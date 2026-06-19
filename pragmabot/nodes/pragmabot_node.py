#!/usr/bin/env python3
"""Main ROS node for PragmaBot with Gradio UI."""

import datetime
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

import gradio as gr
import rospy
from openai import OpenAI

from pragmabot.scene_observer import SceneObserver
from pragmabot.simple_config import get_config
from pragmabot.utils import get_package_path
from pragmabot.vlm_success_detector import VLMSuccessDetector
from pragmabot.vlm_task_planner import VLMTaskPlanner
from pragmabot.vlm_client import VLMClient
from pragmabot.vlm_exp_summarizer import VLMExperienceSummarizer
from pragmabot.conversation_builder import ConversationBuilder
from pragmabot.vlm_scene_describer import VLMSceneDescriber
from pragmabot.memory_manager import MemoryManager


logger = logging.getLogger(__name__)


class PragmaBot:
    """PragmaBot node with Gradio UI for task planning."""

    def __init__(self) -> None:
        """Initialize the PragmaBot ROS node, services, and Gradio UI."""
        self.data_folder = get_package_path() / "data"
        self.log_folder = self.data_folder / "logs"

        # Initialize ROS node and TF listener
        rospy.init_node("pragmabot_node")

        # Configure Python logging AFTER rospy.init_node(), because rospy overrides the root logger's handlers/level during initialization.
        logging.basicConfig(
            level=logging.INFO, format="[%(levelname)s] [%(name)s]: %(message)s", stream=sys.stdout, force=True
        )

        # Initialize variables
        self.color_image = None
        self.depth_image = None
        self.intrinsics = None
        self.observation_time = None

        self.instruction = None
        self.initial_scene_description = None
        self.action_taken = None
        self.stm = []
        self.ltm = []
        self.conversation_log = []
        self.display_index = 0
        self.time_step = 0

        # Load configuration from YAML file
        self.config = get_config()
        self.scene_observer = SceneObserver(self.config.topics)

        # If vlm model contains gpt, initialize the client
        if "gpt" in self.config.vlm.vlm_model:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            rospy.loginfo(f"Using {self.config.vlm.vlm_model} with OpenAI")
        else:
            rospy.logfatal("Unsupported VLM model: %s", self.config.vlm.vlm_model)
            sys.exit(1)
        self.vlm_client = VLMClient(self.client, self.config.vlm)

        self.memory_manager = MemoryManager(self.vlm_client, self.conversation_log)
        self.scene_describer = VLMSceneDescriber(self.vlm_client, self.conversation_log)
        self.exp_summarizer = VLMExperienceSummarizer(self.vlm_client, self.conversation_log)
        self.success_detector = VLMSuccessDetector(self.vlm_client, self.conversation_log)
        self.task_planner = VLMTaskPlanner(self.vlm_client, self.conversation_log)

        self.start_gradio_interface()

    def run(self):
        """Block on rospy.spin() until shutdown."""
        rospy.spin()

    def handle_planning_request(self, chatbot):
        """Run one planning step: observe scene, plan action, and execute.

        Captures a camera observation, optionally generates a scene description,
        queries the VLM action planner, and dispatches the resulting action
        to the function call handler and behavior tree.

        Args:
            chatbot: Gradio chatbot component (unused, kept for callback API).
        """
        self.time_step += 1

        (
            self.color_image,
            self.depth_image,
            self.intrinsics,
            self.observation_time,
        ) = self.scene_observer.get_scene_observation()

        if self.config.activate_ltm or self.config.save_to_ltm:
            # Need initial scene description for LTM retrieval, but only want to generate it once at the first step (not every step)
            if self.initial_scene_description is None:
                self.initial_scene_description = self.scene_describer.get_scene_description(
                    self.instruction, self.color_image
                )

        if self.config.activate_ltm:
            # Retrieve long-term memory if activated and not already retrieved for this task
            if len(self.ltm) == 0:
                self.ltm, _, _, _, _ = self.memory_manager.retrieve_relevant_experiences(
                    self.instruction,
                    self.initial_scene_description,
                    self.config.retrieval_top_k,
                    self.config.use_random_retrieval,
                )

        next_action, _, _ = self.task_planner.plan_action(self.instruction, self.color_image, self.stm, self.ltm)
        self.append_to_stm_if_activated("action", next_action)
        self.action_taken = next_action.chosen_action

        if self.config.rosbag_replay:
            # Send a request for success detection if no action execution
            self.handle_evaluation_request(chatbot)
        else:
            raise NotImplementedError(
                "Implement your own action execution logic here, and then call the success detection service after executing the action in the real world."
            )

    def handle_evaluation_request(self, chatbot):
        """Evaluate action success by comparing before/after observations.

        Captures a new observation, sends the before/after images to the VLM
        success detector, and handles task completion or re-planning.

        Args:
            chatbot: Gradio chatbot component (unused, kept for callback API).
        """

        color_image_before_action = self.color_image

        (
            self.color_image,
            self.depth_image,
            self.intrinsics,
            self.observation_time,
        ) = self.scene_observer.get_scene_observation()

        success_evaluation = self.success_detector.perform_success_detection(
            self.instruction,
            self.action_taken,
            color_image_before_action,
            self.color_image,
        )
        self.append_to_stm_if_activated("evaluation", success_evaluation)

        if success_evaluation.is_task_completed:
            if self.config.save_to_ltm:
                self.handle_experience_summarization(chatbot)
        else:

            if not success_evaluation.is_action_successful:
                info_msg = "The human operator might have reset the scene after the action failure."
                self.append_to_stm_if_activated("additional_info", info_msg)

            # if using rosbag, call planning directly
            if self.config.rosbag_replay:
                self.handle_planning_request(chatbot)
            else:
                raise NotImplementedError(
                    "Implement your own logic here to decide when to call the planning function again for re-planning, based on the success evaluation results."
                )

    def handle_experience_summarization(self, chatbot):
        """Summarize the current short-term memory into a long-term experience entry.

        Args:
            chatbot: Gradio chatbot component (unused, kept for callback API).
        """
        summarized_experience = self.exp_summarizer.summarize_stm_to_ltm(
            self.instruction, self.initial_scene_description, self.stm
        )
        self.memory_manager.save_experience(self.instruction, self.initial_scene_description, summarized_experience)

    def save_conversation_log(self):
        """Save the conversation log to a timestamped JSON file.

        The file is saved to the logs folder with a name derived from the
        instruction and current timestamp. Only called on exit.

        Returns:
            The file path if saved successfully, or None on failure.
        """
        if self.instruction is None:
            rospy.logwarn("No valid instruction — skipping log save.")
            return

        # Sanitize instruction for filename
        safe_instruction = "".join(c if c.isalnum() else "_" for c in self.instruction[:50]).strip().lower()
        if safe_instruction.endswith("_"):
            safe_instruction = safe_instruction[:-1]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{safe_instruction}.json"
        filepath = self.log_folder / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    self.conversation_log,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            rospy.loginfo(f"Conversation log saved to: {filepath}")
            return filepath
        except Exception as e:
            rospy.logerr(f"Failed to save conversation log: {e}")
            return None

    def load_conversation_log(self, file_path, chatbot):
        """Load a conversation log from a JSON file and update the chatbot.

        Args:
            file_path: Path to the JSON log file to load.
            chatbot: Gradio chatbot component (unused, kept for callback API).

        Returns:
            The updated conversation log list.
        """
        if not file_path or not os.path.exists(file_path):
            error_msg = f"Error: File not found - {file_path}"
            rospy.logerr(error_msg)
            self.conversation_log.append({"role": "user", "content": error_msg})
            return self.conversation_log

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded_log = json.load(f)
                if isinstance(loaded_log, list):
                    self.conversation_log = loaded_log
                    rospy.loginfo(f"Loaded conversation log from: {file_path}")
                else:
                    raise ValueError("Invalid log format — expected a list of messages.")
        except Exception as e:
            error_msg = f"Failed to load conversation log: {e}"
            rospy.logerr(error_msg)
            self.conversation_log.append({"role": "user", "content": error_msg})

        return self.conversation_log

    def append_to_stm_if_activated(self, key: str, data: object) -> None:
        """Build a JSON entry and append it to the short-term memory list.

        Args:
            key: Dictionary key for the data (e.g., ``'action'``, ``'evaluation'``).
            data: Payload to store. Pydantic models are auto-serialized via
                ``model_dump``; dicts and strings are stored as-is.
        """
        if self.config.activate_stm is False:
            return

        # Automatically extract data from Pydantic models
        if hasattr(data, "model_dump"):
            processed_data = data.model_dump(exclude_none=True, exclude_unset=True)
        else:
            processed_data = data

        entry_with_timestep = {
            "time_step": self.time_step,
            key: processed_data,
        }

        entry_json = json.dumps(entry_with_timestep, indent=2)
        self.stm.append(f"{entry_json}\n")

        builder = ConversationBuilder(self.conversation_log)
        builder.log_assistant_message(entry_json, is_json=True)

    def start_gradio_interface(self):
        """Build and launch the Gradio web UI using a Right Sidebar Layout."""

        def get_new_task(user_instruction, chatbot):
            """Handle a new user instruction submission."""
            init_task(chatbot)
            self.conversation_log.append({"role": "user", "content": f"# Instruction: {user_instruction}"})
            self.instruction = user_instruction
            # The gui is updated immediately after this function returns. So the display_index should be updated here.
            self.display_index = len(self.conversation_log)
            return "", self.conversation_log

        def init_task(chatbot):
            """Reset all task-related state for a new planning session."""
            self.color_image = None
            self.depth_image = None
            self.intrinsics = None
            self.observation_time = None

            self.instruction = None
            self.initial_scene_description = None
            self.action_taken = None
            self.stm.clear()
            self.ltm.clear()
            self.conversation_log.clear()
            self.display_index = 0
            self.time_step = 0
            return

        def update_chat(chatbot):
            """Yield new conversation log entries to the Gradio chatbot UI."""
            total_messages = len(self.conversation_log)

            if total_messages == 0:
                # If the chat was cleared, just reset the display (this should already be reset by the clear button)
                self.display_index = 0
                yield []
            elif total_messages > self.display_index:
                new_messages = self.conversation_log[self.display_index :]
                self.display_index = total_messages
                yield chatbot + new_messages
            else:
                yield chatbot

        custom_css = """
        #chatbot .bubble {
            max-width: 80% !important;
            padding-left: 12px !important;
            padding-right: 12px !important;
        }

        #chatbot .bubble.left {
            margin-right: auto !important;
        }

        #chatbot .bubble.right {
            margin-left: auto !important;
        }

        #chatbot {
            height: calc(100vh - 100px) !important; 
            min-height: 600px;
        }
        
        .right-sidebar {
            border-left: 1px solid #e5e7eb;
            padding-left: 20px !important;
        }
        """

        with gr.Blocks(css=custom_css, title="PragmaBot") as demo:
            timer = gr.Timer(value=2)  # updates every 2 seconds

            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        type="messages",
                        elem_id="chatbot",
                        label="VLM Conversation Log",
                    )

                with gr.Column(scale=1, elem_classes="right-sidebar"):
                    gr.Markdown("### 🤖 Instruction")
                    instruction_msg = gr.Textbox(
                        show_label=False,
                        placeholder="Type your instruction here. Press enter to submit...",
                        elem_id="user-instruction",
                    )

                    with gr.Row():
                        clear_button = gr.Button("🗑️ Clear", elem_id="clear")
                        update_button = gr.Button("🔄 Update", elem_id="update")

                    gr.HTML("<hr style='margin-top: 30px; margin-bottom: 30px;'>")

                    # 2. Secondary Actions remain at the bottom
                    gr.Markdown("### 📂 Log Management")
                    load_log_button = gr.UploadButton(
                        "Click to Select JSON Log", file_types=[".json"], elem_id="load-log"
                    )

            gr.Markdown(f"> 💡 Default logs location: `{self.log_folder}`")

            # --- Event Bindings ---
            timer.tick(update_chat, inputs=chatbot, outputs=chatbot)

            instruction_msg.submit(
                get_new_task, [instruction_msg, chatbot], [instruction_msg, chatbot], queue=False
            ).then(self.handle_planning_request, chatbot, None)

            clear_button.click(lambda: init_task(chatbot), None, chatbot, queue=False)
            update_button.click(update_chat, chatbot, chatbot, queue=False)
            load_log_button.upload(
                self.load_conversation_log, inputs=[load_log_button, chatbot], outputs=chatbot, queue=False
            )

        demo.launch(share=self.config.gradio_share, inline=False, prevent_thread_lock=True, server_name="0.0.0.0")


if __name__ == "__main__":
    pragmabot = PragmaBot()

    try:
        pragmabot.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt received. Shutting down...")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
    finally:
        rospy.loginfo("Saving conversation log before shutdown...")
        pragmabot.save_conversation_log()
        rospy.loginfo("Cleanup completed. Node terminated.")
