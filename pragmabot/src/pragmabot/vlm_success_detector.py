"""VLM-based action success detection and task completion evaluation."""

import logging
from typing import Any, Dict, List

from PIL import Image
from pydantic import BaseModel, Field

from pragmabot.utils import encode_pil_image_to_base64
from pragmabot.vlm_client import VLMClient
from pragmabot.conversation_builder import ConversationBuilder

logger = logging.getLogger(__name__)


# --- Prompt Templates ---

DETECTOR_SYSTEM_PROMPT = """
You are a helpful assistant for a legged robot equipped with a single arm and a two-finger gripper. You specialize in detecting whether an action or task has been successfully completed, and you propose the most likely causes of failure. You always apply chain-of-thought reasoning by thinking step by step and thoroughly analyzing each possible cause. Each action execution involves perception and manipulation. The robot has a limited accuracy in the action execution. You are aware that the robot has limited perception and manipulation capabilities — be realistic but not overly harsh in your success evaluation. This is a robot, not a human! Crucially, any violation of implicit social norms (e.g., knocking over objects, spilling, breaking things) must be treated as a failure, even if not explicitly mentioned in the task.
"""

DETECTOR_TASK_PROMPT = """
Task given to the robot: {task}
Action the robot just attempted: {action_to_detect}
"""

DETECTOR_RULE_PROMPT = """
Based on the above image, task and action, output your structured evaluation.

SUCCESS CRITERIA FOR ACTION AND TASK
  - For picking up objects, if the object is not lying flat on the table at the original location, you might consider it a success. It can be tilted, still touching the table, or appear to be not lifted, and it still might be considered a success.
  - For pushing an object toward another object, if these objects are getting much closer, you may consider it a success (they do not need to be touching each other or super close). If the push causes the target object to tip over (but not broken), it is still considered as a success.
  - For placing, if the object is roughly placed on the target object, it might be considered as a success. It does not need to be perfectly placed on the center of the target object.
  
FAILURE CRITERIA FOR ACTION AND TASK
  - Always a failure if the robot breaks or drops other objects unintentionally, you must consider it a failure for the current action and thus the overall task (regardless of how it performs).
  - Action failure if the action clearly did not achieve its goal (e.g., gripper missed entirely, object didn’t move at all).
  
REASONING REQUIREMENTS (IF ACTION FAILED)
  - Describe the scene before and after, especially spatial changes and object states.
  - There is a success/failure for the action, and a success/failure for the overall task. Don't confuse them.
"""


class SuccessEvaluation(BaseModel):
    """Structured VLM response for evaluating action success and task completion."""

    scene_description: str = Field(
        ...,
        description="Description of the object's surroundings, especially its spatial relationships with nearby objects. Also, describe what objects have been moved, removed, or added between the two images (before and after the action).",
    )
    is_action_successful: bool = Field(
        ...,
        description="Indicate whether the current action was successful or failed. The action is considered a failure if the robot violates social norms (even not explicitly stated). Note that this is not about the overall instruction or task.",
    )
    is_task_completed: bool = Field(
        ...,
        description="True if the whole task has been successfully completed. The task is not completed if the robot violates social norms (even not explicitly stated). Note this is not about the single action taken, but the overall task.",
    )


class VLMSuccessDetector:
    """Detect whether a robot action was successful using before/after image comparison."""

    def __init__(self, vlm_client: VLMClient, conversation_log: List[Dict[str, Any]]) -> None:
        """Initialize with a VLM client and shared conversation log.

        Args:
            vlm_client: Client for querying the VLM API.
            conversation_log: Shared mutable list for the Gradio UI log.
        """
        self.vlm_client = vlm_client
        self.conversation_log = conversation_log

    def perform_success_detection(
        self,
        task: str,
        action_to_detect: str,
        color_image_before: Image.Image,
        color_image_after: Image.Image,
    ) -> SuccessEvaluation:
        """Evaluate whether the attempted action succeeded by comparing before/after images.

        Sends before and after images along with the task and action description
        to the VLM, which determines whether the action succeeded and whether
        the overall task is complete.

        Args:
            task: The high-level task instruction.
            action_to_detect: Description of the specific action just attempted.
            color_image_before: RGB image captured before the action.
            color_image_after: RGB image captured after the action.

        Returns:
            A SuccessEvaluation containing action success, task completion,
            and scene description.

        Raises:
            ValueError: If the VLM fails to parse a structured response.
        """

        logger.info("VLM success detector running...")

        builder = ConversationBuilder(self.conversation_log)
        builder.log_user_message("## VLM success detector running...")

        # VLM system prompt
        builder.add_system_message(DETECTOR_SYSTEM_PROMPT)

        # VLM instruction prompt
        vlm_task_prompt = DETECTOR_TASK_PROMPT.format(task=task, action_to_detect=action_to_detect)
        builder.add_user_text(vlm_task_prompt)

        # Image before action
        color_image_before_base64 = encode_pil_image_to_base64(color_image_before)
        builder.add_user_image("Caption: Observation before the action was attempted.", color_image_before_base64)

        # Image after action
        color_image_after_base64 = encode_pil_image_to_base64(color_image_after)
        builder.add_user_image(
            "Caption: Observation after the action was attempted. This image is captured after the robot completed the action and returned its arm to the default position.",
            color_image_after_base64,
        )

        # VLM rule prompt
        builder.add_user_text(DETECTOR_RULE_PROMPT, log_heading="")

        # Query VLM API
        success_eval_obj, _, _ = self.vlm_client.query_structured(builder, SuccessEvaluation)
        return success_eval_obj
