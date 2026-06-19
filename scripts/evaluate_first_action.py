#!/usr/bin/env python3
"""Evaluate first-action accuracy across images in the evaluation set.

Runs the full pipeline (scene description, LTM retrieval, action planning)
on each image under ``pragmabot/data/images/`` and writes results
to a timestamped Markdown file.
"""

import json
import os
import datetime
from pathlib import Path

import cv2
import matplotlib
from PIL import Image

matplotlib.use("Agg")

from openai import OpenAI

from pragmabot.utils import get_package_path, encode_pil_image_to_base64

from pragmabot.vlm_task_planner import VLMTaskPlanner
from pragmabot.vlm_scene_describer import VLMSceneDescriber
from pragmabot.memory_manager import MemoryManager
from pragmabot.vlm_client import VLMClient
from pragmabot.simple_config import get_config


class ActionAccuracyEvaluator:
    def __init__(self) -> None:
        self.data_folder = get_package_path() / "data"
        self.ltm_file_path = self.data_folder / "ltm" / "ltm.csv"
        self.image_eval_folder = self.data_folder / "images"

        # Load configuration from YAML file
        self.config = get_config()

        # If vlm model contains gpt, initialize the client
        if "gpt" in self.config.vlm.vlm_model:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported VLM model: {self.config.vlm.vlm_model}")
        self.vlm_client = VLMClient(self.client, self.config.vlm)

        # make a new log file under output_folder with the vlm_model name and current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = Path(__file__).parent / "output" / f"{self.config.vlm.vlm_model}_{timestamp}.md"

    def run_eval(self, color_image, instruction):
        time_step = 1

        initial_scene_description = None
        stm = []
        ltm = []
        sorted_exp_str = ""

        scene_describer = VLMSceneDescriber(self.vlm_client)
        task_planner = VLMTaskPlanner(self.vlm_client)
        memory_manager = MemoryManager(self.vlm_client)

        if self.config.activate_ltm:
            # Generate scene description after we have camera data
            initial_scene_description = scene_describer.get_scene_description(instruction, color_image)

            # Retrieve long-term memory if activated and not already retrieved for this task
            ltm, _, _, _, sorted_exp_str = memory_manager.retrieve_relevant_experiences(
                instruction, initial_scene_description, self.config.retrieval_top_k, self.config.use_random_retrieval
            )

        next_action, response_time, prompt_tokens = task_planner.plan_action(
            instruction,
            color_image,
            stm,
            ltm,
        )

        action_with_timestep = {
            "time_step": time_step,
            "action": next_action.model_dump(exclude_none=True, exclude_unset=True),
        }
        action_with_timestep_json = json.dumps(action_with_timestep, indent=2)

        # Log the function call in a pretty JSON format
        print("Planned Action:", action_with_timestep_json)

        # Append the image and instruction to the markdown log file current_log_file
        with open(self.current_log_file, "a") as log_file:
            # Write instruction
            log_file.write(f"# Instruction:\n{instruction}\n\n")
            # Write image
            color_image_base64 = encode_pil_image_to_base64(color_image)
            log_file.write(f"![Robot Observation]({color_image_base64})\n\n")
            # Write sorted experiences
            if self.config.activate_ltm:
                log_file.write(f"Sorted Long-Term Memories:\n```\n{sorted_exp_str}\n```\n\n")
            # Write action taken
            log_file.write(f"Number of prompt tokens: {prompt_tokens}\n\n")
            log_file.write(f"VLM Response Time: {response_time} seconds\n\n")
            log_file.write(f"Action Taken:\n```json\n{action_with_timestep_json}\n```\n\n")

    def run_evals(self):
        for file_path in sorted(self.image_eval_folder.iterdir()):
            # open the image file
            color_image = cv2.imread(str(file_path))

            # convert to Pillow image
            color_image_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

            # remove the extension from the file name
            instruction = file_path.stem.replace("_", " ")
            # remove any numeric digits from the instruction
            instruction = "".join([i for i in instruction if not i.isdigit()])
            # remove trailing and leading spaces
            instruction = instruction.strip()
            # remove double spaces
            instruction = " ".join(instruction.split())
            # add a period at the end if not present
            if not instruction.endswith("."):
                instruction += "."

            print("----------------------------------------")
            print("Evaluating image file:", file_path.name)
            print("Instruction:", instruction)
            self.run_eval(color_image_pil, instruction)


if __name__ == "__main__":
    evaluator = ActionAccuracyEvaluator()
    evaluator.run_evals()
