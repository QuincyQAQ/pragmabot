"""VLM-based action planning with short-term and long-term memory integration."""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from pydantic import BaseModel, Field

from pragmabot.utils import encode_pil_image_to_base64
from pragmabot.vlm_client import VLMClient
from pragmabot.conversation_builder import ConversationBuilder

logger = logging.getLogger(__name__)


# --- Prompt Templates ---

PLANNER_SYSTEM_PROMPT = """
You are a helpful assistant for a legged robot equipped with a single arm and a two-finger gripper. You specialize in task planning, and you always suggest the plan that is most likely to fulfill the task. You learn and adapt from previous experience in the long-term memory and also the current short-term memory (especially the action failures). Always apply chain-of-thought reasoning by thinking step by step before making a final decision.
"""

PLANNER_TASK_PROMPT = """
The robot received this instruction from the user: {task}
"""

PLANNER_INSTRUCTION_PROMPT = """
Given the current scene and task, choose the next best action.

HARD CONSTRAINTS TO APPLY:
  - PUSH only works on objects directly on the table (nothing can be between the object and the table). PICK and PLACE would also work.
  - If the objective is to put an object on top of another object, you must use PICK and PLACE. PUSH would not work. Similarly, you cannot PUSH an object off another object because it is not directly on the table.
  - When two objects are next to each other, it's infeasible to directly PUSH the object that is behind the other one.
  - If the robot is holding something, it must PLACE that object before attempting to grasp another.
  - When the target object is tiny or flat, which is hard to grasp, you cannot use PICK.
  - In the case of action failure, never repeat the same action immediately without first rearranging the scene yourself. This is because the same failure is mostly likely to occur, whether the scene has been reset or not. Instead, consider whether other actions could be taken.
  - Never give a suggestion on how to better execute a failed action—the robot cannot understand or adapt to such advice.
  
GENERAL RULES:
  - Could interacting with other objects help? You are allowed to interact with any other objects on the table.
  - Any object on the table can be used as a tool (all are clean and safe to handle). Be creative! But remember to always PICK up the tool first before using it.
  - In the reasoning process, first propose a promising action, then check whether it violates any constraints, one by one. If it does, discard it and propose another. Repeat this until you find a promising action that satisfies all constraints. 

ACTION PARAMETERS:
  - For the pick up, if the object needs to be grasped at a specific section, you must specify that as well.
  - For the place action, you need to specify which object to place the target object on (not next to). If the object needs to be placed at a specific section, you must specify that as well.
  - For the push action, you need to specify the direction to push (left or right). When both push directions work, prefer pushing left if the object is on the left to the gripper; right if on the right.

Output your final decision in the specified structured format. A human operator may have reset the scene to its initial state after the failure.
"""

STM_PROMPT = """
Here is the action history for the current task so far for reference:

```json
{stm_str}
```

This contains the actions proposed and executed, and the feedback after the action execution. Learn from the experience (including past and active logs) and adapt if needed.
"""

LTM_PROMPT = """
Here are the past relevant experiences from long-term memory.

```json
{ltm_str}
```

They illustrate how the robot successfully planned action sequences to complete tasks, sometimes after initial failures. They reflect the robot’s capabilities and limitations. Always first carefully reason how the current scenario is similar to these past ones, including how the target object is similar. Identify any lessons learned that could apply to the current task. This is part of the chain-of-thought reasoning and helps you avoid repeating past failures.
"""


class RobotSkill(str, Enum):
    """Available robot manipulation skills."""

    PUSH = "push"
    PICK = "pick"
    PLACE = "place"


class PushDirection(str, Enum):
    """Allowed push directions for push actions."""

    LEFT = "left"
    RIGHT = "right"


class NextBestAction(BaseModel):
    """Structured VLM response for the next best robot action to take."""

    scene_description: str = Field(
        ...,
        description="An accurate and concise description of the object's surroundings, especially its spatial relationships with nearby objects. Directions should always be described with respect to the camera, not relative to the object.",
    )
    applicable_knowledge: Optional[str] = Field(
        None,
        description="If past experiences from long-term memory are provided, first identify all similar scenarios, and then check whether there is applicable knowledge to avoid any similar failures. Only output this when long-term memory is provided. ",
    )
    chain_of_thought_reasoning: str = Field(
        ...,
        description="Carefully think about the outcome and the feasibility of the actions. Think step by step and check the constraints one by one. Pay attention to the spatial relationship. Choose the best immediate action that is most likely to accomplish the task.",
    )
    chosen_action: str = Field(
        ...,
        description="The action chosen to be executed (only a single action). This needs to a clear text description of the next immediate action, which a VLM success detector will use to determine whether this action was successful or not.",
    )
    chosen_skill: RobotSkill = Field(..., description="The skill to use for the action, chosen from a predefined set.")
    target_object: str = Field(
        ...,
        description="The object the robot should interact with. This is either the object to push, pick up, or object we should place somewhere.",
    )
    should_grasp_at_specific_section: Optional[bool] = Field(
        None,
        description="Whether the robot should grasp the object at a specific section. Required for pick actions and shall not be provided for other actions.",
    )
    placement_object: Optional[str] = Field(
        None,
        description="The object on which the target object should be placed (not next to). Required for place actions and should not be provided for other actions.",
    )
    should_place_at_specific_section: Optional[bool] = Field(
        None,
        description="Whether the robot should place the object at a specific section to ensure stability and proper placement. Required for place actions and shall not be provided for other actions.",
    )
    push_direction: Optional[PushDirection] = Field(
        None,
        description="The direction to push the object, chosen from a predefined set. Prefer pushing left if object is on left side of image; right if on right. Required for push actions and shall not be provided for other actions.",
    )


class VLMTaskPlanner:
    """Plan the next best robot action using a VLM with memory-augmented reasoning."""

    def __init__(
        self,
        vlm_client: VLMClient,
        conversation_log: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Initialize the planner with a VLM client and shared conversation log.

        Args:
            vlm_client: Client for querying the VLM API.
            conversation_log: Optional shared mutable list for the Gradio UI log.
        """
        self.vlm_client = vlm_client
        self.conversation_log = conversation_log

    def plan_action(
        self,
        task: str,
        color_image: Image.Image,
        stm: List[str],
        ltm: List[str],
    ) -> Tuple[NextBestAction, float, int]:
        """Plan the next best action given the current scene and task.

        Constructs a VLM prompt with the task, camera observation, short-term
        memory, and long-term memory, then queries the VLM for the next best
        action.

        Args:
            task: The user-provided task instruction.
            color_image: Current RGB camera observation as a PIL Image.
            stm: List of short-term memory entries (action/feedback strings).
            ltm: List of long-term memory entries to include in the prompt.

        Returns:
            A tuple of (action, response_time, prompt_tokens)
            where action is the parsed NextBestAction, response_time is in seconds,
            and prompt_tokens is the number of input tokens used.

        Raises:
            ValueError: If the VLM fails to parse a structured response.
        """

        logger.info("VLM task planner running...")

        builder = ConversationBuilder(self.conversation_log)
        builder.log_user_message("## VLM task planner running...")

        # VLM system prompt
        builder.add_system_message(PLANNER_SYSTEM_PROMPT)

        # VLM instruction prompt
        vlm_task_prompt = PLANNER_TASK_PROMPT.format(task=task)
        builder.add_user_text(vlm_task_prompt)

        color_image_base64 = encode_pil_image_to_base64(color_image)
        builder.add_user_image("Here's the robot's current observation.", color_image_base64)

        # Add short-term memory
        if len(stm) > 0:
            stm_str = "\n".join(stm)
            vlm_stm_prompt = STM_PROMPT.format(stm_str=stm_str)
            builder.add_user_text(vlm_stm_prompt, log_heading="")

        builder.add_user_text(PLANNER_INSTRUCTION_PROMPT, log_heading="")

        # Add long-term memory
        if len(ltm) > 0:
            ltm_str = "\n".join(ltm)
            vlm_ltm_prompt = LTM_PROMPT.format(ltm_str=ltm_str)
            builder.add_user_text(vlm_ltm_prompt, log_heading="")

        # Query VLM API
        action_obj, response_time, prompt_tokens = self.vlm_client.query_structured(builder, NextBestAction)
        return action_obj, response_time, prompt_tokens
