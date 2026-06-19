"""Summarize short-term robot experiences into long-term memory entries."""

import json
import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from pragmabot.vlm_client import VLMClient
from pragmabot.conversation_builder import ConversationBuilder

logger = logging.getLogger(__name__)


# --- Prompt Templates ---

EXPERIENCE_SYSTEM_PROMPT = """
You are a helpful assistant for a legged robot equipped with a single arm and a two-finger gripper. You specialize in summarizing the robot's experiences and extracting generalizable lessons learned. You always apply chain-of-thought reasoning to thoroughly analyze each situation before performing the conversion.
"""

EXPERIENCE_TASK_PROMPT = """
Instruction: {instruction}
Scene: {scene_description}

Please summarize the following step-by-step actions and feedback. Make sure to include every single step the robot took. In addition, highlight the key lesson learned, including similar scenarios the robot may encounter and how it should sequence its actions to complete the task without any failure in the future. Please use only ASCII code.

```json
{stm_str}
```
"""


class ExperienceSummary(BaseModel):
    """Structured VLM response for a summarized robot experience."""

    chain_of_thought_reasoning: str = Field(
        ..., description="Reasoning about the what happened and the lessons learned."
    )
    summarized_experience: str = Field(..., description="The summarized experience, including the lessons learned.")


class VLMExperienceSummarizer:
    """Summarize robot short-term memory into long-term experience entries."""

    def __init__(self, vlm_client: VLMClient, conversation_log: List[Dict[str, Any]]) -> None:
        """Initialize with a VLM client and shared conversation log.

        Args:
            vlm_client: Client for querying the VLM API.
            conversation_log: Shared mutable list for the Gradio UI log.
        """
        self.vlm_client = vlm_client
        self.conversation_log = conversation_log

    def summarize_stm_to_ltm(
        self,
        instruction: str,
        scene_description: str,
        stm: List[str],
    ) -> str:
        """Summarize short-term memory into a long-term experience entry.

        Sends the task instruction, scene description, and step-by-step
        action history to the VLM, which returns a summarized experience
        with lessons learned.

        Args:
            instruction: The task instruction string.
            scene_description: The scene description at the time of the task.
            stm: List of short-term memory entries (action/feedback strings).

        Returns:
            A summarized experience string including lessons learned.

        Raises:
            ValueError: If the VLM fails to parse a structured response.
        """

        logger.info("Task completed. VLM experience summarizer running...")

        builder = ConversationBuilder(self.conversation_log)
        builder.log_user_message("## Task completed. VLM experience summarizer running...")

        # VLM system prompt
        builder.add_system_message(EXPERIENCE_SYSTEM_PROMPT)

        # VLM instruction prompt
        stm_str = "\n".join(stm)
        vlm_instruction_prompt = EXPERIENCE_TASK_PROMPT.format(
            instruction=instruction, scene_description=scene_description, stm_str=stm_str
        )
        builder.add_user_text(vlm_instruction_prompt)

        # Query VLM API
        exp_summary_obj, _, _ = self.vlm_client.query_structured(builder, ExperienceSummary)

        answer_json = json.dumps(exp_summary_obj.model_dump(exclude_none=True, exclude_unset=True), indent=2)
        builder.log_assistant_message(answer_json, is_json=True)

        return exp_summary_obj.summarized_experience
