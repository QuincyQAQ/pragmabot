"""VLM-based scene description generation."""

import json
import logging
from typing import Any, Dict, List, Optional

from PIL import Image
from pydantic import BaseModel, Field

from pragmabot.utils import encode_pil_image_to_base64
from pragmabot.vlm_client import VLMClient
from pragmabot.conversation_builder import ConversationBuilder

logger = logging.getLogger(__name__)


# --- Prompt Templates ---

SCENE_SYSTEM_PROMPT = """
You are a helpful assistant for a legged robot equipped with a single arm and a two-finger gripper. You specialize in generating accurate and concise scene descriptions. You always apply chain-of-thought reasoning to ensure accurate and comprehensive scene understanding.
"""

SCENE_TASK_PROMPT = """
The robot received this instruction from the user: {task}

Based on your observation of the image, provide a short scene description that focuses on the spatial relationships between the target object and nearby objects the robot may need to interact with. Describe only the scene, do not include the robot's task in the description.
"""


class SceneDescription(BaseModel):
    """Structured VLM response for a scene description."""

    chain_of_thought_reasoning: str = Field(
        ..., description="Describe what was observed in the image to generate the scene description."
    )
    scene_description: str = Field(
        ..., description="A brief summary of the scene, focusing on relevant spatial relationships."
    )


class VLMSceneDescriber:
    """Generate concise scene descriptions using a VLM."""

    def __init__(self, vlm_client: VLMClient, conversation_log: Optional[List[Dict[str, Any]]] = None) -> None:
        """Initialize with a VLM client and shared conversation log.

        Args:
            vlm_client: Client for querying the VLM API.
            conversation_log: Optional shared mutable list for the Gradio UI log.
        """
        self.vlm_client = vlm_client
        self.conversation_log = conversation_log

    def get_scene_description(
        self,
        task: str,
        color_image: Image.Image,
    ) -> str:
        """Generate a scene description from the current camera observation.

        Sends the task instruction and current camera image to the VLM and
        returns a concise description of spatial relationships in the scene.

        Args:
            task: The user-provided task instruction.
            color_image: Current RGB camera observation as a PIL Image.

        Returns:
            A string describing the scene and spatial relationships.

        Raises:
            ValueError: If the VLM fails to parse a structured response.
        """
        logger.info("VLM scene describer running...")

        builder = ConversationBuilder(self.conversation_log)
        builder.log_user_message("## VLM scene describer running...")

        # VLM system prompt
        builder.add_system_message(SCENE_SYSTEM_PROMPT)

        # VLM instruction prompt
        vlm_task_prompt = SCENE_TASK_PROMPT.format(task=task)
        builder.add_user_text(vlm_task_prompt)

        color_image_base64 = encode_pil_image_to_base64(color_image)
        builder.add_user_image("Here's the robot's current observation.", color_image_base64)

        # Query VLM API
        scene_desc_obj, _, _ = self.vlm_client.query_structured(builder, SceneDescription)

        answer_json = json.dumps(scene_desc_obj.model_dump(exclude_none=True, exclude_unset=True), indent=2)
        builder.log_assistant_message(answer_json, is_json=True)

        return scene_desc_obj.scene_description
