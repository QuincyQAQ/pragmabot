"""Shared utility functions"""

import io
import logging
from pathlib import Path

import base64
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = [
    "get_package_path",
    "get_scenario_key",
    "encode_pil_image_to_base64",
]


def get_package_path() -> Path:
    """Return the root path of the pragmabot package."""
    return Path(__file__).parents[2]


def get_scenario_key(instruction: str, initial_scene_description: str) -> str:
    """Format an instruction and scene description into a scenario key for LTM retrieval."""
    return f"Instruction: {instruction}\nScene: {initial_scene_description}"


def encode_pil_image_to_base64(image: Image.Image, image_format: str = "JPEG", quality: int = 85) -> str:
    """Encode a PIL Image as a base64 data URI string.

    The image is converted to RGB if needed, then encoded in the requested
    format and returned as a ``data:<mime>;base64,...`` URI.

    Args:
        image: The PIL Image to encode.
        image_format: Output format (e.g., ``'JPEG'``, ``'PNG'``).
        quality: JPEG quality (ignored for non-JPEG formats).

    Returns:
        A base64-encoded data URI string.

    Raises:
        ValueError: If *image* is None.
    """
    if image is None:
        raise ValueError("Image cannot be None")

    # Convert to RGB if not already (e.g., RGBA, L, P, etc.)
    if image.mode != "RGB":
        # Use a white background for RGBA → RGB to avoid black transparency artifacts
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            image = background
        else:
            image = image.convert("RGB")

    buffered = io.BytesIO()
    if image_format.upper() == "JPEG":
        image.save(buffered, format="JPEG", quality=quality, optimize=True)
    else:
        image.save(buffered, format=image_format, optimize=True)

    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime_type = "image/jpeg" if image_format.upper() == "JPEG" else f"image/{image_format.lower()}"
    return f"data:{mime_type};base64,{base64_image}"
