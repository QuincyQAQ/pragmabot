"""Builder for constructing VLM API messages and UI conversation log entries in sync."""

from inspect import cleandoc
from typing import Any, Dict, List, Optional


class ConversationBuilder:
    """Build VLM API messages and UI conversation log entries simultaneously.

    Eliminates the error-prone pattern of maintaining two parallel lists
    (``messages`` for the VLM API and ``conversation_log`` for the Gradio UI)
    by providing a single interface that appends to both at once, using the
    correct format for each.

    Usage::

        builder = ConversationBuilder(conversation_log)
        builder.log_user_message("## VLM task planner running...")
        builder.add_system_message(system_prompt)
        builder.add_user_text(task_prompt)
        builder.add_user_image("Current observation.", img_b64)

        parsed, response = vlm_client.query_structured(builder, NextBestAction)
        builder.log_assistant_message(answer_json, is_json=True)
    """

    def __init__(self, conversation_log: Optional[List[Dict[str, Any]]] = None) -> None:
        """Initialize the builder.

        Args:
            conversation_log: Optional shared mutable list for the Gradio UI log.
                If ``None``, UI logging is silently skipped.
        """
        self.messages: List[Dict[str, Any]] = []
        self.conversation_log = conversation_log

    def add_system_message(self, content: str, log_heading: str = "### System prompt") -> None:
        """Add a system message to the VLM messages and log it to the UI.

        Args:
            content: The system prompt text.
            log_heading: Heading shown in the conversation log before the prompt.
        """
        content = cleandoc(content)
        self.messages.append({"role": "system", "content": content})
        if self.conversation_log is not None:
            self.conversation_log.append({"role": "user", "content": log_heading})
            self.conversation_log.append({"role": "user", "content": content})

    def add_user_text(self, text: str, log_heading: str = "### User prompt") -> None:
        """Add a user text message wrapped in the OpenAI content-array format.

        The VLM API receives ``[{"type": "text", "text": ...}]`` while the
        conversation log receives plain text.

        Args:
            text: The user prompt text.
            log_heading: Heading shown in the conversation log before the prompt.
        """
        text = cleandoc(text)
        self.messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        )
        if self.conversation_log is not None:
            if log_heading:
                self.conversation_log.append({"role": "user", "content": log_heading})
            self.conversation_log.append({"role": "user", "content": text})

    def add_user_image(self, caption: str, image_base64: str, detail: str = "high") -> None:
        """Add an image message with caption to both VLM messages and the log.

        The VLM API receives the multi-part ``image_url`` format while the
        conversation log receives a Markdown image embed.

        Args:
            caption: Text caption displayed alongside the image.
            image_base64: Base64-encoded image data URL.
            detail: Image detail level for the VLM API (default ``"high"``).
        """
        caption = cleandoc(caption)
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": caption},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64, "detail": detail},
                    },
                ],
            }
        )
        if self.conversation_log is not None:
            self.conversation_log.append(
                {
                    "role": "user",
                    "content": f"{caption} ![]({image_base64})",
                }
            )

    def log_assistant_message(self, content: str, is_json: bool = False) -> None:
        """Append an assistant entry to the conversation log only (not to VLM messages).

        Args:
            content: The assistant message content (e.g. timing info, JSON results).
            is_json: Whether the content is JSON formatted.
        """
        if is_json:
            content = f"```json\n{content}\n```"
        else:
            content = cleandoc(content)

        if self.conversation_log is not None:
            self.conversation_log.append({"role": "assistant", "content": content})

    def log_user_message(self, content: str) -> None:
        """Append a user entry to the conversation log only (not to VLM messages).

        Useful for status banners like ``"## VLM task planner running..."``.

        Args:
            content: The user message content.
        """
        content = cleandoc(content)
        if self.conversation_log is not None:
            self.conversation_log.append({"role": "user", "content": content})
