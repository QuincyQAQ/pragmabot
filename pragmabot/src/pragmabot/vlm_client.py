"""Client wrapper for querying Vision-Language Model (VLM) APIs."""

import time
from typing import Optional, Union, List, Tuple, Type, TypeVar
from omegaconf import DictConfig

from openai import OpenAI

from pragmabot.conversation_builder import ConversationBuilder

T = TypeVar("T")


class VLMClient:
    """Wrapper around the OpenAI chat completions API for VLM queries."""

    def __init__(self, client: OpenAI, config: DictConfig) -> None:
        """Initialize with an OpenAI client and model configuration.

        Args:
            client: OpenAI API client instance.
            config: VLM configuration (model name, etc.).
        """
        self.client = client
        self.config = config

    def query_structured(
        self,
        builder: ConversationBuilder,
        response_format: Type[T],
    ) -> Tuple[T, float, int]:
        """Send messages, log timing, parse response, and handle errors.

        Args:
            builder: A ``ConversationBuilder`` containing the accumulated messages
                and an optional conversation log reference.
            response_format: A Pydantic model class that defines the expected
                structured response schema.

        Returns:
            A tuple of ``(parsed_response, response_time, prompt_tokens)``.

        Raises:
            ValueError: If the VLM fails to produce a parseable response.
        """
        start_time = time.time()
        response = self.client.chat.completions.parse(
            model=self.config.vlm_model,
            messages=builder.messages,
            response_format=response_format,
            temperature=0.2,
        )
        elapsed = time.time() - start_time

        builder.log_assistant_message(
            f"VLM reasoning time: {elapsed:.2f} s. # prompt tokens: {response.usage.prompt_tokens}."
        )

        answer = response.choices[0].message
        if not answer.parsed:
            raise ValueError(f"VLM refused to parse. Content: {answer.content}")

        return answer.parsed, elapsed, response.usage.prompt_tokens

    def get_text_embedding(
        self,
        text: Union[str, List[str]],
        builder: Optional[ConversationBuilder] = None,
    ) -> Tuple[List[List[float]], float, int]:
        """Get text embeddings from the OpenAI embedding model.

        Args:
            text: A string or list of strings to embed.
            builder: An optional ``ConversationBuilder`` containing the accumulated messages
                and an optional conversation log reference.

        Returns:
            A tuple of (embeddings, elapsed_time, prompt_tokens).
        """
        start_time = time.time()
        response = self.client.embeddings.create(
            model=self.config.text_embedding_model,
            input=text,
        )
        elapsed = time.time() - start_time

        if builder:
            builder.log_assistant_message(
                f"Embedding query time: {elapsed:.2f} s. # prompt tokens: {response.usage.prompt_tokens}."
            )

        embeddings = [d.embedding for d in response.data]
        return embeddings, elapsed, response.usage.prompt_tokens
