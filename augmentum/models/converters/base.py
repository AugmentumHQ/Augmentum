"""Base protocol and enums for message converters."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable


class PostProcessMode(Enum):
    """Post-processing modes for message conversion.

    Controls how messages are normalized before sending to providers.
    Modeled after SillyTavern's message post-processing pipeline.
    """

    NONE = "none"
    MERGE = "merge"
    SEMI = "semi"
    STRICT = "strict"


@runtime_checkable
class MessageConverter(Protocol):
    """Protocol for converting between internal and provider message formats.

    Each LLM provider has its own message format. Implementations of this
    protocol translate between Augmentum's internal dict-based messages and
    the provider's native format.
    """

    def convert_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        prefill: str = "",
    ) -> dict[str, Any]:
        """Convert internal messages to provider-specific request payload.

        Args:
            messages: List of message dicts with at least 'role' and 'content'.
            tools: Optional tool definitions to include in the request.
            prefill: Optional assistant prefill text.

        Returns:
            Provider-specific request payload dict.
        """
        ...

    def convert_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert provider response to internal format.

        Args:
            data: Raw provider response dict.

        Returns:
            Normalized response dict.
        """
        ...
