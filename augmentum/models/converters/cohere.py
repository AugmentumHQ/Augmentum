"""Cohere API message converter.

Converts between Augmentum's internal message format and the
Cohere chat API.  Cohere is largely OpenAI-compatible but uses
``k`` instead of ``top_k`` and ``p`` instead of ``top_p``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from augmentum.models.converters.utils import prepend_name


class CohereConverter:
    """Convert internal messages to the Cohere chat format.

    Implements the ``MessageConverter`` protocol.
    """

    def convert_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        prefill: str = "",
    ) -> dict[str, Any]:
        """Convert internal messages to Cohere payload parts.

        Returns a dict with ``messages`` (list of Cohere messages).
        Prepends names into content, otherwise passes through.
        """
        msgs = [prepend_name(deepcopy(m)) for m in messages]
        return {"messages": msgs}

    def convert_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pass through — Cohere responses are OpenAI-compatible."""
        return data

    def map_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Map parameter names to Cohere equivalents.

        - ``top_k`` → ``k``
        - ``top_p`` → ``p``
        """
        result = dict(params)
        top_k = result.pop("top_k", None)
        if top_k is not None:
            result["k"] = top_k
        top_p = result.pop("top_p", None)
        if top_p is not None:
            result["p"] = top_p
        return result
