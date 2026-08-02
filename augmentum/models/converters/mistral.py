"""Mistral API message converter.

Converts between Augmentum's internal message format and the
Mistral chat completions API.  Mistral is largely OpenAI-compatible
but requires tool-call IDs to be exactly 9 hex characters and uses
``random_seed`` instead of ``seed``.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

from augmentum.models.converters.utils import (
    merge_consecutive_messages,
    prepend_name,
)


def _hash_tool_id(tool_id: str) -> str:
    """Hash a tool-call ID to a 9-char hex string via SHA-512."""
    return hashlib.sha512(tool_id.encode()).hexdigest()[:9]


class MistralConverter:
    """Convert internal messages to the Mistral chat format.

    Implements the ``MessageConverter`` protocol.

    Args:
        enable_prefix: When ``True``, add ``prefix: True`` to the last
            assistant message so Mistral continues from that point.
    """

    def __init__(self, enable_prefix: bool = False) -> None:
        self._enable_prefix = enable_prefix

    # ------------------------------------------------------------------
    # MessageConverter protocol
    # ------------------------------------------------------------------

    def convert_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        prefill: str = "",
    ) -> dict[str, Any]:
        """Convert internal messages to Mistral payload parts.

        Returns a dict with ``messages`` (list of Mistral messages).
        Steps:
        1. Prepend names into content.
        2. Convert mid-conversation system→user (system after assistant).
        3. Hash tool IDs to 9-char hex.
        4. Merge consecutive same-role messages.
        5. Optionally add ``prefix: True`` to last assistant.
        """
        msgs = deepcopy(messages)

        # 1. Prepend names
        msgs = [prepend_name(m) for m in msgs]

        # 2. Convert system messages that appear after an assistant message
        #    to user role (Mistral only allows system at the start).
        seen_assistant = False
        for msg in msgs:
            if msg["role"] == "assistant":
                seen_assistant = True
            elif msg["role"] == "system" and seen_assistant:
                msg["role"] = "user"

        # 3. Build ID mapping and hash tool IDs
        id_map: dict[str, str] = {}
        for msg in msgs:
            for tc in msg.get("tool_calls", []):
                original_id = tc.get("id", "")
                if original_id and original_id not in id_map:
                    id_map[original_id] = _hash_tool_id(original_id)
                tc["id"] = id_map.get(original_id, original_id)

            # Tool result messages reference tool_call_id
            original_tcid = msg.get("tool_call_id", "")
            if original_tcid and original_tcid in id_map:
                msg["tool_call_id"] = id_map[original_tcid]

        # 4. Merge consecutive same-role messages (skip tool/assistant-with-tool_calls)
        merged: list[dict[str, Any]] = []
        for msg in msgs:
            if (
                merged
                and merged[-1]["role"] == msg["role"]
                and msg["role"] not in ("tool",)
                and not msg.get("tool_calls")
                and not merged[-1].get("tool_calls")
            ):
                merged[-1]["content"] = (
                    merged[-1].get("content", "") + "\n\n" + msg.get("content", "")
                )
            else:
                merged.append(msg)

        # 5. Optionally add prefix to last assistant
        if self._enable_prefix and merged:
            for i in range(len(merged) - 1, -1, -1):
                if merged[i]["role"] == "assistant":
                    merged[i]["prefix"] = True
                    break

        return {"messages": merged}

    def convert_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pass through — Mistral responses are OpenAI-compatible."""
        return data

    def map_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Map parameter names to Mistral equivalents.

        - ``seed`` → ``random_seed`` (skipped if value is -1)
        """
        result = dict(params)
        seed = result.pop("seed", None)
        if seed is not None and seed != -1:
            result["random_seed"] = seed
        return result
