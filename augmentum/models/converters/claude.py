"""Claude Messages API message converter.

Converts between Augmentum's internal message format and the
Anthropic Claude Messages API format.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from augmentum.models.converters.utils import (
    ZWS,
    extract_system_prefix,
    prepend_name,
)

# ---------------------------------------------------------------------------
# Model capability detection
# ---------------------------------------------------------------------------

# ``opus-4`` / ``sonnet-4`` match every 4.x point release (4-6/4-7/4-8…)
# via substring search, so they don't need to be enumerated. Fable 5 is
# its own family and must be listed explicitly.
_THINKING_MODEL_RE = re.compile(
    r"claude-(3-7|opus-4|sonnet-4|haiku-4-5|fable-5)"
)

# Frontier generation — adaptive thinking is the ONLY thinking mode
# (``thinking:{budget_tokens}`` 400s on Opus 4.7+/Fable-5), assistant
# prefill 400s, and context is 1M. Covers Opus 4.6/4.7/4.8, Sonnet 4.6,
# Fable 5. Verified 2026-06-15 (Anthropic model-migration docs).
_ADAPTIVE_MODEL_RE = re.compile(
    r"claude-(opus-4-6|opus-4-7|opus-4-8|sonnet-4-6|fable-5)"
)

# Opus 4.7, Opus 4.8, and Fable 5 reject ``temperature``/``top_p``/
# ``top_k`` UNCONDITIONALLY (400), not just when thinking is on. Opus 4.6
# / Sonnet 4.6 still accept sampling params when thinking is off, so they
# are deliberately excluded here.
_NO_SAMPLING_MODEL_RE = re.compile(
    r"claude-(opus-4-7|opus-4-8|fable-5)"
)

# Assistant prefill 400s across the whole frontier generation.
_NO_PREFILL_MODEL_RE = _ADAPTIVE_MODEL_RE


def is_thinking_model(model: str) -> bool:
    """Return True if the model supports extended thinking."""
    return bool(_THINKING_MODEL_RE.search(model))


def is_adaptive_model(model: str) -> bool:
    """Return True if the model supports adaptive thinking (output_config)."""
    return bool(_ADAPTIVE_MODEL_RE.search(model))


def is_no_sampling_model(model: str) -> bool:
    """Return True if the model rejects temperature/top_p/top_k entirely."""
    return bool(_NO_SAMPLING_MODEL_RE.search(model))


def is_no_prefill_model(model: str) -> bool:
    """Return True if the model does not support assistant prefill."""
    return bool(_NO_PREFILL_MODEL_RE.search(model))


# ---------------------------------------------------------------------------
# Thinking budget
# ---------------------------------------------------------------------------

_EFFORT_PERCENTAGES: dict[str, float] = {
    "min": 0.0,
    "low": 0.10,
    "medium": 0.25,
    "high": 0.50,
    "max": 0.95,
}

_MIN_BUDGET = 1024


def calculate_thinking_budget(effort: str, max_tokens: int) -> int:
    """Calculate thinking budget tokens from effort level and max_tokens."""
    pct = _EFFORT_PERCENTAGES.get(effort, _EFFORT_PERCENTAGES["medium"])
    budget = int(max_tokens * pct)
    return max(budget, _MIN_BUDGET)


def get_thinking_config(
    model: str, effort: str, max_tokens: int
) -> dict[str, Any]:
    """Return the thinking configuration dict for a Claude request.

    Adaptive models get ``{type: "adaptive"}`` with an effort field.
    Traditional thinking models get ``{type: "enabled", budget_tokens: N}``.
    """
    if is_adaptive_model(model):
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
    budget = calculate_thinking_budget(effort, max_tokens)
    return {
        "thinking": {"type": "enabled", "budget_tokens": budget},
    }


# ---------------------------------------------------------------------------
# Prompt caching
# ---------------------------------------------------------------------------


def apply_prompt_caching(
    system: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cache_ttl: str = "5m",
) -> None:
    """Mutate *system* and *tools* in-place to add cache_control markers.

    Adds ``cache_control: {type: "ephemeral", ttl: <cache_ttl>}`` to the
    last system block and the last tool definition.
    """
    cc = {"type": "ephemeral", "ttl": cache_ttl}
    if system:
        system[-1]["cache_control"] = cc
    if tools:
        tools[-1]["cache_control"] = cc


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


def convert_response(data: dict[str, Any]) -> dict[str, Any]:
    """Normalise a Claude Messages API response to an OpenAI-ish shape.

    Extracts text, thinking, and tool_use blocks from ``content``.
    Maps ``stop_reason`` to ``finish_reason`` and ``input_tokens`` to
    ``prompt_tokens``.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in data.get("content", []):
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {}),
                },
            })

    # Map stop_reason -> finish_reason
    stop = data.get("stop_reason", "end_turn")
    finish_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "stop_sequence": "stop",
    }
    finish_reason = finish_map.get(stop, stop)

    usage_in = data.get("usage", {})
    # Anthropic splits cache stats into ``cache_creation_input_tokens``
    # (writes — billed at ~1.25× fresh) and ``cache_read_input_tokens``
    # (hits — billed at ~0.1× fresh).
    #
    # CRITICAL: Anthropic's ``input_tokens`` EXCLUDES both of those — it is
    # the freshly-evaluated remainder, not the prompt total. Treating it as
    # the total (the pre-2026-07 behaviour here) under-reported prompt_tokens
    # by the entire cached prefix — on a well-cached 200k turn that is a 99%
    # undercount — and made ``input_tokens - cache_read`` go negative, so the
    # max(0, …) clamp silently reported ZERO misses on every cache hit.
    # Reconstruct the true total the way the provider bills it.
    cache_read = int(usage_in.get("cache_read_input_tokens") or 0)
    cache_write = int(usage_in.get("cache_creation_input_tokens") or 0)
    raw_input = int(usage_in.get("input_tokens", 0) or 0)
    prompt_tokens = raw_input + cache_read + cache_write
    output_tokens = usage_in.get("output_tokens", 0)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "prompt_cache_hit_tokens": cache_read,
        # Misses are the genuinely-fresh remainder only. Writes are billed
        # ABOVE fresh rate, so folding them in here would understate cost.
        "prompt_cache_miss_tokens": raw_input,
        "prompt_cache_write_tokens": cache_write,
    }

    result: dict[str, Any] = {
        "content": "\n\n".join(text_parts),
        "thinking": "\n\n".join(thinking_parts) if thinking_parts else None,
        "finish_reason": finish_reason,
        "usage": usage,
        "model": data.get("model", ""),
    }
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


def _parse_data_uri(uri: str) -> tuple[str, str]:
    """Extract (media_type, base64_data) from a ``data:`` URI."""
    # data:image/png;base64,iVBOR...
    header, _, b64data = uri.partition(",")
    media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
    return media_type, b64data


def _to_content_array(content: Any) -> list[dict[str, Any]]:
    """Ensure content is in Claude's array-of-blocks format."""
    if isinstance(content, list):
        return deepcopy(content)
    text = str(content) if content else ZWS
    return [{"type": "text", "text": text}]


def _convert_image_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style image_url parts to Claude image blocks."""
    result: list[dict[str, Any]] = []
    for part in parts:
        if part.get("type") == "image_url":
            url = part.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                media_type, b64data = _parse_data_uri(url)
                result.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64data,
                    },
                })
            else:
                # URL-based images — pass as URL source
                result.append({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                })
        elif part.get("type") == "text":
            result.append({"type": "text", "text": part.get("text", "")})
        else:
            result.append(deepcopy(part))
    return result


def _convert_tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert OpenAI tool_calls to Claude tool_use content blocks."""
    blocks: list[dict[str, Any]] = []
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "input": args,
        })
    return blocks


def _convert_tool_result(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI tool-role message to a Claude user+tool_result message."""
    content_text = msg.get("content", "") or ZWS
    return {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": msg.get("tool_call_id", ""),
            "content": content_text,
        }],
    }


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


class ClaudeConverter:
    """Converts internal messages to Claude Messages API format.

    Implements the ``MessageConverter`` protocol.
    """

    def convert_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        prefill: str = "",
    ) -> dict[str, Any]:
        """Convert internal messages to Claude request payload parts.

        Returns a dict with ``system`` (list of content blocks) and
        ``messages`` (list of Claude messages).
        """
        msgs = deepcopy(messages)

        # 1. Extract leading system messages
        system_msgs, remaining = extract_system_prefix(msgs)

        # Build system content blocks
        system_blocks: list[dict[str, Any]] = []
        for sm in system_msgs:
            text = sm.get("content", "") or ZWS
            system_blocks.append({"type": "text", "text": text})

        # 2. Convert mid-conversation system messages to user role
        converted: list[dict[str, Any]] = []
        for msg in remaining:
            msg = deepcopy(msg)
            if msg.get("role") == "system":
                msg["role"] = "user"
            converted.append(msg)

        # 3. Handle names — prepend to content text
        converted = [prepend_name(m) for m in converted]

        # 4. Convert each message to Claude format
        claude_msgs: list[dict[str, Any]] = []
        for msg in converted:
            role = msg.get("role", "user")

            # Tool result messages
            if role == "tool":
                claude_msgs.append(_convert_tool_result(msg))
                continue

            # Build content
            raw_content = msg.get("content", "")
            if isinstance(raw_content, list):
                content = _convert_image_parts(raw_content)
            else:
                content = _to_content_array(raw_content)

            # Assistant messages with tool calls
            if role == "assistant" and msg.get("tool_calls"):
                tool_blocks = _convert_tool_calls(msg)
                # Include any text content before tool calls
                if content and content[0].get("text", "").strip():
                    claude_msgs.append({
                        "role": "assistant",
                        "content": content + tool_blocks,
                    })
                else:
                    claude_msgs.append({
                        "role": "assistant",
                        "content": tool_blocks,
                    })
                continue

            # Ensure non-empty content
            if not content:
                content = [{"type": "text", "text": ZWS}]

            claude_msgs.append({"role": role, "content": content})

        # 5. Merge consecutive same-role messages
        claude_msgs = _merge_claude_messages(claude_msgs)

        # 6. Optional prefill
        if prefill and claude_msgs:
            if claude_msgs[-1]["role"] == "assistant":
                # Append to existing assistant message
                claude_msgs[-1]["content"].append(
                    {"type": "text", "text": prefill}
                )
            else:
                claude_msgs.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": prefill}],
                })

        # 7. Ensure conversation starts with user
        if claude_msgs and claude_msgs[0]["role"] != "user":
            claude_msgs.insert(0, {
                "role": "user",
                "content": [{"type": "text", "text": ZWS}],
            })

        return {
            "system": system_blocks,
            "messages": claude_msgs,
        }

    def convert_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalise a Claude response to internal format."""
        return convert_response(data)


def _merge_claude_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge consecutive same-role Claude messages (content is always a list)."""
    if not messages:
        return []
    result: list[dict[str, Any]] = [deepcopy(messages[0])]
    for msg in messages[1:]:
        msg = deepcopy(msg)
        if result[-1]["role"] == msg["role"]:
            # Merge content lists
            prev_content = result[-1]["content"]
            new_content = msg["content"]
            if isinstance(prev_content, list) and isinstance(new_content, list):
                prev_content.extend(new_content)
            else:
                # Fallback — shouldn't happen with proper conversion
                result.append(msg)
        else:
            result.append(msg)
    return result
