"""Shared utilities for message conversion.

Functions for merging, alternating, and normalizing message lists
before they are sent to LLM providers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from augmentum.models.converters.base import PostProcessMode

# Zero-width space used as placeholder content for inserted messages.
ZWS = "\u200b"


def merge_consecutive_messages(
    messages: list[dict[str, Any]],
    separator: str = "\n\n",
) -> list[dict[str, Any]]:
    """Merge consecutive messages that share the same role.

    Args:
        messages: List of message dicts.
        separator: String inserted between merged content blocks.

    Returns:
        New list with consecutive same-role messages combined.
    """
    if not messages:
        return []

    result: list[dict[str, Any]] = []
    for msg in messages:
        msg = deepcopy(msg)
        if result and result[-1]["role"] == msg["role"]:
            result[-1]["content"] = result[-1]["content"] + separator + msg["content"]
        else:
            result.append(msg)
    return result


def prepend_name(msg: dict[str, Any]) -> dict[str, Any]:
    """If a message has a ``name`` field, prepend it to content and remove it.

    Args:
        msg: A single message dict.

    Returns:
        New message dict with name prepended to content (or unchanged copy).
    """
    msg = deepcopy(msg)
    name = msg.pop("name", None)
    if name:
        msg["content"] = f"{name}: {msg['content']}"
    return msg


def force_alternating(
    messages: list[dict[str, Any]],
    *,
    placeholder: str = ZWS,
) -> list[dict[str, Any]]:
    """Ensure strict user/assistant alternation by inserting placeholders.

    When two consecutive messages share a role, a placeholder message with
    the opposite role is inserted between them.

    Args:
        messages: List of message dicts (system messages should already
            be extracted or converted before calling this).
        placeholder: Content for inserted filler messages.

    Returns:
        New list with strict alternation guaranteed.
    """
    if not messages:
        return []

    result: list[dict[str, Any]] = [deepcopy(messages[0])]
    for msg in messages[1:]:
        msg = deepcopy(msg)
        prev_role = result[-1]["role"]
        if msg["role"] == prev_role:
            # Insert opposite role as filler.
            filler_role = "assistant" if prev_role == "user" else "user"
            result.append({"role": filler_role, "content": placeholder})
        result.append(msg)
    return result


def extract_system_prefix(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split leading system messages from the rest.

    Args:
        messages: Full message list.

    Returns:
        Tuple of (system_prefix, remaining_messages). Both are new lists
        (deep-copied) so the caller can mutate freely.
    """
    system: list[dict[str, Any]] = []
    idx = 0
    for msg in messages:
        if msg.get("role") == "system":
            system.append(deepcopy(msg))
            idx += 1
        else:
            break
    remaining = deepcopy(messages[idx:])
    return system, remaining


def _convert_mid_system_to_user(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert non-leading system messages to user role.

    Leading system messages are preserved; any system message after the
    first non-system message is converted to ``user`` role.
    """
    result: list[dict[str, Any]] = []
    past_prefix = False
    for msg in messages:
        msg = deepcopy(msg)
        if not past_prefix and msg["role"] == "system":
            result.append(msg)
            continue
        past_prefix = True
        if msg["role"] == "system":
            msg["role"] = "user"
        result.append(msg)
    return result


def post_process(
    messages: list[dict[str, Any]],
    mode: PostProcessMode,
) -> list[dict[str, Any]]:
    """Apply a SillyTavern-style post-processing pipeline.

    Modes:
        NONE:   Return messages unchanged (deep copy).
        MERGE:  Prepend names, then merge consecutive same-role messages.
        SEMI:   Convert mid-conversation system→user, then apply MERGE.
        STRICT: Apply SEMI, then insert placeholders for strict alternation.

    Args:
        messages: List of message dicts.
        mode: Processing mode to apply.

    Returns:
        New list of processed messages.
    """
    if mode == PostProcessMode.NONE:
        return deepcopy(messages)

    result = messages

    # SEMI and STRICT: convert mid-conversation system to user first.
    if mode in (PostProcessMode.SEMI, PostProcessMode.STRICT):
        result = _convert_mid_system_to_user(result)

    # MERGE, SEMI, STRICT: prepend names then merge.
    result = [prepend_name(m) for m in result]
    result = merge_consecutive_messages(result)

    # STRICT: force alternation.
    if mode == PostProcessMode.STRICT:
        result = force_alternating(result)

    return result
