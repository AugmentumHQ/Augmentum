"""Token accounting and deterministic compaction helpers for Coder context."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from augmentum.utils.tokenizer import count_tokens_messages


DEFAULT_CODER_COMPACT_TOKENS = 16_000
DEFAULT_CODER_DIGEST_TOKENS = 40_000
# Cap on the auto-derived compaction threshold. Bumped 192K → 256K on
# 2026-05-31 so the policy "10% reserve, full usable utilization"
# actually applies for 256K-class models (Claude Opus, Qwen 3.6 long-
# context variants) instead of getting clamped to 75% of window. The
# matching test (test_coder_context_limit_scales_with_model_window)
# expects the full 230-240K range for a 261K-window model.
MAX_DYNAMIC_COMPACT_TOKENS = 256_000
MAX_DYNAMIC_DIGEST_TOKENS = 96_000


def _positive_env_int(name: str) -> int | None:
    try:
        value = int(os.environ.get(name, "") or "0")
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _coerce_context_window(context_window: int | None = None) -> int:
    try:
        value = int(context_window or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, value)


def _resolve_reserve_pct() -> float:
    """Live-tunable reserve fraction. Reads
    ``settings.coder_context_reserve_pct``; falls back to 0.10 if unset
    or malformed. Bounded to (0.02, 0.40) so a misconfigured setting
    can't starve the model of output budget or shrink the usable
    window into uselessness.
    """
    default = 0.10
    try:
        from augmentum.config import settings as _settings
        raw = getattr(_settings, "coder_context_reserve_pct", default)
        pct = float(raw if raw is not None else default)
    except Exception:
        return default
    return max(0.02, min(0.40, pct))


def _context_reserve_tokens(context_window: int) -> int:
    """Reserve output/tool-schema/reasoning headroom inside a model window.

    Default is 10% of the window (was 15% before 2026-05-31). Bumped
    because 15% on a 131K-window model left only ~72K usable after the
    0.65 utilization multiplier — over a third of the window unused.
    User-tunable via ``coder_context_reserve_pct``.
    """
    context_window = _coerce_context_window(context_window)
    if context_window <= 0:
        return 0

    reserve = max(2_048, int(context_window * _resolve_reserve_pct()))
    reserve = min(reserve, 32_768)
    # On tiny windows, do not spend more than about a third on reserve.
    return min(reserve, max(1_024, context_window // 3))


def derive_coder_context_token_limit(context_window: int | None = None) -> int:
    """Derive the Coder auto-compaction threshold from a model window.

    Pre-2026-05-31 this returned ``int(usable * 0.65)``, leaving a
    second slice of the window unused beyond the reserve. The
    multiplier is now 1.0 — the compaction threshold equals the full
    usable region (window minus reserve). For a 131K-window model this
    moves the threshold from ~72K to ~118K, which is the actual
    intended "compact before I'd otherwise overflow" semantic.
    """
    context_window = _coerce_context_window(context_window)
    if context_window <= 0:
        return DEFAULT_CODER_COMPACT_TOKENS

    reserve = _context_reserve_tokens(context_window)
    usable = max(1_000, context_window - reserve)
    return max(1_000, min(usable, MAX_DYNAMIC_COMPACT_TOKENS))


def coder_context_token_limit(context_window: int | None = None) -> int:
    """Return the token ceiling used by the Coder auto-compactor.

    ``AUGMENTUM_CODER_COMPACT_TOKENS`` remains an explicit override.
    Without it, the ceiling scales from the active model context window
    while leaving headroom for tool schemas, outputs, and reasoning.
    """
    override = _positive_env_int("AUGMENTUM_CODER_COMPACT_TOKENS")
    if override is not None:
        return max(1_000, override)
    return derive_coder_context_token_limit(context_window)


def derive_coder_digest_token_budget(context_window: int | None = None) -> int:
    """Derive the all-or-nothing project-digest budget from a model window."""
    context_window = _coerce_context_window(context_window)
    if context_window <= 0:
        return DEFAULT_CODER_DIGEST_TOKENS

    reserve = _context_reserve_tokens(context_window)
    usable = max(1_000, context_window - reserve)
    compact_limit = derive_coder_context_token_limit(context_window)
    budget = int(usable * 0.35)
    budget = min(budget, int(compact_limit * 0.60), MAX_DYNAMIC_DIGEST_TOKENS)
    return max(1_000, budget)


def coder_digest_token_budget(context_window: int | None = None) -> int:
    """Return the token ceiling for the optional Coder project digest."""
    override = _positive_env_int("AUGMENTUM_CODER_DIGEST_BUDGET")
    if override is not None:
        return override
    budget = derive_coder_digest_token_budget(context_window)
    compact_override = _positive_env_int("AUGMENTUM_CODER_COMPACT_TOKENS")
    if compact_override is not None or _coerce_context_window(context_window) > 0:
        budget = min(budget, int(coder_context_token_limit(context_window) * 0.60))
    return max(1_000, budget)


def _content_of(message: dict | object) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _role_of(message: dict | object) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def _clone_message(message: dict | object) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    data: dict[str, Any] = {
        "role": _role_of(message),
        "content": _content_of(message),
    }
    for attr in ("tool_call_id", "name", "tool_calls"):
        value = getattr(message, attr, None)
        if value is not None:
            data[attr] = value
    return data


def _one_line(text: str, *, limit: int = 220) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "..."


def count_context_tokens(messages: list[dict | object]) -> int:
    """Count prompt-message tokens using Augmentum's shared tokenizer."""
    return count_tokens_messages(messages)


def token_budget_payload(
    messages: list[dict | object],
    *,
    scope: str,
    limit: int | None = None,
    context_window: int | None = None,
    iteration: int | None = None,
    compacted: bool = False,
) -> dict[str, Any]:
    """Build a stable token-budget payload for stream metadata/events."""
    ceiling = int(limit or coder_context_token_limit(context_window))
    count = count_context_tokens(messages)
    payload: dict[str, Any] = {
        "scope": scope,
        "tokens": count,
        "limit": ceiling,
        "ratio": round(count / ceiling, 4) if ceiling else 0.0,
        "message_count": len(messages),
        "compacted": bool(compacted),
    }
    window = _coerce_context_window(context_window)
    if window:
        payload["context_window"] = window
    if iteration is not None:
        payload["iteration"] = int(iteration)
    return payload


@dataclass(slots=True)
class ConversationCompactionResult:
    messages: list[dict[str, Any]]
    compacted: bool
    tokens_before: int
    tokens_after: int
    dropped_messages: int = 0
    summary_message: dict[str, Any] | None = None

    def to_payload(self, *, limit: int | None = None) -> dict[str, Any]:
        ceiling = int(limit or coder_context_token_limit())
        return {
            "compacted": self.compacted,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": max(0, self.tokens_before - self.tokens_after),
            "limit": ceiling,
            "messages_before": self.dropped_messages + len(self.messages)
            - (1 if self.summary_message else 0),
            "messages_after": len(self.messages),
            "dropped_messages": self.dropped_messages,
        }


def _is_runtime_carrier(message: dict | object) -> bool:
    from augmentum.modes.coder.chat_egress import RUNTIME_CARRIER_HEADER

    return (_content_of(message) or "").startswith(RUNTIME_CARRIER_HEADER)


def _is_compacted_block(message: dict | object) -> bool:
    return (_content_of(message) or "").lstrip().startswith("<compacted")


def _first_user_index(messages: list[dict | object]) -> int:
    """Index of the task definition — the first REAL user message.

    Skips per-turn runtime carriers (role=user scaffolding); without
    the skip, a carrier riding ahead of the task steals the preserved
    anchor slot and the actual task gets condensed away.
    """
    for i, message in enumerate(messages):
        if _role_of(message) == "user" and not _is_runtime_carrier(message):
            return i
    return 0 if messages else -1


# Immutable opening of the manual/one-shot compacted block. NO counts
# here — the block extends append-only (see the extend path in
# ``compact_conversation_messages``); a mutating header at the head of
# the block breaks the llama-server prefix cache on every re-compaction
# (measured 2026-07-02, kv_prefix_stability stable_pct 0.13).
_MANUAL_WRAPPER_OPEN = (
    "<compacted reason=\"manual-or-context-pressure\">\n"
    "Earlier messages were condensed to preserve context. Segments are "
    "ordered oldest first; messages after this block are verbatim. Use "
    "this as memory of the earlier chat; inspect files/tools again only "
    "when exact current state matters.\n\n"
)

_COMPACTED_CLOSER = "</compacted>"


def _render_condensed_segment(messages: list[dict | object]) -> str:
    """One immutable per-pass segment: header (counts live HERE, written
    once) + one line per condensed message."""
    lines: list[str] = []
    tool_count = 0
    for idx, message in enumerate(messages, start=1):
        role = _role_of(message) or "message"
        # Never one-line an existing compacted block (the extend path
        # excludes it positionally; this guards drifted histories) or a
        # stale runtime carrier (per-turn scaffolding, not chat).
        if _is_compacted_block(message) or _is_runtime_carrier(message):
            continue
        content = _one_line(_content_of(message))
        if role == "tool":
            tool_count += 1
        if not content:
            content = "(empty)"
        lines.append(f"- {idx}. {role}: {content}")
        if len(lines) >= 80:
            remaining = len(messages) - len(lines)
            if remaining > 0:
                lines.append(f"- ... {remaining} more messages omitted")
            break
    header_bits = [f"{len(messages)} messages"]
    if tool_count:
        header_bits.append(f"{tool_count} tool messages")
    return (
        f"## Condensed segment ({', '.join(header_bits)})\n"
        + "\n".join(lines)
    )


def _extend_compacted_content(existing: str, segment: str) -> str:
    """Append ``segment`` inside an existing block's wrapper, keeping
    every byte up to the old closer identical (prefix-cache safe)."""
    base = (existing or "").rstrip()
    if base.endswith(_COMPACTED_CLOSER):
        base = base[: -len(_COMPACTED_CLOSER)].rstrip("\n")
    return f"{base}\n\n{segment}\n{_COMPACTED_CLOSER}"


def compact_conversation_messages(
    messages: list[dict | object],
    *,
    keep_recent: int = 12,
    force: bool = False,
    limit: int | None = None,
) -> ConversationCompactionResult:
    """Compact a Coder conversation/history list deterministically.

    Preserves the first user turn plus the most recent ``keep_recent``
    messages, replacing the middle with one assistant summary message.
    This is deliberately non-LLM: it is safe to call from `/compact`
    and from request-prep paths without adding latency or model cost.
    """
    cloned = [_clone_message(m) for m in messages]
    ceiling = int(limit or coder_context_token_limit())
    before = count_context_tokens(cloned)

    if not cloned:
        return ConversationCompactionResult([], False, before, before)
    if not force and before < ceiling:
        return ConversationCompactionResult(cloned, False, before, before)

    keep_recent = max(4, min(int(keep_recent or 12), 40))
    first_user = _first_user_index(cloned)
    if first_user < 0:
        return ConversationCompactionResult(cloned, False, before, before)

    # Append-stable extension: an existing compacted block right after
    # the task definition (from a previous /compact OR the in-loop
    # auto-compactor — both start with ``<compacted``) is never
    # re-rendered. Re-condensing it crushed the whole prior summary to
    # one line AND rewrote the head of history, invalidating the
    # llama-server prefix cache (measured 2026-07-02). Instead condense
    # only the messages AFTER it and append a new segment inside its
    # wrapper.
    existing_idx = -1
    if first_user + 1 < len(cloned) and _is_compacted_block(cloned[first_user + 1]):
        existing_idx = first_user + 1

    region_start = (existing_idx if existing_idx >= 0 else first_user) + 1
    tail_start = max(region_start, len(cloned) - keep_recent)
    dropped = cloned[region_start:tail_start]
    if len(dropped) < 2:
        return ConversationCompactionResult(cloned, False, before, before)

    segment = _render_condensed_segment(dropped)
    if existing_idx >= 0:
        existing = cloned[existing_idx]
        content = _extend_compacted_content(_content_of(existing), segment)
        role = _role_of(existing) or "assistant"
    else:
        content = _MANUAL_WRAPPER_OPEN + segment + "\n" + _COMPACTED_CLOSER
        role = "assistant"
    summary = {
        "id": f"msg_compact_{int(time.time() * 1000)}",
        "role": role,
        "content": content,
        "metadata": {
            "compacted": True,
            "dropped_messages": len(dropped),
            "tokens_before": before,
        },
    }
    compacted = cloned[: region_start] + [summary] + cloned[tail_start:]
    if existing_idx >= 0:
        # The extended summary REPLACES the old block (it contains it
        # as a byte-prefix) — drop the stale copy that region_start
        # left in place.
        del compacted[existing_idx]
    after = count_context_tokens(compacted)
    # Even an explicit /compact should never make the next prompt larger.
    # "force" means "try even when under the configured ceiling", not
    # "accept a counterproductive summary".
    if after >= before:
        return ConversationCompactionResult(cloned, False, before, before)
    summary["metadata"]["tokens_after"] = after
    return ConversationCompactionResult(
        compacted,
        True,
        before,
        after,
        dropped_messages=len(dropped),
        summary_message=summary,
    )
