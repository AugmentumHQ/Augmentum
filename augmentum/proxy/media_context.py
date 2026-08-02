"""Inject "currently listening to X" context into chat requests.

Small, provider-agnostic helper the chat routes call when the frontend
sends ``X-Augmentum-Media-Context``. Decodes a minimal JSON blob from
the header and prepends a short system message so the LLM knows what
book the user is in — no explicit "which book" prompting required.

Design constraints (from the "works on 4B" requirement):

- Pure string formatting — no LLM calls, no DB lookups here.
- Produces a single factual sentence. Small models latch onto this
  without being confused, large models use it as soft grounding.
- Silently no-ops when the header is missing / malformed — never
  corrupts a request.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from augmentum.models.base import Message
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request

    from augmentum.models.internal import InternalChatRequest

log = get_logger(__name__)


def _fmt_timecode(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, r = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{r:02d}"
    return f"{m}:{r:02d}"


def _build_sentence(ctx: dict) -> str:
    title = str(ctx.get("title") or "").strip()
    if not title:
        return ""
    author = str(ctx.get("author") or "").strip()
    chapter_idx = ctx.get("chapterIdx")
    chapter_title = str(ctx.get("chapterTitle") or "").strip()
    current_s = float(ctx.get("currentTimeS") or 0)
    is_playing = bool(ctx.get("isPlaying"))

    parts = [f'The user is currently listening to "{title}"']
    if author:
        parts.append(f"by {author}")
    if chapter_idx is not None and chapter_title:
        parts.append(f"at chapter {int(chapter_idx) + 1}: \"{chapter_title}\"")
    elif chapter_title:
        parts.append(f"at \"{chapter_title}\"")

    state = "playing" if is_playing else "paused"
    if current_s > 0:
        parts.append(f"({state} at {_fmt_timecode(current_s)})")
    else:
        parts.append(f"({state})")
    return " ".join(parts) + "."


def inject_media_context(
    internal_req: InternalChatRequest,
    request: Request,
    mode: str = "",
) -> None:
    """Prepend a 'currently listening to X' system message if the header is set.

    Companion-scoped: this is ambient presence awareness ("what are you
    hearing right now"), which belongs to the companion (``becca_direct``),
    NOT to every route. Injecting it into narrative polluted the story /
    leaked into RP reasoning, and — because it lands at system index 0 and
    changes every turn as playback advances — it invalidated the whole KV
    prefix (``kv_prefix_stability`` contract=violated, stable_pct ~0.006).
    So we no-op for every mode except the companion. Other modes that want
    media grounding should ask for it via a tool, not a forced prefix.

    Also no-ops on missing / malformed headers. Idempotent per request (we
    don't dedupe across requests — a new header every turn is expected
    as the user's position advances).
    """
    if mode != "becca_direct":
        return
    raw = request.headers.get("X-Augmentum-Media-Context", "")
    if not raw:
        return
    try:
        ctx = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        log.debug("media_context_header_invalid", raw=raw[:200])
        return
    if not isinstance(ctx, dict):
        return

    sentence = _build_sentence(ctx)
    if not sentence:
        return

    # Prepend a single system message. We don't merge into an existing
    # system message because:
    #   1) model providers differ on whether multiple systems are joined
    #      (OpenAI: yes; Ollama: depends on backend tokenizer).
    #   2) Keeping this as its own line makes it easy to audit in logs.
    # InternalChatRequest.messages is list[Message] (Pydantic), not a
    # list of dicts — inserting a bare dict here poisons the list and
    # every downstream `msg.role` access raises AttributeError.
    try:
        msgs = list(internal_req.messages or [])
        msgs.insert(0, Message(role="system", content=sentence))
        internal_req.messages = msgs
    except Exception as exc:
        log.debug("media_context_inject_failed", error=str(exc))
