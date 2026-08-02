"""Capture-mode LLM cleanup pass.

When ``note.start_capture`` is active, every utterance is appended to
the active note as raw STT output. That produces a stream-of-thought
transcript with the usual STT artifacts: missing punctuation,
homophone errors ("their" / "there"), run-on sentences, no paragraph
breaks. Readable, but rough.

This module runs a single LLM call at end-of-capture to clean up the
dictated slice. The pass is deliberately conservative:

  * **Slice scope** — only ``content[note_capture_baseline_chars:]``
    is sent to the model. Pre-capture content (whatever the note held
    before the user said "take notes on this") is untouched.
  * **Semantic preservation** — the system prompt forbids changing
    meaning, adding new content, summarizing, or reordering thoughts.
    Punctuation, capitalization, paragraph breaks, and homophone
    repair are the entire mandate.
  * **Failure mode** — any timeout, parse error, or empty result
    leaves the raw transcript in place. Users never lose what they
    dictated to a flaky model call. The setting
    ``companion_note_capture_cleanup`` is a hard gate so operators
    can disable the pass entirely.

The cleaned text replaces the captured slice in place:
``content[:baseline] + cleaned``. The note's ``updated_at`` is
refreshed so downstream listeners (UI sticky overlay, file_index
metadata) pick up the change.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from augmentum.config import settings
from augmentum.models.base import InternalChatRequest, Message
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_CLEANUP_SYSTEM_PROMPT = (
    "You clean up raw speech-to-text dictation captured into a personal "
    "note. Your output replaces the raw transcript IN PLACE.\n\n"
    "STRICT RULES:\n"
    "1. Preserve every idea and word order. Do NOT summarize, "
    "paraphrase, reorder, add content, or remove content.\n"
    "2. Fix obvious homophone errors only when context makes them "
    "unambiguous (e.g. 'their/there/they're', 'to/too/two', "
    "'your/you're').\n"
    "3. Add sentence punctuation and capitalization where it's missing.\n"
    "4. Insert paragraph breaks between distinct thoughts when the "
    "transition is clear.\n"
    "5. Do NOT add headers, bullet lists, markdown, or commentary about "
    "what you changed.\n\n"
    "Reply with ONLY the cleaned text. No preamble, no explanation, "
    "no quoting."
)


async def cleanup_captured_text(
    raw: str,
    *,
    app_state: Any,
) -> str:
    """Return the cleaned version of ``raw``, or ``raw`` unchanged on failure.

    The function NEVER raises — capture mode loses notes if it does.
    Every error path returns the original text.
    """
    raw = raw or ""
    stripped = raw.strip()
    if not stripped:
        return raw

    if not getattr(settings, "companion_note_capture_cleanup", True):
        return raw

    registry = getattr(app_state, "provider_registry", None) if app_state else None
    if registry is None:
        log.debug("capture_cleanup_skipped_no_registry")
        return raw

    timeout_s = max(
        0.5,
        float(getattr(settings, "companion_note_capture_cleanup_timeout_ms", 8000)) / 1000.0,
    )

    try:
        backend, resolved_model = await registry.resolve_model_for_role(
            "utility",
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("capture_cleanup_resolve_failed", error=str(exc)[:160])
        return raw

    if backend is None:
        return raw

    req = InternalChatRequest(
        model=resolved_model or "",
        messages=[
            Message(role="system", content=_CLEANUP_SYSTEM_PROMPT),
            Message(role="user", content=stripped),
        ],
        stream=False,
        temperature=0.2,
        # No-thinking on this hop. The pass is mechanical clean-up; a
        # chain-of-thought trace burns the budget without improving
        # output. Models that don't honor the kwarg ignore it.
        chat_template_kwargs={"enable_thinking": False},
        # Soft ceiling — long captures need room. Cap at 4x the input
        # length plus padding so even a chatty mid-sized capture fits,
        # without giving the model headroom to add hundreds of new
        # tokens of "helpful" extra content.
        max_tokens=min(8192, max(512, len(stripped) // 2 + 256)),
    )

    try:
        resp = await asyncio.wait_for(backend.chat(req), timeout=timeout_s)
    except asyncio.TimeoutError:
        log.info(
            "capture_cleanup_timeout",
            model=resolved_model,
            input_chars=len(stripped),
            timeout_s=timeout_s,
        )
        return raw
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "capture_cleanup_backend_error",
            model=resolved_model, error=str(exc)[:200],
        )
        return raw

    message = getattr(resp, "message", None)
    content = (getattr(message, "content", "") or "").strip()
    if not content:
        log.info("capture_cleanup_empty_response", model=resolved_model)
        return raw

    # Sanity check: cleaned text shouldn't be wildly shorter than the
    # input (model summarized despite the prompt) or wildly longer
    # (model hallucinated additions). Allow 0.5x - 2.5x as the
    # acceptable envelope; outside this we fall back to raw to avoid
    # mangling the user's notes.
    in_len = len(stripped)
    out_len = len(content)
    if out_len < in_len * 0.5 or out_len > in_len * 2.5:
        log.info(
            "capture_cleanup_length_outlier_rejected",
            model=resolved_model,
            in_chars=in_len,
            out_chars=out_len,
        )
        return raw

    log.info(
        "capture_cleanup_ok",
        model=resolved_model,
        in_chars=in_len,
        out_chars=out_len,
    )
    return content


async def apply_cleanup_to_note(
    notes_store: Any,
    note_id: str,
    *,
    user_id: str,
    baseline_chars: int,
    app_state: Any,
) -> tuple[bool, str]:
    """Slice + clean + write back. Returns (changed, new_content).

    Returns ``(False, "")`` on any failure or no-op (cleanup disabled,
    baseline at or past end of content, capture empty). The caller can
    branch on ``changed`` to decide whether to re-emit a WS update to
    refresh the sticky overlay.
    """
    if notes_store is None or not note_id or not user_id:
        return False, ""
    try:
        note = await notes_store.get(note_id, user_id=user_id)
    except Exception:
        note = None
    if not note:
        return False, ""

    content = note.get("content") or ""
    if baseline_chars < 0 or baseline_chars >= len(content):
        # Nothing captured (baseline at or past end) — skip.
        return False, ""

    prefix = content[:baseline_chars]
    captured = content[baseline_chars:]
    cleaned = await cleanup_captured_text(captured, app_state=app_state)
    if cleaned == captured:
        return False, ""

    new_content = prefix + cleaned
    try:
        await notes_store.update(
            note_id,
            {
                "content": new_content,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            user_id=user_id,
        )
    except Exception as exc:
        log.warning(
            "capture_cleanup_update_failed",
            note_id=note_id, error=str(exc)[:200],
        )
        return False, ""
    return True, new_content
