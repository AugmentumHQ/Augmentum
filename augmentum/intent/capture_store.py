"""Opt-in training-data capture for the voice intent router.

When ``intent_capture_enabled`` is on, every voice_router verdict is appended
to the user-scoped ``intent_capture`` table: the transcript + the context
features the model saw, paired with the teacher model's goal/confidence. That
gives a ready-to-export ``(X, y)`` row for distilling a small on-device intent
classifier and publishing the set to HuggingFace.

Design notes:
  * **Fire-and-forget, never fatal.** A capture failure must never break the
    voice turn — every write is wrapped and logged at ``warning``.
  * **User-scoped.** Refuses to write without a real ``user_id`` (no anon row).
  * **No new connection.** Reuses the caller's aiosqlite connection
    (``state_manager.backend.conn``) like every other store.
"""

from __future__ import annotations

import time
import uuid

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_SCHEMA_VERSION = 1

# Voice-router goal vocabulary — mirrors architect/voice_router.py::_GOALS.
# A user correction must land on one of these so the exported label stays in
# the same space as the teacher's. Kept local (not imported) so this store
# has no dependency on the architect package.
VALID_GOALS: frozenset[str] = frozenset(("act", "converse", "clarify", "idle", "drop"))


async def record_intent_capture(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    session_id: str = "",
    surface: str = "voice_router",
    input_text: str = "",
    last_assistant_response: str = "",
    last_dispatch_summary: str = "",
    active_surface: str = "",
    seconds_since_last_tts: float | None = None,
    media_active: bool = False,
    explicit_capture: bool = False,
    goal: str = "",
    effective_goal: str = "",
    coherent: bool = True,
    addressed: bool = False,
    confidence: float = 0.0,
    teacher_model: str = "",
    parsed_from: str = "",
    reasoning: str = "",
    latency_ms: int = 0,
) -> None:
    """Append one routing decision to ``intent_capture``. Best-effort.

    Caller is responsible for gating on ``settings.intent_capture_enabled``;
    this function just refuses anon writes and swallows/logs any error so the
    voice path is never disrupted.
    """
    if not user_id:
        return  # never write into the anon row
    if not (input_text or "").strip():
        return  # nothing to learn from an empty transcript

    try:
        await conn.execute(
            """
            INSERT INTO intent_capture (
                id, user_id, session_id, surface,
                input_text, last_assistant_response, last_dispatch_summary,
                active_surface, seconds_since_last_tts, media_active, explicit_capture,
                goal, effective_goal, coherent, addressed, confidence,
                teacher_model, parsed_from, reasoning, latency_ms,
                corrected_goal, schema_version, captured_at
            ) VALUES (?, ?, ?, ?,  ?, ?, ?,  ?, ?, ?, ?,  ?, ?, ?, ?, ?,  ?, ?, ?, ?,  ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                user_id,
                session_id or "",
                surface or "voice_router",
                (input_text or "")[:2000],
                (last_assistant_response or "")[:2000],
                (last_dispatch_summary or "")[:1000],
                active_surface or "",
                float(seconds_since_last_tts) if seconds_since_last_tts is not None else None,
                1 if media_active else 0,
                1 if explicit_capture else 0,
                goal or "",
                effective_goal or "",
                1 if coherent else 0,
                1 if addressed else 0,
                float(confidence or 0.0),
                teacher_model or "",
                parsed_from or "",
                (reasoning or "")[:500],
                int(latency_ms or 0),
                "",  # corrected_goal — filled later by the correction flywheel
                _SCHEMA_VERSION,
                time.time(),
            ),
        )
        await conn.commit()
    except Exception as e:  # noqa: BLE001 — capture is never allowed to break voice
        log.warning("intent_capture_write_failed", error=str(e), surface=surface)


async def update_corrected_goal(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    capture_id: str,
    corrected_goal: str,
) -> bool:
    """Set the user's corrected label on one capture row.

    This is the correction flywheel: the export's supervised label is
    ``corrected_goal or effective_goal`` (intent_capture_routes.py), so a
    mislabelled voice route the user fixes becomes the gold target instead
    of the teacher's guess. The ``corrected_goal`` column shipped in
    migration 271 but had no writer until now (the dead-column gap in
    project_uncertainty_handling_map).

    User-scoped (never touches another user's rows). Returns True iff a row
    was updated. Best-effort — swallows/logs so a correction can't 500.
    """
    if not user_id or not capture_id:
        return False
    goal = (corrected_goal or "").strip().lower()
    if goal not in VALID_GOALS:
        return False
    try:
        cur = await conn.execute(
            "UPDATE intent_capture SET corrected_goal = ? WHERE id = ? AND user_id = ?",
            (goal, capture_id, user_id),
        )
        await conn.commit()
        return (cur.rowcount or 0) > 0
    except Exception as e:  # noqa: BLE001 — a correction must never break the caller
        log.warning("intent_capture_correct_failed", error=str(e), capture_id=capture_id)
        return False
