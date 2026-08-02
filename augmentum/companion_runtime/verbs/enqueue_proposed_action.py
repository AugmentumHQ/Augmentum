"""enqueue_proposed_action — Phase 3c consumer verb.

Subscribes to ``companion.action_proposed`` (emitted by
:mod:`propose_action`) and persists the proposal as a row in
``companion_initiative_queue``. Closes the open end of the
substrate loop: substrate-threshold-cross → proposed-action event →
queued row → existing initiative surfacing path picks it up.

Why a separate verb rather than folding the write into propose_action:
the two have different responsibilities and different blast radii.
``propose_action`` is pure substrate-to-substrate (no DB write, read-
only safety class). Enqueuing a row is a write that can show up in
the user's notes / today reflection / glance HUD. Splitting the
write into its own verb keeps the kill switches independent — you
can flip propose_action on first, watch it land in the verb_log,
and only flip enqueue_proposed_action when the proposal quality
looks right.

The schema (`migration 156`) allows the kinds share|wonder|offer|
notice|... — the proposed activity kind from propose_action
(revisit_thread / creation / reach_out / no_op) is carried verbatim
in the payload, and the row's ``kind`` is the broader category
('offer' for actionable proposals, 'notice' for no-op).
"""

from __future__ import annotations

import json
import time

from augmentum.companion_runtime.event_bus import (
    DispatchClass,
    SafetyClass,
    verb,
)
from augmentum.companion_runtime.verbs import VerbRegistry
from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Broader-category mapping so the initiative-surfacing path can
# distinguish actionable offers from passive notices. ``no_op``
# proposals (the propose_action floor short-circuit) are still
# worth queuing as 'notice' so the surfacing UI can show that
# Becca considered something but chose to hold.
_KIND_CATEGORY = {
    "revisit_thread": "offer",
    "creation": "offer",
    "reach_out": "offer",
    "no_op": "notice",
}


# 5 minute cooldown — the upstream propose_action already throttles
# at 10 min, so this is belt-and-braces against a future second
# producer (e.g., a sibling verb in Phase 4+) flooding the queue.
_ENQUEUE_COOLDOWN_MS = 5 * 60 * 1000


@verb(
    "companion.action_proposed",
    name="enqueue_proposed_action",
    reads=(),
    writes=("companion_initiative_queue",),
    dispatch_class=DispatchClass.EVENT_DRIVEN,
    safety_class=SafetyClass.WRITE_SELF,
    cooldown_ms=_ENQUEUE_COOLDOWN_MS,
)
async def enqueue_proposed_action(event, ctx) -> None:
    """Write one ``companion.action_proposed`` event to the queue."""
    if not getattr(settings, "companion_action_enqueue_enabled", False):
        return

    runtime = ctx.runtime
    owner = getattr(runtime, "owner_user_id", "") or ""
    if not owner:
        return

    payload_in = getattr(event, "payload", None) or {}
    proposed_kind = str(payload_in.get("kind") or "").strip()
    drive = str(payload_in.get("drive") or "").strip()
    try:
        urgency = float(payload_in.get("urgency") or 0.0)
    except (TypeError, ValueError):
        urgency = 0.0

    if not proposed_kind:
        log.debug("enqueue_proposed_action_no_kind", payload=payload_in)
        return

    category = _KIND_CATEGORY.get(proposed_kind, "offer")
    payload_out = {
        "proposed_kind": proposed_kind,
        "drive": drive,
        "urgency": round(urgency, 3),
        "trigger_field": payload_in.get("trigger_field"),
        "source": "propose_action",
    }
    payload_json = json.dumps(payload_out, separators=(",", ":"))

    backend = runtime.backend
    try:
        cur = await backend.conn.execute(
            "INSERT INTO companion_initiative_queue "
            "(companion_id, proposed_at, kind, payload, importance, "
            "score, status, target_user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                runtime.companion_id,
                time.time(),
                category,
                payload_json,
                "low",
                round(urgency, 3),
                "queued",
                owner,
            ),
        )
        row_id = cur.lastrowid
        await backend.conn.commit()
        await cur.close()
    except Exception:
        log.exception("enqueue_proposed_action_failed", owner=owner)
        return

    ctx.cite("companion_initiative_queue", row_id=row_id)
    ctx.db_ops += 1
    log.debug(
        "enqueue_proposed_action_queued",
        row_id=row_id,
        user_id=owner,
        category=category,
        proposed_kind=proposed_kind,
        drive=drive,
        urgency=round(urgency, 3),
    )


VerbRegistry.register(enqueue_proposed_action)
