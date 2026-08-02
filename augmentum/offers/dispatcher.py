"""Offer dispatcher — the entry point ``propose_offer`` tool calls.

Walks the substrate from the tool call to the published notification:

1. **Catalog lookup.** ``(kind, target_id)`` → ``CatalogEntry`` via the
   global registry. Unknown → return ``ok=False, reason="unknown"``.
2. **Suppression check.** If the user previously chose *Never* on this
   (kind, target_id), return ``ok=False, suppressed=True``. Only Never
   writes a suppression — "Not now" dismisses one chip and leaves no
   trace, so declining an offer never blocks the next request.
3. **Rate limits.** Per-turn cap, per-session pending cap, per-day
   global cap. Each return a ``ok=False, reason="rate_limit:..."``
   on overshoot.
4. **Preview build.** Catalog entry decides whether it's relevant
   right now (e.g. MCP server already installed → not relevant);
   ``None`` from the builder is "skip without surfacing."
5. **Publish.** ``publish_and_dispatch`` writes to the
   ``notifications`` table on channel ``system.offer`` and fans out
   via the existing notification hub.

The tool result is returned to the model so it can adjust prose when
it sees ``suppressed=True`` or a rate-limit reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from augmentum.notifications.hub import NotificationHub, publish_and_dispatch
from augmentum.notifications.store import NotificationAction
from augmentum.offers.catalog.base import get_entry
from augmentum.offers.store import is_suppressed
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite


log = get_logger(__name__)


OFFER_CHANNEL_ID: str = "system.offer"
OFFER_SOURCE: str = "chat.propose_offer"


PROPOSE_OFFER_RESULT_KEYS = ("ok", "offer_id", "suppressed", "reason")


@dataclass(frozen=True)
class ProposeOfferResult:
    """Return shape for ``propose_offer``.

    Modeled as a dataclass for tests; the tool layer flattens to a
    plain dict before returning to the chat LLM.
    """

    ok: bool
    offer_id: str = ""
    suppressed: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "offer_id": self.offer_id,
            "suppressed": self.suppressed,
            "reason": self.reason,
        }


# ── Rate-limit primitives ────────────────────────────────────────


async def _count_pending_offers(
    conn: aiosqlite.Connection, *, user_id: str, thread_id: str = "",
) -> int:
    """Count non-dismissed offers for this user (optionally per session).

    Pending = ``dismissed_at IS NULL``. Read state doesn't matter —
    a read-but-undismissed offer still counts against the cap because
    it still occupies a chip slot in the chat stream.
    """

    if thread_id:
        cur = await conn.execute(
            """SELECT COUNT(*) FROM notifications
                WHERE user_id = ? AND channel_id = ?
                  AND dismissed_at IS NULL
                  AND thread_id = ?""",
            (user_id, OFFER_CHANNEL_ID, thread_id),
        )
    else:
        cur = await conn.execute(
            """SELECT COUNT(*) FROM notifications
                WHERE user_id = ? AND channel_id = ?
                  AND dismissed_at IS NULL""",
            (user_id, OFFER_CHANNEL_ID),
        )
    row = await cur.fetchone()
    await cur.close()
    return int(row[0]) if row else 0


async def _count_offers_today(
    conn: aiosqlite.Connection, *, user_id: str,
) -> int:
    """Count offers created today (UTC) — for the per-day global cap.

    Counts both still-pending and already-actioned rows. The user-
    intent here is "stop a chatty model from emitting 50 offers a
    day even if the user accepts each one quickly."
    """

    start_of_day = datetime.now(UTC).replace(
        hour=0, minute=0, second=0, microsecond=0,
    ).isoformat()
    cur = await conn.execute(
        """SELECT COUNT(*) FROM notifications
            WHERE user_id = ? AND channel_id = ?
              AND created_at >= ?""",
        (user_id, OFFER_CHANNEL_ID, start_of_day),
    )
    row = await cur.fetchone()
    await cur.close()
    return int(row[0]) if row else 0


# ── Per-turn counter (in-memory, per-handler) ────────────────────


# The chat handler stamps the current turn id on the request scope
# before invoking tools; the dispatcher counts emissions per turn id
# in a process-local map. Stale entries are pruned by the LRU cap
# below — no explicit cleanup is needed because the keys are unique
# per turn and a turn never spans process restart.
_PER_TURN_COUNTS: dict[str, int] = {}
_PER_TURN_LRU_CAP: int = 4096


def _bump_turn_count(turn_id: str) -> int:
    if not turn_id:
        return 0
    n = _PER_TURN_COUNTS.get(turn_id, 0) + 1
    _PER_TURN_COUNTS[turn_id] = n
    # Prune oldest entries when the dict bloats. Dict iteration in
    # Python 3.7+ is insertion-ordered, so popping items from the
    # front behaves as an LRU.
    while len(_PER_TURN_COUNTS) > _PER_TURN_LRU_CAP:
        try:
            oldest = next(iter(_PER_TURN_COUNTS))
            _PER_TURN_COUNTS.pop(oldest, None)
        except StopIteration:
            break
    return n


def reset_turn_counts() -> None:
    """Test helper — clear the in-memory turn counter."""

    _PER_TURN_COUNTS.clear()


# ── Main entrypoint ──────────────────────────────────────────────


def _expires_at(days: int) -> str:
    return (datetime.now(UTC) + timedelta(days=int(days))).isoformat()


async def propose_offer(
    conn: aiosqlite.Connection,
    *,
    hub: NotificationHub | None,
    user_id: str,
    kind: str,
    target_id: str,
    reason: str,
    extra: dict[str, Any] | None = None,
    thread_id: str = "",
    turn_id: str = "",
    mode: str = "",
    max_per_turn: int = 2,
    max_pending_per_session: int = 5,
    max_per_day: int = 20,
    expiry_days: int = 7,
) -> ProposeOfferResult:
    """Validate, gate, and (if everything passes) publish the offer.

    ``hub`` may be ``None`` in tests that don't care about live
    fan-out — the persisted row is still the source of truth, so the
    chip still appears on the next ``/api/notify/feed`` poll. The
    dispatcher does NOT fail when the hub is missing.

    Caller-supplied caps are pulled from settings at the call site —
    so they live-update without restarting the process.
    """

    if not user_id:
        return ProposeOfferResult(ok=False, reason="missing_user")
    if not kind or not target_id:
        return ProposeOfferResult(ok=False, reason="missing_target")

    entry = get_entry(kind, target_id)
    if entry is None:
        return ProposeOfferResult(ok=False, reason="unknown_target")

    # Mode-context gate. Catalog entries can declare ``allowed_modes``
    # to restrict where they're sensible to surface — e.g. powers only
    # in coder mode, since "enable a power" from passthrough creates
    # an inert chip the user can accept but nothing observably changes.
    # Empty ``allowed_modes`` (the default) means any mode; empty
    # ``mode`` from the caller falls open so non-handler call paths
    # (tests, scripts) aren't broken.
    if entry.allowed_modes and mode and mode not in entry.allowed_modes:
        return ProposeOfferResult(ok=False, reason="mode_mismatch")

    # Suppression check first — cheapest gate and the one that most
    # respects user intent. Always runs even if the model is spamming.
    if await is_suppressed(
        conn, user_id=user_id, kind=kind, target_id=target_id,
    ):
        return ProposeOfferResult(ok=False, suppressed=True, reason="suppressed")

    # Per-turn cap (in-memory, fast).
    if turn_id and _bump_turn_count(turn_id) > int(max_per_turn):
        return ProposeOfferResult(ok=False, reason="rate_limit:turn")

    # Per-session pending cap (DB query — one count).
    pending = await _count_pending_offers(
        conn, user_id=user_id, thread_id=thread_id or "",
    )
    if pending >= int(max_pending_per_session):
        return ProposeOfferResult(ok=False, reason="rate_limit:pending")

    # Per-day global cap.
    today = await _count_offers_today(conn, user_id=user_id)
    if today >= int(max_per_day):
        return ProposeOfferResult(ok=False, reason="rate_limit:day")

    # Build the preview through the catalog entry. None means "skip
    # — not relevant for this user right now."
    preview = None
    if entry.build_preview is not None:
        preview = await entry.build_preview(target_id, user_id)
        if preview is None:
            return ProposeOfferResult(ok=False, reason="not_relevant")

    # Compose the notification payload. The action handler reads it
    # to know which catalog entry to invoke on Accept.
    payload: dict[str, Any] = {
        "kind": kind,
        "target_id": target_id,
        "scope": entry.scope,
    }
    if preview is not None:
        preview_dict = preview.to_dict()
        if preview_dict:
            payload["preview"] = preview_dict
    if extra:
        payload["extra"] = dict(extra)

    actions = [
        NotificationAction(id="accept", label="Install", style="primary"),
        NotificationAction(id="snooze", label="Not now", style="default"),
        NotificationAction(id="never", label="Never", style="ghost"),
    ]

    # ``publish_and_dispatch`` is hub-tolerant — pass a real hub when
    # one's available, fall back to a fresh empty hub when not (the
    # dispatch attempt is a no-op since there are no subscribers).
    dispatch_hub = hub if hub is not None else NotificationHub()

    offer_id = await publish_and_dispatch(
        conn,
        hub=dispatch_hub,
        user_id=user_id,
        channel_id=OFFER_CHANNEL_ID,
        source=OFFER_SOURCE,
        title=entry.title,
        body=reason or "",
        dedupe_key=f"{kind}:{target_id}",
        thread_id=thread_id or "",
        actions=actions,
        payload=payload,
        expires_at=_expires_at(expiry_days),
        icon=entry.icon or "",
    )

    log.info(
        "offer_proposed",
        user_id=user_id,
        kind=kind,
        target_id=target_id,
        offer_id=offer_id,
        scope=entry.scope,
    )
    return ProposeOfferResult(ok=True, offer_id=offer_id)
