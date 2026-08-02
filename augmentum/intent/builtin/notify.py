"""Notification verbs — mute a channel, dismiss from the feed.

Wiring program Phase 1 (2026-06-12). The notification substrate
(``augmentum/notifications/``) has had per-channel mutes and dismiss
since it shipped, but only the bell UI could reach them — "mute the
job alerts" out loud went nowhere. These verbs wrap the same store
functions the ``/api/notify`` routes use (not HTTP — same process,
same conn), honoring the same ``notifications_enabled`` gate.

Both are Tier-3 only: "dismiss that" is referent-dependent and
"mute X" carries a free-slot channel name — exactly the shapes the
no-regex-switchboard rule keeps away from Tier 1.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)

# "Mute forever" sentinel — same convention the bell UI writes.
_FOREVER_ISO = "9999-12-31T00:00:00+00:00"


def _conn(session: SessionContext):
    """Resolve the aiosqlite conn the notification store needs.

    Mirrors notifications_routes._get_conn: the substrate is
    persistence-bound, so a non-SQLite backend (in-memory tests)
    yields None and the verbs answer honestly instead of pretending.
    """
    sm = getattr(session.app_state, "state_manager", None) if session.app_state else None
    backend = getattr(sm, "backend", None)
    return getattr(backend, "conn", None)


def _enabled() -> bool:
    from augmentum.config import settings
    return bool(getattr(settings, "notifications_enabled", False))


def _gate(session: SessionContext) -> ActionResult | None:
    if not _enabled():
        return ActionResult(
            short_circuit=True,
            speak="Notifications are turned off in settings.",
        )
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I'm not sure whose notifications to touch.",
        )
    if _conn(session) is None:
        return ActionResult(
            short_circuit=True,
            speak="I can't reach the notification store right now.",
        )
    return None


# ---------------------------------------------------------------------------
# notify.mute
# ---------------------------------------------------------------------------

async def _notify_mute(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    blocked = _gate(session)
    if blocked is not None:
        return blocked
    from augmentum.notifications.store import mute_channel, resolved_channels

    conn = _conn(session)
    channels = await resolved_channels(conn, user_id=session.user_id)
    query = str(args.get("channel") or "").strip().lower()
    if not query:
        names = ", ".join(c.name for c in channels[:6]) or "none yet"
        return ActionResult(
            short_circuit=True,
            speak=f"Which notifications — {names}?",
            clarify={"missing": ["channel"], "args": dict(args)},
        )

    matches = [
        c for c in channels
        if query in c.name.lower() or query in c.channel_id.lower()
    ]
    if not matches:
        names = ", ".join(c.name for c in channels[:6]) or "none configured"
        return ActionResult(
            short_circuit=True,
            speak=(
                f"I don't see a notification channel like "
                f"{query[:40]} — there's {names}."
            ),
        )
    if len(matches) > 1:
        names = ", or ".join(c.name for c in matches[:4])
        return ActionResult(
            short_circuit=True,
            speak=f"Which one — {names}?",
            clarify={"missing": ["channel"], "args": dict(args)},
        )

    target = matches[0]
    unmute = bool(args.get("off"))
    hours = args.get("hours")
    if unmute:
        until_iso: str | None = None
        spoken = f"Unmuted {target.name}."
    elif hours:
        try:
            h = max(0.25, float(hours))
        except (TypeError, ValueError):
            h = 1.0
        until = datetime.now(UTC) + timedelta(hours=h)
        until_iso = until.isoformat()
        nice = f"{int(h)} hours" if h >= 2 else (
            "an hour" if h >= 1 else f"{int(h * 60)} minutes"
        )
        spoken = f"Muted {target.name} for {nice}."
    else:
        until_iso = _FOREVER_ISO
        spoken = f"Muted {target.name} until you turn them back on."

    await mute_channel(
        conn, user_id=session.user_id,
        channel_id=target.channel_id, until_iso=until_iso,
    )
    log.info(
        "notify_mute_verb",
        user_id=session.user_id, channel_id=target.channel_id,
        unmute=unmute,
    )
    return ActionResult(
        short_circuit=True,
        speak=spoken,
        digest=f"{'unmuted' if unmute else 'muted'} {target.name} notifications",
    )


register_action(
    id="notify.mute",
    summary=(
        "Silently mute (or unmute) one of the user's notification "
        "channels — job alerts, companion notes, system warnings — "
        "for a while or until turned back on. Call for 'mute the job "
        "notifications', 'stop pinging me about downloads', 'turn "
        "those alerts back on'. Sibling: making ONE notification go "
        "away is notify.dismiss."
    ),
    examples=[
        "mute the job notifications", "stop notifying me about downloads",
        "silence those alerts for a couple hours",
        "turn the system notifications back on",
    ],
    arg_schema={
        "channel": {
            "type": "string",
            "description": (
                "Which notifications, in the user's words — matched "
                "against the channel names."
            ),
        },
        "hours": {
            "type": "number",
            "description": "Mute duration in hours. Omit = indefinitely.",
        },
        "off": {
            "type": "boolean",
            "description": "True to UNMUTE the channel instead.",
        },
    },
    fanout=_TIER3_ONLY,
    handler=_notify_mute,
    delivery="verbal",
)


# ---------------------------------------------------------------------------
# notify.dismiss
# ---------------------------------------------------------------------------

async def _notify_dismiss(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    blocked = _gate(session)
    if blocked is not None:
        return blocked
    from augmentum.notifications.store import dismiss, list_for_user

    conn = _conn(session)
    items = await list_for_user(
        conn, user_id=session.user_id,
        include_read=True, include_dismissed=False, limit=50,
    )
    if not items:
        return ActionResult(
            short_circuit=True,
            speak="Nothing's waiting in your notifications.",
        )

    query = str(args.get("query") or "").strip().lower()
    if query:
        items = [
            n for n in items
            if query in n.title.lower() or query in (n.body or "").lower()
        ]
        if not items:
            return ActionResult(
                short_circuit=True,
                speak=f"No notification mentions {query[:40]}.",
            )

    if bool(args.get("all")):
        count = 0
        for n in items:
            if await dismiss(
                conn, user_id=session.user_id,
                notification_id=n.notification_id,
            ):
                count += 1
        log.info("notify_dismiss_all_verb", user_id=session.user_id, count=count)
        return ActionResult(
            short_circuit=True,
            speak=f"Cleared {count} notifications.",
            digest=f"dismissed {count} notifications",
        )

    newest = items[0]  # feed is newest-first
    await dismiss(
        conn, user_id=session.user_id,
        notification_id=newest.notification_id,
    )
    log.info(
        "notify_dismiss_verb",
        user_id=session.user_id, notification_id=newest.notification_id,
    )
    return ActionResult(
        short_circuit=True,
        speak=f"Dismissed: {newest.title[:80]}.",
        digest=f"dismissed notification: {newest.title[:60]}",
    )


register_action(
    id="notify.dismiss",
    summary=(
        "Silently dismiss notifications from the user's feed — the "
        "newest one, everything, or ones matching a phrase. Call for "
        "'dismiss that', 'clear my notifications', 'get rid of the "
        "download ones'. Sibling: silencing a channel going forward "
        "is notify.mute."
    ),
    examples=[
        "dismiss that notification", "clear my notifications",
        "get rid of the download alerts", "clear all of those",
    ],
    arg_schema={
        "all": {
            "type": "boolean",
            "description": "Dismiss everything currently in the feed.",
        },
        "query": {
            "type": "string",
            "description": (
                "Only dismiss notifications mentioning this phrase. "
                "Omit with all=false to dismiss just the newest one."
            ),
        },
    },
    fanout=_TIER3_ONLY,
    handler=_notify_dismiss,
    delivery="verbal",
)
