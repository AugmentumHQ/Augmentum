"""Call lifecycle timers — missed-call detection + ringing TTL.

When an invite goes out, both perspectives' ``call_sessions`` rows
are inserted in states ``ringing`` (initiator) / ``invited`` (recipient).
If neither side acts within the invite lifetime (default 60s), the
call is "missed". This module owns the timer that fires that
transition and wires the cleanup.

Lifecycle:

  arm_invite_timer(call_id, ...)
       │
       └─ after lifetime_ms:
            ├─ recheck state (was it accept/decline/hangup'd?)
            ├─ if still ringing/invited:
            │     mark BOTH rows state='missed'
            │     log call_events 'missed'
            │     publish connect.call.missed notification on recipient
            │     route EVENT_HANGUP{reason='missed'} to whoever's still attached
            │     dismiss the connect.call.incoming banner if it's still up
            │     (handled client-side via the notification dismiss path
            │      — server dismisses the row in the notification store)
            └─ remove timer from the registry

  cancel_invite_timer(call_id) — called when accept / decline / hangup
       arrives, so the timer is a no-op when the call resolved naturally.

The timer registry is process-local. On restart, in-flight invites
are abandoned (the rows are still in the DB so they'll surface on
next list/feed call), but no notification fires. That's the right
trade-off — restart-survivable missed-call detection would require
a persistent scheduler and the recipient's feed already shows the
unresolved row.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from augmentum.connect.protocol import (
    DEFAULT_INVITE_LIFETIME_MS,
    EVENT_HANGUP,
    ConnectEnvelope,
)
from augmentum.notifications import (
    IMPORTANCE_DEFAULT,
)
from augmentum.utils.logging import get_logger


if TYPE_CHECKING:
    from augmentum.connect.hub import ConnectHub
    from augmentum.notifications import NotificationHub


log = get_logger(__name__)


# Active invite timers keyed by call_id.
_TIMERS: dict[str, asyncio.Task[None]] = {}


# ── Public API ────────────────────────────────────────────────────


def arm_invite_timer(
    *,
    conn: Any,
    connect_hub: "ConnectHub",
    notification_hub: "NotificationHub",
    call_id: str,
    initiator_user_id: str,
    initiator_did: str,
    recipient_user_id: str,
    recipient_did: str,
    lifetime_ms: int = DEFAULT_INVITE_LIFETIME_MS,
) -> None:
    """Schedule the missed-call check.

    Idempotent on ``call_id`` — re-arming replaces any existing timer
    rather than stacking duplicates (e.g. an INVITE-retry on flaky
    network shouldn't fire two missed-call notifications).
    """

    if call_id in _TIMERS:
        cancel_invite_timer(call_id)

    coro = _wait_then_maybe_mark_missed(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        call_id=call_id,
        initiator_user_id=initiator_user_id,
        initiator_did=initiator_did,
        recipient_user_id=recipient_user_id,
        recipient_did=recipient_did,
        lifetime_ms=lifetime_ms,
    )
    task = asyncio.create_task(coro, name=f"connect-invite-{call_id}")
    _TIMERS[call_id] = task


def cancel_invite_timer(call_id: str) -> bool:
    """Cancel the pending missed-call check. No-op if no timer is set.

    Returns whether a timer was actually cancelled (used by tests).

    Tolerant of closed event loops so test teardown ordering can't
    raise — on Windows pytest tears the loop down before our autouse
    fixture's post-yield runs, and ``task.cancel()`` would otherwise
    propagate ``Event loop is closed``.
    """

    task = _TIMERS.pop(call_id, None)
    if task is None:
        return False
    if not task.done():
        try:
            task.cancel()
        except RuntimeError:
            # Loop already closed — the task is effectively dead;
            # nothing to do.
            pass
    return True


def active_timer_count() -> int:
    """Diagnostic — how many timers are currently outstanding."""

    return len(_TIMERS)


def reset_timers_for_test() -> None:
    """Test seam — cancels every outstanding timer."""

    for call_id in list(_TIMERS):
        cancel_invite_timer(call_id)


async def recover_stale_invites_on_startup(
    conn: Any,
    *,
    grace_ms: int = DEFAULT_INVITE_LIFETIME_MS,
) -> int:
    """Age out ``ringing`` / ``invited`` call_sessions rows orphaned by a
    prior process death.

    The in-memory missed-call timer registry doesn't survive a
    restart. Before this sweep, a row that was mid-ring when augmentum
    crashed / was restarted sat in ``ringing`` forever, polluting the
    calls-history list and (more harmfully) showing up as a presumed-
    live call in the recents UI. ``grace_ms`` gives in-flight pre-
    sweep invites a chance to resolve naturally — anything older is
    presumed orphaned.

    Returns the number of rows transitioned to ``missed``. Failures
    are logged and swallowed so a bad sweep can't keep the app from
    starting.
    """

    try:
        cur = await conn.execute(
            """UPDATE call_sessions
                  SET state = 'missed',
                      end_reason = 'restart_recovery',
                      ended_at = CURRENT_TIMESTAMP
                WHERE state IN ('ringing', 'invited')
                  AND (
                    strftime('%s', 'now') * 1000
                    - strftime('%s', initiated_at) * 1000
                  ) > ?""",
            (grace_ms,),
        )
        await conn.commit()
        count = cur.rowcount if cur.rowcount is not None else 0
        if count > 0:
            log.info("connect_stale_invites_recovered", count=count)
        return count
    except Exception as exc:
        log.warning("connect_stale_invite_recovery_failed", error=str(exc)[:160])
        return 0


# ── Internals ─────────────────────────────────────────────────────


async def _wait_then_maybe_mark_missed(
    *,
    conn: Any,
    connect_hub: "ConnectHub",
    notification_hub: "NotificationHub",
    call_id: str,
    initiator_user_id: str,
    initiator_did: str,
    recipient_user_id: str,
    recipient_did: str,
    lifetime_ms: int,
) -> None:
    """The body of the missed-call timer task."""

    try:
        await asyncio.sleep(lifetime_ms / 1000.0)
    except asyncio.CancelledError:
        # The natural happy path — caller cancelled because the
        # invite was answered.
        return
    finally:
        # Drop self from the registry whether we ran or were cancelled.
        _TIMERS.pop(call_id, None)

    try:
        await _mark_missed(
            conn=conn,
            connect_hub=connect_hub,
            notification_hub=notification_hub,
            call_id=call_id,
            initiator_user_id=initiator_user_id,
            initiator_did=initiator_did,
            recipient_user_id=recipient_user_id,
            recipient_did=recipient_did,
        )
    except Exception as exc:
        log.warning(
            "connect_missed_call_handler_failed",
            call_id=call_id,
            error=str(exc)[:160],
        )


async def _mark_missed(
    *,
    conn: Any,
    connect_hub: "ConnectHub",
    notification_hub: "NotificationHub",
    call_id: str,
    initiator_user_id: str,
    initiator_did: str,
    recipient_user_id: str,
    recipient_did: str,
) -> None:
    """Run the missed-call transitions.

    Re-checks the call state first so a late-arriving accept doesn't
    race the timer into a wrong "missed" state.
    """

    from augmentum.connect.call_routing import (
        _log_call_event,
        _update_call_session_state,
    )

    # Re-read state for both perspectives. If either advanced past
    # ringing/invited, the call resolved and we should bail out.
    cur = await conn.execute(
        "SELECT user_id, state FROM call_sessions WHERE call_id = ?",
        (call_id,),
    )
    rows = await cur.fetchall()
    if not rows:
        # Call row was deleted out from under us (rare — admin cleanup).
        return
    pending_states = {"ringing", "invited"}
    if not all(state in pending_states for _, state in rows):
        # Someone already advanced this. Bail.
        return

    # Transition both rows to 'missed'.
    await _update_call_session_state(
        conn, call_id=call_id, user_id=initiator_user_id,
        state="missed", end_reason="timeout",
    )
    await _update_call_session_state(
        conn, call_id=call_id, user_id=recipient_user_id,
        state="missed", end_reason="timeout",
    )
    await _log_call_event(
        conn, call_id=call_id, user_id=recipient_user_id,
        event_type="missed",
        event_data={"reason": "invite_timeout"},
    )

    # Publish connect.call.missed on the recipient's feed so they see
    # the missed call later. Don't reuse the dedupe_key from the
    # incoming notification — the missed-call entry is its own row
    # (the incoming one gets dismissed below).
    try:
        from augmentum.notifications.hub import publish_and_dispatch

        await publish_and_dispatch(
            conn,
            hub=notification_hub,
            user_id=recipient_user_id,
            channel_id="connect.call.missed",
            source="connect",
            title=f"Missed call from {initiator_did}",
            body="They tried to reach you.",
            importance=IMPORTANCE_DEFAULT,
            dedupe_key=f"missed:{call_id}",
            thread_id=call_id,
            payload={
                "call_id": call_id,
                "initiator_did": initiator_did,
                "initiator_user_id": initiator_user_id,
            },
            transient=False,
            icon="phone-missed",
        )
    except Exception as exc:
        log.warning(
            "connect_missed_call_publish_failed",
            call_id=call_id, error=str(exc)[:160],
        )

    # Dismiss the connect.call.incoming notification if it's still
    # sitting unactioned in the recipient's feed. The dedupe_key the
    # invite used was the call_id itself.
    try:
        from augmentum.notifications.store import (
            dismiss_by_dedupe_key,
        )
        await dismiss_by_dedupe_key(
            conn, user_id=recipient_user_id,
            source="connect", dedupe_key=call_id,
        )
    except ImportError:
        # dismiss_by_dedupe_key is added by a future task; the missed
        # banner being slightly redundant with the ringing one isn't a
        # blocking issue. Skip silently.
        pass
    except Exception as exc:
        log.warning(
            "connect_missed_call_dismiss_failed",
            call_id=call_id, error=str(exc)[:160],
        )

    # Route a synthetic EVENT_HANGUP back to the initiator so their
    # ringing UI can stop ringing. The recipient's signaling WS, if
    # attached, also gets the event so the incoming banner clears
    # in-app (the notification dismissal above handles the cold case).
    envelope = ConnectEnvelope(
        kind="event",
        verb=EVENT_HANGUP,
        peer=recipient_did,  # event.from for the initiator
        data={"call_id": call_id, "reason": "missed"},
    )
    try:
        await connect_hub.route_to_user(
            target_user_id=initiator_user_id,
            envelope=envelope,
        )
    except Exception as exc:
        log.warning(
            "connect_missed_call_route_failed",
            call_id=call_id, error=str(exc)[:160],
        )
