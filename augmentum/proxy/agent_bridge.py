"""Harness agent bridge — terminal agents reach the user through Augmentum.

An external coding agent (Claude Code, pi, cursor... anything on the /v1
proxy) registers presence with ``agent_checkin``, asks the user things with
``ask_user`` (permission approvals, questions, end-of-run review offers),
and polls ``check_reply`` for the answer. The user sees an Augmentum
notification on whatever device they're on — Approve/Deny are one tap;
questions and reviews take a free-text reply — so managing a terminal
agent no longer requires being at the keyboard.

Reuses, not reinvents: notifications ride ``publish_and_dispatch`` (WS +
web push, action buttons), button clicks come back through the existing
``notifications/actions.py`` registry, and state lives in two migrated
user-scoped tables (migration 316). ATP tools live in
``augmentum/tools/agent_bridge_tools.py``; HTTP routes for the UI side in
``harness_routes.py``.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

CHANNEL = "harness.agent.request"

AGENT_STATUSES = ("working", "waiting", "done")
REQUEST_KINDS = ("approve", "question", "review", "notify", "assignment")

# Sessions with a heartbeat in this window count as "active".
ACTIVE_WINDOW_MINUTES = 30


def _conn(app_state: Any):
    sm = getattr(app_state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm is not None else None
    return getattr(backend, "conn", None)


# ── agent sessions ─────────────────────────────────────────────────────

async def checkin(
    app_state: Any, *, user_id: str, harness: str, project: str,
    agent_id: str = "", title: str = "", status: str = "working",
    summary: str = "",
) -> dict[str, Any] | None:
    """Upsert an agent session heartbeat. Returns the session dict (with
    server-minted id when new) plus any freshly answered requests for it,
    so a check-in doubles as a reply pickup."""
    conn = _conn(app_state)
    if conn is None or not user_id:
        return None
    if status not in AGENT_STATUSES:
        status = "working"
    if agent_id:
        cur = await conn.execute(
            "SELECT id FROM harness_agent_sessions WHERE id = ? AND user_id = ?",
            [agent_id, user_id],
        )
        if await cur.fetchone() is None:
            agent_id = ""  # unknown/foreign id → mint fresh
    if not agent_id:
        agent_id = f"ag_{uuid.uuid4().hex[:12]}"
        await conn.execute(
            "INSERT INTO harness_agent_sessions "
            "(id, user_id, harness, project, title, status, summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [agent_id, user_id, harness, project, title, status, summary],
        )
    else:
        await conn.execute(
            "UPDATE harness_agent_sessions SET title = COALESCE(NULLIF(?, ''), title), "
            "status = ?, summary = COALESCE(NULLIF(?, ''), summary), "
            "last_seen = datetime('now') WHERE id = ? AND user_id = ?",
            [title, status, summary, agent_id, user_id],
        )
    await conn.commit()

    # Piggyback: any answered-but-unfetched requests for this agent.
    cur = await conn.execute(
        "SELECT id, kind, title, reply_action, reply_text, answered_at "
        "FROM harness_agent_requests WHERE user_id = ? AND agent_session_id = ? "
        "AND status = 'answered' ORDER BY answered_at DESC LIMIT 10",
        [user_id, agent_id],
    )
    answered = [
        {"request_id": r[0], "kind": r[1], "title": r[2],
         "reply_action": r[3], "reply_text": r[4], "answered_at": r[5]}
        for r in await cur.fetchall()
    ]

    # If the agent reports a terminal status, finalize any of ITS assignment
    # runs still marked 'working' (the row advances queued → working → done in
    # the Agents history). Best-effort — a status update must never fail check-in.
    if status in ("done", "failed"):
        with contextlib.suppress(Exception):
            fin = await conn.execute(
                "SELECT linked_run_id FROM harness_agent_requests "
                "WHERE user_id = ? AND agent_session_id = ? AND kind = 'assignment' "
                "AND status = 'delivered' AND linked_run_id != ''",
                [user_id, agent_id],
            )
            run_ids = [r[0] for r in await fin.fetchall()]
            if run_ids:
                from augmentum.coder import coding_driver
                for rid in run_ids:
                    await coding_driver.update_run(
                        app_state, user_id=user_id, run_id=rid,
                        status=status, summary=summary)
                ph = ",".join("?" for _ in run_ids)
                await conn.execute(
                    "UPDATE harness_agent_requests SET status = 'done' "
                    f"WHERE user_id = ? AND agent_session_id = ? AND linked_run_id IN ({ph})",
                    [user_id, agent_id, *run_ids])
                await conn.commit()

    # Assignments: tasks queued FOR this agent (the reverse channel — the
    # user/companion handed it work). Deliver once, then mark 'delivered' so
    # the same task isn't picked up on every heartbeat.
    acur = await conn.execute(
        "SELECT id, title, body, linked_run_id FROM harness_agent_requests "
        "WHERE user_id = ? AND agent_session_id = ? AND kind = 'assignment' "
        "AND status = 'pending' ORDER BY created_at ASC LIMIT 5",
        [user_id, agent_id],
    )
    assignments = [
        {"request_id": r[0], "title": r[1], "task": r[2], "run_id": r[3]}
        for r in await acur.fetchall()
    ]
    if assignments:
        placeholders = ",".join("?" for _ in assignments)
        await conn.execute(
            "UPDATE harness_agent_requests SET status = 'delivered', "
            f"answered_at = datetime('now') WHERE id IN ({placeholders})",
            [a["request_id"] for a in assignments],
        )
        await conn.commit()
        # Delivery to a LIVE session = the work is now in flight; advance each
        # linked status row queued → working so the history reflects reality.
        with contextlib.suppress(Exception):
            from augmentum.coder import coding_driver
            for a in assignments:
                if a.get("run_id"):
                    await coding_driver.update_run(
                        app_state, user_id=user_id, run_id=a["run_id"], status="working")

    return {"agent_id": agent_id, "status": status,
            "answered_requests": answered, "assignments": assignments}


async def create_assignment(
    app_state: Any, *, user_id: str, agent_session_id: str, task: str,
    title: str = "", harness: str = "", project: str = "", linked_run_id: str = "",
) -> dict[str, Any] | None:
    """Queue a task FOR an agent (the reverse channel). The agent picks it up
    at its next check-in. Returns the created request id.

    ``linked_run_id`` ties this assignment to the coding_runs status row shown
    in the Agents history, so check-in can advance it (queued → working → done)
    as the agent on the user's machine reports back."""
    conn = _conn(app_state)
    if conn is None or not user_id or not agent_session_id or not (task or "").strip():
        return None
    # The target agent must belong to this user.
    cur = await conn.execute(
        "SELECT id FROM harness_agent_sessions WHERE id = ? AND user_id = ?",
        [agent_session_id, user_id],
    )
    if await cur.fetchone() is None:
        return None
    request_id = f"agrq_{uuid.uuid4().hex[:12]}"
    await conn.execute(
        "INSERT INTO harness_agent_requests "
        "(id, user_id, agent_session_id, kind, title, body, linked_run_id) "
        "VALUES (?, ?, ?, 'assignment', ?, ?, ?)",
        [request_id, user_id, agent_session_id,
         title or (task[:60] + ("…" if len(task) > 60 else "")), task,
         linked_run_id or ""],
    )
    await conn.commit()
    # Best-effort nudge so the user sees the hand-off land on their devices.
    with __import__("contextlib").suppress(Exception):
        await _notify(
            app_state, conn, user_id=user_id, kind="notify",
            title=f"Task sent to agent: {title or task[:48]}", body="",
            request_id="", harness=harness, project=project,
        )
    return {"request_id": request_id}


async def list_agents(app_state: Any, *, user_id: str) -> list[dict[str, Any]]:
    """Active agent sessions (recent heartbeat) with pending-request counts."""
    conn = _conn(app_state)
    if conn is None or not user_id:
        return []
    cur = await conn.execute(
        "SELECT s.id, s.harness, s.project, s.title, s.status, s.summary, "
        "s.created_at, s.last_seen, "
        "(SELECT COUNT(*) FROM harness_agent_requests r "
        " WHERE r.agent_session_id = s.id AND r.status = 'pending') "
        "FROM harness_agent_sessions s WHERE s.user_id = ? "
        f"AND s.last_seen >= datetime('now', '-{ACTIVE_WINDOW_MINUTES} minutes') "
        "ORDER BY s.last_seen DESC LIMIT 50",
        [user_id],
    )
    return [
        {"agent_id": r[0], "harness": r[1], "project": r[2], "title": r[3],
         "status": r[4], "summary": r[5], "created_at": r[6],
         "last_seen": r[7], "pending_requests": r[8]}
        for r in await cur.fetchall()
    ]


# ── requests (ask → notify → reply → poll) ─────────────────────────────

async def create_request(
    app_state: Any, *, user_id: str, agent_session_id: str, kind: str,
    title: str, body: str = "", harness: str = "", project: str = "",
) -> dict[str, Any] | None:
    """Create a request row + fire the notification. ``kind='notify'`` is
    fire-and-forget (no row, no reply expected)."""
    conn = _conn(app_state)
    if conn is None or not user_id or not title:
        return None
    if kind not in REQUEST_KINDS:
        kind = "question"

    request_id = ""
    if kind != "notify":
        request_id = f"agrq_{uuid.uuid4().hex[:12]}"
        await conn.execute(
            "INSERT INTO harness_agent_requests "
            "(id, user_id, agent_session_id, kind, title, body) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [request_id, user_id, agent_session_id, kind, title, body],
        )
        await conn.commit()

    notification_id = await _notify(
        app_state, conn, user_id=user_id, kind=kind, title=title, body=body,
        request_id=request_id, harness=harness, project=project,
    )
    if request_id and notification_id:
        await conn.execute(
            "UPDATE harness_agent_requests SET notification_id = ? WHERE id = ?",
            [notification_id, request_id],
        )
        await conn.commit()
    return {"request_id": request_id, "notification_id": notification_id or ""}


async def _notify(
    app_state: Any, conn, *, user_id: str, kind: str, title: str, body: str,
    request_id: str, harness: str, project: str,
) -> str | None:
    """Publish through the standard hub (WS + web push). Best-effort —
    the request row is the source of truth; the UI panel also polls."""
    try:
        from augmentum.notifications.hub import NotificationHub, publish_and_dispatch
        from augmentum.notifications.store import NotificationAction

        hub = getattr(app_state, "notification_hub", None)
        if hub is None:
            hub = NotificationHub()
            app_state.notification_hub = hub

        actions: list[NotificationAction] = []
        if kind == "approve":
            actions = [
                NotificationAction(id="approve", label="Approve", style="primary"),
                NotificationAction(id="deny", label="Deny", style="danger"),
            ]
        elif kind in ("question", "review"):
            actions = [NotificationAction(id="reply", label="Reply…", style="primary")]

        origin = " · ".join(x for x in (harness, project) if x)
        return await publish_and_dispatch(
            conn,
            hub=hub,
            user_id=user_id,
            channel_id=CHANNEL,
            source="harness.agent",
            title=title if not origin else f"[{origin}] {title}",
            body=body,
            importance=3,
            actions=actions or None,
            payload={"request_id": request_id, "kind": kind},
        )
    except Exception:
        log.warning("agent_bridge_notify_failed", user_id=user_id, exc_info=True)
        return None


async def get_request(
    app_state: Any, *, user_id: str, request_id: str,
) -> dict[str, Any] | None:
    conn = _conn(app_state)
    if conn is None or not user_id or not request_id:
        return None
    cur = await conn.execute(
        "SELECT id, agent_session_id, kind, title, body, status, "
        "reply_action, reply_text, created_at, answered_at "
        "FROM harness_agent_requests WHERE id = ? AND user_id = ?",
        [request_id, user_id],
    )
    r = await cur.fetchone()
    if r is None:
        return None
    return {
        "request_id": r[0], "agent_session_id": r[1], "kind": r[2],
        "title": r[3], "body": r[4], "status": r[5],
        "reply_action": r[6], "reply_text": r[7],
        "created_at": r[8], "answered_at": r[9],
    }


async def answer_request(
    app_state: Any, *, user_id: str, request_id: str,
    action: str = "", text: str = "",
) -> dict[str, Any]:
    """Record the user's answer (button action and/or free text)."""
    conn = _conn(app_state)
    if conn is None or not user_id or not request_id:
        return {"ok": False, "error": "bad request"}
    if not action and not (text or "").strip():
        return {"ok": False, "error": "action or text required"}
    await conn.execute(
        "UPDATE harness_agent_requests SET status = 'answered', "
        "reply_action = ?, reply_text = ?, answered_at = datetime('now') "
        "WHERE id = ? AND user_id = ? AND status = 'pending'",
        [action, (text or "").strip(), request_id, user_id],
    )
    await conn.commit()
    # Verify by read — cursor.rowcount proved unreliable on this
    # connection (reported 0 for an UPDATE that demonstrably landed).
    cur = await conn.execute(
        "SELECT status, reply_action, reply_text FROM harness_agent_requests "
        "WHERE id = ? AND user_id = ?",
        [request_id, user_id],
    )
    row = await cur.fetchone()
    if row is None:
        return {"ok": False, "error": "request not found"}
    if row[0] != "answered":
        return {"ok": False, "error": "request not answerable (already handled?)"}
    if (row[1] or "") != action or (row[2] or "") != (text or "").strip():
        return {"ok": False, "error": "request was already answered earlier"}
    log.info("agent_bridge_request_answered", request_id=request_id,
             action=action or "text")
    return {"ok": True, "request_id": request_id, "action": action}


# ── notification action-button handler ─────────────────────────────────

async def _handle_action(notification, action_id: str, request) -> dict[str, Any]:
    """Approve/Deny straight from the notification button. ``reply``
    returns a hint the UI turns into a text prompt."""
    payload = getattr(notification, "payload", None) or {}
    request_id = str(payload.get("request_id") or "")
    user = request.scope.get("user")
    user_id = getattr(user, "id", "") if user else ""
    if not request_id:
        return {"ok": False, "error": "no request_id on notification"}
    if action_id in ("approve", "deny"):
        return await answer_request(
            request.app.state, user_id=user_id, request_id=request_id,
            action=action_id,
        )
    if action_id == "reply":
        # The UI prompts for text and POSTs /api/harness/agent/reply.
        return {"ok": True, "client_reply_prompt": True, "request_id": request_id}
    return {"ok": False, "error": f"unknown action {action_id!r}"}


def register_bridge_action_handler() -> None:
    try:
        from augmentum.notifications.actions import register_action_handler
        register_action_handler("harness.agent.*", _handle_action)
    except Exception:  # pragma: no cover — notifications optional in tests
        log.debug("agent_bridge_action_handler_skipped", exc_info=True)
