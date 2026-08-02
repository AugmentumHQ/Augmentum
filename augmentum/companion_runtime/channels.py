"""Channel state machine for multi-turn handoffs (Lane 3 §3).

When Becca emits ``<handoff:CHANNEL .../>``, the channel state machine
takes over: she steps aside, the channel surface mounts in the UI, the
user enters; when the user exits, she re-engages with one of the
return-microcopy menus (or stays silent, per the load-bearing rule
in Lane 3 §3.6).

Channels are:
  coder       — workspace, file edits, Plan/Act loop
  agentic     — multi-step plans with approval gates
  bug_finder  — eight-stage code-audit pipeline
  narrative   — fiction / RP (handled with extra isolation in Sprint E)

State machine:

       IDLE → ENTERING → ACTIVE ⇄ USER_IDLE → EXITING → EXITED → IDLE

Bus events:
  channel.entering        {channel, session_id, intent_id, surface_target}
  channel.active          {channel, session_id, mount_duration_ms}
  channel.user_idle       {channel, session_id, idle_ms}
  channel.exiting         {channel, session_id, exit_reason}
  channel.exited          {channel, session_id, duration_s, exchange_count, summary_id}

Memory contract (Lane 3 §3.4): on ``channel.exited``, write ONE Tier-1
companion_event capturing summary fields only — NOT contents. Narrative
mode in particular: NO scene content crosses the boundary.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from augmentum.companion_runtime import affordances
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Transition timeouts (Lane 3 §3.1)
USER_IDLE_AFTER_S = 180.0          # 3min silence → USER_IDLE
USER_IDLE_AUTO_EXIT_AFTER_S = 1800 # 30min in USER_IDLE → EXITING
MOUNT_TIMEOUT_MS = 800             # soft warning if mount exceeds


class ChannelState(str, Enum):
    IDLE = "idle"
    ENTERING = "entering"
    ACTIVE = "active"
    USER_IDLE = "user_idle"
    EXITING = "exiting"
    EXITED = "exited"


@dataclass
class ChannelSession:
    """Runtime-tracked state for one in-flight channel session.

    Held in ``CompanionRuntime._channel_sessions`` (a dict keyed by
    session_id). Removed on ``channel.exited``.
    """
    channel: str
    session_id: str
    user_id: str
    intent_id: str = ""
    started_at: float = field(default_factory=time.time)
    last_user_activity_at: float = field(default_factory=time.time)
    state: ChannelState = ChannelState.IDLE
    exchange_count: int = 0
    error_count: int = 0
    artifacts_created: int = 0
    user_tagged: list[str] = field(default_factory=list)
    reason: str = ""     # the "reason" arg from <handoff:.../>
    brief: str = ""      # the "brief" arg from <handoff:.../>


@dataclass(frozen=True, slots=True)
class ChannelSummary:
    """Summary written to memory tier 1 on exit. NO content fields."""
    channel: str
    session_id: str
    duration_s: float
    exchange_count: int
    started_at: float
    ended_at: float
    exit_reason: str
    broad_topic: str             # channel-supplied, ≤ 25 words, no content
    exit_affect: str | None      # filled by Lane 2 affect labeler later
    artifacts_created: int
    error_count: int
    user_tagged: list[str]


def _classify_exit(session: ChannelSession, exit_reason: str) -> str:
    """Decide which return-microcopy menu to draw from. See Lane 3 §3.5
    for the matrix.

    Returns one of: long_energized | long_frustrated | long_neutral |
    short_energized | short_frustrated | short_neutral | errored |
    task_completed | task_failed.
    """
    duration_s = time.time() - session.started_at
    long = duration_s > 600  # 10 min
    if session.error_count >= 3:
        return "errored"

    if session.channel in ("agentic", "bug_finder"):
        if exit_reason == "channel_complete":
            return "task_completed"
        if exit_reason in ("error", "channel_failed"):
            return "task_failed"
        return "long_neutral" if long else "short_neutral"

    # coder / narrative pattern
    # Sprint D: we don't have an exit_affect classifier yet (Lane 2
    # Sprint F territory), so map to neutral. Long-vs-short still helps.
    affect = "neutral"
    bucket = "long" if long else "short"
    return f"{bucket}_{affect}"


async def enter_channel(
    runtime: CompanionRuntime,
    *,
    channel: str,
    user_id: str,
    intent_id: str,
    reason: str = "",
    brief: str = "",
) -> ChannelSession:
    """Begin a channel session. Emits ``channel.entering`` and
    ``channel.active`` events. Stores the session on the runtime so
    ``exit_channel`` can find it.

    The UI consumes the ``channel.entering`` event to mount the
    channel surface; the ``channel.active`` event signals mount complete.

    Sprint D ships the events + session tracking; the actual channel
    surface mount is a UI subscriber that already handles legacy
    channel-mode mounts (coder workspace, narrative session, etc.).
    """
    session_id = f"ch_{uuid.uuid4().hex[:12]}"
    session = ChannelSession(
        channel=channel, session_id=session_id, user_id=user_id,
        intent_id=intent_id, reason=reason, brief=brief,
        state=ChannelState.ENTERING,
    )
    _sessions(runtime)[session_id] = session

    t_mount_start = time.monotonic()
    await runtime.bus.publish_topic(
        "channel.entering",
        {
            "channel": channel, "session_id": session_id,
            "intent_id": intent_id,
            "surface_target": _surface_for(channel),
            "reason": reason, "brief": brief,
            "user_id": user_id,
        },
        source_companion_id=runtime.companion_id,
    )

    # Mount-complete signal. UI subscribers should publish back a
    # ``channel.mount_complete`` event; for Sprint D we don't block on
    # it (the UI may or may not be wired yet). Just emit channel.active
    # after a tiny grace period.
    await asyncio.sleep(0)  # yield
    mount_ms = int((time.monotonic() - t_mount_start) * 1000)
    if mount_ms > MOUNT_TIMEOUT_MS:
        log.info("channel_mount_slow", channel=channel, ms=mount_ms)

    session.state = ChannelState.ACTIVE
    await runtime.bus.publish_topic(
        "channel.active",
        {"channel": channel, "session_id": session_id, "mount_duration_ms": mount_ms},
        source_companion_id=runtime.companion_id,
    )

    return session


async def exit_channel(
    runtime: CompanionRuntime,
    *,
    session_id: str,
    exit_reason: str = "user_explicit",
) -> ChannelSummary | None:
    """End a channel session. Writes the summary to memory tier 1
    (Lane 3 §3.4: summary fields only, no content). Emits
    ``channel.exiting`` then ``channel.exited``.

    Returns the summary so callers (the route handler) can choose
    re-engagement microcopy via ``return_microcopy``.
    """
    sessions = _sessions(runtime)
    session = sessions.pop(session_id, None)
    if session is None:
        log.warning("channel_exit_unknown_session", session_id=session_id)
        return None

    session.state = ChannelState.EXITING
    await runtime.bus.publish_topic(
        "channel.exiting",
        {"channel": session.channel, "session_id": session_id, "exit_reason": exit_reason},
        source_companion_id=runtime.companion_id,
    )

    ended_at = time.time()
    duration_s = ended_at - session.started_at

    # Broad topic: channel-supplied, sanitized for narrative (Lane 3 §4.2).
    broad_topic = await _broad_topic_for(session)

    summary = ChannelSummary(
        channel=session.channel,
        session_id=session_id,
        duration_s=duration_s,
        exchange_count=session.exchange_count,
        started_at=session.started_at,
        ended_at=ended_at,
        exit_reason=exit_reason,
        broad_topic=broad_topic,
        exit_affect=None,   # Sprint F: Lane 2 affect labeler fills this
        artifacts_created=session.artifacts_created,
        error_count=session.error_count,
        user_tagged=list(session.user_tagged),
    )

    # Persist summary to memory tier 1 — ONE event row, fields only.
    summary_text = (
        f"Channel session: {summary.channel}. Duration {int(summary.duration_s)}s. "
        f"{summary.exchange_count} exchanges. Topic: {summary.broad_topic}. "
        f"Exit: {summary.exit_reason}."
    )
    try:
        memory = getattr(runtime, "memory", None)
        if memory is not None and session.user_id:
            await memory.store_companion_event(
                summary_text,
                user_id=session.user_id,
                importance=0.4,
                source_context={
                    "kind": "channel_summary",
                    "channel": summary.channel,
                    "session_id": session_id,
                    "duration_s": round(summary.duration_s, 1),
                    "exchange_count": summary.exchange_count,
                    "exit_reason": summary.exit_reason,
                },
            )
    except Exception:
        log.exception("channel_summary_persist_failed", session_id=session_id)

    session.state = ChannelState.EXITED
    microcopy = return_microcopy_for(summary, session)
    await runtime.bus.publish_topic(
        "channel.exited",
        {
            "channel": summary.channel,
            "session_id": session_id,
            "duration_s": round(summary.duration_s, 1),
            "exchange_count": summary.exchange_count,
            "exit_reason": summary.exit_reason,
            "broad_topic": summary.broad_topic,
            "microcopy": microcopy,
        },
        source_companion_id=runtime.companion_id,
    )

    return summary


def return_microcopy_for(summary: ChannelSummary, session: ChannelSession | None = None) -> str:
    """Pick the re-engagement line based on the exit profile. Returns
    empty string for silent return (Lane 3 §3.6)."""
    if session is None:
        # Synthesize a minimal session for classification
        session = ChannelSession(
            channel=summary.channel,
            session_id=summary.session_id,
            user_id="",
            started_at=summary.started_at,
            exchange_count=summary.exchange_count,
            error_count=summary.error_count,
        )
    exit_class = _classify_exit(session, summary.exit_reason)
    return affordances.return_microcopy(summary.channel, exit_class)


# ── Internal helpers ─────────────────────────────────────────────────

def _sessions(runtime: CompanionRuntime) -> dict[str, ChannelSession]:
    """Lazy-init the per-runtime channel session map."""
    sessions = getattr(runtime, "_channel_sessions", None)
    if sessions is None:
        sessions = {}
        try:
            runtime._channel_sessions = sessions   # type: ignore[attr-defined]
        except Exception:
            log.warning("channels_session_attr_set_failed", exc_info=True)
    return sessions


def _surface_for(channel: str) -> str:
    """The UI surface id to mount for this channel."""
    return {
        "coder": "coder",
        "agentic": "agentic",
        "narrative": "narrative",
        "bug_finder": "bug_finder",
    }.get(channel, channel)


async def _broad_topic_for(session: ChannelSession) -> str:
    """Channel-supplied summary, ≤ 25 words. For narrative, returns a
    literal "a narrative session" string with NO content extraction —
    enforced by Lane 3 §4.2's isolation guarantee.

    Other channels (coder, agentic, bug_finder) can be enriched in
    Sprint D+ by reading the channel's own session table (coder
    workspace name, agentic task statement, etc.). For Sprint D ship
    the conservative fallback for all channels — exit summaries are
    additive; thicker topics can land later without breaking schema.
    """
    if session.channel == "narrative":
        return "a narrative session"
    if session.reason:
        return session.reason[:180]
    return f"a {session.channel} session"


__all__ = [
    "ChannelState",
    "ChannelSession",
    "ChannelSummary",
    "enter_channel",
    "exit_channel",
    "return_microcopy_for",
]
