"""Notification fuser — the difference between an echo machine and awareness.

The phone can hand the companion 40 notifications. Reading them back is the echo
machine (Bee's failure mode: raw capture, no synthesis). This fuser does the
opposite: it correlates the stream into the *few* things that actually matter,
each with its evidence chain, and emits NOTHING when nothing correlates.

The rules here are deliberately conservative — one notification is never an
insight (that's an echo); it takes a *pattern* (repetition from one person, a
genuinely time-critical category) to clear the bar. The judgment gate downstream
then decides silent / pull / speak, and the regret loop damps any shape the user
keeps dismissing. So this fuser's only job is: find the real signal, attach the
why, and set how-much-it-matters honestly.

Correlations implemented:
  * **person pressure** — ≥2 unread messages from the SAME person inside the
    window collapse into one "X has been trying to reach you (N messages)"
    insight. Value scales with count; confidence with how concentrated it is.
    NOT time-critical — it's pull-worthy, not interrupt-worthy (the gate will
    file it for the Today digest unless the user's already in conversation).
  * **missed calls** — repeated ``call`` notifications from one person → a
    time-critical "X tried to call you (N times)" insight (a person calling
    twice is reaching for you now).

Reads only ``ctx.signals['notifications']`` (a list of
``NotificationObservation``) — the live pass loads it; the fuser stays pure and
fully unit-testable with synthetic lists.
"""

from __future__ import annotations

from collections import defaultdict

from augmentum.companion_runtime.perception.acquisition.notifications import (
    NotificationObservation,
)
from augmentum.companion_runtime.perception.fusion import FusionContext
from augmentum.companion_runtime.perception.insight import Insight

NOTIFICATION_FUSER_NAME = "notifications"

# A person is "pressing" once they've sent at least this many unread messages in
# the window — below it, a single ping is just a ping (echo territory).
_PRESSURE_MIN = 2
# Repeated calls from one person within the window → they're reaching for you now.
_CALL_MIN = 2
# How recent a notification must be to count toward a live pattern (seconds).
# Past this, it's history, not pressure — the gate would only file it anyway.
_FRESH_WINDOW_S = 6 * 3600.0

_CALL_CATEGORIES = frozenset({"call", "missed_call"})


def _person_key(obs: NotificationObservation) -> str:
    """Best-effort identity for grouping — the named person, else the title
    (chat apps put the sender in the title), else the app (a generic bucket)."""
    return (obs.person or obs.title or obs.source_app or obs.source_pkg).strip().lower()


def _display_person(obs: NotificationObservation) -> str:
    return (obs.person or obs.title or obs.source_app or "Someone").strip() or "Someone"


def fuse_notifications(ctx: FusionContext) -> list[Insight]:
    """Correlate the recent notification stream into insights. Pure."""
    raw = ctx.signals.get("notifications") if ctx.signals else None
    if not raw:
        return []

    fresh: list[NotificationObservation] = []
    cutoff = ctx.now - _FRESH_WINDOW_S
    for obs in raw:
        if not isinstance(obs, NotificationObservation):
            continue
        # posted_at==0 means the client didn't stamp it; treat as fresh rather
        # than dropping a real signal on a missing timestamp.
        if obs.posted_at and obs.posted_at < cutoff:
            continue
        fresh.append(obs)
    if not fresh:
        return []

    insights: list[Insight] = []
    insights.extend(_fuse_person_pressure(fresh))
    insights.extend(_fuse_missed_calls(fresh))
    return insights


def _fuse_person_pressure(fresh: list[NotificationObservation]) -> list[Insight]:
    by_person: dict[str, list[NotificationObservation]] = defaultdict(list)
    for obs in fresh:
        if not obs.is_message:
            continue
        by_person[_person_key(obs)].append(obs)

    out: list[Insight] = []
    for _key, msgs in by_person.items():
        if len(msgs) < _PRESSURE_MIN:
            continue
        person = _display_person(msgs[0])
        n = len(msgs)
        app = (msgs[0].source_app or "").strip()
        via = f" on {app}" if app else ""
        # Value rises with the count but saturates — 2 messages already matters,
        # 8 doesn't matter 4× as much. Confidence rises with concentration.
        value = min(0.45 + 0.10 * (n - _PRESSURE_MIN), 0.85)
        confidence = min(0.55 + 0.08 * (n - _PRESSURE_MIN), 0.9)
        evidence = [
            f"{n} unread message{'s' if n != 1 else ''} from {person}{via}",
        ]
        # Quote the most recent line as the "why" — concrete beats abstract.
        last_body = (msgs[0].body or "").strip()
        if last_body:
            evidence.append(f"latest: “{last_body[:120]}”")
        out.append(Insight(
            kind="social.pressure",
            summary=f"{person} has been trying to reach you "
                    f"— {n} unread message{'s' if n != 1 else ''}{via}.",
            shape="social.pressure",
            evidence=evidence,
            value=value,
            confidence=confidence,
            time_critical=False,  # pull-worthy, not an interruption
        ))
    return out


def _fuse_missed_calls(fresh: list[NotificationObservation]) -> list[Insight]:
    by_person: dict[str, list[NotificationObservation]] = defaultdict(list)
    for obs in fresh:
        if obs.category not in _CALL_CATEGORIES:
            continue
        by_person[_person_key(obs)].append(obs)

    out: list[Insight] = []
    for _key, calls in by_person.items():
        if len(calls) < _CALL_MIN:
            continue
        person = _display_person(calls[0])
        n = len(calls)
        value = min(0.6 + 0.1 * (n - _CALL_MIN), 0.9)
        out.append(Insight(
            kind="comms.missed_call",
            summary=f"{person} tried to call you {n} times.",
            shape="comms.missed_call",
            evidence=[f"{n} call notifications from {person} in the last few hours"],
            value=value,
            confidence=0.85,
            time_critical=True,  # someone calling twice is reaching for you now
        ))
    return out
