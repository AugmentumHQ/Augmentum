"""time.set_timer — architect-callable countdown timer.

User: "set a 10-minute timer", "remind me in an hour to take a break",
"start a 5 minute timer". Imperative-only — no question forms.

Implementation:
  1. Parse the duration slot into seconds (handles "10 minutes",
     "an hour", "30 seconds", "1h 30m", etc.)
  2. Schedule an asyncio task that fires after the duration
  3. On fire: run the optional END-ACTION (``then_verb`` — "in 20
     minutes pause the music"; gated to DEFERRED_ACTION_STAKES at set
     time AND fire time), then DELIVER via the notifications pipeline
     (time.timer channel — banner + sound live, Web Push offline),
     then publish ``surface.companion.timer_fired`` for perception.
  4. Speak immediate confirmation

In-memory: a server restart loses pending timers — acceptable for
short countdowns; anchored "at 5pm" asks belong to the
restart-survivable ``verb_fire`` standing-task kind instead.
"""

from __future__ import annotations

import asyncio
import re
import time as _time
from typing import Any

from augmentum.intent.action import (
    ActionResult,
    SessionContext,
)
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# In-memory active-timer registry. Keyed by (user_id, timer_id).
# Stores the asyncio task + metadata so we can introspect / cancel.
# A future "cancel my timer" primitive reads from this.
_ACTIVE_TIMERS: dict[tuple[str, str], dict[str, Any]] = {}


# Duration parsing — "10 minutes" → 600. Handles natural variations:
#   "10 minutes", "10 min", "10m"
#   "an hour" / "one hour" / "1h"
#   "half an hour" → 1800
#   "30 seconds" / "30s"
#   "1 hour 30 minutes" / "1h30m"
# Returns total seconds or None when unparseable.
_NUMBER_WORDS = {
    "an": 1, "a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "ninety": 90,
}

_UNIT_TO_SECONDS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
}


def _parse_duration(text: str) -> int | None:
    """Parse a duration phrase into total seconds.

    Returns None when the parse fails so the caller can surface a
    clarifying response ("How long?") rather than dispatching a
    timer with bogus duration.
    """
    if not text:
        return None
    s = text.strip().lower()
    # "half an hour" / "half hour"
    if "half" in s and ("hour" in s or "hr" in s):
        return 1800
    if "quarter" in s and ("hour" in s or "hr" in s):
        return 900

    # Generic number+unit scan. Supports compound forms like
    # "1 hour 30 minutes" by summing each (number, unit) pair.
    total = 0
    matched_any = False
    pattern = re.compile(
        r"(?P<num>\d+|" + "|".join(_NUMBER_WORDS.keys()) + r")"
        r"\s*(?P<unit>" + "|".join(_UNIT_TO_SECONDS.keys()) + r")\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(s):
        num_text = match.group("num").lower()
        unit = match.group("unit").lower()
        if num_text.isdigit():
            num = int(num_text)
        else:
            num = _NUMBER_WORDS.get(num_text, 0)
        if num <= 0:
            continue
        secs_per_unit = _UNIT_TO_SECONDS.get(unit, 0)
        if secs_per_unit <= 0:
            continue
        total += num * secs_per_unit
        matched_any = True

    if not matched_any or total <= 0:
        return None
    # Hard upper bound — refuse "set a 100 year timer" pranks.
    if total > 24 * 3600:
        return None
    return total


def _format_duration(seconds: int) -> str:
    """Render seconds to a human label for the spoken ack."""
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    if seconds < 3600:
        m = seconds // 60
        s = seconds % 60
        if s:
            return f"{m} minute{'s' if m != 1 else ''} {s} second{'s' if s != 1 else ''}"
        return f"{m} minute{'s' if m != 1 else ''}"
    h = seconds // 3600
    rem = seconds % 3600
    m = rem // 60
    if m:
        return f"{h} hour{'s' if h != 1 else ''} {m} minute{'s' if m != 1 else ''}"
    return f"{h} hour{'s' if h != 1 else ''}"


async def _run_then_action(
    runtime: Any, *, user_id: str, then_verb: str, then_args: dict[str, str],
) -> str:
    """Execute the timer's end-action headlessly through the registry
    pipeline. Returns a short result line for the notification body.

    The stakes gate was already enforced at SET time, but re-check here
    — the registry could have changed between set and fire.
    """
    from augmentum.intent.registry import DEFERRED_ACTION_STAKES, REGISTRY
    action = REGISTRY.get(then_verb)
    if action is None or getattr(action, "stakes", "") not in DEFERRED_ACTION_STAKES:
        return f"(couldn't run {then_verb} — not available)"
    from augmentum.companion_runtime import tools as tool_bridge
    from augmentum.companion_runtime.tool_protocol import ToolCall
    call = ToolCall(
        kind="tool", name=then_verb, args=dict(then_args),
        raw="timer_then_action", span=(0, 0),
    )
    # session_id stays empty: marks the surface_emit bus publish as a
    # HEADLESS fire so the widget's bus forwarder dispatches it.
    result = await tool_bridge.execute_tool(call, runtime, user_id=user_id)
    if result.ok:
        content = ""
        if isinstance(result.payload, dict):
            content = str(result.payload.get("content") or "").strip()
        return content or f"{then_verb} done."
    reason = result.error.message if result.error else "unknown error"
    return f"(couldn't run {then_verb}: {reason[:120]})"


async def _deliver_fire(
    runtime: Any, *, user_id: str, label: str, duration_s: int,
    then_line: str = "",
) -> None:
    """Make the fired timer reach the user: notification row + live WS
    banner + Web Push when offline. This is the delivery the original
    slice lacked — the bus event alone has no consumer, so timers
    confirmed but never rang (found 2026-06-11)."""
    app_state = getattr(runtime, "_app_state", None) if runtime else None
    conn = getattr(getattr(runtime, "backend", None), "conn", None)
    if conn is None:
        log.warning("timer_fire_no_conn", user_id=user_id)
        return
    hub = getattr(app_state, "notification_hub", None) if app_state else None
    if hub is None:
        from augmentum.notifications.hub import NotificationHub
        hub = NotificationHub()
        if app_state is not None:
            app_state.notification_hub = hub
    from augmentum.notifications.hub import publish_and_dispatch
    title = label.strip().capitalize() if label.strip() else "Timer done"
    body = f"Your {_format_duration(duration_s)} timer is up."
    if then_line:
        body = f"{body}\n{then_line}"
    await publish_and_dispatch(
        conn,
        hub=hub,
        user_id=user_id,
        channel_id="time.timer",
        source="time.set_timer",
        title=title,
        body=body,
        importance=None,
    )


async def _fire_timer(
    user_id: str,
    timer_id: str,
    duration_s: int,
    label: str,
    runtime: Any,
    then_verb: str = "",
    then_args: dict[str, str] | None = None,
) -> None:
    """Sleep for the duration, then run the end-action (if any) and
    deliver the fire to the user.

    Cancellation: the task is stored in _ACTIVE_TIMERS; a future
    "cancel my timer" primitive can lift it out and call cancel().
    On normal expiry, we remove ourselves from the registry before
    publishing so any inspector sees the timer as gone.
    """
    try:
        await asyncio.sleep(duration_s)
    except asyncio.CancelledError:
        _ACTIVE_TIMERS.pop((user_id, timer_id), None)
        log.info("timer_cancelled", user_id=user_id, timer_id=timer_id)
        raise

    _ACTIVE_TIMERS.pop((user_id, timer_id), None)

    # End-action first — "in 20 minutes pause the music" should pause
    # the music AT the fire, then tell the user it happened.
    then_line = ""
    if then_verb and runtime is not None:
        try:
            then_line = await _run_then_action(
                runtime, user_id=user_id,
                then_verb=then_verb, then_args=then_args or {},
            )
        except Exception:  # noqa: BLE001 — a broken action still rings
            log.warning("timer_then_action_failed", verb=then_verb, exc_info=True)
            then_line = f"(couldn't run {then_verb})"

    try:
        await _deliver_fire(
            runtime, user_id=user_id, label=label,
            duration_s=duration_s, then_line=then_line,
        )
    except Exception:  # noqa: BLE001 — delivery failure shouldn't kill the bus event
        log.warning("timer_fire_delivery_failed", user_id=user_id, exc_info=True)

    bus = getattr(runtime, "bus", None) if runtime else None
    if bus is None:
        log.info(
            "timer_fired_no_bus",
            user_id=user_id, timer_id=timer_id,
            label=label[:80],
        )
        return

    try:
        await bus.publish_topic(
            "surface.companion.timer_fired",
            {
                "user_id": user_id,
                "timer_id": timer_id,
                "duration_s": duration_s,
                "label": label[:160] if label else "",
                "then_verb": then_verb,
            },
        )
        log.info(
            "timer_fired",
            user_id=user_id, timer_id=timer_id,
            duration_s=duration_s, label=label[:80],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("timer_fire_publish_failed", error=str(exc)[:200])


async def _set_timer_handler(
    text: str,
    session: SessionContext,
    args: dict[str, Any],
) -> ActionResult | None:
    """Architect handler — parses duration, schedules the task,
    speaks confirmation.
    """
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I can't set a timer for a signed-out session.",
        )

    duration_text = (args.get("duration") or "").strip()
    label = (args.get("message") or args.get("label") or "").strip()

    duration_s = _parse_duration(duration_text)
    # Fallback: some Tier 1 regex patterns ("set a 10 minute timer")
    # match without a capture group, leaving duration_text empty. The
    # full transcript still contains the duration, so re-scan it. Only
    # applies when the slot didn't already provide a valid duration.
    if not duration_s:
        duration_s = _parse_duration(text)
    if not duration_s:
        return ActionResult(
            short_circuit=True,
            speak=(
                "How long should the timer run? Try something like "
                "'set a 10 minute timer' or 'remind me in an hour'."
            ),
        )

    runtime = getattr(session.app_state, "companion_runtime", None) \
        if getattr(session, "app_state", None) else None

    # End-action ("in 20 minutes pause the music"): a registry verb to
    # dispatch at fire time. Gated to trivial_reversible at SET time so
    # the refusal is conversational, not a silent fizzle later. Args
    # arrive as a JSON object string from native tool calls.
    then_verb = (args.get("then_verb") or "").strip().lower()
    then_args: dict[str, str] = {}
    raw_then_args = args.get("then_args")
    if isinstance(raw_then_args, str) and raw_then_args.strip():
        try:
            import json as _json
            parsed = _json.loads(raw_then_args)
            if isinstance(parsed, dict):
                then_args = {str(k): str(v) for k, v in parsed.items()}
        except Exception:  # noqa: BLE001 — bad args = action without args
            log.warning("timer_then_args_unparseable", raw=raw_then_args[:120])
    elif isinstance(raw_then_args, dict):
        then_args = {str(k): str(v) for k, v in raw_then_args.items()}

    if then_verb:
        from augmentum.intent.registry import DEFERRED_ACTION_STAKES, REGISTRY
        then_action = REGISTRY.get(then_verb)
        if then_action is None:
            return ActionResult(
                short_circuit=True,
                speak=(
                    f"I can set the timer, but I don't have an action "
                    f"called {then_verb} to run when it ends."
                ),
            )
        if getattr(then_action, "stakes", "") not in DEFERRED_ACTION_STAKES:
            return ActionResult(
                short_circuit=True,
                speak=(
                    f"I can't schedule {then_verb} to run unattended — "
                    "that one needs you present."
                ),
            )

    # Schedule on the running loop. asyncio.create_task ties the
    # task to the loop, not to this handler's call frame, so it
    # outlives the request that created it.
    timer_id = f"t_{int(_time.time() * 1000)}_{session.user_id[-6:]}"
    task = asyncio.create_task(
        _fire_timer(
            session.user_id, timer_id, duration_s, label, runtime,
            then_verb=then_verb, then_args=then_args,
        ),
        name=f"timer-{timer_id}",
    )
    _ACTIVE_TIMERS[(session.user_id, timer_id)] = {
        "task": task,
        "duration_s": duration_s,
        "label": label,
        "then_verb": then_verb,
        "started_at": _time.time(),
    }

    pretty = _format_duration(duration_s)
    if then_verb:
        speak = f"Done — in {pretty} I'll run {then_verb.replace('.', ' ')}."
        if label:
            speak = f"Done — in {pretty}: {label}."
    elif label:
        speak = f"Timer set for {pretty} — I'll remind you to {label}."
    else:
        speak = f"Timer set for {pretty}."

    log.info(
        "timer_set",
        user_id=session.user_id, timer_id=timer_id,
        duration_s=duration_s, label=label[:80],
    )

    return ActionResult(
        short_circuit=True,
        speak=speak,
        surface_emit={
            "channel": "timer.set",
            "payload": {
                "timer_id": timer_id,
                "duration_s": duration_s,
                "label": label,
                "ends_at": _time.time() + duration_s,
            },
        },
    )


register_action(
    id="time.set_timer",
    summary=(
        "Set a countdown timer. Notifies the user when it expires. "
        "Duration is parsed from natural phrasing (10 minutes, an "
        "hour, 30 seconds). Optionally runs an action when it ends: "
        "pass then_verb (a tool id like media.pause or weather.today) "
        "and then_args for asks like 'in 20 minutes pause the music' "
        "or 'check the weather in an hour and tell me'."
    ),
    examples=[
        "set a 10 minute timer",
        "set a timer for 5 minutes",
        "start a 30 second timer",
        "remind me in an hour",
        "remind me in 20 minutes to take a break",
        "set a half hour timer",
    ],
    handler=_set_timer_handler,
    delivery="artifact",
    arg_schema={
        "duration": {
            "type": "string",
            "description": "How long the timer should run, in natural phrasing.",
        },
        "label": {
            "type": "string",
            "description": "Optional reminder label.",
        },
        "then_verb": {
            "type": "string",
            "description": (
                "Optional tool id to run when the timer fires (e.g. "
                "media.pause, weather.today). Only small reversible "
                "actions are allowed."
            ),
        },
        "then_args": {
            "type": "string",
            "description": (
                "JSON object of arguments for then_verb, e.g. "
                "'{\"location\": \"portland\"}'."
            ),
        },
    },
    required=["duration"],
    surfaces=["becca", "chat"],
    stakes="trivial_reversible",
    templates=[
        # Direct timer-set forms with optional label
        "(set|start|begin) [a] {duration} timer",
        "(set|start) [a] timer for {duration}",
        # Reminder forms — slot "duration" captures the time, optional
        # "label" captures what to remind about.
        "remind me in {duration} to {label}",
        "remind me in {duration}",
        # Filler-tolerant + polite-imperative + gerund forms. Real STT
        # transcripts often start with hedging ("well", "so", "um"),
        # wrap in politeness ("could you", "would you", "please"), and
        # use gerunds ("setting a timer") instead of clean imperatives.
        # These templates capture those messy-but-clear command forms
        # so the architect doesn't fall through to the LLM (which has
        # historically hallucinated compliance for the very actions
        # the user just asked for).
        "[(well|so|um|uh|hey|okay)] [about] (could you|would you|can you|will you|please) (set|start|begin|create|make) [(a|the)] {duration} timer [(for me|for us)]",
        "[(well|so|um|uh|hey|okay)] [about] (could you|would you|can you|will you|please) (set|start|begin|create|make) [(a|the)] timer [(for me|for us)] (for|of) {duration} [from now]",
        "[(well|so|um|uh|hey|okay)] [about] (setting|starting|making|creating) [(a|the)] {duration} timer [(for me|for us)]",
        "[(well|so|um|uh|hey|okay)] [about] (setting|starting|making|creating) [(a|the)] timer [(for me|for us)] (for|of) {duration} [from now]",
        # "give me a 5 minute timer" / "give me a timer for 5 minutes"
        "give me [(a|the)] {duration} timer [(for me|for us)]",
        "give me [(a|the)] timer [(for me|for us)] (for|of) {duration} [from now]",
    ],
)
