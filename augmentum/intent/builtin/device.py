"""device.* — phone-as-capability-provider verbs.

The first verb whose result lives on the user's *phone*, not the server.
``device.bluetooth_list`` asks the paired Android device "what Bluetooth
devices are you connected to / paired with?" and returns that into the
companion loop, which narrates it.

The round-trip rides the always-on notification WebSocket via
:class:`augmentum.notifications.device_bus.DeviceCommandBus` — no new
socket, no trusted cert needed (the phone's foreground service holds the
connection open already). This is the proving ground for the whole
phone-as-capability-provider direction: if this loop works on glass,
every later phone capability (NFC, more sensors, media handoff target
hints) is the same shape — register a verb, the bus carries it.

No regex anywhere: the MODEL decides this verb is relevant from its
schema (``select_companion_tools`` relevance-ranks it into the roster);
the transcript is never pattern-matched. See
docs/superpowers/specs/2026-06-17-phone-capability-provider.md.
"""
from __future__ import annotations

import json
from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)


def _summarize_bluetooth(devices: list[dict[str, Any]]) -> str:
    """A spoken line built from the phone's actual device list.

    Built deterministically (delivery='artifact' relays it verbatim) so
    real device names never pass through a model that could invent them
    — the same lesson weather.today learned with numbers.
    """
    if not devices:
        return "Your phone isn't paired with any Bluetooth devices right now."

    def _name(d: dict[str, Any]) -> str:
        return str(d.get("name") or d.get("address") or "an unnamed device")

    connected = [d for d in devices if d.get("connected")]
    if connected:
        names = [_name(d) for d in connected]
        if len(names) == 1:
            head = f"You're connected to {names[0]}."
        else:
            head = (
                "You're connected to "
                + ", ".join(names[:-1])
                + f" and {names[-1]}."
            )
        others = len(devices) - len(connected)
        if others > 0:
            head += f" {others} more paired but not connected."
        return head

    names = [_name(d) for d in devices[:4]]
    more = "" if len(devices) <= 4 else f", plus {len(devices) - 4} more"
    return (
        "You're not connected to anything over Bluetooth right now. "
        f"Paired devices: {', '.join(names)}{more}."
    )


async def _bluetooth_list_handler(
    _text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I can't reach your phone for a signed-out session.",
        )

    from augmentum.config import settings

    if not getattr(settings, "companion_device_tools_enabled", False):
        return ActionResult(
            short_circuit=True,
            speak="Phone device tools are turned off right now.",
        )

    app_state = getattr(session, "app_state", None)
    if app_state is None:
        return ActionResult(
            short_circuit=True,
            speak="I couldn't reach your phone just now.",
        )

    from augmentum.notifications.device_bus import get_device_bus

    bus = get_device_bus(app_state)
    timeout = float(
        getattr(settings, "companion_device_command_timeout_s", 8.0) or 8.0
    )
    result = await bus.request(
        user_id=session.user_id, action="bluetooth_list", timeout=timeout,
    )

    if not isinstance(result, dict) or result.get("ok") is False:
        err = result.get("error") if isinstance(result, dict) else "error"
        if err == "not_connected" or err == "no_hub":
            return ActionResult(
                short_circuit=True,
                speak=(
                    "Your phone isn't connected to me right now, so I "
                    "can't check its Bluetooth."
                ),
            )
        if err == "timeout":
            return ActionResult(
                short_circuit=True,
                speak=(
                    "Your phone didn't answer in time — I couldn't read "
                    "its Bluetooth."
                ),
            )
        return ActionResult(
            short_circuit=True,
            speak="I couldn't reach your phone's Bluetooth just now.",
        )

    devices = result.get("devices")
    if not isinstance(devices, list):
        devices = []
    # Normalise to the fields the summary + follow-ups use; ignore extras.
    clean = [
        {
            "name": str(d.get("name") or "")[:80],
            "address": str(d.get("address") or "")[:32],
            "connected": bool(d.get("connected")),
            "type": str(d.get("type") or "")[:32],
        }
        for d in devices
        if isinstance(d, dict)
    ]

    return ActionResult(
        short_circuit=True,
        speak=_summarize_bluetooth(clean),
        # Structured data rides the model-visible payload so "switch to
        # my headphones" / "is the car still connected" answer from
        # context without another round-trip to the phone.
        prompt_addendum=f"[bluetooth devices] {json.dumps(clean)}",
    )


# ─────────────────────────────────────────────────────────────────────
# Shared phone round-trip + framing helpers for the action verbs below.
# ─────────────────────────────────────────────────────────────────────

# Friendly spoken lines for the bus-level error envelope every verb can
# hit. Phone-side verb errors (e.g. "no_app") get their own lines in the
# handler that knows what it asked for.
_BUS_ERROR_SPEAK = {
    "not_connected": (
        "Your phone isn't connected to me right now, so I can't do that "
        "on it."
    ),
    "no_hub": (
        "Your phone isn't connected to me right now, so I can't do that "
        "on it."
    ),
    "timeout": "Your phone didn't answer in time, so I'm not sure that went through.",
}


async def _phone_request(
    session: SessionContext, action: str, params: dict[str, Any],
) -> tuple[dict[str, Any] | None, ActionResult | None]:
    """Emit a CLIENT-EXECUTED device effect for an action verb.

    Instead of a fragile server→phone round-trip over the notification
    WebSocket (which fails whenever that socket is mid-reconnect — "your
    phone isn't fully connected"), we queue a ``device`` surface event and
    the app — which is right here running the turn and already holds
    ``DeviceCommands`` — executes the native intent itself (the assist
    overlay runs it directly; the WebView runs it via the native bridge,
    from a foreground Activity so there's no background-launch block).

    Returns ``({"ok": True, "client_effect": True}, None)`` once queued, or
    ``(None, error_action)`` if the gate fails. The verb narrates
    optimistically; the app confirms via its on-screen receipt. No phone
    echo, so result-derived fields (scheduled_for, …) are absent by design.

    Read-style verbs that need the phone's DATA back (device.bluetooth_list)
    do NOT use this — they still round-trip via the device bus directly.
    """
    if not session.user_id:
        return None, ActionResult(
            short_circuit=True,
            speak="I can't reach your phone for a signed-out session.",
        )

    from augmentum.config import settings

    if not getattr(settings, "companion_device_tools_enabled", False):
        return None, ActionResult(
            short_circuit=True,
            speak="Phone device tools are turned off right now.",
        )

    app_state = getattr(session, "app_state", None)
    if app_state is None:
        return None, ActionResult(
            short_circuit=True, speak="I couldn't set that up just now.",
        )

    try:
        from augmentum.intent.dispatch import get_referent_cache
        refs = get_referent_cache(app_state, session.user_id, session.session_id)
        refs.pending_surface_events.append({
            "type": "intent_action",
            "v": 1,
            "action": f"device.{action}",
            "tier": 3,
            "short_circuit": True,
            "surface": {
                "channel": "device",
                "payload": {"action": action, "params": params},
            },
        })
    except Exception:
        log.warning("device_effect_emit_failed", action=action, exc_info=True)
        return None, ActionResult(
            short_circuit=True, speak="I couldn't set that up just now.",
        )

    return {"ok": True, "client_effect": True}, None


def _fmt_clock(hour: int, minute: int) -> str:
    """24h ints → a spoken 12-hour string ("7:30 AM", "9 PM")."""
    suffix = "AM" if hour < 12 else "PM"
    h12 = hour % 12 or 12
    return f"{h12}:{minute:02d} {suffix}" if minute else f"{h12} {suffix}"


def _humanize_seconds(total: int) -> str:
    """Whole seconds → "1 hour 30 minutes" / "10 minutes" / "45 seconds"."""
    hours, rem = divmod(int(total), 3600)
    minutes, seconds = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds and not hours:  # drop seconds once we're into hours
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    return " ".join(parts) or "0 seconds"


# ─────────────────────────────────────────────────────────────────────
# device.set_alarm — phone clock-app alarm (AlarmClock.ACTION_SET_ALARM)
# ─────────────────────────────────────────────────────────────────────

async def _set_alarm_handler(
    _text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    from augmentum.intent.device_normalize import parse_clock_time

    raw = str(args.get("time") or "").strip()
    when = parse_clock_time(raw)
    if when is None:
        return ActionResult(
            short_circuit=True,
            speak=(
                "I didn't catch what time to set the alarm for — when "
                "should it go off?"
            ),
        )

    label = str(args.get("label") or "").strip()[:80]
    params: dict[str, Any] = dict(when.to_payload())
    if label:
        params["label"] = label

    result, err = await _phone_request(session, "set_alarm", params)
    if err is not None:
        return err

    # Speak the resolved wall-clock time. For a relative alarm the phone
    # resolves it against its own clock and echoes it back as
    # "scheduled_for" (server tz isn't the user's).
    if when.is_relative:
        scheduled = str(result.get("scheduled_for") or "").strip() if result else ""
        spoke = (
            f"Alarm set for {scheduled}."
            if scheduled
            else f"Alarm set for {_humanize_seconds(when.in_seconds or 0)} from now."
        )
    else:
        spoke = f"Alarm set for {_fmt_clock(when.hour or 0, when.minute or 0)}."
    if label:
        spoke = spoke[:-1] + f" — {label}."
    return ActionResult(short_circuit=True, speak=spoke)


register_action(
    id="device.set_alarm",
    summary=(
        "Set an alarm in the clock app on the user's phone so it rings "
        "even if Augmentum is closed. Use for 'wake me at 7', 'set an "
        "alarm for 6:30am'. Pass the time exactly as the user said it "
        "(e.g. '7am', 'quarter past six', 'in 8 hours') — do not convert "
        "it to numbers yourself. Only meaningful on their Android phone."
    ),
    examples=["set an alarm for 7am", "wake me up at 6:30", "alarm for half past five"],
    handler=_set_alarm_handler,
    arg_schema={
        "time": {
            "type": "string",
            "description": (
                "The alarm time in the user's own words: '7am', '7:30 pm', "
                "'19:00', 'noon', 'quarter past seven', or a relative "
                "'in 8 hours'."
            ),
        },
        "label": {
            "type": "string",
            "description": "Optional short label, e.g. 'gym' or 'meeting'.",
        },
    },
    required=["time"],
    surfaces=["becca", "chat", "voice"],
    stakes="disruptive",
    fanout=_TIER3_ONLY,
)


# ─────────────────────────────────────────────────────────────────────
# device.set_timer — phone countdown timer (AlarmClock.ACTION_SET_TIMER)
# ─────────────────────────────────────────────────────────────────────

async def _set_timer_handler(
    _text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    from augmentum.intent.device_normalize import parse_duration

    raw = str(args.get("duration") or "").strip()
    seconds = parse_duration(raw)
    if seconds is None or seconds <= 0:
        return ActionResult(
            short_circuit=True,
            speak="How long should the timer run for?",
        )

    label = str(args.get("label") or "").strip()[:80]
    params: dict[str, Any] = {"seconds": int(seconds)}
    if label:
        params["label"] = label

    result, err = await _phone_request(session, "set_timer", params)
    if err is not None:
        return err

    spoke = f"Timer started for {_humanize_seconds(int(seconds))}."
    if label:
        spoke = spoke[:-1] + f" — {label}."
    return ActionResult(short_circuit=True, speak=spoke)


register_action(
    id="device.set_timer",
    summary=(
        "Start a countdown timer in the clock app on the user's phone "
        "(rings even if Augmentum is closed). Use for 'set a timer for 10 "
        "minutes', 'timer for an hour and a half'. This is the phone's "
        "hardware timer — distinct from any in-app playback sleep timer. "
        "Pass the duration as the user said it; do not convert to seconds."
    ),
    examples=["set a timer for 10 minutes", "timer for 90 seconds", "give me an hour and a half"],
    handler=_set_timer_handler,
    arg_schema={
        "duration": {
            "type": "string",
            "description": (
                "How long the timer runs, in the user's words: '10 "
                "minutes', '90 seconds', '1h30m', 'an hour and a half'."
            ),
        },
        "label": {
            "type": "string",
            "description": "Optional short label, e.g. 'pasta' or 'laundry'.",
        },
    },
    required=["duration"],
    surfaces=["becca", "chat", "voice"],
    stakes="disruptive",
    fanout=_TIER3_ONLY,
)


# ─────────────────────────────────────────────────────────────────────
# device.dial — open the dialer pre-filled (ACTION_DIAL). The user taps
# Call; we never place the call ourselves (consent by construction).
# ─────────────────────────────────────────────────────────────────────

async def _dial_handler(
    _text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    number = str(args.get("number") or "").strip()
    name = str(args.get("name") or "").strip()[:60]
    # Keep only dial-safe characters; the phone wraps it in a tel: URI.
    clean = "".join(c for c in number if c.isdigit() or c in "+*#")
    if not clean:
        who = f" for {name}" if name else ""
        return ActionResult(
            short_circuit=True,
            speak=(
                f"I don't have a phone number to dial{who}. Tell me the "
                "number and I'll pull up the dialer."
            ),
        )

    result, err = await _phone_request(session, "dial", {"number": clean})
    if err is not None:
        return err

    who = name or clean
    return ActionResult(
        short_circuit=True,
        speak=(
            f"I've pulled up the dialer for {who} — tap call to ring them. "
            "I won't place the call myself."
        ),
    )


register_action(
    id="device.dial",
    summary=(
        "Open the phone's dialer pre-filled with a number so the user can "
        "tap Call. Use when they say 'call <number>' / 'dial <number>'. "
        "You only OPEN the dialer — the user places the call. Never claim "
        "you called someone. You must supply an actual phone number; if "
        "the user only gave a contact name and no number, say you don't "
        "have the number rather than guessing."
    ),
    examples=["call 555 0142", "dial 911", "ring this number 8005551234"],
    handler=_dial_handler,
    arg_schema={
        "number": {
            "type": "string",
            "description": "The phone number to dial (digits, +, *, #).",
        },
        "name": {
            "type": "string",
            "description": "Optional name of who it belongs to, for the spoken line only.",
        },
    },
    required=["number"],
    surfaces=["becca", "chat", "voice"],
    stakes="disruptive",
    fanout=_TIER3_ONLY,
)


# ─────────────────────────────────────────────────────────────────────
# device.compose_text — open the SMS composer pre-filled (ACTION_SENDTO).
# The user taps Send; nothing is sent automatically.
# ─────────────────────────────────────────────────────────────────────

async def _compose_text_handler(
    _text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    number = str(args.get("number") or "").strip()
    clean = "".join(c for c in number if c.isdigit() or c in "+*#")
    body = str(args.get("body") or "").strip()[:1000]
    if not clean:
        return ActionResult(
            short_circuit=True,
            speak=(
                "I need a phone number to open a text to. What's the "
                "number?"
            ),
        )

    params: dict[str, Any] = {"number": clean}
    if body:
        params["body"] = body
    result, err = await _phone_request(session, "compose_text", params)
    if err is not None:
        return err

    drafted = " with your message drafted" if body else ""
    return ActionResult(
        short_circuit=True,
        speak=(
            f"I've opened a text to {clean}{drafted} — hit send when you're "
            "ready. I won't send it for you."
        ),
    )


register_action(
    id="device.compose_text",
    summary=(
        "Open the phone's messaging app pre-filled with a recipient number "
        "and (optionally) a drafted message body, so the user can review "
        "and tap Send. Use for 'text <number> ...', 'send a message to "
        "<number>'. You only OPEN the composer — the user sends it. Never "
        "claim you sent a text. Requires an actual phone number."
    ),
    examples=["text 5550142 running late", "message 8005551234", "send a text saying on my way to 5550199"],
    handler=_compose_text_handler,
    arg_schema={
        "number": {
            "type": "string",
            "description": "Recipient phone number (digits, +, *, #).",
        },
        "body": {
            "type": "string",
            "description": "Optional message text to pre-fill; the user still taps Send.",
        },
    },
    required=["number"],
    surfaces=["becca", "chat", "voice"],
    stakes="disruptive",
    fanout=_TIER3_ONLY,
)


# ─────────────────────────────────────────────────────────────────────
# device.add_contact — open the contact editor pre-filled (Insert). The
# user reviews and saves; we never write the contact silently.
# ─────────────────────────────────────────────────────────────────────

async def _add_contact_handler(
    _text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    name = str(args.get("name") or "").strip()[:120]
    if not name:
        return ActionResult(
            short_circuit=True,
            speak="What name should I put on the new contact?",
        )
    phone = str(args.get("phone") or "").strip()[:40]
    email = str(args.get("email") or "").strip()[:160]

    params: dict[str, Any] = {"name": name}
    if phone:
        params["phone"] = phone
    if email:
        params["email"] = email
    result, err = await _phone_request(session, "add_contact", params)
    if err is not None:
        return err

    return ActionResult(
        short_circuit=True,
        speak=(
            f"I've started a new contact for {name} — review the details "
            "and tap save."
        ),
    )


register_action(
    id="device.add_contact",
    summary=(
        "Open the phone's contact editor pre-filled with a name and "
        "optional phone/email so the user can review and save. Use for "
        "'add <name> to my contacts', 'save this number as <name>'. You "
        "only OPEN the editor — the user saves it."
    ),
    examples=["add Sarah to my contacts", "save 5550142 as Mike", "new contact for Dr. Lee email lee@clinic.com"],
    handler=_add_contact_handler,
    arg_schema={
        "name": {"type": "string", "description": "Contact's display name."},
        "phone": {"type": "string", "description": "Optional phone number."},
        "email": {"type": "string", "description": "Optional email address."},
    },
    required=["name"],
    surfaces=["becca", "chat", "voice"],
    stakes="disruptive",
    fanout=_TIER3_ONLY,
)


# ─────────────────────────────────────────────────────────────────────
# device.launch_app — open an installed app by name. The phone resolves
# the spoken label against its launchable apps and starts it.
# ─────────────────────────────────────────────────────────────────────

async def _launch_app_handler(
    _text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    app = str(args.get("app") or "").strip()[:80]
    if not app:
        return ActionResult(
            short_circuit=True, speak="Which app should I open?",
        )

    result, err = await _phone_request(session, "launch_app", {"query": app})
    if err is not None:
        return err

    # The phone resolves the real label and returns it; speak that, not
    # the user's possibly-misheard query (the device-name lesson).
    if result and result.get("error") == "no_match":
        return ActionResult(
            short_circuit=True,
            speak=f"I couldn't find an app called {app} on your phone.",
        )
    label = str((result or {}).get("label") or app)[:80]
    return ActionResult(short_circuit=True, speak=f"Opening {label}.")


register_action(
    id="device.launch_app",
    summary=(
        "Open an installed app on the user's phone by name (e.g. 'open "
        "Spotify', 'launch Maps'). The phone matches the name against its "
        "installed apps. Only meaningful on their Android phone."
    ),
    examples=["open Spotify", "launch the camera", "open Google Maps"],
    handler=_launch_app_handler,
    arg_schema={
        "app": {
            "type": "string",
            "description": "The app name to open, as the user said it (e.g. 'Spotify').",
        },
    },
    required=["app"],
    surfaces=["becca", "chat", "voice"],
    stakes="disruptive",
    fanout=_TIER3_ONLY,
)


register_action(
    id="device.bluetooth_list",
    summary=(
        "List the Bluetooth devices the user's phone is connected to or "
        "paired with (headphones, car, speaker, watch). Use when they ask "
        "what they're connected to, whether their car/headphones are "
        "connected, or which audio outputs are available. Reads the phone "
        "live — only meaningful when they're on their Android phone."
    ),
    examples=[
        "what bluetooth am I connected to",
        "am I connected to my car",
        "are my headphones connected",
        "what devices is my phone paired with",
    ],
    handler=_bluetooth_list_handler,
    # The speak line IS the data, built from the phone's real device
    # list and spoken verbatim — device names must not be synthesised by
    # a model that can invent them (weather.today's number lesson).
    delivery="artifact",
    arg_schema={},
    required=[],
    surfaces=["becca", "chat", "voice"],
    stakes="trivial_reversible",
    fanout=_TIER3_ONLY,
)
