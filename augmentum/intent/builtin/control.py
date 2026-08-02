"""Conversation-control actions — stop, repeat, slower, louder, bye,
nevermind.

These are the "fast-path" candidates — they're addressed to the
conversation infrastructure rather than to Becca's reasoning, so they
should never wait on an LLM. The frontend mirror (ui/scripts/voice/
intent-fast-path.js) keeps a copy of these patterns and acts on them
before the transcript even reaches the server in the wake-PTT path;
the server-side handlers below are the authoritative fallback for
surfaces that don't run the fast-path (cast voice, future voice over
non-WS transports, text Ask bars).

All six emit ``surface_emit`` payloads keyed to the frontend
intent-action-router. None of them speak — the action itself is the
feedback (TTS pauses, voice closes, etc.).
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import (
    ActionFanout,
    ActionResult,
    SessionContext,
)
from augmentum.intent.registry import register_action

# Fast-path actions opt out of LLM tool exposure — there's no scenario
# where the model should "call" a stop action. They're pure surface
# controls keyed off recognized phrasings.
_CONTROL_FANOUT = ActionFanout(tier1=True, tier2=True, tier3=False, fast_path=True)


async def _control_stop(
    _text: str, _session: SessionContext, _args: dict[str, Any],
) -> ActionResult:
    return ActionResult(
        short_circuit=True,
        surface_emit={"channel": "tts.cancel", "payload": {}},
    )


register_action(
    id="control.stop",
    summary="Stop the current TTS playback immediately.",
    examples=["stop", "shut up", "be quiet", "enough", "quiet"],
    patterns=[
        r"^(?:stop|shut up|be quiet|enough|quiet)$",
        r"\b(?:please )?stop talking\b",
    ],
    handler=_control_stop,
    delivery="artifact",
    fanout=_CONTROL_FANOUT,
)


async def _control_repeat(
    _text: str, _session: SessionContext, _args: dict[str, Any],
) -> ActionResult:
    return ActionResult(
        short_circuit=True,
        surface_emit={"channel": "tts.repeat_last", "payload": {}},
    )


register_action(
    id="control.repeat",
    summary="Replay the last TTS clip.",
    examples=["what?", "say that again", "huh?", "repeat that", "come again"],
    patterns=[
        r"^(?:what|huh|sorry|pardon)$",
        r"\b(?:say|can you say) that again\b",
        r"\brepeat that\b",
        r"\bcome again\b",
    ],
    handler=_control_repeat,
    delivery="artifact",
    fanout=_CONTROL_FANOUT,
)


async def _control_slower(
    _text: str, _session: SessionContext, _args: dict[str, Any],
) -> ActionResult:
    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "tts.resynth_last",
            "payload": {"speed_factor": 0.85},
        },
    )


register_action(
    id="control.slower",
    summary="Re-speak the last reply at a slower rate.",
    examples=["slower", "slow down", "speak slower"],
    patterns=[r"^(?:slower|slow down)$", r"\bspeak slower\b"],
    handler=_control_slower,
    delivery="artifact",
    fanout=_CONTROL_FANOUT,
)


async def _control_louder(
    _text: str, _session: SessionContext, _args: dict[str, Any],
) -> ActionResult:
    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "tts.volume_bump",
            "payload": {"delta": 0.15},
        },
    )


register_action(
    id="control.louder",
    summary="Bump TTS playback volume.",
    examples=["louder", "speak up", "I can't hear you"],
    patterns=[
        r"^(?:louder|speak up)$",
        r"\bi can'?t hear you\b",
    ],
    handler=_control_louder,
    delivery="artifact",
    fanout=_CONTROL_FANOUT,
)


async def _control_goodbye(
    _text: str, _session: SessionContext, _args: dict[str, Any],
) -> ActionResult:
    # Closes the follow-up window early — frontend handles the actual
    # WS teardown via the surface event. Speak a short ack so the user
    # gets confirmation the session closed cleanly.
    return ActionResult(
        short_circuit=True,
        surface_emit={"channel": "conversation.close", "payload": {}},
        speak="Okay.",
    )


register_action(
    id="control.goodbye",
    summary="End the conversation; close any follow-up window.",
    examples=[
        "bye becca", "goodbye becca", "thanks bye",
        "that's all", "thanks that's all", "we're done",
    ],
    patterns=[
        r"\b(?:bye|goodbye)(?:\s+becca)?\b",
        r"\bthanks?(?:[,\s]+(?:bye|that'?s all))\b",
        r"\bthat'?s all\b",
        r"\bwe'?re done\b",
    ],
    handler=_control_goodbye,
    delivery="artifact",
    fanout=_CONTROL_FANOUT,
)


async def _control_nevermind(
    _text: str, _session: SessionContext, _args: dict[str, Any],
) -> ActionResult:
    # Abort the in-flight turn. Cancels TTS if she was mid-response,
    # drops the current capture if she was still listening.
    return ActionResult(
        short_circuit=True,
        surface_emit={"channel": "turn.abort", "payload": {}},
    )


register_action(
    id="control.nevermind",
    summary="Abort the current turn — cancel TTS and drop the capture.",
    examples=["never mind", "nvm", "forget it", "cancel that"],
    patterns=[
        r"\bnever ?mind\b",
        r"^nvm$",
        r"\bforget it\b",
        r"\bcancel that\b",
    ],
    handler=_control_nevermind,
    delivery="artifact",
    fanout=_CONTROL_FANOUT,
)


async def _control_strike(
    _text: str, _session: SessionContext, _args: dict[str, Any],
) -> ActionResult:
    # "Scratch that / disregard that last recording / strike it from
    # context." Distinct from nevermind: nevermind aborts the IN-FLIGHT
    # turn (turn.abort); strike removes the LAST COMMITTED exchange from
    # the model's working context. This is the user's manual escape
    # hatch for mangled-STT poisoning — when a misheard utterance already
    # got a reply and is now sitting in history, one phrase pops both.
    #
    # The actual history pop is server-side (the WS route owns
    # ``session.messages``); this surface_emit just removes the matching
    # bubbles from the transcript log so what she "forgot" disappears on
    # screen too. The spoken ack confirms it landed — important because a
    # silent strike of a real exchange would read as her glitching.
    return ActionResult(
        short_circuit=True,
        surface_emit={"channel": "conversation.strike", "payload": {}},
        speak="Okay, scratched that.",
    )


register_action(
    id="conversation.strike",
    summary=(
        "Strike the last exchange from context — for 'scratch that' when "
        "mangled speech poisoned the conversation."
    ),
    examples=[
        "scratch that", "strike that", "disregard that last recording",
        "ignore my last message", "that wasn't meant for you",
        "pretend I didn't say that", "strike it from context",
    ],
    patterns=[
        r"\bscratch that\b",
        r"\bstrike that\b",
        r"\bdisregard (?:that|the last|what i (?:just )?said)\b",
        r"\bignore (?:that last|my last|the last) "
        r"(?:bit|message|recording|thing|one)\b",
        r"\bdelete (?:that|the) last\b",
        r"\bthat was(?:n'?t| not)(?: meant)? for you\b",
        r"\bpretend i (?:didn'?t|did not) say that\b",
        r"\bstrike (?:that|it) from (?:context|the record)\b",
    ],
    handler=_control_strike,
    delivery="artifact",
    fanout=_CONTROL_FANOUT,
)
