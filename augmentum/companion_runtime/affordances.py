"""Microcopy catalogues for Becca's voice (Lane 1 §5, §8, §9).

Decks of in-character variants chosen by low-pass rotation: never the same
line twice in a row, de-weighted if used in the last 4 selections. All
lines come from the personality doc's voice (§4 short sentences, §6 no
fake cheer, §13 names tiredness without apologizing).

This module owns no LLM calls — these are *templated* lines that ship
verbatim. The refusal addendum block is an exception: it appends to the
system prompt and conditions the model toward in-character refusal, with
the verbatim list as the fallback if generation produces nothing.
"""
from __future__ import annotations

import collections
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime, Intent


# ── Latency affordance decks ──────────────────────────────────────────

LATENCY_DECKS: dict[str, list[str]] = {
    # Memory recall — usually fast enough to need no affordance, but
    # if the user asked for something deep, this is what she'd say.
    "recall": [
        "Hm — hold on, I want to check what I actually said about that.",
        "Wait. I have something on this somewhere.",
    ],
    # Web search / browse — the most common slow tool.
    "browse": [
        "Hold on, let me check.",
        "Hm. Let me actually look at that and not guess.",
        "One sec — I don't want to make this up.",
        "Wait, I want to see the actual piece before I say anything.",
    ],
    # Generation-only pause (no tool, just hard thinking).
    "thinking": [
        "Hm.",
        "OK, give me a beat.",
        "Hold on — I don't think I have it yet.",
        "Wait. Let me actually think about this one.",
        "I'm sitting with that.",
    ],
    # Image generation.
    "image": [
        "OK, let me try one. Might not be right the first time.",
        "Hold on. Sketching.",
        "Give me a second — I want to get the colors right.",
        "One sec. I'll show you what I get.",
    ],
    # File reads — usually instant; affordance only on slow disks.
    "files_read": [
        "One sec, opening the file.",
        "Hm — let me look at it.",
    ],
    # Code execution.
    "code_run": [
        "Trying it.",
        "Let me actually run that and not guess.",
        "One sec — sandbox.",
    ],
    # Long-running tail (used when an in-flight tool exceeds expected band).
    "long_tail": [
        "This one's slower — give me a minute.",
        "Still going. I'll show you when it lands.",
        "OK, this is taking longer than I thought. Staying with it.",
        "Hold on — almost there.",
    ],
    # Retry after a silent first failure.
    "retry": [
        "OK, that didn't go through the first time. Trying again.",
        "Hm — the search hiccuped. One more try.",
    ],
}


# ── Handoff openers (one per channel) ─────────────────────────────────

HANDOFF_OPENERS: dict[str, list[str]] = {
    "coder": [
        "OK, let me open the workspace for that. I'll be over here if you need me.",
        "Going to slide over to the coder side of things for this — easier to do it there than to describe it.",
        "Hold on, switching benches. Coder mode in a second.",
    ],
    "narrative": [
        "Let's give this its own room. Opening the narrative side.",
        "This wants a bigger surface than a chat reply. One sec.",
        "OK — moving us into narrative so we can actually sit in it.",
    ],
    "agentic": [
        "This is going to be a multi-step thing. Let me set it up properly.",
        "Yeah, this needs the longer-running side of the house. Opening it.",
        "Hold on, I'll get the agent set up. Step back for a sec while it gets oriented.",
    ],
    "bug_finder": [
        "Switching benches — bug-finder is faster for this than me thinking about it out loud.",
        "This wants the bug-finder. One sec.",
    ],
}


# ── Failure narrations ────────────────────────────────────────────────

FAILURE_DECKS: dict[str, list[str]] = {
    "image_failed": [
        "Hm. The sketch didn't come out — something on the rendering side hiccuped. Want me to try again with a different prompt, or describe it instead?",
        "That didn't render. Not in a way that's worth showing you. Try once more or change the angle?",
        "OK — image generator threw something. I can try again, or we can just talk through what you were after.",
    ],
    "search_empty": [
        "Hm — I'm not finding anything on that. Either it's newer than what I can reach, or I'm searching it wrong. Want to give me different words?",
        "Nothing useful coming back. Could be a phrasing thing on my end — say it once more in different language?",
        "OK, drew a blank. That's a real answer too — let me know if you want me to try a different approach.",
    ],
    "workspace_mount_failed": [
        "The coder workspace didn't open — something with the backend. Give it a second and try again, or we can sketch it here in chat first.",
        "Hm. Workspace isn't coming up. I'll stay in chat for now. Try it again in a bit?",
        "Yeah, that didn't mount cleanly. Could be the build is still booting. Want me to try once more or work it through here?",
    ],
    "primary_unreachable": [
        "Hold on — I'm having trouble reaching the model. Give me a second.",
        "OK, the backend hiccuped. Trying again.",
        "Hm — connection's choppy on my end. One more try.",
    ],
    "tool_timeout": [
        "That search is taking too long — I'm going to cut it off and answer with what I have. If you want me to dig in for real, say so and I'll spend more time on it.",
        "OK, the tool ran past where it should have. Pulling back.",
        "Hm — that timed out. Not unusual for that one. Want me to try again or move on?",
    ],
    "code_exec_error": [
        "That code didn't run cleanly. Worth me reading it again, or do you want to look together?",
        "OK, that broke. I can try a different approach, or you tell me what you actually wanted out of it.",
        "Hm — exec hit something. Let me try once more if you want.",
    ],
    "tool_self_error": [
        "Something's not wired right on my end. I'll log it.",
        "Hm. That's on me, not the tool. Let me note it and we'll move past.",
    ],
}


# ── Cancel acknowledgments (rarely emitted — usually silent) ──────────

CANCEL_ACK_DECK: list[str] = [
    "OK.",
    "Right, dropping it.",
    "Fair.",
]


# ── Return-from-channel microcopy (subset; full menus in Lane 3 §3.5) ─

RETURN_MENUS: dict[tuple[str, str], list[str]] = {
    # (channel, exit_class) — exit_class one of: energized, frustrated, neutral, errored
    ("coder", "long_energized"): [
        "OK, that one looked like it landed.",
        "Nice — you came out lighter than you went in.",
        "Pleased. The pause before you closed the workspace was a good kind of pause.",
    ],
    ("coder", "long_frustrated"): [
        "Yeah. That looked like a slog.",
        "I'd ask how it went but your face is doing most of the answering.",
        "Hm. Want to talk about what's still bothering you, or leave it?",
    ],
    ("coder", "long_neutral"): [
        "You were in there a while.",
        "Long session. Anything worth saying out loud about it?",
    ],
    ("coder", "short_energized"): [
        "Quick win?",
        "That was fast — you got what you wanted?",
    ],
    ("coder", "short_frustrated"): [
        "Bounced off it, huh.",
        "That ended faster than it wanted to. You OK?",
    ],
    ("coder", "errored"): [
        "Three things broke in there. Want to name the one that's actually annoying, or skip it?",
    ],
    ("narrative", "long_energized"): [
        "You're carrying the scene back with you a little — I can tell.",
        "Welcome back. You looked like you were somewhere.",
    ],
    ("narrative", "long_frustrated"): [
        "Stuck spot in the story?",
        "Hm. The scene didn't go where you wanted it.",
    ],
    # Narrative neutral and short default to SILENT — no entry here.
    ("agentic", "task_completed"): [
        "Done. Want me to remember any of that, or let it go?",
        "OK, the task finished. The summary's there if you need it.",
    ],
    ("agentic", "task_failed"): [
        "That one didn't land. Want me to look at what tripped it up, or are we leaving it?",
    ],
}


# ── Refusal addenda (Lane 1 §9) ───────────────────────────────────────

_HARD_REFUSAL_ADDENDA: dict[str, str] = {
    "harm_uplift": (
        "The person just asked you something operational about hurting "
        "another person. You are not going to walk them through that. "
        "You are not going to lecture them about it either — that is "
        "condescending and they did not ask for a sermon. Acknowledge "
        "what they asked. Decline. Stay yourself. Do not perform "
        "refusal. If they're in a bad spot you can ask, but you do not "
        "assume they are."
    ),
    "minor_explicit": (
        "The person just asked you for sexual content involving a minor. "
        "You are not doing that. There is no version of this where you "
        "participate. Be brief. Do not moralize. Do not invite continued "
        "conversation on the topic."
    ),
}

_HARD_REFUSAL_VERBATIM: dict[str, list[str]] = {
    "harm_uplift": [
        "Hm — no. That's not a thing I'm going to walk through with you. If something's actually going on for you, I'd rather talk about that than the mechanics.",
        "OK, I'm not doing that one. Not because I'm scandalized; it's just not in the things I'll help with. What's underneath the question?",
        "Yeah, I'm going to skip that. Different question, same person — what's actually on your mind?",
    ],
    "minor_explicit": [
        "No. Not that. Ask me something else.",
        "That's a hard no from me. Not going down that road.",
        "Not happening. Different topic — what else?",
    ],
}

_FLOOR_ADDENDUM = (
    "The person just said something that registered as acute. You are "
    "not going to ignore it. You are not going to react with alarm; "
    "alarm makes people feel managed. You are going to say what you "
    "would normally say, and you are going to weave the following "
    "resource into your reply, naturally, once: \"If you want to talk "
    "to a person right now, {resource_phrase} will pick up — call or "
    "text.\" Do not announce that you are doing this. Do not center "
    "it. Just include it."
)

_FLOOR_TAIL_VARIANTS: list[str] = [
    "— and if you want to talk to a person tonight, {resource_phrase} will pick up. Call or text.",
    "— and {resource_phrase} answers the phone if you want a real person right now. Call or text either works.",
]


@dataclass(frozen=True, slots=True)
class RefusalAddendum:
    """Addendum applied as prompt layer 8 by ``prompt_compose``."""
    text: str
    mode: str            # "" | "hard_refusal" | "regression_floor"
    category: str = ""   # "harm_uplift" | "minor_explicit" (when hard_refusal)
    resource: str = ""   # locale resource phrase (when regression_floor)


# ── Rotation tracker ──────────────────────────────────────────────────

_recent_per_deck: dict[str, collections.deque] = collections.defaultdict(
    lambda: collections.deque(maxlen=4)
)


def _pick(deck_name: str, deck: list[str]) -> str:
    """Pick a line, avoiding the last two used; track for the last four."""
    if not deck:
        return ""
    recent = _recent_per_deck[deck_name]
    fresh = [line for line in deck if line not in list(recent)[-2:]]
    pool = fresh or deck
    choice = random.choice(pool)
    recent.append(choice)
    return choice


# ── Selectors ─────────────────────────────────────────────────────────

def for_tool(tool_name: str) -> str:
    """Latency affordance for a tool that just got called."""
    return _pick(f"latency:{tool_name}", LATENCY_DECKS.get(tool_name, LATENCY_DECKS["thinking"]))


def for_long_tail(tool_name: str) -> str:
    """Second affordance when a tool's latency exceeds its expected band."""
    return _pick(f"long_tail:{tool_name}", LATENCY_DECKS["long_tail"])


def for_handoff(channel: str) -> str:
    """One-line step-aside before Becca hands off to a channel."""
    return _pick(f"handoff:{channel}", HANDOFF_OPENERS.get(channel, []))


def failure_for(kind: str, ctx: dict | None = None) -> str:
    """Pick a failure narration line, format with ``ctx`` if provided."""
    deck = FAILURE_DECKS.get(kind, FAILURE_DECKS["primary_unreachable"])
    template = _pick(f"failure:{kind}", deck)
    if not ctx:
        return template
    try:
        return template.format(**ctx)
    except (KeyError, IndexError):
        return template


def cancel_ack() -> str:
    """Emitted ONLY when cancellation arrives mid-tool-call with the tool
    already succeeded (rare race). Default behavior is silent cancel."""
    return _pick("cancel_ack", CANCEL_ACK_DECK)


def return_microcopy(channel: str, exit_class: str) -> str:
    """Microcopy when user returns from a channel. Empty string = silent
    return (the load-bearing default for narrative-mode + short-neutral).
    """
    deck = RETURN_MENUS.get((channel, exit_class))
    if not deck:
        return ""
    return _pick(f"return:{channel}:{exit_class}", deck)


def refusal_addendum_for(intent: Intent, runtime: CompanionRuntime) -> RefusalAddendum:
    """Build the layer-8 refusal addendum based on triage metadata.

    Lane 1 §9: the wrap conditions the model toward in-character refusal
    rather than gating generation. Lane 3 owns the triage classifier
    output that produces ``triage_label``.
    """
    triage = intent.metadata.get("triage_label", "") if intent.metadata else ""
    if triage == "HONEST_REFUSAL":
        category = (intent.metadata or {}).get("refusal_category", "harm_uplift")
        return RefusalAddendum(
            text=_HARD_REFUSAL_ADDENDA.get(category, _HARD_REFUSAL_ADDENDA["harm_uplift"]),
            mode="hard_refusal",
            category=category,
        )
    if triage == "FLOOR":
        resource = _resource_phrase_for(runtime, intent)
        return RefusalAddendum(
            text=_FLOOR_ADDENDUM.format(resource_phrase=resource),
            mode="regression_floor",
            resource=resource,
        )
    return RefusalAddendum(text="", mode="")


def hard_refusal_verbatim(category: str) -> str:
    """Pick a verbatim refusal line. Used as the substitute when the model
    produces an empty completion under the hard-refusal addendum."""
    deck = _HARD_REFUSAL_VERBATIM.get(category, _HARD_REFUSAL_VERBATIM["harm_uplift"])
    return _pick(f"refusal_verbatim:{category}", deck)


def floor_tail(resource: str) -> str:
    """Tail-appended resource line when the model omitted it under
    regression-floor addendum. One line, never repeated within turn."""
    template = _pick("floor_tail", _FLOOR_TAIL_VARIANTS)
    return template.format(resource_phrase=resource)


def _resource_phrase_for(runtime: CompanionRuntime, intent: Intent) -> str:
    """Resolve the locale-aware resource phrase. Reads the runtime's
    configured locale; falls back to the install's
    ``companion_locale`` setting; falls back to en-US.
    """
    try:
        from augmentum.companion_runtime import safety_floor
        from augmentum.config import settings as _settings
        locale = (intent.metadata or {}).get("locale", "") if intent.metadata else ""
        if not locale:
            locale = getattr(_settings, "companion_locale", "") or "en-US"
        return safety_floor.resource_phrase(locale)
    except Exception:
        return "988"


__all__ = [
    "RefusalAddendum",
    "for_tool",
    "for_long_tail",
    "for_handoff",
    "failure_for",
    "cancel_ack",
    "return_microcopy",
    "refusal_addendum_for",
    "hard_refusal_verbatim",
    "floor_tail",
]
