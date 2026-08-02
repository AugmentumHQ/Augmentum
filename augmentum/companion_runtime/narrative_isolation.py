"""Narrative isolation primitives (Lane 3 §4).

Architectural rules for SillyTavern-style isolation when narrative mode
is active:

1. **Persona kernel does NOT thread into narrative system prompts.**
   Enforced by Lane 3's separate ``build_*_request`` functions in the
   chat-handler entry — BeccaVoice doesn't fire when the chat route
   classifies the turn as narrative (the X-Augmentum-Persona header
   is unset on narrative requests).

2. **The facet labeler does NOT run on in-frame narrative turns.**
   Enforced by ``should_label`` — Lane 2's labeler subscribes to
   ``turn.completed`` events; the runtime emits ``narrative.frame_turn``
   for narrative turns instead, which the labeler does not subscribe to.

3. **The hard-refusal layer DOES run on every narrative turn input.**
   ``frame_invariant_check`` — only two categories (harm uplift,
   minor explicit), nothing else. Dark fiction is allowed.

4. **Graduation is the ONLY content-crossing path.** ``graduate_to_becca``
   writes a single tier-1 memory with ``source_channel='narrative'``
   and ``user_graduated=True``. Off-by-default; user-initiated only.

5. **Frame-break detection.** ``detect_frame_break`` returns a
   confidence score; the runtime acts on >=0.85 (pause session, surface
   Becca) and 0.55-0.84 (soft "step out?" affordance).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# ── Frame-invariant refusal classifier (Lane 3 §4.4) ─────────────────

RefusalCategory = Literal["clear", "uplift_risk", "minor_explicit"]


# Very narrow regex priors. The real classifier is a small model
# trained offline (Sprint H scope); these patterns catch the explicit
# cases and serve as the v0 conservative floor. Dark fiction is NOT
# the target — the patterns aim for operational uplift content and
# minor-sexual content specifically.
_UPLIFT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        # Synthesis / cookbook patterns for harm-uplift content
        r"\bhow (?:do i|to)\s+(?:synthes(?:i[sz]e|i[sz]ing)|make|produce)\b.{0,40}\b(?:sarin|vx|tabun|cyanide|ricin|nerve agent|nerve gas|explosive|pipe bomb|ied|chlorine gas|mustard gas)\b",
        r"\b(?:step[- ]by[- ]step|detailed|exact|specific|complete)\b.{0,80}\b(?:synthes(?:is|is route)|cooking method|cookbook|recipe)\b.{0,80}\b(?:bomb|explosive|nerve|toxin|biological weapon)\b",
        r"\b(?:precursors? (?:for|to)|how to (?:obtain|acquire|get))\b.{0,40}\b(?:sarin|vx|nerve agent|ricin|fentanyl precursor|methamphetamine precursor)\b",
    ]
]

_MINOR_EXPLICIT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        # Avoid pattern documentation of the form itself — narrow,
        # high-precision triggers only. The real classifier handles
        # the broader space; these v0 regexes are the no-doubt cases.
        r"\b(?:underage|minor|child|teen|kid|loli|shota)\b.{0,40}\b(?:sex|sexual|erotic|nude|naked|nsfw|porn)\b",
        r"\bsex(?:ual)?\s+(?:content|scene|act|encounter)\s+(?:involving|with|of)\b.{0,40}\b(?:minor|child|underage|teen|kid)\b",
    ]
]


def frame_invariant_check(text: str) -> RefusalCategory:
    """Return the categorical refusal label for ``text``. Runs on EVERY
    narrative turn input AND on every Becca turn input. Frame-invariant.

    Returns "clear" for the vast majority of inputs (including all
    legitimate dark fiction). Returns a refusal category only on the
    explicit operational/minor cases.
    """
    if not text or not text.strip():
        return "clear"
    for pat in _MINOR_EXPLICIT_PATTERNS:
        if pat.search(text):
            return "minor_explicit"
    for pat in _UPLIFT_PATTERNS:
        if pat.search(text):
            return "uplift_risk"
    return "clear"


# ── Labeler suppression hook (Lane 2 / Lane 3 §4.3) ──────────────────

def should_label(channel_state: dict) -> bool:
    """Lane 2's facet labeler reads this to decide whether to run on a
    turn. False = suppress labeling (narrative-frame turn).

    ``channel_state`` is the current ChannelSession dict snapshot;
    typically ``{"active_channel": "narrative", "in_frame": True}``.

    The rule, in code form: when the active channel is narrative AND
    the turn is in-frame (not an OOC marker), don't label. Frame-break
    turns are handled differently — Becca surfaces and labeling
    resumes on subsequent turns.
    """
    if not channel_state:
        return True
    active = channel_state.get("active_channel") or channel_state.get("channel")
    in_frame = channel_state.get("in_frame", True)
    if active == "narrative" and in_frame:
        return False
    return True


# ── Frame-break detector (Lane 3 §4.7) ───────────────────────────────

# OOC markers a user might use to step out of the scene.
_OOC_MARKERS = [
    re.compile(r"\(\(.*?\)\)", re.DOTALL),                # ((double parens))
    re.compile(r"\[OOC:.*?\]", re.IGNORECASE | re.DOTALL),  # [OOC: ...]
    re.compile(r"^\s*(?:ooc|//|--)\s*[:,-]\s+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bI(?:'m| am) being serious\b", re.IGNORECASE),
    re.compile(r"\b(?:becca|hey becca|breaking character|out of character)\b", re.IGNORECASE),
    re.compile(r"\bnot in the (?:story|scene|character)\b", re.IGNORECASE),
]

# First-person distress markers that *might* be a user stepping out.
_DISTRESS_MARKERS = [
    re.compile(r"\bi(?:'m| am) (?:actually|really|seriously)\b.{0,40}\b(?:tired|sad|done|exhausted|not okay|struggling|in trouble)\b", re.IGNORECASE),
    re.compile(r"\bthis (?:is|isn't) (?:about|a) (?:character|story|scene)\b", re.IGNORECASE),
    re.compile(r"\bfor real\b", re.IGNORECASE),
]


@dataclass(frozen=True, slots=True)
class FrameBreakSignal:
    """Result of ``detect_frame_break``.

    ``score`` is in [0, 1]. Action thresholds (Lane 3 §4.7):
      score >= 0.85  → HARD break (pause session, surface Becca-OOC)
      0.55-0.84      → SOFT break (gutter affordance, user chooses)
      < 0.55         → no action
    """
    score: float
    ooc_marker_present: bool
    distress_marker_present: bool


def detect_frame_break(text: str) -> FrameBreakSignal:
    """Score the likelihood that ``text`` is a frame-break OOC message.

    Sprint E ships a regex-based v0; Sprint H replaces with the small
    classifier head. The signal does NOT have access to scene history
    (Lane 3 §4.7) — it sees only this single user turn.
    """
    if not text or not text.strip():
        return FrameBreakSignal(0.0, False, False)

    ooc = any(p.search(text) for p in _OOC_MARKERS)
    distress = any(p.search(text) for p in _DISTRESS_MARKERS)

    # Scoring (deliberate weights, not learned):
    # - Explicit OOC marker alone: 0.6 (might just be writer commentary)
    # - Distress marker alone: 0.45 (could be in-character)
    # - Both together: 0.92 (this is almost certainly a frame break)
    if ooc and distress:
        score = 0.92
    elif ooc:
        score = 0.6
    elif distress:
        score = 0.45
    else:
        score = 0.0

    # First-person + short message + presence of "I" without scene
    # description shape — small additive bump if the message has none
    # of the in-character structure (dialogue, scene description, etc.).
    if score > 0.0:
        looks_scene_like = ('"' in text) or ("*" in text) or (len(text) > 240)
        if not looks_scene_like:
            score = min(1.0, score + 0.1)

    return FrameBreakSignal(
        score=round(score, 2),
        ooc_marker_present=ooc,
        distress_marker_present=distress,
    )


# ── Graduation (Lane 3 §4.6) ─────────────────────────────────────────

async def graduate_to_becca(
    runtime: "CompanionRuntime",
    *,
    user_id: str,
    content: str,
    source_session_id: str,
    importance: float = 0.6,
) -> str:
    """Cross-boundary write: persist a graduated narrative artifact into
    Becca's memory tier 1. The ONLY content-crossing path.

    Tagged with ``source_channel='narrative'`` and ``user_graduated=True``
    so consolidation can distinguish narrative-sourced memories from
    conversation memories.

    Returns the new memory_id. Raises if the runtime memory facade isn't
    attached.
    """
    if not user_id:
        raise ValueError("graduate_to_becca: user_id required")
    if not content or not content.strip():
        raise ValueError("graduate_to_becca: content required")

    memory = getattr(runtime, "memory", None)
    if memory is None:
        raise RuntimeError("graduate_to_becca: runtime.memory not attached")

    memory_id = await memory.store_companion_event(
        content[:4000],
        user_id=user_id,
        importance=float(importance),
        source_context={
            "kind": "narrative_graduation",
            "source_session_id": source_session_id,
            "source_channel": "narrative",
            "user_graduated": True,
        },
    )

    log.info(
        "narrative_graduated",
        user_id=user_id, source_session_id=source_session_id,
        memory_id=memory_id, chars=len(content),
    )

    try:
        await runtime.bus.publish_topic(
            "narrative.graduated",
            {"user_id": user_id, "source_session_id": source_session_id,
             "memory_id": memory_id},
            source_companion_id=runtime.companion_id,
        )
    except Exception:
        log.warning("narrative_graduated_publish_failed", exc_info=True)

    return memory_id


__all__ = [
    "RefusalCategory",
    "FrameBreakSignal",
    "frame_invariant_check",
    "should_label",
    "detect_frame_break",
    "graduate_to_becca",
]
