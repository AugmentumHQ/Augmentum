"""Map the companion's affect → a CSM emotion tag for her fine-tuned voice.

The fine-tuned CSM voices (trained by services/sesame-csm/training) respond to a
leading ``(emotion)`` tag in the input text — the EARS label set. Augmentum
already tracks the companion's affect (``runtime._last_affect_tag``, the PAD
behaviour system). This bridges the two: her mood drives how she *sounds*, not
just what she says.

Cross-modal by design — one affect state, surfaced in voice. Gated by the
``companion_csm_emotion_tag`` setting (default off until tuned by ear) and only
applied for CSM voices.

CAVEAT (why this is opt-in): ``_last_affect_tag`` is the last *published change*
and persists, so a naive read can leave her stuck in a stale mood. Callers
should pass a recency-gated affect (see ``emotion_for_affect`` docstring) and we
tune the mapping by listening once a fine-tuned voice is live.
"""
from __future__ import annotations

# Companion affect tag (left) → CSM emotion label (right). The right-hand side
# MUST be a label the voice was trained on (EARS set). Unmapped affect → "" =
# no tag = neutral delivery (safe default).
_AFFECT_TO_CSM: dict[str, str] = {
    # warm / positive-low-arousal
    "content": "contentment", "contentment": "contentment", "settled": "",
    "calm": "serenity", "serene": "serenity", "serenity": "serenity",
    "peaceful": "serenity", "relaxed": "serenity",
    "affectionate": "adoration", "fond": "adoration", "tender": "adoration",
    "warm": "adoration", "loving": "adoration", "adoration": "adoration",
    # positive-high-arousal
    "happy": "amusement", "playful": "amusement", "amused": "amusement",
    "amusement": "amusement", "cheerful": "amusement", "delighted": "amusement",
    "excited": "amazement", "thrilled": "amazement", "amazed": "amazement",
    "awed": "amazement", "amazement": "amazement", "wonder": "amazement",
    "proud": "pride", "pride": "pride",
    "relieved": "relief", "relief": "relief",
    "curious": "interest", "interested": "interest", "intrigued": "interest",
    "interest": "interest", "engaged": "interest",
    # negative
    "sad": "sadness", "down": "sadness", "melancholy": "sadness",
    "blue": "sadness", "wistful": "sadness", "sadness": "sadness",
    "anxious": "fear", "worried": "fear", "nervous": "fear",
    "afraid": "fear", "scared": "fear", "fear": "fear", "uneasy": "fear",
    "frustrated": "anger", "annoyed": "anger", "irritated": "anger",
    "angry": "anger", "anger": "anger", "cross": "anger",
    "disappointed": "disappointment", "disappointment": "disappointment",
    "confused": "confusion", "confusion": "confusion", "puzzled": "confusion",
    "distressed": "distress", "distress": "distress", "upset": "distress",
}

# CSM labels the harness's voices are trained on — guard against drift.
_VALID_CSM = {
    "contentment", "serenity", "adoration", "amusement", "amazement", "pride",
    "relief", "interest", "sadness", "fear", "anger", "disappointment",
    "confusion", "distress", "neutral",
}


def emotion_for_affect(affect_tag: str | None) -> str:
    """Companion affect tag → CSM emotion label, or "" for neutral delivery.

    Pass a *recency-gated* affect tag (e.g. only if it changed within the last
    N seconds) — this function does no recency check itself; it only maps. ""
    / unknown / equilibrium tags map to "" so she speaks plainly.
    """
    t = (affect_tag or "").strip().lower()
    if not t:
        return ""
    csm = _AFFECT_TO_CSM.get(t, "")
    return csm if csm in _VALID_CSM else ""


# Companion affect tag → natural-language control descriptor for OpenAI-omni
# style TTS (Higgs Audio v3 via sglang-omni, and anything that interprets a
# leading parenthetical style cue). Higgs takes free-form descriptors, so this
# is a CURATED, companion-appropriate vocabulary rather than a fixed label set:
# negatives are rendered as restrained/warm equivalents (a companion sounds
# *concerned*, not hostile, when her mood dips) so authentic affect never reads
# as anger AT the user. Unmapped affect → "" = plain delivery.
_AFFECT_TO_OMNI: dict[str, str] = {
    # warm / positive-low-arousal
    "content": "warm", "contentment": "warm", "settled": "gentle",
    "calm": "calm", "serene": "calm", "serenity": "calm",
    "peaceful": "gentle", "relaxed": "relaxed",
    "affectionate": "tender", "fond": "tender", "tender": "tender",
    "warm": "warm", "loving": "tender", "adoration": "tender",
    # positive-high-arousal
    "happy": "cheerful", "cheerful": "cheerful", "delighted": "delighted",
    "playful": "playful", "amused": "playful", "amusement": "playful",
    "excited": "excited", "thrilled": "excited", "amazed": "amazed",
    "awed": "amazed", "amazement": "amazed", "wonder": "amazed",
    "proud": "warm", "pride": "warm",
    "relieved": "relieved", "relief": "relieved",
    "curious": "curious", "interested": "curious", "intrigued": "curious",
    "interest": "curious", "engaged": "curious",
    # negative — rendered restrained/warm, never hostile toward the user
    "sad": "gentle", "down": "gentle", "melancholy": "wistful",
    "blue": "wistful", "wistful": "wistful", "sadness": "gentle",
    "anxious": "soft", "worried": "soft", "nervous": "soft",
    "afraid": "soft", "scared": "soft", "fear": "soft", "uneasy": "soft",
    "frustrated": "measured", "annoyed": "measured", "irritated": "measured",
    "angry": "serious", "anger": "serious", "cross": "measured",
    "disappointed": "gentle", "disappointment": "gentle",
    "confused": "thoughtful", "confusion": "thoughtful", "puzzled": "thoughtful",
    "distressed": "concerned", "distress": "concerned", "upset": "concerned",
}


def omni_descriptor_for_affect(affect_tag: str | None) -> str:
    """Companion affect tag → a natural control descriptor for omni/Higgs TTS.

    Like :func:`emotion_for_affect` but targets free-form parenthetical style
    cues (e.g. ``(warm)``, ``(excited)``) instead of CSM's fixed EARS labels.
    Pass a *recency-gated* affect tag; unknown/empty → "" (plain delivery).
    """
    t = (affect_tag or "").strip().lower()
    if not t:
        return ""
    return _AFFECT_TO_OMNI.get(t, "")
