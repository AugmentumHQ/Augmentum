"""Emotion extraction from RP text for TTS prosody control.

Pure-function module — no external dependencies beyond ``re``.
Extracts emotional cues from asterisked/hyphenated RP markers and
entity emotional state, producing output in two formats:

1. **Instruct string** (for Qwen3-TTS ``instruct`` parameter):
   ``"speak in a soft whisper"``
2. **Inline tags** (for Fish Speech / providers that use text-embedded tags):
   ``"[whisper]Hello there[/whisper]"``

The voice pipeline picks the right format based on the TTS provider.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# RP action keyword → emotion mapping
# ---------------------------------------------------------------------------
# Order matters: first match wins.
# Each entry maps to: (instruct_string, fish_tag)
# - instruct_string: Qwen3-TTS style natural-language instruction
# - fish_tag: Fish Speech inline tag name

_ACTION_MAP: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"whisper", re.IGNORECASE), "speak in a soft whisper", "whisper"),
    (re.compile(r"murmur", re.IGNORECASE), "speak in a soft murmur", "whisper in small voice"),
    (re.compile(r"sigh", re.IGNORECASE), "speak with a weary sigh", "tired sigh"),
    (re.compile(r"sob|cry|weep|tear", re.IGNORECASE), "speak through tears", "crying"),
    (re.compile(r"laugh|chuckle|giggle", re.IGNORECASE), "speak with amusement", "laughing"),
    (re.compile(r"shout|yell|scream", re.IGNORECASE), "speak loudly and forcefully", "shouting"),
    (re.compile(r"trembl|shiver|shake", re.IGNORECASE), "speak with a trembling, fearful voice", "scared trembling voice"),
    (re.compile(r"growl|snarl", re.IGNORECASE), "speak with a low, threatening tone", "angry growl"),
    (re.compile(r"hiss", re.IGNORECASE), "speak in a sharp hiss", "sharp hiss"),
    (re.compile(r"stammer|stutter", re.IGNORECASE), "speak hesitantly with a stammer", "nervous stuttering"),
    (re.compile(r"gasp", re.IGNORECASE), "speak breathlessly", "breathless"),
    (re.compile(r"sing", re.IGNORECASE), "speak in a melodic, sing-song voice", "singing melodically"),
    (re.compile(r"purr", re.IGNORECASE), "speak in a low, sultry purr", "sultry low voice"),
    (re.compile(r"squeal", re.IGNORECASE), "speak with a high-pitched squeal", "excited high pitch"),
    (re.compile(r"groan|moan", re.IGNORECASE), "speak with a tired groan", "tired groan"),
    (re.compile(r"sneer|mock", re.IGNORECASE), "speak with contempt", "contemptuous"),
    (re.compile(r"plead|beg", re.IGNORECASE), "speak with desperate pleading", "desperate pleading"),
]

# Regex to extract asterisked (*action*) and hyphenated (-action-) RP markers
_RP_MARKER_RE = re.compile(r"\*([^*]+)\*|-([^-]+)-")

# ---------------------------------------------------------------------------
# Entity emotional state → emotion mapping
# ---------------------------------------------------------------------------
# Substring matching: "slightly sad" matches "sad".

_ENTITY_STATE_MAP: list[tuple[str, str, str]] = [
    ("happy", "speak happily and warmly", "happy"),
    ("joy", "speak happily and warmly", "joyful"),
    ("sad", "speak with sadness", "sad"),
    ("angry", "speak with anger", "angry"),
    ("furious", "speak with intense fury", "furious"),
    ("calm", "speak calmly and steadily", "calm"),
    ("anxious", "speak with nervous anxiety", "anxious"),
    ("afraid", "speak with fear", "scared"),
    ("scared", "speak with fear", "scared"),
    ("excited", "speak with excitement and energy", "excited"),
    ("tender", "speak tenderly and gently", "tender gentle voice"),
    ("loving", "speak with warmth and affection", "warm affectionate"),
    ("bitter", "speak with bitterness", "bitter"),
    ("tired", "speak with exhaustion", "exhausted"),
    ("confused", "speak with uncertainty and confusion", "confused"),
    ("amused", "speak with amusement", "amused"),
    ("annoyed", "speak with mild irritation", "annoyed"),
    ("worried", "speak with concern and worry", "worried"),
    ("surprised", "speak with surprise", "surprised"),
    ("embarrassed", "speak with embarrassment", "embarrassed"),
    ("shy", "speak quietly and shyly", "shy quiet voice"),
    ("confident", "speak with bold confidence", "confident"),
    ("proud", "speak with pride", "proud"),
    ("disgusted", "speak with disgust", "disgusted"),
    ("hopeful", "speak with gentle hopefulness", "hopeful"),
    ("melancholy", "speak with quiet melancholy", "melancholy"),
    ("playful", "speak playfully and lightly", "playful"),
]

# States that should produce no instruct (normal speech)
_NEUTRAL_STATES = {"neutral", "normal", "calm", "composed", "steady"}


def extract_emotion_instruct(
    raw_text: str,
    entity_emotional_state: str = "",
) -> str:
    """Extract an emotion instruct string from RP text and/or entity state.

    Priority:
    1. RP markers in the text (``*sighs*``, ``-whispers-``) — per-sentence
    2. Entity emotional state (``"sad"``, ``"slightly anxious"``) — per-turn fallback

    Returns a short natural-language instruction or ``""`` if no signal.
    Used by Qwen3-TTS (instruct parameter) and as fallback voice style.
    """
    # 1) Try RP markers first
    for match in _RP_MARKER_RE.finditer(raw_text):
        action_text = match.group(1) or match.group(2) or ""
        for pattern, instruct, _tag in _ACTION_MAP:
            if pattern.search(action_text):
                return instruct

    # 2) Fall back to entity emotional state
    if entity_emotional_state:
        state_lower = entity_emotional_state.lower().strip()
        if state_lower and state_lower not in _NEUTRAL_STATES:
            for keyword, instruct, _tag in _ENTITY_STATE_MAP:
                if keyword in state_lower:
                    return instruct

    return ""


def extract_emotion_tag(
    raw_text: str,
    entity_emotional_state: str = "",
) -> str:
    """Extract a Fish Speech emotion tag name from RP text and/or entity state.

    Same priority as ``extract_emotion_instruct`` but returns a tag name
    for Fish Speech's inline syntax: ``[tag]text[/tag]``.

    Returns a tag string or ``""`` if no signal.
    """
    # 1) Try RP markers first
    for match in _RP_MARKER_RE.finditer(raw_text):
        action_text = match.group(1) or match.group(2) or ""
        for pattern, _instruct, tag in _ACTION_MAP:
            if pattern.search(action_text):
                return tag

    # 2) Fall back to entity emotional state
    if entity_emotional_state:
        state_lower = entity_emotional_state.lower().strip()
        if state_lower and state_lower not in _NEUTRAL_STATES:
            for keyword, _instruct, tag in _ENTITY_STATE_MAP:
                if keyword in state_lower:
                    return tag

    return ""


def wrap_with_emotion_tag(text: str, tag: str) -> str:
    """Prepend Fish Speech inline emotion tag to text.

    Fish Speech uses parenthesized tags at the start of text:
    ``wrap_with_emotion_tag("Hello!", "excited")`` → ``"(excited)Hello!"``

    Tags apply to the text that follows them — no closing tag needed.
    Multiple tags can be chained: ``"(excited)(laughing)Hello!"``

    If tag is empty, returns the text unchanged.
    """
    if not tag:
        return text
    return f"({tag}){text}"


# ---------------------------------------------------------------------------
# Chatterbox Turbo paralinguistic tags
# ---------------------------------------------------------------------------
# Turbo supports inline tags: [laugh], [cough], [chuckle].
# These are embedded directly in the text and pronounced by the model.
# We convert RP markers to inline Turbo tags before sending to the provider.

_TURBO_TAG_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"chuckle", re.IGNORECASE), "[chuckle]"),  # specific match first
    (re.compile(r"laugh|giggle", re.IGNORECASE), "[laugh]"),
    (re.compile(r"cough", re.IGNORECASE), "[cough]"),
]


def inject_turbo_tags(raw_text: str) -> str:
    """Convert RP markers to Chatterbox Turbo inline paralinguistic tags.

    ``*laughs softly* Hello there`` → ``[laugh] Hello there``
    ``She said *coughs* excuse me`` → ``She said [cough] excuse me``

    Turbo supports: ``[laugh]``, ``[cough]``, ``[chuckle]``
    Unrecognized markers are stripped (they'd confuse TTS anyway).
    """
    def _replace_marker(match: re.Match) -> str:
        action_text = match.group(1) or match.group(2) or ""
        for pattern, tag in _TURBO_TAG_MAP:
            if pattern.search(action_text):
                return tag
        return ""  # strip unrecognized RP markers

    return _RP_MARKER_RE.sub(_replace_marker, raw_text).strip()


def is_turbo_provider(provider_id: str) -> bool:
    """Check if a provider ID is Chatterbox Turbo."""
    return provider_id == "chatterbox-turbo"
