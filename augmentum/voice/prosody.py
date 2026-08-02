"""Embedding Space Cartography & Dynamic Prosodic Steering for Kokoro TTS.

Maps Kokoro's 54-voice embedding space to discover perceptual axes (warmth,
breathiness, depth, energy, accent), then modulates style embeddings in
real-time based on text content to add natural prosodic variation.

Technique: latent space arithmetic on voice embeddings — established for
image generation (StyleGAN) but novel for TTS voice embeddings.

The key insight: voice embeddings encode more than speaker identity — they
encode prosodic tendencies (pitch range, breathiness, energy). By finding
*directions* in embedding space that correspond to these qualities, we can
shift a voice's delivery per-clause without changing its identity.

Shifts are intentionally small (0.05-0.2 magnitude) to preserve voice
identity while adding micro-expressiveness.
"""
from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING

import numpy as np

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.voice.kokoro_tts import KokoroTTS

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Semantic direction vectors (voice pairs that differ in a target quality)
# ---------------------------------------------------------------------------

# Each entry: (positive_voice, negative_voice, label)
# Direction = normalize(embed(positive) - embed(negative))
_AXIS_DEFINITIONS: list[tuple[str, str, str]] = [
    ("af_nicole", "af_heart",   "breathiness"),   # breathy ↔ clear
    ("af_bella",  "af_sarah",   "warmth"),         # warm ↔ bright
    ("am_adam",   "af_sky",     "depth"),           # deep ↔ light
    ("am_puck",   "af_nicole",  "energy"),          # energetic ↔ soft
    ("bf_emma",   "af_bella",   "accent_gb"),       # British ↔ American
]

# Steering magnitudes per text signal (intentionally small to preserve identity)
_QUESTION_MAGNITUDE = 0.10
_EXCLAMATION_MAGNITUDE = 0.08
_WHISPER_MAGNITUDE = 0.18
_SENTIMENT_MAGNITUDE = 0.08
_DIALOGUE_MAGNITUDE = 0.06

# Patterns for text analysis
_WHISPER_RE = re.compile(
    r"\[whisper\]|\(whisper(?:ing|ed)?\)|\(soft(?:ly)?\)|\(quiet(?:ly)?\)",
    re.IGNORECASE,
)
_DIALOGUE_RE = re.compile(r'["\u201c\u201d]')
_EMPHASIS_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__|\*(.+?)\*")


class ProsodyCartographer:
    """Maps and navigates Kokoro's voice embedding space.

    Discovers perceptual axes from voice pairs, caches direction vectors,
    and provides per-clause embedding modulation.

    Usage::

        cart = ProsodyCartographer.instance(kokoro_tts)
        steered = cart.steer(base_embedding, "What do you think?")
        # steered has subtle question-intonation shift
    """

    _instance: ProsodyCartographer | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._axes: dict[str, np.ndarray] = {}  # name → unit direction vector
        self._computed = False

    @classmethod
    def instance(cls, kokoro: KokoroTTS | None = None) -> ProsodyCartographer:
        """Get or create singleton. Call with kokoro on first access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        if kokoro and not cls._instance._computed:
            cls._instance._compute_axes(kokoro)
        return cls._instance

    def _compute_axes(self, kokoro: KokoroTTS) -> None:
        """Extract direction vectors from voice embedding pairs.

        Called once on first use, results cached for session lifetime.
        """
        if self._computed:
            return
        with self._lock:
            if self._computed:
                return

            available = set(kokoro.get_voices())
            for pos_name, neg_name, label in _AXIS_DEFINITIONS:
                if pos_name not in available or neg_name not in available:
                    log.debug("prosody_axis_skipped", label=label,
                              reason=f"voice not available: {pos_name} or {neg_name}")
                    continue
                try:
                    pos_embed = kokoro._kokoro.get_voice_style(pos_name)
                    neg_embed = kokoro._kokoro.get_voice_style(neg_name)
                    direction = pos_embed - neg_embed
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        self._axes[label] = direction / norm
                except Exception as exc:
                    log.debug("prosody_axis_error", label=label, error=str(exc))

            self._computed = True
            log.info("prosody_cartography_complete", axes=list(self._axes.keys()))

    @property
    def available_axes(self) -> list[str]:
        """List discovered perceptual axes."""
        return list(self._axes.keys())

    def steer(
        self,
        embedding: np.ndarray,
        text: str,
        *,
        manual_axes: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Modulate a voice embedding based on text content and manual overrides.

        Args:
            embedding: base voice embedding (numpy array from Kokoro)
            text: the text clause to be spoken
            manual_axes: optional dict of {axis_name: magnitude} for manual control
                        (e.g. from UI sliders: {"warmth": 0.3, "breathiness": -0.1})

        Returns:
            modulated embedding (same shape, small delta preserves voice identity)
        """
        if not self._computed or not self._axes:
            return embedding

        result = embedding.copy() if isinstance(embedding, np.ndarray) else embedding
        if not isinstance(result, np.ndarray):
            return result  # string voice name, can't steer

        # --- Automatic steering from text analysis ---

        # Question → shift toward higher energy (rising intonation proxy)
        if "energy" in self._axes and text.rstrip().endswith("?"):
            result = result + self._axes["energy"] * _QUESTION_MAGNITUDE

        # Exclamation → more energy + slight warmth
        if "!" in text:
            if "energy" in self._axes:
                result = result + self._axes["energy"] * _EXCLAMATION_MAGNITUDE
            if "warmth" in self._axes:
                result = result + self._axes["warmth"] * (_EXCLAMATION_MAGNITUDE * 0.5)

        # Whisper markers → breathiness
        if "breathiness" in self._axes and _WHISPER_RE.search(text):
            result = result + self._axes["breathiness"] * _WHISPER_MAGNITUDE

        # Quoted dialogue → slight depth shift (different "character")
        if "depth" in self._axes and _is_dialogue(text):
            result = result + self._axes["depth"] * _DIALOGUE_MAGNITUDE

        # Simple sentiment → warmth modulation
        if "warmth" in self._axes:
            sentiment = _quick_sentiment(text)
            if abs(sentiment) > 0.1:
                result = result + self._axes["warmth"] * (sentiment * _SENTIMENT_MAGNITUDE)

        # --- Manual slider overrides ---
        if manual_axes:
            for axis_name, magnitude in manual_axes.items():
                if axis_name in self._axes and abs(magnitude) > 0.01:
                    result = result + self._axes[axis_name] * magnitude

        return result

    def steer_text(
        self,
        kokoro: KokoroTTS,
        voice_spec: str,
        text: str,
        *,
        manual_axes: dict[str, float] | None = None,
    ) -> str | np.ndarray:
        """Convenience: resolve voice spec + apply steering in one call.

        For blend specs, resolves the blend first, then steers. For plain
        voice names, gets the embedding, steers, and returns the array.
        """
        resolved = kokoro._resolve_voice(voice_spec)

        # If it's a string (plain voice name), get the actual embedding
        if isinstance(resolved, str):
            try:
                resolved = kokoro._kokoro.get_voice_style(resolved)
            except Exception:
                return resolved  # can't steer, return as-is

        return self.steer(resolved, text, manual_axes=manual_axes)


# ---------------------------------------------------------------------------
# Text analysis helpers (lightweight, no ML dependencies)
# ---------------------------------------------------------------------------

def _is_dialogue(text: str) -> bool:
    """Detect if text is likely dialogue (quoted speech)."""
    quotes = _DIALOGUE_RE.findall(text)
    return len(quotes) >= 2  # opening + closing quote


# Positive/negative word sets for fast sentiment (no NLTK/transformers needed)
_POS_WORDS = frozenset({
    "love", "happy", "wonderful", "beautiful", "amazing", "great", "excellent",
    "fantastic", "joy", "delight", "warm", "kind", "gentle", "sweet", "lovely",
    "brilliant", "perfect", "excited", "grateful", "blessed", "smile", "laugh",
    "hope", "dream", "bright", "sunshine", "heart", "cherish", "adore", "treasure",
})
_NEG_WORDS = frozenset({
    "hate", "sad", "terrible", "horrible", "awful", "angry", "furious", "pain",
    "grief", "sorrow", "dark", "cold", "cruel", "harsh", "bitter", "despair",
    "fear", "dread", "rage", "suffer", "agony", "doom", "death", "cry", "scream",
    "destroy", "broken", "lost", "alone", "empty", "bleak", "miserable", "wretched",
})


def _quick_sentiment(text: str) -> float:
    """Fast keyword-based sentiment: -1.0 (negative) to +1.0 (positive).

    Not trying to be accurate — just enough to detect obvious emotional tone
    for subtle embedding shifts. False neutrals are fine; false positives
    are kept small by the magnitude scaling in steer().
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total  # -1 to +1, weighted by hit count


def split_prosodic_clauses(text: str) -> list[str]:
    """Split text into prosodic clauses for per-clause steering.

    Splits on natural prosodic boundaries (sentence ends, semicolons,
    em-dashes, long commas) rather than just sentence boundaries.
    Each clause gets its own embedding modulation.

    Returns list of non-empty clause strings.
    """
    # Split on major prosodic boundaries
    clauses = re.split(
        r'(?<=[.!?])\s+'           # sentence boundaries
        r'|(?<=[;:])\s+'           # semicolons, colons
        r'|(?<=—)\s*'             # em-dashes
        r'|\s*—\s*'              # em-dashes (both sides)
        r'|(?<=,)\s+(?=[A-Z])',   # comma before capitalized word (clause break)
        text,
    )
    return [c.strip() for c in clauses if c.strip()]
