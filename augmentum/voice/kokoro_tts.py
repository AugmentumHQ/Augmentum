"""In-process Kokoro TTS via kokoro-onnx.

Generates speech directly in the Augmentum process — no sidecar container,
no HTTP overhead.  Uses the ONNX Runtime backend for CPU inference.

Features:
  - Streaming: async generator yields audio chunks per text segment
  - Voice mixing: weighted blend of voice embeddings via numpy
  - Format encoding: raw PCM → MP3/WAV via ffmpeg (already in Docker)
  - Thread-safe: ONNX Runtime handles concurrent inference

Models (~88 MB INT8):
  https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0

Requires: pip install kokoro-onnx
"""

from __future__ import annotations

import asyncio
import os
import struct
import threading
from typing import AsyncGenerator

import numpy as np

from augmentum.utils.logging import get_logger
from augmentum.utils.paths import resolve_model_dir

log = get_logger(__name__)

# Resolves to /home/augmentum/.kokoro in the Docker container (Dockerfile
# bakes weights there). Native installs get a platform-appropriate user
# cache dir — see augmentum/utils/paths.py.
_DEFAULT_MODEL_DIR = resolve_model_dir("kokoro")
_SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# Voice metadata — grades from hexgrad/Kokoro-82M VOICES.md, descriptions
# from community consensus (Reddit, HuggingFace, GitHub discussions).
# ---------------------------------------------------------------------------
VOICE_META: dict[str, dict] = {
    # American English — Female
    "af_heart":   {"grade": "A",  "gender": "F", "lang": "en-US", "desc": "Flagship composite — balanced, clear, versatile"},
    "af_bella":   {"grade": "A-", "gender": "F", "lang": "en-US", "desc": "Warm narrator — audiobook favorite, artifact-free"},
    "af_nicole":  {"grade": "B-", "gender": "F", "lang": "en-US", "desc": "Breathy ASMR — intimate, whisper quality"},
    "af_aoede":   {"grade": "C+", "gender": "F", "lang": "en-US", "desc": "Clear and steady female"},
    "af_kore":    {"grade": "C+", "gender": "F", "lang": "en-US", "desc": "Bright and expressive female"},
    "af_sarah":   {"grade": "C+", "gender": "F", "lang": "en-US", "desc": "Friendly, younger tone — tutorials and education"},
    "af_alloy":   {"grade": "C",  "gender": "F", "lang": "en-US", "desc": "Neutral female"},
    "af_nova":    {"grade": "C",  "gender": "F", "lang": "en-US", "desc": "Neutral female"},
    "af_sky":     {"grade": "C-", "gender": "F", "lang": "en-US", "desc": "Light, airy female"},
    "af_jessica": {"grade": "D",  "gender": "F", "lang": "en-US", "desc": "Limited training data"},
    "af_river":   {"grade": "D",  "gender": "F", "lang": "en-US", "desc": "Limited training data"},
    # American English — Male
    "am_fenrir":  {"grade": "C+", "gender": "M", "lang": "en-US", "desc": "Strong, steady male"},
    "am_michael": {"grade": "C+", "gender": "M", "lang": "en-US", "desc": "Warm, approachable — best male for conversation"},
    "am_puck":    {"grade": "C+", "gender": "M", "lang": "en-US", "desc": "Energetic male"},
    "am_adam":    {"grade": "F+", "gender": "M", "lang": "en-US", "desc": "Deep but lowest quality — use in blends only"},
    "am_echo":    {"grade": "D",  "gender": "M", "lang": "en-US", "desc": "Limited training data"},
    "am_eric":    {"grade": "D",  "gender": "M", "lang": "en-US", "desc": "Limited training data"},
    "am_liam":    {"grade": "D",  "gender": "M", "lang": "en-US", "desc": "Limited training data"},
    "am_onyx":    {"grade": "D",  "gender": "M", "lang": "en-US", "desc": "Limited training data"},
    "am_santa":   {"grade": "D-", "gender": "M", "lang": "en-US", "desc": "Novelty voice"},
    # British English
    "bf_emma":      {"grade": "B-", "gender": "F", "lang": "en-GB", "desc": "Best British female — refined, clear"},
    "bf_isabella":  {"grade": "C",  "gender": "F", "lang": "en-GB", "desc": "British female"},
    "bf_alice":     {"grade": "D",  "gender": "F", "lang": "en-GB", "desc": "Limited training data"},
    "bf_lily":      {"grade": "D",  "gender": "F", "lang": "en-GB", "desc": "Limited training data"},
    "bm_george":    {"grade": "C",  "gender": "M", "lang": "en-GB", "desc": "British male — good for blending with am_michael"},
    "bm_fable":     {"grade": "C",  "gender": "M", "lang": "en-GB", "desc": "British male"},
    "bm_lewis":     {"grade": "D+", "gender": "M", "lang": "en-GB", "desc": "Limited training data"},
    "bm_daniel":    {"grade": "D",  "gender": "M", "lang": "en-GB", "desc": "Limited training data"},
    # French
    "ff_siwis":     {"grade": "B-", "gender": "F", "lang": "fr", "desc": "Best French voice — SIWIS corpus trained"},
    # Japanese
    "jf_alpha":     {"grade": "C+", "gender": "F", "lang": "ja", "desc": "Best Japanese female"},
    "jf_gongitsune": {"grade": "C", "gender": "F", "lang": "ja", "desc": "Japanese female"},
    "jf_tebukuro":  {"grade": "C",  "gender": "F", "lang": "ja", "desc": "Japanese female"},
    "jf_nezumi":    {"grade": "C-", "gender": "F", "lang": "ja", "desc": "Japanese female"},
    "jm_kumo":      {"grade": "C-", "gender": "M", "lang": "ja", "desc": "Japanese male"},
    # Hindi
    "hf_alpha":     {"grade": "C",  "gender": "F", "lang": "hi", "desc": "Hindi female"},
    "hf_beta":      {"grade": "C",  "gender": "F", "lang": "hi", "desc": "Hindi female"},
    "hm_omega":     {"grade": "C",  "gender": "M", "lang": "hi", "desc": "Hindi male"},
    "hm_psi":       {"grade": "C",  "gender": "M", "lang": "hi", "desc": "Hindi male"},
    # Italian
    "if_sara":      {"grade": "C",  "gender": "F", "lang": "it", "desc": "Italian female"},
    "im_nicola":    {"grade": "C",  "gender": "M", "lang": "it", "desc": "Italian male"},
    # Mandarin Chinese
    "zf_xiaobei":   {"grade": "D",  "gender": "F", "lang": "zh", "desc": "Chinese female"},
    "zf_xiaoni":    {"grade": "D",  "gender": "F", "lang": "zh", "desc": "Chinese female"},
    "zf_xiaoxiao":  {"grade": "D",  "gender": "F", "lang": "zh", "desc": "Chinese female"},
    "zf_xiaoyi":    {"grade": "D",  "gender": "F", "lang": "zh", "desc": "Chinese female"},
    "zm_yunjian":   {"grade": "D",  "gender": "M", "lang": "zh", "desc": "Chinese male"},
    "zm_yunxi":     {"grade": "D",  "gender": "M", "lang": "zh", "desc": "Chinese male"},
    "zm_yunxia":    {"grade": "D",  "gender": "M", "lang": "zh", "desc": "Chinese male"},
    "zm_yunyang":   {"grade": "D",  "gender": "M", "lang": "zh", "desc": "Chinese male"},
    # Spanish
    "ef_dora":      {"grade": "C",  "gender": "F", "lang": "es", "desc": "Spanish female"},
    "em_alex":      {"grade": "C",  "gender": "M", "lang": "es", "desc": "Spanish male"},
    "em_santa":     {"grade": "D",  "gender": "M", "lang": "es", "desc": "Novelty voice"},
    # Brazilian Portuguese
    "pf_dora":      {"grade": "C",  "gender": "F", "lang": "pt-BR", "desc": "Portuguese female"},
    "pm_alex":      {"grade": "C",  "gender": "M", "lang": "pt-BR", "desc": "Portuguese male"},
    "pm_santa":     {"grade": "D",  "gender": "M", "lang": "pt-BR", "desc": "Novelty voice"},
}

# Grades that qualify as "recommended" tier in the UI
_RECOMMENDED_GRADES = {"A", "A-", "B-", "C+"}

# Pre-made voice blends from community recommendations
RECOMMENDED_BLENDS: list[dict] = [
    {"name": "Warm Narrator",      "spec": "af_bella*0.6+af_heart*0.4",     "desc": "Rich and professional — audiobooks, marketing",           "gender": "F", "lang": "en-US"},
    {"name": "Clear & Friendly",   "spec": "af_sarah*0.5+af_bella*0.5",     "desc": "Approachable and clear — tutorials, education",           "gender": "F", "lang": "en-US"},
    {"name": "British Blend",      "spec": "bf_emma*0.6+bf_isabella*0.4",   "desc": "Refined British accent — narration, documentaries",       "gender": "F", "lang": "en-GB"},
    {"name": "Gentle ASMR",        "spec": "af_nicole*0.7+af_sky*0.3",      "desc": "Soft and intimate — meditation, sleep stories",           "gender": "F", "lang": "en-US"},
    {"name": "Deep Gentleman",     "spec": "am_michael*0.5+bm_george*0.5",  "desc": "Cross-accent male warmth — narration, conversation",     "gender": "M", "lang": "en"},
    {"name": "Androgynous",        "spec": "af_sarah*0.6+am_adam*0.4",      "desc": "Gender-neutral — unique character, creative projects",    "gender": "N", "lang": "en-US"},
]


# Voice prefix → kokoro lang code mapping
_PREFIX_TO_LANG: dict[str, str] = {
    "a": "en-us",   # American English
    "b": "en-gb",   # British English
    "j": "ja",      # Japanese
    "z": "cmn",     # Mandarin Chinese
    "f": "fr-fr",   # French
    "h": "hi",      # Hindi
    "i": "it",      # Italian
    "e": "es",      # Spanish
    "p": "pt-br",   # Brazilian Portuguese
}


def _voice_lang(voice: str) -> str:
    """Derive the kokoro lang code from a voice name or blend spec.

    Voice names follow the pattern ``{lang}{gender}_{name}`` — the first
    character encodes the language.  For blends (``af_bella*0.6+bf_emma*0.4``)
    the first component's language wins.
    """
    # Extract the first voice name from a blend spec
    first = voice.split("+")[0].split("*")[0].split("(")[0].strip()
    if first:
        return _PREFIX_TO_LANG.get(first[0], "en-us")
    return "en-us"


class KokoroTTS:
    """In-process Kokoro TTS engine.

    Loads the ONNX model once (class-level) and reuses across all requests.
    Voice mixing is done via numpy weighted interpolation of style embeddings.

    Usage::

        tts = KokoroTTS.instance()
        async for chunk in tts.stream_speech("Hello!", voice="af_heart"):
            websocket.send_bytes(chunk)
    """

    _instance: KokoroTTS | None = None
    _instance_quality: str = ""  # quality setting when instance was created
    _lock = threading.Lock()

    def __init__(self, model_dir: str = ""):
        self._model_dir = model_dir or _DEFAULT_MODEL_DIR
        self._kokoro = None  # kokoro_onnx.Kokoro instance
        self._loaded = False
        self._quality = ""  # actual quality loaded ("int8" or "fp16")
        self._requested_quality = ""  # what the user asked for
        # Cache blended voices for the session to avoid re-computing
        self._voice_cache: dict[str, np.ndarray] = {}
        self._voice_cache_lock = threading.Lock()

    @classmethod
    def instance(cls, model_dir: str = "") -> KokoroTTS:
        """Get or create the singleton TTS instance.

        If the user changes tts_kokoro_quality in settings, the singleton
        is invalidated and a new instance loads the correct model variant.
        """
        from augmentum.config import settings
        current_quality = getattr(settings, "tts_kokoro_quality", "int8")

        # Invalidate if quality setting changed since last load
        if (cls._instance is not None
                and cls._instance._loaded
                and cls._instance_quality
                and cls._instance_quality != current_quality):
            log.info("kokoro_quality_changed",
                     old=cls._instance_quality, new=current_quality)
            cls._instance = None

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_dir=model_dir)
                    cls._instance_quality = current_quality
        return cls._instance

    @classmethod
    def configure(cls, model_dir: str = "") -> None:
        """Set model directory before loading.  Call during app startup."""
        inst = cls.instance(model_dir=model_dir)
        inst._model_dir = model_dir or _DEFAULT_MODEL_DIR

    def load_model(self, quality: str = "") -> None:
        """Load the Kokoro ONNX model.  Thread-safe, idempotent.

        Args:
            quality: "int8" (CPU, fast, 88MB) or "fp16" (GPU, better quality, 169MB).
                     Empty string auto-detects from config.
        """
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            voices_path = os.path.join(self._model_dir, "voices-v1.0.bin")

            # Resolve quality setting
            if not quality:
                from augmentum.config import settings
                quality = getattr(settings, "tts_kokoro_quality", "int8")

            # Model selection priority based on quality setting:
            # fp16 → GPU-accelerated, better audio quality, 169MB
            # int8 → CPU-optimized, fast, 88MB (default)
            _requested_quality = quality
            if quality == "fp16":
                model_path = os.path.join(self._model_dir, "kokoro-v1.0.fp16.onnx")
                if not os.path.exists(model_path):
                    log.warning("kokoro_fp16_not_found", fallback="int8")
                    model_path = os.path.join(self._model_dir, "kokoro-v1.0.int8.onnx")
                    quality = "int8"  # track actual loaded quality
            else:
                model_path = os.path.join(self._model_dir, "kokoro-v1.0.int8.onnx")
                if not os.path.exists(model_path):
                    # Try fp16 or fp32 as fallback
                    for alt in ("kokoro-v1.0.fp16.onnx", "kokoro-v1.0.onnx"):
                        alt_path = os.path.join(self._model_dir, alt)
                        if os.path.exists(alt_path):
                            model_path = alt_path
                            quality = "fp16" if "fp16" in alt else "fp32"
                            break

            if not os.path.exists(model_path) or not os.path.exists(voices_path):
                log.warning("kokoro_models_not_found", path=self._model_dir,
                            model=model_path, voices=voices_path)
                return

            try:
                from kokoro_onnx import Kokoro

                self._kokoro = Kokoro(model_path, voices_path)
                self._loaded = True
                self._quality = quality
                self._requested_quality = _requested_quality
                KokoroTTS._instance_quality = quality  # track for hot-swap detection
                log.info("kokoro_tts_loaded",
                         model=os.path.basename(model_path),
                         quality=quality,
                         voices=len(self._kokoro.get_voices()))
            except ImportError:
                log.warning("kokoro_onnx_not_available",
                            note="pip install kokoro-onnx for built-in TTS")
            except Exception as exc:
                log.warning("kokoro_tts_load_error", error=str(exc))

    @property
    def is_available(self) -> bool:
        return self._loaded and self._kokoro is not None

    @classmethod
    def status(cls) -> dict:
        """Return Kokoro runtime status for the settings UI."""
        inst = cls._instance
        if inst is None or not inst._loaded:
            return {"loaded": False}
        return {
            "loaded": True,
            "requested": inst._requested_quality,
            "actual": inst._quality,
            "fallback": inst._requested_quality != inst._quality,
        }

    def get_voices(self) -> list[str]:
        """List available voice names."""
        if not self.is_available:
            return []
        return self._kokoro.get_voices()

    def _resolve_voice(self, voice: str) -> str | np.ndarray:
        """Resolve a voice spec to a name or blended numpy array.

        Supports:
          - Simple name: ``"af_heart"``
          - Equal blend: ``"af_heart+af_sky"`` (equal weights)
          - Weighted blend: ``"af_heart*0.7+af_sky*0.3"``
          - Kokoro-style weighted: ``"af_heart(2)+af_sky(1)"``
        """
        if not voice or not self.is_available:
            return "af_heart"  # default voice

        # Check blend cache (lock protects concurrent read/write)
        with self._voice_cache_lock:
            if voice in self._voice_cache:
                return self._voice_cache[voice]

        # Load saved voice walk embedding from disk
        if voice.startswith("walk:"):
            walk_name = voice[5:]
            try:
                import os
                from augmentum.config import settings
                walk_dir = os.path.join(settings.data_dir or "/data", "voice_walks")
                walk_path = os.path.join(walk_dir, f"{walk_name}.npy")
                if os.path.exists(walk_path):
                    loaded = np.load(walk_path)
                    with self._voice_cache_lock:
                        self._voice_cache[voice] = loaded
                    return loaded
            except Exception as exc:
                log.warning("voice_walk_load_error", name=walk_name, error=str(exc))
            return "af_heart"  # fallback

        # Parse blend spec if it contains "+"
        if "+" in voice:
            try:
                parts = voice.split("+")
                if len(parts) < 2:
                    return voice

                blended = None
                total_weight = 0.0
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue

                    # Parse weight: "name*0.7" or "name(2)" or just "name"
                    w = 1.0
                    name = part
                    if "*" in part:
                        name, weight_str = part.rsplit("*", 1)
                        name = name.strip()
                        w = float(weight_str.strip())
                    elif "(" in part and part.endswith(")"):
                        name = part[:part.index("(")]
                        w = float(part[part.index("(") + 1:-1])

                    style = self._kokoro.get_voice_style(name)
                    if blended is None:
                        blended = style * w
                    else:
                        blended = np.add(blended, style * w)
                    total_weight += w

                # Normalize to unit weights if total > 1
                if blended is not None and total_weight > 1.0:
                    blended = blended / total_weight

                if blended is not None:
                    with self._voice_cache_lock:
                        self._voice_cache[voice] = blended
                    return blended
            except Exception as exc:
                log.warning("kokoro_blend_error", voice=voice, error=str(exc))

        return voice  # plain voice name

    async def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> bytes:
        """Generate speech for a text string.  Returns encoded audio bytes."""
        if not self.is_available:
            log.warning("kokoro_generate_unavailable",
                        note="Model not loaded — no audio will be produced")
            return b""

        resolved_voice = self._resolve_voice(voice)
        lang = _voice_lang(voice)

        # Prosodic steering: modulate embedding based on text content.
        # Walked voices are already optimized points in Kokoro's style space;
        # additional per-sentence steering can move them away from the stable
        # match and make alignment artifacts more likely.
        if not voice.startswith("walk:"):
            resolved_voice = _apply_prosodic_steering(self, resolved_voice, text)

        # Run inference in thread to avoid blocking the event loop
        samples, sr = await asyncio.to_thread(
            self._kokoro.create,
            text,
            voice=resolved_voice,
            speed=speed,
            lang=lang,
        )

        # Harmonic bandwidth extension: 24kHz → 48kHz
        samples, sr = await asyncio.to_thread(_apply_hbe, samples, sr)

        # Encode to target format
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return await _encode_audio(pcm16, sr, response_format)

    async def stream_speech(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        response_format: str = "mp3",
        stream_chunks: bool = False,
    ) -> AsyncGenerator[bytes, None]:
        """Stream speech audio chunks for a text string.

        WAV emission has two strategies, chosen by the caller to match
        what the client at the other end of the response can decode.
        The pre-existing implementation embedded a full ``_pcm_to_wav``
        header in every chunk — that produced a stream of concatenated
        mini-WAV files. Clients survived only because the streaming
        reader treats subsequent header bytes as PCM (44 bytes = ~0.9 ms
        at 24 kHz, sub-perceptual click) and because chunks were large
        enough that short replies were a single chunk anyway. On longer
        replies the blob fallback (iOS Audio element) gets a glued
        multi-WAV body and refuses to play. See PocketTTS.stream_speech
        for the original write-up of the same diagnosis.

        * ``stream_chunks=True`` — emit ONE sentinel-size WAV header up
          front, then raw PCM int16 frames from each chunk as the model
          produces them. First byte ~100 ms after warmup. Only safe on
          clients that consume the response as a true byte stream
          (Android Chrome / desktop via ReadableStream.getReader).
        * ``stream_chunks=False`` — accumulate all PCM across every
          chunk, emit one buffered WAV with real chunk sizes at the
          end. Loses streaming latency but plays in every client path
          including iOS Audio elements. Safe default.

        Non-WAV formats (mp3, opus, etc.) go through the original per-
        chunk encode-and-yield path — those container formats are
        designed to concatenate naturally (each MP3 frame is self-
        contained), so the chunking strategy doesn't affect playability.
        """
        if not self.is_available:
            log.warning("kokoro_stream_unavailable",
                        note="Model not loaded — no audio will be streamed")
            return

        resolved_voice = self._resolve_voice(voice)

        lang = _voice_lang(voice)

        # Prosodic steering: modulate embedding based on text content. Keep
        # walked voices stable for intelligibility.
        if not voice.startswith("walk:"):
            resolved_voice = _apply_prosodic_steering(self, resolved_voice, text)

        log.info("kokoro_tts_input", chars=len(text), lang=lang, sample=text[:60])
        log.debug("kokoro_tts_text", text=text, lang=lang)

        is_wav = response_format.lower() in ("wav", "pcm")

        try:
            stream = self._kokoro.create_stream(
                text,
                voice=resolved_voice,
                speed=speed,
                lang=lang,
            )

            if is_wav and stream_chunks:
                # Live-stream WAV: one sentinel header, then raw PCM from
                # every chunk. The header carries the FIRST sr observed;
                # if HBE flips rate mid-stream we log + keep emitting at
                # the original rate (the alternative is to resample,
                # which would warp pitch).
                first_sr: int | None = None
                async for samples, sr in stream:
                    samples, sr = await asyncio.to_thread(_apply_hbe, samples, sr)
                    if first_sr is None:
                        first_sr = sr
                        yield _wav_header_streaming(first_sr)
                    elif sr != first_sr:
                        log.warning(
                            "kokoro_stream_sr_mismatch",
                            first_sr=first_sr, chunk_sr=sr,
                            note="header committed at first_sr; chunk emitted unchanged",
                        )
                    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    if pcm16:
                        yield pcm16
                return

            if is_wav:
                # Buffered WAV: collect all PCM, emit one well-formed
                # file with real sizes at the end. Header is committed
                # at the first chunk's sr.
                first_sr = None
                pcm_parts: list[bytes] = []
                async for samples, sr in stream:
                    samples, sr = await asyncio.to_thread(_apply_hbe, samples, sr)
                    if first_sr is None:
                        first_sr = sr
                    elif sr != first_sr:
                        log.warning(
                            "kokoro_stream_sr_mismatch",
                            first_sr=first_sr, chunk_sr=sr,
                            note="header committed at first_sr; chunk emitted unchanged",
                        )
                    pcm_parts.append((np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
                if not pcm_parts or first_sr is None:
                    return
                all_pcm = b"".join(pcm_parts)
                if not all_pcm:
                    return
                # _pcm_to_wav builds a full file with real chunk sizes;
                # split into header + body so we can stride the body for
                # cancellation-friendly yields.
                full = _pcm_to_wav(all_pcm, first_sr)
                yield full[:44]
                body = full[44:]
                stride = 16 * 1024
                for i in range(0, len(body), stride):
                    yield body[i:i + stride]
                return

            # Non-WAV (mp3, opus, etc.): original per-chunk encode-and-
            # yield path. Container formats are designed to concatenate
            # — each MP3 frame is self-contained — so the chunking
            # strategy doesn't affect playability.
            async for samples, sr in stream:
                samples, sr = await asyncio.to_thread(_apply_hbe, samples, sr)
                pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                encoded = await _encode_audio(pcm16, sr, response_format)
                if encoded:
                    yield encoded

        except Exception as exc:
            log.warning("kokoro_stream_error", error=str(exc))

    def clear_voice_cache(self) -> None:
        """Clear the blended voice cache."""
        with self._voice_cache_lock:
            self._voice_cache.clear()


# ---------------------------------------------------------------------------
# Enhancement pipeline helpers
# ---------------------------------------------------------------------------


def _apply_hbe(samples: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Apply Harmonic Bandwidth Extension if enabled.  Thread-safe."""
    try:
        from augmentum.config import settings
        if not getattr(settings, "tts_kokoro_hbe", True):
            return samples, sr
    except ImportError as exc:
        # Defaults to enabled when settings isn't importable.
        log.debug("kokoro_hbe_settings_import_failed", error=str(exc))

    try:
        from augmentum.voice.hbe import extend_bandwidth
        return extend_bandwidth(samples, sr)
    except Exception as exc:
        log.debug("hbe_skipped", error=str(exc))
        return samples, sr


def _apply_prosodic_steering(
    kokoro: KokoroTTS,
    resolved_voice: str | np.ndarray,
    text: str,
) -> str | np.ndarray:
    """Apply prosodic embedding steering if enabled.  Thread-safe."""
    try:
        from augmentum.config import settings
        if not getattr(settings, "tts_kokoro_prosody", True):
            return resolved_voice
    except ImportError as exc:
        # Defaults to enabled when settings isn't importable.
        log.debug("kokoro_prosody_settings_import_failed", error=str(exc))

    if not isinstance(resolved_voice, np.ndarray):
        # Need the actual embedding to steer — resolve string to array
        try:
            resolved_voice = kokoro._kokoro.get_voice_style(resolved_voice)
        except Exception:
            return resolved_voice

    try:
        from augmentum.voice.prosody import ProsodyCartographer
        cart = ProsodyCartographer.instance(kokoro)
        return cart.steer(resolved_voice, text)
    except Exception as exc:
        log.debug("prosody_steering_skipped", error=str(exc))
        return resolved_voice


# ---------------------------------------------------------------------------
# Audio Encoding
# ---------------------------------------------------------------------------


async def _encode_audio(
    pcm16_bytes: bytes,
    sample_rate: int,
    fmt: str = "mp3",
) -> bytes:
    """Encode raw PCM16 mono bytes to the target format via ffmpeg.

    ``fmt == "pcm"`` returns the raw int16-LE bytes unchanged. This
    matches OpenAI's documented contract for ``response_format=pcm``
    (raw mono 24kHz signed little-endian, no container) and lets
    Android's AudioTrack consume the response directly without
    parsing a header. Web UI / voice mode / lipsync test never request
    pcm (they use mp3, wav, or the streaming path), so this branch is
    Android-only in practice.
    """
    if fmt == "pcm":
        return pcm16_bytes

    if fmt == "wav":
        return _pcm_to_wav(pcm16_bytes, sample_rate)

    if fmt in ("mp3", "opus", "aac", "flac"):
        return await _encode_ffmpeg(pcm16_bytes, sample_rate, fmt)

    # Fallback to WAV for unknown formats.
    return _pcm_to_wav(pcm16_bytes, sample_rate)


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16 mono bytes in a minimal WAV header."""
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,              # Subchunk1Size (PCM)
        1,               # AudioFormat (PCM = 1)
        1,               # NumChannels (mono)
        sample_rate,
        sample_rate * 2,  # ByteRate
        2,               # BlockAlign
        16,              # BitsPerSample
        b"data",
        data_size,
    )
    return header + pcm_bytes


def _wav_header_streaming(sample_rate: int) -> bytes:
    """Build a 44-byte WAV header with sentinel chunk sizes for the
    live-stream emission path in :meth:`KokoroTTS.stream_speech`.

    RIFF and data sizes are set to ``0x7FFFFFFF`` because we don't know
    the final length when the first byte goes out. Clients that consume
    the response as a true byte stream (Web Audio via ReadableStream.
    getReader) ignore the size field and read PCM until the stream
    closes — exactly what we want.

    DO NOT use this on a code path where the response might land in a
    buffered Blob (iOS Safari Audio element, anything that falls back
    to ``response.blob()``). The Blob would carry a WAV claiming a 2 GB
    data chunk over a 30 KB body and the decoder would refuse to play
    it. Use :func:`_pcm_to_wav` for those callers — the cost is waiting
    for full synthesis before the first byte, but the audio actually
    plays.

    Mirrors ``PocketTTS._wav_header_streaming`` — kept module-local in
    Kokoro so the streaming-vs-buffered shape is visible alongside the
    rest of Kokoro's WAV plumbing.
    """
    sentinel = 0x7FFFFFFF
    byte_rate = sample_rate * 2  # mono, 16-bit
    return (
        b"RIFF" + struct.pack("<I", sentinel)
        + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH",
            16,            # fmt chunk size
            1,             # PCM
            1,             # mono
            sample_rate,   # sample rate
            byte_rate,     # byte rate
            2,             # block align
            16,            # bits per sample
        )
        + b"data" + struct.pack("<I", sentinel)
    )


async def _encode_ffmpeg(
    pcm16_bytes: bytes,
    sample_rate: int,
    fmt: str,
) -> bytes:
    """Encode PCM16 mono to target format via ffmpeg subprocess."""
    codec_args: list[str] = []
    if fmt == "mp3":
        # -q:a 2 ≈ 190kbps VBR — preserves more high-frequency detail from
        # Kokoro's 24kHz output than the default q4 (~165kbps).
        codec_args = ["-codec:a", "libmp3lame", "-q:a", "2"]
    elif fmt == "opus":
        # 96k preserves prosody detail; 48k was audibly lossy at 24kHz source.
        codec_args = ["-codec:a", "libopus", "-b:a", "96k"]
    elif fmt == "aac":
        codec_args = ["-codec:a", "aac", "-b:a", "160k"]
    elif fmt == "flac":
        codec_args = ["-codec:a", "flac"]

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        *codec_args,
        "-f", fmt, "pipe:1",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=pcm16_bytes)
        if proc.returncode != 0:
            log.warning("kokoro_encode_error", fmt=fmt,
                        stderr=stderr[:200].decode(errors="replace") if stderr else "")
            return _pcm_to_wav(pcm16_bytes, sample_rate)
        return stdout
    except FileNotFoundError:
        log.warning("kokoro_ffmpeg_not_found", note="ffmpeg required for MP3/Opus encoding")
        return _pcm_to_wav(pcm16_bytes, sample_rate)
    except Exception as exc:
        log.warning("kokoro_encode_failed", error=str(exc))
        return _pcm_to_wav(pcm16_bytes, sample_rate)
