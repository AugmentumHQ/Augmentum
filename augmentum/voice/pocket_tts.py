"""PocketTTS — Kyutai Pocket TTS engine, the ultra-light CPU tier.

A small CPU-only TTS engine that runs alongside Kokoro for hosts
without GPU headroom or when latency-on-CPU is the right trade-off.

Pocket TTS (`kyutai/pocket-tts <https://huggingface.co/kyutai/pocket-tts>`_)
is a ~100M-param flow-based language model from the Moshi team that
runs CPU-real-time and uses the Mimi codec.

  * **Languages**: 6 (English, French, German, Italian, Portuguese, Spanish).
  * **Voice cloning**: built-in from a short reference clip.
  * **Footprint**: ~236MB weights.
  * **No streaming output** from the underlying model — we still chunk by
    sentence for perceived latency, but each chunk is generated atomically.
  * **Not thread-safe** — inference is serialized through a lock. The
    underlying ``pocket_tts`` package explicitly documents batch=1 only.

Pocket ships 8 named voices (alba, marius, javert, jean, fantine,
cosette, eponine, azelma) drawn from its training-voice library.

Imports of the ``pocket_tts`` package are deferred to method calls so
this module loads cleanly even when the dependency isn't installed —
``is_available`` reflects whether the pip install + weights exist.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import re
import sys
import threading
from typing import Any, AsyncGenerator

import numpy as np

from augmentum.utils.logging import get_logger
from augmentum.voice.kokoro_tts import _encode_audio  # reuse PCM→{wav,mp3,opus,…}

log = get_logger(__name__)


# Pocket TTS caches its weights under ``~/.cache/pocket_tts`` by default;
# we let the upstream cache handle discovery + download. The model_dir
# setting only matters when the user wants a non-default location.
_DEFAULT_MODEL_DIR = ""  # empty = use upstream default cache
_DEFAULT_LANGUAGE = "english"
_DEFAULT_VOICE = "alba"

# Fallback voice list if the package import fails — kept for the test
# environment where ``pocket_tts`` isn't installed. The live catalog is
# loaded from the package at import time below.
_FALLBACK_VOICE_NAMES: tuple[str, ...] = (
    "alba", "marius", "javert", "jean",
    "fantine", "cosette", "eponine", "azelma",
)

# Gender hints per known voice — derived from the names + the source clips
# in the Kyutai TTS voice library. Override-safe; only used for UI labels.
_VOICE_GENDER: dict[str, str] = {
    # v1 / classic 8
    "alba": "female", "fantine": "female", "cosette": "female",
    "eponine": "female", "azelma": "female",
    "marius": "male", "javert": "male", "jean": "male",
    # v3 additions (vctk + voice-zero + voice-donations + multilingual defaults)
    "anna": "female", "eve": "female", "jane": "female", "mary": "female",
    "vera": "female", "caro_davy": "female", "estelle": "female",
    "lola": "female",
    "charles": "male", "george": "male", "michael": "male", "paul": "male",
    "bill_boerst": "male", "peter_yearsley": "male", "stuart_bell": "male",
    "giovanni": "male", "juergen": "male", "rafael": "male",
}

# Voices that aren't English — when the user picks one of these, Pocket TTS
# expects the model to be loaded with the matching language (the package
# enforces this at synth time). Used for the UI lang tag only; the actual
# language switching is a settings concern.
_VOICE_LANGUAGE: dict[str, str] = {
    "estelle": "fr", "giovanni": "it", "juergen": "de",
    "lola": "es", "rafael": "pt",
}


def _load_package_voice_catalog() -> tuple[str, ...]:
    """Pull the live voice catalog from the installed pocket-tts package.

    The catalog grows across package releases — v1 had 8 voices, v3 has
    26 (including the multilingual defaults). Reading from the package
    instead of hardcoding means a ``pip install -U pocket-tts`` surfaces
    new voices in the picker without a code change.

    Returns the fallback list when the package isn't importable (test
    environments, fresh installs without the dep).
    """
    try:
        from pocket_tts.models.tts_model import (  # type: ignore[import-untyped]
            _ORIGINS_OF_PREDEFINED_VOICES,
        )
        return tuple(sorted(_ORIGINS_OF_PREDEFINED_VOICES.keys()))
    except Exception:  # noqa: BLE001 — module/attr may not exist
        return _FALLBACK_VOICE_NAMES


_DEFAULT_VOICE_NAMES: tuple[str, ...] = _load_package_voice_catalog()

VOICE_META: dict[str, dict] = {
    n: {
        "gender": _VOICE_GENDER.get(n, "neutral"),
        "lang": _VOICE_LANGUAGE.get(n, "en"),
        "desc": f"Pocket TTS voice {n}",
    }
    for n in _DEFAULT_VOICE_NAMES
}


# Pocket TTS officially "can handle infinitely long text inputs" and its
# generate_audio_stream maintains internal prosody across the entire input
# it sees. Pre-splitting into per-sentence calls resets prosody at every
# seam, which is audible as a chunk-boundary cadence. The safety threshold
# below caps memory blow-up on pathological inputs (very long pastes); the
# fallback splitter still kicks in above it. 8 KB ≈ ~1200 words ≈ ~8 min
# of speech — well beyond a normal chat message.
_POCKET_NATIVE_MAX_CHARS = 8000


def _segments_for_pocket(text: str) -> list[str]:
    """Return text as a single segment when the engine can handle it natively.

    Falls back to ``_split_sentences`` only for inputs above the native cap
    so audio quality follows the engine's intended shape. Empty input
    returns ``[]`` so callers can iterate without special-casing.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= _POCKET_NATIVE_MAX_CHARS:
        return [text]
    return _split_sentences(text, max_chars=_POCKET_NATIVE_MAX_CHARS)


def _split_sentences(text: str, max_chars: int = 400) -> list[str]:
    """Sentence-aware split so streaming chunks land on natural prosodic boundaries."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    out: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        buf = ""
        for clause in re.split(r"(?<=[,;:])\s+", sent):
            clause = clause.strip()
            if not clause:
                continue
            if len(clause) > max_chars:
                if buf:
                    out.append(buf)
                    buf = ""
                for i in range(0, len(clause), max_chars):
                    out.append(clause[i:i + max_chars])
                continue
            if buf and len(buf) + 1 + len(clause) > max_chars:
                out.append(buf)
                buf = clause
            else:
                buf = f"{buf} {clause}" if buf else clause
        if buf:
            out.append(buf)
    return out


class PocketTTS:
    """Singleton wrapper around the Kyutai Pocket TTS model.

    Usage::

        tts = PocketTTS.instance()
        async for chunk in tts.stream_speech("Hello!", voice="alba"):
            ...

    The first ``load_model()`` call downloads ~236MB of weights from
    Hugging Face into ``~/.cache/pocket_tts``. Subsequent process
    starts load from cache instantly.
    """

    _instance: PocketTTS | None = None
    _instance_lock = threading.Lock()

    def __init__(self, model_dir: str = "", language: str = "") -> None:
        self._model_dir = model_dir or _DEFAULT_MODEL_DIR
        self._language = language or _DEFAULT_LANGUAGE
        self._model: Any = None              # pocket_tts.TTSModel
        # voice name → opaque ``voice_state`` (returned by the model). Loaded
        # lazily on first request per voice so startup isn't blocked on the
        # full voice library; the default voice is preloaded during warmup.
        self._voice_states: dict[str, Any] = {}
        self._voice_states_lock = threading.Lock()
        # Inference lock — pocket_tts is explicitly NOT thread-safe.
        self._inference_lock = threading.Lock()
        self._loaded = False
        self._sample_rate = 24_000  # overridden by model after load
        # Set by load_model after probing the loaded TTSModel. When True,
        # synthesis uses generate_audio_stream() for chunked output;
        # otherwise we fall back to generate_audio() (full tensor).
        self._supports_streaming = False
        # Set by load_model from the loaded TTSModel's ``has_voice_cloning``
        # attribute. False when the with-cloning weights couldn't be
        # downloaded — built-in voices still work, but cloned voices
        # (WAV-conditioned) fall back to the default voice.
        self._supports_voice_cloning = False

    # ── singleton ─────────────────────────────────────────────────────
    @classmethod
    def instance(cls, model_dir: str = "", language: str = "") -> PocketTTS:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(model_dir=model_dir, language=language)
        return cls._instance

    @classmethod
    def configure(cls, model_dir: str = "", language: str = "") -> None:
        inst = cls.instance(model_dir=model_dir, language=language)
        if model_dir:
            inst._model_dir = model_dir
        if language:
            inst._language = language

    # ── loading ───────────────────────────────────────────────────────
    def load_model(self) -> None:
        """Load the Pocket TTS model + preload the default voice state.

        Idempotent. Safe to call from multiple threads — protected by
        the singleton lock; the actual import + weight download happens
        exactly once even under contention.

        Tuning knobs applied:
          * ``temp=0.5`` (vs upstream default 0.7) — lower sampling
            temperature gives more consistent voice across re-reads of
            the same text, which matters for the chat/read-aloud loop
            where the user expects "the same voice both times".
          * ``config="without-voice-cloning"`` would load the smaller
            variant. We use the default (full ``kyutai/pocket-tts``)
            so ``get_state_for_audio_prompt`` accepts arbitrary WAV
            paths — the clone-via-WAV plumbing needs the full model.
        """
        if self._loaded and self._model is not None:
            return
        try:
            # Deferred import — keeps this module loadable without the
            # ``pocket-tts`` pip package installed. ``is_available``
            # tells callers whether the engine can actually do work.
            from pocket_tts import TTSModel  # type: ignore[import-untyped]
        except ImportError:
            log.warning(
                "pocket_tts_package_missing",
                hint="pip install pocket-tts",
            )
            return
        # Bridge the Augmentum-level HuggingFace token setting into the
        # HF Hub auth chain. Pocket TTS's with-cloning weights at
        # ``kyutai/pocket-tts`` are gated and require a token; users
        # typically configure ``huggingface_token`` once in the Augmentum
        # settings panel (which routes to ``AUGMENTUM_HUGGINGFACE_TOKEN``
        # in the container). The pocket-tts package reads ``HF_TOKEN``
        # / ``HUGGING_FACE_HUB_TOKEN`` instead — bridge them here so
        # users don't have to set the token in two places.
        try:
            from augmentum.config import settings as _cfg
            # Source precedence:
            #   1. HF_TOKEN env (explicit user override)
            #   2. HUGGING_FACE_HUB_TOKEN env (HF SDK convention)
            #   3. AUGMENTUM_HUGGINGFACE_TOKEN env (compose passthrough)
            #   4. settings.huggingface_token (UI field — may be encrypted)
            #   5. settings.image_huggingface_token (plaintext mirror that
            #      Augmentum's settings layer maintains alongside the
            #      encrypted primary; what actually has a usable value)
            # The settings encryption layer occasionally returns the raw
            # ``enc:...`` ciphertext when the decryption key isn't available;
            # filter those out so we don't ship garbage as a token.
            def _usable(value: str) -> str:
                value = (value or "").strip()
                if not value or value.startswith("enc:"):
                    return ""
                return value

            _hf_tok = (
                _usable(os.environ.get("HF_TOKEN", ""))
                or _usable(os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
                or _usable(os.environ.get("AUGMENTUM_HUGGINGFACE_TOKEN", ""))
                or _usable(getattr(_cfg, "huggingface_token", ""))
                or _usable(getattr(_cfg, "image_huggingface_token", ""))
            )
            if _hf_tok:
                # Only inject when not already set, so a real HF_TOKEN env
                # wins over a stale Augmentum-side setting.
                if not _usable(os.environ.get("HF_TOKEN", "")):
                    os.environ["HF_TOKEN"] = _hf_tok
                if not _usable(os.environ.get("HUGGING_FACE_HUB_TOKEN", "")):
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_tok
                log.info(
                    "pocket_tts_hf_token_bridged",
                    source=(
                        "env" if os.environ.get("AUGMENTUM_HUGGINGFACE_TOKEN")
                        else "settings"
                    ),
                    token_prefix=_hf_tok[:6] + "...",
                )
        except Exception as exc:  # noqa: BLE001 — token bridge is best-effort
            log.debug("pocket_tts_hf_token_bridge_failed", error=str(exc)[:160])

        try:
            kwargs: dict[str, Any] = {"temp": 0.5}
            if self._language:
                kwargs["language"] = self._language
            # Int8 quantization: per upstream docs, ~27% inference speedup
            # + ~48% memory reduction on x86 (FBGEMM), no measurable WER
            # impact. Free win when the setting is enabled. Without
            # ``torchao`` installed the package falls back to a slower
            # but still-quantized path; the speed-up is smaller but it
            # still works.
            try:
                from augmentum.config import settings as _cfg
                if getattr(_cfg, "tts_pocket_quantize", True):
                    kwargs["quantize"] = True
            except Exception:  # noqa: BLE001 — degrade silently
                kwargs["quantize"] = True
            self._model = TTSModel.load_model(**kwargs)
            self._sample_rate = int(getattr(self._model, "sample_rate", 24_000))
            # Detect whether the loaded variant supports incremental
            # generation. Newer ``pocket-tts`` ships ``generate_audio_stream``
            # which yields tensor chunks as they're produced — that gives
            # us the ability to start emitting audio bytes before the full
            # sentence has been generated. Older releases only have
            # ``generate_audio`` (full-tensor return); we fall back to that.
            self._supports_streaming = hasattr(self._model, "generate_audio_stream")
            # Voice cloning is gated on HuggingFace — the package falls
            # back to the without-cloning variant when HF_TOKEN isn't set
            # or the user hasn't accepted the terms at
            # https://huggingface.co/kyutai/pocket-tts. Surface the state
            # so the operator sees this at boot instead of discovering it
            # only when the first cloned voice fails.
            self._supports_voice_cloning = bool(
                getattr(self._model, "has_voice_cloning", False),
            )
            # Preload the default voice so the first user-triggered synth
            # doesn't pay the get_state_for_audio_prompt cost (which can
            # be hundreds of ms on cold start).
            self._voice_states[_DEFAULT_VOICE] = self._model.get_state_for_audio_prompt(
                _DEFAULT_VOICE,
            )
            self._loaded = True
            log.info(
                "pocket_tts_loaded",
                language=self._language,
                sample_rate=self._sample_rate,
                default_voice=_DEFAULT_VOICE,
                streaming=self._supports_streaming,
                voice_cloning=self._supports_voice_cloning,
                temp=0.5,
            )
            if not self._supports_voice_cloning:
                log.warning(
                    "pocket_tts_voice_cloning_unavailable",
                    hint=(
                        "Built-in voices work, but cloned voices (clips in "
                        "/data/voices/) won't. Accept the terms at "
                        "https://huggingface.co/kyutai/pocket-tts and set "
                        "HF_TOKEN to unlock."
                    ),
                )
            self._warmup()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "pocket_tts_load_failed",
                error=str(exc),
                exc_info=True,
            )
            self._model = None
            self._loaded = False

    # ── status ────────────────────────────────────────────────────────
    @property
    def is_available(self) -> bool:
        return self._loaded and self._model is not None

    @classmethod
    def status(cls) -> dict:
        inst = cls._instance
        return {
            "available": bool(inst and inst.is_available),
            "voices": len(inst._voice_states) if inst else 0,
            "language": inst._language if inst else _DEFAULT_LANGUAGE,
            "model_dir": inst._model_dir if inst else _DEFAULT_MODEL_DIR,
        }

    def get_voices(self) -> list[str]:
        """Return the built-in voice names.

        Note: Pocket TTS supports voice cloning from any reference WAV
        (path or ``hf://`` URL). ``stream_speech`` / ``generate`` accept
        such paths as the ``voice`` argument; this list only enumerates
        the named built-ins.
        """
        return list(_DEFAULT_VOICE_NAMES)

    # ── voice resolution ──────────────────────────────────────────────
    @staticmethod
    def _resolve_clone_path(name: str) -> str:
        """Map a bare voice name to a cloned voice WAV under ``/data/voices``.

        Returns the empty string when the name doesn't match any cloned
        file — the caller passes the original name through to Pocket's
        own resolver, which handles the built-in voice library and
        ``hf://`` / http URLs.
        """
        if not name:
            return ""
        # Skip paths + URLs + Pocket built-ins (no clone lookup needed).
        if (name in _DEFAULT_VOICE_NAMES
                or "://" in name
                or name.startswith(("./", "/", "~"))
                or name.endswith((".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"))):
            return ""
        import os
        from augmentum.config import settings as _cfg
        voice_dir = os.path.join(_cfg.data_dir or "/data", "voices")
        for ext in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"):
            candidate = os.path.join(voice_dir, f"{name}{ext}")
            if os.path.isfile(candidate):
                return candidate
        return ""

    def _resolve_voice_state(self, voice: str) -> Any:
        """Get-or-load the ``voice_state`` for ``voice``.

        ``voice`` can be:
          * a built-in name (``alba``, ``marius``, etc.)
          * a cloned voice name registered in ``/data/voices/`` — resolved
            transparently so cloned voices integrate with the same UI
            picker Chatterbox uses
          * a local WAV path
          * a ``hf://`` URL (Pocket TTS resolves via huggingface_hub)
          * an https URL (Pocket TTS fetches directly)

        Returns the default voice state on any resolution failure —
        false silence isn't an option for TTS, so we degrade to whatever
        voice we know we have.
        """
        if not self.is_available:
            return None
        v = (voice or _DEFAULT_VOICE).strip() or _DEFAULT_VOICE
        with self._voice_states_lock:
            cached = self._voice_states.get(v)
            if cached is not None:
                return cached
        # Try cloned-voice library before Pocket's own resolver; the bare
        # name "my_voice" should hit /data/voices/my_voice.wav, not get
        # rejected by Pocket as an unknown built-in.
        resolved = self._resolve_clone_path(v) or v
        try:
            state = self._model.get_state_for_audio_prompt(resolved)
            with self._voice_states_lock:
                # Cache under the ORIGINAL name so subsequent requests
                # with the bare name hit the cache directly.
                self._voice_states[v] = state
            if resolved != v:
                log.info("pocket_tts_clone_voice_loaded", name=v, path=resolved)
            return state
        except Exception as exc:  # noqa: BLE001
            # Returning the default voice state is the fallback. We
            # *intentionally* do NOT cache the fallback under ``v`` —
            # subsequent requests for the same name retry. This matters
            # because the most common reason for a load failure (during
            # initial deployment) is the HF Hub lock-directory race
            # where the first concurrent request to a voice repo blocks
            # before the lock directory has materialised on disk. The
            # retry succeeds once the directory exists.
            log.warning(
                "pocket_tts_voice_load_failed",
                voice=v, resolved=resolved,
                error=str(exc)[:200],
                fallback="default",
            )
            with self._voice_states_lock:
                return self._voice_states.get(_DEFAULT_VOICE)

    # ── synthesis ─────────────────────────────────────────────────────
    @staticmethod
    def _to_pcm16(samples: np.ndarray) -> bytes:
        return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    @contextlib.contextmanager
    def _capture_upstream_stdout(self):
        """Redirect upstream PocketTTS ``print()`` output into structlog.

        The bundled ``pocket_tts`` package emits per-inference perf
        breadcrumbs via plain ``print()``:

            starting timer now!
            Prompting text took 35 ms
            Average generation step time: 19 ms
            Generated: 3680 ms of audio in 1008 ms so 3.65x faster than real-time

        These bypass the global log level (so ``AUGMENTUM_LOG_LEVEL=
        WARNING`` doesn't silence them) and carry no structured
        fields — at high TTS volume they dominate stdout.

        We redirect ``sys.stdout`` for the duration of the inference
        call, capture the lines, and emit them as a single structured
        ``log.debug`` event on exit. Production deployments (default
        log level INFO) hide it; operators chasing TTS latency flip
        the level to DEBUG and see the same data with structured
        fields attached.

        The inference lock is held around every call site, so the
        stdout redirect window is short and serialised — no cross-
        talk risk from other module-level ``print()`` calls inside
        the same window.
        """
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                yield
            finally:
                output = buf.getvalue().strip()
                if output:
                    lines = [ln for ln in output.splitlines() if ln.strip()]
                    log.debug("pocket_tts_perf", lines=lines)

    def _generate_samples(self, voice_state: Any, text: str) -> np.ndarray:
        """Run full-tensor inference under the serialization lock.

        Pocket TTS isn't thread-safe — the upstream docs explicitly say
        server mode does not support concurrent requests. The lock is
        the simplest correct behavior; if throughput matters later,
        the right answer is a request queue, not finer-grained locking.
        """
        with self._inference_lock, self._capture_upstream_stdout():
            audio = self._model.generate_audio(voice_state, text)
        # Pocket TTS returns a torch tensor; .numpy() converts to ndarray.
        try:
            samples = audio.numpy()
        except AttributeError:
            samples = np.asarray(audio)
        return np.asarray(samples, dtype=np.float32).reshape(-1)

    def _generate_chunks(self, voice_state: Any, text: str):
        """Yield audio chunks as they're produced by the model.

        Uses ``generate_audio_stream`` when available, falling back to
        a single-shot ``generate_audio`` yield. This is the latency
        win — for a 1.5s utterance the first chunk can arrive in
        ~80-100ms instead of waiting ~500ms for the whole tensor.

        Runs inside the serialization lock; the generator yields chunks
        while the lock is held, so the streaming context lives across
        multiple yields. Callers that read this stream from an
        ``asyncio.to_thread`` wrapper see the chunks as they're produced.
        """
        with self._inference_lock, self._capture_upstream_stdout():
            for tensor_chunk in self._model.generate_audio_stream(
                voice_state, text,
            ):
                try:
                    samples = tensor_chunk.numpy()
                except AttributeError:
                    samples = np.asarray(tensor_chunk)
                yield np.asarray(samples, dtype=np.float32).reshape(-1)

    def _warmup(self) -> None:
        """Amortise first-inference cost with one short throwaway synth."""
        try:
            state = self._voice_states.get(_DEFAULT_VOICE)
            if state is None:
                return
            self._generate_samples(state, "Hi.")
            log.info("pocket_tts_warmup_done")
        except Exception as exc:  # noqa: BLE001
            log.debug("pocket_tts_warmup_failed", error=str(exc))

    async def stream_speech(
        self,
        text: str,
        voice: str = _DEFAULT_VOICE,
        speed: float = 1.0,
        response_format: str = "mp3",
        stream_chunks: bool = False,
    ) -> AsyncGenerator[bytes, None]:
        """Yield encoded audio bytes as the model produces them.

        Two WAV emission strategies, chosen by the caller (which knows
        what the client at the other end can decode):

        * ``stream_chunks=True`` — emit ONE WAV header up front with
          sentinel chunk sizes (2**31 - 1) and stream raw PCM int16
          frames from each segment as the model's incremental generator
          produces them. First audio byte arrives ~100 ms after the
          model warms up; the WebAudio reader on the client parses the
          header and reads PCM until the stream closes. Works only on
          clients that consume the response as a true byte stream
          (Android Chrome, desktop Firefox/Chrome via ReadableStream
          getReader). DOES NOT work for clients that buffer the
          response into a Blob — iOS Safari's Audio element won't
          decode a WAV that claims a 2 GB data chunk over a 30 KB body.

        * ``stream_chunks=False`` — accumulate all PCM across every
          segment, emit one buffered WAV at the end with the REAL
          chunk sizes. Loses the streaming latency win (user waits
          for full synthesis before first byte arrives) but plays in
          every client path including iOS Audio elements. This is
          the safe default and what audio_routes uses when the
          client doesn't ask for streaming.

        Both paths share the same per-segment producer
        (``_stream_segment_pcm``); only the header strategy and chunk
        cadence differ.

        For MP3 (and older Pocket builds without
        ``generate_audio_stream``) we fall through to per-segment full-
        tensor synth + ffmpeg encode — that path was always per-segment
        even in the original implementation.

        ``speed`` is accepted for API parity but currently a no-op —
        Pocket TTS has no native speed control. Resampling for tempo
        adjustment would hurt quality.
        """
        if not self.is_available:
            log.warning("pocket_tts_stream_unavailable")
            return
        voice_state = self._resolve_voice_state(voice)
        if voice_state is None:
            log.warning("pocket_tts_no_voice_state", voice=voice)
            return
        if speed != 1.0:
            log.debug(
                "pocket_tts_speed_ignored",
                speed=speed,
                note="Pocket TTS has no native speed control",
            )

        use_streaming_wav = (
            self._supports_streaming
            and response_format.lower() in ("wav", "pcm")
        )

        if use_streaming_wav and stream_chunks:
            # Live-stream path: ONE sentinel-size WAV header, then PCM
            # chunks from each segment as the model emits them. Caller
            # must guarantee the client consumes the response as a true
            # byte stream — otherwise the buffered Blob path on the
            # client side cannot decode this header shape.
            yield self._wav_header_streaming(self._sample_rate)
            for segment in _segments_for_pocket(text):
                async for pcm_chunk in self._stream_segment_pcm(voice_state, segment):
                    yield pcm_chunk
            return

        if use_streaming_wav:
            # Buffered path: collect PCM bytes from every segment, then
            # emit one valid WAV with real lengths. Plays in every
            # client path including iOS Audio elements.
            pcm_parts: list[bytes] = []
            for segment in _segments_for_pocket(text):
                async for pcm_chunk in self._stream_segment_pcm(voice_state, segment):
                    pcm_parts.append(pcm_chunk)
            all_pcm = b"".join(pcm_parts)
            if not all_pcm:
                return
            yield self._wav_header_sized(self._sample_rate, len(all_pcm))
            # Yield the body in modest slices so cancellation upstream
            # gets a chance to fire between yields rather than waiting
            # for one giant write.
            stride = 16 * 1024
            for i in range(0, len(all_pcm), stride):
                yield all_pcm[i:i + stride]
            return

        for segment in _segments_for_pocket(text):
            # Fallback: full-tensor synthesis + encode-as-blob.
            try:
                samples = await asyncio.to_thread(
                    self._generate_samples, voice_state, segment,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("pocket_tts_infer_error", error=str(exc))
                continue
            pcm16 = self._to_pcm16(samples)
            if not pcm16:
                continue
            encoded = await _encode_audio(pcm16, self._sample_rate, response_format)
            if encoded:
                yield encoded

    @staticmethod
    def _wav_header_streaming(sample_rate: int) -> bytes:
        """Build a 44-byte WAV header with sentinel chunk sizes for the
        live-stream emission path.

        RIFF and data sizes are set to ``0x7FFFFFFF`` because we don't
        know the final length when the first byte goes out. Clients that
        consume the response as a true byte stream (Web Audio via
        ReadableStream.getReader) ignore the size field and read PCM
        until the stream closes — exactly what we want.

        DO NOT use this on a code path where the response might land in
        a buffered Blob (iOS Safari Audio element, anything that falls
        back to ``response.blob()``). The Blob would carry a WAV
        claiming a 2 GB data chunk over a 30 KB body and the decoder
        would refuse to play it. Use ``_wav_header_sized`` for those
        callers — the cost is waiting for full synthesis before the
        first byte, but the audio actually plays.
        """
        import struct
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

    @staticmethod
    def _wav_header_sized(sample_rate: int, data_size: int) -> bytes:
        """Build a 44-byte WAV header for a PCM int16 mono payload of
        ``data_size`` bytes.

        Used by the buffered-WAV emission path in :meth:`stream_speech`:
        we accumulate all PCM in memory across segments, then emit one
        header with the real size and the full body. This guarantees a
        well-formed file in every client path — true streaming, blob
        buffering on iOS, reverse-proxy buffering on HTTP/2 — at the
        cost of waiting for the full synthesis before the first byte.

        Replaces the earlier sentinel-size header that only worked when
        the client genuinely consumed the response as a chunked stream;
        the moment that response landed in an Audio element via Blob,
        the file became unplayable (data chunk claimed 2 GB).
        """
        import struct
        byte_rate = sample_rate * 2  # mono, 16-bit
        return (
            b"RIFF" + struct.pack("<I", 36 + data_size)
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
            + b"data" + struct.pack("<I", data_size)
        )

    async def _stream_segment_pcm(
        self, voice_state: Any, segment: str,
    ) -> AsyncGenerator[bytes, None]:
        """Yield raw PCM int16 chunks for one text segment (no WAV header).

        The header is emitted ONCE by :meth:`stream_speech` once all
        segments have been generated — embedding a header per segment
        would create a stream of concatenated WAV files that no browser
        can decode as a single audio source.

        Runs the model's streaming generator in a thread via a queue
        bridge — each tensor chunk produced by the model is converted
        to PCM int16 and pushed into the queue, which the async caller
        drains. The generator's lock is held in the producer thread for
        the duration of one segment (matching the upstream "no
        concurrent requests" guarantee).

        Cancellation: if the caller stops iterating (e.g. client
        disconnect mid-stream), the ``finally`` block sets the cancel
        flag so the producer thread breaks out of the model loop on
        its next yield. The thread is joined via ``await task`` so we
        don't leak background work — important under load when many
        cancelled requests would otherwise pile up.

        Error fallback: if the streaming generator raises (e.g. some
        text segments hit an edge in the model), we fall through to
        the full-tensor path so the user still hears something for
        that segment rather than a silent gap.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=32)
        cancelled = threading.Event()
        stream_failed: list[str] = []  # used as nullable str — mutable cell

        def _safe_put(item: bytes | None) -> None:
            """Enqueue from the loop, fail-loud on overflow.

            Pre-2026-05-31: ``call_soon_threadsafe(queue.put_nowait, pcm)``
            scheduled a bare ``put_nowait`` on the loop. When the consumer
            fell behind (32-slot queue full), ``QueueFull`` raised on the
            loop side and went into the default exception handler — the
            producer never saw it, kept pushing chunks, and audio dropped
            silently mid-stream. Now overflow logs once and signals the
            cancel event so the producer stops cleanly.
            """
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                if not cancelled.is_set():
                    log.warning(
                        "pocket_tts_queue_overflow",
                        queue_max=queue.maxsize,
                        note="consumer slower than producer — stopping stream",
                    )
                    cancelled.set()

        def _producer():
            try:
                for samples in self._generate_chunks(voice_state, segment):
                    if cancelled.is_set():
                        break
                    pcm = self._to_pcm16(samples)
                    if pcm:
                        loop.call_soon_threadsafe(_safe_put, pcm)
            except Exception as exc:  # noqa: BLE001
                stream_failed.append(str(exc)[:200])
            finally:
                # Sentinel for clean consumer shutdown — also queue-bound, so
                # route it through the same overflow-safe path.
                loop.call_soon_threadsafe(_safe_put, None)

        task = loop.run_in_executor(None, _producer)

        emitted_any_pcm = False
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                emitted_any_pcm = True
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnect / pipeline cancel — signal the producer
            # to bail and re-raise so the caller sees the cancel.
            cancelled.set()
            raise
        finally:
            cancelled.set()
            # Drain the producer cleanly — silent on already-completed.
            try:
                await task
            except Exception as exc:  # noqa: BLE001
                log.debug("pocket_tts_producer_join_failed", error=str(exc)[:160])

        # Error recovery: if the streaming path raised and we produced
        # NOTHING usable, retry the segment via full-tensor synthesis so
        # the user hears the sentence instead of a silent gap.
        if stream_failed and not emitted_any_pcm:
            log.warning(
                "pocket_tts_stream_fallback_to_full",
                segment_chars=len(segment),
                error=stream_failed[0],
            )
            try:
                samples = await asyncio.to_thread(
                    self._generate_samples, voice_state, segment,
                )
                pcm = self._to_pcm16(samples)
                if pcm:
                    yield pcm
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "pocket_tts_full_fallback_failed",
                    error=str(exc)[:200],
                )

    async def generate(
        self,
        text: str,
        voice: str = _DEFAULT_VOICE,
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> bytes:
        """Synthesize the whole text, returning one encoded blob."""
        if not self.is_available:
            log.warning("pocket_tts_generate_unavailable")
            return b""
        voice_state = self._resolve_voice_state(voice)
        if voice_state is None:
            log.warning("pocket_tts_no_voice_state", voice=voice)
            return b""
        if speed != 1.0:
            log.debug("pocket_tts_speed_ignored", speed=speed)
        pcm = bytearray()
        for segment in _segments_for_pocket(text):
            try:
                samples = await asyncio.to_thread(
                    self._generate_samples, voice_state, segment,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("pocket_tts_infer_error", error=str(exc))
                continue
            pcm += self._to_pcm16(samples)
        if not pcm:
            return b""
        return await _encode_audio(bytes(pcm), self._sample_rate, response_format)
