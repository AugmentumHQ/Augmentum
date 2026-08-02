"""TTS streaming helpers — provider resolution and audio streaming.

Extracted from pipeline.py for modularity.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.proxy.audio_routes import (
    _build_tts_stream,
    _get_default_provider,
    _get_provider_by_id,
    _is_csm_provider,
)
from augmentum.utils.http_client import normalize_base_url
from augmentum.utils.logging import get_logger
from augmentum.utils.model_load import load_model_off_loop

if TYPE_CHECKING:
    from starlette.websockets import WebSocket

    from augmentum.voice.pipeline import VoiceSession

log = get_logger(__name__)

# Provider IDs that support stream_pcm (real-time PCM streaming)
# Disabled for now — Qwen optimized backend drops connections mid-stream.
# Re-enable when upstream fixes streaming stability.
_STREAM_PCM_PROVIDERS: frozenset[str] = frozenset()


async def _resolve_tts_provider(
    conn,
    voice: str,
) -> tuple[dict | None, str]:
    """Resolve TTS provider and effective voice name.

    Uses the voice→provider auto-resolution map from audio_routes.
    If voice contains '::' prefix, routes to that provider explicitly.
    Otherwise looks up which provider owns the voice name and routes
    automatically — so "Vivian" goes to Qwen, "af_heart" goes to Kokoro,
    cloned voices go to Chatterbox, all without the user specifying.

    Returns (provider_dict, effective_voice).
    """
    from augmentum.proxy.audio_routes import resolve_voice_provider
    return await resolve_voice_provider(conn, voice)


async def prefetch_tts_audio(
    text: str,
    conn,
    voice: str = "",
    speed: float = 1.0,
    *,
    provider: dict | None = None,
    instruct: str = "",
    response_format: str = "",
    user_id: str = "",
    session_id: str = "",
) -> list[bytes]:
    """Fetch TTS audio for a sentence into a memory buffer.

    Used by the prefetch pipeline to pre-generate the next sentence's
    audio while the current sentence is still playing, eliminating
    inter-sentence silence.

    Pass *provider* to skip the per-sentence DB lookup when the provider
    has already been resolved for this turn.

    Pass *response_format* to override the default format (e.g. "wav"
    for Qwen to skip MP3 encoding overhead).

    Returns a list of audio byte chunks, or empty list on failure.
    """
    if provider is None:
        provider, voice = await _resolve_tts_provider(conn, voice)
    else:
        # Still need to strip provider_id prefix from voice string
        if voice and "::" in voice:
            _, voice = voice.split("::", 1)
    if not provider:
        return []

    # Built-in TTS engines (Kokoro, Pocket) synth in-process. The voice WS
    # route at voice_routes.py:945 already drives them directly; the prefetch
    # of the next sentence has no HTTP path to take, so don't even try —
    # httpx would otherwise reject base_url="builtin" and we'd log a warning
    # per sentence. Returning [] matches the comment at voice_routes.py:933.
    if provider.get("base_url") == "builtin":
        return []

    base_url = normalize_base_url(provider["base_url"])
    tts_voice = voice or provider["default_voice"]
    tts_model = provider["default_model"]
    fmt = response_format or settings.voice_tts_format

    try:
        stream = _build_tts_stream(
            base_url, tts_model, tts_voice, text,
            fmt, speed, provider["api_key"],
            pre_cleaned=True, instruct=instruct,
            provider_id=provider.get("id", ""),
            user_id=user_id,
            session_id=session_id,
        )
        chunks: list[bytes] = []
        async for chunk in stream:
            chunks.append(chunk)
        return chunks
    except Exception as exc:
        log.warning("voice_tts_prefetch_error", error=str(exc), text_len=len(text))
        return []


async def send_prefetched_audio(
    chunks: list[bytes],
    websocket: WebSocket,
    session: VoiceSession | None = None,
) -> bool:
    """Send pre-fetched TTS audio chunks over WebSocket.

    Used by the prefetch pipeline to deliver audio that was synthesized
    concurrently while the previous sentence was playing.

    Returns True if fully sent, False if interrupted.
    """
    try:
        for chunk in chunks:
            if session and session.interrupted:
                return False
            await websocket.send_bytes(chunk)
        return True
    except (ConnectionError, RuntimeError) as exc:
        log.warning("voice_tts_prefetch_send_error", error=str(exc))
        return False


async def _notify_tts_failure(
    websocket: WebSocket,
    session: VoiceSession | None,
    provider_id: str,
    detail: str,
) -> None:
    """Surface a TTS failure to the voice client instead of silently
    skipping the speaking phase — otherwise the call UI jumps straight
    from tokenizing back to listening with no explanation. Emitted at
    most once per session (this runs per sentence; a multi-sentence
    reply must not flood the client). The flag resets on the next
    successful synth so a later re-failure notifies again."""
    if session is not None:
        if getattr(session, "_tts_failure_notified", False):
            return
        session._tts_failure_notified = True
    # best-effort: the client may already be gone — nothing to notify
    with contextlib.suppress(Exception):
        await websocket.send_json({
            "type": "error",
            "message": (
                f"Voice output unavailable "
                f"({provider_id or 'no TTS provider'}): {detail}"
            )[:300],
        })


def _clear_tts_failure_flag(session: VoiceSession | None) -> None:
    if session is not None and getattr(session, "_tts_failure_notified", False):
        session._tts_failure_notified = False


async def stream_tts_sentence(
    text: str,
    websocket: WebSocket,
    conn,
    voice: str = "",
    speed: float = 1.0,
    session: VoiceSession | None = None,
    *,
    provider: dict | None = None,
    instruct: str = "",
    stream_pcm: bool = False,
    response_format: str = "",
    user_id: str = "",
) -> bool:
    """Send a sentence to TTS and stream audio chunks back over WebSocket.

    Pass *provider* to skip the per-sentence DB lookup when the provider
    has already been resolved for this turn.

    Pass *response_format* to override the default format (e.g. "wav"
    for Qwen to skip MP3 encoding overhead).

    Returns True if fully sent, False if interrupted.
    """
    if provider is None:
        provider, voice = await _resolve_tts_provider(conn, voice)
    else:
        if voice and "::" in voice:
            _, voice = voice.split("::", 1)

    if not provider:
        log.warning("voice_no_tts_provider")
        await _notify_tts_failure(
            websocket, session, "", "no TTS provider is configured",
        )
        return True

    # Phoneme-driven lip-sync (Phase 1): Kokoro provider + setting opt-in.
    # Synchronous synth so the audio duration is known when the schedule is
    # built; emits the schedule before audio chunks so the client has it
    # ready by the time playback starts. Failure here falls through to the
    # existing streaming path — never regresses the audio.
    engine = getattr(settings, "voice_lipsync_engine", "amplitude")
    is_kokoro = provider.get("id") == "kokoro-builtin"
    if is_kokoro and engine in ("phoneme", "auto"):
        ok = await _stream_kokoro_with_schedule(
            text=text,
            websocket=websocket,
            voice=voice or provider.get("default_voice", "af_heart"),
            speed=speed,
            session=session,
            response_format=response_format or settings.voice_tts_format,
        )
        if ok is not None:
            return ok
        # Helper signalled "not handled" — fall through to streaming path

    # In-process built-in TTS engines (kokoro-builtin, pockettts-builtin)
    # have no base_url. The HTTP path below would fail
    # with "Request URL is missing an 'http://' or 'https://' protocol"
    # when the lip-sync schedule path didn't apply (engine=amplitude,
    # default). Dispatch directly to the engine instead — mirrors the
    # audio_routes tts_synthesize_bytes flow.
    from augmentum.proxy.audio_routes import _BUILTIN_TTS_IDS, _builtin_tts_engine
    if provider.get("id", "") in _BUILTIN_TTS_IDS:
        eng = await _builtin_tts_engine(provider["id"])
        if eng is None:
            log.warning(
                "voice_tts_builtin_unavailable",
                provider=provider.get("id", ""),
            )
            await _notify_tts_failure(
                websocket, session, provider.get("id", ""),
                "engine failed to load — check server logs",
            )
            return False
        voice_name = voice or provider.get("default_voice", "")
        fmt = response_format or settings.voice_tts_format
        try:
            async for chunk in eng.stream_speech(
                text, voice=voice_name, speed=speed, response_format=fmt,
            ):
                if session and session.interrupted:
                    return False
                if chunk:
                    await websocket.send_bytes(chunk)
            _clear_tts_failure_flag(session)
            return True
        except Exception as exc:
            log.warning(
                "voice_tts_stream_error",
                error=str(exc),
                provider=provider.get("id", ""),
                voice=voice_name,
                text_len=len(text),
            )
            await _notify_tts_failure(
                websocket, session, provider.get("id", ""), str(exc)[:150],
            )
            return False

    base_url = normalize_base_url(provider["base_url"])
    tts_voice = voice or provider["default_voice"]
    tts_model = provider["default_model"]
    fmt = response_format or settings.voice_tts_format

    # Cross-modal: her current mood → a leading (emotion) tag the fine-tuned
    # CSM voice was trained on, so she *sounds* how she feels. Set per-turn on
    # the session (recency-gated, behind companion_csm_emotion_tag); only for
    # CSM, only when an emotion is active this turn.
    _emo = getattr(session, "csm_emotion", "") if session else ""
    if _emo and _is_csm_provider(provider.get("id", "")):
        text = f"({_emo}) {text}"
    else:
        # OpenAI-omni style providers (Higgs Audio v3 via sglang-omni, etc.)
        # interpret a leading natural descriptor — (warm), (excited), (gentle).
        # Drive it from her recency-gated affect (session.voice_affect, behind
        # companion_voice_emotion_tag). Skip if the text already opens with a
        # parenthetical cue (the model or a manual tag wrote one) so we never
        # double-stack. Provider-agnostic: any /v1/audio/speech endpoint that
        # honours style cues benefits; ones that don't simply ignore the prefix.
        _aff = getattr(session, "voice_affect", "") if session else ""
        if _aff and provider.get("id", "") == "openai-tts" and not text.lstrip().startswith("("):
            from augmentum.voice.companion_emotion import omni_descriptor_for_affect
            _desc = omni_descriptor_for_affect(_aff)
            if _desc:
                text = f"({_desc}) {text}"

    try:
        # Prefer the explicit user_id arg; fall back to the session's
        # user_id when present. Fabric-routed providers REQUIRE this
        # to authenticate to the peer; non-fabric paths ignore it.
        eff_user_id = user_id or (session.user_id if session else "")
        stream = _build_tts_stream(
            base_url, tts_model, tts_voice, text,
            fmt, speed, provider["api_key"],
            pre_cleaned=True, instruct=instruct, stream_pcm=stream_pcm,
            provider_id=provider.get("id", ""),
            user_id=eff_user_id,
            # Conversation id for context-aware engines (CSM). The voice
            # session id is the natural conversation key — every sentence of
            # a turn shares it, so the sidecar builds rolling self-context.
            session_id=(session.session_id if session else ""),
        )
        async for chunk in stream:
            if session and session.interrupted:
                return False
            await websocket.send_bytes(chunk)
        _clear_tts_failure_flag(session)
        return True
    except Exception as exc:
        log.warning(
            "voice_tts_stream_error",
            error=str(exc),
            provider=provider.get("id", ""),
            voice=tts_voice,
            text_len=len(text),
        )
        await _notify_tts_failure(
            websocket, session, provider.get("id", ""), str(exc)[:150],
        )
        # Return False so the caller knows TTS failed (don't silently pretend success)
        return False


async def _stream_kokoro_with_schedule(
    *,
    text: str,
    websocket: WebSocket,
    voice: str,
    speed: float,
    session: VoiceSession | None,
    response_format: str,
) -> bool | None:
    """Synchronous Kokoro synth + viseme-schedule emission for phoneme lip-sync.

    Returns True/False on success/interruption like stream_tts_sentence, or
    None if the helper couldn't run (caller should fall through to the
    streaming path so audio still reaches the user).
    """
    import asyncio
    import json

    import numpy as np

    try:
        from augmentum.voice.kokoro_tts import (
            KokoroTTS,
            _apply_hbe,
            _apply_prosodic_steering,
            _encode_audio,
            _voice_lang,
        )
        from augmentum.voice.phoneme_lipsync import (
            is_lang_supported,
            text_to_schedule,
        )
    except Exception as exc:
        log.warning("voice_phoneme_imports_unavailable", error=str(exc))
        return None

    kokoro = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
    if not kokoro.is_available:
        await load_model_off_loop(kokoro.load_model)
    if not kokoro.is_available:
        log.warning("voice_phoneme_kokoro_unavailable")
        return None

    lang = _voice_lang(voice)

    try:
        resolved_voice = kokoro._resolve_voice(voice)
        if not voice.startswith("walk:"):
            resolved_voice = _apply_prosodic_steering(kokoro, resolved_voice, text)
        samples, sr = await asyncio.to_thread(
            kokoro._kokoro.create,
            text,
            voice=resolved_voice,
            speed=speed,
            lang=lang,
        )
        samples, sr = await asyncio.to_thread(_apply_hbe, samples, sr)
    except Exception as exc:
        log.warning("voice_phoneme_synth_failed", error=str(exc), text_len=len(text))
        return None

    if session and session.interrupted:
        return False

    duration_ms = int(round(len(samples) / sr * 1000)) if sr else 0

    # Encode the audio BEFORE emitting the schedule. On encode failure this
    # helper returns None and the caller re-synthesizes over the streaming
    # fallback path — so a schedule emitted first would leave the client
    # holding a viseme schedule that never matches the audio it actually
    # plays (per-sentence lip/audio desync). Encoding first guarantees the
    # schedule ships only when THIS audio is what the client will hear.
    try:
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        audio_bytes = await _encode_audio(pcm16, sr, response_format)
    except Exception as exc:
        log.warning("voice_phoneme_encode_failed", error=str(exc))
        return None  # let caller try the streaming fallback

    if session and session.interrupted:
        return False

    schedule = text_to_schedule(text, duration_ms, lang=lang) if is_lang_supported(lang) else None
    if schedule is not None:
        try:
            await websocket.send_text(json.dumps({
                "type": "viseme_schedule",
                "sentence": text,
                "duration_ms": duration_ms,
                "events": schedule["events"],
            }))
        except Exception as exc:
            # Schedule emission is best-effort; audio must still play.
            log.warning("voice_phoneme_schedule_send_failed", error=str(exc))

    try:
        await websocket.send_bytes(audio_bytes)
    except (ConnectionError, RuntimeError) as exc:
        log.warning("voice_phoneme_send_failed", error=str(exc))
        return False

    # Success: clear the one-shot TTS-failure notify latch so a later
    # re-failure notifies the user again. This branch bypasses the two
    # _clear_tts_failure_flag calls in stream_tts_sentence, so without this
    # a single earlier failure would permanently suppress every future one.
    _clear_tts_failure_flag(session)
    return True


async def maybe_emit_normalized_schedule(
    text: str,
    websocket: WebSocket,
    provider: dict | None,
) -> None:
    """Emit a normalized viseme schedule for external (non-Kokoro) providers.

    Phase 2 of phoneme lip-sync. The Kokoro path emits an absolute-time
    schedule from :func:`_stream_kokoro_with_schedule` because its synth
    is in-process and the audio duration is exact at emission time.
    External providers stream audio over HTTP and don't expose duration
    until the client decodes the response, so we emit a *normalized*
    schedule (timing in ``[0.0, 1.0]``) here and let the client rescale
    using its decoded ``audio.duration`` at the moment it activates the
    schedule.

    Gates:

    - Provider must not be ``kokoro-builtin`` (that path emits its own
      absolute schedule). Built-in TTS providers that don't have a
      Kokoro-style schedule path are also excluded — they fall through
      to amplitude.
    - ``voice_lipsync_engine`` must be ``phoneme`` or ``auto``.
    - ``voice_lipsync_universal`` must be enabled (feature flag, default
      off until validated).
    - Text must pass the ASCII-heavy English gate.

    Best-effort: never raises, never blocks audio. Schedule emission is
    a quality enhancement; a failure here just means amplitude takes
    over for that sentence.
    """
    import json

    if not provider:
        return
    if provider.get("id") == "kokoro-builtin":
        return

    engine = getattr(settings, "voice_lipsync_engine", "amplitude")
    if engine not in ("phoneme", "auto"):
        return
    if not getattr(settings, "voice_lipsync_universal", False):
        return

    try:
        from augmentum.voice.phoneme_lipsync import (
            looks_english,
            text_to_normalized_schedule,
        )
        if not looks_english(text):
            return
        sched = text_to_normalized_schedule(text, lang="en-us")
        if not sched or not sched.get("events"):
            return
        await websocket.send_text(json.dumps({
            "type": "viseme_schedule",
            "sentence": text,
            "duration_ms": None,
            "events": sched["events"],
            "normalized": True,
        }))
    except Exception as exc:
        log.warning(
            "voice_normalized_schedule_failed",
            error=str(exc),
            provider=provider.get("id", "") if provider else "",
            text_len=len(text),
        )
