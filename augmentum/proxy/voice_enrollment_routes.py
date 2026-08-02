"""Voice enrollment API — register and manage speaker voiceprints.

Provides endpoints for the enrollment flow:
  1. Check enrollment status (has the user enrolled?)
  2. Submit enrollment audio samples (3 spoken phrases)
  3. Delete enrollment (re-enroll or opt out)
  4. Record a skip (don't re-prompt)

Enrollments are scoped to ``user_id`` in multi-tenant deployments — the
legacy ``scope`` column (which held the client IP) is preserved on write
for backward compatibility but no longer used for filtering.
"""

from __future__ import annotations

import asyncio

import uuid

from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger
from augmentum.voice.speaker import SpeakerVerifier, VoicePrint

log = get_logger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice-enrollment"])

# Shared verifier instance — initialized lazily on first enrollment
_verifier: SpeakerVerifier | None = None


def _get_verifier() -> SpeakerVerifier:
    global _verifier
    if _verifier is None:
        _verifier = SpeakerVerifier()
        _verifier.load_model()
    return _verifier


def _user_id(request: Request) -> str:
    """Extract user_id from the authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


def _normalize_scope(host: str) -> str:
    """Normalize localhost variants so IPv4/IPv6 loopback map to the same scope."""
    if host in ("::1", "::ffff:127.0.0.1", "localhost"):
        return "127.0.0.1"
    return host


def _legacy_scope(request: Request) -> str:
    """Derive the legacy IP-based scope (kept for column continuity only)."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        host = forwarded.split(",")[0].strip()
    else:
        client = request.client
        host = client.host if client else "default"
    return _normalize_scope(host)


async def _get_conn(request: Request):
    """Get the SQLite connection from app state."""
    sm = getattr(request.app.state, "state_manager", None)
    if not sm:
        return None
    from augmentum.state.backends.sqlite import SQLiteBackend
    if isinstance(sm.backend, SQLiteBackend):
        return sm.backend.conn
    return None


# ---------------------------------------------------------------------------
# Enrollment Status
# ---------------------------------------------------------------------------


@router.get("/enrollment")
async def get_enrollment_status(request: Request) -> JSONResponse:
    """Check if the authenticated user has an active voice enrollment."""
    conn = await _get_conn(request)
    if not conn:
        return JSONResponse({"enrolled": False, "error": "Database unavailable"})

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    cursor = await conn.execute(
        "SELECT id, quality, samples, enrolled_at FROM voice_enrollments "
        "WHERE user_id = ? ORDER BY enrolled_at DESC LIMIT 1",
        (uid,),
    )
    row = await cursor.fetchone()

    if row:
        return JSONResponse({
            "enrolled": True,
            "enrollment_id": row[0],
            "quality": row[1],
            "samples": row[2],
            "enrolled_at": row[3],
        })

    # Check if user declined enrollment
    cursor = await conn.execute(
        "SELECT value FROM app_settings WHERE key = ?",
        (f"voice_enrollment_declined:{uid}",),
    )
    declined_row = await cursor.fetchone()
    if declined_row:
        return JSONResponse({"enrolled": False, "declined": True})

    return JSONResponse({"enrolled": False})


# ---------------------------------------------------------------------------
# Decline / Skip Enrollment
# ---------------------------------------------------------------------------


@router.post("/enrollment/decline")
async def decline_enrollment(request: Request) -> JSONResponse:
    """Record that the user declined voice enrollment (don't re-prompt)."""
    conn = await _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    await conn.execute(
        "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
        (f"voice_enrollment_declined:{uid}", "true"),
    )
    await conn.commit()

    log.info("voice_enrollment_declined", user_id=uid)
    return JSONResponse({"declined": True})


# ---------------------------------------------------------------------------
# Enrollment (submit audio samples)
# ---------------------------------------------------------------------------

# Phrases the user reads aloud during enrollment.
# Mix of short and long to build a robust voiceprint across utterance lengths.
ENROLLMENT_PHRASES = [
    "Hey, how's it going today?",                                        # short, casual
    "My voice is my password, verify me.",                                # medium, classic
    "The quick brown fox jumps over the lazy dog near the riverbank.",    # long, pangram
    "I enjoy listening to music and reading books on rainy afternoons.",  # long, natural
    "Sure, sounds good to me.",                                          # short, casual
]


@router.get("/enrollment/phrases")
async def get_enrollment_phrases() -> JSONResponse:
    """Return the phrases the user should speak for enrollment."""
    return JSONResponse({"phrases": ENROLLMENT_PHRASES})


@router.post("/enrollment")
async def enroll_voice(
    request: Request,
    sample1: UploadFile = File(...),
    sample2: UploadFile = File(...),
    sample3: UploadFile = File(...),
    sample4: UploadFile = File(...),
    sample5: UploadFile = File(...),
) -> JSONResponse:
    """Submit 5 audio samples to create a voice enrollment.

    Each sample can be WebM/Opus (from browser MediaRecorder) or WAV.
    The server transcodes to 16kHz mono PCM16 for the speaker model.
    """
    conn = await _get_conn(request)
    if not conn:
        return JSONResponse(
            {"error": "Database unavailable"},
            status_code=503,
        )

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Read and transcode audio samples to PCM16. _to_pcm16 spawns ffmpeg via
    # subprocess.run (sync, up to 10s timeout per call) and reads/writes temp
    # files — running it inline would block the event loop for ~50s on a
    # 5-sample enrollment, starving the healthcheck and any concurrent
    # streams. Hand off to a worker thread per sample.
    samples: list[bytes] = []
    for f in [sample1, sample2, sample3, sample4, sample5]:
        data = await f.read()
        pcm = await asyncio.to_thread(_to_pcm16, data, f.filename or "sample.webm")
        samples.append(pcm)

    # Acoustic-space alignment: run enrollment audio through the same
    # DTLN + WebRTC NS/AGC pipeline that the live voice path applies
    # per-frame. Without this, the enrolled embedding sits in raw-mic
    # space while every runtime verify embedding sits in denoised+leveled
    # space — same speaker scores 0.22 one turn and 0.57 the next.
    samples = await asyncio.to_thread(_preprocess_for_speaker_model, samples)

    # Compute voiceprint
    try:
        verifier = _get_verifier()
    except Exception as exc:
        log.warning("enrollment_model_error", error=str(exc))
        return JSONResponse(
            {"error": "Speaker verification model unavailable"},
            status_code=503,
        )

    voiceprint = verifier.enroll(samples)
    if not voiceprint:
        from augmentum.voice.speaker import MIN_ENROLLMENT_SAMPLES
        return JSONResponse(
            {"error": f"Enrollment failed — need at least {MIN_ENROLLMENT_SAMPLES} "
                      "usable samples. Please speak clearly in a quiet environment "
                      "and try again."},
            status_code=400,
        )

    # Store voiceprint — replace old one atomically (delete + insert in
    # single commit so a failed re-enrollment doesn't lose the old voiceprint)
    legacy_scope = _legacy_scope(request)
    enrollment_id = str(uuid.uuid4())

    await conn.execute(
        "DELETE FROM voice_enrollments WHERE user_id = ?", (uid,),
    )
    await conn.execute(
        "DELETE FROM app_settings WHERE key = ?",
        (f"voice_enrollment_declined:{uid}",),
    )
    await conn.execute(
        "INSERT INTO voice_enrollments "
        "(id, scope, user_id, voiceprint, enrolled_at, quality, samples) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (enrollment_id, legacy_scope, uid, voiceprint.to_json(),
         voiceprint.enrolled_at, voiceprint.quality_score,
         voiceprint.sample_count),
    )
    await conn.commit()

    log.info("voice_enrolled", user_id=uid, quality=voiceprint.quality_score,
             samples=voiceprint.sample_count)

    return JSONResponse({
        "enrolled": True,
        "enrollment_id": enrollment_id,
        "quality": voiceprint.quality_score,
        "samples": voiceprint.sample_count,
    })


@router.delete("/enrollment")
async def delete_enrollment(request: Request) -> JSONResponse:
    """Delete the authenticated user's voice enrollment."""
    conn = await _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    await conn.execute(
        "DELETE FROM voice_enrollments WHERE user_id = ?", (uid,),
    )
    await conn.commit()

    log.info("voice_enrollment_deleted", user_id=uid)
    return JSONResponse({"deleted": True})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def load_voiceprint(conn, user_id: str) -> VoicePrint | None:
    """Load the most recent voiceprint for a user.

    Used by voice_routes.py to get the enrolled voiceprint for
    speaker verification during voice calls.
    """
    if not user_id:
        return None
    cursor = await conn.execute(
        "SELECT voiceprint FROM voice_enrollments "
        "WHERE user_id = ? ORDER BY enrolled_at DESC LIMIT 1",
        (user_id,),
    )
    row = await cursor.fetchone()
    if row and row[0]:
        try:
            return VoicePrint.from_json(row[0])
        except Exception:
            return None
    return None


async def is_enrollment_declined(conn, user_id: str) -> bool:
    """Check if the user has declined voice enrollment."""
    if not user_id:
        return False
    cursor = await conn.execute(
        "SELECT value FROM app_settings WHERE key = ?",
        (f"voice_enrollment_declined:{user_id}",),
    )
    row = await cursor.fetchone()
    return row is not None


def _preprocess_for_speaker_model(pcm_samples: list[bytes]) -> list[bytes]:
    """Run enrollment PCM through the same DTLN + AGC/NS chain that the
    live voice path applies before speaker verification.

    Why: ``voice_routes.py`` runs every captured frame through
    ``SpeechEnhancer`` (highpass + DTLN) and ``AudioProcessor``
    (WebRTC NS + AGC) before handing the PCM to ``SpeakerVerifier``.
    Enrollment used to skip that chain, so the enrolled voiceprint lived
    in a different acoustic space than every live embedding it was being
    compared to. Cosine scores for the same speaker swung 0.22 → 0.57
    turn-to-turn — not a threshold problem, an embedding-space problem.

    We construct fresh enhancer/processor instances using the same
    settings the live path reads, warm DTLN's LSTM with ~256 ms of
    silence (so the first real sample isn't fed through cold state),
    and process all 5 samples in sequence through the shared instances
    to mimic a live session's continuous noise tracking.

    Tail bytes that don't fill a full 32 ms DTLN frame are passed through
    unprocessed — the verifier's own VAD-trim usually strips them as
    trailing silence.
    """
    from augmentum.config import settings
    from augmentum.voice.audio_processor import AudioProcessor
    from augmentum.voice.denoiser import _BLOCK_LEN, SpeechEnhancer

    use_enhancer = settings.voice_denoise_enabled or settings.voice_highpass_hz > 0
    use_audio_proc = settings.voice_audio_agc or settings.voice_audio_ns

    if not use_enhancer and not use_audio_proc:
        return pcm_samples

    enhancer: SpeechEnhancer | None = None
    if use_enhancer:
        enhancer = SpeechEnhancer(
            highpass_hz=settings.voice_highpass_hz,
            model_dir=settings.voice_denoise_model_dir,
        )
        try:
            enhancer.load_model()
        except Exception as exc:
            log.warning("enrollment_enhancer_load_failed", error=str(exc))
            enhancer = None

    audio_proc: AudioProcessor | None = None
    if use_audio_proc:
        audio_proc = AudioProcessor(
            agc_enabled=settings.voice_audio_agc,
            ns_enabled=settings.voice_audio_ns,
            agc_target_dbfs=settings.voice_audio_agc_target_dbfs,
            ns_level=settings.voice_audio_ns_level,
        )

    frame_bytes = _BLOCK_LEN * 2  # 512 samples * 2 bytes (PCM16)

    if enhancer is not None:
        silence_frame = b"\x00" * frame_bytes
        for _ in range(8):  # ~256 ms of warmup
            enhancer.process_frame(silence_frame)

    processed: list[bytes] = []
    for pcm in pcm_samples:
        parts: list[bytes] = []
        offset = 0
        while offset + frame_bytes <= len(pcm):
            frame = pcm[offset:offset + frame_bytes]
            if enhancer is not None:
                frame = enhancer.process_frame(frame)
            if audio_proc is not None:
                frame = audio_proc.process_frame(frame)
            parts.append(frame)
            offset += frame_bytes
        if offset < len(pcm):
            parts.append(pcm[offset:])
        processed.append(b"".join(parts))

    log.info(
        "enrollment_preprocessed",
        samples=len(processed),
        denoise=enhancer is not None and enhancer.has_neural,
        highpass=settings.voice_highpass_hz if enhancer is not None else 0,
        agc=audio_proc is not None and settings.voice_audio_agc,
        ns=audio_proc is not None and settings.voice_audio_ns,
    )
    return processed


def _to_pcm16(audio_bytes: bytes, filename: str) -> bytes:
    """Convert audio to raw PCM16 16kHz mono for the speaker model.

    Handles WebM/Opus (from browser MediaRecorder) and WAV inputs.
    Always transcodes through ffmpeg to guarantee 16kHz mono format —
    WAV files from different sources may have varying sample rates.
    """
    # Always transcode via ffmpeg to ensure 16kHz mono PCM16
    from augmentum.voice.pipeline import _transcode_to_wav
    wav_bytes, _, _ = _transcode_to_wav(audio_bytes)

    # If transcoding succeeded, strip the WAV header to get raw PCM
    if wav_bytes[:4] == b"RIFF":
        return _strip_wav_header(wav_bytes)

    # ffmpeg unavailable — if it's a WAV, strip header (may be wrong sample rate)
    if len(audio_bytes) > 44 and audio_bytes[:4] == b"RIFF":
        return _strip_wav_header(audio_bytes)

    # Fallback: return as-is (speaker model may fail)
    return audio_bytes


def _strip_wav_header(data: bytes) -> bytes:
    """Strip WAV header to get raw PCM data, if present."""
    if len(data) > 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        # Find the "data" subchunk
        pos = 12
        while pos < len(data) - 8:
            chunk_id = data[pos:pos + 4]
            chunk_size = int.from_bytes(data[pos + 4:pos + 8], "little")
            if chunk_id == b"data":
                return data[pos + 8:pos + 8 + chunk_size]
            pos += 8 + chunk_size
        # Fallback: skip standard 44-byte header
        return data[44:]
    return data
