"""REST + WebSocket routes for wake-word model training, listing, and
streaming detection.

* ``POST /api/wake_word/train`` — enqueue a wake-word training job.
  Returns ``{job_id, status}``. Client polls /api/jobs/{job_id} for
  progress.
* ``GET  /api/wake_word/models`` — list the trained models on this
  instance.
* ``WS   /ws/voice/wake`` — streaming detection. Client sends PCM16
  16 kHz mono frames; server replies with ``{"type": "wake_detected",
  ...}`` JSON when a trained phrase is recognized. Also emits the
  detection on the PresenceBus as ``input.wake_detected``.

Training takes ~5 minutes on the GPU; the JobRunner pattern is the
right shape for that duration. Detection is real-time, ~5 ms ORT
inference per 250 ms hop on CPU.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import secrets
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket
from pydantic import BaseModel

from augmentum.companion_runtime.bus import emit_safe
from augmentum.config import settings as app_settings
from augmentum.utils.logging import get_logger
from augmentum.voice.wake_word.service import (
    get_or_create_service,
    load_models_from_db,
)

log = get_logger(__name__)

router = APIRouter(tags=["wake-word"])


# ── Request / response models ────────────────────────────────────────


class WakeWordTrainRequest(BaseModel):
    """POST /api/wake_word/train body."""

    avatar_id: str
    phrase: str
    voices: list[str] | None = None
    epochs: int | None = None
    # When true, the handler loads the user's personal recordings from
    # /data/wake_word_personal/{user_id}/{avatar_id}/ and mixes them
    # into the positives pool. See POST /api/wake_word/personal_samples
    # for how recordings get there.
    use_personal_samples: bool = False


class WakeWordTrainResponse(BaseModel):
    job_id: str
    avatar_id: str
    phrase: str
    status: str = "pending"


# ── Helpers ──────────────────────────────────────────────────────────


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


# ── Routes ───────────────────────────────────────────────────────────


@router.post("/api/wake_word/train", response_model=WakeWordTrainResponse)
async def train_wake_word(
    request: Request, body: WakeWordTrainRequest,
) -> WakeWordTrainResponse:
    """Enqueue a wake-word training job for the given avatar+phrase.

    Idempotent on the avatar_id — re-running replaces the previous model
    and bumps its version row. Training runs server-side via the JobRunner;
    poll /api/jobs/{job_id} for progress and the final result.
    """
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")

    avatar_id = (body.avatar_id or "").strip()
    phrase = (body.phrase or "").strip()
    if not avatar_id:
        raise HTTPException(status_code=400, detail="avatar_id is required")
    if not phrase:
        raise HTTPException(status_code=400, detail="phrase is required")

    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if jobs_store is None or job_runner is None:
        raise HTTPException(
            status_code=503, detail="background job queue unavailable",
        )

    payload: dict[str, Any] = {
        "avatar_id": avatar_id,
        "phrase": phrase,
    }
    if body.voices:
        payload["voices"] = list(body.voices)
    if body.epochs is not None:
        payload["epochs"] = int(body.epochs)
    if body.use_personal_samples:
        payload["use_personal_samples"] = True

    job_id = await jobs_store.create(
        user_id=user_id,
        job_type="wake_word_training",
        payload=payload,
        priority=5,
        # max_attempts=2 — single container restart shouldn't kill a
        # 15-minute training run permanently. Handler is idempotent
        # (re-runs replace the model + bump version).
        max_attempts=2,
    )
    job_runner.wake()
    log.info(
        "wake_word_training_enqueued",
        user_id=user_id, job_id=job_id, avatar_id=avatar_id, phrase=phrase,
    )

    return WakeWordTrainResponse(
        job_id=job_id, avatar_id=avatar_id, phrase=phrase,
    )


@router.get("/api/wake_word/corpora")
async def get_corpora_status(request: Request) -> dict[str, Any]:
    """Report whether the real-audio negatives corpus is installed.

    Lets the settings panel render an "install / installed" chip without
    waiting on the jobs queue. Also surfaces the bytes-on-disk so the
    operator knows what space is being used.
    """
    if not _user_id(request):
        raise HTTPException(status_code=401, detail="authentication required")

    # Lazy import — corpus module pulls torchaudio.
    from augmentum.voice.wake_word import negatives_corpus

    installed = negatives_corpus.is_installed()
    summary = negatives_corpus.installed_summary() if installed else {}

    # Surface whether a corpus-download job is currently in flight so the
    # UI can show progress without polling /api/jobs separately.
    in_flight: dict[str, Any] | None = None
    store = getattr(request.app.state, "jobs_store", None)
    if store is not None:
        try:
            recent = await store.list_for_user(
                user_id=_user_id(request),
                job_type="wake_word_corpus_download",
                limit=5,
            )
            for j in recent:
                status = j.get("status")
                if status in ("pending", "running"):
                    in_flight = {
                        "job_id": j.get("job_id"),
                        "status": status,
                        "progress": j.get("progress") or 0.0,
                        "stage": j.get("stage") or "",
                    }
                    break
        except Exception:
            log.warning("wake_corpus_status_jobs_lookup_failed", exc_info=True)

    return {
        "installed": installed,
        "summary": summary,
        "in_flight_job": in_flight,
    }


@router.post("/api/wake_word/corpora")
async def install_corpora(request: Request) -> dict[str, Any]:
    """Enqueue the LibriSpeech dev-clean download job.

    The corpus install is the structural fix for synthetic-only training
    artifacts — silence false positives, TTS self-trigger, threshold
    mis-calibration. Idempotent: a re-run when the corpus is already on
    disk short-circuits inside the handler. Returns ``{job_id, status}``;
    client polls /api/jobs/{job_id} (or re-polls this endpoint) for
    progress.
    """
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")

    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if jobs_store is None or job_runner is None:
        raise HTTPException(
            status_code=503, detail="background job queue unavailable",
        )

    # If a job is already pending/running, return it instead of stacking
    # duplicates — the corpus is global, racing two downloads would just
    # produce thrash on the same partial file.
    try:
        recent = await jobs_store.list_for_user(
            user_id=user_id,
            job_type="wake_word_corpus_download",
            limit=5,
        )
        for j in recent:
            if j.get("status") in ("pending", "running"):
                return {
                    "job_id": j.get("job_id"),
                    "status": j.get("status"),
                    "reused": True,
                }
    except Exception:
        log.warning("wake_corpus_install_jobs_lookup_failed", exc_info=True)

    job_id = await jobs_store.create(
        user_id=user_id,
        job_type="wake_word_corpus_download",
        payload={},
        priority=5,
        # max_attempts=3 — download + extract is best-effort; a single
        # network blip shouldn't permanently fail an idempotent job.
        max_attempts=3,
    )
    job_runner.wake()
    log.info(
        "wake_word_corpus_download_enqueued",
        user_id=user_id, job_id=job_id,
    )
    return {"job_id": job_id, "status": "pending", "reused": False}


@router.get("/api/wake_word/models")
async def list_models(request: Request) -> dict[str, Any]:
    """List the trained wake-word models on this instance."""
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")

    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    if backend is None:
        raise HTTPException(status_code=503, detail="state store unavailable")
    conn = backend.conn

    cur = await conn.execute(
        """
        SELECT avatar_id, phrase, model_path, version, trained_at,
               train_metrics, is_builtin
        FROM wake_word_models
        ORDER BY is_builtin DESC, trained_at DESC
        """
    )
    rows = await cur.fetchall()
    await cur.close()

    models = []
    for row in rows:
        try:
            metrics = json.loads(row[5]) if row[5] else {}
        except Exception:
            metrics = {}
        models.append({
            "avatar_id": row[0],
            "phrase": row[1],
            "model_path": row[2],
            "version": row[3],
            "trained_at": row[4],
            "metrics": metrics,
            "is_builtin": bool(row[6]),
        })
    return {"models": models}


# ── Personal voice samples ───────────────────────────────────────────
#
# Real positives from the user's own voice — the structural fix for the
# FRR-on-out-of-training-voices problem the eval harness caught after
# the v3/v4 bakes. Each recording is a 0.5-2.0s WAV of the user saying
# the wake phrase; training mixes these into the positives pool so the
# model anchors on the actual speaker who'll be using it.
#
# Layout under {data_dir}/wake_word_personal/{user_id}/{avatar_id}/:
#   {sample_id}.wav   — 16 kHz mono PCM16, normalized
#   {sample_id}.json  — sidecar: created_at, duration_ms, rms_dbfs

# Validation thresholds. Anything below these is rejected at upload —
# the user gets a helpful error in the UI instead of training on bad
# data.
_PERSONAL_MIN_DURATION_S = 0.5
_PERSONAL_MAX_DURATION_S = 2.0
_PERSONAL_MIN_RMS_DBFS = -40.0
_PERSONAL_MAX_BYTES = 1_000_000   # 1 MB — clean 2s 16kHz 16-bit is ~64KB
_PERSONAL_TARGET_SR = 16000


def _is_safe_path_id(s: str) -> bool:
    """Refuse path traversal + funky characters in user-supplied IDs."""
    return bool(s) and len(s) <= 64 and all(c.isalnum() or c in "_-" for c in s)


def _personal_dir(user_id: str, avatar_id: str) -> Path:
    return Path(app_settings.data_dir) / "wake_word_personal" / user_id / avatar_id


def _decode_wav_to_mono16k(wav_bytes: bytes) -> tuple[np.ndarray, int, float]:
    """Decode a WAV blob to mono float32 samples. Returns (samples, sr, duration_s).

    Accepts 16-bit and 32-bit PCM. Mixes multichannel down to mono. Does
    NOT resample yet — caller decides whether to keep native rate or
    convert.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
    except (wave.Error, EOFError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"could not parse WAV: {exc}",
        ) from exc

    if sampwidth == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported sample width {sampwidth} (need 16-bit or 32-bit PCM)",
        )

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    duration_s = len(samples) / sr if sr > 0 else 0.0
    return samples, sr, duration_s


def _resample_to(samples: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample via torchaudio if rates differ. Pulls torch lazily."""
    if from_sr == to_sr:
        return samples
    import torch
    import torchaudio
    t = torch.from_numpy(samples).unsqueeze(0)
    t = torchaudio.functional.resample(t, from_sr, to_sr)
    return t.squeeze(0).numpy()


def _write_pcm16_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    """Write samples (float32 in [-1, 1]) as 16-bit PCM mono WAV."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _count_personal(user_id: str, avatar_id: str) -> int:
    d = _personal_dir(user_id, avatar_id)
    return sum(1 for _ in d.glob("*.wav")) if d.exists() else 0


def _load_personal_meta(meta_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


@router.post("/api/wake_word/personal_samples")
async def upload_personal_sample(
    request: Request,
    avatar_id: str = Form(...),
    audio: UploadFile = File(...),
) -> dict[str, Any]:
    """Accept a WAV recording of the wake phrase for personal training.

    Validates duration (0.5-2.0s), audio level (above -40 dBFS), size
    (under 1 MB), then resamples to 16 kHz mono and stores under the
    user's personal-samples dir. Multiple recordings accumulate; the
    training job picks them all up on next bake.
    """
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")
    if not _is_safe_path_id(avatar_id):
        raise HTTPException(status_code=400, detail="invalid avatar_id")

    wav_bytes = await audio.read()
    if len(wav_bytes) == 0:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(wav_bytes) > _PERSONAL_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file too large: {len(wav_bytes)} bytes > {_PERSONAL_MAX_BYTES}",
        )

    samples, sr, duration_s = _decode_wav_to_mono16k(wav_bytes)
    if duration_s < _PERSONAL_MIN_DURATION_S or duration_s > _PERSONAL_MAX_DURATION_S:
        raise HTTPException(
            status_code=400,
            detail=(
                f"duration {duration_s:.2f}s out of range "
                f"[{_PERSONAL_MIN_DURATION_S}, {_PERSONAL_MAX_DURATION_S}]"
            ),
        )

    rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-10))
    if rms_dbfs < _PERSONAL_MIN_RMS_DBFS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"audio too quiet ({rms_dbfs:.1f} dBFS) — record louder. "
                f"Aim for normal speaking volume (~-25 dBFS)."
            ),
        )

    if sr != _PERSONAL_TARGET_SR:
        samples = _resample_to(samples, sr, _PERSONAL_TARGET_SR)

    sample_id = secrets.token_hex(8)
    out_dir = _personal_dir(user_id, avatar_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{sample_id}.wav"
    meta_path = out_dir / f"{sample_id}.json"

    _write_pcm16_wav(wav_path, samples, _PERSONAL_TARGET_SR)
    meta = {
        "id": sample_id,
        "avatar_id": avatar_id,
        "created_at": int(time.time()),
        "duration_ms": int(round(duration_s * 1000)),
        "rms_dbfs": round(rms_dbfs, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    log.info(
        "wake_word_personal_sample_saved",
        user_id=user_id, avatar_id=avatar_id, sample_id=sample_id,
        duration_ms=meta["duration_ms"], rms_dbfs=meta["rms_dbfs"],
    )
    return {
        "sample": meta,
        "count": _count_personal(user_id, avatar_id),
    }


@router.get("/api/wake_word/personal_samples")
async def list_personal_samples(
    request: Request, avatar_id: str,
) -> dict[str, Any]:
    """List the user's personal recordings for an avatar. Newest first."""
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")
    if not _is_safe_path_id(avatar_id):
        raise HTTPException(status_code=400, detail="invalid avatar_id")

    d = _personal_dir(user_id, avatar_id)
    samples: list[dict[str, Any]] = []
    if d.exists():
        for meta_path in d.glob("*.json"):
            meta = _load_personal_meta(meta_path)
            if meta:
                samples.append(meta)
    samples.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return {"samples": samples, "count": len(samples)}


@router.delete("/api/wake_word/personal_samples/{sample_id}")
async def delete_personal_sample(
    request: Request, sample_id: str, avatar_id: str,
) -> dict[str, Any]:
    """Drop a bad take. Both wav + sidecar removed; missing files are
    silently ignored so a half-deleted state self-heals on next call.
    """
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")
    if not _is_safe_path_id(sample_id) or not _is_safe_path_id(avatar_id):
        raise HTTPException(status_code=400, detail="invalid id")

    d = _personal_dir(user_id, avatar_id)
    wav_path = d / f"{sample_id}.wav"
    meta_path = d / f"{sample_id}.json"
    if wav_path.exists():
        wav_path.unlink()
    if meta_path.exists():
        meta_path.unlink()
    log.info(
        "wake_word_personal_sample_deleted",
        user_id=user_id, avatar_id=avatar_id, sample_id=sample_id,
    )
    return {"ok": True, "count": _count_personal(user_id, avatar_id)}


# ── Streaming detection WebSocket ────────────────────────────────────


@router.websocket("/ws/voice/wake")
async def voice_wake_websocket(websocket: WebSocket) -> None:
    """Streaming wake-word detection.

    Client → Server: PCM16 16 kHz mono binary frames (any length; the
    service buffers and runs inference on a 250 ms hop).
    Server → Client (JSON):
      ``{"type": "ready", "avatar_ids": [...]}`` on connect
      ``{"type": "wake_detected", "avatar_id", "phrase", "score", "t"}``
      ``{"type": "error", "message": "..."}``

    Query params:
      ``avatar_ids`` — comma-separated list of wake-word models to
        activate for this session. Required. Maps to rows in the
        ``wake_word_models`` table. Missing models are silently
        skipped; if none load successfully the connection still stays
        up but will never detect.

    Auth: same WS-ticket flow as ``/ws/voice``. AuthMiddleware sets
    ``scope['user']``; missing → 4001 close.
    """
    await websocket.accept()

    user = websocket.scope.get("user")
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    params = websocket.query_params
    raw_ids = (params.get("avatar_ids") or "").strip()
    if not raw_ids:
        await websocket.send_json({
            "type": "error",
            "message": "avatar_ids query param is required",
        })
        await websocket.close(code=4000, reason="missing avatar_ids")
        return
    avatar_ids = [s.strip() for s in raw_ids.split(",") if s.strip()]

    source_id = f"wake-ws-{id(websocket)}"
    app_state = websocket.app.state

    # Load detectors (idempotent — if a model is already enabled at the
    # same threshold/path it's a no-op).
    try:
        loaded = await load_models_from_db(app_state, avatar_ids=avatar_ids)
    except Exception as exc:
        log.warning("wake_ws_load_failed", error=str(exc))
        await websocket.send_json({
            "type": "error",
            "message": f"failed to load wake-word models: {exc}",
        })
        await websocket.close(code=1011, reason="model load failed")
        return

    service = get_or_create_service(app_state)
    # Scope this source's detection set explicitly. Without this the
    # service would fan PCM frames to every detector ever loaded — and
    # detectors loaded by a prior session with a different avatar
    # selection would keep firing for this session too.
    subscribed = service.subscribe_source(source_id, loaded)
    log.info(
        "wake_ws_connected",
        user_id=user.id, requested=avatar_ids, loaded=loaded,
        subscribed=subscribed, active=len(service.active_models()),
    )

    try:
        await websocket.send_json({
            "type": "ready", "avatar_ids": subscribed,
        })
    except Exception:
        return

    # Periodic telemetry — confirms the mic is actually piping PCM and
    # surfaces the highest sub-threshold score seen, so we can tell at a
    # glance whether the model is silent because no audio is arriving vs
    # because audio is arriving but no utterance is getting close to the
    # threshold. Without this, debugging from server logs alone is
    # essentially blind.
    _TELEMETRY_INTERVAL_SEC = 8.0
    frames_in = 0
    bytes_in = 0
    last_telemetry_at = time.monotonic()

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            raw = message.get("bytes")
            if not raw:
                # Ignore stray JSON frames — this endpoint is binary-only.
                continue

            frames_in += 1
            bytes_in += len(raw)
            now_mono = time.monotonic()
            if now_mono - last_telemetry_at >= _TELEMETRY_INTERVAL_SEC:
                diag = service.flush_diagnostics(source_id)
                log.info(
                    "wake_ws_audio_diagnostic",
                    source_id=source_id, frames=frames_in, bytes=bytes_in,
                    elapsed_sec=round(now_mono - last_telemetry_at, 1),
                    per_avatar=diag,
                )
                frames_in = 0
                bytes_in = 0
                last_telemetry_at = now_mono

            detections = service.feed(raw, source_id)
            if not detections:
                continue

            for d in detections:
                payload = {
                    "type": "wake_detected",
                    "avatar_id": d.avatar_id,
                    "phrase": d.phrase,
                    "score": round(d.score, 4),
                    "t": d.t,
                }
                try:
                    await websocket.send_json(payload)
                except Exception as exc:
                    log.info("wake_ws_send_failed", error=str(exc)[:200])
                    return
                # Fire-and-forget bus emit. Other in-process subscribers
                # (BeccaObserver, behavior modules) can react without
                # going through the WS layer. ``track`` holds a ref so
                # GC can't drop the emit before subscribers receive it.
                from augmentum.utils.bg_tasks import track
                track(emit_safe(
                    app_state, "input.wake_detected",
                    {
                        "avatar_id": d.avatar_id,
                        "phrase": d.phrase,
                        "score": d.score,
                        "source_id": d.source_id,
                        "user_id": user.id,
                    },
                ))
    finally:
        service.reset_source(source_id)
        log.info("wake_ws_disconnected", user_id=user.id, source_id=source_id)
        try:
            await websocket.close()
        except Exception:
            log.debug("wake_ws_close_failed", exc_info=True)
