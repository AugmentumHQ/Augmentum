"""Wake-word detection — ONNX inference over a sliding 1-second window.

Loads ONNX models trained by ``training.py`` and runs streaming
inference on PCM16 16 kHz mono audio frames. On positive detection
above the model's trained ``best_threshold`` and outside the refractory
window, returns a :class:`WakeDetection` the caller can publish on the
PresenceBus and/or forward to the source connection as a JSON event.

Source-agnostic by design: a browser mic via ``/ws/voice/wake`` feeds
frames under one ``source_id``; a fabric-routed input from a peer node
feeds under another. The service multiplexes detectors against sources.

Components:
  * :class:`WakeWordDetector` — one model, one ORT session, per-source
    rolling buffers. Owned by :class:`WakeWordService`.
  * :class:`WakeWordService` — orchestrator. Owns the active detector
    set. Public API: ``enable_model``, ``disable_model``, ``feed``,
    ``reset_source``, ``dispose``.

Inference cost: ~5 ms per window on CPU. Hop = 250 ms (4 inferences/sec
per active model). Refractory = 2.0 s — no double-fire within the same
utterance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Audio constants — must match training ──────────────────────────────

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000  # 1 second
N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160

# Run inference every INFERENCE_HOP_SAMPLES new samples (= 250 ms at
# 16 kHz). Slower hops save CPU but raise worst-case wake latency; the
# trained models are robust to the phrase landing anywhere in the
# window so we don't need a smaller hop for accuracy.
INFERENCE_HOP_SAMPLES = 4000

# After a positive detection, suppress further detections from the same
# source for this many seconds. Prevents a single utterance triggering
# multiple times as the 1-second window slides through it.
REFRACTORY_SEC = 2.0

# Fallback sigmoid cutoff if a model's train_metrics didn't record one.
_DEFAULT_THRESHOLD = 0.5

# Quality floor — refuse to load a model whose training run never
# learned to discriminate. Below these, the model false-positives on
# room tone / TTS playback and creates a feedback loop where the
# avatar's own voice triggers its own wake word. The thresholds are
# loose enough to allow first-cut models, tight enough to exclude the
# old single-word "becca" model (val_acc 0.60, pos 168) which was the
# observed loop source.
_MIN_VAL_ACC_FOR_USE = 0.85
_MIN_POSITIVES_FOR_USE = 300

# Runtime threshold floors — pipeline-aware. ``_sweep_threshold`` picks
# F1-optimal against the val set; whether that's trustworthy depends on
# whether the val set looks like production audio.
#
# ``synthetic_only`` models have a val set built entirely from Kokoro
# phrases. Real-world silence/room-tone lives in a different acoustic
# distribution and falsely scores 0.3-0.5 against these models, so the
# stored F1-optimal (often 0.25) is dramatically mis-calibrated. Floor
# at 0.65 to suppress idle-room false-positives.
#
# ``real_audio`` models have a val set that includes LibriSpeech +
# synthetic silence/noise — the same acoustic distributions as
# production. The stored threshold is far more trustworthy, but the
# eval harness (scripts/wake_word_eval.py) shows F1-optimal can still
# land slightly low because val under-samples the long tail of weak
# positives. Floor at 0.50 to keep recall while still rejecting
# clearly-non-wake audio.
_MIN_RUNTIME_THRESHOLD_SYNTHETIC = 0.65
_MIN_RUNTIME_THRESHOLD_REAL_AUDIO = 0.50

# Silence gate — RMS below this is treated as essentially-no-audio
# and skipped before inference. Avoids the "fires on room tone"
# failure mode entirely (the model was never trained on silence as a
# negative class). Value is in linear amplitude on the [-1, 1] scale;
# ~0.003 corresponds to -50 dBFS. Normal speech sits around -25 dBFS
# (0.056 linear), so this gate is well below any real wake utterance.
_SILENCE_RMS_FLOOR = 0.003

# Shared torchaudio mel front-end — lazy because torchaudio import is
# heavy and not every process running this module will need it.
_mel_transform = None
_db_transform = None


def _ensure_mel():
    """Build the shared torchaudio mel transforms on first call.

    Identical config to ``training.MelFrontend`` — the ONNX model was
    exported expecting log-mel input at exactly these shapes.
    """
    global _mel_transform, _db_transform
    if _mel_transform is not None:
        return _mel_transform, _db_transform
    import torchaudio  # heavy; defer until first inference
    _mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=2.0,
    )
    _db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")
    return _mel_transform, _db_transform


def _compute_log_mel(samples: np.ndarray) -> np.ndarray:
    """Compute the (1, 1, N_MELS, T) log-mel features the trained ONNX
    model expects. T = 101 for a 1-second 16 kHz window with 10 ms hop.
    """
    import torch
    mel_t, db_t = _ensure_mel()
    audio = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)  # (1, samples)
    mel = mel_t(audio)              # (1, n_mels, T)
    log_mel = db_t(mel)             # (1, n_mels, T)
    return log_mel.unsqueeze(1).numpy()  # (1, 1, n_mels, T)


@dataclass(frozen=True, slots=True)
class WakeDetection:
    """One positive detection. Returned by ``WakeWordService.feed``."""

    avatar_id: str
    phrase: str
    score: float
    source_id: str
    t: float


class WakeWordDetector:
    """One trained model + its ORT session + per-source rolling buffers.

    Stateless across sources at the model level; per-source state lives
    in three dicts keyed by ``source_id`` so a single detector can
    handle the browser mic, a fabric peer, and future device inputs in
    parallel without crosstalk.
    """

    def __init__(
        self,
        avatar_id: str,
        phrase: str,
        model_path: str | Path,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        import onnxruntime as ort
        self.avatar_id = avatar_id
        self.phrase = phrase
        self.threshold = float(threshold)
        self.model_path = str(model_path)
        # File mtime at load time — surfaces to ``WakeWordService.enable_
        # model`` so a personal-voice retrain (which overwrites the ONNX
        # at the same path) actually swaps the in-memory session instead
        # of silently reusing the prior detector. Without this the user
        # retrains, the cache hit short-circuits, and the OLD model
        # keeps running — making the retrain UI feel broken.
        try:
            self.loaded_mtime = Path(self.model_path).stat().st_mtime
        except OSError:
            self.loaded_mtime = 0.0
        # CPU only — inference is ~5 ms; GPU dispatch overhead would
        # negate the win and steal cycles from the main voice pipeline.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3  # warnings only
        self._session = ort.InferenceSession(
            self.model_path, sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._buffers: dict[str, np.ndarray] = {}
        self._since_last_infer: dict[str, int] = {}
        self._last_detect_at: dict[str, float] = {}
        # Per-source rolling diagnostic — max sigmoid score + max RMS seen
        # since the last telemetry flush. Lets the route handler emit a
        # periodic "audio is arriving, here's what scores it's producing"
        # log without spamming. Reset by ``flush_diagnostics``.
        self._diag_max_score: dict[str, float] = {}
        self._diag_max_rms: dict[str, float] = {}
        self._diag_infers: dict[str, int] = {}
        log.info(
            "wake_word_detector_loaded",
            avatar_id=avatar_id, phrase=phrase,
            threshold=self.threshold, model=self.model_path,
        )

    def _state_for(self, source_id: str) -> tuple[np.ndarray, int]:
        buf = self._buffers.get(source_id)
        if buf is None:
            buf = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            self._buffers[source_id] = buf
            self._since_last_infer[source_id] = 0
        return buf, self._since_last_infer[source_id]

    def feed(
        self,
        samples_f32: np.ndarray,
        source_id: str,
    ) -> WakeDetection | None:
        """Append ``samples_f32`` to the source's rolling buffer; run
        inference when enough new samples have accumulated. Returns a
        :class:`WakeDetection` on a positive trigger past threshold and
        outside the refractory window, else None.
        """
        buf, since = self._state_for(source_id)
        n = len(samples_f32)
        if n >= WINDOW_SAMPLES:
            buf[:] = samples_f32[-WINDOW_SAMPLES:]
        else:
            buf[:-n] = buf[n:]
            buf[-n:] = samples_f32
        since += n
        if since < INFERENCE_HOP_SAMPLES:
            self._since_last_infer[source_id] = since
            return None
        self._since_last_infer[source_id] = 0

        # Refractory check before inference — saves the ORT round-trip
        # while a recent detection still holds the lock.
        now = time.monotonic()
        last_at = self._last_detect_at.get(source_id, 0.0)
        if now - last_at < REFRACTORY_SEC:
            return None

        # Silence gate — skip ORT entirely when the 1-second window is
        # essentially quiet. The model never saw silence in its
        # synthetic negative pool, so room-tone with no speech can
        # cross threshold (we observed scores 0.34 on idle rooms). One
        # np.std on 16k samples is cheap; this catches the dominant
        # false-positive class without retraining.
        rms = float(np.sqrt(np.mean(buf * buf)))
        if rms > self._diag_max_rms.get(source_id, 0.0):
            self._diag_max_rms[source_id] = rms
        if rms < _SILENCE_RMS_FLOOR:
            return None

        try:
            mel = _compute_log_mel(buf)
            outputs = self._session.run(None, {"mel": mel.astype(np.float32)})
            logits = outputs[0]
            score = float(1.0 / (1.0 + np.exp(-logits[0, 0])))
        except Exception as exc:
            log.warning(
                "wake_word_inference_error",
                avatar_id=self.avatar_id, error=str(exc),
            )
            return None

        self._diag_infers[source_id] = self._diag_infers.get(source_id, 0) + 1
        if score > self._diag_max_score.get(source_id, 0.0):
            self._diag_max_score[source_id] = score

        if score < self.threshold:
            return None

        self._last_detect_at[source_id] = now
        log.info(
            "wake_word_detected",
            avatar_id=self.avatar_id, phrase=self.phrase,
            source_id=source_id, score=round(score, 3),
        )
        return WakeDetection(
            avatar_id=self.avatar_id, phrase=self.phrase,
            score=score, source_id=source_id, t=time.time(),
        )

    def reset_source(self, source_id: str) -> None:
        self._buffers.pop(source_id, None)
        self._since_last_infer.pop(source_id, None)
        self._last_detect_at.pop(source_id, None)
        self._diag_max_score.pop(source_id, None)
        self._diag_max_rms.pop(source_id, None)
        self._diag_infers.pop(source_id, None)

    def flush_diagnostics(self, source_id: str) -> dict[str, float | int]:
        """Return + reset rolling diagnostic counters for ``source_id``.

        Used by the route handler to emit a periodic "audio arrived,
        scores looked like X" log without polluting the inference path.
        Empty dict for an unknown source.
        """
        if source_id not in self._diag_infers and source_id not in self._diag_max_rms:
            return {}
        out: dict[str, float | int] = {
            "infers": self._diag_infers.get(source_id, 0),
            "max_score": round(self._diag_max_score.get(source_id, 0.0), 3),
            "max_rms": round(self._diag_max_rms.get(source_id, 0.0), 4),
        }
        self._diag_max_score[source_id] = 0.0
        self._diag_max_rms[source_id] = 0.0
        self._diag_infers[source_id] = 0
        return out


class WakeWordService:
    """Multiplex of zero-or-more :class:`WakeWordDetector`s, scoped by
    source.

    Each source (a WS connection, a fabric peer, etc) declares its own
    subset of avatar_ids via :meth:`subscribe_source`. ``feed(pcm,
    source_id)`` only routes to that source's detectors — so when one
    WS asks for ``wake-hey-jarvis`` and another for ``wake-hey-cortana``,
    their detections don't bleed across, and disconnecting one source
    doesn't leave its detectors firing for the next.

    The detector instances themselves are still pooled at the service
    level (one ONNX session per model, shared by all subscribers) — only
    routing is per-source.

    The service is meant to be a process singleton on ``app.state``.
    Use :func:`get_or_create_service` to fetch/create it.
    """

    def __init__(self) -> None:
        # Pool of loaded detectors, keyed by avatar_id. Multiple sources
        # share one detector instance — only routing diverges.
        self._detectors: dict[str, WakeWordDetector] = {}
        # source_id → set of avatar_ids that source is subscribed to.
        # ``feed`` reads this to scope detection per source.
        self._subscriptions: dict[str, set[str]] = {}

    # ── Registration ────────────────────────────────────────────────

    def enable_model(
        self,
        avatar_id: str,
        phrase: str,
        model_path: str | Path,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        """Load (or reload) a detector for ``avatar_id``.

        Idempotent when called with the same model_path + threshold + the
        ONNX file on disk hasn't been overwritten since the prior load.
        The mtime comparison is what makes personal-voice retrains
        actually swap the model — without it, the cached detector
        (loaded with the prior bake's weights) would keep firing because
        the path string + threshold match.
        """
        existing = self._detectors.get(avatar_id)
        if existing is not None \
                and existing.model_path == str(model_path) \
                and abs(existing.threshold - float(threshold)) < 1e-6:
            try:
                disk_mtime = Path(model_path).stat().st_mtime
            except OSError:
                disk_mtime = 0.0
            if disk_mtime <= existing.loaded_mtime + 1e-3:
                return
            log.info(
                "wake_word_detector_reloading_changed_file",
                avatar_id=avatar_id, path=str(model_path),
                loaded_mtime=existing.loaded_mtime, disk_mtime=disk_mtime,
            )
        self._detectors[avatar_id] = WakeWordDetector(
            avatar_id=avatar_id, phrase=phrase,
            model_path=model_path, threshold=threshold,
        )

    def disable_model(self, avatar_id: str) -> None:
        if avatar_id in self._detectors:
            del self._detectors[avatar_id]
            log.info("wake_word_detector_disabled", avatar_id=avatar_id)
        # Also drop from any source's subscription list so disabled
        # models can't leak via stale subscriptions.
        for ids in self._subscriptions.values():
            ids.discard(avatar_id)

    def active_models(self) -> list[dict]:
        return [
            {"avatar_id": d.avatar_id, "phrase": d.phrase,
             "threshold": d.threshold, "model_path": d.model_path}
            for d in self._detectors.values()
        ]

    # ── Source subscription ─────────────────────────────────────────

    def subscribe_source(self, source_id: str, avatar_ids: list[str]) -> list[str]:
        """Set which loaded detectors apply to ``source_id``.

        Replaces any prior subscription for the same source. Returns the
        list of avatar_ids that actually got subscribed (skips ids that
        aren't loaded — typically because they failed the quality gate
        in :func:`load_models_from_db` or were never enabled at all).

        After this call, ``feed(pcm, source_id)`` will only route to
        these detectors. Disconnecting via :meth:`reset_source` clears
        the subscription — so a model that one source dropped does
        not continue firing for the next.
        """
        active = [aid for aid in avatar_ids if aid in self._detectors]
        self._subscriptions[source_id] = set(active)
        log.debug(
            "wake_word_source_subscribed",
            source_id=source_id, requested=list(avatar_ids), active=active,
        )
        return active

    # ── Streaming ───────────────────────────────────────────────────

    def feed(self, pcm_bytes: bytes, source_id: str) -> list[WakeDetection]:
        """Feed PCM16 16 kHz mono frames to this source's subscribed
        detectors only.

        Returns the list of detections that fired this call (zero or
        more — typically zero, occasionally one, rarely more if
        multiple wake words happened to land in the same hop).
        """
        subs = self._subscriptions.get(source_id)
        if not subs:
            return []
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        out: list[WakeDetection] = []
        for avatar_id in subs:
            det = self._detectors.get(avatar_id)
            if det is None:
                continue
            d = det.feed(samples, source_id)
            if d is not None:
                out.append(d)
        return out

    def reset_source(self, source_id: str) -> None:
        """Drop per-source rolling state + subscription. Call on source
        disconnect — without this, a detector keeps a stale rolling
        buffer for that source plus an orphaned subscription that no
        one will ever clear.
        """
        for det in self._detectors.values():
            det.reset_source(source_id)
        self._subscriptions.pop(source_id, None)

    def flush_diagnostics(self, source_id: str) -> dict[str, dict]:
        """Aggregate per-source diagnostic counters across all detectors
        subscribed by ``source_id``. Returns ``{avatar_id: {...}}``.
        """
        subs = self._subscriptions.get(source_id) or set()
        out: dict[str, dict] = {}
        for avatar_id in subs:
            det = self._detectors.get(avatar_id)
            if det is None:
                continue
            d = det.flush_diagnostics(source_id)
            if d:
                out[avatar_id] = d
        return out

    def dispose(self) -> None:
        self._detectors.clear()
        self._subscriptions.clear()


# ── Module-level helpers ───────────────────────────────────────────────

def get_or_create_service(app_state) -> WakeWordService:
    """Singleton accessor — one service per process, hung off app.state.

    Lazy because not every deployment loads wake-word models on boot
    (e.g., dev runs without the bake).
    """
    svc = getattr(app_state, "wake_word_service", None)
    if svc is None:
        svc = WakeWordService()
        app_state.wake_word_service = svc
        log.info("wake_word_service_created")
    return svc


async def load_models_from_db(
    app_state,
    avatar_ids: list[str] | None = None,
) -> list[str]:
    """Enable detectors from the ``wake_word_models`` table.

    If ``avatar_ids`` is provided, only those rows are loaded;
    otherwise every row in the table is loaded. Returns the list of
    avatar_ids actually enabled (skips rows whose model file is
    missing on disk).
    """
    sm = getattr(app_state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    if backend is None:
        log.warning("wake_word_load_no_state_manager")
        return []

    svc = get_or_create_service(app_state)
    conn = backend.conn

    if avatar_ids:
        placeholders = ",".join("?" * len(avatar_ids))
        sql = (
            f"SELECT avatar_id, phrase, model_path, train_metrics "
            f"FROM wake_word_models WHERE avatar_id IN ({placeholders})"
        )
        cur = await conn.execute(sql, tuple(avatar_ids))
    else:
        sql = (
            "SELECT avatar_id, phrase, model_path, train_metrics "
            "FROM wake_word_models"
        )
        cur = await conn.execute(sql)
    rows = await cur.fetchall()
    await cur.close()

    loaded: list[str] = []
    for row in rows:
        avatar_id, phrase, model_path, metrics_json = row[0], row[1], row[2], row[3]
        if not model_path or not Path(model_path).exists():
            log.warning(
                "wake_word_model_missing",
                avatar_id=avatar_id, model_path=model_path,
            )
            continue
        try:
            metrics = json.loads(metrics_json) if metrics_json else {}
        except Exception:
            metrics = {}
        # Quality gate — refuse models that didn't learn to discriminate.
        # Without this, a low-val-acc model fires on background noise +
        # the avatar's own TTS, creating a call-after-call feedback loop.
        val_acc = float(metrics.get("best_val_acc") or 0.0)
        pos_count = int(metrics.get("positives_count") or 0)
        if val_acc < _MIN_VAL_ACC_FOR_USE or pos_count < _MIN_POSITIVES_FOR_USE:
            log.warning(
                "wake_word_model_quality_too_low",
                avatar_id=avatar_id, phrase=phrase,
                val_acc=round(val_acc, 3), positives=pos_count,
                min_val_acc=_MIN_VAL_ACC_FOR_USE,
                min_positives=_MIN_POSITIVES_FOR_USE,
                note=(
                    "model excluded from detection — retrain with the "
                    "full data budget (504 positives, 1050 negatives) "
                    "or pick a higher-quality phrase."
                ),
            )
            continue
        raw_threshold = float(metrics.get("best_threshold", _DEFAULT_THRESHOLD))
        pipeline = str(metrics.get("negatives_pipeline") or "synthetic_only")
        floor = (
            _MIN_RUNTIME_THRESHOLD_REAL_AUDIO
            if pipeline == "real_audio"
            else _MIN_RUNTIME_THRESHOLD_SYNTHETIC
        )
        threshold = max(raw_threshold, floor)
        if threshold > raw_threshold:
            log.info(
                "wake_word_threshold_floored",
                avatar_id=avatar_id, phrase=phrase,
                stored=raw_threshold, runtime_floor=floor,
                applied=threshold, pipeline=pipeline,
                note=(
                    "F1-optimal threshold from training was below the "
                    "runtime minimum for this pipeline; clamping up."
                ),
            )
        try:
            svc.enable_model(
                avatar_id=avatar_id, phrase=phrase,
                model_path=model_path, threshold=threshold,
            )
            loaded.append(avatar_id)
        except Exception as exc:
            log.warning(
                "wake_word_enable_failed",
                avatar_id=avatar_id, error=str(exc),
            )
    return loaded


__all__ = [
    "WakeDetection",
    "WakeWordDetector",
    "WakeWordService",
    "get_or_create_service",
    "load_models_from_db",
]
