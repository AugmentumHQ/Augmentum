"""Self-calibrating workspace VRAM estimates.

Augmentum's load planner uses hand-tuned constants for prompt-eval
workspace size — 384 MiB with Flash Attention, 640 MiB without.
Those numbers came from peak-residency measurements on a handful of
dense Q4_K_M / Q8_0 GGUFs at batch 512. Real loads sometimes deviate:

* Architecture-specific quirks (MoE expert routing temporarily fans
  out the activations buffer mid-forward).
* Context-length scaling that the linear ``hidden_bytes * factor``
  estimator under-counts at very long contexts.
* CUDA driver versions that change the default scratch-pool slack.

T2-7 closes that gap with an exponential moving average of the
ratio between observed peak compute usage and the predicted reserve.
After enough samples accumulate, future loads scale the predicted
reserve by that EMA — over-predicting drops the reserve back to fit,
under-predicting pads it just enough to avoid OOM.

The calibration is persisted as JSON next to the model profile cache
so it survives restart and accumulates across sessions on the same
hardware. Per-bucket (FA-on / FA-off) so the two compute estimates
calibrate independently — the workspace shape with Flash Attention
is meaningfully different from without, and conflating the samples
would smear them.

Safety bounds: the EMA ratio is clamped to ``[0.7, 1.5]`` before
being applied so a single anomalous reading (a failed load that
truncates the parse, a rogue value from a future llama.cpp log
format change) can't push us into the OOM zone or starve us into
a layer-zero CPU-only fallback. Until ``MIN_SAMPLES_TO_TRUST``
samples have been collected, the factor returns 1.0 (no
calibration applied — original constants stand).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class WorkspaceCalibration:
    """JSON-backed EMA of observed/predicted compute reserve ratios.

    Buckets are arbitrary string keys — Augmentum uses ``"fa_on"`` and
    ``"fa_off"``. The class stays generic so future axes (architecture
    family, batch-size band) can plug in without a schema change.
    """

    # EMA blending weight for new observations. 0.3 means a single
    # outlier moves the smoothed value by ~30%; full convergence to a
    # new equilibrium takes ~10 samples. Conservative enough to ride
    # over noise, fast enough that the calibration adapts to a model-
    # family change within a few loads.
    EMA_ALPHA: float = 0.3

    # Number of samples collected before the calibration is applied.
    # Below this, ``get_factor`` returns 1.0 and predictions use the
    # uncalibrated baseline. Five is the smallest count where the EMA
    # has converged enough to be useful but small enough that a fresh
    # install starts adapting within an evening of casual use.
    MIN_SAMPLES_TO_TRUST: int = 5

    # Hard clamps on the applied factor. Calibration that drifts
    # outside this range is almost certainly a parser bug or a load
    # that captured non-comparable phases (e.g. a transient peak from
    # a tool the user isn't using) — clamp rather than letting it run
    # the load off the rails.
    RATIO_CLAMP_LOW: float = 0.7
    RATIO_CLAMP_HIGH: float = 1.5

    # Reject ratios outside this range as outliers BEFORE updating the
    # EMA. The clamps above protect the read path; this protects the
    # accumulator from being polluted by a single ridiculous sample.
    OUTLIER_MIN: float = 0.1
    OUTLIER_MAX: float = 10.0

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._buckets: dict[str, dict[str, float]] = {}
        self._load()

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def get_factor(self, bucket: str) -> float:
        """Calibration multiplier for ``bucket``.

        Returns 1.0 (no adjustment) when fewer than
        ``MIN_SAMPLES_TO_TRUST`` samples have been collected. Above
        that, returns the EMA ratio clamped to
        ``[RATIO_CLAMP_LOW, RATIO_CLAMP_HIGH]``.
        """
        entry = self._buckets.get(bucket)
        if entry is None:
            return 1.0
        if entry.get("samples", 0) < self.MIN_SAMPLES_TO_TRUST:
            return 1.0
        ratio = float(entry.get("ratio", 1.0))
        return max(self.RATIO_CLAMP_LOW, min(self.RATIO_CLAMP_HIGH, ratio))

    def record(
        self,
        bucket: str,
        *,
        observed_bytes: float,
        predicted_bytes: float,
    ) -> None:
        """Update the EMA for ``bucket`` from a load observation.

        Called from the manager's "model loaded" handler with the
        observed compute residual (actual VRAM minus model weights
        minus KV) and the predicted reserve we used for the load.
        Outlier ratios are dropped on the floor — they corrupt more
        than they teach.
        """
        if predicted_bytes <= 0 or observed_bytes <= 0:
            return
        ratio = float(observed_bytes) / float(predicted_bytes)
        if ratio < self.OUTLIER_MIN or ratio > self.OUTLIER_MAX:
            log.info(
                "workspace_calibration_outlier_dropped bucket=%s ratio=%.3f "
                "observed_mib=%d predicted_mib=%d",
                bucket,
                ratio,
                round(observed_bytes / 1024**2),
                round(predicted_bytes / 1024**2),
            )
            return

        entry = self._buckets.setdefault(bucket, {"ratio": 1.0, "samples": 0})
        prev_samples = int(entry.get("samples", 0))
        if prev_samples == 0:
            # First sample seeds the EMA directly; smoothing from
            # 1.0 would bias every fresh install toward "no
            # calibration" for ~10 loads.
            entry["ratio"] = ratio
        else:
            prev_ratio = float(entry.get("ratio", 1.0))
            entry["ratio"] = (
                (1.0 - self.EMA_ALPHA) * prev_ratio
                + self.EMA_ALPHA * ratio
            )
        entry["samples"] = prev_samples + 1
        self._save()

        log.info(
            "workspace_calibration_updated bucket=%s ratio=%.3f samples=%d "
            "observation_mib=%d predicted_mib=%d",
            bucket,
            entry["ratio"],
            int(entry["samples"]),
            round(observed_bytes / 1024**2),
            round(predicted_bytes / 1024**2),
        )

    def snapshot(self) -> dict[str, Any]:
        """Read-only view, primarily for diagnostics / tests."""
        return {
            bucket: {
                "ratio": float(entry.get("ratio", 1.0)),
                "samples": int(entry.get("samples", 0)),
                "applied_factor": self.get_factor(bucket),
            }
            for bucket, entry in self._buckets.items()
        }

    def reset(self) -> None:
        """Drop all samples — used by tests and an admin reset path."""
        self._buckets = {}
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("workspace_calibration_reset_failed error=%s", str(exc))

    # ---------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------

    def _load(self) -> None:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            log.warning(
                "workspace_calibration_load_failed path=%s error=%s",
                str(self._path),
                str(exc),
            )
            return

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.warning(
                "workspace_calibration_corrupt_json path=%s error=%s",
                str(self._path),
                str(exc),
            )
            return

        buckets = data.get("buckets") if isinstance(data, dict) else None
        if not isinstance(buckets, dict):
            return

        # Sanitize: only keep entries with the expected shape. A
        # malformed entry shouldn't poison the in-memory state.
        sanitized: dict[str, dict[str, float]] = {}
        for bucket, entry in buckets.items():
            if not isinstance(entry, dict):
                continue
            try:
                ratio = float(entry["ratio"])
                samples = int(entry["samples"])
            except (KeyError, TypeError, ValueError):
                continue
            if ratio <= 0 or samples < 0:
                continue
            sanitized[str(bucket)] = {"ratio": ratio, "samples": samples}
        self._buckets = sanitized

    def _save(self) -> None:
        # Atomic write so a crash mid-save can't leave a half-written
        # JSON the next load would reject.
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps({"buckets": self._buckets}, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except OSError as exc:
            # Persistence failure is non-fatal — the in-memory EMA
            # still works for the lifetime of the process. Logging
            # gives operators a head's-up if the model dir is read-only.
            log.warning(
                "workspace_calibration_save_failed path=%s error=%s",
                str(self._path),
                str(exc),
            )
