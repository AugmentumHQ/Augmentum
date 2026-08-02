"""Tests for WorkspaceCalibration — EMA + persistence + bounds."""

from __future__ import annotations

import json
from pathlib import Path

from augmentum.models.workspace_calibration import WorkspaceCalibration

# ---------------------------------------------------------------------------
# Pure-EMA invariants
# ---------------------------------------------------------------------------


class TestEMABehavior:
    def test_first_sample_seeds_directly(self, tmp_path: Path) -> None:
        """The first observation sets the ratio directly — no smoothing.

        Smoothing the first sample from the default 1.0 would bias
        every fresh install toward "no calibration" for ~10 loads.
        """
        cal = WorkspaceCalibration(tmp_path / "cal.json")
        cal.record("fa_on", observed_bytes=512.0 * 1024**2, predicted_bytes=384.0 * 1024**2)
        snap = cal.snapshot()
        # 512 / 384 = 1.333...
        assert snap["fa_on"]["samples"] == 1
        assert abs(snap["fa_on"]["ratio"] - (512.0 / 384.0)) < 1e-9

    def test_subsequent_samples_blend_via_ema(self, tmp_path: Path) -> None:
        """Second sample blends prev*(1-α) + new*α with α=0.3."""
        cal = WorkspaceCalibration(tmp_path / "cal.json")
        cal.record("fa_on", observed_bytes=384.0 * 1024**2, predicted_bytes=384.0 * 1024**2)  # ratio 1.0
        cal.record("fa_on", observed_bytes=576.0 * 1024**2, predicted_bytes=384.0 * 1024**2)  # ratio 1.5

        # Expected: 0.7 * 1.0 + 0.3 * 1.5 = 0.7 + 0.45 = 1.15
        snap = cal.snapshot()
        assert abs(snap["fa_on"]["ratio"] - 1.15) < 1e-9
        assert snap["fa_on"]["samples"] == 2

    def test_buckets_independent(self, tmp_path: Path) -> None:
        """Recording fa_on must not touch fa_off and vice-versa."""
        cal = WorkspaceCalibration(tmp_path / "cal.json")
        cal.record("fa_on", observed_bytes=300 * 1024**2, predicted_bytes=384 * 1024**2)  # 0.78
        cal.record("fa_off", observed_bytes=900 * 1024**2, predicted_bytes=640 * 1024**2)  # 1.40625

        snap = cal.snapshot()
        assert abs(snap["fa_on"]["ratio"] - (300 / 384)) < 1e-9
        assert abs(snap["fa_off"]["ratio"] - (900 / 640)) < 1e-9

    def test_zero_or_negative_inputs_drop(self, tmp_path: Path) -> None:
        """Bad inputs must not corrupt the EMA."""
        cal = WorkspaceCalibration(tmp_path / "cal.json")
        cal.record("fa_on", observed_bytes=0, predicted_bytes=384.0 * 1024**2)
        cal.record("fa_on", observed_bytes=384.0 * 1024**2, predicted_bytes=0)
        cal.record("fa_on", observed_bytes=-1, predicted_bytes=384.0 * 1024**2)
        assert cal.snapshot() == {}


# ---------------------------------------------------------------------------
# get_factor — sample-count gate + bounds
# ---------------------------------------------------------------------------


class TestGetFactor:
    def test_returns_one_below_min_samples(self, tmp_path: Path) -> None:
        """Below MIN_SAMPLES_TO_TRUST, callers must get 1.0 — uncalibrated."""
        cal = WorkspaceCalibration(tmp_path / "cal.json")
        for _ in range(WorkspaceCalibration.MIN_SAMPLES_TO_TRUST - 1):
            cal.record("fa_on", observed_bytes=192 * 1024**2, predicted_bytes=384 * 1024**2)  # 0.5
        assert cal.get_factor("fa_on") == 1.0

    def test_returns_calibration_at_or_above_min_samples(self, tmp_path: Path) -> None:
        """At MIN_SAMPLES_TO_TRUST samples the calibration kicks in."""
        cal = WorkspaceCalibration(tmp_path / "cal.json")
        # Drive the EMA to a known target by feeding the same ratio
        # MIN_SAMPLES_TO_TRUST times. After enough samples the EMA
        # converges arbitrarily close to the input.
        for _ in range(WorkspaceCalibration.MIN_SAMPLES_TO_TRUST):
            cal.record("fa_on", observed_bytes=288 * 1024**2, predicted_bytes=384 * 1024**2)  # 0.75
        factor = cal.get_factor("fa_on")
        # Within clamp range and close to 0.75.
        assert WorkspaceCalibration.RATIO_CLAMP_LOW <= factor <= WorkspaceCalibration.RATIO_CLAMP_HIGH
        assert factor == cal.snapshot()["fa_on"]["ratio"]

    def test_clamps_to_safe_range(self, tmp_path: Path) -> None:
        """Even after many extreme samples, the applied factor is clamped."""
        cal = WorkspaceCalibration(tmp_path / "cal.json")
        # Feed a stream of 0.3 ratios — ten of them converge well below
        # the 0.7 clamp.
        for _ in range(20):
            cal.record("fa_on", observed_bytes=int(384 * 0.3) * 1024**2,
                       predicted_bytes=384 * 1024**2)
        snap = cal.snapshot()
        assert snap["fa_on"]["ratio"] < WorkspaceCalibration.RATIO_CLAMP_LOW
        # But the APPLIED factor stays at the floor.
        assert cal.get_factor("fa_on") == WorkspaceCalibration.RATIO_CLAMP_LOW

        # And the inverse for the high clamp.
        cal2 = WorkspaceCalibration(tmp_path / "cal2.json")
        for _ in range(20):
            cal2.record("fa_off", observed_bytes=int(640 * 3.0) * 1024**2,
                        predicted_bytes=640 * 1024**2)
        snap2 = cal2.snapshot()
        assert snap2["fa_off"]["ratio"] > WorkspaceCalibration.RATIO_CLAMP_HIGH
        assert cal2.get_factor("fa_off") == WorkspaceCalibration.RATIO_CLAMP_HIGH


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------


class TestOutlierRejection:
    def test_outlier_does_not_update(self, tmp_path: Path) -> None:
        """A ratio below OUTLIER_MIN or above OUTLIER_MAX is dropped
        before touching the accumulator.

        Scenario: parser bug captures a transient 1 GiB peak and
        attributes it as compute buffer. Without rejection the EMA
        would lurch to a ratio of 2.6 from a single sample,
        permanently dragging future calibration up. With rejection
        the bad sample never moves the smoothed value.
        """
        cal = WorkspaceCalibration(tmp_path / "cal.json")
        # Seed with a normal ratio.
        cal.record("fa_on", observed_bytes=384 * 1024**2, predicted_bytes=384 * 1024**2)
        baseline_ratio = cal.snapshot()["fa_on"]["ratio"]

        # Now feed a wild outlier: 50× the predicted.
        cal.record(
            "fa_on",
            observed_bytes=50 * 384 * 1024**2,
            predicted_bytes=384 * 1024**2,
        )

        snap = cal.snapshot()
        # Sample count and ratio MUST NOT have moved.
        assert snap["fa_on"]["samples"] == 1
        assert snap["fa_on"]["ratio"] == baseline_ratio


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_round_trip_through_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "cal.json"
        cal1 = WorkspaceCalibration(path)
        for _ in range(8):
            cal1.record("fa_on", observed_bytes=300 * 1024**2,
                        predicted_bytes=384 * 1024**2)

        # New instance reads the same file.
        cal2 = WorkspaceCalibration(path)
        snap = cal2.snapshot()
        assert "fa_on" in snap
        assert snap["fa_on"]["samples"] == 8
        # Ratio survives within float-equality tolerance.
        assert abs(snap["fa_on"]["ratio"] - cal1.snapshot()["fa_on"]["ratio"]) < 1e-9

    def test_corrupt_json_starts_fresh(self, tmp_path: Path) -> None:
        """A garbled file must not crash construction."""
        path = tmp_path / "cal.json"
        path.write_text("not valid json{{{", encoding="utf-8")

        cal = WorkspaceCalibration(path)
        # Empty state — no samples carried over from the corruption.
        assert cal.snapshot() == {}
        # And we can still record + read normally.
        cal.record("fa_on", observed_bytes=384 * 1024**2,
                   predicted_bytes=384 * 1024**2)
        assert "fa_on" in cal.snapshot()

    def test_malformed_entries_are_dropped(self, tmp_path: Path) -> None:
        """Entries missing fields or with wrong types skip silently."""
        path = tmp_path / "cal.json"
        path.write_text(
            json.dumps({
                "buckets": {
                    "valid": {"ratio": 0.9, "samples": 7},
                    "missing_samples": {"ratio": 1.1},
                    "wrong_type": {"ratio": "not a number", "samples": 3},
                    "negative_ratio": {"ratio": -0.5, "samples": 2},
                }
            }),
            encoding="utf-8",
        )
        cal = WorkspaceCalibration(path)
        snap = cal.snapshot()
        assert "valid" in snap
        assert "missing_samples" not in snap
        assert "wrong_type" not in snap
        assert "negative_ratio" not in snap

    def test_missing_file_starts_empty(self, tmp_path: Path) -> None:
        cal = WorkspaceCalibration(tmp_path / "nonexistent.json")
        assert cal.snapshot() == {}

    def test_save_creates_parent_dir(self, tmp_path: Path) -> None:
        """If the calibration dir doesn't exist, persistence creates it."""
        path = tmp_path / "deep" / "nested" / "cal.json"
        cal = WorkspaceCalibration(path)
        cal.record("fa_on", observed_bytes=384 * 1024**2, predicted_bytes=384 * 1024**2)
        assert path.exists()
        # Round-trip works.
        cal2 = WorkspaceCalibration(path)
        assert "fa_on" in cal2.snapshot()

    def test_reset_clears_state_and_file(self, tmp_path: Path) -> None:
        path = tmp_path / "cal.json"
        cal = WorkspaceCalibration(path)
        cal.record("fa_on", observed_bytes=384 * 1024**2, predicted_bytes=384 * 1024**2)
        assert path.exists()

        cal.reset()
        assert cal.snapshot() == {}
        # File removed too — next instance starts from scratch on disk.
        assert not path.exists()
