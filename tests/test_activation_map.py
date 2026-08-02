"""Tests for the Activation Neural Map."""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "engine"))

from activation_map import ActivationMap, ActivationMapManager, fingerprint_logits


def test_fingerprint_deterministic():
    """Same logits produce same fingerprint."""
    logits = np.random.randn(32000).astype(np.float32)
    fp1 = fingerprint_logits(logits)
    fp2 = fingerprint_logits(logits)
    assert fp1 == fp2


def test_fingerprint_different_logits():
    """Different logits produce different fingerprints."""
    logits_a = np.random.randn(32000).astype(np.float32)
    logits_b = np.random.randn(32000).astype(np.float32)
    assert fingerprint_logits(logits_a) != fingerprint_logits(logits_b)


def test_record_and_predict():
    """Basic record -> predict cycle."""
    with tempfile.TemporaryDirectory() as tmp:
        m = ActivationMap(db_path=os.path.join(tmp, "test.db"), min_confidence=0.5)
        fp = b"\x00" * 16
        tokens = [100, 200, 300]

        m.record(fp, tokens)
        assert m.predict(fp) is None  # total_observations=1, needs >=3

        m.record(fp, tokens)
        m.record(fp, tokens)
        draft = m.predict(fp)
        assert draft == tokens, f"Expected {tokens}, got {draft}"


def test_hebbian_strengthening():
    """Repeated observations increase confidence."""
    with tempfile.TemporaryDirectory() as tmp:
        m = ActivationMap(db_path=os.path.join(tmp, "test.db"))
        fp = b"\x01" * 16
        tokens = [42, 43, 44]

        for _ in range(10):
            m.record(fp, tokens)

        entry = m._cache[fp]
        assert entry.total_observations == 10
        assert entry.confidence > 0.9


def test_hebbian_distribution():
    """Multiple different continuations build a distribution."""
    with tempfile.TemporaryDirectory() as tmp:
        m = ActivationMap(db_path=os.path.join(tmp, "test.db"))
        fp = b"\x02" * 16

        for _ in range(5):
            m.record(fp, [100, 200])
        for _ in range(3):
            m.record(fp, [999, 888])

        entry = m._cache[fp]
        assert entry.total_observations == 8
        assert len(entry.distribution) == 2
        assert entry.distribution[(100, 200)] == 5
        assert entry.distribution[(999, 888)] == 3
        assert entry.confidence < 0.7  # 5/8 = 0.625


def test_distribution_dominance():
    """When a new pattern dominates, it becomes the top prediction."""
    with tempfile.TemporaryDirectory() as tmp:
        m = ActivationMap(db_path=os.path.join(tmp, "test.db"), min_confidence=0.3)
        fp = b"\x03" * 16

        m.record(fp, [100])
        m.record(fp, [100])
        m.record(fp, [200])
        m.record(fp, [200])
        m.record(fp, [200])

        entry = m._cache[fp]
        assert entry.top_continuation == [200]  # 3 vs 2
        assert entry.confidence == 0.6  # 3/5


def test_temperature_sampling():
    """temp>0 samples from distribution instead of always picking top."""
    with tempfile.TemporaryDirectory() as tmp:
        m = ActivationMap(db_path=os.path.join(tmp, "test.db"), min_confidence=0.3)
        fp = b"\x06" * 16

        # 50/50 split
        for _ in range(50):
            m.record(fp, [100])
        for _ in range(50):
            m.record(fp, [200])

        # With temp=0, always returns the first one found (or either, both count=50)
        greedy = m.predict(fp, temperature=0.0)
        assert greedy in ([100], [200])

        # With temp=1.0, should sample both over many trials
        np.random.seed(42)
        seen = set()
        for _ in range(100):
            result = m.predict(fp, temperature=1.0)
            if result:
                seen.add(tuple(result))
        assert len(seen) == 2, f"Expected both continuations sampled, got {seen}"


def test_get_draft_distribution():
    """get_draft_distribution returns full probability table."""
    with tempfile.TemporaryDirectory() as tmp:
        m = ActivationMap(db_path=os.path.join(tmp, "test.db"))
        fp = b"\x07" * 16

        for _ in range(7):
            m.record(fp, [10, 20])
        for _ in range(3):
            m.record(fp, [30, 40])

        dist = m.get_draft_distribution(fp)
        assert dist is not None
        assert len(dist) == 2
        # Sorted by frequency descending
        assert dist[0][0] == [10, 20]
        assert abs(dist[0][1] - 0.7) < 0.01
        assert dist[1][0] == [30, 40]
        assert abs(dist[1][1] - 0.3) < 0.01


def test_persistence():
    """Map survives save/load cycle with distribution data."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")

        m1 = ActivationMap(db_path=db, min_confidence=0.5)
        fp = b"\x04" * 16
        for _ in range(5):
            m1.record(fp, [10, 20, 30])
        for _ in range(2):
            m1.record(fp, [40, 50])
        m1.save()

        m2 = ActivationMap(db_path=db, min_confidence=0.5)
        assert len(m2._cache) == 1
        entry = m2._cache[fp]
        assert entry.total_observations == 7
        assert len(entry.distribution) == 2
        draft = m2.predict(fp)
        assert draft == [10, 20, 30]  # most frequent


def test_stats():
    """Stats reflect actual usage."""
    with tempfile.TemporaryDirectory() as tmp:
        m = ActivationMap(db_path=os.path.join(tmp, "test.db"), min_confidence=0.5)
        fp = b"\x05" * 16

        for _ in range(5):
            m.record(fp, [1, 2, 3])

        m.predict(fp)
        m.predict(b"\xff" * 16)

        s = m.stats()
        assert s["entries"] == 1
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["observation_count"] == 5
        assert s["predictions_made"] == 1


def test_eviction():
    """Eviction removes lowest-confidence entries."""
    with tempfile.TemporaryDirectory() as tmp:
        m = ActivationMap(db_path=os.path.join(tmp, "test.db"), max_entries=10)

        for i in range(15):
            fp = bytes([i]) + b"\x00" * 15
            m.record(fp, [i])

        assert len(m._cache) <= 10


def test_manager_per_model_isolation():
    """Manager keeps separate maps per model."""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = ActivationMapManager(base_dir=tmp, min_confidence=0.5)

        mgr.load_model("ModelA-7B-Q4")
        fp_a = b"\xaa" * 16
        for _ in range(5):
            mgr.active.record(fp_a, [100, 200])
        assert mgr.active.predict(fp_a) == [100, 200]

        mgr.load_model("ModelB-13B-Q8")
        assert mgr.active.predict(fp_a) is None

        fp_b = b"\xbb" * 16
        for _ in range(5):
            mgr.active.record(fp_b, [300, 400])

        mgr.load_model("ModelA-7B-Q4")
        assert mgr.active.predict(fp_a) == [100, 200]
        assert mgr.active.predict(fp_b) is None

        stats = mgr.stats()
        assert "ModelA-7B-Q4" in stats["stored_models"]
        assert "ModelB-13B-Q8" in stats["stored_models"]


def test_manager_unload_saves():
    """Unloading saves the map, reloading restores it."""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = ActivationMapManager(base_dir=tmp, min_confidence=0.5)

        mgr.load_model("TestModel")
        fp = b"\xcc" * 16
        for _ in range(5):
            mgr.active.record(fp, [42, 43])
        mgr.unload_model()

        assert mgr.active is None

        mgr.load_model("TestModel")
        assert mgr.active.predict(fp) == [42, 43]


if __name__ == "__main__":
    tests = [
        test_fingerprint_deterministic,
        test_fingerprint_different_logits,
        test_record_and_predict,
        test_hebbian_strengthening,
        test_hebbian_distribution,
        test_distribution_dominance,
        test_temperature_sampling,
        test_get_draft_distribution,
        test_persistence,
        test_stats,
        test_eviction,
        test_manager_per_model_isolation,
        test_manager_unload_saves,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
