"""Tests for discovery clustering and frecency scoring."""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from augmentum.discovery.clustering import (
    ClusterData,
    _compute_centroid,
    _cosine_similarity,
    _generate_cluster_name,
    compute_depth_level,
    extract_signal_text,
)
from augmentum.discovery.frecency import (
    SHORT_HALF_LIFE_DAYS,
    compute_combined_frecency,
    compute_decay,
    compute_frecency_from_signals,
)

# -----------------------------------------------------------------------
# ClusterData
# -----------------------------------------------------------------------

class TestClusterData:
    def test_auto_generates_id(self) -> None:
        c = ClusterData()
        assert c.cluster_id.startswith("c_")
        assert len(c.cluster_id) == 14  # "c_" + 12 hex chars

    def test_preserves_explicit_id(self) -> None:
        c = ClusterData(cluster_id="c_abc123")
        assert c.cluster_id == "c_abc123"

    def test_default_frecency(self) -> None:
        c = ClusterData()
        assert c.frecency_short == 0.0
        assert c.frecency_long == 0.0

    def test_depth_level_clamped(self) -> None:
        c = ClusterData(depth_level=0)
        assert c.depth_level == 1
        c2 = ClusterData(depth_level=99)
        assert c2.depth_level == 5

    def test_depth_level_valid_range(self) -> None:
        for level in range(1, 6):
            c = ClusterData(depth_level=level)
            assert c.depth_level == level

    def test_timestamps_populated(self) -> None:
        c = ClusterData()
        assert c.created_at != ""
        assert c.updated_at != ""


# -----------------------------------------------------------------------
# Cosine similarity
# -----------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert math.isclose(_cosine_similarity(v, v), 1.0, abs_tol=1e-9)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert math.isclose(_cosine_similarity(a, b), 0.0, abs_tol=1e-9)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert math.isclose(_cosine_similarity(a, b), -1.0, abs_tol=1e-9)


# -----------------------------------------------------------------------
# Centroid
# -----------------------------------------------------------------------

class TestComputeCentroid:
    def test_single_vector(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert _compute_centroid([v]) == v

    def test_average(self) -> None:
        v1 = [2.0, 4.0]
        v2 = [4.0, 6.0]
        result = _compute_centroid([v1, v2])
        assert math.isclose(result[0], 3.0)
        assert math.isclose(result[1], 5.0)

    def test_empty_returns_empty(self) -> None:
        assert _compute_centroid([]) == []


# -----------------------------------------------------------------------
# Cluster name
# -----------------------------------------------------------------------

class TestGenerateClusterName:
    def test_short_text(self) -> None:
        assert _generate_cluster_name("Hello world") == "Hello world"

    def test_long_text_truncated_to_8_words(self) -> None:
        text = "one two three four five six seven eight nine ten"
        result = _generate_cluster_name(text)
        assert len(result.split()) == 8

    def test_max_55_chars(self) -> None:
        text = "longword " * 10
        result = _generate_cluster_name(text.strip())
        assert len(result) <= 55


# -----------------------------------------------------------------------
# Depth level
# -----------------------------------------------------------------------

class TestComputeDepthLevel:
    def test_empty_signals(self) -> None:
        assert compute_depth_level([]) == 1

    def test_single_low_type(self) -> None:
        signals = [{"signal_type": "page_visit"}]
        assert compute_depth_level(signals) == 1

    def test_two_types_low_weight(self) -> None:
        signals = [
            {"signal_type": "page_visit"},
            {"signal_type": "search_query"},
        ]
        assert compute_depth_level(signals) == 2

    def test_diverse_high_weight(self) -> None:
        signals = [
            {"signal_type": "page_visit"},
            {"signal_type": "discuss"},
            {"signal_type": "note_save"},
            {"signal_type": "ai_action"},
            {"signal_type": "video_watch"},
            {"signal_type": "video_watch"},
            {"signal_type": "video_watch"},
            {"signal_type": "note_save"},
        ]
        assert compute_depth_level(signals) >= 4

    def test_level_5_requires_4_types_and_high_weight(self) -> None:
        signals = [
            {"signal_type": "note_save"},  # 3.0
            {"signal_type": "note_save"},  # 3.0
            {"signal_type": "discuss"},     # 2.5
            {"signal_type": "discuss"},     # 2.5
            {"signal_type": "ai_action"},   # 2.0
            {"signal_type": "ai_action"},   # 2.0
            {"signal_type": "video_watch"}, # 1.0
        ]
        # 16.0 total, 4 types
        assert compute_depth_level(signals) == 5


# -----------------------------------------------------------------------
# Extract signal text
# -----------------------------------------------------------------------

class TestExtractSignalText:
    def test_page_visit_uses_title(self) -> None:
        s = {"signal_type": "page_visit", "source_title": "My Article"}
        assert extract_signal_text(s) == "My Article"

    def test_search_query_uses_metadata(self) -> None:
        s = {
            "signal_type": "search_query",
            "source_title": "",
            "metadata": {"query": "python asyncio"},
        }
        assert extract_signal_text(s) == "python asyncio"

    def test_video_includes_channel(self) -> None:
        s = {
            "signal_type": "video_watch",
            "source_title": "Cool Video",
            "metadata": {"channel": "TechChannel"},
        }
        assert "Cool Video" in extract_signal_text(s)
        assert "TechChannel" in extract_signal_text(s)


# -----------------------------------------------------------------------
# Frecency — decay
# -----------------------------------------------------------------------

class TestFrecencyDecay:
    def test_decay_at_zero(self) -> None:
        assert math.isclose(compute_decay(0, half_life_days=7), 1.0)

    def test_decay_at_half_life(self) -> None:
        hours = SHORT_HALF_LIFE_DAYS * 24
        assert math.isclose(compute_decay(hours, half_life_days=SHORT_HALF_LIFE_DAYS), 0.5, abs_tol=1e-9)

    def test_old_signal_very_small(self) -> None:
        hours = 90 * 24  # 90 days
        val = compute_decay(hours, half_life_days=SHORT_HALF_LIFE_DAYS)
        assert val < 0.1


# -----------------------------------------------------------------------
# Frecency — combined
# -----------------------------------------------------------------------

class TestCombinedFrecency:
    def test_formula(self) -> None:
        result = compute_combined_frecency(10.0, 5.0)
        assert math.isclose(result, 10.0 * 0.6 + 5.0 * 0.4)

    def test_dampened(self) -> None:
        normal = compute_combined_frecency(10.0, 5.0)
        dampened = compute_combined_frecency(10.0, 5.0, dampened=True)
        assert math.isclose(dampened, normal * 0.3)


# -----------------------------------------------------------------------
# Frecency — from signals
# -----------------------------------------------------------------------

class TestFrecencyFromSignals:
    def test_recent_signal(self) -> None:
        now = datetime.now(UTC)
        signals = [
            {
                "signal_type": "page_visit",
                "created_at": now.isoformat(),
            }
        ]
        short, long = compute_frecency_from_signals(signals, now=now)
        # At t=0, decay=1.0, weight=0.5
        assert math.isclose(short, 0.5, abs_tol=1e-6)
        assert math.isclose(long, 0.5, abs_tol=1e-6)

    def test_old_signal_has_lower_frecency(self) -> None:
        now = datetime.now(UTC)
        old = now - timedelta(days=30)
        signals = [
            {
                "signal_type": "page_visit",
                "created_at": old.isoformat(),
            }
        ]
        short, long = compute_frecency_from_signals(signals, now=now)
        # Short half-life: 30 days / 7 day half-life => very decayed
        assert short < 0.05
        # Long half-life: 30 days / 30 day half-life => ~0.25
        assert math.isclose(long, 0.5 * 0.5, abs_tol=0.01)

    def test_empty_signals(self) -> None:
        short, long = compute_frecency_from_signals([])
        assert short == 0.0
        assert long == 0.0

    def test_same_domain_signals_saturate(self) -> None:
        """10 signals from one domain should weigh ~3x, not 10x."""
        now = datetime.now(UTC)
        signals = [
            {
                "signal_type": "video_watch",  # weight 1.0
                "created_at": now.isoformat(),
                "source_domain": "youtube.com",
            }
            for _ in range(10)
        ]
        short, _ = compute_frecency_from_signals(signals, now=now)
        # 10 fresh signals would naively sum to 10.0. With log(11)/10 saturation
        # the total lands near log(11) ≈ 2.4 — well under 10, well over 1.
        assert 2.0 < short < 4.0

    def test_diverse_domains_do_not_saturate(self) -> None:
        """5 signals from 5 different domains should weigh ~5x."""
        now = datetime.now(UTC)
        signals = [
            {
                "signal_type": "video_watch",
                "created_at": now.isoformat(),
                "source_domain": f"site-{i}.example",
            }
            for i in range(5)
        ]
        short, _ = compute_frecency_from_signals(signals, now=now)
        # Each domain has n=1 → factor=1.0 → full weight, total = 5 * 1.0 = 5.0.
        assert math.isclose(short, 5.0, abs_tol=1e-6)

    def test_mixed_saturation(self) -> None:
        """Big binge from one source + a few diverse signals — diverse ones survive."""
        now = datetime.now(UTC)
        signals = [
            {"signal_type": "video_watch", "created_at": now.isoformat(),
             "source_domain": "binge.example"}
            for _ in range(50)
        ] + [
            {"signal_type": "page_visit", "created_at": now.isoformat(),
             "source_domain": "diverse-a.example"},
            {"signal_type": "page_visit", "created_at": now.isoformat(),
             "source_domain": "diverse-b.example"},
        ]
        short, _ = compute_frecency_from_signals(signals, now=now)
        # Binge saturates to ~log(51) ≈ 3.93; two diverse @ 0.5 weight each = 1.0.
        # Total should land in the 4.5-5.5 range — binge does NOT dominate.
        assert 4.0 < short < 6.0
