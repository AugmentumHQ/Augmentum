"""Tests for augmentum.discovery.recommender."""
from __future__ import annotations

import pytest

from augmentum.discovery.recommender import build_search_query, distribute_slots


# ---------------------------------------------------------------------------
# distribute_slots
# ---------------------------------------------------------------------------

class TestDistributeSlots:
    def test_default_15(self):
        slots = distribute_slots()
        assert slots["core"] + slots["frontier"] + slots["adjacent"] == 15
        assert slots["core"] == 9   # round(15 * 0.6) = 9
        assert slots["frontier"] == 4  # round(15 * 0.27) = 4
        assert slots["adjacent"] == 2  # 15 - 9 - 4 = 2

    def test_small_total(self):
        slots = distribute_slots(3)
        # Each zone gets at least 1
        assert slots["core"] >= 1
        assert slots["frontier"] >= 1
        assert slots["adjacent"] >= 1

    def test_all_zones_present(self):
        for total in (5, 10, 15, 20, 30):
            slots = distribute_slots(total)
            assert all(v >= 1 for v in slots.values()), f"Zero slot at total={total}"


# ---------------------------------------------------------------------------
# build_search_query
# ---------------------------------------------------------------------------

class TestBuildSearchQuery:
    def test_contains_cluster_name(self):
        query = build_search_query("machine learning")
        assert "machine learning" in query

    def test_depth_1_modifiers(self):
        for _ in range(20):
            query = build_search_query("rust", depth_level=1)
            assert any(
                mod in query
                for mod in ["introduction to", "what is", "beginner guide"]
            )

    def test_depth_5_modifiers(self):
        for _ in range(20):
            query = build_search_query("rust", depth_level=5)
            assert any(
                mod in query
                for mod in ["production", "scaling", "state of the art"]
            )

    def test_clamps_depth(self):
        # Depth below 1 uses level 1
        q_low = build_search_query("rust", depth_level=-5)
        assert any(
            mod in q_low
            for mod in ["introduction to", "what is", "beginner guide"]
        )
        # Depth above 5 uses level 5
        q_high = build_search_query("rust", depth_level=99)
        assert any(
            mod in q_high
            for mod in ["production", "scaling", "state of the art"]
        )

    def test_strips_trailing_period(self):
        query = build_search_query("machine learning.")
        assert not query.endswith(".")
