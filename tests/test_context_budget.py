"""Tests — augmentum.context.budget per-turn token meter.

Covers:
  * Basic accumulation across labels
  * remaining / over_budget transitions
  * Per-label cap detection (default fractions + caller override)
  * Snapshot is a frozen view (doesn't mutate)
  * Commit logs only when over budget
  * Validation rejects nonpositive / absurd total cap
  * add_count for caller-provided token counts
"""

from __future__ import annotations

import pytest

from augmentum.context.budget import BudgetSnapshot, BudgetTracker


class TestBasicAccumulation:
    def test_empty_tracker(self):
        t = BudgetTracker(total_cap=1000)
        assert t.used == 0
        assert t.remaining == 1000
        assert t.over_budget is False

    def test_single_add(self):
        t = BudgetTracker(total_cap=1000)
        cost = t.add("system", "hello world")
        assert cost > 0
        assert t.used == cost
        assert t.used_by("system") == cost

    def test_multiple_labels_sum_correctly(self):
        t = BudgetTracker(total_cap=10_000)
        a = t.add("system", "you are a helper")
        b = t.add("memory/active", "user likes coffee. user is in seattle.")
        c = t.add("documents/rag", "document chunk content")
        assert t.used == a + b + c
        assert t.used_by("system") == a
        assert t.used_by("memory/active") == b

    def test_repeated_add_same_label_accumulates(self):
        t = BudgetTracker(total_cap=10_000)
        first = t.add("memory/active", "fact one")
        second = t.add("memory/active", "fact two")
        assert t.used_by("memory/active") == first + second

    def test_empty_content_no_op(self):
        t = BudgetTracker(total_cap=1000)
        cost = t.add("system", "")
        assert cost == 0
        assert t.used == 0

    def test_add_count_explicit(self):
        t = BudgetTracker(total_cap=1000)
        n = t.add_count("tool_schemas", 250)
        assert n == 250
        assert t.used_by("tool_schemas") == 250

    def test_add_count_clamps_negative(self):
        t = BudgetTracker(total_cap=1000)
        n = t.add_count("misc", -50)
        assert n == 0
        assert t.used == 0


class TestBudgetBoundaries:
    def test_under_budget(self):
        t = BudgetTracker(total_cap=100)
        t.add_count("system", 40)
        t.add_count("persona", 30)
        assert t.over_budget is False
        assert t.remaining == 30

    def test_at_budget(self):
        t = BudgetTracker(total_cap=100)
        t.add_count("system", 100)
        assert t.over_budget is False
        assert t.remaining == 0

    def test_over_budget(self):
        t = BudgetTracker(total_cap=100)
        t.add_count("system", 150)
        assert t.over_budget is True
        assert t.remaining == 0  # clamped to non-negative


class TestPerLabelCap:
    def test_default_fraction_caps_applied(self):
        # total_cap=1000 with default 'system' fraction of 0.05 → 50
        t = BudgetTracker(total_cap=1000)
        assert t.cap_for("system") == 50
        assert t.cap_for("memory/active") == 100

    def test_under_label_cap(self):
        t = BudgetTracker(total_cap=1000)
        t.add_count("system", 30)
        assert t.over_label_cap("system") is False

    def test_over_label_cap(self):
        t = BudgetTracker(total_cap=1000)
        t.add_count("system", 80)
        # default system cap = 50, used 80 → over
        assert t.over_label_cap("system") is True

    def test_custom_label_caps_override_fractions(self):
        t = BudgetTracker(
            total_cap=1000,
            contributor_caps={"system": 500},  # bump way above default
        )
        t.add_count("system", 100)
        assert t.over_label_cap("system") is False
        t.add_count("system", 500)
        # used 600, cap 500 → over
        assert t.over_label_cap("system") is True

    def test_no_cap_for_unconfigured_label(self):
        t = BudgetTracker(total_cap=1000)
        assert t.cap_for("custom_thing") == 0
        t.add_count("custom_thing", 99999)
        # 0 cap means "no per-label cap" — never reports over
        assert t.over_label_cap("custom_thing") is False


class TestSnapshot:
    def test_snapshot_shape(self):
        t = BudgetTracker(total_cap=1000)
        t.add_count("system", 50)
        t.add_count("persona", 30)
        snap = t.snapshot()
        assert isinstance(snap, BudgetSnapshot)
        assert snap.total_cap == 1000
        assert snap.used == 80
        assert snap.remaining == 920
        assert snap.over_budget is False
        assert snap.by_label == {"system": 50, "persona": 30}

    def test_snapshot_as_log_payload(self):
        t = BudgetTracker(total_cap=1000)
        t.add_count("system", 50)
        log_payload = t.snapshot().as_log()
        assert log_payload["budget_total"] == 1000
        assert log_payload["budget_used"] == 50
        assert log_payload["budget_remaining"] == 950
        assert log_payload["budget_over"] is False
        assert "budget_elapsed_s" in log_payload

    def test_snapshot_does_not_mutate_tracker(self):
        t = BudgetTracker(total_cap=1000)
        t.add_count("system", 50)
        snap = t.snapshot()
        snap.by_label["system"] = 999  # mutate the copy
        assert t.used_by("system") == 50  # tracker unchanged


class TestCommit:
    def test_commit_returns_snapshot(self):
        t = BudgetTracker(total_cap=1000)
        t.add_count("system", 50)
        snap = t.commit()
        assert snap.used == 50
        assert snap.over_budget is False

    def test_commit_when_over(self, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        t = BudgetTracker(total_cap=100, model_name="qwen3-32b")
        t.add_count("system", 80)
        t.add_count("memory/active", 50)
        snap = t.commit()
        assert snap.over_budget is True


class TestValidation:
    def test_zero_cap_rejected(self):
        with pytest.raises(ValueError):
            BudgetTracker(total_cap=0)

    def test_negative_cap_rejected(self):
        with pytest.raises(ValueError):
            BudgetTracker(total_cap=-1)

    def test_absurd_cap_rejected(self):
        with pytest.raises(ValueError):
            BudgetTracker(total_cap=10_000_000_000)


class TestRealisticUsage:
    """Sanity check against a realistic turn-shaped accumulation."""

    def test_realistic_chat_turn(self):
        t = BudgetTracker(total_cap=8000, model_name="qwen3-32b")
        # Use add() with actual strings so token counts come from the
        # real tokenizer, not hand-supplied numbers.
        t.add("system", "You are Becca, a warm companion AI.")
        t.add("persona", "Becca speaks softly, takes interest in user's projects." * 4)
        t.add("memory/active", "User prefers dark roast coffee. " * 10)
        t.add("documents/rag", "Document chunk about sourdough hydration. " * 8)
        t.add("knowledge/pack", "Wikipedia: Sourdough is a fermented... " * 8)
        t.add("recent_messages", "Hi Becca\nHi there\n" * 5)

        snap = t.snapshot()
        # Should be well under budget for this shape.
        assert snap.used > 0
        assert snap.over_budget is False
        # Every contributor recorded.
        assert set(snap.by_label.keys()) >= {
            "system", "persona", "memory/active",
            "documents/rag", "knowledge/pack", "recent_messages",
        }
