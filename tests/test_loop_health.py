"""LoopHealthCoordinator — arbitration, suppression, telemetry."""

from __future__ import annotations

from augmentum.coder.loop_health import (
    NUDGE_PRIORITY,
    LoopHealthCoordinator,
    _priority,
)
from augmentum.loops.breakers import live_threshold


def _make() -> LoopHealthCoordinator:
    return LoopHealthCoordinator.create(
        threshold=live_threshold,
        tasks=[],
        tracked_read_tools=frozenset({"file_read", "code_grep"}),
    )


class TestArbitration:
    def test_no_pending_returns_none(self):
        h = _make()
        winner, suppressed = h.arbitrate()
        assert winner is None and suppressed == []

    def test_single_nudge_wins(self):
        h = _make()
        h.submit("silent_success_nudge", "body", {"streak": 3})
        winner, suppressed = h.arbitrate()
        assert winner.kind == "silent_success_nudge"
        assert winner.body == "body"
        assert suppressed == []

    def test_highest_priority_wins_rest_suppressed(self):
        h = _make()
        h.submit("single_read_nudge", "advisory")
        h.submit("same_file_edit_nudge", "corrective")
        h.submit("task_stale_nudge", "hygiene")
        winner, suppressed = h.arbitrate()
        assert winner.kind == "same_file_edit_nudge"
        assert [s.kind for s in suppressed] == [
            "task_stale_nudge", "single_read_nudge",
        ]

    def test_intervention_suppresses_all_nudges(self):
        h = _make()
        h.submit("same_file_edit_nudge", "corrective")
        h.note_intervention("loop_reorient")
        winner, suppressed = h.arbitrate()
        assert winner is None
        assert [s.kind for s in suppressed] == ["same_file_edit_nudge"]

    def test_iteration_state_resets(self):
        h = _make()
        h.note_intervention("loop_reorient")
        h.arbitrate()
        # Next iteration: intervention no longer suppresses.
        h.submit("probe_no_signal_nudge", "body")
        winner, _ = h.arbitrate()
        assert winner is not None

    def test_unknown_kind_sorts_last(self):
        h = _make()
        h.submit("brand_new_guard_nudge", "unknown")
        h.submit("single_read_nudge", "known-lowest")
        winner, suppressed = h.arbitrate()
        assert winner.kind == "single_read_nudge"
        assert suppressed[0].kind == "brand_new_guard_nudge"


class TestTelemetry:
    def test_counters_track_fired_suppressed_interventions(self):
        h = _make()
        h.submit("same_file_edit_nudge", "a")
        h.submit("single_read_nudge", "b")
        h.arbitrate()
        h.note_intervention("escalated_to_buddy")
        h.submit("probe_no_signal_nudge", "c")
        h.arbitrate()
        assert h.summary() == {
            "same_file_edit_nudge": 1,
            "suppressed:single_read_nudge": 1,
            "escalated_to_buddy": 1,
            "suppressed:probe_no_signal_nudge": 1,
        }

    def test_healthy_turn_summary_empty(self):
        h = _make()
        h.arbitrate()
        assert h.summary() == {}


class TestConstruction:
    def test_owns_all_trackers(self):
        h = _make()
        for attr in (
            "task_spine", "write_churn", "probe_signal",
            "command_carousel", "progress_ledger", "duplicate_calls",
            "code_intel",
        ):
            assert getattr(h, attr) is not None

    def test_priority_order_is_stable(self):
        # Delegated to deepseek-v4-flash via scripts/claude_delegate.py,
        # reviewed + applied by hand (dropped an unused pytest import).
        assert len(NUDGE_PRIORITY) == len(set(NUDGE_PRIORITY))
        assert _priority(NUDGE_PRIORITY[0]) == 0
        assert _priority("nonexistent_nudge") == len(NUDGE_PRIORITY)

    def test_priority_table_covers_known_guards(self):
        # Every nudge kind emitted from the native loop must rank.
        for kind in (
            "flaky_test_nudge", "same_file_edit_nudge",
            "identical_result_nudge", "duplicate_call_nudge",
            "command_carousel_nudge", "probe_no_signal_nudge",
            "progress_stall_nudge", "silent_success_nudge",
            "task_stale_nudge", "symbol_grep_nudge", "single_read_nudge",
        ):
            assert kind in NUDGE_PRIORITY, kind
