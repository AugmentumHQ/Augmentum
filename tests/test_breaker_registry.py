"""Tests for ``augmentum.loops.breakers``.

Pins the threshold values that the coder's live behaviour depends on
(so a Phase 2 PR can't accidentally raise a breaker without flipping
its baseline), and verifies the tier-filtered registry returns the
right subset for LIGHT / MEDIUM / HEAVY.
"""
from __future__ import annotations

from augmentum.loops.breakers import (
    ACTION_STAGNATION_BREAK,
    ALL_BREAKERS,
    COORDINATION_ONLY_NUDGE_AT,
    FAILING_SHELL_NUDGE_AT,
    HYBRID_MAX_ITERS,
    HYBRID_MAX_ITERS_UNGATED,
    INSPECTION_STREAK_BREAK_AFTER_NUDGE,
    INSPECTION_STREAK_NUDGE,
    INSPECTION_TOOLS,
    MUTATING_TOOL_NAMES,
    NATIVE_SERIAL_TOOL_NAMES,
    NO_WRITE_PROGRESS_BREAK,
    PARALLEL_READ_TOOLS,
    SAME_FILE_EDIT_BREAK,
    SAME_VALIDATION_REPEAT_BREAK,
    SILENT_SUCCESS_NUDGE_AT,
    TASK_STALE_NUDGE_AT,
    TEST_FAILURE_STREAK_BREAK,
    VALIDATION_ERROR_STREAK_BREAK,
    Breaker,
    BreakerRegistry,
)
from augmentum.loops.tier import HEAVY, LIGHT, MEDIUM

# ── Threshold preservation ────────────────────────────────────────────


class TestThresholdsMatchHistoricalDefaults:
    """The threshold defaults are tuned values, not arbitrary numbers —
    each was set after seeing a specific failure mode in the wild
    (see phase_act.py comments). Pin them so a careless edit can't
    raise them without an intentional change to this test."""

    def test_validation_error_streak(self):
        assert VALIDATION_ERROR_STREAK_BREAK == 5

    def test_same_validation_repeat(self):
        assert SAME_VALIDATION_REPEAT_BREAK == 2

    def test_action_stagnation(self):
        assert ACTION_STAGNATION_BREAK == 20

    def test_test_failure_streak(self):
        assert TEST_FAILURE_STREAK_BREAK == 8

    def test_same_file_edit_break(self):
        assert SAME_FILE_EDIT_BREAK == 15

    def test_no_write_progress(self):
        assert NO_WRITE_PROGRESS_BREAK == 10

    def test_silent_success_nudge(self):
        assert SILENT_SUCCESS_NUDGE_AT == 3

    def test_failing_shell_nudge(self):
        assert FAILING_SHELL_NUDGE_AT == 4

    def test_task_stale_nudge(self):
        assert TASK_STALE_NUDGE_AT == 8

    def test_coordination_only_nudge(self):
        assert COORDINATION_ONLY_NUDGE_AT == 3

    def test_inspection_streak_nudge(self):
        assert INSPECTION_STREAK_NUDGE == 5

    def test_inspection_streak_break_after_nudge(self):
        assert INSPECTION_STREAK_BREAK_AFTER_NUDGE == 3

    def test_hybrid_max_iters(self):
        assert HYBRID_MAX_ITERS == 150

    def test_hybrid_max_iters_ungated(self):
        assert HYBRID_MAX_ITERS_UNGATED == 500


# ── Tier filtering ────────────────────────────────────────────────────


class TestRegistryFilter:
    def test_light_has_only_termination_gate(self):
        reg = BreakerRegistry.for_intensity(LIGHT)
        assert reg.names() == ("termination_quality_gate",)
        assert reg.max_iterations == 8

    def test_medium_loads_standard_high_signal_breakers(self):
        reg = BreakerRegistry.for_intensity(MEDIUM)
        names = set(reg.names())
        # Must include the spec's high-signal stop conditions
        assert "validation_error_streak" in names
        assert "same_validation_error_repeat" in names
        assert "action_stagnation_break" in names
        assert "failing_shell_nudge" in names
        assert "termination_quality_gate" in names
        # Must NOT include full-suite breakers
        assert "test_failure_streak" not in names
        assert "same_file_edit_break" not in names
        assert "inspection_loop_nudge" not in names
        assert reg.max_iterations == 25

    def test_heavy_loads_all_breakers(self):
        reg = BreakerRegistry.for_intensity(HEAVY)
        assert len(reg.names()) == len(ALL_BREAKERS)
        assert reg.max_iterations == 150

    def test_max_iterations_override_for_ungated(self):
        """``safeguards_enabled=False`` raises the ceiling without
        flipping intensity. The override knob is how the coder will
        wire it post-PR-2.4."""
        reg = BreakerRegistry.for_intensity(
            HEAVY, max_iterations_override=HYBRID_MAX_ITERS_UNGATED,
        )
        assert reg.max_iterations == 500


class TestRegistryLookup:
    def test_by_name_returns_match(self):
        reg = BreakerRegistry.for_intensity(HEAVY)
        b = reg.by_name("action_stagnation_break")
        assert b is not None
        assert isinstance(b, Breaker)
        assert b.threshold == 20

    def test_by_name_returns_none_for_unknown(self):
        reg = BreakerRegistry.for_intensity(HEAVY)
        assert reg.by_name("nonexistent_breaker") is None

    def test_filter_by_kind_break(self):
        reg = BreakerRegistry.for_intensity(HEAVY)
        breaks = reg.filter("break")
        assert all(b.kind == "break" for b in breaks)
        # MEDIUM has 3 break breakers + 2 nudges
        med = BreakerRegistry.for_intensity(MEDIUM)
        assert len(med.filter("break")) == 3

    def test_filter_by_kind_nudge(self):
        med = BreakerRegistry.for_intensity(MEDIUM)
        nudges = med.filter("nudge")
        assert len(nudges) == 2
        assert {n.name for n in nudges} == {
            "termination_quality_gate", "failing_shell_nudge",
        }


# ── Tool sets (used by both breakers + LoopRunner act loop) ───────────


class TestToolSets:
    def test_mutating_tools_set(self):
        assert frozenset({
            "code_edit", "code_edit_batch", "file_write", "apply_patch",
        }) == MUTATING_TOOL_NAMES

    def test_inspection_tools_does_not_include_shell_exec(self):
        """2026-04-22 removal — shell_exec is the legitimate
        build/install surface; counting it as inspection mis-fired
        the breaker on real work."""
        assert "shell_exec" not in INSPECTION_TOOLS
        assert "shell_read" in INSPECTION_TOOLS

    def test_native_serial_includes_mutating(self):
        assert MUTATING_TOOL_NAMES.issubset(NATIVE_SERIAL_TOOL_NAMES)

    def test_native_serial_includes_browser_evaluate(self):
        """browser_evaluate spawns Chromium per call — must stay
        serial so two parallel evaluates don't fight over the
        workspace's CPU/GPU budget."""
        assert "browser_evaluate" in NATIVE_SERIAL_TOOL_NAMES
        assert "browser_evaluate" not in PARALLEL_READ_TOOLS

    def test_native_serial_includes_wave2_browser_tools(self):
        """Every browser_* tool contends on the same Chromium/session
        state — the Wave-2 primitives (2026-07-02) serialize like
        their siblings. Required because browser_wait/browser_extract
        are in READ_ONLY_TOOLS: without serial membership the native
        loop would fan them out in the parallel-read wave."""
        assert "browser_wait" in NATIVE_SERIAL_TOOL_NAMES
        assert "browser_extract" in NATIVE_SERIAL_TOOL_NAMES
        assert "browser_fill_form" in NATIVE_SERIAL_TOOL_NAMES
        assert "browser_wait" not in PARALLEL_READ_TOOLS
        assert "browser_extract" not in PARALLEL_READ_TOOLS

    def test_observe_is_serial(self):
        """observe appends to /workspace/.augmentum/observations.jsonl
        via read-modify-write — two parallel observe calls would
        race on the file."""
        assert "observe" in NATIVE_SERIAL_TOOL_NAMES
        assert "observe" not in PARALLEL_READ_TOOLS

    def test_parallel_reads_are_read_only(self):
        """The hybrid loop fans these out in parallel — they must
        all be side-effect-free."""
        # http_request + db_inspect are read-only probes; the rest
        # are pure code/doc reads
        assert "http_request" in PARALLEL_READ_TOOLS
        assert "db_inspect" in PARALLEL_READ_TOOLS
        # No mutating tool in the parallel set
        assert MUTATING_TOOL_NAMES.isdisjoint(PARALLEL_READ_TOOLS)
