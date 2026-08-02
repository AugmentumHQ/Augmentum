"""Unit tests for ``augmentum.loops.ledger.ObservationLedger``.

Phase 2 / PR-2.2 extracted the four-bucket ledger out of CoderState
so the shared LoopRunner can own one directly. These tests pin the
behaviour the coder + (later) agentic loops depend on:

* validation-error dedup with signature tracking (the breaker
  ``same_validation_error_repeat`` reads ``repeat_count``)
* soft-failure TTL prune (the cross-turn ledger doesn't grow
  unboundedly when the model keeps retrying then stops)
* intent-keyed tool-call dedup + clear-by-path
* background-process dedup
* shared-reference invariant from :meth:`from_lists` so the
  CoderState wrapper keeps working
"""
from __future__ import annotations

import time

from augmentum.loops.ledger import (
    FAILURE_LEDGER_TTL_SECONDS,
    ObservationLedger,
)


# ── Validation errors ─────────────────────────────────────────────────


class TestValidationErrors:
    def test_dedup_bumps_count_keeps_one_entry(self):
        l = ObservationLedger()
        l.record_validation_error(tool_name="shell_exec", error="oops a")
        l.record_validation_error(tool_name="shell_exec", error="oops b")
        assert len(l.recent_validation_errors) == 1
        assert l.recent_validation_errors[0]["count"] == 2

    def test_repeat_count_resets_on_signature_change(self):
        l = ObservationLedger()
        l.record_validation_error(tool_name="shell_exec", error="missing arg 'cmd'.")
        l.record_validation_error(tool_name="shell_exec", error="missing arg 'cmd'.")
        assert l.recent_validation_errors[0]["repeat_count"] == 2
        # Different signature -> repeat_count resets
        l.record_validation_error(tool_name="shell_exec", error="bad timeout value.")
        assert l.recent_validation_errors[0]["repeat_count"] == 1

    def test_cap_kept_at_max(self):
        l = ObservationLedger()
        for i in range(6):
            l.record_validation_error(
                tool_name=f"tool_{i}", error="x", max_kept=3,
            )
        assert len(l.recent_validation_errors) == 3

    def test_clear_returns_true_only_when_nonempty(self):
        l = ObservationLedger()
        assert l.clear_validation_errors() is False
        l.record_validation_error(tool_name="x", error="y")
        assert l.clear_validation_errors() is True
        assert l.recent_validation_errors == []


# ── Soft tool failures ────────────────────────────────────────────────


class TestToolFailures:
    def test_dedup_by_tool_and_target(self):
        l = ObservationLedger()
        l.record_tool_failure(
            tool_name="code_edit", target="/foo.py", error="re-read first",
        )
        l.record_tool_failure(
            tool_name="code_edit", target="/foo.py", error="re-read first",
        )
        l.record_tool_failure(
            tool_name="code_edit", target="/bar.py", error="re-read first",
        )
        assert len(l.recent_tool_failures) == 2

    def test_prune_drops_stale_entries(self):
        l = ObservationLedger()
        l.record_tool_failure(tool_name="x", target="t", error="e")
        l.recent_tool_failures[0]["last_at"] = time.time() - (
            FAILURE_LEDGER_TTL_SECONDS + 60
        )
        assert l.prune_stale_tool_failures() == 1
        assert l.recent_tool_failures == []

    def test_prune_keeps_fresh_entries(self):
        l = ObservationLedger()
        l.record_tool_failure(tool_name="x", target="t", error="e")
        assert l.prune_stale_tool_failures() == 0
        assert len(l.recent_tool_failures) == 1


# ── Tool-call repeat detection ────────────────────────────────────────


class TestToolCalls:
    def test_path_intent_key_dedup(self):
        l = ObservationLedger()
        l.record_tool_call(
            tool_name="file_read", tool_input={"path": "/a.py"}, iteration=1,
        )
        l.record_tool_call(
            tool_name="file_read", tool_input={"path": "/a.py"}, iteration=2,
        )
        l.record_tool_call(
            tool_name="file_read", tool_input={"path": "/b.py"}, iteration=3,
        )
        assert l.repeat_count(
            tool_name="file_read", tool_input={"path": "/a.py"},
        ) == 2
        assert l.repeat_count(
            tool_name="file_read", tool_input={"path": "/b.py"},
        ) == 1

    def test_untracked_tool_returns_zero(self):
        l = ObservationLedger()
        l.record_tool_call(
            tool_name="task_list", tool_input={"items": []}, iteration=1,
        )
        assert l.recent_tool_calls == []
        assert l.repeat_count(
            tool_name="task_list", tool_input={"items": []},
        ) == 0

    def test_command_intent_key(self):
        l = ObservationLedger()
        l.record_tool_call(
            tool_name="shell_exec",
            tool_input={"command": "ls /workspace"},
            iteration=1,
        )
        l.record_tool_call(
            tool_name="shell_exec",
            tool_input={"command": "ls /workspace"},
            iteration=2,
        )
        assert l.hit_repeat_cap(
            tool_name="shell_exec",
            tool_input={"command": "ls /workspace"},
            cap=2,
        )

    def test_query_intent_key_falls_back(self):
        l = ObservationLedger()
        # ``pattern`` preferred when present
        l.record_tool_call(
            tool_name="code_grep",
            tool_input={"pattern": "foo", "query": "ignored"},
            iteration=1,
        )
        # ``query`` used when pattern absent
        l.record_tool_call(
            tool_name="code_grep",
            tool_input={"query": "bar"},
            iteration=2,
        )
        assert l.repeat_count(
            tool_name="code_grep", tool_input={"pattern": "foo"},
        ) == 1
        assert l.repeat_count(
            tool_name="code_grep", tool_input={"query": "bar"},
        ) == 1

    def test_clear_for_path_drops_only_that_path(self):
        l = ObservationLedger()
        l.record_tool_call(
            tool_name="file_read", tool_input={"path": "/a.py"}, iteration=1,
        )
        l.record_tool_call(
            tool_name="file_read", tool_input={"path": "/b.py"}, iteration=2,
        )
        l.record_tool_call(
            tool_name="shell_exec",
            tool_input={"command": "cat /a.py"}, iteration=3,
        )
        assert l.clear_tool_calls_for_path("/a.py") is True
        # /a.py file_read cleared; /b.py file_read survives;
        # shell_exec cat /a.py NOT cleared (not a path tool)
        assert {(e["tool"], e["key"]) for e in l.recent_tool_calls} == {
            ("file_read", "/b.py"),
            ("shell_exec", "cat /a.py"),
        }


# ── Background processes ──────────────────────────────────────────────


class TestBackgroundProcesses:
    def test_dedup_by_trimmed_command(self):
        l = ObservationLedger()
        l.record_background_process(command="node server.js &", iteration=1)
        l.record_background_process(command="node server.js &", iteration=5)
        assert len(l.background_processes) == 1
        assert l.background_processes[0]["count"] == 2

    def test_long_command_truncated(self):
        l = ObservationLedger()
        long_cmd = "python -m foo " + "x" * 200
        l.record_background_process(command=long_cmd, iteration=1)
        assert len(l.background_processes[0]["command"]) <= 121

    def test_empty_command_ignored(self):
        l = ObservationLedger()
        l.record_background_process(command="", iteration=1)
        l.record_background_process(command="   ", iteration=2)
        assert l.background_processes == []


# ── Shared-reference invariant ────────────────────────────────────────


class TestSharedReferenceInvariant:
    """:meth:`from_lists` shares list refs so CoderState's existing
    dataclass-field surface keeps reflecting ledger mutations. Without
    this, the wrapper would orphan its lists every time a ledger
    method ran."""

    def test_mutations_visible_through_shared_list(self):
        shared = []
        l = ObservationLedger.from_lists(
            recent_validation_errors=shared,
            recent_tool_failures=[],
            recent_tool_calls=[],
            background_processes=[],
        )
        l.record_validation_error(tool_name="x", error="y")
        assert len(shared) == 1
        assert shared[0]["tool"] == "x"

    def test_clear_in_place_preserves_shared_ref(self):
        shared = [{"tool": "x", "error": "y"}]
        l = ObservationLedger.from_lists(
            recent_validation_errors=shared,
            recent_tool_failures=[],
            recent_tool_calls=[],
            background_processes=[],
        )
        assert l.clear_validation_errors() is True
        assert shared == []  # external ref also empty (same list)

    def test_prune_in_place_preserves_shared_ref(self):
        shared = []
        l = ObservationLedger.from_lists(
            recent_validation_errors=[],
            recent_tool_failures=shared,
            recent_tool_calls=[],
            background_processes=[],
        )
        l.record_tool_failure(tool_name="x", target="t", error="e")
        assert len(shared) == 1
        shared[0]["last_at"] = time.time() - (FAILURE_LEDGER_TTL_SECONDS + 60)
        l.prune_stale_tool_failures()
        # prune used `[:]` slice-assign so the shared ref still points
        # at the same list object now empty
        assert shared == []
