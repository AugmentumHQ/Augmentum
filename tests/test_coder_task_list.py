"""Tests for TaskListTool + sticky-reminder state machinery.

Covers:
  - Tool input validation (empty items, non-dict items, bad status, multiple in_progress)
  - Tool success path (single in_progress, wholesale replace, activeForm fallback)
  - State helpers: set_tasks, active_task, record_validation_error (dedup),
    clear_validation_errors
  - Serialisation round-trip via to_dict / from_row
"""

from __future__ import annotations

import json

import pytest

from augmentum.coder.state import CoderState, CoderPhase
from augmentum.coder.tools import TaskListTool


def _make_tool():
    state = CoderState(session_id="s1", workspace_id="w1")
    tool = TaskListTool(container_manager=None, workspace_id="w1", state=state)
    return tool, state


# --- Tool validation ------------------------------------------------------


class TestTaskListValidation:
    async def test_missing_items_validation_error(self):
        tool, _ = _make_tool()
        r = await tool.execute()
        assert r.validation_error is True
        assert "items" in r.error

    async def test_non_list_items_validation_error(self):
        tool, _ = _make_tool()
        r = await tool.execute(items="not a list")
        assert r.validation_error is True

    async def test_non_dict_item_validation_error(self):
        tool, _ = _make_tool()
        r = await tool.execute(items=["not a dict"])
        assert r.validation_error is True
        assert "must be an object" in r.error

    async def test_empty_content_validation_error(self):
        tool, _ = _make_tool()
        r = await tool.execute(items=[{"content": "", "status": "pending"}])
        assert r.validation_error is True
        assert "content" in r.error

    async def test_bad_status_validation_error(self):
        tool, _ = _make_tool()
        r = await tool.execute(items=[{"content": "x", "status": "doing"}])
        assert r.validation_error is True
        assert "status" in r.error

    async def test_multiple_in_progress_validation_error(self):
        tool, _ = _make_tool()
        r = await tool.execute(items=[
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"},
        ])
        assert r.validation_error is True
        assert "in_progress" in r.error


# --- Tool success path ----------------------------------------------------


class TestTaskListSuccess:
    async def test_single_task_in_progress(self):
        tool, state = _make_tool()
        r = await tool.execute(items=[
            {"content": "Read INSTALL.md", "activeForm": "Reading INSTALL.md",
             "status": "in_progress"},
        ])
        assert r.success is True
        assert len(state.tasks) == 1
        assert state.tasks[0]["status"] == "in_progress"
        assert state.tasks[0]["activeForm"] == "Reading INSTALL.md"

    async def test_wholesale_replace(self):
        """Each call replaces the full list — no merge semantics."""
        tool, state = _make_tool()
        await tool.execute(items=[
            {"content": "a", "status": "pending"},
            {"content": "b", "status": "pending"},
        ])
        assert len(state.tasks) == 2
        await tool.execute(items=[{"content": "c", "status": "in_progress"}])
        assert len(state.tasks) == 1
        assert state.tasks[0]["content"] == "c"

    async def test_activeform_defaults_to_content(self):
        tool, state = _make_tool()
        await tool.execute(items=[{"content": "Run tests", "status": "pending"}])
        assert state.tasks[0]["activeForm"] == "Run tests"

    async def test_output_contains_all_items(self):
        tool, _ = _make_tool()
        r = await tool.execute(items=[
            {"content": "First",  "status": "completed"},
            {"content": "Second", "status": "in_progress"},
            {"content": "Third",  "status": "pending"},
        ])
        assert "First" in r.output and "[x]" in r.output
        assert "Second" in r.output and "[~]" in r.output
        assert "Third" in r.output and "[ ]" in r.output

    async def test_zero_in_progress_ok(self):
        """Empty plan or all-pending is fine; invariant is at-MOST-one."""
        tool, _ = _make_tool()
        r = await tool.execute(items=[
            {"content": "a", "status": "pending"},
            {"content": "b", "status": "completed"},
        ])
        assert r.success is True


# --- State helpers --------------------------------------------------------


class TestStateHelpers:
    def test_active_task_returns_in_progress(self):
        s = CoderState(session_id="x", workspace_id="y")
        s.set_tasks([
            {"content": "a", "activeForm": "a", "status": "completed"},
            {"content": "b", "activeForm": "b", "status": "in_progress"},
            {"content": "c", "activeForm": "c", "status": "pending"},
        ])
        assert s.active_task()["content"] == "b"

    def test_active_task_none_when_no_in_progress(self):
        s = CoderState(session_id="x", workspace_id="y")
        s.set_tasks([{"content": "a", "activeForm": "a", "status": "pending"}])
        assert s.active_task() is None

    def test_record_validation_error_dedupes_by_tool(self):
        s = CoderState(session_id="x", workspace_id="y")
        s.record_validation_error(tool_name="shell_exec", error="first")
        s.record_validation_error(tool_name="shell_exec", error="second")
        s.record_validation_error(tool_name="shell_exec", error="third")
        assert len(s.recent_validation_errors) == 1
        assert s.recent_validation_errors[0]["count"] == 3
        assert s.recent_validation_errors[0]["error"] == "third"

    def test_repeat_count_tracks_same_signature(self):
        """Same-signature failures bump repeat_count; the circuit breaker
        watches this field for the "model stuck on identical bad call"
        pattern (the common file_write-without-path loop)."""
        s = CoderState(session_id="x", workspace_id="y")
        err = "file_write called without a 'path' argument. Required: path + content."
        s.record_validation_error(tool_name="file_write", error=err)
        s.record_validation_error(tool_name="file_write", error=err)
        s.record_validation_error(tool_name="file_write", error=err)
        entry = s.recent_validation_errors[0]
        assert entry["count"] == 3
        assert entry["repeat_count"] == 3

    def test_repeat_count_resets_on_signature_change(self):
        """Different first-sentence = different signature → repeat_count
        resets to 1. count still increments (it's per-tool, not per-error).
        Without this distinction, the breaker would fire on legit "model
        wandering through different bad calls" sequences."""
        s = CoderState(session_id="x", workspace_id="y")
        s.record_validation_error(
            tool_name="file_write",
            error="file_write called without a 'path' argument.",
        )
        s.record_validation_error(
            tool_name="file_write",
            error="file_write called without a 'path' argument.",
        )
        # Now a DIFFERENT failure mode for the same tool.
        s.record_validation_error(
            tool_name="file_write",
            error="file_write content is too large (50000 tokens > 8000 cap).",
        )
        entry = s.recent_validation_errors[0]
        assert entry["count"] == 3
        assert entry["repeat_count"] == 1  # reset on signature change

    def test_repeat_count_unaffected_by_other_tools(self):
        """Per-tool tracking: shell_exec failures shouldn't bump
        file_write's repeat_count."""
        s = CoderState(session_id="x", workspace_id="y")
        err = "file_write called without a 'path' argument."
        s.record_validation_error(tool_name="file_write", error=err)
        s.record_validation_error(tool_name="shell_exec", error="bad command")
        s.record_validation_error(tool_name="file_write", error=err)
        fw = next(e for e in s.recent_validation_errors if e["tool"] == "file_write")
        assert fw["repeat_count"] == 2

    def test_record_validation_error_keeps_multiple_tools(self):
        s = CoderState(session_id="x", workspace_id="y")
        s.record_validation_error(tool_name="shell_exec", error="a")
        s.record_validation_error(tool_name="file_read", error="b")
        s.record_validation_error(tool_name="code_grep", error="c")
        assert len(s.recent_validation_errors) == 3

    def test_record_validation_error_caps_at_3(self):
        s = CoderState(session_id="x", workspace_id="y")
        for tool in ("t1", "t2", "t3", "t4", "t5"):
            s.record_validation_error(tool_name=tool, error="x")
        # FIFO trim — oldest (t1, t2) dropped.
        kept = [e["tool"] for e in s.recent_validation_errors]
        assert len(kept) == 3
        assert kept == ["t3", "t4", "t5"]

    def test_clear_validation_errors(self):
        s = CoderState(session_id="x", workspace_id="y")
        s.record_validation_error(tool_name="x", error="y")
        s.clear_validation_errors()
        assert s.recent_validation_errors == []

    def test_truncates_long_error_message(self):
        s = CoderState(session_id="x", workspace_id="y")
        s.record_validation_error(tool_name="x", error="a" * 500)
        assert len(s.recent_validation_errors[0]["error"]) == 200


# --- Serialisation round-trip --------------------------------------------


class TestSerialisation:
    def test_to_dict_includes_tasks_and_errors(self):
        s = CoderState(session_id="x", workspace_id="y")
        s.set_tasks([{"content": "a", "activeForm": "a", "status": "pending"}])
        s.record_validation_error(tool_name="shell_exec", error="bad")
        d = s.to_dict()
        assert json.loads(d["tasks"])[0]["content"] == "a"
        errors = json.loads(d["recent_validation_errors"])
        assert errors[0]["tool"] == "shell_exec"

    def test_from_row_round_trip(self):
        s = CoderState(session_id="x", workspace_id="y")
        s.set_tasks([{"content": "a", "activeForm": "a", "status": "in_progress"}])
        s.record_validation_error(tool_name="shell_exec", error="bad")
        d = s.to_dict()
        # Simulate SQLite read path
        row = {
            **d,
            "phase": d["phase"],
        }
        restored = CoderState.from_row(row)
        assert restored.tasks == s.tasks
        assert restored.recent_validation_errors == s.recent_validation_errors

    def test_from_row_missing_fields_defaults_to_empty(self):
        """Backwards compat: a row written before this feature still loads."""
        row = {
            "session_id": "x", "workspace_id": "y", "phase": "waiting",
            # No tasks / recent_validation_errors keys at all.
        }
        restored = CoderState.from_row(row)
        assert restored.tasks == []
        assert restored.recent_validation_errors == []


# --- Sticky reminder rendering --------------------------------------------


def _make_handler():
    """Minimal handler for testing pure rendering methods."""
    from augmentum.modes.coder.handler import CoderHandler
    h = CoderHandler(backend=None, session_id="test_sess")
    return h


class TestStickyReminder:
    def test_goal_always_present(self):
        h = _make_handler()
        out = h._build_sticky_reminder(
            goal="Build nsnake", iteration=1, max_iters=100, writes=0,
        )
        assert "<system-reminder>" in out
        assert "</system-reminder>" in out
        assert "Goal: Build nsnake" in out

    def test_goal_truncation(self):
        h = _make_handler()
        out = h._build_sticky_reminder(
            goal="x" * 1000, iteration=1, max_iters=100, writes=0,
        )
        # Goal clamped to 280 chars to keep reminder tight
        goal_line = [l for l in out.splitlines() if l.startswith("Goal:")][0]
        assert len(goal_line) < 300

    def test_empty_task_list_shows_hint(self):
        h = _make_handler()
        out = h._build_sticky_reminder(
            goal="g", iteration=1, max_iters=100, writes=0,
        )
        assert "task_list" in out   # The hint mentions the tool name

    def test_task_list_rendered_with_markers(self):
        h = _make_handler()
        h._state.set_tasks([
            {"content": "Done one",  "activeForm": "Doing", "status": "completed"},
            {"content": "Cur one",   "activeForm": "Doing", "status": "in_progress"},
            {"content": "Todo one",  "activeForm": "Doing", "status": "pending"},
        ])
        out = h._build_sticky_reminder(
            goal="g", iteration=2, max_iters=100, writes=1,
        )
        assert "[x] Done one" in out
        assert "[~] Cur one" in out
        assert "[ ] Todo one" in out
        assert "← current" in out   # active task marker

    def test_blockers_section_rendered(self):
        h = _make_handler()
        h._state.record_validation_error(tool_name="shell_exec", error="command is required")
        h._state.record_validation_error(tool_name="shell_exec", error="command is required")
        out = h._build_sticky_reminder(
            goal="g", iteration=3, max_iters=100, writes=0,
        )
        assert "Recent blockers" in out
        assert "shell_exec" in out
        assert "(×2)" in out

    def test_blockers_section_omitted_when_empty(self):
        h = _make_handler()
        out = h._build_sticky_reminder(
            goal="g", iteration=1, max_iters=100, writes=0,
        )
        assert "Recent blockers" not in out

    def test_budget_signal_present(self):
        h = _make_handler()
        out = h._build_sticky_reminder(
            goal="g", iteration=42, max_iters=100, writes=7,
        )
        assert "Iteration 42/100" in out
        assert "7 writes" in out

    def test_inject_appends_when_tail_not_reminder(self):
        from augmentum.models.engine import Message
        h = _make_handler()
        messages = [
            Message(role="system", content="sys"),
            Message(role="user",   content="original task"),
        ]
        h._inject_sticky_reminder(
            messages, goal="g", iteration=1, max_iters=100, writes=0,
        )
        assert len(messages) == 3
        assert messages[-1].content.startswith("<system-reminder>")

    def test_inject_replaces_trailing_reminder(self):
        """Two reminders in a row → last one wins, no history growth."""
        from augmentum.models.engine import Message
        h = _make_handler()
        messages = [
            Message(role="system", content="sys"),
            Message(role="user",   content="original task"),
        ]
        h._inject_sticky_reminder(
            messages, goal="g", iteration=1, max_iters=100, writes=0,
        )
        h._inject_sticky_reminder(
            messages, goal="g", iteration=2, max_iters=100, writes=0,
        )
        assert len(messages) == 3   # no growth
        assert "Iteration 2/100" in messages[-1].content

    def test_inject_appends_after_tool_result(self):
        """If a non-reminder user message (tool result) intervenes, append fresh."""
        from augmentum.models.engine import Message
        h = _make_handler()
        messages = [
            Message(role="system", content="sys"),
            Message(role="user",   content="original task"),
        ]
        h._inject_sticky_reminder(
            messages, goal="g", iteration=1, max_iters=100, writes=0,
        )
        # Simulate text-tier tool result being glued onto the last user msg
        messages.append(Message(role="assistant", content="okay"))
        messages.append(Message(role="user", content="[Tool result: x]\nok"))
        h._inject_sticky_reminder(
            messages, goal="g", iteration=2, max_iters=100, writes=0,
        )
        # Reminder appended, tool-result message preserved
        assert messages[-1].content.startswith("<system-reminder>")
        assert messages[-2].content.startswith("[Tool result:")


# --- Same-signature validation breaker -----------------------------------


class TestSameValidationRepeatBreaker:
    """The phase_act helper that fires when the model loops on the
    identical bad call. Imported from the mode-specific module."""

    def _state_with(self, *errors_by_tool):
        s = CoderState(session_id="x", workspace_id="y")
        for tool_name, err in errors_by_tool:
            s.record_validation_error(tool_name=tool_name, error=err)
        return s

    def test_no_repeats_returns_none(self):
        from augmentum.modes.coder.phase_act import _find_repeat_offender
        s = self._state_with(("file_write", "missing path"))
        assert _find_repeat_offender(s) is None

    def test_single_signature_repeat_fires(self):
        from augmentum.modes.coder.phase_act import _find_repeat_offender
        s = self._state_with(
            ("file_write", "called without a 'path' argument"),
            ("file_write", "called without a 'path' argument"),
        )
        offender = _find_repeat_offender(s)
        assert offender is not None
        assert offender["tool"] == "file_write"
        assert offender["repeat_count"] == 2

    def test_signature_change_does_not_fire(self):
        from augmentum.modes.coder.phase_act import _find_repeat_offender
        s = self._state_with(
            ("file_write", "called without a 'path' argument"),
            ("file_write", "content is too large"),
        )
        # Different errors → repeat_count was reset to 1; no break.
        assert _find_repeat_offender(s) is None

    def test_picks_worst_offender_on_ties(self):
        """When multiple tools both repeat past threshold, the bail
        message names the one with the highest repeat_count so the user
        sees the most pathological behavior."""
        from augmentum.modes.coder.phase_act import _find_repeat_offender
        s = self._state_with(
            ("file_write", "without a 'path' argument"),
            ("file_write", "without a 'path' argument"),
            ("file_write", "without a 'path' argument"),
            ("shell_exec", "missing command"),
            ("shell_exec", "missing command"),
        )
        offender = _find_repeat_offender(s)
        assert offender is not None
        assert offender["tool"] == "file_write"  # 3× beats 2×
        assert offender["repeat_count"] == 3

    def test_break_message_names_tool_and_count(self):
        from augmentum.modes.coder.phase_act import _format_repeat_break_message
        msg = _format_repeat_break_message({
            "tool": "file_write",
            "repeat_count": 4,
        })
        assert "file_write" in msg
        assert "4×" in msg or "4x" in msg.lower()
        # Surfaces the most-likely fix.
        assert "code_edit" in msg or "code_edit" in msg.lower()


# --- Per-request state reset --------------------------------------------


class TestStateReset:
    def test_reset_clears_per_request_scratchpads(self):
        h = _make_handler()
        h._state.set_tasks([{"content": "old", "activeForm": "old", "status": "in_progress"}])
        h._state.record_validation_error(tool_name="x", error="y")
        h._state.plan = "old plan"
        h._state.plan_steps = ["a", "b"]
        h._state.current_step = 2
        h._state.step_outputs = {"0": "done"}
        h._state.consecutive_failures = 5
        h._state.error = "old error"

        h._reset_for_new_request()

        assert h._state.tasks == []
        assert h._state.recent_validation_errors == []
        assert h._state.plan == ""
        assert h._state.plan_steps == []
        assert h._state.current_step == 0
        assert h._state.step_outputs == {}
        assert h._state.consecutive_failures == 0
        assert h._state.error is None

    def test_reset_can_preserve_objective_for_continuations(self):
        h = _make_handler()
        h._state.set_tasks([{"content": "old", "activeForm": "old", "status": "in_progress"}])
        h._state.plan = "old plan"
        h._state.plan_steps = ["a", "b"]
        h._state.current_step = 1
        h._state.step_outputs = {"0": "done"}
        h._state.mission = [{"description": "keep going"}]  # minimal truthy payload
        h._state.set_pending_objective_contract({
            "kind": "operate_remote_access",
            "summary": "public access not proven",
        })
        h._state.record_validation_error(tool_name="x", error="y")

        h._reset_for_new_request(preserve_objective=True)

        assert h._state.tasks == [{"content": "old", "activeForm": "old", "status": "in_progress"}]
        assert h._state.plan == "old plan"
        assert h._state.plan_steps == ["a", "b"]
        assert h._state.current_step == 1
        assert h._state.step_outputs == {"0": "done"}
        assert h._state.pending_objective_contract["kind"] == "operate_remote_access"
        assert h._state.recent_validation_errors == []

    def test_reset_preserves_session_invariants(self):
        """files_read + tool_calls_made are session-level; reset leaves them alone."""
        h = _make_handler()
        h._state.record_file_read("/workspace/a.py")
        h._state.record_file_read("/workspace/b.py")
        h._state.tool_calls_made = 42

        h._reset_for_new_request()

        assert "/workspace/a.py" in h._state.files_read
        assert "/workspace/b.py" in h._state.files_read
        assert h._state.tool_calls_made == 42

    def test_reset_clears_recent_tool_calls(self):
        """New request should start with empty recent_tool_calls dedup buffer."""
        h = _make_handler()
        h._state.record_tool_call(
            tool_name="file_read", tool_input={"path": "/a"}, iteration=1,
        )
        assert h._state.recent_tool_calls
        h._reset_for_new_request()
        assert h._state.recent_tool_calls == []

    def test_reset_clears_pending_objective_contract_without_preserve(self):
        h = _make_handler()
        h._state.set_pending_objective_contract({
            "kind": "operate_remote_access",
            "summary": "public access not proven",
        })

        h._reset_for_new_request()

        assert h._state.pending_objective_contract == {}


# --- Redundant-fetch dedup (tracker + reminder + cap) -------------------


class TestRecentToolCalls:
    def test_file_read_tracked_by_path(self):
        s = CoderState(session_id="x", workspace_id="y")
        s.record_tool_call(
            tool_name="file_read", tool_input={"path": "/README.md"}, iteration=1,
        )
        assert len(s.recent_tool_calls) == 1
        assert s.recent_tool_calls[0]["key"] == "/README.md"
        assert s.recent_tool_calls[0]["count"] == 1

    def test_same_path_increments_count_not_new_entry(self):
        s = CoderState(session_id="x", workspace_id="y")
        for _ in range(3):
            s.record_tool_call(
                tool_name="file_read", tool_input={"path": "/a"}, iteration=1,
            )
        assert len(s.recent_tool_calls) == 1
        assert s.recent_tool_calls[0]["count"] == 3

    def test_different_paths_are_separate_entries(self):
        s = CoderState(session_id="x", workspace_id="y")
        s.record_tool_call(tool_name="file_read", tool_input={"path": "/a"}, iteration=1)
        s.record_tool_call(tool_name="file_read", tool_input={"path": "/b"}, iteration=1)
        assert len(s.recent_tool_calls) == 2

    def test_shell_tracked_by_command(self):
        s = CoderState(session_id="x", workspace_id="y")
        s.record_tool_call(
            tool_name="shell_read", tool_input={"command": "ls -la"}, iteration=1,
        )
        assert s.recent_tool_calls[0]["key"] == "ls -la"

    def test_untracked_tools_are_noops(self):
        """task_list and ask_user don't reflect 'gathering info' — skip them."""
        s = CoderState(session_id="x", workspace_id="y")
        s.record_tool_call(
            tool_name="task_list", tool_input={"items": []}, iteration=1,
        )
        s.record_tool_call(
            tool_name="ask_user", tool_input={"questions": []}, iteration=1,
        )
        assert s.recent_tool_calls == []

    def test_buffer_bounded(self):
        s = CoderState(session_id="x", workspace_id="y")
        for i in range(20):
            s.record_tool_call(
                tool_name="file_read", tool_input={"path": f"/f{i}"}, iteration=1,
            )
        assert len(s.recent_tool_calls) == 8

    def test_hit_repeat_cap_false_below_threshold(self):
        s = CoderState(session_id="x", workspace_id="y")
        for _ in range(4):
            s.record_tool_call(
                tool_name="file_read", tool_input={"path": "/a"}, iteration=1,
            )
        assert not s.hit_repeat_cap(
            tool_name="file_read", tool_input={"path": "/a"}, cap=5,
        )

    def test_hit_repeat_cap_true_at_threshold(self):
        s = CoderState(session_id="x", workspace_id="y")
        for _ in range(5):
            s.record_tool_call(
                tool_name="file_read", tool_input={"path": "/a"}, iteration=1,
            )
        assert s.hit_repeat_cap(
            tool_name="file_read", tool_input={"path": "/a"}, cap=5,
        )

    def test_reminder_includes_already_inspected_section(self):
        h = _make_handler()
        h._state.record_tool_call(
            tool_name="file_read", tool_input={"path": "/README.md"}, iteration=1,
        )
        h._state.record_tool_call(
            tool_name="file_read", tool_input={"path": "/README.md"}, iteration=2,
        )
        h._state.record_tool_call(
            tool_name="shell_read", tool_input={"command": "ls -la"}, iteration=3,
        )
        out = h._build_sticky_reminder(
            goal="build", iteration=4, max_iters=100, writes=0,
        )
        assert "Already inspected" in out
        assert "/README.md" in out
        assert "(×2)" in out
        assert "ls -la" in out
        assert "Don't re-fetch" in out

    def test_reminder_omits_already_inspected_when_empty(self):
        h = _make_handler()
        out = h._build_sticky_reminder(
            goal="build", iteration=1, max_iters=100, writes=0,
        )
        assert "Already inspected" not in out

    def test_reminder_truncates_long_shell_commands(self):
        h = _make_handler()
        # Make it unambiguously > 80 chars
        long_cmd = (
            "find /workspace -name '*.py' -exec grep -l 'matching_pattern_here' "
            "{} \\; -print | sort | uniq | head -50"
        )
        assert len(long_cmd) > 80
        h._state.record_tool_call(
            tool_name="shell_read", tool_input={"command": long_cmd}, iteration=1,
        )
        out = h._build_sticky_reminder(
            goal="x", iteration=2, max_iters=100, writes=0,
        )
        # Command is truncated in the display so a 200-char command
        # doesn't dominate the reminder
        assert "…" in out


# --- Unknown-tool validation flag ----------------------------------------


class TestUnknownToolFlag:
    async def test_unknown_tool_returns_validation_error(self):
        """Hallucinated tool names must flag validation_error so the
        circuit breaker can count them toward the malformed-call streak."""
        from augmentum.modes.coder.handler import _execute_tool
        result = await _execute_tool(
            tool_map={},     # No tools registered
            tool_name="imaginary_search",
            tool_input={"q": "anything"},
        )
        assert result.success is False
        assert result.validation_error is True
        assert "imaginary_search" in result.error
        assert "Available:" in result.error


# --- Malformed-JSON surfacing --------------------------------------------


class TestParseErrorSurfacing:
    async def test_tool_tracked_prepends_json_hint(self):
        """When _parse_error_raw is on the raw tc dict and the tool fails,
        the resulting error must mention malformed JSON + the raw string."""
        from augmentum.modes.coder.handler import CoderHandler
        from augmentum.coder.tools import FileReadTool

        class _Backend:
            pass

        h = CoderHandler(backend=_Backend(), session_id="s1")
        # File-index-less FileReadTool will fail validation on empty path
        tool = FileReadTool(container_manager=None, workspace_id="w", state=h._state)
        tool_map = {"file_read": tool}

        tc = {
            "id":   "call_1",
            "name": "file_read",
            "input": {},   # empty because JSON parse failed
            "_parse_error_raw": '{"path": "/foo',   # truncated JSON
        }
        chunks = []
        async for ev in h._run_tool_tracked(
            tc=tc, tool_map=tool_map, tier=None,   # tier irrelevant, text path
            messages=[], model="x", counters={},
        ):
            chunks.append(ev)

        # Find the tool_result meta chunk
        result_chunks = [
            c for c in chunks
            if c.augmentum and c.augmentum.get("status") == "tool_result"
        ]
        assert result_chunks, "Expected a tool_result chunk"
        preview = result_chunks[0].augmentum["tool_result"]["output_preview"]
        assert "not valid JSON" in preview
        assert '{"path": "/foo' in preview
