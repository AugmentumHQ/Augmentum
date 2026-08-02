"""Tests for the 2026-04-20 loop-wiring fixes.

Origin: user showed a trace where "hey there" triggered a full plan
phase and "what's in this project?" thrashed for ~40 iterations on
failing test_run calls. The audit found four gaps:

  1. Task completion never terminated the hybrid loop — `tasks`, a
     pending/in_progress/completed state machine seeded from the plan,
     was rendered in the sticky reminder but never read by the loop.

  2. The no-tool-calls exit over-fired the continuation nudge: it
     only terminated cleanly on "recent writes", so informational
     queries (read + answer + stop, zero writes) got nudged back into
     an action-hunting loop.

  3. Greetings / small-talk entered plan→act. "hey there" shouldn't
     generate a plan.

  4. TestRunTool reported Passed=1 when pytest wasn't installed — the
     fallback in _parse_results flipped to success for any output
     lacking "error"/"fail"/"traceback" keywords. The thrashing-streak
     breaker never fired because every failure was being reported as
     a pass.

This file covers all four fixes plus the sticky-reminder soft-failure
section that was added alongside.

Run: python -m pytest tests/test_coder_loop_wiring.py -v
"""
from __future__ import annotations

import pytest

from augmentum.coder.models import FileEntry
from augmentum.coder.state import CoderState, CoderPhase
from augmentum.coder.tools import (
    TestRunTool as _TestRunTool,  # alias so pytest doesn't try to collect it
    _resolve_workspace_path,
    _shell_command_failure,
)
from augmentum.modes.coder.handler import (
    CoderHandler,
    _ACTION_STAGNATION_BREAK,
    _classify_turn_intent,
    _explicitly_requests_execution,
    _HYBRID_MIN_TURN_PROSE_CHARS,
    _INSPECTION_COLD_START_GRACE,
    _READ_REPEAT_REFUSAL_CAP,
    _is_explanatory_request,
    _is_read_only_request,
    _generate_clarification,
    _has_content_loop,
    _has_unclaimed_code_block,
    _is_conversational_greeting,
    _is_vague_improvement,
    _is_vague_request,
    _plan_is_question,
    _strip_tool_json,
    _soft_failure_target,
)
from augmentum.modes.coder.intent import Tier, TierClassification
from augmentum.modes.analytical.tool_calling import ToolCallingTier
from augmentum.models.base import InternalStreamChunk, Message

from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeBackend,
    _FakeChunk,
    _FakeTool,
    _force_native_tier,
    _make_request,
    _make_turn_context,
    _tc_delta,
)


# ---------------------------------------------------------------------------
# Fix 1a: tasks-completed termination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_tasks_completed_terminates_hybrid_loop(monkeypatch):
    """When every seeded task is status=completed, the hybrid loop
    breaks with termination_reason='tasks_completed' without calling
    the model again."""
    _force_native_tier(monkeypatch)
    fake_tool = _FakeTool("file_read")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_tool],
    )

    class _StubBackend:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            # First iteration: emit a tool call so counters tick.
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-1", "file_read", {"path": "/x.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                # Shouldn't be reached; tasks will be all-completed
                # before the second backend call lands.
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _StubBackend()
    handler = CoderHandler(
        backend, session_id="sess-tc",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-tc",
    )
    # Seed one in-progress task; after iteration 1 it'll be flipped
    # to completed and the loop should terminate at iteration 2's top.
    handler._state.set_tasks([
        {"content": "read x.py", "activeForm": "reading x.py",
         "status": "in_progress"},
    ])

    # Simulate the model marking the task complete after iteration 1
    # — we do this between chat_stream calls by hooking a post-tool
    # callback. Simplest: flip the task after the first iteration
    # by mutating state on backend's second call.
    orig_chat_stream = backend.chat_stream

    async def _wrapped(request):
        # Mark task complete right before the SECOND iteration so the
        # tasks-completed check at top-of-iter 2 fires.
        if backend.calls >= 1:
            handler._state.tasks = [
                {**t, "status": "completed"}
                for t in handler._state.tasks
            ]
        async for c in orig_chat_stream(request):
            yield c

    backend.chat_stream = _wrapped

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("read x.py"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "tasks_completed"
    ]
    assert break_chunks, "Expected tasks_completed to fire"
    assert break_chunks[0].augmentum.get("task_count") == 1


@pytest.mark.asyncio
async def test_tasks_completed_does_not_fire_on_iteration_1(monkeypatch):
    """Preseeded all-completed tasks must NOT short-circuit iteration 1
    — that guard prevents a resumed session from skipping the act phase
    entirely."""
    _force_native_tier(monkeypatch)
    fake_tool = _FakeTool("file_read")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_tool],
    )

    class _StopsImmediately:
        async def chat_stream(self, request):
            # Model stops with prose on iter 1.
            yield _FakeChunk(content_delta="Everything already done." * 3)
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _StopsImmediately(), session_id="sess-preseed",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-preseed",
    )
    handler._state.set_tasks([
        {"content": "task A", "activeForm": "A", "status": "completed"},
    ])

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("resume"), workspace_context="",
    ):
        chunks.append(c)

    # iteration-1 tasks_completed would be a false positive; ensure
    # it did NOT fire.
    tc_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "tasks_completed"
    ]
    assert not tc_chunks, (
        "Preseeded completed tasks must not short-circuit iter 1"
    )


@pytest.mark.asyncio
async def test_tasks_completed_remote_access_goal_forces_followup(monkeypatch):
    """`tasks_completed` must not short-circuit an unresolved operate goal.

    Regression for the tunnel flow: even if the task list says "done", a
    remote-access request still needs either public verification or a plain
    blocker explanation. The loop should nudge once, then allow one more
    backend round instead of immediately stopping on the completed tasks.
    """
    _force_native_tier(monkeypatch)
    shell_tool = _FakeTool(
        "shell_exec",
        output=(
            "your url is: https://bright-rice-sleep.loca.lt\n"
            "Server HTTP: 302\n"
        ),
    )

    class _OperateThenExplain:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(
                        0,
                        "tc-remote-shell",
                        "shell_exec",
                        {"command": "lt --port 8080"},
                    ),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(content_delta=(
                    "Localtunnel is giving an interstitial and then bad "
                    "gateway here, so I can't provide a clean remote link "
                    "from this environment."
                ))
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _OperateThenExplain()
    handler = CoderHandler(
        backend,
        session_id="sess-tasks-complete-operate",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-tasks-complete-operate",
    )
    handler._turn_intent_for_turn = _classify_turn_intent(
        "start the app and expose it so I can access it remotely",
    )
    handler._state.set_tasks([
        {
            "content": "expose the app remotely",
            "activeForm": "exposing the app remotely",
            "status": "in_progress",
        },
    ])

    orig_execute = shell_tool.execute

    async def _complete_tasks_during_tool(**kwargs):
        try:
            return await orig_execute(**kwargs)
        finally:
            handler._state.tasks = [
                {**t, "status": "completed"}
                for t in handler._state.tasks
            ]

    shell_tool.execute = _complete_tasks_during_tool
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [shell_tool],
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("start the app and expose it so I can access it remotely"),
        workspace_context="",
    ):
        chunks.append(c)

    assert backend.calls >= 2, (
        "Completed tasks should not end an unresolved remote-access turn "
        "before the model can explain the blocker"
    )
    evidence_nudges = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "operate_evidence_nudge"
    ]
    assert evidence_nudges, "Expected operate_evidence_nudge before stopping"
    assert any(
        "can't provide a clean remote link" in (c.content_delta or "")
        for c in chunks
    ), "Expected a blocker explanation after the tasks_completed nudge"
    assert handler._state.pending_objective_contract == {}


# ---------------------------------------------------------------------------
# Fix 1b: meaningful-answer stop (no tool calls + prose)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prose_stop_terminates_without_nudge(monkeypatch):
    """When the model stops with substantive prose (multi-sentence
    real-content answer) under a passive analysis request and zero
    writes, the loop terminates cleanly without emitting a
    continuation_nudge. Pre-fix: the nudge fired here because the
    judge only looked at recent writes.

    Updated 2026-05-10 (Phase 3.6): the prose threshold is now
    length+sentence-count via ``classify_prose``, not a flat 40-char
    floor. Test prose updated to match the new substantive contract.
    A 50-char block of repeated 'x' (no sentence terminators) is
    correctly classified as BAILOUT and would nudge; that's the
    intended Phase 3.6 behavior. The contract this test pins is
    'real prose accepts', not 'any string >= 50 chars accepts'."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [],
    )

    # Multi-sentence real-content prose — passes the Phase 3.6 gate
    # because it's >= 30 chars AND has >= 2 sentence terminators.
    prose = (
        "The repo follows a layered architecture. "
        "The proxy layer routes API calls. "
        "The mode layer dispatches to per-mode handlers. "
        "Persistence is SQLite via aiosqlite."
    )

    class _AnswersWithoutTools:
        async def chat_stream(self, request):
            yield _FakeChunk(content_delta=prose)
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _AnswersWithoutTools(), session_id="sess-prose",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-prose",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("summarize the repo"), workspace_context="",
    ):
        chunks.append(c)

    nudges = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "continuation_nudge"
    ]
    assert not nudges, (
        "Substantive prose stop should NOT trigger continuation_nudge"
    )


@pytest.mark.asyncio
async def test_empty_stop_still_nudges(monkeypatch):
    """If the model stops with NO tool calls AND no substantive prose,
    the continuation nudge still fires once — we don't want to lose
    the recovery path for genuine stalls."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [],
    )

    class _SilentStop:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                # Empty prose, no tools — should nudge.
                yield _FakeChunk(content_delta="")
                yield _FakeChunk(done=True, finish_reason="stop")
            else:
                # After nudge, stop again with empty prose so loop can
                # exit via model_stop_after_nudge.
                yield _FakeChunk(content_delta="")
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _SilentStop(), session_id="sess-silent",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-silent",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("do the thing"), workspace_context="",
    ):
        chunks.append(c)

    nudges = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "continuation_nudge"
    ]
    assert nudges, "Empty stop should still nudge (recovery signal)"


# ---------------------------------------------------------------------------
# Fix 3: conversational short-circuit
# ---------------------------------------------------------------------------


class TestConversationalGreeting:
    """Detection heuristic must cover greetings but not real requests."""

    @pytest.mark.parametrize("msg", [
        "hey",
        "hi",
        "hello",
        "hey there",
        "thanks",
        "thank you",
        "ok cool",
        "good morning",
        "yo",
        "sup",
        "what's up",
        "hey claude",
        "thanks!",
        "hi.",
        "hello!",
    ])
    def test_greetings_detected(self, msg):
        assert _is_conversational_greeting(msg), (
            f"{msg!r} should be detected as conversational"
        )

    @pytest.mark.parametrize("msg", [
        "read main.py",
        "what's in this project?",
        "fix the bug in auth.py",
        "run the tests",
        "create a fibonacci script",
        "",
        "help me understand the workspace structure and how modules connect",
    ])
    def test_real_requests_not_flagged(self, msg):
        assert not _is_conversational_greeting(msg), (
            f"{msg!r} should NOT be detected as conversational"
        )


# ---------------------------------------------------------------------------
# Fix 4: TestRunTool shell-failure detection
# ---------------------------------------------------------------------------


class TestExplanatoryRequestDetection:
    @pytest.mark.parametrize("msg", [
        "explain this project",
        "describe the python files in this repo",
        "walk me through what files this project has",
        "can you look at this project to explain what python files it has?",
        "what's in this project?",
    ])
    def test_explanatory_requests_detected(self, msg):
        assert _is_explanatory_request(msg), (
            f"{msg!r} should be treated as explanatory/read-only"
        )

    @pytest.mark.parametrize("msg", [
        "run the tests and explain the failures",
        "build the project and describe the output",
        "execute the script and summarize what it does",
    ])
    def test_execution_requests_detected(self, msg):
        assert _explicitly_requests_execution(msg), (
            f"{msg!r} should count as explicit execution"
        )

    @pytest.mark.parametrize("msg", [
        "fix the bug in auth.py",
        "add a new settings route",
        "refactor the cache layer",
        "list the files in src",
    ])
    def test_non_explanatory_requests_not_flagged(self, msg):
        assert not _is_explanatory_request(msg), (
            f"{msg!r} should stay actionable"
        )


class TestTurnIntentClassification:
    def test_review_prompt_classifies_read_only(self):
        intent = _classify_turn_intent("review this repo for bugs")
        assert intent.kind.value == "review"
        assert intent.read_only_by_default is True
        assert not intent.explicit_execution

    def test_audit_prompt_is_read_only(self):
        assert _is_read_only_request("audit this project for security risks")

    def test_review_run_prompt_preserves_execution_override(self):
        intent = _classify_turn_intent(
            "run the tests and review the failures",
        )
        assert intent.kind.value == "review"
        assert intent.explicit_execution is True
        assert _is_read_only_request(
            "run the tests and review the failures",
        ) is False


class TestShellCommandFailure:
    """_shell_command_failure catches shell-level execution failures."""

    @pytest.mark.parametrize("output,expected", [
        (
            "bash: line 1: pytest: command not found",
            "command not found",
        ),
        ("pytest: not found", "command not found"),
        (
            "ModuleNotFoundError: No module named 'pytest'",
            "missing Python module",
        ),
        (
            "ImportError: No module named pytest",
            "missing Python module",
        ),
        (
            "python3: can't open file '//snake.html': [Errno 2] No such "
            "file or directory",
            "file or directory not found",
        ),
        (
            "exec: \"go\": executable file not found in $PATH",
            "executable not in PATH",
        ),
    ])
    def test_detects_shell_failures(self, output, expected):
        assert _shell_command_failure(output) == expected

    @pytest.mark.parametrize("output", [
        "",
        "3 passed, 0 failed",
        "test_foo: PASSED",
        "1 test passed\nAll green",
        # Contains "error" as a test name, not a shell failure
        "def test_error_handling passed",
    ])
    def test_legitimate_output_not_flagged(self, output):
        assert _shell_command_failure(output) is None


@pytest.mark.asyncio
async def test_test_run_tool_reports_error_on_missing_pytest():
    """TestRunTool.execute must surface a shell-level failure when
    pytest isn't installed, instead of reporting Passed=1."""
    class _FakeCM:
        async def run_command(self, *args, **kwargs):
            return await self._run_command(*args, **kwargs)

        async def _run_command(self, ws, cmd, timeout):
            return "bash: line 1: pytest: command not found\n"

    state = CoderState(
        session_id="s", workspace_id="w", phase=CoderPhase.EXECUTING,
    )
    tool = _TestRunTool(
        container_manager=_FakeCM(), workspace_id="ws-x", state=state,
    )
    result = await tool.execute(command="pytest -x")

    assert result.success is False, (
        f"pytest-not-found must NOT report success. Output: {result.output!r}"
    )
    assert "command not found" in (result.error or "")
    # The old buggy fallback produced "Passed: 1"; the new code must not.
    assert "Passed: 1" not in (result.output or "")


@pytest.mark.asyncio
async def test_test_run_tool_fallback_errors_on_ambiguous_output():
    """When no framework parser matches AND output is ambiguous, the
    fallback now defaults to errors=1 (was passed=1 pre-fix)."""
    class _FakeCM:
        async def run_command(self, *args, **kwargs):
            return await self._run_command(*args, **kwargs)

        async def _run_command(self, ws, cmd, timeout):
            # Output with no pass/fail markers, no error keywords — the
            # old fallback returned passed=1 here.
            return "Running custom checker...\nDone.\n"

    state = CoderState(
        session_id="s", workspace_id="w", phase=CoderPhase.EXECUTING,
    )
    tool = _TestRunTool(
        container_manager=_FakeCM(), workspace_id="ws-x", state=state,
    )
    result = await tool.execute(command="./custom_check.sh")
    assert result.success is False


# ---------------------------------------------------------------------------
# Fix 2: soft-failure tracking + sticky render
# ---------------------------------------------------------------------------


class TestSoftFailureTracking:
    """CoderState.record_tool_failure dedupes by (tool, target)."""

    def _state(self) -> CoderState:
        return CoderState(
            session_id="s", workspace_id="w", phase=CoderPhase.EXECUTING,
        )

    def test_records_single_failure(self):
        s = self._state()
        s.record_tool_failure(
            tool_name="code_edit", target="/snake.html",
            error="Cannot edit: file not read",
        )
        assert len(s.recent_tool_failures) == 1
        assert s.recent_tool_failures[0]["count"] == 1
        assert s.recent_tool_failures[0]["target"] == "/snake.html"

    def test_dedupes_same_tool_and_target(self):
        s = self._state()
        for _ in range(5):
            s.record_tool_failure(
                tool_name="code_edit", target="/snake.html",
                error="Cannot edit: stale read",
            )
        assert len(s.recent_tool_failures) == 1
        assert s.recent_tool_failures[0]["count"] == 5

    def test_tracks_same_tool_different_targets(self):
        s = self._state()
        s.record_tool_failure(
            tool_name="code_edit", target="/snake.html",
            error="Cannot edit",
        )
        s.record_tool_failure(
            tool_name="code_edit", target="/fib.html",
            error="Cannot edit",
        )
        assert len(s.recent_tool_failures) == 2

    def test_fifo_cap(self):
        s = self._state()
        for i in range(10):
            s.record_tool_failure(
                tool_name="code_edit", target=f"/f{i}.html",
                error="err",
            )
        assert len(s.recent_tool_failures) <= 4

    def test_clear_tool_failures(self):
        s = self._state()
        s.record_tool_failure(
            tool_name="file_read", target="/x", error="ENOENT",
        )
        s.clear_tool_failures()
        assert s.recent_tool_failures == []

    def test_serialization_roundtrip(self):
        s = self._state()
        s.record_tool_failure(
            tool_name="code_edit", target="/x.py", error="stale read",
        )
        d = s.to_dict()
        restored = CoderState.from_row(d)
        assert restored.recent_tool_failures == s.recent_tool_failures

    # -----------------------------------------------------------------
    # Phase 2.2 — cross-turn persistence + TTL aging
    # -----------------------------------------------------------------

    def test_prune_stale_drops_old_entries(self):
        """Entries with last_at older than ttl_seconds get dropped."""
        import time as _time
        s = self._state()
        s.record_tool_failure(
            tool_name="code_edit", target="/old.py", error="stale",
        )
        # Manually age the entry past TTL
        s.recent_tool_failures[0]["last_at"] = _time.time() - 7200  # 2h ago
        dropped = s.prune_stale_tool_failures(ttl_seconds=1800)  # 30 min TTL
        assert dropped == 1
        assert s.recent_tool_failures == []

    def test_prune_keeps_fresh_entries(self):
        """Recent entries (within TTL) survive prune."""
        s = self._state()
        s.record_tool_failure(
            tool_name="code_edit", target="/fresh.py", error="stale",
        )
        # last_at is now() — should survive a 30 min TTL prune trivially
        dropped = s.prune_stale_tool_failures(ttl_seconds=1800)
        assert dropped == 0
        assert len(s.recent_tool_failures) == 1

    def test_record_auto_prunes_before_inserting(self):
        """record_tool_failure prunes before checking dedupe so an old
        entry for the same (tool, target) isn't refreshed by a new
        record — it gets aged out, and the new record creates a fresh
        entry. Without this, a tool that fails once, succeeds for an
        hour, then fails again would carry a count of 2 instead of 1."""
        import time as _time
        s = self._state()
        s.record_tool_failure(
            tool_name="code_edit", target="/x.py", error="first",
        )
        # Age the existing entry past TTL
        s.recent_tool_failures[0]["last_at"] = _time.time() - 7200
        # New failure on same key — old should be pruned, count starts at 1
        s.record_tool_failure(
            tool_name="code_edit", target="/x.py", error="much later",
        )
        assert len(s.recent_tool_failures) == 1
        assert s.recent_tool_failures[0]["count"] == 1
        assert s.recent_tool_failures[0]["error"] == "much later"

    def test_recurring_failures_stay_fresh(self):
        """A failure that keeps happening should keep refreshing
        last_at via the dedupe path — it never ages out as long as
        the model keeps hitting it."""
        s = self._state()
        for _ in range(3):
            s.record_tool_failure(
                tool_name="code_edit", target="/recurring.py", error="err",
            )
        # All three hits collapsed; last_at is fresh
        assert len(s.recent_tool_failures) == 1
        assert s.recent_tool_failures[0]["count"] == 3
        # Pruning at TTL should not affect a fresh entry
        dropped = s.prune_stale_tool_failures(ttl_seconds=1800)
        assert dropped == 0


@pytest.mark.asyncio
async def test_reset_for_new_request_preserves_recent_tool_failures(monkeypatch):
    """Phase 2.2 — failures are now cross-turn, not per-turn. The
    handler's _reset_for_new_request must NOT wipe recent_tool_failures
    (it used to). Stale entries age out via prune; recurring ones
    persist so the next turn's sticky reminder shows the pattern."""
    from augmentum.coder.state import CoderPhase as _CoderPhase
    from augmentum.modes.coder.handler import CoderHandler as _Handler

    handler = _Handler(
        backend=None, session_id="s", workspace_id="w",
        container_manager=None,
    )
    handler._state.phase = _CoderPhase.EXECUTING
    handler._state.record_tool_failure(
        tool_name="code_edit", target="/persistent.py", error="stale read",
    )
    assert len(handler._state.recent_tool_failures) == 1

    handler._reset_for_new_request()
    # Failure entry must survive — pre-2.2 it was wiped here
    assert len(handler._state.recent_tool_failures) == 1
    assert handler._state.recent_tool_failures[0]["target"] == "/persistent.py"


@pytest.mark.asyncio
async def test_reset_for_new_request_prunes_stale_failures(monkeypatch):
    """At reset time, entries older than TTL get pruned automatically
    so a long pause doesn't carry ancient errors forward."""
    import time as _time
    from augmentum.coder.state import CoderPhase as _CoderPhase
    from augmentum.modes.coder.handler import CoderHandler as _Handler

    handler = _Handler(
        backend=None, session_id="s", workspace_id="w",
        container_manager=None,
    )
    handler._state.phase = _CoderPhase.EXECUTING
    handler._state.record_tool_failure(
        tool_name="code_edit", target="/ancient.py", error="from yesterday",
    )
    # Manually age past the 30-min TTL
    handler._state.recent_tool_failures[0]["last_at"] = _time.time() - 7200
    handler._reset_for_new_request()
    assert handler._state.recent_tool_failures == []


class TestSoftFailureTarget:
    """_soft_failure_target picks the right arg per tool."""

    def test_path_tools(self):
        assert _soft_failure_target("file_read", {"path": "/a"}) == "/a"
        assert _soft_failure_target("code_edit", {"path": "/b"}) == "/b"
        assert _soft_failure_target("dir_tree", {"path": "/c"}) == "/c"

    def test_command_tools(self):
        assert _soft_failure_target(
            "shell_exec", {"command": "ls -la"},
        ) == "ls -la"
        assert _soft_failure_target(
            "test_run", {"command": "pytest"},
        ) == "pytest"

    def test_query_tools(self):
        assert _soft_failure_target(
            "code_grep", {"pattern": "foo"},
        ) == "foo"
        assert _soft_failure_target(
            "code_search", {"query": "bar"},
        ) == "bar"

    def test_unknown_tool_returns_empty(self):
        assert _soft_failure_target("exotic_tool", {"x": 1}) == ""

    def test_handles_non_dict_input(self):
        assert _soft_failure_target("file_read", None) == ""


class TestStickyReminderRendersFailures:
    """Repeated soft failures appear in the sticky reminder."""

    def _handler(self) -> CoderHandler:
        return CoderHandler(
            backend=None, session_id="s",
            container_manager=_ExtendedContainerManager(),
            workspace_id="w",
        )

    def test_repeated_failure_appears(self):
        h = self._handler()
        # Three identical rejections → count=3, above the "only show
        # if >=2" threshold.
        for _ in range(3):
            h._state.record_tool_failure(
                tool_name="code_edit", target="/snake.html",
                error="Cannot edit: file not read",
            )
        reminder = h._build_sticky_reminder(
            goal="fix snake", iteration=5, max_iters=100, writes=0,
        )
        assert "Recent repeated failures" in reminder
        assert "code_edit /snake.html" in reminder
        assert "(×3)" in reminder

    def test_single_failure_not_shown(self):
        """A one-off failure is noise; only repeated ones render."""
        h = self._handler()
        h._state.record_tool_failure(
            tool_name="file_read", target="/missing",
            error="ENOENT",
        )
        reminder = h._build_sticky_reminder(
            goal="x", iteration=2, max_iters=100, writes=0,
        )
        assert "Recent repeated failures" not in reminder

    def test_empty_when_no_failures(self):
        h = self._handler()
        reminder = h._build_sticky_reminder(
            goal="g", iteration=1, max_iters=100, writes=0,
        )
        assert "Recent repeated failures" not in reminder


# ---------------------------------------------------------------------------
# Fix 6: force-narrate before stop (fallback summary)
# ---------------------------------------------------------------------------
#
# Observed 2026-04-20 on Qwen 3.6 35B A3B: "can you run it for me?" turn
# ran 15 shell_exec calls trying to build nsnake, never emitted prose,
# user saw an empty "Done" banner with zero context. Tool OUTPUTS are
# invisible in our UI — only tool names render — so silent stops are
# catastrophic UX. Fix: track cumulative prose, emit fallback summary
# when a turn does real work but narrates nothing.


class TestFallbackSummary:
    """Turn-end fallback when the model narrated less than the threshold."""

    def _handler(self) -> CoderHandler:
        return CoderHandler(
            backend=None, session_id="s",
            container_manager=_ExtendedContainerManager(),
            workspace_id="w",
        )

    def test_renders_reason_and_counts(self):
        h = self._handler()
        h._state.tool_calls_made = 7
        out = h._render_fallback_summary(
            iteration=5, total_writes=2,
            termination_reason="model_stop_after_nudge",
            same_file_edits={},
            messages=[],
        )
        assert "stopped after a prompt to continue" in out
        assert "5 iteration" in out
        assert "7 tool call" in out
        assert "2 file write" in out

    def test_renders_thrashed_paths(self):
        h = self._handler()
        out = h._render_fallback_summary(
            iteration=10, total_writes=9,
            termination_reason="same_file_edit_break",
            same_file_edits={"/snake.html": 8, "/fib.html": 1},
            messages=[],
        )
        assert "/snake.html (×8)" in out
        assert "broke on thrashing one file" in out
        # One-count paths shouldn't appear as thrashed
        assert "/fib.html (×1)" not in out

    def test_renders_repeated_failures_from_state(self):
        h = self._handler()
        for _ in range(3):
            h._state.record_tool_failure(
                tool_name="shell_exec",
                target="make",
                error="g++ not found",
            )
        out = h._render_fallback_summary(
            iteration=6, total_writes=0,
            termination_reason="model_stop",
            same_file_edits={},
            messages=[],
        )
        assert "Repeated failures" in out
        assert "shell_exec make" in out
        assert "×3" in out

    def test_extracts_reads_and_writes_from_messages(self):
        h = self._handler()
        msgs = [
            Message(role="assistant", content="", tool_calls=[
                {"id": "tc-1", "function": {
                    "name": "file_read",
                    "arguments": {"path": "/README.md"},
                }},
            ]),
            Message(role="tool", content="content", tool_call_id="tc-1"),
            Message(role="assistant", content="", tool_calls=[
                {"id": "tc-2", "function": {
                    "name": "file_write",
                    "arguments": {"path": "/new.py"},
                }},
            ]),
            Message(role="tool", content="written", tool_call_id="tc-2"),
        ]
        out = h._render_fallback_summary(
            iteration=3, total_writes=1,
            termination_reason="model_stop",
            same_file_edits={},
            messages=msgs,
        )
        assert "/README.md" in out
        assert "/new.py" in out

    def test_renders_last_error(self):
        h = self._handler()
        msgs = [
            Message(
                role="tool",
                content="ERROR: g++: command not found",
                tool_call_id="tc-1",
            ),
        ]
        out = h._render_fallback_summary(
            iteration=2, total_writes=0,
            termination_reason="backend_error",
            same_file_edits={},
            messages=msgs,
        )
        assert "Last error" in out
        assert "g++: command not found" in out

    def test_includes_user_facing_explanation(self):
        """Fallback must end with a 'what to do' line so users aren't
        stranded when the model silently fails."""
        h = self._handler()
        h._state.tool_calls_made = 1
        out = h._render_fallback_summary(
            iteration=1, total_writes=0,
            termination_reason="model_stop",
            same_file_edits={},
            messages=[],
        )
        assert "rerun" in out.lower() or "rephrase" in out.lower()


@pytest.mark.asyncio
async def test_fallback_summary_fires_on_silent_stop(monkeypatch):
    """When the model runs tools but emits no prose, the turn-end
    fallback summary appears so the user isn't left with empty Done."""
    _force_native_tier(monkeypatch)

    fake_tool = _FakeTool("file_read")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_tool],
    )

    class _RunsToolsSilently:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            # First call: run a tool with NO prose
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-1", "file_read", {"path": "/x.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                # Subsequent calls: stop with NO prose (triggers the
                # nudge, then another silent stop → fallback fires)
                yield _FakeChunk(content_delta="")
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _RunsToolsSilently(), session_id="sess-silent-work",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-silent",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("do the thing"), workspace_context="",
    ):
        chunks.append(c)

    fallback_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "fallback_summary"
    ]
    assert fallback_chunks, (
        "Silent work turn must emit a fallback summary"
    )
    body = "".join(
        c.content_delta or "" for c in fallback_chunks
    )
    # Must mention something actionable
    assert "Result" in body
    assert "Activity" in body


@pytest.mark.asyncio
async def test_fallback_summary_skipped_when_model_narrated(monkeypatch):
    """If the model emitted substantive prose, the fallback is redundant
    and must NOT fire. Avoids double-summary noise."""
    _force_native_tier(monkeypatch)

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [],
    )

    long_prose = "Here is a thorough summary of what happened. " * 5

    class _Narrates:
        async def chat_stream(self, request):
            yield _FakeChunk(content_delta=long_prose)
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _Narrates(), session_id="sess-narrated",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-narrated",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("do the thing"), workspace_context="",
    ):
        chunks.append(c)

    fallback_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "fallback_summary"
    ]
    assert not fallback_chunks, (
        "Substantive prose → no fallback. Got: "
        f"{[c.content_delta for c in fallback_chunks]}"
    )


@pytest.mark.asyncio
async def test_fallback_summary_skipped_when_no_work_done(monkeypatch):
    """An empty turn (no tools called, no writes) doesn't need a fallback
    — the user's input just didn't trigger activity. Avoids summarising
    nothing."""
    _force_native_tier(monkeypatch)

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [],
    )

    class _DoesNothing:
        async def chat_stream(self, request):
            yield _FakeChunk(content_delta="")
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _DoesNothing(), session_id="sess-idle",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-idle",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("hmm"), workspace_context="",
    ):
        chunks.append(c)

    fallback_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "fallback_summary"
    ]
    assert not fallback_chunks, "Idle turn shouldn't emit fallback"


def test_new_nudge_text_demands_narrative():
    """The continuation nudges must push the model to output a
    structured report, not vaguely 'continue or explain'. Regression
    guard against drifting back to the old vague wording.

    Updated 2026-05-10 (Phase 3.6): nudge text moved from inline in
    ``_act_hybrid`` to a module-level ``_NUDGE_MESSAGES`` dict keyed
    by ``nudge_kind`` so the gate can pick the variant matching the
    bail mode (insistence / bailout / no-progress). All three variants
    must carry the same hard contract: the model must commit to either
    a tool call or a structured "what you tried / why blocked / what
    you need" report.
    """
    from augmentum.modes.coder.phase_act import (
        NUDGE_BAILOUT,
        NUDGE_INSISTENCE,
        NUDGE_NO_PROGRESS,
        _NUDGE_MESSAGES,
    )

    # Each variant exists.
    assert NUDGE_INSISTENCE in _NUDGE_MESSAGES
    assert NUDGE_BAILOUT in _NUDGE_MESSAGES
    assert NUDGE_NO_PROGRESS in _NUDGE_MESSAGES

    # The no-progress variant is the spiritual successor to the pre-3.6
    # default nudge — preserves the structured "what / why / next" frame
    # that we know works (project memory note: "meaningful-prose stops
    # don't over-nudge"). Pin the key phrases.
    no_progress = _NUDGE_MESSAGES[NUDGE_NO_PROGRESS]
    assert "user CAN'T see" in no_progress
    assert "What you tried" in no_progress
    assert "Next step" in no_progress

    # The insistence variant must reference user-demanded completion
    # so the model understands *why* it's being pushed harder than usual.
    insistence = _NUDGE_MESSAGES[NUDGE_INSISTENCE]
    assert "explicitly" in insistence.lower() or "demanded" in insistence.lower()
    assert "completion" in insistence.lower() or "finished" in insistence.lower()

    # The bailout variant must name the pattern explicitly so the model
    # can correct course. "Bail" or "evasive" framing keeps the nudge
    # actionable rather than generic.
    bailout = _NUDGE_MESSAGES[NUDGE_BAILOUT]
    assert "bail" in bailout.lower()

    # Guard against regressing to the old vague nudge.
    for msg in _NUDGE_MESSAGES.values():
        assert "Continue toward the user's goal, or explain" not in msg


# ---------------------------------------------------------------------------
# Fix 7: vague-improvement detection
# ---------------------------------------------------------------------------


class TestVagueImprovementRequests:
    """'improve it', 'fix it', 'make it better' should be flagged."""

    @pytest.mark.parametrize("msg", [
        "improve it",
        "improve",
        "make it better",
        "make this better",
        "fix it",
        "fix this",
        "clean it up",
        "clean up",
        "refactor",
        "refactor it",
        "optimize it",
        "optimise it",
        "enhance it",
        "polish it",
        "can you improve it?",
        "can you improve it for me?",
        "could you fix this?",
        "would you clean it up please?",
        "help me improve",
        "help me refactor this",
        "make improvements",
        "make some improvements",
        # trailing punctuation / casing
        "Improve it.",
        "Fix it!",
        "IMPROVE IT",
    ])
    def test_detects_vague_improvements(self, msg):
        assert _is_vague_improvement(msg), (
            f"{msg!r} should be detected as vague improvement"
        )
        assert _is_vague_request(msg), (
            f"{msg!r} should also register as vague request"
        )

    @pytest.mark.parametrize("msg", [
        "improve the error handling in auth.py",
        "refactor the login function to use async",
        "fix the bug where users can't submit",
        "clean up unused imports in main.py",
        "optimize the database query in search()",
        "make the tests faster by mocking the network",
        "make it better at handling large inputs",
        # These contain "fix" but with a concrete target
        "fix the race condition in the cache",
    ])
    def test_concrete_requests_not_flagged(self, msg):
        assert not _is_vague_improvement(msg), (
            f"{msg!r} is concrete, should not be flagged"
        )

    def test_clarification_covers_improvement_axes(self):
        text = _generate_clarification("improve it")
        lowered = text.lower()
        # Covers the common "what kind of improvement" axes
        assert "refactor" in lowered
        assert "test" in lowered
        assert "bug" in lowered or "fix" in lowered
        assert "feature" in lowered


# ---------------------------------------------------------------------------
# Fix 8: preemptive repeat-read refusal
# ---------------------------------------------------------------------------


class TestPreemptiveRefusalState:
    """CoderState.repeat_count / clear_tool_calls_for_path semantics."""

    def _state(self) -> CoderState:
        return CoderState(
            session_id="s", workspace_id="w", phase=CoderPhase.EXECUTING,
        )

    def test_repeat_count_starts_zero(self):
        s = self._state()
        assert s.repeat_count(
            tool_name="file_read", tool_input={"path": "/a"},
        ) == 0

    def test_repeat_count_tracks_record_tool_call(self):
        s = self._state()
        for _ in range(4):
            s.record_tool_call(
                tool_name="file_read",
                tool_input={"path": "/a.py"},
                iteration=0,
            )
        assert s.repeat_count(
            tool_name="file_read", tool_input={"path": "/a.py"},
        ) == 4

    def test_repeat_count_scoped_per_tool(self):
        s = self._state()
        s.record_tool_call(
            tool_name="file_read",
            tool_input={"path": "/a.py"},
            iteration=0,
        )
        # Different tool, same path → separate counter
        assert s.repeat_count(
            tool_name="code_grep",
            tool_input={"path": "/a.py", "pattern": "x"},
        ) == 0

    def test_repeat_count_scoped_per_path(self):
        s = self._state()
        s.record_tool_call(
            tool_name="file_read",
            tool_input={"path": "/a.py"},
            iteration=0,
        )
        assert s.repeat_count(
            tool_name="file_read", tool_input={"path": "/b.py"},
        ) == 0

    def test_clear_tool_calls_for_path_resets_counter(self):
        s = self._state()
        for _ in range(5):
            s.record_tool_call(
                tool_name="file_read",
                tool_input={"path": "/a.py"},
                iteration=0,
            )
        s.clear_tool_calls_for_path("/a.py")
        assert s.repeat_count(
            tool_name="file_read", tool_input={"path": "/a.py"},
        ) == 0

    def test_clear_only_affects_named_path(self):
        s = self._state()
        s.record_tool_call(
            tool_name="file_read",
            tool_input={"path": "/a.py"},
            iteration=0,
        )
        s.record_tool_call(
            tool_name="file_read",
            tool_input={"path": "/b.py"},
            iteration=0,
        )
        s.clear_tool_calls_for_path("/a.py")
        assert s.repeat_count(
            tool_name="file_read", tool_input={"path": "/a.py"},
        ) == 0
        assert s.repeat_count(
            tool_name="file_read", tool_input={"path": "/b.py"},
        ) == 1

    def test_clear_noop_when_path_missing(self):
        s = self._state()
        # Should not crash / mutate
        s.clear_tool_calls_for_path("/not-tracked.py")
        assert s.recent_tool_calls == []

    def test_untracked_tools_return_zero(self):
        s = self._state()
        assert s.repeat_count(
            tool_name="task_list", tool_input={"action": "add"},
        ) == 0


@pytest.mark.asyncio
async def test_explanatory_request_nudges_shell_exec_without_blocking():
    """Descriptive requests get a read-only NUDGE, not a hard block.

    Before PR 2 (2026-04-27) this fired a validation_error refusal that
    aborted the call. The classifier overfits substring matches like
    'explain' inside an unrelated request, so blocking caused user-
    visible failures (see transcript bug). The new contract: the
    handler emits an advisory system_reminder and lets the call
    proceed; the model adapts via the same channel it already uses for
    other reminders. counters['validation_errors'] is NOT incremented
    because a nudge is not a malformed call."""
    shell_tool = _FakeTool("shell_exec", output="ran")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-explain-nudge",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-explain-nudge",
    )
    messages = [
        Message(
            role="user",
            content="Can you look at this project to explain what python files it has?",
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-shell",
                "name": "shell_exec",
                "input": {"command": "python /workspace/fun_terminal.py"},
            },
            {"shell_exec": shell_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    # Tool DID run (no longer blocked).
    assert shell_tool.calls == [{"command": "python /workspace/fun_terminal.py"}]
    # Soft nudge was emitted before the call.
    nudges = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "read_only_nudge"
        and c.augmentum.get("read_only_nudge") is True
    ]
    assert nudges, "Expected explanatory read-only nudge to fire"
    assert nudges[0].augmentum.get("intent") == "explanatory"
    # Nudge must NOT pollute the validation-error streak counter.
    assert counters.get("validation_errors", 0) == 0


@pytest.mark.asyncio
async def test_review_request_nudges_shell_exec_without_blocking():
    """Review/audit turns also get a nudge, not a block (parallel of
    the explanatory test above)."""
    shell_tool = _FakeTool("shell_exec", output="ran")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-review-nudge",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-review-nudge",
    )
    messages = [
        Message(
            role="user",
            content="Review this project for bugs and risks.",
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-shell",
                "name": "shell_exec",
                "input": {"command": "pytest -q"},
            },
            {"shell_exec": shell_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    assert shell_tool.calls == [{"command": "pytest -q"}]
    nudges = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "read_only_nudge"
        and c.augmentum.get("read_only_nudge") is True
    ]
    assert nudges, "Expected review read-only nudge to fire"
    assert nudges[0].augmentum.get("intent") == "review/audit"
    assert counters.get("validation_errors", 0) == 0


@pytest.mark.asyncio
async def test_explanatory_request_allows_shell_exec_when_user_explicitly_requests_run():
    """The explanatory guard should yield when the user explicitly says run."""
    shell_tool = _FakeTool("shell_exec", output="ran")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-explain-run",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-explain-run",
    )
    messages = [
        Message(
            role="user",
            content="Run fun_terminal.py and explain what it does.",
        ),
    ]

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-shell",
                "name": "shell_exec",
                "input": {"command": "python /workspace/fun_terminal.py"},
            },
            {"shell_exec": shell_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            {},
        )
    ]

    assert shell_tool.calls == [{"command": "python /workspace/fun_terminal.py"}]
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("read_only_guard") is True
    ]
    assert not guarded


@pytest.mark.asyncio
async def test_explanatory_continue_turn_preserves_prior_run_intent():
    """A later continuation should inherit the prior explicit run request."""
    shell_tool = _FakeTool("shell_exec", output="ran")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-explain-continue-run",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-explain-continue-run",
    )
    messages = [
        Message(
            role="user",
            content="Run fun_terminal.py and explain what it does.",
        ),
        Message(
            role="user",
            content="continue please",
        ),
    ]

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-shell",
                "name": "shell_exec",
                "input": {"command": "python /workspace/fun_terminal.py"},
            },
            {"shell_exec": shell_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            {},
        )
    ]

    assert shell_tool.calls == [{"command": "python /workspace/fun_terminal.py"}]
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("read_only_guard") is True
    ]
    assert not guarded


@pytest.mark.asyncio
async def test_verifier_guard_blocks_install_bootstrap_shell_exec():
    """Verifier turns should not install tooling unless the user asked for setup."""
    shell_tool = _FakeTool("shell_exec", output="should-not-run")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-verifier-install-guard",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-verifier-install-guard",
    )
    handler._controller_power_summary = {
        "id": "test-author",
        "kind": "verifier",
        "display_name": "Test Author",
    }
    messages = [
        Message(
            role="user",
            content="Add focused regression tests for the slug helper and run the relevant verification.",
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-shell",
                "name": "shell_exec",
                "input": {"command": "pip3 install pytest -q"},
            },
            {"shell_exec": shell_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    assert shell_tool.calls == []
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("verifier_guard") is True
    ]
    assert guarded, "Expected verifier guard to block install churn"
    assert counters.get("validation_errors") == 1


@pytest.mark.asyncio
async def test_hybrid_routes_shell_exec_through_serial_guards(monkeypatch):
    """Hybrid must not send shell_exec through the parallel read path.

    shell_exec has permission/verifier/root guards in _run_tool_tracked.
    A full _act_hybrid run should hit those guards, not bypass them via
    the read fanout scheduler.
    """
    _force_native_tier(monkeypatch)
    shell_tool = _FakeTool("shell_exec", output="should-not-run")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [shell_tool],
    )

    class _ShellThenStop:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(
                        0,
                        "tc-shell",
                        "shell_exec",
                        {"command": "pip3 install pytest -q"},
                    ),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
                return
            yield _FakeChunk(
                content_delta=(
                    "I did not install packages because verifier turns must "
                    "avoid bootstrap churn unless the user explicitly asks."
                ),
            )
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _ShellThenStop(),
        session_id="sess-hybrid-shell-serial",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-hybrid-shell-serial",
    )
    handler._controller_power_summary = {
        "id": "test-author",
        "kind": "verifier",
        "display_name": "Test Author",
    }

    chunks = [
        c async for c in handler._act_hybrid(
            _make_request(
                "Add focused regression tests and run the relevant verification.",
            ),
            workspace_context="",
        )
    ]

    assert shell_tool.calls == []
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("verifier_guard") is True
    ]
    assert guarded, "Expected hybrid shell_exec to hit verifier_guard"


@pytest.mark.asyncio
async def test_hybrid_serializes_code_edit_before_test_run(monkeypatch):
    """Stateful calls in one model batch should run in model order."""
    _force_native_tier(monkeypatch)
    order: list[str] = []

    class _OrderedTool(_FakeTool):
        async def execute(self, **kwargs):
            order.append(self.name)
            return await super().execute(**kwargs)

    tools = [
        _OrderedTool("code_edit", output="edited"),
        _OrderedTool("test_run", output="tests passed"),
    ]
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: tools,
    )

    class _EditAndTest:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(
                        0,
                        "tc-edit",
                        "code_edit",
                        {
                            "path": "/workspace/app.py",
                            "search": "old",
                            "replace": "new",
                        },
                    ),
                    _tc_delta(1, "tc-test", "test_run", {}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
                return
            yield _FakeChunk(
                content_delta=(
                    "Updated /workspace/app.py and ran the test command "
                    "successfully. No known gaps remain."
                ),
            )
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _EditAndTest(),
        session_id="sess-hybrid-edit-test-order",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-hybrid-edit-test-order",
    )

    async for _ in handler._act_hybrid(
        _make_request("Edit app.py and then run the tests."),
        workspace_context="",
    ):
        pass

    assert order == ["code_edit", "test_run"]


@pytest.mark.asyncio
async def test_hybrid_persists_turn_summary_for_text_tier_tools(monkeypatch):
    """Structured/text-tier tool results are user messages, not tool roles.

    The hybrid loop should still persist a turn summary so weak-model
    fallback tiers keep cross-turn memory.
    """
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.select_tier",
        lambda backend, model_name: ToolCallingTier.TEXT,
    )
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [
            _FakeTool("file_read", output="file body"),
        ],
    )

    class _TextToolThenAnswer:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(
                    content_delta=(
                        '{"tool":"file_read","input":{"path":"/workspace/a.py"}}'
                    ),
                )
                yield _FakeChunk(done=True, finish_reason="stop")
                return
            yield _FakeChunk(
                content_delta=(
                    "I read /workspace/a.py and found the relevant content. "
                    "No edits were needed for this inspection turn."
                ),
            )
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _TextToolThenAnswer(),
        session_id="sess-text-tier-summary",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-text-tier-summary",
    )

    async for _ in handler._act_hybrid(
        _make_request("Inspect /workspace/a.py."),
        workspace_context="",
    ):
        pass

    assert handler._state.turn_summaries
    assert (
        handler._state.turn_summaries[-1]["user_goal"]
        == "Inspect /workspace/a.py."
    )


@pytest.mark.asyncio
async def test_hybrid_sticky_reminder_uses_effective_tier_cap(monkeypatch):
    """The reminder should display the actual capped budget for this turn."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [],
    )

    class _CaptureRequest:
        def __init__(self):
            self.seen_messages: list[Message] = []

        async def chat_stream(self, request):
            self.seen_messages = list(request.messages)
            yield _FakeChunk(
                content_delta=(
                    "I inspected the request shape and have enough context "
                    "to answer without any tool calls."
                ),
            )
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _CaptureRequest()
    handler = CoderHandler(
        backend,
        session_id="sess-hybrid-sticky-tier-cap",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-hybrid-sticky-tier-cap",
    )
    handler._turn_tier_for_turn = TierClassification(
        tier=Tier.REFLEX,
        reason="unit-test",
    )

    async for _ in handler._act_hybrid(
        _make_request("Give me the quick answer."),
        workspace_context="",
    ):
        pass

    reminder = next(
        m.content for m in reversed(backend.seen_messages)
        if m.role == "user" and m.content.startswith("<system-reminder>")
    )
    assert "Iteration " in reminder
    assert "/2" in reminder
    assert "/150" not in reminder


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        ("dir_tree", {"path": "/workspace", "depth": 4}),
        ("file_list", {"path": "/workspace"}),
        ("shell_exec", {"command": "ls -la /workspace"}),
        ("shell_read", {"command": "find /workspace -maxdepth 2"}),
    ],
)
@pytest.mark.asyncio
async def test_workspace_tree_guard_blocks_root_repo_rediscovery(
    tool_name: str,
    tool_input: dict,
):
    """Populated authoritative repos should not re-list /workspace root."""
    tool = _FakeTool(tool_name, output="should-not-run")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id=f"sess-tree-guard-{tool_name}",
        container_manager=_ExtendedContainerManager(),
        workspace_id=f"ws-tree-guard-{tool_name}",
    )
    handler._workspace_tree_authoritative_for_turn = True
    handler._workspace_tree_file_count_for_turn = 142
    messages = [
        Message(
            role="user",
            content="Review this cloned repository, identify one small real improvement, and implement it.",
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-tree-guard",
                "name": tool_name,
                "input": tool_input,
            },
            {tool_name: tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    assert tool.calls == []
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("workspace_tree_guard") is True
    ]
    assert guarded, f"Expected workspace_tree_guard for {tool_name}"
    assert counters.get("validation_errors") == 1


@pytest.mark.asyncio
async def test_workspace_tree_guard_allows_narrow_subdir_inspection():
    """The guard should only block broad /workspace root rediscovery."""
    tool = _FakeTool("dir_tree", output="src/\n  click/")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-tree-guard-subdir",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-tree-guard-subdir",
    )
    handler._workspace_tree_authoritative_for_turn = True
    handler._workspace_tree_file_count_for_turn = 142
    messages = [
        Message(
            role="user",
            content="Inspect src/click and find one small improvement.",
        ),
    ]

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-tree-subdir",
                "name": "dir_tree",
                "input": {"path": "/workspace/src", "depth": 3},
            },
            {"dir_tree": tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            {},
        )
    ]

    assert tool.calls == [{"path": "/workspace/src", "depth": 3}]
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("workspace_tree_guard") is True
    ]
    assert not guarded


@pytest.mark.asyncio
async def test_workspace_tree_guard_blocks_root_rediscovery_for_cloned_repo_signal():
    """A git-clone workspace should not re-list /workspace root just to rediscover the repo."""
    tool = _FakeTool("dir_tree", output="should-not-run")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-tree-guard-cloned-repo",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-tree-guard-cloned-repo",
    )
    handler._workspace_git_url_for_turn = "https://github.com/pallets/click.git"
    messages = [
        Message(
            role="user",
            content="Review this cloned repository and improve one small real issue.",
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-tree-guard-cloned-repo",
                "name": "dir_tree",
                "input": {"path": "/workspace", "depth": 3},
            },
            {"dir_tree": tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    assert tool.calls == []
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("workspace_tree_guard") is True
    ]
    assert guarded, "Expected workspace_tree_guard for a cloned workspace"
    assert counters.get("validation_errors") == 1


@pytest.mark.asyncio
async def test_workspace_tree_guard_blocks_root_rediscovery_for_root_probe_signal():
    """A successful controller root probe should also block broad /workspace rediscovery."""
    tool = _FakeTool("shell_exec", output="should-not-run")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-tree-guard-root-probe-signal",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-tree-guard-root-probe-signal",
    )
    handler._workspace_root_probe_populated_for_turn = True
    messages = [
        Message(
            role="user",
            content="Review this cloned repository and improve one small real issue.",
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-tree-guard-root-probe-signal",
                "name": "shell_exec",
                "input": {"command": "ls -la /workspace"},
            },
            {"shell_exec": tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    assert tool.calls == []
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("workspace_tree_guard") is True
    ]
    assert guarded, "Expected workspace_tree_guard from root probe signal"
    assert counters.get("validation_errors") == 1


@pytest.mark.asyncio
async def test_workspace_root_probe_marks_repo_and_injects_context():
    """A controller root probe should detect repo-shaped roots before the model re-lists them."""
    class _RepoRootManager(_ExtendedContainerManager):
        async def _run_command(self, workspace_id, cmd, timeout=None):
            raise RuntimeError("force file_list fallback")

        async def file_list(self, workspace_id, path):
            assert path == "/workspace"
            return [
                FileEntry(name=".augmentum", path="/workspace/.augmentum", is_dir=True),
                FileEntry(name=".git", path="/workspace/.git", is_dir=True),
                FileEntry(name="README.md", path="/workspace/README.md", is_dir=False),
                FileEntry(name="src", path="/workspace/src", is_dir=True),
                FileEntry(name="tests", path="/workspace/tests", is_dir=True),
            ]

    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-root-probe",
        container_manager=_RepoRootManager(),
        workspace_id="ws-root-probe",
    )

    await handler._probe_workspace_root_for_turn()

    assert handler._workspace_root_probe_populated_for_turn is True
    assert "README.md" in handler._workspace_root_probe_context_block
    assert "src/" in handler._workspace_root_probe_context_block
    assert "Treat this as an existing repository" in handler._workspace_root_probe_context_block


@pytest.mark.asyncio
async def test_workspace_root_probe_uses_candidate_existence_checks_before_root_listing():
    """The root probe should detect common repo anchors even if /workspace listing is misleading."""
    existing = {
        "/workspace/.git",
        "/workspace/README.md",
        "/workspace/pyproject.toml",
        "/workspace/src",
        "/workspace/tests",
    }
    dirs = {
        "/workspace/.git",
        "/workspace/src",
        "/workspace/tests",
    }

    class _CandidateProbeManager(_ExtendedContainerManager):
        async def _run_command(self, workspace_id, cmd, timeout=None):
            assert workspace_id == "ws-root-probe-candidates"
            if cmd[:2] == ["test", "-e"] and cmd[2] in existing:
                return ""
            if cmd[:2] == ["test", "-d"] and cmd[2] in dirs:
                return ""
            raise RuntimeError("missing")

        async def file_list(self, workspace_id, path):
            raise AssertionError("file_list fallback should not be needed")

    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-root-probe-candidates",
        container_manager=_CandidateProbeManager(),
        workspace_id="ws-root-probe-candidates",
    )

    await handler._probe_workspace_root_for_turn()

    assert handler._workspace_root_probe_populated_for_turn is True
    assert ".git/" in handler._workspace_root_probe_context_block
    assert "README.md" in handler._workspace_root_probe_context_block
    assert "pyproject.toml" in handler._workspace_root_probe_context_block
    assert "src/" in handler._workspace_root_probe_context_block


@pytest.mark.asyncio
async def test_populated_repo_guard_blocks_generic_creation_ask_user():
    """On a populated repo, generic 'what should I build?' questions are invalid."""
    ask_tool = _FakeTool("ask_user", output="should-not-run")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-populated-ask-user",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-populated-ask-user",
    )
    handler._workspace_tree_authoritative_for_turn = True
    handler._workspace_tree_file_count_for_turn = 142
    messages = [
        Message(
            role="user",
            content="Review this cloned repository and improve one small but real issue.",
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-ask-user",
                "name": "ask_user",
                "input": {
                    "questions": [
                        {
                            "prompt": "This workspace appears empty. What would you like me to create?",
                            "options": [
                                "Python web application",
                                "Node.js REST API",
                            ],
                        },
                    ],
                },
            },
            {"ask_user": ask_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    assert ask_tool.calls == []
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("populated_repo_guard") is True
    ]
    assert guarded, "Expected populated_repo_guard to block generic creation ask_user"
    assert counters.get("validation_errors") == 1


@pytest.mark.asyncio
async def test_populated_repo_guard_blocks_clone_repo_script_ask_user():
    """The live empty-repo fallback phrasing should also be blocked on cloned repos."""
    ask_tool = _FakeTool("ask_user", output="should-not-run")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-populated-ask-user-clone-script",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-populated-ask-user-clone-script",
    )
    handler._workspace_git_url_for_turn = "https://github.com/pallets/click.git"
    messages = [
        Message(
            role="user",
            content="Review this cloned repository and improve one small real issue.",
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-ask-user-clone-script",
                "name": "ask_user",
                "input": {
                    "questions": [
                        {
                            "prompt": (
                                "The workspace appears empty - no repository has "
                                "been cloned yet. How would you like me to proceed?"
                            ),
                            "options": [
                                "Clone a specific repository (provide URL)",
                                "Create a sample project to demonstrate the improvement process",
                                "Provide instructions for setting up a different environment",
                            ],
                        },
                    ],
                },
            },
            {"ask_user": ask_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    assert ask_tool.calls == []
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("populated_repo_guard") is True
    ]
    assert guarded, "Expected populated_repo_guard to block clone-repo script ask_user"
    assert counters.get("validation_errors") == 1


@pytest.mark.asyncio
async def test_populated_repo_guard_blocks_clone_repo_script_from_root_probe_signal():
    """The ask_user guard should also fire when controller root probing found a repo."""
    ask_tool = _FakeTool("ask_user", output="should-not-run")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-populated-ask-user-root-probe",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-populated-ask-user-root-probe",
    )
    handler._workspace_root_probe_populated_for_turn = True
    messages = [
        Message(
            role="user",
            content="Review this cloned repository and improve one small real issue.",
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-ask-user-root-probe",
                "name": "ask_user",
                "input": {
                    "questions": [
                        {
                            "prompt": "This workspace appears empty (only contains a workspace guide). What would you like me to build or improve?",
                            "options": [
                                "Build a Python CLI tool",
                                "Build a REST API service",
                                "Paste repository contents here",
                            ],
                        },
                    ],
                },
            },
            {"ask_user": ask_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    assert ask_tool.calls == []
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("populated_repo_guard") is True
    ]
    assert guarded, "Expected populated_repo_guard from root probe signal"
    assert counters.get("validation_errors") == 1


@pytest.mark.asyncio
async def test_populated_repo_guard_allows_narrow_repo_question():
    """Targeted clarification on an existing repo should still be allowed."""
    ask_tool = _FakeTool("ask_user", output="ran")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-populated-ask-user-allow",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-populated-ask-user-allow",
    )
    handler._workspace_tree_authoritative_for_turn = True
    handler._workspace_tree_file_count_for_turn = 142
    messages = [
        Message(
            role="user",
            content="Review this cloned repository and improve one small but real issue.",
        ),
    ]

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-ask-user",
                "name": "ask_user",
                "input": {
                    "questions": [
                        {
                            "prompt": "Do you want the improvement to prioritize docs clarity or test coverage?",
                            "options": [
                                "Docs clarity",
                                "Test coverage",
                            ],
                        },
                    ],
                },
            },
            {"ask_user": ask_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            {},
        )
    ]

    assert ask_tool.calls == [
        {
            "questions": [
                {
                    "prompt": "Do you want the improvement to prioritize docs clarity or test coverage?",
                    "options": ["Docs clarity", "Test coverage"],
                },
            ],
        },
    ]
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("populated_repo_guard") is True
    ]
    assert not guarded


def test_response_contradicts_populated_repo_detects_empty_workspace_claim():
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-populated-response-guard",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-populated-response-guard",
    )
    handler._workspace_tree_authoritative_for_turn = True
    handler._workspace_tree_file_count_for_turn = 142

    assert handler._response_contradicts_populated_repo(
        "This workspace is essentially empty and only contains workspace.md.",
    )
    assert handler._response_contradicts_populated_repo(
        "Please clone a specific repository and provide the URL first.",
    )
    assert handler._response_contradicts_populated_repo(
        "The workspace appears empty and no repository has been cloned yet.",
    )
    assert handler._response_contradicts_populated_repo(
        "I can create a sample project instead if you want.",
    )


def test_response_contradicts_populated_repo_ignores_real_repo_summary():
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-populated-response-allow",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-populated-response-allow",
    )
    handler._workspace_tree_authoritative_for_turn = True
    handler._workspace_tree_file_count_for_turn = 142

    assert not handler._response_contradicts_populated_repo(
        "This looks like an existing Python project with src/, tests/, docs/, and a pyproject.toml.",
    )


def test_workspace_has_populated_repo_for_turn_when_git_clone_workspace():
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-populated-git-url",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-populated-git-url",
    )
    handler._workspace_git_url_for_turn = "https://github.com/pallets/click.git"

    assert handler._workspace_has_populated_repo_for_turn() is True


def test_strip_tool_json_removes_task_list_markup():
    text = (
        "<task_list>\n"
        "{\"items\": [{\"content\": \"Review repo\", \"status\": \"in_progress\"}]}\n"
        "</task_list>\n"
        "Real answer here."
    )
    assert _strip_tool_json(text).strip() == "Real answer here."


@pytest.mark.asyncio
async def test_verifier_guard_blocks_redundant_env_discovery_after_env_info():
    """Once env_info has run, verifier turns should stop shell-probing Python basics."""
    shell_tool = _FakeTool("shell_exec", output="should-not-run")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-verifier-env-guard",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-verifier-env-guard",
    )
    handler._controller_power_summary = {
        "id": "test-author",
        "kind": "verifier",
        "display_name": "Test Author",
    }
    messages = [
        Message(
            role="user",
            content="Create a tiny helper and add one focused regression test.",
        ),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"name": "env_info", "input": {}}],
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-shell",
                "name": "shell_exec",
                "input": {"command": "which python3 || which python"},
            },
            {"shell_exec": shell_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    assert shell_tool.calls == []
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("verifier_guard") is True
    ]
    assert guarded, "Expected verifier guard to block redundant env discovery"
    assert counters.get("validation_errors") == 1


@pytest.mark.asyncio
async def test_verifier_guard_blocks_raw_pytest_shell_exec():
    """Verifier turns should route test commands through test_run, not shell_exec."""
    shell_tool = _FakeTool("shell_exec", output="should-not-run")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-verifier-test-run-guard",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-verifier-test-run-guard",
    )
    handler._controller_power_summary = {
        "id": "test-author",
        "kind": "verifier",
        "display_name": "Test Author",
    }
    messages = [
        Message(
            role="user",
            content="Add one focused regression test and verify it.",
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-shell",
                "name": "shell_exec",
                "input": {"command": "cd /workspace && python3 -m pytest test_slugify.py -v"},
            },
            {"shell_exec": shell_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    assert shell_tool.calls == []
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("verifier_guard") is True
    ]
    assert guarded, "Expected verifier guard to redirect raw test shell commands"
    assert counters.get("validation_errors") == 1


@pytest.mark.asyncio
async def test_browser_verifier_blocks_browser_discovery_shell_exec():
    """Browser verifier turns should not spend shell turns on runtime discovery probes."""
    shell_tool = _FakeTool("shell_exec", output="should-not-run")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-browser-discovery-guard",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-browser-discovery-guard",
    )
    handler._controller_power_summary = {
        "id": "browser-verification",
        "kind": "verifier",
        "display_name": "Browser Verification",
    }
    messages = [
        Message(
            role="user",
            content="Create a simple signup page and sanity-check the browser-facing flow.",
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-shell",
                "name": "shell_exec",
                "input": {"command": "which chromium chromium-browser google-chrome 2>/dev/null || echo 'No browser found'"},
            },
            {"shell_exec": shell_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    assert shell_tool.calls == []
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("verifier_guard") is True
    ]
    assert guarded, "Expected browser verifier to block runtime discovery probes"
    assert counters.get("validation_errors") == 1


@pytest.mark.asyncio
async def test_browser_verifier_blocks_second_browser_shell_attempt():
    """After one browser shell attempt, the verifier should force a different proof path."""
    shell_tool = _FakeTool("shell_exec", output="should-not-run")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-browser-repeat-guard",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-browser-repeat-guard",
    )
    handler._controller_power_summary = {
        "id": "browser-verification",
        "kind": "verifier",
        "display_name": "Browser Verification",
    }
    messages = [
        Message(
            role="user",
            content="Create a simple signup page and sanity-check the browser-facing flow.",
        ),
        Message(
            role="assistant",
            content="",
            tool_calls=[{
                "name": "shell_exec",
                "input": {
                    "command": "xvfb-run --auto-server-name :1 -a chromium --headless file:///workspace/index.html",
                },
            }],
        ),
    ]
    counters: dict = {}

    chunks = [
        c async for c in handler._run_tool_tracked(
            {
                "id": "tc-shell",
                "name": "shell_exec",
                "input": {"command": "xvfb-run --auto-server-name :1 -a chromium --headless file:///workspace/index.html 2>&1 | head -50"},
            },
            {"shell_exec": shell_tool},
            ToolCallingTier.NATIVE,
            messages,
            "test-model",
            counters,
        )
    ]

    assert shell_tool.calls == []
    guarded = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("tool_result", {}).get("verifier_guard") is True
    ]
    assert guarded, "Expected browser verifier to block repeated browser shell attempts"
    assert counters.get("validation_errors") == 1


@pytest.mark.asyncio
async def test_preemptive_refusal_blocks_further_reads(monkeypatch):
    """After _READ_REPEAT_REFUSAL_CAP-1 successful reads of the same
    path, the next read gets refused with validation_error=True and
    the container round-trip is skipped. Cap is env-tunable via
    AUGMENTUM_CODER_READ_REPEAT_CAP; the test computes the expected
    real-call count from the constant so raising the default doesn't
    require re-editing this test."""
    _force_native_tier(monkeypatch)

    # Tool that reports how many times .execute ran
    class _CountingRead:
        def __init__(self):
            self.calls = 0

        @property
        def name(self):
            return "file_read"

        @property
        def description(self):
            return "fake"

        @property
        def category(self):
            from augmentum.tools.base import ToolCategory
            return ToolCategory.CODE

        @property
        def input_schema(self):
            return {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }

        @property
        def timeout(self):
            return 5.0

        async def execute(self, *, path="", **_):
            from augmentum.tools.base import ToolResult
            self.calls += 1
            return ToolResult(success=True, output=f"content-of-{path}")

    tool = _CountingRead()
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [tool],
    )

    # Emit 2 × cap iterations so both "real calls" and "refusals"
    # appear in the stream regardless of the env-var value.
    total_iters = _READ_REPEAT_REFUSAL_CAP * 2

    class _KeepsReading:
        def __init__(self):
            self.iter = 0

        async def chat_stream(self, request):
            self.iter += 1
            if self.iter <= total_iters:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(
                        0, f"tc-{self.iter}", "file_read",
                        {"path": "/workspace/snake.html"},
                    ),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _KeepsReading(), session_id="sess-refuse",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-refuse",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("read stuff"), workspace_context="",
    ):
        chunks.append(c)

    # First (cap - 1) reads run; from the cap-th onwards, refusal.
    # The validation_error_streak breaker will fire after N consecutive
    # refusals, so the total real calls cap out at cap - 1.
    assert tool.calls == _READ_REPEAT_REFUSAL_CAP - 1, (
        f"Expected {_READ_REPEAT_REFUSAL_CAP - 1} real reads before "
        f"refusal, got {tool.calls}"
    )
    refusal_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "tool_result"
        and (c.augmentum.get("tool_result") or {}).get("preemptive_refusal")
    ]
    assert refusal_chunks, "Expected at least one preemptive_refusal chunk"


@pytest.mark.asyncio
async def test_write_clears_repeat_count(monkeypatch):
    """After a mutation, the same path's read counter resets so
    'edit → verify' workflows aren't blocked."""
    _force_native_tier(monkeypatch)

    from augmentum.tools.base import ToolCategory, ToolResult

    class _Read:
        @property
        def name(self): return "file_read"
        @property
        def description(self): return "r"
        @property
        def category(self): return ToolCategory.CODE
        @property
        def input_schema(self):
            return {"type": "object", "properties": {"path": {"type": "string"}}}
        @property
        def timeout(self): return 5.0
        async def execute(self, *, path="", **_):
            return ToolResult(success=True, output=f"content-{path}")

    class _Write:
        @property
        def name(self): return "file_write"
        @property
        def description(self): return "w"
        @property
        def category(self): return ToolCategory.CODE
        @property
        def input_schema(self):
            return {"type": "object", "properties": {
                "path": {"type": "string"}, "content": {"type": "string"},
            }}
        @property
        def timeout(self): return 5.0
        async def execute(self, *, path="", content="", **_):
            return ToolResult(success=True, output="written")

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_Read(), _Write()],
    )

    handler = CoderHandler(
        backend=None, session_id="sess-clear",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-clear",
    )
    # Prime the counter: 2 reads of /foo.py
    for _ in range(2):
        handler._state.record_tool_call(
            tool_name="file_read",
            tool_input={"path": "/foo.py"},
            iteration=0,
        )
    assert handler._state.repeat_count(
        tool_name="file_read", tool_input={"path": "/foo.py"},
    ) == 2

    # Simulate a write having happened
    handler._state.clear_tool_calls_for_path("/foo.py")

    assert handler._state.repeat_count(
        tool_name="file_read", tool_input={"path": "/foo.py"},
    ) == 0


def test_refusal_cap_constant():
    """Guard against someone accidentally turning the threshold off."""
    assert _READ_REPEAT_REFUSAL_CAP >= 2, (
        "Cap must be at least 2 — allowing ≥2 real reads before refusal"
    )
    assert _READ_REPEAT_REFUSAL_CAP <= 5, (
        "Cap above 5 lets thrashing run too long before the hard block"
    )


# ---------------------------------------------------------------------------
# Fix 10: qwen-code ports — cold-start grace + action stagnation
# ---------------------------------------------------------------------------


def test_cold_start_constants_sane():
    """Grace should be small (1-3 iters). Longer lets real inspection
    loops run too long; shorter doesn't grant meaningful exploration."""
    assert 1 <= _INSPECTION_COLD_START_GRACE <= 3

    # Action-stagnation threshold is env-tunable. Default bumped to 20
    # after user feedback that 8 fired too early on coherent work.
    # Lower bound keeps it above pathological-thrashing territory;
    # upper bound keeps the safety backstop meaningful.
    assert 8 <= _ACTION_STAGNATION_BREAK <= 40, (
        "Too few breaks normal debug cycles; too many wastes iterations"
    )


@pytest.mark.asyncio
async def test_action_stagnation_breaks_on_same_tool_variant_args(monkeypatch):
    """When the model calls the SAME tool with DIFFERENT args for 8
    iterations in a row, break with 'action_stagnation'. Our existing
    stagnation detector only catches identical batches — this catches
    the `make`, `make 2>&1`, `make clean` parameter-thrashing pattern."""
    _force_native_tier(monkeypatch)

    fake_tool = _FakeTool("shell_exec")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_tool],
    )

    class _ShellSpammer:
        def __init__(self):
            self.iter = 0
            # Generate enough distinct shell_exec variants to exceed
            # the action-stagnation threshold. Computed dynamically
            # from the env-tunable constant so raising the default
            # doesn't break this test.
            self.num_variants = _ACTION_STAGNATION_BREAK + 2

        async def chat_stream(self, request):
            self.iter += 1
            if self.iter <= self.num_variants:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(
                        0, f"tc-{self.iter}", "shell_exec",
                        {"command": f"make variant-{self.iter}"},
                    ),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _ShellSpammer(), session_id="sess-stag",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-stag",
    )
    chunks: list[InternalStreamChunk] = []
    # NB: goal has no creation verb so the inspection-only detector
    # (which would fire earlier at streak=4) is out of scope; this
    # test isolates the action-stagnation path.
    async for c in handler._act_hybrid(
        _make_request("figure out why the tests are slow"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "action_stagnation_break"
    ]
    assert break_chunks, "Expected action_stagnation_break to fire"
    assert break_chunks[0].augmentum.get("streak") >= _ACTION_STAGNATION_BREAK
    assert break_chunks[0].augmentum.get("tool") == "shell_exec"


@pytest.mark.asyncio
async def test_action_stagnation_exempts_read_only_exploration(monkeypatch):
    """Observed 2026-04-20 false positive on a "explain this codebase"
    task: the model legitimately read 8 distinct files in a row
    (file_read with different paths each time) and got broken out of
    the task. Read-only exploration tools in _PREEMPTIVE_REFUSAL_TOOLS
    have their own per-args protection; identical-args thrashing is
    the real pathology, not distinct-args legitimate exploration.
    Break must NOT fire here."""
    _force_native_tier(monkeypatch)

    fake_read = _FakeTool("file_read")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_read],
    )

    class _ExploresTenFiles:
        def __init__(self):
            self.iter = 0

        async def chat_stream(self, request):
            self.iter += 1
            # Read 10 DIFFERENT files — this is legitimate exploration.
            if self.iter <= 10:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(
                        0, f"tc-{self.iter}", "file_read",
                        {"path": f"/workspace/file{self.iter}.py"},
                    ),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _ExploresTenFiles(), session_id="sess-explore",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-explore",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("explain this codebase"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "action_stagnation_break"
    ]
    assert not break_chunks, (
        "Legitimate multi-file exploration must NOT trigger "
        "action_stagnation — read-only tools with distinct args are "
        "progress, not thrashing"
    )


@pytest.mark.asyncio
async def test_safeguards_off_bypasses_action_stagnation(monkeypatch):
    """With per-workspace ``safeguards_enabled=False``, the same
    parameter-thrashing pattern that fires ``action_stagnation_break``
    in the default-on case must NOT fire — strong models that
    legitimately run long need the soft breakers out of the way.
    The hard iteration ceiling still bounds runaway loops."""
    _force_native_tier(monkeypatch)

    fake_tool = _FakeTool("shell_exec")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_tool],
    )

    class _ShellSpammer:
        def __init__(self):
            self.iter = 0
            self.num_variants = _ACTION_STAGNATION_BREAK + 2

        async def chat_stream(self, request):
            self.iter += 1
            if self.iter <= self.num_variants:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(
                        0, f"tc-{self.iter}", "shell_exec",
                        {"command": f"make variant-{self.iter}"},
                    ),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

    handler = CoderHandler(
        _ShellSpammer(), session_id="sess-sg-off",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-sg-off",
    )
    handler._state.safeguards_enabled = False

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("figure out why the tests are slow"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "action_stagnation_break"
    ]
    assert not break_chunks, (
        "Safeguards-off must bypass action_stagnation_break"
    )


@pytest.mark.asyncio
async def test_action_stagnation_resets_on_different_tool(monkeypatch):
    """Switching to a different tool name resets the streak so normal
    multi-tool work isn't punished."""
    _force_native_tier(monkeypatch)

    read_tool = _FakeTool("file_read")
    shell_tool = _FakeTool("shell_exec")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [read_tool, shell_tool],
    )

    class _Alternator:
        def __init__(self):
            self.iter = 0

        async def chat_stream(self, request):
            self.iter += 1
            # Alternate shell_exec and file_read every few iters.
            # shell x3 → file_read x3 → shell x3 → stop.
            if self.iter <= 9:
                which = "shell_exec" if ((self.iter - 1) // 3) % 2 == 0 else "file_read"
                args = (
                    {"command": f"cmd{self.iter}"} if which == "shell_exec"
                    else {"path": f"/f{self.iter}.py"}
                )
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.iter}", which, args),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _Alternator(), session_id="sess-alt",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-alt",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("explore and build"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "action_stagnation_break"
    ]
    assert not break_chunks, (
        "Alternating tools shouldn't trip the detector"
    )


@pytest.mark.asyncio
async def test_cold_start_grace_allows_initial_exploration(monkeypatch):
    """On a creation goal, the first 2 inspection-only iterations
    shouldn't count toward the inspection_only_streak. Before the fix,
    'make me a snake game' + `dir_tree` + `file_read` was approaching
    the nudge threshold by iter 5; now it's not counted until iter 3."""
    _force_native_tier(monkeypatch)

    read_tool = _FakeTool("file_read")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [read_tool],
    )

    class _TwoReadsThenStop:
        def __init__(self):
            self.iter = 0

        async def chat_stream(self, request):
            self.iter += 1
            # Two iterations of reads (within cold-start grace), then
            # model stops with a prose plan.
            if self.iter <= 2:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(
                        0, f"tc-{self.iter}", "file_read",
                        {"path": f"/file{self.iter}.py"},
                    ),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                # Substantive prose stop — should terminate cleanly
                # without the inspection-loop nudge firing.
                yield _FakeChunk(
                    content_delta="Here's my plan: I'll create a snake game in snake.html using HTML canvas. Starting with the boilerplate now.",
                )
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _TwoReadsThenStop(), session_id="sess-cold",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-cold",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("make me a snake game"), workspace_context="",
    ):
        chunks.append(c)

    # The inspection-loop nudge should NOT have fired during cold start.
    inspection_nudges = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "inspection_loop_nudge"
    ]
    assert not inspection_nudges, (
        "Cold-start grace should prevent the nudge during first 2 iters"
    )
    # And the inspection-loop break should definitely not fire.
    inspection_breaks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "inspection_loop_break"
    ]
    assert not inspection_breaks


# ---------------------------------------------------------------------------
# Fix 11: intent-without-action + content-loop detection
# ---------------------------------------------------------------------------
#
# Observed 2026-04-20 on Qwen 3.6: model writes shell commands inside
# ```bash``` fences as prose, emits zero tool calls, stops. Our
# continuation judge saw substantive prose and fired
# model_stop_with_answer — but nothing actually ran. Same session also
# exhibited within-response decoding loops ("Let me run these now.
# Running diagnostics: ...` Let me run these now. Running diagnostics:
# ...` Let me run these..."). Neither Codex nor Claude Code defends
# against either pattern.


class TestHasUnclaimedCodeBlock:

    @pytest.mark.parametrize("text", [
        "I'll run these:\n```bash\ncurl http://localhost\n```",
        "```sh\nls -la\n```",
        "```shell\nmake && make install\n```",
        "```zsh\necho hi\n```",
        "```console\n$ pwd\n```",
        # case-insensitive
        "```BASH\ncommand\n```",
    ])
    def test_detects_action_fences(self, text):
        assert _has_unclaimed_code_block(text)

    @pytest.mark.parametrize("text", [
        "plain prose without any code",
        "```python\nprint('hi')\n```",
        "```javascript\nconsole.log()\n```",
        "```\nunfenced block\n```",
        "",
    ])
    def test_ignores_non_action_fences(self, text):
        assert not _has_unclaimed_code_block(text)


class TestHasContentLoop:

    def test_detects_repeated_substring(self):
        segment = "Let me run these now. Running diagnostics now: curl http. "
        text = segment * 3 + " additional prose that extends the length. "
        assert _has_content_loop(text)

    def test_ignores_short_text(self):
        # Short text is never flagged even if it repeats.
        assert not _has_content_loop("hi hi hi hi hi hi hi hi hi hi")

    def test_ignores_normal_varied_prose(self):
        text = (
            "This file implements the user authentication flow. The login "
            "path handles credentials via Argon2id and issues an opaque "
            "session token. The logout path invalidates the token and "
            "clears any cached state. Rate limiting is enforced at the "
            "middleware layer via a sliding window counter."
        )
        assert not _has_content_loop(text)

    def test_requires_min_repeats(self):
        # Two occurrences shouldn't fire — only three.
        segment = "This is a specific phrase that will be repeated. " * 2 + (
            " different content of sufficient length to reach the threshold "
            "for content-loop evaluation without repeating again."
        )
        assert not _has_content_loop(segment)

    def test_threshold_exactly_three(self):
        segment = "Exactly the same chunk of words repeating here with enough tokens. "
        text = segment * 3 + " extra prose for length buffer. " * 3
        assert _has_content_loop(text)


@pytest.mark.asyncio
async def test_unclaimed_code_block_forces_nudge(monkeypatch):
    """When the model emits a ```bash``` fence in prose with zero
    tool calls, the continuation judge must NOT accept it as
    delivered_answer — a targeted nudge must fire instead."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("shell_exec")],
    )

    # Model emits bash-block prose and no tools on the FIRST call.
    # On the SECOND call (post-nudge), model stops silently — this
    # lets the loop exit via model_stop_after_nudge.
    class _BashInProse:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(content_delta=(
                    "I'll diagnose the server now. Let me run a curl "
                    "to check if it's responding:\n\n"
                    "```bash\n"
                    "curl -v http://localhost:8000/snake.html\n"
                    "```\n\n"
                    "This should tell us whether the server is actually "
                    "bound and listening on port 8000 from inside the "
                    "container. If the connection is refused, we'll "
                    "need to rebind to 0.0.0.0 and map the port."
                ))
                yield _FakeChunk(done=True, finish_reason="stop")
            else:
                yield _FakeChunk(content_delta="")
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _BashInProse(), session_id="sess-bash",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-bash",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("diagnose the web server"), workspace_context="",
    ):
        chunks.append(c)

    # The unclaimed-code-block nudge must fire.
    nudge_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "unclaimed_code_block_nudge"
    ]
    assert nudge_chunks, (
        "Expected unclaimed_code_block_nudge when bash is in prose "
        "but no shell_exec tool call was emitted"
    )
    assert nudge_chunks[0].augmentum.get("unclaimed_code") is True


@pytest.mark.asyncio
async def test_content_loop_forces_nudge(monkeypatch):
    """When the model's response contains a repeated n-gram 3+ times,
    the judge must NOT accept it as delivered_answer."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [],
    )

    class _RepeatsItself:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                # Same 8-token substring three times. Must be long
                # enough to exceed the min-text-chars floor.
                segment = (
                    "Let me run these diagnostics now to check the "
                    "server status very carefully. "
                )
                yield _FakeChunk(content_delta=segment * 3 + (
                    " Additional filler prose to ensure the total "
                    "response exceeds the minimum text-char threshold "
                    "and the detector actually fires."
                ))
                yield _FakeChunk(done=True, finish_reason="stop")
            else:
                yield _FakeChunk(content_delta="")
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _RepeatsItself(), session_id="sess-loop",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-loop",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("explain the issue"), workspace_context="",
    ):
        chunks.append(c)

    nudge_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "content_loop_nudge"
    ]
    assert nudge_chunks, (
        "Expected content_loop_nudge when response repeats itself"
    )
    assert nudge_chunks[0].augmentum.get("content_loop") is True


def test_codex_prompt_line_present_in_act_system():
    """Regression guard for the Codex-ported 'implement, don't
    describe' rule in the act system prompt."""
    from augmentum.coder.prompts import ACT_SYSTEM
    # The key phrase — wording should survive minor rewording, but the
    # intent ("don't describe, implement") must stay.
    assert "Implement" in ACT_SYSTEM and "describe" in ACT_SYSTEM
    # Strong signal that bash-in-fences is specifically called out.
    assert "```bash" in ACT_SYSTEM or "markdown" in ACT_SYSTEM.lower()


# ---------------------------------------------------------------------------
# Fix 12: plan-question short-circuit
# ---------------------------------------------------------------------------
#
# Observed 2026-04-20 on "Write tests for the main functionality":
# plan phase correctly emitted "Question: What kind of tests? - Unit
# tests - Integration tests - ..." but act phase ran anyway. The model
# then burned 6 iterations exploring and broke on inspection-only-loop.
# The question was correct; the follow-through was wrong.


class TestPlanIsQuestion:

    @pytest.mark.parametrize("text", [
        "Question: What kind of tests?",
        "Question: Which file?\n- Option A\n- Option B",
        "   Question: Which framework?",
        "QUESTION: case-insensitive match",
        # leading preamble is tolerated
        "Let me think about this.\n\nQuestion: What kind of task?",
    ])
    def test_detects_question_format(self, text):
        assert _plan_is_question(text)

    @pytest.mark.parametrize("text", [
        "Plan: do the thing\n\n1. Step one\n2. Step two",
        "# Plan\n\nStep 1: ...",
        "",
        "Just prose with no structure.",
        # Way-long preamble — questions shouldn't buried deep
        ("x " * 500) + "Question: something",
    ])
    def test_plan_format_passes_through(self, text):
        assert not _plan_is_question(text)


@pytest.mark.asyncio
async def test_plan_phase_suppresses_thinking_delta_preamble(monkeypatch):
    """Reasoning-model backends emit their "The user is asking…Plan:…"
    prose via ``thinking_delta`` (not ``content_delta``). The plan
    phase's marker filter must apply to BOTH channels — otherwise the
    preamble leaks into the UI's thinking bubble even though the
    content filter suppressed it. Regression guard for the 2026-04-22
    Glowstone transcript where "The user is asking 'what is this?'…"
    showed up ahead of the real plan."""
    _force_native_tier(monkeypatch)

    class _ReasoningPlanEmitter:
        """Backend that emits the preamble via thinking_delta and the
        plan body via content_delta — mirrors DeepSeek-R1 / Qwen /
        Gemma reasoning output."""

        async def chat_stream(self, request):
            yield _FakeChunk(thinking_delta=(
                "The user is asking 'what is this?' which is an "
                "informational request about the project. I need to "
                "read relevant files to understand and summarise. "
            ))
            yield _FakeChunk(
                thinking_delta="Plan: describe project contents",
                content_delta=(
                    "Plan: describe project contents\n"
                    "1. Read README.md\n"
                    "2. Summarise what the project is"
                ),
            )
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _ReasoningPlanEmitter(), session_id="sess-preamble",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-preamble",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._plan_phase(
        _make_request("hey what is this?"),
        _make_turn_context(),
    ):
        chunks.append(c)

    # Contract update 2026-07-02 (live reasoning relay): the pre-marker
    # preamble is no longer DISCARDED — it streams as ``reasoning_delta``
    # chunks, which the UI renders in the collapsible reasoning block.
    # What must still hold: nothing pre-marker reaches the channels the
    # plan bubble renders (content_delta, and thinking_delta on
    # non-reasoning chunks).
    all_content = "".join(c.content_delta or "" for c in chunks)
    plan_bubble_thinking = "".join(
        c.thinking_delta or "" for c in chunks
        if (c.augmentum or {}).get("status") != "reasoning_delta"
    )
    reasoning = "".join(
        c.thinking_delta or "" for c in chunks
        if (c.augmentum or {}).get("status") == "reasoning_delta"
    )
    combined = all_content + plan_bubble_thinking

    assert "The user is asking" not in combined, (
        "Plan-phase filter must keep the preamble out of BOTH "
        "content_delta AND the plan-bubble thinking channel — reasoning "
        "models leak through thinking_delta and the UI renders it inline"
    )
    assert "I need to read" not in combined
    # The plan body itself must still reach the UI
    assert "Plan: describe project contents" in combined
    # ...and the preamble is surfaced as reasoning (collapsible block),
    # not silently dropped.
    assert "The user is asking" in reasoning


@pytest.mark.asyncio
async def test_plan_question_skips_act_phase(monkeypatch):
    """When _plan_phase emits 'Question: ...', handle_stream must NOT
    run the act phase — just stream the question and end the turn."""
    _force_native_tier(monkeypatch)

    # Fake tool present so the test can fail loudly if act phase
    # sneaks in and calls it.
    act_tool = _FakeTool("file_read")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [act_tool],
    )

    class _QuestionEmitter:
        """Backend that emits a question during plan phase."""
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            # First call is plan phase — emit a Question.
            if self.calls == 1:
                yield _FakeChunk(content_delta=(
                    "Question: What kind of tests would you like?\n"
                    "- Unit tests\n"
                    "- Integration tests\n"
                    "- Both"
                ))
                yield _FakeChunk(done=True, finish_reason="stop")
            else:
                # Any subsequent call is the act phase — shouldn't
                # happen under correct behaviour.
                yield _FakeChunk(
                    content_delta="act phase should not have started",
                )
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _QuestionEmitter()
    handler = CoderHandler(
        backend, session_id="sess-q",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-q",
    )
    # Phase starts at WAITING so plan phase runs.
    from augmentum.coder.state import CoderPhase as _CP
    handler._state.phase = _CP.WAITING

    chunks: list[InternalStreamChunk] = []
    async for c in handler.handle_stream(
        _make_request("write tests for the main functionality"),
    ):
        chunks.append(c)

    # The question must be streamed to the user.
    all_content = "".join(
        c.content_delta or "" for c in chunks
    )
    assert "Question:" in all_content
    assert "Unit tests" in all_content

    # Act phase must NOT have run — only the single plan-phase backend
    # call should have happened.
    assert backend.calls == 1, (
        f"Act phase ran when it shouldn't have; backend called {backend.calls}× "
        "(plan + act instead of plan-only)"
    )

    # The handler should emit a planning.question meta chunk.
    question_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("phase") == "planning"
        and c.augmentum.get("status") == "question"
    ]
    assert question_chunks, "Expected planning.question meta chunk"

    # State is back at WAITING for the next user turn.
    assert handler._state.phase == _CP.WAITING


@pytest.mark.asyncio
async def test_continue_after_question_does_not_resume_stale_question(monkeypatch):
    """A prior clarification question is not resumable objective state.

    Regression for traces where the model asked a good question, then a later
    "continue" revived that stale question/plan instead of entering a fresh
    plan→act turn.
    """
    _force_native_tier(monkeypatch)

    class _QuestionThenPlan:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(content_delta=(
                    "Question: What would you like to know?\n"
                    "- The workspace container\n"
                    "- The repository\n"
                    "- Both"
                ))
                yield _FakeChunk(done=True, finish_reason="stop")
                return

            yield _FakeChunk(content_delta=(
                "Plan: inspect the environment\n"
                "1. Run env_info\n"
                "2. Run dir_tree on /workspace\n"
                "3. Summarize the environment"
            ))
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [],
    )

    backend = _QuestionThenPlan()
    cm = _ExtendedContainerManager()

    async def _cancel_workspace_execs(_workspace_id):
        return 0

    cm.cancel_workspace_execs = _cancel_workspace_execs
    handler = CoderHandler(
        backend, session_id="sess-q-continue",
        container_manager=cm,
        workspace_id="ws-q-continue",
    )

    first_chunks: list[InternalStreamChunk] = []
    async for c in handler.handle_stream(
        _make_request("write tests for the main functionality"),
    ):
        first_chunks.append(c)

    assert any(
        c.augmentum and c.augmentum.get("status") == "question"
        for c in first_chunks
    )
    assert handler._state.plan == ""
    assert handler._state.tasks == []

    second_chunks: list[InternalStreamChunk] = []
    async for c in handler.handle_stream(
        _make_request("continue"),
    ):
        second_chunks.append(c)

    all_content = "".join(c.content_delta or "" for c in second_chunks)
    assert "Plan: inspect the environment" in all_content
    continuation_chunks = [
        c for c in second_chunks
        if c.augmentum and c.augmentum.get("status") == "continuation"
    ]
    assert not continuation_chunks
    assert backend.calls >= 2


@pytest.mark.asyncio
async def test_plan_with_real_plan_runs_act(monkeypatch):
    """Regression guard: when plan phase emits a real 'Plan: ...', the
    act phase must run normally. Guards against over-triggering the
    question short-circuit."""
    _force_native_tier(monkeypatch)

    act_tool = _FakeTool("file_read")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [act_tool],
    )

    class _PlanThenAct:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            # Plan-phase call
            if self.calls == 1:
                yield _FakeChunk(content_delta=(
                    "Plan: read README\n\n"
                    "1. file_read /workspace/README.md\n"
                    "2. Summarise the project"
                ))
                yield _FakeChunk(done=True, finish_reason="stop")
            else:
                # Act-phase call — emit prose so the turn terminates
                # via model_stop_with_answer.
                yield _FakeChunk(content_delta=(
                    "This project is a test repository. "
                    "It contains a README explaining the purpose."
                ))
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _PlanThenAct()
    handler = CoderHandler(
        backend, session_id="sess-p",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-p",
    )
    from augmentum.coder.state import CoderPhase as _CP
    handler._state.phase = _CP.WAITING

    chunks: list[InternalStreamChunk] = []
    async for c in handler.handle_stream(
        _make_request("summarise this project"),
    ):
        chunks.append(c)

    # Both plan and act should have hit the backend.
    assert backend.calls >= 2, (
        f"Real plan should run act phase too; backend called {backend.calls}× "
        "(expected plan + at least one act iter)"
    )

    # And the question meta chunk must NOT have fired.
    question_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("phase") == "planning"
        and c.augmentum.get("status") == "question"
    ]
    assert not question_chunks


@pytest.mark.asyncio
async def test_operate_turn_progress_note_without_action_forces_nudge(monkeypatch):
    """Actionable operate turns must not stop on future-action prose alone.

    Regression for the live tunnel run: the model said what it would do next
    ("Now let me install ngrok and start it") and ended the turn without any
    tool call. That should trigger a targeted nudge, not a clean stop.
    """
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("shell_exec")],
    )

    class _PlanThenProgressNote:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(content_delta=(
                    "Plan: switch tunnel providers\n\n"
                    "Step 1: Kill the localtunnel process\n"
                    "Step 2: Try a different tunnel provider\n"
                    "Step 3: Verify the public URL responds"
                ))
                yield _FakeChunk(done=True, finish_reason="stop")
            elif self.calls == 2:
                yield _FakeChunk(content_delta=(
                    "Localtunnel is no longer viable here. "
                    "Now let me install ngrok and start a new tunnel."
                ))
                yield _FakeChunk(done=True, finish_reason="stop")
            else:
                yield _FakeChunk(content_delta=(
                    "ngrok requires an auth token in this environment, "
                    "so I can't expose it that way without new credentials."
                ))
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _PlanThenProgressNote()
    handler = CoderHandler(
        backend, session_id="sess-operate-progress",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-operate-progress",
    )
    handler._state.phase = CoderPhase.WAITING

    chunks: list[InternalStreamChunk] = []
    async for c in handler.handle_stream(
        _make_request("start the app and expose it through a public tunnel"),
    ):
        chunks.append(c)

    assert backend.calls >= 3, (
        "Operate turn should not stop on a progress note with no action; "
        "expected a follow-up backend round after the nudge"
    )
    nudge_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "progress_without_action_nudge"
    ]
    assert nudge_chunks, "Expected progress_without_action_nudge for operate turn"
    assert any(
        "requires an auth token" in (c.content_delta or "")
        for c in chunks
    ), "Expected the follow-up blocker explanation to reach the user"


@pytest.mark.asyncio
async def test_operate_turn_requires_remote_evidence_or_blocker(monkeypatch):
    """Remote-access operate turns must not finish on local-only proof.

    Regression for the tunnel run: the model had a localhost check and a
    printed loca.lt URL, then claimed the app was exposed. That should not
    count as completion until it either verifies the public URL or explains
    the concrete blocker plainly.
    """
    _force_native_tier(monkeypatch)
    shell_tool = _FakeTool(
        "shell_exec",
        output=(
            "your url is: https://bright-rice-sleep.loca.lt\n"
            "Server HTTP: 302\n"
        ),
    )
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [shell_tool],
    )

    class _RemoteAccessEvidenceBackend:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(content_delta=(
                    "Plan: expose the app for remote access\n\n"
                    "1. Start the app locally\n"
                    "2. Create a public tunnel\n"
                    "3. Verify the public URL"
                ))
                yield _FakeChunk(done=True, finish_reason="stop")
            elif self.calls == 2:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(
                        0,
                        "tc-shell-1",
                        "shell_exec",
                        {"command": "lt --port 8080"},
                    ),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            elif self.calls == 3:
                yield _FakeChunk(content_delta=(
                    "The app is running and exposed via tunnel. "
                    "You can access it at https://bright-rice-sleep.loca.lt"
                ))
                yield _FakeChunk(done=True, finish_reason="stop")
            else:
                yield _FakeChunk(content_delta=(
                    "Localtunnel is only giving an interstitial and then bad "
                    "gateway behavior here, so I can't provide a clean remote "
                    "link from this environment."
                ))
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _RemoteAccessEvidenceBackend()
    handler = CoderHandler(
        backend,
        session_id="sess-operate-evidence",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-operate-evidence",
    )
    handler._state.phase = CoderPhase.WAITING

    chunks: list[InternalStreamChunk] = []
    async for c in handler.handle_stream(
        _make_request("clone the repo, run it, and expose it so I can access it remotely"),
    ):
        chunks.append(c)

    assert backend.calls >= 4, (
        "Remote-access operate turn should not stop on a tunnel URL plus "
        "localhost evidence; expected a follow-up round after the nudge"
    )
    evidence_nudges = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "operate_evidence_nudge"
    ]
    assert evidence_nudges, "Expected operate_evidence_nudge for unverified remote access"
    assert any(
        "can't provide a clean remote link" in (c.content_delta or "")
        for c in chunks
    ), "Expected the blocker explanation to reach the user after the nudge"
    assert handler._state.pending_objective_contract == {}


# ---------------------------------------------------------------------------
# Fix 14: workspace path resolution
# ---------------------------------------------------------------------------
#
# Observed 2026-04-20 on a Qwen 3.5 4B trace: model passed relative
# paths (`README.md`, `DTLN_model.py`) to file_read. The tool ran
# `cat README.md` inside the container; _run_command didn't pass
# workdir; cat ran from Docker default (/) and every read returned
# ENOENT. Model burned 10 iters and broke on inspection-only-loop —
# not because of a loop bug, but because every tool call failed.


class TestResolveWorkspacePath:

    @pytest.mark.parametrize("absolute", [
        "/workspace/foo.py",
        "/workspace/src/main.py",
        "/etc/hosts",
        "/",
        "/tmp/x",
    ])
    def test_absolute_paths_unchanged(self, absolute):
        assert _resolve_workspace_path(absolute) == absolute

    @pytest.mark.parametrize("relative,expected", [
        ("README.md", "/workspace/README.md"),
        ("src/main.py", "/workspace/src/main.py"),
        ("./src/x.py", "/workspace/src/x.py"),
        ("./main.py", "/workspace/main.py"),
        # Leading/trailing whitespace is stripped
        ("   foo.py  ", "/workspace/foo.py"),
    ])
    def test_relative_paths_prepended(self, relative, expected):
        assert _resolve_workspace_path(relative) == expected

    def test_home_path_preserved(self):
        # ~/foo is user-intentional; don't hijack it
        assert _resolve_workspace_path("~/foo") == "~/foo"
        assert _resolve_workspace_path("~/.ssh/config") == "~/.ssh/config"

    @pytest.mark.parametrize("disguised,actual", [
        # `..` segments collapse so confinement checks (and humans
        # reading tool logs) see the path that will actually be hit.
        ("/workspace/../etc/passwd", "/etc/passwd"),
        ("/workspace/src/../../tmp/x", "/tmp/x"),
        ("../etc/passwd", "/etc/passwd"),
        ("./src/../main.py", "/workspace/main.py"),
        ("/workspace/./src/x.py", "/workspace/src/x.py"),
        ("src//x.py", "/workspace/src/x.py"),
    ])
    def test_dot_segments_normalized(self, disguised, actual):
        assert _resolve_workspace_path(disguised) == actual

    @pytest.mark.parametrize("empty", ["", "   ", "\t\n", None])
    def test_empty_inputs_return_empty(self, empty):
        if empty is None:
            # resolver is called with string types in practice — guard
            # if somehow called with None
            assert _resolve_workspace_path("") == ""
        else:
            assert _resolve_workspace_path(empty) == ""


def test_container_exec_sets_workdir():
    """Regression guard: the container-layer `_run_command` must pass
    workdir='/workspace' to exec, otherwise relative paths in tool
    commands fail silently. The bug this test covers was the root
    cause of the 2026-04-20 'every file_read returns ENOENT' trace."""
    import inspect
    from augmentum.coder import containers
    source = inspect.getsource(containers.ContainerManager._run_command)
    # The call to container.exec() inside _run_command must carry
    # workdir="/workspace". Without this, ENOENT on every cat call
    # passing a relative path.
    assert 'workdir="/workspace"' in source, (
        "_run_command must pass workdir='/workspace' to container.exec — "
        "otherwise relative paths from tool commands fail silently"
    )


# ---------------------------------------------------------------------------
# Silent-success fog detector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_silent_success_nudge_fires_after_three_consecutive(monkeypatch):
    """When shell_exec returns the literal silent-success marker 3
    iterations in a row, the one-shot silent_success_nudge must fire.
    Catches the "fog" pathology from the 2026-04-22 transcript."""
    _force_native_tier(monkeypatch)

    silent = _FakeTool(
        "shell_exec",
        output="(exit 0, command succeeded with no stdout)",
    )
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [silent],
    )

    class _AllSilent:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls <= 5:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.calls}", "shell_exec",
                              {"command": f"pkill -f x{self.calls}"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _AllSilent(), session_id="sess-silent",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-silent",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("start the server"), workspace_context="",
    ):
        chunks.append(c)

    nudge_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "silent_success_nudge"
    ]
    assert nudge_chunks, (
        "Expected silent_success_nudge after 3 consecutive silent shell "
        "successes — see phase_act.py _SILENT_SUCCESS_NUDGE_AT"
    )
    assert nudge_chunks[0].augmentum.get("streak") >= 3
    # One-shot: the nudge fires exactly once per turn even if more
    # silent iterations happen after.
    assert len(nudge_chunks) == 1


@pytest.mark.asyncio
async def test_shell_exec_build_task_does_not_trip_inspection_break(monkeypatch):
    """Regression guard for the 2026-04-22 Glowstone build transcript:
    user said "build this for me", agent used shell_exec (correctly)
    for gradle/apt-get/which java, and inspection_loop_break fired at
    iter 5 because shell_exec was in _INSPECTION_TOOLS. Removing it
    from the set means build/run/deploy turns that use only shell_exec
    don't trip the detector."""
    _force_native_tier(monkeypatch)

    shell = _FakeTool("shell_exec", output="gradle 8.0 ...")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [shell],
    )

    class _AllShellExec:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls <= 10:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.calls}", "shell_exec",
                              {"command": f"gradle build {self.calls}"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _AllShellExec(), session_id="sess-build",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-build",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("build this for me"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "inspection_loop_break"
    ]
    assert break_chunks == [], (
        "Build tasks using shell_exec must NOT trip inspection_loop_break "
        "— shell_exec is an action, not inspection"
    )


@pytest.mark.asyncio
async def test_shell_read_only_still_trips_inspection_break(monkeypatch):
    """Complement: shell_read STAYS in _INSPECTION_TOOLS because it's
    explicitly read-only. A creation-verb goal where the agent only
    uses shell_read + file_read should still trip the breaker."""
    _force_native_tier(monkeypatch)

    shell_read = _FakeTool("shell_read", output="ok")
    read = _FakeTool("file_read", output="ok")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [shell_read, read],
    )

    class _ReadOnly:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls <= 10:
                name = "shell_read" if self.calls % 2 == 0 else "file_read"
                args = (
                    {"command": f"cat x{self.calls}"} if name == "shell_read"
                    else {"path": f"/f{self.calls}.py"}
                )
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.calls}", name, args),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _ReadOnly(), session_id="sess-readonly",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-readonly",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("create a new feature"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "inspection_loop_break"
    ]
    assert break_chunks, (
        "Creation-verb goal with only shell_read + file_read should "
        "still trip the breaker"
    )


@pytest.mark.asyncio
async def test_failing_shell_without_edit_nudge_fires(monkeypatch):
    """When shell_exec fails 4 iterations in a row with no file writes
    between, the failing_shell_nudge must fire. Catches the "retry the
    same thing expecting different result" trap that no_write_progress
    (requires mutating-tool attempt) and silent_success (requires
    success with empty stdout) both miss."""
    _force_native_tier(monkeypatch)

    bad_shell = _FakeTool("shell_exec", succeeds=False)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [bad_shell],
    )

    class _AlwaysFails:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls <= 6:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.calls}", "shell_exec",
                              {"command": f"cargo build {self.calls}"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _AlwaysFails(), session_id="sess-failshell",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-failshell",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("build this"), workspace_context="",
    ):
        chunks.append(c)

    nudge_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "failing_shell_nudge"
    ]
    assert nudge_chunks, (
        "Expected failing_shell_nudge after 4 consecutive failing "
        "shell_exec calls with no edits between"
    )
    assert nudge_chunks[0].augmentum.get("streak") >= 4
    # One-shot per turn
    assert len(nudge_chunks) == 1


@pytest.mark.asyncio
async def test_failing_shell_resets_on_successful_shell(monkeypatch):
    """A single successful shell_exec resets the failing-shell streak —
    normal debug cycles (fail, fail, succeed, fail) shouldn't trip."""
    _force_native_tier(monkeypatch)

    class _Alternating:
        """Fail / fail / succeed / fail / fail / succeed / ..."""

        def __init__(self):
            self.calls = 0
            from augmentum.tools.base import ToolCategory
            self._cat = ToolCategory.SHELL

        @property
        def name(self):
            return "shell_exec"

        @property
        def description(self):
            return "fake shell"

        @property
        def category(self):
            return self._cat

        @property
        def input_schema(self):
            return {"type": "object", "properties": {}}

        @property
        def timeout(self):
            return 5.0

        async def execute(self, **kwargs):
            from augmentum.tools.base import ToolResult
            self.calls += 1
            if self.calls % 3 == 0:
                return ToolResult(success=True, output="ok, things worked")
            return ToolResult(success=False, error="build failed")

    alt = _Alternating()
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [alt],
    )

    class _Caller:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls <= 9:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.calls}", "shell_exec",
                              {"command": f"cmd{self.calls}"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _Caller(), session_id="sess-alt-shell",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-alt-shell",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("build then verify"), workspace_context="",
    ):
        chunks.append(c)

    nudge_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "failing_shell_nudge"
    ]
    assert not nudge_chunks, (
        "A successful shell every 3 iters must reset the streak; "
        "the nudge should not fire"
    )


def test_turn_summary_records_successful_shell_commands():
    """The turn summary must include shell_commands so the next turn's
    <prior_turns> block can show what was built/installed/ran. Without
    this, a build turn looks like "nothing happened" to the next turn
    and the model re-runs env discovery."""
    from augmentum.modes.coder.handler import CoderHandler
    from augmentum.models.base import Message

    handler = CoderHandler(
        None,
        session_id="sess-turn-summary",
        container_manager=None,
        workspace_id="ws-turn",
    )

    assistant_msg = Message(
        role="assistant",
        content="running the build",
        tool_calls=[
            {
                "id": "tc-1",
                "type": "function",
                "function": {
                    "name": "shell_exec",
                    "arguments": '{"command": "cargo build --release"}',
                },
            },
            {
                "id": "tc-2",
                "type": "function",
                "function": {
                    "name": "shell_exec",
                    "arguments": '{"command": "cargo test"}',
                },
            },
        ],
    )
    tool_msg_1 = Message(role="tool", content="ok", tool_call_id="tc-1")
    tool_msg_2 = Message(role="tool", content="ok", tool_call_id="tc-2")

    summary = handler._build_turn_summary(
        messages=[assistant_msg, tool_msg_1, tool_msg_2],
        user_goal="build the rust project",
        termination_reason="model_stop",
    )

    assert "shell_commands" in summary
    assert "cargo build --release" in summary["shell_commands"]
    assert "cargo test" in summary["shell_commands"]


@pytest.mark.asyncio
async def test_silent_success_resets_on_any_real_output(monkeypatch):
    """A single shell call that returns real stdout resets the streak —
    otherwise a debug cycle (check, check, check, confirm with ls,
    check, check) would wrongly trip the nudge on the second half."""
    _force_native_tier(monkeypatch)

    class _MixedShell:
        """Alternates between silent and noisy outputs."""
        def __init__(self):
            self.calls = 0
            self._name = "shell_exec"
            from augmentum.tools.base import ToolCategory
            self._cat = ToolCategory.SHELL

        @property
        def name(self):
            return "shell_exec"

        @property
        def description(self):
            return "fake shell"

        @property
        def category(self):
            return self._cat

        @property
        def input_schema(self):
            return {"type": "object", "properties": {}}

        @property
        def timeout(self):
            return 5.0

        async def execute(self, **kwargs):
            from augmentum.tools.base import ToolResult
            self.calls += 1
            if self.calls % 2 == 0:
                return ToolResult(
                    success=True,
                    output="some real content here",
                )
            return ToolResult(
                success=True,
                output="(exit 0, command succeeded with no stdout)",
            )

    mixed = _MixedShell()
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [mixed],
    )

    class _Caller:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls <= 6:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.calls}", "shell_exec",
                              {"command": f"cmd{self.calls}"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _Caller(), session_id="sess-mixed",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-mixed",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("run some commands"), workspace_context="",
    ):
        chunks.append(c)

    nudge_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "silent_success_nudge"
    ]
    assert not nudge_chunks, (
        "Alternating silent/noisy shell calls must reset the streak; "
        "the nudge should not fire"
    )


# ---------------------------------------------------------------------------
# Task-list staleness detector (PR2.1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_stale_nudge_fires_when_list_unchanged(monkeypatch):
    """When the model has tasks marked in_progress but never updates
    the list across N iterations, fire a one-shot task_stale_nudge."""
    _force_native_tier(monkeypatch)
    fake_tool = _FakeTool("file_read")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_tool],
    )
    # Tighten threshold so the test doesn't need 8+ iterations.
    monkeypatch.setenv("AUGMENTUM_CODER_TASK_STALE_STREAK", "3")
    import augmentum.coder.task_spine as _ts
    monkeypatch.setattr(_ts, "TASK_STALE_NUDGE_AT", 3)

    class _NeverUpdatesTasks:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            # Always emit the same file_read — model is "working" but
            # never touches task_list.
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, f"tc-{self.calls}", "file_read",
                          {"path": f"/tmp/x{self.calls}.py"}),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _NeverUpdatesTasks(), session_id="sess-stale",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-stale",
    )
    handler._state.set_tasks([
        {"content": "implement feature X", "activeForm": "implementing",
         "status": "in_progress"},
        {"content": "write tests", "activeForm": "testing",
         "status": "pending"},
    ])

    chunks: list[InternalStreamChunk] = []
    iters = 0
    async for c in handler._act_hybrid(
        _make_request("implement feature X"), workspace_context="",
    ):
        chunks.append(c)
        iters += 1
        if iters > 200:  # safety cap
            break

    stale_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "task_stale_nudge"
    ]
    assert stale_chunks, "Expected task_stale_nudge to fire"
    # One-shot — must not fire repeatedly even if model continues to
    # ignore the list.
    assert len(stale_chunks) == 1, (
        f"task_stale_nudge must fire once per turn, got {len(stale_chunks)}"
    )


@pytest.mark.asyncio
async def test_task_stale_nudge_skipped_when_list_empty(monkeypatch):
    """No tasks set → no nudge. The detector only fires when the model
    has committed to a list."""
    _force_native_tier(monkeypatch)
    fake_tool = _FakeTool("file_read")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_tool],
    )
    monkeypatch.setenv("AUGMENTUM_CODER_TASK_STALE_STREAK", "2")
    import augmentum.coder.task_spine as _ts
    monkeypatch.setattr(_ts, "TASK_STALE_NUDGE_AT", 2)

    class _Caller:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls <= 5:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.calls}", "file_read",
                              {"path": "/x.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(content_delta="Done." * 20)
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _Caller(), session_id="sess-empty",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-empty",
    )
    # No tasks set.

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("read a file"), workspace_context="",
    ):
        chunks.append(c)

    stale_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "task_stale_nudge"
    ]
    assert not stale_chunks, (
        "Empty task list must not trigger task_stale_nudge"
    )
