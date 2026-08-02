"""Tests for the 2026-04-20 thrashing-loop breakers + Phase 1.2 tier cap.

Observed in a "what's in this project?" session: the agent spent ~40
iterations running failing `test_run` calls interleaved with edits to
fib.html, then switched to snake.html, then back, with no circuit
breaker firing. User had to ^C multiple times to stop it.

Three fixes added:

  1. ``test_run`` failure streak: 5 consecutive iterations where all
     test_run calls failed and none passed → break the loop.
  2. Single-file edit cap: editing the SAME file 8+ times in one turn
     → break with "agent is thrashing on <path>" message.
  3. PLAN_SYSTEM updated to distinguish INFORMATIONAL / VAGUE /
     ACTIONABLE queries so "what's in this project?" produces a
     read-only plan rather than an edit frenzy.

Run: python -m pytest tests/test_coder_thrashing_breakers.py -v
"""
from __future__ import annotations

import pytest

from augmentum.coder.prompts import PLAN_SYSTEM
from augmentum.models.base import InternalStreamChunk
from augmentum.modes.coder.handler import (
    _SAME_FILE_EDIT_BREAK,
    _TEST_FAILURE_STREAK_BREAK,
    CoderHandler,
)
from augmentum.modes.coder.phase_act import _NO_WRITE_PROGRESS_BREAK
from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeChunk,
    _FakeTool,
    _force_native_tier,
    _make_request,
    _tc_delta,
)

# ---------------------------------------------------------------------------
# test_run failure streak
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streak_of_failing_test_runs_breaks_loop(monkeypatch):
    """When test_run fails _TEST_FAILURE_STREAK_BREAK iters in a row
    with no success, the loop breaks with a user-visible message."""
    _force_native_tier(monkeypatch)
    # test_run tool that always reports failed tests
    failing_test = _FakeTool("test_run", succeeds=False)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [failing_test],
    )

    class _AlwaysRunsFailingTests:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, f"tc-{self.calls}", "test_run", {}),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _AlwaysRunsFailingTests(), session_id="sess-test-fail",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-test-fail",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("run the tests"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "test_failure_streak_break"
    ]
    assert break_chunks, "Expected test_failure_streak_break to fire"
    assert break_chunks[0].augmentum.get("streak") >= _TEST_FAILURE_STREAK_BREAK


@pytest.mark.asyncio
async def test_passing_test_run_resets_streak(monkeypatch):
    """A single passing test_run must reset the streak so normal
    debug-cycle work (3 fails, 1 pass, 3 fails, 1 pass) doesn't break."""
    _force_native_tier(monkeypatch)

    # Alternating tool: first three fail, then one passes, then more fail.
    class _AlternatingTool:
        def __init__(self):
            self.calls = 0

        @property
        def name(self):
            return "test_run"

        @property
        def description(self):
            return "fake test_run"

        @property
        def category(self):
            from augmentum.tools.base import ToolCategory
            return ToolCategory.SHELL

        @property
        def input_schema(self):
            return {"type": "object", "properties": {}}

        @property
        def timeout(self):
            return 5.0

        async def execute(self, **kw):
            from augmentum.tools.base import ToolResult
            self.calls += 1
            # Fail 1-3, pass 4, fail 5-7, pass 8, ...
            if self.calls % 4 == 0:
                return ToolResult(success=True, output="3 passed")
            return ToolResult(success=False, error="tests failed")

    test_tool = _AlternatingTool()
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [test_tool],
    )

    class _AlwaysTests:
        def __init__(self):
            self.iter = 0

        async def chat_stream(self, request):
            self.iter += 1
            # Run ~10 iterations of test_run; one pass every 4 keeps
            # the streak reset, so the break should NOT fire.
            if self.iter <= 10:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.iter}", "test_run", {}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _AlwaysTests(), session_id="sess-alt",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-alt",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("keep testing"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "test_failure_streak_break"
    ]
    assert break_chunks == [], (
        "Streak should reset on passes; break should NOT fire with a "
        "pass every 4 failures"
    )


# ---------------------------------------------------------------------------
# Single-file edit cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thrashing_single_file_edits_break_loop(monkeypatch):
    """When the SAME file is edited _SAME_FILE_EDIT_BREAK times in
    one turn, the loop breaks with a "thrashing on <path>" message."""
    _force_native_tier(monkeypatch)

    edit_tool = _FakeTool("code_edit", output="edit applied")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [edit_tool],
    )

    # Record the file as read so code_edit guard doesn't block
    class _EditLoop:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, f"tc-{self.calls}", "code_edit",
                          {"path": "/workspace/foo.py",
                           "search": "x", "replace": "y"}),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _EditLoop(), session_id="sess-thrash",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-thrash",
    )
    # Pre-register the file as read so code_edit's guard passes
    handler._state.record_file_read("/workspace/foo.py")

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("fix the thing"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "same_file_edit_break"
    ]
    assert break_chunks, "Expected same_file_edit_break to fire"
    assert break_chunks[0].augmentum.get("path") == "/workspace/foo.py"
    assert break_chunks[0].augmentum.get("edit_count") >= _SAME_FILE_EDIT_BREAK


@pytest.mark.asyncio
async def test_edits_across_different_files_do_not_thrash(monkeypatch):
    """The cap is per-path — legit multi-file work shouldn't trigger."""
    _force_native_tier(monkeypatch)

    edit_tool = _FakeTool("code_edit", output="edit applied")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [edit_tool],
    )

    class _MultiFileEdit:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls <= 10:
                # Each iteration touches a DIFFERENT file
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.calls}", "code_edit",
                              {"path": f"/workspace/f{self.calls}.py",
                               "search": "x", "replace": "y"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _MultiFileEdit(), session_id="sess-multi",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-multi",
    )
    for i in range(1, 11):
        handler._state.record_file_read(f"/workspace/f{i}.py")

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("touch every file"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "same_file_edit_break"
    ]
    assert break_chunks == [], (
        "Multi-file edits should NOT trigger same-file-edit-break"
    )


# ---------------------------------------------------------------------------
# PLAN_SYSTEM prompt update
# ---------------------------------------------------------------------------


def test_plan_system_distinguishes_informational_queries():
    """PLAN_SYSTEM must explicitly mention INFORMATIONAL classification
    and tell the model to output a read-only plan for such queries —
    previously the prompt biased every request toward action plans,
    which triggered the "what's in this project?" edit-frenzy bug.

    Note: previously this test also asserted the presence of a worked
    "what's in this project?" example. That example was removed because
    it anchored the model on a fixed env-inspect template — when the
    request was different (e.g. "explain the recommendations") the
    model still copied the example structure verbatim. Abstract
    guidance + the verb list above carries the INFORMATIONAL contract."""
    assert "INFORMATIONAL" in PLAN_SYSTEM
    # At least one representative informational verb should be listed
    assert "what" in PLAN_SYSTEM.lower()
    # The read-only constraint must be stated
    assert "READ-ONLY" in PLAN_SYSTEM or "read-only" in PLAN_SYSTEM


def test_plan_system_still_handles_vague_requests():
    """Backward compat — the VAGUE + clarifying-question path must
    still be described; we added a new branch, didn't delete the old."""
    assert "VAGUE" in PLAN_SYSTEM
    assert "clarifying" in PLAN_SYSTEM.lower() or "Question:" in PLAN_SYSTEM


def test_plan_system_still_handles_actionable_requests():
    """The original ACTIONABLE plan path must still be there."""
    assert "ACTIONABLE" in PLAN_SYSTEM
    assert "Plan:" in PLAN_SYSTEM


def test_plan_system_explicitly_forbids_edits_in_informational_plans():
    """The whole point: informational plans must NOT produce edits.
    Prompt should say this directly so the model doesn't slip into
    action mode when classifying informational."""
    lower = PLAN_SYSTEM.lower()
    # Some phrasing along the lines of "every step must be a read" or
    # "NO edits / NO writes"
    assert (
        "no edits" in lower
        or "no writes" in lower
        or "every step must be a read" in lower
        or "read operation" in lower
    )


# ---------------------------------------------------------------------------
# Write-without-progress detector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failing_edits_with_no_writes_break_loop(monkeypatch):
    """When code_edit is CALLED but every call fails (writes stays 0)
    for _NO_WRITE_PROGRESS_BREAK iterations, the loop must stop with
    ``no_write_progress_break``. Catches the "agent is working but
    nothing is sticking" degenerate state — complement to
    ``inspection_loop_break`` (no writes attempted) and
    ``same_file_edit_break`` (writes landing but thrashing)."""
    _force_native_tier(monkeypatch)

    # Edit tool that ALWAYS returns success=False
    failing_edit = _FakeTool("code_edit", succeeds=False)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [failing_edit],
    )

    class _AlwaysFailsEdit:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            # Rotate through a few paths so same_file_edit_break doesn't
            # race us — this test is specifically for "writes attempted
            # but nothing lands", not "thrashing on one file".
            path = f"/workspace/f{self.calls % 4}.py"
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, f"tc-{self.calls}", "code_edit",
                          {"path": path, "search": "x", "replace": "y"}),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _AlwaysFailsEdit(), session_id="sess-nowrite",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-nowrite",
    )
    # Pre-register the paths as read so code_edit's read-before-write
    # guard doesn't pre-empt with its own validation error (which would
    # steal the iter from the no_write_progress counter).
    for i in range(4):
        handler._state.record_file_read(f"/workspace/f{i}.py")

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("fix the thing"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "no_write_progress_break"
    ]
    assert break_chunks, "Expected no_write_progress_break to fire"
    assert break_chunks[0].augmentum.get("streak") >= _NO_WRITE_PROGRESS_BREAK


@pytest.mark.asyncio
async def test_successful_write_resets_no_write_progress_streak(monkeypatch):
    """One landing edit must reset the streak — the detector is
    specifically for zero-progress pathology, not slow progress. A
    debug cycle of (fail, fail, land, fail, fail, land) should never
    trip this breaker."""
    _force_native_tier(monkeypatch)

    class _AlternatingEdit:
        """Succeed every 4th call — mimics normal debug/iterate."""

        def __init__(self):
            self.calls = 0
            self._name = "code_edit"
            from augmentum.tools.base import ToolCategory
            self._cat = ToolCategory.CODE

        @property
        def name(self):
            return "code_edit"

        @property
        def description(self):
            return "fake code_edit"

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
            if self.calls % 4 == 0:
                return ToolResult(success=True, output="edit applied",
                                  metadata={"path": kwargs.get("path", "")})
            return ToolResult(success=False, error="fake failure")

    alt = _AlternatingEdit()
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [alt],
    )

    class _EditEvery:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            # Bounded so the test terminates even if the detector
            # never fires (which is what we want to verify).
            if self.calls <= _NO_WRITE_PROGRESS_BREAK + 8:
                path = f"/workspace/f{self.calls % 3}.py"
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.calls}", "code_edit",
                              {"path": path, "search": "x", "replace": "y"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _EditEvery(), session_id="sess-reset",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-reset",
    )
    for i in range(3):
        handler._state.record_file_read(f"/workspace/f{i}.py")

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("fix the thing"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "no_write_progress_break"
    ]
    assert break_chunks == [], (
        "Streak should reset on every successful write; break must "
        "NOT fire with a successful edit every 4 iterations"
    )


# ---------------------------------------------------------------------------
# Phase 1.2 — tier-aware iteration cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflex_tier_caps_iterations_below_env_max(monkeypatch):
    """REFLEX tier must bound _act_hybrid below the env-level
    _HYBRID_MAX_ITERS ceiling. Otherwise classifying as reflex is
    cosmetic — the loop still spins for 150 iters."""
    from augmentum.modes.coder.intent import (
        TIER_LIMITS,
        Tier,
        TierClassification,
    )

    _force_native_tier(monkeypatch)
    # Push the env cap up so only the tier cap can stop the loop.
    monkeypatch.setattr(
        "augmentum.modes.coder.phase_act._HYBRID_MAX_ITERS", 100,
    )

    busy_tool = _FakeTool("file_read", succeeds=True)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [busy_tool],
    )

    class _AlwaysWantsMore:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, f"tc-{self.calls}", "file_read",
                          {"file_path": f"x{self.calls}.py"}),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")

        async def chat(self, request):
            return None

    backend = _AlwaysWantsMore()
    handler = CoderHandler(
        backend, session_id="sess-tier-cap",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-tier-cap",
    )
    handler._turn_tier_for_turn = TierClassification(
        tier=Tier.REFLEX, reason="test_setup",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("add the import"), workspace_context="",
    ):
        chunks.append(c)

    reflex_max = TIER_LIMITS[Tier.REFLEX].max_iterations
    assert backend.calls <= reflex_max, (
        f"REFLEX tier should cap at {reflex_max} iterations, "
        f"got {backend.calls} (env max=100, default tier cap would be 25)"
    )


@pytest.mark.asyncio
async def test_composed_tier_default_does_not_tighten_below_env_max(monkeypatch):
    """COMPOSED is the default. With env max set tighter than COMPOSED's
    25-iter cap, the env max should win (tier cap is min'd, not
    forced)."""
    from augmentum.modes.coder.intent import Tier, TierClassification

    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.phase_act._HYBRID_MAX_ITERS", 5,
    )

    busy_tool = _FakeTool("file_read", succeeds=True)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [busy_tool],
    )

    class _AlwaysWantsMore:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, f"tc-{self.calls}", "file_read",
                          {"file_path": f"x{self.calls}.py"}),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")

        async def chat(self, request):
            return None

    backend = _AlwaysWantsMore()
    handler = CoderHandler(
        backend, session_id="sess-tier-composed",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-tier-composed",
    )
    handler._turn_tier_for_turn = TierClassification(
        tier=Tier.COMPOSED, reason="test_setup",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("compose"), workspace_context="",
    ):
        chunks.append(c)

    # Env max=5 is tighter than COMPOSED's 25; env max should win.
    assert backend.calls <= 5, (
        f"Env max should bound the loop when tighter than tier cap, "
        f"got {backend.calls}"
    )


@pytest.mark.asyncio
async def test_reflex_classification_skips_plan_phase(monkeypatch):
    """When classify_tier returns REFLEX, _handle_stream_body must
    bypass plan_phase entirely. Verified by intercepting _plan_phase
    and asserting it's never invoked, plus the 'skipped_reflex'
    metadata chunk is emitted."""
    from augmentum.modes.coder.intent import Tier

    _force_native_tier(monkeypatch)

    plan_calls = {"count": 0}

    async def _spy_plan_phase(self, *args, **kwargs):
        plan_calls["count"] += 1
        if False:
            yield None  # make it an async generator

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.CoderHandler._plan_phase",
        _spy_plan_phase,
    )

    edit_tool = _FakeTool("code_edit", succeeds=True)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [edit_tool],
    )

    class _SingleEditBackend:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-1", "code_edit", {
                        "file_path": "main.py",
                        "search": "x",
                        "replace": "import json\nx",
                    }),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(content_delta="<task_complete/>")
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _SingleEditBackend(), session_id="sess-reflex",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-reflex",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler.handle_stream(
        _make_request("Add the missing json import to main.py"),
    ):
        chunks.append(c)

    assert plan_calls["count"] == 0, (
        f"_plan_phase should be skipped for REFLEX, got {plan_calls['count']} calls"
    )

    skipped_reflex_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "skipped_reflex"
    ]
    assert skipped_reflex_chunks, (
        "Expected at least one 'skipped_reflex' meta chunk on a "
        "REFLEX-classified turn"
    )
    assert skipped_reflex_chunks[0].augmentum.get("tier") == Tier.REFLEX.value


@pytest.mark.asyncio
async def test_non_reflex_classification_does_not_skip_plan(monkeypatch):
    """Symmetric guard: COMPOSED / SURGICAL turns must still go through
    plan_phase. Without this, a future regression that changes the
    tier check could silently disable planning for everyone."""
    _force_native_tier(monkeypatch)

    plan_calls = {"count": 0}

    async def _spy_plan_phase(self, *args, **kwargs):
        plan_calls["count"] += 1
        if False:
            yield None

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.CoderHandler._plan_phase",
        _spy_plan_phase,
    )
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.CoderHandler._act_phase",
        _spy_plan_phase,  # also stub act so the test exits cleanly
    )

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [],
    )

    class _NoOpBackend:
        async def chat_stream(self, request):
            yield _FakeChunk(content_delta="ok")
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _NoOpBackend(), session_id="sess-composed",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-composed",
    )

    chunks: list[InternalStreamChunk] = []
    # "Refactor across the codebase" → COMPOSED
    async for c in handler.handle_stream(
        _make_request("Refactor the auth handler across the codebase"),
    ):
        chunks.append(c)

    assert plan_calls["count"] >= 1, (
        "Plan phase must run for COMPOSED tier; got 0 calls"
    )
    skipped_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "skipped_reflex"
    ]
    assert not skipped_chunks, (
        "skipped_reflex must not fire for non-REFLEX classifications"
    )


# ---------------------------------------------------------------------------
# Phase 3.2 — verification_failed chunk emission
# ---------------------------------------------------------------------------


class _VerifyFailingTool(_FakeTool):
    """Returns success=True but stamps verification_failed in metadata
    — simulates what the real write tools do when a write produces an
    unparseable file. Used to exercise the chunk-emission contract in
    _run_tool_tracked without needing a real ContainerManager + ast
    round-trip."""

    def __init__(self, name: str = "code_edit", *, path: str = "/workspace/x.py"):
        super().__init__(name, succeeds=True, output="edit applied")
        self._path = path

    async def execute(self, **kwargs):
        from augmentum.tools.base import ToolResult
        self.calls.append(dict(kwargs))
        return ToolResult(
            success=True,
            output="edit applied\n\nVerification failed (1 blocking issue(s)):\n  - [python_parse] Syntax error",
            metadata={
                "path":                 self._path,
                "verification_failed":  True,
            },
        )


@pytest.mark.asyncio
async def test_verification_failed_metadata_emits_meta_chunk(monkeypatch):
    """When a write tool sets verification_failed in metadata, the act
    loop must emit a discrete verification_failed chunk so the UI can
    render a trust-failure badge on the iteration card.

    The model-facing path (output append) is covered by tests in
    test_coder_verify_integration.py; this test pins the UI-facing
    chunk emission contract."""
    _force_native_tier(monkeypatch)

    bad_edit_tool = _VerifyFailingTool("code_edit", path="/workspace/bad.py")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [bad_edit_tool],
    )

    class _OneEditThenStop:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-1", "code_edit", {
                        "path":    "/workspace/bad.py",
                        "search":  "x",
                        "replace": "broken",
                    }),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(content_delta="<task_complete/>")
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _OneEditThenStop(), session_id="sess-vf-emit",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-vf-emit",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("fix the bug"), workspace_context="",
    ):
        chunks.append(c)

    vf_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "verification_failed"
    ]
    assert vf_chunks, (
        "Expected exactly one verification_failed meta chunk after the "
        "code_edit that flagged metadata"
    )
    assert len(vf_chunks) == 1
    payload = vf_chunks[0].augmentum
    assert payload.get("tool") == "code_edit"
    assert payload.get("path") == "/workspace/bad.py"
    assert payload.get("tool_call_id") == "tc-1"


@pytest.mark.asyncio
async def test_clean_write_does_not_emit_verification_failed(monkeypatch):
    """Symmetric guard: a successful tool with no verification_failed
    metadata must NOT emit the chunk. Without this, a future regression
    that always emits would silently mark every iteration failed."""
    _force_native_tier(monkeypatch)

    clean_tool = _FakeTool("code_edit", output="edit applied")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [clean_tool],
    )

    class _OneCleanEdit:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-1", "code_edit", {"path": "/workspace/x.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(content_delta="<task_complete/>")
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _OneCleanEdit(), session_id="sess-clean",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-clean",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("simple edit"), workspace_context="",
    ):
        chunks.append(c)

    vf_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "verification_failed"
    ]
    assert not vf_chunks, (
        "verification_failed must not fire on writes with no "
        "verification_failed metadata flag"
    )
