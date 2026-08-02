"""Tests for the ``finish_task`` pseudo-tool.

The tool exists for two reasons:

1. Safe exit under ``tool_choice="required"``. When the plan-prose
   detector escalates a chatty model to forced tool use, the model
   can no longer emit a terminal text response. ``finish_task`` is
   the explicit "I'm done" tool so a genuinely-complete task can
   exit cleanly rather than loop forever.
2. Unambiguous completion signal for weak models. Small LLMs
   routinely repeat themselves because the loop can't tell "I
   answered" from "I restated the question". A tool call is
   structurally distinct from prose and short-circuits the
   ambiguity.

Tests cover:
- Validation: empty summary rejected with a validation_error.
- State mutation: a successful call flips ``finish_requested``
  and stores the ``finish_summary``.
- Loop integration: ``_act_hybrid`` terminates on the next
  iteration with ``termination_reason="finish_task_called"`` and
  emits the summary as the final user-visible prose.
- Per-request reset: a prior turn's finish flag cannot
  short-circuit the first iteration of a new turn.

Run: python -m pytest tests/test_coder_finish_task.py -v
"""
from __future__ import annotations

import asyncio

import pytest

from augmentum.coder.state import CoderState
from augmentum.coder.tools import FinishTaskTool
from augmentum.modes.coder.handler import CoderHandler
from augmentum.models.base import InternalStreamChunk, Message

from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeChunk,
    _FakeTool,
    _force_native_tier,
    _make_request,
    _tc_delta,
)


# ---------------------------------------------------------------------------
# Tool-level behaviour
# ---------------------------------------------------------------------------


def _make_state() -> CoderState:
    return CoderState(session_id="sess-finish", workspace_id="ws-finish")


@pytest.mark.asyncio
async def test_finish_task_rejects_empty_summary():
    """A finish_task with no summary is a validation error — the summary
    IS the user-facing answer, so empty means nothing to show."""
    state = _make_state()
    tool = FinishTaskTool(
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-finish", state=state,
    )

    result = await tool.execute(summary="")
    assert not result.success
    assert result.validation_error
    assert "summary" in (result.error or "").lower()
    # Must NOT have mutated state on a validation failure — that would
    # trigger a spurious loop termination on the next iteration.
    assert state.finish_requested is False
    assert state.finish_summary == ""


@pytest.mark.asyncio
async def test_finish_task_rejects_whitespace_only_summary():
    """Whitespace-only summaries are the same as empty — reject."""
    state = _make_state()
    tool = FinishTaskTool(
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-finish", state=state,
    )
    result = await tool.execute(summary="   \n\t  ")
    assert not result.success
    assert state.finish_requested is False


@pytest.mark.asyncio
async def test_finish_task_sets_state_on_success():
    """A valid call flips ``finish_requested`` and stores the summary
    verbatim (trimmed of surrounding whitespace only)."""
    state = _make_state()
    tool = FinishTaskTool(
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-finish", state=state,
    )
    summary = "Added auth middleware, tests pass, migration deferred to PR 2."
    result = await tool.execute(summary=f"  {summary}  ")

    assert result.success
    assert state.finish_requested is True
    assert state.finish_summary == summary
    assert result.metadata["summary_chars"] == len(summary)


@pytest.mark.asyncio
async def test_finish_task_preserves_internal_whitespace():
    """Only surrounding whitespace is stripped — newlines inside the
    summary are part of the user-facing message."""
    state = _make_state()
    tool = FinishTaskTool(
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-finish", state=state,
    )
    summary = "Line one.\nLine two.\n\nLine four."
    result = await tool.execute(summary=summary)
    assert result.success
    assert state.finish_summary == summary


# ---------------------------------------------------------------------------
# Loop integration — _act_hybrid honours the finish signal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finish_task_terminates_hybrid_loop(monkeypatch):
    """When the model calls finish_task, the next loop iteration
    breaks with termination_reason='finish_task_called' — the model
    is NOT called again after its finish_task call returns."""
    _force_native_tier(monkeypatch)

    # Real FinishTaskTool — we want the actual state mutation path,
    # not a fake. The loop looks up tools by name via create_coder_tools,
    # so inject a FinishTaskTool bound to the handler's state.
    class _LoopHarness:
        state: CoderState | None = None

    def _fake_create_tools(cm, ws, state, **_kw):
        _LoopHarness.state = state
        return [FinishTaskTool(
            container_manager=cm, workspace_id=ws, state=state,
        )]

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        _fake_create_tools,
    )

    class _CallsFinishThenHangs:
        """Iteration 1: call finish_task. Iteration 2: if reached, fail
        loudly — the loop should have terminated before this."""

        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(
                        0, "tc-1", "finish_task",
                        {"summary": "All done — wrote foo.py."},
                    ),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                # This branch means the termination check failed. Emit
                # a diagnostic the test can assert against.
                yield _FakeChunk(
                    content_delta="ERROR: loop did not terminate",
                )
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _CallsFinishThenHangs()
    handler = CoderHandler(
        backend, session_id="sess-finish-loop",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-finish-loop",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("write foo.py"), workspace_context="",
    ):
        chunks.append(c)

    # Meta chunk for finish_task_called must be emitted.
    break_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "finish_task_called"
    ]
    assert break_chunks, (
        "Expected a finish_task_called meta chunk. Statuses seen: "
        + repr([c.augmentum.get("status") if c.augmentum else None
                for c in chunks])
    )

    # Final complete chunk must record the termination reason.
    complete_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "complete"
    ]
    assert complete_chunks, "Expected a terminal 'complete' meta chunk"
    final_reason = complete_chunks[-1].augmentum.get("termination_reason")
    assert final_reason == "finish_task_called", final_reason

    # Summary must surface as streamed content — the user sees the
    # model's own words, not a synthesized reconstruction.
    streamed_text = "".join(
        c.content_delta or "" for c in chunks if c.content_delta
    )
    assert "All done — wrote foo.py." in streamed_text

    # Backend must NOT have been called a second time — the loop
    # terminated before re-prompting.
    assert backend.calls == 1, (
        f"Loop re-prompted after finish_task (calls={backend.calls})"
    )

    # Sanity: the "hang" branch's error marker never leaked through.
    assert "ERROR: loop did not terminate" not in streamed_text


@pytest.mark.asyncio
async def test_native_filters_finish_task_and_ends_with_final_prose(monkeypatch):
    """Native mode should finish like CLI agents do: final prose, no tool."""

    def _fake_create_tools(cm, ws, state, **_kw):
        return [
            FinishTaskTool(container_manager=cm, workspace_id=ws, state=state),
            _FakeTool("file_list"),
        ]

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        _fake_create_tools,
    )

    class _FinalProseBackend:
        def __init__(self):
            self.calls = 0
            self.requests = []

        async def chat_stream(self, request):
            self.calls += 1
            self.requests.append(request)
            # Substantive final prose — under the Termination Quality
            # Gate this needs to be either >=200 chars or >=30 chars with
            # 2+ sentences. Real Qwen-3.6 completion summaries are 200+
            # chars; this is the minimum shape that still verifies
            # "model legitimately stops with a final answer".
            yield _FakeChunk(
                content_delta=(
                    "Native done. Inspected the workspace and listed "
                    "all three Python files in the project root."
                ),
            )
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _FinalProseBackend()
    handler = CoderHandler(
        backend,
        session_id="sess-native-finish",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-native-finish",
    )

    chunks: list[InternalStreamChunk] = []
    req = _make_request("finish briefly")
    req.top_p = 0.77
    req.max_tokens = 1234
    async for c in handler._act_native(req, workspace_context=""):
        chunks.append(c)

    tool_names = [
        schema["function"]["name"]
        for schema in backend.requests[0].tools
    ]
    assert tool_names == ["file_list"]
    streamed_text = "".join(c.content_delta or "" for c in chunks)
    assert "Native done. Inspected the workspace" in streamed_text
    complete_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "complete"
    ]
    # Termination Quality Gate (phase 2026-05-27): native now anchors
    # the termination_reason on the gate's verdict tag rather than the
    # bare ``model_stop``. A substantive prose response under an action
    # intent with zero writes resolves to
    # ``model_stop:substantive_under_active`` — the gate "accepted a
    # real answer even though no writes happened".
    assert (
        complete_chunks[-1].augmentum.get("termination_reason")
        == "model_stop:substantive_under_active"
    )
    assert backend.calls == 1
    assert backend.requests[0].tool_choice is None
    assert backend.requests[0].chat_template_kwargs == {"enable_thinking": False}
    assert backend.requests[0].top_p == 0.77
    assert backend.requests[0].max_tokens == 1234


@pytest.mark.asyncio
async def test_native_first_iteration_leaves_tool_choice_auto(monkeypatch):
    """Native mode should mirror CLI agents: schemas in, model chooses."""
    fake_tool = _FakeTool("file_list")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_tool],
    )

    class _StopsImmediately:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            # Substantive prose so the Termination Quality Gate accepts
            # the stop (≥30 chars + ≥2 sentences). Pre-gate, native
            # accepted any prose verbatim.
            yield _FakeChunk(
                content_delta=(
                    "Visible final answer. No tool calls were needed "
                    "for this question."
                ),
            )
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _StopsImmediately()
    handler = CoderHandler(
        backend,
        session_id="sess-native-file-list",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-native-file-list",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(_make_request("hello"), workspace_context=""):
        chunks.append(c)

    assert backend.requests[0].tool_choice is None
    complete_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "complete"
    ]
    # See note in test_native_filters_finish_task_and_ends_with_final_prose
    # — substantive prose under an action intent with zero writes
    # accepts via the gate's "substantive_under_active" verdict.
    assert (
        complete_chunks[-1].augmentum.get("termination_reason")
        == "model_stop:substantive_under_active"
    )


@pytest.mark.asyncio
async def test_native_injects_bounded_context_prelude(monkeypatch):
    """Native should receive the light Augmentum prelude when supplied."""
    fake_tool = _FakeTool("file_list")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_tool],
    )

    class _StopsImmediately:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            # Substantive prose so the Termination Quality Gate accepts
            # the stop (≥30 chars + ≥2 sentences). Pre-gate, native
            # accepted any prose verbatim.
            yield _FakeChunk(
                content_delta=(
                    "Visible final answer. No tool calls were needed "
                    "for this question."
                ),
            )
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _StopsImmediately()
    handler = CoderHandler(
        backend,
        session_id="sess-native-context",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-native-context",
    )

    async for _ in handler._act_native(
        _make_request("hello"),
        workspace_context="## Native Context Prelude\nFacts",
    ):
        pass

    system_text = backend.requests[0].messages[0].content
    assert "## Native Context Prelude" in system_text
    assert "Facts" in system_text
    assert system_text.index("## Capability map") < system_text.index(
        "## Native Context Prelude",
    )


@pytest.mark.asyncio
async def test_native_retries_generic_empty_stop(monkeypatch):
    """A blank model stop is not a final answer, but the retry is generic."""
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )

    class _BlankThenFinal:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                yield _FakeChunk(content_delta="\n\n")
                yield _FakeChunk(done=True, finish_reason="stop")
                return
            # The retry response must clear the Termination Quality
            # Gate too; pre-2026-05-27 the bare "Visible final answer."
            # (21 chars, 1 sentence) would pass because native accepted
            # any prose, but the gate now classifies that as BAILOUT.
            yield _FakeChunk(
                content_delta=(
                    "Visible final answer. No additional tool calls "
                    "were required for this turn."
                ),
            )
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _BlankThenFinal()
    handler = CoderHandler(
        backend,
        session_id="sess-native-empty-stop",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-native-empty-stop",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(_make_request("answer this"), workspace_context=""):
        chunks.append(c)

    assert len(backend.requests) == 2
    retry_prompt = backend.requests[1].messages[-1].content.lower()
    assert "visible answer" in retry_prompt
    assert "current state" in retry_prompt
    assert "container" not in retry_prompt
    assert " ip" not in retry_prompt
    streamed_text = "".join(c.content_delta or "" for c in chunks)
    assert "Visible final answer." in streamed_text
    assert "tool calls" in streamed_text
    retry_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "empty_model_stop_retry"
    ]
    assert retry_chunks


@pytest.mark.asyncio
async def test_native_accepts_tool_call_then_final_prose(monkeypatch):
    """Native uses the shared hybrid parser: tool call, result, final stop."""
    fake_tool = _FakeTool("file_list", output="file  index.html")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_tool],
    )

    class _ToolThenFinal:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(
                        0, "tc-native-list", "file_list",
                        {"path": "/workspace"},
                    ),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
                return
            assert self.requests[1].messages[-1].role == "tool"
            assert self.requests[1].messages[-1].tool_call_id == "tc-native-list"
            assert "index.html" in self.requests[1].messages[-1].content
            yield _FakeChunk(content_delta="Found index.html in /workspace.")
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _ToolThenFinal()
    handler = CoderHandler(
        backend,
        session_id="sess-native-tool-loop",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-native-tool-loop",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(_make_request("list files"), workspace_context=""):
        chunks.append(c)

    assert len(backend.requests) == 2
    assert backend.requests[0].tool_choice is None
    assert fake_tool.calls == [{"path": "/workspace"}]
    statuses = [
        c.augmentum.get("status")
        for c in chunks
        if c.augmentum
    ]
    assert "tool_call" in statuses
    assert "tool_result" in statuses
    streamed_text = "".join(c.content_delta or "" for c in chunks)
    assert "Found index.html in /workspace." in streamed_text


@pytest.mark.asyncio
async def test_native_parallelizes_read_tools_but_serializes_mutations(monkeypatch):
    """Native should fan out reads, then avoid racing state-changing tools."""

    class _TrackedTool(_FakeTool):
        def __init__(self, name: str, tracker: dict):
            super().__init__(name, output=f"{name} ok")
            self._tracker = tracker

        async def execute(self, **kwargs):
            from augmentum.tools.base import ToolResult
            self.calls.append(dict(kwargs))
            self._tracker["active"] += 1
            self._tracker["max_active"] = max(
                self._tracker["max_active"],
                self._tracker["active"],
            )
            self._tracker["started"].append(self.name)
            await asyncio.sleep(0.01)
            self._tracker["active"] -= 1
            return ToolResult(success=True, output=f"{self.name} ok")

    read_tracker = {"active": 0, "max_active": 0, "started": []}
    shell_tracker = {"active": 0, "max_active": 0, "started": []}
    tools = [
        _TrackedTool("file_read", read_tracker),
        _TrackedTool("file_list", read_tracker),
        _TrackedTool("shell_exec", shell_tracker),
    ]
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: tools,
    )

    class _BatchThenFinal:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "read-1", "file_read", {"path": "/workspace/a.py"}),
                    _tc_delta(1, "read-2", "file_list", {"path": "/workspace"}),
                    _tc_delta(2, "sh-1", "shell_exec", {"command": "echo one"}),
                    _tc_delta(3, "sh-2", "shell_exec", {"command": "echo two"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
                return
            yield _FakeChunk(content_delta="Done.")
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _BatchThenFinal()
    handler = CoderHandler(
        backend,
        session_id="sess-native-schedule",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-native-schedule",
    )

    async for _ in handler._act_native(_make_request("do things"), workspace_context=""):
        pass

    assert read_tracker["max_active"] == 2
    assert shell_tracker["max_active"] == 1
    assert shell_tracker["started"] == ["shell_exec", "shell_exec"]


@pytest.mark.asyncio
async def test_finish_task_flag_cleared_between_requests(monkeypatch):
    """A prior turn's finish_requested must NOT short-circuit a new
    request's first iteration. _reset_for_new_request clears it."""
    _force_native_tier(monkeypatch)
    fake_tool = _FakeTool("file_read")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_tool],
    )

    class _SingleRead:
        async def chat_stream(self, request):
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, "tc-1", "file_read", {"path": "/a.py"}),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _SingleRead(), session_id="sess-reset",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-reset",
    )
    # Simulate the stale state a prior turn would leave behind.
    handler._state.finish_requested = True
    handler._state.finish_summary = "prior-turn summary"

    # _reset_for_new_request is the method the handler calls at the
    # start of each new request; exercise it directly.
    handler._reset_for_new_request()

    assert handler._state.finish_requested is False
    assert handler._state.finish_summary == ""
