"""Subagent return-path verification gate (agents/verify.py + loop wiring).

Pins the behavior this shipment introduced — the leaf-node twin of the
lead goal-judge:

- ``judge_subagent_result`` parses a per-criterion verdict, fails OPEN
  (``ok=None``) on backend/parse failure, and returns ``ok=None`` when
  handed no criteria.
- ``run_subagent`` runs the judge ONLY when ``spec.verify`` is set AND
  criteria are present; a passing verdict marks ``verification="passed"``,
  a failing verdict re-enters the loop (bounded by ``verify_max_reentry``)
  with the unmet criteria injected, and exhaustion marks
  ``verification="failed"`` with a recovery hint — without trapping the
  subagent.
- A judge that gives no signal fails open to ``verification="error"`` and
  the stop is honored.
- ``build_initial_user_message`` appends the ``<criteria_check>`` self-
  report instruction iff criteria were handed down.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from augmentum.agents.budget import SubagentBudget
from augmentum.agents.context_bridge import build_initial_user_message
from augmentum.agents.loop import (
    SubagentSpec,
    _compute_recovery_hint,
    run_subagent,
)
from augmentum.agents.verify import (
    judge_subagent_result,
)
from augmentum.tools.base import Tool, ToolCategory, ToolResult

# ----------------------------------------------------------------------
# judge_subagent_result — verdict parsing + fail-open contract
# ----------------------------------------------------------------------

class _JudgeBackend:
    """Returns a single canned reply (or raises) for the judge call."""

    def __init__(self, reply: str | None = None, raises: bool = False):
        self._reply = reply
        self._raises = raises
        self.calls: list = []

    async def chat(self, request):
        self.calls.append(request)
        if self._raises:
            raise RuntimeError("judge backend down")
        return SimpleNamespace(
            message=SimpleNamespace(content=self._reply, thinking=None),
        )


@pytest.mark.asyncio
async def test_judge_passes_when_all_criteria_met():
    be = _JudgeBackend('{"ok": true, "unmet": [], "reason": "route + test both present"}')
    v = await judge_subagent_result(
        be, model="m", task="add /health with a test",
        success_criteria=("endpoint added", "test passes"),
        output="Added /health and a passing test.",
        tool_summary="file_write app/routes.py → success\ntest_run → success",
    )
    assert v.ok is True
    assert v.label == "passed"
    assert v.unmet == ()
    # Judge call is non-streaming + deterministic + tool-free.
    req = be.calls[0]
    assert req.stream is False and req.tools is None and req.temperature == 0.0


@pytest.mark.asyncio
async def test_judge_fails_and_names_unmet_criteria():
    be = _JudgeBackend(
        '{"ok": false, "unmet": ["test passes"], "reason": "no test was run"}'
    )
    v = await judge_subagent_result(
        be, model="m", task="add a route with a test",
        success_criteria=("endpoint added", "test passes"),
        output="Added the route.",
    )
    assert v.ok is False
    assert v.label == "failed"
    assert "test passes" in v.unmet


@pytest.mark.asyncio
async def test_judge_fails_open_on_backend_error():
    be = _JudgeBackend(raises=True)
    v = await judge_subagent_result(
        be, model="m", task="x", success_criteria=("c",), output="y",
    )
    assert v.ok is None
    assert v.label == "error"


@pytest.mark.asyncio
async def test_judge_fails_open_on_unparseable_reply():
    be = _JudgeBackend("not json at all")
    v = await judge_subagent_result(
        be, model="m", task="x", success_criteria=("c",), output="y",
    )
    assert v.ok is None


@pytest.mark.asyncio
async def test_judge_tolerates_fenced_json():
    be = _JudgeBackend('```json\n{"ok": true, "unmet": [], "reason": "ok"}\n```')
    v = await judge_subagent_result(
        be, model="m", task="x", success_criteria=("c",), output="y",
    )
    assert v.ok is True


@pytest.mark.asyncio
async def test_judge_no_criteria_returns_no_signal():
    be = _JudgeBackend('{"ok": false}')
    v = await judge_subagent_result(
        be, model="m", task="x", success_criteria=(), output="y",
    )
    assert v.ok is None
    # Defensive: never even calls the backend with no criteria.
    assert be.calls == []


# ----------------------------------------------------------------------
# recovery hint — failed verification on a clean complete
# ----------------------------------------------------------------------

def test_recovery_hint_names_unmet_on_failed_verification():
    hint = _compute_recovery_hint(
        stop_reason="complete", stop_detail="", stuck_pattern=None,
        role="fixer", iterations=4, instance_id="sa_v",
        verification="failed",
        verification_unmet=["test passes", "no regressions"],
        verification_reason="tests were never run",
    )
    assert "fixer" in hint
    assert "test passes" in hint
    assert "tests were never run" in hint
    assert "do not trust" in hint.lower()


def test_recovery_hint_empty_on_passed_verification():
    hint = _compute_recovery_hint(
        stop_reason="complete", stop_detail="", stuck_pattern=None,
        role="fixer", iterations=4, instance_id="sa_v",
        verification="passed",
    )
    assert hint == ""


# ----------------------------------------------------------------------
# run_subagent — the gate end to end
# ----------------------------------------------------------------------

class _StubTool(Tool):
    def __init__(self, name: str, output: str = "ok") -> None:
        self._name = name
        self._output = output

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"stub {self._name}"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ_ONLY

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=self._output)


@dataclass
class _Usage:
    prompt_tokens: int = 10
    completion_tokens: int = 5


@dataclass
class _Msg:
    content: str = ""
    tool_calls: list | None = None
    role: str = "assistant"


@dataclass
class _Resp:
    message: _Msg
    usage: _Usage


class _SeqBackend:
    """Pops prepared responses in order; raises if over-called. The SAME
    backend serves both the model loop and the verification judge, so the
    response sequence interleaves model turns and judge verdicts."""

    def __init__(self, responses: list[_Resp]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, request):
        self.calls += 1
        if not self._responses:
            raise RuntimeError("backend over-called")
        return self._responses.pop(0)


def _prose(text: str) -> _Resp:
    return _Resp(message=_Msg(content=text, tool_calls=None), usage=_Usage())


def _verdict(json_text: str) -> _Resp:
    return _Resp(message=_Msg(content=json_text, tool_calls=None), usage=_Usage())


def _spec(backend_role="fixer", *, verify, criteria, reentry=1) -> SubagentSpec:
    return SubagentSpec(
        role=backend_role,
        model="stub",
        system_prompt="",
        initial_user_message="do the thing",
        tools=(_StubTool("file_read"),),
        budget=SubagentBudget(max_iterations=6, max_wallclock_seconds=10, max_tokens=10_000),
        verify=verify,
        task_prompt="do the thing",
        success_criteria=criteria,
        verify_max_reentry=reentry,
        instance_id="sa_verify_test",
    )


@pytest.mark.asyncio
async def test_no_verification_when_disabled():
    backend = _SeqBackend([_prose("all done")])
    spec = _spec(verify=False, criteria=("c",))
    result = await run_subagent(spec, backend=backend)
    assert result.stop_reason == "complete"
    assert result.verification == "unchecked"
    # No judge call — exactly one model turn consumed.
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_no_verification_when_no_criteria():
    backend = _SeqBackend([_prose("all done")])
    spec = _spec(verify=True, criteria=())
    result = await run_subagent(spec, backend=backend)
    assert result.verification == "unchecked"
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_verification_passes():
    backend = _SeqBackend([
        _prose("done — wrote the route and ran the test"),
        _verdict('{"ok": true, "unmet": [], "reason": "evidence present"}'),
    ])
    spec = _spec(verify=True, criteria=("route added", "test passes"))
    result = await run_subagent(spec, backend=backend)
    assert result.stop_reason == "complete"
    assert result.verification == "passed"
    assert result.recovery_hint == ""
    assert backend.calls == 2  # model + judge


@pytest.mark.asyncio
async def test_verification_failed_reenters_then_passes():
    backend = _SeqBackend([
        _prose("done (attempt 1)"),
        _verdict('{"ok": false, "unmet": ["test passes"], "reason": "no test run"}'),
        _prose("now I ran the test (attempt 2)"),
        _verdict('{"ok": true, "unmet": [], "reason": "test now passes"}'),
    ])
    spec = _spec(verify=True, criteria=("route added", "test passes"), reentry=1)
    result = await run_subagent(spec, backend=backend)
    assert result.verification == "passed"
    # model1 + judge1 + model2 + judge2
    assert backend.calls == 4
    assert result.output == "now I ran the test (attempt 2)"


@pytest.mark.asyncio
async def test_verification_failed_exhausted_marks_failed():
    backend = _SeqBackend([
        _prose("done but incomplete"),
        _verdict('{"ok": false, "unmet": ["test passes"], "reason": "no test run"}'),
    ])
    spec = _spec(verify=True, criteria=("test passes",), reentry=0)
    result = await run_subagent(spec, backend=backend)
    assert result.stop_reason == "complete"
    assert result.verification == "failed"
    assert result.verification_reason == "no test run"
    assert "test passes" in result.verification_unmet
    # Recovery hint must warn the lead not to trust the report.
    assert "do not trust" in result.recovery_hint.lower()
    assert backend.calls == 2


@pytest.mark.asyncio
async def test_verification_fails_open_on_judge_error():
    backend = _SeqBackend([
        _prose("done"),
        _verdict("garbage not-json"),  # judge can't parse → ok=None
    ])
    spec = _spec(verify=True, criteria=("c",), reentry=1)
    result = await run_subagent(spec, backend=backend)
    # Fail open: stop honored, marked error (not failed, not passed).
    assert result.stop_reason == "complete"
    assert result.verification == "error"
    assert result.recovery_hint == ""
    assert backend.calls == 2


# ----------------------------------------------------------------------
# context bridge — the <criteria_check> self-report instruction
# ----------------------------------------------------------------------

def test_criteria_check_instruction_present_with_criteria():
    msg = build_initial_user_message(
        prompt="add a route",
        context_mode="slim",
        success_criteria=("route added", "test passes"),
    )
    assert "<criteria_check>" in msg
    assert "<success_criteria>" in msg


def test_criteria_check_instruction_absent_without_criteria():
    msg = build_initial_user_message(
        prompt="add a route",
        context_mode="slim",
    )
    assert "<criteria_check>" not in msg


@pytest.mark.asyncio
async def test_dispatch_footer_labels_unverified_on_judge_error():
    """verification='error' must reach the lead as an explicit UNVERIFIED
    note, not just a cryptic `verify:error` token — a silent fail-open
    reading like a checked pass is the trust bug (uncertainty-map tier 2)."""
    from types import SimpleNamespace

    from augmentum.coder.state import CoderState
    from augmentum.coder.tools import TaskDispatchTool

    result = SimpleNamespace(
        output="I did the thing.",
        stop_reason="complete",
        stop_detail="",
        stuck_pattern="",
        iterations=3,
        tool_calls=5,
        tokens_in=100,
        tokens_out=50,
        wallclock_ms=1200,
        recovery_hint="",
        verification="error",
        verification_reason="",
        verification_unmet=[],
    )
    outcome = SimpleNamespace(
        result=result,
        subagent_id="sub1",
        role="explore",
        model_spec="m",
        model_resolved="m",
    )

    class _Dispatcher:
        async def dispatch(self, req):
            return outcome

    tool = TaskDispatchTool(
        container_manager=None,
        workspace_id="ws",
        state=CoderState(session_id="s", workspace_id="ws"),
        dispatcher=_Dispatcher(),
    )
    tr = await tool.execute(role="explore", prompt="do it")
    # Fail-open: still success (the stop is honored)…
    assert tr.success is True
    # …but the lead-facing output says loudly that nothing was verified.
    assert "[unverified]" in tr.output
    assert "could not run" in tr.output
    assert tr.metadata["verification"] == "error"


@pytest.mark.asyncio
async def test_dispatch_footer_no_unverified_note_on_pass():
    from types import SimpleNamespace

    from augmentum.coder.state import CoderState
    from augmentum.coder.tools import TaskDispatchTool

    result = SimpleNamespace(
        output="Done.",
        stop_reason="complete",
        stop_detail="",
        stuck_pattern="",
        iterations=1,
        tool_calls=1,
        tokens_in=10,
        tokens_out=5,
        wallclock_ms=100,
        recovery_hint="",
        verification="passed",
        verification_reason="",
        verification_unmet=[],
    )
    outcome = SimpleNamespace(
        result=result, subagent_id="s1", role="explore",
        model_spec="m", model_resolved="m",
    )

    class _Dispatcher:
        async def dispatch(self, req):
            return outcome

    tool = TaskDispatchTool(
        container_manager=None,
        workspace_id="ws",
        state=CoderState(session_id="s", workspace_id="ws"),
        dispatcher=_Dispatcher(),
    )
    tr = await tool.execute(role="explore", prompt="do it")
    assert tr.success is True
    assert "[unverified]" not in tr.output
    assert "verify:passed" in tr.output
