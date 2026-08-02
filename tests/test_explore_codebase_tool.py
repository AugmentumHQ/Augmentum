"""explore_codebase — the concrete-surface alias that makes local models delegate.

Pins the 2026-06-19 build: local/open models reliably don't call the abstract
``task_dispatch`` meta-tool (delegation is an RL-trained behaviour they lack), but
DO call an in-distribution ``explore_codebase(query)`` verb — proven on
Qwen3-Coder-30B (SELF -> DELEGATED when the same explore subagent was reframed).
``ExploreCodebaseTool`` is that reframe: it supplies the meta-cognition the model
skipped (role=explore + framed prompt + derived success_criteria) and delegates to
the inherited ``TaskDispatchTool`` dispatch path, so persistence / streaming /
verification are unchanged.

The load-bearing assertions: (1) a one-arg ``query`` becomes a real explore
dispatch carrying non-empty success_criteria; (2) it's offered alongside
``task_dispatch`` whenever a dispatcher is wired.
"""

from __future__ import annotations

import pytest

from augmentum.agents.dispatch import DispatchOutcome
from augmentum.agents.loop import SubagentResult
from augmentum.coder.tools import (
    ExploreCodebaseTool,
    TaskDispatchTool,
    create_coder_tools,
)


class _FakeDispatcher:
    """Captures the DispatchRequest and returns a canned successful outcome."""

    def __init__(self) -> None:
        self.last_req = None

    async def dispatch(self, req):
        self.last_req = req
        result = SubagentResult(
            role=req.role,
            instance_id="sa_test_1",
            output="Found 3 call sites: a.py:10, b.py:22, c.py:30.",
            tokens_in=500,
            tokens_out=120,
            wallclock_ms=2000,
            iterations=4,
            tool_calls=6,
            stop_reason="complete",
            verification="passed",
        )
        return DispatchOutcome(
            subagent_id="sa_test_1",
            role=req.role,
            model_spec="",
            model_resolved="local-model",
            result=result,
        )


def _make_tool() -> tuple[ExploreCodebaseTool, _FakeDispatcher]:
    disp = _FakeDispatcher()
    tool = ExploreCodebaseTool(
        container_manager=None,
        workspace_id="ws-1",
        state=None,
        dispatcher=disp,
        user_id="usr_1",
    )
    return tool, disp


def test_name_and_schema_are_concrete():
    tool, _ = _make_tool()
    assert tool.name == "explore_codebase"
    schema = tool.input_schema
    assert schema["required"] == ["query"]
    assert set(schema["properties"]) == {"query"}  # one obvious arg, no role/criteria
    # The surface must NOT mention the abstract machinery that scares models off.
    assert "subagent" not in tool.description.lower()
    assert "role" not in tool.description.lower()


@pytest.mark.asyncio
async def test_query_becomes_an_explore_dispatch_with_criteria():
    tool, disp = _make_tool()
    res = await tool.execute(query="every call site of resolve_backend_for_model")
    assert res.success
    req = disp.last_req
    # The model passed ONE arg; the tool supplied the meta-cognition.
    assert req.role == "explore"
    assert "resolve_backend_for_model" in req.prompt
    assert len(req.success_criteria) >= 2          # derived, non-empty
    assert any("file:line" in c for c in req.success_criteria)
    assert req.context_mode_override == "workspace"
    assert req.user_id == "usr_1"
    # The subagent's findings come back as the tool output.
    assert "a.py:10" in res.output


@pytest.mark.asyncio
async def test_tolerates_prompt_alias():
    # Some models fill `prompt` instead of `query`; don't punish that.
    tool, disp = _make_tool()
    res = await tool.execute(prompt="where session tokens are validated")
    assert res.success
    assert "session tokens" in disp.last_req.prompt


@pytest.mark.asyncio
async def test_empty_query_is_validation_error_not_dispatch():
    tool, disp = _make_tool()
    res = await tool.execute(query="   ")
    assert not res.success
    assert res.validation_error
    assert disp.last_req is None                    # never dispatched


def test_offered_alongside_task_dispatch_when_dispatcher_wired():
    disp = _FakeDispatcher()
    tools = create_coder_tools(None, "ws-1", None, subagent_dispatcher=disp)
    names = {t.name for t in tools}
    assert "explore_codebase" in names
    assert "task_dispatch" in names                 # additive, not a replacement
    # It must be the right class (inherits the dispatch path).
    explore = next(t for t in tools if t.name == "explore_codebase")
    assert isinstance(explore, TaskDispatchTool)


def test_absent_when_no_dispatcher():
    tools = create_coder_tools(None, "ws-1", None)  # subagent_dispatcher=None
    names = {t.name for t in tools}
    assert "explore_codebase" not in names
    assert "task_dispatch" not in names
