"""SSOS → standard executor unification (2026-07-02).

Every SSOS tool execution (heuristic Gen-1 + marker Gen-2) routes
through the handler's ``_execute_tool`` when bound, so param coercion,
user/session context, metrics, cards, and tool-card presentation match
the native loop exactly — one executor, one visual language.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.modes.passthrough.orchestrator import (
    _CAPABILITIES_BY_NAME,
    SSOSOrchestrator,
)


def _orch(registry=None):
    return SSOSOrchestrator(
        registry or MagicMock(), user_id="u1", app_state=MagicMock(),
    )


@pytest.mark.asyncio
async def test_run_named_tool_routes_through_bound_executor():
    registry = MagicMock()
    fake_tool = MagicMock()
    fake_tool.name = "web_search"
    registry.get.return_value = fake_tool

    orch = _orch(registry)
    executor = AsyncMock(return_value=("some results", {"results": [{"url": "x"}]}))
    orch.bind_executor(executor)

    cap = _CAPABILITIES_BY_NAME["web_search"]
    out, meta = await orch.run_named_tool(cap, "rust async runtimes")

    executor.assert_awaited_once()
    called_tool, called_params = executor.await_args.args
    assert called_tool is fake_tool
    assert called_params["query"] == "rust async runtimes"
    assert called_params["num_results"] == 5
    assert out == "some results"
    assert meta == {"results": [{"url": "x"}]}
    # The tool itself was NOT called directly — the executor owns it.
    fake_tool.execute.assert_not_called()


@pytest.mark.asyncio
async def test_executor_error_shape_is_failure():
    """_execute_tool signals failure as an 'Error: …' string — the shim
    must read that as unsuccessful so SSOS synthesizes the graceful
    fallback instead of citing an error message as results."""
    registry = MagicMock()
    fake_tool = MagicMock()
    fake_tool.name = "web_search"
    registry.get.return_value = fake_tool

    orch = _orch(registry)
    orch.bind_executor(AsyncMock(return_value=("Error: backend down", {})))

    cap = _CAPABILITIES_BY_NAME["web_search"]
    out, meta = await orch.run_named_tool(cap, "anything")
    assert out is None and meta == {}


@pytest.mark.asyncio
async def test_unbound_orchestrator_falls_back_to_direct_execute():
    """Standalone construction (no handler) still works — legacy direct
    execution with minimal context injection."""
    fake_result = MagicMock(success=True, output="direct", metadata={}, error="")

    async def _direct(**kwargs):
        return fake_result

    fake_tool = MagicMock()
    fake_tool.name = "calculator"
    fake_tool.timeout = 5.0
    fake_tool.execute = _direct

    orch = _orch()
    r = await orch._run_tool(fake_tool, {"expression": "1+1"})
    assert r.success and r.output == "direct"


@pytest.mark.asyncio
async def test_run_tool_never_raises():
    fake_tool = MagicMock()
    fake_tool.name = "web_search"
    orch = _orch()
    boom = AsyncMock(side_effect=RuntimeError("kaboom"))
    orch.bind_executor(boom)
    r = await orch._run_tool(fake_tool, {"query": "x"})
    assert not r.success and "kaboom" in r.error
    assert await orch._run_tool(None, {}) is not None
