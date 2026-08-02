"""Consent gate on model-invoked actions — Slice 1 of the FC-loop hardening.

The architect router tier-gates high-stakes intent on the voice path, but a tool
the MODEL calls directly hits ``ActionTool.execute`` with no such check. These
tests pin the fail-safe: irreversible actions refuse to auto-fire (the handler is
never called), while recoverable ones execute normally.
"""
from __future__ import annotations

import pytest

from augmentum.intent.action import Action, ActionResult, SessionContext
from augmentum.intent.tool_adapter import ActionTool


def _make_action(stakes: str, *, fired: list[str]) -> Action:
    async def _handler(_text: str, _session: SessionContext, _args: dict) -> ActionResult:
        fired.append("called")
        return ActionResult(short_circuit=True, speak="done")

    return Action(
        id=f"test.{stakes}",
        summary="test verb",
        examples=[],
        handler=_handler,
        stakes=stakes,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stakes", ["irrevocable", "safety_critical"])
async def test_high_stakes_actions_are_gated(stakes):
    fired: list[str] = []
    tool = ActionTool(_make_action(stakes, fired=fired), app_state=None)
    result = await tool.execute(_user_id="u1", _session_id="s1")

    assert result.success is False
    assert fired == []  # handler must NOT have run
    assert "confirm" in (result.output or "").lower()
    assert stakes in (result.output or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("stakes", ["trivial_reversible", "disruptive", "costly"])
async def test_recoverable_actions_execute_normally(stakes):
    fired: list[str] = []
    tool = ActionTool(_make_action(stakes, fired=fired), app_state=None)
    result = await tool.execute(_user_id="u1", _session_id="s1")

    assert result.success is True
    assert fired == ["called"]
    assert "done" in (result.output or "")
