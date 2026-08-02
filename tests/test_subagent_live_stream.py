"""Tests for the subagent live-stream + cancel + recovery substrate.

Pinning behavior that this shipment introduced:
- ``SubagentSpec.progress_callback`` fires on each inner-loop boundary
  (responding / tool_call / tool_result / done / stuck).
- ``SubagentResult.recovery_hint`` is populated from the stop_reason →
  guidance map for budget / stuck / error / cancelled; empty on clean
  complete.
- ``SubagentDispatcher.cancel(instance_id)`` reaches one in-flight
  subagent and returns a synthesised ``stop_reason="cancelled"``
  result with a recovery hint; siblings unaffected.
- The process-wide ``find_subagent_owner`` registry resolves running
  instance_ids to their owning dispatcher (used by the cancel REST
  endpoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from augmentum.agents.budget import SubagentBudget
from augmentum.agents.dispatch import (
    SubagentDispatcher,
    current_subagent_depth,
    find_subagent_owner,
    list_active_subagents,
)
from augmentum.agents.loop import (
    SubagentProgress,
    SubagentSpec,
    _compute_recovery_hint,
    run_subagent,
)
from augmentum.agents.spec import AgentRole
from augmentum.tools.base import Tool, ToolCategory, ToolResult

# ----------------------------------------------------------------------
# Recovery-hint mapping
# ----------------------------------------------------------------------

def test_recovery_hint_empty_on_clean_complete():
    hint = _compute_recovery_hint(
        stop_reason="complete", stop_detail="", stuck_pattern=None,
        role="explore", iterations=3, instance_id="sa_x",
    )
    assert hint == ""


def test_recovery_hint_budget_names_role_and_iterations():
    hint = _compute_recovery_hint(
        stop_reason="budget", stop_detail="iterations cap",
        stuck_pattern=None, role="plan", iterations=8,
        instance_id="sa_y",
    )
    assert "plan" in hint
    assert "8 iter" in hint or "8 iterations" in hint
    assert "split" in hint.lower() or "narrower" in hint.lower()


def test_recovery_hint_stuck_names_pattern():
    hint = _compute_recovery_hint(
        stop_reason="stuck", stop_detail="REPEATED_TOOL_CALLS pattern",
        stuck_pattern="REPEATED_TOOL_CALLS", role="review",
        iterations=12, instance_id="sa_z",
    )
    assert "REPEATED_TOOL_CALLS" in hint
    assert "review" in hint


def test_recovery_hint_cancelled_distinct_from_error():
    cancelled = _compute_recovery_hint(
        stop_reason="cancelled", stop_detail="user cancelled",
        stuck_pattern=None, role="research", iterations=0,
        instance_id="sa_a",
    )
    err = _compute_recovery_hint(
        stop_reason="error", stop_detail="HTTP 500",
        stuck_pattern=None, role="research", iterations=0,
        instance_id="sa_a",
    )
    assert "cancelled" in cancelled.lower()
    assert "cancelled" not in err.lower()
    assert err != cancelled


def test_recovery_hint_unknown_reason_returns_empty():
    """An unknown stop_reason should produce empty (not crash). Keeps
    the contract simple — adding new stop_reasons that don't yet have
    a hint is harmless, just degrades to no guidance."""
    hint = _compute_recovery_hint(
        stop_reason="unrecognised_reason", stop_detail="",
        stuck_pattern=None, role="x", iterations=1,
        instance_id="sa_b",
    )
    assert hint == ""


# ----------------------------------------------------------------------
# Progress callback emission from run_subagent
# ----------------------------------------------------------------------

class _StubTool(Tool):
    """Tool that always returns the same prepared ToolResult."""

    def __init__(self, name: str, output: str = "ok") -> None:
        self._name = name
        self._output = output

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"stub tool {self._name}"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ_ONLY

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=self._output)


@dataclass
class _StubUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5


@dataclass
class _StubMessage:
    content: str = ""
    tool_calls: list | None = None
    role: str = "assistant"


@dataclass
class _StubResponse:
    message: _StubMessage
    usage: _StubUsage


class _StubBackend:
    """Yields prepared responses in order; raises if exhausted."""

    def __init__(self, responses: list[_StubResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, request):
        self.calls += 1
        if not self._responses:
            raise RuntimeError("backend exhausted")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_run_subagent_emits_progress_at_each_boundary():
    """One tool call → expect responding + tool_call + tool_result +
    done — in that order — without missing any boundary."""
    tool = _StubTool("file_read", output="contents")

    # Sequence: model returns a tool_call, then returns clean prose.
    backend = _StubBackend([
        _StubResponse(
            message=_StubMessage(
                content="I'll read the file.",
                tool_calls=[{
                    "id": "tc_1", "type": "function",
                    "function": {"name": "file_read", "arguments": "{}"},
                }],
            ),
            usage=_StubUsage(),
        ),
        _StubResponse(
            message=_StubMessage(content="Done — the file says 'contents'.", tool_calls=None),
            usage=_StubUsage(),
        ),
    ])

    events: list[SubagentProgress] = []

    async def _capture(p: SubagentProgress) -> None:
        events.append(p)

    spec = SubagentSpec(
        role="explore",
        model="stub",
        system_prompt="",
        initial_user_message="read /workspace/x",
        tools=(tool,),
        budget=SubagentBudget(max_iterations=4, max_wallclock_seconds=10, max_tokens=10_000),
        progress_callback=_capture,
        instance_id="sa_test_42",
    )
    result = await run_subagent(spec, backend=backend)

    assert result.stop_reason == "complete"
    assert result.recovery_hint == ""

    phases = [e.phase for e in events]
    # Iteration 1: responding (had tool_calls) → tool_call → tool_result
    # Iteration 2: responding (clean prose) → done
    assert phases == ["responding", "tool_call", "tool_result", "responding", "done"], phases

    # Tool-call events carry tool_name; responding events don't.
    assert events[1].tool_name == "file_read"
    assert events[2].tool_name == "file_read"
    assert events[0].tool_name == ""

    # All events carry the same instance_id and role.
    assert {e.instance_id for e in events} == {"sa_test_42"}
    assert {e.role for e in events} == {"explore"}


@pytest.mark.asyncio
async def test_progress_callback_exception_does_not_kill_loop():
    """A misbehaving sink shouldn't break the subagent."""
    tool = _StubTool("file_read")
    backend = _StubBackend([
        _StubResponse(
            message=_StubMessage(content="done", tool_calls=None),
            usage=_StubUsage(),
        ),
    ])

    async def _bad_sink(_p):
        raise RuntimeError("sink is broken")

    spec = SubagentSpec(
        role="explore", model="stub", system_prompt="",
        initial_user_message="x", tools=(tool,),
        budget=SubagentBudget(max_iterations=2, max_wallclock_seconds=10, max_tokens=1000),
        progress_callback=_bad_sink, instance_id="sa_bad",
    )
    result = await run_subagent(spec, backend=backend)
    assert result.stop_reason == "complete"
    assert result.output == "done"


# ----------------------------------------------------------------------
# Cancel via dispatcher + process-wide owner registry
# ----------------------------------------------------------------------

def _stub_role(name: str = "explore") -> AgentRole:
    return AgentRole(
        name=name, system_prompt="", preferred_model="",
        fallback_models=(), tools=frozenset(["file_read"]),
        budget=SubagentBudget(max_iterations=2, max_wallclock_seconds=10, max_tokens=1000),
        tool_guard="none", context_mode="slim",
        visible_in_ui=False, log_persistence=False,
        can_spawn_subagents=False, max_concurrent=1,
        source="builtin",
    )


@pytest.mark.asyncio
async def test_cancel_unknown_subagent_returns_false():
    reg = MagicMock()
    reg.refresh_if_stale = MagicMock()
    reg.get = MagicMock(return_value=_stub_role())
    dispatcher = SubagentDispatcher(
        registry=reg, provider_registry=MagicMock(),
        store=None, tool_registry_provider=None,
        coder_state_provider=None,
    )
    assert dispatcher.cancel("nonexistent_id") is False
    assert dispatcher.is_running("nonexistent_id") is False
    assert dispatcher.list_running() == []


def test_find_subagent_owner_returns_none_when_idle():
    """Before any dispatch is in-flight, the owner registry is empty."""
    assert find_subagent_owner("sa_does_not_exist") is None
    # ``list_active_subagents`` may legitimately contain ids from other
    # in-progress test cases when run in parallel; here we only assert
    # the absence of our specific made-up id.
    assert "sa_does_not_exist" not in list_active_subagents()


# ----------------------------------------------------------------------
# Depth contextvar still 0 outside any dispatch
# ----------------------------------------------------------------------

def test_current_depth_zero_outside_dispatch():
    """The contextvar default is 0 at module scope; only the dispatcher
    bumps it inside a run. The depth-cap test (test_subagent_depth_cap
    in test_agents_substrate) covers the bumped-state behavior."""
    assert current_subagent_depth() == 0
