"""System-driven explore dispatch — the subagent-router Power runs it for the model.

Pins the 2026-06-20 build: local models won't call the delegation tool even when
offered (validated live — Qwen3-Coder used code_grep/file_list, never delegated).
So when the ``subagent-router`` Power is the active controller pick on an
explore-shaped ask at ``pre_plan``, ``CoderHandler._maybe_auto_dispatch_explore``
dispatches ``explore_codebase`` ITSELF and injects the findings into the plan
context — no model cooperation required.

Tests the gating (only fires for the right Power + explore-shaped text + setting
on, at most once) and that a fire actually dispatches and sets the context block.
"""

from __future__ import annotations

import pytest

import augmentum.config as config_mod
from augmentum.agents.dispatch import DispatchOutcome
from augmentum.agents.loop import SubagentResult
from augmentum.modes.coder.handler import (
    CoderHandler,
    _text_is_explore_shaped,
)


class _FakeDispatcher:
    def __init__(self) -> None:
        self.calls = 0
        self.last_req = None

    async def dispatch(self, req):
        self.calls += 1
        self.last_req = req
        result = SubagentResult(
            role=req.role, instance_id="sa_x", output="a.py:1 defines Command; b.py:9 uses it.",
            tokens_in=200, tokens_out=50, wallclock_ms=900, iterations=3,
            tool_calls=4, stop_reason="complete", verification="passed",
        )
        return DispatchOutcome(subagent_id="sa_x", role=req.role, model_spec="",
                               model_resolved="local", result=result)


class _Stub:
    """Minimal stand-in carrying just what _maybe_auto_dispatch_explore touches."""

    def __init__(self, dispatcher, *, power_id="subagent-router", fired=False):
        self._auto_explore_fired = fired
        self._auto_explore_context_block = ""
        self._controller_power_summary = {"id": power_id} if power_id else None
        self._dispatcher = dispatcher
        self._container_manager = None
        self._workspace_id = "ws-1"
        self._state = None
        self._profile_store = None
        self._service_store = None
        self._user_id = "usr_1"
        self.chunks: list[dict] = []

    def _get_subagent_dispatcher(self):
        return self._dispatcher

    def _meta_chunk(self, **kw):
        self.chunks.append(kw)
        return kw


async def _drain(stub, text):
    gen = CoderHandler._maybe_auto_dispatch_explore(stub, latest_user_text=text, model="m")
    out = [ev async for ev in gen]
    return out


# ---------------------------------------------------------------- predicate


def test_explore_shaped_predicate():
    assert _text_is_explore_shaped("find every call site of resolve_backend")
    assert _text_is_explore_shaped("Where is the session token validated?")
    assert _text_is_explore_shaped("trace how click.command becomes a command")
    assert _text_is_explore_shaped("help me understand how the router works")
    # Not explore-shaped: a narrow single-file edit / unrelated ask.
    assert not _text_is_explore_shaped("add a docstring to foo() in bar.py")
    assert not _text_is_explore_shaped("bump the version number")


# ---------------------------------------------------------------- dispatch + gates


@pytest.mark.asyncio
async def test_fires_for_subagent_router_on_explore_text():
    disp = _FakeDispatcher()
    stub = _Stub(disp)
    await _drain(stub, "find every place the Command class is used")
    assert disp.calls == 1
    assert disp.last_req.role == "explore"
    # Findings injected into the plan context block.
    assert "a.py:1" in stub._auto_explore_context_block
    assert "<auto_exploration" in stub._auto_explore_context_block
    statuses = [c.get("status") for c in stub.chunks]
    assert "auto_explore_started" in statuses
    assert "auto_explore_done" in statuses
    assert stub._auto_explore_fired is True


@pytest.mark.asyncio
async def test_skips_when_power_is_not_subagent_router():
    disp = _FakeDispatcher()
    stub = _Stub(disp, power_id="failure-triage")
    await _drain(stub, "find every place the Command class is used")
    assert disp.calls == 0
    assert stub._auto_explore_context_block == ""


@pytest.mark.asyncio
async def test_skips_when_text_not_explore_shaped():
    disp = _FakeDispatcher()
    stub = _Stub(disp)
    await _drain(stub, "rename the variable foo to bar in one file")
    assert disp.calls == 0


@pytest.mark.asyncio
async def test_fires_at_most_once_per_turn():
    disp = _FakeDispatcher()
    stub = _Stub(disp, fired=True)  # already fired this turn
    await _drain(stub, "find every caller of X")
    assert disp.calls == 0


@pytest.mark.asyncio
async def test_skips_when_setting_disabled(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "coder_subagent_auto_explore", False, raising=False)
    disp = _FakeDispatcher()
    stub = _Stub(disp)
    await _drain(stub, "find every caller of X")
    assert disp.calls == 0


@pytest.mark.asyncio
async def test_skips_when_no_dispatcher():
    stub = _Stub(None)  # coder_subagents_enabled off → dispatcher None
    await _drain(stub, "find every caller of X")
    assert stub._auto_explore_context_block == ""
