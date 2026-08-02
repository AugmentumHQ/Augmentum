"""Goal judge (MiMo-Code borrow) — verdict parsing + fail-open contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.coder.goal_judge import (
    MAX_JUDGE_REENTRY,
    judge_goal_satisfied,
)
from augmentum.models.base import InternalChatRequest, Message


def _request():
    return InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="add a healthcheck route")],
    )


class FakeBackend:
    def __init__(self, reply: str | None = None, raises: bool = False):
        self._reply = reply
        self._raises = raises
        self.calls: list = []

    async def chat(self, request):
        self.calls.append(request)
        if self._raises:
            raise RuntimeError("backend down")
        return SimpleNamespace(
            message=SimpleNamespace(content=self._reply, thinking=None),
        )


@pytest.mark.asyncio
async def test_satisfied_verdict_parses():
    be = FakeBackend('{"ok": true, "reason": "route added and test passes"}')
    v = await judge_goal_satisfied(
        be, source_request=_request(),
        user_goal="add a healthcheck route",
        final_response="Added /health and a test; both pass.",
        edited_paths=["app/routes.py"], total_writes=2,
    )
    assert v.ok is True
    assert "test passes" in v.reason
    # Judge call is non-streaming, tool-free, deterministic.
    req = be.calls[0]
    assert req.stream is False and req.tools is None
    assert req.temperature == 0.0


@pytest.mark.asyncio
async def test_unsatisfied_verdict_parses():
    be = FakeBackend('{"ok": false, "reason": "no test was written"}')
    v = await judge_goal_satisfied(
        be, source_request=_request(),
        user_goal="add a route with a test",
        final_response="Added the route.",
        total_writes=1,
    )
    assert v.ok is False and not v.impossible
    assert "no test" in v.reason


@pytest.mark.asyncio
async def test_impossible_escape_hatch():
    be = FakeBackend(
        '{"ok": false, "impossible": true, "reason": "depends on a service '
        'that does not exist"}'
    )
    v = await judge_goal_satisfied(
        be, source_request=_request(),
        user_goal="x", final_response="y", total_writes=1,
    )
    assert v.ok is False and v.impossible is True


@pytest.mark.asyncio
async def test_fenced_json_and_wrapping_prose_tolerated():
    be = FakeBackend(
        'Here is my analysis:\n```json\n{"ok": true, "reason": "done"}\n```'
    )
    v = await judge_goal_satisfied(
        be, source_request=_request(),
        user_goal="x", final_response="y", total_writes=1,
    )
    assert v.ok is True


@pytest.mark.asyncio
async def test_fail_open_on_backend_error():
    be = FakeBackend(raises=True)
    v = await judge_goal_satisfied(
        be, source_request=_request(),
        user_goal="x", final_response="y", total_writes=1,
    )
    assert v.ok is None  # no signal — caller honors the stop


@pytest.mark.asyncio
async def test_fail_open_on_garbage_output():
    be = FakeBackend("I think it went pretty well overall!")
    v = await judge_goal_satisfied(
        be, source_request=_request(),
        user_goal="x", final_response="y", total_writes=1,
    )
    assert v.ok is None


@pytest.mark.asyncio
async def test_empty_goal_skips_call():
    be = FakeBackend('{"ok": true, "reason": "?"}')
    v = await judge_goal_satisfied(
        be, source_request=_request(),
        user_goal="   ", final_response="y", total_writes=1,
    )
    assert v.ok is None
    assert be.calls == []  # never spent the round-trip


def test_reentry_cap_is_small():
    # Coder iterations are expensive; the cap must stay conservative
    # (MiMo uses 12 for goal-mode sessions — ours is per ordinary turn).
    assert MAX_JUDGE_REENTRY <= 3


def test_setting_registered():
    from augmentum.config import settings
    assert hasattr(settings, "coder_goal_judge_enabled")
    from augmentum.proxy.config_routes import _TOOL_SETTINGS
    assert "coder_goal_judge_enabled" in _TOOL_SETTINGS


def test_phase_act_imports_with_judge_wiring():
    # The judge gate sits inside the native loop — an import proves
    # the surgery didn't break the module.
    import augmentum.modes.coder.phase_act  # noqa: F401
