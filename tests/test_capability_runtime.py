"""Direct-mode model_invoke — bypasses the chat pipeline (the live-run lesson)."""

from __future__ import annotations

import pytest

from augmentum.selfedit.capabilities.runtime import build_direct_model_invoke


class _Resp:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _Backend:
    def __init__(self, captured):
        self._captured = captured

    async def chat(self, req):
        self._captured["req"] = req
        return _Resp(f"out:{req.model}")


class _Registry:
    def __init__(self, captured):
        self._captured = captured

    async def resolve_backend_for_model(self, model):
        return _Backend(self._captured), (model or "default-model")


class _State:
    def __init__(self):
        self.captured: dict = {}
        self.provider_registry = _Registry(self.captured)


async def test_direct_invoke_calls_backend_directly():
    st = _State()
    mi = build_direct_model_invoke(st, model="m1")
    out = await mi("hello")
    assert out == "out:m1"
    req = st.captured["req"]
    # direct: no temperature forced (reasoning models 400), thinking on, single user msg
    assert req.temperature is None
    assert req.think is True
    assert req.messages[0].role == "user" and req.messages[0].content == "hello"


async def test_direct_invoke_uses_default_when_no_model():
    st = _State()
    out = await build_direct_model_invoke(st)("p")
    assert out == "out:default-model"


async def test_direct_invoke_raises_without_registry():
    class _Empty:
        provider_registry = None
    with pytest.raises(RuntimeError):
        await build_direct_model_invoke(_Empty())("p")
