"""acp_app — production wiring that assembles the real coder loop for a session."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import augmentum.proxy.handler_factory as HF
from augmentum.coder.acp_app import (
    build_coder_handler,
    build_coder_request,
    make_app_loop_runner,
)

_EXECUTOR = object()


def _session():
    return SimpleNamespace(session_id="acp-abc", user_id="u1", executor=_EXECUTOR)


class _Registry:
    def __init__(self, backend="BACKEND", resolved="resolved-model") -> None:
        self._backend = backend
        self._resolved = resolved

    async def resolve_backend_with_fabric(self, model, *, user_id):
        return (self._backend, self._resolved)


def _app_state(registry=None):
    return SimpleNamespace(provider_registry=registry or _Registry())


# -- request assembly --------------------------------------------------------

def test_build_request_has_coder_kv_keys_and_prompt() -> None:
    req = build_coder_request(_session(), "hello", model="m")
    assert req.model == "m"
    assert req.kv_mode == "coder"
    assert req.kv_session_key  # derived, non-empty
    assert req.stream is True
    assert req.messages[-1].role == "user"
    assert req.messages[-1].content == "hello"


# -- handler assembly (factory reused, editor deltas applied) ----------------

class _FakeCoderHandler:  # name is checked by build_coder_handler's guard
    __name__ = "CoderHandler"

    def __init__(self) -> None:
        self._executor = None
        self._permission_callback = None


# rebind __class__.__name__ so type(h).__name__ == "CoderHandler"
_FakeCoderHandler.__qualname__ = "CoderHandler"
CoderHandler = _FakeCoderHandler
CoderHandler.__name__ = "CoderHandler"


@pytest.mark.asyncio
async def test_build_handler_resolves_backend_and_applies_editor_deltas(monkeypatch) -> None:
    captured = {}

    def fake_factory(mode, backend, session_id, app_state, *, workspace_id, user_id, coder_strategy):
        captured.update(
            backend=backend, session_id=session_id, workspace_id=workspace_id,
            user_id=user_id, coder_strategy=coder_strategy,
        )
        return CoderHandler()

    monkeypatch.setattr(HF, "get_handler_for_mode", fake_factory)
    sess = _session()
    handler = await build_coder_handler(_app_state(), sess, model="qwen")

    assert type(handler).__name__ == "CoderHandler"
    # editor delta 1: tools act on the editor, not a container
    assert handler._executor is _EXECUTOR
    # editor delta 2: a permission callback is installed (editor owns approval)
    assert callable(handler._permission_callback)
    assert await handler._permission_callback("write_file", {}) is True
    # canonical assembly got the right identity + native strategy
    assert captured["backend"] == "BACKEND"
    assert captured["session_id"] == "acp-abc"
    assert captured["workspace_id"] == "acp-abc"
    assert captured["user_id"] == "u1"
    assert captured["coder_strategy"] == "native"


@pytest.mark.asyncio
async def test_build_handler_raises_when_model_unavailable() -> None:
    app_state = _app_state(_Registry(backend=None, resolved=""))
    with pytest.raises(RuntimeError, match="model unavailable"):
        await build_coder_handler(app_state, _session(), model="ghost")


@pytest.mark.asyncio
async def test_build_handler_raises_on_passthrough_fallback(monkeypatch) -> None:
    class PassthroughHandler:
        pass

    monkeypatch.setattr(HF, "get_handler_for_mode", lambda *a, **k: PassthroughHandler())
    with pytest.raises(RuntimeError, match="fell back"):
        await build_coder_handler(_app_state(), _session(), model="qwen")


@pytest.mark.asyncio
async def test_build_handler_raises_without_provider_registry() -> None:
    with pytest.raises(RuntimeError, match="provider registry unavailable"):
        await build_coder_handler(SimpleNamespace(provider_registry=None), _session(), model="m")


# -- the full production loop_runner (async build_handler is awaited) ---------

class _FakeStream:
    def __init__(self, chunks) -> None:
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed = True


def _chunk(**kw):
    kw.setdefault("thinking_delta", None)
    kw.setdefault("content_delta", None)
    kw.setdefault("done", False)
    return SimpleNamespace(**kw)


@pytest.mark.asyncio
async def test_make_app_loop_runner_drives_loop_through_async_build(monkeypatch) -> None:
    seen = {}

    class StreamingCoderHandler:
        __name__ = "CoderHandler"

        def __init__(self) -> None:
            self._executor = None
            self._permission_callback = None

        def handle_stream(self, request):
            seen["request"] = request
            return _FakeStream([
                _chunk(thinking_delta="plan"),
                _chunk(content_delta="hi", done=True),
            ])

    StreamingCoderHandler.__name__ = "CoderHandler"
    monkeypatch.setattr(HF, "get_handler_for_mode", lambda *a, **k: StreamingCoderHandler())

    runner = make_app_loop_runner(_app_state(), model="qwen")
    sess = _session()
    events = [ev async for ev in runner(sess, "do the thing")]

    assert events == [("thought", {"text": "plan"}), ("text", {"text": "hi"})]
    # build_request received the SESSION (for KV keys) + the prompt text
    assert seen["request"].kv_mode == "coder"
    assert seen["request"].messages[-1].content == "do the thing"
