"""Tests for the /api/fabric/inference endpoint.

This is the dedicated cross-peer LLM dispatch endpoint — distinct from
/v1/chat/completions. Built after the 2026-05-23 incident where reusing
the regular chat handler for cross-peer requests caused stream-killing
crashes when the receiver's orchestration deps (embedder, pack DB,
memory store) were unhealthy. The new endpoint is pure compute: it
resolves the LOCAL backend, dispatches the LLM call, and streams the
response back. Zero orchestration.

These tests pin the load-bearing invariants:

  * Verified-peer-only: non-peer requests get 403 (so a leaked URL
    can't be used as an auth-bypass for LLM dispatch).
  * Local-only backend resolution: never falls through to fabric
    routing (no A→B→C→… fan-out loops).
  * 404 on model unavailable: clean error, not silent misdispatch to
    a default backend with the wrong model name.
  * Disabled-fabric returns 503 (fail-closed when the flag is off).
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# We test the handler function directly with mocks rather than spinning
# up the whole FastAPI app — gives us tight unit-level coverage of the
# load-bearing branches without booting the lifespan.
def _make_request(*, fabric_peer: dict | None, body: dict | None, headers: dict | None = None):
    """Build a minimal Request-like object for the endpoint tests."""
    request = MagicMock()
    request.scope = {"fabric_peer": fabric_peer} if fabric_peer is not None else {}
    request.headers = headers or {}
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.provider_registry = MagicMock()

    async def _json_body():
        if body is None:
            raise ValueError("no body")
        return body

    request.json = _json_body
    return request


def _wire_registry(request: MagicMock, *, backend, clean_model: str, known: bool = True):
    """Stub provider_registry.resolve_backend_for_model + _model_map.

    The local-known gate is load-bearing because resolve_backend_for_model
    falls back to the default backend instead of returning None — without
    this gate the endpoint would silently dispatch to the wrong backend.
    """
    request.app.state.provider_registry.resolve_backend_for_model = AsyncMock(
        return_value=(backend, clean_model),
    )
    request.app.state.provider_registry._model_map = (
        {clean_model: "engine"} if known else {}
    )


@pytest.mark.asyncio
async def test_inference_403_when_not_fabric_peer():
    """Non-peer callers (e.g. a leaked URL hit from a logged-in browser)
    must 403. This endpoint is fabric-only — bypassing the normal chat
    handler means it also bypasses normal LLM rate limiting, content
    safety, audit logging, etc. Letting non-peers in would be an
    auth-bypass vulnerability.
    """
    from augmentum.proxy.fabric_routes import fabric_inference

    request = _make_request(fabric_peer=None, body={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_inference(request)
        assert excinfo.value.status_code == 403
        assert "verified fabric peer" in excinfo.value.detail


@pytest.mark.asyncio
async def test_inference_503_when_fabric_disabled():
    """Defense-in-depth: even a verified peer envelope shouldn't unlock
    the endpoint when fabric is operator-disabled. The flag is the
    operator's kill switch."""
    from augmentum.proxy.fabric_routes import fabric_inference

    request = _make_request(
        fabric_peer={"sender_node_id": "peer-abc"},
        body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = False
        with pytest.raises(HTTPException) as excinfo:
            await fabric_inference(request)
        assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_inference_400_on_missing_model():
    """Body must include a non-empty model field. Defensive — peer-side
    bugs that send an empty model shouldn't cascade to silent misdispatch."""
    from augmentum.proxy.fabric_routes import fabric_inference

    request = _make_request(
        fabric_peer={"sender_node_id": "peer-abc"},
        body={"model": "", "messages": [{"role": "user", "content": "hi"}]},
    )
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_inference(request)
        assert excinfo.value.status_code == 400
        assert "model required" in excinfo.value.detail


@pytest.mark.asyncio
async def test_inference_400_on_missing_messages():
    """Body must include a non-empty messages list."""
    from augmentum.proxy.fabric_routes import fabric_inference

    request = _make_request(
        fabric_peer={"sender_node_id": "peer-abc"},
        body={"model": "m", "messages": []},
    )
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_inference(request)
        assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_inference_404_when_model_not_local():
    """If the model isn't in the receiver's local map, return 404.

    Critical recursion guard: we MUST refuse here rather than letting
    resolve_backend_for_model fall back to the default backend
    (which would silently dispatch with the wrong model name) or to
    fabric routing (which would fan out to another peer, opening
    A→B→C→… loops).
    """
    from augmentum.proxy.fabric_routes import fabric_inference

    request = _make_request(
        fabric_peer={"sender_node_id": "peer-abc"},
        body={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
    )
    fake_backend = MagicMock()  # default backend; must NOT be called for inference
    _wire_registry(request, backend=fake_backend, clean_model="ghost", known=False)

    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        resp = await fabric_inference(request)

    assert resp.status_code == 404
    body = json.loads(resp.body)
    assert body["error"]["type"] == "model_unavailable"
    assert "ghost" in body["error"]["model"]


@pytest.mark.asyncio
async def test_inference_nonstream_returns_openai_response():
    """Non-streaming path: calls backend.chat() and returns an OpenAI
    chat completion shape."""
    from augmentum.models.base import InternalChatResponse, Message, Usage
    from augmentum.proxy.fabric_routes import fabric_inference

    fake_backend = MagicMock()
    fake_backend.chat = AsyncMock(return_value=InternalChatResponse(
        message=Message(role="assistant", content="hi from peer"),
        model="m",
        finish_reason="stop",
        usage=Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
    ))

    request = _make_request(
        fabric_peer={"sender_node_id": "peer-abc"},
        body={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    _wire_registry(request, backend=fake_backend, clean_model="m")

    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        resp = await fabric_inference(request)

    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["choices"][0]["message"]["content"] == "hi from peer"
    assert body["usage"]["total_tokens"] == 5
    # Critical: backend.chat was called, NOT chat_stream — we honored
    # the stream=False request.
    fake_backend.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_inference_does_not_use_fabric_resolver():
    """Recursion safety. The endpoint must call resolve_backend_for_model
    (local-only), NEVER resolve_backend_with_fabric. If it did, a
    request that landed on a peer without the model could fan out to
    yet another peer, opening A→B→C→… loops.
    """
    from augmentum.models.base import InternalChatResponse, Message, Usage
    from augmentum.proxy.fabric_routes import fabric_inference

    fake_backend = MagicMock()
    fake_backend.chat = AsyncMock(return_value=InternalChatResponse(
        message=Message(role="assistant", content="ok"),
        model="m", finish_reason="stop",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    ))

    request = _make_request(
        fabric_peer={"sender_node_id": "peer-abc"},
        body={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    _wire_registry(request, backend=fake_backend, clean_model="m")
    request.app.state.provider_registry.resolve_backend_with_fabric = AsyncMock(
        side_effect=AssertionError(
            "fabric_inference must NEVER call the fabric resolver — recursion bug"
        ),
    )

    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        await fabric_inference(request)

    request.app.state.provider_registry.resolve_backend_for_model.assert_awaited_once()
    request.app.state.provider_registry.resolve_backend_with_fabric.assert_not_awaited()


@pytest.mark.asyncio
async def test_inference_forwards_session_id_for_kv_continuity():
    """Optional session_id should be forwarded as kv_session_key so the
    receiver's engine slot stays warm across turns of the same
    originating conversation — but NAMESPACED by the sender node id so
    it can't collide with another peer's (or a local chat's) slot.
    """
    from augmentum.models.base import InternalChatResponse, Message, Usage
    from augmentum.proxy.fabric_routes import fabric_inference

    captured: dict = {}

    async def _capture_chat(req):
        captured["kv_session_key"] = req.kv_session_key
        captured["kv_mode"] = req.kv_mode
        return InternalChatResponse(
            message=Message(role="assistant", content="ok"),
            model="m", finish_reason="stop",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    fake_backend = MagicMock()
    fake_backend.chat = _capture_chat

    request = _make_request(
        fabric_peer={"sender_node_id": "peer-abc"},
        body={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "session_id": "s_originating_session_id",
        },
    )
    _wire_registry(request, backend=fake_backend, clean_model="m")

    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        await fabric_inference(request)

    # Namespaced by sender node — NOT the raw session_id.
    assert captured["kv_session_key"] == "fabric:peer-abc:s_originating_session_id"
    assert captured["kv_mode"] == "fabric"


@pytest.mark.asyncio
async def test_inference_kv_session_isolated_across_peers():
    """Two peers sending the SAME opaque session_id must resolve to
    DIFFERENT kv_session_keys, so peer A's KV prefix can never be
    restored into peer B's turn (cross-tenant context leak below the
    user_id layer — the slot key is what _session_fingerprint uses for
    save/restore affinity).
    """
    from augmentum.models.base import InternalChatResponse, Message, Usage
    from augmentum.proxy.fabric_routes import fabric_inference

    seen: list[str] = []

    async def _capture_chat(req):
        seen.append(req.kv_session_key)
        return InternalChatResponse(
            message=Message(role="assistant", content="ok"),
            model="m", finish_reason="stop",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    for node in ("peer-aaa", "peer-bbb"):
        fake_backend = MagicMock()
        fake_backend.chat = _capture_chat
        request = _make_request(
            fabric_peer={"sender_node_id": node},
            body={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                # Identical client-minted id from both peers — the
                # collision case.
                "session_id": "s_shared",
            },
        )
        _wire_registry(request, backend=fake_backend, clean_model="m")
        with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
            mock_settings.fabric_enabled = True
            await fabric_inference(request)

    assert seen == ["fabric:peer-aaa:s_shared", "fabric:peer-bbb:s_shared"]
    assert seen[0] != seen[1]


@pytest.mark.asyncio
async def test_inference_stream_returns_asgi_response():
    """Stream=true returns a custom Response that overrides __call__
    for raw ASGI streaming (NOT a Starlette StreamingResponse — its
    silent-exit behavior was the source of the 2026-05-23 debug
    nightmare). Verifies the right code path."""
    from augmentum.proxy.fabric_routes import fabric_inference

    fake_backend = MagicMock()
    request = _make_request(
        fabric_peer={"sender_node_id": "peer-abc"},
        body={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    _wire_registry(request, backend=fake_backend, clean_model="m")

    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        resp = await fabric_inference(request)

    # The response must be an ASGI app (have a __call__) but NOT a
    # standard JSONResponse — that would indicate we accidentally fell
    # through to a non-streaming path.
    assert callable(resp), "stream response must be ASGI-callable"
    assert type(resp).__name__ == "_RawASGIResponse"


@pytest.mark.asyncio
async def test_inference_reconstructs_full_preference_surface():
    """The 2026-06-12 chat-UI-parity fix: every user preference the
    sender forwards (thinking controls, native tools, stop sequences,
    sampler options, Continue semantics) must land on the receiver's
    InternalChatRequest so its local backend applies the same
    family-specific mapping a local chat would get.
    """
    from augmentum.models.base import InternalChatResponse, Message, Usage
    from augmentum.proxy.fabric_routes import fabric_inference

    captured: dict = {}

    async def _capture_chat(req):
        captured["req"] = req
        return InternalChatResponse(
            message=Message(role="assistant", content="ok"),
            model="m", finish_reason="stop",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    fake_backend = MagicMock()
    fake_backend.chat = _capture_chat

    tools = [{"type": "function", "function": {"name": "web_search"}}]
    request = _make_request(
        fabric_peer={"sender_node_id": "peer-abc"},
        body={
            "model": "m",
            "messages": [
                {"role": "user", "content": "what is this?",
                 "images": ["data:image/png;base64,AAAA"]},
                {"role": "assistant", "content": "a cat",
                 "thinking": "photo of a cat"},
                {"role": "user", "content": "now search it"},
            ],
            "stream": False,
            "temperature": 0.6,
            "top_p": 0.9,
            "max_tokens": 256,
            "stop": ["</s>"],
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "seed": 7,
            "tools": tools,
            "tool_choice": "auto",
            "format": "json",
            "think": True,
            "chat_template_kwargs": {"enable_thinking": True},
            "preserve_thinking": True,
            "reasoning_effort": "high",
            "raw_options": {"top_k": 40},
            "continue_last_assistant": True,
            "is_background_task": True,
        },
    )
    _wire_registry(request, backend=fake_backend, clean_model="m")

    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        await fabric_inference(request)

    req = captured["req"]
    assert req.stop == ["</s>"]
    assert req.frequency_penalty == 0.1
    assert req.presence_penalty == 0.2
    assert req.seed == 7
    assert req.tools == tools
    assert req.tool_choice == "auto"
    assert req.format == "json"
    assert req.think is True
    assert req.chat_template_kwargs == {"enable_thinking": True}
    assert req.preserve_thinking is True
    assert req.reasoning_effort == "high"
    assert req.raw_options == {"top_k": 40}
    assert req.continue_last_assistant is True
    assert req.is_background_task is True
    # Message extras survive coercion.
    assert req.messages[0].images == ["data:image/png;base64,AAAA"]
    assert req.messages[1].thinking == "photo of a cat"


@pytest.mark.asyncio
async def test_inference_think_false_and_mistyped_fields_are_safe():
    """think=False stays an explicit False (always-on families need
    the peer's backend to emit enable_thinking=false), and mis-typed
    preference fields from a version-skewed sender degrade to local
    defaults instead of crashing the endpoint.
    """
    from augmentum.models.base import InternalChatResponse, Message, Usage
    from augmentum.proxy.fabric_routes import fabric_inference

    captured: dict = {}

    async def _capture_chat(req):
        captured["req"] = req
        return InternalChatResponse(
            message=Message(role="assistant", content="ok"),
            model="m", finish_reason="stop",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    fake_backend = MagicMock()
    fake_backend.chat = _capture_chat

    request = _make_request(
        fabric_peer={"sender_node_id": "peer-abc"},
        body={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "think": False,
            # Mis-typed garbage — must not crash, must not leak through.
            "stop": "not-a-list",
            "tools": "not-a-list",
            "tool_choice": 42,
            "chat_template_kwargs": "not-a-dict",
            "raw_options": ["not", "a", "dict"],
            "seed": "not-an-int",
            "format": 123,
            "reasoning_effort": {"not": "a string"},
        },
    )
    _wire_registry(request, backend=fake_backend, clean_model="m")

    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        await fabric_inference(request)

    req = captured["req"]
    assert req.think is False
    assert req.stop is None
    assert req.tools is None
    assert req.tool_choice is None
    assert req.chat_template_kwargs is None
    assert req.raw_options is None
    assert req.seed is None
    assert req.format is None
    assert req.reasoning_effort is None


def test_chunk_serialiser_emits_reasoning_and_tool_calls():
    """Regression pin for the always-None getattr bug: the serializer
    read ``chunk.thinking`` but InternalStreamChunk's field is
    ``thinking_delta`` — reasoning never crossed the wire, which is
    exactly "thinking doesn't work over fabric". Also pins tool-call
    deltas riding chunk.augmentum → delta.tool_calls.
    """
    from augmentum.models.base import InternalStreamChunk
    from augmentum.proxy.fabric_routes import _internal_chunk_to_openai_sse_dict

    chunk = InternalStreamChunk(
        content_delta="", thinking_delta="reasoning...", role="assistant",
    )
    sse = _internal_chunk_to_openai_sse_dict(chunk, "chatcmpl-x", "m")
    assert sse["choices"][0]["delta"]["reasoning_content"] == "reasoning..."

    tc = [{"index": 0, "id": "call_1", "type": "function",
           "function": {"name": "web_search", "arguments": "{}"}}]
    chunk2 = InternalStreamChunk(content_delta="")
    chunk2.augmentum = {"tool_calls": tc}
    sse2 = _internal_chunk_to_openai_sse_dict(chunk2, "chatcmpl-x", "m")
    assert sse2["choices"][0]["delta"]["tool_calls"] == tc
