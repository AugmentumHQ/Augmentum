"""Tests for FabricBackend's ModelBackend implementation.

Verifies the proxy actually constructs the right HTTP calls,
handles errors gracefully, and respects the ModelBackend contract.
We mock httpx (no real peer needed) and use the canonical OpenAI
SSE shape.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from augmentum.fabric.capabilities import LLMInferenceCapability
from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.fabric_backend import FabricBackend


async def _warm_drive_load(_model_id):
    """Warm-peer stand-in for FabricBackend._drive_peer_load: an empty
    async generator (no cold_start/ready markers → no model_load stage),
    so chat_stream goes straight to inference in tests that aren't
    exercising the load handshake."""
    return
    yield  # noqa — unreachable; makes this an async generator


def _make_backend(*, http_client=None, addr="192.168.1.10:6443"):
    cap = LLMInferenceCapability(
        backend="peer", model_id="test-model",
        model_family="qwen3", params_b=72.0, ctx_max=32768,
    )
    backend = FabricBackend(
        http_client=http_client or MagicMock(),
        peer_node_id="peer-abc",
        peer_addr=addr,
        advertised_capability=cap,
    )
    # Bypass the pre-dispatch load gate by default — tests that
    # exercise it explicitly construct a FabricBackend directly and
    # don't go through this helper. Existing tests focus on the
    # chat dispatch surface, not load coordination.
    backend.ensure_peer_model_loaded = AsyncMock()
    # chat_stream consumes _drive_peer_load directly (to emit the
    # model_load stage); warm-peer stand-in yields nothing.
    backend._drive_peer_load = _warm_drive_load
    return backend


@pytest.mark.asyncio
async def test_list_models_from_capability():
    """FabricBackend reports the single model it was built for, not
    the full peer model list. The coordinator already knows the full
    list elsewhere.
    """
    backend = _make_backend()
    models = await backend.list_models()
    assert len(models) == 1
    assert models[0].name == "test-model"
    assert models[0].context_length == 32768


@pytest.mark.asyncio
async def test_show_model_returns_capability_fields():
    backend = _make_backend()
    details = await backend.show_model("test-model")
    assert details.family == "qwen3"
    assert "72.0B" in details.parameter_size


@pytest.mark.asyncio
async def test_chat_proxies_to_peer_addr():
    """chat() POSTs to {peer_addr}/api/fabric/inference with the
    OpenAI payload shape + Phase 3.x signed peer headers. Verify
    URL, headers, and body shape.

    The /api/fabric/inference endpoint is purpose-built for cross-peer
    LLM dispatch — it bypasses the receiver's orchestration-heavy chat
    handler (mode classification, memory recall, knowledge packs,
    narrative state, tools) which the initiating peer has already run.
    """
    import aiosqlite

    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "model": "test-model",
        "choices": [{
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=fake_resp)

    # Phase 3.x: backend needs identity + user_id to sign requests.
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.commit()
    identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
    try:
        cap = LLMInferenceCapability(
            backend="peer", model_id="test-model", model_family="qwen3",
            params_b=72.0, ctx_max=32768,
        )
        from augmentum.models.fabric_backend import FabricBackend
        backend = FabricBackend(
            http_client=fake_client, peer_node_id="peer-abc",
            peer_addr="192.168.1.10:6443", advertised_capability=cap,
            identity=identity, user_id="user-42",
        )
        backend.ensure_peer_model_loaded = AsyncMock()
        backend._drive_peer_load = _warm_drive_load
        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="hi")],
            temperature=0.7,
        )
        resp = await backend.chat(request)

        assert resp.message.content == "hello"
        assert resp.usage.total_tokens == 6
        # Verify the URL + headers.
        call_args = fake_client.post.call_args
        url = call_args[0][0]
        assert url.endswith("/api/fabric/inference")
        assert "192.168.1.10:6443" in url
        # X-Fabric-Sender is OUR node_id (the sender from receiver's POV),
        # not the destination peer's id (which is "peer-abc" here).
        headers = call_args[1]["headers"]
        assert headers["X-Fabric-Sender"] == identity.node_id
        assert headers["X-Fabric-User-Id"] == "user-42"
        assert "X-Fabric-Signature" in headers
        assert "X-Fabric-Timestamp" in headers
        # Body shape. Phase 3.y: payload is serialised + sent via
        # ``content=`` (raw bytes) so we can hash it for the signed
        # canonical bytes; decode and inspect.
        import json as _json
        body_bytes = call_args[1]["content"]
        assert isinstance(body_bytes, bytes)
        payload = _json.loads(body_bytes)
        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert payload["messages"][0]["role"] == "user"
        assert payload["temperature"] == 0.7
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_chat_omits_signed_headers_when_no_identity():
    """Backward-compat: when constructed without identity (the legacy
    path / some tests), no X-Fabric-* signed headers are emitted.
    The receiver's middleware will treat this as a non-fabric
    request and pass it through to user auth (which will likely 401).
    """
    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": "x"},
                     "finish_reason": "stop"}],
        "usage": {},
    }
    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=fake_resp)

    backend = _make_backend(http_client=fake_client)  # no identity/user_id
    request = InternalChatRequest(
        model="test-model", messages=[Message(role="user", content="hi")],
    )
    await backend.chat(request)
    headers = fake_client.post.call_args[1]["headers"]
    # No signed headers when identity wasn't supplied.
    assert "X-Fabric-Sender" not in headers
    assert "X-Fabric-Signature" not in headers


@pytest.mark.asyncio
async def test_chat_raises_on_peer_4xx():
    """A 401/4xx from the peer is surfaced as a clean RuntimeError
    rather than silently producing empty/garbage output.
    """
    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 401
    fake_resp.text = "Unauthorized"
    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=fake_resp)

    backend = _make_backend(http_client=fake_client)
    request = InternalChatRequest(model="test-model", messages=[Message(role="user", content="x")])
    with pytest.raises(RuntimeError, match="401"):
        await backend.chat(request)


@pytest.mark.asyncio
async def test_chat_raises_on_connect_error():
    """When the peer is unreachable, raise a clean RuntimeError
    naming the issue.
    """
    fake_client = MagicMock()
    fake_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    backend = _make_backend(http_client=fake_client)
    request = InternalChatRequest(model="test-model", messages=[Message(role="user", content="x")])
    with pytest.raises(RuntimeError, match="peer unreachable"):
        await backend.chat(request)


@pytest.mark.asyncio
async def test_chat_stream_parses_sse_chunks():
    """chat_stream consumes OpenAI-format SSE and yields
    InternalStreamChunks correctly.
    """
    async def fake_aiter_lines():
        yield "data: " + json.dumps({
            "model": "test-model",
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}],
        })
        yield "data: " + json.dumps({
            "model": "test-model",
            "choices": [{"delta": {"content": "Hello "}, "finish_reason": None}],
        })
        yield "data: " + json.dumps({
            "model": "test-model",
            "choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}],
        })
        yield "data: [DONE]"

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 200
    fake_resp.aiter_lines = fake_aiter_lines

    class _AsyncCM:
        async def __aenter__(self):
            return fake_resp
        async def __aexit__(self, *a, **k):
            return False

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_AsyncCM())

    backend = _make_backend(http_client=fake_client)
    request = InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )
    chunks = []
    async for chunk in backend.chat_stream(request):
        chunks.append(chunk)

    # Should see: role chunk, content "Hello ", content "world",
    # then done.
    assert len(chunks) == 4
    assert chunks[0].role == "assistant"
    assert chunks[1].content_delta == "Hello "
    assert chunks[2].content_delta == "world"
    assert chunks[2].finish_reason == "stop"
    assert chunks[3].done is True


@pytest.mark.asyncio
async def test_chat_stream_raises_on_peer_4xx():
    """4xx from peer's stream endpoint surfaces as clean RuntimeError."""
    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 401
    fake_resp.aread = AsyncMock(return_value=b"Unauthorized")

    class _AsyncCM:
        async def __aenter__(self):
            return fake_resp
        async def __aexit__(self, *a, **k):
            return False

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_AsyncCM())

    backend = _make_backend(http_client=fake_client)
    request = InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )
    with pytest.raises(RuntimeError, match="401"):
        async for _ in backend.chat_stream(request):
            pass


@pytest.mark.asyncio
async def test_http_scheme_defaults_to_https():
    """Bare host:port assumes HTTPS (Caddy + TLS terminus)."""
    backend = _make_backend(addr="192.168.1.10:6443")
    assert backend._http_scheme() == "https"


@pytest.mark.asyncio
async def test_http_scheme_respects_full_url():
    """When the operator supplied a full URL, don't override it."""
    backend = _make_backend(addr="http://peer.local:8080")
    assert backend._http_scheme() == "http"


@pytest.mark.asyncio
async def test_chat_stream_emits_request_id_header():
    """Phase 9.3: every chat_stream call mints a request_id and
    sends it as X-Fabric-Request-Id so the peer-side middleware
    can register the in-flight task for later MSG_CANCEL_REQUEST.
    """
    import aiosqlite

    from augmentum.fabric.capabilities import LLMInferenceCapability
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.models.fabric_backend import FabricBackend
    from augmentum.state.settings_store import SettingsStore

    # Build a stream context manager that yields one [DONE] chunk
    # so the generator completes cleanly.
    class _StreamCM:
        def __init__(self):
            self.status_code = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def aiter_lines(self):
            yield "data: [DONE]"

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_StreamCM())

    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.commit()
    identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
    try:
        cap = LLMInferenceCapability(model_id="m")
        backend = FabricBackend(
            http_client=fake_client, peer_node_id="peer-x",
            peer_addr="192.168.1.10:6443", advertised_capability=cap,
            identity=identity, user_id="user-42",
        )
        backend.ensure_peer_model_loaded = AsyncMock()
        backend._drive_peer_load = _warm_drive_load
        request = InternalChatRequest(
            model="m", messages=[Message(role="user", content="hi")],
        )
        async for _ in backend.chat_stream(request):
            pass
        kwargs = fake_client.stream.call_args[1]
        headers = kwargs["headers"]
        assert "X-Fabric-Request-Id" in headers
        assert headers["X-Fabric-Request-Id"].startswith("req-")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_chat_404_model_missing_raises_peer_model_missing_error():
    """Phase 9-failure-surfacing: when the peer 404s with a body
    suggesting the model isn't there, the FabricBackend raises
    PeerModelMissingError (not generic RuntimeError) AND asks the
    coordinator to drop the stale capability.
    """
    from augmentum.models.fabric_backend import (
        PeerModelMissingError,
    )

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 404
    fake_resp.text = "model not found: flux-dev"
    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=fake_resp)

    coord = MagicMock()
    coord.invalidate_peer_capability = MagicMock()

    backend = _make_backend(http_client=fake_client)
    backend._coordinator = coord  # inject directly; _make_backend has no kwarg yet

    request = InternalChatRequest(
        model="flux-dev", messages=[Message(role="user", content="x")],
    )
    with pytest.raises(PeerModelMissingError):
        await backend.chat(request)

    # The capability for flux-dev on this peer should be dropped.
    coord.invalidate_peer_capability.assert_called_once_with(
        backend._peer_node_id, kind="llm.inference", model_id="flux-dev",
    )


@pytest.mark.asyncio
async def test_chat_generic_4xx_raises_peer_protocol_error():
    """4xx that doesn't match the model-missing signal becomes
    PeerProtocolError, NOT PeerModelMissingError. The peer is
    behaving badly but the model is still advertised; don't drop
    the capability.
    """
    from augmentum.models.fabric_backend import (
        PeerModelMissingError,
        PeerProtocolError,
    )

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 400
    fake_resp.text = "bad request: malformed prompt"
    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=fake_resp)

    coord = MagicMock()
    coord.invalidate_peer_capability = MagicMock()

    backend = _make_backend(http_client=fake_client)
    backend._coordinator = coord
    request = InternalChatRequest(
        model="flux-dev", messages=[Message(role="user", content="x")],
    )
    with pytest.raises(PeerProtocolError):
        await backend.chat(request)
    with pytest.raises(PeerProtocolError):
        # Subclass relationship: PeerProtocolError is NOT
        # PeerModelMissingError. Verify the type discriminator.
        try:
            raise PeerProtocolError("x")
        except PeerModelMissingError:
            pytest.fail("PeerProtocolError should not be caught as PeerModelMissingError")
        except PeerProtocolError:
            raise PeerProtocolError("x")

    coord.invalidate_peer_capability.assert_not_called()


@pytest.mark.asyncio
async def test_chat_connect_error_raises_peer_unreachable_error():
    """Transport-level failure → PeerUnreachableError (transient
    signal vs. PeerProtocolError which is "peer is misbehaving").
    """
    from augmentum.models.fabric_backend import PeerUnreachableError

    fake_client = MagicMock()
    fake_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

    backend = _make_backend(http_client=fake_client)
    request = InternalChatRequest(
        model="x", messages=[Message(role="user", content="x")],
    )
    with pytest.raises(PeerUnreachableError):
        await backend.chat(request)


@pytest.mark.asyncio
async def test_chat_stream_cancel_sends_ws_backstop():
    """Phase 9.4: when the chat_stream async generator is cancelled,
    we MUST push a MSG_CANCEL_REQUEST envelope to the peer over the
    coordinator's WS. Belt-and-suspenders for the TCP-close path
    which is best-effort under proxies / kernel buffering.
    """
    import asyncio as _asyncio
    import aiosqlite

    from augmentum.fabric.capabilities import LLMInferenceCapability
    from augmentum.fabric.identity import FabricIdentity
    from augmentum.models.fabric_backend import FabricBackend
    from augmentum.state.settings_store import SettingsStore

    # Stream that hangs forever so the caller can cancel mid-stream.
    class _HangingStreamCM:
        def __init__(self):
            self.status_code = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def aiter_lines(self):
            # Yield one chunk then hang
            yield "data: " + '{"choices":[{"delta":{"content":"hi"}}]}'
            await _asyncio.Event().wait()  # never completes

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_HangingStreamCM())

    # Coordinator with the send_to_peer method intercepted.
    coord = MagicMock()
    coord.send_to_peer = AsyncMock(return_value=True)

    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.commit()
    identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
    try:
        cap = LLMInferenceCapability(model_id="m")
        backend = FabricBackend(
            http_client=fake_client, peer_node_id="peer-x",
            peer_addr="192.168.1.10:6443", advertised_capability=cap,
            identity=identity, user_id="u", coordinator=coord,
        )
        backend.ensure_peer_model_loaded = AsyncMock()
        backend._drive_peer_load = _warm_drive_load
        request = InternalChatRequest(
            model="m", messages=[Message(role="user", content="hi")],
        )
        gen = backend.chat_stream(request)
        # Pull one chunk
        await gen.__anext__()
        # Now cancel by closing the generator.
        await gen.aclose()

        # Wait briefly for the finally-block backstop to fire.
        await _asyncio.sleep(0.05)
        coord.send_to_peer.assert_called_once()
        call_kwargs = coord.send_to_peer.call_args[1]
        assert call_kwargs["msg_type"] == "cancel_request"
        assert call_kwargs["payload"]["request_id"].startswith("req-")
    finally:
        await conn.close()


# ── Pre-dispatch model-load coordination ─────────────────────────────


def _load_resp(status: str, *, current_model: str = "", reason: str = "") -> MagicMock:
    """Build a fake httpx.Response for /api/fabric/load_model + /load_status."""
    r = MagicMock(spec=httpx.Response)
    r.status_code = 200
    body = {"status": status, "current_model": current_model}
    if reason:
        body["reason"] = reason
    r.json.return_value = body
    return r


async def _make_signed_backend() -> "FabricBackend":
    """Construct a real FabricBackend with identity + user_id wired so
    ensure_peer_model_loaded actually runs its signed-call path.
    """
    import aiosqlite

    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.commit()
    identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
    cap = LLMInferenceCapability(model_id="my-model")
    backend = FabricBackend(
        http_client=MagicMock(),  # tests override per-call
        peer_node_id="peer-z",
        peer_addr="192.168.1.10:6443",
        advertised_capability=cap,
        identity=identity,
        user_id="user-x",
    )
    backend._test_conn = conn  # caller cleans up
    return backend


@pytest.mark.asyncio
async def test_ensure_peer_model_loaded_noops_when_ready():
    """Common warm path: peer's /load_model returns status=ready in
    the same handshake. We never enter the polling loop."""
    backend = await _make_signed_backend()
    try:
        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=_load_resp("ready"))
        fake_client.get = AsyncMock()
        backend._client = fake_client

        await backend.ensure_peer_model_loaded("my-model")

        # Single POST to /load_model, zero polls.
        assert fake_client.post.call_count == 1
        assert "/api/fabric/load_model" in fake_client.post.call_args[0][0]
        fake_client.get.assert_not_called()
    finally:
        await backend._test_conn.close()


@pytest.mark.asyncio
async def test_ensure_peer_model_loaded_polls_until_ready():
    """Cold load path: /load_model returns status=loading, we poll
    /load_status until it reports ready. Two polls then ready —
    third call returns ready and we stop.
    """
    backend = await _make_signed_backend()
    try:
        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=_load_resp("loading"))
        # Two loading responses then ready.
        get_responses = [
            _load_resp("loading"),
            _load_resp("loading"),
            _load_resp("ready"),
        ]
        fake_client.get = AsyncMock(side_effect=get_responses)
        backend._client = fake_client

        await backend.ensure_peer_model_loaded("my-model")

        assert fake_client.post.call_count == 1
        # 3 polls — two loading + one ready that broke the loop.
        assert fake_client.get.call_count == 3
    finally:
        await backend._test_conn.close()


@pytest.mark.asyncio
async def test_ensure_peer_model_loaded_raises_on_failure():
    """When the receiver's background load task errors (OOM, etc.),
    /load_status returns status=failed with a reason. We raise
    PeerProtocolError carrying the reason so the operator sees why.
    """
    from augmentum.models.fabric_backend import PeerProtocolError

    backend = await _make_signed_backend()
    try:
        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=_load_resp("loading"))
        fake_client.get = AsyncMock(
            return_value=_load_resp("failed", reason="OutOfMemoryError: 12.4 GB needed"),
        )
        backend._client = fake_client

        with pytest.raises(PeerProtocolError, match="OutOfMemoryError"):
            await backend.ensure_peer_model_loaded("my-model")
    finally:
        await backend._test_conn.close()


@pytest.mark.asyncio
async def test_ensure_peer_model_loaded_raises_model_missing_on_404():
    """Receiver 404s /load_model when the requested model isn't on
    disk. We raise PeerModelMissingError + invalidate the stale
    capability so the next dispatch attempt routes elsewhere.
    """
    from augmentum.models.fabric_backend import PeerModelMissingError

    backend = await _make_signed_backend()
    try:
        # Attach a mock coordinator so capability invalidation can fire.
        coord = MagicMock()
        backend._coordinator = coord

        not_found = MagicMock(spec=httpx.Response)
        not_found.status_code = 404
        not_found.text = "model 'my-model' not found on this peer"
        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=not_found)
        backend._client = fake_client

        with pytest.raises(PeerModelMissingError):
            await backend.ensure_peer_model_loaded("my-model")

        coord.invalidate_peer_capability.assert_called_once()
    finally:
        await backend._test_conn.close()


@pytest.mark.asyncio
async def test_ensure_peer_model_loaded_raises_peer_unreachable():
    """Network failure during the load handshake surfaces as
    PeerUnreachableError, same shape as a chat-dispatch network
    failure — caller can treat them uniformly.
    """
    from augmentum.models.fabric_backend import PeerUnreachableError

    backend = await _make_signed_backend()
    try:
        fake_client = MagicMock()
        fake_client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused"),
        )
        backend._client = fake_client

        with pytest.raises(PeerUnreachableError, match="unreachable"):
            await backend.ensure_peer_model_loaded("my-model")
    finally:
        await backend._test_conn.close()


@pytest.mark.asyncio
async def test_ensure_peer_model_loaded_signs_both_calls():
    """Both POST /load_model and GET /load_status must carry signed
    X-Fabric-* headers so the receiver's middleware authenticates
    them — these are NOT in _PUBLIC_PATHS.
    """
    backend = await _make_signed_backend()
    try:
        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=_load_resp("loading"))
        fake_client.get = AsyncMock(return_value=_load_resp("ready"))
        backend._client = fake_client

        await backend.ensure_peer_model_loaded("my-model")

        post_headers = fake_client.post.call_args[1]["headers"]
        get_headers = fake_client.get.call_args[1]["headers"]
        for headers in (post_headers, get_headers):
            assert "X-Fabric-Sender" in headers
            assert "X-Fabric-Signature" in headers
            assert "X-Fabric-Timestamp" in headers
    finally:
        await backend._test_conn.close()


@pytest.mark.asyncio
async def test_ensure_peer_model_loaded_empty_model_is_noop():
    """Defensive: empty model_id should no-op rather than POST a
    nonsense load request. The role-resolver "use default" empty-
    name idiom must not trigger a load round-trip.
    """
    backend = await _make_signed_backend()
    try:
        fake_client = MagicMock()
        fake_client.post = AsyncMock()
        fake_client.get = AsyncMock()
        backend._client = fake_client

        await backend.ensure_peer_model_loaded("")

        fake_client.post.assert_not_called()
        fake_client.get.assert_not_called()
    finally:
        await backend._test_conn.close()


# ── Empty-user_id signing (cross-peer internal dispatch paths) ───────


@pytest.mark.asyncio
async def test_load_gate_signs_with_empty_user_id():
    """Internal LLM dispatch sites (jobs, narrative refresh, draft_section,
    reasoning executor, …) resolve backends without a request context, so
    they pass ``user_id=""`` through to FabricBackend. The sender must
    still sign the envelope — receiver-side middleware accepts empty
    user_id_claim under the per-peer service user model (1c213d1). If
    the sender bails out of signing the receiver sees no fabric headers
    and the AuthMiddleware 401s with no fabric_peer_* log on the
    receiver side — the diagnostic fingerprint of this bug.
    """
    import aiosqlite

    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        cap = LLMInferenceCapability(model_id="my-model")
        backend = FabricBackend(
            http_client=MagicMock(),
            peer_node_id="peer-z",
            peer_addr="192.168.1.10:6443",
            advertised_capability=cap,
            identity=identity,
            user_id="",  # internal dispatch — no request context
        )

        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=_load_resp("ready"))
        backend._client = fake_client

        await backend.ensure_peer_model_loaded("my-model")

        post_headers = fake_client.post.call_args[1]["headers"]
        # The whole point of the fix: signed envelope must be present
        # even with empty user_id.
        assert "X-Fabric-Sender" in post_headers
        assert "X-Fabric-Signature" in post_headers
        assert "X-Fabric-Timestamp" in post_headers
        # User-id header IS present, just empty — receiver's middleware
        # accepts that since 1c213d1.
        assert post_headers.get("X-Fabric-User-Id") == ""
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_chat_signs_with_empty_user_id():
    """Same fix applies to the chat dispatch path: _fabric_headers must
    sign with empty user_id, not bail out to unsigned headers.
    """
    import aiosqlite

    from augmentum.fabric.identity import FabricIdentity
    from augmentum.state.settings_store import SettingsStore

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=fake_resp)

    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " updated_at TEXT DEFAULT (datetime('now')))"
        )
        await conn.commit()
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        cap = LLMInferenceCapability(model_id="test-model")
        backend = FabricBackend(
            http_client=fake_client,
            peer_node_id="peer-z",
            peer_addr="192.168.1.10:6443",
            advertised_capability=cap,
            identity=identity,
            user_id="",
        )
        backend.ensure_peer_model_loaded = AsyncMock()
        backend._drive_peer_load = _warm_drive_load

        req = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="hi")],
        )
        await backend.chat(req)

        post_headers = fake_client.post.call_args[1]["headers"]
        assert "X-Fabric-Sender" in post_headers
        assert "X-Fabric-Signature" in post_headers
        assert post_headers.get("X-Fabric-User-Id") == ""
    finally:
        await conn.close()


# ── URL-encoded model_id in load_status query ────────────────────────


@pytest.mark.asyncio
async def test_load_status_url_encodes_special_chars_in_model_id():
    """Model IDs containing '+', spaces, '&', or '=' would break
    signature canonicalisation if the sender signed the raw string but
    httpx URL-encoded it on the wire. The receiver reconstructs the
    signed path from ASGI's raw query_string (already-encoded form),
    so the sender must sign the encoded form too.
    """
    backend = await _make_signed_backend()
    try:
        fake_client = MagicMock()
        # Need to enter the polling loop so we get a GET call to inspect.
        fake_client.post = AsyncMock(return_value=_load_resp("loading"))
        fake_client.get = AsyncMock(return_value=_load_resp("ready"))
        backend._client = fake_client

        await backend.ensure_peer_model_loaded("weird model+name=test")

        # URL passed to httpx.get must be the encoded form.
        get_url = fake_client.get.call_args[0][0]
        assert "weird%20model%2Bname%3Dtest" in get_url
        # And the signed path must match the encoded URL — middleware
        # rebuilds canonical bytes from the encoded query_string, so
        # signing the raw string would mismatch.
        get_headers = fake_client.get.call_args[1]["headers"]
        assert "X-Fabric-Signature" in get_headers
        # Sanity: the URL doesn't contain the raw decoded form anywhere.
        assert "weird model+name=test" not in get_url
    finally:
        await backend._test_conn.close()


# ── Structural-failure detection (peer load_status reason parsing) ──


def test_is_structural_load_failure_catches_sigsegv():
    """The Qwen3.6-A3B-IQ4_NL crash in production logs: peer's
    llama-server exits during startup with code -11. Should be
    structural so the capability gets invalidated and dropdown
    stops offering it.
    """
    from augmentum.models.fabric_backend import _is_structural_load_failure

    assert _is_structural_load_failure(
        "RuntimeError: llama-server exited during startup with code -11"
    )


def test_is_structural_load_failure_catches_oom():
    from augmentum.models.fabric_backend import _is_structural_load_failure

    assert _is_structural_load_failure("CUDA out of memory at layer 18")
    assert _is_structural_load_failure("cuda oom during prefill")


def test_is_structural_load_failure_catches_gated_delta_net():
    from augmentum.models.fabric_backend import _is_structural_load_failure

    assert _is_structural_load_failure(
        "sched_reserve: fused Gated Delta Net (chunked) not supported"
    )


def test_is_structural_load_failure_ignores_transient():
    """Don't invalidate capability on transient signals — the model
    might load fine on the next attempt (peer was busy with another
    load, request raced a swap, etc.).
    """
    from augmentum.models.fabric_backend import _is_structural_load_failure

    assert not _is_structural_load_failure("model not ready yet")
    assert not _is_structural_load_failure("superseded by newer load request")
    assert not _is_structural_load_failure("")
    assert not _is_structural_load_failure("timeout while polling")



# ── Preference forwarding (chat-UI parity over fabric) ──────────────
#
# The 2026-06-12 fix: _build_payload was a "minimal viable serialiser"
# that dropped everything beyond model/messages/temperature/top_p/
# max_tokens — so the thinking toggle, native tools, stop sequences,
# sampler prefs, and Continue-button semantics all silently vanished
# when a chat was routed to a fabric peer. These tests pin the full
# preference surface to the wire.


def test_build_payload_forwards_full_preference_surface():
    backend = _make_backend()
    request = InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        temperature=0.6,
        top_p=0.9,
        max_tokens=512,
        stop=["</s>", "Observation:"],
        frequency_penalty=0.1,
        presence_penalty=0.2,
        seed=42,
        tools=[{"type": "function", "function": {"name": "web_search"}}],
        tool_choice="auto",
        format="json",
        think=True,
        chat_template_kwargs={"enable_thinking": True, "reasoning_budget": 2048},
        preserve_thinking=True,
        reasoning_effort="high",
        raw_options={"top_k": 40, "min_p": 0.05},
        continue_last_assistant=True,
        is_background_task=True,
        kv_session_key="s_chat-123",
    )
    payload = backend._build_payload(request, stream=True)

    assert payload["stream"] is True
    assert payload["stop"] == ["</s>", "Observation:"]
    assert payload["frequency_penalty"] == 0.1
    assert payload["presence_penalty"] == 0.2
    assert payload["seed"] == 42
    assert payload["tools"][0]["function"]["name"] == "web_search"
    assert payload["tool_choice"] == "auto"
    assert payload["format"] == "json"
    assert payload["think"] is True
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    assert payload["preserve_thinking"] is True
    assert payload["reasoning_effort"] == "high"
    assert payload["raw_options"]["top_k"] == 40
    assert payload["continue_last_assistant"] is True
    assert payload["is_background_task"] is True
    assert payload["session_id"] == "s_chat-123"


def test_build_payload_think_false_is_explicit():
    """think=False must cross the wire as an explicit boolean — for
    always-on-by-default families (Qwen 3.x) the OFF state requires
    the peer to emit ``enable_thinking: false``; omission would leave
    the peer's template default (thinking ON) in charge and the UI
    toggle would appear broken.
    """
    backend = _make_backend()
    request = InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
    )
    payload = backend._build_payload(request, stream=False)

    assert payload["think"] is False
    # Unset optionals stay absent so the peer's defaults apply.
    for key in ("stop", "tools", "tool_choice", "format",
                "chat_template_kwargs", "preserve_thinking",
                "reasoning_effort", "raw_options",
                "continue_last_assistant", "is_background_task",
                "session_id"):
        assert key not in payload, key


def test_messages_serialise_images_and_thinking():
    from augmentum.models.fabric_backend import _messages_to_openai

    msgs = [
        Message(role="user", content="what is this?",
                images=["data:image/png;base64,AAAA"]),
        Message(role="assistant", content="a cat",
                thinking="the user attached a photo of a cat"),
    ]
    out = _messages_to_openai(msgs)

    assert out[0]["images"] == ["data:image/png;base64,AAAA"]
    assert out[1]["thinking"] == "the user attached a photo of a cat"
    # Plain messages stay minimal — no None-valued keys.
    plain = _messages_to_openai([Message(role="user", content="hi")])
    assert "images" not in plain[0]
    assert "thinking" not in plain[0]


@pytest.mark.asyncio
async def test_chat_stream_surfaces_reasoning_and_tool_calls():
    """Streamed ``delta.reasoning_content`` becomes thinking_delta and
    ``delta.tool_calls`` rides in chunk.augmentum — mirroring
    OpenAICompatBackend's parser so chat_egress treats fabric-routed
    inference identically to local.
    """
    tc = [{"index": 0, "id": "call_1", "type": "function",
           "function": {"name": "web_search", "arguments": "{\"q\":"}}]

    async def fake_aiter_lines():
        yield "data: " + json.dumps({
            "model": "test-model",
            "choices": [{"delta": {"role": "assistant",
                                   "reasoning_content": "let me think"},
                         "finish_reason": None}],
        })
        yield "data: " + json.dumps({
            "model": "test-model",
            "choices": [{"delta": {"content": "Answer"},
                         "finish_reason": None}],
        })
        yield "data: " + json.dumps({
            "model": "test-model",
            "choices": [{"delta": {"tool_calls": tc},
                         "finish_reason": "tool_calls"}],
        })
        yield "data: [DONE]"

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 200
    fake_resp.aiter_lines = fake_aiter_lines

    class _AsyncCM:
        async def __aenter__(self):
            return fake_resp
        async def __aexit__(self, *a, **k):
            return False

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_AsyncCM())

    backend = _make_backend(http_client=fake_client)
    request = InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )
    chunks = [c async for c in backend.chat_stream(request)]

    assert chunks[0].thinking_delta == "let me think"
    assert chunks[1].content_delta == "Answer"
    assert chunks[2].augmentum["tool_calls"] == tc
    assert chunks[2].finish_reason == "tool_calls"
    assert chunks[3].done is True
