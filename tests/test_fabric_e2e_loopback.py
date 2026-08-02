"""End-to-end fabric loopback tests.

Validates the FULL cross-peer pipeline by wiring two coordinators
+ a peer-side ASGI app + an httpx AsyncClient over ASGITransport,
so the FabricBackend's HTTPS calls actually traverse a real
FastAPI request/response cycle through FabricPeerMiddleware.

The chains under test (matched to the Phase 9.1 audit):

  - Auth + body-integrity round-trip: signed envelope from A
    arrives at B, FabricPeerMiddleware verifies, downstream handler
    sees ``scope["user"]`` populated.
  - Streaming chunks: B emits SSE; A's FabricBackend parses + yields
    InternalStreamChunks in order.
  - Lifecycle events: B's middleware emits MSG_JOB_STARTED /
    MSG_JOB_COMPLETED back over A's coordinator WS; A's coordinator
    records them in _peer_call_events.
  - Cancellation backstop: when A cancels mid-stream, A's
    FabricBackend pushes MSG_CANCEL_REQUEST via coordinator.

These are integration tests; they spin up actual ASGI apps in-process
+ exchange real HTTP envelopes. No external services.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from augmentum.fabric.coordinator import FabricCoordinator
from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.peer_auth import persist_remote_node
from augmentum.fabric.peer_middleware import FabricPeerMiddleware
from augmentum.models.base import (
    InternalChatRequest,
    Message,
)
from augmentum.models.fabric_backend import FabricBackend
from augmentum.state.settings_store import SettingsStore

# ── Fixture: paired two-peer fabric ────────────────────────────────


async def _make_db_with_fabric_nodes() -> aiosqlite.Connection:
    """SQLite memory DB with the schema both coordinators need."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.execute(
        """CREATE TABLE fabric_nodes (
            id TEXT PRIMARY KEY, hostname TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT 'peer',
            pubkey_ed25519 TEXT NOT NULL, pubkey_fingerprint TEXT NOT NULL,
            addr TEXT NOT NULL DEFAULT '', tier TEXT NOT NULL DEFAULT 'local',
            fabric_share_enabled INTEGER NOT NULL DEFAULT 1,
            paired_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_seen_at TEXT, icon TEXT NOT NULL DEFAULT '')"""
    )
    await conn.commit()
    return conn


class _StubUser:
    """Duck-typed augmentum.auth.session_manager.User. The
    middleware reads .id and .is_active off whatever scope["user"]
    becomes; the downstream handler in this test just checks
    presence.
    """
    def __init__(self, user_id: str = "test-user"):
        self.id = user_id
        self.is_active = True


class _StubSessionManager:
    """Minimal session_manager that always returns the same stub user.

    The middleware provisions a per-peer service user from the signed
    envelope (``get_or_create_fabric_peer_user``); older revisions looked
    the claimed user up by id. Stub both so the test tracks whichever path
    the middleware takes.
    """
    def __init__(self, user: _StubUser):
        self._user = user

    async def get_user_by_id(self, _uid: str):
        return self._user

    async def get_or_create_fabric_peer_user(
        self, _sender_node_id: str, *, hostname: str = "",
    ):
        return self._user


async def _peer_b_setup(stream_chunks: list[str]):
    """Set up the receiving peer's full stack: DB, coordinator,
    FastAPI app with FabricPeerMiddleware + a mock chat handler
    that yields ``stream_chunks`` as SSE.

    Returns (conn_b, coord_b, identity_b, app_b) so tests can
    inspect state + drive requests through the app.
    """
    conn_b = await _make_db_with_fabric_nodes()
    identity_b = await FabricIdentity.from_settings_store(SettingsStore(conn_b))
    coord_b = FabricCoordinator(identity_b, conn_b)

    app_b = FastAPI()
    app_b.state.fabric_coordinator = coord_b
    # Stub state_manager wrapper for the middleware's DB access path.
    sm_wrap = MagicMock()
    sm_wrap.backend = MagicMock()
    sm_wrap.backend.conn = conn_b
    app_b.state.state_manager = sm_wrap
    app_b.state.session_manager = _StubSessionManager(_StubUser("test-user"))

    chat_call_log: list[dict] = []

    @app_b.post("/v1/chat/completions")
    async def _chat_handler(request: Request):
        body = await request.json()
        chat_call_log.append({
            "body": body,
            "user_id": request.scope["user"].id if request.scope.get("user") else None,
            "fabric_peer": request.scope.get("fabric_peer"),
            "request_id": request.headers.get("x-fabric-request-id", ""),
        })

        async def _gen():
            for chunk in stream_chunks:
                yield chunk

        return StreamingResponse(_gen(), media_type="text/event-stream")

    # FabricPeerMiddleware wraps the app (auth + register-inflight +
    # lifecycle event emission).
    app_b.add_middleware(FabricPeerMiddleware)

    return conn_b, coord_b, identity_b, app_b, chat_call_log


# ── Test 1: auth + streaming chunks end-to-end ────────────────────


@pytest.mark.asyncio
async def test_e2e_signed_request_authenticates_and_streams():
    """Happy path: A signs a request, B's middleware verifies,
    B's handler streams chunks, A's FabricBackend parses them
    back into InternalStreamChunks. Validates Links 1-3 from
    the Phase 9.1 audit.
    """
    # Sample SSE stream — three content deltas + DONE.
    chunks = [
        f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant'}}]})}\n\n",
        f"data: {json.dumps({'choices': [{'delta': {'content': 'Hello '}}]})}\n\n",
        f"data: {json.dumps({'choices': [{'delta': {'content': 'world'}, 'finish_reason': 'stop'}]})}\n\n",
        "data: [DONE]\n\n",
    ]

    conn_a = await _make_db_with_fabric_nodes()
    conn_b, coord_b, identity_b, app_b, chat_log = await _peer_b_setup(chunks)
    try:
        identity_a = await FabricIdentity.from_settings_store(SettingsStore(conn_a))
        coord_a = FabricCoordinator(identity_a, conn_a)

        # Pair: B's identity persisted into A's fabric_nodes (so A
        # would know about B for routing); A's identity persisted
        # into B's fabric_nodes (so B's middleware can verify A's
        # signature via lookup_peer_pubkey).
        await persist_remote_node(
            conn_a, node_id=identity_b.node_id, hostname="b",
            role="peer", pubkey_b64=identity_b.public_key_b64,
            addr="b:6443",
        )
        await persist_remote_node(
            conn_b, node_id=identity_a.node_id, hostname="a",
            role="peer", pubkey_b64=identity_a.public_key_b64,
            addr="a:6443",
        )

        # Build an httpx AsyncClient that routes to B's ASGI app.
        # base_url is what FabricBackend's URL construction targets.
        transport = httpx.ASGITransport(app=app_b)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://b:6443",
        ) as client:
            from augmentum.fabric.capabilities import LLMInferenceCapability
            cap = LLMInferenceCapability(model_id="m")
            backend = FabricBackend(
                http_client=client, peer_node_id=identity_b.node_id,
                peer_addr="b:6443", advertised_capability=cap,
                identity=identity_a, user_id="test-user",
                coordinator=coord_a,
            )
            request = InternalChatRequest(
                model="m", messages=[Message(role="user", content="hi")],
            )
            received = []
            async for chunk in backend.chat_stream(request):
                received.append(chunk)

        # Chunks parsed correctly + in order.
        assert len(received) >= 3
        # Find content deltas (the 'role' chunk may or may not be
        # surfaced depending on parser shape).
        content = "".join(c.content_delta for c in received if c.content_delta)
        assert content == "Hello world"

        # B's handler ran exactly once + saw the authenticated user.
        assert len(chat_log) == 1
        assert chat_log[0]["user_id"] == "test-user"
        assert chat_log[0]["fabric_peer"]["sender_node_id"] == identity_a.node_id
        # Body was deserialized correctly through the body-integrity
        # buffer-replay path.
        assert chat_log[0]["body"]["model"] == "m"
        assert chat_log[0]["body"]["messages"][0]["content"] == "hi"
        # Request-id was forwarded for cancellation correlation.
        assert chat_log[0]["request_id"].startswith("req-")
    finally:
        await conn_a.close()
        await conn_b.close()


# ── Test 2: lifecycle events arrive at originator's coordinator ───


@pytest.mark.asyncio
async def test_e2e_lifecycle_events_emitted_to_originator():
    """When B's middleware processes a fabric request, it pushes
    job_started + job_completed envelopes back over A's coordinator
    WS. To verify this without a real WS, stub coord_b.send_to_peer
    + assert the envelopes that WOULD have been sent.

    (The actual WS round-trip is exercised in the coordinator
    dispatcher unit tests; this proves the middleware INITIATES
    the events at the right moments.)
    """
    chunks = [
        f"data: {json.dumps({'choices': [{'delta': {'content': 'x'}, 'finish_reason': 'stop'}]})}\n\n",
        "data: [DONE]\n\n",
    ]

    conn_a = await _make_db_with_fabric_nodes()
    conn_b, coord_b, identity_b, app_b, _ = await _peer_b_setup(chunks)
    try:
        # Intercept B's coordinator outbound — the middleware uses
        # this to push lifecycle events back to A.
        coord_b.send_to_peer = AsyncMock(return_value=True)

        identity_a = await FabricIdentity.from_settings_store(SettingsStore(conn_a))
        coord_a = FabricCoordinator(identity_a, conn_a)

        await persist_remote_node(
            conn_a, node_id=identity_b.node_id, hostname="b",
            role="peer", pubkey_b64=identity_b.public_key_b64,
            addr="b:6443",
        )
        await persist_remote_node(
            conn_b, node_id=identity_a.node_id, hostname="a",
            role="peer", pubkey_b64=identity_a.public_key_b64,
            addr="a:6443",
        )

        transport = httpx.ASGITransport(app=app_b)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://b:6443",
        ) as client:
            from augmentum.fabric.capabilities import LLMInferenceCapability
            cap = LLMInferenceCapability(model_id="m")
            backend = FabricBackend(
                http_client=client, peer_node_id=identity_b.node_id,
                peer_addr="b:6443", advertised_capability=cap,
                identity=identity_a, user_id="test-user",
                coordinator=coord_a,
            )
            request = InternalChatRequest(
                model="m", messages=[Message(role="user", content="hi")],
            )
            async for _ in backend.chat_stream(request):
                pass

        # coord_b.send_to_peer was called at least twice: once for
        # job_started (right after auth) + once for job_completed
        # (after the handler returns cleanly).
        msg_types = [call.kwargs["msg_type"]
                     for call in coord_b.send_to_peer.call_args_list]
        assert "job_started" in msg_types
        assert "job_completed" in msg_types
        # All envelopes target A (the originator).
        for call in coord_b.send_to_peer.call_args_list:
            assert call.args[0] == identity_a.node_id
    finally:
        await conn_a.close()
        await conn_b.close()


# ── Note: cancellation E2E coverage ──────────────────────────────
#
# A third test for "cancel mid-stream + verify backstop fires" was
# initially attempted here but httpx.ASGITransport buffers the full
# response body before returning it (it's not a true streaming
# transport — see httpx#2492). That means we can't observe a partial
# stream mid-flight under ASGITransport, which is the exact moment
# the cancellation backstop needs to fire.
#
# Coverage for that path is instead achieved via:
#   - test_chat_stream_cancel_sends_ws_backstop in test_fabric_backend
#     (unit, with a hanging async generator stub — proves the
#     backstop path in FabricBackend.chat_stream fires correctly
#     on caller-cancel)
#   - test_handle_inbound_envelope_dispatches_cancel in
#     test_fabric_coordinator (unit, with a real signed envelope
#     and a registered task — proves the receiver-side path looks
#     up the right in-flight task and cancels it)
#
# A true E2E test would need uvicorn-on-a-port (heavier infra
# than ASGITransport). The unit tests cover both halves of the
# chain; the wire format between them is exercised by Phase 9's
# protocol tests.
