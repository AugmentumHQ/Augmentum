"""Tests for Phase 6 knowledge-pack search routing.

Three layers:

  - search_remote_packs: outbound POST shape (signed envelope, body
    integrity, correct path) + error mapping (unreachable, HTTP 4xx).
  - RoutingDirector.fanout_knowledge_search: partition correctness
    (local vs peer subsets), parallel dispatch, failure absorption.
  - inject_pack_context._search_packs: fabric-off path is the pre-
    fabric pack_mgr.search call verbatim (default-off invariant).

The router & injection paths use stub Coordinators + PackManagers
to avoid pulling in the real subsystems (SQLite, embeddings, etc.).
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import httpx
import pytest

from augmentum.fabric.coordinator import FabricCoordinator
from augmentum.fabric.director import RoutingDirector
from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.knowledge_client import (
    RemoteSearchError,
    search_remote_packs,
)
from augmentum.fabric.peer_auth import PairedPeer
from augmentum.state.settings_store import SettingsStore

# ── Helpers ───────────────────────────────────────────────────────


async def _make_identity_db():
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.commit()
    return conn


def _peer(node_id: str, addr: str = "192.168.1.20:6443") -> PairedPeer:
    return PairedPeer(
        node_id=node_id, hostname=f"h-{node_id[:4]}", role="peer",
        pubkey_b64="dGVzdA==", fingerprint=f"SHA256:{node_id[:8]}",
        addr=addr, tier="local", fabric_share_enabled=True,
        paired_at="2026-05-16 00:00:00", last_seen_at=None,
    )


def _ok_search_response(results: list[dict]) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "query": "any", "pack_ids": ["wikipedia_en"], "results": results,
    }
    return resp


# ── search_remote_packs ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_remote_packs_posts_signed_body_to_correct_path():
    """The outbound search MUST be a signed POST to /api/knowledge/search
    with the body covered by sha256 — otherwise the receiver's
    FabricPeerMiddleware rejects it.
    """
    conn = await _make_identity_db()
    try:
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        results_payload = [{
            "content": "a chunk", "title": "t", "section": "s", "url": "",
            "pack_id": "wikipedia_en", "source": "vector", "score": 0.8,
        }]
        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=_ok_search_response(results_payload))

        out = await search_remote_packs(
            http_client=fake_client, identity=identity,
            user_id="user-42", peer_addr="192.168.1.20:6443",
            query="renaissance painting",
            pack_ids=["wikipedia_en", "wikipedia_de"], limit=5,
        )

        # Path is correct.
        call_args = fake_client.post.call_args
        url = call_args[0][0]
        assert url.endswith("/api/knowledge/search")

        # Body is the exact bytes used for signing (content=body).
        body = call_args[1]["content"]
        assert isinstance(body, bytes)
        parsed = json.loads(body)
        assert parsed["q"] == "renaissance painting"
        assert parsed["pack_ids"] == ["wikipedia_en", "wikipedia_de"]
        assert parsed["limit"] == 5

        # Signed headers are present.
        headers = call_args[1]["headers"]
        assert headers["X-Fabric-Sender"] == identity.node_id
        assert headers["X-Fabric-User-Id"] == "user-42"
        assert "X-Fabric-Signature" in headers
        assert "X-Fabric-Timestamp" in headers
        assert headers["Content-Type"] == "application/json"

        assert out == results_payload
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_search_remote_packs_no_pack_ids_short_circuits():
    """An empty pack list should not even issue an HTTP request — the
    receiver would just match zero packs anyway.
    """
    conn = await _make_identity_db()
    try:
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        fake_client = MagicMock()
        fake_client.post = AsyncMock()  # should not be called

        out = await search_remote_packs(
            http_client=fake_client, identity=identity, user_id="u",
            peer_addr="any", query="q", pack_ids=[], limit=5,
        )
        assert out == []
        fake_client.post.assert_not_called()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_search_remote_packs_unreachable_raises_remote_search_error():
    """A connection error becomes RemoteSearchError so the director's
    fanout can absorb it + return empty rather than propagating up
    into the chat-injection path.
    """
    conn = await _make_identity_db()
    try:
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        fake_client = MagicMock()
        fake_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(RemoteSearchError) as excinfo:
            await search_remote_packs(
                http_client=fake_client, identity=identity, user_id="u",
                peer_addr="192.168.1.99:6443", query="q",
                pack_ids=["any"], limit=5,
            )
        assert "unreachable" in str(excinfo.value).lower()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_search_remote_packs_4xx_raises_remote_search_error():
    """Peer returning 401/403/etc. is mapped to RemoteSearchError with
    the response detail, so we never silently treat a rejected
    request as "no results".
    """
    conn = await _make_identity_db()
    try:
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        bad_resp = MagicMock(spec=httpx.Response)
        bad_resp.status_code = 401
        bad_resp.text = "unauthorized"
        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=bad_resp)

        with pytest.raises(RemoteSearchError) as excinfo:
            await search_remote_packs(
                http_client=fake_client, identity=identity, user_id="u",
                peer_addr="192.168.1.20:6443", query="q",
                pack_ids=["any"], limit=5,
            )
        assert "401" in str(excinfo.value)
    finally:
        await conn.close()


# ── RoutingDirector.fanout_knowledge_search ───────────────────────


async def _make_director_with_peer(pack_id_on_peer: str):
    """Coordinator + director with one connected peer advertising one pack."""
    conn = await _make_identity_db()
    identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
    coord = FabricCoordinator(identity, conn)
    await coord.register_paired_peer(_peer("peer-1"))
    ws = MagicMock()
    ws.close = AsyncMock()
    await coord.attach_connection("peer-1", ws)
    coord.record_remote_capabilities("peer-1", [{
        "kind": "knowledge.search", "schema_version": 1,
        "pack_id": pack_id_on_peer, "pack_name": pack_id_on_peer,
        "chunk_count": 1000, "embedding_dim": 384,
        "active": True, "pack_format": "augpack",
    }])
    fake_http = MagicMock()
    director = RoutingDirector(coord, fake_http)
    return conn, director, fake_http


@pytest.mark.asyncio
async def test_fanout_partitions_local_vs_peer_correctly():
    """A mix of local + peer-only packs goes to the right places:
    pack_mgr gets only the local subset, peers get only their subset.
    """
    conn, director, fake_http = await _make_director_with_peer("wikipedia_en")
    try:
        # Stub the peer response.
        peer_results = [{"content": "from peer", "title": "t",
                         "section": "", "url": "", "pack_id": "wikipedia_en",
                         "source": "vector", "score": 0.9}]
        fake_http.post = AsyncMock(return_value=_ok_search_response(peer_results))

        # Local search receives only the local pack ID.
        local_calls = []
        async def local_search_fn(pack_ids):
            local_calls.append(list(pack_ids))
            return ["LOCAL_RESULT"]  # stub object — director doesn't introspect

        merged = await director.fanout_knowledge_search(
            query="q", requested_pack_ids=["local_pack", "wikipedia_en"],
            local_pack_ids={"local_pack"}, user_id="u", limit=5,
            local_search_fn=local_search_fn,
        )

        # Local subset routed correctly.
        assert local_calls == [["local_pack"]]
        # Peer subset routed correctly (URL contains the peer addr).
        assert fake_http.post.call_count == 1
        peer_url = fake_http.post.call_args[0][0]
        assert "192.168.1.20:6443" in peer_url
        # Merged result contains both legs.
        assert "LOCAL_RESULT" in merged
        assert peer_results[0] in merged
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_fanout_no_peer_match_skips_remote_dispatch():
    """A requested pack that nobody has should produce no remote calls
    AND not appear in the local subset either (PackManager already
    silently drops unknown packs, but the director shouldn't waste
    a network round-trip).
    """
    conn, director, fake_http = await _make_director_with_peer("wikipedia_en")
    try:
        fake_http.post = AsyncMock()

        local_calls = []
        async def local_search_fn(pack_ids):
            local_calls.append(list(pack_ids))
            return []

        await director.fanout_knowledge_search(
            query="q",
            requested_pack_ids=["nonexistent_pack"],
            local_pack_ids={"local_pack"},  # we have local_pack, but it wasn't requested
            user_id="u", limit=5, local_search_fn=local_search_fn,
        )

        assert local_calls == []  # no local search either
        fake_http.post.assert_not_called()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_fanout_absorbs_peer_failure_returns_local_results():
    """If a peer is unreachable, the director MUST NOT propagate the
    exception — chat must not break just because a peer is down.
    Local results still come back; peer leg yields empty.
    """
    conn, director, fake_http = await _make_director_with_peer("wikipedia_en")
    try:
        fake_http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        async def local_search_fn(pack_ids):
            return ["LOCAL_OK"]

        merged = await director.fanout_knowledge_search(
            query="q",
            requested_pack_ids=["local_pack", "wikipedia_en"],
            local_pack_ids={"local_pack"}, user_id="u", limit=5,
            local_search_fn=local_search_fn,
        )

        # Local result survives; peer's exception was absorbed.
        assert "LOCAL_OK" in merged
        assert len(merged) == 1
    finally:
        await conn.close()


# ── _search_packs (the injection.py shim) ─────────────────────────


@pytest.mark.asyncio
async def test_search_packs_fabric_off_is_passthrough():
    """When app_state.fabric_director is None, _search_packs must call
    pack_mgr.search verbatim with the original args (incl. rerank).
    This is the default-off invariant for Phase 6.
    """
    from augmentum.knowledge.injection import _search_packs

    pack_mgr = MagicMock()
    pack_mgr.search = AsyncMock(return_value=["ORIGINAL_RESULTS"])
    pack_mgr.installed = MagicMock(return_value=[{"pack_id": "x"}])

    app_state = MagicMock()
    app_state.fabric_director = None

    out = await _search_packs(
        pack_mgr=pack_mgr, app_state=app_state, user_id="u",
        query="q", pack_ids=["x"], limit=5,
    )

    assert out == ["ORIGINAL_RESULTS"]
    # Called exactly once, with the original rerank pass-through.
    pack_mgr.search.assert_awaited_once()
    kwargs = pack_mgr.search.await_args.kwargs
    assert kwargs["pack_ids"] == ["x"]
    assert kwargs["limit"] == 5
    assert "rerank" in kwargs  # original rerank semantics preserved


@pytest.mark.asyncio
async def test_search_packs_all_local_skips_fanout():
    """Even with the director present, if all requested packs are local
    we take the simple path. Avoids unnecessary fanout when fabric is
    on but no peer involvement is needed for THIS query.
    """
    from augmentum.knowledge.injection import _search_packs

    pack_mgr = MagicMock()
    pack_mgr.search = AsyncMock(return_value=["ORIGINAL_RESULTS"])
    pack_mgr.installed = MagicMock(return_value=[
        {"pack_id": "wiki"}, {"pack_id": "devdocs"},
    ])

    director = MagicMock()
    director.fanout_knowledge_search = AsyncMock()
    app_state = MagicMock()
    app_state.fabric_director = director

    out = await _search_packs(
        pack_mgr=pack_mgr, app_state=app_state, user_id="u",
        query="q", pack_ids=["wiki", "devdocs"], limit=5,
    )

    assert out == ["ORIGINAL_RESULTS"]
    # The fanout was NOT called — we took the simple path.
    director.fanout_knowledge_search.assert_not_called()
