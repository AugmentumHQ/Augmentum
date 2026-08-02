"""Tests for the RoutingDirector.

Critical invariants:

  - Local-first: when local can serve, NEVER return a peer. This is
    the load-bearing assertion. A bug here silently routes every
    request across the network.
  - No-peer-fallback: when no peer advertises the model AND local
    can't serve, return None (stay local; upstream raises a clean
    error).
  - Connected-only: an offline peer with the model still isn't
    selected.
  - No infinite recursion: when the supplied local_backend is itself
    a FabricBackend, maybe_route_llm returns None (treats it as
    "already routing").
"""
from __future__ import annotations

from unittest.mock import MagicMock

import aiosqlite
import httpx
import pytest

from augmentum.fabric.capabilities import (
    LLMInferenceCapability,
    serialise,
)
from augmentum.fabric.coordinator import FabricCoordinator
from augmentum.fabric.director import RoutingDirector
from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.peer_auth import PairedPeer
from augmentum.models.fabric_backend import FabricBackend
from augmentum.state.settings_store import SettingsStore


async def _make_env() -> tuple[aiosqlite.Connection, FabricCoordinator, RoutingDirector]:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings ("
        "  key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')))"
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
    identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
    coord = FabricCoordinator(identity, conn)
    http = httpx.AsyncClient()
    director = RoutingDirector(coord, http)
    return conn, coord, director


def _peer(node_id: str, addr: str = "192.168.1.10:6443") -> PairedPeer:
    return PairedPeer(
        node_id=node_id, hostname=f"h-{node_id[:4]}", role="peer",
        pubkey_b64="dGVzdA==", fingerprint=f"SHA256:{node_id[:8]}",
        addr=addr, tier="local",
        fabric_share_enabled=True, paired_at="2026-05-16 00:00:00",
        last_seen_at=None,
    )


class _FakeWebSocket:
    def __init__(self):
        self.closed = False
    async def close(self, code=1000, reason=""):
        self.closed = True


@pytest.mark.asyncio
async def test_local_first_when_local_can_serve():
    """The single most-important assertion: when local serves, we
    NEVER override to a peer. Even when peers advertise the same
    model.
    """
    conn, coord, director = await _make_env()
    try:
        # A peer is connected AND advertises the model we want.
        await coord.register_paired_peer(_peer("peer-1"))
        await coord.attach_connection("peer-1", _FakeWebSocket())
        coord.record_remote_capabilities("peer-1", [
            serialise(LLMInferenceCapability(model_id="Qwen3.5-72B", model_family="qwen3")),
        ])

        # Local backend (anything non-FabricBackend) claims to serve.
        local_backend = MagicMock()
        result = await director.maybe_route_llm(
            model="Qwen3.5-72B", user_id="u1", session_id="s1",
            local_backend=local_backend,
        )
        # Director MUST return None: local-first.
        assert result is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_no_peer_returns_none_even_when_local_cant_serve():
    """When no peer matches AND local can't serve, return None.
    Don't try to mask the failure; let the upstream error reach
    the user.
    """
    conn, coord, director = await _make_env()
    try:
        # No peer paired at all.
        local = MagicMock()

        # local_known=False signals "the resolver fell back to default
        # because the local map doesn't have this model" -- without it
        # the director honours local-first and never even tries peers.
        result = await director.maybe_route_llm(
            model="some-model", user_id="u1", session_id="s1",
            local_backend=local, local_known=False,
        )
        assert result is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_routes_to_peer_when_local_cant_and_peer_can():
    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("peer-x"))
        await coord.attach_connection("peer-x", _FakeWebSocket())
        coord.record_remote_capabilities("peer-x", [
            serialise(LLMInferenceCapability(model_id="qwen3-x", model_family="qwen3")),
        ])

        local = MagicMock()
        result = await director.maybe_route_llm(
            model="qwen3-x", user_id="u1", session_id="s1",
            local_backend=local, local_known=False,
        )
        # Got a FabricBackend pointing at peer-x.
        assert isinstance(result, FabricBackend)
        assert result._peer_node_id == "peer-x"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_does_not_route_to_offline_peer():
    """A peer that's paired but disconnected shouldn't be selected,
    even if they advertised the model before going offline.
    """
    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("dark-peer"))
        # NB: NO attach_connection() call -- peer is paired but offline.
        coord.record_remote_capabilities("dark-peer", [
            serialise(LLMInferenceCapability(model_id="model-z")),
        ])

        local = MagicMock()
        result = await director.maybe_route_llm(
            model="model-z", user_id="u1", session_id="s1",
            local_backend=local, local_known=False,
        )
        # No connected peer has it, so director declines to route.
        assert result is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_does_not_recurse_when_backend_is_already_fabric():
    """If a request is somehow being re-dispatched and ``local_backend``
    is already a FabricBackend, the director should NOT try to route
    again. Otherwise we'd ping-pong infinitely between peers.
    """
    conn, coord, director = await _make_env()
    try:
        # Mock FabricBackend (don't need a real peer for this).
        fake_fabric = MagicMock(spec=FabricBackend)
        result = await director.maybe_route_llm(
            model="anything", user_id="u1", session_id="s1",
            local_backend=fake_fabric,
        )
        # Treats existing FabricBackend as "already routing" -> stay local.
        assert result is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_handles_model_with_no_peer_match():
    """Multiple peers advertise different models; none matches the
    requested model. Director returns None.
    """
    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("p1"))
        await coord.register_paired_peer(_peer("p2"))
        await coord.attach_connection("p1", _FakeWebSocket())
        await coord.attach_connection("p2", _FakeWebSocket())
        coord.record_remote_capabilities("p1", [
            serialise(LLMInferenceCapability(model_id="model-A")),
        ])
        coord.record_remote_capabilities("p2", [
            serialise(LLMInferenceCapability(model_id="model-B")),
        ])

        local = MagicMock()
        result = await director.maybe_route_llm(
            model="model-NOT-AVAILABLE", user_id="u1", session_id="s1",
            local_backend=local, local_known=False,
        )
        assert result is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_peer_diagnostic_for_llm_surfaces_connection_state():
    """peer_diagnostic_for_llm collects the structured state the
    resolver attaches to ModelUnavailableError. Two peers paired —
    one connected with the wanted model, one offline — the diagnostic
    must distinguish both and flag which advertises the wanted model.
    """
    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("online-peer"))
        await coord.register_paired_peer(_peer("offline-peer"))
        await coord.attach_connection("online-peer", _FakeWebSocket())
        # offline-peer paired but NOT attached.
        coord.record_remote_capabilities("online-peer", [
            serialise(LLMInferenceCapability(model_id="wanted")),
        ])
        coord.record_remote_capabilities("offline-peer", [
            serialise(LLMInferenceCapability(model_id="something-else")),
        ])

        diag = director.peer_diagnostic_for_llm("wanted")
        assert diag["wanted_model"] == "wanted"
        assert "online-peer" in diag["connected_peers"]
        assert "offline-peer" in diag["offline_peers"]
        assert diag["peers"]["online-peer"]["advertises_wanted"] is True
        assert diag["peers"]["online-peer"]["connected"] is True
        assert diag["peers"]["offline-peer"]["advertises_wanted"] is False
        assert diag["peers"]["offline-peer"]["connected"] is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_resolver_raises_model_unavailable_when_no_peer_has_model():
    """End-to-end through the resolver helper: when the model isn't
    in the local map AND no connected peer advertises it, raise
    ModelUnavailableError with the peer diagnostic attached. Pre-fix
    this returned (default_backend, model_name) silently, then the
    default backend errored confusingly during dispatch.

    Bypasses the real ProviderRegistry by stubbing resolve_backend_for_model
    so the test stays focused on the fabric resolver logic without needing
    a full backend registry setup.
    """
    from augmentum.models.provider_registry import (
        ModelUnavailableError,
        ProviderRegistry,
    )

    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("p"))
        await coord.attach_connection("p", _FakeWebSocket())
        coord.record_remote_capabilities("p", [
            serialise(LLMInferenceCapability(model_id="some-other-model")),
        ])

        # Construct a minimal registry shell. ``_init_backends`` runs in
        # __init__ but we override resolve_backend_for_model to bypass
        # the real lookup chain. Model map stays empty → local_known=False.
        registry = ProviderRegistry.__new__(ProviderRegistry)
        registry._backends = {}
        registry._model_map = {}
        registry._fabric_director = director

        async def _stub_resolve(model_name):
            # Mirror the buggy real behaviour: silently hand back a stub
            # backend with the original model name (this is what the
            # ModelUnavailableError exists to protect against).
            return (MagicMock(), model_name)

        registry.resolve_backend_for_model = _stub_resolve

        with pytest.raises(ModelUnavailableError) as ei:
            await registry.resolve_backend_with_fabric(
                "peer-only-mystery-model", user_id="u", session_id="s",
            )
        exc = ei.value
        assert exc.model == "peer-only-mystery-model"
        assert "p" in exc.peer_diagnostic["connected_peers"]
        assert exc.peer_diagnostic["wanted_model"] == "peer-only-mystery-model"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_resolver_uses_manager_loaded_model_when_map_misses():
    """Probe-bypass safety net: if ``list_models()`` was flaky during the
    last model-map refresh, the requested model may be missing from
    ``_model_map`` even though llama-server has it actively loaded.
    The resolver consults each backend's ``_manager.model_id`` as the
    authoritative answer so the user isn't locked out for the TTL
    window after every UI refresh.
    """
    from augmentum.models.provider_registry import (
        ModelUnavailableError,
        ProviderRegistry,
    )

    conn, coord, director = await _make_env()
    try:
        # A connected peer exists but doesn't serve the requested model.
        # Pre-fix this would raise ModelUnavailableError because the model
        # isn't in `_model_map` either.
        await coord.register_paired_peer(_peer("p"))
        await coord.attach_connection("p", _FakeWebSocket())
        coord.record_remote_capabilities("p", [
            serialise(LLMInferenceCapability(model_id="some-other-model")),
        ])

        registry = ProviderRegistry.__new__(ProviderRegistry)
        local_backend = MagicMock()
        local_backend._manager = MagicMock()
        local_backend._manager.model_id = "rocinante-xl-16b"
        registry._backends = {"llamacpp": local_backend}
        registry._model_map = {}  # probe failed / hadn't completed yet
        registry._fabric_director = director

        async def _stub_resolve(model_name):
            return (local_backend, model_name)

        registry.resolve_backend_for_model = _stub_resolve

        # MUST NOT raise — manager reports the model is loaded.
        backend, clean_model = await registry.resolve_backend_with_fabric(
            "rocinante-xl-16b", user_id="u", session_id="s",
        )
        assert backend is local_backend
        assert clean_model == "rocinante-xl-16b"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_resolver_manager_bypass_only_matches_exact_model_id():
    """The manager-bypass must NOT match when the loaded model is
    different from the requested one — otherwise a request for an
    unknown model would silently route to whatever is loaded.
    """
    from augmentum.models.provider_registry import (
        ModelUnavailableError,
        ProviderRegistry,
    )

    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("p"))
        await coord.attach_connection("p", _FakeWebSocket())
        coord.record_remote_capabilities("p", [
            serialise(LLMInferenceCapability(model_id="some-other-model")),
        ])

        registry = ProviderRegistry.__new__(ProviderRegistry)
        local_backend = MagicMock()
        local_backend._manager = MagicMock()
        local_backend._manager.model_id = "rocinante-xl-16b"  # loaded
        registry._backends = {"llamacpp": local_backend}
        registry._model_map = {}
        registry._fabric_director = director

        async def _stub_resolve(model_name):
            return (local_backend, model_name)

        registry.resolve_backend_for_model = _stub_resolve

        # Request a DIFFERENT model — should raise, not silently route.
        with pytest.raises(ModelUnavailableError):
            await registry.resolve_backend_with_fabric(
                "totally-different-model", user_id="u", session_id="s",
            )
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_resolver_does_not_raise_for_empty_model_name():
    """Empty model_name is the role-resolver "use the default first model"
    idiom (resolve_model_for_role chain). Must NOT trigger
    ModelUnavailableError even when local has no models — the caller
    expects a graceful fallback to handle itself.
    """
    from augmentum.models.provider_registry import ProviderRegistry

    conn, coord, director = await _make_env()
    try:
        registry = ProviderRegistry.__new__(ProviderRegistry)
        registry._backends = {}
        registry._model_map = {}
        registry._fabric_director = director

        stub_backend = MagicMock()

        async def _stub_resolve(model_name):
            return (stub_backend, model_name)

        registry.resolve_backend_for_model = _stub_resolve

        # Should NOT raise — empty model name is a sentinel for "default".
        backend, clean_model = await registry.resolve_backend_with_fabric(
            "", user_id="u", session_id="s",
        )
        assert clean_model == ""
        assert backend is stub_backend
    finally:
        await conn.close()


# ── Phase 8.x: operator-pinned peer dispatch ───────────────────────────


def test_parse_fabric_pin_strips_suffix():
    from augmentum.models.provider_registry import _parse_fabric_pin

    clean, pin = _parse_fabric_pin("Llama-3.3-70B@fabric:abc12345")
    assert clean == "Llama-3.3-70B"
    assert pin == "abc12345"


def test_parse_fabric_pin_no_suffix_returns_passthrough():
    from augmentum.models.provider_registry import _parse_fabric_pin

    clean, pin = _parse_fabric_pin("Llama-3.3-70B")
    assert clean == "Llama-3.3-70B"
    assert pin == ""


def test_parse_fabric_pin_ignores_backend_suffix():
    """The existing @<backend_key> convention (e.g., @ollama) must not
    be misread as a fabric pin. Backend keys never contain colons."""
    from augmentum.models.provider_registry import _parse_fabric_pin

    clean, pin = _parse_fabric_pin("Llama-3.3-70B@ollama")
    assert clean == "Llama-3.3-70B@ollama"
    assert pin == ""


def test_parse_fabric_pin_malformed_treated_as_unpinned():
    from augmentum.models.provider_registry import _parse_fabric_pin

    # Empty suffix
    clean, pin = _parse_fabric_pin("Llama-3.3-70B@fabric:")
    assert pin == ""
    # Empty model
    clean, pin = _parse_fabric_pin("@fabric:abc12")
    assert pin == ""


@pytest.mark.asyncio
async def test_route_llm_to_pinned_peer_resolves_by_prefix():
    """Pinned dispatch resolves a short node_id prefix to the unique
    connected peer and returns a FabricBackend bound to it.
    """
    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("abc123def456"))
        await coord.attach_connection("abc123def456", _FakeWebSocket())
        coord.record_remote_capabilities("abc123def456", [
            serialise(LLMInferenceCapability(model_id="qwen3-72b")),
        ])

        result = await director.route_llm_to_pinned_peer(
            model="qwen3-72b", peer_id_prefix="abc123def456",
            user_id="u", session_id="s",
        )
        assert isinstance(result, FabricBackend)
        assert result._peer_node_id == "abc123def456"
        # Response stamping carries the pinned wire name back to the
        # chat renderer / model_used persistence.
        assert result._pinned_wire_name == "qwen3-72b@fabric:abc123def456"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_route_llm_to_pinned_peer_hard_fails_when_offline():
    """Pin to a paired-but-disconnected peer → None. Resolver translates
    that into a ModelUnavailableError ("don't auto-fall-back when the
    operator picked specifically that box").
    """
    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("offline-peer1"))
        # NB: NO attach_connection — peer is paired but disconnected.
        coord.record_remote_capabilities("offline-peer1", [
            serialise(LLMInferenceCapability(model_id="qwen3-72b")),
        ])

        result = await director.route_llm_to_pinned_peer(
            model="qwen3-72b", peer_id_prefix="offline-peer1",
            user_id="u", session_id="s",
        )
        assert result is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_route_llm_to_pinned_peer_hard_fails_when_no_longer_serving():
    """Operator pinned to a peer that *was* serving the model when the
    dropdown was emitted, but the peer's heartbeat dropped that
    capability before dispatch. Return None — don't 404 the operator
    via a peer call.
    """
    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("abc123def456"))
        await coord.attach_connection("abc123def456", _FakeWebSocket())
        coord.record_remote_capabilities("abc123def456", [
            serialise(LLMInferenceCapability(model_id="different-model")),
        ])

        result = await director.route_llm_to_pinned_peer(
            model="qwen3-72b", peer_id_prefix="abc123def456",
            user_id="u", session_id="s",
        )
        assert result is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_route_llm_to_pinned_peer_rejects_ambiguous_prefix():
    """If two connected peers share the same prefix, the pin is
    ambiguous → None. (Won't happen in practice with 12-char prefixes,
    but defensive coverage so a future shortening can't silently
    misroute.)
    """
    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("abc1"))
        await coord.register_paired_peer(_peer("abc2"))
        await coord.attach_connection("abc1", _FakeWebSocket())
        await coord.attach_connection("abc2", _FakeWebSocket())
        coord.record_remote_capabilities("abc1", [
            serialise(LLMInferenceCapability(model_id="m")),
        ])
        coord.record_remote_capabilities("abc2", [
            serialise(LLMInferenceCapability(model_id="m")),
        ])

        # Prefix "abc" matches both -- ambiguous.
        result = await director.route_llm_to_pinned_peer(
            model="m", peer_id_prefix="abc",
            user_id="u", session_id="s",
        )
        assert result is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_resolver_routes_pinned_dispatch_to_chosen_peer():
    """Resolver detects the @fabric: suffix and bypasses the scoring
    path even when local would have served. Hard contract:
    pinned-by-operator means "go to that box."
    """
    from augmentum.models.provider_registry import ProviderRegistry

    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("xyz789abc012"))
        await coord.attach_connection("xyz789abc012", _FakeWebSocket())
        coord.record_remote_capabilities("xyz789abc012", [
            serialise(LLMInferenceCapability(model_id="qwen3-72b")),
        ])

        registry = ProviderRegistry.__new__(ProviderRegistry)
        registry._backends = {}
        registry._model_map = {"qwen3-72b": "fake-local"}  # even local has it
        registry._fabric_director = director

        # If the resolver fell back to the unpinned path, local_known
        # would be True and the director would stay local. The fact
        # that we return a FabricBackend proves the pin won.
        backend, clean_model = await registry.resolve_backend_with_fabric(
            "qwen3-72b@fabric:xyz789abc012", user_id="u", session_id="s",
        )
        assert isinstance(backend, FabricBackend)
        assert backend._peer_node_id == "xyz789abc012"
        assert clean_model == "qwen3-72b"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_resolver_pinned_offline_raises_model_unavailable():
    """The user-visible "pinned peer is offline" hard-fail. Operator
    sees the typed error, not a generic dispatch failure.
    """
    from augmentum.models.provider_registry import (
        ModelUnavailableError,
        ProviderRegistry,
    )

    conn, coord, director = await _make_env()
    try:
        await coord.register_paired_peer(_peer("offlineXYZ123"))
        # Paired but not connected.
        coord.record_remote_capabilities("offlineXYZ123", [
            serialise(LLMInferenceCapability(model_id="qwen3-72b")),
        ])

        registry = ProviderRegistry.__new__(ProviderRegistry)
        registry._backends = {}
        registry._model_map = {}
        registry._fabric_director = director

        with pytest.raises(ModelUnavailableError) as ei:
            await registry.resolve_backend_with_fabric(
                "qwen3-72b@fabric:offlineXYZ12", user_id="u", session_id="s",
            )
        # Error carries the clean model + the peer diagnostic.
        assert ei.value.model == "qwen3-72b"
    finally:
        await conn.close()
