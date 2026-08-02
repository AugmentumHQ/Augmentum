"""Connect instance-identity resolution — Phase 0 of the comms platform.

Covers the real instance handle (derived from settings / public host),
its sanitisation, and that DID resolution treats both the configured
handle AND the legacy ``this-instance`` sentinel as local (back-compat).
"""

from __future__ import annotations

import pytest

from augmentum.config import settings
from augmentum.connect import contacts
from augmentum.connect.contacts import (
    THIS_INSTANCE_SENTINEL,
    _sanitize_handle,
    instance_handle,
    is_local_instance,
    local_did_for,
    resolve_peer_did,
)


@pytest.fixture
def handle(monkeypatch):
    """Set connect_instance_handle (and clear public host) for the test."""

    def _set(value: str, *, public_host: str = "") -> str:
        monkeypatch.setattr(settings, "connect_instance_handle", value, raising=False)
        monkeypatch.setattr(settings, "augmentum_public_host", public_host, raising=False)
        return instance_handle()

    return _set


class TestSanitizeHandle:
    def test_strips_scheme_port_and_path(self) -> None:
        assert _sanitize_handle("https://myhost.ts.net:6443/connect") == "myhost.ts.net"

    def test_lowercases(self) -> None:
        assert _sanitize_handle("MyHost.Example.COM") == "myhost.example.com"

    def test_keeps_bare_ip(self) -> None:
        assert _sanitize_handle("192.168.1.10:6443") == "192.168.1.10"

    def test_empty_returns_empty(self) -> None:
        assert _sanitize_handle("") == ""
        assert _sanitize_handle("   ") == ""

    def test_strips_unsafe_chars(self) -> None:
        # Spaces / @ / underscores are not DNS-safe and get dropped.
        assert _sanitize_handle("bad host@name") == "badhostname"


class TestInstanceHandle:
    def test_defaults_to_sentinel_when_unconfigured(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "connect_instance_handle", "", raising=False)
        monkeypatch.setattr(settings, "augmentum_public_host", "", raising=False)
        assert instance_handle() == THIS_INSTANCE_SENTINEL

    def test_uses_explicit_handle(self, handle) -> None:
        assert handle("myhost.example.com") == "myhost.example.com"

    def test_falls_back_to_public_host(self, handle) -> None:
        # Handle empty → derive from the public-host override (sanitised).
        assert handle("", public_host="https://lan.box:6443/") == "lan.box"

    def test_explicit_handle_wins_over_public_host(self, handle) -> None:
        assert handle("primary.name", public_host="other.host") == "primary.name"


class TestIsLocalInstance:
    def test_sentinel_is_local(self, handle) -> None:
        handle("myhost.example.com")
        assert is_local_instance(THIS_INSTANCE_SENTINEL) is True

    def test_configured_handle_is_local(self, handle) -> None:
        handle("myhost.example.com")
        assert is_local_instance("myhost.example.com") is True

    def test_configured_handle_is_case_insensitive(self, handle) -> None:
        handle("myhost.example.com")
        assert is_local_instance("MyHost.Example.com") is True

    def test_other_host_is_not_local(self, handle) -> None:
        handle("myhost.example.com")
        assert is_local_instance("peer.example.com") is False

    def test_empty_is_not_local(self, handle) -> None:
        handle("myhost.example.com")
        assert is_local_instance("") is False


class TestResolutionWithHandle:
    def test_local_did_for_uses_configured_handle(self, handle) -> None:
        handle("myhost.example.com")
        assert local_did_for("usr_abc") == "usr_abc@myhost.example.com"

    def test_resolve_configured_handle_is_local(self, handle) -> None:
        handle("myhost.example.com")
        resolved = resolve_peer_did("usr_abc@myhost.example.com")
        assert resolved is not None
        assert resolved.kind == "local"
        assert resolved.address == "usr_abc"

    def test_legacy_sentinel_still_local_with_handle_set(self, handle) -> None:
        # Back-compat: DIDs minted before instance identity existed (stored
        # in old connect_contacts rows) must keep resolving local even after
        # the operator names the instance.
        handle("myhost.example.com")
        resolved = resolve_peer_did(f"usr_old@{THIS_INSTANCE_SENTINEL}")
        assert resolved is not None
        assert resolved.kind == "local"
        assert resolved.address == "usr_old"

    def test_foreign_host_still_fabric_with_handle_set(self, handle) -> None:
        handle("myhost.example.com")
        resolved = resolve_peer_did("usr_x@peer.example.com")
        assert resolved is not None
        assert resolved.kind == "fabric"
        assert resolved.address == "peer.example.com"

    def test_roundtrip_with_handle(self, handle) -> None:
        handle("myhost.example.com")
        resolved = resolve_peer_did(local_did_for("usr_round"))
        assert resolved is not None
        assert resolved.kind == "local"
        assert resolved.address == "usr_round"


@pytest.mark.asyncio
async def test_display_name_resolves_for_configured_handle(handle) -> None:
    handle("myhost.example.com")

    class _Conn:
        async def execute(self, *_a, **_k):
            class _Cur:
                async def fetchone(self_inner):
                    return ("Alice",)

                async def close(self_inner):
                    return None

            return _Cur()

    # A DID carrying our real handle must be treated as same-instance and
    # hit the local display-name lookup (returns "Alice"), not fall back
    # to the raw local-part.
    result = await contacts.display_name_for_did(_Conn(), "usr_abc@myhost.example.com")
    assert result == "Alice"
