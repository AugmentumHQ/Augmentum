"""Connect peer-DID resolver — same-instance / fabric / malformed."""

from __future__ import annotations

import logging

import pytest

from augmentum.connect.contacts import (
    THIS_INSTANCE_SENTINEL,
    display_name_for_did,
    local_did_for,
    resolve_peer_did,
)


class _FailingConn:
    async def execute(self, *_args, **_kwargs):
        raise RuntimeError("db unavailable")


class TestResolvePeerDid:
    def test_same_instance_returns_local_user_id(self) -> None:
        resolved = resolve_peer_did(f"bob@{THIS_INSTANCE_SENTINEL}")
        assert resolved is not None
        assert resolved.kind == "local"
        assert resolved.address == "bob"

    def test_fabric_peer_returns_host(self) -> None:
        # Any non-"this-instance" suffix is treated as a fabric peer host.
        resolved = resolve_peer_did("alice@peer.example.com")
        assert resolved is not None
        assert resolved.kind == "fabric"
        assert resolved.address == "peer.example.com"

    def test_empty_returns_none(self) -> None:
        assert resolve_peer_did("") is None
        assert resolve_peer_did(None) is None  # type: ignore[arg-type]

    def test_missing_at_sign_returns_none(self) -> None:
        # No @ means we can't parse the routing target.
        assert resolve_peer_did("just-a-name") is None

    def test_empty_user_part_returns_none(self) -> None:
        # "@instance" has no user — malformed.
        assert resolve_peer_did("@this-instance") is None

    def test_empty_instance_part_returns_none(self) -> None:
        assert resolve_peer_did("bob@") is None

    def test_did_form_returns_none_for_now(self) -> None:
        # Forward-compat: did:* forms aren't recognised yet. When the
        # minimum-viable DID layer lands, this test gets inverted.
        assert resolve_peer_did("did:augmentum:abc123") is None

    def test_local_did_for_roundtrips(self) -> None:
        # local_did_for(x) → resolve(...) → local(x). The two helpers
        # must agree or peer routing breaks.
        did = local_did_for("alex")
        resolved = resolve_peer_did(did)
        assert resolved is not None
        assert resolved.kind == "local"
        assert resolved.address == "alex"

    def test_user_with_at_in_username_uses_rpartition(self) -> None:
        # Defensive — username can't currently contain '@' but
        # rpartition guarantees we split on the LAST @ if it ever
        # does. The instance part is always the rightmost segment.
        resolved = resolve_peer_did("weird@user@this-instance")
        assert resolved is not None
        assert resolved.kind == "local"
        assert resolved.address == "weird@user"


@pytest.mark.asyncio
async def test_display_name_for_did_logs_lookup_failure_and_falls_back(caplog) -> None:
    caplog.set_level(logging.WARNING)

    result = await display_name_for_did(_FailingConn(), f"bob@{THIS_INSTANCE_SENTINEL}")

    assert result == "bob"
    assert "Failed to resolve local Connect DID display name" in caplog.text
