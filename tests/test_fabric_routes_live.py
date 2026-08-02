"""LIVE in-process API exercise of the federation HTTP surface.

Boots the real app (create_app via the `client` fixture), attaches a real
fabric identity, and drives the actual HTTP endpoints — proving the routes
are wired and that the live path emits cryptographically valid output
(the minted card verifies; the status reports a real safety code; the QR
renders). DB-backed logic (pin store, bundles, knocks) is covered by the
unit suites against real aiosqlite + migrations; here we verify the live
HTTP wiring + crypto end-to-end.
"""
from __future__ import annotations

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from augmentum.config import settings
from augmentum.fabric.contact_card import parse_card
from augmentum.fabric.identity import FabricIdentity


@pytest.fixture
def fabric_live(client, monkeypatch):
    """Enable fabric/federation and attach a real identity to the live app."""
    monkeypatch.setattr(settings, "fabric_enabled", True, raising=False)
    monkeypatch.setattr(settings, "fabric_federation_enabled", True, raising=False)
    priv = Ed25519PrivateKey.generate()
    identity = FabricIdentity(
        node_id="livenode", private_key=priv, public_key=priv.public_key(),
    )
    client.app.state.fabric_identity = identity
    return client, identity


def test_live_federation_status(fabric_live):
    client, identity = fabric_live
    r = client.get("/api/fabric/federation/status")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["enabled"] is True and j["ready"] is True
    assert j["me"]["safety_code"]  # a real, derived safety code
    assert j["reach_setting"]["label"]  # friendly posture copy, not "knock"


def test_live_mint_contact_card_is_cryptographically_valid(fabric_live):
    client, identity = fabric_live
    r = client.post("/api/fabric/contact-card", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    # The card minted by the LIVE HTTP path verifies against the instance key.
    parsed = parse_card(body["card"])
    assert parsed.instance_did_key == identity.did_key
    # Professional share block present.
    assert body["share"]["safety_code"]
    assert body["share"]["qr_url"].startswith("/api/fabric/contact-card/qr")


def test_live_contact_card_qr_renders_png(fabric_live):
    client, _ = fabric_live
    r = client.get("/api/fabric/contact-card/qr?code=K7P29QX4")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic — a real image


def test_live_routes_refuse_when_fabric_disabled(client, monkeypatch):
    monkeypatch.setattr(settings, "fabric_enabled", False, raising=False)
    r = client.get("/api/fabric/federation/status")
    assert r.status_code == 200
    assert r.json()["enabled"] is False  # honest "not turned on" state
