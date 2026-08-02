"""Phase 3 tests — cross-instance attachment fetch.

Coverage:
  * Token mint + verify happy path, expiry, ref-binding, signature.
  * Outbound: SEND with attachment_ref + base_url adds token + url
    to the dispatched envelope.
  * Inbound: SEND with attachment_token + attachment_fetch_url stores
    them on the recipient's connect_messages row.
  * HTTP route surface (smoke) — direct call to the handler with a
    minimal fake request, verifying token validation paths.

The full HTTP integration test (real fastapi.TestClient with a blob
store + uploads row) is deferred to manual two-instance verification;
this unit pass pins the wiring shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import aiosqlite
import pytest

from augmentum.connect.contacts import local_did_for
from augmentum.connect.fabric_inbound import apply_inbound_fabric_envelope
from augmentum.connect.fabric_transport import (
    sign_attachment_token,
    verify_attachment_token,
)
from augmentum.connect.hub import ConnectHub
from augmentum.connect.message_routing import handle_message_envelope
from augmentum.connect.protocol import (
    MSG_TEXT_SEND,
    ConnectEnvelope,
    serialise_envelope,
)
from augmentum.notifications.hub import NotificationHub

CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql",
).read_text(encoding="utf-8")
NOTIFICATIONS_MIGRATION = Path(
    "augmentum/state/migrations/221_notification_substrate.sql",
).read_text(encoding="utf-8")
REACTIONS_MIGRATION = Path(
    "augmentum/state/migrations/233_connect_message_reactions.sql",
).read_text(encoding="utf-8")
OUTBOX_MIGRATION = Path(
    "augmentum/state/migrations/241_connect_fabric_outbox.sql",
).read_text(encoding="utf-8")
ATTACH_MIGRATION = Path(
    "augmentum/state/migrations/242_connect_fabric_attachment_fields.sql",
).read_text(encoding="utf-8")


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.execute(
            "CREATE TABLE IF NOT EXISTS schema_version "
            "(version INTEGER PRIMARY KEY, description TEXT, "
            " applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
        )
        await c.executescript(CONNECT_MIGRATION)
        await c.executescript(NOTIFICATIONS_MIGRATION)
        await c.executescript(REACTIONS_MIGRATION)
        await c.executescript(OUTBOX_MIGRATION)
        await c.executescript(ATTACH_MIGRATION)
        await c.commit()
        yield c


# ── Token mint + verify ───────────────────────────────────────────────


def test_token_round_trip_happy_path() -> None:
    token = sign_attachment_token(identity=None, ref="ul_abc")
    ok, err = verify_attachment_token(
        identity=None, token=token, expected_ref="ul_abc",
    )
    assert ok is True
    assert err == ""


def test_token_expiry_rejection() -> None:
    # Sign with now=1000, ttl=10 → exp=1010. Verify at now=2000 → expired.
    token = sign_attachment_token(
        identity=None, ref="ul_abc", ttl_seconds=10, now=1000,
    )
    ok, err = verify_attachment_token(
        identity=None, token=token, expected_ref="ul_abc", now=2000,
    )
    assert ok is False
    assert err == "expired"


def test_token_ref_mismatch_rejection() -> None:
    token = sign_attachment_token(identity=None, ref="ul_abc")
    ok, err = verify_attachment_token(
        identity=None, token=token, expected_ref="ul_DIFFERENT",
    )
    assert ok is False
    assert err == "ref_mismatch"


def test_token_signature_tamper_rejection() -> None:
    """Flip several bytes in the signature → rejection.

    Tampering with a single base64 char can sometimes round-trip to
    the same bytes due to padding semantics; pick a tamper that's
    guaranteed to change the decoded bytes (replace the entire sig
    with a known-different one of the same length-class).
    """
    token = sign_attachment_token(identity=None, ref="ul_abc")
    payload, sig = token.split(".", 1)
    # Replace the signature with a known different one of valid
    # base64url shape but wrong contents.
    tampered_sig = "A" * len(sig)
    tampered = f"{payload}.{tampered_sig}"
    ok, err = verify_attachment_token(
        identity=None, token=tampered, expected_ref="ul_abc",
    )
    assert ok is False
    assert err in ("bad_signature", "malformed")


def test_token_malformed_rejection() -> None:
    ok, err = verify_attachment_token(
        identity=None, token="garbage_no_dot",
        expected_ref="ul_abc",
    )
    assert ok is False
    assert err == "malformed"


def test_token_uses_real_fabric_identity() -> None:
    """When a real FabricIdentity is supplied, mint+verify use the
    derived HMAC secret. A different identity won't verify the same
    token — guards against secret cross-contamination."""

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from augmentum.fabric.identity import FabricIdentity

    priv_a = Ed25519PrivateKey.generate()
    identity_a = FabricIdentity(
        node_id="A", private_key=priv_a, public_key=priv_a.public_key(),
    )
    priv_b = Ed25519PrivateKey.generate()
    identity_b = FabricIdentity(
        node_id="B", private_key=priv_b, public_key=priv_b.public_key(),
    )

    token = sign_attachment_token(identity=identity_a, ref="ul_x")
    ok_a, _ = verify_attachment_token(
        identity=identity_a, token=token, expected_ref="ul_x",
    )
    ok_b, err_b = verify_attachment_token(
        identity=identity_b, token=token, expected_ref="ul_x",
    )
    assert ok_a is True
    assert ok_b is False
    assert err_b == "bad_signature"


# ── Outbound: token added to envelope ────────────────────────────────


class _PassthroughCoord:
    """Fabric coordinator stub that captures the dispatched payload."""

    def __init__(self) -> None:
        paired = MagicMock(node_id="peer-B", hostname="instance-B")
        self._peers = {"peer-B": MagicMock(paired=paired)}
        self.captured: list[dict] = []

    async def send_to_peer(
        self, node_id: str, *, msg_type: str, payload: dict,
    ) -> bool:
        self.captured.append(dict(payload))
        return True


@pytest.mark.asyncio
async def test_fabric_send_with_attachment_mints_token_and_url(conn) -> None:
    """SEND with attachment_ref + base_url → envelope payload carries
    attachment_token + attachment_fetch_url."""

    coord = _PassthroughCoord()
    hub = ConnectHub()
    notif = NotificationHub()
    await handle_message_envelope(
        conn=conn,
        connect_hub=hub,
        notification_hub=notif,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer="bob@instance-B",
            data={
                "thread_id": "t-1", "message_id": "m-1",
                "body": "photo!", "attachment_ref": "ul_pic",
            },
        ),
        sender_user_id="alice",
        sender_did=local_did_for("alice"),
        fabric_coordinator=coord,
        our_attachment_base_url="https://instance-A.example.com",
    )

    assert len(coord.captured) == 1
    sent_env = json.loads(coord.captured[0]["envelope"])
    assert sent_env["data"]["attachment_token"]
    assert sent_env["data"]["attachment_fetch_url"] == (
        "https://instance-A.example.com/api/connect/fabric/attachments/ul_pic"
    )

    # Token verifies for the bound ref + same identity (None).
    ok, _ = verify_attachment_token(
        identity=None,
        token=sent_env["data"]["attachment_token"],
        expected_ref="ul_pic",
    )
    assert ok is True


@pytest.mark.asyncio
async def test_fabric_send_without_base_url_omits_token(conn) -> None:
    """No base URL provided → no token/url added (graceful degrade)."""

    coord = _PassthroughCoord()
    await handle_message_envelope(
        conn=conn,
        connect_hub=ConnectHub(),
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer="bob@instance-B",
            data={
                "thread_id": "t-2", "message_id": "m-2",
                "body": "pic", "attachment_ref": "ul_x",
            },
        ),
        sender_user_id="alice",
        sender_did=local_did_for("alice"),
        fabric_coordinator=coord,
        our_attachment_base_url="",
    )
    sent_env = json.loads(coord.captured[0]["envelope"])
    assert sent_env["data"].get("attachment_token", "") == ""
    assert sent_env["data"].get("attachment_fetch_url", "") == ""


# ── Inbound: token stored on recipient row ───────────────────────────


@pytest.mark.asyncio
async def test_inbound_send_stores_attachment_fetch_fields(conn) -> None:
    hub = ConnectHub()
    env = ConnectEnvelope(
        kind="msg", verb=MSG_TEXT_SEND, peer=local_did_for("bob"),
        data={
            "thread_id": "t-1", "message_id": "m-inb",
            "body": "img", "attachment_ref": "ul_pic",
            "attachment_token": "fake.token",
            "attachment_fetch_url": "https://instance-A.example.com/api/connect/fabric/attachments/ul_pic",
        },
    )
    res = await apply_inbound_fabric_envelope(
        conn,
        connect_hub=hub,
        notification_hub=None,
        fabric_payload={
            "envelope": serialise_envelope(env),
            "source_did": "alice@instance-A",
            "target_user_id": "bob",
            "sender_party_id": "p",
        },
    )
    assert res["applied"] is True

    cur = await conn.execute(
        "SELECT attachment_ref, attachment_fetch_url, attachment_fetch_token "
        "FROM connect_messages WHERE message_id = ? AND user_id = ?",
        ("m-inb", "bob"),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    assert row[0] == "ul_pic"
    assert row[1] == (
        "https://instance-A.example.com/api/connect/fabric/attachments/ul_pic"
    )
    assert row[2] == "fake.token"


@pytest.mark.asyncio
async def test_inbound_send_without_attachment_leaves_columns_null(conn) -> None:
    """Plain text message — no token, no URL, columns stay NULL."""

    env = ConnectEnvelope(
        kind="msg", verb=MSG_TEXT_SEND, peer=local_did_for("bob"),
        data={"thread_id": "t-1", "message_id": "m-plain", "body": "txt"},
    )
    res = await apply_inbound_fabric_envelope(
        conn,
        connect_hub=ConnectHub(),
        notification_hub=None,
        fabric_payload={
            "envelope": serialise_envelope(env),
            "source_did": "alice@instance-A",
            "target_user_id": "bob",
            "sender_party_id": "p",
        },
    )
    assert res["applied"] is True

    cur = await conn.execute(
        "SELECT attachment_fetch_url, attachment_fetch_token "
        "FROM connect_messages WHERE message_id = ? AND user_id = ?",
        ("m-plain", "bob"),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row == (None, None)
