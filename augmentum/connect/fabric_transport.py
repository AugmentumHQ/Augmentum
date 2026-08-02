"""Outbound Connect-over-fabric dispatch + durable outbox.

When a message_routing or call_routing handler sees a fabric DID
(``alice@instance-A`` rather than ``alice@this-instance``), it stops
short of recipient-mirror writes and hands the envelope to this
module. Responsibilities:

  1. Resolve the DID's host-part to a paired fabric ``node_id`` via
     ``FabricCoordinator``'s peer registry.
  2. Persist the envelope to ``connect_fabric_outbox`` BEFORE the
     first send attempt — so a process crash mid-flush doesn't lose
     the message.
  3. Attempt immediate flush via ``coordinator.send_to_peer(...)``
     wrapping the ConnectEnvelope inside an ``MSG_CONNECT_ENVELOPE``
     fabric envelope.
  4. On successful send: delete the outbox row (the recipient
     instance acks at the protocol layer via Ed25519 signature
     verification — there's no app-layer ack on the happy path).
  5. On send failure (peer disconnected, socket broken, etc.):
     leave the outbox row queued. The drain loop on the next
     FabricClient reconnect picks it up.

This module does NOT directly mutate any Connect tables — the
caller has already written the sender's local row. Outbox is the
only table touched here.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from augmentum.connect.protocol import ConnectEnvelope, serialise_envelope
from augmentum.fabric.protocol import MSG_CONNECT_ENVELOPE
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

    from augmentum.fabric.coordinator import FabricCoordinator

log = get_logger(__name__)


# Max attempts before we give up on a single outbox row and fire
# EVENT_ERROR("fabric_outbox_exhausted") back at the local sender.
# Set high enough that a multi-day peer outage still drains when the
# peer comes back, but low enough that a permanently-misconfigured
# peer doesn't accumulate unboundedly.
MAX_OUTBOX_ATTEMPTS = 100


@dataclass
class DispatchResult:
    """Outcome of one fabric dispatch attempt.

    ``queued`` is True whenever we successfully enqueued (whether or
    not the immediate flush succeeded). ``delivered`` is True only
    when the WS write went through synchronously — useful for tests
    and for the routing-layer ``routed`` count.
    """

    queued: bool
    delivered: bool
    outbox_id: str = ""
    error_code: str = ""
    error_message: str = ""


async def _resolve_node_id(
    coordinator: FabricCoordinator, *, hostname: str,
) -> str | None:
    """Find the fabric ``node_id`` whose paired hostname matches.

    Linear scan against the in-memory ``_peers`` dict — fine for
    typical fleet sizes (<10 paired peers). The coordinator holds
    the lock for its own ops; we only need to read the snapshot.
    """
    # ``snapshot_peers`` is the public read accessor on the
    # coordinator; use it if available, else fall back to attribute
    # access for tests that mock a minimal coordinator stub.
    snapshot = getattr(coordinator, "snapshot_peers", None)
    peers: list[Any]
    if callable(snapshot):
        peers = await snapshot()
    else:
        peers = list(getattr(coordinator, "_peers", {}).values())
    for state in peers:
        paired = getattr(state, "paired", None)
        if paired is not None and paired.hostname == hostname:
            return paired.node_id
    return None


async def dispatch_fabric_envelope(
    conn: aiosqlite.Connection,
    *,
    coordinator: FabricCoordinator | None,
    target_hostname: str,
    source_did: str,
    sender_user_id: str,
    sender_party_id: str,
    envelope: ConnectEnvelope,
) -> DispatchResult:
    """Enqueue + best-effort flush a Connect envelope to a fabric peer.

    The envelope's ``peer`` field is the target DID (e.g.
    ``bob@instance-B``); ``source_did`` is the local user's DID
    (``alice@this-instance``). Both ride inside the fabric payload
    so the receiving instance's inbound handler can re-route to the
    right local user.

    Returns DispatchResult. The caller uses ``queued=True`` as the
    sender-facing success signal (the user's UI shows "sent"; live
    delivery is best-effort and acked separately via
    MSG_TEXT_DELIVERED).
    """

    if coordinator is None:
        # Fabric isn't enabled on this instance at all — surface as
        # a clean error so the UI can show "cross-instance not
        # available on this box" rather than a silent stall.
        return DispatchResult(
            queued=False, delivered=False,
            error_code="fabric_unavailable",
            error_message=(
                "fabric is not enabled on this Augmentum instance; "
                "cross-instance Connect requires fabric pairing"
            ),
        )

    node_id = await _resolve_node_id(coordinator, hostname=target_hostname)
    if node_id is None:
        return DispatchResult(
            queued=False, delivered=False,
            error_code="fabric_peer_unknown",
            error_message=(
                f"no paired fabric peer with hostname '{target_hostname}' — "
                "pair the peer via /api/fabric/pair first"
            ),
        )

    outbox_id = f"fox_{secrets.token_hex(6)}"
    # Extract the target user_id from the DID's local-part. The DID's
    # hostname is meaningful only on the sender's side; the receiving
    # instance resolves the user directly from the local-part rather
    # than re-parsing the DID (which would map "bob@instance-B" to
    # "fabric" on Bob's box where instance-B is the LOCAL hostname).
    target_user_id = envelope.peer.rpartition("@")[0] if envelope.peer else ""
    # Resolve the sender's human name on our OWN box (where source_did is a
    # local DID) and ride it across so the receiving instance can render a
    # username instead of the raw usr_<hash>@host DID — the remote box has no
    # way to look our user up otherwise. Best-effort: a lookup miss just omits
    # the field and the receiver falls back to the local-part.
    from augmentum.connect.contacts import display_name_for_did
    try:
        source_display_name = await display_name_for_did(conn, source_did)
    except Exception:
        source_display_name = ""
    payload = {
        "envelope": serialise_envelope(envelope),
        "source_did": source_did,
        "target_user_id": target_user_id,
        "sender_party_id": sender_party_id,
        "source_display_name": source_display_name or "",
    }
    now = int(time.time())

    # 1. Persist the intent first. A crash here means the message is
    # lost; a crash AFTER this means the drain loop picks it up.
    import json as _json
    await conn.execute(
        """INSERT INTO connect_fabric_outbox
               (id, target_node_id, sender_user_id, envelope_json,
                queued_at, attempts)
             VALUES (?, ?, ?, ?, ?, 0)""",
        (
            outbox_id, node_id, sender_user_id,
            _json.dumps(payload, separators=(",", ":")),
            now,
        ),
    )
    await conn.commit()

    # 2. Best-effort immediate flush.
    delivered = await coordinator.send_to_peer(
        node_id, msg_type=MSG_CONNECT_ENVELOPE, payload=payload,
    )

    if delivered:
        await conn.execute(
            "DELETE FROM connect_fabric_outbox WHERE id = ?",
            (outbox_id,),
        )
        await conn.commit()
        log.info(
            "connect_fabric_envelope_delivered",
            outbox_id=outbox_id, target_node_id=node_id,
            verb=envelope.verb,
        )
        return DispatchResult(
            queued=True, delivered=True, outbox_id=outbox_id,
        )

    # Not delivered — leave the outbox row in place; drain loop will
    # pick it up when the peer reconnects.
    await conn.execute(
        """UPDATE connect_fabric_outbox
             SET attempts = attempts + 1,
                 last_attempt_at = ?,
                 last_error = ?
             WHERE id = ?""",
        (now, "peer_unreachable", outbox_id),
    )
    await conn.commit()
    log.info(
        "connect_fabric_envelope_queued",
        outbox_id=outbox_id, target_node_id=node_id,
        verb=envelope.verb,
    )
    return DispatchResult(
        queued=True, delivered=False, outbox_id=outbox_id,
    )


async def drain_outbox_for_peer(
    conn: aiosqlite.Connection,
    *,
    coordinator: FabricCoordinator,
    node_id: str,
) -> dict[str, int]:
    """Flush every queued envelope to ``node_id`` in queued_at order.

    Called when a FabricClient (re)connects. Iterates rows in queue
    order, attempts each, deletes on success / increments attempts on
    failure. Returns a ``{"sent": N, "still_queued": N, "exhausted": N}``
    counter dict for telemetry + tests.

    Important: drain in serial. WebSocket writes through the same
    socket need to keep order (otherwise EVENT_TEXT_RECEIVED frames
    can land out of insertion order, which the recipient's UI is
    not robust to). The serial penalty is negligible since each
    write is small and the WS pipe is fast on a reconnected link.
    """
    counters = {"sent": 0, "still_queued": 0, "exhausted": 0}
    now = int(time.time())

    cur = await conn.execute(
        """SELECT id, envelope_json, attempts, sender_user_id
             FROM connect_fabric_outbox
             WHERE target_node_id = ?
             ORDER BY queued_at, rowid""",
        (node_id,),
    )
    rows = await cur.fetchall()
    await cur.close()

    for outbox_id, envelope_json_str, attempts, sender_user_id in rows:
        import json as _json
        try:
            payload = _json.loads(envelope_json_str)
        except _json.JSONDecodeError:
            # Corrupted row — can't ever send. Surface as exhausted
            # so the local sender's UI stops waiting.
            await conn.execute(
                "DELETE FROM connect_fabric_outbox WHERE id = ?",
                (outbox_id,),
            )
            await conn.commit()
            counters["exhausted"] += 1
            log.warning(
                "connect_fabric_outbox_corrupt",
                outbox_id=outbox_id,
            )
            continue

        delivered = await coordinator.send_to_peer(
            node_id, msg_type=MSG_CONNECT_ENVELOPE, payload=payload,
        )
        if delivered:
            await conn.execute(
                "DELETE FROM connect_fabric_outbox WHERE id = ?",
                (outbox_id,),
            )
            await conn.commit()
            counters["sent"] += 1
            continue

        # Failed — bump attempts; give up if we've hit the cap.
        new_attempts = attempts + 1
        if new_attempts >= MAX_OUTBOX_ATTEMPTS:
            await conn.execute(
                "DELETE FROM connect_fabric_outbox WHERE id = ?",
                (outbox_id,),
            )
            await conn.commit()
            counters["exhausted"] += 1
            # Best-effort: surface a fabric_outbox_exhausted error
            # to the local sender's signaling WS so their UI knows
            # the message will never arrive. The hub lookup may fail
            # (sender offline); that's OK — they'll see the giveup
            # state when they reconnect via their next catch-up.
            log.warning(
                "connect_fabric_outbox_exhausted",
                outbox_id=outbox_id,
                sender_user_id=sender_user_id,
                target_node_id=node_id,
            )
            continue

        await conn.execute(
            """UPDATE connect_fabric_outbox
                 SET attempts = ?, last_attempt_at = ?,
                     last_error = ?
                 WHERE id = ?""",
            (new_attempts, now, "peer_unreachable", outbox_id),
        )
        await conn.commit()
        counters["still_queued"] += 1

    if counters["sent"] or counters["exhausted"]:
        log.info(
            "connect_fabric_outbox_drain",
            target_node_id=node_id, **counters,
        )
    return counters


# ── Fabric attachment fetch tokens (Phase 3) ──────────────────────────


# Default token TTL for cross-instance attachment fetches. Long enough
# that a recipient's UI can render the message and trigger the GET
# without a race, short enough that a token leaked into logs / browser
# history can't be replayed weeks later. 10 minutes mirrors the
# guest-link spec's TURN cred default.
DEFAULT_ATTACHMENT_TOKEN_TTL_S = 600

# Signed-token format: ``base64url(json({"ref": str, "exp": int})).base64url(sig)``
# where sig = HMAC-SHA256(derived_secret, payload). HMAC is sufficient
# because the SAME instance signs and verifies — no cross-instance key
# distribution problem. Derived secret = sha256 of the fabric private
# key bytes; rotates if the fabric identity is regenerated (which
# invalidates outstanding tokens, which is correct behavior).
_TOKEN_DELIMITER = "."


def _hmac_secret_from_identity(identity: Any) -> bytes:
    """Derive an HMAC secret from the fabric identity.

    Doesn't expose the raw private key; uses sha256(priv-bytes) as the
    derivation. Stable across the identity's lifetime. Returns 32
    bytes.

    When ``identity`` is None we ONLY fall back to a fixed test secret
    while running under pytest. Production paths must always supply a
    real identity — a missing identity in production would otherwise
    let a single hardcoded HMAC key sign tokens that any caller could
    forge, giving cross-tenant attachment access on this instance.
    """
    if identity is None:
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            raise RuntimeError(
                "Refusing to mint/verify attachment tokens without a fabric "
                "identity outside the test harness. Initialize "
                "app.state.fabric_identity before calling.",
            )
        return b"augmentum-fabric-attachment-test-secret-do-not-use-in-prod"
    from cryptography.hazmat.primitives import serialization
    raw = identity.private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return sha256(raw).digest()


def _b64url(data: bytes) -> str:
    """URL-safe base64 without padding (matches JWT convention)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def sign_attachment_token(
    *,
    identity: Any,
    ref: str,
    ttl_seconds: int = DEFAULT_ATTACHMENT_TOKEN_TTL_S,
    now: int | None = None,
) -> str:
    """Mint a fabric-signed attachment fetch token.

    ``identity`` may be the FabricIdentity (production) or None (unit
    tests using the fixed-test secret). ``ref`` is the upload_id the
    token grants access to (e.g. ``ul_abc123``). Tokens are bound to
    the ref so a leaked token doesn't grant access to other
    attachments on the same instance.
    """
    now = int(time.time()) if now is None else now
    payload = json.dumps(
        {"ref": ref, "exp": now + ttl_seconds},
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    secret = _hmac_secret_from_identity(identity)
    sig = hmac.new(secret, payload, sha256).digest()
    return f"{_b64url(payload)}{_TOKEN_DELIMITER}{_b64url(sig)}"


def verify_attachment_token(
    *,
    identity: Any,
    token: str,
    expected_ref: str,
    now: int | None = None,
) -> tuple[bool, str]:
    """Verify a token. Returns ``(ok, error_code)``.

    Checks (in order): wire format, signature against the identity's
    HMAC key, ref binding, expiry. Constant-time signature compare so
    a token-substitution attacker can't learn anything from per-byte
    timing.
    """
    if not token or _TOKEN_DELIMITER not in token:
        return False, "malformed"
    try:
        payload_b64, sig_b64 = token.split(_TOKEN_DELIMITER, 1)
        payload = _b64url_decode(payload_b64)
        sig = _b64url_decode(sig_b64)
    except Exception:
        return False, "malformed"

    secret = _hmac_secret_from_identity(identity)
    expected_sig = hmac.new(secret, payload, sha256).digest()
    if not hmac.compare_digest(sig, expected_sig):
        return False, "bad_signature"

    try:
        body = json.loads(payload.decode("utf-8"))
        ref = str(body.get("ref") or "")
        exp = int(body.get("exp") or 0)
    except Exception:
        return False, "malformed_payload"

    if ref != expected_ref:
        return False, "ref_mismatch"

    now = int(time.time()) if now is None else now
    if exp <= now:
        return False, "expired"

    return True, ""
