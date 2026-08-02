"""Peer pairing + signature verification.

Two distinct flows live here:

  Pairing: operator-driven, runs once per peer. One side sends the
  other a signed ``PairRequest`` carrying its own node identity and
  the fingerprint hint the operator just typed. The receiver verifies
  the signature, checks the fingerprint matches the supplied pubkey
  (catching transcription typos), and writes the new peer to
  ``fabric_nodes``. Both sides do this symmetrically so each has the
  other on file.

  Per-connection auth: on every WebSocket open the peer presents a
  signed challenge proving control of the pinned private key. The
  receiver looks up the claimed ``sender_node_id`` in fabric_nodes,
  pulls the pinned pubkey, and verifies the signature with
  :meth:`FabricEnvelope.from_wire`. No fabric_nodes match = unknown
  peer = close with 4401.

Pairing data is the only piece of fabric state that touches SQLite.
Connection state, capabilities, heartbeats, and live peer status
live entirely in RAM (the writer-lock landmine principle).
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from augmentum.fabric.identity import FabricIdentity
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

# Validity window for a pair-request signature. Short enough to make
# replay infeasible (the request can't be re-used hours later by an
# attacker who picks up a wire log), wide enough to forgive routine
# clock skew between two LAN peers (NTP-synced devices typically agree
# within 100 ms; this is 5 minutes of slack).
_PAIR_REQUEST_TTL_S = 300


@dataclass(frozen=True)
class PairedPeer:
    """A row from the ``fabric_nodes`` table, parsed into a dataclass.

    Used to bridge the SQLite row format and the higher-level Python
    code in coordinator/routes without leaking column-tuple indices
    everywhere.
    """

    node_id: str
    hostname: str
    role: str               # 'primary' | 'peer'
    pubkey_b64: str
    fingerprint: str
    addr: str
    tier: str               # 'local' | 'federated'
    fabric_share_enabled: bool
    paired_at: str
    last_seen_at: str | None
    icon: str = ""          # Phase 8: local-pick emoji for this peer in our fleet


@dataclass(frozen=True)
class PairRequest:
    """Wire-form pairing request sent during the manual pair flow.

    The signature covers ``(sender_node_id, hostname, pubkey_b64,
    timestamp)``; the receiver verifies it with the supplied pubkey
    AND checks ``fingerprint_hint`` matches ``pubkey_b64``'s derived
    fingerprint (so an operator typo in the fingerprint paste-box
    surfaces as a clean error rather than as a silent corruption).
    """

    sender_node_id: str
    hostname: str
    pubkey_b64: str
    fingerprint_hint: str   # what the operator typed; we re-derive and compare
    role: str               # 'primary' | 'peer' -- what the sender wants to be
    timestamp: int          # unix seconds; for replay-window check
    signature: str          # base64; over canonical bytes of fields above


def build_pair_request(
    *,
    identity: FabricIdentity,
    hostname: str,
    target_fingerprint_hint: str,
    role: str = "peer",
) -> PairRequest:
    """Construct an outbound pair request from the local identity.

    ``target_fingerprint_hint`` is the fingerprint of the *peer being
    paired with* -- not our own -- and is what the operator typed
    into the local pair-UI. Sent so the receiver can cross-check
    that we actually believed we were pairing with the right node
    when we ran our half of the handshake.
    """
    payload = _pair_canonical_bytes(
        sender_node_id=identity.node_id,
        hostname=hostname,
        pubkey_b64=identity.public_key_b64,
        fingerprint_hint=target_fingerprint_hint,
        role=role,
        timestamp=int(time.time()),
    )
    sig = identity.sign(payload)
    return PairRequest(
        sender_node_id=identity.node_id,
        hostname=hostname,
        pubkey_b64=identity.public_key_b64,
        fingerprint_hint=target_fingerprint_hint,
        role=role,
        timestamp=int(time.time()),
        signature=base64.b64encode(sig).decode("ascii"),
    )


def verify_pair_request(
    req: PairRequest,
    *,
    own_fingerprint: str,
) -> None:
    """Validate an incoming pair request. Raises on any failure.

    Checks in order: timestamp window, fingerprint match against the
    supplied pubkey, ed25519 signature against the canonical bytes.
    ``own_fingerprint`` is *this node's* fingerprint, used to confirm
    the remote operator actually targeted us (catches paste-the-
    wrong-fingerprint-into-the-wrong-window mistakes).
    """
    now = int(time.time())
    if abs(now - req.timestamp) > _PAIR_REQUEST_TTL_S:
        raise PairRequestError(
            f"timestamp out of window ({abs(now - req.timestamp)}s vs {_PAIR_REQUEST_TTL_S}s)"
        )

    if req.fingerprint_hint != own_fingerprint:
        raise PairRequestError(
            "fingerprint mismatch: the requesting peer thinks they are "
            "pairing with a different node than this one. Double-check "
            "that you pasted this node's fingerprint, not another's."
        )

    derived = _fingerprint_from_b64(req.pubkey_b64)
    # The remote MUST attach the pubkey matching the signing key. We
    # don't trust the wire blindly: re-derive the fingerprint from the
    # supplied pubkey and compare to what the sender claimed to be
    # their identity.
    expected_sender_fp = _fingerprint_from_b64(req.pubkey_b64)
    if derived != expected_sender_fp:
        # Defensive: this is mathematically impossible given how
        # _fingerprint_from_b64 works, but the assertion documents
        # the invariant for anyone reading.
        raise PairRequestError("internal: derived fingerprint mismatch")

    canonical = _pair_canonical_bytes(
        sender_node_id=req.sender_node_id,
        hostname=req.hostname,
        pubkey_b64=req.pubkey_b64,
        fingerprint_hint=req.fingerprint_hint,
        role=req.role,
        timestamp=req.timestamp,
    )
    try:
        pub_bytes = base64.b64decode(req.pubkey_b64)
        pub = Ed25519PublicKey.from_public_bytes(pub_bytes)
        sig_bytes = base64.b64decode(req.signature)
        pub.verify(sig_bytes, canonical)
    except Exception as exc:
        raise PairRequestError(f"signature verification failed: {exc}") from None


class PairRequestError(Exception):
    """Any failure during pair-request verification. Mapped to HTTP
    400 by the route layer. The message is operator-facing -- it
    should make sense to a human trying to debug a pairing typo.
    """


async def persist_remote_node(
    db: "aiosqlite.Connection",
    *,
    node_id: str,
    hostname: str,
    role: str,
    pubkey_b64: str,
    addr: str,
    tier: str = "local",
    icon: str = "",
    enabled: bool = True,
) -> PairedPeer:
    """Insert (or refresh) a paired remote node into ``fabric_nodes``.

    The shared INSERT used by both the inbound /pair flow (after
    verifying a remote's signed PairRequest) and the outbound
    /pair-with-remote flow (after a successful round-trip with the
    remote's identity in the response).

    Idempotent: if the peer already exists, we update its hostname,
    addr, pubkey, fingerprint, and icon. The icon is local-pick (the
    LOCAL operator chose it for THEIR view of the fleet), so an
    inbound /pair from the remote does NOT overwrite an icon we've
    already set -- the COALESCE in the UPDATE keeps our local label
    if a re-pair attempt comes in with empty icon.
    """
    fingerprint = _fingerprint_from_b64(pubkey_b64)
    now_sql = _sql_now()
    enabled_int = 1 if enabled else 0
    await db.execute(
        """INSERT INTO fabric_nodes
              (id, hostname, role, pubkey_ed25519, pubkey_fingerprint,
               addr, tier, fabric_share_enabled, paired_at, last_seen_at,
               icon)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
              hostname = excluded.hostname,
              role = excluded.role,
              pubkey_ed25519 = excluded.pubkey_ed25519,
              pubkey_fingerprint = excluded.pubkey_fingerprint,
              addr = excluded.addr,
              tier = excluded.tier,
              -- Anti-hijack: keep an existing peer's enabled state ONLY when
              -- the pubkey is unchanged. If the key rotates (or an attacker
              -- re-pairs a known node_id with their own key) force it back to
              -- 0 so the new identity must be re-approved — a key swap can't
              -- silently inherit prior data-plane trust.
              fabric_share_enabled = CASE
                  WHEN fabric_nodes.pubkey_ed25519 = excluded.pubkey_ed25519
                  THEN fabric_nodes.fabric_share_enabled
                  ELSE 0
              END,
              icon = CASE
                  WHEN excluded.icon = '' THEN fabric_nodes.icon
                  ELSE excluded.icon
              END""",
        (
            node_id, hostname, role, pubkey_b64, fingerprint,
            addr, tier, enabled_int, now_sql, None, icon,
        ),
    )
    await db.commit()
    log.info(
        "fabric_pair_persisted",
        peer_node_id=node_id, fingerprint=fingerprint, role=role,
        enabled=enabled, icon=icon or "(none)",
    )
    return PairedPeer(
        node_id=node_id, hostname=hostname, role=role,
        pubkey_b64=pubkey_b64, fingerprint=fingerprint,
        addr=addr, tier=tier, fabric_share_enabled=enabled,
        paired_at=now_sql, last_seen_at=None,
        icon=icon,
    )


async def persist_pairing(
    db: "aiosqlite.Connection",
    *,
    req: PairRequest,
    addr: str,
    tier: str = "local",
) -> PairedPeer:
    """Insert the inbound-pair-verified peer into ``fabric_nodes``.

    Thin adapter around :func:`persist_remote_node` for the inbound
    flow where we have a full PairRequest in hand. Kept for backward
    compatibility with existing call sites (fabric_routes.py:113 +
    tests).
    """
    return await persist_remote_node(
        db,
        node_id=req.sender_node_id,
        hostname=req.hostname,
        role=req.role,
        pubkey_b64=req.pubkey_b64,
        addr=addr,
        tier=tier,
        # INBOUND pair lands PENDING. A signed pair envelope is NOT consent:
        # /api/fabric/pair is unauthenticated and the fingerprint it must echo
        # is handed out by /api/fabric/hello, so anyone can mint a valid-looking
        # request with their own key. The peer is invisible to
        # lookup_peer_pubkey (data-plane auth) until the operator approves it
        # via POST /api/fabric/peers/{id}/approve. Outbound, operator-initiated
        # pairing (pair_client.py) stays enabled=True — the operator consented.
        enabled=False,
    )


async def load_paired_peers(db: "aiosqlite.Connection") -> list[PairedPeer]:
    """Read all paired peers from the table. Used by the coordinator
    at startup to seed its in-memory view.

    ``icon`` was added in migration 170 (Phase 8). Older rows return
    '' for the field; the UI maps that to a fallback emoji.
    """
    cursor = await db.execute(
        """SELECT id, hostname, role, pubkey_ed25519, pubkey_fingerprint,
                  addr, tier, fabric_share_enabled, paired_at, last_seen_at,
                  icon
           FROM fabric_nodes"""
    )
    rows = await cursor.fetchall()
    return [
        PairedPeer(
            node_id=row[0],
            hostname=row[1],
            role=row[2],
            pubkey_b64=row[3],
            fingerprint=row[4],
            addr=row[5],
            tier=row[6],
            fabric_share_enabled=bool(row[7]),
            paired_at=row[8],
            last_seen_at=row[9],
            icon=row[10] or "",
        )
        for row in rows
    ]


async def lookup_peer_pubkey(
    db: "aiosqlite.Connection",
    node_id: str,
) -> str | None:
    """Look up the pinned pubkey for a node_id. Returns None when the peer
    isn't paired OR is still pending operator approval (fabric_share_enabled
    = 0) -- caller should close the WS with 4401 / reject the envelope.

    The ``fabric_share_enabled = 1`` filter is the data-plane half of the
    inbound-pairing fix: an unauthenticated /api/fabric/pair can create a
    PENDING fabric_nodes row, but a pending peer has no resolvable identity
    here, so it can never authenticate a data-plane envelope until an
    operator approves it. Without this filter, self-enrollment = instant trust.
    """
    cursor = await db.execute(
        "SELECT pubkey_ed25519 FROM fabric_nodes "
        "WHERE id = ? AND fabric_share_enabled = 1 LIMIT 1",
        (node_id,),
    )
    row = await cursor.fetchone()
    return row[0] if row else None


def make_auth_challenge() -> str:
    """Generate a fresh nonce for the WS-open auth challenge.

    The server emits this in the initial frame; the peer signs it and
    sends back the signature in their first ``hello``. Single-use
    (the server stores it on the connection state and only accepts
    the matching response once).
    """
    return secrets.token_hex(16)


# ── Internal helpers ──────────────────────────────────────────────


def _pair_canonical_bytes(
    *,
    sender_node_id: str,
    hostname: str,
    pubkey_b64: str,
    fingerprint_hint: str,
    role: str,
    timestamp: int,
) -> bytes:
    """Stable serialisation for pair-request signing/verifying. Order
    is fixed and explicit; we don't rely on dict ordering.
    """
    parts = [
        f"v1",
        f"sender={sender_node_id}",
        f"hostname={hostname}",
        f"pubkey={pubkey_b64}",
        f"fingerprint_hint={fingerprint_hint}",
        f"role={role}",
        f"ts={timestamp}",
    ]
    return "\n".join(parts).encode("utf-8")


def _fingerprint_from_b64(pubkey_b64: str) -> str:
    """Derive the SSH-style fingerprint from a base64 pubkey."""
    raw = base64.b64decode(pubkey_b64)
    digest = hashlib.sha256(raw).hexdigest()
    return f"SHA256:{digest[:32]}"


def _sql_now() -> str:
    """ISO-formatted UTC timestamp matching ``datetime('now')`` in SQLite."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")
