"""Fabric wire protocol: signed JSON envelopes over WebSocket.

Every fabric message rides in a :class:`FabricEnvelope` carrying a
protocol version, sender node_id, monotonic sequence number, payload,
and ed25519 signature over the canonical-form contents. The receiver
verifies the signature against the sender's public key (looked up in
the local ``fabric_nodes`` row paired with that node_id).

Versioning lives at the envelope layer so future protocol revisions
can ship deltas without renegotiating from scratch (Syncthing BEP
pattern: each side advertises its supported version range in the
``hello`` payload; both sides downgrade to the highest mutually-
supported version).

This module is pure Python with no I/O, no global state, and no
side effects on import.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

# Current wire-protocol version. Bump only for breaking changes;
# additive payload fields don't require a version bump.
PROTOCOL_VERSION = 1

# Message types currently defined. Phase 1 ships the four below;
# Phase 2 adds "capabilities" + "capabilities_update" and Phase 3
# adds "request" + "response" for cross-peer dispatch.
MSG_HELLO = "hello"                # first message post-WS-accept
MSG_HEARTBEAT = "heartbeat"        # liveness + (future) state push
MSG_ACK = "ack"                    # acknowledge a received message
MSG_ERROR = "error"                # protocol/auth/format error
# Phase 9: server→peer outbound message types (control plane).
# All require coordinator.send_to_peer + an existing WS connection.
MSG_CANCEL_REQUEST = "cancel_request"  # {request_id: str} — abort an in-flight HTTPS request
MSG_JOB_STARTED = "job_started"        # {request_id, kind, model?} — peer started serving
MSG_JOB_PROGRESS = "job_progress"      # {request_id, fraction|stage|tokens?} — periodic update
MSG_JOB_COMPLETED = "job_completed"    # {request_id, ok: bool} — terminal success
MSG_JOB_FAILED = "job_failed"          # {request_id, reason: str} — terminal failure
# Wedge B: Connect (calls + 1:1 text) over fabric. Payload carries a
# wire-form ConnectEnvelope inside ``connect_envelope`` plus a
# ``target_did`` so the receiver knows which local user the inner
# envelope addresses. See augmentum/connect/fabric_transport.py +
# augmentum/connect/fabric_inbound.py.
MSG_CONNECT_ENVELOPE = "connect_envelope"

_VALID_MSG_TYPES = frozenset({
    MSG_HELLO, MSG_HEARTBEAT, MSG_ACK, MSG_ERROR,
    MSG_CANCEL_REQUEST,
    MSG_JOB_STARTED, MSG_JOB_PROGRESS, MSG_JOB_COMPLETED, MSG_JOB_FAILED,
    MSG_CONNECT_ENVELOPE,
})


@dataclass(frozen=True)
class FabricEnvelope:
    """A single fabric protocol message.

    Constructed via :meth:`build` (signs on the way out) and parsed via
    :meth:`from_wire` (verifies on the way in). Direct construction is
    reserved for tests that want to inject unsigned envelopes; the
    production path always rides through build/from_wire.
    """

    protocol_version: int
    msg_type: str
    seq: int
    sender_node_id: str
    payload: dict
    signature: str  # base64 ed25519 signature over canonical bytes

    @classmethod
    def build(
        cls,
        *,
        msg_type: str,
        seq: int,
        sender_node_id: str,
        payload: dict,
        signing_key: Ed25519PrivateKey,
    ) -> "FabricEnvelope":
        """Build + sign an envelope. The signature covers a
        deterministic, canonical-form serialisation of the contents so
        two peers signing/verifying the same logical message reach
        the same bytes (no JSON key-ordering ambiguity).
        """
        if msg_type not in _VALID_MSG_TYPES:
            raise ValueError(f"unknown msg_type: {msg_type!r}")
        canonical = _canonical_bytes(
            protocol_version=PROTOCOL_VERSION,
            msg_type=msg_type,
            seq=seq,
            sender_node_id=sender_node_id,
            payload=payload,
        )
        sig = signing_key.sign(canonical)
        return cls(
            protocol_version=PROTOCOL_VERSION,
            msg_type=msg_type,
            seq=seq,
            sender_node_id=sender_node_id,
            payload=dict(payload),  # defensive copy
            signature=base64.b64encode(sig).decode("ascii"),
        )

    def to_wire(self) -> str:
        """Serialise to a JSON string suitable for WS ``send_text``."""
        return json.dumps(
            {
                "v": self.protocol_version,
                "t": self.msg_type,
                "seq": self.seq,
                "from": self.sender_node_id,
                "payload": self.payload,
                "sig": self.signature,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_wire(
        cls,
        raw: str,
        *,
        expected_sender_pubkey_b64: str,
    ) -> "FabricEnvelope":
        """Parse a wire-form envelope and verify its signature.

        Raises :class:`FabricProtocolError` on any malformed input,
        unknown msg_type, version mismatch, or signature failure.
        Returns the validated envelope on success.

        The caller must supply the expected sender's pubkey (looked up
        in the local ``fabric_nodes`` table from the ``sender_node_id``
        BEFORE calling -- this method does NOT trust the wire). A
        mismatch is the strongest tamper signal we have at this
        layer; treat it as ``CLOSE_BAD_SIGNATURE``.
        """
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FabricProtocolError(f"invalid JSON: {exc}") from None

        if not isinstance(obj, dict):
            raise FabricProtocolError("envelope must be a JSON object")

        try:
            version = int(obj["v"])
            msg_type = str(obj["t"])
            seq = int(obj["seq"])
            sender_node_id = str(obj["from"])
            payload = obj["payload"]
            signature_b64 = str(obj["sig"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FabricProtocolError(f"missing/wrong-type field: {exc}") from None

        if not isinstance(payload, dict):
            raise FabricProtocolError("payload must be a JSON object")
        if msg_type not in _VALID_MSG_TYPES:
            raise FabricProtocolError(f"unknown msg_type: {msg_type!r}")
        if version != PROTOCOL_VERSION:
            # Phase 1 is single-version. Future phases negotiate in hello.
            raise FabricProtocolError(
                f"unsupported protocol version: {version} (this node speaks {PROTOCOL_VERSION})"
            )

        canonical = _canonical_bytes(
            protocol_version=version,
            msg_type=msg_type,
            seq=seq,
            sender_node_id=sender_node_id,
            payload=payload,
        )
        try:
            pub_bytes = base64.b64decode(expected_sender_pubkey_b64)
            pub = Ed25519PublicKey.from_public_bytes(pub_bytes)
            sig_bytes = base64.b64decode(signature_b64)
            pub.verify(sig_bytes, canonical)
        except Exception as exc:
            raise FabricProtocolError(f"signature verification failed: {exc}") from None

        return cls(
            protocol_version=version,
            msg_type=msg_type,
            seq=seq,
            sender_node_id=sender_node_id,
            payload=payload,
            signature=signature_b64,
        )


class FabricProtocolError(Exception):
    """Any wire-form parse / verify failure. Raised exclusively by
    :meth:`FabricEnvelope.from_wire`. Callers should map this to a WS
    close code (1003 for malformed, 4401 for bad signature) and never
    surface details to the wire side -- a noisy ``error`` envelope
    response leaks parse-state to potential attackers.
    """


def _canonical_bytes(
    *,
    protocol_version: int,
    msg_type: str,
    seq: int,
    sender_node_id: str,
    payload: dict,
) -> bytes:
    """Deterministic byte-serialisation used both for signing and
    verifying. Stable across Python versions and dict orderings.
    """
    return json.dumps(
        {
            "v": protocol_version,
            "t": msg_type,
            "seq": seq,
            "from": sender_node_id,
            "payload": payload,
        },
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")


def make_hello_payload(*, hostname: str, public_key_b64: str) -> dict[str, Any]:
    """Standard hello payload. Sent as the first message after WS
    accept by both sides to confirm protocol version + identity.

    Hostname is informational (used in UI); pubkey is the auth-layer
    identity already pinned at pairing time. The receiver verifies
    that ``public_key_b64`` matches the ``pubkey_ed25519`` it has on
    file for ``sender_node_id`` -- a mismatch means impersonation
    attempt or stale fabric_nodes row.
    """
    return {
        "hostname": hostname,
        "public_key": public_key_b64,
    }
