"""Tests for the fabric wire protocol.

Pins the contract every higher fabric phase relies on:

  - envelopes signed by a known identity roundtrip cleanly through
    encode/decode
  - tampered payloads / signatures are rejected
  - protocol version mismatches surface as clean errors
  - unknown msg_type / malformed JSON / missing fields all raise
    FabricProtocolError (never a bare crash)
"""
from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from augmentum.fabric.protocol import (
    MSG_HEARTBEAT,
    MSG_HELLO,
    PROTOCOL_VERSION,
    FabricEnvelope,
    FabricProtocolError,
)


def _identity_pair():
    """Generate (private_key, public_key_b64) without using FabricIdentity.

    Tests at this layer should avoid the settings_store roundtrip;
    protocol.py knows nothing about identity persistence.
    """
    import base64

    from cryptography.hazmat.primitives import serialization

    priv = Ed25519PrivateKey.generate()
    pub_bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv, base64.b64encode(pub_bytes).decode("ascii")


def test_envelope_roundtrip():
    priv, pub_b64 = _identity_pair()
    env = FabricEnvelope.build(
        msg_type=MSG_HELLO,
        seq=1,
        sender_node_id="abc123",
        payload={"hostname": "desktop", "public_key": pub_b64},
        signing_key=priv,
    )
    wire = env.to_wire()
    parsed = FabricEnvelope.from_wire(wire, expected_sender_pubkey_b64=pub_b64)
    assert parsed.msg_type == MSG_HELLO
    assert parsed.seq == 1
    assert parsed.sender_node_id == "abc123"
    assert parsed.payload["hostname"] == "desktop"
    assert parsed.protocol_version == PROTOCOL_VERSION


def test_signature_verification_rejects_tampered_payload():
    priv, pub_b64 = _identity_pair()
    env = FabricEnvelope.build(
        msg_type=MSG_HEARTBEAT, seq=5, sender_node_id="x",
        payload={"key": "value"}, signing_key=priv,
    )
    wire = env.to_wire()
    obj = json.loads(wire)
    # Tamper the payload after signing.
    obj["payload"] = {"key": "different_value"}
    tampered = json.dumps(obj)
    with pytest.raises(FabricProtocolError, match="signature"):
        FabricEnvelope.from_wire(tampered, expected_sender_pubkey_b64=pub_b64)


def test_signature_verification_rejects_wrong_pubkey():
    priv_a, pub_a = _identity_pair()
    _, pub_b = _identity_pair()
    env = FabricEnvelope.build(
        msg_type=MSG_HELLO, seq=1, sender_node_id="x",
        payload={}, signing_key=priv_a,
    )
    with pytest.raises(FabricProtocolError, match="signature"):
        FabricEnvelope.from_wire(env.to_wire(), expected_sender_pubkey_b64=pub_b)


def test_protocol_version_mismatch_rejected():
    priv, pub_b64 = _identity_pair()
    env = FabricEnvelope.build(
        msg_type=MSG_HELLO, seq=1, sender_node_id="x",
        payload={}, signing_key=priv,
    )
    # Manually bump the version in the wire form.
    obj = json.loads(env.to_wire())
    obj["v"] = 999
    bumped = json.dumps(obj)
    with pytest.raises(FabricProtocolError, match="protocol version"):
        FabricEnvelope.from_wire(bumped, expected_sender_pubkey_b64=pub_b64)


def test_unknown_msg_type_rejected():
    priv, pub_b64 = _identity_pair()
    # Manually craft an envelope with an unknown msg_type using the
    # canonical-bytes helper directly (build() refuses the unknown type).
    import base64

    from augmentum.fabric.protocol import _canonical_bytes
    canonical = _canonical_bytes(
        protocol_version=PROTOCOL_VERSION,
        msg_type="zzz_not_a_real_type",
        seq=1,
        sender_node_id="x",
        payload={},
    )
    sig = priv.sign(canonical)
    raw = json.dumps({
        "v": PROTOCOL_VERSION,
        "t": "zzz_not_a_real_type",
        "seq": 1,
        "from": "x",
        "payload": {},
        "sig": base64.b64encode(sig).decode("ascii"),
    })
    with pytest.raises(FabricProtocolError, match="msg_type"):
        FabricEnvelope.from_wire(raw, expected_sender_pubkey_b64=pub_b64)


def test_malformed_json_raises_protocol_error():
    _, pub_b64 = _identity_pair()
    with pytest.raises(FabricProtocolError, match="JSON"):
        FabricEnvelope.from_wire("not valid json{{{", expected_sender_pubkey_b64=pub_b64)


def test_missing_field_raises_protocol_error():
    _, pub_b64 = _identity_pair()
    raw = json.dumps({"v": 1, "t": "hello"})  # missing seq/from/payload/sig
    with pytest.raises(FabricProtocolError, match="field"):
        FabricEnvelope.from_wire(raw, expected_sender_pubkey_b64=pub_b64)


def test_build_rejects_unknown_msg_type():
    priv, _ = _identity_pair()
    with pytest.raises(ValueError, match="msg_type"):
        FabricEnvelope.build(
            msg_type="bogus", seq=1, sender_node_id="x",
            payload={}, signing_key=priv,
        )


def test_payload_must_be_dict():
    priv, pub_b64 = _identity_pair()
    # Manually craft an envelope with a non-dict payload.
    import base64

    from augmentum.fabric.protocol import _canonical_bytes
    canonical = _canonical_bytes(
        protocol_version=PROTOCOL_VERSION, msg_type="hello",
        seq=1, sender_node_id="x", payload={},  # canonical needs a dict
    )
    sig = priv.sign(canonical)
    raw = json.dumps({
        "v": PROTOCOL_VERSION, "t": "hello", "seq": 1, "from": "x",
        "payload": ["not", "a", "dict"],
        "sig": base64.b64encode(sig).decode("ascii"),
    })
    with pytest.raises(FabricProtocolError, match="payload"):
        FabricEnvelope.from_wire(raw, expected_sender_pubkey_b64=pub_b64)
