"""P5: end-to-end device sealing + at-rest encryption.

E2E (host can't read DMs):
  - a message sealed to the recipient DEVICE opens with the full
    device→master chain validated.
  - neither host key opens it (only the recipient device key does).
  - a device whose binding doesn't chain to the pinned master is
    rejected; a binding for a different device than the signer is
    rejected.

At-rest:
  - round-trips under the derived key; wrong key / tamper fails;
    a too-weak secret is refused.
"""
from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from augmentum.fabric import e2e
from augmentum.fabric.at_rest import (
    AtRestError,
    decrypt_at_rest,
    derive_at_rest_key,
    encrypt_at_rest,
)
from augmentum.fabric.author_binding import mint_binding
from augmentum.fabric.didkey import encode_ed25519_did


def _ed():
    priv = Ed25519PrivateKey.generate()
    raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return encode_ed25519_did(raw), priv.sign


def _full_chain():
    """Build a master, a device subkey bound to it, and the recipient
    device sealing key. Returns the pieces a real send/open needs."""
    master_did, master_sign = _ed()
    device_did, device_sign = _ed()
    binding = mint_binding(
        master_sign=master_sign, master_did=master_did,
        subkey_did=device_did, issued_at=1,
    )
    recipient_dev = e2e.generate_device_sealing_key()
    return master_did, device_did, device_sign, binding, recipient_dev


def test_e2e_round_trip_full_chain():
    master_did, device_did, device_sign, binding, recipient_dev = _full_chain()
    sealed = e2e.seal_message(
        payload={"text": "host can't read this"},
        recipient_device_sealing_pub_b64=e2e.device_sealing_pub_b64(recipient_dev),
        device_sign=device_sign, device_did=device_did, seq=1, ts=1,
    )
    opened = e2e.open_message(
        sealed, recipient_device_priv=recipient_dev,
        expected_master_did=master_did, device_binding=binding,
    )
    assert opened["payload"] == {"text": "host can't read this"}
    assert opened["device_did"] == device_did


def test_e2e_host_cannot_read():
    master_did, device_did, device_sign, binding, recipient_dev = _full_chain()
    sealed = e2e.seal_message(
        payload={"text": "private"},
        recipient_device_sealing_pub_b64=e2e.device_sealing_pub_b64(recipient_dev),
        device_sign=device_sign, device_did=device_did, seq=1, ts=1,
    )
    assert "private" not in repr(sealed)
    # A different device key (a host's, an eavesdropper's) cannot open it.
    other_dev = e2e.generate_device_sealing_key()
    with pytest.raises(e2e.E2EError):
        e2e.open_message(
            sealed, recipient_device_priv=other_dev,
            expected_master_did=master_did, device_binding=binding,
        )


def test_e2e_rejects_device_not_under_pinned_master():
    _, device_did, device_sign, binding, recipient_dev = _full_chain()
    wrong_master, _ = _ed()
    sealed = e2e.seal_message(
        payload={"x": 1},
        recipient_device_sealing_pub_b64=e2e.device_sealing_pub_b64(recipient_dev),
        device_sign=device_sign, device_did=device_did, seq=1, ts=1,
    )
    # Recipient pinned a DIFFERENT master → the binding won't chain.
    with pytest.raises(e2e.E2EError):
        e2e.open_message(
            sealed, recipient_device_priv=recipient_dev,
            expected_master_did=wrong_master, device_binding=binding,
        )


def test_e2e_rejects_binding_for_other_device():
    master_did, master_sign = _ed()
    device_did, device_sign = _ed()
    other_device, _ = _ed()
    recipient_dev = e2e.generate_device_sealing_key()
    # Binding vouches for a DIFFERENT device than the one that signs.
    binding = mint_binding(
        master_sign=master_sign, master_did=master_did,
        subkey_did=other_device, issued_at=1,
    )
    sealed = e2e.seal_message(
        payload={"x": 1},
        recipient_device_sealing_pub_b64=e2e.device_sealing_pub_b64(recipient_dev),
        device_sign=device_sign, device_did=device_did, seq=1, ts=1,
    )
    with pytest.raises(e2e.E2EError):
        e2e.open_message(
            sealed, recipient_device_priv=recipient_dev,
            expected_master_did=master_did, device_binding=binding,
        )


# ── at-rest ──────────────────────────────────────────────────────────


def test_at_rest_round_trip():
    key = derive_at_rest_key(b"a-high-entropy-stored-secret-32b!")
    blob = encrypt_at_rest(b"stored message content", key)
    assert b"stored message" not in base64.b64decode(blob["ct"])  # not cleartext
    assert decrypt_at_rest(blob, key) == b"stored message content"


def test_at_rest_wrong_key_fails():
    key1 = derive_at_rest_key(b"secret-one-secret-one-secret!!!!")
    key2 = derive_at_rest_key(b"secret-two-secret-two-secret!!!!")
    blob = encrypt_at_rest(b"x", key1)
    with pytest.raises(AtRestError):
        decrypt_at_rest(blob, key2)


def test_at_rest_tamper_detected():
    key = derive_at_rest_key(b"secret-secret-secret-secret!!!!!")
    blob = encrypt_at_rest(b"x", key)
    ct = bytearray(base64.b64decode(blob["ct"]))
    ct[0] ^= 0xFF
    blob["ct"] = base64.b64encode(bytes(ct)).decode("ascii")
    with pytest.raises(AtRestError):
        decrypt_at_rest(blob, key)


def test_at_rest_rejects_weak_secret():
    with pytest.raises(AtRestError):
        derive_at_rest_key(b"short")
