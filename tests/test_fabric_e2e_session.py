"""Conversation E2E: simple direct path + companion-on-standby safety.

The load-bearing test here is the standby gate: a requested companion is
NOT sealed to while COMPANION_E2E_SECURITY_CONFIRMED is False — that's
the guarantee the whole 'on standby until we confirm security' design
rests on.
"""
from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from augmentum.fabric import e2e, e2e_session
from augmentum.fabric.author_binding import mint_binding
from augmentum.fabric.didkey import encode_ed25519_did


def _signing():
    priv = Ed25519PrivateKey.generate()
    raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return encode_ed25519_did(raw), priv.sign


def _device():
    """A recipient device: signing did + X25519 sealing key + priv."""
    dev_did, _ = _signing()
    sealing = e2e.generate_device_sealing_key()
    return e2e_session.Recipient(
        device_did=dev_did,
        sealing_pub_b64=e2e.device_sealing_pub_b64(sealing),
    ), sealing


def _sender_with_binding():
    """A sender: master key, a device subkey bound to it, and the binding."""
    master_did, master_sign = _signing()
    device_did, device_sign = _signing()
    binding = mint_binding(
        master_sign=master_sign, master_did=master_did,
        subkey_did=device_did, issued_at=1,
    )
    return master_did, device_did, device_sign, binding


# ── direct 1:1 ───────────────────────────────────────────────────────


def test_direct_one_to_one_round_trip():
    master_did, dev_did, sign, binding = _sender_with_binding()
    bob, bob_priv = _device()

    res = e2e_session.resolve_recipients([bob])
    assert len(res.recipients) == 1 and res.companion_active is False

    sealed = e2e_session.seal_for_recipients(
        payload={"text": "just us two"}, recipients=res.recipients,
        sender_sign=sign, sender_device_did=dev_did, seq=1, ts=1,
    )
    opened = e2e_session.open_for_me(
        sealed, my_device_did=bob.device_did, my_device_priv=bob_priv,
        sender_master_did=master_did, sender_device_binding=binding,
    )
    assert opened["payload"] == {"text": "just us two"}


def test_multi_device_recipient():
    # Bob has two devices; the message is sealed to both, each opens its own.
    master_did, dev_did, sign, binding = _sender_with_binding()
    phone, phone_priv = _device()
    laptop, laptop_priv = _device()

    res = e2e_session.resolve_recipients([phone, laptop])
    sealed = e2e_session.seal_for_recipients(
        payload={"m": 1}, recipients=res.recipients,
        sender_sign=sign, sender_device_did=dev_did, seq=1, ts=1,
    )
    assert set(sealed) == {phone.device_did, laptop.device_did}
    for dev, priv in ((phone, phone_priv), (laptop, laptop_priv)):
        out = e2e_session.open_for_me(
            sealed, my_device_did=dev.device_did, my_device_priv=priv,
            sender_master_did=master_did, sender_device_binding=binding,
        )
        assert out["payload"] == {"m": 1}


# ── companion-on-standby (the safety guarantee) ──────────────────────


def test_companion_blocked_while_security_unconfirmed():
    # The gate is False by default — a REQUESTED companion is NOT included.
    assert e2e_session.COMPANION_E2E_SECURITY_CONFIRMED is False
    bob, _ = _device()
    companion, _ = _device()

    res = e2e_session.resolve_recipients(
        [bob], companion=companion, companion_requested=True,
    )
    assert res.companion_active is False
    assert res.reason == "companion_on_standby_security_unconfirmed"
    # The companion device is NOT in the recipient set — it cannot be
    # sealed to, so it cannot read anything.
    assert companion.device_did not in {r.device_did for r in res.recipients}


def test_companion_not_added_when_not_requested():
    bob, _ = _device()
    companion, _ = _device()
    res = e2e_session.resolve_recipients([bob], companion=companion)
    assert res.companion_active is False
    assert res.reason == "companion_not_requested"


def test_companion_joins_only_when_gate_lifted(monkeypatch):
    # Simulate a reviewed, signed-off security confirmation.
    monkeypatch.setattr(
        e2e_session, "COMPANION_E2E_SECURITY_CONFIRMED", True, raising=False,
    )
    master_did, dev_did, sign, binding = _sender_with_binding()
    bob, _ = _device()
    companion, comp_priv = _device()

    res = e2e_session.resolve_recipients(
        [bob], companion=companion, companion_requested=True,
    )
    assert res.companion_active is True
    assert any(r.is_companion for r in res.recipients)

    sealed = e2e_session.seal_for_recipients(
        payload={"text": "ai may read"}, recipients=res.recipients,
        sender_sign=sign, sender_device_did=dev_did, seq=1, ts=1,
    )
    # Now the companion CAN open its envelope (proving the gate is the
    # only thing that was holding it back — the plumbing is real).
    out = e2e_session.open_for_me(
        sealed, my_device_did=companion.device_did, my_device_priv=comp_priv,
        sender_master_did=master_did, sender_device_binding=binding,
    )
    assert out["payload"] == {"text": "ai may read"}


def test_unsealed_device_cannot_open():
    # A device the message was NOT sealed to gets nothing.
    master_did, dev_did, sign, binding = _sender_with_binding()
    bob, _ = _device()
    outsider, outsider_priv = _device()
    sealed = e2e_session.seal_for_recipients(
        payload={"x": 1}, recipients=[bob],
        sender_sign=sign, sender_device_did=dev_did, seq=1, ts=1,
    )
    with pytest.raises(e2e.E2EError):
        e2e_session.open_for_me(
            sealed, my_device_did=outsider.device_did,
            my_device_priv=outsider_priv,
            sender_master_did=master_did, sender_device_binding=binding,
        )
