"""End-to-end device-to-device sealing (P5).

The host-trusted model (both servers can read content) is the honest v1
posture. P5 closes it for direct messages: messages are sealed
**device to device**, so neither the sender's host nor the recipient's
host can read them — only the recipient's device key opens them.

This composes the earlier phases rather than inventing new crypto:

  * the **sign-then-seal** primitive from P3 (``relay_seal``) does the
    confidentiality + inner-signature work, but here the keys are
    per-DEVICE (X25519 for sealing, Ed25519 subkey for signing), not the
    per-instance keys — that's what takes the host out of the trust path.
  * the **author binding** from P2 proves the sending device's subkey is
    vouched for by the user's master author key, which the recipient
    pinned + verified in the P1 ceremony. So opening a message yields a
    device id that chains: device → master → ceremony-verified human.

Full chain on ``open_message`` success: content was unreadable to both
hosts (device X25519), the sender device actually signed it (inner
Ed25519), and that device is authorised by the pinned master (binding).
"""

from __future__ import annotations

from typing import Any

from augmentum.fabric import relay_seal
from augmentum.fabric.author_binding import verify_binding
from augmentum.fabric.didkey import did_equal


class E2EError(ValueError):
    """Raised when an E2E message can't be opened or its device chain
    doesn't validate."""


def generate_device_sealing_key():
    """A device's X25519 key for receiving E2E messages (published in
    the user's contact card alongside the signing subkey)."""
    return relay_seal.generate_sealing_key()


def device_sealing_pub_b64(priv) -> str:
    return relay_seal.sealing_pub_b64(priv)


def seal_message(
    *,
    payload: Any,
    recipient_device_sealing_pub_b64: str,
    device_sign,
    device_did: str,
    seq: int,
    ts: int,
) -> dict[str, Any]:
    """Sender device signs ``payload`` and seals it to the recipient
    DEVICE key. ``device_sign`` is the sending device subkey's Ed25519
    signer; ``device_did`` is that subkey's did:key."""
    return relay_seal.seal(
        payload=payload,
        recipient_sealing_pub_b64=recipient_device_sealing_pub_b64,
        origin_sign=device_sign,
        source_did=device_did,
        seq=seq,
        ts=ts,
    )


def open_message(
    sealed: dict[str, Any],
    *,
    recipient_device_priv,
    expected_master_did: str,
    device_binding: dict[str, Any],
) -> dict[str, Any]:
    """Open an E2E message and validate the full device→master chain.

    Steps (any failure raises :class:`E2EError`):
      1. unseal with the recipient device key + verify the inner device
         signature (host couldn't read; device really signed).
      2. verify the sending device's subkey is bound to
         ``expected_master_did`` — the master the recipient pinned and
         verified in the ceremony.
      3. confirm the bound subkey IS the device that signed (no
         binding-for-a-different-device swap).

    Returns ``{"payload", "device_did", "seq", "ts"}``.
    """
    try:
        inner = relay_seal.unseal(sealed, recipient_sealing_priv=recipient_device_priv)
    except relay_seal.SealError as exc:
        raise E2EError(f"could not open E2E message: {exc}") from exc

    device_did = inner["source_did"]
    try:
        bound_subkey = verify_binding(
            device_binding, expected_master_did=expected_master_did
        )
    except Exception as exc:
        raise E2EError(f"sending device is not authorised by the pinned master: {exc}") from exc

    if not did_equal(bound_subkey, device_did):
        raise E2EError("author binding is for a different device than the signer")

    return {
        "payload": inner["payload"],
        "device_did": device_did,
        "seq": inner["seq"],
        "ts": inner["ts"],
    }
