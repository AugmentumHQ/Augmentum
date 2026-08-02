"""Signed delivery receipts over CONTIGUOUS sequence (P4 hardening).

A relay-path medium finding: a signed receipt that just says "I got a
message" is theater — it doesn't prove the stream wasn't silently
truncated. A meaningful receipt must attest that the recipient has every
message up to a sequence number with **no gaps**. So:

  * ``contiguous_high_water`` collapses a set of received seqs to the
    highest N such that 1..N are ALL present. A gap caps the receipt
    there, so a dropped message can't be acked past.
  * ``mint_receipt`` signs ``{source_did, up_to_seq, ts}`` with the
    recipient's identity key; ``verify_receipt`` checks it under the
    recipient's did:key. The sender uses it to know what's durably
    delivered and what to retransmit.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from augmentum.fabric.canonical import canonical_bytes
from augmentum.fabric.didkey import decode_ed25519_did

_RECEIPT_VERSION = 1


class ReceiptError(ValueError):
    """Raised when a delivery receipt is malformed or its signature fails."""


def contiguous_high_water(received_seqs: Iterable[int]) -> int:
    """Highest N such that every seq in 1..N was received. 0 if seq 1 is
    missing. A gap stops the count — you cannot ack past a hole."""
    seen = set(int(s) for s in received_seqs)
    n = 0
    while (n + 1) in seen:
        n += 1
    return n


def mint_receipt(
    *,
    sign,
    recipient_did: str,
    source_did: str,
    up_to_seq: int,
    ts: int,
) -> dict[str, Any]:
    """Recipient signs an ack of contiguous delivery up to ``up_to_seq``
    for the stream from ``source_did``. ``sign`` is the recipient's
    identity signer; ``recipient_did`` is that identity."""
    decode_ed25519_did(recipient_did)
    decode_ed25519_did(source_did)
    statement = {
        "ctx": "augmentum-fabric-receipt-v1",  # domain separation
        "v": _RECEIPT_VERSION,
        "recipient_did": recipient_did,
        "source_did": source_did,
        "up_to_seq": int(up_to_seq),
        "ts": int(ts),
    }
    sig = sign(canonical_bytes(statement))
    return {**statement, "sig": base64.b64encode(sig).decode("ascii")}


def verify_receipt(receipt: dict[str, Any], *, expected_recipient_did: str) -> int:
    """Verify a receipt is signed by ``expected_recipient_did`` and return
    the acked contiguous ``up_to_seq``. Raises :class:`ReceiptError`."""
    if not isinstance(receipt, dict) or receipt.get("v") != _RECEIPT_VERSION:
        raise ReceiptError("unsupported or malformed receipt")
    sig_b64 = receipt.get("sig")
    if not isinstance(sig_b64, str) or not sig_b64:
        raise ReceiptError("receipt missing signature")

    recipient_did = str(receipt.get("recipient_did", ""))
    from augmentum.fabric.didkey import did_equal
    if not did_equal(recipient_did, expected_recipient_did):
        raise ReceiptError("receipt recipient_did mismatch")

    statement = {
        "ctx": "augmentum-fabric-receipt-v1",  # domain separation
        "v": _RECEIPT_VERSION,
        "recipient_did": recipient_did,
        "source_did": str(receipt.get("source_did", "")),
        "up_to_seq": int(receipt.get("up_to_seq", 0)),
        "ts": int(receipt.get("ts", 0)),
    }
    try:
        pub_raw = decode_ed25519_did(recipient_did)
        Ed25519PublicKey.from_public_bytes(pub_raw).verify(
            base64.b64decode(sig_b64), canonical_bytes(statement)
        )
    except Exception as exc:
        raise ReceiptError("receipt signature verification failed") from exc
    return statement["up_to_seq"]
