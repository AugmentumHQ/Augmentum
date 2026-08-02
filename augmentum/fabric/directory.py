"""Optional directory module: self-certifying Number + descriptor (P4).

DORMANT BY DEFAULT. After D4, the dialable Number and the directory were
cut from the out-of-box path (contact cards are the trust root). This
module exists only for communities that *choose* to run a directory —
and it is built so that even a malicious directory **cannot impersonate**
anyone; the worst it can do is refuse to answer.

Two primitives make that true:

  * **Number** — a one-way, ≥80-bit fingerprint of the identity key,
    derived purely from the did:key bytes. 24 payload digits + a Luhn
    check digit. ``verify_number`` re-derives it, so a directory can't
    hand you a Number that doesn't match the key it points at.
  * **Descriptor** — what a directory stores under a Number: the did:key
    + a reachable endpoint, **signed by the instance identity key**.
    ``verify_descriptor`` checks (a) the signature validates under the
    descriptor's own did:key AND (b) ``derive_number(did_key) == number``.
    So a tampered descriptor fails the signature (attacker lacks the
    key) and a mismatched Number fails the derivation. Wrong endpoint ⇒
    the signed fabric handshake fails ⇒ denial, never MITM.

≥80 bits of canonical width is what makes the Number non-grindable (the
v1 43-bit lesson); it only matters when a directory does bare-Number
lookup, which is why it lives here in the optional module.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from augmentum.fabric.canonical import canonical_bytes
from augmentum.fabric.didkey import decode_ed25519_did

_NUMBER_DIGITS = 24            # ~79.7 bits — the anti-grind canonical width
_NUMBER_DOMAIN = b"augmentum-fabric-number-v1"
_DESCRIPTOR_VERSION = 1


class DescriptorError(ValueError):
    """Raised when a directory descriptor is malformed, unsigned, or its
    Number doesn't match its key."""


def _luhn_check_digit(digits: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - (total % 10)) % 10)


def derive_number(did_key: str) -> str:
    """did:key → self-certifying 25-digit Number (24 payload + 1 Luhn).

    Derived only from the key bytes, so anyone can recompute it; a
    directory cannot mint a Number for a key it doesn't actually hold a
    descriptor for that would survive :func:`verify_number`.
    """
    raw = decode_ed25519_did(did_key)
    digest = hashlib.sha256(_NUMBER_DOMAIN + raw).digest()
    payload = f"{int.from_bytes(digest, 'big') % (10 ** _NUMBER_DIGITS):0{_NUMBER_DIGITS}d}"
    return payload + _luhn_check_digit(payload)


def verify_number(did_key: str, number: str) -> bool:
    """True iff ``number`` is the canonical Number for ``did_key``.
    Whitespace/grouping tolerant. Never raises."""
    digits = "".join(c for c in number if c.isdigit())
    try:
        return digits == derive_number(did_key)
    except ValueError:
        return False


def format_number(number: str) -> str:
    """Group a 25-digit Number into IBAN-style 5-digit blocks for display."""
    digits = "".join(c for c in number if c.isdigit())
    return " ".join(digits[i:i + 5] for i in range(0, len(digits), 5))


# ── descriptor (what a directory stores; self-certifying) ────────────


def mint_descriptor(
    *,
    sign,
    did_key: str,
    endpoint: str,
    issued_at: int,
) -> dict[str, Any]:
    """Build a signed directory descriptor for this instance.

    ``sign`` is the instance identity's Ed25519 signer; ``did_key`` is
    that identity. The Number is derived (not supplied) so it always
    matches the key.
    """
    statement = {
        "ctx": "augmentum-fabric-directory-descriptor-v1",  # domain separation
        "v": _DESCRIPTOR_VERSION,
        "number": derive_number(did_key),
        "did_key": did_key,
        "endpoint": endpoint,
        "issued_at": int(issued_at),
    }
    sig = sign(canonical_bytes(statement))
    return {**statement, "sig": base64.b64encode(sig).decode("ascii")}


def verify_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Verify a descriptor's signature AND Number↔key binding.

    Returns the validated ``{number, did_key, endpoint, issued_at}``.
    Raises :class:`DescriptorError` on any failure. A malicious directory
    that altered the endpoint fails the signature; one that altered the
    Number or key fails the derivation check.
    """
    if not isinstance(descriptor, dict) or descriptor.get("v") != _DESCRIPTOR_VERSION:
        raise DescriptorError("unsupported or malformed descriptor")
    sig_b64 = descriptor.get("sig")
    if not isinstance(sig_b64, str) or not sig_b64:
        raise DescriptorError("descriptor missing signature")

    did_key = str(descriptor.get("did_key", ""))
    number = str(descriptor.get("number", ""))
    statement = {
        "ctx": "augmentum-fabric-directory-descriptor-v1",  # domain separation
        "v": _DESCRIPTOR_VERSION,
        "number": number,
        "did_key": did_key,
        "endpoint": str(descriptor.get("endpoint", "")),
        "issued_at": int(descriptor.get("issued_at", 0)),
    }
    try:
        pub_raw = decode_ed25519_did(did_key)
        Ed25519PublicKey.from_public_bytes(pub_raw).verify(
            base64.b64decode(sig_b64), canonical_bytes(statement)
        )
    except Exception as exc:
        raise DescriptorError("descriptor signature verification failed") from exc

    if not verify_number(did_key, number):
        raise DescriptorError("descriptor Number does not match its did:key")

    return {
        "number": number,
        "did_key": did_key,
        "endpoint": statement["endpoint"],
        "issued_at": statement["issued_at"],
    }
