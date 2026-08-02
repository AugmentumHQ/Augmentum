"""At-rest encryption for stored federated content (P5).

A relay-path medium finding: even with the wire sealed, message content
was written to the DB in cleartext, so a disk seizure / backup leak
exposed everything the transport protected. P5 wraps stored content in an
authenticated at-rest envelope.

Symmetric ChaCha20-Poly1305 under a per-instance at-rest key. The key is
derived (HKDF) from a stored high-entropy secret (the same
``encrypt_api_key`` custody the identity key uses), so it's not the raw
secret that lands in code or config. AEAD means tamper of the stored
blob is detected on read, not silently returned.

Pure pyca; no new dependency.
"""

from __future__ import annotations

import base64
import os
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_AT_REST_VERSION = 1
_HKDF_INFO = b"augmentum-fabric-at-rest-v1"


class AtRestError(ValueError):
    """Raised when an at-rest blob can't be decrypted (wrong key or
    tampered ciphertext)."""


def derive_at_rest_key(secret: bytes) -> bytes:
    """Derive a 32-byte at-rest key from a stored high-entropy secret."""
    if len(secret) < 16:
        raise AtRestError("at-rest secret must be >= 16 bytes of entropy")
    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=_HKDF_INFO,
    ).derive(secret)


def encrypt_at_rest(plaintext: bytes, key: bytes) -> dict[str, Any]:
    """Encrypt ``plaintext`` for storage. Returns a JSON-safe envelope."""
    nonce = os.urandom(12)
    aad = bytes([_AT_REST_VERSION])
    ct = ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)
    return {
        "v": _AT_REST_VERSION,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ct": base64.b64encode(ct).decode("ascii"),
    }


def decrypt_at_rest(blob: dict[str, Any], key: bytes) -> bytes:
    """Decrypt a stored envelope. Raises :class:`AtRestError` on a wrong
    key or any tamper (AEAD authentication failure)."""
    if not isinstance(blob, dict) or blob.get("v") != _AT_REST_VERSION:
        raise AtRestError("unsupported or malformed at-rest blob")
    try:
        nonce = base64.b64decode(blob["nonce"])
        ct = base64.b64decode(blob["ct"])
        aad = bytes([_AT_REST_VERSION])
        return ChaCha20Poly1305(key).decrypt(nonce, ct, aad)
    except Exception as exc:
        raise AtRestError("at-rest decryption failed") from exc
