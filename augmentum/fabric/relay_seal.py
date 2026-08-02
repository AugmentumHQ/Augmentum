"""Sealed relay envelopes — pyca-only sign-then-seal (P3).

When two instances can't connect directly (CGNAT, both behind NAT), a
relay forwards bytes between them. D3 mandates the relay sees only
ciphertext, and the second red-team pinned three things the naive plan
got wrong:

  * **RSC-3 — named libsodium primitives don't exist in pyca.**
    ``crypto_box_seal`` / ``crypto_sign_ed25519_pk_to_curve25519`` are
    PyNaCl, not pyca/cryptography (our only crypto dep). So we build the
    seal from pyca parts: ephemeral **X25519** ECDH → **HKDF-SHA256** →
    **ChaCha20-Poly1305** AEAD.
  * **RSC-2 — don't reuse the Ed25519 identity key for ECDH.** Converting
    the signing key to X25519 is the footgun that bit Tor/Matrix. Each
    instance mints a SEPARATE X25519 *sealing* key; that key's public
    half is what others seal to.
  * **RSC-1 — sealing strips authenticity.** Encrypting alone destroys
    the §8 caller-ID guarantee (anyone can encrypt to you). So we
    **sign-then-seal**: the origin signs the payload with its Ed25519
    identity key, and that signature travels *inside* the seal. The
    recipient unseals, then verifies the inner signature against the
    claimed ``source_did`` before trusting it. Caller-ID survives the
    relay; the relay still sees nothing.

Replay defense: a monotonic ``seq`` + timestamp inside the sealed inner
payload (use :class:`ReplayWindow` to enforce per-source monotonicity).

Wire (outer, what the relay sees — all opaque)::

    {"v":1, "eph_pub": b64, "nonce": b64, "ct": b64}

Inner (decrypted, authenticated)::

    {"source_did": "...", "seq": n, "ts": t, "payload": {...},
     "origin_sig": b64(Ed25519 over canonical{source_did,seq,ts,payload})}
"""

from __future__ import annotations

import base64
import os
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from augmentum.fabric.canonical import canonical_bytes
from augmentum.fabric.didkey import decode_ed25519_did

_SEAL_VERSION = 1
_SEAL_CTX = "augmentum-fabric-relay-seal-v1"  # domain separation tag
_HKDF_INFO = b"augmentum-fabric-relay-seal-v1"


class SealError(ValueError):
    """Raised when a sealed envelope can't be opened or fails authenticity."""


# ── separate X25519 sealing key (RSC-2) ──────────────────────────────


def generate_sealing_key() -> x25519.X25519PrivateKey:
    """Mint a fresh X25519 sealing key (separate from the Ed25519
    identity). Each instance keeps one; its public half is published in
    contact cards / .well-known for others to seal to."""
    return x25519.X25519PrivateKey.generate()


def sealing_pub_b64(priv: x25519.X25519PrivateKey) -> str:
    raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def _derive_key(shared: bytes, eph_pub: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=eph_pub,        # ephemeral pub as salt (fresh per message)
        info=_HKDF_INFO,
    ).derive(shared)


# ── sign-then-seal (RSC-1) ───────────────────────────────────────────


def seal(
    *,
    payload: Any,
    recipient_sealing_pub_b64: str,
    origin_sign,
    source_did: str,
    seq: int,
    ts: int,
) -> dict[str, Any]:
    """Sign ``payload`` with the origin identity, then seal it to the
    recipient's X25519 sealing key.

    ``origin_sign`` is the origin instance identity's ``bytes -> bytes``
    Ed25519 signer; ``source_did`` is that identity's did:key. The
    signature is computed over the *plaintext* and carried inside the
    seal, so the recipient can attribute the message after decrypting.
    """
    decode_ed25519_did(source_did)  # fail loudly if malformed

    # Bind the INTENDED RECIPIENT into the signed statement (the recipient's
    # sealing pubkey). Without this the origin signature only attests
    # "source said payload" — not "to this recipient" — so a legitimate
    # recipient could re-seal the origin-signed bytes to a third party who
    # would believe the origin addressed them (surreptitious forwarding).
    inner_signed = {
        "ctx": _SEAL_CTX,
        "source_did": source_did,
        "recipient_seal": recipient_sealing_pub_b64,
        "seq": int(seq),
        "ts": int(ts),
        "payload": payload,
    }
    origin_sig = origin_sign(canonical_bytes(inner_signed))
    inner = {**inner_signed, "origin_sig": base64.b64encode(origin_sig).decode("ascii")}
    inner_bytes = canonical_bytes(inner)

    recipient_pub = x25519.X25519PublicKey.from_public_bytes(
        base64.b64decode(recipient_sealing_pub_b64)
    )
    eph_priv = x25519.X25519PrivateKey.generate()
    eph_pub = eph_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    shared = eph_priv.exchange(recipient_pub)
    key = _derive_key(shared, eph_pub)

    nonce = os.urandom(12)
    # AAD binds the version + ephemeral pub so neither can be swapped.
    aad = canonical_bytes({"v": _SEAL_VERSION, "eph": base64.b64encode(eph_pub).decode("ascii")})
    ct = ChaCha20Poly1305(key).encrypt(nonce, inner_bytes, aad)
    return {
        "v": _SEAL_VERSION,
        "eph_pub": base64.b64encode(eph_pub).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ct": base64.b64encode(ct).decode("ascii"),
    }


def unseal(
    sealed: dict[str, Any],
    *,
    recipient_sealing_priv: x25519.X25519PrivateKey,
) -> dict[str, Any]:
    """Open a sealed envelope and verify its inner origin signature.

    Returns ``{"source_did", "seq", "ts", "payload"}`` — but ONLY after
    the inner Ed25519 signature validates against ``source_did``. So the
    returned ``source_did`` is authenticated (RSC-1), not self-asserted.
    Raises :class:`SealError` on any decrypt or authenticity failure.
    """
    if not isinstance(sealed, dict) or sealed.get("v") != _SEAL_VERSION:
        raise SealError("unsupported or malformed sealed envelope")
    try:
        eph_pub = base64.b64decode(sealed["eph_pub"])
        nonce = base64.b64decode(sealed["nonce"])
        ct = base64.b64decode(sealed["ct"])
    except Exception as exc:
        raise SealError("malformed sealed fields") from exc

    try:
        eph_pub_key = x25519.X25519PublicKey.from_public_bytes(eph_pub)
        shared = recipient_sealing_priv.exchange(eph_pub_key)
        key = _derive_key(shared, eph_pub)
        aad = canonical_bytes({"v": _SEAL_VERSION, "eph": base64.b64encode(eph_pub).decode("ascii")})
        inner_bytes = ChaCha20Poly1305(key).decrypt(nonce, ct, aad)
    except Exception as exc:
        raise SealError("seal decryption failed") from exc

    import json
    try:
        inner = json.loads(inner_bytes)
        source_did = str(inner["source_did"])
        origin_sig = base64.b64decode(inner["origin_sig"])
        recipient_seal = str(inner["recipient_seal"])
        inner_signed = {
            "ctx": _SEAL_CTX,
            "source_did": source_did,
            "recipient_seal": recipient_seal,
            "seq": int(inner["seq"]),
            "ts": int(inner["ts"]),
            "payload": inner["payload"],
        }
    except Exception as exc:
        raise SealError("malformed inner payload") from exc

    # Confirm WE are the intended recipient: the signed recipient_seal must
    # be our own sealing public key. Defeats surreptitious re-sealing — a
    # message addressed (and signed) to someone else won't validate here
    # even though we could decrypt a copy re-sealed to our key.
    own_pub_b64 = sealing_pub_b64(recipient_sealing_priv)
    if recipient_seal != own_pub_b64:
        raise SealError("sealed message was addressed to a different recipient")

    # Verify the inner origin signature — the caller-ID authentication
    # that survives the relay (RSC-1).
    try:
        pub_raw = decode_ed25519_did(source_did)
        Ed25519PublicKey.from_public_bytes(pub_raw).verify(
            origin_sig, canonical_bytes(inner_signed)
        )
    except Exception as exc:
        raise SealError("inner origin signature failed — unauthenticated") from exc

    return {
        "source_did": source_did,
        "seq": inner_signed["seq"],
        "ts": inner_signed["ts"],
        "payload": inner_signed["payload"],
    }


class ReplayWindow:
    """Per-source monotonic-seq replay guard.

    The recipient feeds each unsealed ``(source_did, seq)`` here; a seq
    less-than-or-equal-to the highest already seen from that source is a
    replay and rejected. Contiguity isn't required (relays may reorder),
    but monotonic progress is — a stored, strictly-increasing high-water
    mark per source. In-memory here; the durable store is the caller's.
    """

    def __init__(self) -> None:
        self._high: dict[str, int] = {}

    def check_and_advance(self, source_did: str, seq: int) -> bool:
        """Return True if ``seq`` is fresh (and record it); False if it's
        a replay/stale."""
        last = self._high.get(source_did)
        if last is not None and seq <= last:
            return False
        self._high[source_did] = seq
        return True
