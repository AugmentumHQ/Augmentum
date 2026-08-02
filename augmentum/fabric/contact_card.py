"""Signed contact cards — the default federation trust root (P1).

D4 deleted the global directory, so the **contact card** (a link or QR
the inviter shares) is how two instances first learn each other's
identity. A card carries the issuing instance's did:key, a reachable
endpoint, the inviting user's author key, a handle, and an opaque token,
all signed by the **instance** key.

What the signature DOES prove: integrity in transit + that the minting
party held the instance private key at mint time (self-certifying — the
card claims "I am did X" and signs with X). What it does NOT prove: that
X is the party the human *means* to talk to. A malicious host mints a
valid card for a key it controls. So a freshly-parsed card is pinned
**"verified=False"** (TOFU) and only the out-of-band ceremony
(``ceremony.py``) upgrades it. The honest invite labeling (INV-2) and
the binding verified-state UI (D1-01) depend on never conflating the two.

Wire format (``canonical.py`` byte-canonical, then Ed25519-signed)::

    {
      "v": 1,
      "instance_did_key": "did:key:z...",   # canonical issuer identity
      "endpoint": "https://host:port",      # mutable reachability hint
      "author_did_key": "did:key:z...",     # inviting user's authenticity key
      "handle": "alice@host",               # display only, NOT a trust input
      "token": "<opaque bearer>",           # invite/admission token
      "issued_at": 1718000000
    }
    + "sig": base64(Ed25519 over canonical_bytes(payload-without-sig))
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from augmentum.fabric.canonical import canonical_bytes
from augmentum.fabric.didkey import decode_ed25519_did, encode_ed25519_did

_CARD_VERSION = 1


class ContactCardError(ValueError):
    """Raised when a contact card is malformed or its signature is invalid."""


@dataclass(frozen=True)
class ContactCard:
    instance_did_key: str
    endpoint: str
    author_did_key: str
    handle: str
    token: str
    issued_at: int

    def _payload(self) -> dict[str, Any]:
        return {
            "ctx": "augmentum-fabric-contact-card-v1",  # domain separation
            "v": _CARD_VERSION,
            "instance_did_key": self.instance_did_key,
            "endpoint": self.endpoint,
            "author_did_key": self.author_did_key,
            "handle": self.handle,
            "token": self.token,
            "issued_at": self.issued_at,
        }


def mint_card(
    *,
    sign,
    instance_did_key: str,
    endpoint: str,
    author_did_key: str,
    handle: str,
    token: str,
    issued_at: int,
) -> dict[str, Any]:
    """Build a signed contact-card dict ready to embed in a link/QR.

    ``sign`` is a callable ``bytes -> bytes`` (the instance identity's
    :meth:`FabricIdentity.sign`). ``issued_at`` is passed in (no clock
    here — ``Date.now`` is unavailable in some run contexts and explicit
    timestamps keep minting deterministic/testable).
    """
    # did:keys must be well-formed before we sign (fail loudly at mint,
    # not silently at parse on the other side).
    decode_ed25519_did(instance_did_key)
    if author_did_key:
        decode_ed25519_did(author_did_key)

    card = ContactCard(
        instance_did_key=instance_did_key,
        endpoint=endpoint,
        author_did_key=author_did_key,
        handle=handle,
        token=token,
        issued_at=int(issued_at),
    )
    payload = card._payload()
    sig = sign(canonical_bytes(payload))
    return {**payload, "sig": base64.b64encode(sig).decode("ascii")}


def parse_card(data: dict[str, Any]) -> ContactCard:
    """Parse + cryptographically verify a contact-card dict.

    Verifies the Ed25519 signature against the card's OWN
    ``instance_did_key`` (self-certifying). Raises :class:`ContactCardError`
    on any malformation or signature failure.

    NOTE: a valid signature means "intact + minted by the holder of this
    key", NOT "trusted". The caller pins the result as verified=False and
    must run the ceremony to upgrade. Do not treat a parsed card as a
    verified identity.
    """
    if not isinstance(data, dict):
        raise ContactCardError("card must be a JSON object")
    if data.get("v") != _CARD_VERSION:
        raise ContactCardError(f"unsupported card version {data.get('v')!r}")

    sig_b64 = data.get("sig")
    if not isinstance(sig_b64, str) or not sig_b64:
        raise ContactCardError("card missing signature")

    try:
        instance_did = str(data["instance_did_key"])
        card = ContactCard(
            instance_did_key=instance_did,
            endpoint=str(data.get("endpoint", "")),
            author_did_key=str(data.get("author_did_key", "")),
            handle=str(data.get("handle", "")),
            token=str(data.get("token", "")),
            issued_at=int(data["issued_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContactCardError(f"malformed card fields: {exc}") from None

    try:
        pub_raw = decode_ed25519_did(card.instance_did_key)
    except ValueError as exc:
        raise ContactCardError(f"bad instance_did_key: {exc}") from None

    try:
        sig = base64.b64decode(sig_b64)
        Ed25519PublicKey.from_public_bytes(pub_raw).verify(
            sig, canonical_bytes(card._payload())
        )
    except Exception as exc:
        raise ContactCardError("contact-card signature verification failed") from exc

    return card


def normalize_did(did: str) -> str:
    """Re-encode a did:key to its canonical form (raises if malformed).

    Used before pinning so the stored value is canonical and
    byte-comparison via ``did_equal`` is reliable.
    """
    return encode_ed25519_did(decode_ed25519_did(did))
