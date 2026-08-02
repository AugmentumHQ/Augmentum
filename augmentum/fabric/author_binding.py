"""Per-user authenticity keys: master → device-subkey bindings (P2).

The AK-1 / D2 fix. A user's authenticity shouldn't rest on the instance
key (the host can forge all its users) NOR on a single device key (lose
the phone, lose your identity). So each user has a long-lived **master
author key**; each device gets its own **subkey**, and the master signs
a binding statement vouching for that subkey. Verifiers trust a subkey
iff it carries a valid binding to a master they've pinned (folded into
the verification ceremony, so a host can't substitute the master).

This module is the binding crypto + a 3-state trust badge. Master-key
custody (generation, where the private master lives, web vs device) is
deliberately out of scope here and disclosed as a residual — this builds
the *mechanism* the custody layer will use.

Binding statement (``canonical.py`` bytes, master-signed)::

    {"v":1,"master_did":"did:key:z...","subkey_did":"did:key:z...",
     "purpose":"device","issued_at":<int>}
    + "sig": base64(master Ed25519 over canonical_bytes(statement))
"""

from __future__ import annotations

import base64
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from augmentum.fabric.canonical import canonical_bytes
from augmentum.fabric.didkey import decode_ed25519_did, did_equal

_BINDING_VERSION = 1

# Trust badge states the UI renders for an author key (3-state, AK-3).
BADGE_UNBOUND = "unbound"          # no binding presented (e.g. web-custody loss)
BADGE_MASTER_BOUND = "master_bound"  # valid binding to a pinned master
BADGE_BROKEN = "broken"            # binding present but invalid → treat as hostile


class AuthorBindingError(ValueError):
    """Raised when an author binding is malformed or its signature fails."""


def mint_binding(
    *,
    master_sign,
    master_did: str,
    subkey_did: str,
    issued_at: int,
    purpose: str = "device",
) -> dict[str, Any]:
    """Master vouches for a subkey. ``master_sign`` is the master key's
    ``bytes -> bytes`` signer. Returns the signed binding dict."""
    decode_ed25519_did(master_did)
    decode_ed25519_did(subkey_did)
    statement = {
        "ctx": "augmentum-fabric-author-binding-v1",  # domain separation
        "v": _BINDING_VERSION,
        "master_did": master_did,
        "subkey_did": subkey_did,
        "purpose": purpose,
        "issued_at": int(issued_at),
    }
    sig = master_sign(canonical_bytes(statement))
    return {**statement, "sig": base64.b64encode(sig).decode("ascii")}


def verify_binding(binding: dict[str, Any], *, expected_master_did: str) -> str:
    """Verify a binding and return the bound subkey did:key.

    Checks: version, the statement's ``master_did`` byte-matches
    ``expected_master_did`` (the master the verifier pinned — NOT a
    self-asserted one), and the signature validates under that master.
    Raises :class:`AuthorBindingError` on any failure.
    """
    if not isinstance(binding, dict) or binding.get("v") != _BINDING_VERSION:
        raise AuthorBindingError("unsupported or malformed binding")
    sig_b64 = binding.get("sig")
    if not isinstance(sig_b64, str) or not sig_b64:
        raise AuthorBindingError("binding missing signature")

    master_did = str(binding.get("master_did", ""))
    if not did_equal(master_did, expected_master_did):
        raise AuthorBindingError("binding master_did does not match the pinned master")

    statement = {
        "ctx": "augmentum-fabric-author-binding-v1",  # domain separation
        "v": _BINDING_VERSION,
        "master_did": master_did,
        "subkey_did": str(binding.get("subkey_did", "")),
        "purpose": str(binding.get("purpose", "device")),
        "issued_at": int(binding.get("issued_at", 0)),
    }
    try:
        pub_raw = decode_ed25519_did(master_did)
        sig = base64.b64decode(sig_b64)
        Ed25519PublicKey.from_public_bytes(pub_raw).verify(
            sig, canonical_bytes(statement)
        )
    except AuthorBindingError:
        raise
    except Exception as exc:
        raise AuthorBindingError("binding signature verification failed") from exc

    return statement["subkey_did"]


def author_badge(binding: dict[str, Any] | None, *, expected_master_did: str) -> str:
    """3-state trust badge for an author key (never raises).

    * ``BADGE_UNBOUND`` — no binding presented. Could be a fresh/web key
      or a host that stripped the binding; the UI treats it as "not
      proven", not "trusted".
    * ``BADGE_MASTER_BOUND`` — valid binding to the pinned master.
    * ``BADGE_BROKEN`` — a binding was presented but is invalid: treat as
      hostile (a tampered/forged binding), louder than merely unbound.
    """
    if not binding:
        return BADGE_UNBOUND
    try:
        verify_binding(binding, expected_master_did=expected_master_did)
        return BADGE_MASTER_BOUND
    except AuthorBindingError:
        return BADGE_BROKEN
