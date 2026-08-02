"""Authoritative caller-ID for inbound federated frames (P2).

The §8 / INV-2 fix. An inbound Connect frame (message, call invite,
knock) arrives over a peer connection whose envelope signature the
middleware already verified — so we KNOW which instance key actually
signed it. The frame BODY may additionally *claim* a ``source_did``.
Those two must agree, or the body is forged.

The rule, enforced here and nowhere else (single source of truth):

    The caller-ID is the did:key of the **envelope-verified** signer.
    A body-claimed source that does not byte-match it is rejected. We
    never trust a self-asserted ``source_did``.

This is pure: it takes the verified signer's public key (from the
middleware's pin lookup) and the body's claim, and returns the
authoritative did:key or raises. The DB/middleware wiring lives in the
route layer; keeping the rule pure makes it exhaustively testable.
"""

from __future__ import annotations

import base64

from augmentum.fabric.didkey import did_equal, encode_ed25519_did


class CallerIdForgeryError(ValueError):
    """Raised when a frame's claimed source_did does not match the
    envelope-verified signer — i.e. a spoofed caller-ID."""


def authoritative_source_did(verified_pubkey: bytes | str) -> str:
    """did:key of the envelope-verified signer.

    Accepts raw 32 bytes or a base64 string (the form stored in
    ``fabric_nodes.pubkey_ed25519``). This — not the body — IS the
    caller-ID.
    """
    raw = (
        base64.b64decode(verified_pubkey)
        if isinstance(verified_pubkey, str)
        else verified_pubkey
    )
    return encode_ed25519_did(raw)


def assert_caller(verified_pubkey: bytes | str, claimed_source_did: str | None) -> str:
    """Return the authoritative caller did:key, refusing a forged claim.

    * Computes the caller-ID from the verified signer key.
    * If the body claims a ``source_did``, it MUST byte-match (via
      :func:`did_equal`); otherwise :class:`CallerIdForgeryError`.
    * An absent/empty claim is fine — we simply stamp the authoritative
      value (the body doesn't get to omit its way out of attribution).

    Comparison is on decoded key bytes, so a different-string-same-key
    claim is accepted and a same-looking-different-key claim is caught.
    """
    authoritative = authoritative_source_did(verified_pubkey)
    if claimed_source_did and not did_equal(claimed_source_did, authoritative):
        raise CallerIdForgeryError(
            "frame source_did does not match the envelope-verified signer"
        )
    return authoritative
