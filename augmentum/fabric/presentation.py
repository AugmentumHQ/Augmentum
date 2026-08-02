"""The human-friendly presentation layer for federation.

Everything underneath this speaks in did:keys, postures, TOFU pins, and
exception classes. None of that should ever reach a non-technical user.
This module is the single place that turns the technical truth into warm,
plain-language copy — so the API and UI present one consistent, professional
voice instead of each surface inventing its own.

Pure functions, no I/O. Give it the raw state, get back what a person
should see.
"""

from __future__ import annotations

import hashlib
from typing import Any

from augmentum.fabric.didkey import decode_ed25519_did

# Trust states, in plain language. The copy is deliberately reassuring
# but honest — "Not verified yet" invites action without alarming.
_STATES = {
    "verified": {
        "label": "Verified",
        "hint": "You confirmed this is really them.",
        "tone": "good",
        "icon": "✓",
    },
    "unverified": {
        "label": "Not verified yet",
        "hint": "Tap Verify to be sure no one is impersonating them.",
        "tone": "warn",
        "icon": "•",
    },
    "changed": {
        "label": "Identity changed",
        "hint": "Their security code changed. Re-verify before you trust this contact.",
        "tone": "alert",
        "icon": "!",
    },
}


def connection_presentation(peer: dict[str, Any]) -> dict[str, str]:
    """Turn a peer record into the trust UI the human should see.

    ``peer`` is the serialized peer dict (verified bool, optional
    key_changed). Returns ``{state, label, hint, tone, icon}`` — ready to
    render as a chip + tooltip with no technical terms.
    """
    if peer.get("key_changed"):
        state = "changed"
    elif peer.get("verified"):
        state = "verified"
    else:
        state = "unverified"
    return {"state": state, **_STATES[state]}


def identity_code(did_key: str) -> str:
    """A short, stable, human-readable 'safety code' for an identity.

    The full did:key is unreadable; this is the friendly form people can
    glance at and compare — 12 hex chars from the key fingerprint, grouped
    like a product key (``A1B2-C3D4-E5F6``). Stable for a given key, so two
    people seeing the same code are looking at the same identity.

    Returns '' for a malformed did so callers can fall back gracefully.
    """
    try:
        raw = decode_ed25519_did(did_key)
    except ValueError:
        return ""
    digest = hashlib.sha256(raw).hexdigest().upper()[:12]
    return "-".join(digest[i:i + 4] for i in range(0, 12, 4))


# Friendly copy for the things that can go wrong, keyed by the technical
# reason strings the backend produces (admission reasons, gate reasons,
# error classes). The default keeps it calm and non-blaming.
_FRIENDLY_ERRORS = {
    "instance_denylisted": "You blocked this server, so its messages don't come through.",
    "posture_private": "Your settings only let people you've added reach you.",
    "posture_allowlist": "Only contacts you've approved can reach you right now.",
    "knocked": "Someone new asked to connect — find them under Requests.",
    "ContactCardError": "That invite couldn't be read. Ask them to send a fresh one.",
    "contact-card signature verification failed":
        "That invite looks tampered with. Ask them to send a fresh one.",
    "forged": "We couldn't confirm who that was from, so we didn't deliver it.",
    "replay": "That message was a duplicate and was skipped.",
    "unknown_user": "That person isn't on this server.",
    "fabric disabled": "Connect isn't turned on yet.",
}


def friendly_error(reason: str) -> str:
    """Map a backend reason/exception-name to calm, plain user copy.

    Falls back to a friendly generic message rather than leaking a class
    name or stack detail. Matches on an exact key first, then a
    case-insensitive substring so wrapped messages still resolve.
    """
    if not reason:
        return "Something didn't go through. Please try again."
    if reason in _FRIENDLY_ERRORS:
        return _FRIENDLY_ERRORS[reason]
    low = reason.lower()
    for key, copy in _FRIENDLY_ERRORS.items():
        if key.lower() in low:
            return copy
    return "Something didn't go through. Please try again."


def connection_blurb(peer: dict[str, Any]) -> str:
    """A one-line, human summary of a contact for list rows / headers,
    e.g. 'Alice · Verified' or 'bob@host · Not verified yet'."""
    name = peer.get("display_name") or peer.get("handle") or "Unknown"
    pres = connection_presentation(peer)
    return f"{name} · {pres['label']}"
