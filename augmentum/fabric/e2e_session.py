"""Conversation-level E2E: direct first, AI-participant on standby.

Built on the device-to-device primitive in ``e2e.py``. The model is
deliberately simple: an E2E message is sealed to a *list of recipient
devices*. For a direct 1:1 chat that list is just the other person's
device(s) — nothing fancy. The same list is what would later let the
user's own sovereign AI join a conversation as an additional endpoint.

THE SAFETY DESIGN (what the user asked for): the companion-as-participant
capability is fully wired here, but it is **on standby behind a hard code
gate** — ``COMPANION_E2E_SECURITY_CONFIRMED``. While that constant is
False, the companion can NEVER be added to an E2E conversation's
recipients, *even if an operator turns on the user-facing setting*. That
is intentional defense-in-depth: a single setting flip must not be able
to silently insert a third party (even the user's own AI) into an
end-to-end conversation before the threat model is reviewed and signed
off. Flipping the constant is a deliberate, reviewed act — not a toggle.

Note on true host-blindness: sealing must ultimately run on the CLIENT
for the host to be blind. This module is the recipient/policy model and
wire shape the client will drive; running it server-side gives
encrypted-in-transit/at-rest but not host-blind E2E. The standby gate is
correct either way.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from augmentum.fabric import e2e
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# HARD STANDBY GATE. Do NOT flip to True until the "companion as an
# authorized E2E device" threat model has been independently reviewed and
# signed off. This constant — not any user setting — is the real switch.
COMPANION_E2E_SECURITY_CONFIRMED = False
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Recipient:
    """One endpoint a message is sealed to. For direct E2E this is the
    peer's device; for the (gated) companion path, the companion's
    device. Sealing only needs the device's X25519 sealing pubkey; the
    sender's author binding is supplied at OPEN time, not here."""
    device_did: str          # Ed25519 signing-subkey did:key (addressing)
    sealing_pub_b64: str     # X25519 sealing pubkey to encrypt to
    label: str = ""          # human label, e.g. "Bob's phone" / "Companion"
    is_companion: bool = False


@dataclass(frozen=True)
class RecipientResolution:
    recipients: list[Recipient]
    companion_active: bool
    reason: str


def resolve_recipients(
    human_recipients: list[Recipient],
    *,
    companion: Recipient | None = None,
    companion_requested: bool = False,
) -> RecipientResolution:
    """Decide who an E2E message is sealed to.

    ALWAYS includes the human peer's device(s). The companion is included
    ONLY when **both** the operator requested it AND the hard security
    gate is lifted. Otherwise the companion stays out and we report it as
    on standby — the message is still plain direct E2E.
    """
    recipients = list(human_recipients)

    if companion is None or not companion_requested:
        return RecipientResolution(recipients, False, "companion_not_requested")

    if not COMPANION_E2E_SECURITY_CONFIRMED:
        # Requested, but the security gate is closed — stay on standby.
        log.info("e2e_companion_on_standby_security_unconfirmed")
        return RecipientResolution(
            recipients, False, "companion_on_standby_security_unconfirmed",
        )

    recipients.append(replace(companion, is_companion=True))
    return RecipientResolution(recipients, True, "companion_included")


def seal_for_recipients(
    *,
    payload: Any,
    recipients: list[Recipient],
    sender_sign,
    sender_device_did: str,
    seq: int,
    ts: int,
) -> dict[str, dict[str, Any]]:
    """Seal one message to every recipient device.

    Returns ``{recipient_device_did: sealed_blob}``. For a direct 1:1
    chat this is a single entry. Each blob is independently
    sign-then-sealed (the sender signs once per recipient; relay_seal
    binds each to that recipient's key — no cross-recipient replay).
    """
    out: dict[str, dict[str, Any]] = {}
    for r in recipients:
        out[r.device_did] = e2e.seal_message(
            payload=payload,
            recipient_device_sealing_pub_b64=r.sealing_pub_b64,
            device_sign=sender_sign,
            device_did=sender_device_did,
            seq=seq,
            ts=ts,
        )
    return out


def open_for_me(
    sealed_by_device: dict[str, dict[str, Any]],
    *,
    my_device_did: str,
    my_device_priv,
    sender_master_did: str,
    sender_device_binding: dict[str, Any],
) -> dict[str, Any]:
    """Open the envelope sealed to THIS device and validate the sender.

    ``sender_master_did`` is the master key you verified for the sender in
    the ceremony; ``sender_device_binding`` proves the sending device is
    authorised by that master. Raises ``e2e.E2EError`` if no envelope was
    sealed to us or the chain doesn't validate.
    """
    blob = sealed_by_device.get(my_device_did)
    if blob is None:
        raise e2e.E2EError("no E2E envelope was sealed to this device")
    return e2e.open_message(
        blob,
        recipient_device_priv=my_device_priv,
        expected_master_did=sender_master_did,
        device_binding=sender_device_binding,
    )
