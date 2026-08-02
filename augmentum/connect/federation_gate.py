"""Inbound admission gate for Connect-over-fabric (live-path wiring).

This is where the federated-PBX trust layer meets the EXISTING Connect
transport. The live path addresses peers as ``user@hostname`` and
verifies the *sending instance* via its pinned Ed25519 key (the fabric
envelope signature, already checked before we run). The new did:key /
contact-card / knock layer is a parallel model; this gate bridges them
without breaking the working transport.

Design constraints honored:

  * **Default-off.** When ``fabric_federation_enabled`` is False (the
    default), the gate allows everything — existing installs are 100%
    unchanged. It only enforces when an operator opts in.
  * **Authenticated identity = the verified instance.** We derive the
    sending instance's did:key from its pinned pubkey (never from the
    frame body) and key denylist/revocation on it (SEC-11 discipline).
  * **Deny-by-default for strangers.** Relationship-creating verbs
    (first message / call invite) from a sender the recipient has no
    contact with are gated by the recipient instance's
    ``fabric_admission_posture``; under ``knock`` they're queued
    (intro-withheld) instead of delivered.
  * **Known contacts bypass.** An existing ``connect_contacts`` row is
    the live-path equivalent of a pin — those flow through untouched.

Per-message replay is NOT enforced here: the live WS is an authenticated
point-to-point connection, not a store-and-forward relay, so the relay
replay guard (relay_seal/durable_guards) doesn't apply to this path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augmentum.config import settings
from augmentum.connect.protocol import MSG_INVITE, MSG_TEXT_SEND
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

# Verbs that establish a NEW relationship (and thus deserve posture
# gating). Everything else references an existing thread/call and is a
# no-op without one, so it doesn't need the stranger gate.
_RELATIONSHIP_VERBS = frozenset({MSG_TEXT_SEND, MSG_INVITE})


@dataclass(frozen=True)
class GateResult:
    allow: bool
    reason: str = ""
    knocked: bool = False  # True iff a knock was queued (handled, not delivered)


async def resolve_instance_did(
    conn: aiosqlite.Connection, sender_node_id: str,
) -> str:
    """did:key of the verified sending instance, from its pinned pubkey.

    Reads the Ed25519 key pinned in ``fabric_nodes`` for the
    signature-verified ``sender_node_id`` — never anything from the frame
    body. Returns '' if the node isn't pinned (can't happen on a verified
    frame, but we fail safe by skipping the did-keyed checks)."""
    if not sender_node_id:
        return ""
    try:
        cur = await conn.execute(
            "SELECT pubkey_ed25519 FROM fabric_nodes WHERE id=?",
            (sender_node_id,),
        )
        row = await cur.fetchone()
    except Exception:
        return ""
    if not row or not row[0]:
        return ""
    from augmentum.fabric.caller_id import authoritative_source_did
    try:
        return authoritative_source_did(str(row[0]))
    except Exception:
        return ""


async def gate_inbound(
    conn: aiosqlite.Connection,
    *,
    sender_node_id: str,
    source_did: str,
    target_user_id: str,
    verb: str,
    body: str = "",
) -> GateResult:
    """Decide whether an inbound Connect-over-fabric frame may proceed.

    Returns ``GateResult(allow=True)`` to deliver normally;
    ``allow=False`` to drop (with a reason; ``knocked=True`` when the
    stranger was queued as a knock rather than refused outright).
    """
    # Default-off: existing installs see no change.
    if not getattr(settings, "fabric_federation_enabled", False):
        return GateResult(allow=True)

    instance_did = await resolve_instance_did(conn, sender_node_id)

    # 1. Instance-level denylist / revocation — drops every frame from a
    #    peer instance the operator blocked or that revoked its key.
    if instance_did:
        from augmentum.fabric.revocation import is_denied
        try:
            if await is_denied(conn, instance_did):
                return GateResult(allow=False, reason="instance_denylisted")
        except Exception as exc:
            log.warning("connect_gate_denylist_check_failed", error=str(exc)[:160])

    # 2. Posture only applies to relationship-creating verbs from a
    #    sender the recipient doesn't already know.
    if verb not in _RELATIONSHIP_VERBS:
        return GateResult(allow=True)

    from augmentum.connect.contact_store import get_contact
    try:
        known = await get_contact(
            conn, user_id=target_user_id, peer_did=source_did,
        )
    except Exception:
        known = None
    if known is not None:
        return GateResult(allow=True)  # existing contact = the live-path pin

    posture = getattr(settings, "fabric_admission_posture", "knock") or "knock"
    if posture == "open":
        return GateResult(allow=True)  # preserves pre-federation behavior
    if posture in ("private", "allowlist"):
        # allowlist has no live-path source beyond contacts (already
        # checked), so it behaves like private for unknown senders.
        return GateResult(allow=False, reason=f"posture_{posture}")

    # 3. knock posture: queue an intro-withheld request, drop delivery.
    #    The authenticated identity is the instance did:key; the claimed
    #    user@host rides as the (untrusted) handle.
    if instance_did:
        from augmentum.fabric.knock import KnockRefused, submit_knock
        try:
            await submit_knock(
                conn,
                to_user_id=target_user_id,
                from_did_key=instance_did,
                posture="knock",
                from_handle=source_did,
                intro_text=(body or "")[:280],
            )
            log.info(
                "connect_gate_knock_queued",
                to_user_id=target_user_id, from_handle=source_did,
            )
        except KnockRefused as refused:
            log.info(
                "connect_gate_knock_refused",
                to_user_id=target_user_id, reason=refused.reason,
            )
        except Exception as exc:
            log.warning("connect_gate_knock_failed", error=str(exc)[:160])
    # Either way the stranger's frame is not delivered — that IS the
    # deny-by-default behavior.
    return GateResult(allow=False, reason="knocked", knocked=True)


def gate_result_dict(verb: str, result: GateResult) -> dict[str, Any]:
    """Shape a dropped-frame GateResult as the inbound dispatcher's
    standard result dict. ``applied`` is True for a queued knock (it was
    handled, just not delivered) and False for a hard refusal."""
    return {
        "applied": result.knocked,
        "verb": verb,
        "error": result.reason,
        "gated": True,
    }
