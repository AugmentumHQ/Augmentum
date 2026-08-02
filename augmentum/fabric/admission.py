"""The inbound admission choke-point for the RELAY path (SEC-11 keystone).

Augmentum federation has TWO inbound transports with TWO trust models, and
this module serves ONE of them — read this before assuming it's bypassed:

  * **Live point-to-point WS** (the path wired today):
    ``connect/fabric_inbound.py`` -> ``connect/federation_gate.gate_inbound``.
    The Ed25519-verified peer channel + the pinned ``fabric_nodes`` key +
    hostname binding ARE the auth; the verified instance vouches for its own
    local-parts, and per-message replay is N/A on an authenticated
    point-to-point link. That gate enforces instance denylist/revocation +
    the deny-by-default stranger posture. See its module docstring.

  * **Store-and-forward RELAY** (this module — a primitive ahead of its
    consumer; not yet wired into a live relay handler, exercised by
    ``tests/test_fabric_admission.py``): here there is NO authenticated
    channel — anyone can submit a relayed blob — so the caller's did:key
    signature MUST be checked against the body claim and replays MUST be
    blocked by seq. ``authenticate_and_admit`` is the foot-gun-free
    choke-point for THAT path: it sources identity only from the verified
    envelope and runs every gate in the right order.

When the relay handler is built, call ``authenticate_and_admit`` for every
inbound RELAYED frame (message, call invite, knock) BEFORE acting on it. It:

  1. derives the authoritative caller did:key from the **envelope-verified
     signer key** and rejects any mismatching body-claimed source
     (caller_id / §8);
  2. refuses revoked / denylisted identities (revocation);
  3. rejects replays via the durable per-(owner,source) seq guard
     (durable_guards / SEC-8) when a seq is supplied;
  4. admits a **pinned** peer (carrying its verified/unverified trust
     label for the UI — D1-01);
  5. otherwise applies the recipient's admission **posture**
     (private / allowlist / knock / open).

It never rings, surfaces, or writes message content — it returns a
decision; the caller acts on it. Knock enqueuing (intro-withheld, rate
limited) is left to ``knock.submit_knock`` so this stays a pure gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.fabric.caller_id import CallerIdForgeryError, assert_caller
from augmentum.fabric.durable_guards import check_and_advance_seq
from augmentum.fabric.knock import VALID_POSTURES
from augmentum.fabric.peer_identity_store import get_peer
from augmentum.fabric.revocation import is_denied
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

# Decision actions.
ADMIT = "admit"      # known/allowed — deliver
KNOCK = "knock"      # stranger under knock posture — enqueue a knock
DENIED = "denied"    # posture/denylist refused
FORGED = "forged"    # body source_did != envelope-verified signer
REPLAY = "replay"    # seq replay


@dataclass(frozen=True)
class AdmissionDecision:
    action: str            # one of the constants above
    source_did: str        # the AUTHORITATIVE caller did:key ('' if forged)
    reason: str
    pinned: bool = False
    verified: bool = False
    trust_label: str = ""

    @property
    def allowed(self) -> bool:
        return self.action == ADMIT


async def authenticate_and_admit(
    conn: aiosqlite.Connection,
    *,
    verified_pubkey: bytes | str,
    claimed_source_did: str | None,
    to_user_id: str,
    recipient_posture: str,
    allowlisted: bool = False,
    seq: int | None = None,
    owner_id: str = "",
) -> AdmissionDecision:
    """Single inbound gate. ``verified_pubkey`` is the signer key the
    envelope middleware already verified — NOT anything from the body.
    ``claimed_source_did`` is the body's self-asserted source (optional);
    it must match or the frame is FORGED.

    ``recipient_posture`` is the target user's ``fabric_admission_posture``
    (private/allowlist/knock/open); ``allowlisted`` says whether the
    authoritative source is on that user's allowlist (the caller resolves
    it). Supply ``seq`` (+ ``owner_id`` for per-user E2E) to enforce the
    durable replay guard.
    """
    # 1. Authoritative caller-ID from the VERIFIED key. Forged claim → out.
    try:
        source_did = assert_caller(verified_pubkey, claimed_source_did)
    except CallerIdForgeryError:
        log.warning("fabric_admission_forged_caller_id", to_user_id=to_user_id)
        return AdmissionDecision(FORGED, "", "claimed source does not match signer")

    # 2. Revoked / denylisted identities never get in.
    if await is_denied(conn, source_did):
        return AdmissionDecision(DENIED, source_did, "source is revoked or denylisted")

    # 3. Replay guard (durable). Only when the frame carries a seq.
    if seq is not None:
        fresh = await check_and_advance_seq(
            conn, source_did=source_did, seq=seq, owner_id=owner_id,
        )
        if not fresh:
            return AdmissionDecision(REPLAY, source_did, "stale or duplicate sequence")

    # 4. Pinned peer → admit, carrying the trust label the UI must show.
    peer = await get_peer(conn, user_id=to_user_id, peer_did_key=source_did)
    if peer is not None:
        return AdmissionDecision(
            ADMIT, source_did, "known contact",
            pinned=True, verified=peer.verified, trust_label=peer.trust_label,
        )

    # 5. Stranger → posture decides.
    posture = recipient_posture if recipient_posture in VALID_POSTURES else "knock"
    if posture == "private":
        return AdmissionDecision(DENIED, source_did, "recipient accepts no strangers")
    if posture == "allowlist":
        if allowlisted:
            return AdmissionDecision(ADMIT, source_did, "allowlisted stranger")
        return AdmissionDecision(DENIED, source_did, "stranger not on allowlist")
    if posture == "open":
        return AdmissionDecision(
            ADMIT, source_did, "open posture — auto-surfaced",
            pinned=False, verified=False, trust_label="pinned, not verified",
        )
    # Default: knock — caller should enqueue via knock.submit_knock.
    return AdmissionDecision(KNOCK, source_did, "stranger — queue a knock")
