"""Deny-by-default knock admission (P2).

The knock tier is how a stranger (an identity the recipient has NOT
pinned) reaches a user — and it is deliberately the weakest path:

  * **Posture-gated.** The recipient's ``fabric_admission_posture``
    decides whether a knock is even accepted: ``private`` (none),
    ``allowlist`` (only pre-approved keys), ``knock`` (default — accept
    into a pending queue), ``open`` (accept + auto-surface).
  * **No ring.** A pending knock never triggers a call/notification.
  * **Intro withheld.** The intro text is stored but not surfaced until
    the recipient accepts — a stranger can't push 280 chars at you
    pre-consent (SDC-3 / the Signal message-request model).
  * **Rate-limited on scarce axes.** Per source did:key AND per source
    IP, plus a per-recipient pending ceiling. Limits are on things that
    cost the attacker (IP), not on the free-to-mint identity alone.

Accepting a knock pins the source identity (TOFU, unverified) so normal
contact-card verification can follow.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.fabric.contact_card import normalize_did
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

VALID_POSTURES = ("private", "allowlist", "knock", "open")
DEFAULT_POSTURE = "knock"  # D5: gated-knock default

# Rate-limit ceilings (the scarce-axis fix). Tunable later via settings.
_MAX_PENDING_PER_SOURCE = 1   # one outstanding knock per source identity
_MAX_PENDING_PER_IP = 5       # per source IP across all targets
_MAX_PENDING_PER_RECIPIENT = 50


class KnockRefused(Exception):
    """A knock was refused by posture or rate limit (not an error — the
    structural spam defense doing its job). ``reason`` is machine-readable."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason


@dataclass(frozen=True)
class Knock:
    id: str
    to_user_id: str
    from_did_key: str
    from_handle: str
    status: str
    intro_flagged: bool


def _row_to_knock(row) -> Knock:
    return Knock(
        id=row[0], to_user_id=row[1], from_did_key=row[2],
        from_handle=row[3], status=row[4], intro_flagged=bool(row[5]),
    )


_PUBLIC_COLS = "id, to_user_id, from_did_key, from_handle, status, intro_flagged"


async def _count(conn: aiosqlite.Connection, where: str, params: tuple) -> int:
    cur = await conn.execute(
        f"SELECT COUNT(*) FROM fabric_knocks WHERE {where}", params
    )
    row = await cur.fetchone()
    return int(row[0]) if row else 0


async def submit_knock(
    conn: aiosqlite.Connection,
    *,
    to_user_id: str,
    from_did_key: str,
    posture: str,
    from_handle: str = "",
    intro_text: str = "",
    intro_flagged: bool = False,
    src_ip: str = "",
    allowlisted: bool = False,
) -> Knock:
    """Submit a knock, enforcing posture + rate limits. Raises
    :class:`KnockRefused` when the structural defenses reject it.

    ``posture`` is the recipient's admission posture. ``allowlisted``
    is True iff ``from_did_key`` is on the recipient's allowlist (the
    caller resolves that; this module just enforces the rule).
    """
    if not to_user_id:
        raise ValueError("submit_knock requires a recipient user_id")
    canonical = normalize_did(from_did_key)

    if posture not in VALID_POSTURES:
        posture = DEFAULT_POSTURE

    if posture == "private":
        raise KnockRefused("posture_private", "recipient accepts no knocks")
    if posture == "allowlist" and not allowlisted:
        raise KnockRefused("not_allowlisted", "recipient accepts only known keys")

    # Rate limits (scarce axes first). Count only PENDING — accepted/
    # rejected knocks don't hold a slot.
    per_source = await _count(
        conn, "from_did_key=? AND status='pending'", (canonical,)
    )
    if per_source >= _MAX_PENDING_PER_SOURCE:
        raise KnockRefused("source_rate_limited", "outstanding knock already pending")

    if src_ip:
        per_ip = await _count(
            conn, "src_ip=? AND status='pending'", (src_ip,)
        )
        if per_ip >= _MAX_PENDING_PER_IP:
            raise KnockRefused("ip_rate_limited", "too many pending knocks from this address")

    per_recipient = await _count(
        conn, "to_user_id=? AND status='pending'", (to_user_id,)
    )
    if per_recipient >= _MAX_PENDING_PER_RECIPIENT:
        raise KnockRefused("recipient_queue_full", "recipient's knock queue is full")

    knock_id = uuid.uuid4().hex
    await conn.execute(
        "INSERT INTO fabric_knocks "
        "(id, to_user_id, from_did_key, from_handle, intro_text, intro_flagged, src_ip) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (knock_id, to_user_id, canonical, from_handle, intro_text,
         1 if intro_flagged else 0, src_ip),
    )
    await conn.commit()
    log.info(
        "fabric_knock_received",
        to_user_id=to_user_id, from_did_key=canonical,
        posture=posture, flagged=intro_flagged,
    )
    return Knock(
        id=knock_id, to_user_id=to_user_id, from_did_key=canonical,
        from_handle=from_handle, status="pending", intro_flagged=intro_flagged,
    )


async def list_pending(conn: aiosqlite.Connection, *, to_user_id: str) -> list[Knock]:
    """List a user's pending knocks. Intro text is NOT included — it
    stays withheld until accept (use :func:`accept_knock` to reveal)."""
    cur = await conn.execute(
        f"SELECT {_PUBLIC_COLS} FROM fabric_knocks "
        "WHERE to_user_id=? AND status='pending' ORDER BY created_at DESC",
        (to_user_id,),
    )
    return [_row_to_knock(r) for r in await cur.fetchall()]


async def accept_knock(
    conn: aiosqlite.Connection, *, to_user_id: str, knock_id: str,
) -> dict:
    """Accept a knock: flip status, REVEAL the withheld intro, and TOFU-pin
    the source identity so normal verification can follow.

    Returns ``{"from_did_key", "intro_text", "pinned"}``. Raises
    ValueError if the knock isn't this user's pending knock.
    """
    cur = await conn.execute(
        "SELECT from_did_key, from_handle, intro_text FROM fabric_knocks "
        "WHERE id=? AND to_user_id=? AND status='pending'",
        (knock_id, to_user_id),
    )
    row = await cur.fetchone()
    if row is None:
        raise ValueError("no such pending knock for this user")
    from_did, from_handle, intro_text = row[0], row[1], row[2]

    await conn.execute(
        "UPDATE fabric_knocks SET status='accepted', decided_at=datetime('now') "
        "WHERE id=? AND to_user_id=?",
        (knock_id, to_user_id),
    )
    await conn.commit()

    # Pin the now-known identity (still unverified — ceremony comes next).
    from augmentum.fabric.peer_identity_store import pin_peer
    pinned = await pin_peer(
        conn, user_id=to_user_id, peer_did_key=from_did,
        handle=from_handle, source="knock",
    )
    log.info("fabric_knock_accepted", to_user_id=to_user_id, from_did_key=from_did)
    return {
        "from_did_key": from_did,
        "intro_text": intro_text,  # revealed only now
        "pinned": pinned,
    }


async def reject_knock(
    conn: aiosqlite.Connection, *, to_user_id: str, knock_id: str,
) -> bool:
    """Reject a pending knock. Returns True if one was rejected."""
    cur = await conn.execute(
        "UPDATE fabric_knocks SET status='rejected', decided_at=datetime('now') "
        "WHERE id=? AND to_user_id=? AND status='pending'",
        (knock_id, to_user_id),
    )
    await conn.commit()
    return cur.rowcount > 0
