"""Contact store — DAO over connect_contacts.

Phase 1 contacts are simple: a per-user list of peer DIDs the user
has interacted with or explicitly added. The schema (migration 219)
already supports tags, blocked, presence cache, discovery source —
this module exposes a coherent surface around it without leaking
schema details into HTTP handlers.

Discovery is mutual-enablement: when two users with matching
``connect_discoverable_*`` settings come online, both instances
auto-create contact rows (Phase 2 cross-instance wiring). For
Phase 1, contacts are added either:

  - manually via ``add_contact()`` (HTTP POST /contacts)
  - implicitly on first inbound message / call (the routing layer
    can call ``ensure_contact()`` to materialise a row without
    overriding existing data)

Soft-delete isn't useful here — contacts are user-owned data; an
explicit remove should actually remove the row. The "blocked" flag
is the asymmetric override that prevents new traffic without
removing the relationship.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# ── Helpers ────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_contact_id() -> str:
    return uuid.uuid4().hex[:20]


# ── Data shape ─────────────────────────────────────────────────────


@dataclass
class ContactRow:
    contact_id: str
    user_id: str
    peer_did: str
    peer_display_name: str
    peer_avatar_url: str
    discovery_source: str
    last_seen_status: str
    last_seen_at: str | None
    blocked: bool
    tags: list[str]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "user_id": self.user_id,
            "peer_did": self.peer_did,
            "peer_display_name": self.peer_display_name,
            "peer_avatar_url": self.peer_avatar_url,
            "discovery_source": self.discovery_source,
            "last_seen_status": self.last_seen_status,
            "last_seen_at": self.last_seen_at,
            "blocked": self.blocked,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ── CRUD ───────────────────────────────────────────────────────────


async def add_contact(
    conn: Any,
    *,
    user_id: str,
    peer_did: str,
    peer_display_name: str = "",
    peer_avatar_url: str = "",
    discovery_source: str = "handle_added",
    tags: list[str] | None = None,
) -> ContactRow:
    """Add a contact. Idempotent on (user_id, peer_did) — re-adding
    returns the existing row (mirrors ``get_or_create_thread``)."""

    contact_id = new_contact_id()
    now = _now_iso()
    await conn.execute(
        """INSERT OR IGNORE INTO connect_contacts
                (contact_id, user_id, peer_did, peer_display_name,
                 peer_avatar_url, discovery_source,
                 last_seen_status, last_seen_at,
                 blocked, tags, created_at, updated_at)
              VALUES (?, ?, ?, ?, ?, ?, 'offline', NULL,
                      0, ?, ?, ?)""",
        (
            contact_id, user_id, peer_did, peer_display_name,
            peer_avatar_url, discovery_source,
            json.dumps(tags or []),
            now, now,
        ),
    )
    await conn.commit()
    row = await get_contact(conn, user_id=user_id, peer_did=peer_did)
    if row is None:
        raise RuntimeError("connect_contacts insert/read race produced no row")
    return row


async def ensure_contact(
    conn: Any,
    *,
    user_id: str,
    peer_did: str,
    discovery_source: str = "implicit",
) -> ContactRow:
    """Like ``add_contact`` but doesn't overwrite an existing row's
    display name / discovery source. Used by routing on first inbound
    contact so a manually-added contact's display name isn't clobbered
    by a noisy initial message."""

    existing = await get_contact(conn, user_id=user_id, peer_did=peer_did)
    if existing is not None:
        return existing
    return await add_contact(
        conn, user_id=user_id, peer_did=peer_did,
        discovery_source=discovery_source,
    )


async def remember_peer_display_name(
    conn: Any, *, user_id: str, peer_did: str, display_name: str,
) -> None:
    """Cache a peer's human name on the contact row (creating it if absent).

    Used by the fabric inbound dispatcher: a cross-instance peer's username
    lives on THEIR box, so we persist the name they ship with their traffic
    here once, and the contacts/calls list routes resolve it locally. Fills
    the name only when it's currently empty so a user's manual rename of a
    contact isn't clobbered by subsequent inbound traffic. No-op on an empty
    name (never overwrite a real name with nothing)."""

    name = (display_name or "").strip()
    if not name or not peer_did:
        return
    now = _now_iso()
    # Create the row if this is the first contact (discovery_source=fabric);
    # INSERT OR IGNORE leaves an existing row — including its name — untouched.
    await conn.execute(
        """INSERT OR IGNORE INTO connect_contacts
                (contact_id, user_id, peer_did, peer_display_name,
                 peer_avatar_url, discovery_source,
                 last_seen_status, last_seen_at,
                 blocked, tags, created_at, updated_at)
              VALUES (?, ?, ?, ?, '', 'fabric', 'offline', NULL,
                      0, '[]', ?, ?)""",
        (new_contact_id(), user_id, peer_did, name, now, now),
    )
    # Backfill an empty name on a pre-existing row (e.g. one created before
    # name-carriage shipped) without touching a user-set name.
    await conn.execute(
        """UPDATE connect_contacts
              SET peer_display_name = ?, updated_at = ?
            WHERE user_id = ? AND peer_did = ?
              AND (peer_display_name IS NULL OR peer_display_name = '')""",
        (name, now, user_id, peer_did),
    )
    await conn.commit()


async def stale_local_peers(
    conn: Any, *, user_id: str, peer_user_ids: list[str],
) -> set[str]:
    """Which of ``peer_user_ids`` are no longer contactable on THIS instance.

    A contact row outlives the thing it points at. Two ways that happens, both
    observed live:

      - **The account was deleted.** The row survives with the peer's user_id
        dangling, so the list renders an uncallable ``usr_<hex>`` ghost forever.
      - **A guest grant was revoked.** The guest user row remains (it owns
        messages), but every grant carries ``revoked_at`` — the relationship is
        over even though the account exists.

    Only SAME-INSTANCE peers are judged. A fabric peer legitimately has no
    local ``users`` row, so applying this test to one would delete every
    cross-instance contact — the exact opposite of the intent. Callers pass
    local user-parts only.

    Returns the subset that is stale (never raises: a missing guest-grants
    table degrades to "no guest is stale" rather than blanking the list).
    """

    ids = [p for p in dict.fromkeys(peer_user_ids) if p]
    if not ids:
        return set()

    placeholders = ",".join("?" * len(ids))
    try:
        cur = await conn.execute(
            f"SELECT id, COALESCE(role, '') FROM users WHERE id IN ({placeholders})",
            ids,
        )
        live_roles = {row[0]: (row[1] or "") for row in await cur.fetchall()}
    except Exception:  # noqa: BLE001 - no users table (isolated schema) => judge nothing
        # Fail OPEN, not closed. If we cannot see the users table we cannot
        # prove anyone is gone, and guessing "stale" would hide every contact.
        return set()

    # Account gone entirely.
    stale = {pid for pid in ids if pid not in live_roles}

    # Account present but the guest relationship was revoked. Grants are scoped
    # to the HOST (user_id), which is the contact's owner — a guest revoked by
    # this user may still be live for someone else on the box.
    guests = [pid for pid, role in live_roles.items() if role == "guest"]
    if guests:
        gph = ",".join("?" * len(guests))
        try:
            cur = await conn.execute(
                f"""SELECT guest_user_id
                      FROM connect_guest_grants
                     WHERE user_id = ?
                       AND guest_user_id IN ({gph})
                       AND (revoked_at IS NULL OR revoked_at = '')""",
                [user_id, *guests],
            )
            still_granted = {row[0] for row in await cur.fetchall()}
        except Exception:  # noqa: BLE001 - absent table must not blank contacts
            still_granted = set(guests)
        stale.update(g for g in guests if g not in still_granted)

    return stale


async def list_contacts(
    conn: Any, *, user_id: str,
    include_blocked: bool = False,
    tag: str | None = None,
    include_stale: bool = False,
) -> list[ContactRow]:
    """All contacts, alpha by peer_did with blocked sorted last.

    ``include_stale`` keeps rows whose peer is deleted or whose guest grant was
    revoked. It defaults to False so every people list, picker and dialer drops
    dead entries — but call HISTORY passes True, because the name cached on a
    stale contact is the only remaining way to label a past call with someone
    who has since been removed. Hiding a contact and forgetting who they were
    are different operations.
    """

    where = ["user_id = ?"]
    params: list[Any] = [user_id]
    if not include_blocked:
        where.append("blocked = 0")
    cur = await conn.execute(
        f"""SELECT contact_id, user_id, peer_did, peer_display_name,
                   peer_avatar_url, discovery_source,
                   last_seen_status, last_seen_at,
                   blocked, tags, created_at, updated_at
              FROM connect_contacts
             WHERE {' AND '.join(where)}
             ORDER BY blocked ASC, peer_did ASC""",
        params,
    )
    rows = [_row_to_contact(r) for r in await cur.fetchall()]
    if tag:
        rows = [c for c in rows if tag in c.tags]
    if include_stale or not rows:
        return rows

    # Resolve DIDs here rather than in the caller: the local/fabric split is a
    # routing concept, and a caller that got it wrong would silently drop every
    # federated contact.
    from augmentum.connect.contacts import resolve_peer_did

    local_of: dict[str, str] = {}
    for c in rows:
        resolved = resolve_peer_did(c.peer_did)
        if resolved is not None and resolved.kind == "local":
            local_of[c.contact_id] = resolved.address
    if not local_of:
        return rows

    stale = await stale_local_peers(
        conn, user_id=user_id, peer_user_ids=list(local_of.values()),
    )
    if not stale:
        return rows
    return [c for c in rows if local_of.get(c.contact_id) not in stale]


async def get_contact(
    conn: Any, *, user_id: str,
    peer_did: str | None = None,
    contact_id: str | None = None,
) -> ContactRow | None:
    """Look up by peer_did OR contact_id (both scoped by user_id)."""

    if not peer_did and not contact_id:
        raise ValueError("peer_did or contact_id required")
    if peer_did:
        cur = await conn.execute(
            """SELECT contact_id, user_id, peer_did, peer_display_name,
                      peer_avatar_url, discovery_source,
                      last_seen_status, last_seen_at,
                      blocked, tags, created_at, updated_at
                 FROM connect_contacts
                WHERE user_id = ? AND peer_did = ?""",
            (user_id, peer_did),
        )
    else:
        cur = await conn.execute(
            """SELECT contact_id, user_id, peer_did, peer_display_name,
                      peer_avatar_url, discovery_source,
                      last_seen_status, last_seen_at,
                      blocked, tags, created_at, updated_at
                 FROM connect_contacts
                WHERE user_id = ? AND contact_id = ?""",
            (user_id, contact_id),
        )
    row = await cur.fetchone()
    return _row_to_contact(row) if row else None


async def remove_contact(
    conn: Any, *, user_id: str, contact_id: str,
) -> bool:
    """Hard-delete. Returns whether a row was deleted."""

    cur = await conn.execute(
        "DELETE FROM connect_contacts WHERE user_id = ? AND contact_id = ?",
        (user_id, contact_id),
    )
    await conn.commit()
    return cur.rowcount > 0


async def set_blocked(
    conn: Any, *, user_id: str, contact_id: str, blocked: bool,
) -> bool:
    """Toggle the blocked flag. Returns whether a row was updated."""

    cur = await conn.execute(
        """UPDATE connect_contacts
              SET blocked = ?, updated_at = ?
            WHERE user_id = ? AND contact_id = ?""",
        (1 if blocked else 0, _now_iso(), user_id, contact_id),
    )
    await conn.commit()
    return cur.rowcount > 0


async def is_blocked(
    conn: Any, *, user_id: str, peer_did: str,
) -> bool:
    """Has ``user_id`` blocked ``peer_did``?

    Used by message_routing / call_routing on the RECIPIENT side to
    drop inbound traffic from a blocked sender without revealing the
    block to that sender (silent-block semantics — sender sees a
    successful send but never gets delivery/read, peer never gets
    a banner or WS event).

    Returns False when no contact row exists — blocking requires an
    explicit row, so a never-contacted DID is implicitly unblocked.
    """

    if not user_id or not peer_did:
        return False
    cur = await conn.execute(
        "SELECT blocked FROM connect_contacts "
        "WHERE user_id = ? AND peer_did = ? LIMIT 1",
        (user_id, peer_did),
    )
    row = await cur.fetchone()
    return bool(row[0]) if row else False


async def set_tags(
    conn: Any, *, user_id: str, contact_id: str, tags: list[str],
) -> bool:
    cur = await conn.execute(
        """UPDATE connect_contacts
              SET tags = ?, updated_at = ?
            WHERE user_id = ? AND contact_id = ?""",
        (json.dumps(tags), _now_iso(), user_id, contact_id),
    )
    await conn.commit()
    return cur.rowcount > 0


async def update_presence(
    conn: Any, *, user_id: str, peer_did: str,
    status: str, seen_at: str | None = None,
) -> bool:
    """Cache the peer's presence — last-seen snapshot for offline UI
    rendering. Authoritative presence still comes from the WS layer."""

    if status not in ("online", "away", "dnd", "offline"):
        raise ValueError(f"unknown presence status '{status}'")
    cur = await conn.execute(
        """UPDATE connect_contacts
              SET last_seen_status = ?, last_seen_at = ?, updated_at = ?
            WHERE user_id = ? AND peer_did = ?""",
        (status, seen_at or _now_iso(), _now_iso(), user_id, peer_did),
    )
    await conn.commit()
    return cur.rowcount > 0


# ── Row converter ─────────────────────────────────────────────────


def _row_to_contact(row: Any) -> ContactRow:
    raw_tags = row[9] or "[]"
    try:
        tags = json.loads(raw_tags)
        if not isinstance(tags, list):
            tags = []
    except (json.JSONDecodeError, TypeError):
        tags = []
    return ContactRow(
        contact_id=row[0],
        user_id=row[1],
        peer_did=row[2],
        peer_display_name=row[3] or "",
        peer_avatar_url=row[4] or "",
        discovery_source=row[5] or "handle_added",
        last_seen_status=row[6] or "offline",
        last_seen_at=row[7],
        blocked=bool(row[8]),
        tags=[str(t) for t in tags],
        created_at=row[10] or "",
        updated_at=row[11] or "",
    )
