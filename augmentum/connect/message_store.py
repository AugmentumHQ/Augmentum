"""Connect message store — aiosqlite DAO for connect_threads + connect_messages.

Wire-shape contract:

* ``insert_message`` is idempotent on ``(message_id, user_id)`` — re-sending
  the same wire envelope (e.g. on reconnect) is a no-op rather than a
  duplicate row. The DB trigger in migration 219 keeps the thread's
  denormalised tail snapshot (last_message_at / last_message_preview /
  unread_count) in sync automatically.

* ``get_or_create_thread`` is idempotent on ``(user_id, peer_did)``: the
  unique index in migration 219 (idx_connect_threads_pair) collapses
  repeat creates into a single row. Each user has their own copy of
  the thread, even when both users live on the same instance — that's
  the per-user-isolation tax baked into the schema.

* Soft-delete: the row stays (audit trail), ``body`` is cleared, and
  ``deleted_at`` is stamped. ``list_messages_for_thread`` returns
  these rows so the UI can render a tombstone in-place, matching
  Signal / Matrix.

* Read receipts are bulk: ``mark_thread_read`` sets ``read_at`` on every
  unread row in the thread up to and including ``last_read_message_id``,
  clears the thread's ``unread_count``, and emits a single ack — saves
  the chatter of per-message receipts.

Per-user isolation: every function accepts ``user_id`` and scopes by
it. There is no cross-user read path. This module is invoked from
``message_routing.py`` (which inserts BOTH sides' rows on same-
instance message delivery) and from ``connect_routes.py`` (HTTP
read paths).

Per CLAUDE.md: ISO-8601 UTC strings for all timestamps to stay
consistent with the rest of the persistence layer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# ── Helpers ────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_thread_id() -> str:
    """Fresh thread id when a client doesn't supply one."""

    return uuid.uuid4().hex[:24]


def new_message_id() -> str:
    """Fresh message id when a client doesn't supply one."""

    return uuid.uuid4().hex[:24]


# ── Data shapes ────────────────────────────────────────────────────


@dataclass
class ThreadRow:
    thread_id: str
    user_id: str
    peer_did: str
    peer_display_name: str
    last_message_at: str
    last_message_preview: str
    unread_count: int
    muted: bool
    pinned: bool
    archived: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "peer_did": self.peer_did,
            "peer_display_name": self.peer_display_name,
            "last_message_at": self.last_message_at,
            "last_message_preview": self.last_message_preview,
            "unread_count": self.unread_count,
            "muted": self.muted,
            "pinned": self.pinned,
            "archived": self.archived,
            "created_at": self.created_at,
        }


@dataclass
class MessageRow:
    message_id: str
    thread_id: str
    user_id: str
    sender_did: str
    body: str
    format: str
    attachment_ref: str
    reply_to: str
    sent_at: str
    received_at: str
    delivered_at: str | None
    read_at: str | None
    edited_at: str | None
    deleted_at: str | None
    transcript: str
    # Fabric attachment fetch — populated only for messages that
    # crossed an instance boundary with attachment_ref set. None for
    # plain text + same-instance attachment messages.
    attachment_fetch_url: str | None = None
    attachment_fetch_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "sender_did": self.sender_did,
            "body": self.body,
            "format": self.format,
            "attachment_ref": self.attachment_ref,
            "reply_to": self.reply_to,
            "sent_at": self.sent_at,
            "received_at": self.received_at,
            "delivered_at": self.delivered_at,
            "read_at": self.read_at,
            "edited_at": self.edited_at,
            "deleted_at": self.deleted_at,
            "transcript": self.transcript,
            "attachment_fetch_url": self.attachment_fetch_url,
            "attachment_fetch_token": self.attachment_fetch_token,
        }


# ── Thread primitives ──────────────────────────────────────────────


async def get_or_create_thread(
    conn: Any,
    *,
    thread_id: str,
    user_id: str,
    peer_did: str,
    peer_display_name: str = "",
) -> ThreadRow:
    """Idempotent — relies on idx_connect_threads_pair uniqueness.

    When two clients race-create the same (user_id, peer_did) thread
    with different thread_ids, the FIRST winner persists; the second
    is silently discarded by the unique constraint and the row that
    actually exists is returned. Tests rely on this.
    """

    now = _now_iso()
    await conn.execute(
        """INSERT OR IGNORE INTO connect_threads
                (thread_id, user_id, peer_did, peer_display_name,
                 last_message_at, last_message_preview,
                 unread_count, muted, pinned, archived, created_at)
              VALUES (?, ?, ?, ?, ?, '', 0, 0, 0, 0, ?)""",
        (thread_id, user_id, peer_did, peer_display_name, now, now),
    )
    await conn.commit()

    # Re-read by (user_id, peer_did) — handles the race where the
    # caller's thread_id was discarded in favor of an earlier row.
    cur = await conn.execute(
        """SELECT thread_id, user_id, peer_did, peer_display_name,
                  COALESCE(last_message_at, '') AS last_message_at,
                  last_message_preview, unread_count,
                  muted, pinned, archived, created_at
             FROM connect_threads
            WHERE user_id = ? AND peer_did = ?""",
        (user_id, peer_did),
    )
    row = await cur.fetchone()
    if row is None:
        raise RuntimeError("connect_threads insert/read race produced no row")
    return _row_to_thread(row)


async def list_threads_for_user(
    conn: Any,
    *,
    user_id: str,
    include_archived: bool = False,
    limit: int = 100,
) -> list[ThreadRow]:
    """Most-recent first by ``last_message_at`` (with thread_id tie-break
    for tests on hosts with low-resolution clocks)."""

    where = ["user_id = ?"]
    params: list[Any] = [user_id]
    if not include_archived:
        where.append("archived = 0")
    cur = await conn.execute(
        f"""SELECT thread_id, user_id, peer_did, peer_display_name,
                   COALESCE(last_message_at, '') AS last_message_at,
                   last_message_preview, unread_count,
                   muted, pinned, archived, created_at
              FROM connect_threads
             WHERE {' AND '.join(where)}
             ORDER BY pinned DESC,
                      last_message_at DESC,
                      thread_id DESC
             LIMIT ?""",
        (*params, limit),
    )
    return [_row_to_thread(r) for r in await cur.fetchall()]


async def get_thread(
    conn: Any, *, thread_id: str, user_id: str,
) -> ThreadRow | None:
    cur = await conn.execute(
        """SELECT thread_id, user_id, peer_did, peer_display_name,
                  COALESCE(last_message_at, '') AS last_message_at,
                  last_message_preview, unread_count,
                  muted, pinned, archived, created_at
             FROM connect_threads
            WHERE thread_id = ? AND user_id = ?""",
        (thread_id, user_id),
    )
    row = await cur.fetchone()
    return _row_to_thread(row) if row else None


async def set_thread_flag(
    conn: Any, *, thread_id: str, user_id: str,
    flag: str, value: bool,
) -> bool:
    """Toggle one of muted / pinned / archived. Returns whether a row
    was updated (i.e. whether the thread exists for this user)."""

    if flag not in {"muted", "pinned", "archived"}:
        raise ValueError(f"unknown thread flag '{flag}'")
    cur = await conn.execute(
        f"UPDATE connect_threads SET {flag} = ? "
        "WHERE thread_id = ? AND user_id = ?",
        (1 if value else 0, thread_id, user_id),
    )
    await conn.commit()
    return cur.rowcount > 0


# ── Message primitives ─────────────────────────────────────────────


async def insert_message(
    conn: Any,
    *,
    message_id: str,
    thread_id: str,
    user_id: str,
    sender_did: str,
    body: str,
    format: str = "plain",
    attachment_ref: str = "",
    reply_to: str = "",
    sent_at: str | None = None,
    transcript: str = "",
) -> bool:
    """Insert one perspective's row. Returns True if a new row was
    inserted, False if the (message_id, user_id) pair already exists.

    The trigger in migration 219 keeps the thread tail / unread count
    in sync — callers don't need to update connect_threads themselves.
    """

    cur = await conn.execute(
        """INSERT OR IGNORE INTO connect_messages
                (message_id, thread_id, user_id, sender_did,
                 body, format, attachment_ref, reply_to,
                 sent_at, received_at, transcript)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            message_id, thread_id, user_id, sender_did,
            body, format, attachment_ref, reply_to,
            sent_at or _now_iso(),
            _now_iso(),
            transcript,
        ),
    )
    await conn.commit()
    return cur.rowcount > 0


async def list_messages_for_thread(
    conn: Any,
    *,
    thread_id: str,
    user_id: str,
    limit: int = 200,
    before_sent_at: str | None = None,
    after_sent_at: str | None = None,
) -> list[MessageRow]:
    """Newest-first within the thread.

    ``before_sent_at`` paginates older (sent_at of the oldest row
    already loaded). ``after_sent_at`` is the catch-up direction —
    "give me everything strictly newer than this cursor" — used by
    the UI on reconnect to pull messages that arrived while the WS
    was down. Either cursor can be supplied; the order is still
    newest-first so the catch-up caller iterates ``reverse()`` for
    chronological replay.
    """

    where = ["thread_id = ?", "user_id = ?"]
    params: list[Any] = [thread_id, user_id]
    if before_sent_at:
        where.append("sent_at < ?")
        params.append(before_sent_at)
    if after_sent_at:
        where.append("sent_at > ?")
        params.append(after_sent_at)
    cur = await conn.execute(
        f"""SELECT message_id, thread_id, user_id, sender_did,
                   body, format, attachment_ref, reply_to,
                   sent_at, received_at, delivered_at, read_at,
                   edited_at, deleted_at, transcript
              FROM connect_messages
             WHERE {' AND '.join(where)}
             ORDER BY sent_at DESC, message_id DESC
             LIMIT ?""",
        (*params, limit),
    )
    return [_row_to_message(r) for r in await cur.fetchall()]


async def get_message(
    conn: Any, *, message_id: str, user_id: str,
) -> MessageRow | None:
    cur = await conn.execute(
        """SELECT message_id, thread_id, user_id, sender_did,
                  body, format, attachment_ref, reply_to,
                  sent_at, received_at, delivered_at, read_at,
                  edited_at, deleted_at, transcript
             FROM connect_messages
            WHERE message_id = ? AND user_id = ?""",
        (message_id, user_id),
    )
    row = await cur.fetchone()
    return _row_to_message(row) if row else None


async def mark_thread_read(
    conn: Any,
    *,
    thread_id: str,
    user_id: str,
    last_read_message_id: str = "",
) -> int:
    """Mark every unread row in the thread (up to ``last_read_message_id``
    if supplied) as read, and clear the thread's unread_count.

    Returns the number of message rows newly stamped — callers can use
    this to decide whether to skip the read-receipt routing (no rows
    changed => no notification to emit).
    """

    now = _now_iso()
    if last_read_message_id:
        # Bound the receipt to messages sent on or before the named one.
        cur_last = await conn.execute(
            "SELECT sent_at FROM connect_messages "
            "WHERE message_id = ? AND user_id = ?",
            (last_read_message_id, user_id),
        )
        last_row = await cur_last.fetchone()
        if last_row is None:
            return 0
        last_sent_at = last_row[0]
        cur = await conn.execute(
            """UPDATE connect_messages
                  SET read_at = ?
                WHERE thread_id = ?
                  AND user_id = ?
                  AND read_at IS NULL
                  AND deleted_at IS NULL
                  AND sent_at <= ?""",
            (now, thread_id, user_id, last_sent_at),
        )
    else:
        cur = await conn.execute(
            """UPDATE connect_messages
                  SET read_at = ?
                WHERE thread_id = ?
                  AND user_id = ?
                  AND read_at IS NULL
                  AND deleted_at IS NULL""",
            (now, thread_id, user_id),
        )
    marked = cur.rowcount
    await conn.execute(
        "UPDATE connect_threads SET unread_count = 0 "
        "WHERE thread_id = ? AND user_id = ?",
        (thread_id, user_id),
    )
    await conn.commit()
    return marked


async def soft_delete_message(
    conn: Any, *, message_id: str, user_id: str,
) -> bool:
    """Clear body + stamp deleted_at. Returns whether a row was found."""

    cur = await conn.execute(
        """UPDATE connect_messages
              SET body = '', deleted_at = ?
            WHERE message_id = ? AND user_id = ?
              AND deleted_at IS NULL""",
        (_now_iso(), message_id, user_id),
    )
    await conn.commit()
    return cur.rowcount > 0


async def edit_message(
    conn: Any, *, message_id: str, user_id: str, body: str,
) -> bool:
    """Replace body + stamp edited_at. No-op on already-deleted rows."""

    cur = await conn.execute(
        """UPDATE connect_messages
              SET body = ?, edited_at = ?
            WHERE message_id = ? AND user_id = ?
              AND deleted_at IS NULL""",
        (body, _now_iso(), message_id, user_id),
    )
    await conn.commit()
    return cur.rowcount > 0


async def clear_thread_for_user(
    conn: Any, *, user_id: str, thread_id: str,
) -> int:
    """Hard-delete every message row in ``thread_id`` owned by
    ``user_id``. Local-only — the peer's instance keeps its mirror,
    matching iMessage/WhatsApp "Clear Chat History" semantics. The
    thread row itself is kept so the conversation can resume, but its
    last_message_* preview fields are reset so the thread list doesn't
    keep showing a quote of a message that no longer exists.

    Returns the number of message rows removed.
    """

    cur = await conn.execute(
        "DELETE FROM connect_messages WHERE user_id = ? AND thread_id = ?",
        (user_id, thread_id),
    )
    removed = cur.rowcount
    await conn.execute(
        """UPDATE connect_threads
              SET last_message_at = NULL,
                  last_message_preview = '',
                  unread_count = 0
            WHERE user_id = ? AND thread_id = ?""",
        (user_id, thread_id),
    )
    await conn.commit()
    return removed


async def stamp_delivered(
    conn: Any, *, message_id: str, user_id: str,
) -> bool:
    """Stamp delivered_at = now if not already stamped. Idempotent."""

    cur = await conn.execute(
        """UPDATE connect_messages
              SET delivered_at = ?
            WHERE message_id = ? AND user_id = ?
              AND delivered_at IS NULL""",
        (_now_iso(), message_id, user_id),
    )
    await conn.commit()
    return cur.rowcount > 0


# ── Row converters (internal) ──────────────────────────────────────


def _row_to_thread(row: Any) -> ThreadRow:
    return ThreadRow(
        thread_id=row[0],
        user_id=row[1],
        peer_did=row[2],
        peer_display_name=row[3] or "",
        last_message_at=row[4] or "",
        last_message_preview=row[5] or "",
        unread_count=int(row[6] or 0),
        muted=bool(row[7]),
        pinned=bool(row[8]),
        archived=bool(row[9]),
        created_at=row[10] or "",
    )


def _row_to_message(row: Any) -> MessageRow:
    return MessageRow(
        message_id=row[0],
        thread_id=row[1],
        user_id=row[2],
        sender_did=row[3],
        body=row[4] or "",
        format=row[5] or "plain",
        attachment_ref=row[6] or "",
        reply_to=row[7] or "",
        sent_at=row[8] or "",
        received_at=row[9] or "",
        delivered_at=row[10],
        read_at=row[11],
        edited_at=row[12],
        deleted_at=row[13],
        transcript=row[14] or "",
    )


async def get_fabric_attachment_fields(
    conn: Any, *, message_id: str, user_id: str,
) -> tuple[str | None, str | None]:
    """Lookup the cross-instance attachment fetch fields for a message.

    Returns ``(fetch_url, fetch_token)`` — both None for same-instance
    messages (where the local attachment route handles the bytes
    directly). Used by the catch-up endpoint + the live event
    enrichment path to surface fabric-delivered attachments to the
    recipient's UI. Tolerates absence of the columns (older DB) by
    returning (None, None).
    """
    try:
        cur = await conn.execute(
            "SELECT attachment_fetch_url, attachment_fetch_token "
            "FROM connect_messages WHERE message_id = ? AND user_id = ?",
            (message_id, user_id),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        # Columns not present (migration 240 not yet applied in a
        # test DB) — fall back to the same-instance assumption.
        return None, None
    if row is None:
        return None, None
    return row[0], row[1]
