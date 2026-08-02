"""Shared stale-write guard for user-content tables.

**The problem this closes.** Most user-content tables are written with a
blind ``INSERT ... ON CONFLICT DO UPDATE``. The ``WHERE user_id = ?`` clause
on those upserts is a *tenant* guard — it stops another account clobbering
your row — but it is NOT a *staleness* guard. Two tabs, or a phone and a
desktop, editing the same character card means the second save silently
overwrites the first with no error and no trace. The user finds out later,
when the edit they made simply isn't there.

``ui_sessions`` already solved this (``proxy/chat_routes.py``). This module
is that solution lifted out so every other table can use it, rather than the
fix living on one surface — see the "fix the CLASS, not the symptom" rule in
CLAUDE.md.

**Two clocks, kept separate.** This is the subtlety worth reading before
using this module:

* ``updated_at`` (TEXT, ``datetime('now')``) is the SERVER write time. It
  says when the row was persisted. It is useless for staleness detection
  because it is stamped by whichever write landed last — including the one
  we are trying to reject.
* The *client edit stamp* (ms since epoch) is stamped by the client that
  made the edit, at edit time. That is the only value that can tell us
  "the stored copy contains a change this client has not seen".

We compare client stamps to client stamps, never the server column. This
mirrors the discipline already documented in ``state/sync_store.py``, which
uses the server clock strictly as a pull cursor and never trusts the
device's clock for ordering.

**Two storage shapes.** Existing tables carry the client stamp differently
and we meet them where they are rather than migrating blobs:

* JSON-blob tables (``ui_sessions``, ``ui_characters``) keep it inside the
  blob at ``$.updatedAt`` — described by ``json_column``/``json_path``.
* Column-shaped tables (``prompt_presets``, ``regex_scripts``, …) get a
  dedicated ``client_updated_at INTEGER`` column in migration 325 —
  described by ``stamp_column``.

**Fails OPEN, deliberately.** Any lookup error returns an empty map, which
means "no rejections". A guard that fails closed would block saves outright
on an exotic SQLite build lacking ``json_extract``. Losing the guard is
recoverable; refusing to save the user's work is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class StampSource:
    """Where a table keeps its client edit-stamp, and how rows are scoped.

    Instances live only in the ``_GUARDED`` registry below — the table and
    column names are therefore compile-time constants, never user input,
    which is why interpolating them into SQL below is safe. Ids and user
    ids are always bound as parameters.
    """

    table: str
    id_column: str = "id"
    # Exactly one of these two must be set.
    json_column: str | None = None
    json_path: str = "$.updatedAt"
    stamp_column: str | None = None
    # Legacy pre-auth rows have NULL user_id and stay claimable by the
    # first owner who writes them — same scoping the upserts use.
    user_scoped: bool = True

    def __post_init__(self) -> None:
        if bool(self.json_column) == bool(self.stamp_column):
            raise ValueError(
                f"StampSource({self.table}): set exactly one of "
                "json_column or stamp_column",
            )


# Registry of guarded tables. Adding a table here plus its route-level call
# is the whole integration. ``voice_mixes`` is keyed by ``name``, not ``id``.
_GUARDED: dict[str, StampSource] = {
    # Chats stamp their blob at ``$.updatedAt`` and the chats GET does not
    # overwrite it, so that path stays as-is.
    "ui_sessions": StampSource("ui_sessions", json_column="data"),
    # Cards must NOT use ``$.updatedAt``: list_characters() overwrites that
    # key with the server's ISO ``updated_at`` column for display, so the
    # value the client echoes back is a date string, not a ms integer —
    # int() would fail, the stamp would read 0, and the guard would
    # silently never fire. A distinct key keeps the two clocks apart.
    "ui_characters": StampSource(
        "ui_characters", json_column="data", json_path="$.clientUpdatedAt",
    ),
    "prompt_presets": StampSource("prompt_presets", stamp_column="client_updated_at"),
    "regex_scripts": StampSource("regex_scripts", stamp_column="client_updated_at"),
    "custom_flows": StampSource("custom_flows", stamp_column="client_updated_at"),
    "voice_mixes": StampSource(
        "voice_mixes", id_column="name", stamp_column="client_updated_at",
    ),
    "lorebook_entries": StampSource(
        "lorebook_entries", stamp_column="client_updated_at",
    ),
}


def incoming_stamp(payload: dict) -> int:
    """Read the stamp a write should be judged against.

    Prefers ``baseUpdatedAt`` — the stamp the client held when it LOADED
    the row — and falls back to ``updatedAt``, the stamp of the edit it is
    now writing. The distinction decides how strong the guard is, so it is
    worth being precise about:

    * **With ``baseUpdatedAt`` (strong, true optimistic concurrency).**
      "Has anyone written since I loaded this?" If the stored stamp is
      newer than the base, another tab or device committed a change this
      client never saw, and the write is a genuine conflict — regardless
      of which side clicked save last.

    * **With only ``updatedAt`` (weak, replay-only).** "Is the content I'm
      sending older than what's stored?" This catches a stale tab whose
      autosave replays a copy loaded before someone else's edit, which is
      the common silent-clobber path. It CANNOT catch two tabs editing
      concurrently: whichever saves second stamps a later time and is
      accepted, overwriting the first. That is the pre-existing behaviour
      on chats, preserved here rather than silently changed.

    Key order is ``baseUpdatedAt`` → ``clientUpdatedAt`` → ``updatedAt``.
    The last is a legacy fallback for chats, which already send their edit
    stamp as ``updatedAt``. New surfaces should send the first two and
    leave ``updatedAt`` alone: several GET routes overwrite it with the
    server's ISO ``updated_at`` column for display, so it is NOT reliably
    a client stamp.

    Returns 0 when all are absent or unparseable — legacy clients that
    send no stamp are always accepted, so rolling this out never breaks an
    older tab mid-session.
    """
    for key in ("baseUpdatedAt", "clientUpdatedAt", "updatedAt"):
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return 0


def edit_stamp(payload: dict) -> int:
    """Read the stamp a write should be PERSISTED with.

    Deliberately different from :func:`incoming_stamp`. That one answers
    "what should this write be judged against?" and prefers the client's
    base. This one answers "what does the row now contain?", which is
    always the new edit's own ``updatedAt`` — storing the base would pin
    the row in the past and make the next write look spuriously fresh.

    Returns 0 when the client sends no stamp. A 0 row is treated as
    unguarded on the next write, which is the same legacy-tolerant
    behaviour described in the module docstring: we never invent a stamp
    from the server clock, because mixing the two clocks is exactly the
    bug this module exists to avoid.
    """
    for key in ("clientUpdatedAt", "updatedAt"):
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return 0


async def stored_stamps(
    conn,
    table: str,
    ids: list[str],
    *,
    user_id: str = "",
) -> dict[str, int]:
    """Map row id -> client edit-stamp of the STORED row.

    Missing rows are simply absent from the result (a first write has
    nothing to be stale against). Fails OPEN — see the module docstring.
    """
    if not ids:
        return {}
    src = _GUARDED.get(table)
    if src is None:
        log.warning("write_guard_unregistered_table", table=table)
        return {}

    expr = (
        f"json_extract({src.json_column}, '{src.json_path}')"
        if src.json_column
        else src.stamp_column
    )
    placeholders = ",".join("?" * len(ids))
    query = (
        f"SELECT {src.id_column}, {expr} FROM {src.table} "
        f"WHERE {src.id_column} IN ({placeholders})"
    )
    params: list = list(ids)
    if src.user_scoped:
        if user_id:
            query += " AND (user_id = ? OR user_id IS NULL)"
            params.append(user_id)
        else:
            query += " AND user_id IS NULL"

    try:
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
    except Exception as exc:
        log.warning(
            "write_guard_lookup_failed", table=table, error=str(exc)[:160],
        )
        return {}

    out: dict[str, int] = {}
    for row_id, stamp in rows:
        try:
            out[row_id] = int(stamp or 0)
        except (TypeError, ValueError):
            out[row_id] = 0
    return out


async def find_stale(
    conn,
    table: str,
    incoming: dict[str, int],
    *,
    user_id: str = "",
) -> list[str]:
    """Return the ids whose stored copy is NEWER than the incoming edit.

    ``incoming`` maps row id -> client stamp. Entries with a falsy stamp
    are skipped (legacy clients are accepted unguarded).
    """
    candidates = {rid: ts for rid, ts in incoming.items() if ts}
    if not candidates:
        return []
    stored = await stored_stamps(
        conn, table, list(candidates), user_id=user_id,
    )
    return [
        rid for rid, ts in candidates.items() if stored.get(rid, 0) > ts
    ]


async def is_stale(
    conn,
    table: str,
    row_id: str,
    stamp: int,
    *,
    user_id: str = "",
) -> bool:
    """Single-row convenience wrapper over :func:`find_stale`."""
    if not stamp:
        return False
    stale = await find_stale(conn, table, {row_id: stamp}, user_id=user_id)
    return bool(stale)


def stale_payload(row_id: str) -> dict:
    """The uniform 409 body.

    One wire contract across every guarded surface, matching what the chat
    client already handles, so clients need a single branch rather than one
    per table.
    """
    return {"ok": False, "stale": True, "id": row_id}
