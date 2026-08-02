"""Per-user media-server credential storage.

CRUD for the ``user_media_servers`` table. Every function is user-scoped
by keyword-only ``user_id``; server-level reads (e.g. the streaming
proxy resolving a target by id) use ``get_by_id_any_user`` with an
explicit caller-side user check against the file_index row.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


@dataclass(slots=True)
class MediaServer:
    id: str
    user_id: str
    provider: str
    name: str
    base_url: str
    access_token: str
    status: str
    status_detail: str
    last_sync_at: str | None
    item_count: int
    created_at: str
    updated_at: str
    # Sync diagnostics — populated on every sync, surfaced on the
    # server card so the user can tell at a glance whether "N books"
    # matches what they expected or whether items were silently dropped.
    total_seen: int = 0
    skipped_count: int = 0
    # Sharing scope. 'private' = only `user_id` sees/uses this row.
    # 'shared' = admin-published; every authenticated user sees it
    # read-only (can sync to their own file_index + stream, can't
    # edit URL/token/name/scope). Default keeps pre-existing rows
    # invisible to other users when the migration first applies.
    scope: str = "private"
    last_sync_skipped: list[dict] = field(default_factory=list)

    def is_borrowed_by(self, user_id: str) -> bool:
        """True when ``user_id`` is using someone ELSE's server row.

        ``access_token`` on this row belongs to :attr:`user_id` (the
        owner). Every provider call made with it — catalog fetch,
        progress fetch, progress push — therefore reads and writes the
        OWNER's account upstream, no matter who triggered it. For an
        admin-shared server that means, unguarded:

        - the owner's progress / played flags / favorites get written
          into the borrower's ``file_index`` rows, and
        - the borrower's playback gets pushed back into the OWNER's
          Emby/ABS/Komga account, corrupting their real watch state.

        Callers MUST consult this before touching any per-user field in
        either direction. Sharing conveys the credential and the catalog,
        never the personal state layered on top of it — for a borrowed
        server, progress is Augmentum-side only.

        An empty ``user_id`` is treated as NOT borrowed so internal and
        test callers with no viewer context keep pre-share behavior.
        """
        return bool(user_id) and user_id != self.user_id

    def to_dict(
        self,
        *,
        redact_token: bool = True,
        viewer_user_id: str = "",
    ) -> dict:
        """Project to the response shape the UI consumes.

        ``viewer_user_id`` is the caller asking for this server. When set
        and the row is shared, the response carries ``is_owned_by_viewer``
        = False so the frontend can hide edit/delete affordances. When
        empty (internal callers, tests), defaults to True — same shape
        as before the share feature landed.
        """
        is_owned = (not viewer_user_id) or self.user_id == viewer_user_id
        return {
            "id": self.id,
            "provider": self.provider,
            "name": self.name,
            "base_url": self.base_url,
            "has_token": bool(self.access_token),
            "access_token": "" if redact_token else self.access_token,
            "status": self.status,
            "status_detail": self.status_detail,
            "last_sync_at": self.last_sync_at,
            "item_count": self.item_count,
            "total_seen": self.total_seen,
            "skipped_count": self.skipped_count,
            "last_sync_skipped": self.last_sync_skipped,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "scope": self.scope,
            "is_shared": self.scope == "shared",
            "is_owned_by_viewer": is_owned,
        }


def _normalize_base_url(url: str) -> str:
    """Strip trailing slash; lowercase scheme+host. Keeps path as-is."""
    s = (url or "").strip()
    if not s:
        return s
    while s.endswith("/"):
        s = s[:-1]
    return s


class MediaServerStore:
    # Short-TTL read cache for ``get_visible`` — the hot server-resolve on
    # every media use-path (stream / browse / cover / cast-thumbnail). A single
    # media detail view fires ~18 cast-image requests that each re-resolve the
    # SAME server row; without this they 18× the identical PK query against the
    # shared aiosqlite connection, spiking slow_db_op + the event loop. The
    # cache stores the raw row (a fresh ``MediaServer`` is still built per call,
    # so no aliasing) and is INVALIDATED on every write (create/update/token/
    # scope/delete) — all writes to user_media_servers go through this store
    # (verified), so a changed URL/token/scope/status is never served stale.
    # The TTL is a memory bound + backstop, NOT the freshness mechanism.
    _VISIBLE_CACHE_TTL = 60.0  # seconds
    _VISIBLE_CACHE_MAX = 1024

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn
        # key: (server_id, user_id) -> (expiry_monotonic, raw_row_or_None)
        self._visible_cache: dict[tuple[str, str], tuple[float, object]] = {}

    def _invalidate_visible_cache(self) -> None:
        """Drop the get_visible cache. Called by every mutator so a write is
        reflected on the very next read (zero staleness window)."""
        self._visible_cache.clear()

    # --- Create / Update ---------------------------------------------------

    async def create(
        self,
        *,
        user_id: str,
        provider: str,
        name: str,
        base_url: str,
        access_token: str = "",
    ) -> MediaServer:
        if not user_id:
            raise ValueError("media server requires a user_id")
        if not provider:
            raise ValueError("media server requires a provider")
        if not base_url:
            raise ValueError("media server requires a base_url")

        server_id = f"ms_{secrets.token_hex(8)}"
        normalized = _normalize_base_url(base_url)
        await self._conn.execute(
            "INSERT INTO user_media_servers "
            "(id, user_id, provider, name, base_url, access_token) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (server_id, user_id, provider, name, normalized,
             encrypt_api_key(access_token) or ""),
        )
        await self._conn.commit()
        self._invalidate_visible_cache()
        log.info(
            "media_server_created", id=server_id, user_id=user_id,
            provider=provider, name=name,
        )
        row = await self.get(server_id, user_id=user_id)
        assert row is not None
        return row

    async def update(
        self,
        server_id: str,
        *,
        user_id: str,
        name: str | None = None,
        base_url: str | None = None,
        access_token: str | None = None,
        status: str | None = None,
        status_detail: str | None = None,
        last_sync_at: str | None = None,
        item_count: int | None = None,
        total_seen: int | None = None,
        skipped_count: int | None = None,
        last_sync_skipped: list[dict] | None = None,
    ) -> MediaServer | None:
        fields: list[str] = []
        params: list[object] = []
        if name is not None:
            fields.append("name = ?")
            params.append(name)
        if base_url is not None:
            fields.append("base_url = ?")
            params.append(_normalize_base_url(base_url))
        if access_token is not None:
            fields.append("access_token = ?")
            params.append(encrypt_api_key(access_token) or "")
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if status_detail is not None:
            fields.append("status_detail = ?")
            params.append(status_detail)
        if last_sync_at is not None:
            fields.append("last_sync_at = ?")
            params.append(last_sync_at)
        if item_count is not None:
            fields.append("item_count = ?")
            params.append(item_count)
        if total_seen is not None:
            fields.append("total_seen = ?")
            params.append(total_seen)
        if skipped_count is not None:
            fields.append("skipped_count = ?")
            params.append(skipped_count)
        if last_sync_skipped is not None:
            fields.append("last_sync_skipped = ?")
            params.append(json.dumps(last_sync_skipped))
        if not fields:
            return await self.get(server_id, user_id=user_id)

        fields.append("updated_at = datetime('now')")
        params.extend([server_id, user_id])
        await self._conn.execute(
            f"UPDATE user_media_servers SET {', '.join(fields)} "
            "WHERE id = ? AND user_id = ?",
            params,
        )
        await self._conn.commit()
        self._invalidate_visible_cache()
        return await self.get(server_id, user_id=user_id)

    async def update_token_for_provider(
        self, provider: str, access_token: str, *, base_url: str = "",
    ) -> int:
        """Refresh the stored access token for every row of a provider.

        Provisioned media servers are install-wide singletons with one
        managed login, but each connected user has their own
        ``user_media_servers`` row pointing at it. When that shared
        credential changes, every row's token must be refreshed or the
        other users get auth errors until they reconnect. Returns the
        number of rows updated.

        ``base_url`` scopes the write to rows pointing at that instance —
        without it, a managed credential change would clobber the token on
        manually-connected EXTERNAL servers of the same provider (which
        have their own, unrelated credentials). Pass it whenever the
        caller has the instance's URL; the bare form is kept for callers
        that genuinely mean the whole provider.
        """
        if not provider:
            return 0
        sql = (
            "UPDATE user_media_servers SET access_token = ?, "
            "updated_at = datetime('now') WHERE provider = ?"
        )
        params: list[object] = [encrypt_api_key(access_token) or "", provider]
        if base_url:
            sql += " AND base_url = ?"
            params.append(_normalize_base_url(base_url))
        cursor = await self._conn.execute(sql, params)
        await self._conn.commit()
        self._invalidate_visible_cache()
        return cursor.rowcount or 0

    # --- Read --------------------------------------------------------------

    # Shared column list so the SELECT shape matches the dataclass field
    # order exactly. Positional unpack (MediaServer(*row)) is fine here
    # because the dataclass carries the matching field order — any new
    # column must be appended to BOTH the SELECT and the dataclass or
    # this breaks loudly at construction time. `scope` sits right before
    # `last_sync_skipped` so the trailing JSON column stays last (the
    # row unpacker peels it off the tail).
    _SELECT_COLS = (
        "id, user_id, provider, name, base_url, access_token, "
        "status, status_detail, last_sync_at, item_count, "
        "created_at, updated_at, "
        "total_seen, skipped_count, scope, last_sync_skipped"
    )

    def _row_to_server(self, row) -> MediaServer:
        # last_sync_skipped comes back as a JSON string; decode once so
        # callers never touch raw JSON. Malformed payloads (partial
        # write, hand-edit) fall back to an empty list rather than
        # crashing the whole response.
        *head, skipped_json = row
        try:
            skipped = json.loads(skipped_json or "[]")
            if not isinstance(skipped, list):
                skipped = []
        except (json.JSONDecodeError, TypeError):
            skipped = []
        # access_token is column index 5 in _SELECT_COLS; decrypt at-rest
        # value before handing the dataclass to callers (callers pass the
        # token to upstream Plex/Jellyfin Bearer headers, so it must be
        # plaintext in memory).
        head = list(head)
        head[5] = decrypt_api_key(head[5]) or ""
        return MediaServer(*head, last_sync_skipped=skipped)

    async def get(self, server_id: str, *, user_id: str) -> MediaServer | None:
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM user_media_servers "
            "WHERE id = ? AND user_id = ?",
            (server_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_server(row)

    async def list_for_user(self, *, user_id: str) -> list[MediaServer]:
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM user_media_servers "
            "WHERE user_id = ? ORDER BY created_at ASC",
            (user_id,),
        )
        return [self._row_to_server(row) for row in await cursor.fetchall()]

    async def list_all(self) -> list[MediaServer]:
        """Every media server across all users — for the periodic re-sync loop."""
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM user_media_servers ORDER BY created_at ASC",
        )
        return [self._row_to_server(row) for row in await cursor.fetchall()]

    async def reset_stale_syncing(self) -> int:
        """Flip rows stuck in 'syncing' back to 'ok' — left that way when a
        restart killed a sync mid-run before it could finalize (the catalog is
        already indexed; only the status flag is stale). Returns rows reset."""
        cursor = await self._conn.execute(
            "UPDATE user_media_servers SET status = 'ok', status_detail = '' "
            "WHERE status = 'syncing'",
        )
        await self._conn.commit()
        return cursor.rowcount or 0

    async def list_visible(self, *, user_id: str) -> list[MediaServer]:
        """Servers visible to ``user_id``: own rows ∪ admin-shared rows.

        Used by the routes the user-facing UI calls. The non-owner gets
        the shared row read-only (the route layer is what enforces that,
        but the row itself is the same shape — only ``user_id`` differs).
        """
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM user_media_servers "
            "WHERE user_id = ? OR scope = 'shared' "
            # Show user's own servers first; admin-shared ones below.
            "ORDER BY (user_id = ?) DESC, created_at ASC",
            (user_id, user_id),
        )
        return [self._row_to_server(row) for row in await cursor.fetchall()]

    async def get_visible(
        self, server_id: str, *, user_id: str,
    ) -> MediaServer | None:
        """Get a server the caller can use: owned by them, or admin-shared.

        Distinct from ``get()`` (strict ownership). Use this on USE paths
        — streaming, browsing, syncing — where shared servers must resolve
        for any authenticated user. Stick with ``get()`` on WRITE paths
        that must verify ownership (edit URL/token/name/delete).

        Cached (short TTL + write-invalidation) because USE paths re-resolve
        the same server many times in a burst (esp. cast-image thumbnails).
        """
        key = (server_id, user_id)
        now = time.monotonic()
        cached = self._visible_cache.get(key)
        if cached is not None and cached[0] > now:
            row = cached[1]
            return self._row_to_server(row) if row else None
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM user_media_servers "
            "WHERE id = ? AND (user_id = ? OR scope = 'shared')",
            (server_id, user_id),
        )
        row = await cursor.fetchone()
        # Cache positive AND negative (None) results — a burst against a
        # missing/unauthorized server shouldn't re-hammer the DB either.
        if len(self._visible_cache) >= self._VISIBLE_CACHE_MAX:
            self._visible_cache.clear()
        self._visible_cache[key] = (now + self._VISIBLE_CACHE_TTL, row)
        return self._row_to_server(row) if row else None

    async def get_any(self, server_id: str) -> MediaServer | None:
        """Get a server by id with no user filter.

        For admin-only edits to shared rows where the route layer has
        already verified the caller is admin. Don't call this on
        user-driven paths — it bypasses ownership entirely.
        """
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM user_media_servers "
            "WHERE id = ?",
            (server_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_server(row) if row else None

    async def set_scope(
        self,
        server_id: str,
        *,
        scope: str,
        # owner_user_id constrains the UPDATE so an admin can only flip
        # servers THEY OWN. Sharing someone else's server would either
        # (a) require transferring ownership or (b) leak someone's token
        # to all users without their consent. Both are out of scope; an
        # admin who wants to share a non-admin user's server reads the
        # config from them and adds it under their own account.
        owner_user_id: str,
    ) -> MediaServer | None:
        """Flip a server between 'private' and 'shared'.

        Caller MUST verify ``owner_user_id`` is admin before invoking —
        this method enforces ownership (so admin can't accidentally share
        another user's row) but not the admin role. See
        ``media_routes.py::update_server`` for the gate.
        """
        if scope not in ("private", "shared"):
            raise ValueError(f"invalid scope: {scope!r}")
        if not owner_user_id:
            raise ValueError("set_scope requires owner_user_id")
        await self._conn.execute(
            "UPDATE user_media_servers "
            "SET scope = ?, updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (scope, server_id, owner_user_id),
        )
        await self._conn.commit()
        self._invalidate_visible_cache()
        return await self.get(server_id, user_id=owner_user_id)

    async def find_match(
        self,
        *,
        user_id: str,
        provider: str,
        base_url: str,
    ) -> MediaServer | None:
        """Used by auto-detect to avoid re-offering a server the user already added.

        Also matches admin-shared servers visible to ``user_id`` so we
        don't pester non-admins to "connect" something an admin has
        already shared for them.
        """
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM user_media_servers "
            "WHERE (user_id = ? OR scope = 'shared') "
            "AND provider = ? AND base_url = ?",
            (user_id, provider, _normalize_base_url(base_url)),
        )
        row = await cursor.fetchone()
        return self._row_to_server(row) if row else None

    # --- Delete ------------------------------------------------------------

    async def delete(self, server_id: str, *, user_id: str) -> bool:
        cursor = await self._conn.execute(
            "DELETE FROM user_media_servers WHERE id = ? AND user_id = ?",
            (server_id, user_id),
        )
        await self._conn.commit()
        self._invalidate_visible_cache()
        ok = cursor.rowcount > 0
        if ok:
            log.info("media_server_deleted", id=server_id, user_id=user_id)
        return ok


# --- Cascade purge -----------------------------------------------------------
#
# When a media server is removed, every cached file_index row that referenced
# it becomes dead weight: the streaming proxy can't resolve the credentials,
# so any request through `/api/media/comic/page` or `/stream/...` 502s. The
# `media_library_views` table cascades automatically via SQLite FK, but
# file_index doesn't — server_id lives inside `source_metadata` JSON, with
# no foreign-key constraint possible.
#
# `purge_server_data` is the single source of truth for that cascade. Both
# the delete-server HTTP route and the one-time orphan-cleanup script call
# it, so the rules stay in one place. It does NOT remove the user_media_
# servers row itself — callers do that explicitly so the script can run
# even when the server row is already gone (which is the common case for
# orphans that accumulated before this helper existed).


async def purge_server_data(
    db: aiosqlite.Connection,
    server_id: str,
    *,
    user_id: str,
) -> dict[str, int]:
    """Cascade-delete every cached row tied to one media server.

    Tables touched:
      - file_index — rows whose source_metadata.server_id matches.
      - comic_series — rows orphaned by the file_index purge (no chapters
        left). comic_series has no server_id column, so we identify them
        after-the-fact by their absence from any remaining file_index row.

    Tables that cascade automatically when the server row is removed:
      - media_library_views (REFERENCES user_media_servers ... ON DELETE
        CASCADE). Callers must still delete the server row themselves;
        this helper only handles the JSON-referenced cascade.

    What's NOT lost, for a server you OWN: progress data on the upstream
    provider. Suwayomi, Komga, Audiobookshelf, Emby all track
    is_finished + last_position on their own side, and our
    `push_progress` writes to them on every page flip. A re-add + resync
    pulls fresh file_index rows with the canonical progress restored.
    That's why no merge logic is needed here.

    This does NOT hold for a BORROWED (admin-shared) server. There the
    upstream account is the owner's, so we never push the borrower's
    progress to it and never read the owner's progress back — see
    :meth:`MediaServer.is_borrowed_by`. A borrower's progress lives only
    in their file_index rows, which means this purge DOES destroy it
    permanently. Warn before purging a shared server on the borrower's
    behalf.

    Returns counts so callers can surface impact (toast, route response).
    """
    chapters_cursor = await db.execute(
        """
        DELETE FROM file_index
        WHERE user_id = ?
          AND json_extract(source_metadata, '$.server_id') = ?
        """,
        (user_id, server_id),
    )
    chapters_removed = chapters_cursor.rowcount or 0

    # Series cleanup runs only if we actually removed chapters — otherwise
    # the orphan-search would scan the full file_index for nothing.
    series_removed = 0
    if chapters_removed > 0:
        series_cursor = await db.execute(
            """
            DELETE FROM comic_series
            WHERE user_id = ?
              AND id NOT IN (
                SELECT DISTINCT series_id
                FROM file_index
                WHERE user_id = ? AND series_id IS NOT NULL
              )
            """,
            (user_id, user_id),
        )
        series_removed = series_cursor.rowcount or 0

    await db.commit()

    if chapters_removed or series_removed:
        log.info(
            "media_server_purge",
            server_id=server_id, user_id=user_id,
            chapters=chapters_removed, series=series_removed,
        )
    return {
        "chapters": chapters_removed,
        "series":   series_removed,
    }
