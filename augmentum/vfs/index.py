"""File index service — unified catalog across all file subsystems."""

from __future__ import annotations

import contextlib
import json
import secrets
from datetime import datetime
from typing import TYPE_CHECKING, Any

from augmentum.vfs.classify import derive_kind
from augmentum.vfs.models import FileEntry
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


# Trash-semantics guard. `is_trashed = 1` carries two distinct meanings
# now (migration 323):
#
#   detached_at IS NULL     — the user deleted this. Belongs in Trash,
#                             restorable, auto-purged after 30 days.
#   detached_at IS NOT NULL — a media server was un-shared or deleted out
#                             from under this row. The user didn't delete
#                             anything; we hid it because it can no longer
#                             be streamed. Its progress/history is being
#                             PRESERVED for a possible re-share.
#
# Every path that implements *trash semantics* (listing the Trash bin,
# counting it, aging rows out, hard-deleting them) must carry this so
# detached rows are exempt. Paths that merely implement *invisibility*
# (`is_trashed = 0`, ~30 queries) need no change — detached rows are
# correctly invisible there, which is the whole point of reusing the flag.
#
# Getting this wrong is silent and delayed: without it, `purge_all_old_trash`
# deletes the preserved history 30 days later with no user action and no
# trace.
NOT_DETACHED = "detached_at IS NULL"


_SORT_CLAUSES = {
    "newest": "created_at DESC",
    "oldest": "created_at ASC",
    "name":   "name COLLATE NOCASE ASC",
    "size":   "size_bytes DESC",
    # Release-year sorts read ``source_metadata.year`` first, falling
    # back to the secondary ``source_metadata.extra.year`` copy that
    # older importers wrote. Rows with no year flush to 0 so they sit
    # at the far end of the ordering rather than throwing — predictable
    # for "Year ↓ / Year ↑" UI without surprising holes mid-list.
    "year_desc": (
        "CAST(COALESCE("
        "  json_extract(source_metadata, '$.year'),"
        "  json_extract(source_metadata, '$.extra.year'),"
        "  0"
        ") AS INTEGER) DESC, name COLLATE NOCASE ASC"
    ),
    "year_asc": (
        "CAST(COALESCE("
        "  json_extract(source_metadata, '$.year'),"
        "  json_extract(source_metadata, '$.extra.year'),"
        "  0"
        ") AS INTEGER) ASC, name COLLATE NOCASE ASC"
    ),
}


# Explicit column list for every list/search query. Excludes the
# ``embedding`` BLOB (3KB/row × 64k rows = 196MB on disk) — it's
# read on every list query but never sent to the UI (``to_dict``
# drops it). Switching from ``SELECT *`` to this list cuts ~180KB
# of pointless disk I/O off every page request and removes the
# Python-side bytes-allocation churn from the hot path.
#
# Order matches ``_row_to_entry``'s positional indexing.
_ENTRY_COLUMNS = (
    "id, user_id, source, source_id, name, mime_type, "
    "size_bytes, real_path, description, tags, thumbnail, "
    "is_directory, parent_id, source_metadata, "
    "created_at, updated_at, is_favorite, is_trashed, trashed_at, "
    "kind, last_played_at, series_id"
)
# Same column set, prefixed for the FTS join in ``search``.
_ENTRY_COLUMNS_FI = ", ".join(f"fi.{c.strip()}" for c in _ENTRY_COLUMNS.split(","))


def _sort_clause(sort: str | None) -> str:
    return _SORT_CLAUSES.get(sort or "newest", _SORT_CLAUSES["newest"])


_MEDIA_SOURCES = {"audiobookshelf", "librivox", "emby", "jellyfin"}

# Sort options that only make sense when the rows carry media-server
# source_metadata (author string, progress timestamps). When any other
# source is in play, ``_resolve_sort`` falls back to "newest" so the
# list doesn't end up ordered by an absent JSON field.
_MEDIA_SORT_CLAUSES = {
    # Lowercase-compared so "andy weir" sorts alongside "Andy Weir"
    # regardless of tagging casing.
    "author":   "LOWER(COALESCE(json_extract(source_metadata, '$.author'), '')) ASC, name COLLATE NOCASE ASC",
    # "Recently played" — sorts by the dedicated ``last_played_at``
    # column (migration 195) that's set EXCLUSIVELY by the progress
    # endpoint and never touched by catalog sync. Falls back to
    # ``updated_at`` for rows that have never been played so they
    # still order sensibly at the bottom of the rail. Previous impl
    # was bare ``updated_at DESC`` — that signal got destroyed every
    # time catalog sync ran because the upsert bumps updated_at on
    # every synced row uniformly, washing out playback ordering. See
    # 195_file_index_last_played_at.sql for the full rationale.
    "progress": "COALESCE(last_played_at, updated_at) DESC",
}


def _resolve_sort(sort: str | None, *, on_media_source: bool) -> str:
    """Return the ORDER BY fragment (no 'ORDER BY' prefix)."""
    if sort in _MEDIA_SORT_CLAUSES:
        # ``progress`` is column-backed (last_played_at, migration 195)
        # and therefore safe on rows from ANY source — never-played
        # rows just fall through to updated_at via COALESCE. Without
        # this carve-out, the cast Continue rail's resume query (no
        # kind/source set → on_media_source=False) silently downgrades
        # to "newest", which made the new column never get consulted
        # and froze the rail order.
        if sort == "progress" or on_media_source:
            return _MEDIA_SORT_CLAUSES[sort]
        return _SORT_CLAUSES["newest"]
    return _sort_clause(sort)


# Maps ``media_status`` query param values to a SQL predicate on
# ``source_metadata``. Each predicate is scoped such that non-media
# rows (no progress_pct field) don't match any filter except "all".
_MEDIA_STATUS_PREDICATES = {
    "in_progress": (
        "COALESCE(json_extract(source_metadata, '$.is_finished'), 0) = 0 "
        "AND COALESCE(json_extract(source_metadata, '$.progress_pct'), 0) > 0"
    ),
    "finished": (
        "COALESCE(json_extract(source_metadata, '$.is_finished'), 0) = 1"
    ),
    "not_started": (
        "COALESCE(json_extract(source_metadata, '$.is_finished'), 0) = 0 "
        "AND COALESCE(json_extract(source_metadata, '$.progress_pct'), 0) = 0"
    ),
}


def _entity_kind_clause(
    entity_kind: str | None = None,
    entity_kinds: list[str] | None = None,
    exclude_entity_kinds: list[str] | None = None,
    *,
    alias: str = "",
) -> tuple[str, list]:
    prefix = f"{alias}." if alias else ""
    clauses: list[str] = []
    params: list = []
    field = f"json_extract({prefix}source_metadata, '$.entity_kind')"
    if entity_kind:
        clauses.append(f"COALESCE({field}, '') = ?")
        params.append(entity_kind)
    elif entity_kinds:
        placeholders = ", ".join("?" for _ in entity_kinds)
        clauses.append(f"COALESCE({field}, '') IN ({placeholders})")
        params.extend(entity_kinds)
    if exclude_entity_kinds:
        placeholders = ", ".join("?" for _ in exclude_entity_kinds)
        clauses.append(
            f"(COALESCE({field}, '') = '' OR COALESCE({field}, '') NOT IN ({placeholders}))"
        )
        params.extend(exclude_entity_kinds)
    if not clauses:
        return "", []
    return " AND ".join(clauses), params


import re as _re

# FTS5 operators / reserved characters that break MATCH parsing when a
# user types them raw. We strip rather than escape so "AND"-typed-by-user
# becomes a prefix-search token, not a boolean operator surprise.
#
# Apostrophes (ASCII + typographic) are stripped too: FTS5's default
# ``unicode61`` tokenizer treats them as token breaks, so indexed text
# like "Ranger's Apprentice" is stored as ["ranger", "s", "apprentice"].
# A user query of "ranger's" must match the same tokens — we split on
# the apostrophe here so each token gets its own prefix-star.
_FTS_STRIP_RE = _re.compile(r"[\"\(\)\*\^\{\}\[\]:+\-!~?<>=&|'\u2018\u2019\u201C\u201D`.,;/\\]")


def _build_fts_query(raw: str) -> str:
    """Turn a user-typed search string into a prefix-match FTS5 MATCH.

    Search-as-you-type must return results after *every* keystroke —
    "ra" should match "Ranger's Apprentice", not wait until the user
    types the full word. Default FTS5 MATCH is full-token-match only,
    so we tokenize the input and append ``*`` to each piece.

    We also strip FTS5-reserved characters so a stray apostrophe or
    parenthesis doesn't syntax-error the whole query (which would fall
    through to the LIKE backup and be slower + less accurate).

    Examples::

        _build_fts_query("ra")             → 'ra*'
        _build_fts_query("ranger's app")   → 'ranger* s* app*'
        _build_fts_query("AND OR")         → 'AND* OR*'   (treated as tokens)
        _build_fts_query("  ")             → ''           (caller should list_recent)
    """
    if not raw:
        return ""
    cleaned = _FTS_STRIP_RE.sub(" ", raw)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return ""
    # Append '*' to every token so each is a prefix match. FTS5 treats
    # consecutive tokens as implicit AND, so "ran* app*" matches rows
    # containing BOTH a token starting with "ran" AND one starting with
    # "app" — which is exactly the natural expectation.
    return " ".join(f"{t}*" for t in tokens)


class FileIndexService:
    """Manages the file_index table — register, search, list, CRUD."""

    def __init__(
        self,
        db: aiosqlite.Connection,
        *,
        autocommit: bool = True,
    ) -> None:
        self._db = db
        # ``autocommit=False`` suppresses the per-row commit in
        # :meth:`register` so a bulk caller can batch many rows into one
        # transaction and commit on its own cadence. Catalog sync indexes
        # tens of thousands of rows; at one commit per row that is one
        # queued operation per row on the aiosqlite worker thread, which
        # is what made a media scan audibly stutter concurrent voice and
        # chat traffic. ONLY set this False through
        # :mod:`augmentum.vfs.bulk`, which owns the commit cadence, the
        # dedicated connection, and the event-loop yield. A caller that
        # sets it False and forgets to commit loses every buffered row.
        self._autocommit = autocommit
        # Optional jobs_store — wired by server.py after both services
        # exist. Used by :meth:`register` to enqueue a ``file_caption``
        # job for image rows. Left None in tests / standalone usage; the
        # enqueue path no-ops when unset.
        self._jobs_store: Any = None

    def set_jobs_store(self, jobs_store: Any) -> None:
        """Inject the JobsStore handle used for background enrichment
        enqueue. Idempotent; pass ``None`` to disable."""
        self._jobs_store = jobs_store

    async def register(
        self,
        *,
        user_id: str,
        source: str,
        source_id: str,
        name: str,
        mime_type: str = "",
        size_bytes: int = 0,
        real_path: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        thumbnail: str | None = None,
        source_metadata: dict | None = None,
        parent_id: str | None = None,
        is_directory: bool = False,
        scan_status: str = "pending",
        mtime: int | None = None,
        scan_error: dict | None = None,
        metadata_confidence: float = 0.5,
        series_id: str | None = None,
    ) -> str:
        """Register a file in the index. Upserts on (user_id, source, source_id).

        ``scan_status`` / ``mtime`` / ``scan_error`` / ``metadata_confidence`` /
        ``series_id`` are the comic-library scan-pipeline columns (migration
        101). Existing callers that don't set them keep pre-101 behavior: scan
        status defaults to ``'pending'`` so non-comic sources (audiobookshelf,
        librivox, etc.) aren't flagged as errored.
        """
        if not user_id:
            raise ValueError(
                f"file_index.register requires a user_id (source={source}, source_id={source_id})"
            )
        file_id = f"fi_{secrets.token_hex(8)}"
        tags_json = json.dumps(tags or [])
        # Preserve local-only playback state across catalog sync. Sync
        # rebuilds source_metadata from upstream-only fields (see
        # media/sync.py:475-512); without this merge, every sync wiped
        # the JSON-side ``last_read_at`` / ``current_time_s`` /
        # ``progress_pct`` / ``is_finished`` fields the progress endpoint
        # had written, breaking resume + leaving the Continue rail's
        # ordering based on outdated upstream-side progress. The
        # dedicated ``last_played_at`` COLUMN (migration 195) is the
        # authoritative recency signal for the rail; this merge keeps
        # the JSON fields in sync for surfaces that still read them
        # (the per-card progress bar, "Resume" CTA, etc.).
        meta_in = dict(source_metadata or {})
        try:
            cur = await self._db.execute(
                "SELECT source_metadata FROM file_index "
                "WHERE user_id = ? AND source = ? AND source_id = ?",
                (user_id, source, source_id),
            )
            existing_row = await cur.fetchone()
            if existing_row and existing_row[0]:
                existing_meta = json.loads(existing_row[0])
                if isinstance(existing_meta, dict):
                    for k in (
                        "last_read_at", "current_time_s", "progress_pct",
                        "is_finished", "selected_episode_id",
                    ):
                        if k in existing_meta and k not in meta_in:
                            meta_in[k] = existing_meta[k]
        except (json.JSONDecodeError, TypeError):
            pass  # corrupt row — let UPSERT overwrite cleanly
        meta_json = json.dumps(meta_in)
        kind = derive_kind(mime_type, name)
        scan_error_json = json.dumps(scan_error) if scan_error else None

        await self._db.execute(
            """INSERT INTO file_index
               (id, user_id, source, source_id, name, mime_type, size_bytes,
                real_path, description, tags, thumbnail, is_directory, parent_id,
                source_metadata, kind, scan_status, mtime, scan_error,
                metadata_confidence, series_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       datetime('now'), datetime('now'))
               ON CONFLICT(user_id, source, source_id) DO UPDATE SET
                 name = excluded.name,
                 mime_type = excluded.mime_type,
                 size_bytes = excluded.size_bytes,
                 real_path = excluded.real_path,
                 description = excluded.description,
                 tags = excluded.tags,
                 thumbnail = excluded.thumbnail,
                 source_metadata = excluded.source_metadata,
                 kind = excluded.kind,
                 scan_status = excluded.scan_status,
                 mtime = excluded.mtime,
                 scan_error = excluded.scan_error,
                 metadata_confidence = excluded.metadata_confidence,
                 series_id = excluded.series_id,
                 updated_at = datetime('now')""",
            (file_id, user_id, source, source_id, name, mime_type, size_bytes,
             real_path, description, tags_json, thumbnail, int(is_directory),
             parent_id, meta_json, kind, scan_status, mtime, scan_error_json,
             metadata_confidence, series_id),
        )
        if self._autocommit:
            await self._db.commit()
        log.info("file_indexed", source=source, source_id=source_id, name=name)
        # Auto-enqueue caption job for image rows. The handler is
        # idempotent (skips when description is already set), so this
        # is safe on upserts too. Failure to enqueue must not break
        # registration — captioning is a nice-to-have, not a contract.
        if (
            self._jobs_store is not None
            and mime_type.lower().startswith("image/")
        ):
            try:
                await self._jobs_store.create(
                    user_id=user_id,
                    job_type="file_caption",
                    payload={"file_id": file_id, "user_id": user_id},
                    priority=2,  # below user-initiated work
                    max_attempts=2,
                )
            except Exception:
                log.warning("file_caption_enqueue_failed", file_id=file_id, exc_info=True)
        return file_id

    async def update_scan_state(
        self,
        file_id: str,
        *,
        user_id: str,
        scan_status: str | None = None,
        scan_error: dict | None = None,
        metadata_confidence: float | None = None,
        mtime: int | None = None,
        series_id: str | None = None,
    ) -> bool:
        """Update scan-pipeline fields on a file_index row.

        Used by the comic scan orchestrator to flip ``scanning → ok|error``
        without re-registering the whole row. Any field left ``None`` is not
        touched. ``scan_error=None`` leaves the existing error; pass an empty
        dict to clear it explicitly.
        """
        if not user_id:
            return False
        fields: list[str] = []
        params: list = []
        if scan_status is not None:
            fields.append("scan_status = ?")
            params.append(scan_status)
        if scan_error is not None:
            fields.append("scan_error = ?")
            params.append(json.dumps(scan_error) if scan_error else None)
        if metadata_confidence is not None:
            fields.append("metadata_confidence = ?")
            params.append(metadata_confidence)
        if mtime is not None:
            fields.append("mtime = ?")
            params.append(mtime)
        if series_id is not None:
            fields.append("series_id = ?")
            params.append(series_id)
        if not fields:
            return False
        fields.append("updated_at = datetime('now')")
        params.extend([file_id, user_id])
        cursor = await self._db.execute(
            f"UPDATE file_index SET {', '.join(fields)} WHERE id = ? AND user_id = ?",
            params,
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def unregister(self, source: str, source_id: str, *, user_id: str) -> bool:
        """Remove a file from the index."""
        cursor = await self._db.execute(
            "DELETE FROM file_index WHERE source = ? AND source_id = ? AND user_id = ?",
            (source, source_id, user_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def get(self, file_id: str, *, user_id: str) -> FileEntry | None:
        """Get a single file by ID."""
        cursor = await self._db.execute(
            f"SELECT {_ENTRY_COLUMNS} FROM file_index WHERE id = ? AND user_id = ?",
            (file_id, user_id),
        )
        row = await cursor.fetchone()
        return self._row_to_entry(row) if row else None

    async def get_by_source(self, source: str, source_id: str, *, user_id: str) -> FileEntry | None:
        """Get by source system reference."""
        cursor = await self._db.execute(
            f"SELECT {_ENTRY_COLUMNS} FROM file_index "
            "WHERE source = ? AND source_id = ? AND user_id = ?",
            (source, source_id, user_id),
        )
        row = await cursor.fetchone()
        return self._row_to_entry(row) if row else None

    async def update_source_metadata(
        self, file_id: str, metadata: dict, *, user_id: str,
    ) -> bool:
        """Overwrite the ``source_metadata`` JSON blob for one row.

        Used by features that carry mutable per-item state there — media
        playback progress, YouTube view counts, etc. Scoped by user_id so
        a stray file_id can't let one tenant mutate another's row.
        """
        cursor = await self._db.execute(
            "UPDATE file_index SET source_metadata = ?, updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (json.dumps(metadata), file_id, user_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        source: str | None = None,
        sources: list[str] | None = None,
        exclude_sources: list[str] | None = None,
        mime_filter: str | None = None,
        kind: str | None = None,
        entity_kind: str | None = None,
        entity_kinds: list[str] | None = None,
        exclude_entity_kinds: list[str] | None = None,
        media_status: str | None = None,
        year_from: int = 0,
        year_to: int = 0,
        genre: str = "",
        limit: int = 20,
        offset: int = 0,
        sort: str | None = None,
        match_author: bool = False,
    ) -> list[FileEntry]:
        """Hybrid search: FTS5 text match + optional vector similarity.

        sort=None preserves FTS rank order; pass 'newest'/'oldest'/'name'/'size'
        to override. ``media_status`` filters by source_metadata playback
        state — only meaningful when ``source`` is a media-server slug;
        silently ignored otherwise. ``year_from`` / ``year_to`` bracket
        rows by ``source_metadata.extra.year`` (0 means "no bound");
        rows with no year drop from the result whenever either bound is
        set so explicit year filters return predictable results.
        ``genre`` is a case-insensitive substring match against any
        entry in ``source_metadata.extra.genres``.
        """
        results: list[FileEntry] = []

        # FTS5 search — explicit column list to match _row_to_entry indices.
        # Uses ``_ENTRY_COLUMNS_FI`` (same set as list queries; embedding
        # excluded to skip the BLOB I/O on the hot search path).
        _COLS = _ENTRY_COLUMNS_FI
        # Prefix-aware query build: turns "ra" into "ra*" so as-you-type
        # search doesn't wait for a completed token. See _build_fts_query
        # for the tokenization + sanitisation rules.
        fts_query = _build_fts_query(query)
        if not fts_query:
            # Empty after sanitisation — no meaningful search to run.
            return results
        # Always exclude trashed rows from search hits — list_recent and
        # list_by_source already do this; the FTS path used to leak them
        # because the JOIN didn't carry the filter forward.
        #
        # Author-aware search (``match_author=True``): adds an OR clause
        # against ``source_metadata.author`` (not in the FTS index) so
        # clicking an author name from a Discovery card returns rows.
        # OPT-IN because the OR + LIKE + json_extract forces a full
        # table scan with JSON parsing on every row — measured at 40-80s
        # per call against a 64k-row file_index, blocking every other
        # aiosqlite coroutine on the shared worker thread. Default off:
        # the chat strip and most callers want plain FTS, which is fast
        # (microseconds for typical queries).
        if match_author:
            author_like = f"%{query.lower()}%"
            sql = f"""
                SELECT {_COLS} FROM file_index fi
                LEFT JOIN file_index_fts fts
                  ON fi.rowid = fts.rowid AND file_index_fts MATCH ?
                WHERE fi.user_id = ? AND fi.is_trashed = 0
                  AND (
                    fts.rowid IS NOT NULL
                    OR LOWER(COALESCE(json_extract(fi.source_metadata, '$.author'), '')) LIKE ?
                  )
            """
            params: list = [fts_query, user_id, author_like]
        else:
            # FTS-only fast path — INNER JOIN on FTS so the planner uses
            # the FTS index and only materializes rows that actually
            # matched the query.
            sql = f"""
                SELECT {_COLS} FROM file_index fi
                JOIN file_index_fts fts
                  ON fi.rowid = fts.rowid AND file_index_fts MATCH ?
                WHERE fi.user_id = ? AND fi.is_trashed = 0
            """
            params = [fts_query, user_id]
        # `sources` wins when both provided — it's the "source group"
        # (e.g. the Audiobooks chip expanding to {audiobookshelf, librivox}).
        # Inlining placeholders is safe because we cap N to a small set of
        # known slugs; parameterised binding still carries the values.
        if sources:
            placeholders = ", ".join("?" for _ in sources)
            sql += f" AND fi.source IN ({placeholders})"
            params.extend(sources)
        elif source:
            sql += " AND fi.source = ?"
            params.append(source)
        if exclude_sources:
            placeholders = ", ".join("?" for _ in exclude_sources)
            sql += f" AND fi.source NOT IN ({placeholders})"
            params.extend(exclude_sources)
        if kind:
            sql += " AND fi.kind = ?"
            params.append(kind)
        entity_clause, entity_params = _entity_kind_clause(
            entity_kind=entity_kind,
            entity_kinds=entity_kinds,
            exclude_entity_kinds=exclude_entity_kinds,
            alias="fi",
        )
        if entity_clause:
            sql += f" AND ({entity_clause})"
            params.extend(entity_params)
        if mime_filter:
            sql += " AND fi.mime_type LIKE ?"
            params.append(f"{mime_filter}%")
        # Media playback status filter — only applied when caller asked
        # for one. Predicate is whitelisted via dict lookup so the value
        # can't inject SQL; worst case of an unknown status is silently
        # omitting the filter.
        status_pred = _MEDIA_STATUS_PREDICATES.get((media_status or "").strip())
        if status_pred:
            # json_extract lives on the base table, so reference it via
            # the fi. alias to stay unambiguous.
            sql += f" AND ({status_pred.replace('source_metadata', 'fi.source_metadata')})"
        # Year range — apply only when at least one bound is non-zero.
        # Read the canonical top-level path first, fall back to the
        # secondary $.extra.year copy. CAST because json_extract
        # returns text by default; COALESCE before CAST so a missing
        # field reads as 0 rather than throwing.
        if year_from or year_to:
            year_expr = (
                "CAST(COALESCE("
                "  json_extract(fi.source_metadata, '$.year'),"
                "  json_extract(fi.source_metadata, '$.extra.year'),"
                "  0"
                ") AS INTEGER)"
            )
            sql += f" AND {year_expr} > 0"
            if year_from:
                sql += f" AND {year_expr} >= ?"
                params.append(year_from)
            if year_to:
                sql += f" AND {year_expr} <= ?"
                params.append(year_to)
        # Genre — EXISTS over the genres array, canonical path first.
        if genre:
            sql += (
                " AND EXISTS ("
                "SELECT 1 FROM json_each("
                "  COALESCE("
                "    json_extract(fi.source_metadata, '$.genres'),"
                "    json_extract(fi.source_metadata, '$.extra.genres'),"
                "    '[]'"
                "  )"
                ") g WHERE LOWER(g.value) LIKE ?"
                ")"
            )
            params.append(f"%{genre}%")
        # Media-aware sort keys (author / progress) require us to rewrite
        # the ORDER BY against the fi. alias before use. A sources set is
        # "on media" when every slug in it is a media slug — catches the
        # Audiobooks group expansion without special-casing.
        on_media_source = (
            bool(sources) and all(s in _MEDIA_SOURCES for s in sources)
        ) or bool(source and source in _MEDIA_SOURCES)
        resolved = _resolve_sort(sort, on_media_source=on_media_source) if sort else ""
        if resolved:
            # Prepend fi. to bare column names; json_extract stays as-is.
            order = resolved if "json_extract" in resolved or "LOWER(" in resolved else f"fi.{resolved}"
        else:
            order = "rank"
        sql += f" ORDER BY {order} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        try:
            cursor = await self._db.execute(sql, params)
            rows = await cursor.fetchall()
            for row in rows:
                entry = self._row_to_entry(row)
                if entry:
                    results.append(entry)
        except Exception as err:
            # FTS5 can fail on malformed queries (unbalanced quotes, reserved
            # operators, etc.). Log so we can spot abuse / tooling bugs instead
            # of silently degrading to LIKE every request.
            log.warning(
                "file_search_fts_fallback",
                user_id=user_id, query=query[:120], err=str(err),
            )
            sql = f"""
                SELECT {_ENTRY_COLUMNS} FROM file_index
                WHERE user_id = ? AND is_trashed = 0 AND (name LIKE ? OR description LIKE ?)
            """
            params = [user_id, f"%{query}%", f"%{query}%"]
            if sources:
                placeholders = ", ".join("?" for _ in sources)
                sql += f" AND source IN ({placeholders})"
                params.extend(sources)
            elif source:
                sql += " AND source = ?"
                params.append(source)
            if exclude_sources:
                placeholders = ", ".join("?" for _ in exclude_sources)
                sql += f" AND source NOT IN ({placeholders})"
                params.extend(exclude_sources)
            if kind:
                sql += " AND kind = ?"
                params.append(kind)
            entity_clause, entity_params = _entity_kind_clause(
                entity_kind=entity_kind,
                entity_kinds=entity_kinds,
                exclude_entity_kinds=exclude_entity_kinds,
            )
            if entity_clause:
                sql += f" AND ({entity_clause})"
                params.extend(entity_params)
            sql += f" ORDER BY {_sort_clause(sort)} LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            cursor = await self._db.execute(sql, params)
            rows = await cursor.fetchall()
            for row in rows:
                entry = self._row_to_entry(row)
                if entry:
                    results.append(entry)

        return results

    async def list_by_source(
        self,
        source: str,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        sort: str | None = None,
    ) -> list[FileEntry]:
        """List all files from a specific source."""
        cursor = await self._db.execute(
            f"SELECT {_ENTRY_COLUMNS} FROM file_index "
            f"WHERE source = ? AND user_id = ? AND is_trashed = 0 "
            f"ORDER BY {_sort_clause(sort)} LIMIT ? OFFSET ?",
            (source, user_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [e for r in rows if (e := self._row_to_entry(r))]

    async def list_recent(
        self,
        *,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        sort: str | None = None,
        source: str | None = None,
        sources: list[str] | None = None,
        exclude_sources: list[str] | None = None,
        kind: str | None = None,
        entity_kind: str | None = None,
        entity_kinds: list[str] | None = None,
        exclude_entity_kinds: list[str] | None = None,
        exclude_kinds: list[str] | None = None,
        media_status: str | None = None,
        year_from: int = 0,
        year_to: int = 0,
        genre: str = "",
    ) -> list[FileEntry]:
        """List indexed files, optionally filtered by source/kind/status.

        ``sources`` (a whitelist) takes precedence over ``source`` (a single
        slug) so the Files panel can drive the Audiobooks chip as a group
        query without a second round of SQL.

        ``exclude_sources`` (a blacklist) is applied on top of whichever
        include-filter is active — used by the Local scope toggle to mean
        "everything except cloud providers" without having to enumerate
        every local source. Intersecting include+exclude is allowed and
        naturally returns empty for contradictory combos (e.g. ``sources=
        ['audiobookshelf']`` with ``exclude_sources=['audiobookshelf']``).

        ``year_from`` / ``year_to`` bracket rows by the
        ``source_metadata.extra.year`` field that Emby/Jellyfin sync
        writes for shows and movies. 0 on either bound means "no
        bound." Rows without a year drop from the result whenever any
        bound is set so explicit year filters return predictable
        results (no surprise undated entries in a "1990 — 2010" view).
        """
        sql = f"SELECT {_ENTRY_COLUMNS} FROM file_index WHERE user_id = ? AND is_trashed = 0"
        params: list = [user_id]
        if sources:
            placeholders = ", ".join("?" for _ in sources)
            sql += f" AND source IN ({placeholders})"
            params.extend(sources)
        elif source:
            sql += " AND source = ?"
            params.append(source)
        if exclude_sources:
            placeholders = ", ".join("?" for _ in exclude_sources)
            sql += f" AND source NOT IN ({placeholders})"
            params.extend(exclude_sources)
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        entity_clause, entity_params = _entity_kind_clause(
            entity_kind=entity_kind,
            entity_kinds=entity_kinds,
            exclude_entity_kinds=exclude_entity_kinds,
        )
        if entity_clause:
            sql += f" AND ({entity_clause})"
            params.extend(entity_params)
        if exclude_kinds:
            placeholders = ", ".join("?" for _ in exclude_kinds)
            sql += f" AND (kind IS NULL OR kind NOT IN ({placeholders}))"
            params.extend(exclude_kinds)
        status_pred = _MEDIA_STATUS_PREDICATES.get((media_status or "").strip())
        if status_pred:
            sql += f" AND ({status_pred})"
        # Year range — canonical $.year first, fall back to $.extra.year.
        if year_from or year_to:
            year_expr = (
                "CAST(COALESCE("
                "  json_extract(source_metadata, '$.year'),"
                "  json_extract(source_metadata, '$.extra.year'),"
                "  0"
                ") AS INTEGER)"
            )
            sql += f" AND {year_expr} > 0"
            if year_from:
                sql += f" AND {year_expr} >= ?"
                params.append(year_from)
            if year_to:
                sql += f" AND {year_expr} <= ?"
                params.append(year_to)
        # Genre — canonical $.genres first, fall back to $.extra.genres.
        if genre:
            sql += (
                " AND EXISTS ("
                "SELECT 1 FROM json_each("
                "  COALESCE("
                "    json_extract(source_metadata, '$.genres'),"
                "    json_extract(source_metadata, '$.extra.genres'),"
                "    '[]'"
                "  )"
                ") g WHERE LOWER(g.value) LIKE ?"
                ")"
            )
            params.append(f"%{genre}%")
        if sources:
            on_media_source = all(s in _MEDIA_SOURCES for s in sources)
        elif source:
            on_media_source = source in _MEDIA_SOURCES
        else:
            on_media_source = (kind == "audio")
        resolved = _resolve_sort(sort, on_media_source=on_media_source)
        sql += f" ORDER BY {resolved} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [e for r in rows if (e := self._row_to_entry(r))]

    async def clear_embedding(
        self, file_id: str, *, user_id: str,
    ) -> bool:
        """Reset a file's embedding so the enrichment loop regenerates it.

        Used by the captioner: when a new description lands, the prior
        embedding (typically computed from filename alone, before the
        caption arrived) is now stale. Clearing it puts the row back on
        the ``WHERE embedding IS NULL`` partial index so the next
        ``enrich_pending`` pass regenerates from
        ``name + description + tags`` — now with real semantic content.

        Also removes the corresponding vec0 row so similarity search
        doesn't return a stale vector while waiting for regeneration.
        """
        if not file_id or not user_id:
            return False
        await self._db.execute(
            "UPDATE file_index SET embedding = NULL, "
            "    last_enrichment_attempt = NULL, "
            "    updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (file_id, user_id),
        )
        # Best-effort vec cleanup. Failure here is non-fatal — orphans
        # in file_index_vec naturally drop from search_by_embedding
        # (which INNER JOINs file_index), so a missed delete is a disk
        # cost, not a correctness issue.
        try:
            await self._db.execute(
                "DELETE FROM file_index_vec WHERE file_id = ?", (file_id,),
            )
        except Exception:
            log.debug("clear_embedding_vec_unavailable", file_id=file_id, exc_info=True)
        await self._db.commit()
        return True

    async def _upsert_file_vec(
        self, file_id: str, embedding: bytes,
    ) -> None:
        """Mirror an embedding write into the ``file_index_vec`` vec0 table.

        Called from ``update_enrichment`` and ``enrich_file_atomic`` after
        the base-table commit so the search-time vector leg stays in sync.
        Mirrors the pattern in ``state/discovery_store.upsert_cluster_vec``:
        DELETE + INSERT (vec0 doesn't support UPSERT) and swallow errors
        from environments where sqlite-vec isn't loaded (test fixtures).
        """
        try:
            await self._db.execute(
                "DELETE FROM file_index_vec WHERE file_id = ?", (file_id,),
            )
            await self._db.execute(
                "INSERT INTO file_index_vec (file_id, embedding) VALUES (?, ?)",
                (file_id, embedding),
            )
            await self._db.commit()
        except Exception:
            log.warning(
                "file_index_vec_upsert_unavailable", file_id=file_id, exc_info=True,
            )

    async def search_by_embedding(
        self,
        embedding: bytes,
        *,
        user_id: str,
        limit: int = 20,
        sources: list[str] | None = None,
    ) -> list[FileEntry]:
        """Pure vector similarity search against ``file_index_vec``.

        Caller must pre-compute the query embedding (typically via
        ``EmbeddingService.embed_query``). Returns ranked
        :class:`FileEntry` objects ordered by ascending L2 distance,
        already filtered by ``user_id`` and ``is_trashed = 0``.

        This is the pure-vec leg for the Reference Resolver. Hybrid
        retrieval combines it with the existing FTS-based ``search()``
        via RRF in the resolver module — not in this class.

        vec0 KNN doesn't accept WHERE clauses on auxiliary columns, so
        the user/source filter happens post-fetch. We over-fetch by 3×
        to compensate for shared-tenant deployments where many vec
        rows belong to other users; trims back to ``limit`` after the
        filter.
        """
        try:
            over = max(limit * 3, limit)
            sql = """
                SELECT {cols}, vec.distance AS _vec_distance
                FROM file_index_vec vec
                INNER JOIN file_index fi ON fi.id = vec.file_id
                WHERE vec.embedding MATCH ? AND vec.k = ?
                  AND fi.user_id = ?
                  AND fi.is_trashed = 0
                ORDER BY vec.distance
            """.format(cols=_ENTRY_COLUMNS_FI)
            cursor = await self._db.execute(sql, (embedding, over, user_id))
            rows = await cursor.fetchall()
        except Exception:
            log.warning("file_index_vec_search_unavailable", exc_info=True)
            return []

        out: list[FileEntry] = []
        allowed = set(sources or [])
        for r in rows:
            entry = self._row_to_entry(r)
            if entry is None:
                continue
            if allowed and entry.source not in allowed:
                continue
            # Project the vec distance onto the standard FileEntry.score
            # field so callers can rank uniformly with FTS results.
            # distance is L2 in [0, ~2.0]; convert to a similarity-like
            # score where smaller distance => higher score.
            entry.score = max(0.0, 1.0 - float(r[-1]))
            out.append(entry)
            if len(out) >= limit:
                break
        return out

    async def update_enrichment(
        self,
        file_id: str,
        *,
        user_id: str,
        description: str | None = None,
        tags: list[str] | None = None,
        thumbnail: str | None = None,
        embedding: bytes | None = None,
        source_metadata: dict | None = None,
    ) -> bool:
        """Update AI-generated enrichment fields.

        ``source_metadata`` merges into the existing JSON blob (rather
        than overwriting) so adapters that seeded keys at register time
        — e.g. media_server's ``cover_url`` — aren't clobbered by later
        enrichment passes. Pass an empty dict to no-op the merge.
        """
        updates = []
        params: list = []
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if thumbnail is not None:
            updates.append("thumbnail = ?")
            params.append(thumbnail)
        if embedding is not None:
            updates.append("embedding = ?")
            params.append(embedding)
        if source_metadata:
            cursor = await self._db.execute(
                "SELECT source_metadata FROM file_index WHERE id = ? AND user_id = ?",
                (file_id, user_id),
            )
            row = await cursor.fetchone()
            existing: dict = {}
            if row and row[0]:
                try:
                    existing = json.loads(row[0]) or {}
                except (ValueError, TypeError):
                    existing = {}
            existing.update(source_metadata)
            updates.append("source_metadata = ?")
            params.append(json.dumps(existing))
        if not updates:
            return False
        updates.append("updated_at = datetime('now')")
        params.extend([file_id, user_id])
        await self._db.execute(
            f"UPDATE file_index SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
            params,
        )
        await self._db.commit()
        # Mirror embedding into the vec0 table for similarity search.
        if embedding is not None:
            await self._upsert_file_vec(file_id, embedding)
        return True

    async def enrich_file_atomic(
        self,
        file_id: str,
        *,
        user_id: str,
        description: str | None = None,
        tags: list[str] | None = None,
        thumbnail: str | None = None,
        embedding: bytes | None = None,
        source_metadata_merge: dict | None = None,
        stamp_attempt: bool = False,
    ) -> bool:
        """Apply all enrichment fields to a file in ONE transaction.

        Used by the background enrichment loop, which previously called
        ``update_enrichment`` 2-4 times per file (each its own commit +
        fsync) and then ``_stamp_enrichment_attempt`` for a 5th. On a
        32-file batch that was ~150 commits per pass, each holding the
        writer lock and blocking every other connection. Folding the
        per-file work into a single ``BEGIN IMMEDIATE`` / ``COMMIT``
        cuts that to one commit per file.

        Pass ``stamp_attempt=True`` to also bump
        ``last_enrichment_attempt`` in the same transaction so the
        loop's hour-backoff applies even when no enrichment fields
        were produced (e.g., a malformed EPUB whose extractor returned
        nothing).
        """
        updates: list[str] = []
        params: list = []
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if thumbnail is not None:
            updates.append("thumbnail = ?")
            params.append(thumbnail)
        if embedding is not None:
            updates.append("embedding = ?")
            params.append(embedding)

        needs_meta_merge = bool(source_metadata_merge)
        if not updates and not needs_meta_merge and not stamp_attempt:
            return False

        await self._db.execute("BEGIN IMMEDIATE")
        try:
            if needs_meta_merge:
                # Read inside the transaction so a concurrent enrichment
                # for the same file can't lose either side's keys.
                cursor = await self._db.execute(
                    "SELECT source_metadata FROM file_index "
                    "WHERE id = ? AND user_id = ?",
                    (file_id, user_id),
                )
                row = await cursor.fetchone()
                existing: dict = {}
                if row and row[0]:
                    try:
                        existing = json.loads(row[0]) or {}
                    except (ValueError, TypeError):
                        existing = {}
                existing.update(source_metadata_merge)
                updates.append("source_metadata = ?")
                params.append(json.dumps(existing))

            if stamp_attempt:
                updates.append("last_enrichment_attempt = datetime('now')")
            updates.append("updated_at = datetime('now')")
            params.extend([file_id, user_id])

            await self._db.execute(
                f"UPDATE file_index SET {', '.join(updates)} "
                "WHERE id = ? AND user_id = ?",
                params,
            )
            await self._db.commit()
        except Exception:
            try:
                await self._db.rollback()
            except Exception as rb_exc:
                log.debug(
                    "vfs_enrich_file_rollback_failed",
                    error=str(rb_exc),
                )
            raise
        # Mirror embedding into the vec0 table after the main commit so a
        # vec failure can't roll back the embedding write itself. Skipped
        # when no embedding was supplied (most enrichment passes don't
        # update the vector every time).
        if embedding is not None:
            await self._upsert_file_vec(file_id, embedding)
        return True

    async def stats(self, *, user_id: str) -> dict:
        """Return file counts by source + kind, plus favorites/trash totals.

        Shape:
          {
            "by_source": {name: {count, size_bytes}},
            "by_kind":   {name: {count, size_bytes}},
            "favorites": N, "trash": N,
            "total_count": N, "total_size": N,
          }
        """
        cursor = await self._db.execute(
            "SELECT source, COUNT(*), SUM(size_bytes) FROM file_index "
            "WHERE user_id = ? AND is_trashed = 0 GROUP BY source",
            (user_id,),
        )
        by_source = {r[0]: {"count": r[1], "size_bytes": r[2] or 0}
                     for r in await cursor.fetchall()}

        cursor = await self._db.execute(
            "SELECT kind, COUNT(*), SUM(size_bytes) FROM file_index "
            "WHERE user_id = ? AND is_trashed = 0 AND kind != '' GROUP BY kind",
            (user_id,),
        )
        by_kind = {r[0]: {"count": r[1], "size_bytes": r[2] or 0}
                   for r in await cursor.fetchall()}

        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM file_index WHERE user_id = ? AND is_favorite = 1 AND is_trashed = 0",
            (user_id,),
        )
        favorites = (await cursor.fetchone())[0]
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM file_index WHERE user_id = ? "
            f"AND is_trashed = 1 AND {NOT_DETACHED}",
            (user_id,),
        )
        trash = (await cursor.fetchone())[0]

        total_count = sum(v["count"] for v in by_source.values())
        total_size = sum(v["size_bytes"] for v in by_source.values())
        return {
            "by_source": by_source, "by_kind": by_kind,
            "favorites": favorites, "trash": trash,
            "total_count": total_count, "total_size": total_size,
        }

    async def count(self, *, user_id: str, kind: str | None = None) -> int:
        """Return file count for user, optionally narrowed to one kind.

        Used by callers (e.g. cast_routes' Comics rail) that need to
        pre-size a fetch to the user's actual library so they don't
        silently truncate large catalogues.
        """
        if kind:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM file_index "
                "WHERE user_id = ? AND kind = ? AND is_trashed = 0",
                (user_id, kind),
            )
        else:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM file_index WHERE user_id = ? AND is_trashed = 0",
                (user_id,),
            )
        return (await cursor.fetchone())[0]

    async def toggle_favorite(self, file_id: str, *, user_id: str) -> bool:
        """Toggle is_favorite flag. Returns new state."""
        cursor = await self._db.execute(
            "UPDATE file_index SET is_favorite = 1 - is_favorite, updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (file_id, user_id),
        )
        await self._db.commit()
        if cursor.rowcount == 0:
            return False
        row = await (await self._db.execute(
            "SELECT is_favorite FROM file_index WHERE id = ?", (file_id,)
        )).fetchone()
        return bool(row[0]) if row else False

    async def list_favorites(
        self,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        sort: str | None = None,
        query: str | None = None,
    ) -> list[FileEntry]:
        """List favorited files, optionally filtered by a name/description
        substring. Uses LIKE rather than FTS5 because favorites are a
        curated subset (typically hundreds, not tens of thousands) where
        a full-table LIKE scan is fast enough without a join into the
        ``file_index_fts`` virtual table.
        """
        sql = (
            f"SELECT {_ENTRY_COLUMNS} FROM file_index "
            "WHERE user_id = ? AND is_favorite = 1 AND is_trashed = 0"
        )
        params: list = [user_id]
        needle = (query or "").strip().lower()
        if needle:
            sql += (
                " AND (LOWER(name) LIKE ? "
                "OR LOWER(COALESCE(description, '')) LIKE ?)"
            )
            params.extend([f"%{needle}%", f"%{needle}%"])
        sql += f" ORDER BY {_sort_clause(sort)} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [e for r in rows if (e := self._row_to_entry(r))]

    async def soft_delete(self, file_id: str, *, user_id: str) -> bool:
        """Move file to trash (soft delete)."""
        cursor = await self._db.execute(
            "UPDATE file_index SET is_trashed = 1, trashed_at = datetime('now'), "
            "updated_at = datetime('now') WHERE id = ? AND user_id = ? AND is_trashed = 0",
            (file_id, user_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def restore(self, file_id: str, *, user_id: str) -> bool:
        """Restore file from trash.

        Refuses detached rows (migration 323). They never surface in the
        Trash listing, but this takes a raw ``file_id``, so a direct call
        would otherwise un-hide an item whose server is unreachable —
        recreating the exact ghost state the detach existed to clear.
        Re-attachment happens on resync after a re-share, not here.
        """
        cursor = await self._db.execute(
            "UPDATE file_index SET is_trashed = 0, trashed_at = NULL, "
            "updated_at = datetime('now') WHERE id = ? AND user_id = ? "
            f"AND is_trashed = 1 AND {NOT_DETACHED}",
            (file_id, user_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def list_trash(
        self,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        sort: str | None = None,
        query: str | None = None,
    ) -> list[FileEntry]:
        """List trashed files, optionally sorted and filtered.

        Default sort is ``trashed_at DESC`` — most-recently-deleted first,
        which matches the "I just deleted the wrong thing, let me restore
        it" flow. An explicit ``sort`` value overrides. Like favorites,
        substring search is LIKE-based since the trash set is small.

        Detached rows (migration 323) are excluded: the user didn't
        delete them, so offering them a Restore button here would
        resurrect a row pointing at a server they still can't reach.
        """
        sql = (
            f"SELECT {_ENTRY_COLUMNS} FROM file_index "
            f"WHERE user_id = ? AND is_trashed = 1 AND {NOT_DETACHED}"
        )
        params: list = [user_id]
        needle = (query or "").strip().lower()
        if needle:
            sql += (
                " AND (LOWER(name) LIKE ? "
                "OR LOWER(COALESCE(description, '')) LIKE ?)"
            )
            params.extend([f"%{needle}%", f"%{needle}%"])
        order = _sort_clause(sort) if sort else "trashed_at DESC"
        sql += f" ORDER BY {order} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [e for r in rows if (e := self._row_to_entry(r))]

    async def purge_trash(self, *, user_id: str, older_than_days: int | None = None) -> int:
        """Permanently delete trashed files.

        Backs the user's "Empty Trash" action. Detached rows are exempt
        (migration 323) — they never appeared in the Trash listing, so
        emptying it must not delete them.
        """
        if older_than_days is not None:
            cursor = await self._db.execute(
                "DELETE FROM file_index WHERE user_id = ? AND is_trashed = 1 "
                f"AND {NOT_DETACHED} AND trashed_at < datetime('now', ?)",
                (user_id, f"-{older_than_days} days"),
            )
        else:
            cursor = await self._db.execute(
                "DELETE FROM file_index WHERE user_id = ? AND is_trashed = 1 "
                f"AND {NOT_DETACHED}",
                (user_id,),
            )
        await self._db.commit()
        return cursor.rowcount

    async def list_trashed_older_than(
        self, days: int, *, limit: int = 1000,
    ) -> list[FileEntry]:
        """All-users variant of list_trash, filtered by age. Used by the
        maintenance loop to dispatch per-source adapter cleanup BEFORE
        the rows themselves are deleted (so blob refcounts get released
        instead of leaking).

        Detached rows are exempt (migration 323) — they're preserved
        history, not aged-out deletions.
        """
        cursor = await self._db.execute(
            f"SELECT {_ENTRY_COLUMNS} FROM file_index WHERE is_trashed = 1 "
            f"AND {NOT_DETACHED} "
            "AND trashed_at < datetime('now', ?) "
            "ORDER BY trashed_at ASC LIMIT ?",
            (f"-{int(days)} days", int(limit)),
        )
        rows = await cursor.fetchall()
        return [e for r in rows if (e := self._row_to_entry(r))]

    async def purge_all_old_trash(self, older_than_days: int = 30) -> int:
        """Purge trashed files older than N days across all users.

        The most dangerous of the trash-semantics paths: it runs
        unattended, across every user, and hard-deletes. Without the
        detached exclusion (migration 323) it would destroy preserved
        share history 30 days after an un-share, with no user action and
        no trace.
        """
        cursor = await self._db.execute(
            "DELETE FROM file_index WHERE is_trashed = 1 "
            f"AND {NOT_DETACHED} "
            "AND trashed_at < datetime('now', ?)",
            (f"-{older_than_days} days",),
        )
        await self._db.commit()
        return cursor.rowcount

    async def update_tags(self, file_id: str, *, tags: list[str], user_id: str) -> bool:
        """Replace tags for a file."""
        cursor = await self._db.execute(
            "UPDATE file_index SET tags = ?, updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (json.dumps(tags), file_id, user_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    def _row_to_entry(self, row) -> FileEntry | None:
        if not row:
            return None
        # Column order matches ``_ENTRY_COLUMNS`` — the embedding BLOB
        # is no longer selected, so the indexes shift up after position
        # 10 vs. the old ``SELECT *`` layout.
        # 0=id 1=user_id 2=source 3=source_id 4=name 5=mime_type
        # 6=size_bytes 7=real_path 8=description 9=tags 10=thumbnail
        # 11=is_directory 12=parent_id 13=source_metadata
        # 14=created_at 15=updated_at 16=is_favorite 17=is_trashed
        # 18=trashed_at 19=kind 20=last_played_at 21=series_id
        def g(i):
            return row[i] if i < len(row) else None
        try:
            tags = json.loads(g(9)) if g(9) else []
        except (json.JSONDecodeError, TypeError):
            tags = []
        try:
            meta = json.loads(g(13)) if g(13) else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return FileEntry(
            id=g(0) or "",
            user_id=g(1) or "",
            source=g(2) or "",
            source_id=g(3) or "",
            name=g(4) or "",
            mime_type=g(5) or "",
            size_bytes=g(6) or 0,
            real_path=g(7),
            description=g(8) or "",
            tags=tags,
            thumbnail=g(10),
            embedding=None,  # never selected; field kept for back-compat
            is_directory=bool(g(11)),
            parent_id=g(12),
            source_metadata=meta,
            created_at=g(14) or "",
            updated_at=g(15) or "",
            is_favorite=bool(g(16)),
            is_trashed=bool(g(17)),
            trashed_at=g(18),
            kind=g(19) or "",
            last_played_at=g(20) or "",
            series_id=g(21),
        )
