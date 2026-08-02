"""Comic series identity store.

Series get stable IDs from first-ingest; canonical_name / cover / description
update freely as metadata improves. Archives FK to ``comic_series.id`` via
``file_index.series_id``, so favorites, bookmarks, and collections that
reference a series never break when the series renames.

The "same name → same id" guarantee comes from looking up on ``sort_name``
(lowercased, articles stripped, whitespace collapsed). Any scan that ingests
"The Walking Dead" once and "Walking Dead" later resolves to the same series.

Written for Phase A of the comic library plan
(``docs/superpowers/plans/2026-04-20-comic-library-phase-a-scan-infrastructure.md``).
Tier-2 enrichment (Phase D) will flip ``metadata_source`` from ``filename`` to
``external_api`` and raise ``metadata_confidence`` accordingly.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import asdict, dataclass, field
from typing import Any

import aiosqlite
import structlog

log = structlog.get_logger(__name__)


# --- Module-level store registration ---------------------------------------
# Mirrors the ``augmentum.vfs.set_file_index`` pattern: server startup
# instantiates one ``ComicSeriesStore`` against the shared aiosqlite
# connection and registers it here; downstream callers (sync.py,
# per-page route, future Phase B surface) pull the configured instance
# without having to thread the DB connection through every layer.

_comic_series_store: ComicSeriesStore | None = None  # forward-declared


def set_comic_series_store(store: ComicSeriesStore) -> None:
    global _comic_series_store
    _comic_series_store = store


def get_comic_series_store() -> ComicSeriesStore | None:
    """Return the configured store, or ``None`` if startup hasn't wired it.

    Callers should treat ``None`` as a benign "feature not yet configured"
    state — the same posture :mod:`augmentum.vfs` takes for file_index.
    """
    return _comic_series_store


_ARTICLE_RE = re.compile(r"^\s*(?:the|a|an)\s+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def derive_sort_name(name: str) -> str:
    """Normalize a series name for deduplication + sort ordering.

    - lowercase
    - strip leading article (``The``, ``A``, ``An``)
    - collapse whitespace
    - keep punctuation (JoJo's Bizarre Adventure vs JoJos should differ —
      only normalize the things that commonly vary across the same series).

    Pure function; testable without a DB. Any two filenames that *should*
    resolve to the same series must produce the same ``sort_name``.
    """
    s = (name or "").strip()
    s = _ARTICLE_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s)
    return s.lower().strip()


@dataclass(slots=True)
class ComicSeries:
    id: str
    user_id: str
    canonical_name: str
    sort_name: str
    alias_names: list[str] = field(default_factory=list)
    publisher: str | None = None
    author: str | None = None
    description: str | None = None
    cover_file_id: str | None = None
    status: str | None = None
    year_started: int | None = None
    year_ended: int | None = None
    genres: list[str] = field(default_factory=list)
    language_iso: str | None = None
    age_rating: str | None = None
    metadata_source: str | None = None
    metadata_confidence: float = 0.5
    archive_count_reported: int | None = None
    accent_color: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row_to_series(row: aiosqlite.Row) -> ComicSeries:
    return ComicSeries(
        id=row["id"],
        user_id=row["user_id"],
        canonical_name=row["canonical_name"],
        sort_name=row["sort_name"],
        alias_names=json.loads(row["alias_names"] or "[]"),
        publisher=row["publisher"],
        author=row["author"],
        description=row["description"],
        cover_file_id=row["cover_file_id"],
        status=row["status"],
        year_started=row["year_started"],
        year_ended=row["year_ended"],
        genres=json.loads(row["genres"] or "[]"),
        language_iso=row["language_iso"],
        age_rating=row["age_rating"],
        metadata_source=row["metadata_source"],
        metadata_confidence=row["metadata_confidence"],
        archive_count_reported=row["archive_count_reported"],
        accent_color=row["accent_color"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class ComicSeriesStore:
    """CRUD + identity resolution for ``comic_series``.

    Always user-scoped. A series in user A's library is isolated from
    user B's even if the canonical_name matches — two users with their own
    "Berserk" entries is fine and expected.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        autocommit: bool = True,
    ) -> None:
        self._conn = conn
        self._conn.row_factory = aiosqlite.Row
        # See FileIndexService.__init__ — same rationale. Comic catalog
        # sync calls create_or_resolve_series (and often update_series)
        # from inside the same per-item loop as the file_index write, so
        # a comic library paid two to three commits per item. Bulk
        # callers batch through augmentum.vfs.bulk.
        self._autocommit = autocommit

    async def _maybe_commit(self) -> None:
        if self._autocommit:
            await self._conn.commit()

    # --- Identity resolution ------------------------------------------------

    async def create_or_resolve_series(
        self,
        *,
        user_id: str,
        name: str,
        metadata_source: str = "filename",
        metadata_confidence: float = 0.5,
        publisher: str | None = None,
        author: str | None = None,
        language_iso: str | None = None,
        year_started: int | None = None,
    ) -> str:
        """Return the series_id for this (user_id, name).

        On first call for a given sort_name, creates a new row and returns the
        new ID. On subsequent calls, returns the existing ID without touching
        any existing metadata — tier-2 enrichment updates are a separate
        ``update_series`` call so identity resolution stays idempotent.
        """
        if not user_id:
            raise ValueError("comic_series requires a user_id")
        if not name or not name.strip():
            raise ValueError("comic_series requires a non-empty name")

        sort = derive_sort_name(name)

        cursor = await self._conn.execute(
            "SELECT id FROM comic_series WHERE user_id = ? AND sort_name = ?",
            (user_id, sort),
        )
        row = await cursor.fetchone()
        if row is not None:
            return row["id"]

        series_id = f"cs_{secrets.token_hex(12)}"
        await self._conn.execute(
            """INSERT INTO comic_series
               (id, user_id, canonical_name, sort_name,
                metadata_source, metadata_confidence,
                publisher, author, language_iso, year_started)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                series_id, user_id, name.strip(), sort,
                metadata_source, metadata_confidence,
                publisher, author, language_iso, year_started,
            ),
        )
        await self._maybe_commit()
        log.info(
            "comic_series_created",
            id=series_id, user_id=user_id, name=name.strip(),
            source=metadata_source, confidence=metadata_confidence,
        )
        return series_id

    # --- Read ----------------------------------------------------------------

    async def get(self, series_id: str, *, user_id: str) -> ComicSeries | None:
        cursor = await self._conn.execute(
            "SELECT * FROM comic_series WHERE id = ? AND user_id = ?",
            (series_id, user_id),
        )
        row = await cursor.fetchone()
        return _row_to_series(row) if row else None

    async def list_series(
        self,
        *,
        user_id: str,
        sort: str = "sort_name",
        limit: int = 50,
        offset: int = 0,
    ) -> list[ComicSeries]:
        """List all series for a user.

        ``sort`` ∈ {``sort_name``, ``updated_at``, ``created_at``}.
        Pagination is offset-based here because Phase A doesn't need cursors —
        the library surface (Phase B) layers cursor pagination on top.
        """
        sort_sql = {
            "sort_name": "sort_name ASC",
            "updated_at": "updated_at DESC",
            "created_at": "created_at DESC",
        }.get(sort, "sort_name ASC")
        cursor = await self._conn.execute(
            f"SELECT * FROM comic_series WHERE user_id = ? "
            f"ORDER BY {sort_sql} LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [_row_to_series(r) for r in rows]

    async def count(self, *, user_id: str) -> int:
        cursor = await self._conn.execute(
            "SELECT COUNT(*) as n FROM comic_series WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        return int(row["n"]) if row else 0

    # --- Update --------------------------------------------------------------

    async def update_series(
        self,
        series_id: str,
        *,
        user_id: str,
        canonical_name: str | None = None,
        alias_names: list[str] | None = None,
        publisher: str | None = None,
        author: str | None = None,
        description: str | None = None,
        cover_file_id: str | None = None,
        status: str | None = None,
        year_started: int | None = None,
        year_ended: int | None = None,
        genres: list[str] | None = None,
        language_iso: str | None = None,
        age_rating: str | None = None,
        metadata_source: str | None = None,
        metadata_confidence: float | None = None,
        archive_count_reported: int | None = None,
        accent_color: str | None = None,
    ) -> bool:
        """Partial update. Any field left ``None`` is preserved.

        Renaming (``canonical_name``) also updates ``sort_name`` derived from
        the new name — but the series ID is stable, so inbound references
        (file_index.series_id, favorites, bookmarks) keep working.
        """
        if not user_id:
            return False

        fields: list[str] = []
        params: list[Any] = []

        if canonical_name is not None:
            fields.append("canonical_name = ?")
            params.append(canonical_name.strip())
            fields.append("sort_name = ?")
            params.append(derive_sort_name(canonical_name))
        if alias_names is not None:
            fields.append("alias_names = ?")
            params.append(json.dumps(alias_names))
        if publisher is not None:
            fields.append("publisher = ?")
            params.append(publisher)
        if author is not None:
            fields.append("author = ?")
            params.append(author)
        if description is not None:
            fields.append("description = ?")
            params.append(description)
        if cover_file_id is not None:
            fields.append("cover_file_id = ?")
            params.append(cover_file_id)
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if year_started is not None:
            fields.append("year_started = ?")
            params.append(year_started)
        if year_ended is not None:
            fields.append("year_ended = ?")
            params.append(year_ended)
        if genres is not None:
            fields.append("genres = ?")
            params.append(json.dumps(genres))
        if language_iso is not None:
            fields.append("language_iso = ?")
            params.append(language_iso)
        if age_rating is not None:
            fields.append("age_rating = ?")
            params.append(age_rating)
        if metadata_source is not None:
            fields.append("metadata_source = ?")
            params.append(metadata_source)
        if metadata_confidence is not None:
            fields.append("metadata_confidence = ?")
            params.append(metadata_confidence)
        if archive_count_reported is not None:
            fields.append("archive_count_reported = ?")
            params.append(archive_count_reported)
        if accent_color is not None:
            fields.append("accent_color = ?")
            params.append(accent_color)

        if not fields:
            return False

        fields.append("updated_at = datetime('now')")
        params.extend([series_id, user_id])

        cursor = await self._conn.execute(
            f"UPDATE comic_series SET {', '.join(fields)} "
            f"WHERE id = ? AND user_id = ?",
            params,
        )
        await self._maybe_commit()
        return cursor.rowcount > 0

    # --- Delete --------------------------------------------------------------

    async def delete(self, series_id: str, *, user_id: str) -> bool:
        """Delete a series row. Does NOT null out ``file_index.series_id`` —
        callers that need cascade behavior should do it explicitly.
        """
        cursor = await self._conn.execute(
            "DELETE FROM comic_series WHERE id = ? AND user_id = ?",
            (series_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0
