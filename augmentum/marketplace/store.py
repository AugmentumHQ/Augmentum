"""MarketplaceStore -- CRUD over the curated catalog table.

Listings are server-level data (every user sees the same catalog), so
no user_id scoping. Per-user state (whether an item is installed,
ratings the user has left) lives in the existing artifacts table and
the future ``marketplace_reviews`` table.

Reads are the dominant operation; writes happen on startup (catalog
load) and rarely otherwise (admin upserts). The store keeps both paths
simple -- no caching layer here; SQLite indexed reads are already
sub-millisecond at our catalog scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class MarketplaceListing:
    id: str
    publisher: str
    title: str
    kind: str
    runtime_preferred: str
    runtime_alternates: tuple[str, ...]
    tagline: str
    description: str
    thumbnail_url: str
    source_url: str
    embed_url: str
    install_via: str                                # source_id of the underlying installer
    install_payload: dict[str, Any]
    capabilities: dict[str, Any]
    metadata: dict[str, Any]
    rating: float | None
    install_count: int
    signature: str
    listed_at: str
    delisted_at: str | None = None
    # Discover surface fields (migration 254). category is the top-level
    # group ("providers" | "games" | "characters" | "powers" |
    # "reasoning-flows" | "knowledge" | "other"); tags is a flat
    # filter-chip list; featured drives the homepage rail.
    category: str = ""
    tags: tuple[str, ...] = ()
    featured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "publisher": self.publisher,
            "title": self.title,
            "kind": self.kind,
            "runtime_preferred": self.runtime_preferred,
            "runtime_alternates": list(self.runtime_alternates),
            "tagline": self.tagline,
            "description": self.description,
            "thumbnail_url": self.thumbnail_url,
            "source_url": self.source_url,
            "embed_url": self.embed_url,
            "install_via": self.install_via,
            "install_payload": dict(self.install_payload),
            "capabilities": dict(self.capabilities),
            "metadata": dict(self.metadata),
            "rating": self.rating,
            "install_count": self.install_count,
            "signature": self.signature,
            "listed_at": self.listed_at,
            "delisted_at": self.delisted_at,
            "category": self.category,
            "tags": list(self.tags),
            "featured": self.featured,
        }


class MarketplaceStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    # ── Reads ────────────────────────────────────────────────────────

    async def list_active(
        self,
        *,
        kind: str | None = None,
        publisher: str | None = None,
        limit: int = 200,
    ) -> list[MarketplaceListing]:
        query = (
            "SELECT * FROM marketplace_listings "
            "WHERE delisted_at IS NULL"
        )
        params: list[Any] = []
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        if publisher:
            query += " AND publisher = ?"
            params.append(publisher)
        query += " ORDER BY listed_at DESC LIMIT ?"
        params.append(int(limit))
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [_row_to_listing(dict(zip(cols, r))) for r in rows]

    async def list_for_discover(
        self,
        *,
        category: str | None = None,
        kind: str | None = None,
        publisher: str | None = None,
        featured_only: bool = False,
        search: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> list[MarketplaceListing]:
        """Filtered list for the Discover surface.

        Layered on top of `list_active`'s filters with category /
        featured / search. Search matches title, tagline, and tags
        with a case-insensitive LIKE; cheap at our catalog scale.
        """
        query = (
            "SELECT * FROM marketplace_listings WHERE delisted_at IS NULL"
        )
        params: list[Any] = []
        if category:
            query += " AND category = ?"
            params.append(category)
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        if publisher:
            query += " AND publisher = ?"
            params.append(publisher)
        if featured_only:
            query += " AND featured = 1"
        if search:
            # Title + tagline + tags JSON blob. LIKE on tags JSON is
            # a flat substring match — good enough for chip-shaped
            # tags. FTS5 would be the upgrade if the catalog grows
            # beyond a few hundred entries.
            query += (
                " AND (title LIKE ? OR tagline LIKE ? OR tags LIKE ?)"
            )
            needle = f"%{search}%"
            params.extend([needle, needle, needle])
        # Ordering: featured first, then newest. Keeps the homepage
        # rail at the top of any unfiltered query without a second
        # round-trip.
        query += " ORDER BY featured DESC, listed_at DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [_row_to_listing(dict(zip(cols, r))) for r in rows]

    async def get(self, listing_id: str) -> MarketplaceListing | None:
        cursor = await self._conn.execute(
            "SELECT * FROM marketplace_listings WHERE id = ?",
            (listing_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        return _row_to_listing(dict(zip(cols, row)))

    async def count_active(self) -> int:
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM marketplace_listings WHERE delisted_at IS NULL"
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    # ── Writes ───────────────────────────────────────────────────────

    async def upsert(self, listing: MarketplaceListing) -> None:
        """Insert or replace a listing. Used by the catalog loader."""
        await self._conn.execute(
            """INSERT INTO marketplace_listings
               (id, publisher, title, kind, runtime_preferred,
                runtime_alternates, tagline, description, thumbnail_url,
                source_url, embed_url, install_via, install_payload,
                capabilities, metadata, rating, install_count, signature,
                category, tags, featured)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 publisher=excluded.publisher,
                 title=excluded.title,
                 kind=excluded.kind,
                 runtime_preferred=excluded.runtime_preferred,
                 runtime_alternates=excluded.runtime_alternates,
                 tagline=excluded.tagline,
                 description=excluded.description,
                 thumbnail_url=excluded.thumbnail_url,
                 source_url=excluded.source_url,
                 embed_url=excluded.embed_url,
                 install_via=excluded.install_via,
                 install_payload=excluded.install_payload,
                 capabilities=excluded.capabilities,
                 metadata=excluded.metadata,
                 rating=excluded.rating,
                 signature=excluded.signature,
                 category=excluded.category,
                 tags=excluded.tags,
                 featured=excluded.featured,
                 delisted_at=NULL""",
            (
                listing.id,
                listing.publisher,
                listing.title,
                listing.kind,
                listing.runtime_preferred,
                json.dumps(list(listing.runtime_alternates)),
                listing.tagline,
                listing.description,
                listing.thumbnail_url,
                listing.source_url,
                listing.embed_url,
                listing.install_via,
                json.dumps(listing.install_payload),
                json.dumps(listing.capabilities),
                json.dumps(listing.metadata),
                listing.rating,
                int(listing.install_count or 0),
                listing.signature,
                listing.category,
                json.dumps(list(listing.tags)),
                1 if listing.featured else 0,
            ),
        )
        await self._conn.commit()

    async def delist_missing(self, keep_ids: set[str]) -> int:
        """Soft-delete any listing not in ``keep_ids``.

        DEPRECATED for multi-loader use — sweeps GLOBALLY across all
        publishers, which means the titles loader would delist provider
        rows and vice versa. Kept for backwards compatibility with
        single-loader callers; new code should use
        ``delist_missing_for_publisher`` so each loader's delist sweep
        is scoped to the rows it owns.
        """
        if not keep_ids:
            return 0
        placeholders = ",".join(["?"] * len(keep_ids))
        cursor = await self._conn.execute(
            f"UPDATE marketplace_listings "
            f"SET delisted_at = datetime('now') "
            f"WHERE delisted_at IS NULL AND id NOT IN ({placeholders})",
            tuple(keep_ids),
        )
        await self._conn.commit()
        return cursor.rowcount or 0

    async def delist_missing_for_publisher(
        self, keep_ids: set[str], *, publisher: str,
    ) -> int:
        """Publisher-scoped delist sweep.

        Each loader owns a publisher namespace (titles loader →
        ``augmentum``, providers loader → ``augmentum-providers``,
        future community loader → ``community:<handle>``) and only
        sweeps rows under its own publisher. This prevents loaders
        from clobbering each other's catalog.

        Pass ``keep_ids=set()`` to delist EVERY active row for the
        publisher — useful when a loader source is empty.
        """
        # No early return on empty set — empty set means "delist all
        # publisher rows" here, which is a valid sweep result.
        placeholders = ",".join(["?"] * len(keep_ids)) if keep_ids else ""
        not_in_clause = f" AND id NOT IN ({placeholders})" if keep_ids else ""
        cursor = await self._conn.execute(
            f"UPDATE marketplace_listings "
            f"SET delisted_at = datetime('now') "
            f"WHERE delisted_at IS NULL AND publisher = ?{not_in_clause}",
            (publisher, *keep_ids),
        )
        await self._conn.commit()
        return cursor.rowcount or 0

    async def delist_community_listings(self) -> int:
        """Soft-delist every active ``community:*`` listing.

        Used by the startup reconciler when the community feed is
        disabled: the feed is the only loader for community rows, so if
        it's off, any lingering rows are stale placeholders (e.g. the
        sample "example" Character/Power cards) that should not surface.
        Publisher-prefix scoped so augmentum / augmentum-providers rows
        are never touched.
        """
        cursor = await self._conn.execute(
            "UPDATE marketplace_listings "
            "SET delisted_at = datetime('now') "
            "WHERE delisted_at IS NULL AND publisher LIKE 'community:%'",
        )
        await self._conn.commit()
        return cursor.rowcount or 0

    async def increment_install_count(self, listing_id: str) -> None:
        await self._conn.execute(
            "UPDATE marketplace_listings "
            "SET install_count = install_count + 1 "
            "WHERE id = ?",
            (listing_id,),
        )
        await self._conn.commit()

    # ── Per-user install audit ───────────────────────────────────────

    async def record_install(
        self,
        *,
        user_id: str,
        listing_id: str,
        install_via: str,
        kind: str,
        resource_id: str,
    ) -> str:
        """Write a row to marketplace_installs.

        Idempotent against re-installs of the same listing: if a row
        exists with uninstalled_at IS NULL (the unique partial index),
        the INSERT fails silently — caller can treat that as a no-op
        success because the listing is already installed.

        Returns the install audit id (or empty string if the unique
        constraint kicked in — meaning a prior active row exists).
        """
        import uuid
        install_id = f"mki_{uuid.uuid4().hex[:16]}"
        try:
            await self._conn.execute(
                """INSERT INTO marketplace_installs
                   (id, user_id, listing_id, install_via, kind, resource_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (install_id, user_id, listing_id, install_via, kind, resource_id),
            )
            await self._conn.commit()
            return install_id
        except Exception:
            # Unique constraint on (user_id, listing_id) WHERE
            # uninstalled_at IS NULL caught a duplicate. That just
            # means the listing is already installed; treat as a
            # silent no-op rather than letting the route 500.
            return ""

    async def mark_uninstalled(
        self, listing_id: str, *, user_id: str = "",
    ) -> int:
        """Flag active install rows for ``listing_id`` as uninstalled.

        An empty ``user_id`` clears EVERY user's active record for the
        listing — used when an install-wide shared resource (e.g. a
        media-server container) is torn down, so no one's catalog still
        shows it installed. A specific ``user_id`` clears just that user's
        row. Returns the number of rows flagged. Idempotent: a listing
        with no active rows flags nothing and returns 0.
        """
        if not listing_id:
            return 0
        if user_id:
            cur = await self._conn.execute(
                "UPDATE marketplace_installs SET uninstalled_at = datetime('now') "
                "WHERE listing_id = ? AND user_id = ? AND uninstalled_at IS NULL",
                (listing_id, user_id),
            )
        else:
            cur = await self._conn.execute(
                "UPDATE marketplace_installs SET uninstalled_at = datetime('now') "
                "WHERE listing_id = ? AND uninstalled_at IS NULL",
                (listing_id,),
            )
        await self._conn.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    async def installed_listing_ids_for_user(
        self, user_id: str, listing_ids: list[str],
    ) -> set[str]:
        """Return the subset of ``listing_ids`` the user has installed.

        Reads from marketplace_installs (active rows only). Empty
        listing_ids returns empty set without hitting the DB.
        """
        if not user_id or not listing_ids:
            return set()
        placeholders = ",".join(["?"] * len(listing_ids))
        cursor = await self._conn.execute(
            f"SELECT listing_id FROM marketplace_installs "
            f"WHERE user_id = ? AND uninstalled_at IS NULL "
            f"AND listing_id IN ({placeholders})",
            (user_id, *listing_ids),
        )
        rows = await cursor.fetchall()
        return {r[0] for r in rows}

    async def install_wide_active_service_definitions(self) -> set[str]:
        """Return definition_ids of currently-enabled managed services.

        Provider service installs are install-wide (not per-user), so
        Discover should mark a provider listing as installed for ALL
        users when its underlying definition has an enabled
        managed_services row. This complements the per-user lookup
        in ``installed_listing_ids_for_user``.

        Falls back to empty set if the table doesn't exist or query
        fails (defensive — the catalog endpoint must keep rendering
        even if managed_services is unavailable).
        """
        try:
            cursor = await self._conn.execute(
                "SELECT DISTINCT definition_id FROM managed_services "
                "WHERE enabled = 1",
            )
            rows = await cursor.fetchall()
            return {r[0] for r in rows}
        except Exception:
            return set()


# ── helpers ──────────────────────────────────────────────────────────


def _row_to_listing(row: dict) -> MarketplaceListing:
    def _decode_json(raw: object, default):
        if isinstance(raw, dict) or isinstance(raw, list):
            return raw
        if isinstance(raw, str) and raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return default
        return default

    return MarketplaceListing(
        id=str(row.get("id", "")),
        publisher=str(row.get("publisher", "augmentum")),
        title=str(row.get("title", "")),
        kind=str(row.get("kind", "")),
        runtime_preferred=str(row.get("runtime_preferred", "")),
        runtime_alternates=tuple(
            str(a) for a in _decode_json(row.get("runtime_alternates"), [])
        ),
        tagline=str(row.get("tagline", "")),
        description=str(row.get("description", "")),
        thumbnail_url=str(row.get("thumbnail_url", "")),
        source_url=str(row.get("source_url", "")),
        embed_url=str(row.get("embed_url", "")),
        install_via=str(row.get("install_via", "")),
        install_payload=_decode_json(row.get("install_payload"), {}) or {},
        capabilities=_decode_json(row.get("capabilities"), {}) or {},
        metadata=_decode_json(row.get("metadata"), {}) or {},
        rating=row.get("rating"),
        install_count=int(row.get("install_count") or 0),
        signature=str(row.get("signature", "")),
        listed_at=str(row.get("listed_at", "")),
        delisted_at=row.get("delisted_at"),
        category=str(row.get("category") or ""),
        tags=tuple(str(t) for t in _decode_json(row.get("tags"), [])),
        featured=bool(row.get("featured") or 0),
    )
