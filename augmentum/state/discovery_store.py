"""Discovery Engine persistence — signals, history, and content library."""
from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_dict(cursor: aiosqlite.Cursor, row) -> dict:
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


# Titles matching these patterns are filtered from history listings.
_JUNK_TITLE_PATTERNS = [
    "Page Not Found", "404 Not Found", "403 Forbidden", "Access Denied",
    "Just a moment", "Attention Required", "Checking your browser",
    "Please enable", "Enable JavaScript", "404 Error", "500 Internal",
    "502 Bad Gateway", "503 Service", "Unauthorized", "Sign in",
]


class DiscoveryStore:
    """Read/write discovery engine state: signals, browse history, content library."""

    # Per-signal-type deduplication window in minutes. A new signal arriving
    # within the window for the same (user, url, type) merges into the
    # existing row instead of creating a new one — `browse_history.visit_count`
    # still increments either way, so we don't lose re-engagement counting.
    #
    # The windows reflect what counts as the "same event" for each kind of
    # signal:
    #   - page_visit / video_open: 30 min (page-reload + tab-revisit guard)
    #   - search_query: 5 min (typo retry / pagination)
    #   - video_watch / video_summary: 24 hr (daily re-watch is the same
    #     engagement; don't double-count for frecency purposes)
    #   - ai_action / discuss / note_save: 0 (deliberate actions — every
    #     one is meaningful, never dedup)
    # Unknown types fall back to 30 min for backwards-compat.
    _DEDUP_WINDOW_MIN: dict[str, int] = {
        "page_visit": 30,
        "video_open": 30,
        "video_seek": 30,
        "search_query": 5,
        "video_watch": 1440,
        "video_summary": 1440,
        "ai_action": 0,
        "discuss": 0,
        "note_save": 0,
    }
    _DEFAULT_DEDUP_WINDOW_MIN: int = 30

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    # -----------------------------------------------------------------------
    # Signals
    # -----------------------------------------------------------------------

    async def log_signal(
        self,
        *,
        signal_type: str,
        source_url: str,
        source_title: str,
        content_type: str,
        weight: float,
        metadata: dict,
        user_id: str = "",
    ) -> dict:
        """Log an interaction signal, deduplicating per signal-type window.

        Returns the signal dict with a ``deduplicated`` boolean flag.
        Signals are scoped per-user so one tenant's browsing can't pollute
        another tenant's discovery feed. Requires user_id. The dedup window
        varies by signal type — see ``_DEDUP_WINDOW_MIN`` for the policy.
        Window of 0 disables dedup entirely (every call creates a new row).
        """
        if not user_id:
            raise ValueError("interaction_signals insert requires user_id")
        window_min = self._DEDUP_WINDOW_MIN.get(
            signal_type, self._DEFAULT_DEDUP_WINDOW_MIN,
        )
        existing = None
        # window_min == 0 means "never dedup" — skip the lookup entirely
        # so deliberate actions (note_save, ai_action, discuss) always
        # create a fresh row even when fired back-to-back.
        if window_min > 0:
            window = (datetime.now(UTC) - timedelta(minutes=window_min)).isoformat()
            dedup_query = (
                """SELECT id, metadata FROM interaction_signals
                   WHERE source_url = ? AND signal_type = ? AND created_at >= ?"""
            )
            dedup_params: list = [source_url, signal_type, window]
            if user_id:
                dedup_query += " AND user_id = ?"
                dedup_params.append(user_id)
            dedup_query += " ORDER BY created_at DESC LIMIT 1"
            cursor = await self._conn.execute(dedup_query, dedup_params)
            existing = await cursor.fetchone()
        if existing:
            existing_id, existing_meta_raw = existing
            # Merge metadata if new metadata provided
            if metadata:
                try:
                    merged = {**json.loads(existing_meta_raw or "{}"), **metadata}
                except (json.JSONDecodeError, TypeError):
                    merged = metadata
                await self._conn.execute(
                    "UPDATE interaction_signals SET metadata = ? WHERE id = ?",
                    (json.dumps(merged), existing_id),
                )
                await self._conn.commit()

            cursor2 = await self._conn.execute(
                "SELECT * FROM interaction_signals WHERE id = ?", (existing_id,)
            )
            row = await cursor2.fetchone()
            result = _row_to_dict(cursor2, row)
            result["metadata"] = json.loads(result.get("metadata") or "{}")
            result["deduplicated"] = True
            return result

        sig_id = secrets.token_hex(8)
        domain = _extract_domain(source_url)
        now = _now_iso()
        await self._conn.execute(
            """INSERT INTO interaction_signals
               (id, signal_type, source_url, source_title, source_domain,
                content_type, metadata, weight, user_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sig_id,
                signal_type,
                source_url,
                source_title,
                domain,
                content_type,
                json.dumps(metadata),
                weight,
                user_id,
                now,
            ),
        )
        await self._conn.commit()

        return {
            "id": sig_id,
            "signal_type": signal_type,
            "source_url": source_url,
            "source_title": source_title,
            "source_domain": domain,
            "content_type": content_type,
            "metadata": metadata,
            "weight": weight,
            "cluster_id": None,
            "created_at": now,
            "deduplicated": False,
        }

    async def list_signals(
        self,
        *,
        signal_type: str | None = None,
        source_url: str | None = None,
        cluster_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        user_id: str = "",
    ) -> list[dict]:
        """List signals with optional type/URL/cluster filters (per-user)."""
        clauses: list[str] = []
        params: list = []
        if signal_type is not None:
            clauses.append("signal_type = ?")
            params.append(signal_type)
        if source_url is not None:
            clauses.append("source_url = ?")
            params.append(source_url)
        if cluster_id is not None:
            clauses.append("cluster_id = ?")
            params.append(cluster_id)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params += [limit, offset]
        cursor = await self._conn.execute(
            f"SELECT * FROM interaction_signals {where} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = _row_to_dict(cursor, row)
            d["metadata"] = json.loads(d.get("metadata") or "{}")
            results.append(d)
        return results

    # -----------------------------------------------------------------------
    # History
    # -----------------------------------------------------------------------

    async def upsert_history(
        self,
        *,
        url: str,
        title: str,
        domain: str,
        content_type: str,
        thumbnail: str,
        metadata: dict,
        user_id: str = "",
    ) -> dict:
        """Insert or update a browse history entry for ``user_id``.

        Increments visit_count and updates last_visited on revisit.
        browse_history.url has a UNIQUE index from the single-tenant era —
        treat this as a per-(url, user) upsert by looking up the caller's
        row first and skipping if another tenant holds the url.
        Requires user_id.

        On revisit, ``title`` IS overwritten with the latest value. We
        DON'T separately refresh title/thumbnail on rows the user hasn't
        revisited — see ``list_history`` for the rationale. If you need
        live metadata, hit the source URL directly; this row is the
        snapshot of what the user saw at visit time.
        """
        if not user_id:
            raise ValueError("browse_history insert requires user_id")
        now = _now_iso()
        hist_query = "SELECT id, visit_count, metadata FROM browse_history WHERE url = ?"
        hist_params: list = [url]
        if user_id:
            hist_query += " AND user_id = ?"
            hist_params.append(user_id)
        cursor = await self._conn.execute(hist_query, hist_params)
        existing = await cursor.fetchone()
        if existing:
            existing_id, visit_count, existing_meta_raw = existing
            new_count = visit_count + 1
            try:
                merged_meta = {**json.loads(existing_meta_raw or "{}"), **metadata}
            except (json.JSONDecodeError, TypeError):
                merged_meta = metadata
            await self._conn.execute(
                """UPDATE browse_history
                   SET title = ?, visit_count = ?, last_visited = ?, metadata = ?
                   WHERE id = ?""",
                (title, new_count, now, json.dumps(merged_meta), existing_id),
            )
            await self._conn.commit()
            cursor2 = await self._conn.execute(
                "SELECT * FROM browse_history WHERE id = ?", (existing_id,)
            )
            row = await cursor2.fetchone()
            d = _row_to_dict(cursor2, row)
            d["metadata"] = json.loads(d.get("metadata") or "{}")
            return d

        entry_id = secrets.token_hex(8)
        try:
            await self._conn.execute(
                """INSERT INTO browse_history
                   (id, url, title, domain, content_type, thumbnail, metadata,
                    visit_count, user_id, first_visited, last_visited)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (
                    entry_id,
                    url,
                    title,
                    domain,
                    content_type,
                    thumbnail,
                    json.dumps(metadata),
                    user_id,
                    now,
                    now,
                ),
            )
            await self._conn.commit()
        except aiosqlite.IntegrityError:
            # Legacy UNIQUE(url) — another tenant already has this URL.
            # Best-effort: skip instead of crashing the visit pipeline. The
            # next stage_c migration widens the unique key to (user_id, url).
            log.debug("history_url_collision", url=url[:120], user_id=user_id)
            return {
                "id": "",
                "url": url,
                "title": title,
                "domain": domain,
                "content_type": content_type,
                "thumbnail": thumbnail,
                "metadata": metadata,
                "cluster_id": None,
                "visit_count": 0,
                "first_visited": now,
                "last_visited": now,
                "collision": True,
            }
        return {
            "id": entry_id,
            "url": url,
            "title": title,
            "domain": domain,
            "content_type": content_type,
            "thumbnail": thumbnail,
            "metadata": metadata,
            "cluster_id": None,
            "visit_count": 1,
            "first_visited": now,
            "last_visited": now,
        }

    async def list_history(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        query: str = "",
        days: int = 0,
        user_id: str = "",
    ) -> list[dict]:
        """List browse history for ``user_id``, newest first.

        Optionally filter by text query (title/url/domain) and/or recency (days).

        Title/thumbnail are the snapshot from visit time (latest revisit
        wins via ``upsert_history``). Rows aren't background-refreshed —
        history is a record of what the user *saw*, not what the URL
        currently shows. Discovery's recommender doesn't read title/
        thumbnail from history (only URLs, for dedup), so staleness here
        is purely a History-panel concern and doesn't pollute For-You
        rankings or curator note quality.
        """
        clauses: list[str] = ["length(COALESCE(title, '')) > 3"]
        params: list = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        # Exclude junk titles (error pages, captchas, etc.)
        for pattern in _JUNK_TITLE_PATTERNS:
            clauses.append("title NOT LIKE ?")
            params.append(f"%{pattern}%")
        if query:
            like = f"%{query}%"
            clauses.append("(title LIKE ? OR url LIKE ? OR domain LIKE ?)")
            params += [like, like, like]
        if days > 0:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            clauses.append("last_visited >= ?")
            params.append(cutoff)
        where = "WHERE " + " AND ".join(clauses)
        params += [limit, offset]
        cursor = await self._conn.execute(
            f"SELECT * FROM browse_history {where} "
            f"ORDER BY last_visited DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = _row_to_dict(cursor, row)
            d["metadata"] = json.loads(d.get("metadata") or "{}")
            results.append(d)
        return results

    async def delete_history(
        self, history_id: str, *, user_id: str = "",
    ) -> bool:
        """Delete a history entry by ID, scoped to the owning user."""
        if not user_id:
            raise ValueError("browse_history delete requires user_id")
        cursor = await self._conn.execute(
            "DELETE FROM browse_history WHERE id = ? AND user_id = ?",
            (history_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def list_hidden_urls(
        self, *, days: int = 90, user_id: str = "",
    ) -> set[str]:
        """URLs the caller explicitly hid from Discovery.

        Folded into the recommender's seen-set so hidden items do not reappear.
        """
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        try:
            query = (
                """SELECT DISTINCT source_url FROM interaction_signals
                   WHERE signal_type = 'discovery_hide_url'
                     AND created_at >= ?"""
            )
            params: list = [cutoff]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            cursor = await self._conn.execute(query, params)
            rows = await cursor.fetchall()
            return {row[0] for row in rows if row[0]}
        except Exception:
            return set()

    async def check_visited_urls(
        self, urls: list[str], *, user_id: str = "",
    ) -> set[str]:
        """Return the subset of ``urls`` already present in browse_history.

        Lighter than :meth:`check_visited` — only the indexed ``url``
        column is read, no row payload or JSON parse. Use this when the
        caller only needs existence (e.g. dedup in the recommender's
        quality pipeline). The full-row variant remains for the
        `/api/discovery/check-visited` route that surfaces metadata to
        the browser.
        """
        if not urls:
            return set()
        placeholders = ",".join("?" * len(urls))
        query = f"SELECT url FROM browse_history WHERE url IN ({placeholders})"
        params: list = list(urls)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return {row[0] for row in rows if row[0]}

    async def check_visited(
        self, urls: list[str], *, user_id: str = "",
    ) -> dict[str, dict]:
        """Batch-check which URLs the authenticated user has visited."""
        if not urls:
            return {}
        placeholders = ",".join("?" * len(urls))
        query = f"SELECT * FROM browse_history WHERE url IN ({placeholders})"
        params: list = list(urls)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        result: dict[str, dict] = {}
        for row in rows:
            d = _row_to_dict(cursor, row)
            d["metadata"] = json.loads(d.get("metadata") or "{}")
            result[d["url"]] = d
        return result

    # -----------------------------------------------------------------------
    # Content Library
    # -----------------------------------------------------------------------

    async def store_chunk(
        self,
        *,
        source_url: str,
        source_title: str,
        source_type: str,
        content: str,
        embedding: bytes | None,
        cluster_id: str | None,
        user_id: str = "",
    ) -> dict:
        """Insert a content chunk into the library.

        Also inserts into content_library_vec if embedding is provided
        (silently skips if the vec0 extension is unavailable).
        """
        chunk_id = secrets.token_hex(8)
        now = _now_iso()
        await self._conn.execute(
            """INSERT INTO content_library
               (chunk_id, source_url, source_title, source_type, content,
                embedding, cluster_id, retrieved_count, created_at, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
            (chunk_id, source_url, source_title, source_type, content,
             embedding, cluster_id, now, user_id),
        )
        await self._conn.commit()

        if embedding is not None:
            try:
                await self._conn.execute(
                    "INSERT INTO content_library_vec (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, embedding),
                )
                await self._conn.commit()
            except Exception as exc:
                log.debug(
                    "discovery_chunk_vec_insert_skipped",
                    chunk_id=chunk_id,
                    error=str(exc),
                )

        return {
            "chunk_id": chunk_id,
            "source_url": source_url,
            "source_title": source_title,
            "source_type": source_type,
            "content": content,
            "embedding": embedding,
            "cluster_id": cluster_id,
            "retrieved_count": 0,
            "created_at": now,
        }

    async def get_chunk(self, chunk_id: str, *, user_id: str = "") -> dict | None:
        """Fetch a single chunk by ID."""
        if user_id:
            cursor = await self._conn.execute(
                "SELECT * FROM content_library WHERE chunk_id = ? AND user_id = ?",
                (chunk_id, user_id),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM content_library WHERE chunk_id = ?", (chunk_id,)
            )
        row = await cursor.fetchone()
        if not row:
            return None
        return _row_to_dict(cursor, row)

    async def get_chunks_by_source(self, source_url: str, *, user_id: str = "") -> list[dict]:
        """Return all chunks for a given source URL."""
        if user_id:
            cursor = await self._conn.execute(
                "SELECT * FROM content_library WHERE source_url = ? AND user_id = ? "
                "ORDER BY created_at ASC",
                (source_url, user_id),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM content_library WHERE source_url = ? ORDER BY created_at ASC",
                (source_url,),
            )
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, row) for row in rows]

    async def search_library(
        self,
        query_embedding: bytes,
        *,
        limit: int = 5,
        min_score: float = 0.65,
        user_id: str = "",
    ) -> list[dict]:
        """Search content library by embedding similarity.

        Uses the vec0 KNN index (``MATCH ? AND k = ?``) — NOT a
        ``vec_distance_cosine`` full scan. The earlier shape forced SQLite
        to compute cosine distance for every row + walk overflow pages on
        the JOIN, which on the shared aiosqlite connection blocked every
        other DB-touching route until it finished. Over-fetch then prune
        by ``min_score`` so the floor still holds after KNN.

        Returns an empty list if the vec0 extension is unavailable.
        """
        try:
            if user_id:
                cursor = await self._conn.execute(
                    """SELECT cl.*, (1 - v.distance) AS similarity
                       FROM content_library_vec v
                       JOIN content_library cl ON cl.chunk_id = v.chunk_id
                       WHERE v.embedding MATCH ? AND k = ? AND cl.user_id = ?
                       ORDER BY v.distance""",
                    (query_embedding, max(limit * 2, 10), user_id),
                )
            else:
                cursor = await self._conn.execute(
                    """SELECT cl.*, (1 - v.distance) AS similarity
                       FROM content_library_vec v
                       JOIN content_library cl ON cl.chunk_id = v.chunk_id
                       WHERE v.embedding MATCH ? AND k = ?
                       ORDER BY v.distance""",
                    (query_embedding, max(limit * 2, 10)),
                )
            rows = await cursor.fetchall()
            results = [_row_to_dict(cursor, row) for row in rows]
            return [r for r in results if r.get("similarity", 0.0) >= min_score][:limit]
        except Exception:
            log.warning("search_library_vec_unavailable", exc_info=True)
            return []

    async def increment_retrieved(self, chunk_id: str, *, user_id: str = "") -> None:
        """Bump the retrieved_count for a chunk."""
        if user_id:
            await self._conn.execute(
                "UPDATE content_library SET retrieved_count = retrieved_count + 1 "
                "WHERE chunk_id = ? AND user_id = ?",
                (chunk_id, user_id),
            )
        else:
            await self._conn.execute(
                "UPDATE content_library SET retrieved_count = retrieved_count + 1 "
                "WHERE chunk_id = ?",
                (chunk_id,),
            )
        await self._conn.commit()

    async def has_source(self, source_url: str, *, user_id: str = "") -> bool:
        """Return True if any chunk exists for the given source URL."""
        if user_id:
            cursor = await self._conn.execute(
                "SELECT 1 FROM content_library WHERE source_url = ? AND user_id = ? LIMIT 1",
                (source_url, user_id),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT 1 FROM content_library WHERE source_url = ? LIMIT 1",
                (source_url,),
            )
        return await cursor.fetchone() is not None

    # -----------------------------------------------------------------------
    # Clusters
    # -----------------------------------------------------------------------

    async def upsert_cluster(
        self, cluster: dict, *, user_id: str = "",
    ) -> None:
        """INSERT OR REPLACE an interest cluster row.

        Clusters are per-user — one tenant's topic model doesn't feed into
        another tenant's feed. Requires user_id.
        """
        owner = user_id or cluster.get("user_id", "")
        if not owner:
            raise ValueError("interest_clusters insert requires user_id")
        await self._conn.execute(
            """INSERT OR REPLACE INTO interest_clusters
               (cluster_id, name, centroid_embedding, frecency_short, frecency_long,
                depth_level, signal_count, narration, knowledge_gaps, adjacent_topics,
                dampened, user_id, kind, entity_ref, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cluster["cluster_id"],
                cluster.get("name", ""),
                cluster.get("centroid_embedding"),
                cluster.get("frecency_short", 0.0),
                cluster.get("frecency_long", 0.0),
                cluster.get("depth_level", 1),
                cluster.get("signal_count", 0),
                cluster.get("narration"),
                cluster.get("knowledge_gaps"),
                cluster.get("adjacent_topics"),
                cluster.get("dampened", 0),
                owner,
                # Entity-vs-topic kind (migration 264). REPLACE rewrites
                # the whole row, so omitting these here would silently
                # reset an entity cluster to 'topic' on every frecency
                # round-trip.
                cluster.get("kind", "topic") or "topic",
                cluster.get("entity_ref", "") or "",
                cluster.get("created_at", _now_iso()),
                cluster.get("updated_at", _now_iso()),
            ),
        )
        await self._conn.commit()

    async def list_clusters(
        self, *, include_dampened: bool = False, user_id: str = "",
    ) -> list[dict]:
        """List the caller's clusters ordered by combined frecency DESC."""
        clauses: list[str] = []
        params: list = []
        if not include_dampened:
            clauses.append("dampened = 0")
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cursor = await self._conn.execute(
            f"""SELECT * FROM interest_clusters {where}
                ORDER BY (frecency_short * 0.6 + frecency_long * 0.4) DESC""",
            params,
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, row) for row in rows]

    async def find_nearest_cluster(
        self,
        embedding: bytes,
        threshold: float = 0.75,
    ) -> dict | None:
        """Vec search for the nearest cluster centroid.

        Returns {cluster_id, similarity} if above threshold, else None.
        Silently returns None if vec0 is unavailable.

        Uses the vec0 KNN index (``MATCH ? AND k = 1``) — see
        ``search_library`` for the rationale on why the alias-filtered
        cosine variant is unsafe on the shared connection.
        """
        try:
            cursor = await self._conn.execute(
                """SELECT cluster_id, (1 - distance) AS similarity
                   FROM interest_clusters_vec
                   WHERE centroid_embedding MATCH ? AND k = 1
                   ORDER BY distance""",
                (embedding,),
            )
            row = await cursor.fetchone()
            if row is None or row[1] < threshold:
                return None
            return {"cluster_id": row[0], "similarity": row[1]}
        except Exception:
            log.warning("find_nearest_cluster_vec_unavailable", exc_info=True)
            return None

    async def upsert_cluster_vec(self, cluster_id: str, embedding: bytes) -> None:
        """Delete + insert into interest_clusters_vec (vec0 virtual table)."""
        try:
            await self._conn.execute(
                "DELETE FROM interest_clusters_vec WHERE cluster_id = ?",
                (cluster_id,),
            )
            await self._conn.execute(
                "INSERT INTO interest_clusters_vec (cluster_id, centroid_embedding) VALUES (?, ?)",
                (cluster_id, embedding),
            )
            await self._conn.commit()
        except Exception:
            log.warning("upsert_cluster_vec_unavailable", cluster_id=cluster_id)

    async def update_signal_cluster(self, signal_id: str, cluster_id: str) -> None:
        """Assign a signal to a cluster."""
        await self._conn.execute(
            "UPDATE interaction_signals SET cluster_id = ? WHERE id = ?",
            (cluster_id, signal_id),
        )
        await self._conn.commit()

    async def dampen_cluster(
        self, cluster_id: str, *, user_id: str = "",
    ) -> None:
        """Mark a cluster as dampened (de-prioritized), owner only."""
        if not user_id:
            raise ValueError("interest_clusters dampen requires user_id")
        await self._conn.execute(
            "UPDATE interest_clusters SET dampened = 1, updated_at = ? "
            "WHERE cluster_id = ? AND user_id = ?",
            (_now_iso(), cluster_id, user_id),
        )
        await self._conn.commit()

    # -----------------------------------------------------------------------
    # Content pruning
    # -----------------------------------------------------------------------

    async def prune_old_content(self, retention_days: int) -> int:
        """Replace content text for old, unretrieved chunks with a placeholder.

        Chunks with retrieved_count > 0 are never pruned.
        Returns the number of chunks pruned.
        """
        cutoff = (
            datetime.now(UTC) - timedelta(days=retention_days)
        ).isoformat()
        cursor = await self._conn.execute(
            """UPDATE content_library
               SET content = '[pruned: content older than retention window]'
               WHERE created_at < ? AND retrieved_count = 0
                 AND content != '[pruned: content older than retention window]'""",
            (cutoff,),
        )
        await self._conn.commit()
        return cursor.rowcount
