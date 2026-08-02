"""Dream Journal — CRUD operations for dream entries stored in SQLite."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta

from contextlib import asynccontextmanager

import aiosqlite

from augmentum.config import settings
from augmentum.dream.models import DreamEntry, DreamEntryType


@asynccontextmanager
async def _reuse_conn(conn: aiosqlite.Connection):
    """Async context manager that yields an existing connection without closing it."""
    yield conn
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# sqlite-vec is loaded lazily — the module import succeeds even without
# the sidecar shared library being available, but loadable_path() is the
# trigger that requires the .so/.dll. Treat both as soft failures so the
# journal works in environments without the extension (CI without the
# wheel, distros that didn't bundle it, etc.).
_VEC_AVAILABLE = False
try:
    import sqlite_vec
    _VEC_AVAILABLE = True
except ImportError:
    sqlite_vec = None  # type: ignore[assignment]


class DreamJournal:
    """Persistent store for dream entries, backed by SQLite.

    Maintains a persistent connection (``_db``) opened during
    ``initialize()``.  The PortraitManager and DreamEngine access
    the connection through this attribute for direct queries on
    the portrait / cycle / memory-log tables that live alongside
    dream entries in the same database.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        # Whether the sqlite-vec extension is loaded on this connection.
        # The journal opens its OWN aiosqlite connection separate from
        # SQLiteBackend's, so vec0 isn't automatically available — we
        # have to load it ourselves in initialize(). When false, the
        # semantic-recall path is unavailable and callers fall back to
        # chronological recall. Set during initialize().
        self._vec_enabled = False
        # Optional app_state reference — set by lifecycle wiring after
        # construction so on-write consolidation can resolve a backend
        # via the provider registry. Tests construct the journal without
        # app_state and on-write consolidation gracefully no-ops.
        self._app_state: object | None = None

    def attach_app_state(self, app_state: object) -> None:
        """Wire app_state in post-construction. Called from lifecycle.

        Lets the journal resolve the provider registry for on-write
        consolidation without making app_state a constructor dependency
        (which would force every test to mock it). Safe to call multiple
        times — last write wins.
        """
        self._app_state = app_state

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open a persistent connection and verify tables exist."""
        # Apply the canonical augmentum.db pragma set (WAL +
        # busy_timeout=30s + synchronous=NORMAL etc.). Pre-fix this
        # connection set only journal_mode=WAL with no busy_timeout,
        # so during contention it would raise "database is locked"
        # instantly instead of waiting through a brief writer-stall.
        from augmentum.state.backends.sqlite import (
            apply_augmentum_pragmas,
            install_safe_rollback,
        )

        self._db = await aiosqlite.connect(self._db_path)
        await apply_augmentum_pragmas(self._db)
        # Structural safety net: any DML that raises auto-rollbacks so a
        # transient ``database is locked`` can't leave the connection in
        # in_transaction=True and pin a WAL snapshot on the next SELECT.
        # This is THE mode that caused the 2026-05-22 8-hour WAL pin —
        # a failed _persist_cycle INSERT here without rollback poisoned
        # the connection for the rest of the day.
        install_safe_rollback(self._db)
        # Load sqlite-vec on this connection so dream_entries_vec is
        # queryable. Best-effort — without it the journal still works,
        # just without semantic recall (chronological fallback applies).
        # Vec is only "enabled" if BOTH the extension loads AND the
        # dream_entries_vec virtual table actually exists. Test fixtures
        # frequently load the extension but skip the vec table, so the
        # extension-loaded check alone would falsely advertise capability
        # and trigger a 130MB FastEmbed model download in unit tests.
        extension_loaded = False
        if _VEC_AVAILABLE:
            try:
                await self._db.enable_load_extension(True)
                await self._db.load_extension(sqlite_vec.loadable_path())
                await self._db.enable_load_extension(False)
                extension_loaded = True
                log.info("dream_journal.vec_loaded")
            except Exception:
                log.debug("dream_journal.vec_load_failed", exc_info=True)
        if extension_loaded:
            try:
                cursor = await self._db.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type='table' AND name='dream_entries_vec'"
                )
                self._vec_enabled = bool(await cursor.fetchone())
            except Exception:
                log.debug("dream_journal.vec_table_check_failed", exc_info=True)
        # Verify all required dream tables exist
        required = ["dream_entries", "dream_portraits", "dream_cycles", "dream_memory_log"]
        cursor = await self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ({})".format(
                ",".join("?" for _ in required)
            ),
            required,
        )
        found = {row[0] for row in await cursor.fetchall()}
        missing = set(required) - found
        if missing:
            raise RuntimeError(
                f"Dream tables missing (migration 058 may not have run): {sorted(missing)}"
            )
        log.info("dream_journal.initialized", db_path=self._db_path, vec_enabled=self._vec_enabled)

    async def close(self) -> None:
        """Close the persistent connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def store_entry(
        self,
        persona_id: str,
        content: str,
        entry_type: DreamEntryType | str,
        source_memories: list[str],
        source_sessions: list[str],
        context_window: dict,
        dream_cycle_id: str,
        embedding: bytes | None = None,
        *,
        user_id: str = "",
    ) -> str:
        """Insert a new dream entry and return its ID.

        If ``embedding`` is not supplied AND content is non-empty, computes
        a document-typed embedding via ``EmbeddingService.embed_one`` and
        also inserts a row into ``dream_entries_vec`` so the entry is
        immediately searchable via ``find_similar_entries``. Embedding
        computation runs in a thread to keep the event loop free; failures
        are best-effort and do NOT block the row insert — losing semantic
        searchability is preferable to losing the dream entry itself.
        """
        if not user_id:
            raise ValueError("dream_entries insert requires user_id")
        entry_id = uuid.uuid4().hex[:16]
        if isinstance(entry_type, DreamEntryType):
            entry_type_val = entry_type.value
        else:
            entry_type_val = str(entry_type)

        # Compute embedding lazily — only if not provided and we'll
        # actually be able to use it (vec extension loaded). Avoid the
        # ~2ms FastEmbed call when vec is unavailable since the result
        # would be unused.
        embedding_floats: list[float] | None = None
        if embedding is None and content and self._vec_enabled:
            try:
                from augmentum.memory.embeddings import EmbeddingService
                embedding_floats = await asyncio.to_thread(
                    EmbeddingService.embed_one, content,
                )
                embedding = EmbeddingService.to_blob(embedding_floats)
            except Exception:
                log.warning("dream_journal.embed_failed", entry_id=entry_id, exc_info=True)
                embedding = None
                embedding_floats = None

        # On-write consolidation — if a near-duplicate already exists in
        # the configured window, merge into it instead of inserting a new
        # row. Mirrors memory's try_consolidate. Cheap when no candidates
        # exist (one vec query) and absorbs the LLM cost only on actual
        # matches. Best-effort: any failure falls through to normal insert.
        if (
            settings.dream_compaction_enabled
            and embedding is not None
            and content
            and self._vec_enabled
        ):
            try:
                merged_id = await self._maybe_consolidate_on_write(
                    new_content=content,
                    new_embedding_blob=embedding,
                    persona_id=persona_id,
                    user_id=user_id,
                )
                if merged_id is not None:
                    log.info(
                        "dream_journal.on_write_consolidated",
                        target_id=merged_id, persona_id=persona_id,
                    )
                    return merged_id
            except Exception:
                log.debug("dream_journal.on_write_consolidate_failed", exc_info=True)

        async with self._connect() as db:
            await db.execute(
                """INSERT INTO dream_entries
                   (id, persona_id, content, entry_type, source_memories,
                    source_sessions, context_window, embedding, dream_cycle_id,
                    user_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry_id, persona_id, content, entry_type_val,
                    json.dumps(source_memories), json.dumps(source_sessions),
                    json.dumps(context_window), embedding, dream_cycle_id,
                    user_id,
                ),
            )

            # FTS index (best-effort — table may not exist in all environments)
            if content:
                try:
                    await db.execute(
                        "INSERT INTO dream_entries_fts(rowid, content) VALUES ((SELECT rowid FROM dream_entries WHERE id = ?), ?)",
                        (entry_id, content),
                    )
                except Exception as exc:
                    # FTS table optional in test environments; debug-log
                    # so an unexpected FTS failure (vs absent) is findable.
                    log.debug("dream_journal_fts_insert_skipped", entry_id=entry_id, error=str(exc))

            # Vec index for semantic recall. Only insert when we have
            # both an embedding AND vec is enabled; when one is missing
            # the entry is still stored, just skipped from vec search
            # (chronological fallback handles it).
            if embedding is not None and self._vec_enabled:
                try:
                    await db.execute(
                        "INSERT INTO dream_entries_vec(id, embedding) VALUES (?, ?)",
                        (entry_id, embedding),
                    )
                except Exception:
                    log.warning(
                        "dream_journal.vec_insert_failed",
                        entry_id=entry_id, exc_info=True,
                    )

            await db.commit()

        log.debug("dream_journal.stored", entry_id=entry_id, persona_id=persona_id)
        return entry_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_entry(self, entry_id: str, *, user_id: str = "") -> DreamEntry | None:
        """Fetch a single entry by ID, or None if not found."""
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            query = ("SELECT id, persona_id, content, entry_type, source_memories,"
                     " source_sessions, context_window, embedding, weight,"
                     " pinned, dream_cycle_id, created_at, expires_at"
                     " FROM dream_entries WHERE id = ?")
            params: list = [entry_id]
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None
        return self._row_to_entry(row)

    async def list_entries(
        self,
        persona_id: str,
        limit: int = 50,
        offset: int = 0,
        entry_type: DreamEntryType | str | None = None,
        *,
        user_id: str = "",
    ) -> tuple[list[DreamEntry], int]:
        """Return paginated entries for a persona, plus the total count."""
        type_filter = ""
        params_filter: list = [persona_id]

        if entry_type is not None:
            type_val = entry_type.value if isinstance(entry_type, DreamEntryType) else str(entry_type)
            type_filter = " AND entry_type = ?"
            params_filter.append(type_val)

        if user_id:
            type_filter += " AND user_id = ?"
            params_filter.append(user_id)

        base_where = "WHERE persona_id = ?" + type_filter

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row

            # Total count
            async with db.execute(
                f"SELECT COUNT(*) FROM dream_entries {base_where}",
                params_filter,
            ) as cursor:
                row = await cursor.fetchone()
                total = row[0] if row else 0

            # Paginated rows
            async with db.execute(
                f"""
                SELECT id, persona_id, content, entry_type, source_memories,
                       source_sessions, context_window, embedding, weight,
                       pinned, dream_cycle_id, created_at, expires_at
                FROM dream_entries
                {base_where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                [*params_filter, limit, offset],
            ) as cursor:
                rows = await cursor.fetchall()

        entries = [self._row_to_entry(r) for r in rows]
        return entries, total

    # ------------------------------------------------------------------
    # Semantic recall
    # ------------------------------------------------------------------

    async def find_similar_entries(
        self,
        query: str,
        persona_id: str,
        *,
        user_id: str = "",
        limit: int = 10,
        min_similarity: float = 0.4,
        exclude_expired: bool = True,
    ) -> list[DreamEntry]:
        """Return entries semantically closest to ``query``, scoped to user.

        Uses the sqlite-vec ``dream_entries_vec`` index with a Nomic
        query-typed embedding. Returns ``[]`` when vec is unavailable,
        when the query is empty, when no entries have embeddings yet,
        when embedding the query fails, or when no entries clear the
        ``min_similarity`` gate — caller is expected to fall back to
        chronological recall in those cases.

        ``min_similarity`` is in similarity space (0 = unrelated, 1 =
        identical), matching ``MemoryStore._vector_search``'s convention.
        Default 0.4 is slightly more permissive than memory's 0.5 because
        dream entries are longer and more abstract than memory facts —
        even relevant entries often score lower against question-shaped
        queries. Caller can override per-user via the
        ``ui.dreamRecallMinSimilarity`` setting.

        Returned list is ordered by similarity (closest first), capped
        at ``limit``. user_id and expires_at filters happen post-vec
        because vec0 doesn't support arbitrary WHERE on adjacent
        columns; we over-fetch from vec then filter to compensate.
        """
        if not self._vec_enabled or not query or not query.strip():
            return []
        try:
            from augmentum.memory.embeddings import EmbeddingService
            query_vec = await asyncio.to_thread(EmbeddingService.embed_query, query)
            blob = EmbeddingService.to_blob(query_vec)
        except Exception:
            log.debug("dream_journal.query_embed_failed", exc_info=True)
            return []

        # Fetch 3× requested so user/persona/expires filters still leave
        # ``limit`` usable results in the common case. The vec index has
        # no per-user partitioning so cross-tenant rows show up here and
        # have to be filtered out below.
        over_fetch = max(limit * 3, limit)

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            try:
                async with db.execute(
                    "SELECT id, distance FROM dream_entries_vec "
                    "WHERE embedding MATCH ? AND k = ? "
                    "ORDER BY distance",
                    (blob, over_fetch),
                ) as cursor:
                    vec_rows = await cursor.fetchall()
            except Exception:
                log.debug("dream_journal.vec_query_failed", exc_info=True)
                return []

            if not vec_rows:
                return []

            # Apply similarity threshold here, before fetching full rows.
            # Same convention as MemoryStore._vector_search:
            #   similarity = 1 - distance, keep if similarity >= threshold,
            #   equivalently distance <= 1 - threshold.
            # Without this gate the recall path is just "top-N regardless of
            # actual relevance" — a casual chat about anything pulls in
            # whatever was most recently embedded, dragging the model's
            # attention toward unrelated topics.
            max_distance = 1.0 - min_similarity
            vec_rows = [r for r in vec_rows if r[1] <= max_distance]
            if not vec_rows:
                log.debug(
                    "dream_journal.find_similar_no_match",
                    persona_id=persona_id, min_similarity=min_similarity,
                )
                return []

            ids = [r[0] for r in vec_rows]
            distance_map = {r[0]: r[1] for r in vec_rows}
            placeholders = ",".join("?" * len(ids))

            uid_filter = ""
            extra_params: list = []
            if user_id:
                uid_filter += " AND user_id = ?"
                extra_params.append(user_id)
            if exclude_expired:
                uid_filter += " AND expires_at IS NULL"

            async with db.execute(
                f"""SELECT id, persona_id, content, entry_type, source_memories,
                          source_sessions, context_window, embedding, weight,
                          pinned, dream_cycle_id, created_at, expires_at
                    FROM dream_entries
                    WHERE id IN ({placeholders}) AND persona_id = ?{uid_filter}""",
                [*ids, persona_id, *extra_params],
            ) as cursor:
                entry_rows = await cursor.fetchall()

        by_id = {r["id"]: self._row_to_entry(r) for r in entry_rows}

        # Reassemble in vec-distance order and cap to limit
        ordered: list[DreamEntry] = []
        for vid in ids:
            entry = by_id.get(vid)
            if entry is None:
                continue
            ordered.append(entry)
            if len(ordered) >= limit:
                break

        log.debug(
            "dream_journal.find_similar",
            persona_id=persona_id, returned=len(ordered),
            best_distance=distance_map.get(ids[0]) if ids else None,
        )
        return ordered

    async def find_consolidation_candidates(
        self,
        embedding_blob: bytes,
        persona_id: str,
        *,
        user_id: str,
        sim_low: float,
        sim_high: float,
        exclude_id: str | None = None,
        limit: int = 5,
    ) -> list[tuple[DreamEntry, float]]:
        """Find existing entries whose similarity to ``embedding_blob`` is
        within ``[sim_low, sim_high]`` — the on-write consolidation window.

        Used by ``store_entry`` BEFORE inserting a new entry to check
        whether the new content is close enough to an existing entry that
        they should be merged rather than both stored. Mirrors memory's
        ``try_consolidate`` search but operates on dream entries.

        Returns ``[(entry, similarity), ...]`` ordered by similarity
        (highest first), capped at ``limit``. ``exclude_id`` lets callers
        skip a specific id (e.g., the entry being updated). Empty list
        when vec is unavailable, no matches in range, or query fails —
        callers proceed with normal insert.
        """
        if not self._vec_enabled:
            return []
        # vec0 returns top-K by distance — over-fetch so the post-filter
        # to the [low, high] band still has enough candidates after user
        # / persona scoping. The band is narrow so 4× limit is plenty.
        over_fetch = max(limit * 4, 10)
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            try:
                async with db.execute(
                    "SELECT id, distance FROM dream_entries_vec "
                    "WHERE embedding MATCH ? AND k = ? "
                    "ORDER BY distance",
                    (embedding_blob, over_fetch),
                ) as cursor:
                    vec_rows = await cursor.fetchall()
            except Exception:
                log.debug("dream_journal.consolidation_vec_failed", exc_info=True)
                return []

            if not vec_rows:
                return []

            # Filter by similarity band BEFORE the row fetch — saves an
            # I/O on the inevitable misses.
            in_band: list[tuple[str, float]] = []
            for r in vec_rows:
                sim = 1.0 - float(r[1])
                if sim_low <= sim <= sim_high:
                    if exclude_id is None or r[0] != exclude_id:
                        in_band.append((r[0], sim))
            if not in_band:
                return []

            ids = [t[0] for t in in_band]
            sim_by_id = {t[0]: t[1] for t in in_band}
            placeholders = ",".join("?" * len(ids))
            async with db.execute(
                f"""SELECT id, persona_id, content, entry_type, source_memories,
                          source_sessions, context_window, embedding, weight,
                          pinned, dream_cycle_id, created_at, expires_at
                    FROM dream_entries
                    WHERE id IN ({placeholders}) AND persona_id = ? AND user_id = ?
                          AND expires_at IS NULL""",
                [*ids, persona_id, user_id],
            ) as cursor:
                rows = await cursor.fetchall()

        # Reassemble in similarity-descending order
        by_id = {r["id"]: self._row_to_entry(r) for r in rows}
        out: list[tuple[DreamEntry, float]] = []
        for vid, sim in sorted(in_band, key=lambda t: -t[1]):
            entry = by_id.get(vid)
            if entry is not None:
                out.append((entry, sim))
            if len(out) >= limit:
                break
        return out

    async def _maybe_consolidate_on_write(
        self,
        *,
        new_content: str,
        new_embedding_blob: bytes,
        persona_id: str,
        user_id: str,
    ) -> str | None:
        """Try on-write consolidation. Returns target id if merged, else None.

        Wires together :meth:`find_consolidation_candidates`, the
        :func:`augmentum.dream.consolidator.try_consolidate_dream` LLM
        merge, and a direct UPDATE of the target entry's content +
        embedding. If anything fails (no app_state, no registry, LLM
        error, etc.) returns None so the caller falls through to the
        normal INSERT path.

        Per-user scoping: ``find_consolidation_candidates`` filters to
        ``user_id`` so candidates are guaranteed to belong to the same
        user. The UPDATE also includes ``user_id`` in the WHERE so a
        stale cross-user id slipping through somehow still wouldn't
        write across tenants.
        """
        candidates = await self.find_consolidation_candidates(
            embedding_blob=new_embedding_blob,
            persona_id=persona_id,
            user_id=user_id,
            sim_low=settings.dream_consolidation_low,
            sim_high=settings.dream_consolidation_high,
            limit=3,
        )
        if not candidates:
            return None

        backend, model = await self._resolve_consolidation_backend()
        if backend is None:
            return None

        from augmentum.dream.consolidator import try_consolidate_dream
        result = await try_consolidate_dream(
            new_content=new_content,
            candidates=candidates,
            backend=backend,
            model=model,
            sim_low=settings.dream_consolidation_low,
            sim_high=settings.dream_consolidation_high,
        )
        if result is None:
            return None

        merged_text, _importance, target_id = result
        ok = await self.merge_entries(
            keep_id=target_id,
            # No drop here — there's no second row yet, we're consolidating
            # BEFORE inserting. Reuse merge_entries' UPDATE-keep path by
            # passing a sentinel drop_id that won't match anything; the
            # soft-delete UPDATE no-ops on rowcount=0.
            drop_id="__nodrop_on_write__",
            merged_content=merged_text,
            user_id=user_id,
        )
        if not ok:
            return None
        return target_id

    async def _resolve_consolidation_backend(self) -> tuple[object, str | None]:
        """Resolve LLM backend for on-write consolidation via app_state.

        Mirrors ``DreamCompactor._resolve_backend`` — same role chain
        (``utility``), same override knob (``dream_compaction_model``).
        Returns ``(None, None)`` when the registry isn't reachable so
        callers gracefully fall through.
        """
        app_state = getattr(self, "_app_state", None)
        if app_state is None:
            return (None, None)
        registry = getattr(app_state, "provider_registry", None)
        if registry is None:
            return (None, None)
        try:
            backend, model = await registry.resolve_model_for_role(
                "utility",
                override=settings.dream_compaction_model,
                settings=settings,
            )
            return (backend, model)
        except Exception:
            log.debug("dream_journal.consolidation_backend_resolve_failed", exc_info=True)
            return (None, None)

    async def merge_entries(
        self,
        keep_id: str,
        drop_id: str,
        merged_content: str,
        *,
        user_id: str,
    ) -> bool:
        """Merge ``drop_id`` into ``keep_id``: replace keep's content with
        ``merged_content``, refresh keep's embedding, soft-delete drop.

        Soft-delete uses ``expires_at`` (matches ``compact_journal``'s
        existing soft-delete semantics) so the drop entry is excluded from
        all read paths but remains in the table for audit / rollback.
        Both rows are filtered by ``user_id`` so cross-user merges are
        physically impossible at the SQL level. Returns True on success,
        False if keep_id wasn't found for this user.
        """
        if not user_id:
            raise ValueError("dream_entries merge requires user_id")
        # Compute a fresh embedding for merged_content. Skip silently if
        # vec or the embedding service isn't available — keep's embedding
        # would just be slightly stale, not broken.
        new_blob: bytes | None = None
        if self._vec_enabled and merged_content:
            try:
                from augmentum.memory.embeddings import EmbeddingService
                vec = await asyncio.to_thread(EmbeddingService.embed_one, merged_content)
                new_blob = EmbeddingService.to_blob(vec)
            except Exception:
                log.debug("dream_journal.merge_embed_failed", exc_info=True)

        expires_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        async with self._connect() as db:
            # Update keep's content (and embedding column if we computed one)
            if new_blob is not None:
                cursor = await db.execute(
                    "UPDATE dream_entries SET content = ?, embedding = ? "
                    "WHERE id = ? AND user_id = ?",
                    (merged_content, new_blob, keep_id, user_id),
                )
            else:
                cursor = await db.execute(
                    "UPDATE dream_entries SET content = ? "
                    "WHERE id = ? AND user_id = ?",
                    (merged_content, keep_id, user_id),
                )
            if cursor.rowcount == 0:
                # keep_id not found for this user — refuse merge
                return False

            # Refresh keep's vec row (delete old + insert new), best-effort
            if new_blob is not None and self._vec_enabled:
                try:
                    await db.execute(
                        "DELETE FROM dream_entries_vec WHERE id = ?",
                        (keep_id,),
                    )
                    await db.execute(
                        "INSERT INTO dream_entries_vec(id, embedding) VALUES (?, ?)",
                        (keep_id, new_blob),
                    )
                except Exception:
                    log.debug("dream_journal.merge_vec_refresh_failed", exc_info=True)

            # Soft-delete drop (mark expired, scoped to same user)
            await db.execute(
                "UPDATE dream_entries SET expires_at = ? "
                "WHERE id = ? AND user_id = ? AND expires_at IS NULL",
                (expires_at, drop_id, user_id),
            )
            # Drop row's vec entry stays — find_similar_entries already
            # filters expires_at IS NULL via the dream_entries join.
            # Removing it would invalidate audit trail.

            await db.commit()
        return True

    async def backfill_embeddings(
        self,
        persona_id: str,
        *,
        user_id: str = "",
        batch_size: int = 32,
        max_entries: int = 500,
    ) -> int:
        """Compute embeddings for entries that don't have one yet.

        Targets the existing dream-entry rows whose ``embedding`` column
        is NULL (typically pre-existing data from before semantic recall
        was wired). Batches via ``EmbeddingService.embed`` for efficiency,
        writes both the column and the vec index. Returns the number of
        rows successfully backfilled.

        Bounded by ``max_entries`` per call so a long-time user with
        thousands of entries doesn't block startup; the next call picks
        up where this one left off.
        """
        if not self._vec_enabled:
            return 0
        try:
            from augmentum.memory.embeddings import EmbeddingService
        except Exception:
            log.debug("dream_journal.backfill_embed_unavailable", exc_info=True)
            return 0

        params: list = [persona_id]
        uid_filter = ""
        if user_id:
            uid_filter = " AND user_id = ?"
            params.append(user_id)
        params.append(max_entries)

        async with self._connect() as db:
            async with db.execute(
                f"""SELECT id, content FROM dream_entries
                    WHERE persona_id = ? AND embedding IS NULL
                          AND content IS NOT NULL AND content != ''{uid_filter}
                    ORDER BY created_at ASC
                    LIMIT ?""",
                params,
            ) as cursor:
                pending = await cursor.fetchall()

            if not pending:
                return 0

            backfilled = 0
            for batch_start in range(0, len(pending), batch_size):
                batch = pending[batch_start:batch_start + batch_size]
                texts = [row[1] for row in batch]
                try:
                    vectors = await asyncio.to_thread(EmbeddingService.embed, texts)
                except Exception:
                    log.warning("dream_journal.backfill_batch_embed_failed", exc_info=True)
                    continue

                for (entry_id, _content), vec in zip(batch, vectors, strict=False):
                    blob = EmbeddingService.to_blob(vec)
                    try:
                        await db.execute(
                            "UPDATE dream_entries SET embedding = ? WHERE id = ?",
                            (blob, entry_id),
                        )
                        # Vec inserts are unique-id; if a row somehow
                        # already exists (shouldn't, but defensive),
                        # delete then re-insert.
                        await db.execute(
                            "DELETE FROM dream_entries_vec WHERE id = ?",
                            (entry_id,),
                        )
                        await db.execute(
                            "INSERT INTO dream_entries_vec(id, embedding) VALUES (?, ?)",
                            (entry_id, blob),
                        )
                        backfilled += 1
                    except Exception:
                        log.warning(
                            "dream_journal.backfill_row_failed",
                            entry_id=entry_id, exc_info=True,
                        )

                await db.commit()

        log.info(
            "dream_journal.backfill_complete",
            persona_id=persona_id, backfilled=backfilled, scanned=len(pending),
        )
        return backfilled

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update_entry(
        self,
        entry_id: str,
        content: str | None = None,
        weight: float | None = None,
        pinned: bool | None = None,
        *,
        user_id: str = "",
    ) -> None:
        """Update one or more fields on an existing entry."""
        if not user_id:
            raise ValueError("dream_entries update requires user_id")
        sets: list[str] = []
        values: list = []

        if content is not None:
            sets.append("content = ?")
            values.append(content)
        if weight is not None:
            sets.append("weight = ?")
            values.append(weight)
        if pinned is not None:
            sets.append("pinned = ?")
            values.append(1 if pinned else 0)

        if not sets:
            return  # nothing to update

        values.extend([entry_id, user_id])
        sql = (
            f"UPDATE dream_entries SET {', '.join(sets)} "
            "WHERE id = ? AND user_id = ?"
        )

        async with self._connect() as db:
            await db.execute(sql, values)
            await db.commit()

        log.debug("dream_journal.updated", entry_id=entry_id)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_entry(self, entry_id: str, *, user_id: str = "") -> None:
        """Permanently remove an entry and its FTS / vector index rows."""
        if not user_id:
            raise ValueError("dream_entries delete requires user_id")
        async with self._connect() as db:
            # Best-effort cleanup of FTS / vector indexes for this entry. Both
            # rowid lookups are scoped via the parent's WHERE so they only
            # match if the entry actually belongs to user_id.
            for aux_table in ("dream_entries_fts", "dream_entries_vec"):
                try:
                    await db.execute(
                        f"DELETE FROM {aux_table} WHERE rowid = ("
                        "SELECT rowid FROM dream_entries WHERE id = ? AND user_id = ?)",
                        (entry_id, user_id),
                    )
                except Exception:
                    log.debug(
                        "dream_journal.aux_delete_failed",
                        table=aux_table, entry_id=entry_id, exc_info=True,
                    )

            await db.execute(
                "DELETE FROM dream_entries WHERE id = ? AND user_id = ?",
                (entry_id, user_id),
            )
            await db.commit()

        log.debug("dream_journal.deleted", entry_id=entry_id)

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def compact_journal(
        self,
        persona_id: str,
        max_age_days: int = 30,
        *,
        user_id: str = "",
        count_threshold: int | None = None,
    ) -> dict:
        """Soft-delete unpinned entries older than max_age_days.

        Pinned entries are always preserved. Returns a dict with stats:
        ``{compacted, kept, deleted, total_active, gated}``.

        ``count_threshold`` (when set) defers time-trim entirely until
        the user has more than that many non-expired entries — the
        DreamCompactor relies on semantic dedup/cluster passes to do
        the bulk of pruning while the journal is small, only falling
        back to age-based deletion once volume actually warrants it.
        Without this gate, a curated 60-entry journal would still lose
        anything older than 30 days for no good reason. Returns
        ``{..., gated: True}`` so callers can distinguish "ran but
        nothing to do" from "skipped due to threshold".
        """
        cutoff = (datetime.now(UTC) - timedelta(days=max_age_days)).strftime("%Y-%m-%d %H:%M:%S")
        expires_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        uid_filter = " AND user_id = ?" if user_id else ""

        async with self._connect() as db:
            # Count threshold gate — total non-expired entries, all ages.
            # If we're below it, semantic compaction is the right tool;
            # don't lose content to age yet.
            if count_threshold is not None:
                total_params: list = [persona_id]
                if user_id:
                    total_params.append(user_id)
                async with db.execute(
                    f"""
                    SELECT COUNT(*) FROM dream_entries
                    WHERE persona_id = ? AND expires_at IS NULL{uid_filter}
                    """,
                    total_params,
                ) as cursor:
                    row = await cursor.fetchone()
                    total_active = row[0] if row else 0
                if total_active <= count_threshold:
                    log.debug(
                        "dream_journal.compact_gated",
                        persona_id=persona_id, total_active=total_active,
                        count_threshold=count_threshold,
                    )
                    return {
                        "compacted": 0, "kept": 0, "deleted": 0,
                        "total_active": total_active, "gated": True,
                    }

            # Count how many would be soft-deleted
            count_params: list = [persona_id, cutoff]
            if user_id:
                count_params.append(user_id)
            async with db.execute(
                f"""
                SELECT COUNT(*) FROM dream_entries
                WHERE persona_id = ?
                  AND pinned = 0
                  AND created_at <= ?
                  AND expires_at IS NULL{uid_filter}
                """,
                count_params,
            ) as cursor:
                row = await cursor.fetchone()
                compacted = row[0] if row else 0

            # Count pinned (kept regardless)
            pinned_params: list = [persona_id]
            if user_id:
                pinned_params.append(user_id)
            async with db.execute(
                f"SELECT COUNT(*) FROM dream_entries WHERE persona_id = ? AND pinned = 1{uid_filter}",
                pinned_params,
            ) as cursor:
                row = await cursor.fetchone()
                kept_pinned = row[0] if row else 0

            # Soft-delete: set expires_at on old unpinned entries
            update_params: list = [expires_at, persona_id, cutoff]
            if user_id:
                update_params.append(user_id)
            await db.execute(
                f"""
                UPDATE dream_entries
                SET expires_at = ?
                WHERE persona_id = ?
                  AND pinned = 0
                  AND created_at <= ?
                  AND expires_at IS NULL{uid_filter}
                """,
                update_params,
            )
            await db.commit()

        stats = {
            "compacted": compacted, "kept": kept_pinned, "deleted": 0,
            "gated": False,
        }
        log.info("dream_journal.compacted", persona_id=persona_id, **stats)
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        """Return a context manager for the DB connection.

        Uses the persistent ``_db`` when available (server runtime), or
        opens a fresh connection per call (test environments that skip
        ``initialize()``). Both paths apply the canonical augmentum.db
        pragma set so any writer in the system sees the same
        busy_timeout and journal mode — no contention surprises.
        """
        if self._db is not None:
            return _reuse_conn(self._db)
        return self._fresh_connect()

    @asynccontextmanager
    async def _fresh_connect(self):
        """Open a one-shot connection with augmentum.db pragmas applied.

        Test path. Production code uses the persistent ``self._db``
        from ``initialize()``.
        """
        from augmentum.state.backends.sqlite import (
            apply_augmentum_pragmas,
            install_safe_rollback,
        )

        conn = await aiosqlite.connect(self._db_path)
        try:
            await apply_augmentum_pragmas(conn)
            # Required on every persistent aiosqlite conn touching
            # augmentum.db — without it a failed DML pins the WAL via
            # stuck transaction state and cascades into "database is
            # locked" errors. See install_safe_rollback's docstring
            # for the long-form background and the 2026-05-22 incident.
            install_safe_rollback(conn)
            yield conn
        finally:
            await conn.close()

    @staticmethod
    def _row_to_entry(row: aiosqlite.Row) -> DreamEntry:
        """Convert a DB row to a DreamEntry dataclass."""
        return DreamEntry(
            id=row["id"],
            persona_id=row["persona_id"],
            content=row["content"],
            entry_type=DreamEntryType(row["entry_type"]),
            source_memories=json.loads(row["source_memories"]),
            source_sessions=json.loads(row["source_sessions"]),
            context_window=json.loads(row["context_window"]),
            embedding=row["embedding"],
            weight=row["weight"],
            pinned=bool(row["pinned"]),
            dream_cycle_id=row["dream_cycle_id"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )
