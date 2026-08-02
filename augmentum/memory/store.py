"""Memory store — CRUD operations with hybrid vector + FTS5 search."""

from __future__ import annotations

import asyncio
import json
import re as _re
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from math import exp
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.memory.embeddings import EmbeddingService
from augmentum.memory.events import log_event
from augmentum.memory.models import ExtractedFact, Memory, MemoryTier, MemoryType, SourceType
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import ModelBackend
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)

# Cosine similarity threshold for deduplication (above = same fact, update).
# Lowered from 0.92 to 0.88 to catch paraphrases like
# "I'm a software engineer" vs "I work as a software developer".
# Default: 0.88 — now configurable via settings.memory_dedup_threshold
DEDUP_THRESHOLD = 0.88
# Cosine similarity for contradiction detection (between this and dedup = related but different → supersede)
# Default: 0.78 — now configurable via settings.memory_contradiction_threshold
CONTRADICTION_THRESHOLD = 0.78
# Recency half-life in days (memories older than this get 50% weight)
RECENCY_HALF_LIFE_DAYS = 30.0

# Scopes that are STRICTLY ISOLATED from the general memory pool: their rows
# must NEVER surface in a scope=None ("general store") read, only in an
# explicit same-scope read. This is the MIRROR of the harness scope_strict
# read path — C1 stopped harness from reading other scopes; this stops the
# general store / chat / companion / management-UI (which read with scope=None)
# from reading harness rows. Isolation is bidirectional. Keep "harness" in sync
# with proxy/harness.py::HARNESS_SCOPE; add a scope here when a new surface must
# be physically unreachable from the general pool (coder/narrative use separate
# tables, so they need no entry).
ISOLATED_SCOPES: frozenset[str] = frozenset({"harness"})


def is_isolated_scope(scope: str | None) -> bool:
    """True when ``scope`` belongs to an isolated family — either the exact
    scope name or any ``<name>:...`` sub-scope (the harness layer keys
    per-project scopes as ``harness:<harness>:<project>``)."""
    if not scope:
        return False
    return any(scope == s or scope.startswith(s + ":") for s in ISOLATED_SCOPES)


def _isolation_sql(col: str) -> tuple[str, list[str]]:
    """SQL predicate (+ params) excluding all isolated-scope rows, including
    ``<scope>:*`` sub-scopes. Returns ("", []) when there are no isolated
    scopes. The predicate allows NULL (unscoped) rows through."""
    if not ISOLATED_SCOPES:
        return "", []
    parts: list[str] = []
    params: list[str] = []
    for s in sorted(ISOLATED_SCOPES):
        parts.append(f"{col} != ? AND {col} NOT LIKE ?")
        params.extend([s, s + ":%"])
    return f"({col} IS NULL OR ({' AND '.join(parts)}))", params

# Task-scoped flag set by :meth:`MemoryStore.batch_write`. While true,
# nested calls into ``store()``, ``update_content()`` and the other write
# helpers skip ``_write_lock`` re-acquisition (the batch holds it once)
# and skip their internal ``commit()`` (the batch commits once at the
# end). Because it's a ContextVar, only the task that opened the batch
# sees the flag — concurrent callers in other tasks still wait on the
# lock and commit normally. This is what turns an N-fact extraction
# burst from N transactions into 1.
_batch_active: ContextVar[bool] = ContextVar("memory_batch_active", default=False)


# Test-pollution patterns. Match content that's clearly from a test harness
# rather than a real user/assistant turn — bracketed UUIDs / "Live test memory
# fact" / canonical Wikipedia-style facts that ship as test fixtures. Generic
# across all users: nobody legitimately stores "[test-abc123]" or "Paris is the
# capital of France [cross-...]" as a personal fact.
_TEST_SENTINEL_RE = _re.compile(
    r"\[(?:test|cross|fixture|sentinel)-[0-9a-f]+\]"
    r"|^live test memory fact:"
    r"|^temporary fact for deletion test\b"
    r"|^cross-client persistence test\b",
    _re.IGNORECASE,
)


def _looks_like_test_sentinel(content: str) -> bool:
    """True if the memory content matches a known test-fixture pattern.

    Generic guard — works for any user/domain because it keys on harness
    markers (bracketed UUIDs, fixture phrasing) rather than user vocabulary.
    """
    if not content:
        return False
    return bool(_TEST_SENTINEL_RE.search(content))


class MemoryStore:
    """Cross-session memory store with hybrid retrieval (vector + FTS5 + RRF)."""

    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend
        self._write_lock = asyncio.Lock()
        self._consolidation_backend: ModelBackend | None = None

    def set_consolidation_backend(self, backend: ModelBackend, model: str = "") -> None:
        """Set the LLM backend for on-write consolidation."""
        self._consolidation_backend = backend

    @property
    def _consolidation_model(self) -> str | None:
        """Resolve consolidation model dynamically from runtime settings."""
        from augmentum.config import settings
        return settings.memory_llm_extraction_model or None

    @property
    def _conn(self):
        return self._backend.conn

    @property
    def _vec_enabled(self) -> bool:
        return self._backend.vec_enabled

    @property
    def db_path(self) -> str:
        """Path to the underlying SQLite file (used by dream + adjacent subsystems
        that need to open their own connections to the same DB).
        """
        return self._backend._db_path

    # ------------------------------------------------------------------
    # Batch transaction support
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def batch_write(self):
        """Coalesce N writes into a single SQLite transaction.

        Use when storing multiple facts in succession (e.g. after a
        batch extraction). Without this, each ``store_fact`` / ``update_content``
        commits independently — N facts = N writer-lock acquisitions
        and N fsyncs, which is the largest single source of write
        contention in this codebase. Inside the context, nested calls
        skip both lock re-acquisition (we hold it for the whole batch)
        and their internal commit (we commit once at the end).

        Concurrent writers in OTHER asyncio tasks are unaffected —
        ``_batch_active`` is a ContextVar so the flag is task-local.
        They wait on ``_write_lock`` as normal.

        Errors anywhere inside the block roll back the entire batch.
        """
        async with self._write_lock:
            token = _batch_active.set(True)
            try:
                await self._conn.execute("BEGIN")
                try:
                    yield
                    await self._conn.commit()
                except Exception:
                    await self._conn.rollback()
                    raise
            finally:
                _batch_active.reset(token)

    async def _maybe_commit(self) -> None:
        """Commit unless we're inside a :meth:`batch_write` context."""
        if not _batch_active.get():
            await self._conn.commit()

    async def _maybe_rollback(self) -> None:
        """Rollback unless we're inside a :meth:`batch_write` context.

        Inside a batch, the outer context handles rollback for the whole
        transaction — a nested rollback would end the batch transaction
        early and leave subsequent statements running in autocommit.
        """
        if not _batch_active.get():
            await self._conn.rollback()

    @asynccontextmanager
    async def _write_section(self):
        """Acquire ``_write_lock`` unless we're already inside a batch.

        Plain ``async with self._write_lock`` would deadlock when called
        from inside :meth:`batch_write` (the batch holds the lock for its
        full duration). This wrapper is the right thing for every
        write-protecting site in the file.
        """
        if _batch_active.get():
            yield
        else:
            async with self._write_lock:
                yield

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store(
        self,
        content: str,
        memory_type: MemoryType | str,
        user_id: str = "default",
        session_id: str | None = None,
        importance: float = 0.5,
        confidence: float = 0.8,
        source_type: SourceType | str | None = None,
        source_context: dict | None = None,
        scope: str | None = None,
        durability: object | None = None,
        scope_strict: bool = False,
    ) -> str:
        """Store a memory, deduplicating against existing entries.

        Returns the memory ID (existing if deduped, new if inserted).
        """
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        if isinstance(source_type, str):
            source_type = SourceType(source_type)

        # PII scrub pre-embedding so the vector keys on the scrubbed
        # form (otherwise two captures of the same email-bearing fact
        # would embed differently and dedup would miss). EXPLICIT facts
        # are exempt — the user's chosen phrasing is the opt-in.
        from augmentum.config import settings
        if settings.memory_pii_scrub_enabled and source_type != SourceType.EXPLICIT:
            from augmentum.memory.scrub import scrub_pii
            content, kinds = scrub_pii(content)
            if kinds:
                log.info("memory_pii_scrubbed", kinds=kinds, user_id=user_id)

        embedding = await asyncio.to_thread(EmbeddingService.embed_one, content)

        # Lock the search-then-write section to prevent duplicate inserts.
        # _write_section() is a no-op when called inside batch_write() —
        # the batch already holds the lock for the whole transaction.
        async with self._write_section():
            return await self._store_inner(
                content=content, embedding=embedding, memory_type=memory_type,
                user_id=user_id, session_id=session_id, importance=importance,
                confidence=confidence, source_type=source_type,
                source_context=source_context, scope=scope,
                durability=durability, scope_strict=scope_strict,
            )

    async def _store_inner(
        self, content: str, embedding: list[float], memory_type: MemoryType,
        user_id: str, session_id: str | None, importance: float,
        confidence: float, source_type: SourceType | None,
        source_context: dict | None, scope: str | None,
        durability: object | None = None, scope_strict: bool = False,
    ) -> str:
        """Inner store logic — must be called under _write_lock."""
        from augmentum.config import settings

        # Test-sentinel guard. Tests that write to the live DB instead of
        # an in-memory fixture pollute the long-term store; we'd rather
        # surface the bug than silently accumulate. EXPLICIT facts skip
        # the guard so a user can legitimately ask "remember this test".
        if source_type != SourceType.EXPLICIT and _looks_like_test_sentinel(content):
            log.warning("memory_test_sentinel_rejected", content=content[:80])
            return ""

        # Fast path: normalized text exact-match dedup (no embedding needed)
        existing_id = await self._text_dedup_check(
            content, user_id, scope=scope, scope_strict=scope_strict,
        )
        if existing_id:
            # Corroboration: an exact re-mention is evidence the fact matters.
            # Bump access + check promotion so earned-permanence can lift a
            # PROVISIONAL fact toward ACTIVE on re-mention. This path previously
            # no-op'd, so a re-stated fact never climbed — the promotion ladder
            # was unreachable for exact repeats. Gated, so legacy behavior is
            # unchanged when earned-permanence is off.
            if settings.memory_earned_permanence:
                await self._conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, "
                    "updated_at = ? WHERE id = ? AND user_id = ?",
                    (datetime.now(UTC).isoformat(), existing_id, user_id),
                )
                await self._maybe_commit()
                await self._maybe_promote(existing_id, user_id=user_id)
            log.debug("memory_text_deduped", memory_id=existing_id, content=content[:80])
            return existing_id

        # Check for near-duplicates and contradictions via vector search.
        # NOT filtered by memory_type — cross-type dedup catches overlaps.
        # limit=20 (was 5) so dense clusters don't push the canonical
        # match outside the scan window at scale — a user with 100s of
        # memories can have 5+ near-siblings of one cluster crowding the
        # top, hiding the real dedup target deeper in the neighbor list.
        if self._vec_enabled:
            # Scope-isolated contradiction/dedup search — matches recall's scope
            # filter so a scoped write (e.g. ``harness``) only supersedes/dedups
            # against its own scope (+ unscoped), never a different scope's
            # memory. ``scope=None`` is unfiltered, preserving prior behavior.
            similar = await self._vector_search_scored(
                embedding, user_id, limit=20, scope=scope, scope_strict=scope_strict,
            )
            consolidation_candidates: list[tuple[Memory, float]] = []

            dedup_thresh = settings.memory_dedup_threshold
            contradiction_thresh = settings.memory_contradiction_threshold

            for existing_mem, distance in similar:
                similarity = 1.0 - distance
                if similarity >= dedup_thresh:
                    await self._update_existing(existing_mem, content, embedding)
                    log.debug("memory_deduped", memory_id=existing_mem.id, content=content[:80])
                    return existing_mem.id
                if similarity >= contradiction_thresh:
                    log.info(
                        "memory_superseded",
                        old_id=existing_mem.id,
                        old_content=existing_mem.content[:60],
                        new_content=content[:60],
                        similarity=round(similarity, 3),
                    )
                    return await self._insert_and_supersede(
                        old_id=existing_mem.id,
                        content=content, embedding=embedding,
                        memory_type=memory_type, user_id=user_id,
                        session_id=session_id, importance=importance,
                        confidence=confidence, source_type=source_type,
                        source_context=source_context, scope=scope,
                    )
                # Track candidates in consolidation range (0.60-0.78)
                _CONSOLIDATION_LOW = 0.60
                _CONSOLIDATION_HIGH = 0.78
                if _CONSOLIDATION_LOW <= similarity < _CONSOLIDATION_HIGH:
                    consolidation_candidates.append((existing_mem, similarity))

        # Insert new memory (transaction ensures atomicity with vec table)
        memory_id = str(uuid.uuid4())
        blob = EmbeddingService.to_blob(embedding)
        now = datetime.now(UTC).isoformat()

        # Tier routing:
        # 1. EXPLICIT facts always start at ACTIVE (user-stated = trusted).
        # 2. Otherwise, if the LLM classified the fact as TRANSIENT, route
        #    to PROVISIONAL with the longer transient TTL — ephemeral
        #    state expires unless cosine shadow-touch proves it recurring.
        # 3. Otherwise, low-confidence (< 0.7) facts get the historical
        #    PROVISIONAL placement with the 7-day TTL.
        # 4. Otherwise (DURABLE / UNKNOWN + confidence ≥ 0.7), start ACTIVE.
        # See docs/superpowers/specs/2026-05-31-memory-establishment-rebalance.md.
        from augmentum.memory.models import Durability
        initial_tier = MemoryTier.ACTIVE.value
        provisional_expires = None
        if source_type != SourceType.EXPLICIT:
            from datetime import timedelta
            is_transient = (
                settings.memory_durability_classification_enabled
                and isinstance(durability, Durability)
                and durability == Durability.TRANSIENT
            )
            if is_transient:
                initial_tier = MemoryTier.PROVISIONAL.value
                ttl_days = max(1, int(settings.memory_durability_transient_ttl_days))
                provisional_expires = (datetime.now(UTC) + timedelta(days=ttl_days)).isoformat()
                log.info(
                    "memory_provisional",
                    reason="durability_transient",
                    ttl_days=ttl_days,
                    chars=len(content),
                )
                log.debug("memory_provisional_content", content=content[:60], reason="durability_transient")
            elif confidence < 0.7:
                initial_tier = MemoryTier.PROVISIONAL.value
                provisional_expires = (datetime.now(UTC) + timedelta(days=7)).isoformat()
                log.info(
                    "memory_provisional",
                    reason="low_confidence",
                    confidence=confidence,
                    chars=len(content),
                )
                log.debug("memory_provisional_content", content=content[:60], confidence=confidence)
            elif (
                settings.memory_earned_permanence
                and source_type == SourceType.EXTRACTED
            ):
                # Earned-permanence (subtractive memory, Slice 2): a passively
                # EXTRACTED fact is UNPROVEN — quarantine it in PROVISIONAL
                # (never injected) until corroboration (re-mention / topical
                # recurrence) promotes it. Deliberate writes (EXPLICIT user
                # "remember…", USER_MANUAL) skipped this whole block / aren't
                # EXTRACTED, so they still land ACTIVE. This is what stops an
                # off-hand line becoming a durable, recited "fact" on first
                # mention (the echo chamber). confidence is unreliable (the
                # extractor hardcodes 0.8), so source — not confidence — is the
                # signal here. See 2026-06-20-memory-subtractive-design.md.
                initial_tier = MemoryTier.PROVISIONAL.value
                provisional_expires = (datetime.now(UTC) + timedelta(days=7)).isoformat()
                log.info(
                    "memory_provisional",
                    reason="earned_permanence",
                    chars=len(content),
                )
                log.debug("memory_provisional_content", content=content[:60], reason="earned_permanence")

        # Extract evidence from source_context if available
        evidence = ""
        if source_context and isinstance(source_context, dict):
            evidence = source_context.get("evidence", "")

        try:
            await self._conn.execute(
                "INSERT INTO memories "
                "(id, user_id, session_id, content, memory_type, importance, confidence, "
                " embedding, valid_from, source_type, source_context, scope, "
                " tier, provisional_expires_at, evidence, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory_id,
                    user_id,
                    session_id,
                    content,
                    memory_type.value,
                    importance,
                    confidence,
                    blob,
                    now,
                    source_type.value if source_type else None,
                    json.dumps(source_context) if source_context else None,
                    scope,
                    initial_tier,
                    provisional_expires,
                    evidence,
                    now,
                    now,
                ),
            )

            # Insert into vec0 virtual table
            if self._vec_enabled:
                await self._conn.execute(
                    "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                    (memory_id, blob),
                )

            await self._maybe_commit()
        except Exception:
            await self._maybe_rollback()
            raise

        log.debug("memory_stored", memory_id=memory_id, type=memory_type.value, content=content[:80])
        return memory_id

    async def store_fact(
        self,
        fact: ExtractedFact,
        user_id: str = "default",
        session_id: str | None = None,
        is_explicit: bool = False,
        scope: str | None = None,
    ) -> str:
        """Store an ExtractedFact. Convenience wrapper around store()."""
        source_type = SourceType.EXPLICIT if is_explicit else SourceType.EXTRACTED
        # Bridge evidence from ExtractedFact.evidence into source_context
        # so _store_inner can persist it to the evidence DB column
        ctx = dict(fact.source_context) if fact.source_context else {}
        if fact.evidence and "evidence" not in ctx:
            ctx["evidence"] = fact.evidence
        return await self.store(
            content=fact.content,
            memory_type=fact.type,
            user_id=user_id,
            session_id=session_id,
            importance=fact.importance,
            confidence=fact.confidence,
            source_type=source_type,
            source_context=ctx or None,
            scope=scope,
            durability=getattr(fact, "durability", None),
        )

    async def recall(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 10,
        memory_types: list[MemoryType] | None = None,
        min_score: float = 0.0,
        scope: str | None = None,
        hyde_text: str = "",
        scope_strict: bool = False,
    ) -> list[Memory]:
        """Hybrid retrieval: vector + FTS5 + RRF.

        Returns memories sorted by combined relevance * recency * importance.

        ``hyde_text`` (optional) — a hypothetical-answer expansion of the
        query. When supplied, its embedding contributes a third RRF leg
        alongside the query vector and FTS5 keyword search. Short queries
        benefit most; longer queries see negligible change. Caller
        decides whether to compute it (see augmentum.memory.hyde).
        """
        vec_results: list[tuple[Memory, float]] = []
        hyde_results: list[tuple[Memory, float]] = []
        fts_results: list[tuple[Memory, float]] = []

        # Vector search (run in thread to avoid blocking the event loop)
        if self._vec_enabled:
            import asyncio
            embedding = await asyncio.to_thread(EmbeddingService.embed_query, query)
            vec_results = await self._vector_search_scored(
                embedding, user_id, limit=20, memory_types=memory_types,
                scope=scope, scope_strict=scope_strict,
            )
            if hyde_text and hyde_text.strip():
                hyde_emb = await asyncio.to_thread(EmbeddingService.embed_query, hyde_text)
                hyde_results = await self._vector_search_scored(
                    hyde_emb, user_id, limit=20, memory_types=memory_types,
                    scope=scope, scope_strict=scope_strict,
                )

        # FTS5 keyword search
        fts_results = await self._fts_search(
            query, user_id, limit=20, memory_types=memory_types,
            scope=scope, scope_strict=scope_strict,
        )

        # Reciprocal Rank Fusion (multi-source). The HyDE leg is dropped
        # from the merge when empty, so non-HyDE callers see identical
        # behavior to before this parameter existed.
        legs = [vec_results, fts_results]
        if hyde_results:
            legs.append(hyde_results)
        merged = self._rrf_merge_multi(legs, k=60)

        # Phase 2: Exclude PROVISIONAL tier — never inject unvalidated facts
        merged = [
            (mem, s) for mem, s in merged
            if (mem.tier if isinstance(mem.tier, str) else mem.tier.value) != MemoryTier.PROVISIONAL
        ]

        # Apply scoring: rrf × strength × importance × tier × source × surprise
        # Uses spaced-repetition decay and surprise scoring
        scored: list[tuple[Memory, float]] = []
        total_retrievals = await self._get_total_retrieval_count(user_id)

        for mem, rrf_score in merged:
            # Spaced-repetition strength (replaces uniform recency decay)
            strength = self._memory_strength(mem)
            # Use living importance (grows with access, decays over time)
            eff_importance = self._effective_importance(mem)
            score = rrf_score * strength * eff_importance

            # Tier weighting
            tier_val = mem.tier if isinstance(mem.tier, str) else mem.tier.value
            if tier_val == MemoryTier.CORE:
                score *= 1.3
            elif tier_val == MemoryTier.ARCHIVE:
                score *= 0.7

            # Boost explicit/manual memories
            if mem.source_type in ("user_manual", "explicit"):
                score *= 1.5

            # Surprise score: memories rarely retrieved but matching now
            # are surprisingly relevant — boost gently to surface "wow" recalls
            surprise = self._surprise_score(mem, total_retrievals)
            score *= (1.0 + 0.15 * surprise)

            scored.append((mem, score))

        # Filter by minimum score threshold
        if min_score > 0:
            scored = [(m, s) for m, s in scored if s >= min_score]

        scored.sort(key=lambda x: x[1], reverse=True)

        # Cross-encoder reranking for precision (after RRF + scoring).
        # Off-loaded to a thread because the BAAI cross-encoder runs ~1-10s
        # on 15-30 docs and was previously stalling the event loop on every
        # chat turn that triggered memory recall.
        scored = await asyncio.to_thread(self._rerank_memories, query, scored, limit)

        # Phase 3: Hebbian co-occurrence boost (after reranking, before final cut)
        scored = await self._apply_hebbian_boost(scored, user_id)

        # Phase 3: Associate expansion — pull co-occurring memories beyond top-K
        scored = await self._expand_associates(scored, user_id, limit)

        # Update access counts, retrieval_count, last_accessed_at, and check promotion
        result = [mem for mem, _ in scored[:limit]]
        if result:
            ids = [m.id for m in result]
            placeholders = ",".join("?" * len(ids))
            if user_id:
                await self._conn.execute(
                    f"UPDATE memories SET access_count = access_count + 1, "
                    f"retrieval_count = COALESCE(retrieval_count, 0) + 1, "
                    f"last_accessed = datetime('now'), "
                    f"last_accessed_at = datetime('now') "
                    f"WHERE id IN ({placeholders}) AND user_id = ?",
                    [*ids, user_id],
                )
            else:
                await self._conn.execute(
                    f"UPDATE memories SET access_count = access_count + 1, "
                    f"retrieval_count = COALESCE(retrieval_count, 0) + 1, "
                    f"last_accessed = datetime('now'), "
                    f"last_accessed_at = datetime('now') "
                    f"WHERE id IN ({placeholders})",
                    ids,
                )
            await self._conn.commit()

            # Phase 3: Record co-occurrence for all pairs in result set
            await self._record_cooccurrence(result, user_id)

            # Check if any recalled memories should be promoted
            for mem in result:
                await self._maybe_promote(mem.id, user_id=user_id)

        return result

    async def update_content(
        self,
        memory_id: str,
        new_content: str,
        new_importance: float | None = None,
        *,
        user_id: str,
    ) -> bool:
        """Update an existing memory's content and re-embed it.

        Used by the reconciliation extraction when the LLM identifies that
        a new user statement updates/corrects an existing memory.
        Returns True if the update succeeded.
        """
        async with self._write_section():
            # 1. Fetch existing memory (scoped to user)
            mem = await self.get(memory_id, user_id=user_id)
            if mem is None or mem.valid_until is not None:
                return False

            # 2. Re-embed the new content
            embedding = await asyncio.to_thread(EmbeddingService.embed_one, new_content)
            blob = EmbeddingService.to_blob(embedding)
            now = datetime.now(UTC).isoformat()

            try:
                # 3. Update the memories table
                if new_importance is not None:
                    await self._conn.execute(
                        "UPDATE memories SET content = ?, embedding = ?, importance = ?, "
                        "updated_at = ? WHERE id = ? AND user_id = ?",
                        (new_content, blob, new_importance, now, memory_id, user_id),
                    )
                else:
                    await self._conn.execute(
                        "UPDATE memories SET content = ?, embedding = ?, "
                        "updated_at = ? WHERE id = ? AND user_id = ?",
                        (new_content, blob, now, memory_id, user_id),
                    )

                # 4. Update the vec table
                if self._vec_enabled:
                    await self._conn.execute(
                        "UPDATE memories_vec SET embedding = ? WHERE memory_id = ?",
                        (blob, memory_id),
                    )

                await self._maybe_commit()
                log.info(
                    "memory_content_updated",
                    memory_id=memory_id,
                    old_content=mem.content[:60],
                    new_content=new_content[:60],
                )
                return True
            except Exception:
                await self._maybe_rollback()
                raise

    async def forget(self, memory_id: str, *, user_id: str) -> bool:
        """Soft-delete a memory by setting valid_until to now."""
        now = datetime.now(UTC).isoformat()
        cursor = await self._conn.execute(
            "UPDATE memories SET valid_until = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ? AND valid_until IS NULL",
            (now, now, memory_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def edit(self, memory_id: str, new_content: str, *, user_id: str) -> bool:
        """Update memory content and re-embed."""
        embedding = await asyncio.to_thread(EmbeddingService.embed_one, new_content)
        blob = EmbeddingService.to_blob(embedding)
        now = datetime.now(UTC).isoformat()

        async with self._write_section():
            try:
                cursor = await self._conn.execute(
                    "UPDATE memories SET content = ?, embedding = ?, updated_at = ? "
                    "WHERE id = ? AND user_id = ?",
                    (new_content, blob, now, memory_id, user_id),
                )

                if self._vec_enabled and cursor.rowcount > 0:
                    await self._conn.execute(
                        "UPDATE memories_vec SET embedding = ? WHERE memory_id = ?",
                        (blob, memory_id),
                    )

                await self._conn.commit()
                return cursor.rowcount > 0
            except Exception:
                await self._conn.rollback()
                raise

    async def get(self, memory_id: str, *, user_id: str) -> Memory | None:
        """Get a single memory by ID, scoped to the owning user."""
        cursor = await self._conn.execute(
            "SELECT * FROM memories WHERE id = ? AND user_id = ?",
            (memory_id, user_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_memory(dict(row))

    async def list_all(
        self,
        user_id: str = "default",
        memory_type: MemoryType | None = None,
        include_expired: bool = False,
        limit: int = 50,
        offset: int = 0,
        scope: str | None = None,
        tier: str | MemoryTier | None = None,
    ) -> list[Memory]:
        """Paginated listing of memories."""
        conditions = ["user_id = ?"]
        params: list = [user_id]

        if not include_expired:
            conditions.append("valid_until IS NULL")
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)
        if scope is not None:
            conditions.append("(scope = ? OR scope IS NULL)")
            params.append(scope)
        elif ISOLATED_SCOPES:
            # General (unscoped) listing must not pool isolated-surface rows
            # (e.g. harness coding conventions, incl. harness:* sub-scopes)
            # into the user's memory store.
            iso_sql, iso_params = _isolation_sql("scope")
            conditions.append(iso_sql)
            params.extend(iso_params)
        if tier is not None:
            tier_val = tier.value if isinstance(tier, MemoryTier) else str(tier)
            conditions.append("tier = ?")
            params.append(tier_val)

        where = " AND ".join(conditions)
        params.extend([limit, offset])

        cursor = await self._conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(dict(row)) for row in rows]

    async def count(self, user_id: str = "default") -> dict[str, int]:
        """Count memories by type for a user (excludes isolated scopes so the
        counts match the general ``list_all`` view)."""
        iso_sql = ""
        iso_params: list = []
        if ISOLATED_SCOPES:
            pred, iso_params = _isolation_sql("scope")
            iso_sql = f" AND {pred}"
        cursor = await self._conn.execute(
            "SELECT memory_type, COUNT(*) as cnt FROM memories "
            f"WHERE user_id = ? AND valid_until IS NULL{iso_sql} GROUP BY memory_type",
            (user_id, *iso_params),
        )
        rows = await cursor.fetchall()
        result = {row[0]: row[1] for row in rows}
        cursor2 = await self._conn.execute(
            f"SELECT COUNT(*) FROM memories WHERE user_id = ? AND valid_until IS NULL{iso_sql}",
            (user_id, *iso_params),
        )
        total = (await cursor2.fetchone())[0]
        result["total"] = total
        return result

    async def supersede(self, old_id: str, new_content: str, *, user_id: str, **kwargs) -> str:
        """Mark an existing memory as expired and store a new version.

        Returns the new memory ID.
        """
        # user_id is passed both as the explicit kwarg (for the UPDATE filter
        # here) and as part of store()'s kwargs — drop any stale copy the
        # caller may have also stuffed into kwargs to avoid double-keyword.
        kwargs.pop("user_id", None)
        scope = kwargs.get("scope")
        scope_strict = kwargs.get("scope_strict", False)
        new_id = await self.store(new_content, user_id=user_id, **kwargs)
        now = datetime.now(UTC).isoformat()
        if scope is not None and scope_strict:
            # Hard isolation guard: only ever expire a memory in the SAME strict
            # scope, so a mis-passed id can never cross a scope boundary (a
            # harness write must not be able to supersede a companion memory).
            await self._conn.execute(
                "UPDATE memories SET valid_until = ?, superseded_by = ?, updated_at = ? "
                "WHERE id = ? AND user_id = ? AND scope = ? AND valid_until IS NULL",
                (now, new_id, now, old_id, user_id, scope),
            )
        else:
            await self._conn.execute(
                "UPDATE memories SET valid_until = ?, superseded_by = ?, updated_at = ? "
                "WHERE id = ? AND user_id = ? AND valid_until IS NULL",
                (now, new_id, now, old_id, user_id),
            )
        await self._conn.commit()
        return new_id

    async def get_history(self, memory_id: str, *, user_id: str) -> list[Memory]:
        """Get the version history of a memory (following superseded_by chain backwards)."""
        chain: list[Memory] = []
        current = await self.get(memory_id, user_id=user_id)
        if current:
            chain.append(current)

        # Walk backwards: find who was superseded by this (scoped to user)
        visited = {memory_id}
        while True:
            cursor = await self._conn.execute(
                "SELECT * FROM memories WHERE superseded_by = ? AND user_id = ?",
                (chain[-1].id if chain else memory_id, user_id),
            )
            row = await cursor.fetchone()
            if row is None:
                break
            mem = self._row_to_memory(dict(row))
            if mem.id in visited:
                break
            visited.add(mem.id)
            chain.append(mem)

        # Walk forward: follow superseded_by chain
        if current and current.superseded_by:
            next_id = current.superseded_by
            while next_id and next_id not in visited:
                visited.add(next_id)
                mem = await self.get(next_id, user_id=user_id)
                if mem is None:
                    break
                chain.insert(0, mem)
                next_id = mem.superseded_by

        # Sort oldest first
        chain.sort(key=lambda m: m.created_at)
        return chain

    async def get_compaction_candidates(
        self,
        user_id: str = "default",
        max_age_days: float = 30.0,
        max_access_count: int = 2,
        max_importance: float = 0.3,
    ) -> list[Memory]:
        """Find memories eligible for compaction based on age, access, and importance.

        CORE tier memories are NEVER compacted — only the user can delete them.
        """
        cutoff = (datetime.now(UTC) - timedelta(days=max_age_days)).isoformat()
        cursor = await self._conn.execute(
            "SELECT * FROM memories WHERE user_id = ? AND valid_until IS NULL "
            "AND tier != 'core' "
            "AND updated_at < ? AND access_count < ? AND importance < ? "
            "ORDER BY updated_at ASC LIMIT 100",
            (user_id, cutoff, max_access_count, max_importance),
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(dict(row)) for row in rows]

    async def update_tier(
        self,
        memory_id: str,
        tier: str | MemoryTier,
        *,
        user_id: str,
        source: str = "system",
        log_change: bool = True,
    ) -> bool:
        """Update a memory's tier.

        ``source`` lands in the event detail ("manual" for user-initiated
        changes — the stream UI only surfaces those). ``log_change=False``
        skips the tier_change event entirely; _maybe_promote uses it because
        it logs its own richer "promotion" event (logging both produced
        duplicate cards in the memory timeline).
        """
        tier_val = tier.value if isinstance(tier, MemoryTier) else str(tier)
        now = datetime.now(UTC).isoformat()
        cursor = await self._conn.execute(
            "UPDATE memories SET tier = ?, last_compacted_at = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (tier_val, now, now, memory_id, user_id),
        )
        await self._conn.commit()
        if cursor.rowcount > 0 and log_change:
            try:
                await log_event(
                    self._conn, "tier_change",
                    user_id=user_id,
                    memory_id=memory_id,
                    detail={"to_tier": tier_val, "source": source},
                )
            except Exception:
                log.debug("tier_change_event_failed", exc_info=True)
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Tier promotion
    # ------------------------------------------------------------------

    # Promotion thresholds: access_count >= N AND importance >= X → promote
    _PROMOTE_TO_ACTIVE_ACCESS = 3     # tier 1 (archive) → active after 3 accesses
    _PROMOTE_TO_ACTIVE_IMPORTANCE = 0.4
    _PROMOTE_TO_CORE_ACCESS = 5       # active → core after 5 accesses
    _PROMOTE_TO_CORE_IMPORTANCE = 0.6

    async def bump_access(self, memory_id: str, *, user_id: str, by: int = 1) -> bool:
        """Increment a memory's access_count — the corroboration signal the
        promotion ladder reads. Public so the Evidence Bus can advance the
        ladder one step per *independent* corroborating source (triangulation),
        reusing the proven promotion path rather than a parallel one. Returns
        True if a row was updated."""
        by = max(1, int(by))
        cursor = await self._conn.execute(
            "UPDATE memories SET access_count = access_count + ? "
            "WHERE id = ? AND user_id = ? AND valid_until IS NULL",
            (by, memory_id, user_id),
        )
        await self._maybe_commit()
        return bool(cursor.rowcount)

    async def _maybe_promote(self, memory_id: str, *, user_id: str) -> None:
        """Check if a memory should be promoted based on access_count + importance.

        Promotion path: archive → active → core.
        CORE memories survive compaction and are always injected via CoreProfileManager.
        """
        mem = await self.get(memory_id, user_id=user_id)
        if mem is None or mem.valid_until is not None:
            return

        current_tier = mem.tier if isinstance(mem.tier, str) else mem.tier.value

        # PROVISIONAL → ACTIVE: promote when corroborated enough (re-mention /
        # topical recurrence). Threshold is tunable via
        # memory_corroboration_promote_access (earned-permanence); falls back to
        # the historical 3 when the setting is absent.
        if current_tier == MemoryTier.PROVISIONAL:
            from augmentum.config import settings as _settings
            _promote_at = max(1, int(getattr(
                _settings, "memory_corroboration_promote_access", 3,
            )))
            if mem.access_count >= _promote_at:
                await self.update_tier(memory_id, MemoryTier.ACTIVE, user_id=user_id, log_change=False)
                # Clear the TTL (scoped — _maybe_promote already fetched the
                # memory under this user_id, so this just prevents parallel
                # races from touching another tenant's row).
                await self._conn.execute(
                    "UPDATE memories SET provisional_expires_at = NULL "
                    "WHERE id = ? AND user_id = ?",
                    (memory_id, user_id),
                )
                await self._conn.commit()
                log.info(
                    "memory_promoted_from_provisional",
                    memory_id=memory_id,
                    access_count=mem.access_count,
                    content=mem.content[:60],
                )
                try:
                    await log_event(
                        self._conn, "promotion",
                        user_id=user_id,
                        memory_id=memory_id,
                        detail={"from_tier": "provisional", "to_tier": "active", "reason": f"access_count >= {_promote_at}"},
                    )
                except Exception:
                    log.debug("promotion_event_failed", exc_info=True)
            return  # Don't check further promotions for PROVISIONAL

        eff_importance = self._effective_importance(mem)

        if (
            current_tier == MemoryTier.ACTIVE
            and mem.access_count >= self._PROMOTE_TO_CORE_ACCESS
            and eff_importance >= self._PROMOTE_TO_CORE_IMPORTANCE
        ):
            await self.update_tier(memory_id, MemoryTier.CORE, user_id=user_id, log_change=False)
            log.info(
                "memory_promoted_to_core",
                memory_id=memory_id,
                access_count=mem.access_count,
                importance=mem.importance,
                effective_importance=round(eff_importance, 3),
                content=mem.content[:60],
            )
            try:
                await log_event(
                    self._conn, "promotion",
                    user_id=user_id,
                    memory_id=memory_id,
                    detail={"from_tier": "active", "to_tier": "core", "reason": f"access_count >= {self._PROMOTE_TO_CORE_ACCESS} and importance >= {self._PROMOTE_TO_CORE_IMPORTANCE}"},
                )
            except Exception:
                log.debug("promotion_event_failed", exc_info=True)
        elif (
            current_tier == MemoryTier.ARCHIVE
            and mem.access_count >= self._PROMOTE_TO_ACTIVE_ACCESS
            and eff_importance >= self._PROMOTE_TO_ACTIVE_IMPORTANCE
        ):
            await self.update_tier(memory_id, MemoryTier.ACTIVE, user_id=user_id, log_change=False)
            log.info(
                "memory_promoted_to_active",
                memory_id=memory_id,
                access_count=mem.access_count,
                importance=mem.importance,
                effective_importance=round(eff_importance, 3),
                content=mem.content[:60],
            )
            try:
                await log_event(
                    self._conn, "promotion",
                    user_id=user_id,
                    memory_id=memory_id,
                    detail={"from_tier": "archive", "to_tier": "active", "reason": f"access_count >= {self._PROMOTE_TO_ACTIVE_ACCESS} and importance >= {self._PROMOTE_TO_ACTIVE_IMPORTANCE}"},
                )
            except Exception:
                log.debug("promotion_event_failed", exc_info=True)

    # ------------------------------------------------------------------
    # Phase 2: PROVISIONAL tier lifecycle
    # ------------------------------------------------------------------

    async def cleanup_provisional(self) -> int:
        """Delete expired PROVISIONAL memories (and their vec entries). Returns count deleted."""
        try:
            # First collect IDs to also clean vec table
            cursor = await self._conn.execute(
                "SELECT id FROM memories WHERE tier = 'provisional' "
                "AND provisional_expires_at IS NOT NULL "
                "AND provisional_expires_at < datetime('now')"
            )
            expired_ids = [row[0] for row in await cursor.fetchall()]
            if not expired_ids:
                return 0

            placeholders = ",".join("?" * len(expired_ids))

            # Delete from vec table first (no FK cascade)
            if self._vec_enabled:
                try:
                    await self._conn.execute(
                        f"DELETE FROM memories_vec WHERE memory_id IN ({placeholders})",
                        expired_ids,
                    )
                except Exception as exc:
                    log.debug(
                        "memory_store_vec_cleanup_skipped",
                        count=len(expired_ids),
                        error=str(exc),
                    )

            # Delete from memories (triggers FTS5 cleanup)
            await self._conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                expired_ids,
            )
            await self._conn.commit()
            log.info("provisional_cleanup", deleted=len(expired_ids))
            return len(expired_ids)
        except Exception:
            log.warning("provisional_cleanup_failed", exc_info=True)
            return 0

    async def retroactive_demote(self) -> int:
        """Demote long-idle ACTIVE memories to ARCHIVE.

        Sweeps across all users in a single pass. A memory is demoted
        iff ALL of:
          - tier = ACTIVE (CORE is exempt — already cleared a higher bar)
          - source_type != EXPLICIT (user said it; user owns it)
          - importance < memory_demotion_importance_floor
          - COALESCE(retrieval_count, 0) < memory_demotion_min_retrievals
          - last_accessed_at older than memory_demotion_idle_days
            (falls back to created_at when last_accessed_at is NULL —
            an ACTIVE memory that's been here since day one and never
            been retrieved is exactly the kind we want to demote)

        Records every transition in ``memory_tier_history`` so the
        inspector UI can audit + revert. ARCHIVE memories remain
        searchable (scored 0.7x in recall) and can promote back via
        the existing access-count path; nothing is destroyed.

        Returns the number of demotions performed.
        """
        from augmentum.config import settings
        if not settings.memory_retroactive_demotion_enabled:
            return 0
        idle_days = max(1, int(settings.memory_demotion_idle_days))
        min_retrievals = max(0, int(settings.memory_demotion_min_retrievals))
        importance_floor = float(settings.memory_demotion_importance_floor)
        try:
            cursor = await self._conn.execute(
                """
                SELECT id, user_id FROM memories
                WHERE tier = 'active'
                  AND (source_type IS NULL OR source_type != 'explicit')
                  AND importance < ?
                  AND COALESCE(retrieval_count, 0) < ?
                  AND COALESCE(last_accessed_at, created_at) IS NOT NULL
                  AND COALESCE(last_accessed_at, created_at) < datetime('now', ?)
                  AND valid_until IS NULL
                """,
                (importance_floor, min_retrievals, f"-{idle_days} days"),
            )
            candidates = [(row[0], row[1] or "") for row in await cursor.fetchall()]
            if not candidates:
                return 0

            now = datetime.now(UTC).isoformat()
            reason = (
                f"idle>{idle_days}d retrievals<{min_retrievals} "
                f"importance<{importance_floor}"
            )

            placeholders = ",".join("?" * len(candidates))
            mem_ids = [cid for cid, _uid in candidates]

            await self._conn.execute(
                f"UPDATE memories SET tier = 'archive', updated_at = ? "
                f"WHERE id IN ({placeholders})",
                [now, *mem_ids],
            )
            # Batch history inserts. One row per demotion so the inspector
            # can show per-memory transitions.
            await self._conn.executemany(
                "INSERT INTO memory_tier_history "
                "(id, user_id, memory_id, from_tier, to_tier, reason, transitioned_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (str(uuid.uuid4()), uid, mid, "active", "archive", reason, now)
                    for mid, uid in candidates
                ],
            )
            await self._conn.commit()
            log.info(
                "memory_retroactive_demoted",
                count=len(candidates),
                idle_days=idle_days,
                min_retrievals=min_retrievals,
                importance_floor=importance_floor,
            )
            return len(candidates)
        except Exception:
            log.warning("memory_retroactive_demote_failed", exc_info=True)
            return 0

    async def revert_tier_transition(self, memory_id: str, user_id: str) -> bool:
        """Revert the most recent tier transition for a memory.

        Reads the latest ``memory_tier_history`` row, flips the memory's
        tier back to ``from_tier``, writes a corresponding "revert"
        history row noting the user undo. Returns True on success.
        """
        try:
            cursor = await self._conn.execute(
                "SELECT from_tier, to_tier FROM memory_tier_history "
                "WHERE memory_id = ? AND user_id = ? "
                "ORDER BY transitioned_at DESC LIMIT 1",
                (memory_id, user_id),
            )
            row = await cursor.fetchone()
            if row is None:
                return False
            prev_tier, current_tier = row[0], row[1]
            now = datetime.now(UTC).isoformat()
            await self._conn.execute(
                "UPDATE memories SET tier = ?, updated_at = ? "
                "WHERE id = ? AND user_id = ?",
                (prev_tier, now, memory_id, user_id),
            )
            await self._conn.execute(
                "INSERT INTO memory_tier_history "
                "(id, user_id, memory_id, from_tier, to_tier, reason, transitioned_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), user_id, memory_id, current_tier, prev_tier,
                 "user_revert", now),
            )
            await self._conn.commit()
            log.info("memory_tier_reverted", memory_id=memory_id,
                     from_tier=current_tier, to_tier=prev_tier)
            return True
        except Exception:
            log.warning("memory_tier_revert_failed", memory_id=memory_id, exc_info=True)
            return False

    async def shadow_touch_provisional(
        self,
        query_text: str,
        user_id: str,
        *,
        threshold: float | None = None,
        max_bumps: int | None = None,
        keyword_fallback: list[str] | None = None,
    ) -> int:
        """Bump access_count on PROVISIONAL memories topically related to a batch.

        Replaces the historical ``content LIKE %keyword%`` shadow-touch
        with embedding cosine similarity against the batch's query text.
        Lexical overlap promoted unrelated facts whenever a high-frequency
        word ('model', 'project', 'report') happened to appear; cosine
        keys on actual topical recurrence regardless of the user's domain.

        Returns the number of memories bumped. Bumps are capped via
        ``max_bumps`` so a semantically broad batch can't cascade-promote
        the entire PROVISIONAL pool in one shot.

        Fallback path: when ``_vec_enabled`` is False (sqlite-vec not
        loaded) and ``keyword_fallback`` is supplied, behaves identically
        to the old LIKE-match logic so deployments without the extension
        still see *some* promotion signal. Document the dependency in
        ops notes — cosine is the recommended path.
        """
        from augmentum.config import settings

        eff_threshold = threshold if threshold is not None else settings.memory_shadow_touch_threshold
        eff_cap = max_bumps if max_bumps is not None else settings.memory_shadow_touch_max_per_batch
        text = (query_text or "").strip()
        if not text or not user_id:
            return 0

        # Preferred path: embedding cosine. Compute the batch embedding once
        # and walk the top neighbors. ``_vector_search_scored`` already
        # filters to the requested user, so we only post-filter on tier.
        if self._vec_enabled:
            try:
                embedding = await asyncio.to_thread(EmbeddingService.embed_query, text)
            except Exception:
                log.debug("shadow_touch_embed_failed", exc_info=True)
                embedding = None
            if embedding is not None:
                neighbors = await self._vector_search_scored(
                    embedding, user_id, limit=max(20, eff_cap * 4),
                )
                bumped = 0
                bumped_ids: list[str] = []
                for mem, distance in neighbors:
                    if bumped >= eff_cap:
                        break
                    tier_val = mem.tier if isinstance(mem.tier, str) else mem.tier.value
                    if tier_val != MemoryTier.PROVISIONAL.value:
                        continue
                    similarity = 1.0 - distance
                    if similarity < eff_threshold:
                        continue
                    bumped_ids.append(mem.id)
                    bumped += 1
                if bumped_ids:
                    placeholders = ",".join("?" * len(bumped_ids))
                    await self._conn.execute(
                        f"UPDATE memories SET access_count = access_count + 1 "
                        f"WHERE id IN ({placeholders})",
                        bumped_ids,
                    )
                    await self._maybe_commit()
                    # Corroboration: topical recurrence just bumped these
                    # PROVISIONAL facts — check promotion so they can climb to
                    # ACTIVE once corroborated enough. Previously the bump
                    # happened but nothing ever called _maybe_promote on a
                    # PROVISIONAL row (recall excludes it), so earned facts were
                    # stranded until TTL expiry. Gated to earned-permanence.
                    if settings.memory_earned_permanence:
                        for mid in bumped_ids:
                            await self._maybe_promote(mid, user_id=user_id)
                    log.debug(
                        "shadow_touch_cosine",
                        bumped=bumped,
                        threshold=eff_threshold,
                        user_id=user_id,
                    )
                return bumped

        # Fallback path: caller supplied keywords for the legacy LIKE match.
        # Kept so deployments without sqlite-vec preserve current behavior
        # rather than silently lose shadow-touch entirely. The new defaults
        # (cap = 5) still apply via early-exit in the LIKE branch.
        kws = [k for k in (keyword_fallback or []) if k]
        if not kws:
            return 0
        try:
            like_clauses = " OR ".join(["content LIKE ?"] * len(kws))
            params: list = [user_id]
            params.extend(f"%{kw}%" for kw in kws)
            cursor = await self._conn.execute(
                f"UPDATE memories SET access_count = access_count + 1 "
                "WHERE user_id = ? AND tier = 'provisional' "
                "AND valid_until IS NULL "
                f"AND ({like_clauses})",
                params,
            )
            await self._maybe_commit()
            return cursor.rowcount or 0
        except Exception:
            log.debug("shadow_touch_like_fallback_failed", exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Phase 3: Hebbian co-occurrence
    # ------------------------------------------------------------------

    async def _apply_hebbian_boost(
        self,
        scored: list[tuple[Memory, float]],
        user_id: str,
    ) -> list[tuple[Memory, float]]:
        """Apply Hebbian co-occurrence boost to scored results."""
        if len(scored) < 2:
            return scored

        ids = [m.id for m, _ in scored]
        max_cooccur = 1  # prevent division by zero

        # Fetch max co-occurrence count for this user
        try:
            cursor = await self._conn.execute(
                "SELECT MAX(count) FROM memory_cooccurrence WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
            if row and row[0]:
                max_cooccur = max(1, row[0])
        except Exception:
            return scored  # table may not exist yet

        # Batch fetch all co-occurrence pairs for these IDs in one query
        try:
            placeholders = ",".join("?" * len(ids))
            cursor = await self._conn.execute(
                f"SELECT id_a, id_b, count FROM memory_cooccurrence "
                f"WHERE user_id = ? "
                f"AND id_a IN ({placeholders}) AND id_b IN ({placeholders}) "
                f"AND count >= 3",
                [user_id, *ids, *ids],
            )
            cooccur_rows = await cursor.fetchall()
        except Exception:
            return scored  # table may not exist yet

        # Build lookup dict: (id_a, id_b) -> count  (canonical order)
        cooccur_map: dict[tuple[str, str], int] = {}
        for row in cooccur_rows:
            cooccur_map[(row[0], row[1])] = row[2]

        # For each memory, sum co-occurrence with other results
        id_set = set(ids)
        boosted: list[tuple[Memory, float]] = []
        for mem, score in scored:
            cooccur_sum = 0
            for other_id in id_set:
                if other_id == mem.id:
                    continue
                # Canonical ordering: smaller ID first
                pair = tuple(sorted([mem.id, other_id]))
                count = cooccur_map.get(pair, 0)
                if count >= 3:
                    cooccur_sum += count

            # Hebbian boost: 1.0 to 1.3 (capped)
            hebbian = min(1.3, 1.0 + 0.2 * (cooccur_sum / max_cooccur))
            boosted.append((mem, score * hebbian))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    async def _expand_associates(
        self,
        scored: list[tuple[Memory, float]],
        user_id: str,
        limit: int,
    ) -> list[tuple[Memory, float]]:
        """Expand results with top co-occurring associates beyond top-K."""
        if len(scored) < 2:
            return scored

        result_ids = {m.id for m, _ in scored}
        seed_ids = [m.id for m, _ in scored[:limit]]

        # Batch fetch all co-occurrence rows for the seed IDs in two queries
        # (one for each direction: id_a=seed and id_b=seed)
        associates: list[tuple[str, int]] = []  # (memory_id, cooccur_count)
        try:
            placeholders = ",".join("?" * len(seed_ids))
            # Direction 1: seed is id_a, associate is id_b
            cursor = await self._conn.execute(
                f"SELECT id_a, id_b, count FROM memory_cooccurrence "
                f"WHERE user_id = ? AND id_a IN ({placeholders}) AND count >= 3 "
                f"ORDER BY count DESC",
                [user_id, *seed_ids],
            )
            rows_ab = await cursor.fetchall()
            # Direction 2: seed is id_b, associate is id_a
            cursor = await self._conn.execute(
                f"SELECT id_b, id_a, count FROM memory_cooccurrence "
                f"WHERE user_id = ? AND id_b IN ({placeholders}) AND count >= 3 "
                f"ORDER BY count DESC",
                [user_id, *seed_ids],
            )
            rows_ba = await cursor.fetchall()

            # Collect unique associates not already in results
            for rows in (rows_ab, rows_ba):
                for row in rows:
                    assoc_id = row[1]
                    if assoc_id not in result_ids:
                        associates.append((assoc_id, row[2]))
                        result_ids.add(assoc_id)
        except Exception:
            return scored  # table may not exist yet

        # Pick top-2 associates, batch-fetch their Memory objects
        associates.sort(key=lambda x: x[1], reverse=True)
        top_assoc_ids = [aid for aid, _ in associates[:2]]
        if not top_assoc_ids:
            return scored

        # Batch fetch memories by ID — scope to user_id for tenant isolation.
        placeholders = ",".join("?" * len(top_assoc_ids))
        try:
            if user_id:
                cursor = await self._conn.execute(
                    f"SELECT * FROM memories WHERE id IN ({placeholders}) AND user_id = ?",
                    [*top_assoc_ids, user_id],
                )
            else:
                cursor = await self._conn.execute(
                    f"SELECT * FROM memories WHERE id IN ({placeholders})",
                    top_assoc_ids,
                )
            rows = await cursor.fetchall()
        except Exception:
            return scored

        added = 0
        for row in rows:
            assoc_mem = self._row_to_memory(dict(row))
            if assoc_mem.valid_until is None:
                tier_val = assoc_mem.tier if isinstance(assoc_mem.tier, str) else assoc_mem.tier.value
                if tier_val != MemoryTier.PROVISIONAL:
                    scored.append((assoc_mem, 0.01))  # low score, won't outrank direct results
                    added += 1

        if added:
            log.debug("hebbian_associate_expansion", added=added)

        return scored

    async def _record_cooccurrence(
        self,
        memories: list[Memory],
        user_id: str,
    ) -> None:
        """Record co-occurrence for all pairs in a retrieval result set."""
        if len(memories) < 2:
            return

        # Only count ACTIVE+ tier memories
        active_mems = [
            m for m in memories
            if (m.tier if isinstance(m.tier, str) else m.tier.value)
            in (MemoryTier.CORE, MemoryTier.ACTIVE, "core", "active")
        ]
        if len(active_mems) < 2:
            return

        # Build all pairs first, then one executemany. The original double-
        # loop did one ``await`` per pair = O(N²) round-trips through
        # aiosqlite's worker thread. With 20 active memories that's 190
        # round-trips; with 50 it's 1225 — fired on every memory recall,
        # blocking every other DB-bound coroutine for the duration.
        pairs: list[tuple] = []
        for i in range(len(active_mems)):
            for j in range(i + 1, len(active_mems)):
                id_a, id_b = sorted([active_mems[i].id, active_mems[j].id])
                pairs.append((user_id, id_a, id_b))

        if not pairs:
            return

        try:
            await self._conn.executemany(
                "INSERT INTO memory_cooccurrence (user_id, id_a, id_b, count, last_updated) "
                "VALUES (?, ?, ?, 1, datetime('now')) "
                "ON CONFLICT(user_id, id_a, id_b) DO UPDATE SET "
                "count = count + 1, last_updated = datetime('now')",
                pairs,
            )
            await self._conn.commit()
        except Exception:
            # Table may not exist yet (pre-migration)
            log.debug("cooccurrence_record_failed", exc_info=True)

    async def decay_cooccurrence(self, decay_factor: float = 0.99) -> int:
        """Weekly decay: multiply all co-occurrence counts by decay_factor.

        Prevents stale associations from dominating. Call weekly via background task.
        Returns number of rows affected.
        """
        try:
            cursor = await self._conn.execute(
                "UPDATE memory_cooccurrence SET count = MAX(1, CAST(count * ? AS INTEGER))",
                (decay_factor,),
            )
            # Clean up zero/one counts that have decayed fully
            await self._conn.execute(
                "DELETE FROM memory_cooccurrence WHERE count <= 0"
            )
            await self._conn.commit()
            return cursor.rowcount
        except Exception:
            log.debug("cooccurrence_decay_failed", exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Private: search helpers
    # ------------------------------------------------------------------

    async def _vector_search(
        self,
        embedding: list[float],
        user_id: str,
        threshold: float = 0.5,
        limit: int = 10,
    ) -> list[Memory]:
        """Find memories by vector similarity (cosine distance)."""
        results = await self._vector_search_scored(embedding, user_id, limit=limit)
        # threshold is cosine distance (0 = identical, 2 = opposite)
        # Convert similarity threshold: distance < (1 - threshold)
        max_distance = 1 - threshold
        return [mem for mem, dist in results if dist <= max_distance]

    async def _vector_search_scored(
        self,
        embedding: list[float],
        user_id: str,
        limit: int = 20,
        memory_types: list[MemoryType] | None = None,
        scope: str | None = None,
        scope_strict: bool = False,
    ) -> list[tuple[Memory, float]]:
        """Vector search returning (memory, distance) pairs."""
        if not self._vec_enabled:
            return []

        blob = EmbeddingService.to_blob(embedding)

        # KNN k: the vec table is global (all users, all scopes) and the
        # user/scope filters run AFTER the KNN cut below. With the default
        # k = limit*2 a small isolated scope (e.g. a harness:<project> scope
        # holding a handful of rows) essentially never survives the global
        # top-k, making scoped recall silently blind. When a scope filter
        # will discard candidates, over-fetch a bounded larger pool.
        k = limit * 2 if scope is None else max(limit * 2, 256)
        cursor = await self._conn.execute(
            "SELECT memory_id, distance FROM memories_vec "
            "WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (blob, k),  # fetch extra to filter
        )
        vec_rows = await cursor.fetchall()

        if not vec_rows:
            return []

        # Batch fetch all candidate memories in one query
        vec_ids = [row[0] for row in vec_rows]
        distance_map: dict[str, float] = {row[0]: row[1] for row in vec_rows}

        placeholders = ",".join("?" * len(vec_ids))
        cursor = await self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            vec_ids,
        )
        mem_rows = await cursor.fetchall()
        mem_by_id: dict[str, Memory] = {}
        for row in mem_rows:
            mem = self._row_to_memory(dict(row))
            mem_by_id[mem.id] = mem

        # Filter and build results in original distance order
        results: list[tuple[Memory, float]] = []
        for vid in vec_ids:
            mem = mem_by_id.get(vid)
            if mem is None or mem.user_id != user_id or mem.valid_until is not None:
                continue
            if memory_types and MemoryType(mem.memory_type) not in memory_types:
                continue
            # Scope filter. Default: match the scope OR include unscoped
            # (universal) memories. scope_strict=True (the isolated harness
            # scope) excludes unscoped, so another surface's memory — including
            # personal companion facts stored unscoped — can never surface here.
            if scope is not None:
                if scope_strict:
                    if mem.scope != scope:
                        continue
                elif mem.scope is not None and mem.scope != scope:
                    continue
            elif is_isolated_scope(mem.scope):
                # General (unscoped) recall must never surface an isolated
                # scope's memory (e.g. harness coding conventions, incl.
                # harness:* sub-scopes).
                continue
            results.append((mem, distance_map[vid]))
            if len(results) >= limit:
                break

        return results

    async def _fts_search(
        self,
        query: str,
        user_id: str,
        limit: int = 20,
        memory_types: list[MemoryType] | None = None,
        scope: str | None = None,
        scope_strict: bool = False,
    ) -> list[tuple[Memory, float]]:
        """Full-text search via FTS5. Returns (memory, rank) pairs."""
        # Strip FTS5 special characters and split into individual words for OR matching
        safe_query = _re.sub(r'["\*\(\)\-\+\^~:]', " ", query)
        words = safe_query.split()
        if not words:
            return []
        fts_query = " ".join(words)  # unquoted words → FTS5 implicit OR matching

        try:
            # Type filter goes INSIDE the MATCH expression — memory_type is an
            # indexed FTS5 column, and column-filter syntax (`memory_type:(...)`)
            # is only valid there. The previous form appended it as bare SQL
            # after `MATCH ?`, which is a SQL parse error ("unrecognized token
            # ':'") — every typed recall's FTS leg silently died and recall
            # degraded to vector-only.
            if memory_types:
                type_values = " OR ".join(f'"{t.value}"' for t in memory_types)
                fts_query = f"content:({fts_query}) AND memory_type:({type_values})"

            # Scope filter: default matches the scope OR unscoped (universal);
            # scope_strict excludes unscoped (the isolated harness scope).
            scope_filter = ""
            scope_params: list = []
            if scope is not None:
                scope_filter = (
                    " AND m.scope = ?" if scope_strict
                    else " AND (m.scope = ? OR m.scope IS NULL)"
                )
                scope_params = [scope]
            elif ISOLATED_SCOPES:
                # General (unscoped) FTS must not return isolated-scope rows
                # (incl. harness:* sub-scopes).
                pred, scope_params = _isolation_sql("m.scope")
                scope_filter = f" AND {pred}"

            cursor = await self._conn.execute(
                "SELECT m.*, rank FROM memories_fts "
                "JOIN memories m ON m.rowid = memories_fts.rowid "
                "WHERE memories_fts MATCH ? "
                f"AND m.user_id = ? AND m.valid_until IS NULL{scope_filter} "
                "ORDER BY rank LIMIT ?",
                [fts_query, user_id, *scope_params, limit],
            )
            rows = await cursor.fetchall()
        except Exception:
            log.debug("fts_search_failed", query=query[:80], exc_info=True)
            return []

        results: list[tuple[Memory, float]] = []
        for row in rows:
            d = dict(row)
            rank = d.pop("rank", 0.0)
            mem = self._row_to_memory(d)
            results.append((mem, abs(rank)))  # FTS5 rank is negative (lower = better)

        return results

    @staticmethod
    def _rrf_merge(
        vec_results: list[tuple[Memory, float]],
        fts_results: list[tuple[Memory, float]],
        k: int = 60,
    ) -> list[tuple[Memory, float]]:
        """Reciprocal Rank Fusion — merge two ranked lists.

        RRF score = sum(1 / (k + rank)) for each list the item appears in.
        """
        return MemoryStore._rrf_merge_multi([vec_results, fts_results], k=k)

    @staticmethod
    def _rrf_merge_multi(
        ranked_lists: list[list[tuple[Memory, float]]],
        k: int = 60,
    ) -> list[tuple[Memory, float]]:
        """Reciprocal Rank Fusion — merge N ranked lists.

        RRF score = sum(1 / (k + rank)) for each list the item appears in.
        """
        scores: dict[str, float] = {}
        memories: dict[str, Memory] = {}

        for ranked_list in ranked_lists:
            for rank_idx, (mem, _) in enumerate(ranked_list):
                scores[mem.id] = scores.get(mem.id, 0) + 1 / (k + rank_idx + 1)
                memories[mem.id] = mem

        merged = [(memories[mid], score) for mid, score in scores.items()]
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    @staticmethod
    def _rerank_memories(
        query: str,
        scored: list[tuple[Memory, float]],
        limit: int,
    ) -> list[tuple[Memory, float]]:
        """Rerank top memory candidates with cross-encoder for precision.

        Only reranks when enabled in config. Falls back to original order on error.
        """
        from augmentum.config import settings

        if not settings.reranker_enabled or not scored:
            return scored

        try:
            from augmentum.memory.reranker import RerankService

            # Rerank the top candidates (more than limit to give reranker room)
            candidates = scored[:max(limit * 3, 15)]
            documents = [mem.content for mem, _ in candidates]
            ranked = RerankService.rerank(query, documents, top_k=None)

            # Rebuild scored list with reranker scores
            reranked: list[tuple[Memory, float]] = []
            for orig_idx, rerank_score in ranked:
                mem, _ = candidates[orig_idx]
                reranked.append((mem, rerank_score))
            return reranked
        except Exception:
            log.debug("memory_rerank_failed_using_rrf_order", exc_info=True)
            return scored

    @staticmethod
    def _recency_weight(updated_at: str) -> float:
        """Exponential decay weight based on age. Returns 0-1. (Legacy, kept for compat.)"""
        if not updated_at:
            return 0.5
        try:
            updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            now = datetime.now(UTC)
            age_days = (now - updated).total_seconds() / 86400
            return exp(-0.693 * age_days / RECENCY_HALF_LIFE_DAYS)
        except (ValueError, TypeError):
            return 0.5

    @staticmethod
    def _effective_importance(mem: Memory) -> float:
        """Living importance score: grows with access, decays without it.

        Base importance is the LLM's initial estimate (or default 0.6).
        Access boost rewards memories proven useful through natural retrieval.
        Time decay gently erodes importance of memories not accessed recently,
        making them eligible for compaction and less likely to surface.

        Returns 0.1–1.0.
        """
        base = mem.importance or 0.5
        access = mem.access_count or 0

        # Access boost: +0.02 per access, capped so it can't exceed 1.0
        access_boost = min(access * 0.02, 0.3)

        # Time decay: importance erodes if memory hasn't been accessed.
        # Half-life of 90 days — very gentle. A memory accessed last week
        # barely decays. A memory untouched for 6 months loses ~75%.
        decay = 1.0
        try:
            ts = mem.last_accessed_at or mem.updated_at or mem.created_at
            if ts:
                accessed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                days_since = max(0, (datetime.now(UTC) - accessed).total_seconds() / 86400)
                # exp(-0.693 * days / 90) → half-life of 90 days
                decay = max(0.3, exp(-0.693 * days_since / 90.0))
        except (ValueError, TypeError):
            decay = 0.7

        effective = (base + access_boost) * decay
        return max(0.1, min(1.0, effective))

    @staticmethod
    def _memory_strength(mem: Memory) -> float:
        """Spaced-repetition strength: memories proven through spaced access stay strong.

        Based on the spacing effect (Ebbinghaus, 1885): memories reinforced at
        intervals are stronger than those accessed in bursts. A memory accessed
        5 times over 2 weeks is stronger than one accessed 5 times in one hour.

        Returns 0.2–1.0.
        """
        access = mem.access_count or 0

        # Parse age for decay calculation
        try:
            ts = mem.last_accessed_at or mem.updated_at
            if not ts:
                return 0.3
            accessed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            hours_since = max(0, (datetime.now(UTC) - accessed).total_seconds() / 3600)
        except (ValueError, TypeError):
            hours_since = 24 * 7  # assume 1 week if unparseable

        if access == 0:
            # Never accessed — minimal strength, fast decay
            return max(0.2, 0.3 * (0.99 ** hours_since))

        if access == 1:
            # Single access — might be a fluke match, moderate decay
            return max(0.2, 0.5 * (0.995 ** hours_since))

        # Multiple accesses — compute spacing
        try:
            created = datetime.fromisoformat((mem.created_at or mem.updated_at).replace("Z", "+00:00"))
            days_alive = max(1, (datetime.now(UTC) - created).total_seconds() / 86400)
        except (ValueError, TypeError):
            days_alive = 7

        accesses_per_week = (access / days_alive) * 7

        # Well-spaced (< 3/week): strong plateau — proven consistently useful
        if accesses_per_week < 3:
            base_strength = 0.85
            decay_rate = 0.999  # very slow decay for proven memories
        # Moderate (3-10/week): solid but could be transient
        elif accesses_per_week < 10:
            base_strength = 0.7
            decay_rate = 0.997
        # Bursty (> 10/week): likely topical, lower plateau
        else:
            base_strength = 0.5
            decay_rate = 0.995

        return max(0.2, base_strength * (decay_rate ** hours_since))

    @staticmethod
    def _surprise_score(mem: Memory, total_retrievals: int) -> float:
        """How surprising is it to retrieve this memory right now?

        Three components combined (max of each → final surprise):
        1. Frequency surprise: rarely retrieved memories matching now
        2. Recency surprise: memories not accessed recently matching now
        3. Importance surprise: LOW-importance memories matching are more
           surprising than high-importance ones. A core identity fact
           surfacing is expected; a casual mention surfacing is a
           "you remembered that?" moment that builds trust.

        Returns 0.0–1.0.
        """
        if total_retrievals <= 0:
            return 0.0

        # Frequency: memories in 10%+ of retrievals are expected (surprise → 0)
        retrieval_rate = (mem.retrieval_count or 0) / total_retrievals
        freq_surprise = 1.0 - min(1.0, retrieval_rate * 10)

        # Recency: memories not accessed recently are MORE surprising when they match
        recency_surprise = 0.0
        try:
            ts = mem.last_accessed_at or mem.updated_at
            if ts:
                accessed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                hours_since = (datetime.now(UTC) - accessed).total_seconds() / 3600
                recency_surprise = min(1.0, hours_since / (24 * 7))  # maxes at 1 week
        except (ValueError, TypeError):
            pass

        # Importance: low-importance facts surfacing is more surprising.
        # importance=0.3 → surprise=0.7, importance=0.9 → surprise=0.1
        importance_surprise = max(0.0, 1.0 - (mem.importance or 0.5))

        return max(freq_surprise, recency_surprise, importance_surprise)

    async def _get_total_retrieval_count(self, user_id: str) -> int:
        """Get total retrieval operations for this user (for surprise scoring)."""
        try:
            cursor = await self._conn.execute(
                "SELECT SUM(COALESCE(retrieval_count, 0)) FROM memories "
                "WHERE user_id = ? AND valid_until IS NULL",
                (user_id,),
            )
            row = await cursor.fetchone()
            return max(1, row[0] or 0)
        except Exception:
            return 1  # prevent division by zero

    @staticmethod
    def _row_to_memory(row: dict) -> Memory:
        """Convert a database row to a Memory dataclass."""
        return Memory(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            importance=row.get("importance", 0.5),
            confidence=row.get("confidence", 0.8),
            session_id=row.get("session_id"),
            embedding=None,  # Don't load blob into memory by default
            valid_from=row.get("valid_from", ""),
            valid_until=row.get("valid_until"),
            superseded_by=row.get("superseded_by"),
            source_type=row.get("source_type"),
            source_context=row.get("source_context"),
            access_count=row.get("access_count", 0),
            last_accessed=row.get("last_accessed"),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
            scope=row.get("scope"),
            tier=row.get("tier", MemoryTier.ACTIVE),
            last_compacted_at=row.get("last_compacted_at"),
            provisional_expires_at=row.get("provisional_expires_at"),
            evidence=row.get("evidence", ""),
            retrieval_count=row.get("retrieval_count", 0),
            last_accessed_at=row.get("last_accessed_at"),
            source_memory_ids=row.get("source_memory_ids", "[]"),
        )

    async def _insert_and_supersede(
        self,
        old_id: str,
        content: str,
        embedding: list[float],
        memory_type: MemoryType,
        user_id: str,
        session_id: str | None,
        importance: float,
        confidence: float,
        source_type: SourceType | None,
        source_context: dict | None,
        scope: str | None = None,
    ) -> str:
        """Insert a new memory and mark the old one as superseded (atomic)."""
        memory_id = str(uuid.uuid4())
        blob = EmbeddingService.to_blob(embedding)
        now = datetime.now(UTC).isoformat()

        try:
            # Insert new
            await self._conn.execute(
                "INSERT INTO memories "
                "(id, user_id, session_id, content, memory_type, importance, confidence, "
                " embedding, valid_from, source_type, source_context, scope, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory_id, user_id, session_id, content, memory_type.value,
                    importance, confidence, blob, now,
                    source_type.value if source_type else None,
                    json.dumps(source_context) if source_context else None,
                    scope, now, now,
                ),
            )
            if self._vec_enabled:
                await self._conn.execute(
                    "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                    (memory_id, blob),
                )

            # Supersede old — scope to user_id so a UUID collision
            # (near-impossible) can't cross tenant boundaries.
            if user_id:
                await self._conn.execute(
                    "UPDATE memories SET valid_until = ?, superseded_by = ?, updated_at = ? "
                    "WHERE id = ? AND user_id = ? AND valid_until IS NULL",
                    (now, memory_id, now, old_id, user_id),
                )
            else:
                await self._conn.execute(
                    "UPDATE memories SET valid_until = ?, superseded_by = ?, updated_at = ? "
                    "WHERE id = ? AND valid_until IS NULL",
                    (now, memory_id, now, old_id),
                )
            await self._maybe_commit()
        except Exception:
            await self._maybe_rollback()
            raise

        return memory_id

    @staticmethod
    def _normalize_content(text: str) -> str:
        """Normalize memory content for text-based dedup comparison."""
        text = text.lower().strip()
        for contraction, expanded in [
            ("i'm", "i am"), ("i've", "i have"), ("i'll", "i will"),
            ("i'd", "i would"), ("don't", "do not"), ("doesn't", "does not"),
            ("can't", "cannot"), ("won't", "will not"), ("it's", "it is"),
            ("that's", "that is"), ("there's", "there is"),
        ]:
            text = text.replace(contraction, expanded)
        text = _re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    async def _text_dedup_check(
        self, content: str, user_id: str, scope: str | None = None,
        scope_strict: bool = False,
    ) -> str | None:
        """Fast text-normalized dedup: catches 'I'm X' vs 'I am X' without embeddings.

        Two scan passes so high-value old memories don't slip past the window:
        the recency pass (LIMIT 1000) catches the common case where a user
        re-asserts a fact within the same era; the importance pass catches
        cases where someone re-asserts a long-known core fact months later
        and the recency pass already aged out the original.

        Scope-isolated to match recall: a scoped write only dedups against its
        own scope (+ unscoped/universal); ``scope=None`` is unfiltered as before.
        """
        normalized = self._normalize_content(content)
        if len(normalized) < 5:
            return None
        iso_params: list = []
        if scope is None:
            # Unscoped write: dedup against the general pool only, never an
            # isolated scope's rows (keeps the isolation symmetric on writes).
            if ISOLATED_SCOPES:
                pred, iso_params = _isolation_sql("scope")
                scope_sql = f" AND {pred}"
            else:
                scope_sql = ""
        elif scope_strict:
            scope_sql = " AND scope = ?"
        else:
            scope_sql = " AND (scope = ? OR scope IS NULL)"
        for order_clause in ("ORDER BY created_at DESC", "ORDER BY importance DESC, confidence DESC"):
            params = [user_id, scope, *iso_params] if scope is not None else [user_id, *iso_params]
            cursor = await self._conn.execute(
                "SELECT id, content FROM memories WHERE user_id = ? AND valid_until IS NULL "
                f"{scope_sql} {order_clause} LIMIT 1000",
                params,
            )
            for row in await cursor.fetchall():
                if self._normalize_content(row["content"]) == normalized:
                    return row["id"]
        return None

    async def _update_existing(self, existing: Memory, new_content: str, embedding: list[float]) -> None:
        """Update an existing near-duplicate memory."""
        blob = EmbeddingService.to_blob(embedding)
        now = datetime.now(UTC).isoformat()
        if existing.user_id:
            await self._conn.execute(
                "UPDATE memories SET content = ?, embedding = ?, access_count = access_count + 1, "
                "updated_at = ? WHERE id = ? AND user_id = ?",
                (new_content, blob, now, existing.id, existing.user_id),
            )
        else:
            await self._conn.execute(
                "UPDATE memories SET content = ?, embedding = ?, access_count = access_count + 1, "
                "updated_at = ? WHERE id = ?",
                (new_content, blob, now, existing.id),
            )
        if self._vec_enabled:
            await self._conn.execute(
                "UPDATE memories_vec SET embedding = ? WHERE memory_id = ?",
                (blob, existing.id),
            )
        await self._maybe_commit()
        # Corroboration: a near-duplicate re-mention just bumped access_count
        # above — check promotion so an earned PROVISIONAL fact can climb (this
        # call was missing, so the bump never triggered the ladder). Gated.
        from augmentum.config import settings
        if settings.memory_earned_permanence and existing.user_id:
            await self._maybe_promote(existing.id, user_id=existing.user_id)
