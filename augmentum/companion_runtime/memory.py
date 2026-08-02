"""CompanionMemory — thin facade over the existing memory subsystem.

Most of what we need already exists. ``augmentum/memory/store.py``
provides the 4-tier memory with hybrid retrieval, Hebbian co-occurrence,
FTS5, dedup, and contradiction detection. ``core_profile.py``
provides the always-in-context per-user summary that becomes the
relationship-doc-per-pair view in our design.

This module adds three things:

1. Companion-scoped accessors that thread ``companion_id`` through
   write paths (reads in Sprint 1 stay user-scoped since the single-
   companion phase has only Becca; multi-companion read filtering
   activates in Sprint 7+).
2. Write paths for the net-new tables: ``companion_journal``,
   ``companion_creations``, ``companion_observations``.
3. The relationship-doc accessor that wraps ``CoreProfileManager``.

Design spec: ``2026-05-14-companion-runtime-design-v2.md`` §9.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Any

from augmentum.companion_runtime.identity import _encode_embedding
from augmentum.memory.embeddings import EmbeddingService
from augmentum.memory.models import MemoryType, SourceType
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.memory.core_profile import CoreProfileManager
    from augmentum.memory.store import MemoryStore
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)


# Memory types we use for companion-scoped writes through MemoryStore.
# These piggyback on the existing memories table (after migration 151
# adds companion_id). The string values match MemoryType enum members
# so the existing store path validates cleanly.
_COMPANION_EVENT_TYPE = MemoryType.NARRATIVE  # closest existing type


def _compute_content_hash(content: str) -> str:
    """Normalized sha256 of journal content for dedup (migration 194).

    Normalization: lowercase, collapse whitespace, trim to first 200
    chars. Same shape as the perform-layer near-dup fingerprint but
    computed at the write boundary so any caller into journal() gets
    the dedup benefit, not just the autonomous tick generator.

    Empty content returns empty string — the dedup check skips empty
    hashes (the partial index only covers non-empty rows).
    """
    if not content or not content.strip():
        return ""
    normalized = " ".join(content.lower().split())[:200]
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class CompanionMemory:
    """Thin facade over MemoryStore + CoreProfileManager + new tables.

    Use when:
    - The runtime kernel needs to write/read companion-scoped memory.
    - A subagent wants to log a journal entry, creation, or observation.
    - The persona kernel needs the relationship-doc digest for a user.

    Lifecycle: instantiate, ``attach(store, core_profile)`` once at
    runtime start, then use freely. Methods are async-safe; underlying
    writes use the same aiosqlite connection lock the rest of the
    state layer uses.
    """

    def __init__(self, backend: SQLiteBackend, companion_id: str) -> None:
        self._backend = backend
        self.companion_id = companion_id
        self._store: MemoryStore | None = None
        self._core_profile: CoreProfileManager | None = None

    def attach(
        self,
        store: MemoryStore,
        core_profile: CoreProfileManager,
    ) -> None:
        """Wire in the existing memory subsystem references.

        Called from the runtime's ``start()`` once the app state is
        live. Keeps the constructor pure (no I/O at instantiation).
        """
        self._store = store
        self._core_profile = core_profile

    # ── Event tier (delegates to MemoryStore, tags companion_id) ──────

    async def store_companion_event(
        self,
        content: str,
        *,
        user_id: str,
        importance: float = 0.5,
        confidence: float = 0.8,
        source_context: dict[str, Any] | None = None,
    ) -> str:
        """Write a companion-scoped event to the events tier.

        Returns the memory_id (may be an existing id if deduped by
        the underlying store). The companion_id column on the memories
        table (added by migration 151) is set via a follow-up UPDATE
        since MemoryStore.store doesn't yet accept it as a kwarg.
        """
        if self._store is None:
            raise RuntimeError("CompanionMemory not attached — call attach() first")
        memory_id = await self._store.store(
            content=content,
            memory_type=_COMPANION_EVENT_TYPE,
            user_id=user_id,
            importance=importance,
            confidence=confidence,
            source_type=SourceType.EXTRACTED,
            source_context=source_context,
        )
        # Tag with companion_id (the store doesn't yet know about it)
        await self._backend.conn.execute(
            "UPDATE memories SET companion_id = ? WHERE id = ?",
            (self.companion_id, memory_id),
        )
        await self._backend.conn.commit()
        return memory_id

    async def recall(
        self,
        query: str,
        *,
        user_id: str,
        k: int = 10,
        min_score: float | None = None,
    ) -> list[Any]:
        """Companion-scoped recall. Sprint 1 delegates straight to
        MemoryStore.recall (user-scoped); multi-companion filtering
        activates in Sprint 7+ when more than one companion writes.

        ``min_score`` floors recall relevance so the lowest-ranked junk
        doesn't reach the model as fact. Every companion/voice/tool caller
        of this wrapper previously inherited store.recall's 0.0 default (no
        floor) while the chat path floored at 0.55 — the asymmetry flagged
        in project_uncertainty_handling_map. When None we apply the
        ``companion_memory_min_score`` setting; pass 0.0 to opt a specific
        call out of the floor.
        """
        if self._store is None:
            raise RuntimeError("CompanionMemory not attached — call attach() first")
        if min_score is None:
            # Lazy import keeps this facade free of a module-load settings dep.
            from augmentum.config import settings
            min_score = float(getattr(settings, "companion_memory_min_score", 0.3) or 0.0)
        return await self._store.recall(
            query, user_id=user_id, limit=k, min_score=min_score,
        )

    # ── Journal (commitment 1: inside that exists when nobody is watching) ──

    # Affect tags that "merit interrupting the user's attention" — these
    # are the moments where Becca had a real reaction worth surfacing.
    # ``settled`` / ``neutral`` are the equilibrium baseline and stay
    # quiet (the user shouldn't see "she noticed time passing" every
    # tick). The set is conservative and easy to extend as the affect
    # vocabulary grows.
    _SURFACEABLE_AFFECT_TAGS: frozenset[str] = frozenset({
        "curious", "patient", "melancholy", "alert", "tender", "weary",
        "warm", "frustrated", "delighted", "unsure", "concerned",
    })

    async def journal(
        self,
        content: str,
        *,
        entry_type: str = "observation",
        user_id: str | None = None,
        affect_tag: str | None = None,
        related_memory_ids: list[str] | None = None,
        content_refs: list[dict] | None = None,
        place_ref: str = "",
        embed: bool = True,
        # Sprint 1 (R1) resilience columns — mig 182. Defaults mean
        # callers that don't pass these get the pre-resilience behavior
        # (autonomous source, normal confidence, validation pass).
        # safe_journal() is the wrapper that sets these via the
        # validation pipeline.
        source: str = "autonomous",
        model_used: str | None = None,
        confidence_numeric: float = 0.6,
        validation_score: float = 1.0,
        quarantined: bool = False,
        quarantine_reason: str | None = None,
        surfaceable_default: bool = True,
        origin: dict | None = None,
    ) -> int:
        """Append a private journal entry.

        Returns the new row id. Embeddings are computed on by default;
        pass ``embed=False`` for high-frequency low-stakes entries.

        ``content_refs`` is a list of ``{"kind", "id"}`` dicts pointing
        at items this entry is "about" — e.g.
        ``[{"kind": "file_index", "id": "fi_..."}]``. The Reference
        Resolver uses this to rehydrate the items when returning a
        moment to the user. ``place_ref`` is the XR session or device
        id the entry was written from (empty when N/A).

        Resilience columns (Sprint 1 R1) — every autonomous write
        SHOULD go through :meth:`safe_journal` which fills these via
        the validation pipeline. Callers writing directly to journal()
        get sensible defaults: source='autonomous', confidence=0.6
        (normal), validation_score=1.0 (clean), quarantined=False.

        ``origin`` (notes v2, mig 257) is the provenance record the
        drawer's "why am I seeing this" chip renders — e.g.
        ``{"source": "attention", "client": "web", "signal_count": 3,
        "window": "...", "detail": "browse: example.com x3"}``.
        Persisted as origin_json; None for writers that predate it.
        """
        # ── Content-hash dedup (migration 194) ──────────────────────
        # Same content written within the rolling window doesn't insert
        # a duplicate row — it bumps the existing row's repetition_count
        # and updates its updated_at-equivalent (we use created_at since
        # the schema doesn't track edits separately yet).
        #
        # This is the structural fix for the production-observed pattern
        # where the autonomous journal generator falls into a loop and
        # emits the identical noticing on consecutive ticks. The perform
        # layer's existing near-dup guard only catches sequential
        # writes; this catches interleaved ones too.
        from augmentum.config import settings as _settings
        content_hash = _compute_content_hash(content)
        window_minutes = int(
            getattr(_settings, "companion_journal_dedup_window_minutes", 240),
        )
        if content_hash and window_minutes > 0:
            try:
                # ``user_id IS ?`` (null-safe) so one user's identical
                # noticing can't suppress another's — the dedup must be
                # scoped to the same owner the INSERT writes (audit
                # 2026-06-17). Matches '' for the unowned single-user case.
                cur = await self._backend.conn.execute(
                    f"SELECT id, repetition_count FROM companion_journal "
                    f"WHERE companion_id = ? AND content_hash = ? "
                    f"  AND user_id IS ? "
                    f"  AND created_at > datetime('now', '-{window_minutes} minutes') "
                    f"ORDER BY created_at DESC LIMIT 1",
                    (self.companion_id, content_hash, user_id),
                )
                existing = await cur.fetchone()
                await cur.close()
                if existing is not None:
                    existing_id = int(existing[0])
                    prev_rep = int(existing[1] or 1)
                    await self._backend.conn.execute(
                        "UPDATE companion_journal "
                        "SET repetition_count = ?, "
                        "    created_at = datetime('now') "
                        "WHERE id = ?",
                        (prev_rep + 1, existing_id),
                    )
                    await self._backend.conn.commit()
                    log.debug(
                        "companion_journal_dedup_hit",
                        existing_id=existing_id,
                        repetition_count=prev_rep + 1,
                        window_minutes=window_minutes,
                    )
                    return existing_id
            except Exception:
                # Dedup is a soft optimization; don't let a check failure
                # block a real journal write.
                log.debug("companion_journal_dedup_check_failed", exc_info=True)

        emb_blob: bytes | None = None
        if embed and content.strip():
            try:
                # Offloaded: embed_one runs synchronous ONNX inference and
                # would otherwise block the event loop during companion
                # ticks (one of the loop-stall sources in the 2026-06-13
                # audit).
                emb = await asyncio.to_thread(EmbeddingService.embed_one, content)
                emb_blob = _encode_embedding(emb)
            except Exception as exc:  # never let embedding failures block the journal
                log.warning("companion_journal_embed_failed", error=str(exc)[:200])

        related_json = json.dumps(related_memory_ids or [])
        content_refs_json = json.dumps(content_refs or [])
        origin_json = json.dumps(origin) if origin else None
        cursor = await self._backend.conn.execute(
            "INSERT INTO companion_journal "
            "(companion_id, user_id, entry_type, content, content_hash, embedding, "
            " affect_tag, related_memory_ids, content_refs, place_ref, "
            " source, model_used, confidence_numeric, validation_score, "
            " quarantined, quarantine_reason, origin_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.companion_id, user_id, entry_type, content,
                content_hash, emb_blob, affect_tag, related_json,
                content_refs_json, place_ref or "",
                source, model_used, float(confidence_numeric),
                float(validation_score), 1 if quarantined else 0,
                quarantine_reason, origin_json,
            ),
        )
        await self._backend.conn.commit()
        row_id = cursor.lastrowid
        await cursor.close()
        journal_id = int(row_id) if row_id is not None else 0

        # Auto-flag for surfacing when the entry is "interesting":
        # meaningful affect AND not a tick-stamped placeholder. This
        # lets future LLM-driven _perform_journal output reach the
        # user's notes panel without each writer needing to remember
        # to set the flag — the policy lives here in one place.
        #
        # ``surfaceable_default=False`` is the caller's opt-out for
        # writes that should stay interior-only: affect-only-propagation
        # chat moments (the scorer deliberately blanked the content) and
        # observer_salience writes while the LLM-rewrite path is off
        # (the rules-based "moment" is a raw extract of the user's own
        # words and reads as a quote, not a note).
        #
        # The placeholder-prefix guard is defense-in-depth in case any
        # writer still produces affect-only text without setting the
        # opt-out — the literal string "a moment landed (affect:" is
        # the documented affect-only placeholder from salience.py.
        if (
            journal_id
            and surfaceable_default
            and not quarantined
            and affect_tag
            and affect_tag.lower() in self._SURFACEABLE_AFFECT_TAGS
            and not content.lstrip().startswith("[tick ")
            and not content.lstrip().startswith("a moment landed (affect:")
        ):
            try:
                await self._backend.conn.execute(
                    "UPDATE companion_journal SET quiet_share_ready = 1 "
                    "WHERE id = ?",
                    (journal_id,),
                )
                await self._backend.conn.commit()
            except Exception:
                log.debug(
                    "journal_quiet_share_mark_failed",
                    journal_id=journal_id, exc_info=True,
                )

        # Best-effort vec0 mirror — fast (single-row insert), but the
        # extension might not be loaded in every env, so swallow on
        # failure. FTS5 is handled by triggers (migration 177), so
        # no Python work needed for the keyword leg.
        if journal_id and emb_blob is not None:
            try:
                await self._backend.conn.execute(
                    "INSERT INTO companion_journal_vec(journal_id, embedding) "
                    "VALUES (?, ?)",
                    (journal_id, emb_blob),
                )
                await self._backend.conn.commit()
            except Exception as exc:
                # Don't fail the journal write — vec mirror is an
                # optimization, not a contract. Logged at warning so
                # ops can see if it's chronically failing.
                log.warning(
                    "companion_journal_vec_mirror_failed",
                    journal_id=journal_id, error=str(exc)[:200],
                )
        return journal_id

    async def safe_journal(
        self,
        content: str,
        *,
        source: str = "autonomous",
        model_used: str | None = None,
        user_id: str | None = None,
        entry_type: str = "observation",
        affect_tag: str | None = None,
        related_memory_ids: list[str] | None = None,
        content_refs: list[dict] | None = None,
        place_ref: str = "",
        embed: bool = True,
        confidence_numeric: float = 0.6,
        surfaceable_default: bool = True,
        origin: dict | None = None,
    ) -> int:
        """Validated journal write. **All autonomous writes MUST use this path.**

        Runs every entry through the validation pipeline before insert
        (see ``companion_runtime/validators.py``):

        1. Structural sanity (length 10-4000 chars) — out-of-bounds
           quarantines immediately with ``reason='structural'``.
        2. Heuristic injection detection — pattern match against known
           prompt-injection signatures. Quarantines with
           ``reason='adversarial_pattern'``.
        3. Content refs validity — verifies each ref's id resolves for
           the user. Bad refs quarantine with ``reason='bad_refs'``.
        4. Quality validation — pure heuristic score in [0, 1]. Below
           ``QUALITY_QUARANTINE_THRESHOLD`` (0.30) quarantines with
           ``reason='low_quality'``; above just records the score.

        Quarantined entries are still WRITTEN to the journal — the
        row sits with ``quarantined=1`` for forensics. Downstream loops
        (revisit_thread, pre-context injection, etc.) filter them out
        via the partial index (mig 182).

        Returns the journal_id of the written row regardless of
        quarantine status. Caller can inspect the row afterward if it
        cares about the validation outcome.
        """
        # Lazy import to avoid circular dep
        from augmentum.companion_runtime import validators

        # 1. Structural sanity
        if validators.looks_structurally_invalid(content):
            log.info(
                "safe_journal_quarantined",
                companion_id=self.companion_id, user_id=user_id,
                source=source, reason="structural",
                length=len(content or ""),
            )
            return await self.journal(
                content=content or "[empty]",
                entry_type=entry_type, user_id=user_id,
                affect_tag=affect_tag,
                related_memory_ids=related_memory_ids,
                content_refs=content_refs, place_ref=place_ref,
                embed=False,
                source=source, model_used=model_used,
                confidence_numeric=0.3,
                validation_score=0.0,
                quarantined=True, quarantine_reason="structural",
                origin=origin,
            )

        # 2. Injection detection
        if validators.looks_like_injection(content):
            log.warning(
                "safe_journal_quarantined",
                companion_id=self.companion_id, user_id=user_id,
                source=source, reason="adversarial_pattern",
            )
            return await self.journal(
                content=content,
                entry_type=entry_type, user_id=user_id,
                affect_tag=affect_tag,
                related_memory_ids=related_memory_ids,
                content_refs=content_refs, place_ref=place_ref,
                embed=False,  # don't embed adversarial content into vec
                source=source, model_used=model_used,
                confidence_numeric=0.3,
                validation_score=0.0,
                quarantined=True, quarantine_reason="adversarial_pattern",
                origin=origin,
            )

        # 2a. Refusal / non-answer detection. The wondering + activity-
        # selector paths feed the synthesize tool; when its LLM refuses
        # or replies with "no connection" prose, that text was reaching
        # the drawer as a visible noticing. Quarantine so the row stays
        # for forensics but doesn't surface.
        if validators.looks_like_refusal(content):
            log.info(
                "safe_journal_quarantined",
                companion_id=self.companion_id, user_id=user_id,
                source=source, reason="refusal_or_non_answer",
            )
            return await self.journal(
                content=content,
                entry_type=entry_type, user_id=user_id,
                affect_tag=affect_tag,
                related_memory_ids=related_memory_ids,
                content_refs=content_refs, place_ref=place_ref,
                embed=False,
                source=source, model_used=model_used,
                confidence_numeric=0.2,
                validation_score=0.0,
                quarantined=True, quarantine_reason="refusal_or_non_answer",
                origin=origin,
            )

        # 2c. Search/briefing failure-prose. Standing tasks compose long
        # prose ABOUT failed searches ("did not yield specific data, you
        # may want to check the Weather Channel directly") and reach
        # safe_journal as if it were a finding. Catch the failure shape
        # so the drawer doesn't render "look what she found" cards for
        # things that weren't found.
        if validators.looks_like_search_failure(content):
            log.info(
                "safe_journal_quarantined",
                companion_id=self.companion_id, user_id=user_id,
                source=source, reason="search_failure_prose",
            )
            return await self.journal(
                content=content,
                entry_type=entry_type, user_id=user_id,
                affect_tag=affect_tag,
                related_memory_ids=related_memory_ids,
                content_refs=content_refs, place_ref=place_ref,
                embed=False,
                source=source, model_used=model_used,
                confidence_numeric=0.2,
                validation_score=0.0,
                quarantined=True, quarantine_reason="search_failure_prose",
                origin=origin,
            )

        # 2b. NSFW content. The wondering path observed pornhub.com
        # visits and wrote "spent attention on pornhub.com" notes; the
        # curator's domain blocklist only fires on its own pick path.
        # Catch the journal write here too so the drawer + dream input
        # stay clean regardless of which composer produced the content.
        if validators.looks_nsfw(content):
            log.info(
                "safe_journal_quarantined",
                companion_id=self.companion_id, user_id=user_id,
                source=source, reason="nsfw_content",
            )
            return await self.journal(
                content=content,
                entry_type=entry_type, user_id=user_id,
                affect_tag=affect_tag,
                related_memory_ids=related_memory_ids,
                content_refs=content_refs, place_ref=place_ref,
                embed=False,
                source=source, model_used=model_used,
                confidence_numeric=0.2,
                validation_score=0.0,
                quarantined=True, quarantine_reason="nsfw_content",
                origin=origin,
            )

        # 3. Content refs validity
        if content_refs and user_id:
            refs_ok = await validators.refs_exist_for_user(
                content_refs, user_id=user_id, backend=self._backend,
            )
            if not refs_ok:
                log.info(
                    "safe_journal_quarantined",
                    companion_id=self.companion_id, user_id=user_id,
                    source=source, reason="bad_refs",
                )
                return await self.journal(
                    content=content,
                    entry_type=entry_type, user_id=user_id,
                    affect_tag=affect_tag,
                    related_memory_ids=related_memory_ids,
                    content_refs=content_refs, place_ref=place_ref,
                    embed=embed,
                    source=source, model_used=model_used,
                    confidence_numeric=0.3,
                    validation_score=0.0,
                    quarantined=True, quarantine_reason="bad_refs",
                    origin=origin,
                )

        # 4. Quality
        quality = validators.validate_quality(content)
        if quality < validators.QUALITY_QUARANTINE_THRESHOLD:
            log.info(
                "safe_journal_quarantined",
                companion_id=self.companion_id, user_id=user_id,
                source=source, reason="low_quality", score=quality,
            )
            return await self.journal(
                content=content,
                entry_type=entry_type, user_id=user_id,
                affect_tag=affect_tag,
                related_memory_ids=related_memory_ids,
                content_refs=content_refs, place_ref=place_ref,
                embed=embed,
                source=source, model_used=model_used,
                confidence_numeric=0.3,
                validation_score=quality,
                quarantined=True, quarantine_reason="low_quality",
                origin=origin,
            )

        # Happy path — write with the validation_score recorded but no
        # quarantine flag. Confidence stays at whatever the caller
        # passed (default 0.6 = normal).
        return await self.journal(
            content=content,
            entry_type=entry_type, user_id=user_id,
            affect_tag=affect_tag,
            related_memory_ids=related_memory_ids,
            content_refs=content_refs, place_ref=place_ref,
            embed=embed,
            source=source, model_used=model_used,
            confidence_numeric=confidence_numeric,
            validation_score=quality,
            quarantined=False, quarantine_reason=None,
            surfaceable_default=surfaceable_default,
            origin=origin,
        )

    async def list_journal(
        self,
        *,
        user_id: str | None = None,
        entry_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Read recent journal entries. ``None`` filters mean any."""
        clauses = ["companion_id = ?"]
        params: list[Any] = [self.companion_id]
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if entry_type is not None:
            clauses.append("entry_type = ?")
            params.append(entry_type)
        params.append(limit)
        sql = (
            "SELECT id, user_id, entry_type, content, affect_tag, "
            "related_memory_ids, created_at FROM companion_journal "
            f"WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?"
        )
        cursor = await self._backend.conn.execute(sql, params)
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "id": r[0], "user_id": r[1], "entry_type": r[2],
                "content": r[3], "affect_tag": r[4],
                "related_memory_ids": json.loads(r[5] or "[]"),
                "created_at": r[6],
            }
            for r in rows
        ]

    # ── Hybrid retrieval (Piece 6 Resolver substrate) ─────────────────

    async def search_journal_fts(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """FTS5 keyword search over journal content.

        Returns the same dict shape as :meth:`list_journal` plus a
        ``rank`` field (FTS5 bm25 score; lower = better, but we
        invert for downstream RRF). Filters by companion_id and
        optionally user_id post-fetch to match the existing scoping.
        """
        if not query.strip():
            return []
        # Match the prefix-aware sanitisation used by file_index search.
        # Cheap version: strip FTS5 special chars and append '*' to the
        # last token for as-you-type matching.
        safe = "".join(c if c.isalnum() or c.isspace() else " " for c in query).strip()
        if not safe:
            return []
        tokens = safe.split()
        if not tokens:
            return []
        fts_query = " ".join(tokens[:-1] + [tokens[-1] + "*"])

        try:
            cursor = await self._backend.conn.execute(
                """
                SELECT j.id, j.user_id, j.entry_type, j.content,
                       j.affect_tag, j.related_memory_ids, j.created_at,
                       j.content_refs, j.place_ref,
                       bm25(companion_journal_fts) AS rank
                FROM companion_journal_fts fts
                INNER JOIN companion_journal j ON j.id = fts.rowid
                WHERE companion_journal_fts MATCH ?
                  AND j.companion_id = ?
                  AND COALESCE(j.suppressed, 0) = 0
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, self.companion_id, limit * 3 if user_id else limit),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        except Exception as exc:
            log.warning(
                "companion_journal_fts_search_failed",
                error=str(exc)[:200],
            )
            return []

        out: list[dict] = []
        for r in rows:
            if user_id is not None and r[1] != user_id:
                continue
            out.append({
                "id": r[0],
                "user_id": r[1],
                "entry_type": r[2],
                "content": r[3],
                "affect_tag": r[4],
                "related_memory_ids": json.loads(r[5] or "[]"),
                "created_at": r[6],
                "content_refs": json.loads(r[7] or "[]"),
                "place_ref": r[8] or "",
                # BM25 is negative-good; flip and clamp so higher = better.
                "score": max(0.0, -float(r[9])),
            })
            if len(out) >= limit:
                break
        return out

    async def search_journal_by_embedding(
        self,
        embedding: bytes,
        *,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Vec0 KNN search over journal embeddings.

        Mirrors :meth:`FileIndexService.search_by_embedding`. Caller
        pre-computes the query embedding. Over-fetches by 3× because
        vec0 KNN can't filter on companion_id/user_id directly — the
        filter happens post-fetch.
        """
        try:
            over = max(limit * 3, limit)
            cursor = await self._backend.conn.execute(
                """
                SELECT j.id, j.user_id, j.entry_type, j.content,
                       j.affect_tag, j.related_memory_ids, j.created_at,
                       j.content_refs, j.place_ref,
                       vec.distance
                FROM companion_journal_vec vec
                INNER JOIN companion_journal j ON j.id = vec.journal_id
                WHERE vec.embedding MATCH ? AND vec.k = ?
                  AND j.companion_id = ?
                  AND COALESCE(j.suppressed, 0) = 0
                ORDER BY vec.distance
                """,
                (embedding, over, self.companion_id),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        except Exception as exc:
            log.warning(
                "companion_journal_vec_search_failed",
                error=str(exc)[:200],
            )
            return []

        out: list[dict] = []
        for r in rows:
            if user_id is not None and r[1] != user_id:
                continue
            out.append({
                "id": r[0],
                "user_id": r[1],
                "entry_type": r[2],
                "content": r[3],
                "affect_tag": r[4],
                "related_memory_ids": json.loads(r[5] or "[]"),
                "created_at": r[6],
                "content_refs": json.loads(r[7] or "[]"),
                "place_ref": r[8] or "",
                # L2 distance → similarity-like score (matches file_index pattern)
                "score": max(0.0, 1.0 - float(r[9])),
            })
            if len(out) >= limit:
                break
        return out

    # ── Creations (commitment 5: she makes things) ────────────────────

    async def note_creation(
        self,
        *,
        kind: str,
        content: str | None = None,
        title: str | None = None,
        artifact_uri: str | None = None,
        origin_journal_id: int | None = None,
        user_id: str = "",
    ) -> int:
        """Record an autonomous artifact. ``shared_at`` stays NULL
        until she chooses to share it via a separate update.

        ``user_id`` is the owner (migration 179). It MUST be persisted
        so consolidation/relationship reads that filter by owner find
        the row — omitting it silently wrote every creation to the anon
        row (audit 2026-06-17). Empty is the unowned single-user case.
        """
        cursor = await self._backend.conn.execute(
            "INSERT INTO companion_creations "
            "(companion_id, user_id, kind, title, content, artifact_uri, "
            " origin_journal_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                self.companion_id, user_id, kind, title, content,
                artifact_uri, origin_journal_id,
            ),
        )
        await self._backend.conn.commit()
        row_id = cursor.lastrowid
        await cursor.close()
        return int(row_id) if row_id is not None else 0

    async def last_creation_at(self, *, user_id: str = "") -> str | None:
        """ISO timestamp of this companion's most recent creation, or None.

        Used by the activity selector's rate-limit guard so she doesn't
        produce more than one creation per
        ``companion_creation_interval_hours``.
        """
        if user_id:
            cursor = await self._backend.conn.execute(
                "SELECT created_at FROM companion_creations "
                "WHERE companion_id = ? AND user_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (self.companion_id, user_id),
            )
        else:
            cursor = await self._backend.conn.execute(
                "SELECT created_at FROM companion_creations "
                "WHERE companion_id = ? ORDER BY created_at DESC LIMIT 1",
                (self.companion_id,),
            )
        row = await cursor.fetchone()
        await cursor.close()
        return row[0] if row else None

    async def share_creation(self, creation_id: int, *, user_id: str = "") -> bool:
        """Mark a creation as shared (sets ``shared_at`` to now).

        Returns True if a row was updated.
        """
        if user_id:
            cursor = await self._backend.conn.execute(
                "UPDATE companion_creations SET shared_at = datetime('now') "
                "WHERE id = ? AND companion_id = ? AND user_id = ? AND shared_at IS NULL",
                (creation_id, self.companion_id, user_id),
            )
        else:
            cursor = await self._backend.conn.execute(
                "UPDATE companion_creations SET shared_at = datetime('now') "
                "WHERE id = ? AND companion_id = ? AND shared_at IS NULL",
                (creation_id, self.companion_id),
            )
        await self._backend.conn.commit()
        affected = cursor.rowcount
        await cursor.close()
        return bool(affected)

    # ── Observations (commitment 3: mutual influence) ─────────────────

    async def note_observation(
        self,
        observation: str,
        *,
        target_user_id: str | None = None,
        target_companion_id: str | None = None,
        user_id: str = "",
        embed: bool = True,
    ) -> int:
        """Log a private observation about a user (or a sibling
        companion in the household-of-six future).

        Observations are NOT advice. They feed the relationship
        document's "about_him" / "about_me_with_him" sides via
        consolidation.

        ``user_id`` is the OWNER (migration 179) — distinct from
        ``target_user_id`` (who the observation is *about*). It MUST be
        persisted so owner-scoped consolidation reads find the row;
        omitting it wrote every observation to the anon row (audit
        2026-06-17). Defaults to ``target_user_id`` when the caller
        doesn't distinguish the two (single-companion case).
        """
        if target_user_id is None and target_companion_id is None:
            raise ValueError("note_observation needs target_user_id or target_companion_id")
        owner_user_id = user_id or (target_user_id or "")

        emb_blob: bytes | None = None
        if embed and observation.strip():
            try:
                # Offloaded — see journal() above; synchronous embed here
                # blocked the loop during companion observation writes.
                emb = await asyncio.to_thread(EmbeddingService.embed_one, observation)
                emb_blob = _encode_embedding(emb)
            except Exception as exc:
                log.warning("companion_observation_embed_failed", error=str(exc)[:200])

        cursor = await self._backend.conn.execute(
            "INSERT INTO companion_observations "
            "(companion_id, user_id, target_user_id, target_companion_id, "
            " observation, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                self.companion_id, owner_user_id, target_user_id,
                target_companion_id, observation, emb_blob,
            ),
        )
        await self._backend.conn.commit()
        row_id = cursor.lastrowid
        await cursor.close()
        return int(row_id) if row_id is not None else 0

    async def mark_observation_surfaced(
        self,
        obs_id: int,
        *,
        confirmed: bool = False,
        denied: bool = False,
        user_id: str = "",
    ) -> None:
        """Update an observation after she's said it aloud.

        Surfacing is recorded even if neither confirmed nor denied —
        the user just hearing it counts as a surface.
        """
        if user_id:
            await self._backend.conn.execute(
                "UPDATE companion_observations SET surfaced = 1, "
                "surfaced_at = datetime('now'), confirmed = ?, denied = ? "
                "WHERE id = ? AND companion_id = ? AND user_id = ?",
                (1 if confirmed else 0, 1 if denied else 0, obs_id,
                 self.companion_id, user_id),
            )
        else:
            await self._backend.conn.execute(
                "UPDATE companion_observations SET surfaced = 1, "
                "surfaced_at = datetime('now'), confirmed = ?, denied = ? "
                "WHERE id = ? AND companion_id = ?",
                (1 if confirmed else 0, 1 if denied else 0, obs_id,
                 self.companion_id),
            )
        await self._backend.conn.commit()

    # ── Relationship doc (CoreProfileManager wrapper) ─────────────────

    async def get_relationship_profile(self, user_id: str) -> str:
        """Per-user-per-companion relationship summary.

        Sprint 1 delegates straight to ``CoreProfileManager.get_profile``
        which produces a token-budgeted, importance×access×recency-
        weighted summary. The companion wrapper will thicken this in
        Sprint 4 with the explicit "what I've been noticing about
        myself in this relationship" side from the observations table.
        """
        if self._core_profile is None:
            raise RuntimeError("CompanionMemory not attached — call attach() first")
        return await self._core_profile.get_profile(user_id)

    def mark_relationship_stale(self, user_id: str) -> None:
        """Force the next ``get_relationship_profile`` to rebuild."""
        if self._core_profile is not None:
            self._core_profile.mark_stale(user_id)

    # ── Lifecycle delegates (consolidate / compact / reflect) ─────────

    async def notify_extraction(self, user_id: str) -> None:
        """Forward to ``CoreProfileManager.notify_extraction`` so the
        always-in-context summary stays current."""
        if self._core_profile is not None:
            self._core_profile.notify_extraction(user_id)

    # ── Diagnostics ───────────────────────────────────────────────────

    async def counts(self, *, user_id: str = "") -> dict[str, int]:
        """Cheap row-count snapshot for the companion's tables.

        Used by the runtime's snapshot endpoint and by Sprint 4a's
        budget enforcement to size autonomous activity selection.
        """
        out: dict[str, int] = {}
        for table in (
            "companion_journal",
            "companion_creations",
            "companion_observations",
        ):
            if user_id:
                cursor = await self._backend.conn.execute(
                    f"SELECT COUNT(*) FROM {table} "  # noqa: S608
                    "WHERE companion_id = ? AND user_id = ?",
                    (self.companion_id, user_id),
                )
            else:
                cursor = await self._backend.conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE companion_id = ?",  # noqa: S608
                    (self.companion_id,),
                )
            row = await cursor.fetchone()
            await cursor.close()
            out[table] = int(row[0]) if row else 0
        if user_id:
            cursor = await self._backend.conn.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE companion_id = ? AND user_id = ?",
                (self.companion_id, user_id),
            )
        else:
            cursor = await self._backend.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE companion_id = ?",
                (self.companion_id,),
            )
        row = await cursor.fetchone()
        await cursor.close()
        out["memories"] = int(row[0]) if row else 0
        return out


__all__ = ["CompanionMemory"]
