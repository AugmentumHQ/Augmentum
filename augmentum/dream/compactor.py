"""Background dream compaction — semantic dedup + cluster summarization.

Mirrors :class:`augmentum.memory.compactor.MemoryCompactor` for the dream
subsystem. Two phases per cycle:

1. **Pair-merge** — pairwise similarity scan over active entries, LLM-merge
   any pair above ``dream_dedup_threshold``. Soft-deletes the dropped entry
   via ``expires_at`` (matches journal's existing soft-delete semantics so
   read paths transparently exclude it).
2. **Cluster-summarize** — greedy clustering by ``dream_cluster_threshold``,
   clusters of ``dream_cluster_min_size``+ get LLM-summarized into a single
   new entry (with ``source_memories`` union'd from the cluster), originals
   soft-deleted.

Key differences from the memory compactor:

* Reuses **stored embeddings** from ``dream_entries.embedding`` instead of
  re-embedding every cycle — major efficiency win since dream entries are
  longer than memory facts and embedding cost adds up.
* Uses ``expires_at`` soft-delete (already proven by ``compact_journal``)
  rather than the memory store's ``forget()``.
* Operates strictly per-user — the background loop iterates the active
  users table same way the memory compactor does, calls compact(user_id)
  per user, every sub-call passes user_id so cross-tenant merges are
  physically impossible at the SQL level.
* Time-trim is gated behind ``dream_time_trim_count_threshold`` — small
  journals rely entirely on semantic compaction; only larger journals
  fall back to age-based pruning. Prevents a curated journal from losing
  content just because it's old.

LLM model resolution mirrors memory's role chain: tries
``settings.dream_compaction_model`` override first, falls through to the
``"utility"`` role chain. Same prompts as memory's compactor (merge pair
/ summarize cluster) but the prompts are dream-flavored — they ask for
emotional essence preservation rather than fact-style specifics.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.dream.models import DreamEntry, DreamEntryType
from augmentum.utils.logging import get_logger
from augmentum.utils.vector import cosine_similarity, parse_merged_response

if TYPE_CHECKING:
    from augmentum.dream.journal import DreamJournal
    from augmentum.models.base import ModelBackend

log = get_logger(__name__)


# Dream-flavored prompts — preserve voice/essence rather than fact-specifics.
# Compare to memory/compactor.py which says "preserve specifics (names,
# numbers, tools, preferences)" — that's right for facts but wrong for
# reflective prose. Here we want the consolidated entry to read like a
# single thought, not a stitched summary.
_MERGE_SYSTEM = """\
You are merging two related reflections from an AI's private journal into one consolidated reflection. Rules:
- Preserve the emotional essence and voice of both originals.
- Combine into a single self-contained reflection, written in the same first-person reflective style.
- Do NOT add observations not present in either original.
- Drop near-duplicate phrasing; keep distinct insights.
- Return valid JSON: {"merged": "...", "importance": 0.7}
"""

_CLUSTER_SUMMARY_SYSTEM = """\
You are consolidating a cluster of related reflections from an AI's private journal into a single broader reflection. Rules:
- Capture the recurring theme without losing specific details that gave each entry its weight.
- Write as one flowing reflection in first-person, the same voice as the originals.
- Do NOT introduce observations or sentiments not present in the cluster.
- Aim for breadth over completeness — the new entry should feel like a general impression formed from many specific moments, not a list.
- Return valid JSON: {"summary": "...", "importance": 0.7}
"""


class DreamCompactor:
    """Periodic background compaction of the dream journal.

    See module docstring for design rationale. Settings (all admin-global,
    not per-user) controlling thresholds/intervals live in
    ``augmentum.config.Settings.dream_compaction_*``.
    """

    def __init__(
        self,
        journal: DreamJournal,
        registry: object | None = None,
        settings_store: object | None = None,
        app_state: object | None = None,
    ) -> None:
        self._journal = journal
        self._registry = registry
        self._settings_store = settings_store
        self._app_state = app_state
        self._task: asyncio.Task | None = None
        self._resolved_backend: ModelBackend | None = None
        self._resolved_model: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background compaction loop. Idempotent."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())
        self._task.add_done_callback(self._on_task_done)
        log.info(
            "dream_compactor_started",
            interval_hours=settings.dream_compaction_interval_hours,
        )

    def _on_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.error("dream_compactor_crashed", error=str(exc), exc_info=exc)
            self._task = None

    async def stop(self) -> None:
        """Stop the background loop. Idempotent."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
            log.info("dream_compactor_stopped")

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    async def _resolve_backend(self) -> ModelBackend | None:
        """Resolve the LLM backend via the same role chain MemoryCompactor uses.

        Tries the dream-specific override first, falls through to the
        ``"utility"`` role chain so deployments don't have to configure
        compaction model separately from other utility tasks.
        """
        if self._registry is None:
            return None
        try:
            backend, model = await self._registry.resolve_model_for_role(
                "utility",
                override=settings.dream_compaction_model,
                settings=settings,
            )
            self._resolved_backend = backend
            self._resolved_model = model
            return backend
        except Exception:
            log.debug("dream_compactor_backend_resolve_failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compact(self, user_id: str, persona_id: str = "default") -> dict:
        """Run one compaction cycle for ``user_id``. Returns stats."""
        if not user_id:
            raise ValueError("dream compaction requires user_id")
        if not settings.dream_compaction_enabled:
            return {"enabled": False}

        stats: dict = {
            "user_id": user_id,
            "deduped_pairs": 0,
            "summarized_clusters": 0,
            "summarized_entries": 0,
            "time_trimmed": 0,
            "time_trim_gated": False,
        }

        backend = await self._resolve_backend()

        # Phase 0 — count-gated time trim. When the journal is small the
        # dedup/cluster phases handle pruning; only fall back to age-based
        # deletion above the threshold.
        try:
            trim_stats = await self._journal.compact_journal(
                persona_id=persona_id,
                max_age_days=settings.dream_compaction_max_age_days,
                user_id=user_id,
                count_threshold=settings.dream_time_trim_count_threshold,
            )
            stats["time_trimmed"] = trim_stats.get("compacted", 0)
            stats["time_trim_gated"] = trim_stats.get("gated", False)
        except Exception:
            log.warning("dream_compaction_time_trim_failed", user_id=user_id, exc_info=True)

        # Phases 1-2 require LLM
        if backend is None:
            # stats already contains user_id; pass it once via the dict
            # rather than as a separate kwarg (would duplicate-arg error).
            log.info("dream_compaction_skipped_no_backend", **stats)
            return stats

        try:
            stats["deduped_pairs"] = await self._dedup_sweep(user_id, persona_id, backend)
        except Exception:
            log.warning("dream_compaction_dedup_failed", user_id=user_id, exc_info=True)

        try:
            cluster_stats = await self._summarize_clusters(user_id, persona_id, backend)
            stats["summarized_clusters"] = cluster_stats[0]
            stats["summarized_entries"] = cluster_stats[1]
        except Exception:
            log.warning("dream_compaction_cluster_failed", user_id=user_id, exc_info=True)

        log.info("dream_compaction_complete", **stats)
        return stats

    # ------------------------------------------------------------------
    # Phase 1: pair dedup
    # ------------------------------------------------------------------

    async def _dedup_sweep(
        self,
        user_id: str,
        persona_id: str,
        backend: ModelBackend,
    ) -> int:
        """Pairwise similarity scan; LLM-merge near-duplicates.

        Uses STORED embeddings from the dream_entries column rather than
        re-embedding — the backfill task ensures this is populated, and
        re-embedding hundreds of entries every 12 hours would dominate the
        cost otherwise.
        """
        from augmentum.memory.embeddings import EmbeddingService

        threshold = settings.dream_dedup_threshold

        # Pull all active (non-expired, non-pinned) entries for this user.
        # We exclude pinned because the user explicitly marked those.
        rows = await self._load_active_with_embeddings(user_id, persona_id, include_pinned=False)
        if len(rows) < 2:
            return 0

        # Decode stored embedding blobs into float lists once
        decoded: list[tuple[DreamEntry, list[float]]] = []
        for entry, blob in rows:
            try:
                decoded.append((entry, EmbeddingService.from_blob(blob)))
            except Exception as exc:
                log.debug("dream_embedding_decode_failed", entry_id=entry.id, error=str(exc))
                continue
        if len(decoded) < 2:
            return 0

        merged = 0
        merged_ids: set[str] = set()

        # O(N²) over typically <500 entries — fine in Python; the LLM
        # call is the actual cost ceiling, capped via early-exit on first
        # match per `keep` to avoid over-aggressive consolidation.
        for i in range(len(decoded)):
            keep_entry, keep_vec = decoded[i]
            if keep_entry.id in merged_ids:
                continue
            for j in range(i + 1, len(decoded)):
                drop_entry, drop_vec = decoded[j]
                if drop_entry.id in merged_ids:
                    continue
                sim = cosine_similarity(keep_vec, drop_vec)
                if sim < threshold:
                    continue

                merged_text = await self._llm_merge_pair(keep_entry, drop_entry, backend)
                if merged_text is None:
                    continue
                ok = await self._journal.merge_entries(
                    keep_id=keep_entry.id,
                    drop_id=drop_entry.id,
                    merged_content=merged_text,
                    user_id=user_id,
                )
                if ok:
                    merged += 1
                    merged_ids.add(drop_entry.id)
                    log.info(
                        "dream_compaction_pair_merged",
                        kept_id=keep_entry.id, dropped_id=drop_entry.id,
                        similarity=round(sim, 3), user_id=user_id,
                    )
                    break  # keep moves on; one merge per keep per pass

        return merged

    async def _llm_merge_pair(
        self,
        a: DreamEntry,
        b: DreamEntry,
        backend: ModelBackend,
    ) -> str | None:
        """LLM-merge two near-duplicate entries; return new content or None."""
        from augmentum.models.base import InternalChatRequest, Message

        user_prompt = (
            f"Reflection 1: {a.content}\n"
            f"Reflection 2: {b.content}\n\n"
            "Merge these into one consolidated reflection."
        )
        try:
            response = await backend.chat(InternalChatRequest(
                model=self._resolved_model or "",
                messages=[
                    Message(role="system", content=_MERGE_SYSTEM),
                    Message(role="user", content=user_prompt),
                ],
                stream=False, temperature=0.2, max_tokens=300,
            ))
            parsed = parse_merged_response(response.message.content or "")
            return parsed[0] if parsed else None
        except Exception:
            log.debug("dream_compaction_pair_merge_llm_failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Phase 2: cluster summarization
    # ------------------------------------------------------------------

    async def _summarize_clusters(
        self,
        user_id: str,
        persona_id: str,
        backend: ModelBackend,
    ) -> tuple[int, int]:
        """Greedy cluster, LLM-summarize, replace cluster with one new entry.

        Returns ``(clusters_summarized, entries_consolidated)``.
        Bounded by ``dream_compaction_max_clusters_per_run`` so a single
        pass doesn't burn the model on a backlog of clusters.
        """
        from augmentum.memory.embeddings import EmbeddingService

        threshold = settings.dream_cluster_threshold
        min_size = settings.dream_cluster_min_size
        max_clusters = settings.dream_compaction_max_clusters_per_run

        rows = await self._load_active_with_embeddings(user_id, persona_id, include_pinned=False)
        if len(rows) < min_size:
            return (0, 0)

        decoded: list[tuple[DreamEntry, list[float]]] = []
        for entry, blob in rows:
            try:
                decoded.append((entry, EmbeddingService.from_blob(blob)))
            except Exception as exc:
                log.debug("dream_embedding_decode_failed", entry_id=entry.id, error=str(exc))
                continue
        if len(decoded) < min_size:
            return (0, 0)

        # Greedy clustering — each unassigned entry seeds a new cluster,
        # absorbs any other unassigned entries above the cluster threshold.
        # Same algorithm as MemoryCompactor._summarize_clusters.
        assigned: set[int] = set()
        clusters: list[list[int]] = []

        for i in range(len(decoded)):
            if i in assigned:
                continue
            seed_vec = decoded[i][1]
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, len(decoded)):
                if j in assigned:
                    continue
                if cosine_similarity(seed_vec, decoded[j][1]) >= threshold:
                    cluster.append(j)
                    assigned.add(j)
            if len(cluster) >= min_size:
                clusters.append(cluster)
                if len(clusters) >= max_clusters:
                    break

        clusters_summarized = 0
        entries_consolidated = 0

        for cluster in clusters:
            members = [decoded[i][0] for i in cluster]
            summary_text = await self._llm_summarize_cluster(members, backend)
            if summary_text is None:
                continue

            new_id = await self._insert_cluster_summary(
                user_id=user_id,
                persona_id=persona_id,
                content=summary_text,
                members=members,
            )
            if new_id is None:
                continue

            # Soft-delete originals
            soft_deleted_ids: list[str] = []
            for m in members:
                ok = await self._soft_delete_entry(m.id, user_id)
                if ok:
                    soft_deleted_ids.append(m.id)

            clusters_summarized += 1
            entries_consolidated += len(soft_deleted_ids)
            log.info(
                "dream_compaction_cluster_summarized",
                new_entry_id=new_id, replaced=len(soft_deleted_ids),
                user_id=user_id,
            )

        return (clusters_summarized, entries_consolidated)

    async def _llm_summarize_cluster(
        self,
        members: list[DreamEntry],
        backend: ModelBackend,
    ) -> str | None:
        from augmentum.models.base import InternalChatRequest, Message

        bullet_list = "\n".join(f"- {m.content}" for m in members)
        user_prompt = (
            f"Cluster of related reflections:\n{bullet_list}\n\n"
            "Consolidate into a single broader reflection capturing the recurring theme."
        )
        try:
            response = await backend.chat(InternalChatRequest(
                model=self._resolved_model or "",
                messages=[
                    Message(role="system", content=_CLUSTER_SUMMARY_SYSTEM),
                    Message(role="user", content=user_prompt),
                ],
                stream=False, temperature=0.2, max_tokens=400,
            ))
            parsed = parse_merged_response(response.message.content or "")
            return parsed[0] if parsed else None
        except Exception:
            log.debug("dream_compaction_cluster_llm_failed", exc_info=True)
            return None

    async def _insert_cluster_summary(
        self,
        *,
        user_id: str,
        persona_id: str,
        content: str,
        members: list[DreamEntry],
    ) -> str | None:
        """Write the consolidated cluster summary as a NEW dream entry.

        Inherits the union of source_memories / source_sessions from the
        cluster so attribution is preserved. Cycle id is set to the
        oldest member's cycle so historical context isn't lost. Reuses
        ``DreamJournal.store_entry`` to get embedding + vec index for free.
        """
        # Deterministic union, sorted for stable serialization
        src_mem: list[str] = sorted({
            mid for m in members for mid in (m.source_memories or [])
        })
        src_sess: list[str] = sorted({
            sid for m in members for sid in (m.source_sessions or [])
        })
        oldest_cycle = min(
            (m.dream_cycle_id for m in members if m.dream_cycle_id),
            default="compaction",
        )
        # Use REFLECTION as the consolidated entry type — clusters mix
        # types (impressions, threads, reflections) and reflection is the
        # most general form.
        try:
            new_id = await self._journal.store_entry(
                persona_id=persona_id,
                content=content,
                entry_type=DreamEntryType.REFLECTION,
                source_memories=src_mem,
                source_sessions=src_sess,
                context_window={"compaction": True, "cluster_size": len(members)},
                dream_cycle_id=oldest_cycle,
                user_id=user_id,
            )
            return new_id
        except Exception:
            log.warning("dream_compaction_cluster_insert_failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_active_with_embeddings(
        self,
        user_id: str,
        persona_id: str,
        *,
        include_pinned: bool,
    ) -> list[tuple[DreamEntry, bytes]]:
        """Load ``[(entry, embedding_blob), ...]`` for active entries.

        Filters: ``user_id`` (required), ``persona_id``, ``expires_at IS
        NULL``, ``embedding IS NOT NULL``. Excludes pinned by default
        because user explicitly marked those as worth keeping. Source
        for both dedup and cluster phases.
        """
        conn = self._journal._db
        if conn is None:
            return []
        pinned_filter = "" if include_pinned else " AND pinned = 0"
        try:
            cursor = await conn.execute(
                f"""SELECT id, persona_id, content, entry_type, source_memories,
                          source_sessions, context_window, embedding, weight,
                          pinned, dream_cycle_id, created_at, expires_at
                    FROM dream_entries
                    WHERE persona_id = ? AND user_id = ?
                          AND expires_at IS NULL
                          AND embedding IS NOT NULL{pinned_filter}""",
                (persona_id, user_id),
            )
            rows = await cursor.fetchall()
        except Exception:
            log.debug("dream_compaction_load_failed", exc_info=True)
            return []
        out: list[tuple[DreamEntry, bytes]] = []
        # _row_to_entry needs Row-like access; use json/string fields directly.
        for r in rows:
            try:
                entry = DreamEntry(
                    id=r[0], persona_id=r[1], content=r[2],
                    entry_type=DreamEntryType(r[3]),
                    source_memories=json.loads(r[4]) if r[4] else [],
                    source_sessions=json.loads(r[5]) if r[5] else [],
                    context_window=json.loads(r[6]) if r[6] else {},
                    embedding=r[7], weight=r[8], pinned=bool(r[9]),
                    dream_cycle_id=r[10], created_at=r[11], expires_at=r[12],
                )
                if r[7]:
                    out.append((entry, r[7]))
            except Exception as exc:
                log.debug("dream_row_decode_failed", row_id=r[0] if r else None, error=str(exc))
                continue
        return out

    async def _soft_delete_entry(self, entry_id: str, user_id: str) -> bool:
        """Mark an entry expired (scoped to user_id). Returns True on hit."""
        conn = self._journal._db
        if conn is None:
            return False
        expires_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        try:
            cursor = await conn.execute(
                "UPDATE dream_entries SET expires_at = ? "
                "WHERE id = ? AND user_id = ? AND expires_at IS NULL",
                (expires_at, entry_id, user_id),
            )
            await conn.commit()
            return cursor.rowcount > 0
        except Exception:
            log.debug("dream_compaction_soft_delete_failed", entry_id=entry_id, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Per-user compaction loop, mirroring MemoryCompactor._loop.

        Iterates the active users table each cycle so each tenant's
        journal gets pruned independently. Empty users-table or auth-less
        installs fall back to the literal "default" user (legacy single-
        tenant compatibility).
        """
        interval_s = max(60.0, settings.dream_compaction_interval_hours * 3600)
        while True:
            await asyncio.sleep(interval_s)
            if not settings.dream_compaction_enabled:
                continue
            try:
                user_ids = await self._list_active_users()
                for uid in user_ids:
                    try:
                        await self.compact(uid)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.warning(
                            "dream_compaction_loop_user_error",
                            user_id=uid, exc_info=True,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("dream_compaction_loop_error", exc_info=True)

    async def _list_active_users(self) -> list[str]:
        """Resolve the set of user_ids to compact this cycle.

        Reads from the ``users`` table when available (post-multi-tenancy
        installs); falls back to ``["default"]`` for legacy single-user
        installs that don't have the auth tables.
        """
        conn = self._journal._db
        if conn is None:
            return ["default"]
        try:
            cursor = await conn.execute(
                "SELECT id FROM users WHERE is_active = 1"
            )
            rows = await cursor.fetchall()
            user_ids = [r[0] for r in rows]
            return user_ids if user_ids else ["default"]
        except Exception:
            return ["default"]
