"""Background memory compaction — prune, archive, cluster-summarize.

Runs periodically to keep the memory store lean:
1. Utility-based deletion: old + low-access + low-importance
2. Tier demotion: old + low-access to ARCHIVE
3. Cluster summarization: group ARCHIVE by similarity, LLM-summarize (optional)
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.memory.models import MemoryTier
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.memory.store import MemoryStore
    from augmentum.models.base import ModelBackend

log = get_logger(__name__)

_SUMMARIZE_SYSTEM = """\
You summarize a cluster of related user memories into one concise statement. Rules:
- Combine all key information into a single, self-contained sentence.
- Preserve specifics (names, numbers, tools, preferences).
- Return valid JSON: {"summary": "...", "importance": 0.7}
"""


class MemoryCompactor:
    """Periodic background compaction of the memory store."""

    def __init__(
        self,
        store: MemoryStore,
        backend: ModelBackend | None = None,
        model: str | None = None,
        interval_hours: float | None = None,
        registry: object | None = None,
    ) -> None:
        self._store = store
        self._backend = backend
        self._default_model = model
        self._registry = registry
        self._interval = (interval_hours or settings.memory_compaction_interval_hours) * 3600
        self._task: asyncio.Task | None = None
        # Resolved per-cycle backend/model (updated by _resolve_backend)
        self._resolved_backend: ModelBackend | None = backend
        self._resolved_model: str | None = model

    @property
    def _model(self) -> str | None:
        """Resolve the LLM model dynamically — respects runtime config changes."""
        return settings.memory_llm_extraction_model or self._default_model or None

    async def _resolve_backend(self) -> ModelBackend | None:
        """Resolve the LLM backend via role chain on each compaction cycle."""
        if self._registry is None:
            return self._backend
        try:
            backend, model = await self._registry.resolve_model_for_role(
                "utility",
                override=settings.memory_llm_extraction_model,
                settings=settings,
            )
            self._resolved_backend = backend
            self._resolved_model = model
            return backend
        except Exception:
            log.debug("compactor_backend_resolve_failed", exc_info=True)
            return self._backend

    def start(self) -> None:
        """Start the background compaction loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())
        log.info("compactor_started", interval_hours=self._interval / 3600)

    async def stop(self) -> None:
        """Stop the background compaction loop."""
        if self._task is not None:
            self._task.cancel()
            import contextlib
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
            log.info("compactor_stopped")

    async def compact(self, user_id: str = "default") -> dict:
        """Run one compaction cycle. Returns stats."""
        stats = {"deleted": 0, "archived": 0, "summarized": 0}

        # Resolve backend fresh each cycle (picks up registry / setting changes)
        active_backend = await self._resolve_backend()

        max_age = settings.memory_compaction_max_age_days

        # Phase 1: Utility-based deletion
        candidates = await self._store.get_compaction_candidates(
            user_id=user_id,
            max_age_days=max_age,
            max_access_count=2,
            max_importance=0.3,
        )
        for mem in candidates:
            await self._store.forget(mem.id, user_id=user_id)
            stats["deleted"] += 1

        # Phase 2: Tier demotion to ARCHIVE
        candidates = await self._store.get_compaction_candidates(
            user_id=user_id,
            max_age_days=max_age,
            max_access_count=5,
            max_importance=0.6,
        )
        for mem in candidates:
            if mem.tier != MemoryTier.ARCHIVE:
                # No per-memory event row: a compaction run can demote up to
                # 100 memories — last_compacted_at already records the action.
                await self._store.update_tier(mem.id, MemoryTier.ARCHIVE, user_id=user_id, log_change=False)
                stats["archived"] += 1

        # Phase 3: Dedup sweep — find and merge near-duplicate pairs
        deduped = await self._dedup_sweep(user_id, active_backend)
        stats["deduped"] = deduped

        # Phase 4: Cluster summarization (requires LLM)
        if active_backend is not None:
            summarized = await self._summarize_clusters(user_id, active_backend)
            stats["summarized"] = summarized

        log.info("compaction_complete", user_id=user_id, **stats)
        return stats

    async def _dedup_sweep(self, user_id: str, active_backend: ModelBackend | None = None) -> int:
        """Scan active memories for near-duplicate pairs and merge them."""
        from augmentum.memory.embeddings import EmbeddingService

        dedup_thresh = settings.memory_dedup_threshold
        all_mems = await self._store.list_all(user_id=user_id, limit=300)
        if len(all_mems) < 2:
            return 0

        mem_embeddings: list[list[float]] = []
        valid_indices: list[int] = []
        for i, mem in enumerate(all_mems):
            try:
                emb = await asyncio.to_thread(EmbeddingService.embed_one, mem.content)
                mem_embeddings.append(emb)
                valid_indices.append(i)
            except Exception as exc:
                log.debug("memory_embed_failed", mem_id=mem.id, error=str(exc))
                continue

        if len(valid_indices) < 2:
            return 0

        merged = 0
        forgotten_ids: set[str] = set()

        for a_pos in range(len(valid_indices)):
            mem_a = all_mems[valid_indices[a_pos]]
            if mem_a.id in forgotten_ids:
                continue
            for b_pos in range(a_pos + 1, len(valid_indices)):
                mem_b = all_mems[valid_indices[b_pos]]
                if mem_b.id in forgotten_ids:
                    continue
                sim = _cosine_similarity(mem_embeddings[a_pos], mem_embeddings[b_pos])
                if sim < dedup_thresh:
                    continue

                if (mem_a.importance > mem_b.importance or
                        (mem_a.importance == mem_b.importance
                         and mem_a.access_count >= mem_b.access_count)):
                    keep, drop = mem_a, mem_b
                else:
                    keep, drop = mem_b, mem_a

                # Try LLM merge if backend available
                if active_backend is not None:
                    consolidated = await self._llm_merge_pair(keep, drop, active_backend)
                    if consolidated:
                        merged_text, _ = consolidated
                        await self._store.edit(keep.id, merged_text, user_id=user_id)
                        log.info("compaction_dedup_consolidated",
                                 kept_id=keep.id, dropped_id=drop.id,
                                 similarity=round(sim, 3))
                    else:
                        log.info("compaction_dedup_kept_primary",
                                 kept_id=keep.id, dropped_id=drop.id)
                else:
                    log.info("compaction_dedup_kept_primary",
                             kept_id=keep.id, dropped_id=drop.id)

                await self._store.forget(drop.id, user_id=user_id)
                forgotten_ids.add(drop.id)
                merged += 1

        return merged

    async def _llm_merge_pair(self, mem_a, mem_b, active_backend: ModelBackend | None = None) -> tuple[str, float] | None:
        """Merge two similar memories into one enriched statement via LLM."""
        from augmentum.models.base import InternalChatRequest, Message

        backend = active_backend or self._backend
        if backend is None:
            return None

        system = (
            "Merge these two related user memories into one concise, enriched statement. "
            "Preserve ALL specific details (names, numbers, dates, preferences). "
            "Do NOT add information not present in either memory. "
            'Return valid JSON: {"merged": "...", "importance": 0.8}'
        )
        user_prompt = f"Memory 1: {mem_a.content}\nMemory 2: {mem_b.content}"

        try:
            from augmentum.config import settings as _settings
            _think = bool(getattr(_settings, "onboard_reasoning_thinking", True))
            _max_tok = int(getattr(_settings, "onboard_reasoning_max_tokens", 8192)) if _think else 200
            response = await backend.chat(InternalChatRequest(
                model=self._model or "",
                messages=[
                    Message(role="system", content=system),
                    Message(role="user", content=user_prompt),
                ],
                stream=False, temperature=0.1,
                max_tokens=_max_tok,
                think=_think,
            ))
            return _parse_summary_response(response.message.content)
        except Exception:
            log.debug("dedup_llm_merge_failed", exc_info=True)
            return None

    async def _summarize_clusters(self, user_id: str, active_backend: ModelBackend | None = None) -> int:
        """Group ARCHIVE memories by similarity and summarize clusters of 3+."""
        archived = await self._store.list_all(
            user_id=user_id, limit=200, tier=MemoryTier.ARCHIVE,
        )
        if len(archived) < 3:
            return 0

        # Simple greedy clustering by content similarity via embeddings
        from augmentum.memory.embeddings import EmbeddingService

        # Embed all archived memories
        embeddings: list[tuple[int, list[float]]] = []
        for i, mem in enumerate(archived):
            try:
                emb = EmbeddingService.embed_one(mem.content)
                embeddings.append((i, emb))
            except Exception as exc:
                log.debug("memory_archive_embed_failed", mem_id=mem.id, error=str(exc))
                continue

        if len(embeddings) < 3:
            return 0

        # Greedy clustering: for each unassigned memory, find similar ones
        assigned: set[int] = set()
        clusters: list[list[int]] = []

        for i, emb_i in embeddings:
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j, emb_j in embeddings:
                if j in assigned:
                    continue
                sim = _cosine_similarity(emb_i, emb_j)
                if sim >= 0.65:
                    cluster.append(j)
                    assigned.add(j)
            if len(cluster) >= 3:
                clusters.append(cluster)

        summarized = 0
        for cluster in clusters:
            mems = [archived[i] for i in cluster]
            contents = [m.content for m in mems]

            summary = await self._llm_summarize(contents, active_backend)
            if summary is None:
                continue

            merged_text, importance = summary

            # Store the summary as a new ACTIVE memory
            from augmentum.memory.models import MemoryType, SourceType

            await self._store.store(
                content=merged_text,
                memory_type=MemoryType.FACT,
                user_id=user_id,
                importance=importance,
                source_type=SourceType.SYSTEM,
                source_context={"extraction": "compaction", "cluster_size": len(cluster)},
            )

            # Soft-delete the cluster members
            for mem in mems:
                await self._store.forget(mem.id, user_id=user_id)

            summarized += len(cluster)

        return summarized

    async def _llm_summarize(self, contents: list[str], active_backend: ModelBackend | None = None) -> tuple[str, float] | None:
        """Summarize a cluster of memories via LLM."""
        from augmentum.models.base import InternalChatRequest, Message

        backend = active_backend or self._backend
        if backend is None:
            return None

        bullet_list = "\n".join(f"- {c}" for c in contents)
        user_prompt = f"Summarize these related user memories:\n{bullet_list}"

        from augmentum.config import settings as _settings
        _think = bool(getattr(_settings, "onboard_reasoning_thinking", True))
        _max_tok = int(getattr(_settings, "onboard_reasoning_max_tokens", 8192)) if _think else 300
        request = InternalChatRequest(
            model=self._model or "",
            messages=[
                Message(role="system", content=_SUMMARIZE_SYSTEM),
                Message(role="user", content=user_prompt),
            ],
            stream=False,
            temperature=0.1,
            max_tokens=_max_tok,
            think=_think,
        )

        try:
            response = await backend.chat(request)
            return _parse_summary_response(response.message.content)
        except Exception:
            log.warning("cluster_summarization_failed", exc_info=True)
            return None

    async def _loop(self) -> None:
        """Background loop that runs compact() periodically for every user.

        Before multi-tenancy, this called ``self.compact()`` with no args and
        only the literal ``"default"`` user's memories ever got pruned —
        every other tenant's store grew unbounded. We now iterate the users
        table each cycle so each tenant gets their own compaction pass.
        """
        while True:
            await asyncio.sleep(self._interval)
            try:
                try:
                    cursor = await self._store._conn.execute(
                        "SELECT id FROM users WHERE is_active = 1",
                    )
                    user_ids = [row[0] for row in await cursor.fetchall()]
                except Exception:
                    # Legacy single-user install — users table may not exist.
                    user_ids = ["default"]

                if not user_ids:
                    user_ids = ["default"]

                for uid in user_ids:
                    try:
                        await self.compact(uid)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.warning(
                            "compaction_loop_user_error",
                            user_id=uid, exc_info=True,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("compaction_loop_error", exc_info=True)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_summary_response(raw: str) -> tuple[str, float] | None:
    """Parse the LLM summary response."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    summary = str(data.get("summary") or data.get("merged") or "").strip()
    if len(summary) < 5:
        return None

    try:
        importance = max(0.0, min(1.0, float(data.get("importance", 0.7))))
    except (TypeError, ValueError):
        importance = 0.7

    return summary, importance
