"""Memory reflection — generate higher-level understanding from accumulated facts.

Periodically clusters ACTIVE-tier memories and generates reflective insights
that capture patterns about the user. Reflections land ACTIVE (they synthesize
already-stored memories, so they have evidentiary basis) and EARN promotion to
CORE through the same corroboration ladder as everything else — a machine-made
abstraction does not outrank user-confirmed facts in always-on context.
(Legacy force-to-CORE-on-write is behind ``memory_reflection_force_core``.)

Inspired by the Generative Agents (Park et al., 2023) reflection mechanism:
synthesize higher-level abstractions from concrete observations.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from augmentum.memory.models import (
    ExtractedFact,
    Memory,
    MemoryTier,
    MemoryType,
    SourceType,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.memory.store import MemoryStore
    from augmentum.models.base import ModelBackend

log = get_logger(__name__)

# Maximum reflections per user — prevents reflection bloat
MAX_REFLECTIONS = 5

# Eligibility thresholds — only battle-tested facts seed reflections
MIN_IMPORTANCE = 0.7
MIN_ACCESS_COUNT = 3
MIN_CONFIDENCE = 0.8

# Minimum cluster size
MIN_CLUSTER_SIZE = 3

_REFLECTION_PROMPT = """\
You are reviewing verified facts about a user to generate deeper understanding.

## Verified facts
{facts}

## Task
Write 1-2 sentences that capture the PATTERN or UNDERSTANDING these facts \
reveal about this person. This should be a genuine insight about who they \
are, what they value, or how they think — not a summary of the facts.

Good: "Values reliability and offline capability, likely due to a mobile \
lifestyle with frequent travel"
Bad: "Lives in <city>, travels by plane, and needs offline tools"

Focus on understanding, not listing. Start with what the pattern reveals, \
not what the individual facts say.

Return valid JSON:
{{"reflection": "your insight here"}}
Return {{"reflection": ""}} if the facts don't reveal a meaningful pattern.
"""


async def generate_reflections(
    store: MemoryStore,
    backend: ModelBackend,
    model: str,
    user_id: str = "default",
) -> list[str]:
    """Generate reflective insights from accumulated ACTIVE-tier memories.

    Returns list of stored reflection memory IDs.
    """
    from augmentum.models.base import InternalChatRequest, Message

    # Count existing reflections — cap at MAX_REFLECTIONS
    existing_reflections = await _count_reflections(store, user_id)
    if existing_reflections >= MAX_REFLECTIONS:
        log.info("reflection_cap_reached", user_id=user_id, count=existing_reflections)
        return []

    # Fetch eligible memories (high-confidence, frequently accessed, aged)
    eligible = await _fetch_eligible_memories(store, user_id)
    if len(eligible) < MIN_CLUSTER_SIZE:
        log.debug("reflection_insufficient_memories", user_id=user_id, eligible=len(eligible))
        return []

    # Cluster by embedding similarity
    clusters = _cluster_memories(eligible)

    stored_ids: list[str] = []
    remaining_slots = MAX_REFLECTIONS - existing_reflections

    for cluster in clusters[:remaining_slots]:
        if len(cluster) < MIN_CLUSTER_SIZE:
            continue

        # Build the prompt
        # Sanitize memory content to prevent prompt injection
        def _sanitize(text: str) -> str:
            """Strip potential injection markers from memory content."""
            import re
            # Remove instruction-like patterns
            text = re.sub(r"(?i)(ignore|disregard|forget)\s+(all\s+)?(previous\s+)?(instructions?|rules?|prompts?)", "", text)
            # Remove system/assistant role markers
            text = re.sub(r"(?i)(system|assistant|user)\s*:", "", text)
            return text.strip()[:200]

        facts_text = "\n".join(
            f"- {_sanitize(mem.content)}" for mem in cluster
        )
        prompt = _REFLECTION_PROMPT.format(facts=facts_text)

        try:
            request = InternalChatRequest(
                model=model,
                messages=[
                    Message(role="user", content=prompt),
                ],
                stream=False,
                temperature=0.3,
                max_tokens=200,
            )
            response = await backend.chat(request)
            raw = (response.message.content or "").strip()

            # Parse reflection
            reflection_text = _parse_reflection(raw)
            if not reflection_text:
                continue

            # Store the reflection. It's a synthesis OVER already-stored
            # memories, so it has evidentiary basis (lands ACTIVE, not the
            # PROVISIONAL quarantine) — but a machine-made abstraction must
            # EARN always-on CORE via the same corroboration ladder as
            # everything else. Force-promoting it to CORE on write (legacy)
            # let an unverified pattern outrank user-confirmed facts in the
            # always-injected set — the "looks earned but isn't" path the
            # Earned Understanding design exists to close.
            source_ids = [mem.id for mem in cluster]
            mem_id = await store.store(
                content=reflection_text,
                memory_type=MemoryType.FACT,
                user_id=user_id,
                importance=0.9,
                confidence=1.0,
                source_type=SourceType.SYSTEM,
                source_context={
                    "extraction": "reflection",
                    "source_memory_ids": source_ids,
                },
            )

            # Legacy escape hatch: force straight to CORE on write.
            from augmentum.config import settings as _settings
            if getattr(_settings, "memory_reflection_force_core", False):
                await store.update_tier(mem_id, MemoryTier.CORE, user_id=user_id)

            stored_ids.append(mem_id)
            log.info(
                "reflection_generated",
                user_id=user_id,
                cluster_size=len(cluster),
                reflection=reflection_text[:80],
            )

        except Exception:
            log.warning("reflection_generation_failed", user_id=user_id, exc_info=True)

    return stored_ids


async def _count_reflections(store: MemoryStore, user_id: str) -> int:
    """Count existing reflection memories for a user."""
    try:
        cursor = await store._conn.execute(
            "SELECT COUNT(*) FROM memories "
            "WHERE user_id = ? AND valid_until IS NULL "
            "AND source_context LIKE '%\"extraction\": \"reflection\"%'",
            (user_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


async def _fetch_eligible_memories(store: MemoryStore, user_id: str) -> list[Memory]:
    """Fetch memories eligible for reflection.

    Requirements:
    - ACTIVE or CORE tier (not PROVISIONAL or ARCHIVE)
    - effective_importance >= 0.7 (base importance + access boost - time decay)
    - access_count >= 3 (proven useful through natural retrieval)
    - confidence >= 0.8
    - Not already a reflection

    SQL pre-filters with a lower base importance (0.4) to catch memories
    whose effective importance has grown through access. Final filtering
    is done in Python using _effective_importance().
    """
    try:
        cursor = await store._conn.execute(
            "SELECT * FROM memories "
            "WHERE user_id = ? AND valid_until IS NULL "
            "AND tier IN ('active', 'core') "
            "AND importance >= 0.4 "
            "AND access_count >= ? "
            "AND confidence >= ? "
            "AND (source_context IS NULL OR source_context NOT LIKE '%\"extraction\": \"reflection\"%') "
            "ORDER BY importance DESC, access_count DESC "
            "LIMIT 100",
            (user_id, MIN_ACCESS_COUNT, MIN_CONFIDENCE),
        )
        rows = await cursor.fetchall()
        candidates = [store._row_to_memory(dict(r)) for r in rows]

        # Filter by effective importance (accounts for access growth + time decay)
        return [
            m for m in candidates
            if store._effective_importance(m) >= MIN_IMPORTANCE
        ]
    except Exception:
        log.warning("reflection_fetch_failed", user_id=user_id, exc_info=True)
        return []


def _cluster_memories(memories: list[Memory], threshold: float = 0.6) -> list[list[Memory]]:
    """Simple threshold-based clustering by embedding similarity.

    Groups memories where pairwise cosine similarity exceeds threshold.
    Returns list of clusters, each cluster is a list of Memory objects.
    """
    from augmentum.memory.embeddings import EmbeddingService

    if not memories:
        return []

    # Get embeddings (they're already stored but may not be loaded as lists)
    texts = [m.content for m in memories]
    try:
        embeddings = EmbeddingService.embed(texts)
    except Exception:
        return []

    # Simple greedy clustering
    used = set()
    clusters: list[list[Memory]] = []

    for i in range(len(memories)):
        if i in used:
            continue
        cluster = [memories[i]]
        used.add(i)

        for j in range(i + 1, len(memories)):
            if j in used:
                continue
            sim = _cosine(embeddings[i], embeddings[j])
            if sim >= threshold:
                cluster.append(memories[j])
                used.add(j)

        if len(cluster) >= MIN_CLUSTER_SIZE:
            clusters.append(cluster)

    # Sort: largest clusters first
    clusters.sort(key=len, reverse=True)
    return clusters


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_reflection(raw: str) -> str:
    """Parse reflection from LLM response."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(text)
        reflection = data.get("reflection", "").strip()
        if len(reflection) < 20:
            return ""
        return reflection
    except json.JSONDecodeError:
        # Try to extract as plain text if JSON fails
        if len(text) >= 20 and not text.startswith("{"):
            return text[:300]
        return ""
