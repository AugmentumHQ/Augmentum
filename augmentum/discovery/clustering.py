"""Discovery Phase 2 — interest clustering and signal assignment."""
from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Weight map for depth-level computation
# ---------------------------------------------------------------------------
SIGNAL_WEIGHT_MAP: dict[str, float] = {
    "page_visit": 0.5,
    "search_query": 0.5,
    "video_open": 0.5,
    "video_watch": 1.0,
    "video_seek": 0.8,
    "ai_action": 2.0,
    "discuss": 2.5,
    "note_save": 3.0,
    "video_summary": 2.0,
}


# ---------------------------------------------------------------------------
# ClusterData
# ---------------------------------------------------------------------------
@dataclass
class ClusterData:
    """In-memory representation of an interest cluster."""

    cluster_id: str = ""
    name: str = ""
    centroid: list[float] = field(default_factory=list)
    frecency_short: float = 0.0
    frecency_long: float = 0.0
    depth_level: int = 1
    signal_count: int = 0
    narration: str | None = None
    knowledge_gaps: str | None = None
    adjacent_topics: str | None = None
    dampened: bool = False
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.cluster_id:
            self.cluster_id = f"c_{uuid.uuid4().hex[:12]}"
        if self.depth_level < 1:
            self.depth_level = 1
        elif self.depth_level > 5:
            self.depth_level = 5
        now = datetime.now(UTC).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_centroid(vectors: list[list[float]]) -> list[float]:
    """Element-wise average of a list of vectors."""
    if not vectors:
        return []
    dim = len(vectors[0])
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


# Single-word cluster names that are common-English noise — visiting a page
# titled just "BEGINNER" or "EXPLORE" was minting a cluster of that name,
# which then drove generic SearXNG queries returning dictionary landing
# pages every poll cycle. The downstream curator + filter_for_llm reject
# the dictionary results, but the wasted round-trip happens anyway. Reject
# the cluster mint upstream when the auto-generated name is a single
# generic token. Multi-word names and technical-looking single words
# (kubernetes, transformer, rustlang, etc.) still pass cleanly.
_SINGLE_WORD_NOISE_STOPLIST: frozenset[str] = frozenset({
    # Generic verbs (common pollutants observed in logs)
    "open", "close", "explore", "introduction", "logic", "replace",
    "dragged", "click", "tap", "swipe", "press", "select", "search",
    "find", "look", "view", "see", "show", "hide", "start", "stop",
    "pause", "play", "skip", "next", "back", "home", "load", "loaded",
    "save", "saved", "submit", "cancel", "delete", "remove", "add",
    "create", "edit", "update", "refresh", "reload", "retry", "help",
    # Authentication / navigation chrome
    "login", "logout", "signin", "signup", "register", "settings",
    "options", "menu", "about", "contact", "support", "faq", "terms",
    "privacy", "welcome", "hello", "thanks",
    # Generic qualifiers + adjectives
    "top", "best", "worst", "good", "bad", "easy", "hard", "fast", "slow",
    "big", "small", "new", "old", "free", "paid", "premium", "trial",
    "demo", "beta", "alpha",
    # Generic nouns
    "beginner", "advanced", "intermediate", "expert", "professional",
    "thing", "stuff", "item", "page", "site", "app", "web", "online",
    "guide", "tutorial", "lesson", "course", "class", "tip", "trick",
    "review", "rating", "score", "result", "data", "info", "news",
    "today", "yesterday", "tomorrow", "now", "soon", "later",
})

# Hide noisy partial words that almost never carry signal alone.
_MIN_CLUSTER_NAME_CHARS: int = 3


def _generate_cluster_name(text: str) -> str:
    """First 8 words, max 55 chars."""
    words = text.split()[:8]
    name = " ".join(words)
    if len(name) > 55:
        name = name[:55].rsplit(" ", 1)[0]
    return name


def is_meaningful_cluster_name(name: str) -> bool:
    """Return True when the auto-generated name is worth minting a cluster for.

    The single-word + common-English case is the named failure mode — see
    ``_SINGLE_WORD_NOISE_STOPLIST`` and the curator dictionary-spam pattern
    in logs. Callers that mint clusters from raw signal text should skip
    the create when this returns False; the signal still exists on disk,
    it just doesn't anchor an interest cluster until a richer signal lands
    nearby in embedding space.
    """
    cleaned = (name or "").strip().lower()
    if len(cleaned) < _MIN_CLUSTER_NAME_CHARS:
        return False
    tokens = cleaned.split()
    if len(tokens) >= 2:
        return True
    return tokens[0] not in _SINGLE_WORD_NOISE_STOPLIST


# ---------------------------------------------------------------------------
# Depth level
# ---------------------------------------------------------------------------

def compute_depth_level(signals: list[dict]) -> int:
    """Compute depth level 1-5 from signal type diversity and total weight.

    Each signal contributes its mapped weight from SIGNAL_WEIGHT_MAP.
    """
    if not signals:
        return 1

    types: set[str] = set()
    total_weight = 0.0
    for s in signals:
        sig_type = s.get("signal_type", "")
        types.add(sig_type)
        total_weight += SIGNAL_WEIGHT_MAP.get(sig_type, s.get("weight", 1.0))

    n_types = len(types)

    if total_weight >= 15 and n_types >= 4:
        return 5
    if total_weight >= 8 and n_types >= 3:
        return 4
    if total_weight >= 4 and n_types >= 2:
        return 3
    if total_weight >= 2 or n_types >= 2:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Narration — short label string for the dormant `narration` cluster field.
# Discovery UI surfaces it as a human-readable summary instead of the raw
# cluster name. Template-driven for v1; LLM-mediated narration could replace
# this for high-signal clusters in a later pass.
# ---------------------------------------------------------------------------

_SIGNAL_TYPE_HUMAN: dict[str, str] = {
    "page_visit": "reading",
    "video_watch": "watching",
    "video_open": "exploring",
    "video_seek": "skimming",
    "video_summary": "summarizing",
    "search_query": "searching",
    "note_save": "saving notes on",
    "ai_action": "discussing with AI",
    "discuss": "talking about",
}


def compose_narration(*, name: str, signal_count: int, signals: list[dict]) -> str:
    """One-line narration for an interest cluster.

    Template: ``"Exploring {name} — {N} items, mostly {action}"``
    where action is the most-frequent signal type, humanized via
    :data:`_SIGNAL_TYPE_HUMAN`. Returns empty string when there isn't
    enough material to say anything coherent.

    The signals list can be sampled — caller decides how many to pass
    (typically up to 100 from ``store.list_signals``). The narration
    reflects the SAMPLE, not the full cluster history.
    """
    name = (name or "").strip()
    if not name or signal_count <= 0:
        return ""

    type_counts: dict[str, int] = {}
    for s in signals:
        t = s.get("signal_type", "")
        if t:
            type_counts[t] = type_counts.get(t, 0) + 1

    items_word = "item" if signal_count == 1 else "items"
    if not type_counts:
        return f"Exploring {name} — {signal_count} {items_word}"

    top_type = max(type_counts.items(), key=lambda p: p[1])[0]
    action = _SIGNAL_TYPE_HUMAN.get(top_type, top_type.replace("_", " "))
    return f"Exploring {name} — {signal_count} {items_word}, mostly {action}"


# ---------------------------------------------------------------------------
# Signal text extraction
# ---------------------------------------------------------------------------

def extract_signal_text(signal: dict) -> str:
    """Extract representative text from a signal for embedding."""
    from augmentum.discovery.text_clean import clean_text_for_query

    sig_type = signal.get("signal_type", "")
    title = signal.get("source_title", "")
    metadata = signal.get("metadata", {})
    if isinstance(metadata, str):
        import json
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    if sig_type == "search_query":
        raw = metadata.get("query", title)
    elif sig_type in ("video_open", "video_watch", "video_seek", "video_summary"):
        channel = metadata.get("channel", "")
        parts = [title]
        if channel:
            parts.append(channel)
        raw = " ".join(parts)
    else:
        raw = title

    return clean_text_for_query(raw)


# ---------------------------------------------------------------------------
# Cluster assignment
# ---------------------------------------------------------------------------

async def assign_signal_to_cluster(
    store,
    signal_id: str,
    signal_text: str,
    signal_type: str,
    signal_weight: float,
    *,
    user_id: str = "",
) -> str | None:
    """Embed signal text, find or create cluster, assign signal.

    Returns the cluster_id the signal was assigned to, or None on failure.
    Cluster create/lookup is scoped to ``user_id`` so two users' interest
    graphs never merge.
    """
    try:
        from augmentum.memory.embeddings import EmbeddingService

        embedding = await asyncio.to_thread(EmbeddingService.embed_query, signal_text)
        emb_blob = EmbeddingService.to_blob(embedding)
    except Exception:
        log.warning("cluster_embed_failed", signal_id=signal_id)
        return None

    # Try to find an existing nearby cluster (NOTE: find_nearest_cluster
    # uses the vec0 virtual table which doesn't carry user_id — callers
    # must verify ownership of the returned cluster_id. The threshold is
    # semantic, so the worst case is a per-user cluster merge across
    # tenants; we post-filter by user_id via list_clusters lookup below.)
    nearest = await store.find_nearest_cluster(emb_blob, threshold=0.75)
    if nearest is not None:
        candidate_id = nearest["cluster_id"]
        # Confirm ownership before reusing.
        user_clusters = {
            c["cluster_id"]
            for c in await store.list_clusters(include_dampened=True, user_id=user_id)
        }
        if candidate_id in user_clusters:
            await store.update_signal_cluster(signal_id, candidate_id)
            return candidate_id

    # Create a new cluster owned by this user — unless the auto-generated
    # name is single-word common-English noise (see is_meaningful_cluster_name).
    # The signal row still exists; it just doesn't anchor a cluster yet.
    candidate_name = _generate_cluster_name(signal_text)
    if not is_meaningful_cluster_name(candidate_name):
        log.info(
            "cluster_create_skipped_noise_name",
            signal_id=signal_id,
            name=candidate_name[:40],
        )
        return None
    cluster = ClusterData(
        name=candidate_name,
        centroid=embedding,
    )
    await store.upsert_cluster({
        "cluster_id": cluster.cluster_id,
        "name": cluster.name,
        "centroid_embedding": emb_blob,
        "frecency_short": 0.0,
        "frecency_long": 0.0,
        "depth_level": 1,
        "signal_count": 1,
        "narration": None,
        "knowledge_gaps": None,
        "adjacent_topics": None,
        "dampened": 0,
        "created_at": cluster.created_at,
        "updated_at": cluster.updated_at,
    }, user_id=user_id)
    await store.upsert_cluster_vec(cluster.cluster_id, emb_blob)
    await store.update_signal_cluster(signal_id, cluster.cluster_id)
    return cluster.cluster_id
