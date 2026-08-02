"""FILCO-style sentence-level span filtering for retrieved chunks.

EXPERIMENTAL: Not integrated into the production pipeline. Benchmark
showed identical precision/recall (keywords survive filtering) but 7%
token reduction. Real benefit is LLM answer quality, which our keyword
benchmark can't measure. Kept for future integration.

After retrieving full chunks, filters out individual sentences that
aren't relevant to the query. Reduces injected context size and
improves precision by removing noise within otherwise-relevant chunks.

Reference: arxiv.org/abs/2311.08377 (Learning to Filter Context for RAG)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.documents.scoring import ScoredChunk

log = get_logger(__name__)

# Sentence-ending pattern (handles ". ", "! ", "? ", newlines)
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|\n{2,}')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping non-trivial ones."""
    parts = _SENTENCE_RE.split(text)
    return [s.strip() for s in parts if s.strip() and len(s.strip()) > 20]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def filter_chunk_spans(
    chunks: list[ScoredChunk],
    query: str,
    min_sentence_score: float = 0.3,
    min_sentences: int = 1,
    max_sentences: int = 8,
) -> list[ScoredChunk]:
    """Filter each chunk to only its query-relevant sentences.

    For each chunk:
    1. Split into sentences
    2. Score each sentence against the query using embedding similarity
    3. Keep sentences above min_sentence_score
    4. Reassemble the chunk with only relevant sentences

    If a chunk has fewer than min_sentences relevant, keep the top
    min_sentences by score to avoid dropping the chunk entirely.

    Returns new ScoredChunk objects with filtered content.
    """
    if not chunks:
        return chunks

    from augmentum.memory.embeddings import EmbeddingService

    # Embed the query once
    query_vec = EmbeddingService.embed_query(query)

    filtered: list[ScoredChunk] = []

    for sc in chunks:
        content = sc.chunk.get("content", "")
        sentences = _split_sentences(content)

        if len(sentences) <= 1:
            # Single sentence or can't split — keep as-is
            filtered.append(sc)
            continue

        # Embed all sentences in batch
        sent_vecs = EmbeddingService.embed(sentences)

        # Score each sentence
        scored_sents = [
            (sent, _cosine_similarity(query_vec, svec))
            for sent, svec in zip(sentences, sent_vecs)
        ]
        scored_sents.sort(key=lambda x: x[1], reverse=True)

        # Keep sentences above threshold
        kept = [(s, sc) for s, sc in scored_sents if sc >= min_sentence_score]

        # Ensure minimum sentences
        if len(kept) < min_sentences:
            kept = scored_sents[:min_sentences]

        # Cap at max
        kept = kept[:max_sentences]

        # Restore original order
        kept_texts = {s for s, _ in kept}
        ordered = [s for s in sentences if s in kept_texts]

        if ordered:
            from augmentum.documents.scoring import ScoredChunk as SC

            new_content = " ".join(ordered)
            filtered.append(SC(
                chunk={**sc.chunk, "content": new_content},
                tier=sc.tier,
                score=sc.score,
            ))
        else:
            filtered.append(sc)  # fallback: keep original

    return filtered
