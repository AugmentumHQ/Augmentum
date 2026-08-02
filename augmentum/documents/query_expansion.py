"""Embedding-based query expansion — zero-LLM synonym discovery.

EXPERIMENTAL: Not integrated into the production pipeline. Used only
by the benchmark harness (test_rag_pipeline_bench.py) for A/B testing.
Benchmark showed +3pp precision on cross-reference but -6pp on specific
queries. Not recommended for production without conditional application.

Finds semantically similar terms from the corpus vocabulary to bridge
vocabulary gaps (e.g., user says "salary", document says "compensation").
Uses the same embedding model as the rest of the pipeline.

Reference: arxiv.org/abs/2509.07794 (Query Expansion Survey)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Module-level vocabulary index (built once per corpus)
_vocab_embeddings: dict[str, list[float]] = {}
_vocab_ready = False


def build_vocab_index(chunks: list[str], max_terms: int = 2000) -> int:
    """Build a term→embedding index from corpus chunk text.

    Call once after ingesting documents. Extracts the most frequent
    content words and embeds them for similarity lookup.

    Returns the number of terms indexed.
    """
    global _vocab_embeddings, _vocab_ready  # noqa: PLW0603

    from augmentum.documents.store import _STOP_WORDS

    # Extract word frequencies from all chunks
    word_freq: dict[str, int] = {}
    for chunk in chunks:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', chunk.lower())
        for w in words:
            if w not in _STOP_WORDS:
                word_freq[w] = word_freq.get(w, 0) + 1

    # Take top N most frequent terms (these are the "document vocabulary")
    sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    terms = [t for t, _ in sorted_terms[:max_terms]]

    if not terms:
        _vocab_ready = False
        return 0

    # Embed all terms as documents (they represent document vocabulary)
    from augmentum.memory.embeddings import EmbeddingService

    embeddings = EmbeddingService.embed(terms)
    _vocab_embeddings = dict(zip(terms, embeddings))
    _vocab_ready = True

    log.debug("vocab_index_built", terms=len(terms))
    return len(terms)


def expand_query_terms(
    query_words: list[str],
    top_k: int = 3,
    min_similarity: float = 0.5,
) -> list[str]:
    """Find semantically similar corpus terms for each query word.

    Returns a flat list of expansion terms (deduplicated, excluding
    the original query words).
    """
    if not _vocab_ready or not _vocab_embeddings:
        return []

    from augmentum.memory.embeddings import EmbeddingService

    # Embed query words
    query_embeddings = EmbeddingService.embed_queries(query_words)

    # Pre-compute vocab as lists for dot product
    vocab_terms = list(_vocab_embeddings.keys())
    vocab_vecs = list(_vocab_embeddings.values())

    expansions: set[str] = set()
    query_set = {w.lower() for w in query_words}

    for q_word, q_vec in zip(query_words, query_embeddings):
        # Cosine similarity (vectors are already normalized by nomic)
        scores = []
        for v_term, v_vec in zip(vocab_terms, vocab_vecs):
            if v_term == q_word.lower():
                continue
            dot = sum(a * b for a, b in zip(q_vec, v_vec))
            scores.append((v_term, dot))

        scores.sort(key=lambda x: x[1], reverse=True)

        for term, score in scores[:top_k]:
            if score >= min_similarity and term not in query_set:
                expansions.add(term)

    return list(expansions)


def reset() -> None:
    """Reset the vocab index (for testing)."""
    global _vocab_embeddings, _vocab_ready  # noqa: PLW0603
    _vocab_embeddings = {}
    _vocab_ready = False
