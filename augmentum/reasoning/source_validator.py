"""Post-draft source validation — flags unsourced claims in draft outputs.

After a draft step completes, this module compares each paragraph of the
draft against the search results that were collected earlier.  Paragraphs
whose embedding similarity to every source chunk falls below a threshold
are flagged as potentially unsourced.  If the ratio of unsourced
paragraphs exceeds a limit the overall validation fails, and a warning is
recorded for downstream review/respond steps to consider.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re

from augmentum.memory.embeddings import EmbeddingService

# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

@dataclasses.dataclass
class SourceValidationResult:
    """Outcome of comparing draft paragraphs against search sources."""

    is_valid: bool
    unsourced_ratio: float
    unsourced_paragraphs: list[str]  # first 100 chars of each
    sourced_count: int
    total_count: int


# ------------------------------------------------------------------
# Cosine similarity
# ------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ------------------------------------------------------------------
# Chunk splitting helpers
# ------------------------------------------------------------------

_SOURCE_DELIMITERS = re.compile(
    r"(?:^|\n)(?:## Source:|### Result|---)\s*\n",
    re.MULTILINE,
)


def _split_paragraphs(text: str, min_len: int = 30) -> list[str]:
    """Split text on blank lines and drop short fragments."""
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_len]


def _split_source_chunks(text: str, min_len: int = 20) -> list[str]:
    """Split search context into source chunks.

    Tries structured delimiters first (``## Source:``, ``### Result``,
    ``---``).  Falls back to double-newline paragraphs.
    """
    parts = _SOURCE_DELIMITERS.split(text)
    chunks = [p.strip() for p in parts if len(p.strip()) >= min_len]
    if chunks:
        return chunks
    # Fallback: plain paragraph splitting
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_len]


# ------------------------------------------------------------------
# Main validation
# ------------------------------------------------------------------

async def validate_draft_sources(
    draft_text: str,
    search_context: str,
    *,
    min_similarity: float = 0.45,
    max_unsourced_ratio: float = 0.5,
) -> SourceValidationResult:
    """Compare draft paragraphs against search sources.

    Parameters
    ----------
    draft_text:
        The full output of a draft step.
    search_context:
        Collected search results that the draft was based on.
    min_similarity:
        Minimum cosine similarity for a paragraph to be considered sourced.
    max_unsourced_ratio:
        Maximum fraction of unsourced paragraphs before validation fails.

    Returns
    -------
    SourceValidationResult
        Contains validity flag, ratio, and truncated unsourced paragraphs.
    """
    paragraphs = _split_paragraphs(draft_text)
    source_chunks = _split_source_chunks(search_context)

    # Early exit — nothing meaningful to validate
    if not paragraphs or not source_chunks:
        return SourceValidationResult(
            is_valid=True,
            unsourced_ratio=0.0,
            unsourced_paragraphs=[],
            sourced_count=len(paragraphs),
            total_count=len(paragraphs),
        )

    # Embed paragraphs as queries, source chunks as documents
    para_embeddings = await asyncio.to_thread(EmbeddingService.embed_queries, paragraphs)
    source_embeddings = await asyncio.to_thread(EmbeddingService.embed, source_chunks)

    unsourced: list[str] = []
    sourced_count = 0

    for i, para_vec in enumerate(para_embeddings):
        max_sim = max(
            _cosine_similarity(para_vec, src_vec)
            for src_vec in source_embeddings
        )
        if max_sim >= min_similarity:
            sourced_count += 1
        else:
            unsourced.append(paragraphs[i][:100])

    total = len(paragraphs)
    unsourced_ratio = len(unsourced) / total if total > 0 else 0.0
    is_valid = unsourced_ratio <= max_unsourced_ratio

    return SourceValidationResult(
        is_valid=is_valid,
        unsourced_ratio=unsourced_ratio,
        unsourced_paragraphs=unsourced,
        sourced_count=sourced_count,
        total_count=total,
    )
