"""Chunk deduplication via word n-gram overlap.

Uses word 3-grams instead of character n-grams to avoid false positives
on templated content where structure is similar but facts differ.
"Module-025 budget $625,000" vs "Module-010 budget $250,000" share
character 4-grams but NOT word 3-grams.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from augmentum.documents.scoring import ScoredChunk


def deduplicate(
    chunks: list[ScoredChunk],
    overlap_threshold: float = 0.7,
    ngram_size: int = 3,
) -> list[ScoredChunk]:
    """Remove chunks with >overlap_threshold word-n-gram overlap with a
    higher-scored chunk already accepted.

    Uses word 3-grams for content-aware dedup. Catches true duplicates
    (overlapping chunks with identical text) while preserving templated
    sections that share structure but contain different facts.

    Threshold raised from 0.6 to 0.7 to reduce false positives on
    structurally similar but factually distinct content.
    """
    if len(chunks) <= 1:
        return chunks

    accepted: list[ScoredChunk] = []
    accepted_ngrams: list[set[tuple]] = []

    for sc in chunks:
        content = sc.chunk.get("content", "")
        ngrams = _word_ngrams(content, ngram_size)

        is_duplicate = False
        for prev_ngrams in accepted_ngrams:
            if not ngrams or not prev_ngrams:
                continue
            overlap = len(ngrams & prev_ngrams) / len(ngrams)
            if overlap > overlap_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            accepted.append(sc)
            accepted_ngrams.append(ngrams)

    return accepted


def _word_ngrams(text: str, n: int = 3) -> set[tuple]:
    """Generate word n-gram set from text.

    Words are lowercased and include numbers/special terms so that
    "$625,000" and "Module-025" are preserved as distinct tokens.
    """
    words = re.findall(r'\S+', text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}
