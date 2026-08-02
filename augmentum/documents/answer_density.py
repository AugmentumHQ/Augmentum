"""Answer density scoring — pre-score chunks by information content.

EXPERIMENTAL: Not integrated into the production pipeline. Benchmark
showed zero measurable impact — avg density 0.16, boost too small to
re-order chunks already ranked by RRF. Kept for future investigation.

Chunks with high concentrations of specific facts (numbers, named entities,
technical terms, definitions) are more likely to contain actual answers
vs. filler prose. Score is computed at ingest time and stored per chunk.

Novel technique: standard RAG scores by query-document similarity only.
This adds an intrinsic "how much information is in this chunk" signal
that's query-independent.
"""

from __future__ import annotations

import re


# Patterns that indicate factual/specific content
_NUMBER_RE = re.compile(
    r'\$[\d,]+(?:\.\d+)?'      # dollar amounts: $145,000
    r'|\b\d{1,3}(?:,\d{3})+\b' # large numbers: 2,847
    r'|\b\d+(?:\.\d+)?%'        # percentages: 83.3%
    r'|\b\d+(?:\.\d+)?\s*(?:month|year|day|hour|week|minute)s?\b'  # durations
    r'|\b\d+(?:\.\d+)?\s*(?:MB|GB|TB|KB|Hz|kHz|MHz|GHz|ms|px|dpi|mAh)\b'  # tech units
    r'|\b\d{4}-\d{2}-\d{2}\b'  # dates: 2026-03-23
    r'|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
)

# Parenthetical definitions: ($145,000), (AAA), ("Company")
_DEFINITION_RE = re.compile(
    r'\([^)]{2,60}\)'
)

# Quoted terms: "Base Compensation", 'non-compete'
_QUOTED_RE = re.compile(
    r'["\u201c][^"\u201d]{2,50}["\u201d]'
    r"|'[^']{2,50}'"
)

# Named entities: capitalized multi-word phrases (not sentence starts)
_NAMED_ENTITY_RE = re.compile(
    r'(?<=[.!?]\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'  # after sentence boundary
    r'|(?<=\n)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'       # after newline
    r'|[A-Z]{2,}(?:\s+[A-Z]{2,})*'                   # acronyms: AAA, HMAC-SHA256
)

# Technical terms: words with special characters common in specs
_TECHNICAL_RE = re.compile(
    r'\b(?:v\d+\.\d+|[A-Z][a-z]*-\d+|https?://\S+)\b'  # versions, model names, URLs
    r'|/api/\S+'                                           # API paths
    r'|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'                   # CamelCase
)


def compute_density(text: str) -> float:
    """Compute information density score for a chunk of text.

    Returns a float in [0, 1] range:
    - 0.0 = pure filler/prose with no specific facts
    - 0.5 = moderate density (typical informative paragraph)
    - 1.0 = extremely dense (table rows, spec lists, legal clauses with numbers)

    The score is the ratio of "information tokens" (numbers, entities,
    definitions, technical terms) to total words, capped at 1.0.
    """
    if not text or len(text) < 10:
        return 0.0

    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return 0.0

    # Count information markers
    numbers = len(_NUMBER_RE.findall(text))
    definitions = len(_DEFINITION_RE.findall(text))
    quoted = len(_QUOTED_RE.findall(text))
    named_entities = len(_NAMED_ENTITY_RE.findall(text))
    technical = len(_TECHNICAL_RE.findall(text))

    # Weight different markers (numbers/amounts are strongest signal)
    info_score = (
        numbers * 3.0
        + definitions * 2.0
        + quoted * 1.5
        + named_entities * 1.0
        + technical * 1.5
    )

    # Normalize by word count, cap at 1.0
    density = min(1.0, info_score / total_words)

    return round(density, 4)


def boost_by_density(
    score: float,
    density: float,
    weight: float = 0.5,
) -> float:
    """Apply density boost to a retrieval score.

    score * (1 + density * weight)

    With default weight=0.5:
    - density=0.0 (filler): no change
    - density=0.3 (moderate): +15% boost
    - density=0.8 (very dense): +40% boost
    """
    return score * (1.0 + density * weight)
