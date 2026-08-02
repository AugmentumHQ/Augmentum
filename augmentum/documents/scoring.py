"""Document RAG scoring pipeline — score gate, cliff detection, budget."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ScoredChunk:
    chunk: dict           # original result dict from search
    tier: str             # "high" | "uncertain" | "irrelevant"
    score: float          # the score used for gating


def query_quality(query: str) -> float:
    """Score query signal strength from 0.0 (no signal) to 1.0 (strong signal).

    Queries with mostly stop words/pronouns ("tell me more about that")
    have low quality. Queries with specific content words ("termination
    notice period") have high quality.

    Used to dampen score gate when query has insufficient signal —
    prevents meaningless queries from getting "high" confidence results
    when the reranker is not available.
    """
    from augmentum.documents.store import _STOP_WORDS

    words = re.findall(r'\w+', query.lower())
    if not words:
        return 0.0
    content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    # Ratio of content words to total words, capped at 1.0
    ratio = len(content_words) / len(words)
    # Bonus for having multiple content words (more specific query)
    count_bonus = min(1.0, len(content_words) / 3.0)
    return min(1.0, (ratio + count_bonus) / 2.0)


def score_gate(
    results: list[dict],
    reranker_enabled: bool,
    dual_source: bool = True,
    query: str = "",
) -> list[ScoredChunk]:
    """Classify results into confidence tiers.

    Thresholds calibrated separately for:
    - Reranker scores (0-1 range, from cross-encoder)
    - Dual-source RRF (both vec + FTS contributed, ~0.001-0.03 range)
    - Single-source RRF (only vec or FTS, max ~0.016)

    RRF formula: score = sum(1/(k + rank + 1)) with k=60.
    - Rank 0 single source: 1/61 = 0.0164
    - Rank 0 both sources: 2/61 = 0.0328
    """
    if reranker_enabled:
        # Sigmoid-normalized cross-encoder scores:
        #   > 0.73 = raw > 1.0 (strongly relevant)
        #   0.50 = raw 0.0 (neutral)
        #   < 0.27 = raw < -1.0 (irrelevant)
        high_threshold = 0.65
        uncertain_threshold = 0.45
    elif dual_source:
        high_threshold = 0.025
        uncertain_threshold = 0.012
    else:
        high_threshold = 0.015
        uncertain_threshold = 0.008

    # Query quality dampening (without reranker only):
    # Low-quality queries ("tell me more", "how long is it?") should
    # not get "high" confidence results since RRF scores are positional.
    # When reranker is enabled, it handles this via semantic scoring.
    quality_dampen = 1.0
    if not reranker_enabled and query:
        qq = query_quality(query)
        if qq < 0.3:
            # Very low quality: demote everything to uncertain/irrelevant
            quality_dampen = 0.3
        elif qq < 0.5:
            # Low quality: shift thresholds up (harder to reach "high")
            quality_dampen = 0.6

    scored = []
    for r in results:
        s = r.get("score", 0) * quality_dampen
        if s >= high_threshold:
            tier = "high"
        elif s >= uncertain_threshold:
            tier = "uncertain"
        else:
            tier = "irrelevant"
        scored.append(ScoredChunk(chunk=r, tier=tier, score=s))

    return scored


def cliff_detect(
    scored: list[ScoredChunk],
    cliff_ratio: float = 0.3,
    max_results: int = 10,
) -> list[ScoredChunk]:
    """Drop chunks where score < top_score * cliff_ratio.

    cliff_ratio: minimum relative relevance vs best result. 0.3 = must
    score at least 30% of the top result.

    Defensive sort ensures correctness regardless of input order.
    """
    if not scored:
        return []

    scored = sorted(scored, key=lambda s: s.score, reverse=True)

    viable = [s for s in scored if s.tier != "irrelevant"]
    if not viable:
        return []

    top_score = viable[0].score
    threshold = top_score * cliff_ratio

    clipped = []
    for s in viable:
        if s.score < threshold:
            break
        clipped.append(s)
        if len(clipped) >= max_results:
            break

    return clipped


_CHARS_PER_TOKEN = 4


def apply_budget(
    chunks: list[ScoredChunk],
    max_tokens: int = 1500,
) -> list[ScoredChunk]:
    """Pack chunks in score order until token budget exhausted.

    Last chunk truncated at sentence boundary if it would exceed budget.
    Uses ~4 chars/token approximation.
    """
    max_chars = max_tokens * _CHARS_PER_TOKEN
    budgeted: list[ScoredChunk] = []
    chars_used = 0

    for sc in chunks:
        content = sc.chunk.get("content", "")
        chunk_chars = len(content)

        if chars_used + chunk_chars <= max_chars:
            budgeted.append(sc)
            chars_used += chunk_chars
        else:
            remaining = max_chars - chars_used
            if remaining > 100:
                truncated = _truncate_at_sentence(content, remaining)
                if truncated:
                    sc_copy = ScoredChunk(
                        chunk={**sc.chunk, "content": truncated},
                        tier=sc.tier,
                        score=sc.score,
                    )
                    budgeted.append(sc_copy)
            break

    return budgeted


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at the last sentence boundary within max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
        idx = truncated.rfind(sep)
        if idx > max_chars * 0.5:
            return truncated[:idx + 1].rstrip()
    idx = truncated.rfind(" ")
    if idx > max_chars * 0.5:
        return truncated[:idx].rstrip() + "..."
    return truncated.rstrip() + "..."


def determine_sufficiency(
    chunks: list[ScoredChunk],
    strategy: str = "direct",
    sub_query_results: list[list[dict]] | None = None,
) -> str:
    """Determine context sufficiency from chunk confidence tiers.

    - "sufficient": at least one high-confidence chunk
    - "partial": only uncertain-tier chunks passed
    - "none": no chunks passed
    """
    if not chunks:
        return "none"

    has_high = any(sc.tier == "high" for sc in chunks)

    # EXPERIMENTAL: decompose-aware check (Phase 2, behind flag)
    if strategy == "decompose" and sub_query_results:
        for sq_results in sub_query_results:
            if not sq_results:
                return "partial"

    if has_high:
        return "sufficient"
    return "partial"
