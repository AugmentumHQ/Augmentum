"""Query analyzer — classifies and rewrites queries before document search."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

_VALID_STRATEGIES = frozenset({"direct", "rewrite", "decompose", "skip"})

_SYSTEM_PROMPT = """\
You classify document search queries. Given the user's message and \
available documents, decide the optimal search strategy.

Strategies:
- "direct": query has specific terms likely to appear in the document. \
Return the query as-is or with minor cleanup.
- "rewrite": query is vague or uses informal/different vocabulary than \
likely document language. Rewrite using formal terms, synonyms, and \
likely document vocabulary. Aim for 5-15 words.
- "decompose": query asks about multiple distinct topics. Split into \
2-3 focused sub-queries (max 3). Each should be self-contained.
- "skip": no document search needed. Greetings, meta-questions \
("can you rephrase that?"), continuations ("go on"), thanks.

Available documents: {filenames}

Return ONLY valid JSON:
{{"strategy": "...", "queries": [...], "reason": "...", "confidence": 0.0-1.0}}"""

_MAX_QUERIES = 3
_MAX_QUERY_LEN = 200
_CACHE_TTL = 60.0


@dataclass
class QueryAnalysis:
    strategy: str = "direct"
    queries: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 1.0


# Module-level cache: hash -> (result, timestamp)
_analysis_cache: dict[str, tuple[QueryAnalysis, float]] = {}


def _cache_key(query: str, doc_names: list[str]) -> str:
    raw = f"{query}|{'|'.join(sorted(doc_names))}"
    return hashlib.md5(raw.encode()).hexdigest()


def _parse_analysis_response(raw: str, original_query: str) -> QueryAnalysis:
    """Parse LLM response with 7-step fallback for resilience."""
    data = None

    # Step 1: Try direct JSON parse
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # Step 2: Extract from markdown code fence
    if data is None and raw:
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except (json.JSONDecodeError, TypeError):
                pass

    # Step 3: Fall back to direct strategy
    if not isinstance(data, dict):
        return QueryAnalysis(
            strategy="direct", queries=[original_query],
            reason="parse_fallback", confidence=1.0,
        )

    # Step 4: Validate strategy
    strategy = data.get("strategy", "direct")
    if strategy not in _VALID_STRATEGIES:
        strategy = "direct"

    # Step 5: Validate queries
    queries = data.get("queries", [])
    if not isinstance(queries, list):
        queries = [original_query]
    queries = [str(q) for q in queries if isinstance(q, str) and q.strip()]

    # Empty queries: populate for non-skip, allow for skip
    if not queries and strategy != "skip":
        queries = [original_query]

    # Step 6: Cap at max
    queries = queries[:_MAX_QUERIES]

    # Step 7: Cap query length
    queries = [q[:_MAX_QUERY_LEN] for q in queries]

    reason = str(data.get("reason", ""))
    confidence = float(data.get("confidence", 0.8))

    return QueryAnalysis(
        strategy=strategy, queries=queries,
        reason=reason, confidence=confidence,
    )


class QueryAnalyzer:
    """Classifies queries and optionally rewrites them before search."""

    def __init__(self, backend: object | None = None) -> None:
        self._backend = backend

    async def analyze(
        self,
        query: str,
        doc_names: list[str],
        has_full_docs: bool = False,
    ) -> QueryAnalysis:
        # Short-circuit: all docs in full mode, no search-mode docs
        if has_full_docs and not doc_names:
            return QueryAnalysis(
                strategy="skip", queries=[],
                reason="all_docs_full_mode", confidence=1.0,
            )

        # No backend or disabled
        if not self._backend or not settings.document_rag_query_analysis:
            return QueryAnalysis(
                strategy="direct", queries=[query],
                reason="no_backend", confidence=1.0,
            )

        # Cache check
        key = _cache_key(query, doc_names)
        cached = _analysis_cache.get(key)
        if cached:
            result, ts = cached
            if time.time() - ts < _CACHE_TTL:
                return result
            else:
                del _analysis_cache[key]

        # LLM call with timeout
        try:
            result = await asyncio.wait_for(
                self._llm_classify(query, doc_names),
                timeout=settings.document_rag_query_analysis_timeout,
            )
        except TimeoutError:
            log.warning("query_analyzer_timeout", query=query[:60])
            result = QueryAnalysis(
                strategy="direct", queries=[query],
                reason="timeout", confidence=1.0,
            )
        except Exception as exc:
            log.warning("query_analyzer_failed", error=str(exc)[:200])
            result = QueryAnalysis(
                strategy="direct", queries=[query],
                reason="error", confidence=1.0,
            )

        # Cache result
        _analysis_cache[key] = (result, time.time())
        return result

    async def _llm_classify(
        self, query: str, doc_names: list[str],
    ) -> QueryAnalysis:
        filenames = ", ".join(doc_names) if doc_names else "(none)"
        system = _SYSTEM_PROMPT.format(filenames=filenames)

        from augmentum.models.base import Message

        response = await self._backend.chat(
            messages=[Message(role="user", content=query)],
            system_prompt=system,
            temperature=0.0,
            max_tokens=150,
        )

        raw = response.content.strip() if response and response.content else ""
        log.debug("query_analyzer_raw", raw=raw[:200].encode("ascii", "replace").decode())
        return _parse_analysis_response(raw, query)
