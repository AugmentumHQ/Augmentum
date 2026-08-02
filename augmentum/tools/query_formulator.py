"""Deterministic query formulation for server-side search orchestration.

Transforms conversational user messages into optimized search queries
without an LLM call.  Uses location/timezone from settings and topic
hints from the intent classifier.
"""

from __future__ import annotations

import re
from datetime import datetime

from augmentum.config import settings
from augmentum.tools.intent import QueryIntent
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Conversational noise to strip
# ---------------------------------------------------------------------------

_STRIP_PREFIXES = [
    r"^(?:can you|could you|would you|will you)\s+(?:please\s+)?",
    r"^(?:please|hey|hi|okay|ok|so|well|actually)\s+",
    r"^(?:I want to|I need to|I'd like to|help me)\s+",
    r"^(?:search for|search|look up|look into|find me|find|google)\s+",
    r"^(?:show me|tell me about|tell me|give me)\s+",
    r"^(?:what do you know about|what can you tell me about)\s+",
    r"^(?:what (?:is|are|was|were) (?:the )?)",
    r"^(?:how (?:does|do|is|are|did|was) (?:the )?)",
    r"^(?:when (?:did|does|do|is|was|were) (?:the )?)",
    r"^(?:where (?:is|are|was|were|can I find) (?:the )?)",
]

_STRIP_PREFIX_RES = [re.compile(p, re.IGNORECASE) for p in _STRIP_PREFIXES]

# Trailing noise
_STRIP_SUFFIXES_RE = re.compile(
    r"\s*(?:please|thanks|thank you|thx|for me|if you can|would you|right now)\s*[?.!]*$",
    re.IGNORECASE,
)


def _clean_query(text: str) -> str:
    """Strip conversational fluff and normalize for search engines."""
    result = text.strip()
    # Strip voice input context note injected by _inject_voice_context().
    # It's bracketed: [This message was transcribed from voice input ...]
    result = re.sub(
        r"\[This message was transcribed from voice input[^\]]*\]",
        "",
        result,
    ).strip()
    for pat in _STRIP_PREFIX_RES:
        result = pat.sub("", result).strip()
    result = _STRIP_SUFFIXES_RE.sub("", result).strip()
    # Strip punctuation that hurts search quality
    result = result.rstrip(".,!;:?")
    # Normalize special characters
    result = re.sub(r"[""''`]", '"', result)  # smart quotes → standard
    result = re.sub(r"[–—]", "-", result)     # em/en dash → hyphen
    result = re.sub(r"\s+", " ", result)       # collapse whitespace
    return result.strip() if result.strip() else text.strip()


# ---------------------------------------------------------------------------
# Temporal qualification
# ---------------------------------------------------------------------------


def _add_temporal(query: str, intent: QueryIntent) -> str:
    """Add date qualifiers for time-sensitive queries."""
    if not intent.temporal:
        return query

    # Don't add if query already has a year
    if re.search(r"\b20\d{2}\b", query):
        return query

    try:
        tz = None
        if settings.timezone:
            from zoneinfo import ZoneInfo
            try:
                tz = ZoneInfo(settings.timezone)
            except Exception:
                # ZoneInfoNotFoundError on platforms without tzdata —
                # fall through to system local.
                pass
        now = datetime.now(tz) if tz else datetime.now()
        month_year = now.strftime("%B %Y")
        return f"{query} {month_year}"
    except Exception:
        return query


# ---------------------------------------------------------------------------
# Location injection
# ---------------------------------------------------------------------------

_LOCATION_KEYWORDS = frozenset({
    "near me", "nearby", "local", "around here", "in my area",
    "my city", "my town", "close to me", "in the area",
})


def _inject_location(query: str, intent: QueryIntent) -> str:
    """Replace 'near me' / 'local' with actual location if configured."""
    location = settings.location.strip()
    if not location:
        return query

    query_lower = query.lower()
    for kw in _LOCATION_KEYWORDS:
        if kw in query_lower:
            # Replace the location keyword with actual location
            result = re.sub(re.escape(kw), location, query, flags=re.IGNORECASE)
            return result

    # If query has location-dependent topics but no explicit location keyword,
    # append location for weather/news queries
    if intent.source_type == "fresh" and any(
        t in intent.topics for t in ("weather", "forecast", "news", "local")
    ):
        return f"{query} {location}"

    return query


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------

_COMPARE_RE = re.compile(
    r"^(.+?)\s+(?:vs\.?|versus|compared to|or|against)\s+(.+)$",
    re.IGNORECASE,
)


def _decompose(query: str, intent: QueryIntent) -> list[str]:
    """Split comparison queries into parallel searches."""
    m = _COMPARE_RE.match(query)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        # For data queries, search each item separately for specs
        if intent.source_type == "data":
            return [f"{a} specifications", f"{b} specifications"]
        return [f"{a} vs {b}"]
    return [query]


# ---------------------------------------------------------------------------
# Site-scoped variant
# ---------------------------------------------------------------------------


def _add_site_variant(queries: list[str], intent: QueryIntent) -> list[str]:
    """Add a site-scoped search variant for the best preferred source."""
    if not intent.site_hints:
        return queries

    # Take the first (highest quality) hint
    domain = intent.site_hints[0]
    base_query = queries[0]

    # Don't double up if already has site:
    if "site:" in base_query:
        return queries

    site_query = f"{base_query} site:{domain}"
    return queries + [site_query]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def formulate_queries(intent: QueryIntent, user_message: str) -> list[str]:
    """Transform a user message into optimized search queries.

    Returns 1-3 queries suitable for parallel execution against SearxNG.
    Uses settings.location and settings.timezone for geo/temporal grounding.
    """
    # Start with cleaned query
    cleaned = _clean_query(user_message)

    # Inject location
    cleaned = _inject_location(cleaned, intent)

    # Add temporal qualification
    cleaned = _add_temporal(cleaned, intent)

    # Decompose comparisons
    queries = _decompose(cleaned, intent)

    # Add site-scoped variant from preferred sources
    queries = _add_site_variant(queries, intent)

    # Cap at 3 queries
    queries = queries[:3]

    log.info(
        "queries_formulated",
        query_count=len(queries),
        location=settings.location or "(none)",
    )
    log.debug(
        "queries_formulated_content",
        original=user_message[:100],
        queries=queries,
    )
    return queries
