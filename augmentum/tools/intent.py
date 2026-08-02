"""Zero-cost intent classification for server-side tool orchestration.

Analyzes user messages with keyword/pattern heuristics to determine what
tools are needed and what kind of sources to target — without an LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from augmentum.tools.preferred_sources import GOOD, get_sources_by_category
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class QueryIntent:
    """Classified intent for a user query."""

    action: str = "none"        # search, fetch_url, calculate, convert, code, files, none
    source_type: str = "fresh"  # fresh, reference, data
    topics: list[str] = field(default_factory=list)
    temporal: bool = False      # query wants current/recent information
    url: str | None = None      # extracted URL
    math_expr: str | None = None  # extracted math expression
    confidence: float = 0.0
    site_hints: list[str] = field(default_factory=list)  # preferred domains for site-scoped queries


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# URL detection
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

# Math expression (basic arithmetic, parens, common functions)
_MATH_RE = re.compile(
    r"([\d]+(?:\.\d+)?\s*[\+\-\*/\^\%]\s*[\d]+(?:\.\d+)?(?:\s*[\+\-\*/\^\%]\s*[\d]+(?:\.\d+)?)*)",
)

# Conversational prefixes to ignore during classification
_CONV_PREFIX_RE = re.compile(
    r"^(?:can you|could you|please|hey|hi|okay|ok|so|well|actually|"
    r"I want to|I need to|I'd like to|help me|show me|tell me|"
    r"find me|look up|search for|google|look into)\s+",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Temporal signal keywords
# ---------------------------------------------------------------------------

_TEMPORAL_KEYWORDS = frozenset({
    "today", "tonight", "tomorrow", "yesterday", "this week", "this month",
    "this year", "right now", "currently", "current", "latest", "recent",
    "breaking", "live", "real-time", "realtime", "update", "updates",
    "new", "newest", "now", "2025", "2026", "2027",
})

# ---------------------------------------------------------------------------
# Source type keyword mappings
# ---------------------------------------------------------------------------

# fresh: current events, live data, weather, scores
_FRESH_KEYWORDS = frozenset({
    "news", "weather", "forecast", "score", "scores", "stock", "stocks",
    "price", "prices", "market", "markets", "election", "results",
    "live", "happening", "today", "tonight", "game", "games",
    "match", "matches", "standings", "rankings", "trending", "viral",
    "release", "released", "announcement", "announced", "update",
    "outbreak", "earthquake", "hurricane", "storm", "fire", "flood",
    "traffic", "flight", "flights", "deal", "deals", "sale",
})

# reference: encyclopedic, historical, definitional
_REFERENCE_KEYWORDS = frozenset({
    "who is", "who was", "who were", "what is", "what are", "what was",
    "how does", "how do", "how is", "how did", "how was",
    "define", "definition", "meaning", "explain", "explanation",
    "history", "historical", "biography", "invented", "discovered",
    "founded", "origin", "origins", "theory", "theorem",
    "difference between", "why is", "why do", "why does", "why did",
    "overview", "summary", "introduction",
})

# data: statistics, specs, benchmarks, measurements
_DATA_KEYWORDS = frozenset({
    "statistics", "stats", "data", "dataset", "benchmark", "benchmarks",
    "specifications", "specs", "spec", "compare", "comparison", "versus",
    "vs", "population", "gdp", "rate", "rates", "percentage",
    "calories", "nutrition", "ingredients", "dimensions", "capacity",
    "performance", "speed", "size", "weight", "height", "length",
    "temperature", "distance", "cost", "salary", "salaries", "income",
    "ranking", "ranked", "top 10", "top 5", "list of",
})

# calculate: direct math requests
_CALC_KEYWORDS = frozenset({
    "calculate", "compute", "solve", "what is", "how much is",
    "sum of", "product of", "square root", "factorial",
    "percent of", "percentage of", "divided by", "multiplied by",
})

# convert: unit conversion
_CONVERT_RE = re.compile(
    r"(?:convert|how many|what is)\s+[\d.]+\s*\w+\s+(?:to|in|into)\s+\w+",
    re.IGNORECASE,
)

# datetime: time/date queries
_DATETIME_RE = re.compile(
    r"(?:what\s+(?:time|day|date)\s+is|current\s+(?:time|date|day)|"
    r"today'?s?\s+date|what'?s?\s+the\s+(?:time|date|day)|"
    r"what\s+is\s+the\s+(?:time|date|day)|what\s+is\s+today|"
    r"how\s+many\s+days?\s+(?:until|since|between|from|to)|"
    r"what\s+(?:is|was)\s+the\s+day\s+(?:of|on))",
    re.IGNORECASE,
)

# build_app: application/website generation
_BUILD_APP_RE = re.compile(
    r"\b(?:build|create|make|generate)\s+(?:me\s+)?(?:a|an|the)?\s*"
    r"(?:[\w\s]{1,30}\s+)?(?:app|application|website|web\s*app|web\s*page|"
    r"site|tool|dashboard|game|calculator|form|page)\b",
    re.IGNORECASE,
)

# code: run code
_CODE_KEYWORDS = frozenset({
    "run this code", "execute this", "run this script", "run this program",
    "run the code", "execute the code",
})

# files: user's own files / artifacts
_FILE_KEYWORDS = (
    "my file", "my files", "my document", "my documents", "my image",
    "my images", "my photo", "my photos", "my picture", "my pictures",
    "my artifact", "my artifacts", "what did i create", "what have i created",
    "find my", "show my files", "show my documents", "show my images",
    "uploaded", "my uploads", "what files", "list my", "my ebook",
    "my ebooks", "my spreadsheet", "my presentation",
)

# ---------------------------------------------------------------------------
# Topic matching against preferred_sources categories
# ---------------------------------------------------------------------------

# Map common query words to preferred_sources category names
_TOPIC_KEYWORDS: dict[str, list[str]] = {
    # Weather & environment
    "weather": ["weather", "forecast", "climate"],
    "forecast": ["weather", "forecast"],
    "climate": ["climate", "environment"],
    "earthquake": ["earthquakes", "geology"],
    "hurricane": ["hurricane", "weather"],
    # News & current events
    "news": ["news", "world", "politics"],
    "politics": ["politics", "government", "legislation"],
    "election": ["politics", "government"],
    # Finance & markets
    "stock": ["stocks", "finance", "markets"],
    "stocks": ["stocks", "finance", "markets"],
    "crypto": ["cryptocurrency", "finance"],
    "bitcoin": ["cryptocurrency", "finance"],
    "market": ["markets", "finance", "stocks"],
    "price": ["finance", "markets"],
    # Sports
    "nfl": ["nfl", "football", "sports", "scores"],
    "nba": ["nba", "basketball", "sports", "scores"],
    "mlb": ["mlb", "baseball", "sports", "scores"],
    "football": ["football", "sports", "scores"],
    "basketball": ["basketball", "sports", "scores"],
    "baseball": ["baseball", "sports", "scores"],
    "soccer": ["soccer", "sports", "scores"],
    "hockey": ["hockey", "sports", "scores"],
    "scores": ["scores", "sports"],
    "sports": ["sports", "scores"],
    # Technology & hardware
    "python": ["python", "programming"],
    "javascript": ["javascript", "programming"],
    "programming": ["programming", "development"],
    "gpu": ["gpu", "benchmarks", "hardware", "specs"],
    "cpu": ["cpu", "benchmarks", "hardware", "specs"],
    "processor": ["cpu", "hardware", "specs"],
    "graphics card": ["gpu", "hardware", "specs"],
    "ram": ["hardware", "specs"],
    "ssd": ["hardware", "specs"],
    "motherboard": ["hardware", "specs"],
    "laptop": ["technology", "specs", "hardware"],
    "phone": ["mobile", "phones", "specs"],
    "monitor": ["monitors", "specs", "technology"],
    "benchmark": ["benchmarks", "hardware", "specs"],
    "benchmarks": ["benchmarks", "hardware", "specs"],
    "specs": ["specs", "hardware"],
    "specifications": ["specs", "hardware"],
    "review": ["reviews", "technology"],
    "reviews": ["reviews", "technology"],
    "software": ["technology", "software"],
    "ryzen": ["cpu", "hardware", "specs", "benchmarks"],
    "intel": ["cpu", "hardware", "specs", "benchmarks"],
    "nvidia": ["gpu", "hardware", "specs", "benchmarks"],
    "radeon": ["gpu", "hardware", "specs", "benchmarks"],
    "geforce": ["gpu", "hardware", "specs", "benchmarks"],
    "rtx": ["gpu", "hardware", "specs", "benchmarks"],
    # Science & health
    "health": ["health", "medical"],
    "medical": ["medical", "health"],
    "science": ["science", "research"],
    "physics": ["physics", "science"],
    "chemistry": ["chemistry", "science"],
    "biology": ["biology", "science"],
    # Food & nutrition
    "recipe": ["recipe", "cooking", "food"],
    "cooking": ["cooking", "recipe", "food"],
    "calories": ["nutrition", "food", "health"],
    "nutrition": ["nutrition", "food", "health"],
    # Entertainment
    "movie": ["movies", "entertainment"],
    "movies": ["movies", "entertainment"],
    "music": ["music", "entertainment"],
    "game": ["games", "entertainment"],
    "games": ["games", "entertainment"],
}


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


def classify_intent(message: str) -> QueryIntent:
    """Classify a user message into a QueryIntent using heuristics.

    Zero LLM cost. Returns an intent with confidence 0.0-1.0.
    Confidence < 0.3 means the query is ambiguous — caller should
    fall back to the LLM-driven tool-calling pipeline.
    """
    intent = QueryIntent()
    original = message.strip()
    if not original:
        return intent

    # Strip conversational prefix for analysis
    cleaned = _CONV_PREFIX_RE.sub("", original).strip()
    lower = cleaned.lower()
    original_lower = original.lower()

    # --- URL detection (highest priority) ---
    url_match = _URL_RE.search(original)
    if url_match:
        intent.action = "fetch_url"
        intent.url = url_match.group(0)
        intent.confidence = 0.9
        return intent

    # --- Date/time detection ---
    if _DATETIME_RE.search(original):
        intent.action = "datetime"
        intent.confidence = 0.85
        return intent

    # --- Build application detection ---
    if _BUILD_APP_RE.search(original):
        intent.action = "build_app"
        intent.confidence = 0.85
        return intent

    # --- File search detection (user's own files / artifacts) ---
    if any(kw in original_lower for kw in _FILE_KEYWORDS):
        intent.action = "files"
        intent.confidence = 0.85
        return intent

    # --- Code block detection ---
    if "```" in original:
        for kw in _CODE_KEYWORDS:
            if kw in original_lower:
                intent.action = "code"
                intent.confidence = 0.8
                return intent

    # --- Unit conversion detection ---
    if _CONVERT_RE.search(original):
        intent.action = "convert"
        intent.confidence = 0.8
        return intent

    # --- Math expression detection ---
    math_match = _MATH_RE.search(original)
    has_calc_keyword = any(kw in original_lower for kw in _CALC_KEYWORDS)
    if math_match:
        intent.action = "calculate"
        intent.math_expr = math_match.group(1).strip()
        intent.confidence = 0.85 if has_calc_keyword else 0.7
        return intent
    if has_calc_keyword and re.search(r"\d", original):
        # "Calculate 15% of 200" — keyword + number but no operator match
        intent.action = "calculate"
        intent.math_expr = cleaned
        intent.confidence = 0.6
        return intent

    # --- Search intent detection ---
    # Score each source type
    fresh_score = 0
    ref_score = 0
    data_score = 0

    # Check temporal signals
    for kw in _TEMPORAL_KEYWORDS:
        if kw in original_lower:
            intent.temporal = True
            fresh_score += 2
            break

    # Check fresh keywords
    for kw in _FRESH_KEYWORDS:
        if kw in original_lower:
            fresh_score += 3

    # Check reference keywords (multi-word patterns)
    for kw in _REFERENCE_KEYWORDS:
        if kw in original_lower:
            ref_score += 3

    # Check data keywords
    for kw in _DATA_KEYWORDS:
        if kw in original_lower:
            data_score += 3

    # Check question patterns
    if original.rstrip().endswith("?"):
        # Questions generally indicate search intent
        fresh_score += 1
        ref_score += 1

    # Determine dominant source type
    max_score = max(fresh_score, ref_score, data_score)
    total_score = fresh_score + ref_score + data_score

    if total_score == 0:
        # No search signals detected — could be a greeting, creative writing, etc.
        intent.action = "none"
        intent.confidence = 0.1
        return intent

    if fresh_score == max_score:
        intent.source_type = "fresh"
    elif ref_score == max_score:
        intent.source_type = "reference"
    else:
        intent.source_type = "data"

    intent.action = "search"

    # Confidence based on signal strength
    if max_score >= 6:
        intent.confidence = 0.85
    elif max_score >= 3:
        intent.confidence = 0.6
    else:
        intent.confidence = 0.35

    # --- Topic matching for site hints ---
    topics: set[str] = set()
    words = set(re.findall(r"\b\w+\b", original_lower))
    for word in words:
        if word in _TOPIC_KEYWORDS:
            topics.update(_TOPIC_KEYWORDS[word])

    # Resolve ambiguous topic overlaps — when tech + sports keywords coexist,
    # prefer the more specific context
    tech_words = words & {"gpu", "cpu", "benchmark", "benchmarks", "ram", "fps", "performance", "spec", "specs"}
    sport_words = words & {"nfl", "nba", "mlb", "football", "basketball", "baseball", "soccer", "hockey", "game", "match"}
    if tech_words and not sport_words:
        topics -= {"sports", "scores"}
    elif sport_words and not tech_words:
        topics -= {"benchmarks", "technology"}

    intent.topics = list(topics)

    # Find EXCELLENT/GOOD domains for matched topics
    if topics:
        site_hints: list[str] = []
        seen_domains: set[str] = set()
        for topic in topics:
            for domain, info in get_sources_by_category(topic):
                if info.quality >= GOOD and domain not in seen_domains:
                    site_hints.append(domain)
                    seen_domains.add(domain)
                if len(site_hints) >= 3:
                    break
            if len(site_hints) >= 3:
                break
        intent.site_hints = site_hints

    log.info(
        "intent_classified",
        action=intent.action,
        source_type=intent.source_type,
        confidence=round(intent.confidence, 2),
        temporal=intent.temporal,
        topics=intent.topics[:5],
        site_hints=intent.site_hints[:3],
    )
    return intent
