"""Browse tab API routes — search, fetch, image proxy, AI analysis, save."""

from __future__ import annotations

import asyncio
import json
import math
import re
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from augmentum.config import settings
from augmentum.models.base import InternalChatRequest, Message
from augmentum.proxy.reputation import (
    _EMBEDDABLE_VIDEO_DOMAINS,
    _get_domain_scores,
    _rank_results,
    _seed_preferred_sources,
    _update_reputation,
)
from augmentum.utils.logging import get_logger
from augmentum.utils.safe_http import SafeHttpClient, SafeHttpError

log = get_logger(__name__)

router = APIRouter(prefix="/api/browse", tags=["browse"])


# Shared SSRF-safe client for fetch + image proxy
_safe_client = SafeHttpClient()
_image_client = SafeHttpClient(max_response_size=10_485_760)  # 10 MB for images

# Realistic browser headers — avoids basic bot detection
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
}

# Minimum extracted chars before we consider a fetch "thin" and try fallbacks
_MIN_CONTENT_CHARS = 200

# `site:X` operator in a query — used to enforce the UI's provider-chip
# scope by URL hostname regardless of whether each SearXNG engine honors
# the operator itself.
_SITE_OPERATOR_RE = re.compile(r"\bsite:(\S+)", re.IGNORECASE)

# Accepted values for the SearXNG `time_range` query parameter. Engines
# that don't implement time_range_support silently ignore the param —
# we always forward it when set, never validate per-engine.
_VALID_TIME_RANGES = frozenset({"day", "week", "month", "year"})

# Accepted values for the UI's sort dropdown. Maps to post-rerank sort
# behaviour in `browse_search` — the reranker still does its tier work
# first (so embeddable-video boost still applies in the videos category),
# then sort_by overrides ordering at the end.
_VALID_SORTS = frozenset({"date", "duration_asc", "duration_desc"})

# Duration bucket cutoffs (seconds). Match YouTube's filter labels so
# users coming from the YouTube UI get the same mental model.
_DURATION_BUCKETS: dict[str, tuple[int, int]] = {
    "short": (0, 240),         # < 4 min
    "medium": (240, 1200),     # 4-20 min
    "long": (1200, 10**9),     # > 20 min
}

# Matches "1:23", "12:34", "1:23:45". SearXNG video engines return
# duration in this format for the major sources (YouTube, Bing Videos,
# Dailymotion); Vimeo returns a raw int (seconds). Either is handled.
_DURATION_HMS_RE = re.compile(r"^\s*(\d+):(\d{1,2})(?::(\d{1,2}))?\s*$")


def _parse_video_duration(raw) -> int | None:
    """Return duration in seconds, or None if unparseable.

    SearXNG returns video duration as either "MM:SS" / "HH:MM:SS" (most
    engines) or a raw int (Vimeo). Returns None for empty/garbage so
    callers can decide whether to keep or drop unknown-duration results
    when a filter is active — we keep them rather than over-filter.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        return max(0, int(raw))
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    # Try pure int first ("180" → 180 seconds)
    if text.isdigit():
        return int(text)
    m = _DURATION_HMS_RE.match(text)
    if not m:
        return None
    parts = [int(p) if p else 0 for p in m.groups()]
    if parts[2] == 0 and m.group(3) is None:
        # MM:SS form
        return parts[0] * 60 + parts[1]
    # HH:MM:SS form
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _filter_by_duration(results: list[dict], bucket: str) -> list[dict]:
    """Drop results whose parsed duration falls outside `bucket`.

    Results with unparseable / missing duration are kept (graceful) —
    only Vimeo and the major video engines reliably populate `length`,
    and hard-dropping non-video results that incidentally got tagged
    via the embeddable-host check would silently empty out genre-mixed
    result sets.
    """
    if bucket not in _DURATION_BUCKETS:
        return results
    lo, hi = _DURATION_BUCKETS[bucket]
    kept: list[dict] = []
    for r in results:
        secs = _parse_video_duration(r.get("duration"))
        if secs is None:
            kept.append(r)
            continue
        if lo <= secs < hi:
            kept.append(r)
    return kept


def _apply_sort(results: list[dict], sort_by: str) -> list[dict]:
    """Re-order results by user-selected sort, after the reputation rerank.

    `sort_by` overrides tier ordering — when the user explicitly asks
    for newest or longest they want the literal sort, not a tier-filtered
    one. Stable so the rerank's internal tie-break is preserved within
    equal-key groups.
    """
    if sort_by == "date":
        return sorted(
            results,
            key=lambda r: r.get("published_date") or "",
            reverse=True,
        )
    if sort_by in ("duration_asc", "duration_desc"):
        reverse = sort_by == "duration_desc"
        # Unknown durations sink to the bottom regardless of direction —
        # they shouldn't sandbag the explicit sort users asked for.
        return sorted(
            results,
            key=lambda r: (
                _parse_video_duration(r.get("duration")) is None,
                -(_parse_video_duration(r.get("duration")) or 0) if reverse
                else (_parse_video_duration(r.get("duration")) or 0),
            ),
        )
    return results


def _extract_site_scope(query: str) -> str:
    """Return the normalized domain from `site:X` in the query, or ''."""
    m = _SITE_OPERATOR_RE.search(query or "")
    if not m:
        return ""
    raw = m.group(1).strip().strip(".").lower().removeprefix("www.")
    return raw


def _url_matches_site(url: str, site: str) -> bool:
    """True if `url`'s hostname equals `site` or is a subdomain of it."""
    if not site:
        return True
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if not host:
        return False
    return host == site or host.endswith("." + site)

# Wayback Machine CDX API for finding archived snapshots
_WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"
_WAYBACK_RAW_URL = "https://web.archive.org/web/{timestamp}id_/{url}"

# 1x1 transparent PNG fallback for broken images
_TRANSPARENT_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

# ---------------------------------------------------------------------------
# AI action system prompts
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Content-aware AI prompts — specialized per page type
# ---------------------------------------------------------------------------
# The "_default" key is the fallback for any page type not listed.
# Note-specific actions (expand, formalize, fix, etc.) don't vary by page type.

_AI_PROMPT_VARIANTS: dict[str, dict[str, str]] = {
    "summarize": {
        "_default": (
            "Provide a concise summary of the following web page content. "
            "Focus on main arguments, key findings, and conclusions. Use bullet points."
        ),
        "product": (
            "Summarize this product page. Extract: product name, price (if shown), "
            "key features and specs, ratings/review consensus, and pros/cons. "
            "Use bullet points. Be concise."
        ),
        "recipe": (
            "Summarize this recipe. Include: dish name, prep and cook time, "
            "number of servings, difficulty level, dietary notes (vegan, gluten-free, etc.), "
            "and the core technique in 1-2 sentences. List key ingredients."
        ),
        "technical": (
            "Summarize this technical documentation. Cover: what it documents, "
            "key APIs/functions/classes, required parameters, prerequisites, "
            "and common usage patterns. Use code-friendly formatting."
        ),
        "research": (
            "Summarize this research paper or study. Include: research question/hypothesis, "
            "methodology, key findings with statistics, sample size, limitations, "
            "and practical implications. Be precise about claims vs. conclusions."
        ),
        "forum": (
            "Summarize this discussion thread. Include: the original question or problem, "
            "the accepted or top-voted answer, alternative approaches mentioned, "
            "and important caveats or warnings from the replies."
        ),
        "video": (
            "Summarize this video content. Cover the main topics discussed, "
            "key points with approximate timestamps where available, "
            "demonstrations or examples shown, and the speaker's conclusions. Use bullet points."
        ),
        "video_article": (
            "This page contains both written content and embedded video(s). "
            "Summarize both: the article's main points and the video content. "
            "Note where the video supplements, demonstrates, or differs from the written text."
        ),
        "reference": (
            "Summarize this reference article. Cover: the subject, essential facts, "
            "historical context, significance, and key relationships to other topics. "
            "Organize by the article's own section structure when possible."
        ),
        "event": (
            "Summarize this event page. Extract: event name, date/time, venue/location, "
            "ticket price or registration info, featured speakers/performers, "
            "and what attendees should expect."
        ),
        "job": (
            "Summarize this job posting. Extract: role title, company, location "
            "(remote/hybrid/onsite), salary range if listed, key requirements, "
            "nice-to-haves, benefits, and application deadline."
        ),
    },
    "keypoints": {
        "_default": (
            "Extract the 5-10 most important points. "
            "Present as a numbered list with brief explanations."
        ),
        "product": (
            "Extract the key decision-making points for this product: "
            "standout features, price-to-value assessment, what reviewers love, "
            "what reviewers complain about, and who this product is best for."
        ),
        "recipe": (
            "Extract the key points for making this recipe successfully: "
            "critical ingredients that can't be substituted, technique tips, "
            "common mistakes to avoid, and timing that matters."
        ),
        "technical": (
            "Extract the key technical takeaways: essential functions/APIs, "
            "required vs. optional parameters, breaking changes or gotchas, "
            "version requirements, and the most common usage pattern."
        ),
        "research": (
            "Extract the key findings: primary results with effect sizes or "
            "p-values, sample characteristics, methodology strengths/weaknesses, "
            "and how findings relate to prior work."
        ),
        "forum": (
            "Extract the key takeaways from this discussion: the solution that worked, "
            "alternative approaches, commands or code to use, "
            "and warnings about what NOT to do."
        ),
        "video": (
            "Extract key moments and takeaways from this video. "
            "Include approximate timestamps for each point. "
            "Focus on actionable insights and demonstrations."
        ),
        "reference": (
            "Extract the essential facts: key dates, figures, relationships, "
            "defining characteristics, and any ongoing debates or controversies."
        ),
    },
    "extract": {
        "_default": (
            "Extract the actionable data from this content: statistics, quotes, "
            "dates, names, factual claims, and any structured data (tables, lists). "
            "Present in organized markdown format."
        ),
        "product": (
            "Extract the product data: full specs table, price, availability, "
            "model/SKU numbers, ratings breakdown (stars distribution), "
            "and key quotes from reviews. Present as structured markdown."
        ),
        "recipe": (
            "Extract the clean recipe — no blog preamble, no life story. Output:\n"
            "1. Ingredients list with exact measurements\n"
            "2. Numbered preparation steps\n"
            "3. Cook time, temperature, and yield\n"
            "4. Storage and reheating instructions (if mentioned)"
        ),
        "technical": (
            "Extract the technical reference data: code snippets, CLI commands, "
            "configuration values, environment variables, API endpoints, "
            "function signatures with parameter types, and return values. "
            "Preserve code formatting."
        ),
        "research": (
            "Extract the research data: statistical results (tables, p-values, "
            "confidence intervals), sample demographics, methodology parameters, "
            "figure descriptions, and key citations."
        ),
        "forum": (
            "Extract the solution(s) from this discussion: code snippets, "
            "commands, configuration changes, file paths, links to resources, "
            "and step-by-step instructions. Ignore commentary and meta-discussion."
        ),
        "video": (
            "Extract structured content from this video: topic timestamps, "
            "key terms and definitions, resources/links mentioned, "
            "tools or software demonstrated, and step-by-step procedures shown."
        ),
        "reference": (
            "Extract structured reference data: key dates and events, "
            "named figures and their roles, statistics and measurements, "
            "classifications and categories, and cited sources."
        ),
    },
    "explain": {
        "_default": (
            "Explain the following content clearly. Simplify complex concepts, "
            "define technical terms, and provide context for why this matters."
        ),
        "technical": (
            "Explain this technical content clearly. Define jargon, "
            "explain what each function/API does in plain language, "
            "provide simple usage examples, and note common pitfalls."
        ),
        "research": (
            "Explain this research in accessible terms. Unpack the methodology, "
            "explain what the statistics mean practically, "
            "contextualize the findings for a non-specialist, "
            "and note the strength of the evidence."
        ),
        "video": (
            "Explain the concepts discussed in this video content. "
            "Define technical terms used, provide additional context, "
            "and clarify any complex demonstrations or examples."
        ),
    },
}

# Actions that don't vary by page type
_AI_PROMPTS_STATIC: dict[str, str] = {
    "ask": (
        "Respond to the user's request about the provided content. Use the "
        "content as the source of truth for claims about that source, but "
        "interpret the request by intent rather than exact keyword matching. "
        "If the user asks for a transformation such as summarizing, explaining, "
        "translating, outlining, extracting facts, or finding tasks, perform "
        "that transformation on the provided content. If they ask about a term, "
        "concept, comparison, implication, or follow-up that is not named "
        "verbatim, connect it to relevant passages when possible; use brief "
        "background knowledge only to define or explain, and clearly separate "
        "that from what the content itself says. If the content does not contain "
        "enough evidence to answer fully, say that briefly, then give the "
        "closest useful answer from the content and what would be needed to "
        "answer fully. Do not answer with only 'not in the article', 'term does "
        "not match', or similar. Output markdown."
    ),
    "expand": (
        "Expand and flesh out the following notes with additional detail, "
        "examples, and context. Maintain the original structure and voice. "
        "Output as markdown."
    ),
    "formalize": (
        "Rewrite the following casual or rough notes into polished, professional "
        "prose. Maintain all factual content. Use clear structure with headings "
        "where appropriate. Output as markdown."
    ),
    "fix": (
        "Fix grammar, spelling, punctuation, and improve the flow of the following "
        "text. Make minimal changes — preserve the author's voice and intent. "
        "Output the corrected text as markdown."
    ),
    "extract_tasks": (
        "Extract all action items, tasks, and TODOs from this content. "
        "Present as a markdown checklist using - [ ] format. "
        "Group by topic if multiple categories exist."
    ),
    "translate": (
        "Translate the following content. Maintain formatting and structure. "
        "Output as markdown."
    ),
    "outline": (
        "Create a structured outline of this content with numbered sections, "
        "subsections, and key points under each. Output as markdown."
    ),
}

# Combined set of valid action names
_AI_PROMPTS = {**{k: v["_default"] for k, v in _AI_PROMPT_VARIANTS.items()}, **_AI_PROMPTS_STATIC}


def _get_ai_prompt(action: str, page_type: str = "") -> str:
    """Get the best prompt for an action + page type combination."""
    # Static actions don't vary
    if action in _AI_PROMPTS_STATIC:
        return _AI_PROMPTS_STATIC[action]
    # Try page-type-specific variant, fall back to _default
    variants = _AI_PROMPT_VARIANTS.get(action, {})
    return variants.get(page_type, variants.get("_default", "Analyze the following content."))


def _build_ai_messages(
    action: str,
    content: str,
    question: str = "",
    page_type: str = "",
) -> tuple[str, str]:
    """Build the system + user messages for browse/note AI actions."""
    system_prompt = _get_ai_prompt(action, page_type)
    question = str(question or "").strip()

    if action == "ask":
        if question:
            return (
                system_prompt,
                f"User request:\n{question}\n\n---\n\nProvided content:\n{content}",
            )
        return system_prompt, f"Provided content:\n{content}"

    if action == "translate" and question:
        return (
            system_prompt,
            f"Target language:\n{question}\n\n---\n\nProvided content:\n{content}",
        )

    if question:
        return (
            system_prompt,
            f"User focus or request:\n{question}\n\n---\n\nProvided content:\n{content}",
        )

    return system_prompt, f"Provided content:\n{content}"


def _detect_page_type(url: str, source: str = "", has_videos: bool = False) -> str:
    """Detect page type from URL patterns, source field, and content signals."""
    # Schema-detected types (source is "schema:Product", "schema:Recipe", etc.)
    if source.startswith("schema:"):
        schema_type = source.split(":", 1)[1].lower()
        type_map = {
            "product": "product", "recipe": "recipe", "event": "event",
            "localbusiness": "product", "jobposting": "job",
        }
        if schema_type in type_map:
            return type_map[schema_type]

    # Source-based
    if source == "youtube-embed":
        return "video"
    if source == "wikipedia-api":
        return "reference"

    # URL pattern-based
    hostname = urlparse(url).hostname or ""
    domain = hostname.lower().removeprefix("www.")
    path = urlparse(url).path.lower()

    # Forums / Q&A
    _FORUM_DOMAINS = {
        "reddit.com", "old.reddit.com",
        "stackoverflow.com", "stackexchange.com", "superuser.com",
        "serverfault.com", "askubuntu.com",
        "news.ycombinator.com",
        "discourse.org", "forum.",
    }
    if any(domain == d or domain.endswith("." + d) for d in _FORUM_DOMAINS):
        return "forum"

    # Technical documentation
    if (
        domain.startswith("docs.") or domain.startswith("developer.")
        or "devdocs" in domain or domain == "mdn.io"
        or domain.endswith(".readthedocs.io") or domain.endswith(".rtfd.io")
        or "/docs/" in path or "/api/" in path or "/reference/" in path
    ):
        return "technical"

    # Research / academic
    _RESEARCH_DOMAINS = {
        "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "scholar.google.com",
        "nature.com", "science.org", "sciencedirect.com",
        "ieee.org", "acm.org", "springer.com", "wiley.com",
        "biorxiv.org", "medrxiv.org", "ssrn.com", "jstor.org",
        "plos.org", "cell.com", "thelancet.com", "bmj.com",
    }
    if any(domain == d or domain.endswith("." + d) for d in _RESEARCH_DOMAINS):
        return "research"

    # Reference / encyclopedic
    if "wikipedia.org" in domain or "britannica.com" in domain or "wikiwand.com" in domain:
        return "reference"

    # Pages with significant embedded video content
    if has_videos:
        return "video_article"

    return "article"

_MAX_CONTENT_CHARS = 30_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_html_tags(html: str) -> str:
    """Crude HTML tag stripping fallback."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Patterns that indicate the "content" is actually junk
_JUNK_PATTERNS = [
    # Error pages — numeric codes with descriptors
    re.compile(r"^(400|403|404|500|502|503)\s*(Bad Request|Forbidden|Not Found|Error)", re.IGNORECASE),
    # Error pages without the status code prefix — common on hand-rolled 404s
    # that just render "Page not found" or "That page doesn't exist".
    re.compile(r"\b(page|page you('re| are) looking for|that page) (was )?not found\b", re.IGNORECASE),
    re.compile(r"\b(page|content|article) (doesn'?t exist|has been (removed|deleted)|no longer (available|exists))\b", re.IGNORECASE),
    re.compile(r"\b(oops|sorry)[!.,]?\s*(we (can'?t|cannot)|the (page|content)).{0,60}(find|locate|not found)", re.IGNORECASE),
    re.compile(r"blocked by.{0,30}(security|firewall|policy)", re.IGNORECASE),
    re.compile(r"Access Denied", re.IGNORECASE),
    # Anti-bot / JS-required walls
    re.compile(r"please enable (javascript|JS)", re.IGNORECASE),
    re.compile(r"enable.{0,20}(javascript|JS).{0,30}(ad\s*block|adblocker)", re.IGNORECASE),
    re.compile(r"you need to enable javascript", re.IGNORECASE),
    re.compile(r"this (site|page) requires javascript", re.IGNORECASE),
    re.compile(r"browser.{0,20}(not supported|out of date|needs? updating)", re.IGNORECASE),
    # Adblock walls — both directions of the phrase ("please disable" vs
    # "ad-block detected"). Audit flagged the original pattern as too
    # narrow; these cover the common framings.
    re.compile(r"(please|kindly)?\s*disable.{0,30}ad\s*[-]?block(?:er)?", re.IGNORECASE),
    re.compile(r"ad\s*[-]?block(?:er)?.{0,30}(detected|enabled|turn off|disable)", re.IGNORECASE),
    re.compile(r"(turn off|whitelist).{0,30}ad\s*[-]?block(?:er)?", re.IGNORECASE),
    re.compile(r"we (rely on|depend on) ad(vertising)? revenue", re.IGNORECASE),
    # CAPTCHA / challenge
    re.compile(r"verify.{0,20}(human|not a robot|captcha)", re.IGNORECASE),
    re.compile(r"checking (your|the) browser", re.IGNORECASE),
    # Cloudflare Turnstile / "Just a moment" interstitials — these are
    # the verbatim text a server-side fetch sees when challenge JS won't run.
    re.compile(r"just a moment\b", re.IGNORECASE),
    re.compile(r"\b(one moment|please wait)\b.{0,40}(verifying|checking|securely)", re.IGNORECASE),
    re.compile(r"ray\s*id:?\s*[a-f0-9]{8,}", re.IGNORECASE),
    re.compile(r"\battention required\b.{0,40}cloudflare", re.IGNORECASE),
    re.compile(r"performance\s*&\s*security\s+by\s+cloudflare", re.IGNORECASE),
    # Cookie consent as sole content. Tightened to require a
    # consent-action verb within ~200 chars so a real article that
    # mentions "we use cookies for analytics" in passing doesn't trip.
    # Banners always pair the cookie disclosure with accept/agree/manage.
    re.compile(
        r"\b(we|this site|our (site|website)|we and our partners) use[sd]? cookies\b"
        r".{0,200}\b(accept|agree|consent|dismiss|manage|customize|preferences)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(by (using|continuing|clicking|browsing))\b.{0,80}cookies"
        r".{0,100}\b(accept|agree|continue|consent)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\b(accept|manage|customize) (all )?cookies\b.{0,40}(continue|preferences|settings)", re.IGNORECASE),
    re.compile(r"\bce site (web )?utilise (les )?cookies\b.{0,120}\b(accepter|consentement|paramètres)\b", re.IGNORECASE | re.DOTALL),  # FR
    re.compile(r"\bdiese (website|seite) (verwendet|nutzt) cookies\b.{0,120}\b(akzeptieren|zustimmen|einstellungen)\b", re.IGNORECASE | re.DOTALL),  # DE
    re.compile(r"\beste sitio (web )?(usa|utiliza) cookies\b.{0,120}\b(aceptar|consentimiento|preferencias)\b", re.IGNORECASE | re.DOTALL),  # ES
    # Paywalls / login walls. Phrased narrowly so a normal page with a
    # login link in its header doesn't false-positive — these patterns
    # only fire when the text *explicitly* asks the user to sign up /
    # subscribe / log in *to continue or read*, which is the signature
    # of a gating interstitial, not incidental navigation.
    re.compile(r"(sign in|log in).{0,30}(to (continue|read|view|access)|required)", re.IGNORECASE),
    re.compile(r"(subscribe|subscription required).{0,30}(to (continue|read|view|access))", re.IGNORECASE),
    re.compile(r"create (a|an) (free )?account.{0,30}(to (continue|read|view|access))", re.IGNORECASE),
    re.compile(r"this (article|content|page|story) is (for|available to) (subscribers|members)", re.IGNORECASE),
    re.compile(r"(you('?ve| have)? reached|you'?ve hit).{0,30}(article limit|free (article|read) limit)", re.IGNORECASE),
]


def _is_junk_content(text: str) -> bool:
    """Detect if extracted text is actually an error page, JS-required wall, etc."""
    if not text:
        return True
    # PDF binary leaking through
    if text.startswith("%PDF-") or text[:50].count("\x00") > 5:
        return True
    # Very short — check against junk patterns
    check = text[:500]
    return any(p.search(check) for p in _JUNK_PATTERNS)


def _is_binary_response(html: str) -> bool:
    """Detect binary content (PDFs, images) that leaked through as text."""
    if not html:
        return False
    head = html[:100]
    return (
        head.startswith("%PDF-")
        or head.startswith("\x89PNG")
        or head.startswith("GIF8")
        or head[:4] in ("\xff\xd8\xff\xe0", "\xff\xd8\xff\xe1")  # JPEG
        or head.count("\x00") > 10  # binary content
    )


def _extract_title(html: str) -> str:
    """Extract <title> from raw HTML."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_with_trafilatura(html: str, *, with_metadata: bool = False):
    """Run trafilatura extraction (may not be installed)."""
    try:
        import trafilatura  # type: ignore[import-untyped]
    except ImportError:
        return None
    if with_metadata:
        return trafilatura.extract(
            html,
            with_metadata=True,
            include_links=True,
            include_images=True,
            output_format="txt",
        )
    return trafilatura.extract(html)


def _extract_metadata_with_trafilatura(html: str) -> dict:
    """Get metadata dict from trafilatura."""
    try:
        import trafilatura  # type: ignore[import-untyped]
    except ImportError:
        return {}
    try:
        doc = trafilatura.bare_extraction(html)
        if doc:
            return {
                "author": doc.get("author", ""),
                "date": doc.get("date", ""),
                "sitename": doc.get("sitename", ""),
                "title": doc.get("title", ""),
                "description": doc.get("description", ""),
            }
    except Exception as exc:
        log.debug("trafilatura_metadata_extract_failed", error=str(exc))
    return {}


async def _fetch_with_chrome_tls(url: str, timeout: float = 20.0) -> tuple[str, dict]:
    """Fetch URL with Chrome TLS fingerprint + SSRF protection.

    Uses curl_cffi to match Chrome's exact TLS handshake (JA3/JA4),
    with SafeHttpClient's hostname/IP validation for SSRF prevention.
    Falls back to httpx if curl_cffi is unavailable.
    """
    # SSRF checks first
    hostname = _safe_client._validate_url(url)
    await _safe_client._check_resolved_ips(hostname)

    try:
        from curl_cffi.requests import AsyncSession

        # Let curl_cffi generate its own matching headers for the
        # impersonated browser — only add extras it doesn't set.
        # Overriding User-Agent would create a TLS/header mismatch
        # that bot detectors catch.
        async with AsyncSession(
            impersonate="chrome131",
            max_redirects=5,
            timeout=timeout,
        ) as session:
            response = await session.get(url, allow_redirects=True)

            body = response.content
            if len(body) > _safe_client._max_response_size:
                raise SafeHttpError(
                    f"Response too large: {len(body)} bytes "
                    f"(limit {_safe_client._max_response_size})"
                )

            # Validate final URL after redirects
            final_hostname = urlparse(str(response.url)).hostname
            if final_hostname and final_hostname != hostname:
                await _safe_client._check_resolved_ips(final_hostname)

            text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else body
            return text, {
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "content_length": len(body),
            }
    except ImportError:
        log.debug("curl_cffi_not_available_falling_back_to_httpx")
        # Fallback to httpx with browser headers (no TLS fingerprint match)
        from augmentum.utils.safe_http import _PinnedTransport

        transport = _PinnedTransport(_safe_client)
        async with httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=5,
            timeout=httpx.Timeout(timeout),
            transport=transport,
            headers=_BROWSER_HEADERS,
        ) as client:
            response = await client.get(url)

            body = response.content
            if len(body) > _safe_client._max_response_size:
                raise SafeHttpError(
                    f"Response too large: {len(body)} bytes "
                    f"(limit {_safe_client._max_response_size})"
                )

            final_hostname = urlparse(str(response.url)).hostname
            if final_hostname and final_hostname != hostname:
                await _safe_client._check_resolved_ips(final_hostname)

            text = body.decode("utf-8", errors="replace")
            return text, {
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "content_length": len(body),
            }


async def _fetch_wayback(url: str) -> tuple[str, dict] | None:
    """Try to fetch a recent Wayback Machine snapshot of the URL.

    Uses the CDX API to find the most recent snapshot, then fetches
    the raw archived HTML (id_ modifier strips the Wayback toolbar).
    Returns (html, metadata) or None if no snapshot found.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            # Query CDX for most recent snapshot
            cdx_resp = await client.get(
                _WAYBACK_CDX_URL,
                params={
                    "url": url,
                    "output": "json",
                    "limit": "1",
                    "sort": "closest",
                    "to": "20261231",  # search backwards from future
                    "fl": "timestamp,statuscode,original",
                    "filter": "statuscode:200",
                },
            )
            if cdx_resp.status_code != 200:
                return None

            rows = cdx_resp.json()
            if len(rows) < 2:  # first row is header
                return None

            timestamp = rows[1][0]
            # Fetch the raw HTML (id_ strips Wayback's injected toolbar)
            raw_url = _WAYBACK_RAW_URL.format(timestamp=timestamp, url=url)
            page_resp = await client.get(raw_url, follow_redirects=True)
            if page_resp.status_code != 200:
                return None

            body = page_resp.text
            if len(body) < 500:
                return None

            log.info("wayback_fallback_used", url=url, timestamp=timestamp)
            return body, {
                "url": url,
                "status_code": 200,
                "content_type": "text/html",
                "content_length": len(body),
                "source": "wayback",
                "wayback_timestamp": timestamp,
            }
    except Exception:
        log.debug("wayback_fetch_failed", url=url, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Layer 0: Zero-cost structured data shortcuts
# ---------------------------------------------------------------------------

def _extract_jsonld_article(html: str) -> dict | None:
    """Extract full article text from JSON-LD structured data.

    Many news sites embed Schema.org Article with articleBody in
    <script type="application/ld+json"> — this gives us the full text
    without any heuristic extraction.
    """
    try:
        import extruct  # type: ignore[import-untyped]
        data = extruct.extract(html, syntaxes=["json-ld"], errors="ignore")
    except ImportError:
        # Fallback: regex extraction if extruct not installed
        data = {"json-ld": []}
        for m in re.finditer(
            r'<script\s+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html, re.DOTALL | re.IGNORECASE,
        ):
            try:
                parsed = json.loads(m.group(1))
                if isinstance(parsed, list):
                    data["json-ld"].extend(parsed)
                else:
                    data["json-ld"].append(parsed)
            except (json.JSONDecodeError, TypeError):
                continue
    except Exception:
        return None

    # Search for Article types with articleBody
    for item in data.get("json-ld", []):
        items_to_check = [item]
        # Handle @graph arrays
        if isinstance(item, dict) and "@graph" in item:
            items_to_check = item["@graph"]

        for obj in items_to_check:
            if not isinstance(obj, dict):
                continue
            obj_type = obj.get("@type", "")
            if isinstance(obj_type, list):
                obj_type = " ".join(obj_type)
            if not any(t in obj_type for t in ("Article", "NewsArticle", "BlogPosting", "Report")):
                continue

            body = obj.get("articleBody", "")
            if body and len(body) > _MIN_CONTENT_CHARS:
                return {
                    "text": body,
                    "title": obj.get("headline", ""),
                    "author": (
                        obj.get("author", {}).get("name", "")
                        if isinstance(obj.get("author"), dict)
                        else str(obj.get("author", ""))
                    ),
                    "date": obj.get("datePublished", ""),
                    "sitename": obj.get("publisher", {}).get("name", "")
                        if isinstance(obj.get("publisher"), dict) else "",
                    "description": obj.get("description", ""),
                }
    return None


def _extract_jsonld_structured(html: str) -> dict | None:
    """Extract structured data from JSON-LD for non-article page types.

    Handles: Product, Recipe, Event, LocalBusiness, JobPosting,
    RealEstateListing, Vehicle, Restaurant, and more.  Returns a dict
    with 'html' (rendered card), 'title', 'type' keys — or None.
    """
    try:
        import extruct  # type: ignore[import-untyped]
        data = extruct.extract(html, syntaxes=["json-ld"], errors="ignore")
    except ImportError:
        data = {"json-ld": []}
        for m in re.finditer(
            r'<script\s+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html, re.DOTALL | re.IGNORECASE,
        ):
            try:
                parsed = json.loads(m.group(1))
                if isinstance(parsed, list):
                    data["json-ld"].extend(parsed)
                else:
                    data["json-ld"].append(parsed)
            except (json.JSONDecodeError, TypeError):
                continue
    except Exception:
        return None

    # Flatten @graph arrays
    all_objects: list[dict] = []
    for item in data.get("json-ld", []):
        if isinstance(item, dict) and "@graph" in item:
            all_objects.extend(item["@graph"])
        elif isinstance(item, dict):
            all_objects.append(item)

    for obj in all_objects:
        if not isinstance(obj, dict):
            continue
        obj_type = obj.get("@type", "")
        # @type can be a single string OR a list. The list case is common
        # for hybrid objects ("@type": ["WebPage", "Article"]) — we walk
        # the list looking for the first type that has a renderer rather
        # than blindly taking [0], otherwise pages tagged as
        # ["WebPage", "Article"] would miss the Article renderer.
        type_candidates: list[str]
        if isinstance(obj_type, list):
            type_candidates = [t for t in obj_type if isinstance(t, str)]
        elif isinstance(obj_type, str):
            type_candidates = [obj_type]
        else:
            continue

        renderer = None
        for t in type_candidates:
            renderer = _SCHEMA_RENDERERS.get(t)
            if renderer:
                break
        if renderer:
            try:
                result = renderer(obj)
                if result:
                    return result
            except Exception as exc:
                log.debug("schema_renderer_failed", types=type_candidates, error=str(exc))
                continue
    return None


def _esc(text) -> str:
    """HTML-escape for schema renderer output."""
    if not text:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _render_product(obj: dict) -> dict | None:
    """Render a Schema.org Product as a clean reader card."""
    name = obj.get("name", "")
    if not name:
        return None

    parts = [f'<div class="browse-schema-card browse-schema-product">']
    # Image
    img = obj.get("image")
    if isinstance(img, list):
        img = img[0] if img else ""
    if isinstance(img, dict):
        img = img.get("url", "")
    if img:
        parts.append(f'<img class="browse-schema-img" src="{_esc(img)}" alt="{_esc(name)}" loading="lazy">')

    parts.append('<div class="browse-schema-details">')
    parts.append(f'<h2 class="browse-schema-title">{_esc(name)}</h2>')

    # Brand
    brand = obj.get("brand", {})
    if isinstance(brand, dict):
        brand = brand.get("name", "")
    if brand:
        parts.append(f'<div class="browse-schema-brand">{_esc(brand)}</div>')

    # Price
    offers = obj.get("offers", {})
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    if isinstance(offers, dict):
        price = offers.get("price", "")
        currency = offers.get("priceCurrency", "USD")
        avail = offers.get("availability", "")
        if price:
            symbol = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "JPY": "\u00a5"}.get(currency, currency + " ")
            parts.append(f'<div class="browse-schema-price">{_esc(symbol)}{_esc(price)}</div>')
        if avail:
            avail_text = avail.replace("https://schema.org/", "").replace("http://schema.org/", "")
            avail_class = "in-stock" if "InStock" in avail_text else "out-of-stock"
            parts.append(f'<div class="browse-schema-avail {avail_class}">{_esc(avail_text)}</div>')

    # Rating — use `is not None` so a legitimate 0 rating isn't dropped
    # as missing; also reject sentinel strings JSON-LD producers use to
    # signal "no rating yet" ("N/A", "null", empty).
    rating = obj.get("aggregateRating", {})
    if isinstance(rating, dict):
        val = rating.get("ratingValue")
        if val is not None and str(val).strip().lower() not in ("", "n/a", "null", "none"):
            count = rating.get("reviewCount", rating.get("ratingCount", ""))
            parts.append(f'<div class="browse-schema-rating">{_esc(val)}/5{f" ({_esc(count)} reviews)" if count else ""}</div>')

    # Description
    desc = obj.get("description", "")
    if desc:
        parts.append(f'<p class="browse-schema-desc">{_esc(desc[:1000])}</p>')

    # Specs (from additionalProperty)
    props = obj.get("additionalProperty", [])
    if isinstance(props, list) and props:
        parts.append('<table class="browse-schema-specs">')
        for prop in props[:15]:
            if isinstance(prop, dict):
                pn = _esc(prop.get("name", ""))
                pv = _esc(str(prop.get("value", "")))
                if pn and pv:
                    parts.append(f"<tr><th>{pn}</th><td>{pv}</td></tr>")
        parts.append("</table>")

    parts.append("</div></div>")

    return {"html": "\n".join(parts), "title": name, "type": "Product"}


def _render_recipe(obj: dict) -> dict | None:
    """Render a Schema.org Recipe as a clean reader card."""
    name = obj.get("name", "")
    if not name:
        return None

    parts = [f'<div class="browse-schema-card browse-schema-recipe">']
    img = obj.get("image")
    if isinstance(img, list):
        img = img[0] if img else ""
    if isinstance(img, dict):
        img = img.get("url", "")
    if img:
        parts.append(f'<img class="browse-schema-img" src="{_esc(img)}" alt="{_esc(name)}" loading="lazy">')

    parts.append('<div class="browse-schema-details">')
    parts.append(f'<h2 class="browse-schema-title">{_esc(name)}</h2>')

    # Meta row (prep time, cook time, servings)
    meta = []
    if obj.get("prepTime"):
        meta.append(f"Prep: {_esc(_format_duration(obj['prepTime']))}")
    if obj.get("cookTime"):
        meta.append(f"Cook: {_esc(_format_duration(obj['cookTime']))}")
    if obj.get("totalTime"):
        meta.append(f"Total: {_esc(_format_duration(obj['totalTime']))}")
    if obj.get("recipeYield"):
        meta.append(f"Serves: {_esc(str(obj['recipeYield']))}")
    if meta:
        parts.append(f'<div class="browse-schema-meta">{" &middot; ".join(meta)}</div>')

    # Rating — see _render_product for the is-not-None rationale.
    rating = obj.get("aggregateRating", {})
    if isinstance(rating, dict):
        val = rating.get("ratingValue")
        if val is not None and str(val).strip().lower() not in ("", "n/a", "null", "none"):
            count = rating.get("reviewCount", "")
            parts.append(f'<div class="browse-schema-rating">{_esc(val)}/5{f" ({_esc(count)} reviews)" if count else ""}</div>')

    # Description
    desc = obj.get("description", "")
    if desc:
        parts.append(f'<p class="browse-schema-desc">{_esc(desc[:500])}</p>')

    # Ingredients
    ingredients = obj.get("recipeIngredient", [])
    if isinstance(ingredients, list) and ingredients:
        parts.append('<h3>Ingredients</h3><ul class="browse-schema-ingredients">')
        for ing in ingredients:
            parts.append(f"<li>{_esc(str(ing))}</li>")
        parts.append("</ul>")

    # Instructions
    instructions = obj.get("recipeInstructions", [])
    if instructions:
        parts.append('<h3>Instructions</h3><ol class="browse-schema-instructions">')
        if isinstance(instructions, list):
            for step in instructions:
                if isinstance(step, dict):
                    text = step.get("text", "")
                elif isinstance(step, str):
                    text = step
                else:
                    continue
                if text:
                    parts.append(f"<li>{_esc(text)}</li>")
        elif isinstance(instructions, str):
            for line in instructions.split("\n"):
                if line.strip():
                    parts.append(f"<li>{_esc(line.strip())}</li>")
        parts.append("</ol>")

    # Nutrition
    nutrition = obj.get("nutrition", {})
    if isinstance(nutrition, dict):
        nutri_parts = []
        for key in ("calories", "fatContent", "carbohydrateContent", "proteinContent", "fiberContent", "sodiumContent"):
            val = nutrition.get(key, "")
            if val:
                label = key.replace("Content", "").capitalize()
                nutri_parts.append(f"{_esc(label)}: {_esc(str(val))}")
        if nutri_parts:
            parts.append(f'<div class="browse-schema-nutrition">{" &middot; ".join(nutri_parts)}</div>')

    parts.append("</div></div>")
    return {"html": "\n".join(parts), "title": name, "type": "Recipe"}


def _render_event(obj: dict) -> dict | None:
    """Render a Schema.org Event."""
    name = obj.get("name", "")
    if not name:
        return None

    parts = [f'<div class="browse-schema-card browse-schema-event">']
    parts.append('<div class="browse-schema-details">')
    parts.append(f'<h2 class="browse-schema-title">{_esc(name)}</h2>')

    # Date
    start = obj.get("startDate", "")
    end = obj.get("endDate", "")
    if start:
        parts.append(f'<div class="browse-schema-meta">{_esc(start)}{f" &ndash; {_esc(end)}" if end else ""}</div>')

    # Location
    location = obj.get("location", {})
    if isinstance(location, dict):
        loc_name = location.get("name", "")
        address = location.get("address", "")
        if isinstance(address, dict):
            address = ", ".join(filter(None, [
                address.get("streetAddress", ""),
                address.get("addressLocality", ""),
                address.get("addressRegion", ""),
            ]))
        if loc_name or address:
            parts.append(f'<div class="browse-schema-location">{_esc(loc_name)}{f" &mdash; {_esc(address)}" if address else ""}</div>')

    # Description
    desc = obj.get("description", "")
    if desc:
        parts.append(f'<p class="browse-schema-desc">{_esc(desc[:500])}</p>')

    # Offers/tickets
    offers = obj.get("offers", {})
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    if isinstance(offers, dict) and offers.get("price"):
        parts.append(f'<div class="browse-schema-price">From {_esc(offers.get("priceCurrency", "$"))}{_esc(offers["price"])}</div>')

    parts.append("</div></div>")
    return {"html": "\n".join(parts), "title": name, "type": "Event"}


def _render_local_business(obj: dict) -> dict | None:
    """Render a Schema.org LocalBusiness / Restaurant."""
    name = obj.get("name", "")
    if not name:
        return None

    parts = [f'<div class="browse-schema-card browse-schema-business">']
    img = obj.get("image")
    if isinstance(img, list):
        img = img[0] if img else ""
    if isinstance(img, dict):
        img = img.get("url", "")
    if img:
        parts.append(f'<img class="browse-schema-img" src="{_esc(img)}" alt="{_esc(name)}" loading="lazy">')

    parts.append('<div class="browse-schema-details">')
    parts.append(f'<h2 class="browse-schema-title">{_esc(name)}</h2>')

    # Address
    address = obj.get("address", {})
    if isinstance(address, dict):
        addr_str = ", ".join(filter(None, [
            address.get("streetAddress", ""),
            address.get("addressLocality", ""),
            address.get("addressRegion", ""),
            address.get("postalCode", ""),
        ]))
        if addr_str:
            parts.append(f'<div class="browse-schema-location">{_esc(addr_str)}</div>')

    # Phone
    phone = obj.get("telephone", "")
    if phone:
        parts.append(f'<div class="browse-schema-phone">{_esc(phone)}</div>')

    # Rating — see _render_product for the is-not-None rationale.
    rating = obj.get("aggregateRating", {})
    if isinstance(rating, dict):
        val = rating.get("ratingValue")
        if val is not None and str(val).strip().lower() not in ("", "n/a", "null", "none"):
            count = rating.get("reviewCount", "")
            parts.append(f'<div class="browse-schema-rating">{_esc(val)}/5{f" ({_esc(count)} reviews)" if count else ""}</div>')

    # Hours
    hours = obj.get("openingHoursSpecification", [])
    if isinstance(hours, list) and hours:
        parts.append('<div class="browse-schema-hours"><strong>Hours:</strong> ')
        hour_strs = []
        for h in hours[:7]:
            if isinstance(h, dict):
                day = h.get("dayOfWeek", "")
                if isinstance(day, list):
                    day = ", ".join(d.replace("https://schema.org/", "").replace("http://schema.org/", "") for d in day)
                else:
                    day = day.replace("https://schema.org/", "").replace("http://schema.org/", "")
                opens = h.get("opens", "")
                closes = h.get("closes", "")
                if day and opens:
                    hour_strs.append(f"{_esc(day)} {_esc(opens)}-{_esc(closes)}")
        parts.append("; ".join(hour_strs))
        parts.append("</div>")

    # Description
    desc = obj.get("description", "")
    if desc:
        parts.append(f'<p class="browse-schema-desc">{_esc(desc[:500])}</p>')

    # Price range
    price_range = obj.get("priceRange", "")
    if price_range:
        parts.append(f'<div class="browse-schema-price">{_esc(price_range)}</div>')

    parts.append("</div></div>")
    return {"html": "\n".join(parts), "title": name, "type": "LocalBusiness"}


def _render_job_posting(obj: dict) -> dict | None:
    """Render a Schema.org JobPosting."""
    title = obj.get("title", "")
    if not title:
        return None

    parts = [f'<div class="browse-schema-card browse-schema-job">']
    parts.append('<div class="browse-schema-details">')
    parts.append(f'<h2 class="browse-schema-title">{_esc(title)}</h2>')

    # Company
    org = obj.get("hiringOrganization", {})
    if isinstance(org, dict):
        parts.append(f'<div class="browse-schema-brand">{_esc(org.get("name", ""))}</div>')

    # Location
    location = obj.get("jobLocation", {})
    if isinstance(location, dict):
        address = location.get("address", {})
        if isinstance(address, dict):
            loc_str = ", ".join(filter(None, [
                address.get("addressLocality", ""),
                address.get("addressRegion", ""),
            ]))
            if loc_str:
                parts.append(f'<div class="browse-schema-location">{_esc(loc_str)}</div>')

    # Salary
    salary = obj.get("baseSalary", {})
    if isinstance(salary, dict):
        val = salary.get("value", {})
        if isinstance(val, dict):
            min_v = val.get("minValue", "")
            max_v = val.get("maxValue", "")
            currency = salary.get("currency", "USD")
            if min_v or max_v:
                parts.append(f'<div class="browse-schema-price">{_esc(currency)} {_esc(min_v)} - {_esc(max_v)}</div>')

    # Employment type
    emp_type = obj.get("employmentType", "")
    if emp_type:
        if isinstance(emp_type, list):
            emp_type = ", ".join(emp_type)
        parts.append(f'<div class="browse-schema-meta">{_esc(emp_type)}</div>')

    # Description
    desc = obj.get("description", "")
    if desc:
        # Job descriptions are often HTML
        clean = re.sub(r"<[^>]+>", " ", desc)
        clean = re.sub(r"\s+", " ", clean).strip()
        parts.append(f'<p class="browse-schema-desc">{_esc(clean[:1000])}</p>')

    parts.append("</div></div>")
    return {"html": "\n".join(parts), "title": title, "type": "JobPosting"}


def _format_duration(iso: str) -> str:
    """Convert ISO 8601 duration (PT30M, PT1H15M) to human-readable."""
    if not iso:
        return ""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso, re.IGNORECASE)
    if not m:
        return iso
    hours, mins, secs = m.groups()
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if mins:
        parts.append(f"{mins}m")
    if secs and not hours:
        parts.append(f"{secs}s")
    return " ".join(parts) or iso


# Dispatcher: Schema.org @type → renderer function
_SCHEMA_RENDERERS: dict[str, callable] = {
    "Product": _render_product,
    "Recipe": _render_recipe,
    "Event": _render_event,
    "MusicEvent": _render_event,
    "SportsEvent": _render_event,
    "LocalBusiness": _render_local_business,
    "Restaurant": _render_local_business,
    "FoodEstablishment": _render_local_business,
    "Store": _render_local_business,
    "JobPosting": _render_job_posting,
}


def _discover_amp_url(html: str) -> str | None:
    """Find AMP version URL from <link rel="amphtml"> in page head."""
    m = re.search(
        r'<link\s+[^>]*rel=["\']amphtml["\'][^>]*href=["\']([^"\']+)["\']',
        html, re.IGNORECASE,
    )
    if m:
        return m.group(1)
    # Also check reversed attribute order
    m = re.search(
        r'<link\s+[^>]*href=["\']([^"\']+)["\'][^>]*rel=["\']amphtml["\']',
        html, re.IGNORECASE,
    )
    return m.group(1) if m else None


async def _fetch_rss_article(feed_url: str, page_url: str) -> str | None:
    """Try to find the current page's content in an RSS feed.

    Fetches the feed, finds the entry matching page_url, and returns
    its full content if available.
    """
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            headers=_BROWSER_HEADERS,
        ) as client:
            resp = await client.get(feed_url)
            if resp.status_code != 200:
                return None

        feed_xml = resp.text
        # Quick regex extraction — avoids feedparser dependency
        # Look for <item> or <entry> containing our URL
        parsed_page = urlparse(page_url)
        page_path = parsed_page.path.rstrip("/")

        # Find all content blocks in the feed
        for tag in ("content:encoded", "content", "summary"):
            pattern = rf"<link[^>]*>{re.escape(page_url)}.*?<{tag}[^>]*>(.*?)</{tag}>"
            m = re.search(pattern, feed_xml, re.DOTALL | re.IGNORECASE)
            if m:
                content = m.group(1).strip()
                # Unescape CDATA
                content = content.replace("<![CDATA[", "").replace("]]>", "")
                if len(content) > _MIN_CONTENT_CHARS:
                    log.info("rss_content_found", url=page_url, feed=feed_url)
                    return content

        # Also try matching by path (URLs in feeds sometimes differ slightly)
        if page_path:
            for tag in ("content:encoded", "content", "summary"):
                pattern = rf"<link[^>]*>[^<]*{re.escape(page_path)}[^<]*</link>.*?<{tag}[^>]*>(.*?)</{tag}>"
                m = re.search(pattern, feed_xml, re.DOTALL | re.IGNORECASE)
                if m:
                    content = m.group(1).strip().replace("<![CDATA[", "").replace("]]>", "")
                    if len(content) > _MIN_CONTENT_CHARS:
                        log.info("rss_content_found_by_path", url=page_url)
                        return content
    except Exception:
        log.debug("rss_fetch_failed", url=feed_url, exc_info=True)

    return None


# Domains/patterns that serve tracking pixels, ads, or junk images
_JUNK_IMAGE_DOMAINS = {
    "pixel", "track", "beacon", "analytics", "doubleclick", "facebook.com/tr",
    "google-analytics", "googlesyndication", "adservice", "adsystem",
    "sharethis", "addthis", "gravatar",  # author avatars
}

_JUNK_IMAGE_PATTERNS = re.compile(
    r"(tracking|pixel|beacon|spacer|blank|logo|icon|badge|avatar|emoji|"
    r"share|social|facebook|twitter|pinterest|linkedin|whatsapp|telegram|"
    r"arrow|bullet|separator|divider|spinner|loader|ad[_-]|"
    r"1x1|transparent\.(?:gif|png))",
    re.IGNORECASE,
)


def _filter_junk_images(content) -> None:
    """Remove non-content images: tracking pixels, icons, social buttons, ads.

    Keeps images that are likely editorial content (photos, diagrams, charts).
    Operates on an lxml element tree in-place.
    """
    for img in list(content.iter("img")):
        src = img.get("src", "")
        alt = img.get("alt", "")
        classes = img.get("class", "")
        width = img.get("width", "")
        height = img.get("height", "")

        remove = False

        # 1. Explicit tiny dimensions — tracking pixels and spacers
        try:
            w = int(width) if width else 999
            h = int(height) if height else 999
            if w <= 3 or h <= 3:
                remove = True
            elif w <= 32 and h <= 32:
                remove = True  # likely an icon
        except (ValueError, TypeError):
            pass

        # 2. Source URL contains junk patterns
        if not remove and _JUNK_IMAGE_PATTERNS.search(src):
            remove = True

        # 3. Source URL from known tracking/ad domains
        if not remove:
            src_lower = src.lower()
            for domain in _JUNK_IMAGE_DOMAINS:
                if domain in src_lower:
                    remove = True
                    break

        # 4. Class names suggest non-content image
        if not remove and re.search(
            r"(avatar|icon|logo|badge|emoji|social|share|tracking|ad-|lazy-placeholder)",
            classes, re.IGNORECASE
        ):
            remove = True

        # 5. Data URIs — almost always lazy-load placeholders or tiny inline icons
        #    Real content images are never inline base64 (they'd be huge)
        if not remove and src.startswith("data:"):
            remove = True

        # 6. Empty src
        if not remove and not src:
            remove = True

        # KEEP if: inside a <figure> (editorial images are almost always in figures)
        if remove:
            parent = img.getparent()
            if parent is not None and parent.tag == "figure":
                remove = False  # trust figure-wrapped images

        # KEEP if: has substantial alt text (suggests editorial intent)
        if remove and alt and len(alt) > 20:
            remove = False

        if remove:
            parent = img.getparent()
            if parent is not None:
                # If img is inside a figure, remove the whole figure
                if parent.tag == "figure":
                    grandparent = parent.getparent()
                    if grandparent is not None:
                        grandparent.remove(parent)
                else:
                    parent.remove(img)


def _strip_site_styling(content) -> None:
    """Remove all visual styling from extracted HTML.

    Strips inline styles, classes, and color attributes so our reader
    CSS provides a consistent look across all themes. Preserves
    structural attributes (href, src, alt, colspan, etc.).

    This is what makes reader mode work — like Safari Reader or
    Firefox Reader View, we apply our own typography and colors.
    """
    # Attributes to always remove (visual styling)
    _STRIP_ATTRS = {
        "style", "class", "bgcolor", "color", "background",
        "border", "cellpadding", "cellspacing", "align", "valign",
        "face", "size",  # font attributes
    }

    # Attributes to always keep (structural/semantic)
    _KEEP_ATTRS = {
        "href", "src", "alt", "title", "id", "name",
        "colspan", "rowspan", "scope", "headers",
        "width", "height",  # keep for images (helps layout)
        "type", "start",  # list attributes
        "data-src", "data-srcset", "srcset", "data-lazy-src",
        "loading", "decoding",  # image loading hints
        "target", "rel",  # link attributes
        "data-placeholder",  # our placeholder attr
        "contenteditable",
        # Internationalization & accessibility
        "dir", "lang",  # RTL text, language tagging
        "role",  # ARIA landmark roles
        "aria-label", "aria-describedby",  # useful accessibility hints
        # Responsive images
        "sizes", "media",  # <source> and <img> responsive attrs
        # Media
        "controls", "poster",  # video/audio fallback
        "data-original",  # lazy-load variant
    }

    # --- Pre-pass: collapse <picture> → <img> ---
    # Modern sites ship a low-quality <source> placeholder (blurred base64 or
    # tiny spacer) alongside a real <img> fallback. Browsers pick the
    # <source>, but the <img> still renders — resulting in two stacked
    # images. Replace each <picture> with its inner <img> and drop all
    # <source> children. Falls back to the picture's first image-bearing
    # descendant if no direct <img> child exists.
    for picture in list(content.iter("picture")):
        parent = picture.getparent()
        if parent is None:
            continue
        img = picture.find("img")
        if img is None:
            # Some sites put the img deeper inside picture (wrapped in noscript etc.)
            img = next(iter(picture.iter("img")), None)
        idx = list(parent).index(picture)
        tail = picture.tail
        if img is not None:
            img.tail = tail
            parent.insert(idx, img)
        parent.remove(picture)

    # --- Pre-pass: fix lazy-loaded images (data-src → src) ---
    for img in content.iter("img"):
        src = img.get("src", "")
        if not src or src.startswith("data:"):
            # Copy data-src or data-lazy-src to src so the image actually loads
            fallback = img.get("data-src") or img.get("data-lazy-src") or img.get("data-original")
            if fallback:
                img.set("src", fallback)

    # --- Pre-pass: remove elements that can't render in a reader ---
    # These are interactive/JS-dependent elements that show as blank or broken
    for tag_name in ("canvas", "dialog", "template", "slot",
                     "object", "embed", "applet"):
        for el in list(content.iter(tag_name)):
            try:
                el.getparent().remove(el)
            except Exception as exc:
                log.debug("dom_scrub_remove_failed", scrub="interactive_element", tag=tag_name, error=str(exc))

    # Remove loose form controls (inputs/buttons outside <form> tags)
    for tag_name in ("input", "select", "textarea", "button"):
        for el in list(content.iter(tag_name)):
            # Keep if inside a preserved form-like structure (rare)
            try:
                el.getparent().remove(el)
            except Exception as exc:
                log.debug("dom_scrub_remove_failed", scrub="form_control", tag=tag_name, error=str(exc))

    # Remove custom web components (any tag containing a hyphen = custom element)
    for el in list(content.iter()):
        if hasattr(el, "tag") and isinstance(el.tag, str) and "-" in el.tag:
            # Preserve known valid hyphenated tags
            if el.tag not in ("font-face", "color-profile", "missing-glyph"):
                try:
                    el.getparent().remove(el)
                except Exception as exc:
                    log.debug("dom_scrub_remove_failed", scrub="custom_element", tag=el.tag, error=str(exc))

    # Remove aria-hidden="true" elements (hidden content made visible by style stripping)
    for el in content.xpath("//*[@aria-hidden='true']"):
        try:
            el.getparent().remove(el)
        except Exception as exc:
            log.debug("dom_scrub_remove_failed", scrub="aria_hidden", error=str(exc))

    # Remove elements with hidden display patterns in inline styles
    # (before we strip styles, check for intentionally hidden content)
    for el in content.xpath("//*[@style]"):
        style = (el.get("style") or "").lower()
        if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
            try:
                el.getparent().remove(el)
            except Exception as exc:
                log.debug("dom_scrub_remove_failed", scrub="hidden_style", error=str(exc))

    # --- Attribute stripping ---
    for el in content.iter():
        if not hasattr(el, "attrib"):
            continue

        # Keep Wikipedia/semantic classes we style in CSS
        cls = el.get("class", "")
        _preserve_class = False
        if cls:
            _semantic_classes = ("infobox", "reflist", "reference", "citation",
                                "mw-", "hatnote", "thumb", "tright", "tleft",
                                "gallery", "wikitable", "sortable")
            if any(sc in cls for sc in _semantic_classes):
                _preserve_class = True

        # Collect attrs to remove (can't modify dict during iteration)
        to_remove = []
        for attr in el.attrib:
            attr_lower = attr.lower()
            if attr_lower == "class":
                if not _preserve_class:
                    to_remove.append(attr)
            elif attr_lower in _STRIP_ATTRS:
                to_remove.append(attr)
            elif attr_lower not in _KEEP_ATTRS and attr_lower.startswith("data-"):
                # Remove data-* attrs except the ones we need
                if attr_lower not in ("data-src", "data-srcset", "data-lazy-src",
                                      "data-placeholder", "data-original"):
                    to_remove.append(attr)
            elif attr_lower.startswith("aria-") and attr_lower not in _KEEP_ATTRS:
                # Strip non-essential aria attrs but keep label/describedby
                to_remove.append(attr)

        for attr in to_remove:
            del el.attrib[attr]

    # Strip all on* event handler attributes (XSS vector — onclick, onload, etc.)
    # Runs after main attribute cleaning to catch anything that slipped through.
    for el in content.iter():
        if not hasattr(el, "attrib"):
            continue
        event_attrs = [a for a in el.attrib if a.lower().startswith("on")]
        for attr in event_attrs:
            del el.attrib[attr]

    # Also remove any remaining <style> tags that survived earlier cleanup
    for style_el in list(content.iter("style")):
        parent = style_el.getparent()
        if parent is not None:
            parent.remove(style_el)


def _extract_rich_html(raw_html: str, base_url: str) -> str:
    """Extract main content area with images preserved.

    Trafilatura is great for text but strips images. This function
    finds the article/main content area, removes scripts/nav/ads,
    keeps images and links, and rewrites URLs through the proxy.
    """
    try:
        from lxml import html as lxml_html  # type: ignore
    except ImportError:
        # lxml not available — fall back to regex cleaning
        return _extract_rich_html_regex(raw_html, base_url)

    try:
        doc = lxml_html.fromstring(raw_html)
    except Exception:
        return _extract_rich_html_regex(raw_html, base_url)

    # Remove always-unwanted elements (never contain article content)
    # Preserve YouTube/Vimeo embeds — they're legitimate content iframes.
    _EMBED_WHITELIST = (
        # Video
        "youtube.com/embed", "youtube-nocookie.com/embed",
        "player.vimeo.com", "dailymotion.com/embed",
        "tiktok.com/embed", "clips.twitch.tv/embed",
        "platform.twitter.com/embed",
        # Video (new platforms)
        "player.bilibili.com/player.html",
        "rumble.com/embed",
        "odysee.com/$/embed",
        "player.kick.com",
        "embed.nebula.tv",
        "archive.org/embed",
        # Peertube — federated, match pattern not domain
        # (handled separately in iframe filtering below)
        # Music
        "open.spotify.com/embed",
        "w.soundcloud.com/player",
        "bandcamp.com/EmbeddedPlayer",
        # Code
        "codepen.io/",
        "gist.github.com/",
        "codesandbox.io/embed",
        "stackblitz.com/edit",
        "jsfiddle.net/",
        # Social
        "www.instagram.com/p/",
        "www.facebook.com/plugins",
        # Docs / other
        "docs.google.com/",
        "slides.google.com/",
        "giphy.com/embed",
    )
    for tag in doc.iter("script", "style", "noscript", "form"):
        try:
            tag.getparent().remove(tag)
        except Exception:
            log.debug("browse_strip_tag_failed", tag=tag.tag, exc_info=True)
    for tag in doc.iter("iframe"):
        src = (tag.get("src") or tag.get("data-src") or "").lower()
        if any(domain in src for domain in _EMBED_WHITELIST):
            # Keep video embeds — rewrite src through our proxy-safe CSP
            continue
        # Also allow Peertube embeds (federated — any domain with /videos/embed/ path)
        if "/videos/embed/" in src:
            continue
        try:
            tag.getparent().remove(tag)
        except Exception as exc:
            log.debug("dom_scrub_remove_failed", scrub="non_whitelisted_iframe", src=src[:120], error=str(exc))

    # Remove nav/footer/header/aside only at the TOP level (outside article body).
    # Content-level asides (pull quotes, callout boxes) are preserved because
    # they live inside <article>/<main> and won't be reached by this pass.
    body = doc.find(".//body")
    if body is not None:
        for tag in list(body):
            if tag.tag in ("nav", "footer", "header"):
                try:
                    body.remove(tag)
                except Exception as exc:
                    log.debug("dom_scrub_remove_failed", scrub="top_level_chrome", tag=tag.tag, error=str(exc))

    # Wikipedia-specific cleanup: remove edit links, navboxes, and empty ref lists
    # but KEEP infoboxes and reference content
    _is_wikipedia = "wikipedia.org" in base_url
    if _is_wikipedia:
        # Remove [edit] links
        for el in doc.xpath("//span[@class='mw-editsection']"):
            try:
                el.getparent().remove(el)
            except Exception as exc:
                log.debug("dom_scrub_remove_failed", scrub="wikipedia_editsection", error=str(exc))
        # Remove navigation boxes at the bottom
        for el in doc.xpath(
            "//*[contains(@class,'navbox') or contains(@class,'sistersitebox') "
            "or contains(@class,'catlinks') or contains(@class,'mw-jump-link') "
            "or contains(@class,'noprint') or contains(@class,'mbox-small')]"
        ):
            try:
                el.getparent().remove(el)
            except Exception as exc:
                log.debug("dom_scrub_remove_failed", scrub="wikipedia_navbox", error=str(exc))

    # Remove elements with ad/tracking class patterns — but preserve content sidebars
    _REMOVE_CLASS_RE = re.compile(
        r"(^ad[-_]|[-_]ad[-_]|[-_]ad$|advert|adsense|adslot|"
        r"cookie[-_]|popup|modal[-_]overlay|newsletter[-_]signup|paywall|"
        r"social[-_]share|share[-_]buttons|related[-_]articles|"
        r"site[-_]header|site[-_]footer|site[-_]nav|"
        r"mega[-_]?menu|breadcrumb|skip[-_]link)",
        re.IGNORECASE,
    )
    for el in doc.xpath("//*[@class or @id]"):
        cls = el.get("class", "") + " " + el.get("id", "")
        if _REMOVE_CLASS_RE.search(cls):
            try:
                el.getparent().remove(el)
            except Exception as exc:
                log.debug("dom_scrub_remove_failed", scrub="ad_class_pattern", error=str(exc))

    # Find the best content container — more specific selectors first
    content = None
    for selector in [".article-body", ".post-content", ".entry-content",
                     ".story-body", ".article-content", ".post-body",
                     "#article-body", "#story-body",
                     "[role='article']", "article", "main", "[role='main']",
                     ".mw-parser-output",  # Wikipedia
                     # Commerce / listings
                     ".product-detail", ".product-info", ".listing-details",
                     ".recipe-content", ".recipe-body",
                     "#product-description", "#listing-content",
                     # Forums / discussions
                     ".question", ".post", "#question", ".thread-content",
                     ".content", "#content"]:
        try:
            found = doc.cssselect(selector) if "." in selector or "#" in selector or "[" in selector else doc.iter(selector)
            candidates = list(found)
            if candidates:
                content = candidates[0]
                break
        except Exception as exc:
            log.debug("content_selector_failed", selector=selector, error=str(exc))
            continue

    if content is None:
        content = doc.find(".//body")
    if content is None:
        return _extract_rich_html_regex(raw_html, base_url)

    # Post-extraction cleanup INSIDE the content container.
    # Remove elements that are always noise even inside articles.
    for tag in content.iter("aside"):
        # Keep asides with substantial text (pull quotes, callouts)
        text_len = len((tag.text_content() or "").strip())
        cls = (tag.get("class") or "").lower()
        if text_len < 50 or "ad" in cls or "promo" in cls or "related" in cls:
            try:
                tag.getparent().remove(tag)
            except Exception as exc:
                log.debug("dom_scrub_remove_failed", scrub="aside_promo", error=str(exc))
    # Remove remaining nav/footer inside content (secondary navs, article footers)
    for tag in content.iter("nav", "footer"):
        cls = (tag.get("class") or "").lower()
        # Keep footers with references/citations
        if "reference" in cls or "citation" in cls or "footnote" in cls:
            continue
        try:
            tag.getparent().remove(tag)
        except Exception as exc:
            log.debug("dom_scrub_remove_failed", scrub="secondary_chrome", tag=tag.tag, error=str(exc))

    # Filter images — keep only meaningful content images
    _filter_junk_images(content)

    # Strip all site styling — our reader CSS handles presentation.
    # This is what makes the reader work consistently across all themes.
    _strip_site_styling(content)

    # Serialize back to HTML string
    from lxml.html import tostring as html_tostring
    result = html_tostring(content, encoding="unicode")

    # Rewrite links and image URLs through our proxy
    result = _rewrite_links(result, base_url)

    # Convert YouTube links into embedded players
    result = _youtube_links_to_embeds(result)

    # Convert Spotify/SoundCloud links into embedded players
    result = _music_links_to_embeds(result)

    # Convert video platform links into embedded players
    result = _video_links_to_embeds(result)

    return result


def _extract_rich_html_regex(raw_html: str, base_url: str) -> str:
    """Fallback rich HTML extraction without lxml — regex-based cleaning."""
    # Remove scripts, styles, nav elements
    # Preserve whitelisted embeds before stripping other iframes
    _REGEX_EMBED_KEEP = (
        r"youtube\.com/embed|youtube-nocookie\.com/embed|player\.vimeo\.com"
        r"|open\.spotify\.com/embed|w\.soundcloud\.com/player|bandcamp\.com/EmbeddedPlayer"
        r"|codepen\.io/|codesandbox\.io/embed|stackblitz\.com/edit|jsfiddle\.net/"
        r"|docs\.google\.com/|slides\.google\.com/|giphy\.com/embed"
        r"|player\.bilibili\.com|rumble\.com/embed|odysee\.com/\$/embed"
        r"|player\.kick\.com|embed\.nebula\.tv|archive\.org/embed|/videos/embed/"
    )
    cleaned = re.sub(
        rf'<iframe[^>]*(?:{_REGEX_EMBED_KEEP})[^>]*>.*?</iframe>',
        lambda m: m.group(0).replace('<iframe', '<PRESERVED_IFRAME'),
        raw_html, flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"<(script|style|nav|footer|aside|iframe|noscript|svg)[^>]*>.*?</\1>",
                     "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Restore preserved video embeds
    cleaned = cleaned.replace('<PRESERVED_IFRAME', '<iframe')

    # Try to find article or main content
    for pattern in [r"<article[^>]*>(.*?)</article>",
                    r"<main[^>]*>(.*?)</main>",
                    r'<div[^>]*class="[^"]*(?:content|article|post)[^"]*"[^>]*>(.*?)</div>']:
        m = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if m and len(m.group(1)) > 500:
            cleaned = m.group(1)
            break

    # Rewrite URLs
    cleaned = _rewrite_links(cleaned, base_url)

    return cleaned


# Matches YouTube URLs in href attributes or bare text
_YT_LINK_RE = re.compile(
    r'<a\s[^>]*href=["\']'
    r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?[^"\']*v=|youtu\.be/)'
    r'([A-Za-z0-9_-]{11})[^"\']*["\'][^>]*>'
    r'(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)

# Also catch proxied YouTube links (already rewritten by _rewrite_links)
_YT_PROXIED_LINK_RE = re.compile(
    r'<a\s[^>]*href=["\'][^"\']*youtube\.com%2Fwatch%3F[^"\']*v%3D'
    r'([A-Za-z0-9_-]{11})[^"\']*["\'][^>]*>'
    r'(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)


def _youtube_links_to_embeds(html: str) -> str:
    """Convert YouTube <a> links into inline embedded players.

    Finds links to youtube.com/watch and youtu.be in the article HTML
    and replaces them with responsive iframe embeds. Only converts links
    where the link text looks like a video reference (not inline mentions
    in prose).
    """
    def _make_embed(video_id: str, link_text: str) -> str:
        clean_text = re.sub(r'<[^>]+>', '', link_text).strip()
        return (
            f'<div class="yt-article-embed">'
            f'<iframe src="https://www.youtube.com/embed/{video_id}" '
            f'allowfullscreen frameborder="0" loading="lazy" '
            f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>'
            f'{f"<div class=yt-article-embed-caption>{clean_text}</div>" if clean_text and clean_text != video_id else ""}'
            f'</div>'
        )

    def _replace_link(m: re.Match) -> str:
        video_id = m.group(1)
        link_text = m.group(2).strip()
        # Don't replace inline mentions (short link text embedded in prose)
        # Only replace if the link is its own block or the text is a URL/title
        clean = re.sub(r'<[^>]+>', '', link_text).strip()
        if len(clean) < 3:
            return m.group(0)  # keep tiny links as-is
        return _make_embed(video_id, link_text)

    result = _YT_LINK_RE.sub(_replace_link, html)
    result = _YT_PROXIED_LINK_RE.sub(_replace_link, result)
    return result


# Matches Spotify links: open.spotify.com/track|album|playlist|episode/xxx
_SPOTIFY_LINK_RE = re.compile(
    r'<a\s[^>]*href=["\']'
    r'(?:https?://)?open\.spotify\.com/(track|album|playlist|episode|show)/([A-Za-z0-9]+)[^"\']*["\'][^>]*>'
    r'(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)


def _music_links_to_embeds(html: str) -> str:
    """Convert Spotify links into inline embedded players.

    Finds links to open.spotify.com in the article HTML and replaces
    them with responsive iframe embeds.
    """
    def _replace_spotify(m: re.Match) -> str:
        media_type = m.group(1)  # track, album, playlist, episode, show
        media_id = m.group(2)
        # Height: tracks/episodes = 152, albums/playlists/shows = 352
        # Episodes (video podcasts) get taller embed like albums/playlists
        height = "352" if media_type in ("album", "playlist", "show", "episode") else "152"
        embed_url = f"https://open.spotify.com/embed/{media_type}/{media_id}"
        return (
            f'<iframe src="{embed_url}" width="100%" height="{height}" '
            f'frameborder="0" allow="encrypted-media" loading="lazy" '
            f'style="border-radius:12px"></iframe>'
        )

    return _SPOTIFY_LINK_RE.sub(_replace_spotify, html)


def _video_links_to_embeds(html: str) -> str:
    """Convert video platform links to embedded players."""

    # Bilibili: bilibili.com/video/BVxxxxxx → player embed
    html = re.sub(
        r'<a[^>]*href="https?://(?:www\.)?bilibili\.com/video/(BV[a-zA-Z0-9]+)[^"]*"[^>]*>[^<]*</a>',
        r'<iframe src="https://player.bilibili.com/player.html?bvid=\1&danmaku=0&autoplay=0" '
        r'width="100%" style="aspect-ratio:16/9" frameborder="0" allowfullscreen loading="lazy"></iframe>',
        html,
    )

    # Peertube: {instance}/videos/watch/{id} or {instance}/w/{id}
    html = re.sub(
        r'<a[^>]*href="(https?://[^"]+)/(?:videos/watch|w)/([a-zA-Z0-9-]+)[^"]*"[^>]*>[^<]*</a>',
        r'<iframe src="\1/videos/embed/\2" width="100%" style="aspect-ratio:16/9" '
        r'frameborder="0" allowfullscreen sandbox="allow-same-origin allow-scripts" loading="lazy"></iframe>',
        html,
    )

    # Rumble: rumble.com/{path}.html → embed
    html = re.sub(
        r'<a[^>]*href="https?://(?:www\.)?rumble\.com/([a-zA-Z0-9]+)[^"]*\.html[^"]*"[^>]*>[^<]*</a>',
        r'<iframe src="https://rumble.com/embed/\1/" width="100%" style="aspect-ratio:16/9" '
        r'frameborder="0" allowfullscreen loading="lazy"></iframe>',
        html,
    )

    # Odysee: odysee.com/@channel/video → embed
    html = re.sub(
        r'<a[^>]*href="https?://odysee\.com/(@[^":/]+)/([^":/]+):([a-f0-9]+)[^"]*"[^>]*>[^<]*</a>',
        r'<iframe src="https://odysee.com/$/embed/\2/\3" width="100%" style="aspect-ratio:16/9" '
        r'frameborder="0" allowfullscreen loading="lazy"></iframe>',
        html,
    )

    # Archive.org: archive.org/details/{id} → embed
    html = re.sub(
        r'<a[^>]*href="https?://(?:www\.)?archive\.org/details/([^"/?#]+)[^"]*"[^>]*>[^<]*</a>',
        r'<iframe src="https://archive.org/embed/\1" width="100%" style="aspect-ratio:4/3" '
        r'frameborder="0" allowfullscreen loading="lazy"></iframe>',
        html,
    )

    # Kick: kick.com/{channel} clips → embed
    html = re.sub(
        r'<a[^>]*href="https?://(?:www\.)?kick\.com/([a-zA-Z0-9_-]+)(?:/clip/[^"]+)?[^"]*"[^>]*>[^<]*</a>',
        r'<iframe src="https://player.kick.com/\1" width="100%" style="aspect-ratio:16/9" '
        r'frameborder="0" allowfullscreen loading="lazy"></iframe>',
        html,
    )

    return html


def _rewrite_links(html_content: str, base_url: str) -> str:
    """Rewrite all URL attributes to route through the browse proxy.

    Catches: src, href, srcset, data-src, data-srcset, data-lazy-src, poster.
    Images go through /api/browse/image, links through /api/browse/fetch.
    """
    from urllib.parse import quote as url_quote

    # Attributes that contain image URLs
    _IMG_ATTRS = {"src", "data-src", "data-lazy-src", "data-original", "poster"}
    _LINK_ATTRS = {"href"}

    def _rewrite_attr(m: re.Match) -> str:
        attr = m.group(1).lower()
        quote_char = m.group(2)
        url = m.group(3).strip()
        # Skip anchors, javascript:, data:, mailto:, already-proxied
        if url.startswith(("#", "javascript:", "data:", "mailto:", "/api/browse/")):
            return m.group(0)
        abs_url = urljoin(base_url, url)
        encoded = url_quote(abs_url, safe="")
        if attr in _IMG_ATTRS:
            return f'{m.group(1)}={quote_char}/api/browse/image?url={encoded}{quote_char}'
        if attr in _LINK_ATTRS:
            return f'{m.group(1)}={quote_char}/api/browse/fetch?url={encoded}{quote_char}'
        # Default: proxy as image (safer for unknown attrs)
        return f'{m.group(1)}={quote_char}/api/browse/image?url={encoded}{quote_char}'

    # Match all URL-bearing attributes
    result = re.sub(
        r'(href|src|data-src|data-lazy-src|data-original|poster)\s*=\s*(["\'])(.*?)\2',
        _rewrite_attr,
        html_content,
        flags=re.IGNORECASE,
    )

    # Handle srcset (comma-separated list of URLs with size descriptors)
    def _rewrite_srcset(m: re.Match) -> str:
        attr = m.group(1)
        quote_char = m.group(2)
        raw = m.group(3)
        parts = []
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            tokens = entry.split()
            if tokens:
                url = tokens[0].strip()
                if url.startswith(("data:", "/api/browse/")):
                    parts.append(entry)
                    continue
                abs_url = urljoin(base_url, url)
                encoded = url_quote(abs_url, safe="")
                tokens[0] = f"/api/browse/image?url={encoded}"
            parts.append(" ".join(tokens))
        return f'{attr}={quote_char}{", ".join(parts)}{quote_char}'

    result = re.sub(
        r'(srcset|data-srcset)\s*=\s*(["\'])(.*?)\2',
        _rewrite_srcset,
        result,
        flags=re.IGNORECASE,
    )

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/search")
async def browse_search(
    request: Request,
    q: str = "",
    categories: str = "general",
    page: int = 1,
    time_range: str = "",
    sort_by: str = "",
    duration: str = "",
) -> JSONResponse:
    """Search via SearXNG and return deduplicated results.

    Video-surface params (only meaningful when ``categories`` includes
    ``videos`` or when the result set is all-video, but accepted for any
    category — they degrade harmlessly elsewhere):

    - ``time_range``: ``day|week|month|year`` — forwarded to SearXNG as
      the standard ``time_range`` param. Engines that don't support it
      ignore it silently.
    - ``sort_by``: ``date|duration_asc|duration_desc`` — post-rerank
      reorder. Overrides reputation tiering for the final visible order.
    - ``duration``: ``short|medium|long`` — post-rerank filter on parsed
      duration. Results with no parseable duration are kept (graceful).
    """
    if not q.strip():
        return JSONResponse({"error": "Query parameter 'q' is required"}, status_code=400)

    http_client = getattr(request.app.state, "http_client", None)
    if not http_client:
        return JSONResponse({"error": "HTTP client not available"}, status_code=503)

    base_url = settings.searxng_base_url
    searxng_params: dict[str, str | int] = {
        "q": q.strip(),
        "format": "json",
        "categories": categories,
        "pageno": page,
    }
    if time_range and time_range in _VALID_TIME_RANGES:
        searxng_params["time_range"] = time_range
    try:
        response = await http_client.get(
            f"{base_url}/search",
            params=searxng_params,
            timeout=15.0,
        )
        response.raise_for_status()
    except Exception as exc:
        log.warning("browse_search_failed", query=q, error=str(exc))
        return JSONResponse({"error": f"Search failed: {exc}"}, status_code=502)

    try:
        data = response.json()
    except Exception:
        return JSONResponse({"error": "Failed to parse search response"}, status_code=502)

    raw_results = data.get("results", [])

    # If the query scopes to a specific site (chip click or user-typed
    # `site:X`), enforce it by URL hostname. SearXNG fans queries out to
    # every engine in the category bucket, but engines like Dailymotion
    # don't honor `site:` as an operator — they match it as a literal
    # phrase against titles/descriptions, so unrelated results with
    # `site:youtube.com` in the title leak through. Post-filtering by
    # hostname makes the chip contract deterministic: "YouTube chip →
    # only youtube.com hostnames." Subdomains (m./music./etc.) are kept.
    site_scope = _extract_site_scope(q)
    if site_scope:
        raw_results = [r for r in raw_results if _url_matches_site(r.get("url", ""), site_scope)]

    # Quality filter — drop error pages / CAPTCHA walls / terms-of-service
    # stubs, strip HTML highlight tags out of snippets, rank by relevance.
    # Softer than filter_for_llm: keeps non-English results, file
    # downloads (browse_fetch renders PDFs/code/etc.), and "unfetchable"
    # platform domains (browse_fetch shows a clean open-in-browser card).
    try:
        from augmentum.discovery.quality import filter_for_browse
        filtered = filter_for_browse(raw_results, query=q.strip())
    except Exception:
        log.debug("browse_search_quality_filter_failed", exc_info=True)
        filtered = raw_results

    # Normalize into the result-card shape the UI consumes. Quality filter
    # has already deduped + ranked; here we just project the fields.
    deduped: list[dict] = []
    for result in filtered:
        url = result.get("url", "")
        host = (urlparse(url).hostname or "").lower().removeprefix("www.")
        host_parent = ".".join(host.split(".")[-2:]) if host.count(".") > 1 else host
        is_video = host in _EMBEDDABLE_VIDEO_DOMAINS or host_parent in _EMBEDDABLE_VIDEO_DOMAINS
        deduped.append({
            "title": result.get("title", "Untitled"),
            "url": url,
            "snippet": result.get("content", ""),
            "engine": result.get("engine", ""),
            "engines": result.get("engines", []),
            "published_date": result.get("publishedDate", ""),
            "thumbnail": result.get("thumbnail") or result.get("img_src", ""),
            "category": result.get("category", ""),
            "img_format": result.get("img_format", ""),
            # Video-card fields — SearXNG's video engines set `length`
            # (duration str) and `author` (channel). Missing on non-video
            # engines; the UI only renders them when present.
            "is_video": is_video,
            "duration": str(result.get("length") or result.get("duration") or ""),
            "author": result.get("author", ""),
            # YouTube engine (and a few others) populate `views` as int.
            # Surfaced for the video card so the UI can show "1.2M views"
            # alongside the date — matches YouTube/Google video cards.
            "views": result.get("views"),
        })
        if len(deduped) >= 20:
            break

    # has_more reflects whether SearXNG itself has another page. Use the
    # raw-result count (before our quality filter removes some) so we
    # don't show a stuck "Load More" when filtering happened to drop a
    # page down to 9 entries. SearXNG's default page size is 10 — fewer
    # than 10 from the engines means we're at the tail.
    has_more = len(raw_results) >= 10

    # Seed preferred sources and sync reputation on first search (once per process)
    if not getattr(router, "_sources_seeded", False):
        router._sources_seeded = True
        await _seed_preferred_sources(request)
        # Merge learned reputation into the web tool's quality lookup
        # so AI-initiated searches also benefit from browse tab learning
        all_scores = await _get_domain_scores(request)
        if all_scores:
            from augmentum.tools.preferred_sources import merge_learned_reputation
            updated = merge_learned_reputation(all_scores)
            if updated:
                log.info("reputation_merged_to_web_tool", domains_updated=updated)

    # Re-rank by domain reputation + topic relevance to the query.
    # Passing q lets _rank_results demote off-topic EXCELLENT sources (so
    # a biology query doesn't surface arxiv at the top) and the diversity
    # cap inside the ranker prevents a single domain from monopolising
    # the visible results.
    scores = await _get_domain_scores(request)
    is_video = categories.strip().lower() in ("videos", "video")
    if scores or is_video or q.strip():
        deduped = _rank_results(
            deduped,
            scores or {},
            boost_embeddable_video=is_video,
            prefer_english=is_video,
            query=q.strip(),
        )

    # Duration filter — post-rerank so the reranker still gets its full
    # tier signal. Unknown durations pass through (see _filter_by_duration).
    if duration:
        deduped = _filter_by_duration(deduped, duration)

    # Explicit user sort overrides the rerank's tier order. Applied last
    # so the result list reflects exactly what the dropdown says.
    if sort_by and sort_by in _VALID_SORTS:
        deduped = _apply_sort(deduped, sort_by)

    # Attach reputation indicator to each result
    for r in deduped:
        hostname = urlparse(r.get("url", "")).hostname or ""
        domain = hostname.lower().removeprefix("www.")
        s = scores.get(domain)
        if s is None:
            r["reputation"] = "unknown"
        elif s >= 5:
            r["reputation"] = "excellent"
        elif s >= 0:
            r["reputation"] = "good"
        else:
            r["reputation"] = "bad"

    return JSONResponse({
        "results": deduped,
        "query": q.strip(),
        "page": page,
        "has_more": has_more,
        "filters": {
            "time_range": time_range if time_range in _VALID_TIME_RANGES else "",
            "sort_by": sort_by if sort_by in _VALID_SORTS else "",
            "duration": duration if duration in _DURATION_BUCKETS else "",
        },
    })


# Video platform patterns: (url_regex, embed_url_template, oembed_url, site_name)
_VIDEO_PLATFORMS = [
    # Vimeo: vimeo.com/123456789
    (
        re.compile(r"(?:https?://)?(?:www\.)?vimeo\.com/(\d+)", re.IGNORECASE),
        "https://player.vimeo.com/video/{id}?byline=0&portrait=0",
        "https://vimeo.com/api/oembed.json?url=https://vimeo.com/{id}",
        "Vimeo",
    ),
    # Dailymotion: dailymotion.com/video/x8abc12
    (
        re.compile(r"(?:https?://)?(?:www\.)?dailymotion\.com/video/([a-zA-Z0-9]+)", re.IGNORECASE),
        "https://www.dailymotion.com/embed/video/{id}",
        "https://www.dailymotion.com/services/oembed?url=https://www.dailymotion.com/video/{id}&format=json",
        "Dailymotion",
    ),
    # TikTok: tiktok.com/@user/video/1234567890
    (
        re.compile(r"(?:https?://)?(?:www\.)?tiktok\.com/@[^/]+/video/(\d+)", re.IGNORECASE),
        "https://www.tiktok.com/embed/v2/{id}",
        "https://www.tiktok.com/oembed?url={url}&format=json",
        "TikTok",
    ),
    # Twitch clips: clips.twitch.tv/ClipName or twitch.tv/*/clip/ClipName
    (
        re.compile(r"(?:https?://)?clips\.twitch\.tv/([A-Za-z0-9_-]+)", re.IGNORECASE),
        "https://clips.twitch.tv/embed?clip={id}&parent=localhost&autoplay=false",
        None,
        "Twitch",
    ),
    (
        re.compile(r"(?:https?://)?(?:www\.)?twitch\.tv/\w+/clip/([A-Za-z0-9_-]+)", re.IGNORECASE),
        "https://clips.twitch.tv/embed?clip={id}&parent=localhost&autoplay=false",
        None,
        "Twitch",
    ),
    # Twitter/X video posts
    (
        re.compile(r"(?:https?://)?(?:twitter\.com|x\.com)/\w+/status/(\d+)", re.IGNORECASE),
        "https://platform.twitter.com/embed/Tweet.html?id={id}",
        "https://publish.twitter.com/oembed?url={url}&format=json",
        "X (Twitter)",
    ),
    # Twitch VODs (recorded streams — separate URL shape from clips)
    (
        re.compile(r"(?:https?://)?(?:www\.)?twitch\.tv/videos/(\d+)", re.IGNORECASE),
        "https://player.twitch.tv/?video={id}&parent=localhost&autoplay=false",
        None,
        "Twitch",
    ),
    # Reddit video posts — redditmedia.com serves the embed iframe. Use the
    # base post URL (sub + id) since the post slug varies and isn't required
    # by the embed. showmedia=true so v.redd.it videos actually render (the
    # default embed suppresses inline media to reduce tracker footprint).
    (
        re.compile(r"(?:https?://)?(?:www\.|old\.|new\.)?reddit\.com/r/(\w+)/comments/([a-z0-9]+)", re.IGNORECASE),
        "https://www.redditmedia.com/r/{id[0]}/comments/{id[1]}/?ref_source=embed&ref=share&embed=true&showmedia=true",
        None,
        "Reddit",
    ),
    # Archive.org items — /details/<id> → /embed/<id> handles video, audio,
    # books (with a reader), and software (with JSMESS). One of the longest
    # tails of media types on the modern web, and the embed URL is stable.
    (
        re.compile(r"(?:https?://)?(?:www\.)?archive\.org/details/([A-Za-z0-9_\-.]+)", re.IGNORECASE),
        "https://archive.org/embed/{id}",
        None,
        "Internet Archive",
    ),
    # Spotify — tracks, albums, playlists, episodes, shows. Embed URL is the
    # same shape for every content type, just swap the kind. Preserve the
    # kind via a regex that captures both segments.
    (
        re.compile(r"(?:https?://)?open\.spotify\.com/(track|album|playlist|episode|show)/([A-Za-z0-9]+)", re.IGNORECASE),
        "https://open.spotify.com/embed/{id[0]}/{id[1]}",
        "https://open.spotify.com/oembed?url={url}",
        "Spotify",
    ),
    # Apple Music — tracks/albums. Embed subdomain is the same shape.
    (
        re.compile(r"(?:https?://)?music\.apple\.com/([a-z]{2}/(?:album|playlist|song)/[^?#\s]+)", re.IGNORECASE),
        "https://embed.music.apple.com/{id}",
        None,
        "Apple Music",
    ),
    # Instagram posts/reels — /embed suffix returns a ready-made iframe body.
    # Captures the kind so reels vs feed posts keep their respective layouts.
    (
        re.compile(r"(?:https?://)?(?:www\.)?instagram\.com/(p|reel|tv)/([A-Za-z0-9_-]+)", re.IGNORECASE),
        "https://www.instagram.com/{id[0]}/{id[1]}/embed",
        None,
        "Instagram",
    ),
    # Bluesky posts — embed.bsky.app serves an iframeable widget. Capture
    # handle + post id so the URL maps cleanly.
    (
        re.compile(r"(?:https?://)?bsky\.app/profile/([^/\s]+)/post/([A-Za-z0-9]+)", re.IGNORECASE),
        "https://embed.bsky.app/embed/{id[0]}/app.bsky.feed.post/{id[1]}",
        None,
        "Bluesky",
    ),
    # Imgur galleries/albums. Imgur single-image pages (/a/ID without /embed)
    # auto-redirect, so /embed returns the rendered player for gifs + albums.
    (
        re.compile(r"(?:https?://)?(?:www\.)?imgur\.com/(a|gallery)/([A-Za-z0-9]+)", re.IGNORECASE),
        "https://imgur.com/{id[0]}/{id[1]}/embed",
        None,
        "Imgur",
    ),
]


# Direct-media URL patterns. Recognised by file extension so we can skip the
# article extractor entirely for URLs that point at a raw media file. Grouped
# by render strategy so the frontend knows what tag to wrap around the src.
#
# Kept broad on purpose — searxng returns .mp4/.mp3/.pdf/.webp URLs constantly
# (especially for academic papers, image hosts, and CDN hotlinks) and the
# article extractor would just render a blank page for them.
_DIRECT_MEDIA_EXT: dict[str, str] = {
    # Video
    ".mp4": "video", ".webm": "video", ".ogv": "video", ".mov": "video",
    ".m4v": "video",
    # Imgur .gifv → looping silent mp4 served under the same path with .mp4
    ".gifv": "video",
    # Audio
    ".mp3": "audio", ".ogg": "audio", ".oga": "audio", ".wav": "audio",
    ".flac": "audio", ".m4a": "audio", ".opus": "audio", ".aac": "audio",
    ".weba": "audio",
    # Images
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image",
    ".webp": "image", ".avif": "image", ".svg": "image", ".bmp": "image",
    # Documents
    ".pdf": "pdf",
}


async def _try_direct_media(url: str, request) -> dict | None:
    """Render a direct-media URL natively instead of running it through the
    article extractor, which produces a blank page for raw media.

    Keyed purely off the URL suffix (cheap, no network) — if a URL looks like
    a media file, we trust it. Content-Type sniffing would be more robust but
    would add a HEAD round-trip to every fetch, and false positives here are
    self-correcting (the native element just shows a broken-media icon).
    """
    parsed = urlparse(url)
    path = (parsed.path or "").lower()
    kind: str | None = None
    gif_like = False  # autoplay silent loop — gifv / .gif-promoted-to-video
    for ext, k in _DIRECT_MEDIA_EXT.items():
        if path.endswith(ext):
            kind = k
            # imgur .gifv → swap to .mp4: imgur serves a silent looped MP4
            # at the same path with .mp4 as the extension and it plays
            # dramatically better than trying to treat the .gifv page as a
            # video (which isn't — it's an HTML wrapper). Mark as gif-like
            # so the emitted <video> gets autoplay/muted/loop — matching how
            # imgur renders them on its own site.
            if ext == ".gifv":
                url = url[: -len(".gifv")] + ".mp4"
                gif_like = True
            break
    if not kind:
        return None

    safe_url = url  # already-unquoted; <video>/<img> src accepts raw URL
    hostname = parsed.hostname or ""
    filename = (path.rsplit("/", 1)[-1] or "media").strip()

    if kind == "video":
        loop_attrs = ' autoplay muted loop' if gif_like else ''
        body = (
            f'<div class="browse-direct-media browse-direct-media--video">'
            f'<video controls preload="metadata" playsinline{loop_attrs} src="{safe_url}">'
            f'Your browser can\'t play this video. '
            f'<a href="{safe_url}" target="_blank" rel="noopener">Download</a>'
            f'</video></div>'
        )
    elif kind == "audio":
        body = (
            f'<div class="browse-direct-media browse-direct-media--audio">'
            f'<audio controls preload="metadata" src="{safe_url}">'
            f'Your browser can\'t play this audio. '
            f'<a href="{safe_url}" target="_blank" rel="noopener">Download</a>'
            f'</audio></div>'
        )
    elif kind == "image":
        body = (
            f'<div class="browse-direct-media browse-direct-media--image">'
            f'<img src="{safe_url}" alt="{filename}" loading="lazy">'
            f'</div>'
        )
    else:  # pdf
        body = (
            f'<div class="browse-direct-media browse-direct-media--pdf">'
            f'<iframe src="{safe_url}" loading="lazy" '
            f'title="PDF: {filename}" style="width:100%;height:80vh;border:0"></iframe>'
            f'<p style="margin-top:0.5rem"><a href="{safe_url}" target="_blank" rel="noopener">'
            f'Open PDF in new tab</a> if it didn\'t load above.</p></div>'
        )

    return {
        "html": body,
        "text": f"{kind.capitalize()} file: {filename}",
        "title": filename,
        "author": "",
        "date": "",
        "sitename": hostname,
        "word_count": 0,
        "reading_time_min": 0,
        "url": url,
        "favicon_url": f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3D{hostname}%26sz%3D32",
        "source": f"direct-{kind}",
        "page_type": "video" if kind == "video" else ("reference" if kind == "pdf" else "article"),
    }


def _fill_template(template: str, groups: tuple[str, ...], url: str) -> str:
    """Fill an embed/oembed URL template from regex capture groups.

    Supports both single-group (``{id}``) and multi-group (``{id[0]}``,
    ``{id[1]}``, ...) forms, plus ``{url}`` for oEmbed endpoints that take
    the original URL as a query param.
    """
    out = template
    if len(groups) == 1:
        out = out.replace("{id}", groups[0])
    for i, g in enumerate(groups):
        out = out.replace(f"{{id[{i}]}}", g)
    out = out.replace("{url}", url)
    return out


async def _try_video_embed(url: str, request) -> dict | None:
    """Check if URL matches a known video platform and return embed response."""
    for pattern, embed_template, oembed_template, site_name in _VIDEO_PLATFORMS:
        m = pattern.search(url)
        if not m:
            continue

        groups = m.groups()
        embed_url = _fill_template(embed_template, groups, url)
        title = ""
        author = ""
        thumbnail = ""

        # Fetch metadata via oEmbed if available
        if oembed_template:
            http_client = getattr(request.app.state, "http_client", None)
            if http_client:
                try:
                    oembed_url = _fill_template(oembed_template, groups, url)
                    resp = await http_client.get(oembed_url, timeout=8.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        title = data.get("title", "")
                        author = data.get("author_name", "")
                        thumbnail = data.get("thumbnail_url", "")
                except Exception:
                    log.debug("oembed_template_fetch_failed", url=url, exc_info=True)

        # Build embed HTML — add platform class for layout-specific CSS.
        # The "Open in <site>" link below the iframe is the only reliable
        # fallback for X-Frame-Options / CSP frame-ancestors rejections: a
        # blocked iframe still fires `load` cross-origin, so JS can't tell
        # the difference. Showing the link unconditionally means users never
        # get stranded on a silent blank embed.
        platform_class = site_name.lower().replace(" ", "-").replace("(", "").replace(")", "")
        embed_html = (
            f'<div class="yt-article-embed embed-{platform_class}">'
            f'<iframe src="{embed_url}" '
            f'allowfullscreen frameborder="0" loading="lazy" '
            f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture">'
            f'</iframe>'
            f'<div class="yt-article-embed-fallback">'
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">Open on {site_name} →</a>'
            f'</div></div>'
        )

        parsed = urlparse(url)
        favicon_domain = parsed.hostname or site_name.lower().replace(" ", "")
        return {
            "html": embed_html,
            "text": title or f"{site_name} video",
            "title": title or f"{site_name} Video",
            "author": author,
            "date": "",
            "sitename": site_name,
            "word_count": 0,
            "reading_time_min": 0,
            "url": url,
            "favicon_url": f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3D{favicon_domain}%26sz%3D32",
            "source": f"{site_name.lower()}-embed",
        }

    return None


# oEmbed discovery regex — finds <link rel="alternate" type="application/json+oembed">
_OEMBED_LINK_RE = re.compile(
    r'<link[^>]+type=["\']application/json\+oembed["\'][^>]+href=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_OEMBED_LINK_RE2 = re.compile(
    r'<link[^>]+href=["\']([^"\']+)["\'][^>]+type=["\']application/json\+oembed["\']',
    re.IGNORECASE,
)
# OpenGraph video meta tag
_OG_VIDEO_RE = re.compile(
    r'<meta[^>]+property=["\']og:video(?::secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_OG_VIDEO_RE2 = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:video(?::secure_url)?["\']',
    re.IGNORECASE,
)
_OG_TITLE_RE = re.compile(
    r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_OG_SITE_RE = re.compile(
    r'<meta[^>]+property=["\']og:site_name["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


async def _try_oembed_discovery(url: str, request) -> dict | None:
    """Generic video embed discovery via oEmbed and OpenGraph.

    Fetches just the page head to find oEmbed or og:video tags,
    then renders an embedded player. Works for any site that supports
    the oEmbed standard or OpenGraph video tags.
    """
    http_client = getattr(request.app.state, "http_client", None)
    if not http_client:
        return None

    # Fetch just the first ~15KB to find <head> tags — don't download the whole page
    try:
        resp = await http_client.get(url, timeout=8.0, headers={
            "User-Agent": "Mozilla/5.0 (compatible; Augmentum/1.0)",
            "Range": "bytes=0-15000",
        })
        head_html = resp.text[:15000]
    except Exception:
        return None

    parsed = urlparse(url)
    site_name = ""
    title = ""

    # Try OpenGraph metadata first (available on most pages)
    og_title_m = _OG_TITLE_RE.search(head_html)
    if og_title_m:
        title = og_title_m.group(1)
    og_site_m = _OG_SITE_RE.search(head_html)
    if og_site_m:
        site_name = og_site_m.group(1)

    # Strategy 1: oEmbed discovery — the gold standard
    oembed_url = None
    for pattern in (_OEMBED_LINK_RE, _OEMBED_LINK_RE2):
        m = pattern.search(head_html)
        if m:
            oembed_url = m.group(1).replace("&amp;", "&")
            break

    if oembed_url:
        try:
            oembed_resp = await http_client.get(oembed_url, timeout=8.0)
            if oembed_resp.status_code == 200:
                data = oembed_resp.json()
                embed_html_raw = data.get("html", "")
                if embed_html_raw and "<iframe" in embed_html_raw.lower():
                    # Wrap in our container + append an "Open on <site>" link
                    # so users have a path out when the embed silently fails
                    # to render (blocked by X-Frame-Options, auth wall, etc.)
                    provider = data.get("provider_name") or site_name or (parsed.hostname or "site")
                    embed_html = (
                        f'<div class="yt-article-embed">{embed_html_raw}'
                        f'<div class="yt-article-embed-fallback">'
                        f'<a href="{url}" target="_blank" rel="noopener noreferrer">Open on {provider} →</a>'
                        f'</div></div>'
                    )
                    return {
                        "html": embed_html,
                        "text": data.get("title", title) or f"{site_name} video",
                        "title": data.get("title", title) or f"Video",
                        "author": data.get("author_name", ""),
                        "date": "",
                        "sitename": site_name or data.get("provider_name", parsed.hostname or ""),
                        "word_count": 0,
                        "reading_time_min": 0,
                        "url": url,
                        "favicon_url": f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3D{parsed.hostname}%26sz%3D32",
                        "source": "oembed-discovery",
                    }
        except Exception:
            log.debug("oembed_discovery_failed", url=url, exc_info=True)

    # Strategy 2: OpenGraph og:video tag — fallback
    og_video_url = None
    for pattern in (_OG_VIDEO_RE, _OG_VIDEO_RE2):
        m = pattern.search(head_html)
        if m:
            og_video_url = m.group(1).replace("&amp;", "&")
            break

    if og_video_url and ("embed" in og_video_url or "player" in og_video_url):
        provider = site_name or parsed.hostname or "site"
        embed_html = (
            f'<div class="yt-article-embed">'
            f'<iframe src="{og_video_url}" '
            f'allowfullscreen frameborder="0" loading="lazy" '
            f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture">'
            f'</iframe>'
            f'<div class="yt-article-embed-fallback">'
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">Open on {provider} →</a>'
            f'</div></div>'
        )
        return {
            "html": embed_html,
            "text": title or f"Video from {parsed.hostname}",
            "title": title or f"Video",
            "author": "",
            "date": "",
            "sitename": site_name or parsed.hostname or "",
            "word_count": 0,
            "reading_time_min": 0,
            "url": url,
            "favicon_url": f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3D{parsed.hostname}%26sz%3D32",
            "source": "og-video",
        }

    return None


# -----------------------------------------------------------------------
# File-type handlers — render non-HTML content appropriately
# -----------------------------------------------------------------------

# Extension → (category, mime_hint)
_FILE_TYPES: dict[str, tuple[str, str]] = {
    # PDF
    ".pdf": ("pdf", "application/pdf"),
    # Images
    ".png": ("image", "image/png"), ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"), ".gif": ("image", "image/gif"),
    ".webp": ("image", "image/webp"), ".svg": ("image", "image/svg+xml"),
    ".avif": ("image", "image/avif"), ".bmp": ("image", "image/bmp"),
    ".ico": ("image", "image/x-icon"),
    # Text / code
    ".txt": ("text", "text/plain"), ".log": ("text", "text/plain"),
    ".csv": ("text", "text/csv"), ".tsv": ("text", "text/tsv"),
    ".json": ("code", "application/json"), ".xml": ("code", "text/xml"),
    ".yaml": ("code", "text/yaml"), ".yml": ("code", "text/yaml"),
    ".toml": ("code", "text/toml"),
    ".md": ("markdown", "text/markdown"),
    ".py": ("code", "text/x-python"), ".js": ("code", "text/javascript"),
    ".ts": ("code", "text/typescript"), ".html": ("skip", ""),
    ".css": ("code", "text/css"), ".sh": ("code", "text/x-shellscript"),
    ".bash": ("code", "text/x-shellscript"),
    ".rs": ("code", "text/x-rust"), ".go": ("code", "text/x-go"),
    ".java": ("code", "text/x-java"), ".c": ("code", "text/x-c"),
    ".cpp": ("code", "text/x-c++"), ".h": ("code", "text/x-c"),
    ".rb": ("code", "text/x-ruby"), ".php": ("code", "text/x-php"),
    ".sql": ("code", "text/x-sql"), ".r": ("code", "text/x-r"),
    ".swift": ("code", "text/x-swift"), ".kt": ("code", "text/x-kotlin"),
    ".lua": ("code", "text/x-lua"), ".pl": ("code", "text/x-perl"),
    ".ini": ("code", "text/plain"), ".cfg": ("code", "text/plain"),
    ".conf": ("code", "text/plain"), ".env": ("code", "text/plain"),
    # Jupyter notebooks — JSON, but rendered cell-by-cell
    ".ipynb": ("notebook", "application/x-ipynb+json"),
    # Audio
    ".mp3": ("audio", "audio/mpeg"), ".wav": ("audio", "audio/wav"),
    ".ogg": ("audio", "audio/ogg"), ".flac": ("audio", "audio/flac"),
    ".m4a": ("audio", "audio/mp4"), ".aac": ("audio", "audio/aac"),
    ".wma": ("audio", "audio/x-ms-wma"),
    # Video files (native HTML5 playback)
    ".mp4": ("video", "video/mp4"), ".webm": ("video", "video/webm"),
    ".ogv": ("video", "video/ogg"), ".mov": ("video", "video/quicktime"),
    ".m4v": ("video", "video/mp4"), ".avi": ("video", "video/x-msvideo"),
    ".mkv": ("video", "video/x-matroska"),
    # Documents (download-only — can't render in browser)
    ".docx": ("download", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ".xlsx": ("download", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ".pptx": ("download", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    ".doc": ("download", "application/msword"),
    ".xls": ("download", "application/vnd.ms-excel"),
    ".ppt": ("download", "application/vnd.ms-powerpoint"),
    ".odt": ("download", "application/vnd.oasis.opendocument.text"),
    ".ods": ("download", "application/vnd.oasis.opendocument.spreadsheet"),
    ".epub": ("download", "application/epub+zip"),
    ".zip": ("download", "application/zip"),
    ".tar": ("download", "application/x-tar"),
    ".gz": ("download", "application/gzip"),
    ".7z": ("download", "application/x-7z-compressed"),
    ".rar": ("download", "application/vnd.rar"),
}

# Notebook output MIME priority — first match wins per output. Richer
# formats (interactive HTML, then images, then plain text) come first
# so a cell with multiple representations renders the best one.
_NB_OUTPUT_MIME_ORDER = (
    "text/html",
    "image/svg+xml",
    "image/png",
    "image/jpeg",
    "image/gif",
    "text/markdown",
    "text/plain",
)

# Strip ANSI CSI sequences from notebook tracebacks. Jupyter saves the
# colored traceback verbatim and the escape codes garble the rendered
# error block otherwise.
_NB_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Notebook text/html outputs occasionally embed <script> tags
# (Plotly, Bokeh, ipywidgets). Strip them — we can't safely execute
# untrusted JS inside the reader.
_NB_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)


def _render_notebook_markdown(src: str) -> str:
    """Minimal markdown → HTML for notebook markdown cells.

    Mirrors the existing markdown file branch but escapes first so
    arbitrary user content stays safe. Headings, bold, italic, inline
    code, lists. Anything richer (tables, LaTeX, image references) falls
    back to the escaped + paragraph-wrapped form.
    """
    md = _esc(src)
    md = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', md, flags=re.MULTILINE)
    md = re.sub(r'^### (.+)$', r'<h3>\1</h3>', md, flags=re.MULTILINE)
    md = re.sub(r'^## (.+)$', r'<h2>\1</h2>', md, flags=re.MULTILINE)
    md = re.sub(r'^# (.+)$', r'<h1>\1</h1>', md, flags=re.MULTILINE)
    md = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', md)
    md = re.sub(r'\*(.+?)\*', r'<em>\1</em>', md)
    md = re.sub(r'`(.+?)`', r'<code>\1</code>', md)
    md = re.sub(r'^- (.+)$', r'<li>\1</li>', md, flags=re.MULTILINE)
    md = md.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f"<p>{md}</p>"


def _render_notebook_output(out: dict) -> str:
    """Render a single notebook output cell to HTML.

    Returns an empty string for unsupported output types so unknown
    formats don't break the surrounding render.
    """
    otype = out.get("output_type", "")

    if otype == "stream":
        text = out.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        text = _NB_ANSI_RE.sub("", text or "")
        if not text.strip():
            return ""
        stream_name = out.get("name", "stdout")
        cls = "nb-out-stderr" if stream_name == "stderr" else "nb-out-stdout"
        return f'<pre class="nb-output {cls}">{_esc(text)}</pre>'

    if otype in ("display_data", "execute_result"):
        data = out.get("data", {})
        if not isinstance(data, dict):
            return ""

        for mime in _NB_OUTPUT_MIME_ORDER:
            if mime not in data:
                continue
            payload = data[mime]
            if isinstance(payload, list):
                payload = "".join(payload)
            if not payload:
                continue

            if mime == "image/svg+xml":
                # SVG can contain script tags too.
                svg = _NB_SCRIPT_RE.sub("", payload)
                return f'<div class="nb-output nb-out-svg">{svg}</div>'
            if mime.startswith("image/"):
                # PNG/JPEG/GIF are base64-encoded in the notebook spec.
                return (
                    f'<img class="nb-output nb-out-img" '
                    f'src="data:{mime};base64,{payload.strip()}" '
                    f'alt="output" loading="lazy">'
                )
            if mime == "text/html":
                clean = _NB_SCRIPT_RE.sub("", payload)
                return f'<div class="nb-output nb-out-html">{clean}</div>'
            if mime == "text/markdown":
                return (
                    f'<div class="nb-output nb-out-md">'
                    f'{_render_notebook_markdown(payload)}'
                    f'</div>'
                )
            # text/plain — final fallback
            return f'<pre class="nb-output nb-out-text">{_esc(payload)}</pre>'
        return ""

    if otype == "error":
        tb = out.get("traceback", []) or []
        if isinstance(tb, list):
            tb = "\n".join(tb)
        tb = _NB_ANSI_RE.sub("", tb)
        if not tb.strip():
            ename = out.get("ename", "Error")
            evalue = out.get("evalue", "")
            tb = f"{ename}: {evalue}"
        return f'<pre class="nb-output nb-out-error">{_esc(tb)}</pre>'

    return ""


# Language hints for code syntax highlighting (Prism.js class names)
_CODE_LANG_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".json": "json", ".xml": "xml", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".css": "css", ".html": "html",
    ".sh": "bash", ".bash": "bash", ".sql": "sql",
    ".rs": "rust", ".go": "go", ".java": "java",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".rb": "ruby",
    ".php": "php", ".r": "r", ".swift": "swift",
    ".kt": "kotlin", ".lua": "lua", ".pl": "perl",
}


def _handle_file_type(url: str, raw_html: str, fetch_meta: dict) -> dict | None:
    """Detect non-HTML file types and render them appropriately.

    Returns a response dict, or None if the URL is a normal web page.
    """
    # Detect file extension from URL (strip query/fragment)
    url_path = urlparse(url).path.lower().rstrip("/")
    ext = ""
    for e in _FILE_TYPES:
        if url_path.endswith(e):
            ext = e
            break

    # Also detect PDF from content (binary magic bytes)
    if not ext and raw_html and raw_html[:10].startswith("%PDF"):
        ext = ".pdf"

    # Also detect binary content without known extension
    if not ext and _is_binary_response(raw_html):
        return {
            "html": "", "text": "",
            "title": _extract_title(raw_html) or url,
            "author": "", "date": "",
            "sitename": urlparse(url).hostname or "",
            "word_count": 0, "reading_time_min": 0,
            "url": str(fetch_meta.get("url", url)),
            "favicon_url": f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3D{urlparse(url).hostname}%26sz%3D32",
            "source": "direct",
            "error": "This URL points to a binary file that cannot be displayed.",
        }

    if not ext:
        return None  # Normal web page — proceed with HTML extraction

    category, _ = _FILE_TYPES[ext]
    if category == "skip":
        return None  # .html files are normal web pages

    parsed = urlparse(url)
    filename = url_path.rsplit("/", 1)[-1] if "/" in url_path else f"file{ext}"
    proxy_url = f"/api/browse/image?url={quote_plus(url)}"
    favicon = f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3D{parsed.hostname}%26sz%3D32"
    clean_title = filename.rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()

    base = {
        "author": "", "date": "",
        "sitename": parsed.hostname or "",
        "url": str(fetch_meta.get("url", url)),
        "favicon_url": favicon,
    }

    if category == "pdf":
        log.info("browse_file_pdf", url=url)
        # Try text extraction
        pdf_text = ""
        try:
            import pymupdf  # type: ignore[import-untyped]
            if raw_html and raw_html[:10].startswith("%PDF"):
                doc = pymupdf.open(stream=raw_html.encode("latin-1"), filetype="pdf")
                pages = [doc[i].get_text() for i in range(min(len(doc), 20))]
                pdf_text = "\n\n".join(p.strip() for p in pages if p.strip())
                doc.close()
        except Exception:
            # pymupdf optional / PDF parse can fail on corrupt streams —
            # fall through to the iframe-only embed below.
            log.debug("pdf_text_extract_failed", url=url, exc_info=True)
        return {
            **base,
            "html": (
                f'<div class="browse-pdf-embed">'
                f'<iframe src="{proxy_url}" class="browse-pdf-viewer"></iframe>'
                f'<div class="browse-pdf-actions">'
                f'<a href="{proxy_url}" download="{_esc(filename)}" class="browse-pdf-download">Download PDF</a>'
                f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer" class="browse-pdf-original">Open Original</a>'
                f'</div></div>'
            ),
            "text": pdf_text or f"PDF document: {filename}",
            "title": clean_title,
            "word_count": len(pdf_text.split()) if pdf_text else 0,
            "reading_time_min": max(1, len(pdf_text.split()) // 238) if pdf_text else 0,
            "source": "pdf",
        }

    if category == "image":
        log.info("browse_file_image", url=url)
        return {
            **base,
            "html": (
                f'<div class="browse-image-embed">'
                f'<img src="{proxy_url}" alt="{_esc(filename)}" class="browse-image-full">'
                f'<div class="browse-pdf-actions">'
                f'<a href="{proxy_url}" download="{_esc(filename)}">Download Image</a>'
                f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">Open Original</a>'
                f'</div></div>'
            ),
            "text": f"Image: {filename}",
            "title": clean_title,
            "word_count": 0, "reading_time_min": 0,
            "source": "image",
        }

    if category in ("text", "code"):
        log.info("browse_file_text", url=url, ext=ext)
        lang = _CODE_LANG_MAP.get(ext, "")
        lang_class = f' class="language-{lang}"' if lang else ""
        # raw_html is actually the file content since it was fetched as text
        content = raw_html or ""
        # Pretty-print structured formats so users can actually read them
        # instead of staring at a single-line dump. Fail open: leave the
        # content untouched on parse error so we don't lose information.
        stripped = content.strip()
        if ext == ".json" and stripped:
            try:
                content = json.dumps(
                    json.loads(stripped), indent=2, ensure_ascii=False
                )
            except Exception:
                log.debug("browse_json_pretty_print_failed", exc_info=True)
        elif ext == ".xml" and stripped:
            try:
                import xml.dom.minidom as _minidom
                pretty = _minidom.parseString(stripped).toprettyxml(indent="  ")
                # minidom returns the XML decl on its own line; drop blank
                # lines between elements introduced by toprettyxml so the
                # output is dense but still readable.
                lines = [ln for ln in pretty.splitlines() if ln.strip()]
                content = "\n".join(lines)
            except Exception:
                # Malformed XML — leave content untouched per the
                # docstring "Fail open" contract above.
                log.debug("browse_xml_pretty_print_failed", exc_info=True)
        if len(content) > 100_000:
            content = content[:100_000] + "\n\n... [truncated at 100KB]"
        escaped = _esc(content)
        return {
            **base,
            "html": (
                f'<div class="browse-code-embed">'
                f'<div class="browse-code-header">{_esc(filename)}</div>'
                f'<pre><code{lang_class}>{escaped}</code></pre>'
                f'<div class="browse-pdf-actions">'
                f'<a href="{proxy_url}" download="{_esc(filename)}">Download</a>'
                f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">Open Original</a>'
                f'</div></div>'
            ),
            "text": content,
            "title": filename,
            "word_count": len(content.split()),
            "reading_time_min": max(1, len(content.split()) // 238),
            "source": "text",
        }

    if category == "notebook":
        log.info("browse_file_notebook", url=url)
        try:
            nb = json.loads(raw_html)
            if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
                return None
        except Exception:
            log.debug("notebook_parse_failed", url=url, exc_info=True)
            return None

        kernel_lang = (
            (nb.get("metadata") or {}).get("kernelspec", {}).get("language")
            or "python"
        )
        lang_class = _CODE_LANG_MAP.get(f".{kernel_lang}", kernel_lang)

        rendered: list[str] = []
        text_parts: list[str] = []
        word_count = 0

        for cell in nb["cells"]:
            if not isinstance(cell, dict):
                continue
            ctype = cell.get("cell_type", "")
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            src = src or ""

            if ctype == "markdown":
                rendered.append(
                    f'<div class="nb-cell nb-cell-md">'
                    f'{_render_notebook_markdown(src)}'
                    f'</div>'
                )
                text_parts.append(src)
                word_count += len(src.split())

            elif ctype == "code":
                src_html = _esc(src)
                exec_count = cell.get("execution_count")
                count_badge = (
                    f'<span class="nb-exec-count">[{exec_count}]</span>'
                    if exec_count is not None
                    else '<span class="nb-exec-count nb-exec-empty">[&nbsp;]</span>'
                )
                outputs_html = []
                for out in cell.get("outputs", []) or []:
                    if isinstance(out, dict):
                        rendered_out = _render_notebook_output(out)
                        if rendered_out:
                            outputs_html.append(rendered_out)

                outputs_block = (
                    f'<div class="nb-outputs">{"".join(outputs_html)}</div>'
                    if outputs_html
                    else ""
                )
                rendered.append(
                    f'<div class="nb-cell nb-cell-code">'
                    f'<div class="nb-code-row">'
                    f'{count_badge}'
                    f'<pre class="nb-code"><code class="language-{_esc(lang_class)}">'
                    f'{src_html}</code></pre>'
                    f'</div>'
                    f'{outputs_block}'
                    f'</div>'
                )
                text_parts.append(src)
                word_count += len(src.split())

            elif ctype == "raw":
                rendered.append(
                    f'<div class="nb-cell nb-cell-raw"><pre>{_esc(src)}</pre></div>'
                )
                text_parts.append(src)
                word_count += len(src.split())

        nb_html = (
            f'<div class="browse-notebook-embed">'
            f'<div class="browse-code-header">{_esc(filename)} '
            f'<span class="nb-kernel">({_esc(kernel_lang)})</span></div>'
            f'<div class="nb-body">{"".join(rendered)}</div>'
            f'<div class="browse-pdf-actions">'
            f'<a href="{proxy_url}" download="{_esc(filename)}">Download Notebook</a>'
            f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">Open Original</a>'
            f'</div></div>'
        )

        full_text = "\n\n".join(text_parts)
        return {
            **base,
            "html": nb_html,
            "text": full_text[:50_000],
            "title": clean_title,
            "word_count": word_count,
            "reading_time_min": max(1, word_count // 238),
            "source": "notebook",
        }

    if category == "markdown":
        log.info("browse_file_markdown", url=url)
        content = raw_html or ""
        # Simple markdown → HTML rendering
        import re as _re
        md_html = _esc(content)
        md_html = _re.sub(r'^### (.+)$', r'<h3>\1</h3>', md_html, flags=_re.MULTILINE)
        md_html = _re.sub(r'^## (.+)$', r'<h2>\1</h2>', md_html, flags=_re.MULTILINE)
        md_html = _re.sub(r'^# (.+)$', r'<h1>\1</h1>', md_html, flags=_re.MULTILINE)
        md_html = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', md_html)
        md_html = _re.sub(r'\*(.+?)\*', r'<em>\1</em>', md_html)
        md_html = _re.sub(r'`(.+?)`', r'<code>\1</code>', md_html)
        md_html = _re.sub(r'^- (.+)$', r'<li>\1</li>', md_html, flags=_re.MULTILINE)
        md_html = md_html.replace('\n\n', '</p><p>').replace('\n', '<br>')
        md_html = f'<p>{md_html}</p>'
        return {
            **base,
            "html": md_html,
            "text": content,
            "title": clean_title,
            "word_count": len(content.split()),
            "reading_time_min": max(1, len(content.split()) // 238),
            "source": "markdown",
        }

    if category == "video":
        log.info("browse_file_video", url=url)
        return {
            **base,
            "html": (
                f'<div class="browse-video-embed">'
                f'<video controls preload="metadata" class="browse-video-player">'
                f'<source src="{proxy_url}" type="{_FILE_TYPES[ext][1]}">'
                f'Your browser does not support video playback.'
                f'</video>'
                f'<div class="browse-pdf-actions">'
                f'<a href="{proxy_url}" download="{_esc(filename)}">Download Video</a>'
                f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">Open Original</a>'
                f'</div></div>'
            ),
            "text": f"Video: {filename}",
            "title": clean_title,
            "word_count": 0, "reading_time_min": 0,
            "source": "video",
        }

    if category == "audio":
        log.info("browse_file_audio", url=url)
        return {
            **base,
            "html": (
                f'<div class="browse-audio-embed">'
                f'<audio controls preload="metadata" class="browse-audio-player">'
                f'<source src="{proxy_url}" type="{_FILE_TYPES[ext][1]}">'
                f'Your browser does not support audio playback.'
                f'</audio>'
                f'<div class="browse-pdf-actions">'
                f'<a href="{proxy_url}" download="{_esc(filename)}">Download Audio</a>'
                f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">Open Original</a>'
                f'</div></div>'
            ),
            "text": f"Audio: {filename}",
            "title": clean_title,
            "word_count": 0, "reading_time_min": 0,
            "source": "audio",
        }

    if category == "download":
        log.info("browse_file_download", url=url, ext=ext)
        type_name = {
            ".docx": "Word Document", ".xlsx": "Excel Spreadsheet",
            ".pptx": "PowerPoint Presentation", ".doc": "Word Document",
            ".xls": "Excel Spreadsheet", ".ppt": "PowerPoint Presentation",
            ".odt": "OpenDocument Text", ".ods": "OpenDocument Spreadsheet",
            ".epub": "EPUB Book",
            ".zip": "ZIP Archive", ".tar": "TAR Archive",
            ".gz": "Gzip Archive", ".7z": "7-Zip Archive", ".rar": "RAR Archive",
        }.get(ext, "Document")
        return {
            **base,
            "html": (
                f'<div class="browse-download-embed">'
                f'<div class="browse-download-icon">📄</div>'
                f'<div class="browse-download-info">'
                f'<div class="browse-download-name">{_esc(filename)}</div>'
                f'<div class="browse-download-type">{type_name}</div>'
                f'</div>'
                f'<div class="browse-pdf-actions">'
                f'<a href="{proxy_url}" download="{_esc(filename)}">Download {type_name}</a>'
                f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">Open Original</a>'
                f'</div></div>'
            ),
            "text": f"{type_name}: {filename}",
            "title": clean_title,
            "word_count": 0, "reading_time_min": 0,
            "source": "download",
        }

    return None


# Wikipedia article title regex — matches /wiki/Article_Title
_WIKI_ARTICLE_RE = re.compile(
    r"(?:https?://)?([a-z]{2,3})\.(?:wikipedia|wikimedia|wiktionary|wikiquote|wikibooks|wikisource|wikinews|wikiversity|wikivoyage)\.org/wiki/([^#?]+)",
    re.IGNORECASE,
)


async def _try_wikipedia_api(url: str, request) -> dict | None:
    """Fetch Wikipedia articles via the REST API for clean, reliable HTML.

    The Wikimedia REST API (api.wikimedia.org) returns pre-rendered HTML
    that's cleaner than scraping. Works for Wikipedia, Wiktionary,
    Wikiquote, Wikibooks, and other Wikimedia projects.
    """
    m = _WIKI_ARTICLE_RE.match(url)
    if not m:
        return None

    lang = m.group(1)
    raw_title = m.group(2)

    # Skip special pages and file/media pages — let normal fetch handle them
    if raw_title.startswith(("Special:", "File:", "Category:", "Talk:",
                             "User:", "Template:", "Help:", "Portal:",
                             "Wikipedia:", "MediaWiki:")):
        return None

    # Also skip commons.wikimedia.org entirely — it's a media repository, not articles
    if "commons.wikimedia.org" in url:
        return None

    # Decode percent-encoded title
    from urllib.parse import unquote
    title = unquote(raw_title).replace("_", " ")
    api_title = raw_title  # keep URL-encoded form for API

    http_client = getattr(request.app.state, "http_client", None)
    if not http_client:
        return None

    # Determine which Wikimedia project
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    project = "wikipedia"
    for p in ("wiktionary", "wikiquote", "wikibooks", "wikisource", "wikinews", "wikivoyage", "wikiversity"):
        if p in hostname:
            project = p
            break

    # Try the Wikipedia REST API (returns clean HTML)
    api_url = f"https://{lang}.{project}.org/api/rest_v1/page/html/{api_title}"
    try:
        resp = await http_client.get(
            api_url,
            timeout=15.0,
            headers={"User-Agent": "Augmentum/1.0 (self-hosted AI proxy)"},
        )
        if resp.status_code != 200:
            return None  # fall through to normal fetch
        article_html = resp.text
    except Exception:
        return None

    # Also get summary for metadata
    summary_title = ""
    summary_desc = ""
    summary_thumb = ""
    try:
        summary_resp = await http_client.get(
            f"https://{lang}.{project}.org/api/rest_v1/page/summary/{api_title}",
            timeout=8.0,
            headers={"User-Agent": "Augmentum/1.0 (self-hosted AI proxy)"},
        )
        if summary_resp.status_code == 200:
            summary = summary_resp.json()
            summary_title = summary.get("titles", {}).get("display", title)
            summary_desc = summary.get("extract", "")
            summary_thumb = summary.get("thumbnail", {}).get("source", "")
    except Exception:
        # Wikipedia REST summary is enrichment — main article HTML is
        # already rendered above; missing summary just means no card.
        log.debug("wikipedia_summary_fetch_failed", title=title, exc_info=True)

    # Clean the REST API HTML — it's already good but needs image proxy + style strip
    # The API returns full HTML with inline styles for math, tables, etc.
    # We rewrite image URLs through our proxy and strip external stylesheets
    clean_html = article_html

    # Remove REST API base styles link (we have our own)
    clean_html = re.sub(r'<link\s+rel="stylesheet"[^>]*>', '', clean_html)

    # Rewrite image URLs through our proxy
    def _proxy_wiki_img(match):
        src = match.group(1) or match.group(2)
        if src and not src.startswith("/api/browse/"):
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                src = f"https://{lang}.{project}.org{src}"
            return match.group(0).replace(
                match.group(1) or match.group(2),
                f"/api/browse/image?url={quote_plus(src)}",
            )
        return match.group(0)

    clean_html = re.sub(
        r'src="([^"]+)"|src=\'([^\']+)\'',
        _proxy_wiki_img,
        clean_html,
    )

    # Extract plain text for the text view (strip HTML tags)
    text = re.sub(r'<[^>]+>', ' ', clean_html)
    text = re.sub(r'\s+', ' ', text).strip()[:20000]

    display_title = summary_title or title
    site_name = f"{project.title()} ({lang.upper()})"

    return {
        "html": clean_html,
        "text": text,
        "title": display_title,
        "author": "",
        "date": "",
        "sitename": site_name,
        "word_count": len(text.split()),
        "reading_time_min": max(1, len(text.split()) // 238),
        "url": url,
        "favicon_url": f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3D{hostname}%26sz%3D32",
        "source": "wikipedia-api",
    }


# ---------------------------------------------------------------------------
# Reddit intercept
#
# Reddit blocks most server-side scraping with a "prove you're human" wall,
# so trafilatura fetches of reddit.com return nothing useful. Their .json
# endpoint — which used to be the escape hatch — now 403s on cloud / data-
# centre IPs just like the HTML. The one path still open anonymously is
# RSS: they need feed readers to work, so old.reddit.com/r/.../.rss and
# old.reddit.com/r/{sub}/comments/{id}/.rss still return a full Atom feed
# containing the post plus every comment as individual entries.
#
# RSS is lossier than JSON — no scores, no threaded replies, no num_comments
# summary, no preview images structured — but it's enough to render the
# post body and a flat comment list, which is the 90% use case.
#
# User-Agent: Reddit still rejects generic UAs. A descriptive one works.
# ---------------------------------------------------------------------------

_REDDIT_HOSTS = {
    "reddit.com", "www.reddit.com", "old.reddit.com",
    "new.reddit.com", "np.reddit.com", "i.reddit.com", "m.reddit.com",
}

# /r/<sub>/comments/<id>[/slug[/comment_id]]
_REDDIT_POST_RE = re.compile(
    r"^/r/([a-zA-Z0-9_]+)/comments/([a-zA-Z0-9]+)(?:/[^/?#]*)?(?:/([a-zA-Z0-9]+))?/?$"
)

# /r/<sub>[/sort]
_REDDIT_SUB_RE = re.compile(
    r"^/r/([a-zA-Z0-9_]+)(?:/(hot|new|top|rising|controversial|best))?/?$"
)

# Reddit gates their RSS / JSON endpoints by User-Agent as well as IP.
# As of April 2026 an identifying "Augmentum/1.0 ..." UA gets a 403 on
# both www and old subdomains even for the .rss feed that's meant for
# feed readers — Reddit's anti-bot heuristics treat any non-browser UA
# as suspect. A standard browser UA string passes. This is the same
# tradeoff Reddit RSS readers make; we keep it explicit here so a future
# reviewer sees the reasoning and doesn't "fix" it back to a polite UA.
_REDDIT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _classify_reddit_url(url: str) -> tuple[str, str, str | None] | None:
    """Return (kind, subreddit, post_id_or_None) for a Reddit URL, or None.

    kind is 'post' for an individual post / comment permalink and 'listing'
    for a subreddit homepage or sort page. Other Reddit URLs (user pages,
    search, wiki) return None and fall through to the normal fetch path —
    not worth building dedicated renderers for every Reddit surface.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if (parsed.hostname or "").lower() not in _REDDIT_HOSTS:
        return None
    path = parsed.path or "/"
    m = _REDDIT_POST_RE.match(path)
    if m:
        return ("post", m.group(1), m.group(2))
    m = _REDDIT_SUB_RE.match(path)
    if m:
        return ("listing", m.group(1), None)
    return None


def _reddit_age(created_utc: float | int | None) -> str:
    """Relative age label: '3 hours ago', '2 days ago', '1 year ago'."""
    if not created_utc:
        return ""
    import time
    delta = time.time() - float(created_utc)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)} min ago"
    if delta < 86400:
        return f"{int(delta / 3600)} hr ago"
    if delta < 30 * 86400:
        return f"{int(delta / 86400)} days ago"
    if delta < 365 * 86400:
        return f"{int(delta / (30 * 86400))} months ago"
    return f"{int(delta / (365 * 86400))} yr ago"


# Any src/href pointing to these hosts gets rewritten through our
# /api/browse/image proxy. The CSP only allows images from 'self' + a
# small curated list, and Reddit has many CDN hosts (i.redd.it,
# preview.redd.it, external-preview.redd.it, v.redd.it, the older
# *.redditmedia.com domains, imgur when rehost-embedded, etc.) that
# would all need allowlisting otherwise. Proxying covers them in one
# sweep and inherits the proxy's SSRF validation + size cap + fallback
# to a transparent PNG on failure. `external-` covered by the wildcard
# check below — we match the substring, not the full hostname.
def _proxy_external_media(html_fragment: str, *, base_url: str = "") -> str:
    """Rewrite img/source/poster src attributes in rendered HTML so they
    flow through /api/browse/image. Used on Reddit, GitHub README, and
    any other intercept whose output may reference CDN hosts the CSP
    doesn't cover (Reddit's ten-odd media CDNs, GitHub's camo /
    user-images / shields.io badges, etc.). The proxy endpoint handles
    TLS + SSRF checks + a transparent-PNG fallback, so universal
    proxying is safe and costs nothing per-domain.

    Idempotent: srcs already pointing at /api/browse/ are skipped. When
    a base_url is supplied, relative srcs are resolved against it
    before being routed through the proxy — otherwise relative srcs
    are left alone (we can't rewrite what we can't resolve)."""
    if not html_fragment or "<" not in html_fragment:
        return html_fragment

    def _rewrite(match: re.Match) -> str:
        full = match.group(0)
        src = match.group(1) or match.group(2) or ""
        if not src:
            return full
        if src.startswith(("/api/browse/", "data:", "blob:")):
            return full
        # Absolute or protocol-relative: proxy directly.
        if src.startswith("//"):
            abs_src = "https:" + src
        elif src.startswith(("http://", "https://")):
            abs_src = src
        elif base_url:
            abs_src = urljoin(base_url, src)
        else:
            return full  # relative w/ no base — can't resolve, leave alone
        proxied = f"/api/browse/image?url={quote_plus(abs_src)}"
        return full.replace(src, proxied)

    # Match src="..." or src='...' on img/source + poster="..." on
    # video. Narrow to those attrs so we don't touch href.
    return re.sub(
        r'\b(?:src|poster)=(?:"([^"]+)"|\'([^\']+)\')',
        _rewrite,
        html_fragment,
    )


# Backwards-compat alias (one caller still uses the old name inside
# _reddit_decode_html further down). Kept as a thin forwarder rather
# than a search-and-replace so the semantics stay obvious at the site
# that's explicitly about Reddit.
def _proxy_reddit_media(html_fragment: str) -> str:
    return _proxy_external_media(html_fragment)


def _reddit_decode_html(raw_html: str) -> str:
    """Reddit returns body_html as HTML-entity-escaped HTML (i.e. the
    server-side sanitised version run through html.escape). Unescape
    once to get real HTML tags, then rewrite media srcs through our
    proxy so CSP doesn't block Reddit's image CDNs. The frontend's
    DOMPurify pass still runs over the final output."""
    if not raw_html:
        return ""
    import html as _html
    decoded = _html.unescape(raw_html)
    return _proxy_reddit_media(decoded)


def _esc(s: object) -> str:
    """Local HTML-escape wrapper. Uses the stdlib escape so we don't pull
    the render module's escapeHtml here."""
    import html as _html
    return _html.escape(str(s or ""), quote=True)


_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _parse_reddit_atom(xml_text: str) -> tuple[str, list[dict]]:
    """Parse a Reddit Atom feed into (feed_title, entries).

    Each entry dict carries: author, title, content_html (HTML-escaped
    HTML per the Atom spec), updated (ISO8601), kind ('post'|'comment'),
    id, link, subreddit. For per-post feeds the first entry is the post
    and the rest are comments (flat — Atom doesn't carry threading).
    """
    from xml.etree import ElementTree as ET  # stdlib

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ("", [])

    feed_title_el = root.find(f"{_ATOM_NS}title")
    feed_title = (feed_title_el.text or "").strip() if feed_title_el is not None else ""

    entries: list[dict] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        author_el = entry.find(f"{_ATOM_NS}author/{_ATOM_NS}name")
        author = (author_el.text or "").strip() if author_el is not None else ""
        if author.startswith("/u/"):
            author = author[3:]

        title_el = entry.find(f"{_ATOM_NS}title")
        title = (title_el.text or "").strip() if title_el is not None else ""

        content_el = entry.find(f"{_ATOM_NS}content")
        content = (content_el.text or "") if content_el is not None else ""

        updated_el = entry.find(f"{_ATOM_NS}updated")
        updated = (updated_el.text or "").strip() if updated_el is not None else ""

        id_el = entry.find(f"{_ATOM_NS}id")
        entry_id = (id_el.text or "").strip() if id_el is not None else ""
        # Reddit tags: t3_* = post, t1_* = comment
        if entry_id.startswith("t3_"):
            kind = "post"
        elif entry_id.startswith("t1_"):
            kind = "comment"
        else:
            kind = ""

        link_el = entry.find(f"{_ATOM_NS}link")
        link = link_el.attrib.get("href", "") if link_el is not None else ""

        category_el = entry.find(f"{_ATOM_NS}category")
        subreddit_term = ""
        if category_el is not None:
            subreddit_term = (
                category_el.attrib.get("label", "")
                or f"r/{category_el.attrib.get('term', '')}"
            )

        entries.append({
            "author": author or "[deleted]",
            "title": title,
            "content_html": content,
            "updated": updated,
            "kind": kind,
            "id": entry_id,
            "link": link,
            "subreddit": subreddit_term,
        })
    return (feed_title, entries)


def _reddit_iso_to_age(iso: str) -> str:
    """Convert ISO8601 to a relative age label. Returns '' on parse fail."""
    if not iso:
        return ""
    try:
        from datetime import datetime, timezone
        import time as _time
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (datetime.now(timezone.utc) - dt).total_seconds()
        if delta < 0:
            delta = 0
        return _reddit_age(_time.time() - delta)
    except Exception:
        return ""


def _render_reddit_post_rss(
    feed_title: str, entries: list[dict]
) -> tuple[str, str, str, dict]:
    """Render a per-post Atom feed (first entry = post, rest = comments).

    RSS doesn't carry threading info — all comments arrive flat, ordered
    roughly by Reddit's default sort. We render them as a flat list with
    a permalink out to the comment on Reddit for anyone wanting to see
    context.
    """
    if not entries:
        return ("", "", "Reddit", {})

    post = entries[0]
    comments = [e for e in entries[1:] if e.get("kind") == "comment"]

    title = post.get("title") or feed_title or "Reddit post"
    subreddit = post.get("subreddit") or ""
    author = post.get("author") or "[deleted]"
    age = _reddit_iso_to_age(post.get("updated", ""))
    body_html = _reddit_decode_html(post.get("content_html") or "")

    header_html = (
        f'<div class="reddit-post-header">'
        f'<div class="reddit-sub-line">'
        f'<span class="reddit-sub">{_esc(subreddit)}</span>'
        f'<span class="reddit-author">Posted by u/{_esc(author)}</span>'
        f'<span class="reddit-age">{_esc(age)}</span>'
        f'</div>'
        f'<div class="reddit-post-meta">'
        f'<span class="reddit-comments">{len(comments)} comments</span>'
        f'</div>'
        f'</div>'
    )

    comment_parts: list[str] = []
    for c in comments[:200]:  # DOM cap for megathreads
        cauthor = c.get("author") or "[deleted]"
        cage = _reddit_iso_to_age(c.get("updated", ""))
        cbody = _reddit_decode_html(c.get("content_html") or "")
        clink = c.get("link") or ""
        if not cbody.strip():
            continue
        if clink:
            link_btn = (
                f'<a class="reddit-comment-permalink" href="{_esc(clink)}" '
                f'target="_blank" rel="noopener noreferrer" title="Open on Reddit">'
                f'&#x2197;</a>'
            )
        else:
            link_btn = ""
        comment_parts.append(
            f'<div class="reddit-comment" data-depth="0">'
            f'<div class="reddit-comment-meta">'
            f'<strong class="reddit-author">u/{_esc(cauthor)}</strong>'
            f'<span class="reddit-age">{_esc(cage)}</span>'
            f'{link_btn}'
            f'</div>'
            f'<div class="reddit-comment-body">{cbody}</div>'
            f'</div>'
        )
    comments_html = "".join(comment_parts)
    if not comments_html:
        comments_html = '<p class="reddit-no-comments">No comments yet.</p>'

    body_block = (
        f'<div class="reddit-post-body">{body_html}</div>'
        if body_html else ""
    )
    article_html = (
        f'<div class="reddit-article">'
        f'{header_html}'
        f'{body_block}'
        f'<hr class="reddit-divider">'
        f'<h2 class="reddit-comments-heading">Comments ({len(comments)})</h2>'
        f'<div class="reddit-comments">{comments_html}</div>'
        f'</div>'
    )

    metadata = {
        "author": author,
        "date": age,
        "num_comments": len(comments),
        "subreddit": subreddit,
    }
    return (article_html, title, "Reddit", metadata)


def _render_reddit_listing_rss(
    subreddit: str, entries: list[dict]
) -> tuple[str, str, str, dict]:
    """Render a subreddit Atom feed as post cards. Each card links through
    /api/browse/fetch so clicking re-fires the intercept with the specific
    post URL."""
    cards: list[str] = []
    for e in entries[:40]:
        if e.get("kind") != "post":
            continue
        e_title = e.get("title") or ""
        e_author = e.get("author") or ""
        e_age = _reddit_iso_to_age(e.get("updated", ""))
        e_link = e.get("link") or ""
        if not e_link:
            continue
        fetch_link = f"/api/browse/fetch?url={quote_plus(e_link)}"
        preview_src = _reddit_decode_html(e.get("content_html") or "")
        preview_txt = re.sub(r"<[^>]+>", " ", preview_src)
        preview_txt = re.sub(r"\s+", " ", preview_txt).strip()[:220]
        if preview_txt:
            ellipsis = "\u2026" if len(preview_txt) == 220 else ""
            preview_html = (
                f'<p class="reddit-listing-preview">'
                f'{_esc(preview_txt)}{ellipsis}</p>'
            )
        else:
            preview_html = ""
        cards.append(
            f'<div class="reddit-listing-card">'
            f'<h3 class="reddit-listing-title">'
            f'<a href="{_esc(fetch_link)}">{_esc(e_title)}</a>'
            f'</h3>'
            f'<div class="reddit-listing-meta">'
            f'<span class="reddit-author">u/{_esc(e_author)}</span>'
            f'<span class="reddit-age">{_esc(e_age)}</span>'
            f'</div>'
            f'{preview_html}'
            f'</div>'
        )
    if not cards:
        cards.append('<p class="reddit-empty">No posts found.</p>')

    title = f"r/{subreddit}"
    article_html = (
        f'<div class="reddit-listing">'
        f'<h1 class="reddit-listing-heading">{_esc(title)}</h1>'
        f'{"".join(cards)}'
        f'</div>'
    )
    metadata = {"subreddit": subreddit, "post_count": len(entries)}
    return (article_html, title, "Reddit", metadata)


async def _try_reddit_api(url: str, request) -> dict | None:
    """Intercept Reddit URLs via the RSS (Atom) endpoint.

    Reddit's .json path now 403s on cloud / datacentre IPs as of spring
    2026. Their Atom feeds still work anonymously because feed readers
    depend on them — and they include the full post body plus every
    comment as flat entries, which is enough for a readable article view.

    Falls through (returns None) for URL shapes we don't render — user
    pages, search, wiki — and for any fetch failure, so the caller drops
    down to trafilatura.
    """
    classified = _classify_reddit_url(url)
    if not classified:
        return None
    kind, subreddit, post_id = classified

    parsed = urlparse(url)
    if kind == "post" and post_id:
        rss_url = f"https://old.reddit.com/r/{subreddit}/comments/{post_id}/.rss"
    else:
        # Preserve a trailing sort (hot/new/top/rising/controversial/best)
        # if the user pasted one — Atom feeds exist for each.
        sort_match = re.match(
            r"^/r/[a-zA-Z0-9_]+/(hot|new|top|rising|controversial|best)",
            parsed.path,
        )
        sort = sort_match.group(1) if sort_match else ""
        if sort:
            rss_url = f"https://old.reddit.com/r/{subreddit}/{sort}/.rss"
        else:
            rss_url = f"https://old.reddit.com/r/{subreddit}/.rss"

    # Only User-Agent — sending an explicit Accept header (even a
    # browser-standard one) fingerprints us as a bot against Reddit's
    # anti-scraping heuristics and swaps the 200 for a 403 HTML wall.
    # Httpx's default Accept ("*/*") doesn't trigger the check.
    headers = {"User-Agent": _REDDIT_UA}

    # Reddit sets a `session_tracker` cookie on every response; on any
    # subsequent request that replays it, Reddit challenges us with a
    # 403. The shared app.state.http_client shares a cookie jar across
    # every request, so using it would block the very next Reddit fetch
    # (and sometimes the same one that set the cookie, if we'd already
    # fetched Reddit earlier in the session). Using a fresh short-lived
    # client keeps the cookie jar empty and each call clean — at the
    # cost of no connection pooling, which matters essentially zero for
    # Reddit's per-request pace.
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            resp = await client.get(rss_url)
    except Exception:
        return None
    if resp.status_code != 200:
        log.info("reddit_rss_non_200", url=url, rss_url=rss_url, status=resp.status_code)
        return None
    ctype = resp.headers.get("content-type", "") or ""
    if "xml" not in ctype and "atom" not in ctype:
        # Reddit occasionally returns a 200 HTML error page during
        # anti-bot rollouts; don't try to parse HTML as Atom.
        log.info("reddit_rss_wrong_ctype", url=url, rss_url=rss_url, ctype=ctype)
        return None

    feed_title, entries = _parse_reddit_atom(resp.text)
    if not entries:
        return None

    if kind == "post":
        article_html, title, site, meta = _render_reddit_post_rss(feed_title, entries)
    else:
        article_html, title, site, meta = _render_reddit_listing_rss(subreddit, entries)

    if not article_html:
        return None

    text = re.sub(r"<[^>]+>", " ", article_html)
    text = re.sub(r"\s+", " ", text).strip()[:20000]

    hostname = parsed.hostname or "reddit.com"
    return {
        "html": article_html,
        "text": text,
        "title": title,
        "author": meta.get("author", ""),
        "date": meta.get("date", ""),
        "sitename": site,
        "word_count": len(text.split()),
        "reading_time_min": max(1, len(text.split()) // 238),
        "url": url,
        "favicon_url": f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3D{hostname}%26sz%3D32",
        "source": "reddit-rss",
        "page_type": "forum",
    }


# ---------------------------------------------------------------------------
# Hacker News intercept (Firebase API, no auth, no meaningful rate limit)
# ---------------------------------------------------------------------------
_HN_HOSTS = frozenset({"news.ycombinator.com", "ycombinator.com"})
_HN_ITEM_RE = re.compile(r"^/item")


def _classify_hn_url(url: str) -> str | None:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if (parsed.hostname or "").lower() not in _HN_HOSTS:
        return None
    if not _HN_ITEM_RE.match(parsed.path or ""):
        return None
    try:
        from urllib.parse import parse_qs
        item_id = (parse_qs(parsed.query or "").get("id") or [""])[0]
    except Exception:
        return None
    return item_id if item_id.isdigit() else None


async def _hn_fetch(client, item_id: str) -> dict | None:
    try:
        resp = await client.get(
            f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json",
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _hn_epoch_to_age(ts: int | float | None) -> str:
    if not ts:
        return ""
    import time as _time
    delta = _time.time() - float(ts)
    if delta < 0:
        delta = 0
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)} min ago"
    if delta < 86400:
        return f"{int(delta / 3600)} hr ago"
    if delta < 30 * 86400:
        return f"{int(delta / 86400)} days ago"
    if delta < 365 * 86400:
        return f"{int(delta / (30 * 86400))} months ago"
    return f"{int(delta / (365 * 86400))} yr ago"


async def _hn_render_comment(client, comment_id: int, depth: int = 0, max_depth: int = 4) -> str:
    """Recursively render an HN comment subtree. Per-node fanout and
    total depth are capped — a 1000-child thread would otherwise fire
    a storm of Firebase requests and freeze the render. Extras are
    replaced with a 'more replies on HN' link."""
    if depth > max_depth:
        return ""
    data = await _hn_fetch(client, str(comment_id))
    if not data or data.get("deleted") or data.get("dead"):
        return ""
    author = data.get("by") or "[deleted]"
    age = _hn_epoch_to_age(data.get("time"))
    body_html = data.get("text") or ""
    if not body_html:
        return ""
    replies = data.get("kids") or []
    replies_html = ""
    if depth < max_depth:
        per_node_cap = 20
        child_htmls = await asyncio.gather(
            *[_hn_render_comment(client, kid, depth + 1, max_depth)
              for kid in replies[:per_node_cap]],
            return_exceptions=True,
        )
        replies_html = "".join(h for h in child_htmls if isinstance(h, str))
        if len(replies) > per_node_cap:
            replies_html += (
                f'<div class="hn-more-replies">'
                f'+{len(replies) - per_node_cap} more replies on HN</div>'
            )
    indent_style = f"margin-left:{min(depth * 16, 64)}px" if depth else ""
    return (
        f'<div class="hn-comment" data-depth="{depth}" style="{indent_style}">'
        f'<div class="hn-comment-meta">'
        f'<strong class="hn-author">{_esc(author)}</strong>'
        f'<span class="hn-age">{_esc(age)}</span>'
        f'</div>'
        f'<div class="hn-comment-body">{body_html}</div>'
        f'{replies_html}'
        f'</div>'
    )


async def _try_hn_api(url: str, request) -> dict | None:
    item_id = _classify_hn_url(url)
    if not item_id:
        return None

    async with httpx.AsyncClient(
        timeout=15.0,
        headers={"User-Agent": "Augmentum/1.0 (+reader)"},
    ) as client:
        root = await _hn_fetch(client, item_id)
        if not root:
            return None

        title = root.get("title") or "Hacker News"
        author = root.get("by") or "[deleted]"
        score = root.get("score") or 0
        age = _hn_epoch_to_age(root.get("time"))
        story_url = root.get("url") or ""
        story_text_html = root.get("text") or ""
        descendants = root.get("descendants") or 0
        top_kids = root.get("kids") or []

        if story_url and not story_text_html:
            body_html = (
                f'<div class="hn-link-card">'
                f'<a class="hn-link" href="{_esc(story_url)}" target="_blank" rel="noopener noreferrer">'
                f'{_esc(story_url)}</a>'
                f'</div>'
            )
        elif story_text_html:
            body_html = f'<div class="hn-text-body">{story_text_html}</div>'
        else:
            body_html = ""

        comment_cap = 40
        comment_htmls = await asyncio.gather(
            *[_hn_render_comment(client, kid, 0) for kid in top_kids[:comment_cap]],
            return_exceptions=True,
        )
        comments_html = "".join(h for h in comment_htmls if isinstance(h, str))
        if len(top_kids) > comment_cap:
            comments_html += (
                f'<div class="hn-more-replies">'
                f'+{len(top_kids) - comment_cap} more top-level comments on HN</div>'
            )
        if not comments_html:
            comments_html = '<p class="hn-no-comments">No comments yet.</p>'

        header_html = (
            f'<div class="hn-post-header">'
            f'<div class="hn-sub-line">'
            f'<span class="hn-author">Posted by {_esc(author)}</span>'
            f'<span class="hn-age">{_esc(age)}</span>'
            f'</div>'
            f'<div class="hn-post-meta">'
            f'<span class="hn-score">{score} points</span>'
            f'<span class="hn-comments">{descendants} comments</span>'
            f'</div>'
            f'</div>'
        )
        article_html = (
            f'<div class="hn-article">'
            f'{header_html}'
            f'{body_html}'
            f'<hr class="hn-divider">'
            f'<h2 class="hn-comments-heading">Comments ({descendants})</h2>'
            f'<div class="hn-comments">{comments_html}</div>'
            f'</div>'
        )
        text = re.sub(r"<[^>]+>", " ", article_html)
        text = re.sub(r"\s+", " ", text).strip()[:20000]
        return {
            "html": article_html,
            "text": text,
            "title": title,
            "author": author,
            "date": age,
            "sitename": "Hacker News",
            "word_count": len(text.split()),
            "reading_time_min": max(1, len(text.split()) // 238),
            "url": url,
            "favicon_url": "/api/browse/image?url=https%3A%2F%2Fnews.ycombinator.com%2Ffavicon.ico",
            "source": "hn-api",
            "page_type": "forum",
        }


# ---------------------------------------------------------------------------
# GitHub intercept (REST API; 60 req/hr/IP anonymous is plenty for a
# reader that fetches a README once per clicked repo).
# ---------------------------------------------------------------------------
_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})
_GITHUB_REPO_RE = re.compile(r"^/([\w.-]+)/([\w.-]+)/?$")
# /<owner>/<repo>/blob/<branch>/<path...>
_GITHUB_BLOB_RE = re.compile(r"^/([\w.-]+)/([\w.-]+)/blob/([^/]+)/(.+)$")
_GITHUB_RESERVED = frozenset({
    "settings", "explore", "topics", "search", "features", "pricing",
    "orgs", "users", "about", "marketplace", "apps", "new",
    "notifications", "login", "signup", "pulls", "issues", "codespaces",
    "sponsors", "enterprise", "trending", "collections",
})

# File extension → hljs language. hljs accepts the raw extension as
# language too (js, py, rb, etc.) but some of these normalise to the
# canonical name hljs expects. Unknown extensions render unhighlighted
# which is fine — still monospaced, still readable.
_GITHUB_EXT_TO_LANG = {
    "py": "python", "pyi": "python", "pyx": "python",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "tsx": "typescript", "jsx": "jsx",
    "rb": "ruby", "rs": "rust", "go": "go",
    "java": "java", "kt": "kotlin", "swift": "swift",
    "c": "c", "h": "c", "cpp": "cpp", "cc": "cpp", "cxx": "cpp",
    "hpp": "cpp", "hh": "cpp", "cs": "csharp",
    "php": "php", "pl": "perl", "lua": "lua", "sh": "bash",
    "bash": "bash", "zsh": "bash", "fish": "bash",
    "sql": "sql", "r": "r", "scala": "scala",
    "yaml": "yaml", "yml": "yaml", "toml": "ini", "ini": "ini",
    "json": "json", "jsonc": "json", "json5": "json",
    "xml": "xml", "html": "html", "htm": "html",
    "css": "css", "scss": "scss", "sass": "scss", "less": "less",
    "md": "markdown", "markdown": "markdown", "rst": "", "txt": "",
    "dockerfile": "dockerfile",
    "mk": "makefile", "make": "makefile", "makefile": "makefile",
    "vue": "xml", "svelte": "xml",
    "elm": "elm", "hs": "haskell", "ml": "ocaml",
    "dart": "dart", "ex": "elixir", "exs": "elixir",
    "erl": "erlang", "clj": "clojure", "cljs": "clojure",
    "nix": "nix", "tf": "hcl", "hcl": "hcl",
    "proto": "protobuf", "graphql": "graphql", "gql": "graphql",
}


def _github_lang_for_path(path: str) -> str:
    """Extract hljs language from a file path. Special-cases the
    well-known no-extension files."""
    lower = path.lower()
    # Filenames that encode their own language without an extension
    name = lower.rsplit("/", 1)[-1]
    if name in ("dockerfile", "containerfile"):
        return "dockerfile"
    if name in ("makefile", "gnumakefile"):
        return "makefile"
    if name in ("readme", "license", "changelog", "authors"):
        return ""  # plain text
    if "." not in name:
        return ""
    ext = name.rsplit(".", 1)[-1]
    return _GITHUB_EXT_TO_LANG.get(ext, ext)


def _classify_github_blob_url(url: str) -> tuple[str, str, str, str] | None:
    """Return (owner, repo, branch, path) for /blob/ URLs, else None."""
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if (parsed.hostname or "").lower() not in _GITHUB_HOSTS:
        return None
    m = _GITHUB_BLOB_RE.match(parsed.path or "")
    if not m:
        return None
    return (m.group(1), m.group(2), m.group(3), m.group(4))


async def _try_github_blob_api(url: str, request) -> dict | None:
    """Intercept github.com/<owner>/<repo>/blob/<branch>/<path> URLs.
    Fetches the raw file from raw.githubusercontent.com and wraps it in
    a syntax-highlight-ready <pre><code>. Frontend hljs does coloring."""
    classified = _classify_github_blob_url(url)
    if not classified:
        return None
    owner, repo, branch, path = classified

    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    async with httpx.AsyncClient(
        timeout=15.0,
        headers={"User-Agent": "Augmentum/1.0 (+reader)"},
    ) as client:
        try:
            resp = await client.get(raw_url)
        except Exception:
            return None
        if resp.status_code != 200:
            return None

        # Cap at 512 KB of source — larger files (LFS-tracked binaries,
        # giant data files) aren't readable anyway and we don't want to
        # ship multi-megabyte HTML over the wire.
        max_size = 512 * 1024
        content_bytes = resp.content[:max_size]
        truncated = len(resp.content) > max_size
        # Crude binary sniff: reject if NUL bytes present in the first
        # 4 KB. Prevents rendering compiled artifacts as "text".
        if b"\x00" in content_bytes[:4096]:
            return None
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = content_bytes.decode("latin-1")
            except Exception:
                return None

    lang = _github_lang_for_path(path)
    line_count = content.count("\n") + 1
    size_kb = round(len(resp.content) / 1024, 1)
    filename = path.rsplit("/", 1)[-1]

    # hljs wants the language in the class name; empty lang = auto-detect.
    # Pre-compute fragments that conditionally include escaped quotes so
    # Python 3.11's "no backslashes in f-string expressions" rule doesn't
    # bite us on inline ternaries.
    lang_class = f' class="language-{_esc(lang)}"' if lang else ""
    body = _esc(content)
    if truncated:
        truncated_note = (
            f'<div class="gh-blob-truncated">'
            f'File truncated to the first {max_size // 1024} KB '
            f'— open the raw file on GitHub for the rest.'
            f'</div>'
        )
    else:
        truncated_note = ""
    lang_chip = f'<span class="gh-blob-lang">{_esc(lang)}</span>' if lang else ""

    header_html = (
        f'<div class="gh-blob-header">'
        f'<h1 class="gh-blob-path">'
        f'<a href="https://github.com/{_esc(owner)}/{_esc(repo)}" target="_blank" rel="noopener noreferrer">{_esc(owner)}/{_esc(repo)}</a>'
        f' <span class="gh-blob-sep">/</span> '
        f'<span class="gh-blob-filename">{_esc(filename)}</span>'
        f'</h1>'
        f'<div class="gh-blob-meta">'
        f'<span class="gh-blob-branch">branch: <code>{_esc(branch)}</code></span>'
        f'<span class="gh-blob-lines">{line_count} lines</span>'
        f'<span class="gh-blob-size">{size_kb} KB</span>'
        f'{lang_chip}'
        f'<a class="gh-blob-raw" href="{_esc(raw_url)}" target="_blank" rel="noopener noreferrer">Raw</a>'
        f'</div>'
        f'<div class="gh-blob-path-sub">{_esc(path)}</div>'
        f'</div>'
    )
    article_html = (
        f'<div class="gh-blob-article">'
        f'{header_html}'
        f'{truncated_note}'
        f'<pre class="gh-blob-code"><code{lang_class}>{body}</code></pre>'
        f'</div>'
    )
    text = content[:20000]
    return {
        "html": article_html,
        "text": text,
        "title": f"{filename} · {owner}/{repo}",
        "author": owner,
        "date": "",
        "sitename": "GitHub",
        "word_count": len(text.split()),
        "reading_time_min": max(1, len(text.split()) // 238),
        "url": url,
        "favicon_url": "/api/browse/image?url=https%3A%2F%2Fgithub.com%2Ffavicon.ico",
        "source": "github-blob",
        "page_type": "reference",
    }


# ---------------------------------------------------------------------------
# GitHub Gist intercept
# ---------------------------------------------------------------------------
_GIST_HOSTS = frozenset({"gist.github.com", "www.gist.github.com"})
_GIST_ID_RE = re.compile(r"^/[\w.-]+/([a-fA-F0-9]+)/?$")


def _classify_gist_url(url: str) -> str | None:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if (parsed.hostname or "").lower() not in _GIST_HOSTS:
        return None
    m = _GIST_ID_RE.match(parsed.path or "")
    return m.group(1) if m else None


async def _try_github_gist_api(url: str, request) -> dict | None:
    """Intercept gist.github.com URLs. Renders every file in the gist
    as its own titled block with syntax highlighting."""
    gist_id = _classify_gist_url(url)
    if not gist_id:
        return None

    async with httpx.AsyncClient(
        timeout=15.0,
        headers={
            "User-Agent": "Augmentum/1.0 (+reader)",
            "Accept": "application/vnd.github+json",
        },
    ) as client:
        try:
            resp = await client.get(f"https://api.github.com/gists/{gist_id}")
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        try:
            gist = resp.json()
        except Exception:
            return None

    owner = (gist.get("owner") or {}).get("login") or "anonymous"
    description = gist.get("description") or ""
    created = gist.get("created_at") or ""
    files = gist.get("files") or {}
    if not files:
        return None

    file_blocks: list[str] = []
    total_lines = 0
    for filename, f in files.items():
        content = f.get("content") or ""
        # language hint from API (falls back to extension)
        gist_lang = (f.get("language") or "").lower()
        if not gist_lang:
            gist_lang = _github_lang_for_path(filename)
        else:
            # Map common gist.language names to hljs ids
            gist_lang = _GITHUB_EXT_TO_LANG.get(gist_lang, gist_lang)
        lang_class = f' class="language-{_esc(gist_lang)}"' if gist_lang else ""
        line_count = content.count("\n") + 1
        total_lines += line_count
        size_kb = round((f.get("size") or 0) / 1024, 1)
        lang_chip = (
            f'<span class="gist-file-lang">{_esc(gist_lang)}</span>'
            if gist_lang else ""
        )
        file_blocks.append(
            f'<div class="gist-file">'
            f'<div class="gist-file-header">'
            f'<span class="gist-filename">{_esc(filename)}</span>'
            f'<span class="gist-file-lines">{line_count} lines</span>'
            f'<span class="gist-file-size">{size_kb} KB</span>'
            f'{lang_chip}'
            f'</div>'
            f'<pre class="gist-file-code"><code{lang_class}>{_esc(content)}</code></pre>'
            f'</div>'
        )

    age = _reddit_iso_to_age(created)
    desc_block = f'<p class="gist-description">{_esc(description)}</p>' if description else ""
    age_chip = f'<span class="gist-age">Created {_esc(age)}</span>' if age else ""
    header_html = (
        f'<div class="gist-header">'
        f'<h1 class="gist-title">Gist \u00b7 {_esc(owner)}</h1>'
        f'{desc_block}'
        f'<div class="gist-meta">'
        f'<span class="gist-files-count">{len(files)} files</span>'
        f'<span class="gist-lines-count">{total_lines} lines total</span>'
        f'{age_chip}'
        f'</div>'
        f'</div>'
    )
    article_html = (
        f'<div class="gist-article">'
        f'{header_html}'
        f'<hr class="gist-divider">'
        f'{"".join(file_blocks)}'
        f'</div>'
    )
    # Plain-text view concatenates every file body — useful for the AI
    # actions (summarize, explain) that work off extracted text.
    text_parts = [f"# {fn}\n{f.get('content', '')}" for fn, f in files.items()]
    text = "\n\n".join(text_parts)[:20000]
    return {
        "html": article_html,
        "text": text,
        "title": f"Gist by {owner}",
        "author": owner,
        "date": age,
        "sitename": "GitHub Gist",
        "word_count": len(text.split()),
        "reading_time_min": max(1, len(text.split()) // 238),
        "url": url,
        "favicon_url": "/api/browse/image?url=https%3A%2F%2Fgithub.com%2Ffavicon.ico",
        "source": "github-gist",
        "page_type": "reference",
    }


def _classify_github_url(url: str) -> tuple[str, str] | None:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if (parsed.hostname or "").lower() not in _GITHUB_HOSTS:
        return None
    m = _GITHUB_REPO_RE.match(parsed.path or "")
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    if owner in _GITHUB_RESERVED:
        return None
    if repo.endswith(".git"):
        repo = repo[:-4]
    return (owner, repo)


def _rewrite_github_relative_links(
    html: str, owner: str, repo: str, branch: str,
) -> str:
    """Rewrite relative href attributes in a GitHub-rendered README so
    they point back to the canonical github.com repo rather than 404ing
    on our origin. README links like `[docs](./CONTRIBUTING.md)`,
    `[docs](docs/guide.md)`, and `[docs](/LICENSE)` all become full
    github.com/<owner>/<repo>/blob/<branch>/... URLs — which will then
    re-enter our intercept chain on click (file view for .md/.py/etc.,
    image proxy for raster assets, raw pass-through for binary).
    Anchor-only links (#section) are left alone so TOC jumps work.
    Absolute https:// links are left alone."""
    if not html:
        return html
    base = f"https://github.com/{owner}/{repo}/blob/{branch}/"
    site_root = "https://github.com/"

    def _rewrite(match: re.Match) -> str:
        full = match.group(0)
        href = match.group(1) or match.group(2) or ""
        if not href:
            return full
        # Already absolute or opaque — leave alone.
        if href.startswith((
            "http://", "https://", "mailto:", "tel:",
            "data:", "blob:", "#", "javascript:", "/api/browse/",
        )):
            return full
        if href.startswith("/"):
            resolved = urljoin(site_root, href.lstrip("/"))
            return full.replace(href, resolved)
        # Resolve relative path against the repo root on the default
        # branch. Root-relative links above target github.com itself.
        resolved = urljoin(base, href)
        return full.replace(href, resolved)

    return re.sub(
        r'\bhref=(?:"([^"]+)"|\'([^\']+)\')',
        _rewrite,
        html,
    )


async def _try_github_api(url: str, request) -> dict | None:
    classified = _classify_github_url(url)
    if not classified:
        return None
    owner, repo = classified

    async with httpx.AsyncClient(
        timeout=15.0,
        headers={
            "User-Agent": "Augmentum/1.0 (+reader)",
            "Accept": "application/vnd.github+json",
        },
    ) as client:
        meta_task = client.get(f"https://api.github.com/repos/{owner}/{repo}")
        readme_task = client.get(
            f"https://api.github.com/repos/{owner}/{repo}/readme",
            headers={"Accept": "application/vnd.github.html+json"},
        )
        try:
            meta_resp, readme_resp = await asyncio.gather(meta_task, readme_task)
        except Exception:
            return None

        if meta_resp.status_code != 200:
            return None

        try:
            meta = meta_resp.json()
        except Exception:
            return None

        full_name = meta.get("full_name") or f"{owner}/{repo}"
        description = meta.get("description") or ""
        stars = meta.get("stargazers_count") or 0
        forks = meta.get("forks_count") or 0
        language = meta.get("language") or ""
        license_info = (meta.get("license") or {}).get("spdx_id") or ""
        updated = meta.get("pushed_at") or meta.get("updated_at") or ""
        topics = meta.get("topics") or []
        homepage = meta.get("homepage") or ""
        archived = bool(meta.get("archived"))

        readme_html = ""
        default_branch = meta.get("default_branch") or "main"
        if readme_resp.status_code == 200:
            ctype = readme_resp.headers.get("content-type", "") or ""
            if "html" in ctype:
                readme_html = readme_resp.text
            else:
                try:
                    rj = readme_resp.json()
                    import base64 as _b64
                    raw = _b64.b64decode(rj.get("content", "")).decode("utf-8", "replace")
                    readme_html = f"<pre class='gh-readme-raw'>{_esc(raw)}</pre>"
                except Exception:
                    readme_html = ""

        # Route every image / badge / embedded media through our proxy
        # so CSP's img-src 'self' covers everything — GitHub's README
        # HTML references camo.githubusercontent.com (rewritten badges),
        # user-images.githubusercontent.com (user uploads), shields.io,
        # and raw.githubusercontent.com, not all of which are on the
        # CSP allowlist.
        if readme_html:
            readme_html = _proxy_external_media(
                readme_html,
                base_url=f"https://github.com/{owner}/{repo}/raw/{default_branch}/",
            )
            # Rewrite relative links so README shortcuts like
            # [docs](./CONTRIBUTING.md) point at the repo's canonical
            # github.com URL instead of 404ing under our origin.
            readme_html = _rewrite_github_relative_links(
                readme_html, owner, repo, default_branch,
            )

        age = _reddit_iso_to_age(updated)
        topic_chips = "".join(
            f'<span class="gh-topic">{_esc(t)}</span>' for t in topics[:12]
        )
        archived_badge = (
            '<span class="gh-badge gh-badge-archived">Archived</span>'
            if archived else ""
        )
        # f-string expressions can't contain backslashes, so pre-compute
        # the optional fragments.
        desc_block = f'<p class="gh-description">{_esc(description)}</p>' if description else ""
        language_chip = f'<span class="gh-language">{_esc(language)}</span>' if language else ""
        license_chip = f'<span class="gh-license">{_esc(license_info)}</span>' if license_info else ""
        updated_chip = f'<span class="gh-updated">Updated {_esc(age)}</span>' if age else ""
        topics_block = f'<div class="gh-topics">{topic_chips}</div>' if topic_chips else ""
        homepage_block = (
            f'<p class="gh-homepage"><a href="{_esc(homepage)}" target="_blank" rel="noopener noreferrer">{_esc(homepage)}</a></p>'
            if homepage else ""
        )
        header_html = (
            f'<div class="gh-header">'
            f'<div class="gh-title-row">'
            f'<h1 class="gh-full-name">{_esc(full_name)}</h1>'
            f'{archived_badge}'
            f'</div>'
            f'{desc_block}'
            f'<div class="gh-meta">'
            f'<span class="gh-stars">\u2605 {stars}</span>'
            f'<span class="gh-forks">Forks: {forks}</span>'
            f'{language_chip}{license_chip}{updated_chip}'
            f'</div>'
            f'{topics_block}{homepage_block}'
            f'</div>'
        )

        body_html = (
            f'<div class="gh-readme">{readme_html}</div>'
            if readme_html else '<p class="gh-no-readme">This repository has no README.</p>'
        )
        article_html = (
            f'<div class="gh-article">'
            f'{header_html}'
            f'<hr class="gh-divider">'
            f'{body_html}'
            f'</div>'
        )
        text = re.sub(r"<[^>]+>", " ", article_html)
        text = re.sub(r"\s+", " ", text).strip()[:20000]
        return {
            "html": article_html,
            "text": text,
            "title": full_name,
            "author": owner,
            "date": age,
            "sitename": "GitHub",
            "word_count": len(text.split()),
            "reading_time_min": max(1, len(text.split()) // 238),
            "url": url,
            "favicon_url": "/api/browse/image?url=https%3A%2F%2Fgithub.com%2Ffavicon.ico",
            "source": "github-api",
            "page_type": "reference",
        }


# ---------------------------------------------------------------------------
# Stack Exchange intercept (300 req/day/IP anonymous).
# ---------------------------------------------------------------------------
_STACK_Q_RE = re.compile(r"^/questions/(\d+)(?:/[^/?#]*)?/?$")
_STACK_HOST_SITE_MAP = {
    "stackoverflow.com": "stackoverflow",
    "www.stackoverflow.com": "stackoverflow",
    "superuser.com": "superuser",
    "www.superuser.com": "superuser",
    "serverfault.com": "serverfault",
    "www.serverfault.com": "serverfault",
    "askubuntu.com": "askubuntu",
    "www.askubuntu.com": "askubuntu",
    "mathoverflow.net": "mathoverflow",
    "www.mathoverflow.net": "mathoverflow",
    "stackapps.com": "stackapps",
    "www.stackapps.com": "stackapps",
}


def _classify_stack_url(url: str) -> tuple[str, str] | None:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    hostname = (parsed.hostname or "").lower()
    m = _STACK_Q_RE.match(parsed.path or "")
    if not m:
        return None
    question_id = m.group(1)
    site = _STACK_HOST_SITE_MAP.get(hostname)
    if not site and hostname.endswith(".stackexchange.com"):
        subdomain = hostname[:-len(".stackexchange.com")]
        if subdomain and "." not in subdomain:
            site = f"{subdomain}.stackexchange"
    if not site:
        return None
    return (site, question_id)


# ---------------------------------------------------------------------------
# Hugging Face intercept suite
#
# HF exposes a rich public API under /api/{models,datasets,spaces}/{repo}
# that returns structured metadata — pipeline tag, library, downloads,
# likes, tags, file list, cardData, last-modified. README markdown is
# available at /<repo>/raw/{branch}/README.md. No auth required for
# public resources; gated models return 401 on the README (we keep the
# metadata card and note that the README is gated).
#
# URL coverage:
#   /<user>/<model>                       → model page
#   /<user>/<model>/blob/<branch>/<path>  → file view (GitHub-style)
#   /datasets/<user>/<ds>                 → dataset page
#   /datasets/<ds>                        → dataset page (legacy bare)
#   /spaces/<user>/<space>                → Space page
#   /papers/<arxiv_id>                    → canonicalized to arxiv.org/abs
#
# Bare single-segment URLs (/gpt2, /bert-base-uncased) are ambiguous
# (user page vs. legacy model) — we fall through to trafilatura for
# those; HF's server-rendered HTML still has enough content for a
# useful reader view.
# ---------------------------------------------------------------------------
_HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co"})
# Two-segment paths that are actually models (not reserved top-levels).
# Reserved roots that MUST NOT be treated as a model owner.
_HF_RESERVED = frozenset({
    "datasets", "spaces", "papers", "blog", "docs", "learn", "models",
    "search", "new", "login", "signup", "logout", "settings", "pricing",
    "enterprise", "join", "huggingchat", "chat", "posts", "collections",
    "tasks", "organizations", "deepresearch", "api",
})
_HF_MODEL_RE = re.compile(r"^/([\w.-]+)/([\w.-]+)/?$")
_HF_MODEL_BLOB_RE = re.compile(r"^/([\w.-]+)/([\w.-]+)/blob/([^/]+)/(.+)$")
_HF_DATASET_RE = re.compile(r"^/datasets/([\w.-]+)(?:/([\w.-]+))?/?$")
_HF_DATASET_BLOB_RE = re.compile(r"^/datasets/([\w.-]+)/([\w.-]+)/blob/([^/]+)/(.+)$")
_HF_SPACE_RE = re.compile(r"^/spaces/([\w.-]+)/([\w.-]+)/?$")


def _classify_hf_url(url: str) -> tuple[str, tuple] | None:
    """Return (kind, args) for HF URLs we have handlers for, else None.

    kind ∈ {'model', 'model_blob', 'dataset', 'dataset_blob', 'space'}.
    args is a positional tuple matching the handler's expected inputs.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if (parsed.hostname or "").lower() not in _HF_HOSTS:
        return None
    path = parsed.path or "/"

    # Model file view first (more specific than the plain model regex)
    m = _HF_MODEL_BLOB_RE.match(path)
    if m and m.group(1) not in _HF_RESERVED:
        return ("model_blob", (m.group(1), m.group(2), m.group(3), m.group(4)))

    # Dataset file view
    m = _HF_DATASET_BLOB_RE.match(path)
    if m:
        return ("dataset_blob", (m.group(1), m.group(2), m.group(3), m.group(4)))

    # Dataset page (handle both /datasets/ns/name and legacy /datasets/name)
    m = _HF_DATASET_RE.match(path)
    if m:
        ns, name = m.group(1), m.group(2)
        repo = f"{ns}/{name}" if name else ns
        return ("dataset", (repo,))

    # Space page
    m = _HF_SPACE_RE.match(path)
    if m:
        return ("space", (f"{m.group(1)}/{m.group(2)}",))

    # Model page — two-segment, not reserved
    m = _HF_MODEL_RE.match(path)
    if m and m.group(1) not in _HF_RESERVED:
        return ("model", (f"{m.group(1)}/{m.group(2)}",))

    return None


def _hf_render_markdown(md: str) -> str:
    """Render HF README markdown to HTML via markdown-it-py. Sanitised
    downstream by DOMPurify on the frontend, so we enable the full
    commonmark set including HTML passthrough."""
    if not md:
        return ""
    try:
        from markdown_it import MarkdownIt
        md_parser = MarkdownIt("commonmark", {"html": True, "linkify": True, "breaks": False})
        md_parser.enable(["table", "strikethrough"])
        return md_parser.render(md)
    except Exception:
        import html as _html
        return f"<pre class='hf-readme-raw'>{_html.escape(md)}</pre>"


def _hf_format_downloads(n: int | None) -> str:
    if not n:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _hf_build_tag_chips(tags: list[str]) -> str:
    """Filter library-prefix and pipeline-prefix internal tags; render
    the rest as visible chips. HF tags include a lot of duplicated
    metadata like 'library:transformers' and 'pipeline:text-generation'
    which are already shown as their own chips — skip those."""
    if not tags:
        return ""
    skip_prefixes = ("library:", "pipeline_tag:", "region:", "endpoints_compatible",
                     "arxiv:", "base_model:", "inference:", "autotrain", "co2_")
    cleaned = [t for t in tags if not any(t.startswith(p) for p in skip_prefixes)]
    if not cleaned:
        return ""
    return "".join(
        f'<span class="hf-tag">{_esc(t)}</span>' for t in cleaned[:16]
    )


def _hf_build_file_list(siblings: list[dict], repo_kind: str, repo_id: str) -> str:
    """Render the HF siblings list as a compact file list. Each file
    links back through /api/browse/fetch so blob-view intercepts fire
    when the user clicks a code-shaped file."""
    if not siblings:
        return ""
    path_prefix = {
        "model": f"https://huggingface.co/{repo_id}/blob/main/",
        "dataset": f"https://huggingface.co/datasets/{repo_id}/blob/main/",
        "space": f"https://huggingface.co/spaces/{repo_id}/blob/main/",
    }[repo_kind]
    items: list[str] = []
    for f in siblings[:25]:
        rfilename = f.get("rfilename") or ""
        if not rfilename:
            continue
        fetch_link = f"/api/browse/fetch?url={quote_plus(path_prefix + rfilename)}"
        items.append(
            f'<li class="hf-file-item">'
            f'<a href="{_esc(fetch_link)}">{_esc(rfilename)}</a>'
            f'</li>'
        )
    if not items:
        return ""
    more_note = ""
    if len(siblings) > 25:
        more_note = f'<li class="hf-file-more">+{len(siblings) - 25} more files</li>'
    return (
        f'<div class="hf-files">'
        f'<div class="hf-files-heading">Files</div>'
        f'<ul class="hf-files-list">{"".join(items)}{more_note}</ul>'
        f'</div>'
    )


async def _try_hf_api(url: str, request) -> dict | None:
    """Intercept Hugging Face model / dataset / space / blob URLs."""
    classified = _classify_hf_url(url)
    if not classified:
        return None
    kind, args = classified

    async with httpx.AsyncClient(
        timeout=15.0,
        headers={
            "User-Agent": "Augmentum/1.0 (+reader)",
            "Accept": "application/json",
        },
        follow_redirects=True,  # datasets sometimes 307 to namespaced form
    ) as client:

        # --- Blob views: reuse the same rendering as GitHub blobs ---
        if kind == "model_blob":
            owner, repo, branch, path = args
            return await _hf_render_blob(
                client, f"{owner}/{repo}", branch, path, url,
                repo_kind="model",
            )
        if kind == "dataset_blob":
            owner, repo, branch, path = args
            return await _hf_render_blob(
                client, f"{owner}/{repo}", branch, path, url,
                repo_kind="dataset",
            )

        # --- Metadata lookups ---
        repo_id = args[0]
        api_path = {
            "model": f"/api/models/{repo_id}",
            "dataset": f"/api/datasets/{repo_id}",
            "space": f"/api/spaces/{repo_id}",
        }[kind]
        try:
            meta_resp = await client.get(f"https://huggingface.co{api_path}")
        except Exception:
            return None
        if meta_resp.status_code != 200:
            log.info("hf_api_non_200", url=url, status=meta_resp.status_code)
            return None
        try:
            meta = meta_resp.json()
        except Exception:
            return None

        canonical_id = meta.get("id") or repo_id
        # --- Fetch README in parallel (best-effort; gated models 401) ---
        readme_path = {
            "model": f"/{canonical_id}/raw/main/README.md",
            "dataset": f"/datasets/{canonical_id}/raw/main/README.md",
            "space": f"/spaces/{canonical_id}/raw/main/README.md",
        }[kind]
        readme_md = ""
        readme_gated = False
        try:
            r = await client.get(f"https://huggingface.co{readme_path}")
            if r.status_code == 200:
                readme_md = r.text
            elif r.status_code in (401, 403):
                readme_gated = True
        except Exception:
            log.debug(
                "hf_readme_fetch_failed",
                readme_path=readme_path,
                exc_info=True,
            )

        # --- Render ---
        author = meta.get("author") or canonical_id.split("/", 1)[0]
        pipeline = meta.get("pipeline_tag") or ""
        library = meta.get("library_name") or ""
        downloads = meta.get("downloads") or 0
        likes = meta.get("likes") or 0
        license_info = (meta.get("cardData") or {}).get("license") or ""
        tags = meta.get("tags") or []
        siblings = meta.get("siblings") or []
        last_modified = meta.get("lastModified") or meta.get("last_modified") or ""
        age = _reddit_iso_to_age(last_modified)

        # Strip YAML frontmatter from the rendered README — it's machine
        # metadata, already displayed as chips above. Matches --- ... ---
        # at the very start of the document only.
        if readme_md.startswith("---"):
            end = readme_md.find("\n---", 3)
            if end > 0:
                readme_md = readme_md[end + 4:].lstrip("\n")

        readme_html = _hf_render_markdown(readme_md)
        # Proxy images in the rendered README through our CSP-safe path.
        if readme_html:
            base_url = f"https://huggingface.co/{canonical_id}/resolve/main/"
            readme_html = _proxy_external_media(readme_html, base_url=base_url)

        # --- Header chips ---
        chips: list[str] = []
        kind_label = {"model": "Model", "dataset": "Dataset", "space": "Space"}[kind]
        chips.append(f'<span class="hf-kind-chip hf-kind-{kind}">{_esc(kind_label)}</span>')
        if pipeline:
            chips.append(f'<span class="hf-pipeline">{_esc(pipeline)}</span>')
        if library:
            chips.append(f'<span class="hf-library">{_esc(library)}</span>')
        if license_info:
            chips.append(f'<span class="hf-license">{_esc(license_info)}</span>')
        chips.append(f'<span class="hf-dl">\u2b07 {_hf_format_downloads(downloads)}</span>')
        chips.append(f'<span class="hf-likes">\u2661 {likes}</span>')
        if age:
            chips.append(f'<span class="hf-updated">Updated {_esc(age)}</span>')

        tag_chips = _hf_build_tag_chips(tags)
        file_list = _hf_build_file_list(siblings, kind, canonical_id)

        gated_banner = ""
        if readme_gated:
            gated_banner = (
                '<div class="hf-gated-banner">'
                'This repository is gated. Sign in on Hugging Face to view the full README.'
                '</div>'
            )

        body_html = (
            f'<div class="hf-readme">{readme_html}</div>'
            if readme_html
            else ('<p class="hf-no-readme">No README is published for this repository.</p>' if not readme_gated else "")
        )

        tags_block = f'<div class="hf-tags">{tag_chips}</div>' if tag_chips else ""
        header_html = (
            f'<div class="hf-header">'
            f'<h1 class="hf-title">{_esc(canonical_id)}</h1>'
            f'<div class="hf-chips">{"".join(chips)}</div>'
            f'{tags_block}'
            f'</div>'
        )
        article_html = (
            f'<div class="hf-article">'
            f'{header_html}'
            f'{gated_banner}'
            f'<hr class="hf-divider">'
            f'{body_html}'
            f'{file_list}'
            f'</div>'
        )
        text = re.sub(r"<[^>]+>", " ", article_html)
        text = re.sub(r"\s+", " ", text).strip()[:20000]
        return {
            "html": article_html,
            "text": text,
            "title": canonical_id,
            "author": author,
            "date": age,
            "sitename": "Hugging Face",
            "word_count": len(text.split()),
            "reading_time_min": max(1, len(text.split()) // 238),
            "url": url,
            "favicon_url": "/api/browse/image?url=https%3A%2F%2Fhuggingface.co%2Ffavicon.ico",
            "source": "huggingface-api",
            "page_type": "reference",
        }


async def _hf_render_blob(
    client, repo_id: str, branch: str, path: str, url: str, *, repo_kind: str,
) -> dict | None:
    """Render a Hugging Face file view. Uses /resolve/ (HF's raw file
    path) in the same way the GitHub blob intercept uses
    raw.githubusercontent.com."""
    url_prefix = {
        "model": f"https://huggingface.co/{repo_id}/resolve/{branch}/",
        "dataset": f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/",
    }[repo_kind]
    raw_url = url_prefix + path
    try:
        resp = await client.get(raw_url)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    max_size = 512 * 1024
    content_bytes = resp.content[:max_size]
    truncated = len(resp.content) > max_size
    if b"\x00" in content_bytes[:4096]:
        return None
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content = content_bytes.decode("latin-1")
        except Exception:
            return None

    lang = _github_lang_for_path(path)  # reuse the extension map
    line_count = content.count("\n") + 1
    size_kb = round(len(resp.content) / 1024, 1)
    filename = path.rsplit("/", 1)[-1]

    lang_class = f' class="language-{_esc(lang)}"' if lang else ""
    lang_chip = f'<span class="hf-blob-lang">{_esc(lang)}</span>' if lang else ""
    if truncated:
        truncated_note = (
            f'<div class="hf-blob-truncated">'
            f'Truncated to the first {max_size // 1024} KB — fetch the raw file on Hugging Face for the rest.'
            f'</div>'
        )
    else:
        truncated_note = ""

    repo_href = (
        f"https://huggingface.co/{repo_id}"
        if repo_kind == "model"
        else f"https://huggingface.co/datasets/{repo_id}"
    )
    header_html = (
        f'<div class="hf-blob-header">'
        f'<h1 class="hf-blob-path">'
        f'<a href="{_esc(repo_href)}" target="_blank" rel="noopener noreferrer">{_esc(repo_id)}</a>'
        f' <span class="hf-blob-sep">/</span> '
        f'<span class="hf-blob-filename">{_esc(filename)}</span>'
        f'</h1>'
        f'<div class="hf-blob-meta">'
        f'<span class="hf-blob-branch">branch: <code>{_esc(branch)}</code></span>'
        f'<span class="hf-blob-lines">{line_count} lines</span>'
        f'<span class="hf-blob-size">{size_kb} KB</span>'
        f'{lang_chip}'
        f'<a class="hf-blob-raw" href="{_esc(raw_url)}" target="_blank" rel="noopener noreferrer">Raw</a>'
        f'</div>'
        f'<div class="hf-blob-path-sub">{_esc(path)}</div>'
        f'</div>'
    )
    article_html = (
        f'<div class="hf-blob-article">'
        f'{header_html}'
        f'{truncated_note}'
        f'<pre class="hf-blob-code"><code{lang_class}>{_esc(content)}</code></pre>'
        f'</div>'
    )
    text = content[:20000]
    return {
        "html": article_html,
        "text": text,
        "title": f"{filename} \u00b7 {repo_id}",
        "author": repo_id.split("/", 1)[0],
        "date": "",
        "sitename": "Hugging Face",
        "word_count": len(text.split()),
        "reading_time_min": max(1, len(text.split()) // 238),
        "url": url,
        "favicon_url": "/api/browse/image?url=https%3A%2F%2Fhuggingface.co%2Ffavicon.ico",
        "source": "huggingface-blob",
        "page_type": "reference",
    }


async def _try_stackexchange_api(url: str, request) -> dict | None:
    classified = _classify_stack_url(url)
    if not classified:
        return None
    site, question_id = classified

    params_base = {"order": "desc", "sort": "votes", "site": site, "filter": "withbody"}
    async with httpx.AsyncClient(
        timeout=15.0,
        headers={"User-Agent": "Augmentum/1.0 (+reader)"},
    ) as client:
        try:
            q_resp = await client.get(
                f"https://api.stackexchange.com/2.3/questions/{question_id}",
                params=params_base,
            )
            a_resp = await client.get(
                f"https://api.stackexchange.com/2.3/questions/{question_id}/answers",
                params={**params_base, "pagesize": 20},
            )
        except Exception:
            return None
        if q_resp.status_code != 200:
            return None
        try:
            q_data = q_resp.json()
            a_data = a_resp.json() if a_resp.status_code == 200 else {"items": []}
        except Exception:
            return None

    questions = q_data.get("items") or []
    if not questions:
        return None
    q = questions[0]
    title = q.get("title") or "Question"
    author = ((q.get("owner") or {}).get("display_name") or "[deleted]")
    score = q.get("score") or 0
    view_count = q.get("view_count") or 0
    answer_count = q.get("answer_count") or 0
    tags = q.get("tags") or []
    body_html = q.get("body") or ""
    is_answered = bool(q.get("is_answered"))
    age = _hn_epoch_to_age(q.get("creation_date"))

    answers = a_data.get("items") or []
    answer_blocks: list[str] = []
    for a in answers[:10]:
        a_author = ((a.get("owner") or {}).get("display_name") or "[deleted]")
        a_score = a.get("score") or 0
        a_accepted = bool(a.get("is_accepted"))
        a_body = a.get("body") or ""
        a_age = _hn_epoch_to_age(a.get("creation_date"))
        accepted_badge = '<span class="se-accepted-badge">\u2714 Accepted</span>' if a_accepted else ""
        accepted_cls = " se-accepted" if a_accepted else ""
        answer_blocks.append(
            f'<div class="se-answer{accepted_cls}">'
            f'<div class="se-answer-meta">'
            f'<span class="se-score">{a_score} points</span>'
            f'{accepted_badge}'
            f'<span class="se-author">{_esc(a_author)}</span>'
            f'<span class="se-age">{_esc(a_age)}</span>'
            f'</div>'
            f'<div class="se-answer-body">{a_body}</div>'
            f'</div>'
        )
    answers_html = "".join(answer_blocks)
    if not answers_html:
        answers_html = '<p class="se-no-answers">No answers yet.</p>'

    tag_chips = "".join(f'<span class="se-tag">{_esc(t)}</span>' for t in tags[:10])
    tags_block = f'<div class="se-tags">{tag_chips}</div>' if tag_chips else ""
    answered_chip = '<span class="se-answered">\u2714 Answered</span>' if is_answered else ""
    header_html = (
        f'<div class="se-header">'
        f'<h1 class="se-title">{_esc(title)}</h1>'
        f'<div class="se-meta">'
        f'<span class="se-score">{score} points</span>'
        f'<span class="se-views">{view_count} views</span>'
        f'<span class="se-answer-count">{answer_count} answers</span>'
        f'<span class="se-author">Asked by {_esc(author)}</span>'
        f'<span class="se-age">{_esc(age)}</span>'
        f'{answered_chip}'
        f'</div>'
        f'{tags_block}'
        f'</div>'
    )
    article_html = (
        f'<div class="se-article">'
        f'{header_html}'
        f'<div class="se-question-body">{body_html}</div>'
        f'<hr class="se-divider">'
        f'<h2 class="se-answers-heading">Answers ({answer_count})</h2>'
        f'<div class="se-answers">{answers_html}</div>'
        f'</div>'
    )
    text = re.sub(r"<[^>]+>", " ", article_html)
    text = re.sub(r"\s+", " ", text).strip()[:20000]
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    sitename_map = {
        "stackoverflow": "Stack Overflow",
        "superuser": "Super User",
        "serverfault": "Server Fault",
        "askubuntu": "Ask Ubuntu",
        "mathoverflow": "MathOverflow",
    }
    sitename = sitename_map.get(site) or "Stack Exchange"
    return {
        "html": article_html,
        "text": text,
        "title": title,
        "author": author,
        "date": age,
        "sitename": sitename,
        "word_count": len(text.split()),
        "reading_time_min": max(1, len(text.split()) // 238),
        "url": url,
        "favicon_url": f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3D{hostname}%26sz%3D32",
        "source": "stackexchange-api",
        "page_type": "forum",
    }


# ---------------------------------------------------------------------------
# RSS / Atom feed renderer (when the URL itself IS a feed)
#
# The "RSS fallback" infrastructure further down handles the case where
# we've already fetched an HTML page and want to re-hydrate it from a
# linked feed. THIS handler is the opposite: the user navigates directly
# to feed.xml / /atom / /rss and we render the entry list as a clean
# reading surface instead of dumping pretty-printed XML.
#
# Detection: sniff the first 2KB for <rss / <feed / <rdf:RDF. This
# catches Content-Type mismatches (servers that label feeds as text/xml
# or application/xml) without depending on the URL ending in .xml.
# ---------------------------------------------------------------------------
_FEED_SNIFF_RE = re.compile(
    r"(?:<\?xml[^>]*\?>\s*)?"
    r"(?:<\?[a-z][^>]*\?>\s*)*"   # other processing instructions (xml-stylesheet etc.)
    r"(?:<!--.*?-->\s*)*"
    r"<(?:rss\b|feed\b|rdf:RDF\b)",
    re.IGNORECASE | re.DOTALL,
)


def _looks_like_feed(raw_html: str) -> bool:
    """Quick sniff — does this look like an RSS/Atom/RDF feed root?

    Looks at the first 2KB only so we don't slow down the normal HTML
    path. The XML prolog + optional comments are allowed before the
    root element.
    """
    if not raw_html:
        return False
    head = raw_html[:2048].lstrip()
    if not head.startswith("<"):
        return False
    return bool(_FEED_SNIFF_RE.match(head))


def _feed_text(elem) -> str:
    """Concatenate an ElementTree element's text + children's tail."""
    if elem is None:
        return ""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        if child.text:
            parts.append(child.text)
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _feed_strip_ns(tag: str) -> str:
    """Drop the XML namespace prefix from an ElementTree tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _try_render_feed(url: str, raw_html: str, fetch_meta: dict) -> dict | None:
    """Render an RSS/Atom/RDF feed as a list of entries.

    Returns None when the body doesn't sniff as a feed, when XML
    parsing fails, or when no entries are found — in all those cases
    the caller falls back to file-type rendering or article extraction.
    """
    if not _looks_like_feed(raw_html):
        return None

    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(raw_html.encode("utf-8", errors="replace"))
    except Exception:
        log.debug("feed_parse_failed", url=url, exc_info=True)
        return None

    root_tag = _feed_strip_ns(root.tag).lower()
    entries: list[dict] = []
    feed_title = ""
    feed_desc = ""
    feed_link = ""

    def _find_child(parent, *names):
        """Return the first child whose stripped tag matches any name."""
        wanted = {n.lower() for n in names}
        for child in parent:
            if _feed_strip_ns(child.tag).lower() in wanted:
                return child
        return None

    def _findall_child(parent, name):
        name_lc = name.lower()
        return [c for c in parent if _feed_strip_ns(c.tag).lower() == name_lc]

    if root_tag == "rss":
        channel = _find_child(root, "channel")
        if channel is None:
            return None
        feed_title = _feed_text(_find_child(channel, "title"))
        feed_desc = _feed_text(_find_child(channel, "description", "subtitle"))
        link_el = _find_child(channel, "link")
        feed_link = _feed_text(link_el)
        for item in _findall_child(channel, "item"):
            title = _feed_text(_find_child(item, "title"))
            link_el = _find_child(item, "link")
            link = _feed_text(link_el)
            if not link and link_el is not None:
                link = link_el.get("href", "")
            pub = _feed_text(_find_child(item, "pubdate", "date", "published"))
            author = _feed_text(_find_child(item, "author", "creator"))
            # Prefer content:encoded over description (full body vs. summary).
            content_el = None
            for child in item:
                if _feed_strip_ns(child.tag).lower() == "encoded":
                    content_el = child
                    break
            summary = _feed_text(content_el) if content_el is not None else ""
            if not summary:
                summary = _feed_text(_find_child(item, "description", "summary"))
            entries.append({
                "title": title,
                "link": link,
                "date": pub,
                "author": author,
                "summary": summary,
            })
    elif root_tag == "feed":
        # Atom
        feed_title = _feed_text(_find_child(root, "title"))
        feed_desc = _feed_text(_find_child(root, "subtitle", "description"))
        for child in root:
            if _feed_strip_ns(child.tag).lower() == "link":
                href = child.get("href", "")
                rel = child.get("rel", "alternate")
                if rel in ("alternate", "") and href:
                    feed_link = href
                    break
        for entry in _findall_child(root, "entry"):
            title = _feed_text(_find_child(entry, "title"))
            link = ""
            for child in entry:
                if _feed_strip_ns(child.tag).lower() == "link":
                    href = child.get("href", "")
                    rel = child.get("rel", "alternate")
                    if rel in ("alternate", "") and href:
                        link = href
                        break
            pub = (
                _feed_text(_find_child(entry, "published"))
                or _feed_text(_find_child(entry, "updated"))
            )
            author_el = _find_child(entry, "author")
            author = ""
            if author_el is not None:
                name_el = _find_child(author_el, "name")
                author = _feed_text(name_el) if name_el is not None else _feed_text(author_el)
            summary = (
                _feed_text(_find_child(entry, "content"))
                or _feed_text(_find_child(entry, "summary"))
            )
            entries.append({
                "title": title,
                "link": link,
                "date": pub,
                "author": author,
                "summary": summary,
            })
    elif root_tag == "rdf":
        # RSS 1.0 / RDF — channel sibling of items
        feed_title = _feed_text(_find_child(root, "title"))
        feed_desc = _feed_text(_find_child(root, "description"))
        for item in _findall_child(root, "item"):
            title = _feed_text(_find_child(item, "title"))
            link = _feed_text(_find_child(item, "link"))
            pub = _feed_text(_find_child(item, "date", "pubdate"))
            author = _feed_text(_find_child(item, "creator"))
            summary = _feed_text(_find_child(item, "description"))
            entries.append({
                "title": title,
                "link": link,
                "date": pub,
                "author": author,
                "summary": summary,
            })
    else:
        return None

    if not entries:
        return None

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    # ---- Render ----
    items_html: list[str] = []
    text_parts: list[str] = [feed_title or hostname]
    if feed_desc:
        text_parts.append(feed_desc)

    for e in entries[:60]:
        title = (e["title"] or "Untitled").strip()
        link = (e["link"] or "").strip()
        pub_short = (e["date"] or "")[:25].strip()
        author = (e["author"] or "").strip()
        # Strip HTML, collapse whitespace, trim
        raw_summary = e["summary"] or ""
        summary_text = re.sub(r"<[^>]+>", " ", raw_summary)
        summary_text = re.sub(r"\s+", " ", summary_text).strip()
        if len(summary_text) > 400:
            summary_text = summary_text[:400].rsplit(" ", 1)[0] + "…"

        meta_bits: list[str] = []
        if pub_short:
            meta_bits.append(_esc(pub_short))
        if author:
            meta_bits.append(_esc(author))
        meta_line = " &middot; ".join(meta_bits)

        if link:
            title_html = (
                f'<a href="/api/browse/fetch?url={quote_plus(link)}" '
                f'data-browse-url="{_esc(link)}">{_esc(title)}</a>'
            )
        else:
            title_html = _esc(title)

        items_html.append(
            f'<article class="browse-feed-entry">'
            f'<h3 class="browse-feed-entry-title">{title_html}</h3>'
            + (f'<div class="browse-feed-entry-meta">{meta_line}</div>' if meta_line else "")
            + (f'<p class="browse-feed-entry-summary">{_esc(summary_text)}</p>' if summary_text else "")
            + f'</article>'
        )
        text_parts.append(f"{title}{f' — {summary_text}' if summary_text else ''}")

    site_link_html = (
        f'<a class="browse-feed-site-link" href="{_esc(feed_link)}" '
        f'target="_blank" rel="noopener noreferrer">{_esc(feed_link)}</a>'
        if feed_link else ""
    )

    feed_html = (
        f'<div class="browse-feed">'
        f'<header class="browse-feed-header">'
        f'<h1>{_esc(feed_title or hostname or "Feed")}</h1>'
        + (f'<p class="browse-feed-description">{_esc(feed_desc)}</p>' if feed_desc else "")
        + site_link_html
        + f'</header>'
        f'<div class="browse-feed-entries">{"".join(items_html)}</div>'
        f'</div>'
    )

    full_text = "\n\n".join(text_parts)

    return {
        "html": feed_html,
        "text": full_text[:20_000],
        "title": feed_title or hostname or "Feed",
        "author": "",
        "date": "",
        "sitename": hostname,
        "word_count": len(full_text.split()),
        "reading_time_min": max(1, len(full_text.split()) // 238),
        "url": str(fetch_meta.get("url", url)),
        "favicon_url": (
            f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons"
            f"%3Fdomain%3D{hostname}%26sz%3D32"
        ),
        "source": "feed",
        "page_type": "feed",
    }


# ---------------------------------------------------------------------------
# arXiv intercept
#
# arxiv.org/abs/<id> pages render fine via trafilatura but lose the
# structured metadata (authors, primary category, DOI, journal ref,
# version history). The export.arxiv.org Atom API returns all of it
# cleanly. Also handles /pdf/<id> URLs since _canonicalize_url already
# rewrites those to /abs/<id>.
#
# Versioning: 2301.01234 and 2301.01234v2 both resolve via id_list; the
# API always returns the canonical (latest) version. Old-style IDs
# (math.GT/0301001) work too — the export API accepts both forms.
# ---------------------------------------------------------------------------
_ARXIV_ABS_RE = re.compile(
    r"^(?:https?://)?(?:www\.|export\.)?arxiv\.org/abs/([\w./-]+?)(?:v\d+)?/?(?:[?#].*)?$",
    re.IGNORECASE,
)


async def _try_arxiv_api(url: str, request) -> dict | None:
    """Render arXiv abstract pages via the export.arxiv.org Atom API.

    Returns clean title, authors, abstract, categories, and download
    links. Falls through (None) on parse failure so trafilatura still
    has a shot at the regular /abs/ HTML.
    """
    m = _ARXIV_ABS_RE.match(url)
    if not m:
        return None
    arxiv_id = m.group(1)

    http_client = getattr(request.app.state, "http_client", None)
    if not http_client:
        return None

    try:
        resp = await http_client.get(
            f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
            timeout=12.0,
            headers={"User-Agent": "Augmentum/1.0 (self-hosted AI proxy)"},
        )
        if resp.status_code != 200:
            return None
        atom_xml = resp.text
    except Exception as exc:
        log.debug("arxiv_api_fetch_failed", url=url, error=str(exc))
        return None

    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(atom_xml)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        title = (entry.findtext("atom:title", "", ns) or "").strip()
        title = re.sub(r"\s+", " ", title)
        summary = (entry.findtext("atom:summary", "", ns) or "").strip()
        # arXiv abstracts often have hard-wrapped lines mid-sentence;
        # collapse single linebreaks but preserve paragraph breaks.
        summary = re.sub(r"(?<!\n)\n(?!\n)", " ", summary)
        summary = re.sub(r"[ \t]+", " ", summary).strip()

        published = (entry.findtext("atom:published", "", ns) or "")[:10]
        updated = (entry.findtext("atom:updated", "", ns) or "")[:10]

        authors = []
        for a in entry.findall("atom:author", ns):
            name = (a.findtext("atom:name", "", ns) or "").strip()
            if name:
                authors.append(name)

        categories: list[str] = []
        for c in entry.findall("atom:category", ns):
            term = c.get("term", "")
            if term and term not in categories:
                categories.append(term)

        primary_cat = ""
        pc = entry.find("arxiv:primary_category", ns)
        if pc is not None:
            primary_cat = pc.get("term", "")

        journal_ref = (entry.findtext("arxiv:journal_ref", "", ns) or "").strip()
        doi = (entry.findtext("arxiv:doi", "", ns) or "").strip()
        comment = (entry.findtext("arxiv:comment", "", ns) or "").strip()

        # The <id> field is the canonical arXiv URL — use its version
        # suffix as authoritative if we didn't get one in the user's URL.
        canonical_id = (entry.findtext("atom:id", "", ns) or "").strip()
        canonical_match = re.search(r"/abs/([\w./-]+?)(?:v\d+)?$", canonical_id)
        bare_id = canonical_match.group(1) if canonical_match else arxiv_id
    except Exception:
        log.debug("arxiv_parse_failed", url=url, exc_info=True)
        return None

    if not title or not summary:
        return None  # malformed response — let trafilatura handle it

    pdf_url = f"https://arxiv.org/pdf/{bare_id}"
    html_url = f"https://arxiv.org/html/{bare_id}"
    eprint_url = f"https://arxiv.org/e-print/{bare_id}"

    # ---- Render ----
    authors_html = ", ".join(_esc(a) for a in authors) or "Unknown"

    cat_chips = "".join(
        f'<span class="browse-schema-tag">{_esc(c)}</span>'
        for c in categories[:8]
    )

    meta_lines: list[str] = []
    if primary_cat and primary_cat not in categories[:1]:
        meta_lines.append(f"<strong>Primary:</strong> {_esc(primary_cat)}")
    if journal_ref:
        meta_lines.append(f"<strong>Journal:</strong> {_esc(journal_ref)}")
    if doi:
        meta_lines.append(
            f'<strong>DOI:</strong> '
            f'<a href="https://doi.org/{_esc(doi)}" target="_blank" rel="noopener noreferrer">'
            f'{_esc(doi)}</a>'
        )
    if comment:
        meta_lines.append(f"<strong>Note:</strong> {_esc(comment)}")
    if updated and updated != published:
        meta_lines.append(f"<strong>Updated:</strong> {_esc(updated)}")
    meta_html = "<br>".join(meta_lines)

    abstract_paragraphs = "".join(
        f"<p>{_esc(p.strip())}</p>"
        for p in summary.split("\n\n")
        if p.strip()
    ) or f"<p>{_esc(summary)}</p>"

    article_html = (
        f'<div class="browse-schema-card browse-schema-arxiv">'
        f'<div class="browse-schema-details">'
        f'<h1 class="browse-schema-title">{_esc(title)}</h1>'
        f'<div class="browse-schema-meta">{authors_html}</div>'
        f'<div class="browse-schema-meta">'
        f'<strong>arXiv:{_esc(bare_id)}</strong>'
        + (f' &middot; {_esc(published)}' if published else "")
        + f'</div>'
        + (f'<div class="browse-schema-tags">{cat_chips}</div>' if cat_chips else "")
        + (f'<div class="browse-schema-meta">{meta_html}</div>' if meta_html else "")
        + f'<h2>Abstract</h2>'
        f'<div class="browse-schema-desc">{abstract_paragraphs}</div>'
        f'<div class="browse-pdf-actions">'
        f'<a href="/api/browse/fetch?url={quote_plus(pdf_url)}" '
        f'data-browse-url="{_esc(pdf_url)}">PDF</a>'
        f'<a href="/api/browse/fetch?url={quote_plus(html_url)}" '
        f'data-browse-url="{_esc(html_url)}">HTML version</a>'
        f'<a href="{_esc(eprint_url)}" target="_blank" rel="noopener noreferrer">Source (e-print)</a>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    short_authors = ", ".join(authors[:3])
    if len(authors) > 3:
        short_authors += f" +{len(authors) - 3} more"
    text = f"{title}\n\nAuthors: {', '.join(authors)}\n\nAbstract: {summary}"

    return {
        "html": article_html,
        "text": text,
        "title": title,
        "author": short_authors,
        "date": published,
        "sitename": "arXiv",
        "word_count": len(text.split()),
        "reading_time_min": max(1, len(text.split()) // 238),
        "url": url,
        "favicon_url": (
            "/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons"
            "%3Fdomain%3Darxiv.org%26sz%3D32"
        ),
        "source": "arxiv-api",
        "page_type": "paper",
    }


# ---------------------------------------------------------------------------
# Discourse forum intercept
#
# Discourse runs ~hundreds of thousands of public forums (HuggingFace,
# Meta Discourse, Rust Lang, BBC R&D, every Rails-shaped community) and
# uses one shared URL shape: `/t/<slug>/<id>[/<post_no>]`. Every site
# exposes `/t/<id>.json` unauthenticated, which returns the full topic
# stream (no slug required) plus author/category/tags metadata.
#
# Detection: probe the .json endpoint. Non-Discourse URLs that happen
# to match `/t/<digits>` (rare) return 404 or non-JSON and we fall
# through. Cheap because we're going to fetch the page anyway.
#
# Renders OP + replies, capped at 50 posts. `cooked` is server-rendered
# HTML — script tags stripped, image src rewritten through our proxy so
# we don't leak the user's IP to the forum's CDN.
# ---------------------------------------------------------------------------
_DISCOURSE_TOPIC_RE = re.compile(
    r"^/t/(?:[^/]+/)?(\d+)(?:/(\d+))?/?$"
)

# Hosts that LOOK like /t/<digits> URLs but aren't Discourse — keep the
# false-positive probe rate down. Twitter/X uses /<user>/status/<id> not
# /t/, but other apps with /t/ paths (URL shorteners, etc.) get hit.
_DISCOURSE_BLOCKLIST = frozenset({
    "t.co", "tinyurl.com", "bit.ly", "goo.gl", "ow.ly", "buff.ly",
    "github.com", "gitlab.com",  # have /t/ in some workflow paths
})


def _discourse_proxy_html(html_fragment: str, base_url: str) -> str:
    """Sanitise Discourse `cooked` HTML and proxy its images.

    Strips <script> tags (Discourse shouldn't emit any but plugins might),
    rewrites <img src> through /api/browse/image so we don't leak the
    user's IP to the forum's image CDN, and resolves relative URLs.
    """
    if not html_fragment:
        return ""
    # Drop scripts defensively.
    html_fragment = re.sub(
        r"<script\b[^>]*>.*?</script>",
        "",
        html_fragment,
        flags=re.IGNORECASE | re.DOTALL,
    )

    def _rewrite_img(match: re.Match) -> str:
        src = match.group(1) or match.group(2) or ""
        if not src or src.startswith("/api/browse/image"):
            return match.group(0)
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = urljoin(base_url, src)
        return match.group(0).replace(
            match.group(1) or match.group(2),
            f"/api/browse/image?url={quote_plus(src)}",
        )

    return re.sub(
        r'<img\b[^>]*?\bsrc="([^"]+)"|<img\b[^>]*?\bsrc=\'([^\']+)\'',
        _rewrite_img,
        html_fragment,
    )


async def _try_discourse_api(url: str, request) -> dict | None:
    """Render Discourse topic pages via the unauthenticated .json endpoint."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().removeprefix("www.")
    if not hostname or hostname in _DISCOURSE_BLOCKLIST:
        return None

    m = _DISCOURSE_TOPIC_RE.match(parsed.path or "/")
    if not m:
        return None

    topic_id = m.group(1)
    post_no_focus = m.group(2)  # if user linked to /t/slug/id/<post_no>

    http_client = getattr(request.app.state, "http_client", None)
    if not http_client:
        return None

    api_url = f"{parsed.scheme or 'https'}://{parsed.netloc}/t/{topic_id}.json"
    try:
        resp = await http_client.get(
            api_url,
            timeout=10.0,
            headers={
                "User-Agent": "Augmentum/1.0 (self-hosted AI proxy)",
                "Accept": "application/json",
            },
        )
        if resp.status_code != 200:
            return None
        ctype = resp.headers.get("content-type", "")
        if "json" not in ctype.lower():
            return None
        data = resp.json()
    except Exception as exc:
        log.debug("discourse_api_fetch_failed", url=url, error=str(exc))
        return None

    # Shape sniff — Discourse topic JSON has these required keys.
    if not isinstance(data, dict) or "post_stream" not in data or "title" not in data:
        return None

    title = (data.get("title") or "").strip()
    posts = (data.get("post_stream") or {}).get("posts") or []
    if not isinstance(posts, list) or not posts:
        return None

    base_url = f"{parsed.scheme or 'https'}://{parsed.netloc}"
    category_name = data.get("category_name") or ""
    tags = data.get("tags") or []
    created_at = (data.get("created_at") or "")[:10]
    views = data.get("views")
    reply_count = data.get("posts_count")

    # Header
    chips: list[str] = []
    if category_name:
        chips.append(f'<span class="browse-schema-tag">{_esc(category_name)}</span>')
    for tag in tags[:10]:
        chips.append(f'<span class="browse-schema-tag">{_esc(str(tag))}</span>')
    chips_html = (
        f'<div class="browse-schema-tags">{"".join(chips)}</div>' if chips else ""
    )

    meta_bits: list[str] = []
    if reply_count:
        meta_bits.append(f"{reply_count} posts")
    if views:
        meta_bits.append(f"{views} views")
    if created_at:
        meta_bits.append(_esc(created_at))
    meta_line = " &middot; ".join(meta_bits)

    # Posts
    post_blocks: list[str] = []
    text_parts: list[str] = [title]
    for p in posts[:50]:
        if not isinstance(p, dict):
            continue
        username = p.get("username") or p.get("name") or "unknown"
        display_name = p.get("name") or username
        post_no = p.get("post_number", "")
        created = (p.get("created_at") or "")[:10]
        cooked = p.get("cooked") or ""
        body_html = _discourse_proxy_html(cooked, base_url)

        # Strip HTML for the plain-text representation.
        body_text = re.sub(r"<[^>]+>", " ", cooked)
        body_text = re.sub(r"\s+", " ", body_text).strip()
        if body_text:
            text_parts.append(f"@{username}: {body_text}")

        avatar_tmpl = p.get("avatar_template") or ""
        avatar_url = ""
        if avatar_tmpl:
            # Discourse avatar templates have {size} placeholders.
            avatar_path = avatar_tmpl.replace("{size}", "48")
            if avatar_path.startswith("//"):
                avatar_url = "https:" + avatar_path
            elif avatar_path.startswith("/"):
                avatar_url = urljoin(base_url, avatar_path)
            else:
                avatar_url = avatar_path
            avatar_url = f"/api/browse/image?url={quote_plus(avatar_url)}"

        focused = (
            ' browse-discourse-post--focused'
            if post_no_focus and str(post_no) == post_no_focus
            else ""
        )

        avatar_html = (
            f'<img class="browse-discourse-avatar" src="{_esc(avatar_url)}" '
            f'alt="" loading="lazy">'
            if avatar_url else ""
        )
        post_blocks.append(
            f'<article class="browse-discourse-post{focused}">'
            f'<header class="browse-discourse-post-header">'
            f'{avatar_html}'
            f'<span class="browse-discourse-post-author">{_esc(display_name)}'
            + (f' <span class="browse-discourse-post-username">@{_esc(username)}</span>'
               if display_name != username else "")
            + f'</span>'
            + (f' <span class="browse-discourse-post-date">{_esc(created)}</span>' if created else "")
            + (f' <span class="browse-discourse-post-no">#{post_no}</span>' if post_no else "")
            + f'</header>'
            f'<div class="browse-discourse-post-body">{body_html}</div>'
            f'</article>'
        )

    article_html = (
        f'<div class="browse-discourse-topic">'
        f'<header class="browse-discourse-header">'
        f'<h1 class="browse-discourse-title">{_esc(title)}</h1>'
        + (f'<div class="browse-discourse-meta">{meta_line}</div>' if meta_line else "")
        + chips_html
        + f'</header>'
        f'<div class="browse-discourse-posts">{"".join(post_blocks)}</div>'
        f'<div class="browse-pdf-actions">'
        f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">Open on {_esc(hostname)} →</a>'
        f'</div>'
        f'</div>'
    )

    full_text = "\n\n".join(text_parts)

    return {
        "html": article_html,
        "text": full_text[:30_000],
        "title": title,
        "author": (posts[0].get("username") if posts else "") or "",
        "date": created_at,
        "sitename": hostname,
        "word_count": len(full_text.split()),
        "reading_time_min": max(1, len(full_text.split()) // 238),
        "url": url,
        "favicon_url": (
            f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons"
            f"%3Fdomain%3D{hostname}%26sz%3D32"
        ),
        "source": "discourse-api",
        "page_type": "forum",
    }


# ---------------------------------------------------------------------------
# RSS / Atom feed fallback
#
# Many blog and publication sites (Substack, Ghost, WordPress, Tumblr,
# Medium publications, corporate blogs, news sites) render dynamically
# but expose an RSS or Atom feed that includes the full post body inside
# `<content>` / `<description>` / `<content:encoded>`. When trafilatura
# returns junk or too little text, we try to find that feed via the
# `<link rel="alternate" type="application/rss+xml">` (or atom) hint in
# the raw HTML, fetch it, and pull the entry whose `<link>` matches the
# URL we were originally trying to render.
#
# Skipped silently when: no feed link in head, feed fetch fails, no
# entry matches the URL, or the entry body is a short excerpt rather
# than the full post. In all those cases we let Wayback take over.
# ---------------------------------------------------------------------------
_RSS_MIN_CONTENT_CHARS = 500


def _discover_rss_url(raw_html: str, base_url: str) -> str | None:
    """Return the first RSS or Atom feed URL declared in <head> via
    <link rel="alternate" ...>, or None."""
    if not raw_html:
        return None
    # Match <link> tags declaring alternate feeds. Narrow the search to
    # the <head> section when possible — some sites list feeds in body
    # too but head-level links are almost always canonical.
    head_end = raw_html.lower().find("</head>")
    scope = raw_html[: head_end if head_end > 0 else 8192]
    pattern = re.compile(
        r'<link\b([^>]*?)>',
        re.IGNORECASE,
    )
    for attrs in pattern.findall(scope):
        # Accept attr ordering in any order: find rel / type / href.
        rel_m = re.search(r'rel=["\']([^"\']+)["\']', attrs, re.IGNORECASE)
        if not rel_m or "alternate" not in rel_m.group(1).lower():
            continue
        type_m = re.search(r'type=["\']([^"\']+)["\']', attrs, re.IGNORECASE)
        if not type_m:
            continue
        mime = type_m.group(1).lower()
        if "rss" not in mime and "atom" not in mime:
            continue
        href_m = re.search(r'href=["\']([^"\']+)["\']', attrs, re.IGNORECASE)
        if not href_m:
            continue
        href = href_m.group(1).strip()
        if not href:
            continue
        return urljoin(base_url, href)
    return None


def _parse_feed_for_url(feed_xml: str, target_url: str) -> dict | None:
    """Find the entry in the RSS/Atom feed whose <link> matches the
    target URL (with trailing slash forgiveness) and return its content
    as a partial data dict. Returns None when no match or when the
    content is too short to be a real body (many feeds publish only
    excerpts).
    """
    if not feed_xml:
        return None
    from xml.etree import ElementTree as ET
    try:
        root = ET.fromstring(feed_xml)
    except ET.ParseError:
        return None

    # Namespace-agnostic tag lookup via {*}. RSS uses plain tags like
    # <item>, <link>, <description>; Atom wraps in the Atom namespace.
    entries = list(root.iter("{*}entry")) + list(root.iter("item"))
    if not entries:
        return None

    target_trimmed = target_url.rstrip("/")
    import html as _html

    for entry in entries:
        # Atom: <link href="..."/>. RSS: <link>url</link>.
        entry_link = ""
        for link_el in list(entry.iter("{*}link")) + list(entry.iter("link")):
            href = link_el.attrib.get("href") or (link_el.text or "")
            href = href.strip()
            if href:
                entry_link = href
                break
        if not entry_link:
            continue
        if entry_link.rstrip("/") != target_trimmed:
            continue

        # Title
        title_el = next(iter(entry.iter("{*}title")), None) or next(iter(entry.iter("title")), None)
        title = (title_el.text or "").strip() if title_el is not None else ""

        # Body. Order of preference: content:encoded (full body in RSS 2.0
        # extension), Atom <content>, Atom <summary>, RSS <description>.
        body = ""
        for tag in (
            "{http://purl.org/rss/1.0/modules/content/}encoded",
            "{*}content",
            "{*}summary",
            "description",
        ):
            el = next(iter(entry.iter(tag)), None)
            if el is not None and (el.text or "").strip():
                body = el.text or ""
                break
        if not body:
            continue

        body = _html.unescape(body)
        text = re.sub(r"<[^>]+>", " ", body)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < _RSS_MIN_CONTENT_CHARS:
            # Feed ships only an excerpt — not worth using as the body;
            # caller will fall through to Wayback for a real extraction.
            return None
        return {
            "title": title,
            "html": body,
            "text": text[:20000],
        }
    return None


async def _try_rss_fallback(url: str, raw_html: str) -> dict | None:
    """Find and fetch an RSS/Atom feed for `url`. Returns a partial data
    dict (title + html + text) or None. Caller is responsible for
    merging the returned fields into its response envelope."""
    if not raw_html:
        return None
    feed_url = _discover_rss_url(raw_html, url)
    if not feed_url:
        return None
    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "Augmentum/1.0 (+reader)"},
        ) as client:
            resp = await client.get(feed_url)
        if resp.status_code != 200:
            return None
        ctype = (resp.headers.get("content-type") or "").lower()
        if "xml" not in ctype and "rss" not in ctype and "atom" not in ctype:
            # Feed URL served HTML — probably a landing page, not a feed.
            return None
    except Exception:
        return None

    result = _parse_feed_for_url(resp.text, url)
    if result:
        result["source"] = "rss-fallback"
        log.info("browse_rss_fallback_hit", url=url, feed_url=feed_url,
                 chars=len(result.get("text", "")))
    return result


async def _detect_embedded_videos(html: str, request: Request) -> list[dict]:
    """Scan article HTML for embedded video iframes. Fetch transcripts for YouTube.

    Returns list of: {platform, video_id, title, channel, thumbnail, transcript, embed_url}
    """
    if not html:
        return []

    # Extract YouTube video IDs from embed iframes
    yt_pattern = re.compile(
        r'(?:youtube\.com|youtube-nocookie\.com)/embed/([a-zA-Z0-9_-]{11})',
        re.IGNORECASE,
    )
    yt_ids = list(dict.fromkeys(yt_pattern.findall(html)))  # dedupe, preserve order

    results: list[dict] = []

    # Detect non-YouTube embedded video platforms (no metadata fetch needed)
    _extra_platform_patterns: list[tuple[str, str, re.Pattern[str]]] = [
        ("bilibili", "player.bilibili.com", re.compile(r'player\.bilibili\.com[^"]*bvid=([^&"]+)', re.I)),
        ("archive", "archive.org/embed", re.compile(r'archive\.org/embed/([^"/?#\s]+)', re.I)),
        ("peertube", "/videos/embed/", re.compile(r'(https?://[^"]+/videos/embed/[a-zA-Z0-9-]+)', re.I)),
    ]
    for platform, marker, pat in _extra_platform_patterns:
        if marker in html:
            for match in pat.finditer(html):
                embed_url = match.group(0) if platform == "peertube" else ""
                results.append({
                    "platform": platform,
                    "embed_url": embed_url or match.group(0),
                    "title": "",
                })

    if not yt_ids:
        return results

    http_client = getattr(request.app.state, "http_client", None)

    async def _fetch_yt_video(vid: str) -> dict | None:
        """Fetch metadata + transcript for a single YouTube video."""
        entry: dict = {
            "platform": "youtube",
            "video_id": vid,
            "title": "",
            "channel": "",
            "thumbnail": f"https://img.youtube.com/vi/{vid}/hqdefault.jpg",
            "transcript": "",
            "embed_url": f"https://www.youtube.com/embed/{vid}",
        }

        # oEmbed for metadata
        if http_client:
            try:
                oembed_resp = await http_client.get(
                    "https://www.youtube.com/oembed",
                    params={"url": f"https://www.youtube.com/watch?v={vid}", "format": "json"},
                    timeout=5.0,
                )
                if oembed_resp.status_code == 200:
                    oembed = oembed_resp.json()
                    entry["title"] = oembed.get("title", "")
                    entry["channel"] = oembed.get("author_name", "")
            except Exception as exc:
                log.debug("youtube_oembed_history_fetch_failed", vid=vid, error=str(exc))

        # Transcript
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            ytt = YouTubeTranscriptApi()
            raw_t = await asyncio.to_thread(ytt.fetch, vid, languages=["en"])
            segments = [s.text for s in raw_t]
            if segments:
                entry["transcript"] = " ".join(segments)
        except Exception:
            # Many videos don't have transcripts — that's fine; debug
            # log so a YouTube API change (auth, rate limit, schema) is
            # findable rather than blending into "no transcript".
            log.debug("yt_transcript_fetch_failed", vid=vid, exc_info=True)

        return entry

    # Fetch all videos in parallel (cap at 5 to avoid hammering)
    tasks = [_fetch_yt_video(vid) for vid in yt_ids[:5]]
    fetched = await asyncio.gather(*tasks, return_exceptions=True)
    for item in fetched:
        if isinstance(item, dict):
            results.append(item)

    return results


# ---------------------------------------------------------------------------
# URL canonicalization
#
# Run before any fetch. Collapses URL variants that would otherwise hit
# different intercepts or different content, and strips tracking junk
# that pollutes reputation scoring and share links. All transformations
# preserve user intent — they only fix up aliases, short URLs, and
# promotional parameters, never change the destination page.
# ---------------------------------------------------------------------------
_TRACKING_PARAMS: frozenset[str] = frozenset({
    # Standard UTM family
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_name", "utm_cid", "utm_reader", "utm_viz_id", "utm_pubreferrer",
    # Facebook, Google, Yandex, Microsoft click IDs
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "yclid", "twclid",
    # Affiliate / referral
    "ref", "referrer", "referral", "source", "share",
    # Amazon / shopping trackers
    "tag", "linkCode", "camp", "creative", "creativeASIN", "ie", "qid",
    "psc", "th", "sprefix", "sr", "crid", "keywords",
    # Medium / Substack source tags
    "source", "subId1", "subId2",
})


def _canonicalize_url(url: str) -> str:
    """Normalise a URL before fetch.

    Handles:
      * Twitter aliases: x.com / vxtwitter.com / fxtwitter.com / nitter.net
        → canonical twitter.com
      * arXiv /pdf/ → /abs/ so we get HTML, not a PDF blob
      * Reddit short URL redd.it/<id> → full comments URL
      * AMP paths (/amp/, /amp/s/, ?amp=1) stripped when we can detect a
        non-AMP sibling
      * Tracking params stripped (utm_*, fbclid, gclid, etc.)
      * www → preserved (some sites 301 without it, some 301 with it;
        leaving alone matches user clicks)
    """
    try:
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
        p = urlparse(url)
    except Exception:
        return url
    host = (p.hostname or "").lower()
    path = p.path or "/"

    # --- Domain aliases ---
    if host in ("x.com", "www.x.com", "vxtwitter.com", "www.vxtwitter.com",
                "fxtwitter.com", "www.fxtwitter.com", "fixupx.com"):
        host = "twitter.com"
    elif host in ("nitter.net",):
        host = "twitter.com"

    # --- arXiv PDF → abstract page (ID contains dot: 2301.01234) ---
    if host.endswith("arxiv.org") and path.startswith("/pdf/"):
        # /pdf/2301.01234.pdf → /abs/2301.01234
        m = re.match(r"^/pdf/([\w.-]+?)(?:\.pdf)?$", path)
        if m:
            path = f"/abs/{m.group(1)}"

    # --- Hugging Face paper pages → arXiv abstract. HF papers index the
    #     same arXiv ID; canonicalizing sends the URL through our arXiv
    #     handling instead of rendering HF's lightweight wrapper page.
    if host in ("huggingface.co", "www.huggingface.co") and path.startswith("/papers/"):
        m = re.match(r"^/papers/([\w.-]+)/?$", path)
        if m:
            arxiv_id = m.group(1)
            return f"https://arxiv.org/abs/{arxiv_id}"

    # --- redd.it short URL ---
    if host == "redd.it" or host == "www.redd.it":
        # https://redd.it/abc123 → https://reddit.com/comments/abc123
        m = re.match(r"^/([a-zA-Z0-9]+)/?$", path)
        if m:
            host = "reddit.com"
            path = f"/comments/{m.group(1)}"

    # --- Strip /amp/ suffix — most sites serve the canonical at the same
    # path without /amp/. This is heuristic; when it guesses wrong the
    # server 404s and we fall back to the original URL via a retry-free
    # path (trafilatura still works on the AMP HTML if the strip fails).
    if path.endswith("/amp/"):
        path = path[: -len("amp/")]
    elif path.endswith("/amp"):
        path = path[: -len("/amp")]

    # --- Strip tracking query params ---
    qs = parse_qsl(p.query, keep_blank_values=False)
    clean_qs = [(k, v) for (k, v) in qs if k.lower() not in _TRACKING_PARAMS]
    # Remove ?amp=1 / amp=true style
    clean_qs = [(k, v) for (k, v) in clean_qs if not (k.lower() == "amp" and v.lower() in ("1", "true"))]
    new_query = urlencode(clean_qs, doseq=True) if clean_qs else ""

    # --- Reassemble ---
    new_netloc = host + (f":{p.port}" if p.port else "")
    canonical = urlunparse((
        p.scheme or "https",
        new_netloc,
        path,
        p.params or "",
        new_query,
        "",  # drop fragment — reader view doesn't need anchors on initial load
    ))
    return canonical


# ---------------------------------------------------------------------------
# Hostile-domain short-circuit
#
# Sites that reliably return nothing useful from any server-side fetch
# (login walls, anti-bot challenges, heavy JS SPAs that only the browser
# can render). Rather than spend 15 seconds on a doomed fetch and then
# show a generic error, we detect these up-front and return a structured
# "unsupported" payload so the frontend can render a clean card with an
# open-in-browser CTA. No third-party mirror scraping, no Wayback auto-
# fallback — just honest UX when a site isn't viewable through us.
# ---------------------------------------------------------------------------
_HOSTILE_DOMAINS: frozenset[str] = frozenset({
    # Social — login walls
    "facebook.com", "m.facebook.com", "web.facebook.com",
    "instagram.com", "m.instagram.com",
    "linkedin.com", "m.linkedin.com",
    "pinterest.com", "pinterest.ca", "pinterest.co.uk", "pinterest.fr",
    "quora.com",
    "threads.net",
    "nextdoor.com",
    # Shopping — bot walls
    "amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.it",
    "amazon.es", "amazon.ca", "amazon.com.au", "amazon.co.jp", "amazon.in",
    "walmart.com", "target.com", "bestbuy.com", "homedepot.com", "lowes.com",
    "costco.com", "samsclub.com", "wayfair.com", "ikea.com",
    # Streaming — player, not readable content
    "netflix.com", "hulu.com", "disneyplus.com", "max.com", "hbomax.com",
    "primevideo.com", "peacocktv.com", "paramountplus.com",
    "crunchyroll.com", "funimation.com",
    # Music players — embed available separately but the page itself
    # isn't readable
    "music.apple.com", "music.amazon.com",
    "music.youtube.com",
    # Maps / location apps
    "maps.google.com", "www.google.com/maps",
    "maps.apple.com",
    # Productivity — require auth
    "docs.google.com", "drive.google.com", "sheets.google.com",
    "slides.google.com", "meet.google.com", "calendar.google.com",
    "outlook.com", "office.com", "sharepoint.com",
    "notion.so", "www.notion.so",  # public pages work but need JS
    "figma.com",
    "airtable.com",
    "miro.com",
    # Messaging
    "discord.com", "discordapp.com", "slack.com", "teams.microsoft.com",
    "whatsapp.com", "web.whatsapp.com",
    # Banking / finance — never want to render these in a reader
    "chase.com", "bankofamerica.com", "wellsfargo.com", "citi.com",
    "paypal.com", "venmo.com", "cashapp.com",
    "robinhood.com", "coinbase.com",
})


def _is_hostile_domain(hostname: str) -> bool:
    if not hostname:
        return False
    h = hostname.lower().removeprefix("www.")
    if h in _HOSTILE_DOMAINS:
        return True
    # Walk up subdomains for wildcard match — api.facebook.com still
    # hostile even though we only listed facebook.com.
    parts = h.split(".")
    for i in range(1, len(parts) - 1):
        if ".".join(parts[i:]) in _HOSTILE_DOMAINS:
            return True
    return False


def _unsupported_site_response(url: str, *, reason: str = "") -> dict:
    """Build the structured payload the frontend renders as the clean
    open-in-browser card. Includes favicon + hostname + optional reason
    text. No content, no error — just a "we can't show this here" hint.
    """
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    pretty_host = hostname.removeprefix("www.") or "this site"
    return {
        "unsupported": True,
        "url": url,
        "title": pretty_host,
        "text": "",
        "html": "",
        "sitename": pretty_host,
        "hostname": pretty_host,
        "reason": reason or "",
        "favicon_url": (
            f"/api/browse/image?url=https%3A%2F%2Fwww.google.com"
            f"%2Fs2%2Ffavicons%3Fdomain%3D{hostname}%26sz%3D64"
        ),
        "source": "unsupported",
        "page_type": "unsupported",
    }


@router.get("/fetch")
async def browse_fetch(request: Request, url: str = "") -> JSONResponse:
    """Fetch and extract article content with multi-layer fallback chain.

    Layer 0: Structured data shortcuts (JSON-LD, AMP, RSS) — zero-cost
    Layer 1: Chrome TLS fetch + trafilatura extraction
    Layer 2: Wayback Machine archived snapshot
    """
    if not url.strip():
        return JSONResponse({"error": "URL parameter is required"}, status_code=400)

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return JSONResponse({"error": "URL must start with http:// or https://"}, status_code=400)

    # Canonicalize once at the top so every downstream path (intercepts,
    # trafilatura, reputation scoring) sees the same URL. Strips trackers
    # and collapses domain aliases.
    url = _canonicalize_url(url)

    # Piece 8' — emit a surface event so the companion runtime can
    # observe what the user is paying attention to. Fire-and-forget on
    # the bus; never raises, never blocks. Carries only the URL + user_id
    # so the recent deque can hold many of these cheaply. The companion
    # never auto-acts on a single browse event; it feeds aggregated
    # signal into perception + initiative scoring.
    try:
        _u = request.scope.get("user")
        _user_id = getattr(_u, "id", "") if _u else ""
        # `client` = the auth session's source (web / android /
        # cast_receiver). The topical aggregator drops attention from
        # clients outside `companion_attention_sources`, and the
        # provenance chip shows it — a TV logged in as you shouldn't
        # write your attention stream (2026-06-08 incident).
        _client = getattr(_u, "session_source", "web") if _u else "web"
        from augmentum.companion_runtime.bus import emit_safe
        await emit_safe(
            request.app.state,
            "surface.browse.opened",
            {"url": url, "user_id": _user_id, "client": _client},
        )
    except Exception as exc:
        # Observability is best-effort; never break the user's read.
        log.debug("browse_open_signal_failed", error=str(exc))

    # Short-circuit known-hostile domains so users don't wait 15s for a
    # guaranteed failure. Returns a structured payload with a clear
    # "open in browser" prompt.
    _host = (urlparse(url).hostname or "").lower()
    if _is_hostile_domain(_host):
        return JSONResponse(_unsupported_site_response(
            url,
            reason="This site needs a real browser session — it doesn't render useful content to a reader.",
        ))

    # ===================================================================
    # Wikipedia/Wikimedia intercept — use REST API for clean HTML
    # Wikimedia sites serve heavy JS-dependent pages that fail on fetch.
    # Their REST API returns pre-rendered, clean article HTML.
    # ===================================================================
    _wiki_resp = await _try_wikipedia_api(url, request)
    if _wiki_resp:
        return JSONResponse(_wiki_resp)

    # ===================================================================
    # Reddit intercept — use the public .json endpoints.
    # Reddit's HTML now serves a "prove you're human" wall to most
    # server-side fetches; its JSON endpoints are unauthenticated and
    # return full post + comments data. Handles post permalinks AND
    # subreddit listings. Returns None for URLs we don't render
    # (user pages, search, wiki) and falls through to the generic fetch.
    # ===================================================================
    _reddit_resp = await _try_reddit_api(url, request)
    if _reddit_resp:
        return JSONResponse(_reddit_resp)

    # ===================================================================
    # Hacker News intercept — Firebase API for clean story + comments.
    # ===================================================================
    _hn_resp = await _try_hn_api(url, request)
    if _hn_resp:
        return JSONResponse(_hn_resp)

    # ===================================================================
    # GitHub — dispatch in specificity order: blob file → gist → repo.
    # A blob URL like /owner/repo/blob/main/src/foo.py must match the
    # blob intercept before falling to the repo intercept (which only
    # matches /owner/repo exactly anyway, so no overlap — but keep the
    # order consistent for future pattern additions).
    # ===================================================================
    _gh_blob_resp = await _try_github_blob_api(url, request)
    if _gh_blob_resp:
        return JSONResponse(_gh_blob_resp)

    _gh_gist_resp = await _try_github_gist_api(url, request)
    if _gh_gist_resp:
        return JSONResponse(_gh_gist_resp)

    _gh_resp = await _try_github_api(url, request)
    if _gh_resp:
        return JSONResponse(_gh_resp)

    # ===================================================================
    # Hugging Face — model / dataset / space / blob views.
    # Uses /api/{models,datasets,spaces}/{repo} metadata + /raw README.
    # Paper URLs are canonicalized to arxiv.org earlier in this function.
    # ===================================================================
    _hf_resp = await _try_hf_api(url, request)
    if _hf_resp:
        return JSONResponse(_hf_resp)

    # ===================================================================
    # arXiv intercept — clean abstract + authors + DOI + PDF/HTML links
    # via the export.arxiv.org Atom API. Runs AFTER HF (HF papers
    # canonicalize to arxiv.org/abs upstream).
    # ===================================================================
    _arxiv_resp = await _try_arxiv_api(url, request)
    if _arxiv_resp:
        return JSONResponse(_arxiv_resp)

    # ===================================================================
    # Stack Exchange intercept — question + top answers. Covers
    # Stack Overflow, *.stackexchange.com, superuser, serverfault, etc.
    # ===================================================================
    _se_resp = await _try_stackexchange_api(url, request)
    if _se_resp:
        return JSONResponse(_se_resp)

    # ===================================================================
    # Discourse forum intercept — most public Discourse instances expose
    # /t/<id>.json unauthenticated. Probe-based: a fast 404 on
    # non-Discourse URLs that happen to match /t/<digits>.
    # ===================================================================
    _discourse_resp = await _try_discourse_api(url, request)
    if _discourse_resp:
        return JSONResponse(_discourse_resp)

    # ===================================================================
    # YouTube URL intercept — render embedded player instead of fetching
    # YouTube pages are JS-heavy SPAs that return garbage when fetched.
    # ===================================================================
    from augmentum.tools.youtube import _extract_video_id
    yt_video_id = _extract_video_id(url)
    if yt_video_id:
        # Fetch metadata via oEmbed (free, no API key)
        yt_title = ""
        yt_channel = ""
        yt_thumbnail = f"https://img.youtube.com/vi/{yt_video_id}/hqdefault.jpg"
        http_client = getattr(request.app.state, "http_client", None)
        if http_client:
            try:
                oembed_resp = await http_client.get(
                    "https://www.youtube.com/oembed",
                    params={"url": url, "format": "json"},
                    timeout=8.0,
                )
                if oembed_resp.status_code == 200:
                    oembed = oembed_resp.json()
                    yt_title = oembed.get("title", "")
                    yt_channel = oembed.get("author_name", "")
                    yt_thumbnail = oembed.get("thumbnail_url", yt_thumbnail)
            except Exception as exc:
                log.debug("youtube_oembed_article_fetch_failed", url=url[:120], error=str(exc))

        embed_html = (
            f'<div class="yt-article-embed">'
            f'<iframe src="https://www.youtube.com/embed/{yt_video_id}?rel=0&modestbranding=1" '
            f'allowfullscreen frameborder="0" loading="lazy" '
            f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture">'
            f'</iframe></div>'
        )

        # Fetch related videos for rabbit-hole browsing in browse tab
        related_html = ""
        if yt_title and http_client:
            try:
                from augmentum.tools.youtube import _humanize_views, _humanize_date
                query = re.sub(r"[^\w\s]", "", yt_title).strip().split()[:6]
                search_resp = await http_client.get(
                    f"{settings.searxng_base_url.rstrip('/')}/search",
                    params={"q": " ".join(query), "format": "json", "categories": "videos"},
                    timeout=10.0,
                )
                if search_resp.status_code == 200:
                    search_data = search_resp.json()
                    # Quality filter: reject junk, non-English, noise
                    _raw_related = search_data.get("results", [])
                    try:
                        from augmentum.discovery.quality import filter_for_video_ui
                        _raw_related = filter_for_video_ui(
                            _raw_related, context="related",
                            exclude_ids={yt_video_id},
                        )
                    except Exception:
                        # Quality filter failure leaves raw results — debug
                        # log so filter regressions don't silently degrade
                        # related-video relevance.
                        log.debug("video_related_filter_failed", exc_info=True)
                    related_cards = []
                    seen = {yt_video_id}
                    for r in _raw_related:
                        vid = _extract_video_id(r.get("url", ""))
                        if not vid or vid in seen:
                            continue
                        seen.add(vid)
                        thumb = f"/api/browse/image?url={quote_plus(f'https://img.youtube.com/vi/{vid}/mqdefault.jpg')}"
                        browse_url = f"https://www.youtube.com/watch?v={vid}"
                        dur = r.get("length") or r.get("duration", "")
                        dur_badge = f'<span class="yt-card-duration">{_esc(str(dur))}</span>' if dur else ""
                        onerr = "this.style.display='none'"
                        related_cards.append(
                            f'<a class="browse-yt-related-card" href="/api/browse/fetch?url={quote_plus(browse_url)}" '
                            f'data-browse-url="{_esc(browse_url)}">'
                            f'<div class="yt-related-thumb">'
                            f'<img src="{_esc(thumb)}" alt="{_esc(r.get("title", ""))}" loading="lazy" onerror="{onerr}">'
                            f'{dur_badge}'
                            f'</div>'
                            f'<div class="yt-related-info">'
                            f'<div class="yt-related-title">{_esc(r.get("title", ""))}</div>'
                            f'<div class="yt-related-channel">{_esc(r.get("author", ""))}</div>'
                            f'</div></a>'
                        )
                        if len(related_cards) >= 5:
                            break
                    if related_cards:
                        related_html = (
                            f'<details class="browse-yt-related">'
                            f'<summary class="browse-yt-related-toggle">More Videos</summary>'
                            f'<div class="yt-related-grid">{"".join(related_cards)}</div>'
                            f'</details>'
                        )
            except Exception as exc:
                log.debug("youtube_related_videos_build_failed", error=str(exc))

        # Also try to get transcript for text content + readable HTML
        transcript_text = ""
        transcript_html = ""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            ytt = YouTubeTranscriptApi()
            raw_t = await asyncio.to_thread(ytt.fetch, yt_video_id, languages=["en"])
            segments = [{"start": s.start, "text": s.text} for s in raw_t]
            transcript_text = " ".join(s["text"] for s in segments)
            # Group into readable paragraphs (by time gap or sentence boundaries)
            if segments:
                paragraphs: list[str] = []
                current: list[str] = []
                last_start = 0.0
                for seg in segments:
                    gap = seg["start"] - last_start
                    # New paragraph on 8+ second gap or every ~5 sentences
                    if current and (gap > 8 or len(current) >= 5):
                        paragraphs.append(" ".join(current))
                        current = []
                    current.append(seg["text"])
                    last_start = seg["start"]
                if current:
                    paragraphs.append(" ".join(current))
                transcript_html = (
                    '<div class="browse-yt-transcript">'
                    '<h3 style="margin:var(--space-lg) 0 var(--space-sm);font-size:1em;color:var(--text-primary)">Transcript</h3>'
                    + "".join(f"<p>{_esc(p)}</p>" for p in paragraphs)
                    + "</div>"
                )
        except Exception as exc:
            log.debug("youtube_transcript_fetch_failed", vid=yt_video_id, error=str(exc))

        # Log YouTube video to discovery history
        try:
            _disc_store = getattr(request.app.state, "discovery_store", None)
            if _disc_store:
                from augmentum.config import settings as _settings
                if _settings.discovery_enabled:
                    await _disc_store.upsert_history(
                        url=f"https://www.youtube.com/watch?v={yt_video_id}",
                        title=yt_title or "",
                        domain="youtube.com",
                        content_type="video",
                        thumbnail=f"https://img.youtube.com/vi/{yt_video_id}/hqdefault.jpg",
                    )
        except Exception:
            log.debug(
                "browse_history_upsert_yt_failed",
                vid=yt_video_id,
                exc_info=True,
            )

        parsed = urlparse(url)
        return JSONResponse({
            "html": embed_html + related_html + transcript_html,
            "text": transcript_text or f"YouTube video: {yt_title}",
            "title": yt_title or f"YouTube Video",
            "author": yt_channel,
            "date": "",
            "sitename": "YouTube",
            "word_count": len(transcript_text.split()) if transcript_text else 0,
            "reading_time_min": max(1, len(transcript_text.split()) // 238) if transcript_text else 0,
            "url": url,
            "favicon_url": f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3Dyoutube.com%26sz%3D32",
            "source": "youtube-embed",
            "page_type": "video",
            "videos": [],
        })

    # ===================================================================
    # Direct-media intercept — URLs pointing at a raw .mp4/.mp3/.pdf/etc.
    # Runs before platform detection because these URLs don't match any
    # platform regex and the article extractor would render a blank page
    # for them. Suffix-based so it's effectively free when it misses.
    # ===================================================================
    _direct_media = await _try_direct_media(url, request)
    if _direct_media:
        return JSONResponse(_direct_media)

    # ===================================================================
    # Video platform intercepts — embed players for known video sites
    # Each uses oEmbed for metadata + platform-specific embed URL.
    # ===================================================================
    _video_embed = await _try_video_embed(url, request)
    if _video_embed:
        return JSONResponse(_video_embed)

    # ===================================================================
    # Generic video embed discovery — oEmbed + OpenGraph fallback
    # Works for any site that supports the oEmbed standard (hundreds of
    # video/audio platforms) or exposes og:video meta tags.
    # ===================================================================
    _discovered_embed = await _try_oembed_discovery(url, request)
    if _discovered_embed:
        return JSONResponse(_discovered_embed)

    source = "direct"
    extracted: str | None = None
    raw_html: str = ""
    fetch_meta: dict = {"url": url}
    jsonld_meta: dict | None = None  # metadata from JSON-LD if found

    # ===================================================================
    # Layer 1: Fetch with Chrome TLS fingerprint
    # ===================================================================
    try:
        raw_html, fetch_meta = await _fetch_with_chrome_tls(url)
    except SafeHttpError as exc:
        return JSONResponse({"error": f"Cannot access this URL: {exc}"}, status_code=403)
    except Exception as exc:
        log.warning("browse_fetch_failed", url=url, error=str(exc))
        return JSONResponse({"error": f"Fetch failed: {exc}"}, status_code=502)

    # ===================================================================
    # Pre-check: feed-shaped bodies render as feed listings before
    # _handle_file_type would dump them as pretty XML. Catches feeds at
    # /feed, /rss, /atom.xml regardless of extension or Content-Type.
    # ===================================================================
    _feed_response = _try_render_feed(url, raw_html, fetch_meta)
    if _feed_response:
        return JSONResponse(_feed_response)

    # ===================================================================
    # Pre-check: handle non-HTML file types (PDF, images, text, code, audio)
    # Each renders appropriately instead of trying to parse as a web page.
    # ===================================================================
    _file_response = _handle_file_type(url, raw_html, fetch_meta)
    if _file_response:
        return JSONResponse(_file_response)

    # ===================================================================
    # Layer 0: Try structured data shortcuts on the raw HTML
    # (Runs BEFORE trafilatura — faster and more reliable when available)
    # ===================================================================

    # 0a: JSON-LD articleBody — full text in structured data
    jsonld = await asyncio.to_thread(_extract_jsonld_article, raw_html)
    if jsonld and len(jsonld.get("text", "")) > _MIN_CONTENT_CHARS:
        extracted = jsonld["text"]
        jsonld_meta = jsonld
        source = "json-ld"
        log.info("layer0_jsonld", url=url, chars=len(extracted))

    # 0a2: Schema.org structured data (Product, Recipe, Event, etc.)
    # Renders structured content as clean HTML cards instead of garbled text.
    structured_card_html = None
    if not extracted:
        structured = await asyncio.to_thread(_extract_jsonld_structured, raw_html)
        if structured and structured.get("html"):
            structured_card_html = structured["html"]
            if not jsonld_meta:
                jsonld_meta = {"title": structured.get("title", "")}
            source = f"schema:{structured.get('type', 'structured')}"
            log.info("layer0_structured", url=url, schema_type=structured.get("type"))

    # 0b: AMP version — clean static HTML, no JS needed
    if not extracted or len(extracted) < _MIN_CONTENT_CHARS:
        amp_url = _discover_amp_url(raw_html)
        if amp_url:
            amp_url = urljoin(url, amp_url)  # resolve relative URLs
            try:
                amp_html, amp_meta = await _fetch_with_chrome_tls(amp_url, timeout=15.0)
                amp_text = await asyncio.to_thread(_extract_with_trafilatura, amp_html)
                if amp_text and len(amp_text) > len(extracted or ""):
                    raw_html = amp_html
                    extracted = amp_text
                    fetch_meta = amp_meta
                    source = "amp"
                    log.info("layer0_amp", url=url, amp_url=amp_url, chars=len(extracted))
            except Exception:
                log.debug("layer0_amp_failed", url=url, exc_info=True)

    # 0c: RSS feed — full-content feeds give us clean article text
    if not extracted or len(extracted) < _MIN_CONTENT_CHARS:
        rss_url = _discover_rss_url(raw_html, url)
        if rss_url:
            rss_url = urljoin(url, rss_url)
            rss_content = await _fetch_rss_article(rss_url, url)
            if rss_content and len(rss_content) > len(extracted or ""):
                # RSS content is often HTML — extract clean text
                rss_text = await asyncio.to_thread(_extract_with_trafilatura, rss_content)
                if not rss_text:
                    rss_text = _strip_html_tags(rss_content)
                if rss_text and len(rss_text) > len(extracted or ""):
                    extracted = rss_text
                    source = "rss"
                    log.info("layer0_rss", url=url, chars=len(extracted))

    # ===================================================================
    # Layer 1 continued: trafilatura extraction (if shortcuts didn't work)
    # ===================================================================
    if not extracted or len(extracted) < _MIN_CONTENT_CHARS:
        try:
            traf = await asyncio.to_thread(_extract_with_trafilatura, raw_html)
            if traf and len(traf) > len(extracted or ""):
                extracted = traf
                source = "direct"
        except Exception:
            log.debug("browse_trafilatura_failed", url=url, exc_info=True)

    if not extracted:
        extracted = _strip_html_tags(raw_html)

    # Check if what we got is actually junk (error pages, JS walls, etc.)
    junk_detected = False
    if extracted and _is_junk_content(extracted):
        log.info("browse_junk_detected", url=url, chars=len(extracted),
                 preview=extracted[:80])
        junk_detected = True
        extracted = ""  # force fallbacks

    # ===================================================================
    # Layer 1.5: RSS / Atom feed fallback
    # Dynamic blogs and publications (Substack, Ghost, WordPress, Tumblr,
    # many news sites) often ship a full-body feed even when their HTML
    # is too JS-heavy for trafilatura. Try to find and use the feed
    # before falling back to Wayback — feed content is canonical (the
    # author's own publish), Wayback is a snapshot.
    # ===================================================================
    rss_rehydrated = False
    if (not extracted or len(extracted) < _MIN_CONTENT_CHARS) and raw_html:
        rss_result = await _try_rss_fallback(url, raw_html)
        if rss_result:
            # Keep the original metadata (title from page if extracted,
            # otherwise use RSS's; author/date/sitename from page; body
            # from RSS).
            if rss_result.get("text") and len(rss_result["text"]) >= _MIN_CONTENT_CHARS:
                extracted = rss_result["text"]
                # Expose the RSS HTML body upstream as raw_html so the
                # normal render path picks it up.
                raw_html = rss_result["html"]
                source = "rss-feed"
                rss_rehydrated = True
                # RSS content is canonical, so clear junk flag if it was set.
                junk_detected = False

    # ===================================================================
    # Layer 2: Wayback Machine fallback (if still thin or junk)
    # ===================================================================
    if not extracted or len(extracted) < _MIN_CONTENT_CHARS:
        log.info("browse_thin_content", url=url, chars=len(extracted or ""))
        wb_result = await _fetch_wayback(url)
        if wb_result:
            wb_html, wb_meta = wb_result
            wb_extracted: str | None = None
            try:
                wb_extracted = await asyncio.to_thread(
                    _extract_with_trafilatura, wb_html
                )
            except Exception:
                log.debug("wayback_extraction_failed", exc_info=True)

            if wb_extracted and len(wb_extracted) > len(extracted or ""):
                raw_html = wb_html
                extracted = wb_extracted
                fetch_meta = wb_meta
                source = "wayback"
                log.info(
                    "browse_wayback_improved",
                    url=url,
                    chars=len(extracted),
                    timestamp=wb_meta.get("wayback_timestamp", ""),
                )

    # ===================================================================
    # Extract metadata + rich HTML for rendering
    # ===================================================================
    metadata = await asyncio.to_thread(_extract_metadata_with_trafilatura, raw_html)

    # Prefer JSON-LD metadata if we got it (more reliable than heuristics)
    if jsonld_meta:
        metadata["title"] = jsonld_meta.get("title") or metadata.get("title", "")
        metadata["author"] = jsonld_meta.get("author") or metadata.get("author", "")
        metadata["date"] = jsonld_meta.get("date") or metadata.get("date", "")
        metadata["sitename"] = jsonld_meta.get("sitename") or metadata.get("sitename", "")

    title = metadata.get("title") or _extract_title(raw_html) or url

    # Build rich HTML for display — trafilatura strips images,
    # so we extract the main content area ourselves and clean it.
    html_content = await asyncio.to_thread(_extract_rich_html, raw_html, url)

    # If we extracted structured data (product, recipe, etc.), prepend the
    # clean schema card ABOVE the page extract so key info is immediately visible.
    if structured_card_html:
        html_content = structured_card_html + "\n" + (html_content or "")

    # ---------------------------------------------------------------
    # Detect embedded videos and fetch transcripts for YouTube
    # ---------------------------------------------------------------
    embedded_videos = await _detect_embedded_videos(html_content or "", request)
    video_transcript_text = ""
    if embedded_videos:
        # Append transcript text to extracted text so AI actions get full context
        parts = []
        for ev in embedded_videos:
            if ev.get("transcript"):
                parts.append(f"[Video: {ev.get('title', 'Untitled')}]\n{ev['transcript']}")
        if parts:
            video_transcript_text = "\n\n".join(parts)
            if extracted:
                extracted = extracted + "\n\n---\n\n" + video_transcript_text
            else:
                extracted = video_transcript_text

        # Enhance HTML: add "Ask about this video" buttons to embeds
        for ev in embedded_videos:
            vid = ev.get("video_id", "")
            if not vid or not ev.get("transcript"):
                continue
            ask_btn = (
                f'<div class="browse-video-ask" data-video-id="{_esc(vid)}" '
                f'data-video-title="{_esc(ev.get("title", ""))}">'
                f'<button class="browse-action-pill browse-video-ask-btn">'
                f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">'
                f'<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
                f'<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
                f' Ask about this video</button></div>'
            )
            # Insert button after the iframe wrapper for this video
            embed_marker = f"youtube.com/embed/{vid}"
            embed_marker_nc = f"youtube-nocookie.com/embed/{vid}"
            if html_content and (embed_marker in html_content or embed_marker_nc in html_content):
                # Insert after the closing </iframe></div>
                for marker in (embed_marker, embed_marker_nc):
                    if marker in html_content:
                        # Find the end of the iframe wrapper div
                        idx = html_content.find(marker)
                        if idx >= 0:
                            close_idx = html_content.find("</div>", idx)
                            if close_idx >= 0:
                                insert_pos = close_idx + len("</div>")
                                html_content = html_content[:insert_pos] + ask_btn + html_content[insert_pos:]
                                break

    # Reading stats
    word_count = len(extracted.split()) if extracted else 0
    reading_time = max(1, math.ceil(word_count / 238))

    # Favicon
    parsed = urlparse(url)
    favicon_url = f"/api/browse/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3D{parsed.hostname}%26sz%3D32"

    # Build response — include error hint if content was junk and fallbacks failed
    resp: dict = {
        "html": html_content or "",
        "text": extracted or "",
        "title": title,
        "author": metadata.get("author", ""),
        "date": metadata.get("date", ""),
        "sitename": metadata.get("sitename", "") or parsed.hostname,
        "word_count": word_count,
        "reading_time_min": reading_time,
        "url": str(fetch_meta.get("url", url)),
        "favicon_url": favicon_url,
        "source": source,
        "page_type": _detect_page_type(url, source=source, has_videos=bool(embedded_videos)),
        "videos": embedded_videos if embedded_videos else [],
    }

    if junk_detected and (not extracted or len(extracted) < _MIN_CONTENT_CHARS):
        resp["error"] = (
            "This page is behind a sign-in, paywall, or JavaScript challenge. "
            "Try opening the original link directly if you have access."
        )

    # Update domain reputation based on extraction result
    is_success = bool(extracted and len(extracted) >= _MIN_CONTENT_CHARS and not junk_detected)
    is_structured = source in ("json-ld", "amp", "rss")
    await _update_reputation(
        request, url,
        success=is_success,
        junk=junk_detected,
        structured_data=is_structured,
    )

    # Log to discovery history
    try:
        _disc_store = getattr(request.app.state, "discovery_store", None)
        if _disc_store:
            from augmentum.config import settings as _settings
            if _settings.discovery_enabled:
                from urllib.parse import urlparse as _urlparse
                _final_url = str(fetch_meta.get("url", url))
                _hostname = _urlparse(_final_url).hostname or ""
                _domain = _hostname.lower().removeprefix("www.")
                await _disc_store.upsert_history(
                    url=_final_url,
                    title=title or "",
                    domain=_domain,
                    content_type="article",
                    thumbnail="",
                )
    except Exception:
        # history logging is best-effort; debug log so a DB-side
        # regression doesn't silently lose history-shelf entries.
        log.debug("browse_history_upsert_article_failed", url=url, exc_info=True)

    # JS-shell guard: when the generic extraction path returns near-empty
    # text (forecast.weather.gov, ESPN scoreboards, Yahoo Finance tickers,
    # any site whose visible content is rendered client-side after JS),
    # fall through to the same friendly "open in a real browser" card the
    # hostile-domain short-circuit uses. Without this the reader renders
    # the bare HTML shell as a wall of <script>/<div> markup, which looks
    # like raw code to the user and never contains the answer they wanted.
    #
    # Specialized intercepts (Wikipedia/Reddit/HN/GitHub/HF/StackExchange/
    # YouTube) returned earlier; this only fires for the generic path.
    if not extracted or len(extracted.strip()) < _MIN_CONTENT_CHARS:
        reason = (
            "This page renders its content after the browser runs JavaScript, "
            "so the reader couldn't extract anything useful. Open it in a "
            "real browser to view."
        )
        unsupported = _unsupported_site_response(url, reason=reason)
        # Preserve the title we already extracted — better than the bare
        # hostname the helper falls back to.
        if title:
            unsupported["title"] = title
        return JSONResponse(unsupported)

    return JSONResponse(resp)


_IMG_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"\x00\x00\x01\x00", "image/x-icon"),
)


def _sniff_image_type(body: bytes) -> str:
    """Detect image MIME via magic bytes. Handles WebP/AVIF (RIFF/ISOBMFF) too."""
    if not body:
        return ""
    for magic, mime in _IMG_MAGIC:
        if body.startswith(magic):
            return mime
    if body[:4] == b"RIFF" and body[8:12] == b"WEBP":
        return "image/webp"
    # AVIF / HEIC: ISOBMFF box at offset 4 = "ftyp" + brand
    if len(body) >= 12 and body[4:8] == b"ftyp":
        brand = body[8:12]
        if brand in (b"avif", b"avis"):
            return "image/avif"
        if brand in (b"heic", b"heix", b"mif1"):
            return "image/heic"
    if body[:5] == b"<?xml" or body[:4] == b"<svg":
        return "image/svg+xml"
    return ""


@router.get("/image")
async def browse_image(request: Request, url: str = "", ref: str = "") -> Response:
    """Proxy an image through the SSRF-safe client with binary support.

    Validates SSRF, caps size, sniffs content-type via magic bytes when
    upstream lies (application/octet-stream, text/plain, missing header).

    Sends a same-origin Referer by default so hotlink-protected CDNs
    (Medium, Substack, news sites) serve the image. Callers can override
    with ?ref=<url> to match the article origin when that's what the
    CDN's hotlink policy requires.

    Returns a real 404 on upstream failure so the frontend's `onerror`
    handler fires and hides the broken element. Only bad/SSRF-rejected
    input URLs return a transparent PNG (avoids leaking validation
    details).
    """
    if not url.strip() or not url.startswith(("http://", "https://")):
        return Response(content=_TRANSPARENT_PNG, media_type="image/png")

    # Derive Referer: caller-provided origin, else the image's own origin.
    # Same-origin Referer satisfies the common hotlink-protection pattern
    # (block cross-origin, allow same-origin) without claiming to come
    # from somewhere we're not.
    try:
        ref_candidate = ref if ref.startswith(("http://", "https://")) else url
        parsed_ref = urlparse(ref_candidate)
        referer = f"{parsed_ref.scheme}://{parsed_ref.netloc}/" if parsed_ref.netloc else ""
    except Exception:
        referer = ""

    max_size = 10_485_760  # 10 MB
    try:
        hostname = _image_client._validate_url(url)
        await _image_client._check_resolved_ips(hostname)

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Augmentum/1.0)",
            "Accept": "image/*,video/*,application/pdf;q=0.8,*/*;q=0.5",
        }
        if referer:
            headers["Referer"] = referer

        async with httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=5,
            timeout=httpx.Timeout(10.0),
            headers=headers,
        ) as client:
            response = await client.get(url)

            if response.status_code >= 400:
                log.warning(
                    "image_proxy_upstream_error",
                    url=url[:200], status=response.status_code,
                )
                return Response(status_code=404)

            body = response.content
            if len(body) > max_size:
                log.warning("image_proxy_too_large", url=url[:200], size=len(body))
                return Response(status_code=404)

            content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
            ok = (
                content_type.startswith("image/")
                or content_type.startswith("video/")
                or content_type.startswith("audio/")
                or content_type == "application/pdf"
            )
            if not ok:
                # Upstream lied / omitted header — sniff magic bytes.
                sniffed = _sniff_image_type(body)
                if sniffed:
                    content_type = sniffed
                else:
                    log.warning(
                        "image_proxy_bad_content_type",
                        url=url[:200], content_type=content_type or "<empty>",
                    )
                    return Response(status_code=404)

        return Response(
            content=body,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except Exception as exc:
        log.warning("image_proxy_failed", url=url[:200], error=str(exc))
        return Response(status_code=404)


@router.post("/ai")
async def browse_ai(request: Request) -> StreamingResponse:
    """Run an AI action on page content and stream the response."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    action = body.get("action", "")
    content = body.get("content", "")
    question = body.get("question", "")
    model = body.get("model", "")
    video_context = body.get("video_context", "")
    page_type = body.get("page_type", "article")

    if action not in _AI_PROMPTS:
        return JSONResponse(
            {"error": f"Unknown action: {action}. Valid: {', '.join(_AI_PROMPTS)}"},
            status_code=400,
        )

    if not content and not video_context:
        return JSONResponse({"error": "No content provided"}, status_code=400)

    # Combine article content + video transcripts for unified AI context
    if video_context:
        combined = content
        if combined:
            combined += "\n\n---\n\n[Embedded Video Transcripts]\n" + video_context
        else:
            combined = video_context
        content = combined

    # Cap content
    content = content[:_MAX_CONTENT_CHARS]

    system_prompt, user_message = _build_ai_messages(
        action=action,
        content=content,
        question=question,
        page_type=page_type,
    )

    # Resolve backend
    provider_registry = getattr(request.app.state, "provider_registry", None)
    if not provider_registry:
        return JSONResponse({"error": "No LLM backend available"}, status_code=503)

    try:
        backend, resolved_model = await provider_registry.resolve_model_for_role(
            "utility",
            override=model,
            settings=settings,
        )
    except Exception:
        log.warning("browse_ai_model_resolve_failed", model=model)
        try:
            backend, resolved_model = await provider_registry.resolve_backend_with_fabric("")
        except Exception:
            backend, resolved_model = None, ""

    if not backend:
        return JSONResponse({"error": "No LLM backend available"}, status_code=503)

    chat_request = InternalChatRequest(
        model=resolved_model,
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ],
        stream=True,
        temperature=0.3,
        max_tokens=2048,
    )

    async def _sse_stream():
        try:
            async for chunk in backend.chat_stream(chat_request):
                if chunk.content_delta:
                    payload = json.dumps({"delta": chunk.content_delta})
                    yield f"data: {payload}\n\n"
                if chunk.done:
                    yield "data: [DONE]\n\n"
                    return
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("browse_ai_stream_error", error=str(exc))
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/save")
async def browse_save(request: Request) -> JSONResponse:
    """Save page content to the document store for RAG."""
    user = request.scope.get("user")
    user_id = user.id if user else ""
    if not user_id:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    url = body.get("url", "")
    title = body.get("title", "")
    content = body.get("content", "")

    if not content:
        return JSONResponse({"error": "No content to save"}, status_code=400)

    store = getattr(request.app.state, "document_store", None)
    if not store:
        return JSONResponse({"error": "Document store not enabled"}, status_code=503)

    # Create a text document from the page content
    filename = f"browse_{title[:60] or 'page'}.txt".replace("/", "_").replace("\\", "_")
    header = f"Title: {title}\nURL: {url}\n\n"
    data = (header + content).encode("utf-8")

    try:
        result = await store.ingest(data, filename, "text/plain", user_id=user_id)
        # Boost domain reputation — user found this content valuable
        if url:
            await _update_reputation(request, url, success=True, user_action=True)
        return JSONResponse(result, status_code=201)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        log.error("browse_save_failed", url=url, exc_info=True)
        return JSONResponse({"error": "Failed to save document"}, status_code=500)


# ---------------------------------------------------------------------------
# Live Extraction Test
# ---------------------------------------------------------------------------

_DEFAULT_TEST_QUERIES = [
    ("general", "artificial intelligence latest research"),
    ("news", "climate change policy 2026"),
    ("science", "CRISPR gene therapy results"),
    ("it", "rust programming language"),
    ("general", "best Python web frameworks"),
]


@router.get("/test")
async def browse_extraction_test(
    request: Request,
    q: str = "",
    categories: str = "",
    top: int = 5,
) -> StreamingResponse:
    """Live extraction test — searches, fetches top results, reports stats.

    Streams NDJSON so the frontend (or curl) can display results in realtime.
    Each line is a JSON object: either a "search" event, a "result" event,
    or a final "summary" event.

    Usage:
      GET /api/browse/test                    — run default queries
      GET /api/browse/test?q=quantum+computing — single custom query
      GET /api/browse/test?top=3              — top 3 per query
    """
    import time

    queries: list[tuple[str, str]]
    if q:
        queries = [(categories or "general", q)]
    else:
        queries = list(_DEFAULT_TEST_QUERIES)

    http_client = getattr(request.app.state, "http_client", None)
    base_url = settings.searxng_base_url

    async def _stream():
        totals = {
            "tested": 0, "success": 0, "thin": 0, "failed": 0,
            "sources": {"direct": 0, "json-ld": 0, "amp": 0, "rss": 0, "wayback": 0},
            "avg_chars": 0, "total_chars": 0, "avg_time_ms": 0, "total_time_ms": 0,
        }

        for cat, query in queries:
            yield json.dumps({
                "event": "search",
                "query": query,
                "category": cat,
            }) + "\n"

            # Search
            results = []
            if http_client:
                try:
                    resp = await http_client.get(
                        f"{base_url}/search",
                        params={"q": query, "format": "json", "categories": cat},
                        timeout=15.0,
                    )
                    if resp.status_code == 200:
                        raw = resp.json().get("results", [])
                        seen: set[str] = set()
                        for r in raw:
                            u = r.get("url", "")
                            if u and u not in seen:
                                seen.add(u)
                                results.append({"title": r.get("title", ""), "url": u})
                            if len(results) >= top:
                                break
                except Exception as exc:
                    yield json.dumps({
                        "event": "error",
                        "message": f"Search failed: {exc}",
                    }) + "\n"
                    continue

            if not results:
                yield json.dumps({
                    "event": "error",
                    "message": f"No results for '{query}'",
                }) + "\n"
                continue

            # Fetch each result
            for r in results:
                totals["tested"] += 1
                t0 = time.monotonic()
                result_data: dict = {
                    "event": "result",
                    "url": r["url"],
                    "title": r["title"],
                }

                try:
                    # Use the internal fetch logic directly
                    raw_html, fetch_meta = await _fetch_with_chrome_tls(r["url"], timeout=15.0)

                    source = "direct"
                    extracted: str | None = None

                    # Layer 0a: JSON-LD
                    jsonld = await asyncio.to_thread(_extract_jsonld_article, raw_html)
                    if jsonld and len(jsonld.get("text", "")) > _MIN_CONTENT_CHARS:
                        extracted = jsonld["text"]
                        source = "json-ld"

                    # Layer 0b: AMP
                    if not extracted or len(extracted) < _MIN_CONTENT_CHARS:
                        amp_url = _discover_amp_url(raw_html)
                        if amp_url:
                            amp_url = urljoin(r["url"], amp_url)
                            try:
                                amp_html, _ = await _fetch_with_chrome_tls(amp_url, timeout=10.0)
                                amp_text = await asyncio.to_thread(
                                    _extract_with_trafilatura, amp_html
                                )
                                if amp_text and len(amp_text) > len(extracted or ""):
                                    extracted = amp_text
                                    source = "amp"
                            except Exception:
                                log.debug("amp_extraction_failed", exc_info=True)

                    # Layer 0c: RSS
                    if not extracted or len(extracted) < _MIN_CONTENT_CHARS:
                        rss_url = _discover_rss_url(raw_html, r["url"])
                        if rss_url:
                            rss_url = urljoin(r["url"], rss_url)
                            rss_content = await _fetch_rss_article(rss_url, r["url"])
                            if rss_content and len(rss_content) > len(extracted or ""):
                                rss_text = _strip_html_tags(rss_content)
                                if rss_text and len(rss_text) > len(extracted or ""):
                                    extracted = rss_text
                                    source = "rss"

                    # Layer 1: trafilatura
                    if not extracted or len(extracted) < _MIN_CONTENT_CHARS:
                        traf = await asyncio.to_thread(_extract_with_trafilatura, raw_html)
                        if traf and len(traf) > len(extracted or ""):
                            extracted = traf
                            source = "direct"

                    if not extracted:
                        extracted = _strip_html_tags(raw_html)

                    # Filter junk/binary
                    is_binary = _is_binary_response(raw_html)
                    is_junk = bool(extracted and _is_junk_content(extracted))
                    if is_binary or is_junk:
                        extracted = ""

                    # Layer 2: Wayback
                    if not extracted or len(extracted) < _MIN_CONTENT_CHARS:
                        wb = await _fetch_wayback(r["url"])
                        if wb:
                            wb_text = await asyncio.to_thread(
                                _extract_with_trafilatura, wb[0]
                            )
                            if wb_text and len(wb_text) > len(extracted or ""):
                                extracted = wb_text
                                source = "wayback"

                    elapsed = int((time.monotonic() - t0) * 1000)
                    chars = len(extracted or "")

                    result_data["source"] = source
                    result_data["chars"] = chars
                    result_data["time_ms"] = elapsed
                    result_data["preview"] = (extracted or "")[:200]
                    result_data["has_amp"] = bool(_discover_amp_url(raw_html))
                    result_data["has_rss"] = bool(_discover_rss_url(raw_html, r["url"]))
                    result_data["has_jsonld"] = jsonld is not None
                    result_data["is_binary"] = is_binary
                    result_data["is_junk"] = is_junk

                    if is_binary:
                        result_data["status"] = "binary"
                        totals["failed"] += 1
                    elif is_junk and chars < _MIN_CONTENT_CHARS:
                        result_data["status"] = "junk"
                        totals["failed"] += 1
                    elif chars >= _MIN_CONTENT_CHARS:
                        result_data["status"] = "ok"
                        totals["success"] += 1
                    else:
                        result_data["status"] = "thin"
                        totals["thin"] += 1

                    totals["sources"][source] = totals["sources"].get(source, 0) + 1
                    totals["total_chars"] += chars
                    totals["total_time_ms"] += elapsed

                except Exception as exc:
                    elapsed = int((time.monotonic() - t0) * 1000)
                    result_data["status"] = "error"
                    result_data["error"] = str(exc)
                    result_data["time_ms"] = elapsed
                    totals["failed"] += 1
                    totals["total_time_ms"] += elapsed

                yield json.dumps(result_data) + "\n"

        # Summary
        n = totals["tested"] or 1
        totals["avg_chars"] = totals["total_chars"] // n
        totals["avg_time_ms"] = totals["total_time_ms"] // n
        totals["success_rate"] = f"{totals['success'] / n * 100:.0f}%"

        yield json.dumps({"event": "summary", **totals}) + "\n"

    return StreamingResponse(
        _stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
