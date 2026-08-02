"""Coder-specific web tools — documentation search and page fetching.

Tailored for coding workflows: searches prioritize official documentation,
API references, and trusted programming sites. Results are re-ranked
to surface quality technical content over blog spam.

Uses the existing WebSearchTool and WebFetchTool infrastructure but with
coder-specific configuration: preferred doc domains, language-aware
query expansion, and content extraction optimized for code examples.

2026-05-31: ``doc_search`` now fans out internally to TWO parallel
SearXNG queries — one with the doc-suffix enhancement, one with the
raw query — and merges results. This gives a single tool the model
can call while broadening coverage for "compare X and Y" / "what's
the consensus on Z" style asks that the doc-only path missed.
"""
from __future__ import annotations

import asyncio
import re

from augmentum.coder.tools import _CoderTool, _truncate
from augmentum.tools.base import ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Word-boundary check used to decide whether to auto-append "documentation"
# to the search query. Substring matching ("'api' in 'fastapi'") was a
# false-positive trap that skipped the suffix on every fastapi/scrapy/etc.
# query — those are exactly the queries that benefit most from it.
_HAS_DOC_KEYWORD = re.compile(
    r"\b(docs?|documentation|api|reference|how to)\b",
    re.IGNORECASE,
)

# ── Deterministic query distillation ─────────────────────────────────
# Small models paste raw failure artifacts into doc_search: whole
# tracebacks, absolute workspace paths, hex addresses, line numbers.
# Search engines answer keyword queries, not stack dumps — the good
# result never enters the pool and no amount of reranking can recover
# it. This is a model-free translation layer (works identically for a
# 2B and a frontier model; no classifier-slot dependency): extract the
# exception line from traceback-shaped input, strip filesystem/memory
# noise, and cap runaway length. The original query is preserved in the
# tool output so the model sees what actually ran.

_EXC_LINE = re.compile(
    r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt|Exit))\s*:\s*(.*)$",
    re.MULTILINE,
)
_TRACEBACK_MARK = re.compile(r"Traceback \(most recent call last\)|^\s*File \"", re.MULTILINE)
_NOISE = re.compile(
    r"(?:[A-Za-z]:)?(?:/|\\)[\w.\\/-]{8,}"   # absolute-ish paths
    r"|0x[0-9a-fA-F]{4,}"                     # memory addresses
    r"|\bline \d+\b"                           # traceback line numbers
    r"|\bat \d+:\d+\b",                        # js-style positions
)
_ERROR_SHAPED = re.compile(r"\b\w+(?:Error|Exception)\b|\berror\b", re.IGNORECASE)
_MAX_QUERY_WORDS = 12


def _distill_query(query: str) -> tuple[str, bool]:
    """Reduce a raw model query to search-engine-shaped keywords.

    Returns ``(distilled, changed)``. Conservative: returns the input
    untouched unless it is traceback-shaped, noise-laden, or longer
    than a search engine can use — short clean queries pass through.
    """
    q = query.strip()
    # Traceback pasted wholesale → the exception line is the query.
    if _TRACEBACK_MARK.search(q):
        last = None
        for last in _EXC_LINE.finditer(q):  # noqa: B007 — used after loop
            pass  # keep the LAST exception line (the actual failure)
        if last is not None:
            exc_type, exc_msg = last.group(1), _NOISE.sub(" ", last.group(2))
            q = f"{exc_type} {exc_msg}".strip()
    cleaned = _NOISE.sub(" ", q)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split(" ")
    if len(words) > _MAX_QUERY_WORDS:
        cleaned = " ".join(words[:_MAX_QUERY_WORDS])
    cleaned = cleaned or query.strip()
    return cleaned, cleaned != query.strip()


def _query_tokens(query: str) -> frozenset[str]:
    """Lowercase word tokens for repeat/overlap comparison."""
    return frozenset(
        t for t in re.split(r"[^\w.]+", query.lower()) if len(t) > 2
    )

# Official documentation domains — highest trust for coding queries
_DOC_DOMAINS = {
    # Python
    "docs.python.org", "pypi.org", "peps.python.org",
    # JavaScript / TypeScript
    "developer.mozilla.org", "nodejs.org", "typescriptlang.org", "npmjs.com",
    # Rust
    "doc.rust-lang.org", "docs.rs", "crates.io",
    # Go
    "go.dev", "pkg.go.dev",
    # Java / Kotlin
    "docs.oracle.com", "kotlinlang.org",
    # C / C++
    "cppreference.com", "cplusplus.com",
    # C# / .NET
    "learn.microsoft.com", "dotnet.microsoft.com",
    # Ruby
    "ruby-doc.org", "rubygems.org",
    # PHP
    "php.net",
    # Swift
    "developer.apple.com",
    # General
    "devdocs.io", "stackoverflow.com", "github.com",
    # Frameworks
    "flask.palletsprojects.com", "fastapi.tiangolo.com", "djangoproject.com",
    "expressjs.com", "react.dev", "vuejs.org", "nextjs.org",
    "tailwindcss.com", "getbootstrap.com",
    # Databases
    "postgresql.org", "dev.mysql.com", "redis.io", "mongodb.com",
    # DevOps
    "docs.docker.com", "kubernetes.io", "nginx.org",
}

# Domains to avoid (SEO spam, low-quality rewrites)
_AVOID_DOMAINS = {
    "w3schools.com", "geeksforgeeks.org", "tutorialspoint.com",
    "javatpoint.com", "programiz.com",
}


def _domain_from_url(url: str) -> str:
    """Extract domain from URL."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _anchor_from_url(url: str) -> str:
    """Return the URL fragment (after #) or empty string."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).fragment or ""
    except Exception:
        return ""


def _slice_around_anchor(content: str, anchor: str) -> tuple[str, bool]:
    """Try to slice extracted content around the URL anchor's heading.

    Returns ``(sliced, hit)``. On hit, ``sliced`` starts just before the
    line containing the anchor (so the heading itself stays visible).
    On miss, returns the original content unchanged.

    Strategy: convert anchor to candidate display strings (replace -/_
    with space, split dotted paths), search case-insensitively for the
    longest one that appears in the content. Real-world anchors are
    things like ``asyncio.TaskGroup``, ``background-tasks``,
    ``creating-a-react-app`` — all of which the rendered page usually
    contains as a heading word-for-word.
    """
    if not anchor:
        return content, False

    candidates = [
        anchor,
        anchor.replace("-", " ").replace("_", " "),
        anchor.split(".")[-1],  # last segment of dotted path
    ]
    # Longest first — match the most specific candidate before falling back.
    candidates = sorted({c.strip() for c in candidates if c.strip()}, key=len, reverse=True)

    lower = content.lower()
    for cand in candidates:
        idx = lower.find(cand.lower())
        if idx == -1:
            continue
        # Back up to the start of the line so the heading stays visible.
        line_start = content.rfind("\n", 0, idx) + 1
        return content[line_start:], True
    return content, False


# URL path segments that mark a result as a real doc page rather than a
# domain homepage. Anything matching gets an extra boost on top of the
# domain-trust multiplier, so /library/asyncio-task.html beats /.
_DOC_PATH_SIGNALS = (
    "/library/", "/api/", "/reference/", "/tutorial/", "/guide/",
    "/guides/", "/manual/", "/docs/", "/documentation/", "/howto/",
    "/learn/", "/handbook/",
)


def _rerank_results(results: list[dict]) -> list[dict]:
    """Re-rank search results to prioritize documentation sites."""
    for r in results:
        url = r.get("url", "")
        domain = _domain_from_url(url).lower()

        # Base score from search position
        score = r.get("_score", 1.0)

        # Boost official docs
        if domain in _DOC_DOMAINS:
            score *= 2.5
        # Boost any .dev, .io domain with "docs" in path
        elif domain.endswith((".dev", ".io")) and "/docs" in url:
            score *= 1.8
        # Boost GitHub repos (not gists)
        elif "github.com" in domain and "/blob/" not in url and "/gist" not in url:
            score *= 1.3
        # Penalize low-quality sites
        elif domain in _AVOID_DOMAINS:
            score *= 0.3

        # Path-depth boost — a result whose URL path looks like a real
        # doc page (/library/foo.html, /tutorial/bar/) outranks the
        # domain's homepage (/) when both share the same trust tier.
        url_lower = url.lower()
        if any(seg in url_lower for seg in _DOC_PATH_SIGNALS):
            score *= 1.5

        r["_score"] = score

    results.sort(key=lambda r: r.get("_score", 0), reverse=True)
    return results


# ── Engine pool ───────────────────────────────────────────────────────
# One reliable pool for EVERY query — no per-query intent routing. We
# deliberately do NOT try to guess "is this a research / web-platform /
# repo query" from the words: keyword rules encode one author's phrasing
# habits and don't survive contact with a real user base. Instead every
# query hits the same broad set and the RANKER (query-token relevance
# floor + low-value-landing drop + per-domain diversity cap in
# filter_for_docs) decides what surfaces — that's query-agnostic and works
# the same for everyone.
#
# The only per-engine judgment here is universal, not intent: two engines
# are EXCLUDED because their results defeat token-relevance ranking for
# unrelated queries (verified against 469 real doc_search queries,
# 2026-08-01) —
#   * docker hub — returns unrelated images (percona for 'llama.cpp',
#     airbyte for an fps query),
#   * MDN — returns any incidental token-match (WebGL 'blend' for a
#     raymarching query, 'Percent-encoding' for RoPE) and the floor can't
#     drop it because the token legitimately overlaps.
# Real MDN/docker CONTENT still reaches web-platform/docker queries via the
# general engines (both rank high on general search for their real topics);
# we only drop the ENGINES that inject them into everything else.
#
# SearXNG silently skips any listed engine that's disabled or currently
# suspended, so this degrades gracefully across installs — a user's own
# enabled general engines still carry the query; the keyless code/research
# APIs (github, stackoverflow via stackexchange, arxiv, hackernews,
# lobste.rs) never rate-limit and are always available.
_DOC_ENGINES: tuple[str, ...] = (
    "google", "bing", "mojeek", "brave", "startpage", "wikipedia",  # general web
    "github", "stackoverflow", "hackernews", "lobste.rs",           # keyless code/discussion
)
# google is back IN the pool: with Tor egress (compose.tor.yaml) the exit IP
# rotates, so Google stops fingerprint-banning us and becomes the single
# highest-value engine again — verified 2026-08-01, it's what surfaces the
# iquilezles.org result for "inigo quilez smooth minimum sdf" that every
# other engine missed. Without Tor, Google returns empty/suspended and the
# other engines carry the query (SearXNG silently skips a dead engine).

# Named source options the MODEL can target via the ``sources`` param.
# Intent comes from the caller that HAS it — the model that knows it wants
# a paper picks ``papers`` and phrases for arXiv — not from keyword rules
# guessing intent from the query (which would encode one author's habits).
#
# The three specialist sources (papers/reference/images) are the ones kept
# OUT of the default pool because their keyword-dense results token-match
# unrelated queries and drown / mislead the ranker (arxiv was #1 for 54% of
# real queries when pooled by default; MDN floods on incidental token hits).
# Out of the default they can't flood; as an explicit ``sources`` pick they
# are one request away when genuinely wanted. Values map to SearXNG engine
# names; unknown/disabled engines are silently skipped per install.
_DOC_SOURCES: dict[str, tuple[str, ...]] = {
    "web": ("bing", "mojeek", "brave", "startpage", "wikipedia"),
    "code": ("github",),
    "qa": ("stackoverflow",),
    "discussion": ("hackernews", "lobste.rs", "reddit"),
    "papers": ("arxiv",),           # specialist — excluded from default
    "reference": ("mdn",),          # specialist — excluded from default
    "images": ("docker hub",),      # specialist — excluded from default
}


# Below this many ranked results, the SearXNG pool is treated as thin/
# degraded (its scraped engines were likely rate-limited) and — if a
# browser sidecar is available — doc_search escalates to a real-browser
# SERP render, which bypasses the bot-detection blocking the httpx path.
_DOC_THIN_THRESHOLD: int = 4


def _resolve_sources(sources: list[str] | None) -> list[str]:
    """Map model-chosen source ids to the SearXNG engine set. Unspecified
    (or all-unknown) falls back to the default reliable pool."""
    if not sources:
        return list(_DOC_ENGINES)
    engines: list[str] = []
    for s in sources:
        for e in _DOC_SOURCES.get(str(s).strip().lower(), ()):
            if e not in engines:
                engines.append(e)
    return engines or list(_DOC_ENGINES)


async def _searxng_query(
    client, base_url: str, query: str, *, engines: list[str] | None = None
) -> list[dict]:
    """One SearXNG round-trip. Returns the result list or raises.

    Extracted so ``doc_search`` can fan out to two queries in parallel
    (doc-tuned + broader-web) without duplicating the HTTP plumbing.
    When ``engines`` is given we pin that exact set (the reliable
    ``_DOC_ENGINES`` pool); otherwise fall back to the ``general,it``
    category firehose for backward compatibility.
    """
    params = {"q": query, "format": "json", "language": "en", "pageno": 1}
    if engines:
        params["engines"] = ",".join(engines)
    else:
        params["categories"] = "general,it"
    resp = await client.get(f"{base_url.rstrip('/')}/search", params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"SearXNG returned HTTP {resp.status_code}")
    data = resp.json()
    return data.get("results", []) or []


class DocSearchTool(_CoderTool):
    """Search programming documentation and trusted sources.

    Optimized for coding: prioritizes official docs, API references,
    and Stack Overflow. Avoids SEO spam and low-quality tutorial sites.
    """

    @property
    def name(self) -> str:
        return "doc_search"

    @property
    def description(self) -> str:
        return (
            "Search programming sources and re-rank to surface relevant results "
            "above spam. By default searches a broad reliable pool (general web + "
            "GitHub + Stack Overflow + tech discussion). "
            "To TARGET a specific source, pass `sources` — and phrase the query "
            "the way that source expects:\n"
            "  • papers (arXiv): paper title / author-year / method name\n"
            "  • code (GitHub): repo or topic keywords\n"
            "  • qa (Stack Overflow): the error text or 'how do I X'\n"
            "  • reference (MDN): the exact CSS/DOM/JS API name\n"
            "  • discussion (HN/lobste.rs): the project or topic name\n"
            "  • images (Docker Hub): the image name\n"
            "Only set `sources` when you specifically want that source; omit it "
            "for the general search. "
            "Query shape matters: 3-8 keywords, not whole tracebacks/paths/prose. "
            "Re-running the same query returns the same results — rephrase, target "
            "a different source, or doc_fetch a result instead."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (e.g., 'python asyncio gather', 'express middleware error handling', 'fastapi vs flask 2026 consensus')",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language context (optional, helps prioritize results). Skip when the query is comparison / news / community-opinion shaped — the broader leg picks those up unbiased.",
                    "default": "",
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["web", "code", "qa", "discussion", "papers", "reference", "images"],
                    },
                    "description": "Optional — target specific source(s) instead of the default general pool. Use 'papers' for arXiv research, 'code' for GitHub repos, 'qa' for Stack Overflow, 'reference' for MDN web-platform docs, 'discussion' for HN/lobste.rs, 'images' for Docker Hub. Omit for a general search. When you set this, phrase the query for that source.",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self, *, query: str = "", language: str = "", sources: list[str] | None = None, **_kwargs
    ) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(success=False, error="query is required", validation_error=True)

        from augmentum.config import settings

        # Distill first (model-free): traceback → exception line, strip
        # paths/addresses/line numbers, cap runaway length. Small models
        # paste failure artifacts verbatim; engines answer keywords.
        distilled, was_distilled = _distill_query(query)

        # ── Per-turn repeat-query memory ──────────────────────────────
        # Tool instances are rebuilt each turn (create_coder_tools), so this
        # cache is turn-scoped by construction. An exact repeat returns
        # the CACHED result with a corrective banner instead of a fresh
        # SearXNG round trip — re-searching identical words was a loop
        # shape none of the iteration-level detectors could see (search →
        # fetch → same search interleaves, so identical-call never trips).
        if not hasattr(self, "_seen_queries"):
            self._seen_queries: dict[frozenset, tuple[str, str]] = {}
        seen_key = _query_tokens(distilled + " " + language)
        cached = self._seen_queries.get(seen_key)
        if cached is not None:
            prev_query, prev_output = cached
            return ToolResult(
                success=True,
                output=(
                    "[Repeat search: you already ran this query this turn "
                    f"('{prev_query}') and the results are UNCHANGED — shown "
                    "again below. Do NOT search these words again. Either "
                    "doc_fetch one of the URLs below, or rephrase with "
                    "DIFFERENT terms: library name + symbol + error class.]\n\n"
                    + prev_output
                ),
                metadata={"query": query, "repeat": True},
            )
        # Soft warning for near-duplicates (trivially reworded repeats).
        near_dup_note = ""
        for prev_key, (prev_query, _prev_out) in self._seen_queries.items():
            union = seen_key | prev_key
            if union and len(seen_key & prev_key) / len(union) >= 0.75:
                near_dup_note = (
                    f"\n[Note: this is nearly the same as your earlier search "
                    f"'{prev_query}'. If these results don't answer the "
                    "question, searching variations of the same words won't "
                    "either — doc_fetch a promising URL or change the terms.]"
                )
                break

        # Build the two parallel queries. The DOC leg has the
        # "documentation" suffix + language prefix (current behavior).
        # The BROAD leg uses the raw query so coverage extends to
        # general-web results that a doc-suffix would suppress (e.g.,
        # "what's the consensus on switching from X to Y in 2026" —
        # genuine comparison/news asks that doc_search 1.0 missed).
        raw_query = distilled
        if language:
            raw_query = f"{language} {raw_query}"
        doc_query = raw_query
        # Error-shaped queries skip the suffix: "TypeError unhashable
        # documentation" suppresses the Stack Overflow / issue-tracker
        # results that actually answer it.
        if not _HAS_DOC_KEYWORD.search(doc_query) and not _ERROR_SHAPED.search(doc_query):
            doc_query += " documentation"

        searxng_url = getattr(settings, "searxng_url", "http://searxng:8080")

        # Run both legs in parallel. ``return_exceptions=True`` so a
        # partial failure (one leg blocked, the other healthy) still
        # delivers results — Cline/Codex-style "best-effort retrieval"
        # rather than all-or-nothing.
        # Default = the ``general,it`` category (engines=None → categories
        # path in _searxng_query). We use categories, NOT an explicit
        # engines= list, because through the Tor egress proxy
        # (compose.tor.yaml) SearXNG's engines= param returns nothing for
        # 2+ engines — even "google,bing" zeroes out — while categories
        # routes fine and revives Google. The reintroduced flooders
        # (MDN/docker-hub) are handled by the ranker (floor + low-value-
        # landing + per-domain diversity cap), verified junk-free. Only
        # when the model TARGETS a source do we pin engines= (single
        # specialist works through the proxy).
        routed = _resolve_sources(sources) if sources else None
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                doc_results, broad_results = await asyncio.gather(
                    _searxng_query(client, searxng_url, doc_query, engines=routed),
                    _searxng_query(client, searxng_url, raw_query, engines=routed),
                    return_exceptions=True,
                )
        except Exception as exc:
            return ToolResult(success=False, error=f"Search failed: {str(exc)[:200]}")

        # Collect non-exception results from both legs. Either may be
        # an Exception object (network blip, SearXNG 503) — log + skip.
        pool: list[dict] = []
        doc_count, broad_count = 0, 0
        if isinstance(doc_results, list):
            pool.extend(doc_results)
            doc_count = len(doc_results)
        else:
            log.debug("doc_search.doc_leg_failed", error=str(doc_results)[:160])
        if isinstance(broad_results, list):
            pool.extend(broad_results)
            broad_count = len(broad_results)
        else:
            log.debug("doc_search.broad_leg_failed", error=str(broad_results)[:160])

        # Merge by URL — the doc leg comes first, so when both return
        # the same URL we keep the doc-leg entry (preserves rank).
        # Empty/duplicate URLs are dropped.
        seen: set[str] = set()
        merged: list[dict] = []

        def _absorb(results: list[dict]) -> None:
            for r in results:
                url = (r.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                merged.append(r)

        _absorb(pool)

        # Drop AVOID domains pre-rank — they get refused by doc_fetch
        # anyway. If they survive into the result list the model picks
        # one, fails to fetch, and wastes a turn on an opaque error.
        def _rank(items: list[dict]) -> list[dict]:
            keep = [
                r for r in items
                if _domain_from_url(r.get("url", "")).lower() not in _AVOID_DOMAINS
            ]
            # ``filter_for_docs`` is the canonical scorer; fall back to the
            # local re-rank if the discovery quality module errors out.
            try:
                from augmentum.discovery.quality import filter_for_docs
                return filter_for_docs(keep, query, language=language)
            except Exception:
                for i, r in enumerate(keep):
                    r["_score"] = 1.0 / (i + 1)
                return _rerank_results(keep)

        ranked = _rank(merged)

        # Scarcity escalation: when SearXNG's scraped engines are rate-
        # limited/suspended the pool comes back thin or junk-only (the
        # recurring 2026-08 problem). A REAL Chrome fingerprint bypasses the
        # bot-detection that blocks the httpx path, so — only when the pool
        # is thin AND a browser sidecar is available — render a DuckDuckGo
        # SERP through it and re-rank the combined set. Best-effort: a
        # failure just leaves the SearXNG results as-is.
        browser_used = False
        if len(ranked) < _DOC_THIN_THRESHOLD and self._cm is not None:
            try:
                from augmentum.coder import browser_sidecar as _bs
                extra = await _bs.search_serp(self._cm, self._workspace_id, distilled)
            except Exception:
                extra = []
            if extra:
                _absorb(extra)
                ranked = _rank(merged)
                browser_used = True

        if not ranked:
            no_hit = f"No results found for '{query}'"
            # Remember the miss too — re-running a no-hit query verbatim
            # is the most common spam-loop shape.
            self._seen_queries[seen_key] = (distilled, no_hit)
            return ToolResult(
                success=True,
                output=no_hit,
                metadata={"query": query, "results": 0},
            )

        # Format top results
        _via = " · +browser" if browser_used else ""
        header = f"Search: '{query}'  ({doc_count} doc · {broad_count} broad{_via} → {len(ranked)} merged)"
        if was_distilled:
            header += f"\n(query distilled to: '{distilled}' — paths/tracebacks stripped)"
        lines = [header + "\n"]
        for i, r in enumerate(ranked[:5]):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("content", "")[:300]
            domain = _domain_from_url(url)
            doc_badge = " [official docs]" if domain in _DOC_DOMAINS else ""

            lines.append(f"{i+1}. {title}{doc_badge}")
            lines.append(f"   {url}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")

        output = _truncate("\n".join(lines)) + near_dup_note
        # Remember for the rest of the turn (exact-repeat banner above).
        self._seen_queries[seen_key] = (distilled, output)

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "query": query,
                "distilled_query": distilled if was_distilled else "",
                "results": len(ranked[:5]),
                "doc_leg_count": doc_count,
                "broad_leg_count": broad_count,
                "merged_pool_size": len(merged),
                "browser_escalated": browser_used,
            },
        )


class DocFetchTool(_CoderTool):
    """Fetch and extract content from a documentation page.

    Extracts clean text from web pages, optimized for code documentation:
    preserves code blocks, API signatures, and parameter descriptions.
    """

    @property
    def name(self) -> str:
        return "doc_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch and read content from a documentation URL. "
            "Extracts clean text with code examples preserved. "
            "Use after doc_search to read a specific page."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The documentation URL to fetch",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return per call (default: 20000)",
                    "default": 20000,
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "Character offset into the extracted page to start "
                        "from (default: 0). When a page is larger than "
                        "max_chars, the full text is saved to a workspace "
                        "file and the footer reports the next offset — call "
                        "again with that offset to read the remainder, or "
                        "grep/read the saved file directly."
                    ),
                    "default": 0,
                },
            },
            "required": ["url"],
        }

    # Upper bound on the model-supplied ``max_chars`` — one intentional window
    # of a doc page. The full extracted text is always persisted to the
    # workspace (see below), so a large page is paginated, never lost. The
    # universal clamp in handler.py is the final backstop.
    _MAX_CHARS_CEILING = 50_000

    # Raw-HTML transport cap. This bounds memory/SSRF blast radius but must NOT
    # double as a content cap: a normal single-page doc (MDN, big API refs)
    # routinely ships >500KB of raw HTML that trafilatura collapses to <50KB of
    # text. The old 500KB cap hard-FAILED those pages before extraction ever
    # ran ("Response too large"). 5MB matches the SafeHttpClient default and
    # lets the common case just work; genuinely huge bodies still get clamped.
    _MAX_RESPONSE_BYTES = 5_242_880

    async def execute(self, *, url: str = "", max_chars: int = 20000, offset: int = 0, **_kwargs) -> ToolResult:
        if not url or not url.strip():
            return ToolResult(success=False, error="url is required", validation_error=True)
        # Clamp pathological / unbounded max_chars to a sane ceiling.
        try:
            max_chars = max(1000, min(int(max_chars), self._MAX_CHARS_CEILING))
        except (TypeError, ValueError):
            max_chars = 20000
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0

        # Security: only allow http/https
        if not url.startswith(("http://", "https://")):
            return ToolResult(success=False, error="Only http:// and https:// URLs are supported")

        domain = _domain_from_url(url)
        if domain in _AVOID_DOMAINS:
            return ToolResult(
                success=False,
                error=f"Domain {domain} is not a trusted documentation source. Try an official docs site instead.",
            )

        try:
            from augmentum.utils.safe_http import SafeHttpClient

            # Read up to a generous cap, then let extraction shrink it —
            # degrade, don't hard-fail on a large-but-legitimate page.
            client = SafeHttpClient(max_response_size=self._MAX_RESPONSE_BYTES)
            html, _metadata = await client.fetch(url, timeout=15.0)

            # Try trafilatura for clean extraction
            content = None
            try:
                import trafilatura
                content = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    favor_recall=True,
                )
            except ImportError:
                pass

            if not content:
                # Fallback: strip HTML tags
                import re
                content = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r"<[^>]+>", " ", content)
                content = re.sub(r"\s+", " ", content).strip()

            if not content:
                # Degrade, don't hard-fail: a page that yields no extractable
                # text (empty body, pure-JS shell, binary) is a valid-but-thin
                # result, not an error. Matches WebFetchTool's behavior so the
                # model gets a consistent, actionable signal ("try another URL")
                # instead of an exception-shaped failure.
                return ToolResult(
                    success=True,
                    output=f"Content from {url}\n{'─' * 40}\n\n(page returned no extractable text content)",
                    metadata={"url": url, "domain": domain, "chars": 0, "total_chars": 0},
                )

            # Anchor-aware slice — when the URL ends in #fragment, jump to
            # the heading instead of returning the page intro. Saves a
            # round trip for the common "fetch the docs section the model
            # already named" case. Only when reading from the top: a paginating
            # offset call means the model is deliberately walking the body.
            anchor = _anchor_from_url(url)
            anchor_hit = False
            if anchor and offset == 0:
                content, anchor_hit = _slice_around_anchor(content, anchor)

            total_chars = len(content)

            # Persist the FULL extracted text to the workspace so a large page
            # is navigable (grep/read/offset) instead of truncated-and-lost.
            # Best-effort: a save failure just means no "read the file" hint.
            saved_path = await self._persist_extracted(url, content)

            # Window the requested slice at a paragraph boundary.
            window = content[offset:offset + max_chars]
            next_offset = offset + len(window)
            more = next_offset < total_chars
            if more:
                cut = window.rfind("\n\n")
                if cut > max_chars // 2:
                    window = window[:cut]
                    next_offset = offset + cut

            doc_badge = " [trusted source]" if domain in _DOC_DOMAINS else ""
            anchor_note = f" [anchored at #{anchor}]" if anchor_hit else ""
            range_note = f" [chars {offset}-{offset + len(window)} of {total_chars}]" if (offset or more) else ""
            header = f"Content from {url}{doc_badge}{anchor_note}{range_note}\n{'─' * 40}\n\n"

            footer = ""
            if more:
                footer = (
                    f"\n\n{'─' * 40}\n"
                    f"[{total_chars - next_offset} more chars] Read the rest with "
                    f"doc_fetch(url=..., offset={next_offset})"
                )
                if saved_path:
                    footer += f", or grep/read the full page saved at {saved_path}"

            return ToolResult(
                success=True,
                output=header + window + footer,
                metadata={
                    "url": url,
                    "domain": domain,
                    "chars": len(window),
                    "total_chars": total_chars,
                    "offset": offset,
                    "next_offset": next_offset if more else None,
                    "has_more": more,
                    "saved_path": saved_path,
                    "anchor": anchor,
                    "anchor_hit": anchor_hit,
                },
            )

        except Exception as exc:
            return ToolResult(success=False, error=f"Fetch failed: {str(exc)[:200]}")

    async def _persist_extracted(self, url: str, content: str) -> str:
        """Write the full extracted page to a stable workspace path.

        Returns the path on success, or "" if no workspace/executor is
        available or the write fails (best-effort — a save miss only costs
        the "read the saved file" hint, never the fetch itself).
        """
        executor = getattr(self, "_executor", None)
        if executor is None:
            return ""
        import hashlib

        digest = hashlib.sha1(url.encode("utf-8", "replace")).hexdigest()[:16]
        path = f"/workspace/.augmentum/fetch/{digest}.txt"
        try:
            await executor.run_command(
                ["mkdir", "-p", "/workspace/.augmentum/fetch"], timeout=5.0
            )
            await executor.write_file(path, f"Source: {url}\n\n{content}")
        except Exception:
            log.warning("doc_fetch.persist_failed", url=url[:120], exc_info=True)
            return ""
        return path
