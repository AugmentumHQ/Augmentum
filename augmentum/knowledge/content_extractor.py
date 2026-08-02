"""Agnostic content extractor for the Cardsmith pipeline.

Given a URL (or a wiki path + base host), fetch the document and return
structured context the Cardsmith prompt can reference. Supports three tiers
that degrade gracefully:

  Tier 1 — MediaWiki (Fandom + Wikipedia + custom MediaWiki sites)
    Uses the MediaWiki API for clean structured data. Rich infobox,
    category metadata, link graph.

  Tier 2 — HTML wiki / structured doc (Notion public, GitBook, indie
    wiki sites). Rendered HTML with discoverable sections + internal
    links. No API, but lxml can parse what's there.

  Tier 3 — Generic page (worldbuilding blog, TVTropes, single-page
    lore docs). Best-effort text + thumbnail extraction. Few or no
    follow-up links to chase.

The extractor is deliberately conservative — strips chrome (navboxes,
edit links, citations) but doesn't try to interpret unusual templates.
The model on the other end is the smart layer.

Output: ``ContentDoc`` — a normalized shape across all tiers. Field
richness varies (Tier 1 has full infobox + categories; Tier 3 may have
title + summary only) but the consumer can rely on the shape.

Backward compat: ``WikiContext = ContentDoc`` and ``fetch_wiki_context``
remain as aliases so existing imports keep working.
"""

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, unquote, urljoin, urlparse

from augmentum.utils.logging import get_logger
from augmentum.utils.safe_http import SafeHttpClient, SafeHttpError

log = get_logger(__name__)


# ── Cache ──────────────────────────────────────────────────────────────────
# The Cardsmith hits content_preview before /start, then /turn fetches
# additional documents. Cache by URL with a short TTL so we don't refetch.
_CACHE_MAX = 128
_CACHE_TTL_SECONDS = 60 * 30  # 30 minutes
_cache: OrderedDict[str, tuple[float, ContentDoc]] = OrderedDict()


def _cache_get(url: str) -> ContentDoc | None:
    entry = _cache.get(url)
    if entry is None:
        return None
    ts, ctx = entry
    if time.time() - ts > _CACHE_TTL_SECONDS:
        _cache.pop(url, None)
        return None
    _cache.move_to_end(url)
    return ctx


def _cache_put(url: str, ctx: ContentDoc) -> None:
    _cache[url] = (time.time(), ctx)
    _cache.move_to_end(url)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def clear_content_cache() -> None:
    """Test helper — drop all cached entries."""
    _cache.clear()


# Backward-compat alias.
clear_wiki_cache = clear_content_cache


# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class Link:
    """An extractable internal link from a document.

    The Cardsmith may emit ``fetch_targets[]`` referencing these paths to
    follow them. Path is relative (e.g. ``/wiki/Sapin_Kingdom``) when the
    source is a structured wiki, full URL otherwise.
    """

    title: str  # link text — what the user / model sees
    path: str  # relative path or full URL
    is_internal: bool  # True if same host as source

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "path": self.path, "is_internal": self.is_internal}


@dataclass
class ContentDoc:
    """Normalized output of the content extractor.

    Field richness varies by tier:
      Tier 1 (MediaWiki): all fields populated
      Tier 2 (HTML wiki): everything except infobox + categories may be empty
      Tier 3 (generic): title + summary populated; sections/links best-effort
    """

    url: str
    source_kind: str  # "fandom" | "wikipedia" | "mediawiki" | "html" | "generic"
    title: str
    summary: str  # lead paragraph(s), plain text, ~1500 char cap

    # Rich structure (Tier 1 most-populated; Tier 2-3 may be empty)
    infobox: dict[str, str] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    thumbnail_url: str = ""
    categories: list[str] = field(default_factory=list)

    # Agnostic enrichment — works across all tiers
    aliases: list[str] = field(default_factory=list)
    extracted_links: list[Link] = field(default_factory=list)

    # Card-type classification (used by Cardsmith to suggest type)
    detected_type: str = "single"
    confidence: float = 0.5

    @property
    def host_kind(self) -> str:
        """Backward-compat alias — old code reads ``host_kind`` instead of ``source_kind``."""
        return self.source_kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "source_kind": self.source_kind,
            "host_kind": self.source_kind,  # backward compat
            "title": self.title,
            "summary": self.summary,
            "infobox": self.infobox,
            "sections": self.sections,
            "thumbnail_url": self.thumbnail_url,
            "detected_type": self.detected_type,
            "confidence": self.confidence,
            "categories": self.categories,
            "aliases": self.aliases,
            "extracted_links": [link.to_dict() for link in self.extracted_links],
        }

    def for_prompt(self) -> str:
        """Render as a structured XML-ish block for the system prompt."""
        lines = [
            f'<source kind="{self.source_kind}">',
            f"  <url>{self.url}</url>",
            f"  <title>{self.title}</title>",
            f"  <detected_type>{self.detected_type}</detected_type>",
        ]
        if self.summary:
            lines.append(f"  <summary>{_clip(self.summary, 1200)}</summary>")
        if self.aliases:
            lines.append(
                f"  <aliases>{', '.join(_xml_escape(a) for a in self.aliases[:8])}</aliases>"
            )
        if self.infobox:
            lines.append("  <infobox>")
            for k, v in list(self.infobox.items())[:20]:
                lines.append(
                    f'    <field name="{_xml_escape(k)}">{_xml_escape(_clip(v, 200))}</field>'
                )
            lines.append("  </infobox>")
        if self.sections:
            lines.append("  <sections>")
            for heading, content in list(self.sections.items())[:8]:
                lines.append(f'    <section heading="{_xml_escape(heading)}">')
                lines.append(f"      {_xml_escape(_clip(content, 800))}")
                lines.append("    </section>")
            lines.append("  </sections>")
        if self.categories:
            lines.append(f"  <categories>{', '.join(self.categories[:8])}</categories>")
        if self.extracted_links:
            lines.append("  <links>")
            for link in self.extracted_links[:30]:
                lines.append(
                    f'    <link path="{_xml_escape(link.path)}">{_xml_escape(link.title)}</link>'
                )
            lines.append("  </links>")
        lines.append("</source>")
        return "\n".join(lines)


# Backward-compat alias for Phase 2 code paths.
WikiContext = ContentDoc


class ContentExtractError(Exception):
    pass


# Backward-compat alias.
WikiExtractError = ContentExtractError


# ── Public entry ──────────────────────────────────────────────────────────

async def fetch_content_doc(
    url: str,
    *,
    use_cache: bool = True,
    base_host: str = "",
) -> ContentDoc:
    """Fetch and structure any URL.

    The base_host param is unused for absolute URLs; it's a no-op kept for
    symmetry with ``fetch_path``. Resolves wiki paths via ``fetch_path``.
    """
    if not url or not url.strip():
        raise ContentExtractError("URL is empty")

    url = url.strip()
    if use_cache:
        cached = _cache_get(url)
        if cached is not None:
            return cached

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ContentExtractError("Only http/https URLs supported")

    host = (parsed.hostname or "").lower()
    if host.endswith(".fandom.com") or host == "fandom.com":
        ctx = await _fetch_mediawiki(url, host, source_kind="fandom")
    elif "wikipedia.org" in host:
        ctx = await _fetch_mediawiki(url, host, source_kind="wikipedia")
    else:
        ctx = await _fetch_generic(url)

    _cache_put(url, ctx)
    return ctx


async def fetch_path(
    path_or_url: str,
    *,
    base_host: str,
    use_cache: bool = True,
) -> ContentDoc:
    """Resolve a wiki path or relative URL against ``base_host``, then fetch.

    Used by the agentic fetch loop when the model emits
    ``{"path": "Sapin_Kingdom"}`` — we resolve against the host of the
    original URL the user pasted.
    """
    if not path_or_url or not path_or_url.strip():
        raise ContentExtractError("Path is empty")
    s = path_or_url.strip()

    if s.startswith(("http://", "https://")):
        return await fetch_content_doc(s, use_cache=use_cache)

    # Treat as a wiki page title or relative path. Build a /wiki/<path> URL
    # against the base host. Strip any leading slash + url-encode.
    if s.startswith("/"):
        # Absolute path on the base host (e.g. "/wiki/Sapin_Kingdom").
        url = f"https://{base_host.lstrip('/')}{s}"
    else:
        # Bare title like "Sapin_Kingdom" or "Sapin Kingdom".
        url = f"https://{base_host.lstrip('/')}/wiki/{quote(s.replace(' ', '_'))}"
    return await fetch_content_doc(url, use_cache=use_cache)


# Backward-compat alias used by Phase 2 routes.
fetch_wiki_context = fetch_content_doc


# ── MediaWiki path (Fandom + Wikipedia + custom MediaWiki) ────────────────

def _api_endpoint(host: str, source_kind: str) -> str:
    if source_kind == "wikipedia":
        return f"https://{host}/w/api.php"
    return f"https://{host}/api.php"


_WIKI_TITLE_RE = re.compile(r"/wiki/([^?#]+)")


def _extract_title_from_url(url: str) -> str:
    m = _WIKI_TITLE_RE.search(url)
    if not m:
        return ""
    return unquote(m.group(1)).replace("_", " ")


async def _fetch_mediawiki(
    url: str, host: str, *, source_kind: str,
) -> ContentDoc:
    title = _extract_title_from_url(url)
    if not title:
        raise ContentExtractError("Could not parse article title from URL")

    api = _api_endpoint(host, source_kind)
    client = SafeHttpClient()

    # 1. Lead summary + thumbnail + categories
    query_url = (
        f"{api}?action=query&format=json&redirects=1"
        f"&prop=extracts%7Cpageimages%7Ccategories"
        f"&exintro=1&explaintext=1"
        f"&piprop=thumbnail&pithumbsize=400"
        f"&cllimit=20&clshow=!hidden"
        f"&titles={quote(title)}"
    )
    try:
        text, _ = await client.fetch(query_url, timeout=15.0)
    except SafeHttpError as exc:
        raise ContentExtractError(f"Fetch blocked: {exc}") from exc
    except Exception as exc:
        raise ContentExtractError(f"Fetch failed: {exc}") from exc

    try:
        data = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ContentExtractError("Wiki API returned invalid JSON") from exc

    pages = data.get("query", {}).get("pages", {}) or {}
    if not pages:
        raise ContentExtractError(f"Article '{title}' not found")
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ContentExtractError(f"Article '{title}' not found")

    real_title = page.get("title", title)
    summary = (page.get("extract") or "").strip()
    thumbnail = (page.get("thumbnail") or {}).get("source", "") or ""
    categories = [
        (c.get("title", "") or "").replace("Category:", "")
        for c in (page.get("categories") or [])
        if c.get("title")
    ]

    # 2. Section HTML for per-section paragraphs + infobox + links
    parse_url = (
        f"{api}?action=parse&format=json&redirects=1"
        f"&prop=text%7Csections%7Cdisplaytitle"
        f"&page={quote(title)}"
    )
    sections: dict[str, str] = {}
    infobox: dict[str, str] = {}
    extracted_links: list[Link] = []
    aliases: list[str] = []
    html_text = ""
    try:
        ptext, _ = await client.fetch(parse_url, timeout=20.0)
        pdata = json.loads(ptext)
        html_text = pdata.get("parse", {}).get("text", {}).get("*", "") or ""
    except (SafeHttpError, ValueError, TypeError) as exc:
        log.debug("wiki_parse_step_failed", url=url, error=str(exc))

    if html_text:
        sections = _extract_sections(html_text)
        infobox = _extract_infobox(html_text)
        if not summary:
            summary = _extract_lead_paragraphs(html_text)
        extracted_links = _extract_mediawiki_links(html_text, host=host)
        aliases = _extract_aliases(html_text, infobox=infobox, title=real_title)

    detected_type, confidence = _classify_type(
        title=real_title,
        infobox=infobox,
        categories=categories,
    )

    return ContentDoc(
        url=url,
        source_kind=source_kind,
        title=real_title,
        summary=summary,
        infobox=infobox,
        sections=sections,
        thumbnail_url=thumbnail,
        detected_type=detected_type,
        confidence=confidence,
        categories=categories,
        aliases=aliases,
        extracted_links=extracted_links,
    )


# ── HTML extraction ───────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[\d+\]")
_EDIT_RE = re.compile(r"\[\s*edit\s*\]", re.IGNORECASE)


def _norm_text(s: str) -> str:
    s = _CITATION_RE.sub("", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" :-")


def _drop_chrome(root) -> None:
    for sel in (
        '//span[contains(@class, "mw-editsection")]',
        '//table[contains(@class, "navbox")]',
        '//table[contains(@class, "infobox")]',
        '//aside[contains(@class, "portable-infobox")]',
        '//div[contains(@class, "thumb")]',
        '//div[contains(@class, "reference")]',
        '//sup[contains(@class, "reference")]',
    ):
        for el in root.xpath(sel):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)


def _extract_sections(html_text: str) -> dict[str, str]:
    try:
        from lxml import html as lxml_html
    except ImportError:
        return {}
    try:
        root = lxml_html.fromstring(html_text)
    except Exception:
        return {}

    _drop_chrome(root)
    body_candidates = root.xpath('//div[contains(@class, "mw-parser-output")]')
    body = body_candidates[0] if body_candidates else root

    sections: dict[str, str] = {}
    current_heading: str | None = None
    current_parts: list[str] = []
    capped = False
    max_paragraphs = 3

    def _flush() -> None:
        if current_heading and current_parts:
            sections[current_heading] = _trim("\n\n".join(current_parts), 1500)

    def _heading_text_from_child(child) -> str | None:
        tag = (child.tag or "").lower() if isinstance(child.tag, str) else ""
        if tag in ("h2", "h3", "h4"):
            return _norm_text(
                _EDIT_RE.sub("", child.text_content() or ""),
            ).strip(" []") or None
        if tag == "div":
            cls = child.get("class") or ""
            if "mw-heading" in cls:
                inner = child.xpath('./h2 | ./h3 | ./h4')
                if inner:
                    return _norm_text(
                        _EDIT_RE.sub("", inner[0].text_content() or ""),
                    ).strip(" []") or None
        return None

    for child in body.iterchildren():
        heading = _heading_text_from_child(child)
        if heading is not None:
            _flush()
            current_heading = heading
            current_parts = []
            capped = False
            continue
        tag = (child.tag or "").lower() if isinstance(child.tag, str) else ""
        if tag == "p" and current_heading and not capped:
            text = _norm_text(_EDIT_RE.sub("", child.text_content() or ""))
            if text:
                current_parts.append(text)
                if len(current_parts) >= max_paragraphs:
                    capped = True
    _flush()

    return {
        k: v for k, v in sections.items()
        if k.lower() not in _NOISE_SECTIONS
    }


_NOISE_SECTIONS: frozenset[str] = frozenset({
    "references", "external links", "see also", "notes", "navigation",
    "gallery", "trivia", "site navigation",
})


def _extract_lead_paragraphs(html_text: str, *, max_chars: int = 1500) -> str:
    try:
        from lxml import html as lxml_html
    except ImportError:
        return ""
    try:
        root = lxml_html.fromstring(html_text)
    except Exception:
        return ""

    _drop_chrome(root)
    body_candidates = root.xpath('//div[contains(@class, "mw-parser-output")]')
    body = body_candidates[0] if body_candidates else root

    lead_parts: list[str] = []
    for child in body.iterchildren():
        tag = (child.tag or "").lower() if isinstance(child.tag, str) else ""
        if tag in ("h2", "h3", "h4"):
            break
        if tag == "div" and "mw-heading" in (child.get("class") or ""):
            break
        if tag == "p":
            text = _norm_text(_EDIT_RE.sub("", child.text_content() or ""))
            if text:
                lead_parts.append(text)
                if len(lead_parts) >= 3:
                    break

    return _trim("\n\n".join(lead_parts), max_chars)


def _extract_infobox(html_text: str) -> dict[str, str]:
    try:
        from lxml import html as lxml_html
    except ImportError:
        return {}
    try:
        root = lxml_html.fromstring(html_text)
    except Exception:
        return {}

    infobox: dict[str, str] = {}

    for aside in root.xpath('//aside[contains(@class, "portable-infobox")]'):
        for item in aside.xpath('.//div[contains(@class, "pi-data")]'):
            label_el = item.xpath('.//*[contains(@class, "pi-data-label")]')
            value_el = item.xpath('.//*[contains(@class, "pi-data-value")]')
            if label_el and value_el:
                k = _norm_text(label_el[0].text_content() or "")
                v = _norm_text(value_el[0].text_content() or "")
                if k and v and len(v) < 500:
                    infobox[k] = v
        if infobox:
            return infobox

    for tbl in root.xpath('//table[contains(@class, "infobox")]'):
        for tr in tbl.xpath('.//tr'):
            ths = tr.xpath('./th')
            tds = tr.xpath('./td')
            if ths and tds:
                k = _norm_text(ths[0].text_content() or "")
                v = _norm_text(tds[0].text_content() or "")
                if k and v and len(v) < 500:
                    infobox[k] = v
        if infobox:
            return infobox

    return infobox


# ── Aliases extraction (for the reference index) ──────────────────────────

# Infobox keys that signal "this is an alternate name for the entity"
_ALIAS_INFOBOX_KEYS: frozenset[str] = frozenset({
    "alias", "aliases", "nickname", "nicknames", "also known as", "aka",
    "real name", "true name", "epithet", "epithets", "title", "titles",
    "former names", "alternate names", "other names",
})


def _extract_aliases(
    html_text: str,
    *,
    infobox: dict[str, str],
    title: str,
) -> list[str]:
    """Mine alternate names for the entity from infobox + lead paragraph bolds.

    Sources, in priority order:
      1. Infobox alias-flavored fields (split on commas/slashes)
      2. Bold tokens in the lead paragraph (the wiki convention is to
         bold the canonical name + major aliases in the very first sentence)
      3. Title variations (split parens, trim leading articles)

    Returns deduped list, capped at 12. Title itself is NOT included
    (it's the primary key separately).
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = s.strip(" ,;.")
        if not s or len(s) > 60:
            return
        if s.lower() == title.lower():
            return
        if s.lower() in seen:
            return
        seen.add(s.lower())
        out.append(s)

    # 1. Infobox alias-flavored fields
    for k, v in infobox.items():
        if k.lower() in _ALIAS_INFOBOX_KEYS:
            for piece in re.split(r"[,/;|]| or ", v):
                _add(piece.strip())

    # 2. Bold tokens in lead paragraph
    try:
        from lxml import html as lxml_html
    except ImportError:
        return out[:12]

    try:
        root = lxml_html.fromstring(html_text)
    except Exception:
        return out[:12]

    body_candidates = root.xpath('//div[contains(@class, "mw-parser-output")]')
    body = body_candidates[0] if body_candidates else root

    # Find the first <p> in the lead (before any heading) and pull its <b>/<strong> children
    for child in body.iterchildren():
        tag = (child.tag or "").lower() if isinstance(child.tag, str) else ""
        if tag in ("h2", "h3", "h4"):
            break
        if tag == "div" and "mw-heading" in (child.get("class") or ""):
            break
        if tag == "p":
            for bold in child.xpath('.//b | .//strong'):
                text = _norm_text(bold.text_content() or "")
                if text:
                    _add(text)
            break  # only the first lead paragraph

    # 3. Title variations — strip parens
    paren_match = re.match(r"^(.+?)\s*\(.+?\)\s*$", title)
    if paren_match:
        _add(paren_match.group(1))

    return out[:12]


# ── Link extraction (for the agentic fetch loop) ──────────────────────────

# MediaWiki path prefixes we DON'T want as fetchable targets — these are
# meta/admin/file pages, not content the Cardsmith should chase.
_LINK_NOISE_PREFIXES: tuple[str, ...] = (
    "Special:", "File:", "Image:", "Category:", "Template:", "Help:",
    "Talk:", "User:", "User_talk:", "Project:", "MediaWiki:",
    "Module:", "Forum:", "Blog:", "Thread:", "Board:", "Source:",
)


def _extract_mediawiki_links(html_text: str, *, host: str) -> list[Link]:
    """Pull internal /wiki/<X> links from an article body.

    Only same-host links count as internal — external links are excluded
    (we don't want the model wandering off-site without confirmation).
    Filters out namespace-prefixed pages (Category:, Template:, etc.) since
    they're meta, not content.
    """
    try:
        from lxml import html as lxml_html
    except ImportError:
        return []
    try:
        root = lxml_html.fromstring(html_text)
    except Exception:
        return []

    _drop_chrome(root)

    body_candidates = root.xpath('//div[contains(@class, "mw-parser-output")]')
    body = body_candidates[0] if body_candidates else root

    seen_paths: set[str] = set()
    links: list[Link] = []

    for a in body.xpath('.//a[@href]'):
        href = a.get("href") or ""
        title = _norm_text(a.text_content() or "")
        if not title:
            continue
        # Match /wiki/<X> patterns
        if "/wiki/" not in href:
            continue
        # External URLs (http/https that point elsewhere) excluded
        if href.startswith(("http://", "https://")):
            try:
                parsed = urlparse(href)
                if (parsed.hostname or "").lower() != host:
                    continue
                path = parsed.path
            except ValueError:
                # urlparse raises on malformed IPv6 hosts / non-ASCII
                continue
        else:
            path = href.split("#")[0]  # strip fragment

        # Pull the actual page slug
        m = _WIKI_TITLE_RE.search(path)
        if not m:
            continue
        slug = unquote(m.group(1))
        if any(slug.startswith(p) for p in _LINK_NOISE_PREFIXES):
            continue
        canonical = "/wiki/" + slug
        if canonical in seen_paths:
            continue
        seen_paths.add(canonical)
        links.append(Link(title=title, path=canonical, is_internal=True))
        if len(links) >= 60:  # cap to avoid bloating prompts
            break

    return links


def _trim(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


# ── Type classification ───────────────────────────────────────────────────

_ENSEMBLE_TITLE_HINTS = (
    "team", "squad", "clan", "faction", "organization", "guild", "house",
    "order of", "brotherhood", "society", "league", "academy", "council",
)
_WORLD_TITLE_HINTS = (
    "world of", "realm of", "kingdom of", "land of", "city of",
)
_ENSEMBLE_INFOBOX_KEYS = {"members", "leader", "leaders", "founders", "headquarters"}
_WORLD_INFOBOX_KEYS = {"location", "region", "capital", "government", "population"}
_ENSEMBLE_CATEGORY_HINTS = ("groups", "factions", "teams", "organizations", "clans", "guilds")
_WORLD_CATEGORY_HINTS = ("locations", "places", "settings", "regions", "cities", "kingdoms", "realms")
_SINGLE_CATEGORY_HINTS = ("characters", "people", "individuals")


def _classify_type(
    *, title: str, infobox: dict[str, str], categories: list[str],
) -> tuple[str, float]:
    title_l = title.lower()
    cat_l = [c.lower() for c in categories]
    info_keys = {k.lower() for k in infobox}

    if any(any(hint in c for hint in _WORLD_CATEGORY_HINTS) for c in cat_l):
        return "world_rpg", 0.85
    if any(any(hint in c for hint in _ENSEMBLE_CATEGORY_HINTS) for c in cat_l):
        return "ensemble", 0.85
    if any(any(hint in c for hint in _SINGLE_CATEGORY_HINTS) for c in cat_l):
        return "single", 0.85

    if any(hint in title_l for hint in _WORLD_TITLE_HINTS):
        return "world_rpg", 0.7
    if any(hint in title_l for hint in _ENSEMBLE_TITLE_HINTS):
        return "ensemble", 0.7

    if info_keys & _WORLD_INFOBOX_KEYS:
        return "world_rpg", 0.6
    if info_keys & _ENSEMBLE_INFOBOX_KEYS:
        return "ensemble", 0.6

    return "single", 0.5


# ── Generic HTML / Tier 2-3 fallback ──────────────────────────────────────

async def _fetch_generic(url: str) -> ContentDoc:
    """Tier 2/3 fallback for non-MediaWiki URLs."""
    client = SafeHttpClient()
    try:
        text, _meta = await client.fetch(url, timeout=20.0)
    except SafeHttpError as exc:
        raise ContentExtractError(f"Fetch blocked: {exc}") from exc
    except Exception as exc:
        raise ContentExtractError(f"Fetch failed: {exc}") from exc

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    title, summary, thumbnail, sections, links, aliases = _parse_generic_html(text, url, host)

    return ContentDoc(
        url=url,
        source_kind="html" if sections else "generic",
        title=title or url,
        summary=summary,
        sections=sections,
        thumbnail_url=thumbnail,
        detected_type="single",
        confidence=0.4,
        aliases=aliases,
        extracted_links=links,
    )


def _parse_generic_html(
    html_text: str, url: str, host: str,
) -> tuple[str, str, str, dict[str, str], list[Link], list[str]]:
    """Tier 2/3 extraction. Returns (title, summary, thumbnail, sections, links, aliases)."""
    try:
        from lxml import html as lxml_html
    except ImportError:
        return "", _trim(html_text, 1500), "", {}, [], []

    try:
        root = lxml_html.fromstring(html_text)
    except Exception:
        return "", "", "", {}, [], []

    title_el = root.xpath('//title')
    title = title_el[0].text_content().strip() if title_el else ""

    # Drop chrome before structural extraction.
    for el in root.xpath('//script | //style | //nav | //aside | //footer | //header'):
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)

    # Find primary content container.
    article = root.xpath('//article | //main | //div[@id="content"]')
    target = article[0] if article else (root.xpath('//body') or [root])[0]

    # Sections: walk top-level descendants of target, group <p> under preceding heading.
    sections: dict[str, str] = {}
    current_heading: str | None = None
    current_parts: list[str] = []
    cap = False
    max_p = 3

    def _flush():
        if current_heading and current_parts:
            sections[current_heading] = _trim("\n\n".join(current_parts), 1500)

    # Use iter() with element filtering to handle deeper structures (HTML
    # wikis don't always put h2/p as direct children of <article>).
    for el in target.iter():
        tag = (el.tag or "").lower() if isinstance(el.tag, str) else ""
        if tag in ("h2", "h3", "h4"):
            _flush()
            current_heading = _norm_text(el.text_content() or "") or None
            current_parts = []
            cap = False
        elif tag == "p" and current_heading and not cap:
            text = _norm_text(el.text_content() or "")
            if text:
                current_parts.append(text)
                if len(current_parts) >= max_p:
                    cap = True
    _flush()

    # Drop noise sections.
    sections = {k: v for k, v in sections.items() if k.lower() not in _NOISE_SECTIONS}

    # Summary: lead paragraphs before first h2.
    lead_parts: list[str] = []
    for el in target.iter():
        tag = (el.tag or "").lower() if isinstance(el.tag, str) else ""
        if tag in ("h2", "h3", "h4"):
            break
        if tag == "p":
            text = _norm_text(el.text_content() or "")
            if text:
                lead_parts.append(text)
                if len(lead_parts) >= 3:
                    break
    summary = _trim("\n\n".join(lead_parts), 2000) or _trim(target.text_content(), 2000)

    # Thumbnail: og:image meta first, else first <img> in target with src.
    og = root.xpath('//meta[@property="og:image"]/@content')
    thumb = og[0] if og else ""
    if not thumb:
        imgs = target.xpath('.//img[@src]/@src')
        if imgs:
            thumb = urljoin(url, imgs[0])

    # Aliases: bold tokens in first <p> of target.
    aliases: list[str] = []
    seen_aliases: set[str] = set()
    for el in target.iter():
        tag = (el.tag or "").lower() if isinstance(el.tag, str) else ""
        if tag == "p":
            for bold in el.xpath('.//b | .//strong'):
                txt = _norm_text(bold.text_content() or "").strip(" ,;.")
                if (
                    txt
                    and len(txt) <= 60
                    and txt.lower() != title.lower()
                    and txt.lower() not in seen_aliases
                ):
                    seen_aliases.add(txt.lower())
                    aliases.append(txt)
            break  # first p only
        if tag in ("h2", "h3"):
            break

    # Links: same-host <a href> from target.
    links: list[Link] = []
    seen_paths: set[str] = set()
    for a in target.xpath('.//a[@href]'):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#") or href.startswith("mailto:"):
            continue
        link_text = _norm_text(a.text_content() or "")
        if not link_text:
            continue
        # Resolve to absolute URL
        absolute = urljoin(url, href)
        try:
            parsed = urlparse(absolute)
        except ValueError:
            # malformed URL — skip; urlparse only raises on bad IPv6 etc.
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        link_host = (parsed.hostname or "").lower()
        is_internal = link_host == host
        if not is_internal:
            continue  # cross-domain excluded for now
        # Use absolute URL as path for non-MediaWiki sources (no /wiki/ prefix).
        if absolute in seen_paths:
            continue
        seen_paths.add(absolute)
        links.append(Link(title=link_text, path=absolute, is_internal=True))
        if len(links) >= 60:
            break

    return title, summary, thumb, sections, links, aliases[:12]


# ── Helpers ───────────────────────────────────────────────────────────────

def _clip(s: str, n: int) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0] + "…"


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )
