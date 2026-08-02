"""Kiwix OPDS catalog client — fetch, cache, and browse ZIM file metadata."""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KIWIX_CATALOG_URL = "https://opds.library.kiwix.org/catalog/v2/entries"

CATEGORY_MAP: dict[str, str] = {
    "wikipedia": "Wikipedia",
    "stack_exchange": "Stack Exchange",
    "gutenberg": "Books",
    "wikibooks": "Books",
    "devdocs": "Dev",
    "ifixit": "How-To",
    "wikiversity": "Education",
    "libretexts": "Education",
    "freecodecamp": "Education",
    "ted": "Education",
    "wiktionary": "Reference",
    "wikiquote": "Reference",
    "wikivoyage": "Travel",
    "wikinews": "News",
    "wikisource": "Books",
}

# Featured packs — hand-picked for usefulness in RAG.
# Prefer nopic/mini flavours (text-only, smaller, faster to embed).
# Matched by (name, flavour) tuple; if flavour is empty, matches any.
FEATURED_PACKS: list[dict[str, str]] = [
    {"name": "wikipedia_en_all", "flavour": "mini", "reason": "All of Wikipedia, text-only"},
    {"name": "mdwiki_en_all", "flavour": "", "reason": "363K medical articles"},
    {"name": "wikibooks_en_all", "flavour": "nopic", "reason": "Open textbooks"},
    {"name": "freecodecamp_en_all", "flavour": "", "reason": "Learn to code"},
    {"name": "devdocs_en_python", "flavour": "", "reason": "Python API reference"},
    {"name": "devdocs_en_javascript", "flavour": "", "reason": "JavaScript reference"},
]

# Legacy flat list for backward compat with override setting
FEATURED_PACK_IDS: list[str] = [p["name"] for p in FEATURED_PACKS]

# ── "Coder" virtual category ──────────────────────────────────────────
# Pre-curated coding reference packs, surfaced as a first-class catalog
# filter so the coder surface's quick-access button (and anyone in
# Settings → Knowledge) can see just the packs that make `pack_search`
# useful. Matched by pack-name PREFIX so language variants
# (devdocs_en_python, devdocs_en_java, ...) all qualify without
# enumerating every language. This is a curation, not a heuristic —
# extend the list deliberately.
CODER_PACK_PREFIXES: tuple[str, ...] = (
    "devdocs_",            # per-language API references (python, js, java, go, rust, ...)
    "freecodecamp_",       # course/reference corpus
    "stackoverflow_",      # SO dumps (when present in the catalog)
    "stack_exchange_",
)
CODER_CATEGORY = "Coder"


def is_coder_pack(entry_id: str) -> bool:
    """True when a catalog entry id belongs to the curated coder shelf."""
    return entry_id.startswith(CODER_PACK_PREFIXES)

# XML namespaces
_ATOM = "http://www.w3.org/2005/Atom"
_DC = "http://purl.org/dc/terms/"
_OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"


# ---------------------------------------------------------------------------
# CatalogEntry dataclass
# ---------------------------------------------------------------------------


@dataclass
class CatalogEntry:
    """A single entry from the Kiwix OPDS catalog."""

    id: str
    title: str
    description: str
    language: str
    raw_category: str
    article_count: int
    media_count: int
    size_bytes: int
    download_url: str
    thumbnail_url: str
    issued_date: str
    flavour: str = ""
    tags: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def category(self) -> str:
        """Display category mapped from raw_category."""
        return CATEGORY_MAP.get(self.raw_category, "Other")

    @property
    def display_size(self) -> str:
        """Human-readable file size."""
        n = self.size_bytes
        if n == 0:
            return "0 B"
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024:
                return f"{n:.1f} {unit}" if isinstance(n, float) else f"{n} {unit}"
            n /= 1024  # type: ignore[assignment]
        return f"{n:.1f} PB"

    @property
    def flavour_label(self) -> str:
        """Human-readable flavour description."""
        f = self.flavour.strip("_").lower()
        has_pics = "_pictures:yes" in ";".join(self.tags)
        has_ftindex = "_ftindex:yes" in ";".join(self.tags)
        if f == "mini":
            return "Text only (smallest)"
        if f == "nopic":
            return "Full text, no images"
        if f == "maxi":
            return "Full with images"
        if not f:
            parts = []
            if has_pics:
                parts.append("with images")
            if has_ftindex:
                parts.append("searchable")
            return ", ".join(parts) if parts else ""
        return f

    @property
    def display_title(self) -> str:
        """Title with flavour suffix to differentiate variants."""
        if self.flavour and self.flavour.strip("_"):
            return f"{self.title} ({self.flavour.strip('_')})"
        return self.title

    @property
    def license(self) -> str:
        """Heuristic license label based on entry id."""
        pk = self.id.lower()
        if "wikipedia" in pk or "wikimedia" in pk:
            return "CC BY-SA"
        if "gutenberg" in pk or "wikisource" in pk:
            return "Public Domain"
        if "stack_exchange" in pk or "stackoverflow" in pk:
            return "CC BY-SA"
        if "devdocs" in pk:
            return "Various"
        if "ifixit" in pk:
            return "CC BY-NC-SA"
        if "ted" in pk:
            return "CC BY-NC-ND"
        return "Unknown"

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def from_opds(cls, el: ET.Element) -> CatalogEntry:
        """Parse an Atom XML ``<entry>`` element from the Kiwix OPDS feed."""

        def _atom(tag: str) -> str:
            child = el.find(f"{{{_ATOM}}}{tag}")
            return (child.text or "").strip() if child is not None else ""

        def _dc(tag: str) -> str:
            child = el.find(f"{{{_DC}}}{tag}")
            return (child.text or "").strip() if child is not None else ""

        def _plain(tag: str) -> str:
            child = el.find(tag)
            return (child.text or "").strip() if child is not None else ""

        # Kiwix OPDS uses default xmlns=Atom, so ALL elements (even custom
        # ones like <name>, <category>, <articleCount>) are in the Atom namespace.
        # Only dc:issued uses the DC namespace.
        title = _atom("title")
        description = _atom("summary")
        language = _atom("language") or _dc("language")
        issued_date = _dc("issued") or _atom("updated")
        raw_category = _atom("category") or _plain("category")
        article_count = int(_atom("articleCount") or _plain("articleCount") or "0")
        media_count = int(_atom("mediaCount") or _plain("mediaCount") or "0")

        # Pack ID: <name> is the canonical ID in Kiwix; fall back to <id>
        entry_id = _atom("name") or _plain("name") or _atom("id")

        flavour = _atom("flavour") or _plain("flavour")

        tags_raw = _atom("tags") or _plain("tags")
        tags = [t.strip() for t in tags_raw.split(";") if t.strip()] if tags_raw else []

        # Links
        download_url = ""
        thumbnail_url = ""
        size_bytes = 0
        for link in el.findall(f"{{{_ATOM}}}link"):
            rel = link.get("rel", "")
            ltype = link.get("type", "")
            href = link.get("href", "")
            if "acquisition" in rel and ltype == "application/x-zim":
                download_url = href
                length_str = link.get("length", "0")
                try:
                    size_bytes = int(length_str)
                except ValueError:
                    size_bytes = 0
            elif "thumbnail" in rel or "image" in rel:
                thumbnail_url = href

        return cls(
            id=entry_id,
            title=title,
            description=description,
            language=language,
            raw_category=raw_category,
            article_count=article_count,
            media_count=media_count,
            size_bytes=size_bytes,
            download_url=download_url,
            thumbnail_url=thumbnail_url,
            issued_date=issued_date,
            flavour=flavour,
            tags=tags,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry to a dict including computed properties."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "language": self.language,
            "raw_category": self.raw_category,
            "article_count": self.article_count,
            "media_count": self.media_count,
            "size_bytes": self.size_bytes,
            "download_url": self.download_url,
            "thumbnail_url": self.thumbnail_url,
            "issued_date": self.issued_date,
            "flavour": self.flavour,
            "tags": self.tags,
            # Computed
            "category": self.category,
            "display_title": self.display_title,
            "display_size": self.display_size,
            "flavour_label": self.flavour_label,
            "license": self.license,
        }


# ---------------------------------------------------------------------------
# CatalogClient
# ---------------------------------------------------------------------------

_PAGE_SIZE = 500  # entries per OPDS page request


class CatalogClient:
    """Fetch and cache the Kiwix OPDS catalog."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl: int = 86400,
        base_url: str = KIWIX_CATALOG_URL,
    ) -> None:
        self._cache_dir = cache_dir
        self._cache_ttl = cache_ttl
        self._base_url = base_url
        # In-memory cache: lang → (timestamp, entries)
        self._mem_cache: dict[str, tuple[float, list[CatalogEntry]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def categories(self) -> list[str]:
        """Unique sorted display category names from CATEGORY_MAP, with
        the curated virtual "Coder" shelf pinned first."""
        return [CODER_CATEGORY, *sorted(set(CATEGORY_MAP.values()))]

    async def browse(
        self,
        lang: str = "en",
        category: str | None = None,
        max_size_bytes: int | None = None,
        sort: str = "recommended",
        query: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[CatalogEntry]:
        """Fetch entries, apply filters, sort, and paginate."""
        entries = await self._get_all(lang)

        # Filter. "Coder" is a virtual category: a curated prefix list
        # (CODER_PACK_PREFIXES) rather than a Kiwix category mapping.
        if category == CODER_CATEGORY:
            entries = [e for e in entries if is_coder_pack(e.id)]
        elif category:
            entries = [e for e in entries if e.category == category]
        if max_size_bytes is not None:
            entries = [e for e in entries if e.size_bytes <= max_size_bytes]
        if query:
            q = query.lower()
            entries = [
                e for e in entries
                if q in e.title.lower() or q in (e.description or "").lower()
            ]

        # Sort
        entries = _sort_entries(entries, sort)

        # Paginate
        return entries[offset : offset + limit]

    async def featured(
        self,
        lang: str = "eng",
        override: str | None = None,
    ) -> list[CatalogEntry]:
        """Return hand-picked featured packs, matched by name+flavour."""
        entries = await self._get_all(lang)
        if override:
            # Legacy override: comma-separated pack names (any flavour)
            ids = {s.strip() for s in override.split(",") if s.strip()}
            return [e for e in entries if e.id in ids]

        # Match by (name, flavour) from FEATURED_PACKS
        result: list[CatalogEntry] = []
        for fp in FEATURED_PACKS:
            for e in entries:
                if e.id == fp["name"]:
                    if not fp["flavour"] or e.flavour.strip("_") == fp["flavour"]:
                        result.append(e)
                        break
        return result

    async def total_count(self, lang: str = "eng") -> int:
        """Total number of entries available for a language."""
        entries = await self._get_all(lang)
        return len(entries)

    # ------------------------------------------------------------------
    # Internal caching
    # ------------------------------------------------------------------

    async def _get_all(self, lang: str) -> list[CatalogEntry]:
        """Return all entries for lang, using in-memory → disk → network."""
        cache_key = f"catalog_{lang}"
        now = time.time()

        # 1. In-memory cache (TTL=0 means always re-fetch)
        if self._cache_ttl > 0 and cache_key in self._mem_cache:
            ts, cached = self._mem_cache[cache_key]
            if (now - ts) < self._cache_ttl:
                return cached

        # 2. Disk cache (skip if TTL=0)
        if self._cache_ttl > 0 and self._cache_dir is not None:
            disk_entries = self._load_disk_cache(lang, now)
            if disk_entries is not None:
                self._mem_cache[cache_key] = (now, disk_entries)
                return disk_entries

        # 3. Network fetch (paginated)
        log.info("catalog.fetch_start", lang=lang)
        all_entries: list[CatalogEntry] = []
        start = 0
        while True:
            page, total = await self._fetch_page(lang, count=_PAGE_SIZE, start=start)
            all_entries.extend(page)
            start += len(page)
            if not page or start >= total:
                break
        log.info("catalog.fetch_done", lang=lang, count=len(all_entries))

        # Store caches
        self._mem_cache[cache_key] = (now, all_entries)
        if self._cache_dir is not None:
            self._write_disk_cache(lang, all_entries)

        return all_entries

    def _disk_cache_path(self, lang: str) -> Path:
        assert self._cache_dir is not None
        return self._cache_dir / f"catalog_{lang}.json"

    def _load_disk_cache(self, lang: str, now: float) -> list[CatalogEntry] | None:
        path = self._disk_cache_path(lang)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ts: float = data.get("timestamp", 0.0)
            if self._cache_ttl > 0 and (now - ts) >= self._cache_ttl:
                return None
            entries = [_entry_from_dict(d) for d in data.get("entries", [])]
            log.debug("catalog.disk_cache_hit", lang=lang, count=len(entries))
            return entries
        except Exception as exc:
            log.warning("catalog.disk_cache_error", lang=lang, error=str(exc))
            return None

    def _write_disk_cache(self, lang: str, entries: list[CatalogEntry]) -> None:
        path = self._disk_cache_path(lang)
        try:
            payload = {
                "timestamp": time.time(),
                "entries": [e.to_dict() for e in entries],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            log.debug("catalog.disk_cache_write", lang=lang, path=str(path))
        except Exception as exc:
            log.warning("catalog.disk_cache_write_error", lang=lang, error=str(exc))

    # ------------------------------------------------------------------
    # Network layer
    # ------------------------------------------------------------------

    async def _fetch_page(
        self, lang: str, count: int = _PAGE_SIZE, start: int = 0
    ) -> tuple[list[CatalogEntry], int]:
        """Fetch one page from the Kiwix OPDS API."""
        params: dict[str, Any] = {
            "lang": lang,
            "count": count,
            "start": start,
        }
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(self._base_url, params=params)
            resp.raise_for_status()

        root = ET.fromstring(resp.text)
        entries: list[CatalogEntry] = []
        for el in root.findall(f"{{{_ATOM}}}entry"):
            try:
                entries.append(CatalogEntry.from_opds(el))
            except Exception as exc:
                log.warning("catalog.parse_entry_error", error=str(exc))

        # Extract totalResults — Kiwix puts it in Atom namespace (default xmlns)
        total = len(entries)
        for ns in (_ATOM, _OPENSEARCH, ""):
            tag = f"{{{ns}}}totalResults" if ns else "totalResults"
            total_el = root.find(tag)
            if total_el is not None and total_el.text:
                try:
                    total = int(total_el.text)
                    break
                except ValueError:
                    pass

        return entries, total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sort_entries(entries: list[CatalogEntry], sort: str) -> list[CatalogEntry]:
    """Sort entries by the given strategy."""
    featured_names = {p["name"] for p in FEATURED_PACKS}
    if sort == "recommended":
        def _key(e: CatalogEntry) -> tuple[int, int, int, int]:
            # 1. Featured packs first
            is_featured = 0 if e.id in featured_names else 1
            # 2. Prefer text-only flavours for RAG (nopic/mini over maxi)
            f = e.flavour.strip("_").lower()
            flavour_rank = {"mini": 0, "nopic": 1, "": 2, "maxi": 3}.get(f, 2)
            # 3. Higher article count = more useful
            return (is_featured, flavour_rank, -e.article_count, e.size_bytes)
        return sorted(entries, key=_key)
    if sort == "smallest":
        return sorted(entries, key=lambda e: e.size_bytes)
    if sort == "largest":
        return sorted(entries, key=lambda e: -e.size_bytes)
    if sort == "newest":
        return sorted(entries, key=lambda e: e.issued_date or "", reverse=True)
    if sort in ("articles", "most_articles"):
        return sorted(entries, key=lambda e: -e.article_count)
    return entries


def _entry_from_dict(d: dict[str, Any]) -> CatalogEntry:
    """Reconstruct a CatalogEntry from a to_dict() payload."""
    return CatalogEntry(
        id=d.get("id", ""),
        title=d.get("title", ""),
        description=d.get("description", ""),
        language=d.get("language", ""),
        raw_category=d.get("raw_category", ""),
        article_count=d.get("article_count", 0),
        media_count=d.get("media_count", 0),
        size_bytes=d.get("size_bytes", 0),
        download_url=d.get("download_url", ""),
        thumbnail_url=d.get("thumbnail_url", ""),
        issued_date=d.get("issued_date", ""),
        flavour=d.get("flavour", ""),
        tags=d.get("tags", []),
    )
