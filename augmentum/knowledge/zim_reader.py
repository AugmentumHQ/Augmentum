"""ZIM file reader wrapper using libzim for reading and searching ZIM archives.

Retrieval is section-aligned via ``search_passages()``: each article is
cleaned of HTML chrome (CSS, scripts, infoboxes, navboxes) and split by
``<h1>``-``<h6>`` headings into ~900-char passages so the reranker scores
at its trained granularity and a single passage fits the per-mode char
budget. Whole-article retrieval is not supported — MDWiki articles run
100KB+ while the passthrough chat budget is 1.5KB. See the 2026-05-02
all-dropped diagnosis in injection.py.
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

try:
    import libzim  # type: ignore[import-untyped]
except ImportError:
    libzim = None  # type: ignore[assignment]


@dataclass
class ZimPassage:
    """A section-level slice of a ZIM article.

    Each passage carries enough context to render a useful citation
    ("[MDWiki → Type 2 diabetes → Treatment]") without forcing the model
    to swallow the whole article.
    """

    title: str       # Article title (e.g. "Type 2 diabetes")
    section: str     # Heading text for this passage ("" for intro before first heading)
    url: str         # ZIM path (no fragment — libzim doesn't expose anchor IDs cleanly)
    content: str     # Cleaned plain text


@dataclass
class ZimSuggestion:
    """A single typeahead candidate from libzim's SuggestionSearcher.

    Returned by ``ZimReader.suggest()`` for the browse panel's
    pack-scoped search input. ``path`` is the ZIM entry path the
    browse iframe should navigate to on click; ``title`` is the
    human-readable label rendered in the dropdown.
    """

    title: str
    path: str


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Block-level chrome to strip before any text extraction. Order matters —
# strip script/style first so their CSS/JS contents don't leak through the
# tag-stripper as visible text. Then strip MediaWiki structural noise that
# inflates passage size without contributing answer content.
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
# MediaWiki classes that signal "page chrome, not article body."
# infobox/navbox/sidebar/metadata cover the bulk; toc strips the table of
# contents (already implied by section structure); hatnote covers
# "redirected from"-style preambles; mw-editsection is the [edit] link.
_CHROME_TABLE_RE = re.compile(
    r"<table\b[^>]*class=\"[^\"]*"
    r"(?:infobox|navbox|sidebar|metadata|toc|hatnote|mw-editsection)"
    r"[^\"]*\"[^>]*>.*?</table>",
    re.DOTALL | re.IGNORECASE,
)
_CHROME_DIV_RE = re.compile(
    r"<div\b[^>]*class=\"[^\"]*"
    r"(?:navbox|catlinks|printfooter|toc|hatnote|mw-editsection|reference|references)"
    r"[^\"]*\"[^>]*>.*?</div>",
    re.DOTALL | re.IGNORECASE,
)
# Citation footnote markers like [1], [ 24 ], [12], [ 3 ] left over after
# reference stripping. MediaWiki injects extra whitespace around the digits
# in some templates so the brackets and the number are not always adjacent.
_FOOTNOTE_MARKER_RE = re.compile(r"\[\s*\d+\s*\]")
# HTML comments occasionally carry CSS/JS or MediaWiki templating directives.
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Heading capture: split point for sections. Captures heading text in group(2).
_HEADING_RE = re.compile(
    r"<(h[1-6])\b[^>]*>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)
# Sentence boundary for fallback chunking inside oversized sections.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _clean_html_for_text(html: str) -> str:
    """Remove block-level chrome before text extraction.

    Order: comments → script/style → chrome tables → chrome divs. Each pass
    deletes whole blocks (opening tag through matching closing tag) so the
    CSS/JS body inside ``<style>``/``<script>`` does not survive into the
    plain-text output. Without this pass, ZIM article text leads with
    ``"/* start https://mdwiki.org/ */ .mw-parser-output ..."`` — pure
    noise that displaces real answer content from the budget.
    """
    html = _COMMENT_RE.sub(" ", html)
    html = _SCRIPT_STYLE_RE.sub(" ", html)
    html = _CHROME_TABLE_RE.sub(" ", html)
    html = _CHROME_DIV_RE.sub(" ", html)
    return html


def _html_segment_to_text(html: str) -> str:
    """Strip remaining tags + footnote markers + collapse whitespace."""
    text = _TAG_RE.sub(" ", html)
    text = _FOOTNOTE_MARKER_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _chunk_section(section: str, text: str, max_chars: int) -> list[tuple[str, str]]:
    """Yield ``(section, chunk)`` pairs, each chunk ≤ max_chars.

    Splits on sentence boundaries. A sentence longer than max_chars rides
    alone — over-budget passages are still better than mid-sentence cuts
    that mangle the meaning.
    """
    if len(text) <= max_chars:
        return [(section, text)]
    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    current_len = 0
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        s_len = len(sentence)
        if current_len + s_len + 1 > max_chars and current:
            chunks.append((section, " ".join(current)))
            current = [sentence]
            current_len = s_len
        else:
            current.append(sentence)
            current_len += s_len + 1
    if current:
        chunks.append((section, " ".join(current)))
    return chunks


def extract_passages(html: str, *, max_chars: int = 900, min_chars: int = 100) -> list[tuple[str, str]]:
    """Split a ZIM article's HTML into ``(section, plain_text)`` passages.

    Algorithm:
        1. Strip block-level chrome (script/style/infobox/navbox/etc.).
        2. Find h1-h6 boundaries.
        3. Take the prelude before the first heading as the intro passage
           (section="").
        4. For each heading, take the body until the next heading.
        5. Strip remaining tags + collapse whitespace per segment.
        6. Drop segments shorter than ``min_chars`` (noise like single-line
           "See also" stubs).
        7. Split oversized segments into sentence-aligned chunks of
           ≤ ``max_chars``.

    Returns:
        List of ``(section_heading, plain_text)`` tuples in document order.
        Empty list if the article has no usable content after cleaning.
    """
    cleaned = _clean_html_for_text(html)

    matches = list(_HEADING_RE.finditer(cleaned))
    if not matches:
        text = _html_segment_to_text(cleaned)
        if len(text) < min_chars:
            return []
        return _chunk_section("", text, max_chars)

    out: list[tuple[str, str]] = []

    intro_html = cleaned[: matches[0].start()]
    intro_text = _html_segment_to_text(intro_html)
    if len(intro_text) >= min_chars:
        out.extend(_chunk_section("", intro_text, max_chars))

    for i, m in enumerate(matches):
        heading = _html_segment_to_text(m.group(2))
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        body_text = _html_segment_to_text(cleaned[body_start:body_end])
        if len(body_text) < min_chars:
            continue
        out.extend(_chunk_section(heading, body_text, max_chars))

    return out


class ZimReader:
    """Read and search a ZIM archive via libzim.

    Uses the Xapian full-text index embedded in ZIM files for keyword search.
    Some ZIM files lack a search index — in that case searches return empty.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._archive: Any = None
        self._searcher: Any = None
        self._suggester: Any = None
        # Lock guards libzim suggester calls from concurrent access.
        # Mirrors libkiwix's ``LockableSuggestionSearcher`` — at scale
        # we'll have parallel typeahead requests on the same pack
        # and the underlying suggester is not documented as thread-
        # safe. Created eagerly because asyncio.Lock construction
        # is loop-agnostic in 3.10+; only acquire/release touches
        # the running loop.
        self._suggest_lock = asyncio.Lock()

        if libzim is None:
            log.warning("libzim not installed, ZIM support disabled")
            return

        try:
            self._archive = libzim.Archive(str(self._path))
        except Exception:
            log.warning("zim_archive_open_failed", path=str(self._path), exc_info=True)
            return

        try:
            self._searcher = libzim.Searcher(self._archive)
        except Exception:
            log.warning("zim_no_search_index", path=str(self._path))
            self._searcher = None

        # SuggestionSearcher is the typeahead-optimized index.
        # Some libzim builds and some ZIMs (especially scraper-
        # generated archives without a title index) lack it; in
        # that case ``suggest()`` falls back to the full-text
        # Searcher so typeahead still functions, just less crisp.
        try:
            if hasattr(libzim, "SuggestionSearcher"):
                self._suggester = libzim.SuggestionSearcher(self._archive)
        except Exception:
            log.info("zim_no_suggest_index", path=str(self._path))
            self._suggester = None

    @property
    def article_count(self) -> int:
        """Return entry count from the archive."""
        if self._archive is None:
            return 0
        return self._archive.entry_count

    async def suggest(self, query: str, limit: int = 8) -> list[ZimSuggestion]:
        """Typeahead suggestions via libzim's SuggestionSearcher.

        Surfaces title-prefix matches for the browse panel's
        pack-scoped search input. Returns up to ``limit`` candidates
        as ``ZimSuggestion`` records (``title`` + ``path``).

        The libzim call runs on a worker thread via ``asyncio.to_thread``
        and is wrapped in an ``asyncio.Lock`` so concurrent typeahead
        requests on the same pack serialize. libzim's suggester /
        searcher are not documented as thread-safe; libkiwix wraps
        them in mutexes for the same reason.

        Empty query returns ``[]``. Falls back to the full-text
        Searcher when the SuggestionSearcher is unavailable, so
        users on libzim builds without a title index still get
        usable (though less prefix-aware) suggestions.
        """
        if self._archive is None or libzim is None:
            return []
        if not query or not query.strip():
            return []
        async with self._suggest_lock:
            return await asyncio.to_thread(self._suggest_sync, query, limit)

    def _suggest_sync(self, query: str, limit: int) -> list[ZimSuggestion]:
        """Sync side of ``suggest()``. Runs on a worker thread under
        the coroutine-level lock acquired in ``suggest()`` — do not
        call directly from request handlers.
        """
        paths: list[str] = []
        if self._suggester is not None:
            try:
                search_obj = self._suggester.suggest(query)
                paths = list(search_obj.getResults(0, limit))
            except Exception:
                log.warning(
                    "zim_suggest_native_failed", query=query, exc_info=True,
                )
                paths = []
        if not paths and self._searcher is not None:
            # Fallback: full-text searcher. Title-prefix matches lose
            # rank quality but the candidate list is still useful for
            # typeahead, especially on packs the user has just opened.
            try:
                q = libzim.Query().set_query(query)
                search_obj = self._searcher.search(q)
                paths = list(search_obj.getResults(0, limit))
            except Exception:
                log.warning(
                    "zim_suggest_fallback_failed", query=query, exc_info=True,
                )
                return []
        out: list[ZimSuggestion] = []
        for p in paths:
            try:
                entry = self._archive.get_entry_by_path(p)
                out.append(ZimSuggestion(title=entry.title, path=p))
            except Exception as exc:
                log.debug("zim_suggest_entry_failed", path=p, error=str(exc))
                continue
        return out

    # ------------------------------------------------------------------
    # Passage cache (persistent SQLite sidecar)
    # ------------------------------------------------------------------
    #
    # Why this exists: ZIM passage extraction is a regex pass over each
    # article's HTML (~100KB-500KB). For a query that returns 10 article
    # candidates, the per-query cost is ~800-1500ms — dominant in the
    # whole search pipeline.
    #
    # The cache stores per-article (title + passage list) keyed by ZIM
    # entry path. Hit: ~5ms (one SQLite lookup, one JSON parse). Miss:
    # the original extraction, then a write. The cache file is a sidecar
    # at {zim_path}.passages.cache.sqlite — never modifies the ZIM
    # itself, safe to delete to free space.
    #
    # Bounded by knowledge_passage_cache_max_articles (default 5000 ≈
    # ~50MB per pack at typical passage sizes). Inline LRU eviction by
    # cached_at when the cap is exceeded — cheap because eviction only
    # fires on writes, not reads.

    def _passage_cache_path(self) -> Path:
        """SQLite sidecar path for this ZIM's passage cache."""
        return self._path.with_suffix(self._path.suffix + ".passages.cache.sqlite")

    def _passage_cache_init(self, conn: sqlite3.Connection) -> None:
        """Create schema if missing. Cheap — runs first time per process."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS passages (
                article_url TEXT PRIMARY KEY,
                article_title TEXT,
                passages_json TEXT NOT NULL,
                cached_at INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cached_at ON passages(cached_at)")

    def _passage_cache_read(
        self, article_url: str,
    ) -> tuple[str, list[tuple[str, str]]] | None:
        """Look up a cached extraction. Returns (title, passages) or None."""
        if not settings.knowledge_passage_cache_enabled:
            return None
        path = self._passage_cache_path()
        if not path.exists():
            return None
        try:
            conn = sqlite3.connect(str(path), timeout=2.0)
            try:
                cursor = conn.execute(
                    "SELECT article_title, passages_json FROM passages WHERE article_url = ?",
                    (article_url,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                title = row[0] or ""
                # Stored as a list of [section, content] pairs to keep the
                # JSON small. tuple-ify back for downstream consumers.
                passages = [(p[0], p[1]) for p in json.loads(row[1])]
                return title, passages
            finally:
                conn.close()
        except (sqlite3.Error, json.JSONDecodeError, OSError):
            log.debug("zim_passage_cache_read_failed", path=str(path), exc_info=True)
            return None

    def _passage_cache_write(
        self,
        article_url: str,
        article_title: str,
        passages: list[tuple[str, str]],
    ) -> None:
        """Insert/replace a cache entry. Inline-evicts on overflow.

        Schema is created lazily on first write; the file doesn't exist
        until the first article is cached. Failures are swallowed — the
        cache is best-effort and a write error shouldn't break the user-
        facing search. WAL mode keeps concurrent reads from blocking
        writes (and vice versa) when multiple workers eventually share
        the file.
        """
        if not settings.knowledge_passage_cache_enabled:
            return
        path = self._passage_cache_path()
        try:
            conn = sqlite3.connect(str(path), timeout=2.0)
            try:
                self._passage_cache_init(conn)
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute(
                    "INSERT OR REPLACE INTO passages "
                    "(article_url, article_title, passages_json, cached_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        article_url,
                        article_title,
                        json.dumps([list(p) for p in passages]),
                        int(time.time()),
                    ),
                )
                # Inline eviction. Counts the table; if over cap, deletes
                # the oldest 10% by cached_at. Cheap: a single COUNT +
                # one DELETE on the index. Only fires on writes so reads
                # stay snappy.
                cursor = conn.execute("SELECT COUNT(*) FROM passages")
                cur_count = int(cursor.fetchone()[0])
                cap = settings.knowledge_passage_cache_max_articles
                if cap > 0 and cur_count > cap:
                    to_drop = max(1, (cur_count - cap) + cap // 10)
                    conn.execute(
                        "DELETE FROM passages WHERE article_url IN ("
                        "  SELECT article_url FROM passages "
                        "  ORDER BY cached_at ASC LIMIT ?"
                        ")",
                        (to_drop,),
                    )
                    log.debug(
                        "zim_passage_cache_evicted",
                        path=str(path),
                        dropped=to_drop,
                        retained=cur_count - to_drop,
                    )
                conn.commit()
            finally:
                conn.close()
        except (sqlite3.Error, OSError):
            log.debug("zim_passage_cache_write_failed", path=str(path), exc_info=True)

    def search_passages(
        self,
        query: str,
        *,
        max_articles: int = 12,
        max_chars: int = 900,
    ) -> list[ZimPassage]:
        """Section-level passage retrieval.

        For each article from the keyword search, splits into section-aligned
        passages with HTML chrome (CSS, scripts, infoboxes, navboxes) removed
        and each passage clipped to ``max_chars``. The pack-search RRF/rerank
        pipeline then scores passages individually — its native granularity.

        Args:
            query: Free-text query.
            max_articles: Hard cap on articles fetched. Each article yields
                multiple passages, so 10-15 is usually enough to give the
                reranker a decent candidate pool without exploding latency.
            max_chars: Per-passage size cap. ~900 fits one passage in the
                1500-char passthrough budget with formatting overhead.

        Returns:
            List of ``ZimPassage``. Order roughly mirrors article relevance
            (libzim search rank, document order within article) — but the
            downstream reranker does the real ordering.
        """
        if self._searcher is None:
            return []

        passages: list[ZimPassage] = []
        try:
            query_obj = libzim.Query().set_query(query)
            search_obj = self._searcher.search(query_obj)
            results = search_obj.getResults(0, max_articles)
            for path in results:
                try:
                    # Cache hit: skip the expensive HTML decode + regex
                    # passage split entirely. Typical cached read is ~5ms
                    # vs ~100-200ms uncached for a Wikipedia-scale article.
                    cached = self._passage_cache_read(path)
                    if cached is not None:
                        cached_title, cached_passages = cached
                        for section, text in cached_passages:
                            passages.append(
                                ZimPassage(
                                    title=cached_title,
                                    section=section,
                                    url=path,
                                    content=text,
                                )
                            )
                        continue

                    entry = self._archive.get_entry_by_path(path)
                    item = entry.get_item()
                    mimetype = item.mimetype
                    if not mimetype.startswith("text/"):
                        continue
                    raw = item.content.tobytes().decode("utf-8", errors="replace")
                    article_title = entry.title
                    extracted = list(extract_passages(raw, max_chars=max_chars))
                    for section, text in extracted:
                        passages.append(
                            ZimPassage(
                                title=article_title,
                                section=section,
                                url=path,
                                content=text,
                            )
                        )
                    # Write-through cache. Even an empty extraction is
                    # worth caching — saves the regex pass next time we
                    # see this article (some ZIM entries are stubs that
                    # always extract to nothing).
                    self._passage_cache_write(path, article_title, extracted)
                except Exception:
                    log.warning("zim_entry_read_error", exc_info=True)
        except Exception:
            log.warning("zim_search_error", query=query, exc_info=True)

        return passages

    # ------------------------------------------------------------------
    # Reader-surface extras (illustration, metadata, random)
    # ------------------------------------------------------------------
    #
    # Reader-side ZIM features beyond search/suggest. Used by the browse
    # panel for per-pack favicon, metadata sidebar, and "random article"
    # discovery. All sync — quick libzim lookups; the route layer wraps
    # in asyncio.to_thread where it matters.

    def get_illustration(self, size: int = 48) -> tuple[bytes, str] | None:
        """Return (content_bytes, mimetype) for the pack illustration at the
        given size, or None if the pack has no illustration.

        libzim raises when illustration is missing — common on older ZIMs.
        We swallow silently because absence is normal, not an error.
        """
        if self._archive is None or libzim is None:
            return None
        try:
            item = self._archive.get_illustration_item(size)
            return item.content.tobytes(), item.mimetype
        except Exception:
            return None

    def get_metadata_full(self) -> dict[str, Any]:
        """Return all archive metadata as a dict.

        Standard ZIM metadata keys: Title, Description, Language, Creator,
        Publisher, Date, Tags, Name, Flavour, Counter, etc. Counter (the
        per-MIME-type entry count) is parsed into a ``{mime: int}`` sub-dict;
        all others are returned as UTF-8 strings.

        Missing keys are simply not present in the returned dict — callers
        should ``.get(key, default)`` rather than assume presence.
        """
        if self._archive is None or libzim is None:
            return {}
        out: dict[str, Any] = {}
        try:
            keys = list(self._archive.metadata_keys)
        except Exception:
            log.warning("zim_metadata_keys_failed", path=str(self._path), exc_info=True)
            return {}
        for key in keys:
            try:
                raw = self._archive.get_metadata(key)
                value = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                if key == "Counter":
                    # "text/html=15432;image/jpeg=8234" → {"text/html": 15432, ...}.
                    # Tolerates spaces after semicolons (some scrapers emit them).
                    counter: dict[str, int] = {}
                    for piece in value.split(";"):
                        piece = piece.strip()
                        if not piece or "=" not in piece:
                            continue
                        mime, _, count = piece.rpartition("=")
                        try:
                            counter[mime.strip()] = int(count.strip())
                        except ValueError:
                            continue
                    out[key] = counter
                else:
                    out[key] = value
            except Exception:
                log.debug("zim_metadata_read_failed", key=key, exc_info=True)
        return out

    def get_random_article(self, *, max_attempts: int = 5) -> str | None:
        """Return the path of a random HTML article, or None.

        Filters: text/html MIME only, not the main page, not a disambiguation
        page. Redirects are followed before the MIME check so we never hand
        the caller a 302-loop target. After ``max_attempts`` non-HTML / dup
        draws we give up — image-only packs or packs of only disambiguation
        entries legitimately have no article to return.
        """
        if self._archive is None or libzim is None:
            return None
        try:
            main_path = self._archive.main_entry.get_item().path
        except Exception:
            main_path = None
        for _ in range(max_attempts):
            try:
                entry = self._archive.get_random_entry()
                if entry.is_redirect:
                    entry = entry.get_redirect_entry()
                item = entry.get_item()
                if not item.mimetype.startswith("text/html"):
                    continue
                if main_path and item.path == main_path:
                    continue
                if "(disambiguation)" in (entry.title or "").lower():
                    continue
                return item.path
            except Exception:
                log.debug("zim_random_entry_failed", exc_info=True)
                continue
        return None

    def close(self) -> None:
        """Release archive and searcher references."""
        self._archive = None
        self._searcher = None
