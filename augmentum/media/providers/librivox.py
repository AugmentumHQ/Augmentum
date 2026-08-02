"""LibriVox provider — free, public-domain audiobooks.

Catalog is hosted at https://librivox.org (JSON feed API); audio files
live on archive.org and are streamed through the existing /api/media/stream
proxy with Range passthrough. This provider is a *built-in*: users see it
without connecting anything, so it bypasses user_media_servers entirely —
the sentinel ``server_id='builtin-librivox'`` on file_index rows tells
the route layer to skip the credential store.

Design choices:
    - external_id is the archive.org identifier (last path segment of
      url_iarchive). This makes build_cover_url / stream resolution
      trivial: every archive.org URL is derivable from the identifier.
    - stream_path is empty on LibriVox rows. Books are multi-file
      (one MP3 per chapter), so the stream route resolves a specific
      file via ``?file=<index>`` against ``extra.audio_files``; there's
      no single "whole book" URL to store here.
    - login/verify_token raise — they should never be called for a
      built-in. Defensive: if a future refactor wires LibriVox through
      the add-server flow by accident, the error is loud, not silent.
    - fetch_catalog returns [] — LibriVox has ~20k books and mirroring
      the whole catalog into every user's file_index would be absurd.
      sync.py explicitly skips provider=='librivox' as a belt-and-braces.
"""

from __future__ import annotations

import html
import re
import time
from typing import TYPE_CHECKING, Any

from augmentum.media.providers.base import (
    BrowseResult,
    CatalogItem,
    ProviderInfo,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)

# LibriVox descriptions ship with inline HTML (<i>, <b>, <br>, the occasional
# <a>). The detail panel renders descriptions as plain text, so escape-on-render
# would produce visible tags. Stripping here keeps the flow readable without
# trusting upstream HTML. We also unescape entities so &amp; / &#8212; / etc.
# arrive as real characters.
_TAG_RE = re.compile(r"<[^>]+>")
# Collapse whitespace runs (post-tag-strip) to a single space so paragraph
# breaks don't leave awkward double spaces.
_WS_RE = re.compile(r"\s+")


def _clean_html_text(raw: str) -> str:
    if not raw:
        return ""
    stripped = _TAG_RE.sub(" ", raw)
    unescaped = html.unescape(stripped)
    return _WS_RE.sub(" ", unescaped).strip()


LIBRIVOX_API = "https://librivox.org/api/feed/audiobooks"
ARCHIVE_METADATA = "https://archive.org/metadata"
ARCHIVE_DOWNLOAD = "https://archive.org/download"
ARCHIVE_COVER = "https://archive.org/services/img"
# archive.org's Advanced Search over the LibriVox collection. LibriVox's
# own feed API ignores `search=`, `genre=`, and `genre_id=` — every query
# returns the catalogue's default front page. Verified live 2026-04-20.
# archive.org runs a real search index over the same recordings and does
# respect `subject:<name>` + free-text, so we build browse on top of it.
ARCHIVE_SEARCH = "https://archive.org/advancedsearch.php"
ARCHIVE_COLLECTION = "librivoxaudio"

_TIMEOUT_S = 15.0
_DETAILS_TIMEOUT_S = 20.0


def _english_first(results: list[BrowseResult]) -> list[BrowseResult]:
    """Stable sort that lifts English entries above the rest.

    Archive.org's search and LibriVox's feed both mix languages in their
    default order. LibriVox is ~90% English so users expect English-first
    without having to filter — but keeping non-English visible (below) is
    the right trade for a library with plenty of French/German/Spanish
    classics. Python's sorted is stable, so within each group the
    upstream relevance/freshness order is preserved.
    """
    return sorted(
        results,
        key=lambda r: 0 if (r.extra.get("language") or "").strip().lower() == "english" else 1,
    )

# Every upstream archive.org / librivox.org call follows redirects.
# archive.org 301s http→https and also redirects between CDN edges.
_REDIRECT_KW = {"follow_redirects": True}


class LibrivoxProvider:
    """Built-in, credential-free provider. Implements MediaProvider."""

    name = "librivox"

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    # --- Identity ----------------------------------------------------------

    async def ping(self, base_url: str) -> ProviderInfo | None:
        """Always reports as initialized — this is a built-in, not a remote.

        The ``base_url`` arg is ignored; kept for Protocol conformance.
        Returning a real ProviderInfo lets the Media Servers UI render
        the "Free Public Domain Library" card as a first-class tile.
        """
        return ProviderInfo(
            provider=self.name,
            base_url="https://librivox.org",
            server_name="LibriVox",
            version="builtin",
            is_initialized=True,
        )

    async def login(self, base_url: str, username: str, password: str) -> str:
        # LibriVox has no account system. If this is ever called, the
        # caller is mistakenly treating the built-in like a user-owned
        # server — surface that loudly rather than returning "".
        raise ValueError(
            "LibriVox is a built-in free library; it does not use login",
        )

    async def verify_token(self, base_url: str, token: str) -> bool:
        # Always valid — there's no token to expire.
        return True

    # --- Catalog sync (intentionally empty) --------------------------------

    async def fetch_catalog(
        self, base_url: str, token: str,
    ) -> list[CatalogItem]:
        """LibriVox is browse-only; never synced into file_index wholesale."""
        return []

    async def fetch_progress(
        self, base_url: str, token: str,
    ) -> dict[str, dict]:
        return {}

    async def push_progress(
        self,
        base_url: str,
        token: str,
        *,
        external_id: str,
        current_time_s: float,
        duration_s: float,
        is_finished: bool = False,
    ) -> bool:
        # Progress is owned by Augmentum; file_index already stores it.
        # Return True so the caller records the local write as successful.
        return True

    # --- URL construction --------------------------------------------------

    def build_stream_url(
        self, base_url: str, stream_path: str, token: str,
    ) -> str:
        """Resolve one chapter MP3 to its absolute archive.org URL.

        The route layer constructs ``stream_path`` as
        ``{archive_identifier}/{filename}`` (no leading slash) for the
        requested chapter and calls here. We just stitch on the archive
        download base.
        """
        if not stream_path:
            return ""
        path = stream_path.lstrip("/")
        return f"{ARCHIVE_DOWNLOAD}/{path}"

    def build_cover_url(
        self, base_url: str, external_id: str, token: str,
    ) -> str:
        """Archive.org's services/img endpoint returns a scaled cover.

        ``external_id`` is the archive identifier (e.g.
        ``pride_and_prejudice_0711_librivox``). archive.org picks a
        reasonable default image for every item.
        """
        if not external_id:
            return ""
        return f"{ARCHIVE_COVER}/{external_id}"

    # --- Browse (live catalog search) --------------------------------------

    async def browse(
        self,
        *,
        query: str = "",
        category: str = "",
        page: int = 1,
        page_size: int = 24,
    ) -> list[BrowseResult]:
        """Query archive.org's advanced search over the LibriVox collection.

        Why not LibriVox's own feed: the feed API's ``search=`` is
        silently inert — every term returns the catalogue's default
        front page. ``genre_id=`` is ignored and ``genre=<name>`` 500s.
        Verified live 2026-04-20. archive.org runs a proper search
        engine over the same audio recordings and respects:
            - ``subject:<name>``  (genre chips — Horror, Mystery, …)
            - free-text          (title / author / description)
            - ``page``           (1-indexed)

        The returned rows carry enough metadata to render the grid
        (title, author, description, runtime, cover). Per-chapter
        reader credits are not part of archive.org's search index;
        pin-time falls back to :func:`normalise_details_to_catalog`
        (archive.org /metadata) for chapter data.
        """
        category = (category or "").strip()
        q = (query or "").strip()
        clauses = [f"collection:{ARCHIVE_COLLECTION}"]
        if category:
            # Lowercase subject query — archive.org's subject index is
            # case-insensitive but lowercasing keeps the cache key stable.
            clauses.append(f'subject:"{category.lower()}"')
        if q:
            # Parenthesise the user term so it AND-combines cleanly with
            # the collection/subject clauses without quote-escaping games.
            clauses.append(f"({q})")
        params: list[tuple[str, Any]] = [
            ("q", " AND ".join(clauses)),
            ("fl[]", "identifier"),
            ("fl[]", "title"),
            ("fl[]", "creator"),
            ("fl[]", "description"),
            ("fl[]", "subject"),
            ("fl[]", "runtime"),
            ("fl[]", "language"),
            ("rows", str(max(1, page_size))),
            ("page", str(max(1, page))),
            ("output", "json"),
        ]

        try:
            resp = await self._http.get(
                ARCHIVE_SEARCH, params=params, timeout=_TIMEOUT_S, **_REDIRECT_KW,
            )
        except Exception as exc:
            log.warning("librivox_browse_failed", query=q, category=category, error=str(exc))
            return []
        if resp.status_code != 200:
            log.warning(
                "librivox_browse_non_200",
                status=resp.status_code, query=q, category=category,
            )
            return []

        try:
            body = resp.json()
        except Exception as exc:
            log.warning("librivox_browse_invalid_json", error=str(exc))
            return []

        docs = (body.get("response") or {}).get("docs") or []
        if not isinstance(docs, list):
            return []

        out: list[BrowseResult] = []
        for raw in docs:
            result = _browse_result_from_archive_doc(raw)
            if result is not None:
                out.append(result)
        return _english_first(out)

    async def recently_added(
        self, *, days: int = 30, limit: int = 24,
    ) -> list[BrowseResult]:
        """Books cataloged on LibriVox in the last ``days`` days.

        Backed by LibriVox's ``?since=<unix_ts>`` feed param — the only
        sort-by-freshness path upstream. archive.org's search index
        doesn't expose cataloging date, so this is the LibriVox-only
        route that makes "what's new" possible.

        Uses the LibriVox feed directly (not archive.org search) because:
            1. ``since=`` works only on the feed.
            2. The feed returns the full extended record in one call, so
               we get coverart_jpg / sections / authors for free — good
               shape for rendering the landing grid and, if the user
               pins, for skipping the extra feed call at pin time.

        Returns [] on upstream failure — the route falls back to the
        regular browse path in that case so the overlay never opens empty.
        """
        if days < 1:
            days = 1
        since_ts = int(time.time()) - days * 86400
        params: dict[str, Any] = {
            "format":   "json",
            "since":    since_ts,
            "extended": 1,
            "coverart": 1,
            "limit":    max(1, min(100, limit)),   # LibriVox caps at 100
        }
        try:
            resp = await self._http.get(
                LIBRIVOX_API, params=params, timeout=_TIMEOUT_S, **_REDIRECT_KW,
            )
        except Exception as exc:
            log.warning("librivox_recent_failed", error=str(exc))
            return []
        if resp.status_code != 200:
            log.warning("librivox_recent_non_200", status=resp.status_code)
            return []
        try:
            body = resp.json()
        except Exception as exc:
            log.warning("librivox_recent_invalid_json", error=str(exc))
            return []
        raw_books = body.get("books") or []
        if not isinstance(raw_books, list):
            return []
        out: list[BrowseResult] = []
        for raw in raw_books:
            result = _browse_result_from_feed(raw)
            if result is not None:
                out.append(result)
        return _english_first(out)

    async def fetch_book_by_id(self, librivox_id: str) -> dict | None:
        """Fetch the full LibriVox record for one book (sections + readers).

        Used at pin time to get the per-chapter data we need to build a
        playable library row. Returns None on any failure so the caller
        can fall back to the archive.org /metadata path.

        Why this over archive.org /metadata: LibriVox's sections[] carries
        per-chapter readers, chapter-range-aware titles ("Chapters 1-3"),
        and a canonical listen_url pointing to the 64kbps MP3 — one call,
        authoritative data. archive.org /metadata works but requires
        filtering and doesn't know about readers.
        """
        if not librivox_id:
            return None
        try:
            resp = await self._http.get(
                LIBRIVOX_API,
                params={
                    "format":   "json",
                    "id":       str(librivox_id),
                    "extended": 1,
                    # coverart=1 adds coverart_jpg / coverart_thumbnail /
                    # coverart_pdf when LibriVox has produced cover art for
                    # the book. Free upgrade over archive.org's services/img
                    # guess — same round trip.
                    "coverart": 1,
                },
                timeout=_DETAILS_TIMEOUT_S, **_REDIRECT_KW,
            )
        except Exception as exc:
            log.warning(
                "librivox_fetch_by_id_failed",
                librivox_id=librivox_id, error=str(exc),
            )
            return None
        if resp.status_code != 200:
            return None
        try:
            body = resp.json()
        except Exception:
            return None
        books = body.get("books") or []
        if not isinstance(books, list) or not books:
            return None
        return books[0]

    # --- Details (called by /api/media/details and /api/media/pin) ---------

    async def fetch_item_details(
        self,
        base_url: str,
        token: str,
        *,
        external_id: str,
    ) -> dict | None:
        """Fetch archive.org metadata and normalise into our detail shape.

        ``external_id`` is the archive identifier. We use the archive.org
        Metadata API (not the LibriVox per-book endpoint) because:
            1. It's authoritative for the file list (chapter MP3s + order)
            2. It carries per-file duration, which LibriVox's feed omits
            3. It's one call, not N

        Returns None on upstream failure so the route layer falls back to
        cached source_metadata without crashing.
        """
        if not external_id:
            return None
        url = f"{ARCHIVE_METADATA}/{external_id}"
        try:
            resp = await self._http.get(
                url, timeout=_DETAILS_TIMEOUT_S, **_REDIRECT_KW,
            )
        except Exception as exc:
            log.warning(
                "librivox_details_failed",
                external_id=external_id, error=str(exc),
            )
            return None
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except Exception as exc:
            log.warning(
                "librivox_details_invalid_json",
                external_id=external_id, error=str(exc),
            )
            return None


# --- Normalisers -----------------------------------------------------------


# Archive.org mirrors LibriVox descriptions, which always open with a stock
# preamble ("LibriVox recording of <Title> by <Author>. Read in <lang> by
# <Reader>.") before the actual blurb. Strip the lead-in so the card shows
# meaningful prose first. Match is anchored to the start and non-greedy so
# unusual descriptions (missing preamble, multiple sentences) pass through
# unchanged.
_LV_DESCRIPTION_PREAMBLE = re.compile(
    r"^LibriVox recording of [^.]+?\.(?:\s+Read (?:in [^.]+?by|by)[^.]+?\.)?",
    re.IGNORECASE,
)


def _strip_librivox_preamble(desc: str) -> str:
    """Drop the stock 'LibriVox recording of X by Y. Read by …' lead-in.

    Archive.org's description field mirrors LibriVox's own, which always
    starts with this preamble. Keeping it dominates the card; users
    already know the collection is LibriVox. We only strip when the
    preamble is clearly present so edge-case descriptions aren't mangled.
    """
    if not desc:
        return ""
    # Common shape: "LibriVox recording of <title> by <author>. Read in
    # <lang> by <reader>[. <actual summary…>]"
    m = _LV_DESCRIPTION_PREAMBLE.match(desc)
    if not m:
        return desc.strip()
    return desc[m.end():].lstrip(" .:-—")


def _parse_runtime(raw: object) -> int:
    """archive.org 'runtime' is 'H:MM:SS' or similar. Returns milliseconds.

    Handles the upstream data-cleanliness quirk where some rows store
    ``HH:MM.SS`` (period instead of the second colon) for what should be
    ``HH:MM:SS`` — e.g. ``"16:31.09"`` for a 16h 31m 9s book. Naive
    split-on-colon parses that as 16m 31.09s, a 60× undercount that
    makes multi-hour books look like short tracks on the browse card.
    We promote the period to a colon only when exactly one of each is
    present, leaving pure-decimal seconds (``"90.5"``) and
    sub-second-precision forms (``"1:30:45.5"``) untouched.
    """
    if raw is None:
        return 0
    s = str(raw).strip()
    if not s:
        return 0
    if s.count(":") == 1 and s.count(".") == 1:
        s = s.replace(".", ":")
    try:
        parts = [float(p) for p in s.split(":")]
    except ValueError:
        return 0
    if len(parts) == 1:
        seconds = parts[0]
    elif len(parts) == 2:
        seconds = parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        return 0
    return int(seconds * 1000)


# Archive.org subject lists include catalogue tags ('librivox', 'audiobook',
# 'audiobooks') that aren't meaningful genres to a listener. Filter them
# out so chip display shows only topical subjects (horror, mystery, …).
_SUBJECT_NOISE = frozenset({"librivox", "audiobook", "audiobooks", "audio"})


def _browse_result_from_archive_doc(raw: dict) -> BrowseResult | None:
    """archive.org Advanced Search doc → BrowseResult.

    archive.org doesn't return the LibriVox feed's per-author life dates,
    chapter count, or external links — those fields land empty here and
    the detail panel hides blank rows. Pin-time pulls the authoritative
    metadata from archive.org /metadata for chapter data.
    """
    if not isinstance(raw, dict):
        return None
    identifier = (raw.get("identifier") or "").strip()
    if not identifier:
        return None
    title = (raw.get("title") or "").strip() or "Untitled"
    # `creator` can be a string or a list — normalise to a comma-joined string.
    creator = raw.get("creator") or ""
    if isinstance(creator, list):
        author = ", ".join(c for c in (str(x).strip() for x in creator) if c)
    else:
        author = str(creator).strip()
    description = _strip_librivox_preamble(str(raw.get("description") or ""))
    duration_ms = _parse_runtime(raw.get("runtime"))

    subjects = raw.get("subject") or []
    if not isinstance(subjects, list):
        subjects = [subjects]
    genres = [
        str(s).strip() for s in subjects
        if isinstance(s, (str, int, float)) and str(s).strip().lower() not in _SUBJECT_NOISE
    ]

    language = str(raw.get("language") or "").strip()
    # archive.org uses ISO 639-2 codes ("eng", "fre", …). Expand the common
    # ones for display; leave unknowns untouched so we never hide info.
    lang_expand = {
        "eng": "English", "fre": "French", "fra": "French", "ger": "German",
        "deu": "German", "spa": "Spanish", "ita": "Italian", "rus": "Russian",
        "por": "Portuguese", "dut": "Dutch", "nld": "Dutch", "jpn": "Japanese",
        "chi": "Chinese", "zho": "Chinese", "ara": "Arabic", "pol": "Polish",
        "swe": "Swedish", "dan": "Danish", "nor": "Norwegian", "fin": "Finnish",
        "lat": "Latin", "grc": "Ancient Greek", "heb": "Hebrew",
    }
    language = lang_expand.get(language.lower(), language)

    return BrowseResult(
        external_id=identifier,
        name=title,
        author=author,
        narrator="",   # not in archive.org's search index; pin-time fills it
        duration_ms=duration_ms,
        cover_url=f"{ARCHIVE_COVER}/{identifier}",
        description=description,
        license="public-domain",
        extra={
            # librivox_id stays empty when browsing via archive.org —
            # pin-time will fall back to archive.org /metadata for
            # chapter data (no per-section reader credits). Same goes
            # for coverart_jpg / url_zip_file: those live only on the
            # LibriVox feed, so archive-browsed rows get the archive.org
            # services/img cover + no ZIP link until the user pins.
            "librivox_id":        "",
            "librivox_url":       "",
            "language":           language,
            "genres":             genres,
            "num_sections":       0,
            "copyright_year":     "",
            "totaltime":          "",
            "url_text_source":    "",
            "url_project":        "",
            "url_rss":            "",
            "url_other":          "",
            "url_zip_file":       "",
            "coverart_jpg":       "",
            "coverart_thumbnail": "",
            "authors":            [],
            "translators":        [],
        },
    )


def _browse_result_from_feed(raw: dict) -> BrowseResult | None:
    """One LibriVox feed 'book' → BrowseResult.

    Returns None when we can't derive an archive identifier — those rows
    can't be played through our streaming proxy, so omitting them from
    browse results is the right call.
    """
    if not isinstance(raw, dict):
        return None
    archive_id = _archive_identifier(raw)
    if not archive_id:
        return None

    title = (raw.get("title") or "").strip() or "Untitled"
    authors = raw.get("authors") or []
    author = _join_authors(authors)

    try:
        duration_ms = int(float(raw.get("totaltimesecs") or 0) * 1000)
    except (TypeError, ValueError):
        duration_ms = 0

    # Descriptions arrive with HTML markup. Strip to plain text once here so
    # every downstream consumer (detail panel, search snippet, etc.) gets the
    # same clean prose without per-surface sanitisation.
    description = _clean_html_text(raw.get("description") or "")
    genres_raw = raw.get("genres") or []
    genres: list[str] = []
    if isinstance(genres_raw, list):
        for g in genres_raw:
            if isinstance(g, dict):
                name = (g.get("name") or "").strip()
                if name:
                    genres.append(name)
            elif isinstance(g, str) and g.strip():
                genres.append(g.strip())

    # Structured authors keep life dates for "Jane Austen (1775–1817)" in
    # the detail panel; plain author string (above) still powers list cards.
    authors_detailed: list[dict] = []
    raw_authors = raw.get("authors") or []
    if isinstance(raw_authors, list):
        for a in raw_authors:
            if isinstance(a, dict):
                first = (a.get("first_name") or "").strip()
                last = (a.get("last_name") or "").strip()
                name = " ".join(p for p in (first, last) if p)
                if name:
                    authors_detailed.append({
                        "name": name,
                        "dob":  str(a.get("dob") or "").strip(),
                        "dod":  str(a.get("dod") or "").strip(),
                    })

    # Translators: same shape as authors on the feed; relevant for non-
    # English-origin works ("translated by Constance Garnett" is a
    # meaningful distinguisher between two translations of the same book).
    translators: list[dict] = []
    raw_translators = raw.get("translators") or []
    if isinstance(raw_translators, list):
        for t in raw_translators:
            if isinstance(t, dict):
                first = (t.get("first_name") or "").strip()
                last = (t.get("last_name") or "").strip()
                name = " ".join(p for p in (first, last) if p)
                if name:
                    translators.append({
                        "name": name,
                        "dob":  str(t.get("dob") or "").strip(),
                        "dod":  str(t.get("dod") or "").strip(),
                    })

    # LibriVox's own cover art when available (coverart=1). Prefer the
    # thumbnail for the grid view; the full JPG is reserved for the
    # detail panel. Both live on archive.org CDN, so no extra proxy work.
    coverart_jpg = str(raw.get("coverart_jpg") or "").strip()
    coverart_thumb = str(raw.get("coverart_thumbnail") or "").strip()
    # Prefer thumbnail for the browse grid: ~20KB vs ~200KB for the full
    # JPG and a 20-card page would otherwise pull 4MB of images on open.
    grid_cover = coverart_thumb or coverart_jpg or f"{ARCHIVE_COVER}/{archive_id}"

    return BrowseResult(
        external_id=archive_id,
        name=title,
        author=author,
        narrator="",   # LibriVox feed doesn't expose reader list per-book
        duration_ms=duration_ms,
        cover_url=grid_cover,
        description=description,
        license="public-domain",
        extra={
            "librivox_id":     str(raw.get("id") or ""),
            "librivox_url":    raw.get("url_librivox") or "",
            "language":        raw.get("language") or "",
            "genres":          genres,
            "num_sections":    int(raw.get("num_sections") or 0),
            "copyright_year":  str(raw.get("copyright_year") or ""),
            # Human-formatted "13:06:44" — avoids recomputing from seconds.
            "totaltime":       str(raw.get("totaltime") or ""),
            # Optional enrichment links. Stored even when empty so downstream
            # code can test presence with a simple truthy check.
            "url_text_source": str(raw.get("url_text_source") or ""),
            "url_project":     str(raw.get("url_project") or ""),
            "url_rss":         str(raw.get("url_rss") or ""),
            "url_other":       str(raw.get("url_other") or ""),
            "url_zip_file":    str(raw.get("url_zip_file") or ""),
            # LibriVox's own cover art (if produced for this book). Empty
            # when LibriVox didn't do a custom cover — caller falls back
            # to cover_url above (archive.org services/img).
            "coverart_jpg":       coverart_jpg,
            "coverart_thumbnail": coverart_thumb,
            # Full author records (with DOB/DOD) for the detail panel.
            "authors":         authors_detailed,
            "translators":     translators,
        },
    )


def _archive_identifier(raw: dict) -> str:
    """Pull the archive.org identifier out of a LibriVox feed row.

    Prefers ``url_iarchive`` (canonical details page). Falls back to
    ``url_zip_file`` (the real key name — ``url_zip`` doesn't exist in
    the feed schema, verified live 2026-04-20). Also handles archive.org
    ``/compress/<id>`` URLs which is what url_zip_file actually uses.
    Returns '' if no identifier is derivable.
    """
    for key in ("url_iarchive", "url_zip_file", "url_zip"):
        url = raw.get(key) or ""
        if not url:
            continue
        # Recognised anchors:
        #   /details/<id>           — iarchive
        #   /download/<id>/foo.mp3  — direct download
        #   /compress/<id>          — url_zip_file (compress endpoint)
        parts = [p for p in url.split("/") if p]
        for anchor in ("details", "download", "compress"):
            if anchor in parts:
                idx = parts.index(anchor)
                if idx + 1 < len(parts):
                    candidate = parts[idx + 1]
                    # Strip any trailing query string.
                    candidate = candidate.split("?", 1)[0]
                    if candidate:
                        return candidate
    return ""


def _join_authors(authors: list) -> str:
    """ABS does the same trick — flatten objects to a comma-joined string."""
    if not isinstance(authors, list):
        return ""
    names: list[str] = []
    for a in authors:
        if isinstance(a, dict):
            first = (a.get("first_name") or "").strip()
            last = (a.get("last_name") or "").strip()
            joined = " ".join(p for p in (first, last) if p)
            if joined:
                names.append(joined)
        elif isinstance(a, str):
            s = a.strip()
            if s:
                names.append(s)
    return ", ".join(names)


def normalise_librivox_sections(
    *,
    librivox_book: dict,
    browse_result: BrowseResult,
) -> dict:
    """Build a file_index payload directly from LibriVox's sections[].

    Preferred pin-time path. sections[] on the extended feed carries:
        - section_number (for ordering)
        - title (chapter or chapter-range-aware, e.g. "Chapters 1-3")
        - listen_url (absolute archive.org MP3 URL we can parse for a
          filename, so the stream route doesn't need a separate metadata
          round-trip at playback time)
        - playtime (seconds, authoritative)
        - readers[] ([{reader_id, display_name}])

    Returns the same shape as normalise_details_to_catalog so the pin
    route doesn't care which path produced it.
    """
    sections = librivox_book.get("sections") or []
    if not isinstance(sections, list):
        sections = []

    # Sort by numeric section_number to guarantee chapter order even if
    # upstream ships out-of-order (rare but harmless to defend against).
    def _section_num(s: dict) -> int:
        try:
            return int(str(s.get("section_number") or 0))
        except (TypeError, ValueError):
            return 0

    sections = sorted(sections, key=_section_num)

    audio_files: list[dict] = []
    chapters: list[dict] = []
    total_duration_s = 0.0
    narrator_set: list[str] = []
    narrator_seen: set[str] = set()
    cursor = 0.0

    for i, sec in enumerate(sections):
        if not isinstance(sec, dict):
            continue
        listen_url = sec.get("listen_url") or ""
        filename = _filename_from_listen_url(listen_url)
        if not filename:
            continue
        try:
            playtime_s = float(sec.get("playtime") or 0)
        except (TypeError, ValueError):
            playtime_s = 0.0
        title = (sec.get("title") or f"Section {sec.get('section_number') or i + 1}").strip()

        readers = sec.get("readers") or []
        reader_names: list[str] = []
        if isinstance(readers, list):
            for r in readers:
                if isinstance(r, dict):
                    name = (r.get("display_name") or "").strip()
                    if name:
                        reader_names.append(name)
                        if name not in narrator_seen:
                            narrator_seen.add(name)
                            narrator_set.append(name)

        audio_files.append({
            "name":       filename,
            "duration_s": playtime_s,
            "size":       0,   # LibriVox feed doesn't expose byte size
            "title":      title,
            "track":      _section_num(sec) or (i + 1),
        })

        chapters.append({
            "title":      title,
            "start":      cursor,
            "end":        cursor + playtime_s,
            "file_index": len(audio_files) - 1,
            "narrators":  reader_names,
        })
        total_duration_s += playtime_s
        cursor += playtime_s

    # Book-level narrator: de-duped union of per-section readers. Falls
    # back to what the browse result carried (usually empty for LibriVox
    # because the feed strips readers at the top level) or a sentinel.
    if narrator_set:
        if len(narrator_set) == 1:
            narrator = narrator_set[0]
        elif len(narrator_set) <= 3:
            narrator = ", ".join(narrator_set)
        else:
            narrator = f"{narrator_set[0]} + {len(narrator_set) - 1} others"
    else:
        narrator = browse_result.narrator or ""

    duration_ms = (
        int(total_duration_s * 1000) if total_duration_s > 0
        else browse_result.duration_ms
    )

    return {
        "archive_identifier": browse_result.external_id,
        "audio_files":        audio_files,
        "chapters":           chapters,
        "duration_ms":        duration_ms,
        "size_bytes":         0,   # no size data from sections path
        "narrator":           narrator,
        "narrators":          narrator_set,   # full deduped list for UI
        "genres":             browse_result.extra.get("genres") or [],
        "language":           browse_result.extra.get("language", ""),
        "librivox_url":       browse_result.extra.get("librivox_url", ""),
        "librivox_id":        browse_result.extra.get("librivox_id", ""),
        "authors_detailed":   browse_result.extra.get("authors") or [],
        "translators":        browse_result.extra.get("translators") or [],
        "copyright_year":     browse_result.extra.get("copyright_year", ""),
        "totaltime":          browse_result.extra.get("totaltime", ""),
        "url_text_source":    browse_result.extra.get("url_text_source", ""),
        "url_project":        browse_result.extra.get("url_project", ""),
        "url_rss":            browse_result.extra.get("url_rss", ""),
        "url_other":          browse_result.extra.get("url_other", ""),
        "url_zip_file":       browse_result.extra.get("url_zip_file", ""),
        "coverart_jpg":       browse_result.extra.get("coverart_jpg", ""),
        "coverart_thumbnail": browse_result.extra.get("coverart_thumbnail", ""),
    }


def _filename_from_listen_url(url: str) -> str:
    """Extract the MP3 filename from a section's listen_url.

    Shape: https://www.archive.org/download/<identifier>/<filename>
    Returns '' if the URL doesn't match (e.g. section stored as a
    relative path or missing entirely).
    """
    if not url:
        return ""
    # Split on /download/ and take the tail; then the last segment.
    if "/download/" not in url:
        return ""
    tail = url.split("/download/", 1)[1]
    parts = tail.split("/", 1)
    if len(parts) < 2:
        return ""
    filename = parts[1]
    # Strip any query string (defensive — LibriVox doesn't add one, but
    # a CDN rewrite might).
    return filename.split("?", 1)[0]


def normalise_details_to_catalog(
    *,
    archive_meta: dict,
    browse_result: BrowseResult,
) -> dict:
    """Fallback: merge archive.org /metadata + browse hit into a file_index row.

    Used only when normalise_librivox_sections fails (e.g. LibriVox feed
    didn't return sections for a book, or the pin flow was called
    without a librivox_id). Returns the same shape so callers don't care
    which path succeeded.
    """
    files = archive_meta.get("files") or []
    if not isinstance(files, list):
        files = []

    audio_files: list[dict] = []
    total_duration_s = 0.0
    total_size = 0

    for f in files:
        if not isinstance(f, dict):
            continue
        fmt = (f.get("format") or "").lower()
        # Prefer 64 kbps MP3 (LibriVox's canonical chapter format). Skip
        # the whole-book zip, the text source, the preview, the
        # community-submitted metadata, and the 128 kbps re-encode to
        # avoid doubling every chapter.
        if "mp3" not in fmt or "vbr" in fmt or "128" in fmt:
            continue
        name = f.get("name") or ""
        if not name:
            continue
        try:
            length_s = _parse_length(f.get("length"))
        except Exception:
            length_s = 0.0
        try:
            size = int(f.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        audio_files.append({
            "name":        name,
            "duration_s":  length_s,
            "size":        size,
            "title":       (f.get("title") or "").strip() or name,
            "track":       _safe_int(f.get("track")),
        })
        total_duration_s += length_s
        total_size += size

    # Sort by track number when present, else by filename (archive.org
    # uses zero-padded numeric prefixes, so lexical == track order).
    audio_files.sort(key=lambda a: (a["track"] or 10_000, a["name"]))

    chapters: list[dict] = []
    cursor = 0.0
    for i, af in enumerate(audio_files):
        start = cursor
        end = cursor + af["duration_s"]
        chapters.append({
            "title":      af["title"],
            "start":      start,
            "end":        end,
            "file_index": i,
        })
        cursor = end

    # Fallback duration to browse result's total if archive.org files were
    # missing durations (happens on a minority of items).
    duration_ms = (
        int(total_duration_s * 1000) if total_duration_s > 0
        else browse_result.duration_ms
    )

    arch_metadata = archive_meta.get("metadata") or {}
    narrator = (
        arch_metadata.get("creator") if isinstance(arch_metadata.get("creator"), str)
        else ""
    )

    return {
        "archive_identifier": browse_result.external_id,
        "audio_files":        audio_files,
        "chapters":           chapters,
        "duration_ms":        duration_ms,
        "size_bytes":         total_size,
        "narrator":           (narrator or browse_result.narrator or "").strip(),
        "narrators":          [],   # archive.org path has no per-section reader info
        "genres":             browse_result.extra.get("genres") or [],
        "language":           browse_result.extra.get("language", ""),
        "librivox_url":       browse_result.extra.get("librivox_url", ""),
        "librivox_id":        browse_result.extra.get("librivox_id", ""),
        # Enrichment fields: same keys as the sections path so downstream
        # doesn't care which produced it. All may be empty on this fallback.
        "authors_detailed":   browse_result.extra.get("authors") or [],
        "translators":        browse_result.extra.get("translators") or [],
        "copyright_year":     browse_result.extra.get("copyright_year", ""),
        "totaltime":          browse_result.extra.get("totaltime", ""),
        "url_text_source":    browse_result.extra.get("url_text_source", ""),
        "url_project":        browse_result.extra.get("url_project", ""),
        "url_rss":            browse_result.extra.get("url_rss", ""),
        "url_other":          browse_result.extra.get("url_other", ""),
        "url_zip_file":       browse_result.extra.get("url_zip_file", ""),
        "coverart_jpg":       browse_result.extra.get("coverart_jpg", ""),
        "coverart_thumbnail": browse_result.extra.get("coverart_thumbnail", ""),
    }


def _parse_length(raw: object) -> float:
    """archive.org returns file length as either seconds (float str) or MM:SS.

    Returns 0.0 on anything unparseable.
    """
    if raw is None:
        return 0.0
    s = str(raw).strip()
    if not s:
        return 0.0
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:
                m, sec = parts
                return float(m) * 60 + float(sec)
            if len(parts) == 3:
                h, m, sec = parts
                return float(h) * 3600 + float(m) * 60 + float(sec)
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _safe_int(raw: object) -> int | None:
    if raw is None:
        return None
    try:
        return int(str(raw).split("/")[0])   # "3/12" → 3
    except (ValueError, TypeError):
        return None
