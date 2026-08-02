"""Suwayomi HTTP client (GraphQL transport).

Docs: https://suwayomi.org/. Default port 4567. Suwayomi (formerly
Tachidesk-Server) runs Tachiyomi-style extensions as a server — your
"library" is the subset of manga you've curated through Suwayomi's own
UI, and the chapters come from whatever source extension backs each
manga (MangaDex, Mangakakalot, Manganelo, etc.).

Suwayomi **v2.x** moved the data API to GraphQL at ``/api/graphql``.
The old ``/api/v1/library``, ``/api/v1/manga/*/chapters`` REST endpoints
no longer exist — the SPA catch-all now serves ``index.html`` for those
paths, which is how this migration got discovered in the first place
(200 OK with HTML body breaking ``resp.json()``).

Image serving is still REST — the Suwayomi reader itself builds URLs
like ``/api/v1/manga/{mangaId}/chapter/{sourceOrder}/page/{N}`` inside
the ``fetchChapterPages`` mutation, so those endpoints are stable and
the per-page delivery route in ``media_routes.py`` needs no change
beyond parsing the new external_id shape.

Architectural notes:

- **Auth**. HTTP Basic when enabled, empty when Suwayomi is running with
  ``authMode = none``. ``simple_login`` and ``ui_login`` (cookie/JWT)
  are **not** supported yet — users on those modes should switch their
  server to ``basic_auth`` or ``none``. Detecting auth mismatch: our
  ``_gql`` helper treats an HTML response body as a misconfiguration
  and raises with the first 200 chars of the body, so future cases
  like "user points us at a non-GraphQL endpoint" are self-diagnosing
  instead of failing as cryptic JSON errors.

- **Catalog granularity**: one ``CatalogItem`` per **chapter**, not per
  manga. A user with 20 manga × 200 chapters = 4000 rows in file_index
  — fine at scale, on par with any real Komga library.

- **External ID shape**: ``{manga_id}.{source_order}.{chapter_db_id}``
  (three parts). Changed from the two-part ``{manga_id}.{source_order}``
  we used against the REST API, because ``updateChapter`` mutation
  requires the chapter's DB ``id`` — which GraphQL exposes but the
  old REST chapter lookup made implicit. Parse first two for image
  URLs, third for the update mutation. Older 2-part rows (if any
  survive a migration) degrade gracefully: images still work,
  progress writes skip with a debug log.

- **Progress**: Suwayomi stores ``lastPageRead`` (int, 0-indexed) and
  ``isRead`` (bool) per chapter. Maps cleanly to the audio-shaped
  protocol: ``current_time_s = last_page_read``,
  ``duration_s = page_count``.

- **Live-fetched chapters**: if a chapter isn't cached locally, Suwayomi
  live-fetches from the source's website. If the source is down,
  ``/page`` returns 5xx — the reader surfaces the error and lets the
  user retry.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from augmentum.media.providers.base import CatalogItem, ProviderInfo
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)


# --- Timeouts ---------------------------------------------------------------
# Short for interactive paths (ping, verify, progress push), longer for the
# library walk which can involve N+1 source-side chapter hydration on servers
# where the cache is cold.

_TIMEOUT_S = 10.0
_LOGIN_TIMEOUT_S = 15.0
_CATALOG_TIMEOUT_S = 60.0

# First-page caps — library queries are paginated via ``first: Int``, so we
# bound them to avoid an unbounded-response worst case. Observed real-world
# libraries run ~500 series with 20–80 chapters each (~20k chapters total);
# 5000 series is well into "this person hoards every manga ever" territory
# and 100k chapters covers the progress walk for such a library comfortably.
# If someone hits these caps, they'll see a warning log and we can switch
# to cursor-based pagination — deferred until there's a real user who needs
# it, because paginating adds one round-trip per 500-manga page.
_MAX_LIBRARY_MANGAS = 5000
_MAX_PROGRESS_CHAPTERS = 100000

# Follow redirects — Suwayomi behind a reverse proxy is a common deployment
# and HTTPS redirects are standard there.
_REDIRECT_KW = {"follow_redirects": True}

_GRAPHQL_PATH = "/api/graphql"


# --- GraphQL documents ------------------------------------------------------
# Kept as module-level constants (not string-builders) so test mocks can
# match them exactly — a typo in either side blows up obviously rather than
# silently sending a malformed query.

_ABOUT_QUERY = "{ aboutServer { name version } }"

# Minimal auth probe — requires @RequireAuth on the server side, so a wrong
# Basic token here produces a 401 response whereas ``aboutServer`` is public
# and always 200s. We use this for login validation specifically.
_AUTH_PROBE_QUERY = (
    "{ mangas(condition: { inLibrary: true }, first: 1) { nodes { id } } }"
)

_LIBRARY_QUERY = """
query Library($first: Int!) {
  mangas(condition: { inLibrary: true }, first: $first) {
    nodes {
      id
      title
      author
      artist
      description
      status
      genre
      sourceId
      realUrl
      thumbnailUrl
      inLibraryAt
      lastFetchedAt
      chapters {
        nodes {
          id
          mangaId
          name
          sourceOrder
          chapterNumber
          scanlator
          pageCount
          isRead
          isBookmarked
          isDownloaded
          lastPageRead
          uploadDate
          realUrl
          fetchedAt
        }
      }
    }
  }
}
"""

# Server-side filtered progress walk: returns only chapters the user has
# actually touched. Much cheaper than re-walking the library for progress.
_PROGRESS_QUERY = """
query Progress($first: Int!) {
  chapters(
    filter: {
      inLibrary: { equalTo: true },
      or: [
        { isRead: { equalTo: true } },
        { lastPageRead: { greaterThan: 0 } }
      ]
    },
    first: $first
  ) {
    nodes {
      id
      mangaId
      sourceOrder
      isRead
      lastPageRead
      pageCount
    }
  }
}
"""

_ITEM_DETAILS_QUERY = """
query ItemDetails($mangaId: Int!) {
  manga(id: $mangaId) {
    id
    title
    author
    artist
    description
    status
    genre
    sourceId
    realUrl
    thumbnailUrl
    inLibraryAt
    lastFetchedAt
    chapters {
      nodes {
        id
        mangaId
        name
        sourceOrder
        chapterNumber
        scanlator
        pageCount
        isRead
        isBookmarked
        isDownloaded
        lastPageRead
        uploadDate
        realUrl
        fetchedAt
      }
    }
  }
}
"""

_UPDATE_CHAPTER_MUTATION = """
mutation UpdateChapter($input: UpdateChapterInput!) {
  updateChapter(input: $input) {
    chapter { id lastPageRead isRead }
  }
}
"""

# Must be called once per chapter before its /api/v1/manga/X/chapter/Y/page/N
# image endpoints return 200. Suwayomi's upstream extension (MangaDex /
# whatever source backs the manga) hasn't been asked to enumerate pages
# until this fires — without it, the page REST endpoints 404 because the
# page count is 0 in the local DB and the scraper never ran.
_FETCH_CHAPTER_PAGES_MUTATION = """
mutation FetchChapterPages($input: FetchChapterPagesInput!) {
  fetchChapterPages(input: $input) {
    pages
    chapter { id pageCount }
  }
}
"""


# --- Auth header helpers ----------------------------------------------------

def _encode_basic(username: str, password: str) -> str:
    """Return ``base64(user:pass)`` for the HTTP Basic header."""
    raw = f"{username}:{password}".encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _auth_headers(token: str) -> dict[str, str]:
    """Return the auth header dict, or empty if auth is disabled."""
    if not token:
        return {}
    return {"Authorization": f"Basic {token}"}


class SuwayomiProvider:
    name = "suwayomi"

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    # --- GraphQL transport -------------------------------------------------

    async def _gql(
        self,
        base_url: str,
        token: str,
        query: str,
        *,
        variables: dict[str, Any] | None = None,
        timeout: float = _TIMEOUT_S,
    ) -> dict[str, Any]:
        """POST a GraphQL document. Return ``data`` dict or raise.

        Failure modes are distinct and translated into the right exception
        types so callers can act on them:

        - **401/403** → ``ValueError`` ("Authentication rejected"). Login
          paths turn this into the user-visible "Invalid username or password".
        - **non-200** → ``RuntimeError`` with the status code.
        - **HTML body on 200** → ``RuntimeError`` with the first 200 chars
          of the body. This catches "pointed at the SPA catch-all because
          you're on a pre-GraphQL Suwayomi" and similar misconfigurations
          loudly instead of as an opaque JSONDecodeError.
        - **GraphQL ``errors`` array** → ``RuntimeError`` with the first
          error's message. GraphQL errors are usually very specific
          ("Cannot query field 'foo' on type 'Bar'") so one hop is enough.
        """
        url = f"{base_url.rstrip('/')}{_GRAPHQL_PATH}"
        body: dict[str, Any] = {"query": query}
        if variables is not None:
            body["variables"] = variables

        resp = await self._http.post(
            url,
            json=body,
            headers={"Content-Type": "application/json", **_auth_headers(token)},
            timeout=timeout,
            **_REDIRECT_KW,
        )

        if resp.status_code in (401, 403):
            raise ValueError("Authentication rejected by Suwayomi")
        if resp.status_code != 200:
            raise RuntimeError(f"GraphQL request failed: HTTP {resp.status_code}")

        # Guard against "200 OK but wrong body" — the signature of the
        # REST→GraphQL migration that caused this rewrite. A proper GraphQL
        # response always comes back as application/json; anything else
        # means something is terribly misrouted.
        content_type = (resp.headers.get("content-type") or "").lower()
        if "application/json" not in content_type:
            snippet = resp.text[:200].replace("\n", " ")
            raise RuntimeError(
                f"GraphQL endpoint returned non-JSON (content-type={content_type!r}). "
                f"Body starts: {snippet!r}. Is the server running a version that "
                f"supports GraphQL at {_GRAPHQL_PATH}?"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise RuntimeError(f"GraphQL response is not valid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Unexpected GraphQL payload shape: {type(payload).__name__}"
            )

        errors = payload.get("errors")
        if errors:
            first = errors[0] if isinstance(errors, list) and errors else {}
            msg = first.get("message") if isinstance(first, dict) else str(first)
            raise RuntimeError(f"GraphQL error: {msg}")

        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("GraphQL response missing `data` field")
        return data

    # --- Detection + auth --------------------------------------------------

    async def ping(self, base_url: str) -> ProviderInfo | None:
        """Unauthenticated ``aboutServer`` query.

        Confirms the endpoint speaks GraphQL AND is Suwayomi-shaped. Name
        has to mention "Suwayomi" or "Tachidesk" so the fingerprint doesn't
        accept arbitrary GraphQL servers.
        """
        url = base_url.rstrip("/")
        try:
            data = await self._gql(url, "", _ABOUT_QUERY)
        except Exception as exc:
            log.debug("suwayomi_ping_failed", base_url=url, error=str(exc))
            return None
        about = data.get("aboutServer")
        if not isinstance(about, dict):
            return None
        name = str(about.get("name") or "").lower()
        if "suwayomi" not in name and "tachidesk" not in name:
            return None
        return ProviderInfo(
            provider=self.name,
            base_url=url,
            version=str(about.get("version") or ""),
            server_name=about.get("name") or "",
        )

    async def login(
        self, base_url: str, username: str, password: str,
    ) -> str:
        """Return the Basic auth token after verifying it works.

        Three cases:

        1. Empty creds → ping-only validation, return "". Works for
           ``authMode = none`` deployments.
        2. Creds provided, server has auth enabled → Basic probe against
           the auth-gated ``mangas`` query. 401 → user-visible error;
           200 → return the token.
        3. Creds provided, server has auth disabled → Basic header is
           ignored upstream, probe returns 200, we return the token.
           Subsequent calls will send a harmless ignored header. Not
           ideal but harmless; users who change modes should re-add
           the server.
        """
        url = base_url.rstrip("/")

        if not username and not password:
            info = await self.ping(base_url)
            if info is None:
                raise RuntimeError("Suwayomi server unreachable or not recognized")
            return ""

        token = _encode_basic(username, password)
        try:
            await self._gql(
                url, token, _AUTH_PROBE_QUERY, timeout=_LOGIN_TIMEOUT_S,
            )
        except ValueError as exc:
            # _gql raises ValueError on 401/403 specifically
            raise ValueError("Invalid username or password") from exc
        return token

    async def verify_token(self, base_url: str, token: str) -> bool:
        url = base_url.rstrip("/")
        try:
            await self._gql(url, token, _AUTH_PROBE_QUERY)
            return True
        except Exception:
            return False

    # --- Catalog -----------------------------------------------------------

    async def fetch_catalog(
        self, base_url: str, token: str,
    ) -> list[CatalogItem]:
        """Pull every chapter in the user's hard library in one query.

        The GraphQL shape lets us request manga + nested chapters in one
        round-trip, replacing the REST API's per-manga chapter fetches.
        """
        url = base_url.rstrip("/")
        data = await self._gql(
            url, token, _LIBRARY_QUERY,
            variables={"first": _MAX_LIBRARY_MANGAS},
            timeout=_CATALOG_TIMEOUT_S,
        )

        mangas_node = (data.get("mangas") or {}).get("nodes")
        if not isinstance(mangas_node, list):
            log.warning(
                "suwayomi_library_unexpected_shape",
                type=type(mangas_node).__name__,
            )
            return []

        items: list[CatalogItem] = []
        manga_seen = 0
        for manga in mangas_node:
            if not isinstance(manga, dict):
                continue
            manga_id = manga.get("id")
            if manga_id is None:
                continue
            manga_seen += 1
            manga_payload = _build_manga_payload(manga)
            chapter_nodes = (manga.get("chapters") or {}).get("nodes") or []
            if not isinstance(chapter_nodes, list):
                continue
            for chapter in chapter_nodes:
                item = _chapter_to_catalog_item(chapter, manga_payload)
                if item is not None:
                    items.append(item)

        log.info(
            "suwayomi_catalog_fetched",
            items=len(items), manga=manga_seen,
        )
        # Hitting the cap means the user has more series than our one-shot
        # library query returns — some are silently clipped. Log loudly so
        # we know to wire cursor pagination. See _MAX_LIBRARY_MANGAS note.
        if manga_seen >= _MAX_LIBRARY_MANGAS:
            log.warning(
                "suwayomi_library_cap_hit",
                manga_seen=manga_seen,
                cap=_MAX_LIBRARY_MANGAS,
                message="Library may be clipped. Wire cursor pagination.",
            )
        return items

    # --- Streaming ---------------------------------------------------------

    def build_stream_url(
        self, base_url: str, stream_path: str, token: str,
    ) -> str:
        """Return the chapter-level URL. Per-page serving happens at the
        comic-page route, which builds ``/page/{N}`` under this root.
        """
        path = stream_path if stream_path.startswith("/") else f"/{stream_path}"
        return f"{base_url.rstrip('/')}{path}"

    def build_cover_url(
        self, base_url: str, external_id: str, token: str,
    ) -> str:
        """Return the manga thumbnail URL.

        ``external_id`` is 3-parted as of v2; we only need the manga_id
        at index 0. The thumbnail endpoint is still REST in v2 — Suwayomi
        kept it that way because GraphQL doesn't fit binary image delivery.
        """
        manga_id = external_id.split(".", 1)[0] if "." in external_id else external_id
        return f"{base_url.rstrip('/')}/api/v1/manga/{manga_id}/thumbnail"

    # --- Progress ----------------------------------------------------------

    async def fetch_progress(
        self, base_url: str, token: str,
    ) -> dict[str, dict]:
        """Return ``{external_id: progress_record}`` for every touched chapter.

        Server-side filtered so only chapters with ``lastPageRead > 0``
        OR ``isRead = true`` come back — way cheaper than the REST API's
        per-manga walk. Empty progress maps (user never read anything
        in their library) are valid.
        """
        url = base_url.rstrip("/")
        try:
            data = await self._gql(
                url, token, _PROGRESS_QUERY,
                variables={"first": _MAX_PROGRESS_CHAPTERS},
            )
        except Exception as exc:
            log.warning("suwayomi_progress_fetch_failed", error=str(exc))
            return {}

        chapter_nodes = (data.get("chapters") or {}).get("nodes") or []
        if not isinstance(chapter_nodes, list):
            return {}

        out: dict[str, dict] = {}
        for chapter in chapter_nodes:
            if not isinstance(chapter, dict):
                continue
            manga_id = chapter.get("mangaId")
            source_order = chapter.get("sourceOrder")
            chapter_id = chapter.get("id")
            if manga_id is None or source_order is None or chapter_id is None:
                continue
            last_page = float(chapter.get("lastPageRead") or 0)
            page_count = float(chapter.get("pageCount") or 0)
            is_read = bool(chapter.get("isRead") or False)
            external_id = f"{manga_id}.{source_order}.{chapter_id}"
            out[external_id] = {
                "current_time_s": last_page,
                "duration_s":     page_count,
                "progress":       (last_page / page_count) if page_count else 0.0,
                "is_finished":    is_read,
            }
        return out

    async def fetch_item_details(
        self, base_url: str, token: str, *, external_id: str,
    ) -> dict | None:
        """Return ``{manga, chapter}`` for a specific chapter.

        external_id is ``{manga_id}.{source_order}`` or
        ``{manga_id}.{source_order}.{chapter_id}`` — we only need the
        first two parts. If the chapter isn't in the returned manga's
        chapter list (deleted upstream?), returns None.
        """
        parts = external_id.split(".")
        if len(parts) < 2:
            return None
        try:
            manga_id = int(parts[0])
            source_order = int(parts[1])
        except ValueError:
            return None

        url = base_url.rstrip("/")
        try:
            data = await self._gql(
                url, token, _ITEM_DETAILS_QUERY,
                variables={"mangaId": manga_id},
            )
        except Exception as exc:
            log.debug(
                "suwayomi_item_details_failed",
                external_id=external_id, error=str(exc),
            )
            return None

        manga = data.get("manga")
        if not isinstance(manga, dict):
            return None

        chapter = None
        chapter_nodes = (manga.get("chapters") or {}).get("nodes") or []
        for candidate in chapter_nodes:
            if isinstance(candidate, dict) and candidate.get("sourceOrder") == source_order:
                chapter = candidate
                break
        if chapter is None:
            return None

        return {"manga": manga, "chapter": chapter}

    async def prepare_chapter(
        self, base_url: str, token: str, *, chapter_db_id: int,
    ) -> int:
        """Call ``fetchChapterPages`` so Suwayomi populates its local page
        cache for this chapter before we proxy image bytes.

        Suwayomi's REST page endpoints (``/api/v1/manga/X/chapter/Y/page/N``)
        404 on an uncached chapter because ``pageCount`` is 0 until the
        mutation runs and hits the source extension to enumerate pages.
        Calling this is idempotent — already-cached chapters return
        immediately. Returns the resulting pageCount, or 0 on failure.
        """
        try:
            data = await self._gql(
                base_url.rstrip("/"), token, _FETCH_CHAPTER_PAGES_MUTATION,
                variables={"input": {"chapterId": chapter_db_id}},
                timeout=_CATALOG_TIMEOUT_S,  # source fetches can be slow
            )
        except Exception as exc:
            log.warning(
                "suwayomi_prepare_chapter_failed",
                chapter_db_id=chapter_db_id, error=str(exc),
            )
            return 0
        chapter = (data.get("fetchChapterPages") or {}).get("chapter") or {}
        return int(chapter.get("pageCount") or 0)

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
        """Write chapter progress via ``updateChapter`` mutation.

        Requires the chapter's **DB id** (not manga_id + sourceOrder),
        which we encode as the third part of ``external_id``. Legacy
        2-part external_ids from the REST era can't be updated — we
        log-and-skip rather than attempting an extra GraphQL round-trip
        to resolve. A subsequent catalog sync will upgrade the row.
        """
        parts = external_id.split(".")
        if len(parts) < 3:
            log.debug(
                "suwayomi_progress_push_skipped_legacy_external_id",
                external_id=external_id,
            )
            return False
        try:
            chapter_id = int(parts[2])
        except ValueError:
            return False

        if is_finished:
            page_count = max(1, int(round(duration_s)))
            patch: dict[str, Any] = {
                "isRead":       True,
                "lastPageRead": page_count - 1,
            }
        else:
            # Always force isRead=False on the unfinished branch — without
            # this, an explicit "mark as unread" UI action couldn't undo a
            # prior isRead=True upstream (we'd be patching only lastPageRead
            # while the upstream isRead flag stayed sticky). With it,
            # toggling read state in the Augmentum UI round-trips correctly.
            last_page = max(0, int(round(current_time_s)))
            patch = {"lastPageRead": last_page, "isRead": False}

        variables = {"input": {"id": chapter_id, "patch": patch}}
        try:
            await self._gql(
                base_url.rstrip("/"), token, _UPDATE_CHAPTER_MUTATION,
                variables=variables,
            )
            return True
        except Exception as exc:
            log.debug(
                "suwayomi_progress_push_failed",
                external_id=external_id, error=str(exc),
            )
            return False


# --- Translation helpers ---------------------------------------------------
# Kept at module level so they're unit-testable independent of network IO.

def _build_manga_payload(manga: dict[str, Any]) -> dict[str, Any]:
    """Flatten a GraphQL MangaType into the dict we stash in ``CatalogItem.extra``.

    Kept here (not in the class) because this is a pure data translation —
    easier to test and easier to extend when new fields land upstream.
    """
    return {
        "suwayomi_manga_id":   manga.get("id"),
        "series_name":         manga.get("title") or "",
        "author":              manga.get("author") or "",
        "artist":              manga.get("artist") or "",
        "description":         manga.get("description") or "",
        "genres":              manga.get("genre") or [],
        "status":              (str(manga.get("status") or "")).lower() or None,
        "source_id":           manga.get("sourceId") or "",
        "thumbnail_url":       manga.get("thumbnailUrl") or "",
        "last_fetched_at":     manga.get("lastFetchedAt"),
        "in_library_at":       manga.get("inLibraryAt"),
        "real_url":            manga.get("realUrl") or "",
    }


def _chapter_to_catalog_item(
    chapter: dict[str, Any],
    manga_payload: dict[str, Any],
) -> CatalogItem | None:
    """Translate one GraphQL ChapterType node into a ``CatalogItem``.

    Returns ``None`` for chapters missing the fields we need to address
    them (sourceOrder, id, manga_id) — these can't be served, so no
    point indexing them.
    """
    if not isinstance(chapter, dict):
        return None
    source_order = chapter.get("sourceOrder")
    chapter_id = chapter.get("id")
    manga_id = manga_payload.get("suwayomi_manga_id")
    if source_order is None or manga_id is None or chapter_id is None:
        return None

    external_id = f"{manga_id}.{source_order}.{chapter_id}"
    page_count = int(chapter.get("pageCount") or 0)
    last_page_read = int(chapter.get("lastPageRead") or 0)
    is_read = bool(chapter.get("isRead") or False)
    progress_pct = (last_page_read / page_count) if page_count else 0.0
    chapter_number = chapter.get("chapterNumber")

    stream_path = f"/api/v1/manga/{manga_id}/chapter/{source_order}"

    name = (
        chapter.get("name")
        or f"{manga_payload.get('series_name') or 'Untitled'} Ch. {chapter_number or source_order}"
    )

    return CatalogItem(
        external_id=external_id,
        name=name,
        kind="comic",
        mime_type="application/vnd.comicbook+zip",
        size_bytes=0,
        duration_ms=0,
        progress_pct=progress_pct,
        cover_url=f"/api/v1/manga/{manga_id}/thumbnail",
        author=manga_payload.get("author") or "",
        narrator="",
        stream_path=stream_path,
        extra={
            **manga_payload,
            # Kept under two names during transition — ``chapter_index`` is
            # what existing callers (sync, per-page route) know, and it
            # equals ``sourceOrder`` in v2 semantics. ``chapter_source_order``
            # is the new-world name; ``chapter_db_id`` is the thing only
            # GraphQL exposes (needed for the update mutation).
            "chapter_index":        source_order,
            "chapter_source_order": source_order,
            "chapter_db_id":        chapter_id,
            "chapter_number":       chapter_number,
            "chapter_name":         chapter.get("name") or "",
            "page_count":           page_count,
            "current_page":         last_page_read,
            "is_finished":          is_read,
            "is_downloaded":        bool(chapter.get("isDownloaded") or False),
            "upload_date":          chapter.get("uploadDate"),
            "scanlator":            chapter.get("scanlator") or "",
            "real_url":             chapter.get("realUrl") or "",
            "suwayomi_chapter_id":  chapter_id,
        },
    )
