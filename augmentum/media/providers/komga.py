"""Komga HTTP client.

Docs: https://komga.org/docs/openapi/. Default port 25600. HTTP Basic auth —
we store ``base64(username:password)`` as the ``access_token`` in
``user_media_servers`` and set ``Authorization: Basic <token>`` on every
call. Komga also supports API keys on newer versions; Basic remains the
lowest-common-denominator so we wire that up first.

Architectural notes:

- **Hierarchy**: Komga organizes content as Library → Series → Book. One
  ``file_index`` row is emitted per *Book* (a single CBZ/CBR). Series
  metadata is stashed in ``extra`` and reconciled separately via
  ``ComicSeriesStore`` on sync (outside this module).

- **Stream path**: Komga exposes three shapes —
  1. ``/api/v1/books/{id}/file`` — full archive bytes (range-capable)
  2. ``/api/v1/books/{id}/pages/{n}`` — per-page image
  3. ``/api/v1/books/{id}/pages/{n}/thumbnail`` — per-page thumbnail
  We store ``stream_path = /api/v1/books/{id}/file`` for parity with the
  existing byte-range proxy. The comic-specific per-page delivery route
  (built in a later phase) resolves book_id → per-page URLs.

- **Progress**: Komga records ``{page, completed, readDate}`` per book.
  We map to the audio-shaped ``MediaProvider`` protocol by using
  ``current_time_s = current_page`` and ``duration_s = page_count``.
  Semantically odd but it keeps the existing progress plumbing unchanged.
  Consumers that know the item is a comic can reinterpret.

- **Auth header in URLs**: Unlike ABS, Komga does NOT support token-in-URL
  for streaming. Every request needs the Authorization header. The range
  proxy (:mod:`augmentum.proxy.media_routes.stream`) handles this by
  adding the header server-side; public URLs returned by
  ``build_stream_url`` are intended for authenticated proxying, not
  direct client use.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from augmentum.media.providers.base import CatalogItem, ProviderInfo
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)


_TIMEOUT_S = 10.0
_LOGIN_TIMEOUT_S = 15.0
_CATALOG_TIMEOUT_S = 60.0

# Every upstream call follows redirects — Komga behind Traefik/Caddy commonly
# lives at ``/komga/`` with a path rewrite, and proxies may 308 on trailing
# slashes. Matches the ABS provider's posture (scoped per-call, not client-level).
_REDIRECT_KW = {"follow_redirects": True}


_AUTHOR_ROLE_PRIORITY = (
    "writer",
    "story",
    "author",
    "creator",
)


def _encode_basic(username: str, password: str) -> str:
    """Return the ``base64(user:pass)`` value for an HTTP Basic header."""
    raw = f"{username}:{password}".encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _basic_header(token: str) -> dict[str, str]:
    """Auth header dict for a stored (base64-encoded) Komga credential."""
    return {"Authorization": f"Basic {token}"}


# Current-user endpoint for Basic-auth validation. Komga moved the users API
# from v1 → v2 (verified against the bundled OpenAPI 1.24.x in
# docs/integrations/media-servers/komga/openapi.json, which has NO
# /api/v1/users* paths). Try v2 first, fall back to v1 for older servers.
_ME_PATHS = ("/api/v2/users/me", "/api/v1/users/me")


def _search_is(value: Any) -> dict[str, Any]:
    """Komga search operator payload for an exact match."""
    return {"operator": "is", "value": value}


def _search_body(condition: dict[str, Any] | None = None) -> dict[str, Any]:
    """Request body for ``POST /api/v1/*/list`` endpoints."""
    return {"condition": condition} if condition else {}


def _alternate_titles(raw: Any) -> list[str]:
    """Flatten Komga's ``alternateTitles`` objects into plain strings."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(title)
    return out


def _english_alternate_title(raw: Any) -> str:
    """Return the first alternate title labelled as English, if any."""
    if not isinstance(raw, list):
        return ""
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        label = str(item.get("label") or "").strip().lower()
        if not title:
            continue
        if label in {"en", "eng", "english"} or "english" in label:
            return title
    return ""


def _author_names(authors: Any) -> str:
    """Prefer writer-like roles while keeping the original author order."""
    if not isinstance(authors, list):
        return ""
    ranked: list[tuple[int, int, str]] = []
    for idx, author in enumerate(authors):
        if not isinstance(author, dict):
            continue
        name = str(author.get("name") or "").strip()
        if not name:
            continue
        role = str(author.get("role") or "").strip().lower()
        priority = next(
            (
                i for i, token in enumerate(_AUTHOR_ROLE_PRIORITY)
                if token in role
            ),
            len(_AUTHOR_ROLE_PRIORITY),
        )
        ranked.append((priority, idx, name))
    ranked.sort()
    out: list[str] = []
    seen: set[str] = set()
    for _, _, name in ranked:
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return ", ".join(out)


class KomgaProvider:
    name = "komga"

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    # --- Detection + auth --------------------------------------------------

    async def ping(self, base_url: str) -> ProviderInfo | None:
        """``GET /api/v2/actuator/info`` — unauthenticated version stamp.

        Returns Spring Boot actuator info: ``{build: {version, name}, komga:
        {version}}``. We treat any 200 with a recognizable ``komga`` key as
        a valid fingerprint. Some deployments lock actuator behind auth —
        in that case we fall back to ``/api/v1/claim`` which is always
        unauthenticated and returns the "claimed" status.
        """
        url = base_url.rstrip("/")
        try:
            resp = await self._http.get(
                f"{url}/api/v2/actuator/info",
                timeout=_TIMEOUT_S, **_REDIRECT_KW,
            )
            if resp.status_code == 200:
                body = resp.json() or {}
                version = ""
                if isinstance(body, dict):
                    # Shape varies across Komga versions: sometimes
                    # {komga: {version}}, sometimes {build: {version}}.
                    version = (
                        ((body.get("komga") or {}).get("version"))
                        or ((body.get("build") or {}).get("version"))
                        or ""
                    )
                    if "komga" in body or version:
                        return ProviderInfo(
                            provider=self.name,
                            base_url=url,
                            version=str(version),
                        )

            # Fallback: /api/v1/claim is always unauthenticated and always
            # present on Komga. Returns {"isClaimed": true|false}.
            claim = await self._http.get(
                f"{url}/api/v1/claim",
                timeout=_TIMEOUT_S, **_REDIRECT_KW,
            )
            if claim.status_code == 200:
                body = claim.json() or {}
                if isinstance(body, dict) and "isClaimed" in body:
                    return ProviderInfo(
                        provider=self.name,
                        base_url=url,
                        is_initialized=bool(body.get("isClaimed", True)),
                    )
            return None
        except Exception as exc:
            log.debug("komga_ping_failed", base_url=url, error=str(exc))
            return None

    async def first_run_setup(
        self, base_url: str, username: str, password: str,
    ) -> None:
        """Idempotently claim a fresh Komga server (create the first admin).

        A new Komga has no users — Basic-auth login can't work until the
        server is "claimed". ``POST /api/v1/claim`` with the
        ``X-Komga-Email`` / ``X-Komga-Password`` headers creates the first
        admin (this is what the web UI's first-run screen does). No-op if
        already claimed: ``GET /api/v1/claim`` reports ``isClaimed`` and is
        always unauthenticated. ``username`` must be an email (Komga
        validates the header as one).
        """
        url = base_url.rstrip("/")
        try:
            resp = await self._http.get(
                f"{url}/api/v1/claim", timeout=_TIMEOUT_S, **_REDIRECT_KW,
            )
            if resp.status_code == 200 and (resp.json() or {}).get("isClaimed"):
                return
        except Exception:  # noqa: BLE001 — not ready yet; attempt claim
            pass
        await self._http.post(
            f"{url}/api/v1/claim",
            headers={"X-Komga-Email": username, "X-Komga-Password": password},
            timeout=_LOGIN_TIMEOUT_S, **_REDIRECT_KW,
        )

    async def login(self, base_url: str, username: str, password: str) -> str:
        """Validate credentials via the current-user endpoint with HTTP Basic.

        Returns ``base64(user:pass)`` on success — stored as the Komga
        server's ``access_token``. Raises ``ValueError`` on 401 (bad
        credentials) and ``RuntimeError`` on other transport failures so
        the UI can distinguish user-actionable errors from server faults.

        Tries ``/api/v2/users/me`` first (current Komga — verified against
        the bundled OpenAPI 1.24.x, which dropped all ``/api/v1/users*``)
        and falls back to ``/api/v1/users/me`` for older servers.
        """
        url = base_url.rstrip("/")
        token = _encode_basic(username, password)
        last_status: int | None = None
        for path in _ME_PATHS:
            resp = await self._http.get(
                f"{url}{path}",
                headers=_basic_header(token),
                timeout=_LOGIN_TIMEOUT_S, **_REDIRECT_KW,
            )
            if resp.status_code == 200:
                return token
            if resp.status_code in (401, 403):
                raise ValueError("Invalid username or password")
            last_status = resp.status_code  # 404 on this API version → try next
        raise RuntimeError(f"Login failed: HTTP {last_status}")

    async def change_password(
        self, base_url: str, username: str,
        current_password: str, new_password: str,
    ) -> str:
        """Change the managed account's password; return the new Basic token.

        Komga: ``PATCH /api/v{2,1}/users/me/password`` with ``{password}``
        authenticated by the CURRENT Basic credential. Because the stored
        token IS ``base64(user:pass)``, the old one is invalid the instant
        the password changes — so we recompute and return the new token.
        """
        url = base_url.rstrip("/")
        cur_token = _encode_basic(username, current_password)
        last_status: int | None = None
        for path in ("/api/v2/users/me/password", "/api/v1/users/me/password"):
            resp = await self._http.patch(
                f"{url}{path}",
                headers={**_basic_header(cur_token), "Content-Type": "application/json"},
                json={"password": new_password},
                timeout=_LOGIN_TIMEOUT_S, **_REDIRECT_KW,
            )
            if resp.status_code in (200, 204):
                return _encode_basic(username, new_password)
            if resp.status_code in (401, 403):
                raise ValueError("Current password rejected")
            last_status = resp.status_code  # 404 → try the next API version
        raise RuntimeError(f"change password failed: HTTP {last_status}")

    async def verify_token(self, base_url: str, token: str) -> bool:
        url = base_url.rstrip("/")
        for path in _ME_PATHS:
            try:
                resp = await self._http.get(
                    f"{url}{path}",
                    headers=_basic_header(token),
                    timeout=_TIMEOUT_S, **_REDIRECT_KW,
                )
            except Exception:
                return False
            if resp.status_code == 200:
                return True
            if resp.status_code in (401, 403):
                return False
            # 404 → endpoint not on this version; try the next candidate.
        return False

    # --- Catalog -----------------------------------------------------------

    async def fetch_catalog(self, base_url: str, token: str) -> list[CatalogItem]:
        """Paginated traversal: series → books per series.

        One ``CatalogItem`` per Book. Series-level metadata (title,
        publisher, language, author) lives in ``extra`` so the sync layer
        can resolve it against :mod:`comic_series_store` without a second
        fetch. Pagination size 500 keeps per-request payloads reasonable
        on a 50k-book library (~100 series-page requests + ~100 book-page
        requests in the worst case).
        """
        url = base_url.rstrip("/")
        headers = _basic_header(token)
        items: list[CatalogItem] = []
        series_seen = 0

        series_page = 0
        while True:
            resp = await self._http.post(
                f"{url}/api/v1/series/list",
                params={"page": series_page, "size": 500},
                json=_search_body(),
                headers=headers,
                timeout=_CATALOG_TIMEOUT_S, **_REDIRECT_KW,
            )
            if resp.status_code != 200:
                log.warning(
                    "komga_series_page_failed",
                    page=series_page, status=resp.status_code,
                )
                break
            body = resp.json() or {}
            series_list = body.get("content") or []
            if not series_list:
                break
            for series in series_list:
                series_seen += 1
                items.extend(
                    await self._fetch_series_books(url, headers, series)
                )
            if body.get("last") is True:
                break
            series_page += 1

        log.info(
            "komga_catalog_fetched",
            items=len(items), series=series_seen,
        )
        return items

    async def _fetch_series_books(
        self,
        url: str,
        headers: dict[str, str],
        series: dict[str, Any],
    ) -> list[CatalogItem]:
        """Fetch every book in one series, translate to ``CatalogItem``s."""
        series_id = series.get("id", "")
        if not series_id:
            return []
        series_meta = series.get("metadata") or {}
        series_booksMeta = series.get("booksMetadata") or {}
        alt_titles = _alternate_titles(series_meta.get("alternateTitles"))
        series_payload = {
            "komga_series_id":  series_id,
            "series_name":      series_meta.get("title") or series.get("name") or "",
            "publisher":        series_meta.get("publisher") or "",
            "language":         series_meta.get("language") or "",
            "status":           (series_meta.get("status") or "").lower() or None,
            "genres":           series_meta.get("genres") or [],
            "tags":             series_meta.get("tags") or [],
            "age_rating":       series_meta.get("ageRating") or "",
            "description":      series_meta.get("summary") or "",
            "alternate_titles": alt_titles,
            "english_title":    _english_alternate_title(series_meta.get("alternateTitles")),
            "title_sort":       series_meta.get("titleSort") or "",
            "reading_direction": (
                str(series_meta.get("readingDirection") or "").lower() or ""
            ),
            "series_cover_url": f"/api/v1/series/{series_id}/thumbnail",
            "total_book_count": series.get("booksCount") or 0,
            "authors":          series_booksMeta.get("authors") or [],
        }

        out: list[CatalogItem] = []
        page = 0
        while True:
            resp = await self._http.post(
                f"{url}/api/v1/books/list",
                params={"page": page, "size": 500},
                json=_search_body({
                    "seriesId": _search_is(series_id),
                }),
                headers=headers,
                timeout=_CATALOG_TIMEOUT_S, **_REDIRECT_KW,
            )
            if resp.status_code != 200:
                log.warning(
                    "komga_books_page_failed",
                    series_id=series_id, page=page, status=resp.status_code,
                )
                break
            body = resp.json() or {}
            books = body.get("content") or []
            if not books:
                break
            for book in books:
                item = _book_to_catalog_item(book, series_payload)
                if item is not None:
                    out.append(item)
            if body.get("last") is True:
                break
            page += 1
        return out

    # --- Streaming ---------------------------------------------------------

    def build_stream_url(self, base_url: str, stream_path: str, token: str) -> str:
        """Return the authenticated archive URL for byte-range proxying.

        Komga requires the Authorization header on every request; the
        caller (range proxy) must add the header server-side. The URL
        returned here is intended for server-to-server fetching, not
        direct client consumption.
        """
        path = stream_path if stream_path.startswith("/") else f"/{stream_path}"
        return f"{base_url.rstrip('/')}{path}"

    def build_cover_url(self, base_url: str, external_id: str, token: str) -> str:
        """Book thumbnail endpoint — 300x450 JPEG by default.

        Same auth constraint as ``build_stream_url``: the cover route
        must fetch with the Authorization header attached server-side.
        """
        return f"{base_url.rstrip('/')}/api/v1/books/{external_id}/thumbnail"

    # --- Progress ----------------------------------------------------------

    async def fetch_progress(self, base_url: str, token: str) -> dict[str, dict]:
        """Return ``{book_id: progress_record}`` for every book with progress.

        Komga returns read-progress embedded on each Book via
        ``/api/v1/books/{id}`` but there's no bulk endpoint. Cheapest path
        is ``/api/v1/books?read_status=IN_PROGRESS,READ`` which paginates
        every book the user has interacted with — typically a fraction of
        the full library.

        Returns the audio-shaped record even though the domain is pages:
        ``current_time_s`` maps to ``current_page`` and ``duration_s`` to
        ``page_count``.
        """
        url = base_url.rstrip("/")
        headers = _basic_header(token)
        out: dict[str, dict] = {}
        page = 0
        while True:
            resp = await self._http.post(
                f"{url}/api/v1/books/list",
                params={
                    "page": page,
                    "size": 500,
                },
                json=_search_body({
                    "anyOf": [
                        {"readStatus": _search_is("IN_PROGRESS")},
                        {"readStatus": _search_is("READ")},
                    ],
                }),
                headers=headers,
                timeout=_TIMEOUT_S, **_REDIRECT_KW,
            )
            if resp.status_code != 200:
                break
            body = resp.json() or {}
            books = body.get("content") or []
            if not books:
                break
            for book in books:
                book_id = book.get("id") or ""
                read_progress = book.get("readProgress") or {}
                if not book_id or not read_progress:
                    continue
                current_page = float(read_progress.get("page") or 0)
                page_count = float(
                    (book.get("media") or {}).get("pagesCount") or 0
                )
                is_finished = bool(read_progress.get("completed") or False)
                out[book_id] = {
                    "current_time_s": current_page,
                    "duration_s":     page_count,
                    "progress":       (current_page / page_count) if page_count else 0.0,
                    "is_finished":    is_finished,
                }
            if body.get("last") is True:
                break
            page += 1
        return out

    async def fetch_item_details(
        self, base_url: str, token: str, *, external_id: str,
    ) -> dict | None:
        """``GET /api/v1/books/{id}`` with full metadata.

        Komga's listing and detail endpoints return the same shape, but
        the detail endpoint can include ``/pages`` sub-resources for
        callers that want page dimensions. We keep the call simple —
        per-page info is fetched lazily by the reader as needed.
        """
        url = base_url.rstrip("/")
        try:
            resp = await self._http.get(
                f"{url}/api/v1/books/{external_id}",
                headers=_basic_header(token),
                timeout=_TIMEOUT_S, **_REDIRECT_KW,
            )
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception as exc:
            log.debug(
                "komga_item_details_failed",
                external_id=external_id, error=str(exc),
            )
            return None

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
        """Write page-level progress via ``PATCH /api/v1/books/{id}/read-progress``.

        Protocol translation: the audio-shaped ``current_time_s`` is
        treated as the current page, ``duration_s`` as the page count.
        If ``is_finished`` is set we emit ``completed: true`` and let
        Komga infer the page. Otherwise we send ``{page: int(current_page)}``.
        """
        url = base_url.rstrip("/")
        if is_finished:
            payload: dict[str, Any] = {"completed": True}
        else:
            page_int = max(1, int(round(current_time_s)))
            payload = {"page": page_int}
        try:
            resp = await self._http.patch(
                f"{url}/api/v1/books/{external_id}/read-progress",
                json=payload,
                headers=_basic_header(token),
                timeout=_TIMEOUT_S, **_REDIRECT_KW,
            )
            return 200 <= resp.status_code < 300
        except Exception as exc:
            log.debug(
                "komga_progress_push_failed",
                external_id=external_id, error=str(exc),
            )
            return False


def _book_to_catalog_item(
    book: dict[str, Any],
    series_payload: dict[str, Any],
) -> CatalogItem | None:
    """Translate one Komga Book into a ``CatalogItem``.

    Returns ``None`` for items we can't stream (no id, no media).
    ``series_payload`` carries the owning series' metadata so the sync
    layer can resolve it against :class:`ComicSeriesStore` without a
    second fetch.
    """
    book_id = book.get("id", "")
    if not book_id:
        return None

    media = book.get("media") or {}
    metadata = book.get("metadata") or {}
    # Page-level progress embedded on the listing response.
    read_progress = book.get("readProgress") or {}
    current_page = int(read_progress.get("page") or 0)
    page_count = int(media.get("pagesCount") or 0)
    progress_pct = (current_page / page_count) if page_count else 0.0
    is_finished = bool(read_progress.get("completed") or False)

    # Stream path: full-archive download. The comic-specific page route
    # (built later) resolves book_id → per-page image URLs as needed.
    stream_path = f"/api/v1/books/{book_id}/file"

    # Mime: Komga reports ``application/vnd.comicbook+zip`` /
    # ``application/vnd.comicbook-rar`` / ``application/pdf`` etc via
    # ``media.mediaType``. Fall back to CBZ since that's the common case.
    mime_type = (
        media.get("mediaType")
        or "application/vnd.comicbook+zip"
    )

    # Volume/number: Komga exposes both ``number`` (display) and
    # ``numberSort`` (float, handles 12.5 decimals). Prefer the sort
    # variant for ordering, fall back to display.
    number = metadata.get("number") or ""
    number_sort = metadata.get("numberSort")

    name = (
        metadata.get("title")
        or series_payload.get("series_name")
        or book.get("name")
        or "Untitled"
    )
    authors = metadata.get("authors") or series_payload.get("authors") or []
    author_names = _author_names(authors)
    cover_url = (
        series_payload.get("series_cover_url")
        or f"/api/v1/books/{book_id}/thumbnail"
    )

    return CatalogItem(
        external_id=book_id,
        name=name,
        kind="comic",
        mime_type=mime_type,
        size_bytes=int(book.get("sizeBytes") or 0),
        duration_ms=0,
        progress_pct=progress_pct,
        # Series posters feel better in the series-first UI than an
        # arbitrary volume cover. Keep the book poster path in ``extra``
        # for any future item-level views.
        cover_url=cover_url,
        author=author_names,
        narrator="",
        stream_path=stream_path,
        extra={
            **series_payload,
            "volume":           number,
            "volume_sort":      float(number_sort) if number_sort is not None else None,
            "release_date":     metadata.get("releaseDate") or "",
            "summary":          metadata.get("summary") or "",
            "page_count":       page_count,
            "current_page":     current_page,
            "is_finished":      is_finished,
            "created_at":       book.get("created") or "",
            "last_modified":    book.get("lastModified") or "",
            "komga_book_id":    book_id,
            "book_cover_url":   f"/api/v1/books/{book_id}/thumbnail",
            "authors":          authors,
        },
    )
