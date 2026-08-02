"""Audiobookshelf HTTP client.

Docs: api.audiobookshelf.org. Default port 13378. Bearer-token auth with
an optional `POST /login` exchange to turn user/pass into a token. The
`/ping` endpoint is unauthenticated and returns `{"success": true}` —
used by the detector as a fingerprint.

One architectural note: ABS serves audiobooks as a single logical "item"
with multiple audio tracks (chapters). We surface one file_index row
*per book*, not per chapter, and let the stream proxy request the
book-level playback URL (ABS handles chapter seeking internally).
Chapter metadata is stashed in ``extra['chapters']`` so the detail
panel can render a chapter list without another round trip.
"""

from __future__ import annotations

import asyncio
import re

from typing import TYPE_CHECKING

from augmentum.media.normalize import author_for_match
from augmentum.media.providers.base import CatalogItem, ProviderInfo
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)


_TIMEOUT_S = 10.0
_LOGIN_TIMEOUT_S = 15.0
_CATALOG_TIMEOUT_S = 60.0
_DETAIL_FETCH_CONCURRENCY = 8
_LIBRARY_PAGE_SIZE = 1000

# Every upstream call follows redirects. Reverse proxies (Caddy, Traefik,
# Nginx-PM) in front of ABS routinely 308 http→https or slash-normalize,
# and the shared httpx client doesn't follow redirects by default.
# Without this, a correctly-configured ABS behind a proxy silently rejects
# every login with "HTTP 308" — confusing, user-unactionable. Scoped per
# call (not client-level) so the rest of the app's SSRF posture stays put.
_REDIRECT_KW = {"follow_redirects": True}


class AudiobookshelfProvider:
    name = "audiobookshelf"

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    # --- Detection + auth --------------------------------------------------

    async def ping(self, base_url: str) -> ProviderInfo | None:
        """GET /ping — returns `{success: true}` on a real ABS server.

        We also hit /status (unauthenticated) to capture the `isInit`
        flag, which tells the UI whether this is a freshly-installed
        server that still needs its first user. `ping` alone isn't
        enough to fingerprint — any trivial 200-returning endpoint
        would match — so we require BOTH paths to resolve correctly.
        """
        url = base_url.rstrip("/")
        try:
            ping_resp = await self._http.get(
                f"{url}/ping", timeout=_TIMEOUT_S, **_REDIRECT_KW,
            )
            if ping_resp.status_code != 200:
                return None
            ping_body = ping_resp.json()
            if not isinstance(ping_body, dict) or ping_body.get("success") is not True:
                return None

            status_resp = await self._http.get(
                f"{url}/status", timeout=_TIMEOUT_S, **_REDIRECT_KW,
            )
            if status_resp.status_code != 200:
                return None
            status_body = status_resp.json()
            if not isinstance(status_body, dict) or "isInit" not in status_body:
                return None

            return ProviderInfo(
                provider=self.name,
                base_url=url,
                is_initialized=bool(status_body.get("isInit", True)),
            )
        except Exception as exc:
            log.debug("audiobookshelf_ping_failed", base_url=url, error=str(exc))
            return None

    async def first_run_setup(
        self, base_url: str, username: str, password: str,
    ) -> None:
        """Idempotently create the initial root user on a fresh server.

        A freshly provisioned Audiobookshelf reports ``isInit: false`` and
        has no users, so ``/login`` can't work yet. ``POST /init`` with
        ``{"newRoot": {username, password}}`` creates the all-powerful root
        account (ABS hashes the password server-side). No-op if the server
        is already initialized (re-install / restart): ``GET /status``
        reports ``isInit`` without auth.
        """
        url = base_url.rstrip("/")
        try:
            resp = await self._http.get(
                f"{url}/status", timeout=_TIMEOUT_S, **_REDIRECT_KW,
            )
            if resp.status_code == 200 and (resp.json() or {}).get("isInit"):
                return
        except Exception:  # noqa: BLE001 — not ready yet; attempt init
            pass
        await self._http.post(
            f"{url}/init",
            json={"newRoot": {"username": username, "password": password}},
            timeout=_LOGIN_TIMEOUT_S,
            **_REDIRECT_KW,
        )

    async def login(self, base_url: str, username: str, password: str) -> str:
        """POST /login — returns user.token on success.

        Raises ValueError on auth failure (surface-able to UI) or
        RuntimeError on transport/shape issues.
        """
        url = base_url.rstrip("/")
        resp = await self._http.post(
            f"{url}/login",
            json={"username": username, "password": password},
            timeout=_LOGIN_TIMEOUT_S,
            **_REDIRECT_KW,
        )
        if resp.status_code in (401, 403):
            raise ValueError("Invalid username or password")
        if resp.status_code != 200:
            raise RuntimeError(f"Login failed: HTTP {resp.status_code}")
        body = resp.json()
        token = ((body or {}).get("user") or {}).get("token", "")
        if not token:
            raise RuntimeError("Login response missing token")
        return token

    async def change_password(
        self, base_url: str, username: str,
        current_password: str, new_password: str,
    ) -> str:
        """Change the managed account's password; return a fresh token.

        ABS: ``PATCH /api/me/password`` with ``{password, newPassword}``
        and the current Bearer token. The JWT survives, but we re-login to
        mint/store a clean token matching the persisted credential.
        """
        url = base_url.rstrip("/")
        token = await self.login(base_url, username, current_password)
        resp = await self._http.patch(
            f"{url}/api/me/password",
            headers={"Authorization": f"Bearer {token}"},
            json={"password": current_password, "newPassword": new_password},
            timeout=_LOGIN_TIMEOUT_S,
            **_REDIRECT_KW,
        )
        if resp.status_code not in (200, 204):
            raise RuntimeError(f"change password failed: HTTP {resp.status_code}")
        return await self.login(base_url, username, new_password)

    async def verify_token(self, base_url: str, token: str) -> bool:
        url = base_url.rstrip("/")
        try:
            # /api/me is the minimal authenticated probe; returns the user row.
            resp = await self._http.get(
                f"{url}/api/me",
                headers={"Authorization": f"Bearer {token}"},
                timeout=_TIMEOUT_S,
                **_REDIRECT_KW,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # --- Catalog -----------------------------------------------------------

    async def fetch_catalog(self, base_url: str, token: str) -> list[CatalogItem]:
        """Pull every book across every library.

        Two-stage: list libraries → list items per library. We do NOT
        pass ``minified=1`` — ABS's minified form strips ``audioFiles``
        from library-item payloads, which is exactly the data we need
        to build the stream path. Without it, every item fails the
        sync loop's `stream_path` guard and silently disappears.
        """
        url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"}

        lib_resp = await self._http.get(
            f"{url}/api/libraries", headers=headers, timeout=_TIMEOUT_S,
            **_REDIRECT_KW,
        )
        if lib_resp.status_code != 200:
            raise RuntimeError(f"List libraries failed: HTTP {lib_resp.status_code}")
        libraries = (lib_resp.json() or {}).get("libraries", [])

        # Return every parseable item — even ones with empty stream_path.
        # sync.py partitions into indexable vs skipped and surfaces the
        # skip count + first-N titles through the store so the UI can
        # show "312 items · 141 skipped" without docker logs.
        items: list[CatalogItem] = []
        first_skipped_shape: dict | None = None
        detail_recoveries = 0
        for lib in libraries:
            lib_id = lib.get("id", "")
            lib_kind = (lib.get("mediaType") or "book").lower()  # 'book' | 'podcast'
            if not lib_id:
                continue
            page = 0
            while True:
                page_resp = await self._http.get(
                    f"{url}/api/libraries/{lib_id}/items",
                    params={"limit": _LIBRARY_PAGE_SIZE, "page": page},
                    headers=headers, timeout=_CATALOG_TIMEOUT_S,
                    **_REDIRECT_KW,
                )
                if page_resp.status_code != 200:
                    log.warning(
                        "audiobookshelf_library_failed",
                        library_id=lib_id, page=page, status=page_resp.status_code,
                    )
                    break
                body = page_resp.json() or {}
                results = body.get("results") or []
                if not results:
                    break

                sem = asyncio.Semaphore(_DETAIL_FETCH_CONCURRENCY)

                async def _bound_resolve(raw: dict):
                    async with sem:
                        return await self._resolve_catalog_item(
                            base_url=url, token=token, raw=raw, lib_kind=lib_kind,
                        )

                resolved = await asyncio.gather(*[
                    _bound_resolve(raw) for raw in results
                ])
                for raw, (item, recovered) in zip(results, resolved):
                    if item is None:
                        continue
                    if recovered:
                        detail_recoveries += 1
                    # Capture the shape of the first unplayable item so if the
                    # wider world comes up with a brand-new ABS shape we get
                    # one diagnostic line (not 453). Log-level only — counts
                    # and titles land in the store via sync.py for UI display.
                    if (
                        not item.stream_path
                        and not item.extra.get("index_without_stream")
                        and first_skipped_shape is None
                    ):
                        first_skipped_shape = _shape_summary(raw)
                    items.append(item)

                total = body.get("total")
                try:
                    total_i = int(total) if total is not None else 0
                except (TypeError, ValueError):
                    total_i = 0
                if len(results) < _LIBRARY_PAGE_SIZE:
                    break
                if total_i and ((page + 1) * _LIBRARY_PAGE_SIZE) >= total_i:
                    break
                page += 1

        if first_skipped_shape:
            log.warning(
                "audiobookshelf_items_skipped_no_stream",
                sample=first_skipped_shape,
            )

        log.info(
            "audiobookshelf_catalog_fetched",
            items=len(items), libraries=len(libraries),
            detail_recoveries=detail_recoveries,
        )
        return items

    async def _resolve_catalog_item(
        self,
        *,
        base_url: str,
        token: str,
        raw: dict,
        lib_kind: str,
    ) -> tuple[CatalogItem | None, bool]:
        """Map one listing row, hydrating folder books when the listing is thin.

        Some Audiobookshelf builds return folder-based books in library
        listings with only ``numAudioFiles`` and no per-file ``ino``. Those
        rows are not playable as-is, but ``GET /api/items/{id}?expanded=1``
        does include the full audio file records. We reuse the same mapper on
        that expanded payload so this remains one parsing code path.
        """
        item = _item_from_abs(raw, lib_kind=lib_kind)
        if item is None or item.stream_path:
            return item, False
        if item.extra.get("skip_reason") != "folder_needs_detail_fetch":
            return item, False

        detailed = await self.fetch_item_details(
            base_url, token, external_id=item.external_id,
        )
        if not isinstance(detailed, dict):
            return item, False

        hydrated = _item_from_abs(detailed, lib_kind=lib_kind)
        if hydrated is None:
            return item, False

        # Preserve non-stream fields from the listing when the expanded row
        # omits them, while still trusting the expanded row for audio files.
        hydrated.name = hydrated.name or item.name
        hydrated.author = hydrated.author or item.author
        hydrated.narrator = hydrated.narrator or item.narrator
        hydrated.cover_url = hydrated.cover_url or item.cover_url
        if not hydrated.extra.get("chapters"):
            hydrated.extra["chapters"] = item.extra.get("chapters") or []
        if hydrated.stream_path:
            hydrated.extra["recovered_via_detail"] = True
        return hydrated, bool(hydrated.stream_path)

    # --- Streaming ---------------------------------------------------------

    def build_stream_url(self, base_url: str, stream_path: str, token: str) -> str:
        """Resolve an item-level stream URL.

        ABS exposes `/api/items/{id}/play` for session-based playback, but
        for a simple HTML5 <audio> the direct file stream at
        `/api/items/{id}/file/{fileId}` is what we want. The caller
        (sync.py) stores that path in ``stream_path`` so the provider
        just appends the base URL here.
        """
        path = stream_path if stream_path.startswith("/") else f"/{stream_path}"
        sep = "&" if "?" in path else "?"
        return f"{base_url.rstrip('/')}{path}{sep}token={token}"

    def build_cover_url(self, base_url: str, external_id: str, token: str) -> str:
        """`/api/items/{id}/cover` is the canonical art endpoint.

        Returns a raw JPEG/PNG; the token is required because ABS treats
        covers as authenticated content (a deliberate choice so scraped
        public URLs can't fingerprint your library).
        """
        return (
            f"{base_url.rstrip('/')}/api/items/{external_id}/cover"
            f"?token={token}"
        )

    # --- Progress ----------------------------------------------------------

    async def fetch_progress(self, base_url: str, token: str) -> dict[str, dict]:
        """Return a map of external_id -> progress record.

        `/api/me` returns the whole user row including `mediaProgress`, so
        we get every item's state in one call instead of N+1. Progress
        records carry enough to fully resume playback: currentTime,
        duration, progress (0-1), and isFinished.
        """
        resp = await self._http.get(
            f"{base_url.rstrip('/')}/api/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=_TIMEOUT_S,
            **_REDIRECT_KW,
        )
        if resp.status_code != 200:
            return {}
        body = resp.json() or {}
        progress_list = body.get("mediaProgress") or []
        out: dict[str, dict] = {}
        for rec in progress_list:
            library_item_id = rec.get("libraryItemId") or rec.get("id")
            episode_id = rec.get("episodeId") or ""
            item_id = (
                f"{library_item_id}:{episode_id}"
                if library_item_id and episode_id else
                library_item_id
            )
            if not item_id:
                continue
            out[str(item_id)] = {
                "current_time_s": float(rec.get("currentTime") or 0),
                "duration_s":     float(rec.get("duration") or 0),
                "progress":       float(rec.get("progress") or 0),
                "is_finished":    bool(rec.get("isFinished") or False),
            }
        return out

    async def fetch_item_details(
        self,
        base_url: str,
        token: str,
        *,
        external_id: str,
        episode_id: str = "",
    ) -> dict | None:
        """Pull the full item record — listing omits chapters/audioFiles detail.

        ABS exposes `/api/items/{id}?expanded=1&include=progress` which
        returns the LibraryItem with rich metadata (full authors array,
        narrators, series, chapters, per-file audio records) plus the
        current user's progress on this item. One fetch powers the whole
        detail panel; the result is small (< 100 KB typical).
        """
        url = base_url.rstrip("/")
        try:
            resp = await self._http.get(
                f"{url}/api/items/{external_id}",
                params={
                    "expanded": 1,
                    "include": "progress",
                    **({"episode": episode_id} if episode_id else {}),
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=_TIMEOUT_S,
                **_REDIRECT_KW,
            )
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception as exc:
            log.debug("audiobookshelf_item_details_failed",
                      external_id=external_id, error=str(exc))
            return None

    async def push_progress(
        self,
        base_url: str,
        token: str,
        *,
        external_id: str,
        episode_id: str = "",
        current_time_s: float,
        duration_s: float,
        is_finished: bool = False,
    ) -> bool:
        """Write user progress back to the server.

        ABS exposes `PATCH /api/me/progress/{libraryItemId}`. Returns True
        on 2xx. Failures are swallowed by the caller — a dropped progress
        update isn't worth disrupting the user's playback.
        """
        progress = 0.0 if duration_s <= 0 else min(1.0, max(0.0, current_time_s / duration_s))
        payload = {
            "currentTime": current_time_s,
            "duration":    duration_s,
            "progress":    progress,
            "isFinished":  is_finished,
        }
        path = (
            f"/api/me/progress/{external_id}/{episode_id}"
            if episode_id else
            f"/api/me/progress/{external_id}"
        )
        try:
            resp = await self._http.patch(
                f"{base_url.rstrip('/')}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=_TIMEOUT_S,
                **_REDIRECT_KW,
            )
            return 200 <= resp.status_code < 300
        except Exception as exc:
            log.debug("audiobookshelf_progress_push_failed",
                      external_id=external_id, error=str(exc))
            return False

    def episode_stream_path(self, raw: dict, *, episode_id: str) -> str:
        """Resolve the audio content path for a podcast episode.

        ABS exposes podcast episode audio through ``episode.audioTrack.contentUrl``
        on the expanded library-item payload. The value is already a server-
        relative path, so the shared ``build_stream_url`` helper can finish it.
        """
        media = raw.get("media") or {}
        episodes = media.get("episodes") or []
        if not isinstance(episodes, list):
            return ""
        target_id = str(episode_id or "").strip()
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            if str(episode.get("id") or "").strip() != target_id:
                continue
            audio_track = episode.get("audioTrack") or {}
            content_url = str(audio_track.get("contentUrl") or "").strip()
            if content_url:
                return content_url
        return ""


def _shape_summary(raw: dict) -> dict:
    """Compact snapshot of a library-item payload for diagnostic logging.

    Keeps the log line short — sample sizes, keys, and a few candidate
    id/path fields — so we can see what shape ABS actually returned
    without dumping 200+ keys of each book.
    """
    media = raw.get("media") or {}
    first_af = (media.get("audioFiles") or [{}])[0] if media.get("audioFiles") else {}
    first_lf = (raw.get("libraryFiles") or [{}])[0] if raw.get("libraryFiles") else {}
    return {
        "top_keys":          list(raw.keys())[:15],
        "media_keys":        list(media.keys())[:15],
        "audio_file_count":  len(media.get("audioFiles") or []),
        "library_file_count": len(raw.get("libraryFiles") or []),
        "first_audio_keys":  list(first_af.keys())[:15] if first_af else [],
        "first_library_keys": list(first_lf.keys())[:15] if first_lf else [],
        "first_library_type": (first_lf.get("fileType") if first_lf else ""),
        "item_id_present":   bool(raw.get("id")),
        "is_file":           bool(raw.get("isFile")),
        "media_type":        raw.get("mediaType") or media.get("type") or "",
    }


def _abs_series(metadata: dict) -> tuple[str, str]:
    """Extract (series_name, sequence) from ABS metadata, both shapes.

    The detailed item endpoint returns ``series`` as an array of
    ``{"name", "sequence"}``; library listings flatten it to a
    ``seriesName`` string that often carries the volume as ``"Name #10"``.
    Returns ("", "") when the book isn't part of a series. ABS feeds the
    series machine that already exists in sync.py / source_metadata — it
    just was never populated for audiobooks.
    """
    raw = metadata.get("series")
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict):
            return (
                str(first.get("name") or "").strip(),
                str(first.get("sequence") or "").strip(),
            )
        if isinstance(first, str) and first.strip():
            return (first.strip(), "")
    name = str(metadata.get("seriesName") or "").strip()
    seq = ""
    if name:
        m = re.search(r"#\s*([0-9]+(?:\.[0-9]+)?)\s*$", name)
        if m:
            seq = m.group(1)
            name = name[: m.start()].strip()
    return (name, seq)


def _join_people(raw_people: object) -> str:
    """Flatten ABS person arrays into a display string.

    Library listings often carry ``authors`` / ``narrators`` as plain strings,
    while expanded item payloads can return objects like ``{"id": ..., "name":
    ...}``. Accept both shapes so the same mapper works for list rows and
    detail-hydrated rows.
    """
    if not isinstance(raw_people, list):
        return ""
    names: list[str] = []
    for person in raw_people:
        if isinstance(person, str):
            text = person.strip()
        elif isinstance(person, dict):
            text = str(person.get("name") or "").strip()
        else:
            text = ""
        if text:
            names.append(text)
    return ", ".join(names)


def _item_from_abs(raw: dict, *, lib_kind: str) -> CatalogItem | None:
    """Translate one ABS library item into a CatalogItem.

    Returns None for items we can't stream (no media files), so the
    sync path can silently skip them. We deliberately favor graceful
    degradation here — a malformed upstream row shouldn't abort the
    whole sync.
    """
    item_id = raw.get("id", "")
    if not item_id:
        return None
    media = raw.get("media") or {}
    metadata = media.get("metadata") or {}
    # Books have `audioFiles` with an `ino` (inode) or file path. Podcasts
    # expose a list of episodes — we index one container row per podcast
    # and let the detail panel / player pick a concrete episode at play time.
    # libraryFiles is ABS's newer per-file list that always carries ino
    # even when audioFiles is absent (e.g. item not yet scanned for audio).
    audio_files = media.get("audioFiles") or []
    library_files = raw.get("libraryFiles") or []
    is_file = bool(raw.get("isFile"))
    num_audio_files = int((media.get("numAudioFiles") or 0))

    # --- Stream path derivation ------------------------------------------------
    # Three real shapes observed in the wild, tried in order:
    #
    #   1. Modern ABS library listings include media.audioFiles[].ino
    #      (or audioFiles[].metadata.ino on some older scans).
    #   2. Newer "hydrated" listings expose libraryFiles[] where each file
    #      has fileType ("audio"/"image"/"metadata") + its own ino.
    #   3. Current ABS listings (seen on real deployments) don't ship either
    #      of the above — just `numAudioFiles` as a counter plus the
    #      LibraryItem's top-level `ino`. When isFile=true, the library
    #      item IS a single file whose inode is exactly what the stream
    #      endpoint wants. When isFile=false (folder-based), we genuinely
    #      can't derive the stream path from the listing and would need a
    #      per-item fetch — left as a follow-up, logged for visibility.
    def _first_audio_ino() -> str:
        for af in audio_files:
            ino = af.get("ino") or (af.get("metadata") or {}).get("ino") or ""
            if ino:
                return str(ino)
        for lf in library_files:
            ftype = (lf.get("fileType") or "").lower()
            if ftype == "audio":
                ino = lf.get("ino") or (lf.get("metadata") or {}).get("ino") or ""
                if ino:
                    return str(ino)
        # Single-file library item: its own ino is the file's ino.
        if is_file and num_audio_files > 0:
            top_ino = raw.get("ino") or ""
            if top_ino:
                return str(top_ino)
        return ""

    ino = _first_audio_ino()
    first_audio = audio_files[0] if audio_files else {}
    stream_path = f"/api/items/{item_id}/file/{ino}" if ino else ""

    # Classify *why* we couldn't derive a stream path so the UI can
    # group skipped titles by reason. The categories match the real
    # failure modes seen in ABS shapes; unknown ones get a generic
    # bucket that still tells the user it's a shape we haven't seen.
    skip_reason = ""
    if not stream_path and lib_kind != "podcast":
        if num_audio_files == 0 and not audio_files and not library_files:
            skip_reason = "no_audio_files"         # metadata-only entry
        elif not is_file and num_audio_files > 0:
            skip_reason = "folder_needs_detail_fetch"  # multi-file book; listing lacks files
        else:
            skip_reason = "unknown_shape"

    # Fall back to media.size (aggregate) when per-file sizes aren't
    # present in the listing — matches the same version that omits
    # audioFiles above.
    size_bytes = (
        sum(int((f.get("metadata") or {}).get("size") or 0) for f in audio_files)
        or int(media.get("size") or 0)
    )
    duration_ms = int(float(media.get("duration") or 0) * 1000)
    cover_rel = media.get("coverPath") or ""
    progress_pct = 0.0

    # Diagnostic: when an item has a real id but no way to play, log the
    # shape so the next investigator can see what ABS actually returned.
    # Debug-level — catalog sync emits a count summary at info for the UI.
    if not stream_path:
        log.debug(
            "audiobookshelf_item_no_stream",
            item_id=item_id,
            has_media=bool(media),
            audio_file_count=len(audio_files),
            library_file_count=len(library_files),
            media_keys=list(media.keys())[:10],
            top_keys=list(raw.keys())[:15],
        )

    mime_type = "audio/mpeg"
    if first_audio:
        fmt = (first_audio.get("metadata", {}).get("format") or "").lower()
        if "mp4" in fmt or "m4a" in fmt or "m4b" in fmt:
            mime_type = "audio/mp4"
        elif "ogg" in fmt or "opus" in fmt:
            mime_type = "audio/ogg"
        elif "flac" in fmt:
            mime_type = "audio/flac"

    chapters = [
        {
            "title": ch.get("title") or "",
            "start": float(ch.get("start") or 0),
            "end": float(ch.get("end") or 0),
        }
        for ch in (media.get("chapters") or [])
    ]

    name = metadata.get("title") or raw.get("name") or "Untitled"
    # ABS library listings return flattened string fields (authorName /
    # narratorName) where the detailed item endpoint returns arrays
    # (authors[] / narrators[]). Handle both so we don't drop credit
    # info depending on which endpoint the catalog fetch uses.
    # Drop non-author role credits ABS bakes into the authors array
    # ("Jenny McKeon - translator") so the author line + match key stay
    # clean. See augmentum/media/normalize.author_for_match.
    author = author_for_match(
        _join_people(metadata.get("authors"))
        or metadata.get("authorName")
        or metadata.get("author")
        or ""
    )
    narrator = (
        _join_people(metadata.get("narrators"))
        or metadata.get("narratorName")
        or ""
    )
    _abs_series_name, _abs_series_seq = _abs_series(metadata)

    return CatalogItem(
        external_id=item_id,
        name=name,
        kind="audio" if lib_kind == "book" else "audio",
        mime_type=mime_type,
        size_bytes=size_bytes,
        duration_ms=duration_ms,
        progress_pct=progress_pct,
        cover_url=cover_rel,           # server-relative; resolved at UI time
        author=author,
        narrator=narrator,
        stream_path=stream_path,
        extra={
            "library_kind": lib_kind,
            "entity_kind": "podcast" if lib_kind == "podcast" else "book",
            "index_without_stream": lib_kind == "podcast",
            # Series — feeds the existing source_metadata.series_name slot
            # (sync.py) so "More in this series" + series grouping light up
            # for audiobooks the way they do for comics.
            **({"series_name": _abs_series_name} if _abs_series_name else {}),
            **({"series_sequence": _abs_series_seq} if _abs_series_seq else {}),
            "episode_count": int(media.get("numEpisodes") or 0),
            "chapters": chapters,
            "audio_files": [
                {
                    "ino": f.get("ino") or f.get("metadata", {}).get("ino") or "",
                    "index": f.get("index"),
                    "duration_ms": int(float(f.get("duration") or 0) * 1000),
                }
                for f in audio_files
            ],
            # Skip reason is set only when the row couldn't derive a
            # stream_path. sync.py reads this to populate
            # user_media_servers.last_sync_skipped for the UI.
            **({"skip_reason": skip_reason} if skip_reason else {}),
        },
    )
