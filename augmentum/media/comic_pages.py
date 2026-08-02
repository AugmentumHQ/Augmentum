"""Shared comic page fetching — used by the per-page image route AND the
comic-narration synth job.

Comic pages live on external providers (Komga, Suwayomi) and are delivered
one image at a time, with per-provider indexing + auth. This module owns the
URL-building + fetch + Suwayomi prepare-retry so the streaming route
(`media_routes.comic_page`) and the offline OCR/narration job share one
implementation instead of drifting.

All functions are app-state driven (no `Request`), so a background job can use
them. The route resolves the same pieces from `request.app.state`.
"""

from __future__ import annotations

import httpx

from augmentum.media.providers.komga import KomgaProvider
from augmentum.media.providers.suwayomi import SuwayomiProvider
from augmentum.media.store import MediaServerStore
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def page_meta(entry) -> dict | None:
    """Pull (server_id, provider, external_id, extra) off a file-index entry."""
    meta = entry.source_metadata if isinstance(getattr(entry, "source_metadata", None), dict) else {}
    server_id = meta.get("server_id", "")
    provider = meta.get("provider", "")
    external_id = meta.get("external_id", "")
    if not server_id or not provider or not external_id:
        return None
    extra = meta.get("extra") if isinstance(meta.get("extra"), dict) else {}
    return {
        "server_id": server_id,
        "provider": provider,
        "external_id": external_id,
        "extra": extra,
        "meta": meta,
    }


def build_page_url(
    server_base: str,
    provider: str,
    external_id: str,
    page_1indexed: int,
    *,
    want_thumb: bool = False,
    want_raw: bool = False,
) -> str | None:
    """Per-provider upstream URL for page ``page_1indexed`` (1-indexed externally)."""
    base = server_base.rstrip("/")
    if provider == "komga":
        suffix = "/thumbnail" if want_thumb else ("/raw" if want_raw else "")
        return f"{base}/api/v1/books/{external_id}/pages/{page_1indexed}{suffix}"
    if provider == "suwayomi":
        parts = external_id.split(".")
        if len(parts) < 2:
            return None
        # Suwayomi pages are 0-indexed upstream.
        return f"{base}/api/v1/manga/{parts[0]}/chapter/{parts[1]}/page/{page_1indexed - 1}"
    return None


def _auth_headers(server) -> dict:
    headers: dict = {}
    if getattr(server, "access_token", ""):
        headers["Authorization"] = f"Basic {server.access_token}"
    return headers


async def fetch_page_bytes(
    http_client: httpx.AsyncClient,
    server,
    provider: str,
    external_id: str,
    extra: dict,
    page_1indexed: int,
    *,
    timeout_s: float = 20.0,
) -> bytes | None:
    """Fetch one page's image bytes, or None. Mirrors the route's logic,
    including the Suwayomi prepare-chapter retry on a 404."""
    url = build_page_url(server.base_url, provider, external_id, page_1indexed)
    if not url:
        return None
    headers = _auth_headers(server)

    async def _once() -> tuple[int, bytes | None]:
        try:
            async with http_client.stream(
                "GET", url, headers=headers, timeout=timeout_s, follow_redirects=True,
            ) as up:
                if up.status_code < 200 or up.status_code >= 400:
                    await up.aread()
                    return up.status_code, None
                return up.status_code, await up.aread()
        except httpx.RequestError as exc:
            log.warning("comic_page_fetch_failed", provider=provider, page=page_1indexed, error=str(exc)[:160])
            return 0, None

    status, body = await _once()
    if status == 404 and provider == "suwayomi" and body is None:
        chapter_db_id = (extra or {}).get("chapter_db_id")
        if chapter_db_id:
            try:
                await SuwayomiProvider(http_client).prepare_chapter(
                    server.base_url, server.access_token, chapter_db_id=int(chapter_db_id),
                )
            except Exception as exc:  # noqa: BLE001 — best-effort prepare
                log.warning("comic_page_prepare_failed", page=page_1indexed, error=str(exc)[:160])
            else:
                status, body = await _once()
    return body


async def resolve_page_count(http_client, server, provider, external_id, extra) -> int:
    """Authoritative page count: refresh from the provider, fall back to the
    cached ``extra.page_count``."""
    page_count = int((extra or {}).get("page_count") or 0)
    try:
        if provider == "suwayomi":
            chapter_db_id = (extra or {}).get("chapter_db_id")
            if chapter_db_id:
                fresh = await SuwayomiProvider(http_client).prepare_chapter(
                    server.base_url, server.access_token, chapter_db_id=int(chapter_db_id),
                )
                if fresh > 0:
                    page_count = fresh
        elif provider == "komga":
            raw = await KomgaProvider(http_client).fetch_item_details(
                server.base_url, server.access_token, external_id=external_id,
            )
            if isinstance(raw, dict):
                pc = int((raw.get("media") or {}).get("pagesCount") or 0)
                if pc > 0:
                    page_count = pc
    except Exception as exc:  # noqa: BLE001 — fall back to cached count
        log.warning("comic_page_count_refresh_failed", provider=provider, error=str(exc)[:160])
    return page_count


async def open_comic_source(app, entry, *, user_id: str) -> dict | None:
    """Resolve everything the synth job needs to page through a comic:
    ``{server, provider, external_id, extra, http_client, page_count}`` or None.
    """
    pm = page_meta(entry)
    if not pm or pm["provider"] not in ("komga", "suwayomi"):
        return None
    sm = getattr(app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    conn = getattr(backend, "conn", None)
    http_client = getattr(app.state, "http_client", None)
    if conn is None or http_client is None:
        return None
    store = MediaServerStore(conn)
    server = await store.get_visible(pm["server_id"], user_id=user_id)
    if not server:
        return None
    page_count = await resolve_page_count(
        http_client, server, pm["provider"], pm["external_id"], pm["extra"],
    )
    return {
        "server": server,
        "provider": pm["provider"],
        "external_id": pm["external_id"],
        "extra": pm["extra"],
        "http_client": http_client,
        "page_count": page_count,
    }
