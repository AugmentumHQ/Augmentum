"""Community catalog feed loader — Phase 2 of the Discover surface.

Fetches a JSON index from ``settings.discover_community_feed_url``
(augmentumhq.com/community/index.json by default) and upserts each
entry into ``marketplace_listings`` under a publisher-scoped namespace
``community:<handle>``. Each contributor owns their slice; the loader
delists only what the index dropped on the most recent successful
pull — fetch failures leave the catalog untouched (better stale than
empty during a flaky network).

Index schema (v1):

    {
      "version": 1,
      "generated_at": "2026-06-10T00:00:00Z",
      "listings": [
        {
          "id": "community:alice/spotify-bridge",
          "publisher": "community:alice",
          "title": "Spotify Bridge",
          "kind": "power",
          "category": "powers",          // optional, derived from kind if missing
          "tagline": "...",
          "description": "...",
          "thumbnail_url": "...",
          "source_url": "https://raw.githubusercontent.com/alice/.../POWER.md",
          "install_via": "community-power",
          "install_payload": {            // forwarded to the dispatcher
            "manifest_url": "https://raw.githubusercontent.com/alice/.../manifest.yaml"
          },
          "capabilities": {...},
          "metadata": {"version": "1.0", "license": "MIT", ...},
          "tags": ["spotify", "music"],
          "featured": false,
          "signature": ""                 // reserved for Phase 3
        },
        ...
      ]
    }

Trust model: the feed_url is the trust root. Operators who don't
trust augmentumhq.com point the setting at their own mirror.
Per-entry signatures are Phase 3.

Scheduling: ``schedule_community_feed_refresh`` is the lifespan helper;
it pulls once at startup (off the critical path) and then on the
``discover_community_feed_refresh_minutes`` cadence. Idempotent —
each pull is independent.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from augmentum.config import settings
from augmentum.marketplace.store import MarketplaceListing, MarketplaceStore
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_COMMUNITY_PUBLISHER_PREFIX = "community:"
_MAX_INDEX_BYTES = 2 * 1024 * 1024              # 2 MB cap; catalog is JSON of metadata
_FETCH_TIMEOUT_S = 20.0
_REQUIRED_FIELDS = ("id", "title", "kind", "install_via")


class CommunityFeedError(Exception):
    """Raised by the loader on unrecoverable failure (parse / missing
    listings array). Network failures don't raise — they log and
    leave the catalog untouched."""


async def _fetch_index(url: str) -> dict | None:
    """Pull the index JSON. Returns the parsed dict or None on
    transient failure. Caps response at _MAX_INDEX_BYTES.

    Supports ``file://`` URLs for local-fixture testing — operators
    point ``discover_community_feed_url`` at a path inside the
    container to dry-run the loader without standing up an HTTP
    server. The file path is treated as trusted (operator opt-in).
    """
    # file:// fallback — read directly off disk. Keep size cap.
    if url.startswith("file://"):
        try:
            from pathlib import Path
            local_path = Path(url[len("file://"):])
            if not local_path.is_file():
                log.warning(
                    "community_feed_file_not_found", url=url,
                )
                return None
            size = local_path.stat().st_size
            if size > _MAX_INDEX_BYTES:
                log.warning(
                    "community_feed_too_large",
                    url=url, max_bytes=_MAX_INDEX_BYTES,
                )
                return None
            return json.loads(local_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log.warning(
                "community_feed_parse_failed",
                url=url, error=str(exc)[:200],
            )
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "community_feed_file_read_failed",
                url=url, error=str(exc)[:200],
            )
            return None

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_FETCH_TIMEOUT_S),
            follow_redirects=True,
            headers={"User-Agent": "Augmentum/1.0 (discover-community-feed)"},
        ) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                buf = bytearray()
                async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                    buf.extend(chunk)
                    if len(buf) > _MAX_INDEX_BYTES:
                        log.warning(
                            "community_feed_too_large",
                            url=url, max_bytes=_MAX_INDEX_BYTES,
                        )
                        return None
        return json.loads(bytes(buf).decode("utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("community_feed_parse_failed", url=url, error=str(exc)[:200])
        return None
    except httpx.HTTPError as exc:
        log.warning("community_feed_fetch_failed", url=url, error=str(exc)[:200])
        return None
    except Exception as exc:  # noqa: BLE001 — defensive
        log.warning("community_feed_unexpected_failure", url=url, error=str(exc)[:200])
        return None


def _validate_entry(entry: dict) -> MarketplaceListing | None:
    """Lenient per-entry validation. Returns None to skip on invalid
    inputs — one bad entry mustn't block the rest of the catalog."""
    for f in _REQUIRED_FIELDS:
        if not entry.get(f):
            return None
    publisher = str(entry.get("publisher") or "").strip()
    if not publisher.startswith(_COMMUNITY_PUBLISHER_PREFIX):
        # Enforce publisher namespace — keeps the delist sweep scoped
        # so community entries can't accidentally clobber augmentum-
        # curated rows. The index publisher is responsible for prefix.
        return None

    raw_tags = entry.get("tags") or []
    if not isinstance(raw_tags, list):
        raw_tags = []

    runtime_alternates = entry.get("runtime_alternates") or []
    if not isinstance(runtime_alternates, list):
        runtime_alternates = []

    # Category: explicit if present, else derived from kind.
    category = str(entry.get("category") or "").strip()
    if not category:
        kind = str(entry["kind"])
        category_map = {
            "character": "characters",
            "reasoning_flow": "reasoning-flows",
            "power": "powers",
            "knowledge_pack": "knowledge",
            "streamed_game": "games",
            "js13k_game": "games",
            "web_app": "games",
        }
        category = category_map.get(kind, "other")

    try:
        return MarketplaceListing(
            id=str(entry["id"]),
            publisher=publisher,
            title=str(entry["title"]),
            kind=str(entry["kind"]),
            runtime_preferred=str(entry.get("runtime_preferred") or ""),
            runtime_alternates=tuple(str(a) for a in runtime_alternates),
            tagline=str(entry.get("tagline") or ""),
            description=str(entry.get("description") or ""),
            thumbnail_url=str(entry.get("thumbnail_url") or ""),
            source_url=str(entry.get("source_url") or ""),
            embed_url=str(entry.get("embed_url") or ""),
            install_via=str(entry["install_via"]),
            install_payload=entry.get("install_payload") or {},
            capabilities=entry.get("capabilities") or {},
            metadata=entry.get("metadata") or {},
            rating=_as_float(entry.get("rating")),
            install_count=int(entry.get("install_count") or 0),
            signature=str(entry.get("signature") or ""),
            listed_at="",
            category=category,
            tags=tuple(str(t) for t in raw_tags if str(t).strip()),
            featured=bool(entry.get("featured") or False),
        )
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None


async def load_community_into_store(
    store: MarketplaceStore,
    *,
    feed_url: str | None = None,
) -> dict[str, int]:
    """Pull the community index and upsert into marketplace_listings.

    Returns ``{"loaded": N, "skipped": M, "delisted": K}``. On fetch
    failure returns zeros without touching the catalog (so a stale
    network doesn't blank out community items).
    """
    url = feed_url or getattr(settings, "discover_community_feed_url", "")
    if not url:
        log.info("community_feed_disabled_no_url")
        return {"loaded": 0, "skipped": 0, "delisted": 0}

    index = await _fetch_index(url)
    if index is None:
        # Transient or permanent failure — leave the catalog as-is.
        return {"loaded": 0, "skipped": 0, "delisted": 0}

    listings_raw = index.get("listings") if isinstance(index, dict) else None
    if not isinstance(listings_raw, list):
        log.warning("community_feed_no_listings_array", url=url)
        return {"loaded": 0, "skipped": 0, "delisted": 0}

    # Group entries by publisher so the delist sweep is scoped per
    # contributor — one publisher dropping their submissions doesn't
    # delist another publisher's still-active listings.
    by_publisher: dict[str, set[str]] = {}
    loaded = 0
    skipped = 0
    for entry in listings_raw:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        listing = _validate_entry(entry)
        if listing is None:
            skipped += 1
            continue
        await store.upsert(listing)
        by_publisher.setdefault(listing.publisher, set()).add(listing.id)
        loaded += 1

    # Delist any community: rows that vanished from this pull.
    # Scoped per publisher to preserve the multi-contributor model.
    delisted = 0
    for publisher, seen in by_publisher.items():
        delisted += await store.delist_missing_for_publisher(
            seen, publisher=publisher,
        )

    log.info(
        "marketplace_community_loaded",
        loaded=loaded, skipped=skipped, delisted=delisted,
        url=url,
    )
    return {"loaded": loaded, "skipped": skipped, "delisted": delisted}


# ── Lifespan scheduler ───────────────────────────────────────────────


def schedule_community_feed_refresh(
    app, store: MarketplaceStore,
) -> asyncio.Task | None:
    """Spawn the background refresh loop and stash the task on app.state.

    Returns the task (or None if disabled) so the caller can hold a
    reference — without one, asyncio garbage-collects the task and
    the periodic pull silently stops.
    """
    if not getattr(settings, "discover_community_feed_enabled", True):
        log.info("community_feed_disabled_by_setting")
        # Reconcile: the feed is the only loader for community:* rows, so
        # while it's disabled any active ones are stale placeholders (the
        # sample "example" Character/Power cards). Delist them once, off
        # the critical path, so Discover doesn't show empty lanes.
        async def _cleanup():
            await asyncio.sleep(5)
            try:
                n = await store.delist_community_listings()
                if n:
                    log.info("community_placeholders_delisted", count=n)
            except Exception:  # noqa: BLE001 — never let cleanup break startup
                log.warning("community_placeholder_cleanup_failed", exc_info=True)

        task = asyncio.create_task(_cleanup(), name="discover_community_cleanup")
        if app is not None:
            app.state.discover_community_cleanup_task = task
        return None

    interval_minutes = int(
        getattr(settings, "discover_community_feed_refresh_minutes", 360),
    )
    interval_s = max(60, interval_minutes * 60)

    async def _loop():
        # First pull happens after a short delay so it doesn't block
        # the rest of lifespan startup. Subsequent pulls run on
        # interval_s cadence.
        await asyncio.sleep(15)
        while True:
            try:
                stats = await load_community_into_store(store)
                log.info("marketplace_community_load_summary", **stats)
            except Exception:  # noqa: BLE001 — never let a refresh kill the loop
                log.warning("community_feed_refresh_failed", exc_info=True)
            try:
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                return

    task = asyncio.create_task(_loop(), name="discover_community_feed")
    if app is not None:
        app.state.discover_community_feed_task = task
    return task
