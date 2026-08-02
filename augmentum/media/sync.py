"""Pull a server's catalog into ``file_index``.

The catalog rows land with:
  source     = 'audiobookshelf' / 'emby' / 'jellyfin' / ...
  source_id  = server's opaque item ID (e.g. ABS library item id)
  kind       = derived from mime_type ('audio' / 'video')
  source_metadata = {
      server_id:     <user_media_servers.id>
      provider:      <same as `source`>
      stream_path:   server-relative path the streaming proxy requests
      duration_ms:   total ms, for progress bar math
      progress_pct:  0-100, freshened by progress_routes
      cover_url:     server-relative path (resolved lazily on the UI)
      author:        audiobook author / show creator
      narrator:      audiobook narrator (ABS)
      chapters:      [{title, start, end}, ...] (ABS audiobooks)
      extra:         provider-specific bag
  }

Deletion handling: we do NOT reconcile removed items on a sync. If a
book vanishes from the server, the file_index row stays (it's
technically stale) until a full-resync tool lands. Phase 1 keeps the
happy path small; stale-cleanup is a follow-up.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from augmentum.media.comic_series_store import get_comic_series_store
from augmentum.media.library_store import get_media_library_store
from augmentum.media.normalize import normalize_name
from augmentum.media.providers.audiobookshelf import AudiobookshelfProvider
from augmentum.media.providers.base import (
    CatalogItem,
    provider_supports_library_discovery,
)
from augmentum.media.providers.emby import EmbyProvider
from augmentum.media.providers.jellyfin import JellyfinProvider
from augmentum.media.providers.komga import KomgaProvider
from augmentum.media.providers.librivox import LibrivoxProvider
from augmentum.media.providers.suwayomi import SuwayomiProvider
from augmentum.media.store import MediaServer, MediaServerStore
from augmentum.utils.logging import get_logger
from augmentum.vfs import register_file

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)


async def enqueue_media_sync(
    app_state: Any, *, user_id: str, server_id: str,
) -> str | None:
    """Queue a background catalog sync for a media server, dedup-aware.

    Fired on PROVISION (so a freshly added/connected server indexes into
    ``file_index`` automatically — without this it sits at 0 items until a
    manual Sync, and the companion can't find anything to play) and by the
    periodic re-sync. Returns the job id, or None when there's no jobs infra
    or a sync for this server is already pending/running.
    """
    if not user_id or not server_id:
        return None
    jobs_store = getattr(app_state, "jobs_store", None)
    job_runner = getattr(app_state, "job_runner", None)
    if jobs_store is None or job_runner is None:
        return None
    try:
        existing = await jobs_store.list_for_user(
            user_id=user_id, job_type="media_sync", limit=50,
        )
        for job in existing:
            if (
                job.get("status") in {"pending", "running"}
                and str((job.get("payload") or {}).get("server_id") or "") == server_id
            ):
                return job.get("id")
        job_id = await jobs_store.create(
            user_id=user_id,
            job_type="media_sync",
            payload={"server_id": server_id},
            priority=15,
            max_attempts=2,
        )
        job_runner.wake()
        log.info(
            "media_sync_enqueued_on_provision",
            server_id=server_id, user_id=user_id, job_id=job_id,
        )
        return job_id
    except Exception:
        log.warning(
            "media_sync_enqueue_failed",
            server_id=server_id, user_id=user_id, exc_info=True,
        )
        return None


def _build_provider(provider: str, http_client: httpx.AsyncClient):
    if provider == "audiobookshelf":
        return AudiobookshelfProvider(http_client)
    if provider == "emby":
        return EmbyProvider(http_client)
    if provider == "jellyfin":
        return JellyfinProvider(http_client)
    if provider == "librivox":
        # Parity-only: LibriVox.fetch_catalog returns [] and sync_server
        # short-circuits below before any provider method runs. Keeping
        # the branch means a hypothetical future user-owned LibriVox
        # mirror could reuse this factory without rewiring.
        return LibrivoxProvider(http_client)
    if provider == "komga":
        return KomgaProvider(http_client)
    if provider == "suwayomi":
        return SuwayomiProvider(http_client)
    raise ValueError(f"Unknown media provider: {provider}")


# Providers whose items carry comic series metadata. Kept as a set for
# cheap membership checks in the indexer; expand when new comic-shaped
# providers land (Kavita, OPDS generic).
_COMIC_PROVIDERS: frozenset[str] = frozenset({"komga", "suwayomi"})


def _comic_series_name(item: CatalogItem) -> str:
    """Best-effort series-name extraction from a comic ``CatalogItem``.

    Komga emits ``extra.series_name`` from the owning Series metadata;
    Suwayomi emits the same key from ``manga.title``. Fall back to the
    item's own display name if the provider didn't populate it.
    """
    return str(item.extra.get("series_name") or item.name or "").strip()


def _provider_allows_empty_token(provider: str) -> bool:
    """Providers that can operate with no stored credential."""
    return provider in {"suwayomi"}


def _sync_success_detail(
    *,
    indexed: int,
    total_seen: int,
    skipped_count: int,
    recovered_count: int = 0,
) -> str:
    if total_seen <= 0:
        return "No items found"
    if indexed == total_seen and not skipped_count and not recovered_count:
        return f"Indexed all {indexed} items"
    parts = [f"Indexed {indexed} of {total_seen}"]
    if skipped_count:
        parts.append(f"{skipped_count} skipped")
    if recovered_count:
        noun = "item" if recovered_count == 1 else "items"
        parts.append(f"{recovered_count} {noun} recovered via detail fetch")
    return " · ".join(parts)


def _should_index_without_stream(item: CatalogItem) -> bool:
    entity_kind = str(item.extra.get("entity_kind") or "").strip().lower()
    if item.extra.get("index_without_stream"):
        return True
    return entity_kind in {"series", "season"}


def _comic_alias_names(item: CatalogItem) -> list[str] | None:
    """Best-effort alternate titles for series search and display."""
    raw = item.extra.get("alternate_titles") or []
    if not isinstance(raw, list):
        return None
    canonical = _comic_series_name(item).casefold()
    out: list[str] = []
    seen: set[str] = set()
    for candidate in raw:
        title = str(candidate or "").strip()
        if not title:
            continue
        key = title.casefold()
        if key == canonical or key in seen:
            continue
        seen.add(key)
        out.append(title)
    return out or None


def _comic_series_update_fields(item: CatalogItem, provider: str) -> dict:
    """Translate provider extras into ``ComicSeriesStore.update_series`` kwargs."""
    extra = item.extra if isinstance(item.extra, dict) else {}
    fields: dict = {}

    alias_names = _comic_alias_names(item)
    if alias_names:
        fields["alias_names"] = alias_names

    publisher = str(extra.get("publisher") or "").strip()
    if publisher:
        fields["publisher"] = publisher

    author = str(item.author or extra.get("author") or "").strip()
    if author:
        fields["author"] = author

    description = str(extra.get("description") or extra.get("summary") or "").strip()
    if description:
        fields["description"] = description

    status = str(extra.get("status") or "").strip().lower()
    if status:
        fields["status"] = status

    genres = extra.get("genres")
    if isinstance(genres, list):
        clean_genres = [str(g).strip() for g in genres if str(g).strip()]
        if clean_genres:
            fields["genres"] = clean_genres

    language = str(extra.get("language_iso") or extra.get("language") or "").strip().lower()
    if language:
        fields["language_iso"] = language

    age_rating = str(extra.get("age_rating") or "").strip()
    if age_rating:
        fields["age_rating"] = age_rating

    archive_count = extra.get("total_book_count")
    try:
        if archive_count is not None and str(archive_count).strip() != "":
            fields["archive_count_reported"] = int(archive_count)
    except (TypeError, ValueError):
        pass

    fields["metadata_source"] = provider
    fields["metadata_confidence"] = 0.9
    return {k: v for k, v in fields.items() if v not in ("", None)}


# Per-user state that arrives from the provider under the SERVER OWNER's
# credential. On a borrowed (admin-shared) server these describe the
# owner's viewing life, not the syncing user's, and must never reach the
# borrower's file_index row.
#
# Two independent channels deliver these, which is why stripping happens
# at the item level rather than only around fetch_progress():
#   1. fetch_progress() — explicit, all providers.
#   2. fetch_catalog() itself — Emby/Jellyfin request
#      ``EnableUserData: true`` so every catalog row carries the owner's
#      UserData (emby_compat ``_parse_item``), and Komga folds readProgress
#      into its parse. Audiobookshelf is clean here; it only leaks via (1).
_OWNER_USER_STATE_EXTRA_KEYS = (
    "current_time_s",
    "is_finished",
    "unplayed_count",
    "is_favorite",
    "play_count",
    "current_page",
    "last_read_at",
)


def _strip_owner_user_state(item: CatalogItem) -> None:
    """Erase provider-side per-user fields from ``item``, in place.

    Called for every item of a borrowed server. Clearing the CatalogItem
    is only half the job — ``_index_item`` must also OMIT these keys from
    the payload rather than writing zeros, so that
    ``FileIndexService.register``'s preserve-merge keeps the borrower's
    own locally-tracked progress. See ``_index_item``.
    """
    item.progress_pct = 0.0
    for key in _OWNER_USER_STATE_EXTRA_KEYS:
        item.extra.pop(key, None)


async def sync_server(
    server: MediaServer,
    *,
    store: MediaServerStore,
    http_client: httpx.AsyncClient,
    progress_callback: Callable[[float, str], Awaitable[None]] | None = None,
    target_user_id: str = "",
    bulk: Any = None,
) -> tuple[int, str]:
    """Fetch and index every catalog item for a single server.

    ``bulk`` is an optional :class:`augmentum.vfs.bulk.BulkIndexSession`.
    Background callers (the ``media_sync`` job) pass one so the indexing
    writes land on a dedicated connection with batched commits and an
    event-loop yield between batches, instead of enqueuing one commit per
    item onto the shared connection's single aiosqlite worker thread and
    stuttering concurrent voice/chat traffic. Passing None preserves the
    original per-row-commit behavior for direct/synchronous callers.

    ``target_user_id`` is the user the resulting per-user rows are
    written under (file_index, library_views, comic_series). For a
    private server it equals ``server.user_id`` (the owner). For an
    admin-shared server it's the caller — the non-admin user clicking
    "Sync" — so their personal catalog gets populated without the
    rows leaking under the admin's id. Defaults to ``server.user_id``
    to preserve the historical behavior for non-shared servers.

    The server-row writes (``store.update`` on status / last_sync_at /
    item_count / total_seen) still use ``server.user_id`` — for shared
    servers the non-owner's writes silently no-op against the admin's
    row, which is what we want (admin's stats stay admin's; non-owner
    progress shows up on the job row, not the server row).

    Returns (items_indexed, status_detail). On failure, status_detail
    carries a human-readable message that ends up in the UI.
    """
    # uid = the user the per-user rows belong to. Captured once so a
    # later refactor that adds new "write X under user Y" sites has
    # a single source of truth to reach for.
    uid = target_user_id or server.user_id
    # LibriVox is a built-in free library with a ~20k-book catalog; it's
    # never synced wholesale into file_index (see sync.py module docstring
    # and the LibriVox plan). If a caller somehow routes a librivox server
    # through here, short-circuit loudly rather than doing 20,000 inserts.
    if server.provider == "librivox":
        log.info("media_sync_skipped_librivox", server_id=server.id)
        return 0, "LibriVox is browse-only; items are added individually via pin"

    if not server.access_token and not _provider_allows_empty_token(server.provider):
        return 0, "No access token — add credentials first"

    try:
        provider = _build_provider(server.provider, http_client)
    except ValueError as exc:
        return 0, str(exc)

    async def _report(progress: float, stage: str) -> None:
        if progress_callback is None:
            return
        await progress_callback(progress, stage)

    library_store = get_media_library_store()
    library_lookup: dict[str, object] = {}
    if (
        library_store is not None
        and provider_supports_library_discovery(provider)
    ):
        try:
            discovered = await provider.discover_libraries(
                server.base_url, server.access_token,
            )
            await library_store.upsert_discovered(
                user_id=uid,
                server_id=server.id,
                provider=server.provider,
                libraries=discovered,
            )
            library_lookup = await library_store.active_by_provider_id(
                user_id=uid,
                server_id=server.id,
            )
        except Exception as exc:
            log.warning(
                "media_library_discovery_failed",
                server_id=server.id,
                provider=server.provider,
                error=str(exc),
            )

    await _report(0.10, "Fetching catalog")
    try:
        items = await provider.fetch_catalog(server.base_url, server.access_token)
    except Exception as exc:
        log.warning(
            "media_sync_failed", server_id=server.id,
            provider=server.provider, error=str(exc),
        )
        await store.update(
            server.id, user_id=server.user_id,
            status="error", status_detail=f"Sync failed: {exc}",
        )
        return 0, f"Sync failed: {exc}"

    # Progress is a one-call batch fetch (ABS exposes mediaProgress on
    # /api/me). Failures here aren't fatal — we just lose resume state
    # until the next sync, which is fine degradation.
    # A borrowed server's token is the OWNER's, so fetch_progress would
    # return the owner's resume points and played flags. Skip the call
    # entirely — it's both a leak and a wasted round-trip. The borrower's
    # progress is tracked Augmentum-side (see MediaServer.is_borrowed_by).
    borrowed = server.is_borrowed_by(uid)
    progress_by_id: dict[str, dict] = {}
    if borrowed:
        log.info(
            "media_sync_owner_progress_skipped",
            server_id=server.id, user_id=uid, owner_id=server.user_id,
        )
    else:
        await _report(0.35, "Fetching progress")
        try:
            progress_by_id = await provider.fetch_progress(
                server.base_url, server.access_token,
            )
        except Exception as exc:
            log.warning(
                "media_progress_fetch_failed", server_id=server.id, error=str(exc),
            )

    # Partition items into indexable vs skipped. We collect the first
    # 30 skipped titles + their reason codes so the Media Servers UI
    # can surface "141 skipped · see why" without asking the user to
    # dig through docker logs. 30 is enough to spot a pattern (usually
    # all the same reason) without bloating the row past ~2KB of JSON.
    SKIPPED_SAMPLE_CAP = 30
    indexed = 0
    skipped_count = 0
    recovered_count = 0
    skipped_sample: list[dict] = []
    total_items = len(items)
    if total_items == 0:
        await _report(0.95, "No items found")
    for pos, item in enumerate(items, start=1):
        if item.extra.get("recovered_via_detail"):
            recovered_count += 1
        if not item.stream_path and not _should_index_without_stream(item):
            skipped_count += 1
            if len(skipped_sample) < SKIPPED_SAMPLE_CAP:
                skipped_sample.append({
                    "title":  item.name,
                    "author": item.author,
                    "reason": item.extra.get("skip_reason") or "unknown",
                })
            continue
        library_view_id = str(item.extra.get("library_view_id") or "").strip()
        if library_view_id and library_lookup:
            library_row = library_lookup.get(library_view_id)
            if library_row is not None:
                surface_group = getattr(library_row, "surface_group", "")
                if getattr(library_row, "is_hidden", False):
                    skipped_count += 1
                    if len(skipped_sample) < SKIPPED_SAMPLE_CAP:
                        skipped_sample.append({
                            "title": item.name,
                            "author": item.author,
                            "reason": "hidden_library",
                        })
                    continue
                if surface_group and surface_group not in {"movies", "shows", "music_videos"}:
                    skipped_count += 1
                    if len(skipped_sample) < SKIPPED_SAMPLE_CAP:
                        skipped_sample.append({
                            "title": item.name,
                            "author": item.author,
                            "reason": "unsupported_library_group",
                        })
                    continue
        if borrowed:
            # Channel 2: the catalog parse itself already folded the
            # owner's UserData into this item. Clear it before it can
            # reach the borrower's row. progress_by_id is empty here, so
            # the branch below is a no-op — both channels are closed.
            _strip_owner_user_state(item)
        prog = progress_by_id.get(item.external_id)
        if prog:
            item.progress_pct = prog.get("progress", 0.0) * 100.0
            item.extra["current_time_s"] = prog.get("current_time_s", 0.0)
            item.extra["is_finished"] = prog.get("is_finished", False)
        await _index_item(
            server=server, item=item, target_user_id=uid, bulk=bulk,
            strip_user_state=borrowed,
        )
        indexed += 1
        if bulk is not None:
            # Commit cadence + the yield that lets voice/chat/ticks
            # interleave. Must run per item, not per progress report.
            await bulk.tick()
        if total_items and (pos == total_items or pos == 1 or pos % 25 == 0):
            await _report(
                0.45 + (0.50 * (pos / total_items)),
                f"Indexing {pos}/{total_items}",
            )

    if bulk is not None:
        # Durability before we claim success. The server row is written on
        # the SHARED connection while catalog rows sit on the sidecar, so
        # without this flush a crash in the gap would leave status='ok'
        # and item_count set against rows that were never committed.
        await bulk.flush()

    now_iso = datetime.now(tz=UTC).isoformat()
    summary = _sync_success_detail(
        indexed=indexed,
        total_seen=total_items,
        skipped_count=skipped_count,
        recovered_count=recovered_count,
    )
    await store.update(
        server.id, user_id=server.user_id,
        status="ok", status_detail=summary, last_sync_at=now_iso,
        item_count=indexed,
        total_seen=total_items,
        skipped_count=skipped_count,
        last_sync_skipped=skipped_sample,
    )
    log.info(
        "media_sync_done", server_id=server.id, provider=server.provider,
        items=indexed, total_seen=total_items, skipped=skipped_count,
        recovered=recovered_count,
    )
    await _report(1.0, summary)
    return indexed, ""


async def _index_item(
    *,
    server: MediaServer,
    item: CatalogItem,
    target_user_id: str,
    bulk: Any = None,
    strip_user_state: bool = False,
) -> None:
    """Upsert one catalog item into file_index.

    ``strip_user_state`` is set for a borrowed (admin-shared) server. It
    OMITS the per-user playback keys from ``source_metadata`` rather than
    writing zeros — the distinction is load-bearing. ``register``'s
    preserve-merge only restores a stored value when the key is ABSENT
    from the incoming payload, so writing ``progress_pct: 0.0`` would
    clobber the borrower's own progress on every sync, while omitting it
    lets their locally-tracked state survive untouched.

    ``bulk`` is an optional :class:`augmentum.vfs.bulk.BulkIndexSession`.
    When present, the file_index and comic_series writes go through its
    dedicated connection and batched commit cadence instead of the
    process-wide shared connection — see that module for why. When None
    (the synchronous route path, tests), behavior is unchanged.

    Uses `source_id = f"{server.id}:{item.external_id}"` so the same
    external id on two of the user's servers produces two rows. ABS
    item IDs are already globally unique, but the prefix keeps the
    invariant cleanly when we add Emby/Jellyfin.

    ``target_user_id`` is the user the file_index/comic_series rows
    are written under. For private servers this equals
    ``server.user_id`` (the owner). For admin-shared servers it's
    the caller running the sync — see ``sync_server`` for the full
    rationale.

    Comic items (``kind='comic'`` from Komga/Suwayomi/future Kavita) get
    one extra step: we resolve the owning series through
    :class:`ComicSeriesStore` so every chapter/volume in the same series
    shares a stable ``series_id``. Failure to resolve is non-fatal —
    the item still lands in file_index without a series_id, and the
    Library surface degrades to "unknown series" grouping instead of
    losing the row.
    """
    source = server.provider
    source_id = f"{server.id}:{item.external_id}"
    # Progress is carried in source_metadata; `current_time_s` comes from
    # the provider's progress fetch when present. Stored in seconds so the
    # audio element can seek directly without unit conversion.
    current_time_s = float(item.extra.get("current_time_s") or 0)
    is_finished = bool(item.extra.get("is_finished") or False)
    duration_s = item.duration_ms / 1000.0 if item.duration_ms else 0.0

    # Comic-shaped items: resolve the owning series through the store.
    # Remote providers deliver authoritative series metadata via their
    # APIs, so confidence is high (0.9). We don't set 1.0 — that's
    # reserved for user-verified metadata where the human has confirmed
    # the identity. Remote APIs can still get an item wrong.
    series_id: str | None = None
    metadata_confidence = 0.5
    if item.kind == "comic" and server.provider in _COMIC_PROVIDERS:
        series_store = bulk.series_store if bulk is not None else get_comic_series_store()
        series_name = _comic_series_name(item)
        if series_store and series_name:
            try:
                series_id = await series_store.create_or_resolve_series(
                    user_id=target_user_id,
                    name=series_name,
                    metadata_source=server.provider,
                    metadata_confidence=0.9,
                    publisher=item.extra.get("publisher") or None,
                    author=item.author or item.extra.get("author") or None,
                    language_iso=item.extra.get("language") or None,
                )
                metadata_confidence = 0.9
                update_fields = _comic_series_update_fields(item, server.provider)
                if update_fields:
                    await series_store.update_series(
                        series_id,
                        user_id=target_user_id,
                        **update_fields,
                    )
            except Exception as exc:
                # Store lookup/insert failed — log and proceed without
                # grouping. The row still lands; it just won't join with
                # its siblings until a later sync resolves it.
                log.warning(
                    "comic_series_resolve_failed",
                    user_id=target_user_id,
                    provider=server.provider,
                    series_name=series_name,
                    error=str(exc),
                )

    payload = dict(
        user_id=target_user_id,
        source=source,
        source_id=source_id,
        name=item.name,
        mime_type=item.mime_type,
        size_bytes=item.size_bytes,
        real_path=None,                 # not on disk — streamed
        thumbnail=None,
        scan_status="ok",               # remote catalog is authoritative
        metadata_confidence=metadata_confidence,
        series_id=series_id,
        source_metadata={
            "server_id":           server.id,
            "provider":            server.provider,
            "external_id":         item.external_id,
            "stream_path":         item.stream_path,
            "duration_ms":         item.duration_ms,
            "duration_s":          duration_s,
            "progress_pct":        item.progress_pct,
            "current_time_s":      current_time_s,
            "is_finished":         is_finished,
            "cover_url":           item.cover_url,
            "has_cover":           bool(item.cover_url),
            "author":              item.author,
            "narrator":            item.narrator,
            "entity_kind":         item.extra.get("entity_kind") or "",
            "library_view_id":     item.extra.get("library_view_id") or "",
            "library_name":        item.extra.get("library_name") or "",
            "provider_collection_type": item.extra.get("provider_collection_type") or "",
            "provider_item_type":  item.extra.get("provider_item_type") or "",
            "parent_external_id":  item.extra.get("parent_external_id") or "",
            "grandparent_external_id": item.extra.get("grandparent_external_id") or "",
            "year":                item.extra.get("year") or 0,
            "genres":              item.extra.get("genres") or [],
            "series_name":         item.extra.get("series_name") or "",
            "series_sequence":     item.extra.get("series_sequence") or "",
            "series_normalized":   normalize_name(item.extra.get("series_name") or ""),
            "season_number":       item.extra.get("season_number") or 0,
            "episode_number":      item.extra.get("episode_number") or 0,
            "unplayed_count":      item.extra.get("unplayed_count") or 0,
            "overview":            item.extra.get("overview") or "",
            # Canonical forms for related-item lookups: equality on
            # these fields powers "other books by this author" without
            # the raw-string brittleness of "J.F. Brink" vs "J F Brink".
            "author_normalized":   normalize_name(item.author),
            "narrator_normalized": normalize_name(item.narrator),
            "chapters":            item.extra.get("chapters") or [],
            "extra":               {k: v for k, v in item.extra.items()
                                    if k not in ("chapters", "current_time_s",
                                                 "is_finished")},
        },
    )

    if strip_user_state:
        # Drop, don't zero — see the docstring. Whatever the borrower has
        # tracked Augmentum-side for these fields stays put.
        meta = payload["source_metadata"]
        for key in ("progress_pct", "current_time_s", "is_finished",
                    "unplayed_count"):
            meta.pop(key, None)
        nested = meta.get("extra")
        if isinstance(nested, dict):
            for key in _OWNER_USER_STATE_EXTRA_KEYS:
                nested.pop(key, None)

    if bulk is None:
        await register_file(**payload)
        return
    # Mirror register_file's contract: a single bad row must not abort a
    # 63k-item scan. Swallow, log, move on — the row is simply missing
    # until the next sync. Note this leaves the batch's transaction
    # intact; safe_rollback only fires when the DML itself raised.
    try:
        await bulk.file_index.register(**payload)
    except Exception:
        log.warning(
            "file_register_failed",
            source=source, source_id=source_id, exc_info=True,
        )
