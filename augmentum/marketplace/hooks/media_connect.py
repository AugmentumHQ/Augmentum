"""media_connect hook — wire a provisioned service into the media stack.

This is the existing behaviour extracted from
``install_dispatchers._install_service_manifest`` into the hook
registry. When a manifest declares ``integration.media_connect``,
the installed service gets a per-user ``user_media_servers`` row,
managed credentials login, and a catalog sync — exactly the path
the standalone ``kind=media_server`` loader follows for Jellyfin /
Suwayomi / Audiobookshelf.

The hook config (from the manifest's ``integration.media_connect``
dict) may carry:
  * ``provider`` — the media provider name (defaults to the service id).
"""

from __future__ import annotations

from typing import Any

from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS, HookMeta
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def _install(request: Any, manifest: Any, sd: Any, user_id: str) -> None:
    """Provision the per-user media connection after container start."""
    cfg = manifest.integration.get("media_connect", {}) or {}
    provider = str(cfg.get("provider") or manifest.service_id)

    from augmentum.marketplace.install_dispatchers import _connect_media_server

    try:
        await _connect_media_server(
            request, sd=sd, service_id=manifest.service_id,
            provider=provider, user_id=user_id,
        )
    except Exception:
        log.warning(
            "media_connect_hook_failed",
            service_id=manifest.service_id, exc_info=True,
        )


async def _uninstall(request: Any, manifest: Any, sd: Any, user_id: str) -> None:
    """Remove the per-user media connection on uninstall.

    The shared container is stopped by the generic uninstall dispatcher
    before hooks run; this hook only cleans up the user's connection row
    + cached library. It's best-effort — a missing row or already-purged
    cache must not block uninstall.
    """
    cfg = manifest.integration.get("media_connect", {}) or {}
    provider = str(cfg.get("provider") or manifest.service_id)

    from augmentum.marketplace.install_dispatchers import _media_server_store
    from augmentum.media.store import purge_server_data

    store = _media_server_store(request)
    idx = getattr(request.app.state, "file_index", None)
    if store is None:
        return

    removed = 0
    try:
        from augmentum.media.store import MediaServerStore
    except ImportError:
        log.warning("media_connect_uninstall_no_store", service_id=manifest.service_id)
        return

    try:
        rows = await store.list_visible(user_id=user_id)
    except Exception:
        log.warning("media_connect_uninstall_list_failed", exc_info=True)
        return

    for s in rows:
        if s.provider != provider or s.user_id != user_id:
            continue
        try:
            if idx is not None:
                await purge_server_data(idx._db, s.id, user_id=user_id)
            await store.delete(s.id, user_id=user_id)
            removed += 1
        except Exception:
            log.warning(
                "media_connect_uninstall_row_failed",
                server_id=s.id, exc_info=True,
            )

    if removed:
        log.info(
            "media_connect_uninstalled",
            service_id=manifest.service_id, connections_removed=removed,
        )


KNOWN_INTEGRATION_HOOKS["media_connect"] = (
    _install,
    _uninstall,
    HookMeta(
        label="Media Library",
        icon="📚",
        companion_hint="Companion can play your media",
        status_provider="media_connect",
    ),
)
