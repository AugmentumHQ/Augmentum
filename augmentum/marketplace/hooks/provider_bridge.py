"""provider_bridge hook — register a provisioned service as an Augmentum
provider (music, translate, image, …).

Phase 3: Subsonic music bridge (Navidrome/Jellyfin/Airsonic/Gonic).
Subsequent phases add translate, image, and other provider types.
"""

from __future__ import annotations

from typing import Any

from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS, HookMeta
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def _install(request: Any, manifest: Any, sd: Any, user_id: str) -> None:
    """Register a provisioned service as a provider.

    Supported protocols:
      * ``subsonic`` — music server (Navidrome, Jellyfin, Airsonic).
        Pings the server, verifies credentials, creates a per-user
        connection via ``_connect_media_server`` so the library
        appears in Files and ``media.play`` can search + stream.
    """
    cfg = manifest.integration.get("provider_bridge", {}) or {}
    protocol = str(cfg.get("protocol") or "").strip().lower()

    if protocol == "subsonic":
        await _install_subsonic(request, manifest, sd, user_id)
    else:
        log.info(
            "provider_bridge_unknown_protocol",
            service_id=manifest.service_id, protocol=protocol,
        )


async def _uninstall(request: Any, manifest: Any, sd: Any, user_id: str) -> None:
    """Remove the provider registration."""
    cfg = manifest.integration.get("provider_bridge", {}) or {}
    protocol = str(cfg.get("protocol") or "").strip().lower()

    if protocol == "subsonic":
        await _uninstall_subsonic(request, manifest, sd, user_id)


# ── Subsonic (music server) ────────────────────────────────────────────


async def _install_subsonic(
    request: Any, manifest: Any, sd: Any, user_id: str,
) -> None:
    """Wire a Subsonic-speaking music server into Augmentum."""
    # 1) Ping the server to verify it's alive and auth works.
    base_url = f"http://augmentum-{manifest.service_id}:{sd.internal_port}"
    from augmentum.media.subsonic_client import SubsonicClient
    from augmentum.providers.service_auth import managed_service_credentials

    username, password = managed_service_credentials(manifest.service_id)
    client = SubsonicClient(base_url, username=username, password=password)
    if not await client.ping():
        log.warning(
            "provider_bridge_subsonic_unreachable",
            service_id=manifest.service_id, base_url=base_url,
        )
        # Non-fatal — the container is running, maybe just slow to boot.
        # The per-user connection row still gets created; the first sync
        # will retry.

    # 2) Create the per-user connection row + enqueue catalog sync.
    #    Reuses the exact same path as kind=media_server installs so
    #    the server immediately shows up in Files with live status.
    from augmentum.marketplace.install_dispatchers import _connect_media_server
    provider = manifest.service_id
    try:
        await _connect_media_server(
            request, sd=sd, service_id=manifest.service_id,
            provider=provider, user_id=user_id,
        )
    except Exception:
        log.warning(
            "provider_bridge_connect_failed",
            service_id=manifest.service_id, hook="provider_bridge",
            exc_info=True,
        )

    log.info(
        "provider_bridge_subsonic_connected",
        service_id=manifest.service_id, base_url=base_url,
    )


async def _uninstall_subsonic(
    request: Any, manifest: Any, sd: Any, user_id: str,
) -> None:
    """Remove the per-user Subsonic connection."""
    from augmentum.marketplace.install_dispatchers import _media_server_store
    from augmentum.media.store import purge_server_data

    store = _media_server_store(request)
    idx = getattr(request.app.state, "file_index", None)
    if store is None:
        return

    provider = manifest.service_id
    try:
        rows = await store.list_visible(user_id=user_id)
    except Exception:
        log.warning("provider_bridge_uninstall_list_failed", exc_info=True)
        return

    removed = 0
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
                "provider_bridge_uninstall_row_failed",
                server_id=s.id, exc_info=True,
            )

    if removed:
        log.info(
            "provider_bridge_subsonic_uninstalled",
            service_id=manifest.service_id, connections_removed=removed,
        )


KNOWN_INTEGRATION_HOOKS["provider_bridge"] = (
    _install,
    _uninstall,
    HookMeta(
        label="Music & Media",
        icon="🔌",
        companion_hint="Companion can play your music",
        status_provider="subsonic",
    ),
)
