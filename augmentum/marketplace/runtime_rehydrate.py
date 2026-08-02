"""Rebuild runtime service definitions for installed manifest services.

Marketplace ``kind: "service"`` apps register their ServiceDefinition at
install time only (``catalog.register_runtime``), in memory. Without this
module, every Augmentum restart left each installed app with a
managed_services row but NO definition — so ``restore_enabled`` failed
("Unknown service"), the container never came back, and Discover could
not show ports, status, or a Start action.

Called from server boot AFTER the marketplace catalog load and service
manager construction, BEFORE restore_enabled is scheduled. Cheap: pure
DB reads + in-memory registration, no Docker calls.
"""
from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def rehydrate_manifest_services(store: Any, mgr: Any) -> int:
    """Re-register runtime definitions for every installed manifest
    service found in the marketplace catalog. Returns the count."""
    from augmentum.marketplace.manifest import (
        ManifestError,
        parse_manifest,
        to_service_definition,
    )

    active: set[str] = await store.install_wide_active_service_definitions()
    if not active:
        return 0

    listings = await store.list_for_discover(kind="service", limit=1000)
    count = 0
    for listing in listings:
        payload = listing.install_payload or {}
        svc_id = str(((payload.get("service") or {}).get("id")) or "")
        if not svc_id or svc_id not in active:
            continue
        if mgr.get_definition(svc_id) is not None:
            continue  # shipped-catalog or already-registered id
        try:
            sd = to_service_definition(parse_manifest(payload))
        except ManifestError:
            log.warning(
                "manifest_rehydrate_invalid_manifest",
                service_id=svc_id, listing_id=listing.id,
            )
            continue
        # Reuse the front-door port allocated at install (persisted in
        # config_json) — allocating fresh would orphan the caddy snippet.
        try:
            cfg = await mgr.read_config_json(svc_id)
            https_port = int(cfg.get("https_port") or 0)
        except Exception:  # noqa: BLE001 — enrichment only
            https_port = 0
        # Fall back to the actual caddy snippet on disk — pre-persistence
        # installs have no config_json entry but their snippet file is the
        # authoritative record of which port caddy actually binds. Without
        # this, rehydration builds definitions with https_port=0 and the
        # allocator hands their real port to a new install (the n8n/
        # Navidrome collision).
        if not https_port:
            from augmentum.providers.caddy_front_door import snippet_port_for
            https_port = snippet_port_for(svc_id)
        if https_port:
            sd = _dc_replace(sd, https_port=https_port)
            # Persist the recovered port so the NEXT restart doesn't need
            # the snippet fallback again (one-shot reconcile for pre-
            # persistence installs). Best-effort — a write failure here
            # is non-fatal; the snippet fallback fires again next boot.
            try:
                await mgr.update_config_json(svc_id, {"https_port": https_port})
            except Exception:  # noqa: BLE001
                pass
        mgr.catalog.register_runtime(sd)
        count += 1
        log.info(
            "manifest_service_rehydrated",
            service_id=svc_id, https_port=https_port,
        )
    return count
