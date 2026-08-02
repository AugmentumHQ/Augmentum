"""Fabric lifespan hook: one-line wiring point for server.py.

server.py has heavy churn from many parallel branches. To keep
fabric integration light-touch, the actual ``app.state`` setup lives
here -- server.py just calls ``await start_fabric_if_enabled(app)``
during lifespan startup and ``await stop_fabric(app)`` during
shutdown. Two lines edited there, full lifecycle owned by this
module.

Every helper here is a no-op when ``settings.fabric_enabled`` is
False, so callers don't need to gate at the call site.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

log = get_logger(__name__)


async def start_fabric_if_enabled(app: FastAPI) -> None:
    """Initialise fabric subsystems on the app, gated on the flag.

    Idempotent: calling twice is safe (second call returns immediately
    because ``app.state.fabric_coordinator`` will already exist).

    Required preconditions on app.state at call time:
      - ``settings_store`` -- for identity load
      - ``state_manager.backend.conn`` -- for fabric_nodes access
    These exist by the time the route layer is wired up, so call
    this AFTER ``_restore_settings`` and AFTER the state backend is
    open.
    """
    # Default-off guard. Even with the import landed, no fabric code
    # runs until an operator explicitly flips the setting on.
    if not settings.fabric_enabled:
        log.info("fabric_disabled_skipping_startup")
        return

    if getattr(app.state, "fabric_coordinator", None) is not None:
        # Already started; calling again is a no-op.
        return

    # Imports are local to avoid paying startup cost (cryptography
    # initialisation) when fabric is disabled. Solo installs never
    # touch this module.
    from augmentum.fabric.client import FabricClient
    from augmentum.fabric.coordinator import FabricCoordinator
    from augmentum.fabric.identity import (
        FabricIdentity,
        FabricIdentityCorruptError,
    )

    settings_store = getattr(app.state, "settings_store", None)
    if settings_store is None:
        log.warning("fabric_startup_skipped_no_settings_store")
        return

    sm = getattr(app.state, "state_manager", None)
    db_conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    if db_conn is None:
        log.warning("fabric_startup_skipped_no_state_backend")
        return

    try:
        identity = await FabricIdentity.from_settings_store(settings_store)
    except FabricIdentityCorruptError:
        # Fail-closed identity refused to silently regenerate. Degrade
        # fabric (the rest of the app boots normally); the operator must
        # restore the key from their BIP39 backup or a data_dir snapshot.
        # Surfaced on app.state so a health endpoint can report it.
        log.error("fabric_startup_aborted_identity_corrupt", exc_info=True)
        app.state.fabric_identity_corrupt = True
        return
    coordinator = FabricCoordinator(identity, db_conn)
    await coordinator.initialise()
    _register_capability_extractors(app, coordinator)
    # Tier 0: start the heartbeat sweeper. Detects + detaches peers
    # whose WS socket is open but whose process has stopped sending
    # heartbeats (hung process, OOM-soft, frozen GC). Pre-fix the
    # coordinator would stay convinced a hung peer was live and the
    # router kept dispatching to it. Shutdown stops it.
    coordinator.start_heartbeat_sweeper()

    # Dedicated fabric httpx client with verify=False. Same trust model
    # as pair_client.py / discovery.py / FabricClient WS: peer identity is
    # the pinned Ed25519 fingerprint + signed envelope, NOT the TLS chain.
    # The system-wide app.state.http_client validates certs (correct for
    # OpenAI / HuggingFace / etc.) — using it for peer-to-peer HTTPS would
    # fail on every self-signed Caddy LAN cert. Closed in stop_fabric().
    import httpx as _httpx
    fabric_http_client = _httpx.AsyncClient(verify=False, follow_redirects=False)
    app.state.fabric_http_client = fabric_http_client

    client = FabricClient(identity, coordinator, fabric_http_client)
    client_task = asyncio.create_task(client.run(), name="fabric_client_supervisor")

    # Phase 3: routing director consumes the coordinator's capability
    # registry + fabric_http_client for cross-peer dispatch.
    from augmentum.fabric.director import RoutingDirector
    director = RoutingDirector(coordinator, fabric_http_client)

    # Attach the director to the provider_registry so
    # ``resolve_backend_with_fabric`` picks it up automatically. This
    # is what makes every LLM dispatch site fabric-aware without each
    # caller threading the director manually.
    provider_registry = getattr(app.state, "provider_registry", None)
    if provider_registry is not None and hasattr(provider_registry, "set_fabric_director"):
        provider_registry.set_fabric_director(director)

    app.state.fabric_identity = identity
    app.state.fabric_coordinator = coordinator
    app.state.fabric_client = client
    app.state.fabric_client_task = client_task
    app.state.fabric_director = director

    # Wedge B: if the Connect hubs already exist (test fixture or
    # eager init), wire them so inbound MSG_CONNECT_ENVELOPE frames
    # can route. Otherwise the lazy-create paths in connect_routes
    # wire on first attach.
    existing_connect_hub = getattr(app.state, "connect_hub", None)
    if existing_connect_hub is not None:
        coordinator.connect_hub = existing_connect_hub
    existing_notification_hub = getattr(app.state, "notification_hub", None)
    if existing_notification_hub is not None:
        coordinator.notification_hub = existing_notification_hub

    # audio_routes consults the coordinator via a module-level handle
    # so resolve_voice_provider / _refresh_voice_provider_map don't
    # have to thread it through every call site. Setter is idempotent.
    try:
        from augmentum.proxy.audio_routes import register_fabric_coordinator
        register_fabric_coordinator(coordinator)
    except Exception:
        log.warning("fabric_audio_coordinator_register_failed", exc_info=True)

    log.info(
        "fabric_started",
        node_id=identity.node_id,
        fingerprint=identity.fingerprint,
        peer_count=coordinator.peer_count(),
    )


async def stop_fabric(app: FastAPI) -> None:
    """Tear down fabric subsystems on lifespan shutdown.

    No-op when fabric was never started (default-off path or a
    startup that bailed before installing the coordinator).
    """
    client = getattr(app.state, "fabric_client", None)
    task = getattr(app.state, "fabric_client_task", None)
    coordinator = getattr(app.state, "fabric_coordinator", None)

    if client is not None:
        try:
            await client.stop()
        except Exception:
            log.debug("fabric_client_stop_failed", exc_info=True)

    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    if coordinator is not None:
        try:
            await coordinator.shutdown()
        except Exception:
            log.debug("fabric_coordinator_shutdown_failed", exc_info=True)

    fabric_http_client = getattr(app.state, "fabric_http_client", None)
    if fabric_http_client is not None:
        try:
            await fabric_http_client.aclose()
        except Exception:
            log.debug("fabric_http_client_close_failed", exc_info=True)
        app.state.fabric_http_client = None

    # Drop the director reference on the provider_registry so a
    # subsequent solo-mode startup doesn't reuse a stale one.
    provider_registry = getattr(app.state, "provider_registry", None)
    if provider_registry is not None and hasattr(provider_registry, "set_fabric_director"):
        provider_registry.set_fabric_director(None)

    # Clear the audio module's coordinator handle so a fresh start
    # doesn't read a stale reference. Solo-install path never reaches
    # here (start_fabric_if_enabled returned early), so the import is
    # safe to attempt unconditionally.
    try:
        from augmentum.proxy.audio_routes import register_fabric_coordinator
        register_fabric_coordinator(None)
    except Exception:
        log.debug("fabric_audio_coordinator_unregister_failed", exc_info=True)

    log.info("fabric_stopped")


def _register_capability_extractors(app, coordinator) -> None:
    """Attach all capability extractors to the coordinator.

    Reads the existing in-process registries / managers from app.state
    and constructs one extractor per kind, passing in the source dep.
    Sources that aren't present on this install simply don't get an
    extractor -- the coordinator handles that gracefully.

    Imports are local so a solo install (fabric disabled) never pays
    the import cost.
    """
    from augmentum.fabric.extractors import (
        AudioCapabilityExtractor,
        CastRenderCapabilityExtractor,
        ImageCapabilityExtractor,
        KnowledgeSearchExtractor,
        LLMCapabilityExtractor,
    )

    llama_manager = getattr(app.state, "llama_manager", None)
    provider_registry = getattr(app.state, "provider_registry", None)
    image_pipeline_registry = getattr(app.state, "image_pipeline_registry", None)
    image_persistence = getattr(app.state, "image_persistence", None)
    image_model_manager = getattr(app.state, "image_model_manager", None)
    pack_manager = getattr(app.state, "pack_manager", None)

    sm = getattr(app.state, "state_manager", None)
    # Heartbeat extractors run a SELECT every 5s. Prefer the dedicated
    # read connection (configured in proxy/server.py) so they don't
    # queue behind writers on the main aiosqlite worker thread —
    # ``audio_providers`` SELECT was logging 3.7s contended in the
    # 2026-05-26 trace. Fall back to the main conn if read_conn isn't
    # configured (early lifespan, tests).
    audio_db_conn = getattr(app.state, "read_conn", None)
    if audio_db_conn is None:
        audio_db_conn = (
            getattr(getattr(sm, "backend", None), "conn", None) if sm else None
        )

    coordinator.register_extractor(
        LLMCapabilityExtractor(
            provider_registry=provider_registry,
            llama_manager=llama_manager,
        )
    )
    coordinator.register_extractor(
        ImageCapabilityExtractor(
            persistence=image_persistence,
            pipeline_registry=image_pipeline_registry,
            model_manager=image_model_manager,
        )
    )
    coordinator.register_extractor(
        KnowledgeSearchExtractor(pack_manager=pack_manager)
    )
    coordinator.register_extractor(
        AudioCapabilityExtractor(db_conn=audio_db_conn)
    )
    coordinator.register_extractor(
        CastRenderCapabilityExtractor()
    )
