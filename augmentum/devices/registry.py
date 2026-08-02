"""DeviceRegistry — single dispatch point for the substrate.

Holds driver instances, persists devices, mediates discovery, runs
invocations. Every method is user-scoped; drivers cannot bypass this —
they receive a Device handed to them by the registry, never one they
read for themselves.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from augmentum.devices.capabilities import (
    get_capability,
    has_capability,
    resolve_action,
)
from augmentum.devices.device import Device, DiscoveredDevice, make_device_id
from augmentum.devices.discovery.coordinator import (
    DiscoveryResult,
    merge_discovered_with_saved,
    run_discovery_sweep,
)
from augmentum.devices.discovery.subnet_sweep import (
    DEFAULT_FALLBACK_SUBNETS,
    sweep_multiple_subnets,
    sweep_subnet,
)
from augmentum.devices.events import EventBus
from augmentum.devices.invocation import (
    Event,
    InvocationContext,
    InvocationError,
    InvocationResult,
    PairResult,
)
from augmentum.devices.sessions import CapabilitySession, SessionRuntime
from augmentum.devices.store import (
    DevicePairingStore,
    DevicePlayHistoryStore,
    DeviceStore,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

    from augmentum.devices.cast_tokens import CastTokenStore
    from augmentum.devices.driver import DeviceDriver, DriverContext

log = get_logger(__name__)


# Capability IDs whose successful actions get logged to play_history.
# Drives the smart-match heuristic for voice/LLM commands like
# "play the lofi music on the living room TV."
_HISTORY_LOGGED_CAPABILITIES: frozenset[str] = frozenset({
    "media.audio_play@1",
    "media.video_play@1",
    "media.queue@1",
    "display.image_show@1",
    "display.web_show@1",
})


_HISTORY_LOGGED_ACTIONS: frozenset[str] = frozenset({"play", "show", "load_url", "add"})


def _content_kind_for_capability(capability_id: str) -> str:
    if capability_id == "media.audio_play@1":
        return "audio"
    if capability_id == "media.video_play@1":
        return "video"
    if capability_id == "media.queue@1":
        return "queue_item"
    if capability_id == "display.image_show@1":
        return "image"
    if capability_id == "display.web_show@1":
        return "web"
    return ""


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _expand_capabilities(caps: list[str]) -> list[str]:
    """Walk extends chains so a device declaring video_play also declares audio_play."""
    expanded: list[str] = []
    seen: set[str] = set()
    queue = list(caps or [])
    while queue:
        cap_id = queue.pop(0)
        if cap_id in seen:
            continue
        seen.add(cap_id)
        cap = get_capability(cap_id)
        if cap is None:
            # Unknown capability is preserved as-is — drivers may declare
            # forward-compat capability IDs not yet in the catalog.
            expanded.append(cap_id)
            continue
        expanded.append(cap_id)
        if cap.extends:
            queue.append(cap.extends)
    return expanded


class DeviceRegistry:
    """Single dispatch point. Lifecycle: register drivers, then `start()`."""

    def __init__(
        self,
        *,
        device_store: DeviceStore,
        pairing_store: DevicePairingStore,
        history_store: DevicePlayHistoryStore,
        sessions: SessionRuntime,
        bus: EventBus,
        http_client: httpx.AsyncClient | None = None,
        cast_token_store: CastTokenStore | None = None,
    ) -> None:
        self._device_store = device_store
        self._pairing_store = pairing_store
        self._history_store = history_store
        self._sessions = sessions
        self._bus = bus
        self._http = http_client
        self._cast_token_store = cast_token_store
        self._drivers: dict[str, DeviceDriver] = {}
        self._driver_ctx: DriverContext | None = None
        self._started = False

    @property
    def cast_token_store(self) -> CastTokenStore | None:
        return self._cast_token_store

    # ---- driver registration -------------------------------------------------

    def register_driver(self, driver: DeviceDriver) -> None:
        if driver.id in self._drivers:
            log.warning("driver_already_registered", driver=driver.id)
            return
        self._drivers[driver.id] = driver

    def list_drivers(self) -> list[DeviceDriver]:
        return list(self._drivers.values())

    def get_driver(self, driver_id: str) -> DeviceDriver | None:
        return self._drivers.get(driver_id)

    # ---- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        from augmentum.devices.driver import DriverContext  # avoid cycle

        async def _settings_get(key: str) -> Any:  # placeholder; wire later
            return None

        self._driver_ctx = DriverContext(
            http_client=self._http,
            event_bus=self._bus,
            settings_get=_settings_get,
        )
        for driver in self._drivers.values():
            try:
                await driver.start(self._driver_ctx)
            except Exception as exc:
                log.warning("driver_start_failed", driver=driver.id, error=str(exc))
        self._started = True
        log.info("device_registry_started", drivers=list(self._drivers))

    async def stop(self) -> None:
        for driver in self._drivers.values():
            try:
                await driver.stop()
            except Exception as exc:
                log.warning("driver_stop_failed", driver=driver.id, error=str(exc))
        self._bus.close_all()
        self._started = False

    # ---- persistence facade --------------------------------------------------

    async def list(self, *, user_id: str, only_online: bool = False) -> list[Device]:
        devices = await self._device_store.list_for_user(user_id=user_id)
        if only_online:
            devices = [d for d in devices if d.status == "online"]
        return devices

    async def get(self, device_id: str, *, user_id: str) -> Device | None:
        return await self._device_store.get(device_id, user_id=user_id)

    async def save(
        self,
        *,
        user_id: str,
        driver: str,
        native_id: str,
        label: str,
        capabilities: list[str],
        address: dict[str, Any] | None = None,
        auth: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        bindings: list[dict[str, Any]] | None = None,
        status: str = "unverified",
    ) -> Device:
        if not user_id:
            raise ValueError("device save requires user_id")
        if driver not in self._drivers:
            raise ValueError(f"unknown driver: {driver}")

        device_id = make_device_id(driver=driver, native_id=native_id, user_id=user_id)
        device = Device(
            id=device_id,
            user_id=user_id,
            driver=driver,
            native_id=native_id,
            label=label,
            capabilities=_expand_capabilities(capabilities),
            address=address or {},
            auth=auth or {},
            status=status,
            last_seen_at=_now_iso() if status == "online" else "",
            metadata=metadata or {},
            config=config or {},
            bindings=bindings or [],
        )
        return await self._device_store.upsert(device, user_id=user_id)

    async def update(
        self,
        device_id: str,
        *,
        user_id: str,
        label: str | None = None,
        config: dict[str, Any] | None = None,
        capabilities: list[str] | None = None,
    ) -> Device | None:
        return await self._device_store.update_fields(
            device_id,
            user_id=user_id,
            label=label,
            config=config,
            capabilities=_expand_capabilities(capabilities) if capabilities is not None else None,
        )

    async def delete(self, device_id: str, *, user_id: str) -> bool:
        # Tear down any active sessions on this device before removal.
        device = await self._device_store.get(device_id, user_id=user_id)
        if device is None:
            return False
        self._sessions.remove_all_for_device(user_id=user_id, device_id=device_id)
        return await self._device_store.delete(device_id, user_id=user_id)

    # ---- discovery -----------------------------------------------------------

    async def discover(
        self,
        *,
        user_id: str,
        drivers: list[str] | None = None,
        timeout_s: float = 3.0,
    ) -> DiscoveryResult:
        start = time.monotonic()
        discovered, errors = await run_discovery_sweep(
            list(self._drivers.values()),
            timeout_s=timeout_s,
            only_drivers=drivers,
            user_id=user_id,
        )

        saved = await self._device_store.list_for_user(user_id=user_id)
        truly_new, online_ids, heal_map = merge_discovered_with_saved(discovered, saved)

        saved_by_id: dict[str, Any] = {s.id: s for s in saved}

        # Mark online devices as online + heal stale rows in place. The
        # heal pass fixes rows that were saved when the device wasn't
        # fully discoverable (e.g. manual-add with empty capabilities,
        # provider-bridged devices that didn't have a live session yet).
        now = _now_iso()
        for sid in online_ids:
            update_kwargs: dict[str, Any] = {
                "status": "online",
                "last_seen_at": now,
            }
            existing = saved_by_id.get(sid)
            fresh = heal_map.get(sid)
            if existing is not None and fresh is not None:
                if not (existing.capabilities or []) and fresh.capabilities:
                    update_kwargs["capabilities"] = _expand_capabilities(list(fresh.capabilities))
                # Manual-add rows lack server_id / session_id; replace
                # the address only when the freshly discovered payload
                # has materially more information.
                existing_addr = existing.address or {}
                fresh_addr = fresh.address or {}
                if fresh_addr and (
                    "session_id" in fresh_addr and "session_id" not in existing_addr
                    or "server_id" in fresh_addr and "server_id" not in existing_addr
                    or str(existing.native_id).startswith("manual:")
                ):
                    update_kwargs["address"] = dict(fresh_addr)
                    if fresh.metadata:
                        update_kwargs["metadata"] = dict(fresh.metadata)
            try:
                await self._device_store.update_fields(
                    sid,
                    user_id=user_id,
                    **update_kwargs,
                )
            except Exception as exc:
                log.warning("device_status_update_failed", id=sid, error=str(exc))

        return DiscoveryResult(
            discovered=truly_new,
            online_saved_ids=online_ids,
            offline_saved_ids=[],
            errors=errors,
            duration_s=time.monotonic() - start,
        )

    async def probe(
        self,
        *,
        user_id: str,
        driver: str,
        host: str,
        port: int | None = None,
        hint: dict[str, Any] | None = None,
    ) -> DiscoveredDevice | None:
        d = self._drivers.get(driver)
        if d is None:
            raise ValueError(f"unknown driver: {driver}")
        return await d.probe(host=host, port=port, hint=hint)

    async def sweep(
        self,
        *,
        user_id: str,
        subnet: str | None = None,
        timeout_s: float = 8.0,
    ) -> tuple[list[DiscoveredDevice], dict[str, str], float]:
        """TCP-based subnet sweep — discovers devices when multicast fails.

        Used by the `/api/devices/sweep` route as a Docker-Desktop-friendly
        alternative to SSDP. The user's request source IP gives a strong
        subnet hint; if absent, falls back to common consumer-router
        defaults.
        """
        drivers = list(self._drivers.values())
        if subnet:
            return await sweep_subnet(
                drivers=drivers,
                subnet=subnet,
                timeout_s=timeout_s,
            )
        return await sweep_multiple_subnets(
            drivers=drivers,
            subnets=DEFAULT_FALLBACK_SUBNETS,
            timeout_s_per_subnet=max(2.0, timeout_s / len(DEFAULT_FALLBACK_SUBNETS)),
        )

    async def sweep_candidates(
        self,
        *,
        user_id: str,
        candidates: list[dict[str, Any]],
        timeout_s: float = 6.0,
    ) -> list[DiscoveredDevice]:
        """Validate browser-supplied (host, port) candidates.

        The browser is the only thing guaranteed to be on the user's LAN
        (the augmentum container may be behind Docker NAT, on a VPS, etc).
        Browser-side probing finds 'something speaks HTTP at this address';
        we then validate each candidate by fetching the actual UPnP
        description from the server side and parsing it into a real
        DiscoveredDevice.
        """
        if not candidates:
            return []

        drivers = list(self._drivers.values())
        if not drivers:
            return []

        sem = asyncio.Semaphore(20)
        seen: set[tuple[str, str]] = set()
        results: list[DiscoveredDevice] = []
        results_lock = asyncio.Lock()

        async def _probe_one(driver, host: str, port: int | None) -> None:
            async with sem:
                try:
                    discovered = await driver.probe(host=host, port=port)
                except Exception as exc:
                    log.debug(
                        "sweep_candidate_probe_failed",
                        driver=driver.id, host=host, port=port, error=str(exc),
                    )
                    return
                if discovered is None:
                    return
                async with results_lock:
                    key = (discovered.driver, discovered.native_id)
                    if key in seen:
                        return
                    seen.add(key)
                    results.append(discovered)

        tasks = []
        for cand in candidates:
            host = str(cand.get("host") or cand.get("ip") or "").strip()
            port_raw = cand.get("port")
            try:
                port = int(port_raw) if port_raw is not None else None
            except (TypeError, ValueError):
                port = None
            if not host:
                continue
            for driver in drivers:
                if "manual_only" in (driver.discovery_modes or ()):
                    continue
                tasks.append(asyncio.create_task(_probe_one(driver, host, port)))

        if not tasks:
            return []

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=max(2.0, float(timeout_s)),
            )
        except TimeoutError:
            for t in tasks:
                if not t.done():
                    t.cancel()

        return results

    # ---- invocation ----------------------------------------------------------

    async def invoke(
        self,
        *,
        user_id: str,
        device_id: str,
        capability: str,
        action: str,
        args: dict[str, Any] | None = None,
    ) -> InvocationResult:
        device = await self._device_store.get(device_id, user_id=user_id)
        if device is None:
            return InvocationResult.failure(
                "device_not_found",
                code="device_not_found",
            )

        if not has_capability(capability):
            return InvocationResult.failure(
                f"unknown capability: {capability}",
                code="unknown_capability",
            )

        if not device.supports(capability):
            return InvocationResult.failure(
                f"device does not support {capability}",
                code="capability_not_supported",
                extra={"supported": list(device.capabilities)},
            )

        resolved = resolve_action(capability, action)
        if resolved is None:
            return InvocationResult.failure(
                f"action {action} not defined on {capability}",
                code="unknown_action",
            )

        driver_id = device.driver_for(capability)
        driver = self._drivers.get(driver_id)
        if driver is None:
            return InvocationResult.failure(
                f"driver not registered: {driver_id}",
                code="driver_unavailable",
            )

        ctx = InvocationContext(
            user_id=user_id,
            http_client=self._http,
        )

        try:
            result = await driver.invoke(device, capability, action, dict(args or {}), ctx)
        except InvocationError as exc:
            return InvocationResult.failure(
                str(exc),
                code=exc.code,
                retryable=exc.retryable,
                extra=dict(exc.details),
            )
        except Exception as exc:
            log.warning(
                "driver_invoke_unhandled",
                driver=driver_id,
                capability=capability,
                action=action,
                error=str(exc),
            )
            return InvocationResult.failure(
                f"driver_error: {exc}",
                code="driver_error",
            )

        if not isinstance(result, InvocationResult):
            return InvocationResult.failure(
                "driver returned non-InvocationResult",
                code="driver_protocol_violation",
            )

        # Stateful action success → manage a session
        cap_meta, action_meta = resolved
        if result.ok and action_meta.is_stateful:
            session = self._sessions.list_for_device(user_id=user_id, device_id=device_id)
            existing = next(
                (s for s in session if s.capability_id == capability), None,
            )
            if action == "stop" and existing is not None:
                # Revoke any cast tokens issued for this session — once the
                # cast is over, leaked tokens are dead weight regardless of
                # remaining TTL.
                if self._cast_token_store is not None:
                    self._cast_token_store.revoke_session(existing.id)
                self._sessions.remove(existing.id, user_id=user_id)
            elif existing is None and action in ("play", "show", "load_url"):
                # Carry display metadata through to the session so the
                # remote-control UI (cast-remote pill) can show cover
                # art + author next to the title.
                args_dict = args or {}
                extra: dict[str, Any] = {
                    "device_label": device.label,
                }
                if args_dict.get("poster_url"):
                    extra["poster_url"] = str(args_dict["poster_url"])
                if args_dict.get("author"):
                    extra["author"] = str(args_dict["author"])
                if args_dict.get("artist"):
                    extra["artist"] = str(args_dict["artist"])
                if args_dict.get("album"):
                    extra["album"] = str(args_dict["album"])
                new_session = CapabilitySession(
                    user_id=user_id,
                    device_id=device_id,
                    driver=driver_id,
                    capability_id=capability,
                    title=str(args_dict.get("title") or ""),
                    state=dict(result.state),
                    thumbnail=str(args_dict.get("poster_url") or ""),
                    extra=extra,
                )
                self._sessions.put(new_session)
                if not result.session_id:
                    result.session_id = new_session.id
            elif existing is not None:
                existing.update_state(result.state)
                if not result.session_id:
                    result.session_id = existing.id

        # Smart-match history logging
        if (
            result.ok
            and capability in _HISTORY_LOGGED_CAPABILITIES
            and action in _HISTORY_LOGGED_ACTIONS
        ):
            try:
                await self._history_store.log(
                    user_id=user_id,
                    device_id=device_id,
                    capability_id=capability,
                    action=action,
                    file_id=str((args or {}).get("file_id") or ""),
                    content_key=str((args or {}).get("content_key") or (args or {}).get("content_url") or ""),
                    content_label=str((args or {}).get("title") or ""),
                    content_kind=_content_kind_for_capability(capability),
                    success=True,
                )
            except Exception as exc:
                log.warning("play_history_log_failed", error=str(exc))

        return result

    async def snapshot(
        self,
        *,
        user_id: str,
        device_id: str,
        capability: str,
    ) -> dict[str, Any] | None:
        device = await self._device_store.get(device_id, user_id=user_id)
        if device is None or not device.supports(capability):
            return None
        driver = self._drivers.get(device.driver_for(capability))
        if driver is None:
            return None
        ctx = InvocationContext(user_id=user_id, http_client=self._http)
        try:
            return await driver.snapshot(device, capability, ctx)
        except Exception as exc:
            log.debug(
                "driver_snapshot_failed",
                driver=driver.id,
                capability=capability,
                error=str(exc),
            )
            return None

    async def subscribe(
        self,
        *,
        user_id: str,
        device_id: str | None = None,
        capability: str | None = None,
    ) -> AsyncIterator[Event]:
        return self._bus.subscribe(
            user_id=user_id,
            device_id=device_id,
            capability_id=capability,
        )

    # ---- sessions ------------------------------------------------------------

    async def list_sessions(self, *, user_id: str) -> list[CapabilitySession]:
        return self._sessions.list(user_id=user_id)

    async def end_session(self, *, user_id: str, session_id: str) -> bool:
        session = self._sessions.get(session_id, user_id=user_id)
        if session is None:
            return False
        # Best-effort send a `stop` invocation to the driver.
        try:
            await self.invoke(
                user_id=user_id,
                device_id=session.device_id,
                capability=session.capability_id,
                action="stop",
                args={},
            )
        except Exception as exc:
            log.debug("session_stop_invoke_failed", error=str(exc))
        return self._sessions.remove(session_id, user_id=user_id)

    # ---- pairing -------------------------------------------------------------

    async def pair_start(self, *, user_id: str, device_id: str) -> PairResult:
        device = await self._device_store.get(device_id, user_id=user_id)
        if device is None:
            return PairResult(
                state="failed",
                message="device_not_found",
            )
        driver = self._drivers.get(device.driver)
        if driver is None:
            return PairResult(
                state="failed",
                message="driver_unavailable",
            )
        ctx = InvocationContext(user_id=user_id, http_client=self._http)
        return await driver.pair_start(device, ctx)

    async def pair_complete(
        self,
        *,
        user_id: str,
        device_id: str,
        code: str,
    ) -> PairResult:
        device = await self._device_store.get(device_id, user_id=user_id)
        if device is None:
            return PairResult(state="failed", message="device_not_found")
        driver = self._drivers.get(device.driver)
        if driver is None:
            return PairResult(state="failed", message="driver_unavailable")
        ctx = InvocationContext(user_id=user_id, http_client=self._http)
        return await driver.pair_complete(device, code, ctx)
