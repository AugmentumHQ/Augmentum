"""Parallel discovery sweep + cross-driver dedup.

Each driver's `discover()` runs in parallel under one shared timeout. The
combined results are deduplicated against saved devices (by exact native_id
match for the same driver, fuzzy match by metadata fingerprint across
drivers) and surfaced as a single list.

Saved devices found in this sweep get a fresh `last_seen_at`. Saved
devices NOT found accumulate a missed-pass counter; after `offline_threshold`
consecutive misses they flip to `status='offline'`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from augmentum.devices.device import DiscoveredDevice
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.devices.device import Device
    from augmentum.devices.driver import DeviceDriver

log = get_logger(__name__)


@dataclass(slots=True)
class DiscoveryResult:
    """Outcome of one sweep — fresh discoveries plus enriched saved devices."""

    discovered: list[DiscoveredDevice] = field(default_factory=list)
    online_saved_ids: list[str] = field(default_factory=list)
    offline_saved_ids: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovered": [d.to_dict() for d in self.discovered],
            "online_saved_ids": list(self.online_saved_ids),
            "offline_saved_ids": list(self.offline_saved_ids),
            "errors": dict(self.errors),
            "duration_s": float(self.duration_s),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _fingerprint(device: DiscoveredDevice | Device) -> str:
    """Cross-driver dedup fingerprint.

    Same physical device may show up via two drivers (Cast + DLNA). We
    fingerprint on (host IP, manufacturer, model_name) when available;
    label is a fallback signal but not authoritative.
    """
    address = getattr(device, "address", {}) or {}
    metadata = getattr(device, "metadata", {}) or {}
    host = str(address.get("host") or "").strip().lower()
    manufacturer = str(metadata.get("manufacturer") or "").strip().lower()
    model = str(metadata.get("model_name") or metadata.get("model") or "").strip().lower()
    return f"{host}|{manufacturer}|{model}"


async def run_discovery_sweep(
    drivers: list[DeviceDriver],
    *,
    timeout_s: float = 3.0,
    only_drivers: list[str] | None = None,
    user_id: str = "",
) -> tuple[list[DiscoveredDevice], dict[str, str]]:
    """Run all selected drivers' discover() in parallel under one timeout.

    Returns (discovered, errors). Drivers in `paired_only` or
    `manual_only` mode are skipped — discovery is meaningless for them.
    """
    selected: list[DeviceDriver] = []
    for driver in drivers:
        if only_drivers and driver.id not in only_drivers:
            continue
        modes = set(driver.discovery_modes or ())
        if "paired_only" in modes or "manual_only" in modes:
            continue
        selected.append(driver)

    if not selected:
        return ([], {})

    async def _one(driver: DeviceDriver) -> tuple[str, list[DiscoveredDevice] | Exception]:
        try:
            result = await asyncio.wait_for(
                driver.discover(timeout_s=timeout_s, user_id=user_id),
                timeout=timeout_s + 1.0,
            )
            return (driver.id, list(result or []))
        except asyncio.TimeoutError:
            return (driver.id, asyncio.TimeoutError("discovery_timeout"))
        except Exception as exc:
            return (driver.id, exc)

    outcomes = await asyncio.gather(*[_one(d) for d in selected])

    discovered: list[DiscoveredDevice] = []
    errors: dict[str, str] = {}
    for driver_id, result in outcomes:
        if isinstance(result, Exception):
            errors[driver_id] = str(result)
            log.debug("driver_discover_error", driver=driver_id, error=str(result))
            continue
        for item in result:
            if isinstance(item, DiscoveredDevice):
                if not item.driver:
                    item.driver = driver_id
                discovered.append(item)

    return (discovered, errors)


def merge_discovered_with_saved(
    discovered: list[DiscoveredDevice],
    saved: list[Device],
) -> tuple[list[DiscoveredDevice], list[str], dict[str, DiscoveredDevice]]:
    """Filter freshly-discovered devices against saved ones.

    Returns (truly_new, online_saved_ids, heal_map):

    - `truly_new` are DiscoveredDevices that don't match any saved device.
    - `online_saved_ids` are saved-device IDs that this sweep proved online.
    - `heal_map` maps saved_id → the matching DiscoveredDevice, so the
       registry can repair rows that were saved with stale or incomplete
       data (e.g. manually-added rows that later get a real session_id).

    The fingerprint dedup is intentional: if the user saved a TV via
    DLNA and the same TV shows up via Cast in this sweep, we don't want
    to surface a "new" Cast device — we update the saved DLNA device's
    `bindings` to add the Cast binding.
    """
    truly_new: list[DiscoveredDevice] = []
    online_ids: set[str] = set()
    heal_map: dict[str, DiscoveredDevice] = {}

    saved_exact: dict[tuple[str, str], Device] = {}
    saved_fingerprint: dict[str, Device] = {}
    saved_by_host: dict[tuple[str, str], Device] = {}
    for sd in saved:
        saved_exact[(sd.driver, sd.native_id)] = sd
        for binding in (sd.bindings or []):
            saved_exact[(
                str(binding.get("driver") or ""),
                str(binding.get("native_id") or ""),
            )] = sd
        fp = _fingerprint(sd)
        if fp.strip("|"):
            saved_fingerprint.setdefault(fp, sd)
        # Host-based match helps heal manual-add rows (native_id =
        # "manual:host:port", empty fingerprint) when the same host
        # later shows up in discovery with real metadata.
        host = str((sd.address or {}).get("host") or "").strip().lower()
        if host:
            saved_by_host.setdefault((sd.driver, host), sd)

    for found in discovered:
        exact = saved_exact.get((found.driver, found.native_id))
        if exact is not None:
            online_ids.add(exact.id)
            heal_map[exact.id] = found
            continue
        fp = _fingerprint(found)
        if fp.strip("|"):
            via_fp = saved_fingerprint.get(fp)
            if via_fp is not None:
                online_ids.add(via_fp.id)
                heal_map[via_fp.id] = found
                continue
        found_host = str((found.address or {}).get("host") or "").strip().lower()
        if found_host:
            via_host = saved_by_host.get((found.driver, found_host))
            if via_host is not None:
                online_ids.add(via_host.id)
                heal_map[via_host.id] = found
                continue
        truly_new.append(found)

    return (truly_new, sorted(online_ids), heal_map)
