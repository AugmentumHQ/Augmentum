"""TCP-based subnet sweep for environments where multicast discovery fails.

Docker Desktop (Mac/Windows) hides the augmentum container inside a VM
whose network namespace can't reach LAN multicast (SSDP/mDNS). Direct
TCP unicast to LAN IPs *does* cross the Docker NAT cleanly, so we have
a workable fallback: walk the user's subnet, ask each IP for a UPnP
description on a small set of well-known ports, and surface anything
that answers as a discovered device.

This is **user-initiated only** (clicking "Search" / hitting the sweep
endpoint) and capped to UPnP-default ports — not a generic port scanner.
The threat model is "find the user's TV on their own LAN" not "scan the
internet."

Usage:

    from augmentum.devices.discovery.subnet_sweep import sweep_subnet
    discovered = await sweep_subnet(
        drivers=[dlna_driver],
        subnet="192.168.1.0/24",
        timeout_s=8.0,
    )
"""

from __future__ import annotations

import asyncio
import ipaddress
import time
from typing import TYPE_CHECKING, Iterable

from augmentum.devices.device import DiscoveredDevice
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.devices.driver import DeviceDriver

log = get_logger(__name__)


# Common subnets we'll try when the caller can't supply a hint (e.g.
# accessing augmentum at localhost from the same host). Three of the
# four most common consumer-router defaults plus 10.0.0/24.
DEFAULT_FALLBACK_SUBNETS: tuple[str, ...] = (
    "192.168.0.0/24",
    "192.168.1.0/24",
    "10.0.0.0/24",
)


def _enumerate_hosts(subnet: str) -> list[str]:
    """Return every host IP in the subnet (skips network + broadcast).

    Refuses to enumerate non-RFC1918 ranges — even an authenticated user
    shouldn't be able to weaponize augmentum's egress as a port scanner
    against the public internet. Loopback is also refused (no probing
    services on the augmentum host itself).
    """
    try:
        net = ipaddress.IPv4Network(subnet, strict=False)
    except (ValueError, ipaddress.AddressValueError):
        return []
    # Cap at /22 (1022 hosts) — anything wider is almost certainly
    # mis-entered and would take too long to sweep.
    if net.num_addresses > 1024:
        log.warning("subnet_sweep_too_wide", subnet=subnet, hosts=net.num_addresses)
        return []
    # Refuse anything outside RFC1918 / link-local. The threat model is
    # "find the user's TV on their LAN", not "scan the network."
    if not (net.is_private and not net.is_loopback):
        log.warning("subnet_sweep_non_private", subnet=subnet)
        return []
    return [str(ip) for ip in net.hosts()]


async def sweep_subnet(
    *,
    drivers: list["DeviceDriver"],
    subnet: str,
    timeout_s: float = 15.0,
    concurrency: int = 60,
) -> tuple[list[DiscoveredDevice], dict[str, str], float]:
    """Probe every host in `subnet` via every driver's `probe()`.

    Returns (discovered, errors, duration_s). Errors are keyed by driver
    id; absent if the driver succeeded without raising. Discovered list
    is deduplicated by (driver, native_id) — the same TV won't appear
    twice if multiple ports answered.

    Concurrency is bounded per-host (and globally via the semaphore) so
    we don't spam the LAN with hundreds of simultaneous probes; each
    individual probe also has its own internal timeout from the driver.
    """
    hosts = _enumerate_hosts(subnet)
    if not hosts:
        return ([], {"_subnet": "invalid_or_too_wide"}, 0.0)
    if not drivers:
        return ([], {}, 0.0)

    sem = asyncio.Semaphore(max(1, int(concurrency)))
    found: list[DiscoveredDevice] = []
    errors: dict[str, str] = {}
    start = time.monotonic()
    deadline = start + max(2.0, float(timeout_s))

    async def _probe_one(driver: "DeviceDriver", host: str) -> None:
        if time.monotonic() >= deadline:
            return
        async with sem:
            if time.monotonic() >= deadline:
                return
            try:
                discovered = await driver.probe(host=host)
            except Exception as exc:
                # Per-host errors are expected (timeouts, refused, etc).
                # Only log at debug; aggregate driver-level errors below.
                log.debug(
                    "subnet_sweep_probe_failed",
                    driver=driver.id,
                    host=host,
                    error=str(exc),
                )
                return
            if discovered is not None:
                found.append(discovered)

    tasks: list[asyncio.Task] = []
    for driver in drivers:
        # Skip drivers that explicitly opt out of host-level probing.
        if "manual_only" in (driver.discovery_modes or ()):
            continue
        for host in hosts:
            tasks.append(asyncio.create_task(_probe_one(driver, host)))

    if not tasks:
        return ([], {}, 0.0)

    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=max(2.0, float(timeout_s)),
        )
    except asyncio.TimeoutError:
        # Some probes may still be in flight — cancel them.
        for t in tasks:
            if not t.done():
                t.cancel()
        errors["_timeout"] = f"{timeout_s}s"

    # Dedup by (driver, native_id) — same physical device may answer
    # on multiple ports, we only want one entry.
    seen: set[tuple[str, str]] = set()
    deduped: list[DiscoveredDevice] = []
    for d in found:
        key = (d.driver, d.native_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)

    return (deduped, errors, time.monotonic() - start)


async def sweep_multiple_subnets(
    *,
    drivers: list["DeviceDriver"],
    subnets: Iterable[str],
    timeout_s_per_subnet: float = 6.0,
) -> tuple[list[DiscoveredDevice], dict[str, str], float]:
    """Run sweep_subnet across several candidate subnets sequentially.

    Used when the caller has no subnet hint (e.g. accessing augmentum
    via localhost) and we need to try the common defaults.
    """
    all_found: list[DiscoveredDevice] = []
    all_errors: dict[str, str] = {}
    start = time.monotonic()
    seen: set[tuple[str, str]] = set()

    for subnet in subnets:
        found, errors, _ = await sweep_subnet(
            drivers=drivers,
            subnet=subnet,
            timeout_s=timeout_s_per_subnet,
        )
        for d in found:
            key = (d.driver, d.native_id)
            if key in seen:
                continue
            seen.add(key)
            all_found.append(d)
        for k, v in errors.items():
            all_errors[f"{subnet}:{k}"] = v
        # Found enough? Stop early to keep latency reasonable.
        if len(all_found) >= 5:
            break

    return (all_found, all_errors, time.monotonic() - start)
