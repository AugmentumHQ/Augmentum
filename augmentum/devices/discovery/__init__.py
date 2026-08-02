"""Discovery coordinator — parallel sweep across drivers, dedup, merge."""

from __future__ import annotations

from augmentum.devices.discovery.coordinator import (
    DiscoveryResult,
    merge_discovered_with_saved,
    run_discovery_sweep,
)

__all__ = [
    "DiscoveryResult",
    "merge_discovered_with_saved",
    "run_discovery_sweep",
]
