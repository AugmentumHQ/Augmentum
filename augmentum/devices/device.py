"""Device dataclasses.

A `Device` is a registered or saved instance bound to a driver. A
`DiscoveredDevice` is a transient discovery-result that hasn't been
persisted yet — the registry decides whether to merge it with an existing
saved device or surface it as a new one.

Auth dictionaries on `Device` are plaintext in-memory; persistence layer
encrypts at rest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def make_device_id(*, driver: str, native_id: str, user_id: str) -> str:
    """Deterministic device ID. Same triple → same ID across restarts."""
    seed = f"{driver}|{native_id}|{user_id}".encode("utf-8")
    digest = hashlib.sha1(seed, usedforsecurity=False).hexdigest()  # noqa: S324
    return f"dev_{digest[:12]}"


@dataclass(slots=True)
class Device:
    """Persisted or live-cached device instance."""

    id: str
    user_id: str
    driver: str
    native_id: str
    label: str
    capabilities: list[str] = field(default_factory=list)
    address: dict[str, Any] = field(default_factory=dict)
    auth: dict[str, Any] = field(default_factory=dict)
    status: str = "unverified"
    last_seen_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    bindings: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def supports(self, capability_id: str) -> bool:
        return capability_id in self.capabilities or any(
            binding.get("capabilities") and capability_id in binding["capabilities"]
            for binding in self.bindings
        )

    def driver_for(self, capability_id: str) -> str:
        """Which driver should handle this capability?

        Prefers the primary driver if it supports it; otherwise falls back
        to the first binding that declares support.
        """
        if capability_id in self.capabilities:
            return self.driver
        for binding in self.bindings:
            caps = binding.get("capabilities") or []
            if capability_id in caps:
                return str(binding.get("driver") or "")
        return self.driver

    def to_dict(self, *, include_auth: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "user_id": self.user_id,
            "driver": self.driver,
            "native_id": self.native_id,
            "label": self.label,
            "capabilities": list(self.capabilities),
            "address": dict(self.address or {}),
            "status": self.status,
            "last_seen_at": self.last_seen_at,
            "metadata": dict(self.metadata or {}),
            "config": dict(self.config or {}),
            "bindings": [dict(b) for b in (self.bindings or [])],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_auth:
            out["auth"] = dict(self.auth or {})
        return out


@dataclass(slots=True)
class DiscoveredDevice:
    """Transient discovery result. Not yet persisted."""

    driver: str
    native_id: str
    label: str
    capabilities: list[str] = field(default_factory=list)
    address: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "driver": self.driver,
            "native_id": self.native_id,
            "label": self.label,
            "capabilities": list(self.capabilities),
            "address": dict(self.address or {}),
            "metadata": dict(self.metadata or {}),
            "confidence": float(self.confidence or 0.0),
        }
