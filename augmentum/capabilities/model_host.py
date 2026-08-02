"""Capability adapter for model/provider runtime state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from augmentum.capabilities.frontdesk import CapabilityContext


def _mapping(value: Any) -> dict[Any, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    return []


class ModelHost:
    id = "models"
    title = "ModelHost"
    order = 10
    endpoint_prefixes = ["/api/models", "/api/providers", "/v1/models", "/api/tags"]

    def describe(self, context: CapabilityContext) -> dict[str, Any]:
        registry = getattr(context.app_state, "provider_registry", None)
        backends = self._backends(registry)
        model_map = self._model_map(registry)
        discovered = self._discovered(registry)

        return {
            "id": self.id,
            "title": self.title,
            "available": registry is not None,
            "status": "ready" if registry is not None else "unavailable",
            "endpoint_prefixes": list(self.endpoint_prefixes),
            "default_backend": self._default_backend(registry),
            "available_backends": sorted(str(name) for name in backends),
            "backend_count": len(backends),
            "model_map_count": len(model_map),
            "discovered_count": len(discovered),
            "items": self._backend_items(backends),
        }

    def _backends(self, registry: Any) -> dict[Any, Any]:
        if registry is None:
            return {}
        return _mapping(getattr(registry, "backends", {}))

    def _model_map(self, registry: Any) -> dict[Any, Any]:
        if registry is None:
            return {}
        return _mapping(getattr(registry, "_model_map", {}))

    def _discovered(self, registry: Any) -> list[Any]:
        if registry is None:
            return []
        return _sequence(getattr(registry, "_discovered", []))

    def _default_backend(self, registry: Any) -> str:
        if registry is None:
            return ""
        configured = getattr(registry, "_default", "") or getattr(
            registry,
            "default_backend_name",
            "",
        )
        return str(configured or "")

    def _backend_items(self, backends: dict[Any, Any]) -> list[dict[str, Any]]:
        return [
            {
                "id": str(name),
                "kind": "backend",
                "class": backend.__class__.__name__,
            }
            for name, backend in sorted(backends.items(), key=lambda item: str(item[0]))
        ]
