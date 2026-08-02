"""Read-oriented capability aggregation for Augmentum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

ROUTE_MAP = {
    "compatibility": [
        "/api/chat",
        "/api/generate",
        "/api/tags",
        "/v1/chat/completions",
        "/v1/models",
    ],
    "native": [
        "/api/capabilities",
        "/api/models",
        "/api/providers",
        "/api/tools",
        "/api/powers",
        "/api/resources",
        "/api/sessions",
        "/api/devices",
        "/api/surfaces",
        "/api/jobs",
    ],
}


@dataclass(frozen=True)
class CapabilityContext:
    """Request/runtime context used by capability hosts."""

    app_state: Any
    user_id: str = ""
    session_id: str = ""
    workspace_id: str = ""
    request_source: str = "api"


class CapabilityHost(Protocol):
    id: str
    title: str
    order: int
    endpoint_prefixes: list[str]

    def describe(self, context: CapabilityContext) -> dict[str, Any]: ...


def context_from_request(request: Any) -> CapabilityContext:
    user = request.scope.get("user") if hasattr(request, "scope") else None
    return CapabilityContext(
        app_state=request.app.state,
        user_id=str(getattr(user, "id", "") or ""),
        request_source="api",
    )


class CapabilityFrontdesk:
    """Aggregates capability host descriptors into a stable inventory."""

    def __init__(self, hosts: list[CapabilityHost]) -> None:
        self._hosts = sorted(hosts, key=lambda host: (host.order, host.id))

    def inventory(self, context: CapabilityContext) -> dict[str, Any]:
        return {
            "version": 1,
            "frontdesk": {
                "schema": "augmentum.capabilities.inventory",
                "schema_version": 1,
            },
            "hosts": {
                host.id: self._describe_host(host, context)
                for host in self._hosts
            },
            "routes": ROUTE_MAP,
        }

    def host(self, host_id: str, context: CapabilityContext) -> dict[str, Any] | None:
        for host in self._hosts:
            if host.id == host_id:
                return self._describe_host(host, context)
        return None

    def _describe_host(self, host: CapabilityHost, context: CapabilityContext) -> dict[str, Any]:
        try:
            descriptor = host.describe(context)
        except Exception as exc:
            log.warning(
                "capability_host_descriptor_failed",
                host_id=host.id,
                error=str(exc),
                exc_info=True,
            )
            return {
                "id": host.id,
                "title": host.title,
                "available": False,
                "status": "error",
                "endpoint_prefixes": list(getattr(host, "endpoint_prefixes", [])),
                "count": 0,
                "items": [],
                "unavailable_reason": "descriptor_failed",
            }
        descriptor.setdefault("id", host.id)
        descriptor.setdefault("title", host.title)
        descriptor.setdefault("available", True)
        descriptor.setdefault("endpoint_prefixes", list(getattr(host, "endpoint_prefixes", [])))
        descriptor.setdefault("items", [])
        return descriptor


def build_default_frontdesk() -> CapabilityFrontdesk:
    from augmentum.capabilities.model_host import ModelHost
    from augmentum.capabilities.standard_hosts import (
        DeviceHost,
        EventHost,
        JobHost,
        ModeHost,
        PowerHost,
        SessionHost,
        SurfaceHost,
    )
    from augmentum.capabilities.tool_host import ToolHost

    return CapabilityFrontdesk(
        [
            ModelHost(),
            ModeHost(),
            ToolHost(),
            SessionHost(),
            DeviceHost(),
            SurfaceHost(),
            PowerHost(),
            JobHost(),
            EventHost(),
        ],
    )
