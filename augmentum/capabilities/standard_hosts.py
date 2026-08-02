"""Default first-party capability host adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from augmentum.capabilities.frontdesk import CapabilityContext
from augmentum.classifier.router import MODE_PREFIXES, Mode


def _safe_list(callable_obj: Any) -> list[Any]:
    try:
        items = callable_obj()
    except Exception:
        return []
    return list(items or [])


class ModeHost:
    id = "modes"
    title = "ModeHost"
    order = 20
    endpoint_prefixes = ["/api/chat", "/v1/chat/completions"]

    def describe(self, _context: CapabilityContext) -> dict[str, Any]:
        prefix_by_mode = {mode.value: prefix for prefix, mode in MODE_PREFIXES.items()}
        return {
            "id": self.id,
            "title": self.title,
            "available": True,
            "status": "ready",
            "endpoint_prefixes": list(self.endpoint_prefixes),
            "items": [
                {
                    "id": mode.value,
                    "prefix": prefix_by_mode.get(mode.value, ""),
                }
                for mode in Mode
            ],
        }


class SessionHost:
    id = "sessions"
    title = "SessionHost"
    order = 40
    endpoint_prefixes = ["/api/sessions", "/api/auth"]

    def describe(self, context: CapabilityContext) -> dict[str, Any]:
        state_manager = getattr(context.app_state, "state_manager", None)
        session_manager = getattr(context.app_state, "session_manager", None)
        return {
            "id": self.id,
            "title": self.title,
            "available": state_manager is not None,
            "status": "ready" if state_manager is not None else "unavailable",
            "endpoint_prefixes": list(self.endpoint_prefixes),
            "items": [
                {
                    "id": "conversation-state",
                    "available": state_manager is not None,
                },
                {
                    "id": "auth-session",
                    "available": session_manager is not None,
                },
            ],
        }


class StaticHost:
    id = ""
    title = ""
    order = 0
    endpoint_prefixes: list[str] = []
    items: tuple[dict[str, Any], ...] = ()

    def describe(self, _context: CapabilityContext) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "available": True,
            "status": "ready",
            "endpoint_prefixes": list(self.endpoint_prefixes),
            "count": len(self.items),
            "items": list(self.items),
        }


class DeviceHost(StaticHost):
    id = "devices"
    title = "DeviceHost"
    order = 50
    endpoint_prefixes = ["/api/devices", "/api/controllers", "/api/game-stream", "/api/xr"]
    items = (
        {
            "id": "cast",
            "title": "Cast Receivers",
            "endpoint_prefixes": ["/api/devices", "/cast"],
        },
        {
            "id": "controller",
            "title": "Controller Clients",
            "endpoint_prefixes": ["/api/controllers"],
        },
        {
            "id": "game-stream",
            "title": "Game Stream Clients",
            "endpoint_prefixes": ["/api/game-stream"],
        },
        {
            "id": "surface-receiver",
            "title": "Surface Receivers",
            "endpoint_prefixes": ["/api/surfaces"],
        },
        {
            "id": "xr",
            "title": "XR Clients",
            "endpoint_prefixes": ["/api/xr"],
        },
    )


class SurfaceHost(StaticHost):
    id = "surfaces"
    title = "SurfaceHost"
    order = 60
    endpoint_prefixes = ["/api/surfaces", "/api/media", "/api/games", "/api/titles", "/api/xr"]
    items = (
        {
            "id": "chat",
            "title": "Chat",
            "endpoint_prefixes": ["/api/chat", "/v1/chat/completions", "/ui"],
        },
        {
            "id": "avatar.stage",
            "title": "Avatar Stage",
            "endpoint_prefixes": ["/api/avatar"],
        },
        {
            "id": "browser.surface",
            "title": "Browser Surface",
            "endpoint_prefixes": ["/api/surfaces"],
        },
        {
            "id": "comic.reader.webtoon",
            "title": "Comic Reader",
            "endpoint_prefixes": ["/api/titles"],
        },
        {
            "id": "game.stream.controller",
            "title": "Game Stream Controller",
            "endpoint_prefixes": ["/api/games", "/api/game-stream", "/api/controllers"],
        },
        {
            "id": "media.watch",
            "title": "Media Watch",
            "endpoint_prefixes": ["/api/media", "/api/devices"],
        },
        {
            "id": "xr.room",
            "title": "XR Room",
            "endpoint_prefixes": ["/api/xr"],
        },
    )


def _power_item(manifest: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(manifest, "id", "")),
        "display_name": str(getattr(manifest, "display_name", "")),
        "kind": str(getattr(manifest, "kind", "")),
        "source_kind": str(getattr(manifest, "source_kind", "")),
        "mode_scope": list(getattr(manifest, "mode_scope", []) or []),
        "preferred_tools": list(getattr(manifest, "preferred_tools", []) or []),
        "required_tools": list(getattr(manifest, "required_tools", []) or []),
    }


class PowerHost:
    id = "powers"
    title = "PowerHost"
    order = 70
    endpoint_prefixes = ["/api/powers"]

    def describe(self, context: CapabilityContext) -> dict[str, Any]:
        registry = getattr(context.app_state, "power_registry", None)
        powers = _safe_list(registry.list_powers) if registry is not None else []
        items = sorted((_power_item(power) for power in powers), key=lambda item: item["id"])
        return {
            "id": self.id,
            "title": self.title,
            "available": registry is not None,
            "status": "ready" if registry is not None else "unavailable",
            "endpoint_prefixes": list(self.endpoint_prefixes),
            "count": len(items),
            "items": items,
        }


class JobHost:
    id = "jobs"
    title = "JobHost"
    order = 80
    endpoint_prefixes = ["/api/jobs"]

    def describe(self, _context: CapabilityContext) -> dict[str, Any]:
        handlers_dir = Path(__file__).resolve().parents[1] / "jobs" / "handlers"
        handlers = []
        if handlers_dir.exists():
            handlers = sorted(
                {
                    path.stem.replace("_", "-")
                    for path in handlers_dir.glob("*.py")
                    if path.name != "__init__.py"
                },
            )
        return {
            "id": self.id,
            "title": self.title,
            "available": True,
            "status": "ready",
            "endpoint_prefixes": list(self.endpoint_prefixes),
            "count": len(handlers),
            "items": [{"id": handler, "kind": "handler"} for handler in handlers],
        }


class EventHost(StaticHost):
    id = "events"
    title = "EventHost"
    order = 90
    endpoint_prefixes = ["/api/chat", "/api/jobs", "/api/surfaces"]
    items = (
        {
            "id": "streaming",
            "title": "LLM streaming chunks",
            "endpoint_prefixes": ["/api/chat", "/api/generate", "/v1/chat/completions"],
        },
        {
            "id": "jobs",
            "title": "Background job status",
            "endpoint_prefixes": ["/api/jobs"],
        },
        {
            "id": "surfaces",
            "title": "Surface and device updates",
            "endpoint_prefixes": ["/api/surfaces", "/api/devices"],
        },
    )
