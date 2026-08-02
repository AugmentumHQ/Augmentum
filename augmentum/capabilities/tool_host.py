"""Tool capability host adapter."""

from __future__ import annotations

from typing import Any

from augmentum.capabilities.frontdesk import CapabilityContext
from augmentum.tools.base import ToolCategory
from augmentum.tools.registry import _PHASE_CATEGORIES


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _safe_list(callable_obj: Any) -> list[Any]:
    try:
        items = callable_obj()
    except Exception:
        return []
    return list(items or [])


def _tool_descriptor(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "input_schema", {}) or {}
    return {
        "id": str(getattr(tool, "name", "")),
        "description": str(getattr(tool, "description", "")),
        "category": _enum_value(getattr(tool, "category", "")),
        "input_schema": schema if isinstance(schema, dict) else {},
        "requires_services": list(getattr(tool, "requires_services", []) or []),
        "produces": list(getattr(tool, "produces", []) or []),
        "consumes": list(getattr(tool, "consumes", []) or []),
        "long_running": bool(getattr(tool, "long_running", False)),
    }


def _ui_tool_descriptor(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "input_schema", {}) or {}
    schema = schema if isinstance(schema, dict) else {}
    props = schema.get("properties", {})
    props = props if isinstance(props, dict) else {}
    required = set(schema.get("required", []) or [])
    type_map = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
    }
    params = []
    for name, definition in props.items():
        definition = definition if isinstance(definition, dict) else {}
        param_type = definition.get("type", "string")
        params.append(
            {
                "name": str(name),
                "type": type_map.get(str(param_type), str(param_type)),
                "required": name in required,
                "description": str(definition.get("description", "")),
            },
        )
    return {
        "name": str(getattr(tool, "name", "")),
        "description": str(getattr(tool, "description", "")),
        "category": _enum_value(getattr(tool, "category", "")),
        "params": params,
    }


class ToolHost:
    id = "tools"
    title = "ToolHost"
    order = 30
    endpoint_prefixes = ["/api/tools"]

    def _registry(self, context: CapabilityContext) -> Any:
        return getattr(context.app_state, "tool_registry", None)

    def describe(self, context: CapabilityContext) -> dict[str, Any]:
        registry = self._registry(context)
        tools = _safe_list(registry.list_tools) if registry is not None else []
        items = sorted((_tool_descriptor(tool) for tool in tools), key=lambda item: item["id"])
        return {
            "id": self.id,
            "title": self.title,
            "available": registry is not None,
            "status": "ready" if registry is not None else "unavailable",
            "summary": "Registered tools and phase availability.",
            "endpoint_prefixes": list(self.endpoint_prefixes),
            "scopes": ["global", "session"],
            "categories": [category.value for category in ToolCategory],
            "phase_categories": {
                phase: [category.value for category in categories]
                for phase, categories in _PHASE_CATEGORIES.items()
            },
            "policy": {
                "invocation": "mode_and_user_selection",
                "explainable": True,
            },
            "events": [
                "tool.call.started",
                "tool.call.finished",
            ],
            "links": {
                "self": "/api/capabilities/tools",
                "primary": "/api/tools",
                "metrics": "/api/tools/metrics",
            },
            "count": len(items),
            "items": items,
        }

    def list_ui_tools(
        self, context: CapabilityContext, *, surface: str = "",
    ) -> dict[str, Any] | list[Any]:
        """UI tool catalog, optionally filtered to one exposure surface.

        ``surface="flow"`` keeps only tools whose ``SurfaceExposure.flow``
        is True — the flow editor's grid and anything else offering tools
        for reasoning-flow steps should pass it, so conversational action
        verbs never show up as flow-step options. No/unknown surface →
        the full catalog (historical behavior).
        """
        registry = self._registry(context)
        if registry is None:
            return []
        raw = _safe_list(registry.list_tools)
        if surface == "flow":
            raw = [
                t for t in raw
                if getattr(getattr(t, "surfaces", None), "flow", True)
            ]
        tools = sorted(
            (_ui_tool_descriptor(tool) for tool in raw),
            key=lambda item: item["name"],
        )
        return {
            "tools": tools,
            "categories": [category.value for category in ToolCategory],
        }

    def metrics(self, context: CapabilityContext) -> dict[str, Any]:
        registry = self._registry(context)
        if registry is None:
            return {}
        metrics = getattr(registry, "metrics", None)
        if metrics is None:
            return {}
        try:
            return metrics.snapshot()
        except Exception:
            return {}
