"""REST API routes for Augmentum capability discovery."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.capabilities import build_default_frontdesk, context_from_request
from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/capabilities", tags=["capabilities"])


# ── Per-user TTL cache ────────────────────────────────────────────
# /api/capabilities is polled by the UI every few seconds and runs
# list_flows + frontdesk inventory + an image-providers count on
# every call — collectively ~500–800ms even on warm caches. The
# payload only changes when the user adjusts settings or adds a
# flow / provider, so a short TTL is safe and kills the polling cost.
#
# Cache key = user_id (flows are per-user). 5s TTL means a settings
# change is visible within a single poll cycle. Memory-only; flushes
# on server restart, which is the right invalidation point for any
# server-level field changes.
_CAPABILITY_CACHE_TTL_S = 5.0
_capability_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _capability_cache_key(request: Request) -> str:
    user = request.scope.get("user")
    return getattr(user, "id", "") or "anon"


def _capability_cache_get(key: str) -> dict[str, Any] | None:
    entry = _capability_cache.get(key)
    if entry is None:
        return None
    ts, payload = entry
    if time.monotonic() - ts >= _CAPABILITY_CACHE_TTL_S:
        _capability_cache.pop(key, None)
        return None
    return payload


def _capability_cache_set(key: str, payload: dict[str, Any]) -> None:
    _capability_cache[key] = (time.monotonic(), payload)


def build_capability_inventory(request: Request) -> dict:
    frontdesk = build_default_frontdesk()
    return frontdesk.inventory(context_from_request(request))


def _backend_names(registry: Any) -> list[str]:
    backends = getattr(registry, "backends", {}) if registry else {}
    if not isinstance(backends, Mapping):
        return []
    return [str(name) for name in backends]


async def _cloud_image_available(request: Request) -> bool:
    state_manager = getattr(request.app.state, "state_manager", None)
    if state_manager is None:
        return False

    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = getattr(state_manager, "backend", None)
    if not isinstance(backend, SQLiteBackend):
        return False

    try:
        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM image_providers WHERE is_enabled = 1",
        )
        row = await cursor.fetchone()
    except Exception:
        return False
    return bool(row and row[0] > 0)


async def _flow_names(request: Request) -> list[dict[str, Any]]:
    flow_store = getattr(request.app.state, "flow_store", None)
    if flow_store is None:
        return []
    try:
        flows_list = await flow_store.list_flows()
    except Exception as exc:
        log.warning("flow_list_failed", error=str(exc))
        return []
    return [
        {"id": flow.id, "name": flow.name, "is_default": flow.is_default}
        for flow, _ in flows_list
    ]


async def build_capability_payload(request: Request) -> dict[str, Any]:
    """Return the compatibility capability payload plus frontdesk inventory.

    Cached per-user with a short TTL (see _CAPABILITY_CACHE_TTL_S) so
    aggressive UI polling doesn't run the underlying list_flows +
    inventory work on every call.
    """
    cache_key = _capability_cache_key(request)
    cached = _capability_cache_get(cache_key)
    if cached is not None:
        return cached
    payload = await _build_capability_payload_uncached(request)
    _capability_cache_set(cache_key, payload)
    return payload


async def _build_capability_payload_uncached(request: Request) -> dict[str, Any]:
    """Uncached payload build. Don't call directly — use
    build_capability_payload() so polling clients hit the cache."""
    registry = getattr(request.app.state, "provider_registry", None)
    backends = _backend_names(registry)
    local_backends = [backend for backend in backends if backend in ("ollama", "llamacpp")]
    discovered = [backend for backend in backends if backend.startswith("local-")]
    discovered_services = getattr(registry, "_discovered", []) if registry else []
    if not isinstance(discovered_services, list):
        discovered_services = []

    flow_names = await _flow_names(request)
    is_dmr = bool(settings.ollama_base_url and "model-runner" in settings.ollama_base_url)

    payload: dict[str, Any] = {
        "image_enabled": settings.image_enabled or await _cloud_image_available(request),
        "memory_enabled": settings.memory_enabled,
        "mcp_enabled": settings.mcp_enabled,
        "voice_enabled": settings.voice_enabled,
        "backends": backends,
        "has_backends": len(backends) > 0,
        "has_local_backends": len(local_backends) > 0 or len(discovered) > 0,
        "local_backends": local_backends + discovered,
        "has_llamacpp": "llamacpp" in backends,
        "has_ollama": "ollama" in backends,
        "has_engine": "engine" in backends,
        # The optional vLLM safetensors engine (installed from Discover). When
        # present, the model manager surfaces the safetensors format + library.
        "has_vllm": "vllm" in backends,
        "is_docker_model_runner": is_dmr,
        "discovered": discovered,
        "discovered_services": discovered_services,
        "persistence_degraded": getattr(request.app.state, "persistence_degraded", False),
        "modes": {
            "passthrough": "Chat - normal conversation, tools optional",
            "analytical": "Thinker - forced multi-step reasoning for deeper answers",
            "narrative": "Narrative - character cards, world state, roleplay",
            "agentic": "Creator - plans and delivers tangible artifacts",
            "coder": "Coder - terminal + container, AI coding agent",
            "direct": "Direct - raw verbatim pass-through, no Augmentum features (memory/tools/etc.)",
        },
        "model_prefixes": {
            "p/": "Force Chat (passthrough) mode (e.g. p/llama3.2)",
            "a/": "Force Thinker (analytical) mode",
            "n/": "Force Narrative mode",
            "g/": "Force Creator (agentic) mode",
            "c/": "Force Coder mode",
            "d/": "Force Direct mode - raw verbatim pass-through, no Augmentum features",
        },
        "headers": {
            "X-Augmentum-Mode": {
                "description": "Override the automatic mode classifier. 'direct' is raw verbatim pass-through (no memory/tools/injection).",
                "values": ["passthrough", "analytical", "narrative", "agentic", "coder", "direct"],
                "example": "analytical",
            },
            "X-Augmentum-Session": {
                "description": "Explicit session ID for conversation continuity and memory",
                "values": "Any string (UUID recommended)",
                "example": "abc123-session-id",
            },
            "X-Augmentum-Flow": {
                "description": "Select a specific reasoning flow by ID (overrides auto-routing). Only used in analytical mode.",
                "values": "Flow ID string (see 'flows' field below)",
                "example": flow_names[0]["id"] if flow_names else "",
            },
            "X-Augmentum-Flow-Tune": {
                "description": "Per-message flow overrides. Skip steps or disable tools for one request.",
                "values": "JSON object",
                "example": '{"skip_steps": [3], "disable_tools": ["web_search"]}',
            },
            "X-Augmentum-Complexity": {
                "description": "Skip the classify step and force a complexity level. Saves one LLM call.",
                "values": ["simple", "moderate", "complex"],
                "example": "simple",
            },
            "X-Augmentum-Tools": {
                "description": "Enable/disable tools for passthrough and analytical modes",
                "values": "Comma-separated tool names, 'all', or 'none'",
                "example": "web_search,calculator",
            },
            "X-Augmentum-Workspace": {
                "description": "Workspace ID for coder mode",
                "values": "Workspace ID string",
            },
            "X-Augmentum-Memory-Query": {
                "description": "Override the memory recall query. By default, the last user message is used. Set this to recall memories about a specific topic instead.",
                "values": "Any text string",
                "example": "user preferences for Python code style",
            },
            "X-Augmentum-Client-ID": {
                "description": "Client identity for session isolation in multi-user setups",
                "values": "Any unique client identifier",
            },
        },
        "flows": flow_names,
        "endpoints": {
            "ollama_chat": "/api/chat",
            "openai_chat": "/v1/chat/completions",
            "models_ollama": "/api/tags",
            "models_openai": "/v1/models",
            "tts": "/v1/audio/speech",
            "stt": "/v1/audio/transcriptions",
            "embeddings": "/v1/embeddings",
            "images": "/v1/images/generations",
            "images_edits": "/v1/images/edits",
            "memory_search": "/v1/memory/search",
            "memory_store": "/v1/memory/store",
            # MCP server (Streamable HTTP) — exposes Augmentum's tools +
            # memory to MCP clients (Claude Desktop/Code, Cursor). Live when
            # ``mcp_enabled`` is true; auth via Authorization: Bearer <sk-aug key>.
            "mcp": "/mcp/",
            "flows": "/api/reasoning/flows",
            "health": "/api/health",
        },
    }
    payload.update(build_capability_inventory(request))
    return payload


@router.get("/{host_id}")
async def get_capability_host(request: Request, host_id: str) -> JSONResponse:
    """Return one capability host descriptor."""
    frontdesk = build_default_frontdesk()
    host = frontdesk.host(host_id, context_from_request(request))
    if host is None:
        return JSONResponse({"error": "Capability host not found"}, status_code=404)
    return JSONResponse(host)
