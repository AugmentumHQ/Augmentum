"""ATP — Augmentum Tool Protocol (Phase 1)

A single, clean, OpenAI-function-calling-shaped surface for external
harnesses (pi, other agents, scripts) to discover and call Augmentum's
registered tools.

Design constraints (zero-disruption):
  * Reads from ``app.state.tool_registry`` — NEVER adds or removes tools.
  * Uses the SAME auth middleware stack as every other route — the
    ``user`` is already resolved by the time this handler runs.
  * Tool whitelist is EXPLICIT and conservative — tools opt IN here,
    not out. Adding a tool to the registry never auto-exposes it via ATP.
  * Lives at ``/v1/tools/`` — a new namespace. No existing client
    routes through ``/v1/tools``.
  * Stateless beyond what the tool registry already provides — no new
    stores, no new tables, no new singletons.
  * The auto-route system (``auto_routes.py``) is NOT modified. ATP is
    the separate, explicit path for external harnesses. The auto-route
    path (``/api/tools/``) remains for Augmentum's own web UI.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from augmentum.tools.base import ToolResult, invoke_tool
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/v1/tools", tags=["atp"])


# ── Whitelist: tools exposed via ATP ──────────────────────────────────
# Format: {tool_name: {"streaming": bool}}
# Only add tools that:
#   1. Are stateless OR handle user_id correctly via extract_user_id()
#   2. Do not require a Docker workspace or other transient resource
#   3. Have a health_check() that accurately reports availability
#   4. Are useful to an external harness (not Augmentum-web-UI-specific)

ATP_TOOLS: dict[str, dict[str, bool]] = {
    # Web & Research
    "web_search":       {"streaming": False},
    "web_fetch":        {"streaming": False},
    "research":         {"streaming": False},
    # Server-side research subagent: launches the multi-step Deep Research
    # flow on Augmentum's own model slots, returns a task id immediately;
    # poll with task_status until completed. (flow_tool.py / flow_status.py)
    "flow_deep_research": {"streaming": False},
    "task_status":      {"streaming": False},
    "wikipedia":        {"streaming": False},
    "youtube":          {"streaming": False},

    # Compute & Verify
    "calculator":       {"streaming": False},
    "json_tool":        {"streaming": False},
    "python_exec":      {"streaming": False},
    "math_verify":      {"streaming": False},
    "consistency_check": {"streaming": False},

    # Documents & Data
    "document_parse":   {"streaming": False},
    "text_analysis":    {"streaming": False},
    "unit_converter":   {"streaming": False},
    "hash_tool":        {"streaming": False},

    # Memory & Context
    "memory_recall":    {"streaming": False},
    "memory_store":     {"streaming": False},  # staged, human-gated (harness_memory.py)
    "context_peek":     {"streaming": False},
    "search_files":     {"streaming": False},

    # Image
    "image_search":     {"streaming": False},

    # Media
    "media_recommendations": {"streaming": False},

    # Browser (agent-browser sidecar; sessions are per-user, screenshots
    # come back as artifact URLs — see augmentum/tools/browser_tools.py)
    "browser_navigate":   {"streaming": False},
    "browser_screenshot": {"streaming": False},
    "browser_action":     {"streaming": False},
    "browser_evaluate":   {"streaming": False},
    "browser_wait":       {"streaming": False},
    "browser_ensure_auth": {"streaming": False},

    # Recipes — named per-user macros that replay a sequence of ATP tool
    # calls in one shot (recipe_tool.py). Steps run through this same route's
    # gate, so a recipe can only reach tools its owner already could.
    "atp_recipe":         {"streaming": False},
    # Self-minted soft procedural memory (workflow_tool.py). The model saves
    # a playbook that worked and refines it; matches auto-surface into the
    # harness briefing by FTS on the when_to_use trigger.
    "workflow":           {"streaming": False},

    # Artifacts — real deliverables from any harness; results carry
    # /api/artifacts/<id>/download URLs.
    "create_document":    {"streaming": False},
    "create_chart":       {"streaming": False},
    "create_spreadsheet": {"streaming": False},

    # Vision & OCR (text-only models "see" screenshots/images —
    # augmentum/tools/vision_tools.py; artifact refs are ownership-checked)
    "vision_describe":    {"streaming": False},
    "ocr_extract":        {"streaming": False},

    # Offline docs/reference retrieval (knowledge packs — the self-hosted
    # Context7; reuses the coder pack_search via tools/pack_search_atp.py)
    "pack_search":        {"streaming": False},

    # Per-user persistent Docker sandbox (reuses coder workspaces —
    # tools/sandbox_tools.py)
    "sandbox_shell":      {"streaming": False},

    # Agent bridge: presence + ask/approve/review through Augmentum
    # notifications, replies from any device (tools/agent_bridge_tools.py)
    "agent_checkin":      {"streaming": False},
    "ask_user":           {"streaming": False},
    "check_reply":        {"streaming": False},
}


# ── Meta-tier: discoverable long tail ─────────────────────────────────
# Beyond the curated ATP_TOOLS (full schemas in /list), any registry tool
# that is ALREADY exposed to the chat LLM surface is callable via ATP after
# discovery: same authenticated user, same _context isolation, so the
# capability boundary is identical to what the user's own chat can do.
# Tools NOT chat-exposed (ATP-only, coder-internal, UI-only) stay out.

def _discoverable(tool) -> bool:
    return tool.name not in ATP_TOOLS and getattr(tool.surfaces, "chat", False)


async def _tool_healthy(tool) -> bool:
    """Health-gate a tool, preferring an async check when the tool has
    one (sync ``health_check`` can only consult caches — e.g. the
    browser sidecar's discovery cache — so it can be stale on the first
    call after startup)."""
    checker = getattr(tool, "health_check_async", None)
    if checker is not None:
        try:
            return bool(await checker())
        except Exception:
            log.warning("atp_async_health_check_failed", tool=tool.name, exc_info=True)
            return False
    return tool.health_check()


def _resolve_user_id(request: Request) -> str:
    """Extract the authenticated user_id from the request scope.

    The auth middleware (augmentum/auth/middleware.py) populates
    ``request.scope["user"]`` before any route handler runs. This is
    the same extraction used by auto_routes.py and every other route.
    """
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


def _inject_user_context(arguments: dict, user_id: str, request: Request) -> dict:
    """Force-set user context so a crafted body cannot impersonate.

    Mirrors the defense in auto_routes.py: strip any caller-supplied
    identity fields and overwrite with the authenticated user. Harness +
    project identity (the memory-scope key) is likewise derived from the
    request headers server-side, never from the body.
    """
    from augmentum.proxy.harness import detect_harness, detect_project

    arguments.pop("_user_id", None)
    ctx = arguments.get("_context")
    if not isinstance(ctx, dict):
        ctx = {}
    ctx["user_id"] = user_id
    ctx["harness"] = detect_harness(request)
    ctx["project"] = detect_project(request)
    arguments["_context"] = ctx
    return arguments


# ── Endpoints ──────────────────────────────────────────────────────────

@router.get("/health")
async def health_check(request: Request) -> dict:
    """Lightweight liveness check. Returns tool registry status."""
    registry = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        return {"ok": False, "error": "Tool registry not initialized"}
    total = len(registry.list_tools())
    available = sum(1 for t in registry.list_tools() if t.health_check())
    return {"ok": True, "tools_total": total, "tools_available": available}


@router.get("/list")
async def list_tools(request: Request) -> dict:
    """Return every ATP-exposed tool in OpenAI function-calling schema format.

    Only tools that pass ``health_check()`` are included — if SearXNG is
    down, web_search won't appear. Clients should call this once at
    startup and cache until they reconnect.
    """
    registry = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        return {"tools": []}

    tools: list[dict] = []
    for name, meta in sorted(ATP_TOOLS.items()):
        tool = registry.resolve(name)
        if tool is None:
            log.debug("atp_tool_not_in_registry", tool=name)
            continue
        if not await _tool_healthy(tool):
            log.debug("atp_tool_unhealthy", tool=name)
            continue

        entry: dict[str, Any] = {
            "name": name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "streaming": meta["streaming"],
            "category": tool.category.value if tool.category else "unknown",
        }
        if tool.error_hints:
            entry["error_hints"] = tool.error_hints
        tools.append(entry)

    return {"tools": tools}


@router.get("/discover")
async def discover_tools(request: Request, q: str = "") -> dict:
    """Search the discoverable long tail (registry tools beyond the curated
    whitelist that the chat surface already exposes). Returns name/
    description/category for matches; full parameter schemas only when the
    match set is small (≤5) so a broad query can't flood the caller's
    context. Call a discovered tool through the normal POST /call."""
    registry = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        return {"tools": []}
    q_l = (q or "").strip().lower()
    matches = []
    for tool in registry.list_tools():
        if not _discoverable(tool):
            continue
        hay = f"{tool.name} {tool.description}".lower()
        if q_l and not all(term in hay for term in q_l.split()):
            continue
        if not tool.health_check():
            continue
        matches.append(tool)
    include_schemas = 0 < len(matches) <= 5
    out = []
    for tool in sorted(matches, key=lambda t: t.name):
        entry: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value if tool.category else "unknown",
        }
        if include_schemas:
            entry["parameters"] = tool.input_schema
        out.append(entry)
    return {"tools": out, "schemas_included": include_schemas}


@router.post("/call")
async def call_tool(request: Request) -> JSONResponse:
    """Execute a tool and return the result.

    Request body::

        {"tool": "web_search", "arguments": {"query": "..."}}

    Response::

        {"ok": true, "output": "...", "metadata": {...}}
        {"ok": false, "error": "...", "warnings": [...]}

    User isolation is automatic — the authenticated user's ID is
    force-injected into ``_context`` so no caller can impersonate.
    """
    registry = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "error": "Tool registry not initialized"},
        )

    # Parse request
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    tool_name = (body.get("tool") or "").strip()
    if not tool_name:
        raise HTTPException(status_code=400, detail="Missing 'tool' field")

    tool = registry.resolve(tool_name)
    if tool_name not in ATP_TOOLS and (tool is None or not _discoverable(tool)):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Tool '{tool_name}' is not available via ATP. "
                f"Whitelisted: {', '.join(sorted(ATP_TOOLS))}. "
                "More tools may be found via GET /v1/tools/discover?q=..."
            ),
        )

    if tool is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found in registry",
        )

    if not await _tool_healthy(tool):
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": f"Tool '{tool_name}' is currently unavailable",
            },
        )

    # Prepare arguments with user isolation
    arguments = body.get("arguments", {})
    if not isinstance(arguments, dict):
        raise HTTPException(
            status_code=400, detail="'arguments' must be a JSON object"
        )

    user_id = _resolve_user_id(request)
    arguments = _inject_user_context(arguments, user_id, request)

    # Execute
    # Some tools have explicit keyword-only signatures (e.g.
    # ``execute(self, *, query: str)``) and reject extra kwargs like
    # ``_context``. Try with context first (for tools that need user_id),
    # fall back without it (for stateless tools that don't).
    # invoke() applies schema coercion and list fan-out. It returns a
    # typed ToolResult rather than RAISING TypeError, so the historical
    # "retry without _context" fallback now keys off failure_kind — an
    # `except TypeError` here would never fire again.
    result: ToolResult = await invoke_tool(tool, arguments)
    if not result.success and result.failure_kind == "invalid_input":
        retry = {k: v for k, v in arguments.items()
                 if k not in ("_context", "_user_id")}
        if retry != arguments:
            result = await invoke_tool(tool, retry)
    if not result.success and result.failure_kind == "internal_error":
        log.warning(
            "atp_tool_execution_failed", tool=tool_name, error=result.error,
        )
        raise HTTPException(
            status_code=500, detail=f"Tool execution failed: {result.error}",
        )

    if not result.success:
        status_code = 400 if result.validation_error else 500
        return JSONResponse(
            status_code=status_code,
            content={
                "ok": False,
                "error": result.error,
                "warnings": result.warnings,
            },
        )

    return JSONResponse({
        "ok": True,
        "output": result.output,
        "metadata": result.metadata,
        "warnings": result.warnings,
        "card": result.card,
    })
