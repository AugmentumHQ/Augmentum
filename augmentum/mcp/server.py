"""MCP server — exposes Augmentum's built-in tools to external MCP clients.

This creates a FastMCP server that wraps each Augmentum tool so that
MCP clients (Claude Desktop, other agents, etc.) can discover and call them.
The server is mounted on the FastAPI application at ``/mcp``.

Multi-tenancy: ``/mcp`` is gated by Augmentum's ASGI auth middleware (it's
NOT in _PUBLIC_PATHS), so every request arriving at a tool handler has
already passed auth. User identity is read from ``ctx.request_context.request``
(the underlying Starlette Request) inside each user-scoped tool. Memory
tools refuse to run for the anonymous tenant.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context, FastMCP

from augmentum.tools.base import invoke_tool
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.memory.store import MemoryStore
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)


def _user_id_from_ctx(ctx: Context | None) -> str:
    """Extract the authenticated user_id from an MCP tool Context.

    Returns "" when the request scope has no user attached (anonymous
    bypass — should never happen in production because the auth
    middleware gates /mcp, but treated defensively as a refusal signal).
    """
    if ctx is None:
        return ""
    try:
        request = ctx.request_context.request
    except (ValueError, AttributeError):
        return ""
    if request is None:
        return ""
    scope = getattr(request, "scope", None) or {}
    user = scope.get("user") if isinstance(scope, dict) else None
    return getattr(user, "id", "") if user else ""

# JSON Schema type → Python annotation map
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def create_mcp_server(
    tool_registry: ToolRegistry,
    memory_store: MemoryStore | None = None,
    app: Any = None,
) -> FastMCP:
    """Build a FastMCP server that exposes Augmentum tools, resources, and prompts.

    Each tool in the registry is wrapped as an MCP tool with a generic
    ``execute(**params)`` interface. Memory tools are exposed as dedicated
    typed tools, scoped to the calling user. When ``app`` is supplied,
    character cards and knowledge packs become MCP resources, and prompt
    presets / reasoning flows become MCP prompts — all user-scoped via
    Context (see ``_user_id_from_ctx``).
    """
    from mcp.server.transport_security import TransportSecuritySettings

    mcp = FastMCP(
        "augmentum",
        # Serve the Streamable-HTTP endpoint at the app's root so that
        # mounting under ``/mcp`` (see mount_mcp_server) yields the clean,
        # intuitive ``/mcp/`` URL. FastMCP defaults this to ``/mcp``, which
        # under the ``/mcp`` mount would double to ``/mcp/mcp/`` — a silent
        # 404 for anyone configuring the obvious ``/mcp/`` in their client.
        streamable_http_path="/",
        # FastMCP's DNS-rebinding protection defaults to ON with an EMPTY
        # host allowlist — which 421s every request (found live
        # 2026-07-18). Augmentum is reached under many hosts (LAN IP,
        # tailnet name, user domains) that can't be enumerated here, and
        # /mcp/ already sits behind the session-auth middleware; a
        # DNS-rebound origin can't attach the session cookie (different
        # domain) or a Bearer key, so the MCP-level Host check adds
        # nothing but breakage.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        ),
        instructions=(
            "Augmentum intelligence layer. Tools: web search, Python execution, "
            "math verification, file operations, text analysis, memory recall. "
            "Resources: your character cards (augmentum://characters/{id}) and "
            "installed knowledge packs (augmentum://knowledge/{pack_id}). "
            "Prompts: your saved prompt presets and reasoning flows."
        ),
    )

    # ------------------------------------------------------------------
    # Expose each registered tool dynamically
    # ------------------------------------------------------------------
    _register_builtin_tools(mcp, tool_registry)

    # ------------------------------------------------------------------
    # Memory tools (if memory store is available)
    # ------------------------------------------------------------------
    if memory_store is not None:
        _register_memory_tools(mcp, memory_store)

    # ------------------------------------------------------------------
    # Resources + prompts (need access to app.state for backend / stores)
    # ------------------------------------------------------------------
    if app is not None:
        _register_resources(mcp, app)
        _register_prompts(mcp, app)
        _register_bug_finder_tools(mcp, app)

    log.info("mcp_server_created")
    return mcp


def _build_typed_handler(tool: Any) -> Any:
    """Build an async handler with a proper signature from the tool's input_schema.

    FastMCP uses function signatures to generate MCP tool schemas and validate
    arguments. We dynamically construct a function whose signature matches
    the tool's JSON Schema so MCP clients see the correct parameters.
    """
    schema = tool.input_schema
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Build inspect.Parameter list for the function signature
    params: list[inspect.Parameter] = []
    for pname, pdef in props.items():
        annotation = _TYPE_MAP.get(pdef.get("type", "string"), str)
        default = pdef.get("default", inspect.Parameter.empty)
        if pname not in required and default is inspect.Parameter.empty:
            default = None
        params.append(
            inspect.Parameter(
                pname,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )

    # If no properties in schema, accept generic **kwargs
    if not params:

        async def _generic_handler(**kwargs: Any) -> str:
            result = await invoke_tool(tool, kwargs)
            return result.output if result.success else f"Error: {result.error}"

        _generic_handler.__name__ = tool.name
        _generic_handler.__doc__ = tool.description
        return _generic_handler

    # Build a handler with proper signature using exec (safe — schema is trusted)
    async def _handler(**kwargs: Any) -> str:
        # invoke(): external MCP clients are the LEAST trusted arg source
        # we have — no prompt of ours shapes them — so schema coercion
        # matters more here than anywhere.
        result = await invoke_tool(tool, kwargs)
        return result.output if result.success else f"Error: {result.error}"

    # Apply the correct signature with typed parameters
    sig = inspect.Signature(params, return_annotation=str)
    _handler.__signature__ = sig
    _handler.__name__ = tool.name
    _handler.__doc__ = tool.description
    _handler.__annotations__ = {p.name: p.annotation for p in params}
    _handler.__annotations__["return"] = str

    return _handler


def _register_builtin_tools(mcp: FastMCP, registry: ToolRegistry) -> None:
    """Register each Augmentum tool as an MCP tool."""
    for tool in registry.list_tools():
        fn = _build_typed_handler(tool)
        mcp.add_tool(fn, name=tool.name, description=tool.description)
        log.debug("mcp_tool_exposed", name=tool.name)


def _register_memory_tools(mcp: FastMCP, store: MemoryStore) -> None:
    """Register memory-specific MCP tools, scoped per authenticated user.

    Each handler pulls the user_id from the request scope on every call so
    distinct MCP clients (even ones sharing the same FastMCP server
    instance) never see each other's memories.
    """

    _UNAUTH = "Authentication required: memory operations need a logged-in Augmentum user."

    @mcp.tool(name="memory_recall", description="Search Augmentum's semantic memory for the calling user.")
    async def memory_recall(query: str, ctx: Context, limit: int = 5) -> str:
        """Search the calling user's semantic memory."""
        user_id = _user_id_from_ctx(ctx)
        if not user_id:
            return _UNAUTH
        memories = await store.recall(query, user_id=user_id, limit=limit)
        if not memories:
            return "No relevant memories found."
        lines = []
        for mem in memories:
            conf = f" (confidence: {mem.confidence:.2f})" if mem.confidence < 1.0 else ""
            lines.append(f"- [{mem.memory_type.value}] {mem.content}{conf}")
        return "\n".join(lines)

    @mcp.tool(name="memory_store", description="Store a fact or preference in the calling user's memory.")
    async def memory_store_tool(content: str, ctx: Context, memory_type: str = "fact") -> str:
        """Store a memory under the calling user."""
        user_id = _user_id_from_ctx(ctx)
        if not user_id:
            return _UNAUTH
        from augmentum.memory.models import MemoryType
        try:
            mt = MemoryType(memory_type)
        except ValueError:
            return f"Invalid memory type: {memory_type}. Use: fact, preference, entity, narrative, analysis."
        memory_id = await store.store(content, mt, user_id=user_id)
        return f"Stored memory {memory_id}"

    @mcp.tool(name="memory_count", description="Get the count of memories belonging to the calling user.")
    async def memory_count(ctx: Context) -> str:
        """Count memories for the calling user."""
        user_id = _user_id_from_ctx(ctx)
        if not user_id:
            return _UNAUTH
        count = await store.count(user_id=user_id)
        # ``count`` returns a dict[str, int] keyed by memory_type; flatten for display.
        if isinstance(count, dict):
            total = sum(count.values())
            return f"Total memories: {total}"
        return f"Total memories: {count}"


def _register_resources(mcp: FastMCP, app: Any) -> None:
    """Register MCP resources for character cards and knowledge packs.

    Resources are user-scoped: the discovery tools filter by the calling
    user's ``user_id``, and the URI-template handlers refuse to read a
    resource owned by a different user.
    """
    import json

    _UNAUTH = "Authentication required."

    @mcp.tool(
        name="list_character_cards",
        description="List the calling user's character cards. Returns name + augmentum:// URI for each.",
    )
    async def list_character_cards(ctx: Context) -> str:
        user_id = _user_id_from_ctx(ctx)
        if not user_id:
            return _UNAUTH
        backend = getattr(app.state, "backend", None)
        if backend is None or getattr(backend, "conn", None) is None:
            return "Character card store unavailable."
        cursor = await backend.conn.execute(
            "SELECT id, name FROM ui_characters WHERE user_id = ? ORDER BY updated_at DESC LIMIT 500",
            [user_id],
        )
        rows = await cursor.fetchall()
        if not rows:
            return "No character cards found."
        lines = [f"- {r[1]}  ({r[0]})  →  augmentum://characters/{r[0]}" for r in rows]
        return "\n".join(lines)

    @mcp.resource(
        "augmentum://characters/{character_id}",
        name="character_card",
        description="A user's character card (personality, scenario, example dialogue). User-scoped.",
        mime_type="application/json",
    )
    async def read_character_card(character_id: str, ctx: Context) -> str:
        user_id = _user_id_from_ctx(ctx)
        if not user_id:
            return json.dumps({"error": "authentication_required"})
        backend = getattr(app.state, "backend", None)
        if backend is None or getattr(backend, "conn", None) is None:
            return json.dumps({"error": "store_unavailable"})
        cursor = await backend.conn.execute(
            "SELECT id, name, data, avatar, created_at, updated_at "
            "FROM ui_characters WHERE id = ? AND user_id = ?",
            [character_id, user_id],
        )
        row = await cursor.fetchone()
        if not row:
            # Either nonexistent or owned by a different user — same error
            # surface so we don't leak existence-vs-ownership distinctions.
            return json.dumps({"error": "not_found", "character_id": character_id})
        try:
            card = json.loads(row[2])
        except json.JSONDecodeError:
            card = {}
        card["id"] = row[0]
        card["name"] = row[1]
        if row[3]:
            card["avatar"] = row[3]
        card["createdAt"] = row[4]
        card["updatedAt"] = row[5]
        return json.dumps(card, ensure_ascii=False)

    @mcp.tool(
        name="list_knowledge_packs",
        description="List installed knowledge packs (Wikipedia, Stack Exchange, etc.). Returns pack_id + URI.",
    )
    async def list_knowledge_packs(ctx: Context) -> str:
        user_id = _user_id_from_ctx(ctx)
        if not user_id:
            return _UNAUTH
        pack_manager = getattr(app.state, "pack_manager", None)
        if pack_manager is None:
            return "Knowledge pack subsystem not enabled."
        try:
            packs = pack_manager.installed
        except Exception:
            log.warning("mcp_list_packs_failed", exc_info=True)
            return "Failed to list knowledge packs."
        if not packs:
            return "No knowledge packs installed."
        lines = []
        for p in packs:
            pid = p.get("pack_id", "")
            name = p.get("name", pid)
            chunks = p.get("chunk_count", 0)
            lines.append(f"- {name}  ({chunks} chunks)  →  augmentum://knowledge/{pid}")
        return "\n".join(lines)

    @mcp.resource(
        "augmentum://knowledge/{pack_id}",
        name="knowledge_pack",
        description="Metadata for an installed knowledge pack (name, license, chunk count, embedding model).",
        mime_type="application/json",
    )
    async def read_knowledge_pack(pack_id: str, ctx: Context) -> str:
        user_id = _user_id_from_ctx(ctx)
        if not user_id:
            return json.dumps({"error": "authentication_required"})
        pack_manager = getattr(app.state, "pack_manager", None)
        if pack_manager is None:
            return json.dumps({"error": "subsystem_disabled"})
        try:
            packs = pack_manager.installed
        except Exception:
            log.warning("mcp_read_pack_failed", pack=pack_id, exc_info=True)
            return json.dumps({"error": "store_unavailable"})
        for p in packs:
            if p.get("pack_id") == pack_id:
                return json.dumps(p, ensure_ascii=False, default=str)
        return json.dumps({"error": "not_found", "pack_id": pack_id})


def _register_prompts(mcp: FastMCP, app: Any) -> None:
    """Register MCP prompts for the user's prompt presets and the modular composer.

    Presets are user-scoped: discovery lists only the calling user's saved
    presets, and ``apply_prompt_preset`` refuses to load another user's row.
    The ``compose_modular_prompt`` slot exposes Augmentum's modular system-
    prompt composer to any MCP client — no preset row required.
    """

    @mcp.tool(
        name="list_prompt_presets",
        description="List the calling user's saved prompt presets. Returns id + name + default flag.",
    )
    async def list_prompt_presets(ctx: Context) -> str:
        user_id = _user_id_from_ctx(ctx)
        if not user_id:
            return "Authentication required."
        backend = getattr(app.state, "backend", None)
        if backend is None or getattr(backend, "conn", None) is None:
            return "Preset store unavailable."
        from augmentum.modes.narrative.prompt_presets import PromptPresetStore
        store = PromptPresetStore(backend.conn)
        presets = await store.list_presets(user_id=user_id)
        if not presets:
            return "No saved prompt presets."
        lines = []
        for p in presets:
            default_tag = "  [default]" if p.is_default else ""
            lines.append(f"- {p.name} ({p.id}){default_tag}")
        return "\n".join(lines)

    @mcp.prompt(
        name="apply_prompt_preset",
        description=(
            "Load a saved Augmentum prompt preset by id and return its composed "
            "system prompt as a user message. Modular presets are rendered through "
            "the toggle composer; literal presets return their stored system_prompt."
        ),
    )
    async def apply_prompt_preset(preset_id: str, ctx: Context) -> list[dict]:
        user_id = _user_id_from_ctx(ctx)
        if not user_id:
            return [{"role": "user", "content": "Authentication required."}]
        backend = getattr(app.state, "backend", None)
        if backend is None or getattr(backend, "conn", None) is None:
            return [{"role": "user", "content": "Preset store unavailable."}]
        from augmentum.modes.narrative.prompt_presets import (
            PromptPresetStore,
            compose_modular_system_prompt,
        )
        store = PromptPresetStore(backend.conn)
        preset = await store.get_preset(preset_id, user_id=user_id)
        if preset is None:
            return [{"role": "user", "content": f"Preset {preset_id!r} not found."}]
        modular_cfg = preset.load_modular_config()
        if modular_cfg:
            text = compose_modular_system_prompt(modular_cfg)
        else:
            text = preset.system_prompt or "(preset has empty system_prompt)"
        return [{"role": "user", "content": text}]

    @mcp.prompt(
        name="compose_modular_prompt",
        description=(
            "Compose a system prompt from Augmentum's modular toggles "
            "(role/tense/pov/pov_mode/length/tone/content). Returns the rendered "
            "prompt as a user message. Defaults match Augmentum's narrative-mode "
            "modular preset."
        ),
    )
    async def compose_modular_prompt(
        ctx: Context,
        role: str = "roleplayer",
        tense: str = "present",
        pov: str = "third",
        pov_mode: str = "character",
        length: str = "moderate",
        tone: str = "neutral",
        content: str = "sfw",
    ) -> list[dict]:
        # No auth gate — this is a pure composer with no user data access.
        # ctx is required by FastMCP's prompt signature contract; unused here.
        del ctx
        from augmentum.modes.narrative.prompt_presets import (
            compose_modular_system_prompt,
        )
        cfg = {
            "role": role, "tense": tense, "pov": pov, "pov_mode": pov_mode,
            "length": length, "tone": tone, "content": content,
        }
        text = compose_modular_system_prompt(cfg)
        return [{"role": "user", "content": text}]


def _register_bug_finder_tools(mcp: FastMCP, app: Any) -> None:
    """Register the bug-finder run/list/get tools.

    Exposes the same surface as the REST API
    (``POST/GET /api/bug-finder/runs``) for external MCP clients —
    Claude Desktop, Cursor, Cline etc. — so they can dispatch audits
    against the calling user's workspaces and read the structured
    report once complete.

    All three are user-scoped via ``_user_id_from_ctx`` and apply the
    same capability gate as the REST route.
    """
    import json
    import uuid as _uuid

    _UNAUTH = "Authentication required."

    @mcp.tool(
        name="bug_finder_run",
        description=(
            "Kick off an autonomous bug-finder audit on an existing "
            "Augmentum coder workspace. Returns the run_id + job_id "
            "immediately; poll bug_finder_get_run for the report once "
            "the run completes (5-30 minutes typical). The pipeline "
            "runs planner → detector → verifier → fixer subagents and "
            "writes the report to Augmentum's audit history."
        ),
    )
    async def bug_finder_run(
        workspace_id: str,
        primary_model: str,
        ctx: Context,
        verifier_model: str = "",
        focus_paths: list | None = None,
        threat_model: str = "",
        force_below_minimum: bool = False,
    ) -> str:
        user_id = _user_id_from_ctx(ctx)
        if not user_id:
            return _UNAUTH
        workspace_id = (workspace_id or "").strip()
        primary_model = (primary_model or "").strip()
        if not workspace_id:
            return "workspace_id is required."
        if not primary_model:
            return "primary_model is required."

        from augmentum.bug_finder.capability import (
            capability_floor_label,
            is_capable,
        )
        if not force_below_minimum and not is_capable(primary_model):
            return (
                f"primary_model '{primary_model}' is below the bug-finder "
                "capability floor. The detector / verifier / fixer prompts "
                "target capable instruction-followers; below-floor models "
                "tend to produce malformed JSON the parsers silently "
                "reject. Recommended floor: "
                f"{capability_floor_label()}. "
                "Pass force_below_minimum=true to override."
            )

        jobs_store = getattr(app.state, "jobs_store", None)
        job_runner = getattr(app.state, "job_runner", None)
        if jobs_store is None or job_runner is None:
            return "Background job queue unavailable on this Augmentum instance."

        run_id = f"bfr_{_uuid.uuid4().hex[:12]}"
        focus = [
            str(p).strip() for p in (focus_paths or [])
            if str(p).strip()
        ]
        payload: dict[str, Any] = {
            "run_id": run_id,
            "workspace_id": workspace_id,
            "primary_model": primary_model,
            "verifier_model": (verifier_model or "").strip(),
            "focus_paths": focus,
            "threat_model": (threat_model or "").strip(),
        }
        if force_below_minimum:
            payload["force_below_minimum"] = True

        job_id = await jobs_store.create(
            user_id=user_id,
            job_type="bug_finder_run",
            payload=payload,
            priority=5,
            max_attempts=1,
        )
        job_runner.wake()
        return (
            f"Bug-finder run enqueued.\n"
            f"  run_id: {run_id}\n"
            f"  job_id: {job_id}\n"
            f"  workspace: {workspace_id}\n"
            f"  primary_model: {primary_model}\n"
            "Use bug_finder_get_run with the run_id once the audit "
            "finishes (typically 5-30 minutes)."
        )

    @mcp.tool(
        name="bug_finder_list_runs",
        description=(
            "List recent bug-finder runs belonging to the calling user, "
            "newest first. Lightweight — no embedded report blob."
        ),
    )
    async def bug_finder_list_runs(ctx: Context, limit: int = 20) -> str:
        user_id = _user_id_from_ctx(ctx)
        if not user_id:
            return _UNAUTH
        from augmentum.bug_finder.store import BugFinderRunStore
        backend = getattr(app.state, "backend", None)
        conn = getattr(backend, "conn", None)
        if conn is None:
            return "Bug-finder store unavailable."
        store = BugFinderRunStore(conn)
        rows = await store.list_runs(
            user_id=user_id, limit=max(1, min(int(limit), 200)),
        )
        if not rows:
            return "No bug-finder runs found."
        lines: list[str] = []
        for r in rows:
            status = (r.get("stop_reason") or "running").strip()
            total = r.get("findings_total") or 0
            fixed = r.get("findings_fixed") or 0
            conf = r.get("findings_confirmed") or 0
            lines.append(
                f"- {r.get('run_id', '')}  "
                f"[{status}]  ws={r.get('workspace_id', '')}  "
                f"{total} findings ({fixed} fixed, {conf} conf)"
            )
        return "\n".join(lines)

    @mcp.tool(
        name="bug_finder_get_run",
        description=(
            "Fetch the full bug-finder report for one run_id. Returns a "
            "structured summary plus all findings, the cost ledger, and "
            "the workspace baseline. Use this after bug_finder_run "
            "returns to read the audit result."
        ),
    )
    async def bug_finder_get_run(run_id: str, ctx: Context) -> str:
        user_id = _user_id_from_ctx(ctx)
        if not user_id:
            return _UNAUTH
        from augmentum.bug_finder.store import BugFinderRunStore
        backend = getattr(app.state, "backend", None)
        conn = getattr(backend, "conn", None)
        if conn is None:
            return "Bug-finder store unavailable."
        store = BugFinderRunStore(conn)
        row = await store.get_run(run_id, user_id=user_id)
        if row is None:
            return f"Run '{run_id}' not found."
        status = (row.get("stop_reason") or "running").strip()
        if status == "running":
            return (
                f"Run {run_id} still running.\n"
                f"  workspace: {row.get('workspace_id', '')}\n"
                f"  started_at: {row.get('started_at', '')}\n"
                "Check back later."
            )
        report = row.get("report") or {}
        # Return a structured JSON so clients can render or further
        # process. Strings inside Augmentum-shaped payloads are already
        # serializable.
        envelope = {
            "run_id": row.get("run_id"),
            "stop_reason": status,
            "stop_detail": row.get("stop_detail") or "",
            "workspace_id": row.get("workspace_id"),
            "started_at": row.get("started_at"),
            "completed_at": row.get("completed_at"),
            "findings_total": row.get("findings_total"),
            "findings_fixed": row.get("findings_fixed"),
            "findings_confirmed": row.get("findings_confirmed"),
            "findings_fix_failed": row.get("findings_fix_failed"),
            "tokens_in": row.get("total_tokens_in"),
            "tokens_out": row.get("total_tokens_out"),
            "wallclock_ms": row.get("total_wallclock_ms"),
            "report": report,
        }
        return json.dumps(envelope, indent=2, default=str)


def mount_mcp_server(app: Any, mcp: FastMCP) -> None:
    """Mount the MCP server's Streamable HTTP app onto a FastAPI application.

    This makes the MCP server available at ``/mcp`` for external clients.
    """
    starlette_app = mcp.streamable_http_app()
    app.mount("/mcp", starlette_app)
    log.info("mcp_server_mounted", path="/mcp")
