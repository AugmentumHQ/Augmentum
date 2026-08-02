"""REST API routes for custom tool chain flows."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.tools.custom_flows import validate_flow_tools
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/flows", tags=["flows"])

# Per-flow rate limiting: max 10 runs per minute per flow ID
_FLOW_RUN_LIMIT = 10
_FLOW_RUN_WINDOW = 60.0  # seconds
_flow_run_timestamps: dict[str, deque[float]] = defaultdict(lambda: deque())


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


def _get_store(request: Request):
    return getattr(request.app.state, "custom_flow_store", None)


async def _resolve_flow_backend(provider_registry, requested_model: str = ""):
    """Resolve the LLM target for flow generation/execution.

    Blank model names inherit the user's primary chat model through the
    provider registry's empty-model fallback path.
    """
    model = (requested_model or "").strip()
    try:
        return await provider_registry.resolve_backend_with_fabric(model)
    except Exception:
        log.warning("flow_model_resolve_failed", model=model or "(primary)", exc_info=True)
        if model:
            try:
                return await provider_registry.resolve_backend_with_fabric("")
            except Exception:
                log.warning("flow_primary_fallback_failed", exc_info=True)
        return None, ""


async def _resync_flow_tools(request: Request) -> None:
    """Re-register flow tools after a CRUD operation."""
    try:
        tool_registry = getattr(request.app.state, "tool_registry", None)
        store = _get_store(request)
        bg_manager = getattr(request.app.state, "background_chain_manager", None)
        provider_registry = getattr(request.app.state, "provider_registry", None)
        backend = provider_registry.default_backend if provider_registry else None
        if tool_registry and store and bg_manager and backend:
            from augmentum.proxy.handler_factory import register_flow_tools_async

            await register_flow_tools_async(
                tool_registry, store, bg_manager, backend,
                provider_registry=provider_registry,
            )
    except Exception:
        log.warning("flow_tool_resync_failed", exc_info=True)


@router.get("")
async def list_flows(request: Request) -> JSONResponse:
    """List all custom flows."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"flows": []})
    uid = _user_id(request)
    flows = await store.list_flows(user_id=uid)
    # Lazy seed: first time this user opens flows, copy the default templates.
    # (Tier 0 multi-tenant rollout — defaults are no longer server-wide.)
    if not flows and uid:
        try:
            await store.seed_defaults(user_id=uid)
            flows = await store.list_flows(user_id=uid)
        except Exception:
            log.warning("flow_seed_defaults_failed", user_id=uid, exc_info=True)
    return JSONResponse({"flows": flows})


@router.post("")
async def create_flow(request: Request) -> JSONResponse:
    """Create a new custom flow."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Flow store not available"}, status_code=503)
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)
    steps = body.get("steps", [])
    if not steps:
        return JSONResponse({"error": "steps array is required"}, status_code=400)
    # Validate step structure
    for i, s in enumerate(steps):
        if "id" not in s or "tool" not in s:
            return JSONResponse(
                {"error": f"Step {i} must have 'id' and 'tool' fields"},
                status_code=400,
            )
    try:
        flow = await store.create_flow(
            name=name,
            steps=steps,
            description=body.get("description", ""),
            trigger_pattern=body.get("trigger_pattern", ""),
            user_id=_user_id(request),
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    await _resync_flow_tools(request)
    tool_registry = getattr(request.app.state, "tool_registry", None)
    warnings = validate_flow_tools(steps, tool_registry)
    result: dict = {"flow": flow}
    if warnings:
        result["warnings"] = warnings
    return JSONResponse(result, status_code=201)


@router.post("/generate")
async def generate_flow(request: Request) -> JSONResponse:
    """Generate a flow from a natural language description using the LLM.

    Body: {"description": "search for a topic then write python to analyze it"}
    Returns the generated flow definition (not yet saved).
    """
    body = await request.json()
    description = body.get("description", "").strip()
    model = body.get("model", "").strip()
    if not description:
        return JSONResponse({"error": "description is required"}, status_code=400)

    tool_registry = getattr(request.app.state, "tool_registry", None)
    provider_registry = getattr(request.app.state, "provider_registry", None)
    if not tool_registry or not provider_registry:
        return JSONResponse({"error": "Required services not available"}, status_code=503)

    # Resolve the requested model or inherit the user's primary chat model.
    backend, model = await _resolve_flow_backend(provider_registry, model)
    if not backend:
        return JSONResponse({"error": "No backend available"}, status_code=503)

    from augmentum.tools.custom_flows import generate_flow_via_llm

    try:
        flow = await generate_flow_via_llm(description, backend, tool_registry, model=model)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except Exception as exc:
        log.error("flow_generate_failed", error=str(exc), exc_info=True)
        # Connection errors often stringify to "" and used to surface as a
        # blank "Flow generation failed: " toast — fall back to the type name.
        detail = str(exc) or type(exc).__name__
        return JSONResponse(
            {"error": f"Flow generation failed: {detail}"}, status_code=500,
        )

    # Merge validation warnings from generator + route-level check
    gen_warnings = flow.pop("_warnings", [])
    route_warnings = validate_flow_tools(flow.get("steps", []), tool_registry)
    all_warnings = gen_warnings + [w for w in route_warnings if w not in gen_warnings]
    result: dict = {"flow": flow}
    if all_warnings:
        result["warnings"] = all_warnings
    return JSONResponse(result)


@router.get("/match")
async def match_flow(request: Request, q: str = "") -> JSONResponse:
    """Check which flow matches a query."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"match": None})
    flow = await store.match_query(q, user_id=_user_id(request))
    return JSONResponse({"match": flow})


@router.get("/export")
async def export_flows(request: Request) -> JSONResponse:
    """Export all flows as JSON."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"flows": []})
    flows = await store.export_all(user_id=_user_id(request))
    return JSONResponse({"flows": flows})


@router.post("/import")
async def import_flows(request: Request) -> JSONResponse:
    """Import flows from JSON."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Flow store not available"}, status_code=503)
    body = await request.json()
    flows_data = body.get("flows", [])
    count = await store.import_flows(flows_data, user_id=_user_id(request))
    return JSONResponse({"imported": count})


@router.get("/{flow_id}")
async def get_flow(request: Request, flow_id: str) -> JSONResponse:
    """Get a single flow by ID."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Flow store not available"}, status_code=503)
    flow = await store.get_flow(flow_id, user_id=_user_id(request))
    if not flow:
        return JSONResponse({"error": "Flow not found"}, status_code=404)
    return JSONResponse(flow)


# Allowlist of fields the PUT body may set on a flow. Anything else (most
# importantly ``user_id``) is dropped so a client can't smuggle a tenant
# override through the **fields kwargs of CustomFlowStore.update_flow.
_FLOW_UPDATE_FIELDS = frozenset({"name", "description", "trigger_pattern", "steps", "enabled"})


@router.put("/{flow_id}")
async def update_flow(request: Request, flow_id: str) -> JSONResponse:
    """Update a flow."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Flow store not available"}, status_code=503)
    body = await request.json()
    fields = {k: v for k, v in body.items() if k in _FLOW_UPDATE_FIELDS}
    try:
        flow = await store.update_flow(flow_id, user_id=_user_id(request), **fields)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not flow:
        return JSONResponse({"error": "Flow not found"}, status_code=404)
    await _resync_flow_tools(request)
    steps = body.get("steps", [])
    tool_registry = getattr(request.app.state, "tool_registry", None)
    warnings = validate_flow_tools(steps, tool_registry) if steps else []
    result: dict = {"flow": flow}
    if warnings:
        result["warnings"] = warnings
    return JSONResponse(result)


@router.delete("/{flow_id}")
async def delete_flow(request: Request, flow_id: str) -> JSONResponse:
    """Delete a flow."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Flow store not available"}, status_code=503)
    deleted = await store.delete_flow(flow_id, user_id=_user_id(request))
    if not deleted:
        return JSONResponse({"error": "Flow not found"}, status_code=404)
    await _resync_flow_tools(request)
    return JSONResponse({"deleted": True})


@router.post("/{flow_id}/run")
async def run_flow(request: Request, flow_id: str) -> JSONResponse:
    """Manually trigger a flow with a query.

    Body: {"query": "...", "model": "optional-model-name"}
    Returns the execution results.
    """
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Flow store not available"}, status_code=503)
    flow = await store.get_flow(flow_id, user_id=_user_id(request))
    if not flow:
        return JSONResponse({"error": "Flow not found"}, status_code=404)

    # Rate limiting: max N runs per minute per (user, flow) — keyed by
    # user_id so one user can't exhaust the limit for another.
    uid = _user_id(request)
    now = time.monotonic()
    ts_deque = _flow_run_timestamps[(uid, flow_id)]
    while ts_deque and ts_deque[0] < now - _FLOW_RUN_WINDOW:
        ts_deque.popleft()
    if len(ts_deque) >= _FLOW_RUN_LIMIT:
        return JSONResponse(
            {"error": "Rate limit exceeded — max 10 runs per minute per flow"},
            status_code=429,
        )
    ts_deque.append(now)

    body = await request.json()
    query = body.get("query", "")

    # Execute the flow
    from augmentum.tools.chain import execute_chain
    from augmentum.tools.custom_flows import flow_to_plan

    plan = flow_to_plan(flow)

    # Inject {{query}} into any step inputs
    for step in plan.steps:
        if step.input:
            for k, v in step.input.items():
                if isinstance(v, str) and "{{query}}" in v:
                    step.input[k] = v.replace("{{query}}", query)

    tool_registry = getattr(request.app.state, "tool_registry", None)
    provider_registry = getattr(request.app.state, "provider_registry", None)
    if not tool_registry or not provider_registry:
        return JSONResponse({"error": "Required services not available"}, status_code=503)

    model = body.get("model", "")
    backend, model = await _resolve_flow_backend(provider_registry, model)
    if not backend:
        return JSONResponse({"error": "No backend available"}, status_code=503)

    from augmentum.models.base import InternalChatRequest, Message
    ctx = InternalChatRequest(
        model=model,
        messages=[Message(role="user", content=query)],
        stream=False,
    )

    # Constrain replan to the plan's authored tool set so a failed step
    # can't pull a random tool from the registry and surprise the user.
    _allowed_chain_tools = {s.tool for s in plan.steps if getattr(s, "tool", None)}
    results = await execute_chain(
        plan, backend, tool_registry, request_context=ctx,
        allowed_tool_names=_allowed_chain_tools or None,
    )

    return JSONResponse({
        "flow_id": flow_id,
        "flow_name": flow["name"],
        "results": {
            str(k): {
                "step_id": v.step_id,
                "tool": v.tool_name,
                "output": v.output[:2000],
                "success": v.success,
            }
            for k, v in results.items()
        },
    })
