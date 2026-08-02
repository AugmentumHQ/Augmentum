"""REST API routes for reasoning flow management."""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict, deque

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

from augmentum.reasoning.models import FlowCreateRequest, FlowUpdateRequest, ReasoningFlow
from augmentum.reasoning.store import FlowStore
from augmentum.reasoning.templates import get_template, list_templates
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/reasoning", tags=["reasoning"])

_STORE_503 = JSONResponse({"error": "Flow store not initialized"}, status_code=503)

# Rate-limit `/test` runs per (user_id, flow_id) so a stuck-button or a
# pathological loop can't burn through the user's tokens. The window is
# generous — this is a developer-facing dry-run, not a public endpoint.
_TEST_RUN_LIMIT = 10
_TEST_RUN_WINDOW = 60.0  # seconds
_test_run_timestamps: dict[tuple[str, str], deque[float]] = defaultdict(deque)


def _get_store(request: Request) -> FlowStore | None:
    return getattr(request.app.state, "flow_store", None)


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request scope (set by AuthMiddleware)."""
    user = request.scope.get("user")
    return user.id if user else ""


# ------------------------------------------------------------------
# List / Get
# ------------------------------------------------------------------


@router.get("/flows")
async def list_flows(request: Request) -> JSONResponse:
    """List all reasoning flows."""
    store = _get_store(request)
    if not store:
        return _STORE_503
    try:
        flows_with_counts = await store.list_flows(user_id=_user_id(request))
    except Exception as e:
        log.error("list_flows_failed", error=str(e), exc_info=True)
        return JSONResponse({"error": f"Failed to list flows: {e}"}, status_code=503)
    data = []
    for flow, step_count in flows_with_counts:
        d = flow.model_dump(exclude={"steps"})
        d["step_count"] = step_count
        data.append(d)
    return JSONResponse(data)


@router.get("/flows/{flow_id}")
async def get_flow(flow_id: str, request: Request) -> JSONResponse:
    """Get a flow with all its steps."""
    store = _get_store(request)
    if not store:
        return _STORE_503
    flow = await store.get_flow(flow_id, user_id=_user_id(request))
    if not flow:
        return JSONResponse({"error": "Flow not found"}, status_code=404)
    return JSONResponse(flow.model_dump())


# ------------------------------------------------------------------
# Create
# ------------------------------------------------------------------


@router.post("/flows")
async def create_flow(body: FlowCreateRequest, request: Request) -> JSONResponse:
    """Create a new reasoning flow.

    If ``template`` is provided, seeds from a built-in template and applies
    any overrides from the request body.
    """
    store = _get_store(request)
    if not store:
        return _STORE_503

    if body.template:
        flow = get_template(body.template)
        if not flow:
            return JSONResponse(
                {"error": f"Unknown template: {body.template}"},
                status_code=400,
            )
        flow.name = body.name
        flow.is_builtin = False
        flow.is_default = False
        if body.description:
            flow.description = body.description
        if body.steps:
            flow.steps = body.steps
    else:
        flow = ReasoningFlow(
            name=body.name,
            description=body.description,
            icon=body.icon,
            auto_select=body.auto_select,
            trigger_domains=body.trigger_domains,
            trigger_keywords=body.trigger_keywords,
            pinned_models=body.pinned_models,
            auto_search=body.auto_search,
            max_tool_calls_per_step=body.max_tool_calls_per_step,
            steps=body.steps,
        )

    flow = await store.create_flow(flow, user_id=_user_id(request))
    return JSONResponse(flow.model_dump(), status_code=201)


# ------------------------------------------------------------------
# Update
# ------------------------------------------------------------------


@router.put("/flows/{flow_id}")
async def update_flow(flow_id: str, body: FlowUpdateRequest, request: Request) -> JSONResponse:
    """Update a flow. Cannot update built-in flows (clone them instead)."""
    store = _get_store(request)
    if not store:
        return _STORE_503
    updates = body.model_dump(exclude_none=True)
    if not updates:
        return JSONResponse({"error": "No updates provided"}, status_code=400)

    # Convert steps from dicts if present
    if "steps" in updates and updates["steps"] is not None:
        updates["steps"] = [
            s.model_dump() if hasattr(s, "model_dump") else s
            for s in updates["steps"]
        ]

    flow = await store.update_flow(flow_id, updates, user_id=_user_id(request))
    if not flow:
        return JSONResponse(
            {"error": "Flow not found or is a built-in (clone it instead)"},
            status_code=404,
        )
    return JSONResponse(flow.model_dump())


# ------------------------------------------------------------------
# Delete
# ------------------------------------------------------------------


@router.delete("/flows/{flow_id}")
async def delete_flow(flow_id: str, request: Request) -> JSONResponse:
    """Delete a flow. Cannot delete built-in flows."""
    store = _get_store(request)
    if not store:
        return _STORE_503
    deleted = await store.delete_flow(flow_id, user_id=_user_id(request))
    if not deleted:
        return JSONResponse(
            {"error": "Flow not found or is a built-in"},
            status_code=404,
        )
    return JSONResponse({"deleted": True})


# ------------------------------------------------------------------
# Clone
# ------------------------------------------------------------------


@router.post("/flows/{flow_id}/clone")
async def clone_flow(flow_id: str, request: Request) -> JSONResponse:
    """Clone a flow (including built-ins). Creates an editable copy."""
    store = _get_store(request)
    if not store:
        return _STORE_503

    # Optional name from query param
    name = request.query_params.get("name", "")

    clone = await store.clone_flow(flow_id, new_name=name, user_id=_user_id(request))
    if not clone:
        return JSONResponse({"error": "Source flow not found"}, status_code=404)
    return JSONResponse(clone.model_dump(), status_code=201)


# ------------------------------------------------------------------
# Default
# ------------------------------------------------------------------


@router.put("/flows/{flow_id}/default")
async def set_default_flow(flow_id: str, request: Request) -> JSONResponse:
    """Set a flow as the default."""
    store = _get_store(request)
    if not store:
        return _STORE_503
    ok = await store.set_default(flow_id, user_id=_user_id(request))
    if not ok:
        return JSONResponse({"error": "Flow not found"}, status_code=404)
    return JSONResponse({"default": True})


# ------------------------------------------------------------------
# Import / Export
# ------------------------------------------------------------------


@router.get("/flows/{flow_id}/export")
async def export_flow(flow_id: str, request: Request) -> JSONResponse:
    """Export a flow as JSON for sharing."""
    store = _get_store(request)
    if not store:
        return _STORE_503
    data = await store.export_flow(flow_id, user_id=_user_id(request))
    if not data:
        return JSONResponse({"error": "Flow not found"}, status_code=404)
    return JSONResponse(data)


@router.post("/flows/import")
async def import_flow(request: Request) -> JSONResponse:
    """Import a flow from JSON."""
    store = _get_store(request)
    if not store:
        return _STORE_503
    body = await request.json()
    try:
        flow = await store.import_flow(body, user_id=_user_id(request))
    except (ValidationError, ValueError) as e:
        return JSONResponse(
            {"error": f"Invalid flow data: {e}"},
            status_code=400,
        )
    return JSONResponse(flow.model_dump(), status_code=201)


# ------------------------------------------------------------------
# Test run — execute a flow against a sample query, stream events.
# Used by the editor's "Test run" button so authors can validate a
# flow without going through chat. Builds an *ephemeral* engine —
# does not touch session state, prompt cache, or memory.
# ------------------------------------------------------------------


class TestFlowRequest(BaseModel):
    query: str
    model: str = ""
    complexity: str | None = None  # "simple" | "moderate" | "complex" | None
    allow_tools: bool = False


class RoutingPreviewRequest(BaseModel):
    query: str


@router.post("/flows/routing-preview")
async def routing_preview(body: RoutingPreviewRequest, request: Request) -> JSONResponse:
    """Dry-run the auto-routing decision for a sample query.

    Uses the SAME scoring helper as the live resolver
    (``resolver.score_flow_for_query``), so what the flow editor shows is
    exactly what dispatch would do. Returns the ranked candidates with
    their matched keywords, the would-be winner, and — when the user's
    default flow isn't Auto Routing — the fact that keyword matching is
    skipped entirely.
    """
    from augmentum.reasoning.resolver import (
        MIN_AUTO_ROUTE_SCORE,
        score_flow_for_query,
    )

    store = _get_store(request)
    if not store:
        return _STORE_503
    query = (body.query or "").strip()
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)
    if len(query) > 2000:
        return JSONResponse({"error": "query too long (max 2000 chars)"}, status_code=400)

    user_id = _user_id(request)

    # Mirror resolver step 3: an explicit (non-Auto-Routing) default flow
    # short-circuits keyword matching for every chat query.
    default = await store.get_default_flow(user_id=user_id)
    if default and default.name != "Auto Routing":
        return JSONResponse({
            "mode": "default_flow",
            "winner": {"id": default.id, "name": default.name},
            "threshold": MIN_AUTO_ROUTE_SCORE,
            "candidates": [],
            "note": (
                f'Your default flow is "{default.name}", so keyword routing '
                "is skipped — it always wins. Set Auto Routing as default "
                "to route by keywords."
            ),
        })

    flows_with_counts = await store.list_flows(user_id=user_id)
    candidates = []
    for flow_summary, _count in flows_with_counts:
        if flow_summary.name == "Auto Routing":
            continue
        full_flow = await store.get_flow(flow_summary.id, user_id=user_id)
        if not full_flow or not full_flow.auto_select:
            continue
        score, matched = score_flow_for_query(full_flow, query)
        if score > 0:
            candidates.append({
                "id": full_flow.id,
                "name": full_flow.name,
                "score": score,
                "matched": matched,
                "is_builtin": full_flow.is_builtin,
            })
    # Rank exactly like the resolver: highest score first; ties keep list
    # order (which is what dispatch does — worth surfacing, not hiding).
    candidates.sort(key=lambda c: -c["score"])

    winner = candidates[0] if candidates and candidates[0]["score"] >= MIN_AUTO_ROUTE_SCORE else None
    tie = (
        len(candidates) >= 2
        and winner is not None
        and candidates[1]["score"] == candidates[0]["score"]
    )
    return JSONResponse({
        "mode": "auto_routing",
        "winner": winner,
        "tie": tie,
        "threshold": MIN_AUTO_ROUTE_SCORE,
        "candidates": candidates[:10],
        "note": (
            "No flow reaches the score threshold — this query falls "
            "through to the standard analytical pipeline."
        ) if winner is None else (
            "Two flows tie on score — whichever was created first wins. "
            "Differentiate their keywords to make routing deterministic."
        ) if tie else "",
    })


@router.post("/flows/{flow_id}/test")
async def test_flow(flow_id: str, body: TestFlowRequest, request: Request):
    """Stream-execute a flow against a sample query (SSE).

    Events (one per `data:` line, JSON-encoded):
      { "type": "step",   "phase": <name>, "status": "running"|"complete"|... }
      { "type": "delta",  "phase": <name>, "content": "<chunk>" }
      { "type": "tool",   "phase": <name>, "tool": <name>, "status": ... }
      { "type": "done" }
      { "type": "error",  "message": "<reason>" }
    """
    store = _get_store(request)
    if not store:
        return _STORE_503

    user_id = _user_id(request)
    flow = await store.get_flow(flow_id, user_id=user_id)
    if not flow:
        return JSONResponse({"error": "Flow not found"}, status_code=404)

    query = (body.query or "").strip()
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)
    if len(query) > 4000:
        return JSONResponse({"error": "query too long (max 4000 chars)"}, status_code=400)

    # Per-(user, flow) rate limit. Slot the user_id into the key so a
    # shared built-in flow can't be drained by one user against another.
    rate_key = (user_id or "anon", flow_id)
    now = time.monotonic()
    ts_deque = _test_run_timestamps[rate_key]
    while ts_deque and ts_deque[0] < now - _TEST_RUN_WINDOW:
        ts_deque.popleft()
    if len(ts_deque) >= _TEST_RUN_LIMIT:
        return JSONResponse(
            {"error": f"Rate limit exceeded — max {_TEST_RUN_LIMIT} test runs per minute"},
            status_code=429,
        )
    ts_deque.append(now)

    # Resolve dependencies. Some are optional — the engine treats them
    # as best-effort. ProviderRegistry is not optional: without it we
    # can't pick a backend.
    state = request.app.state
    provider_registry = getattr(state, "provider_registry", None)
    if provider_registry is None:
        return JSONResponse({"error": "Provider registry unavailable"}, status_code=503)
    try:
        backend, resolved_model = await provider_registry.resolve_backend_with_fabric(
            (body.model or "").strip()
        )
    except Exception as e:
        log.warning("test_flow_backend_resolve_failed", model=body.model, error=str(e))
        return JSONResponse({"error": f"No backend available for model: {e}"}, status_code=503)
    if backend is None:
        return JSONResponse({"error": "No backend available"}, status_code=503)

    tool_registry = getattr(state, "tool_registry", None)
    prompt_cache = getattr(state, "prompt_cache", None)
    circuit_breaker = getattr(state, "circuit_breaker", None)

    # `allow_tools=False` → empty frozenset → engine treats as "user
    # explicitly chose no tools" and skips the registry path. This
    # keeps test runs cheap and side-effect-free by default.
    enabled_tools: list[str] | None = None if body.allow_tools else []

    flow_tune: dict | None = None
    if body.complexity and body.complexity in {"simple", "moderate", "complex"}:
        flow_tune = {"complexity": body.complexity}

    async def stream_events():
        # Lazy-import here so the `/test` route doesn't drag the engine
        # module into every reasoning-routes import (cold-start cost).
        from augmentum.modes.analytical.engine import AnalyticalEngine
        from augmentum.reasoning.executor import execute_flow_stream

        try:
            engine = AnalyticalEngine(
                backend,
                tool_registry=tool_registry,
                prompt_cache=prompt_cache,
                provider_registry=provider_registry,
                circuit_breaker=circuit_breaker,
                enabled_tools=enabled_tools,
                user_id=user_id,
            )
            engine._state.query = query

            # Initial event so the client knows the stream is live before
            # the first model call returns.
            yield _sse({
                "type": "start",
                "flow": flow.name,
                "model": resolved_model or body.model or "",
                "step_count": len([s for s in flow.steps if s.enabled]),
            })

            async for chunk in execute_flow_stream(
                flow, engine, backend, resolved_model or body.model, query,
                tool_registry=tool_registry,
                provider_registry=provider_registry,
                conversation_context="",  # ephemeral — no session history
                search_context="",
                user_system="",
                flow_tune=flow_tune,
            ):
                event = _chunk_to_event(chunk)
                if event is not None:
                    yield _sse(event)

            yield _sse({"type": "done"})
        except Exception as e:
            log.warning("test_flow_run_failed", flow=flow_id, error=str(e), exc_info=True)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx: don't buffer SSE
        },
    )


def _sse(payload: dict) -> str:
    """Encode a dict as an SSE `data:` line."""
    return f"data: {json.dumps(payload)}\n\n"


def _chunk_to_event(chunk) -> dict | None:
    """Translate an InternalStreamChunk into a compact test-run event.

    The executor emits:
      - phase status updates  (augmentum.phase + augmentum.phase_status)
      - per-token content deltas (content_delta filled in)
      - tool calls (augmentum.tool_calls list)
    We collapse these into the {start|step|delta|tool|done} vocabulary
    the editor's accordion knows how to render.
    """
    meta = chunk.augmentum or {}
    phase = meta.get("phase") or ""
    if meta.get("phase_status"):
        evt = {
            "type": "step",
            "phase": phase,
            "status": meta["phase_status"],
        }
        if meta.get("complexity"):
            evt["complexity"] = meta["complexity"]
        if meta.get("step_model"):
            evt["model"] = meta["step_model"]
        return evt
    tool_calls = meta.get("tool_calls") or []
    if tool_calls:
        # Forward each tool call as its own event so the UI can show
        # them as inline rows under the step.
        # We only forward the first per chunk to avoid SSE spam — the
        # executor emits them one-by-one anyway.
        tc = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
        return {
            "type": "tool",
            "phase": phase,
            "tool": tc.get("tool") if isinstance(tc, dict) else "",
            "status": tc.get("status") if isinstance(tc, dict) else "",
        }
    delta = chunk.content_delta or meta.get("phase_content_delta") or ""
    if delta:
        return {"type": "delta", "phase": phase, "content": delta}
    return None


# ------------------------------------------------------------------
# Templates
# ------------------------------------------------------------------


@router.get("/templates")
async def get_templates() -> JSONResponse:
    """List available built-in flow templates."""
    return JSONResponse(list_templates())


@router.post("/templates/{template_name}/create")
async def create_from_template(template_name: str, request: Request) -> JSONResponse:
    """Create a new editable flow from a built-in template."""
    template_flow = get_template(template_name)
    if not template_flow:
        return JSONResponse({"error": f"Template '{template_name}' not found"}, status_code=404)

    store: FlowStore = getattr(request.app.state, "flow_store", None)
    if not store:
        return JSONResponse({"error": "Flow store not available"}, status_code=503)

    # Create an editable copy (not builtin, not default)
    new_flow = ReasoningFlow(
        id=uuid.uuid4().hex[:16],
        name=f"{template_flow.name} (copy)",
        description=template_flow.description,
        steps=[s.model_copy(update={"id": uuid.uuid4().hex[:16]}) for s in template_flow.steps],
        trigger_domains=template_flow.trigger_domains,
        trigger_keywords=template_flow.trigger_keywords,
        auto_search=template_flow.auto_search,
        auto_select=template_flow.auto_select,
        max_tool_calls_per_step=template_flow.max_tool_calls_per_step,
        is_builtin=False,
        is_default=False,
    )
    # FlowStore has no `.save()` — was an undefined-method bug masked
    # because the create-from-template path is rarely exercised. Use the
    # canonical create_flow with the caller's user_id so the new copy is
    # owned by them, not unscoped.
    await store.create_flow(new_flow, user_id=_user_id(request))
    log.info("flow_created_from_template", template=template_name, flow_id=new_flow.id)
    return JSONResponse(new_flow.model_dump(), status_code=201)
