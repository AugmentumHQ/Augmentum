"""REST API routes for agentic task management."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/agentic", tags=["agentic"])


def _get_task_store(request: Request):
    return getattr(request.app.state, "task_store", None)


def _current_user_id(request: Request) -> str:
    """Pull user_id off the ASGI scope populated by the auth middleware.

    Empty string means "anonymous / auth disabled" — routes return an empty
    set for anonymous callers rather than leaking every tenant's tasks.
    """
    user = request.scope.get("user")
    return user.id if user else ""


@router.get("/tasks")
async def list_tasks(
    request: Request,
    session_id: str = "",
) -> JSONResponse:
    """List agentic tasks for the authenticated user in a session."""
    store = _get_task_store(request)
    if not store:
        return JSONResponse({"tasks": []})

    if not session_id:
        return JSONResponse({"error": "session_id required"}, status_code=400)

    uid = _current_user_id(request)
    tasks = await store.list_for_session(session_id, user_id=uid)

    return JSONResponse({
        "tasks": [
            {
                "id": t.id,
                "session_id": t.session_id,
                "flow_id": t.flow_id,
                "status": t.status.value,
                "autonomy_level": t.autonomy_level,
                "title": t.title,
                "current_step": t.current_step,
                "total_steps": t.total_steps,
                "tool_calls_made": t.tool_calls_made,
                "error": t.error,
            }
            for t in tasks
        ]
    })


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request) -> JSONResponse:
    """Get a specific task by ID for the authenticated user.

    Returns 404 for tasks that belong to a different tenant — we deliberately
    do not distinguish "not found" from "not yours" in the response so the
    endpoint can't be used as a task-ID oracle across users.
    """
    store = _get_task_store(request)
    if not store:
        return JSONResponse({"error": "Task store not available"}, status_code=503)

    uid = _current_user_id(request)
    task = await store.get(task_id, user_id=uid)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    artifacts: list[dict] = []
    artifact_store = getattr(request.app.state, "artifact_store", None)
    if artifact_store:
        try:
            rows = await artifact_store.list_for_task(task_id, user_id=uid)
            for a in rows:
                aid = a.get("id", "")
                name = (
                    a.get("display_name") or a.get("filename") or "Artifact"
                )
                fmt = a.get("format", "") or a.get("page_type", "")
                artifacts.append({
                    "id": aid,
                    "name": name,
                    "display_name": name,
                    "title": name,
                    "format": fmt,
                    "kind": fmt,
                    "size_bytes": int(a.get("size_bytes") or 0),
                    "page_type": a.get("page_type", ""),
                    "download_url": (
                        a.get("download_url")
                        or (f"/api/artifacts/{aid}/download" if aid else "")
                    ),
                })
        except Exception:
            log.warning("agentic_task_artifact_list_failed",
                        task_id=task_id, exc_info=True)

    total = task.total_steps or 0
    cur = task.current_step or 0
    progress = round((cur / total) * 100) if total > 0 else 0

    return JSONResponse({
        "id": task.id,
        "session_id": task.session_id,
        "flow_id": task.flow_id,
        "status": task.status.value,
        "autonomy_level": task.autonomy_level,
        "title": task.title,
        "plan_md": task.plan_md,
        "current_step": cur,
        "total_steps": total,
        "progress": progress,
        "step_outputs": {str(k): v for k, v in task.step_outputs.items()},
        "tool_calls_made": task.tool_calls_made,
        "error": task.error,
        "artifacts": artifacts,
    })


# ---------------------------------------------------------------------------
# Image candidate picker — /candidates (read) /expand (widen pool) /pick (commit)
# ---------------------------------------------------------------------------


def _resolve_default_backend(request: Request):
    """Pick a ModelBackend for off-stream picker LLM calls.

    Picker endpoints run outside the agentic streaming loop, so we don't
    have a handler caller. Use the registry's default backend — same one
    the agentic handler picks at construction time.
    """
    reg = getattr(request.app.state, "provider_registry", None)
    return getattr(reg, "default_backend", None) if reg else None


def _serialise_candidates(task) -> dict:
    """Read-side projection of image_candidates + slide_image_picks."""
    return {
        "candidates": {str(k): v for k, v in (task.image_candidates or {}).items()},
        "picks": {str(k): v for k, v in (task.slide_image_picks or {}).items()},
    }


def _parsed_slides_for_task(task) -> list[dict]:
    """Pull the deck draft + parse slides the same way the handler does."""
    from augmentum.modes.agentic.handler import (
        _parse_slide_draft,
        _pick_artifact_draft_from_outputs,
        _resolve_artifact_topic,
    )

    draft = _pick_artifact_draft_from_outputs(task)
    if not draft:
        return []
    slides, _ = _parse_slide_draft(
        draft, fallback_title=_resolve_artifact_topic(task) or "Content",
    )
    return slides


async def _resolve_pptx_artifact(request: Request, task_id: str, user_id: str):
    """Find the pptx artifact for this task. Returns (artifact_dict, artifact_store) or (None, None)."""
    store = getattr(request.app.state, "artifact_store", None)
    if not store:
        return None, None
    try:
        rows = await store.list_for_task(task_id, user_id=user_id)
    except Exception as exc:
        log.warning("picker_artifact_list_failed",
                    task_id=task_id, error=str(exc))
        return None, store
    for a in rows:
        if (a.get("format") or "").lower() == "pptx":
            return a, store
    return None, store


async def _rerender_pptx_with_picks(request: Request, task, user_id: str) -> bool:
    """Re-render the deck PPTX after a pick mutation, replacing the file in place.

    Reads source_json (the deck's structured spec), projects the new
    picks onto its slides, calls _render_pptx, and writes the bytes
    back to the same artifact. Returns True on success, False if there
    is no artifact to update yet (deck hasn't been delivered).
    """
    artifact, store = await _resolve_pptx_artifact(request, task.id, user_id)
    if not artifact:
        return False
    try:
        source = json.loads(artifact.get("source_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        log.warning("picker_source_json_unparseable", artifact_id=artifact.get("id"))
        return False
    slides = source.get("slides") or []
    if not slides:
        return False
    from augmentum.tools.artifact_pipeline import _apply_pipeline_image_picks
    from augmentum.tools.artifact_presentation import PresentationTool, _render_pptx

    new_slides = _apply_pipeline_image_picks(
        slides, task.slide_image_picks or {}, task.image_candidates or {},
    )
    # Resolve URLs to local paths exactly as PresentationTool.execute does.
    tool = PresentationTool(store)
    resolved = await tool._resolve_images(new_slides)
    try:
        data = _render_pptx(
            source.get("title", "Presentation"),
            source.get("subtitle", ""),
            source.get("author", ""),
            resolved,
            theme_name=source.get("theme", ""),
        )
    except Exception as exc:
        log.warning("picker_rerender_failed",
                    artifact_id=artifact.get("id"), error=str(exc))
        return False

    # Persist updated source + file.
    new_source = dict(source)
    new_source["slides"] = new_slides
    await store.update_source(
        artifact["id"], json.dumps(new_source), user_id=user_id,
    )
    await store.update_file(artifact["id"], data, user_id=user_id)
    return True


async def _backfill_candidates_if_empty(
    request: Request, task, user_id: str,
) -> None:
    """Silently run the Illustrate Slides crafter for resumed tasks.

    Tasks created before the picker shipped have empty image_candidates.
    On the user's first picker interaction we run the same query crafter
    + image_search loop the agentic flow now runs, so the pool is ready
    by the time the rest of the handler returns. One-time hit per task.
    """
    if task.image_candidates:
        return
    from augmentum.tools.artifact_pipeline import (
        build_backend_pipeline_caller,
        craft_initial_slide_queries,
    )

    slides = _parsed_slides_for_task(task)
    if not slides:
        return
    registry = getattr(request.app.state, "tool_registry", None)
    image_search = registry.resolve("image_search") if registry else None
    if not image_search:
        return
    # Use the backend chat for the LLM call — backfill happens outside an
    # active streaming request so we don't have a handler caller handy.
    backend = _resolve_default_backend(request)
    if backend is None:
        return
    caller = build_backend_pipeline_caller(backend)
    crafted = await craft_initial_slide_queries(slides, caller)
    if not crafted:
        return

    candidates_by_slide: dict[int, list[dict]] = {}
    picks_by_slide: dict[int, dict] = {}
    for slide_data in crafted:
        idx = slide_data["index"]
        query = slide_data.get("query", "")
        if not query:
            continue
        try:
            result = await image_search.execute(
                query=query,
                count=4,
                prefer_charts=bool(slide_data.get("prefer_charts")),
                task_id=task.id,
                session_id=task.session_id,
                user_id=user_id,
            )
        except Exception as exc:
            log.warning("backfill_image_search_failed",
                        slide_index=idx, error=str(exc))
            continue
        if not result.success or not result.metadata:
            continue
        pool: list[dict] = []
        for img in result.metadata.get("images") or []:
            if not isinstance(img, dict):
                continue
            embed_url = img.get("embed_url") or img.get("url") or ""
            if not embed_url:
                continue
            pool.append({
                "candidate_id": uuid.uuid4().hex[:12],
                "query": query,
                "description": slide_data.get("description", ""),
                "prefer_charts": bool(slide_data.get("prefer_charts")),
                "embed_url": embed_url,
                "thumb_url": img.get("thumb_url") or embed_url,
                "source": img.get("source", ""),
                "title": img.get("title", ""),
            })
        if pool:
            candidates_by_slide[idx] = pool
            picks_by_slide[idx] = {"primary": pool[0]["candidate_id"], "additional": []}

    task.image_candidates = candidates_by_slide
    task.slide_image_picks = picks_by_slide
    store = _get_task_store(request)
    if store:
        await store.update_image_candidates(task.id, candidates_by_slide, user_id=user_id)
        await store.update_slide_image_picks(task.id, picks_by_slide, user_id=user_id)


@router.get("/tasks/{task_id}/candidates")
async def get_task_candidates(task_id: str, request: Request) -> JSONResponse:
    """Return the per-slide candidate pool + current picks.

    Tasks created before this substrate shipped have empty pools; we
    silently backfill on first read so the picker UI never sees ghost
    state for resumed decks.
    """
    store = _get_task_store(request)
    if not store:
        return JSONResponse({"error": "Task store not available"}, status_code=503)
    uid = _current_user_id(request)
    task = await store.get(task_id, user_id=uid)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    if not task.image_candidates:
        await _backfill_candidates_if_empty(request, task, uid)
    return JSONResponse(_serialise_candidates(task))


@router.post("/tasks/{task_id}/expand")
async def expand_task_candidates(task_id: str, request: Request) -> JSONResponse:
    """Widen the candidate pool for a slide (or every slide).

    Body: {"scope": "deck"|"slide", "slide_index"?: int, "target_count": int}.

    Forms `target_count - current_pool_size` deliberately diverse queries
    per slide via the expansion crafter, runs image_search, appends the
    new candidates. Returns the updated pool + picks.
    """
    store = _get_task_store(request)
    if not store:
        return JSONResponse({"error": "Task store not available"}, status_code=503)
    uid = _current_user_id(request)
    task = await store.get(task_id, user_id=uid)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        body = {}
    scope = (body.get("scope") or "slide").lower()
    target_count = max(1, min(int(body.get("target_count") or 1), 4))
    slide_index_arg = body.get("slide_index")
    if scope == "slide" and slide_index_arg is None:
        return JSONResponse(
            {"error": "scope=slide requires slide_index"}, status_code=400,
        )

    if not task.image_candidates:
        await _backfill_candidates_if_empty(request, task, uid)

    slides = _parsed_slides_for_task(task)
    if not slides:
        return JSONResponse(
            {"error": "Deck draft not available — task hasn't reached Illustrate Slides yet"},
            status_code=409,
        )

    registry = getattr(request.app.state, "tool_registry", None)
    image_search = registry.resolve("image_search") if registry else None
    if not image_search:
        return JSONResponse({"error": "image_search tool not registered"}, status_code=503)
    backend = _resolve_default_backend(request)
    if backend is None:
        return JSONResponse({"error": "backend not available"}, status_code=503)

    from augmentum.tools.artifact_pipeline import (
        build_backend_pipeline_caller,
        craft_expansion_queries,
    )

    caller = build_backend_pipeline_caller(backend)
    candidates = dict(task.image_candidates)

    target_indices: list[int]
    if scope == "deck":
        # Expand every slide that already has at least one candidate;
        # silent backfill above ensures the pool is populated.
        target_indices = sorted(candidates.keys())
    else:
        try:
            idx = int(slide_index_arg)
        except (TypeError, ValueError):
            return JSONResponse({"error": "slide_index must be an int"}, status_code=400)
        target_indices = [idx]

    for idx in target_indices:
        pool = list(candidates.get(idx) or [])
        existing_queries = [
            {"query": c.get("query", ""), "description": c.get("description", "")}
            for c in pool
        ]
        deficit = target_count - len(pool)
        if deficit <= 0:
            continue
        # Find the slide by 1-based index.
        if idx - 1 < 0 or idx - 1 >= len(slides):
            continue
        slide = slides[idx - 1]
        new_queries = await craft_expansion_queries(
            slide_title=slide.get("title", ""),
            slide_body=slide.get("body", ""),
            existing_queries=existing_queries,
            k=deficit,
            llm_caller=caller,
        )
        for q in new_queries:
            try:
                result = await image_search.execute(
                    query=q["query"],
                    count=4,
                    prefer_charts=bool(q.get("prefer_charts")),
                    task_id=task.id,
                    session_id=task.session_id,
                    user_id=uid,
                )
            except Exception as exc:
                log.warning("expand_image_search_failed",
                            slide_index=idx, query=q["query"], error=str(exc))
                continue
            if not result.success or not result.metadata:
                continue
            for img in result.metadata.get("images") or []:
                if not isinstance(img, dict):
                    continue
                embed_url = img.get("embed_url") or img.get("url") or ""
                if not embed_url:
                    continue
                pool.append({
                    "candidate_id": uuid.uuid4().hex[:12],
                    "query": q["query"],
                    "description": q.get("description", ""),
                    "prefer_charts": bool(q.get("prefer_charts")),
                    "embed_url": embed_url,
                    "thumb_url": img.get("thumb_url") or embed_url,
                    "source": img.get("source", ""),
                    "title": img.get("title", ""),
                })
        candidates[idx] = pool

    task.image_candidates = candidates
    await store.update_image_candidates(task.id, candidates, user_id=uid)
    return JSONResponse(_serialise_candidates(task))


@router.post("/tasks/{task_id}/pick")
async def pick_task_candidate(task_id: str, request: Request) -> JSONResponse:
    """Commit a swap / append / remove against a slide's pick state.

    Body: {"slide_index": int, "candidate_id": str, "mode": "swap"|"append"|"remove"}.

    On success the deck PPTX is re-rendered with the new picks and the
    artifact is replaced in place. Returns the updated picks state and
    a flag indicating whether the re-render succeeded.
    """
    store = _get_task_store(request)
    if not store:
        return JSONResponse({"error": "Task store not available"}, status_code=503)
    uid = _current_user_id(request)
    task = await store.get(task_id, user_id=uid)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        slide_index = int(body.get("slide_index"))
    except (TypeError, ValueError):
        return JSONResponse({"error": "slide_index must be an int"}, status_code=400)
    candidate_id = str(body.get("candidate_id") or "").strip()
    mode = (body.get("mode") or "swap").lower()
    if mode not in ("swap", "append", "remove"):
        return JSONResponse({"error": "mode must be swap/append/remove"}, status_code=400)

    pool = list((task.image_candidates or {}).get(slide_index) or [])
    if mode in ("swap", "append"):
        if not candidate_id or not any(c.get("candidate_id") == candidate_id for c in pool):
            return JSONResponse({"error": "candidate_id not in pool"}, status_code=404)

    picks = dict(task.slide_image_picks or {})
    current = dict(picks.get(slide_index) or {"primary": "", "additional": []})
    if mode == "swap":
        # If the swapped-in candidate was in additional, drop it from there.
        current["additional"] = [
            c for c in (current.get("additional") or []) if c != candidate_id
        ]
        current["primary"] = candidate_id
    elif mode == "append":
        if candidate_id == current.get("primary"):
            return JSONResponse(
                {"error": "candidate is already primary"}, status_code=409,
            )
        additional = list(current.get("additional") or [])
        if candidate_id in additional:
            return JSONResponse(
                {"error": "candidate already appended"}, status_code=409,
            )
        if len(additional) >= 3:
            return JSONResponse(
                {"error": "additional_images already at cap (3)"},
                status_code=409,
            )
        additional.append(candidate_id)
        current["additional"] = additional
    elif mode == "remove":
        # remove from additional list; if candidate matches primary, clear it.
        if current.get("primary") == candidate_id:
            current["primary"] = ""
        current["additional"] = [
            c for c in (current.get("additional") or []) if c != candidate_id
        ]

    picks[slide_index] = current
    task.slide_image_picks = picks
    await store.update_slide_image_picks(task.id, picks, user_id=uid)

    rerendered = await _rerender_pptx_with_picks(request, task, uid)
    return JSONResponse({
        **_serialise_candidates(task),
        "rerendered": rerendered,
    })
