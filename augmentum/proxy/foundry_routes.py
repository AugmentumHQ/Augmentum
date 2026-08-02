"""Game foundry trigger + status endpoints.

* ``POST /api/foundry/runs``       — start a foundry run (background task)
* ``GET  /api/foundry/runs/{id}``  — poll status + per-pass scores

The run drives the closed loop (``run_foundry``): generate → [3d: Blender
asset + visual-verify] → deploy → autonomous play → score → defect relay →
regenerate. It executes as an in-process background task (like the game-agent
orchestrator) with results held in ``app.state.foundry_runs``.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any, Literal, cast

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from augmentum.coder.foundry.contract import GameBuildSpec
from augmentum.coder.foundry.events import FoundryEventBus
from augmentum.coder.foundry.loop import run_foundry
from augmentum.coder.foundry.wire import wire_default_stages
from augmentum.config import settings

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/foundry", tags=["foundry"])


def _runs(request: Request) -> dict[str, dict]:
    store = getattr(request.app.state, "foundry_runs", None)
    if store is None:
        store = cast(dict[str, dict], {})
        request.app.state.foundry_runs = store
    return cast(dict[str, dict], store)


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return getattr(user, "id", "") or ""


class FoundryRunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=120)
    concept: str = Field(..., min_length=1, max_length=4000)
    objective: str = Field(..., min_length=1, max_length=2048)
    slug: str = Field("", max_length=64)
    dimension: Literal["2d", "3d"] = "2d"
    passes: int = Field(2, ge=1, le=6)
    play_seconds: int = Field(90, ge=15, le=600)
    controls: dict[str, str] | None = None


@router.post("/runs")
async def start_run(body: FoundryRunBody, request: Request) -> Any:
    user_id = _user_id(request)
    if not user_id:
        return JSONResponse({"error": "authentication required"}, status_code=401)

    slug = (body.slug or _slugify(body.title)).strip("-") or "game"
    spec = GameBuildSpec(
        slug=slug, title=body.title, concept=body.concept,
        objective=body.objective, dimension=body.dimension,
        controls=body.controls or GameBuildSpec.__dataclass_fields__[
            "controls"].default_factory(),
    )
    run_id = f"f_{uuid.uuid4().hex[:10]}"
    bus = FoundryEventBus()
    # The theater feed reaches BOTH the loop (pass/score events) and the play
    # stage (session + live decisions) through the same emit.
    try:
        stages_with_events = wire_default_stages(
            request, user_id=user_id, workspace_id=body.workspace_id,
            model=body.model,
            verify_model=(getattr(settings, "coder_visual_verify_model", "") or ""),
            on_event=bus.emit,
        )
    except Exception as exc:
        log.warning("foundry_wire_failed", error=str(exc))
        return JSONResponse({"error": f"could not wire foundry: {exc}"}, status_code=500)

    store = _runs(request)
    store[run_id] = {"run_id": run_id, "status": "running", "slug": slug,
                     "dimension": body.dimension, "passes": [], "_bus": bus}

    async def _drive() -> None:
        try:
            result = await run_foundry(
                spec,
                generate=stages_with_events["generate"], play=stages_with_events["play"],
                asset=stages_with_events["asset"], verify=stages_with_events["verify"],
                passes=body.passes, play_seconds=body.play_seconds,
                on_event=bus.emit,
            )
            store[run_id].update({
                "status": "completed", "improved": result.improved,
                "summary": result.summary(),
                "passes": [asdict(p) for p in result.passes],
            })
        except Exception as exc:  # a run failure is recorded, never crashes the app
            log.warning("foundry_run_failed", run_id=run_id, error=str(exc), exc_info=True)
            bus.emit("error", message=str(exc))
            store[run_id].update({"status": "error", "error": str(exc)})
        finally:
            bus.close()

    task = asyncio.create_task(_drive(), name=f"foundry-{run_id}")
    # Keep a handle so the task isn't garbage-collected mid-flight.
    store[run_id]["_task"] = task
    return {"run_id": run_id, "status": "running"}


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request) -> Any:
    """SSE stream of the run's live theater events (backlog, then live).

    Reconnect-safe: a subscriber joining mid-run (or after a page reload)
    replays the backlog first, then tails live until the run closes.
    """
    rec = _runs(request).get(run_id)
    bus = rec.get("_bus") if rec else None
    if not isinstance(bus, FoundryEventBus):
        return JSONResponse({"error": "no such foundry run"}, status_code=404)

    async def _stream() -> AsyncIterator[bytes]:
        async for event in bus.subscribe():
            yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> Any:
    rec = _runs(request).get(run_id)
    if rec is None:
        return JSONResponse({"error": "no such foundry run"}, status_code=404)
    # Drop the non-serializable internals (task handle + event bus).
    return {k: v for k, v in rec.items() if k not in ("_task", "_bus")}


def _slugify(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("-")
    return "".join(out)[:64] or "game"
