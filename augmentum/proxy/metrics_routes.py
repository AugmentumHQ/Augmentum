"""Prometheus-compatible metrics endpoint."""
from __future__ import annotations

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def prometheus_metrics(request: Request):
    """Prometheus text exposition format."""
    from augmentum.utils.metrics import REGISTRY

    # Update dynamic gauges before rendering
    app = request.app

    # Image queue depth
    if hasattr(app.state, "generation_queue") and app.state.generation_queue:
        from augmentum.utils.metrics import IMAGE_QUEUE_DEPTH
        IMAGE_QUEUE_DEPTH.set(app.state.generation_queue.queue_size)

    body = REGISTRY.render()
    return Response(
        content=body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
