"""Cache management API endpoints served under /api/cache/."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/cache", tags=["cache"])


@router.get("/stats")
async def get_cache_stats(request: Request) -> JSONResponse:
    """Get cache hit/miss statistics and deduplication metrics."""
    prompt_cache = getattr(request.app.state, "prompt_cache", None)
    prefix_cache = getattr(request.app.state, "prefix_cache", None)
    deduplicator = getattr(request.app.state, "request_deduplicator", None)

    stats: dict = {}

    if prompt_cache is not None:
        stats["prompt_cache"] = {
            **prompt_cache.stats.to_dict(),
            "size": prompt_cache.size,
        }

    if prefix_cache is not None:
        stats["prefix_cache"] = prefix_cache.get_stats()

    if deduplicator is not None:
        stats["deduplicator"] = deduplicator.get_stats()

    return JSONResponse(stats)


@router.post("/clear")
async def clear_cache(request: Request) -> JSONResponse:
    """Clear all cached entries."""
    prompt_cache = getattr(request.app.state, "prompt_cache", None)
    prefix_cache = getattr(request.app.state, "prefix_cache", None)

    result: dict = {}

    if prompt_cache is not None:
        count = await prompt_cache.clear()
        result["prompt_cache_cleared"] = count

    if prefix_cache is not None:
        count = prefix_cache.clear()
        result["prefix_cache_cleared"] = count

    return JSONResponse(result)


@router.get("/entries")
async def list_cache_entries(request: Request) -> JSONResponse:
    """List metadata for all cached entries (no response bodies)."""
    prompt_cache = getattr(request.app.state, "prompt_cache", None)
    prefix_cache = getattr(request.app.state, "prefix_cache", None)

    result: dict = {}

    if prompt_cache is not None:
        result["prompt_cache"] = await prompt_cache.get_entries_metadata()

    if prefix_cache is not None:
        result["prefix_cache"] = prefix_cache.get_all_entries()

    return JSONResponse(result)
