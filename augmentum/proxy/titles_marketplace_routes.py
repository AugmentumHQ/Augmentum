"""AXF Marketplace routes -- browse catalog + one-click install.

Distinct from ``augmentum/proxy/marketplace_routes.py`` (which is the
Docker provider marketplace -- managing service containers like
ollama, llamacpp). This file is the *title* marketplace: a curated
catalog of playable things (games, streamed bundles, web apps).
Mounted at ``/api/titles/marketplace`` so it lives under the AXF
namespace.

The catalog is server-level data; reads are public to authenticated
users. Installs delegate through ``TitleService`` so the resulting
artifact uses the same orchestration as any other title import.

Master toggle: ``marketplace_enabled``. When false, every endpoint
returns 503 -- the catalog table is still loaded so flipping the
toggle on doesn't require a restart, but the surface is muted.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.config import settings
from augmentum.titles import TitleServiceError
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/titles/marketplace", tags=["titles-marketplace"])


# ── helpers ──────────────────────────────────────────────────────────


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _gate(request: Request) -> JSONResponse | None:
    if not getattr(settings, "marketplace_enabled", False):
        return JSONResponse(
            {"error": "Marketplace is disabled"}, status_code=503,
        )
    if getattr(request.app.state, "marketplace_store", None) is None:
        return JSONResponse(
            {"error": "Marketplace not available"}, status_code=503,
        )
    return None


def _store(request: Request):
    return getattr(request.app.state, "marketplace_store", None)


def _title_service(request: Request):
    return getattr(request.app.state, "title_service", None)


# ── Routes ───────────────────────────────────────────────────────────


@router.get("/")
async def list_listings(
    request: Request,
    kind: str | None = None,
    publisher: str | None = None,
    limit: int = 200,
) -> JSONResponse:
    """List active marketplace listings."""
    if (gate := _gate(request)) is not None:
        return gate
    if not _user_id(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    listings = await _store(request).list_active(
        kind=kind, publisher=publisher,
        limit=max(1, min(500, int(limit))),
    )
    return JSONResponse({
        "listings": [l.to_dict() for l in listings],
    })


@router.get("/{listing_id}")
async def get_listing(request: Request, listing_id: str) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    if not _user_id(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    listing = await _store(request).get(listing_id)
    if not listing or listing.delisted_at:
        return JSONResponse({"error": "Listing not found"}, status_code=404)
    return JSONResponse({"listing": listing.to_dict()})


@router.post("/{listing_id}/install")
async def install_listing(request: Request, listing_id: str) -> JSONResponse:
    """Install a marketplace listing into the user's library.

    Delegates to ``TitleService.import_title`` with ``source_id =
    "marketplace"`` so the underlying Source the listing names in
    ``install_via`` does the actual work. Returns the new title
    manifest, same shape as ``POST /api/titles/``.
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    svc = _title_service(request)
    if svc is None:
        return JSONResponse(
            {"error": "Title service unavailable"}, status_code=503,
        )
    try:
        manifest = await svc.import_title(
            user_id=uid,
            source_id="marketplace",
            manifest_data={"listing_id": listing_id},
        )
    except TitleServiceError as exc:
        msg = str(exc)
        if "not found" in msg.lower():
            return JSONResponse({"error": msg}, status_code=404)
        return JSONResponse({"error": msg}, status_code=400)
    return JSONResponse({"title": manifest.to_dict()}, status_code=201)
