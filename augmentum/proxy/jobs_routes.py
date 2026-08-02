"""Background-job status + control endpoints.

Reads the user's jobs, posts cancellations. Job *creation* is not
exposed here — consumers (e.g. a future transcription route) enqueue
jobs from their own domain endpoints so they can validate domain
payloads before writing to the queue.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _get_store(request: Request):
    return getattr(request.app.state, "jobs_store", None)


@router.get("/")
async def list_jobs(
    request: Request,
    status: str | None = None,
    type: str | None = None,
    limit: int = 100,
) -> JSONResponse:
    """List the authenticated user's jobs, newest first.

    ``status`` filters to one of pending/running/completed/failed/cancelled.
    ``type`` filters to a single job_type. Both optional.
    """
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Jobs store not available"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    limit = max(1, min(500, int(limit)))
    jobs = await store.list_for_user(
        user_id=uid, status=status, job_type=type, limit=limit,
    )
    return JSONResponse({"jobs": jobs})


@router.get("/{job_id}")
async def get_job(request: Request, job_id: str) -> JSONResponse:
    """Single-job status for polling."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Jobs store not available"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    job = await store.get(job_id, user_id=uid)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse(job)


@router.post("/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str) -> JSONResponse:
    """Request cancellation.

    Pending jobs terminate immediately; running jobs flip the cancel
    flag and wind down cooperatively — returns 200 in both cases. Jobs
    already in a terminal state return 409.
    """
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Jobs store not available"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    ok = await store.request_cancel(job_id, user_id=uid)
    if not ok:
        # Either missing-or-not-yours, or already terminal. Fetch to
        # distinguish — the 404 vs 409 split matters for the UI.
        existing = await store.get(job_id, user_id=uid)
        if not existing:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        return JSONResponse(
            {"error": "Job already in terminal state", "status": existing["status"]},
            status_code=409,
        )
    return JSONResponse({"ok": True})
