"""HTTP endpoints for the reviewable-turn flow.

See ``augmentum/coder/reviews.py`` for the registry and
``augmentum/coder/turn_snapshot.py`` for the pre-write capture layer
that makes the reject path reversible. Flow:

* Agent turn ends → ``CoderHandler._publish_turn_review`` creates a
  :class:`ReviewBundle` and registers it.
* Frontend observes the ``complete`` meta chunk carrying
  ``review_turn_id``, fetches the bundle via ``GET …/pending`` (or the
  single-bundle variant), renders the inline review panel.
* User clicks Accept / Reject / Partial → corresponding endpoint
  applies the decision:

  - **Accept** — disk is already current (writes hit directly per
    snapshot-then-observe); the route just stamps a git commit
    "Turn: <user_message>" and resolves the bundle.
  - **Reject** — route calls ``snapshot.restore()`` with ALL touched
    paths. Each restore writes the captured pre-turn bytes (or deletes
    the file if it didn't exist pre-turn). Non-reversible paths are
    returned in the response so the user sees what partial rollback
    left behind.
  - **Partial** — body carries ``accepted_paths`` and
    ``rejected_paths``; restore runs on rejected, commit covers
    accepted. A shell_exec side-effect from earlier in the turn
    cannot be partially undone — the response surfaces a warning
    when ``tool_calls_made`` suggests mixed shell activity.

All responses are JSON. Multi-tenant scope: turns are filtered by the
caller's user_id; cross-tenant access returns 403.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/coder/reviews", tags=["coder-reviews"])


def _get_registry(request: Request):
    return getattr(request.app.state, "review_registry", None)


def _get_container_manager(request: Request):
    return getattr(request.app.state, "container_manager", None)


def _current_user_id(request: Request) -> str:
    """Scope requests to the caller. Empty in single-tenant-dev setups.

    Matches the ``coder_permission_routes`` convention — the registry's
    ``pending_for("")`` returns every bundle so no-auth deployments
    still work.
    """
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------


@router.get("/pending")
async def list_pending(request: Request) -> JSONResponse:
    """Return all pending review bundles for the current user.

    Frontend polls this (or consumes it once on seeing the turn's
    ``complete`` meta chunk). ``enabled: false`` signals the registry
    isn't wired on this deployment — UI should hide the review panel
    entirely rather than show an empty state.
    """
    registry = _get_registry(request)
    if registry is None:
        return JSONResponse({"enabled": False, "pending": []})

    user_id = _current_user_id(request)
    pending = [b.to_dict() for b in registry.pending_for(user_id)]
    return JSONResponse({"enabled": True, "pending": pending})


@router.get("/{turn_id}")
async def get_one(turn_id: str, request: Request) -> JSONResponse:
    """Fetch a single bundle by turn_id. Frontend uses this when it
    sees a ``review_turn_id`` in a ``complete`` meta chunk — avoids
    the polling round-trip for the common case of "just-finished turn".
    """
    registry = _get_registry(request)
    if registry is None:
        return JSONResponse({"error": "reviews disabled"}, status_code=400)

    bundle = registry.get(turn_id)
    if bundle is None:
        return JSONResponse({"error": "unknown turn"}, status_code=404)

    user_id = _current_user_id(request)
    if user_id and bundle.user_id and bundle.user_id != user_id:
        return JSONResponse({"error": "not owner"}, status_code=403)

    return JSONResponse(bundle.to_dict())


# ---------------------------------------------------------------------------
# Decision endpoints
# ---------------------------------------------------------------------------


def _authorised_bundle_or_error(
    request: Request, turn_id: str,
) -> tuple[object | None, JSONResponse | None]:
    """Common lookup + ownership check; returns (bundle, None) on success
    or (None, error_response). Callers then do the action-specific work."""
    registry = _get_registry(request)
    if registry is None:
        return None, JSONResponse({"error": "reviews disabled"}, status_code=400)

    bundle = registry.get(turn_id)
    if bundle is None:
        return None, JSONResponse({"error": "unknown turn"}, status_code=404)

    user_id = _current_user_id(request)
    if user_id and bundle.user_id and bundle.user_id != user_id:
        return None, JSONResponse({"error": "not owner"}, status_code=403)
    return bundle, None


@router.post("/{turn_id}/accept")
async def accept(turn_id: str, request: Request) -> JSONResponse:
    """User approves the whole turn as-is.

    Disk already reflects the agent's writes (snapshot-then-observe
    — writes go direct). Nothing to apply; we just stamp a git commit
    for the turn and remove the bundle from pending. The commit
    message embeds the user_message so ``git log`` reads as an
    intent-level history rather than per-tool noise.
    """
    bundle, err = _authorised_bundle_or_error(request, turn_id)
    if err is not None:
        return err

    cm = _get_container_manager(request)
    commit_hash: str | None = None
    if cm is not None:
        commit_hash = await _git_commit_turn(
            cm, bundle.workspace_id,
            paths=[f.path for f in bundle.files],
            message=_commit_message(bundle),
        )

    registry = _get_registry(request)
    registry.resolve(turn_id, "accepted")

    return JSONResponse({
        "status":    "accepted",
        "turn_id":   turn_id,
        "commit":    commit_hash,
        "files":     [f.path for f in bundle.files],
    })


@router.post("/{turn_id}/reject")
async def reject(turn_id: str, request: Request) -> JSONResponse:
    """User rejects the whole turn.

    Restore every touched path from its pre-turn snapshot. Paths
    flagged ``reversible=False`` (snapshot skipped due to read error
    or size limit) come back in the ``failed_paths`` response so the
    user knows partial rollback happened. For Sprint 1 we do NOT run
    a git revert on top — the restore writes ARE the rollback, and
    tracking a separate git revert would double-commit. Users can
    always ``git reset HEAD~1`` manually if they want to erase the
    accept-commit history too.
    """
    bundle, err = _authorised_bundle_or_error(request, turn_id)
    if err is not None:
        return err

    paths = [f.path for f in bundle.files]
    failed = await bundle.snapshot.restore(paths)
    restored = [p for p in paths if p not in failed]

    registry = _get_registry(request)
    registry.resolve(turn_id, "rejected")

    return JSONResponse({
        "status":         "rejected",
        "turn_id":        turn_id,
        "restored_paths": restored,
        "failed_paths":   failed,
    })


@router.post("/{turn_id}/partial")
async def partial(turn_id: str, request: Request) -> JSONResponse:
    """Per-file decision — body is ``{accepted_paths, rejected_paths}``.

    Runs restore over rejected paths and git-commit over accepted
    paths. Paths present in neither array are silently accepted (we
    default to keep-what-you-didn't-reject — the opposite would be
    lossy). Failed restores surface in ``failed_paths`` and the
    caller can decide to re-try or surface to the user.
    """
    bundle, err = _authorised_bundle_or_error(request, turn_id)
    if err is not None:
        return err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    accepted_paths: list[str] = [
        str(p) for p in (body.get("accepted_paths") or [])
    ]
    rejected_paths: list[str] = [
        str(p) for p in (body.get("rejected_paths") or [])
    ]

    known = {f.path for f in bundle.files}
    # Silently coerce: ignore paths not in the bundle (stale UI state),
    # treat unmentioned bundle paths as accepted (see docstring).
    accepted_set = known & set(accepted_paths)
    rejected_set = known & set(rejected_paths)
    implicitly_accepted = known - accepted_set - rejected_set
    accepted_final = accepted_set | implicitly_accepted

    failed: list[str] = []
    if rejected_set:
        failed = await bundle.snapshot.restore(sorted(rejected_set))

    commit_hash: str | None = None
    cm = _get_container_manager(request)
    if cm is not None and accepted_final:
        commit_hash = await _git_commit_turn(
            cm, bundle.workspace_id,
            paths=sorted(accepted_final),
            message=_commit_message(bundle, partial=True),
        )

    registry = _get_registry(request)
    registry.resolve(turn_id, "partial")

    return JSONResponse({
        "status":         "partial",
        "turn_id":        turn_id,
        "accepted_paths": sorted(accepted_final),
        "rejected_paths": sorted(rejected_set),
        "failed_paths":   failed,
        "commit":         commit_hash,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _commit_message(bundle, *, partial: bool = False) -> str:
    """Build a human-readable commit message for an accepted turn.

    Format: ``Turn: <trimmed user message>`` so ``git log`` reads as
    a history of user intents rather than per-tool noise. Partial
    accepts get a ``[partial]`` marker so log readers know some files
    were rejected out of this turn.
    """
    msg = (bundle.user_message or "Agent turn").strip().replace("\n", " ")
    if len(msg) > 80:
        msg = msg[:77] + "..."
    prefix = "Turn (partial)" if partial else "Turn"
    return f"{prefix}: {msg}"


async def _git_commit_turn(
    container_manager,
    workspace_id: str,
    *,
    paths: list[str],
    message: str,
) -> str | None:
    """``git add <paths> && git commit -m <message>`` inside the workspace.

    Returns the short commit hash on success, None on failure (the
    workspace might not be a git repo, or there may be nothing to
    commit — both are benign). Logged rather than raised.
    """
    if not paths:
        return None
    try:
        # git add with each path individually avoids shell-quoting
        # headaches on unusual names. Runs from /workspace so relative
        # paths in the file bundle still resolve.
        for path in paths:
            await container_manager.run_command(
                workspace_id,
                ["sh", "-c", f"cd /workspace && git add {path!r}"],
            )
        out = await container_manager.run_command(
            workspace_id,
            ["sh", "-c",
             f"cd /workspace && git commit -m {message!r} --allow-empty-message "
             "&& git rev-parse --short HEAD"],
        )
        # rev-parse output is the last non-empty line.
        lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
        return lines[-1] if lines else None
    except Exception as exc:
        log.warning(
            "coder.review_commit_failed",
            workspace_id=workspace_id, error=str(exc),
        )
        return None
