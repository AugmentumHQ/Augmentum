"""REST + WebSocket routes for Coder mode (workspace containers and terminal)."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import shlex
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
import websockets
from fastapi import APIRouter, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.websockets import WebSocketState

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["coder"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CreateWorkspaceRequest(BaseModel):
    name: str
    git_url: str | None = Field(default=None)
    git_branch: str | None = Field(default=None)
    base_image: str = Field(default="augmentum-workspace")
    # Validated at the schema layer against the profile catalog in
    # ``augmentum/coder/profiles.py``. Using a field_validator (instead of
    # ``Literal[...]``) so adding a profile is a single data-model change,
    # while still rejecting unknown values with a Pydantic 422 response.
    tooling_profile: str = Field(default="browser")

    @field_validator("tooling_profile")
    @classmethod
    def _validate_tooling_profile(cls, v: str) -> str:
        from augmentum.coder import profiles as _profiles
        if not _profiles.has_profile(v):
            valid = ", ".join(p.id for p in _profiles.all_profiles())
            raise ValueError(
                f"unknown tooling_profile {v!r}; expected one of: {valid}"
            )
        return v
    # Opt-in: publish the common dev-server ports (3000, 5173, 8000,
    # etc.) to 127.0.0.1 on the host so dev servers running inside the
    # workspace are reachable from the user's browser. OFF by default
    # because not every workspace wants LAN-accessible ports, and port
    # publishing adds to attack surface even on loopback (shared-host
    # side-channels).
    publish_ports: bool = Field(default=False)
    # Workspace type — ``regular`` (default) is a standard coder
    # workspace; ``bug_finder`` is a workspace dedicated to autonomous
    # audit runs (surfaces an extra Bug Finder tab in the workbench).
    # Chosen once at creation; switch is done by creating a different
    # workspace.
    kind: Literal["regular", "bug_finder"] = Field(default="regular")


class SafeguardsRequest(BaseModel):
    enabled: bool


class AlwaysOnRequest(BaseModel):
    # True exempts the workspace from the idle reaper; False opts in.
    # See migration 211 + ``ContainerManager.sweep_idle`` for the
    # full lifecycle contract.
    always_on: bool


class BackgroundRunRequest(BaseModel):
    """Queue a headless coder mission (see jobs/handlers/coder_background_run.py).

    ``model`` is REQUIRED — the user picks it at queue time (the UI sends
    its currently-selected coder model). Never auto-selected server-side.
    """

    prompt: str
    model: str
    coder_strategy: str = ""


class ResearchRunRequest(BaseModel):
    """Queue an autonomous improvement loop (see
    jobs/handlers/coder_research_run.py).

    ``model`` and ``direction`` are REQUIRED — the user picks the model
    and states what "better" means at queue time. Never assumed
    server-side. ``intentions`` may be empty when the workspace has an
    ``OBJECTIVES.md`` at its root (the handler falls back to it).
    """

    model: str
    objective_command: str
    direction: Literal["minimize", "maximize"]
    intentions: str = ""
    experiments: int = Field(default=5, ge=1, le=50)
    objective_timeout: float = Field(default=300.0, ge=5.0, le=1800.0)
    turn_max_seconds: float = Field(default=1200.0, ge=60.0, le=3600.0)
    min_delta: float = Field(default=0.0, ge=0.0)
    coder_strategy: str = ""


class RenameRequest(BaseModel):
    # Purely cosmetic — the docker container name is derived from the
    # workspace_id, so a rename doesn't churn Docker.
    name: str = Field(..., min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        s = (v or "").strip()
        if not s:
            raise ValueError("Name cannot be empty or whitespace-only")
        return s


class BugFinderVerifierRequest(BaseModel):
    # Empty string clears the override (single-model self-verification
    # default kicks back in).
    verifier_model: str = ""


class ProfileEntryRequest(BaseModel):
    action: Literal["upsert", "delete"] = "upsert"
    category: str
    key: str
    value: Any = None
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    workspace_scoped: bool = True


class ProfileUpdateRequest(BaseModel):
    entries: list[ProfileEntryRequest] = Field(default_factory=list)


class ServiceStartRequest(BaseModel):
    command: str
    name: str = ""
    cwd: str = "/workspace"
    env: dict[str, str] = Field(default_factory=dict)
    ports: list[int] = Field(default_factory=list)


class ServiceProbeRequest(BaseModel):
    service_id: str = ""
    url: str = ""
    port: int | None = None
    timeout: float = 5.0


class ConversationCompactRequest(BaseModel):
    messages: list[dict[str, Any]] | None = None
    keep_recent: int = Field(default=12, ge=4, le=40)
    force: bool = True
    model: str = ""


def _preview_proxy_path(workspace_id: str, container_port: int) -> str:
    """Build the same-origin proxy path the iframe loads for a dev port.

    Routed via Augmentum so the iframe is always same-origin (CSP 'self'
    permits it) and the dev server is reachable from any device that can
    reach Augmentum — phone-on-LAN, tablet, etc. — without punching holes
    in host port publishing.
    """
    return f"/api/coder/preview/{workspace_id}/{int(container_port)}/"


async def _maybe_sync_workspace_gate(
    mgr, workspace_id: str, ports: list[dict],
) -> list[str]:
    """If the workspace is LAN-accessible and gate_domain is configured,
    apply (or remove) Caddy gate snippets for listening ports.

    Returns a list of gate URLs for listening ports (empty when gate is
    not configured or workspace is not LAN-accessible).
    """
    from augmentum.coder.containers import _workspace_slug
    from augmentum.providers.caddy_front_door import (
        GATE_LISTEN_PORT,
        apply_workspace_gate,
        gate_domain,
        remove_workspace_gate,
    )

    gd = gate_domain()
    if not gd:
        return []

    try:
        info = await mgr._get_workspace(workspace_id)
    except Exception:
        return []

    listening = [
        p for p in ports
        if p.get("listening") and int(p.get("host_port") or 0) > 0
    ]
    slug = _workspace_slug(info.name, workspace_id)

    if not info.lan_accessible or not listening:
        try:
            await remove_workspace_gate(mgr._docker, slug)
        except Exception:
            log.debug("workspace_gate_cleanup_skipped", slug=slug, exc_info=True)
        return []

    first = listening[0]
    host_port = int(first["host_port"])
    try:
        ok = await apply_workspace_gate(mgr._docker, slug, host_port)
    except Exception:
        log.warning("workspace_gate_sync_failed", slug=slug, exc_info=True)
        return []

    if not ok:
        log.warning(
            "workspace_gate_not_live",
            slug=slug, host_port=host_port,
        )
        return []

    return [f"https://{slug}.{gd}:{GATE_LISTEN_PORT}"]


def _build_preview_summary(ports: list[dict], workspace_id: str) -> dict:
    """Summarize raw port rows into a small preview-state contract."""
    published = [p for p in ports if int(p.get("host_port") or 0) > 0]
    ready = [p for p in published if bool(p.get("listening"))]
    urls = [
        _preview_proxy_path(workspace_id, int(p["container_port"])) for p in ready
    ]
    if ready:
        state = "ready"
    elif published:
        state = "published_idle"
    else:
        state = "not_published"
    return {
        "state": state,
        "published": bool(published),
        "ready": bool(ready),
        "ready_count": len(ready),
        "primary_url": urls[0] if urls else None,
        "urls": urls,
    }


class WriteFileRequest(BaseModel):
    path: str
    content: str
    # If True, create a user-attributed checkpoint after writing so the
    # edit shows up in the checkpoint timeline as "User edit: <path>"
    # instead of being silently bundled into the next agent commit.
    # Frontend sets this for editor saves; skips it for .keep stubs
    # produced by new-folder creation.
    checkpoint: bool = False


class GitTokenRequest(BaseModel):
    host: str
    token: str
    username: str = Field(default="oauth2")


class GitRemoteRequest(BaseModel):
    url: str


class GitCommitRequest(BaseModel):
    message: str = Field(default="Update from Augmentum")


class GitStageRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)


class GitCheckoutRequest(BaseModel):
    branch: str


class GitBranchRequest(BaseModel):
    name: str
    # Source branch to fork from. Empty = HEAD. Common alternatives
    # are "main", "develop", or an explicit ref.
    from_: str = Field(default="", alias="from")

    class Config:
        populate_by_name = True


# Inspector panel — editable objective + observations + state aggregate.
# See docs/superpowers/specs/2026-05-28-coder-inspector-design.md.


class InspectorObjectiveRequest(BaseModel):
    content: str = Field(..., min_length=1)
    # Optional optimistic-concurrency token (file mtime at last read).
    # When provided and the current mtime differs, server returns 409 so
    # the inspector can surface a conflict UI instead of silently
    # overwriting a model edit that landed during the user's typing.
    if_mtime_unchanged: float | None = Field(default=None)


class InspectorObservationCreateRequest(BaseModel):
    category: str = Field(..., min_length=1, max_length=32)
    fact: str = Field(..., min_length=1, max_length=1024)
    confidence: str = Field(default="user_asserted")


class InspectorObservationUpdateRequest(BaseModel):
    category: str | None = Field(default=None, max_length=32)
    fact: str | None = Field(default=None, max_length=1024)
    confidence: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


def _get_manager(request: Request) -> object | None:
    return getattr(request.app.state, "container_manager", None)


def _get_conn(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    return getattr(getattr(sm, "backend", None), "conn", None) if sm else None


async def _owns_workspace(request: Request, workspace_id: str) -> bool:
    """Return True if the authenticated user owns ``workspace_id``.

    Used as a thin scoping layer around WorkspaceManager: the manager's own
    SQL was never rewritten to accept user_id, so the routes gate access
    here before delegating. Matches the "check at the edge" pattern used
    elsewhere when a manager is expensive to refactor.

    Side effect: on successful ownership match, bumps the workspace's
    ``last_active`` via ``ContainerManager.mark_active`` so the idle
    reaper (``sweep_idle`` in containers.py) doesn't reap a workspace
    whose user is actively polling its routes. The mark is debounced
    at the manager level to ~30s, so high-frequency endpoints
    (inspector-state at ~5s) don't write the DB on every hit. Failure
    to mark is logged-and-swallowed by the manager; never affects the
    ownership decision.
    """
    uid = _user_id(request)
    if not uid:
        return False
    conn = _get_conn(request)
    if conn is None:
        return False
    try:
        cursor = await conn.execute(
            "SELECT 1 FROM project_checkouts WHERE id = ? AND user_id = ? LIMIT 1",
            (workspace_id, uid),
        )
        row = await cursor.fetchone()
        if row is None:
            return False
    except Exception:
        return False
    mgr = _get_manager(request)
    if mgr is not None:
        try:
            await mgr.mark_active(workspace_id)
        except Exception:
            # Activity tracking is best-effort; the reaper is a
            # nice-to-have, not load-bearing on ownership. mark_active
            # already logs its own DB-write failures — this catches an
            # unexpected raise from the call as a whole, which we still
            # want visible rather than fully silent.
            log.warning(
                "owns_workspace_mark_active_failed",
                workspace_id=workspace_id, exc_info=True,
            )
    return True


def _unavailable() -> JSONResponse:
    return JSONResponse(
        {"error": "Container manager not available (Docker not configured)"},
        status_code=503,
    )


async def _get_git_store(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    if conn is None:
        return None
    from augmentum.coder.git_credentials import GitTokenStore
    return GitTokenStore(conn)


async def _get_coder_persistence(request: Request):
    conn = _get_conn(request)
    if conn is None:
        return None
    from augmentum.state.coder_persistence import CoderPersistence
    return CoderPersistence(conn)


async def _get_profile_store(request: Request):
    conn = _get_conn(request)
    if conn is None:
        return None
    from augmentum.coder.profile import CoderProfileStore
    return CoderProfileStore(conn)


async def _get_service_store(request: Request):
    conn = _get_conn(request)
    if conn is None:
        return None
    from augmentum.coder.services import CoderServiceStore
    return CoderServiceStore(conn)


async def _get_ledger_store(request: Request):
    conn = _get_conn(request)
    if conn is None:
        return None
    from augmentum.coder.ledger import CoderTurnLedgerStore
    return CoderTurnLedgerStore(conn)


async def _workspace_owner_id(request: Request, workspace_id: str) -> str:
    """Resolve the owning user for ``workspace_id``."""
    conn = _get_conn(request)
    if conn is None:
        return ""
    try:
        cursor = await conn.execute(
            "SELECT user_id FROM project_checkouts WHERE id = ? LIMIT 1",
            (workspace_id,),
        )
        row = await cursor.fetchone()
        return (row[0] or "") if row else ""
    except Exception:
        return ""


_DOCKER_SUBNETS = [
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
]


def _is_docker_internal(client_ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(client_ip)
        return any(addr in subnet for subnet in _DOCKER_SUBNETS)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Workspace endpoints
# ---------------------------------------------------------------------------


@router.get("/api/coder/tooling-profiles")
async def list_tooling_profiles() -> JSONResponse:
    """Profile catalog for the workspace creation modal.

    Returns the profiles in declaration order. The UI uses this as the
    single source of truth — the JS dropdown is built from this response,
    not a parallel JS array. Adding/removing a profile in
    ``augmentum/coder/profiles.py`` propagates to the UI automatically.
    """
    from augmentum.coder import profiles as _profiles

    out = []
    for prof in _profiles.all_profiles():
        out.append({
            "id": prof.id,
            "label": prof.label,
            "description": prof.description,
            "inherits": prof.inherits,
            "est_size_mb": prof.est_size_mb,
            "est_setup_sec": prof.est_setup_sec,
            "network_policy": prof.network_policy,
            "notice": prof.notice,
        })
    return JSONResponse({"profiles": out})


@router.get("/api/coder/workspaces")
async def list_workspaces(request: Request) -> JSONResponse:
    """List the authenticated user's workspaces."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        all_workspaces = await mgr.list_workspaces()
        # Intersect manager output with user-owned IDs from the DB.
        conn = _get_conn(request)
        owned_ids: set[str] = set()
        if conn is not None:
            cursor = await conn.execute(
                "SELECT id FROM project_checkouts WHERE user_id = ?",
                (uid,),
            )
            owned_ids = {row[0] for row in await cursor.fetchall()}
        workspaces = [w for w in all_workspaces if w.id in owned_ids]
        return JSONResponse({"workspaces": [w.__dict__ for w in workspaces]})
    except Exception as exc:
        log.warning("list_workspaces_failed", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/workspaces")
async def create_workspace(body: CreateWorkspaceRequest, request: Request) -> JSONResponse:
    """Create a new workspace container owned by the authenticated user."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        # user_id is threaded all the way to ContainerManager._persist_workspace
        # so the INSERT stamps ownership atomically. The previous post-INSERT
        # UPDATE workaround had a tiny race window where a concurrent reader
        # could see a NULL-user_id row; this closes it.
        info = await mgr.create_workspace(
            name=body.name,
            base_image=body.base_image,
            git_url=body.git_url,
            git_branch=body.git_branch,
            publish_ports=body.publish_ports,
            tooling_profile=body.tooling_profile,
            user_id=uid,
            kind=body.kind,
        )
        return JSONResponse(info.__dict__, status_code=201)
    except Exception as exc:
        log.warning("create_workspace_failed", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/workspaces/self-test")
async def create_self_test_workspace(request: Request) -> JSONResponse:
    """Create a workspace seeded with the live host augmentum source.

    Developer convenience for testing coder behavior on Augmentum's own
    codebase. Requires the host source to be bind-mounted read-only at
    ``/host-augmentum-src`` (see compose.yaml). The workspace receives a
    fresh copy of every tracked file plus every untracked-but-not-
    ignored file — capturing the current working-tree state, including
    uncommitted edits, which a ``git_url`` clone could not see.

    Safe to call repeatedly; each call creates an independent workspace.
    Returns 412 with a configuration hint if the mount is missing so a
    plain misconfiguration doesn't masquerade as a 500.
    """
    import os

    host_source = "/host-augmentum-src"

    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not os.path.isdir(host_source):
        return JSONResponse(
            {"error": (
                f"Host augmentum source not mounted at {host_source}. "
                "Add the read-only bind mount under the augmentum service "
                "in compose.yaml and restart the stack."
            )},
            status_code=412,
        )
    if not os.path.isfile(os.path.join(host_source, "compose.yaml")):
        return JSONResponse(
            {"error": (
                f"{host_source} does not look like the augmentum repo "
                "(no compose.yaml at the root). Check the bind-mount source."
            )},
            status_code=412,
        )

    try:
        rel_paths = await _list_host_source_files(host_source)
        if not rel_paths:
            return JSONResponse(
                {"error": (
                    f"Source walk under {host_source} produced no files. "
                    "Check that the bind mount resolves to a populated tree."
                )},
                status_code=500,
            )

        info = await mgr.create_workspace(
            name="augmentum-self-test",
            tooling_profile="standard",
            user_id=uid,
            kind="regular",
        )

        # Wait for the workspace setup script (which runs the bare-repo
        # clone into /workspace) to finish before seeding — otherwise
        # put_archive races the clone and one side loses. The setup
        # script touches /workspace/.augmentum/ready as its last step.
        ready_deadline = 180.0
        polled = 0.0
        ready = False
        while polled < ready_deadline:
            try:
                out = await mgr._run_command(
                    info.id,
                    ["sh", "-c",
                     "test -f /workspace/.augmentum/ready && echo READY"],
                    timeout=5.0,
                )
                if "READY" in out:
                    ready = True
                    break
            except Exception:
                pass
            await asyncio.sleep(1.0)
            polled += 1.0
        if not ready:
            return JSONResponse(
                {"error": (
                    f"Workspace {info.id} did not become ready within "
                    f"{int(ready_deadline)}s; refusing to seed into a "
                    "half-initialized container."
                )},
                status_code=504,
            )

        seeded = await mgr.seed_from_host_paths(
            workspace_id=info.id,
            dest_path="/workspace",
            host_root=host_source,
            rel_paths=rel_paths,
        )
        return JSONResponse(
            {**info.__dict__, "seeded_files": seeded},
            status_code=201,
        )
    except Exception as exc:
        log.warning("self_test_workspace_failed", error=str(exc), exc_info=True)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _list_host_source_files(host_root: str) -> list[str]:
    """Return relative paths to seed via a directory walk with hard-coded
    excludes approximating ``.gitignore``.

    The augmentum container image doesn't ship ``git``, so we can't lean
    on ``git ls-files`` to get a precise gitignore-aware listing. The
    exclude list below covers the heavy-hitters: caches, virtualenvs,
    build outputs, the ``data/`` tree (which holds user-generated SQLite
    blobs and would balloon the seed), and the ``.git`` directory
    itself — a fresh seed without history is enough for a test
    workspace, and the alternative is shipping a multi-hundred-MB pack
    file over Docker's put-archive API.
    """
    import os

    exclude_dirs = {
        "__pycache__", ".venv", "venv", "env",
        "dist", "build", ".eggs",
        "node_modules", ".vscode", ".idea",
        "data", ".data", "htmlcov", ".git",
        "services", "forensics", "audit", "_pilot_mirror",
        "tmp", "refvrm",
        ".pytest_cache", ".ruff_cache", ".mypy_cache",
        ".augmentum-dev-cache",
        ".next", "target",
    }
    exclude_prefixes = (
        "ui/lib/emulator-js/data/",
        "poses/external/",
        "poses/affordance-screenshots/",
        "poses/synthesized-screenshots/",
        "poses/idle-frames/",
        "poses/vrma/",
        "ui/mockups/",
    )
    exclude_suffixes = (".pyc", ".pyo", ".swp", ".swo", ".db", ".sqlite")

    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(host_root):
        rel_dir = os.path.relpath(dirpath, host_root).replace(os.sep, "/")
        if rel_dir == ".":
            rel_dir = ""
        kept: list[str] = []
        for d in dirnames:
            if d in exclude_dirs:
                continue
            candidate = f"{rel_dir}/{d}/" if rel_dir else f"{d}/"
            if any(candidate.startswith(p) for p in exclude_prefixes):
                continue
            kept.append(d)
        dirnames[:] = kept
        for fn in filenames:
            if fn.endswith(exclude_suffixes):
                continue
            rel = f"{rel_dir}/{fn}" if rel_dir else fn
            paths.append(rel)
    return paths


@router.put("/api/coder/workspaces/{workspace_id}/bug-finder-verifier")
async def set_workspace_bug_finder_verifier(
    workspace_id: str, body: BugFinderVerifierRequest, request: Request,
) -> JSONResponse:
    """Set or clear the Bug Finder verifier-model override for one workspace.

    Empty string clears the override — the workspace falls back to
    single-model self-verification (the user's selected primary model
    used for all four roles). When set, that model is used for the
    verifier role only on this workspace's audit runs.
    """
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    conn = _get_conn(request)
    if conn is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    value = (body.verifier_model or "").strip() or None
    try:
        await conn.execute(
            "UPDATE project_checkouts SET bug_finder_verifier_model=? WHERE id=?",
            (value, workspace_id),
        )
        await conn.commit()
        return JSONResponse({
            "workspace_id": workspace_id,
            "verifier_model": value or "",
        })
    except Exception as exc:
        log.warning(
            "set_bug_finder_verifier_failed",
            workspace=workspace_id, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


class PlanningModeRequest(BaseModel):
    # One of "default" / "plan" / "auto" — see
    # migrations/207_coder_workspaces_planning_mode.sql for the
    # semantics of each. Unknown values are rejected with 400.
    mode: Literal["default", "plan", "auto"]


@router.put("/api/coder/workspaces/{workspace_id}/planning-mode")
async def set_workspace_planning_mode(
    workspace_id: str, body: PlanningModeRequest, request: Request,
) -> JSONResponse:
    """Set the per-workspace coder planning mode.

    Plan-mode cycle (Shift+Tab in the composer):

      * ``default`` — Plan + Act with per-tool permission prompts.
      * ``plan``    — Read-only exploration; write/shell tools are
                      filtered out of the tool list and the system
                      prompt nudges "explore and propose, do not
                      edit". Used as the "explore first" checkpoint
                      before unleashing edits.
      * ``auto``    — Auto-approve mode; the permission policy
                      resolver short-circuits to "allow" for every
                      tool, skipping the modal entirely. Used during
                      trusted long-form work after the user has
                      validated the agent's behavior.

    Takes effect on the NEXT turn — already in-flight turns continue
    under the mode they started under.
    """
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    conn = _get_conn(request)
    if conn is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    try:
        await conn.execute(
            "UPDATE project_checkouts SET planning_mode=? WHERE id=?",
            (body.mode, workspace_id),
        )
        await conn.commit()
        return JSONResponse({"workspace_id": workspace_id, "mode": body.mode})
    except Exception as exc:
        log.warning(
            "set_planning_mode_failed", workspace=workspace_id, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.put("/api/coder/workspaces/{workspace_id}/safeguards")
async def set_workspace_safeguards(
    workspace_id: str, body: SafeguardsRequest, request: Request,
) -> JSONResponse:
    """Toggle the hybrid-loop soft circuit-breakers for one workspace.

    When ``enabled=False`` the act loop bypasses the streak / stagnation /
    inspection breakers and relies only on the hard iteration ceiling
    (raised in phase_act.py when the flag is off). Use for strong
    API-backed or strong local models that legitimately run long.
    """
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    conn = _get_conn(request)
    if conn is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    try:
        await conn.execute(
            "UPDATE project_checkouts SET safeguards_enabled=? WHERE id=?",
            (1 if body.enabled else 0, workspace_id),
        )
        await conn.commit()
        return JSONResponse({"workspace_id": workspace_id, "enabled": body.enabled})
    except Exception as exc:
        log.warning(
            "set_safeguards_failed", workspace=workspace_id, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.put("/api/coder/workspaces/{workspace_id}/always-on")
async def set_workspace_always_on(
    workspace_id: str, body: AlwaysOnRequest, request: Request,
) -> JSONResponse:
    """Toggle the container-lifecycle policy for one workspace.

    ``always_on=true`` exempts the workspace from the idle reaper —
    the container stays running across browser sessions, useful for
    workspaces hosting a dev server or daemon the user wants to leave
    up while they switch tabs.

    ``always_on=false`` opts into the reaper. The container is
    auto-stopped after ``coder_idle_timeout`` seconds of no activity
    on the workspace's HTTP routes / chat completions. The DB row
    and volume survive; the user can restart with one click or by
    sending a new chat message.

    Takes effect on the next reaper tick (≤ 2 minutes). The flag is
    persisted on ``project_checkouts.always_on`` (migration 211).
    """
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    try:
        info = await mgr.set_always_on(
            workspace_id, always_on=body.always_on,
        )
        return JSONResponse(
            {"workspace_id": workspace_id, "always_on": info.always_on},
        )
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception as exc:
        log.warning(
            "set_always_on_failed", workspace=workspace_id, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/preview-token/{workspace_id}")
async def mint_preview_token(
    workspace_id: str, request: Request,
) -> JSONResponse:
    """Mint a one-time token for the isolated preview origin handoff.

    Authenticated route (main session cookie). The UI calls this BEFORE
    setting the preview iframe src; the returned token is embedded as
    ``?_pvt=`` on the iframe URL and consumed by the isolated proxy on
    first request, which then sets a preview-session cookie scoped to
    the isolated origin for all subsequent in-iframe requests.

    Returns 404 if the workspace doesn't belong to the caller, 503 if
    the token store is unavailable (init failure), 501 if isolation is
    disabled. See docs/superpowers/specs/2026-05-27-preview-origin-
    isolation-design.md.
    """
    from augmentum.config import settings

    if not settings.coder_preview_isolation_enabled:
        return JSONResponse(
            {"error": "Preview isolation is disabled"}, status_code=501,
        )
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    store = getattr(request.app.state, "preview_token_store", None)
    if store is None:
        return JSONResponse(
            {"error": "Preview token store unavailable"}, status_code=503,
        )
    try:
        token, expires_at = store.mint(
            user_id=uid,
            workspace_id=workspace_id,
            ttl_s=float(settings.coder_preview_token_ttl_seconds),
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    # expires_at is wall-clock from the store (time.time() + ttl).
    import time as _time
    expires_in = max(1, int(round(expires_at - _time.time())))
    return JSONResponse({
        "token": token,
        "expires_in": expires_in,
        "isolated_origin": _isolated_preview_origin(request),
    })


def _isolated_preview_origin(request: Request) -> str:
    """Derive the isolated-preview origin from settings + request host.

    Settings override (``coder_preview_isolated_origin``) wins when set.
    Otherwise use the request's host (with the port stripped) and the
    configured isolated port. Returns an empty string if isolation is
    off — callers must check the setting first.
    """
    from augmentum.config import settings
    explicit = (settings.coder_preview_isolated_origin or "").strip()
    if explicit:
        return explicit.rstrip("/")
    host_header = request.headers.get("host") or ""
    host_only = host_header.split(":", 1)[0].strip()
    if not host_only:
        return ""
    # Defensive: reject Host-header injection attempts. Only plain
    # hostname / IP characters are accepted.
    if not re.match(r"^[A-Za-z0-9._\-]+$", host_only):
        return ""
    scheme = "https" if request.url.scheme == "https" else "http"
    port = int(settings.coder_preview_isolated_port)
    return f"{scheme}://{host_only}:{port}"


@router.get("/api/coder/workspaces/{workspace_id}/profile")
async def get_workspace_profile(workspace_id: str, request: Request) -> JSONResponse:
    """Return editable workspace profile facts for the authenticated user."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    store = await _get_profile_store(request)
    if store is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    try:
        from augmentum.coder.profile import render_profile_block

        entries = await store.query_for_workspace(
            user_id=uid,
            workspace_id=workspace_id,
        )
        return JSONResponse({
            "workspace_id": workspace_id,
            "entries": [entry.to_dict() for entry in entries],
            "rendered": render_profile_block(entries, query="", max_entries=16),
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.put("/api/coder/workspaces/{workspace_id}/profile")
async def update_workspace_profile(
    workspace_id: str,
    body: ProfileUpdateRequest,
    request: Request,
) -> JSONResponse:
    """Edit or delete workspace profile facts."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    store = await _get_profile_store(request)
    if store is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    try:
        from augmentum.coder.profile import ACTIVE_PROFILE_CATEGORIES

        changed: list[dict] = []
        for item in body.entries:
            category = item.category.strip().lower()
            key = item.key.strip()
            if category not in ACTIVE_PROFILE_CATEGORIES or not key:
                return JSONResponse(
                    {"error": f"Invalid profile key: {item.category}.{item.key}"},
                    status_code=400,
                )
            target_workspace = workspace_id if item.workspace_scoped else ""
            if item.action == "delete":
                deleted = await store.delete(
                    user_id=uid,
                    workspace_id=target_workspace,
                    category=category,
                    key=key,
                )
                changed.append({"action": "delete", "category": category, "key": key, "deleted": deleted})
            else:
                entry = await store.upsert(
                    user_id=uid,
                    workspace_id=target_workspace,
                    category=category,
                    key=key,
                    value=item.value,
                    confidence=item.confidence,
                )
                changed.append({"action": "upsert", "entry": entry.to_dict()})
        entries = await store.query_for_workspace(user_id=uid, workspace_id=workspace_id)
        return JSONResponse({
            "workspace_id": workspace_id,
            "changed": changed,
            "entries": [entry.to_dict() for entry in entries],
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/coder/workspaces/{workspace_id}/services")
async def list_workspace_services(workspace_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    store = await _get_service_store(request)
    try:
        from augmentum.coder.services import WorkspaceServiceManager

        services = await WorkspaceServiceManager(
            mgr,
            workspace_id,
            store=store,
            user_id=uid,
        ).list()
        return JSONResponse({"services": [svc.to_dict() for svc in services]})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/workspaces/{workspace_id}/services")
async def start_workspace_service(
    workspace_id: str,
    body: ServiceStartRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    store = await _get_service_store(request)
    try:
        from augmentum.coder.services import WorkspaceServiceManager

        svc = await WorkspaceServiceManager(
            mgr,
            workspace_id,
            store=store,
            user_id=uid,
        ).start(
            command=body.command,
            name=body.name,
            cwd=body.cwd,
            env=body.env,
            ports=body.ports,
        )
        return JSONResponse({"service": svc.to_dict()}, status_code=201)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/coder/workspaces/{workspace_id}/services/{service_id}/logs")
async def get_workspace_service_logs(
    workspace_id: str,
    service_id: str,
    request: Request,
    lines: int = 120,
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    store = await _get_service_store(request)
    try:
        from augmentum.coder.services import WorkspaceServiceManager

        text = await WorkspaceServiceManager(
            mgr,
            workspace_id,
            store=store,
            user_id=uid,
        ).logs(service_id, lines=lines)
        return JSONResponse({"service_id": service_id, "logs": text})
    except KeyError:
        return JSONResponse({"error": "Service not found"}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.delete("/api/coder/workspaces/{workspace_id}/services/{service_id}")
async def stop_workspace_service(
    workspace_id: str,
    service_id: str,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    store = await _get_service_store(request)
    try:
        from augmentum.coder.services import WorkspaceServiceManager

        svc_manager = WorkspaceServiceManager(mgr, workspace_id, store=store, user_id=uid)
        svc = await svc_manager.stop(service_id)
        if store is not None:
            await store.delete(service_id, user_id=uid, workspace_id=workspace_id)
        return JSONResponse({"service": svc.to_dict()})
    except KeyError:
        return JSONResponse({"error": "Service not found"}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/workspaces/{workspace_id}/services/{service_id}/stop")
async def soft_stop_workspace_service(
    workspace_id: str,
    service_id: str,
    request: Request,
) -> JSONResponse:
    """Stop a service WITHOUT deleting its row.

    Sibling to the DELETE handler, which removes the row entirely (the
    agent's "I'm done with this service" semantic). This route is what
    the user-facing services panel binds Stop to, so the row persists
    and the matching /start route can revive it with one click.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    store = await _get_service_store(request)
    try:
        from augmentum.coder.services import WorkspaceServiceManager

        svc = await WorkspaceServiceManager(
            mgr, workspace_id, store=store, user_id=uid,
        ).stop(service_id)
        return JSONResponse({"service": svc.to_dict()})
    except KeyError:
        return JSONResponse({"error": "Service not found"}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/workspaces/{workspace_id}/services/{service_id}/start")
async def restart_workspace_service(
    workspace_id: str,
    service_id: str,
    request: Request,
) -> JSONResponse:
    """Re-launch a stopped service with its saved configuration.

    Counterpart to /stop. Closes the user-controlled toggle loop — the
    services panel uses these two routes to turn a service off and back
    on without having to remember its command / cwd / env / ports.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    store = await _get_service_store(request)
    try:
        from augmentum.coder.services import WorkspaceServiceManager

        svc = await WorkspaceServiceManager(
            mgr, workspace_id, store=store, user_id=uid,
        ).restart(service_id)
        return JSONResponse({"service": svc.to_dict()})
    except KeyError:
        return JSONResponse({"error": "Service not found"}, status_code=404)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/workspaces/{workspace_id}/services/probe")
async def probe_workspace_service(
    workspace_id: str,
    body: ServiceProbeRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    store = await _get_service_store(request)
    try:
        from augmentum.coder.services import WorkspaceServiceManager

        probe = await WorkspaceServiceManager(
            mgr,
            workspace_id,
            store=store,
            user_id=uid,
        ).probe(
            service_id=body.service_id,
            url=body.url,
            port=body.port,
            timeout=body.timeout,
        )
        return JSONResponse({"probe": probe})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/coder/oracle-stats")
async def get_coder_oracle_stats(request: Request, limit: int = 500) -> JSONResponse:
    """Verification-spine telemetry rollup (spec 2026-07-06, Phase 2).

    Aggregates the per-turn ``oracle`` metrics block over the caller's
    recent finished runs — headline number is ``no_oracle_done_rate``
    (write-turns that shipped with no oracle after the last write).
    Observational only; nothing reads this to gate a turn.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = await _get_ledger_store(request)
    if store is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    stats = await store.oracle_stats(user_id=uid, limit=limit)
    return JSONResponse({"stats": stats})


@router.get("/api/coder/runs/{run_id}")
async def get_coder_run(run_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = await _get_ledger_store(request)
    if store is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    run = await store.get_run(run_id, user_id=uid)
    if run is None:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    return JSONResponse({"run": run})


@router.get("/api/coder/runs/{run_id}/verification")
async def get_coder_run_verification(run_id: str, request: Request) -> JSONResponse:
    """The persisted independent-verification verdict for a completed run.

    The brief carries the verdict live in its envelope; this is the cold-open
    fallback (a brief reopened from a stale notification, past the envelope).
    Returns an ``unchecked`` verdict when none was recorded — never 404s, so
    the brief always has something honest to render.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = _get_conn(request)
    verdict = None
    if conn is not None:
        from augmentum.coder.run_verifier import load_verdict
        verdict = await load_verdict(conn, run_id=run_id, user_id=uid)
    if verdict is None:
        verdict = {
            "tier": "unchecked", "oracle": "none", "reason": "",
            "verifier_model": "", "self_verified": False,
        }
    verdict.setdefault("contract_unmet", [])
    return JSONResponse({"verification": verdict})


@router.get("/api/coder/runs/{turn_run_id}/citations")
async def get_coder_run_citations(turn_run_id: str, request: Request) -> JSONResponse:
    """Claim→proof provenance for a coder turn (the citation ledger).

    ``turn_run_id`` is the ledger ``ctr_`` id — the same id the brief renders
    the diff for (``review_turn_id`` / ``mountReviewPanel``), so each citation
    deep-links straight into the changed lines the user is deciding on. Returns
    ``[]`` when a run emitted no citations — a valid, honest empty state.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = _get_conn(request)
    rows: list = []
    if conn is not None:
        from augmentum.coder.citations import load_citations
        rows = await load_citations(conn, turn_run_id=turn_run_id, user_id=uid)
    return JSONResponse({"citations": rows})


@router.get("/api/coder/runs/{run_id}/events")
async def get_coder_run_events(
    run_id: str,
    request: Request,
    limit: int = 500,
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = await _get_ledger_store(request)
    if store is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    events = await store.list_events(run_id, user_id=uid, limit=limit)
    if not events:
        run = await store.get_run(run_id, user_id=uid)
        if run is None:
            return JSONResponse({"error": "Run not found"}, status_code=404)
    return JSONResponse({"events": events})


# ---------------------------------------------------------------------------
# Background-run reconnect — see augmentum/coder/run_broker.py
# ---------------------------------------------------------------------------


@router.get("/api/coder/workspaces/{workspace_id}/active-run")
async def get_active_coder_run(workspace_id: str, request: Request) -> JSONResponse:
    """Return the in-flight run for ``workspace_id`` if any.

    UI calls this on mount so a reconnecting client (mobile screen
    wake, tab restore) can attach to ``/api/coder/runs/{id}/stream``
    instead of starting a new turn or leaving the user with a frozen
    Send button. Authoritative source is the in-process broker;
    falls back to the SQL ledger if the broker entry has already
    been evicted but the run row is still ``status='running'`` —
    that only happens transiently across the eviction sweep.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    broker = getattr(request.app.state, "coder_run_broker", None)
    if broker is not None:
        entry = broker.get_active_for_workspace(
            user_id=uid, workspace_id=workspace_id,
        )
        if entry is not None:
            return JSONResponse({
                "run_id": entry.run_id,
                "started_at": entry.started_at,
                "seq": entry.seq,
                "source": "broker",
            })

    conn = _get_conn(request)
    if conn is None:
        return JSONResponse({"run_id": None})
    try:
        cursor = await conn.execute(
            """
            SELECT id, started_at FROM coder_turn_runs
            WHERE workspace_id = ? AND user_id = ? AND status = 'running'
            ORDER BY started_at DESC LIMIT 1
            """,
            (workspace_id, uid),
        )
        row = await cursor.fetchone()
    except Exception:
        return JSONResponse({"run_id": None})
    if row is None:
        return JSONResponse({"run_id": None})
    return JSONResponse({
        "run_id": row[0],
        "started_at": row[1],
        "source": "ledger",
    })


@router.post("/api/coder/workspaces/{workspace_id}/background-runs")
async def queue_background_run(
    workspace_id: str, body: BackgroundRunRequest, request: Request,
) -> JSONResponse:
    """Queue a headless coder mission against this workspace.

    The job runner drives the normal coder turn stack with itself as the
    "client" (see ``jobs/handlers/coder_background_run.py``); the user
    gets a ``coder.run.complete`` / ``coder.run.failed`` notification at
    the end, and can watch live any time via the standard active-run
    reattach path. List/cancel ride the existing jobs surface:
    ``GET /api/jobs/?type=coder_background_run`` and
    ``POST /api/jobs/{id}/cancel``.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    prompt = (body.prompt or "").strip()
    model = (body.model or "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)
    if not model:
        # Deliberate: the user picks the model at queue time. No silent
        # server-side default (never auto-select on the user's behalf).
        return JSONResponse({"error": "model is required"}, status_code=400)

    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if jobs_store is None or job_runner is None:
        return JSONResponse(
            {"error": "Background jobs unavailable"}, status_code=503,
        )

    job_id = await jobs_store.create(
        user_id=uid,
        job_type="coder_background_run",
        payload={
            "workspace_id": workspace_id,
            "prompt": prompt,
            "model": model,
            "coder_strategy": (body.coder_strategy or "").strip(),
        },
        priority=5,
        # 2, not 1: a container restart mid-mission re-queues once — the
        # agent re-enters against the workspace's current state (same
        # convergence contract as interactive interrupted-turn
        # continuation). Routine handler errors don't re-raise as
        # retryable, so this only activates on the crash path.
        max_attempts=2,
    )
    job_runner.wake()
    log.info(
        "coder_background_run_queued",
        user_id=uid, workspace_id=workspace_id, job_id=job_id, model=model,
    )
    return JSONResponse({"job_id": job_id, "status": "pending"})


@router.post("/api/coder/workspaces/{workspace_id}/research-runs")
async def queue_research_run(
    workspace_id: str, body: ResearchRunRequest, request: Request,
) -> JSONResponse:
    """Queue an autonomous improvement loop against this workspace.

    The job runner drives repeated headless coder turns (propose →
    implement → measure the user's objective command → keep on
    improvement, git-revert otherwise); progress is visible in the
    workspace itself (RESEARCH_LOG.md + git checkpoints + the normal
    live-run reattach path). List/cancel ride the jobs surface:
    ``GET /api/jobs/?type=coder_research_run`` and
    ``POST /api/jobs/{id}/cancel``.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    model = (body.model or "").strip()
    objective_command = (body.objective_command or "").strip()
    if not model:
        # Deliberate: the user picks the model at queue time. No silent
        # server-side default (never auto-select on the user's behalf).
        return JSONResponse({"error": "model is required"}, status_code=400)
    if not objective_command:
        return JSONResponse(
            {"error": "objective_command is required"}, status_code=400,
        )

    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if jobs_store is None or job_runner is None:
        return JSONResponse(
            {"error": "Background jobs unavailable"}, status_code=503,
        )

    job_id = await jobs_store.create(
        user_id=uid,
        job_type="coder_research_run",
        payload={
            "workspace_id": workspace_id,
            "model": model,
            "objective_command": objective_command,
            "direction": body.direction,
            "intentions": (body.intentions or "").strip(),
            "experiments": body.experiments,
            "objective_timeout": body.objective_timeout,
            "turn_max_seconds": body.turn_max_seconds,
            "min_delta": body.min_delta,
            "coder_strategy": (body.coder_strategy or "").strip(),
        },
        priority=5,
        # 1, not 2: the loop's bookkeeping (best score, last-good sha,
        # experiment history) lives in handler memory — a blind requeue
        # would restart from experiment 1 against a mutated workspace.
        # Partial results are already durable (RESEARCH_LOG.md + kept
        # checkpoints), so a crash loses only the in-flight experiment.
        max_attempts=1,
    )
    job_runner.wake()
    log.info(
        "coder_research_run_queued",
        user_id=uid, workspace_id=workspace_id, job_id=job_id, model=model,
        direction=body.direction, experiments=body.experiments,
    )
    return JSONResponse({"job_id": job_id, "status": "pending"})


@router.post("/api/coder/runs/{run_id}/interject")
async def interject_into_coder_run(
    run_id: str, request: Request,
) -> JSONResponse:
    """Append a cooperative user message to an in-flight run's inbox.

    Two delivery modes — both queue the message; what differs is when
    the handler drains and how the model sees it:

      * ``mode="queue"`` (default) — drains at end-of-turn. Becomes
        the user content of a new turn (fresh run_id) once the
        current loop exits naturally. Safe for "I have a follow-up
        for after this is done" messages — doesn't derail in-flight
        work. This is the mode users expect when they just type
        while the agent is running.
      * ``mode="steer"`` — drains at the next iteration boundary of
        the CURRENT turn. Appended as a user message in the next
        model call's prompt. Use for "actually, switch focus to X"
        course corrections that should land mid-turn. Latency to
        delivery depends on iteration cadence — a long-running shell
        exec can delay a steer by 30s+.

    Returns the queued entry id + delivery mode + queue depth so the
    UI can render a "queued" badge that turns into "delivered" once
    the handler stamps ``delivered_at``.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = await _get_ledger_store(request)
    if store is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    run = await store.get_run(run_id, user_id=uid)
    if run is None:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    broker = getattr(request.app.state, "coder_run_broker", None)
    if broker is None:
        return JSONResponse(
            {"error": "Broker unavailable — run cannot be interjected"},
            status_code=503,
        )

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    content = str(body.get("content") or "").strip()
    attachments = body.get("attachments") or []
    if not isinstance(attachments, list):
        attachments = []
    mode = str(body.get("mode") or "queue").strip().lower()
    if not content and not attachments:
        return JSONResponse(
            {"error": "Either content or attachments is required"},
            status_code=400,
        )

    entry = broker.enqueue_user_message(
        run_id,
        content=content,
        attachments=attachments,
        mode=mode,
    )
    if entry is None:
        # Two reasons we land here: run is done (race against natural
        # completion) or inbox is full. Surface a 409 in either case;
        # the client can fall back to starting a new turn.
        return JSONResponse(
            {
                "error": "Cannot interject — run is finished or inbox full",
                "depth": broker.inbox_depth(run_id),
            },
            status_code=409,
        )

    return JSONResponse({
        "queued": True,
        "run_id": run_id,
        "msg_id": entry["id"],
        "mode": entry["mode"],
        "queue_depth": broker.inbox_depth(run_id),
    })


@router.post("/api/coder/runs/{run_id}/pause")
async def pause_coder_run(run_id: str, request: Request) -> JSONResponse:
    """Pause an in-flight run at the next iteration boundary.

    Does NOT cancel — the agent will finish whatever it's currently
    running (shell command, file write, model call in progress) and
    then block on the pause gate. Inbox interjections stay queued
    during the pause; they drain on the next iteration after resume.

    Returns 409 if the run isn't actually pause-able (unknown,
    finished, or already paused). The UI can use this to update the
    Pause/Resume button state without prompting.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = await _get_ledger_store(request)
    if store is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    run = await store.get_run(run_id, user_id=uid)
    if run is None:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    broker = getattr(request.app.state, "coder_run_broker", None)
    if broker is None:
        return JSONResponse({"error": "Broker unavailable"}, status_code=503)
    ok = broker.pause(run_id)
    if not ok:
        return JSONResponse(
            {"paused": False, "reason": "run not active or already paused"},
            status_code=409,
        )
    return JSONResponse({"paused": True, "run_id": run_id})


@router.post("/api/coder/runs/{run_id}/resume")
async def resume_coder_run(run_id: str, request: Request) -> JSONResponse:
    """Resume a paused run. Idempotent: resuming a non-paused run is a 409.

    Releases the pause gate so the handler's awaiter wakes. Any queued
    interjections drain on the next iteration (or end-of-turn, by
    their mode).
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = await _get_ledger_store(request)
    if store is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    run = await store.get_run(run_id, user_id=uid)
    if run is None:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    broker = getattr(request.app.state, "coder_run_broker", None)
    if broker is None:
        return JSONResponse({"error": "Broker unavailable"}, status_code=503)
    ok = broker.resume(run_id)
    if not ok:
        return JSONResponse(
            {"resumed": False, "reason": "run not active or not paused"},
            status_code=409,
        )
    return JSONResponse({"resumed": True, "run_id": run_id})


@router.post("/api/coder/workspaces/{workspace_id}/rewind")
async def rewind_coder_workspace(
    workspace_id: str, request: Request,
) -> JSONResponse:
    """Rewind the most recent coder turn for this workspace.

    Reverts the workspace files via the live TurnSnapshot, pops the
    matching ``turn_summaries`` entry, and clears per-request scratchpads
    (plan / tasks / mission). If a turn is still in flight it is cancelled
    with reason ``user_rewind`` first; the route waits briefly for the
    cancellation to settle before mutating state.

    The frontend drops the matching user + assistant messages from the
    chat session tree on a successful response — the chat tree's source
    of truth is client-side, so the backend doesn't touch it here.

    Side effects outside the workspace (HTTP requests, started services,
    external DB writes, ``git push``) cannot be undone. The response
    ``warnings`` list flags this for the user.

    Response shape (see ``RewindOutcome.to_dict``):
        {ok, run_id, cancelled_in_flight, restored_paths,
         irreversible_paths, turn_summary_popped, warnings, error}
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    # Optional body: {"mode": "both" | "files" | "conv"}. Empty/missing
    # falls back to "both" (the legacy, all-restore behaviour). See
    # ``augmentum.coder.rewind.rewind_last_turn`` for mode semantics.
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    requested_mode = str(body.get("mode") or "both").strip().lower()

    from augmentum.coder.rewind import rewind_last_turn
    outcome = await rewind_last_turn(
        user_id=uid,
        workspace_id=workspace_id,
        app_state=request.app.state,
        mode=requested_mode,
    )
    status = 200 if outcome.ok else 404
    return JSONResponse(outcome.to_dict(), status_code=status)


@router.post("/api/coder/runs/{run_id}/cancel")
async def cancel_coder_run(run_id: str, request: Request) -> JSONResponse:
    """Explicitly cancel an in-flight run.

    The Stop button in the chat UI used to abort the fetch — which
    worked because the request task held the agent loop. With the
    broker model, the run survives client disconnect on purpose, so
    Stop has to call this route instead. ``broker.cancel`` flips the
    flag and ``Task.cancel()`` the detached task; the existing
    ``cancel_workspace_execs`` cleanup fires inside that task's
    finally so any in-flight shell exec gets SIGTERM.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = await _get_ledger_store(request)
    if store is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    run = await store.get_run(run_id, user_id=uid)
    if run is None:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    broker = getattr(request.app.state, "coder_run_broker", None)
    if broker is None:
        return JSONResponse({"cancelled": False, "reason": "no_broker"})

    # Reason hint propagates into the handler's CancelledError path
    # so the next turn's <prior_turns> block can tell the model *why*
    # the turn ended. Body is optional — legacy clients that POST no
    # body keep the default "user_cancel" behaviour.
    reason = "user_cancel"
    try:
        body = await request.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        raw = body.get("reason")
        if isinstance(raw, str) and raw.strip():
            reason = raw.strip()[:50]

    cancelled = broker.cancel(run_id, reason=reason)
    return JSONResponse(
        {"cancelled": cancelled, "run_id": run_id, "reason": reason},
    )


@router.get("/api/coder/runs/{run_id}/stream")
async def stream_coder_run(
    run_id: str,
    request: Request,
    since: int = 0,
) -> Response:
    """Reattach to a coder run as NDJSON.

    Two modes:

    - **Live**: broker still has the entry. Subscribe from
      ``since`` and stream chunks as they're produced, in the same
      NDJSON wire format as ``/api/chat`` (per-line JSON with
      ``message.content``, ``augmentum``, ``done``). Frontend
      reuses ``CoderStream._processChunk``.

    - **Replay**: broker evicted the entry (run finished while you
      were away). Emit each ``coder_turn_events`` row as a chunk,
      then a final synthetic chunk carrying ``final_assistant_message``
      from the persisted conversation so the UI can render the
      assistant's text without a second round-trip.

    No-buffer headers (``X-Accel-Buffering: no``) so nginx/Cloudflare
    don't hold the response open in front of the user.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = await _get_ledger_store(request)
    if store is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    run = await store.get_run(run_id, user_id=uid)
    if run is None:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    broker = getattr(request.app.state, "coder_run_broker", None)
    entry = broker.get(run_id) if broker is not None else None

    async def _live_stream():
        emitted = False
        try:
            async for buffered in broker.subscribe(run_id, since_seq=since):
                emitted = True
                yield (
                    _coder_chunk_to_ndjson_line(buffered.chunk, buffered.seq)
                    + b"\n"
                )
        except asyncio.CancelledError:
            return
        # The live/replay choice is made from a broker snapshot taken
        # before the response starts streaming; the sweeper can evict
        # the entry in between, in which case subscribe() yields
        # nothing. Fall through to the ledger replay so the client
        # gets the run's history instead of an empty stream.
        if not emitted and broker.get(run_id) is None:
            async for line in _replay_stream():
                yield line

    async def _replay_stream():
        events = await store.list_events(run_id, user_id=uid, limit=2000)
        for evt in events:
            seq = int(evt.get("seq") or 0)
            if seq <= since:
                continue
            payload = evt.get("payload") or {}
            line = _coder_event_replay_line(
                evt.get("type") or "",
                evt.get("phase") or "",
                evt.get("status") or "",
                payload,
                run_id=run_id,
                seq=seq,
            )
            yield line + b"\n"
        # Final chunk: pull the last assistant message from persisted
        # conversation so the UI can render the actual prose. The
        # ledger holds structured events only — see
        # docs/superpowers/specs/coder-background-runs (or just the
        # broker module docstring) for the rationale.
        final_text = ""
        persistence = await _get_coder_persistence(request)
        if persistence is not None and run.get("workspace_id"):
            try:
                messages = await persistence.load_conversation(
                    str(run["workspace_id"]), user_id=uid,
                )
                for msg in reversed(messages or []):
                    if (msg.get("role") or "") == "assistant":
                        content = msg.get("content")
                        if isinstance(content, str):
                            final_text = content
                        break
            except Exception:
                log.debug("coder_stream_final_msg_load_failed", exc_info=True)
        yield _coder_final_state_line(
            run_id=run_id,
            run=run,
            final_assistant_message=final_text,
        ) + b"\n"

    generator = _live_stream() if entry is not None else _replay_stream()

    return StreamingResponse(
        generator,
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _coder_chunk_to_ndjson_line(chunk, seq: int) -> bytes:
    """Encode an ``InternalStreamChunk`` for the /stream NDJSON wire.

    Mirrors the subset of ``_chunk_to_ollama_chat_ndjson`` the
    frontend actually consumes — content, thinking, augmentum,
    done. ``seq`` is stamped into augmentum so the client can
    resume from a known cursor on the next reconnect.
    """
    aug = dict(chunk.augmentum) if isinstance(chunk.augmentum, dict) else {}
    aug["seq"] = seq
    body: dict[str, Any] = {
        "message": {
            "role": "assistant",
            "content": chunk.content_delta or "",
        },
        "done": bool(chunk.done),
        "augmentum": aug,
    }
    if getattr(chunk, "thinking_delta", None):
        aug["model_thinking_delta"] = chunk.thinking_delta
    return json.dumps(body).encode()


def _coder_event_replay_line(
    event_type: str,
    phase: str,
    status: str,
    payload: dict,
    *,
    run_id: str,
    seq: int,
) -> bytes:
    """Encode a stored ``coder_turn_events`` row as an NDJSON chunk.

    The payload column already holds the original ``augmentum`` dict
    minus ``mode`` (stripped at write time — see
    ``CoderTurnLedger.observe_chunk``). We restore the run_id +
    seq fields so the frontend treats the replayed chunk identically
    to a live one.
    """
    aug = dict(payload) if isinstance(payload, dict) else {}
    aug.setdefault("run_id", run_id)
    aug.setdefault("phase", phase)
    aug.setdefault("status", status)
    aug["seq"] = seq
    aug["replay"] = True
    body = {
        "message": {"role": "assistant", "content": ""},
        "done": False,
        "augmentum": aug,
    }
    return json.dumps(body).encode()


def _coder_final_state_line(
    *,
    run_id: str,
    run: dict,
    final_assistant_message: str,
) -> bytes:
    """Emit the synthetic terminal chunk for a replayed run.

    Carries the final assistant prose (loaded from persisted
    conversation) so the frontend's ``onComplete`` callback fires
    with real content rather than an empty string. ``done=True``
    signals end-of-stream.
    """
    aug = {
        "run_id": run_id,
        "status": run.get("status") or "complete",
        "phase": "complete",
        "final_state": True,
        "iterations": run.get("iterations"),
        "tool_calls": run.get("tool_calls"),
        "finish_reason": run.get("finish_reason") or "",
        "changed_files": run.get("changed_files") or [],
        "commands_run": run.get("commands_run") or [],
        "checkpoint_id": run.get("checkpoint_id") or "",
    }
    body = {
        "message": {
            "role": "assistant",
            "content": final_assistant_message or "",
        },
        "done": True,
        "augmentum": aug,
    }
    return json.dumps(body).encode()


@router.get("/api/coder/workspaces/{workspace_id}/ports")
async def workspace_ports(workspace_id: str, request: Request) -> JSONResponse:
    """List dev-server ports for a workspace + which are currently
    listening inside the container.

    Returns ``{ports: [...], preview: {...}}``.
    Empty host_port means the port is not published yet; existing
    workspaces can enable publishing later via the publish endpoint.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        ports = await mgr.list_ports(workspace_id)
        gate_urls = await _maybe_sync_workspace_gate(mgr, workspace_id, ports)
        resp_data = {
            "ports": ports,
            "preview": _build_preview_summary(ports, workspace_id),
        }
        if gate_urls:
            resp_data["gate_urls"] = gate_urls
        return JSONResponse(resp_data)
    except Exception as exc:
        log.warning("list_ports_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({
            "ports": [],
            "preview": _build_preview_summary([], workspace_id),
            "error": str(exc),
        })


@router.post("/api/coder/workspaces/{workspace_id}/ports/publish")
async def publish_workspace_ports(workspace_id: str, request: Request) -> JSONResponse:
    """Enable common dev-server port publishing for an existing workspace.

    This may recreate the container against the same persistent /workspace
    volume because Docker port bindings cannot be changed in-place once a
    container already exists.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        info, changed = await mgr.enable_published_ports(workspace_id)
        return JSONResponse({"workspace": info.__dict__, "changed": changed})
    except Exception as exc:
        log.warning("publish_workspace_ports_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/workspaces/{workspace_id}/lan")
async def set_workspace_lan_accessible(
    workspace_id: str, request: Request,
) -> JSONResponse:
    """Toggle LAN accessibility for a workspace.

    Recreates the container so port bindings switch between loopback
    (127.0.0.1) and LAN-reachable (0.0.0.0). The workspace volume
    survives — data is never at risk. Deliberate user action only.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    enabled = bool(body.get("enabled", False))
    try:
        info, changed = await mgr.set_lan_accessible(workspace_id, enabled)
        return JSONResponse({
            "workspace": info.__dict__,
            "changed": changed,
            "lan_accessible": info.lan_accessible,
        })
    except Exception as exc:
        log.warning(
            "set_lan_accessible_failed",
            workspace=workspace_id, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/coder/workspaces/{workspace_id}/ready")
async def workspace_ready(workspace_id: str, request: Request) -> JSONResponse:
    """Check if workspace setup has completed (readiness probe).

    Also reports clone status so the frontend can surface auth or branch
    errors when a workspace was created with a git URL. Pre-2026-04-21,
    clone failures would leave the container's primary process stuck on
    a password prompt (no TTY attached), so the ready marker was never
    written and the frontend polled until its 120s timeout — now the
    setup always reaches the ready marker and records clone_ok/
    clone_failed for distinguishing success.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        await mgr._run_command(
            workspace_id,
            ["test", "-f", "/workspace/.augmentum/ready"],
            timeout=3.0,
        )
    except Exception:
        return JSONResponse({"ready": False})

    # Probe clone status. Clone log is only populated when a clone was
    # attempted, so its absence means "no git_url was given".
    clone_status = "none"
    clone_log = ""
    try:
        await mgr._run_command(
            workspace_id,
            ["test", "-f", "/workspace/.augmentum/clone_ok"],
            timeout=3.0,
        )
        clone_status = "ok"
    except Exception:
        try:
            await mgr._run_command(
                workspace_id,
                ["test", "-f", "/workspace/.augmentum/clone_failed"],
                timeout=3.0,
            )
            clone_status = "failed"
            try:
                clone_log = await mgr._run_command(
                    workspace_id,
                    ["sh", "-c", "tail -c 2000 /workspace/.augmentum/clone.log 2>/dev/null"],
                    timeout=3.0,
                )
            except Exception as exc:
                log.debug("coder_clone_log_tail_failed", workspace_id=workspace_id, error=str(exc))
        except Exception as exc:
            log.debug("coder_clone_failed_marker_check_failed", workspace_id=workspace_id, error=str(exc))

    return JSONResponse({
        "ready": True,
        "clone_status": clone_status,
        "clone_log": clone_log.strip() if clone_log else "",
    })


@router.post("/api/coder/workspaces/{workspace_id}/start")
async def start_workspace(workspace_id: str, request: Request) -> JSONResponse:
    """Start a stopped workspace container."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        info = await mgr.start(workspace_id)
        return JSONResponse(info.__dict__)
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception as exc:
        log.warning("start_workspace_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/workspaces/{workspace_id}/stop")
async def stop_workspace(workspace_id: str, request: Request) -> JSONResponse:
    """Stop a running workspace container."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        info = await mgr.stop(workspace_id)
        return JSONResponse(info.__dict__)
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception as exc:
        log.warning("stop_workspace_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/workspaces/{workspace_id}/pause")
async def pause_workspace(workspace_id: str, request: Request) -> JSONResponse:
    """Suspend the workspace via the cgroup freezer (running → paused).

    Sub-second resume via ``/start`` — RAM stays held, CPU drops to 0.
    The manager method is idempotent against drift (404 / 409 reconciled
    to ``stopped``); see ContainerManager.pause for the lifecycle contract.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        info = await mgr.pause(workspace_id)
        return JSONResponse(info.__dict__)
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception as exc:
        log.warning("pause_workspace_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.put("/api/coder/workspaces/{workspace_id}/name")
async def rename_workspace(
    workspace_id: str, body: RenameRequest, request: Request,
) -> JSONResponse:
    """Update the user-visible workspace name (DB metadata only).

    The Docker container's internal name is derived from ``workspace_id``
    (see ``_workspace_container_name``) so rename touches no Docker state
    and never collides with running containers.
    """
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    conn = _get_conn(request)
    if conn is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)
    try:
        cursor = await conn.execute(
            "UPDATE project_checkouts SET name=? WHERE id=?",
            (body.name, workspace_id),
        )
        await conn.commit()
        if cursor.rowcount == 0:
            return JSONResponse({"error": "Workspace not found"}, status_code=404)
        return JSONResponse({"workspace_id": workspace_id, "name": body.name})
    except Exception as exc:
        log.warning(
            "rename_workspace_failed", workspace=workspace_id, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.delete("/api/coder/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str, request: Request) -> JSONResponse:
    """Archive a workspace (default) or completely remove it.

    Default (``?purge=0``): ARCHIVE — remove the container, keep the
    ``/workspace`` volume, mark the row archived. Reclaims the container's
    runtime footprint while keeping files + task history restorable.

    ``?purge=1``: completely remove — delete the row AND the volume. This
    is the destructive path the "Completely remove" checkbox opts into.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    purge = request.query_params.get("purge") in ("1", "true", "yes")
    try:
        if purge:
            await mgr.delete(workspace_id, keep_volume=False)
            return JSONResponse({"status": "deleted"})
        await mgr.archive(workspace_id)
        return JSONResponse({"status": "archived"})
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception as exc:
        log.warning(
            "delete_workspace_failed",
            workspace=workspace_id, purge=purge, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/coder/archives")
async def list_archives(request: Request) -> JSONResponse:
    """List the authenticated user's archived workspaces (name, size, tasks)."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    try:
        return JSONResponse({"archives": await mgr.list_archived(user_id=uid)})
    except Exception as exc:
        log.warning("list_archives_failed", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/archives/{workspace_id}/restore")
async def restore_archive(workspace_id: str, request: Request) -> JSONResponse:
    """Respawn a container onto an archived workspace's surviving volume."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        info = await mgr.restore(workspace_id)
        return JSONResponse({"status": "restored", "workspace_id": info.id})
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except Exception as exc:
        log.warning("restore_archive_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.delete("/api/coder/archives/{workspace_id}")
async def purge_archive(workspace_id: str, request: Request) -> JSONResponse:
    """Completely remove an archived workspace — its row AND its volume."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        await mgr.delete(workspace_id, keep_volume=False)
        return JSONResponse({"status": "deleted"})
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception as exc:
        log.warning("purge_archive_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# File endpoints
# ---------------------------------------------------------------------------


@router.get("/api/coder/files/{workspace_id}")
async def list_files(workspace_id: str, request: Request) -> JSONResponse:
    """List files in a workspace directory."""
    path = request.query_params.get("path", "/workspace")
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        entries = await mgr.file_list(workspace_id, path)
        return JSONResponse({"files": [e.__dict__ for e in entries]})
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception as exc:
        log.warning("list_files_failed", workspace=workspace_id, path=path, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/coder/files/{workspace_id}/read")
async def read_file(workspace_id: str, request: Request) -> JSONResponse:
    """Read a file from a workspace container."""
    path = request.query_params.get("path")
    if not path:
        return JSONResponse({"error": "path query parameter required"}, status_code=400)
    err = _validate_workspace_path(path)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        content = await mgr.file_read(workspace_id, path)
        return JSONResponse({"content": content, "path": path})
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception as exc:
        log.warning("read_file_failed", workspace=workspace_id, path=path, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.put("/api/coder/files/{workspace_id}/write")
async def write_file(workspace_id: str, body: WriteFileRequest, request: Request) -> JSONResponse:
    """Write content to a file in a workspace container."""
    err = _validate_workspace_path(body.path)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        await mgr.file_write(workspace_id, body.path, body.content)
        checkpoint_hash = None
        if body.checkpoint:
            short = body.path.replace("/workspace/", "", 1)
            checkpoint_hash = await mgr.git_checkpoint(
                workspace_id, f"User edit: {short}",
            )
        return JSONResponse({
            "status": "written",
            "path": body.path,
            "checkpoint": checkpoint_hash,
        })
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception as exc:
        log.warning(
            "write_file_failed", workspace=workspace_id, path=body.path, error=str(exc)
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


def _safe_relative_path(rel: str) -> str | None:
    """Return the sanitized relative path or None if rejected.

    Rejects absolute paths, backslashes (Windows drag-drop often slips
    these through), parent-dir components, and empty segments. Keeps
    the containing-directory prefix for nested uploads (e.g. from a
    folder drag).
    """
    if not rel:
        return None
    rel = rel.replace("\\", "/").lstrip("/")
    parts = []
    for p in rel.split("/"):
        if p in ("", "."):
            continue
        if p == "..":
            return None
        parts.append(p)
    if not parts:
        return None
    return "/".join(parts)


@router.post("/api/coder/files/{workspace_id}/upload")
async def upload_files(
    workspace_id: str,
    request: Request,
    dest_path: str = Form(default="/workspace"),
    files: list[UploadFile] = File(...),
) -> JSONResponse:
    """Upload one or more files (or a folder via repeated entries) into
    the workspace. Binary-safe: bytes flow through Docker's tar extract
    endpoint, not a shell-quoted printf.

    Clients pass each file with its intended relative path as the
    filename (e.g. ``src/main.py``) — browsers do this automatically
    for ``webkitdirectory`` and DataTransferItem folder drops.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    # Clamp dest_path to /workspace. A malicious client could otherwise
    # write into /etc, /root, etc.
    if not dest_path.startswith("/workspace"):
        return JSONResponse({"error": "dest_path must be under /workspace"}, status_code=400)

    payload: list[tuple[str, bytes]] = []
    skipped: list[str] = []
    total_bytes = 0
    MAX_BYTES = 200 * 1024 * 1024  # 200 MB per upload batch
    for f in files:
        rel = _safe_relative_path(f.filename or "")
        if rel is None:
            skipped.append(f.filename or "<unnamed>")
            continue
        data = await f.read()
        total_bytes += len(data)
        if total_bytes > MAX_BYTES:
            return JSONResponse(
                {"error": f"Upload exceeds {MAX_BYTES // (1024 * 1024)} MB limit"},
                status_code=413,
            )
        payload.append((rel, data))

    if not payload:
        return JSONResponse({"error": "No valid files", "skipped": skipped}, status_code=400)

    try:
        await mgr.file_upload(workspace_id, dest_path, payload)
        return JSONResponse({
            "uploaded": len(payload),
            "skipped": skipped,
            "bytes": total_bytes,
            "dest_path": dest_path,
        })
    except Exception as exc:
        log.warning("upload_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/coder/files/{workspace_id}/download")
async def download_file(workspace_id: str, request: Request) -> Response:
    """Stream a single file from the workspace as an attachment.

    Uses Docker's get-archive so binary files (images, PDFs, archives)
    round-trip correctly — the JSON read endpoint decodes as UTF-8 and
    would mojibake anything that isn't text.
    """
    path = request.query_params.get("path")
    if not path:
        return JSONResponse({"error": "path query parameter required"}, status_code=400)
    if not path.startswith("/workspace") or ".." in path.split("/"):
        return JSONResponse({"error": "path must be under /workspace"}, status_code=400)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        data = await mgr.file_download(workspace_id, path)
        filename = path.rsplit("/", 1)[-1] or "file"
        safe = filename.replace('"', '')
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{safe}"'},
        )
    except FileNotFoundError:
        return JSONResponse({"error": "File not found"}, status_code=404)
    except Exception as exc:
        log.warning("download_file_failed", workspace=workspace_id, path=path, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


# Extension → media-type map for the inline-serve route below. Only
# extensions worth rendering inline in the UI are listed — anything
# else falls back to application/octet-stream which the browser will
# offer as a download (so the route can't be abused to serve, e.g.,
# arbitrary HTML from a workspace as a same-origin payload).
_INLINE_MEDIA_TYPES = {
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".svg":  "image/svg+xml",
}


@router.get("/api/coder/files/{workspace_id}/raw")
async def read_file_raw(workspace_id: str, request: Request) -> Response:
    """Serve a workspace file with inline Content-Type for embedding.

    Sibling to ``/download`` which forces an attachment Content-Disposition.
    This route is for the conversation pane's inline ``<img>`` embeds
    (browser_screenshot artifacts) — same file-fetch path, just doesn't
    force the browser to download. Extension allowlist prevents the
    route from being misused as a same-origin HTML loader.
    """
    path = request.query_params.get("path")
    if not path:
        return JSONResponse({"error": "path query parameter required"}, status_code=400)
    if not path.startswith("/workspace") or ".." in path.split("/"):
        return JSONResponse({"error": "path must be under /workspace"}, status_code=400)
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    media_type = _INLINE_MEDIA_TYPES.get(f".{ext}", "application/octet-stream")
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        data = await mgr.file_download(workspace_id, path)
        return Response(
            content=data,
            media_type=media_type,
            # Short cache so a re-screenshot at the same path renders
            # promptly. Browser-screenshot paths are timestamp-suffixed
            # in practice, so this is belt-and-suspenders.
            headers={"Cache-Control": "private, max-age=60"},
        )
    except FileNotFoundError:
        return JSONResponse({"error": "File not found"}, status_code=404)
    except Exception as exc:
        log.warning("read_file_raw_failed", workspace=workspace_id, path=path, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# File management — delete / rename / mkdir.
#
# These three routes back the inspector file-tree's context menu so the
# user can manage workspace files without dropping to a terminal. Same
# auth + path-validation contract as the read/write routes; container
# ops go through ContainerManager.file_{delete,rename,mkdir}.
#
# Operations are advisory — the agent may also mutate the filesystem
# in the same turn the user is clicking around. The tree refreshes
# after each call so the UI catches up.
# ---------------------------------------------------------------------------


def _validate_workspace_path(path: str) -> str | None:
    """Reject paths outside /workspace or containing parent-dir parts.

    Returns an error message on rejection; ``None`` when the path is
    safe to pass to ContainerManager's file ops. Stricter than
    ``_safe_relative_path`` because these routes accept ABSOLUTE paths
    (the file tree returns absolute paths from ``ls``), so the check
    is "must live under /workspace" rather than "no leading slash."
    """
    if not path:
        return "path required"
    if "\x00" in path:
        return "path contains null byte"
    if not path.startswith("/workspace"):
        return "path must be under /workspace"
    # Even one ``..`` segment is enough to escape: ``/workspace/../etc``.
    for p in path.split("/"):
        if p == "..":
            return "path must not contain '..'"
    # Refuse to operate on the workspace root itself. Deleting /workspace
    # would brick the container; renaming/mkdir there is meaningless.
    if path.rstrip("/") == "/workspace":
        return "cannot operate on workspace root"
    return None


class DeleteFileRequest(BaseModel):
    path: str
    recursive: bool = False
    # UI deletes are reversible by default (move to trash + Undo). Set
    # permanent=True to hard-rm — used by "empty trash" / explicit purges.
    permanent: bool = False


class RestoreFileRequest(BaseModel):
    trash_id: str


class PurgeTrashRequest(BaseModel):
    # Blank trash_id empties the whole trash.
    trash_id: str = ""


class RenameFileRequest(BaseModel):
    old_path: str
    new_path: str
    # ``mv`` silently clobbers an existing destination (and NESTS a
    # directory moved onto an existing directory). The UI's rename and
    # drag-to-move flows send overwrite=False, get a 409 when the
    # target exists, and re-send with overwrite=True only after the
    # user confirms. Defaults False so no caller overwrites by accident.
    overwrite: bool = False


class MkdirRequest(BaseModel):
    path: str


@router.delete("/api/coder/files/{workspace_id}")
async def delete_file(
    workspace_id: str, body: DeleteFileRequest, request: Request,
) -> JSONResponse:
    """Delete a file (or directory tree if ``recursive=True``).

    Idempotent on the container side — deleting a path that no longer
    exists succeeds quietly. Concurrent deletes by the agent and the
    UI shouldn't surface a 500 to whoever races.
    """
    err = _validate_workspace_path(body.path)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        if body.permanent:
            await mgr.file_delete(
                workspace_id, body.path, recursive=body.recursive,
            )
            return JSONResponse({"deleted": True, "path": body.path, "trashed": False})
        # Default: reversible soft delete. Returns a trash_id the UI
        # surfaces as an Undo action.
        trash_id = await mgr.file_trash(workspace_id, body.path)
        return JSONResponse({
            "deleted": True, "path": body.path,
            "trashed": True, "trash_id": trash_id,
        })
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception as exc:
        log.warning(
            "delete_file_failed",
            workspace=workspace_id, path=body.path, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/files/{workspace_id}/restore")
async def restore_file(
    workspace_id: str, body: RestoreFileRequest, request: Request,
) -> JSONResponse:
    """Restore a trashed item (from the delete Undo action or trash drawer)."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        result = await mgr.file_restore(workspace_id, body.trash_id)
        if not result.get("restored"):
            # Occupied original / missing entry — a real 409, not a 500.
            return JSONResponse(result, status_code=409)
        return JSONResponse(result)
    except Exception as exc:
        log.warning(
            "restore_file_failed",
            workspace=workspace_id, trash_id=body.trash_id, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/coder/files/{workspace_id}/trash")
async def list_trash(workspace_id: str, request: Request) -> JSONResponse:
    """List trashed items for the trash drawer (newest first)."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        return JSONResponse({"items": await mgr.file_list_trash(workspace_id)})
    except Exception as exc:
        log.warning("list_trash_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"items": []})


@router.delete("/api/coder/files/{workspace_id}/trash")
async def purge_trash(
    workspace_id: str, body: PurgeTrashRequest, request: Request,
) -> JSONResponse:
    """Permanently delete one trash entry, or empty the whole trash."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        await mgr.file_purge_trash(workspace_id, body.trash_id)
        return JSONResponse({"purged": True, "trash_id": body.trash_id})
    except Exception as exc:
        log.warning("purge_trash_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/files/{workspace_id}/rename")
async def rename_file(
    workspace_id: str, body: RenameFileRequest, request: Request,
) -> JSONResponse:
    """Rename (or move) a path inside the workspace.

    Both old + new paths are validated against the same /workspace
    containment rules so a rename can't relocate a file out of the
    workspace volume.
    """
    err = (
        _validate_workspace_path(body.old_path)
        or _validate_workspace_path(body.new_path)
    )
    if err:
        return JSONResponse({"error": err}, status_code=400)
    if body.old_path == body.new_path:
        return JSONResponse(
            {"error": "old_path and new_path are identical"}, status_code=400,
        )
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        if not body.overwrite:
            probe = await mgr.run_command(
                workspace_id,
                ["bash", "-c",
                 f"test -e {shlex.quote(body.new_path)} && echo EXISTS || true"],
                timeout=5.0,
            )
            if "EXISTS" in (probe or ""):
                return JSONResponse(
                    {
                        "error": "Destination already exists",
                        "code": "destination_exists",
                        "new_path": body.new_path,
                    },
                    status_code=409,
                )
        await mgr.file_rename(workspace_id, body.old_path, body.new_path)
        return JSONResponse({
            "renamed": True,
            "old_path": body.old_path,
            "new_path": body.new_path,
        })
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception as exc:
        log.warning(
            "rename_file_failed",
            workspace=workspace_id,
            old_path=body.old_path,
            new_path=body.new_path,
            error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/files/{workspace_id}/mkdir")
async def make_directory(
    workspace_id: str, body: MkdirRequest, request: Request,
) -> JSONResponse:
    """Create a directory (and parents) inside the workspace."""
    err = _validate_workspace_path(body.path)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        await mgr.file_mkdir(workspace_id, body.path)
        return JSONResponse({"created": True, "path": body.path})
    except KeyError:
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    except Exception as exc:
        log.warning(
            "mkdir_failed",
            workspace=workspace_id, path=body.path, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


#
# Companion to the dev-server reverse proxy at /api/coder/preview/...:
# this route serves workspace files directly so the user can preview
# HTML / markdown / images / PDFs / etc. without publishing a port or
# starting a process.
#
# The renderer registry lives in augmentum/coder/preview_types.py.
# Adding a new previewable extension is a single line there — both the
# route AND the frontend's context-menu eligibility check pick it up.
# ---------------------------------------------------------------------------


def _validate_preview_path(path: str) -> str | None:
    """Return an error message if the path is unsafe to serve, else None.

    Constraints:

    - Must start with ``/workspace/`` (not just ``/workspace`` — a bare
      directory has nothing to render).
    - No ``..`` segments (defeats the workspace root).
    - No backslashes (Windows-style path injection from a misbehaving
      client; the container fs is POSIX).
    """
    if not path or not path.startswith("/workspace/"):
        return "path must start with /workspace/"
    if "\\" in path:
        return "backslashes are not allowed in path"
    if ".." in path.split("/"):
        return "path may not contain .. segments"
    return None


def _preview_base_href(workspace_id: str, path: str) -> str:
    """Compute the <base href> URL for an HTML/markdown render so that
    relative asset references (sibling files, ../shared/img.png) resolve
    back through the same preview-file route.

    The trailing slash is what tells the browser "this is a directory"
    so `<img src="logo.png">` becomes ``{base}logo.png`` and not
    ``{parent_of_base}logo.png``.
    """
    parent = path.rsplit("/", 1)[0]
    # Keep the absolute-from-workspace shape; the route is keyed by the
    # full path including /workspace/, so we hand the browser the same
    # shape it'd use to fetch any sibling.
    return f"/api/coder/preview-file/{workspace_id}{parent}/"


@router.get("/api/coder/preview-types")
async def preview_types_listing(request: Request) -> JSONResponse:
    """Return the registry of previewable file types.

    The frontend fetches this once at coder-mode init and uses it to
    decide whether to show the "Preview" context-menu item for each file
    in the tree — no parallel list to maintain.

    Auth-gated like every other /api/coder/* route; the data isn't
    sensitive, but exposing it pre-auth would be a free reconnaissance
    surface ("does this server even have a coder subsystem?").
    """
    from augmentum.coder.preview_types import extensions_by_kind, list_extensions

    if not _user_id(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse({
        "extensions": list_extensions(),
        "by_kind": extensions_by_kind(),
    })


@router.get("/api/coder/preview-file/{workspace_id}/{path:path}")
async def preview_file(workspace_id: str, path: str, request: Request) -> Response:
    """Serve a workspace file rendered for in-iframe preview.

    Unlike the dev-server reverse proxy at /api/coder/preview/{ws}/{port}/...,
    this route doesn't require a running process inside the container —
    it just reads the file and applies the renderer registered in
    ``preview_types.py``. Use cases: AI-generated harness.html, a README
    you're polishing, a design SVG, a screenshot, a PDF spec.

    Path semantics:

    - The path arrives WITHOUT the leading slash (FastAPI strips it from
      a ``{path:path}`` param), so we prepend ``/`` before validating
      and reading. Real on-disk paths look like ``/workspace/foo/bar.html``.
    - Relative asset references inside HTML/markdown resolve via a
      ``<base href>`` tag injected at render time, pointing at this
      same route under the file's parent directory.
    - Unknown extensions return 415, not 404 — distinguishing "file is
      there but I won't serve it" from "file doesn't exist".
    """
    from augmentum.coder.preview_types import by_extension, extension_for_path

    # FastAPI strips the leading slash from a ``{path:path}`` capture, so
    # rebuild the absolute path we'll hand to the container fs adapter.
    abs_path = "/" + path if not path.startswith("/") else path
    err = _validate_preview_path(abs_path)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    ext = extension_for_path(abs_path)
    type_info = by_extension(ext) if ext else None
    if type_info is None:
        return JSONResponse(
            {"error": f"file type not previewable: {ext or '(no extension)'}"},
            status_code=415,
        )
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        data = await mgr.file_download(workspace_id, abs_path)
    except FileNotFoundError:
        return JSONResponse({"error": "File not found"}, status_code=404)
    except Exception as exc:
        log.warning(
            "preview_file_read_failed",
            workspace=workspace_id, path=abs_path, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)

    # Base href only matters for renderers that consume it (HTML, markdown).
    # The registry signature accepts it unconditionally; passthrough
    # renderers ignore it.
    base_href = _preview_base_href(workspace_id, abs_path)
    rendered = type_info.renderer(data, base_href)
    return Response(
        content=rendered,
        media_type=type_info.media_type,
        # Short private cache so the iframe doesn't re-fetch every
        # tile-resize / focus event. The agent edits files
        # constantly, so a long cache would feel stale; 30s is the
        # same shape used by other workspace file routes.
        headers={
            "Cache-Control": "private, max-age=30",
            # X-Content-Type-Options stops browsers from MIME-sniffing
            # a text file as HTML — relevant for ``.txt``/``.log``
            # served as text/html via the wrapper renderer, but also
            # defense-in-depth across the surface.
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/api/coder/workspaces/{workspace_id}/export")
async def export_workspace(workspace_id: str, request: Request) -> Response:
    """Stream a gzipped tar of /workspace as an attachment.

    Excludes common heavy dep/build dirs (node_modules, .venv, etc.)
    by default — pass ``?include_deps=1`` to get everything.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    include_deps = request.query_params.get("include_deps") == "1"
    excludes = [] if include_deps else None

    try:
        info = await mgr._get_workspace(workspace_id)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    ws_name = (info.name or "workspace").replace('"', '').replace("/", "-")
    filename = f"{ws_name}.tar.gz"

    # Peek the first chunk BEFORE committing the 200 + attachment headers.
    # A StreamingResponse can't change its status once the body starts, so a
    # generator that raises (or yields nothing) mid-stream used to be served
    # as a silent 0-byte "success" — the browser saved an empty .tar.gz and
    # nothing told the user it failed. Materializing the first chunk here lets
    # a startup failure (stopped container, missing volume) surface as a real
    # error status instead.
    agen = mgr.workspace_archive_stream(workspace_id, excludes=excludes)
    try:
        first = await agen.__anext__()
    except StopAsyncIteration:
        log.warning("export_stream_empty", workspace=workspace_id)
        await agen.aclose()
        return JSONResponse(
            {"error": "Export produced no data (workspace volume empty or unavailable)."},
            status_code=500,
        )
    except Exception as exc:
        log.warning("export_stream_failed", workspace=workspace_id, error=str(exc))
        await agen.aclose()
        return JSONResponse({"error": f"Export failed: {exc}"}, status_code=500)

    async def _stream():
        try:
            yield first
            async for chunk in agen:
                yield chunk
        except Exception as exc:
            # Past the first byte the 200 is already sent — we can't signal an
            # error status, but truncating loudly beats a silent partial file.
            log.warning("export_stream_failed_midway", workspace=workspace_id, error=str(exc))
        finally:
            await agen.aclose()

    return StreamingResponse(
        _stream(),
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/api/coder/workspaces/import")
async def import_workspace(
    request: Request,
    name: str = Form(...),
    tooling_profile: str = Form(default="browser"),
    archive: UploadFile = File(...),
) -> JSONResponse:
    """Create a workspace from an Augmentum ``.tar.gz`` export.

    Round-trip counterpart to ``/export`` — spins up a fresh container,
    then extracts the archive into ``/workspace`` so file state is
    restored exactly. The new workspace gets a fresh id, container, and
    volume (volume-level copy is a separate problem).
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    cleaned_name = (name or "").strip() or "imported"
    if len(cleaned_name) > 80:
        return JSONResponse(
            {"error": "Name too long (max 80 chars)"}, status_code=400,
        )

    # Cap upload size in line with other workspace upload routes; the
    # bound here is also a soft sanity check that a stray non-archive
    # file (a .iso, a video) doesn't get half-uploaded before we reject.
    MAX_BYTES = 500 * 1024 * 1024  # 500 MB
    archive_bytes = await archive.read()
    if len(archive_bytes) > MAX_BYTES:
        return JSONResponse(
            {"error": f"Archive exceeds {MAX_BYTES // (1024 * 1024)} MB limit"},
            status_code=413,
        )
    if not archive_bytes:
        return JSONResponse({"error": "Archive is empty"}, status_code=400)

    info = None
    try:
        info = await mgr.create_workspace(
            name=cleaned_name,
            tooling_profile=tooling_profile,
            user_id=uid,
        )
        await mgr.import_archive_into(info.id, archive_bytes)
        return JSONResponse(info.__dict__, status_code=201)
    except ValueError as exc:
        # Archive validation failure — invalid gzip, empty, etc. The
        # workspace was already created; tear it down so a bad upload
        # doesn't leave a half-imported skeleton row behind.
        if info is not None:
            try:
                await mgr.delete(info.id, keep_volume=False)
            except Exception:
                log.warning(
                    "import_cleanup_failed",
                    workspace=info.id if info else None,
                    exc_info=True,
                )
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        log.warning("import_workspace_failed", error=str(exc))
        if info is not None:
            try:
                await mgr.delete(info.id, keep_volume=False)
            except Exception:
                log.warning("import_cleanup_failed", exc_info=True)
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Codebase Index
# ---------------------------------------------------------------------------


@router.post("/api/coder/index/{workspace_id}")
async def build_codebase_index(workspace_id: str, request: Request) -> JSONResponse:
    """Build or update the semantic codebase index for a workspace."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        from augmentum.coder.indexer import build_index
        force = request.query_params.get("force", "").lower() == "true"
        # Piggyback the code-intel (symbol) index on the same trigger —
        # background so the semantic build's latency doesn't stack.
        from augmentum.config import settings as _settings
        if getattr(_settings, "coder_code_intel_enabled", True):
            import asyncio as _asyncio

            from augmentum.coder import code_intel as _ci
            _t = _asyncio.create_task(_ci.build_code_intel(mgr, workspace_id, force=force))
            _t.add_done_callback(lambda t: t.cancelled() or t.exception())
        stats = await build_index(mgr, workspace_id, force=force)
        return JSONResponse(stats)
    except Exception as exc:
        log.warning("build_index_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/coder/index/{workspace_id}/progress")
async def codebase_index_progress(workspace_id: str, request: Request) -> JSONResponse:
    """Live progress of the in-flight (or last) index build for the file UI.

    Cheap in-memory read — the UI polls this ~1s while a build runs and
    stops once ``state`` is ``done`` (or ``idle`` when nothing has run).
    """
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    from augmentum.coder.indexer import get_index_progress
    return JSONResponse(get_index_progress(workspace_id) or {"state": "idle"})


@router.get("/api/coder/search/{workspace_id}")
async def search_codebase(workspace_id: str, request: Request) -> JSONResponse:
    """Semantic search across the indexed codebase."""
    q = request.query_params.get("q", "")
    if not q:
        return JSONResponse({"error": "q query param required"}, status_code=400)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    limit = int(request.query_params.get("limit", "10"))
    try:
        from augmentum.coder.indexer import search_index
        results = await search_index(workspace_id, q, limit=limit)
        return JSONResponse({"results": [
            {"file_path": r.file_path, "start_line": r.start_line,
             "end_line": r.end_line, "score": round(r.score, 3),
             "content": r.content[:500]}
            for r in results
        ]})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/coder/workspaces/{workspace_id}/search-text")
async def search_workspace_text_route(
    workspace_id: str, request: Request,
) -> JSONResponse:
    """Literal / regex text search across the live working tree.

    Backs the Files-panel search pane. Complements
    ``/api/coder/search/{workspace_id}`` (the semantic index): this leg
    greps the files as they exist RIGHT NOW — exact strings, no index
    staleness — so a file the agent wrote seconds ago is searchable.

    Query params:
      - ``q``: the search text (required).
      - ``regex``: ``1`` to treat ``q`` as a regex; default literal.
      - ``case``: ``1`` for case-sensitive; default insensitive.
      - ``glob``: optional file filter, e.g. ``*.py``.
      - ``limit``: max matches (default 500, capped at 2000).

    Response is structured per-match (path/line/text/highlight spans)
    with an explicit ``truncated`` flag — the UI never has to guess
    whether it saw everything.
    """
    qp = request.query_params
    q = qp.get("q") or ""
    if not q:
        return JSONResponse({"error": "q query param required"}, status_code=400)
    if len(q) > 1000:
        return JSONResponse({"error": "query too long"}, status_code=400)
    glob = (qp.get("glob") or "").strip()
    if "\x00" in glob or "\n" in glob:
        return JSONResponse({"error": "invalid glob"}, status_code=400)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    truthy = ("1", "true", "yes")
    try:
        from augmentum.coder.text_search import search_workspace_text
        result = await search_workspace_text(
            mgr, workspace_id, q,
            regex=(qp.get("regex") or "").strip() in truthy,
            case_sensitive=(qp.get("case") or "").strip() in truthy,
            glob=glob,
            max_results=qp.get("limit") or 500,
        )
        return JSONResponse(result)
    except Exception as exc:
        log.warning(
            "search_text_failed",
            workspace=workspace_id, query=q[:80], error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Checkpoints (git auto-commit per agent action)
# ---------------------------------------------------------------------------


@router.get("/api/coder/checkpoints/{workspace_id}")
async def list_checkpoints(workspace_id: str, request: Request, limit: int = 20) -> JSONResponse:
    """Get the checkpoint timeline for a workspace."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        entries = await mgr.git_log(workspace_id, limit=limit)
        return JSONResponse({"checkpoints": entries})
    except Exception as exc:
        log.warning("list_checkpoints_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/checkpoints/{workspace_id}")
async def create_checkpoint(workspace_id: str, request: Request) -> JSONResponse:
    """Manually stamp a named checkpoint capturing the current workspace state.

    Useful before a risky terminal operation or after manual edits the
    user wants preserved. Returns {checkpoint: "<hash>" | None}; None
    means there was nothing uncommitted to capture.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        body = await request.json()
    except Exception:
        body = {}
    message = (body.get("message") or "").strip() or "Manual checkpoint"
    # Cap the message so a huge payload can't bloat the git log.
    if len(message) > 200:
        message = message[:200]
    try:
        sha = await mgr.git_checkpoint(workspace_id, message)
        return JSONResponse({"checkpoint": sha, "message": message})
    except Exception as exc:
        log.warning("manual_checkpoint_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/checkpoints/{workspace_id}/revert")
async def revert_checkpoint(workspace_id: str, request: Request) -> JSONResponse:
    """Revert a workspace to a specific checkpoint."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    body = await request.json()
    commit_hash = body.get("hash", "")
    if not commit_hash:
        return JSONResponse({"error": "hash required"}, status_code=400)
    try:
        ok = await mgr.git_revert(workspace_id, commit_hash)
        if ok:
            return JSONResponse({"status": "reverted", "to": commit_hash})
        return JSONResponse({"error": "revert failed"}, status_code=500)
    except Exception as exc:
        log.warning("revert_checkpoint_failed", workspace=workspace_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/coder/checkpoints/{workspace_id}/show")
async def checkpoint_show(workspace_id: str, request: Request) -> JSONResponse:
    """Return a single commit's own diff (git show) for the history browser.

    Distinct from ``/diff`` (commit-vs-current): this is what changed IN
    the commit. The response splits the metadata header (full hash /
    author / unix time / subject) from the unified-diff body so the UI
    can render both without re-parsing.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    commit_hash = (request.query_params.get("hash") or "").strip()
    # Hashes are hex (optionally short). Reject anything else before it
    # reaches the shell arg — defense in depth atop the literal-arg pass.
    if not re.fullmatch(r"[0-9a-fA-F]{4,40}", commit_hash):
        return JSONResponse({"error": "valid commit hash required"}, status_code=400)
    try:
        raw = await mgr.git_show(workspace_id, commit_hash)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    # git show with our --format prints 4 header lines, then a blank line,
    # then the diff. Split defensively — a merge/empty commit may vary.
    lines = (raw or "").split("\n")
    meta: dict = {"hash": commit_hash}
    body_start = 0
    if len(lines) >= 4:
        meta = {
            "hash": lines[0].strip() or commit_hash,
            "author": lines[1].strip(),
            "timestamp": int(lines[2]) if lines[2].strip().isdigit() else 0,
            "subject": lines[3].strip(),
        }
        body_start = 4
        if body_start < len(lines) and lines[body_start].strip() == "":
            body_start += 1
    diff_body = "\n".join(lines[body_start:])
    truncated = False
    if len(diff_body) > _GIT_DIFF_MAX_BYTES:
        diff_body = diff_body[:_GIT_DIFF_MAX_BYTES] + (
            f"\n\n... (truncated, {len(diff_body)} bytes — open the file for full content)"
        )
        truncated = True
    return JSONResponse({
        "hash": commit_hash, "meta": meta,
        "diff": diff_body, "truncated": truncated,
    })


@router.get("/api/coder/checkpoints/{workspace_id}/diff")
async def checkpoint_diff(workspace_id: str, request: Request) -> JSONResponse:
    """Get the diff between a checkpoint and current state."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    commit_hash = request.query_params.get("hash", "")
    if not commit_hash:
        return JSONResponse({"error": "hash query param required"}, status_code=400)
    try:
        diff = await mgr.git_diff(workspace_id, commit_hash)
        return JSONResponse({"diff": diff, "hash": commit_hash})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Conversation persistence
# ---------------------------------------------------------------------------


@router.get("/api/coder/conversation/{workspace_id}")
async def get_conversation(workspace_id: str, request: Request) -> JSONResponse:
    """Load conversation history for a workspace the caller owns."""
    persistence = await _get_coder_persistence(request)
    if persistence is None:
        return JSONResponse({"messages": []})
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"messages": []})
    uid = _user_id(request)
    try:
        return JSONResponse({
            "messages": await persistence.load_conversation(
                workspace_id,
                user_id=uid,
            ),
        })
    except Exception as exc:
        # A load failure is NOT an empty conversation — returning [] with
        # a 200 makes the UI silently offer "start a new conversation"
        # and the user loses history they actually have. Surface it so
        # the client can show a retry instead of a clean slate.
        log.warning("get_conversation_failed", error=str(exc), exc_info=True)
        return JSONResponse(
            {"error": "conversation_load_failed", "messages": []},
            status_code=503,
        )


@router.delete("/api/coder/conversation/{workspace_id}")
async def clear_conversation(workspace_id: str, request: Request) -> JSONResponse:
    """Wipe the conversation history for a workspace the caller owns.

    Frontend surfaces this as the ``/clear`` slash command. Intent is
    "fresh session, keep the workspace files intact" — the message log
    is emptied but the container and any files on disk are untouched.
    Per-session agent state that persists at the row level (plan,
    tasks, plan_steps) also clears so the next turn doesn't inherit
    stale goals. ``turn_summaries``, ``files_read``, and
    ``tool_calls_made`` live only in in-memory ``CoderState`` which is
    rebuilt per request, so they already reset naturally.
    """
    persistence = await _get_coder_persistence(request)
    if persistence is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    uid = _user_id(request)
    try:
        await persistence.delete_session(workspace_id, user_id=uid)
        return JSONResponse({"ok": True})
    except Exception as exc:
        log.warning("clear_conversation_failed", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/conversation/{workspace_id}/compact")
async def compact_conversation(
    workspace_id: str,
    body: ConversationCompactRequest,
    request: Request,
) -> JSONResponse:
    """Deterministically compact the saved Coder conversation.

    Frontend surfaces this as ``/compact``. The endpoint accepts the
    caller's current in-memory messages so a just-sent turn can be
    compacted before the periodic save catches up; when omitted, it
    falls back to the persisted conversation.
    """
    persistence = await _get_coder_persistence(request)
    if persistence is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    uid = _user_id(request)
    try:
        from augmentum.coder.context_tokens import (
            coder_context_token_limit,
            compact_conversation_messages,
        )

        context_window = 0
        registry = getattr(request.app.state, "provider_registry", None)
        if registry is not None:
            try:
                backend, resolved_model = await registry.resolve_backend_with_fabric(
                    body.model or "",
                )
                get_context_length = getattr(backend, "get_context_length", None)
                if callable(get_context_length):
                    context_window = int(
                        await get_context_length(resolved_model or body.model or "")
                        or 0
                    )
            except Exception:
                log.debug(
                    "coder_compact_context_length_probe_failed",
                    model=body.model,
                    exc_info=True,
                )
                context_window = 0
        limit = coder_context_token_limit(context_window)
        messages = body.messages
        if messages is None:
            messages = await persistence.load_conversation(
                workspace_id,
                user_id=uid,
            )
        result = compact_conversation_messages(
            messages or [],
            keep_recent=body.keep_recent,
            force=body.force,
            limit=limit,
        )
        if result.compacted:
            await persistence.save_conversation(
                workspace_id,
                result.messages,
                user_id=uid,
            )
        token_payload = result.to_payload(limit=limit)
        if context_window > 0:
            token_payload["context_window"] = context_window
        return JSONResponse({
            "ok": True,
            "compacted": result.compacted,
            "messages": result.messages,
            "tokens": token_payload,
        })
    except Exception as exc:
        log.warning("compact_conversation_failed", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.api_route("/api/coder/conversation/{workspace_id}", methods=["PUT", "POST"])
async def save_conversation(workspace_id: str, request: Request) -> JSONResponse:
    """Save conversation history for a workspace the caller owns.

    Accepts POST in addition to PUT so ``navigator.sendBeacon`` (which is
    hardwired to POST) can reach this endpoint during page unload — that
    path is the only way to persist the conversation when the user closes
    the tab mid-session.
    """
    persistence = await _get_coder_persistence(request)
    if persistence is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    uid = _user_id(request)
    try:
        body = await request.json()
        messages = body.get("messages", [])
        await persistence.save_conversation(workspace_id, messages, user_id=uid)
        return JSONResponse({"ok": True})
    except Exception as exc:
        log.warning("save_conversation_failed", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Git Token endpoints
# ---------------------------------------------------------------------------


@router.post("/api/coder/git-tokens")
async def store_git_token(body: GitTokenRequest, request: Request) -> JSONResponse:
    """Store a git credential token for a host."""
    store = await _get_git_store(request)
    if store is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    await store.set_token(
        body.host,
        body.token,
        username=body.username,
        user_id=uid,
    )
    return JSONResponse({"status": "ok", "host": body.host})


@router.get("/api/coder/git-tokens")
async def list_git_tokens(request: Request) -> JSONResponse:
    """List configured git hosts (tokens redacted)."""
    store = await _get_git_store(request)
    if store is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    tokens = await store.list_tokens(user_id=uid)
    return JSONResponse({"tokens": tokens})


@router.delete("/api/coder/git-tokens/{host}")
async def delete_git_token(host: str, request: Request) -> JSONResponse:
    """Remove a git token for a host."""
    store = await _get_git_store(request)
    if store is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    await store.delete_token(host, user_id=uid)
    return JSONResponse({"status": "deleted", "host": host})


@router.get("/api/coder/git-credential")
async def git_credential_proxy(
    request: Request,
    host: str = "",
    workspace_id: str = "",
) -> JSONResponse:
    """Credential helper callback — returns token for a git host.
    Only responds to requests from Docker-internal networks.
    """
    client_ip = request.client.host if request.client else ""
    if not _is_docker_internal(client_ip):
        log.warning("git_credential_blocked", client_ip=client_ip, host=host)
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    if not host:
        return JSONResponse({"error": "Missing host parameter"}, status_code=400)

    store = await _get_git_store(request)
    if store is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    owner_id = ""
    if workspace_id:
        owner_id = await _workspace_owner_id(request, workspace_id)

    token_data = await store.get_token(host, user_id=owner_id)
    if token_data is None:
        return JSONResponse({"error": f"No token for {host}"}, status_code=404)

    # Return the raw token as plain text. JSONResponse would wrap it in
    # quotes — e.g. body `"ghp_abc..."` — which the shell credential
    # helper would then pass to git as the password *including* the
    # literal quote characters, causing auth to fail for users who had
    # actually stored a valid token. PlainTextResponse emits just the
    # token bytes.
    return PlainTextResponse(token_data["token"], status_code=200)


# ---------------------------------------------------------------------------
# Git Operations
# ---------------------------------------------------------------------------


@router.get("/api/coder/workspaces/{workspace_id}/git/status")
async def git_status(workspace_id: str, request: Request) -> JSONResponse:
    """Get git branch, dirty state, remote URL, and recent log."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        script = (
            "cd /workspace && "
            "echo \"BRANCH=$(git branch --show-current 2>/dev/null || echo main)\" && "
            "echo \"DIRTY=$(git status --porcelain 2>/dev/null | wc -l)\" && "
            "echo \"REMOTE=$(git remote get-url origin 2>/dev/null || echo '')\" && "
            "echo \"AHEAD=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo 0)\" && "
            "echo \"BEHIND=$(git rev-list --count HEAD..@{u} 2>/dev/null || echo 0)\" && "
            "echo 'LOG_START' && git log --oneline -5 2>/dev/null && echo 'LOG_END'"
        )
        output = await mgr._run_command(workspace_id, ["bash", "-c", script], timeout=10.0)

        # Parse key=value pairs from output
        result: dict = {"branch": "main", "dirty": False, "remote": "", "ahead": 0, "behind": 0, "log": []}
        for line in output.strip().split("\n"):
            if line.startswith("BRANCH="):
                result["branch"] = line.split("=", 1)[1]
            elif line.startswith("DIRTY="):
                result["dirty"] = int(line.split("=", 1)[1]) > 0
            elif line.startswith("REMOTE="):
                result["remote"] = line.split("=", 1)[1]
            elif line.startswith("AHEAD="):
                result["ahead"] = int(line.split("=", 1)[1])
            elif line.startswith("BEHIND="):
                result["behind"] = int(line.split("=", 1)[1])

        # Extract log lines between markers
        if "LOG_START" in output and "LOG_END" in output:
            log_section = output.split("LOG_START")[1].split("LOG_END")[0].strip()
            if log_section:
                result["log"] = log_section.split("\n")

        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# git status --porcelain parser shared by the per-file decoration
# endpoint. Maps the two-letter index/worktree status pair to a single
# user-visible flag for the file tree badge.
#
# Codes (from man git-status):
#   ' M' = modified, unstaged
#   'M ' = modified, staged
#   'MM' = modified both
#   '??' = untracked
#   'A ' = added (staged)
#   ' D' = deleted, unstaged
#   'D ' = deleted, staged
#   'R ' = renamed (porcelain emits "old -> new")
#   'UU' / 'AA' / 'DD' = conflicted
# We collapse to a single character so the UI badge stays consistent.
def _collapse_porcelain_status(xy: str) -> str:
    if not xy or len(xy) < 2:
        return ""
    x, y = xy[0], xy[1]
    # Conflict markers are uppercase pairs.
    if (x == "U" or y == "U") or (x == "A" and y == "A") or (x == "D" and y == "D"):
        return "C"
    if x == "?" or y == "?":
        return "U"
    if x == "R" or y == "R":
        return "R"
    if x == "D" or y == "D":
        return "D"
    if x == "A" or y == "A":
        return "A"
    if x == "M" or y == "M":
        return "M"
    return ""


@router.get("/api/coder/workspaces/{workspace_id}/git/file-status")
async def git_file_status(workspace_id: str, request: Request) -> JSONResponse:
    """Per-file git status for the file-tree decoration layer.

    Returns ``{files: [{path, status}], head_oid}`` where ``status`` is
    one of M / A / D / U / R / C and ``path`` is the absolute path
    inside the workspace volume so the UI can match it against the
    file-tree row's ``data-path`` attribute without normalization.

    Empty list when the workspace isn't a git repo or the working
    tree is clean. Best-effort: returns 200 with an empty list rather
    than 500 on transient git failures so a momentary error doesn't
    blank the decorations.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        # ``-z`` would be safer for filenames with newlines, but the
        # porcelain v1 format reliably terminates with `\n` for the
        # paths we see in coder workspaces; sticking with the simpler
        # parser. If a future workspace has weird names we can switch.
        script = (
            "cd /workspace && "
            "git status --porcelain 2>/dev/null && "
            "echo HEAD_OID=$(git rev-parse HEAD 2>/dev/null || echo '')"
        )
        output = await mgr._run_command(
            workspace_id, ["bash", "-c", script], timeout=10.0,
        )
    except Exception:
        # No container, no git, etc. — return empty rather than 500.
        return JSONResponse({"files": [], "head_oid": ""})

    files: list[dict] = []
    head_oid = ""
    for raw in (output or "").splitlines():
        if raw.startswith("HEAD_OID="):
            head_oid = raw.split("=", 1)[1].strip()
            continue
        if len(raw) < 4:
            continue
        # Porcelain v1 shape: "XY path" — exactly one space after the
        # two-letter status code.
        xy = raw[:2]
        rest = raw[3:]
        status = _collapse_porcelain_status(xy)
        if not status:
            continue
        # Rename entries are "old -> new"; surface the new path so the
        # decoration lands on the row the user actually sees.
        if " -> " in rest:
            rest = rest.split(" -> ", 1)[1]
        # Strip the porcelain quoting (active when a path contains
        # special chars). Best-effort — we don't unescape every C-style
        # sequence, just the wrapping quotes.
        if rest.startswith("\"") and rest.endswith("\""):
            rest = rest[1:-1]
        files.append({
            "path": f"/workspace/{rest}" if not rest.startswith("/") else rest,
            "status": status,
        })

    return JSONResponse({"files": files, "head_oid": head_oid})


# Defaults that approximate "what a developer wants to see in a fuzzy
# file finder" — skip vendored / generated / caches. Match against
# path components so prune is fast and predictable.
_FILES_FLAT_EXCLUDE_DIRS = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".next", ".cache",
    "target", ".gradle", ".idea", ".vscode",
})


@router.get("/api/coder/workspaces/{workspace_id}/files-flat")
async def files_flat(workspace_id: str, request: Request) -> JSONResponse:
    """Flat list of all workspace files for the command palette.

    Returns ``{files: [{path, name}]}`` capped at 5000 entries so a
    pathological workspace can't choke the palette. The exclude list
    skips vendored + generated dirs that a fuzzy finder would only
    pollute its results with.

    Uses ``find`` in the container — fast even on large workspaces
    because we ``-prune`` heavy dirs and bail at ``-print`` not
    ``-name``-filter.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    # Build the find expression: prune known-uninteresting dirs by
    # NAME (matches at any depth), then print everything else that's
    # a regular file.
    prune_clauses = " -o ".join(
        f"-name {name!r}" for name in sorted(_FILES_FLAT_EXCLUDE_DIRS)
    )
    script = (
        f"cd /workspace && find . "
        f"-type d \\( {prune_clauses} \\) -prune "
        f"-o -type f -print 2>/dev/null | head -5000"
    )
    try:
        output = await mgr._run_command(
            workspace_id, ["bash", "-c", script], timeout=15.0,
        )
    except Exception as exc:
        log.warning(
            "files_flat_failed", workspace=workspace_id, error=str(exc),
        )
        return JSONResponse({"files": []})

    files: list[dict] = []
    for raw in (output or "").splitlines():
        rel = raw.strip()
        if not rel or rel == ".":
            continue
        # Strip the leading "./" find emits.
        if rel.startswith("./"):
            rel = rel[2:]
        name = rel.rsplit("/", 1)[-1] if "/" in rel else rel
        # Hidden files at the top level often aren't what the user
        # wants in a fuzzy finder (.env, .DS_Store) — keep nested
        # hidden files like .github/workflows/foo.yml so config and
        # CI files remain findable.
        if name.startswith(".") and "/" not in rel:
            continue
        files.append({
            "path": f"/workspace/{rel}",
            "name": name,
        })

    return JSONResponse({"files": files, "truncated": len(files) >= 5000})


# Patterns that indicate the push/pull failed for credential or
# host-trust reasons. Matched case-insensitively against the combined
# stderr+stdout of the git invocation. The UI auto-opens the settings
# modal on a 401, so a false negative here leaves the user staring at
# a generic error with no path forward — keep this list inclusive.
_GIT_AUTH_FAILURE_PHRASES = (
    "authentication failed",
    "could not read username",
    "could not read password",
    "fatal: authentication",
    "invalid credentials",
    "bad credentials",
    "remote: invalid username or password",
    "401 unauthorized",
    "http/1.1 401",
    "http/2 401",
    "403 forbidden",
    "permission denied (publickey",
    "permission denied (password",
    "permission to ",  # matches "Permission to user/repo denied to other-user"
    "no supported authentication methods",
    "host key verification failed",
    "are you sure you want to continue connecting",
    "407 proxy authentication required",
)
_GIT_AUTH_CODE_TOKENS = (" 401 ", " 403 ", " 407 ", "[401]", "[403]", "[407]")


def _is_git_auth_failure(text: str) -> bool:
    """True when ``text`` looks like a git auth / host-trust failure.

    Substring-only — patterns are picked to be specific enough that a
    successful operation's prose doesn't accidentally match. The
    "permission to " phrase is intentionally broad to catch GitHub's
    standard rejection message regardless of repo/user names.
    """
    lowered = (text or "").lower()
    for phrase in _GIT_AUTH_FAILURE_PHRASES:
        if phrase in lowered:
            return True
    for code in _GIT_AUTH_CODE_TOKENS:
        if code in lowered:
            return True
    return False


@router.post("/api/coder/workspaces/{workspace_id}/git/push")
async def git_push(workspace_id: str, request: Request) -> JSONResponse:
    """Auto-commit uncommitted changes and push to remote.

    Auth-failure detection covers both:
      - exceptions raised by ``_run_command`` (subprocess failure)
      - successful command output that contains a protocol-level
        rejection (``git push`` exits 0 with the rejection in stdout)

    Both paths feed ``_is_git_auth_failure`` so the UI consistently
    gets a 401 + structured error when credentials are at fault.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        await mgr._run_command(workspace_id, ["bash", "-c",
            "cd /workspace && git add -A && "
            "git diff --cached --quiet 2>/dev/null || git commit -m 'Auto-commit before push'"
        ], timeout=10.0)
        output = await mgr._run_command(workspace_id, ["bash", "-c",
            "cd /workspace && git push -u origin HEAD 2>&1"
        ], timeout=60.0)
        if _is_git_auth_failure(output):
            return JSONResponse(
                {
                    "error": "Authentication failed — configure a git token in settings",
                    "raw": output.strip(),
                },
                status_code=401,
            )
        return JSONResponse({"status": "ok", "output": output.strip()})
    except Exception as exc:
        error = str(exc)
        if _is_git_auth_failure(error):
            return JSONResponse(
                {
                    "error": "Authentication failed — configure a git token in settings",
                    "raw": error,
                },
                status_code=401,
            )
        return JSONResponse({"error": error}, status_code=500)


def _classify_pull_failure(text: str) -> tuple[str, str]:
    """Map raw git pull output to (error_code, human_message).

    Returns ``("", "")`` when the output doesn't look like a known
    failure shape — caller falls through to the raw error path.
    Codes are stable so the UI can branch on them (open settings
    modal, show stash button, etc.) instead of re-pattern-matching.
    """
    lowered = (text or "").lower()
    if _is_git_auth_failure(text):
        return "auth", "Authentication failed — configure a git token in settings."
    # Dirty-tree refusal — the canonical "would be overwritten" message
    # appears verbatim in every dirty-tree pull I've checked. Catching
    # the substring is enough.
    if (
        "would be overwritten by merge" in lowered
        or "your local changes to the following files would be overwritten" in lowered
        or "please commit your changes or stash them" in lowered
    ):
        return (
            "dirty_tree",
            "You have uncommitted local changes that would conflict with the pull. "
            "Commit them first (or stash if you don't want to keep them) and try again.",
        )
    if "not possible to fast-forward" in lowered or "non-fast-forward" in lowered:
        return (
            "non_fast_forward",
            "Remote has diverged from your branch. A merge or rebase is needed — "
            "open a terminal and run ``git pull --rebase`` or ``git merge`` once "
            "you've reviewed the incoming commits.",
        )
    if "couldn't find remote ref" in lowered or "remote ref does not exist" in lowered:
        return (
            "no_upstream",
            "The remote doesn't have this branch yet. Push it first, or set an "
            "upstream via Git Settings.",
        )
    if (
        "could not resolve host" in lowered
        or "name or service not known" in lowered
        or "no route to host" in lowered
    ):
        return (
            "network",
            "Couldn't reach the remote — check your network or the remote URL.",
        )
    return "", ""


@router.post("/api/coder/workspaces/{workspace_id}/git/pull")
async def git_pull(workspace_id: str, request: Request) -> JSONResponse:
    """Pull latest changes from remote.

    ``--ff-only`` deliberately refuses non-fast-forward merges so the
    UI button can't silently produce a merge commit the user didn't
    intend. Common failure shapes (dirty tree, diverged branches,
    auth) are mapped to structured ``error_code`` values so the UI
    can branch on them without re-pattern-matching the raw output.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        output = await mgr._run_command(workspace_id, ["bash", "-c",
            "cd /workspace && git pull --ff-only 2>&1"
        ], timeout=60.0)
        code, msg = _classify_pull_failure(output)
        if code:
            status_code = 401 if code == "auth" else 409
            return JSONResponse(
                {"error": msg, "error_code": code, "raw": output.strip()},
                status_code=status_code,
            )
        return JSONResponse({"status": "ok", "output": output.strip()})
    except Exception as exc:
        text = str(exc)
        code, msg = _classify_pull_failure(text)
        if code:
            status_code = 401 if code == "auth" else 409
            return JSONResponse(
                {"error": msg, "error_code": code, "raw": text},
                status_code=status_code,
            )
        return JSONResponse({"error": text}, status_code=500)


@router.post("/api/coder/workspaces/{workspace_id}/git/remote")
async def set_git_remote(workspace_id: str, body: GitRemoteRequest, request: Request) -> JSONResponse:
    """Set or update the origin remote URL."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        # body.url goes through bash -c, so escape it as a single shell arg.
        # Without this, a URL containing `'` could break out and execute
        # arbitrary commands inside the workspace container.
        safe_url = shlex.quote(body.url)
        await mgr._run_command(workspace_id, ["bash", "-c",
            f"cd /workspace && (git remote set-url origin {safe_url} 2>/dev/null || "
            f"git remote add origin {safe_url})"
        ], timeout=10.0)
        return JSONResponse({"status": "ok", "remote": body.url})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/workspaces/{workspace_id}/git/commit")
async def git_commit(workspace_id: str, body: GitCommitRequest, request: Request) -> JSONResponse:
    """Commit currently-staged changes.

    Pre-2026-05-31 this auto-staged every modified file with
    ``git add -A`` before committing — which silently shipped
    untracked files (``.env``, build artifacts) the user hadn't
    reviewed. New behaviour: commit ONLY what's already staged. If
    nothing is staged the route refuses (409) and points the caller
    at the stage endpoint or the commit panel's checkboxes.

    Callers who want the old auto-stage behaviour can call
    ``/git/stage`` with the path list first, then ``/git/commit``.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    msg = (body.message or "").strip()
    if not msg:
        return JSONResponse(
            {"error": "Commit message is required."}, status_code=400,
        )
    try:
        # Refuse the commit when nothing's staged so the user gets a
        # clear error instead of a confusing "nothing to commit" from
        # git after they thought they staged something.
        staged_check = await mgr._run_command(
            workspace_id,
            ["bash", "-c", "cd /workspace && git diff --cached --quiet 2>&1; echo $?"],
            timeout=5.0,
        )
        if (staged_check or "").strip().endswith("0"):
            return JSONResponse(
                {
                    "error": "Nothing is staged. Stage files first via the commit panel "
                             "or POST /git/stage.",
                    "error_code": "nothing_staged",
                },
                status_code=409,
            )
        safe_msg = msg.replace("'", "'\\''")
        output = await mgr._run_command(workspace_id, ["bash", "-c",
            f"cd /workspace && git commit -m '{safe_msg}' 2>&1"
        ], timeout=10.0)
        return JSONResponse({"status": "ok", "output": output.strip()})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# Maximum diff output size returned to the UI. Big enough for normal
# review-before-commit but capped so a single huge generated file
# can't spike memory or render time.
_GIT_DIFF_MAX_BYTES = 256 * 1024


@router.get("/api/coder/workspaces/{workspace_id}/git/diff")
async def git_diff(workspace_id: str, request: Request) -> JSONResponse:
    """Return the unstaged or staged diff for the commit-review pane.

    Query params:
      - ``staged``: ``"1"`` for the index diff (about-to-commit),
        anything else for the worktree diff (about-to-stage).
      - ``path``: optional single-file scope. Validated against
        ``_validate_workspace_path`` so the diff can't be coerced to
        read outside the workspace.

    Output is capped at ``_GIT_DIFF_MAX_BYTES`` with a trailing
    ``... (truncated, N bytes total)`` marker so the UI can warn the
    user. Untracked files are surfaced as ``? <path>`` lines after
    the diff body so the panel can offer to stage them.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    staged = (request.query_params.get("staged") or "").strip() in ("1", "true", "yes")
    path = (request.query_params.get("path") or "").strip()
    path_arg = ""
    if path:
        err = _validate_workspace_path(path)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        # Convert /workspace/foo to a path relative to cwd=/workspace.
        rel = path[len("/workspace/"):] if path.startswith("/workspace/") else path
        path_arg = " -- " + shlex.quote(rel)

    cached_flag = "--cached" if staged else ""
    try:
        diff_text = await mgr._run_command(
            workspace_id,
            [
                "bash", "-c",
                f"cd /workspace && git diff {cached_flag} --no-color {path_arg} 2>&1",
            ],
            timeout=15.0,
        )
        # Untracked files don't show in ``git diff`` even with no
        # filter; surface them so the UI can present them as "ready
        # to stage" entries alongside the diff.
        untracked = await mgr._run_command(
            workspace_id,
            [
                "bash", "-c",
                "cd /workspace && git ls-files --others --exclude-standard 2>/dev/null",
            ],
            timeout=10.0,
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    truncated = False
    total_bytes = len(diff_text or "")
    if total_bytes > _GIT_DIFF_MAX_BYTES:
        diff_text = diff_text[:_GIT_DIFF_MAX_BYTES] + (
            f"\n\n... (truncated, {total_bytes} bytes total — open the file for full content)"
        )
        truncated = True

    untracked_paths = [
        f"/workspace/{line.strip()}"
        for line in (untracked or "").splitlines()
        if line.strip()
    ]

    return JSONResponse({
        "staged": staged,
        "path": path,
        "diff": diff_text,
        "truncated": truncated,
        "total_bytes": total_bytes,
        "untracked": untracked_paths,
    })


def _validate_stage_paths(paths: list[str]) -> tuple[list[str], str | None]:
    """Filter incoming stage/unstage paths through workspace path
    validation. Returns ``(rel_paths, error)`` where ``rel_paths`` are
    workspace-relative (no leading slash) so they can be passed
    directly to ``git add`` / ``git restore --staged``.
    """
    if not isinstance(paths, list) or not paths:
        return [], "paths must be a non-empty list"
    rels: list[str] = []
    for p in paths:
        if not isinstance(p, str):
            return [], "paths must be strings"
        err = _validate_workspace_path(p)
        if err:
            return [], f"{p}: {err}"
        rel = p[len("/workspace/"):] if p.startswith("/workspace/") else p
        if rel:
            rels.append(rel)
    return rels, None


@router.post("/api/coder/workspaces/{workspace_id}/git/stage")
async def git_stage(workspace_id: str, body: GitStageRequest, request: Request) -> JSONResponse:
    """Stage one or more paths (``git add``)."""
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    rels, err = _validate_stage_paths(body.paths)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    safe = " ".join(shlex.quote(r) for r in rels)
    try:
        output = await mgr._run_command(
            workspace_id,
            ["bash", "-c", f"cd /workspace && git add -- {safe} 2>&1"],
            timeout=15.0,
        )
        return JSONResponse({"status": "ok", "staged": len(rels), "output": output.strip()})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/workspaces/{workspace_id}/git/unstage")
async def git_unstage(workspace_id: str, body: GitStageRequest, request: Request) -> JSONResponse:
    """Unstage one or more paths (``git restore --staged``).

    Falls back to ``git reset HEAD --`` on initial-commit branches
    where ``git restore`` would fail because there's no HEAD yet.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    rels, err = _validate_stage_paths(body.paths)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    safe = " ".join(shlex.quote(r) for r in rels)
    try:
        output = await mgr._run_command(
            workspace_id,
            ["bash", "-c",
             f"cd /workspace && (git restore --staged -- {safe} 2>&1 || "
             f"git reset HEAD -- {safe} 2>&1)"],
            timeout=15.0,
        )
        return JSONResponse({"status": "ok", "unstaged": len(rels), "output": output.strip()})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/coder/workspaces/{workspace_id}/git/branches")
async def git_branches(workspace_id: str, request: Request) -> JSONResponse:
    """List local + remote-tracking branches, marking the current one.

    Returns ``{current, branches: [{name, current, remote_tracking, last_commit}]}``.
    Pure git ``for-each-ref`` — no ``git branch`` parsing because the
    porcelain format is more stable across git versions.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    try:
        # %(HEAD) is the asterisk marker; %(refname:short) is "main"
        # (not "refs/heads/main"); %(upstream:short) is empty when
        # unset; %(subject) is the latest commit subject — handy as
        # context in a branch picker.
        fmt = "%(HEAD)|%(refname:short)|%(upstream:short)|%(objectname:short)|%(subject)"
        output = await mgr._run_command(
            workspace_id,
            [
                "bash", "-c",
                "cd /workspace && git for-each-ref --sort=-committerdate "
                f"--format={shlex.quote(fmt)} refs/heads/ 2>&1",
            ],
            timeout=10.0,
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    branches: list[dict] = []
    current = ""
    for raw in (output or "").splitlines():
        parts = raw.split("|", 4)
        if len(parts) < 4:
            continue
        head_marker, name, upstream, sha = parts[0], parts[1], parts[2], parts[3]
        subject = parts[4] if len(parts) >= 5 else ""
        is_current = head_marker.strip() == "*"
        if is_current:
            current = name
        branches.append({
            "name": name,
            "current": is_current,
            "upstream": upstream,
            "sha": sha,
            "subject": subject,
        })

    return JSONResponse({"current": current, "branches": branches})


@router.post("/api/coder/workspaces/{workspace_id}/git/checkout")
async def git_checkout(workspace_id: str, body: GitCheckoutRequest, request: Request) -> JSONResponse:
    """Switch to an existing branch.

    Refuses when the working tree is dirty — git will too, but we
    pre-check so the error path returns a structured ``dirty_tree``
    code the UI can route to "commit or stash first" guidance instead
    of dumping the raw git error.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    branch = (body.branch or "").strip()
    if not branch:
        return JSONResponse({"error": "branch is required"}, status_code=400)
    # Branch names are validated by git itself; we just guard against
    # shell injection by quoting. ``git`` rejects leading ``-`` etc.
    safe_branch = shlex.quote(branch)
    try:
        dirty = await mgr._run_command(
            workspace_id,
            ["bash", "-c", "cd /workspace && git status --porcelain 2>/dev/null | head -1"],
            timeout=5.0,
        )
        if dirty.strip():
            return JSONResponse(
                {
                    "error": "Working tree is dirty — commit or stash your changes before switching branches.",
                    "error_code": "dirty_tree",
                },
                status_code=409,
            )
        output = await mgr._run_command(
            workspace_id,
            ["bash", "-c", f"cd /workspace && git checkout {safe_branch} 2>&1"],
            timeout=15.0,
        )
        return JSONResponse({"status": "ok", "branch": branch, "output": output.strip()})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/coder/workspaces/{workspace_id}/git/branch")
async def git_create_branch(workspace_id: str, body: GitBranchRequest, request: Request) -> JSONResponse:
    """Create a new branch (and switch to it).

    ``from`` (optional) is the source ref — defaults to current HEAD.
    Useful when you want to fork from ``main`` while sitting on a
    different branch.
    """
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    name = (body.name or "").strip()
    if not name:
        return JSONResponse({"error": "branch name is required"}, status_code=400)
    # Be conservative on the name characters — git accepts a wider set
    # but the common ones (alnum, slash, dash, underscore, dot) cover
    # every UI flow. This blocks shell metas + leading dashes that
    # would confuse ``git branch``.
    import re as _re
    if not _re.match(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,127}$", name):
        return JSONResponse(
            {
                "error": "Branch names may use letters, digits, dots, dashes, slashes, and underscores "
                         "(must start with a letter or digit).",
            },
            status_code=400,
        )
    safe_name = shlex.quote(name)
    safe_from = shlex.quote((body.from_ or "").strip()) if body.from_ else ""
    try:
        cmd = (
            f"cd /workspace && git checkout -b {safe_name} {safe_from} 2>&1"
            if safe_from
            else f"cd /workspace && git checkout -b {safe_name} 2>&1"
        )
        output = await mgr._run_command(
            workspace_id,
            ["bash", "-c", cmd],
            timeout=15.0,
        )
        return JSONResponse({"status": "ok", "branch": name, "output": output.strip()})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Live-preview capture WebSocket
# ---------------------------------------------------------------------------


@router.websocket("/ws/coder/preview-capture/{workspace_id}")
async def preview_capture_ws(websocket: WebSocket, workspace_id: str) -> None:
    """Live-preview frame channel (owner-only).

    The coder UI holds this socket open whenever a preview is showing;
    ``browser_screenshot`` uses it to grab the frame the user's real GPU
    already rendered instead of re-rendering a heavy WebGL page headless.
    See ``augmentum/coder/preview_capture.py``. The socket's mere presence
    is the "a GPU frame is available" signal.
    """
    from augmentum.coder.preview_capture import broker

    await websocket.accept()
    # Same ownership gate as terminal_ws — a tenant must not capture another
    # tenant's preview by guessing a workspace_id.
    ws_user = websocket.scope.get("user")
    ws_uid = ws_user.id if ws_user else ""
    sm = getattr(websocket.app.state, "state_manager", None)
    conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    if not ws_uid or conn is None:
        await websocket.close(code=4403, reason="Unauthorized")
        return
    try:
        cursor = await conn.execute(
            "SELECT 1 FROM project_checkouts WHERE id = ? AND user_id = ? LIMIT 1",
            (workspace_id, ws_uid),
        )
        if not await cursor.fetchone():
            await websocket.close(code=4403, reason="Workspace not found")
            return
    except Exception:
        await websocket.close(code=1011, reason="Ownership check failed")
        return

    async def _send(msg: dict) -> None:
        await websocket.send_json(msg)

    broker.register(workspace_id, _send)
    try:
        while True:
            data = await websocket.receive_json()
            if not isinstance(data, dict):
                continue
            if data.get("type") == "result":
                cid = str(data.get("id") or "")
                if cid:
                    broker.resolve(cid, {
                        "data_url": str(data.get("data_url") or ""),
                        "width": int(data.get("width") or 0),
                        "height": int(data.get("height") or 0),
                        "reason": str(data.get("reason") or ""),
                    })
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as exc:  # malformed frame / transport error — drop the socket
        log.warning("preview_capture_ws.error", workspace=workspace_id, error=str(exc))
    finally:
        broker.unregister(workspace_id, _send)
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Terminal WebSocket
# ---------------------------------------------------------------------------


@router.websocket("/ws/terminal/{workspace_id}")
async def terminal_ws(websocket: WebSocket, workspace_id: str) -> None:
    """Bidirectional terminal bridge over WebSocket (owner-only)."""
    await websocket.accept()
    mgr = getattr(websocket.app.state, "container_manager", None)
    if not mgr:
        await websocket.close(code=1011, reason="Container manager not available")
        return

    # Gate on ownership — the auth middleware attaches user to scope via WS
    # ticket. Without this check, a tenant who guessed a workspace_id could
    # shell into another tenant's container.
    ws_user = websocket.scope.get("user")
    ws_uid = ws_user.id if ws_user else ""
    sm = getattr(websocket.app.state, "state_manager", None)
    conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    if not ws_uid or conn is None:
        await websocket.close(code=4403, reason="Unauthorized")
        return
    try:
        cursor = await conn.execute(
            "SELECT 1 FROM project_checkouts WHERE id = ? AND user_id = ? LIMIT 1",
            (workspace_id, ws_uid),
        )
        if not await cursor.fetchone():
            await websocket.close(code=4403, reason="Workspace not found")
            return
    except Exception:
        await websocket.close(code=1011, reason="Ownership check failed")
        return

    try:
        exec_obj = await mgr.exec_shell(workspace_id)
    except Exception as exc:
        log.error("terminal_exec_failed", workspace=workspace_id, error=str(exc))
        await websocket.close(code=1011, reason=str(exc))
        return

    stream = exec_obj.start(detach=False)
    exec_id = exec_obj.id if hasattr(exec_obj, "id") else ""

    async def container_to_browser() -> None:
        try:
            while True:
                msg = await stream.read_out()
                if msg is None:
                    break
                await websocket.send_bytes(msg.data)
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass
        except Exception as exc:
            log.warning("container_read_error", workspace=workspace_id, error=str(exc))

    async def browser_to_container() -> None:
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                if data.get("type") == "input":
                    await stream.write_in(data["data"].encode())
                elif data.get("type") == "resize" and exec_id:
                    try:
                        await mgr.resize_exec(
                            exec_id, data.get("rows", 24), data.get("cols", 80)
                        )
                    except Exception:
                        log.warning("terminal_resize_failed", exec_id=exec_id, exc_info=True)
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass
        except Exception as exc:
            log.warning("browser_read_error", workspace=workspace_id, error=str(exc))

    tasks = [
        asyncio.create_task(container_to_browser()),
        asyncio.create_task(browser_to_container()),
    ]
    try:
        await asyncio.gather(*tasks)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        # Skip the close if either side already disconnected — calling
        # websocket.close() after that path raises "Unexpected ASGI
        # message 'websocket.close', after sending 'websocket.close'"
        # and dumps a noisy traceback for what is just a normal
        # client-driven terminal close.
        if (
            websocket.application_state != WebSocketState.DISCONNECTED
            and websocket.client_state != WebSocketState.DISCONNECTED
        ):
            try:
                await websocket.close()
            except Exception:
                log.debug("terminal_ws_close_failed", exc_info=True)


@router.websocket("/ws/coder/acp")
async def coder_acp_ws(websocket: WebSocket) -> None:
    """ACP editor endpoint — Augmentum's coder loop, in-process, over WebSocket.

    The editor machine runs a thin stdio<->WSS bridge (``python -m
    augmentum.coder.acp_stdio --bridge``); Zed speaks ACP to that bridge over
    stdio and the bridge tunnels the frames here. The real coder loop runs
    IN-PROCESS (shared model slots with chat/voice), acting on the user's editor
    via ``RemoteEditorExecutor`` rather than a Docker workspace.

    Auth: the sk-aug key on the WS ``Authorization``/``x-api-key`` header is
    validated by the auth middleware into ``scope['user']`` (same as the
    terminal WS). That tenant is bound to every ACP session on this connection.
    """
    await websocket.accept()
    ws_user = websocket.scope.get("user")
    uid = ws_user.id if ws_user else ""
    if not uid:
        await websocket.close(code=4403, reason="Unauthorized")
        return

    try:
        from augmentum.coder.acp_agent import AugmentumACPAgent
        from augmentum.coder.acp_app import make_app_loop_runner
        from augmentum.coder.acp_ws import serve_acp_over_websocket
    except Exception:
        # The ACP SDK (agent-client-protocol) is an OPTIONAL dependency — a
        # deployment without it simply doesn't offer the editor endpoint.
        log.warning("coder_acp_ws_sdk_unavailable", exc_info=True)
        await websocket.close(code=1011, reason="ACP editor support not installed")
        return

    # The user picks the coder model client-side (bridge forwards it as ?model=);
    # empty falls through to the provider registry's default resolution.
    model = websocket.query_params.get("model", "")
    runner = make_app_loop_runner(websocket.app.state, model=model)
    agent = AugmentumACPAgent(loop_runner=runner, default_user_id=uid)

    log.info("coder_acp_ws_open", user_id=uid, model=model or "(default)")
    try:
        await serve_acp_over_websocket(websocket, agent)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception:
        log.warning("coder_acp_ws_failed", exc_info=True)
    finally:
        if (
            websocket.application_state != WebSocketState.DISCONNECTED
            and websocket.client_state != WebSocketState.DISCONNECTED
        ):
            try:
                await websocket.close()
            except Exception:
                log.debug("coder_acp_ws_close_failed", exc_info=True)


# ---------------------------------------------------------------------------
# Dev-server preview reverse proxy
#
# Tunnels the workspace's published dev-server ports through Augmentum so
# the iframe in the coder UI is always same-origin (CSP 'self' permits
# it) and reachable from any device that can reach Augmentum — phone on
# LAN, tablet, etc. Without this, the iframe pointed at
# http://127.0.0.1:<host_port>/ which (a) violated CSP frame-src when the
# user accessed Augmentum at any host other than 127.0.0.1, and (b) could
# only ever resolve from a browser on the same machine as the Docker
# host.
#
# Caveat for dev servers that hardcode root-relative asset URLs (Vite,
# Next, CRA, etc.): set the dev server's base path to the proxy URL so
# generated <script src="/main.js"> turns into the correctly-prefixed
# path. Vite: `base: '/api/coder/preview/<id>/<port>/'`. Next:
# `basePath`. CRA: `homepage`. The proxy injects a <base href> tag into
# served HTML as a best-effort fallback for relative-URL templates.
# ---------------------------------------------------------------------------


# Hop-by-hop and connection-management headers that must NOT be forwarded
# either direction in a reverse proxy. RFC 7230 §6.1.
_HOP_BY_HOP_HEADERS = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
})

# Request headers we strip on the way upstream because they describe the
# Augmentum-facing leg, not the dev-server-facing leg.
_DROP_REQUEST_HEADERS = _HOP_BY_HOP_HEADERS | {"host", "content-length"}

# Response headers we strip on the way back. content-encoding/length are
# dropped on rewritten HTML responses (mutating the body invalidates
# them); on streamed pass-through responses we leave content-encoding
# alone so the browser still decompresses correctly.
_DROP_RESPONSE_HEADERS_BASE = _HOP_BY_HOP_HEADERS | frozenset({
    # X-Frame-Options: DENY from a dev server would block our iframe.
    # The dev server doesn't know it's being framed; drop it and let
    # Augmentum's own CSP frame-ancestors govern instead.
    "x-frame-options",
    # Same reasoning: a dev server's frame-ancestors could break us.
    # We strip the response CSP entirely — any policy needed for the
    # dev content lives at the Augmentum layer.
    "content-security-policy",
    "content-security-policy-report-only",
})

# Try Docker Desktop's host alias first; on Linux Docker the alias only
# resolves if compose.yaml adds `extra_hosts: ["host.docker.internal:
# host-gateway"]`, otherwise the bridge gateway 172.17.0.1 wins. We
# cache the working host on app.state after the first success so we're
# not paying a failed DNS lookup per request.
_PROXY_UPSTREAM_HOSTS = ("host.docker.internal", "172.17.0.1")
_PROXY_HOST_CACHE_KEY = "_coder_preview_upstream_host"


def _strip_response_headers(
    headers: httpx.Headers, *, body_rewritten: bool,
) -> list[tuple[str, str]]:
    drop = set(_DROP_RESPONSE_HEADERS_BASE)
    if body_rewritten:
        # Length and encoding describe the upstream bytes we just
        # mutated. Starlette will set content-length from the new body.
        drop.add("content-length")
        drop.add("content-encoding")
    return [
        (k, v) for k, v in headers.multi_items()
        if k.lower() not in drop
    ]


async def _resolve_proxy_host(app, host_port: int) -> str | None:
    """Return the first upstream host that accepts a TCP connect on
    ``host_port``. Result is cached on app.state for subsequent calls.

    We probe rather than rely on DNS alone because Docker Desktop
    resolves ``host.docker.internal`` even on Linux installs where the
    gateway route doesn't actually traverse it — the resolution succeeds
    but the connect hangs. Probing once per process keeps the hot path
    a dict lookup.
    """
    cached = getattr(app.state, _PROXY_HOST_CACHE_KEY, None)
    if cached:
        return cached
    for host in _PROXY_UPSTREAM_HOSTS:
        try:
            fut = asyncio.open_connection(host, host_port)
            _, writer = await asyncio.wait_for(fut, timeout=1.5)
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, ConnectionError):
                # Probe-only close; underlying socket may already be in
                # a torn-down state — pin host anyway.
                pass
            setattr(app.state, _PROXY_HOST_CACHE_KEY, host)
            return host
        except (OSError, TimeoutError):
            continue
    return None


async def _resolve_published_port(
    request: Request, workspace_id: str, container_port: int,
) -> int | None:
    """Return the host_port mapped to ``container_port`` for this
    workspace, or ``None`` if the port is not currently published or
    not exposed by the workspace at all (SSRF guard)."""
    mgr = _get_manager(request)
    if mgr is None:
        return None
    try:
        ports = await mgr.list_ports(workspace_id)
    except Exception:
        return None
    for row in ports:
        if int(row.get("container_port") or 0) != container_port:
            continue
        host_port = int(row.get("host_port") or 0)
        return host_port if host_port > 0 else None
    return None


# Root-relative href/src/action URL rewriter for HTML responses. Best-
# effort fallback for dev-server templates that don't honor <base href>:
# /main.js becomes /api/coder/preview/<id>/<port>/main.js. Skips //
# (protocol-relative), absolute URLs, fragment-only, and data:/blob:/
# javascript: schemes by virtue of requiring a leading single slash.
_HTML_ROOT_ATTR_RE = re.compile(
    rb'((?:href|src|action|formaction|poster|data-src)\s*=\s*)'
    rb'(["\'])'
    rb'(/(?!/)[^"\'\s>]*)'
    rb'\2',
    re.IGNORECASE,
)


# Tiny script injected into HTML responses on the isolated preview origin.
# Hooks fetch + XHR to detect 401 responses carrying the
# X-Augmentum-Preview-Expired header, and notifies the parent so it can
# re-mint a token and reload the iframe. The script is self-contained,
# IIFE-scoped, ~700 bytes minified — adds no runtime overhead unless a
# 401 actually fires, and idempotent against frameworks that
# monkey-patch fetch on their own (we wrap whatever's there at script-
# evaluation time). Sent with target='*' because the parent validates
# origin on receipt — that's where the security boundary lives.
_PREVIEW_EXPIRY_NOTIFY_SCRIPT = (
    b"<script>(function(){"
    b"function n(){try{window.parent.postMessage({type:'augmentum.preview.expired'},'*');}catch(e){}}"
    b"var f=window.fetch;"
    b"if(typeof f==='function'){"
    b"window.fetch=function(){"
    b"return f.apply(this,arguments).then(function(r){"
    b"if(r&&r.status===401&&r.headers&&r.headers.get('x-augmentum-preview-expired')==='true')n();"
    b"return r;"
    b"});"
    b"};"
    b"}"
    b"var X=window.XMLHttpRequest;"
    b"if(X&&X.prototype){"
    b"var o=X.prototype.open;"
    b"X.prototype.open=function(){"
    b"this.addEventListener('load',function(){"
    b"if(this.status===401&&this.getResponseHeader('x-augmentum-preview-expired')==='true')n();"
    b"});"
    b"return o.apply(this,arguments);"
    b"};"
    b"}"
    b"})();</script>"
)


# Console/error capture shim. Hooks console.error/warn + the window 'error'
# event + unhandledrejection in the REAL preview iframe, batches (500ms), and
# postMessages to the parent (coder UI), which relays to the /preview-console
# beacon. This is the only path by which the model sees errors from the user's
# live, stateful, interacted-with session — the headless browser tools cold-load
# a fresh page and structurally miss them. Fully guarded so a shim bug can never
# break the previewed app.
_PREVIEW_CONSOLE_CAPTURE_SCRIPT = (
    b"<script>(function(){"
    b"var Q=[],t=null;"
    b"function flush(){t=null;if(!Q.length)return;var b=Q.splice(0,Q.length);"
    b"try{window.parent.postMessage({type:'augmentum.preview.console',entries:b},'*');}catch(e){}}"
    b"function push(ty,tx,u,ln){if(!tx)return;"
    b"Q.push({type:ty,text:String(tx).slice(0,600),url:u||location.href,line:ln||0,ts:Date.now()/1000});"
    b"if(Q.length>50)Q.splice(0,Q.length-50);"
    b"if(!t)t=setTimeout(flush,500);}"
    b"function j(a){try{return Array.prototype.map.call(a,function(x){"
    b"return (x&&x.stack)?x.stack:(typeof x==='object'?JSON.stringify(x):String(x));}).join(' ');}catch(e){return '';}}"
    b"var ce=console.error;console.error=function(){try{push('error',j(arguments));}catch(e){}return ce.apply(this,arguments);};"
    b"var cw=console.warn;console.warn=function(){try{push('warn',j(arguments));}catch(e){}return cw.apply(this,arguments);};"
    b"window.addEventListener('error',function(ev){try{"
    b"push('exception',(ev.message||'')+(ev.filename?(' @'+ev.filename):''),ev.filename,ev.lineno);}catch(e){}},true);"
    b"window.addEventListener('unhandledrejection',function(ev){try{"
    b"var r=ev.reason;push('unhandledrejection',(r&&r.stack)||(r&&r.message)||String(r));}catch(e){}});"
    b"})();</script>"
)


def _preview_console_capture_on() -> bool:
    """Whether to inject the preview console-capture shim (settings-gated)."""
    from augmentum.config import settings as _s
    return bool(getattr(_s, "coder_preview_console_capture", True))


# Live-capture agent. Lets the coder capture the frame the USER's real GPU
# already rendered instead of re-rendering a heavy WebGL/Three.js page in the
# headless, GPU-less workspace (which is 6-45s+ or times out). Two parts:
#  1) hook getContext to force preserveDrawingBuffer, so WebGL canvases are
#     readable via toDataURL (Three.js defaults it OFF -> a blank capture).
#     Injected FIRST in <head>, before any app script creates its context.
#  2) on a parent 'augmentum.preview.capture' message, grab the largest canvas
#     as a PNG data URL (inside rAF, so it's a freshly-composited frame) and
#     postMessage it back. The parent relays it over the preview-capture WS to
#     browser_screenshot. Fully guarded — a shim bug can never break the app.
_PREVIEW_CAPTURE_SCRIPT = (
    b"<script>(function(){"
    b"try{var GC=HTMLCanvasElement.prototype.getContext;"
    b"HTMLCanvasElement.prototype.getContext=function(t,a){"
    b"if(t&&/webgl/i.test(t)){a=a||{};if(a.preserveDrawingBuffer!==true)a.preserveDrawingBuffer=true;}"
    b"return GC.call(this,t,a);};}catch(e){}"
    b"function big(){var cs=document.querySelectorAll('canvas'),b=null,ar=0;"
    b"for(var i=0;i<cs.length;i++){var c=cs[i],a=(c.width||0)*(c.height||0);if(a>ar){ar=a;b=c;}}return b;}"
    b"function grab(id){var o={type:'augmentum.preview.capture.result',id:id,data_url:null};"
    b"try{var c=big();if(!c){o.reason='no-canvas';}else{o.data_url=c.toDataURL('image/png');o.width=c.width;o.height=c.height;}}"
    b"catch(e){o.reason=String((e&&e.message)||e).slice(0,120);}"
    b"try{window.parent.postMessage(o,'*');}catch(e){}}"
    b"window.addEventListener('message',function(ev){var d=ev.data;"
    b"if(!d||typeof d!=='object'||d.type!=='augmentum.preview.capture')return;"
    b"try{requestAnimationFrame(function(){grab(d.id);});}catch(e){grab(d.id);}});"
    b"})();</script>"
)


def _preview_live_capture_on() -> bool:
    """Whether to inject the live-preview capture agent (settings-gated)."""
    from augmentum.config import settings as _s
    return bool(getattr(_s, "coder_preview_live_capture_enabled", True))


# Vite-shaped absolute imports inside JS module bodies. The HTML
# rewriter prefixes <script src="/foo"> attributes, but the dev
# server's emitted JS (Vite client, HMR runtime, pre-bundled deps)
# performs its own ``import "/src/App.tsx"`` / ``import("/@react-refresh")``
# statements that arrive as raw string literals. Browsers resolve those
# absolute paths against the iframe ORIGIN (not the <base href>), so
# without a body rewrite every Vite app shows a blank screen under the
# same-origin proxy. The isolation-origin path sidesteps this by giving
# the iframe its own root, but that's off by default and requires Caddy
# wiring — this rewriter is the unblocker for the default path.
#
# Anchored on a leading quote (single, double, or backtick) followed by
# ``/`` and a Vite-canonical first segment. Tight enough that user code
# strings like ``"src/file"`` (no leading slash) and ``"/api/foo"``
# (different prefix) pass through untouched. Covers:
#   - /@vite/client, /@vite/env, /@react-refresh, /@id/..., /@fs/...
#   - /@<plugin>/... — generic /@<word> for plugin-mounted virtual modules
#   - /src/...
#   - /node_modules/.vite/deps/... (dependency pre-bundle)
#   - /node_modules/<pkg>/... (less common but used by some dev servers)
_JS_VITE_IMPORT_RE = re.compile(
    rb'(["\'`])'
    rb'(/(?:'
    rb'@[A-Za-z0-9_\-]+(?:/[^"\'`?\s]*)?'   # /@vite, /@react-refresh, /@fs/...
    rb'|src/[^"\'`?\s]*'                     # /src/anything
    rb'|node_modules/[^"\'`?\s]*'           # /node_modules/anything
    rb'))'
    rb'(["\'`?])',
)


def _rewrite_js_module_body(body: bytes, base_path: str) -> bytes:
    """Prefix Vite/Next/Nuxt-style absolute import paths with the proxy
    base. See _JS_VITE_IMPORT_RE for what we match (and don't).

    Idempotent against pre-prefixed paths: if a path already starts with
    the base prefix, leave it alone so re-streams don't double-prefix.
    """
    prefix = base_path.encode("ascii", "ignore")  # ends with /
    def _sub(m: re.Match) -> bytes:
        opener = m.group(1)
        path = m.group(2)
        closer = m.group(3)
        if path.startswith(prefix):
            return m.group(0)
        # path starts with '/', prefix ends with '/' — drop one to avoid '//'
        return opener + prefix + path[1:] + closer
    return _JS_VITE_IMPORT_RE.sub(_sub, body)


# Content-types that should pass through the JS rewriter. Vite serves
# .tsx/.jsx/.css?direct/etc. with application/javascript so the type is
# more reliable than the URL extension. We deliberately do NOT touch
# application/json or text/css — JSON gets imported as data and CSS
# url() references are resolved against the document, not the script.
_JS_REWRITE_CONTENT_TYPES = (
    "application/javascript",
    "text/javascript",
    "application/ecmascript",
    "text/ecmascript",
    # Vite emits these for typescript/JSX in dev:
    "application/x-javascript",
)


def _is_js_response(content_type: str) -> bool:
    ct = (content_type or "").lower().split(";", 1)[0].strip()
    return ct in _JS_REWRITE_CONTENT_TYPES


# Runtime URL interceptor injected at the top of every HTML response.
# Wraps platform APIs that construct URLs so any same-origin absolute
# path from within the iframe gets routed through the proxy prefix.
# Strategy:
#   - fetch(input, init)          → rewrite string URL or Request.url
#   - XMLHttpRequest.open()       → rewrite url arg
#   - new WebSocket(url, protos)  → rewrite same-origin; reroute
#                                   localhost/127.0.0.1 HMR fallbacks
#                                   to the proxy origin so HMR survives
#   - new EventSource(url, init)  → same as fetch (HMR over SSE)
#
# This is framework-agnostic: static rewrites (HTML attrs, JS module
# imports) cover what the browser parses BEFORE running any code; this
# interceptor covers everything constructed at runtime. Together they
# make Vite, Next, Nuxt, SvelteKit, webpack-dev-server, Astro, etc.
# work under a path-prefix proxy without any per-framework rules.
#
# IIFE-scoped, defensive against double-wrap (data-augmentum-interceptor
# attribute on a marker), and runs at <head> entry so it patches APIs
# before any user/framework script can capture the originals. The
# {{BASE}} placeholder is replaced at injection time with the workspace-
# specific proxy base path.
_PREVIEW_URL_INTERCEPTOR_TMPL = b"""<script data-augmentum-interceptor="1">(function(){
if(window.__augmentumPreviewInterceptorInstalled)return;
window.__augmentumPreviewInterceptorInstalled=true;
var BASE="{{BASE}}";
var ORIGIN=location.origin;
var BASE_NO_SLASH=BASE.replace(/\\/$/,"");
function isPrefixed(p){return p===BASE_NO_SLASH||p.indexOf(BASE)===0;}
function prefixPath(p){return BASE+(p.charAt(0)==="/"?p.slice(1):p);}
function rewriteString(s){
  if(typeof s!=="string"||!s)return s;
  if(s.indexOf(ORIGIN+"/")===0){
    var path=s.slice(ORIGIN.length);
    if(isPrefixed(path))return s;
    return ORIGIN+prefixPath(path);
  }
  if(s.charAt(0)==="/"&&s.charAt(1)!=="/"){
    if(isPrefixed(s))return s;
    return prefixPath(s);
  }
  return s;
}
function rewriteInput(input){
  if(input==null)return input;
  if(typeof URL!=="undefined"&&input instanceof URL){
    if(input.origin===ORIGIN&&!isPrefixed(input.pathname)){
      var u=new URL(input.href);
      u.pathname=prefixPath(input.pathname);
      return u;
    }
    return input;
  }
  return rewriteString(input);
}
// fetch
if(typeof window.fetch==="function"){
  var _fetch=window.fetch.bind(window);
  window.fetch=function(input,init){
    try{
      if(typeof Request!=="undefined"&&input instanceof Request){
        var newUrl=rewriteString(input.url);
        if(newUrl!==input.url)input=new Request(newUrl,input);
      }else{
        input=rewriteInput(input);
      }
    }catch(e){}
    return _fetch(input,init);
  };
}
// XHR
if(window.XMLHttpRequest&&XMLHttpRequest.prototype){
  var _open=XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open=function(method,url){
    var args=Array.prototype.slice.call(arguments);
    try{args[1]=rewriteInput(url);}catch(e){}
    return _open.apply(this,args);
  };
}
// WebSocket: same-origin gets prefixed; localhost HMR fallbacks get
// rerouted to the proxy origin so HMR works without per-framework
// vite.config / webpack-config tweaks.
if(typeof window.WebSocket==="function"){
  var _WS=window.WebSocket;
  function PatchedWS(url,protocols){
    try{
      var s=typeof url==="string"?url:String(url);
      var u=new URL(s,ORIGIN);
      var sameOrigin=u.origin===ORIGIN||u.host===location.host;
      var isLocal=u.hostname==="localhost"||u.hostname==="127.0.0.1";
      if(sameOrigin||isLocal){
        if(!isPrefixed(u.pathname)){
          u.pathname=prefixPath(u.pathname||"/");
        }
        if(isLocal){
          u.protocol=location.protocol==="https:"?"wss:":"ws:";
          u.host=location.host;
        }
        s=u.toString();
      }
      return protocols!==undefined?new _WS(s,protocols):new _WS(s);
    }catch(e){
      return protocols!==undefined?new _WS(url,protocols):new _WS(url);
    }
  }
  PatchedWS.prototype=_WS.prototype;
  PatchedWS.CONNECTING=_WS.CONNECTING;
  PatchedWS.OPEN=_WS.OPEN;
  PatchedWS.CLOSING=_WS.CLOSING;
  PatchedWS.CLOSED=_WS.CLOSED;
  try{Object.defineProperty(window,"WebSocket",{value:PatchedWS,writable:true,configurable:true});}
  catch(e){window.WebSocket=PatchedWS;}
}
// EventSource (HMR over SSE for some frameworks)
if(typeof window.EventSource==="function"){
  var _ES=window.EventSource;
  function PatchedES(url,init){return new _ES(rewriteString(url),init);}
  PatchedES.prototype=_ES.prototype;
  PatchedES.CONNECTING=_ES.CONNECTING;
  PatchedES.OPEN=_ES.OPEN;
  PatchedES.CLOSED=_ES.CLOSED;
  try{Object.defineProperty(window,"EventSource",{value:PatchedES,writable:true,configurable:true});}
  catch(e){window.EventSource=PatchedES;}
}
})();</script>"""


def _build_url_interceptor(base_path: str) -> bytes:
    """Render the runtime URL interceptor with the workspace's base path
    baked into the BASE constant. base_path always ends with '/'."""
    safe_base = base_path.replace("\\", "\\\\").replace('"', '\\"')
    return _PREVIEW_URL_INTERCEPTOR_TMPL.replace(
        b"{{BASE}}", safe_base.encode("ascii", "ignore"),
    )


def _rewrite_html(
    body: bytes,
    base_path: str,
    *,
    inject_expiry_notify: bool = False,
    inject_console_capture: bool = False,
    inject_live_capture: bool = False,
) -> bytes:
    """Inject <base href> + runtime URL interceptor and prefix
    root-relative attribute URLs.

    When ``inject_expiry_notify`` is True (set by the proxy when serving
    on the isolated preview origin), also injects a small IIFE that
    hooks fetch + XHR to detect preview-session expiry and notifies the
    parent window via postMessage. The parent has a corresponding
    listener that re-mints a token and reloads the iframe.
    """
    base_bytes = base_path.encode("ascii", "ignore")
    base_tag = b'<base href="' + base_bytes + b'">'
    # Interceptor goes FIRST so it patches fetch/XHR/WebSocket/EventSource
    # before any framework script can capture the originals into closures.
    interceptor = _build_url_interceptor(base_path)
    head_inject = base_tag + interceptor
    if inject_expiry_notify:
        head_inject += _PREVIEW_EXPIRY_NOTIFY_SCRIPT
    if inject_console_capture:
        head_inject += _PREVIEW_CONSOLE_CAPTURE_SCRIPT
    if inject_live_capture:
        head_inject += _PREVIEW_CAPTURE_SCRIPT

    head_match = re.search(rb"<head[^>]*>", body, re.IGNORECASE)
    if head_match:
        idx = head_match.end()
        body = body[:idx] + head_inject + body[idx:]
    else:
        body = head_inject + body

    prefix = base_bytes  # already ends with /
    def _sub(m: re.Match) -> bytes:
        path = m.group(3)
        # Idempotency guard: if the URL is already prefixed (model
        # hardcoded the proxy path into the HTML, or this body has
        # already been rewritten by an upstream layer), don't prepend
        # the prefix a second time. Without this, paths like
        # /api/coder/preview/<ws>/<port>/foo.html get rewritten to
        # /api/coder/preview/<ws>/<port>/api/coder/preview/<ws>/<port>/
        # foo.html and 404 against the proxy router.
        if path.startswith(prefix) or path == prefix.rstrip(b"/"):
            return m.group(0)
        return m.group(1) + m.group(2) + prefix + path[1:] + m.group(2)
    body = _HTML_ROOT_ATTR_RE.sub(_sub, body)

    # Inline <script> bodies aren't attributes — the attribute rewriter
    # above misses them. Vite's index.html ships an inline preamble
    # `<script type="module">import RefreshRuntime from "/@react-refresh"…`
    # whose 404 takes down React fast-refresh entirely
    # ("@vitejs/plugin-react can't detect preamble"). Pipe each inline
    # script body through the JS module rewriter so the same Vite-shaped
    # specifiers get prefixed. Conservative scope: only `type="module"`
    # scripts (the only ones that can have bare-absolute import specifiers
    # without a build step). Plain inline scripts (jQuery snippets,
    # analytics tags) pass through untouched.
    def _sub_inline_script(m: re.Match) -> bytes:
        opening_tag = m.group(1)
        body_inner = m.group(2)
        closing_tag = m.group(3)
        return opening_tag + _rewrite_js_module_body(body_inner, base_path) + closing_tag
    body = _INLINE_MODULE_SCRIPT_RE.sub(_sub_inline_script, body)

    return body


# Inline ES-module scripts. Matches `<script type="module">…</script>`
# capturing the opening tag, inner body, and closing tag separately so
# the rewriter can transform just the body. Case-insensitive and tolerant
# of extra attributes on the script tag. Only `type="module"` — see
# _rewrite_html for why classic scripts are excluded.
_INLINE_MODULE_SCRIPT_RE = re.compile(
    rb'(<script\b[^>]*\btype\s*=\s*["\']module["\'][^>]*>)'
    rb'(.*?)'
    rb'(</script\s*>)',
    re.IGNORECASE | re.DOTALL,
)


def _rewrite_location(value: str, base_path: str, upstream_origin: str) -> str:
    """Rewrite a Location header so 3xx redirects stay inside the proxy."""
    prefix = base_path.rstrip("/")
    parts = urlsplit(value)
    if parts.scheme and parts.netloc:
        if f"{parts.scheme}://{parts.netloc}" != upstream_origin:
            return value
        path = parts.path or "/"
        # Idempotency: a dev server that's been told its base path is
        # the proxy prefix may emit redirects already containing it.
        if path.startswith(prefix + "/") or path == prefix:
            rewritten = path
        else:
            rewritten = prefix + path
        return urlunsplit(("", "", rewritten, parts.query, parts.fragment))
    if value.startswith("/"):
        if value.startswith(prefix + "/") or value == prefix:
            return value
        return prefix + value
    return value


async def _check_preview_auth(
    request: Request, workspace_id: str,
) -> Response | None:
    """Validate authorization for a preview-proxy request.

    Returns a Response (302 redirect, 401, 403, 404, 503) that the
    caller must return when the request must NOT proceed to the
    upstream dev server. Returns ``None`` when the proxy can continue.

    Two paths:

    - **Main origin** (no preview-listener header): existing behavior —
      validates the main session cookie via :func:`_owns_workspace`.

    - **Isolated origin** (X-Augmentum-Preview-Listener header set by
      Caddy on the :6444 upstream): three sub-cases —

      1. Request carries ``?_pvt=`` query → consume the one-time token,
         mint a preview-session cookie scoped to the isolated origin,
         302 redirect to the same URL with the query stripped.
      2. Request carries ``preview_session`` cookie → validate via the
         session store, extend sliding TTL, proceed.
      3. Neither → 401 with ``X-Augmentum-Preview-Expired: true`` so
         the parent UI's postMessage handler re-mints.

    Cross-cutting checks:

    - Token / session's bound workspace_id MUST match the URL's
      workspace_id (defense in depth — a token minted for workspace A
      can't be used to access workspace B even if the URL says so).
    - On isolation off, the isolated-origin requests still flow but
      the cookie path mints a session regardless — this is the
      transition path. Setting flips don't break in-flight previews.
    """
    from augmentum.config import settings

    is_isolated = bool(request.scope.get("augmentum_preview_isolated"))
    if not is_isolated:
        # Main origin — existing behavior unchanged.
        if not await _owns_workspace(request, workspace_id):
            return JSONResponse({"error": "Workspace not found"}, status_code=404)
        return None

    # ---- Isolated origin path -------------------------------------------
    token = request.query_params.get("_pvt", "")
    token_store = getattr(request.app.state, "preview_token_store", None)
    session_store = getattr(request.app.state, "preview_session_store", None)
    if token_store is None or session_store is None:
        return JSONResponse(
            {"error": "Preview auth unavailable"}, status_code=503,
        )

    if token:
        # Sub-case 1: redeem one-time token, set cookie, 302.
        record = token_store.consume(token)
        if record is None:
            return JSONResponse(
                {"error": "Invalid or expired preview token"},
                status_code=401,
            )
        if record.workspace_id != workspace_id:
            log.warning(
                "preview_token_workspace_mismatch",
                token_workspace=record.workspace_id,
                requested_workspace=workspace_id,
                user_id=record.user_id,
            )
            return JSONResponse(
                {"error": "Token workspace mismatch"}, status_code=403,
            )
        cookie_value = session_store.mint(
            user_id=record.user_id, workspace_id=workspace_id,
        )
        # Build the redirect target: same path, query stripped of _pvt.
        remaining_qs = "&".join(
            f"{k}={v}" for k, v in request.query_params.items() if k != "_pvt"
        )
        redirect_path = request.url.path
        if remaining_qs:
            redirect_path += "?" + remaining_qs
        response = Response(status_code=302)
        response.headers["Location"] = redirect_path
        response.set_cookie(
            key="preview_session",
            value=cookie_value,
            httponly=True,
            secure=(request.url.scheme == "https"),
            samesite="lax",
            max_age=int(settings.coder_preview_session_ttl_seconds),
            path="/",
        )
        return response

    # Sub-case 2/3: cookie auth.
    cookie = request.cookies.get("preview_session", "")
    if not cookie:
        return JSONResponse(
            {"error": "Preview session required"},
            status_code=401,
            headers={"X-Augmentum-Preview-Expired": "true"},
        )
    record = session_store.get(cookie)
    if record is None:
        return JSONResponse(
            {"error": "Preview session expired"},
            status_code=401,
            headers={"X-Augmentum-Preview-Expired": "true"},
        )
    if record.workspace_id != workspace_id:
        log.warning(
            "preview_session_workspace_mismatch",
            session_workspace=record.workspace_id,
            requested_workspace=workspace_id,
            user_id=record.user_id,
        )
        return JSONResponse(
            {"error": "Session workspace mismatch"}, status_code=403,
        )
    # Stash user_id in scope so logs / downstream handlers can attribute.
    # NOT scope["user"] — AuthMiddleware contract owns that key.
    request.scope["augmentum_preview_user_id"] = record.user_id
    return None


async def _proxy_http(
    request: Request, workspace_id: str, container_port: int, path: str,
) -> Response:
    auth_response = await _check_preview_auth(request, workspace_id)
    if auth_response is not None:
        return auth_response
    host_port = await _resolve_published_port(
        request, workspace_id, container_port,
    )
    if host_port is None:
        return JSONResponse(
            {"error": "Port not published for this workspace"},
            status_code=404,
        )
    upstream_host = await _resolve_proxy_host(request.app, host_port)
    if upstream_host is None:
        return JSONResponse(
            {"error": "Cannot reach workspace dev server from Augmentum"},
            status_code=502,
        )

    base_path = _preview_proxy_path(workspace_id, container_port)
    upstream_origin = f"http://{upstream_host}:{host_port}"
    upstream_url = f"{upstream_origin}/{path}"
    if request.url.query:
        upstream_url += "?" + request.url.query

    fwd_headers = [
        (k, v) for k, v in request.headers.items()
        if k.lower() not in _DROP_REQUEST_HEADERS
    ]
    body = await request.body()

    # We can't use `async with httpx.AsyncClient` and yield from a
    # generator after exiting the context — the connection pool would
    # tear down mid-stream. Manage the client/response lifetime manually
    # so the streaming generator can close them after the last chunk.
    client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))
    try:
        req = client.build_request(
            request.method, upstream_url, headers=fwd_headers, content=body,
        )
        resp = await client.send(req, stream=True)
    except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
        await client.aclose()
        log.warning(
            "coder_preview_upstream_unreachable",
            workspace=workspace_id, port=container_port, error=str(exc),
        )
        return JSONResponse(
            {"error": "Dev server stopped responding"}, status_code=502,
        )

    if 300 <= resp.status_code < 400 and resp.headers.get("location"):
        resp.headers["location"] = _rewrite_location(
            resp.headers["location"], base_path, upstream_origin,
        )

    content_type = (resp.headers.get("content-type") or "").lower()
    is_html = "text/html" in content_type
    is_js = _is_js_response(content_type)

    if is_html:
        try:
            full_body = await resp.aread()
        finally:
            await resp.aclose()
            await client.aclose()
        rewritten = _rewrite_html(
            full_body,
            base_path,
            inject_expiry_notify=bool(
                request.scope.get("augmentum_preview_isolated"),
            ),
            inject_console_capture=_preview_console_capture_on(),
            inject_live_capture=_preview_live_capture_on(),
        )
        out_headers = _strip_response_headers(resp.headers, body_rewritten=True)
        return Response(
            content=rewritten,
            status_code=resp.status_code,
            headers=dict(out_headers),
            media_type=content_type or "text/html",
        )

    if is_js:
        # JS bodies need Vite-style absolute imports prefixed with the
        # proxy base so they resolve against the proxy instead of the
        # iframe origin. See _rewrite_js_module_body for the matcher.
        # Buffered (not streamed) because we mutate the body.
        try:
            full_body = await resp.aread()
        finally:
            await resp.aclose()
            await client.aclose()
        rewritten = _rewrite_js_module_body(full_body, base_path)
        out_headers = _strip_response_headers(resp.headers, body_rewritten=True)
        return Response(
            content=rewritten,
            status_code=resp.status_code,
            headers=dict(out_headers),
            media_type=content_type or "application/javascript",
        )

    out_headers = _strip_response_headers(resp.headers, body_rewritten=False)

    async def _streamer():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            try:
                await resp.aclose()
            except Exception as exc:
                log.debug("coder_proxy_resp_aclose_failed", error=str(exc))
            try:
                await client.aclose()
            except Exception as exc:
                log.debug("coder_proxy_client_aclose_failed", error=str(exc))

    return StreamingResponse(
        _streamer(),
        status_code=resp.status_code,
        headers=dict(out_headers),
        media_type=content_type or None,
    )


# Methods we accept on the proxy. Kept explicit (rather than wildcard)
# so any new method requires a deliberate add — keeps the surface area
# inspectable.
_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


@router.api_route(
    "/api/coder/preview/{workspace_id}/{container_port:int}/{path:path}",
    methods=_PROXY_METHODS,
)
async def coder_preview_proxy(
    workspace_id: str, container_port: int, path: str, request: Request,
) -> Response:
    return await _proxy_http(request, workspace_id, container_port, path)


@router.api_route(
    "/api/coder/preview/{workspace_id}/{container_port:int}/",
    methods=_PROXY_METHODS,
)
async def coder_preview_proxy_root(
    workspace_id: str, container_port: int, request: Request,
) -> Response:
    return await _proxy_http(request, workspace_id, container_port, "")


@router.post("/api/coder/workspaces/{workspace_id}/preview-console")
async def coder_preview_console_beacon(
    workspace_id: str, request: Request,
) -> JSONResponse:
    """Ingest console/error events captured from the user's live preview iframe.

    The preview shim (``_PREVIEW_CONSOLE_CAPTURE_SCRIPT``) postMessages batches to
    the coder UI, which relays them here. They land in the per-workspace
    ``preview_console`` buffer that ``browser_snapshot`` + the turn-top auto-inject
    read, so the model sees the errors the USER actually saw — not just what its
    disconnected headless cold-load reproduces.

    Authenticated + workspace-scoped. Best-effort: malformed input is dropped, it
    never 500s (a noisy beacon must not break the preview).
    """
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": True, "high_water": 0})
    entries = body.get("entries") if isinstance(body, dict) else None
    if not isinstance(entries, list):
        return JSONResponse({"ok": True, "high_water": 0})
    from augmentum.coder.preview_console import record
    return JSONResponse({"ok": True, "high_water": record(workspace_id, entries)})


# ---------------------------------------------------------------------------
# Dev-server preview WebSocket bridge (HMR, devtools, anything that
# upgrades). Same ownership + port-validation gate as the HTTP proxy so
# an authenticated user can't poke arbitrary loopback ports.
# ---------------------------------------------------------------------------


async def _ws_owns_workspace(websocket: WebSocket, workspace_id: str) -> bool:
    user = websocket.scope.get("user")
    uid = user.id if user else ""
    if not uid:
        return False
    sm = getattr(websocket.app.state, "state_manager", None)
    conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    if conn is None:
        return False
    try:
        cursor = await conn.execute(
            "SELECT 1 FROM project_checkouts WHERE id = ? AND user_id = ? LIMIT 1",
            (workspace_id, uid),
        )
        return await cursor.fetchone() is not None
    except Exception:
        return False


@router.websocket(
    "/api/coder/preview/{workspace_id}/{container_port}/{path:path}",
)
async def coder_preview_proxy_ws(
    websocket: WebSocket, workspace_id: str, container_port: int, path: str,
) -> None:
    """Bridge a browser WebSocket to the dev server inside the workspace.

    Vite/Webpack HMR, devtools live-reload, and any custom socket the dev
    server hosts. Subprotocols are forwarded both directions so the
    upstream and the browser see the negotiation they expect.
    """
    if not await _ws_owns_workspace(websocket, workspace_id):
        await websocket.close(code=4403)
        return

    mgr = getattr(websocket.app.state, "container_manager", None)
    if mgr is None:
        await websocket.close(code=1011)
        return
    try:
        ports = await mgr.list_ports(workspace_id)
    except Exception:
        await websocket.close(code=1011)
        return
    host_port = 0
    for row in ports:
        if int(row.get("container_port") or 0) == container_port:
            host_port = int(row.get("host_port") or 0)
            break
    if host_port <= 0:
        await websocket.close(code=4404)
        return
    upstream_host = await _resolve_proxy_host(websocket.app, host_port)
    if upstream_host is None:
        await websocket.close(code=1011)
        return

    query = websocket.scope.get("query_string", b"")
    qs = ("?" + query.decode("latin-1")) if query else ""
    upstream_url = f"ws://{upstream_host}:{host_port}/{path}{qs}"

    requested_subprotocols = list(
        websocket.scope.get("subprotocols") or [],
    ) or None

    try:
        upstream = await websockets.connect(
            upstream_url,
            subprotocols=requested_subprotocols,
            open_timeout=10,
            max_size=None,  # dev servers happily push large HMR payloads
        )
    except Exception as exc:
        log.warning(
            "coder_preview_ws_upstream_failed",
            workspace=workspace_id, port=container_port, error=str(exc),
        )
        await websocket.close(code=1011)
        return

    # Honor the subprotocol the upstream selected, otherwise the browser
    # closes immediately with a protocol-mismatch error.
    selected = upstream.subprotocol
    await websocket.accept(subprotocol=selected)

    async def _browser_to_upstream() -> None:
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    return
                if msg.get("bytes") is not None:
                    await upstream.send(msg["bytes"])
                elif msg.get("text") is not None:
                    await upstream.send(msg["text"])
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass
        except Exception as exc:
            log.debug(
                "coder_preview_ws_b2u_error",
                workspace=workspace_id, error=str(exc),
            )

    async def _upstream_to_browser() -> None:
        try:
            async for msg in upstream:
                if isinstance(msg, bytes):
                    await websocket.send_bytes(msg)
                else:
                    await websocket.send_text(msg)
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass
        except Exception as exc:
            log.debug(
                "coder_preview_ws_u2b_error",
                workspace=workspace_id, error=str(exc),
            )

    tasks = [
        asyncio.create_task(_browser_to_upstream()),
        asyncio.create_task(_upstream_to_browser()),
    ]
    try:
        # Either side closing collapses both legs.
        _, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
    finally:
        try:
            await upstream.close()
        except Exception as exc:
            log.debug("coder_preview_ws_upstream_close_failed", error=str(exc))
        if (
            websocket.application_state != WebSocketState.DISCONNECTED
            and websocket.client_state != WebSocketState.DISCONNECTED
        ):
            try:
                await websocket.close()
            except Exception:
                log.debug("coder_preview_ws_close_failed", exc_info=True)


# ---------------------------------------------------------------------------
# Inspector panel — see 2026-05-28-coder-inspector-design.md
#
# The right-side inspector for coder mode surfaces the same hidden-state
# that the model reads each turn (objective, observations, workspace
# facts, turn history, costs) — but in a cooperative form: edits to the
# objective and observations mirror to the container files the kernel
# re-reads on the next turn. The model and the user share the same
# source of truth.
# ---------------------------------------------------------------------------


def _seconds_since(epoch: float | None) -> float | None:
    """Helper for the inspector run-status header (idle time / elapsed)."""
    if not epoch:
        return None
    import time as _time
    return max(0.0, _time.time() - float(epoch))


@router.get("/api/coder/workspaces/{workspace_id}/inspector-state")
async def get_inspector_state(
    workspace_id: str, request: Request,
) -> JSONResponse:
    """Aggregate snapshot of the workspace's coder state for the panel.

    Read-only. The inspector polls this every 1.5s while open. It
    returns everything the panel needs to render except the editable
    bodies (objective / observations), which have their own endpoints
    so edits round-trip without re-fetching the whole state.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)

    conn = _get_conn(request)
    if conn is None:
        return JSONResponse({"error": "Database unavailable"}, status_code=503)

    # Live run status from the broker (in-process truth).
    broker = getattr(request.app.state, "coder_run_broker", None)
    active_entry = None
    if broker is not None:
        active_entry = broker.get_active_for_workspace(
            user_id=uid, workspace_id=workspace_id,
        )

    run_status: dict[str, Any] = {"state": "idle"}
    if active_entry is not None:
        run_status = {
            "state": "running",
            "run_id": active_entry.run_id,
            "started_at": active_entry.started_at,
            "elapsed_s": _seconds_since(active_entry.started_at),
            "cancel_requested": active_entry.cancel_requested,
            "seq": active_entry.seq,
        }
    else:
        # Surface the last completed run summary so the panel can show
        # "Idle · last turn: cancelled 2m ago" instead of a blank header.
        try:
            cursor = await conn.execute(
                """
                SELECT id, status, model, strategy, finish_reason,
                       completed_at, started_at, iterations, tool_calls,
                       input_cost_usd, output_cost_usd
                FROM coder_turn_runs
                WHERE project_id = ? AND user_id = ?
                ORDER BY started_at DESC LIMIT 1
                """,
                (workspace_id, uid),
            )
            row = await cursor.fetchone()
        except Exception:
            row = None
        if row is not None:
            run_status = {
                "state": "idle",
                "last_run_id": row[0],
                "last_status": row[1],
                "last_model": row[2],
                "last_strategy": row[3],
                "last_finish_reason": row[4],
                "last_completed_at": row[5],
                "last_started_at": row[6],
                "last_iterations": int(row[7] or 0),
                "last_tool_calls": int(row[8] or 0),
                "last_idle_s": _seconds_since(row[5]),
            }

    # CoderState: mission + turn_summaries + iter_count are persisted.
    from augmentum.state.coder_persistence import CoderPersistence
    persistence = CoderPersistence(conn)
    state_doc: dict[str, Any] = {}
    try:
        loaded = await persistence.load_session_state(
            workspace_id, user_id=uid,
        )
    except Exception:
        loaded = None
    if loaded is not None:
        try:
            state_doc = {
                "mission": [p.to_dict() for p in (loaded.mission or [])],
                "turn_summaries": list(loaded.turn_summaries or []),
                "phase": getattr(loaded.phase, "value", str(loaded.phase)),
                "tasks": list(loaded.tasks or []),
                "recent_tool_failures": list(loaded.recent_tool_failures or []),
                "pending_objective_contract": dict(
                    loaded.pending_objective_contract or {},
                ),
                "current_step": loaded.current_step,
                "total_steps": loaded.total_steps,
                "files_read_count": len(loaded.files_read or {}),
                "working_set": sorted(loaded.working_set or [])[:32],
            }
        except Exception as exc:
            log.debug(
                "coder_inspector_state_serialize_failed",
                workspace_id=workspace_id, error=str(exc),
            )
            state_doc = {}

    # Identity is auto-detected at turn start and persisted via the
    # kernel's identity.toml. Read it through the kernel rather than
    # re-parsing the toml here so the detector logic stays in one place.
    identity_doc: dict[str, Any] = {}
    mgr = _get_manager(request)
    if mgr is not None:
        try:
            from dataclasses import asdict as _asdict

            from augmentum.coder.workspace_kernel import WorkspaceKernel
            kernel = WorkspaceKernel(mgr, workspace_id)
            manifest = await kernel.read_identity()
            identity_doc = _asdict(manifest)
        except Exception as exc:
            log.debug(
                "coder_inspector_identity_read_failed",
                workspace_id=workspace_id, error=str(exc),
            )

    # Cost aggregation — sums of the per-turn cost columns we capture
    # at finish_run time. Local models contribute $0; cloud models
    # contribute real LiteLLM-computed spend.
    from augmentum.coder.inspector_store import aggregate_session_costs
    cost = await aggregate_session_costs(
        conn, user_id=uid, workspace_id=workspace_id, session_id=workspace_id,
    )

    # Verification-spine rollup (spec 2026-07-06 Phase 2) — same store
    # the oracle-stats route uses, scoped to this workspace's runs.
    from augmentum.coder.ledger import CoderTurnLedgerStore
    oracle: dict[str, Any] = {}
    try:
        oracle = await CoderTurnLedgerStore(conn).oracle_stats(
            user_id=uid, project_id=workspace_id, limit=200,
        )
    except Exception as exc:
        log.warning(
            "coder_inspector_oracle_stats_failed",
            workspace_id=workspace_id, error=str(exc),
        )

    return JSONResponse({
        "workspace_id": workspace_id,
        "run_status": run_status,
        "state": state_doc,
        "identity": identity_doc,
        "cost": cost,
        "oracle": oracle,
    })


@router.get("/api/coder/workspaces/{workspace_id}/objective")
async def get_objective(workspace_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    from augmentum.coder.inspector_store import read_objective
    try:
        body = await read_objective(mgr, workspace_id)
    except Exception as exc:
        log.warning(
            "coder_inspector_objective_read_failed",
            workspace=workspace_id, error=str(exc),
        )
        return JSONResponse({"error": "Failed to read objective"}, status_code=500)
    return JSONResponse(body)


@router.put("/api/coder/workspaces/{workspace_id}/objective")
async def put_objective(
    workspace_id: str,
    body: InspectorObjectiveRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    from augmentum.coder.inspector_store import (
        ObjectiveConflictError,
        write_objective,
    )
    try:
        result = await write_objective(
            mgr,
            workspace_id,
            content=body.content,
            if_mtime_unchanged=body.if_mtime_unchanged,
        )
    except ObjectiveConflictError as conflict:
        return JSONResponse(
            {
                "error": "objective_conflict",
                "current_content": conflict.current_content,
                "current_mtime": conflict.current_mtime,
            },
            status_code=409,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        log.warning(
            "coder_inspector_objective_write_failed",
            workspace=workspace_id, error=str(exc),
        )
        return JSONResponse({"error": "Failed to write objective"}, status_code=500)
    return JSONResponse(result)


@router.get("/api/coder/workspaces/{workspace_id}/observations")
async def list_workspace_observations(
    workspace_id: str,
    request: Request,
    categories: str = "",
    limit: int = 200,
    offset: int = 0,
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    cats = [c.strip() for c in (categories or "").split(",") if c.strip()]
    from augmentum.coder.inspector_store import list_observations
    try:
        body = await list_observations(
            mgr, workspace_id,
            categories=cats or None,
            limit=max(1, min(int(limit or 200), 500)),
            offset=max(0, int(offset or 0)),
        )
    except Exception as exc:
        log.warning(
            "coder_inspector_observations_list_failed",
            workspace=workspace_id, error=str(exc),
        )
        return JSONResponse(
            {"error": "Failed to list observations"}, status_code=500,
        )
    return JSONResponse(body)


@router.post("/api/coder/workspaces/{workspace_id}/observations")
async def create_workspace_observation(
    workspace_id: str,
    body: InspectorObservationCreateRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    from augmentum.coder.inspector_store import create_observation
    try:
        result = await create_observation(
            mgr, workspace_id,
            user_id=uid,
            category=body.category,
            fact=body.fact,
            confidence=body.confidence,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        log.warning(
            "coder_inspector_observation_create_failed",
            workspace=workspace_id, error=str(exc),
        )
        return JSONResponse(
            {"error": "Failed to create observation"}, status_code=500,
        )
    return JSONResponse(result, status_code=201)


@router.patch("/api/coder/workspaces/{workspace_id}/observations/{idx}")
async def update_workspace_observation(
    workspace_id: str,
    idx: int,
    body: InspectorObservationUpdateRequest,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    from augmentum.coder.inspector_store import (
        ObservationNotFoundError,
        update_observation,
    )
    try:
        result = await update_observation(
            mgr, workspace_id, idx,
            user_id=uid,
            category=body.category,
            fact=body.fact,
            confidence=body.confidence,
        )
    except ObservationNotFoundError:
        return JSONResponse(
            {"error": "Observation not found"}, status_code=404,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        log.warning(
            "coder_inspector_observation_update_failed",
            workspace=workspace_id, idx=idx, error=str(exc),
        )
        return JSONResponse(
            {"error": "Failed to update observation"}, status_code=500,
        )
    return JSONResponse(result)


@router.delete("/api/coder/workspaces/{workspace_id}/observations/{idx}")
async def delete_workspace_observation(
    workspace_id: str,
    idx: int,
    request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not await _owns_workspace(request, workspace_id):
        return JSONResponse({"error": "Workspace not found"}, status_code=404)
    mgr = _get_manager(request)
    if mgr is None:
        return _unavailable()
    from augmentum.coder.inspector_store import (
        ObservationNotFoundError,
        delete_observation,
    )
    try:
        await delete_observation(mgr, workspace_id, idx)
    except ObservationNotFoundError:
        return JSONResponse(
            {"error": "Observation not found"}, status_code=404,
        )
    except Exception as exc:
        log.warning(
            "coder_inspector_observation_delete_failed",
            workspace=workspace_id, idx=idx, error=str(exc),
        )
        return JSONResponse(
            {"error": "Failed to delete observation"}, status_code=500,
        )
    return JSONResponse({"deleted": True})


@router.get("/api/coder/tuning")
async def get_coder_tuning(request: Request) -> JSONResponse:
    """Live-tunable harness manifest + current override values.

    Surfaces the breaker registry from ``augmentum/loops/breakers.py``
    plus the current ``settings.coder_breaker_<name>`` overrides so
    the inspector's Loop-tuning panel can render rows with name,
    description, registered default, current override, and effective
    threshold without the UI hardcoding the manifest.

    Reads — admin-only writes flow through the existing
    ``PUT /api/config/tools`` endpoint since the keys are already
    registered in ``_TOOL_SETTINGS``. No new persistence path here.
    """
    from augmentum.config import settings as _settings
    from augmentum.loops.breakers import (
        ALL_BREAKERS,
        HYBRID_MAX_ITERS,
        HYBRID_MAX_ITERS_UNGATED,
        NATIVE_NUDGE_MAX_DEFAULT,
    )

    breakers: list[dict] = []
    for b in ALL_BREAKERS:
        # threshold=0 entries are structural gates (e.g. the TQG), not
        # tunable knobs. Skip them so the UI doesn't render a confused
        # "default 0" row.
        if b.threshold <= 0:
            continue
        settings_key = f"coder_breaker_{b.name}"
        override = int(getattr(_settings, settings_key, 0) or 0)
        breakers.append({
            "name": b.name,
            "settings_key": settings_key,
            "registered_default": b.resolved_threshold,
            "override": override,
            "effective": override if override > 0 else b.resolved_threshold,
            "kind": b.kind,
            "bucket": b.bucket,
            "description": b.description,
            "env_var": b.env_var,
        })

    def _cap_row(name: str, key: str, fallback: int, env_var: str, desc: str) -> dict:
        override = int(getattr(_settings, key, 0) or 0)
        return {
            "name": name,
            "settings_key": key,
            "registered_default": fallback,
            "override": override,
            "effective": override if override > 0 else fallback,
            "kind": "cap",
            "bucket": "standard",
            "description": desc,
            "env_var": env_var,
        }

    iter_caps = [
        _cap_row(
            "hybrid_max_iters",
            "coder_hybrid_max_iters",
            HYBRID_MAX_ITERS,
            "AUGMENTUM_CODER_MAX_ITERS",
            "Maximum iterations per turn before the loop bails (workspace safeguards on).",
        ),
        _cap_row(
            "hybrid_max_iters_ungated",
            "coder_hybrid_max_iters_ungated",
            HYBRID_MAX_ITERS_UNGATED,
            "AUGMENTUM_CODER_MAX_ITERS_UNGATED",
            "Maximum iterations per turn when workspace safeguards are off.",
        ),
        _cap_row(
            "native_nudge_max",
            "coder_native_nudge_max",
            NATIVE_NUDGE_MAX_DEFAULT,
            "AUGMENTUM_CODER_NATIVE_NUDGE_MAX",
            "Native strategy: max consecutive prose-no-tools nudges before accepting the stop. Raise if chatty local models bail before acting; lower to terminate prose-only loops faster.",
        ),
    ]

    # Request pacing — a bool toggle + float seconds, distinct from the
    # int "0 = default" breaker/cap knobs above (hence its own block, not an
    # iter_cap row). Lets the user throttle the agentic loop to stay under a
    # fast provider's rate limit. Writes flow through PUT /api/config/tools
    # (both keys are registered in _TOOL_SETTINGS).
    pacing = {
        "enabled": bool(getattr(_settings, "coder_request_delay_enabled", False)),
        "seconds": float(getattr(_settings, "coder_request_delay_seconds", 0.0) or 0.0),
    }

    return JSONResponse({
        "breakers": breakers,
        "iter_caps": iter_caps,
        "pacing": pacing,
    })
