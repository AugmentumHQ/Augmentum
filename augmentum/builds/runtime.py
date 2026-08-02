"""Runtime helpers for in-process application builds."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from augmentum.builds.store import legacy_status, normalize_status
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

STALE_RUNNING_BUILD_SECONDS = 600
STALE_RUNNING_BUILD_REASON = (
    "Build stopped updating before completion. It was likely interrupted by a "
    "server restart or worker failure."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _status_is_terminal(status: str) -> bool:
    return normalize_status(status) in {"completed", "failed", "canceled"}


def _quality_status_from(*sources: dict | None) -> str:
    for source in sources:
        if not isinstance(source, dict):
            continue
        value = source.get("qualityStatus") or source.get("quality_status")
        if value:
            return str(value)
    return "clean"


def _list_from(source: dict | None, *keys: str) -> list:
    if not isinstance(source, dict):
        return []
    for key in keys:
        value = source.get(key)
        if isinstance(value, list):
            return value
    return []


def visible_to_user(build: dict, user_id: str) -> bool:
    owner = build.get("user_id") or ""
    if owner:
        return bool(user_id) and owner == user_id
    return True


def select_active_build(
    builds: dict[str, dict],
    *,
    user_id: str,
    build_id: str = "",
    session_id: str = "",
) -> tuple[str, dict] | tuple[None, None]:
    if build_id:
        build = builds.get(build_id)
        if build and visible_to_user(build, user_id):
            return build_id, build
        return None, None

    candidates: list[tuple[str, dict]] = []
    for bid, build in builds.items():
        if not visible_to_user(build, user_id):
            continue
        if session_id and (build.get("session_id") or "") != session_id:
            continue
        candidates.append((bid, build))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[1].get("started_at", item[0]))
    return candidates[-1]


def build_status_snapshot(build: dict, build_id: str) -> dict[str, Any]:
    """Render a legacy-compatible build-status payload from ACTIVE_BUILDS."""
    cp = build.get("_checkpoint", {})
    project = build.get("project") if isinstance(build.get("project"), dict) else {}
    quality_status = _quality_status_from(build, project)
    warnings = _list_from(build, "warnings", "qualityWarnings", "quality_warnings") or _list_from(project, "warnings", "qualityWarnings", "quality_warnings")
    blocking_errors = _list_from(build, "blockingErrors", "blocking_errors") or _list_from(project, "blockingErrors", "blocking_errors")
    completed_files = cp.get("completed_files", [])
    planned_files = cp.get("planned_files", [])
    completed_set = set(completed_files)
    remaining_files = [
        f["path"] if isinstance(f, dict) else f
        for f in planned_files
        if (f["path"] if isinstance(f, dict) else f) not in completed_set
    ]

    status = legacy_status(build.get("status", "running"))
    snap = {
        "active": status == "running",
        "id": build_id,
        "build_id": build_id,
        "name": build.get("name", ""),
        "status": status,
        "kind": build.get("kind", "application"),
        "session_id": build.get("session_id", ""),
        "task_id": build.get("task_id", ""),
        "artifact_id": build.get("artifact_id", ""),
        "passes": build.get("passes", []),
        "error": build.get("error"),
        "created_at": build.get("created_at"),
        "updated_at": build.get("updated_at"),
        "completed_at": build.get("completed_at"),
        "lastHeartbeat": build.get("lastHeartbeat"),
        "lastProgressAt": build.get("lastProgressAt"),
        "startedAtIso": build.get("started_at_iso") or build.get("created_at") or "",
        "model": build.get("model", ""),
        # Failure forensics — only populated on terminal=error builds.
        "failedPass": build.get("failedPass", ""),
        "lastCompletedPass": build.get("lastCompletedPass", ""),
        "errorDetail": build.get("errorDetail", ""),
        "filesComplete": completed_files,
        "filesRemaining": remaining_files,
        "totalTokens": build.get("totalTokens", 0),
        "llmCalls": build.get("llmCalls", 0),
        "qualityStatus": quality_status,
        "quality_status": quality_status,
        "warnings": warnings,
        "blockingErrors": blocking_errors,
        "blocking_errors": blocking_errors,
        "currentFile": build.get("currentFile", ""),
        # Workspace backing this build — lets the library Play surface preview
        # the LIVE dev server via the coder preview proxy while it's still
        # building (vs the static artifact once complete).
        "workspace_id": build.get("workspace_id", ""),
        # Ordered tool-step trail (build_progress tool_call phases). The live
        # monitor renders it; builder_eval judges it against the Power's
        # verification floor. Bounded upstream in the event sink.
        "steps": build.get("steps", []),
        # Inline quality verdict (floor judge), set on the complete stage event.
        "verdict": build.get("verdict") or {},
        # Spec-derived behavior contract with per-behavior pass/fail.
        "behaviors": build.get("behaviors") or [],
        "resume_count": int(build.get("resume_count") or 0),
        # Checkpoint gate: a budget/stuck stop pauses (non-fatal) and asks the
        # user to continue (resume) or keep what's built. The UI renders a gate
        # for this instead of a failure banner.
        "awaiting_continue": bool(build.get("awaiting_continue")) or status == "paused",
        "stop_reason": build.get("stop_reason", ""),
    }

    if cp.get("files"):
        snap["checkpoint"] = {
            "files": cp["files"],
            "planned_files": planned_files,
            "completed_files": completed_files,
        }

    # Paused builds are non-terminal but idle — attach the produced artifact so
    # the "keep what's built" side of the gate has something real to show.
    if _status_is_terminal(status) or status == "paused":
        project = build.get("project")
        if not project and cp.get("files"):
            project = {
                "name": build.get("name", ""),
                "files": cp["files"],
                "planned_files": planned_files,
                "completed_files": completed_files,
                "resumable": True,
                "qualityStatus": quality_status,
                "warnings": warnings,
                "blockingErrors": blocking_errors,
            }
        if project:
            project.setdefault("qualityStatus", quality_status)
            project.setdefault("quality_status", quality_status)
            project.setdefault("warnings", warnings)
            project.setdefault("blockingErrors", blocking_errors)
            snap["project"] = project
            artifact_id = project.get("artifactId") or project.get("artifact_id") or ""
            if artifact_id:
                snap["artifact_id"] = artifact_id
    return snap


def apply_project_progress(build: dict, progress: dict[str, Any]) -> None:
    """Merge an app-builder ``project_progress`` event into build state.

    The chat stream emits compact per-pass events, while persisted build
    status expects the legacy monitor shape with a pass list and checkpoint
    file sets. Keeping the merge here prevents the direct, iterate, and
    agentic paths from drifting apart.
    """
    if not isinstance(progress, dict):
        return
    build["lastProgressAt"] = _utc_now_iso()
    if progress.get("name"):
        build["name"] = progress["name"]
    if progress.get("pass"):
        existing = next(
            (p for p in build.setdefault("passes", [])
             if p.get("name") == progress["pass"]),
            None,
        )
        if existing:
            existing["status"] = progress.get("status", "running")
            existing["detail"] = progress.get("detail", "")
            existing["iterations"] = progress.get("iteration", existing.get("iterations", 0))
            if progress.get("max_iterations"):
                existing["max_iterations"] = progress["max_iterations"]
        else:
            build.setdefault("passes", []).append({
                "name": progress["pass"],
                "status": progress.get("status", "running"),
                "detail": progress.get("detail", ""),
                "iterations": progress.get("iteration", 0),
                "max_iterations": progress.get("max_iterations", 0),
            })
    if progress.get("files"):
        cp = build.setdefault("_checkpoint", {})
        cp["files"] = progress["files"]
        cp["planned_files"] = progress.get("planned_files", [])
        cp["completed_files"] = progress.get("completed_files", [])
    if progress.get("totalTokens") is not None:
        build["totalTokens"] = progress["totalTokens"]
        build["llmCalls"] = progress.get("llmCalls", 0)
    if progress.get("currentFile") is not None:
        build["currentFile"] = progress.get("currentFile", "")
    quality_status = _quality_status_from(progress)
    if quality_status != "clean" or progress.get("qualityStatus") or progress.get("quality_status"):
        build["qualityStatus"] = quality_status
        build["quality_status"] = quality_status
    warnings = _list_from(progress, "warnings", "qualityWarnings", "quality_warnings")
    if warnings:
        build["warnings"] = warnings
    blocking_errors = _list_from(progress, "blockingErrors", "blocking_errors")
    if blocking_errors:
        build["blockingErrors"] = blocking_errors
        build["blocking_errors"] = blocking_errors


async def load_persisted_build_run(
    store: Any,
    *,
    build_id: str = "",
    session_id: str = "",
    user_id: str,
    stale_after_seconds: int = STALE_RUNNING_BUILD_SECONDS,
) -> dict | None:
    """Load a persisted run and fail it if it has gone stale.

    Agentic builds are persisted without an ``ACTIVE_BUILDS`` entry, so the
    stale decision is based on DB update age instead of in-memory presence.
    """
    if not store:
        return None
    run = (
        await store.get(build_id, user_id=user_id)
        if build_id else
        await store.latest_for_session(session_id, user_id=user_id)
    )
    if not run:
        return None
    if normalize_status(run.get("status") or "") not in {"running", "queued"}:
        return run

    marker = getattr(store, "mark_running_stale", None)
    if not marker:
        return run
    marked = await marker(
        run.get("id") or build_id,
        user_id=user_id,
        max_age_seconds=stale_after_seconds,
        reason=STALE_RUNNING_BUILD_REASON,
    )
    if marked:
        log.warning("build_run.stale_marked", build_id=run.get("id") or build_id)
        return await store.get(run.get("id") or build_id, user_id=user_id)
    return run


def build_status_from_run(run: dict | None) -> dict[str, Any]:
    if not run:
        return {"active": False}

    progress = run.get("progress") or {}
    result = run.get("result") or {}
    project = result.get("project") or progress.get("project")
    artifact_id = run.get("artifact_id") or result.get("artifact_id") or ""
    if project and artifact_id and not project.get("artifactId"):
        project = {**project, "artifactId": artifact_id}
    quality_status = _quality_status_from(progress, project)
    warnings = _list_from(progress, "warnings", "qualityWarnings", "quality_warnings") or _list_from(project, "warnings", "qualityWarnings", "quality_warnings")
    blocking_errors = _list_from(progress, "blockingErrors", "blocking_errors") or _list_from(project, "blockingErrors", "blocking_errors")

    status = legacy_status(run.get("status", ""))
    snap = {
        "active": status == "running",
        "id": run.get("id", ""),
        "build_id": run.get("id", ""),
        "name": run.get("name", ""),
        "status": status,
        "kind": run.get("kind", "application"),
        "session_id": run.get("session_id", ""),
        "task_id": run.get("task_id", ""),
        "artifact_id": artifact_id,
        "passes": progress.get("passes", []),
        "error": run.get("error"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "completed_at": run.get("completed_at"),
        "lastHeartbeat": progress.get("lastHeartbeat"),
        "lastProgressAt": progress.get("lastProgressAt"),
        # Persisted runs carry start time as created_at (ISO from SQLite
        # datetime('now')); model lives on the request blob.
        "startedAtIso": run.get("created_at") or "",
        "model": (run.get("request") or {}).get("model", ""),
        # Failure forensics persisted on the progress blob.
        "failedPass": progress.get("failedPass", "") or (project or {}).get("failed_pass", ""),
        "lastCompletedPass": progress.get("lastCompletedPass", "") or (project or {}).get("last_completed_pass", ""),
        "errorDetail": progress.get("errorDetail", "") or (project or {}).get("error_detail", ""),
        "filesComplete": progress.get("filesComplete", []),
        "filesRemaining": progress.get("filesRemaining", []),
        "totalTokens": progress.get("totalTokens", 0),
        "llmCalls": progress.get("llmCalls", 0),
        "qualityStatus": quality_status,
        "quality_status": quality_status,
        "warnings": warnings,
        "blockingErrors": blocking_errors,
        "blocking_errors": blocking_errors,
        "currentFile": progress.get("currentFile", ""),
        "workspace_id": run.get("workspace_id", "") or (result or {}).get("workspace_id", ""),
        # Forward-compat: the in-memory build accumulates the tool trail; if a
        # future persist path writes it onto the progress blob it surfaces here.
        "steps": progress.get("steps", []),
        # Inline quality verdict (floor judge) persisted on the result blob.
        "verdict": (result or {}).get("verdict") or {},
        # Spec-derived behavior contract with per-behavior pass/fail.
        "behaviors": (result or {}).get("behaviors")
        or (project or {}).get("behaviors") or [],
        "resume_count": int(run.get("resume_count") or 0),
        # Checkpoint gate survives a restart: a persisted 'paused' row still
        # renders the continue/stop gate rather than a failure.
        "awaiting_continue": status == "paused",
        "stop_reason": (result or {}).get("stop_reason", ""),
    }
    if progress.get("checkpoint"):
        snap["checkpoint"] = progress["checkpoint"]
    if project:
        project.setdefault("qualityStatus", quality_status)
        project.setdefault("quality_status", quality_status)
        project.setdefault("warnings", warnings)
        project.setdefault("blockingErrors", blocking_errors)
        snap["project"] = project
    return snap


async def heartbeat_build_run(
    store: Any,
    *,
    build_id: str,
    user_id: str,
    state: dict[str, Any],
    interval: float = 30.0,
) -> None:
    """Keep a persisted running build fresh during long LLM calls.

    Only writes when the progress payload or the build's name has
    actually changed since the last heartbeat. Without this dirty
    check, an idle build (LLM hung, no file output) would still fire
    a write every ``interval`` seconds, holding the SQLite writer lock
    just long enough to push concurrent writers over busy_timeout
    during heavy traffic.
    """
    if not store or not build_id or not user_id:
        return
    last_signature: str | None = None
    while True:
        await asyncio.sleep(interval)
        if normalize_status(state.get("status") or "running") != "running":
            return
        # Always refresh the in-memory heartbeat so observers using the
        # change event still tick. This is local and never blocks anyone.
        state["lastHeartbeat"] = _utc_now_iso()
        ev = state.get("_change_event")
        if ev is not None:
            ev.set()
        # Compute the *persisted* signature — payload + name — and skip
        # the DB write when it hasn't moved. lastHeartbeat intentionally
        # stays out of the signature: refreshing it on disk every 30s
        # without any real progress is exactly the write-amplification
        # we're trying to avoid.
        progress = progress_payload_from_state(state)
        name = state.get("name", "")
        try:
            signature = json.dumps(
                {"name": name, "progress": progress},
                sort_keys=True, default=str,
            )
        except (TypeError, ValueError):
            # Non-serializable payload (shouldn't happen — store.update
            # would have failed too). Fall through and write to surface
            # the underlying issue.
            signature = None
        if signature is not None and signature == last_signature:
            continue
        try:
            await store.update(
                build_id,
                user_id=user_id,
                name=name,
                progress=progress,
            )
            last_signature = signature
        except Exception:
            log.warning("build_run.heartbeat_failed", build_id=build_id, exc_info=True)


def progress_payload_from_state(build: dict) -> dict[str, Any]:
    cp = build.get("_checkpoint", {})
    project = build.get("project") if isinstance(build.get("project"), dict) else {}
    quality_status = _quality_status_from(build, project)
    warnings = _list_from(build, "warnings", "qualityWarnings", "quality_warnings") or _list_from(project, "warnings", "qualityWarnings", "quality_warnings")
    blocking_errors = _list_from(build, "blockingErrors", "blocking_errors") or _list_from(project, "blockingErrors", "blocking_errors")
    completed_files = cp.get("completed_files", [])
    planned_files = cp.get("planned_files", [])
    completed_set = set(completed_files)
    remaining_files = [
        f["path"] if isinstance(f, dict) else f
        for f in planned_files
        if (f["path"] if isinstance(f, dict) else f) not in completed_set
    ]
    return {
        "passes": build.get("passes", []),
        "filesComplete": completed_files,
        "filesRemaining": remaining_files,
        "totalTokens": build.get("totalTokens", 0),
        "llmCalls": build.get("llmCalls", 0),
        "lastHeartbeat": build.get("lastHeartbeat"),
        "lastProgressAt": build.get("lastProgressAt"),
        "qualityStatus": quality_status,
        "quality_status": quality_status,
        "warnings": warnings,
        "blockingErrors": blocking_errors,
        "blocking_errors": blocking_errors,
        "failedPass": build.get("failedPass", ""),
        "lastCompletedPass": build.get("lastCompletedPass", ""),
        "errorDetail": build.get("errorDetail", ""),
        "currentFile": build.get("currentFile", ""),
        "steps": build.get("steps", []),
        "checkpoint": {
            "files": cp.get("files", []),
            "planned_files": planned_files,
            "completed_files": completed_files,
        } if cp.get("files") else None,
        "project": build.get("project"),
    }
