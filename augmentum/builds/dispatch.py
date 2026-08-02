"""Shared app-build kickoff — the single seam both conversational entry points
route through so build mode is ONE builder, not two.

`build_application` can be triggered two ways:

  * **build mode / direct-invoke** — the builder tool is enabled and the
    passthrough handler fires it as a long-running direct tool
    (``handler._run_long_running_direct_tool_stream``);
  * **gated offer Accept** — the model proposes the tool in chat/passthrough
    and the user clicks Accept (``offers.catalog.gated_tool``).

Both now call :func:`start_app_build`, which runs the SAME coder-workspace
builder as the Library "Build an app" button (``builds.facade.run_build``): a
real Docker workspace + Playwright behavior gate, tracked in ``ACTIVE_BUILDS``
and served by ``/api/builds/{id}`` + ``/stream``. The retired in-process
quickjs pipeline (``tools.artifact_application``) is no longer on either path.

Docker is a hard prerequisite for Augmentum, so there is no lightweight
fallback — if the workspace stack is unavailable we return a clear error
instead of silently shipping an unverified app.
"""

from __future__ import annotations

import asyncio
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _pulse(build_state: dict) -> None:
    ev = build_state.get("_change_event")
    if ev is not None:
        ev.set()


async def start_app_build(
    app_state: Any,
    *,
    objective: str,
    user_id: str,
    session_id: str = "",
    model: str = "",
) -> dict:
    """Kick off an app build in the background; return a started-ack dict.

    Returns ``{"ok": True, "build_id": str, "name": str}`` on success, or
    ``{"ok": False, "error": str, "detail": str}`` if the workspace stack or a
    model backend isn't available. Never raises for the common failure modes —
    the caller surfaces ``detail`` to the user.
    """
    cm = getattr(app_state, "container_manager", None)
    artifact_store = getattr(app_state, "artifact_store", None)
    build_store = getattr(app_state, "build_run_store", None)
    provider_registry = getattr(app_state, "provider_registry", None)
    power_registry = getattr(app_state, "power_registry", None)

    missing = [
        label for label, dep in (
            ("container manager", cm),
            ("artifact store", artifact_store),
            ("build store", build_store),
            ("model registry", provider_registry),
        ) if dep is None
    ]
    if missing:
        return {
            "ok": False,
            "error": "workspace_unavailable",
            "detail": (
                "The coder workspace stack isn't available "
                f"({', '.join(missing)}). It needs the Docker services running."
            ),
        }

    backend, clean_model = await provider_registry.resolve_backend_with_fabric(
        model, user_id=user_id,
    )
    if backend is None:
        return {
            "ok": False,
            "error": "no_model",
            "detail": "No model backend is available — select a model first.",
        }
    resolved_model = clean_model or model

    # Deferred imports keep this module import-light and avoid load-time cycles
    # (handler ↔ build_routes both reference ACTIVE_BUILDS).
    from augmentum.builds.runtime import _utc_now_iso
    from augmentum.modes.passthrough.handler import ACTIVE_BUILDS
    from augmentum.proxy.build_routes import _make_build_event_sink
    from augmentum.tools.artifact_application import derive_project_name

    build_id = f"build_{int(asyncio.get_event_loop().time() * 1000)}"
    project_name = derive_project_name(objective)
    # Snapshot shape mirrors build_routes.create_build so GET /api/builds/{id}
    # + /stream serve ONE consistent monitor payload regardless of entry point.
    # run_build creates the build_runs row itself (non-resume) — we don't
    # pre-create it, matching the Library path.
    build_state = {
        "id": build_id,
        "kind": "application",
        "user_id": user_id,
        "session_id": session_id,
        "task_id": "",
        "started_at": asyncio.get_event_loop().time(),
        "started_at_iso": _utc_now_iso(),
        "model": resolved_model,
        "name": project_name,
        "status": "running",
        "passes": [],
        "error": None,
        "project": None,
        "workspace_id": "",
        "_checkpoint": {},
        "_change_event": asyncio.Event(),
    }
    ACTIVE_BUILDS[build_id] = build_state
    sink = _make_build_event_sink(build_state)

    async def _run_build_bg() -> None:
        from augmentum.builds.facade import run_build
        try:
            await run_build(
                objective=objective,
                user_id=user_id,
                backend=backend,
                model=resolved_model,
                container_manager=cm,
                artifact_store=artifact_store,
                build_run_store=build_store,
                power_registry=power_registry,
                session_id=session_id,
                build_id=build_id,
                event_sink=sink,
            )
        except asyncio.CancelledError:
            build_state["status"] = "cancelled"
            _pulse(build_state)
            raise
        except Exception as exc:  # noqa: BLE001 — surface as a failed build
            import traceback
            log.warning(
                "app_build.bg_failed",
                build_id=build_id, error=str(exc), exc_info=True,
            )
            build_state["status"] = "error"
            build_state["error"] = str(exc)
            build_state["errorDetail"] = traceback.format_exc()
            _pulse(build_state)

    # Hold a ref so the background build isn't GC'd mid-run.
    from augmentum.utils.bg_tasks import track
    track(_run_build_bg())
    log.info(
        "app_build.started",
        build_id=build_id, model=resolved_model, session=session_id,
    )
    return {"ok": True, "build_id": build_id, "name": project_name}
