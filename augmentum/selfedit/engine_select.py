"""Assemble the self-edit EditDriver from a chosen ENGINE — the connector that
makes the debt loop (and any self-edit) actually fix things.

The debt endpoint already passes ``app.state.selfedit_driver`` to ``run_debt_loop``
and runs ``dry_run = not live``; the slot is just empty, so it triages + dry-runs
but never edits. This factory fills that slot from the user's engine choice, reusing
the backend-agnostic drivers built this session:

  - ``native``      → a LOCAL model via the coder's agentic loop (sovereign, no
                      token) — ``NativeModelDriver`` + ``run_external_edit_driver``;
  - ``claude_code`` / ``codex`` → the external platform via the registry
                      (``build_selected_edit_driver`` → ``select_driver``).

So: Workshop debt → green lane (``live``) → ``run_debt_loop(driver=this)`` → per
mechanical finding: isolate candidate → the ENGINE edits it → the candidate audit
verifies (finding gone + no regression = the mechanical oracle) → rest at gated →
promote (apply-diff). The score climbs, autonomously, on the safe lane.

``native_loop`` (the coder loop adapter over a workspace) and the platform creds
are injected, so this is testable; assembling the driver constructs nothing live
by itself.
"""

from __future__ import annotations

from typing import Any

from augmentum.selfedit.external_edit_driver import (
    build_selected_edit_driver,
    run_external_edit_driver,
)
from augmentum.selfedit.orchestrator import EditDriver
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

ENGINE_NATIVE = "native"          # local model (sovereign default)
ENGINE_CLAUDE = "claude_code"
ENGINE_CODEX = "codex"

# Setting key a UI/startup reads to pick the self-edit engine.
ENGINE_SETTING_KEY = "selfedit_engine"
DEFAULT_ENGINE = ENGINE_NATIVE


async def build_selfedit_driver(*, conn: Any, engine: str = DEFAULT_ENGINE,
                                cwd: str = "/workspace", oauth_token: str = "",
                                api_key: str = "", native_loop: Any = None,
                                registry: Any = None, native_role: str = "utility",
                                model: str = "", max_iters: int = 40,
                                _select: Any = None) -> EditDriver | None:
    """Build the EditDriver for ``app.state.selfedit_driver`` from ``engine``.
    Returns None if that engine isn't usable here (no native loop/registry / no
    available platform), so the debt loop stays a safe dry-run, not a broken run.

    For ``native``: pass a ready ``native_loop``, OR a ``registry`` — the loop is
    then self-constructed (``make_native_loop`` over ``build_local_chat`` against
    the model list) = the sovereign path, no token."""
    if engine == ENGINE_NATIVE:
        if native_loop is None and registry is not None:
            from augmentum.selfedit.native_loop import build_local_chat, make_native_loop
            native_loop = make_native_loop(
                chat=build_local_chat(registry, role=native_role, model=model),
                max_iters=max_iters)
        if native_loop is None:
            log.info("selfedit_native_engine_unavailable",
                     reason="no native_loop and no registry")
            return None
        from augmentum.coder.external.native_model_driver import NativeModelDriver
        driver = NativeModelDriver(run_loop=native_loop, model=model)
        return run_external_edit_driver(conn=conn, driver=driver)

    # External platforms (claude_code | codex) via the registry.
    return await build_selected_edit_driver(
        conn=conn, prefer=engine, cwd=cwd, oauth_token=oauth_token,
        api_key=api_key, model=model, _select=_select)


async def wire_selfedit_driver(app_state: Any, conn: Any, *, engine: str = DEFAULT_ENGINE,
                               repo_dir: str = "", cwd: str = "/workspace",
                               oauth_token: str = "", api_key: str = "",
                               native_loop: Any = None, registry: Any = None,
                               native_role: str = "utility", model: str = "",
                               max_iters: int = 40, _select: Any = None) -> bool:
    """THE startup wiring: set ``app_state.selfedit_driver`` (+ ``selfedit_repo_dir``)
    so the Workshop debt green-lane and any self-edit promote can actually run.
    Returns True if a driver was wired (live), False if not (debt stays a SAFE
    dry-run). The whole server-side activation is ONE call to this.

    ``native`` needs a ``native_loop`` (the local code-editing agentic loop — see
    the contract in the build-ref); ``claude_code``/``codex`` need a credential
    (operator-level for the shared app.state driver) and are wirable today."""
    driver = await build_selfedit_driver(
        conn=conn, engine=engine, cwd=cwd, oauth_token=oauth_token, api_key=api_key,
        native_loop=native_loop, registry=registry, native_role=native_role,
        model=model, max_iters=max_iters, _select=_select)
    app_state.selfedit_driver = driver
    if repo_dir:
        app_state.selfedit_repo_dir = repo_dir
    log.info("selfedit_driver_wired", engine=engine, live=driver is not None,
             repo_dir=bool(repo_dir))
    return driver is not None
