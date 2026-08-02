"""Backend-agnostic self-edit driver — run the editing agent on ANY external
coder backend (Claude Code today; Codex and others the moment their driver lands).

``edit_driver.run_engine_edit_driver`` is Claude-CLI-specific (it builds
``build_claude_argv`` + routes Claude auth). This module bridges the self-edit
``EditDriver`` seam onto the already-multi-backend ``ExternalCoderDriver``
abstraction (``coder/external/base.py``): it consumes the **normalized
``CoderEvent`` stream** — the ONE shape every engine emits — so self-edit gains
every backend the registry knows, through a single consumer. No engine-specifics
live here; auth/wire-format/permission mapping are the driver's job.

The driver is injected (testable with a fake that yields ``CoderEvent``s). Live
selection is the registry's ``select_driver(prefer=...)``; ``build_selected_edit_driver``
ties them together and returns None when no backend is available (caller falls
back to the native/CLI path).

Not exported from ``augmentum.selfedit.__init__`` (importing it pulls in
``coder.external``); import it where you wire the loop, like ``edit_driver``.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Any

from augmentum.coder.external import run_store
from augmentum.coder.external.base import ExternalCoderDriver, ExternalTask
from augmentum.selfedit.live import emit_progress
from augmentum.selfedit.orchestrator import EditDriver, EditRequest, EditResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Self-edit runs the agent UNATTENDED inside an isolated candidate worktree — the
# autonomy/consent decision is upstream (which attempts run at all), and the gate
# + verifier are the safety, so the agent must be free to write. Maps to the
# engine's auto mode per-driver.
_DEFAULT_PERMISSION = "auto"


def run_external_edit_driver(*, conn: Any, driver: ExternalCoderDriver,
                             permission: str = _DEFAULT_PERMISSION,
                             model: str = "") -> EditDriver:
    """Build an ``EditDriver`` that runs ``driver`` (any ``ExternalCoderDriver``)
    over the candidate and records the run, consuming normalized events. Persists
    via the SAME ``run_store`` the Claude path uses — one transcript shape, no
    drift. Never raises (normalizes failure to ``EditResult.ok=False``)."""

    async def _drive(req: EditRequest) -> EditResult:
        run_id = uuid.uuid4().hex
        if conn is not None:
            with contextlib.suppress(Exception):
                await run_store.create_run(
                    conn, run_id=run_id, user_id=req.user_id,
                    workspace_id=req.candidate.name, task=req.objective,
                    permission=permission)

        seq = {"n": 0}

        async def _emit(kind: str, text: str = "", tool: str = "", path: str = "") -> None:
            if conn is None:
                return
            seq["n"] += 1
            with contextlib.suppress(Exception):
                await run_store.add_event(conn, run_id=run_id, user_id=req.user_id,
                                          seq=seq["n"], kind=kind, text=text,
                                          tool=tool, path=path)

        # Escalation: prepend what weaker tiers already learned so a stronger
        # model builds on the groundwork instead of repeating reads/searches.
        prompt = req.objective
        if req.prior_context:
            prompt = f"{req.objective}\n\n{req.prior_context}"
        task = ExternalTask(prompt=prompt, workspace=req.candidate.path,
                            permission=permission, model=model)

        files: list[str] = []
        final_text = ""
        session_ref = ""
        ok = False
        error = ""
        def _arg_preview(raw: Any) -> str:
            """A compact, human descriptor of a tool call's args for the live feed
            (the persisted row only keeps text/tool/path; the live event is richer).
            For an edit, show the unique old→new so the watcher sees the actual change."""
            args = (raw or {}).get("args") if isinstance(raw, dict) else None
            if not isinstance(args, dict):
                return ""
            if "old_string" in args or "new_string" in args:
                old = str(args.get("old_string", "")).strip()[:160]
                new = str(args.get("new_string", "")).strip()[:160]
                return f"- {old}\n+ {new}" if (old or new) else ""
            for k in ("query", "path", "content", "summary"):
                if args.get(k):
                    return str(args[k])[:200]
            return ""

        try:
            async for ev in driver.run(task):
                # Persist the edit CONTENT too: a tool_call row carried only
                # tool+path (you couldn't see WHAT the model wrote without diffing
                # the branch). Fall back to the arg preview (edit old→new, the
                # search query, the written content) so the transcript is legible.
                await _emit(ev.kind, ev.text or _arg_preview(getattr(ev, "raw", None)),
                            ev.tool, ev.path)
                # Live feed (not persisted): richer than the DB row — carries the
                # tool's args (search query, edit old→new) so the user is guided
                # through what the agent is actually doing, coder-mode style.
                emit_progress({
                    "kind": "agent", "sub": ev.kind, "tool": ev.tool or "",
                    "path": ev.path or "", "text": ev.text or "",
                    "detail": _arg_preview(getattr(ev, "raw", None)),
                })
                if ev.kind == "file_change" and ev.path:
                    if ev.path not in files:
                        files.append(ev.path)
                elif ev.kind == "message" and ev.text:
                    final_text = ev.text                  # last assistant message
                elif ev.kind == "completed":
                    ok = True
                    if ev.text:
                        final_text = ev.text
                    session_ref = str(ev.raw.get("session_id")
                                      or ev.raw.get("session_ref") or session_ref)
                elif ev.kind == "failed":
                    ok = False
                    error = ev.text or "run failed"
        except Exception as exc:  # noqa: BLE001 — normalize to a failed edit, never raise
            log.warning("selfedit_external_driver_failed",
                        driver=getattr(driver, "id", ""), error=repr(exc))
            ok, error = False, repr(exc)

        if conn is not None:
            with contextlib.suppress(Exception):
                await run_store.finish_run(
                    conn, run_id=run_id, user_id=req.user_id,
                    status="done" if ok else "failed", outcome=final_text,
                    error=error, files_changed=files, session_id=session_ref)

        log.info("selfedit_external_edit_done", driver=getattr(driver, "id", ""),
                 ok=ok, files=len(files))
        return EditResult(ok=ok, run_id=run_id, final_text=final_text, error=error)

    return _drive


async def build_selected_edit_driver(*, conn: Any, prefer: str = "",
                                     cwd: str = "/workspace", oauth_token: str = "",
                                     api_key: str = "", permission: str = _DEFAULT_PERMISSION,
                                     model: str = "", _select: Any = None) -> EditDriver | None:
    """Pick an available backend via the registry (``select_driver``) and wrap it
    as a self-edit ``EditDriver``. Returns None if no backend is runnable here so
    the caller can fall back. ``prefer`` is a driver id ("claude_code" | "codex").
    ``_select`` overrides the selector for tests."""
    if _select is None:
        from augmentum.coder.external.registry import select_driver as _select
    driver = await _select(prefer, cwd=cwd, claude_oauth_token=oauth_token,
                           claude_api_key=api_key)
    if driver is None:
        log.info("selfedit_no_external_backend", prefer=prefer)
        return None
    return run_external_edit_driver(conn=conn, driver=driver, permission=permission,
                                    model=model)
