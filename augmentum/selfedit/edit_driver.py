"""The edit driver — runs the editing agent (Claude Code) against a candidate.

This is the concrete ``EditDriver`` the orchestrator's seam expects: it drives a
Claude run over the candidate worktree, persists it as a ``claude_run`` (so the
edit transcript is durable + resumable, linked from the attempt's ``run_id``),
and returns an ``EditResult``.

The execution *environment* is injected as a ``command_runner`` — the seam where
the dedicated RW-repo (B1) container, or a host subprocess in the dev-bind case,
plugs in. That keeps the driver's event-handling + persistence fully testable
with a fake runner (feed it canned JSONL), while the container/credential wiring
is the caller's concern. The stream consumer is the SAME
``ClaudeStreamCollector`` the live coder route uses — one consumer, no drift.

Not exported from ``augmentum.selfedit.__init__`` on purpose: importing it pulls
in ``coder.external`` (the run engine), which the pure orchestrator/harness don't
need. Import it where you wire the loop.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from augmentum.coder.external import run_store
from augmentum.coder.external.base import ExternalTask
from augmentum.coder.external.claude_cli import build_claude_argv
from augmentum.coder.external.stream import ClaudeStreamCollector
from augmentum.selfedit.orchestrator import EditDriver, EditRequest, EditResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# A command runner executes ``argv`` in the candidate's context, streaming stdout
# bytes to ``on_chunk``. The container/subprocess seam. Called as:
#   await runner(request=req, argv=[...], on_chunk=fn, environment={...})
CommandRunner = Callable[..., Awaitable[None]]


def run_engine_edit_driver(
    *, conn: Any, command_runner: CommandRunner, token: str = "",
    permission: str = "acceptEdits", config_dir: str = "",
    extra_env: dict | None = None,
) -> EditDriver:
    """Build an ``EditDriver`` that runs Claude over the candidate via
    ``command_runner`` and records the run.

    ``permission`` defaults to ``acceptEdits`` — a self-edit's whole point is to
    write files; a read-only run would always produce a no-op (→ rejected). The
    autonomy gate lives upstream (which attempts run at all), not in starving the
    agent of edit rights."""
    async def _drive(req: EditRequest) -> EditResult:
        run_id = uuid.uuid4().hex

        # Record the run up front so it's visible/resumable even if it crashes.
        if conn is not None:
            try:
                await run_store.create_run(
                    conn, run_id=run_id, user_id=req.user_id,
                    workspace_id=req.candidate.name, task=req.objective,
                    permission=permission,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort persistence
                log.warning("selfedit_driver_create_run_failed", error=repr(exc))

        seq = {"n": 0}

        async def _emit(item: dict) -> None:
            if conn is None:
                return
            seq["n"] += 1
            # a persistence hiccup can't sink the run
            with contextlib.suppress(Exception):
                await run_store.add_event(
                    conn, run_id=run_id, user_id=req.user_id, seq=seq["n"],
                    kind=item.get("kind", ""), text=item.get("text", ""),
                    tool=item.get("tool", ""), path=item.get("path", ""),
                )

        async def _on_session(sid: str) -> None:
            if conn is not None:
                with contextlib.suppress(Exception):
                    await run_store.set_session_id(
                        conn, run_id=run_id, user_id=req.user_id, session_id=sid,
                    )

        collector = ClaudeStreamCollector(emit=_emit, on_session_id=_on_session)
        et = ExternalTask(prompt=req.objective, workspace=req.candidate.path,
                          permission=permission)
        argv = build_claude_argv(et)
        env = dict(extra_env or {})
        if token:
            # Route by credential TYPE: an OAuth subscription token →
            # CLAUDE_CODE_OAUTH_TOKEN, an API key (sk-ant-api…) → ANTHROPIC_API_KEY.
            # Hardcoding the OAuth var broke API-key users. auth_env handles both.
            from augmentum.coder.external.claude_auth import auth_env
            env.update(auth_env(oauth_token=token))
        if config_dir:
            env["CLAUDE_CONFIG_DIR"] = config_dir

        try:
            await command_runner(
                request=req, argv=argv, on_chunk=collector.on_chunk, environment=env,
            )
        except Exception as exc:  # noqa: BLE001 — normalize to a failed edit, never raise
            log.warning("selfedit_driver_run_failed",
                        attempt_id=req.attempt_id, error=repr(exc))
            collector.err = collector.err or repr(exc)
        await collector.flush()

        if conn is not None:
            try:
                await run_store.finish_run(
                    conn, run_id=run_id, user_id=req.user_id, status=collector.status,
                    outcome=collector.outcome, error=collector.err,
                    files_changed=collector.files, raw_jsonl=collector.raw_jsonl,
                    session_id=collector.meta["session_id"],
                    cost_usd=collector.meta["cost_usd"],
                    num_turns=collector.meta["num_turns"],
                    duration_ms=collector.meta["duration_ms"],
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("selfedit_driver_finish_run_failed", error=repr(exc))

        return EditResult(
            ok=collector.ok, run_id=run_id,
            final_text=collector.outcome, error=collector.err,
        )

    return _drive
