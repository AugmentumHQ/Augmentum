"""Coding primitives — the companion wields a coder as one hand.

``code_dispatch`` hands a task to the Coding Driver (internal mission OR an
external harness agent) from any surface the companion speaks — voice, chat,
phone. ``code_status`` reads back the runs so the companion can answer "how's
the refactor going?" off-keyboard. This is the reverse of the agent bridge:
the personal AI reaching out to drive a coder, not the coder pinging the user.

Governance carries through the driver unchanged: explicit workspace/agent
scope, model never auto-selected (the companion must know which), runs
observable in the Agents window, risky steps gated to the phone. The
primitives only exist while ``companion_primitive_registry_active`` is on.
"""

from __future__ import annotations

from typing import Any

from augmentum.companion_runtime.primitives.base import (
    PrimitiveBase,
    PrimitiveContext,
    PrimitiveResult,
)
from augmentum.companion_runtime.primitives.registry import PrimitiveRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class CodeDispatchPrimitive(PrimitiveBase):
    name = "code_dispatch"
    description = (
        "Dispatch a coding task and supervise it. driver='internal' runs a "
        "background mission in a workspace (needs workspace_id + model — never "
        "auto-pick a model); driver='harness' assigns the task to a live "
        "external agent (needs agent_session_id). Returns a run_id you can "
        "check later with code_status; the user gets a notification when it's "
        "done or needs a decision."
    )

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        app_state = getattr(ctx.runtime, "_app_state", None)
        if app_state is None:
            return PrimitiveResult(ok=False, error="code_dispatch: no app_state")
        task = str(kwargs.get("task") or "").strip()
        if not task:
            return PrimitiveResult(ok=False, error="code_dispatch: task is required")
        driver = str(kwargs.get("driver") or "internal").strip().lower()

        from augmentum.coder.coding_driver import (
            HarnessCoderDriver,
            InternalCoderDriver,
        )

        try:
            if driver == "harness":
                agent = str(kwargs.get("agent_session_id") or "").strip()
                if not agent:
                    return PrimitiveResult(
                        ok=False,
                        error="code_dispatch: agent_session_id required for harness")
                res = await HarnessCoderDriver(app_state).dispatch(
                    user_id=ctx.user_id, agent_session_id=agent, task=task,
                    origin_surface="companion")
            else:
                ws = str(kwargs.get("workspace_id") or "").strip()
                model = str(kwargs.get("model") or "").strip()
                if not ws:
                    return PrimitiveResult(ok=False, error="code_dispatch: workspace_id required")
                if not model:
                    # Never auto-select a model for the user.
                    return PrimitiveResult(ok=False, error="code_dispatch: model required")
                res = await InternalCoderDriver(app_state).dispatch(
                    user_id=ctx.user_id, workspace_id=ws, task=task, model=model,
                    origin_surface="companion")
        except Exception as exc:
            log.exception("code_dispatch_failed", error=str(exc))
            return PrimitiveResult(ok=False, error=f"code_dispatch_failed: {exc!s}")

        if not res.get("ok"):
            return PrimitiveResult(ok=False, error=res.get("error") or "dispatch failed")
        return PrimitiveResult(ok=True, payload=res,
                               metadata={"run_id": res.get("run_id", "")})


class CodeStatusPrimitive(PrimitiveBase):
    name = "code_status"
    description = (
        "Read back the user's coding runs (internal missions + external "
        "agents) so you can report progress off-keyboard. Returns each run's "
        "task and status; pass workspace_id to scope."
    )

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        app_state = getattr(ctx.runtime, "_app_state", None)
        if app_state is None:
            return PrimitiveResult(ok=False, error="code_status: no app_state")
        from augmentum.coder import coding_driver
        try:
            runs = await coding_driver.list_internal_runs(
                app_state, user_id=ctx.user_id,
                workspace_id=str(kwargs.get("workspace_id") or "").strip(),
                limit=int(kwargs.get("limit", 10)))
        except Exception as exc:
            log.exception("code_status_failed", error=str(exc))
            return PrimitiveResult(ok=False, error=f"code_status_failed: {exc!s}")
        brief = [{"task": r["task"], "status": r["status"], "run_id": r["id"],
                  "driver": r["driver"]} for r in runs]
        return PrimitiveResult(ok=True, payload=brief,
                               metadata={"count": len(brief)})


PrimitiveRegistry.register(CodeDispatchPrimitive)
PrimitiveRegistry.register(CodeStatusPrimitive)
