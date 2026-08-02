"""Game-agent primitive — wraps ``game_agent.orchestrator.Orchestrator``.

Becca's game-playing capability. Sprint 0 framing: Becca tries games
in dormant time, logs failures as DPO-retrievable lessons. The adapter
launches a session with a supplied adapter (game-specific binding) and
LLM, then awaits the end-of-session payload.

Practically inert in Sprint 2 — Sprint 4a's autonomous tick is the
caller that will routinely invoke this.
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


class GameAgentPrimitive(PrimitiveBase):
    name = "game_agent"
    description = (
        "Run a game-agent session. Caller supplies the surface adapter "
        "and LLM; primitive drives the run loop and returns the session "
        "end payload."
    )

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        adapter = kwargs.get("adapter")
        llm = kwargs.get("llm")
        if adapter is None or llm is None:
            return PrimitiveResult(
                ok=False,
                error="game_agent: need `adapter` and `llm` kwargs",
            )

        try:
            from augmentum.game_agent.orchestrator import Orchestrator
        except Exception as exc:
            return PrimitiveResult(
                ok=False,
                error=f"game_agent_import_failed: {exc!s}",
            )

        try:
            orch = Orchestrator(adapter=adapter, llm=llm, **{
                k: v for k, v in kwargs.items() if k not in ("adapter", "llm")
            })
            payload = await orch.run()
        except Exception as exc:
            log.exception("game_agent_run_failed", error=str(exc))
            return PrimitiveResult(ok=False, error=f"game_agent_run_failed: {exc!s}")

        return PrimitiveResult(ok=True, payload=payload)


PrimitiveRegistry.register(GameAgentPrimitive)
