"""End-to-end orchestrator smoke test against the mock adapter.

This is the single most valuable test in the package -- it exercises
schema validation, log persistence, the fast-path queue, the slow-
path agent, and the surface adapter contract in one go.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from augmentum.game_agent.orchestrator import Orchestrator
from augmentum.game_agent.schema import EventPayload
from augmentum.game_agent.surfaces.mock import MockAdapter, ScriptedEvent


async def _stub_llm_emits_one_action(_prompt: str, _frames: list[bytes]) -> str:
    """A deterministic stub LLM that always emits one ``advance`` action.

    Real backends arrive via the SlowPathLLM Protocol; tests inject
    this stub so we can assert orchestrator behavior without the
    network.
    """

    return json.dumps(
        {
            "observations": ["acting on whatever just arrived"],
            "state_update": "turn taken",
            "actions": [{"semantic": "advance", "duration_ms": 100}],
            "confidence": 0.5,
            "next_check_in_ms": 200,
        }
    )


@pytest.mark.asyncio
async def test_end_to_end_session_writes_well_formed_log(tmp_path: Path) -> None:
    """@example: an end-to-end session produces a parseable log with all kinds.

    Exercises:
      * session header + surface_caps emission
      * scripted-event arrival via the mock
      * slow-path planner emitting actions
      * action worker applying them via the resolver
      * session_end trailer
    """

    log_path = tmp_path / "session.ndjson"
    script = [
        ScriptedEvent(
            delay_ms=50,
            payload=EventPayload(channel="log", data={"event": "spawn"}),
        ),
        ScriptedEvent(
            delay_ms=150,
            payload=EventPayload(channel="log", data={"event": "tick", "n": 1}),
        ),
    ]
    adapter = MockAdapter(script=script)

    orch = Orchestrator(
        log_path=str(log_path),
        surface_kind="mock",
        adapter=adapter,
        llm=_stub_llm_emits_one_action,
        objective="emit some events",
    )

    async def stopper() -> None:
        await asyncio.sleep(0.5)
        orch.stop("completed")

    stop_task = asyncio.create_task(stopper())
    end = await orch.run()
    await stop_task

    # Log was finalized cleanly.
    assert end.reason == "completed"

    # Parse the log and assert the canonical kinds appeared.
    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    kinds = [entry["kind"] for entry in lines]

    assert kinds[0] == "session"
    assert "surface_caps" in kinds
    assert "event" in kinds
    assert "plan" in kinds
    assert "input" in kinds
    assert kinds[-1] == "session_end"

    # The adapter recorded at least one resolver invocation.
    assert adapter.recorded_inputs
    assert adapter.recorded_inputs[0][0] == "advance"


@pytest.mark.asyncio
async def test_orchestrator_logs_slow_path_parse_errors(tmp_path: Path) -> None:
    """@example: a broken LLM response is recorded as agent_error and the loop survives.

    ROOT CAUSE:
      Earlier drafts crashed the slow-path task when the model emitted
      non-JSON; the orchestrator must catch PlanParseError, write an
      agent_error entry, and keep ticking.
    """

    log_path = tmp_path / "session.ndjson"

    async def broken_llm(_prompt: str, _frames: list[bytes]) -> str:
        return "this is not JSON at all"

    adapter = MockAdapter(script=[])
    orch = Orchestrator(
        log_path=str(log_path),
        surface_kind="mock",
        adapter=adapter,
        llm=broken_llm,
        objective="trigger a parse error",
    )

    async def stopper() -> None:
        await asyncio.sleep(0.3)
        orch.stop("completed")

    stop_task = asyncio.create_task(stopper())
    await orch.run()
    await stop_task

    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    error_entries = [e for e in lines if e["kind"] == "agent_error"]
    assert error_entries, "expected at least one agent_error entry"
    assert error_entries[0]["payload"]["where"] == "slow_path"
