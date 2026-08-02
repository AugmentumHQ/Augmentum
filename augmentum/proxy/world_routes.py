"""World-model routes — drive Qwen-AgentWorld on Slot B.

Thin HTTP surface over :class:`augmentum.world_model.AgentWorldDriver`
for the AZR gym orchestrator, dev tooling, and manual probing. The
heavy lifting (domain prompts, history validation, sampling, refusal
when Slot B isn't serving a world model) lives in the driver.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from augmentum.utils.logging import get_logger
from augmentum.world_model import (
    WORLD_DOMAINS,
    AgentWorldDriver,
    WorldModelUnavailable,
)

log = get_logger(__name__)

router = APIRouter(prefix="/api/world", tags=["world-model"])


def _driver(request: Request) -> AgentWorldDriver:
    return AgentWorldDriver(getattr(request.app.state, "secondary_slot", None))


class SimulateBody(BaseModel):
    domain: str
    # Episode so far: agent actions as ``user`` turns, prior observations
    # as ``assistant`` turns, ending on the action to simulate. The
    # domain system prompt is added server-side — don't send one.
    messages: list[dict[str, str]] = Field(min_length=1)
    # Detailed description of the starting environment (cwd, files,
    # versions, app state...). Upstream found this drives sim quality.
    initial_state: str = ""
    system_override: str = ""
    # Serialize the episode into ONE user message (the trained format).
    # False keeps raw multi-turn chat framing — off-distribution, for
    # experiments only.
    serialized: bool = True
    # Thinking shares this budget and runs long on hard steps — keep
    # headroom (upstream evals at 32768).
    max_tokens: int = Field(default=8192, ge=64, le=32768)
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    # Include the model's <think> reasoning and unextracted raw content
    # in the response — useful for debugging simulator behavior, noise
    # for episode rollouts.
    include_thinking: bool = False


@router.get("/status")
async def world_status(request: Request) -> dict[str, Any]:
    """Whether a world model is currently servable, and which domains."""
    return _driver(request).status()


@router.post("/simulate")
async def world_simulate(body: SimulateBody, request: Request) -> dict[str, Any]:
    """Predict the next environment observation for an episode step."""
    if body.domain not in WORLD_DOMAINS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown domain {body.domain!r} — expected one of {list(WORLD_DOMAINS)}",
        )
    try:
        step = await _driver(request).simulate(
            body.domain,
            body.messages,
            initial_state=body.initial_state,
            system_override=body.system_override,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            serialized=body.serialized,
        )
    except WorldModelUnavailable as exc:
        # 409, not 500: the server is fine — the slot just isn't serving
        # a world model. The detail says exactly what to load and where.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result: dict[str, Any] = {
        "observation": step.observation,
        "tag_found": step.tag_found,
        "model": step.model,
        "domain": step.domain,
        "latency_ms": step.latency_ms,
        "prompt_tokens": step.prompt_tokens,
        "completion_tokens": step.completion_tokens,
    }
    if body.include_thinking:
        result["thinking"] = step.thinking
        result["raw_content"] = step.raw_content
    return result
