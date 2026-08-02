"""World-model subsystem — language world models as environment simulators.

Currently drives Qwen-AgentWorld (Slot B resident): given an agent's
action history, the model predicts the environment's next observation
across seven domains (mcp, search, terminal, swe, android, web, os).

Consumers:
- AZR gym (``docs``: agentic-lane episode rollouts — sim mints data,
  real environments gate promotion)
- inference-time lookahead (simulate a tool call before executing)
- ``judging`` — the vendored AgentWorldBench judge contract for the
  sim-vs-real agreement harness

The driver never loads a model itself — the user loads AgentWorld into
Slot B from the model manager ("2nd slot"); the driver only refuses
clearly when the slot is empty or holds a non-world-model.
"""

from augmentum.world_model.driver import (
    RESPONSE_MARKER,
    RESPONSE_TAG,
    WORLD_DOMAINS,
    AgentWorldDriver,
    WorldModelUnavailable,
    WorldStep,
    extract_observation,
    serialize_episode,
)

__all__ = [
    "RESPONSE_MARKER",
    "RESPONSE_TAG",
    "WORLD_DOMAINS",
    "AgentWorldDriver",
    "WorldModelUnavailable",
    "WorldStep",
    "extract_observation",
    "serialize_episode",
]
