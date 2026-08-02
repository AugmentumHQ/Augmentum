"""Augmentum Game Agent.

Universal substrate for AI agents controlling any of Augmentum's four
gaming surfaces (js13k browser games, Luanti-style scriptable
servers, streamed emulators, and Cradle-style curated screen + input).
The cognition is uniform across surfaces; only the surface adapter
varies.

Architecture in one paragraph
-----------------------------
A session has one append-only NDJSON log, one surface adapter, one
fast-path rule engine (reflex layer, ~100 Hz), and one slow-path
planner backed by a user-supplied LLM. The adapter publishes
observations into the log; the rule engine matches predicates and may
emit immediate actions; the planner reads a tail of the log on its
own cadence and produces a strict-JSON plan. The same NDJSON file is
the source of truth for both paths and for any downstream tool
(replay, evaluation, debugging).

Public surface (re-exported here for stability):

* :class:`Orchestrator` -- session driver
* :class:`SlowPathAgent` / :data:`SlowPathLLM` -- LLM-backed planner
* :class:`RuleEngine`, :class:`Rule` -- fast-path reflex layer
* :class:`SemanticInputResolver` -- semantic-id binding registry
* :class:`LiveLog`, :func:`read_log`, :func:`tail_log` -- NDJSON IO
* :data:`SLOW_PATH_PROMPT`, :func:`parse_plan_output` -- prompt + parsing
* Surface adapters: :class:`MockAdapter`, :class:`Js13kAdapter`,
  :class:`LuantiAdapter`, :class:`EmulatorAdapter`,
  :class:`CuratedAdapter`

See ``augmentum/game_agent/README.md`` for the design rationale and
the path for wiring a new surface adapter.
"""

from __future__ import annotations

from augmentum.game_agent.agent import (
    PlanParseError,
    SlowPathAgent,
    SlowPathLLM,
    SlowPathLLMProtocol,
)
from augmentum.game_agent.companion import CompanionPersona, build_identity_prefix
from augmentum.game_agent.log import LiveLog, SessionClock, read_log, tail_log, validate_entries
from augmentum.game_agent.orchestrator import Orchestrator
from augmentum.game_agent.progress import (
    ProgressScore,
    score_from_world,
    score_session,
)
from augmentum.game_agent.prompt import (
    SLOW_PATH_PROMPT,
    build_slow_path_inputs,
    parse_plan_output,
)
from augmentum.game_agent.rules import Rule, RuleEngine, RuleMatch
from augmentum.game_agent.schema import (
    EventChannel,
    EventPayload,
    InputPayload,
    InputSource,
    LogEntry,
    ObservationModality,
    PlanAction,
    PlanPayload,
    SessionEndPayload,
    SessionPayload,
    SurfaceCapsPayload,
    SurfaceKind,
)
from augmentum.game_agent.semantic import SemanticInputResolver, UnknownSemanticError
from augmentum.game_agent.surfaces import (
    BridgedAdapter,
    CuratedAdapter,
    EmitEventFn,
    EmulatorAdapter,
    Js13kAdapter,
    LuantiAdapter,
    MockAdapter,
    ScriptedEvent,
    SurfaceAdapter,
)
from augmentum.game_agent.voice_bridge import VoiceBridge

__all__ = [
    "BridgedAdapter",
    "CompanionPersona",
    "CuratedAdapter",
    "EmitEventFn",
    "EmulatorAdapter",
    "EventChannel",
    "EventPayload",
    "InputPayload",
    "InputSource",
    "Js13kAdapter",
    "LiveLog",
    "LogEntry",
    "LuantiAdapter",
    "MockAdapter",
    "ObservationModality",
    "Orchestrator",
    "PlanAction",
    "PlanParseError",
    "PlanPayload",
    "ProgressScore",
    "Rule",
    "RuleEngine",
    "RuleMatch",
    "SLOW_PATH_PROMPT",
    "ScriptedEvent",
    "SemanticInputResolver",
    "SessionClock",
    "SessionEndPayload",
    "SessionPayload",
    "SlowPathAgent",
    "SlowPathLLM",
    "SlowPathLLMProtocol",
    "SurfaceAdapter",
    "SurfaceCapsPayload",
    "SurfaceKind",
    "UnknownSemanticError",
    "VoiceBridge",
    "build_identity_prefix",
    "build_slow_path_inputs",
    "parse_plan_output",
    "read_log",
    "score_from_world",
    "score_session",
    "tail_log",
    "validate_entries",
]
