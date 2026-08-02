"""Pydantic schemas for the game-agent universal log + adapter contracts.

The live log is an append-only NDJSON file. Every line is a single
:class:`LogEntry` discriminated by ``kind``; every other module in this
package consumes or produces those entries. If you're adding a new
event kind, this file is the source of truth.

Schema versioning
-----------------
Each session declares a top-level ``schema_version`` (currently
``game_agent.v1``). The structural fields on ``LogEntry`` are stable
across schema versions; the surface-specific vocabulary inside
``EventPayload.data`` is *not*, and is versioned separately via the
``log_schema`` field of :class:`SurfaceCapsPayload` (e.g.
``"luanti.v1"``, ``"emulator.v1"``).

This separation lets the agent prompt remain universal while
surface adapters evolve their own event vocabularies independently.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Type aliases ──────────────────────────────────────────────────────

SurfaceKind = Literal["js13k", "luanti", "emulator", "emulatorjs", "curated", "mock"]
"""The real surfaces plus a mock for tests.

* ``js13k``      -- iframe + HTML5 game (catalog from ``augmentum/games/``)
* ``luanti``     -- Lua-mod-bridged voxel server
* ``emulator``   -- streamed-container native emulator (Dolphin et al.)
* ``emulatorjs`` -- in-browser libretro WASM core via EmulatorJS
* ``curated``    -- Cradle-style screen capture + synthetic input
* ``mock``       -- scripted test double
"""

EventChannel = Literal["log", "vlm", "ocr", "memory"]
"""Where an :class:`EventPayload` came from (i.e. who derived it).

* ``log``    -- structured event from the surface adapter (Lua mod, postMessage, etc.)
* ``vlm``    -- slow-path vision-language-model observation
* ``ocr``    -- text extracted from frames by a deterministic OCR pass
* ``memory`` -- recalled long-term memory injected as a synthetic observation
"""

ObservationModality = Literal["log", "frame", "ocr", "memory"]
"""What observation *sources* a surface offers.

Distinct from :data:`EventChannel`: the orchestrator uses this to
decide whether to pull a frame on slow-path turns, whether OCR is
budgeted, etc. ``frame`` here means "I have a screen / canvas you can
sample", not "I emit ``frame`` entries into the log".
"""

InputSource = Literal["agent", "human", "rule"]
"""Who emitted an input.

* ``agent`` -- the slow-path planner decided to act
* ``human`` -- a human is co-piloting (recorded for training/replay)
* ``rule``  -- the fast-path rule engine fired automatically
"""

SchemaVersion = Literal["game_agent.v1"]


# ── Common payload types ──────────────────────────────────────────────


class PlanAction(BaseModel):
    """A single semantic action the agent wants the surface to execute.

    ``semantic`` is a string that must be a member of the active
    :class:`SurfaceCapsPayload.semantic_inputs` list; the surface
    adapter is responsible for resolving it to a wire-format input
    (key event, gamepad press, Lua RPC, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    semantic: str = Field(..., min_length=1, max_length=64)
    duration_ms: int = Field(..., ge=10, le=2000)
    # Optional argument for quickaction semantics (``type_text`` puts
    # the string to type here). Plain button presses leave it None.
    text: str | None = Field(default=None, max_length=16)
    # Chord: up to two extra semantics HELD simultaneously with
    # ``semantic`` for the same duration. This is what makes real-time
    # action games playable (hold run + press jump); turn-based games
    # never need it. Adapters without native chord support degrade to
    # sequential presses in the action worker.
    also: list[str] | None = Field(default=None, max_length=2)


# ── Per-kind payloads ─────────────────────────────────────────────────


class SessionPayload(BaseModel):
    """Header line written once at session start."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1)
    surface: SurfaceKind
    objective: str = Field(..., min_length=1, max_length=2048)
    schema_version: SchemaVersion = "game_agent.v1"
    started_at_unix_ms: int = Field(..., ge=0)


class SurfaceCapsPayload(BaseModel):
    """What a surface can accept and produce.

    The agent prompt is parameterized by this:

    * ``semantic_inputs`` is the *only* vocabulary the slow path may
      emit. Anything outside this list is rejected at planning time.
    * ``log_schema`` names the surface-specific event vocabulary the
      agent should expect to see inside :class:`EventPayload.data`.
    * ``observation_modalities`` tells the orchestrator which channels
      are actually wired; agents will not be asked for VLM analysis
      on a surface that lists only ``["log"]``.
    """

    model_config = ConfigDict(extra="forbid")

    semantic_inputs: list[str] = Field(..., min_length=1, max_length=128)
    log_schema: str = Field(..., min_length=1)
    observation_modalities: list[ObservationModality] = Field(..., min_length=1)
    # Phase G — universal control schema. Optional per-semantic
    # human-readable hint shown to the slow-path agent in the prompt's
    # INPUT_HINTS block. When supplied, the model understands what each
    # semantic does in THIS game (e.g., "confirm = advance Pokémon
    # dialog") rather than reasoning about raw button effects.
    input_hints: dict[str, str] | None = Field(default=None)
    # Phase G — diagnostic / replay surface. When a ComposedProfile
    # was used to derive the vocabulary, log which profiles were
    # active so session replay and journal entries stay unambiguous.
    controller_profile: str | None = Field(default=None, max_length=64)
    game_profile: str | None = Field(default=None, max_length=64)


class InputPayload(BaseModel):
    """An input that was emitted *into* the surface."""

    model_config = ConfigDict(extra="forbid")

    semantic: str = Field(..., min_length=1, max_length=64)
    duration_ms: int = Field(..., ge=0, le=10_000)
    source: InputSource
    # Chord members held simultaneously with ``semantic`` (see
    # PlanAction.also). None for plain presses — the common case — so
    # existing logs replay unchanged.
    also: list[str] | None = Field(default=None, max_length=2)


class EventPayload(BaseModel):
    """An observation that arrived *from* the surface or a deriver.

    ``data`` is intentionally untyped at this layer -- its vocabulary
    is defined by the surface's ``log_schema`` and may carry anything
    JSON-encodable. Validators that need the structured shape should
    parse ``data`` against a schema-specific model.
    """

    model_config = ConfigDict(extra="forbid")

    channel: EventChannel
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


CompanionMood = Literal[
    "neutral", "happy", "sad", "surprised", "concerned", "determined", "amused",
]
"""Coarse emotional state the avatar + voice prosody key off of."""

CompanionIntent = Literal["chat", "react", "encourage", "question", "silent"]
"""What *kind* of utterance this is. ``silent`` is the safe default and
means the companion has nothing to say this turn — typed separately
from an empty ``say`` so a planner can mark "I deliberately stayed
silent" vs "I forgot to fill the field"."""


class PlanPayload(BaseModel):
    """The slow-path agent's output for one turn.

    This is the schema the strict-prompt VLM is constrained to.
    Mirrors :func:`augmentum.game_agent.prompt.SLOW_PATH_PROMPT`
    output contract one-to-one; if you change this you must update
    that prompt.

    Companion-mode fields (``say`` / ``mood`` / ``intent``) are
    *always present* on the schema but default to silent — solo
    sessions just never fill them. The companion prompt addendum
    flips that.
    """

    model_config = ConfigDict(extra="forbid")

    observations: list[str] = Field(default_factory=list, max_length=10)
    state_update: str = Field(default="", max_length=2048)
    actions: list[PlanAction] = Field(default_factory=list, max_length=8)
    confidence: float = Field(..., ge=0.0, le=1.0)
    next_check_in_ms: int = Field(..., ge=50, le=30_000)
    # Companion-mode additions. Defaults make them inert in solo
    # sessions; the companion prompt unlocks them.
    say: str = Field(default="", max_length=400)
    mood: CompanionMood = "neutral"
    intent: CompanionIntent = "silent"
    # Persistent journal patches. The agent emits partial JSON to
    # update its long-running knowledge store keyed by (user, title).
    # The orchestrator validates + merges; CompanionJournal enforces
    # caps on string length and notes list size, so a runaway model
    # can't blow up the prompt budget over many turns. None / missing
    # means "no journal change this turn" -- the prevailing journal
    # carries forward unchanged.
    journal_update: dict[str, Any] | None = Field(default=None)
    # Model-authored reflex rules (tier 0): declarative condition→action
    # specs compiled onto the fast-path RuleEngine so proven repetitive
    # reactions (advance dialog, …) fire at RAM-tick speed without any
    # LLM call. Validated + bounded by augmentum/game_agent/reflex.py;
    # {"id": ..., "retract": true} removes one. None = no change.
    reflex_rules: list[dict[str, Any]] | None = Field(default=None, max_length=6)
    # Cross-title PLAYBOOK patches — same merge shape as journal_update
    # but routed to the per-user playbook: mechanics and interface
    # lessons that TRANSFER between games ("dialog swallows movement",
    # "grass = wild encounters"). Title-specific facts stay in
    # journal_update. None = no change.
    playbook_update: dict[str, Any] | None = Field(default=None)
    # Goal-stack patches: per-horizon {"final"|"medium"|"short":
    # "<text>" | {"text": ..., "metric": {"probe","op","value"}}}.
    # Metrics make progress MEASURABLE (the stall watchdog and the
    # correct-move metric key off them). None = goals unchanged.
    goal_update: dict[str, Any] | None = Field(default=None)


class RuleFiredPayload(BaseModel):
    """A fast-path rule matched and emitted actions."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(..., min_length=1)
    matched: dict[str, Any] = Field(default_factory=dict)
    emitted_actions: list[PlanAction] = Field(default_factory=list)


class AgentErrorPayload(BaseModel):
    """Something went wrong inside a path. Non-fatal by default."""

    model_config = ConfigDict(extra="forbid")

    where: Literal["fast_path", "slow_path", "adapter"]
    message: str = Field(..., min_length=1, max_length=4096)
    recoverable: bool = True


class SessionEndPayload(BaseModel):
    """Trailer line written once at session termination."""

    model_config = ConfigDict(extra="forbid")

    reason: Literal["completed", "aborted", "timeout", "error", "user_stopped"]
    duration_ms: int = Field(..., ge=0)
    # Machine-checked progress scorecard (augmentum.game_agent.progress
    # .ProgressScore.to_dict). Externally defined and NOT authorable by
    # the planner, so two runs of the same game are comparable across a
    # prompt/rule/model change. None only when scoring itself failed —
    # a session must never be lost because grading it raised.
    progress: dict[str, Any] | None = None


# ── Discriminated entry types ─────────────────────────────────────────


class _BaseEntry(BaseModel):
    """Shared shape across every log line.

    ``t`` is milliseconds since session start; the orchestrator sources
    a monotonic clock at start time so replay is deterministic.
    """

    model_config = ConfigDict(extra="forbid")

    t: int = Field(..., ge=0)


class SessionEntry(_BaseEntry):
    kind: Literal["session"] = "session"
    payload: SessionPayload


class SurfaceCapsEntry(_BaseEntry):
    kind: Literal["surface_caps"] = "surface_caps"
    payload: SurfaceCapsPayload


class InputEntry(_BaseEntry):
    kind: Literal["input"] = "input"
    payload: InputPayload


class EventEntry(_BaseEntry):
    kind: Literal["event"] = "event"
    payload: EventPayload


class PlanEntry(_BaseEntry):
    kind: Literal["plan"] = "plan"
    payload: PlanPayload


class RuleFiredEntry(_BaseEntry):
    kind: Literal["rule_fired"] = "rule_fired"
    payload: RuleFiredPayload


class AgentErrorEntry(_BaseEntry):
    kind: Literal["agent_error"] = "agent_error"
    payload: AgentErrorPayload


class SessionEndEntry(_BaseEntry):
    kind: Literal["session_end"] = "session_end"
    payload: SessionEndPayload


LogEntry = Annotated[
    SessionEntry | SurfaceCapsEntry | InputEntry | EventEntry | PlanEntry | RuleFiredEntry | AgentErrorEntry | SessionEndEntry,
    Field(discriminator="kind"),
]
"""A single line of the live log.

Use :func:`pydantic.TypeAdapter(LogEntry).validate_python` to parse a
raw dict into the correct subclass; the ``kind`` field discriminates.
"""
