"""Slow-path planner runner.

This module wires the strict prompt to a user-supplied LLM and turns
its output into a validated :class:`PlanPayload`. The actual LLM call
is a Protocol -- the orchestrator injects whatever backend is wired
(Augmentum's own ``ModelBackend``, the bundled engine, a remote
OpenAI-compatible endpoint, a test stub, etc.).

Failure handling
----------------
* JSON parse / schema errors raise :class:`PlanParseError` -- the
  orchestrator decides whether to retry with a "your last reply was
  invalid, here is the schema again" hint or surrender to fast-path
  alone for the next tick.
* Unknown semantics (model emits an id not in caps) are treated the
  same way: parse error.
* Transport-level failures (timeout, 5xx) raise whatever the LLM
  Protocol implementation raises; not our problem to translate.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeAlias

from augmentum.game_agent.companion import CompanionPersona
from augmentum.game_agent.prompt import (
    FastPlan,
    PlanParseError,
    build_fast_delta,
    build_fast_system_prompt,
    build_full_prompt,
    parse_fast_output,
    parse_plan_output,
)
from augmentum.game_agent.schema import PlanPayload, SurfaceCapsPayload

SlowPathLLM: TypeAlias = Callable[[str, Sequence[bytes]], Awaitable[str]]
"""Async callable: ``(prompt_text, frames_oldest_first) -> raw_model_output``.

The orchestrator wires a concrete implementation; the agent module
itself never reaches the network. This keeps the agent unit-testable
with a synchronous stub.

``frames`` is a sequence of PNG byte strings in oldest-first order.
An empty sequence means "no frames this turn" (text-only inputs).
Sending multiple frames gives the model temporal context -- the
established pattern for video-VLM benchmarks -- so it can reason
about motion, animation progress, and action causality rather than
inferring everything from a single still.
"""


class SlowPathLLMProtocol(Protocol):
    """Class-style equivalent of :data:`SlowPathLLM` for backends that prefer
    a callable object over a bare function."""

    async def __call__(self, prompt: str, frames: Sequence[bytes]) -> str:
        ...


class SlowPathAgent:
    """One slow-path planner attached to one session.

    Use when:
    - The orchestrator needs to invoke the planner at the cadence
      requested by the previous plan's ``next_check_in_ms`` (or on a
      novelty signal from the fast path).

    Expects:
    - The LLM callable is hot and authenticated.
    - The caller supplies the live-log tail it wants the model to see;
      :class:`SlowPathAgent` does no tail-trimming -- token budgeting
      is the orchestrator's responsibility because it owns the log
      reader.

    Returns:
    - A validated :class:`PlanPayload` on success; the orchestrator
      logs it as a :class:`PlanEntry` and applies its actions.
    """

    def __init__(
        self,
        *,
        llm: SlowPathLLM,
        surface_kind: str,
        caps: SurfaceCapsPayload,
        objective: str,
        companion: bool = False,
        persona: CompanionPersona | None = None,
    ) -> None:
        self._llm = llm
        self._surface_kind = surface_kind
        self._caps = caps
        self._objective = objective
        self._companion = companion
        self._persona = persona
        self._state: str = ""

    @property
    def state(self) -> str:
        """The agent scratchpad. Visible to tests + replayers."""

        return self._state

    async def think(
        self,
        *,
        live_log_tail: list[dict[str, object]],
        frames: Sequence[bytes] = (),
        overlay: dict[str, object] | None = None,
        journal: dict[str, object] | None = None,
        frame_note: str = "",
        lore: list[str] | None = None,
        lore_summary: list[str] | None = None,
        playbook: dict[str, object] | None = None,
    ) -> PlanPayload:
        """Run one slow-path turn.

        Builds the prompt, calls the LLM, parses the response, updates
        the agent's persistent scratchpad. The orchestrator is
        responsible for logging the resulting :class:`PlanPayload`.

        @param frames:
            Recent frames in oldest-first order. Multi-frame prompts
            give the model temporal context (motion, animation, action
            causality). Empty sequence means no frames this turn; the
            prompt's FRAMES section reports that honestly so the model
            doesn't hallucinate visual data.
        @param overlay:
            Latest structured world state from RAM probes (or any
            adapter-emitted ``probes`` dict). Rendered as a high-priority
            OVERLAY block in the prompt so the model sees decoded state
            alongside the frames -- the production pattern adopted by
            Claude Plays Pokemon (player coords, party HP, dialog flag)
            and PufferAI/pokegym (``ram_map`` overlay). ``None`` when
            the surface emits no probe events; the OVERLAY block is
            omitted entirely in that case.
        """

        full_prompt = build_full_prompt(
            companion=self._companion,
            surface_kind=self._surface_kind,
            caps=self._caps,
            objective=self._objective,
            state=self._state,
            live_log_tail=live_log_tail,
            n_frames=len(frames),
            persona=self._persona,
            overlay=overlay,
            journal=journal,
            frame_note=frame_note,
            lore=lore,
            lore_summary=lore_summary,
            playbook=playbook,
        )
        raw = await self._llm(full_prompt, frames)
        plan = parse_plan_output(raw, self._caps)
        # State carry-over is the only persistent thing across calls;
        # everything else is in the log.
        self._state = plan.state_update
        return plan


SlowPathChatLLM: TypeAlias = Callable[..., Awaitable[dict]]
"""Async callable for the fast-turn rolling window.

Called as ``chat_llm(messages)`` or ``chat_llm(messages, options)``.
``messages`` is a list of ``{"role", "content", "images"}`` dicts,
oldest first (``images`` = raw PNG bytes or None). ``options`` is an
optional dict of decoding constraints — ``{"json_schema": {...}}``
grammar-locks the reply on llama-server backends (implementations that
can't honor it just ignore it). Returns a dict with at least
``"text"``; timing keys (``latency_ms``, ``tok_s``, ``cached_tokens``,
``completion_tokens``) are optional telemetry.
"""


def _fast_output_schema(caps: SurfaceCapsPayload) -> dict:
    """JSON schema for the micro-plan, action names enum-locked to caps.

    A 2B model under pressure emits broken JSON and invented semantics;
    grammar-constrained decoding makes both structurally impossible —
    llama-server compiles this to a GBNF grammar server-side. Mirrors
    :func:`parse_fast_output`'s contract exactly (which still runs, as
    the safety net for backends that ignored the schema).
    """

    return {
        "type": "object",
        "properties": {
            "a": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "object",
                    # Property order = the order the grammar makes the
                    # model write them: name, then argument, then
                    # duration. ``text`` is REQUIRED (empty string for
                    # plain buttons) — leaving it optional taught the 2B
                    # to emit type_text/navigate_to with no argument at
                    # all (observed live, run 19).
                    "properties": {
                        "s": {"type": "string", "enum": list(caps.semantic_inputs)},
                        "text": {"type": "string", "maxLength": 16},
                        "d": {"type": "integer", "minimum": 10, "maximum": 2000},
                    },
                    "required": ["s", "text", "d"],
                },
            },
            "why": {"type": "string", "maxLength": 100},
            "next_ms": {"type": "integer", "minimum": 50, "maximum": 30000},
            "esc": {"type": "boolean"},
        },
        "required": ["a", "why", "next_ms", "esc"],
    }


class FastTurnRunner:
    """Rolling call-window micro-planner ("call mode").

    Use when:
    - The orchestrator wants sub-second action decisions between FULL
      planning turns. Holds the multi-turn chat window whose append-only
      shape keeps the server-side KV prefix cache hot.

    Lifecycle:
    - :meth:`reset` at session start and after every FULL turn — rebuilds
      the system prompt from the fresh journal/scratchpad and clears the
      window (the one deliberate re-prefill, amortized over the fast
      turns that follow).
    - :meth:`turn` once per fast turn.

    Window discipline:
    - Each turn appends one user (delta + frame) / assistant (raw micro
      JSON) exchange. ``_MAX_EXCHANGES`` is a backstop only — with the
      default full-turn cadence the window resets long before hitting it.
      Overflow clears the window rather than dropping the oldest pair,
      because a head-drop invalidates the KV prefix on EVERY subsequent
      turn while a clear pays the re-prefill once.
    """

    # 48 = 6 full planning cycles × 8 fast turns each. With 32k tokens/slot
    # the window holds comfortably — ~175 tokens/exchange × 48 = ~8.4k, well
    # under budget. In normal operation the window resets after each full plan
    # (~8 turns), so 48 is the emergency backstop, not the expected depth.
    _MAX_EXCHANGES = 48

    def __init__(
        self,
        *,
        chat_llm: SlowPathChatLLM,
        caps: SurfaceCapsPayload,
        objective: str,
        game_context: str = "",
    ) -> None:
        self._chat_llm = chat_llm
        self._caps = caps
        self._objective = objective
        self._game_context = game_context
        self._system = build_fast_system_prompt(
            caps=caps, objective=objective, game_context=game_context
        )
        self._window: list[dict] = []
        self._schema = _fast_output_schema(caps)
        # Telemetry from the most recent turn (None until first turn).
        self.last_meta: dict | None = None

    def reset(
        self,
        *,
        journal: dict[str, object] | None = None,
        state: str = "",
        lore: list[str] | None = None,
        lore_summary: list[str] | None = None,
        playbook: dict[str, object] | None = None,
    ) -> None:
        """Rebuild the static prefix and clear the window."""

        self._system = build_fast_system_prompt(
            caps=self._caps,
            objective=self._objective,
            state=state,
            journal=journal,
            lore=lore,
            lore_summary=lore_summary,
            playbook=playbook,
            game_context=self._game_context,
        )
        self._window = []

    def fresh_window(self) -> bool:
        """True when the rolling window holds no exchanges yet.

        A fresh window means the model has NO memory of prior deltas —
        the orchestrator must send the full state snapshot, not a diff,
        or position/HP/map/party silently vanish until they next change.
        Covers both explicit resets (after FULL turns) and the internal
        overflow wipe at _MAX_EXCHANGES.
        """

        return not self._window

    def window_size(self) -> int:
        """Current number of complete exchanges (user+assistant pairs) in the window."""

        return len(self._window) // 2

    async def turn(
        self,
        *,
        t_ms: int,
        overlay_delta: dict[str, object] | None,
        last_actions: list[str],
        frame: bytes | None,
        reflex_actions: list[str] | None = None,
        fx: list[tuple[str, int]] | None = None,
        scene: str = "",
        goals: str = "",
        stalled_s: int = 0,
        loc: str = "",
        rule: str = "",
        mode: str = "",
        screen: str = "",
        nav: str = "",
        blocked: str = "",
    ) -> FastPlan:
        """Run one fast turn. Raises :class:`PlanParseError` on bad output."""

        delta = build_fast_delta(
            t_ms=t_ms,
            overlay_delta=overlay_delta,
            last_actions=last_actions,
            frame_attached=frame is not None,
            reflex_actions=reflex_actions,
            fx=fx,
            scene=scene,
            goals=goals,
            stalled_s=stalled_s,
            loc=loc,
            rule=rule,
            mode=mode,
            screen=screen,
            nav=nav,
            blocked=blocked,
            exchange=self.window_size(),
            max_exchanges=self._MAX_EXCHANGES,
        )
        user_msg = {
            "role": "user",
            "content": delta,
            "images": [frame] if frame is not None else None,
        }
        messages = (
            [{"role": "system", "content": self._system, "images": None}]
            + self._window
            + [user_msg]
        )
        result = await self._chat_llm(messages, {"json_schema": self._schema})
        raw = str(result.get("text") or "")
        # Carry the exact delta the model decided on into telemetry —
        # loop diagnosis needs "what did it SEE" next to "what did it do".
        self.last_meta = {**result, "delta": delta}
        plan = parse_fast_output(raw, self._caps)
        # Append AFTER a successful parse — a garbage reply must not
        # poison the window (the orchestrator escalates to a FULL turn,
        # which resets us anyway, but don't rely on it).
        #
        # The frame is stripped from the archived copy: a 2B model can't
        # integrate a stack of old 140-token frames, and they were the
        # bulk of the window. Costs one exchange of KV re-prefill per
        # turn (the server's cached copy of the previous user message
        # had the image; ~50-80ms) — cheap against the context relief.
        self._window.append({**user_msg, "images": None})
        self._window.append({"role": "assistant", "content": raw, "images": None})
        if len(self._window) > self._MAX_EXCHANGES * 2:
            self._window = []
        return plan


__all__ = [
    "FastTurnRunner",
    "PlanParseError",
    "SlowPathAgent",
    "SlowPathChatLLM",
    "SlowPathLLM",
    "SlowPathLLMProtocol",
]
