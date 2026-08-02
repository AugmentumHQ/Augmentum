"""Session orchestrator.

Glues the surface adapter, NDJSON live log, fast-path rule engine, and
slow-path planner together. This is the only module that imports from
all of the others; everything below is decoupled.

Lifecycle
---------
1. ``Orchestrator(...)`` construct -- pure, no I/O.
2. ``run()`` -- async, runs the full session until completion. Writes
   session/surface_caps entries, starts the adapter, runs the slow-
   path loop until ``stop()`` is signaled or the slow path returns a
   sentinel.
3. ``stop(reason)`` -- request graceful shutdown; emits a
   :class:`SessionEndEntry` and tears the adapter down.

Concurrency model
-----------------
* One asyncio task runs the slow-path planning loop.
* The adapter spawns its own tasks (network, capture, etc.) and pushes
  observations through the ``emit`` callback we hand it.
* The ``emit`` callback writes to the log, feeds the rule engine, and
  -- if any rule fires -- enqueues the matched actions on the action
  queue.
* A third task drains the action queue, applying each action via the
  semantic resolver and writing an :class:`InputEntry`.

This split lets the slow path block on the LLM without stalling
reflex behavior, and lets the rule engine respond to events without
contending with action execution.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any

import structlog
from pydantic import TypeAdapter

from augmentum.config import settings
from augmentum.game_agent.agent import (
    FastTurnRunner,
    SlowPathAgent,
    SlowPathChatLLM,
    SlowPathLLM,
)
from augmentum.game_agent.companion import CompanionPersona
from augmentum.game_agent.journal import CompanionJournal
from augmentum.game_agent.log import LiveLog, SessionClock, now_ms
from augmentum.game_agent.perception import downscale_frame, prepare_frames
from augmentum.game_agent.progress import DEAD_INPUT_THRESHOLD, score_from_world
from augmentum.game_agent.prompt import FastPlan, PlanParseError
from augmentum.game_agent.rules import RuleEngine
from augmentum.game_agent.schema import (
    EventPayload,
    InputPayload,
    PlanAction,
    PlanPayload,
    SessionEndPayload,
    SessionPayload,
    SurfaceKind,
)
from augmentum.game_agent.surfaces.base import SurfaceAdapter
from augmentum.game_agent.surfaces.bridged import BridgedAdapter
from augmentum.game_agent.voice_bridge import VoiceBridge
from augmentum.game_agent.world import WorldState

log = structlog.get_logger(__name__)

_PlanPayloadAdapter = TypeAdapter(PlanPayload)


class Orchestrator:
    """One game-control session, end to end.

    Use when:
    - A user has chosen a surface, declared an objective, and wired an
      LLM. The orchestrator drives everything else.

    Expects:
    - ``adapter`` is constructed and ready (resolver populated, caps
      stable). The orchestrator owns its lifecycle once :meth:`run`
      starts.
    - ``llm`` is hot.

    Returns:
    - From :meth:`run`, when the session ends -- the
      :class:`SessionEndPayload` written to the log.
    """

    # Default slow-path cadence when the agent hasn't emitted a plan
    # yet (first turn) and when a plan didn't specify a next check.
    _DEFAULT_SLOW_PATH_INTERVAL_MS = 2000
    # Maximum log-tail entries fed to the slow-path agent per turn.
    # Token budgeting is the orchestrator's concern; the agent does not
    # trim. Adjust if your model has more context to spare.
    _SLOW_PATH_TAIL_LIMIT = 64
    # How many recent frames to send the slow path as a time sequence.
    # 3 frames at 1 Hz capture = a ~3-second sliding window: enough to
    # see motion + a textbox appearing + an action's effect, without
    # blowing the LLM's vision-token budget. Frame chunk vs single
    # frame: see ecosystem analysis (Claude Plays Pokemon, lmgame-Bench)
    # -- temporal stacking is the established win for game-VLMs.
    _SLOW_PATH_FRAME_CHUNK = 3

    def __init__(
        self,
        *,
        log_path: str,
        surface_kind: SurfaceKind,
        adapter: SurfaceAdapter,
        llm: SlowPathLLM,
        objective: str,
        rule_engine: RuleEngine | None = None,
        session_id: str | None = None,
        companion: bool = False,
        voice_bridge: VoiceBridge | None = None,
        persona: CompanionPersona | None = None,
        journal: CompanionJournal | None = None,
        fast_llm: SlowPathChatLLM | None = None,
        playbook: CompanionJournal | None = None,
    ) -> None:
        self._surface_kind = surface_kind
        self._adapter = adapter
        self._objective = objective
        self._rules = rule_engine or RuleEngine()
        self._session_id = session_id or f"s_{uuid.uuid4().hex[:8]}"
        self._clock = SessionClock()
        self._live_log = LiveLog(log_path, clock=self._clock)
        self._caps = adapter.caps()
        # Quickaction unlock: when this game's translation layer knows
        # the keyboard layout, expose ``type_text`` — one action that
        # types a whole name and presses OK, replacing dozens of blind
        # grid-navigation turns. Injected into caps (so plan validation
        # accepts it) + INPUT_HINTS (so the model knows it exists).
        from augmentum.game_agent.control.text_entry import has_text_entry
        if (
            has_text_entry(self._caps.game_profile)
            and "type_text" not in self._caps.semantic_inputs
        ):
            hints = dict(self._caps.input_hints or {})
            hints["type_text"] = (
                'QUICKACTION for naming/keyboard screens: {"s":"type_text",'
                '"text":"MAY","d":100} types the whole string and presses OK '
                "in one action. Use it the moment a naming screen appears "
                "instead of navigating letters manually."
            )
            self._caps = self._caps.model_copy(
                update={
                    "semantic_inputs": [*self._caps.semantic_inputs, "type_text"],
                    "input_hints": hints,
                }
            )
        # Same unlock for ``navigate_to`` — when the translation layer
        # ships a walk_grid collision probe, ONE action walks a computed
        # path (the model emits spatial intent; the harness executes it
        # deterministically). See control/navigate.py for the premise.
        from augmentum.game_agent.control.navigate import has_navigation
        if (
            has_navigation(self._caps.game_profile)
            and "navigate_to" not in self._caps.semantic_inputs
        ):
            hints = dict(self._caps.input_hints or {})
            hints["navigate_to"] = (
                'QUICKACTION overworld walking: {"s":"navigate_to","text":'
                '"12,8","d":100} walks a collision-safe path to map tile '
                '(x,y); {"text":"down 5"} walks relative. Replaces chains '
                "of single nav presses — prefer it whenever walking "
                "somewhere. Re-issue if it stops short."
            )
            self._caps = self._caps.model_copy(
                update={
                    "semantic_inputs": [*self._caps.semantic_inputs, "navigate_to"],
                    "input_hints": hints,
                }
            )
        # Probe names the prompt never sees (structural blackboard data).
        from augmentum.game_agent.probes import hidden_probe_names
        self._hidden_probes = hidden_probe_names(self._caps.game_profile)
        # Visit count of the tile the player currently stands on (LOC=).
        self._tile_seen: int = 0
        # Input-context inference (game-agnostic): rolling evidence —
        # per-button effect scores, position motion, text activity —
        # classified into reading/cursor/free_move/locked. Single source
        # of truth for the MODE= line and the modal RULE override.
        from augmentum.game_agent.context import InputContextTracker
        self._context = InputContextTracker()
        self._last_tile_key: Any = None
        # Vision-only motion heuristics: frame-diff evidence for the
        # context tracker on games with NO RAM probes. Stays silent the
        # moment real probes flow (they are strictly better evidence).
        from augmentum.game_agent.motion import MotionSense
        self._motion = MotionSense()
        self._ram_probes_seen = False
        # True while the action worker is mid-press. Fast turns and
        # scene ticks wait for quiescence so the model NEVER decides on
        # a frame captured while its previous presses were still landing
        # (the "acts on the old menu image" failure class).
        self._worker_busy: bool = False
        # Screen value at the moment the narrator last looked — when the
        # screen has changed since, the SCENE= line is dropped as stale.
        self._scene_screen: Any = None
        # Meaningful-action gate bookkeeping: consecutive blocks of the
        # current dead press (release after 2 — livelock guard) and the
        # semantic to report as BLOCKED= on the next fast delta.
        self._dead_block_count: int = 0
        self._blocked_semantic: str = ""
        self._companion = companion
        self._voice = voice_bridge
        self._persona = persona
        # Persistent journal -- the agent's long-running memory across
        # sessions. None when no (user_id, title_id) was supplied; the
        # prompt's JOURNAL block is then omitted entirely.
        self._journal = journal
        # Cross-title playbook: per-user memory that TRANSFERS between
        # games (interface physics, genre mechanics, strategies). Same
        # merge/caps machinery as the journal; separate file, separate
        # prompt block, fed by ``plan.playbook_update``.
        self._playbook = playbook
        self._agent = SlowPathAgent(
            llm=llm,
            surface_kind=surface_kind,
            caps=self._caps,
            objective=objective,
            companion=companion,
            persona=persona,
        )
        # Static game context (title, platform, genre, key controls) for
        # known profiles — injected into every fast-turn system prompt so
        # the model has a named, plain-English anchor after every cold
        # window reset, without having to re-derive the game from the
        # SURFACE_CAPS JSON blob.
        from augmentum.game_agent.game_context import game_context_for
        _game_ctx = game_context_for(self._caps.game_profile)
        # Fast-turn ("call mode") runner — micro action decisions on a
        # rolling chat window between FULL planning turns. None when no
        # chat-capable bridge was wired; the loop then runs full turns
        # only, exactly the pre-call-mode behavior.
        self._fast: FastTurnRunner | None = (
            FastTurnRunner(
                chat_llm=fast_llm,
                caps=self._caps,
                objective=objective,
                game_context=_game_ctx,
            )
            if fast_llm is not None
            else None
        )
        # Raw chat bridge kept for the scene-narrator lane (the "eyes"):
        # a dedicated no-thinking vision call whose only job is to
        # translate the frame into a live-feed description consumed by
        # both the actor (SCENE= in the delta) and the planner
        # (scene_feed in OVERLAY).
        self._chat_llm = fast_llm
        self._scene: str = ""
        self._scene_t_ms: int = 0
        # Universal interface-physics reflexes (game-agnostic): register
        # on every session. Currently: dead-nav-during-dialog -> free
        # confirm (movement is swallowed while a text box is open).
        from augmentum.game_agent.rule_packs.universal import (
            dead_nav_during_dialog_rule,
        )
        self._rules.register(dead_nav_during_dialog_rule())
        # Overlay snapshot at the last LLM turn — fast turns send only
        # the DELTA against this, not the full probe dict.
        self._overlay_sent: dict[str, Any] = {}
        # Semantics emitted by the previous LLM turn (fast or full), fed
        # back as "did=..." so the fast model sees action->effect pairs.
        self._last_emitted: list[str] = []
        # Reflex bookkeeping: ids of model-authored rules currently on
        # the engine (capped), and the semantics reflexes fired since
        # the last fast turn (surfaced in the DELTA so the fast model
        # doesn't double-press what a reflex already pressed).
        self._reflex_ids: set[str] = set()
        self._recent_reflex: list[str] = []
        # Per-button effect feedback: (button, effect_score) from the
        # surface's input_ack events since the last fast turn. The score
        # is a core-level frame-diff — the ground truth for "did that
        # press actually do anything" — so the model attributes success
        # per button instead of guessing from expectation.
        self._recent_fx: list[tuple[str, int]] = []
        # Lifetime input-effectiveness tally for the end-of-session
        # scorecard. _recent_fx is a rolling window (trimmed to 8), so it
        # cannot answer "what fraction of everything we pressed actually
        # worked" — these two counters never reset. See progress.py.
        self._inputs_acked = 0
        self._inputs_effective = 0
        # Meaningful-action gate: the most recent press that provably
        # did nothing (effect_score ~0). Repeating it is blocked.
        self._last_dead_press: str | None = None
        # Keyframe ring: the frame each recent fast turn acted on.
        # FULL turns sample this so the planner sees snapshots aligned
        # to its recent MOVES (loop detection), not just the last ~3
        # seconds of near-identical screen.
        self._frame_ring: list[bytes] = []
        # DIALOGUE_LORE ring: unique decoded in-game text lines (from
        # ``text``-type probes), oldest first. The game's own words are
        # the strongest tutorial/world signal available — tutorial text
        # literally teaches the controls — so they accumulate here and
        # render into both prompts. Bounded so a chatty game can't blow
        # the prompt budget.
        self._lore: list[str] = []
        self._lore_seen: set[str] = set()
        # Compacted summaries of overflowed lore batches: oldest lines are
        # not silently dropped but compressed into breadcrumb entries so
        # the model retains the intro/story arc across the whole session.
        self._lore_summary: list[str] = []
        # The blackboard: provenance-ranked facts + measurable goals.
        # Every lane writes into it; prompts render views of it.
        self._world = WorldState()
        self._goal_event = False  # a metric goal completed -> plan NOW
        # Recent entries kept in-memory so the slow path can read a
        # tail without re-reading the file. The rule engine has its
        # own deque; this one feeds the slow-path prompt.
        self._tail: list[dict[str, Any]] = []
        # Latest values from RAM-probe events, keyed by probe name.
        # The slow-path renders these as an OVERLAY block in the prompt
        # so the model sees structured world state alongside the optional
        # frame -- the production-grade pattern Claude Plays Pokemon and
        # pokegym both use (vision is the sanity check, structured state
        # carries the load). Updated in :meth:`_record` on every event
        # that has a ``probes`` dict in its payload data.
        self._overlay: dict[str, Any] = {}
        # Action queue drained by a worker task -- applying inputs is
        # async so we don't want emit() to wait on the wire.
        self._action_queue: asyncio.Queue[tuple[PlanAction, str]] = asyncio.Queue()
        self._stop_event = asyncio.Event()
        self._next_check_in_ms = self._DEFAULT_SLOW_PATH_INTERVAL_MS
        # Set by stop() so run() knows the reason to write.
        self._stop_reason: str = "completed"

    # ── Public API ────────────────────────────────────────────────

    async def run(self) -> SessionEndPayload:
        """Drive the session until completion."""

        self._clock.start()
        self._write_session_header()
        self._write_caps()

        await self._adapter.start(self._emit_event)

        slow_task = asyncio.create_task(self._slow_path_loop(), name="slow-path")
        action_task = asyncio.create_task(self._action_worker(), name="action-worker")
        scene_task = asyncio.create_task(self._scene_loop(), name="scene-narrator")

        try:
            await self._stop_event.wait()
        finally:
            slow_task.cancel()
            action_task.cancel()
            scene_task.cancel()
            for task in (slow_task, action_task, scene_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001
                    log.error("orchestrator.task_exited", task=task.get_name(), error=str(exc))
            await self._adapter.stop()

        duration_ms = self._clock.elapsed_ms()
        # Grade the run. Wrapped because a scoring bug must never cost us
        # the session trailer — a log with no score is recoverable
        # (progress.score_session can re-score it offline), a log with no
        # session_end is not.
        progress: dict[str, Any] | None = None
        try:
            progress = score_from_world(
                self._world,
                inputs_acked=self._inputs_acked,
                inputs_effective=self._inputs_effective,
                duration_ms=duration_ms,
            ).to_dict()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "orchestrator.progress_score_failed",
                session_id=self._session_id,
                error=str(exc),
            )
        end = SessionEndPayload(
            reason=self._stop_reason,  # type: ignore[arg-type]
            duration_ms=duration_ms,
            progress=progress,
        )
        if progress is not None:
            log.info(
                "game_agent.session_scored",
                session_id=self._session_id,
                verdict=progress.get("verdict"),
                score=progress.get("score"),
                reached_play=progress.get("reached_play"),
            )
        self._live_log.append(
            {"t": self._clock.elapsed_ms(), "kind": "session_end", "payload": end.model_dump()}
        )
        self._live_log.close()
        return end

    def stop(self, reason: str = "user_stopped") -> None:
        """Signal graceful shutdown. Safe to call from any task."""

        self._stop_reason = reason
        self._stop_event.set()

    # ── Adapter -> orchestrator observation sink ──────────────────

    async def _emit_event(self, payload: EventPayload) -> None:
        """Called by the adapter for every observation."""

        entry = {
            "t": self._clock.elapsed_ms(),
            "kind": "event",
            "payload": payload.model_dump(),
        }
        self._record(entry)

        # Rule engine reacts to events synchronously; firings push to
        # the action queue.
        matches = self._rules.tick(self._clock.elapsed_ms(), self._caps)
        for match in matches:
            self._live_log.append(
                {
                    "t": self._clock.elapsed_ms(),
                    "kind": "rule_fired",
                    "payload": {
                        "rule_id": match.rule_id,
                        "matched": match.matched,
                        "emitted_actions": [a.model_dump() for a in match.actions],
                    },
                }
            )
            for action in match.actions:
                await self._action_queue.put((action, "rule"))
                self._recent_reflex.append(action.semantic)
            if len(self._recent_reflex) > 12:
                self._recent_reflex = self._recent_reflex[-12:]

    # ── Internal mechanics ────────────────────────────────────────

    def _record(self, entry: dict[str, Any]) -> None:
        """Persist + push to in-memory tails."""

        self._live_log.append(entry)
        self._tail.append(entry)
        if len(self._tail) > self._SLOW_PATH_TAIL_LIMIT * 4:
            # Trim aggressively so memory stays bounded even on long
            # sessions; the slow path only ever reads the last
            # _SLOW_PATH_TAIL_LIMIT anyway.
            self._tail = self._tail[-self._SLOW_PATH_TAIL_LIMIT * 2 :]
        # Extract structured probe values into the overlay. The bridge
        # emits memory events as ``{"event": "ram", "probes": {name:
        # value, ...}}``; we merge each tick so the overlay always
        # reflects the freshest reading without losing fields the
        # current tick didn't touch (the bridge only re-emits CHANGED
        # probe values to keep wire traffic small).
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        if entry.get("kind") == "event" and isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                probes = data.get("probes")
                if isinstance(probes, dict):
                    t_probe = entry.get("t") or 0
                    # Provenance: RAM events (the bridge's decoder) are
                    # rank 3; scene-derived events (the narrator's
                    # labels, marked probe_source="scene") are rank 2.
                    # The blackboard gates the write; everything below
                    # (overlay, lore, novelty, text evidence) mirrors
                    # only ACCEPTED values so a pixel guess can never
                    # sneak past RAM truth via a side door.
                    if data.get("probe_source") == "scene":
                        accepted = self._world.update_many(
                            probes, source="scene", t_ms=t_probe
                        )
                    else:
                        self._ram_probes_seen = True
                        self._world.update_probes(probes, t_ms=t_probe)
                        accepted = dict(probes)
                    # Hidden probes (walk_grid, …) feed the blackboard +
                    # quickaction compilers but never the prompt overlay.
                    visible = {
                        k: v for k, v in accepted.items()
                        if k not in self._hidden_probes
                    }
                    self._overlay.update(visible)
                    self._collect_lore(visible, t_ms=t_probe)
                    self._note_novelty(accepted, t_ms=t_probe)
                    # Modal-text evidence: a changing *_text probe means
                    # a box is printing RIGHT NOW. The buffers never
                    # clear on close (they hold the last line forever),
                    # so "recently changed" is the honest open-box
                    # signal; each new box re-arms it, a working nav
                    # press ends it (see input_ack below).
                    for name, value in accepted.items():
                        if (
                            name.endswith("_text")
                            and isinstance(value, str)
                            and len(value.strip()) >= 2
                        ):
                            self._context.feed_text_activity(t_probe)
                            break
                    # A scene-accepted DIALOG screen is open-box
                    # evidence even when no quoted text was extracted.
                    if (
                        data.get("probe_source") == "scene"
                        and accepted.get("screen") == "dialog"
                    ):
                        self._context.feed_text_activity(t_probe)
                    for horizon in self._world.check_goals():
                        self._world.mark_progress(t_probe)
                        self._goal_event = True
                        self._live_log.append(
                            {
                                "t": self._clock.elapsed_ms(),
                                "kind": "event",
                                "payload": {
                                    "channel": "log",
                                    "data": {
                                        "event": "goal_completed",
                                        "horizon": horizon,
                                        "goal": self._world.goals[horizon].text,
                                    },
                                },
                            }
                        )
                if data.get("event") == "input_ack" and data.get("button"):
                    score = int(data.get("effect_score") or 0)
                    self._inputs_acked += 1
                    if score >= DEAD_INPUT_THRESHOLD:
                        self._inputs_effective += 1
                    self._recent_fx.append((str(data["button"]), score))
                    self._last_dead_press = (
                        str(data["button"]) if score < DEAD_INPUT_THRESHOLD else None
                    )
                    self._dead_block_count = 0  # fresh evidence resets the guard
                    if len(self._recent_fx) > 8:
                        self._recent_fx = self._recent_fx[-8:]
                    self._context.feed_fx(
                        str(data["button"]), score, entry.get("t") or 0
                    )
                    # A movement press that visibly worked means nothing
                    # modal is swallowing input — close the text window.
                    if score >= DEAD_INPUT_THRESHOLD and str(data["button"]).startswith("nav_"):
                        self._context.end_text_activity()
        self._rules.observe(entry)

    _LORE_MAX_LINES = 40
    _LORE_MAX_CHARS = 200
    # When _lore overflows, compact this many oldest lines into one summary.
    _LORE_COMPACT_BATCH = 8
    # Max summary entries to keep (oldest dropped when exceeded).
    _LORE_SUMMARY_MAX_ENTRIES = 6

    @staticmethod
    def _compact_lore_batch(lines: list[str]) -> str:
        """Compress a batch of overflowed lore lines into one summary entry."""
        n = len(lines)
        if n <= 3:
            return " | ".join(lines)
        preview = [lines[0], lines[1], "...", lines[-1]]
        return f"[{n} lines: {' | '.join(preview)}]"

    def _collect_lore(self, probes: dict[str, Any], *, t_ms: int = 0) -> None:
        """Accumulate string-valued probe changes as dialogue lore.

        Any ``text``-decoded probe (dialog_text, battle_text, …) whose
        value is fresh prose gets one line in the ring. Dedup is exact;
        partially-printed typewriter frames still differ from the final
        line, so we also drop a line that is a strict prefix extension's
        predecessor (the shorter one) when the longer arrives.
        """

        for name, value in probes.items():
            if not isinstance(value, str):
                continue
            text = value.strip()[: self._LORE_MAX_CHARS]
            # Too short to be prose (cursor junk, single glyphs).
            if len(text) < 8 or text in self._lore_seen:
                continue
            # Typewriter growth: when the new line extends the previous
            # one, replace it instead of keeping every print state.
            if self._lore and text.startswith(self._lore[-1]):
                self._lore_seen.discard(self._lore[-1])
                self._lore[-1] = text
            else:
                self._lore.append(text)
            self._lore_seen.add(text)
            if len(self._lore) > self._LORE_MAX_LINES:
                # Compact the oldest batch into a summary entry instead of
                # silently dropping lines.  Intro dialog (GAME FREAK logo,
                # Birch's monologue) gets preserved as a breadcrumb rather
                # than vanishing the moment the ring fills.
                batch = self._lore[: self._LORE_COMPACT_BATCH]
                self._lore = self._lore[self._LORE_COMPACT_BATCH :]
                for _dl in batch:
                    self._lore_seen.discard(_dl)
                self._lore_summary.append(self._compact_lore_batch(batch))
                if len(self._lore_summary) > self._LORE_SUMMARY_MAX_ENTRIES:
                    self._lore_summary.pop(0)
            # A genuinely new dialogue line is progress — the story is
            # advancing. Feeds the novelty-based stall watchdog.
            self._world.note("dialog", text, t_ms=t_ms)
            log.debug("game_agent.lore_line", probe=name, text=text[:60])

    def _note_visual(self, frame: bytes | None) -> None:
        """Record one frame's coarse visual bucket as a novelty visit.

        The probe-free progress dimension. Called from every path that
        already holds a frame — the planner's capture (fast, tracks the
        agent's actual movement) and the motion tick (slow, keeps the
        signal alive while the planner is idle). Sampling from only the
        1.5s motion tick missed whole rooms on a game the agent crossed
        in under a second. ``note`` collapses consecutive repeats, so
        oversampling a static screen costs nothing.
        """

        if frame is None:
            return
        from augmentum.game_agent.perception import (
            _fingerprint_bytes,
            visual_bucket,
        )

        bucket = visual_bucket(_fingerprint_bytes(frame))
        if bucket is not None:
            self._world.note("visual", bucket, t_ms=self._clock.elapsed_ms())

    def _note_novelty(self, probes: dict[str, Any], *, t_ms: int) -> None:
        """Feed the world's novelty tracker from this tick's probes.

        Dimensions are generic — any preset with position probes gets
        tile tracking; any preset with a screen probe gets screen
        tracking. Novelty (first-ever key) bumps the progress pulse the
        stall watchdog measures against.
        """

        f = self._world.facts
        if (
            ("player_x" in probes or "player_y" in probes
             or "map_group" in probes or "map_num" in probes)
            and "player_x" in f and "player_y" in f
        ):
            key = (
                f["map_group"].value if "map_group" in f else None,
                f["map_num"].value if "map_num" in f else None,
                f["player_x"].value,
                f["player_y"].value,
            )
            self._tile_seen = self._world.note("tile", key, t_ms=t_ms)
            # Position MOTION (any tile change, novel or not) is the
            # free-move discriminator for the context tracker.
            if key != self._last_tile_key:
                if self._last_tile_key is not None:
                    self._context.feed_position_change(t_ms)
                self._last_tile_key = key
        if "screen" in probes:
            self._world.note("screen", probes["screen"], t_ms=t_ms)
            self._context.feed_screen_change(t_ms)

    def _write_session_header(self) -> None:
        header = SessionPayload(
            session_id=self._session_id,
            surface=self._surface_kind,
            objective=self._objective,
            started_at_unix_ms=now_ms(),
        )
        self._live_log.append(
            {"t": 0, "kind": "session", "payload": header.model_dump()}
        )

    def _write_caps(self) -> None:
        self._live_log.append(
            {
                "t": self._clock.elapsed_ms(),
                "kind": "surface_caps",
                "payload": self._caps.model_dump(),
            }
        )

    async def _action_worker(self) -> None:
        """Drain the action queue, applying each via the resolver.

        Quickaction macros expand here: ``type_text`` compiles into its
        primitive press sequence (via the game profile's keyboard
        layout) and each primitive runs through the normal apply path —
        acked, effect-scored, and logged individually, so fx feedback
        works for macro steps exactly like hand-picked presses.
        """

        resolver = self._adapter.resolver
        while True:
            self._worker_busy = False
            action, source = await self._action_queue.get()
            self._worker_busy = True
            if action.semantic == "type_text":
                from augmentum.game_agent.control.text_entry import (
                    compile_text_entry,
                )
                seq = compile_text_entry(
                    self._caps.game_profile, action.text or ""
                )
                if not seq:
                    self._log_agent_error(
                        "type_text: no keyboard layout for this game profile"
                    )
                    continue
                self._live_log.append(
                    {
                        "t": self._clock.elapsed_ms(),
                        "kind": "event",
                        "payload": {
                            "channel": "log",
                            "data": {
                                "event": "quickaction",
                                "name": "type_text",
                                "text": action.text or "",
                                "presses": len(seq),
                            },
                        },
                    }
                )
                for step in seq:
                    await self._action_queue.put((step, source))
                continue
            if action.semantic == "navigate_to":
                await self._expand_navigate(action, source)
                continue
            # Chord members: validated against surface caps here (the
            # same gate every action path passes), deduped, primary
            # excluded. Silently-invalid extras become an agent_error
            # so the model learns its chord was trimmed.
            extras: list[str] = []
            if action.also:
                allowed = set(self._caps.semantic_inputs)
                extras = [
                    s for s in dict.fromkeys(action.also)
                    if s in allowed and s != action.semantic
                ]
                bad = [s for s in action.also if s not in allowed]
                if bad:
                    self._log_agent_error(
                        f"chord member(s) {bad!r} not in surface inputs — dropped"
                    )
            payload = InputPayload(
                semantic=action.semantic,
                duration_ms=action.duration_ms,
                source=source,  # type: ignore[arg-type]
                also=extras or None,
            )
            self._live_log.append(
                {
                    "t": self._clock.elapsed_ms(),
                    "kind": "input",
                    "payload": payload.model_dump(),
                }
            )
            try:
                if extras and getattr(resolver, "supports_chord", False):
                    await resolver.apply_chord(
                        [action.semantic, *extras], action.duration_ms
                    )
                elif extras:
                    # Surface can't press simultaneously — degrade to
                    # sequential presses and tell the model the truth
                    # (a "run+jump" that lands as "run, then jump" must
                    # not be mistaken for a real chord).
                    self._live_log.append(
                        {
                            "t": self._clock.elapsed_ms(),
                            "kind": "event",
                            "payload": {
                                "channel": "log",
                                "data": {
                                    "event": "chord_fallback_sequential",
                                    "buttons": [action.semantic, *extras],
                                },
                            },
                        }
                    )
                    await resolver.apply(action.semantic, action.duration_ms)
                    for s in extras:
                        await resolver.apply(s, action.duration_ms)
                else:
                    await resolver.apply(action.semantic, action.duration_ms)
            except Exception as exc:  # noqa: BLE001
                # Adapter-level failure: log and keep going. A repeated
                # failure on the same semantic is the user's bug.
                self._live_log.append(
                    {
                        "t": self._clock.elapsed_ms(),
                        "kind": "agent_error",
                        "payload": {
                            "where": "adapter",
                            "message": f"resolver.apply({action.semantic!r}) raised: {exc}",
                            "recoverable": True,
                        },
                    }
                )

    async def _expand_navigate(self, action: PlanAction, source: str) -> None:
        """Compile one ``navigate_to`` into its collision-safe presses.

        Reads the walk_grid + position facts off the blackboard, BFS-es
        a path, and enqueues per-tile presses through the normal apply
        path (each acked + effect-scored, so fx feedback works for the
        walk exactly like hand-picked presses). Degrades honestly: no
        grid / bad target / no path each log a distinct reason the
        model can read in LIVE_LOG_TAIL.
        """

        from augmentum.game_agent.control.navigate import (
            compile_navigation,
            resolve_nav_target,
        )

        facts = self._world.facts
        grid_f = facts.get("walk_grid")
        px_f, py_f = facts.get("player_x"), facts.get("player_y")
        if grid_f is None or not isinstance(grid_f.value, dict) \
                or px_f is None or py_f is None:
            self._log_agent_error(
                "navigate_to: no walk grid / position available yet — "
                "use single nav presses"
            )
            return
        px, py = int(px_f.value), int(py_f.value)
        target = resolve_nav_target(action.text or "", grid_f.value, px, py)
        if target is None:
            self._log_agent_error(
                f'navigate_to: bad target {action.text!r} — use a NAV '
                'name (exit_north), "x,y", or "down 5"'
            )
            return
        seq, end = compile_navigation(grid_f.value, px, py, *target)
        self._live_log.append(
            {
                "t": self._clock.elapsed_ms(),
                "kind": "event",
                "payload": {
                    "channel": "log",
                    "data": {
                        "event": "quickaction",
                        "name": "navigate_to",
                        "target": list(target),
                        "from": [px, py],
                        "path_end": list(end),
                        "presses": len(seq),
                        "reaches_target": list(end) == list(target),
                    },
                },
            }
        )
        if not seq:
            self._log_agent_error(
                f"navigate_to: no walkable path from ({px},{py}) toward "
                f"{target} in the visible window — try a different target"
            )
            return
        for step in seq:
            await self._action_queue.put((step, source))

    def _log_agent_error(self, message: str) -> None:
        self._live_log.append(
            {
                "t": self._clock.elapsed_ms(),
                "kind": "agent_error",
                "payload": {
                    "where": "slow_path",
                    "message": message,
                    "recoverable": True,
                },
            }
        )

    def _log_llm_timing(self, turn: str, meta: dict[str, Any]) -> None:
        """Append per-turn LLM telemetry straight to the file log.

        Deliberately NOT via :meth:`_record` — timing lines are for the
        operator and the analysis scripts, not for the model's
        LIVE_LOG_TAIL (they'd crowd out real observations).
        """

        self._live_log.append(
            {
                "t": self._clock.elapsed_ms(),
                "kind": "event",
                "payload": {
                    "channel": "vlm",
                    "data": {"event": "llm_timing", "turn": turn, **meta},
                },
            }
        )

    def _overlay_delta(self) -> dict[str, Any] | None:
        """Probe values that changed since the last LLM turn."""

        delta = {
            k: v for k, v in self._overlay.items() if self._overlay_sent.get(k) != v
        }
        self._overlay_sent = dict(self._overlay)
        return delta or None

    def _invalidate_if_stale(self, plan: FastPlan, *, screen_before: str) -> FastPlan:
        """Drop a fast plan whose world moved on while the model thought.

        The frame + delta are captured BEFORE the LLM call; RAM keeps
        streaming during it. If the SCREEN has changed by the time the
        reply arrives (menu closed itself, battle started, a reflex
        advanced past the box), the decision was made against a screen
        that no longer exists — executing it is exactly the "presses
        menu buttons at the empty overworld" loop. Actions are dropped,
        the event is logged for the model's own log tail, and next_ms
        collapses so it looks again immediately with fresh eyes.
        """

        if not plan.actions or not screen_before:
            return plan
        sf = self._world.facts.get("screen")
        screen_after = str(sf.value) if sf is not None else ""
        if screen_after == screen_before:
            return plan
        self._live_log.append(
            {
                "t": self._clock.elapsed_ms(),
                "kind": "event",
                "payload": {
                    "channel": "log",
                    "data": {
                        "event": "stale_decision_dropped",
                        "screen_at_capture": screen_before,
                        "screen_now": screen_after,
                        "dropped": [a.semantic for a in plan.actions],
                    },
                },
            }
        )
        return FastPlan(
            actions=[],
            why=(
                f"[dropped: screen changed {screen_before}->{screen_after} "
                f"mid-think] {plan.why}"
            ),
            next_check_in_ms=250,
            escalate=plan.escalate,
        )

    async def _wait_actions_settled(
        self, *, max_wait_s: float = 6.0, settle_ms: int = 250,
    ) -> None:
        """Block until queued/in-flight inputs finish, then a short settle.

        The stale-context killer: without this, a turn can capture its
        frame while the PREVIOUS turn's presses (or a macro) are still
        landing, so the model reasons about a screen that no longer
        exists (e.g. keeps pressing menu buttons after cancel already
        returned it to the overworld). The settle pause lets the game
        render the last press's effect before we look. Bounded so a
        wedged resolver can never starve the loop.
        """

        waited = False
        deadline = asyncio.get_running_loop().time() + max_wait_s
        while (
            (self._action_queue.qsize() > 0 or self._worker_busy)
            and asyncio.get_running_loop().time() < deadline
        ):
            waited = True
            await asyncio.sleep(0.05)
        if waited:
            await asyncio.sleep(settle_ms / 1000.0)

    async def _run_fast_turn(self) -> FastPlan:
        """One fast ("call mode") turn: newest frame + delta -> micro-plan."""

        assert self._fast is not None
        await self._wait_actions_settled()
        # Fresh window = the model has no memory of prior deltas (post-
        # FULL-turn reset or overflow wipe). Send the FULL state
        # snapshot, not a diff — otherwise position/HP/map/party vanish
        # until they next change, and anything that changed while the
        # planner was thinking is swallowed forever.
        if self._fast.fresh_window():
            self._overlay_sent = {}
        frame: bytes | None = None
        if "frame" in self._caps.observation_modalities:
            snap_frames = getattr(self._adapter, "snapshot_frames", None)
            if callable(snap_frames):
                got = await snap_frames(n=1)
                frame = got[-1] if got else None
            else:
                frame = await self._adapter.snapshot_frame()
        if frame is not None:
            frame = downscale_frame(
                frame,
                max_edge=int(getattr(settings, "game_agent_frame_max_edge", 480)),
            )
        if frame is not None:
            self._frame_ring.append(frame)
            if len(self._frame_ring) > 8:
                self._frame_ring = self._frame_ring[-8:]
        reflex_recent = self._recent_reflex
        self._recent_reflex = []
        fx_recent = self._recent_fx
        self._recent_fx = []
        stall_after_ms = (
            int(getattr(settings, "game_agent_stall_after_s", 45)) * 1000
        )
        # Stall = time since anything NOVEL (new tile/screen/dialogue/
        # goal), not since any fact churned — menu oscillation can no
        # longer reset the watchdog, because a menu is only novel once.
        stalled_ms = self._world.novelty_stalled_for_ms(self._clock.elapsed_ms())
        # MODE= — the agnostic input context inferred from ground truth
        # (what presses actually did, whether position moved, whether
        # text is printing). Works on ANY game, translation layer or not.
        # RULE= — the per-game refinement for the named screen; while
        # text is being presented the modal rule OVERRIDES it (the
        # screen probe keeps saying "overworld" mid-dialog, and telling
        # the model "you are free to move" there is actively harmful).
        from augmentum.game_agent.rule_packs.screen_rules import (
            modal_rule,
            screen_rule,
        )
        now_ms = self._clock.elapsed_ms()
        mode = self._context.mode_line(now_ms)
        screen_fact = self._world.facts.get("screen")
        # Always-on, RAM-fresh screen name — the model must never have
        # to remember which screen it is on from stale turns.
        screen_now = (
            str(screen_fact.value) if screen_fact is not None else ""
        )
        if self._context.infer(now_ms) == "reading":
            rule = modal_rule(self._caps.game_profile)
        else:
            rule = screen_rule(
                self._caps.game_profile,
                screen_fact.value if screen_fact is not None else None,
            )
        # SCENE= staleness: drop it when the screen has CHANGED since the
        # narrator looked (it describes a screen that no longer exists)
        # or when it is simply too old to trust (run 20: a 300s-old
        # title-screen description rode along on the main menu); age-
        # label anything in between so the model can discount.
        scene = self._scene
        if scene:
            cur_screen = screen_fact.value if screen_fact is not None else None
            age_s = (now_ms - self._scene_t_ms) / 1000.0
            if cur_screen != self._scene_screen or age_s > 45:
                scene = ""
            elif age_s > 2.5:
                scene = f"[{age_s:.0f}s old] {scene}"
        # BLOCKED= — the gate ate this press last turn; without the
        # feedback the model re-emits it forever (the run-20 livelock).
        blocked = self._blocked_semantic
        self._blocked_semantic = ""
        # LOC= — novelty of the tile under the player's feet.
        loc = ""
        if self._tile_seen == 1:
            loc = "new"
        elif self._tile_seen > 1:
            loc = f"seenx{self._tile_seen}"
        # NAV= — named walk targets extracted geometrically from the
        # collision window. Symbols instead of coordinates: a small
        # model can pick "exit_north" reliably; it cannot derive (12,8).
        nav = ""
        grid_f = self._world.facts.get("walk_grid")
        px_f = self._world.facts.get("player_x")
        py_f = self._world.facts.get("player_y")
        if (
            grid_f is not None and isinstance(grid_f.value, dict)
            and px_f is not None and py_f is not None
        ):
            from augmentum.game_agent.control.navigate import extract_landmarks
            names = extract_landmarks(
                grid_f.value, int(px_f.value), int(py_f.value)
            )
            if names:
                nav = ",".join(sorted(names))
        plan = await self._fast.turn(
            t_ms=self._clock.elapsed_ms(),
            overlay_delta=self._overlay_delta(),
            last_actions=list(self._last_emitted),
            frame=frame,
            reflex_actions=reflex_recent or None,
            fx=fx_recent or None,
            scene=scene,
            goals=self._world.goals_line(),
            stalled_s=(stalled_ms // 1000) if stalled_ms >= stall_after_ms else 0,
            loc=loc,
            rule=rule,
            mode=mode,
            screen=screen_now,
            nav=nav,
            blocked=blocked,
        )
        plan = self._invalidate_if_stale(plan, screen_before=screen_now)
        if self._fast.last_meta:
            self._log_llm_timing("fast", self._fast.last_meta)
        # Mirror the fast decision into the log as a plan entry so every
        # consumer (UI panel, analysis scripts, training capture) sees
        # one uniform stream. The [fast] marker + empty scratchpad
        # distinguish it from FULL plans.
        self._live_log.append(
            {
                "t": self._clock.elapsed_ms(),
                "kind": "plan",
                "payload": PlanPayload(
                    observations=[f"[fast] {plan.why}"] if plan.why else ["[fast]"],
                    state_update="",
                    actions=plan.actions,
                    confidence=0.6,
                    next_check_in_ms=plan.next_check_in_ms,
                ).model_dump(),
            }
        )
        return plan

    # Planning turns that outlive this watchdog get cancelled — the fast
    # lane must never be permanently orphaned by a hung backend. Generous
    # because big cloud planners with thinking legitimately take minutes.
    _PLANNER_TIMEOUT_S = 180.0

    async def _integrate_plan(self, plan: PlanPayload, latency_ms: float) -> int:
        """Fold a completed FULL plan into the session.

        Runs between fast turns (single event loop — no mid-turn window
        mutation): logs timing, re-grounds the fast window with the
        fresh journal/scratchpad/lore, enqueues the plan's actions, and
        fires companion voice. Returns the refilled fast budget.
        """

        self._log_llm_timing("full", {"latency_ms": round(latency_ms, 1)})
        self._apply_reflex_rules(plan)
        # NOTE: deliberately NOT rebaselining the overlay here — the
        # reset below empties the fast window, and the next fast turn
        # detects that (fresh_window) and sends the FULL state snapshot.
        if self._fast is not None:
            self._fast.reset(
                journal=(
                    self._journal.to_prompt_dict()
                    if self._journal is not None
                    else None
                ),
                state=self._agent.state,
                lore=list(self._lore) if self._lore else None,
                lore_summary=list(self._lore_summary) if self._lore_summary else None,
            )
        self._last_emitted = [a.semantic for a in plan.actions]
        for action in plan.actions:
            await self._action_queue.put((action, "agent"))
        # Companion voice: non-blocking, same as ever.
        if (
            self._companion
            and self._voice is not None
            and plan.say
            and plan.intent != "silent"
            and isinstance(self._adapter, BridgedAdapter)
        ):
            asyncio.create_task(
                self._speak(plan.say),
                name=f"speak-{self._session_id}",
            )
        self._next_check_in_ms = plan.next_check_in_ms
        full_every = max(1, int(getattr(settings, "game_agent_full_turn_every", 8)))
        return full_every - 1

    def _apply_reflex_rules(self, plan: PlanPayload) -> None:
        """Register / retract the plan's model-authored reflex rules.

        Invalid specs and cap overflows become recoverable agent_error
        entries — those land in the planner's LIVE_LOG_TAIL, which is
        how the model learns its rule was rejected.
        """

        if not plan.reflex_rules:
            return
        from augmentum.game_agent.reflex import (
            MAX_ACTIVE_REFLEX_RULES,
            compile_reflex_rule,
        )

        for spec in plan.reflex_rules[:4]:
            if not isinstance(spec, dict):
                continue
            rid = str(spec.get("id") or "").strip()
            if spec.get("retract"):
                if rid and self._rules.unregister(rid):
                    self._reflex_ids.discard(rid)
                continue
            if (
                rid not in self._reflex_ids
                and len(self._reflex_ids) >= MAX_ACTIVE_REFLEX_RULES
            ):
                self._log_agent_error(
                    f"reflex rule {rid!r} rejected: {MAX_ACTIVE_REFLEX_RULES} "
                    "rules already active — retract one first"
                )
                continue
            try:
                rule = compile_reflex_rule(spec)
            except ValueError as exc:
                self._log_agent_error(f"reflex rule rejected: {exc}")
                continue
            self._rules.register(rule)
            self._reflex_ids.add(rule.rule_id)
            log.info(
                "game_agent.reflex_registered",
                session_id=self._session_id,
                rule_id=rule.rule_id,
            )

    # Scene narrator label set: the first word before a colon is the
    # machine-readable screen label (TITLE, OVERWORLD, BATTLE, MENU,
    # DIALOG, CUTSCENE, LOADING, UNKNOWN). Everything after "WORD: " is
    # the plain-text description for SCENE=.
    _SCENE_LABELS = {
        "TITLE", "OVERWORLD", "BATTLE", "MENU", "DIALOG",
        "CUTSCENE", "LOADING", "UNKNOWN",
    }

    async def _scene_tick(self, last_fp: bytes | None) -> bytes | None:
        """One narrator pass. Returns the new fingerprint (or the old one
        when the screen hasn't changed / no frame was available)."""

        from augmentum.game_agent.perception import (
            _fingerprint_bytes,
            _max_channel_diff,
        )
        from augmentum.game_agent.prompt import SCENE_NARRATOR_PROMPT

        # Don't narrate a screen that's mid-press-sequence — the
        # description would be obsolete before anyone reads it.
        if self._action_queue.qsize() > 0 or self._worker_busy:
            return last_fp
        # Screen fact is read BEFORE the frame snapshot: reading it after
        # the (multi-second) LLM call stamped transition-frame scenes
        # with the NEXT screen's name, defeating the staleness drop
        # (run 20: a title-screen description tagged "main_menu").
        sf = self._world.facts.get("screen")
        screen_at_capture = sf.value if sf is not None else None
        frame: bytes | None = None
        snap_frames = getattr(self._adapter, "snapshot_frames", None)
        if callable(snap_frames):
            got = await snap_frames(n=1)
            frame = got[-1] if got else None
        else:
            frame = await self._adapter.snapshot_frame()
        if frame is None:
            return last_fp
        frame = downscale_frame(
            frame, max_edge=int(getattr(settings, "game_agent_frame_max_edge", 480))
        )
        fp = _fingerprint_bytes(frame)
        # NOTE: visual novelty is recorded in _motion_tick, not here —
        # this lane only runs when a chat LLM is wired, and the games
        # that most need a probe-free progress dimension are exactly the
        # ones running without one.
        # Fingerprint gate: don't burn a vision call re-describing an
        # unchanged screen. Same threshold the dedup pass uses.
        if fp is not None and last_fp is not None and _max_channel_diff(fp, last_fp) <= 24:
            return last_fp
        context_bits = []
        if self._scene:
            context_bits.append(f"PREVIOUS: {self._scene}")
        dialog = self._overlay.get("dialog_text")
        if isinstance(dialog, str) and dialog.strip():
            context_bits.append(f'RAM dialog_text: "{dialog[:160]}"')
        context_bits.append("Describe the CURRENT screen.")
        assert self._chat_llm is not None
        result = await self._chat_llm(
            [
                {"role": "system", "content": SCENE_NARRATOR_PROMPT, "images": None},
                {"role": "user", "content": "\n".join(context_bits), "images": [frame]},
            ]
        )
        text = str(result.get("text") or "").strip()[:420]
        if text:
            # Parse the machine-readable screen label prefix: "LABEL: description".
            # Fall back to the full text when the model didn't follow format.
            screen_label = ""
            desc = text
            if ":" in text:
                maybe_label, _, rest = text.partition(":")
                maybe_label = maybe_label.strip().upper()
                if maybe_label in self._SCENE_LABELS:
                    screen_label = maybe_label
                    desc = rest.strip()
            self._scene = desc or text
            self._scene_t_ms = self._clock.elapsed_ms()
            # Narrator output becomes PROBES with probe_source="scene":
            # the same spine RAM probes ride (blackboard rank gate →
            # overlay → lore → novelty → context evidence → reflex
            # window → goal metrics), so every downstream consumer
            # works on a game with ZERO RAM probes. When RAM exists
            # (rank 3) the blackboard rejects these (rank 2) and the
            # mirrors never see them — pixels cannot override memory.
            scene_probes: dict[str, Any] = {}
            if screen_label:
                scene_probes["screen"] = screen_label.lower()
                if screen_label == "DIALOG":
                    # Extract quoted dialog text from the description
                    # for DIALOGUE_LORE accumulation and DELTA= even
                    # without a RAM text probe.
                    _quotes = re.findall(r'"([^"]{8,})"', desc)
                    if not _quotes:
                        _quotes = re.findall(r"'([^']{8,})'", desc)
                    if _quotes and _quotes[0].strip():
                        scene_probes["dialog_text"] = _quotes[0].strip()[:200]
            # Remember which screen this description belongs to — the
            # fast lane drops SCENE= once the screen value moves on.
            # Uses the narrator's label when no RAM probe is available.
            self._scene_screen = screen_label.lower() if screen_label else screen_at_capture
            self._world.update(
                "scene_feed", text, source="scene", t_ms=self._scene_t_ms
            )
            self._log_llm_timing(
                "scene",
                {k: result.get(k) for k in ("latency_ms", "tok_s", "cached_tokens")},
            )
            self._live_log.append(
                {
                    "t": self._clock.elapsed_ms(),
                    "kind": "event",
                    "payload": {
                        "channel": "vlm",
                        "data": {"event": "scene", "text": text},
                    },
                }
            )
            if scene_probes:
                await self._emit_event(
                    EventPayload(
                        channel="vlm",
                        data={
                            "event": "scene_probes",
                            "probes": scene_probes,
                            "probe_source": "scene",
                        },
                    )
                )
        return fp

    async def _motion_tick(self) -> None:
        """Frame-diff input-context evidence for probe-less games.

        Compares the latest frame against the previous motion sample and
        feeds the classified pattern (world motion / text printing /
        screen transition) into the :class:`InputContextTracker` — the
        same evidence RAM probes would have provided. Gated OFF the
        moment real RAM probes flow: decoded truth is strictly better
        than pixel heuristics, and double-feeding would let a wrong
        pixel guess extend a window RAM already closed.
        """

        if self._ram_probes_seen:
            return
        frame: bytes | None = None
        snap_frames = getattr(self._adapter, "snapshot_frames", None)
        if callable(snap_frames):
            got = await snap_frames(n=1)
            frame = got[-1] if got else None
        else:
            frame = await self._adapter.snapshot_frame()
        if frame is None:
            return
        t_ms = self._clock.elapsed_ms()
        # Visual novelty is recorded here rather than in the scene tick,
        # because the scene tick only runs when a chat LLM is wired. A
        # probe-less game on a box with no VLM would otherwise record no
        # progress dimension at all and score zero forever — the exact
        # case the probe-free scorer exists for. See progress.py.
        self._note_visual(frame)
        reading = self._motion.feed(frame)
        if reading is None:
            return
        if reading.screen_changed:
            self._context.feed_screen_change(t_ms)
        elif reading.text_printing:
            self._context.feed_text_activity(t_ms)
        elif reading.world_motion:
            self._context.feed_position_change(t_ms)

    async def _scene_loop(self) -> None:
        """The visual-feed lane: keep ``self._scene`` fresh in parallel.

        Fingerprint-gated so a static screen costs zero LLM calls; every
        failure is swallowed (a missing scene degrades the actor to
        frame-only, never crashes the session)."""

        last_fp: bytes | None = None
        while not self._stop_event.is_set():
            interval = max(
                0.5,
                int(getattr(settings, "game_agent_scene_interval_ms", 1500)) / 1000.0,
            )
            await asyncio.sleep(interval)
            if "frame" not in self._caps.observation_modalities:
                return
            # Motion tick runs EVERY beat — even with the narrator off
            # or no chat LLM at all, probe-less games still get frame-
            # diff input-context evidence (it costs ~1ms, no LLM call).
            try:
                await self._motion_tick()
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "game_agent.motion_tick_failed",
                    session_id=self._session_id,
                    error=str(exc),
                )
            if self._chat_llm is None:
                continue
            if not bool(getattr(settings, "game_agent_scene_narrator_enabled", True)):
                continue
            try:
                last_fp = await self._scene_tick(last_fp)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "game_agent.scene_tick_failed",
                    session_id=self._session_id,
                    error=str(exc),
                )

    @staticmethod
    def _keyframe_sample(ring: list[bytes]) -> list[bytes]:
        """Pick up to 2 spaced keyframes (oldest, middle) from the ring."""

        if len(ring) >= 3:
            return [ring[0], ring[len(ring) // 2]]
        if len(ring) >= 1:
            return [ring[0]]
        return []

    async def _timed_plan_turn(self) -> tuple[PlanPayload, float]:
        """One FULL turn + its wall latency (for the llm_timing entry)."""

        import time as _time

        t0 = _time.monotonic()
        plan = await self._run_slow_path_turn()
        return plan, (_time.monotonic() - t0) * 1000.0

    async def _slow_path_loop(self) -> None:
        """Fast call-mode turns with FULL planning running in parallel.

        The FULL (thinking) turn is a background task: the fast lane
        keeps acting on its current window while the planner thinks, and
        the finished plan integrates at the next turn boundary. Cadence:
        a planning task starts whenever the fast budget is exhausted —
        every ``game_agent_full_turn_every`` turns, or immediately when a
        fast turn escalates (``esc``), fails to parse, or errors. Only
        one planning task is ever in flight. Without a fast lane the
        loop degrades to the serial plan→sleep→plan behavior.
        """

        # First-tick delay gives the adapter a moment to publish
        # initial observations so the agent doesn't plan against an
        # empty log.
        await asyncio.sleep(0.1)
        import time as _time

        fast_budget = 0          # fast turns remaining before the next FULL
        plan_task: asyncio.Task[tuple[PlanPayload, float]] | None = None
        plan_started = 0.0
        try:
            while not self._stop_event.is_set():
                full_every = max(
                    1, int(getattr(settings, "game_agent_full_turn_every", 8))
                )
                fast_ok = (
                    self._fast is not None
                    and bool(getattr(settings, "game_agent_fast_turns_enabled", True))
                    and full_every > 1
                )

                # 1. Integrate a finished planning task (or reap its error).
                if plan_task is not None and plan_task.done():
                    try:
                        plan, latency_ms = plan_task.result()
                    except PlanParseError as exc:
                        self._log_agent_error(f"parse error: {exc}")
                        plan = None
                    except Exception as exc:  # noqa: BLE001
                        self._log_agent_error(f"unhandled: {exc}")
                        plan = None
                    plan_task = None
                    if plan is not None:
                        fast_budget = await self._integrate_plan(plan, latency_ms)
                elif (
                    plan_task is not None
                    and _time.monotonic() - plan_started > self._PLANNER_TIMEOUT_S
                ):
                    plan_task.cancel()
                    plan_task = None
                    self._log_agent_error(
                        f"planning turn exceeded {self._PLANNER_TIMEOUT_S:.0f}s; "
                        "cancelled (fast lane unaffected, will retry)"
                    )

                # 2. Start a planning task when the budget is spent, or
                # immediately when a metric goal just completed (the
                # world changed in a way the planner must re-plan for),
                # or when the fast window is within 2 exchanges of its
                # overflow limit. Without the pre-overflow trigger, the
                # window wipes cold (no FULL plan ran → stale or empty
                # journal/scratchpad in the rebuild) and the model
                # starts from square one on turn 13.
                if self._goal_event:
                    fast_budget = 0
                    self._goal_event = False
                if (
                    plan_task is None
                    and self._fast is not None
                    and self._fast.window_size() >= FastTurnRunner._MAX_EXCHANGES - 2
                ):
                    fast_budget = 0  # force planner before overflow wipe
                if plan_task is None and fast_budget <= 0:
                    plan_task = asyncio.create_task(
                        self._timed_plan_turn(), name=f"plan-{self._session_id}"
                    )
                    plan_started = _time.monotonic()

                # 3. Serial fallback: no fast lane -> just await the plan.
                if not fast_ok:
                    if plan_task is not None:
                        try:
                            plan, latency_ms = await plan_task
                        except PlanParseError as exc:
                            self._log_agent_error(f"parse error: {exc}")
                            plan = None
                        except Exception as exc:  # noqa: BLE001
                            self._log_agent_error(f"unhandled: {exc}")
                            plan = None
                        plan_task = None
                        if plan is None:
                            await asyncio.sleep(
                                self._DEFAULT_SLOW_PATH_INTERVAL_MS / 1000.0
                            )
                            continue
                        await self._integrate_plan(plan, latency_ms)
                        fast_budget = 0  # serial mode: every turn is FULL
                        await asyncio.sleep(self._next_check_in_ms / 1000.0)
                    continue

                # 4. Fast lane — runs whether or not a planner is thinking.
                try:
                    fplan = await self._run_fast_turn()
                except PlanParseError as exc:
                    self._log_agent_error(f"fast-turn parse error: {exc}")
                    fast_budget = 0  # escalate: plan as soon as possible
                    await asyncio.sleep(0.3)
                    continue
                except Exception as exc:  # noqa: BLE001
                    self._log_agent_error(f"fast-turn unhandled: {exc}")
                    fast_budget = 0
                    await asyncio.sleep(self._DEFAULT_SLOW_PATH_INTERVAL_MS / 1000.0)
                    continue
                if fast_budget > 0:
                    fast_budget -= 1
                if fplan.escalate:
                    # Plan NOW (next iteration starts the task) — but never
                    # stack a second planner on one already in flight.
                    fast_budget = 0
                # MEANINGFUL-ACTION GATE: a press identical to one that
                # just scored ~zero effect is wasted by definition —
                # block it at the harness and force a replan instead of
                # feeding the loop. (Audited live: 71% of presses were
                # dead; confirm worst at 82%, with 36 runs of 3+
                # identical presses in one session.)
                kept: list[PlanAction] = []
                for a in fplan.actions:
                    if (
                        self._last_dead_press is not None
                        and a.semantic == self._last_dead_press
                    ):
                        # LIVELOCK GUARD (run-20 lesson): a blocked press
                        # never executes, so its fx never updates, so the
                        # block never clears on its own — blocking the
                        # same press forever froze a whole session on the
                        # title screen. Block at most twice, tell the
                        # model (BLOCKED= next delta), then let it
                        # through: one re-probe re-measures reality.
                        self._dead_block_count += 1
                        if self._dead_block_count > 2:
                            self._last_dead_press = None
                            self._dead_block_count = 0
                            kept.append(a)
                            continue
                        self._blocked_semantic = a.semantic
                        self._live_log.append(
                            {
                                "t": self._clock.elapsed_ms(),
                                "kind": "event",
                                "payload": {
                                    "channel": "log",
                                    "data": {
                                        "event": "dead_press_blocked",
                                        "semantic": a.semantic,
                                    },
                                },
                            }
                        )
                        fast_budget = 0  # the approach isn't working
                    else:
                        kept.append(a)
                for action in kept:
                    await self._action_queue.put((action, "agent"))
                self._last_emitted = [a.semantic for a in kept]
                # Fast cadence floor: 200ms keeps a runaway "next_ms":50
                # from busy-spinning the GPU; the model can still ask for
                # long waits.
                await asyncio.sleep(max(200, fplan.next_check_in_ms) / 1000.0)
        finally:
            if plan_task is not None:
                plan_task.cancel()

    async def _speak(self, text: str) -> None:
        """Synthesize and push companion audio to the bridge.

        Tolerant of every TTS failure: a missing provider, a slow
        backend, or a torn-down bridge all degrade the session to
        text-only rather than crashing the slow-path loop.
        """

        if self._voice is None:
            return
        # Voice override comes from the companion persona when one is
        # attached; otherwise the bridge's default voice is used. A
        # persona with an empty ``voice`` collapses to that default.
        voice_override = self._persona.voice if self._persona is not None else ""
        bytes_b64, mime = await self._voice.synthesize_b64(
            text, voice=voice_override or None,
        )
        if not bytes_b64:
            return
        adapter = self._adapter
        if not isinstance(adapter, BridgedAdapter):
            return
        try:
            await adapter.push_audio(mime=mime, bytes_b64=bytes_b64, utterance=text)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "orchestrator.speak_failed",
                session_id=self._session_id,
                error=str(exc),
            )

    async def _run_slow_path_turn(self) -> PlanPayload:
        """One slow-path planning turn."""

        tail = self._tail[-self._SLOW_PATH_TAIL_LIMIT :]
        frames: list[bytes] = []
        if "frame" in self._caps.observation_modalities:
            # Pull up to 3 recent frames so the agent has temporal
            # context (motion, animation, action causality). Bridged
            # adapters expose snapshot_frames; older adapters that only
            # implement snapshot_frame degrade to a single-frame list.
            snap_frames = getattr(self._adapter, "snapshot_frames", None)
            if callable(snap_frames):
                frames = await snap_frames(n=self._SLOW_PATH_FRAME_CHUNK)
            else:
                f = await self._adapter.snapshot_frame()
                if f is not None:
                    frames = [f]
        # The planner's capture is the HIGH-RATE visual sample: it tracks
        # the agent's actual movement, where the 1.5s motion tick only
        # catches wherever it happened to be. Newest frame only — the
        # older ones in the window were already sampled on earlier turns.
        if frames:
            self._note_visual(frames[-1])
        # Prepend keyframes from recent ACTION turns (oldest, middle) so
        # the planner sees the visual arc of its own moves — the loop
        # detector's raw material — not just the last ~3 seconds. Dedup
        # below collapses them when the screen never changed, which is
        # itself the signal.
        keyframes = self._keyframe_sample(self._frame_ring)
        if keyframes and frames:
            frames = keyframes + list(frames)
        # Perception pass: collapse redundant near-identical frames and
        # (optionally) overlay a labeled Set-of-Marks grid for spatial
        # grounding. Surface-agnostic; toggled per-deployment. ``frame_note``
        # tells the model about the grid + any dedup so it reads the
        # temporal window honestly instead of hallucinating motion across
        # duplicate ticks. See augmentum/game_agent/perception.py.
        frame_note = ""
        if frames:
            prepared = prepare_frames(
                list(frames),
                dedup=getattr(settings, "game_agent_frame_dedup_enabled", True),
                grid=getattr(settings, "game_agent_grid_overlay_enabled", True),
                max_edge=int(getattr(settings, "game_agent_frame_max_edge", 480)),
            )
            frames = prepared.frames
            frame_note = prepared.note
            if keyframes:
                frame_note = (
                    f"{frame_note} The earliest frame(s) are snapshots from "
                    "your recent ACTION turns (not a fixed 1s cadence) — "
                    "compare them to the newest frame: if the screen barely "
                    "changed across several of your actions, you are looping "
                    "and must change approach."
                ).strip()
        # Snapshot the overlay at turn time so the prompt sees a
        # consistent view; concurrent updates during the LLM call don't
        # mutate what the model just read. The scene narrator's latest
        # description rides along as ``scene_feed`` — the planner reads
        # the same eyes the actor does.
        overlay = dict(self._overlay) if self._overlay else {}
        if self._scene:
            overlay["scene_feed"] = self._scene
        goals_line = self._world.goals_line()
        if goals_line:
            overlay["goals"] = goals_line
        # Grounding scaffold for the planner (text Set-of-Marks): the
        # local walkability map around the player, '.'=walkable
        # '#'=blocked '@'=you. The fast lane never sees this raw — it
        # uses navigate_to; the planner uses it to CHOOSE targets.
        grid_f = self._world.facts.get("walk_grid")
        px_f = self._world.facts.get("player_x")
        py_f = self._world.facts.get("player_y")
        if grid_f is not None and isinstance(grid_f.value, dict) \
                and px_f is not None and py_f is not None:
            g = grid_f.value
            rows = list(g.get("rows") or [])
            x0, y0 = int(g.get("x0", 0)), int(g.get("y0", 0))
            cy, cx = int(py_f.value) - y0, int(px_f.value) - x0
            if 0 <= cy < len(rows) and 0 <= cx < len(rows[cy]):
                rows[cy] = rows[cy][:cx] + "@" + rows[cy][cx + 1:]
            overlay["walk_map"] = {
                "top_left_xy": [x0, y0],
                "legend": ".walk #block @you (tile coords for navigate_to)",
                "rows": rows,
            }
        overlay = overlay or None
        # Same snapshot discipline for the journal: render its current
        # state into the prompt; any concurrent edits would be a logic
        # bug since the orchestrator is the only writer.
        journal_view = (
            self._journal.to_prompt_dict() if self._journal is not None else None
        )
        plan = await self._agent.think(
            live_log_tail=tail, frames=frames,
            overlay=overlay, journal=journal_view,
            frame_note=frame_note,
            lore=list(self._lore) if self._lore else None,
            lore_summary=list(self._lore_summary) if self._lore_summary else None,
            playbook=(
                self._playbook.to_prompt_dict()
                if self._playbook is not None
                else None
            ),
        )
        self._live_log.append(
            {
                "t": self._clock.elapsed_ms(),
                "kind": "plan",
                "payload": _PlanPayloadAdapter.dump_python(plan),
            }
        )
        # Apply the plan's goal-stack patch (goals as DATA; metric
        # goals self-complete against the blackboard).
        if plan.goal_update:
            try:
                self._world.apply_goal_update(
                    plan.goal_update, t_ms=self._clock.elapsed_ms()
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "orchestrator.goal_update_failed",
                    session_id=self._session_id,
                    error=str(exc),
                )
        # Apply the agent's playbook patch (cross-title lessons).
        if self._playbook is not None and plan.playbook_update:
            try:
                if self._playbook.apply_update(plan.playbook_update):
                    self._playbook.save()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "orchestrator.playbook_apply_failed",
                    session_id=self._session_id,
                    error=str(exc),
                )
        # Apply the agent's journal patch (if any) and persist.
        # CompanionJournal.apply_update enforces caps so a runaway model
        # can't blow up the prompt budget across turns. Save runs in a
        # try/except internally; failures degrade to "in-memory only"
        # without crashing the loop.
        if self._journal is not None and plan.journal_update:
            try:
                if self._journal.apply_update(plan.journal_update):
                    self._journal.save()
            except Exception as exc:  # noqa: BLE001
                # Defensive -- apply_update / save already handle their
                # own errors, but a logic bug inside merge() shouldn't
                # break the slow path.
                log.warning(
                    "orchestrator.journal_apply_failed",
                    session_id=self._session_id,
                    error=str(exc),
                )
        return plan
