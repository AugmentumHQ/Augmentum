"""CoderState — agent execution state for the Coder mode plan/act loop.

Each agent session tracks its phase, plan, step progress, file working set,
and read-before-edit guard through this dataclass. State is persisted to the
``coder_sessions`` SQLite table and can be round-tripped via ``to_dict`` /
``from_row`` without data loss.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from augmentum.loops.ledger import ObservationLedger
from augmentum.modes.coder.intent import TurnIntent
from augmentum.promises.models import Promise


class CoderPhase(str, Enum):
    """Lifecycle phases of a coder agent session."""

    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    WAITING = "waiting"


@dataclass
class CoderState:
    """Full execution state for a single coder agent session.

    Attributes
    ----------
    session_id:
        The chat session this agent run is attached to.
    workspace_id:
        The Docker workspace container being used.
    phase:
        Current lifecycle phase (planning → executing → reviewing → waiting).
    plan:
        Raw plan text produced by the planning phase.
    plan_steps:
        Parsed list of step descriptions extracted from the plan.
    current_step:
        Zero-based index of the step currently being executed.
    step_outputs:
        Mapping of step index (as str) → tool output produced for that step.
    working_set:
        Set of file paths the agent has declared it is actively editing.
        Auto-populated by ``record_file_read`` so callers rarely need to
        maintain it manually.
    files_read:
        Set of file paths that have been read (fetched from the workspace).
        The read-before-edit guard (``can_edit``) relies on this set.
    tool_calls_made:
        Running count of tool invocations in this session.
    error:
        Last error message, or ``None`` if no error has occurred.
    created_at:
        Unix timestamp when the state was first created.
    updated_at:
        Unix timestamp of the most recent mutation.
    """

    session_id: str
    workspace_id: str
    # Owning Project ID. Phase 1 / PR-1.2 adds this so that turn_summaries +
    # other in-memory state can be looked up by project across container
    # recycle (a fresh workspace gets a new workspace_id but the project
    # stays). Empty string for legacy rows before migration 200.
    project_id: str = ""

    # --- phase ---
    phase: CoderPhase = CoderPhase.WAITING

    # --- plan ---
    plan: str = ""
    plan_steps: list[str] = field(default_factory=list)

    # --- mission (structured plan — supersedes plan_steps for new code) ---
    mission: list[Promise] = field(default_factory=list)

    # --- progress ---
    current_step: int = 0
    step_outputs: dict[str, str] = field(default_factory=dict)

    # --- file tracking ---
    working_set: set[str] = field(default_factory=set)
    # Path → mtime at read time (container epoch seconds). The
    # read-before-edit guard compares current mtime against the stored
    # one: if the file changed externally since our read (user edit,
    # git pull, another agent), ``can_edit`` returns False and forces
    # a re-read. Pre-2026-04-20 this was a plain ``set[str]`` that
    # only answered "has this been read at all?" — allowing silent
    # clobbering of out-of-date content. A stored mtime of
    # ``float("inf")`` means "read but mtime unknown" (e.g. tool
    # didn't report it); we treat that as never-stale for backward
    # compat with pre-fix reads.
    files_read: dict[str, float] = field(default_factory=dict)

    # --- metrics ---
    tool_calls_made: int = 0

    # --- task list (Claude Code / Codex style) ---
    # Structured task list the model updates via the ``task_list`` tool.
    # Each item: ``{content, activeForm, status}`` where status is one of
    # ``pending`` / ``in_progress`` / ``completed``. Invariant (enforced
    # by the tool): at most one item is ``in_progress``. Re-rendered into
    # the system-reminder each iteration so compaction can't eat it.
    tasks: list[dict] = field(default_factory=list)

    # --- sticky reminder signals ---
    # Ring buffer of the most recent validation errors seen across tool
    # calls. Fed into the sticky reminder so the model sees its own
    # pattern of malformed calls — a capability the raw message history
    # loses under compaction. Capped at 3; older entries fall off.
    recent_validation_errors: list[dict] = field(default_factory=list)

    # Ring buffer of REPEATED soft failures — tool calls that returned
    # success=False but NOT for schema reasons. Classic examples: the
    # mtime-guard rejecting an edit on a stale read, file_read on a
    # non-existent path, shell_exec on a missing binary. Dedupe KEY is
    # (tool_name, target) so "code_edit /snake.html" and "code_edit
    # /fib.html" are tracked separately even though both are code_edit.
    # Surfaced in the sticky reminder so the model sees the pattern —
    # observed 2026-04-20 a model retried the same rejected code_edit
    # 20× because the error lived only in the message history and the
    # model stopped attending to older turns. Capped at 4 entries.
    recent_tool_failures: list[dict] = field(default_factory=list)

    # Fingerprint of productive tool calls (successful reads / shell ops /
    # greps) the model has already performed this request. Surfaced in
    # the sticky reminder as "Already inspected" so the model doesn't
    # re-read the same file 5× because it lost track in its own history.
    # Observed 2026-04-20: without this signal, weak models loop on
    # file_read + cat + ls variants of the same paths. The dedup KEY is
    # intent-level (file_read by path, shell by exact command) so re-runs
    # for a legitimate reason (file was modified) can still happen — the
    # model just sees "you already ran this once" and has to decide.
    recent_tool_calls: list[dict] = field(default_factory=list)

    # Background processes the agent spawned this turn via shell_exec —
    # commands containing ``&`` (trailing), ``nohup``, ``setsid``, or
    # ``disown``. Observed 2026-04-22: agent ran ``server &``, later ran
    # ``server &`` again, hit ``Address already in use``, entered a
    # kill/restart/check spiral that only terminated when the 20-iter
    # action_stagnation breaker fired. The agent has no visibility into
    # what it's already started, so it can't reason about conflicts.
    # Surface in the sticky reminder as "Background processes started:".
    # Entries: {iteration, command (trimmed to 120 chars)}. Capped at 8;
    # older entries fall off.
    background_processes: list[dict] = field(default_factory=list)

    # --- loop-termination signal (finish_task tool) ---
    # The ``finish_task`` tool sets ``finish_requested = True`` with the
    # model's exit summary in ``finish_summary``. The act loop checks
    # these at iteration top and terminates cleanly — the model's safe
    # exit when ``tool_choice="required"`` forbids a text-only stop,
    # and also a generally useful "I'm done" signal for any strategy.
    # Reset per-request in ``_reset_for_new_request`` so a prior turn's
    # finish doesn't short-circuit the new one.
    finish_requested: bool = False
    finish_summary: str = ""

    # --- model-initiated compaction (compact tool) ---
    # The ``compact`` tool sets ``compact_requested = True`` with the
    # model's self-written handoff note in ``compact_note`` (the four
    # State/Decisions/Learnings/Next lines — same shape the second-model
    # synthesis emits). The act loop's compaction step consumes the flag
    # on the next iteration and folds history with the model's note as
    # the synthesis segment, bypassing the token threshold (the model is
    # choosing a semantic seam, which usually arrives before the
    # pressure ceiling). ``compact_tool_uses`` caps calls per turn so a
    # weak model can't checkpoint-spam. Reset per-request alongside the
    # finish flags.
    compact_requested: bool = False
    compact_note: str = ""
    compact_tool_uses: int = 0

    # --- reviewable-turn flow ---
    # ``active_turn_id`` is a stable identifier for the currently
    # executing turn — set by the handler at ``_reset_for_new_request``
    # time and threaded into the ReviewBundle when the turn ends.
    # ``active_turn_snapshot`` holds the per-turn pre-write capture
    # (see augmentum/coder/turn_snapshot.py) — mutating tools call
    # ``snapshot_before_write`` on it before writing to disk; the
    # handler collects diffs at turn end. Both reset per-request.
    # Typed as ``Any`` to avoid a runtime import cycle — the real
    # type is :class:`~augmentum.coder.turn_snapshot.TurnSnapshot`.
    active_turn_id: str = ""
    active_turn_snapshot: Any = None

    # --- turn intent (priming tree key) ---
    # Classification of the current turn into a coarse task shape
    # (INSPECT/REVIEW/IMPLEMENT/DEBUG/OPERATE/RESEARCH/UNKNOWN). Set by the
    # handler at request entry via classify_turn_intent and consumed by
    # prompt builders (exemplar loader, tool shortlist, sticky-reminder
    # trim) to tailor the priming the model sees. Reset per-request so
    # the classification reflects the latest user message; the previous
    # turn's intent doesn't bleed into a follow-up. ``None`` means "not
    # yet classified" — callers should treat this as UNKNOWN.
    current_intent: TurnIntent | None = None

    # --- priming telemetry (Sprint 1 measurement) ---
    # Per-branch token counts from the most recent _build_act_system
    # call. Populated as a side effect so the function can stay a pure
    # str-returning helper. Read by the ledger's finish_run at turn
    # close to persist into coder_turn_runs.priming_telemetry. Shape:
    #   {
    #     "intent": "DEBUG", "tier": "native",
    #     "branches": {"rules": 634, "exemplar": 281, ...},
    #     "total_priming_tokens": 1829,
    #     "exemplar_loaded": True
    #   }
    last_priming_telemetry: dict[str, Any] = field(default_factory=dict)

    # --- turn summaries (cross-turn trace persistence) ---
    # FIFO queue of summaries — one per completed _act_hybrid /
    # _act_canonical turn — that the next turn's system prompt injects
    # as a <prior_turns> block. Solves the "model re-reads the same
    # file every turn because it has no memory of last turn's work"
    # problem described 2026-04-20. Each entry is:
    #   {turn_idx, user_goal, files_read, files_edited, outcome,
    #    blockers, created_at}
    # Capped at _TURN_SUMMARY_MAX (10) so long sessions don't bloat the
    # prompt; older entries fall off the front. NOT cleared by
    # _reset_for_new_request — that's the whole point of this field.
    turn_summaries: list[dict] = field(default_factory=list)

    # --- subagent context bridge (set by handler._refresh_kernel_facts) ---
    # Mirror of the handler's cached <workspace_facts> block, parked on the
    # state so the SubagentDispatcher's context bridge
    # (augmentum/agents/context_bridge.py::extract_workspace_facts) can read
    # it when spawning task_dispatch children. Without this the bridge reads
    # an unset attribute and EVERY subagent launches context-blind — the
    # objective never crosses into the child. Set each turn; "" when the
    # kernel is disabled or the workspace has no facts yet.
    kernel_facts_text: str = ""
    # Compact objective + project-shape anchor (~240 chars) handed to even
    # slim-context roles (research, audit_zone) so their answers fit THIS
    # stack instead of drifting generic. Rendered by
    # WorkspaceKernel.render_orientation. Set alongside kernel_facts_text.
    orientation_text: str = ""

    # --- pending objective contract (cross-turn completion requirement) ---
    # Compact persisted reminder of what still must be PROVEN or stated
    # plainly before the current objective can be considered done.
    # Unlike ``tasks`` / ``plan_steps``, this is acceptance-oriented:
    # examples include "remote/public URL must be verified, not just
    # printed" or "state the concrete blocker plainly". Used to keep
    # continuation turns grounded on unresolved end conditions even when
    # the task list says "completed" or the prior turn ended on a nudge.
    pending_objective_contract: dict[str, Any] = field(default_factory=dict)

    # --- loop budget (earned-autonomy model) ---
    # Initialized from defaults at the start of each act phase; the loop
    # mutates these as the agent earns or loses runway based on signals.
    iterations_remaining: int = 20
    iterations_ceiling: int = 75
    iterations_since_progress: int = 0
    fanout_limit: int = 5
    consecutive_failures: int = 0

    # --- error ---
    error: str | None = None

    # --- safeguards toggle (loaded from project_checkouts.safeguards_enabled
    # at turn start; not serialized via to_dict / from_row because the
    # checkout row is the source of truth). When False, phase_act.py
    # bypasses the soft circuit-breakers (action_stagnation, test_failure
    # _streak, same_file_edit_break, etc.) and the hard iteration ceiling
    # is raised. Use for strong models that legitimately run long.
    safeguards_enabled: bool = True

    # --- timestamps ---
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        # Phase 2 / PR-2.2: the ledger shares list references with this
        # dataclass — every existing direct-mutation pattern (handler,
        # tests, JSON round-trip) keeps working while methods delegate.
        # See augmentum/loops/ledger.py for the implementation.
        self.ledger = ObservationLedger.from_lists(
            recent_validation_errors=self.recent_validation_errors,
            recent_tool_failures=self.recent_tool_failures,
            recent_tool_calls=self.recent_tool_calls,
            background_processes=self.background_processes,
        )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def total_steps(self) -> int:
        """Total number of steps in the current plan."""
        return len(self.plan_steps)

    @property
    def progress_pct(self) -> float:
        """Completion percentage in [0.0, 100.0].

        Returns 0.0 when there are no steps to avoid division-by-zero.
        Returns 100.0 only after all steps have been completed (i.e.
        ``current_step == total_steps``).
        """
        if self.total_steps == 0:
            return 0.0
        # Clamp to [0, total_steps] so we never exceed 100 %.
        completed = min(self.current_step, self.total_steps)
        return (completed / self.total_steps) * 100.0

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def advance_step(self, output: str) -> None:
        """Record *output* for the current step then move to the next step.

        If the agent is already past the last step the call is a no-op
        (idempotent guard against double-advance).

        Parameters
        ----------
        output:
            Tool / LLM output that was produced for the step being completed.
        """
        if self.total_steps == 0 or self.current_step >= self.total_steps:
            return
        self.step_outputs[str(self.current_step)] = output
        self.current_step += 1
        self.updated_at = time.time()

    def record_file_read(
        self, path: str, mtime: float | None = None,
    ) -> None:
        """Mark *path* as having been read and add it to the working set.

        Both ``files_read`` and ``working_set`` are updated atomically so
        they remain consistent.

        Parameters
        ----------
        path:
            Workspace-relative or absolute path of the file that was read.
        mtime:
            Container-reported modification time at the moment of the
            read (epoch seconds). Stored so ``can_edit`` can compare
            against current mtime and reject stale reads. Pass ``None``
            for callers that don't have the stat (e.g. old code paths,
            shell-based reads) — we store ``float("inf")`` which
            effectively disables the staleness check for that entry.
        """
        self.files_read[path] = (
            float(mtime) if mtime is not None else float("inf")
        )
        self.working_set.add(path)
        self.updated_at = time.time()

    def can_edit(
        self, path: str, current_mtime: float | None = None,
    ) -> bool:
        """Return ``True`` iff *path* has been read AND is not stale.

        Two checks:
          1. Is the path in ``files_read`` at all? (original guard)
          2. If ``current_mtime`` is provided, is our stored read
             mtime ≥ the current mtime? (i.e. no external edit since.)

        ``current_mtime`` defaults to ``None`` for backward compat —
        callers that can't stat the file (or don't want the check)
        get the original "any-read-is-fine" behaviour. Callers that
        DO stat get the strict staleness guard.

        A stored mtime of ``float("inf")`` (from a pre-mtime read) is
        always considered fresh — we can't know otherwise and blocking
        the edit would break existing sessions.
        """
        if path not in self.files_read:
            return False
        if current_mtime is None:
            return True
        stored = self.files_read[path]
        if stored == float("inf"):
            return True
        return stored >= float(current_mtime)

    def set_tasks(self, items: list[dict]) -> None:
        """Replace the task list wholesale (Claude Code semantics).

        ``items`` is a list of ``{content, activeForm, status}`` dicts;
        invariants (e.g. at most one ``in_progress``) are the caller's
        responsibility — this method does not validate. Updates
        ``updated_at`` so the sticky reminder can show freshness.
        """
        self.tasks = list(items)
        self.updated_at = time.time()

    def active_task(self) -> dict | None:
        """The one task currently ``in_progress``, or None.

        Convenience for the sticky reminder renderer. Returns the first
        in_progress task in list order on the off chance the invariant
        is violated by an earlier tool call.
        """
        for t in self.tasks:
            if isinstance(t, dict) and t.get("status") == "in_progress":
                return t
        return None

    def add_turn_summary(self, summary: dict, max_kept: int = 10) -> None:
        """Append a turn summary and drop the oldest if over ``max_kept``.

        ``summary`` is the dict produced by ``CoderHandler._build_turn_summary``:
        ``{turn_idx, user_goal, files_read, files_edited, outcome,
        blockers, created_at}``. This method only enforces the FIFO cap
        and timestamps; schema validation is the caller's job.
        """
        self.turn_summaries.append(summary)
        if len(self.turn_summaries) > max_kept:
            # Drop oldest first so the most-recent N turns survive.
            self.turn_summaries = self.turn_summaries[-max_kept:]
        self.updated_at = time.time()

    def set_pending_objective_contract(self, contract: dict[str, Any]) -> None:
        """Persist the current unresolved completion contract.

        ``contract`` should be a compact JSON-serialisable dict. Empty or
        falsey input clears the current contract.
        """
        self.pending_objective_contract = dict(contract or {})
        self.updated_at = time.time()

    def clear_pending_objective_contract(self) -> None:
        """Drop any unresolved completion contract from state."""
        if self.pending_objective_contract:
            self.pending_objective_contract = {}
            self.updated_at = time.time()

    def record_validation_error(
        self, *, tool_name: str, error: str, max_kept: int = 3,
    ) -> None:
        """Remember a malformed tool call so the sticky reminder can show it.

        Phase 2 / PR-2.2: delegates to :class:`ObservationLedger`. The
        ledger shares its bucket list reference with this dataclass so
        ``self.recent_validation_errors`` keeps reflecting the current
        state for to_dict / from_row / direct-attribute callers.
        """
        self.ledger.record_validation_error(
            tool_name=tool_name, error=error, max_kept=max_kept,
        )
        self.updated_at = time.time()

    def clear_validation_errors(self) -> None:
        """Drop every validation-error entry. Bumps ``updated_at`` only
        when something was actually cleared."""
        if self.ledger.clear_validation_errors():
            self.updated_at = time.time()

    def record_tool_failure(
        self, *, tool_name: str, target: str, error: str, max_kept: int = 4,
    ) -> None:
        """Remember a REPEATED soft failure (non-schema, success=False).

        Phase 2 / PR-2.2: delegates to :class:`ObservationLedger`.
        Behavioural contract unchanged — dedupe key ``(tool_name,
        target)``; stale entries pruned before recording so the
        cross-turn ledger stays fresh without a scheduler.
        """
        self.ledger.record_tool_failure(
            tool_name=tool_name, target=target, error=error, max_kept=max_kept,
        )
        self.updated_at = time.time()

    def prune_stale_tool_failures(
        self, *, ttl_seconds: float | None = None,
    ) -> int:
        """Drop failure entries older than ``ttl_seconds`` since
        last_at. Defaults to
        :data:`augmentum.loops.ledger.FAILURE_LEDGER_TTL_SECONDS`
        (30 min). Returns the count dropped."""
        dropped = self.ledger.prune_stale_tool_failures(ttl_seconds=ttl_seconds)
        if dropped:
            self.updated_at = time.time()
        return dropped

    def clear_tool_failures(self) -> None:
        """Reset the soft-failure tracker. Phase 2.2 made the ledger
        cross-turn; this method is kept for explicit "fresh slate"
        use cases but is NOT called from ``_reset_for_new_request``
        — that path uses :meth:`prune_stale_tool_failures` instead.
        """
        if self.ledger.clear_tool_failures():
            self.updated_at = time.time()

    def record_tool_call(
        self,
        *,
        tool_name: str,
        tool_input: dict,
        iteration: int,
        max_kept: int = 8,
    ) -> None:
        """Remember a productive tool call so the reminder can show it.

        Phase 2 / PR-2.2: delegates to :class:`ObservationLedger`.
        Intent-keyed dedup (path / command / query depending on tool)
        and the tracked-tool sets live in
        :mod:`augmentum.loops.ledger`.
        """
        self.ledger.record_tool_call(
            tool_name=tool_name, tool_input=tool_input,
            iteration=iteration, max_kept=max_kept,
        )
        self.updated_at = time.time()

    def hit_repeat_cap(
        self, *, tool_name: str, tool_input: dict, cap: int = 5,
    ) -> bool:
        """Has this exact ``(tool, key)`` been called ``cap`` or more
        times? Non-destructive — used as a hard safety net when the
        sticky reminder fails to dissuade a looping weak model."""
        return self.ledger.hit_repeat_cap(
            tool_name=tool_name, tool_input=tool_input, cap=cap,
        )

    def repeat_count(
        self, *, tool_name: str, tool_input: dict,
    ) -> int:
        """Return how many times this ``(tool, intent_key)`` has been
        called. Returns 0 for untracked tools / missing intent keys.
        The look-before-leap sibling of :meth:`hit_repeat_cap`."""
        return self.ledger.repeat_count(
            tool_name=tool_name, tool_input=tool_input,
        )

    def record_background_process(
        self, *, command: str, iteration: int, max_kept: int = 8,
    ) -> None:
        """Note that the agent started a backgrounded shell command.
        Phase 2 / PR-2.2: delegates to :class:`ObservationLedger`."""
        self.ledger.record_background_process(
            command=command, iteration=iteration, max_kept=max_kept,
        )
        self.updated_at = time.time()

    def clear_tool_calls_for_path(self, path: str) -> None:
        """Drop every ``recent_tool_calls`` entry keyed on this path.

        Phase 2 / PR-2.2: delegates to :class:`ObservationLedger`.
        Called after a successful mutation so subsequent re-reads
        start at count=0."""
        if self.ledger.clear_tool_calls_for_path(path):
            self.updated_at = time.time()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a flat dict suitable for SQLite insertion.

        ``set`` fields are converted to sorted JSON lists so that the stored
        representation is deterministic and round-trips cleanly.
        """
        return {
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            # Empty string means "legacy unscoped" in memory, but the DB
            # column FKs projects(id) — '' is a VALUE to the FK checker
            # (unlike NULL) and fails the constraint, killing the entire
            # row write. Persist unscoped as NULL. (from_row maps it back
            # to "".)
            "project_id": self.project_id or None,
            "phase": self.phase.value,
            "plan": self.plan,
            "plan_steps": json.dumps(self.plan_steps),
            "mission": json.dumps([p.to_dict() for p in self.mission]),
            "current_step": self.current_step,
            "step_outputs": json.dumps(self.step_outputs),
            "working_set": json.dumps(sorted(self.working_set)),
            # files_read is now a dict (path → mtime). Serialise with
            # sorted keys for deterministic persistence. ``inf``
            # marshals as JSON-null via the default= handler below.
            "files_read": json.dumps(
                {k: (v if v != float("inf") else None)
                 for k, v in sorted(self.files_read.items())},
            ),
            "tool_calls_made": self.tool_calls_made,
            "tasks": json.dumps(self.tasks),
            "recent_validation_errors": json.dumps(self.recent_validation_errors),
            "recent_tool_failures": json.dumps(self.recent_tool_failures),
            "recent_tool_calls": json.dumps(self.recent_tool_calls),
            "background_processes": json.dumps(self.background_processes),
            "turn_summaries": json.dumps(self.turn_summaries),
            "pending_objective_contract": json.dumps(self.pending_objective_contract),
            "iterations_remaining": self.iterations_remaining,
            "iterations_ceiling": self.iterations_ceiling,
            "iterations_since_progress": self.iterations_since_progress,
            "fanout_limit": self.fanout_limit,
            "consecutive_failures": self.consecutive_failures,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> CoderState:
        """Deserialize a ``CoderState`` from a SQLite row dict.

        Handles JSON decoding of list/dict/set columns and coerces the
        ``phase`` string back to a ``CoderPhase`` enum value.

        Parameters
        ----------
        row:
            Mapping of column names to raw SQLite values (as returned by
            ``aiosqlite`` with ``row_factory = aiosqlite.Row`` or a plain
            ``dict``).
        """
        return cls(
            session_id=row["session_id"],
            workspace_id=row["workspace_id"],
            # project_id was added by migration 200. Older rows have no
            # column or null; default to empty string so downstream
            # code sees the same "legacy unscoped" shape it had before.
            project_id=row.get("project_id") or "",
            phase=CoderPhase(row["phase"]),
            plan=row.get("plan") or "",
            plan_steps=json.loads(row.get("plan_steps") or "[]"),
            mission=[
                Promise.from_dict(d)
                for d in json.loads(row.get("mission") or "[]")
            ],
            current_step=row.get("current_step") or 0,
            step_outputs=json.loads(row.get("step_outputs") or "{}"),
            working_set=set(json.loads(row.get("working_set") or "[]")),
            # files_read: backward compat — old rows stored a list of
            # paths (no mtime). Load those as dict with inf mtime so
            # the can_edit staleness check degrades to the original
            # "read means unstale" behaviour. New rows store a dict.
            files_read=(
                (lambda raw: (
                    {p: float("inf") for p in raw}
                    if isinstance(raw, list)
                    else {
                        p: (float("inf") if v is None else float(v))
                        for p, v in raw.items()
                    }
                ))(json.loads(row.get("files_read") or "{}"))
            ),
            tool_calls_made=row.get("tool_calls_made") or 0,
            tasks=json.loads(row.get("tasks") or "[]"),
            recent_validation_errors=json.loads(
                row.get("recent_validation_errors") or "[]",
            ),
            recent_tool_failures=json.loads(
                row.get("recent_tool_failures") or "[]",
            ),
            recent_tool_calls=json.loads(
                row.get("recent_tool_calls") or "[]",
            ),
            background_processes=json.loads(
                row.get("background_processes") or "[]",
            ),
            turn_summaries=json.loads(row.get("turn_summaries") or "[]"),
            pending_objective_contract=json.loads(
                row.get("pending_objective_contract") or "{}",
            ),
            iterations_remaining=row.get("iterations_remaining") or 20,
            iterations_ceiling=row.get("iterations_ceiling") or 75,
            iterations_since_progress=row.get("iterations_since_progress") or 0,
            fanout_limit=row.get("fanout_limit") or 5,
            consecutive_failures=row.get("consecutive_failures") or 0,
            error=row.get("error"),
            created_at=row.get("created_at") or time.time(),
            updated_at=row.get("updated_at") or time.time(),
        )
