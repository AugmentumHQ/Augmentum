"""Bug-finder pipeline driver.

Implements the eight-stage pipeline from
``docs/superpowers/specs/2026-05-10-bug-finder-mode-design.md``:

    intake → workspace prep → plan → detect →
        verify-is-real → fix → (improve, Phase 2) → report

This module owns the high-level flow + ``ContainerManager`` interactions.
Subagent loops, guards, prompts, and parsers all live in sibling modules
and are composed here.

Phase 1 simplifications (deliberate, see design doc §Phasing):

* Sequential fixers — no per-fix container fork. We get equivalent
  isolation via git snapshot + reset around each attempt, because
  there's no concurrent agent contending for the workspace.
* Stage 7 (improve) is not implemented.
* No auto-apply to the user's repo; patches are emitted as artifacts.

The whole pipeline is one ``async def`` that takes a config and returns
a ``BugFinderRunReport``. It accepts an optional ``JobContext`` so it
can run as a background job that survives client disconnects.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any

from augmentum.bug_finder.agnostic_stage import (
    AgnosticStageResult,
    record_confirmation,
    run_agnostic_stage,
)
from augmentum.bug_finder.budget import SubagentBudget
from augmentum.bug_finder.detector_resilience import (
    DetectorCircuitBreaker,
    run_with_retry,
)
from augmentum.bug_finder.findings import (
    Finding,
    FindingStatus,
    confirmation_histogram,
    merge_runs,
    parse_detector_output,
    rank_findings,
)
from augmentum.bug_finder.guards import (
    detector_guard,
    fixer_guard,
    planner_guard,
)
from augmentum.bug_finder.json_salvage import salvage_json_object
from augmentum.bug_finder.prompts import (
    DETECTOR_SYSTEM_PROMPT,
    DETECTOR_USER_TEMPLATE,
    FIXER_SYSTEM_PROMPT,
    FIXER_USER_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
)
from augmentum.bug_finder.role_models import Role, RoleModelConfig
from augmentum.bug_finder.static_chunker import collect_static_chunks
from augmentum.bug_finder.subagent import (
    DETECTOR_TOOL_NAMES,
    FIXER_TOOL_NAMES,
    PLANNER_TOOL_NAMES,
    VERIFIER_TOOL_NAMES,
    SubagentResult,
    SubagentSpec,
    filter_tools,
    run_subagent,
)
from augmentum.bug_finder.verifier import (
    apply_fix_verify_outcome,
    apply_repro_outcome,
    make_fix_verify_spec,
    make_repro_spec,
    parse_fix_verify_result,
    parse_repro_result,
)
from augmentum.bug_finder.workspace import (
    PreparedWorkspace,
    WorkspaceBaseline,
    prepare_workspace,
)
from augmentum.coder.containers import ContainerManager
from augmentum.coder.state import CoderState
from augmentum.coder.tools import create_coder_tools
from augmentum.jobs.context import JobCancelled, JobContext
from augmentum.models.base import ModelBackend
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# A model resolver returns ``(backend, clean_model_name)`` for the given
# user-facing model identifier. The clean name is what we send to the
# backend's ``chat`` call (some providers want a prefix-stripped form,
# e.g. ``ollama/`` → ``llama3``). Implementations typically delegate to
# ``ProviderRegistry.resolve_backend_for_model``.
BackendResolver = Callable[[str], Awaitable[tuple[ModelBackend, str]]]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserGoal:
    """Structured north-star for one bug-finder run.

    Two operating modes:

    * ``mode="named-bug"`` — the user (or an orchestrator agent above
      us) has a specific bug in mind. ``description`` says what they
      think is wrong; ``repro_hint`` describes how to trigger it. The
      planner narrows attention to ``scope_paths`` and the threat
      model frames the bug class.

    * ``mode="explore"`` — a general sweep for production-grade
      hidden bugs. ``description`` (when present) gives the planner
      a high-level focus area; absent it, the comprehender's
      pillars + risk surfaces drive prioritization.

    The dataclass shape exists so this surface is **callable as a
    sub-agent**: a higher-level orchestrator (coder, companion, MCP
    client) interprets raw user intent and fills this in
    programmatically. The bug-finder doesn't need to handle
    ambiguous chat-speak — that's the orchestrator's job.

    Stays neutral / explore-mode by default so existing callers that
    don't set anything behave as before.
    """

    mode: str = "explore"
    description: str = ""
    repro_hint: str = ""
    scope_paths: tuple[str, ...] = ()
    severity_floor: str = "info"
    time_budget_minutes: int = 0

    def is_named_bug(self) -> bool:
        return self.mode == "named-bug" and bool(self.description.strip())

    def to_prompt_block(self) -> str:
        """Render as a system-prompt prefix block. Empty when no
        meaningful goal information is supplied."""
        if not self.description.strip() and not self.repro_hint.strip():
            return ""
        lines = ["## User goal (authoritative — drives prioritization)\n"]
        if self.mode:
            lines.append(f"**Mode:** {self.mode}")
        if self.description.strip():
            lines.append(f"**Focus:** {self.description.strip()}")
        if self.repro_hint.strip():
            lines.append(f"**Repro hint:** {self.repro_hint.strip()}")
        if self.scope_paths:
            lines.append(
                "**Scope paths:** " + ", ".join(self.scope_paths),
            )
        if self.severity_floor and self.severity_floor != "info":
            lines.append(f"**Severity floor:** {self.severity_floor}")
        return "\n".join(lines)


@dataclass(frozen=True)
class BugFinderIntake:
    """Inputs the user supplies to start a run.

    Workspaces are created and managed by the coder mode — bug finder
    receives an existing ``workspace_id`` and runs against it. There is
    no clone/intake path here.
    """

    workspace_id: str
    focus_paths: tuple[str, ...] = ()
    threat_model: str = ""
    user_goal: UserGoal = field(default_factory=UserGoal)
    """North-star for what this specific run should care about.

    See ``UserGoal``. Default is an empty explore-mode goal so
    existing callers behave exactly as before.
    """
    """User-supplied threat model (free-form markdown). Anthropic's bug-
    finder research names mismatched threat models as the #1 cause of
    valid-but-rejected findings ("40% FP rate where PoCs proved
    exploitability, but the team dismissed them because they didn't fit
    the project's threat model"). When provided, this string is
    prepended to detector + verifier system prompts so both subagents
    work from the same authoritative trust-boundary definition.

    Recommended shape (free-form, but these sections help):
      - Assets: what's valuable / sensitive in this codebase
      - Trust boundaries: where untrusted input enters
      - Attacker capabilities: network? authed? local? supply-chain?
      - In scope: bug classes worth surfacing
      - Out of scope: intentional design choices not to flag
    """

    prior_patterns: str = ""
    """Cross-run pattern memory rendered as a prompt-friendly brief.
    Built by `augmentum.bug_finder.patterns.render_pattern_brief` from
    the `bug_finder_patterns` table for this workspace. Empty by default
    (first run on a workspace has no priors). When set, prepended to the
    planner system prompt so chunk selection compounds across runs:
    files with prior hits earn extra attention without the model
    rediscovering them from scratch.

    Crucially: this is a *prior*, not a directive. The detector still
    has to find the bug — we just point the planner at the right
    chunks. A pattern that was previously fixed should *not* be
    re-flagged; the file getting a chunk doesn't mean the detector
    must produce a finding.
    """


@dataclass(frozen=True)
class BugFinderRunConfig:
    """Per-run configuration.

    Budgets are explicit (no implicit defaults beyond what
    ``SubagentBudget`` provides) so the caller can size them to the
    target's expected complexity.
    """

    intake: BugFinderIntake
    role_models: RoleModelConfig

    planner_budget: SubagentBudget = field(
        default_factory=lambda: SubagentBudget(
            max_iterations=20, max_wallclock_seconds=300, max_tokens=200_000,
        ),
    )
    detector_budget: SubagentBudget = field(
        default_factory=lambda: SubagentBudget(
            max_iterations=15, max_wallclock_seconds=240, max_tokens=120_000,
        ),
    )
    verifier_budget: SubagentBudget = field(
        default_factory=lambda: SubagentBudget(
            max_iterations=20, max_wallclock_seconds=300, max_tokens=150_000,
        ),
    )
    fixer_budget: SubagentBudget = field(
        default_factory=lambda: SubagentBudget(
            max_iterations=30, max_wallclock_seconds=600, max_tokens=300_000,
        ),
    )

    detector_runs_per_chunk: int = 3
    detector_concurrency: int = 4
    max_chunks: int = 40
    max_fix_attempts_per_finding: int = 3
    overall_wallclock_seconds: float = 1800.0

    detector_max_retries: int = 2
    """Bounded retry on a detector subagent that stops with
    ``stop_reason="error"`` (transient 429 / 5xx). 0 disables retry.
    Error runs die fast (~iteration 0), so retries are cheap insurance
    against an isolated transient losing a chunk permanently. Systemic
    outages are handled by the circuit breaker below, not by retrying
    every chunk to exhaustion."""

    detector_retry_base_delay_s: float = 2.0
    """Exponential-backoff base for detector retries: attempt N waits
    ``base * 2**N`` seconds. Held inside the per-detector semaphore slot,
    so retries also naturally throttle the fan-out under load."""

    detector_circuit_breaker_min_samples: int = 8
    """Minimum completed detectors before the circuit breaker can open.
    Guards against a couple of early flukes tripping a healthy run."""

    verifier_max_retries: int = 1
    """Bounded retry on a verifier subagent that stops with
    ``stop_reason="error"``. An errored verifier silently buries a real
    finding as "unconfirmable" (the confirm-stage-rejects-everything
    field failure), so a transient backend blip shouldn't cost a finding.
    Lower than the detector default (verifiers are more expensive and
    there's one per finding, far fewer than detectors)."""

    detector_circuit_breaker_error_rate: float = 0.6
    """Once ``min_samples`` detectors have run and this fraction errored,
    the breaker opens and remaining detectors short-circuit instead of
    hammering a down backend. Turns the 811K-token / 16-minute futile
    grind (06-14 field case) into a fast bail. The detector-health gate
    then reports the run degraded. Set to 1.0 to effectively disable."""

    detector_error_rate_threshold: float = 0.5
    """Detector-health gate. When this fraction (or more) of the
    detector subagents stop with ``stop_reason="error"``, the run is
    DEGRADED rather than ``complete``: the scan didn't functionally run,
    so "no findings" would be a lie. ``_build_report`` rewrites the
    stop_reason to ``"degraded"`` and surfaces the error rate in
    stop_detail + notes. Field data (06-14: 396/399 detectors errored,
    reported as "complete / no findings") is exactly the failure this
    catches — a high-concurrency fan-out collapsing under provider
    rate-limits, mislabeled as a clean pass. Set to 1.0 to disable the
    gate (only an all-errored stage degrades), 0.0 to flag any error."""

    run_mode: str = "planner"
    """Chunk-selection strategy.

    * ``"planner"`` (default) — the standard pipeline. An LLM planner
      reads the codebase, decides which functions are bug-shaped, and
      emits a curated list of N chunks for the detector.
    * ``"static_chunk"`` — skip the planner entirely. The
      :mod:`augmentum.bug_finder.static_chunker` walks every Python
      file in ``focus_paths`` via AST and emits *every* qualifying
      function as a chunk. Detector + verifier + fixer downstream is
      unchanged. Use this for whole-project sweeps where the planner's
      token budget (capped at planner_budget.max_tokens) would
      otherwise drown reading code before producing chunks.

    Other modes can be added without breaking callers: any value
    other than ``"static_chunk"`` falls through to the planner path.
    """

    detector_temperature: float = 0.0
    """Sampling temperature for the detector role specifically.

    The bug_finder pipeline locks every other role to ``0.0`` for
    determinism (see ``tests/test_bug_finder_temperature_lockdown.py``).
    The detector is the one role where research workflows may want to
    sweep temperature — e.g. testing whether ``thinking-on, temp=1.0``
    on a reasoning model lifts recall without destroying precision.
    Defaults to ``0.0`` so the standard pipeline keeps its determinism
    guarantee; the lockdown test allows this site to read from config."""

    detector_enable_thinking: bool | None = None
    """Per-request ``enable_thinking`` chat-template kwarg forwarded to
    the detector subagent. None = let the model's default apply. True =
    explicitly enable reasoning ("thinking") on Qwen 3.x / GLM-4.x /
    EXAONE 4.x / Nemotron 3 Nano. False = explicitly disable."""

    detector_preserve_thinking: bool | None = None
    """When True, the model's ``<think>`` reasoning blocks are kept
    across multi-turn history during the detector inner loop. Qwen 3.6
    consumes this as ``preserve_thinking``; other templates ignore it.
    None = backend default."""

    detector_models: tuple[str, ...] = ()
    """Optional ensemble of detector models. When non-empty, the
    detector loop round-robins through these instead of running
    `role_models.detector` N times in a row. Findings flagged by 2+
    *families* (Claude + GPT, not Claude + Claude) get
    ``families_to_confirm >= 2`` on the report, which is a stronger
    confidence signal than raw run count because it breaks Anthropic's
    cited correlated-error pattern.

    Defaults to () so single-model setups behave exactly as before.
    The total detector invocations per chunk stays
    ``detector_runs_per_chunk`` — this list controls *which* model is
    used on each of those invocations, not the count.
    """

    enable_symbolic_gate: bool = True
    """Run Semgrep against accepted patches and reject fixes that
    introduce new symbolic findings.

    The disproof-oriented verifier already gates on "does the PoC
    pass + tests pass" — but thin test suites miss whole classes of
    regressions. Semgrep's static analysis catches injection /
    deserialization / weak-crypto patterns the test suite probably
    doesn't exercise. The gate is *advisory*: when Semgrep isn't
    installed in the workspace container, it skips gracefully rather
    than blocking the fix.

    Defaults to True — opt-out by setting False at run time when the
    workspace deliberately violates Semgrep's defaults (research code,
    intentional unsafe demos).
    """

    enable_comprehension: bool = True
    """Run the comprehender stage on first contact with a workspace.

    The comprehender walks the codebase once and persists a structural
    map (subsystems, pillars, risk surfaces, entry points) to
    ``bug_finder_codebase_knowledge``. Every subsequent run loads the
    map and injects it as system-prompt context for the planner,
    detector, and verifier — they share an understanding of the code
    instead of re-discovering it each run.

    Defaults to True. Opt out for single-shot scripted runs where the
    one-time comprehension cost (~5-30 minutes of LLM time on a large
    repo) is unwanted. The knowledge map persists across runs, so the
    cost is paid once per workspace (or once per explicit re-comprehend).
    """

    enable_check_writer: bool = True
    """Generate codebase-specific AST checks from comprehender pillars.

    After comprehension, for each load-bearing pillar the comprehender
    identified that isn't already covered by a custom check, the
    check-writer subagent synthesizes a stdlib AST check and persists
    it to ``<workspace>/.augmentum/bug_finder/custom_checks/``. Newly
    written checks fire immediately (their findings join THIS run's
    verify/fix pipeline) AND run for free on every subsequent audit via
    the agnostic substrate stage.

    This is the bug-finder "edits its own tests" loop — paying the LLM
    once to turn an architectural invariant into a permanent,
    deterministic, zero-token check. Requires ``enable_comprehension``
    (no pillars without it) and a host-mounted workspace root.

    Defaults to True. Opt out for one-shot runs where you don't want
    the workspace to accumulate generated checks.
    """

    max_check_writer_pillars: int = 6
    """Cap on how many new custom checks the check-writer generates per
    run. Bounds the one-time token spend; remaining uncovered pillars
    get picked up on later runs. Set 0 to disable generation while
    leaving ``enable_check_writer`` semantics intact."""

    enable_seeded_playbook: bool = True
    """Inject the static, class-specific hunting playbook (``playbook.py``)
    into the planner, TARGETED to the vuln classes this codebase's risk
    surfaces expose. Primes first-contact runs (when self-learned pattern
    memory is empty) with where-to-look / how-to-confirm / common-FP
    guidance. Additive, prompt-only, never reaches the detector or the
    verification core. Defaults to True; opt out for a pure latent-knowledge
    baseline."""

    max_playbook_classes: int = 4
    """Cap on how many playbook class-cards the planner gets — keeps the
    prompt focused on the codebase's actual surfaces rather than dumping
    the whole corpus."""

    enable_chunk_facts_precompute: bool = True
    """Pre-compute deterministic facts for each detector chunk
    (decorator chain on the enclosing function, prior workspace
    pattern memory at this file) and inject them into the detector's
    user message.

    Saves the LLM 2-3 tool-call round-trips per chunk on average —
    those facts are nearly always relevant and microseconds-cheap to
    compute. Defaults to True because the cost is local-AST work in
    exchange for token round-trips, a ~1000:1 cost win.

    Disable when comparing detector behavior with and without
    pre-compute (A/B measurement), or when the workspace lacks any
    substrate worth pre-computing.
    """

    enable_pen_test_leg: bool = False
    """Run the dynamic pen-test leg (boot under-test app + active HTTP
    probing) on confirmed findings before the fix stage.

    Off by default because the leg spawns a subprocess inside the
    workspace container — useful but heavier than the other legs and
    not always applicable (a logic-error in a helper has nothing to
    probe). When on, each confirmed finding gets a bounded pen_tester
    pass that returns ``confirmed`` / ``refuted`` / ``inconclusive``;
    the verdict gets stamped onto the finding's notes. ``refuted``
    findings are downgraded (severity capped at ``low``); ``confirmed``
    findings get a precision-prior bump for the fixer to consume.

    Lifecycle: a per-run ``_UnderTestRegistry`` tracks booted services
    so the orchestrator can tear them all down in a ``try/finally``,
    even on cancellation.
    """

    pen_test_boot_command: str = ""
    """Optional hint passed to the pen_tester for boot_under_test —
    when the orchestrator knows how to invoke the workspace's app
    (from baseline detection), it can save the pen_tester from
    needing to discover the command itself.

    Empty = let the pen_tester figure it out.
    """

    pen_test_boot_port: int = 0
    """Optional port hint for the under-test app. 0 = let the
    pen_tester pick."""

    pen_test_healthcheck_path: str = "/"
    """Healthcheck path the pen_tester uses to verify boot. Default
    ``/`` works for most apps; common alternatives: ``/healthz``,
    ``/readyz``."""

    enable_fuzz_leg: bool = True
    """Run the atheris fuzz leg in parallel with the LLM detector for
    chunks classified as fuzzable.

    Cross-modal confirmation: when the LLM detector AND the fuzz leg
    agree on a finding's site, ``families_to_confirm`` bumps by 1 —
    the spec's "gold-standard FP killer". Fuzz-only crashes (no LLM
    counterpart) also flow into the report as standalone findings.

    Atheris is installed lazily on first use (no image-build cost).
    When the install fails (offline / sandbox blocks apt / exotic
    platform), the leg skips gracefully and the run completes with the
    LLM detector only.
    """

    fuzz_max_seconds_per_chunk: int = 60
    """libfuzzer ``-max_total_time`` budget for each fuzzable chunk.

    60s is a reasonable smoke-test budget — long enough to surface
    shallow bugs (null derefs, IndexError on truncated input, infinite
    recursion on cyclic structures), short enough that 20 fuzzable
    chunks fit inside the default 10-minute total budget below. Raise
    for deeper hunts; lower to keep total run time under tight wallclock.
    """

    fuzz_max_total_seconds: float = 600.0

    deterministic_tools_root: str = ""
    """Host-side filesystem path the deterministic scanner tools
    (``list_routes``, ``security_check``, etc.) read from.

    When empty (default), the orchestrator detects whether the
    augmentum container's ``/app`` directory is a valid source tree
    (i.e. ``/app/augmentum/proxy`` exists) and uses that — this is
    the right behaviour for augmentum-on-augmentum audits where the
    workspace mirrors the host source.

    External codebases (cloned via the bench's ``--workspace-root``
    flag or the external test harness) override this explicitly.
    Set to ``"-"`` to disable deterministic tools entirely.
    """
    """Hard ceiling on total fuzz time per run, across all chunks.

    Once exceeded the orchestrator stops scheduling new fuzz sessions
    and proceeds to verify/fix with whatever already landed. Keeps a
    pathologically-many-chunks plan from blowing the overall wallclock
    on the fuzz leg alone.
    """


@dataclass
class CostLedgerEntry:
    """One row in the per-run cost ledger.

    Recorded once per subagent invocation. The orchestrator does not
    convert tokens to dollars — that's a UI concern; the ledger reports
    the raw inputs so any cost model can be applied downstream.
    """

    stage: str  # "planner" | "detector" | "verifier_repro" | "verifier_fix" | "fixer"
    role: str
    model: str
    instance_id: str
    iterations: int
    tokens_in: int
    tokens_out: int
    wallclock_ms: int
    stop_reason: str
    stuck_pattern: str | None = None
    stop_detail: str = ""
    """The subagent's stop_detail — the ERROR REASON when
    ``stop_reason="error"`` (backend exception text), the budget cap name
    when ``budget``, etc. Field data showed 396/399 detectors erroring
    with the reason recorded nowhere; capturing it here makes the
    dominant failure mode diagnosable from the run report alone."""


@dataclass
class BugFinderRunReport:
    """Final payload returned to the caller / job result."""

    run_id: str
    started_at: float
    completed_at: float
    intake: dict[str, Any]
    workspace_id: str
    baseline: WorkspaceBaseline
    findings: list[Finding]
    confirmation_hist: dict[str, int]
    cost_ledger: list[CostLedgerEntry]
    stop_reason: str  # "complete" | "cancelled" | "wallclock" | "error"
    stop_detail: str = ""
    # True when the verifier model equals the fixer model (single-model
    # self-verification). The local-hardware default; opted out by
    # setting a different per-workspace verifier model. Users read this
    # alongside ``confirmation_hist`` to assess trust in the findings.
    same_model_self_verification: bool = False
    notes: list[str] = field(default_factory=list)

    detector_health: dict[str, Any] = field(default_factory=dict)
    """Pipeline-integrity snapshot computed from the cost ledger by
    ``evaluate_detector_health``: how many detector subagents ran vs.
    errored, the error rate, and whether the run was degraded by it.
    The load-bearing trust signal — a run with ``degraded=True`` did NOT
    functionally scan, regardless of ``findings``. Empty for legacy runs
    persisted before this field existed."""


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------


def _last_json_payload(output: str) -> dict[str, Any] | None:
    """Salvage the last usable JSON object (truncation-tolerant — audit
    2026-06-17)."""
    return salvage_json_object(output)


def _threat_model_prefix_block(threat_model: str) -> str:
    """Render the user's threat model as a system-prompt prefix block.

    Returns ``""`` when no threat model is supplied so callers can pass
    the result through unchanged — the verifier spec builders accept an
    empty prefix as no-op.
    """
    threat = (threat_model or "").strip()
    if not threat:
        return ""
    return (
        "## Threat model (authoritative — detector and verifier consume "
        "the SAME document)\n\n"
        f"{threat}"
    )


def _prefix_threat_model(system_prompt: str, threat_model: str) -> str:
    """Prepend the threat-model block to a subagent system prompt.

    Anthropic's bug-finder research names mismatched threat models as the
    #1 cause of "valid finding but rejected by team" — when the model
    works from a different threat model than the maintainers, findings
    look like noise even when PoCs prove them. Surfacing the same
    authoritative document to detector + verifier closes that gap.

    No-op when ``threat_model`` is empty so existing call-sites that
    don't yet pass a threat model continue to work.
    """
    prefix = _threat_model_prefix_block(threat_model)
    if not prefix:
        return system_prompt
    return prefix + "\n\n---\n\n" + system_prompt


def _prefix_patterns(system_prompt: str, pattern_brief: str) -> str:
    """Prepend the cross-run pattern brief to the planner's system prompt.

    The brief is constructed by `patterns.render_pattern_brief` from
    the workspace's prior pattern rows. We surface it to the planner
    only — the detector should never see "this was flagged before"
    framing because that biases it toward confirming priors rather
    than disproving them.

    No-op when `pattern_brief` is empty so first-run-on-workspace flows
    are unaffected.
    """
    brief = (pattern_brief or "").strip()
    if not brief:
        return system_prompt
    return brief + "\n\n---\n\n" + system_prompt


@dataclass(frozen=True)
class _Chunk:
    file: str
    function: str
    line_start: int
    line_end: int
    rationale: str = ""
    suspected_class: str = ""


def _run_static_chunker(
    *,
    workspace_root: Path | None,
    focus_paths: tuple[str, ...],
    max_chunks: int,
    notes: list[str],
) -> list[_Chunk]:
    """Planner-bypass: walk the workspace via AST and emit one chunk
    per qualifying function. Returns shaped exactly like planner
    output so the detect stage runs unchanged downstream.

    Returns an empty list (with a note) when the workspace root isn't
    resolvable — the caller treats this the same as "planner produced
    no chunks" and exits cleanly.
    """
    if workspace_root is None:
        notes.append(
            "static_chunk mode requested but workspace root unresolved "
            "(deterministic_tools_root='-' or missing); no chunks produced",
        )
        return []
    static = collect_static_chunks(
        workspace_root,
        focus_paths=focus_paths,
        max_chunks=max_chunks,
    )
    notes.append(
        f"static_chunk mode: {len(static)} chunks "
        f"(focus_paths={list(focus_paths) or 'whole workspace'}, "
        f"cap={max_chunks})",
    )
    return [
        _Chunk(
            file=c.file,
            function=c.function,
            line_start=c.line_start,
            line_end=c.line_end,
            rationale=c.rationale,
            suspected_class=c.suspected_class,
        )
        for c in static
    ]


def _parse_planner_chunks(output: str, max_chunks: int) -> list[_Chunk]:
    """Extract the planner's chunk list. Returns at most ``max_chunks`` entries."""
    payload = _last_json_payload(output)
    if not payload or not isinstance(payload.get("chunks"), list):
        return []
    chunks: list[_Chunk] = []
    for raw in payload["chunks"][:max_chunks]:
        if not isinstance(raw, dict):
            continue
        try:
            file = str(raw.get("file") or "").strip()
            function = str(raw.get("function") or "<module>").strip() or "<module>"
            line_start = int(raw.get("line_start") or 0)
            line_end = int(raw.get("line_end") or 0)
        except (TypeError, ValueError):
            continue
        if not file or line_end < line_start:
            continue
        chunks.append(_Chunk(
            file=file,
            function=function,
            line_start=max(0, line_start),
            line_end=max(line_start, line_end),
            rationale=str(raw.get("rationale") or "").strip(),
            suspected_class=str(raw.get("suspected_class") or "").strip().lower(),
        ))
    return chunks


@dataclass(frozen=True)
class _FixerOutput:
    """Decoded final JSON block from the fixer."""

    invariant: str
    patch_summary: str
    files_changed: tuple[str, ...]
    repro_now_passes: bool
    self_test_summary: str


def _parse_fixer_output(output: str) -> _FixerOutput | None:
    """Extract the fixer's structured summary. Returns None on malformed output."""
    payload = _last_json_payload(output)
    if not payload:
        return None
    return _FixerOutput(
        invariant=str(payload.get("invariant") or "").strip(),
        patch_summary=str(payload.get("patch_summary") or "").strip(),
        files_changed=tuple(
            str(p).strip()
            for p in (payload.get("files_changed") or [])
            if str(p).strip()
        ),
        repro_now_passes=bool(payload.get("repro_now_passes")),
        self_test_summary=str(payload.get("self_test_summary") or "").strip(),
    )


# ---------------------------------------------------------------------------
# Container helpers (sequential-fixer snapshot model)
# ---------------------------------------------------------------------------


_SNAPSHOT_REF_PREFIX = "refs/augmentum/bug_finder/snapshot_"
_BASELINE_REF = "refs/augmentum/bug_finder/baseline"


async def _git_run(cm: ContainerManager, workspace_id: str, cmd: str, timeout: float = 60.0) -> str:
    """Run a shell command via the container manager. Bare wrapper so all
    git operations route through one place for easy logging."""
    return await cm.run_command(
        workspace_id,
        ["bash", "-c", f"cd /workspace && {cmd}"],
        timeout=timeout,
    )


async def _git_set_baseline(cm: ContainerManager, workspace_id: str) -> None:
    """Pin a baseline ref at the current HEAD so we can reset back to it
    between fix attempts."""
    await _git_run(
        cm, workspace_id,
        f"git update-ref {_BASELINE_REF} HEAD",
    )


async def _git_snapshot(cm: ContainerManager, workspace_id: str) -> str:
    """Capture the workspace's current state, return a snapshot id."""
    snapshot_id = uuid.uuid4().hex[:12]
    # Stage everything (including untracked) and commit to a snapshot ref.
    cmd = (
        "git add -A && "
        f"git commit -m 'bug-finder snapshot {snapshot_id}' --allow-empty --quiet && "
        f"git update-ref {_SNAPSHOT_REF_PREFIX}{snapshot_id} HEAD && "
        "git reset --soft HEAD~1"  # leave changes in the index (uncommitted)
    )
    try:
        await _git_run(cm, workspace_id, cmd, timeout=30.0)
    except Exception:  # noqa: BLE001
        log.warning(
            "bug_finder_snapshot_failed",
            workspace_id=workspace_id,
            snapshot_id=snapshot_id,
            exc_info=True,
        )
    return snapshot_id


async def _git_reset_to_baseline(
    cm: ContainerManager, workspace_id: str,
) -> None:
    """Reset workspace to the baseline ref. Use before each fix attempt."""
    try:
        await _git_run(
            cm, workspace_id,
            f"git reset --hard {_BASELINE_REF} && git clean -fdx -e '.augmentum/'",
            timeout=60.0,
        )
    except Exception:  # noqa: BLE001
        log.warning(
            "bug_finder_reset_failed",
            workspace_id=workspace_id,
            exc_info=True,
        )


async def _git_capture_patch(cm: ContainerManager, workspace_id: str) -> str:
    """Return the diff between baseline and HEAD-or-working-tree.

    The fixer's edits may live in the working tree, in the index, or as
    commits — capture all three with ``git diff baseline``."""
    try:
        return await _git_run(
            cm, workspace_id,
            f"git diff {_BASELINE_REF}",
            timeout=30.0,
        )
    except Exception:  # noqa: BLE001
        log.warning(
            "bug_finder_capture_patch_failed",
            workspace_id=workspace_id,
            exc_info=True,
        )
        return ""


async def _hash_baseline_and_head(
    cm: ContainerManager,
    workspace_id: str,
    path: str,
) -> tuple[str, str, bool, bool]:
    """Return ``(pre_hash, post_hash, pre_existed, post_existed)`` for
    ``path`` — pre is the SHA256 of the file at ``baseline``, post is
    the SHA256 of the file as currently on disk. Missing files yield
    empty hash + ``existed=False``.

    Used to write SWD-style receipts for each touched file after a fix
    attempt lands. Best-effort: container/shell hiccups return empty
    strings so receipts still record the attempt.
    """
    # Single shell pipeline — cheaper than two round-trips. Each call
    # returns ``<hash>:<exists>`` where exists is 1|0.
    cmd = (
        # baseline state
        f"(git show {_BASELINE_REF}:'{path}' 2>/dev/null | sha256sum | "
        "awk '{print $1\":1\"}') 2>/dev/null || echo ':0'; "
        # post-fix state
        f"(test -f '{path}' && sha256sum '{path}' | "
        "awk '{print $1\":1\"}') 2>/dev/null || echo ':0'"
    )
    try:
        raw = await _git_run(cm, workspace_id, cmd, timeout=15.0)
    except Exception:  # noqa: BLE001
        return "", "", False, False
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    pre_hash, pre_existed = "", False
    post_hash, post_existed = "", False
    if lines:
        h, _, ex = lines[0].partition(":")
        if ex == "1" and h:
            pre_hash = h
            pre_existed = True
    if len(lines) > 1:
        h, _, ex = lines[1].partition(":")
        if ex == "1" and h:
            post_hash = h
            post_existed = True
    return pre_hash, post_hash, pre_existed, post_existed


async def _emit_fix_receipts(
    *,
    cm: ContainerManager,
    workspace_id: str,
    finding: Finding,
    attempt: int,
    patch: str,
    fix_accepted: bool,
    rejection_reason: str,
    model_id: str,
    provider: str,
    run_id: str,
    substrate_host_root: Path | None,
) -> None:
    """Write one SWD-style ``Receipt`` per file touched by ``patch``.

    Each receipt is best-effort and isolated — receipt failures must
    never break a fix. The substrate lives on the host (not the
    container), so we only write when ``substrate_host_root`` is set.

    Receipt status mapping:
      * ``verified`` — fix-verifier accepted and pre/post hashes differ
      * ``drift``    — verifier accepted but pre/post identical (suspicious)
      * ``failed``   — verifier rejected; ``error`` carries the reason
    """
    if substrate_host_root is None:
        return
    if not patch.strip():
        return
    try:
        from augmentum.bug_finder.receipts import (
            Receipt,
            append_receipts,
        )
        from augmentum.bug_finder.swd import ActionIntent, ActionOp
        from augmentum.bug_finder.symbolic_gate import extract_patched_files
    except Exception:  # noqa: BLE001 — receipts module is optional
        return

    touched = extract_patched_files(patch) or []
    if not touched:
        return

    receipts: list[Receipt] = []
    for path in touched:
        try:
            pre_h, post_h, pre_ex, post_ex = await _hash_baseline_and_head(
                cm, workspace_id, path,
            )
        except Exception:  # noqa: BLE001
            pre_h = post_h = ""
            pre_ex = post_ex = False
        # Op classification from existence transitions.
        if not pre_ex and post_ex:
            op = ActionOp.CREATE.value
        elif pre_ex and not post_ex:
            op = ActionOp.DELETE.value
        else:
            op = ActionOp.MODIFY.value
        # Status:
        # * accepted + content changed → verified
        # * accepted + content identical → drift (suspicious — patch
        #   captured but file at HEAD is unchanged relative to baseline)
        # * rejected → failed
        if not fix_accepted:
            status = "failed"
            error = rejection_reason or "fix verifier rejected"
        elif pre_h and post_h and pre_h == post_h:
            status = "drift"
            error = "patch present but baseline=head hashes identical"
        else:
            status = "verified"
            error = ""
        receipts.append(Receipt(
            finding_id=finding.id,
            run_id=run_id,
            op=op,
            path=path,
            intent=ActionIntent.MUTATE.value,
            status=status,
            error=error,
            pre_hash=pre_h, post_hash=post_h,
            pre_existed=pre_ex, post_existed=post_ex,
            model_id=model_id, provider=provider,
            claim_signature=finding.claim_signature,
            reason=f"fix attempt {attempt}: {finding.claim[:160]}",
        ))
    try:
        append_receipts(substrate_host_root, receipts)
    except Exception:  # noqa: BLE001
        log.debug(
            "bug_finder_fix_receipts_append_failed",
            workspace=str(substrate_host_root), exc_info=True,
        )


# ---------------------------------------------------------------------------
# Workspace summary for the planner
# ---------------------------------------------------------------------------


async def _build_workspace_summary(
    cm: ContainerManager, workspace_id: str, baseline: WorkspaceBaseline,
) -> str:
    """One-shot summary the planner sees in its initial message.

    Cheap: just file count + top-level tree (depth 2) + the baseline
    notes. Anything deeper the planner discovers through its own tools.
    """
    parts: list[str] = []
    parts.append(f"language: {baseline.detected_language or 'unknown'}")
    if baseline.test_command:
        parts.append(f"test command: {baseline.test_command}")
    try:
        file_count = await _git_run(
            cm, workspace_id,
            "find . -type f -not -path './.git/*' | wc -l",
            timeout=15.0,
        )
        parts.append(f"file count: {file_count.strip() or 'unknown'}")
    except Exception as exc:  # noqa: BLE001 — best-effort enrichment; summary is still useful without it
        log.debug("bug_finder_file_count_probe_failed", workspace=workspace_id, error=str(exc))
    try:
        tree = await _git_run(
            cm, workspace_id,
            "find . -maxdepth 2 -type d -not -path './.git*' | head -40",
            timeout=15.0,
        )
        if tree.strip():
            parts.append("top-level directories:\n" + tree.strip())
    except Exception as exc:  # noqa: BLE001 — best-effort enrichment
        log.debug("bug_finder_tree_probe_failed", workspace=workspace_id, error=str(exc))
    if baseline.notes:
        parts.append("baseline notes: " + "; ".join(baseline.notes))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Subagent dispatch helpers
# ---------------------------------------------------------------------------


def _resolve_deterministic_root(override: str) -> Path | None:
    """Decide which host path the deterministic tools should scan.

    Resolution order:
      1. Explicit ``override`` (from ``BugFinderRunConfig``). ``"-"`` =
         disabled; non-empty = treat as a host path.
      2. ``/app`` if the augmentum container's source mount is present
         (detected via ``/app/augmentum/proxy``).
      3. ``None`` — deterministic tools are skipped this run.
    """
    override = (override or "").strip()
    if override == "-":
        return None
    if override:
        p = Path(override)
        if p.is_dir():
            return p
        log.warning(
            "bug_finder_deterministic_root_missing",
            override=override,
        )
        return None
    # Default: augmentum-on-augmentum case
    default = Path("/app")
    if (default / "augmentum" / "proxy").is_dir():
        return default
    return None


def _build_tools_for_role(
    cm: ContainerManager,
    workspace_id: str,
    state: CoderState,
    *,
    role: Role,
    user_id: str,
    workspace_root: Path | str | None = None,
    framework: str = "",
):
    """Filter ``create_coder_tools`` output down to the per-role
    allow-list, then mix in the deterministic substrate tools
    (``list_routes`` / ``security_check`` / etc.).

    ``workspace_root`` is the on-disk path the deterministic tools
    scan. When omitted, the deterministic tools are skipped — the
    agent falls back to LLM grepping (the pre-substrate behaviour).
    ``framework`` is the comprehender-detected framework name (or
    empty); defaults to ``"fastapi"`` inside the tool builder.
    """
    all_tools = create_coder_tools(
        cm, workspace_id, state, user_id=user_id,
    )
    allowed = {
        Role.PLANNER: PLANNER_TOOL_NAMES,
        Role.DETECTOR: DETECTOR_TOOL_NAMES,
        Role.VERIFIER: VERIFIER_TOOL_NAMES,
        Role.FIXER: FIXER_TOOL_NAMES,
    }[role]
    coder_tools = filter_tools(all_tools, allowed)
    # The fixer doesn't need the deterministic read-side tools — its
    # job is patching the code the verifier already confirmed. Every
    # other role gets the substrate.
    if role == Role.FIXER or workspace_root is None:
        return coder_tools
    try:
        from augmentum.bug_finder.agent_tools import (
            build_deterministic_tools,
        )
        det = build_deterministic_tools(
            root=Path(workspace_root), framework=framework or "fastapi",
        )
        return tuple(coder_tools) + det
    except Exception:  # noqa: BLE001 — substrate failure must not break the run
        log.warning(
            "bug_finder_deterministic_tools_disabled",
            workspace_id=workspace_id, exc_info=True,
        )
        return coder_tools


def _ledger_entry(
    stage: str,
    result: SubagentResult,
    model: str,
) -> CostLedgerEntry:
    return CostLedgerEntry(
        stage=stage,
        role=result.role,
        model=model,
        instance_id=result.instance_id,
        iterations=result.iterations,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        wallclock_ms=result.wallclock_ms,
        stop_reason=result.stop_reason,
        stuck_pattern=result.stuck_pattern,
        stop_detail=(result.stop_detail or "")[:300],
    )


def _ledger_event_payload(entry: CostLedgerEntry) -> dict[str, Any]:
    """Subset of the ledger row that the SSE consumer needs — keeps
    the event compact and stable across schema growth."""
    return {
        "stage": entry.stage,
        "role": entry.role,
        "model": entry.model,
        "iterations": entry.iterations,
        "tokens_in": entry.tokens_in,
        "tokens_out": entry.tokens_out,
        "wallclock_ms": entry.wallclock_ms,
        "stop_reason": entry.stop_reason,
        "stuck_pattern": entry.stuck_pattern,
    }


# Map fine-grained ledger stages onto the UI's burndown buckets so the
# dashboard's per-stage bars stay stable as internal roles evolve.
_COST_BUCKET = {
    "planner": "planner",
    "detector": "detector",
    "verifier_repro": "verifier",
    "verifier_fix": "verifier",
    "fixer": "fixer",
    "lead": "lead",
    "comprehender": "comprehend",
    "check_writer": "comprehend",
    "pen_tester": "pentest",
}


def _cost_summary(ledger: list[CostLedgerEntry]) -> dict[str, Any]:
    """Cumulative token spend, bucketed by stage, for the live burndown.

    Computed from the authoritative cost ledger (not the per-subagent
    progress snapshots, which can't be summed across concurrent
    same-role detectors). Emitted as a ``cost`` SSE event at each stage
    boundary so the UI can render where the budget is going in real time.
    """
    by_stage: dict[str, dict[str, int]] = {}
    total_in = total_out = total_ms = 0
    for e in ledger:
        bucket = _COST_BUCKET.get(e.stage, e.stage)
        slot = by_stage.setdefault(
            bucket, {"tokens_in": 0, "tokens_out": 0, "wallclock_ms": 0},
        )
        slot["tokens_in"] += e.tokens_in
        slot["tokens_out"] += e.tokens_out
        slot["wallclock_ms"] += e.wallclock_ms
        total_in += e.tokens_in
        total_out += e.tokens_out
        total_ms += e.wallclock_ms
    return {
        "by_stage": by_stage,
        "total_in": total_in,
        "total_out": total_out,
        "total_ms": total_ms,
        "entries": len(ledger),
    }


# Verifier stages share one health bucket — both confirm-or-refute
# findings, and an errored verifier is as untrustworthy as an errored
# detector (it can't have actually judged the finding).
_VERIFIER_STAGES = ("verifier_repro", "verifier_fix")


def evaluate_detector_health(
    ledger: list[CostLedgerEntry], *, threshold: float = 0.5,
) -> dict[str, Any]:
    """Pipeline-integrity check from the cost ledger.

    A bug_finder run that reports ``complete / no findings`` is only
    trustworthy if the detector stage actually RAN. When the high-
    concurrency detector fan-out collapses under provider rate-limits,
    most detector subagents stop with ``stop_reason="error"`` at
    iteration 0 — yet the pipeline still reports "no findings", which
    reads to the user as "your code is clean." It isn't; the scan never
    happened. This computes the detector error rate so ``_build_report``
    can rewrite that lie into an honest ``degraded`` verdict.

    ``stop_reason="error"`` is an unambiguous infrastructure failure
    (backend raised) — distinct from ``budget`` (ran, hit a cap) or
    ``stuck`` (ran, looped). Only ``error`` counts against health.

    Returns a flat dict (serialized verbatim into ``report_json`` /
    surfaced to the UI). ``degraded`` is True iff detectors actually ran
    AND the error rate met the threshold. Verifier error counts ride
    along for visibility but don't (yet) trip the gate — a refused
    finding is a legitimate verdict, only an *errored* verifier is lost
    signal, and the detector gate already catches the dominant case.
    """
    det = [e for e in ledger if e.stage == "detector"]
    ver = [e for e in ledger if e.stage in _VERIFIER_STAGES]
    total = len(det)
    errored = sum(1 for e in det if e.stop_reason == "error")
    budget = sum(1 for e in det if e.stop_reason == "budget")
    rate = (errored / total) if total else 0.0
    ver_total = len(ver)
    ver_errored = sum(1 for e in ver if e.stop_reason == "error")
    ver_rate = (ver_errored / ver_total) if ver_total else 0.0

    # ``degraded`` is the HARD detector gate — it rewrites stop_reason to
    # "degraded" (see _build_report), so it stays detector-only to avoid
    # over-tripping on legitimate verifier refutations.
    detector_degraded = total > 0 and rate >= threshold
    # ``trustworthy`` is the SOFTER, broader verdict surfaced to the user:
    # a run is untrustworthy if detectors collapsed OR enough verifiers
    # errored that the confirm stage couldn't have functionally run. The
    # verifier leg needs a minimum sample (4) so a single errored verifier
    # on a tiny finding set doesn't flip the whole run's trust flag.
    verifier_degraded = ver_total >= 4 and ver_rate >= threshold
    return {
        "detectors_total": total,
        "detectors_errored": errored,
        "detectors_budget": budget,
        "detector_error_rate": round(rate, 3),
        "verifiers_total": ver_total,
        "verifiers_errored": ver_errored,
        "verifier_error_rate": round(ver_rate, 3),
        "threshold": threshold,
        "degraded": detector_degraded,
        "trustworthy": not detector_degraded and not verifier_degraded,
    }


# ---------------------------------------------------------------------------
# Stage drivers
# ---------------------------------------------------------------------------


async def _ensure_knowledge(
    *,
    config: BugFinderRunConfig,
    cm: ContainerManager,
    workspace: PreparedWorkspace,
    resolve_backend: BackendResolver,
    user_id: str,
    ledger: list[CostLedgerEntry],
    knowledge_store: Any,
    notes: list[str],
    progress_callback=None,
) -> str:
    """Return the rendered knowledge brief for this workspace.

    Loads from the store when populated; otherwise runs the
    comprehender to build the map, persists it, and returns the
    rendered brief. Errors during comprehension are logged + reported
    via ``notes`` — the orchestrator continues to planning without a
    brief (degraded but functional).

    Returns ``""`` when comprehension is disabled, the store is
    unavailable, or the comprehender produced no parseable output.
    """
    if not config.enable_comprehension or knowledge_store is None:
        return ""

    from augmentum.bug_finder.comprehender import (
        DEFAULT_COMPREHENDER_BUDGET,
        run_comprehender,
    )
    from augmentum.bug_finder.knowledge_store import render_knowledge_brief
    from augmentum.bug_finder.subagent import COMPREHENDER_TOOL_NAMES

    workspace_id = workspace.workspace_id
    try:
        knowledge = await knowledge_store.get(
            user_id=user_id, workspace_id=workspace_id,
        )
    except Exception as exc:  # noqa: BLE001 — knowledge is advisory
        log.warning(
            "bug_finder_knowledge_load_failed",
            workspace_id=workspace_id, error=str(exc),
        )
        return ""

    if knowledge.is_populated:
        brief = render_knowledge_brief(knowledge)
        log.info(
            "bug_finder_knowledge_reused",
            workspace_id=workspace_id,
            age_seconds=knowledge.age_seconds,
            refresh_count=knowledge.refresh_count,
            chars=len(brief),
        )
        return brief

    # First-contact run. Two phases:
    #   Phase 1: deterministic skeleton (container-side commands) —
    #            seconds, zero tokens. Produces subsystems + routes
    #            + entry-point catalog from file tree + grep + regex.
    #   Phase 2: LLM synthesizer reads the skeleton + samples 5-10
    #            specific files to identify pillars + risk_surfaces +
    #            write the brief. ~30-80k tokens.
    # Even if phase 2 fails, we persist phase 1's results so subsequent
    # runs still benefit from the structural map.
    from augmentum.bug_finder.comprehension_skeleton import build_skeleton
    from augmentum.bug_finder.knowledge_store import (
        EntryPoint,
        Subsystem,
    )

    notes.append("comprehension: first-contact run, building knowledge map")

    skeleton = await build_skeleton(
        cm=cm, workspace_id=workspace_id, root="/workspace",
    )
    log.info(
        "bug_finder_skeleton_complete",
        workspace_id=workspace_id,
        languages=list(skeleton.languages),
        framework=skeleton.framework,
        subsystems=len(skeleton.subsystems),
        routes=len(skeleton.routes),
    )

    # Convert the deterministic shadow into store dataclasses so we
    # can persist the map even when phase 2 fails.
    skeleton_subsystems = tuple(
        Subsystem(
            name=s.path.rsplit("/", 1)[-1] or s.path,
            purpose=(s.top_docstring or "").split("\n", 1)[0][:140],
            paths=(s.path,),
            size_files=s.file_count,
            pillars=(),
        )
        for s in skeleton.subsystems
    )
    skeleton_entry_points = tuple(
        EntryPoint(
            kind="http",
            path=f"{r.method} {r.path}",
            handler=f"{r.file}:{r.line}",
        )
        for r in skeleton.routes
    )

    state = CoderState(
        session_id="bf_comprehender", workspace_id=workspace_id,
    )
    all_tools = _build_tools_for_role(
        cm, workspace_id, state, role=Role.PLANNER, user_id=user_id,
    )
    tools = filter_tools(all_tools, COMPREHENDER_TOOL_NAMES)
    backend, clean_model = await resolve_backend(
        config.role_models.for_role(Role.COMPREHENDER),
    )
    user_goal_block = config.intake.user_goal.to_prompt_block()
    threat_block = _threat_model_prefix_block(config.intake.threat_model)
    skeleton_block = skeleton.render_for_prompt()

    result = None
    try:
        result = await run_comprehender(
            model=clean_model,
            backend=backend,
            tools=tools,
            skeleton_block=skeleton_block,
            user_goal_block=user_goal_block,
            threat_model_block=threat_block,
            budget=DEFAULT_COMPREHENDER_BUDGET,
            progress_callback=progress_callback,
        )
        ledger.append(_ledger_entry(
            "comprehender", result.subagent_result,
            config.role_models.for_role(Role.COMPREHENDER),
        ))
    except Exception as exc:  # noqa: BLE001 — synthesis is best-effort
        log.warning(
            "bug_finder_comprehender_error",
            workspace_id=workspace_id, error=str(exc),
        )
        notes.append(
            f"comprehension synthesis failed: {type(exc).__name__} — "
            "persisting skeleton-only map",
        )

    # Decide what to persist:
    #   * Best case: comprehender produced a parseable JSON — use its
    #     output (richer brief + pillars + risk_surfaces).
    #   * Fallback: persist the skeleton-only map so subsequent runs
    #     at least get subsystems + entry_points pre-rendered.
    persistence_kwargs: dict[str, Any] = {
        "user_id": user_id,
        "workspace_id": workspace_id,
        "brief": "",
        "subsystems": skeleton_subsystems,
        "pillars": (),
        "risk_surfaces": (),
        "entry_points": skeleton_entry_points,
        "commit_sha": skeleton.head_sha,
        "tokens_in": 0,
        "tokens_out": 0,
        "wallclock_seconds": 0.0,
    }
    if result is not None and result.succeeded:
        # Comprehender's output wins. Fall back to skeleton subsystems
        # if the LLM somehow returned an empty list (defensive).
        persistence_kwargs["brief"] = result.output.brief
        persistence_kwargs["subsystems"] = (
            result.output.subsystems or skeleton_subsystems
        )
        persistence_kwargs["pillars"] = result.output.pillars
        persistence_kwargs["risk_surfaces"] = result.output.risk_surfaces
        persistence_kwargs["entry_points"] = (
            result.output.entry_points or skeleton_entry_points
        )
        persistence_kwargs["tokens_in"] = result.subagent_result.tokens_in
        persistence_kwargs["tokens_out"] = result.subagent_result.tokens_out
        persistence_kwargs["wallclock_seconds"] = result.runtime_seconds
    elif result is not None:
        notes.append(
            "comprehender produced no parseable JSON — "
            "persisting skeleton-only map (subsystems + routes)",
        )

    # ledger entry already covers the structured cost; the brief-len
    # below is the "did we actually learn anything" signal for the UI.

    # Persist (either rich comprehender output OR skeleton-only
    # fallback) so subsequent runs benefit.
    try:
        await knowledge_store.upsert(**persistence_kwargs)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "bug_finder_knowledge_persist_failed",
            workspace_id=workspace_id, error=str(exc),
        )
        # Persistence failure shouldn't block this run — we still
        # have the brief in-memory to render.

    # Re-fetch + render so the brief carries refresh metadata + the
    # uniform render path (skeleton-only knowledge falls back to the
    # structured table rendering).
    refreshed = await knowledge_store.get(
        user_id=user_id, workspace_id=workspace_id,
    )
    return render_knowledge_brief(refreshed)


async def _run_check_writer_stage(
    *,
    config: BugFinderRunConfig,
    cm: ContainerManager,
    workspace: PreparedWorkspace,
    deterministic_root: Path | None,
    resolve_backend: BackendResolver,
    user_id: str,
    ledger: list[CostLedgerEntry],
    knowledge_store: Any,
    notes: list[str],
    event_emit,
    run_id: str,
    progress_callback=None,
) -> list[Finding]:
    """Generate codebase-specific AST checks from comprehender pillars.

    For each load-bearing pillar that isn't already covered by a custom
    check, ask the check-writer subagent to synthesize a stdlib AST
    check and persist it to the workspace's ``custom_checks/`` dir.
    Newly written checks are FIRED immediately — their findings are
    returned for the caller to fold into ``scanner_seeded`` so they
    flow through THIS run's verify/fix path — and they also run for
    free on every subsequent audit via the agnostic substrate stage.

    This is the bug-finder's "edits its own tests" loop: one-time LLM
    spend converts a comprehender invariant into a permanent,
    deterministic, zero-token check.

    Best-effort throughout — any failure is logged + noted and returns
    ``[]`` so the run continues. Requires structured pillars (so
    ``enable_comprehension``) and a host-mounted ``deterministic_root``
    (where checks are written + executed).
    """
    if (
        not config.enable_check_writer
        or config.max_check_writer_pillars <= 0
        or knowledge_store is None
        or deterministic_root is None
    ):
        return []

    from augmentum.bug_finder.agnostic_stage import run_custom_checks_stage
    from augmentum.bug_finder.check_writer import (
        DEFAULT_CHECK_WRITER_BUDGET,
        run_check_writer,
        slug_for_pillar,
    )
    from augmentum.bug_finder.custom_check_runner import list_custom_checks
    from augmentum.bug_finder.subagent import COMPREHENDER_TOOL_NAMES

    workspace_id = workspace.workspace_id
    try:
        knowledge = await knowledge_store.get(
            user_id=user_id, workspace_id=workspace_id,
        )
    except Exception as exc:  # noqa: BLE001 — check-writing is advisory
        log.warning(
            "bug_finder_check_writer_knowledge_load_failed",
            workspace_id=workspace_id, error=str(exc),
        )
        return []

    pillars = [
        p for p in knowledge.pillars
        if p.name.strip() and p.statement.strip()
    ]
    if not pillars:
        return []

    # Skip pillars that already have a check (by deterministic slug) so
    # we never regenerate or overwrite a working check.
    existing = {p.stem for p in list_custom_checks(deterministic_root)}
    uncovered = [
        p for p in pillars if slug_for_pillar(p.name) not in existing
    ][: config.max_check_writer_pillars]
    if not uncovered:
        return []

    event_emit("stage", {
        "run_id": run_id, "stage": "writing_checks", "progress": 0.135,
        "note": (
            f"writing {len(uncovered)} codebase-specific check(s) "
            "from comprehender pillars"
        ),
    })

    state = CoderState(
        session_id="bf_check_writer", workspace_id=workspace_id,
    )
    all_tools = _build_tools_for_role(
        cm, workspace_id, state, role=Role.PLANNER, user_id=user_id,
    )
    tools = filter_tools(all_tools, COMPREHENDER_TOOL_NAMES)
    model_id = config.role_models.for_role(Role.PLANNER)
    backend, clean_model = await resolve_backend(model_id)

    written: list[str] = []
    for pillar in uncovered:
        slug = slug_for_pillar(pillar.name)
        try:
            res = await run_check_writer(
                workspace_root=deterministic_root,
                pillar_name=pillar.name,
                pillar_statement=pillar.statement,
                evidence=tuple(pillar.evidence),
                model=clean_model,
                backend=backend,
                tools=tools,
                budget=DEFAULT_CHECK_WRITER_BUDGET,
                progress_callback=progress_callback,
            )
        except Exception as exc:  # noqa: BLE001 — one bad pillar mustn't sink the rest
            log.warning(
                "bug_finder_check_writer_error",
                pillar=pillar.name, error=str(exc),
            )
            event_emit("check_written", {
                "run_id": run_id, "pillar": pillar.name, "check": "",
                "valid": False,
                "reason": f"{type(exc).__name__}: {exc}"[:200],
            })
            continue

        ledger.append(_ledger_entry(
            "check_writer", res.subagent_result, clean_model,
        ))
        if res.valid and res.written_to is not None:
            written.append(slug)
        event_emit("check_written", {
            "run_id": run_id,
            "pillar": pillar.name,
            "valid": bool(res.valid),
            "check": slug if res.valid else "",
            "source_lines": (res.source.count("\n") + 1) if res.valid else 0,
            "reason": res.skip_reason or "",
        })

    if not written:
        event_emit("check_writer_complete", {
            "run_id": run_id, "written": 0, "seeded": 0,
        })
        notes.append("check-writer: no new checks persisted this run")
        return []

    notes.append(
        f"check-writer: generated {len(written)} new custom check(s) "
        "from pillars (permanent — run free on every future audit)",
    )

    # Fire the freshly written checks now so their findings join THIS
    # run's pipeline. Re-uses the agnostic stage's suppression +
    # conversion + pattern machinery, scoped to just the new checks so
    # we don't re-pay the Bandit/Ruff sweep.
    seeded: list[Finding] = []
    try:
        result = run_custom_checks_stage(
            deterministic_root, only_checks=frozenset(written),
        )
        seeded = list(result.seeded_findings)
    except Exception as exc:  # noqa: BLE001 — same-run fire is best-effort
        log.warning(
            "bug_finder_check_writer_fire_failed",
            error=str(exc), exc_info=True,
        )

    for f in seeded:
        event_emit("finding_landed", {
            "run_id": run_id, "finding_id": f.id,
            "file": f.file, "function": f.function, "severity": f.severity,
            "claim_signature": f.claim_signature,
            "claim": (f.claim or "")[:240], "status": f.status,
            "runs_to_confirm": f.runs_to_confirm, "total_runs": f.total_runs,
            "families_to_confirm": f.families_to_confirm,
            "total_families": f.total_families,
        })

    event_emit("check_writer_complete", {
        "run_id": run_id, "written": len(written), "seeded": len(seeded),
    })
    return seeded


async def _run_planner(
    config: BugFinderRunConfig,
    *,
    cm: ContainerManager,
    workspace: PreparedWorkspace,
    resolve_backend: BackendResolver,
    user_id: str,
    ledger: list[CostLedgerEntry],
    knowledge_brief: str = "",
    workspace_priors_brief: str = "",
    playbook_brief: str = "",
    progress_callback=None,
    task_queue: Any | None = None,
    run_id: str = "",
) -> list[_Chunk]:
    state = CoderState(session_id="bf_planner", workspace_id=workspace.workspace_id)
    tools = _build_tools_for_role(
        cm, workspace.workspace_id, state, role=Role.PLANNER, user_id=user_id,
    )
    workspace_summary = await _build_workspace_summary(
        cm, workspace.workspace_id, workspace.baseline,
    )
    focus = list(config.intake.focus_paths) or ["(none — survey whole repo)"]
    user_msg = PLANNER_USER_TEMPLATE.format(
        max_chunks=config.max_chunks,
        workspace_summary=workspace_summary,
        focus_paths="\n".join(f"  - {p}" for p in focus),
    )
    backend, clean_model = await resolve_backend(config.role_models.planner)
    # System-prompt prefix order: knowledge → user goal → threat model →
    # patterns → planner spec. Knowledge first because it's structural
    # context; everything else is "what to do with that knowledge".
    system_prompt = PLANNER_SYSTEM_PROMPT.format(max_chunks=config.max_chunks)
    system_prompt = _prefix_patterns(system_prompt, config.intake.prior_patterns)
    # Workspace-substrate priors land just below caller-supplied
    # ``prior_patterns`` so they share the same framing layer in the
    # prompt. Empty brief is a no-op.
    if workspace_priors_brief:
        system_prompt = (
            workspace_priors_brief + "\n\n---\n\n" + system_prompt
        )
    # Seeded class playbook sits in the same priors layer — targeted
    # where-to-look guidance for first-contact runs. No-op when empty.
    if playbook_brief:
        system_prompt = playbook_brief + "\n\n---\n\n" + system_prompt
    user_goal_block = config.intake.user_goal.to_prompt_block()
    if user_goal_block:
        system_prompt = user_goal_block + "\n\n---\n\n" + system_prompt
    if knowledge_brief:
        system_prompt = knowledge_brief + "\n\n---\n\n" + system_prompt
    # temperature=0.0 — Every bug_finder SubagentSpec uses temperature 0
    # so the pipeline is reproducible. Sub-agents under the bug_finder
    # are audit infrastructure, not creative writing: same input + same
    # config should produce the same output. The default ``None`` lets
    # the API fall back to ~1.0, which empirically produced 0/1/2/3
    # findings on identical vuln_app runs across attempts. See the
    # determinism audit for rationale.
    spec = SubagentSpec(
        role=Role.PLANNER.value,
        model=clean_model,
        system_prompt=system_prompt,
        initial_user_message=user_msg,
        tools=tools,
        budget=config.planner_budget,
        tool_guard=planner_guard,
        instance_id="planner",
        progress_callback=progress_callback,
        temperature=0.0,
    )
    result = await run_subagent(spec, backend=backend)
    ledger.append(_ledger_entry("planner", result, config.role_models.planner))
    chunks = _parse_planner_chunks(result.output, config.max_chunks)
    log.info(
        "bug_finder_planner_done",
        workspace_id=workspace.workspace_id,
        stop_reason=result.stop_reason,
        chunks=len(chunks),
    )

    # Enqueue each chunk as a DETECT task. Idempotent — re-runs of
    # the same planner output collapse to the same task ids. The
    # detect stage drains from the queue; the lead agent (named-bug
    # mode) reads from the same queue and decides what to dispatch.
    if task_queue is not None and run_id and chunks:
        from augmentum.bug_finder.task_queue import TaskKind
        for chunk in chunks:
            try:
                await task_queue.enqueue(
                    run_id=run_id, user_id=user_id,
                    workspace_id=workspace.workspace_id,
                    kind=TaskKind.DETECT,
                    target={
                        "file": chunk.file,
                        "function": chunk.function,
                        "line_start": chunk.line_start,
                        "line_end": chunk.line_end,
                        "rationale": chunk.rationale,
                        "suspected_class": chunk.suspected_class,
                    },
                    reason=chunk.rationale or "planner-selected",
                    priority=5,
                    created_by="planner",
                )
            except Exception as exc:  # noqa: BLE001 — queue is advisory
                log.warning(
                    "bug_finder_task_enqueue_failed",
                    workspace_id=workspace.workspace_id,
                    file=chunk.file, function=chunk.function,
                    error=str(exc),
                )

    return chunks


def _detector_model_for_run(config: BugFinderRunConfig, run_index: int) -> str:
    """Choose which detector model to use for one of the N runs of one
    chunk. Round-robins through `detector_models` when the ensemble is
    configured; otherwise sticks with `role_models.detector`."""
    if not config.detector_models:
        return config.role_models.detector
    return config.detector_models[run_index % len(config.detector_models)]


async def _run_detector_for_chunk(
    chunk: _Chunk,
    *,
    run_index: int,
    config: BugFinderRunConfig,
    cm: ContainerManager,
    workspace: PreparedWorkspace,
    resolve_backend: BackendResolver,
    user_id: str,
    ledger: list[CostLedgerEntry],
    semaphore: asyncio.Semaphore,
    progress_callback=None,
    deterministic_root: Path | None = None,
    breaker: DetectorCircuitBreaker | None = None,
) -> list[Finding]:
    async with semaphore:
        # Circuit breaker: once the backend has proven down (error rate
        # over threshold past the min sample), queued detectors bail here
        # — after acquiring the slot but before any expensive setup — so a
        # systemic outage stops burning budget instead of grinding every
        # remaining chunk to a guaranteed error.
        if breaker is not None and breaker.is_open:
            return []
        state = CoderState(
            session_id=f"bf_detector_r{run_index}", workspace_id=workspace.workspace_id,
        )
        tools = _build_tools_for_role(
            cm, workspace.workspace_id, state, role=Role.DETECTOR, user_id=user_id,
            workspace_root=deterministic_root,
        )
        # Pre-compute deterministic facts (decorators + prior workspace
        # patterns at this file) so the LLM doesn't burn round-trips
        # calling tools to learn things we can answer from disk in
        # microseconds. The detector still has the tools available
        # if it wants to dig deeper.
        precomputed_facts_block = ""
        if config.enable_chunk_facts_precompute and deterministic_root is not None:
            try:
                from augmentum.bug_finder.chunk_facts import (
                    compute_chunk_facts,
                    render_chunk_facts,
                )
                facts = compute_chunk_facts(
                    workspace_root=deterministic_root,
                    file=chunk.file,
                    line_start=chunk.line_start,
                )
                precomputed_facts_block = render_chunk_facts(facts)
            except Exception as exc:  # noqa: BLE001 — best-effort
                log.debug(
                    "bug_finder_chunk_facts_precompute_failed",
                    file=chunk.file, error=str(exc),
                )
        user_msg = DETECTOR_USER_TEMPLATE.format(
            file=chunk.file,
            function=chunk.function,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            rationale=chunk.rationale or "(no rationale supplied)",
            suspected_class=chunk.suspected_class or "(unspecified)",
            precomputed_facts_block=precomputed_facts_block,
        )
        model_for_run = _detector_model_for_run(config, run_index)
        backend, clean_model = await resolve_backend(model_for_run)
        # Thinking-enabled models eat 2-3× more tokens per inner-loop
        # iteration because the reasoning trace is generated before any
        # tool call. The default 240s/120K-token detector budget was
        # tuned for non-thinking models — empirically, thinking-on runs
        # exhaust budget on ~80% of chunks at default. Scale the budget
        # ~3× when thinking is on so the model has room to actually
        # produce findings instead of timing out mid-reasoning.
        if config.detector_enable_thinking:
            detector_budget = SubagentBudget(
                max_iterations=int(config.detector_budget.max_iterations * 1.7),
                max_wallclock_seconds=int(
                    config.detector_budget.max_wallclock_seconds * 3.0,
                ),
                max_tokens=int(config.detector_budget.max_tokens * 2.5),
            )
        else:
            detector_budget = config.detector_budget
        spec = SubagentSpec(
            role=Role.DETECTOR.value,
            model=clean_model,
            system_prompt=_prefix_threat_model(
                DETECTOR_SYSTEM_PROMPT,
                config.intake.threat_model,
            ),
            initial_user_message=user_msg,
            tools=tools,
            budget=detector_budget,
            tool_guard=detector_guard,
            instance_id=f"detector_{chunk.file}_{chunk.function}_r{run_index}"[:80],
            progress_callback=progress_callback,
            temperature=config.detector_temperature,
            enable_thinking=config.detector_enable_thinking,
            preserve_thinking=config.detector_preserve_thinking,
        )
        # Bounded retry on transient backend errors so an isolated 429/5xx
        # doesn't permanently lose this chunk. A permanent error (model
        # unavailable) just exhausts retries and returns; the breaker
        # handles the systemic case once enough detectors have recorded.
        if config.detector_max_retries > 0:
            result = await run_with_retry(
                lambda: run_subagent(spec, backend=backend),
                max_retries=config.detector_max_retries,
                base_delay_s=config.detector_retry_base_delay_s,
            )
        else:
            result = await run_subagent(spec, backend=backend)
        if breaker is not None:
            breaker.record(result.stop_reason)
        ledger.append(_ledger_entry("detector", result, model_for_run))
        return parse_detector_output(result.output)


async def _run_detect_stage(
    chunks: list[_Chunk],
    *,
    config: BugFinderRunConfig,
    cm: ContainerManager,
    workspace: PreparedWorkspace,
    resolve_backend: BackendResolver,
    user_id: str,
    ledger: list[CostLedgerEntry],
    event_emit: Callable[[str, dict[str, Any], bool], None] | None = None,
    run_id: str = "",
    progress_callback=None,
    task_queue: Any | None = None,
    deterministic_root: Path | None = None,
) -> list[Finding]:
    """Run all detectors, fan out via asyncio.gather, dedupe across runs.

    ``event_emit`` (optional) lets the orchestrator surface per-chunk
    progress to the SSE stream so the live dashboard can show which
    chunk is currently being examined. The signature matches the
    ``_emit`` helper in ``run_bug_finder``.
    """
    def _e(kind: str, payload: dict[str, Any]) -> None:
        if event_emit is None:
            return
        payload = {"run_id": run_id, **payload}
        event_emit(kind, payload, terminal=False)

    if not chunks:
        return []
    semaphore = asyncio.Semaphore(max(1, config.detector_concurrency))

    # Shared circuit breaker across every detector this stage spawns. Fed
    # one outcome per detector; once it opens, queued detectors bail
    # cheaply (see ``_run_detector_for_chunk``) so a down backend can't
    # burn the whole budget on guaranteed errors.
    breaker = DetectorCircuitBreaker(
        min_samples=config.detector_circuit_breaker_min_samples,
        error_rate_threshold=config.detector_circuit_breaker_error_rate,
    )

    # Schedule N runs per chunk. The list of tasks is grouped so we can
    # dedupe by chunk first, then aggregate across the N runs.
    per_chunk_run_lists: dict[tuple[str, str, int, int], list[asyncio.Task]] = {}
    for chunk in chunks:
        key = (chunk.file, chunk.function, chunk.line_start, chunk.line_end)
        tasks: list[asyncio.Task] = []
        for r in range(config.detector_runs_per_chunk):
            task = asyncio.create_task(
                _run_detector_for_chunk(
                    chunk,
                    run_index=r,
                    config=config,
                    cm=cm,
                    workspace=workspace,
                    resolve_backend=resolve_backend,
                    user_id=user_id,
                    ledger=ledger,
                    semaphore=semaphore,
                    deterministic_root=deterministic_root,
                    progress_callback=progress_callback,
                    breaker=breaker,
                ),
            )
            tasks.append(task)
        per_chunk_run_lists[key] = tasks

    # Per-run family labels — parallel to runs_results, drives the
    # families_to_confirm count on each Finding. Computed once because
    # round-robin assignment is identical across all chunks.
    from augmentum.bug_finder.role_models import family_for_model
    per_run_models = [
        _detector_model_for_run(config, r)
        for r in range(config.detector_runs_per_chunk)
    ]
    per_run_families = [family_for_model(m) for m in per_run_models]

    # Resolve queue task ids per chunk so we can mark each one as
    # completed when its N detector runs land. Built lazily; if the
    # queue isn't wired or planner didn't enqueue, the marks become
    # no-ops.
    from augmentum.bug_finder.task_queue import TaskKind
    from augmentum.bug_finder.task_queue import _task_id as _build_task_id

    def _task_id_for_chunk(c: _Chunk) -> str:
        target = {
            "file": c.file, "function": c.function,
            "line_start": c.line_start, "line_end": c.line_end,
            "rationale": c.rationale, "suspected_class": c.suspected_class,
        }
        return _build_task_id(run_id, TaskKind.DETECT.value, target)

    chunk_lookup = {
        (c.file, c.function, c.line_start, c.line_end): c
        for c in chunks
    }

    # Await + dedupe per chunk.
    all_findings: list[Finding] = []
    chunks_done = 0
    circuit_open_emitted = False
    for key, tasks in per_chunk_run_lists.items():
        # Surface the breaker the first time it opens so the dashboard
        # shows "detector backend down — bailing early" instead of a
        # silent stall through the remaining (now short-circuited) chunks.
        if breaker.is_open and not circuit_open_emitted:
            circuit_open_emitted = True
            _e("detector_circuit_open", breaker.snapshot())
        file_, function_, *_ = key
        chunk_obj = chunk_lookup.get(key)
        _e("chunk_detect_started", {
            "file": file_, "function": function_,
            "chunks_total": len(chunks),
            "chunks_done": chunks_done,
        })
        runs_results: list[list[Finding]] = []
        failed = False
        for t in tasks:
            try:
                runs_results.append(await t)
            except Exception:  # noqa: BLE001
                log.warning("bug_finder_detector_task_failed", exc_info=True)
                runs_results.append([])
                failed = True
        merged = merge_runs(runs_results, families=per_run_families)
        chunks_done += 1
        _e("chunk_detect_complete", {
            "file": file_, "function": function_,
            "chunks_total": len(chunks),
            "chunks_done": chunks_done,
            "findings_from_chunk": len(merged),
        })
        # Live token burndown — emit cumulative spend after each chunk so
        # the dashboard's budget bar advances during the long detect stage.
        _e("cost", _cost_summary(ledger))
        all_findings.extend(merged)

        # Reflect the per-chunk outcome on the queue so the lead can
        # see what's been done across resumes.
        if task_queue is not None and chunk_obj is not None and run_id:
            tid = _task_id_for_chunk(chunk_obj)
            try:
                if failed:
                    await task_queue.mark_failed(
                        tid, user_id=user_id,
                        reason="one or more detector runs raised",
                    )
                else:
                    await task_queue.mark_completed(
                        tid, user_id=user_id,
                        result_summary=f"{len(merged)} findings from chunk",
                    )
            except Exception as exc:  # noqa: BLE001 — queue advisory
                log.debug(
                    "bug_finder_task_mark_failed",
                    task_id=tid, error=str(exc),
                )
    return all_findings


async def _run_fuzz_stage(
    chunks: list[_Chunk],
    *,
    config: BugFinderRunConfig,
    cm: ContainerManager,
    workspace: PreparedWorkspace,
    user_id: str,
) -> list[Finding]:
    """Per-chunk atheris fuzz pass. Sequential — the shared workspace
    directory means parallel fuzz sessions would race on the same
    artifact prefix.

    Reads each chunk's source from the container, classifies it, and
    (when fuzzable) generates a harness, runs it, and triages crashes
    into Finding rows. Bound by ``fuzz_max_total_seconds`` across all
    chunks; once exceeded, remaining chunks are silently skipped.

    Returns ``[]`` when the leg is disabled, no chunks are fuzzable,
    or atheris install fails — same shape as the symbolic gate's
    graceful-skip pattern.
    """
    if not config.enable_fuzz_leg or not chunks:
        return []

    from augmentum.bug_finder.fuzz import (
        classify_chunk,
        generate_harness,
        run_fuzz_harness,
        triage_fuzz_run,
    )
    from augmentum.bug_finder.fuzz.runner import _safe_chunk_id

    out: list[Finding] = []
    deadline = time.monotonic() + config.fuzz_max_total_seconds
    install_attempted = False  # First call carries the install cost; subsequent are instant

    for chunk in chunks:
        if time.monotonic() >= deadline:
            log.info("bug_finder_fuzz_budget_exhausted")
            break

        # Read chunk source from container. ``cat`` is cheap; if the file
        # can't be read we just skip this chunk's fuzz attempt.
        try:
            source = await cm.run_command(
                workspace.workspace_id,
                ["cat", f"/workspace/{chunk.file}"],
                timeout=15.0,
            )
        except Exception:  # noqa: BLE001
            continue
        if not source:
            continue

        verdict = classify_chunk(source, chunk.function, file_path=chunk.file)
        if not verdict.fuzzable:
            continue

        try:
            harness = generate_harness(
                verdict,
                target_file=chunk.file,
                target_function=chunk.function,
                workspace_root="/workspace",
            )
        except ValueError:
            # Methods + unresolvable module paths fall here; the verdict
            # was technically fuzzable but the v1 writer can't render
            # the harness. Skip silently — pending step 2.5.
            continue

        chunk_id = f"{chunk.file}::{chunk.function}::{chunk.line_start}"
        remaining = max(5, int(deadline - time.monotonic()))
        per_chunk = min(config.fuzz_max_seconds_per_chunk, remaining)

        result = await run_fuzz_harness(
            harness,
            cm=cm,
            workspace_id=workspace.workspace_id,
            chunk_id=chunk_id,
            max_seconds=per_chunk,
            # Be patient on first install (clang + atheris ~ few min);
            # subsequent chunks reuse the install in seconds.
            install_timeout=480.0 if not install_attempted else 60.0,
        )
        install_attempted = True

        if result.skipped:
            # If the FIRST chunk skipped due to install failure, every
            # subsequent chunk will skip for the same reason. Bail early.
            if "install" in result.skip_reason:
                log.warning(
                    "bug_finder_fuzz_install_unrecoverable",
                    reason=result.skip_reason,
                )
                break
            continue

        artifact_dir = (
            f"/workspace/.augmentum/fuzz/{_safe_chunk_id(chunk_id)}/artifacts"
        )
        out.extend(triage_fuzz_run(
            result,
            harness=harness,
            chunk_file=chunk.file,
            chunk_function=chunk.function,
            artifact_dir=artifact_dir,
        ))

    return out


def _merge_cross_modal(
    llm_findings: list[Finding],
    fuzz_findings: list[Finding],
) -> list[Finding]:
    """Combine LLM-detector findings with fuzz findings.

    For each LLM finding whose ``(file, function)`` matches a fuzz
    finding, ``families_to_confirm`` bumps by 1 — that's a cross-modal
    confirmation, the strongest precision signal the pipeline can
    produce. Fuzz findings without a matching LLM finding pass through
    as standalone rows so the user sees the bug either way.

    The LLM finding wins as the "canonical" row (richer claim text,
    severity rubric) when both exist; the fuzz crash is folded in as
    a note + the bumped family count.
    """
    if not fuzz_findings:
        return llm_findings
    if not llm_findings:
        return list(fuzz_findings)

    fuzz_by_site: dict[tuple[str, str], Finding] = {}
    for f in fuzz_findings:
        fuzz_by_site.setdefault((f.file, f.function), f)

    matched_sites: set[tuple[str, str]] = set()
    merged: list[Finding] = []

    for llm in llm_findings:
        site = (llm.file, llm.function)
        fuzz = fuzz_by_site.get(site)
        if fuzz is None:
            merged.append(llm)
            continue
        matched_sites.add(site)
        # Bump cross-modal confirmation. Finding is a mutable dataclass,
        # but we'd rather not surprise other holders of the list —
        # build a fresh instance instead.
        enriched_notes = list(llm.notes) + [
            f"Cross-modal confirmation: {fuzz.claim}",
        ]
        merged.append(Finding(
            id=llm.id,
            file=llm.file,
            function=llm.function,
            claim=llm.claim,
            claim_signature=llm.claim_signature,
            severity=max(
                (llm.severity, fuzz.severity),
                key=lambda s: ["info", "low", "medium", "high", "critical"].index(s),
            ),
            evidence_paths=llm.evidence_paths,
            suggested_repro=llm.suggested_repro or fuzz.suggested_repro,
            status=llm.status,
            runs_to_confirm=llm.runs_to_confirm,
            total_runs=llm.total_runs,
            families_to_confirm=llm.families_to_confirm + 1,
            total_families=max(llm.total_families, 2),
            repro_path=llm.repro_path or fuzz.repro_path,
            repro_command=llm.repro_command or fuzz.repro_command,
            repro_output=llm.repro_output or fuzz.repro_output,
            invariant=llm.invariant,
            patch=llm.patch,
            fix_attempts=llm.fix_attempts,
            notes=enriched_notes,
        ))

    for fuzz in fuzz_findings:
        if (fuzz.file, fuzz.function) not in matched_sites:
            merged.append(fuzz)

    return merged


async def _run_verify_is_real(
    findings: list[Finding],
    *,
    config: BugFinderRunConfig,
    cm: ContainerManager,
    workspace: PreparedWorkspace,
    resolve_backend: BackendResolver,
    user_id: str,
    ledger: list[CostLedgerEntry],
) -> list[Finding]:
    """Stage 5: try to construct a triggering repro for each finding.

    Findings already in ``CONFIRMED`` status (e.g. fuzz-class findings
    where the crash IS the PoC) pass through unchanged — there's no
    point asking the verifier to build a repro for a bug that already
    has one.
    """
    promoted: list[Finding] = []
    for finding in findings:
        if finding.status == FindingStatus.CONFIRMED.value:
            promoted.append(finding)
            continue
        state = CoderState(
            session_id=f"bf_verifier_{finding.id}", workspace_id=workspace.workspace_id,
        )
        tools = _build_tools_for_role(
            cm, workspace.workspace_id, state, role=Role.VERIFIER, user_id=user_id,
        )
        backend, clean_model = await resolve_backend(config.role_models.verifier)
        spec = make_repro_spec(
            finding,
            model=clean_model,
            tools=tools,
            budget=config.verifier_budget,
            system_prompt_prefix=_threat_model_prefix_block(config.intake.threat_model),
        )
        # Retry transient verifier errors so a backend blip doesn't bury a
        # real finding as "unconfirmable" (mirrors the detector path).
        if config.verifier_max_retries > 0:
            # Bind loop vars explicitly (the retry runs to completion within
            # this iteration, but the binding makes that invariant local).
            result = await run_with_retry(
                lambda s=spec, b=backend: run_subagent(s, backend=b),
                max_retries=config.verifier_max_retries,
                base_delay_s=config.detector_retry_base_delay_s,
            )
        else:
            result = await run_subagent(spec, backend=backend)
        ledger.append(_ledger_entry("verifier_repro", result, config.role_models.verifier))
        outcome = parse_repro_result(result)
        promoted.append(apply_repro_outcome(finding, outcome))
    return promoted


async def _attempt_fix(
    finding: Finding,
    attempt: int,
    *,
    config: BugFinderRunConfig,
    cm: ContainerManager,
    workspace: PreparedWorkspace,
    resolve_backend: BackendResolver,
    user_id: str,
    ledger: list[CostLedgerEntry],
    substrate_host_root: Path | None = None,
    run_id: str = "",
) -> Finding:
    """One fix attempt: reset → fixer → capture diff → verifier → resolve.

    When ``substrate_host_root`` is non-None, every touched file gets
    an SWD-style receipt appended to ``receipts.jsonl`` so future runs
    can answer "is the fix we landed last time still on disk?".
    """
    await _git_reset_to_baseline(cm, workspace.workspace_id)

    fixer_state = CoderState(
        session_id=f"bf_fixer_{finding.id}_a{attempt}",
        workspace_id=workspace.workspace_id,
    )
    fixer_tools = _build_tools_for_role(
        cm, workspace.workspace_id, fixer_state, role=Role.FIXER, user_id=user_id,
    )
    fixer_user_msg = FIXER_USER_TEMPLATE.format(
        finding_id=finding.id,
        file=finding.file,
        function=finding.function,
        severity=finding.severity,
        claim=finding.claim,
        claim_signature=finding.claim_signature,
        repro_command=finding.repro_command or "(none)",
        repro_path=finding.repro_path or "(none)",
        evidence_paths="\n".join(f"  - {p}" for p in finding.evidence_paths) or "  (none)",
        suggested_repro=finding.suggested_repro or "(none)",
    )
    fixer_backend, fixer_clean = await resolve_backend(config.role_models.fixer)
    fixer_spec = SubagentSpec(
        role=Role.FIXER.value,
        model=fixer_clean,
        system_prompt=FIXER_SYSTEM_PROMPT,
        initial_user_message=fixer_user_msg,
        tools=fixer_tools,
        budget=config.fixer_budget,
        tool_guard=fixer_guard,
        instance_id=f"fixer_{finding.id}_a{attempt}",
        temperature=0.0,
    )
    fixer_result = await run_subagent(fixer_spec, backend=fixer_backend)
    ledger.append(_ledger_entry("fixer", fixer_result, config.role_models.fixer))

    fixer_decoded = _parse_fixer_output(fixer_result.output)
    patch = await _git_capture_patch(cm, workspace.workspace_id)
    if not patch.strip():
        from augmentum.bug_finder.verifier import FixVerifyOutcome
        outcome = FixVerifyOutcome(
            accept=False, repro_passes_now=False, regressions_detected=False,
            repro_evidence="",
            test_evidence="",
            rejection_reason="fixer produced no diff",
        )
        return apply_fix_verify_outcome(
            finding, outcome, patch_text="", fix_attempts=attempt,
            invariant=(fixer_decoded.invariant if fixer_decoded else ""),
        )

    # Run the fix-verifier in-place. There's no parallel agent, so the
    # invariant the design doc describes ("verifier in fork the fixer
    # never sees") is satisfied positionally — the fixer has already
    # returned and is no longer active.
    verifier_state = CoderState(
        session_id=f"bf_fix_verify_{finding.id}_a{attempt}",
        workspace_id=workspace.workspace_id,
    )
    verifier_tools = _build_tools_for_role(
        cm, workspace.workspace_id, verifier_state, role=Role.VERIFIER, user_id=user_id,
    )
    verify_backend, verify_clean = await resolve_backend(config.role_models.verifier)
    verify_spec = make_fix_verify_spec(
        finding,
        model=verify_clean,
        tools=verifier_tools,
        budget=config.verifier_budget,
        system_prompt_prefix=_threat_model_prefix_block(config.intake.threat_model),
    )
    verify_result = await run_subagent(verify_spec, backend=verify_backend)
    ledger.append(_ledger_entry("verifier_fix", verify_result, config.role_models.verifier))
    fix_verify_outcome = parse_fix_verify_result(verify_result)

    # Symbolic re-verification — only gates a patch that the verifier
    # already accepted. Rejects fixes that pass the PoC but introduce
    # new Semgrep findings (the regression class the test suite misses).
    if config.enable_symbolic_gate and fix_verify_outcome.accept:
        from augmentum.bug_finder.symbolic_gate import (
            check_patch,
            extract_patched_files,
        )
        try:
            patched_files = extract_patched_files(patch)
            gate_result = await check_patch(
                cm, workspace.workspace_id,
                file_paths=patched_files,
            )
        except Exception as exc:  # noqa: BLE001 — gate is advisory
            log.warning(
                "bug_finder_symbolic_gate_error",
                workspace_id=workspace.workspace_id,
                finding_id=finding.id, attempt=attempt, error=str(exc),
            )
        else:
            log.info(
                "bug_finder_symbolic_gate",
                finding_id=finding.id, attempt=attempt,
                passed=gate_result.passed, skipped=gate_result.skipped,
                new_findings=len(gate_result.new_findings),
                duration_ms=gate_result.duration_ms,
            )
            if not gate_result.passed:
                # Override the fix-verifier — patch is rejected because
                # it introduced new symbolic findings.
                from augmentum.bug_finder.verifier import FixVerifyOutcome
                rule_ids = ", ".join(
                    f.rule_id for f in gate_result.new_findings[:5]
                )
                fix_verify_outcome = FixVerifyOutcome(
                    accept=False,
                    repro_passes_now=fix_verify_outcome.repro_passes_now,
                    regressions_detected=True,
                    repro_evidence=fix_verify_outcome.repro_evidence,
                    test_evidence=fix_verify_outcome.test_evidence,
                    rejection_reason=(
                        f"symbolic gate: {len(gate_result.new_findings)} new "
                        f"semgrep finding(s) introduced [{rule_ids}]"
                    ),
                )

    updated = apply_fix_verify_outcome(
        finding,
        fix_verify_outcome,
        patch_text=patch,
        fix_attempts=attempt,
        invariant=(fixer_decoded.invariant if fixer_decoded else ""),
    )

    # Best-effort receipt trail. The fixer's edits already landed
    # via the coder runtime — we're not re-verifying, just attesting
    # what happened for cross-run trust queries.
    try:
        await _emit_fix_receipts(
            cm=cm,
            workspace_id=workspace.workspace_id,
            finding=updated,
            attempt=attempt,
            patch=patch,
            fix_accepted=fix_verify_outcome.accept,
            rejection_reason=fix_verify_outcome.rejection_reason,
            model_id=fixer_clean,
            provider=getattr(
                config.role_models.fixer, "provider", "",
            ) or "",
            run_id=run_id,
            substrate_host_root=substrate_host_root,
        )
    except Exception:  # noqa: BLE001 — receipts are isolated
        log.debug(
            "bug_finder_fix_receipts_failed", exc_info=True,
        )

    return updated


async def _run_pen_test_leg(
    findings: list[Finding],
    *,
    config: BugFinderRunConfig,
    workspace_root_for_probes: Path | None,
    resolve_backend: BackendResolver,
    ledger: list[CostLedgerEntry],
    event_emit: Callable[[str, dict[str, Any]], None] | None = None,
    run_id: str = "",
    progress_callback=None,
    notes: list[str] | None = None,
    job_ctx: JobContext | None = None,
) -> list[Finding]:
    """Stage 5.5: dynamic pen-test pass over confirmed findings.

    Per finding, the pen_tester subagent boots the workspace's app
    (using ``boot_under_test``) and sends probes (``http_attack``) to
    decide whether the defense actually holds. Verdict is stamped on
    the finding's notes; refuted findings are downgraded.

    Lifecycle: a per-run ``_UnderTestRegistry`` is created at the
    start and torn down in ``try/finally``. Booted services don't
    persist across the leg boundary — between findings, an LLM that
    boots service A then never tears it down would leak only until
    teardown_all() runs at the end. This is intentional: the LLM
    has no teardown tool.
    """
    notes = notes if notes is not None else []

    def _e(kind: str, payload: dict[str, Any]) -> None:
        if event_emit is None:
            return
        event_emit(kind, {"run_id": run_id, **payload})

    if workspace_root_for_probes is None:
        notes.append(
            "pen_test leg skipped: no deterministic_root resolved "
            "(receipts trail + boot-tools need a workspace path on disk)",
        )
        return findings

    confirmed = [
        f for f in findings
        if f.status == FindingStatus.CONFIRMED.value
    ]
    if not confirmed:
        notes.append(
            "pen_test leg ran but found no confirmed findings to probe",
        )
        return findings

    # Lazy imports — the pen_test substrate is only loaded when the
    # leg is enabled, so disabled runs don't pay any import cost.
    from augmentum.bug_finder.agent_tools import build_pen_test_tools
    from augmentum.bug_finder.pen_test_boot import _UnderTestRegistry
    from augmentum.bug_finder.pen_tester import (
        DEFAULT_PEN_TESTER_BUDGET,
        run_pen_tester,
        verdict_to_note,
    )
    from augmentum.bug_finder.subagent import (
        PEN_TESTER_TOOL_NAMES,
        filter_tools,
    )

    registry = _UnderTestRegistry()
    # The deterministic substrate (list_routes, decorators_on, etc.)
    # plus the pen_test toolset, all bound to this workspace.
    from augmentum.bug_finder.agent_tools import (
        build_deterministic_tools,
    )
    det_tools = build_deterministic_tools(workspace_root_for_probes)
    pt_tools = build_pen_test_tools(
        workspace_root_for_probes,
        workspace_root_for_receipts=workspace_root_for_probes,
        under_test_registry=registry,
    )
    # The shim's filter_tools is the same allow-list-applying helper
    # used by the lead loop; here it gates by PEN_TESTER_TOOL_NAMES.
    all_tools = tuple(det_tools) + tuple(pt_tools)
    tools = filter_tools(list(all_tools), PEN_TESTER_TOOL_NAMES)

    backend, clean_model = await resolve_backend(
        config.role_models.for_role(Role.PEN_TESTER),
    )

    _e("stage", {
        "stage": "pen_testing",
        "progress": 0.65,
        "candidates": len(confirmed),
    })

    updated_by_id: dict[str, Finding] = {f.id: f for f in findings}
    confirmed_count = 0
    refuted_count = 0
    inconclusive_count = 0

    try:
        for finding in confirmed:
            await _safe_cancel_check(job_ctx)
            try:
                run_result = await run_pen_tester(
                    model=clean_model,
                    backend=backend,
                    tools=tools,
                    finding=finding,
                    boot_command_hint=config.pen_test_boot_command,
                    boot_port_hint=config.pen_test_boot_port,
                    healthcheck_path_hint=config.pen_test_healthcheck_path,
                    budget=DEFAULT_PEN_TESTER_BUDGET,
                    instance_id=f"pen_tester_{finding.id}",
                    progress_callback=progress_callback,
                )
            except Exception as exc:  # noqa: BLE001 — leg must not break run
                log.warning(
                    "bug_finder_pen_tester_invocation_failed",
                    finding_id=finding.id, error=str(exc),
                )
                finding.notes.append(
                    f"pen_test: error {type(exc).__name__} — leg skipped "
                    "this finding",
                )
                continue

            ledger.append(_ledger_entry(
                "pen_tester",
                run_result.subagent_result,
                config.role_models.for_role(Role.PEN_TESTER),
            ))

            verdict = run_result.verdict
            if verdict is None:
                finding.notes.append(
                    "pen_test: inconclusive — output unparseable",
                )
                inconclusive_count += 1
                _e("pen_test_verdict", {
                    "finding_id": finding.id,
                    "verdict": "inconclusive",
                    "rationale": "unparseable",
                })
                continue

            note = verdict_to_note(verdict)
            finding.notes.append(note)
            if verdict.is_refuted:
                # Downgrade severity so the fix loop doesn't burn
                # tokens on a finding the active probe didn't reproduce.
                # Floor at "low" rather than dropping entirely — the
                # static signal still has SOME value.
                original = finding.severity
                if finding.severity in {"critical", "high", "medium"}:
                    finding.severity = "low"
                    finding.notes.append(
                        f"severity downgraded {original}→low "
                        f"after pen_test refutation",
                    )
                refuted_count += 1
            elif verdict.is_confirmed:
                confirmed_count += 1
            else:
                inconclusive_count += 1
            _e("pen_test_verdict", {
                "finding_id": finding.id,
                "verdict": verdict.verdict,
                "evidence_count": len(verdict.evidence),
                "rationale": verdict.rationale[:240],
            })
            updated_by_id[finding.id] = finding
    finally:
        # Always tear down any booted services, even on cancel.
        try:
            verdicts = await registry.teardown_all()
            if verdicts:
                notes.append(
                    f"pen_test teardown: {len(verdicts)} services "
                    + ", ".join(f"{k[:12]}={v}" for k, v in verdicts.items())[:160],
                )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "bug_finder_pen_test_teardown_failed",
                error=str(exc), exc_info=True,
            )

    notes.append(
        f"pen_test leg: {confirmed_count} confirmed, "
        f"{refuted_count} refuted, {inconclusive_count} inconclusive "
        f"(of {len(confirmed)} candidates)",
    )
    return list(updated_by_id.values())


async def _run_fix_loop(
    findings: list[Finding],
    *,
    config: BugFinderRunConfig,
    cm: ContainerManager,
    workspace: PreparedWorkspace,
    resolve_backend: BackendResolver,
    user_id: str,
    ledger: list[CostLedgerEntry],
    substrate_host_root: Path | None = None,
    run_id: str = "",
) -> list[Finding]:
    """Sequential fix loop with per-attempt git reset isolation.

    ``substrate_host_root`` enables SWD receipt emission per attempt
    (see ``_emit_fix_receipts``). When None, the loop runs identically
    to its pre-substrate behavior — no receipts, no extra overhead.
    """
    await _git_set_baseline(cm, workspace.workspace_id)

    out: list[Finding] = []
    for finding in findings:
        if finding.status != FindingStatus.CONFIRMED.value:
            out.append(finding)
            continue
        current = finding
        for attempt in range(1, config.max_fix_attempts_per_finding + 1):
            current = await _attempt_fix(
                current,
                attempt,
                config=config,
                cm=cm,
                workspace=workspace,
                resolve_backend=resolve_backend,
                user_id=user_id,
                ledger=ledger,
                substrate_host_root=substrate_host_root,
                run_id=run_id,
            )
            if current.status == FindingStatus.FIXED.value:
                break
        if current.status != FindingStatus.FIXED.value:
            current = Finding(**{
                **{k: getattr(current, k) for k in current.__dataclass_fields__},
                "status": FindingStatus.FIX_FAILED.value,
                "notes": [
                    *current.notes,
                    f"fix_failed: gave up after {config.max_fix_attempts_per_finding} attempts",
                ],
            })
        # Always reset back to baseline before moving to the next finding
        # so the next fixer sees a clean tree.
        await _git_reset_to_baseline(cm, workspace.workspace_id)
        out.append(current)
    return out


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


async def _safe_progress(
    job_ctx: JobContext | None, progress: float, stage: str,
) -> None:
    """Idempotent progress write — no-op when no JobContext attached."""
    if job_ctx is None:
        return
    try:
        await job_ctx.update_progress(progress, stage)
    except Exception:  # noqa: BLE001
        log.debug("bug_finder_progress_write_failed", exc_info=True)


async def _safe_cancel_check(job_ctx: JobContext | None) -> None:
    """Cancellation point. JobCancelled propagates out of the orchestrator."""
    if job_ctx is None:
        return
    await job_ctx.check_cancel()


def _lead_module_available() -> bool:
    """Cheap lookup: does the lead module export ``is_implemented`` and
    return True. The orchestrator branches on this so the substrate
    can ship before the loop is wired."""
    try:
        from augmentum.bug_finder import lead as _lead_mod
        return bool(getattr(_lead_mod, "is_implemented", lambda: False)())
    except Exception:  # noqa: BLE001
        return False


def _build_lead_dispatchers(
    *,
    config: BugFinderRunConfig,
    cm: ContainerManager,
    workspace: PreparedWorkspace,
    resolve_backend: BackendResolver,
    user_id: str,
    ledger: list[CostLedgerEntry],
    run_id: str,
    emit: Callable[[str, dict[str, Any], bool], None] | None,
    deterministic_root: Path | None = None,
    task_queue: Any | None = None,
) -> dict[str, Any]:
    """Build the per-kind dispatcher map the lead loop invokes.

    Each dispatcher receives a BugFinderTask and returns
    ``(ok, summary)``. The dispatchers wrap the existing detector /
    verifier / fixer subagents so the lead doesn't need to know about
    SubagentSpec wiring.
    """
    from augmentum.bug_finder.task_queue import BugFinderTask, TaskKind

    semaphore = asyncio.Semaphore(max(1, config.detector_concurrency))

    def _emit_finding_landed(f: Finding) -> None:
        if emit is None:
            return
        try:
            emit("finding_landed", {
                "run_id": run_id,
                "finding_id": f.id,
                "file": f.file, "function": f.function,
                "severity": f.severity,
                "claim_signature": f.claim_signature,
                "claim": (f.claim or "")[:240],
                "status": f.status,
                "runs_to_confirm": f.runs_to_confirm,
                "total_runs": f.total_runs,
                "families_to_confirm": f.families_to_confirm,
                "total_families": f.total_families,
            }, False)
        except Exception:  # noqa: BLE001
            pass

    async def dispatch_detect(
        task: BugFinderTask, findings: list[Finding],
    ) -> tuple[bool, str]:
        file_ = str(task.target.get("file") or "").strip()
        if not file_:
            # The lead occasionally emits detect tasks shaped like a
            # multi-file search (e.g. ``{files: "*_routes.py",
            # pattern: "regex"}``). The detector is single-chunk; reject
            # loudly so the lead sees the failure on its next iteration
            # and switches to an investigate task instead.
            return False, (
                "detect requires a single chunk target with `file` "
                "(file/function/line_start/line_end). For pattern searches "
                "across files, enqueue an `investigate` task instead. "
                f"Got target keys: {sorted(task.target.keys())}"
            )
        try:
            chunk = _Chunk(
                file=file_,
                function=str(task.target.get("function") or "<module>"),
                line_start=int(task.target.get("line_start") or 0),
                line_end=int(task.target.get("line_end") or 0),
                rationale=str(task.target.get("rationale") or ""),
                suspected_class=str(task.target.get("suspected_class") or ""),
            )
        except (TypeError, ValueError) as exc:
            return False, f"malformed target: {exc}"

        try:
            new_findings = await _run_detector_for_chunk(
                chunk, run_index=0, config=config, cm=cm,
                workspace=workspace, resolve_backend=resolve_backend,
                user_id=user_id, ledger=ledger, semaphore=semaphore,
                deterministic_root=deterministic_root,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"detector error: {type(exc).__name__}: {exc}"

        # Dedupe: only append findings the lead hasn't seen.
        existing_ids = {f.id for f in findings}
        added = 0
        for f in new_findings:
            if f.id in existing_ids:
                continue
            findings.append(f)
            existing_ids.add(f.id)
            added += 1
            _emit_finding_landed(f)
        return True, f"{added} new finding(s) from chunk"

    async def dispatch_verify(
        task: BugFinderTask, findings: list[Finding],
    ) -> tuple[bool, str]:
        """Run the verifier on a single finding. Mutates the finding's
        status in-place (CONFIRMED / UNCONFIRMABLE) so the lead's next
        iteration reflects the outcome."""
        from augmentum.bug_finder.verifier import (
            apply_repro_outcome,
            make_repro_spec,
            parse_repro_result,
        )

        finding_id = str(task.target.get("finding_id") or "")
        if not finding_id:
            return False, "verify task missing finding_id"

        idx = next(
            (i for i, f in enumerate(findings) if f.id == finding_id), None,
        )
        if idx is None:
            return False, f"finding {finding_id} not in current set"
        finding = findings[idx]
        if finding.status == FindingStatus.CONFIRMED.value:
            return True, "already confirmed; skipping re-verification"

        state = CoderState(
            session_id=f"bf_verifier_{finding.id}",
            workspace_id=workspace.workspace_id,
        )
        verifier_tools = _build_tools_for_role(
            cm, workspace.workspace_id, state,
            role=Role.VERIFIER, user_id=user_id,
        )
        verify_backend, verify_clean = await resolve_backend(
            config.role_models.verifier,
        )
        spec = make_repro_spec(
            finding,
            model=verify_clean,
            tools=verifier_tools,
            budget=config.verifier_budget,
            system_prompt_prefix=_threat_model_prefix_block(
                config.intake.threat_model,
            ),
        )
        result = await run_subagent(spec, backend=verify_backend)
        ledger.append(_ledger_entry(
            "verifier_repro", result, config.role_models.verifier,
        ))
        outcome = parse_repro_result(result)
        promoted = apply_repro_outcome(finding, outcome)
        findings[idx] = promoted
        _emit_finding_landed(promoted)
        return True, (
            f"verifier outcome: {promoted.status} "
            f"({'repro built' if outcome.confirmed else 'no repro'})"
        )

    async def dispatch_fix(
        task: BugFinderTask, findings: list[Finding],
    ) -> tuple[bool, str]:
        """Attempt a fix for a confirmed finding. Mutates the finding's
        status in-place (FIXED / FIX_FAILED). One attempt per dispatch
        — the lead can re-dispatch up to ``max_fix_attempts_per_finding``
        before giving up."""
        finding_id = str(task.target.get("finding_id") or "")
        if not finding_id:
            return False, "fix task missing finding_id"

        idx = next(
            (i for i, f in enumerate(findings) if f.id == finding_id), None,
        )
        if idx is None:
            return False, f"finding {finding_id} not in current set"
        finding = findings[idx]
        if finding.status != FindingStatus.CONFIRMED.value:
            return False, (
                f"finding not confirmed (status={finding.status}); "
                "cannot fix without verified repro"
            )
        if finding.fix_attempts >= config.max_fix_attempts_per_finding:
            return False, (
                f"max_fix_attempts_per_finding ({config.max_fix_attempts_per_finding}) "
                "reached for this finding"
            )

        try:
            await _git_set_baseline(cm, workspace.workspace_id)
        except Exception as exc:  # noqa: BLE001
            return False, f"git_set_baseline error: {type(exc).__name__}"

        attempt = finding.fix_attempts + 1
        try:
            updated = await _attempt_fix(
                finding, attempt,
                config=config, cm=cm, workspace=workspace,
                resolve_backend=resolve_backend, user_id=user_id,
                ledger=ledger,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"fix attempt error: {type(exc).__name__}: {exc}"

        findings[idx] = updated
        _emit_finding_landed(updated)
        return True, f"fix attempt {attempt} → {updated.status}"

    async def dispatch_investigate(
        task: BugFinderTask, findings: list[Finding],
    ) -> tuple[bool, str]:
        """Spawn an investigator subagent, parse its candidates, and
        enqueue each as a child DETECT task. The lead picks them up on
        subsequent iterations."""
        from augmentum.bug_finder.investigator import (
            DEFAULT_INVESTIGATOR_BUDGET,
            run_investigator,
        )

        anchor = task.target.get("thread_anchor") or ""
        scope_hint = task.target.get("scope_hint") or ""
        finding_id = task.target.get("finding_id") or ""

        # Anchor info — if the investigator can resolve a finding from
        # context it deepens the prompt with that finding's claim.
        anchor_finding = None
        if finding_id:
            anchor_finding = next(
                (f for f in findings if f.id == finding_id), None,
            )

        state = CoderState(
            session_id=f"bf_investigator_{task.task_id}",
            workspace_id=workspace.workspace_id,
        )
        all_tools = _build_tools_for_role(
            cm, workspace.workspace_id, state,
            role=Role.PLANNER, user_id=user_id,
            workspace_root=deterministic_root,
        )
        from augmentum.bug_finder.subagent import INVESTIGATOR_TOOL_NAMES
        investigator_tools = filter_tools(all_tools, INVESTIGATOR_TOOL_NAMES)

        backend, clean_model = await resolve_backend(
            config.role_models.for_role(Role.INVESTIGATOR),
        )
        try:
            result = await run_investigator(
                model=clean_model, backend=backend,
                tools=investigator_tools,
                anchor=str(anchor), scope_hint=str(scope_hint),
                anchor_finding=anchor_finding,
                budget=DEFAULT_INVESTIGATOR_BUDGET,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"investigator error: {type(exc).__name__}: {exc}"

        ledger.append(_ledger_entry(
            "investigator", result.subagent_result,
            config.role_models.for_role(Role.INVESTIGATOR),
        ))

        if result.output is None or not result.output.candidates:
            return True, "no candidates surfaced"

        # Enqueue each candidate as a DETECT task with this investigation
        # as its parent. Priority bumps so investigated candidates outrank
        # planner's blind sweep — investigation should compound, not
        # starve.
        enqueued = 0
        if task_queue is None:
            return True, f"{len(result.output.candidates)} candidates (queue unavailable — not enqueued)"
        for cand in result.output.candidates:
            try:
                await task_queue.enqueue(
                    run_id=run_id, user_id=user_id,
                    workspace_id=workspace.workspace_id,
                    kind=TaskKind.DETECT,
                    target={
                        "file": cand.file,
                        "function": cand.function,
                        "line_start": cand.line_start,
                        "line_end": cand.line_end,
                        "rationale": (
                            f"investigator: {result.output.pattern}"
                            if result.output.pattern else "investigator"
                        ),
                        "suspected_class": cand.similar_to or "",
                    },
                    reason=cand.rationale or "investigator-surfaced",
                    priority=min(10, task.priority + 1),
                    parent_task_id=task.task_id,
                    created_by="investigator",
                )
                enqueued += 1
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "bug_finder_investigator_enqueue_failed",
                    file=cand.file, error=str(exc),
                )
        return True, (
            f"investigator surfaced {len(result.output.candidates)} "
            f"candidate(s); {enqueued} enqueued as DETECT tasks"
        )

    async def dispatch_unimplemented(
        task: BugFinderTask, findings: list[Finding],
    ) -> tuple[bool, str]:
        return False, f"dispatcher for kind={task.kind} not yet wired"

    return {
        TaskKind.DETECT.value: dispatch_detect,
        TaskKind.INVESTIGATE.value: dispatch_investigate,
        TaskKind.VERIFY.value: dispatch_verify,
        TaskKind.FIX.value: dispatch_fix,
        TaskKind.CRITIQUE.value: dispatch_unimplemented,
        TaskKind.COMPREHEND_REFRESH.value: dispatch_unimplemented,
    }


async def run_bug_finder(
    config: BugFinderRunConfig,
    *,
    resolve_backend: BackendResolver,
    container_manager: ContainerManager,
    user_id: str = "",
    job_ctx: JobContext | None = None,
    event_sink: Callable[[str, dict[str, Any], bool], None] | None = None,
    knowledge_store: Any | None = None,
    task_queue: Any | None = None,
) -> BugFinderRunReport:
    """Drive the full bug-finder pipeline. See module docstring.

    ``event_sink``: optional ``(kind, payload, terminal)`` callback the
    pipeline invokes at stage boundaries + subagent completion. Used by
    the SSE progress endpoint to surface live activity. Errors in the
    sink are swallowed so a misbehaving consumer can't break a run.
    """

    def _emit(kind: str, payload: dict[str, Any], *, terminal: bool = False) -> None:
        if event_sink is None:
            return
        try:
            event_sink(kind, payload, terminal)
        except Exception:  # noqa: BLE001 — sink isolation
            log.debug("bug_finder_event_sink_failed", kind=kind, exc_info=True)

    run_id = f"bfr_{uuid.uuid4().hex[:12]}"
    started_at = time.time()
    ledger: list[CostLedgerEntry] = []
    notes: list[str] = []
    same_model = config.role_models.same_model_self_verification

    await _safe_progress(job_ctx, 0.05, "preparing workspace")

    # Fire an immediate event so the live dashboard has something the
    # moment it subscribes — workspace prep (container spin-up + repo
    # clone) can take a minute or two with no other signal, and a silent
    # stream reads as "nothing is happening / it's broken".
    _emit("stage", {
        "run_id": run_id,
        "stage": "workspace_ready",
        "progress": 0.05,
        "note": "preparing workspace — container + repo checkout",
    })

    # Stage 2: workspace prep. The workspace was created via coder mode
    # and lives on the user's container — we just borrow it for the
    # audit pass and apply the reward-hacking strip on entry.
    try:
        workspace = await prepare_workspace(
            cm=container_manager,
            workspace_id=config.intake.workspace_id,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("bug_finder_workspace_prep_failed", exc_info=True)
        return BugFinderRunReport(
            run_id=run_id,
            started_at=started_at,
            completed_at=time.time(),
            intake=asdict(config.intake),
            workspace_id="",
            baseline=WorkspaceBaseline(),
            findings=[],
            confirmation_hist={},
            cost_ledger=ledger,
            stop_reason="error",
            stop_detail=f"workspace_prep_failed: {type(exc).__name__}: {exc}"[:256],
            same_model_self_verification=same_model,
        )

    # Helper to build a final report at any exit point.
    def _build_report(stop_reason: str, stop_detail: str, findings: list[Finding]) -> BugFinderRunReport:
        # Release the detector model pin on every exit path (clean
        # completion, early-exit no-chunks, mid-pipeline wallclock cap).
        # The unpin is idempotent — a no-op when no pin was acquired —
        # so this is safe to call even when the resolver leg failed
        # silently above.
        nonlocal pinned_detector_model, detector_manager
        if pinned_detector_model and detector_manager is not None:
            try:
                detector_manager.unpin_model(pinned_detector_model)
            except Exception:  # noqa: BLE001 — unpin must not break the report
                log.debug(
                    "bug_finder_pin_release_failed",
                    model=pinned_detector_model, exc_info=True,
                )
            pinned_detector_model = ""
            detector_manager = None
        ranked = rank_findings(findings)

        # Detector-health gate. Before honoring a clean completion, check
        # the detector stage actually ran. A run where the fan-out
        # collapsed (most detectors errored) reporting "complete / no
        # findings" is a trust-destroying lie for a security tool — the
        # user reads it as "my code is clean" when the scan never
        # happened. Rewrite that into an honest ``degraded`` verdict and
        # surface the error rate. Only upgrades ``complete`` → ``degraded``
        # — never masks a real error/wallclock/cancel, and a no-detector
        # path (no chunks, prep error) is never degraded.
        health = evaluate_detector_health(
            ledger, threshold=config.detector_error_rate_threshold,
        )
        if stop_reason == "complete" and health["degraded"]:
            pct = round(health["detector_error_rate"] * 100)
            msg = (
                f"detector stage degraded: "
                f"{health['detectors_errored']}/{health['detectors_total']} "
                f"detectors errored ({pct}%) — scan did not functionally "
                f"run; this is NOT a clean result"
            )
            notes.append(msg)
            stop_detail = (
                f"{msg} | prior: {stop_detail}" if stop_detail else msg
            )
            stop_reason = "degraded"
            log.warning(
                "bug_finder_run_degraded",
                run_id=run_id,
                detectors_total=health["detectors_total"],
                detectors_errored=health["detectors_errored"],
                error_rate=health["detector_error_rate"],
            )

        return BugFinderRunReport(
            run_id=run_id,
            started_at=started_at,
            completed_at=time.time(),
            intake=asdict(config.intake),
            workspace_id=workspace.workspace_id,
            baseline=workspace.baseline,
            findings=ranked,
            confirmation_hist=confirmation_histogram(ranked),
            cost_ledger=ledger,
            stop_reason=stop_reason,
            stop_detail=stop_detail,
            same_model_self_verification=same_model,
            notes=notes,
            detector_health=health,
        )

    pipeline_started = time.monotonic()

    # Pin the detector's local model (if any) for the duration of the
    # run. Without this, a sibling service (vision aux, voice TTS, a
    # different chat request) can request a different model mid-flight
    # and llama-server will unload the detector's model — every
    # in-progress detector subagent then burns its budget on the
    # subsequent cold reload. Pinning blocks the eviction; sibling
    # services get a ``ModelPinnedError`` they can retry on. Detector
    # model lookup goes through the same resolver as actual requests
    # so cross-family routing (qwen3.6 detector + sonnet verifier) is
    # honoured cleanly. ``pinned_detector_model`` is captured here so
    # the ``finally`` block at the end of the pipeline can release it
    # even if the resolver itself raised mid-startup.
    pinned_detector_model: str = ""
    detector_manager: Any | None = None
    try:
        detector_model_for_pin = (
            config.detector_models[0]
            if config.detector_models else config.role_models.detector
        )
        det_backend, _ = await resolve_backend(detector_model_for_pin)
        candidate_manager = getattr(det_backend, "_manager", None)
        if candidate_manager is not None and hasattr(candidate_manager, "pin_model"):
            # Match the GGUF basename the manager uses internally (the
            # stem). The catalog id is usually the same — if it's not,
            # the pin is a no-op (mismatched key) and we fall through
            # to the unpinned behaviour rather than crashing.
            pinned_detector_model = detector_model_for_pin
            candidate_manager.pin_model(pinned_detector_model)
            detector_manager = candidate_manager
            notes.append(
                f"detector model pinned: {pinned_detector_model}",
            )
    except Exception as exc:  # noqa: BLE001 — pin is best-effort
        log.debug(
            "bug_finder_pin_acquire_failed",
            error=str(exc)[:200],
            model=detector_model_for_pin if 'detector_model_for_pin' in locals() else "",
        )

    # Resolve the deterministic-tools root once per run. All subagent
    # tool-set constructions read this. ``None`` = substrate disabled
    # for this run (the agent falls back to LLM grepping).
    deterministic_root = _resolve_deterministic_root(
        config.deterministic_tools_root,
    )
    if deterministic_root is not None:
        notes.append(
            f"deterministic-tools substrate active "
            f"(root={deterministic_root})",
        )
    findings: list[Finding] = []

    # Per-iteration progress emitter for any subagent that opts in.
    # Builds a callback compatible with SubagentSpec.progress_callback;
    # the loop calls this on every model response / tool call / tool
    # result so the UI can render "what's the agent doing RIGHT NOW".
    def _make_progress_emit(role_label: str):
        async def _on_progress(progress) -> None:  # noqa: ANN001 — SubagentProgress
            try:
                _emit("subagent_progress", {
                    "run_id": run_id,
                    "role": role_label or progress.role,
                    "instance_id": progress.instance_id,
                    "iteration": progress.iteration,
                    "phase": progress.phase,
                    "tool_name": progress.tool_name,
                    "text_preview": progress.text_preview,
                    "tokens_in": progress.tokens_in,
                    "tokens_out": progress.tokens_out,
                    "wallclock_ms": progress.wallclock_ms,
                })
            except Exception:  # noqa: BLE001 — emit isolation
                log.debug("bug_finder_progress_emit_failed", exc_info=True)
        return _on_progress

    _emit(
        "stage",
        {
            "run_id": run_id,
            "stage": "workspace_ready",
            "progress": 0.10,
            "workspace_id": workspace.workspace_id,
            "detected_language": workspace.baseline.detected_language or "",
            "test_command": workspace.baseline.test_command or "",
        },
    )

    # Stage 2.25: agnostic substrate (Bandit + Ruff + custom AST checks).
    # Best-effort, deterministic — runs only when a host-mounted scan
    # root is available. Seeded findings are added to the working set
    # so they flow through the same verify/fix path as LLM findings.
    # The hybrid (substrate + LLM) is what makes the bug-finder
    # trustworthy as a callable subagent — the LLM alone misses
    # low-hanging deterministic patterns, the substrate alone misses
    # everything that needs reasoning.
    scanner_seeded: list[Finding] = []
    if deterministic_root is not None:
        await _safe_progress(job_ctx, 0.11, "agnostic substrate")
        _emit("stage", {
            "run_id": run_id, "stage": "agnostic_substrate", "progress": 0.11,
        })
        try:
            agnostic_result: AgnosticStageResult = run_agnostic_stage(
                deterministic_root,
            )
            scanner_seeded = list(agnostic_result.seeded_findings)
            notes.append(agnostic_result.summary_line())
            _emit("agnostic_complete", {
                "run_id": run_id,
                "raw": agnostic_result.total_raw,
                "seeded": len(scanner_seeded),
                "suppressed": agnostic_result.suppressed_count,
                "wallclock_seconds": agnostic_result.wallclock_seconds,
                "scanner_counts": dict(agnostic_result.scanner_counts),
            })
            for f in scanner_seeded:
                _emit("finding_landed", {
                    "run_id": run_id,
                    "finding_id": f.id,
                    "file": f.file, "function": f.function,
                    "severity": f.severity,
                    "claim_signature": f.claim_signature,
                    "claim": (f.claim or "")[:240],
                    "status": f.status,
                    "runs_to_confirm": f.runs_to_confirm,
                    "total_runs": f.total_runs,
                    "families_to_confirm": f.families_to_confirm,
                    "total_families": f.total_families,
                })
        except Exception as exc:  # noqa: BLE001 — substrate must be best-effort
            log.warning(
                "bug_finder_agnostic_stage_failed",
                error=str(exc), exc_info=True,
            )
            notes.append(f"agnostic substrate skipped: {type(exc).__name__}")

    # Load this-workspace's prior pattern memory ONCE per run. The
    # brief surfaces to planner + lead as a "where to look" prior;
    # never to the detector (that would bias it toward confirming
    # priors instead of reasoning from chunk evidence — see
    # ``_prefix_patterns`` docstring for the design rationale).
    workspace_priors_brief = ""
    if deterministic_root is not None:
        try:
            from augmentum.bug_finder.workspace_substrate import (
                load_workspace_patterns,
                render_pattern_priors,
            )
            ws_patterns = load_workspace_patterns(deterministic_root)
            workspace_priors_brief = render_pattern_priors(ws_patterns)
            if workspace_priors_brief:
                notes.append(
                    f"loaded {len(ws_patterns)} prior patterns from "
                    f"workspace substrate",
                )
        except Exception as exc:  # noqa: BLE001 — priors are best-effort
            log.warning(
                "bug_finder_workspace_priors_failed",
                error=str(exc), exc_info=True,
            )

    try:
        # Stage 2.5: comprehension. Load or build the structural map.
        # First-contact runs pay an LLM cost; every subsequent run on
        # the same workspace reuses the persisted brief for free.
        await _safe_cancel_check(job_ctx)
        await _safe_progress(job_ctx, 0.12, "comprehending")
        _emit("stage", {
            "run_id": run_id, "stage": "comprehending", "progress": 0.12,
        })
        knowledge_brief = await _ensure_knowledge(
            config=config,
            cm=container_manager,
            workspace=workspace,
            resolve_backend=resolve_backend,
            user_id=user_id,
            ledger=ledger,
            knowledge_store=knowledge_store,
            notes=notes,
            progress_callback=_make_progress_emit("comprehender"),
        )
        # Surface comprehension result to the UI so users see whether
        # the brief landed and what it cost. brief_chars=0 + a note in
        # the report tells the dashboard "comprehension failed gracefully".
        _emit("comprehension_complete", {
            "run_id": run_id,
            "brief_chars": len(knowledge_brief),
            "had_prior_map": bool(knowledge_brief),
        })

        # Seeded hunting playbook — TARGETED to the vuln classes this
        # codebase's risk surfaces expose (from the comprehension map),
        # primed on cold-start when the self-learned pattern memory is
        # still empty. Additive, planner-only, best-effort.
        playbook_brief = ""
        if config.enable_seeded_playbook:
            try:
                from augmentum.bug_finder.playbook import (
                    render_playbook_brief,
                    select_playbook,
                )
                _know = await knowledge_store.get(
                    user_id=user_id, workspace_id=workspace.workspace_id,
                )
                _rs_names = tuple(
                    rs.name for rs in _know.risk_surfaces if rs.name
                )
                _entries = select_playbook(
                    risk_surface_names=_rs_names,
                    max_entries=config.max_playbook_classes,
                )
                playbook_brief = render_playbook_brief(_entries)
                if playbook_brief:
                    notes.append(
                        f"seeded playbook: {len(_entries)} class card(s) "
                        f"for surfaces {list(_rs_names) or '[cold-start]'}",
                    )
            except Exception as exc:  # noqa: BLE001 — priors are best-effort
                log.warning(
                    "bug_finder_playbook_failed", error=str(exc), exc_info=True,
                )

        # Purposeful-on-general-input: when the user gave no threat model,
        # synthesize a grounded one from the comprehension brief + detected
        # stack so detectors/verifiers/planner hunt with PURPOSE instead of
        # empty framing (the 06-14 "explore, no threat model" spray). Frozen
        # dataclasses → rebuild config once; every downstream stage then
        # reads the derived model via config.intake.threat_model. Pure
        # prompt-framing — no control-flow change.
        from augmentum.bug_finder.scope_deriver import derive_threat_model
        _derived_tm, _was_derived = derive_threat_model(
            existing_threat_model=config.intake.threat_model,
            knowledge_brief=knowledge_brief,
            detected_language=getattr(workspace.baseline, "detected_language", ""),
            user_goal_description=getattr(
                config.intake.user_goal, "description", "",
            ),
        )
        if _was_derived:
            config = _dc_replace(
                config,
                intake=_dc_replace(config.intake, threat_model=_derived_tm),
            )
            notes.append(
                "threat model auto-derived from comprehension "
                "(general input supplied none)",
            )
            _emit("threat_model_derived", {
                "run_id": run_id, "chars": len(_derived_tm),
            })

        # Stage 2.6: check-writer. Turn comprehender pillars into
        # permanent codebase-specific AST checks. New checks fire this
        # run (folded into scanner_seeded) and run free on every future
        # audit. Best-effort — never blocks the run.
        try:
            await _safe_cancel_check(job_ctx)
            new_checks_seeded = await _run_check_writer_stage(
                config=config,
                cm=container_manager,
                workspace=workspace,
                deterministic_root=deterministic_root,
                resolve_backend=resolve_backend,
                user_id=user_id,
                ledger=ledger,
                knowledge_store=knowledge_store,
                notes=notes,
                event_emit=_emit,
                run_id=run_id,
                progress_callback=_make_progress_emit("check_writer"),
            )
            if new_checks_seeded:
                scanner_seeded = scanner_seeded + new_checks_seeded
        except Exception as exc:  # noqa: BLE001 — check-writing is advisory
            log.warning(
                "bug_finder_check_writer_stage_failed",
                error=str(exc), exc_info=True,
            )
            notes.append(
                f"check-writer stage skipped: {type(exc).__name__}",
            )

        # Stage 3: plan.
        await _safe_cancel_check(job_ctx)
        await _safe_progress(job_ctx, 0.15, "planning")
        _emit("stage", {"run_id": run_id, "stage": "planning", "progress": 0.15})
        if config.run_mode == "static_chunk":
            # Planner-bypass mode: AST-walk the workspace and emit one
            # chunk per function. Reserved for whole-project sweeps
            # where the planner's token budget would otherwise drown
            # reading the codebase before producing chunks. See module
            # docstring on ``static_chunker.py`` for the trade-offs.
            chunks = _run_static_chunker(
                workspace_root=deterministic_root,
                focus_paths=tuple(config.intake.focus_paths),
                max_chunks=config.max_chunks,
                notes=notes,
            )
        else:
            chunks = await _run_planner(
                config,
                cm=container_manager,
                workspace=workspace,
                resolve_backend=resolve_backend,
                user_id=user_id,
                ledger=ledger,
                knowledge_brief=knowledge_brief,
                workspace_priors_brief=workspace_priors_brief,
                playbook_brief=playbook_brief,
                progress_callback=_make_progress_emit("planner"),
                task_queue=task_queue,
                run_id=run_id,
            )
        # Branch on mode: named-bug routes through the lead agent's
        # dynamic decision loop (Phase 3 of the autonomous-direction
        # work). Explore mode keeps the existing static pipeline —
        # statistical coverage matters more there than per-thread
        # focus, and the pipeline is well-tested.
        use_lead = (
            task_queue is not None
            and config.intake.user_goal.is_named_bug()
            and _lead_module_available()
        )

        # Empty-chunks handling. Static pipeline bails — there's
        # nothing to feed the detector stage. Named-bug mode keeps
        # going: the lead can seed work from the user_goal even
        # without the planner's chunks (observed: deepseek-v4-pro
        # often hits budget without committing chunks; the lead's
        # job is precisely to compensate).
        if not chunks:
            if not use_lead:
                # If the scanner substrate seeded anything, keep going —
                # those still deserve to be verified and reported even
                # when the planner has nothing for the LLM to chew on.
                if not scanner_seeded:
                    notes.append("planner produced no chunks; nothing to scan")
                    return _build_report("complete", "no chunks", findings)
                notes.append(
                    "planner produced no chunks; proceeding with "
                    f"{len(scanner_seeded)} scanner-seeded findings",
                )
            # Seed an INVESTIGATE task from the user_goal so the lead
            # has something concrete to dispatch on its first
            # iteration. Investigator output → DETECT children →
            # lead picks them up.
            try:
                from augmentum.bug_finder.task_queue import TaskKind
                await task_queue.enqueue(
                    run_id=run_id, user_id=user_id,
                    workspace_id=workspace.workspace_id,
                    kind=TaskKind.INVESTIGATE,
                    target={
                        "thread_anchor": config.intake.user_goal.description,
                        "scope_hint": ", ".join(config.intake.focus_paths),
                    },
                    reason=(
                        "seed from user_goal (planner emitted no chunks; "
                        "lead must improvise)"
                    ),
                    priority=9,
                    created_by="orchestrator",
                )
                notes.append(
                    "planner emitted 0 chunks — lead seeded with "
                    "an INVESTIGATE task derived from user_goal",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "bug_finder_lead_seed_failed",
                    error=str(exc), exc_info=True,
                )

        # Surface chunk list to the UI so users can watch each chunk
        # tick through detection as it happens.
        _emit("planner_complete", {
            "run_id": run_id,
            "chunks": [
                {
                    "file": c.file, "function": c.function,
                    "line_start": c.line_start, "line_end": c.line_end,
                    "suspected_class": c.suspected_class or "",
                }
                for c in chunks
            ],
        })

        if use_lead:
            await _safe_progress(job_ctx, 0.35, "lead orchestration")
            _emit("stage", {
                "run_id": run_id, "stage": "lead_loop",
                "progress": 0.35, "chunks": len(chunks),
            })
            try:
                from augmentum.bug_finder.lead import (
                    DEFAULT_LEAD_BUDGET,
                    run_lead_loop,
                )
                lead_backend, lead_model = await resolve_backend(
                    config.role_models.for_role(Role.LEAD),
                )
                dispatchers = _build_lead_dispatchers(
                    config=config, cm=container_manager,
                    workspace=workspace, resolve_backend=resolve_backend,
                    user_id=user_id, ledger=ledger, run_id=run_id,
                    emit=_emit,
                    deterministic_root=deterministic_root,
                    task_queue=task_queue,
                )
                user_goal_block = config.intake.user_goal.to_prompt_block()
                # Prepend workspace priors so the lead has the same
                # "where to look" prior the planner sees. Both blocks
                # are optional — a blank prior is a no-op for the lead.
                if workspace_priors_brief:
                    user_goal_block = (
                        workspace_priors_brief
                        + "\n\n---\n\n"
                        + user_goal_block
                    ) if user_goal_block else workspace_priors_brief
                lead_result = await run_lead_loop(
                    model=lead_model, backend=lead_backend,
                    queue=task_queue, run_id=run_id, user_id=user_id,
                    user_goal_block=user_goal_block,
                    initial_findings=findings + scanner_seeded,
                    dispatchers=dispatchers,
                    budget=DEFAULT_LEAD_BUDGET,
                    event_emit=_emit,
                    progress_callback=_make_progress_emit("lead"),
                )
                findings = lead_result.findings
                _emit("cost", _cost_summary(ledger))
                notes.append(
                    "lead orchestration: "
                    f"{lead_result.state.iterations} iters, "
                    f"stop={lead_result.state.stop_reason}, "
                    f"~{lead_result.state.tokens_used:,} tokens",
                )
            except Exception as exc:  # noqa: BLE001 — lead failure → fall back
                log.warning(
                    "bug_finder_lead_failed_fallback", exc_info=True,
                )
                notes.append(
                    f"lead failed ({type(exc).__name__}); "
                    "falling back to static pipeline",
                )
                use_lead = False

        if not use_lead:
            # Stage 4: detect (static pipeline path).
            await _safe_cancel_check(job_ctx)
            await _safe_progress(job_ctx, 0.35, f"detecting ({len(chunks)} chunks)")
            _emit("stage", {
                "run_id": run_id, "stage": "detecting",
                "progress": 0.35, "chunks": len(chunks),
            })
            if time.monotonic() - pipeline_started > config.overall_wallclock_seconds:
                return _build_report("wallclock", "pipeline wallclock exceeded before detect", findings)
            findings = await _run_detect_stage(
                chunks,
                config=config,
                cm=container_manager,
                workspace=workspace,
                resolve_backend=resolve_backend,
                user_id=user_id,
                ledger=ledger,
                event_emit=_emit,
                run_id=run_id,
                progress_callback=_make_progress_emit("detector"),
                task_queue=task_queue,
                deterministic_root=deterministic_root,
            )
            # Surface raw detector output so the UI can render finding
            # cards as they land instead of waiting for the terminal event.
            for f in findings:
                _emit("finding_landed", {
                    "run_id": run_id,
                    "finding_id": f.id,
                    "file": f.file, "function": f.function,
                    "severity": f.severity,
                    "claim_signature": f.claim_signature,
                    "claim": (f.claim or "")[:240],
                    "status": f.status,
                    "runs_to_confirm": f.runs_to_confirm,
                    "total_runs": f.total_runs,
                    "families_to_confirm": f.families_to_confirm,
                    "total_families": f.total_families,
                })

            # Stage 4b: fuzz leg (parallel modality to LLM detector).
            # Runs against the same chunks the planner picked. Fuzzable
            # chunks get an atheris harness + bounded fuzz session; the
            # resulting crashes flow into the findings list via the
            # cross-modal merge below.
            fuzz_findings: list[Finding] = []
            if config.enable_fuzz_leg:
                try:
                    await _safe_progress(job_ctx, 0.45, "fuzzing")
                    _emit("stage", {
                        "run_id": run_id, "stage": "fuzzing",
                        "progress": 0.45, "chunks": len(chunks),
                    })
                    fuzz_findings = await _run_fuzz_stage(
                        chunks,
                        config=config,
                        cm=container_manager,
                        workspace=workspace,
                        user_id=user_id,
                    )
                    if fuzz_findings:
                        notes.append(
                            f"fuzz leg surfaced {len(fuzz_findings)} crashes",
                        )
                except Exception as exc:  # noqa: BLE001 — fuzz failure must not break the run
                    log.warning(
                        "bug_finder_fuzz_stage_failed", exc_info=True,
                    )
                    notes.append(f"fuzz leg skipped: {type(exc).__name__}")

            findings = _merge_cross_modal(findings, fuzz_findings)
            # Scanner-seeded findings flow alongside LLM/fuzz output;
            # they don't compete with cross-modal family-counting since
            # their function placeholder (``<scanner:category>``) won't
            # collide with real handler names.
            if scanner_seeded:
                findings = findings + scanner_seeded

        if not findings:
            notes.append("no findings from any detector")
            return _build_report("complete", "no findings", findings)

        # Stage 5: verify-is-real.
        await _safe_cancel_check(job_ctx)
        await _safe_progress(job_ctx, 0.55, f"verifying ({len(findings)} findings)")
        _emit("stage", {
            "run_id": run_id, "stage": "verifying",
            "progress": 0.55, "findings": len(findings),
        })
        if time.monotonic() - pipeline_started > config.overall_wallclock_seconds:
            return _build_report("wallclock", "pipeline wallclock exceeded before verify", findings)
        findings = await _run_verify_is_real(
            findings,
            config=config,
            cm=container_manager,
            workspace=workspace,
            resolve_backend=resolve_backend,
            user_id=user_id,
            ledger=ledger,
        )
        _emit("cost", _cost_summary(ledger))

        # Post-verify: bump pattern fix_count for any scanner-seeded
        # finding the verifier confirmed. Lets the per-workspace
        # pattern memory learn which scanner rules actually catch real
        # bugs in this codebase vs. produce FPs.
        if deterministic_root is not None and scanner_seeded:
            scanner_finding_ids = {f.id for f in scanner_seeded}
            for f in findings:
                if (
                    f.id in scanner_finding_ids
                    and f.status == FindingStatus.CONFIRMED.value
                ):
                    record_confirmation(deterministic_root, f)

        # Stage 5.5: dynamic pen-test leg. Off by default — when on,
        # each confirmed finding gets an active-probe pass and the
        # verdict stamps onto the finding's notes. Refuted findings
        # have their severity capped at "low" (the defense held).
        # The under-test registry's lifecycle is wrapped in
        # try/finally so booted services don't leak even on cancel.
        if config.enable_pen_test_leg:
            findings = await _run_pen_test_leg(
                findings,
                config=config,
                workspace_root_for_probes=deterministic_root,
                resolve_backend=resolve_backend,
                ledger=ledger,
                event_emit=_emit,
                run_id=run_id,
                progress_callback=_make_progress_emit("pen_tester"),
                notes=notes,
                job_ctx=job_ctx,
            )

        # Stage 6: fix loop.
        confirmed = [f for f in findings if f.status == FindingStatus.CONFIRMED.value]
        if not confirmed:
            notes.append("no findings were confirmable; skipping fix loop")
            return _build_report("complete", "no confirmed findings", findings)

        await _safe_cancel_check(job_ctx)
        await _safe_progress(job_ctx, 0.75, f"fixing ({len(confirmed)} confirmed)")
        _emit("stage", {
            "run_id": run_id, "stage": "fixing",
            "progress": 0.75, "confirmed": len(confirmed),
        })
        if time.monotonic() - pipeline_started > config.overall_wallclock_seconds:
            return _build_report("wallclock", "pipeline wallclock exceeded before fix", findings)
        findings = await _run_fix_loop(
            findings,
            config=config,
            cm=container_manager,
            workspace=workspace,
            resolve_backend=resolve_backend,
            user_id=user_id,
            ledger=ledger,
            substrate_host_root=deterministic_root,
            run_id=run_id,
        )
        _emit("cost", _cost_summary(ledger))

        await _safe_progress(job_ctx, 1.0, "complete")
        report = _build_report("complete", "", findings)
        _emit("done", _terminal_payload(report), terminal=True)
        return report

    except JobCancelled as exc:
        log.info("bug_finder_cancelled", run_id=run_id)
        report = _build_report("cancelled", str(exc), findings)
        _emit("done", _terminal_payload(report), terminal=True)
        return report
    except Exception as exc:  # noqa: BLE001
        log.warning("bug_finder_pipeline_error", run_id=run_id, exc_info=True)
        report = _build_report(
            "error",
            f"{type(exc).__name__}: {exc}"[:256],
            findings,
        )
        _emit("done", _terminal_payload(report), terminal=True)
        return report


def _terminal_payload(report: BugFinderRunReport) -> dict[str, Any]:
    """Compact terminal-event payload — counts + totals, not full findings.

    Subscribers that want the full report poll
    ``GET /api/bug-finder/runs/{run_id}``. Keeping the SSE payload small
    so the stream stays snappy and the replay buffer stays bounded.
    """
    findings = report.findings
    return {
        "run_id": report.run_id,
        "stop_reason": report.stop_reason,
        "stop_detail": report.stop_detail,
        "findings_total": len(findings),
        "findings_fixed": sum(1 for f in findings if f.status == "fixed"),
        "findings_confirmed": sum(
            1 for f in findings if f.status == "confirmed"
        ),
        "findings_fix_failed": sum(
            1 for f in findings if f.status == "fix_failed"
        ),
        "tokens_in": sum(e.tokens_in for e in report.cost_ledger),
        "tokens_out": sum(e.tokens_out for e in report.cost_ledger),
        "wallclock_ms": sum(e.wallclock_ms for e in report.cost_ledger),
        "same_model_self_verification": report.same_model_self_verification,
        "detector_health": report.detector_health,
    }
