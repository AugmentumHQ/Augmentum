"""Orchestrator → coder dispatch contract.

A :class:`CoderDispatch` is the structured task handoff a non-user
orchestrator (Becca, an external CLI via MCP, a scheduled job, a future
autonomy loop) sends to a coder. It carries the technical task, the
verifier set that defines "done", and optional advisory context the
coder treats as soft prior, not authority.

Companion prompt: :data:`augmentum.coder.prompts.DISPATCH_FORK_SYSTEM`
(rendered by :func:`render_dispatch_system`).

See ``docs/superpowers/specs/2026-05-17-coder-promise-verifier-vocabulary.md``
for the vocabulary that populates ``success_criteria``.

Today there's no production caller — the coder runs from a direct-user
turn via ``CoderHandler.process_stream``. This module ships the types so
the Becca-side wiring (and future MCP server) compose against a stable
contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from augmentum.promises.models import Promise

CostTier = Literal["fast", "balanced", "thorough"]
"""Cost tier for the dispatched run.

- ``fast``: single model both phases, smaller context, tighter iter cap,
  Reflect fan-out off. For low-stakes turns.
- ``balanced``: planner uses utility role, actor uses primary chat model,
  Reflect fan-out off. Today's default.
- ``thorough``: full Brief-Plan-Act-Verify-Reflect with PDR-style
  fan-out on verify-fail. Reserved for high-stakes work where the
  orchestrator has budget.
"""

JournalScope = Literal["workspace", "user", "global"]
"""Which past coder journal entries the brief assembler may retrieve.

- ``workspace``: only this container's history (default).
- ``user``: any of this user's coder turns across workspaces.
- ``global``: anonymized cross-user knowledge. Off by default; the
  orchestrator must opt in per dispatch.
"""

PermissionMode = Literal["auto", "confirm_mutations"]
"""Whether mutating tool calls require per-call user approval."""


@dataclass(frozen=True)
class CoderDispatch:
    """The orchestrator → coder contract.

    The orchestrator (Becca, an external CLI, a scheduled job, the
    user typing in coder mode directly) hands the coder one task plus
    the verifier set that defines "done", plus optional advisory
    context. The coder runs in its isolated container and returns a
    structured result via the streaming event channel.

    Three categories of fields:

    1. **Identity** (required): ``workspace_id``, ``user_id``.
    2. **Contract** (required): ``task``, ``success_criteria``.
    3. **Knobs** (optional): everything else — runtime tiering,
       advisory brief, permission policy, journal scope.

    Construct via :meth:`for_direct_user_turn` for the minimal
    no-orchestrator case (the path that's live today). Construct
    directly when an orchestrator is actually composing context.
    """

    # Identity
    workspace_id: str
    """Which container the dispatch targets."""

    user_id: str
    """Owning user. Used for journal scoping + multi-tenant isolation."""

    # Contract
    task: str
    """Technical ask in plain language. The coder restates this verbatim
    at turn start so the orchestrator can spot scope drift."""

    success_criteria: tuple[Promise, ...] = ()
    """The Promise tree the coder must satisfy. Empty = no structured
    contract; the existing TQG (Termination Quality Gate) handles
    completion the way it does for direct-user turns today."""

    # Knobs
    constraints: tuple[str, ...] = ()
    """Free-text constraints the planner should honor (e.g.
    "must not break Argon2id", "ASGI middleware shape preserved")."""

    context_brief: dict[str, Any] | None = None
    """Cross-modal context the orchestrator chose to pass through.

    Advisory only — the coder treats this as soft prior, not authority.
    The schema is intentionally open; the orchestrator owns interpretation.
    A few keys get nicer rendering (``user_mood``, ``recent_decisions``,
    ``memory_snippets``, ``related_files``); everything else renders as
    a flat key-value footer. ``None`` for external-CLI / direct-user
    dispatches where no relational substrate exists.
    """

    cost_tier: CostTier = "balanced"

    iteration_cap: int | None = None
    """Override the strategy default iteration cap. ``None`` = use
    whatever the strategy dispatcher resolves."""

    parallelism: int = 1
    """How many parallel coder dispatches the orchestrator intends to
    spawn against this workspace. The dispatched coder reads this only
    to log it — actually spawning siblings is the orchestrator's
    concern (a single coder never spawns its own siblings)."""

    permission_mode: PermissionMode = "auto"

    journal_scope: JournalScope = "workspace"

    return_to: str = ""
    """Event channel identifier the orchestrator subscribes to for
    structured progress events. Empty = use the default request stream
    (the user's chat channel)."""

    @classmethod
    def for_direct_user_turn(
        cls,
        *,
        workspace_id: str,
        user_id: str,
        task: str,
    ) -> CoderDispatch:
        """Construct the minimal dispatch matching today's direct-user behavior.

        Empty success criteria (TQG handles termination), no advisory
        brief, default tiers. This is what the legacy ``process_stream``
        path would produce if it went through the dispatch contract
        instead of bypassing it.
        """
        return cls(
            workspace_id=workspace_id,
            user_id=user_id,
            task=task,
        )

    @classmethod
    def for_orchestrator_dispatch(
        cls,
        *,
        workspace_id: str,
        user_id: str,
        task: str,
        success_criteria: tuple[Promise, ...] = (),
        constraints: tuple[str, ...] = (),
        context_brief: dict[str, Any] | None = None,
        cost_tier: CostTier = "balanced",
        permission_mode: PermissionMode = "auto",
        journal_scope: JournalScope = "workspace",
        iteration_cap: int | None = None,
        parallelism: int = 1,
        return_to: str = "",
    ) -> CoderDispatch:
        """Construct a full orchestrator dispatch.

        The counterpart to :meth:`for_direct_user_turn`: where that produces
        the minimal contract for the legacy direct-user path (no criteria, no
        brief), this is what a real orchestrator (Becca, a scheduled job, an
        external CLI) uses when it has actually composed a spec — the
        ``success_criteria`` become the coder's inbound contract (seeded into
        ``CoderState.mission`` so the P3 verifier gate has something to judge
        against) and ``context_brief`` carries advisory relational context.

        Everything here is a straight structural copy into the frozen
        dataclass; the classmethod exists so callers name their intent
        (`for_orchestrator_dispatch`) rather than constructing the raw type,
        and so the default knob-set for orchestrated runs lives in one place.
        """
        return cls(
            workspace_id=workspace_id,
            user_id=user_id,
            task=task,
            success_criteria=tuple(success_criteria),
            constraints=tuple(constraints),
            context_brief=context_brief,
            cost_tier=cost_tier,
            permission_mode=permission_mode,
            journal_scope=journal_scope,
            iteration_cap=iteration_cap,
            parallelism=parallelism,
            return_to=return_to,
        )


def render_dispatch_system(dispatch: CoderDispatch, *, fork_prompt: str) -> str:
    """Render the dispatch as a system message for a forked coder.

    Substitutes ``${VAR}`` placeholders in ``fork_prompt`` with fields
    from the dispatch. Pass :data:`DISPATCH_FORK_SYSTEM` from
    ``augmentum.coder.prompts`` as ``fork_prompt`` for the canonical
    template. The resulting string becomes the dispatched coder's
    system prompt body (in addition to the workspace guide and
    phase-specific prompts that still apply on top).

    Uses string replacement (not ``str.format``) so braces in user-
    provided task/constraint text don't trip the substitution.

    Placeholders consumed: ``${TASK}``, ``${CONSTRAINTS}``,
    ``${SUCCESS_CRITERIA}``, ``${CONTEXT_BRIEF}``, ``${COST_TIER}``,
    ``${PARALLELISM}``, ``${PERMISSION_MODE}``.
    """
    rendered = fork_prompt
    rendered = rendered.replace("${TASK}", dispatch.task or "(no task specified)")
    rendered = rendered.replace("${CONSTRAINTS}", _render_constraints(dispatch.constraints))
    rendered = rendered.replace("${SUCCESS_CRITERIA}", _render_success_criteria(dispatch.success_criteria))
    rendered = rendered.replace("${CONTEXT_BRIEF}", _render_brief(dispatch.context_brief))
    rendered = rendered.replace("${COST_TIER}", dispatch.cost_tier)
    rendered = rendered.replace("${PARALLELISM}", str(dispatch.parallelism))
    rendered = rendered.replace("${PERMISSION_MODE}", dispatch.permission_mode)
    return rendered


def _render_constraints(constraints: tuple[str, ...]) -> str:
    if not constraints:
        return "(none specified)"
    return "\n".join(f"- {c}" for c in constraints)


def _render_success_criteria(criteria: tuple[Promise, ...]) -> str:
    if not criteria:
        return (
            "(none specified — the runtime falls back to user-demand "
            "heuristic for termination via the Termination Quality Gate)"
        )
    lines: list[str] = []
    for i, promise in enumerate(criteria, start=1):
        kind = promise.verify.kind.value
        spec = promise.verify.spec or {}
        # Render the spec compactly. Single-key spec gets the value
        # inline; multi-key spec gets dict syntax.
        if not spec:
            spec_repr = ""
        elif len(spec) == 1:
            (k, v), = spec.items()
            spec_repr = f"{k}={v!r}"
        else:
            spec_repr = ", ".join(f"{k}={v!r}" for k, v in sorted(spec.items()))
        desc = promise.description or "(no description)"
        lines.append(f"{i}. {desc}  [verify: {kind}({spec_repr})]")
    return "\n".join(lines)


def _render_brief(brief: dict[str, Any] | None) -> str:
    if not brief:
        return (
            "(no relational context passed through; run on the "
            "technical task only — the orchestrator decided no brief "
            "was relevant for this dispatch)"
        )
    # Conventional keys get nicer rendering. Order matters — most
    # relational-state keys first, then everything else as a flat
    # footer so a freeform orchestrator can extend without us having
    # to know about new keys in advance.
    conventional = ("user_mood", "recent_decisions", "memory_snippets", "related_files")
    lines: list[str] = []
    for key in conventional:
        val = brief.get(key)
        if val in (None, "", [], {}):
            continue
        if isinstance(val, list):
            lines.append(f"- **{key}**:")
            for item in val:
                lines.append(f"  - {item}")
        else:
            lines.append(f"- **{key}**: {val}")
    # Footer: any non-conventional keys.
    for key, val in sorted(brief.items()):
        if key in conventional:
            continue
        if val in (None, "", [], {}):
            continue
        if isinstance(val, list):
            lines.append(f"- **{key}**:")
            for item in val:
                lines.append(f"  - {item}")
        else:
            lines.append(f"- **{key}**: {val}")
    return "\n".join(lines) if lines else "(brief was empty)"
