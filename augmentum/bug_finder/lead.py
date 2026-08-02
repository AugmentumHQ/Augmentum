"""Lead Agent — CC-style dynamic orchestration over the task queue.

Sits above the existing planner / detector / verifier / fixer stages
and decides what to do next at each step. The pipeline is replaced by
a loop:

    while not done:
        prompt   = render_lead_prompt(queue, findings, user_goal, budget)
        decision = await llm_decide(prompt, model, backend)   # one call
        await dispatcher(decision)                            # Python
        if decision.action == "done" or budget_exhausted:
            break

The lead doesn't read source code directly. Its tools are the
dispatchers it can invoke (run detector on this task, run verifier on
this finding, enqueue a new task) — same shape as CC's Task tool with
its narrower-scope subagent dispatches.

The lead's value compounds in named-bug mode: when the user says
"I think there's a session leak when bots are deleted", the lead can
follow that specific thread rather than running a blind plan.
Investigators (next iteration) can add new tasks to the queue when
they find adjacent code worth examining, and the lead sequences
everything.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from augmentum.agents.loop import SubagentSpec, run_subagent
from augmentum.bug_finder.budget import SubagentBudget
from augmentum.bug_finder.findings import Finding
from augmentum.bug_finder.json_salvage import salvage_json_object
from augmentum.bug_finder.role_models import Role
from augmentum.bug_finder.task_queue import (
    BugFinderTask,
    TaskKind,
    TaskQueue,
    TaskStatus,
    render_queue_summary,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Default budget
# ---------------------------------------------------------------------------


# Lead is meta-reasoning, not reading code. Each iteration is a single
# decision (~3-5k tokens) and the lead should rarely need more than 20
# iterations on a single run. Total bound is 20 LLM calls × ~5k tokens.
DEFAULT_LEAD_BUDGET = SubagentBudget(
    max_iterations=20,
    max_wallclock_seconds=600,
    max_tokens=100_000,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


LEAD_SYSTEM_PROMPT = """\
You are the bug-finder LEAD. Your job is to **sequence** the audit:
look at the current task queue and the findings so far, then decide
what to do next.

You DO NOT read source code directly. The detector, investigator, and
verifier do that. Your value is in choosing WHICH thread to pull next.

## Available actions

End every response with a single fenced JSON block. One action per
iteration.

**CRITICAL — task_id discipline:** When dispatching or dropping a
task, the ``task_id`` field MUST be the literal id shown in the queue
summary (looks like ``tsk_a1b2c3d4e5f6...``). Do NOT make up ids like
``flow_routes_detect``, ``p5-detect``, or descriptive shorthand. Copy
the id verbatim from the queue listing — the substring after the
``` ` ``` and before the ``[p…]`` priority marker. If you can't see
a concrete id, enqueue a new task instead.

**CRITICAL — target schemas per kind.** When you enqueue, the
``target`` shape depends on the ``kind``:

- ``detect`` — scans ONE specific chunk for bugs (one file, one
  function, one line range). NOT a multi-file search. Use this
  AFTER you know the exact site. Target shape:
  ```json
  { "file": "augmentum/proxy/auth_routes.py",
    "function": "login",
    "line_start": 42, "line_end": 88,
    "rationale": "<why this chunk>",
    "suspected_class": "<optional bug class hint>" }
  ```

- ``investigate`` — walks the codebase to find sites that match a
  pattern. THIS is the action for "find all the X" / "scan for the Y
  pattern". The investigator returns candidates which the lead then
  enqueues as ``detect`` tasks. Target shape:
  ```json
  { "thread_anchor": "<file:function OR a one-line pattern>",
    "scope_hint": "<optional dir restriction>",
    "finding_id": "<optional — when chasing a specific finding>" }
  ```

- ``verify`` — runs the verifier to construct an executable repro for
  a SPECULATIVE finding (promotes it to CONFIRMED). Target shape:
  ```json
  { "finding_id": "fnd_..." }
  ```

- ``fix`` — attempts a patch for a CONFIRMED finding. Target shape:
  ```json
  { "finding_id": "fnd_..." }
  ```

**Common mistake to avoid:** if the user asks for "find all X across
the codebase", do NOT enqueue a ``detect`` task with ``files``,
``pattern``, ``regex``, or glob fields — the detector doesn't accept
those. Enqueue an ``investigate`` task with the pattern in
``thread_anchor`` instead. The investigator handles the search and
gives back specific chunks for ``detect``.

The format:

```json
{
  "action":     "dispatch" | "enqueue" | "drop_task" | "drop_finding"
                | "done",
  "task_id":    "<id of a pending task — for `dispatch` or `drop_task`>",
  "finding_id": "<id — for `drop_finding`>",
  "new_task":   {                              // for `enqueue` only
    "kind":     "detect" | "investigate" | "verify",
    "target":   { ... },                       // shape varies by kind
    "reason":   "<one sentence>",
    "priority": <1-10>
  },
  "rationale":  "<one sentence — why this action now>"
}
```

## When to use each action

- **dispatch** — Run a pending task that aligns with the user's goal
  or compounds existing findings. The highest-priority pending task
  is the safe default.

- **enqueue** — Add a follow-up task. After a detector reports a
  finding in chunk X, enqueueing an `investigate` task for adjacent
  code can compound discovery without a blind blanket sweep.

- **drop_task** — Remove a pending task that's become irrelevant
  (later evidence showed the bug is elsewhere, the user's goal has
  no overlap with it, etc.). Saves detector budget.

- **drop_finding** — Critique-reject a low-confidence finding before
  spending verifier budget on it. Use sparingly: false drops cost
  more than false verifies in expected value.

- **done** — The queue is empty (or only contains low-priority sweep
  work) AND you've satisfied the user's goal. Provide a one-sentence
  summary in `rationale`.

## Discipline

Be parsimonious. Every dispatch costs detector tokens; every
investigator branch costs more. A 3-finding focused run is more
valuable than a 30-finding scattered sweep. When in doubt, dispatch
the highest-priority pending task and reassess after.

You have a hard cap on iterations and tokens — don't burn budget
deliberating. Pick the next action, dispatch, observe, iterate.

## Deterministic-substrate tools (PREFER these over grepping)

The workspace ships with deterministic analysis tools that return
**ground truth** in milliseconds. ALWAYS reach for these BEFORE
asking an investigator to grep for the same patterns. They have
been hand-tuned over months, return structured findings with
severity + file + line + description, and apply known-intentional
suppressions automatically.

Available tools (the detector + investigator can also call these
during their subagent runs; you get them at the lead level for your
own meta-reasoning):

* **list_routes(method?, path_substr?, file_substr?, limit?)** —
  Returns every HTTP route in the workspace with `{method, path,
  handler, file, line}`. Use this to identify candidate auth /
  billing / admin route handlers BEFORE enqueuing a detect task —
  the route list is ground truth, not a model guess.

* **find_callers_of_endpoint(path_substr, method?)** — Returns
  every frontend JS fetch / WebSocket call that hits an endpoint.
  Empty result = an orphan route (no caller) which is a candidate
  for "is this dead code or server-to-server only?" investigation.

* **security_check** — Returns a STRUCTURED list of security
  findings (SQL injection, SSRF, template XSS, key exposure, stale
  rules). Each carries severity + file + line. Dispatch a verifier
  on each finding instead of re-deriving it.

* **red_team_scan** — Adversarial scanner: data isolation, auth
  bypass, IDOR, token exposure, AI context leaks.

* **code_quality** — Silent catches, console.log in prod, mixed
  error formats, WebSocket contract gaps, tech-debt markers.

* **runtime_checks** — Empty model strings, silent exception
  swallowing, unhandled fetch failures, state outside handlers.

**The strong pattern:** when the user goal mentions "find X", your
FIRST action should be to call the relevant deterministic tool. If
it returns findings, dispatch verifiers on the top N. If it returns
zero, then enqueue an investigator. The investigator is for cases
the deterministic tools don't cover — not as a default search
mechanism.

**Common mistake:** enqueueing an `investigate` task with a regex
pattern in the anchor when the security_check / code_quality tool
would have returned a structured answer in one call. Don't make
the investigator grep when a deterministic tool already grepped
and triaged.
"""


LEAD_USER_TEMPLATE = """\
## Current state

Iteration: {iteration}/{max_iterations}
Findings so far: {findings_count}
Budget remaining: ~{tokens_remaining:,} tokens

{user_goal_block}

{queue_summary}

## Findings summary (top {findings_shown})

{findings_section}

## Decide the next action

Emit a single fenced JSON block. One action per iteration.
"""


# ---------------------------------------------------------------------------
# Decision parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadDecision:
    """Parsed JSON decision returned by the lead's LLM call."""

    action: str                  # "dispatch" | "enqueue" | "drop_task" |
                                 # "drop_finding" | "done"
    task_id: str = ""
    finding_id: str = ""
    new_task: dict[str, Any] | None = None
    rationale: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.action == "done"

    @property
    def is_valid(self) -> bool:
        """Schema-level validity (not semantic validity — e.g.
        ``dispatch`` with an unknown task_id is structurally valid but
        will fail at dispatch time)."""
        if self.action == "dispatch":
            return bool(self.task_id)
        if self.action == "enqueue":
            return (
                self.new_task is not None
                and isinstance(self.new_task, dict)
                and bool(self.new_task.get("kind"))
                and isinstance(self.new_task.get("target"), dict)
            )
        if self.action == "drop_task":
            return bool(self.task_id)
        if self.action == "drop_finding":
            return bool(self.finding_id)
        if self.action == "done":
            return True
        return False


def _last_json_object(output: str) -> dict[str, Any] | None:
    """Salvage the last usable JSON object (truncation-tolerant — audit
    2026-06-17)."""
    return salvage_json_object(output)


def parse_lead_decision(output: str) -> LeadDecision | None:
    """Decode the lead's LLM output into a structured decision.

    Returns ``None`` when no fenced JSON parses or the schema is wrong.
    Caller treats ``None`` as "lead emitted no actionable decision"
    and either retries or falls back to the static pipeline.
    """
    payload = _last_json_object(output)
    if not payload:
        return None
    action = str(payload.get("action") or "").strip().lower()
    if action not in {
        "dispatch", "enqueue", "drop_task", "drop_finding", "done",
    }:
        return None
    new_task = payload.get("new_task")
    if new_task is not None and not isinstance(new_task, dict):
        new_task = None
    decision = LeadDecision(
        action=action,
        task_id=str(payload.get("task_id") or "").strip(),
        finding_id=str(payload.get("finding_id") or "").strip(),
        new_task=new_task,
        rationale=str(payload.get("rationale") or "").strip(),
    )
    return decision if decision.is_valid else None


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _render_findings_section(
    findings: list[Finding], *, max_shown: int = 6,
) -> str:
    if not findings:
        return "_(none yet — lead is sequencing initial work)_"
    # Severity-then-status sort so the highest-impact items lead
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    sorted_findings = sorted(
        findings, key=lambda f: (sev_order.get(f.severity, 5), -f.runs_to_confirm),
    )
    lines: list[str] = []
    for f in sorted_findings[:max_shown]:
        claim = (f.claim or "").replace("\n", " ")[:140]
        lines.append(
            f"- `{f.id}` **[{f.severity}/{f.status}]** "
            f"{f.file}:{f.function} — {claim}"
        )
    if len(findings) > max_shown:
        lines.append(f"- ... and {len(findings) - max_shown} more")
    return "\n".join(lines)


def render_lead_prompt(
    *,
    iteration: int,
    max_iterations: int,
    queue: list[BugFinderTask],
    findings: list[Finding],
    user_goal_block: str,
    tokens_remaining: int,
    findings_shown: int = 6,
) -> str:
    return LEAD_USER_TEMPLATE.format(
        iteration=iteration,
        max_iterations=max_iterations,
        findings_count=len(findings),
        tokens_remaining=max(0, tokens_remaining),
        user_goal_block=(user_goal_block or "_(no specific user goal — explore mode)_"),
        queue_summary=render_queue_summary(queue),
        findings_section=_render_findings_section(findings, max_shown=findings_shown),
        findings_shown=findings_shown,
    )


# ---------------------------------------------------------------------------
# Dispatcher interface
# ---------------------------------------------------------------------------


DispatchHandler = Callable[
    [BugFinderTask, list[Finding]],
    Awaitable[tuple[bool, str]],
]
"""Callback that executes one task. Returns ``(success, summary)``.

The handler receives the LIVE findings list (mutable) — appending new
findings, updating an existing finding's status in-place, or removing
one are all valid. The lead reads from the same list on its next
iteration so dispatcher output is immediately visible to the loop.

The orchestrator wires real handlers (run detector / run verifier /
run fixer / etc.) per kind. Unrecognized kinds get a no-op handler
that returns ``(False, "unhandled kind")`` — the lead sees the failure
on its next iteration and adapts.
"""


@dataclass
class LeadRunState:
    """Mutable state shared across lead iterations.

    Surfaces the running ledger (findings landed so far, total tokens
    burned by the lead's decision calls, last decision rationale) so
    each iteration's prompt reflects current reality.
    """

    findings: list[Finding] = field(default_factory=list)
    decisions: list[LeadDecision] = field(default_factory=list)
    tokens_used: int = 0
    iterations: int = 0
    stop_reason: str = ""
    stop_detail: str = ""


@dataclass(frozen=True)
class LeadRunResult:
    """Outcome of one full lead loop."""

    state: LeadRunState
    findings: list[Finding]

    @property
    def succeeded(self) -> bool:
        return self.state.stop_reason in {"done", "queue_empty"}


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def run_lead_loop(
    *,
    model: str,
    backend,
    queue: TaskQueue,
    run_id: str,
    user_id: str,
    user_goal_block: str,
    initial_findings: list[Finding],
    dispatchers: dict[str, DispatchHandler],
    budget: SubagentBudget = DEFAULT_LEAD_BUDGET,
    event_emit: Callable[[str, dict[str, Any], bool], None] | None = None,
    progress_callback=None,
) -> LeadRunResult:
    """Drive the lead's decision loop until done / budget / queue empty.

    Each iteration: snapshot queue+findings → LLM call returns a JSON
    decision → Python executes it → repeat.

    The loop terminates when:
      * the lead emits ``action=done``,
      * the queue has no actionable tasks AND no enqueue action was
        attempted in the previous iteration (forward progress check),
      * iteration / token budget exceeded,
      * the lead emits 3 consecutive unparseable decisions
        (degenerate-prompt safety),
      * an exception bubbles out of a dispatch handler.

    Returns a ``LeadRunResult`` carrying the accumulated findings + a
    ``LeadRunState`` summarizing what happened. Callers fold the
    findings into the run report and surface ``stop_reason`` in notes.
    """
    state = LeadRunState(findings=list(initial_findings))
    consecutive_parse_failures = 0
    started = time.monotonic()

    def _emit(kind: str, payload: dict[str, Any]) -> None:
        if event_emit is None:
            return
        try:
            event_emit(kind, {"run_id": run_id, **payload}, False)
        except Exception:  # noqa: BLE001 — sink isolation
            log.debug("bug_finder_lead_emit_failed", kind=kind, exc_info=True)

    while state.iterations < budget.max_iterations:
        state.iterations += 1

        if time.monotonic() - started > budget.max_wallclock_seconds:
            state.stop_reason = "wallclock"
            break
        if state.tokens_used >= budget.max_tokens:
            state.stop_reason = "tokens"
            break

        all_tasks = await queue.list_tasks(run_id=run_id, user_id=user_id)
        pending_actionable = any(
            t.status == TaskStatus.PENDING.value for t in all_tasks
        )

        prompt = render_lead_prompt(
            iteration=state.iterations,
            max_iterations=budget.max_iterations,
            queue=all_tasks,
            findings=state.findings,
            user_goal_block=user_goal_block,
            tokens_remaining=max(0, budget.max_tokens - state.tokens_used),
        )

        spec = SubagentSpec(
            role=Role.LEAD.value,
            model=model,
            system_prompt=LEAD_SYSTEM_PROMPT,
            initial_user_message=prompt,
            tools=(),                   # lead has no direct tools — its
                                        # tools are the dispatchers it
                                        # invokes via decisions
            budget=SubagentBudget(
                max_iterations=2,       # single LLM turn per decision
                max_wallclock_seconds=120,
                max_tokens=10_000,
            ),
            instance_id=f"lead_iter_{state.iterations}",
            progress_callback=progress_callback,
            temperature=0.0,
        )
        try:
            sub_result = await run_subagent(spec, backend=backend)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "bug_finder_lead_call_failed",
                iteration=state.iterations, error=str(exc),
            )
            state.stop_reason = "error"
            state.stop_detail = f"{type(exc).__name__}: {exc}"[:200]
            break

        state.tokens_used += sub_result.tokens_in + sub_result.tokens_out
        decision = parse_lead_decision(sub_result.output)

        if decision is None:
            consecutive_parse_failures += 1
            _emit("lead_decision", {
                "iteration": state.iterations,
                "action": "unparseable",
                "rationale": "no fenced JSON / invalid schema",
                "tokens_used": state.tokens_used,
            })
            log.warning(
                "bug_finder_lead_unparseable",
                iteration=state.iterations,
                consecutive=consecutive_parse_failures,
            )
            if consecutive_parse_failures >= 3:
                state.stop_reason = "parse_failure"
                break
            continue
        consecutive_parse_failures = 0
        state.decisions.append(decision)

        _emit("lead_decision", {
            "iteration": state.iterations,
            "action": decision.action,
            "task_id": decision.task_id,
            "finding_id": decision.finding_id,
            "rationale": decision.rationale,
            "tokens_used": state.tokens_used,
        })

        if decision.is_terminal:
            state.stop_reason = "done"
            state.stop_detail = decision.rationale
            break

        # Execute the decision via Python dispatchers.
        try:
            await _execute_decision(
                decision, queue=queue, run_id=run_id, user_id=user_id,
                dispatchers=dispatchers, findings=state.findings,
                emit=_emit,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "bug_finder_lead_dispatch_failed",
                iteration=state.iterations, action=decision.action,
                error=str(exc),
            )
            state.stop_reason = "dispatch_error"
            state.stop_detail = f"{type(exc).__name__}: {exc}"[:200]
            break

        # If we just acted on the only pending task and the lead didn't
        # enqueue anything new, peek again: empty queue + done-eligible
        # state = wrap up.
        if not pending_actionable and decision.action != "enqueue":
            state.stop_reason = "queue_empty"
            break
    else:
        state.stop_reason = state.stop_reason or "iterations"

    log.info(
        "bug_finder_lead_complete",
        run_id=run_id, iterations=state.iterations,
        tokens=state.tokens_used, stop_reason=state.stop_reason,
        decisions=len(state.decisions), findings=len(state.findings),
    )
    return LeadRunResult(state=state, findings=state.findings)


async def _execute_decision(
    decision: LeadDecision,
    *,
    queue: TaskQueue,
    run_id: str,
    user_id: str,
    dispatchers: dict[str, DispatchHandler],
    findings: list[Finding],
    emit: Callable[[str, dict[str, Any]], None],
) -> None:
    """Run the side-effect side of one lead decision."""
    if decision.action == "dispatch":
        # Find the task by id, hand to the matching dispatcher.
        all_tasks = await queue.list_tasks(run_id=run_id, user_id=user_id)
        task = next(
            (t for t in all_tasks if t.task_id == decision.task_id),
            None,
        )
        if task is None:
            log.warning(
                "bug_finder_lead_unknown_task",
                task_id=decision.task_id,
            )
            return
        if not task.is_pending:
            log.debug(
                "bug_finder_lead_dispatch_skipped_nonpending",
                task_id=decision.task_id, status=task.status,
            )
            return
        handler = dispatchers.get(task.kind)
        if handler is None:
            await queue.mark_failed(
                task.task_id, user_id=user_id,
                reason=f"no dispatcher for kind={task.kind}",
            )
            return
        # Atomically take it (flips to in_progress)
        taken = await queue.take_next(
            run_id=run_id, user_id=user_id,
        )
        # take_next returns the highest-priority pending — not always
        # our target. If they differ, we still need to flip OURS via
        # a fresh take. Pragmatic compromise: only proceed if take_next
        # gave us this task; else mark and retry next iter.
        if taken is None or taken.task_id != task.task_id:
            log.debug(
                "bug_finder_lead_take_mismatch",
                wanted=task.task_id,
                got=(taken.task_id if taken else None),
            )
            # The taken task is in_progress now — leave it; lead will
            # see it and decide again.
            return
        ok, summary = await handler(taken, findings)
        if ok:
            await queue.mark_completed(
                taken.task_id, user_id=user_id, result_summary=summary,
            )
        else:
            await queue.mark_failed(
                taken.task_id, user_id=user_id, reason=summary,
            )
        emit("lead_dispatch_complete", {
            "task_id": taken.task_id, "kind": taken.kind,
            "ok": ok, "summary": summary,
        })
        return

    if decision.action == "enqueue":
        nt = decision.new_task or {}
        kind = str(nt.get("kind") or "").strip()
        target = nt.get("target") or {}
        if not kind or not isinstance(target, dict):
            return
        try:
            kind_enum = TaskKind(kind)
        except ValueError:
            log.warning("bug_finder_lead_unknown_kind", kind=kind)
            return
        await queue.enqueue(
            run_id=run_id, user_id=user_id,
            kind=kind_enum, target=target,
            reason=str(nt.get("reason") or "lead-enqueued"),
            priority=int(nt.get("priority") or 5),
            created_by="lead",
        )
        return

    if decision.action == "drop_task":
        await queue.mark_dropped(
            decision.task_id, user_id=user_id,
            reason=decision.rationale or "lead drop",
        )
        return

    if decision.action == "drop_finding":
        # Remove the finding from the running list. Critique pass.
        for i, f in enumerate(findings):
            if f.id == decision.finding_id:
                findings.pop(i)
                break
        return


def is_implemented() -> bool:
    """Used by the orchestrator to branch on lead availability. Flips
    to True once this module ships its decision loop — which it now
    does. The orchestrator may still skip the lead in ``explore`` mode."""
    return True
