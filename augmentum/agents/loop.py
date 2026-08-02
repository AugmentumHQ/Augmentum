"""Bounded subagent loop primitive.

A subagent is a one-shot model→tool→model loop with isolated context
(no parent ``turn_summaries``, no sticky reminder, no plan/act phase
machinery) and three independent stop conditions:

* the model emits no tool calls (``complete``);
* the ``BudgetTracker`` trips (``budget``);
* the ``StuckDetector`` trips (``stuck``).

Tool calls are dispatched through a *guard* (see ``guards.py``) that
implements the reward-hacking defenses before reaching the tool itself.

Lifted from ``bug_finder/subagent.py`` in 2026-05-31. Now generic
across bug_finder and coder subagent consumers.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from augmentum.agents.budget import BudgetTracker, SubagentBudget
from augmentum.agents.guards import ToolGuard
from augmentum.agents.stuck import StuckDetector, Turn
from augmentum.agents.verify import SubagentVerdict, judge_subagent_result
from augmentum.models.base import (
    InternalChatRequest,
    Message,
    ModelBackend,
)
from augmentum.tools.base import Tool, invoke_tool
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_STUCK_HASH_WINDOW = 512


@dataclass(frozen=True)
class SubagentProgress:
    """One snapshot of a subagent's inner loop state.

    Emitted at iteration + tool boundaries so the parent's UI can
    render a live activity feed instead of a blank "running…" card.
    Cheap to construct — the loop emits one per tool call and one per
    model response. Fields are flat strings/ints so chat_egress can
    serialize without bespoke conversion.
    """

    instance_id: str
    role: str
    iteration: int
    phase: str
    """One of: ``responding`` (model just produced output),
    ``tool_call`` (about to execute a tool), ``tool_result`` (result
    landed), ``stuck`` (stuck detector tripped), ``done`` (loop
    exited). The UI uses this to pick an icon/colour for the row."""
    tool_name: str = ""
    """For ``tool_call`` / ``tool_result``: which tool. Empty
    otherwise."""
    text_preview: str = ""
    """First ~200 chars of the most recent model thought or tool
    result. Truncated by the loop, NOT the UI, so chat_egress payloads
    stay bounded."""
    tokens_in: int = 0
    tokens_out: int = 0
    wallclock_ms: int = 0


SubagentProgressCallback = Callable[[SubagentProgress], Awaitable[None]]


@dataclass(frozen=True)
class SubagentSpec:
    """Description of a single subagent run.

    Build one of these for each fan-out, dispatch via ``run_subagent``.
    """

    role: str
    """Role label — used for logging + result metadata; the runtime
    doesn't branch on it."""

    model: str
    """Resolved model id to invoke (post-fallback resolution)."""

    system_prompt: str
    initial_user_message: str

    tools: tuple[Tool, ...]
    """The exact tool set this subagent may invoke. Filtered upstream
    by the dispatcher via ``filter_tools()``."""

    budget: SubagentBudget
    tool_guard: ToolGuard | None = None
    temperature: float | None = None

    enable_thinking: bool | None = None
    """When set, forwarded to llama-server as
    ``chat_template_kwargs.enable_thinking``. Qwen 3.x / GLM-4.x /
    EXAONE 4.x / Nemotron 3 Nano use this to toggle chain-of-thought.
    None = let the model's template default apply."""

    preserve_thinking: bool | None = None
    """When True, the model's ``<think>`` traces are kept across
    multi-turn history (Qwen 3.6 ``preserve_thinking`` chat-template
    kwarg). Otherwise reasoning blocks are stripped from prior
    assistant messages before the next request. None defers to the
    backend default."""

    instance_id: str = field(default="")
    """Optional caller-supplied id for log correlation. If empty the
    runtime mints one."""

    verify: bool = False
    """When True AND ``success_criteria`` is non-empty, an independent
    judge call (``agents/verify.py``) checks the subagent's final report
    against each criterion before its ``complete`` stop is honored. A
    failed verdict re-enters the loop (up to ``verify_max_reentry``) with
    the unmet criteria injected; on exhaustion the result is returned with
    ``verification="failed"`` so the parent sees what's missing instead of
    trusting a confidently-wrong report. Set by the dispatcher from
    ``coder_subagent_verify_enabled`` + the presence of criteria."""

    task_prompt: str = ""
    """The lead's raw delegation prompt (without the orientation /
    workspace-facts wrapper). Handed to the verification judge as the
    ``<task>`` so it judges intent, not the bridged framing."""

    success_criteria: tuple[str, ...] = ()
    """Definition-of-done the lead handed down. Already rendered into
    ``initial_user_message`` for the subagent to read; carried here too so
    the verification judge can check the output against it structurally."""

    verify_max_reentry: int = 1
    """How many times a failed verification re-enters the loop before the
    stop is honored unconditionally. Leaf-node default of 1 (vs the lead
    goal-judge's 2): cheaper for the parent to re-dispatch than for a
    subagent to thrash."""

    progress_callback: SubagentProgressCallback | None = None
    """Optional async callback fired on each inner-loop milestone
    (responding / tool_call / tool_result / stuck / done). Set by the
    dispatcher to bridge subagent activity into the parent's chat
    egress so the UI can render a live activity feed. Callback errors
    are caught and logged — never propagated into the loop, because a
    misbehaving sink shouldn't kill the subagent."""


@dataclass
class ToolCallLog:
    """One row in the subagent's tool-call ledger."""

    iteration: int
    tool: str
    args: dict[str, Any]
    outcome: str  # success | failure | denied | unavailable | exception | argerror
    reason: str = ""
    output_len: int = 0
    elapsed_ms: int = 0


@dataclass
class SubagentResult:
    """Outcome of a single subagent run."""

    role: str
    instance_id: str
    output: str
    tokens_in: int
    tokens_out: int
    wallclock_ms: int
    iterations: int
    tool_calls: int
    stop_reason: str  # complete | budget | stuck | error | cancelled
    stop_detail: str = ""
    stuck_pattern: str | None = None
    tool_call_log: list[ToolCallLog] = field(default_factory=list)
    model_resolved: str = ""
    """The actual model id that ran (for spec.model overrides + fallback chains)."""

    recovery_hint: str = ""
    """Typed guidance for the parent loop / model based on stop_reason.
    Empty when ``stop_reason == "complete"`` AND verification didn't fail.
    Populated by ``_compute_recovery_hint`` so weak parent models get
    explicit instructions ("the subagent looped on repeated reads — narrow
    the scope or try a different role") instead of a generic error string.
    This is the difference between "the subagent failed and the lead burns
    5 iterations guessing why" and "the lead does the right thing on its
    first follow-up."""

    verification: str = "unchecked"
    """Outcome of the success-criteria verification gate:
    ``unchecked`` (gate disabled / no criteria), ``passed``, ``failed``
    (criteria unmet after re-entries exhausted), or ``error`` (judge gave
    no signal — failed open, treat as unchecked for trust). Persisted +
    surfaced so the lead and the UI can tell a verified completion from an
    unverified one."""

    verification_reason: str = ""
    """The judge's one-line reason when ``verification`` is ``failed`` —
    what's concretely missing. Empty otherwise."""

    verification_unmet: list[str] = field(default_factory=list)
    """The specific success criteria the judge found unsatisfied. Empty
    unless ``verification == "failed"``."""


def _compact_args(args: dict[str, Any]) -> str:
    """Stable string form of a tool-call args dict for the stuck hash."""
    try:
        return json.dumps(args, sort_keys=True, default=str)[:_STUCK_HASH_WINDOW]
    except (TypeError, ValueError):
        return str(args)[:_STUCK_HASH_WINDOW]


_PROGRESS_PREVIEW_CHARS = 200

# Subagent context compaction. Without this the loop re-sends the full,
# ever-growing transcript every iteration, so the cumulative max_tokens
# budget trips on the SUM of re-sent histories — a 7-step explore reading
# a handful of files reports ~80k "tokens" and dies with partial output
# even though its live context never exceeded ~15k. Mirrors the main
# coder loop's compact_conversation_messages, but self-contained on
# Message objects and triggered off the MEASURED prompt size
# (resp.usage.prompt_tokens) so it needs no tokenizer.
_SUBAGENT_KEEP_RECENT = 6          # most-recent messages kept verbatim (~3 tool rounds)
_SUBAGENT_SUMMARY_MAX = 2_000      # char cap on the synthetic summary note
_SUBAGENT_COMPACT_FRACTION = 0.4   # compact once a prompt exceeds this fraction of max_tokens
_SUBAGENT_COMPACT_FLOOR = 8_000    # ...but never below this absolute prompt size


def _compact_subagent_messages(
    messages: list[Message],
    *,
    keep_recent: int = _SUBAGENT_KEEP_RECENT,
    examined: list[str] | None = None,
) -> list[Message] | None:
    """Collapse the middle of a subagent transcript into one summary note.

    Preserves the system message + the initial user message (task +
    success_criteria + workspace facts) and the most recent
    ``keep_recent`` messages; the span between is replaced by a single
    user note listing what was already examined, so the subagent doesn't
    re-read the same files. Returns the new list, or ``None`` when
    there's nothing worth compacting (caller skips the swap).

    ``examined`` is the loop-owned CUMULATIVE examined-labels list,
    mutated in place. Before 2026-07-06 each pass rebuilt the list from
    only the span it was dropping — on a second compaction the previous
    summary note (a plain user message, no tool_calls to mine) was
    itself in the dropped span, so everything the first pass recorded
    was forgotten and the model re-read those files. Passing the same
    list across passes keeps the note honest for the whole run.

    Tool-call integrity: the kept tail is advanced forward past any
    leading ``role="tool"`` messages so it never starts with a tool
    result orphaned from the assistant tool_call that requested it —
    some backends 400 on that.
    """
    n = len(messages)
    if n <= keep_recent + 3:
        return None
    head = messages[:2]  # system + initial user
    tail_start = n - keep_recent
    while tail_start < n and messages[tail_start].role == "tool":
        tail_start += 1
    if tail_start >= n:
        return None
    dropped = messages[2:tail_start]
    if len(dropped) < 2:
        return None

    # Summarize what the dropped span DID — chiefly which files/queries
    # it already touched, so the model doesn't waste budget re-reading.
    # Accumulates into the caller's list so labels survive later passes.
    seen: list[str] = examined if examined is not None else []
    for m in dropped:
        for tc in (m.tool_calls or []):
            _id, name, args = _parse_tool_call(tc)
            ref = args.get("path") or args.get("pattern") or args.get("query") or ""
            label = f"{name}({ref})" if ref else name
            if label and label not in seen:
                seen.append(label)
    seen_text = ", ".join(seen[:60]) if seen else "(various reads)"
    note = (
        f"[Context compacted: {len(dropped)} earlier messages summarized to "
        f"stay within budget. Already examined: {seen_text}. Re-read a "
        f"specific file only if you still need its exact contents.]"
    )
    summary = Message(role="user", content=note[:_SUBAGENT_SUMMARY_MAX])
    return head + [summary] + messages[tail_start:]


def _compute_recovery_hint(
    *,
    stop_reason: str,
    stop_detail: str,
    stuck_pattern: str | None,
    role: str,
    iterations: int,
    instance_id: str,
    verification: str = "",
    verification_unmet: list[str] | None = None,
    verification_reason: str = "",
) -> str:
    """Translate a stop_reason into actionable parent-model guidance.

    The parent loop reads this verbatim and includes it in the tool
    result so a weak lead doesn't have to infer "what should I do
    differently?" from a bare ``stop_reason="stuck"``. Each branch
    names the failure mode AND the next move — the same pattern as
    Tool.error_hints, scaled up to subagent results.

    Returns empty string on a clean, verified completion. A completion
    whose verification FAILED still produces a hint — the subagent
    stopped cleanly but didn't satisfy the contract, and the lead must
    know not to trust the report at face value."""
    if stop_reason == "complete":
        if verification == "failed":
            unmet = verification_unmet or []
            unmet_block = (
                "\n".join(f"  - {u}" for u in unmet[:8])
                if unmet else "  (criteria unmet — see the report)"
            )
            return (
                f"Subagent `{role}` finished but an independent verification "
                f"judge found its success criteria NOT satisfied"
                + (f": {verification_reason}" if verification_reason else "")
                + ".\nUnmet criteria:\n" + unmet_block + "\n"
                "Do NOT trust the report as a completed result. Next move: "
                "re-dispatch with a sharper prompt that targets the unmet "
                "criteria directly, finish the remaining work inline, or — if "
                "you judge the criteria were wrong/over-strict — proceed "
                "knowingly. The subagent's partial output above may still be "
                "useful as a starting point."
            )
        return ""
    if stop_reason == "budget":
        return (
            f"Subagent `{role}` exhausted its budget after "
            f"{iterations} iterations ({stop_detail or 'cap reached'}). "
            "The task is too large for this role's budget. Next move: "
            "split it into 2-3 narrower subtasks and dispatch each "
            "separately, OR pick a role with a larger budget, OR raise "
            "the role's `budget.max_iterations` in its frontmatter. "
            "Do NOT re-dispatch the same role on the same prompt — it "
            "will hit the same cap."
        )
    if stop_reason == "stuck":
        pattern = stuck_pattern or "unknown_pattern"
        return (
            f"Subagent `{role}` looped on pattern `{pattern}` and the "
            "stuck detector intervened. Common causes per pattern: "
            "REPEATED_TOOL_CALLS = same args N×, the underlying tool "
            "is wrong or unavailable; REPEATED_OBSERVATIONS = stale "
            "filesystem read; REPEATED_ERRORS = malformed args. Next "
            "move: re-dispatch with a sharper prompt that addresses "
            "the loop directly (e.g., \"Don't re-read /workspace/foo.py; "
            "use the snippet I gave you\"), pick a more capable model "
            "via the `model` field, or do the work inline."
        )
    if stop_reason == "cancelled":
        return (
            f"Subagent `{role}` was cancelled mid-flight by the user "
            "or parent. The partial work shown above may be incomplete. "
            "If you still need the result, re-dispatch with a tighter "
            "scope so it completes faster, or do the work inline."
        )
    if stop_reason == "error":
        return (
            f"Subagent `{role}` failed with a backend error: "
            f"{stop_detail or 'unspecified'}. This is usually a model "
            "API issue (rate limit, transient 5xx) or a misconfigured "
            "backend. Next move: retry once (transient errors clear), "
            "switch to a different `model` via @provider override, or "
            "do the work inline. Inspect run `" + instance_id +
            "` via /api/coder/subagents/" + instance_id + " for full "
            "tool-call log."
        )
    return ""


def _summarize_tool_log(entries: list[ToolCallLog], *, limit: int = 30) -> str:
    """Compact one-line-per-call digest of what the subagent actually DID,
    for the verification judge's ``<subagent_tool_activity>`` block. Carries
    the load-bearing signal — which files were written, whether tests ran and
    passed — that distinguishes a real completion from a claimed one. Last
    ``limit`` entries only, so a long run doesn't blow the judge's prompt."""
    if not entries:
        return ""
    lines: list[str] = []
    for e in entries[-limit:]:
        ref = ""
        if isinstance(e.args, dict):
            for k in ("path", "pattern", "query", "cmd", "command", "file_path"):
                v = e.args.get(k)
                if v:
                    ref = f" {str(v)[:80]}"
                    break
        tail = f" — {e.reason[:80]}" if e.reason else ""
        lines.append(f"{e.tool}{ref} → {e.outcome}{tail}")
    return "\n".join(lines)


def _verify_reentry_note(verdict: SubagentVerdict) -> str:
    """The synthetic user turn injected when verification fails and the
    subagent still has a re-entry left — names the unmet criteria and asks
    for the concrete work, with an explicit honest-exit clause so the model
    can declare a criterion impossible rather than fake it."""
    unmet = "\n".join(f"- {u}" for u in verdict.unmet[:8]) or "- (see reason below)"
    reason = f"Reason: {verdict.reason}\n" if verdict.reason else ""
    return (
        "[Verification: your work does not yet satisfy these success "
        "criteria:\n" + unmet + "\n" + reason +
        "Address them now: take the concrete actions needed, then report "
        "again. If a criterion is genuinely impossible or out of scope, say "
        "so explicitly and explain why — do not silently skip it.]"
    )


async def _safe_emit_progress(
    callback: SubagentProgressCallback | None,
    progress: SubagentProgress,
) -> None:
    """Call the progress callback if set; swallow + log any exception
    so a misbehaving sink never breaks the inner loop. The sink is
    advisory — the subagent must finish its work regardless."""
    if callback is None:
        return
    try:
        await callback(progress)
    except Exception:
        log.warning(
            "subagent_progress_emit_failed",
            instance_id=progress.instance_id,
            phase=progress.phase,
            exc_info=True,
        )


def _tool_to_schema(tool: Tool) -> dict[str, Any]:
    """OpenAI-style function schema for ``tool``."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _parse_tool_call(tc: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Decompose a raw tool_calls entry into ``(call_id, name, args)``."""
    call_id = str(tc.get("id") or "")
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
    if fn is not None:
        name = str(fn.get("name") or "")
        args_raw = fn.get("arguments")
    else:
        name = str(tc.get("name") or "")
        args_raw = tc.get("arguments")

    if isinstance(args_raw, str):
        args_str = args_raw.strip()
        if not args_str:
            return call_id, name, {}
        try:
            parsed = json.loads(args_str)
            return call_id, name, parsed if isinstance(parsed, dict) else {"_value": parsed}
        except json.JSONDecodeError:
            return call_id, name, {"_raw": args_str}
    if isinstance(args_raw, dict):
        return call_id, name, args_raw
    return call_id, name, {}


async def run_subagent(spec: SubagentSpec, *, backend: ModelBackend) -> SubagentResult:
    """Run a bounded inner agent loop.

    Terminates on exactly one of:
      * model emits no tool calls (``stop_reason="complete"``);
      * budget tracker trips (``stop_reason="budget"``);
      * stuck detector trips (``stop_reason="stuck"``);
      * backend raises (``stop_reason="error"``).

    All four cases produce a populated ``SubagentResult`` — no raise
    path under normal operation. Callers treat the result like any
    other tool output.
    """
    instance_id = spec.instance_id or f"sa_{spec.role}_{uuid.uuid4().hex[:8]}"
    started_at = time.monotonic()

    tool_map = {t.name: t for t in spec.tools}
    tool_schemas = [_tool_to_schema(t) for t in spec.tools]

    messages: list[Message] = [
        Message(role="system", content=spec.system_prompt),
        Message(role="user", content=spec.initial_user_message),
    ]

    detector = StuckDetector()
    tracker = BudgetTracker(spec.budget)
    tool_calls_total = 0
    iteration = 0
    log_entries: list[ToolCallLog] = []
    last_thought = ""
    verify_reentries = 0
    examined_labels: list[str] = []  # cumulative across compaction passes

    def _build_result(
        stop_reason: str,
        *,
        output: str = "",
        detail: str = "",
        stuck_pattern: str | None = None,
        verification: str = "unchecked",
        verification_reason: str = "",
        verification_unmet: list[str] | None = None,
    ) -> SubagentResult:
        recovery = _compute_recovery_hint(
            stop_reason=stop_reason,
            stop_detail=detail,
            stuck_pattern=stuck_pattern,
            role=spec.role,
            iterations=tracker.iterations,
            instance_id=instance_id,
            verification=verification,
            verification_unmet=verification_unmet,
            verification_reason=verification_reason,
        )
        return SubagentResult(
            role=spec.role,
            instance_id=instance_id,
            output=output,
            tokens_in=tracker.tokens_in,
            tokens_out=tracker.tokens_out,
            wallclock_ms=int((time.monotonic() - started_at) * 1000),
            iterations=tracker.iterations,
            tool_calls=tool_calls_total,
            stop_reason=stop_reason,
            stop_detail=detail,
            stuck_pattern=stuck_pattern,
            tool_call_log=log_entries,
            model_resolved=spec.model,
            recovery_hint=recovery,
            verification=verification,
            verification_reason=verification_reason,
            verification_unmet=list(verification_unmet or []),
        )

    def _wallclock_ms() -> int:
        return int((time.monotonic() - started_at) * 1000)

    while True:
        exhausted, reason = tracker.exhausted()
        if exhausted:
            log.info(
                "subagent_budget_exhausted",
                instance_id=instance_id,
                role=spec.role,
                reason=reason,
                iterations=tracker.iterations,
                tokens=tracker.tokens_total,
            )
            return _build_result("budget", detail=reason or "")

        iteration += 1

        template_kwargs: dict | None = None
        if spec.enable_thinking is not None:
            template_kwargs = {"enable_thinking": bool(spec.enable_thinking)}
        req = InternalChatRequest(
            model=spec.model,
            messages=list(messages),
            tools=tool_schemas if tool_schemas else None,
            temperature=spec.temperature,
            chat_template_kwargs=template_kwargs,
            preserve_thinking=spec.preserve_thinking,
        )

        try:
            resp = await backend.chat(req)
        except Exception as exc:
            log.warning(
                "subagent_backend_error",
                instance_id=instance_id,
                role=spec.role,
                exc_info=True,
            )
            return _build_result(
                "error",
                detail=f"{type(exc).__name__}: {exc}"[:256],
            )

        messages.append(resp.message)
        tracker.record_iteration(
            tokens_in=resp.usage.prompt_tokens,
            tokens_out=resp.usage.completion_tokens,
        )
        last_thought = resp.message.content or ""

        # Emit ``responding`` checkpoint — model response just landed,
        # before we dispatch any tool calls. Carries the latest thought
        # so the UI can show a streamed-style "thinking" line even
        # though the inner loop itself isn't token-streaming.
        await _safe_emit_progress(spec.progress_callback, SubagentProgress(
            instance_id=instance_id,
            role=spec.role,
            iteration=iteration,
            phase="responding",
            text_preview=last_thought[:_PROGRESS_PREVIEW_CHARS],
            tokens_in=tracker.tokens_in,
            tokens_out=tracker.tokens_out,
            wallclock_ms=_wallclock_ms(),
        ))

        tool_calls = resp.message.tool_calls or []
        if not tool_calls:
            # Verification gate. Before honoring a clean stop, an
            # independent judge checks the final report against the lead's
            # success_criteria. A failed verdict re-enters the loop (bounded
            # by verify_max_reentry) with the unmet criteria injected; on
            # exhaustion we honor the stop but mark verification="failed" so
            # the parent doesn't trust a confidently-wrong report. The judge
            # fails OPEN (verdict.ok is None → verification="error") so a
            # flaky judge never traps the subagent. Only fires when the lead
            # actually handed down criteria — a read-only explore with no
            # contract stays a single cheap pass.
            verification_label = "unchecked"
            if spec.verify and spec.success_criteria:
                await _safe_emit_progress(spec.progress_callback, SubagentProgress(
                    instance_id=instance_id,
                    role=spec.role,
                    iteration=iteration,
                    phase="verifying",
                    text_preview=f"checking {len(spec.success_criteria)} success criteria",
                    tokens_in=tracker.tokens_in,
                    tokens_out=tracker.tokens_out,
                    wallclock_ms=_wallclock_ms(),
                ))
                verdict = await judge_subagent_result(
                    backend,
                    model=spec.model,
                    task=spec.task_prompt or spec.initial_user_message,
                    success_criteria=spec.success_criteria,
                    output=last_thought,
                    tool_summary=_summarize_tool_log(log_entries),
                )
                if verdict.ok is False and verify_reentries < spec.verify_max_reentry:
                    verify_reentries += 1
                    log.info(
                        "subagent_verify_reentry",
                        instance_id=instance_id,
                        role=spec.role,
                        attempt=verify_reentries,
                        unmet=len(verdict.unmet),
                    )
                    messages.append(Message(
                        role="user", content=_verify_reentry_note(verdict),
                    ))
                    continue
                if verdict.ok is False:
                    log.info(
                        "subagent_verify_failed",
                        instance_id=instance_id,
                        role=spec.role,
                        unmet=len(verdict.unmet),
                        reentries=verify_reentries,
                    )
                    await _safe_emit_progress(spec.progress_callback, SubagentProgress(
                        instance_id=instance_id,
                        role=spec.role,
                        iteration=iteration,
                        phase="done",
                        text_preview=("verification failed: " + (verdict.reason or ""))[:_PROGRESS_PREVIEW_CHARS],
                        tokens_in=tracker.tokens_in,
                        tokens_out=tracker.tokens_out,
                        wallclock_ms=_wallclock_ms(),
                    ))
                    return _build_result(
                        "complete",
                        output=last_thought,
                        verification="failed",
                        verification_reason=verdict.reason,
                        verification_unmet=list(verdict.unmet),
                    )
                # ok is True → "passed"; ok is None → "error" (failed open).
                verification_label = verdict.label

            await _safe_emit_progress(spec.progress_callback, SubagentProgress(
                instance_id=instance_id,
                role=spec.role,
                iteration=iteration,
                phase="done",
                text_preview=last_thought[:_PROGRESS_PREVIEW_CHARS],
                tokens_in=tracker.tokens_in,
                tokens_out=tracker.tokens_out,
                wallclock_ms=_wallclock_ms(),
            ))
            return _build_result(
                "complete", output=last_thought, verification=verification_label,
            )

        for tc in tool_calls:
            tool_calls_total += 1
            call_started = time.monotonic()
            call_id, name, args = _parse_tool_call(tc)

            # Loud progress on each tool call so the UI can render
            # "iteration 3 — calling code_grep" before the call runs.
            # Carrying the args text-preview lets the user see at a
            # glance what the subagent is doing.
            await _safe_emit_progress(spec.progress_callback, SubagentProgress(
                instance_id=instance_id,
                role=spec.role,
                iteration=iteration,
                phase="tool_call",
                tool_name=name,
                text_preview=_compact_args(args)[:_PROGRESS_PREVIEW_CHARS],
                tokens_in=tracker.tokens_in,
                tokens_out=tracker.tokens_out,
                wallclock_ms=_wallclock_ms(),
            ))

            guard_reason: str | None = None
            if spec.tool_guard is not None and name:
                try:
                    guard_reason = spec.tool_guard(name, args)
                except Exception:
                    log.warning(
                        "subagent_guard_raised",
                        instance_id=instance_id,
                        tool=name,
                        exc_info=True,
                    )
                    guard_reason = "tool guard raised; denying for safety"

            if guard_reason:
                err = f"command denied by policy: {guard_reason}"
                messages.append(Message(role="tool", content=err, tool_call_id=call_id))
                detector.record(Turn(
                    tool=name,
                    content=_compact_args(args),
                    thought=last_thought,
                    error=err,
                ))
                log_entries.append(ToolCallLog(
                    iteration=iteration,
                    tool=name,
                    args=args,
                    outcome="denied",
                    reason=guard_reason,
                    elapsed_ms=int((time.monotonic() - call_started) * 1000),
                ))
                continue

            tool = tool_map.get(name)
            if tool is None:
                err = f"tool {name!r} not available for role {spec.role}"
                messages.append(Message(role="tool", content=err, tool_call_id=call_id))
                detector.record(Turn(
                    tool=name,
                    content=_compact_args(args),
                    thought=last_thought,
                    error=err,
                ))
                log_entries.append(ToolCallLog(
                    iteration=iteration,
                    tool=name,
                    args=args,
                    outcome="unavailable",
                    reason="tool not in allowed set",
                    elapsed_ms=int((time.monotonic() - call_started) * 1000),
                ))
                continue

            try:
                result = await invoke_tool(tool, args)
            except TypeError as exc:
                err = f"invalid arguments: {exc}"
                messages.append(Message(role="tool", content=err, tool_call_id=call_id))
                detector.record(Turn(
                    tool=name,
                    content=_compact_args(args),
                    thought=last_thought,
                    error=err,
                ))
                log_entries.append(ToolCallLog(
                    iteration=iteration,
                    tool=name,
                    args=args,
                    outcome="argerror",
                    reason=str(exc)[:200],
                    elapsed_ms=int((time.monotonic() - call_started) * 1000),
                ))
                continue
            except Exception as exc:
                err = f"tool exception: {type(exc).__name__}: {exc}"
                messages.append(Message(role="tool", content=err, tool_call_id=call_id))
                detector.record(Turn(
                    tool=name,
                    content=_compact_args(args),
                    thought=last_thought,
                    error=err,
                ))
                log_entries.append(ToolCallLog(
                    iteration=iteration,
                    tool=name,
                    args=args,
                    outcome="exception",
                    reason=f"{type(exc).__name__}: {exc}"[:200],
                    elapsed_ms=int((time.monotonic() - call_started) * 1000),
                ))
                log.warning(
                    "subagent_tool_exception",
                    instance_id=instance_id,
                    tool=name,
                    exc_info=True,
                )
                continue

            content = (result.output or result.error or "")[:_STUCK_HASH_WINDOW * 16]
            messages.append(Message(
                role="tool",
                content=content or "(empty result)",
                tool_call_id=call_id,
            ))
            detector.record(Turn(
                tool=name,
                content=_compact_args(args),
                thought=last_thought,
                observation=content if result.success else "",
                error="" if result.success else (result.error or content),
            ))
            log_entries.append(ToolCallLog(
                iteration=iteration,
                tool=name,
                args=args,
                outcome="success" if result.success else "failure",
                reason=("" if result.success else (result.error or "")[:200]),
                output_len=len(content),
                elapsed_ms=int((time.monotonic() - call_started) * 1000),
            ))

            await _safe_emit_progress(spec.progress_callback, SubagentProgress(
                instance_id=instance_id,
                role=spec.role,
                iteration=iteration,
                phase="tool_result",
                tool_name=name,
                text_preview=content[:_PROGRESS_PREVIEW_CHARS],
                tokens_in=tracker.tokens_in,
                tokens_out=tracker.tokens_out,
                wallclock_ms=_wallclock_ms(),
            ))

        stuck = detector.check()
        if stuck.stuck:
            log.info(
                "subagent_stuck",
                instance_id=instance_id,
                role=spec.role,
                pattern=(stuck.pattern.value if stuck.pattern else None),
            )
            await _safe_emit_progress(spec.progress_callback, SubagentProgress(
                instance_id=instance_id,
                role=spec.role,
                iteration=iteration,
                phase="stuck",
                text_preview=(stuck.detail or "")[:_PROGRESS_PREVIEW_CHARS],
                tokens_in=tracker.tokens_in,
                tokens_out=tracker.tokens_out,
                wallclock_ms=_wallclock_ms(),
            ))
            return _build_result(
                "stuck",
                output=last_thought,
                detail=stuck.detail,
                stuck_pattern=stuck.pattern.value if stuck.pattern else None,
            )

        # Bound the working context before the next iteration. Trigger off
        # the MEASURED size of the prompt we just sent: once it exceeds a
        # fraction of the role's token budget, fold the transcript middle so
        # the next prompt stops growing. This keeps the live context under
        # the model's window AND slows the cumulative-token burn that was
        # tripping the budget mid-exploration. The recent tail + the
        # examined-files note survive, so the model loses raw file dumps,
        # not its train of thought.
        compact_threshold = max(
            _SUBAGENT_COMPACT_FLOOR,
            int(spec.budget.max_tokens * _SUBAGENT_COMPACT_FRACTION),
        )
        if resp.usage.prompt_tokens >= compact_threshold:
            compacted = _compact_subagent_messages(messages, examined=examined_labels)
            if compacted is not None:
                before_n = len(messages)
                messages = compacted
                log.info(
                    "subagent_context_compacted",
                    instance_id=instance_id,
                    role=spec.role,
                    iteration=iteration,
                    prompt_tokens=resp.usage.prompt_tokens,
                    messages_before=before_n,
                    messages_after=len(messages),
                )
                await _safe_emit_progress(spec.progress_callback, SubagentProgress(
                    instance_id=instance_id,
                    role=spec.role,
                    iteration=iteration,
                    phase="compacted",
                    text_preview=(
                        f"compacted {before_n}->{len(messages)} msgs "
                        f"at ~{resp.usage.prompt_tokens} prompt tokens"
                    )[:_PROGRESS_PREVIEW_CHARS],
                    tokens_in=tracker.tokens_in,
                    tokens_out=tracker.tokens_out,
                    wallclock_ms=_wallclock_ms(),
                ))
