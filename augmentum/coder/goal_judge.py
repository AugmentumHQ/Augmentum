"""Goal judge — independent completion check before honoring a stop.

Borrowed from MiMo-Code's ``/goal`` stop-condition gate (reviewed
2026-06-12; their ``session/goal.ts`` + ``prompt.ts`` goalGate): when
the agent claims to be done, an independent judge model reads the
work and returns a structured verdict; "not satisfied" injects the
judge's reason as a synthetic user turn and re-enters the loop. Their
self-reported harness ablation attributes ~5 SWE-bench points to
mechanisms like this — the failure mode it closes is the "optimistic
stop": the model summarizes confidently while the request is half
done.

Augmentum adaptation (vs MiMo):

* Sits ABOVE the heuristic Termination Quality Gate, firing only when
  the TQG already ACCEPTED a stop on a turn that made writes — pure
  Q&A turns and nudge paths never pay the extra call.
* Judges the USER'S REQUEST for the turn (we have no /goal command;
  the latest user input IS the condition).
* Re-entry cap 2 (vs their 12): coder iterations are expensive and
  the TQG + nudge machinery already handles the cheap cases.
* Fail-open on judge error, but LOUDLY (log.warning, not debug) —
  the review note: silent fail-open quietly deletes the guarantee.

Same call shape as ``next_speaker.py`` (the proven second-LLM-call
plumbing): reuse the turn's backend + request so provider/routing carry,
but clear KV slot affinity (``kv_session_key=""``) so this one-shot never
evicts the act loop's warm slot; stream=False, temperature=0, tiny
max_tokens, never raises.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import replace as _dataclass_replace
from typing import TYPE_CHECKING

from augmentum.models.base import InternalChatRequest, Message
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import Backend

log = get_logger(__name__)

# Re-entries per turn. After this many failed judgments the stop is
# honored unconditionally (capped, like MiMo's MAX_GOAL_REACT — a
# judge must never trap the user in a loop).
MAX_JUDGE_REENTRY = 2

_JUDGE_PROMPT = """\
You are a stop-condition judge for a coding agent. The user asked for \
something; the agent now claims to be done. Judge ONLY whether the \
user's request was satisfied — not whether more could theoretically \
be done.

Respond with a JSON object only — no prose, no markdown fences:
{"ok": true, "reason": "<evidence from the work summary that satisfies the request>"}
{"ok": false, "reason": "<what is concretely missing or blocks the request>"}
{"ok": false, "impossible": true, "reason": "<why the request can never be satisfied>"}

Only use "impossible" when the request is genuinely unachievable \
(self-contradictory, depends on an unavailable resource, the agent \
has exhausted reasonable approaches) — never just because progress \
was slow. When in doubt between ok and not-ok over minor polish, \
prefer ok: you exist to catch UNFINISHED work, not to demand \
perfection."""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class GoalVerdict:
    """Judge outcome. ``ok=None`` means no signal (backend/parse
    failure) — callers MUST fail open and honor the stop."""

    ok: bool | None
    impossible: bool = False
    reason: str = ""
    raw_response: str = ""


async def judge_goal_satisfied(
    backend: Backend,
    *,
    source_request: InternalChatRequest,
    user_goal: str,
    final_response: str,
    edited_paths: list[str] | None = None,
    total_writes: int = 0,
    timeout_s: float = 10.0,
) -> GoalVerdict:
    """One judge round-trip. Never raises; ``ok=None`` on failure."""
    goal = (user_goal or "").strip()
    if not goal:
        return GoalVerdict(ok=None)

    paths = ", ".join((edited_paths or [])[:20]) or "(none recorded)"
    summary = (
        f"<user_request>\n{goal[:2000]}\n</user_request>\n\n"
        f"<work_summary>\n"
        f"Files written this turn ({total_writes} writes): {paths}\n"
        f"</work_summary>\n\n"
        f"<agent_final_response>\n{(final_response or '')[:3000]}\n"
        f"</agent_final_response>\n\n"
        f"{_JUDGE_PROMPT}"
    )

    judge_request = _dataclass_replace(
        source_request,
        messages=[Message(role="user", content=summary)],
        # No KV slot affinity: this is a differently-framed one-shot (a
        # 1-message summary), NOT a continuation of the act loop. Inheriting
        # the act request's kv_session_key routes this tiny prompt onto the
        # loop's warm slot and OVERWRITES its KV, forcing the next real turn
        # to re-prefill from cold. Same rationale the plan call already
        # encodes (phase_plan.py). Verified live 2026-07-03 (kv_debug dump:
        # 1-msg judge payload diverging at msg 0 vs the 105-msg loop
        # baseline under a shared session key).
        kv_session_key="",
        stream=False,
        tools=None,
        tool_choice=None,
        temperature=0.0,
        max_tokens=220,
        chat_template_kwargs=None,
        # Force reasoning OFF: this is a deterministic classification, and a
        # reasoning model (Gemini flash-lite / DeepSeek / etc.) would burn the
        # tiny max_tokens in the thought channel and return EMPTY content →
        # ok=None → fail-open → the very premature-stop this judge exists to
        # catch. think=False disables reasoning across providers (openai_compat
        # thinking:{type}, gemini thinkingBudget 0, local enable_thinking=False).
        think=False,
        reasoning_effort=None,
    )

    try:
        response = await backend.chat(judge_request)
    except Exception as exc:  # noqa: BLE001
        # Fail open — but loudly. A flaky judge must never trap the
        # user, AND a silently-failing judge must never masquerade as
        # a working guarantee.
        log.warning(
            "coder_goal_judge_backend_error",
            error=str(exc)[:200],
            model=judge_request.model,
        )
        return GoalVerdict(ok=None)

    from augmentum.models.base import response_text
    # thinking_fallback=True is belt-and-suspenders: if a provider ignores the
    # think=False above and reasons anyway, salvage the verdict from the thought
    # channel rather than reading empty → the completion guarantee survives.
    raw = response_text(response, thinking_fallback=True).strip()
    if not raw:
        log.warning("coder_goal_judge_empty_response")
        return GoalVerdict(ok=None)

    cleaned = _FENCE_RE.sub("", raw).strip()
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        cleaned = cleaned[brace_start : brace_end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("coder_goal_judge_parse_failed", raw_preview=raw[:160])
        return GoalVerdict(ok=None, raw_response=raw)

    ok = parsed.get("ok")
    if not isinstance(ok, bool):
        return GoalVerdict(ok=None, raw_response=raw)
    return GoalVerdict(
        ok=ok,
        impossible=bool(parsed.get("impossible")),
        reason=str(parsed.get("reason") or "")[:400],
        raw_response=raw,
    )


__all__ = ["GoalVerdict", "MAX_JUDGE_REENTRY", "judge_goal_satisfied"]
