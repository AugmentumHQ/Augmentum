"""Subagent output verification — the judge on the delegation return path.

The dispatcher injects the lead's ``success_criteria`` into a subagent's
first message, but nothing checked that the subagent's *output* actually
satisfied them: a confidently-wrong report was indistinguishable from a
correct one, and the lead trusted it verbatim. This module closes that
gap — the leaf-node twin of ``coder/goal_judge.py`` (which guards the
LEAD's stop, not a subagent's).

Shape mirrors ``goal_judge`` deliberately (proven plumbing): a single
second-LLM round-trip, ``stream=False``, ``temperature=0``, tiny
``max_tokens``, never raises. ``ok=None`` means *no signal* (backend or
parse failure) and callers MUST fail open — a flaky judge must never
trap a subagent in a re-entry loop, AND a silently-failing judge must
never masquerade as a working guarantee (hence ``log.warning``, never
``debug``).

Lives in ``agents/`` rather than ``coder/`` on purpose: ``loop.py`` is
generic across bug_finder + coder consumers, so it must not import a
coder-mode module. The judge here is criteria-shaped, not turn-shaped —
it takes ``(task, success_criteria, output, tool_summary)`` and returns
a per-criterion verdict, with no coder-state dependency.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from augmentum.models.base import InternalChatRequest, Message, response_text
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Re-entries per subagent run. After this many failed verdicts the
# subagent's stop is honored unconditionally with ``verification="failed"``
# so the parent sees the unmet criteria and decides — a subagent is a leaf
# node, cheaper for the lead to re-dispatch than for the subagent to thrash.
# Lower than the lead goal-judge's cap of 2 for that reason.
DEFAULT_VERIFY_REENTRY = 1

_JUDGE_PROMPT = """\
You are a verification judge for a delegated coding subagent. A lead agent \
handed a focused task to a subagent with explicit success criteria; the \
subagent has now reported it is done. Judge ONLY whether each success \
criterion is satisfied by the subagent's actual work — not whether more \
could theoretically be done.

Respond with a JSON object only — no prose, no markdown fences:
{"ok": true, "unmet": [], "reason": "<evidence each criterion is satisfied>"}
{"ok": false, "unmet": ["<verbatim criterion not satisfied>", ...], "reason": "<what is concretely missing>"}

Judge against EVIDENCE in the report and the tool activity, not optimism. \
A criterion phrased as a check ("tests pass", "endpoint returns 200") is \
satisfied only if the activity shows it was actually run and the result was \
good — a bare claim of success with no supporting evidence for a \
load-bearing criterion is UNMET. Put the verbatim text of each unsatisfied \
criterion in "unmet". When the only gap is minor wording rather than real \
substance, prefer ok=true: you exist to catch UNFINISHED delegation, not to \
demand perfection."""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class SubagentVerdict:
    """Judge outcome for one subagent run.

    ``ok=None`` means no signal (backend/parse failure) — callers MUST
    fail open and honor the subagent's stop without re-entry.
    """

    ok: bool | None
    unmet: tuple[str, ...] = ()
    reason: str = ""
    raw_response: str = ""

    @property
    def label(self) -> str:
        """``passed`` | ``failed`` | ``error`` for persistence + UI."""
        if self.ok is True:
            return "passed"
        if self.ok is False:
            return "failed"
        return "error"


async def judge_subagent_result(
    backend,
    *,
    model: str,
    task: str,
    success_criteria: tuple[str, ...] | list[str],
    output: str,
    tool_summary: str = "",
    max_tokens: int = 320,
) -> SubagentVerdict:
    """One judge round-trip over a subagent's reported result.

    Never raises; returns ``ok=None`` on backend or parse failure so the
    caller fails open. Reuses the same ``backend`` + ``model`` the
    subagent ran on — fresh context, no sunk-cost in the subagent's own
    transcript, which is the independence that makes the check worth the
    extra call (same independence MiMo's goal gate relies on).
    """
    criteria = tuple(
        c.strip() for c in (success_criteria or ()) if c and str(c).strip()
    )
    if not criteria:
        # Nothing to verify against — caller shouldn't have invoked us,
        # but be defensive: no criteria means no signal, not a failure.
        return SubagentVerdict(ok=None)

    criteria_block = "\n".join(
        f"{i}. {c[:300]}" for i, c in enumerate(criteria, start=1)
    )
    summary = (
        f"<task>\n{(task or '').strip()[:2000]}\n</task>\n\n"
        f"<success_criteria>\n{criteria_block}\n</success_criteria>\n\n"
        f"<subagent_tool_activity>\n{(tool_summary or '(no tools recorded)')[:1500]}\n"
        f"</subagent_tool_activity>\n\n"
        f"<subagent_final_report>\n{(output or '').strip()[:3500]}\n"
        f"</subagent_final_report>\n\n"
        f"{_JUDGE_PROMPT}"
    )

    req = InternalChatRequest(
        model=model,
        messages=[Message(role="user", content=summary)],
        tools=None,
        tool_choice=None,
        stream=False,
        temperature=0.0,
        max_tokens=max_tokens,
        chat_template_kwargs=None,
    )

    try:
        resp = await backend.chat(req)
    except Exception as exc:  # noqa: BLE001
        # Fail open — loudly. A flaky judge must never trap the subagent
        # in a re-entry loop; a silent fail-open must never look like a
        # passed guarantee.
        log.warning(
            "subagent_verify_backend_error",
            error=str(exc)[:200],
            model=model,
        )
        return SubagentVerdict(ok=None)

    raw = response_text(resp, thinking_fallback=False).strip()
    if not raw:
        log.warning("subagent_verify_empty_response", model=model)
        return SubagentVerdict(ok=None)

    cleaned = _FENCE_RE.sub("", raw).strip()
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        cleaned = cleaned[brace_start : brace_end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("subagent_verify_parse_failed", raw_preview=raw[:160])
        return SubagentVerdict(ok=None, raw_response=raw)

    ok = parsed.get("ok")
    if not isinstance(ok, bool):
        return SubagentVerdict(ok=None, raw_response=raw)

    unmet_raw = parsed.get("unmet")
    unmet: tuple[str, ...] = ()
    if isinstance(unmet_raw, list):
        unmet = tuple(str(u)[:300] for u in unmet_raw if u and str(u).strip())
    return SubagentVerdict(
        ok=ok,
        unmet=unmet,
        reason=str(parsed.get("reason") or "")[:400],
        raw_response=raw,
    )


__all__ = ["DEFAULT_VERIFY_REENTRY", "SubagentVerdict", "judge_subagent_result"]
