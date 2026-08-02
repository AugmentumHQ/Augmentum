"""Investigator — follows threads across the codebase.

When the detector flags an interesting pattern in chunk X, the
investigator's job is to ask "where else in this codebase does the
same pattern appear?" and return a list of new chunks (or refined
finding sites) for the lead to enqueue as DETECT tasks.

Example: detector flags ``except Exception`` in ``auth_routes.py:88``
that leaks error context. Investigator reads that code, identifies
the pattern, then greps the codebase for other handlers with the
same shape — and returns ``[{file, function, line, similar_to}, ...]``
as new DETECT candidates the lead enqueues with elevated priority.

Same shape as the comprehender: one read-only LLM subagent with a
bounded budget, emits a structured JSON the orchestrator parses,
errors degrade gracefully (None on parse failure — lead sees empty
candidate list and moves on).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from augmentum.agents.loop import SubagentResult, SubagentSpec, run_subagent
from augmentum.bug_finder.budget import SubagentBudget
from augmentum.bug_finder.findings import Finding
from augmentum.bug_finder.json_salvage import salvage_json_object
from augmentum.bug_finder.role_models import Role
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Default budget
# ---------------------------------------------------------------------------


# Investigators read across the codebase, so they need more headroom
# than a single detector run. 15 iterations + 60k tokens is enough to
# read the anchor + grep + read ~5 candidates + emit JSON. They shouldn't
# fan out wider than the original detector's budget allocation, though,
# or one finding can blow the run's overall budget.
DEFAULT_INVESTIGATOR_BUDGET = SubagentBudget(
    max_iterations=15,
    max_wallclock_seconds=600,
    max_tokens=60_000,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


INVESTIGATOR_SYSTEM_PROMPT = """\
You are the bug-finder INVESTIGATOR. Given a thread anchor (file +
function + reason), find OTHER sites in the codebase that exhibit the
same pattern. Your output is a list of candidate detector targets;
you don't make claims yourself.

## Workflow

The `anchor` you receive may be EITHER:
  (a) a concrete `file:function` reference, OR
  (b) a free-form pattern description / regex (when no concrete
      anchor is known yet — common when seeded from the user_goal).

Adapt accordingly:

1. **Resolve the anchor**.
   * If it looks like `path/to/file.py:function` (or just a real file
     path), use `file_read` to see it.
   * If it looks like a pattern, regex, or natural-language
     description ("blanket except Exception", "raw f-string in
     SQL"), skip directly to step 2 — the pattern is already given.

2. **Name the pattern precisely**. Examples: "blanket
   `except Exception` that re-raises with `str(exc)`"; "missing
   `escapeHtml` in template literal"; "raw f-string in SQL query".
   When the anchor was a pattern already, just rephrase + commit.

3. **Search aggressively**. Use `grep` / `find_files` against the
   `scope_hint` (or the whole repo if empty). For "except Exception"
   patterns, `grep -rn "except Exception"` is the right starting
   move. For SQL-like patterns, search for `f"SELECT`, `f"INSERT`,
   etc. **You MUST run at least one search before reporting empty
   candidates** — an empty candidate list with no search is a bug,
   not a result.

4. **Confirm by reading**. For each promising hit, read 5-20 lines of
   surrounding context to confirm the pattern actually applies — don't
   include false positives (e.g. an `except Exception` that DOES
   sanitize before re-raising).

   When the anchor is wiring-sensitive, use the deterministic tools to
   rule out the obvious FP shapes BEFORE adding a candidate:

   - "trusts unvalidated `scope['user']` / `request.state.x`" patterns —
     call `middleware_chain` to confirm no gating middleware runs first.
   - "missing auth check" patterns — call `decorators_on` with the
     candidate's file:line; a `@require_auth` decorator is invisible
     from a grep that only matched the function body.
   - "tainted variable reaches sink" patterns — call `trace_origin` on
     the variable at the candidate's use site; a typed parameter
     narrowed by an upstream validator is not the same pattern.
   - "risky sink is reachable" patterns — call `who_calls` /
     `is_reachable_from` to confirm the sink is actually wired in.

   These checks are cheap; surfacing FP candidates costs the lead a
   full detector run downstream.

5. **Emit candidates as JSON**. Quality > quantity: 3 high-confidence
   candidates beat 20 weak ones. The lead enqueues each as a fresh
   DETECT task — wasted enqueues cost detector tokens later. But
   **prefer 3 candidates over 0** — surfacing nothing forces the lead
   into a dead end.

## Output

End every response with a single fenced JSON block:

```json
{
  "pattern": "<one-line description of what you found>",
  "candidates": [
    {
      "file": "<path relative to /workspace>",
      "function": "<function name>",
      "line_start": <int>,
      "line_end": <int>,
      "similar_to": "<anchor>",
      "confidence": "high" | "medium" | "low",
      "rationale": "<one-sentence why this is the same pattern>"
    }
  ]
}
```

Empty candidates list (``"candidates": []``) is a valid output —
"I checked, the pattern doesn't recur." The lead acts on that.

Hard cap: at most 15 candidates per investigation. If you found more,
emit the strongest 15 and let the next investigation surface the
rest if the lead pulls that thread further.
"""


INVESTIGATOR_USER_TEMPLATE = """\
Find other sites in the codebase that exhibit the same pattern as the
anchor below.

**Anchor:** `{anchor}`

**Reason this thread is worth pulling:**
{reason}

{anchor_finding_block}

{scope_hint_block}

Output the candidates JSON as the final element of your response.
"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _last_json_object(output: str) -> dict | None:
    """Salvage the last usable JSON object (truncation-tolerant — audit
    2026-06-17)."""
    return salvage_json_object(output)


@dataclass(frozen=True)
class InvestigatorCandidate:
    """One candidate the investigator wants to flag for follow-up."""

    file: str
    function: str
    line_start: int = 0
    line_end: int = 0
    similar_to: str = ""
    confidence: str = "medium"     # "high" | "medium" | "low"
    rationale: str = ""


@dataclass(frozen=True)
class InvestigatorOutput:
    """Decoded final JSON from the investigator."""

    pattern: str
    candidates: tuple[InvestigatorCandidate, ...]


def parse_investigator_output(output: str) -> InvestigatorOutput | None:
    """Return ``None`` when no fenced JSON parses or schema is wrong.

    An empty candidates list IS a valid output ("I checked, no recur");
    that returns an InvestigatorOutput with ``candidates=()`` and
    ``pattern`` populated. The lead sees the empty list and moves on.
    """
    payload = _last_json_object(output)
    if not payload:
        return None
    pattern = str(payload.get("pattern") or "").strip()
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        return None
    candidates: list[InvestigatorCandidate] = []
    for c in raw_candidates[:15]:
        if not isinstance(c, dict):
            continue
        file = str(c.get("file") or "").strip()
        if not file:
            continue
        function = str(c.get("function") or "<module>").strip() or "<module>"
        confidence = str(c.get("confidence") or "medium").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        try:
            line_start = int(c.get("line_start") or 0)
            line_end = int(c.get("line_end") or 0)
        except (TypeError, ValueError):
            line_start = 0
            line_end = 0
        candidates.append(InvestigatorCandidate(
            file=file, function=function,
            line_start=line_start, line_end=line_end,
            similar_to=str(c.get("similar_to") or "").strip(),
            confidence=confidence,
            rationale=str(c.get("rationale") or "").strip(),
        ))
    return InvestigatorOutput(
        pattern=pattern, candidates=tuple(candidates),
    )


# ---------------------------------------------------------------------------
# Subagent runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvestigatorRunResult:
    """Aggregate outcome of one investigation."""

    output: InvestigatorOutput | None
    subagent_result: SubagentResult
    runtime_seconds: float

    @property
    def succeeded(self) -> bool:
        return self.output is not None


async def run_investigator(
    *,
    model: str,
    backend,
    tools,
    anchor: str,
    scope_hint: str = "",
    anchor_finding: Finding | None = None,
    budget: SubagentBudget = DEFAULT_INVESTIGATOR_BUDGET,
    instance_id: str = "",
    progress_callback=None,
) -> InvestigatorRunResult:
    """Run the investigator subagent once on a single anchor.

    ``anchor`` is free-form text — usually ``"file.py:function_name"``,
    but any string the model can use as a starting point works.
    ``scope_hint`` (optional) narrows the search (e.g. "look only in
    augmentum/auth/").
    ``anchor_finding`` (optional) — when present, the finding's claim
    is rendered into the prompt so the investigator knows what kind
    of pattern to look for.

    Returns ``succeeded=False`` when the model's output isn't parseable
    — the dispatcher treats that as "no candidates" and moves on.
    """
    anchor_block = ""
    reason_text = anchor_finding.claim if anchor_finding else ""
    if anchor_finding is not None:
        anchor_block = (
            "**Anchor finding context** "
            f"(severity={anchor_finding.severity}, "
            f"signature={anchor_finding.claim_signature}):\n"
            f"> {anchor_finding.claim}\n\n"
            "Evidence paths from the original finding:\n"
        ) + "\n".join(f"- `{p}`" for p in anchor_finding.evidence_paths)
    scope_block = ""
    if scope_hint:
        scope_block = f"**Scope hint:** restrict search to: `{scope_hint}`"

    spec = SubagentSpec(
        role=Role.INVESTIGATOR.value,
        model=model,
        system_prompt=INVESTIGATOR_SYSTEM_PROMPT,
        initial_user_message=INVESTIGATOR_USER_TEMPLATE.format(
            anchor=anchor or "(no anchor supplied)",
            reason=reason_text or "(no specific reason supplied)",
            anchor_finding_block=anchor_block or "(no anchor finding)",
            scope_hint_block=scope_block or "(no scope hint — search whole repo)",
        ),
        tools=tools,
        budget=budget,
        instance_id=instance_id or f"investigator_{anchor[:40]}",
        progress_callback=progress_callback,
        temperature=0.0,
    )
    start = time.monotonic()
    result = await run_subagent(spec, backend=backend)
    elapsed = time.monotonic() - start
    parsed = parse_investigator_output(result.output)
    log.info(
        "bug_finder_investigator_done",
        anchor=anchor[:80],
        stop_reason=result.stop_reason,
        iterations=result.iterations,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        candidates=(len(parsed.candidates) if parsed else 0),
        wallclock_seconds=round(elapsed, 1),
    )
    return InvestigatorRunResult(
        output=parsed,
        subagent_result=result,
        runtime_seconds=elapsed,
    )


def is_implemented() -> bool:
    return True
