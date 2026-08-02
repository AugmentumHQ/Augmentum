"""Codebase comprehension subagent.

The comprehender is the system's first read of a workspace. Its job is
structural — *understand the code* so every later stage (planner →
detector → verifier → fixer) operates on a shared map instead of
re-deriving the same survey from scratch each run.

What it outputs:

* **subsystems** — major functional groupings with their paths +
  one-sentence purpose.
* **pillars** — load-bearing invariants the codebase relies on
  ("every user-scoped table accepts user_id", "all template literals
  escape user content", "auth middleware runs before route handlers").
* **risk_surfaces** — untrusted-input boundaries worth special
  attention (HTTP routes, file uploads, deserialization sinks).
* **entry_points** — every callable entrance into the system.
* **brief** — a markdown synthesis the planner consumes verbatim.

Single-pass design for v1: one subagent with a generous budget
explores via read-only tools and emits a structured JSON. Per-
subsystem fan-out (cheaper + parallel) is the obvious Phase 2
optimization — defer until we see how single-pass behaves on real
workspaces.

The output is persisted to ``bug_finder_codebase_knowledge`` via the
``KnowledgeStore`` so subsequent runs reuse the map without paying
the comprehension cost again. Re-comprehension fires when the map
is missing (first run) or the user explicitly forgets it; commit-
sha-based staleness detection is a Phase 2 nicety.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from augmentum.bug_finder.budget import SubagentBudget
from augmentum.bug_finder.knowledge_store import (
    EntryPoint,
    Pillar,
    RiskSurface,
    Subsystem,
)
from augmentum.bug_finder.role_models import Role
from augmentum.bug_finder.subagent import (
    SubagentResult,
    SubagentSpec,
    run_subagent,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


COMPREHENDER_SYSTEM_PROMPT = """\
You are the bug-finder COMPREHENDER. A deterministic skeleton has \
already mapped the codebase's structure (languages, files, routes, \
subsystems, settings). You read that skeleton + sample 5-10 specific \
files to identify the parts only judgment can name: **pillars** \
(load-bearing architectural invariants) and **risk surfaces** \
(untrusted-input boundaries).

This is a SYNTHESIS job, not a discovery job. Most of the structural \
data is in the skeleton; you don't need to re-discover it. Trust it.

You are NOT looking for bugs. You are documenting:

  - **Subsystems** — major functional groupings (e.g. "auth", \
"narrative", "documents", "coder"). One subsystem == one cohesive \
set of files with a shared responsibility.
  - **Pillars** — the load-bearing invariants the codebase relies on. \
A pillar is a property the rest of the system assumes is true. \
Examples: "every user-scoped table accepts a user_id parameter on \
every CRUD function"; "template literals always pass user content \
through escapeHtml()"; "auth middleware fires before any user-scoped \
handler". Pillars give the detector the rubric for "is this a bug or \
intentional".
  - **Risk surfaces** — places untrusted input enters the system. HTTP \
routes, WebSockets, file uploads, deserialization sinks, RPC handlers, \
the receiving side of any external API client. For each, note the \
trust boundary (who can reach it) and which downstream sinks it can \
ultimately influence.
  - **Entry points** — a catalog of every callable entrance: HTTP \
routes, background jobs, CLI scripts, MCP tools, scheduled tasks.

## Workflow — TIGHT BUDGET, COMMIT-FIRST

Budget: 10 iterations, 80k tokens. This is plenty because the \
skeleton already did the structural work. The single biggest failure \
mode for this role is "kept exploring, never committed." Don't.

  1. **Read the skeleton first** (in your user message). Note the \
candidate-pillar files it suggests.
  2. **Iteration 1-2**: emit a DRAFT JSON immediately. Use the \
skeleton's subsystems verbatim. Leave pillars + risk_surfaces empty \
for now. This is your safety net.
  3. **Iterations 3-8**: read the candidate-pillar files. For each \
file, ask: what invariant does the surrounding code assume is true? \
What untrusted input enters here? Patch the draft JSON with \
discoveries.
  4. **Iteration 9-10**: emit the FINAL JSON. Re-include the \
skeleton's subsystems + routes; add your pillars + risk_surfaces.

Always emit a complete fenced JSON block at each iteration. The last \
one wins — early drafts are your insurance against budget cut-off.

## What to capture

  - **Subsystems** — major functional groupings (e.g. "auth", \
"narrative", "documents", "coder"). Aim for 8-30 subsystems on a \
1000-file repo; granularity should match the repository's actual \
modularity, not artificially split.

  - **Pillars** — load-bearing invariants the codebase relies on. \
A pillar is a property the rest of the system assumes is true. \
Examples: "every user-scoped table accepts a user_id parameter on \
every CRUD function"; "template literals always pass user content \
through escapeHtml()"; "auth middleware fires before any user-scoped \
handler". Pillars give the detector its rubric for "is this a bug or \
intentional".

  - **Risk surfaces** — places untrusted input enters. HTTP routes, \
WebSockets, file uploads, deserialization sinks. Note trust boundary \
+ which downstream sinks they can reach.

  - **Entry points** — catalog of every callable entrance: HTTP \
routes, background jobs, CLI scripts, MCP tools.

## Output

A SINGLE fenced JSON block at the end of your final response. The \
structure MUST be:

```json
{
  "brief": "<markdown synthesis — 500-2000 words. Lead with a one-paragraph repository summary. Then sections: ## Subsystems, ## Pillars, ## Risk surfaces, ## How findings should be framed (per the pillars).>",
  "subsystems": [
    {
      "name": "<short identifier — auth, narrative, coder, etc.>",
      "purpose": "<one sentence>",
      "paths": ["<dir or file glob>", ...],
      "size_files": <int — approximate file count>,
      "pillars": ["<pillar name>", ...]
    }
  ],
  "pillars": [
    {
      "name": "<short identifier — user_id_scoping, escape_html_template_literals, etc.>",
      "statement": "<the invariant as a complete sentence>",
      "evidence": ["<file>:<line>", ...]
    }
  ],
  "risk_surfaces": [
    {
      "name": "<http_routes, websocket_handlers, upload_endpoints, deserialize_sinks, ...>",
      "entry_points": ["<file>:<function>", ...],
      "trust_boundary": "<user-supplied | third-party-api | filesystem | rpc-peer | ...>",
      "downstream_sinks": ["<what can this reach?>", ...]
    }
  ],
  "entry_points": [
    {
      "kind": "<http | websocket | job | cli | mcp | ...>",
      "path": "<METHOD /api/path  OR  job_name  OR  cli command>",
      "handler": "<file>:<function>"
    }
  ]
}
```

The `brief` is the most important field — downstream subagents read \
it as system-prompt context. Make it concise, opinionated, and \
specific. Vague summaries waste tokens for every subsequent run.

Hard cap: at most 200 entries across all four arrays. Quality > \
quantity. A 30-subsystem map of a large repo is more useful than a \
300-subsystem map that splits hairs.

## CRITICAL: commit early, refine later

You will be cut off at the budget cap. A complete JSON at iteration \
20 with shallow content is **infinitely more valuable** than a \
deeply-explored survey that never gets emitted. The pipeline persists \
your output; refinement happens on the NEXT run (re-comprehension is \
cheap once a draft exists).

Always emit a fenced JSON block at the end of your response, even \
during your refinement iterations. The last fenced block wins; \
intermediate emissions are safe.
"""


COMPREHENDER_USER_TEMPLATE = """\
Synthesize a knowledge map of the workspace.

{skeleton_block}

{user_goal_block}

{threat_model_block}

Your task: read the 5-10 candidate-pillar files listed in the \
skeleton, identify the pillars (architectural invariants) and risk \
surfaces (untrusted-input boundaries), and emit the final JSON. \
Subsystems, routes, and entry_points should come from the skeleton \
verbatim — your value is the pillars + risk_surfaces + the brief.

Commit a draft JSON in iteration 1-2 and refine. Last fenced JSON \
block wins.
"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _scan_object(text: str, start: int) -> tuple[str, int, bool]:
    """Scan a ``{...}`` object starting at ``text[start] == '{'``,
    string/escape aware. Returns ``(substring, end_index, complete)``.

    ``complete`` is False when the object is truncated (the input was cut
    off mid-object — the dominant comprehender failure under a token
    budget); the substring then runs to end-of-text for repair.
    """
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : j + 1], j + 1, True
    return text[start:], len(text), False


def _json_candidates(text: str) -> list[str]:
    """All plausible JSON-object candidates in ``text``, in document order.

    Covers three cases the old fence-only parser missed:
      * fenced blocks (``` ```json ... ``` ```) — the happy path;
      * an OPENING fence with no closing fence — i.e. a budget-truncated
        final block (no closing ``` exists, so the fence regex skips it);
      * JSON embedded in prose without a fence.

    The brace scan returns truncated objects too (``complete=False``) so
    the caller can attempt a repair.
    """
    cands: list[str] = []
    for m in _JSON_BLOCK_RE.finditer(text):
        cands.append(m.group(1).strip())
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            obj, end, complete = _scan_object(text, i)
            cands.append(obj.strip())
            i = end if complete else n
        else:
            i += 1
    return cands


def _open_structure(s: str) -> bool:
    """True when ``s`` ends with an unclosed object/array or string — i.e.
    it was truncated mid-structure."""
    stack = 0
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack += 1
        elif ch in "}]":
            stack = max(0, stack - 1)
    return stack > 0 or in_str


def _close_stack(s: str) -> str:
    """Close any open string + brackets at the end of ``s`` (string-aware),
    trimming a dangling trailing comma/colon first."""
    closers: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            closers.append("}")
        elif ch == "[":
            closers.append("]")
        elif ch in "}]" and closers:
            closers.pop()
    out = s
    if in_str:
        out += '"'
    out = out.rstrip()
    while out and out[-1] in ",:":
        out = out[:-1].rstrip()
    return out + "".join(reversed(closers))


def _last_struct_delim(s: str) -> tuple[int, str] | None:
    """Index + char of the last ``,`` / ``{`` / ``[`` not inside a string."""
    in_str = False
    esc = False
    last: tuple[int, str] | None = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in ",{[":
            last = (i, ch)
    return last


def _repair_truncated(s: str) -> str | None:
    """Best-effort recovery of a truncated JSON object.

    Iteratively closes the open structure and tries to parse; on failure
    it chops the last incomplete element (back to the previous ``,`` /
    container opener) and retries. This recovers a budget-cut final block
    down to whatever complete elements landed before the cut — even when
    the cut left a dangling key (``{"name": ``). Returns ``None`` when the
    input wasn't actually truncated (so a real syntax error elsewhere
    isn't masked) or nothing salvageable remains.

    The result is always re-validated by ``json.loads`` upstream, so a bad
    repair can only fail closed — never inject wrong data. Deliberately
    does NOT strip ``//`` comments — the ``brief`` field is markdown that
    routinely contains ``http://`` URLs.
    """
    if not _open_structure(s):
        return None  # not truncated — don't mask a real error
    candidate = s
    for _ in range(64):  # bounded; each pass drops one trailing element
        closed = _close_stack(candidate)
        try:
            json.loads(closed)
            return closed
        except json.JSONDecodeError:
            pass
        delim = _last_struct_delim(candidate)
        if delim is None:
            return None
        idx, ch = delim
        # ',' → drop it + the partial element after; '{'/'[' → keep the
        # opener (collapses to an empty container), drop the partial body.
        candidate = candidate[:idx] if ch == "," else candidate[: idx + 1]
        if not candidate.strip():
            return None
    return None


def _loads_lenient(blk: str) -> dict | None:
    """``json.loads`` with a single truncation-repair fallback."""
    blk = (blk or "").strip()
    if not blk:
        return None
    try:
        parsed = json.loads(blk)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    repaired = _repair_truncated(blk)
    if repaired is None:
        return None
    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _last_json_payload(output: str) -> dict | None:
    """Pull the last usable JSON object.

    Tries every candidate in reverse document order so the final (most
    refined) emit wins, with the truncation-repair fallback recovering a
    budget-cut final block instead of losing the whole map (audit
    2026-06-17 — "comprehender couldn't produce parseable JSON" → zero
    findings was this path failing).
    """
    if not output:
        return None
    for blk in reversed(_json_candidates(output)):
        parsed = _loads_lenient(blk)
        if parsed is not None:
            return parsed
    return None


@dataclass(frozen=True)
class ComprehenderOutput:
    """Decoded final JSON from the comprehender, ready for KnowledgeStore."""

    brief: str
    subsystems: tuple[Subsystem, ...]
    pillars: tuple[Pillar, ...]
    risk_surfaces: tuple[RiskSurface, ...]
    entry_points: tuple[EntryPoint, ...]


def parse_comprehender_output(output: str) -> ComprehenderOutput | None:
    """Return ``None`` when no fenced JSON parses or the schema is wrong.

    Caller treats ``None`` as "comprehender produced no usable map" and
    falls back to running without a knowledge brief — the planner is
    still functional, just less efficient on first contact.
    """
    payload = _last_json_payload(output)
    if not payload:
        return None
    brief = str(payload.get("brief") or "").strip()
    subsystems = tuple(
        Subsystem(
            name=str(s.get("name") or "").strip(),
            purpose=str(s.get("purpose") or "").strip(),
            paths=tuple(str(p) for p in (s.get("paths") or [])),
            size_files=int(s.get("size_files") or 0),
            pillars=tuple(str(p) for p in (s.get("pillars") or [])),
        )
        for s in (payload.get("subsystems") or [])
        if isinstance(s, dict) and str(s.get("name") or "").strip()
    )
    pillars = tuple(
        Pillar(
            name=str(p.get("name") or "").strip(),
            statement=str(p.get("statement") or "").strip(),
            evidence=tuple(str(e) for e in (p.get("evidence") or [])),
        )
        for p in (payload.get("pillars") or [])
        if isinstance(p, dict) and str(p.get("name") or "").strip()
    )
    risk_surfaces = tuple(
        RiskSurface(
            name=str(r.get("name") or "").strip(),
            entry_points=tuple(str(e) for e in (r.get("entry_points") or [])),
            trust_boundary=str(r.get("trust_boundary") or "").strip(),
            downstream_sinks=tuple(
                str(s) for s in (r.get("downstream_sinks") or [])
            ),
        )
        for r in (payload.get("risk_surfaces") or [])
        if isinstance(r, dict) and str(r.get("name") or "").strip()
    )
    entry_points = tuple(
        EntryPoint(
            kind=str(e.get("kind") or "").strip(),
            path=str(e.get("path") or "").strip(),
            handler=str(e.get("handler") or "").strip(),
        )
        for e in (payload.get("entry_points") or [])
        if isinstance(e, dict) and str(e.get("path") or "").strip()
    )
    # Need at least *something* to call this a usable map.
    if not (brief or subsystems or pillars or risk_surfaces):
        return None
    return ComprehenderOutput(
        brief=brief,
        subsystems=subsystems,
        pillars=pillars,
        risk_surfaces=risk_surfaces,
        entry_points=entry_points,
    )


# ---------------------------------------------------------------------------
# Subagent runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComprehenderRunResult:
    """Aggregate of one comprehension pass — output + cost ledger metadata."""

    output: ComprehenderOutput | None
    subagent_result: SubagentResult
    runtime_seconds: float

    @property
    def succeeded(self) -> bool:
        return self.output is not None


# Default budget — TIGHT because the skeleton has already done the
# expensive structural work. The comprehender now only synthesizes
# pillars + risk_surfaces + brief from ~10 file reads, so 80k tokens
# / 12 iterations / 10 minutes is comfortable headroom. Previous
# 500k-token budget was the explore-everything anti-pattern.
DEFAULT_COMPREHENDER_BUDGET = SubagentBudget(
    max_iterations=12,
    max_wallclock_seconds=600,
    max_tokens=80_000,
)


async def run_comprehender(
    *,
    model: str,
    backend,
    tools,
    skeleton_block: str,
    user_goal_block: str = "",
    threat_model_block: str = "",
    budget: SubagentBudget = DEFAULT_COMPREHENDER_BUDGET,
    instance_id: str = "comprehender",
    progress_callback=None,
) -> ComprehenderRunResult:
    """Run the comprehender subagent once.

    Caller responsibilities:
      * Filter `tools` to the comprehender allow-list (read-only) via
        ``filter_tools(tools, COMPREHENDER_TOOL_NAMES)``.
      * Resolve `backend` + `model` from ``role_models.comprehender``
        (or fall back to planner via ``role_models.for_role``).
      * Persist the parsed output via ``KnowledgeStore.upsert`` on
        success.

    Returns a result with ``succeeded=False`` when the model's final
    output didn't yield a parseable JSON map — the orchestrator
    proceeds without a knowledge brief in that case.
    """
    spec = SubagentSpec(
        role=Role.COMPREHENDER.value,
        model=model,
        system_prompt=COMPREHENDER_SYSTEM_PROMPT,
        initial_user_message=COMPREHENDER_USER_TEMPLATE.format(
            skeleton_block=skeleton_block or "(skeleton unavailable)",
            user_goal_block=user_goal_block or "(no specific user goal)",
            threat_model_block=threat_model_block or "(no explicit threat model)",
        ),
        tools=tools,
        budget=budget,
        instance_id=instance_id,
        progress_callback=progress_callback,
        temperature=0.0,
    )
    start = time.monotonic()
    result = await run_subagent(spec, backend=backend)
    elapsed = time.monotonic() - start
    parsed = parse_comprehender_output(result.output)
    if parsed is None:
        log.warning(
            "bug_finder_comprehender_unparseable",
            stop_reason=result.stop_reason,
            iterations=result.iterations,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
        )
    else:
        log.info(
            "bug_finder_comprehender_done",
            stop_reason=result.stop_reason,
            iterations=result.iterations,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            wallclock_seconds=round(elapsed, 1),
            subsystems=len(parsed.subsystems),
            pillars=len(parsed.pillars),
            risk_surfaces=len(parsed.risk_surfaces),
            entry_points=len(parsed.entry_points),
        )
    return ComprehenderRunResult(
        output=parsed,
        subagent_result=result,
        runtime_seconds=elapsed,
    )


def to_knowledge_kwargs(
    output: ComprehenderOutput,
    *,
    workspace_id: str,
    user_id: str = "",
    commit_sha: str = "",
    subagent: SubagentResult,
    wallclock_seconds: float,
) -> dict:
    """Build the kwargs dict for ``KnowledgeStore.upsert(**kwargs)``."""
    return {
        "user_id": user_id,
        "workspace_id": workspace_id,
        "brief": output.brief,
        "subsystems": output.subsystems,
        "pillars": output.pillars,
        "risk_surfaces": output.risk_surfaces,
        "entry_points": output.entry_points,
        "commit_sha": commit_sha,
        "tokens_in": subagent.tokens_in,
        "tokens_out": subagent.tokens_out,
        "wallclock_seconds": wallclock_seconds,
    }
