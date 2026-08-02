"""Pen-tester — actively probes findings to confirm or refute them.

Phase 1c of the dynamic-probe leg. The pen_tester is given ONE
confirmed finding and tries to either reproduce the exploit against
a running instance of the workspace's app, or convince itself the
defense actually holds. Output is a structured verdict the
orchestrator stamps onto the finding.

The role's special abilities (vs. detector / investigator):
* ``boot_under_test`` — start the workspace's app as a subprocess.
* ``http_attack`` — send probe requests at the booted target.
* ``under_test_status`` — verify the app is still alive between probes.

Plus the full read-only + deterministic substrate so it can survey
routes, decorators, middleware, etc., before crafting probes.

The role's discipline:
* **Active evidence > inferred evidence.** A 200 from a payload the
  app should reject is more weight than a code reading suggesting
  the validation is missing.
* **Refutation is a real outcome.** "Defense held against three
  reasonable payloads" downgrades severity; that's progress, not a
  failed run.
* **No code edits.** This role is read + probe only. The fixer is
  a separate concern.
* **Bounded probe count.** Probe-receipt blow-up is the failure mode
  to avoid. Cap probes per finding (default 12); receipts persisted
  to ``.augmentum/bug_finder/probe_receipts.jsonl`` regardless.

The boot lifecycle is managed by the orchestrator (per-run
``_UnderTestRegistry``) — this role can boot but cannot tear down.
That keeps an off-the-rails pen_tester from leaving processes
running between findings.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from augmentum.agents.loop import SubagentResult, SubagentSpec, run_subagent
from augmentum.bug_finder.budget import SubagentBudget
from augmentum.bug_finder.findings import Finding
from augmentum.bug_finder.role_models import Role
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Default budget — pen-tester gets more headroom than a single detector
# call, less than a full lead loop. Each probe is a tool call; the
# verdict-shape requires a few of them, plus reading the source.
# ---------------------------------------------------------------------------


DEFAULT_PEN_TESTER_BUDGET = SubagentBudget(
    max_iterations=20,
    max_wallclock_seconds=900,
    max_tokens=80_000,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


PEN_TESTER_SYSTEM_PROMPT = """\
You are the bug-finder PEN_TESTER. You receive ONE confirmed finding
and decide whether the defense holds against an actual probe.

You operate read-only on source code AND have probing tools:
  - boot_under_test — start the workspace's app as a subprocess.
    REQUIRES an explicit command + port + healthcheck path. The
    workspace baseline (the user message includes it when known)
    tells you what to pass.
  - http_attack — send one HTTP request to the booted target. Use
    this to send payloads, swap auth tokens, observe responses.
  - under_test_status — verify the app is still alive between probes.

Plus the standard deterministic substrate (list_routes,
decorators_on, middleware_chain, etc.) for surveying the attack
surface before crafting probes.

## Discipline

* **Active evidence over inferred evidence.** A 200 from a payload
  the app should reject is stronger evidence than a code reading.
* **Refutation is a valid verdict.** If three reasonable payloads
  fail to exploit, that's a meaningful "refuted" outcome — emit it.
* **Inconclusive is also valid.** Could not boot the app, no
  reachable HTTP surface, target requires data state we can't
  produce — say so. Better than a bluffed "confirmed".
* **Bounded effort.** Aim for 3-8 high-quality probes per finding.
  Spamming 50 mutations is a sign of an investigation that doesn't
  know what it's looking for.
* **No code edits.** Reading source is fine; modifying it is not
  your role. The fixer handles patches.

## Output

End your response with a fenced JSON block:

```json
{
  "finding_id": "<id from the input>",
  "verdict": "confirmed" | "refuted" | "inconclusive",
  "rationale": "<one to three sentences: what you sent, what came back, and what that proves>",
  "evidence": [
    {
      "method": "GET" | "POST" | ...,
      "url": "<URL probed>",
      "status": <response status>,
      "expected_status": <int — what a sound defense should have returned>,
      "note": "<one sentence interpretation>"
    }
  ]
}
```

Empty evidence list is acceptable when verdict is "inconclusive" with
a reason — e.g. couldn't boot, no HTTP surface. Otherwise include the
strongest 1-5 probe rows that support your verdict.
"""


PEN_TESTER_USER_TEMPLATE = """\
Confirm or refute the finding below via dynamic probing.

**Finding** `{finding_id}`
  file:          {file}
  function:      {function}
  severity:      {severity}
  signature:     {claim_signature}

**Claim:** {claim}

**Suggested repro:** {suggested_repro}

**Evidence paths from static detection:**
{evidence_paths}

{boot_hint_block}

Probe the target. Emit the verdict JSON at the end of your response.
"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
_VALID_VERDICTS: frozenset[str] = frozenset({
    "confirmed", "refuted", "inconclusive",
})


def _last_json_object(output: str) -> dict | None:
    if not output:
        return None
    blocks = [m.group(1).strip() for m in _JSON_BLOCK_RE.finditer(output)]
    if not blocks:
        stripped = output.strip()
        if stripped.startswith("{"):
            blocks = [stripped]
    for blk in reversed(blocks):
        try:
            parsed = json.loads(blk)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


@dataclass(frozen=True)
class ProbeEvidence:
    """One probe row attached to a verdict."""

    method: str
    url: str
    status: int
    expected_status: int = 0
    note: str = ""


@dataclass(frozen=True)
class PenTestVerdict:
    """Structured pen-test outcome for one finding."""

    finding_id: str
    verdict: str                # "confirmed" | "refuted" | "inconclusive"
    rationale: str
    evidence: tuple[ProbeEvidence, ...]

    @property
    def is_confirmed(self) -> bool:
        return self.verdict == "confirmed"

    @property
    def is_refuted(self) -> bool:
        return self.verdict == "refuted"


def parse_pen_tester_output(output: str) -> PenTestVerdict | None:
    """Decode the pen-tester's final JSON. Returns ``None`` when no
    fenced JSON parses or the schema doesn't fit.

    The orchestrator treats ``None`` as "inconclusive — couldn't
    interpret" and moves on; the finding's status doesn't change.
    """
    payload = _last_json_object(output)
    if not payload:
        return None
    finding_id = str(payload.get("finding_id") or "").strip()
    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict not in _VALID_VERDICTS:
        return None
    rationale = str(payload.get("rationale") or "").strip()
    raw_evidence = payload.get("evidence")
    if not isinstance(raw_evidence, list):
        raw_evidence = []
    evidence: list[ProbeEvidence] = []
    for row in raw_evidence:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        try:
            status = int(row.get("status") or 0)
            expected = int(row.get("expected_status") or 0)
        except (TypeError, ValueError):
            status = 0
            expected = 0
        evidence.append(ProbeEvidence(
            method=str(row.get("method") or "").upper().strip() or "GET",
            url=url,
            status=status,
            expected_status=expected,
            note=str(row.get("note") or "").strip(),
        ))
    return PenTestVerdict(
        finding_id=finding_id,
        verdict=verdict,
        rationale=rationale,
        evidence=tuple(evidence),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PenTesterRunResult:
    """Aggregate outcome of one pen-test invocation."""

    verdict: PenTestVerdict | None
    subagent_result: SubagentResult
    runtime_seconds: float

    @property
    def succeeded(self) -> bool:
        return self.verdict is not None


def _build_boot_hint_block(
    *,
    boot_command_hint: str = "",
    boot_port_hint: int = 0,
    healthcheck_path_hint: str = "/",
) -> str:
    """Compose the boot-hint block injected into the user message.

    Empty when the orchestrator has nothing to recommend — then the
    pen_tester decides for itself how to invoke the app (or punts to
    "inconclusive — couldn't determine boot command").
    """
    if not boot_command_hint and not boot_port_hint:
        return (
            "**Boot hint:** none — you'll need to inspect the "
            "workspace (look for a Procfile, docker-compose.yml, or "
            "the readme's quickstart) to determine how to boot. If "
            "you can't find a confident boot command, emit verdict "
            "``inconclusive`` with rationale noting the gap."
        )
    parts = ["**Boot hint** (suggested invocation for boot_under_test):"]
    if boot_command_hint:
        parts.append(f"  command: `{boot_command_hint}`")
    if boot_port_hint:
        parts.append(f"  port: {boot_port_hint}")
    parts.append(f"  healthcheck_path: `{healthcheck_path_hint}`")
    return "\n".join(parts)


async def run_pen_tester(
    *,
    model: str,
    backend: Any,
    tools: tuple,
    finding: Finding,
    boot_command_hint: str = "",
    boot_port_hint: int = 0,
    healthcheck_path_hint: str = "/",
    budget: SubagentBudget = DEFAULT_PEN_TESTER_BUDGET,
    instance_id: str = "",
    progress_callback=None,
) -> PenTesterRunResult:
    """Run the pen_tester subagent once against one finding.

    The caller supplies ``tools`` already filtered down to the
    PEN_TESTER allow-list (read-only + deterministic + pen-test
    primitives). The per-run ``_UnderTestRegistry`` MUST be already
    bound into the boot tools — this function doesn't manage
    lifecycle.

    Returns ``succeeded=False`` when output can't be parsed. Caller
    treats that as "inconclusive" and moves on.
    """
    boot_hint_block = _build_boot_hint_block(
        boot_command_hint=boot_command_hint,
        boot_port_hint=boot_port_hint,
        healthcheck_path_hint=healthcheck_path_hint,
    )
    evidence_paths_block = (
        "\n".join(f"  - `{p}`" for p in finding.evidence_paths)
        if finding.evidence_paths
        else "  (none supplied)"
    )
    user_msg = PEN_TESTER_USER_TEMPLATE.format(
        finding_id=finding.id,
        file=finding.file,
        function=finding.function or "<module>",
        severity=finding.severity,
        claim_signature=finding.claim_signature,
        claim=finding.claim,
        suggested_repro=finding.suggested_repro or "(none)",
        evidence_paths=evidence_paths_block,
        boot_hint_block=boot_hint_block,
    )
    spec = SubagentSpec(
        role=Role.PEN_TESTER.value,
        model=model,
        system_prompt=PEN_TESTER_SYSTEM_PROMPT,
        initial_user_message=user_msg,
        tools=tools,
        budget=budget,
        instance_id=instance_id or f"pen_tester_{finding.id[:24]}",
        progress_callback=progress_callback,
        temperature=0.0,
    )
    start = time.monotonic()
    result = await run_subagent(spec, backend=backend)
    elapsed = time.monotonic() - start
    parsed = parse_pen_tester_output(result.output)
    log.info(
        "bug_finder_pen_tester_done",
        finding_id=finding.id,
        verdict=(parsed.verdict if parsed else "unparsed"),
        evidence_count=(len(parsed.evidence) if parsed else 0),
        iterations=result.iterations,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        wallclock_seconds=round(elapsed, 1),
    )
    return PenTesterRunResult(
        verdict=parsed,
        subagent_result=result,
        runtime_seconds=elapsed,
    )


def is_implemented() -> bool:
    return True


# ---------------------------------------------------------------------------
# Note rendering for the Finding's audit trail
# ---------------------------------------------------------------------------


def verdict_to_note(verdict: PenTestVerdict) -> str:
    """Render a one-line note suitable for ``Finding.notes`` so a
    future reader sees the dynamic outcome alongside the static
    finding context."""
    suffix = ""
    if verdict.rationale:
        rationale = verdict.rationale.strip().splitlines()[0][:160]
        suffix = f" — {rationale}"
    evidence_n = len(verdict.evidence)
    return (
        f"pen_test: {verdict.verdict}"
        f" ({evidence_n} probe{'s' if evidence_n != 1 else ''})"
        f"{suffix}"
    )
