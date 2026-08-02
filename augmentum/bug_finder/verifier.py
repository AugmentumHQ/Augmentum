"""Verification helpers — repro construction and fix verification.

This module is **factories + parsers**, not orchestration. It builds
``SubagentSpec`` objects for the verifier role and parses the resulting
``SubagentResult`` back into structured ``Finding`` updates. The actual
``run_subagent`` calls happen in the orchestrator, which owns container
lifecycle and concurrency.

Two stages live here:

* **Stage 5 — verify-is-real (``make_repro_spec``)**: given a speculative
  finding, the verifier subagent constructs a minimal repro inside the
  container, runs it, and confirms the bug actually triggers. The output
  promotes the finding to ``CONFIRMED`` (with a stored repro command) or
  ``UNCONFIRMABLE``.

* **Stage 6b — fix-verify (``make_fix_verify_spec``)**: after the fixer
  has applied a candidate patch in an isolated container fork, the
  verifier re-runs the previously-confirmed repro (which must now pass)
  AND the project's own tests (which must not regress). Returns a
  ``FixVerifyOutcome``.

Both prompts deliberately constrain the verifier to emit a single JSON
block as its final message. Parsing is permissive — last fenced JSON
block wins, falls back to trailing-JSON-only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from augmentum.bug_finder.budget import SubagentBudget
from augmentum.bug_finder.findings import Finding, FindingStatus
from augmentum.bug_finder.guards import verifier_guard
from augmentum.bug_finder.subagent import SubagentResult, SubagentSpec
from augmentum.tools.base import Tool

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REPRO_SYSTEM_PROMPT = """\
You are the bug-finder VERIFIER. Treat the claim below as a HYPOTHESIS \
TO DISPROVE. Your default assumption is that it is a FALSE POSITIVE; \
only an executable proof-of-concept that demonstrably triggers the bug \
flips it to confirmed.

This adversarial framing is deliberate. Anthropic's own bug-finder \
work measured ~50% false-positive reduction from disproof-prompted \
verifiers vs. confirmation-prompted ones, and "near zero" FPs when \
the verifier also had to build a working PoC. We do BOTH.

You operate inside a Docker workspace container at /workspace with \
read access to the codebase. You may add NEW files under \
/workspace/.augmentum/repros/. You may NOT edit existing source or \
test files. You may NOT consult the detector's reasoning, the \
detector's suggested repro, or any classification hint — work only \
from the file/function/line evidence in the brief.

Workflow:
  1. Read the file and function. Look for reasons the claim is WRONG \
(missing precondition, caller already validates, the named "trigger" \
can't actually reach this code path, error handling already covers it).
  2. If you cannot disprove the claim by reading alone, attempt to \
construct an executable proof-of-concept under \
/workspace/.augmentum/repros/. Prefer the project's own test runner \
(pytest, go test, npm test, etc.); shell script is the fallback.
  3. Run the PoC and observe its actual behavior.
  4. Decide: does the PoC demonstrate the bug triggers, or did your \
disproof attempt succeed (showing the claim cannot reach the failure)?

Confirmed REQUIRES: a runnable artifact whose execution observably \
triggers the claimed consequence. Reasoning about whether the bug \
"could" trigger is NOT confirmation — Anthropic's data shows reasoning- \
only verification produces an unacceptable FP rate.

End your response with a fenced JSON block:

```json
{
  "result": "confirmed" | "unconfirmable",
  "repro_path": "/workspace/.augmentum/repros/<your_file>",
  "repro_command": "<exact command to re-run the repro>",
  "evidence": "<short summary: WHY confirmed (what triggered) or WHY \
unconfirmable (what disproof attempt succeeded, or why no PoC could be \
constructed)>"
}
```

If you cannot construct a triggering PoC after honest effort, report \
``"unconfirmable"`` — do NOT fabricate evidence. A false positive \
caught here is the feature working; a false negative is acceptable. \
Lying to look productive destroys the pipeline.
"""


_FIX_VERIFY_SYSTEM_PROMPT = """\
You are the bug-finder FIX-VERIFIER. A fixer subagent has produced a \
candidate patch for a previously-confirmed finding. Your job is to decide \
whether the patch is acceptable.

You operate inside a fresh container fork at /workspace with the patch \
already applied. The fixer is NOT present. You may not modify the patch, \
the source files it touched, or any test files.

You must answer two questions:

  1. **Does the repro now pass?** Re-run the previously-stored repro \
command. The bug should no longer trigger.
  2. **Are there regressions?** Run the project's own test suite \
(pytest / go test / npm test — detect from the project layout). Compare \
to the baseline pass/fail set captured before the run started.

End your response with a fenced JSON block:

```json
{
  "repro_passes_now": true | false,
  "regressions_detected": true | false,
  "repro_evidence": "<output of the re-run repro>",
  "test_evidence": "<concise summary: which tests, baseline pass count, \
current pass count, any newly-failing test names>",
  "recommendation": "accept" | "reject",
  "rejection_reason": "<empty string when recommending accept>"
}
```

Recommend ``"accept"`` only when BOTH the repro passes AND no \
regressions were detected. If you cannot determine either answer with \
confidence (couldn't find the test suite, the repro file is missing, \
etc.), recommend ``"reject"`` with a precise ``rejection_reason``. \
Better to fail closed than to ship a wrong fix.
"""


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReproOutcome:
    """Structured result of stage 5 (verify-is-real)."""

    confirmed: bool
    repro_path: str
    repro_command: str
    evidence: str
    errored: bool = False
    """True when the verifier subagent failed with a backend ERROR
    (``stop_reason="error"``) rather than completing a real judgment.
    An errored verifier is INFRASTRUCTURE failure, NOT a refutation —
    conflating the two silently buries real findings as "unconfirmable"
    (the field failure: detectors collapse, then verifiers collapse, and
    the run reports 0 confirmed as if the code were clean). Callers use
    this to mark such findings distinctly and to compute verify health."""


@dataclass
class FixVerifyOutcome:
    """Structured result of stage 6b (fix-verify)."""

    accept: bool
    repro_passes_now: bool
    regressions_detected: bool
    repro_evidence: str
    test_evidence: str
    rejection_reason: str


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------


def _format_finding_brief(finding: Finding, *, include_detector_hints: bool = False) -> str:
    """Concise brief for the verifier prompt — file/function/claim/evidence.

    By default (``include_detector_hints=False``) the brief excludes the
    detector's ``suggested_repro`` and ``claim_signature``. This is the
    isolation discipline Anthropic's design calls out: the verifier
    should work from the file evidence only, not from the detector's
    chain-of-thought / classification. Sharing those biases the verifier
    toward confirming a specific class instead of disproving the claim.

    The fix-verify stage (``make_fix_verify_spec``) does pass the hints
    through because by then the bug is already CONFIRMED — the
    fix-verifier needs the same context the fixer had to assess whether
    the patch actually addresses the root cause.
    """
    evidence_lines = "\n".join(f"  - {p}" for p in finding.evidence_paths) or "  (none)"
    brief = (
        f"FINDING ID: {finding.id}\n"
        f"FILE: {finding.file}\n"
        f"FUNCTION: {finding.function}\n"
        f"CLAIM:\n  {finding.claim}\n"
        f"EVIDENCE PATHS:\n{evidence_lines}\n"
    )
    if include_detector_hints:
        suggested = finding.suggested_repro or "(none supplied by detector)"
        brief += (
            f"SEVERITY: {finding.severity}\n"
            f"CLAIM SIGNATURE: {finding.claim_signature}\n"
            f"DETECTOR'S SUGGESTED REPRO:\n  {suggested}\n"
        )
    return brief


def make_repro_spec(
    finding: Finding,
    *,
    model: str,
    tools: tuple[Tool, ...],
    budget: SubagentBudget,
    system_prompt_prefix: str = "",
) -> SubagentSpec:
    """Build a SubagentSpec for stage 5 (verify-is-real) on this finding.

    ``system_prompt_prefix`` is prepended verbatim to the verifier's
    system prompt — used by the orchestrator to surface the threat
    model so the verifier reasons from the same authoritative trust-
    boundary definition the detector did.
    """
    user_msg = (
        "Treat the finding below as a hypothesis to disprove. Try to "
        "show it cannot trigger; if you fail, construct an executable "
        "PoC that demonstrates it does trigger.\n\n"
        + _format_finding_brief(finding, include_detector_hints=False)
    )
    sys_prompt = _REPRO_SYSTEM_PROMPT
    if system_prompt_prefix.strip():
        sys_prompt = system_prompt_prefix.rstrip() + "\n\n---\n\n" + sys_prompt
    return SubagentSpec(
        role="verifier",
        model=model,
        system_prompt=sys_prompt,
        initial_user_message=user_msg,
        tools=tools,
        budget=budget,
        tool_guard=verifier_guard,
        instance_id=f"verify_real_{finding.id}",
        temperature=0.0,
    )


def make_fix_verify_spec(
    finding: Finding,
    *,
    model: str,
    tools: tuple[Tool, ...],
    budget: SubagentBudget,
    system_prompt_prefix: str = "",
) -> SubagentSpec:
    """Build a SubagentSpec for stage 6b (fix-verify).

    Assumes the caller has already applied the fixer's candidate patch in
    the workspace container and that the previously-stored repro at
    ``finding.repro_path`` is intact.
    """
    user_msg = (
        "A candidate fix has been applied to the workspace. Verify it.\n\n"
        + _format_finding_brief(finding, include_detector_hints=True)
        + (
            f"\nSTORED REPRO COMMAND:\n  {finding.repro_command or '(missing)'}\n"
            f"STORED REPRO PATH:\n  {finding.repro_path or '(missing)'}\n"
        )
    )
    sys_prompt = _FIX_VERIFY_SYSTEM_PROMPT
    if system_prompt_prefix.strip():
        sys_prompt = system_prompt_prefix.rstrip() + "\n\n---\n\n" + sys_prompt
    return SubagentSpec(
        role="verifier",
        model=model,
        system_prompt=sys_prompt,
        initial_user_message=user_msg,
        tools=tools,
        budget=budget,
        tool_guard=verifier_guard,
        instance_id=f"verify_fix_{finding.id}",
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _last_json_payload(output: str) -> dict[str, Any] | None:
    """Last fenced JSON block in *output*, or the trailing-JSON object."""
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


def parse_repro_result(result: SubagentResult) -> ReproOutcome:
    """Decode stage-5 verifier output into a ``ReproOutcome``.

    A subagent that hit ``stop_reason != "complete"`` is conservatively
    treated as unconfirmable — we don't promote findings on the basis of
    a stuck/budget/error run.
    """
    if result.stop_reason != "complete":
        return ReproOutcome(
            confirmed=False,
            repro_path="",
            repro_command="",
            evidence=(
                f"verifier did not complete cleanly ({result.stop_reason}): "
                f"{result.stop_detail or ''}"
            ),
            errored=result.stop_reason == "error",
        )
    payload = _last_json_payload(result.output)
    if not payload:
        return ReproOutcome(
            confirmed=False,
            repro_path="",
            repro_command="",
            evidence="verifier produced no parseable JSON result",
        )
    verdict = str(payload.get("result") or "").strip().lower()
    confirmed = verdict == "confirmed"
    return ReproOutcome(
        confirmed=confirmed,
        repro_path=str(payload.get("repro_path") or "").strip(),
        repro_command=str(payload.get("repro_command") or "").strip(),
        evidence=str(payload.get("evidence") or "").strip(),
    )


def parse_fix_verify_result(result: SubagentResult) -> FixVerifyOutcome:
    """Decode stage-6b fix-verifier output into a ``FixVerifyOutcome``.

    Conservative on the same axes as ``parse_repro_result``: a verifier
    that didn't complete is a rejection, not an acceptance.
    """
    if result.stop_reason != "complete":
        return FixVerifyOutcome(
            accept=False,
            repro_passes_now=False,
            regressions_detected=False,
            repro_evidence="",
            test_evidence="",
            rejection_reason=(
                f"fix-verifier did not complete cleanly ({result.stop_reason}): "
                f"{result.stop_detail or ''}"
            ),
        )
    payload = _last_json_payload(result.output)
    if not payload:
        return FixVerifyOutcome(
            accept=False,
            repro_passes_now=False,
            regressions_detected=False,
            repro_evidence="",
            test_evidence="",
            rejection_reason="fix-verifier produced no parseable JSON result",
        )
    recommendation = str(payload.get("recommendation") or "").strip().lower()
    repro_passes = bool(payload.get("repro_passes_now"))
    regressions = bool(payload.get("regressions_detected"))
    # Belt-and-suspenders: even if the subagent recommends "accept",
    # don't honor that unless the two boolean conditions are also true.
    accept = recommendation == "accept" and repro_passes and not regressions
    rejection = "" if accept else str(payload.get("rejection_reason") or "").strip()
    if not accept and not rejection:
        if not repro_passes:
            rejection = "repro still triggers after patch"
        elif regressions:
            rejection = "patch caused regressions in the existing test suite"
        else:
            rejection = "recommendation was not 'accept'"
    return FixVerifyOutcome(
        accept=accept,
        repro_passes_now=repro_passes,
        regressions_detected=regressions,
        repro_evidence=str(payload.get("repro_evidence") or "").strip(),
        test_evidence=str(payload.get("test_evidence") or "").strip(),
        rejection_reason=rejection,
    )


# ---------------------------------------------------------------------------
# Finding mutation helpers
# ---------------------------------------------------------------------------


def apply_repro_outcome(finding: Finding, outcome: ReproOutcome) -> Finding:
    """Return a copy of ``finding`` with stage-5 verifier output folded in."""
    if outcome.confirmed:
        note = f"verify-is-real: confirmed; repro at {outcome.repro_path}"
    elif outcome.errored:
        # Distinguish infra failure from a genuine "couldn't reproduce".
        # The finding is NOT refuted — the verifier never rendered a
        # verdict — so the note says so plainly and the run-trust summary
        # can exclude it from the confirmed/refuted accounting.
        note = f"verify-is-real: NOT JUDGED (verifier errored) — {outcome.evidence[:200]}"
    else:
        note = f"verify-is-real: unconfirmable — {outcome.evidence[:200]}"
    return Finding(
        id=finding.id,
        file=finding.file,
        function=finding.function,
        claim=finding.claim,
        claim_signature=finding.claim_signature,
        severity=finding.severity,
        evidence_paths=finding.evidence_paths,
        suggested_repro=finding.suggested_repro,
        status=(
            FindingStatus.CONFIRMED.value
            if outcome.confirmed
            else FindingStatus.UNCONFIRMABLE.value
        ),
        runs_to_confirm=finding.runs_to_confirm,
        total_runs=finding.total_runs,
        repro_path=outcome.repro_path if outcome.confirmed else "",
        repro_command=outcome.repro_command if outcome.confirmed else "",
        repro_output=outcome.evidence,
        invariant=finding.invariant,
        patch=finding.patch,
        fix_attempts=finding.fix_attempts,
        notes=[*finding.notes, note],
    )


def apply_fix_verify_outcome(
    finding: Finding,
    outcome: FixVerifyOutcome,
    *,
    patch_text: str,
    fix_attempts: int,
    invariant: str = "",
) -> Finding:
    """Return a copy of ``finding`` with stage-6b verifier output folded in.

    Only when ``outcome.accept`` is True does the finding advance to
    ``FIXED``; otherwise it remains ``CONFIRMED`` (so the fixer can
    retry, up to the per-finding attempt cap).
    """
    if outcome.accept:
        return Finding(
            id=finding.id,
            file=finding.file,
            function=finding.function,
            claim=finding.claim,
            claim_signature=finding.claim_signature,
            severity=finding.severity,
            evidence_paths=finding.evidence_paths,
            suggested_repro=finding.suggested_repro,
            status=FindingStatus.FIXED.value,
            runs_to_confirm=finding.runs_to_confirm,
            total_runs=finding.total_runs,
            repro_path=finding.repro_path,
            repro_command=finding.repro_command,
            repro_output=finding.repro_output,
            invariant=invariant or finding.invariant,
            patch=patch_text,
            fix_attempts=fix_attempts,
            notes=[
                *finding.notes,
                f"fix-verify: accepted on attempt {fix_attempts}",
            ],
        )
    return Finding(
        id=finding.id,
        file=finding.file,
        function=finding.function,
        claim=finding.claim,
        claim_signature=finding.claim_signature,
        severity=finding.severity,
        evidence_paths=finding.evidence_paths,
        suggested_repro=finding.suggested_repro,
        status=finding.status,
        runs_to_confirm=finding.runs_to_confirm,
        total_runs=finding.total_runs,
        repro_path=finding.repro_path,
        repro_command=finding.repro_command,
        repro_output=finding.repro_output,
        invariant=invariant or finding.invariant,
        patch=finding.patch,
        fix_attempts=fix_attempts,
        notes=[
            *finding.notes,
            f"fix-verify: rejected on attempt {fix_attempts} — {outcome.rejection_reason[:200]}",
        ],
    )
