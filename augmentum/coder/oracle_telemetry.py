"""Oracle-choice telemetry — pure classifiers for the verification spine.

Phase 2 of docs/superpowers/specs/2026-07-06-coder-verification-spine-design.md:
observe, per coder turn, WHICH oracle the agent reached for (test run, browser
probe, service/http probe, check-shaped shell command — or none) and whether
the last oracle after the last write went red or green. Observational only —
nothing here blocks a turn; the numbers exist so future gating decisions come
from evidence instead of taste.

Everything in this module is a pure function over the tool-call stream the
turn ledger already sees (``tool_call`` / ``tool_result`` event payloads), so
it costs nothing at runtime and is trivially unit-testable. The ledger
accumulates the classified calls and folds the summary into
``coder_turn_runs.metrics_json['oracle']`` plus one ``oracle_summary`` event
at turn close — no new table, no new column.

Honesty notes (what this telemetry can and cannot claim):

* ``kind`` is derived from the TOOL, not the agent's intent. A
  ``browser_screenshot`` counts as a browser-kind oracle call even though a
  screenshot alone is weak evidence — the summary is about *reach*, not
  proof quality.
* ``outcome`` is best-effort: the ``success`` flag on the tool result plus
  a red/green sniff of the 500-char output preview. When neither signals,
  the outcome is ``"unknown"`` — never coerced to green.
* The agent's *stated reason* for skipping verification lives in closeout
  prose, not the event stream; ``no_oracle_done`` is therefore a boolean,
  not an explanation. Deliberate Phase-2 scope cut.
"""
from __future__ import annotations

import re
from typing import Any

# Tools that ARE oracles by construction.
_TEST_TOOLS = frozenset({"test_run"})
_PROBE_TOOLS = frozenset({"service_probe", "http_probe", "db_probe", "verify_preview"})
# Any browser tool counts as reaching for the browser oracle (mirrors the
# ledger's existing ``browser_checks`` semantics). browser_open alone is a
# weak check, but it is still the UI-verification surface being used.
_BROWSER_PREFIX = "browser_"

# Write tools — a turn "made a claim" when one of these succeeded. Mirrors
# the ledger's files_touched/changed_files detection; keep the two in sync.
WRITE_TOOLS = frozenset({"file_write", "code_edit", "code_edit_batch", "apply_patch"})

# Check-shaped shell commands: tests, linters, type checkers, builds. The
# pattern is deliberately word-boundary anchored — "form" ∈ "transformers"
# was a real bug class in keyword routing; don't reintroduce it here.
_SHELL_CHECK_RE = re.compile(
    r"\b("
    r"pytest|unittest|jest|vitest|mocha|playwright"
    r"|npm (?:run )?test|yarn test|pnpm test"
    r"|go test|cargo (?:test|check)|make (?:test|check|lint)"
    r"|ruff|eslint|flake8|pylint|mypy|tsc|black --check"
    r"|npm run build|cargo build|go build|go vet"
    r")\b"
)

# Red/green sniffing over the 500-char output preview. Checked in order:
# explicit failure markers win over pass markers (pytest's "1 failed,
# 3 passed" must read red).
_RED_RE = re.compile(
    r"(\b[1-9]\d* (?:failed|errors?)\b|\bFAILED\b|\bAssertionError\b"
    r"|\bTraceback \(most recent call last\)|\bERROR\b|✗|\bFAIL\b)"
)
_GREEN_RE = re.compile(
    r"(\b\d+ passed\b|\ball checks passed\b|\bok\b|\bPASSED\b|\bpassing\b|✓)",
    re.IGNORECASE,
)


def classify_oracle_kind(tool: str, tool_input: dict[str, Any] | None = None) -> str | None:
    """Classify a tool call as an oracle kind, or None if it isn't one.

    Kinds: ``test`` · ``browser`` · ``probe`` · ``shell_check``.
    """
    name = (tool or "").strip()
    if name in _TEST_TOOLS:
        return "test"
    if name in _PROBE_TOOLS:
        return "probe"
    if name.startswith(_BROWSER_PREFIX):
        return "browser"
    if name in {"shell_exec", "shell_read"}:
        command = str((tool_input or {}).get("command") or "")
        if _SHELL_CHECK_RE.search(command):
            return "shell_check"
    return None


def classify_outcome(*, success: Any, output_preview: str) -> str:
    """Best-effort red/green from the tool result. ``unknown`` when unclear.

    ``success=False`` is authoritative red (the tool itself failed or the
    command exited non-zero where the tool surfaces that). A True/absent
    success flag falls through to preview sniffing — test_run returns
    success=True for a run that PARSED even when tests failed.
    """
    if success is False:
        return "red"
    preview = output_preview or ""
    if _RED_RE.search(preview):
        return "red"
    if _GREEN_RE.search(preview):
        return "green"
    return "unknown"


def summarize(
    *,
    wrote: bool,
    last_write_seq: int,
    oracle_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fold the turn's classified oracle calls into the closeout summary.

    ``oracle_calls`` entries: {"seq": int, "kind": str, "tool": str,
    "outcome": str}. ``no_oracle_done`` is the spine's target metric:
    the turn changed files but ran no oracle AFTER the last write — the
    check that ran before the final edit proved a stale claim.
    """
    after_write = [c for c in oracle_calls if int(c.get("seq") or 0) > last_write_seq]
    relevant = after_write if wrote else oracle_calls
    last_outcome = str(relevant[-1]["outcome"]) if relevant else ""
    return {
        "wrote": wrote,
        "oracle_calls": len(oracle_calls),
        "kinds": sorted({str(c.get("kind") or "") for c in oracle_calls} - {""}),
        "verified_after_last_write": bool(after_write),
        "last_outcome": last_outcome,
        "no_oracle_done": wrote and not after_write,
    }
