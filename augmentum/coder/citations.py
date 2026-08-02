"""Citation ledger — structured claim→proof provenance for a coder turn.

The one genuinely-new primitive from the companion-brief design
(2026-07-27 spec §7.4). A coder turn makes claims by *doing things* —
writing files, running tests, probing services. The Promise tree records
the plan and ``closeout_json`` records the aggregate outcome, but neither
ties a specific claim to the specific evidence that supports it, and
``Promise.evidence`` is an unstructured ``str`` we deliberately do NOT
overload. This ledger is that missing spine: one row per evidence-bearing
tool result, keyed by ``(turn_run_id, tool_call_seq)``.

Two readers consume it:

* **The brief** (``brief-panel.js``) renders each citation as a dropdown
  that deep-links into the ``mountReviewPanel`` diff — so a context-poor
  user can drill from "the agent says it added X" into the exact changed
  lines before approving. The brief is the decision surface; citations
  make that decision honest.
* **The gate** (extended ``run_verifier``) feeds citations to the
  cross-model judge as concrete evidence, alongside the oracle summary.

Emit is cheap and pure: :func:`citations_from_tool_result` is a pure
classifier over the same ``tool_result`` payload the ledger already sees
(``ledger.CoderTurnLedger._accumulate``). The ledger accumulates the rows
in memory (like ``oracle_calls``) and bulk-persists them once at
``finish()`` — never a per-tool INSERT on the live DB.

``line_start``/``line_end`` are ``None`` in the MVP (file_write has no
range; code_edit/apply_patch do — extracted opportunistically when the
edit payload carries explicit line numbers). The column exists from day
one so per-line fidelity lands with no schema change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from augmentum.coder.oracle_telemetry import WRITE_TOOLS
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# evidence_kind values. 'write' is a change-claim; the rest are oracle
# reaches (mirror oracle_telemetry.classify_oracle_kind), carrying an outcome.
EVIDENCE_WRITE = "write"
_MAX_CITATIONS = 300  # per-turn cap — matches the spirit of oracle_calls' 200


@dataclass(frozen=True, slots=True)
class Citation:
    """One evidence row. Scope fields (turn_run_id, user_id, …) are constant
    per turn and supplied at :func:`save_citations` time, not here."""

    tool_call_seq: int
    evidence_kind: str          # write | test | probe | browser | shell_check
    file: str = ""
    line_start: int | None = None
    line_end: int | None = None
    evidence_ref: str = ""      # checkpoint id / command / probe target
    outcome: str = ""           # green | red | unknown (oracle kinds); '' for writes


def _line_range(entry: dict[str, Any]) -> tuple[int | None, int | None]:
    """Best-effort explicit line range from an edit payload. Returns
    (None, None) unless the payload carries unambiguous integer line
    numbers — we never guess a range for a whole-file write."""
    for start_key, end_key in (("start_line", "end_line"), ("line_start", "line_end"), ("line", "line")):
        s = entry.get(start_key)
        e = entry.get(end_key)
        if isinstance(s, int) and isinstance(e, int) and s > 0 and e >= s:
            return s, e
    return None, None


def citations_from_tool_result(
    *,
    seq: int,
    tool: str,
    tool_input: dict[str, Any] | None,
    success: Any,
    checkpoint: str = "",
    oracle_kind: str | None = None,
    outcome: str = "",
    command: str = "",
) -> list[Citation]:
    """Pure: classify one tool result into zero or more citation rows.

    * A SUCCESSFUL write tool → one ``write`` citation per file it touched
      (a batch edit touches many), each with a best-effort line range.
    * An oracle tool (test/probe/browser/shell_check) → one citation
      carrying the outcome and the command/target it probed.

    A failed write is not a claim, so it produces no citation. Nothing else
    is evidence, so returns ``[]``.
    """
    ti = tool_input if isinstance(tool_input, dict) else {}

    if oracle_kind:
        ref = command or str(ti.get("command") or ti.get("url") or ti.get("target") or "")
        return [Citation(
            tool_call_seq=seq, evidence_kind=oracle_kind,
            evidence_ref=ref[:400], outcome=outcome or "unknown",
        )]

    if tool in WRITE_TOOLS and success is not False:
        rows: list[Citation] = []
        seen: set[str] = set()
        # Single-file writes carry path/file_path; batch edits carry an
        # ``edits`` list of {path, ...}. Mirror ledger.changed_files logic.
        for key in ("path", "file_path"):
            p = ti.get(key)
            if isinstance(p, str) and p and p not in seen:
                seen.add(p)
                ls, le = _line_range(ti)
                rows.append(Citation(
                    tool_call_seq=seq, evidence_kind=EVIDENCE_WRITE, file=p,
                    line_start=ls, line_end=le, evidence_ref=checkpoint[:120],
                ))
        edits = ti.get("edits")
        if isinstance(edits, list):
            for edit in edits:
                if not isinstance(edit, dict):
                    continue
                p = edit.get("path")
                if isinstance(p, str) and p and p not in seen:
                    seen.add(p)
                    ls, le = _line_range(edit)
                    rows.append(Citation(
                        tool_call_seq=seq, evidence_kind=EVIDENCE_WRITE, file=p,
                        line_start=ls, line_end=le, evidence_ref=checkpoint[:120],
                    ))
        return rows

    return []


async def save_citations(
    conn: Any,
    *,
    turn_run_id: str,
    user_id: str,
    workspace_id: str = "",
    run_id: str = "",
    citations: list[Citation],
) -> None:
    """Bulk-persist a turn's citation rows. Best-effort — a persistence
    failure must never break turn close (telemetry, not correctness)."""
    if conn is None or not turn_run_id or not user_id or not citations:
        return
    rows = citations[:_MAX_CITATIONS]
    try:
        await conn.executemany(
            "INSERT INTO coder_run_citations "
            "(turn_run_id, user_id, workspace_id, run_id, tool_call_seq, "
            " file, line_start, line_end, evidence_kind, evidence_ref, outcome) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    turn_run_id, user_id, workspace_id or "", run_id or "",
                    c.tool_call_seq, c.file, c.line_start, c.line_end,
                    c.evidence_kind, c.evidence_ref, c.outcome,
                )
                for c in rows
            ],
        )
        await conn.commit()
    except Exception as exc:  # noqa: BLE001
        # Best-effort telemetry must NEVER be able to break (or hang) turn
        # close. Log a compact string, not exc_info — a traceback render can
        # deadlock pytest's output capture on Windows cp1252 stdout.
        log.warning(
            "coder_citations_persist_failed",
            turn_run_id=turn_run_id, error=str(exc)[:200],
        )


async def load_citations(
    conn: Any, *, turn_run_id: str, user_id: str,
) -> list[dict[str, Any]]:
    """Load a turn's citations (user-scoped), ordered by tool call sequence.
    Returns ``[]`` on any miss — a brief with no citations is a valid state."""
    if conn is None or not turn_run_id or not user_id:
        return []
    try:
        cur = await conn.execute(
            "SELECT tool_call_seq, file, line_start, line_end, evidence_kind, "
            "       evidence_ref, outcome "
            "FROM coder_run_citations "
            "WHERE turn_run_id = ? AND user_id = ? "
            "ORDER BY tool_call_seq ASC, id ASC",
            (turn_run_id, user_id),
        )
        fetched = await cur.fetchall()
    except Exception:
        log.warning("coder_citations_load_failed", turn_run_id=turn_run_id, exc_info=True)
        return []
    return [
        {
            "tool_call_seq": r[0], "file": r[1],
            "line_start": r[2], "line_end": r[3],
            "evidence_kind": r[4], "evidence_ref": r[5], "outcome": r[6],
        }
        for r in fetched
    ]


__all__ = [
    "EVIDENCE_WRITE",
    "Citation",
    "citations_from_tool_result",
    "save_citations",
    "load_citations",
]
