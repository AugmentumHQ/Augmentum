"""SQLite store for ``bug_finder_runs`` + ``bug_finder_findings``.

Two complementary tables:

* ``bug_finder_runs`` (migration 142) — one row per pipeline run with
  the full ``BugFinderRunReport`` blob as ``report_json``. Source of
  truth. Denormalized counts on the row power the list view without
  parsing the blob.
* ``bug_finder_findings`` (migration 225) — one row per finding,
  projected from the run blob at completion time. Enables cross-run
  analytics (signature recurrence, regression detection, per-workspace
  trend lines) without rehydrating every blob.

User-scoped per the multi-tenancy pattern — every public method takes
``user_id`` and scopes queries accordingly.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from typing import Any

from augmentum.bug_finder.findings import Finding, FindingStatus
from augmentum.bug_finder.orchestrator import BugFinderRunReport
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Lines like `path/to/file.py:14` or `path/to/file.py:14-22` in
# evidence_paths get parsed into (start, end). When the detector emits
# only a file path with no `:line`, both are None.
_LINE_RE = re.compile(r":(\d+)(?:-(\d+))?$")


class BugFinderRunStore:
    """Per-run history. Wraps a single ``aiosqlite`` connection."""

    def __init__(self, conn: Any) -> None:
        # conn is the aiosqlite connection. Typed as Any to avoid an
        # import dependency from this module on the state package.
        self._conn = conn

    async def start_run(
        self,
        *,
        run_id: str,
        user_id: str,
        job_id: str,
        git_url: str | None,
        workspace_id: str | None,
        started_at: float | None = None,
    ) -> None:
        """Insert a breadcrumb row when a run begins.

        Even if the job crashes mid-run, the user has a record of what
        was attempted. ``complete_run`` later upgrades the row with the
        final report.
        """
        ts = int(started_at or time.time())
        async with self._conn.execute(
            """
            INSERT OR REPLACE INTO bug_finder_runs (
                run_id, user_id, job_id, git_url, workspace_id,
                started_at, stop_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, 'running')
            """,
            (run_id, user_id, job_id, git_url, workspace_id, ts),
        ):
            await self._conn.commit()

    async def complete_run(
        self,
        report: BugFinderRunReport,
        *,
        user_id: str,
        job_id: str = "",
    ) -> None:
        """Write the final report into an existing row (or insert if missing).

        Computes denormalized counts + cost aggregates from the report.
        """
        findings = report.findings
        total = len(findings)
        confirmed = sum(
            1 for f in findings if f.status == FindingStatus.CONFIRMED.value
        )
        fixed = sum(
            1 for f in findings if f.status == FindingStatus.FIXED.value
        )
        fix_failed = sum(
            1 for f in findings if f.status == FindingStatus.FIX_FAILED.value
        )

        ledger = report.cost_ledger or []
        tokens_in = sum(e.tokens_in for e in ledger)
        tokens_out = sum(e.tokens_out for e in ledger)
        wallclock_ms = sum(e.wallclock_ms for e in ledger)

        report_json = json.dumps(_report_to_dict(report), default=str)

        async with self._conn.execute(
            """
            INSERT INTO bug_finder_runs (
                run_id, user_id, job_id, workspace_id, git_url,
                started_at, completed_at,
                stop_reason, stop_detail, containment_warning,
                findings_total, findings_confirmed, findings_fixed, findings_fix_failed,
                total_tokens_in, total_tokens_out, total_wallclock_ms,
                report_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                completed_at        = excluded.completed_at,
                stop_reason         = excluded.stop_reason,
                stop_detail         = excluded.stop_detail,
                containment_warning = excluded.containment_warning,
                workspace_id        = COALESCE(excluded.workspace_id, bug_finder_runs.workspace_id),
                findings_total      = excluded.findings_total,
                findings_confirmed  = excluded.findings_confirmed,
                findings_fixed      = excluded.findings_fixed,
                findings_fix_failed = excluded.findings_fix_failed,
                total_tokens_in     = excluded.total_tokens_in,
                total_tokens_out    = excluded.total_tokens_out,
                total_wallclock_ms  = excluded.total_wallclock_ms,
                report_json         = excluded.report_json
            """,
            (
                report.run_id, user_id, job_id,
                report.workspace_id or None,
                report.intake.get("git_url"),
                int(report.started_at),
                int(report.completed_at),
                report.stop_reason,
                report.stop_detail,
                # containment_warning is not on the dataclass; orchestrator
                # paths that need it set it dynamically. Tolerant getattr
                # keeps a vanilla BugFinderRunReport construction working.
                getattr(report, "containment_warning", "") or "",
                total, confirmed, fixed, fix_failed,
                tokens_in, tokens_out, wallclock_ms,
                report_json,
            ),
        ):
            await self._conn.commit()

        # Project findings into the normalized table. Failures here log
        # but don't fail the run write — the blob is the source of
        # truth, and the normalized rows can always be re-derived via
        # ``backfill_findings_from_runs``.
        try:
            await self._write_normalized_findings(report, user_id=user_id)
        except Exception as exc:  # noqa: BLE001 — log + continue
            log.warning(
                "bug_finder_findings_write_failed",
                run_id=report.run_id, user_id=user_id, error=str(exc),
            )

    async def _write_normalized_findings(
        self,
        report: BugFinderRunReport,
        *,
        user_id: str,
    ) -> None:
        """Replace the run's findings rows with a fresh projection of the
        report. INSERT OR REPLACE on (run_id, finding_id) so re-running
        complete_run is idempotent.

        Deletes existing rows for the run first so findings that drop
        out (re-run of an in-flight run that found fewer bugs the
        second time) don't ghost-haunt the table.
        """
        async with self._conn.execute(
            "DELETE FROM bug_finder_findings WHERE run_id = ? AND user_id = ?",
            (report.run_id, user_id),
        ):
            pass

        if not report.findings:
            await self._conn.commit()
            return

        detected_at = int(report.completed_at or report.started_at or time.time())
        rows: list[tuple[Any, ...]] = []
        for f in report.findings:
            line_start, line_end = _extract_line_range(f)
            rows.append((
                f.id,
                report.run_id,
                user_id,
                report.workspace_id or None,
                f.file,
                f.function or "<module>",
                line_start,
                line_end,
                f.claim,
                f.claim_signature,
                f.severity,
                f.status,
                f.runs_to_confirm,
                f.total_runs,
                # families_to_confirm / total_families default to 0 on
                # the Finding dataclass when ensemble tracking is off
                # (single-model runs) — that's the right value to
                # persist; queries on the column treat 0 as "not tracked".
                f.families_to_confirm,
                f.total_families,
                1 if f.repro_path else 0,
                1 if f.patch else 0,
                f.fix_attempts,
                detected_at,
            ))

        await self._conn.executemany(
            """
            INSERT OR REPLACE INTO bug_finder_findings (
                finding_id, run_id, user_id, workspace_id,
                file, function, line_start, line_end,
                claim, claim_signature, severity, status,
                runs_to_confirm, total_runs,
                families_to_confirm, total_families,
                has_repro, has_patch, fix_attempts,
                detected_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        await self._conn.commit()

    async def get_run(self, run_id: str, *, user_id: str) -> dict[str, Any] | None:
        """Return a single run by id, scoped to ``user_id``."""
        async with self._conn.execute(
            """
            SELECT run_id, user_id, job_id, workspace_id, git_url,
                   started_at, completed_at,
                   stop_reason, stop_detail, containment_warning,
                   findings_total, findings_confirmed, findings_fixed, findings_fix_failed,
                   total_tokens_in, total_tokens_out, total_wallclock_ms,
                   report_json
            FROM bug_finder_runs
            WHERE run_id = ? AND user_id = ?
            """,
            (run_id, user_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    async def mark_orphaned(
        self,
        run_id: str,
        *,
        user_id: str,
        stop_reason: str = "error",
        stop_detail: str = "job terminated without writing a report",
    ) -> bool:
        """Flip a stuck ``running`` row to a terminal state.

        Used by the reconciliation pass that runs whenever the read
        path notices that a row says ``running`` but its underlying
        background_job is already terminal. Without this, a job that
        crashes between ``start_run`` and ``complete_run`` (e.g. the
        process is killed mid-stage and the job_runner exhausts
        attempts) leaves the run record stuck in ``running`` forever
        — every caller polls indefinitely, no error surfaces to the
        UI / subagent.

        Returns True when a row was actually updated. Idempotent: a
        row already in a terminal state is left alone (the writer of
        the original terminal state knows more than we do)."""
        completed_at = int(time.time())
        async with self._conn.execute(
            """
            UPDATE bug_finder_runs
               SET stop_reason = ?,
                   stop_detail = ?,
                   completed_at = COALESCE(completed_at, ?)
             WHERE run_id = ?
               AND user_id = ?
               AND (stop_reason IS NULL OR stop_reason = 'running')
            """,
            (stop_reason, stop_detail, completed_at, run_id, user_id),
        ) as cursor:
            updated = cursor.rowcount or 0
        await self._conn.commit()
        return updated > 0

    async def list_runs(
        self,
        *,
        user_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List a user's recent runs, newest first."""
        async with self._conn.execute(
            """
            SELECT run_id, user_id, job_id, workspace_id, git_url,
                   started_at, completed_at,
                   stop_reason, stop_detail, containment_warning,
                   findings_total, findings_confirmed, findings_fixed, findings_fix_failed,
                   total_tokens_in, total_tokens_out, total_wallclock_ms,
                   NULL AS report_json
            FROM bug_finder_runs
            WHERE user_id = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (user_id, max(1, min(limit, 200))),
        ) as cursor:
            rows = await cursor.fetchall()
        # Hydrate without the heavy report_json blob — list view doesn't
        # need it; the detail endpoint pulls the full row.
        return [_row_to_dict(r, include_report=False) for r in rows]

    # ------------------------------------------------------------------
    # Cross-run analytics (over the normalized table)
    # ------------------------------------------------------------------

    async def list_findings_by_signature(
        self,
        signature: str,
        *,
        user_id: str,
        workspace_id: str = "",
        status: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """All findings of one signature across runs.

        Powers the "show me every CONFIRMED injection across all my
        runs" query — recurrence detection, regression alerting.
        Always scoped to ``user_id``.
        """
        query = (
            "SELECT finding_id, run_id, workspace_id, file, function, "
            "line_start, line_end, claim, claim_signature, severity, "
            "status, runs_to_confirm, total_runs, has_repro, has_patch, "
            "fix_attempts, detected_at "
            "FROM bug_finder_findings "
            "WHERE user_id = ? AND claim_signature = ?"
        )
        params: list[Any] = [user_id, signature]
        if workspace_id:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(max(1, min(limit, 500)))
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [_finding_row_to_dict(r) for r in rows]

    async def signature_recurrence(
        self,
        *,
        user_id: str,
        workspace_id: str = "",
        min_hits: int = 2,
    ) -> list[dict[str, Any]]:
        """Aggregate counts per (signature, file) across all runs.

        Returns the rows where a (signature, file) tuple appears ``>=
        min_hits`` times — i.e., "this bug keeps coming back at this
        location". Used by the pattern-memory subsystem (task #36) to
        seed the planner with "things we've seen before".
        """
        query = (
            "SELECT claim_signature, file, "
            "COUNT(DISTINCT run_id) AS hit_count, "
            "MIN(detected_at) AS first_seen, "
            "MAX(detected_at) AS last_seen, "
            "SUM(CASE WHEN status = 'fixed' THEN 1 ELSE 0 END) AS fixed_count "
            "FROM bug_finder_findings "
            "WHERE user_id = ?"
        )
        params: list[Any] = [user_id]
        if workspace_id:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        query += (
            " GROUP BY claim_signature, file "
            "HAVING hit_count >= ? "
            "ORDER BY hit_count DESC, last_seen DESC"
        )
        params.append(max(1, min_hits))
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        cols = (
            "claim_signature", "file", "hit_count",
            "first_seen", "last_seen", "fixed_count",
        )
        return [dict(zip(cols, r, strict=False)) for r in rows]

    async def workspace_finding_history(
        self,
        workspace_id: str,
        *,
        user_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Every finding ever recorded for one workspace, newest first.

        Powers the per-project trend line: "did the last audit improve
        on the one before". The list is bounded — callers paginate by
        ``detected_at``.
        """
        async with self._conn.execute(
            "SELECT finding_id, run_id, file, function, line_start, "
            "line_end, claim, claim_signature, severity, status, "
            "runs_to_confirm, total_runs, has_repro, has_patch, "
            "fix_attempts, detected_at "
            "FROM bug_finder_findings "
            "WHERE user_id = ? AND workspace_id = ? "
            "ORDER BY detected_at DESC LIMIT ?",
            (user_id, workspace_id, max(1, min(limit, 1000))),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_finding_row_to_dict(r, has_workspace=False) for r in rows]

    async def backfill_findings_from_runs(
        self,
        *,
        user_id: str = "",
        run_id: str = "",
    ) -> int:
        """Re-derive normalized findings rows from existing report_json blobs.

        Called on first migration to populate the new table from runs
        that pre-date it, or as a recovery hatch when the normalized
        table got out of sync. Always treats the blob as source of
        truth.

        Returns the count of runs backfilled.
        """
        query = (
            "SELECT run_id, user_id, workspace_id, started_at, completed_at, "
            "stop_reason, stop_detail, containment_warning, report_json "
            "FROM bug_finder_runs WHERE report_json IS NOT NULL"
        )
        params: list[Any] = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        count = 0
        for row in rows:
            (rid, uid, wsid, started_at, completed_at,
             stop_reason, stop_detail, containment, blob) = row
            try:
                report_dict = json.loads(blob) if blob else None
            except (TypeError, ValueError):
                continue
            if not isinstance(report_dict, dict):
                continue
            report = _rehydrate_report(
                report_dict,
                run_id=rid,
                workspace_id=wsid,
                started_at=started_at,
                completed_at=completed_at,
                stop_reason=stop_reason,
                stop_detail=stop_detail,
                containment_warning=containment,
            )
            await self._write_normalized_findings(report, user_id=uid)
            count += 1
        return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_line_range(finding: Finding) -> tuple[int | None, int | None]:
    """Parse `file:line` or `file:start-end` out of evidence_paths.

    The first evidence path with a numeric trailer wins. Detectors emit
    line ranges inconsistently — this normalizes them so the column is
    queryable without parsing the blob again.
    """
    for path in finding.evidence_paths:
        m = _LINE_RE.search(path)
        if not m:
            continue
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        if end < start:
            start, end = end, start
        return start, end
    return None, None


def _finding_row_to_dict(row: Any, *, has_workspace: bool = True) -> dict[str, Any]:
    cols = (
        "finding_id", "run_id",
    )
    if has_workspace:
        cols += ("workspace_id",)
    cols += (
        "file", "function", "line_start", "line_end",
        "claim", "claim_signature", "severity", "status",
        "runs_to_confirm", "total_runs",
        "has_repro", "has_patch", "fix_attempts", "detected_at",
    )
    return dict(zip(cols, row, strict=False))


def _rehydrate_report(
    report_dict: dict[str, Any],
    *,
    run_id: str,
    workspace_id: str | None,
    started_at: int | None,
    completed_at: int | None,
    stop_reason: str | None,
    stop_detail: str | None,
    containment_warning: str | None,
) -> BugFinderRunReport:
    """Reconstruct just enough of a BugFinderRunReport to drive the
    normalized-findings projection.

    Avoids importing the heavy dataclasses path — only the fields the
    projection actually reads are populated. The Finding rehydration is
    deliberately lenient (missing fields default).
    """
    findings_raw = report_dict.get("findings") or []
    findings: list[Finding] = []
    for row in findings_raw:
        if not isinstance(row, dict):
            continue
        evidence = row.get("evidence_paths") or ()
        if isinstance(evidence, list):
            evidence = tuple(evidence)
        findings.append(Finding(
            id=str(row.get("id") or ""),
            file=str(row.get("file") or ""),
            function=str(row.get("function") or "<module>"),
            claim=str(row.get("claim") or ""),
            claim_signature=str(row.get("claim_signature") or "other"),
            severity=str(row.get("severity") or "medium"),
            evidence_paths=evidence,
            suggested_repro=str(row.get("suggested_repro") or ""),
            status=str(row.get("status") or "speculative"),
            runs_to_confirm=int(row.get("runs_to_confirm") or 0),
            total_runs=int(row.get("total_runs") or 0),
            repro_path=str(row.get("repro_path") or ""),
            repro_command=str(row.get("repro_command") or ""),
            repro_output=str(row.get("repro_output") or ""),
            invariant=str(row.get("invariant") or ""),
            patch=str(row.get("patch") or ""),
            fix_attempts=int(row.get("fix_attempts") or 0),
            notes=list(row.get("notes") or []),
        ))

    # We only need the few fields the projection reads. Building a real
    # BugFinderRunReport here would force WorkspaceBaseline + ledger
    # reconstruction; instead we hand-build a stand-in object with the
    # right attributes. Duck-typed against the projection's reads.
    class _StubReport:
        pass

    stub = _StubReport()
    stub.run_id = run_id
    stub.workspace_id = workspace_id or ""
    stub.started_at = started_at or 0
    stub.completed_at = completed_at or 0
    stub.stop_reason = stop_reason or ""
    stub.stop_detail = stop_detail or ""
    stub.containment_warning = containment_warning or ""
    stub.findings = findings
    stub.intake = report_dict.get("intake") or {}
    stub.cost_ledger = []
    return stub  # type: ignore[return-value]


_COLUMNS = (
    "run_id", "user_id", "job_id", "workspace_id", "git_url",
    "started_at", "completed_at",
    "stop_reason", "stop_detail", "containment_warning",
    "findings_total", "findings_confirmed", "findings_fixed", "findings_fix_failed",
    "total_tokens_in", "total_tokens_out", "total_wallclock_ms",
    "report_json",
)


def _row_to_dict(row: Any, *, include_report: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = dict(zip(_COLUMNS, row, strict=False))
    if include_report and out.get("report_json"):
        try:
            out["report"] = json.loads(out["report_json"])
        except (TypeError, ValueError):
            out["report"] = None
    out.pop("report_json", None)
    return out


def _report_to_dict(report: BugFinderRunReport) -> dict[str, Any]:
    """Make a JSON-safe dict from a BugFinderRunReport.

    ``asdict`` walks the nested dataclasses (Finding, CostLedgerEntry,
    WorkspaceBaseline) automatically. We post-process to coerce any
    non-JSON values (tuples → lists) for sqlite-friendly serialization.
    """
    d = asdict(report)
    return _coerce_json(d)


def _coerce_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_coerce_json(v) for v in value]
    if isinstance(value, list):
        return [_coerce_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce_json(v) for k, v in value.items()}
    return value
