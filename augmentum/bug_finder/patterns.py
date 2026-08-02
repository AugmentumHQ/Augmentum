"""Cross-run pattern memory.

Durable memory of "bugs that have appeared in this workspace before".
Distinct from the read-time `signature_recurrence` aggregation: this
table is *written to* every time a run completes, so the planner can
ask "what should I be particularly attentive to" before each new run
without recomputing from blobs.

The mental model is borrowed from Semgrep (TPs eventually become rules)
and Anthropic's bug-finder research (recurring findings deserve more
detector attention than first-time observations). We don't generate
Semgrep rules — that's a future leap — but we do compound the priors.

User-scoped per the auth pattern.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any

from augmentum.bug_finder.findings import Finding, FindingStatus
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pattern:
    """One row from `bug_finder_patterns`. Read-only."""

    pattern_id: str
    user_id: str
    workspace_id: str
    claim_signature: str
    file: str
    first_seen_at: int
    last_seen_at: int
    last_run_id: str
    hit_count: int
    fix_count: int
    speculative_count: int
    sample_claim: str
    last_severity: str
    note: str

    @property
    def is_unresolved(self) -> bool:
        """Pattern with no successful fixes — likely still present."""
        return self.fix_count == 0


def _pattern_id(user_id: str, workspace_id: str, signature: str, file: str) -> str:
    blob = "|".join((user_id, workspace_id, signature, file))
    return "pat_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PatternStore:
    """Read/write access to `bug_finder_patterns`.

    Wraps an `aiosqlite` connection. Same pattern as `BugFinderRunStore`."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def update_from_findings(
        self,
        findings: list[Finding],
        *,
        run_id: str,
        user_id: str,
        workspace_id: str,
        seen_at: int | None = None,
    ) -> int:
        """Upsert each finding into the pattern table.

        Counters increment per-finding; sample_claim updates only on
        first creation (subsequent runs keep the original — stable text
        is easier for the planner to recognize across runs).

        Returns the number of distinct patterns affected.
        """
        ts = int(seen_at or time.time())
        updated: set[str] = set()
        for f in findings:
            pid = _pattern_id(user_id, workspace_id, f.claim_signature, f.file)
            updated.add(pid)
            fix_delta = 1 if f.status == FindingStatus.FIXED.value else 0
            spec_delta = 1 if f.status == FindingStatus.SPECULATIVE.value else 0
            # First-insert vs update is handled by ON CONFLICT.
            async with self._conn.execute(
                """
                INSERT INTO bug_finder_patterns (
                    pattern_id, user_id, workspace_id,
                    claim_signature, file,
                    first_seen_at, last_seen_at, last_run_id,
                    hit_count, fix_count, speculative_count,
                    sample_claim, last_severity
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(pattern_id) DO UPDATE SET
                    last_seen_at      = excluded.last_seen_at,
                    last_run_id       = excluded.last_run_id,
                    hit_count         = bug_finder_patterns.hit_count + 1,
                    fix_count         = bug_finder_patterns.fix_count + ?,
                    speculative_count = bug_finder_patterns.speculative_count + ?,
                    last_severity     = excluded.last_severity
                """,
                (
                    pid, user_id, workspace_id or "",
                    f.claim_signature, f.file,
                    ts, ts, run_id,
                    fix_delta, spec_delta,
                    (f.claim or "")[:400],
                    f.severity,
                    fix_delta, spec_delta,
                ),
            ):
                pass
        await self._conn.commit()
        return len(updated)

    async def list_patterns(
        self,
        *,
        user_id: str,
        workspace_id: str = "",
        signature: str = "",
        min_hit_count: int = 1,
        unresolved_only: bool = False,
        limit: int = 50,
    ) -> list[Pattern]:
        """Query patterns. Most-recent-first."""
        query = (
            "SELECT pattern_id, user_id, workspace_id, claim_signature, file, "
            "first_seen_at, last_seen_at, last_run_id, "
            "hit_count, fix_count, speculative_count, "
            "sample_claim, last_severity, note "
            "FROM bug_finder_patterns "
            "WHERE user_id = ? AND hit_count >= ?"
        )
        params: list[Any] = [user_id, max(1, min_hit_count)]
        if workspace_id:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        if signature:
            query += " AND claim_signature = ?"
            params.append(signature)
        if unresolved_only:
            query += " AND fix_count = 0"
        query += " ORDER BY last_seen_at DESC LIMIT ?"
        params.append(max(1, min(limit, 500)))
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_pattern(r) for r in rows]

    async def forget_pattern(
        self,
        pattern_id: str,
        *,
        user_id: str,
    ) -> bool:
        """User-initiated forget. Returns True when a row was deleted."""
        async with self._conn.execute(
            "DELETE FROM bug_finder_patterns WHERE pattern_id = ? AND user_id = ?",
            (pattern_id, user_id),
        ) as cursor:
            deleted = cursor.rowcount > 0
        await self._conn.commit()
        return deleted

    async def annotate(
        self,
        pattern_id: str,
        note: str,
        *,
        user_id: str,
    ) -> bool:
        """Attach a free-form note. Helpful when the user knows a pattern
        is intentional (e.g., 'pickle is fine — internal trust') and
        wants to bias future planner runs."""
        async with self._conn.execute(
            "UPDATE bug_finder_patterns SET note = ? "
            "WHERE pattern_id = ? AND user_id = ?",
            (note[:1000], pattern_id, user_id),
        ) as cursor:
            updated = cursor.rowcount > 0
        await self._conn.commit()
        return updated


# ---------------------------------------------------------------------------
# Renderer — produces the planner-prompt prefix block
# ---------------------------------------------------------------------------


def render_pattern_brief(
    patterns: list[Pattern],
    *,
    max_lines: int = 12,
) -> str:
    """Render a compact, prompt-friendly summary of prior patterns.

    Returns "" when there are no patterns — caller passes through to
    the unmodified system prompt.

    The shape is deliberately terse: file, signature, hit count, fix
    rate. Models latch onto short, structured lists better than prose.
    A 'note' (user-supplied) wins display priority over the automatic
    sample_claim.
    """
    if not patterns:
        return ""
    lines: list[str] = []
    lines.append("## Patterns observed in prior runs of this workspace")
    lines.append("")
    lines.append(
        "Findings that appeared in earlier audits. Use this as a prior "
        "— do not trust it blindly; the code may have changed.",
    )
    lines.append("")
    head = f"| {'File':30s} | {'Signature':18s} | Hits | Fixed | Last severity | Note |"
    sep = "|" + ("-" * 32) + "|" + ("-" * 20) + "|" + "-----" + "|" + "-------" + "|" + "----------------|--------|"
    lines.append(head)
    lines.append(sep)
    for p in patterns[:max_lines]:
        note = p.note or p.sample_claim
        if len(note) > 60:
            note = note[:57] + "..."
        lines.append(
            f"| {p.file[:30]:30s} | {p.claim_signature[:18]:18s} "
            f"| {p.hit_count:4d} | {p.fix_count:5d} | {p.last_severity[:14]:14s} | {note} |",
        )
    if len(patterns) > max_lines:
        lines.append("")
        lines.append(f"… and {len(patterns) - max_lines} more pattern(s) not shown.")
    return "\n".join(lines)


def _row_to_pattern(row: Any) -> Pattern:
    return Pattern(
        pattern_id=row[0],
        user_id=row[1],
        workspace_id=row[2],
        claim_signature=row[3],
        file=row[4],
        first_seen_at=row[5],
        last_seen_at=row[6],
        last_run_id=row[7],
        hit_count=row[8],
        fix_count=row[9],
        speculative_count=row[10],
        sample_claim=row[11],
        last_severity=row[12],
        note=row[13],
    )


# Mirror to_dict shape on Pattern for JSON serialization in routes.
def pattern_to_dict(p: Pattern) -> dict[str, Any]:
    return asdict(p)
