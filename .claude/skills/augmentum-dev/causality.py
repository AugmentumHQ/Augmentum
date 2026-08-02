"""Causality layer — distinguish "the codebase regressed" from
"the scanner itself changed."

When a metric moves against baseline, the regression may be:
  (a) real — the underlying code accumulated debt, or
  (b) self-induced — the scanner itself was edited (parser fixed,
      heuristic loosened, suppression removed).

Reporting (b) as a regression is the broken-zero-baseline class of
bug we hit when fixing the wiring parser regex: a parser that had
been silently returning 0 started returning 22 the moment we made
it work, and the audit blamed the codebase.

This module maps every metric to the source file(s) that define
how it's computed, then asks "did any of those files change since
the baseline was set?" If yes, the regression is tagged so the
score / display can downplay it.

The scanner→file map is HAND-CURATED in ``METRIC_OWNERS``. Adding
a new scanner means adding one entry. The map is verified by
``test_causality_map_complete`` to prevent silent drift.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

# Map of (metric_category, metric_key) → list of source files whose
# changes would alter how this metric is computed. Wildcards aren't
# supported; list each file explicitly so the reverse-lookup is
# unambiguous.
#
# A metric_key of "*" matches any sub-metric in that category — useful
# when one scanner file produces multiple metrics.
METRIC_OWNERS: dict[tuple[str, str], list[str]] = {
    # Legacy regex-based scanners — each metric's source file is the
    # whole script.
    ("wiring",       "*"): [".claude/skills/augmentum-dev/scripts/validate_wiring.py"],
    ("dead_code",    "*"): [".claude/skills/augmentum-dev/scripts/dead_code.py"],
    ("code_quality", "*"): [".claude/skills/augmentum-dev/scripts/code_quality.py"],
    ("runtime",      "*"): [".claude/skills/augmentum-dev/scripts/runtime_checks.py"],
    ("security",     "*"): [".claude/skills/augmentum-dev/scripts/security_check.py"],
    ("coverage",     "*"): [".claude/skills/augmentum-dev/scripts/test_coverage.py"],
    ("red_team",     "*"): [".claude/skills/augmentum-dev/scripts/red_team_scan.py"],
    ("doc_facts",    "*"): [
        ".claude/skills/augmentum-dev/scripts/audit.py",  # _check_doc_facts lives here
        ".claude/skills/augmentum-dev/facts/registry.py",
        ".claude/skills/augmentum-dev/model/ingesters/tables.py",
        ".claude/skills/augmentum-dev/model/ingesters/migrations.py",
    ],
    ("exceptions",   "*"): [".claude/skills/augmentum-dev/scripts/audit.py"],
    ("deps",         "*"): [".claude/skills/augmentum-dev/scripts/audit.py"],
    # Phase 1+ model queries — each query module is its own metric source.
    ("model", "orphaned_endpoints"): [
        ".claude/skills/augmentum-dev/queries/orphaned_endpoints.py",
        ".claude/skills/augmentum-dev/model/ingesters/endpoints.py",
        ".claude/skills/augmentum-dev/model/ingesters/js_calls.py",
    ],
    ("model", "incomplete_settings"): [
        ".claude/skills/augmentum-dev/queries/incomplete_settings.py",
        ".claude/skills/augmentum-dev/model/ingesters/settings.py",
    ],
}


@dataclass
class Provenance:
    """Why (we think) a metric moved.

    ``self_changed`` is True when the metric's owning files were
    modified in the comparison window. ``files_changed`` lists the
    specific paths so the audit display can surface them.
    ``last_commit_sha`` is best-effort — empty if git isn't available
    or the file is untracked.
    """
    self_changed: bool
    files_changed: list[str]
    last_commit_sha: str = ""


def _files_for(category: str, key: str) -> list[str]:
    """Resolve owner files for a (category, key) metric, falling back
    to the wildcard ``(category, "*")`` entry."""
    entries = METRIC_OWNERS.get((category, key))
    if entries is not None:
        return entries
    return METRIC_OWNERS.get((category, "*"), [])


def _git_log_since(repo_root: Path, file_rel: str, since_ts: float) -> str:
    """Return the SHA of the most recent commit touching ``file_rel``
    after ``since_ts``, or "" if git isn't available / no such commit.

    Uses --since with a unix timestamp; quietly degrades if the file
    isn't tracked or the repo isn't a git checkout.
    """
    try:
        proc = subprocess.run(
            [
                "git", "-C", str(repo_root), "log",
                "-1", "--format=%H",
                f"--since=@{int(since_ts)}",
                "--",
                file_rel,
            ],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def explain_regression(
    repo_root: Path,
    category: str,
    key: str,
    *,
    since_ts: float,
) -> Provenance:
    """For a metric that regressed against baseline, decide whether
    the scanner itself changed in the comparison window.

    ``since_ts`` is the unix timestamp of when the baseline was
    written. Files modified strictly after that are considered
    self-changes.
    """
    files = _files_for(category, key)
    changed: list[str] = []
    last_sha = ""
    for rel in files:
        path = repo_root / rel
        if not path.exists():
            continue
        if path.stat().st_mtime > since_ts:
            changed.append(rel)
            sha = _git_log_since(repo_root, rel, since_ts)
            if sha and not last_sha:
                last_sha = sha
    return Provenance(
        self_changed=bool(changed),
        files_changed=changed,
        last_commit_sha=last_sha,
    )
