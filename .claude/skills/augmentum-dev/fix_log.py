"""Fix-event recorder.

Every audit run can deposit one row into the model's ``fix_events``
table:
  * commit_sha       — current HEAD (or "" if not in a git repo)
  * ts               — wall clock at audit time
  * audit_delta_json — {metric_key: [baseline, current, delta], ...}
  * detected_pattern — best-effort label classifying the dominant
                       metric movement (e.g. "self_change_only",
                       "wiring_drop", "settings_thrash")

Pattern memory is a Phase 5 concern, but recording the data is the
cheap step we do today so we have history once the analysis lands.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from pathlib import Path

# Cap stored audit_delta_json so a pathological 500-metric audit
# can't bloat the row past a sensible budget. Truncates from the
# tail (oldest insert order); each delta entry is ~80 bytes so the
# cap fits a year of weekly audits in a few MB.
_MAX_DELTA_ENTRIES = 200


def _git_head_sha(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _git_changed_files(repo_root: Path) -> list[str]:
    """Files modified vs HEAD — captures the state the audit observed.

    Returns up to ~50 paths to keep the JSON column manageable.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return files[:50]


def classify_pattern(
    deltas: dict[str, tuple[int, int, int]],
    self_change_keys: set[str],
) -> str:
    """Best-effort label for the dominant movement in this audit run.

    ``deltas`` is {metric_key: (baseline, current, delta)} (delta
    POSITIVE on regression, regardless of higher-is-better semantics).
    ``self_change_keys`` is the subset of metrics whose owning scanner
    files changed in the comparison window (per causality.py).
    """
    if not deltas:
        return "no_change"
    real = {k: v for k, v in deltas.items() if k not in self_change_keys}
    if not real:
        return "self_change_only"
    biggest = max(real.items(), key=lambda kv: abs(kv[1][2]))
    key, (_, _, delta) = biggest
    direction = "drop" if delta < 0 else "spike"
    return f"{key}.{direction}"


def record_fix_event(
    db: sqlite3.Connection,
    repo_root: Path,
    *,
    deltas: dict[str, tuple[int, int, int]],
    self_change_keys: set[str] | None = None,
) -> None:
    """Insert one row into fix_events for this audit run.

    Idempotent on (commit_sha, ts) only in the loose sense — the
    timestamp resolution is per-second, so two audits within the same
    second collapse. Rare in practice; not a correctness issue.
    """
    self_change_keys = self_change_keys or set()
    delta_payload = dict(list(deltas.items())[:_MAX_DELTA_ENTRIES])
    pattern = classify_pattern(delta_payload, self_change_keys)
    db.execute(
        """INSERT INTO fix_events
           (commit_sha, ts, files_changed_json, audit_delta_json, detected_pattern)
           VALUES (?, ?, ?, ?, ?)""",
        (
            _git_head_sha(repo_root),
            time.time(),
            json.dumps(_git_changed_files(repo_root)),
            json.dumps(delta_payload),
            pattern,
        ),
    )
