"""Repair a corrupted augmentum.db.

Designed to run inside a transient container that mounts the augmentum
data volume at /data. The augmentum service container MUST be stopped
before this runs — it has the DB open exclusively while running, and
also any concurrent write would corrupt the rebuild.

Strategy:
    1. Snapshot the current DB to a timestamped backup so we can revert.
    2. Use ``.recover`` (sqlite3's corruption-aware extractor) to dump
       valid rows into a SQL script. ``.dump`` would abort on the bad
       pages; ``.recover`` walks every B-tree and salvages what's
       readable, dropping unrecoverable rows silently.
    3. Replay the recovered SQL into a fresh DB file.
    4. Run ``PRAGMA integrity_check`` on the new file. Abort and keep
       the original if anything other than ``ok`` comes back.
    5. Atomically swap: rename old DB → .corrupt-<ts>, new DB → canonical.

The script never deletes the original DB. Worst case you end up with
``augmentum.db`` (original, possibly still corrupt), ``augmentum.db.
backup-<ts>`` (snapshot before the repair), and ``augmentum.db.new``
(the candidate replacement, possibly bad). You can manually inspect.

Exit codes:
    0  repair succeeded, files swapped
    1  repair failed at any step; original is untouched
    2  ``PRAGMA integrity_check`` reported the new DB is also dirty
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


DB_PATH = Path(os.environ.get("AUGMENTUM_DB_PATH", "/data/augmentum.db"))


def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _log(level: str, msg: str, **kw: object) -> None:
    extras = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[{level}] {msg} {extras}".rstrip(), flush=True)


def _bytes_to_mb(n: int) -> str:
    return f"{n / 1_000_000:.1f}MB"


def _check_preconditions() -> None:
    if not DB_PATH.exists():
        _log("error", "db_not_found", path=str(DB_PATH))
        sys.exit(1)

    # WAL files MUST be absent — if augmentum is still running, they'll
    # exist and the rebuild would race against live writes.
    wal = DB_PATH.with_suffix(DB_PATH.suffix + "-wal")
    shm = DB_PATH.with_suffix(DB_PATH.suffix + "-shm")
    if wal.exists() or shm.exists():
        _log(
            "error", "wal_files_present",
            wal_exists=wal.exists(), shm_exists=shm.exists(),
            note="stop the augmentum container before running this script",
        )
        sys.exit(1)


def _backup() -> Path:
    backup = DB_PATH.with_name(f"{DB_PATH.name}.backup-{_ts()}")
    shutil.copy2(DB_PATH, backup)
    _log("info", "backup_created", path=str(backup), size=_bytes_to_mb(backup.stat().st_size))
    return backup


def _initial_integrity() -> list[str]:
    """Run integrity_check against the current DB. Caller decides whether
    to proceed; this is informational only."""
    try:
        c = sqlite3.connect(DB_PATH)
        rows = c.execute("PRAGMA integrity_check").fetchall()
        c.close()
        return [r[0] for r in rows]
    except sqlite3.DatabaseError as e:
        return [f"<integrity_check itself raised: {e}>"]


def _recover_to_sql(out_sql: Path) -> None:
    """Use sqlite3's ``.recover`` dot-command via the CLI. Python's
    sqlite3 module doesn't expose recover directly, but the underlying
    library does — we shell to the CLI which is the standard tool.
    """
    if not shutil.which("sqlite3"):
        # Fall back to ``.dump`` from Python. Less robust on serious
        # corruption (will abort on the bad B-trees) but works on minor
        # corruption like out-of-order rowids.
        _log("warning", "sqlite3_cli_missing", fallback="iterdump",
             note="install sqlite3 CLI for .recover (corruption-aware extraction)")
        c = sqlite3.connect(DB_PATH)
        with out_sql.open("w") as f:
            for line in c.iterdump():
                f.write(line + "\n")
        c.close()
        return

    cmd = ["sqlite3", str(DB_PATH), ".recover"]
    _log("info", "running_recover", cmd=" ".join(cmd))
    with out_sql.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        _log("warning", "recover_nonzero_exit", code=proc.returncode,
             stderr=proc.stderr[:500])
    sql_size = out_sql.stat().st_size
    _log("info", "recover_complete", sql_size=_bytes_to_mb(sql_size))


def _rebuild_from_sql(in_sql: Path, out_db: Path) -> None:
    if out_db.exists():
        out_db.unlink()
    c = sqlite3.connect(out_db)
    # Match the running app's pragmas so the new file is laid out the
    # same way the app expects.
    for pragma in (
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA foreign_keys=ON",
    ):
        c.execute(pragma)
    with in_sql.open("r") as f:
        c.executescript(f.read())
    c.commit()
    c.close()
    _log("info", "rebuild_complete", path=str(out_db),
         size=_bytes_to_mb(out_db.stat().st_size))


def _verify(db: Path) -> bool:
    c = sqlite3.connect(db)
    try:
        rows = c.execute("PRAGMA integrity_check").fetchall()
        findings = [r[0] for r in rows]
    finally:
        c.close()
    if findings == ["ok"]:
        _log("info", "verify_ok")
        return True
    _log("error", "verify_failed", findings_count=len(findings),
         sample=findings[:5])
    return False


def _swap(new_db: Path) -> Path:
    # Capture the original DB's ownership BEFORE renaming it. The repair
    # script typically runs as root inside the transient container so
    # apt-get + .recover have the privileges they need; the augmentum
    # service container runs as a non-root user and would crash with
    # "attempt to write a readonly database" if the new file landed
    # owned by root. Mirror the original ownership onto the new file
    # so the swap is invisible to the service.
    orig_stat = DB_PATH.stat()
    os.chown(new_db, orig_stat.st_uid, orig_stat.st_gid)

    corrupt_path = DB_PATH.with_name(f"{DB_PATH.name}.corrupt-{_ts()}")
    DB_PATH.rename(corrupt_path)
    new_db.rename(DB_PATH)
    _log("info", "swap_complete",
         active=str(DB_PATH), retired=str(corrupt_path),
         uid=orig_stat.st_uid, gid=orig_stat.st_gid)
    return corrupt_path


def main() -> int:
    _log("info", "repair_started", db=str(DB_PATH))
    _check_preconditions()

    pre_findings = _initial_integrity()
    _log(
        "info", "initial_integrity_check",
        findings_count=len(pre_findings),
        sample=pre_findings[:3] if pre_findings != ["ok"] else "ok",
    )

    backup = _backup()
    sql_dump = DB_PATH.with_suffix(DB_PATH.suffix + ".recover.sql")
    new_db = DB_PATH.with_suffix(DB_PATH.suffix + ".new")

    try:
        _recover_to_sql(sql_dump)
        _rebuild_from_sql(sql_dump, new_db)
    except Exception as e:
        _log("error", "rebuild_aborted", error=str(e))
        for p in (sql_dump, new_db):
            if p.exists():
                p.unlink()
        return 1

    if not _verify(new_db):
        _log("error", "new_db_also_dirty",
             note="original DB untouched; investigate before re-running")
        return 2

    _swap(new_db)

    # Clear the auto-recovery gate stamp. The in-process recovery in
    # augmentum/state/backends/sqlite.py writes this on every recovery
    # attempt and refuses to retry while it exists; once we've done a
    # full offline rebuild, future spontaneous corruption (if any)
    # should be allowed to auto-recover once before falling back to
    # manual repair again.
    stamp = DB_PATH.parent / ".augmentum_recovery_stamp"
    if stamp.exists():
        try:
            stamp.unlink()
            _log("info", "recovery_gate_cleared", stamp=str(stamp))
        except OSError as exc:
            _log("warning", "recovery_gate_clear_failed",
                 path=str(stamp), error=str(exc))

    _log("info", "repair_done",
         backup=str(backup), sql_dump=str(sql_dump),
         note="restart the augmentum container now")
    return 0


if __name__ == "__main__":
    sys.exit(main())
