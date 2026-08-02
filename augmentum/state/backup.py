"""SQLite database backup utilities."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import aiosqlite
import structlog

log = structlog.get_logger(__name__)

_MAX_BACKUPS = 7

# Minimum age (seconds) between successive startup backups. Bouncing
# the container three times in five minutes shouldn't fire three back-
# to-back VACUUM INTOs — each holds an exclusive write lock for several
# seconds and that's exactly the window where the companion runtime's
# state writes + auth's failed-attempt inserts pile up behind the lock
# and surface as ``database is locked`` errors. One hour matches the
# usual cadence of "container did something interesting" without leaving
# huge gaps in the on-disk history if the container is stable.
_BACKUP_MIN_INTERVAL_S = 60 * 60  # 1 hour


async def backup_database(
    source_conn: aiosqlite.Connection,
    db_path: str | Path,
    backup_dir: str | Path | None = None,
) -> str | None:
    """Create a timestamped backup of the SQLite database.

    Uses ``VACUUM INTO`` so the backup is an online-safe, page-atomic
    snapshot taken through SQLite itself rather than a file-level copy
    that can race with checkpoints and produce torn output. The resulting
    file is fully self-contained — no -wal/-shm companions.

    Returns the backup path on success, None on failure.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        log.warning("backup_skipped_no_db", path=str(db_path))
        return None

    if backup_dir is None:
        backup_dir = db_path.parent / "backups"
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    try:
        backup_dir.chmod(0o700)
    except (OSError, PermissionError) as exc:
        log.debug("backup_dir_chmod_skipped", path=str(backup_dir), error=str(exc))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{db_path.stem}_{stamp}{db_path.suffix}"
    backup_path = backup_dir / backup_name

    # VACUUM INTO won't overwrite an existing file. Vanishingly unlikely
    # at one-second timestamp granularity but defend anyway.
    if backup_path.exists():
        log.warning("backup_skipped_exists", path=str(backup_path))
        return None

    # SQLite has no parameter binding for VACUUM INTO's destination —
    # the path is a string literal. Escape single quotes (defensive;
    # machine-generated paths shouldn't contain them).
    sql_path = str(backup_path).replace("'", "''")

    t0 = time.monotonic()
    try:
        # Python 3.11 sqlite3 auto-BEGINs before any DML-shaped statement
        # and won't undo that for VACUUM. Flush any implicit transaction
        # the caller may have left open or SQLite refuses with
        # "cannot VACUUM from within a transaction".
        await source_conn.commit()
        await source_conn.execute(f"VACUUM INTO '{sql_path}'")

        try:
            backup_path.chmod(0o600)
        except (OSError, PermissionError) as exc:
            log.debug("backup_file_chmod_skipped", path=str(backup_path), error=str(exc))

        elapsed = round(time.monotonic() - t0, 2)
        size_mb = round(backup_path.stat().st_size / (1024 * 1024), 1)
        log.info("backup_created", path=str(backup_path), size_mb=size_mb, elapsed_s=elapsed)
        return str(backup_path)
    except Exception as exc:
        log.warning("backup_failed", error=str(exc))
        # VACUUM INTO can leave a partial file if it failed mid-write.
        if backup_path.exists():
            try:
                backup_path.unlink()
            except OSError:
                pass
        return None


def newest_backup_age_s(backup_dir: str | Path) -> float | None:
    """Return age (seconds) of the most recent .db backup, or None
    when no backups exist yet. Cheap directory walk — no DB I/O."""
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return None
    candidates = [f for f in backup_dir.iterdir() if f.suffix == ".db"]
    if not candidates:
        return None
    newest = max(candidates, key=lambda f: f.stat().st_mtime)
    return max(0.0, time.time() - newest.stat().st_mtime)


def should_skip_startup_backup(
    db_path: str | Path,
    *,
    backup_dir: str | Path | None = None,
    min_interval_s: float = _BACKUP_MIN_INTERVAL_S,
) -> bool:
    """Decide whether the startup VACUUM INTO should be skipped.

    Returns True when a fresh backup already exists within the
    interval window. The intent is to avoid back-to-back lock-storm
    restarts: if you bounced the container 30s ago and the previous
    boot's backup landed, another 5-second exclusive write lock
    right now buys nothing and costs every other writer.
    """
    db_path = Path(db_path)
    if backup_dir is None:
        backup_dir = db_path.parent / "backups"
    age = newest_backup_age_s(backup_dir)
    if age is None:
        return False  # no backup yet — always take the first one
    return age < float(min_interval_s)


def rotate_backups(backup_dir: str | Path, max_backups: int = _MAX_BACKUPS) -> int:
    """Delete old backups, keeping only the most recent max_backups.

    Returns the number of backups deleted.
    """
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return 0

    backups = sorted(
        [f for f in backup_dir.iterdir() if f.suffix == ".db"],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    deleted = 0
    for old_backup in backups[max_backups:]:
        try:
            old_backup.unlink()
            # Companion WAL/SHM files only exist on backups created
            # before the VACUUM INTO switch; clean them up too.
            for suffix in ("-wal", "-shm"):
                companion = backup_dir / (old_backup.name + suffix)
                if companion.exists():
                    companion.unlink()
            deleted += 1
        except Exception as exc:
            log.warning("backup_rotation_failed", file=str(old_backup), error=str(exc))

    if deleted:
        log.info("backups_rotated", deleted=deleted, remaining=len(backups) - deleted)
    return deleted
