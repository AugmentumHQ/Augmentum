"""One-time orphan cleanup: purge file_index rows from deleted media servers.

Background:
  Until the cascade was wired into delete_server, removing a media server
  only dropped the user_media_servers row — it didn't cascade through to
  the file_index rows that referenced it via JSON-embedded `server_id`
  (no FK is possible because server_id lives inside the source_metadata
  blob). Those orphans then 502'd every time the user opened what looked
  like a real chapter, because the streaming proxy couldn't resolve
  credentials for a server that no longer existed.

  The cascade now lives in `augmentum.media.store.purge_server_data` and
  fires from the delete-server route. This script applies the same
  cascade retroactively to libraries that already accumulated orphans
  before the cascade existed.

Usage (inside the augmentum container):
    docker exec augmentum-augmentum-1 python /tmp/purge_dead_media_servers.py [--dry-run]

Behavior:
  - Always backs up the database first (unless --dry-run).
  - Identifies (user_id, server_id) pairs referenced in file_index but
    NOT registered in user_media_servers — excluding known sentinel
    server_ids that aren't backed by user_media_servers rows by design
    (e.g. ``builtin-librivox``, the bundled LibriVox provider).
  - For each, deletes file_index rows + orphaned comic_series rows.
  - Reports per-(user, server) counts and a final total.
  - VACUUMs at the end to reclaim disk space.

Re-runnable safely: if there are no orphans, the script reports nothing
and exits cleanly.

Implementation note: uses the synchronous ``sqlite3`` stdlib module
rather than ``aiosqlite``. The async version blocked indefinitely while
the live augmentum service held WAL connections; sync sqlite3 with WAL
journal mode reads in ~0.4s. Synchronous is fine here — this is a
one-shot operational script, not part of the request hot path.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Match the path used inside the container by the augmentum service.
DB_PATH = Path("/data/augmentum.db")

# Sentinel server_ids that aren't backed by user_media_servers rows by
# design. The bundled LibriVox provider uses a constant string instead
# of registering itself, so file_index rows referencing it would look
# like orphans without this exclusion. Mirrors
# ``augmentum.proxy.media_routes.BUILTIN_LIBRIVOX``.
BUILTIN_SENTINELS = frozenset({"builtin-librivox"})


def find_dead_server_refs(
    conn: sqlite3.Connection,
) -> list[tuple[str, str]]:
    """Return [(user_id, server_id), ...] for orphan references in file_index.

    A reference is "dead" if it points at a server_id that is NOT in
    user_media_servers AND is not on the BUILTIN_SENTINELS allow-list.
    NULL server_ids (rows with no upstream server at all) are skipped.
    """
    placeholders = ",".join("?" * len(BUILTIN_SENTINELS))
    sql = f"""
        SELECT DISTINCT
            fi.user_id,
            json_extract(fi.source_metadata, '$.server_id') AS sid
        FROM file_index fi
        WHERE json_extract(fi.source_metadata, '$.server_id') IS NOT NULL
          AND json_extract(fi.source_metadata, '$.server_id') NOT IN (
              SELECT id FROM user_media_servers
          )
          AND json_extract(fi.source_metadata, '$.server_id') NOT IN ({placeholders})
    """
    cursor = conn.execute(sql, tuple(BUILTIN_SENTINELS))
    return [(row[0], row[1]) for row in cursor.fetchall()]


def count_chapters_for(
    conn: sqlite3.Connection, user_id: str, server_id: str,
) -> int:
    cursor = conn.execute(
        """
        SELECT COUNT(*) FROM file_index
        WHERE user_id = ?
          AND json_extract(source_metadata, '$.server_id') = ?
        """,
        (user_id, server_id),
    )
    return int(cursor.fetchone()[0])


def purge(
    conn: sqlite3.Connection, user_id: str, server_id: str,
) -> dict[str, int]:
    """Mirror of ``augmentum.media.store.purge_server_data`` — kept inline
    so the script has no dependency on the augmentum package, which is
    helpful if it's ever run from a recovery shell against a cold DB.
    """
    chapters_cursor = conn.execute(
        """
        DELETE FROM file_index
        WHERE user_id = ?
          AND json_extract(source_metadata, '$.server_id') = ?
        """,
        (user_id, server_id),
    )
    chapters_removed = chapters_cursor.rowcount or 0

    series_removed = 0
    if chapters_removed > 0:
        series_cursor = conn.execute(
            """
            DELETE FROM comic_series
            WHERE user_id = ?
              AND id NOT IN (
                  SELECT DISTINCT series_id
                  FROM file_index
                  WHERE user_id = ? AND series_id IS NOT NULL
              )
            """,
            (user_id, user_id),
        )
        series_removed = series_cursor.rowcount or 0

    conn.commit()
    return {"chapters": chapters_removed, "series": series_removed}


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    if not DB_PATH.exists():
        print(f"ERROR: database not found at {DB_PATH}", file=sys.stderr)
        return 1

    if not dry_run:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = DB_PATH.with_name(f"{DB_PATH.name}.before_purge_{ts}")
        shutil.copy2(DB_PATH, backup)
        # Copy WAL + SHM if present so the backup is consistent on a hot
        # DB. SQLite reads from WAL when present, so leaving them behind
        # would mean the backup misses uncommitted-to-main pages.
        for sidecar in ("-shm", "-wal"):
            src = DB_PATH.with_name(DB_PATH.name + sidecar)
            if src.exists():
                shutil.copy2(src, backup.with_name(backup.name + sidecar))
        print(f"Backed up {DB_PATH} → {backup}")

    # 30-second busy timeout so a brief WAL writer lock from the live
    # augmentum service doesn't fail us with "database is locked".
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    try:
        orphans = find_dead_server_refs(conn)
        if not orphans:
            print("No orphan server references found — nothing to do.")
            return 0

        print(f"Found {len(orphans)} dead (user_id, server_id) reference(s).")

        if dry_run:
            for user_id, server_id in orphans:
                cnt = count_chapters_for(conn, user_id, server_id)
                print(
                    f"  [dry-run] user={user_id} server={server_id}: "
                    f"{cnt} file_index rows would be removed",
                )
            return 0

        total_chapters = 0
        total_series = 0
        for user_id, server_id in orphans:
            counts = purge(conn, user_id, server_id)
            print(
                f"  user={user_id} server={server_id}: "
                f"chapters={counts['chapters']}, series={counts['series']}",
            )
            total_chapters += counts["chapters"]
            total_series += counts["series"]

        # VACUUM reclaims the freelist pages the DELETEs left behind.
        # Cannot run inside a transaction; the prior commits in `purge`
        # have closed it for us.
        print("Running VACUUM to reclaim disk space…")
        conn.execute("VACUUM")
        print(
            f"Done. Removed {total_chapters} file_index row(s) "
            f"and {total_series} orphan series row(s).",
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
