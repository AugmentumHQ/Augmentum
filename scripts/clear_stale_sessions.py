"""Clear stale game_stream_sessions whose containers no longer exist."""
from __future__ import annotations
import sqlite3
import sys

DB = "/data/augmentum.db"
LIVE = ("starting", "ready", "connected", "idle")

def main() -> int:
    conn = sqlite3.connect(DB)
    placeholders = ",".join("?" * len(LIVE))
    rows = conn.execute(
        f"SELECT id, user_id, status, container_id "
        f"FROM game_stream_sessions WHERE status IN ({placeholders})",
        LIVE,
    ).fetchall()
    for r in rows:
        print("stale:", r)
    n = conn.execute(
        f"UPDATE game_stream_sessions "
        f"SET status='crashed', exit_reason='stale_cleanup', "
        f"    updated_at=datetime('now') "
        f"WHERE status IN ({placeholders})",
        LIVE,
    ).rowcount
    conn.commit()
    print(f"cleared {n} session(s)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
