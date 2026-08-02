#!/usr/bin/env python3
"""Augmentum live DB health & reconciliation — run when something looks off.

This is the *runtime* counterpart to `db_safety.py` (which is the static-code
scanner). It opens a real `augmentum.db`, checks it, and reports drift between
DB rows and the files they point at — the failure mode that left ~270 image
files orphaned after a `.recover` salvage dropped their `image_generations`
rows (the files were never deleted, the table was).

  python db_health.py                       # auto-find the DB, summary
  python db_health.py --db /data/augmentum.db
  python db_health.py --full                # integrity_check (slow) not quick_check
  python db_health.py --json

In Docker: `docker exec <container> python /path/to/db_health.py --db /data/augmentum.db`
(the main DB lives in the container's named volume, not the host checkout).

Read-only. Never writes. Exit 0 = healthy, 1 = integrity failure or broken refs,
2 = couldn't find/open a DB.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Windows consoles/pipes default to a non-UTF-8 codepage; ⟷ / … in our output
# would raise UnicodeEncodeError. Make stdout/stderr lenient.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

_COLOR = os.environ.get("TERM") or os.name != "nt"


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def red(s):    return _c("91", s)
def yellow(s): return _c("93", s)
def green(s):  return _c("92", s)
def cyan(s):   return _c("96", s)
def bold(s):   return _c("1", s)
def dim(s):    return _c("2", s)


_CANDIDATE_DBS = [
    "/data/augmentum.db",
    "./data/augmentum.db",
    os.path.expanduser("~/.augmentum/augmentum.db"),
]


def _find_db(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    env = os.environ.get("AUGMENTUM_DATA_DIR")
    cands = ([str(Path(env) / "augmentum.db")] if env else []) + _CANDIDATE_DBS
    for c in cands:
        if Path(c).is_file():
            return Path(c)
    return None


# File-backed tables: (table, [candidate path columns]). Rows reference a file
# on disk; a recovery that drops the table orphans the files. The path column
# name varies, so we try a few.
_FILE_TABLES = [
    ("image_generations", ["file_path", "path"]),
    ("artifacts", ["path", "file_path"]),
    ("uploads", ["path", "file_path"]),
    ("documents", ["path", "file_path", "source_path"]),
]

# Where relative paths might be rooted, relative to the resolved data dir.
# The artifact store nests under `data/artifacts/{hexprefix}/{file}`; uploads
# and documents have their own conventions. Tried in order; whichever resolves
# the most rows wins. Absolute paths in a column skip this entirely.
_REL_BASE_CANDIDATES = [
    "", "data/artifacts", "artifacts", "data/uploads", "uploads",
    "data/documents", "documents", "data", "blobs", "data/blobs",
]


def _q1(cur, sql, default=None):
    try:
        r = cur.execute(sql).fetchone()
        return r[0] if r else default
    except Exception:
        return default


def _columns(cur, table: str) -> set[str]:
    try:
        return {r[1] for r in cur.execute(f"PRAGMA table_info({table})")}
    except Exception:
        return set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--data-dir", help="base dir for resolving relative file paths in the DB (default: the DB's own directory — that's the convention)")
    ap.add_argument("--full", action="store_true", help="integrity_check instead of quick_check (slow on a big DB)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    db = _find_db(args.db)
    if not db:
        msg = "no augmentum.db found — pass --db PATH (in Docker: --db /data/augmentum.db). This tool is for live diagnosis; that's fine if you're not running an instance."
        if args.json:
            print(json.dumps({"ok": None, "error": msg}))
        else:
            print(yellow(msg))
        return 2

    out: dict = {"db": str(db), "ok": True, "problems": []}
    try:
        cur = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        cur.execute("PRAGMA busy_timeout=5000")
    except Exception as e:
        m = f"cannot open {db}: {e}"
        print(json.dumps({"ok": False, "error": m}) if args.json else red(m))
        return 2

    # --- integrity ---
    chk_sql = "PRAGMA integrity_check" if args.full else "PRAGMA quick_check"
    chk = _q1(cur, chk_sql, "?")
    out["integrity_check"] = chk
    if chk != "ok":
        out["ok"] = False
        out["problems"].append(f"{chk_sql} returned: {chk!r}")
    fk = []
    with contextlib.suppress(Exception):
        fk = cur.execute("PRAGMA foreign_key_check").fetchall()
    out["foreign_key_violations"] = len(fk)
    if fk:
        out["ok"] = False
        out["problems"].append(f"{len(fk)} foreign-key violations (e.g. {fk[:3]})")

    # --- stats ---
    page_size = _q1(cur, "PRAGMA page_size", 0) or 0
    page_count = _q1(cur, "PRAGMA page_count", 0) or 0
    out["size_mb"] = round(page_size * page_count / 1e6, 1)
    out["journal_mode"] = _q1(cur, "PRAGMA journal_mode", "?")
    out["freelist_pages"] = _q1(cur, "PRAGMA freelist_count", 0)
    out["user_version"] = _q1(cur, "PRAGMA user_version", 0)
    wal = db.with_suffix(db.suffix + "-wal")
    out["wal_mb"] = round(wal.stat().st_size / 1e6, 1) if wal.is_file() else 0.0
    out["table_count"] = _q1(cur, "SELECT COUNT(*) FROM sqlite_master WHERE type='table'", 0)

    # --- file⟷row reconciliation ---
    data_dir = Path(args.data_dir).resolve() if args.data_dir else db.parent.resolve()
    out["data_dir"] = str(data_dir)
    recon = []
    for tbl, pcols in _FILE_TABLES:
        cols = _columns(cur, tbl)
        if not cols:
            continue
        pcol = next((c for c in pcols if c in cols), None)
        if not pcol:
            continue
        try:
            paths = [r[0] for r in cur.execute(f"SELECT {pcol} FROM {tbl} WHERE {pcol} IS NOT NULL AND {pcol} != ''")]
        except Exception:
            continue
        n_rows = len(paths)
        if n_rows == 0:
            recon.append({"table": tbl, "rows": 0, "broken_refs": 0, "orphan_files": 0, "base": ""})
            continue
        abs_paths = [Path(p) for p in paths if os.path.isabs(p)]
        rel_paths = [p for p in paths if not os.path.isabs(p)]
        # Auto-discover the base dir for the relative ones: whichever candidate
        # resolves the most rows.
        best_base, best_hits = "", 0
        if rel_paths:
            sample = rel_paths[:50]
            for cand in _REL_BASE_CANDIDATES:
                base = data_dir / cand if cand else data_dir
                hits = sum(1 for p in sample if (base / p).is_file())
                if hits > best_hits:
                    best_base, best_hits = cand, hits
        base_dir = data_dir / best_base if best_base else data_dir
        resolved = list(abs_paths) + [base_dir / p for p in rel_paths]
        # If we never found a base that resolved anything for a table that has
        # relative rows, don't pretend they're all broken — say so.
        base_unknown = bool(rel_paths) and best_hits == 0
        missing = 0 if base_unknown else sum(1 for rp in resolved if not rp.is_file())
        referenced = {str(rp.resolve()) for rp in resolved if rp.exists() or not base_unknown}
        orphans = 0
        if not base_unknown:
            for d in {rp.parent for rp in resolved if str(rp.parent) not in (".", "")}:
                try:
                    for f in os.listdir(d):
                        if f.startswith("."):
                            continue
                        fp = str((d / f).resolve())
                        if os.path.isfile(fp) and fp not in referenced:
                            orphans += 1
                except OSError:
                    pass
        entry = {"table": tbl, "rows": n_rows, "broken_refs": missing, "orphan_files": orphans,
                 "base": str(base_dir) if not base_unknown else "", "base_unknown": base_unknown}
        recon.append(entry)
        if base_unknown:
            out["problems"].append(f"{tbl}: {n_rows} rows reference relative paths but no store base resolved any of them — pass --data-dir or run where the store lives")
        if missing:
            out["ok"] = False
            out["problems"].append(f"{tbl}: {missing}/{n_rows} rows point at a missing file")
        if orphans:
            out["problems"].append(f"{tbl}: {orphans} file(s) on disk with no DB row (orphaned — likely a past recovery dropped the rows; see the image-recovery write-up)")
    out["reconciliation"] = recon

    # --- workspace ⟷ docker drift ---
    # Companion to file⟷row recon: project_checkouts rows whose container
    # disappeared from Docker entirely (manual rm, prune, image rebuild)
    # but whose DB row still claims running/paused. always_on=1 rows are
    # additionally surfaced when they haven't been touched in 14+ days
    # (likely leftover flag from a probe run).
    ws_drift = {"checked": False, "phantom_running": [], "stale_always_on": []}
    if "project_checkouts" in {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}:
        ws_drift["checked"] = True
        try:
            import subprocess
            proc = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5,
            )
            docker_names = set(proc.stdout.split()) if proc.returncode == 0 else None
        except Exception:
            docker_names = None
        if docker_names is None:
            ws_drift["docker_unavailable"] = True
        else:
            try:
                phantom = cur.execute(
                    "SELECT id, name, status FROM project_checkouts "
                    "WHERE status IN ('running','paused') "
                    "  AND container_id IS NOT NULL AND container_id != ''"
                ).fetchall()
            except Exception:
                phantom = []
            for ws_id, name, status in phantom:
                if f"augmentum-ws-{ws_id[:8]}" not in docker_names:
                    ws_drift["phantom_running"].append({
                        "id": ws_id[:8], "name": name, "db_status": status,
                    })
        try:
            stale_cutoff = int(time.time()) - 14 * 86400
            stale = cur.execute(
                "SELECT id, name, last_active FROM project_checkouts "
                "WHERE always_on = 1 "
                "  AND (last_active IS NULL OR last_active < ?)",
                (stale_cutoff,),
            ).fetchall()
        except Exception:
            stale = []
        for ws_id, name, last_active in stale:
            idle_days = int((time.time() - last_active) / 86400) if last_active else None
            ws_drift["stale_always_on"].append({
                "id": ws_id[:8], "name": name, "idle_days": idle_days,
            })
        if ws_drift["phantom_running"]:
            out["problems"].append(
                f"project_checkouts: {len(ws_drift['phantom_running'])} row(s) marked running/paused "
                "but no Docker container — list_workspaces auto-fixes on next call",
            )
        if ws_drift["stale_always_on"]:
            out["problems"].append(
                f"project_checkouts: {len(ws_drift['stale_always_on'])} always_on=1 row(s) idle 14+ days "
                "(exempt from reaper — review whether the flag is still wanted)",
            )
    out["workspace_drift"] = ws_drift

    # --- snapshots lying around ---
    snaps = []
    for f in db.parent.glob("*"):
        n = f.name
        if f == db:
            continue
        if any(k in n for k in (".corrupt", ".backup", ".recover", "_corrupt", "before_")):
            with contextlib.suppress(OSError):
                snaps.append({"name": n, "mb": round(f.stat().st_size / 1e6, 1)})
    out["snapshots"] = sorted(snaps, key=lambda s: s["name"])

    cur.close()

    if args.json:
        print(json.dumps(out, indent=2))
        return 0 if out["ok"] else 1

    # --- human report ---
    print(bold(f"DB health — {db}"))
    state = green("OK") if out["ok"] else red("PROBLEMS")
    print(f"  status:        {state}")
    print(f"  {chk_sql}: {green(chk) if chk == 'ok' else red(repr(chk))}")
    print(f"  fk violations: {out['foreign_key_violations']}")
    print(f"  size:          {out['size_mb']} MB  (+{out['wal_mb']} MB WAL)   journal_mode={out['journal_mode']}   tables={out['table_count']}   freelist={out['freelist_pages']}")
    print()
    print(bold("  File ⟷ row reconciliation"))
    if not recon:
        print(dim("    (no file-backed tables found)"))
    for e in recon:
        line = f"    {e['table']:<22} rows={e['rows']:<6} broken_refs={e['broken_refs']:<5} orphan_files={e['orphan_files']:<5}"
        if e.get("base_unknown"):
            line += dim("  (store base not located — pass --data-dir)")
        elif e.get("base") and e["rows"]:
            line += dim(f"  @ {e['base']}")
        print(red(line) if (e["broken_refs"] or e["orphan_files"]) else green(line))
    print()
    if ws_drift["checked"]:
        print(bold("  Workspace ⟷ Docker drift"))
        if ws_drift.get("docker_unavailable"):
            print(dim("    docker CLI unavailable — skipped"))
        else:
            phantom = ws_drift["phantom_running"]
            stale = ws_drift["stale_always_on"]
            if not phantom and not stale:
                print(green("    clean — every running/paused row has a Docker container, no stale always_on flags"))
            if phantom:
                print(red(f"    {len(phantom)} phantom-running row(s) — DB says running/paused but no container:"))
                for p in phantom[:8]:
                    print(red(f"      - {p['id']} {p['name']!r} (db_status={p['db_status']})"))
                if len(phantom) > 8:
                    print(red(f"      … and {len(phantom) - 8} more"))
                print(dim("    list_workspaces() will auto-fix these on next call"))
            if stale:
                print(yellow(f"    {len(stale)} always_on=1 row(s) idle 14+ days:"))
                for s in stale[:8]:
                    days = f"{s['idle_days']}d" if s['idle_days'] is not None else "never"
                    print(yellow(f"      - {s['id']} {s['name']!r} (last_active {days})"))
                if len(stale) > 8:
                    print(yellow(f"      … and {len(stale) - 8} more"))
                print(dim("    reaper-exempt; review the flag if no longer needed"))
        print()

    if out["snapshots"]:
        print(bold(f"  DB snapshots in {db.parent} ({len(out['snapshots'])})"))
        for s in out["snapshots"][-12:]:
            print(dim(f"    {s['name']}  ({s['mb']} MB)"))
        if len(out["snapshots"]) > 12:
            print(dim(f"    … and {len(out['snapshots']) - 12} more — a pile of these means the corruption/recovery cycle is recurring"))
        print()
    if out["problems"]:
        print(yellow(bold("  Problems:")))
        for p in out["problems"]:
            print(yellow(f"    - {p}"))
        print()
        print(dim("  Orphaned files are recoverable: the bytes are there. Restore the rows from"))
        print(dim("  the newest *.backup-* DB (filter to ids whose file still exists), then synthesize"))
        print(dim("  minimal rows for any files newer than that backup. Always VACUUM INTO a fresh"))
        print(dim("  snapshot of the live DB first."))
    else:
        print(green("  No drift — every file-backed row resolves and no orphans."))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
