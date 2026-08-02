#!/usr/bin/env python3
"""strain_report.py — read the strain_samples health time series and surface
the worst windows + the concurrent-client combination present at each.

This is the offline analysis half of the general-purpose strain monitor (the
live half is the ``strain_sample`` WARNING log lines, grep-able like
``event_loop_stall``, and ``GET /api/health/strain``). It answers: "when did
the server strain, and what was concurrently happening across browsers/devices
when it did?"

Usage:
  python strain_report.py                       # auto-find DB, last 120 min
  python strain_report.py --db /data/augmentum.db --minutes 360
  python strain_report.py --top 15              # show the 15 worst samples
  python strain_report.py --json                # machine-readable

In Docker:
  docker exec <container> python /path/to/strain_report.py --db /data/augmentum.db
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import sys
from pathlib import Path

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


# Composite strain score — what makes a sample "bad". Weighted so a real event
# loop stall dominates, but DB writer-lock contention and request pileups also
# register. Active-client count is NOT scored (it's the suspected *cause* we
# correlate against, not strain itself).
def _score(row: dict) -> float:
    return (
        (row.get("event_loop_lag_ms") or 0) * 1.0
        + (row.get("db_write_ms") or 0) * 1.0
        + (row.get("inflight_requests") or 0) * 20.0
        + (row.get("slow_requests") or 0) * 50.0
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--top", type=int, default=12, help="how many worst samples to show")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    db = _find_db(args.db)
    if not db:
        print(red("no augmentum.db found — pass --db PATH (in Docker: --db /data/augmentum.db)."))
        return 2

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strain_samples'")
        if not cur.fetchone():
            print(yellow("strain_samples table not found — run a build with the strain monitor (migration 272) first."))
            return 1
        rows = [
            dict(r)
            for r in con.execute(
                "SELECT * FROM strain_samples WHERE timestamp >= datetime('now', ?) "
                "ORDER BY timestamp ASC",
                (f"-{args.minutes} minutes",),
            ).fetchall()
        ]
    finally:
        con.close()

    if not rows:
        print(yellow(f"no strain samples in the last {args.minutes} min "
                     "(monitor disabled, or instance not running)."))
        return 0

    # ---- summary ----------------------------------------------------------
    def mx(k):
        return max((r.get(k) or 0) for r in rows)

    peak_lag = mx("event_loop_lag_ms")
    peak_db = mx("db_write_ms")
    peak_inflight = mx("inflight_requests")
    peak_clients = mx("active_clients")
    peak_rss = mx("proc_rss_mb")

    # ---- correlation: strain vs concurrent client count -------------------
    by_clients: dict[int, list[dict]] = {}
    for r in rows:
        by_clients.setdefault(int(r.get("active_clients") or 0), []).append(r)

    corr = []
    for n in sorted(by_clients):
        grp = by_clients[n]
        avg_lag = sum((r.get("event_loop_lag_ms") or 0) for r in grp) / len(grp)
        avg_db = sum((r.get("db_write_ms") or 0) for r in grp) / len(grp)
        avg_inflight = sum((r.get("inflight_requests") or 0) for r in grp) / len(grp)
        corr.append({
            "active_clients": n, "samples": len(grp),
            "avg_lag_ms": round(avg_lag, 1), "avg_db_ms": round(avg_db, 1),
            "avg_inflight": round(avg_inflight, 2),
        })

    worst = sorted(rows, key=_score, reverse=True)[: args.top]

    if args.json:
        print(json.dumps({
            "minutes": args.minutes, "samples": len(rows),
            "peaks": {
                "event_loop_lag_ms": peak_lag, "db_write_ms": peak_db,
                "inflight_requests": peak_inflight, "active_clients": peak_clients,
                "proc_rss_mb": peak_rss,
            },
            "by_client_count": corr,
            "worst": worst,
        }, indent=2, default=str))
        return 0

    print(bold(f"\n  Strain report — last {args.minutes} min · {len(rows)} samples · {db}"))
    print(dim("  " + "-" * 68))
    lag_c = red if peak_lag >= 1000 else (yellow if peak_lag >= 250 else green)
    db_c = red if peak_db >= 400 else (yellow if peak_db >= 100 else green)
    print(f"  peak event-loop lag : {lag_c(f'{peak_lag:.0f} ms')}")
    print(f"  peak DB write/lock  : {db_c(f'{peak_db:.0f} ms')}")
    print(f"  peak in-flight reqs : {peak_inflight}")
    print(f"  peak active clients : {peak_clients}")
    print(f"  peak process RSS    : {peak_rss} MB")

    print(bold("\n  Strain vs concurrent clients  (does lag rise with more browsers?)"))
    print(dim("  clients  samples   avg lag   avg db   avg inflight"))
    for c in corr:
        line = (f"  {c['active_clients']:>7}  {c['samples']:>7}   "
                f"{c['avg_lag_ms']:>6.0f}ms  {c['avg_db_ms']:>5.0f}ms   {c['avg_inflight']:>6.2f}")
        print(red(line) if c["avg_lag_ms"] >= 250 else line)

    print(bold(f"\n  Worst {len(worst)} moments  (score = lag + db + 20·inflight + 50·slow)"))
    for r in worst:
        ctx = {}
        with contextlib.suppress(Exception):
            ctx = json.loads(r.get("context_json") or "{}")
        sess = ctx.get("sessions", {})
        eng = r.get("engine_model") or "-"
        sec = r.get("engine_secondary") or ""
        eng_str = f"{eng}{' +B:' + sec if sec else ''}"
        tag = red if _score(r) >= 1000 else yellow
        print(tag(
            f"  {r['timestamp']}  lag={r.get('event_loop_lag_ms', 0):.0f}ms "
            f"db={r.get('db_write_ms', 0):.0f}ms inflight={r.get('inflight_requests', 0)} "
            f"slow={r.get('slow_requests', 0)}"
        ))
        print(dim(
            f"      clients={r.get('active_clients', 0)} users={r.get('active_users', 0)} "
            f"coder={sess.get('coder', r.get('sessions_coder', 0))} "
            f"narr={sess.get('narrative', 0)} agentic={sess.get('agentic', 0)} "
            f"presence={r.get('ws_presence', 0)} notify={r.get('ws_notify', 0)} "
            f"engine={eng_str} rss={r.get('proc_rss_mb', 0)}MB"
        ))

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
