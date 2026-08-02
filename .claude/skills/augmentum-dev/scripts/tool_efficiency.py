#!/usr/bin/env python3
"""Tool-efficiency scanner — mines LIVE tool-usage telemetry for collapse
opportunities and tracks the KPIs each tool-grammar change is judged by.

Unlike the other scanners in this suite (static code analysis), this one
reads RUNTIME data: ``coder_turn_events`` (every tool_call/tool_result the
coder agent made) and the v3 training traces (``tools_used`` per chat turn).
It answers:

  - which tools dominate call volume, and how long are their repeat-chains?
  - which tool pairs always run together (collapse candidates)?
  - which tools fail most, with what failure signatures (wasted iterations)?
  - how much of ``browser_evaluate`` is improvised primitives (waits,
    extraction, clicks) that should be first-class tools?

Every KPI here maps to a model ROUND-TRIP: a collapsed pair or a recovered
failure is a whole context re-send + generation we stop paying for.

Usage:
    python tool_efficiency.py                    # auto-find DB, full report
    python tool_efficiency.py --db /data/augmentum.db --days 14
    python tool_efficiency.py --json             # machine-readable
    python tool_efficiency.py --record           # append KPI history line

The DB usually lives inside the augmentum container (named volume). Run
in-container when the host can't see it:
    docker cp tool_efficiency.py augmentum-augmentum-1:/tmp/
    docker exec -u augmentum augmentum-augmentum-1 python /tmp/tool_efficiency.py

Read-only by design: the DB is opened with ``mode=ro`` so this can never
create root-owned WAL files or write locks against the live app.
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import json
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path

# UTF-8-safe stdout on Windows consoles (mirrors _common.py, but this
# script must also run standalone inside the container where the skill
# package isn't importable).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_DB_CANDIDATES = (
    os.environ.get("AUGMENTUM_DB", ""),
    "/data/augmentum.db",
    "./data/augmentum.db",
    "./augmentum.db",
)

# browser_evaluate improvisation classes — each share is demand evidence
# for a missing first-class primitive.
_EVAL_CLASSES = {
    "improvised_wait": ("setTimeout", "requestAnimationFrame"),
    "dom_extract": ("querySelectorAll", "textContent", "innerText"),
    "direct_click": (".click()",),
    "synthetic_events": ("dispatchEvent", "KeyboardEvent", "PointerEvent"),
}

# Tools whose success=False usually means "the CHECK failed", not "the
# tool broke" — excluded from the failure-rate KPI so red test runs
# don't read as tool debt.
_EXPECTED_FAILURE_TOOLS = {"test_run", "browser_verify", "service_probe"}


def find_db(explicit: str = "") -> str:
    for cand in ((explicit,) if explicit else _DB_CANDIDATES):
        if cand and os.path.exists(cand):
            return cand
    return ""


def _parse_embedded(obj):
    """Event payloads carry tool_call/tool_result as a dict or a str repr
    (legacy rows). ``ast.literal_eval`` handles the repr safely."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            parsed = ast.literal_eval(obj)
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, SyntaxError):
            return None
    return None


def mine_coder(db_path: str, *, days: float = 0) -> dict:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    since = time.time() - days * 86400 if days else 0

    seqs: dict[str, list] = collections.defaultdict(list)
    for run_id, payload in cur.execute(
        "SELECT run_id, payload FROM coder_turn_events "
        "WHERE type='tool_call' AND timestamp >= ? ORDER BY run_id, seq",
        (since,),
    ):
        try:
            tc = _parse_embedded(json.loads(payload).get("tool_call"))
            if not tc:
                continue
            name = tc.get("tool") or tc.get("name") or "?"
            seqs[run_id].append((name, tc.get("input") or {}))
        except (json.JSONDecodeError, TypeError):
            continue

    uni: collections.Counter = collections.Counter()
    big: collections.Counter = collections.Counter()
    chains: dict[str, list[int]] = collections.defaultdict(list)
    eval_counts = {k: 0 for k in _EVAL_CLASSES}
    eval_total = 0
    calls_per_run: list[int] = []

    for names in seqs.values():
        calls_per_run.append(len(names))
        i = 0
        while i < len(names):
            tool, inp = names[i]
            uni[tool] += 1
            if i:
                big[(names[i - 1][0], tool)] += 1
            if tool == "browser_evaluate":
                eval_total += 1
                expr = str(inp.get("expression") or "")
                for cls, needles in _EVAL_CLASSES.items():
                    if any(n in expr for n in needles):
                        eval_counts[cls] += 1
            i += 1
        # chain lengths (consecutive same-tool runs >= 2)
        i = 0
        while i < len(names):
            j = i
            while j < len(names) and names[j][0] == names[i][0]:
                j += 1
            if j - i >= 2:
                chains[names[i][0]].append(j - i)
            i = j

    ok: collections.Counter = collections.Counter()
    bad: collections.Counter = collections.Counter()
    fail_sigs: collections.Counter = collections.Counter()
    for (payload,) in cur.execute(
        "SELECT payload FROM coder_turn_events "
        "WHERE type='tool_result' AND timestamp >= ?",
        (since,),
    ):
        try:
            tr = _parse_embedded(json.loads(payload).get("tool_result"))
            if not tr:
                continue
            name = tr.get("tool") or tr.get("name") or "?"
            if tr.get("success") is False:
                bad[name] += 1
                sig = str(tr.get("error") or tr.get("output_preview") or "")[:70]
                fail_sigs[(name, sig)] += 1
            else:
                ok[name] += 1
        except (json.JSONDecodeError, TypeError):
            continue
    conn.close()

    total_calls = sum(uni.values())
    read_chains = chains.get("file_read", [])
    fail_rates = {
        n: {"fails": bad[n], "total": ok[n] + bad[n],
            "rate": round(bad[n] / (ok[n] + bad[n]), 4)}
        for n in sorted(set(ok) | set(bad), key=lambda n: -bad[n])
        if (ok[n] + bad[n]) >= 30 and n not in _EXPECTED_FAILURE_TOOLS
    }

    kpis = {
        "total_tool_calls": total_calls,
        "runs": len(seqs),
        "median_calls_per_run": (
            statistics.median(calls_per_run) if calls_per_run else 0
        ),
        "file_read_share": round(
            uni.get("file_read", 0) / total_calls, 4) if total_calls else 0,
        "read_chain_count": len(read_chains),
        "read_chain_mean": round(
            statistics.mean(read_chains), 2) if read_chains else 0,
        "read_chain_max": max(read_chains) if read_chains else 0,
        "evaluate_total": eval_total,
        "evaluate_shares": {
            k: round(v / eval_total, 4) if eval_total else 0
            for k, v in eval_counts.items()
        },
        "failure_rates": {
            n: d["rate"] for n, d in fail_rates.items()
            if n in ("file_write", "browser_click", "browser_type",
                     "http_request", "browser_evaluate", "code_edit")
        },
    }
    return {
        "unigrams": uni.most_common(),
        "bigrams": [
            {"a": a, "b": b, "count": c, "self": a == b}
            for (a, b), c in big.most_common()
        ],
        "chains": {
            t: {"n": len(ls), "mean": round(statistics.mean(ls), 2),
                "max": max(ls)}
            for t, ls in sorted(chains.items(), key=lambda kv: -len(kv[1]))
        },
        "failure_rates": fail_rates,
        "failure_signatures": [
            {"tool": t, "sig": s, "count": c}
            for (t, s), c in fail_sigs.most_common(15)
        ],
        "kpis": kpis,
    }


def mine_chat_traces(traces_dir: str, *, days: float = 0) -> dict:
    """Chat-side ``tools_used`` sequences from the v3 training traces.
    Thin today (capture is young) — included so the report grows with it."""
    files = sorted(glob.glob(os.path.join(traces_dir, "*.jsonl")))
    if days and files:
        cutoff = time.time() - days * 86400
        files = [f for f in files if os.path.getmtime(f) >= cutoff]
    uni: collections.Counter = collections.Counter()
    big: collections.Counter = collections.Counter()
    turns = 0
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for line in lines:
            try:
                names = json.loads(line).get("tools_used") or []
            except (json.JSONDecodeError, AttributeError):
                continue
            if not names:
                continue
            turns += 1
            for i, n in enumerate(names):
                uni[n] += 1
                if i:
                    big[(names[i - 1], n)] += 1
    return {
        "files": len(files),
        "tool_turns": turns,
        "unigrams": uni.most_common(),
        "bigrams": [
            {"a": a, "b": b, "count": c} for (a, b), c in big.most_common()
        ],
    }


def render(report: dict, *, top: int) -> str:
    c = report["coder"]
    k = c["kpis"]
    lines = [
        "TOOL EFFICIENCY REPORT",
        f"  db: {report['db']}   window: "
        f"{report['days'] or 'all'} days",
        "",
        f"  {k['total_tool_calls']} tool calls / {k['runs']} runs "
        f"(median {k['median_calls_per_run']} calls/run)",
        "",
        "KPIs",
        f"  file_read share of all calls : {k['file_read_share']:.1%}",
        f"  file_read chain mean/max     : {k['read_chain_mean']} / "
        f"{k['read_chain_max']}  ({k['read_chain_count']} chains)",
        "  browser_evaluate improvised  : "
        + "  ".join(
            f"{cls}={share:.0%}"
            for cls, share in k["evaluate_shares"].items()
        ),
        "  failure rates                : "
        + "  ".join(
            f"{n}={r:.1%}" for n, r in k["failure_rates"].items()
        ),
        "",
        f"TOP TOOLS (of {k['total_tool_calls']})",
    ]
    for name, count in c["unigrams"][:top]:
        lines.append(f"  {count:6d}  {name}")
    lines.append("")
    lines.append("TOP PAIRS (A then B; * = self-chain)")
    for row in c["bigrams"][:top]:
        mark = "*" if row["self"] else " "
        lines.append(
            f"  {row['count']:6d} {mark} {row['a']} -> {row['b']}")
    lines.append("")
    lines.append("REPEAT CHAINS (>=2 consecutive)")
    for tool, st in list(c["chains"].items())[:top]:
        lines.append(
            f"  {tool:22s} n={st['n']:<5d} mean={st['mean']:<6} "
            f"max={st['max']}")
    lines.append("")
    lines.append("FAILURE RATES (>=30 calls; check-style tools excluded)")
    for name, d in list(c["failure_rates"].items())[:top]:
        lines.append(
            f"  {name:22s} {d['rate']:6.1%}  ({d['fails']}/{d['total']})")
    lines.append("")
    lines.append("TOP FAILURE SIGNATURES")
    for row in c["failure_signatures"][:10]:
        lines.append(f"  {row['count']:5d}  {row['tool']}: {row['sig']}")
    ch = report.get("chat") or {}
    if ch.get("tool_turns"):
        lines.append("")
        lines.append(
            f"CHAT-SIDE (training traces: {ch['files']} files, "
            f"{ch['tool_turns']} tool-using turns)")
        for name, count in ch["unigrams"][:top]:
            lines.append(f"  {count:6d}  {name}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="", help="path to augmentum.db")
    ap.add_argument("--traces", default="", help="training_traces dir")
    ap.add_argument("--days", type=float, default=0,
                    help="only events newer than N days (0 = all)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--record", action="store_true",
                    help="append the KPI block to the history file")
    ap.add_argument("--history-file", default="",
                    help="history JSONL (default: <db_dir>/"
                         "tool_efficiency_history.jsonl)")
    args = ap.parse_args()

    db = find_db(args.db)
    if not db:
        print("ERROR: no augmentum.db found (pass --db, or run "
              "in-container against /data/augmentum.db)", file=sys.stderr)
        return 2

    traces = args.traces or str(Path(db).parent / "training_traces")
    report = {
        "generated_at": time.time(),
        "db": db,
        "days": args.days,
        "coder": mine_coder(db, days=args.days),
        "chat": mine_chat_traces(traces, days=args.days)
        if os.path.isdir(traces) else {},
    }

    if args.record:
        hist = args.history_file or str(
            Path(db).parent / "tool_efficiency_history.jsonl")
        entry = {
            "ts": report["generated_at"],
            "days": args.days,
            "kpis": report["coder"]["kpis"],
        }
        try:
            with open(hist, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            print(f"[recorded KPI snapshot -> {hist}]", file=sys.stderr)
        except OSError as exc:
            print(f"[history write failed: {exc}]", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(report, top=args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
