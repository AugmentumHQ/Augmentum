#!/usr/bin/env python3
"""Subsystem health card — render every signal we know about a feature area.

Usage:
    python subsystem_card.py                # list every subsystem with signal
    python subsystem_card.py narrative      # full health card for one
    python subsystem_card.py --top 10       # 10 highest-signal subsystems

Pulls from the codebase model — adding new ingesters automatically
extends the card layout (no changes here needed). Today the card
covers routes, orphans, settings wiring, recent file activity. Once
the ``tests`` and ``handler_signatures`` ingesters land, the card
gains test coverage + multi-tenant audit columns automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_SKILL_DIR = _THIS.parent.parent
sys.path.insert(0, str(_SKILL_DIR))

from model import find_project_root, open_model, refresh  # noqa: E402


def _list_subsystems(db) -> int:
    rows = list(db.execute("""
        SELECT s.subsystem,
               COUNT(DISTINCT e.id)        AS routes,
               COUNT(DISTINCT s2.name_snake) AS settings_count,
               MAX(f.mtime)                 AS last_modified
        FROM files f
        LEFT JOIN endpoints e ON e.handler_file_id = f.id
        LEFT JOIN settings s2
               ON LOWER(s2.name_snake) LIKE LOWER(f.subsystem || '_%')
        JOIN (SELECT DISTINCT subsystem FROM files WHERE subsystem IS NOT NULL) s
             ON s.subsystem = f.subsystem
        WHERE f.subsystem IS NOT NULL
        GROUP BY s.subsystem
        HAVING routes > 0 OR settings_count > 0
        ORDER BY routes DESC, s.subsystem
    """))
    if not rows:
        print("(no subsystems with routes or settings)")
        return 0
    width = max(len(r["subsystem"]) for r in rows)
    print(f"  {'subsystem':<{width}}  {'routes':>6}  {'settings':>8}")
    for r in rows:
        print(f"  {r['subsystem']:<{width}}  {r['routes']:>6}  {r['settings_count']:>8}")
    return 0


def _card(db, name: str) -> int:
    print(f"# Subsystem: {name}")
    print()

    # Routes + orphans
    route_rows = list(db.execute("""
        SELECT
            e.method,
            e.path_template,
            e.handler_name,
            e.handler_line,
            CASE WHEN j.id IS NULL
                  AND e.method != 'OPTIONS'
                  AND e.path_template NOT LIKE '/static/%'
                  AND e.path_template NOT LIKE '/docs%'
                  AND e.path_template != '/openapi.json'
                  AND e.path_template != '/redoc'
                 THEN 1 ELSE 0 END AS is_orphan
        FROM endpoints e
        JOIN files f ON f.id = e.handler_file_id
        LEFT JOIN js_calls j
               ON j.method = e.method
              AND j.path_template = e.path_template
        WHERE f.subsystem = ?
        ORDER BY e.path_template
    """, (name,)))
    orphan_count = sum(1 for r in route_rows if r["is_orphan"])
    print(f"## Routes  ({len(route_rows)} total, {orphan_count} orphan)")
    if not route_rows:
        print("  (no routes registered for this subsystem)")
    else:
        for r in route_rows:
            mark = "  ORPHAN" if r["is_orphan"] else ""
            print(f"  {r['method']:<6} {r['path_template']:<48} -> {r['handler_name']}{mark}")
    print()

    # Settings — match by name_snake prefix (e.g. "narrative_*" for narrative)
    setting_rows = list(db.execute("""
        SELECT
            name_snake,
            in_config_py + in_config_routes_py
            + in_server_restore + in_js_defaults AS layers,
            in_config_py, in_config_routes_py, in_server_restore, in_js_defaults
        FROM settings
        WHERE LOWER(name_snake) LIKE LOWER(? || '_%')
        ORDER BY layers ASC, name_snake
    """, (name,)))
    incomplete = [r for r in setting_rows if 0 < r["layers"] < 4]
    print(f"## Settings  ({len(setting_rows)} total, {len(incomplete)} 3-of-4 wired)")
    if not setting_rows:
        print("  (no settings under this prefix)")
    else:
        for r in setting_rows:
            if r["layers"] == 4:
                continue  # don't print fully-wired (boring)
            missing = []
            if not r["in_config_py"]:
                missing.append("config.py")
            if not r["in_config_routes_py"]:
                missing.append("config_routes.py")
            if not r["in_server_restore"]:
                missing.append("_SETTINGS_RESTORE_MAP")
            if not r["in_js_defaults"]:
                missing.append("settings.js")
            print(f"  {r['name_snake']:<48} {r['layers']}/4  missing: {', '.join(missing)}")
    print()

    # User-scoped tables — match by name prefix (similar heuristic)
    table_rows = list(db.execute("""
        SELECT name, user_scoped, scoping_kind, scoping_migration
        FROM tables
        WHERE LOWER(name) LIKE LOWER(? || '_%') OR LOWER(name) = LOWER(?)
        ORDER BY name
    """, (name, name)))
    if table_rows:
        scoped = sum(1 for r in table_rows if r["user_scoped"])
        print(f"## Tables  ({len(table_rows)} total, {scoped} user-scoped)")
        for r in table_rows:
            scope = (f"scoped via {r['scoping_kind']} in mig {r['scoping_migration']}"
                     if r["user_scoped"] else "server-level")
            print(f"  {r['name']:<48} {scope}")
        print()

    # Recent activity
    activity_rows = list(db.execute("""
        SELECT path, mtime FROM files
        WHERE subsystem = ?
          AND mtime > strftime('%s','now') - 14 * 86400
        ORDER BY mtime DESC LIMIT 8
    """, (name,)))
    if activity_rows:
        print(f"## Recent activity  ({len(activity_rows)} files in last 14d)")
        import datetime
        for r in activity_rows:
            ts = datetime.datetime.fromtimestamp(r["mtime"]).strftime("%Y-%m-%d")
            print(f"  {ts}  {r['path']}")
        print()

    if not (route_rows or setting_rows or table_rows or activity_rows):
        print(f"(no signal for subsystem {name!r} — check spelling with --list)")
        return 1
    return 0


def _top(db, n: int) -> int:
    """The N highest-signal subsystems by orphan + route count."""
    from queries import subsystem_health  # noqa: PLC0415
    rows = list(db.execute(subsystem_health.QUERY))[:n]
    if not rows:
        print("(no subsystem activity)")
        return 0
    width = max(len(r["subsystem"]) for r in rows)
    header = f"  {'subsystem':<{width}}  {'routes':>6} {'orphans':>7} {'recent':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(f"  {r['subsystem']:<{width}}  "
              f"{r['routes']:>6} {r['orphans']:>7} {r['recent_files']:>6}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("name", nargs="?", help="subsystem name (e.g. 'narrative')")
    parser.add_argument("--list", action="store_true",
                        help="list every subsystem with signal")
    parser.add_argument("--top", type=int, default=0,
                        help="show top N highest-signal subsystems")
    args = parser.parse_args(argv)

    project_root = find_project_root(Path.cwd())
    db = open_model(project_root)
    refresh(db, project_root)

    if args.list:
        return _list_subsystems(db)
    if args.top:
        return _top(db, args.top)
    if args.name:
        return _card(db, args.name)
    return _top(db, 10)  # default — top 10


if __name__ == "__main__":
    sys.exit(main())
