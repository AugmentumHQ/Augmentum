"""Named-fact catalog and query helpers.

A FACT is a (description, sql, formatter) tuple. ``sql`` returns one
row; ``formatter`` shapes that row into the string the doc embeds.
The default formatter is ``str(row[0])`` — sufficient for scalar
facts. List facts (e.g. user-scoped table list) override formatter
to produce comma-separated text.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    description: str
    sql: str
    formatter: Callable[[sqlite3.Row], str] = lambda row: str(row[0])


def _list_format(row: sqlite3.Row) -> str:
    """Render a GROUP_CONCAT result as a soft-wrappable list."""
    raw = row[0] or ""
    return raw


# ---------------------------------------------------------------------------
# Phase 0 facts (5 entries)
# ---------------------------------------------------------------------------

FACTS: dict[str, Fact] = {
    "tables.user_scoped.count": Fact(
        description="Number of distinct tables with a user_id column "
                    "(via CREATE TABLE or ALTER TABLE ADD COLUMN).",
        sql="SELECT COUNT(*) FROM tables WHERE user_scoped = 1",
    ),
    "tables.user_scoped.list": Fact(
        description="Alphabetically sorted, comma-separated list of "
                    "user-scoped table names.",
        sql="SELECT GROUP_CONCAT(name, ', ') "
            "FROM (SELECT name FROM tables WHERE user_scoped = 1 ORDER BY name)",
        formatter=_list_format,
    ),
    "migrations.max": Fact(
        description="Highest migration number on disk in "
                    "augmentum/state/migrations/.",
        sql="SELECT COALESCE(MAX(number), 0) FROM migrations",
    ),
    "registrations.count": Fact(
        description="Number of app.include_router() calls in server.py.",
        sql="SELECT COUNT(*) FROM registrations",
    ),
    "registrations.first_line": Fact(
        description="Line number of the first app.include_router() call "
                    "in server.py — the start of the registration block.",
        sql="SELECT MIN(line) FROM registrations",
    ),
    # ---- Phase 1 ----
    "endpoints.count": Fact(
        description="Total registered backend endpoints "
                    "(every @router.<verb> decoration across "
                    "augmentum/proxy/*_routes.py).",
        sql="SELECT COUNT(*) FROM endpoints",
    ),
    "js_calls.count": Fact(
        description="Total fetch / WebSocket / sendBeacon call sites "
                    "across ui/scripts/.",
        sql="SELECT COUNT(*) FROM js_calls",
    ),
    "orphaned_endpoints.count": Fact(
        description="Endpoints with no matching JS call — strict "
                    "(method, normalised_path) join. Excludes OPTIONS "
                    "/ static / docs / openapi paths.",
        sql="""
          SELECT COUNT(*) FROM endpoints e
          LEFT JOIN js_calls j
            ON j.method = e.method AND j.path_template = e.path_template
          WHERE j.id IS NULL
            AND e.method != 'OPTIONS'
            AND e.path_template NOT LIKE '/static/%'
            AND e.path_template NOT LIKE '/docs%'
            AND e.path_template != '/openapi.json'
            AND e.path_template != '/redoc'
        """,
    ),
    # ---- Phase 2 ----
    "settings.count": Fact(
        description="Total settings discovered across the 4 wiring layers "
                    "(config.py, config_routes.py, server.py "
                    "_SETTINGS_RESTORE_MAP, settings.js).",
        sql="SELECT COUNT(*) FROM settings",
    ),
    "settings.fully_wired": Fact(
        description="Settings present in all 4 wiring layers — the "
                    "ship-quality contract for any user-configurable "
                    "feature.",
        sql="""
          SELECT COUNT(*) FROM settings
          WHERE in_config_py = 1
            AND in_config_routes_py = 1
            AND in_server_restore = 1
            AND in_js_defaults = 1
        """,
    ),
    "settings.incomplete": Fact(
        description="Settings wired in 3 of 4 layers — the high-signal "
                    "'missed one layer' subset that flags incomplete "
                    "wiring chores.",
        sql="""
          SELECT COUNT(*) FROM settings
          WHERE (in_config_py + in_config_routes_py
                 + in_server_restore + in_js_defaults) = 3
        """,
    ),
    # ---- Phase 4 ----
    "test_files.count": Fact(
        description="Total test files indexed under tests/.",
        sql="SELECT COUNT(*) FROM test_files",
    ),
    "test_files.test_count_total": Fact(
        description="Sum of `def test_*` functions across the suite "
                    "(parametrize expansion not counted — proxy metric).",
        sql="SELECT COALESCE(SUM(test_count), 0) FROM test_files",
    ),
    "untested_routes.count": Fact(
        description="Route files (augmentum/proxy/*_routes.py) with "
                    "no test file claiming them as a target module.",
        sql="""
          SELECT COUNT(*) FROM files f
          WHERE f.path LIKE 'augmentum/proxy/%_routes.py'
            AND f.path != 'augmentum/proxy/server.py'
            AND NOT EXISTS (
              SELECT 1 FROM test_files tf
              WHERE tf.target_modules LIKE '%' || f.path || '%'
            )
        """,
    ),
    # ---- Phase 5: self-describing scale (derived from files.subsystem /
    # path shape, so they self-heal as dirs are added — no hand-typed
    # "5 modes" / "~30 subsystems" to rot). ----
    "modes.count": Fact(
        description="Number of dispatch-mode handler packages under "
                    "augmentum/modes/ (each subdir = one mode).",
        sql="""
          SELECT COUNT(DISTINCT substr(r, 1, instr(r, '/') - 1))
          FROM (
            SELECT substr(path, length('augmentum/modes/') + 1) AS r
            FROM files WHERE path LIKE 'augmentum/modes/%/%'
          ) WHERE instr(r, '/') > 0
        """,
    ),
    "modes.list": Fact(
        description="Alphabetical, comma-separated list of dispatch-mode "
                    "package names under augmentum/modes/.",
        sql="""
          SELECT GROUP_CONCAT(m, ', ') FROM (
            SELECT DISTINCT substr(r, 1, instr(r, '/') - 1) AS m FROM (
              SELECT substr(path, length('augmentum/modes/') + 1) AS r
              FROM files WHERE path LIKE 'augmentum/modes/%/%'
            ) WHERE instr(r, '/') > 0 ORDER BY m
          )
        """,
        formatter=_list_format,
    ),
    "route_modules.count": Fact(
        description="Number of *_routes.py route modules under "
                    "augmentum/proxy/.",
        sql="SELECT COUNT(*) FROM files WHERE path LIKE 'augmentum/proxy/%_routes.py'",
    ),
    "subsystems.count": Fact(
        description="Number of top-level subsystem packages under augmentum/ "
                    "(distinct second path segment, excluding __pycache__).",
        sql="""
          SELECT COUNT(DISTINCT s) FROM (
            SELECT substr(r, 1, instr(r, '/') - 1) AS s FROM (
              SELECT substr(path, length('augmentum/') + 1) AS r
              FROM files WHERE path LIKE 'augmentum/%/%'
            ) WHERE instr(r, '/') > 0
          ) WHERE s NOT IN ('__pycache__')
        """,
    ),
    "multi_tenant_audit.count": Fact(
        description="Routes in user-scoped subsystems whose handler "
                    "doesn't appear to wire user_id (no _user_id() "
                    "call, no user_id= kwarg). Candidate list — review "
                    "for cross-tenant leak risk.",
        sql="""
          WITH scoped_subsystems AS (
              SELECT DISTINCT
                  substr(t.name, 1, instr(t.name || '_', '_') - 1) AS subsystem
              FROM tables t
              WHERE t.user_scoped = 1 AND t.name LIKE '%_%'
          )
          SELECT COUNT(*) FROM endpoints e
          JOIN files f ON f.id = e.handler_file_id
          JOIN handler_signatures hs ON hs.endpoint_id = e.id
          WHERE hs.accepts_user_id = 0
            AND hs.passes_user_id = 0
            AND f.subsystem IN (SELECT subsystem FROM scoped_subsystems)
            AND e.method != 'OPTIONS'
        """,
    ),
}


# ---------------------------------------------------------------------------
# Render / check helpers
# ---------------------------------------------------------------------------

def render_fact(db: sqlite3.Connection, name: str) -> str:
    """Return the current rendered value of fact ``name``.

    Raises KeyError if the name isn't registered. Raises RuntimeError
    if the underlying query returns no rows (likely an empty model
    that needs ``refresh()``).
    """
    fact = FACTS[name]  # KeyError surfaces unknown facts loudly
    row = db.execute(fact.sql).fetchone()
    if row is None:
        raise RuntimeError(
            f"fact {name!r} returned no rows — model may be empty; "
            f"run model.refresh() first"
        )
    return fact.formatter(row)


def check_fact(db: sqlite3.Connection, name: str, claimed: str) -> bool:
    """Return True if ``claimed`` matches the current rendered value."""
    return render_fact(db, name).strip() == claimed.strip()
