"""Tests for the augmentum-dev codebase model (Phase 0).

Covers:
  * Schema initialises cleanly on a fresh DB.
  * Each ingester is a pure function over a fixture project tree.
  * Tables ingester correctly classifies user_scoped via CREATE and
    via ALTER TABLE ADD COLUMN; preserves the *first* scoping migration.
  * Registrations ingester captures app.include_router() lines.
  * Facts registry queries return well-formed strings.
  * refresh_docs.py round-trip: drift → check fails → apply → check passes.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Resolve and import the skill packages from .claude/skills/augmentum-dev/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL_DIR = _REPO_ROOT / ".claude" / "skills" / "augmentum-dev"
sys.path.insert(0, str(_SKILL_DIR))

from facts import FACTS, render_fact  # noqa: E402
from model import open_model  # noqa: E402
from model.ingesters import endpoints as endpoints_ingest  # noqa: E402
from model.ingesters import files as files_ingest  # noqa: E402
from model.ingesters import handler_signatures as hsig_ingest  # noqa: E402
from model.ingesters import js_calls as js_calls_ingest  # noqa: E402
from model.ingesters import migrations as migrations_ingest  # noqa: E402
from model.ingesters import registrations as registrations_ingest  # noqa: E402
from model.ingesters import settings as settings_ingest  # noqa: E402
from model.ingesters import tables as tables_ingest  # noqa: E402
from model.ingesters import test_files as test_files_ingest  # noqa: E402
from queries import incomplete_settings as incomplete_q  # noqa: E402
from queries import multi_tenant_audit as mt_q  # noqa: E402
from queries import orphaned_endpoints as orphans_q  # noqa: E402
from queries import untested_routes as untested_q  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_project(tmp_path: Path) -> Path:
    """Build a minimal project tree that mirrors the augmentum layout.

    Two migrations exercise both scoping paths:
      001 — CREATE TABLE foo with user_id   (scoping_kind='create')
      002 — ALTER TABLE bar ADD COLUMN user_id   (scoping_kind='alter')
                            ^^^^ scoping happens later than CREATE
      002 also CREATEs unscoped table baz   (user_scoped=0)

    server.py contains 3 include_router calls so registrations_ingest
    has something to find.
    """
    proj = tmp_path / "fixture_project"
    (proj / "augmentum" / "proxy").mkdir(parents=True)
    (proj / "augmentum" / "state" / "migrations").mkdir(parents=True)
    (proj / "ui").mkdir()  # required by find_project_root heuristic

    # Migration 001 — creates foo (scoped at birth) and bar (unscoped).
    (proj / "augmentum" / "state" / "migrations" / "001_init.sql").write_text(
        textwrap.dedent("""
            CREATE TABLE IF NOT EXISTS foo (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                data TEXT
            );
            CREATE TABLE IF NOT EXISTS bar (
                id TEXT PRIMARY KEY,
                payload TEXT
            );
        """).strip(),
        encoding="utf-8",
    )
    # Migration 002 — adds user_id to bar (alter scoping) + creates baz unscoped.
    (proj / "augmentum" / "state" / "migrations" / "002_scope_bar.sql").write_text(
        textwrap.dedent("""
            ALTER TABLE bar ADD COLUMN user_id TEXT REFERENCES users(id);
            CREATE TABLE IF NOT EXISTS baz (
                id TEXT PRIMARY KEY,
                shared_data TEXT
            );
        """).strip(),
        encoding="utf-8",
    )
    # server.py with three registrations.
    (proj / "augmentum" / "proxy" / "server.py").write_text(
        textwrap.dedent("""
            def create_app():
                from fastapi import FastAPI
                app = FastAPI()
                app.include_router(foo_router)
                app.include_router(bar_router)
                # commented: app.include_router(commented_out_router)
                app.include_router(baz_router)
                return app
        """).strip() + "\n",
        encoding="utf-8",
    )
    return proj


@pytest.fixture
def populated_db(fixture_project: Path) -> sqlite3.Connection:
    """Open a model db rooted at the fixture project and run every ingester."""
    db = open_model(fixture_project)
    files_ingest.ingest(fixture_project, db)
    migrations_ingest.ingest(fixture_project, db)
    tables_ingest.ingest(fixture_project, db)
    registrations_ingest.ingest(fixture_project, db)
    return db


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_creates_all_phase0_tables(tmp_path: Path):
    """Opening a model db idempotently creates every Phase 0 table."""
    proj = tmp_path / "blank"
    (proj / "augmentum" / "proxy").mkdir(parents=True)
    (proj / "ui").mkdir()
    db = open_model(proj)
    table_names = {
        row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    for required in (
        "files", "migrations", "tables", "registrations", "endpoints",
        "js_calls", "settings", "css_classes", "js_class_uses",
        "doc_claims", "fix_events", "_model_meta",
    ):
        assert required in table_names, f"missing table: {required}"


def test_schema_apply_is_idempotent(tmp_path: Path):
    """Re-opening the same db must not error or duplicate rows."""
    proj = tmp_path / "blank"
    (proj / "augmentum" / "proxy").mkdir(parents=True)
    (proj / "ui").mkdir()
    open_model(proj).close()
    open_model(proj).close()  # second open must succeed


# ---------------------------------------------------------------------------
# Ingesters
# ---------------------------------------------------------------------------

def test_files_ingester_picks_up_migrations_and_server(populated_db, fixture_project):
    paths = {
        row["path"] for row in populated_db.execute("SELECT path FROM files")
    }
    assert "augmentum/state/migrations/001_init.sql" in paths
    assert "augmentum/state/migrations/002_scope_bar.sql" in paths
    assert "augmentum/proxy/server.py" in paths


def test_migrations_ingester_parses_number_and_slug(populated_db):
    rows = list(populated_db.execute(
        "SELECT number, slug FROM migrations ORDER BY number"
    ))
    assert [(r["number"], r["slug"]) for r in rows] == [
        (1, "init"),
        (2, "scope_bar"),
    ]


def test_tables_ingester_classifies_user_scoping(populated_db):
    rows = {
        row["name"]: row
        for row in populated_db.execute(
            "SELECT name, user_scoped, scoping_migration, scoping_kind, "
            "defining_migration FROM tables"
        )
    }
    # foo: scoped at CREATE in migration 1
    assert rows["foo"]["user_scoped"] == 1
    assert rows["foo"]["scoping_kind"] == "create"
    assert rows["foo"]["scoping_migration"] == 1
    assert rows["foo"]["defining_migration"] == 1
    # bar: defined unscoped in 1, scoped via ALTER in 2
    assert rows["bar"]["user_scoped"] == 1
    assert rows["bar"]["scoping_kind"] == "alter"
    assert rows["bar"]["scoping_migration"] == 2
    assert rows["bar"]["defining_migration"] == 1
    # baz: unscoped, defined in 2
    assert rows["baz"]["user_scoped"] == 0
    assert rows["baz"]["scoping_kind"] is None
    assert rows["baz"]["scoping_migration"] is None
    assert rows["baz"]["defining_migration"] == 2


def test_registrations_ingester_captures_router_calls(populated_db):
    rows = list(populated_db.execute(
        "SELECT line, router_var FROM registrations ORDER BY line"
    ))
    router_vars = [r["router_var"] for r in rows]
    # Three uncommented registrations; commented line is skipped.
    assert router_vars == ["foo_router", "bar_router", "baz_router"]


def test_registrations_ingester_skips_commented_lines(populated_db):
    """The fixture has a commented-out include_router; ensure we don't
    accidentally count it via a too-loose regex."""
    count = populated_db.execute(
        "SELECT COUNT(*) FROM registrations"
    ).fetchone()[0]
    assert count == 3


# ---------------------------------------------------------------------------
# Facts registry
# ---------------------------------------------------------------------------

def test_facts_registry_queries_against_fixture_db(populated_db):
    assert render_fact(populated_db, "tables.user_scoped.count") == "2"
    assert render_fact(populated_db, "tables.user_scoped.list") == "bar, foo"
    assert render_fact(populated_db, "migrations.max") == "2"
    assert render_fact(populated_db, "registrations.count") == "3"
    # First include_router is on the line containing foo_router; the
    # exact line number depends on the fixture text (~line 4 or 5).
    first = int(render_fact(populated_db, "registrations.first_line"))
    assert first > 0


def test_facts_registry_unknown_name_raises(populated_db):
    with pytest.raises(KeyError):
        render_fact(populated_db, "this.fact.does.not.exist")


def test_facts_catalog_completeness():
    """Phase 0+1+2+4 facts. Update this set when new facts ship —
    keeps additions deliberate (a fact without a doc reference is
    dead weight)."""
    expected = {
        # Phase 0
        "tables.user_scoped.count",
        "tables.user_scoped.list",
        "migrations.max",
        "registrations.count",
        "registrations.first_line",
        # Phase 1
        "endpoints.count",
        "js_calls.count",
        "orphaned_endpoints.count",
        # Phase 2
        "settings.count",
        "settings.fully_wired",
        "settings.incomplete",
        # Phase 4
        "test_files.count",
        "test_files.test_count_total",
        "untested_routes.count",
        "multi_tenant_audit.count",
        # Phase 5 — self-describing scale (self-healing counts that
        # replaced hand-typed prose like "5 modes" / "~30 subsystems").
        "modes.count",
        "modes.list",
        "route_modules.count",
        "subsystems.count",
    }
    assert set(FACTS.keys()) == expected


# ---------------------------------------------------------------------------
# Phase 1 — endpoints / js_calls / orphan query
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_with_routes(fixture_project: Path) -> Path:
    """Augment the Phase 0 fixture project with a routes.json + a
    frontend_api_calls.json so the Phase 1 ingesters have something
    to read."""
    import json
    refs = fixture_project / ".claude" / "skills" / "augmentum-dev" / "references"
    refs.mkdir(parents=True)

    # Three endpoints, two of which are referenced from JS, one orphan.
    routes_path = fixture_project / "augmentum" / "proxy" / "foo_routes.py"
    routes_path.parent.mkdir(parents=True, exist_ok=True)
    routes_path.write_text("# placeholder\n", encoding="utf-8")
    (refs / "routes.json").write_text(json.dumps({
        "endpoints": [
            {"method": "GET",  "path": "/api/foo/list",
             "handler": "list_foo",   "file": "augmentum/proxy/foo_routes.py", "line": 10},
            {"method": "POST", "path": "/api/foo/{id}/like",
             "handler": "like_foo",   "file": "augmentum/proxy/foo_routes.py", "line": 20},
            {"method": "GET",  "path": "/api/foo/orphan",
             "handler": "orphan_foo", "file": "augmentum/proxy/foo_routes.py", "line": 30},
            {"method": "OPTIONS", "path": "/api/foo/list",  # excluded by query
             "handler": "preflight",  "file": "augmentum/proxy/foo_routes.py", "line": 40},
            {"method": "GET",  "path": "/static/x",         # excluded by query
             "handler": "static_x",   "file": "augmentum/proxy/foo_routes.py", "line": 50},
        ],
        "count": 5,
    }), encoding="utf-8")

    # Two JS calls — one matches list (verbatim), one matches the
    # parameterised like (different param name → must normalise to match).
    ui_file = fixture_project / "ui" / "scripts" / "foo.js"
    ui_file.parent.mkdir(parents=True, exist_ok=True)
    ui_file.write_text("// placeholder\n", encoding="utf-8")
    (refs / "frontend_api_calls.json").write_text(json.dumps({
        "calls": [
            {"url": "/api/foo/list",         "method": "GET",
             "file": "ui/scripts/foo.js",   "line": 5},
            {"url": "/api/foo/{whatever}/like", "method": "POST",
             "file": "ui/scripts/foo.js",   "line": 12},
        ],
        "count": 2,
    }), encoding="utf-8")
    return fixture_project


@pytest.fixture
def populated_db_with_routes(fixture_with_routes: Path) -> sqlite3.Connection:
    db = open_model(fixture_with_routes)
    files_ingest.ingest(fixture_with_routes, db)
    migrations_ingest.ingest(fixture_with_routes, db)
    tables_ingest.ingest(fixture_with_routes, db)
    registrations_ingest.ingest(fixture_with_routes, db)
    endpoints_ingest.ingest(fixture_with_routes, db)
    js_calls_ingest.ingest(fixture_with_routes, db)
    return db


def test_endpoints_ingester_normalises_param_names(populated_db_with_routes):
    paths = {
        row["path_template"]
        for row in populated_db_with_routes.execute("SELECT path_template FROM endpoints")
    }
    # The {id} template must normalise to {param} so the join with
    # the JS call (which uses {whatever}) works.
    assert "/api/foo/{param}/like" in paths
    assert "/api/foo/list" in paths
    assert "/api/foo/orphan" in paths


def test_js_calls_ingester_normalises_param_names(populated_db_with_routes):
    paths = {
        row["path_template"]
        for row in populated_db_with_routes.execute("SELECT path_template FROM js_calls")
    }
    # Both backend ({id}) and frontend ({whatever}) collapse to {param}.
    assert "/api/foo/{param}/like" in paths


def test_orphan_query_finds_only_unreferenced_endpoint(populated_db_with_routes):
    rows = list(populated_db_with_routes.execute(orphans_q.QUERY))
    paths = [r["path_template"] for r in rows]
    assert paths == ["/api/foo/orphan"], (
        f"expected exactly one orphan; got {paths}"
    )


def test_orphan_query_excludes_options_and_static(populated_db_with_routes):
    """OPTIONS preflights and /static/ paths are intentionally
    excluded; the query must not surface them as orphans even when
    they have no JS reference."""
    rows = list(populated_db_with_routes.execute(orphans_q.QUERY))
    methods = [r["method"] for r in rows]
    paths = [r["path_template"] for r in rows]
    assert "OPTIONS" not in methods
    assert not any(p.startswith("/static/") for p in paths)


def test_orphan_diagnose_clusters_by_subsystem(populated_db_with_routes):
    """The diagnose query must group by subsystem and surface a
    sample paths list."""
    diag = list(populated_db_with_routes.execute(orphans_q.DIAGNOSE))
    assert len(diag) == 1, f"expected one subsystem bucket, got: {[dict(r) for r in diag]}"
    row = diag[0]
    assert row["subsystem"] == "foo"  # foo_routes.py → subsystem "foo"
    assert row["orphan_count"] == 1
    assert "/api/foo/orphan" in (row["sample_paths"] or "")


def test_phase1_facts_render(populated_db_with_routes):
    """The Phase 1 facts must produce reasonable scalar values."""
    assert render_fact(populated_db_with_routes, "endpoints.count") == "5"
    assert render_fact(populated_db_with_routes, "js_calls.count") == "2"
    assert render_fact(populated_db_with_routes, "orphaned_endpoints.count") == "1"


# ---------------------------------------------------------------------------
# Phase 2 — settings ingester + incomplete_settings query
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_with_settings(fixture_with_routes: Path) -> Path:
    """Add a settings_map.json to the fixture project covering the four
    interesting wiring shapes the incomplete_settings query reasons about:

      * fully_wired  — all 4 layers
      * miss_js      — wired in 3, missing settings_js (likely-internal)
      * miss_restore — wired in 3, missing _SETTINGS_RESTORE_MAP (real bug)
      * miss_routes  — wired in 3, missing config_routes
      * miss_config  — wired in 3, missing config.py
      * partial_2    — wired in only 2 layers (NOT flagged — too sparse)
    """
    import json
    refs = fixture_with_routes / ".claude" / "skills" / "augmentum-dev" / "references"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "settings_map.json").write_text(json.dumps({
        "settings": [
            {"backend_key": "fully_wired_thing",
             "frontend_key": "fullyWiredThing", "category": "tool",
             "config_py": True, "config_routes": True,
             "restore_map": True, "settings_js": True, "layers": 4},
            {"backend_key": "miss_js_thing",
             "frontend_key": "", "category": "tool",
             "config_py": True, "config_routes": True,
             "restore_map": True, "settings_js": False, "layers": 3},
            {"backend_key": "miss_restore_thing",
             "frontend_key": "missRestoreThing", "category": "tool",
             "config_py": True, "config_routes": True,
             "restore_map": False, "settings_js": True, "layers": 3},
            {"backend_key": "miss_routes_thing",
             "frontend_key": "missRoutesThing", "category": "tool",
             "config_py": True, "config_routes": False,
             "restore_map": True, "settings_js": True, "layers": 3},
            {"backend_key": "miss_config_thing",
             "frontend_key": "missConfigThing", "category": "tool",
             "config_py": False, "config_routes": True,
             "restore_map": True, "settings_js": True, "layers": 3},
            {"backend_key": "partial_2_thing",
             "frontend_key": "", "category": "string",
             "config_py": True, "config_routes": False,
             "restore_map": True, "settings_js": False, "layers": 2},
        ],
        "totals": {},
    }), encoding="utf-8")
    return fixture_with_routes


@pytest.fixture
def populated_db_with_settings(fixture_with_settings: Path) -> sqlite3.Connection:
    db = open_model(fixture_with_settings)
    files_ingest.ingest(fixture_with_settings, db)
    settings_ingest.ingest(fixture_with_settings, db)
    return db


def test_settings_ingester_loads_layer_flags(populated_db_with_settings):
    rows = {
        row["name_snake"]: row
        for row in populated_db_with_settings.execute(
            "SELECT name_snake, in_config_py, in_config_routes_py, "
            "in_server_restore, in_js_defaults FROM settings"
        )
    }
    fw = rows["fully_wired_thing"]
    assert (fw["in_config_py"], fw["in_config_routes_py"],
            fw["in_server_restore"], fw["in_js_defaults"]) == (1, 1, 1, 1)
    mr = rows["miss_restore_thing"]
    assert (mr["in_config_py"], mr["in_config_routes_py"],
            mr["in_server_restore"], mr["in_js_defaults"]) == (1, 1, 0, 1)


def test_incomplete_settings_query_finds_3of4(populated_db_with_settings):
    """The query must return exactly the 4 settings missing exactly one
    layer — never the fully-wired one or the 2-layer one."""
    rows = list(populated_db_with_settings.execute(incomplete_q.QUERY))
    names = sorted(r["name_snake"] for r in rows)
    assert names == [
        "miss_config_thing",
        "miss_js_thing",
        "miss_restore_thing",
        "miss_routes_thing",
    ]


def test_incomplete_settings_query_attributes_missing_layer(populated_db_with_settings):
    rows = {
        r["name_snake"]: r["missing_layer"]
        for r in populated_db_with_settings.execute(incomplete_q.QUERY)
    }
    assert rows["miss_config_thing"]   == "config.py"
    assert rows["miss_routes_thing"]   == "config_routes.py"
    assert rows["miss_restore_thing"]  == "_SETTINGS_RESTORE_MAP"
    assert rows["miss_js_thing"]       == "settings.js"


def test_incomplete_settings_diagnose_groups_by_missing_layer(populated_db_with_settings):
    diag = list(populated_db_with_settings.execute(incomplete_q.DIAGNOSE))
    by_layer = {r["missing_layer"]: r for r in diag}
    # Exactly one offender per layer in this fixture.
    assert by_layer["_SETTINGS_RESTORE_MAP"]["incomplete_count"] == 1
    assert by_layer["settings.js"]["incomplete_count"] == 1
    assert by_layer["config.py"]["incomplete_count"] == 1
    assert by_layer["config_routes.py"]["incomplete_count"] == 1


def test_phase2_facts_render(populated_db_with_settings):
    assert render_fact(populated_db_with_settings, "settings.count") == "6"
    assert render_fact(populated_db_with_settings, "settings.fully_wired") == "1"
    assert render_fact(populated_db_with_settings, "settings.incomplete") == "4"


# ---------------------------------------------------------------------------
# Phase 4 — tests ingester + untested_routes query
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_with_tests(fixture_with_settings: Path) -> Path:
    """Add tests/ files so the test_files ingester has content + the
    untested_routes query has something to verify against.

    Layout:
      tests/test_foo.py    — targets augmentum/proxy/foo_routes.py via filename + import
      tests/test_misc.py   — no obvious target (covers the ``targets=[]`` path)
    Only test_foo.py is tied to a route file, so foo_routes IS tested
    and any other *_routes.py would surface as untested.
    """
    tests_dir = fixture_with_settings / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_foo.py").write_text(
        "from augmentum.proxy.foo_routes import router\n"
        "def test_one(): pass\n"
        "def test_two(): pass\n"
        "async def test_three(): pass\n",
        encoding="utf-8",
    )
    (tests_dir / "test_misc.py").write_text(
        "def test_something(): pass\n",
        encoding="utf-8",
    )
    # Add a route file that has NO matching test (must surface as untested).
    bare = fixture_with_settings / "augmentum" / "proxy" / "bare_routes.py"
    bare.write_text("# no tests for this one\n", encoding="utf-8")
    return fixture_with_settings


@pytest.fixture
def populated_db_with_tests(fixture_with_tests: Path) -> sqlite3.Connection:
    db = open_model(fixture_with_tests)
    files_ingest.ingest(fixture_with_tests, db)
    settings_ingest.ingest(fixture_with_tests, db)
    test_files_ingest.ingest(fixture_with_tests, db)
    return db


def test_test_files_ingester_counts_def_test_functions(populated_db_with_tests):
    rows = {
        Path(row["path"]).name: row["test_count"]
        for row in populated_db_with_tests.execute(
            "SELECT f.path, t.test_count FROM test_files t "
            "JOIN files f ON f.id = t.file_id"
        )
    }
    assert rows["test_foo.py"] == 3   # def test_one + test_two + async test_three
    assert rows["test_misc.py"] == 1  # def test_something


def test_test_files_ingester_attributes_targets_via_import(populated_db_with_tests):
    """test_foo.py imports from augmentum.proxy.foo_routes — that
    import line must register foo_routes.py as a target."""
    import json
    row = populated_db_with_tests.execute(
        "SELECT t.target_modules FROM test_files t "
        "JOIN files f ON f.id = t.file_id "
        "WHERE f.path = 'tests/test_foo.py'"
    ).fetchone()
    targets = json.loads(row["target_modules"])
    assert "augmentum/proxy/foo_routes.py" in targets


def test_untested_routes_query_finds_routes_without_test_target(populated_db_with_tests):
    """bare_routes.py has no test_*.py importing it → must surface."""
    rows = list(populated_db_with_tests.execute(untested_q.QUERY))
    files = [r["route_file"] for r in rows]
    # foo_routes.py is referenced from test_foo.py — should NOT be untested.
    assert "augmentum/proxy/foo_routes.py" not in files
    # bare_routes.py has no test target — should be flagged.
    assert "augmentum/proxy/bare_routes.py" in files


# ---------------------------------------------------------------------------
# Phase 4 — handler_signatures + multi_tenant_audit
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_with_handler_signatures(fixture_with_tests: Path) -> Path:
    """Build a route file whose handlers exercise each user_id wiring
    pattern (or absence). Then back-fill routes.json so endpoints
    point at these specific handler functions.
    """
    import json

    # Create the route file with three handlers. ``narrative_routes.py``
    # ensures the subsystem is "narrative", which is user-scoped via
    # the existing fixture migrations (we add a narrative_archive
    # table below to satisfy the multi-tenant audit's "scoped_subsystems"
    # CTE).
    routes_path = fixture_with_tests / "augmentum" / "proxy" / "narrative_routes.py"
    routes_path.write_text(
        "def good_handler(request):\n"
        "    user_id = _user_id(request)\n"
        "    store.list_things(user_id=user_id)\n"
        "\n"
        "def passes_only(user_id):\n"
        "    store.update(user_id=user_id)\n"
        "\n"
        "def bad_handler(request):\n"
        "    return {'all_users_data': True}\n",
        encoding="utf-8",
    )
    # Add a user-scoped narrative table so the multi-tenant query's
    # scoped_subsystems CTE includes 'narrative'.
    mig_path = (
        fixture_with_tests / "augmentum" / "state" / "migrations" / "003_narrative.sql"
    )
    mig_path.write_text(
        "CREATE TABLE IF NOT EXISTS narrative_archive (\n"
        "  id TEXT PRIMARY KEY,\n"
        "  user_id TEXT NOT NULL,\n"
        "  body TEXT\n"
        ");\n",
        encoding="utf-8",
    )
    # Route entries pointing at the three handlers.
    refs = fixture_with_tests / ".claude" / "skills" / "augmentum-dev" / "references"
    (refs / "routes.json").write_text(json.dumps({
        "endpoints": [
            {"method": "GET",  "path": "/api/narrative/good",
             "handler": "good_handler",
             "file": "augmentum/proxy/narrative_routes.py", "line": 1},
            {"method": "GET",  "path": "/api/narrative/passes",
             "handler": "passes_only",
             "file": "augmentum/proxy/narrative_routes.py", "line": 4},
            {"method": "GET",  "path": "/api/narrative/bad",
             "handler": "bad_handler",
             "file": "augmentum/proxy/narrative_routes.py", "line": 7},
        ],
        "count": 3,
    }), encoding="utf-8")
    return fixture_with_tests


@pytest.fixture
def populated_db_with_handler_signatures(
    fixture_with_handler_signatures: Path,
) -> sqlite3.Connection:
    db = open_model(fixture_with_handler_signatures)
    files_ingest.ingest(fixture_with_handler_signatures, db)
    migrations_ingest.ingest(fixture_with_handler_signatures, db)
    tables_ingest.ingest(fixture_with_handler_signatures, db)
    endpoints_ingest.ingest(fixture_with_handler_signatures, db)
    hsig_ingest.ingest(fixture_with_handler_signatures, db)
    return db


def test_handler_signatures_detects_user_id_call(
    populated_db_with_handler_signatures,
):
    """``good_handler`` calls _user_id(request) AND passes user_id=
    downstream — both flags must be 1."""
    row = populated_db_with_handler_signatures.execute("""
        SELECT hs.accepts_user_id, hs.passes_user_id
        FROM handler_signatures hs
        JOIN endpoints e ON e.id = hs.endpoint_id
        WHERE e.path_template = '/api/narrative/good'
    """).fetchone()
    assert row["accepts_user_id"] == 1
    assert row["passes_user_id"] == 1


def test_handler_signatures_detects_user_id_param(
    populated_db_with_handler_signatures,
):
    """``passes_only`` declares user_id as a function parameter (not a
    call to _user_id) — accepts_user_id must still be 1."""
    row = populated_db_with_handler_signatures.execute("""
        SELECT hs.accepts_user_id, hs.passes_user_id
        FROM handler_signatures hs
        JOIN endpoints e ON e.id = hs.endpoint_id
        WHERE e.path_template = '/api/narrative/passes'
    """).fetchone()
    assert row["accepts_user_id"] == 1
    assert row["passes_user_id"] == 1


def test_handler_signatures_flags_bad_handler(
    populated_db_with_handler_signatures,
):
    """``bad_handler`` does neither — both flags must be 0 so the
    multi-tenant audit can surface it."""
    row = populated_db_with_handler_signatures.execute("""
        SELECT hs.accepts_user_id, hs.passes_user_id
        FROM handler_signatures hs
        JOIN endpoints e ON e.id = hs.endpoint_id
        WHERE e.path_template = '/api/narrative/bad'
    """).fetchone()
    assert row["accepts_user_id"] == 0
    assert row["passes_user_id"] == 0


def test_multi_tenant_audit_query_finds_only_unwired_in_scoped_subsystem(
    populated_db_with_handler_signatures,
):
    rows = list(populated_db_with_handler_signatures.execute(mt_q.QUERY))
    paths = [r["path_template"] for r in rows]
    # Only bad_handler should be flagged — good and passes wire user_id.
    assert paths == ["/api/narrative/bad"]


def test_multi_tenant_audit_diagnose_groups_by_subsystem(
    populated_db_with_handler_signatures,
):
    diag = list(populated_db_with_handler_signatures.execute(mt_q.DIAGNOSE))
    assert len(diag) == 1
    assert diag[0]["subsystem"] == "narrative"
    assert diag[0]["suspect_count"] == 1


# ---------------------------------------------------------------------------
# refresh_docs.py end-to-end
# ---------------------------------------------------------------------------

def test_refresh_docs_check_passes_on_real_repo():
    """After this PR lands and the wrapped doc claims match reality,
    --check on the actual repo should be clean. Acts as a CI-style
    smoke test against the live tree."""
    script = _SKILL_DIR / "scripts" / "refresh_docs.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"refresh_docs.py --check unexpectedly failed.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_refresh_docs_list_mode_emits_all_facts():
    script = _SKILL_DIR / "scripts" / "refresh_docs.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--list"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    for fact_name in FACTS:
        assert fact_name in proc.stdout, f"--list missed {fact_name}"
