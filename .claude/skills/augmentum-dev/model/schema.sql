-- augmentum-dev codebase model — Phase 0 schema.
--
-- This file defines the FULL Phase 0-6 surface (10 tables) so future
-- ingesters can populate without schema migrations. Phase 0 ingesters
-- only write to: files, migrations, tables, registrations.
--
-- Schema is idempotent (CREATE IF NOT EXISTS everywhere). Indexes live
-- alongside their tables. Re-applying schema.sql is safe.
--
-- Convention: every fact-row carries source_file_id (or file_id) so
-- re-ingesting a single file means DELETE WHERE file_id=? + INSERT.

CREATE TABLE IF NOT EXISTS files (
    id              INTEGER PRIMARY KEY,
    path            TEXT NOT NULL UNIQUE,
    mtime           REAL NOT NULL,
    sha             TEXT NOT NULL,
    lang            TEXT,
    subsystem       TEXT,
    last_ingest_ts  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_subsystem ON files(subsystem);
CREATE INDEX IF NOT EXISTS idx_files_lang ON files(lang);

CREATE TABLE IF NOT EXISTS migrations (
    number          INTEGER PRIMARY KEY,
    slug            TEXT NOT NULL,
    file_id         INTEGER NOT NULL REFERENCES files(id),
    raw_sql         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_migrations_file ON migrations(file_id);

CREATE TABLE IF NOT EXISTS tables (
    name                TEXT PRIMARY KEY,
    defining_migration  INTEGER REFERENCES migrations(number),
    user_scoped         INTEGER NOT NULL DEFAULT 0,
    scoping_migration   INTEGER REFERENCES migrations(number),
    scoping_kind        TEXT,         -- 'create' (column in CREATE TABLE) or 'alter' (added later)
    columns_json        TEXT
);
CREATE INDEX IF NOT EXISTS idx_tables_user_scoped ON tables(user_scoped);

CREATE TABLE IF NOT EXISTS endpoints (
    id                  INTEGER PRIMARY KEY,
    method              TEXT NOT NULL,
    path_template       TEXT NOT NULL,
    handler_file_id     INTEGER NOT NULL REFERENCES files(id),
    handler_line        INTEGER NOT NULL,
    handler_name        TEXT,
    UNIQUE(method, path_template)
);
CREATE INDEX IF NOT EXISTS idx_endpoints_handler ON endpoints(handler_file_id);
CREATE INDEX IF NOT EXISTS idx_endpoints_path ON endpoints(path_template);

CREATE TABLE IF NOT EXISTS registrations (
    id              INTEGER PRIMARY KEY,
    file_id         INTEGER NOT NULL REFERENCES files(id),
    line            INTEGER NOT NULL,
    router_var      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_registrations_file ON registrations(file_id);

CREATE TABLE IF NOT EXISTS js_calls (
    id                  INTEGER PRIMARY KEY,
    file_id             INTEGER NOT NULL REFERENCES files(id),
    line                INTEGER NOT NULL,
    method              TEXT,
    path_template       TEXT,
    has_error_handler   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_js_calls_path ON js_calls(path_template);
CREATE INDEX IF NOT EXISTS idx_js_calls_file ON js_calls(file_id);

CREATE TABLE IF NOT EXISTS settings (
    name_snake          TEXT PRIMARY KEY,
    name_camel          TEXT,
    type                TEXT,
    default_value       TEXT,
    in_config_py        INTEGER NOT NULL DEFAULT 0,
    in_config_routes_py INTEGER NOT NULL DEFAULT 0,
    in_server_restore   INTEGER NOT NULL DEFAULT 0,
    in_js_defaults      INTEGER NOT NULL DEFAULT 0,
    in_js_load          INTEGER NOT NULL DEFAULT 0,
    in_js_sync          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS css_classes (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id),
    line        INTEGER NOT NULL,
    class_name  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_css_classes_name ON css_classes(class_name);

CREATE TABLE IF NOT EXISTS js_class_uses (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id),
    line        INTEGER NOT NULL,
    class_name  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_js_class_uses_name ON js_class_uses(class_name);

CREATE TABLE IF NOT EXISTS doc_claims (
    id                  INTEGER PRIMARY KEY,
    file_id             INTEGER NOT NULL REFERENCES files(id),
    line                INTEGER NOT NULL,
    kind                TEXT NOT NULL,
    claimed_value       TEXT NOT NULL,
    fact_name           TEXT,         -- if this claim is bound to a FACTS entry
    last_checked_ts     REAL
);
CREATE INDEX IF NOT EXISTS idx_doc_claims_fact ON doc_claims(fact_name);

CREATE TABLE IF NOT EXISTS fix_events (
    id                  INTEGER PRIMARY KEY,
    commit_sha          TEXT NOT NULL,
    ts                  REAL NOT NULL,
    files_changed_json  TEXT,
    audit_delta_json    TEXT,
    detected_pattern    TEXT
);
CREATE INDEX IF NOT EXISTS idx_fix_events_pattern ON fix_events(detected_pattern);
CREATE INDEX IF NOT EXISTS idx_fix_events_ts ON fix_events(ts);

-- Tests catalog — one row per test file, with a best-effort link to
-- the production module(s) it exercises. Phase 4: drives untested-
-- routes / untested-modules queries and feeds the subsystem health
-- card with coverage data.
CREATE TABLE IF NOT EXISTS test_files (
    id              INTEGER PRIMARY KEY,
    file_id         INTEGER NOT NULL REFERENCES files(id),
    test_count      INTEGER NOT NULL DEFAULT 0,   -- number of `def test_*` functions
    target_modules  TEXT NOT NULL DEFAULT '[]'    -- JSON array of "augmentum/foo/bar.py" paths
);
CREATE INDEX IF NOT EXISTS idx_test_files_file ON test_files(file_id);

-- Per-handler signatures — Phase 4: AST-derived facts about each
-- route handler (does it accept user_id, does it call a user-scoped
-- store function, etc.). One row per endpoint; joined back via
-- (handler_file_id, handler_line) so query stays robust to handler
-- renames within the same file.
CREATE TABLE IF NOT EXISTS handler_signatures (
    id                   INTEGER PRIMARY KEY,
    -- ON DELETE CASCADE: endpoints is rebuilt-from-scratch by the
    -- endpoints ingester, so child rows must vanish with their
    -- parent or the next refresh hits a FK violation.
    endpoint_id          INTEGER NOT NULL REFERENCES endpoints(id) ON DELETE CASCADE,
    accepts_user_id      INTEGER NOT NULL DEFAULT 0,  -- function calls _user_id(request) or has user_id arg
    passes_user_id       INTEGER NOT NULL DEFAULT 0,  -- function passes user_id= to a downstream call
    raw_signature        TEXT                         -- text of the def line for debugging
);
CREATE INDEX IF NOT EXISTS idx_handler_sigs_endpoint ON handler_signatures(endpoint_id);

-- Schema version sentinel — bump when schema changes.
CREATE TABLE IF NOT EXISTS _model_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
INSERT OR REPLACE INTO _model_meta (key, value) VALUES ('schema_version', '2');
