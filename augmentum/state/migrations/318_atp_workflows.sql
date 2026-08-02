-- ATP workflows: soft, self-minted procedural memory (Hermes/AWM-style).
-- Unlike atp_recipes (deterministic executable macros), a workflow is a
-- natural-language playbook the model reads and ADAPTS: a `when_to_use`
-- trigger + numbered steps + description. The model mints one when it
-- decides something worked, refines it over time (version bump), and the
-- matching workflow is retrieved by FTS on `when_to_use` and injected into
-- the harness briefing before tool-calling. Per-user, scope-isolated
-- (harness:project), auto-minted with easy prune. See
-- augmentum/tools/workflow_tool.py + workflow_store.py.

CREATE TABLE IF NOT EXISTS atp_workflows (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT '',
    scope           TEXT NOT NULL DEFAULT '',  -- harness:<h>:<p> or harness:default
    name            TEXT NOT NULL DEFAULT '',
    when_to_use     TEXT NOT NULL DEFAULT '',  -- semantic trigger (FTS-indexed)
    description     TEXT NOT NULL DEFAULT '',
    steps           TEXT NOT NULL DEFAULT '',  -- numbered markdown steps (free text)
    version         INTEGER NOT NULL DEFAULT 1,
    times_used      INTEGER NOT NULL DEFAULT 0,
    times_succeeded INTEGER NOT NULL DEFAULT 0,
    harness         TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_atp_workflows_user_scope_name
    ON atp_workflows (user_id, scope, name);

-- FTS5 (external content) keyed on the trigger + description + name.
CREATE VIRTUAL TABLE IF NOT EXISTS atp_workflows_fts USING fts5(
    when_to_use,
    description,
    name,
    content=atp_workflows,
    content_rowid=rowid
);

CREATE TRIGGER IF NOT EXISTS atp_workflows_ai AFTER INSERT ON atp_workflows BEGIN
    INSERT INTO atp_workflows_fts(rowid, when_to_use, description, name)
    VALUES (new.rowid, new.when_to_use, new.description, new.name);
END;

CREATE TRIGGER IF NOT EXISTS atp_workflows_ad AFTER DELETE ON atp_workflows BEGIN
    INSERT INTO atp_workflows_fts(atp_workflows_fts, rowid, when_to_use, description, name)
    VALUES ('delete', old.rowid, old.when_to_use, old.description, old.name);
END;

CREATE TRIGGER IF NOT EXISTS atp_workflows_au AFTER UPDATE ON atp_workflows BEGIN
    INSERT INTO atp_workflows_fts(atp_workflows_fts, rowid, when_to_use, description, name)
    VALUES ('delete', old.rowid, old.when_to_use, old.description, old.name);
    INSERT INTO atp_workflows_fts(rowid, when_to_use, description, name)
    VALUES (new.rowid, new.when_to_use, new.description, new.name);
END;
