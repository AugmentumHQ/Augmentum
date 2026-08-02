-- ATP recipes: named, parameterized macros that replay a sequence of ATP
-- tool calls in ONE call. Any harness can crystallize repeated tool
-- choreography (e.g. ensure_auth -> navigate -> screenshot) into a single
-- verb instead of paying the full multi-round-trip tool tax each session.
-- Per-user; steps are a JSON array of {tool, arguments}. See
-- augmentum/tools/recipe_tool.py + recipe_store.py.

CREATE TABLE IF NOT EXISTS atp_recipes (
    id          TEXT PRIMARY KEY,          -- server-minted recipe id
    user_id     TEXT NOT NULL DEFAULT '',
    name        TEXT NOT NULL DEFAULT '',  -- caller-chosen, unique per user
    description TEXT NOT NULL DEFAULT '',
    steps       TEXT NOT NULL DEFAULT '[]',-- JSON: [{"tool": str, "arguments": {}}]
    harness     TEXT NOT NULL DEFAULT '',  -- who authored it (informational)
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_atp_recipes_user_name
    ON atp_recipes (user_id, name);
