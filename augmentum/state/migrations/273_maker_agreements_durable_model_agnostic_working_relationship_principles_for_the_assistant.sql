-- 273_maker_agreements.sql
-- The Working Agreements registry — how *this maker* wants to be worked with.
--
-- The skill graph (mig 193) accumulates what WORKED for the companion.
-- The lesson registry (mig 270) accumulates what she was CORRECTED on.
-- This is the third axis the accumulation thesis names but never wired
-- for the *assistant* relationship: durable, standing operating
-- principles for how the coding/assistant should work with a particular
-- person — "invest once, don't revisit", "tell me the blast radius
-- before irreversible changes", "finish one thing well over starting
-- three". Not situational corrections (those are lessons); these are
-- always-on agreements that condition every turn.
--
-- The point: this relationship stops living only in one model's private
-- scratchpad and becomes the user's own, server-persisted and
-- MODEL-AGNOSTIC — injected at compose time so ANY local model the user
-- runs (and any future instance of the assistant) inherits how they
-- think. It is the durable substrate for "the assistant accrues a
-- relationship too, not just the companion."
--
-- Ships EMPTY. Each user accrues their own agreements; nothing is baked
-- into the repo (OSS-clean, persona-agnostic). User-scoped like every
-- other relationship table.

CREATE TABLE IF NOT EXISTS maker_agreements (
    id            INTEGER PRIMARY KEY,
    user_id       TEXT REFERENCES users(id),         -- user-scoped; the maker this agreement belongs to
    principle     TEXT NOT NULL DEFAULT '',          -- the standing principle, imperative voice ("Tell me the blast radius first")
    rationale     TEXT NOT NULL DEFAULT '',          -- the why (optional) — preserves the reason so it survives rephrasing
    category      TEXT NOT NULL DEFAULT 'general',   -- scope|reliability|aesthetics|process|communication|general
    source        TEXT NOT NULL DEFAULT 'explicit',  -- explicit (stated) | inferred (observed) | imported
    strength      REAL NOT NULL DEFAULT 1.0,         -- how firmly held; room to grow on reinforcement / decay if contradicted
    times_seen    INTEGER NOT NULL DEFAULT 1,        -- how many times this has been restated/reinforced
    status        TEXT NOT NULL DEFAULT 'active',    -- active|retired
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Hot path is "active agreements for this user, strongest first" at
-- every coder turn compose — covered by this partial index.
CREATE INDEX IF NOT EXISTS idx_maker_agreements_user_active
    ON maker_agreements(user_id, status, strength DESC)
    WHERE status = 'active';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (273, 'maker_agreements: durable model-agnostic working-relationship principles for the assistant');
