-- 228_bug_finder_codebase_knowledge.sql
-- Per-workspace structural map of the codebase being audited.
--
-- "Before it can find bugs, it has to know the code base." The
-- comprehender subagent walks each workspace once (1-2 hr of LLM
-- work on a large repo) and emits a structured map: subsystems,
-- pillars (load-bearing invariants), risk surfaces (untrusted input
-- boundaries), routes/entry points. This table persists that map so
-- every subsequent run reuses it without paying the comprehension
-- cost again.
--
-- Re-comprehension is triggered when:
--   - The map is missing (first run on workspace)
--   - last_commit_sha drifts beyond a threshold (significant code
--     change since last comprehension)
--   - The user explicitly requests a re-comprehension
--
-- All non-id fields are nullable so the comprehender can populate
-- incrementally as it gathers data. ``last_updated=0`` marks an
-- in-flight comprehension; consumers read the brief when
-- ``last_updated > 0``.
--
-- User-scoped per the auth pattern. One row per (user_id, workspace_id).

CREATE TABLE IF NOT EXISTS bug_finder_codebase_knowledge (
    workspace_id      TEXT NOT NULL,
    user_id           TEXT NOT NULL DEFAULT '',

    -- Synthesized markdown brief — what the comprehender produces as
    -- prompt-injection content for downstream subagents (planner,
    -- detector, verifier). ~3-10 KB target. Renderable directly into
    -- a system prompt prefix without further processing.
    brief             TEXT NOT NULL DEFAULT '',

    -- Structured artifacts. JSON-serialized lists/dicts so callers
    -- can pull specific dimensions without re-parsing the brief.
    subsystems_json   TEXT NOT NULL DEFAULT '[]',
        -- [{"name": str, "purpose": str, "paths": [...], "size": int,
        --   "pillars": [...]}]
    pillars_json      TEXT NOT NULL DEFAULT '[]',
        -- [{"name": str, "statement": str, "evidence": [file:line, ...]}]
        -- Load-bearing invariants — "every user-scoped table accepts user_id",
        -- "all template-literal user content uses escapeHtml", etc.
    risk_surfaces_json TEXT NOT NULL DEFAULT '[]',
        -- [{"name": str, "entry_points": [...], "trust_boundary": str,
        --   "downstream_sinks": [...]}]
    entry_points_json TEXT NOT NULL DEFAULT '[]',
        -- [{"kind": "http"|"job"|"cli"|..., "path": str,
        --   "handler": "file:function"}]

    -- Refresh metadata
    last_updated      INTEGER NOT NULL DEFAULT 0,
        -- unix-seconds timestamp; 0 = never comprehended (or in-flight)
    last_commit_sha   TEXT NOT NULL DEFAULT '',
        -- HEAD sha at comprehension time; we re-comprehend when the
        -- workspace's current HEAD is significantly ahead.
    refresh_count     INTEGER NOT NULL DEFAULT 0,
        -- How many times we've re-comprehended this workspace.

    -- Cost ledger snapshot from the comprehension run that produced
    -- the current brief — surfaces for the user "this map cost N
    -- tokens" so they can decide whether to refresh.
    tokens_in         INTEGER NOT NULL DEFAULT 0,
    tokens_out        INTEGER NOT NULL DEFAULT 0,
    wallclock_seconds REAL    NOT NULL DEFAULT 0,

    PRIMARY KEY (user_id, workspace_id)
);

CREATE INDEX IF NOT EXISTS idx_bug_finder_knowledge_user
    ON bug_finder_codebase_knowledge(user_id);

CREATE INDEX IF NOT EXISTS idx_bug_finder_knowledge_freshness
    ON bug_finder_codebase_knowledge(user_id, last_updated DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (228, 'Bug finder codebase knowledge (per-workspace comprehension)');
