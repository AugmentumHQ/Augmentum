-- Per-turn archive for the coder mode — full record of each turn the
-- agent runs in a workspace, durable beyond the in-prompt FIFO
-- (turn_summaries cap 10) and beyond conversation compaction.
--
-- Why a new table when ``coder_turn_runs`` exists: that table is
-- metadata-only (model, strategy, finish_reason, costs). This one
-- carries the structured BODY of what happened — user goal, files
-- touched, tool-call sequence, summary, outcome — so the model can
-- recall earlier work via semantic search (next phase) and the
-- inspector can render a full timeline (future phase).
--
-- Bi-temporal timestamps from day 1 per Zep precedent
-- ([[project_coder_archive_research]]):
--   * event_time   — when the event happened in the user's clock
--   * recorded_at  — when we wrote the row (always ≥ event_time, but
--                    catches up-writes after a delay)
-- Without both, queries like "what did the agent think was true as of
-- turn 47" can't distinguish "fact wasn't known yet" from "we wrote
-- this row late." Append-only design means we still surface stale
-- entries until decay/supersession is added (Phase 2 of the LTM spec).
--
-- Workspace-scoped + user-scoped per the multi-tenant pattern. Row
-- cap is enforced in the store layer (coder/turn_archive.py), not
-- here, so the cap is tunable without a migration.
--
-- ``embedding_status`` column reserves the embedding pipeline hook
-- without committing to a specific embedding model. Phase 2 of LTM
-- writes vectors to a separate sqlite-vec table and flips this column
-- from 'pending' → 'embedded' so we can incrementally backfill.

CREATE TABLE IF NOT EXISTS coder_turn_archive (
    archive_id       TEXT PRIMARY KEY,           -- short UUID
    user_id          TEXT NOT NULL DEFAULT '',   -- user-scoped (CLAUDE.md)
    workspace_id     TEXT NOT NULL DEFAULT '',   -- per-project archive
    run_id           TEXT NOT NULL DEFAULT '',   -- coder_turn_runs FK (loose)
    turn_id          TEXT NOT NULL DEFAULT '',   -- state.active_turn_id
    turn_index       INTEGER NOT NULL DEFAULT 0, -- monotonic within workspace

    user_goal        TEXT NOT NULL DEFAULT '',   -- extracted goal text
    outcome          TEXT NOT NULL DEFAULT '',   -- done | incomplete | cancelled | error
    verdict_reason   TEXT NOT NULL DEFAULT '',   -- TQG verdict tag (already_nudged, etc.)
    blockers         TEXT NOT NULL DEFAULT '',   -- last error / blocker text

    -- JSON-encoded lists. We keep raw arrays so future schema growth
    -- (e.g., per-file disposition: created/edited/deleted) lives
    -- inside the JSON without ALTER-TABLE churn.
    files_read       TEXT NOT NULL DEFAULT '[]', -- ["path", ...]
    files_edited     TEXT NOT NULL DEFAULT '[]', -- [{"path","tool","lines_written"}, ...]
    shell_commands   TEXT NOT NULL DEFAULT '[]', -- ["cmd", ...]
    edits            TEXT NOT NULL DEFAULT '[]', -- per-edit search/replace snippets

    -- The compaction-grade summary that would otherwise have been
    -- the only surviving trace once the FIFO rolled over.
    summary          TEXT NOT NULL DEFAULT '',

    -- Token usage at archive time. Indicative, not authoritative —
    -- per-turn cost lives on coder_turn_runs which has provider data.
    tokens_in        INTEGER NOT NULL DEFAULT 0,
    tokens_out       INTEGER NOT NULL DEFAULT 0,

    -- Bi-temporal anchors. event_time is when the turn finished
    -- in wall-clock; recorded_at is when this row was committed.
    -- They match on the common path; diverge on backfill / replay.
    event_time       INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    recorded_at      INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),

    -- Phase-2 hooks: append-only today; add lifecycle later.
    embedding_status TEXT NOT NULL DEFAULT 'pending',  -- pending | embedded | skipped
    superseded_by    TEXT NOT NULL DEFAULT '',         -- archive_id of a newer entry that
                                                       -- contradicts this one (future)
    confidence       REAL NOT NULL DEFAULT 1.0         -- decay applied in retrieval, not write
);

-- The two common reads:
--   1. List by (user, workspace, turn_index DESC) — inspector timeline
--   2. List by (user, workspace, event_time DESC) — recency-ordered recall
-- Both indices share the (user_id, workspace_id) prefix so the per-
-- workspace partition is filterable without scanning.
CREATE INDEX IF NOT EXISTS idx_coder_turn_archive_user_ws_idx
    ON coder_turn_archive(user_id, workspace_id, turn_index DESC);

CREATE INDEX IF NOT EXISTS idx_coder_turn_archive_user_ws_time
    ON coder_turn_archive(user_id, workspace_id, event_time DESC);

-- Embedding-pending sweep — Phase 2 backfill walks this index to find
-- rows the embedder hasn't seen yet.
CREATE INDEX IF NOT EXISTS idx_coder_turn_archive_embedding_status
    ON coder_turn_archive(embedding_status, recorded_at);
