-- 185_cardsmith_sessions.sql
--
-- Durable backing store for in-flight Cardsmith sessions. Previously these
-- lived only in `augmentum.modes.narrative.cardsmith.state._sessions` —
-- an OrderedDict in process memory — so any server restart, crash, or
-- LRU eviction dropped the entire conversation. Users could lose a 15+
-- turn world-building session (with fetched wiki scratchpad and many
-- committed field emissions) to a routine container restart.
--
-- This table is a write-through cache: the in-memory dict remains the
-- read path for hot sessions, and every durable mutation (user message
-- appended, assistant reply finalized, scratchpad updated by the fetch
-- loop, session finalized) writes through to here. `get_session` falls
-- through to this table when an unknown session_id arrives, rehydrating
-- the OrderedDict entry on demand.
--
-- Schema rationale:
--   - `messages` / `fields` / `meta` are JSON blobs. The shapes are
--     model-emitted and frequently nested (lorebook entries are objects,
--     scratchpad rows are objects with zone + content + path). A column-
--     per-field model would balloon the migration list and re-do the
--     work the OrderedDict already does cleanly.
--   - `created_at` / `last_active_at` stored as REAL UNIX seconds to
--     match the in-memory `time.time()` representation — no datetime
--     parsing on hot read paths.
--   - `finalized` is INTEGER 0/1 to match SQLite boolean idiom.
--
-- Eviction policy is unchanged: the in-memory _evict_stale() still runs
-- per the existing TTL + LRU rules. A periodic sweep (added separately)
-- prunes rows older than TTL from this table — keeping disk in sync
-- with the memory eviction.

CREATE TABLE IF NOT EXISTS cardsmith_sessions (
    session_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    card_type       TEXT NOT NULL,
    source          TEXT NOT NULL,
    created_at      REAL NOT NULL,
    last_active_at  REAL NOT NULL,
    messages        TEXT NOT NULL DEFAULT '[]',
    fields          TEXT NOT NULL DEFAULT '{}',
    meta            TEXT NOT NULL DEFAULT '{}',
    finalized       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cardsmith_sessions_user
    ON cardsmith_sessions(user_id, last_active_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (185, 'cardsmith_sessions: durable write-through cache for in-flight cardsmith sessions');
