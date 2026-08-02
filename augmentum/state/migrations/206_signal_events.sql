-- 206_signal_events.sql
-- Signal aggregator substrate — the "Augmentum is telling me what's broken"
-- inbox. Populated by augmentum/signals/aggregator.py on a daily cadence
-- from existing user-scoped sources (bug_finder_runs, companion_journal,
-- coder observation ledgers). No UI yet; no notifications.
--
-- The point of this substrate is to find out, by running it for a few
-- weeks and querying the table with SQL, whether the dedup + categorize
-- layer surfaces patterns that are worth a future Tier-1 inbox UI. If
-- after two weeks of accumulated rows we see things we didn't already
-- know about, the substrate is real. If we see only noise, the
-- categorize layer needs another design pass before going further.
--
-- Schema is intentionally narrow:
--   * One row per (user_id, source, fingerprint). The aggregator
--     UPSERTs — repeat observations bump occurrence_count and
--     last_seen_at instead of creating new rows.
--   * ``category`` is free TEXT (not CHECK-constrained) so we can
--     evolve the vocabulary (bug / gap / drift / gotcha / constraint /
--     polish / other) without a migration.
--   * ``status`` workflow is open → dismissed | resolved. v1 only writes
--     'open'; status changes wait for the inbox UI.
--   * ``details_json`` carries source-specific context (e.g. bug_finder
--     run_id + findings_confirmed; journal entry_id + affect_tag) so
--     a future UI / Becca can deep-link without re-querying.

CREATE TABLE IF NOT EXISTS signal_events (
    id                TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    source            TEXT NOT NULL,         -- 'bug_finder' | 'companion_journal' | 'coder_observations' | ...
    category          TEXT NOT NULL,         -- 'bug' | 'gap' | 'drift' | 'gotcha' | 'constraint' | 'polish' | 'other'
    fingerprint       TEXT NOT NULL,         -- dedup key within (user_id, source)
    summary           TEXT NOT NULL,         -- human-readable one-liner
    details_json      TEXT NOT NULL DEFAULT '{}',
    first_seen_at     INTEGER NOT NULL,
    last_seen_at      INTEGER NOT NULL,
    occurrence_count  INTEGER NOT NULL DEFAULT 1,
    status            TEXT NOT NULL DEFAULT 'open',  -- 'open' | 'dismissed' | 'resolved'
    resolved_at       INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- UPSERT target: every aggregator insert hits this index.
CREATE UNIQUE INDEX IF NOT EXISTS idx_signal_events_dedup
    ON signal_events(user_id, source, fingerprint);

-- Hot-path read: "what's open for this user, newest first".
CREATE INDEX IF NOT EXISTS idx_signal_events_user_status
    ON signal_events(user_id, status, last_seen_at DESC);

-- Secondary read: "show me everything in category X across all users"
-- (admin debugging — rare but useful).
CREATE INDEX IF NOT EXISTS idx_signal_events_category
    ON signal_events(category, last_seen_at DESC);


INSERT OR IGNORE INTO schema_version (version, description)
VALUES (206, 'signal_events: cross-source signal aggregator substrate');
