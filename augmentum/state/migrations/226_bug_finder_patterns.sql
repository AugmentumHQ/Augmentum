-- 226_bug_finder_patterns.sql
-- Cross-run pattern memory. Persists "things we've seen in this
-- workspace before" so subsequent runs' planners can prioritize
-- recurring hotspots.
--
-- This is *durable* memory, distinct from the read-time aggregation
-- query `signature_recurrence` over bug_finder_findings (which is
-- great for ad-hoc analytics but doesn't survive a pruning of old
-- runs). A row here lives until the user explicitly forgets a
-- pattern or it ages out.
--
-- Companion to:
--   - Anthropic's bug-finder memory observations (recurring findings
--     compound over audits)
--   - Semgrep's accumulated-rule-library pattern (TPs become rules)
--   - XBOW's exploit fingerprint storage
--
-- User-scoped per the auth pattern.

CREATE TABLE IF NOT EXISTS bug_finder_patterns (
    pattern_id        TEXT PRIMARY KEY,         -- sha256(user_id|workspace_id|signature|file)
    user_id           TEXT NOT NULL,
    workspace_id      TEXT NOT NULL DEFAULT '', -- '' = cross-workspace pattern

    claim_signature   TEXT NOT NULL,
    file              TEXT NOT NULL,            -- specific file (Phase 1); add glob in Phase 2

    first_seen_at     INTEGER NOT NULL,
    last_seen_at      INTEGER NOT NULL,
    last_run_id       TEXT NOT NULL DEFAULT '',

    -- Counters update with every run that hits this pattern
    hit_count         INTEGER NOT NULL DEFAULT 1,     -- # runs that fired
    fix_count         INTEGER NOT NULL DEFAULT 0,     -- # times it was fixed
    speculative_count INTEGER NOT NULL DEFAULT 0,    -- # times it stayed speculative

    -- One representative claim text + severity for planner context
    sample_claim      TEXT NOT NULL DEFAULT '',
    last_severity     TEXT NOT NULL DEFAULT 'medium',

    -- Optional user-supplied annotation
    note              TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_bug_finder_patterns_user_workspace
    ON bug_finder_patterns(user_id, workspace_id);

CREATE INDEX IF NOT EXISTS idx_bug_finder_patterns_user_signature
    ON bug_finder_patterns(user_id, claim_signature);

CREATE INDEX IF NOT EXISTS idx_bug_finder_patterns_user_lastseen
    ON bug_finder_patterns(user_id, last_seen_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (226, 'Bug finder patterns (cross-run memory, user-scoped)');
