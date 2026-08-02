-- 225_bug_finder_findings_normalized.sql
-- Per-finding normalized rows. Companion to bug_finder_runs (142),
-- which keeps the full BugFinderRunReport as a JSON blob.
--
-- Motivation: cross-run analytics. "Show me every CONFIRMED injection
-- across all runs against workspace X", or "is this signature recurring
-- after we shipped a fix" — both require rehydrating every report blob
-- without this. AIxCC teams ran SARIF-shaped storage for the same
-- reason: blob-only schemas can't power signature-recurrence detection,
-- regression alerts, or cross-workspace pattern aggregation.
--
-- The findings inside report_json remain the source of truth — these
-- rows are projected from them at run-completion time. If they get out
-- of sync (manual surgery, partial backfill), re-derive from blobs.
--
-- User-scoped: every row has user_id matching the parent run. Strict
-- adherence to the auth pattern (CLAUDE.md) so cross-tenant queries
-- can never leak finding text.

CREATE TABLE IF NOT EXISTS bug_finder_findings (
    finding_id        TEXT NOT NULL,
    run_id            TEXT NOT NULL,
    user_id           TEXT NOT NULL,
    workspace_id      TEXT,

    file              TEXT NOT NULL,
    function          TEXT NOT NULL DEFAULT '<module>',
    -- Line range when the detector emitted one in evidence_paths
    -- (parsed as `file.py:14` or `file.py:14-22`). Nullable because
    -- not all detectors include lines.
    line_start        INTEGER,
    line_end          INTEGER,

    claim             TEXT NOT NULL,
    claim_signature   TEXT NOT NULL,
    severity          TEXT NOT NULL,
    status            TEXT NOT NULL,

    -- Cross-run-variance signal: # of detector runs (within one pipeline
    -- run) that flagged this finding, plus the denominator.
    runs_to_confirm   INTEGER NOT NULL DEFAULT 0,
    total_runs        INTEGER NOT NULL DEFAULT 0,

    has_repro         INTEGER NOT NULL DEFAULT 0,
    has_patch         INTEGER NOT NULL DEFAULT 0,
    fix_attempts      INTEGER NOT NULL DEFAULT 0,

    -- Unix epoch seconds — matches bug_finder_runs.completed_at granularity
    -- so cross-run aggregation queries work without arithmetic.
    detected_at       INTEGER NOT NULL,

    PRIMARY KEY (run_id, finding_id)
);

-- Hot indexes for the queries we know we need:
--   - List by user (always — multi-tenant invariant)
--   - Filter by signature for trend analysis
--   - Filter by file for "what's the bug history of this file"
--   - Filter by status to dashboard CONFIRMED-but-unfixed work
--   - Filter by workspace for per-project rollups

CREATE INDEX IF NOT EXISTS idx_bug_finder_findings_user_run
    ON bug_finder_findings(user_id, run_id);

CREATE INDEX IF NOT EXISTS idx_bug_finder_findings_user_signature
    ON bug_finder_findings(user_id, claim_signature);

CREATE INDEX IF NOT EXISTS idx_bug_finder_findings_user_file
    ON bug_finder_findings(user_id, file);

CREATE INDEX IF NOT EXISTS idx_bug_finder_findings_user_status
    ON bug_finder_findings(user_id, status);

CREATE INDEX IF NOT EXISTS idx_bug_finder_findings_workspace
    ON bug_finder_findings(workspace_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (225, 'Bug finder findings (normalized, user-scoped)');
