-- 142_bug_finder_runs.sql
-- Bug Finder run history. Each row is one closed-loop pipeline run
-- (intake → workspace prep → plan → detect → verify → fix → report).
--
-- Schema is intentionally light: denormalized counts for list-view
-- queries, plus a single ``report_json`` blob for the full report.
-- Findings, patches, cost ledger entries are inside that blob — Phase 1
-- doesn't need them queryable as rows. If we hit a UX that needs
-- per-finding indexes (e.g. "show me all bugs of type X across runs"),
-- a Phase 2 migration can normalize them out.
--
-- User-scoped per the auth pattern (CLAUDE.md): user_id column,
-- indexed for the "my runs" list view.

CREATE TABLE IF NOT EXISTS bug_finder_runs (
    run_id            TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    job_id            TEXT,             -- background_jobs.id when run as a job
    workspace_id      TEXT,
    git_url           TEXT,

    started_at        INTEGER NOT NULL,
    completed_at      INTEGER,

    stop_reason       TEXT,
    stop_detail       TEXT,
    containment_warning TEXT,

    -- Denormalized counts for cheap list-view rendering
    findings_total      INTEGER NOT NULL DEFAULT 0,
    findings_confirmed  INTEGER NOT NULL DEFAULT 0,
    findings_fixed      INTEGER NOT NULL DEFAULT 0,
    findings_fix_failed INTEGER NOT NULL DEFAULT 0,

    -- Aggregate cost (sums across the cost_ledger entries inside report_json)
    total_tokens_in    INTEGER NOT NULL DEFAULT 0,
    total_tokens_out   INTEGER NOT NULL DEFAULT 0,
    total_wallclock_ms INTEGER NOT NULL DEFAULT 0,

    -- Full structured report (BugFinderRunReport.to_dict-style JSON)
    report_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_bug_finder_runs_user_started
    ON bug_finder_runs(user_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_bug_finder_runs_job
    ON bug_finder_runs(job_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (142, 'Bug finder run history (user-scoped)');
