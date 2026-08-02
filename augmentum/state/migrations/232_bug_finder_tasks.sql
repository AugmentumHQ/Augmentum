-- 232_bug_finder_tasks.sql
-- Per-run task queue. The substrate for the Lead Agent (CC-style
-- dynamic orchestration) that replaces the fixed "plan once, scan
-- every chunk" pipeline. With this table:
--
--   * Planner enqueues `detect` tasks instead of returning a static
--     chunk list. The orchestrator drains the queue.
--   * Investigators (Phase 2) add follow-up tasks when a finding
--     suggests adjacent code is worth examining.
--   * The lead agent (Phase 3) sequences tasks: pick highest-priority,
--     dispatch the right subagent, read result, update queue.
--   * Survives container restart — a mid-run crash doesn't lose
--     queued work; the resumed run picks up where it left off.
--
-- One row per task. Tasks live for the lifetime of one run; cleanup
-- happens via the run-level rows in `bug_finder_runs`.
--
-- User-scoped per the auth pattern. The composite primary key on
-- (user_id, task_id) prevents accidental cross-user reads even if a
-- task_id collision ever happened (sha256 16-hex collision is
-- effectively zero, but belt + suspenders).

CREATE TABLE IF NOT EXISTS bug_finder_tasks (
    task_id          TEXT NOT NULL,
        -- sha256(run_id|kind|target_json) truncated — stable across
        -- repeated enqueues so the same task collapses idempotently.
    user_id          TEXT NOT NULL DEFAULT '',
    run_id           TEXT NOT NULL,
    workspace_id     TEXT NOT NULL DEFAULT '',

    kind             TEXT NOT NULL,
        -- detect | investigate | verify | fix | critique | comprehend_refresh
    target_json      TEXT NOT NULL DEFAULT '{}',
        -- Kind-specific target payload. For detect: {file, function,
        -- line_start, line_end, rationale}. For investigate:
        -- {thread_anchor, scope_hint}. For verify: {finding_id}. The
        -- JSON shape stays free-form so adding kinds doesn't require
        -- migrations.

    reason           TEXT NOT NULL DEFAULT '',
        -- Free-text rationale shown to the lead agent so it can
        -- weigh urgency. "planner flagged auth-sensitive area";
        -- "investigator: same exception pattern as finding #3"; etc.

    priority         INTEGER NOT NULL DEFAULT 5,
        -- 1 (low) - 10 (high). Lead's take_next() picks highest first.
        -- Investigators add tasks at priority+1 of their parent so a
        -- followed thread compounds rather than starves.

    status           TEXT NOT NULL DEFAULT 'pending',
        -- pending | in_progress | completed | dropped | failed
    parent_task_id   TEXT NOT NULL DEFAULT '',
        -- Empty when the task was originally enqueued by the planner.
        -- Investigators / lead spawn child tasks; this lets the
        -- summary view render task trees.
    created_by       TEXT NOT NULL DEFAULT 'planner',
        -- planner | investigator | lead | critique | resume
    result_summary   TEXT NOT NULL DEFAULT '',
        -- Short text written by the dispatcher when status moves to
        -- a terminal state. Surfaced to the lead so it knows what
        -- happened.

    created_at       INTEGER NOT NULL,
    completed_at     INTEGER NOT NULL DEFAULT 0,

    PRIMARY KEY (user_id, task_id)
);

CREATE INDEX IF NOT EXISTS idx_bug_finder_tasks_user_run
    ON bug_finder_tasks(user_id, run_id);

CREATE INDEX IF NOT EXISTS idx_bug_finder_tasks_user_status
    ON bug_finder_tasks(user_id, run_id, status, priority DESC);

CREATE INDEX IF NOT EXISTS idx_bug_finder_tasks_parent
    ON bug_finder_tasks(user_id, parent_task_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (232, 'Bug finder tasks (Lead agent substrate)');
