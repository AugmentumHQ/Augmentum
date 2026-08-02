-- Coding runs: the unifying record for the Coding Driver — one row per
-- dispatched coding task, whichever engine ran it. The INTERNAL driver
-- enqueues a coder_background_run job (Augmentum drives the loop); the
-- HARNESS driver assigns a task to a live external agent via the bridge
-- (Augmentum delegates + observes). This table is the observable join +
-- diff anchor: it captures base_commit at dispatch so the Agents window can
-- show "what this run changed", plus origin_surface and a durable task
-- record the raw job/agent rows don't carry. Live status is enriched from
-- the job/broker at read time. See augmentum/coder/coding_driver.py.

CREATE TABLE IF NOT EXISTS coding_runs (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL DEFAULT '',
    driver         TEXT NOT NULL DEFAULT 'internal',  -- internal | harness
    engine_ref     TEXT NOT NULL DEFAULT '',  -- job_id (internal) | agent_session_id (harness)
    workspace_id   TEXT NOT NULL DEFAULT '',
    task           TEXT NOT NULL DEFAULT '',
    model          TEXT NOT NULL DEFAULT '',
    base_commit    TEXT NOT NULL DEFAULT '',   -- HEAD short-hash captured at dispatch
    run_id         TEXT NOT NULL DEFAULT '',   -- broker run id, once known
    status         TEXT NOT NULL DEFAULT 'queued',  -- queued|working|done|failed|cancelled
    summary        TEXT NOT NULL DEFAULT '',
    origin_surface TEXT NOT NULL DEFAULT '',   -- coder | companion | voice | phone | ...
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_coding_runs_user
    ON coding_runs (user_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_coding_runs_workspace
    ON coding_runs (user_id, workspace_id);
