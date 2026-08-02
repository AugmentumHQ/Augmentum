-- Generic background-job queue.
--
-- A persistent, restart-survivable primitive for long-running asynchronous
-- work. Each row is one unit of work owned by one user, dispatched through
-- an in-process JobRunner that looks up handlers by ``job_type``.
--
-- Design notes:
--   * ``job_type`` is a free-form discriminator (e.g. ``book_transcribe``,
--     ``video_subtitle``). The runner maintains a registry of handlers
--     keyed by this string; unknown types are marked failed with a clear
--     error instead of crashing the worker.
--   * ``payload`` is a JSON blob interpreted by the handler. The queue
--     itself does not care about its shape.
--   * ``cancel_requested`` is the cooperative-cancel signal. Handlers
--     check it between chunks; the runner checks it before dispatch.
--   * ``attempts`` / ``max_attempts`` cover automatic retry on transient
--     failure. Handlers opt out by raising a non-retryable exception
--     (see augmentum/jobs/runner.py).
--   * On startup, rows with ``status='running'`` are re-queued as
--     ``status='pending'`` (the worker was killed mid-run). This is safe
--     as long as handlers are idempotent — which is a hard requirement
--     called out in the runner's docstring.
--
-- The pending-dispatch index orders by (priority DESC, created_at ASC) so
-- higher-priority jobs jump the line without starving older work at the
-- same priority.

CREATE TABLE IF NOT EXISTS background_jobs (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users(id),
    job_type         TEXT NOT NULL,
    payload          TEXT NOT NULL DEFAULT '{}',
    status           TEXT NOT NULL DEFAULT 'pending',
    progress         REAL NOT NULL DEFAULT 0.0,
    stage            TEXT NOT NULL DEFAULT '',
    result           TEXT,
    error            TEXT,
    priority         INTEGER NOT NULL DEFAULT 0,
    attempts         INTEGER NOT NULL DEFAULT 0,
    max_attempts     INTEGER NOT NULL DEFAULT 3,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at       INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    started_at       INTEGER,
    updated_at       INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    completed_at     INTEGER
);

-- Worker dispatch: pick pending with highest priority, oldest first.
CREATE INDEX IF NOT EXISTS idx_background_jobs_dispatch
    ON background_jobs(status, priority DESC, created_at ASC);

-- Per-user listing (status page, progress polling).
CREATE INDEX IF NOT EXISTS idx_background_jobs_user_status
    ON background_jobs(user_id, status, created_at DESC);

-- Per-type status queries (e.g. "is there already a pending transcription
-- for this book?" — handlers enforce dedup using this index).
CREATE INDEX IF NOT EXISTS idx_background_jobs_type_status
    ON background_jobs(job_type, status);

INSERT INTO schema_version (version, applied_at) VALUES (102, strftime('%s', 'now'));
