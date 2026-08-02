-- 258_standing_task_runs_and_metric_observations.sql
-- Scheduled requests & watches, Phase 1 substrate (spec:
-- docs/superpowers/specs/2026-06-11-scheduled-requests-and-watches-design.md).
--
-- Two tables, no ALTERs. Per-task config this feature adds (intent,
-- condition, delivery, noise_state) lives in the existing
-- companion_standing_tasks.params JSON — the established pattern for
-- one_shot / last_hash / seen-URL baselines.
--
-- companion_standing_task_runs — one row per step()/run_now() execution,
-- INCLUDING runs that found nothing. This is the trust surface: "checked
-- 2h ago, nothing new" is what distinguishes a quiet watch from a dead
-- one (silent non-execution is the canonical complaint against every
-- shipped scheduled-task product). Status vocabulary:
--   fired       — noteworthy result, delivered (notify or digest)
--   silent      — ran clean, nothing new
--   suppressed  — runner said noteworthy, importance judge said no
--                 (logged here, not delivered; verdict in details)
--   error       — runner raised; feeds the consecutive-error auto-pause
-- Retention: newest 20 per task, trimmed opportunistically on insert.
--
-- companion_metric_observations — append-only numeric series for watches
-- that track a number (price, temperature, rate). Keepa-style
-- change-event rows: scaled integers, never mutated; corrections arrive
-- as new observations. "missing" (couldn't extract) is recorded
-- distinctly from a healthy reading — absence of data is not data.
--   status: ok | quarantined (failed sanity bounds, needs confirmation)
--         | missing (extraction produced nothing)

CREATE TABLE IF NOT EXISTS companion_standing_task_runs (
    id INTEGER PRIMARY KEY,
    task_id INTEGER NOT NULL
        REFERENCES companion_standing_tasks(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ran_at TEXT NOT NULL DEFAULT (datetime('now')),
    status TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    details TEXT NOT NULL DEFAULT '{}'        -- JSON: judge verdict,
                                              -- tool_trace, elapsed_ms
);

CREATE INDEX IF NOT EXISTS idx_standing_task_runs_task
    ON companion_standing_task_runs(task_id, ran_at DESC);

CREATE INDEX IF NOT EXISTS idx_standing_task_runs_user
    ON companion_standing_task_runs(user_id, ran_at DESC);

CREATE TABLE IF NOT EXISTS companion_metric_observations (
    id INTEGER PRIMARY KEY,
    task_id INTEGER NOT NULL
        REFERENCES companion_standing_tasks(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    series TEXT NOT NULL DEFAULT 'value',     -- one task may track >1
    observed_at TEXT NOT NULL DEFAULT (datetime('now')),
    value INTEGER,                            -- scaled; NULL = missing
    scale INTEGER NOT NULL DEFAULT 100,       -- value/scale = real number
    unit TEXT NOT NULL DEFAULT '',            -- 'USD', 'F', '%'
    method TEXT NOT NULL DEFAULT '',          -- provider|json-ld|pattern|llm
    status TEXT NOT NULL DEFAULT 'ok',
    evidence TEXT                             -- verbatim quote / field path
);

CREATE INDEX IF NOT EXISTS idx_metric_obs_task
    ON companion_metric_observations(task_id, series, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_metric_obs_user
    ON companion_metric_observations(user_id, observed_at DESC);
