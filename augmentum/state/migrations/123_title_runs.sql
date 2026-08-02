-- 123_title_runs.sql
-- Per-launch telemetry for the Augmentum Experience Framework (AXF).
-- One row per "play session" of any title (js13k, AGSP-streamed, future
-- emulator ROMs, future web bookmarks). Captures launch latency, runtime
-- choice, exit reason, and adapter-reported metrics so we can answer
-- questions like "is the streamed runtime hitting its 60fps target on
-- this user's hardware" without bolting telemetry onto each runtime
-- separately.
--
-- The artifact_id is intentionally not declared as a hard FK -- artifacts
-- can be uninstalled while we still want to retain the play-history row
-- for "you played X for Y hours total over N runs" stats. Cleanup is
-- handled by an explicit retention sweep elsewhere, not by FK CASCADE.

CREATE TABLE IF NOT EXISTS title_runs (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id),
    artifact_id         TEXT NOT NULL,                          -- target title; loose ref
    runtime_id          TEXT NOT NULL,                          -- 'browser-iframe', 'agsp-streamed', ...
    source_id           TEXT NOT NULL DEFAULT '',               -- 'js13k', 'internal', 'agsp-profile', ...
    started_at          TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at            TEXT,
    duration_s          INTEGER,                                -- materialised on end_run for fast SUM()
    exit_reason         TEXT NOT NULL DEFAULT '',               -- clean|crash|idle|force-stop|abandon
    launch_latency_ms   INTEGER,                                -- click -> first frame
    avg_fps             REAL,
    avg_rtt_ms          REAL,                                   -- streamed runtimes only
    avg_bitrate_kbps    INTEGER,                                -- streamed runtimes only
    crashes             INTEGER NOT NULL DEFAULT 0,
    metadata            TEXT NOT NULL DEFAULT '{}'              -- runtime-specific extras
);

CREATE INDEX IF NOT EXISTS idx_title_runs_user
    ON title_runs(user_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_title_runs_artifact
    ON title_runs(artifact_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_title_runs_open
    ON title_runs(user_id, ended_at)
    WHERE ended_at IS NULL;     -- partial index for "what's still running"

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (123, 'title_runs table');
