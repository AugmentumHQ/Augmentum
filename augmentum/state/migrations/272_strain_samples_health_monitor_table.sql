-- 272_strain_samples_health_monitor_table.sql
-- General-purpose server-strain time series for the health monitor.
-- Server-level (NOT user-scoped) — like resource_snapshots, this captures
-- process-/host-wide state, plus a count of how many distinct clients
-- (browsers/tabs/devices) were active at sample time so concurrent
-- multi-browser contention can be correlated against strain after the fact.
-- Rolling retention is enforced in code (StrainMonitor._maybe_prune), mirroring
-- the resource_snapshots 7-day window.
--
-- Plain INTEGER PRIMARY KEY (rowid) — never AUTOINCREMENT (see migration 139).
CREATE TABLE IF NOT EXISTS strain_samples (
    id                  INTEGER PRIMARY KEY,
    timestamp           TEXT NOT NULL DEFAULT (datetime('now')),

    -- Event loop / request pressure
    event_loop_lag_ms   REAL NOT NULL DEFAULT 0,    -- last measured loop lag
    inflight_requests   INTEGER NOT NULL DEFAULT 0, -- HTTP requests in flight at sample
    slow_requests       INTEGER NOT NULL DEFAULT 0, -- slow_request log hits since last sample

    -- Concurrency / multi-client context (the "combination" signal)
    active_clients      INTEGER NOT NULL DEFAULT 0, -- distinct tabs/browsers seen recently
    active_users        INTEGER NOT NULL DEFAULT 0, -- distinct user_ids among active clients
    ws_presence         INTEGER NOT NULL DEFAULT 0, -- companion presence pipelines
    ws_notify           INTEGER NOT NULL DEFAULT 0, -- notification hub connections
    sessions_narrative  INTEGER NOT NULL DEFAULT 0, -- cached narrative engines
    sessions_agentic    INTEGER NOT NULL DEFAULT 0, -- cached agentic handlers
    sessions_coder      INTEGER NOT NULL DEFAULT 0, -- active coder workspaces

    -- Shared single-slot resources
    engine_model        TEXT NOT NULL DEFAULT '',   -- resident primary model
    engine_secondary    TEXT NOT NULL DEFAULT '',   -- resident Slot-B model, if any
    db_write_ms         REAL NOT NULL DEFAULT 0,     -- this sample's own BEGIN IMMEDIATE+insert latency (writer-lock contention proxy)

    -- Host / process resource pressure
    gpu_used_mb         INTEGER NOT NULL DEFAULT 0,
    gpu_free_mb         INTEGER NOT NULL DEFAULT 0,
    ram_used_mb         INTEGER NOT NULL DEFAULT 0,
    ram_free_mb         INTEGER NOT NULL DEFAULT 0,
    proc_rss_mb         INTEGER NOT NULL DEFAULT 0,  -- this process' resident set

    -- Free-form extension slot (per-mode breakdown, ws detail, top client activities)
    context_json        TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_strain_samples_ts
    ON strain_samples (timestamp DESC);
