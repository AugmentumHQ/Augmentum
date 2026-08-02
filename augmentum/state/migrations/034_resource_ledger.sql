-- Resource ledger: tracks model resource usage across subsystems.

CREATE TABLE IF NOT EXISTS resource_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    subsystem TEXT NOT NULL DEFAULT 'llm',
    backend TEXT NOT NULL,
    vram_mb INTEGER NOT NULL DEFAULT 0,
    ram_mb INTEGER NOT NULL DEFAULT 0,
    device TEXT NOT NULL DEFAULT '',
    quantization TEXT NOT NULL DEFAULT '',
    parameter_size TEXT NOT NULL DEFAULT '',
    family TEXT NOT NULL DEFAULT '',
    pipeline_type TEXT NOT NULL DEFAULT '',
    times_seen INTEGER NOT NULL DEFAULT 1,
    first_seen TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(model_name, backend)
);

CREATE TABLE IF NOT EXISTS resource_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    gpu_total_mb INTEGER NOT NULL DEFAULT 0,
    gpu_used_mb INTEGER NOT NULL DEFAULT 0,
    gpu_free_mb INTEGER NOT NULL DEFAULT 0,
    ram_total_mb INTEGER NOT NULL DEFAULT 0,
    ram_used_mb INTEGER NOT NULL DEFAULT 0,
    ram_free_mb INTEGER NOT NULL DEFAULT 0,
    loaded_model_count INTEGER NOT NULL DEFAULT 0,
    loaded_models_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_resource_snapshots_ts
    ON resource_snapshots(timestamp);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (34, 'Resource ledger');
