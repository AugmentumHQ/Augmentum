-- 139_resource_snapshots_drop_autoincrement.sql
-- Drop AUTOINCREMENT from resource_snapshots.id.
--
-- AUTOINCREMENT forces SQLite to maintain a per-table row in the
-- sqlite_sequence shadow table, updated on every insert. resource_snapshots
-- is the schema's busiest insert target (~one row per loaded-model change,
-- thousands per day), so its sqlite_sequence row is a hot write page.
-- That page was one of the two corrupt tables in the 2026-05-10 incident,
-- and was suspected in the 2026-05-09 cluster.
--
-- INTEGER PRIMARY KEY (without AUTOINCREMENT) still produces a rowid
-- that is monotonically increasing under normal use. The only difference
-- is that rowids from deleted rows MAY be reused after a wrap; nothing
-- in the codebase consumes resource_snapshots.id as a stable external
-- identifier (reads are by timestamp, pruning is by timestamp), so the
-- reuse semantics don't matter.

CREATE TABLE resource_snapshots_new (
    id INTEGER PRIMARY KEY,
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

INSERT INTO resource_snapshots_new
    (id, timestamp, gpu_total_mb, gpu_used_mb, gpu_free_mb,
     ram_total_mb, ram_used_mb, ram_free_mb,
     loaded_model_count, loaded_models_json)
SELECT id, timestamp, gpu_total_mb, gpu_used_mb, gpu_free_mb,
       ram_total_mb, ram_used_mb, ram_free_mb,
       loaded_model_count, loaded_models_json
FROM resource_snapshots;

DROP TABLE resource_snapshots;
ALTER TABLE resource_snapshots_new RENAME TO resource_snapshots;

CREATE INDEX IF NOT EXISTS idx_resource_snapshots_ts
    ON resource_snapshots(timestamp);

DELETE FROM sqlite_sequence WHERE name = 'resource_snapshots';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (139, 'Drop AUTOINCREMENT from resource_snapshots');
