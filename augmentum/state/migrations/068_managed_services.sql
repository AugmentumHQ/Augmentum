-- Managed Docker services (provider marketplace)
CREATE TABLE IF NOT EXISTS managed_services (
    id TEXT PRIMARY KEY,
    definition_id TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    image TEXT NOT NULL,
    container_id TEXT,
    host_port INTEGER NOT NULL DEFAULT 0,
    internal_port INTEGER NOT NULL DEFAULT 0,
    config_json TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'stopped',
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_managed_services_category ON managed_services(category);
CREATE INDEX IF NOT EXISTS idx_managed_services_enabled ON managed_services(enabled);
