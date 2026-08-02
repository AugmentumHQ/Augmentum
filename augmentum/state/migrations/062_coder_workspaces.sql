-- Coder mode: workspace containers and templates
CREATE TABLE IF NOT EXISTS coder_workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    container_id TEXT,
    template_id TEXT,
    status TEXT NOT NULL DEFAULT 'stopped',
    git_url TEXT,
    created_at REAL NOT NULL,
    last_active REAL,
    resources_cpu REAL DEFAULT 2.0,
    resources_memory TEXT DEFAULT '2g'
);

CREATE TABLE IF NOT EXISTS coder_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    base_image TEXT NOT NULL DEFAULT 'ubuntu:24.04',
    packages TEXT DEFAULT '[]',
    created_at REAL NOT NULL
);

INSERT INTO schema_version (version, applied_at) VALUES (62, strftime('%s', 'now'));
