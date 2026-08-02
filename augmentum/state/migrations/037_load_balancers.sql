-- Load balancer definitions
CREATE TABLE IF NOT EXISTS load_balancers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    strategy TEXT NOT NULL DEFAULT 'round_robin',
    fallback_enabled INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Member models belonging to a balancer
CREATE TABLE IF NOT EXISTS load_balancer_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    balancer_id TEXT NOT NULL REFERENCES load_balancers(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    backend_key TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    priority INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_used_at TEXT,
    UNIQUE(balancer_id, model_name, backend_key)
);

-- A/B test vote tracking
CREATE TABLE IF NOT EXISTS ab_test_votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    balancer_id TEXT NOT NULL REFERENCES load_balancers(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    backend_key TEXT NOT NULL,
    vote TEXT NOT NULL CHECK(vote IN ('up', 'down')),
    session_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_lb_members_balancer ON load_balancer_members(balancer_id);
CREATE INDEX IF NOT EXISTS idx_ab_votes_balancer ON ab_test_votes(balancer_id);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (37, 'load_balancers');
