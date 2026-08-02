-- Reasoning flow pipeline editor: user-customizable reasoning pipelines.

CREATE TABLE IF NOT EXISTS reasoning_flows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    icon TEXT DEFAULT '',
    version INTEGER DEFAULT 1,
    is_default BOOLEAN DEFAULT 0,
    is_builtin BOOLEAN DEFAULT 0,
    auto_select BOOLEAN DEFAULT 1,
    trigger_domains TEXT DEFAULT '[]',
    trigger_keywords TEXT DEFAULT '[]',
    pinned_models TEXT DEFAULT '[]',
    auto_search BOOLEAN DEFAULT 1,
    max_tool_calls_per_step INTEGER DEFAULT 3,
    autonomy_level INTEGER DEFAULT 2,
    escalation_flow TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reasoning_flow_steps (
    id TEXT PRIMARY KEY,
    flow_id TEXT NOT NULL REFERENCES reasoning_flows(id) ON DELETE CASCADE,
    sort_order INTEGER NOT NULL,
    name TEXT NOT NULL,
    system_prompt TEXT DEFAULT '',
    user_template TEXT DEFAULT '',
    role TEXT DEFAULT 'analyze',
    tool_categories TEXT DEFAULT '[]',
    tool_names TEXT DEFAULT '[]',
    complexity_gate TEXT DEFAULT '[]',
    stream_to_user BOOLEAN DEFAULT 0,
    output_cap INTEGER DEFAULT 800,
    enabled BOOLEAN DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_flow_steps_flow ON reasoning_flow_steps(flow_id, sort_order);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (11, 'Reasoning flow pipeline editor');
