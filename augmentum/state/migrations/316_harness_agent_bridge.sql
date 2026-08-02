-- Harness agent bridge: external coding agents (Claude Code, pi, cursor...)
-- register presence and ask the user things through Augmentum notifications;
-- the user answers from any device (approve buttons / free-text reply) and
-- the agent picks the answer up by polling. See augmentum/proxy/agent_bridge.py.

CREATE TABLE IF NOT EXISTS harness_agent_sessions (
    id          TEXT PRIMARY KEY,          -- server-minted agent session id
    user_id     TEXT NOT NULL DEFAULT '',
    harness     TEXT NOT NULL DEFAULT '',  -- claude_code / pi / cursor / ...
    project     TEXT NOT NULL DEFAULT '',  -- X-Augmentum-Project (sanitized)
    title       TEXT NOT NULL DEFAULT '',  -- what the agent says it's doing
    status      TEXT NOT NULL DEFAULT 'working',  -- working | waiting | done
    summary     TEXT NOT NULL DEFAULT '',  -- last self-reported progress note
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_harness_agent_sessions_user
    ON harness_agent_sessions (user_id, last_seen);

CREATE TABLE IF NOT EXISTS harness_agent_requests (
    id               TEXT PRIMARY KEY,     -- request id (polled by the agent)
    user_id          TEXT NOT NULL DEFAULT '',
    agent_session_id TEXT NOT NULL DEFAULT '',
    kind             TEXT NOT NULL DEFAULT 'question',  -- approve | question | review
    title            TEXT NOT NULL DEFAULT '',
    body             TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'pending',   -- pending | answered | cancelled
    reply_action     TEXT NOT NULL DEFAULT '',  -- approve | deny | '' (free-text only)
    reply_text       TEXT NOT NULL DEFAULT '',
    notification_id  TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    answered_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_harness_agent_requests_user
    ON harness_agent_requests (user_id, status, created_at);
