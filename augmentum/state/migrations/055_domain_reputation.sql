-- 055_domain_reputation.sql
-- Track domain quality scores for browse search result ranking.
-- Scores update automatically on fetch success/failure and user actions.
-- Preferred sources get an initial positive score.

CREATE TABLE IF NOT EXISTS domain_reputation (
    domain TEXT PRIMARY KEY,
    score INTEGER DEFAULT 0,
    fetch_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    fail_count INTEGER DEFAULT 0,
    user_action_count INTEGER DEFAULT 0,
    last_fetched TEXT,
    last_action TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
