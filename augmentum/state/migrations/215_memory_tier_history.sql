-- 215_memory_tier_history.sql
-- Audit trail for memory tier transitions.
--
-- Phase 3 of the memory-establishment rebalance
-- (docs/superpowers/specs/2026-05-31-memory-establishment-rebalance.md)
-- introduces retroactive demotion: a periodic sweep that demotes
-- ACTIVE memories matching a staleness rule down to ARCHIVE. Without
-- an audit record, the user has no way to see WHY a memory dropped
-- in rank or to revert it if the demotion was wrong.
--
-- This table records every tier transition (in either direction) so
-- the inspector UI can show "demoted on 2026-09-15 — idle 184 days,
-- 0 retrievals" and offer a one-click revert. Persists across
-- restarts; orphan-cleanup is deferred until volume is observed (a
-- few hundred rows per user per year is fine without indexed prune).
--
-- Scoped by user_id per the multi-tenant rules — every user-scoped
-- table carries the column. Indexed on memory_id for the planned
-- per-memory revert workflow; secondary index on user_id for the
-- per-user audit list view.

CREATE TABLE IF NOT EXISTS memory_tier_history (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT '',
    memory_id       TEXT NOT NULL,
    from_tier       TEXT NOT NULL,
    to_tier         TEXT NOT NULL,
    reason          TEXT NOT NULL DEFAULT '',
    transitioned_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_tier_history_memory
    ON memory_tier_history(memory_id);

CREATE INDEX IF NOT EXISTS idx_memory_tier_history_user
    ON memory_tier_history(user_id, transitioned_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (215, 'Per-memory tier transition audit trail for retroactive demotion');
