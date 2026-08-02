-- 284_evidence_bus.sql
-- The Evidence Bus (Earned Understanding, P2).
--
-- The unifying intake for "everything that can become a memory": a chat
-- statement, a "Jazz" playlist, a bookmark-with-note, a re-visited article —
-- all land here as Evidence, never as fact. Evidence is cheap, abundant, and
-- INVISIBLE (never injected, never recited). A belief earns durability only
-- when INDEPENDENT sources converge on it (triangulation), so the promotion
-- ladder advances one step per *distinct* corroborating source — a fourth
-- chat mention adds little, a first playlist match adds a lot.
--
-- User-scoped per CLAUDE.md (`user_id` column). `memory_id` links the evidence
-- to the belief it corroborates (nullable — floating evidence not yet bound to
-- a memory is allowed). `source` is the channel (chat_explicit | chat |
-- playlist | bookmark | browse | media_play | ...); `weight` is the raw signal
-- strength before per-source trust weighting (applied at scoring time).
--
-- See docs/superpowers/specs/2026-06-20-earned-understanding-design.md.
CREATE TABLE IF NOT EXISTS evidence (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    companion_id TEXT,
    memory_id    TEXT,                              -- belief this corroborates (nullable)
    subject      TEXT NOT NULL DEFAULT '',          -- topic/belief subject
    claim        TEXT NOT NULL DEFAULT '',          -- what this evidence asserts
    source       TEXT NOT NULL,                     -- channel name
    weight       REAL NOT NULL DEFAULT 1.0,         -- raw signal strength
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_evidence_user_memory ON evidence(user_id, memory_id);
CREATE INDEX IF NOT EXISTS idx_evidence_user_source ON evidence(user_id, source);
