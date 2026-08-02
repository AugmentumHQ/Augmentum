-- 270_companion_lessons.sql
-- The lesson registry — the *inverse* of the skill graph (mig 193).
--
-- The skill graph accumulates what WORKED: "when the situation looks
-- like X, approach Y is good." The lesson registry accumulates what she
-- was CORRECTED on: "when the situation looks like X, the trap is Y —
-- do Z instead." Same accumulation shape (embedding-indexed by the
-- situation, strength that moves with evidence), applied to the
-- learn-from-failure axis the accumulation thesis names but never wired.
--
-- This is the durable half of "absorbing the lesson to succeed next
-- time": a correction stops being a one-turn event and becomes a held
-- lesson that is retrieved at compose time and honored. Cross-modal by
-- construction — a correction captured from a chat reflection conditions
-- the voice path too, because both compose through the same prompt.
--
-- One table (no separate instances/outcomes ledger like the skill
-- graph): an MVP lesson carries its strength + recurrence counts inline.
-- The thesis discipline (no identity mutation without consent) is
-- unaffected — lessons shape HOW she responds within a turn; they do not
-- touch the personality doc, the kernel, or the genesis anchor.

CREATE TABLE IF NOT EXISTS companion_lessons (
    id              INTEGER PRIMARY KEY,
    companion_id    TEXT NOT NULL,
    user_id         TEXT,                          -- NULL = cross-user/shared; default is per-user
    situation       TEXT NOT NULL DEFAULT '',      -- the trigger shape: "when X happens"
    trap            TEXT NOT NULL DEFAULT '',      -- the mistake to avoid (what she did wrong)
    better          TEXT NOT NULL DEFAULT '',      -- what to do instead, in her voice
    embedding       BLOB,                          -- situation embedding for similarity retrieval
    strength        REAL NOT NULL DEFAULT 0.5,     -- how firmly held; rises on recurrence / successful avoidance
    times_seen      INTEGER NOT NULL DEFAULT 1,    -- how many times this correction has recurred
    times_applied   INTEGER NOT NULL DEFAULT 0,    -- how many times she successfully avoided the trap
    source          TEXT NOT NULL DEFAULT 'reflection',  -- reflection|explicit|inferred
    evidence        TEXT NOT NULL DEFAULT '',      -- short snippet/pointer to where it came from
    status          TEXT NOT NULL DEFAULT 'active',-- active|retired
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_companion_lessons_companion_user
    ON companion_lessons(companion_id, user_id, status, strength DESC);

CREATE INDEX IF NOT EXISTS idx_companion_lessons_active
    ON companion_lessons(companion_id, status, updated_at DESC)
    WHERE status = 'active';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (270, 'companion_lessons: learn-from-correction registry (inverse of the skill graph)');
