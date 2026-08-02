-- 202_dance_ratings.sql
--
-- Server-authoritative animation ratings for the Becca companion
-- widget timeline. Replaces the localStorage-only blob in
-- ui/scripts/movement-conductor.js (_ratings keyed by anim_id) so
-- curation follows the user across devices.
--
-- Schema mirrors the JS shape exactly:
--   _ratings = { animId: { kind, slotBonusSec, ts } }
-- becomes
--   one row per (user_id, anim_id), with optional kind + accumulator.
--
-- Fields:
--   kind            'like' | 'dislike' | 'broken' | NULL.
--                   NULL means "rating cleared but slot bonus retained"
--                   (a 'longer' click stores a row even without a kind).
--   slot_bonus_sec  per-id slot extension from 'longer' ratings,
--                   accumulating (cap enforced at write time, not in
--                   the schema, so the cap can move without ALTER).
--   updated_at      ms epoch, matches JS Date.now() like dance_history.
--
-- Composite primary key (user_id, anim_id) — one row per user per
-- animation. Upserts via the UNIQUE constraint.

CREATE TABLE IF NOT EXISTS dance_ratings (
    user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    anim_id        TEXT NOT NULL,
    kind           TEXT,
    slot_bonus_sec INTEGER NOT NULL DEFAULT 0,
    updated_at     INTEGER NOT NULL,
    PRIMARY KEY (user_id, anim_id)
);

CREATE INDEX IF NOT EXISTS idx_dance_ratings_user
    ON dance_ratings(user_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (202, 'dance_ratings - server-authoritative widget curation');
