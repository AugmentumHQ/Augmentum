-- 201_dance_history.sql
--
-- Server-authoritative dance playback history for the Becca companion
-- widget timeline. Replaces the localStorage-only ring buffer in
-- ui/scripts/becca-presence.js so curation persists across devices and
-- becomes the substrate for user-uploaded animations + curated loops
-- (Phase B/C of [[project-companion-widget-text-polish]] follow-up).
--
-- Schema:
--   id          per-row uuid, lets the route surface a deterministic
--               id when the client appends (so server-side trimming
--               can target a specific play).
--   user_id    tenant scope, ON DELETE CASCADE per the
--               user-deletion strands fix.
--   anim_id    logical reference into either ATLAS (code-defined,
--               e.g. 'kebab-dance') or user_animations (Phase B).
--               No FK — anim_id is a soft pointer that survives an
--               atlas rename or upload deletion (history still
--               shows the label that was current at play time).
--   label      denormalized human-readable name at play time.
--   played_at  ms epoch (matches the JS Date.now() shape already in
--               localStorage so a one-shot migration of legacy entries
--               can land later without unit conversion).
--   duration_sec  actual clip duration (REAL — sub-second precision
--               matters for short micro-gestures).
--   mode       conductor mode at play time ('chat-call' / 'narrative'
--               / 'passthrough' / ...). Nullable so legacy/localStorage
--               imports can land without a synthetic value. Used by
--               Phase D filter chips.
--
-- Listing path: user's recent plays, newest first, capped by the
-- widget to ~50 entries. The compound index covers both the timeline
-- query AND a future "per-anim play count" rollup.

CREATE TABLE IF NOT EXISTS dance_history (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    anim_id      TEXT NOT NULL,
    label        TEXT NOT NULL,
    played_at    INTEGER NOT NULL,
    duration_sec REAL NOT NULL DEFAULT 0,
    mode         TEXT
);

CREATE INDEX IF NOT EXISTS idx_dance_history_user_played
    ON dance_history(user_id, played_at DESC);

CREATE INDEX IF NOT EXISTS idx_dance_history_user_anim
    ON dance_history(user_id, anim_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (201, 'dance_history - server-authoritative widget timeline');
