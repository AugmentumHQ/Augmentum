-- 298_notification_observations.sql
-- L0 acquisition: the on-device notification stream, aggregated locally.
--
-- The first real data stream for the Sovereign Perception Pipeline
-- (docs/superpowers/specs/2026-06-25-sovereign-perception-pipeline-design.md).
-- An Android NotificationListenerService reads posted notifications across
-- all apps and uploads normalized entities here, on the user's OWN server —
-- the sovereignty contract: read on-device, aggregate on-device, leaves only
-- via the user's own pull, never a third party.
--
-- These rows are NOT surfaced as-is (that would be the echo machine). They are
-- the raw material a perception fuser correlates into insights: ≥2 unread from
-- one person during a focus block → "Jordan's been trying to reach you", an
-- alert × a calendar event → "flight slipped, don't leave at 4". One row is an
-- echo; the fusion of several is the intelligence.
--
-- USER-SCOPED (multi-tenant rule): every row carries user_id; the store fns
-- take user_id; the ingest route refuses the anon sentinel. Sensitive by
-- nature (notification bodies) — gated behind companion_perception_acquire_
-- notifications (default OFF) AND the Android special-access grant, and pruned
-- to a short retention window since fusion only needs the recent past.

CREATE TABLE IF NOT EXISTS notification_observations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    source_pkg   TEXT NOT NULL DEFAULT '',    -- com.whatsapp (the originating app id)
    source_app   TEXT NOT NULL DEFAULT '',    -- "WhatsApp" (best-effort human label)
    category     TEXT NOT NULL DEFAULT '',    -- android Notification.category: msg|email|call|alarm|...
    title        TEXT NOT NULL DEFAULT '',    -- notification title (often the sender)
    body         TEXT NOT NULL DEFAULT '',    -- notification text
    person       TEXT NOT NULL DEFAULT '',    -- normalized sender (title or EXTRA_PEOPLE), best-effort
    is_message   INTEGER NOT NULL DEFAULT 0,  -- 1 = a person-to-person message (category msg / MessagingStyle)
    posted_at    REAL NOT NULL DEFAULT 0,     -- device epoch seconds the notification was posted
    ingested_at  REAL NOT NULL DEFAULT 0,     -- server epoch seconds we received it
    dedup_key    TEXT NOT NULL DEFAULT '',    -- pkg|notif_key|posted_at — collapses re-posts/updates
    payload      TEXT NOT NULL DEFAULT '{}',  -- JSON: any extra normalized fields (channel, group, count)
    UNIQUE(user_id, dedup_key)
);

CREATE INDEX IF NOT EXISTS idx_notif_obs_user_time
    ON notification_observations (user_id, posted_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (298, 'notification_observations: L0 acquisition store for the on-device notification stream (sovereign perception pipeline)');
