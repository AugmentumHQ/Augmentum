-- 188_trusted_receivers_prefs.sql
--
-- Per-receiver display preferences. Each TV gets its own preference
-- bag so a user can hide the Gallery rail on the living-room TV
-- without it disappearing from the bedroom TV or the phone
-- controller. Stored as a JSON blob so the schema can evolve without
-- migrations every time we expose a new toggle.
--
-- Default '{}' = use server-side defaults (see receiver_prefs.py).
-- Unknown keys at write time are rejected by the schema validator
-- rather than silently stored, so a stale client can't dirty the bag
-- with garbage the next version has to clean up.

ALTER TABLE trusted_receivers
    ADD COLUMN prefs_json TEXT NOT NULL DEFAULT '{}';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (
    188,
    'trusted_receivers.prefs_json for per-receiver display preferences'
);
