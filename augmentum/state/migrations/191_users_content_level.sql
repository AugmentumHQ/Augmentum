-- Per-user content filtering level.
--
-- Values:
--   'unrestricted'  → no filtering (default, preserves existing behaviour)
--   'family'        → server forces SFW on character-import search
--                     (chub.ai + risurealm) regardless of client toggles
--
-- Set by admins via the user management surface. The 'family' level is
-- best-effort SFW — upstream tagging on chub/risurealm isn't a guarantee,
-- but it covers the casual-browse case where a younger user wanders into
-- Story Mode → Import Character and is welcomed by adult cards.
--
-- Only character import is gated today. Other content surfaces (image
-- gen, web browse) may grow level-aware filters later.

ALTER TABLE users ADD COLUMN content_level TEXT NOT NULL DEFAULT 'unrestricted';
