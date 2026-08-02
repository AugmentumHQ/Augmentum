-- 280_users_email_identity.sql
-- Identity foundation (Connect comms platform, Phase 0).
--
-- Give user accounts an email address + a verified flag. Two reasons:
--   1. Invite onboarding (Phase 1) can carry an invitee email so the
--      claim link can be addressed to a person and (later) emailed.
--   2. Future account recovery / notification routing.
--
-- Both columns are additive and back-compat: existing accounts default to
-- empty email / unverified, which is exactly the pre-migration behaviour
-- (no email at all). The `users` table is a server-level table (NOT
-- user-scoped) so no user_id column is involved.
ALTER TABLE users
    ADD COLUMN email TEXT NOT NULL DEFAULT '';

ALTER TABLE users
    ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (280, 'users: email + email_verified for Connect identity / invites');
