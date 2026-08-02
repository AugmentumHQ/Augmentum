-- Tier 1.7 hardening: usernames are now compared case-insensitively
-- (see _canonical_username() in session_manager.py). Any existing rows
-- stored with mixed case would silently fail login post-rollout because
-- get_password_hash() looks up the casefolded form. Normalize them in
-- place; the username regex (_USERNAME_RE) restricts the column to
-- [a-zA-Z0-9_], so SQLite's ASCII lower() matches Python's casefold().
--
-- If two users somehow have case-distinct usernames (e.g. "Alice" and
-- "alice"), this migration trips the UNIQUE constraint on users.username
-- — that's the right outcome: the admin needs to merge or rename one of
-- the rows manually before login lockout will key consistently.

UPDATE users
   SET username = lower(username)
 WHERE username != lower(username);

UPDATE failed_login_attempts
   SET username = lower(username)
 WHERE username != lower(username);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (111, 'normalize_usernames_to_lowercase');
