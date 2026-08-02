-- 308: Narrative background-model choices become PER-USER preferences.
--
-- narrative_memory_model / narrative_extraction_model /
-- narrative_auto_bg_distiller_model / narrative_auto_bg_image_model were
-- install-wide (app_settings): one user picking an API or fabric-peer
-- model silently changed every other user's narrative background calls.
-- Each user chooses local vs API themselves (Matt, 2026-07-02).
--
-- Behavior-preserving migration: copy the current install-wide value to
-- every existing user, then clear the global row so the config default
-- ('' = the session's chat model) applies to users created later.

INSERT INTO user_settings (user_id, key, value, updated_at)
SELECT u.id, s.key, s.value, datetime('now')
FROM users u
JOIN app_settings s ON s.key IN (
    'narrative_memory_model',
    'narrative_extraction_model',
    'narrative_auto_bg_distiller_model',
    'narrative_auto_bg_image_model'
)
WHERE s.value IS NOT NULL AND s.value != ''
ON CONFLICT(user_id, key) DO NOTHING;

DELETE FROM app_settings WHERE key IN (
    'narrative_memory_model',
    'narrative_extraction_model',
    'narrative_auto_bg_distiller_model',
    'narrative_auto_bg_image_model'
);
