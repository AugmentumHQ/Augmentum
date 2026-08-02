-- Language partner character cards.
--
-- A "language partner" is a regular ui_characters row with two extra
-- columns that mark it as a conversation partner for a specific target
-- language. The narrative pipeline already handles persistence, voice,
-- group chats, and tool calling — partners reuse all of that. These
-- columns just let GET /api/learning/partner?lang=X find the right
-- card without parsing every character's data blob.
--
-- Why not stuff it in the JSON `data` blob?
--   The partner-lookup query runs on every visit to the Talk-With-A-Partner
--   surface and on every chat-session resume that needs to know "is this a
--   language session?". Indexed columns keep that path O(log n); scanning
--   N rows and parsing each blob does not.
--
-- Why two columns instead of one?
--   `is_language_partner` is a boolean for "any partner" filters (the
--   future hub-level list of all my partners across all my languages);
--   `lang_code` answers "which language". Together they form the index
--   used by partner_for_lang().
--
-- One partner per (user, lang_code) is enforced in the partner store
-- via UNIQUE(user_id, lang_code) WHERE is_language_partner = 1 — see
-- the partial index below. Users can rename or restyle their partner,
-- but never end up with two partners for the same language.
--
-- Bundled seeds live in augmentum/learning/partners.py and are
-- materialised lazily on first /api/learning/partner access per user,
-- so a fresh tenant doesn't pay for partner rows in languages they
-- never study.

ALTER TABLE ui_characters ADD COLUMN lang_code TEXT NOT NULL DEFAULT '';
ALTER TABLE ui_characters ADD COLUMN is_language_partner INTEGER NOT NULL DEFAULT 0;

-- Partial unique: at most one partner per (user, language). Plain rows
-- (is_language_partner = 0) are unconstrained — the existing PK on id
-- handles those.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ui_characters_partner_unique
    ON ui_characters(user_id, lang_code)
    WHERE is_language_partner = 1;

-- Lookup index for the "list all my partners" path.
CREATE INDEX IF NOT EXISTS idx_ui_characters_partner_lookup
    ON ui_characters(user_id, is_language_partner);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (171, 'ui_characters — language partner flag + lang_code');
