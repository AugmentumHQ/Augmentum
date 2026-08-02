-- 194_companion_journal_content_hash.sql
-- Content-hash dedup for companion_journal — kills repeat noticings.
--
-- Production observation 2026-05-23: same content writes hundreds of
-- times in a single day. The dominant pattern: small utility model
-- falls into an attractor and emits the identical noticing on
-- successive ticks. The existing duplicate guard in
-- ``activity_selector._generate_journal_content`` only compares
-- against the SINGLE most recent non-placeholder entry, so if any
-- other write (mode.changed observation, wake bridge, etc.) interleaves
-- between two identical noticings, the second one passes.
--
-- The structural fix lives below the perform layer: at the
-- ``CompanionMemory.journal()`` write path. We compute a normalized
-- content_hash, look it up in a recent window (configurable, default
-- 4 hours), and on hit bump the existing row's repetition_count
-- instead of inserting a duplicate.
--
-- ``content_hash`` is a sha256 of the normalized content (lowercased,
-- whitespace-collapsed, trimmed to first 200 chars). Same fingerprint
-- the perform layer uses for its own near-dup check; consolidated
-- here so it works regardless of write source.

ALTER TABLE companion_journal
    ADD COLUMN content_hash TEXT NOT NULL DEFAULT '';

-- Partial index on (companion_id, content_hash) for non-empty hashes
-- only. New writes populate the hash; legacy rows have empty strings
-- and never participate in dedup (which is fine — legacy spam will be
-- cleaned in a separate suppression pass).
CREATE INDEX IF NOT EXISTS idx_cj_content_hash_recent
    ON companion_journal(companion_id, content_hash, created_at DESC)
    WHERE content_hash != '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (194, 'companion_journal.content_hash + partial index for dedup');
