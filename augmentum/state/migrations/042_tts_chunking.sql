-- Add TTS chunking mode to audio providers.
-- Controls how text is split before sending to the TTS provider:
--   'sentence' (default) — SentenceBuffer two-tier: clause then sentence boundaries
--   'clause'             — Always use clause-level breaks (~30 chars, commas, etc.)
--   'full'               — Send entire response as one TTS request (no splitting)

ALTER TABLE audio_providers ADD COLUMN tts_chunking TEXT NOT NULL DEFAULT 'sentence';

INSERT OR IGNORE INTO schema_version (version, description) VALUES (42, 'tts_chunking');
