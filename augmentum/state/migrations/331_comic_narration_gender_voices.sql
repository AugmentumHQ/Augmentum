-- Per-speaker voices for comic narration.
--
-- The whole-page VLM read now tags each line with the speaker's apparent
-- gender ([M]/[F]/[N] narration). Synthesis swaps voices per line so a male
-- speaker, a female speaker, and narration boxes each get their own voice.
--
-- `voice` (migration 275) stays the NARRATOR / default voice — existing
-- single-voice narrations keep working untouched. These two columns are
-- purely additive: empty means "fall back to `voice`".
ALTER TABLE comic_narrations ADD COLUMN voice_male TEXT NOT NULL DEFAULT '';
ALTER TABLE comic_narrations ADD COLUMN voice_female TEXT NOT NULL DEFAULT '';
