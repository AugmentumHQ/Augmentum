-- Voice Cast for comic narration — the 5-register-bucket casting model.
--
-- Supersedes the 2-bucket voice_male/voice_female pair (migration 331): a
-- narration is now cast into up to five voices (m_low, m_high, f_low, f_high,
-- narrator), stored as a JSON object here. `voice` stays the narrator/default
-- fallback; the legacy voice_male/voice_female columns remain as a back-compat
-- fallback for rows recorded before this migration (m_low<-voice_male,
-- f_low<-voice_female) and for the 3-bucket UI, so nothing recorded earlier
-- goes silent.
ALTER TABLE comic_narrations ADD COLUMN voice_cast TEXT NOT NULL DEFAULT '{}';
