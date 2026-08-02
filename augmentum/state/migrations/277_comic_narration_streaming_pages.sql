-- 277_comic_narration_streaming_pages.sql
-- Per-page streaming for comic narration (slice 1 of the latency rework).
-- The synth job now emits ONE audio artifact PER PAGE as each page finishes,
-- so the player can start on page 1 in ~20s instead of waiting for the whole
-- chapter. `pages` accumulates those per-page entries:
--   [{ "page": int, "artifact_id": str, "duration_ms": int,
--      "lines": [{order,kind,text,bbox,audio_start_ms,audio_end_ms}] }]
-- audio_start/end_ms are RELATIVE TO THAT PAGE's audio (each page is its own
-- clock). The legacy whole-chapter `narration_artifact_id` + `timeline` columns
-- stay for back-compat but are unused by the streaming path.

ALTER TABLE comic_narrations ADD COLUMN pages TEXT NOT NULL DEFAULT '[]';
