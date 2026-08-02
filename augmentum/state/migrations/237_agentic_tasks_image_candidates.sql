-- 237_agentic_tasks_image_candidates.sql
-- Per-slide image candidate pool + user picks for the agentic Presentation flow.
--
-- image_candidates: JSON keyed by slide index ->
--   [{candidate_id, query, description, embed_url, thumb_url, source, title}, ...]
-- slide_image_picks: JSON keyed by slide index ->
--   {primary: candidate_id, additional: [candidate_id, ...]}

ALTER TABLE agentic_tasks ADD COLUMN image_candidates TEXT NOT NULL DEFAULT '{}';
ALTER TABLE agentic_tasks ADD COLUMN slide_image_picks TEXT NOT NULL DEFAULT '{}';
