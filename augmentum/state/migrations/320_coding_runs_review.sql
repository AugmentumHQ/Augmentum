-- Coding runs: store the review_turn_id so the Agents window can mount the
-- REAL coder review panel (coder-review.js mountReviewPanel — unified diff +
-- Accept/Reject/Partial) instead of a hand-rolled patch view. Both handles
-- (review_turn_id + broker run_id) already come back in the background-run
-- job result; this just gives them a durable home on the run record.

ALTER TABLE coding_runs ADD COLUMN review_turn_id TEXT NOT NULL DEFAULT '';
