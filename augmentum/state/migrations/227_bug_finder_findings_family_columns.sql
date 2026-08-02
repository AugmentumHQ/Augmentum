-- 227_bug_finder_findings_family_columns.sql
-- Adds cross-family confirmation counts to the per-finding rows.
--
-- A finding with `families_to_confirm = 2` was flagged by detectors
-- from two distinct vendor families (e.g. Claude AND GPT). This is a
-- stronger precision signal than `runs_to_confirm` alone because it
-- breaks the correlated-error pattern Anthropic's bug-finder research
-- identified as the dominant FP source. Adding the column to the
-- normalized table makes "list findings confirmed cross-family" a
-- one-query operation in the UI / analytics.
--
-- Defaults to 0 for older rows — they predate the ensemble; their
-- effective "families_to_confirm" is unknown, so 0 is the safe value.

ALTER TABLE bug_finder_findings ADD COLUMN families_to_confirm INTEGER NOT NULL DEFAULT 0;
ALTER TABLE bug_finder_findings ADD COLUMN total_families INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_bug_finder_findings_user_families
    ON bug_finder_findings(user_id, families_to_confirm DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (227, 'Bug finder findings — cross-family confirmation columns');
