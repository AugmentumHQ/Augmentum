-- 174_ui_sessions_group_id.sql
--
-- Promote ui_sessions.data.groupId to a real column so the meta-listing
-- path (`GET /api/chats/?meta=1`) doesn't have to drag the full session
-- blob across the SQLite worker thread just to peek at the group id.
--
-- Background: list_chats was observed taking ~900ms on the meta path
-- because it SELECTed `data` and did a substring scan + JSON parse to
-- find groupId. The blob is the entire chat tree — hundreds of KB
-- per row at scale.
--
-- After this migration:
--   * meta path selects `group_id` directly, never touches `data`
--   * write paths populate the column on every upsert
--   * existing rows are backfilled below

ALTER TABLE ui_sessions ADD COLUMN group_id TEXT;

-- Backfill from existing data blobs. json_extract returns NULL when
-- the key is absent, which is the desired value for non-group sessions.
UPDATE ui_sessions
   SET group_id = json_extract(data, '$.groupId')
 WHERE data LIKE '%"groupId"%';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (174, 'ui_sessions: promote data.groupId to first-class column for cheap meta listing');
