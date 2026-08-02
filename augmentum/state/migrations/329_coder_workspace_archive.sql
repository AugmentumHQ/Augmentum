-- 329_coder_workspace_archive.sql
-- Soft-delete ("archive") for coder workspaces.
--
-- Deleting a workspace already defaulted to keep_volume=True (the durable
-- Project bare repo carries checkpoints), but it hard-DELETEd the
-- project_checkouts row — throwing away the name, mission/task progression,
-- user_id, and project link while leaving the ~GB /workspace volume orphaned
-- and invisible in the UI (104 volumes vs 11 live rows observed on Matt's box).
--
-- New model: the default delete ARCHIVES — removes the container, keeps the
-- volume, and marks the row archived_at instead of deleting it. The row stays
-- so the archive view can show name + accumulated tasks and a native restore
-- can respawn a container onto the surviving volume. "Completely remove"
-- (keep_volume=False) is the opt-in hard delete that drops row + volume.
--
-- archived_at: NULL = active workspace (the normal list). Non-NULL = archived
-- (epoch seconds), shown only in the archive view.
-- archived_size_bytes: /workspace volume size measured once at archive time
-- (the socket proxy forbids /system/df, so we snapshot it here rather than
-- probing live on every archive-list load).
ALTER TABLE project_checkouts ADD COLUMN archived_at INTEGER;
ALTER TABLE project_checkouts ADD COLUMN archived_size_bytes INTEGER;

INSERT INTO schema_version (version, applied_at) VALUES (329, strftime('%s', 'now'));
