-- Comic-narration per-page audio is regenerable playback cache, not a user
-- deliverable — but it was saved as a first-class artifact (registered in
-- the file index), flooding the Files/library surfaces with ~30 entries per
-- narrated chapter. New synths save transient=1 and skip index registration
-- (artifact_storage.save_from_path); this backfills existing rows: mark
-- them transient (listings/library already filter transient=0) and drop
-- their file-index registrations. Playback is unaffected — the player
-- fetches by artifact id.

UPDATE artifacts
   SET transient = 1
 WHERE json_extract(metadata, '$.comic_narration_for') IS NOT NULL
   AND COALESCE(transient, 0) = 0;

DELETE FROM file_index
 WHERE source = 'artifacts'
   AND source_id IN (
       SELECT id FROM artifacts
        WHERE json_extract(metadata, '$.comic_narration_for') IS NOT NULL
   );
