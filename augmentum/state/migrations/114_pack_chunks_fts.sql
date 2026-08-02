-- 114_pack_chunks_fts.sql: Version marker for knowledge-pack FTS5 indexing.
--
-- The chunks_fts virtual table lives INSIDE each .augpack file, not in the
-- main DB — knowledge packs are standalone SQLite databases that travel
-- between machines. New packs get the FTS table at import time
-- (augmentum/knowledge/converter.py and importer.py). Pre-existing packs
-- get a one-time on-load rebuild in PackManager.scan() — see
-- ``_ensure_pack_fts`` in augmentum/knowledge/packs.py.
--
-- This marker exists so operators can grep schema_version and know whether
-- the runtime they're on understands the FTS-enabled pack format.

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (114, 'knowledge-pack FTS5 indexing (chunks_fts inside .augpack files)');
