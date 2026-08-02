-- 324_bios_store_first.sql
-- Make the BIOS vault store-first / verify-second, matching how every
-- mature emulation frontend works.
--
-- The bug this fixes (observed live 2026-07-25, augmentum container):
--
--   19:07:48  bulk_import_complete  bios=0 imported=0 junk=2  unknown=38
--   18:42:10  bulk_import_complete  bios=0 imported=0 junk=32 unknown=0
--
-- 38 BIOS files uploaded, zero installed. The upload fired, returned
-- 200, and threw everything away. Cause: install was GATED on a
-- hand-maintained 67-entry catalog of which only 15 rows carried a
-- SHA1, and the non-hash path demanded an EXACT canonical filename
-- plus an EXACT byte size. `scph5501.bin` at 524287 bytes -- one byte
-- off a redump -- was discarded with everything else.
--
-- RetroArch, EmuDeck and ES-DE all do the opposite: the BIOS
-- directory is a plain folder, files are copied in unconditionally,
-- and hash verification is a separate advisory CHECKER. They
-- converged there because real-world BIOS sets are full of regional
-- revisions, community renames, and firmware with no canonical hash.
-- Identification-as-gate rejects most of what users actually own.
--
-- So identification stops deciding admission and starts producing a
-- LABEL. Three new columns carry it:
--
--   verify_status  'verified'   -- matched a known hash (sha1/md5/crc32)
--                  'named'      -- matched a known filename + size
--                  'unverified' -- stored on the user's say-so
--
--   md5, crc32     the other two digests libretro's System.dat
--                  publishes. We record all three so a file installed
--                  today still verifies after a database refresh adds
--                  its hash upstream -- no re-upload needed.
--
-- `matched_by` gains 'md5' / 'crc32' / 'user_asserted' alongside the
-- existing 'sha1' / 'name_size' / 'manual'. It stays a free-text
-- column (no CHECK constraint) precisely so widening it again later
-- is not another migration.
--
-- Existing rows: everything already in the table got there by passing
-- the old strict gate, so it is at least name+size-correct. We
-- backfill 'verified' where a sha1 was recorded and matched (the old
-- matched_by='sha1' rows) and 'named' otherwise. Nothing is
-- downgraded and no row is deleted.
--
-- The UNIQUE(user_id, system_id, canonical_filename) constraint is
-- deliberately KEPT. It still means "one file per slot per system per
-- user" -- store-first widens what may occupy a slot, it does not
-- allow two files to claim the same one. Unrecognised files occupy a
-- slot named after the file the user actually dropped.

ALTER TABLE user_bios_files ADD COLUMN verify_status TEXT NOT NULL DEFAULT 'unverified';
ALTER TABLE user_bios_files ADD COLUMN md5 TEXT NOT NULL DEFAULT '';
ALTER TABLE user_bios_files ADD COLUMN crc32 TEXT NOT NULL DEFAULT '';

-- Backfill from the provenance the old gate already proved.
UPDATE user_bios_files
   SET verify_status = 'verified'
 WHERE matched_by = 'sha1' AND sha1 != '';

UPDATE user_bios_files
   SET verify_status = 'named'
 WHERE matched_by IN ('name_size', 'manual');

-- The vault renders "N verified / M stored" per system on every open,
-- and the launch path checks required-slot presence per system.
CREATE INDEX IF NOT EXISTS idx_user_bios_files_verify
    ON user_bios_files(user_id, system_id, verify_status);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (324, 'user_bios_files verify_status/md5/crc32 — store-first BIOS vault');
