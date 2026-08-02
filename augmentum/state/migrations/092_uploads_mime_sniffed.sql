-- Server-detected MIME stored alongside the client-claimed value.
--
-- Client `Content-Type` is user-controlled and easily forged (a malicious
-- upload can claim "application/pdf" while shipping EXE bytes).  We now
-- sniff magic bytes on save; the sniffed value goes here, the original
-- client claim stays in `mime_type` for backwards compatibility and for
-- audit / mismatch reporting.
--
-- Empty default for existing rows — they predate sniffing; the enrichment
-- worker will backfill on next access.

ALTER TABLE uploads ADD COLUMN mime_sniffed TEXT NOT NULL DEFAULT '';
