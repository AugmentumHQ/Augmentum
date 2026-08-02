-- 264_entity_clusters.sql
--
-- Consumption-entity discovery, P0+P1 substrate (spec:
-- docs/superpowers/specs/2026-06-12-consumption-entity-discovery-design.md).
--
-- Interests come in two kinds: TOPICS (searches, pages, discussions —
-- things you research) and ENTITIES (audiobooks, comics, shows, music —
-- things you consume). Treating a consumed title as a topic string is
-- the root cause of the curator pairing an audiobook with an unrelated
-- blog post on one shared token ("life"). Entity clusters carry a
-- structured ref into the catalog instead, and route down the
-- catalog-first recommendation ladder, never into topic polling.

ALTER TABLE interest_clusters ADD COLUMN kind TEXT NOT NULL DEFAULT 'topic';
ALTER TABLE interest_clusters ADD COLUMN entity_ref TEXT NOT NULL DEFAULT '';

-- Backfill: clusters whose signals are majority media_play are
-- consumption entities, not research topics. Their entity_ref stays
-- empty (no file_id retained on old rows' cluster linkage in a
-- recoverable shape) — the next play signal re-resolves and fills it.
UPDATE interest_clusters SET kind = 'entity'
WHERE cluster_id IN (
    SELECT cluster_id FROM interaction_signals
    WHERE cluster_id IS NOT NULL
    GROUP BY cluster_id
    HAVING SUM(CASE WHEN signal_type = 'media_play' THEN 1 ELSE 0 END) * 2
           > COUNT(*)
);

CREATE INDEX IF NOT EXISTS idx_clusters_kind ON interest_clusters(kind);
