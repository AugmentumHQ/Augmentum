-- 073_discovery_cluster_vec.sql
-- Vector index for interest cluster centroid similarity search

CREATE VIRTUAL TABLE IF NOT EXISTS interest_clusters_vec USING vec0(
    cluster_id TEXT PRIMARY KEY,
    centroid_embedding float[768]
);
