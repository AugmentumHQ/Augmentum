-- 189_reasoning_flow_steps_flow_id_index.sql
--
-- list_flows() runs a correlated subquery to count steps per flow:
--
--   SELECT f.*, (SELECT COUNT(*) FROM reasoning_flow_steps s
--                WHERE s.flow_id = f.id) AS _step_count
--     FROM reasoning_flows f ORDER BY f.is_default DESC, f.name ASC
--
-- Without an index on reasoning_flow_steps.flow_id every f.id requires
-- a full table scan of the steps table — O(flows × steps) total. With
-- ~30 flows × hundreds of steps that's a 400–800 ms query per call,
-- and /api/capabilities polls this endpoint every few seconds.
-- The index drops it to ~5 ms.
--
-- IF NOT EXISTS so re-runs / parallel deployments are safe.

CREATE INDEX IF NOT EXISTS idx_reasoning_flow_steps_flow_id
  ON reasoning_flow_steps(flow_id);
