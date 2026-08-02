# Memory Store Redesign: Phases 2-4 — store.py Changes

## Migration Summary

| Migration | File | Adds |
|-----------|------|------|
| 049 | `049_provisional_tier.sql` | `provisional_expires_at TEXT`, `evidence TEXT`, partial index on `tier='provisional'` |
| 050 | `050_hebbian_cooccurrence.sql` | `retrieval_count INTEGER`, `last_accessed_at TEXT`, `memory_cooccurrence` table with composite PK and two indexes |
| 051 | `051_reflection_support.sql` | `source_memory_ids TEXT` (JSON array of parent memory IDs for reflection-derived memories) |

**Note on migration 050:** The `source_type` column already exists from `006_memory.sql` and was removed from this migration to avoid a duplicate column error. The existing `access_count`/`last_accessed` columns serve the general tier-promotion system. The new `retrieval_count`/`last_accessed_at` columns serve the Hebbian co-occurrence system specifically — they track retrieval events that feed into the scoring formula, separate from the access counter that drives promotion thresholds.

---

## Phase 2: PROVISIONAL Tier

### 1. Model changes (`models.py`)

Add `PROVISIONAL = "provisional"` to `MemoryTier` enum, after ARCHIVE. Add two optional fields to the `Memory` dataclass:

```
provisional_expires_at: str | None = None
evidence: str = ""
```

Update `_row_to_memory()` in `store.py` to read these from the DB row.

### 2. Excluding PROVISIONAL from `recall()`

**Where:** `_vector_search_scored()` and `_fts_search()`.

Both methods iterate over candidate rows and filter. Add a tier check after fetching the memory:

- In `_vector_search_scored()` (line ~572), after checking `valid_until`, add: skip if `mem.tier == 'provisional'` (or the enum value). This excludes PROVISIONAL memories from vector results before they enter RRF.
- In `_fts_search()` (line ~612), add `AND m.tier != 'provisional'` to the SQL WHERE clause. This is more efficient than post-filtering since FTS5 joins against the memories table anyway.

This ensures PROVISIONAL memories are never returned to the user but remain in the DB for promotion tracking.

### 3. PROVISIONAL tier assignment in `store()` / `_store_inner()`

**Where:** `_store_inner()`, after the INSERT statement (line ~152).

After inserting a new memory, check if `confidence < 0.7`. If so:

1. Compute `provisional_expires_at = datetime.now(UTC) + timedelta(days=7)` as ISO string.
2. Execute `UPDATE memories SET tier = 'provisional', provisional_expires_at = ? WHERE id = ?`.
3. Log at debug level: `memory_stored_as_provisional`.

This keeps the main INSERT clean (default tier = ACTIVE) and applies the downgrade as a follow-up step within the same transaction.

### 4. `cleanup_provisional()` — Expiry deletion

**New async method on `MemoryStore`.**

```
async def cleanup_provisional(self) -> int:
```

Logic:
1. Query: `SELECT id FROM memories WHERE tier = 'provisional' AND provisional_expires_at < datetime('now')`.
2. For each expired memory: delete from `memories_vec` (if vec_enabled), then delete from `memories`.
3. Commit.
4. Return count of deleted memories for logging.

**Scheduling:** Call from two places:
- On every `store()` call, after the insert completes (cheap — the partial index makes the expired check fast).
- From the existing compaction scheduler (if compaction is enabled), piggyback on the hourly cycle.

### 5. `promote_provisional()` — Shadow access tracking

**Where:** Inside `recall()`, after the main retrieval and scoring but *before* the final result filtering.

The key insight: PROVISIONAL memories are excluded from recall results, but we still want to detect when a query *would have* matched them. This requires a separate shadow query:

1. After the main recall completes, run a lightweight vector search against PROVISIONAL memories only:
   `SELECT id FROM memories WHERE tier = 'provisional' AND user_id = ? AND valid_until IS NULL`
   joined with the vec table, limited to top-5, with a minimum similarity threshold (e.g., distance < 0.4).
2. For each matching PROVISIONAL memory, increment `access_count` (reusing the existing column).
3. If `access_count >= 3`, promote:
   - `UPDATE memories SET tier = 'active', provisional_expires_at = NULL WHERE id = ?`
   - Log: `provisional_memory_promoted`.

**Performance note:** This shadow query runs on every `recall()` call. Mitigations:
- Only runs if vec_enabled (no fallback FTS shadow search — not worth the complexity).
- Limited to 5 results with a tight distance threshold.
- The partial index on `tier = 'provisional'` keeps the scan cheap.
- Can be skipped entirely if `SELECT COUNT(*) FROM memories WHERE tier = 'provisional' AND user_id = ?` returns 0 (cached in-memory with a short TTL).

---

## Phase 3: Hebbian Co-occurrence

### 6. Hebbian scoring in `recall()` — Modified scoring formula

**Where:** The scoring loop in `recall()` (lines ~248-265).

Current formula: `rrf_score * recency * importance * tier_weight * source_boost`

New formula: `rrf_score * recency_decay * importance * tier_weight * source_boost * hebbian_boost`

The `hebbian_boost` is computed per-memory:
1. Look up `retrieval_count` from the memory (loaded via `_row_to_memory`, new field).
2. Compute: `hebbian_boost = 1.0 + log1p(retrieval_count) * 0.1` (logarithmic scaling, so diminishing returns).
3. Cap at 1.5 to prevent runaway boosting.

This means a memory retrieved 10 times gets ~1.23x boost, 50 times gets ~1.39x, and it asymptotes at 1.5x.

**Also:** After recall returns results, increment `retrieval_count` and `last_accessed_at` for all returned memories:
```sql
UPDATE memories SET retrieval_count = retrieval_count + 1,
    last_accessed_at = datetime('now') WHERE id IN (?, ?, ...)
```

This is done alongside the existing `access_count` update (line ~281-286) in a single UPDATE or a second UPDATE in the same commit.

### 7. `increment_cooccurrence()` — Pair tracking

**New async method on `MemoryStore`.**

```
async def increment_cooccurrence(self, user_id: str, memory_ids: list[str]) -> None:
```

Logic:
1. Filter `memory_ids` to only include ACTIVE or CORE tier memories (query tiers if not already known). PROVISIONAL and ARCHIVE are excluded.
2. Generate all unique pairs `(id_a, id_b)` where `id_a < id_b` (canonical ordering to avoid duplicates).
3. For each pair, upsert:
   ```sql
   INSERT INTO memory_cooccurrence (user_id, id_a, id_b, count, last_updated)
   VALUES (?, ?, ?, 1, datetime('now'))
   ON CONFLICT(user_id, id_a, id_b) DO UPDATE SET
       count = count + 1, last_updated = datetime('now')
   ```
4. Commit.

**Scheduling:** Call at the end of `recall()`, after the access_count update, passing the IDs of all returned memories. Use `asyncio.create_task()` so it doesn't block the recall response (fire-and-forget with error callback logging).

### 8. `associate_expansion()` — Co-occurrence-based recall expansion

**New async method on `MemoryStore`.**

```
async def associate_expansion(
    self, user_id: str, selected_ids: list[str], max_extra: int = 2,
) -> list[Memory]:
```

Logic:
1. For each ID in `selected_ids`, query the top-2 co-occurring memories:
   ```sql
   SELECT id_b AS assoc_id, count FROM memory_cooccurrence
   WHERE user_id = ? AND id_a = ?
   UNION ALL
   SELECT id_a AS assoc_id, count FROM memory_cooccurrence
   WHERE user_id = ? AND id_b = ?
   ORDER BY count DESC LIMIT 2
   ```
2. Collect all candidate associate IDs. Remove any that are already in `selected_ids`.
3. Rank candidates by their co-occurrence count (sum across all selected memories that link to them).
4. Fetch the top `max_extra` candidates as full `Memory` objects.
5. Filter out expired (`valid_until IS NOT NULL`) and PROVISIONAL tier.
6. Return the list (may be empty or shorter than `max_extra`).

**Integration into `recall()`:** After the final scored list is trimmed to `limit`, call `associate_expansion()` with the selected IDs. Append the returned associates to the result list (beyond the original limit). This means `recall()` may return up to `limit + 2` memories. The caller should be aware of this.

**Alternative:** If strict limit compliance is required, reserve 2 slots: retrieve `limit - 2` from scoring, then fill remaining slots with associates. This is a design choice for the implementer.

### 9. `weekly_decay_cooccurrence()` — Background decay

**New async method on `MemoryStore`.**

```
async def weekly_decay_cooccurrence(self) -> int:
```

Logic:
1. Execute: `UPDATE memory_cooccurrence SET count = CAST(count * 0.99 AS INTEGER)`.
   Note: SQLite CAST truncates, so counts decay to 0 over time. Alternatively, use `ROUND(count * 0.99)` to round instead.
2. Clean up zero-count rows: `DELETE FROM memory_cooccurrence WHERE count <= 0`.
3. Commit.
4. Return number of rows decayed + deleted for logging.

**Scheduling:** Register as a periodic task in the server lifespan (same pattern as `kg_decay_interval`). Default interval: 7 days (168 hours). Config setting: `memory_cooccurrence_decay_interval_hours: float = 168.0`.

---

## Phase 4: Reflection Support

Migration 051 adds `source_memory_ids TEXT DEFAULT '[]'` — a JSON array of memory IDs that a reflection-derived memory was synthesized from. No store.py changes needed for Phase 4 migrations; the column is used by the reflection system (separate module) which calls `store()` and then updates the field.

---

## Config Settings to Add (`config.py`)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `memory_provisional_ttl_days` | float | 7.0 | TTL for provisional memories before auto-deletion |
| `memory_provisional_promote_threshold` | int | 3 | Shadow access count needed to promote provisional to active |
| `memory_provisional_confidence_threshold` | float | 0.7 | Confidence below this stores as provisional |
| `memory_hebbian_boost_cap` | float | 1.5 | Maximum Hebbian boost multiplier |
| `memory_cooccurrence_decay_interval_hours` | float | 168.0 | How often to run co-occurrence decay |
| `memory_associate_expansion_enabled` | bool | True | Whether recall() appends co-occurrence associates |
| `memory_associate_expansion_max` | int | 2 | Max extra memories from associate expansion |

---

## Data Flow Summary

```
store() ─► confidence < 0.7? ─► tier = PROVISIONAL, set TTL
                                    │
recall() ─► vector + FTS (excludes PROVISIONAL) ─► RRF merge
         │                                            │
         │  shadow query (PROVISIONAL only) ──────► access_count++
         │  access_count >= 3? ─► promote to ACTIVE   │
         │                                            ▼
         │                              score = rrf * recency * importance
         │                                      * tier_weight * source_boost
         │                                      * hebbian_boost(retrieval_count)
         │                                            │
         │                                            ▼
         │                              top-K selection
         │                                            │
         │                              associate_expansion(top-K ids)
         │                              ─► append up to +2 co-occurring memories
         │                                            │
         │                                            ▼
         │                              return results
         │                                            │
         └── fire-and-forget: increment_cooccurrence(result_ids)
                              increment retrieval_count + last_accessed_at

cleanup_provisional() ─► delete expired PROVISIONAL (called on store + compaction cycle)
weekly_decay_cooccurrence() ─► count *= 0.99, delete zeros (weekly background task)
```
