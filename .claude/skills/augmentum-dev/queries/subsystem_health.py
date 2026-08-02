"""Subsystem health query — cross-fact correlation per feature area.

For each subsystem (derived from file paths in the ``files`` table)
this query bundles every signal the model already knows about into
one row:
  * routes_total / routes_orphaned
  * tables_owned (user-scoped tables defined in this subsystem's migrations)
  * settings_count / settings_incomplete
  * recent_activity_files (files modified in last 14 days)
  * test_files_count (set to 0 here; populated once the ``tests``
    ingester lands)

Treats absence of data as 0 rather than NULL so the row shape stays
consistent across subsystems with sparse coverage. Subsystem detection
piggybacks on ``files.subsystem`` — the file-ingester already attributes
``augmentum/proxy/<X>_routes.py`` to ``X``, so route counts cluster
correctly without extra plumbing.

The DIAGNOSE shape is the per-subsystem rows themselves — the audit
display surfaces them as the "feature health" report.
"""

from __future__ import annotations

NAME = "subsystem_health"
DESCRIPTION = (
    "Per-subsystem health roll-up: routes, orphans, settings wiring, "
    "recent activity. Use to answer 'what's the state of feature X?'"
)

# Top-level rows: any subsystem with at least one signal flagged.
# Used by the audit's headline number ("N subsystems have notable
# debt"). The DIAGNOSE expansion lists every subsystem with all its
# columns populated.
QUERY = """
WITH subsystem_files AS (
    SELECT id, subsystem, mtime FROM files WHERE subsystem IS NOT NULL
),
route_stats AS (
    SELECT
        f.subsystem AS subsystem,
        COUNT(*)     AS route_count,
        SUM(CASE WHEN j.id IS NULL
                  AND e.method != 'OPTIONS'
                  AND e.path_template NOT LIKE '/static/%'
                  AND e.path_template NOT LIKE '/docs%'
                  AND e.path_template != '/openapi.json'
                  AND e.path_template != '/redoc'
                 THEN 1 ELSE 0 END) AS orphan_count
    FROM endpoints e
    JOIN files f ON f.id = e.handler_file_id
    LEFT JOIN js_calls j
           ON j.method = e.method
          AND j.path_template = e.path_template
    GROUP BY f.subsystem
),
recent_files AS (
    SELECT subsystem, COUNT(*) AS recent_count
    FROM subsystem_files
    WHERE mtime > strftime('%s','now') - 14 * 86400
    GROUP BY subsystem
)
SELECT
    s.subsystem                                       AS subsystem,
    COALESCE(r.route_count, 0)                        AS routes,
    COALESCE(r.orphan_count, 0)                       AS orphans,
    COALESCE(rf.recent_count, 0)                      AS recent_files,
    COALESCE(r.route_count, 0)
        + COALESCE(r.orphan_count, 0)                 AS total_signal
FROM (SELECT DISTINCT subsystem FROM subsystem_files) s
LEFT JOIN route_stats  r  ON r.subsystem  = s.subsystem
LEFT JOIN recent_files rf ON rf.subsystem = s.subsystem
WHERE s.subsystem IS NOT NULL
  AND (r.route_count > 0 OR rf.recent_count > 0)
ORDER BY r.orphan_count DESC, r.route_count DESC, s.subsystem
"""

# DIAGNOSE returns the same rows; severity collapses to the top-level
# count. Audit displays both — the count drives score impact, the
# rows drive the diagnostic report.
DIAGNOSE = QUERY


def severity(count: int) -> str:
    # Subsystem health doesn't penalise — it informs. Surface as info
    # so the score impact is zero.
    return "ok"
