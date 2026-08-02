"""Untested-routes query.

A route file is *untested* when no entry in ``test_files.target_modules``
references its path. The signal is conservative: a single matching
target is enough — multiple test files counting the same target only
strengthens (and we don't reward redundancy).

Stale-route exclusion: route files modified in the last 14 days are
elevated as fresh debt; older untested route files are still surfaced
but tagged ``stale`` in the diagnose output.

Excludes ``server.py`` itself (registers everything; tested
indirectly by every TestClient run) and ``__init__.py`` (rarely
warrants its own test file).
"""

from __future__ import annotations

NAME = "untested_routes"
DESCRIPTION = (
    "Route files (augmentum/proxy/*_routes.py) with no test file "
    "claiming them as a target."
)

QUERY = """
SELECT
    f.path                               AS route_file,
    f.subsystem                          AS subsystem,
    f.mtime                              AS mtime,
    CASE WHEN f.mtime > strftime('%s','now') - 14 * 86400
         THEN 1 ELSE 0 END               AS recent
FROM files f
WHERE f.path LIKE 'augmentum/proxy/%_routes.py'
  AND f.path != 'augmentum/proxy/server.py'
  AND NOT EXISTS (
      SELECT 1 FROM test_files tf
      WHERE tf.target_modules LIKE '%' || f.path || '%'
  )
ORDER BY recent DESC, f.path
"""

DIAGNOSE = """
SELECT
    CASE WHEN f.mtime > strftime('%s','now') - 14 * 86400
         THEN 'recent (added in last 14d)'
         ELSE 'stale (older — likely intentional or long-overdue)'
    END                                  AS bucket,
    COUNT(*)                             AS untested_count,
    GROUP_CONCAT(f.subsystem, ', ')      AS sample_subsystems
FROM files f
WHERE f.path LIKE 'augmentum/proxy/%_routes.py'
  AND f.path != 'augmentum/proxy/server.py'
  AND NOT EXISTS (
      SELECT 1 FROM test_files tf
      WHERE tf.target_modules LIKE '%' || f.path || '%'
  )
GROUP BY bucket
ORDER BY bucket DESC
"""


def severity(count: int) -> str:
    if count > 5:
        return "warn"
    return "ok"
