"""Orphaned-endpoints query.

An endpoint is *orphaned* when no JS call references its
(method, path_template). The strict join misses some real wiring
that the legacy ``dead_code.py`` scanner catches (dynamic paths,
indirect helpers, WebSocket endpoints), so this query is intentionally
ADDITIVE — it surfaces a subset that can be high-signal even before
the smarter resolver is built. The diagnosis layer is what makes the
findings actionable.

OPTIONS endpoints are excluded — those are CORS preflights, never
called from JS directly. Static-asset routes (/static/*) and
docs routes (/docs, /redoc, /openapi.json) are excluded too — they
exist for browser consumption, not fetch().
"""

from __future__ import annotations

NAME = "orphaned_endpoints"
DESCRIPTION = "Endpoints with no matching JS fetch/WebSocket call (strict path/method join)."

# ``EXCLUDE_PATHS`` is encoded into the WHERE clause as repeated NOT
# LIKE predicates rather than a NOT IN list — endpoints that *prefix*
# match (e.g. ``/static/foo/bar``) shouldn't fall out of the filter
# just because the exact path differs.
QUERY = """
SELECT
    e.method,
    e.path_template,
    f.path AS handler_file,
    e.handler_line,
    e.handler_name,
    f.subsystem
FROM endpoints e
JOIN files f ON f.id = e.handler_file_id
LEFT JOIN js_calls j
       ON j.method = e.method
      AND j.path_template = e.path_template
WHERE j.id IS NULL
  AND e.method != 'OPTIONS'
  AND e.path_template NOT LIKE '/static/%'
  AND e.path_template NOT LIKE '/docs%'
  AND e.path_template != '/openapi.json'
  AND e.path_template != '/redoc'
ORDER BY f.subsystem, e.path_template
"""

# Diagnosis: cluster by subsystem and recency. ``new_in_7d`` = handler
# file's mtime is within the last 7 days, suggesting the orphan is
# fresh debt rather than ancient.
DIAGNOSE = """
SELECT
    COALESCE(f.subsystem, '<root>')          AS subsystem,
    COUNT(*)                                  AS orphan_count,
    SUM(CASE WHEN f.mtime > strftime('%s','now') - 7 * 86400
             THEN 1 ELSE 0 END)               AS new_in_7d,
    GROUP_CONCAT(e.path_template, ' | ')      AS sample_paths
FROM endpoints e
JOIN files f ON f.id = e.handler_file_id
LEFT JOIN js_calls j
       ON j.method = e.method
      AND j.path_template = e.path_template
WHERE j.id IS NULL
  AND e.method != 'OPTIONS'
  AND e.path_template NOT LIKE '/static/%'
  AND e.path_template NOT LIKE '/docs%'
  AND e.path_template != '/openapi.json'
  AND e.path_template != '/redoc'
GROUP BY COALESCE(f.subsystem, '<root>')
HAVING orphan_count > 0
ORDER BY new_in_7d DESC, orphan_count DESC
LIMIT 12
"""


def severity(count: int) -> str:
    if count > 100:
        return "warn"
    return "ok"
