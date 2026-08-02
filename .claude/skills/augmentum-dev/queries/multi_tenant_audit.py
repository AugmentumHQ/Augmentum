"""Multi-tenant audit query.

Per CLAUDE.md's data-isolation invariant, every route handler that
touches user-scoped data MUST either (a) call ``_user_id(request)``
to extract the caller's identity, or (b) accept ``user_id`` as a
parameter and pass it to downstream stores.

This query finds handlers that:
  * live in a subsystem with at least one user-scoped table, AND
  * have ``handler_signatures.accepts_user_id = 0``
    AND ``handler_signatures.passes_user_id = 0``

False positives are real — some routes legitimately don't touch
user data even within a user-scoped subsystem (status endpoints,
public catalog, etc.). The query surfaces a candidate list; humans
adjudicate. Add suppressions to a future ``multi_tenant_exceptions.json``
once the noise level is known.

The list of "subsystems with user-scoped tables" is derived from
``tables.user_scoped`` joined back to filename heuristics (table
``narrative_archive`` → narrative subsystem, ``coder_sessions`` →
coder, etc. via prefix match). This avoids hand-curating a list.
"""

from __future__ import annotations

NAME = "multi_tenant_audit"
DESCRIPTION = (
    "Routes in user-scoped subsystems whose handler doesn't appear "
    "to wire user_id (no _user_id() call, no user_id= kwarg). "
    "Candidate list — review for cross-tenant leak risk."
)

# Subsystems are flagged as user-scoped when at least one user-scoped
# table's name starts with the subsystem's prefix. The matching is
# loose on purpose — narrative_archive, narrative_branches all count
# narrative as user-scoped. Built as a CTE rather than a hand-coded
# list so it auto-extends when new user-scoped subsystems land.
QUERY = """
WITH scoped_subsystems AS (
    SELECT DISTINCT
        substr(t.name, 1, instr(t.name || '_', '_') - 1) AS subsystem
    FROM tables t
    WHERE t.user_scoped = 1
      AND t.name LIKE '%_%'
)
SELECT
    e.method,
    e.path_template,
    e.handler_name,
    f.path AS handler_file,
    e.handler_line,
    f.subsystem AS subsystem
FROM endpoints e
JOIN files f ON f.id = e.handler_file_id
JOIN handler_signatures hs ON hs.endpoint_id = e.id
WHERE hs.accepts_user_id = 0
  AND hs.passes_user_id = 0
  AND f.subsystem IN (SELECT subsystem FROM scoped_subsystems)
  -- Skip OPTIONS (CORS preflight, not a data path).
  AND e.method != 'OPTIONS'
ORDER BY f.subsystem, e.path_template
"""

DIAGNOSE = """
WITH scoped_subsystems AS (
    SELECT DISTINCT
        substr(t.name, 1, instr(t.name || '_', '_') - 1) AS subsystem
    FROM tables t
    WHERE t.user_scoped = 1
      AND t.name LIKE '%_%'
)
SELECT
    f.subsystem                         AS subsystem,
    COUNT(*)                            AS suspect_count,
    GROUP_CONCAT(e.method || ' ' || e.path_template, ' | ') AS sample_paths
FROM endpoints e
JOIN files f ON f.id = e.handler_file_id
JOIN handler_signatures hs ON hs.endpoint_id = e.id
WHERE hs.accepts_user_id = 0
  AND hs.passes_user_id = 0
  AND f.subsystem IN (SELECT subsystem FROM scoped_subsystems)
  AND e.method != 'OPTIONS'
GROUP BY f.subsystem
ORDER BY suspect_count DESC
LIMIT 12
"""


def severity(count: int) -> str:
    if count > 50:
        return "warn"  # high — likely real data-isolation gaps
    return "ok"
