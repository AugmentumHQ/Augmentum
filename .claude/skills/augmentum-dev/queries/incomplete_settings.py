"""Incomplete-settings query.

The 4-layer wiring contract (config.py → config_routes.py → server.py
restore-map → settings.js) is the most common source of silent
setting bugs: a setting added to 3 of 4 layers SAVES from the UI,
APPEARS to persist, but does not RESTORE on server restart — or
vice versa. The "missed one layer" case is the high-signal subset.

Severity tiers:
  * 3 of 4 layers wired                    → likely-bug, surfaced as offenders
  * 2 of 4 layers wired                    → likely-internal, not flagged
  * config_py only (1 layer)              → server-internal, not flagged
  * present in JS only                     → likely-orphan, not flagged here
                                              (separate query in a future phase)

The "missed one layer" rule is specific: surface only settings that
are present in 3 layers and absent in the 4th — the symptom of an
incomplete wiring chore.
"""

from __future__ import annotations

NAME = "incomplete_settings"
DESCRIPTION = (
    "Settings wired in 3 of 4 layers (likely-incomplete wiring "
    "— added to most, missed one)."
)

QUERY = """
SELECT
    name_snake,
    name_camel,
    type,
    in_config_py,
    in_config_routes_py,
    in_server_restore,
    in_js_defaults,
    (in_config_py + in_config_routes_py
     + in_server_restore + in_js_defaults) AS layers_wired,
    -- Surface which layer is missing for actionable diagnosis.
    CASE
        WHEN in_config_py        = 0 THEN 'config.py'
        WHEN in_config_routes_py = 0 THEN 'config_routes.py'
        WHEN in_server_restore   = 0 THEN '_SETTINGS_RESTORE_MAP'
        WHEN in_js_defaults      = 0 THEN 'settings.js'
    END AS missing_layer
FROM settings
WHERE (in_config_py + in_config_routes_py
       + in_server_restore + in_js_defaults) = 3
ORDER BY missing_layer, name_snake
"""

DIAGNOSE = """
SELECT
    CASE
        WHEN in_config_py        = 0 THEN 'config.py'
        WHEN in_config_routes_py = 0 THEN 'config_routes.py'
        WHEN in_server_restore   = 0 THEN '_SETTINGS_RESTORE_MAP'
        WHEN in_js_defaults      = 0 THEN 'settings.js'
    END                                       AS missing_layer,
    COUNT(*)                                   AS incomplete_count,
    GROUP_CONCAT(name_snake, ' | ')           AS sample_names
FROM settings
WHERE (in_config_py + in_config_routes_py
       + in_server_restore + in_js_defaults) = 3
GROUP BY missing_layer
ORDER BY incomplete_count DESC
"""


def severity(count: int) -> str:
    if count > 30:
        return "warn"
    return "ok"
