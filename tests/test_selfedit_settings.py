"""Self-edit settings defaults — the safety-critical invariants.

The master switch MUST default OFF and the autonomy posture MUST default to the
safest (propose-only), so a fresh install never self-edits or auto-promotes
without an explicit opt-in. Also verifies the settings are wired across the
server-side layers (config + tool/string registries + restore map).
"""

from __future__ import annotations

from augmentum.config import Settings
from augmentum.proxy import config_routes, server


def test_master_switch_defaults_off():
    s = Settings()
    assert s.selfedit_enabled is False  # never self-edits without an explicit opt-in


def test_autonomy_defaults_to_safest():
    s = Settings()
    assert s.selfedit_autonomy_level == "propose"  # never auto-promotes by default


def test_settings_wired_in_registries():
    # Layer 2: validation registries
    assert "selfedit_enabled" in config_routes._TOOL_SETTINGS
    assert "selfedit_autonomy_level" in config_routes._STRING_SETTINGS
    # Layer 3: restore-on-boot map
    assert "selfedit_enabled" in server._SETTINGS_RESTORE_MAP
    assert "selfedit_autonomy_level" in server._SETTINGS_RESTORE_MAP
