#!/usr/bin/env python3
"""Scaffold all 4 layers of boilerplate for a new Augmentum setting.

Usage:
    python scaffold_setting.py setting_name type default_value

Examples:
    python scaffold_setting.py my_feature_enabled bool false
    python scaffold_setting.py my_model_override str ""
    python scaffold_setting.py my_timeout float 30.0
    python scaffold_setting.py my_max_retries int 3

Prints copy-pasteable snippets for config.py, config_routes.py, server.py,
and settings.js.  Does NOT modify any files (--dry-run is the only mode).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------

def _find_root() -> Path:
    """Walk up from this script to find the project root (has augmentum/ and ui/)."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: Cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)

ROOT = _find_root()

# ---------------------------------------------------------------------------
# Terminal colors (skip on Windows without ANSI support)
# ---------------------------------------------------------------------------

_COLOR = os.environ.get("TERM") or os.name != "nt"

def _red(s: str) -> str:    return f"\033[91m{s}\033[0m" if _COLOR else s
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m" if _COLOR else s
def _green(s: str) -> str:  return f"\033[92m{s}\033[0m" if _COLOR else s
def _cyan(s: str) -> str:   return f"\033[96m{s}\033[0m" if _COLOR else s
def _bold(s: str) -> str:   return f"\033[1m{s}\033[0m" if _COLOR else s
def _dim(s: str) -> str:    return f"\033[2m{s}\033[0m" if _COLOR else s

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_TYPES = {"bool", "str", "int", "float"}


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _parse_default(typ: str, raw: str) -> str:
    """Parse and normalise the default value string for the given type."""
    if typ == "bool":
        if raw.lower() in ("true", "1", "yes"):
            return "True"
        return "False"
    if typ == "str":
        # Strip surrounding quotes if present
        stripped = raw.strip("\"'")
        return f'"{stripped}"'
    if typ == "int":
        return str(int(raw))
    if typ == "float":
        return str(float(raw))
    return raw


def _js_default(typ: str, raw: str) -> str:
    """Return the JS representation of the default value."""
    if typ == "bool":
        return "true" if raw.lower() in ("true", "1", "yes") else "false"
    if typ == "str":
        stripped = raw.strip("\"'")
        return f'"{stripped}"'
    if typ == "int":
        return str(int(raw))
    if typ == "float":
        return str(float(raw))
    return raw


# ---------------------------------------------------------------------------
# Snippet generators
# ---------------------------------------------------------------------------

def _layer1_config(name: str, typ: str, default_py: str) -> str:
    """config.py field declaration."""
    return f"    {name}: {typ} = {default_py}"


def _layer2_config_routes(name: str, typ: str) -> str:
    """config_routes.py validation entry."""
    if typ == "bool":
        return f'    "{name}": (bool, 0, 1),'
    if typ == "int":
        return f'    "{name}": (int, 0, 1000),'
    if typ == "float":
        return f'    "{name}": (float, 0.0, 1.0),'
    # str
    return f'    "{name}": 256,'


def _layer2_dict_name(typ: str) -> str:
    """Which dict in config_routes.py this belongs in."""
    if typ == "str":
        return "_STRING_SETTINGS"
    return "_TOOL_SETTINGS"


def _layer3_server(name: str, typ: str) -> str:
    """server.py _SETTINGS_RESTORE_MAP entry."""
    if typ == "bool":
        return f'    "{name}": _parse_bool,'
    return f'    "{name}": {typ},'


def _layer4_js(name: str, typ: str, default_js: str) -> str:
    """settings.js snippets (DEFAULTS, load, sync)."""
    camel = _snake_to_camel(name)
    defaults_line = f"    {camel}: {default_js},"
    load_line = f"    settings.{camel} = data.{name} ?? DEFAULTS.{camel};"
    sync_line = f"        {name}: settings.{camel},"
    return defaults_line, load_line, sync_line


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) < 4 or "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: scaffold_setting.py <setting_name> <type> <default_value>")
        print()
        print("  setting_name   snake_case name (e.g. my_feature_enabled)")
        print("  type           bool | str | int | float")
        print("  default_value  default value (e.g. false, \"\", 30.0, 3)")
        print()
        print("Examples:")
        print('  python scaffold_setting.py my_feature_enabled bool false')
        print('  python scaffold_setting.py my_model_override str ""')
        print('  python scaffold_setting.py my_timeout float 30.0')
        print('  python scaffold_setting.py my_max_retries int 3')
        return 1

    name = sys.argv[1]
    typ = sys.argv[2].lower()
    raw_default = sys.argv[3]

    # Validate name
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        print(_red(f"ERROR: Setting name must be snake_case: '{name}'"), file=sys.stderr)
        return 1

    # Validate type
    if typ not in _VALID_TYPES:
        print(_red(f"ERROR: Type must be one of {sorted(_VALID_TYPES)}, got '{typ}'"), file=sys.stderr)
        return 1

    # Parse defaults
    try:
        default_py = _parse_default(typ, raw_default)
        default_js = _js_default(typ, raw_default)
    except (ValueError, TypeError) as e:
        print(_red(f"ERROR: Invalid default value '{raw_default}' for type {typ}: {e}"), file=sys.stderr)
        return 1

    camel = _snake_to_camel(name)

    # Header
    print()
    print(_bold("  Augmentum Setting Scaffold"))
    print(_bold("  " + "=" * 40))
    print()
    print(f"  Setting:  {_cyan(name)}")
    print(f"  JS key:   {_cyan(camel)}")
    print(f"  Type:     {typ}")
    print(f"  Default:  {default_py} (Python) / {default_js} (JS)")
    print()
    print(_yellow("  DRY RUN — no files modified. Copy-paste snippets below."))
    print()

    # Layer 1
    print(_bold("  Layer 1 — config.py"))
    print(_dim(f"  File: augmentum/config.py  (inside class Settings)"))
    print()
    print(_green(_layer1_config(name, typ, default_py)))
    print()

    # Layer 2
    dict_name = _layer2_dict_name(typ)
    print(_bold(f"  Layer 2 — config_routes.py"))
    print(_dim(f"  File: augmentum/proxy/config_routes.py  (inside {dict_name})"))
    print()
    print(_green(_layer2_config_routes(name, typ)))
    if typ == "int":
        print(_dim("  # Adjust min/max bounds (0, 1000) as needed"))
    elif typ == "float":
        print(_dim("  # Adjust min/max bounds (0.0, 1.0) as needed"))
    elif typ == "str":
        print(_dim("  # Adjust max length (256) as needed"))
    print()

    # Layer 3
    print(_bold("  Layer 3 — server.py"))
    print(_dim(f"  File: augmentum/proxy/server.py  (inside _SETTINGS_RESTORE_MAP)"))
    print()
    print(_green(_layer3_server(name, typ)))
    print()

    # Layer 4
    defaults_line, load_line, sync_line = _layer4_js(name, typ, default_js)
    print(_bold("  Layer 4 — settings.js"))
    print(_dim(f"  File: ui/scripts/settings.js"))
    print()
    print(_dim("  a) DEFAULTS object:"))
    print(_green(defaults_line))
    print()
    print(_dim("  b) loadToolSettingsFromBackend function:"))
    print(_green(load_line))
    print()
    print(_dim("  c) syncToolSettingsToBackend body:"))
    print(_green(sync_line))
    print()

    # Reminder
    print("  " + "-" * 40)
    print(_yellow("  After inserting, run validate_wiring.py to verify all 4 layers are connected."))
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
