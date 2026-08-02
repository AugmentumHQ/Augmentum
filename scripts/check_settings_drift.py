"""Find any setting key in _TOOL_SETTINGS / _STRING_SETTINGS that's
missing from _SETTINGS_RESTORE_MAP — those have the same bug class as
the companion_auto_summon / local_fabric_icon regressions (saves to DB,
but reverts to the Python default on restart because the restore map
doesn't know how to cast the stored value back).

Run from repo root:

    python scripts/check_settings_drift.py
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
cr = (ROOT / "augmentum/proxy/config_routes.py").read_text(encoding="utf-8")
sv = (ROOT / "augmentum/proxy/server.py").read_text(encoding="utf-8")


def keys_in_block(text: str, marker: str) -> set[str]:
    """Pull dict-literal keys after `<marker>: dict[...] = {` up to the matching `}`."""
    m = re.search(rf"{re.escape(marker)}\s*[:=][^{{]*{{", text)
    if not m:
        return set()
    start = m.end() - 1
    depth = 0
    end = start
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    block = text[start:end]
    return set(re.findall(r'"([a-z][a-z0-9_]+)"\s*:', block))


tool = keys_in_block(cr, "_TOOL_SETTINGS")
string = keys_in_block(cr, "_STRING_SETTINGS")
restore = keys_in_block(sv, "_SETTINGS_RESTORE_MAP")

# Some keys are intentionally NOT persisted (runtime-only / computed).
# Anything still in the gap after this filter is suspect.
INTENTIONAL_RUNTIME_ONLY: set[str] = set()
# Add keys you've confirmed are intentionally runtime-only:
# INTENTIONAL_RUNTIME_ONLY |= {"some_runtime_key", ...}

print(f"_TOOL_SETTINGS keys:        {len(tool)}")
print(f"_STRING_SETTINGS keys:      {len(string)}")
print(f"_SETTINGS_RESTORE_MAP keys: {len(restore)}")
print()

gap = (tool | string) - restore - INTENTIONAL_RUNTIME_ONLY
companion_gap = sorted(k for k in gap if k.startswith("companion"))
other_gap = sorted(k for k in gap if not k.startswith("companion"))

print(f"=== COMPANION gap ({len(companion_gap)}) ===")
for k in companion_gap:
    print(f"  {k}")

print()
print(f"=== OTHER gap ({len(other_gap)}) ===")
for k in other_gap:
    print(f"  {k}")
