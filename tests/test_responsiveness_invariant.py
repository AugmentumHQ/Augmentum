"""Responsiveness invariant — the felt-contract foundation.

A user-ADDRESSED turn must always produce a warm response, regardless of any
interior "economy" state (energy / mana / drives / berries). We do NOT enforce
this with a runtime ``if addressed: respond`` check — the responsive path
already hard-returns on unaddressed turns and consults no economic state today,
so such a check would guard against nothing (theater). Instead we enforce it
STRUCTURALLY: the responsive generation path may never *import* the autonomous /
economic subsystems, so a reply can never become gated on capacity by
construction.

This guard is deliberately laid down BEFORE the economy is wired. The energy /
mana work lands in the AUTONOMOUS path next (``behavior/tick.py`` +
``behavior/activity_selector.py``); this test is the wall that keeps it from
ever leaking onto the path that answers you. Energy shapes what she INITIATES,
never how she MEETS you. If this test ever fails, someone wired capacity into a
reply — that is the drift, caught at its root.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Modules on the RESPONSIVE path — these run when the user addresses her.
_RESPONSIVE_MODULES = [
    "augmentum/companion_runtime/voice.py",
    "augmentum/proxy/voice_routes.py",
]

# Interior "economy" / autonomy subsystems the responsive path must never touch.
_FORBIDDEN_PREFIXES = (
    "augmentum.companion_runtime.energy",
    "augmentum.companion.growth.economy",
    "augmentum.companion_runtime.behavior.activity_selector",
    "augmentum.companion_runtime.behavior.tick",
)


def _repo_root() -> Path:
    # tests/ lives directly under the repo root.
    return Path(__file__).resolve().parents[1]


def _imported_modules(source: str) -> set[str]:
    """Every absolute module name imported by ``source`` (incl. ``from x import y``)."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports (level > 0) — the forbidden set is absolute.
            if node.level != 0 or not node.module:
                continue
            names.add(node.module)
            # Also catch ``from x.behavior import tick`` → ``x.behavior.tick``.
            for alias in node.names:
                names.add(f"{node.module}.{alias.name}")
    return names


@pytest.mark.parametrize("rel_path", _RESPONSIVE_MODULES)
def test_responsive_path_never_imports_economy(rel_path: str) -> None:
    path = _repo_root() / rel_path
    source = path.read_text(encoding="utf-8")
    violations = sorted(
        name
        for name in _imported_modules(source)
        if name.startswith(_FORBIDDEN_PREFIXES)
    )
    assert not violations, (
        f"{rel_path} imports interior-economy modules on the RESPONSIVE path: "
        f"{violations}. A user-addressed reply must never be gated on energy / "
        f"mana / drives. Wire capacity into behavior/tick.py (autonomous) instead."
    )
