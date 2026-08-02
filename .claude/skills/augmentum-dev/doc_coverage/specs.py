"""Concrete coverage specs. Adding a new tracked list = one CoverageSpec.

Each spec is deliberately CLEAN (unambiguous code⟷doc mapping) so the
diagnostics stay high-trust — a noisy check that cries wolf is worse
than no check. Fuzzy correspondences (e.g. thinking-parser families,
where the code key ``glm47`` reads as "GLM-4.7" in prose) are left out
on purpose; add them only with a custom ``match`` that models the alias.
"""

from __future__ import annotations

import re
from pathlib import Path

from .engine import CoverageSpec

# ---------------------------------------------------------------------------
# code-set derivers (live from the tree — cannot rot)
# ---------------------------------------------------------------------------

# Cross-cutting substrate that doesn't warrant a Subsystem Map row.
_INFRA_SUBSYSTEMS = frozenset({"utils", "state", "proxy", "docs"})


def _subsystems(root: Path) -> set[str]:
    base = root / "augmentum"
    out: set[str] = set()
    if base.is_dir():
        for child in base.iterdir():
            if (child.is_dir() and child.name != "__pycache__"
                    and any(child.rglob("*.py"))):
                out.add(child.name)
    return out


def _modes(root: Path) -> set[str]:
    base = root / "augmentum" / "modes"
    out: set[str] = set()
    if base.is_dir():
        for child in base.iterdir():
            if (child.is_dir() and child.name != "__pycache__"
                    and any(child.glob("*.py"))):
                out.add(child.name)
    return out


def _provider_ids(root: Path) -> set[str]:
    """Provider ids declared in models/provider_profiles.py::PROFILES."""
    path = root / "augmentum" / "models" / "provider_profiles.py"
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r'"([a-z0-9_]+)"\s*:\s*ProviderProfile\(', text))


def _provider_card_stems(root: Path) -> set[str]:
    """Card files present under docs/provider-cards/ (minus meta docs)."""
    base = root / "docs" / "provider-cards"
    meta = {"README", "CORRECTIONS", "_TEMPLATE"}
    out: set[str] = set()
    if base.is_dir():
        for p in base.glob("*.md"):
            if p.stem not in meta:
                out.add(p.stem)
    return out


# ---------------------------------------------------------------------------
# scaffold row builders (paste-ready stubs)
# ---------------------------------------------------------------------------

def _subsystem_row(name: str) -> str:
    title = name.replace("_", " ").title()
    return (f"| **{title}** | `augmentum/{name}/` | TODO_routes | TODO_classes "
            f"| TODO: one-line purpose of {name} |")


def _mode_row(name: str) -> str:
    title = name.replace("_", " ").title()
    handler = "".join(w.title() for w in name.split("_")) + "Handler"
    return (f"| {title} | `augmentum/modes/{name}/` | `{handler}` "
            f"| TODO: one-line purpose |")


def _provider_card_stub(pid: str) -> str:
    return (f"MISSING provider-card: create docs/provider-cards/{pid}.md "
            f"(provider id `{pid}` is in provider_profiles.py::PROFILES) "
            f"— copy docs/provider-cards/deepseek.md as the exemplar.")


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

SPECS: list[CoverageSpec] = [
    CoverageSpec(
        name="subsystems",
        describe="augmentum/ packages that have a Subsystem Map row in SKILL.md",
        code_set=_subsystems,
        fix_location="SKILL.md -> ## Subsystem Map (add a row)",
        exempt=_INFRA_SUBSYSTEMS,
        doc_rel=".claude/skills/augmentum-dev/SKILL.md",
        start_marker="## Subsystem Map",
        end_marker="**Finding a route fast**",
        scaffold=_subsystem_row,
    ),
    CoverageSpec(
        name="modes",
        describe="dispatch modes under augmentum/modes/ with a Handler Pattern row",
        code_set=_modes,
        fix_location="SKILL.md -> ## Handler Pattern (Modes) (add a row)",
        doc_rel=".claude/skills/augmentum-dev/SKILL.md",
        start_marker="## Handler Pattern (Modes)",
        end_marker="## Multi-Tenant Data Isolation",
        scaffold=_mode_row,
    ),
    CoverageSpec(
        name="provider_cards",
        describe="OpenAI-compat providers (provider_profiles.py) with a provider-card",
        code_set=_provider_ids,
        fix_location="docs/provider-cards/ (create <id>.md)",
        # Six thin resellers share aggregators.md (see provider-cards/README.md).
        exempt=frozenset({
            "aimlapi", "electronhub", "chutes", "nanogpt", "pollinations",
            "siliconflow",
        }),
        doc_set=_provider_card_stems,
        scaffold=_provider_card_stub,
    ),
]


def spec_by_name(name: str) -> CoverageSpec | None:
    for spec in SPECS:
        if spec.name == name:
            return spec
    return None
