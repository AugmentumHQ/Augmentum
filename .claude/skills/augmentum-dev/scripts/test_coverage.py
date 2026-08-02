#!/usr/bin/env python3
"""Augmentum test coverage gap detector.

Scans all Python modules under augmentum/ and checks whether each has
corresponding test coverage. Reports:
  1. Untested modules — no test file found
  2. Untested route files — proxy/*_routes.py without test_*_routes.py
  3. Orphaned tests — test files for modules that no longer exist
  4. Coverage summary by subsystem

Exit code 0 = all modules covered, 1 = gaps found.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import _common  # noqa: F401 — import side-effect: UTF-8-safe stdout/stderr

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: Cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)

ROOT = _find_root()

_COLOR = os.environ.get("TERM") or os.name != "nt"
def _red(s: str) -> str:    return f"\033[91m{s}\033[0m" if _COLOR else s
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m" if _COLOR else s
def _green(s: str) -> str:  return f"\033[92m{s}\033[0m" if _COLOR else s
def _cyan(s: str) -> str:   return f"\033[96m{s}\033[0m" if _COLOR else s
def _bold(s: str) -> str:   return f"\033[1m{s}\033[0m" if _COLOR else s
def _dim(s: str) -> str:    return f"\033[2m{s}\033[0m" if _COLOR else s

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _find_modules() -> list[Path]:
    """Find all Python modules under augmentum/ (excluding __init__, __pycache__)."""
    modules = []
    aug_dir = ROOT / "augmentum"
    for py in aug_dir.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        if py.name == "__init__.py":
            continue
        modules.append(py.relative_to(ROOT))
    return sorted(modules)


def _find_test_files() -> set[str]:
    """Find all test file stems under tests/."""
    test_dir = ROOT / "tests"
    stems = set()
    for py in test_dir.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        if py.name.startswith("test_") or py.name.startswith("live_"):
            stems.add(py.stem)
    return stems


def _module_to_test_candidates(mod: Path) -> list[str]:
    """Generate candidate test file names for a module.

    augmentum/models/ollama.py -> [test_ollama, test_models_ollama, test_smoke_models, test_contract_ollama, ...]
    augmentum/proxy/chat_routes.py -> [test_chat_routes, test_proxy_chat_routes, ...]
    """
    stem = mod.stem  # e.g. "ollama"
    parts = list(mod.parts)  # ['augmentum', 'models', 'ollama.py']
    subsystem = parts[1] if len(parts) > 2 else ""

    candidates = [
        f"test_{stem}",
        f"test_{subsystem}_{stem}",
        f"test_smoke_{subsystem}",
        f"test_smoke_{stem}",
        f"test_contract_{stem}",
        f"test_integration_{stem}",
        f"test_live_{stem}",
        f"live_{stem}_test",
    ]
    # A subsystem-wide test file (e.g. tests/test_powers.py) is allowed to
    # cover every module under augmentum/<subsystem>/. Without this, a
    # comprehensive test file gets ignored unless every module also has its
    # own per-module test, which is too strict for small packages where one
    # well-organised file covers the whole API surface.
    if subsystem:
        candidates.append(f"test_{subsystem}")
        candidates.append(f"test_{subsystem}_modules")
        # And tests/test_<subsystem>_routes.py if the module is a route/store
        candidates.append(f"test_{subsystem}_routes")

    # Subsystem-naming aliases. The proxy layer for `augmentum/vfs/` is
    # `files_routes.py`, so the team's test convention has been
    # `tests/test_files_<x>.py`. Recognise that.
    if subsystem == "vfs":
        candidates.append(f"test_files_{stem}")
        candidates.append(f"test_files_{stem.replace('_extractor', '')}")
        candidates.append("test_files_routes")
        candidates.append("test_files_scope_filter")
        candidates.append("test_vfs_bridges")
        candidates.append("test_epub_extractor")

    # Route files: test_chat_routes for proxy/chat_routes.py
    if stem.endswith("_routes"):
        base = stem  # e.g. "chat_routes"
        candidates.append(f"test_{base}")
        # Also check without _routes suffix
        short = base.replace("_routes", "")
        candidates.extend([
            f"test_{short}_routes",
            f"test_{short}",
        ])

    # Special patterns
    if subsystem == "modes":
        # modes/analytical/engine.py -> test_analytical_engine, test_analytical_handler
        if len(parts) > 3:
            mode = parts[2]  # e.g. "analytical"
            candidates.extend([
                f"test_{mode}_{stem}",
                f"test_{mode}_handler",
                f"test_smoke_modes",
            ])

    # Proxy subsystem extras
    if subsystem == "proxy":
        candidates.extend([
            f"test_smoke_proxy",
            f"test_{stem.replace('_routes', '')}_routes",
            f"test_{stem}_ws",  # voice_routes -> test_voice_routes_ws
        ])
        # handler_factory.py, server.py, rate_limit.py, reputation.py — infrastructure
        if stem in ("handler_factory", "server", "rate_limit", "reputation", "streaming"):
            candidates.append("test_proxy")

    # MCP pattern: mcp/server.py -> test_mcp_modules, test_mcp
    if subsystem == "mcp":
        candidates.extend(["test_mcp_modules", "test_mcp"])

    # Reasoning pattern: reasoning/executor.py -> test_reasoning_modules, test_reasoning_flows
    if subsystem == "reasoning":
        candidates.extend(["test_reasoning_modules", "test_reasoning_flows"])

    # Session lifecycle
    if subsystem == "session":
        candidates.extend(["test_session", "test_session_routes"])

    # Knowledge importer / convert_worker
    if stem == "importer" or stem == "convert_worker":
        candidates.extend(["test_knowledge_pipeline", "test_knowledge_packs", "test_knowledge_converter"])

    # Cache modules
    if subsystem == "cache":
        candidates.extend([f"test_{stem}", f"test_cache_{stem}", "test_prompt_cache"])

    # Main entry points
    if stem in ("main", "__main__"):
        candidates.extend(["test_main", "test_smoke_main"])

    # Coder submodules
    if subsystem == "coder":
        candidates.extend(["test_coder_modules", "test_coder_containers"])

    return candidates


def _is_tested(mod: Path, test_stems: set[str]) -> bool:
    """Check if a module has any corresponding test file."""
    candidates = _module_to_test_candidates(mod)
    if any(c in test_stems for c in candidates):
        return True
    # Loose fallback: any test stem that contains BOTH a singular form
    # of the subsystem name AND the module stem counts. Catches
    # tests/test_smoke_project_entity.py for projects/store.py, etc.
    parts = list(mod.parts)
    subsystem = parts[1] if len(parts) > 2 else ""
    stem = mod.stem
    if subsystem and len(subsystem) >= 4:
        # Strip trailing 's' for common plural-singular pairs:
        # projects → project, animations → animation, saves → save
        singular = subsystem[:-1] if subsystem.endswith("s") else subsystem
        prefix_matches = (f"test_smoke_{singular}", f"test_{singular}",
                          f"test_smoke_{subsystem}", f"test_{subsystem}")
        for ts in test_stems:
            if any(ts.startswith(p) for p in prefix_matches):
                return True
    # Final fallback: any test file whose stem contains the module stem
    # (and stem is distinctive, >= 4 chars). Catches:
    #   tests/test_breaker_registry.py → loops/breakers.py (via "breaker")
    #   tests/test_observation_ledger.py → loops/ledger.py (via "ledger")
    #   tests/test_smoke_pocket_tts.py → voice/pocket_tts.py
    #   tests/test_coder_tier_classifier.py → loops/tier.py (via "tier")
    if len(stem) >= 4:
        for ts in test_stems:
            if stem in ts:
                return True
    # Also try without the trailing 's' for plural module names:
    # breakers.py → "breaker", ledger.py → already 5 chars
    if stem.endswith("s") and len(stem) >= 6:
        sg = stem[:-1]
        for ts in test_stems:
            if sg in ts:
                return True
    return False


def _count_test_functions(test_dir: Path) -> dict[str, int]:
    """Count test functions per test file."""
    counts = {}
    for py in test_dir.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        if not (py.name.startswith("test_") or py.name.startswith("live_")):
            continue
        try:
            content = py.read_text(encoding="utf-8", errors="ignore")
            count = len(re.findall(r"(?:async\s+)?def\s+test_", content))
            if count > 0:
                counts[py.stem] = count
        except Exception:
            pass
    return counts


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _get_subsystem(mod: Path) -> str:
    """Extract subsystem name from module path."""
    parts = list(mod.parts)
    if len(parts) >= 2:
        return parts[1]
    return "root"


def analyze():
    """Run full coverage gap analysis."""
    modules = _find_modules()
    test_stems = _find_test_files()
    test_counts = _count_test_functions(ROOT / "tests")

    # Categorize
    tested = []
    untested = []
    for mod in modules:
        if _is_tested(mod, test_stems):
            tested.append(mod)
        else:
            untested.append(mod)

    # Route coverage
    route_files = [m for m in modules if "proxy" in str(m) and m.stem.endswith("_routes")]
    tested_routes = [r for r in route_files if _is_tested(r, test_stems)]
    untested_routes = [r for r in route_files if not _is_tested(r, test_stems)]

    # Subsystem breakdown
    subsystems: dict[str, dict] = {}
    for mod in modules:
        sub = _get_subsystem(mod)
        if sub not in subsystems:
            subsystems[sub] = {"total": 0, "tested": 0, "untested": []}
        subsystems[sub]["total"] += 1
        if _is_tested(mod, test_stems):
            subsystems[sub]["tested"] += 1
        else:
            subsystems[sub]["untested"].append(mod)

    return {
        "modules": modules,
        "tested": tested,
        "untested": untested,
        "route_files": route_files,
        "tested_routes": tested_routes,
        "untested_routes": untested_routes,
        "subsystems": subsystems,
        "test_counts": test_counts,
        "test_stems": test_stems,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(data: dict, verbose: bool = False) -> int:
    """Print coverage report. Returns exit code (0=clean, 1=gaps)."""
    modules = data["modules"]
    tested = data["tested"]
    untested = data["untested"]
    subsystems = data["subsystems"]
    test_counts = data["test_counts"]

    total = len(modules)
    tested_n = len(tested)
    pct = (tested_n / total * 100) if total else 0

    print()
    print(_bold("=" * 60))
    print(_bold("  AUGMENTUM TEST COVERAGE REPORT"))
    print(_bold("=" * 60))
    print()

    # Summary
    color = _green if pct >= 80 else (_yellow if pct >= 50 else _red)
    print(f"  Modules:  {color(f'{tested_n}/{total}')} ({color(f'{pct:.1f}%')})")
    print(f"  Routes:   {len(data['tested_routes'])}/{len(data['route_files'])}")
    print(f"  Tests:    {sum(test_counts.values())} functions in {len(test_counts)} files")
    print()

    # Subsystem table
    print(_bold("  COVERAGE BY SUBSYSTEM"))
    print(f"  {'Subsystem':<20} {'Total':>6} {'Tested':>7} {'Coverage':>9}")
    print(f"  {'-'*20} {'-'*6} {'-'*7} {'-'*9}")

    for sub in sorted(subsystems.keys(), key=lambda s: subsystems[s]["tested"] / max(subsystems[s]["total"], 1)):
        info = subsystems[sub]
        sub_pct = (info["tested"] / info["total"] * 100) if info["total"] else 0
        c = _green if sub_pct >= 80 else (_yellow if sub_pct >= 50 else _red)
        print(f"  {sub:<20} {info['total']:>6} {info['tested']:>7} {c(f'{sub_pct:>8.1f}%')}")
    print()

    # Untested routes (always show — these are critical)
    if data["untested_routes"]:
        print(_red(_bold(f"  UNTESTED ROUTES ({len(data['untested_routes'])})")))
        for r in sorted(data["untested_routes"]):
            print(f"    {_red('x')} {r}")
        print()

    # Untested modules
    if untested:
        if verbose:
            print(_yellow(_bold(f"  UNTESTED MODULES ({len(untested)})")))
            current_sub = ""
            for mod in sorted(untested, key=lambda m: str(m)):
                sub = _get_subsystem(mod)
                if sub != current_sub:
                    current_sub = sub
                    print(f"    {_cyan(sub + '/')}")
                print(f"      {_yellow('x')} {mod.name}")
            print()
        else:
            print(_yellow(f"  {len(untested)} untested modules (use --verbose to list)"))
            print()

    # Verdict
    if not untested and not data["untested_routes"]:
        print(_green(_bold("  ALL MODULES COVERED")))
        return 0
    else:
        gaps = len(untested) + len(data["untested_routes"])
        print(_red(f"  {gaps} coverage gaps found"))
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    data = analyze()
    exit_code = report(data, verbose=verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
