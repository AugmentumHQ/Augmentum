#!/usr/bin/env python3
"""Self-test for the doc-coverage engine (doc_coverage/).

Runnable standalone: `python selftest_doc_coverage.py`. Exits non-zero on
any failed assertion. Covers both doc styles (membership + enumerable),
exemption, and the scaffold hook on synthetic fixtures, then asserts a
LIVE invariant that doubles as a regression guard: the SKILL.md Handler
Pattern table must list every dispatch mode under augmentum/modes/.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_THIS = Path(__file__).resolve()
_SKILL_DIR = _THIS.parent.parent
sys.path.insert(0, str(_SKILL_DIR))

import _common  # noqa: E402,F401 — UTF-8-safe stdout
from doc_coverage import SPECS, evaluate, scaffold_for, spec_by_name  # noqa: E402
from doc_coverage.engine import CoverageSpec  # noqa: E402
from model import find_project_root  # noqa: E402

_failures: list[str] = []


def _check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok   {msg}")
    else:
        _failures.append(msg)
        print(f"  FAIL {msg}")


def test_membership_style() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        doc = root / "MAP.md"
        doc.write_text(
            "## Map\n| **Alpha** | alpha | done |\n"
            "| **Bravo** | bravo | done |\n## End\n",
            encoding="utf-8",
        )
        spec = CoverageSpec(
            name="letters",
            describe="test",
            code_set=lambda r: {"alpha", "bravo", "charlie", "infra"},
            fix_location="MAP.md",
            exempt=frozenset({"infra"}),
            doc_rel="MAP.md",
            start_marker="## Map",
            end_marker="## End",
            scaffold=lambda x: f"| **{x.title()}** | {x} | TODO |",
        )
        res = evaluate(spec, root)
        _check(res.documented == ["alpha", "bravo"], "membership: documented = alpha,bravo")
        _check(res.missing == ["charlie"], "membership: missing = charlie")
        _check(res.exempt_present == ["infra"], "membership: exempt = infra")
        _check(scaffold_for(spec, root) == ["| **Charlie** | charlie | TODO |"],
               "membership: scaffold builds the charlie row")


def test_enumerable_style() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        spec = CoverageSpec(
            name="ids",
            describe="test",
            code_set=lambda r: {"a", "b", "c", "legacy"},
            fix_location="cards/",
            exempt=frozenset({"legacy"}),
            doc_set=lambda r: {"a", "c"},
            scaffold=lambda x: f"missing card {x}",
        )
        res = evaluate(spec, root)
        _check(res.documented == ["a", "c"], "enumerable: documented = a,c")
        _check(res.missing == ["b"], "enumerable: missing = b")
        _check(res.exempt_present == ["legacy"], "enumerable: exempt = legacy")


def test_region_isolation() -> None:
    # A mention OUTSIDE the region must NOT count as documentation.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "D.md").write_text(
            "intro charlie is mentioned here\n## Map\n| alpha |\n## End\n",
            encoding="utf-8",
        )
        spec = CoverageSpec(
            name="x", describe="t",
            code_set=lambda r: {"alpha", "charlie"},
            fix_location="D.md", doc_rel="D.md",
            start_marker="## Map", end_marker="## End",
        )
        res = evaluate(spec, root)
        _check("charlie" in res.missing,
               "region: out-of-region mention does not count as documented")


def test_live_specs_present() -> None:
    for name in ("subsystems", "modes", "provider_cards"):
        _check(spec_by_name(name) is not None, f"registry has spec {name!r}")


def test_live_modes_fully_documented() -> None:
    # Regression guard: every dispatch mode MUST have a Handler row.
    root = find_project_root(Path.cwd())
    res = evaluate(spec_by_name("modes"), root)
    _check(not res.missing,
           f"live: all modes documented (missing={res.missing})")


def main() -> int:
    print("doc-coverage engine self-test")
    for fn in (test_membership_style, test_enumerable_style,
               test_region_isolation, test_live_specs_present,
               test_live_modes_fully_documented):
        print(f"\n[{fn.__name__}]")
        fn()
    print()
    if _failures:
        print(f"FAILED — {len(_failures)} assertion(s)")
        return 1
    print("PASSED — all assertions green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
