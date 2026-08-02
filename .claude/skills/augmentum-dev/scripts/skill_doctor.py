#!/usr/bin/env python3
"""skill_doctor — the augmentum-dev skill's self-diagnosis.

One command that answers "is this skill still describing reality, and if
not, what must a HUMAN do about it?" — separating the two tiers of
autonomy so intervention stays minimal:

  TIER 1  self-healing facts   — countable claims fenced with
          <!--fact:NAME-->…<!--/-->; rewritten from the live codebase
          model by the Stop hook. If these are stale, NO human action is
          needed — the next turn heals them. The doctor just reports the
          pending count so nothing looks mysterious.

  TIER 2  coverage detectors   — code⟷doc SET drift (subsystems, modes,
          providers). These need a human, because a description is
          judgement, not a number. The doctor lists exactly which items
          and where the fix goes, and points at scaffold_doc_row.py for
          paste-ready stubs.

Usage:
    skill_doctor.py                # full report
    skill_doctor.py --scaffold     # also print stub rows for every gap
    skill_doctor.py --json         # machine-readable
    skill_doctor.py --heal         # apply Tier-1 fact heal, then report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_SKILL_DIR = _THIS.parent.parent  # scripts/ -> augmentum-dev/
sys.path.insert(0, str(_SKILL_DIR))

from doc_coverage import SPECS, evaluate, scaffold_for  # noqa: E402
from facts import FACTS, render_fact  # noqa: E402
from model import find_project_root, open_model, refresh  # noqa: E402

try:  # colours are nice-to-have; degrade cleanly if _common is absent
    from _common import bold, cyan, dim, green, red, yellow  # type: ignore
except Exception:  # noqa: BLE001
    def _identity(s: str) -> str:
        return s
    bold = cyan = dim = green = red = yellow = _identity  # type: ignore

_DOC_TARGETS = (
    "CLAUDE.md",
    ".claude/skills/augmentum-dev/SKILL.md",
)
_FACT_BLOCK_RE = re.compile(r"<!--fact:([\w.]+)-->(.*?)<!--/-->", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


# ---------------------------------------------------------------------------
# Tier 1 — facts
# ---------------------------------------------------------------------------

def _fact_report(db, root: Path) -> dict:
    """Which fenced facts are stale (heal pending) and which doc fact
    references name no registered fact."""
    stale: list[dict] = []
    unknown: list[str] = []
    fenced_names: set[str] = set()
    for rel in _DOC_TARGETS:
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # Work off masked text so the fact-syntax EXAMPLE in SKILL.md's
        # own docs doesn't count as a real fence / unknown ref.
        masked = _INLINE_CODE_RE.sub("", _CODE_FENCE_RE.sub("", text))
        for m in _FACT_BLOCK_RE.finditer(masked):
            name, body = m.group(1), m.group(2)
            if name not in FACTS:
                unknown.append(f"{rel}: {name}")
                continue
            fenced_names.add(name)
            current = render_fact(db, name)
            if current.strip() != body.strip():
                stale.append({
                    "fact": name, "doc": rel,
                    "claimed": body.strip()[:60],
                    "current": current.strip()[:60],
                })
    return {
        "registered": len(FACTS),
        "fenced_in_docs": len(fenced_names),
        "stale": stale,
        "unknown_refs": sorted(set(unknown)),
    }


# ---------------------------------------------------------------------------
# Tier 2 — coverage
# ---------------------------------------------------------------------------

def _coverage_report(root: Path) -> list[dict]:
    out: list[dict] = []
    for spec in SPECS:
        res = evaluate(spec, root)
        out.append({
            "name": spec.name,
            "describe": spec.describe,
            "fix_location": spec.fix_location,
            "documented": len(res.documented),
            "missing": res.missing,
            "exempt": len(res.exempt_present),
        })
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _render_text(facts: dict, coverage: list[dict], scaffold: bool,
                 root: Path) -> None:
    print(bold("Augmentum-dev skill · self-diagnosis"))
    print("=" * 40)

    # Tier 1
    print("\n" + bold("TIER 1 — self-healing facts") +
          dim(" (auto-applied by the Stop hook; no human action needed)"))
    print(f"  registered facts : {facts['registered']}")
    print(f"  fenced in docs   : {facts['fenced_in_docs']}")
    n_stale = len(facts["stale"])
    if n_stale:
        print(yellow(f"  stale (heal pending): {n_stale}  "
                     f"→ run refresh_docs.py --apply (or just finish a turn)"))
        for s in facts["stale"]:
            print(dim(f"      {s['fact']} [{s['doc']}]: "
                      f"{s['claimed']!r} → {s['current']!r}"))
    else:
        print(green("  stale: 0   ✓ docs match the codebase model"))
    if facts["unknown_refs"]:
        print(red(f"  unknown fact refs: {len(facts['unknown_refs'])} "
                  f"(a doc names a fact with no FACTS entry)"))
        for u in facts["unknown_refs"]:
            print(dim(f"      {u}"))

    # Tier 2
    print("\n" + bold("TIER 2 — coverage drift detectors") +
          dim(" (a description is judgement → needs a human)"))
    total_missing = 0
    for c in coverage:
        total_missing += len(c["missing"])
        head = (f"  {c['name']:<15} {c['documented']:>3} documented · "
                f"{len(c['missing']):>2} missing · {c['exempt']} exempt")
        if c["missing"]:
            print(yellow(head + f"   → {c['fix_location']}"))
        else:
            print(green(head + "   ✓"))

    # Human action
    print("\n" + bold("HUMAN ACTION REQUIRED") +
          f" ({total_missing} item(s) — everything else self-heals)")
    if not total_missing:
        print(green("  none — the skill fully describes the codebase. ✓"))
    else:
        for c in coverage:
            if not c["missing"]:
                continue
            items = ", ".join(c["missing"])
            print(f"  • {c['fix_location']}:")
            print(dim(f"      {items}"))
            print(dim(f"      stubs: scaffold_doc_row.py {c['name']}"))

    # Posture line
    print("\n" + bold("Autonomy posture: ") +
          f"{n_stale} stale fact(s) heal automatically · "
          f"{total_missing} doc cell(s) need a human.")

    if scaffold and total_missing:
        print("\n" + bold("── paste-ready stubs ──"))
        for spec in SPECS:
            lines = scaffold_for(spec, root)
            if lines:
                print(f"\n# {spec.name} → {spec.fix_location}")
                for line in lines:
                    print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scaffold", action="store_true",
                        help="also print paste-ready stub rows for gaps")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    parser.add_argument("--heal", action="store_true",
                        help="apply Tier-1 fact heal before reporting")
    args = parser.parse_args(argv)

    root = find_project_root(Path.cwd())
    db = open_model(root)
    refresh(db, root)

    if args.heal:
        # Reuse refresh_docs' apply path so behaviour is identical to the hook.
        import subprocess  # noqa: PLC0415
        subprocess.run(
            [sys.executable, str(_THIS.parent / "refresh_docs.py"), "--apply"],
            check=False,
        )
        db = open_model(root)
        refresh(db, root)

    facts = _fact_report(db, root)
    coverage = _coverage_report(root)

    if args.json:
        print(json.dumps({"facts": facts, "coverage": coverage}, indent=2))
        return 0

    _render_text(facts, coverage, args.scaffold, root)
    # Exit non-zero if a human has something to do (CI-friendly).
    total_missing = sum(len(c["missing"]) for c in coverage)
    return 1 if (total_missing or facts["unknown_refs"]) else 0


if __name__ == "__main__":
    sys.exit(main())
