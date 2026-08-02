#!/usr/bin/env python3
"""Refresh fact-fenced doc blocks in CLAUDE.md / SKILL.md.

Usage:
    python refresh_docs.py --check     # exit 1 on drift, prints diff
    python refresh_docs.py --apply     # rewrite stale values in place
    python refresh_docs.py --list      # list registered facts + values

Convention: doc files mark facts with HTML comments like

    User-scoped tables (<!--fact:tables.user_scoped.count-->77<!--/-->)

The opening tag names a fact in ``facts.registry.FACTS``; everything
between the opener and the closing ``<!--/-->`` is replaced with the
current rendered value of that fact.

Files refreshed:
  CLAUDE.md
  .claude/skills/augmentum-dev/SKILL.md
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

# Resolve project root + import the skill's model/facts packages.
_THIS = Path(__file__).resolve()
_SKILL_DIR = _THIS.parent.parent  # scripts/ -> augmentum-dev/
sys.path.insert(0, str(_SKILL_DIR))

from facts import FACTS, render_fact  # noqa: E402
from model import find_project_root, open_model, refresh  # noqa: E402

DOC_TARGETS = (
    "CLAUDE.md",
    ".claude/skills/augmentum-dev/SKILL.md",
)

# Matches: <!--fact:NAME-->BODY<!--/-->
# BODY can span lines; non-greedy.
FACT_BLOCK_RE = re.compile(
    r"<!--fact:([\w.]+)-->(.*?)<!--/-->",
    re.DOTALL,
)


def _refresh_text(db, original: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (new_text, [(fact_name, old, new), ...]) where each tuple
    represents a fact whose rendered value differed from the doc."""
    changes: list[tuple[str, str, str]] = []

    def _replace(m: re.Match) -> str:
        name = m.group(1)
        old_body = m.group(2)
        if name not in FACTS:
            # Unknown fact name — leave as-is, surface a warning later.
            return m.group(0)
        new_body = render_fact(db, name)
        if old_body != new_body:
            changes.append((name, old_body, new_body))
        return f"<!--fact:{name}-->{new_body}<!--/-->"

    new_text = FACT_BLOCK_RE.sub(_replace, original)
    return new_text, changes


# Matches fenced code blocks (``` ... ```) and inline code spans
# (`...`). Both are stripped before scanning for unknown fact
# references — documentation examples like
# `<!--fact:NAME-->...<!--/-->` show up in the SKILL.md section that
# explains the syntax and shouldn't trigger warnings.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _unknown_facts(text: str) -> set[str]:
    """Fact names referenced in docs but not registered in FACTS.

    Strips fenced + inline code spans first so documentation examples
    don't fire false-positive warnings.
    """
    masked = _CODE_FENCE_RE.sub("", text)
    masked = _INLINE_CODE_RE.sub("", masked)
    return {
        m.group(1)
        for m in FACT_BLOCK_RE.finditer(masked)
        if m.group(1) not in FACTS
    }


def _print_diff(path: str, original: str, new: str) -> None:
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"{path} (claimed)",
        tofile=f"{path} (current)",
        n=1,
    )
    sys.stdout.writelines(diff)


def _list_facts(db) -> int:
    width = max(len(n) for n in FACTS) + 2
    for name, fact in sorted(FACTS.items()):
        val = render_fact(db, name)
        if len(val) > 70:
            val = val[:67] + "..."
        print(f"  {name:<{width}}  {val}")
        print(f"  {'':<{width}}  {fact.description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="exit 1 on drift, print diff")
    mode.add_argument("--apply", action="store_true",
                      help="rewrite stale values in place")
    mode.add_argument("--list", action="store_true",
                      help="list registered facts + current values")
    args = parser.parse_args(argv)

    project_root = find_project_root(Path.cwd())
    db = open_model(project_root)
    refresh(db, project_root)

    if args.list:
        return _list_facts(db)

    drift = 0
    unknown_warnings: set[str] = set()
    for rel in DOC_TARGETS:
        path = project_root / rel
        if not path.is_file():
            continue
        original = path.read_text(encoding="utf-8")
        new_text, changes = _refresh_text(db, original)
        unknown_warnings |= _unknown_facts(original)
        if not changes:
            continue
        drift += len(changes)
        if args.check:
            print(f"DRIFT  {rel}: {len(changes)} fact(s) stale")
            for name, old, new in changes:
                old_short = old if len(old) < 60 else old[:57] + "..."
                new_short = new if len(new) < 60 else new[:57] + "..."
                print(f"  - {name}")
                print(f"      claimed:  {old_short}")
                print(f"      current:  {new_short}")
            _print_diff(rel, original, new_text)
        else:  # --apply
            path.write_text(new_text, encoding="utf-8")
            print(f"REWROTE  {rel}: {len(changes)} fact(s) refreshed")
            for name, _, new in changes:
                short = new if len(new) < 60 else new[:57] + "..."
                print(f"  + {name}: {short}")

    if unknown_warnings:
        print("\nWARNING: unknown fact names found in docs (no FACTS entry):")
        for n in sorted(unknown_warnings):
            print(f"  - {n}")

    if args.check:
        if drift:
            return 1
        print("OK  no doc drift")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
