"""Declarative doc-coverage engine.

A ``CoverageSpec`` diffs a *code-derived* set of item keys (things that
SHOULD be documented, computed live from the tree so it can't rot)
against the set the docs actually declare. Two doc styles are
supported, because docs come in two shapes:

  - **enumerable** (``doc_set``): the documented set is itself
    enumerable — filenames in a dir, ids in a table. missing = code -
    doc_set - exempt.
  - **membership** (``doc_rel`` + markers + ``match``): the doc is prose
    and you test "is item X mentioned in this region?" (default:
    word-boundary regex). This is the right model for a Markdown table
    where the key appears somewhere in the row.

Everything here is pure + filesystem-only (no LLM, no network, no DB),
so it runs inside every ``audit.py`` pass and in the standalone
``skill_doctor.py`` without cost.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


def _word_match(item: str, region: str) -> bool:
    """Default membership test: item appears as a whole word in region."""
    return re.search(rf"\b{re.escape(item)}\b", region) is not None


@dataclass(frozen=True)
class CoverageSpec:
    """One tracked code⟷doc set correspondence.

    Provide EITHER ``doc_set`` (enumerable style) OR ``doc_rel`` +
    ``start_marker`` (membership style). ``scaffold`` turns a missing
    key into a paste-ready stub line so the residual human step is a
    fill-in-the-blank, not authoring from scratch.
    """

    name: str
    describe: str
    # Code-derived set: the keys that SHOULD be documented.
    code_set: Callable[[Path], set[str]]
    # Human-readable pointer to where a fix goes (doc path or section).
    fix_location: str
    # Keys intentionally not requiring a doc entry.
    exempt: frozenset[str] = frozenset()
    # A stub doc line for a missing key (paste-ready).
    scaffold: Callable[[str], str] = field(default=lambda item: f"- {item}")

    # --- enumerable style ---
    doc_set: Callable[[Path], set[str]] | None = None

    # --- membership style ---
    doc_rel: str = ""            # project-relative doc file
    start_marker: str = ""       # region start ("" = whole file)
    end_marker: str = ""         # region end ("" = to EOF)
    match: Callable[[str, str], bool] = _word_match

    def region_text(self, root: Path) -> str:
        """The doc region a membership spec tests against."""
        if not self.doc_rel:
            return ""
        path = root / self.doc_rel
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        start = text.find(self.start_marker) if self.start_marker else 0
        if start < 0:
            return ""
        rest = text[start:]
        if self.end_marker:
            end = rest.find(self.end_marker, len(self.start_marker))
            if end > 0:
                rest = rest[:end]
        return rest


@dataclass
class CoverageResult:
    name: str
    describe: str
    fix_location: str
    documented: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    exempt_present: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.documented) + len(self.missing) + len(self.exempt_present)

    @property
    def ok(self) -> bool:
        return not self.missing


def evaluate(spec: CoverageSpec, root: Path) -> CoverageResult:
    """Classify every code-set key as documented / missing / exempt."""
    code = spec.code_set(root)
    res = CoverageResult(name=spec.name, describe=spec.describe,
                         fix_location=spec.fix_location)

    if spec.doc_set is not None:
        documented_set = spec.doc_set(root)
        for item in sorted(code):
            if item in spec.exempt:
                res.exempt_present.append(item)
            elif item in documented_set:
                res.documented.append(item)
            else:
                res.missing.append(item)
        return res

    region = spec.region_text(root)
    for item in sorted(code):
        if item in spec.exempt:
            res.exempt_present.append(item)
        elif spec.match(item, region):
            res.documented.append(item)
        else:
            res.missing.append(item)
    return res


def scaffold_for(spec: CoverageSpec, root: Path) -> list[str]:
    """Paste-ready stub lines for every missing key in ``spec``."""
    res = evaluate(spec, root)
    return [spec.scaffold(item) for item in res.missing]
