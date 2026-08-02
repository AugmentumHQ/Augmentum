"""AST-based static chunker — produces detector targets without a planner.

The standard bug_finder pipeline uses an LLM planner to walk the
codebase and emit a curated list of N "interesting" chunks for the
detector to inspect. At temp=0 the planner is deterministically
conservative and at broad scope (whole project, hundreds of files) it
exhausts its token budget reading code before converging on a chunk
list.

This module sidesteps the planner. Given the workspace root + optional
focus_paths, it walks every Python source file via :mod:`ast`, lifts
every top-level function and class method, and emits ``_Chunk`` records
in the same shape the planner produces. The detector + verifier + fixer
pipeline downstream is unchanged.

Trade-off vs planner-driven chunking:

* **Planner**: small, curated, biased toward "looks risky." High signal
  per chunk, but capped at planner_budget tokens and can't see the
  whole repo in one pass.
* **Static**: exhaustive, naive — every function in scope is a chunk.
  Lower signal per chunk because the detector wastes time on
  obviously-clean code, but the *coverage* is unbounded and the run
  shape is "feed code, ask for bugs" — closer to the Mythos-Preview
  experimental scaffold than the planner-driven shape.

Filter policy (defaults):

* Skip ``test_*.py`` files (covered by their own runs in practice).
* Skip ``__init__.py`` (almost always pure re-exports).
* Skip ``__pycache__/``, ``.venv/``, ``node_modules/``, ``dist/``,
  ``build/`` (vendored / generated code, no signal).
* Skip dunder methods *except* ``__init__`` and ``__call__`` (those
  routinely hold actual logic — auth middleware, callable services).
* Skip functions shorter than ``min_function_lines`` (defaults 5) —
  trivial getters/setters waste detector budget.
* Skip the special name ``_`` (gettext alias, never code).

Output is sorted (file, line_start) for deterministic ordering across
runs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class StaticChunk:
    """Identical shape to the orchestrator's planner ``_Chunk``.

    Defined locally so the chunker is import-safe from the orchestrator
    (which would otherwise create a circular import). The orchestrator
    converts these into its own ``_Chunk`` at the dispatch site.
    """

    file: str
    function: str
    line_start: int
    line_end: int
    rationale: str = ""
    suspected_class: str = ""


_DEFAULT_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".venv", "venv", ".git", "node_modules",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "site-packages", "egg-info",
})


_DEFAULT_KEEP_DUNDERS: frozenset[str] = frozenset({
    "__init__", "__call__", "__enter__", "__exit__",
    "__aenter__", "__aexit__",
})


def _is_skippable_filename(name: str, *, skip_tests: bool) -> bool:
    """True when the filename should be skipped wholesale.

    Filters are applied in order so the cheapest checks short-circuit
    first — the chunker walks thousands of files in a typical run."""
    if name == "__init__.py":
        return True
    if skip_tests and (name.startswith("test_") or name.endswith("_test.py")):
        return True
    return False


def _function_is_skippable(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    min_lines: int,
    keep_dunders: frozenset[str],
    skip_private: bool,
) -> bool:
    """Per-function filter — exclude trivial / boilerplate functions
    from the chunk list."""
    name = node.name
    if name == "_":
        return True
    if name.startswith("__") and name.endswith("__") and name not in keep_dunders:
        return True
    if skip_private and name.startswith("_") and not name.startswith("__"):
        return True
    line_start = node.lineno
    line_end = getattr(node, "end_lineno", line_start) or line_start
    if (line_end - line_start + 1) < min_lines:
        return True
    return False


def _extract_chunks_from_tree(
    tree: ast.AST,
    *,
    rel_path: str,
    min_function_lines: int,
    keep_dunders: frozenset[str],
    skip_private: bool,
) -> list[StaticChunk]:
    """Walk the AST and emit one chunk per qualifying function."""
    out: list[StaticChunk] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _function_is_skippable(
                node,
                min_lines=min_function_lines,
                keep_dunders=keep_dunders,
                skip_private=skip_private,
            ):
                continue
            line_start = node.lineno
            line_end = getattr(node, "end_lineno", line_start) or line_start
            # The class context isn't tracked by ast.walk (it flattens).
            # We could rebuild via a visitor, but the detector's prompt
            # already shows the file+function header and the detector
            # has tools to read surrounding context. Leaving
            # ``suspected_class`` empty matches what the planner emits
            # when the chunk is at module scope.
            out.append(StaticChunk(
                file=rel_path,
                function=node.name,
                line_start=line_start,
                line_end=line_end,
                rationale="static AST chunk (planner bypass)",
                suspected_class="",
            ))
    return out


def collect_static_chunks(
    workspace_root: Path,
    *,
    focus_paths: tuple[str, ...] = (),
    max_chunks: int = 50,
    min_function_lines: int = 5,
    skip_tests: bool = True,
    skip_private: bool = False,
    skip_dirs: frozenset[str] = _DEFAULT_SKIP_DIRS,
    keep_dunders: frozenset[str] = _DEFAULT_KEEP_DUNDERS,
) -> list[StaticChunk]:
    """Walk Python source files under ``workspace_root`` and return up
    to ``max_chunks`` function-level chunks.

    ``focus_paths`` narrows the walk to specific subdirectories
    (relative to ``workspace_root``). Empty tuple = whole workspace.

    The chunker is deterministic: identical inputs produce identical
    output, ordered by (file, line_start). A variance bench can re-run
    with the same workspace and expect bit-identical chunk lists.

    Cap policy: when more candidates exist than ``max_chunks``, the
    chunker returns the first N after sort. Callers wanting different
    sampling (random, weighted-by-LOC, etc.) should call this for the
    full list with ``max_chunks=0`` to get everything, then sub-sample.
    """
    if not workspace_root.is_dir():
        log.warning(
            "static_chunker_workspace_missing",
            workspace_root=str(workspace_root),
        )
        return []

    # Build the set of allowed-prefix paths (absolute) from focus_paths.
    # Empty = no filter — walk everything.
    allowed_prefixes: list[Path] = []
    if focus_paths:
        for raw in focus_paths:
            p = (workspace_root / raw.strip()).resolve()
            if p.exists():
                allowed_prefixes.append(p)
            else:
                log.debug(
                    "static_chunker_focus_path_missing",
                    focus_path=raw, resolved=str(p),
                )

    candidates: list[StaticChunk] = []
    for py_file in workspace_root.rglob("*.py"):
        # Cheap dir-prefix check — skip anything under a known noise
        # directory before opening the file.
        if any(part in skip_dirs for part in py_file.parts):
            continue
        if _is_skippable_filename(py_file.name, skip_tests=skip_tests):
            continue
        if allowed_prefixes and not any(
            _is_under(py_file, prefix) for prefix in allowed_prefixes
        ):
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            log.debug(
                "static_chunker_read_failed",
                file=str(py_file), error=str(exc)[:120],
            )
            continue

        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:
            # Vendored / generated / Python 2 holdovers — skip silently.
            log.debug(
                "static_chunker_parse_failed",
                file=str(py_file), error=str(exc)[:120],
            )
            continue

        try:
            rel_path = str(py_file.relative_to(workspace_root))
        except ValueError:
            # rglob shouldn't yield anything outside workspace_root but
            # be defensive against symlinks.
            rel_path = str(py_file)
        # Normalize to forward-slash so paths look the same across OSes
        # in the detector's prompt (planner uses POSIX-style paths too).
        rel_path = rel_path.replace("\\", "/")

        candidates.extend(_extract_chunks_from_tree(
            tree,
            rel_path=rel_path,
            min_function_lines=min_function_lines,
            keep_dunders=keep_dunders,
            skip_private=skip_private,
        ))

    candidates.sort(key=lambda c: (c.file, c.line_start, c.function))
    if max_chunks > 0:
        candidates = candidates[:max_chunks]
    log.info(
        "static_chunker_complete",
        workspace_root=str(workspace_root),
        focus_paths=list(focus_paths),
        chunks=len(candidates),
        cap=max_chunks,
    )
    return candidates


def _is_under(path: Path, prefix: Path) -> bool:
    """True when ``path`` is at or below ``prefix``."""
    try:
        path.resolve().relative_to(prefix)
        return True
    except ValueError:
        return False
