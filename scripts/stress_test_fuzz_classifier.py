"""Walk Augmentum's own source and print how the fuzz classifier rates it.

Sanity-check that on a real corpus:
  - parsers / decoders / signature handlers are flagged fuzzable
  - route handlers / fixtures / generators / async are not
  - the fuzzable percentage is single-digit (most code isn't fuzzable)
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from augmentum.bug_finder.fuzz.classifier import (
    FuzzVerdict,
    classify_function,
)


SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".tox", "node_modules",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".augmentum", "tests",
}


def iter_py_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def classify_file(file: Path, root: Path) -> list[tuple[str, FuzzVerdict]]:
    """Walk one file with class-scope tracking so methods land with
    ``inside_class=True`` and get rejected as such — matching what the
    orchestrator's ``classify_chunk`` will do at run time."""
    try:
        src = file.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    rel = str(file.relative_to(root)).replace("\\", "/")
    out: list[tuple[str, FuzzVerdict]] = []

    def walk(node: ast.AST, class_name: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                verdict = classify_function(
                    child, file_path=rel,
                    inside_class=class_name is not None,
                )
                qualified = (
                    f"{rel}::{class_name}.{child.name}"
                    if class_name else f"{rel}::{child.name}"
                )
                out.append((qualified, verdict))
                walk(child, class_name)
            else:
                walk(child, class_name)

    walk(tree, None)
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    target = root / "augmentum"
    fuzzable: list[tuple[str, FuzzVerdict]] = []
    not_fuzzable: list[tuple[str, FuzzVerdict]] = []
    reason_counter: Counter[str] = Counter()

    for file in iter_py_files(target):
        for name, v in classify_file(file, root):
            if v.fuzzable:
                fuzzable.append((name, v))
            else:
                not_fuzzable.append((name, v))
                # Bucket the reason into coarse categories
                if v.is_method:                   bucket = "method"
                else:
                    r = v.reason
                    if "async" in r:                  bucket = "async"
                    elif "generator" in r:            bucket = "generator"
                    elif "test" in r:                 bucket = "test"
                    elif "decorated" in r:            bucket = "decorator"
                    elif "no positional" in r:        bucket = "no-args"
                    elif "no type hint" in r:         bucket = "untyped-unknown-name"
                    elif "not bytes/str-like" in r:   bucket = "wrong-type"
                    else:                             bucket = "other"
                reason_counter[bucket] += 1

    total = len(fuzzable) + len(not_fuzzable)
    pct = (len(fuzzable) / total * 100.0) if total else 0.0
    print(f"\nClassified {total} function defs under augmentum/")
    print(f"  fuzzable:     {len(fuzzable):5d}  ({pct:.1f}%)")
    print(f"  not-fuzzable: {len(not_fuzzable):5d}")
    print("\nReject reasons:")
    for bucket, count in reason_counter.most_common():
        print(f"  {bucket:25s} {count:5d}")

    print("\n=== Sample fuzzable functions (first 20 alphabetical) ===")
    for name, v in sorted(fuzzable, key=lambda kv: kv[0])[:20]:
        print(f"  [{v.input_kind:>18s}]  {name}  (param: {v.target_param})")

    print(f"\n... {max(0, len(fuzzable) - 20)} more fuzzable targets")

    print("\n=== Sample REJECTED route handlers (decorator bucket, first 8) ===")
    routed = [
        (n, v) for n, v in not_fuzzable
        if "decorated" in v.reason
    ]
    for name, v in sorted(routed, key=lambda kv: kv[0])[:8]:
        print(f"  {name}  ({v.reason})")


if __name__ == "__main__":
    main()
