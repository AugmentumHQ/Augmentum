"""Test the bug_finder substrate against external GitHub projects.

Validates how agnostic our generalization layer is. For each target:
  1. Clones the repo to a tmp dir
  2. Runs the framework adapter's ``list_routes`` (live AST scan —
     no augmentum-dev caches expected on external code)
  3. Reports route count + sample
  4. Tests ``list_settings_files``, ``identify_test_command``
  5. Documents what works vs what doesn't

The test bench is honest: external codebases won't ship
augmentum-dev caches, so the deterministic scanners (security_check,
runtime_checks etc.) won't apply. But the adapter's live scan path
SHOULD work — that's the cross-codebase claim we want to validate.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentum.bug_finder.adapters import FastAPIAdapter, NullAdapter
from augmentum.bug_finder import refs


@dataclass(frozen=True)
class Target:
    name: str
    git_url: str
    framework: str = "fastapi"
    expected_routes_at_least: int = 5
    notes: str = ""


# Hand-picked external targets — varied size, all FastAPI.
_TARGETS: tuple[Target, ...] = (
    Target(
        name="full-stack-fastapi-template",
        git_url="https://github.com/fastapi/full-stack-fastapi-template",
        framework="fastapi",
        expected_routes_at_least=10,
        notes=(
            "Official FastAPI starter, well-organized. Routes under "
            "backend/app/api/routes/. Best baseline for the adapter."
        ),
    ),
    Target(
        name="fastapi-realworld-example-app",
        git_url="https://github.com/nsidnev/fastapi-realworld-example-app",
        framework="fastapi",
        expected_routes_at_least=15,
        notes=(
            "RealWorld pattern reference (Articles/Comments/Users). "
            "Routes under app/api/."
        ),
    ),
    Target(
        name="fastapi-best-practices",
        git_url="https://github.com/zhanymkanov/fastapi-best-practices",
        framework="fastapi",
        expected_routes_at_least=3,
        notes=(
            "Smaller example repo. Tests adapter resilience on "
            "smaller / less-conventional structures."
        ),
    ),
)


def _clone(git_url: str, dest: Path) -> bool:
    """Shallow clone. Returns False on failure."""
    cmd = [
        "git", "clone", "--depth", "1", "--quiet",
        git_url, str(dest),
    ]
    try:
        subprocess.run(
            cmd, check=True, capture_output=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _test_target(target: Target) -> dict:
    """Clone + adapter-scan one target. Returns a summary dict."""
    print(f"\n== {target.name}")
    print(f"   {target.git_url}")
    print(f"   {target.notes}")

    with tempfile.TemporaryDirectory(prefix="bf_ext_") as tmp:
        clone_dir = Path(tmp) / "repo"
        t0 = time.monotonic()
        print(f"   cloning...")
        if not _clone(target.git_url, clone_dir):
            print(f"   FAIL clone failed")
            return {
                "name": target.name, "ok": False,
                "reason": "clone failed",
            }
        clone_elapsed = time.monotonic() - t0
        print(f"   cloned in {clone_elapsed:.1f}s ({_count_files(clone_dir)} files)")

        adapter = FastAPIAdapter()

        # 1. list_routes via live AST scan
        t0 = time.monotonic()
        routes = adapter.list_routes(clone_dir)
        scan_elapsed = time.monotonic() - t0
        print(f"   list_routes: {len(routes)} routes in {scan_elapsed:.1f}s")
        if routes:
            sample = routes[:5]
            for r in sample:
                print(f"     {r.method:8s} {r.path:50s} -> {r.file}:{r.line}")
            if len(routes) > 5:
                print(f"     ... +{len(routes)-5} more")

        # 2. list_settings_files
        settings = adapter.list_settings_files(clone_dir)
        print(f"   list_settings_files: {len(settings)} hints")
        for s in settings[:3]:
            print(f"     {s.kind:15s} {s.file}")

        # 3. test_command
        cmd = adapter.identify_test_command(clone_dir)
        print(f"   identify_test_command: {cmd or '(no signal)'}")

        # 4. has_augmentum_dev_refs (expect False for external)
        has_refs = refs.has_augmentum_dev_refs(clone_dir)
        print(f"   ships augmentum-dev refs: {has_refs}")

        verdict = "PASS" if len(routes) >= target.expected_routes_at_least else "PARTIAL"
        print(f"   verdict: {verdict} "
              f"(expected >={target.expected_routes_at_least}, got {len(routes)})")

        return {
            "name": target.name,
            "ok": verdict == "PASS",
            "verdict": verdict,
            "routes_found": len(routes),
            "routes_expected": target.expected_routes_at_least,
            "settings_hints": len(settings),
            "test_command": cmd,
            "has_aug_dev_refs": has_refs,
            "scan_seconds": round(scan_elapsed, 2),
        }


def _count_files(root: Path) -> int:
    try:
        return sum(1 for _ in root.rglob("*") if _.is_file())
    except OSError:
        return 0


def main() -> int:
    if shutil.which("git") is None:
        print("FATAL: git CLI required", file=sys.stderr)
        return 2

    print("=" * 70)
    print("  BUG FINDER SUBSTRATE — EXTERNAL CODEBASE TEST")
    print("=" * 70)
    print()
    print(f"  Testing {len(_TARGETS)} external FastAPI projects to validate")
    print(f"  the framework adapter's live AST scan works on codebases")
    print(f"  WITHOUT augmentum-dev caches present.")

    results: list[dict] = []
    for target in _TARGETS:
        try:
            results.append(_test_target(target))
        except Exception as exc:  # noqa: BLE001
            print(f"   FAIL FAILED: {type(exc).__name__}: {exc}")
            results.append({
                "name": target.name, "ok": False,
                "reason": f"{type(exc).__name__}: {exc}",
            })

    # ----- summary -----
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r.get("ok"))
    total = len(results)
    print(f"\n  Passed: {passed}/{total}")
    print()
    for r in results:
        mark = "OK" if r.get("ok") else "FAIL"
        line = f"  {mark} {r['name']:35s}"
        if "routes_found" in r:
            line += f" routes={r['routes_found']:>3d}"
            if "scan_seconds" in r:
                line += f"  scan={r['scan_seconds']:.1f}s"
        if not r.get("ok") and "reason" in r:
            line += f"  ({r['reason']})"
        print(line)

    print()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
