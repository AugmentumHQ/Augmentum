#!/usr/bin/env python3
"""Augmentum dead code detector.

Cross-references backend routes against frontend API calls to find:
  1. Orphaned endpoints — backend routes never called from the frontend
  2. Ghost calls — frontend fetch() calls hitting endpoints that don't exist
  3. Unused exports — Python functions/classes imported but never referenced

Reads the auto-generated reference files (routes.json, frontend_api_calls.json)
for instant analysis without re-scanning the full codebase.

Exit code 0 = clean, 1 = findings.
"""

from __future__ import annotations

import ast
import json
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
REFS_DIR = Path(__file__).resolve().parent.parent / "references"
_SUPPRESSIONS_PATH = Path(__file__).resolve().parent / "dead_code_suppressions.json"


def _load_suppressions() -> dict[str, list[str]]:
    """Load (creating an empty skeleton if missing) the dead-code allowlist.

    ``ghost_calls``: entries shaped ``"METHOD /api/path"`` (the call's URL with
    its `{param}` placeholders) for calls the URL-matcher mis-reads as ghosts —
    e.g. a query string mistaken for a path segment. ``orphaned_endpoints``:
    ``"METHOD /api/path"`` for routes that *intentionally* have no JS caller
    (internal / server-to-server / webhook), as opposed to mid-build features
    (use ``audit.py --update-baseline`` for the rolling accepted count).
    """
    keys = ("ghost_calls", "orphaned_endpoints")
    if _SUPPRESSIONS_PATH.is_file():
        try:
            data = json.loads(_SUPPRESSIONS_PATH.read_text(encoding="utf-8"))
            return {k: list(data.get(k, [])) for k in keys}
        except (json.JSONDecodeError, KeyError):
            pass
    skeleton = {
        "_comment": "Reviewed-and-accepted dead-code findings. ghost_calls: 'METHOD /api/path' the matcher mis-flags. orphaned_endpoints: 'METHOD /api/path' that intentionally has no JS caller. Fix real findings; don't suppress them.",
        "ghost_calls": [], "orphaned_endpoints": [],
    }
    _SUPPRESSIONS_PATH.write_text(json.dumps(skeleton, indent=2) + "\n", encoding="utf-8")
    return {"ghost_calls": [], "orphaned_endpoints": []}

_COLOR = os.environ.get("TERM") or os.name != "nt"
def _red(s: str) -> str:    return f"\033[91m{s}\033[0m" if _COLOR else s
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m" if _COLOR else s
def _green(s: str) -> str:  return f"\033[92m{s}\033[0m" if _COLOR else s
def _cyan(s: str) -> str:   return f"\033[96m{s}\033[0m" if _COLOR else s
def _bold(s: str) -> str:   return f"\033[1m{s}\033[0m" if _COLOR else s
def _dim(s: str) -> str:    return f"\033[2m{s}\033[0m" if _COLOR else s

# ---------------------------------------------------------------------------
# Ensure reference files exist
# ---------------------------------------------------------------------------

def _ensure_refs():
    """Refresh reference files if stale."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "refresh_refs",
            Path(__file__).resolve().parent / "refresh_refs.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.refresh_all(quiet=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 1. Orphaned endpoints — backend routes with no frontend caller
# ---------------------------------------------------------------------------

def find_orphaned_endpoints() -> list[dict]:
    """Find backend routes that no frontend code ever calls."""
    routes_file = REFS_DIR / "routes.json"
    calls_file = REFS_DIR / "frontend_api_calls.json"

    if not routes_file.exists() or not calls_file.exists():
        return []

    routes = json.loads(routes_file.read_text(encoding="utf-8"))
    calls = json.loads(calls_file.read_text(encoding="utf-8"))

    # Normalize frontend call URLs (replace {param} with regex-friendly pattern)
    call_urls: set[str] = set()
    call_patterns: list[re.Pattern] = []
    for c in calls.get("calls", []):
        # Strip query strings — they're never part of the route path.
        url = c["url"].split("?", 1)[0].rstrip("/")
        # Also keep the bare-stripped form (some routes register without
        # trailing slash). Add both with and without the trailing-slash form.
        for u in {url, url + "/"} if url else set():
            call_urls.add(u)
            if "{param}" in u:
                parts = u.split("{param}")
                pattern = r"[^/]+".join(re.escape(p) for p in parts)
                call_patterns.append(re.compile(f"^{pattern}$"))

    # Known internal-only routes (not called from UI, used by backends or other services)
    internal_prefixes = {
        "/v1/chat/completions",  # Called by external LLM frontends, not our UI
        "/v1/completions",
        "/api/generate",         # Ollama proxy — called by external clients
        "/api/chat",             # Ollama proxy
        "/api/pull",             # Ollama admin
        "/api/push",
        "/api/create",
        "/api/copy",
        "/api/delete",           # Ollama admin
        "/api/show",             # Ollama admin
        "/api/blobs",            # Ollama admin
        "/api/embed",            # Ollama embed
        "/api/embeddings",       # Ollama/OpenAI embeddings
        "/v1/embeddings",
        "/v1/models",            # OpenAI compat — called by external clients
        "/api/tags",             # Ollama model list — UI uses this but via models.js
        "/api/ps",               # Ollama process list
        "/v1/mcp",               # MCP protocol endpoints
        "/v1/audio",             # OpenAI compat audio — called via voice.js WebSocket
        # OpenAI-compat image surface — called by external clients (DALL-E API parity).
        # UI uses /api/image/* instead.
        "/v1/images",
        "/v1/image-models",
        # OpenAI-compat memory surface — exposed for SDK clients (Continue.dev,
        # Cline, etc.). UI uses /api/memory/* instead.
        "/v1/memory",
        # Coder permissions polled by the in-container coder agent, not the UI.
        "/v1/coder",
        # Prometheus metrics — scraped by ops, not the UI.
        "/metrics",
        # FastAPI root + well-known — server housekeeping, not UI-callable.
        "/.well-known",
    }

    orphaned: list[dict] = []
    for endpoint in routes.get("endpoints", []):
        path = endpoint["path"]
        method = endpoint["method"]

        # Skip internal/proxy routes
        if any(path.startswith(p) for p in internal_prefixes):
            continue

        # Skip WebSocket upgrades (handled differently)
        if method == "WEBSOCKET":
            continue

        # Check direct match
        if path in call_urls:
            continue

        # Check parameterized match (e.g., /api/chats/{id} matches /api/chats/{param})
        matched = False
        for pattern in call_patterns:
            if pattern.match(path):
                matched = True
                break

        # Also check if a parameterized route matches a parameterized call
        # e.g., /api/chats/{chat_id} should match /api/chats/{param}
        if not matched:
            # Normalize path params: /{anything} → /{param}
            normalized = re.sub(r"/\{[^}]+\}", "/{param}", path)
            if normalized in call_urls:
                matched = True
            else:
                for pattern in call_patterns:
                    if pattern.match(normalized):
                        matched = True
                        break

        if not matched:
            orphaned.append({
                "method": method,
                "path": path,
                "handler": endpoint["handler"],
                "file": endpoint["file"],
                "line": endpoint["line"],
            })

    return orphaned

# ---------------------------------------------------------------------------
# 2. Ghost calls — frontend calls to non-existent endpoints
# ---------------------------------------------------------------------------

def find_ghost_calls() -> list[dict]:
    """Find frontend fetch() calls that don't match any backend route."""
    routes_file = REFS_DIR / "routes.json"
    calls_file = REFS_DIR / "frontend_api_calls.json"

    if not routes_file.exists() or not calls_file.exists():
        return []

    routes = json.loads(routes_file.read_text(encoding="utf-8"))
    calls = json.loads(calls_file.read_text(encoding="utf-8"))

    # Build a set of all backend route patterns
    route_urls: set[str] = set()
    route_patterns: list[re.Pattern] = []
    for r in routes.get("endpoints", []):
        path = r["path"]
        route_urls.add(path)
        # Create pattern for parameterized routes
        # Replace params BEFORE escaping so [^/]+ doesn't get escaped
        if "{" in path:
            parts = re.split(r"\{[^}]+\}", path)
            escaped = r"[^/]+".join(re.escape(p) for p in parts)
            route_patterns.append(re.compile(f"^{escaped}$"))

    ghosts: list[dict] = []
    seen: set[str] = set()

    for call in calls.get("calls", []):
        raw_url = call["url"]
        method = call["method"]
        key = f"{method} {raw_url}"
        if key in seen:
            continue
        seen.add(key)

        # Skip WebSocket (different protocol)
        if method == "WS":
            continue

        # Strip query parameters — routes don't include them
        url = raw_url.split("?")[0]

        # Direct match
        if url in route_urls:
            continue

        # Parameterized match
        # Normalize the call's {param} to check against route patterns
        normalized = url.replace("{param}", "placeholder_value")
        matched = False
        for pattern in route_patterns:
            if pattern.match(normalized):
                matched = True
                break

        # Also check if call URL (with {param}) matches route URL (with {var})
        if not matched:
            parts = url.split("{param}")
            call_pattern = r"[^/]+".join(re.escape(p) for p in parts)
            call_re = re.compile(f"^{call_pattern}$")
            for route_url in route_urls:
                plain = re.sub(r"\{[^}]+\}", "x", route_url)
                if call_re.match(plain):
                    matched = True
                    break

        if not matched:
            ghosts.append({
                "method": method,
                "url": raw_url,
                "file": call["file"],
                "line": call["line"],
            })

    return ghosts

# ---------------------------------------------------------------------------
# 3. Test coverage mapping — routes without test files
# ---------------------------------------------------------------------------

def find_untested_routes() -> list[dict]:
    """Find route files that have no corresponding test file."""
    proxy_dir = ROOT / "augmentum" / "proxy"
    tests_dir = ROOT / "tests"

    if not tests_dir.is_dir():
        return []

    # Collect all test file names
    test_names: set[str] = set()
    for tf in tests_dir.rglob("*.py"):
        test_names.add(tf.stem.lower())  # e.g., "test_audio_processor"

    # Check each route file
    untested: list[dict] = []
    for rf in sorted(proxy_dir.glob("*_routes.py")):
        module = rf.stem  # e.g., "browse_routes"
        # Look for test files matching various patterns
        base = module.replace("_routes", "")
        possible_tests = {
            f"test_{module}",           # test_browse_routes
            f"test_{base}",             # test_browse
            f"test_{base}_routes",      # test_browse_routes (same)
            f"test_{base}_api",         # test_browse_api
        }

        if not any(pt in test_names for pt in possible_tests):
            # Check if the module name appears in any test file content
            found_in_test = False
            for tf in tests_dir.rglob("*.py"):
                try:
                    content = tf.read_text(encoding="utf-8", errors="replace")
                    if module in content or f"from augmentum.proxy.{module}" in content:
                        found_in_test = True
                        break
                except Exception:
                    pass

            if not found_in_test:
                untested.append({
                    "route_file": f"augmentum/proxy/{rf.name}",
                    "expected_test": f"tests/test_{base}.py",
                })

    return untested

# ---------------------------------------------------------------------------
# 4. Import/dependency check
# ---------------------------------------------------------------------------

def find_missing_deps() -> list[dict]:
    """Find Python imports that might not be in pyproject.toml dependencies."""
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        return []

    toml_text = pyproject.read_text(encoding="utf-8", errors="replace").lower()

    # Authoritative stdlib set (Python 3.10+). Falls back to a curated set
    # on older interpreters. Avoids the prior maintenance hazard where
    # adding a new stdlib import (`shlex`, `types`, `concurrent`, …) made
    # the scanner cry false positives until someone updated this file.
    stdlib = set(getattr(sys, "stdlib_module_names", set())) or {
        "os", "sys", "re", "json", "math", "time", "datetime", "pathlib",
        "typing", "collections", "functools", "itertools", "contextlib",
        "hashlib", "base64", "uuid", "io", "asyncio", "logging",
        "dataclasses", "enum", "abc", "copy", "textwrap", "html",
        "urllib", "http", "socket", "ssl", "struct", "binascii",
        "importlib", "inspect", "traceback", "warnings", "threading",
        "multiprocessing", "subprocess", "shutil", "tempfile", "glob",
        "csv", "configparser", "secrets", "hmac", "xml", "sqlite3",
        "unittest", "pdb", "string", "codecs", "operator", "signal",
        "gc", "random", "calendar", "difflib", "contextvars", "zoneinfo",
        "ipaddress", "weakref", "queue", "heapq", "bisect", "array",
        "decimal", "fractions", "statistics", "mimetypes", "email",
        "zipfile", "tarfile", "gzip", "bz2", "lzma", "platform",
        "sysconfig", "dis", "token", "tokenize", "pprint", "numbers",
        "shlex", "types", "argparse", "concurrent", "unicodedata",
    }

    # Map import names to pyproject.toml package names
    import_to_pkg = {
        "PIL": "pillow",
        "cv2": "opencv-python",
        "yaml": "pyyaml",
        "sklearn": "scikit-learn",
        "bs4": "beautifulsoup4",
        "docx": "python-docx",
        "pptx": "python-pptx",
        "openpyxl": "openpyxl",
        "aiosqlite": "aiosqlite",
        "httpx": "httpx",
        "fastapi": "fastapi",
        "pydantic": "pydantic",
        "uvicorn": "uvicorn",
        "starlette": "starlette",
        "structlog": "structlog",
        "tiktoken": "tiktoken",
        "numpy": "numpy",
        "torch": "torch",
        "torchaudio": "torchaudio",
        "transformers": "transformers",
        "diffusers": "diffusers",
        "trafilatura": "trafilatura",
        "matplotlib": "matplotlib",
        "pydantic_settings": "pydantic-settings",
    }

    findings: list[dict] = []
    seen_imports: set[str] = set()

    for pyfile in sorted((ROOT / "augmentum").rglob("*.py")):
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        rel = str(pyfile.relative_to(ROOT)).replace("\\", "/")

        # Walk the AST instead of regex-matching `^from|import` — the regex
        # used to match plain English in docstrings (`from research`,
        # `from official documentation`, `import the next page`) and
        # produce noise findings for non-existent packages.
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            pkg: str | None = None
            lineno = getattr(node, "lineno", 1)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _emit_import_finding(
                        alias.name.split(".")[0], lineno, rel,
                        stdlib, import_to_pkg, toml_text,
                        seen_imports, findings,
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                # Skip relative imports (level > 0) — they target the local
                # package, never an external dependency.
                if getattr(node, "level", 0):
                    continue
                _emit_import_finding(
                    node.module.split(".")[0], lineno, rel,
                    stdlib, import_to_pkg, toml_text,
                    seen_imports, findings,
                )

    return findings


def _emit_import_finding(
    pkg: str,
    lineno: int,
    rel: str,
    stdlib: set[str],
    import_to_pkg: dict[str, str],
    toml_text: str,
    seen_imports: set[str],
    findings: list[dict],
) -> None:
    if pkg in stdlib or pkg.startswith("augmentum") or pkg.startswith("_"):
        return
    if pkg in seen_imports:
        return
    seen_imports.add(pkg)
    pyproject_name = import_to_pkg.get(pkg, pkg).lower()
    pkg_lower = pkg.lower()
    # PyPI normalises `_` and `-` interchangeably (PEP 503). Try every
    # spelling so an `import sqlite_vec` doesn't fail to find
    # `sqlite-vec` in pyproject.toml.
    candidates = {
        pyproject_name,
        pyproject_name.replace("_", "-"),
        pyproject_name.replace("-", "_"),
        pkg_lower,
        pkg_lower.replace("_", "-"),
        pkg_lower.replace("-", "_"),
    }
    if any(c in toml_text for c in candidates):
        return
    findings.append({
        "import": pkg,
        "expected_pkg": pyproject_name,
        "file": rel,
        "line": lineno,
    })

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(_bold("\n  Augmentum Dead Code & Coverage Check"))
    print(_bold("  " + "=" * 44) + "\n")

    _ensure_refs()

    # 1. Orphaned endpoints
    print(_cyan("  [1/4] Finding orphaned backend endpoints..."))
    orphaned = find_orphaned_endpoints()

    # 2. Ghost calls
    print(_cyan("  [2/4] Finding ghost frontend API calls..."))
    ghosts = find_ghost_calls()

    # 3. Untested routes
    print(_cyan("  [3/4] Checking test coverage for routes..."))
    untested = find_untested_routes()

    # 4. Import drift
    print(_cyan("  [4/4] Checking import/dependency alignment..."))
    dep_drift = find_missing_deps()

    # Apply the reviewed allowlist (dead_code_suppressions.json).
    sup = _load_suppressions()
    _g_sup, _o_sup = set(sup["ghost_calls"]), set(sup["orphaned_endpoints"])
    ghost_suppressed = [g for g in ghosts if f"{g['method']} {g['url'].split('?')[0]}" in _g_sup or f"{g['method']} {g['url']}" in _g_sup]
    ghosts = [g for g in ghosts if g not in ghost_suppressed]
    orphan_suppressed = [o for o in orphaned if f"{o['method']} {o['path']}" in _o_sup]
    orphaned = [o for o in orphaned if o not in orphan_suppressed]
    n_sup = len(ghost_suppressed) + len(orphan_suppressed)

    print()
    if n_sup:
        print(_dim(f"  ({n_sup} finding(s) suppressed via dead_code_suppressions.json)"))
        if "--verbose" in sys.argv or "-v" in sys.argv:
            for g in ghost_suppressed:
                print(_dim(f"    [suppressed ghost] {g['method']} {g['url']}  ({g['file']}:{g['line']})"))
            for o in orphan_suppressed:
                print(_dim(f"    [suppressed orphan] {o['method']} {o['path']}"))
        print()

    # Report
    has_findings = False

    if orphaned:
        has_findings = True
        print(_yellow(f"  Orphaned Endpoints ({len(orphaned)}) — backend routes with no frontend caller:"))
        for o in orphaned:
            loc = f"{o['file']}:{o['line']}"
            print(f"    {_yellow('~')} {o['method']:6s} {o['path']:45s} {_dim(loc)}")
        print()

    if ghosts:
        has_findings = True
        print(_red(f"  Ghost Calls ({len(ghosts)}) — frontend calls to non-existent endpoints:"))
        for g in ghosts:
            loc = f"{g['file']}:{g['line']}"
            print(f"    {_red('!')} {g['method']:4s} {g['url']:50s} {_dim(loc)}")
        print()

    if untested:
        has_findings = True
        print(_yellow(f"  Untested Routes ({len(untested)}) — route files with no test coverage:"))
        for u in untested:
            expected = f"expected: {u['expected_test']}"
            print(f"    {_yellow('~')} {u['route_file']:45s} {_dim(expected)}")
        print()

    if dep_drift:
        has_findings = True
        print(_dim(f"  Dependency Drift ({len(dep_drift)}) — imports not in pyproject.toml (may be transitive):"))
        for d in dep_drift:
            loc = f"({d['file']}:{d['line']})"
            print(f"    {_dim('-')} import {d['import']:20s} {_dim(loc)}")
        print()

    if not has_findings:
        print(_green("  All clean — no dead code, ghost calls, or coverage gaps."))

    print()
    return 1 if (ghosts) else 0  # Only ghost calls are errors; orphaned are warnings


if __name__ == "__main__":
    sys.exit(main())
