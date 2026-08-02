"""FastAPI framework adapter.

Two routes for discovering routes (no pun intended):

1. **Cached references** — when the codebase has a pre-built
   ``routes.json`` under ``.claude/skills/augmentum-dev/references/``
   (the augmentum-dev convention), we read it directly. Zero scan
   cost. This is the path used in Augmentum itself.

2. **Live AST scan** — for any FastAPI project that doesn't ship
   augmentum-dev caches, we walk the workspace and parse
   ``@router.<verb>('/path')`` decorators directly. Slower (~seconds
   for a 1000-file repo) but works on any FastAPI codebase the
   bug_finder is pointed at.

The adapter is the FIRST concrete adapter shipped — it's the path
through which the bug_finder generalizes to "any FastAPI codebase",
not just Augmentum. Flask / Django / Express adapters will follow
the same shape.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from augmentum.bug_finder.adapters.base import (
    AdapterRouteHint,
    AdapterSettingHint,
    FrameworkAdapter,
)


# Decorator attributes that indicate a route registration.
_ROUTE_VERBS = (
    "get", "post", "put", "delete", "patch",
    "head", "options", "websocket", "route",
)

# Source pattern used in fallback regex mode (when AST parse fails).
_ROUTE_RE = re.compile(
    r'@(?:[a-zA-Z_][\w.]*)\.'
    r'(' + "|".join(_ROUTE_VERBS) + r')'
    r'\(\s*[fr]?["\']([^"\']+)["\']',
)


class FastAPIAdapter(FrameworkAdapter):
    @property
    def name(self) -> str:
        return "fastapi"

    # ----- routes -----

    def list_routes(self, root: Path) -> list[AdapterRouteHint]:
        # Path 1: read the cached references file when present.
        cached = self._read_cached_routes(root)
        if cached:
            return cached
        # Path 2: live AST walk.
        return list(self._scan_routes(root))

    def _read_cached_routes(self, root: Path) -> list[AdapterRouteHint]:
        ref_path = root / ".claude" / "skills" / "augmentum-dev" / \
            "references" / "routes.json"
        if not ref_path.is_file():
            return []
        try:
            data = json.loads(ref_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        # routes.json shape (augmentum-dev convention): ``{"endpoints":
        # [...]}``. Also tolerate a bare list for forward-compat.
        if isinstance(data, dict):
            rows = data.get("endpoints") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        out: list[AdapterRouteHint] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            method = str(r.get("method") or "").strip().upper()
            path = str(r.get("path") or "").strip()
            handler = str(r.get("handler") or "").strip()
            file = str(r.get("file") or "").strip()
            if not (method and path and file):
                continue
            try:
                line = int(r.get("line") or 0)
            except (TypeError, ValueError):
                line = 0
            out.append(AdapterRouteHint(
                method=method, path=path,
                handler=handler or f"{file}:?", file=file, line=line,
            ))
        return out

    def _scan_routes(self, root: Path):
        """Walk every plausible route file under ``root`` and yield
        route hints from decorator AST nodes.

        Convention coverage:
          * ``*_routes.py`` / ``routes.py`` / ``router.py`` (Augmentum
            convention)
          * Any ``.py`` file under a directory named ``routes``,
            ``api``, ``endpoints``, or ``views`` (FastAPI starter
            convention — files named by domain, not suffix)

        Files matching neither convention are skipped — better than a
        full-tree walk on a 10K-file repo. Codebases with bespoke
        conventions can ship augmentum-dev's ``routes.json`` cache to
        bypass the live scan entirely.
        """
        candidates: list[Path] = []

        # Convention 1: suffixed/named route files
        for pattern in ("*_routes.py", "routes.py", "router.py"):
            candidates.extend(root.rglob(pattern))

        # Convention 2: files under a route-shaped directory
        _ROUTE_DIRS = {"routes", "api", "endpoints", "views"}
        _SKIP_DIRS = {
            ".git", ".venv", "venv", "node_modules", "__pycache__",
            ".pytest_cache", ".mypy_cache", ".ruff_cache",
            "dist", "build",
        }
        for path in root.rglob("*.py"):
            parts = path.parts
            if any(p in _SKIP_DIRS for p in parts):
                continue
            # Walk parent directories; include if any matches.
            if any(p.lower() in _ROUTE_DIRS for p in parts[:-1]):
                candidates.append(path)

        seen: set[Path] = set()
        for path in candidates:
            try:
                rp = path.resolve()
            except (OSError, RuntimeError):
                continue
            if rp in seen:
                continue
            seen.add(rp)
            yield from self._parse_routes_file(rp, root)

    def _parse_routes_file(self, path: Path, root: Path):
        """Parse one route-file with AST first, regex fallback."""
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        try:
            rel = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            rel = str(path)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            # Fallback: regex grep for the decorator shape so a single
            # broken file doesn't black-hole the whole adapter pass.
            for m in _ROUTE_RE.finditer(src):
                yield AdapterRouteHint(
                    method=m.group(1).upper(),
                    path=m.group(2),
                    handler=f"{rel}:?", file=rel,
                )
            return
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    hint = self._decorator_to_route_hint(
                        dec, node, rel,
                    )
                    if hint is not None:
                        yield hint

    @staticmethod
    def _decorator_to_route_hint(
        dec: ast.expr,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        rel_file: str,
    ) -> AdapterRouteHint | None:
        """Turn one decorator node into a route hint, or None."""
        # Strip Call wrapper — most route decorators are called
        target = dec.func if isinstance(dec, ast.Call) else dec
        if not isinstance(target, ast.Attribute):
            return None
        verb = target.attr.lower()
        if verb not in _ROUTE_VERBS:
            return None
        # First positional arg of the call carries the path
        path = ""
        if isinstance(dec, ast.Call) and dec.args:
            first = dec.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                path = first.value
        if not path:
            return None
        return AdapterRouteHint(
            method=verb.upper(),
            path=path,
            handler=f"{rel_file}:{func.name}",
            file=rel_file,
            line=func.lineno,
        )

    # ----- settings -----

    def list_settings_files(self, root: Path) -> list[AdapterSettingHint]:
        out: list[AdapterSettingHint] = []
        # Augmentum / FastAPI conventions
        for name, kind in (
            ("config.py",       "python_module"),
            ("settings.py",     "python_module"),
            ("pyproject.toml",  "toml"),
            (".env",            "env"),
            (".env.example",    "env"),
        ):
            for hit in root.rglob(name):
                if any(
                    seg in {".git", ".venv", "venv", "node_modules",
                            "__pycache__", "dist", "build"}
                    for seg in hit.parts
                ):
                    continue
                try:
                    rel = str(hit.relative_to(root)).replace("\\", "/")
                except ValueError:
                    rel = str(hit)
                out.append(AdapterSettingHint(
                    file=rel, kind=kind, name_hint=hit.stem,
                ))
        return out

    def identify_route_file(self, path: Path | str) -> bool:
        name = Path(path).name
        return (
            name.endswith("_routes.py")
            or name in {"routes.py", "router.py"}
        )

    def identify_test_command(self, root: Path) -> str:
        if (root / "pyproject.toml").is_file() or \
                (root / "pytest.ini").is_file() or \
                (root / "setup.cfg").is_file():
            return "pytest"
        return ""
