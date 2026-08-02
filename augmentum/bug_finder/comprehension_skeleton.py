"""Deterministic structural skeleton of a workspace.

Three runs of the LLM-only comprehender burned ~1.5M tokens producing
zero usable maps. The model kept reading without ever committing to a
structured output — a structural problem with "explore + synthesize"
that no prompt tightening fixes.

This module shifts the architecture: ~80% of the structural map is
mechanical (language, files, routes, settings, framework, test
command) and can be derived deterministically from container-side
commands. The LLM only handles the 20% requiring real judgment —
naming pillars (architectural invariants) and risk surfaces
(untrusted-input boundaries).

The skeleton runs in seconds, costs zero tokens, and is reliable.
The comprehender then ingests it as input — its budget drops from
500k → 80k tokens, its job from "discover everything" to "synthesize
from this brief".

The skeleton's compact text rendering is what gets injected into
the comprehender's user message. The structured dataclass is also
returned so downstream callers (orchestrator, persistence) can use
the same data without re-parsing.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from augmentum.coder.containers import ContainerManager
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsystemHint:
    """One directory cluster identified as a candidate subsystem.

    Shallow by design — the LLM is responsible for naming the pillars
    and writing the one-sentence purpose. We just give it the locations
    + a top-of-file docstring sniff.
    """

    path: str                  # repo-relative dir path
    file_count: int            # # of source files inside
    has_routes: bool = False   # any *_routes.py / router.X decorators
    top_docstring: str = ""    # first 200 chars of nearest __init__.py docstring


@dataclass(frozen=True)
class RouteHint:
    """One discovered HTTP route (or job / cli)."""

    method: str                # "GET", "POST", "JOB", "CLI", ...
    path: str                  # "/api/foo", or job name
    file: str                  # source file relative to workspace
    line: int = 0              # 0 if unknown


@dataclass(frozen=True)
class CodebaseSkeleton:
    """The deterministic shadow of a workspace's structure.

    Compact + ready to inject into a prompt verbatim. Fields are
    intentionally small (file_count caps, route caps) so the
    comprehender's input stays bounded even on a 10k-file repo.
    """

    workspace_root: str = "/workspace"
    languages: tuple[str, ...] = ()       # ranked by file count, top 5
    framework: str = ""                   # "fastapi" | "flask" | "django" | ...
    test_command: str = ""                # detected pytest / npm test / etc.
    file_count_total: int = 0
    source_file_count: int = 0
    has_git: bool = False
    head_sha: str = ""

    subsystems: tuple[SubsystemHint, ...] = ()
    routes: tuple[RouteHint, ...] = ()
    background_jobs: tuple[str, ...] = ()
    settings_files: tuple[str, ...] = ()   # config.py / settings.json / .env / etc.
    candidate_pillars_files: tuple[str, ...] = ()
        # Files the comprehender should sample to identify pillars —
        # picked by structural heuristics (large central files, top
        # __init__.py, decorator-heavy auth/security modules).

    discovery_notes: tuple[str, ...] = ()  # advisory text for the LLM

    def render_for_prompt(self) -> str:
        """Compact text form for injection into the comprehender's user
        message. Stable shape so prompts don't drift run-to-run."""
        lines: list[str] = [
            "## Deterministic skeleton (pre-computed — trust this)",
            "",
            f"**Workspace root:** {self.workspace_root}",
        ]
        if self.languages:
            lines.append(f"**Languages:** {', '.join(self.languages[:5])}")
        if self.framework:
            lines.append(f"**Framework:** {self.framework}")
        if self.test_command:
            lines.append(f"**Test command:** `{self.test_command}`")
        lines.append(f"**Files (total / source):** "
                     f"{self.file_count_total:,} / {self.source_file_count:,}")
        if self.head_sha:
            lines.append(f"**HEAD:** `{self.head_sha[:10]}`")

        if self.subsystems:
            lines.append("\n### Subsystem candidates")
            for s in self.subsystems[:40]:
                routes_marker = "  [routes]" if s.has_routes else ""
                lines.append(
                    f"- `{s.path}` — {s.file_count} files{routes_marker}"
                )
                if s.top_docstring:
                    snippet = s.top_docstring.strip().replace("\n", " ")[:180]
                    lines.append(f"    > {snippet}")

        if self.routes:
            lines.append("\n### HTTP routes (deterministic)")
            for r in self.routes[:60]:
                lines.append(
                    f"- `{r.method:6s} {r.path}` — {r.file}"
                    + (f":{r.line}" if r.line else "")
                )
            if len(self.routes) > 60:
                lines.append(f"  ... and {len(self.routes) - 60} more")

        if self.background_jobs:
            lines.append("\n### Background jobs")
            for j in self.background_jobs[:20]:
                lines.append(f"- `{j}`")

        if self.settings_files:
            lines.append("\n### Config / settings files")
            for s in self.settings_files[:10]:
                lines.append(f"- `{s}`")

        if self.candidate_pillars_files:
            lines.append("\n### Files worth reading for pillar inference")
            for f in self.candidate_pillars_files[:15]:
                lines.append(f"- `{f}`")

        if self.discovery_notes:
            lines.append("\n### Notes from the skeleton builder")
            for n in self.discovery_notes:
                lines.append(f"- {n}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Container-side commands
# ---------------------------------------------------------------------------

# Common source-code extensions ranked by likelihood of being primary.
# Order matters: the first match in this list wins the language label.
_LANGUAGE_EXT: tuple[tuple[str, str], ...] = (
    ("py",   "python"),
    ("ts",   "typescript"),
    ("tsx",  "typescript"),
    ("js",   "javascript"),
    ("jsx",  "javascript"),
    ("go",   "go"),
    ("rs",   "rust"),
    ("java", "java"),
    ("kt",   "kotlin"),
    ("c",    "c"),
    ("cpp",  "c++"),
    ("h",    "c"),
    ("hpp",  "c++"),
    ("rb",   "ruby"),
    ("php",  "php"),
    ("cs",   "csharp"),
    ("swift", "swift"),
    ("sh",   "shell"),
)


# Skip directories that don't contribute structural information.
_SKIP_DIRS = (
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    "dist", "build", ".next", ".nuxt", "target",
)


# Framework signatures — order matters; first match wins.
_FRAMEWORK_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fastapi",  ("from fastapi", "import fastapi", "FastAPI(")),
    ("django",   ("from django", "django.urls", "INSTALLED_APPS")),
    ("flask",    ("from flask import", "Flask(__name__)")),
    ("express",  ("require('express')", 'require("express")', "import express")),
    ("nextjs",   ("next/app", "next.config", "next/router")),
    ("nestjs",   ("@nestjs/", "@Module(")),
    ("rails",    ("Rails.application", "config/routes.rb")),
    ("spring",   ("@SpringBootApplication", "org.springframework")),
)


async def _exec(
    cm: ContainerManager, workspace_id: str, cmd: str,
    *, timeout: float = 20.0,
) -> str:
    """Run a shell command in the container, return stdout (or "")."""
    try:
        return await cm.run_command(
            workspace_id, ["bash", "-lc", cmd], timeout=timeout,
        )
    except Exception:  # noqa: BLE001 — skeleton is best-effort
        return ""


async def _detect_languages(
    cm: ContainerManager, workspace_id: str, root: str,
) -> tuple[tuple[str, ...], int, int]:
    """Return (ranked-languages, source_file_count, total_file_count)."""
    # Count files per known extension; skip noisy dirs.
    skip_args = " ".join(
        f"-not -path '{root}/{d}/*' -not -path '*/{d}/*'"
        for d in _SKIP_DIRS
    )
    counts: dict[str, int] = {}
    source_total = 0
    for ext, lang in _LANGUAGE_EXT:
        out = await _exec(
            cm, workspace_id,
            f"find {shlex.quote(root)} -type f -name '*.{ext}' "
            f"{skip_args} 2>/dev/null | wc -l",
            timeout=15.0,
        )
        try:
            n = int(out.strip() or "0")
        except ValueError:
            n = 0
        if n > 0:
            counts[lang] = counts.get(lang, 0) + n
            source_total += n
    # Total file count — for the "files (total / source)" report
    total_out = await _exec(
        cm, workspace_id,
        f"find {shlex.quote(root)} -type f "
        f"-not -path '{root}/.git/*' 2>/dev/null | wc -l",
        timeout=20.0,
    )
    try:
        total = int(total_out.strip() or "0")
    except ValueError:
        total = 0
    # Rank by count, take top 5
    ranked = tuple(
        lang for lang, _n in sorted(
            counts.items(), key=lambda kv: -kv[1],
        )[:5]
    )
    return ranked, source_total, total


async def _detect_framework(
    cm: ContainerManager, workspace_id: str, root: str,
) -> str:
    """Grep for framework-defining import patterns. First match wins."""
    for name, signatures in _FRAMEWORK_SIGNATURES:
        for sig in signatures:
            # ``-q`` returns 0 on first match — bail fast.
            cmd = (
                f"grep -r -l --max-count=1 {shlex.quote(sig)} "
                f"{shlex.quote(root)} "
                + " ".join(f"--exclude-dir={d}" for d in _SKIP_DIRS)
                + " 2>/dev/null | head -1"
            )
            out = await _exec(cm, workspace_id, cmd, timeout=20.0)
            if out.strip():
                return name
    return ""


async def _detect_test_command(
    cm: ContainerManager, workspace_id: str, root: str,
) -> str:
    """Sniff the most likely test command based on present files."""
    checks: tuple[tuple[str, str], ...] = (
        ("pyproject.toml",   "pytest"),
        ("pytest.ini",       "pytest"),
        ("setup.cfg",        "pytest"),
        ("package.json",     "npm test"),
        ("go.mod",           "go test ./..."),
        ("Cargo.toml",       "cargo test"),
        ("Gemfile",          "bundle exec rspec"),
    )
    for fname, cmd in checks:
        out = await _exec(
            cm, workspace_id,
            f"test -f {shlex.quote(root)}/{fname} && echo found",
            timeout=5.0,
        )
        if "found" in out:
            return cmd
    return ""


async def _detect_subsystems(
    cm: ContainerManager, workspace_id: str, root: str,
) -> tuple[SubsystemHint, ...]:
    """Find depth-2 dirs under root + read their __init__.py docstrings."""
    # Depth-2 dirs (e.g. augmentum/auth, augmentum/coder, etc.)
    skip_args = " ".join(
        f"-not -path '*/{d}*'" for d in _SKIP_DIRS
    )
    out = await _exec(
        cm, workspace_id,
        f"find {shlex.quote(root)} -mindepth 2 -maxdepth 3 -type d "
        f"{skip_args} 2>/dev/null | head -60",
        timeout=15.0,
    )
    candidates = [
        line.strip().removeprefix(root + "/").removeprefix("./")
        for line in (out or "").splitlines()
        if line.strip() and line.strip() != root
    ]

    subsystems: list[SubsystemHint] = []
    for path in candidates:
        if not path:
            continue
        # File count for this subsystem (any source extension)
        ext_filters = " ".join(
            f"-o -name '*.{ext}'" for ext, _ in _LANGUAGE_EXT
        )
        # Strip the leading `-o ` from the first filter
        if ext_filters.startswith("-o "):
            ext_filters = ext_filters[3:]
        count_out = await _exec(
            cm, workspace_id,
            f"find {shlex.quote(root)}/{path} -type f \\( {ext_filters} \\) "
            f"2>/dev/null | wc -l",
            timeout=10.0,
        )
        try:
            file_count = int(count_out.strip() or "0")
        except ValueError:
            file_count = 0
        if file_count == 0:
            continue
        # Has routes?
        routes_out = await _exec(
            cm, workspace_id,
            f"find {shlex.quote(root)}/{path} -name '*_routes.py' "
            f"-o -name 'routes.py' -o -name 'urls.py' 2>/dev/null | head -1",
            timeout=5.0,
        )
        has_routes = bool(routes_out.strip())
        # __init__.py docstring sniff
        init_path = f"{root}/{path}/__init__.py"
        docstring = await _exec(
            cm, workspace_id,
            f"head -25 {shlex.quote(init_path)} 2>/dev/null",
            timeout=5.0,
        )
        # Extract docstring text — match the first triple-quoted block
        doc_match = re.search(
            r'"{3}(.+?)"{3}|\'{3}(.+?)\'{3}',
            docstring, re.DOTALL,
        )
        doc_text = ""
        if doc_match:
            doc_text = (doc_match.group(1) or doc_match.group(2) or "").strip()
        subsystems.append(SubsystemHint(
            path=path,
            file_count=file_count,
            has_routes=has_routes,
            top_docstring=doc_text[:200],
        ))
    # Sort by file count descending so the biggest subsystems lead the list
    subsystems.sort(key=lambda s: -s.file_count)
    return tuple(subsystems[:40])


_ROUTE_DECORATOR_RE = re.compile(
    r'@(?:app|router|api[\w_]*|[a-zA-Z_][\w]*)\.'
    r'(get|post|put|delete|patch|head|options|websocket|route)\('
    r'\s*[fr]?["\']([^"\']+)["\']',
)


async def _detect_routes(
    cm: ContainerManager, workspace_id: str, root: str,
) -> tuple[RouteHint, ...]:
    """Grep route files for decorator patterns, parse into RouteHints.

    Bounded to keep token cost predictable: at most ~150 routes from
    at most ~60 files. Repos with more routes will surface them via
    the per-subsystem hints instead.
    """
    # Find candidate route files
    route_files_out = await _exec(
        cm, workspace_id,
        f"find {shlex.quote(root)} \\( -name '*_routes.py' "
        f"-o -name 'routes.py' -o -name 'router.py' "
        f"-o -name 'urls.py' \\) "
        + " ".join(f"-not -path '*/{d}/*'" for d in _SKIP_DIRS)
        + " 2>/dev/null | head -60",
        timeout=15.0,
    )
    files = [
        f.strip().removeprefix(root + "/").removeprefix("./")
        for f in (route_files_out or "").splitlines()
        if f.strip()
    ]
    routes: list[RouteHint] = []
    for rel in files[:60]:
        path = f"{root}/{rel}" if not rel.startswith("/") else rel
        # Grep with line numbers
        out = await _exec(
            cm, workspace_id,
            f"grep -n -E '@[a-zA-Z_][a-zA-Z0-9_]*\\.(get|post|put|delete|patch"
            f"|head|options|websocket|route)\\(' {shlex.quote(path)} "
            f"2>/dev/null | head -40",
            timeout=10.0,
        )
        for line in (out or "").splitlines():
            # Format: "LINE:DECORATOR_TEXT"
            if ":" not in line:
                continue
            line_num_str, body = line.split(":", 1)
            try:
                line_num = int(line_num_str)
            except ValueError:
                continue
            m = _ROUTE_DECORATOR_RE.search(body)
            if not m:
                continue
            method = m.group(1).upper()
            route_path = m.group(2)
            routes.append(RouteHint(
                method=method, path=route_path,
                file=rel, line=line_num,
            ))
            if len(routes) >= 150:
                break
        if len(routes) >= 150:
            break
    return tuple(routes)


async def _detect_settings_files(
    cm: ContainerManager, workspace_id: str, root: str,
) -> tuple[str, ...]:
    """Locate likely config / settings files."""
    candidates = (
        "config.py", "settings.py", "settings.json", "settings.yaml",
        ".env.example", "pyproject.toml", "package.json", "tsconfig.json",
        "go.mod", "Cargo.toml",
    )
    found: list[str] = []
    for fname in candidates:
        out = await _exec(
            cm, workspace_id,
            f"find {shlex.quote(root)} -name {shlex.quote(fname)} "
            + " ".join(f"-not -path '*/{d}/*'" for d in _SKIP_DIRS)
            + " 2>/dev/null | head -3",
            timeout=10.0,
        )
        for line in (out or "").splitlines():
            rel = line.strip().removeprefix(root + "/").removeprefix("./")
            if rel and rel not in found:
                found.append(rel)
    return tuple(found[:15])


async def _detect_git_head(
    cm: ContainerManager, workspace_id: str, root: str,
) -> tuple[bool, str]:
    """Return (has_git, head_sha)."""
    out = await _exec(
        cm, workspace_id,
        f"cd {shlex.quote(root)} && git rev-parse HEAD 2>/dev/null",
        timeout=10.0,
    )
    sha = (out or "").strip().splitlines()[0] if out else ""
    return bool(sha), sha


def _pick_candidate_pillar_files(
    subsystems: tuple[SubsystemHint, ...],
    routes: tuple[RouteHint, ...],
    settings_files: tuple[str, ...],
    framework: str,
) -> tuple[str, ...]:
    """Pick ~10 files the comprehender should read for pillar inference.

    Heuristic: pick the __init__.py of the top 5 subsystems by file
    count, plus 1-2 route files (where security pillars often live),
    plus the main config/settings file.
    """
    out: list[str] = []
    for s in subsystems[:5]:
        out.append(f"{s.path}/__init__.py")
    # First couple of route files — auth/security pillars often anchor here
    seen_files: set[str] = set()
    for r in routes:
        if r.file not in seen_files:
            out.append(r.file)
            seen_files.add(r.file)
        if len(seen_files) >= 2:
            break
    # Main settings file (python or node)
    for cfg in settings_files:
        if cfg in ("config.py", "settings.py", "package.json", "pyproject.toml"):
            out.append(cfg)
            break
    # Deduplicate while preserving order
    deduped: list[str] = []
    seen: set[str] = set()
    for path in out:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return tuple(deduped[:12])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def build_skeleton(
    *,
    cm: ContainerManager,
    workspace_id: str,
    root: str = "/workspace",
) -> CodebaseSkeleton:
    """Build a deterministic structural skeleton of the workspace.

    All work runs container-side via shell commands. Errors are
    swallowed silently — a partial skeleton (e.g. routes empty
    because grep failed) is more useful than an exception that
    breaks the comprehension flow. Discovery notes capture what
    fell through so the comprehender can compensate.
    """
    notes: list[str] = []

    languages, source_total, file_total = await _detect_languages(
        cm, workspace_id, root,
    )
    if not languages:
        notes.append("no source files matched a known language extension")

    framework = await _detect_framework(cm, workspace_id, root)
    test_command = await _detect_test_command(cm, workspace_id, root)
    has_git, head_sha = await _detect_git_head(cm, workspace_id, root)

    subsystems = await _detect_subsystems(cm, workspace_id, root)
    routes = await _detect_routes(cm, workspace_id, root)
    settings_files = await _detect_settings_files(cm, workspace_id, root)

    candidate_files = _pick_candidate_pillar_files(
        subsystems, routes, settings_files, framework,
    )

    if not routes and framework in ("fastapi", "flask", "django", "express"):
        notes.append(
            f"framework={framework} detected but no routes found — "
            "the route file naming may be non-standard",
        )

    skeleton = CodebaseSkeleton(
        workspace_root=root,
        languages=languages,
        framework=framework,
        test_command=test_command,
        file_count_total=file_total,
        source_file_count=source_total,
        has_git=has_git,
        head_sha=head_sha,
        subsystems=subsystems,
        routes=routes,
        background_jobs=(),  # Phase 2 — parse job-registration patterns
        settings_files=settings_files,
        candidate_pillars_files=candidate_files,
        discovery_notes=tuple(notes),
    )
    log.info(
        "bug_finder_skeleton_built",
        workspace_id=workspace_id,
        languages=list(languages),
        framework=framework,
        source_files=source_total,
        subsystems=len(subsystems),
        routes=len(routes),
        notes=len(notes),
    )
    return skeleton
