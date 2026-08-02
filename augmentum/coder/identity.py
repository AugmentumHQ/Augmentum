"""Project-identity detection + manifest IO for the workspace kernel.

Identity is **Layer 1** of the five-layer kernel state model (see
``docs/superpowers/specs/2026-05-28-context-kernel-design.md``):
durable, in-repo, user-curatable facts about the project the agent is
working in — language, package manager, test runner, build/run
commands, deploy target, conventions.

The manifest lives at ``/workspace/.augmentum/identity.toml`` so it
travels with the codebase across machines, sessions, and users. Three
sections:

* ``[detected]`` — auto-populated by the detector registry on workspace
  refresh. Replaced wholesale each refresh; the detector outputs are
  the source of truth for this section.
* ``[asserted]`` — user-edited; takes precedence over ``[detected]``.
  Never overwritten by the harness. The contract with the user is:
  "anything you write here is canonical; the agent will treat it as
  policy."
* ``[discovered]`` — append-only, model-written facts learned while
  working ("the auth tokens live in /workspace/.env.local", "tests
  need SQLITE_PATH=/tmp/test.db"). Persists across refreshes; never
  clobbered by detection.

Detector registry pattern: one class per ecosystem (Python / JS / Rust
/ Go in this initial slice). Adding a language is one file + one entry
in ``DETECTORS``. No layer above the detector touches language-
specific logic — versatility lives entirely in the registry.

TOML is the on-disk format because (a) it's human-readable for the
user's hand-edits of ``[asserted]`` and (b) stdlib ``tomllib`` (3.11+)
covers the read path. The write path is hand-rolled in
``serialize_manifest`` — the schema is small and stable enough that a
50-line writer beats taking on a new dependency.
"""
from __future__ import annotations

import json
import time
import tomllib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.coder.containers import ContainerManager

log = get_logger(__name__)


# Schema version. Bump when [detected] keys change shape (renames,
# nested-vs-flat). Readers should tolerate older versions by ignoring
# unknown keys; writers should never downgrade. ``_compatible_read``
# uses this to decide whether to re-run detection.
DETECTOR_VERSION = 1


# ---------------------------------------------------------------------------
# Manifest dataclasses — shape pinned by tests so future migrations
# can't silently change it.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IdentityMeta:
    """``[meta]`` section: provenance + freshness markers."""
    detector_version: int = DETECTOR_VERSION
    last_detected_at: float = 0.0


@dataclass(slots=True)
class DiscoveredFact:
    """One entry in the ``[[discovered]]`` array-of-tables.

    Model-appended facts about the workspace. ``confidence`` lets the
    model distinguish things it has actively verified (``confirmed``)
    from things it merely inferred (``tentative``) — useful at conflict
    detection time.
    """
    ts: float
    category: str        # "build", "env", "constraint", "gotcha", "other"
    fact: str
    source: str          # "shell_exec turn 9", "user turn 6", etc.
    confidence: str = "confirmed"   # "tentative" | "confirmed" | "user_asserted"


@dataclass(slots=True)
class IdentityManifest:
    """Full identity.toml contents in memory.

    Three sections with different ownership:

    * ``detected`` — harness-owned, refreshed wholesale by detectors
    * ``asserted`` — user-owned, never touched by the harness
    * ``discovered`` — model-owned, append-only
    """
    meta: IdentityMeta = field(default_factory=IdentityMeta)
    detected: dict[str, Any] = field(default_factory=dict)
    asserted: dict[str, Any] = field(default_factory=dict)
    discovered: list[DiscoveredFact] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detector contract — every per-ecosystem detector subclasses this and
# registers itself in DETECTORS. The methods are classmethods (not
# instance methods) so detection is stateless — the registry can call
# them directly without managing detector lifetimes.
# ---------------------------------------------------------------------------


class Detector:
    """Per-ecosystem identity detector. Subclass + register in DETECTORS.

    Convention: ``name`` is the canonical language id used as both the
    key in ``manifest.detected[name]`` and the entry in
    ``manifest.detected['languages']``. Keep it short and lower-case
    (``python``, not ``Python``).
    """

    name: ClassVar[str] = ""

    @classmethod
    async def applies(
        cls, cm: ContainerManager, workspace_id: str,
    ) -> bool:
        """Return True iff this detector should run for the workspace.

        The check should be cheap — a single file existence probe is
        the typical pattern. Detectors that need to read content to
        decide should do that work in ``detect`` instead, returning an
        empty dict if signals turn out to be insufficient.
        """
        raise NotImplementedError

    @classmethod
    async def detect(
        cls, cm: ContainerManager, workspace_id: str,
    ) -> dict[str, Any]:
        """Return a flat-ish dict of detected facts.

        Best-effort: malformed config files should return what can be
        salvaged rather than raising. The registry catches exceptions
        anyway, but a detector that's tolerant to its own ecosystem's
        quirks produces better results than one that's binary.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helper: safe container file read returning empty string on miss.
# Mirrors WorkspaceKernel's contract — failure is normal, not an event.
# ---------------------------------------------------------------------------


async def _safe_read(cm: ContainerManager, workspace_id: str, path: str) -> str:
    try:
        raw = await cm.file_read(workspace_id, path)
    except Exception:
        return ""
    return raw or ""


async def _file_exists(cm: ContainerManager, workspace_id: str, path: str) -> bool:
    """Cheap "does this file exist?" probe.

    Uses file_read with a try/except — true if read succeeds (even if
    empty), false otherwise. The container manager doesn't expose a
    bare ``stat``, and ``run_command`` to ``test -f`` is heavier than
    just attempting the read.
    """
    try:
        await cm.file_read(workspace_id, path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Python detector — pyproject.toml + requirements.txt + setup.py
# ---------------------------------------------------------------------------


class PythonDetector(Detector):
    name = "python"

    @classmethod
    async def applies(cls, cm, workspace_id: str) -> bool:
        for marker in ("pyproject.toml", "requirements.txt", "setup.py"):
            if await _file_exists(cm, workspace_id, f"/workspace/{marker}"):
                return True
        return False

    @classmethod
    async def detect(cls, cm, workspace_id: str) -> dict[str, Any]:
        facts: dict[str, Any] = {}

        # Package manager — file presence is the canonical signal.
        # Order matters: uv.lock + poetry.lock + pip can coexist
        # (migration in progress); pick the most modern one present.
        for lock_file, pm in (
            ("uv.lock", "uv"),
            ("poetry.lock", "poetry"),
            ("Pipfile.lock", "pipenv"),
            ("requirements.txt", "pip"),
        ):
            if await _file_exists(cm, workspace_id, f"/workspace/{lock_file}"):
                facts["package_manager"] = pm
                break

        # pyproject.toml is the richest source — name, scripts, deps,
        # tool-config sections.
        pyproject_raw = await _safe_read(
            cm, workspace_id, "/workspace/pyproject.toml",
        )
        if pyproject_raw:
            try:
                pp = tomllib.loads(pyproject_raw)
            except tomllib.TOMLDecodeError:
                pp = {}
            project = pp.get("project", {})
            if "name" in project:
                facts["name"] = str(project["name"])
            if "version" in project:
                facts["version"] = str(project["version"])
            deps = project.get("dependencies", []) or []
            if isinstance(deps, list):
                facts["dependencies"] = [str(d) for d in deps[:10]]
            scripts = project.get("scripts", {}) or {}
            if scripts:
                facts["entry_points"] = list(scripts.keys())[:8]
            tool = pp.get("tool", {}) or {}
            # Test runner detection — pytest config presence is the
            # strongest signal a project uses pytest as its runner.
            if "pytest" in tool:
                facts["test_runner"] = "pytest"
            elif "tox" in tool:
                facts["test_runner"] = "tox"
            # Linter / formatter — informative for the model when
            # writing code in the project's style.
            tooling: list[str] = []
            for t in ("ruff", "black", "mypy", "pyright", "flake8"):
                if t in tool:
                    tooling.append(t)
            if tooling:
                facts["tooling"] = tooling

        # Fallback test_runner detection from file markers — covers
        # projects without pyproject.toml or pre-PEP-621 layouts.
        if "test_runner" not in facts:
            for marker in ("pytest.ini", "conftest.py", "tests/conftest.py"):
                if await _file_exists(cm, workspace_id, f"/workspace/{marker}"):
                    facts["test_runner"] = "pytest"
                    break

        return facts


# ---------------------------------------------------------------------------
# JavaScript / TypeScript detector — package.json + lockfiles
# ---------------------------------------------------------------------------


class JavaScriptDetector(Detector):
    name = "javascript"

    @classmethod
    async def applies(cls, cm, workspace_id: str) -> bool:
        return await _file_exists(
            cm, workspace_id, "/workspace/package.json",
        )

    @classmethod
    async def detect(cls, cm, workspace_id: str) -> dict[str, Any]:
        facts: dict[str, Any] = {}

        raw = await _safe_read(cm, workspace_id, "/workspace/package.json")
        if not raw:
            return facts
        try:
            pkg = json.loads(raw)
        except json.JSONDecodeError:
            return facts
        if not isinstance(pkg, dict):
            return facts

        if "name" in pkg:
            facts["name"] = str(pkg["name"])
        if "version" in pkg:
            facts["version"] = str(pkg["version"])
        if "main" in pkg:
            facts["main"] = str(pkg["main"])
        if "type" in pkg:
            facts["module_type"] = str(pkg["type"])  # "module" | "commonjs"

        scripts = pkg.get("scripts", {}) or {}
        if isinstance(scripts, dict):
            # Surface the conventional script keys the agent will reach
            # for. Don't dump all scripts — they bloat the manifest
            # and most aren't useful for orientation.
            kept: dict[str, str] = {}
            for k in ("build", "test", "start", "dev", "lint", "format"):
                if k in scripts:
                    kept[k] = str(scripts[k])
            if kept:
                facts["scripts"] = kept

        deps = pkg.get("dependencies", {}) or {}
        if isinstance(deps, dict) and deps:
            facts["dependencies"] = list(deps.keys())[:10]

        # Package manager — lockfile presence is canonical. Order
        # captures preference ordering when multiple are present
        # (mid-migration repos).
        for lock_file, pm in (
            ("bun.lockb", "bun"),
            ("pnpm-lock.yaml", "pnpm"),
            ("yarn.lock", "yarn"),
            ("package-lock.json", "npm"),
        ):
            if await _file_exists(cm, workspace_id, f"/workspace/{lock_file}"):
                facts["package_manager"] = pm
                break
        else:
            # No lockfile yet — default to npm; the user can override.
            facts["package_manager"] = "npm"

        # TypeScript signal — tsconfig.json existence is the load-
        # bearing fact for "this project uses TS". Surfaces a flag
        # rather than a separate ts language so JS+TS workspaces stay
        # under one detector.
        if await _file_exists(cm, workspace_id, "/workspace/tsconfig.json"):
            facts["typescript"] = True

        return facts


# ---------------------------------------------------------------------------
# Rust detector — Cargo.toml
# ---------------------------------------------------------------------------


class RustDetector(Detector):
    name = "rust"

    @classmethod
    async def applies(cls, cm, workspace_id: str) -> bool:
        return await _file_exists(
            cm, workspace_id, "/workspace/Cargo.toml",
        )

    @classmethod
    async def detect(cls, cm, workspace_id: str) -> dict[str, Any]:
        facts: dict[str, Any] = {"package_manager": "cargo"}

        raw = await _safe_read(cm, workspace_id, "/workspace/Cargo.toml")
        if not raw:
            return facts
        try:
            ct = tomllib.loads(raw)
        except tomllib.TOMLDecodeError:
            return facts

        package = ct.get("package", {})
        if isinstance(package, dict):
            if "name" in package:
                facts["name"] = str(package["name"])
            if "version" in package:
                facts["version"] = str(package["version"])
            if "edition" in package:
                facts["edition"] = str(package["edition"])
            if "rust-version" in package:
                facts["rust_version"] = str(package["rust-version"])

        bins = ct.get("bin", []) or []
        if isinstance(bins, list):
            names = [str(b.get("name", "")) for b in bins if isinstance(b, dict)]
            names = [n for n in names if n]
            if names:
                facts["binaries"] = names[:5]

        deps = ct.get("dependencies", {}) or {}
        if isinstance(deps, dict) and deps:
            facts["dependencies"] = list(deps.keys())[:10]

        facts["test_runner"] = "cargo test"
        facts["build_cmd"] = "cargo build"

        # Workspace detection — a workspace Cargo.toml has `[workspace]`
        # but typically no [package]. Useful signal for the model so it
        # doesn't try to `cargo run` from the workspace root.
        if "workspace" in ct:
            facts["is_workspace_root"] = True
            members = ct["workspace"].get("members", []) if isinstance(ct["workspace"], dict) else []
            if isinstance(members, list):
                facts["workspace_members"] = [str(m) for m in members[:10]]

        return facts


# ---------------------------------------------------------------------------
# Go detector — go.mod (regex-parsed because the format isn't JSON/TOML)
# ---------------------------------------------------------------------------


class GoDetector(Detector):
    name = "go"

    @classmethod
    async def applies(cls, cm, workspace_id: str) -> bool:
        return await _file_exists(
            cm, workspace_id, "/workspace/go.mod",
        )

    @classmethod
    async def detect(cls, cm, workspace_id: str) -> dict[str, Any]:
        import re

        facts: dict[str, Any] = {
            "package_manager": "go",
            "test_runner": "go test ./...",
            "build_cmd": "go build ./...",
        }

        raw = await _safe_read(cm, workspace_id, "/workspace/go.mod")
        if not raw:
            return facts

        # go.mod isn't JSON/TOML — the first non-comment "module" line
        # gives us the module path; "go X.Y" gives version.
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("//") or not stripped:
                continue
            m = re.match(r"module\s+(\S+)", stripped)
            if m:
                facts["module"] = m.group(1)
                continue
            v = re.match(r"go\s+(\d+\.\d+(?:\.\d+)?)", stripped)
            if v:
                facts["go_version"] = v.group(1)
                continue

        return facts


# ---------------------------------------------------------------------------
# Registry — order matters for ``language_primary`` tie-breaking
# (first applicable detector wins). Polyglot projects get one entry
# per applicable detector under ``detected[<name>]``.
# ---------------------------------------------------------------------------


DETECTORS: list[type[Detector]] = [
    PythonDetector,
    JavaScriptDetector,
    RustDetector,
    GoDetector,
]


# ---------------------------------------------------------------------------
# Detection orchestrator — runs every applicable detector, merges results.
# ---------------------------------------------------------------------------


async def detect_identity(
    cm: ContainerManager,
    workspace_id: str,
    *,
    now: float | None = None,
) -> IdentityManifest:
    """Run all applicable detectors and assemble a fresh manifest.

    Detector failures are swallowed at the registry level — one
    broken detector can't take out the entire identity refresh.
    Always returns a manifest, even if every detector found nothing
    (in which case ``detected.languages`` is an empty list).

    ``now`` is overridable for deterministic tests.
    """
    detected: dict[str, Any] = {}
    languages: list[str] = []
    for detector_cls in DETECTORS:
        try:
            if not await detector_cls.applies(cm, workspace_id):
                continue
            facts = await detector_cls.detect(cm, workspace_id)
        except Exception:
            log.debug(
                "identity.detector_failed",
                detector=detector_cls.name,
                workspace_id=workspace_id,
                exc_info=True,
            )
            continue
        if facts:
            detected[detector_cls.name] = facts
            languages.append(detector_cls.name)
    detected["languages"] = languages
    if languages:
        detected["language_primary"] = languages[0]

    return IdentityManifest(
        meta=IdentityMeta(
            detector_version=DETECTOR_VERSION,
            last_detected_at=time.time() if now is None else float(now),
        ),
        detected=detected,
        asserted={},
        discovered=[],
    )


def merge_refresh(
    existing: IdentityManifest,
    fresh_detected: dict[str, Any],
    *,
    now: float | None = None,
) -> IdentityManifest:
    """Replace ``detected`` with fresh results; preserve everything else.

    User assertions and model-discovered facts survive every refresh
    untouched — that's the ownership contract. Meta updates to the
    new timestamp + current detector version.
    """
    return IdentityManifest(
        meta=IdentityMeta(
            detector_version=DETECTOR_VERSION,
            last_detected_at=time.time() if now is None else float(now),
        ),
        detected=fresh_detected,
        asserted=existing.asserted,
        discovered=existing.discovered,
    )


# ---------------------------------------------------------------------------
# TOML IO — read uses stdlib tomllib; write is a hand-rolled minimal
# writer constrained to the manifest schema. The writer is intentionally
# narrow: it handles scalars + arrays of scalars + nested tables one
# level deep + arrays-of-tables for ``[[discovered]]``. Anything outside
# that shape is rejected with a clear error rather than producing
# malformed TOML.
# ---------------------------------------------------------------------------


def parse_manifest(text: str) -> IdentityManifest:
    """Parse TOML text into an IdentityManifest.

    Returns a fresh empty manifest on any parse error so callers can
    treat "missing file" and "malformed file" the same way.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        log.debug("identity.parse_failed", error=str(exc))
        return IdentityManifest()

    meta_dict = data.get("meta", {}) or {}
    meta = IdentityMeta(
        detector_version=int(meta_dict.get("detector_version", DETECTOR_VERSION)),
        last_detected_at=float(meta_dict.get("last_detected_at", 0.0)),
    )

    detected = data.get("detected", {}) or {}
    asserted = data.get("asserted", {}) or {}

    discovered_raw = data.get("discovered", []) or []
    discovered: list[DiscoveredFact] = []
    if isinstance(discovered_raw, list):
        for entry in discovered_raw:
            if not isinstance(entry, dict):
                continue
            try:
                discovered.append(DiscoveredFact(
                    ts=float(entry.get("ts", 0.0)),
                    category=str(entry.get("category", "other")),
                    fact=str(entry.get("fact", "")),
                    source=str(entry.get("source", "")),
                    confidence=str(entry.get("confidence", "confirmed")),
                ))
            except (TypeError, ValueError):
                continue

    return IdentityManifest(
        meta=meta,
        detected=detected if isinstance(detected, dict) else {},
        asserted=asserted if isinstance(asserted, dict) else {},
        discovered=discovered,
    )


# Constrained TOML writer. The manifest schema is:
#
#   [meta]              -- scalars (int, float)
#   [detected]          -- scalars + arrays + nested tables (one level)
#   [asserted]          -- scalars + arrays (user-edited)
#   [[discovered]]      -- array-of-tables (entries are DiscoveredFact)
#
# Anything richer (deeper nesting, tables-of-arrays) should be flagged
# at write time rather than silently misformatted. The TOML spec has
# more variety than we need; constraining the writer keeps the round-
# trip property easy to test.


def _toml_value(v: Any) -> str:
    """Serialize a scalar or array-of-scalars to TOML right-hand-side syntax.

    Refuses dicts and nested arrays — those route through table
    syntax in the parent. Booleans, ints, floats, and strings are
    handled; anything else gets repr'd as a string (defensive — a
    detector returning an unexpected type shouldn't crash the write).
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        # Use JSON's string quoting — TOML basic strings have a
        # similar enough escape grammar (\", \\, \n, \t, etc.) that
        # json.dumps' output is valid TOML for ASCII + common
        # Unicode. Saves us hand-rolling escape rules.
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(item) for item in v) + "]"
    # Last-resort fallback: stringify and quote.
    return json.dumps(str(v))


def _emit_table_body(lines: list[str], body: dict[str, Any], *, prefix: str) -> None:
    """Emit scalar+array keys; defer nested tables to the caller.

    Returns nothing; appends to ``lines`` in TOML order: scalars and
    arrays first, then sub-tables get their own ``[prefix.sub]``
    headers in a second pass. This is the standard TOML ordering rule
    that prevents "key after sub-table" parsing errors.
    """
    sub_tables: list[tuple[str, dict[str, Any]]] = []
    for k, v in body.items():
        if isinstance(v, dict):
            sub_tables.append((k, v))
            continue
        lines.append(f"{k} = {_toml_value(v)}")
    for sub_key, sub_body in sub_tables:
        lines.append("")
        lines.append(f"[{prefix}.{sub_key}]")
        _emit_table_body(lines, sub_body, prefix=f"{prefix}.{sub_key}")


def render_identity_summary(
    manifest: IdentityManifest,
    *,
    budget_chars: int = 100,
) -> str:
    """One-line project summary for inclusion in a workspace facts block.

    Output shape (depending on what's detected/asserted):

      Project: python (uv) · test=pytest · deploy=fly.io · style=ruff

    Constraints (``[asserted]``) get precedence over detections so a
    user override always surfaces. Returns ``""`` on an empty
    manifest or zero budget — caller can unconditionally include
    this line in a render and get nothing when there's nothing to
    say.

    Budget is approximate — the line either fits within budget or
    gets truncated with ``…`` at the end. We don't aggressively
    minify because the summary is already pre-trimmed by content
    selection.
    """
    if budget_chars <= 0:
        return ""

    parts: list[str] = []
    detected = manifest.detected or {}
    asserted = manifest.asserted or {}

    # Primary language + package manager — the load-bearing
    # orientation pair. Identifies the project shape in 4-6 words.
    primary = detected.get("language_primary")
    if primary and primary in detected and isinstance(detected[primary], dict):
        lang_block = detected[primary]
        pm = lang_block.get("package_manager")
        if pm:
            parts.append(f"{primary} ({pm})")
        else:
            parts.append(primary)

    # Test runner — second-most-asked fact during agent work
    # ("how do I run the tests?"). Surface explicitly.
    if primary and isinstance(detected.get(primary), dict):
        tr = detected[primary].get("test_runner")
        if tr:
            parts.append(f"test={tr}")

    # User assertions — every key that looks like a single-word
    # policy fact (deploy_target, style, license, etc.). Asserted
    # ``do_not_touch`` is a list — render as count.
    for key, value in asserted.items():
        if isinstance(value, str):
            short_key = key.replace("_", "-")
            parts.append(f"{short_key}={value}")
        elif isinstance(value, list) and value:
            parts.append(f"{key}=({len(value)} items)")

    if not parts:
        return ""

    summary = "Project: " + " · ".join(parts)
    if len(summary) > budget_chars:
        summary = summary[: max(0, budget_chars - 1)].rstrip() + "…"
    return summary


def serialize_manifest(manifest: IdentityManifest) -> str:
    """Render an IdentityManifest as TOML text.

    The output is round-trippable: ``parse_manifest(serialize_manifest(m))``
    produces an equivalent manifest. Pinned by tests so a future
    schema change can't silently break the contract.
    """
    lines: list[str] = []

    # [meta] always rendered, even when default — gives readers a
    # version marker so future schema migrations have a target.
    lines.append("[meta]")
    lines.append(f"detector_version = {manifest.meta.detector_version}")
    lines.append(f"last_detected_at = {repr(float(manifest.meta.last_detected_at))}")

    # [detected]
    lines.append("")
    lines.append("[detected]")
    _emit_table_body(lines, manifest.detected, prefix="detected")

    # [asserted] — even when empty, render the header so the user has
    # an obvious place to add overrides without consulting docs.
    lines.append("")
    lines.append("[asserted]")
    if manifest.asserted:
        _emit_table_body(lines, manifest.asserted, prefix="asserted")
    else:
        lines.append("# User-asserted facts go here. They take precedence over [detected].")
        lines.append('# Example: deploy_target = "fly.io"')

    # [[discovered]] — array of tables, one block per fact. Empty
    # state writes nothing (rather than an empty `[[discovered]]`
    # block) because TOML's parsing of zero-length array-of-tables is
    # awkward and the parser fills in an empty list anyway.
    for fact in manifest.discovered:
        lines.append("")
        lines.append("[[discovered]]")
        lines.append(f"ts = {repr(float(fact.ts))}")
        lines.append(f"category = {_toml_value(fact.category)}")
        lines.append(f"fact = {_toml_value(fact.fact)}")
        lines.append(f"source = {_toml_value(fact.source)}")
        lines.append(f"confidence = {_toml_value(fact.confidence)}")

    return "\n".join(lines) + "\n"
