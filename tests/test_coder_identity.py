"""Tests for the identity layer — detectors + manifest IO.

The identity layer is the first "real" context-kernel layer beyond
plan.md. It detects project facts (language, package manager, test
runner, build commands) at workspace refresh time and writes them to
``/workspace/.augmentum/identity.toml`` so the model has zero-cost
project orientation without a discovery round-trip.

Test surface:

  1. Each detector: applies() + detect() against fixture file contents.
     Detectors run on the host (this process), so we mock container
     file_read to return fixture strings.
  2. The orchestrator (``detect_identity``): runs every applicable
     detector, merges results, handles individual detector failures
     without taking out the whole refresh.
  3. Manifest IO round-trip: serialize_manifest → parse_manifest →
     equivalent manifest. Pinned because the writer is hand-rolled
     and a regression here corrupts every identity.toml.
  4. Merge semantics: ``[asserted]`` and ``[discovered]`` survive
     refresh untouched; only ``[detected]`` is replaced.

Run: python -m pytest tests/test_coder_identity.py -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.coder.identity import (
    DETECTOR_VERSION,
    DETECTORS,
    Detector,
    DiscoveredFact,
    GoDetector,
    IdentityManifest,
    IdentityMeta,
    JavaScriptDetector,
    PythonDetector,
    RustDetector,
    detect_identity,
    merge_refresh,
    parse_manifest,
    serialize_manifest,
)


# ---------------------------------------------------------------------------
# Container-manager fake. Each test maps absolute paths → file contents.
# A read of an absent path raises (mirrors real container behaviour) so
# applies() / detect() exercise their actual file_read-via-try/except
# probes rather than special-casing test paths.
# ---------------------------------------------------------------------------


def _cm(files: dict[str, str]) -> MagicMock:
    cm = MagicMock()

    async def _file_read(workspace_id: str, path: str) -> str:
        if path in files:
            return files[path]
        raise FileNotFoundError(path)

    cm.file_read = AsyncMock(side_effect=_file_read)
    cm.file_write = AsyncMock(return_value=None)
    cm._run_command = AsyncMock(return_value="")
    cm.run_command = cm._run_command
    return cm


# ---------------------------------------------------------------------------
# PythonDetector
# ---------------------------------------------------------------------------


class TestPythonDetector:
    @pytest.mark.asyncio
    async def test_applies_on_pyproject(self):
        cm = _cm({"/workspace/pyproject.toml": "[project]\nname='x'"})
        assert await PythonDetector.applies(cm, "ws") is True

    @pytest.mark.asyncio
    async def test_applies_on_requirements_txt(self):
        cm = _cm({"/workspace/requirements.txt": "fastapi\n"})
        assert await PythonDetector.applies(cm, "ws") is True

    @pytest.mark.asyncio
    async def test_does_not_apply_in_empty_workspace(self):
        cm = _cm({})
        assert await PythonDetector.applies(cm, "ws") is False

    @pytest.mark.asyncio
    async def test_detect_uv_with_pytest_and_ruff(self):
        cm = _cm({
            "/workspace/pyproject.toml": """\
[project]
name = "demo"
version = "0.1.0"
dependencies = ["fastapi", "structlog", "aiosqlite"]
[project.scripts]
demo = "demo.cli:main"
[tool.pytest.ini_options]
testpaths = ["tests"]
[tool.ruff]
line-length = 100
""",
            "/workspace/uv.lock": "# lock",
        })
        facts = await PythonDetector.detect(cm, "ws")
        assert facts["name"] == "demo"
        assert facts["version"] == "0.1.0"
        assert facts["package_manager"] == "uv"
        assert facts["test_runner"] == "pytest"
        assert facts["dependencies"][:3] == ["fastapi", "structlog", "aiosqlite"]
        assert facts["entry_points"] == ["demo"]
        assert "ruff" in facts["tooling"]

    @pytest.mark.asyncio
    async def test_detect_poetry_lock_overrides_pip(self):
        """Detection order: uv > poetry > pipenv > pip. A workspace with
        both poetry.lock and requirements.txt should report poetry."""
        cm = _cm({
            "/workspace/pyproject.toml": "[project]\nname='x'\n",
            "/workspace/poetry.lock": "",
            "/workspace/requirements.txt": "fastapi\n",
        })
        facts = await PythonDetector.detect(cm, "ws")
        assert facts["package_manager"] == "poetry"

    @pytest.mark.asyncio
    async def test_detect_falls_back_to_pytest_marker_files(self):
        """No tool.pytest in pyproject? conftest.py / pytest.ini still
        identifies pytest as the test runner."""
        cm = _cm({
            "/workspace/pyproject.toml": "[project]\nname='x'\n",
            "/workspace/conftest.py": "",
        })
        facts = await PythonDetector.detect(cm, "ws")
        assert facts.get("test_runner") == "pytest"

    @pytest.mark.asyncio
    async def test_detect_malformed_pyproject_returns_partial(self):
        """Malformed pyproject.toml shouldn't crash detection — the
        detector salvages what it can from other markers."""
        cm = _cm({
            "/workspace/pyproject.toml": "[project\nname = broken",
            "/workspace/uv.lock": "",
        })
        facts = await PythonDetector.detect(cm, "ws")
        # No crash, partial result OK.
        assert facts.get("package_manager") == "uv"
        # Malformed pyproject means we couldn't extract name/version.
        assert "name" not in facts


# ---------------------------------------------------------------------------
# JavaScriptDetector
# ---------------------------------------------------------------------------


class TestJavaScriptDetector:
    @pytest.mark.asyncio
    async def test_applies_on_package_json(self):
        cm = _cm({"/workspace/package.json": '{"name": "x"}'})
        assert await JavaScriptDetector.applies(cm, "ws") is True

    @pytest.mark.asyncio
    async def test_detect_npm_with_scripts(self):
        cm = _cm({
            "/workspace/package.json": """{
  "name": "ui",
  "version": "1.2.3",
  "main": "src/index.js",
  "type": "module",
  "scripts": {
    "build": "webpack",
    "test": "jest",
    "start": "node src/index.js",
    "lint": "eslint .",
    "internal": "node tools/internal.js"
  },
  "dependencies": {
    "react": "^18", "react-dom": "^18", "lodash": "^4"
  }
}""",
            "/workspace/package-lock.json": "",
        })
        facts = await JavaScriptDetector.detect(cm, "ws")
        assert facts["name"] == "ui"
        assert facts["version"] == "1.2.3"
        assert facts["main"] == "src/index.js"
        assert facts["module_type"] == "module"
        assert facts["package_manager"] == "npm"
        assert facts["scripts"]["build"] == "webpack"
        assert facts["scripts"]["test"] == "jest"
        # "internal" is not in the curated keep-list — must be elided
        # so manifest stays focused on conventional script names.
        assert "internal" not in facts["scripts"]
        assert facts["dependencies"][:3] == ["react", "react-dom", "lodash"]

    @pytest.mark.asyncio
    async def test_detect_pnpm_lockfile_takes_precedence(self):
        cm = _cm({
            "/workspace/package.json": '{"name":"x"}',
            "/workspace/pnpm-lock.yaml": "",
            "/workspace/package-lock.json": "",  # both present
        })
        facts = await JavaScriptDetector.detect(cm, "ws")
        assert facts["package_manager"] == "pnpm"

    @pytest.mark.asyncio
    async def test_detect_typescript_flag(self):
        cm = _cm({
            "/workspace/package.json": '{"name":"x"}',
            "/workspace/tsconfig.json": '{}',
        })
        facts = await JavaScriptDetector.detect(cm, "ws")
        assert facts.get("typescript") is True

    @pytest.mark.asyncio
    async def test_malformed_package_json_returns_empty(self):
        cm = _cm({"/workspace/package.json": "not valid json {{"})
        facts = await JavaScriptDetector.detect(cm, "ws")
        assert facts == {}

    @pytest.mark.asyncio
    async def test_default_package_manager_is_npm(self):
        """No lockfile? Default to npm — the user can override via
        identity.toml[asserted]."""
        cm = _cm({"/workspace/package.json": '{"name":"x"}'})
        facts = await JavaScriptDetector.detect(cm, "ws")
        assert facts["package_manager"] == "npm"


# ---------------------------------------------------------------------------
# RustDetector
# ---------------------------------------------------------------------------


class TestRustDetector:
    @pytest.mark.asyncio
    async def test_detect_bin_crate(self):
        cm = _cm({
            "/workspace/Cargo.toml": """\
[package]
name = "cli"
version = "0.2.0"
edition = "2021"
rust-version = "1.75"

[[bin]]
name = "cli"
path = "src/main.rs"

[dependencies]
clap = "4"
tokio = "1"
""",
        })
        facts = await RustDetector.detect(cm, "ws")
        assert facts["name"] == "cli"
        assert facts["version"] == "0.2.0"
        assert facts["edition"] == "2021"
        assert facts["rust_version"] == "1.75"
        assert facts["binaries"] == ["cli"]
        assert facts["dependencies"][:2] == ["clap", "tokio"]
        assert facts["test_runner"] == "cargo test"
        assert facts["package_manager"] == "cargo"

    @pytest.mark.asyncio
    async def test_detect_workspace_root(self):
        cm = _cm({
            "/workspace/Cargo.toml": """\
[workspace]
members = ["crates/api", "crates/cli"]
""",
        })
        facts = await RustDetector.detect(cm, "ws")
        assert facts.get("is_workspace_root") is True
        assert facts["workspace_members"] == ["crates/api", "crates/cli"]


# ---------------------------------------------------------------------------
# GoDetector
# ---------------------------------------------------------------------------


class TestGoDetector:
    @pytest.mark.asyncio
    async def test_detect_go_module(self):
        cm = _cm({
            "/workspace/go.mod": """\
module github.com/example/svc

go 1.22

require (
    github.com/spf13/cobra v1.8.0
)
""",
        })
        facts = await GoDetector.detect(cm, "ws")
        assert facts["module"] == "github.com/example/svc"
        assert facts["go_version"] == "1.22"
        assert facts["test_runner"] == "go test ./..."
        assert facts["build_cmd"] == "go build ./..."
        assert facts["package_manager"] == "go"


# ---------------------------------------------------------------------------
# detect_identity orchestrator
# ---------------------------------------------------------------------------


class TestDetectIdentity:
    @pytest.mark.asyncio
    async def test_pure_python_repo(self):
        cm = _cm({
            "/workspace/pyproject.toml": "[project]\nname='x'\n",
            "/workspace/uv.lock": "",
        })
        manifest = await detect_identity(cm, "ws", now=1716000000.0)
        assert manifest.detected["languages"] == ["python"]
        assert manifest.detected["language_primary"] == "python"
        assert "python" in manifest.detected
        assert manifest.detected["python"]["package_manager"] == "uv"
        assert manifest.meta.last_detected_at == 1716000000.0

    @pytest.mark.asyncio
    async def test_polyglot_python_plus_javascript(self):
        """Real-world case: Augmentum itself is Python backend +
        JavaScript frontend. Both detectors must fire and the manifest
        must record both languages."""
        cm = _cm({
            "/workspace/pyproject.toml": "[project]\nname='backend'\n",
            "/workspace/uv.lock": "",
            "/workspace/package.json": '{"name":"frontend","scripts":{"build":"vite build"}}',
            "/workspace/pnpm-lock.yaml": "",
        })
        manifest = await detect_identity(cm, "ws")
        # Order is registry order: Python first, JS second.
        assert manifest.detected["languages"] == ["python", "javascript"]
        assert manifest.detected["language_primary"] == "python"
        assert manifest.detected["python"]["package_manager"] == "uv"
        assert manifest.detected["javascript"]["package_manager"] == "pnpm"
        assert manifest.detected["javascript"]["scripts"]["build"] == "vite build"

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_empty_languages(self):
        cm = _cm({})
        manifest = await detect_identity(cm, "ws")
        assert manifest.detected["languages"] == []
        assert "language_primary" not in manifest.detected

    @pytest.mark.asyncio
    async def test_one_detector_crashing_does_not_kill_others(self):
        """If a detector raises, the orchestrator must log + skip and
        still return facts from the other detectors. This is the
        load-bearing robustness property."""

        class BrokenDetector(Detector):
            name = "broken"

            @classmethod
            async def applies(cls, cm, workspace_id: str) -> bool:
                return True

            @classmethod
            async def detect(cls, cm, workspace_id: str) -> dict[str, Any]:
                raise RuntimeError("simulated detector crash")

        # Temporarily prepend the broken detector.
        DETECTORS.insert(0, BrokenDetector)
        try:
            cm = _cm({
                "/workspace/pyproject.toml": "[project]\nname='x'\n",
            })
            manifest = await detect_identity(cm, "ws")
            # Broken didn't kill the rest.
            assert "python" in manifest.detected
            assert "broken" not in manifest.detected
            assert "broken" not in manifest.detected["languages"]
        finally:
            DETECTORS.pop(0)


# ---------------------------------------------------------------------------
# Manifest serialization round-trip — pin the writer's contract
# ---------------------------------------------------------------------------


class TestManifestRoundTrip:
    def test_empty_manifest_serializes_and_parses(self):
        m = IdentityManifest()
        text = serialize_manifest(m)
        m2 = parse_manifest(text)
        assert m2.detected == m.detected
        assert m2.asserted == m.asserted
        assert m2.discovered == m.discovered
        # Meta carries the version even on empty.
        assert m2.meta.detector_version == DETECTOR_VERSION

    def test_full_shape_round_trips(self):
        m = IdentityManifest(
            meta=IdentityMeta(detector_version=1, last_detected_at=1716000000.0),
            detected={
                "languages": ["python", "javascript"],
                "language_primary": "python",
                "python": {
                    "package_manager": "uv",
                    "test_runner": "pytest",
                    "dependencies": ["fastapi", "structlog"],
                    "tooling": ["ruff", "mypy"],
                },
                "javascript": {
                    "package_manager": "pnpm",
                    "scripts": {"build": "vite", "test": "vitest"},
                },
            },
            asserted={
                "deploy_target": "fly.io",
                "do_not_touch": ["augmentum/state/migrations/*.sql"],
            },
            discovered=[
                DiscoveredFact(
                    ts=1716000400.0, category="env",
                    fact="auth tokens live in /workspace/.env.local",
                    source="shell_exec turn 9", confidence="confirmed",
                ),
                DiscoveredFact(
                    ts=1716001000.0, category="constraint",
                    fact="node 18 locked",
                    source="user turn 6", confidence="user_asserted",
                ),
            ],
        )
        text = serialize_manifest(m)
        m2 = parse_manifest(text)
        assert m2.detected == m.detected
        assert m2.asserted == m.asserted
        assert len(m2.discovered) == 2
        assert m2.discovered[0].fact == m.discovered[0].fact
        assert m2.discovered[1].confidence == "user_asserted"

    def test_parse_handles_malformed_toml(self):
        m = parse_manifest("not really [toml")
        # Fresh empty manifest — caller can treat as "no identity".
        assert m.detected == {}
        assert m.asserted == {}
        assert m.discovered == []

    def test_serialize_includes_help_comment_when_asserted_empty(self):
        """First-time users see an example in the [asserted] section
        so they don't have to consult docs to learn the field shape."""
        m = IdentityManifest()
        text = serialize_manifest(m)
        assert "[asserted]" in text
        assert "User-asserted facts" in text or "user-asserted" in text.lower()


# ---------------------------------------------------------------------------
# merge_refresh — the contract that asserted + discovered survive refresh
# ---------------------------------------------------------------------------


class TestMergeRefresh:
    def test_asserted_survives_detection_refresh(self):
        existing = IdentityManifest(
            detected={"python": {"package_manager": "pip"}},
            asserted={"deploy_target": "fly.io", "style": "ruff"},
            discovered=[],
        )
        fresh_detected = {"python": {"package_manager": "uv"}, "languages": ["python"]}
        merged = merge_refresh(existing, fresh_detected, now=1716000000.0)
        # detected REPLACED.
        assert merged.detected == fresh_detected
        # asserted PRESERVED.
        assert merged.asserted == {"deploy_target": "fly.io", "style": "ruff"}
        # meta UPDATED.
        assert merged.meta.last_detected_at == 1716000000.0

    def test_discovered_survives_detection_refresh(self):
        existing = IdentityManifest(
            detected={"python": {}},
            asserted={},
            discovered=[
                DiscoveredFact(
                    ts=1.0, category="env",
                    fact="X", source="turn 1", confidence="confirmed",
                ),
                DiscoveredFact(
                    ts=2.0, category="api",
                    fact="Y", source="turn 2", confidence="confirmed",
                ),
            ],
        )
        merged = merge_refresh(existing, {"python": {"package_manager": "uv"}})
        assert len(merged.discovered) == 2
        assert merged.discovered[0].fact == "X"
        assert merged.discovered[1].fact == "Y"
