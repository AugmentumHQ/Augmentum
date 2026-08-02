"""Fixtures + mocks for coder eval cases.

Phase 0.3a (this commit): tmp-workspace builder + case loader.
Phase 0.3b (next): ScriptedBackend that drives a real CoderHandler
with canned model responses, plus result-bundle extraction. Sketched
here as TODOs so the format and runner contract are stable.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def coder_eval_workspace(tmp_path: Path):
    """Materialize a case's ``workspace.files`` map into ``tmp_path``.

    Returns a callable so a single test can build multiple workspaces if
    needed (e.g., before/after comparison).
    """
    def _build(files: dict[str, str]) -> Path:
        for rel, content in (files or {}).items():
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return tmp_path
    return _build


def load_case(path: Path) -> dict:
    """Load a yaml case file. Centralized so the runner + ad-hoc loaders
    share validation."""
    with path.open("r", encoding="utf-8") as f:
        case = yaml.safe_load(f)
    required = {"name", "tier", "user_message"}
    missing = required - set(case or {})
    if missing:
        raise ValueError(f"{path}: missing required keys {sorted(missing)}")
    if case["tier"] not in {"reflex", "surgical", "composed", "project"}:
        raise ValueError(f"{path}: tier must be reflex|surgical|composed|project, got {case['tier']!r}")
    return case


# TODO Phase 0.3b: ScriptedBackend mock that yields ``responses`` from
# the case yaml as InternalStreamChunks. Mirrors
# tests/agentic_evals/conftest.py:ScriptedBackend but for CoderHandler's
# generate_stream contract.
#
# TODO Phase 0.3b: snapshot_workspace(tmp_path) -> dict[str, str] for
# building the result bundle's "files" + "files_changed" fields.
#
# TODO Phase 0.3b: result_bundle_from_chunks(chunks, workspace_before,
# workspace_after) -> dict matching properties.py's expected shape.
