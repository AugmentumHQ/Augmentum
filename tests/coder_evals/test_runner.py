"""Parametrized runner for coder eval cases.

Phase 0.3a: validates that every yaml under ``cases/`` parses, declares
required keys, references only known assertion properties, and has a
syntactically-valid responses script. Does NOT yet execute the coder
pipeline — the ScriptedBackend wiring is Phase 0.3b.

Once 0.3b lands, the parametrized test below will:
  1. Materialize ``workspace.files`` into a tmp dir
  2. Wire a ScriptedBackend with the case's ``responses``
  3. Drive ``CoderHandler.generate_stream`` with ``user_message``
  4. Snapshot post-turn workspace + tool-call trace
  5. Build a result bundle and run ``apply_assertions`` against it
"""
from __future__ import annotations

from pathlib import Path

import pytest

from augmentum.modes.coder.intent import Tier, classify_tier
from tests.coder_evals.conftest import load_case
from tests.coder_evals.properties import REGISTRY, apply_assertions

CASES_DIR = Path(__file__).parent / "cases"


def _discover_cases() -> list[tuple[str, Path]]:
    cases = []
    if not CASES_DIR.exists():
        return cases
    for yaml_path in sorted(CASES_DIR.rglob("*.yaml")):
        rel = yaml_path.relative_to(CASES_DIR).with_suffix("")
        cases.append((str(rel).replace("\\", "/"), yaml_path))
    return cases


@pytest.mark.parametrize(
    "case_id,case_path",
    _discover_cases(),
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_coder_eval_case_loads(case_id: str, case_path: Path) -> None:
    """Phase 0.3a: validate case structure without executing the coder.

    Each case must:
      - Parse as yaml
      - Declare required keys (name, tier, user_message) per load_case
      - Use a known tier (reflex|surgical|composed|project)
      - Reference only known assertion properties
      - Declare responses (optional but if present, must be a list)
    """
    case = load_case(case_path)

    # Validate assertions reference known properties
    for spec in case.get("assertions") or []:
        name = spec.get("property")
        assert name in REGISTRY, (
            f"{case_path}: unknown assertion property {name!r}; "
            f"known: {sorted(REGISTRY)}"
        )

    # Validate responses shape if present
    responses = case.get("responses")
    if responses is not None:
        assert isinstance(responses, list), (
            f"{case_path}: responses must be a list"
        )
        for i, r in enumerate(responses):
            assert isinstance(r, dict), (
                f"{case_path}: response #{i} must be a dict"
            )
            # Each response is either content-only, tool_calls-only, or both
            keys = set(r)
            allowed = {"content", "tool_calls", "thinking"}
            extra = keys - allowed
            assert not extra, (
                f"{case_path}: response #{i} has unknown keys {extra}; "
                f"allowed: {allowed}"
            )


def test_corpus_has_at_least_one_case_per_tier() -> None:
    """Smoke test: every tier should have at least one seed case."""
    by_tier: dict[str, int] = {}
    for _, path in _discover_cases():
        case = load_case(path)
        by_tier[case["tier"]] = by_tier.get(case["tier"], 0) + 1
    for tier in ("reflex", "surgical", "composed", "project"):
        assert by_tier.get(tier, 0) >= 1, (
            f"tier {tier!r} has no seed cases; expected at least 1"
        )


@pytest.mark.asyncio
async def test_hybrid_bench_runs_scripted_reflex_case() -> None:
    """Smoke the standalone bench through the real Hybrid Coder loop."""
    from tests.coder_evals.bench import BenchRunConfig, run_case

    result = await run_case(
        BenchRunConfig(
            case_path=CASES_DIR / "reflex" / "case_add_missing_import.yaml",
            model="scripted-coder-eval",
            model_label="scripted",
            backend_kind="scripted",
        ),
    )

    assert result["outcome"] == "perfect"
    assert result["strategy"] == "hybrid"
    assert "code_edit" in result["tools_used"]


# ---------------------------------------------------------------------------
# Phase 1.4: classifier-quality check against the seed corpus.
#
# This exercises classify_tier directly without needing the full coder
# pipeline (Phase 0.3b). It catches misclassification of representative
# user messages — a regression here means Phase 1's tier classifier
# would route real work to the wrong tier.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,case_path",
    _discover_cases(),
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_coder_eval_case_classifies_to_declared_tier(
    case_id: str, case_path: Path,
) -> None:
    """Each seed case's user_message should classify into the case's
    declared tier. Activated by Phase 1.4 — enables tier_classified_as
    assertion on real-shaped corpus messages."""
    case = load_case(case_path)
    workspace_files = case.get("workspace", {}).get("files") or {}
    file_count = len(workspace_files)

    classification = classify_tier(
        latest_text=case["user_message"],
        workspace_file_count=file_count if file_count > 0 else None,
    )

    expected_tier = Tier(case["tier"])
    assert classification.tier == expected_tier, (
        f"{case_path}: declared tier={case['tier']} but classified as "
        f"{classification.tier.value} (reason={classification.reason!r}, "
        f"signals={classification.signals})"
    )

    # Also exercise tier_classified_as against a synthetic result bundle —
    # this is what the full-pipeline runner (Phase 0.3b) will do once
    # tier metadata is extracted from emitted chunks.
    synthetic_result = {
        "files": workspace_files,
        "files_changed": [],
        "tools_used": [],
        "iterations": 0,
        "tokens_used": 0,
        "tier": classification.tier.value,
        "verification": None,
    }
    failures = apply_assertions(
        synthetic_result,
        [
            spec for spec in case.get("assertions") or []
            if spec.get("property") == "tier_classified_as"
        ],
    )
    assert not failures, f"{case_path}: {failures}"
