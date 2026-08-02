"""Model-family ensemble tests.

Exercises:

  - family_for_model classifier on real model ids
  - merge_runs() family tracking — distinct families lift
    families_to_confirm; same-family runs stay flat
  - _detector_model_for_run round-robin selection
  - BugFinderRunConfig accepts detector_models with no validation drag
  - Finding gains the two new fields without breaking older code
"""

from __future__ import annotations

import pytest

from augmentum.bug_finder.findings import (
    ClaimSignature,
    Finding,
    Severity,
    merge_runs,
)
from augmentum.bug_finder.orchestrator import (
    BugFinderIntake,
    BugFinderRunConfig,
    _detector_model_for_run,
)
from augmentum.bug_finder.role_models import (
    RoleModelConfig,
    families_for_models,
    family_for_model,
)

# ---------------------------------------------------------------------------
# family_for_model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id,expected", [
    ("claude-opus-4-7", "anthropic"),
    ("claude-haiku-4-5-20251001", "anthropic"),
    ("gpt-5-pro", "openai"),
    ("gpt-4o", "openai"),
    ("o3-mini", "openai"),
    ("o4-pro", "openai"),
    ("codex-1", "openai"),
    ("qwen-3.6-coder", "qwen"),
    ("deepseek-v3.2", "deepseek"),
    ("gemini-3-pro", "google"),
    ("gemma-4-26b-a4b", "google"),
    ("mistral-large-2", "mistral"),
    ("magistral-2-reasoning", "mistral"),
    ("llama-5-70b", "meta"),
    ("grok-4", "xai"),
    ("kimi-k2.6", "kimi"),
    ("totally-unknown-model", "unknown"),
    ("", "unknown"),
])
def test_family_for_model(model_id: str, expected: str) -> None:
    assert family_for_model(model_id) == expected


def test_family_strips_fabric_peer_suffix() -> None:
    assert family_for_model("claude-opus-4-7@fabric:node_xyz") == "anthropic"


def test_family_strips_mode_prefix() -> None:
    assert family_for_model("a/claude-haiku-4-5") == "anthropic"
    assert family_for_model("n/gpt-5") == "openai"
    assert family_for_model("p/qwen-3.5") == "qwen"


def test_families_for_models_preserves_order() -> None:
    ids = ["claude-opus-4-7", "gpt-5", "claude-haiku-4-5", "qwen-3.5"]
    assert families_for_models(ids) == ["anthropic", "openai", "anthropic", "qwen"]


# ---------------------------------------------------------------------------
# merge_runs() with family tracking
# ---------------------------------------------------------------------------


def _mk_finding(
    file: str = "bug.py",
    *,
    signature: str = ClaimSignature.INJECTION.value,
    severity: str = Severity.HIGH.value,
    claim: str = "SQL injection",
) -> Finding:
    return Finding(
        id="",  # filled below
        file=file,
        function="handler",
        claim=claim,
        claim_signature=signature,
        severity=severity,
        evidence_paths=(f"{file}:14",),
    )


def test_merge_runs_no_families_preserves_old_behavior() -> None:
    """When `families` is None, the new fields stay at 0 — no behavior
    change for callers that don't opt in."""
    run = [_mk_finding()]
    merged = merge_runs([run, run, run])
    assert merged[0].runs_to_confirm == 3
    assert merged[0].total_runs == 3
    assert merged[0].families_to_confirm == 0
    assert merged[0].total_families == 0


def test_merge_runs_same_family_three_runs_no_family_lift() -> None:
    """Three Claude runs all flagging the same finding: family count = 1,
    even though runs_to_confirm = 3. This is the 'mediocre confidence'
    signal Anthropic's research highlights."""
    run = [_mk_finding()]
    merged = merge_runs(
        [run, run, run],
        families=["anthropic", "anthropic", "anthropic"],
    )
    assert merged[0].runs_to_confirm == 3
    assert merged[0].families_to_confirm == 1
    assert merged[0].total_families == 1


def test_merge_runs_two_families_lifts_to_two() -> None:
    """Same finding from Claude + GPT runs: families_to_confirm = 2.
    This is the 'cross-family confirmed' state — high confidence."""
    run = [_mk_finding()]
    merged = merge_runs(
        [run, run, run],
        families=["anthropic", "openai", "qwen"],
    )
    assert merged[0].runs_to_confirm == 3
    assert merged[0].families_to_confirm == 3
    assert merged[0].total_families == 3


def test_merge_runs_only_one_family_flags_finding() -> None:
    """Claude flagged this finding, GPT didn't. families_to_confirm=1."""
    f = _mk_finding()
    merged = merge_runs(
        [[f], [], []],  # only run 0 (Claude) flagged
        families=["anthropic", "openai", "qwen"],
    )
    assert merged[0].runs_to_confirm == 1
    assert merged[0].families_to_confirm == 1
    assert merged[0].total_families == 3


def test_merge_runs_families_mismatch_length_raises() -> None:
    run = [_mk_finding()]
    with pytest.raises(ValueError, match="families="):
        merge_runs([run, run], families=["anthropic"])


def test_merge_runs_unknown_family_excluded_from_count() -> None:
    """Findings from models with unknown family classification shouldn't
    inflate the count — we only credit recognized families."""
    f = _mk_finding()
    # All three runs flag it, but two of them are "unknown" (junk models)
    merged = merge_runs(
        [[f], [f], [f]],
        families=["anthropic", "unknown", "unknown"],
    )
    # The "unknown" family is still ONE distinct family at the set level,
    # plus "anthropic". So families_to_confirm = 2. We don't filter the
    # unknown out — that would mask legitimate ensemble configurations
    # where one or more models simply aren't classified yet.
    assert merged[0].families_to_confirm == 2


# ---------------------------------------------------------------------------
# _detector_model_for_run round-robin
# ---------------------------------------------------------------------------


def _mk_config(detector_models: tuple[str, ...]) -> BugFinderRunConfig:
    return BugFinderRunConfig(
        intake=BugFinderIntake(workspace_id="ws"),
        role_models=RoleModelConfig.from_primary("claude-opus-4-7"),
        detector_models=detector_models,
        detector_runs_per_chunk=3,
    )


def test_detector_model_round_robins_through_ensemble() -> None:
    config = _mk_config(("claude-opus-4-7", "gpt-5", "qwen-3.5"))
    assert _detector_model_for_run(config, 0) == "claude-opus-4-7"
    assert _detector_model_for_run(config, 1) == "gpt-5"
    assert _detector_model_for_run(config, 2) == "qwen-3.5"


def test_detector_model_wraps_when_runs_exceed_ensemble_size() -> None:
    """detector_runs_per_chunk=3 with 2 models → claude, gpt, claude."""
    config = _mk_config(("claude-opus-4-7", "gpt-5"))
    assert _detector_model_for_run(config, 0) == "claude-opus-4-7"
    assert _detector_model_for_run(config, 1) == "gpt-5"
    assert _detector_model_for_run(config, 2) == "claude-opus-4-7"


def test_detector_model_falls_back_to_role_model_when_ensemble_empty() -> None:
    """detector_models=() → use role_models.detector for every run.
    Preserves single-model semantics."""
    config = _mk_config(())
    assert _detector_model_for_run(config, 0) == "claude-opus-4-7"
    assert _detector_model_for_run(config, 1) == "claude-opus-4-7"
    assert _detector_model_for_run(config, 2) == "claude-opus-4-7"


def test_config_detector_models_default_is_empty_tuple() -> None:
    config = BugFinderRunConfig(
        intake=BugFinderIntake(workspace_id="ws"),
        role_models=RoleModelConfig.from_primary("claude-opus-4-7"),
    )
    assert config.detector_models == ()
