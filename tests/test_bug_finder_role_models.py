"""Tests for role-model configuration + cross-family verifier picker.

Pins the same-family correlated-failure mitigation (arXiv 2604.07650:
Spearman 0.64-0.71 correlated errors when detector and verifier share
a model family). The picker selects from available models to break
that correlation; the config helper threads it into RoleModelConfig.
"""

from __future__ import annotations

import pytest

from augmentum.bug_finder.role_models import (
    Role,
    RoleModelConfig,
    family_for_model,
    pick_cross_family_verifier,
)


# ---------------------------------------------------------------------------
# pick_cross_family_verifier
# ---------------------------------------------------------------------------


def test_picker_returns_different_family_when_available() -> None:
    available = ("claude-sonnet-4-6", "gpt-5.4", "qwen3-72b")
    pick = pick_cross_family_verifier("claude-sonnet-4-6", available)
    assert family_for_model(pick) != "anthropic"
    assert pick in available


def test_picker_prefers_anthropic_then_openai_then_google_then_qwen() -> None:
    """The preferred-family ordering: anthropic > openai > google >
    qwen. When the primary is qwen, the picker should reach for
    anthropic first."""
    available = (
        "qwen3-72b",            # primary family
        "deepseek-v3",          # also non-preferred for picker
        "gpt-5.4",              # openai — should NOT be first pick
        "claude-sonnet-4-6",    # anthropic — should be first pick
    )
    pick = pick_cross_family_verifier("qwen3-72b", available)
    assert family_for_model(pick) == "anthropic"


def test_picker_falls_back_to_primary_when_no_cross_family() -> None:
    """When every available model is the same family as primary, the
    picker returns primary itself — the caller's
    same_model_self_verification flag will then surface True so users
    see they're running in the higher-correlated-error mode."""
    available = ("claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5")
    pick = pick_cross_family_verifier("claude-sonnet-4-6", available)
    assert pick == "claude-sonnet-4-6"


def test_picker_skips_primary_in_available() -> None:
    """The picker never returns the primary itself even when it's
    in the available list."""
    available = ("claude-sonnet-4-6",)  # only primary
    assert pick_cross_family_verifier(
        "claude-sonnet-4-6", available,
    ) == "claude-sonnet-4-6"
    # With one other model of a different family, pick that
    available = ("claude-sonnet-4-6", "gpt-5.4")
    assert pick_cross_family_verifier(
        "claude-sonnet-4-6", available,
    ) == "gpt-5.4"


def test_picker_handles_empty_available_models() -> None:
    """Edge case — no models at all means we can't cross. Return
    primary."""
    assert pick_cross_family_verifier("claude-sonnet-4-6", ()) == "claude-sonnet-4-6"
    assert pick_cross_family_verifier("claude-sonnet-4-6", []) == "claude-sonnet-4-6"


def test_picker_skips_unknown_family_models() -> None:
    """A model whose family can't be classified is ambiguous —
    don't pick it for the cross-family verifier slot. We can't
    measure that we've actually broken correlation."""
    available = ("totally-unknown-model-id", "gpt-5.4")
    pick = pick_cross_family_verifier("claude-sonnet-4-6", available)
    assert pick == "gpt-5.4"


def test_picker_handles_unknown_primary() -> None:
    """If the primary's family is unknown, ANY other model breaks the
    nominal correlation. Use whichever's first."""
    available = ("claude-sonnet-4-6", "gpt-5.4")
    pick = pick_cross_family_verifier("unknown-model-xyz", available)
    # Either is fine — what matters is it's NOT the primary
    assert pick != "unknown-model-xyz"


# ---------------------------------------------------------------------------
# RoleModelConfig.from_primary_with_cross_family_verifier
# ---------------------------------------------------------------------------


def test_config_helper_picks_cross_family_when_available() -> None:
    cfg = RoleModelConfig.from_primary_with_cross_family_verifier(
        "claude-sonnet-4-6",
        available_models=("claude-sonnet-4-6", "gpt-5.4", "qwen3-72b"),
    )
    assert cfg.detector == "claude-sonnet-4-6"
    assert cfg.verifier == "gpt-5.4"   # openai preferred
    assert not cfg.same_model_self_verification


def test_config_helper_falls_back_with_self_verification_flagged() -> None:
    cfg = RoleModelConfig.from_primary_with_cross_family_verifier(
        "claude-sonnet-4-6",
        available_models=("claude-sonnet-4-6", "claude-opus-4-7"),
    )
    assert cfg.verifier == "claude-sonnet-4-6"
    # IMPORTANT: this flag is what the orchestrator stamps on the run
    # report — users see "we couldn't break correlation, treat findings
    # accordingly".
    assert cfg.same_model_self_verification


def test_config_helper_honors_explicit_verifier_override() -> None:
    """Caller-supplied verifier always wins, even if cross-family
    candidates exist."""
    cfg = RoleModelConfig.from_primary_with_cross_family_verifier(
        "claude-sonnet-4-6",
        available_models=("claude-sonnet-4-6", "gpt-5.4"),
        explicit_verifier="qwen3-72b",
    )
    assert cfg.verifier == "qwen3-72b"


def test_config_helper_empty_available_works() -> None:
    """Empty available list = same-model. Same behavior as the
    legacy ``from_primary``."""
    cfg = RoleModelConfig.from_primary_with_cross_family_verifier(
        "claude-sonnet-4-6",
    )
    assert cfg.verifier == "claude-sonnet-4-6"
    assert cfg.same_model_self_verification


def test_config_helper_empty_primary_raises() -> None:
    with pytest.raises(ValueError):
        RoleModelConfig.from_primary_with_cross_family_verifier("")


# ---------------------------------------------------------------------------
# Existing config behavior — make sure we didn't break legacy paths
# ---------------------------------------------------------------------------


def test_legacy_from_primary_unchanged() -> None:
    """The simpler ``from_primary`` helper must still behave as
    before — single-model setup with verifier == primary."""
    cfg = RoleModelConfig.from_primary("claude-sonnet-4-6")
    assert cfg.verifier == "claude-sonnet-4-6"
    assert cfg.same_model_self_verification


def test_role_pen_tester_default_falls_through_to_verifier() -> None:
    """Phase 1c invariant — pen_tester defaults to verifier when not
    explicitly set. With cross-family-verifier helper, the pen_tester
    inherits the cross-family pick too."""
    cfg = RoleModelConfig.from_primary_with_cross_family_verifier(
        "claude-sonnet-4-6",
        available_models=("claude-sonnet-4-6", "gpt-5.4"),
    )
    assert cfg.for_role(Role.PEN_TESTER) == cfg.verifier == "gpt-5.4"
