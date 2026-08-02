"""Verification-spine prompt invariants.

Guards the doctrine from
docs/superpowers/specs/2026-07-06-coder-verification-spine-design.md
against silent loss in future prompt refactors. These are protected
strings: a rewrite may rephrase around them, but the falsifiability
doctrine, the claim→oracle rubric, and the exemplar verification
beats must survive.
"""
from __future__ import annotations

from pathlib import Path

from augmentum.coder.exemplar_loader import load_exemplar
from augmentum.coder.prompts import (
    ACT_SYSTEM,
    ACT_SYSTEM_WITH_TOOLS,
    NATIVE_SYSTEM,
    WORKSPACE_GUIDE,
    workspace_guide,
)
from augmentum.modes.coder.intent import TurnIntentKind

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Falsifiability doctrine — the oldest invariant, predates the spine.
# ---------------------------------------------------------------------------

def test_act_system_keeps_falsifiability_doctrine():
    assert "must be able to FAIL" in ACT_SYSTEM
    # The text-tier variant is spliced from ACT_SYSTEM; the doctrine
    # lives in the rules half and must survive the splice.
    assert "must be able to FAIL" in ACT_SYSTEM_WITH_TOOLS


def test_native_system_keeps_oracle_doctrine():
    # One lean bullet on the native tier (9B-promotion tiering): the
    # claim→cheapest-falsifiable-check reflex + the honesty rule.
    assert "cheapest" in NATIVE_SYSTEM
    assert "FAIL" in NATIVE_SYSTEM
    assert "no honest automated check" in NATIVE_SYSTEM.lower()


# ---------------------------------------------------------------------------
# Workspace-guide rubric — injected into every coder system prompt.
# ---------------------------------------------------------------------------

def test_workspace_guide_has_claim_oracle_rubric():
    # Normalize wrapping — prompt prose rewraps freely; the doctrine
    # phrases must survive regardless of line breaks.
    flat = " ".join(WORKSPACE_GUIDE.split())
    assert "## Verification (claim → oracle)" in flat
    # Anchor rows that must not be lost in rewording:
    assert "write a failing test first" in flat
    assert "same verifier" in flat
    assert "seeded deterministic replay" in flat.lower()
    assert "never" in flat.lower() and "third-party" in flat
    assert "No honest automated oracle" in flat
    # The cross-cutting sanity check:
    assert "could its output change if the code were wrong" in flat.lower()


def test_workspace_guide_rubric_survives_profile_addenda():
    base = workspace_guide(None)
    pentest = workspace_guide("pentest")
    for text in (base, pentest):
        assert "## Verification (claim → oracle)" in text
    # Addendum still appends (the rubric edit must not clobber profiles).
    assert "## Pentest profile" in pentest
    assert pentest.startswith(base)


# ---------------------------------------------------------------------------
# Exemplar verification beats.
# ---------------------------------------------------------------------------

def test_all_intent_exemplars_load():
    for kind in TurnIntentKind:
        text = load_exemplar(kind)
        assert isinstance(text, str)
        if kind != TurnIntentKind.UNKNOWN:
            assert text.strip(), f"exemplar for {kind} is empty"


def test_implement_exemplar_teaches_oracle_selection():
    text = load_exemplar(TurnIntentKind.IMPLEMENT)
    assert "Pick the oracle" in text
    assert "FAIL" in text


def test_debug_exemplar_teaches_red_first_same_verifier():
    text = load_exemplar(TurnIntentKind.DEBUG)
    assert "verifier that closes" in text
    assert "Red first" in text


def test_review_exemplar_teaches_could_this_fail():
    text = load_exemplar(TurnIntentKind.REVIEW)
    assert "could this fail if the code were wrong" in text.lower()
    assert "missed coverage" in text.lower()


# ---------------------------------------------------------------------------
# The taxonomy reference doc.
# ---------------------------------------------------------------------------

def test_testing_md_has_oracle_taxonomy():
    text = (REPO_ROOT / "docs" / "testing.md").read_text(encoding="utf-8")
    assert "## Choosing the Oracle" in text
    assert "No honest oracle" in text
    assert "augmentum.contracts.probe" in text
