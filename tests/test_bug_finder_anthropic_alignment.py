"""Tests for the 2026-05-31 bug_finder hardening pass.

These tests pin the behavioral changes that move our pipeline closer
to Anthropic's published bug-finder design
(https://claude.com/blog/using-llms-to-secure-source-code):

1. **Disproof-oriented verifier prompt** — verifier system prompt
   frames the finding as a hypothesis to disprove and requires a
   runnable PoC for confirmation.
2. **Verifier isolation** — by default the verifier brief excludes
   detector hints (suggested_repro, claim_signature) so the verifier
   reasons from raw evidence.
3. **Non-prescriptive detector prompt** — the detector no longer
   ships a fixed bug-class enum that would suppress novel-bug
   discovery; severity rubric is evidence-first.
4. **Threat model prefix** — when the run config carries a
   user-supplied threat model, the same authoritative document
   surfaces to detector AND verifier system prompts.
"""

from __future__ import annotations

from augmentum.bug_finder.findings import Finding, FindingStatus
from augmentum.bug_finder.orchestrator import (
    BugFinderIntake,
    _prefix_threat_model,
    _threat_model_prefix_block,
)
from augmentum.bug_finder.prompts import DETECTOR_SYSTEM_PROMPT
from augmentum.bug_finder.verifier import (
    _FIX_VERIFY_SYSTEM_PROMPT,
    _REPRO_SYSTEM_PROMPT,
    _format_finding_brief,
    make_repro_spec,
)


def _sample_finding() -> Finding:
    return Finding(
        id="f1",
        file="src/foo.py",
        function="parse_header",
        claim="parse_header crashes on empty 'Authorization' header",
        claim_signature="auth-missing-deref",
        severity="high",
        evidence_paths=("src/foo.py:42-58",),
        suggested_repro="curl -H 'Authorization:' /foo",
        status=FindingStatus.SPECULATIVE.value,
    )


# ---------------------------------------------------------------------------
# 1. Disproof framing
# ---------------------------------------------------------------------------


def test_verifier_prompt_is_disproof_oriented():
    """The verifier system prompt frames the claim as a hypothesis to
    disprove and demands a runnable PoC for confirmation."""
    prompt = _REPRO_SYSTEM_PROMPT.lower()
    assert "disprove" in prompt or "false positive" in prompt
    assert "proof-of-concept" in prompt or "poc" in prompt
    assert "runnable" in prompt or "executable" in prompt


def test_verifier_prompt_warns_against_reasoning_only_confirmation():
    """Anthropic's data shows reasoning-only verification doesn't move
    the FP rate; the prompt must explicitly require executing the PoC."""
    assert "reasoning" in _REPRO_SYSTEM_PROMPT.lower()
    # We name the FP-suppression claim verbatim in the prompt as
    # motivational evidence so model takes the disproof framing seriously.
    assert "anthropic" in _REPRO_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# 2. Verifier isolation
# ---------------------------------------------------------------------------


def test_format_finding_brief_excludes_detector_hints_by_default():
    """The verifier-facing brief omits suggested_repro + claim_signature
    + severity so the verifier reasons from raw evidence."""
    brief = _format_finding_brief(_sample_finding())
    assert "suggested" not in brief.lower()
    assert "claim signature" not in brief.lower()
    assert "severity" not in brief.lower()
    assert "auth-missing-deref" not in brief
    # But the core evidence is still present.
    assert "src/foo.py" in brief
    assert "parse_header" in brief
    assert "Authorization" in brief


def test_format_finding_brief_includes_hints_when_opted_in():
    """Fix-verify needs the same context the fixer had; opt-in keeps
    the hints visible there."""
    brief = _format_finding_brief(_sample_finding(), include_detector_hints=True)
    assert "auth-missing-deref" in brief
    assert "curl -H 'Authorization:'" in brief


def test_make_repro_spec_user_msg_lacks_detector_hints():
    """End-to-end: the spec the orchestrator actually dispatches must
    not contain the detector's suggested_repro."""
    spec = make_repro_spec(
        _sample_finding(),
        model="claude-sonnet-4-6",
        tools=(),
        budget=None,  # not exercised in this test
    )
    assert "curl -H 'Authorization:'" not in spec.initial_user_message
    assert "auth-missing-deref" not in spec.initial_user_message
    # Sanity: the actual evidence still got through.
    assert "parse_header" in spec.initial_user_message


# ---------------------------------------------------------------------------
# 3. Non-prescriptive detector + evidence-first severity
# ---------------------------------------------------------------------------


def test_detector_prompt_drops_prescriptive_bug_class_checklist():
    """The detector prompt must NOT name a fixed bug-class enum in its
    framing — Anthropic's data: checklists reduce novel-bug discovery."""
    lower = DETECTOR_SYSTEM_PROMPT.lower()
    # The CHECKLIST is gone from the "scan for these bug classes" framing.
    # The output schema still mentions the field name claim_signature, but
    # the field is now free-text (it used to be a hard enum).
    assert "do not mentally check a fixed taxonomy" in lower
    # The 11-class enum used to be enumerated in the framing prose; now
    # it must only appear, if at all, as illustrative free-text not as a
    # prescribed list to walk through.
    assert "null_deref|bounds_check|race|use_after_free" not in DETECTOR_SYSTEM_PROMPT


def test_detector_prompt_has_evidence_first_severity_rubric():
    """Severity must be scored by preconditions/auth/blast-radius, not
    by bug class."""
    lower = DETECTOR_SYSTEM_PROMPT.lower()
    assert "precondition" in lower
    assert "unauthenticated" in lower or "authenticated" in lower
    # The 4-tier rubric (critical/high/medium/low) is anchored on
    # evidence-driven criteria, not class names.
    for level in ("critical", "high", "medium", "low"):
        assert level in lower


# ---------------------------------------------------------------------------
# 4. Threat model prefix
# ---------------------------------------------------------------------------


def test_threat_model_prefix_block_empty_is_no_op():
    assert _threat_model_prefix_block("") == ""
    assert _threat_model_prefix_block("   \n  ") == ""


def test_threat_model_prefix_block_renders_with_header():
    out = _threat_model_prefix_block("Assets: api keys.\nBoundary: HTTP.")
    assert out.startswith("## Threat model")
    assert "Assets: api keys" in out
    assert "Boundary: HTTP" in out


def test_prefix_threat_model_prepends_then_separates():
    out = _prefix_threat_model("ORIG", "Assets: a, b")
    assert out.startswith("## Threat model")
    assert "ORIG" in out
    # Visual separator between threat model and the original prompt.
    assert "---" in out


def test_prefix_threat_model_empty_returns_original_unchanged():
    assert _prefix_threat_model("ORIG", "") == "ORIG"
    assert _prefix_threat_model("ORIG", None) == "ORIG"


def test_intake_carries_threat_model():
    intake = BugFinderIntake(
        workspace_id="ws1",
        threat_model="Untrusted upload pipeline",
    )
    assert intake.threat_model == "Untrusted upload pipeline"


def test_intake_threat_model_defaults_empty():
    intake = BugFinderIntake(workspace_id="ws1")
    assert intake.threat_model == ""


def test_make_repro_spec_includes_threat_model_prefix_when_supplied():
    spec = make_repro_spec(
        _sample_finding(),
        model="claude-sonnet-4-6",
        tools=(),
        budget=None,
        system_prompt_prefix="## Threat model\n\nAssets: API keys.",
    )
    assert spec.system_prompt.startswith("## Threat model")
    assert "Assets: API keys" in spec.system_prompt
    # Original verifier prompt body still present after the prefix.
    assert "VERIFIER" in spec.system_prompt


def test_make_repro_spec_no_prefix_renders_clean_prompt():
    spec = make_repro_spec(
        _sample_finding(),
        model="claude-sonnet-4-6",
        tools=(),
        budget=None,
    )
    assert spec.system_prompt == _REPRO_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Smoke: fix-verify prompt still requires both repro pass + no regressions
# ---------------------------------------------------------------------------


def test_fix_verify_prompt_requires_both_repro_and_regression_check():
    """The fix-verify prompt is unchanged but pin it so future edits
    don't accidentally drop the dual-check requirement."""
    lower = _FIX_VERIFY_SYSTEM_PROMPT.lower()
    assert "repro" in lower
    assert "regression" in lower
    assert "accept" in lower
