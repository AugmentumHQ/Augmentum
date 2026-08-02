"""Tests for the chunk-facts pre-compute optimization.

The point of this module: replace 2-3 detector tool-call round-trips
per chunk with one cheap local AST + JSON lookup. These tests pin
the contract (correct facts surface; empty-input is a no-op; failures
swallow) and the detector-template wiring.
"""

from __future__ import annotations

from pathlib import Path

from augmentum.bug_finder.chunk_facts import (
    ChunkFacts,
    compute_chunk_facts,
    render_chunk_facts,
)
from augmentum.bug_finder.workspace_substrate import (
    WorkspacePattern,
    upsert_pattern,
)

# ---------------------------------------------------------------------------
# compute_chunk_facts
# ---------------------------------------------------------------------------


def test_compute_with_no_workspace_root_returns_empty() -> None:
    facts = compute_chunk_facts(
        workspace_root=None, file="x.py", line_start=1,
    )
    assert facts.is_empty
    assert facts.decorators == ()
    assert facts.prior_patterns == ()


def test_compute_with_empty_file_returns_empty(tmp_path: Path) -> None:
    facts = compute_chunk_facts(
        workspace_root=tmp_path, file="", line_start=1,
    )
    assert facts.is_empty


def test_compute_surfaces_decorator_chain(tmp_path: Path) -> None:
    """When the chunk's function has decorators, the pre-compute
    captures them so the detector doesn't have to call decorators_on."""
    (tmp_path / "app.py").write_text(
        "@require_auth\n@rate_limit(10)\ndef login():\n    pass\n",
        encoding="utf-8",
    )
    facts = compute_chunk_facts(
        workspace_root=tmp_path, file="app.py", line_start=3,
    )
    assert len(facts.decorators) == 2
    names = [d.name for d in facts.decorators]
    assert "require_auth" in names
    assert "rate_limit" in names


def test_compute_surfaces_prior_workspace_patterns(tmp_path: Path) -> None:
    """Pattern memory at this file flows through too — closing the
    compounding loop one layer deeper than the planner-level brief."""
    upsert_pattern(
        tmp_path, signature="bandit:B608",
        file_pattern="store/notes.py",
        severity="high", sample_claim="SQL string assembly",
    )
    facts = compute_chunk_facts(
        workspace_root=tmp_path,
        file="store/notes.py", line_start=1,
    )
    assert len(facts.prior_patterns) == 1
    assert facts.prior_patterns[0].signature == "bandit:B608"


def test_compute_filters_patterns_to_matching_file(tmp_path: Path) -> None:
    """A pattern at a DIFFERENT file shouldn't appear in this chunk's
    facts — that'd be noise."""
    upsert_pattern(
        tmp_path, signature="bandit:B608",
        file_pattern="auth/routes.py",
        sample_claim="SQL injection",
    )
    upsert_pattern(
        tmp_path, signature="ruff:S105",
        file_pattern="config.py",
        sample_claim="hardcoded password",
    )
    facts = compute_chunk_facts(
        workspace_root=tmp_path,
        file="auth/routes.py", line_start=1,
    )
    sigs = {p.signature for p in facts.prior_patterns}
    assert sigs == {"bandit:B608"}


def test_compute_sorts_patterns_by_recurrence(tmp_path: Path) -> None:
    """High-hit-count patterns come first — most signal per token."""
    upsert_pattern(
        tmp_path, signature="rarely_seen",
        file_pattern="x.py", sample_claim="rare",
    )
    for _ in range(5):
        upsert_pattern(
            tmp_path, signature="often_seen",
            file_pattern="x.py", sample_claim="frequent",
        )
    facts = compute_chunk_facts(
        workspace_root=tmp_path, file="x.py", line_start=1,
    )
    assert facts.prior_patterns[0].signature == "often_seen"


def test_compute_swallows_decorator_failures(tmp_path: Path) -> None:
    """If decorators_on raises (e.g. syntax error in source) the
    pre-compute degrades to empty instead of breaking the detector."""
    (tmp_path / "broken.py").write_text(
        "def f(\n    # syntax error: unclosed paren\n",
        encoding="utf-8",
    )
    # Should not raise even though the file is syntactically broken
    facts = compute_chunk_facts(
        workspace_root=tmp_path, file="broken.py", line_start=1,
    )
    # decorators is empty (couldn't parse) but the call returned cleanly
    assert facts.decorators == ()


# ---------------------------------------------------------------------------
# render_chunk_facts
# ---------------------------------------------------------------------------


def test_render_empty_returns_empty_string() -> None:
    """First-contact workspaces / chunks with no facts must not alter
    the detector's user prompt — pinned for prompt-stability."""
    assert render_chunk_facts(ChunkFacts()) == ""


def test_render_includes_pre_computed_header() -> None:
    """The header is the cue that tells the LLM this block didn't
    come from a tool call — important so it doesn't try to 'verify'
    facts that are already deterministic."""
    facts = ChunkFacts(
        prior_patterns=(WorkspacePattern(
            signature="bandit:B608", file_pattern="x.py",
            hit_count=2, fix_count=0,
        ),),
    )
    rendered = render_chunk_facts(facts)
    assert "Pre-computed facts" in rendered
    assert "deterministic" in rendered


def test_render_decorator_block_includes_FP_killer_cue(tmp_path: Path) -> None:
    """The 'auth decorator may already guard this' cue is the
    load-bearing FP-killer for the most common detector misfire
    (missing-auth claim on a @require_auth handler)."""
    (tmp_path / "r.py").write_text(
        "@require_auth\ndef h():\n    pass\n", encoding="utf-8",
    )
    facts = compute_chunk_facts(
        workspace_root=tmp_path, file="r.py", line_start=2,
    )
    rendered = render_chunk_facts(facts)
    assert "@require_auth" in rendered
    assert "auth/validation decorator" in rendered


def test_render_prior_patterns_block_marks_unresolved_status() -> None:
    """A pattern with fix_count=0 is the strongest precision prior —
    the rendered block must surface that distinction so the detector
    weights it correctly."""
    facts = ChunkFacts(prior_patterns=(
        WorkspacePattern(
            signature="bandit:B608", file_pattern="x.py",
            hit_count=3, fix_count=0, severity="high",
            sample_claim="SQL assembly",
        ),
        WorkspacePattern(
            signature="ruff:F401", file_pattern="x.py",
            hit_count=5, fix_count=2, severity="low",
        ),
    ))
    rendered = render_chunk_facts(facts)
    assert "unresolved" in rendered
    assert "2 fixes" in rendered


def test_render_includes_anti_bias_framing() -> None:
    """Pattern priors must be framed as hotspots, not as claims — the
    detector's job is still to find evidence, not to confirm priors."""
    facts = ChunkFacts(prior_patterns=(WorkspacePattern(
        signature="x", file_pattern="f.py", hit_count=1, fix_count=0,
    ),))
    rendered = render_chunk_facts(facts)
    assert "hotspot priors" in rendered.lower()
    assert "not confirmed claims" in rendered.lower()


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------


def test_detector_template_accepts_precomputed_facts_block() -> None:
    """Format-string regression check — adding a parameter to the
    template that doesn't get filled would raise KeyError on every
    detector call. Pin the slot."""
    from augmentum.bug_finder.prompts import DETECTOR_USER_TEMPLATE
    msg = DETECTOR_USER_TEMPLATE.format(
        file="a.py", function="f", line_start=1, line_end=10,
        rationale="r", suspected_class="x",
        precomputed_facts_block="<TEST BLOCK>",
    )
    assert "<TEST BLOCK>" in msg


def test_detector_template_blank_block_renders_cleanly() -> None:
    """Empty block (no facts) must format without leaving an awkward
    blank line or unfilled placeholder."""
    from augmentum.bug_finder.prompts import DETECTOR_USER_TEMPLATE
    msg = DETECTOR_USER_TEMPLATE.format(
        file="a.py", function="f", line_start=1, line_end=10,
        rationale="r", suspected_class="x",
        precomputed_facts_block="",
    )
    # No trailing "{" / placeholder leakage
    assert "{" not in msg
    assert "}" not in msg


def test_orchestrator_run_config_default_chunk_facts_on() -> None:
    """Pre-compute defaults to ON because the cost is local AST work
    in exchange for token-priced round-trips — a net win."""
    import inspect

    from augmentum.bug_finder.orchestrator import BugFinderRunConfig
    sig = inspect.signature(BugFinderRunConfig)
    assert sig.parameters["enable_chunk_facts_precompute"].default is True


def test_orchestrator_detector_path_references_chunk_facts() -> None:
    """The detector subagent path imports + calls the pre-compute.
    A regression that strips it would silently disable the
    optimization without breaking tests."""
    import inspect

    from augmentum.bug_finder import orchestrator
    src = inspect.getsource(orchestrator._run_detector_for_chunk)
    assert "compute_chunk_facts" in src
    assert "render_chunk_facts" in src
    assert "precomputed_facts_block" in src
