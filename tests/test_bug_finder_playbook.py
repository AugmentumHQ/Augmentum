"""Seeded hunting playbook — targeted selection + rendering + taxonomy
consistency (audit 2026-06-17).

The playbook primes the planner with class-specific priors on cold-start,
TARGETED to the codebase's risk surfaces so a small repo doesn't pay for
classes it has no surface for."""
from __future__ import annotations

from augmentum.bug_finder.findings import ClaimSignature
from augmentum.bug_finder.playbook import (
    PLAYBOOK,
    render_playbook_brief,
    select_playbook,
)


def test_every_entry_signature_is_a_valid_claim_signature():
    valid = {c.value for c in ClaimSignature}
    for e in PLAYBOOK:
        assert e.signature in valid, e.signature


def test_cold_start_returns_universal_classes():
    """No risk surfaces known yet → still seed the essentials."""
    entries = select_playbook(risk_surface_names=(), max_entries=4)
    sigs = {e.signature for e in entries}
    # The universal high-value classes must be present on cold-start.
    assert ClaimSignature.INJECTION.value in sigs
    assert ClaimSignature.AUTH_BYPASS.value in sigs
    assert ClaimSignature.MISSING_VALIDATION.value in sigs


def test_targeted_selection_promotes_surface_relevant_class():
    """A codebase whose comprehension found deserialize_sinks should get
    the missing_validation (deser/SSRF/XXE) card ranked in."""
    entries = select_playbook(
        risk_surface_names=("deserialize_sinks",), max_entries=2,
    )
    sigs = [e.signature for e in entries]
    assert ClaimSignature.MISSING_VALIDATION.value in sigs


def test_signatures_seen_boosts_a_nonuniversal_class():
    """A class recurring in learned memory outranks the universal baseline
    even if its surface isn't flagged — double down on what's actually here."""
    entries = select_playbook(
        risk_surface_names=(),
        signatures_seen=(ClaimSignature.RACE.value,),
        max_entries=1,
    )
    assert entries and entries[0].signature == ClaimSignature.RACE.value


def test_max_entries_caps_output():
    entries = select_playbook(
        risk_surface_names=("http_routes", "upload_endpoints",
                            "deserialize_sinks", "background_jobs"),
        max_entries=2,
    )
    assert len(entries) == 2


def test_render_includes_sections_and_empty_is_blank():
    assert render_playbook_brief([]) == ""
    entries = select_playbook(risk_surface_names=("http_routes",), max_entries=3)
    brief = render_playbook_brief(entries)
    assert "Targeted hunting playbook" in brief
    assert "Where to look" in brief
    assert "Common false positives" in brief
    assert "A correct fix restores" in brief
    # The cards rendered match the selected signatures.
    for e in entries:
        assert e.signature in brief
