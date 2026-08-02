"""Tests pinning the 2026-05-31 powers extension pass.

Five new built-in powers + three stub fleshes shipped to leverage the
subagent dispatch substrate and surface professional-grade defaults
for agent behavior:

* subagent-router       — nudge task_dispatch at the right windows
* observation-keeper    — persist non-obvious discoveries via observe
* multi-tenant-auditor  — enforce Augmentum's #1 security invariant
* changelog-documenter  — pre-finish summary gate
* test-baseline-keeper  — pre-change baseline vs post-change comparison

Stubs fleshed: dependency-doctor, performance-profiler,
workspace-onboarding.

The tests verify the registry picks them up, that the manifests parse,
and that key load-bearing references (task_dispatch, observe,
user_id, etc.) are present in the bodies — so future edits don't
accidentally remove the wiring.
"""

from __future__ import annotations

import pytest

from augmentum.powers.registry import PowerRegistry


@pytest.fixture(scope="module")
def registry() -> PowerRegistry:
    return PowerRegistry()


# ---------------------------------------------------------------------------
# 5 new powers discoverable
# ---------------------------------------------------------------------------


NEW_POWER_IDS = {
    "subagent-router",
    "observation-keeper",
    "multi-tenant-auditor",
    "changelog-documenter",
    "test-baseline-keeper",
}


def test_new_powers_discovered(registry):
    found = {p.slug for p in registry.list_powers()}
    missing = NEW_POWER_IDS - found
    assert not missing, f"new powers not discovered: {missing}"


def test_subagent_router_references_task_dispatch(registry):
    power = registry.get_power("subagent-router")
    assert power is not None
    assert "task_dispatch" in (power.body_markdown or "")
    # Must cover every built-in role so the model has a complete picture.
    for role in ("explore", "plan", "review", "research", "security_review", "threat_model"):
        assert role in power.body_markdown, f"subagent-router missing role {role!r}"


def test_observation_keeper_references_observe_tool(registry):
    power = registry.get_power("observation-keeper")
    assert power is not None
    body = power.body_markdown or ""
    assert "observe" in body
    # Must mention the ledger path the observe tool actually writes to.
    assert "observations.jsonl" in body
    # Must cover at least the high-leverage categories.
    for cat in ("constraint", "gotcha", "env"):
        assert cat in body


def test_multi_tenant_auditor_enforces_user_id_pattern(registry):
    power = registry.get_power("multi-tenant-auditor")
    assert power is not None
    body = power.body_markdown or ""
    # The defining contract from CLAUDE.md.
    assert "user_id" in body
    # Must point at the load-bearing primitives.
    assert "AND user_id = ?" in body
    assert "REFERENCES users(id)" in body
    # Triggers at the right windows — schema/route/store changes.
    assert "pre_plan" in power.activation_windows
    assert "post_write" in power.activation_windows


def test_changelog_documenter_fires_pre_finish(registry):
    power = registry.get_power("changelog-documenter")
    assert power is not None
    assert "pre_finish" in power.activation_windows
    body = power.body_markdown or ""
    # Must ground the summary in real diffs, not memory.
    assert "git diff" in body or "git status" in body
    # Must call out post-merge concerns.
    assert "migration" in body.lower() or "restart" in body.lower()


def test_test_baseline_keeper_captures_before_and_after(registry):
    power = registry.get_power("test-baseline-keeper")
    assert power is not None
    # Must fire on both ends so baseline can be captured first.
    assert "pre_plan" in power.activation_windows
    assert "pre_finish" in power.activation_windows
    body = power.body_markdown or ""
    # Must use the actual test_run tool.
    assert "test_run" in body
    # Must distinguish baseline vs post-change.
    assert "baseline" in body.lower()


# ---------------------------------------------------------------------------
# 3 stub fleshes are no longer stubs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug,min_chars", [
    ("dependency-doctor", 1000),
    ("performance-profiler", 1500),
    ("workspace-onboarding", 1500),
])
def test_stub_power_has_substantive_body(registry, slug, min_chars):
    power = registry.get_power(slug)
    assert power is not None, f"{slug} not registered"
    body = power.body_markdown or ""
    assert len(body) >= min_chars, (
        f"{slug} body is {len(body)} chars, expected ≥ {min_chars} — "
        "looks like the stub flesh-out got reverted."
    )


# ---------------------------------------------------------------------------
# Existing powers wired to task_dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", [
    "contract-keeper",
    "test-author",
    "failure-triage",
])
def test_existing_power_mentions_task_dispatch(registry, slug):
    """The wiring touches done in 2026-05-31 must persist — these
    powers got task_dispatch added to preferred_tools and/or the body
    so the model knows to reach for subagent spawning in their
    activation contexts."""
    power = registry.get_power(slug)
    assert power is not None
    body = power.body_markdown or ""
    mentions_pref = "task_dispatch" in (power.preferred_tools or [])
    mentions_body = "task_dispatch" in body
    assert mentions_pref or mentions_body, (
        f"{slug} no longer references task_dispatch — wiring lost."
    )


# ---------------------------------------------------------------------------
# Sanity: no manifest is empty / broken
# ---------------------------------------------------------------------------


def test_all_powers_have_minimum_metadata(registry):
    for power in registry.list_powers():
        assert power.slug, "power missing slug"
        assert power.display_name, f"{power.slug} missing display_name"
        assert power.kind, f"{power.slug} missing kind"
        assert power.activation_policy, f"{power.slug} missing activation_policy"
        assert power.body_markdown, f"{power.slug} missing body"
