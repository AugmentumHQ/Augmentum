"""Comprehender parser + UserGoal block tests.

Pure-logic. The full subagent run requires a backend + container and
is exercised separately via the smoke script. Here we cover:

  * The parser tolerates a permissive set of LLM output shapes.
  * Empty/malformed outputs return None (orchestrator falls back to
    no-brief mode cleanly).
  * Subsystems/pillars/risk_surfaces all populate from the JSON.
  * UserGoal renders into the expected prompt-block shape.
"""

from __future__ import annotations

import json

from augmentum.bug_finder.comprehender import (
    parse_comprehender_output,
)
from augmentum.bug_finder.orchestrator import UserGoal

# ---------------------------------------------------------------------------
# parse_comprehender_output
# ---------------------------------------------------------------------------


def _block(payload: dict) -> str:
    return "thinking... let me commit:\n\n```json\n" + json.dumps(payload) + "\n```\n"


def test_parse_empty_output_returns_none() -> None:
    assert parse_comprehender_output("") is None
    assert parse_comprehender_output("no json here") is None


def test_parse_malformed_json_returns_none() -> None:
    assert parse_comprehender_output("```json\n{not valid}\n```") is None


def test_parse_empty_payload_returns_none() -> None:
    """An empty schema-valid payload yields no useful map — return None
    so the orchestrator skips persistence."""
    out = parse_comprehender_output(_block({}))
    assert out is None


def test_parse_minimal_brief_only_succeeds() -> None:
    """Just a brief blob is enough — structured data can be empty."""
    out = parse_comprehender_output(_block({
        "brief": "## Tiny project\nA single-file utility.",
    }))
    assert out is not None
    assert out.brief.startswith("## Tiny project")
    assert out.subsystems == ()
    assert out.pillars == ()


def test_parse_full_payload_populates_all_fields() -> None:
    payload = {
        "brief": "## Augmentum\nFastAPI proxy + multi-modal substrate.",
        "subsystems": [
            {
                "name": "auth",
                "purpose": "multi-tenant authentication",
                "paths": ["augmentum/auth"],
                "size_files": 12,
                "pillars": ["user_id_scoping"],
            },
            {
                "name": "bug_finder",
                "purpose": "LLM-driven security audit",
                "paths": ["augmentum/bug_finder"],
                "size_files": 21,
                "pillars": [],
            },
        ],
        "pillars": [
            {
                "name": "user_id_scoping",
                "statement": "Every user-scoped table accepts user_id.",
                "evidence": ["augmentum/auth/store.py:42"],
            },
        ],
        "risk_surfaces": [
            {
                "name": "http_routes",
                "entry_points": ["augmentum/proxy/openai_routes.py:chat"],
                "trust_boundary": "user-supplied",
                "downstream_sinks": ["backend resolution"],
            },
        ],
        "entry_points": [
            {
                "kind": "http",
                "path": "POST /v1/chat/completions",
                "handler": "augmentum/proxy/openai_routes.py:chat",
            },
        ],
    }
    out = parse_comprehender_output(_block(payload))
    assert out is not None
    assert len(out.subsystems) == 2
    assert out.subsystems[0].name == "auth"
    assert out.subsystems[0].pillars == ("user_id_scoping",)
    assert out.subsystems[0].size_files == 12
    assert out.pillars[0].evidence == ("augmentum/auth/store.py:42",)
    assert out.risk_surfaces[0].trust_boundary == "user-supplied"
    assert out.entry_points[0].kind == "http"


def test_parse_skips_subsystems_without_name() -> None:
    """Entries missing the required `name` field are filtered out
    rather than crashing the parse."""
    payload = {
        "brief": "x",
        "subsystems": [
            {"name": "good", "purpose": "ok"},
            {"purpose": "missing-name"},     # dropped
            {"name": "", "purpose": "blank"},  # dropped
            {"name": "also-good", "purpose": "ok"},
        ],
    }
    out = parse_comprehender_output(_block(payload))
    assert out is not None
    names = {s.name for s in out.subsystems}
    assert names == {"good", "also-good"}


def test_parse_tolerates_bare_json_object() -> None:
    """Some models skip the fence entirely and just emit a JSON object.
    Pure-JSON output (no surrounding prose) is recognized; mixed
    prose-then-JSON without a fence is NOT — finding the start of the
    JSON in mixed text is a brittle problem and we'd rather fail loud
    than guess. Models that fail this should learn to fence."""
    payload = {"brief": "bare-json"}
    out = parse_comprehender_output(json.dumps(payload))
    assert out is not None
    assert out.brief == "bare-json"


def test_parse_uses_last_json_block_when_multiple() -> None:
    """Models sometimes emit multiple JSON blocks (thinking aloud, then
    final). Always commit to the LAST one."""
    raw = (
        "Initial thoughts:\n```json\n" + json.dumps({"brief": "draft"}) + "\n```\n\n"
        "Final answer:\n```json\n" + json.dumps({"brief": "final"}) + "\n```\n"
    )
    out = parse_comprehender_output(raw)
    assert out is not None
    assert out.brief == "final"


# ---------------------------------------------------------------------------
# UserGoal.to_prompt_block
# ---------------------------------------------------------------------------


def test_empty_user_goal_renders_empty_string() -> None:
    """An unset goal yields '' — callers concat without an if."""
    assert UserGoal().to_prompt_block() == ""


def test_named_bug_user_goal_renders_all_fields() -> None:
    g = UserGoal(
        mode="named-bug",
        description="possible auth bypass when bot accounts are deleted",
        repro_hint="DELETE /api/users/{id} with a bot user; check auth_sessions",
        scope_paths=("augmentum/auth", "augmentum/proxy/auth_routes.py"),
        severity_floor="medium",
    )
    block = g.to_prompt_block()
    assert "User goal" in block
    assert "named-bug" in block
    assert "auth bypass" in block
    assert "DELETE /api/users" in block
    assert "augmentum/auth" in block
    assert "medium" in block


def test_explore_user_goal_renders_minimally() -> None:
    g = UserGoal(
        mode="explore",
        description="general production-hardening sweep, prefer security findings",
    )
    block = g.to_prompt_block()
    assert "explore" in block
    assert "production-hardening" in block
    # No scope paths, no severity floor → those lines shouldn't appear
    assert "Scope paths" not in block
    assert "Severity floor" not in block


def test_user_goal_is_named_bug_predicate() -> None:
    assert not UserGoal().is_named_bug()
    assert not UserGoal(mode="named-bug").is_named_bug()  # no description
    assert not UserGoal(mode="explore", description="x").is_named_bug()
    assert UserGoal(mode="named-bug", description="x").is_named_bug()
