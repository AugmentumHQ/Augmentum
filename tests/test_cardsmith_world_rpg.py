"""Unit tests for the World/RPG Cardsmith card type.

World/RPG cards route through the same single-card output mapper but
exercise different fields:
- Direct ``description`` (no slot composition)
- Heavy lorebook[] usage with WI V2 metadata (group_name, ignore_budget,
  sticky_turns, match_scenario, probability)
- Optional narrator persona via ``personality`` + ``systemPrompt``
- ``visualTraits`` is world aesthetic, not character traits

Tests verify the prompt dispatcher, supported types, and that the
existing single-card mapper produces sensible output for world content.
"""

from __future__ import annotations

import pytest

from augmentum.modes.narrative.cardsmith import (
    DEFAULT_WORLD_RPG_PROMPT,
    get_prompt,
)
from augmentum.modes.narrative.cardsmith import state
from augmentum.modes.narrative.cardsmith.output_mapper import build_character_payload


@pytest.fixture(autouse=True)
def _clear_registry():
    state._sessions.clear()
    yield
    state._sessions.clear()


# ── Prompt dispatcher ─────────────────────────────────────────────────────

def test_get_prompt_returns_world_rpg_prompt_for_world_rpg_type():
    assert get_prompt("world_rpg") == DEFAULT_WORLD_RPG_PROMPT


def test_world_rpg_prompt_is_distinct_from_single_and_ensemble():
    from augmentum.modes.narrative.cardsmith import (
        DEFAULT_ENSEMBLE_PROMPT,
        DEFAULT_SINGLE_PROMPT,
    )
    assert DEFAULT_WORLD_RPG_PROMPT != DEFAULT_SINGLE_PROMPT
    assert DEFAULT_WORLD_RPG_PROMPT != DEFAULT_ENSEMBLE_PROMPT


def test_world_rpg_prompt_emphasizes_lorebook_usage():
    """The prompt should aggressively guide the model toward heavy lorebook output."""
    p = DEFAULT_WORLD_RPG_PROMPT.lower()
    assert "lorebook" in p
    assert "group_name" in p
    assert "sticky_turns" in p
    assert "ignore_budget" in p
    assert "match_scenario" in p


def test_world_rpg_prompt_avoids_single_character_slots():
    """World cards should NOT use the desc_physical/personality/depth slots."""
    p = DEFAULT_WORLD_RPG_PROMPT
    # Should explicitly tell the model NOT to emit single-character slots
    assert "desc_physical" in p
    # Should mention this in a "don't emit" context
    assert "Don't emit `desc_physical`" in p or "desc_physical, desc_personality, desc_depth" in p


def test_world_rpg_prompt_has_question_blocks():
    """All key world-building questions should be present."""
    p = DEFAULT_WORLD_RPG_PROMPT
    expected_blocks = [
        "Q_HOOK_WORLD",
        "Q_NAME_WORLD",
        "Q_TONE_WORLD",
        "Q_SETTING",
        "Q_PROTAGONIST_ROLE",
        "Q_FACTIONS",
        "Q_RULES",
        "Q_NARRATOR",
        "Q_OPENING_SCENE",
        "Q_LOREBOOK_BUILD",
        "Q_AUGMENTUM_EXTRAS",
        "Q_FINAL",
    ]
    for block in expected_blocks:
        assert f"''''' {block} '''''" in p, f"Missing question block: {block}"


# ── Output mapper routes through single path ──────────────────────────────

def _make_world_session(**fields):
    sess = state.get_or_create_session(
        user_id="test_user",
        card_type="world_rpg",
        source="describe",
        seed_prompt="cyberpunk noir Tokyo",
    )
    for path, value in fields.items():
        sess.commit_field(path, value)
    return sess


def test_world_rpg_uses_direct_description_no_slot_composition():
    sess = _make_world_session(
        name="Sector 7 Tokyo",
        description="A neon-bleached megacity where rain never stops.",
    )
    p = build_character_payload(sess)
    assert p["data"]["description"] == "A neon-bleached megacity where rain never stops."


def test_world_rpg_payload_has_no_character_group_key():
    """World/RPG should NOT add the ensemble character_group dict."""
    sess = _make_world_session(name="World")
    p = build_character_payload(sess)
    assert "character_group" not in p


def test_world_rpg_lorebook_with_world_info_v2_fields():
    sess = _make_world_session(name="World")
    sess.commit_field("lorebook[]", {
        "keys": ["Megacorp"],
        "content": "The five corps that govern Sector 7.",
        "priority": 80,
        "group_name": "factions",
        "sticky_turns": 4,
        "match_scenario": 1,
        "ignore_budget": 0,
    })
    sess.commit_field("lorebook[]", {
        "keys": ["magic"],
        "content": "Magic requires a soul-binding contract.",
        "priority": 50,
        "group_name": "magic_rules",
        "ignore_budget": 1,
    })
    p = build_character_payload(sess)
    entries = p["data"]["lorebook"]
    assert len(entries) == 2
    by_keys = {tuple(e["keys"]): e for e in entries}
    factions = by_keys[("Megacorp",)]
    assert factions["group_name"] == "factions"
    assert factions["sticky_turns"] == 4
    assert factions["match_scenario"] == 1
    rules = by_keys[("magic",)]
    assert rules["group_name"] == "magic_rules"
    assert rules["ignore_budget"] == 1


def test_world_rpg_narrator_persona_lands_in_personality_field():
    sess = _make_world_session(
        name="World",
        personality="Detached cinematic narrator. Third-person past tense.",
    )
    p = build_character_payload(sess)
    assert "Detached cinematic narrator" in p["data"]["personality"]


def test_world_rpg_system_prompt_override_persisted():
    sess = _make_world_session(
        name="World",
        systemPrompt="Narrate in third-person past tense. Keep camera tight on {{user}}.",
    )
    p = build_character_payload(sess)
    assert "third-person past tense" in p["data"]["systemPrompt"]


def test_world_rpg_extensions_metadata_marks_card_type():
    sess = _make_world_session(name="World")
    p = build_character_payload(sess)
    cs_meta = p["data"]["extensions"]["augmentum"]["cardsmith"]
    assert cs_meta["card_type"] == "world_rpg"
    assert cs_meta["seed_prompt"] == "cyberpunk noir Tokyo"


def test_world_rpg_alt_openings_preserved():
    sess = _make_world_session(name="World")
    sess.commit_field("alternateGreetings[]", "Alt opening 1 — cyberpunk dawn.")
    sess.commit_field("alternateGreetings[]", "Alt opening 2 — rainy alley.")
    p = build_character_payload(sess)
    assert len(p["data"]["alternateGreetings"]) == 2
    assert "cyberpunk dawn" in p["data"]["alternateGreetings"][0]


def test_world_rpg_image_style_propagates():
    sess = _make_world_session(name="World", imageStyle="noir")
    p = build_character_payload(sess)
    assert p["data"]["imageStyle"] == "noir"


def test_world_rpg_with_25_lorebook_entries_all_persist():
    """World cards typically have 10-30 entries — make sure none get dropped."""
    sess = _make_world_session(name="World")
    for i in range(25):
        sess.commit_field("lorebook[]", {
            "keys": [f"entry{i}"],
            "content": f"Lore entry number {i}.",
            "priority": 100 + i,
        })
    p = build_character_payload(sess)
    assert len(p["data"]["lorebook"]) == 25


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
