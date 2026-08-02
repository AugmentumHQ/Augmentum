"""Unit tests for the Cardsmith output mapper.

Covers:
- Empty session → valid minimal card
- Scalar field round-trip
- Description paragraph slot composition
- Backward compat: direct ``description`` still works when no slots set
- Tags dedup, empty filter, length cap
- Lorebook entry normalization (TavernCard character_book shape + WI V2 fields)
- Lorebook entries dropped when keys/content missing or malformed
- Regex script normalization (placement enum, find required)
- imageStyle enum filtering
- depthPromptDepth coercion + clamping
- Avatar prompt extraction (last-wins from array)
- extensions.augmentum.cardsmith metadata
"""

from __future__ import annotations

import pytest

from augmentum.modes.narrative.cardsmith.output_mapper import build_character_payload
from augmentum.modes.narrative.cardsmith.state import get_or_create_session


def _make_session(**fields):
    sess = get_or_create_session(
        user_id="test_user",
        card_type="single",
        source="describe",
        seed_prompt="test seed",
    )
    for path, value in fields.items():
        sess.commit_field(path, value)
    return sess


# ── Empty / minimal cases ─────────────────────────────────────────────────

def test_empty_session_produces_default_card():
    sess = _make_session()
    p = build_character_payload(sess)

    assert p["name"] == "New Character"
    assert p["char_id"].startswith("ch_")
    assert p["data"]["description"] == ""
    assert p["data"]["personality"] == ""
    assert p["data"]["tags"] == []
    assert p["data"]["alternateGreetings"] == []
    assert p["data"]["lorebook"] == []
    assert p["data"]["autoCollapseNarrativePanels"] is True
    assert p["regex_scripts"] == []


def test_name_with_only_whitespace_falls_back_to_default():
    sess = _make_session(name="   ")
    p = build_character_payload(sess)
    assert p["name"] == "New Character"


# ── Scalar fields ─────────────────────────────────────────────────────────

def test_scalar_fields_round_trip():
    sess = _make_session(
        name="Lyra Vex",
        personality="Stoic and dryly funny.",
        scenario="A rain-streaked diner.",
        greeting="Hello there.",
        examples="((user)) : hi\n((char)) : *nods*",
        visualTraits="auburn hair, green eyes",
        voice="am_michael",
    )
    p = build_character_payload(sess)
    d = p["data"]

    assert d["name"] == "Lyra Vex"
    assert d["personality"] == "Stoic and dryly funny."
    assert d["scenario"] == "A rain-streaked diner."
    assert d["greeting"] == "Hello there."
    assert "((user))" in d["examples"]
    assert d["visualTraits"] == "auburn hair, green eyes"
    assert d["voice"] == "am_michael"


def test_empty_string_fields_dont_overwrite_defaults():
    sess = _make_session(name="Lyra", personality="   ")
    p = build_character_payload(sess)
    assert p["data"]["personality"] == ""


# ── Description paragraph slot composition ────────────────────────────────

def test_description_slots_compose_with_double_newline():
    sess = _make_session(
        name="Lyra",
        desc_physical="Para 1+2 about looks.",
        desc_personality="Para 3 about personality.",
        desc_depth="Para 4+5 about quirks.",
    )
    p = build_character_payload(sess)
    desc = p["data"]["description"]
    assert "Para 1+2 about looks." in desc
    assert "Para 3 about personality." in desc
    assert "Para 4+5 about quirks." in desc
    # Double-newline join
    assert "\n\n" in desc
    parts = desc.split("\n\n")
    assert parts[0] == "Para 1+2 about looks."
    assert parts[1] == "Para 3 about personality."
    assert parts[2] == "Para 4+5 about quirks."


def test_description_partial_slots_join_only_present():
    sess = _make_session(name="Lyra", desc_physical="Just physical.")
    p = build_character_payload(sess)
    assert p["data"]["description"] == "Just physical."


def test_direct_description_used_when_no_slots():
    """Backward compat: model bypasses slots and emits `description` directly."""
    sess = _make_session(name="Lyra", description="Direct emission.")
    p = build_character_payload(sess)
    assert p["data"]["description"] == "Direct emission."


def test_slots_take_precedence_over_direct_description():
    sess = _make_session(
        name="Lyra",
        description="Direct one.",
        desc_physical="Slot one.",
    )
    p = build_character_payload(sess)
    # Slot composition wins over the direct emission
    assert p["data"]["description"] == "Slot one."


# ── Tags ──────────────────────────────────────────────────────────────────

def test_tags_dedup_case_insensitive_preserves_first_seen_casing():
    sess = _make_session(name="Lyra")
    sess.commit_field("tags[]", "cyberpunk")
    sess.commit_field("tags[]", "Cyberpunk")  # duplicate, different casing
    sess.commit_field("tags[]", "stoic")
    p = build_character_payload(sess)
    assert p["data"]["tags"] == ["cyberpunk", "stoic"]


def test_tags_drops_empty_strings():
    sess = _make_session(name="Lyra")
    for t in ["", "   ", "valid", ""]:
        sess.commit_field("tags[]", t)
    p = build_character_payload(sess)
    assert p["data"]["tags"] == ["valid"]


def test_tags_capped_at_30():
    sess = _make_session(name="Lyra")
    for i in range(50):
        sess.commit_field("tags[]", f"tag{i}")
    p = build_character_payload(sess)
    assert len(p["data"]["tags"]) == 30


# ── Lorebook normalization ────────────────────────────────────────────────

def test_lorebook_minimal_entry_normalized():
    sess = _make_session(name="Lyra")
    sess.commit_field("lorebook[]", '{"keys": ["Konoha"], "content": "Hidden Leaf Village."}')
    p = build_character_payload(sess)
    entries = p["data"]["lorebook"]
    assert len(entries) == 1
    e = entries[0]
    assert e["keys"] == ["Konoha"]
    assert e["content"] == "Hidden Leaf Village."
    assert e["enabled"] is True
    assert e["priority"] == 100  # default
    assert e["position"] == "before_char"  # default
    assert e["constant"] is False


def test_lorebook_preserves_world_info_v2_fields():
    sess = _make_session(name="Lyra")
    sess.commit_field("lorebook[]", (
        '{"keys": ["X"], "content": "Y", '
        '"priority": 50, "group_name": "factions", '
        '"sticky_turns": 3, "match_scenario": 1, '
        '"probability": 70}'
    ))
    p = build_character_payload(sess)
    e = p["data"]["lorebook"][0]
    assert e["priority"] == 50
    assert e["group_name"] == "factions"
    assert e["sticky_turns"] == 3
    assert e["match_scenario"] == 1
    assert e["probability"] == 70


def test_lorebook_keys_can_be_comma_separated_string():
    """Models sometimes emit `"keys": "Konoha, Hidden Leaf"` instead of a list."""
    sess = _make_session(name="Lyra")
    sess.commit_field("lorebook[]", '{"keys": "Konoha, Hidden Leaf", "content": "X"}')
    p = build_character_payload(sess)
    assert p["data"]["lorebook"][0]["keys"] == ["Konoha", "Hidden Leaf"]


def test_lorebook_drops_entry_without_keys():
    sess = _make_session(name="Lyra")
    sess.commit_field("lorebook[]", '{"content": "no keys"}')
    p = build_character_payload(sess)
    assert p["data"]["lorebook"] == []


def test_lorebook_drops_entry_without_content():
    sess = _make_session(name="Lyra")
    sess.commit_field("lorebook[]", '{"keys": ["X"]}')
    p = build_character_payload(sess)
    assert p["data"]["lorebook"] == []


def test_lorebook_drops_non_dict_entry():
    sess = _make_session(name="Lyra")
    # Direct list-as-value would be coerced by state.commit_field — emulate raw
    sess.fields.setdefault("lorebook", []).append("not a dict")
    sess.fields["lorebook"].append({"keys": ["X"], "content": "Valid"})
    p = build_character_payload(sess)
    assert len(p["data"]["lorebook"]) == 1


# ── Regex script normalization ────────────────────────────────────────────

def test_regex_script_minimal():
    sess = _make_session(name="Lyra")
    sess.commit_field(
        "regex_scripts[]",
        '{"find": "gonna", "replace": "going to", "placement": "input"}',
    )
    p = build_character_payload(sess)
    assert len(p["regex_scripts"]) == 1
    r = p["regex_scripts"][0]
    assert r["find_regex"] == "gonna"
    assert r["replace_string"] == "going to"
    assert r["placement"] == "input"
    assert r["character_name"] == "Lyra"
    assert r["enabled"] is True
    assert r["id"].startswith("rgx_")


def test_regex_script_placement_invalid_falls_back_to_output():
    sess = _make_session(name="Lyra")
    sess.commit_field(
        "regex_scripts[]",
        '{"find": "x", "placement": "weird-mode"}',
    )
    p = build_character_payload(sess)
    assert p["regex_scripts"][0]["placement"] == "output"


def test_regex_script_drops_when_find_missing():
    sess = _make_session(name="Lyra")
    sess.commit_field("regex_scripts[]", '{"replace": "y"}')
    p = build_character_payload(sess)
    assert p["regex_scripts"] == []


def test_regex_script_accepts_pattern_alias():
    sess = _make_session(name="Lyra")
    sess.commit_field("regex_scripts[]", '{"pattern": "x", "replace": "y"}')
    p = build_character_payload(sess)
    assert len(p["regex_scripts"]) == 1
    assert p["regex_scripts"][0]["find_regex"] == "x"


def test_regex_script_character_name_matches_card_name():
    sess = _make_session(name="Specific Name")
    sess.commit_field("regex_scripts[]", '{"find": "x"}')
    p = build_character_payload(sess)
    assert p["regex_scripts"][0]["character_name"] == "Specific Name"


# ── imageStyle enum filtering ─────────────────────────────────────────────

def test_image_style_known_value_kept():
    sess = _make_session(name="Lyra", imageStyle="scifi")
    p = build_character_payload(sess)
    assert p["data"]["imageStyle"] == "scifi"


def test_image_style_unknown_value_dropped():
    sess = _make_session(name="Lyra", imageStyle="weird-style-not-in-enum")
    p = build_character_payload(sess)
    assert p["data"]["imageStyle"] == ""


def test_image_style_case_normalized():
    sess = _make_session(name="Lyra", imageStyle="SciFi")
    p = build_character_payload(sess)
    assert p["data"]["imageStyle"] == "scifi"


# ── depthPromptDepth coercion ─────────────────────────────────────────────

def test_depth_prompt_depth_int_kept():
    sess = _make_session(name="Lyra", depthPromptDepth=5)
    p = build_character_payload(sess)
    assert p["data"]["depthPromptDepth"] == 5


def test_depth_prompt_depth_string_coerced():
    sess = _make_session(name="Lyra", depthPromptDepth="3")
    p = build_character_payload(sess)
    assert p["data"]["depthPromptDepth"] == 3


def test_depth_prompt_depth_clamped_high():
    sess = _make_session(name="Lyra", depthPromptDepth=99)
    p = build_character_payload(sess)
    assert p["data"]["depthPromptDepth"] == 10


def test_depth_prompt_depth_clamped_low():
    sess = _make_session(name="Lyra", depthPromptDepth=-5)
    p = build_character_payload(sess)
    assert p["data"]["depthPromptDepth"] == 0


def test_depth_prompt_depth_garbage_keeps_default():
    sess = _make_session(name="Lyra", depthPromptDepth="not a number")
    p = build_character_payload(sess)
    assert p["data"]["depthPromptDepth"] == 4  # default


# ── alternateGreetings ────────────────────────────────────────────────────

def test_alternate_greetings_preserved_in_order():
    sess = _make_session(name="Lyra")
    sess.commit_field("alternateGreetings[]", "Greeting one")
    sess.commit_field("alternateGreetings[]", "Greeting two")
    p = build_character_payload(sess)
    assert p["data"]["alternateGreetings"] == ["Greeting one", "Greeting two"]


def test_alternate_greetings_empty_dropped():
    sess = _make_session(name="Lyra")
    for g in ["", "real one", "  "]:
        sess.commit_field("alternateGreetings[]", g)
    p = build_character_payload(sess)
    assert p["data"]["alternateGreetings"] == ["real one"]


# ── Avatar prompt + extensions metadata ───────────────────────────────────

def test_avatar_prompt_uses_last_emitted():
    sess = _make_session(name="Lyra")
    sess.commit_field("avatar_prompt[]", "first")
    sess.commit_field("avatar_prompt[]", "second")
    sess.commit_field("avatar_prompt[]", "third")
    p = build_character_payload(sess)
    assert p["avatar_prompt"] == "third"
    assert (
        p["data"]["extensions"]["augmentum"]["cardsmith"]["avatar_prompt"]
        == "third"
    )


def test_extensions_augmentum_cardsmith_block():
    sess = _make_session(name="Lyra")
    p = build_character_payload(sess)
    cs_meta = p["data"]["extensions"]["augmentum"]["cardsmith"]
    assert cs_meta["source"] == "describe"
    assert cs_meta["card_type"] == "single"
    assert cs_meta["seed_prompt"] == "test seed"
    assert isinstance(cs_meta["created_at"], int)


# ── Char ID format ────────────────────────────────────────────────────────

def test_char_id_has_correct_prefix_and_format():
    sess = _make_session(name="Lyra")
    p = build_character_payload(sess)
    parts = p["char_id"].split("_")
    assert parts[0] == "ch"
    assert len(parts) == 3  # ch_<ts>_<rand>
    assert len(parts[2]) == 5  # 5-char random hex


def test_char_ids_are_unique_for_concurrent_sessions():
    s1 = _make_session(name="A")
    s2 = _make_session(name="B")
    p1 = build_character_payload(s1)
    p2 = build_character_payload(s2)
    assert p1["char_id"] != p2["char_id"]


# ── Length caps + drift logging ───────────────────────────────────────────

def test_long_description_capped():
    sess = _make_session(name="Lyra", desc_physical="x " * 5000)  # 10000 chars
    p = build_character_payload(sess)
    desc = p["data"]["description"]
    assert len(desc) <= 12100  # cap is 12000 plus ellipsis margin
    assert desc.endswith("…")


def test_long_visual_traits_capped():
    sess = _make_session(name="Lyra", visualTraits="trait, " * 200)  # ~1400 chars
    p = build_character_payload(sess)
    vt = p["data"]["visualTraits"]
    assert len(vt) <= 700
    assert vt.endswith("…")


def test_long_name_capped():
    sess = _make_session(name="A" * 500)
    p = build_character_payload(sess)
    assert len(p["name"]) <= 250
    assert p["name"].endswith("…")


def test_short_fields_not_truncated():
    sess = _make_session(
        name="Lyra",
        scenario="A short scenario.",
        greeting="A short greeting.",
    )
    p = build_character_payload(sess)
    assert p["data"]["scenario"] == "A short scenario."
    assert p["data"]["greeting"] == "A short greeting."
    assert "…" not in p["data"]["scenario"]


def test_unknown_field_logged_but_doesnt_crash(capsys):
    """Drift detection logs unknown keys without breaking the save.

    structlog routes through stdout, not Python logging records — so we
    capture stdout directly via capsys.
    """
    sess = _make_session(name="Lyra", weirdInventedField="some value")
    p = build_character_payload(sess)
    # Save still produces a valid payload
    assert p["name"] == "Lyra"
    # Warning emitted on stdout
    captured = capsys.readouterr()
    assert "cardsmith_unknown_fields" in captured.out
    assert "weirdInventedField" in captured.out


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
