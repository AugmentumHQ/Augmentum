"""Unit tests for the Cardsmith ensemble output mapper.

Covers:
- build_character_payload dispatches to ensemble path on card_type='ensemble'
- Member normalization + merge-by-name across multiple emissions
- Relationship clamping (trust/affection ∈ [-1,1], tension ∈ [0,1])
- Self-relationships dropped
- generation_mode enum filtering + default fallback
- visualTraits composition with <Name> markers
- Description composition (group_dynamic + roster bullet list)
- character_groups row shape (member_names, member_summaries JSON-able)
"""

from __future__ import annotations

import pytest

from augmentum.modes.narrative.cardsmith import state
from augmentum.modes.narrative.cardsmith.output_mapper import build_character_payload


@pytest.fixture(autouse=True)
def _clear_registry():
    state._sessions.clear()
    yield
    state._sessions.clear()


def _make_ensemble_session(**fields):
    sess = state.get_or_create_session(
        user_id="test_user",
        card_type="ensemble",
        source="describe",
        seed_prompt="adventuring party",
    )
    for path, value in fields.items():
        sess.commit_field(path, value)
    return sess


def _commit_member(sess, **kw):
    sess.commit_field("members[]", kw)


def _commit_relationship(sess, **kw):
    sess.commit_field("relationships[]", kw)


# ── Dispatch ──────────────────────────────────────────────────────────────

def test_build_character_payload_dispatches_to_ensemble():
    sess = _make_ensemble_session(name="The Crew")
    p = build_character_payload(sess)
    assert "character_group" in p


def test_single_payload_does_not_have_character_group_key():
    s = state.get_or_create_session(
        user_id="u1", card_type="single", source="describe",
    )
    s.commit_field("name", "Lyra")
    p = build_character_payload(s)
    assert "character_group" not in p


# ── Member merge by name ──────────────────────────────────────────────────

def test_member_roster_then_filled_loop_merges_by_name():
    """Cardsmith emits the roster first (empty fields), then re-emits each
    name during the per-member loop with filled fields. The mapper merges."""
    sess = _make_ensemble_session(name="The Crew")
    # Roster pass — establishes 3 members with empty details
    sess.commit_field("members[]", {"name": "Mira", "role": "leader"})
    sess.commit_field("members[]", {"name": "Jin", "role": "foil"})
    sess.commit_field("members[]", {"name": "Brick", "role": "muscle"})
    # Member loop — fills in details, re-emitting same names
    sess.commit_field("members[]", {
        "name": "Mira", "summary": "Tactical and quiet.", "physical": "lean, scarred",
    })
    sess.commit_field("members[]", {
        "name": "Brick", "summary": "Soft-spoken bruiser.", "physical": "broad-shouldered",
    })

    p = build_character_payload(sess)
    members = p["data"]["extensions"]["augmentum"]["cardsmith"]["members"]

    assert len(members) == 3
    by_name = {m["name"]: m for m in members}
    assert by_name["Mira"]["role"] == "leader"
    assert by_name["Mira"]["summary"] == "Tactical and quiet."
    assert by_name["Mira"]["physical"] == "lean, scarred"
    assert by_name["Jin"]["role"] == "foil"
    # Jin's summary was never filled — should be empty string
    assert by_name["Jin"]["summary"] == ""
    assert by_name["Brick"]["physical"] == "broad-shouldered"


def test_hallucinated_roster_superseded_by_user_supplied_one():
    """Regression for 4B audit finding: model committed Seraphina/Lorien/...
    in turn 1 (filled summaries), then Mira/Jin/... in turn 2, then real
    names Kira/Jek/Marn/Tess (placeholders) in turn 3. Old behavior kept all
    12 names. New behavior: latest placeholder batch is canonical, earlier
    rosters dropped, fills merge only into canonical.
    """
    sess = _make_ensemble_session(name="Crew")
    # Turn 1: hallucinated roster with filled fields.
    for n, summary in [
        ("Seraphina", "A paladin with a strong sense of justice."),
        ("Lorien", "A sneaky rogue."),
        ("Brother Dain", "A cynical cleric."),
        ("Kael", "A naive ranger."),
    ]:
        sess.commit_field("members[]", {
            "name": n, "role": "x", "summary": summary, "physical": "",
        })
    # Turn 2: another hallucination after user said "The Cinderhalls Crew".
    for n, summary in [
        ("Mira", "Tactical and quiet."),
        ("Jin", "Cunning and sly."),
        ("Brick", "Strong and straightforward."),
        ("Lila", "Innocent and trusting."),
    ]:
        sess.commit_field("members[]", {
            "name": n, "role": "x", "summary": summary, "physical": "",
        })
    # Turn 3: user supplies real names — model commits placeholders.
    for n, role in [("Kira", "leader"), ("Jek", "rogue"), ("Marn", "cleric"), ("Tess", "ranger")]:
        sess.commit_field("members[]", {
            "name": n, "role": role, "summary": "",
            "physical": "", "voice_hint": "",
        })
    # Turn 4: per-member loop fills the real roster.
    sess.commit_field("members[]", {
        "name": "Kira", "summary": "A steadfast paladin.",
        "physical": "tall, scarred",
    })
    sess.commit_field("members[]", {
        "name": "Jek", "summary": "Sharp tongue.",
        "physical": "wiry, hooded",
    })

    p = build_character_payload(sess)
    members = p["data"]["extensions"]["augmentum"]["cardsmith"]["members"]
    names = [m["name"] for m in members]

    assert set(names) == {"Kira", "Jek", "Marn", "Tess"}
    assert "Seraphina" not in names
    assert "Mira" not in names

    # Fills landed.
    by_name = {m["name"]: m for m in members}
    assert by_name["Kira"]["summary"] == "A steadfast paladin."
    assert by_name["Kira"]["physical"] == "tall, scarred"
    assert by_name["Marn"]["role"] == "cleric"


def test_no_roster_declaration_falls_back_to_legacy_merge():
    """If the model never emits a placeholder roster block (only filled
    entries), every name remains canonical — same as old behavior."""
    sess = _make_ensemble_session(name="Crew")
    sess.commit_field("members[]", {
        "name": "Mira", "role": "leader", "summary": "Tactical.",
        "physical": "lean", "voice_hint": "",
    })
    sess.commit_field("members[]", {
        "name": "Jin", "role": "foil", "summary": "Sly.",
        "physical": "", "voice_hint": "",
    })
    p = build_character_payload(sess)
    members = p["data"]["extensions"]["augmentum"]["cardsmith"]["members"]
    assert {m["name"] for m in members} == {"Mira", "Jin"}


def test_filled_entries_outside_canonical_roster_dropped():
    """If user re-rosters mid-conversation, fills for old names are ignored."""
    sess = _make_ensemble_session(name="Crew")
    # Old roster with content
    sess.commit_field("members[]", {
        "name": "Old1", "role": "lead", "summary": "old summary",
    })
    # Placeholder roster (the supersede signal)
    sess.commit_field("members[]", {"name": "New1", "role": "lead", "summary": "", "physical": ""})
    sess.commit_field("members[]", {"name": "New2", "role": "foil", "summary": "", "physical": ""})

    p = build_character_payload(sess)
    members = p["data"]["extensions"]["augmentum"]["cardsmith"]["members"]
    names = {m["name"] for m in members}
    assert names == {"New1", "New2"}
    # Old1's "old summary" doesn't appear anywhere.


def test_member_drops_entries_without_name():
    sess = _make_ensemble_session(name="The Crew")
    sess.commit_field("members[]", {"name": "Mira"})
    sess.commit_field("members[]", {"name": "  "})  # whitespace name
    sess.commit_field("members[]", {"role": "leader"})  # no name at all
    sess.commit_field("members[]", "not a dict")

    p = build_character_payload(sess)
    members = p["data"]["extensions"]["augmentum"]["cardsmith"]["members"]
    assert len(members) == 1
    assert members[0]["name"] == "Mira"


# ── Relationship clamp ────────────────────────────────────────────────────

def test_relationship_floats_clamp_to_valid_range():
    sess = _make_ensemble_session(name="The Crew")
    _commit_member(sess, name="A")
    _commit_member(sess, name="B")
    sess.commit_field("relationships[]", {
        "source": "A", "target": "B",
        "trust": 1.5, "affection": -2.0, "tension": 99.0,
    })
    p = build_character_payload(sess)
    rels = p["data"]["extensions"]["augmentum"]["cardsmith"]["relationships"]
    assert len(rels) == 1
    r = rels[0]
    assert r["trust"] == 1.0
    assert r["affection"] == -1.0
    assert r["tension"] == 1.0


def test_relationship_negative_tension_clamped_to_zero():
    sess = _make_ensemble_session(name="The Crew")
    sess.commit_field("relationships[]", {
        "source": "A", "target": "B",
        "tension": -0.5,
    })
    p = build_character_payload(sess)
    rels = p["data"]["extensions"]["augmentum"]["cardsmith"]["relationships"]
    assert rels[0]["tension"] == 0.0


def test_self_relationship_dropped():
    sess = _make_ensemble_session(name="The Crew")
    sess.commit_field("relationships[]", {"source": "A", "target": "A", "trust": 0.5})
    p = build_character_payload(sess)
    assert p["data"]["extensions"]["augmentum"]["cardsmith"]["relationships"] == []


def test_relationship_merge_by_pair():
    sess = _make_ensemble_session(name="The Crew")
    sess.commit_field("relationships[]", {"source": "A", "target": "B", "trust": 0.5})
    sess.commit_field("relationships[]", {"source": "A", "target": "B", "label": "old friends"})
    sess.commit_field("relationships[]", {"source": "A", "target": "B", "tension": 0.3})
    p = build_character_payload(sess)
    rels = p["data"]["extensions"]["augmentum"]["cardsmith"]["relationships"]
    assert len(rels) == 1
    r = rels[0]
    assert r["trust"] == 0.5
    assert r["label"] == "old friends"
    assert r["tension"] == 0.3


# ── generation_mode enum ──────────────────────────────────────────────────

def test_generation_mode_known_values_kept():
    for mode in ("round_robin", "random", "manual", "llm_decide"):
        sess = _make_ensemble_session(name=f"Crew_{mode}", generation_mode=mode)
        p = build_character_payload(sess)
        assert p["character_group"]["generation_mode"] == mode


def test_generation_mode_unknown_falls_back_to_llm_decide():
    sess = _make_ensemble_session(name="Crew", generation_mode="weird-mode")
    p = build_character_payload(sess)
    assert p["character_group"]["generation_mode"] == "llm_decide"


def test_generation_mode_missing_defaults_to_llm_decide():
    sess = _make_ensemble_session(name="Crew")
    p = build_character_payload(sess)
    assert p["character_group"]["generation_mode"] == "llm_decide"


# ── visualTraits composition ──────────────────────────────────────────────

def test_visual_traits_composes_name_markers():
    sess = _make_ensemble_session(name="Crew")
    _commit_member(sess, name="Mira", physical="lean, scarred")
    _commit_member(sess, name="Jin", physical="tall, lanky")
    p = build_character_payload(sess)
    vt = p["data"]["visualTraits"]
    assert "<Mira>" in vt
    assert "<Jin>" in vt
    assert "lean, scarred" in vt
    assert "tall, lanky" in vt


def test_visual_traits_skips_members_without_physical():
    sess = _make_ensemble_session(name="Crew")
    _commit_member(sess, name="Mira", physical="lean, scarred")
    _commit_member(sess, name="Jin")  # no physical
    p = build_character_payload(sess)
    vt = p["data"]["visualTraits"]
    assert "<Mira>" in vt
    assert "<Jin>" not in vt


# ── Description composition ───────────────────────────────────────────────

def test_description_combines_dynamic_and_roster():
    sess = _make_ensemble_session(
        name="Crew",
        group_dynamic="A trio of orphans running a smuggling op.",
    )
    _commit_member(sess, name="Mira", role="leader", summary="Tactical and quiet.")
    _commit_member(sess, name="Jin", role="foil", summary="Dry-witted skeptic.")
    p = build_character_payload(sess)
    desc = p["data"]["description"]
    assert "trio of orphans" in desc
    assert "Members:" in desc
    assert "Mira (leader)" in desc
    assert "Tactical and quiet" in desc
    assert "Jin (foil)" in desc


def test_description_falls_back_to_direct_description_field():
    """Backward compat: model bypasses group_dynamic and emits description."""
    sess = _make_ensemble_session(name="Crew", description="Direct group blurb.")
    p = build_character_payload(sess)
    assert "Direct group blurb" in p["data"]["description"]


# ── character_groups row shape ────────────────────────────────────────────

def test_character_group_row_member_names_and_summaries():
    sess = _make_ensemble_session(name="The Cinderhalls Crew")
    _commit_member(sess, name="Mira", summary="Leader summary.")
    _commit_member(sess, name="Jin", summary="Foil summary.")
    p = build_character_payload(sess)
    g = p["character_group"]
    assert g["name"] == "The Cinderhalls Crew"
    assert g["member_names"] == ["Mira", "Jin"]
    assert g["member_summaries"] == {
        "Mira": "Leader summary.",
        "Jin": "Foil summary.",
    }
    assert g["muted_names"] == []


def test_character_group_member_summaries_is_json_serializable():
    """The route JSON-encodes member_summaries before INSERT."""
    import json
    sess = _make_ensemble_session(name="Crew")
    _commit_member(sess, name="Mira", summary="Has 'quotes' and \"both\" kinds.")
    p = build_character_payload(sess)
    # Round-trip through JSON
    encoded = json.dumps(p["character_group"]["member_summaries"])
    decoded = json.loads(encoded)
    assert decoded["Mira"] == "Has 'quotes' and \"both\" kinds."


def test_character_group_uses_group_dynamic_as_description():
    sess = _make_ensemble_session(name="Crew", group_dynamic="The dynamic blurb.")
    p = build_character_payload(sess)
    assert p["character_group"]["description"] == "The dynamic blurb."


def test_empty_ensemble_still_produces_valid_payload():
    sess = _make_ensemble_session(name="Empty Crew")
    p = build_character_payload(sess)
    assert p["name"] == "Empty Crew"
    assert p["character_group"]["name"] == "Empty Crew"
    assert p["character_group"]["member_names"] == []
    assert p["character_group"]["member_summaries"] == {}


# ── Drift detection works for ensemble too ────────────────────────────────

def test_ensemble_drift_logs_unknown_keys(capsys):
    sess = _make_ensemble_session(name="Crew", weirdEnsembleField="x")
    _ = build_character_payload(sess)
    captured = capsys.readouterr()
    assert "cardsmith_unknown_fields_ensemble" in captured.out
    assert "weirdEnsembleField" in captured.out


# ── Members commit also exposed in extensions for downstream seeding ──────

def test_members_round_trip_through_extensions():
    sess = _make_ensemble_session(name="Crew")
    _commit_member(sess, name="Mira", role="leader", summary="X", physical="Y", voice_hint="Z")
    p = build_character_payload(sess)
    cs_meta = p["data"]["extensions"]["augmentum"]["cardsmith"]
    assert cs_meta["card_type"] == "ensemble"
    assert len(cs_meta["members"]) == 1
    assert cs_meta["members"][0] == {
        "name": "Mira", "role": "leader", "summary": "X",
        "physical": "Y", "voice_hint": "Z",
    }


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
