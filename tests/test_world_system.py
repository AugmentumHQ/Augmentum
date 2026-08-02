"""World-system manifest — parse, tracker authority, dice, sheet, tools.

Spec: docs/superpowers/specs/2026-07-15-world-system-manifest-design.md
The bundled Cyraeth manifest (data/cyraeth_world_manifest.json) doubles
as the parse fixture so the shipped definition stays valid by test.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from augmentum.modes.narrative.world_native_schemas import (
    dispatch_world_native_tool,
    schemas_for_manifest,
)
from augmentum.modes.narrative.world_system import (
    USER_LOCK_TURNS,
    WorldStore,
    match_sheet_command,
    parse_manifest,
    roll_dice,
    sheet_text,
)

_MANIFEST_PATH = Path(__file__).parent.parent / "data" / "cyraeth_world_manifest.json"


def _card_raw() -> dict:
    return {"extensions": {"world_system": json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))}}


@pytest.fixture()
def manifest():
    m = parse_manifest(_card_raw())
    assert m is not None
    return m


@pytest.fixture()
def store(manifest):
    return WorldStore(manifest, {})


# -- parse ------------------------------------------------------------------

def test_absence_is_invisible():
    assert parse_manifest(None) is None
    assert parse_manifest({}) is None
    assert parse_manifest({"extensions": {}}) is None
    assert parse_manifest({"extensions": {"world_system": {"spec": "other_v9"}}}) is None


def test_cyraeth_manifest_parses(manifest):
    assert manifest.name == "Cyraeth"
    assert set(manifest.modules) == {"trackers", "tables", "dice", "sheet"}
    assert manifest.tracker("health").bands[-1] == "Pristine"
    assert manifest.tracker("hollowing").reveal_on == "exposed"
    assert manifest.table("currency") is not None
    assert manifest.dice.player_roller is True


def test_malformed_manifest_never_raises():
    raw = {"extensions": {"world_system": {
        "spec": "augmentum_world_v1", "modules": ["trackers"],
        "trackers": [{"id": "x", "kind": "band", "bands": ["only-one"]}],
    }}}
    # sole tracker invalid -> trackers module pruned -> no modules -> None
    assert parse_manifest(raw) is None


# -- tracker authority (spec D1) ---------------------------------------------

def test_band_moves_one_step(store):
    ok, _, v = store.shift("health", to="Bruised", turn=1)
    assert ok and v == "Bruised"
    ok, msg, v = store.shift("health", to="Dying", turn=2)
    assert not ok and "one band" in msg and v == "Bruised"


def test_force_allows_jump(store):
    ok, _, v = store.shift("health", to="Critical", turn=1,
                           reason="force: dragon breath direct hit")
    assert ok and v == "Critical"


def test_counter_delta_and_floor(store):
    ok, _, v = store.shift("gold_marks", delta=25, turn=1)
    assert ok and v == 25
    ok, _, v = store.shift("gold_marks", delta=-100, turn=2)
    assert ok and v == 0  # min bound, never negative


def test_user_lock_blocks_model(store):
    ok, _, _ = store.shift("stamina", to="Winded", turn=5, by="user",
                           reason="force: user correction")
    assert ok
    ok, msg, _ = store.shift("stamina", to="Steady", turn=5 + USER_LOCK_TURNS - 1)
    assert not ok and "locked" in msg
    ok, _, _ = store.shift("stamina", to="Steady", turn=5 + USER_LOCK_TURNS)
    assert ok


def test_reveal_on_hides_until_moved(store, manifest):
    t = manifest.tracker("hollowing")
    assert not store.revealed(t)
    store.shift("hollowing", to="Exposed", turn=3)
    assert store.revealed(t)


def test_state_block_authoritative_line(store):
    store.shift("health", to="Bruised", turn=1)
    block = store.state_block()
    assert "authoritative" in block
    assert "Bruised" in block
    assert "Hollowing" not in block  # unrevealed stays out of the prompt


# -- dice ---------------------------------------------------------------------

def test_roll_expressions():
    rng = random.Random(7)
    r = roll_dice("d20+2", rng=rng)
    assert r["total"] == sum(r["rolls"]) + 2 and 1 <= r["rolls"][0] <= 20
    assert roll_dice("2d6", rng=rng)["rolls"].__len__() == 2
    assert roll_dice("banana") is None
    assert roll_dice("100d2") is None  # count cap


# -- sheet ---------------------------------------------------------------------

def test_sheet_sections_and_reveal_gate(store):
    sheet = store.sheet()
    ids = [s["id"] for s in sheet["sections"]]
    assert "condition" in ids and "hollowing" not in ids
    store.shift("hollowing", to="Exposed", turn=1)
    ids = [s["id"] for s in store.sheet()["sections"]]
    assert "hollowing" in ids
    assert "Health" in sheet_text(sheet)


def test_sheet_command_matching():
    assert match_sheet_command("/status") == ""
    assert match_sheet_command("/s") == "condition"
    assert match_sheet_command("/inv") == "inventory"
    assert match_sheet_command("she checks her status") is None


# -- tool layer -----------------------------------------------------------------

def test_schemas_gate_on_modules(manifest):
    assert {s["function"]["name"] for s in schemas_for_manifest(manifest)} == {
        "world.track.shift", "world.roll", "world.lookup",
    }
    manifest.modules = ["trackers", "sheet"]
    assert {s["function"]["name"] for s in schemas_for_manifest(manifest)} == {
        "world.track.shift",
    }


def test_dispatch_shift_and_events(store):
    text, events = dispatch_world_native_tool(
        store, turn=1, tool_name="world.track.shift",
        raw_arguments=json.dumps({"tracker": "health", "to": "Bruised",
                                  "reason": "grazed by an arrow"}),
    )
    assert "Bruised" in text
    assert events and events[0]["kind"] == "tracker_shift"


def test_dispatch_roll_with_dc(store):
    text, events = dispatch_world_native_tool(
        store, turn=1, tool_name="world.roll",
        raw_arguments={"expression": "d20+5", "check": "Persuasion", "dc": 1},
    )
    assert "SUCCESS" in text  # d20+5 always >= 6 > DC 1
    assert events[0]["outcome"] == "success"


def test_dispatch_lookup(store):
    text, _ = dispatch_world_native_tool(
        store, turn=1, tool_name="world.lookup",
        raw_arguments={"table": "currency", "query": "penny"},
    )
    assert "Silver Penny" in text
    text, _ = dispatch_world_native_tool(
        store, turn=1, tool_name="world.lookup",
        raw_arguments={"table": "nope"},
    )
    assert "Unknown table" in text


def test_dispatch_never_raises_on_garbage(store):
    text, events = dispatch_world_native_tool(
        store, turn=0, tool_name="world.track.shift", raw_arguments="{not json",
    )
    assert "Unknown tracker" in text and events == []


# -- lore tiering (canon core hysteresis + stable placement) -------------------

def test_lore_core_hysteresis():
    from augmentum.modes.narrative.lore_engine import LoreEngine
    le = LoreEngine()
    # promote after 2 hits within the window
    assert le.update_core_membership(["e1"], turn=10) == set()
    assert le.update_core_membership(["e1"], turn=12) == set()  # e1 not in entries -> pruned
    from augmentum.state.narrative_state import LorebookEntry
    le.set_entries([LorebookEntry(id="e1", keywords=["x"], content="lore")])
    le.update_core_membership(["e1"], turn=13)
    members = le.update_core_membership(["e1"], turn=14)
    assert "e1" in members
    # survives quiet turns inside the idle window
    assert "e1" in le.update_core_membership([], turn=20)
    # demotes after long idle
    assert "e1" not in le.update_core_membership([], turn=14 + le.CORE_DEMOTE_IDLE)
    # round-trips through state dict
    le.update_core_membership(["e1"], turn=50)
    le.update_core_membership(["e1"], turn=51)
    snap = le.to_state_dict()
    le2 = LoreEngine()
    le2.set_entries([LorebookEntry(id="e1", keywords=["x"], content="lore")])
    le2.load_state_dict(snap)
    assert "e1" in le2.update_core_membership([], turn=52)


def test_core_lore_inserted_in_both_live_and_stable():
    from augmentum.models.base import InternalChatRequest, Message
    from augmentum.modes.narrative.context_builder import BuiltContext
    from augmentum.modes.narrative.engine import NarrativeEngine

    eng = NarrativeEngine(session_id="t", context_budget=0)
    req = InternalChatRequest(model="m", messages=[
        Message(role="system", content="CARD"),
        Message(role="user", content="hi"),
    ])
    out = eng._augment_request(
        req, BuiltContext(), context_limit=0,
        supports_mid_system=True, core_lore_text="[World Canon]\n\nLORE",
    )
    # live payload: core sits directly after the card
    assert out.messages[0].content == "CARD"
    assert out.messages[1].content.startswith("[World Canon]")
    # stable snapshot: identical placement (byte-aligned prefix)
    assert out.kv_stable_messages[0].content == "CARD"
    assert out.kv_stable_messages[1].content.startswith("[World Canon]")
    # plain request without core text: untouched shape
    out2 = eng._augment_request(
        req, BuiltContext(), context_limit=0, supports_mid_system=True,
    )
    assert out2.messages[0].content == "CARD"
    assert out2.messages[1].role == "user"
