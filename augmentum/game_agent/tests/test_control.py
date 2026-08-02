"""Control-schema tests — vocabulary, profile loading, composition.

These exercise the three-layer abstraction end-to-end without any
running adapter or session: pure data validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from augmentum.game_agent.control import (
    UNIVERSAL_ACTIONS,
    ProfileRegistry,
    default_registry,
)
from augmentum.game_agent.control.actions import is_universal_action
from augmentum.game_agent.control.controllers import (
    CONTROLLER_INPUTS,
    is_keyboard_code,
    is_known_controller_input,
)
from augmentum.game_agent.control.profile import (
    ProfileLoadError,
    load_controller_profile,
    load_game_profile,
)

# ── Vocabulary stability ──────────────────────────────────────────────


def test_universal_actions_includes_navigation_and_ui() -> None:
    """@example: the core verbs are present in the universal vocabulary.

    ROOT CAUSE:
      A future refactor could silently rename one of these. Asserting
      the canonical set here lets prompt rendering / tests rely on
      the names being stable.
    """

    must_have = {
        "nav_up", "nav_down", "nav_left", "nav_right",
        "confirm", "cancel", "menu",
        "interact", "attack",
    }
    missing = must_have - UNIVERSAL_ACTIONS
    assert missing == set(), f"universal vocabulary missing: {missing}"


def test_is_universal_action_distinguishes_extras() -> None:
    """@example: a game-specific action name is NOT in the universal set."""

    assert is_universal_action("confirm") is True
    assert is_universal_action("registered") is False  # Pokémon RS extension
    assert is_universal_action("") is False


def test_controller_inputs_includes_positional_face_buttons() -> None:
    """@example: SDL positional names are present; letter names are not.

    The positional convention is load-bearing: we DON'T want
    'face_a' in the vocabulary because 'A' isn't in the same physical
    slot across consoles.
    """

    assert "face_south" in CONTROLLER_INPUTS
    assert "face_east" in CONTROLLER_INPUTS
    assert "face_west" in CONTROLLER_INPUTS
    assert "face_north" in CONTROLLER_INPUTS
    assert "face_a" not in CONTROLLER_INPUTS  # deliberately not in the schema
    assert "face_b" not in CONTROLLER_INPUTS


def test_is_known_controller_input_accepts_layer2_names() -> None:
    """@example: layer-2 names recognize as known; arbitrary strings do not."""

    assert is_known_controller_input("dpad_up") is True
    assert is_known_controller_input("mouse_left") is True  # pointer
    assert is_known_controller_input("not_a_button") is False


def test_keyboard_code_validation() -> None:
    """@example: KeyboardEvent.code-shaped strings pass, junk doesn't."""

    assert is_keyboard_code("Space") is True
    assert is_keyboard_code("ArrowLeft") is True
    assert is_keyboard_code("KeyA") is True
    assert is_keyboard_code("Digit5") is True
    assert is_keyboard_code("") is False
    assert is_keyboard_code("9starts_with_digit") is False
    assert is_keyboard_code("has space") is False


# ── Profile loading from JSON ─────────────────────────────────────────


def _write(tmp_path: Path, relpath: str, body: dict) -> Path:
    p = tmp_path / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


def test_controller_profile_loads_clean_json(tmp_path: Path) -> None:
    """@example: a well-formed GBA-shaped profile loads + parses."""

    path = _write(tmp_path, "ctrl.json", {
        "schema": "control.controller.v1",
        "id": "test_gba",
        "description": "",
        "wire": {"kind": "libretro_joypad", "port": 0},
        "buttons": {
            "face_east":  {"wire_code": 8, "label": "A"},
            "face_south": {"wire_code": 0, "label": "B"},
            "dpad_up":    {"wire_code": 4, "label": "Up"},
            "start":      {"wire_code": 3, "label": "Start"},
        },
    })
    profile = load_controller_profile(path)
    assert profile.id == "test_gba"
    assert profile.wire_kind == "libretro_joypad"
    assert profile.has("face_east")
    assert profile.wire_for("face_east").wire_code == 8


def test_controller_profile_rejects_wrong_schema_version(tmp_path: Path) -> None:
    """@example: a v2-claiming profile fails to load against v1 validator."""

    path = _write(tmp_path, "ctrl.json", {
        "schema": "control.controller.v2",   # not yet defined
        "id": "future",
        "wire": {"kind": "libretro_joypad"},
        "buttons": {},
    })
    with pytest.raises(ProfileLoadError):
        load_controller_profile(path)


def test_controller_profile_rejects_unknown_wire_kind(tmp_path: Path) -> None:
    """@example: an unknown wire transport surfaces a clear error."""

    path = _write(tmp_path, "ctrl.json", {
        "schema": "control.controller.v1",
        "id": "weird",
        "wire": {"kind": "morse_code", "port": 0},
        "buttons": {"face_east": {"wire_code": 1, "label": "X"}},
    })
    with pytest.raises(ProfileLoadError, match="unknown wire.kind"):
        load_controller_profile(path)


def test_controller_profile_rejects_unknown_button_name(tmp_path: Path) -> None:
    """@example: garbage Layer-2 input name on a joypad profile errors out."""

    path = _write(tmp_path, "ctrl.json", {
        "schema": "control.controller.v1",
        "id": "x",
        "wire": {"kind": "libretro_joypad"},
        "buttons": {"banana_split": {"wire_code": 1, "label": "X"}},
    })
    with pytest.raises(ProfileLoadError, match="not a known Layer-2"):
        load_controller_profile(path)


def test_keyboard_profile_accepts_keyevent_codes(tmp_path: Path) -> None:
    """@example: a keyboard-wire profile takes arbitrary KeyboardEvent.code names."""

    path = _write(tmp_path, "kbd.json", {
        "schema": "control.controller.v1",
        "id": "pc_keyboard",
        "wire": {"kind": "keyboard"},
        "buttons": {
            "Space":     {"wire_code": "Space",     "label": "Space"},
            "ArrowLeft": {"wire_code": "ArrowLeft", "label": "←"},
            "KeyA":      {"wire_code": "KeyA",      "label": "A"},
        },
    })
    profile = load_controller_profile(path)
    assert profile.has("Space")
    assert profile.wire_for("ArrowLeft").wire_code == "ArrowLeft"


def test_game_profile_loads_clean_json(tmp_path: Path) -> None:
    """@example: a well-formed game profile parses and exposes its actions."""

    path = _write(tmp_path, "game.json", {
        "schema": "control.game.v1",
        "id": "test_game",
        "applies_to_controllers": ["gba"],
        "applies_to_log_schema": "test.v1",
        "actions": {
            "confirm": {"binding": "face_east", "hint": "advance"},
            "nav_up":  {"binding": "dpad_up", "hint": "north"},
        },
    })
    g = load_game_profile(path)
    assert g.id == "test_game"
    assert "confirm" in g.actions
    assert g.actions["confirm"].is_universal is True
    assert g.actions["confirm"].hint == "advance"


def test_game_profile_rejects_bad_action_name(tmp_path: Path) -> None:
    """@example: an action named with spaces or punctuation is rejected."""

    path = _write(tmp_path, "game.json", {
        "schema": "control.game.v1",
        "id": "bad",
        "applies_to_controllers": ["gba"],
        "actions": {
            "do something!": {"binding": "face_east", "hint": ""},
        },
    })
    with pytest.raises(ProfileLoadError, match="must be"):
        load_game_profile(path)


# ── Composition ───────────────────────────────────────────────────────


def test_compose_succeeds_for_compatible_pair() -> None:
    """@example: the shipped GBA + pokemon_rs profiles compose cleanly."""

    composed = default_registry.compose("gba", "pokemon_rs")
    inputs = composed.semantic_inputs()
    # Universal verbs are present
    for verb in ("confirm", "cancel", "menu", "nav_up", "nav_left"):
        assert verb in inputs, f"{verb!r} not in composed inputs: {inputs}"
    # Game-specific extension
    assert "registered" in inputs
    # Hardware passthrough
    assert "shoulder_l" in inputs
    assert "shoulder_r" in inputs


def test_compose_resolves_to_correct_wire_code() -> None:
    """@example: confirm → face_east → libretro button id 8 (A on GBA)."""

    composed = default_registry.compose("gba", "pokemon_rs")
    btn = composed.resolve("confirm")
    assert btn is not None
    assert btn.wire_code == 8
    assert btn.label == "A"


def test_compose_resolves_hardware_passthrough() -> None:
    """@example: shoulder_l is exposed even though it's not a universal action."""

    composed = default_registry.compose("gba", "pokemon_rs")
    btn = composed.resolve("shoulder_l")
    assert btn is not None
    assert btn.wire_code == 10
    assert btn.label == "L"


def test_compose_unknown_semantic_returns_none() -> None:
    """@example: a semantic the profile doesn't expose returns None.

    Callers (the resolver) treat None as the equivalent of an
    UnknownSemanticError -- the agent gets a clean failure path.
    """

    composed = default_registry.compose("gba", "pokemon_rs")
    assert composed.resolve("triple_jump") is None
    assert composed.resolve("") is None


def test_compose_rejects_unsupported_controller() -> None:
    """@example: pokemon_rs declared for GBA only doesn't compose with gambatte."""

    with pytest.raises(ProfileLoadError, match="does not declare support"):
        default_registry.compose("gambatte", "pokemon_rs")


def test_compose_pokemon_rby_on_gambatte() -> None:
    """@example: Pokémon RBY composes cleanly with gambatte (no shoulder buttons).

    ROOT CAUSE:
      RBY's profile must NOT reference shoulder_l/r since the gambatte
      controller doesn't expose them. This guards against a regression
      where someone copies an RS-shaped profile for RBY and leaves the
      shoulders in.
    """

    composed = default_registry.compose("gambatte", "pokemon_rby")
    inputs = composed.semantic_inputs()
    assert "confirm" in inputs
    assert "nav_up" in inputs
    assert "shoulder_l" not in inputs  # not on a Game Boy
    btn = composed.resolve("confirm")
    assert btn.wire_code == 8


def test_compose_pokemon_gsc_on_gambatte() -> None:
    """@example: Pokémon GSC composes on gambatte and exposes the Gen-2 verbs.

    Gen-2 adds the ``registered`` key-item slot (mapped to Select) on
    top of the Gen-1 vocabulary; both should be present, and neither
    should pull in shoulders the GBC doesn't have.
    """

    composed = default_registry.compose("gambatte", "pokemon_gsc")
    inputs = composed.semantic_inputs()
    assert "confirm" in inputs
    assert "cancel" in inputs
    assert "menu" in inputs
    assert "registered" in inputs
    assert "shoulder_l" not in inputs
    # ``registered`` lands on the Select button per the GSC profile.
    registered_btn = composed.resolve("registered")
    assert registered_btn is not None
    # Gambatte's Select sits at libretro joypad id 2.
    assert registered_btn.wire_code == 2


def test_compose_zelda_la_on_gambatte() -> None:
    """@example: Link's Awakening DX composes cleanly with gambatte.

    The Zelda profile binds ``map`` to Select rather than ``registered``;
    asserting both keeps the action vocabulary obvious to readers.
    """

    composed = default_registry.compose("gambatte", "zelda_links_awakening_dx")
    inputs = composed.semantic_inputs()
    assert "confirm" in inputs
    assert "cancel" in inputs
    assert "menu" in inputs
    assert "map" in inputs
    map_btn = composed.resolve("map")
    assert map_btn is not None
    assert map_btn.wire_code == 2  # gambatte Select


def test_default_registry_has_all_shipped_game_profiles() -> None:
    """@example: every bundled game profile id is registered for runtime use.

    Acts as a smoke test: if someone drops a new game profile JSON in
    augmentum/game_agent/control/profiles/games/ but breaks its schema
    so the loader skips it, this assertion fails loudly at test time.
    """

    ids = set(default_registry.game_ids())
    assert {
        "pokemon_rby", "pokemon_rs", "pokemon_gsc",
        "zelda_links_awakening_dx",
        "generic_nes", "generic_snes", "generic_genesis",
        "generic_sms", "generic_gg", "generic_pce",
        "generic_psx", "generic_nds",
    }.issubset(ids)


def test_default_registry_has_all_shipped_controller_profiles() -> None:
    """@example: every bundled controller profile id is loaded.

    Mirrors the game-profile smoke test for the controller side. If a
    new controller JSON breaks the loader silently, this catches it.
    """

    ids = set(default_registry.controller_ids())
    assert {
        "gba", "gambatte",
        "nes", "snes", "genesis", "sms", "gg", "pce", "psx", "nds",
    }.issubset(ids)


# Parametric-style pair tests: every K-series controller composes with its
# matching generic game profile. Asserted button presence catches mis-typed
# bindings inside the JSON (validator catches missing controllers; this
# catches a generic profile that accidentally references face_west on a
# controller that only has face_south/east).


def test_compose_generic_nes_on_nes() -> None:
    """@example: generic_nes composes on nes and exposes the 2-button verbs."""

    composed = default_registry.compose("nes", "generic_nes")
    inputs = composed.semantic_inputs()
    assert {"confirm", "cancel", "menu", "nav_up", "nav_down"}.issubset(inputs)
    assert "special" not in inputs  # NES has no face_west
    assert composed.resolve("confirm").wire_code == 8  # NES A


def test_compose_generic_snes_on_snes() -> None:
    """@example: generic_snes binds all four face buttons + shoulders passthrough."""

    composed = default_registry.compose("snes", "generic_snes")
    inputs = composed.semantic_inputs()
    assert {
        "confirm", "cancel", "special", "use_item",
        "menu", "back", "shoulder_l", "shoulder_r",
    }.issubset(inputs)
    assert composed.resolve("confirm").wire_code == 8   # SNES A
    assert composed.resolve("special").wire_code == 1    # SNES Y
    assert composed.resolve("use_item").wire_code == 9   # SNES X


def test_compose_generic_genesis_on_genesis() -> None:
    """@example: generic_genesis binds the 3-button horizontal layout."""

    composed = default_registry.compose("genesis", "generic_genesis")
    inputs = composed.semantic_inputs()
    assert {"confirm", "cancel", "special", "menu", "nav_up"}.issubset(inputs)
    assert "back" not in inputs   # genesis has no select
    assert "use_item" not in inputs  # genesis has no face_north in 3-button
    assert composed.resolve("confirm").wire_code == 8   # Genesis C
    assert composed.resolve("special").wire_code == 1   # Genesis A (face_west)


def test_compose_generic_sms_on_sms() -> None:
    """@example: SMS has no select; profile must not bind 'back'."""

    composed = default_registry.compose("sms", "generic_sms")
    inputs = composed.semantic_inputs()
    assert "confirm" in inputs
    assert "pause" in inputs
    assert "back" not in inputs


def test_compose_generic_gg_on_gg() -> None:
    """@example: Game Gear binds menu to Start (handheld button)."""

    composed = default_registry.compose("gg", "generic_gg")
    inputs = composed.semantic_inputs()
    assert "confirm" in inputs
    assert "menu" in inputs
    # GG has no select.
    assert "back" not in inputs


def test_compose_generic_pce_on_pce() -> None:
    """@example: PCE inverts the Nintendo confirm/cancel convention.

    The PC Engine's I button (libretro id 0 / face_south) is the
    primary action; the profile must bind confirm to face_south.
    """

    composed = default_registry.compose("pce", "generic_pce")
    inputs = composed.semantic_inputs()
    assert "confirm" in inputs
    # I (libretro 0) is on face_south; confirm should land there.
    assert composed.resolve("confirm").wire_code == 0


def test_compose_generic_psx_on_psx() -> None:
    """@example: Western PSX convention binds confirm to face_south (X).

    Plus all shoulder/trigger/stick-click buttons are surfaced as
    hardware passthrough.
    """

    composed = default_registry.compose("psx", "generic_psx")
    inputs = composed.semantic_inputs()
    assert composed.resolve("confirm").wire_code == 0   # PSX Cross
    assert composed.resolve("cancel").wire_code == 8    # PSX Circle
    assert {
        "shoulder_l", "shoulder_r", "trigger_l", "trigger_r",
        "stick_l_press", "stick_r_press",
    }.issubset(inputs)


def test_compose_generic_nds_on_nds() -> None:
    """@example: NDS uses SNES-shape diamond and inherits Nintendo confirm convention."""

    composed = default_registry.compose("nds", "generic_nds")
    inputs = composed.semantic_inputs()
    assert composed.resolve("confirm").wire_code == 8   # NDS A
    assert composed.resolve("cancel").wire_code == 0    # NDS B
    assert {"shoulder_l", "shoulder_r"}.issubset(inputs)


def test_compose_rejects_binding_to_missing_controller_input(tmp_path: Path) -> None:
    """@example: a game profile binding to face_north on a 2-face controller errors.

    The compose() call validates that EVERY game binding lands on a
    button the controller actually exposes. Catches the most common
    "I copied a Pokémon RS profile to make a Switch RPG one" error
    at registration time, not at the first session POST.
    """

    reg = ProfileRegistry()
    reg.register_controller(load_controller_profile(
        _write(tmp_path, "controllers/limited.json", {
            "schema": "control.controller.v1",
            "id": "limited",
            "wire": {"kind": "libretro_joypad"},
            "buttons": {
                "face_east":  {"wire_code": 8, "label": "A"},
                "face_south": {"wire_code": 0, "label": "B"},
            },
        }),
    ))
    reg.register_game(load_game_profile(
        _write(tmp_path, "games/needs_more.json", {
            "schema": "control.game.v1",
            "id": "needs_more",
            "applies_to_controllers": ["limited"],
            "actions": {
                "confirm": {"binding": "face_east", "hint": ""},
                "special": {"binding": "face_north", "hint": ""},  # not on limited
            },
        }),
    ))
    with pytest.raises(ProfileLoadError, match="binds to.*does not expose"):
        reg.compose("limited", "needs_more")


# ── Hints rendering ───────────────────────────────────────────────────


def test_hints_use_per_game_text_when_present() -> None:
    """@example: pokemon_rs's confirm hint mentions advancing dialog."""

    composed = default_registry.compose("gba", "pokemon_rs")
    hints = composed.hints()
    assert "advance dialog" in hints["confirm"].lower()


def test_hints_describe_passthrough_with_controller_label() -> None:
    """@example: hardware passthrough hint shows the human button label."""

    composed = default_registry.compose("gba", "pokemon_rs")
    hints = composed.hints()
    assert "L" in hints["shoulder_l"]


# ── Registry behavior ────────────────────────────────────────────────


def test_default_registry_loaded_bundled_profiles() -> None:
    """@example: the shipped registry knows the GBA / GB controllers + first games."""

    assert "gba" in default_registry.controller_ids()
    assert "gambatte" in default_registry.controller_ids()
    assert "pokemon_rs" in default_registry.game_ids()
    assert "pokemon_rby" in default_registry.game_ids()


def test_registry_lookup_unknown_raises_with_known_list() -> None:
    """@example: an unknown id surfaces a list of known ones for the user."""

    reg = ProfileRegistry()
    with pytest.raises(ProfileLoadError, match="known: \\[\\]"):
        reg.controller("nope")


def test_registry_overrides_replace_prior_entry(tmp_path: Path) -> None:
    """@example: a later register_controller call wins on id collision.

    Operator-supplied profiles can override bundled ones without
    forking the package.
    """

    reg = ProfileRegistry()
    base = load_controller_profile(_write(tmp_path, "a.json", {
        "schema": "control.controller.v1",
        "id": "x", "wire": {"kind": "libretro_joypad"},
        "buttons": {"face_east": {"wire_code": 8, "label": "Old"}},
    }))
    override = load_controller_profile(_write(tmp_path, "b.json", {
        "schema": "control.controller.v1",
        "id": "x", "wire": {"kind": "libretro_joypad"},
        "buttons": {"face_east": {"wire_code": 8, "label": "New"}},
    }))
    reg.register_controller(base)
    reg.register_controller(override)
    assert reg.controller("x").wire_for("face_east").label == "New"
