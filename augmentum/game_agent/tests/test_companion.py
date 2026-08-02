"""Phase 1 + Phase 5 companion-mode tests.

Phase 1 covers:
* PlanPayload accepts and validates ``say`` / ``mood`` / ``intent``
  with sensible defaults.
* parse_plan_output reconciles inconsistent companion field combos
  (say non-empty but intent=silent, or vice versa).
* build_full_prompt with companion=True actually concatenates the
  addendum (so the model sees the new output contract).
* VoiceBridge fails soft — no conn, no provider, raised exceptions
  all degrade to (None, "") instead of taking down the session.

Phase 5 covers:
* CompanionPersona renders an IDENTITY block when name/persona set.
* build_full_prompt prepends the IDENTITY block above the strict
  planner rules so the planner contract is the last thing the model
  reads.
* VoiceBridge.synthesize accepts a per-call voice override and threads
  it down to tts_synthesize_bytes (so character cards can carry a
  voice without rebuilding the bridge).
* persona_loader returns a CompanionPersona built from ui_characters
  row data, handling V2 nested data, flat data, and bare cards.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from augmentum.game_agent.companion import (
    CompanionPersona,
    build_identity_prefix,
)
from augmentum.game_agent.persona_loader import load_persona
from augmentum.game_agent.prompt import (
    COMPANION_PROMPT_ADDENDUM,
    SLOW_PATH_PROMPT,
    build_full_prompt,
    parse_plan_output,
)
from augmentum.game_agent.schema import PlanPayload, SurfaceCapsPayload
from augmentum.game_agent.voice_bridge import VoiceBridge


def _caps() -> SurfaceCapsPayload:
    return SurfaceCapsPayload(
        semantic_inputs=["a", "b"],
        log_schema="mock.v1",
        observation_modalities=["log"],  # type: ignore[list-item]
    )


# ── Schema ────────────────────────────────────────────────────────────


def test_plan_payload_companion_fields_default_to_silent() -> None:
    """@example: a plain plan has empty say + neutral mood + silent intent."""

    plan = PlanPayload(confidence=0.5, next_check_in_ms=1000)
    assert plan.say == ""
    assert plan.mood == "neutral"
    assert plan.intent == "silent"


def test_plan_payload_accepts_companion_fields() -> None:
    """@example: filled companion fields round-trip via the validator."""

    plan = PlanPayload.model_validate({
        "observations": [], "state_update": "", "actions": [],
        "confidence": 0.7, "next_check_in_ms": 1500,
        "say": "Watch out for that Onix!",
        "mood": "concerned",
        "intent": "react",
    })
    assert plan.say == "Watch out for that Onix!"
    assert plan.mood == "concerned"
    assert plan.intent == "react"


# ── Prompt ────────────────────────────────────────────────────────────


def test_companion_prompt_includes_addendum() -> None:
    """@example: companion=True attaches the addendum; False does not."""

    base = build_full_prompt(
        companion=False, surface_kind="mock", caps=_caps(),
        objective="x", state="", live_log_tail=[], n_frames=0,
    )
    extended = build_full_prompt(
        companion=True, surface_kind="mock", caps=_caps(),
        objective="x", state="", live_log_tail=[], n_frames=0,
    )
    assert "COMPANION MODE" not in base
    assert SLOW_PATH_PROMPT in base and SLOW_PATH_PROMPT in extended
    # Take a verbatim slice of the addendum so we know it actually
    # made it in (not just by accident sharing words with the base).
    needle = COMPANION_PROMPT_ADDENDUM.strip().splitlines()[0]
    assert needle in extended


# ── Output reconciliation ─────────────────────────────────────────────


def test_parse_plan_reconciles_say_with_silent_intent() -> None:
    """@example: say non-empty + intent=silent is internally contradictory.

    ROOT CAUSE:
      Models occasionally fill ``say`` but forget to flip ``intent``.
      We don't want to drop the utterance over a label mismatch.
      Reconciling at the parser keeps the rest of the pipeline simple:
      the orchestrator's "should we speak?" gate becomes a single
      ``plan.intent != 'silent'`` check.
    """

    raw = json.dumps({
        "observations": [], "state_update": "", "actions": [],
        "confidence": 0.5, "next_check_in_ms": 1000,
        "say": "hello",
        "intent": "silent",
    })
    plan = parse_plan_output(raw, _caps())
    assert plan.say == "hello"
    assert plan.intent == "chat"   # bumped, not silent


def test_parse_plan_reconciles_empty_say_with_nonsilent_intent() -> None:
    """@example: empty say + intent=chat collapses to silent."""

    raw = json.dumps({
        "observations": [], "state_update": "", "actions": [],
        "confidence": 0.5, "next_check_in_ms": 1000,
        "say": "",
        "intent": "chat",
    })
    plan = parse_plan_output(raw, _caps())
    assert plan.say == ""
    assert plan.intent == "silent"


# ── VoiceBridge — fails soft ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_voice_bridge_returns_none_when_conn_missing() -> None:
    """@example: no state-manager conn -> (None, '') and no exception."""

    bridge = VoiceBridge(lambda: None)
    audio, mime = await bridge.synthesize("anything")
    assert audio is None
    assert mime == ""


@pytest.mark.asyncio
async def test_voice_bridge_returns_none_for_empty_text() -> None:
    """@example: empty text is a no-op (no TTS call attempted)."""

    bridge = VoiceBridge(lambda: object())  # would crash if called
    audio, mime = await bridge.synthesize("   ")
    assert audio is None
    assert mime == ""


@pytest.mark.asyncio
async def test_voice_bridge_synthesize_b64_returns_empty_on_failure() -> None:
    """@example: b64 wrapper preserves the soft-fail contract."""

    bridge = VoiceBridge(lambda: None)
    b64, mime = await bridge.synthesize_b64("hello")
    assert b64 == ""
    assert mime == ""


# ── Phase 5: CompanionPersona + IDENTITY block ─────────────────────────


def test_identity_prefix_empty_when_no_persona() -> None:
    """@example: a missing or empty persona yields no IDENTITY block."""

    assert build_identity_prefix(None) == ""
    assert build_identity_prefix(CompanionPersona()) == ""


def test_identity_prefix_contains_name_and_persona() -> None:
    """@example: a full persona renders a self-contained IDENTITY block."""

    persona = CompanionPersona(
        name="Aria",
        persona="Curious, encouraging, loves Pokémon trivia.",
    )
    prefix = build_identity_prefix(persona)
    assert prefix.startswith("IDENTITY\n")
    assert "You are Aria." in prefix
    assert "Curious, encouraging" in prefix
    # The IDENTITY block ends with a blank-line separator so the
    # planner rules following it parse as a separate section.
    assert prefix.endswith("\n\n")


def test_build_full_prompt_orders_identity_then_rules_then_addendum() -> None:
    """@example: IDENTITY -> SLOW_PATH_PROMPT -> COMPANION addendum -> inputs.

    ROOT CAUSE:
      The strict-JSON contract has to be the last thing the model
      reads; otherwise some models drift back into prose after the
      IDENTITY paragraph. Putting IDENTITY above SLOW_PATH_PROMPT
      keeps the schema instructions terminal.
    """

    persona = CompanionPersona(name="Aria", persona="Curious.")
    prompt = build_full_prompt(
        companion=True,
        surface_kind="mock",
        caps=_caps(),
        objective="x",
        state="",
        live_log_tail=[],
        n_frames=0,
        persona=persona,
    )
    identity_idx = prompt.find("IDENTITY")
    rules_idx = prompt.find("ROLE")
    addendum_idx = prompt.find("COMPANION MODE")
    inputs_idx = prompt.find("SURFACE_KIND:")
    assert 0 <= identity_idx < rules_idx < addendum_idx < inputs_idx


def test_build_full_prompt_drops_identity_when_companion_false() -> None:
    """@example: a persona supplied to a solo session does not leak into the prompt."""

    persona = CompanionPersona(name="Aria", persona="Curious.")
    prompt = build_full_prompt(
        companion=False,
        surface_kind="mock",
        caps=_caps(),
        objective="x",
        state="",
        live_log_tail=[],
        n_frames=0,
        persona=persona,
    )
    assert "IDENTITY" not in prompt
    assert "You are Aria" not in prompt


# ── Phase 5: VoiceBridge voice override ────────────────────────────────


class _StubConn:
    """Sentinel object; persona_loader only uses .execute / .fetchone."""

    def __init__(self, rows: dict[tuple, tuple[str, str] | None]) -> None:
        self._rows = rows
        self.last_query: str = ""
        self.last_params: list[Any] = []

    async def execute(self, query: str, params: list[Any]):
        self.last_query = query
        self.last_params = list(params)
        return _StubCursor(self._rows.get(tuple(params)))


class _StubCursor:
    def __init__(self, row):
        self._row = row

    async def fetchone(self):
        return self._row


@pytest.mark.asyncio
async def test_voice_bridge_passes_per_call_voice_override(monkeypatch) -> None:
    """@example: synthesize(voice='custom-voice') threads to tts_synthesize_bytes.

    ROOT CAUSE:
      Phase 5 wants per-session voices without rebuilding the shared
      bridge. The override has to win over the constructor default,
      and ``None`` has to fall through cleanly.
    """

    seen_voices: list[str] = []

    async def fake_tts(conn, text, *, voice, speed, response_format):
        seen_voices.append(voice)
        return (b"AUDIO", True)

    monkeypatch.setattr(
        "augmentum.proxy.audio_routes.tts_synthesize_bytes", fake_tts,
    )
    bridge = VoiceBridge(lambda: object(), default_voice="default")
    await bridge.synthesize("hello", voice="custom")
    await bridge.synthesize("hello")  # falls back to default
    await bridge.synthesize("hello", voice=None)  # explicit None == default
    assert seen_voices == ["custom", "default", "default"]


# ── Phase 5: persona_loader ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_persona_returns_none_for_missing_character() -> None:
    """@example: unknown id -> None, caller falls back to anonymous mode."""

    conn = _StubConn({})
    result = await load_persona(conn, "missing-id", user_id="u1")
    assert result is None


@pytest.mark.asyncio
async def test_load_persona_extracts_name_persona_voice_from_flat_card() -> None:
    """@example: flat card JSON -> name + personality summary + voice."""

    data = json.dumps({
        "name": "Aria",
        "personality": "Curious, encouraging, loves Pokémon trivia.",
        "voice": "af_heart",
    })
    conn = _StubConn({("char-1", "u1"): ("Aria", data)})
    persona = await load_persona(conn, "char-1", user_id="u1")
    assert persona is not None
    assert persona.name == "Aria"
    assert "Curious" in persona.persona
    assert persona.voice == "af_heart"


@pytest.mark.asyncio
async def test_load_persona_unwraps_v2_nested_data() -> None:
    """@example: V2 cards with {"data": {...}} get flattened.

    ROOT CAUSE:
      Character cards in the V2 spec carry the actual fields under a
      nested ``data`` key. Without flattening, the loader would see an
      empty persona and the model would never get to introduce itself.
    """

    raw = json.dumps({
        "spec": "chara_card_v2",
        "data": {
            "name": "Aria",
            "personality": "Curious.",
            "scenario": "Co-piloting Pokémon Red together.",
        },
        "voice": "qwen-tts::Vivian",
    })
    conn = _StubConn({("char-2", "u1"): ("Aria-row-name", raw)})
    persona = await load_persona(conn, "char-2", user_id="u1")
    assert persona is not None
    assert persona.name == "Aria"
    assert "Curious" in persona.persona
    assert "Co-piloting" in persona.persona
    assert persona.voice == "qwen-tts::Vivian"


@pytest.mark.asyncio
async def test_load_persona_uses_row_name_when_data_missing_name() -> None:
    """@example: when the JSON has no name, the ui_characters.name column wins."""

    data = json.dumps({"personality": "Curious."})
    conn = _StubConn({("char-3", "u1"): ("RowName", data)})
    persona = await load_persona(conn, "char-3", user_id="u1")
    assert persona is not None
    assert persona.name == "RowName"


@pytest.mark.asyncio
async def test_load_persona_tolerates_broken_json() -> None:
    """@example: a character row with corrupt data JSON degrades to row name only."""

    conn = _StubConn({("char-4", "u1"): ("Aria", "not-json")})
    persona = await load_persona(conn, "char-4", user_id="u1")
    assert persona is not None
    assert persona.name == "Aria"
    assert persona.persona == ""
    assert persona.voice == ""
