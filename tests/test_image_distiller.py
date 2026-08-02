"""Tests for the image prompt distiller."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from augmentum.image.distiller import (
    DISTILLER_SYSTEM_PROMPT,
    SceneContext,
    UserPersona,
    _extract_conversation_rounds,
    build_distiller_prompt,
    distill_scene,
    parse_distiller_response,
)
from augmentum.modes.narrative.card_parser import CharacterCard


def _make_ctx(
    *,
    location: str = "",
    time_of_day: str = "",
    weather: str = "",
    atmosphere: str = "",
    character_card: CharacterCard | None = None,
    state_snapshot_fields: dict[str, str] | None = None,
    memory_ledger: list | None = None,
) -> SceneContext:
    """Build a SceneContext directly for prompt-builder tests."""
    return SceneContext(
        character_card=character_card,
        state_snapshot_fields=state_snapshot_fields or {},
        legacy_world_state=SimpleNamespace(
            location=location,
            time_of_day=time_of_day,
            weather=weather,
            atmosphere=atmosphere,
        ),
        memory_ledger=memory_ledger or [],
        image_style=character_card.image_style if character_card else "",
    )


class TestBuildDistillerPrompt:
    def test_empty_state(self):
        ctx = _make_ctx()
        prompt = build_distiller_prompt(ctx)
        assert "Analyze" in prompt

    def test_scene_context_included(self):
        ctx = _make_ctx(
            location="Dark forest",
            time_of_day="night",
            weather="foggy",
            atmosphere="eerie",
        )
        prompt = build_distiller_prompt(ctx)
        assert "Dark forest" in prompt
        assert "night" in prompt
        assert "foggy" in prompt
        assert "eerie" in prompt

    def test_state_snapshot_takes_priority_over_world_state(self):
        ctx = _make_ctx(
            location="Old location",
            state_snapshot_fields={"location": "New location", "time_of_day": "dusk"},
        )
        prompt = build_distiller_prompt(ctx)
        assert "New location" in prompt
        assert "dusk" in prompt
        # Old legacy world state should be hidden when snapshot is present
        assert "Old location" not in prompt

    def test_memory_ledger_included(self):
        ledger = [
            SimpleNamespace(round_num=3, content="The dragon awoke."),
            SimpleNamespace(round_num=4, content="Elena drew her blade."),
        ]
        ctx = _make_ctx(memory_ledger=ledger)
        prompt = build_distiller_prompt(ctx)
        assert "RECENT KEY EVENTS" in prompt
        assert "dragon awoke" in prompt
        assert "Elena drew" in prompt

    def test_user_instruction(self):
        ctx = _make_ctx()
        prompt = build_distiller_prompt(ctx, user_instruction="focus on the sunset")
        assert "focus on the sunset" in prompt

    def test_character_card_included(self):
        card = CharacterCard(
            name="Aria",
            species="Elf",
            appearance="Silver hair, blue eyes, tall and slender",
            personality="Brave and compassionate",
            description="A ranger from the northern woods",
            scenario="High fantasy adventure",
        )
        ctx = _make_ctx(character_card=card)
        prompt = build_distiller_prompt(ctx)
        assert "Aria" in prompt
        assert "Silver hair" in prompt
        assert "Elf" in prompt
        assert "ranger" in prompt
        assert "CHARACTER CARD" in prompt

    def test_user_persona_included(self):
        ctx = _make_ctx()
        persona = UserPersona(
            name="Marcus",
            appearance="Tall, dark hair, green eyes, leather armor",
            description="A wandering mercenary",
        )
        prompt = build_distiller_prompt(ctx, persona=persona)
        assert "Marcus" in prompt
        assert "leather armor" in prompt
        assert "USER/PLAYER CHARACTER" in prompt

    def test_core_profile_included(self):
        ctx = _make_ctx()
        prompt = build_distiller_prompt(ctx, core_profile="[core_memory]\n- User prefers dark themes")
        assert "USER PROFILE FACTS" in prompt
        assert "dark themes" in prompt

    def test_conversation_messages_included(self):
        ctx = _make_ctx()
        messages = [
            {"role": "user", "content": "I draw my sword and charge at the goblin."},
            {"role": "assistant", "content": "The goblin screeches and raises its crude shield."},
        ]
        prompt = build_distiller_prompt(ctx, conversation_messages=messages)
        assert "draw my sword" in prompt
        assert "goblin screeches" in prompt
        assert "RECENT CONVERSATION" in prompt

    def test_full_context_all_sections(self):
        card = CharacterCard(name="Elena", visual_traits="Red hair, green eyes")
        ctx = _make_ctx(
            location="Castle courtyard",
            character_card=card,
        )
        persona = UserPersona(name="Player", appearance="Knight in plate armor")
        messages = [{"role": "user", "content": "I challenge Elena to a duel."}]

        prompt = build_distiller_prompt(
            ctx,
            user_instruction="dramatic duel scene",
            conversation_messages=messages,
            persona=persona,
            core_profile="[core_memory]\n- Enjoys combat scenes",
        )
        assert "Elena" in prompt
        assert "Red hair" in prompt
        assert "Knight in plate armor" in prompt
        assert "challenge Elena" in prompt
        assert "dramatic duel scene" in prompt
        assert "Castle courtyard" in prompt
        assert "combat scenes" in prompt


class TestExtractConversationRounds:
    def test_extract_last_two_rounds(self):
        from augmentum.models.base import InternalChatRequest, Message
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="system", content="System prompt"),
                Message(role="user", content="First message"),
                Message(role="assistant", content="First response"),
                Message(role="user", content="Second message"),
                Message(role="assistant", content="Second response"),
                Message(role="user", content="Third message"),
                Message(role="assistant", content="Third response"),
            ],
        )
        result = _extract_conversation_rounds(request, rounds=2)
        assert len(result) == 4
        assert result[0]["content"] == "Second message"
        assert result[-1]["content"] == "Third response"

    def test_extract_fewer_messages_than_rounds(self):
        from augmentum.models.base import InternalChatRequest, Message
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="user", content="Only message"),
                Message(role="assistant", content="Only response"),
            ],
        )
        result = _extract_conversation_rounds(request, rounds=5)
        assert len(result) == 2

    def test_skips_system_messages(self):
        from augmentum.models.base import InternalChatRequest, Message
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="system", content="System"),
                Message(role="user", content="Hi"),
                Message(role="assistant", content="Hello"),
            ],
        )
        result = _extract_conversation_rounds(request, rounds=2)
        assert len(result) == 2
        assert all(m["role"] != "system" for m in result)


class TestCharacterCardImageModel:
    def test_image_model_parsed_from_v2_json(self):
        from augmentum.modes.narrative.card_parser import CardParser
        parser = CardParser()
        card_json = '{"spec": "chara_card_v2", "data": {"name": "Test", "image_model": "sd_xl_base"}}'
        card = parser.parse(card_json)
        assert card is not None
        assert card.image_model == "sd_xl_base"

    def test_image_model_from_extensions(self):
        from augmentum.modes.narrative.card_parser import CardParser
        parser = CardParser()
        card_json = '{"spec": "chara_card_v2", "data": {"name": "Test", "extensions": {"image_model": "anime_model"}}}'
        card = parser.parse(card_json)
        assert card is not None
        assert card.image_model == "anime_model"

    def test_image_model_top_level_takes_priority(self):
        from augmentum.modes.narrative.card_parser import CardParser
        parser = CardParser()
        card_json = '{"spec": "chara_card_v2", "data": {"name": "Test", "image_model": "top", "extensions": {"image_model": "ext"}}}'
        card = parser.parse(card_json)
        assert card is not None
        assert card.image_model == "top"

    def test_no_image_model_defaults_empty(self):
        card = CharacterCard(name="Test")
        assert card.image_model == ""


class TestParseDistillerResponse:
    def test_standard_response(self):
        response = """POSITIVE: dark forest, moonlit, fog, gothic atmosphere, eerie lighting
NEGATIVE: bright colors, cheerful, modern, blurry
ASPECT: landscape"""
        result = parse_distiller_response(response)
        assert "dark forest" in result["positive"]
        assert "bright colors" in result["negative"]
        assert result["aspect"] == "landscape"

    def test_missing_fields(self):
        response = "POSITIVE: just a prompt"
        result = parse_distiller_response(response)
        assert result["positive"] == "just a prompt"
        assert result["negative"]  # Should have default
        assert result["aspect"] == "square"  # Default

    def test_empty_response(self):
        result = parse_distiller_response("")
        assert result["positive"] == ""
        assert result["aspect"] == "square"

    def test_case_insensitive(self):
        response = """positive: castle, medieval
negative: modern stuff
aspect: portrait"""
        result = parse_distiller_response(response)
        assert "castle" in result["positive"]
        assert result["aspect"] == "portrait"

    def test_extra_whitespace(self):
        response = "  POSITIVE:  a beautiful scene  \n  NEGATIVE:  ugly stuff  \n  ASPECT:  square  "
        result = parse_distiller_response(response)
        assert result["positive"] == "a beautiful scene"
        assert result["negative"] == "ugly stuff"
        assert result["aspect"] == "square"

    def test_invalid_aspect_falls_back(self):
        response = "POSITIVE: test\nASPECT: widescreen"
        result = parse_distiller_response(response)
        assert result["aspect"] == "square"


class TestDistillerSystemPrompt:
    def test_prompt_contains_instructions(self):
        assert "POSITIVE" in DISTILLER_SYSTEM_PROMPT
        assert "NEGATIVE" in DISTILLER_SYSTEM_PROMPT
        assert "ASPECT" in DISTILLER_SYSTEM_PROMPT
        assert "{model_context}" in DISTILLER_SYSTEM_PROMPT


class TestDistillScene:
    @pytest.mark.asyncio
    async def test_distill_success(self):
        ctx = _make_ctx(location="castle", time_of_day="dawn")

        mock_response = AsyncMock()
        mock_response.message = SimpleNamespace(
            content=(
                "POSITIVE: medieval castle, dawn light, golden hour\n"
                "NEGATIVE: modern, blurry\n"
                "ASPECT: landscape"
            ),
        )

        backend = AsyncMock()
        backend.chat = AsyncMock(return_value=mock_response)

        result = await distill_scene(ctx, backend, "test-model")
        assert "castle" in result["positive"]
        assert result["aspect"] == "landscape"

    @pytest.mark.asyncio
    async def test_distill_fallback_on_error(self):
        ctx = _make_ctx(location="forest", atmosphere="mysterious")

        backend = AsyncMock()
        backend.chat = AsyncMock(side_effect=RuntimeError("Backend down"))

        result = await distill_scene(ctx, backend, "test-model")
        # Should fall back to scene state
        assert "forest" in result["positive"]
