"""Tests for narrative extraction: card_parser, regex_transformer, llm_extractor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.modes.narrative.card_parser import CardParser, CharacterCard
from augmentum.modes.narrative.regex_transformer import (
    RegexScript,
    apply_regex_scripts,
)
from augmentum.modes.narrative.llm_extractor import NarrativeExtraction


class TestCharacterCardDefaults:
    """CharacterCard dataclass defaults."""

    def test_default_display_name(self):
        card = CharacterCard()
        assert card.display_name == "Unknown Character"

    def test_display_name_with_name(self):
        card = CharacterCard(name="Luna")
        assert card.display_name == "Luna"

    def test_trait_summary_basic(self):
        card = CharacterCard(name="Luna", species="Elf", personality="Wise and calm")
        summary = card.trait_summary
        assert "Luna" in summary
        assert "Elf" in summary
        assert "Wise" in summary

    def test_trait_summary_empty(self):
        card = CharacterCard()
        assert card.trait_summary == ""


class TestCardParserV2JSON:
    """Parse Character Card V2 JSON format."""

    def test_parse_v2_json(self):
        parser = CardParser()
        data = {
            "spec": "chara_card_v2",
            "data": {
                "name": "Luna",
                "description": "A mysterious elf mage.",
                "personality": "Calm, wise, secretive",
                "scenario": "In a medieval tavern",
                "first_mes": "Hello, traveler.",
                "mes_example": "<START>\nLuna: *adjusts her cloak*",
            },
        }
        card = parser.parse(json.dumps(data))
        assert card is not None
        assert card.name == "Luna"
        assert "mysterious" in card.description.lower()

    def test_parse_returns_none_for_generic_assistant_prompt(self):
        parser = CardParser()
        # Generic assistant prompts without character signals return None
        card = parser.parse("You are a helpful assistant named Bob.")
        assert card is None

    def test_parse_freeform_with_character_signals(self):
        parser = CardParser()
        card = parser.parse(
            "Personality: Brave and loyal\n"
            "Appearance: Tall with dark hair\n"
            "Scenario: A medieval fantasy adventure"
        )
        assert card is not None


class TestCardParserWPP:
    """Parse W++ (Wiki++) format character cards."""

    def test_parse_wpp_format(self):
        parser = CardParser()
        wpp = (
            '[character= "Luna"]\n'
            '[personality= "wise" + "calm" + "mysterious"]\n'
            '[appearance= "silver hair" + "blue eyes"]\n'
            '[species= "elf"]'
        )
        card = parser.parse(wpp)
        assert card is not None
        assert card.name == "Luna"
        assert card.source_format == "wpp"

    def test_parse_plist_format(self):
        parser = CardParser()
        plist = (
            "Name: Luna Silver\n"
            "Species: Elf\n"
            "Personality: Calm, wise, mysterious\n"
            "Appearance: Silver hair, blue eyes\n"
        )
        card = parser.parse(plist)
        assert card is not None
        assert "Luna" in card.name or "luna" in card.name.lower()


class TestCardParserFreeform:
    """Freeform / unstructured system prompts."""

    def test_generic_text_returns_none(self):
        parser = CardParser()
        # No character signals -> None
        card = parser.parse("You are an ancient dragon who guards the mountain.")
        assert card is None

    def test_freeform_with_personality_field(self):
        parser = CardParser()
        card = parser.parse(
            "Personality: Fierce and territorial\n"
            "You are an ancient dragon who guards the mountain."
        )
        assert card is not None

    def test_visual_traits_section_extracted(self):
        parser = CardParser()
        text = (
            "Personality: Wise and ancient\n"
            "[Visual Traits]\n"
            "Silver hair, blue eyes, pale skin\n\n"
            "She is an elf mage."
        )
        card = parser.parse(text)
        assert card is not None
        assert "silver" in card.visual_traits.lower()


class TestRegexScript:
    """RegexScript dataclass behavior."""

    def test_auto_generates_id(self):
        script = RegexScript(name="test", find_regex="foo", replace_string="bar")
        assert script.id != ""
        assert len(script.id) == 12

    def test_default_placement(self):
        script = RegexScript()
        assert script.placement == "output"
        assert script.enabled is True
        assert script.order_num == 100


class TestApplyRegexScripts:
    """apply_regex_scripts function behavior."""

    def test_simple_replacement(self):
        script = RegexScript(
            find_regex=r"hello",
            replace_string="hi",
            placement="output",
            enabled=True,
        )
        result = apply_regex_scripts("hello world", [script], "output")
        assert result == "hi world"

    def test_placement_filter(self):
        script = RegexScript(
            find_regex=r"hello",
            replace_string="hi",
            placement="input",
            enabled=True,
        )
        # Script is for "input" placement, but we're applying "output"
        result = apply_regex_scripts("hello world", [script], "output")
        assert result == "hello world"  # unchanged

    def test_both_placement_matches(self):
        script = RegexScript(
            find_regex=r"hello",
            replace_string="hi",
            placement="both",
            enabled=True,
        )
        result = apply_regex_scripts("hello world", [script], "output")
        assert result == "hi world"

    def test_disabled_script_skipped(self):
        script = RegexScript(
            find_regex=r"hello",
            replace_string="hi",
            placement="output",
            enabled=False,
        )
        result = apply_regex_scripts("hello world", [script], "output")
        assert result == "hello world"

    def test_invalid_regex_skipped(self):
        script = RegexScript(
            find_regex=r"[invalid",
            replace_string="fix",
            placement="output",
            enabled=True,
        )
        # Should not crash — invalid regex is caught and skipped
        result = apply_regex_scripts("test text", [script], "output")
        assert result == "test text"

    def test_chain_multiple_scripts(self):
        scripts = [
            RegexScript(find_regex=r"foo", replace_string="bar", placement="output", enabled=True, order_num=1),
            RegexScript(find_regex=r"bar", replace_string="baz", placement="output", enabled=True, order_num=2),
        ]
        result = apply_regex_scripts("foo", scripts, "output")
        assert result == "baz"

    def test_empty_text_returns_empty(self):
        script = RegexScript(find_regex=r"x", replace_string="y", placement="output", enabled=True)
        result = apply_regex_scripts("", [script], "output")
        assert result == ""

    def test_empty_scripts_returns_text(self):
        result = apply_regex_scripts("hello", [], "output")
        assert result == "hello"


class TestNarrativeExtraction:
    """NarrativeExtraction dataclass."""

    def test_default_empty(self):
        extraction = NarrativeExtraction()
        assert extraction.characters == []
        assert extraction.world is None
