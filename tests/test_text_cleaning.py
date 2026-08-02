"""Tests for voice/text_cleaning.py -- TTS text normalization."""

from __future__ import annotations

import pytest

from augmentum.voice.text_cleaning import clean_for_tts


class TestUrlRemoval:
    def test_strips_urls(self):
        result = clean_for_tts("Visit https://example.com for details.")
        assert "https://" not in result
        assert "example.com" not in result

    def test_strips_http_urls(self):
        result = clean_for_tts("Go to http://test.org now.")
        assert "http://" not in result


class TestEmojiHandling:
    def test_strips_emoji(self):
        result = clean_for_tts("Hello world! 😀🎉")
        assert "😀" not in result
        assert "🎉" not in result
        assert "Hello world" in result

    def test_strips_various_emoji_ranges(self):
        result = clean_for_tts("Stars 🌟 and fire 🔥 and smiley 😊")
        assert "🌟" not in result
        assert "🔥" not in result
        assert "😊" not in result


class TestAbbreviationExpansion:
    def test_dr_expanded(self):
        result = clean_for_tts("Dr. Smith is here.")
        assert "Doctor" in result

    def test_mr_expanded(self):
        result = clean_for_tts("Mr. Jones arrived.")
        assert "Mister" in result

    def test_etc_expanded(self):
        result = clean_for_tts("Apples, oranges, etc.")
        assert "etcetera" in result

    def test_vs_expanded(self):
        result = clean_for_tts("Good vs. evil.")
        assert "versus" in result


class TestEmptyInput:
    def test_empty_string(self):
        result = clean_for_tts("")
        assert result == ""

    def test_whitespace_only(self):
        result = clean_for_tts("   ")
        assert result == ""


class TestMarkdownStripping:
    def test_strips_bold(self):
        result = clean_for_tts("This is **bold** text.")
        assert "**" not in result
        assert "bold" in result

    def test_strips_headers(self):
        result = clean_for_tts("## Chapter One\nThe story begins.")
        assert "##" not in result
        assert "Chapter One" in result

    def test_strips_code_blocks(self):
        result = clean_for_tts("Here:\n```python\nprint('hi')\n```\nDone.")
        assert "print" not in result
        assert "```" not in result

    def test_links_keep_text(self):
        result = clean_for_tts("Click [here](https://example.com) for more.")
        assert "here" in result
        assert "https://" not in result


class TestNumberExpansion:
    def test_small_number(self):
        result = clean_for_tts("There are 42 cats.")
        assert "forty-two" in result

    def test_percentage(self):
        result = clean_for_tts("That's 90% done.")
        assert "percent" in result

    def test_currency(self):
        result = clean_for_tts("It costs $5 to enter.")
        assert "dollar" in result


class TestRpFormatting:
    def test_keeps_short_action(self):
        result = clean_for_tts("*sighs* Well, fine then.")
        assert "sighs" in result
        assert "Well" in result

    def test_keeps_pronoun_action(self):
        result = clean_for_tts("*She smiles softly* I missed you.")
        assert "She smiles softly" in result
        assert "I missed you" in result

    def test_keeps_short_italic_emphasis(self):
        result = clean_for_tts("This is *really important* today.")
        assert "really important" in result

    def test_keeps_long_italic(self):
        result = clean_for_tts("*The moonlight cast long shadows across the courtyard floor.*")
        assert "moonlight" in result

    def test_strips_ooc_markers(self):
        result = clean_for_tts("Hello! ((this is ooc))")
        assert "ooc" not in result
        assert "Hello" in result
