"""Tests for augmentum.voice.emotion — RP emotion extraction for TTS."""

from __future__ import annotations

from augmentum.voice.emotion import (
    extract_emotion_instruct,
    inject_turbo_tags,
    is_turbo_provider,
)


class TestRPMarkerExtraction:
    """RP asterisk/hyphen markers should produce instruct strings."""

    def test_sigh(self):
        assert extract_emotion_instruct('*sighs deeply* "I know."') == "speak with a weary sigh"

    def test_whisper(self):
        assert extract_emotion_instruct('*whispers* "Come here."') == "speak in a soft whisper"

    def test_shout(self):
        assert extract_emotion_instruct('*shouts angrily* "Get out!"') == "speak loudly and forcefully"

    def test_sob(self):
        assert extract_emotion_instruct('*sobbing* "Why did you leave?"') == "speak through tears"

    def test_laugh(self):
        assert extract_emotion_instruct('*laughs* "That was funny."') == "speak with amusement"

    def test_tremble(self):
        assert extract_emotion_instruct('*trembling* "Please, no..."') == "speak with a trembling, fearful voice"

    def test_growl(self):
        assert extract_emotion_instruct('*growls* "Stay back."') == "speak with a low, threatening tone"

    def test_hyphenated_marker(self):
        assert extract_emotion_instruct('-whispers- "Over here."') == "speak in a soft whisper"

    def test_plead(self):
        assert extract_emotion_instruct('*pleads desperately*') == "speak with desperate pleading"


class TestRPPriorityOverEntity:
    """RP markers should take priority over entity emotional state."""

    def test_whisper_over_angry(self):
        result = extract_emotion_instruct(
            '*whispers* "Don\'t move."',
            entity_emotional_state="angry",
        )
        assert result == "speak in a soft whisper"

    def test_laugh_over_sad(self):
        result = extract_emotion_instruct(
            '*chuckles* "Well, well."',
            entity_emotional_state="sad",
        )
        assert result == "speak with amusement"


class TestEntityStateFallback:
    """Entity emotional state used as fallback when no RP markers present."""

    def test_sad(self):
        assert extract_emotion_instruct("I miss you.", "sad") == "speak with sadness"

    def test_happy(self):
        assert extract_emotion_instruct("Great to see you!", "happy") == "speak happily and warmly"

    def test_angry(self):
        assert extract_emotion_instruct("This is unacceptable.", "angry") == "speak with anger"

    def test_excited(self):
        assert extract_emotion_instruct("Look at this!", "excited") == "speak with excitement and energy"

    def test_compound_state(self):
        """Substring match: 'slightly sad' should match 'sad'."""
        assert extract_emotion_instruct("I see.", "slightly sad") == "speak with sadness"

    def test_compound_anxious(self):
        assert extract_emotion_instruct("What if...", "very anxious") == "speak with nervous anxiety"


class TestNoSignal:
    """No emotion signal should return empty string."""

    def test_plain_text_no_entity(self):
        assert extract_emotion_instruct("Hello, how are you?") == ""

    def test_plain_text_empty_entity(self):
        assert extract_emotion_instruct("Hello.", "") == ""

    def test_neutral_entity(self):
        assert extract_emotion_instruct("Hello.", "neutral") == ""

    def test_normal_entity(self):
        assert extract_emotion_instruct("Hello.", "normal") == ""

    def test_calm_entity(self):
        """Calm is in _NEUTRAL_STATES — no instruct needed."""
        assert extract_emotion_instruct("Hello.", "calm") == ""

    def test_no_matching_marker(self):
        """Asterisked text that doesn't match any keyword."""
        assert extract_emotion_instruct('*nods slowly* "Yes."') == ""


class TestTurboTags:
    """Chatterbox Turbo paralinguistic tag injection."""

    def test_laugh_tag(self):
        assert inject_turbo_tags('*laughs softly* Hello') == "[laugh] Hello"

    def test_cough_tag(self):
        assert inject_turbo_tags('*coughs* excuse me') == "[cough] excuse me"

    def test_chuckle_tag(self):
        assert inject_turbo_tags('*chuckles* oh really') == "[laugh] oh really"

    def test_no_matching_marker(self):
        """Unrecognized RP markers are stripped."""
        assert inject_turbo_tags('*sighs deeply* I am tired') == "I am tired"

    def test_no_markers(self):
        assert inject_turbo_tags('Just normal text') == "Just normal text"

    def test_multiple_markers(self):
        result = inject_turbo_tags('*laughs* Hello *coughs* there')
        assert "[laugh]" in result
        assert "[cough]" in result

    def test_is_turbo_provider(self):
        assert is_turbo_provider("chatterbox-turbo")
        assert not is_turbo_provider("chatterbox-tts")
        assert not is_turbo_provider("kokoro-tts")


class TestCleanForTTSPreserveBrackets:
    """Verify preserve_brackets keeps Turbo tags through cleaning."""

    def test_brackets_preserved(self):
        from augmentum.voice.text_cleaning import clean_for_tts
        result = clean_for_tts("[laugh] Hello there", preserve_brackets=True)
        assert "[laugh]" in result

    def test_brackets_stripped_by_default(self):
        from augmentum.voice.text_cleaning import clean_for_tts
        result = clean_for_tts("[laugh] Hello there", preserve_brackets=False)
        assert "[" not in result
