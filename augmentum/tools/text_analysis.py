"""Text analysis tool — word count, readability, summarization stats."""

from __future__ import annotations

import re
from collections import Counter

from augmentum.tools.base import Tool, ToolCategory, ToolResult


def _count_syllables(word: str) -> int:
    """Estimate syllable count using a simple heuristic."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1

    # Remove trailing 'e'
    if word.endswith("e"):
        word = word[:-1]
    if not word:
        return 1

    count = 0
    prev_vowel = False
    vowels = set("aeiou")
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    return max(1, count)


def analyze_text(text: str) -> dict:
    """Compute comprehensive text statistics."""
    if not text.strip():
        return {"error": "Empty text"}

    # Basic counts
    characters = len(text)
    characters_no_spaces = len(text.replace(" ", ""))
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(1, len(sentences))
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraph_count = max(1, len(paragraphs))
    lines = text.split("\n")
    line_count = len(lines)

    # Word statistics
    word_lengths = [len(w) for w in words]
    avg_word_length = sum(word_lengths) / max(1, word_count)
    avg_sentence_length = word_count / sentence_count

    # Syllable counts
    syllable_counts = [_count_syllables(w) for w in words]
    total_syllables = sum(syllable_counts)
    avg_syllables_per_word = total_syllables / max(1, word_count)

    # Complex words (3+ syllables)
    complex_words = sum(1 for s in syllable_counts if s >= 3)

    # Readability scores
    # Flesch Reading Ease
    flesch = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word

    # Flesch-Kincaid Grade Level
    fk_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59

    # Gunning Fog Index
    fog = 0.4 * (avg_sentence_length + 100 * complex_words / max(1, word_count))

    # Coleman-Liau Index
    avg_chars_per_100 = (characters_no_spaces / max(1, word_count)) * 100
    avg_sentences_per_100 = (sentence_count / max(1, word_count)) * 100
    coleman_liau = 0.0588 * avg_chars_per_100 - 0.296 * avg_sentences_per_100 - 15.8

    # Frequency analysis
    word_freq = Counter(w.lower() for w in words)
    unique_words = len(word_freq)
    lexical_diversity = unique_words / max(1, word_count)

    # Reading time (avg 238 words per minute)
    reading_time_minutes = word_count / 238
    speaking_time_minutes = word_count / 150  # avg speaking rate

    return {
        "characters": characters,
        "characters_no_spaces": characters_no_spaces,
        "words": word_count,
        "unique_words": unique_words,
        "sentences": sentence_count,
        "paragraphs": paragraph_count,
        "lines": line_count,
        "avg_word_length": round(avg_word_length, 2),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_syllables_per_word": round(avg_syllables_per_word, 2),
        "complex_words": complex_words,
        "lexical_diversity": round(lexical_diversity, 4),
        "readability": {
            "flesch_reading_ease": round(flesch, 1),
            "flesch_kincaid_grade": round(fk_grade, 1),
            "gunning_fog_index": round(fog, 1),
            "coleman_liau_index": round(coleman_liau, 1),
        },
        "reading_time_minutes": round(reading_time_minutes, 1),
        "speaking_time_minutes": round(speaking_time_minutes, 1),
        "top_words": dict(word_freq.most_common(10)),
    }


class TextAnalysisTool(Tool):
    """Analyze text for statistics, readability, and word frequency."""

    @property
    def name(self) -> str:
        return "text_analysis"

    @property
    def description(self) -> str:
        return (
            "Analyze text for word count, character count, sentence count, "
            "readability scores (Flesch, Gunning Fog, Coleman-Liau), "
            "reading time, lexical diversity, and word frequency."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to analyze"},
            },
            "required": ["text"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        if not text:
            return ToolResult(success=False, error="No text provided")

        import json

        try:
            stats = analyze_text(text)
            return ToolResult(
                success=True,
                output=json.dumps(stats, indent=2),
                metadata=stats,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Analysis error: {e}")
