"""Tests for TTS text cleaning — verify citations, URLs, and source markers
are stripped before audio synthesis.

Uses realistic model outputs from various LLM providers to test every
citation pattern we've seen in the wild.
"""

from augmentum.voice.text_cleaning import clean_for_tts

# ---------------------------------------------------------------------------
# Realistic LLM outputs with citations/sources
# ---------------------------------------------------------------------------

# DeepSeek-style: inline [n] citations + Sources section
DEEPSEEK_RESPONSE = """The speed of light in a vacuum is approximately 299,792,458 meters per second [1]. This constant, denoted as "c", is fundamental to Einstein's theory of special relativity [2,3].

Light travels at different speeds through different media. In water, it slows to about 225,000 km/s [4]. In diamond, it's even slower at roughly 124,000 km/s.

Sources:
1. NIST Physical Constants - https://physics.nist.gov/constants
2. Einstein, A. (1905). "On the Electrodynamics of Moving Bodies"
3. Wikipedia - Speed of Light: https://en.wikipedia.org/wiki/Speed_of_light
4. Hecht, E. "Optics" (5th ed.)"""

# GPT-style: markdown links + "(Source: ...)" attributions
GPT_RESPONSE = """According to [NASA](https://www.nasa.gov/exploration), the Artemis program aims to return humans to the Moon by 2025 (Source: NASA Official Website).

The program uses the Space Launch System (SLS), which is the most powerful rocket ever built. As reported by The New York Times, the first uncrewed test flight was completed successfully in 2022.

Further reading:
- NASA Artemis Overview: https://www.nasa.gov/artemis
- SpaceNews coverage: https://spacenews.com/tag/artemis/"""

# Qwen/Chinese model style: CJK brackets
QWEN_RESPONSE = """量子计算是一种利用量子力学原理进行计算的技术【1】。与经典计算机不同，量子计算机使用量子位〖qubit〗来处理信息。

Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement【2】. Companies like IBM, Google, and Microsoft are investing heavily in this technology【3,4】.

The potential applications include cryptography, drug discovery, and optimization problems¹²³."""

# Gemma/small model style: messy inline attributions
GEMMA_RESPONSE = """The Great Wall of China is one of the most impressive architectural feats in history. It stretches over 13,000 miles (Source: UNESCO World Heritage) and was built over many centuries.

According to historical records (via Britannica.com), construction began as early as the 7th century BC. The most well-known sections were built during the Ming Dynasty (1368-1644).

(1) UNESCO World Heritage Centre - whc.unesco.org
(2) Encyclopedia Britannica - britannica.com/topic/Great-Wall-of-China"""

# Search-augmented response with footnotes
SEARCH_RESPONSE = """Python 3.12 introduced several performance improvements[^1]. The most notable is a 5% speed increase in the default interpreter[^2].

Key features include:
- Improved error messages with suggestions[^3]
- Per-interpreter GIL (experimental)[^4]
- Type parameter syntax (PEP 695)

[^1]: Python 3.12 Release Notes, python.org/downloads/release/python-3120/
[^2]: https://docs.python.org/3/whatsnew/3.12.html
[^3]: CPython Issue Tracker
[^4]: PEP 684 – A Per-Interpreter GIL"""

# Mixed format with code blocks and stats
MIXED_RESPONSE = """Here's a comparison of web frameworks:

```
Framework    Req/sec    Latency
FastAPI      12,450     3.2ms
Express      8,900      5.1ms
Django       4,200      8.7ms
```

FastAPI is the fastest option [citation needed], achieving 12,450 requests per second in benchmarks (source: TechEmpower Framework Benchmarks, www.techempower.com/benchmarks).

See also: https://fastapi.tiangolo.com/benchmarks/"""

# Model that uses superscripts and parenthesized numbers
ACADEMIC_RESPONSE = """Climate change has accelerated significantly since the industrial revolution¹. Global temperatures have risen by approximately 1.1°C above pre-industrial levels².

The Intergovernmental Panel on Climate Change (IPCC) reports³ that limiting warming to 1.5°C requires reducing CO₂ emissions by 45% from 2010 levels by 2030⁴⁵.

References:
1. IPCC AR6 Synthesis Report (2023)
2. NASA Global Climate Change - climate.nasa.gov
3. IPCC Special Report on 1.5°C
4. United Nations Environment Programme
5. World Meteorological Organization"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCitationMarkers:
    def test_bracket_numbers(self):
        result = clean_for_tts("[1] This is cited. Also see [2,3] and [1-5].")
        assert "[" not in result
        assert "]" not in result
        assert "cited" in result

    def test_footnote_markers(self):
        result = clean_for_tts("Python improved[^1] significantly[^note].")
        assert "[^" not in result
        assert "improved" in result

    def test_parenthesized_numbers(self):
        result = clean_for_tts("This was proven (1) and confirmed (2,3).")
        assert "(1)" not in result
        assert "(2,3)" not in result
        assert "proven" in result

    def test_superscript_numbers(self):
        result = clean_for_tts("Temperatures rose¹ significantly²³.")
        assert "¹" not in result
        assert "²" not in result
        assert "rose" in result

    def test_cjk_brackets(self):
        result = clean_for_tts("This is important【1】and also【source】.")
        assert "【" not in result
        assert "】" not in result
        assert "important" in result

    def test_citation_needed(self):
        result = clean_for_tts("This is fast [citation needed] and reliable.")
        assert "citation" not in result.lower()
        assert "fast" in result


class TestURLsAndDomains:
    def test_full_urls(self):
        result = clean_for_tts("Visit https://example.com/page for details.")
        assert "https" not in result
        assert "example.com" not in result
        assert "Visit" in result

    def test_www_urls(self):
        result = clean_for_tts("Check www.example.com for more info.")
        assert "www" not in result
        assert "Check" in result

    def test_bare_domains(self):
        result = clean_for_tts("See wikipedia.org for the full article.")
        assert "wikipedia.org" not in result
        assert "See" in result

    def test_domain_with_path(self):
        result = clean_for_tts("Details at python.org/downloads/release.")
        assert "python.org" not in result


class TestSourceSections:
    def test_sources_header(self):
        result = clean_for_tts("Good info.\n\nSources:\n1. First source\n2. Second source")
        assert "Sources" not in result
        assert "Good info" in result

    def test_references_header(self):
        result = clean_for_tts("Main text.\n\nReferences:\n- Smith (2023)\n- Jones (2024)")
        assert "References" not in result

    def test_further_reading(self):
        result = clean_for_tts("Content here.\n\nFurther reading:\n- Some book")
        assert "Further reading" not in result

    def test_see_also(self):
        result = clean_for_tts("Main point.\n\nSee also:\n- Related topic")
        assert "See also" not in result


class TestInlineAttributions:
    def test_source_parenthetical(self):
        result = clean_for_tts("The wall is 13,000 miles long (Source: UNESCO).")
        assert "Source:" not in result
        assert "UNESCO" not in result

    def test_via_parenthetical(self):
        result = clean_for_tts("Revenue grew 20% (via Bloomberg).")
        assert "via" not in result
        assert "Bloomberg" not in result

    def test_according_to_with_link(self):
        result = clean_for_tts("According to [NASA](https://nasa.gov), the mission succeeded.")
        assert "nasa.gov" not in result
        assert "mission succeeded" in result


class TestMarkdownLinks:
    def test_link_keeps_text(self):
        result = clean_for_tts("Read the [documentation](https://docs.example.com) for details.")
        assert "documentation" in result
        assert "https" not in result
        assert "example.com" not in result


class TestRealisticResponses:
    """Full end-to-end tests with real model outputs."""

    def _assert_clean(self, text):
        """Assert no citation artifacts remain in cleaned text."""
        result = clean_for_tts(text)
        # No brackets
        assert "[" not in result, f"Bracket found in: {result[:200]}"
        assert "【" not in result, f"CJK bracket found in: {result[:200]}"
        assert "〖" not in result, f"CJK bracket found in: {result[:200]}"
        # No URLs
        assert "http" not in result.lower(), f"URL found in: {result[:200]}"
        assert "www." not in result.lower(), f"www URL found in: {result[:200]}"
        # No .com/.org domains
        for tld in [".com", ".org", ".net", ".gov", ".edu", ".wiki"]:
            assert tld not in result.lower(), f"Domain {tld} found in: {result[:200]}"
        # No superscripts
        for ch in "¹²³⁴⁵⁶⁷⁸⁹⁰":
            assert ch not in result, f"Superscript found in: {result[:200]}"
        # No source/reference headers
        for word in ["Sources:", "References:", "Further reading:", "See also:"]:
            assert word not in result, f"Header found in: {result[:200]}"
        # Content survived (not everything was stripped)
        assert len(result.strip()) > 20, f"Too much content stripped: '{result}'"
        return result

    def test_deepseek_response(self):
        result = self._assert_clean(DEEPSEEK_RESPONSE)
        assert "speed of light" in result.lower()

    def test_gpt_response(self):
        result = self._assert_clean(GPT_RESPONSE)
        assert "artemis" in result.lower()

    def test_qwen_response(self):
        result = self._assert_clean(QWEN_RESPONSE)
        assert "quantum" in result.lower()

    def test_gemma_response(self):
        result = self._assert_clean(GEMMA_RESPONSE)
        assert "great wall" in result.lower()

    def test_search_response(self):
        result = self._assert_clean(SEARCH_RESPONSE)
        assert "python" in result.lower()

    def test_mixed_response(self):
        result = self._assert_clean(MIXED_RESPONSE)
        assert "fastest" in result.lower()

    def test_academic_response(self):
        result = self._assert_clean(ACADEMIC_RESPONSE)
        assert "climate" in result.lower()


class TestUnitExpansionContext:
    """Units expand ONLY next to a number (2026-06-11): standalone
    'in.' and 'mm' collided with plain English — "Come on in." became
    "Come on inches", and the filler "mm" became "millimeters"."""

    def test_preposition_in_survives(self):
        assert "inches" not in clean_for_tts("Come on in.")
        assert "inches" not in clean_for_tts("She walked in. Then sat.")

    def test_filler_mm_survives(self):
        out = clean_for_tts("Mm. That's a good question.")
        assert "millimeter" not in out
        out = clean_for_tts("mm-hmm, I hear you.")
        assert "millimeter" not in out

    def test_twelve_in_the_morning_survives(self):
        out = clean_for_tts("Let's meet at 12 in the morning.")
        assert "inches" not in out

    def test_number_attached_units_expand(self):
        assert "millimeters" in clean_for_tts("The screw is 5mm long.")
        assert "millimeters" in clean_for_tts("It measures 5 mm across.")
        assert "miles per hour" in clean_for_tts("Going 60mph now.")
        assert "kilograms" in clean_for_tts("It weighs 3 kg total.")

    def test_glued_inches_expands(self):
        assert "inches" in clean_for_tts("The board is 5in wide.")

    def test_lowercase_stat_words_survive(self):
        # Case-sensitive gaming stats: prose words must not expand.
        out = clean_for_tts("convert the value to int and check str length")
        assert "intelligence" not in out
        assert "strength" not in out
        out = clean_for_tts("the con artist was charming")
        assert "constitution" not in out

    def test_caps_stats_still_expand(self):
        out = clean_for_tts("Your INT went up and HP dropped.")
        assert "intelligence" in out
        assert "hit points" in out


class TestTtsLexicon:
    """voice_tts_lexicon — operator pronunciation overrides."""

    def _with_lexicon(self, monkeypatch, lexicon_json):
        from augmentum.config import settings
        monkeypatch.setattr(
            settings, "voice_tts_lexicon", lexicon_json, raising=False,
        )

    def test_replacement(self, monkeypatch):
        self._with_lexicon(monkeypatch, '{"SQL": "sequel"}')
        out = clean_for_tts("Run the SQL query again.")
        assert "sequel" in out
        assert "SQL" not in out

    def test_word_boundary(self, monkeypatch):
        self._with_lexicon(monkeypatch, '{"SQL": "sequel"}')
        out = clean_for_tts("The SQLite file is fine.")
        assert "sequelite" not in out.lower()

    def test_empty_value_shields_term(self, monkeypatch):
        # {"vs": ""} protects "vs" from the built-in "versus" expansion.
        self._with_lexicon(monkeypatch, '{"vs": ""}')
        out = clean_for_tts("It was cats vs dogs.")
        assert "versus" not in out
        assert "vs" in out

    def test_invalid_json_is_harmless(self, monkeypatch):
        self._with_lexicon(monkeypatch, '{not json')
        out = clean_for_tts("Hello there, friend.")
        assert "Hello" in out

    def test_empty_lexicon_default(self, monkeypatch):
        self._with_lexicon(monkeypatch, "")
        out = clean_for_tts("It was cats vs dogs.")
        assert "versus" in out  # built-in still applies
