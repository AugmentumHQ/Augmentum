"""Corpus-driven tests for the comic archive filename parser.

The parser is the entry point for the ~80% of real-world archives that
don't carry ComicInfo.xml. These tests exercise every pattern that surfaced
in OSS-reader issue threads (Mihon, Komga, Kavita) — missing any of them
produces low-confidence parses that degrade the library surface.

Test structure:
  - TestBracketClassification: unit tests of _classify_bracketed
  - TestStandardPatterns: digital-release convention (v01, year, group)
  - TestVolumeMarkers: every volume-marker variant
  - TestChapterMarkers: every chapter-marker variant, including decimals
  - TestSeparators: dot / underscore / space word boundaries
  - TestSpecialIssues: Extra / Bonus / Omake / side stories
  - TestOmnibus: volume-range parsing (v1-3)
  - TestConfidence: confidence floor/ceiling behavior
  - TestEdgeCases: empty, garbage, nested, numeric-only
"""

from __future__ import annotations

import pytest

from augmentum.media.comic_filename_parser import (
    ParsedFilename,
    _classify_bracketed,
    parse_filename,
)


# --- Bracket classification ----------------------------------------------


class TestBracketClassification:
    def test_year(self):
        cat, val = _classify_bracketed("2003")
        assert cat == "year"
        assert val == "2003"

    def test_year_boundary_low(self):
        cat, _ = _classify_bracketed("1900")
        assert cat == "year"

    def test_year_boundary_high(self):
        cat, _ = _classify_bracketed("2099")
        assert cat == "year"

    def test_year_rejects_out_of_range(self):
        # 1899 doesn't match our year regex at all
        cat, _ = _classify_bracketed("1899")
        assert cat == "group"  # falls through to "unrecognized" → group

    def test_source(self):
        cat, val = _classify_bracketed("Digital")
        assert cat == "source"
        assert val == "digital"

    def test_source_case_insensitive(self):
        cat, _ = _classify_bracketed("DIGITAL")
        assert cat == "source"

    def test_quality(self):
        cat, val = _classify_bracketed("HD")
        assert cat == "quality"
        assert val == "hd"

    def test_source_and_quality_combined(self):
        cat, val = _classify_bracketed("Digital-HD")
        assert cat == "source_quality"
        assert val == "digital|hd"

    def test_language_code(self):
        cat, val = _classify_bracketed("JP")
        assert cat == "language"
        assert val == "jp"

    def test_language_name(self):
        cat, val = _classify_bracketed("Japanese")
        assert cat == "language"
        assert val == "japanese"

    def test_special_bonus(self):
        cat, _ = _classify_bracketed("Bonus")
        assert cat == "special"

    def test_special_extra(self):
        cat, _ = _classify_bracketed("Extra")
        assert cat == "special"

    def test_special_omake(self):
        cat, _ = _classify_bracketed("Omake")
        assert cat == "special"

    def test_group_fallback(self):
        cat, val = _classify_bracketed("LuCaZ")
        assert cat == "group"
        assert val == "LuCaZ"

    def test_group_with_dash(self):
        cat, val = _classify_bracketed("danke-Empire")
        assert cat == "group"
        assert val == "danke-Empire"

    def test_empty_string(self):
        cat, val = _classify_bracketed("")
        assert cat == "unknown"
        assert val is None


# --- Standard digital-release patterns -----------------------------------


class TestStandardPatterns:
    def test_berserk_full(self):
        p = parse_filename("Berserk v01 (2003) (Digital) (LuCaZ).cbz")
        assert p.series == "Berserk"
        assert p.volume == 1
        assert p.year == 2003
        assert p.source == "digital"
        assert p.scan_group == "LuCaZ"
        assert p.confidence >= 0.85

    def test_group_leading_bracket(self):
        p = parse_filename("[LuCaZ] Berserk v01 (2003).cbz")
        assert p.series == "Berserk"
        assert p.volume == 1
        assert p.year == 2003
        assert p.scan_group == "LuCaZ"

    def test_group_only_no_year(self):
        p = parse_filename("[danke] Attack on Titan v34.cbz")
        assert p.series == "Attack on Titan"
        assert p.volume == 34
        assert p.scan_group == "danke"

    def test_group_with_hyphen(self):
        p = parse_filename("Vagabond v37 (2015) (Digital-HD) (danke-Empire).cbz")
        assert p.series == "Vagabond"
        assert p.volume == 37
        assert p.year == 2015
        assert p.source == "digital"
        assert p.quality == "hd"
        assert p.scan_group == "danke-Empire"


# --- Volume marker variants ----------------------------------------------


class TestVolumeMarkers:
    @pytest.mark.parametrize("filename, expected", [
        ("Series v01.cbz", 1),
        ("Series v1.cbz", 1),
        ("Series V01.cbz", 1),
        ("Series v100.cbz", 100),
        ("Series Vol. 01.cbz", 1),
        ("Series Vol 1.cbz", 1),
        ("Series vol.1.cbz", 1),
        ("Series Volume 1.cbz", 1),
        ("Series Volume 103.cbz", 103),
    ])
    def test_volume_extraction(self, filename: str, expected: int):
        p = parse_filename(filename)
        assert p.volume == expected, f"{filename!r} → volume={p.volume}"

    def test_japanese_volume_marker(self):
        p = parse_filename("\u547c\u3073\u304b\u3051 \u7b2c01\u5dfb.cbz")
        assert p.volume == 1

    def test_volume_beats_chapter_when_both_present(self):
        # v01 Ch. 005 → Vol 1, Chapter 5 — both extracted
        p = parse_filename("One Piece v01 Ch. 005.cbz")
        assert p.volume == 1
        assert p.chapter == 5.0


# --- Chapter marker variants ---------------------------------------------


class TestChapterMarkers:
    @pytest.mark.parametrize("filename, expected", [
        ("Series c001.cbz", 1.0),
        ("Series c1050.cbz", 1050.0),
        ("Series Ch. 1.cbz", 1.0),
        ("Series Ch 1.cbz", 1.0),
        ("Series ch.01.cbz", 1.0),
        ("Series Chapter 1.cbz", 1.0),
        ("Series Chapter 97.cbz", 97.0),
        ("Saga #1.cbz", 1.0),
        ("Saga #1050.cbz", 1050.0),
    ])
    def test_chapter_extraction(self, filename: str, expected: float):
        p = parse_filename(filename)
        assert p.chapter == expected, f"{filename!r} → chapter={p.chapter}"

    def test_decimal_chapter(self):
        p = parse_filename("Berserk Ch. 12.5.cbz")
        assert p.chapter == 12.5

    def test_decimal_chapter_with_bonus(self):
        p = parse_filename("Berserk Ch. 12.5 (Bonus).cbz")
        assert p.chapter == 12.5
        assert p.is_special is True
        # "Bonus" should NOT be classified as scan_group
        assert p.scan_group is None

    def test_japanese_chapter_marker(self):
        # 呪術廻戦 第01話.cbz
        p = parse_filename("\u547c\u3073\u304b\u3051 \u7b2c01\u8a71.cbz")
        assert p.chapter == 1.0


# --- Separator handling ---------------------------------------------------


class TestSeparators:
    def test_underscore_separators(self):
        p = parse_filename("Series_Name_Vol_1.cbz")
        assert p.series == "Series Name"
        assert p.volume == 1

    def test_underscore_with_year(self):
        p = parse_filename("Berserk_v01_(2003).cbz")
        assert p.series == "Berserk"
        assert p.volume == 1
        assert p.year == 2003

    def test_dot_separators(self):
        # Dot-separated titles: we strip dots in cleanup
        p = parse_filename("Series.Name.Vol.1.cbz")
        assert p.volume == 1

    def test_collapses_whitespace(self):
        p = parse_filename("Berserk    v01    (2003).cbz")
        assert p.series == "Berserk"
        assert p.volume == 1

    def test_hyphen_separator_preserved_in_title(self):
        p = parse_filename("Attack on Titan - Shingeki no Kyojin v34.cbz")
        # Leading hyphens stripped, but internal hyphens preserved
        assert "Attack on Titan" in p.series
        assert p.volume == 34


# --- Special issues -------------------------------------------------------


class TestSpecialIssues:
    def test_extra_volume(self):
        p = parse_filename("Berserk Vol Extra.cbz")
        # "Extra" should flip is_special even without a number
        assert p.is_special is True

    def test_bonus_bracket(self):
        p = parse_filename("Berserk (Bonus).cbz")
        assert p.is_special is True
        assert p.scan_group is None  # Bonus != group

    def test_omake(self):
        p = parse_filename("Series Omake.cbz")
        assert p.is_special is True

    def test_side_story(self):
        p = parse_filename("Series Side Story.cbz")
        assert p.is_special is True


# --- Omnibus / volume ranges ----------------------------------------------


class TestOmnibus:
    def test_volume_range(self):
        p = parse_filename("Berserk Deluxe v1-3 (2019).cbz")
        assert p.series == "Berserk Deluxe"
        assert p.volume == 1
        assert p.volume_end == 3
        assert p.year == 2019

    def test_vol_range_with_words(self):
        p = parse_filename("Series Vol. 1-3.cbz")
        assert p.volume == 1
        assert p.volume_end == 3

    def test_single_volume_no_end(self):
        p = parse_filename("Berserk v01.cbz")
        assert p.volume == 1
        assert p.volume_end is None


# --- Confidence scoring --------------------------------------------------


class TestConfidence:
    def test_bare_title_minimum(self):
        p = parse_filename("Just A Title.cbz")
        assert p.confidence == 0.30

    def test_title_plus_volume(self):
        p = parse_filename("Series v01.cbz")
        assert p.confidence == pytest.approx(0.55, abs=0.01)

    def test_title_plus_volume_plus_year(self):
        p = parse_filename("Series v01 (2020).cbz")
        assert p.confidence == pytest.approx(0.75, abs=0.01)

    def test_full_metadata_caps_at_one(self):
        p = parse_filename(
            "[Group] Series v01 Ch. 005 (2020) (Digital).cbz"
        )
        # 0.3 + 0.25 + 0.15 + 0.2 + 0.05 + 0.05 = 1.0
        assert p.confidence == 1.0

    def test_empty_filename_low_confidence(self):
        # Pathological case — a file literally named ".cbz" has no stem
        # that pathlib recognizes as a name. Parser stays stable but the
        # result is meaningless; callers should treat <0.35 as review-queue.
        p = parse_filename("")
        assert p.series == ""
        assert p.confidence == 0.0


# --- Edge cases ----------------------------------------------------------


class TestEdgeCases:
    def test_no_extension(self):
        # Parser should still work without an extension
        p = parse_filename("Berserk v01")
        assert p.series == "Berserk"
        assert p.volume == 1

    def test_empty_string(self):
        p = parse_filename("")
        assert p.series == ""
        assert p.confidence == 0.0

    def test_whitespace_only(self):
        p = parse_filename("   .cbz")
        assert p.series == ""

    def test_year_outside_parens_ignored(self):
        # A bare year in the title shouldn't be captured as year metadata
        p = parse_filename("Series 2003.cbz")
        assert p.year is None  # requires brackets/parens

    def test_numeric_only_title(self):
        p = parse_filename("12345.cbz")
        # Not meaningful, low confidence — title is the number
        assert p.confidence <= 0.3

    def test_preserves_apostrophes(self):
        p = parse_filename("JoJo's Bizarre Adventure Part 4 v12 (2020).cbz")
        assert "JoJo's" in p.series
        assert p.volume == 12
        assert p.year == 2020

    def test_preserves_colons(self):
        p = parse_filename("Series: Subtitle v01.cbz")
        assert ":" in p.series or "Subtitle" in p.series
        assert p.volume == 1

    def test_multiple_years_takes_first(self):
        # Re-releases sometimes have two years — we take the first
        p = parse_filename("Series v01 (2003) (2019).cbz")
        assert p.year == 2003

    def test_ignores_subsequent_groups(self):
        # First bracket is scan group; later brackets get ignored
        p = parse_filename("[First] Series v01 [Second] [Third].cbz")
        assert p.scan_group == "First"

    def test_raw_stem_preserved(self):
        p = parse_filename("Berserk v01 (2003).cbz")
        assert p.raw_stem == "Berserk v01 (2003)"

    def test_only_strips_known_extensions(self):
        # If the "extension" is part of the title (unlikely), don't strip
        p = parse_filename("Series v1.5")
        # .5 is NOT a known archive extension, so we keep it
        assert p.raw_stem == "Series v1.5"


# --- Integration: full corpus parameterized ------------------------------


_CORPUS = [
    # (filename, expected_series, expected_volume, expected_chapter, expected_year)
    ("Berserk v01 (2003) (Digital) (LuCaZ).cbz", "Berserk", 1, None, 2003),
    ("Attack on Titan v34 (2021) [HQ].cbz", "Attack on Titan", 34, None, 2021),
    ("Vinland Saga v25.cbz", "Vinland Saga", 25, None, None),
    ("One Piece - Vol. 103.cbz", "One Piece", 103, None, None),
    ("[LuCaZ] Berserk v01.cbz", "Berserk", 1, None, None),
    ("[Scan-Group] Series Name v01 (2019).cbz", "Series Name", 1, None, 2019),
    ("Solo Leveling c001.cbz", "Solo Leveling", None, 1.0, None),
    ("Chainsaw Man - Ch. 97.cbz", "Chainsaw Man", None, 97.0, None),
    ("One Piece c1050 (Digital).cbz", "One Piece", None, 1050.0, None),
    ("Berserk Ch. 12.5.cbz", "Berserk", None, 12.5, None),
    ("Berserk Deluxe v1-3 (2019).cbz", "Berserk Deluxe", 1, None, 2019),
    ("Akira (1982).cbz", "Akira", None, None, 1982),
    ("Saga #1 (2012).cbz", "Saga", None, 1.0, 2012),
    ("Series Name v01 (Magazine).cbz", "Series Name", 1, None, None),
    ("Berserk    v01   (2003).cbz", "Berserk", 1, None, 2003),
]


@pytest.mark.parametrize(
    "filename, series, volume, chapter, year", _CORPUS,
    ids=[c[0] for c in _CORPUS],
)
def test_corpus(
    filename: str,
    series: str,
    volume: int | None,
    chapter: float | None,
    year: int | None,
):
    p = parse_filename(filename)
    assert p.series == series, f"expected series={series!r} got {p.series!r}"
    assert p.volume == volume, f"expected volume={volume} got {p.volume}"
    assert p.chapter == chapter, f"expected chapter={chapter} got {p.chapter}"
    assert p.year == year, f"expected year={year} got {p.year}"
