"""Author normalisation for related-book matching.

Regression for the 2026-06-17 "Nothing related found" bug: ABS bakes a
contributor role into the author array ("Jenny McKeon - translator"), which
polluted the author credit and broke token-subset matching across a series
whose volumes credit different translators/illustrators.
"""

from __future__ import annotations

from augmentum.media.normalize import (
    author_for_match,
    normalize_name,
    tokens_match_as_related,
)


class TestAuthorForMatch:
    def test_drops_translator_credit(self):
        assert author_for_match("Okina Baba, Jenny McKeon - translator") == "Okina Baba"

    def test_drops_illustrator_credit(self):
        assert author_for_match("Okina Baba, Tsukasa Kiryu - illustrator") == "Okina Baba"

    def test_plain_author_unchanged(self):
        assert author_for_match("Okina Baba") == "Okina Baba"

    def test_reversed_name_kept(self):
        # "Last, First" convention has no role suffix — must not be dropped.
        assert author_for_match("Sanderson, Brandon") == "Sanderson, Brandon"

    def test_compound_hyphen_name_not_mistaken_for_role(self):
        # No spaces around the hyphen → not a " - role" suffix.
        assert author_for_match("Jean-Paul Sartre") == "Jean-Paul Sartre"

    def test_real_coauthors_kept(self):
        assert author_for_match("Jane Austen, Charles Dickens") == "Jane Austen, Charles Dickens"

    def test_never_empties_when_only_role_credit(self):
        # Degenerate data (only a translator) returns the original rather
        # than producing an authorless book.
        assert author_for_match("Jenny McKeon - translator") == "Jenny McKeon - translator"

    def test_empty_passthrough(self):
        assert author_for_match("") == ""


class TestSeriesVolumesMatch:
    def test_volumes_with_different_secondary_credits_still_match(self):
        """The actual bug: Vol 10 credits a translator, another volume an
        illustrator. They share the real author (Okina Baba) and MUST match
        once the role credits are stripped."""
        v10 = normalize_name(author_for_match("Okina Baba, Jenny McKeon - translator"))
        v9 = normalize_name(author_for_match("Okina Baba, Tsukasa Kiryu - illustrator"))
        v1 = normalize_name(author_for_match("Okina Baba"))
        assert tokens_match_as_related(v10, v9)
        assert tokens_match_as_related(v10, v1)

    def test_distinct_authors_still_rejected(self):
        """Stripping roles must not over-match: two different authors who
        share only a surname stay unrelated."""
        a = normalize_name(author_for_match("Jane Smith"))
        b = normalize_name(author_for_match("John Smith"))
        assert not tokens_match_as_related(a, b)


# ── Series axis (2026-06-17) — "More in this series" for audiobooks ─────

from augmentum.media.providers.audiobookshelf import _abs_series  # noqa: E402


class TestAbsSeriesExtraction:
    def test_detail_array_shape(self):
        meta = {"series": [{"name": "So I'm a Spider, So What?", "sequence": "10"}]}
        assert _abs_series(meta) == ("So I'm a Spider, So What?", "10")

    def test_flattened_seriesname_with_hash_sequence(self):
        assert _abs_series({"seriesName": "So I'm a Spider, So What? #10"}) == (
            "So I'm a Spider, So What?", "10",
        )

    def test_flattened_seriesname_no_sequence(self):
        assert _abs_series({"seriesName": "Standalone Saga"}) == ("Standalone Saga", "")

    def test_not_in_a_series(self):
        assert _abs_series({}) == ("", "")
        assert _abs_series({"series": []}) == ("", "")

    def test_string_array_shape(self):
        assert _abs_series({"series": ["Mistborn"]}) == ("Mistborn", "")


class TestSeriesSequenceOrder:
    def test_numeric_volumes_sort_naturally(self):
        from augmentum.proxy.media_routes import _series_sequence_key
        keys = [_series_sequence_key(s) for s in ("2", "10", "1")]
        assert sorted(keys) == [
            _series_sequence_key("1"),
            _series_sequence_key("2"),
            _series_sequence_key("10"),
        ]

    def test_numbered_volumes_sort_before_unlabelled(self):
        from augmentum.proxy.media_routes import _series_sequence_key
        assert _series_sequence_key("10") < _series_sequence_key("Special")
        assert _series_sequence_key("") > _series_sequence_key("99")
