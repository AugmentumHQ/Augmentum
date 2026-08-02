"""Unit tests for the Live TV rail categorizer.

Pure-function tests — no HTTP, no DB, no provider stubs needed.
Verifies the four signal classes the rail builder consumes
(``UserData.IsFavorite``, ``UserData.PlayCount``, EPG flags on
``current_program``, hand-curated network-name hints) plus the
ordering invariants the UI relies on.
"""

from __future__ import annotations

import pytest

from augmentum.media.livetv_rails import Rail, categorize_channels
from augmentum.media.providers.base import CatalogItem


def _ch(
    name: str,
    *,
    number: str = "",
    is_favorite: bool = False,
    play_count: int = 0,
    current_program: dict | None = None,
    source: str = "emby",
    server_id: str = "",
) -> CatalogItem:
    return CatalogItem(
        external_id=name.lower().replace(" ", "-"),
        name=name,
        kind="live_video",
        mime_type="application/vnd.apple.mpegurl",
        extra={
            "channel_number":  number,
            "is_favorite":     is_favorite,
            "play_count":      play_count,
            "current_program": current_program,
            "source_provider": source,
            "server_id":       server_id,
        },
    )


def _ids(rail: Rail) -> list[str]:
    return [c.external_id for c in rail.channels]


def _by_id(rails: list[Rail]) -> dict[str, Rail]:
    return {r.id: r for r in rails}


# ── Empty / degenerate cases ──────────────────────────────────────

def test_empty_input_returns_no_rails():
    assert categorize_channels([]) == []


def test_non_live_channels_are_ignored():
    """A stray VOD ``CatalogItem`` shouldn't slip into the rail set
    — the categorizer filters by ``kind='live_video'`` so callers
    don't have to."""
    vod = CatalogItem(
        external_id="movie-1", name="Some Movie", kind="video",
        mime_type="video/mp4", extra={"channel_number": "1"},
    )
    rails = categorize_channels([vod])
    assert rails == []


# ── Favorites + recent rails (UserData-driven) ────────────────────

def test_favorites_rail_appears_first_when_populated():
    rails = categorize_channels([
        _ch("CNN",  number="201", is_favorite=True),
        _ch("HGTV", number="229"),
    ])
    assert rails[0].id == "favorites"
    assert _ids(rails[0]) == ["cnn"]


def test_favorites_rail_dropped_when_empty():
    rails = categorize_channels([_ch("HGTV", number="229")])
    assert "favorites" not in _by_id(rails)


def test_recent_rail_sorts_by_play_count_desc():
    rails = categorize_channels([
        _ch("HGTV",        number="229", play_count=3),
        _ch("AMC",         number="231", play_count=11),
        _ch("Food Network", number="232", play_count=0),
    ])
    recent = _by_id(rails)["recent"]
    assert _ids(recent) == ["amc", "hgtv"]


def test_recent_rail_tiebreaks_on_channel_number():
    rails = categorize_channels([
        _ch("HGTV", number="229", play_count=2),
        _ch("AMC",  number="231", play_count=2),
    ])
    recent = _by_id(rails)["recent"]
    assert _ids(recent) == ["hgtv", "amc"]


# ── Thematic rails: name hints ────────────────────────────────────

def test_news_rail_from_network_name():
    rails = categorize_channels([_ch("CNN", number="200")])
    assert "news" in _by_id(rails)
    assert _ids(_by_id(rails)["news"]) == ["cnn"]


def test_sports_rail_from_network_name():
    rails = categorize_channels([_ch("ESPN", number="206")])
    assert "sports" in _by_id(rails)


def test_kids_rail_from_network_name():
    rails = categorize_channels([_ch("Cartoon Network", number="298")])
    assert "kids" in _by_id(rails)


def test_unknown_network_only_lands_in_all_channels():
    """The default outcome — *All Channels* — is the floor for
    anything that doesn't match a thematic signal."""
    rails = categorize_channels([_ch("Local Public Access 27", number="27")])
    by_id = _by_id(rails)
    assert set(by_id) == {"all"}


# ── Thematic rails: EPG flags ─────────────────────────────────────

def test_epg_is_news_promotes_into_news_rail():
    """Channel name has zero news hint, but the EPG says it's airing
    a news program right now — should appear in *Live News* anyway."""
    rails = categorize_channels([
        _ch("KCTS-HD", number="9.1", current_program={"is_news": True}),
    ])
    assert "news" in _by_id(rails)


def test_epg_is_sports_promotes_into_sports_rail():
    rails = categorize_channels([
        _ch("KTVU", number="2.1", current_program={"is_sports": True}),
    ])
    assert "sports" in _by_id(rails)


def test_epg_movie_or_series_promotes_into_movies_rail():
    rails = categorize_channels([
        _ch("KGO-HD", number="7.1", current_program={"is_movie": True}),
        _ch("KPIX",   number="5.1", current_program={"is_series": True}),
    ])
    movies = _by_id(rails)["movies"]
    assert {"kgo-hd", "kpix"} == set(_ids(movies))


def test_malformed_current_program_does_not_crash():
    """Defensive: a provider returning ``current_program`` as a list
    or string shouldn't blow up the categorizer."""
    bad = _ch("Weird Channel", number="42")
    bad.extra["current_program"] = ["unexpected", "shape"]
    rails = categorize_channels([bad])
    # Lands only in *All Channels*; thematic predicates short-circuit.
    assert set(r.id for r in rails) == {"all"}


# ── Multi-rail membership ─────────────────────────────────────────

def test_channel_can_appear_in_multiple_thematic_rails():
    """ESPN normally lives in *Sports*; if its EPG says a movie is
    airing right now, it should ALSO show up in *Movies & Shows*
    — multi-rail membership is the design."""
    rails = categorize_channels([
        _ch("ESPN", number="206", current_program={"is_movie": True}),
    ])
    by_id = _by_id(rails)
    assert "sports" in by_id and "espn" in _ids(by_id["sports"])
    assert "movies" in by_id and "espn" in _ids(by_id["movies"])


# ── Local OTA detection ───────────────────────────────────────────

def test_dotted_channel_number_lands_in_ota_rail():
    rails = categorize_channels([
        _ch("KQED",   number="9.1"),
        _ch("KQED+",  number="9.2"),
        _ch("HGTV",   number="229"),
    ])
    ota = _by_id(rails).get("ota")
    assert ota is not None
    assert set(_ids(ota)) == {"kqed", "kqed+"}


def test_integer_channel_number_does_not_land_in_ota():
    rails = categorize_channels([_ch("HGTV", number="229")])
    assert "ota" not in _by_id(rails)


# ── Sort key ──────────────────────────────────────────────────────

def test_all_channels_sorts_by_major_minor_then_name():
    rails = categorize_channels([
        _ch("Channel B", number="7"),
        _ch("Channel D", number="6.2"),
        _ch("Channel C", number="6.10"),
        _ch("Channel A", number="6"),
    ])
    all_rail = _by_id(rails)["all"]
    # 6 < 6.2 < 6.10 < 7
    assert _ids(all_rail) == ["channel-a", "channel-d", "channel-c", "channel-b"]


def test_unparseable_channel_number_sorts_last():
    rails = categorize_channels([
        _ch("Z Last",  number=""),
        _ch("A First", number="100"),
    ])
    all_rail = _by_id(rails)["all"]
    assert _ids(all_rail) == ["a-first", "z-last"]


def test_string_channel_number_sorts_last_not_crashes():
    rails = categorize_channels([
        _ch("Symbol", number="MTV"),
        _ch("Number", number="200"),
    ])
    all_rail = _by_id(rails)["all"]
    assert _ids(all_rail) == ["number", "symbol"]


# ── Rail order invariant ──────────────────────────────────────────

def test_rail_order_is_stable_and_predictable():
    """Favorites → Recent → thematic (news/sports/movies/music/kids)
    → OTA → All. This is the order the UI scrolls top-to-bottom."""
    rails = categorize_channels([
        _ch("CNN",   number="200", is_favorite=True, play_count=4),
        _ch("ESPN",  number="206"),
        _ch("MTV",   number="210"),
        _ch("Cartoon Network", number="298"),
        _ch("AMC",   number="231", current_program={"is_movie": True}),
        _ch("KQED",  number="9.1"),
    ])
    ids = [r.id for r in rails]
    expected = ["favorites", "recent", "news", "sports", "movies", "music", "kids", "ota", "all"]
    assert ids == expected


# ── to_dict serialization ─────────────────────────────────────────

def test_to_dict_preserves_caller_injected_server_id():
    """The route layer aggregates across multiple servers per user.
    It tags each channel with ``server_id`` via the extra dict
    BEFORE handing the list to the categorizer; the serializer
    must surface that to the UI so the play path can route back
    to the right server."""
    rails = categorize_channels([
        _ch("CNN", number="200", server_id="emby-home"),
    ])
    payload = rails[0].to_dict()
    assert payload["channels"][0]["server_id"] == "emby-home"


def test_to_dict_shape_has_required_ui_fields():
    rails = categorize_channels([
        _ch("CNN", number="200",
            current_program={"name": "CNN Newsroom", "is_news": True}),
    ])
    chan = rails[0].to_dict()["channels"][0]
    required = {
        "external_id", "name", "cover_url", "channel_number",
        "source_provider", "server_id", "is_favorite", "play_count",
        "has_logo_primary", "has_logo_light", "has_logo_dark",
        "current_program",
    }
    assert required.issubset(chan)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
