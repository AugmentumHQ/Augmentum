"""Catalog-grounded recommendations — the foundation that makes "recommend
a movie" work from the user's OWN catalog, not just play-history.

Pins the 2026-06-18 fix: the companion's only recommendation path read
``interest_clusters`` (what you've PLAYED), so a fully-indexed library that
had never been played returned "nothing found". ``catalog_recommender``
makes ``file_index`` the always-available base, filtered by the request and
re-ranked by taste. These tests guard that a fresh library still yields
real picks, that type/genre filters reach the catalog, and that play-history
becomes a ranking boost (not a gate).
"""

from __future__ import annotations

import json

import pytest

from augmentum.discovery.catalog_recommender import (
    catalog_picks,
    kind_word_to_filters,
    recommend_picks,
)

_USER = "usr_catalog_test"


async def _fresh_backend(user_id: str = _USER):
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (user_id, "tester", "x"),
    )
    await backend.conn.commit()
    return backend


async def _add_media(
    conn, *, file_id: str, name: str, kind: str, meta: dict,
    user_id: str = _USER, source: str = "jellyfin",
    mime: str = "video/mp4",
) -> None:
    """Seed a file_index row with the ``kind`` column set — media-server
    sync writes it via derive_kind; the catalog recommender filters on it."""
    await conn.execute(
        """INSERT INTO file_index
           (id, user_id, source, source_id, name, mime_type, kind,
            source_metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (file_id, user_id, source, file_id, name, mime, kind,
         json.dumps(meta)),
    )
    await conn.commit()


def _movie(**over) -> dict:
    meta = {
        "entity_kind": "movie",
        "genres": ["Comedy"],
        "year": 2021,
        "is_finished": False,
        "progress_pct": 0.0,
    }
    meta.update(over)
    return meta


def _file_index(backend):
    from augmentum.vfs.index import FileIndexService
    return FileIndexService(backend.conn)


# ── kind-word vocabulary ─────────────────────────────────────────────


def test_kind_word_to_filters_disambiguates():
    assert kind_word_to_filters("movie") == ("video", "movie")
    assert kind_word_to_filters("show") == ("video", "series")
    assert kind_word_to_filters("audiobook") == ("audio", "book")
    assert kind_word_to_filters("manga") == ("comic", "manga")
    # Broad family words leave entity_kind open.
    assert kind_word_to_filters("video") == ("video", "")
    # Unknown → no filter.
    assert kind_word_to_filters("squiggle") == ("", "")
    assert kind_word_to_filters("") == ("", "")


# ── catalog_picks: the core fix ──────────────────────────────────────


@pytest.mark.asyncio
async def test_catalog_picks_surfaces_unwatched_movie_with_no_history():
    """THE bug: a fully-indexed movie library, never played, must still
    yield picks — drawn straight from the catalog."""
    backend = await _fresh_backend()
    await _add_media(backend.conn, file_id="m1", name="The Big Lebowski",
                     kind="video", meta=_movie())
    await _add_media(backend.conn, file_id="m2", name="Superbad",
                     kind="video", meta=_movie())
    await _add_media(backend.conn, file_id="seen", name="Old Watched Film",
                     kind="video",
                     meta=_movie(is_finished=True, progress_pct=100.0))

    picks = await catalog_picks(
        _file_index(backend), user_id=_USER, kind="video",
        entity_kind="movie",
    )
    ids = {p.file_id for p in picks}
    assert ids == {"m1", "m2"}        # the finished film is excluded
    assert all(p.gate == 1 for p in picks)
    assert all(p.file_id for p in picks)


@pytest.mark.asyncio
async def test_catalog_picks_genre_filter_reaches_catalog():
    backend = await _fresh_backend()
    await _add_media(backend.conn, file_id="c1", name="A Funny One",
                     kind="video", meta=_movie(genres=["Comedy"]))
    await _add_media(backend.conn, file_id="h1", name="A Scary One",
                     kind="video", meta=_movie(genres=["Horror"]))

    picks = await catalog_picks(
        _file_index(backend), user_id=_USER, kind="video",
        entity_kind="movie", genre="comedy",
    )
    assert {p.file_id for p in picks} == {"c1"}


@pytest.mark.asyncio
async def test_catalog_picks_taste_reranks_toward_known_genres():
    backend = await _fresh_backend()
    await _add_media(backend.conn, file_id="drama", name="Heavy Drama",
                     kind="video", meta=_movie(genres=["Drama"]))
    await _add_media(backend.conn, file_id="comedy", name="Light Comedy",
                     kind="video", meta=_movie(genres=["Comedy"]))

    taste = {"series": set(), "creators": set(), "genres": {"comedy"}}
    picks = await catalog_picks(
        _file_index(backend), user_id=_USER, kind="video",
        entity_kind="movie", taste=taste, limit=2,
    )
    assert picks[0].file_id == "comedy"   # taste overlap floats up
    assert picks[0].relation == "for_you"


@pytest.mark.asyncio
async def test_catalog_picks_in_progress_is_continuation():
    backend = await _fresh_backend()
    await _add_media(backend.conn, file_id="resume", name="Half-Watched",
                     kind="video", meta=_movie(progress_pct=40.0))
    picks = await catalog_picks(
        _file_index(backend), user_id=_USER, kind="video",
    )
    assert picks and picks[0].file_id == "resume"
    assert picks[0].relation == "continuation"
    assert "resume" in picks[0].why.lower()


@pytest.mark.asyncio
async def test_catalog_picks_skips_nonplayable_kinds():
    backend = await _fresh_backend()
    await _add_media(backend.conn, file_id="img", name="A Photo",
                     kind="image", mime="image/png", meta={})
    picks = await catalog_picks(_file_index(backend), user_id=_USER)
    assert picks == []


@pytest.mark.asyncio
async def test_catalog_picks_user_scoped():
    backend = await _fresh_backend()
    await _add_media(backend.conn, file_id="m1", name="Mine",
                     kind="video", meta=_movie())
    picks = await catalog_picks(
        _file_index(backend), user_id="usr_other", kind="video",
    )
    assert picks == []


@pytest.mark.asyncio
async def test_catalog_picks_empty_file_index_is_safe():
    picks = await catalog_picks(None, user_id=_USER, kind="video")
    assert picks == []


# ── recommend_picks: unified play-history + catalog ──────────────────


@pytest.mark.asyncio
async def test_recommend_picks_falls_back_to_catalog_with_no_history():
    backend = await _fresh_backend()
    await _add_media(backend.conn, file_id="m1", name="Catalog Movie",
                     kind="video", meta=_movie())
    picks = await recommend_picks(
        backend.conn, _file_index(backend), user_id=_USER,
        kind="video", entity_kind="movie",
    )
    assert {p.file_id for p in picks} == {"m1"}


@pytest.mark.asyncio
async def test_recommend_picks_genre_skips_mismatched_history():
    """A genre-specific ask must not surface unrelated play-history picks;
    the catalog leg (which filters genre in SQL) carries it."""
    backend = await _fresh_backend()
    # A played audiobook entity (would be a continuation pick).
    await backend.conn.execute(
        """INSERT INTO interest_clusters
           (cluster_id, name, kind, frecency_short, frecency_long,
            user_id, entity_ref)
           VALUES ('e1', 'Some Saga', 'entity', 10, 5, ?, ?)""",
        (_USER, json.dumps({
            "series_key": "some saga", "series_name": "Some Saga",
            "kind": "audio", "creators": ["x"], "genres": ["fantasy"],
        })),
    )
    await backend.conn.commit()
    # A comedy movie in the catalog.
    await _add_media(backend.conn, file_id="cm", name="Comedy Movie",
                     kind="video", meta=_movie(genres=["Comedy"]))

    picks = await recommend_picks(
        backend.conn, _file_index(backend), user_id=_USER,
        kind="video", entity_kind="movie", genre="comedy",
    )
    assert {p.file_id for p in picks} == {"cm"}


@pytest.mark.asyncio
async def test_recommend_picks_empty_returns_empty():
    backend = await _fresh_backend()
    picks = await recommend_picks(
        backend.conn, _file_index(backend), user_id=_USER, kind="video",
    )
    assert picks == []


# ── tool integration: the original report path ───────────────────────


@pytest.mark.asyncio
async def test_media_recommendations_tool_uses_catalog_with_no_history():
    """End-to-end: ask the companion tool for a movie with a never-played
    catalog → it names a real movie instead of 'nothing found'."""
    from types import SimpleNamespace

    from augmentum.tools.media_recommendations import (
        MediaRecommendationsTool,
    )
    backend = await _fresh_backend()
    await _add_media(backend.conn, file_id="m1", name="The Grand Budapest Hotel",
                     kind="video", meta=_movie())
    app_state = SimpleNamespace(
        state_manager=SimpleNamespace(
            backend=SimpleNamespace(conn=backend.conn),
        ),
        file_index=_file_index(backend),
    )
    tool = MediaRecommendationsTool(app_state)
    result = await tool.execute(_context={"user_id": _USER}, kind="movie")
    assert result.success
    assert "Grand Budapest" in result.output
