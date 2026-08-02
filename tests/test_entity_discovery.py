"""Consumption-entity discovery — P0 guards + P1 substrate.

Pins the 2026-06-12 incident (curator paired the audiobook "My Quiet
Blacksmith Life in Another World" with a grack.com post about main()
internals on the single shared token "life") and tests the lane that
replaces it: series-level entity resolution, the Gate-1 catalog
recommender, curator Phase 0.5 composition, and the routing guards
that keep entities out of every topic-shaped web path.

Spec: docs/superpowers/specs/2026-06-12-consumption-entity-discovery-design.md
"""

from __future__ import annotations

import json

import pytest

_USER = "usr_entity_test"
_BOOK_TITLE = "My Quiet Blacksmith Life in Another World, Vol. 6"
_SERIES = "My Quiet Blacksmith Life in Another World"


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


async def _add_file(
    conn, *, file_id: str, name: str, meta: dict,
    user_id: str = _USER, source: str = "media_server",
) -> None:
    await conn.execute(
        """INSERT INTO file_index
           (id, user_id, source, source_id, name, mime_type, source_metadata)
           VALUES (?, ?, ?, ?, ?, 'audio/mpeg', ?)""",
        (file_id, user_id, source, file_id, name, json.dumps(meta)),
    )
    await conn.commit()


def _book_meta(**over) -> dict:
    meta = {
        "entity_kind": "audiobook",
        "series_name": _SERIES,
        "author": "Tamamaru",
        "author_normalized": "tamamaru",
        "narrator_normalized": "",
        "genres": ["Isekai", "Fantasy"],
        "year": 2023,
        "is_finished": False,
        "progress_pct": 0.0,
        "unplayed_count": 0,
    }
    meta.update(over)
    return meta


# ── P0 regression: the Blacksmith pairing must be impossible ────────


def test_one_generic_token_never_qualifies():
    """The exact incident: 6-token topic vs 'Life before main()' shared
    only 'life' → 1/6 = 0.167 cleared the 0.15 floor. Multi-token
    topics now require ≥2 overlapping tokens."""
    from augmentum.companion_runtime.curator import (
        _MIN_RELEVANCE_SCORE,
        score_relevance,
    )
    score = score_relevance(
        {"title": "Life before main()",
         "snippet": "What happens in a C program before main runs"},
        "My Quiet Blacksmith Life in Another World, Vol.",
    )
    assert score < _MIN_RELEVANCE_SCORE
    assert score == 0.0


def test_two_token_overlap_still_qualifies():
    from augmentum.companion_runtime.curator import score_relevance
    score = score_relevance(
        {"title": "Blacksmithing in another world: isekai craft tropes",
         "snippet": "blacksmith protagonists in fantasy worlds"},
        "My Quiet Blacksmith Life in Another World, Vol.",
    )
    assert score > 0.0


def test_single_token_topics_keep_ratio_rule():
    from augmentum.companion_runtime.curator import score_relevance
    score = score_relevance(
        {"title": "Rust 2.0 release notes", "snippet": ""},
        "rust",
    )
    assert score == 1.0


@pytest.mark.asyncio
async def test_derived_topics_exclude_entity_clusters():
    from augmentum.companion_runtime.curator import _derived_topics
    backend = await _fresh_backend()
    conn = backend.conn
    for cid, name, kind, frec in (
        ("c1", _SERIES, "entity", 99.0),
        ("c2", "rust async runtime", "topic", 5.0),
    ):
        await conn.execute(
            """INSERT INTO interest_clusters
               (cluster_id, name, kind, frecency_short, user_id)
               VALUES (?, ?, ?, ?, ?)""",
            (cid, name, kind, frec, _USER),
        )
    await conn.commit()
    names = await _derived_topics(conn, user_id=_USER)
    assert "rust async runtime" in names
    assert _SERIES not in names   # the latch is dead


def test_migration_backfill_flips_media_play_majority():
    """The 264 backfill marks media_play-majority clusters as entities."""
    from pathlib import Path
    mig = Path(
        "augmentum/state/migrations/264_entity_clusters.sql",
    ).read_text(encoding="utf-8")
    assert "kind = 'entity'" in mig
    assert "media_play" in mig
    # The HAVING clause requires strict majority, not mere presence.
    assert "* 2" in mig and "> COUNT(*)" in mig


# ── series_key: work-level identity ─────────────────────────────────


def test_series_key_strips_volume_suffix():
    from augmentum.discovery.entities import series_key
    assert series_key(_BOOK_TITLE) == series_key(
        "My Quiet Blacksmith Life in Another World, Vol. 7",
    )
    assert "vol" not in series_key(_BOOK_TITLE)


def test_series_key_prefers_series_name():
    from augmentum.discovery.entities import series_key
    assert series_key("Some Chapter Title #42", "One Piece") == "one piece"


def test_series_key_strips_repeated_suffixes():
    from augmentum.discovery.entities import series_key
    assert series_key("Saga Book 3 Part 2") == "saga"


def test_series_key_plain_title_unharmed():
    from augmentum.discovery.entities import series_key
    assert series_key("Project Hail Mary") == "project hail mary"


# ── Entity resolution + cluster assignment ──────────────────────────


@pytest.mark.asyncio
async def test_resolve_entity_from_catalog():
    from augmentum.discovery.entities import resolve_entity
    backend = await _fresh_backend()
    await _add_file(
        backend.conn, file_id="f6", name=_BOOK_TITLE, meta=_book_meta(),
    )
    entity = await resolve_entity(
        backend.conn, user_id=_USER, file_id="f6",
    )
    assert entity is not None
    assert entity["kind"] == "audiobook"
    assert entity["series_name"] == _SERIES
    assert "tamamaru" in entity["creators"]
    assert entity["local_refs"] == ["f6"]


@pytest.mark.asyncio
async def test_resolve_entity_is_user_scoped():
    from augmentum.discovery.entities import resolve_entity
    backend = await _fresh_backend()
    await _add_file(
        backend.conn, file_id="f6", name=_BOOK_TITLE, meta=_book_meta(),
    )
    other = await resolve_entity(
        backend.conn, user_id="usr_other", file_id="f6",
    )
    assert other is None


@pytest.mark.asyncio
async def test_assign_entity_signal_mints_series_level_cluster():
    from augmentum.discovery.entities import assign_entity_signal
    from augmentum.state.discovery_store import DiscoveryStore
    backend = await _fresh_backend()
    store = DiscoveryStore(backend.conn)
    await _add_file(
        backend.conn, file_id="f6", name=_BOOK_TITLE, meta=_book_meta(),
    )
    sig = await store.log_signal(
        signal_type="media_play", source_url="augm:media:f6",
        source_title=_BOOK_TITLE, content_type="audiobook",
        weight=1.0, metadata={"file_id": "f6"}, user_id=_USER,
    )
    cid = await assign_entity_signal(
        store, sig["id"], user_id=_USER, file_id="f6",
        fallback_title=_BOOK_TITLE,
    )
    assert cid
    cur = await backend.conn.execute(
        "SELECT name, kind, entity_ref FROM interest_clusters "
        "WHERE cluster_id = ?", (cid,),
    )
    row = await cur.fetchone()
    assert row[1] == "entity"
    # Display name is series-level — no dangling "Vol." artifact.
    assert "Vol" not in row[0]
    ref = json.loads(row[2])
    assert ref["series_key"]


@pytest.mark.asyncio
async def test_second_volume_joins_same_cluster():
    from augmentum.discovery.entities import assign_entity_signal
    from augmentum.state.discovery_store import DiscoveryStore
    backend = await _fresh_backend()
    store = DiscoveryStore(backend.conn)
    await _add_file(
        backend.conn, file_id="f6", name=_BOOK_TITLE, meta=_book_meta(),
    )
    await _add_file(
        backend.conn, file_id="f7",
        name="My Quiet Blacksmith Life in Another World, Vol. 7",
        meta=_book_meta(),
    )
    ids = []
    for fid in ("f6", "f7"):
        sig = await store.log_signal(
            signal_type="media_play", source_url=f"augm:media:{fid}",
            source_title=_BOOK_TITLE, content_type="audiobook",
            weight=1.0, metadata={"file_id": fid}, user_id=_USER,
        )
        ids.append(await assign_entity_signal(
            store, sig["id"], user_id=_USER, file_id=fid,
            fallback_title=_BOOK_TITLE,
        ))
    assert ids[0] == ids[1]   # vol 6 + vol 7 = ONE interest
    cur = await backend.conn.execute(
        "SELECT entity_ref FROM interest_clusters WHERE cluster_id = ?",
        (ids[0],),
    )
    ref = json.loads((await cur.fetchone())[0])
    assert set(ref["local_refs"]) == {"f6", "f7"}


@pytest.mark.asyncio
async def test_unresolvable_signal_still_avoids_topic_lane():
    from augmentum.discovery.entities import assign_entity_signal
    from augmentum.state.discovery_store import DiscoveryStore
    backend = await _fresh_backend()
    store = DiscoveryStore(backend.conn)
    sig = await store.log_signal(
        signal_type="media_play", source_url="augm:media:gone",
        source_title=_BOOK_TITLE, content_type="audiobook",
        weight=1.0, metadata={"file_id": "gone"}, user_id=_USER,
    )
    cid = await assign_entity_signal(
        store, sig["id"], user_id=_USER, file_id="gone",
        fallback_title=_BOOK_TITLE,
    )
    assert cid   # title-keyed entity cluster, NOT a topic cluster
    cur = await backend.conn.execute(
        "SELECT kind FROM interest_clusters WHERE cluster_id = ?", (cid,),
    )
    assert (await cur.fetchone())[0] == "entity"


# ── Gate 1 recommender ───────────────────────────────────────────────


async def _seed_library(conn) -> dict:
    """Vol 6 (80% in), vol 7 (fresh), same-author standalone, same-genre
    stranger, finished decoy, other user's item."""
    await _add_file(conn, file_id="f6", name=_BOOK_TITLE,
                    meta=_book_meta(progress_pct=80.0))
    await _add_file(
        conn, file_id="f7",
        name="My Quiet Blacksmith Life in Another World, Vol. 7",
        meta=_book_meta(),
    )
    await _add_file(
        conn, file_id="f8",
        name="My Quiet Blacksmith Life in Another World, Vol. 5",
        meta=_book_meta(is_finished=True, progress_pct=100.0),
    )
    await _add_file(
        conn, file_id="standalone",
        name="The Quiet Forge",
        meta=_book_meta(series_name="", genres=["Fantasy"]),
    )
    await _add_file(
        conn, file_id="genre1",
        name="Reborn as a Vending Machine",
        meta=_book_meta(series_name="Vending Machine",
                        author="Hirukuma", author_normalized="hirukuma"),
    )
    from augmentum.discovery.entities import resolve_entity
    entity = await resolve_entity(conn, user_id=_USER, file_id="f6")
    assert entity is not None
    return entity


@pytest.mark.asyncio
async def test_continuation_pick_is_next_unfinished_volume():
    from augmentum.discovery.entity_recommender import recommend_for_entity
    backend = await _fresh_backend()
    entity = await _seed_library(backend.conn)
    picks = await recommend_for_entity(
        backend.conn, entity, user_id=_USER,
    )
    cont = [p for p in picks if p.relation == "continuation"]
    assert cont and cont[0].file_id == "f7"      # vol 5 finished, vol 6 = self
    assert _SERIES in cont[0].why
    assert cont[0].gate == 1


@pytest.mark.asyncio
async def test_same_creator_pick_skips_in_series_and_consumed():
    from augmentum.discovery.entity_recommender import recommend_for_entity
    backend = await _fresh_backend()
    entity = await _seed_library(backend.conn)
    picks = await recommend_for_entity(
        backend.conn, entity, user_id=_USER,
    )
    creator = [p for p in picks if p.relation == "same_creator"]
    assert creator and creator[0].file_id == "standalone"
    assert "Tamamaru" in creator[0].why


@pytest.mark.asyncio
async def test_same_genre_pick_only_unstarted():
    from augmentum.discovery.entity_recommender import recommend_for_entity
    backend = await _fresh_backend()
    entity = await _seed_library(backend.conn)
    picks = await recommend_for_entity(
        backend.conn, entity, user_id=_USER,
    )
    genre = [p for p in picks if p.relation == "same_genre"]
    assert genre and genre[0].file_id == "genre1"


@pytest.mark.asyncio
async def test_recommender_is_user_scoped():
    from augmentum.discovery.entity_recommender import recommend_for_entity
    backend = await _fresh_backend()
    entity = await _seed_library(backend.conn)
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_b', 'b', 'x', datetime('now'))",
    )
    await backend.conn.commit()
    picks = await recommend_for_entity(
        backend.conn, entity, user_id="usr_b",
    )
    assert picks == []


@pytest.mark.asyncio
async def test_top_entity_picks_orders_by_frecency():
    from augmentum.discovery.entities import assign_entity_signal
    from augmentum.discovery.entity_recommender import top_entity_picks
    from augmentum.state.discovery_store import DiscoveryStore
    backend = await _fresh_backend()
    store = DiscoveryStore(backend.conn)
    await _seed_library(backend.conn)
    sig = await store.log_signal(
        signal_type="media_play", source_url="augm:media:f6",
        source_title=_BOOK_TITLE, content_type="audiobook",
        weight=1.0, metadata={"file_id": "f6"}, user_id=_USER,
    )
    await assign_entity_signal(
        store, sig["id"], user_id=_USER, file_id="f6",
        fallback_title=_BOOK_TITLE,
    )
    groups = await top_entity_picks(backend.conn, user_id=_USER)
    assert groups
    entity, picks = groups[0]
    assert entity["series_name"] == _SERIES
    assert any(p.relation == "continuation" for p in picks)


# ── Curator Phase 0.5 composition + dedup ────────────────────────────


@pytest.mark.asyncio
async def test_compose_entity_note_grounded_and_dedupable():
    from augmentum.companion_runtime.curator import (
        _seen_url_recently,
        compose_entity_note,
    )
    from augmentum.discovery.entity_recommender import EntityPick
    backend = await _fresh_backend()
    entity = {"series_name": _SERIES, "title": _BOOK_TITLE,
              "kind": "audiobook"}
    pick = EntityPick(
        relation="continuation", gate=1, file_id="f7",
        title="My Quiet Blacksmith Life in Another World, Vol. 7",
        kind="audiobook",
        why=f"next in {_SERIES}, ready in the library",
    )
    content, refs = compose_entity_note(entity, pick)
    assert _SERIES in content
    assert "ready in the library" in content
    assert refs and refs[0]["kind"] == "library_pick"
    assert refs[0]["file_id"] == "f7"

    # Write a journal row carrying the refs, then verify the dedup key
    # the picker checks matches what the composer stored.
    await backend.conn.execute(
        """INSERT INTO companion_journal
           (user_id, companion_id, entry_type, content, content_refs)
           VALUES (?, 'becca', 'curator_note', ?, ?)""",
        (_USER, content, json.dumps(refs)),
    )
    await backend.conn.commit()
    seen = await _seen_url_recently(
        backend.conn, user_id=_USER, companion_id="becca",
        url=f"augm:rec:{pick.relation}:{pick.file_id}",
    )
    assert seen is True


def test_entity_recs_setting_registered():
    from augmentum.config import settings
    assert hasattr(settings, "companion_entity_recs_enabled")
    from augmentum.proxy.config_routes import _TOOL_SETTINGS
    assert "companion_entity_recs_enabled" in _TOOL_SETTINGS


def test_media_recommendations_in_companion_pool():
    from augmentum.companion_runtime.native_loop import CORE_TOOL_NAMES
    assert "media_recommendations" in CORE_TOOL_NAMES


# ── Companion tool ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_media_recommendations_tool_outputs_grounded_picks():
    from types import SimpleNamespace

    from augmentum.discovery.entities import assign_entity_signal
    from augmentum.state.discovery_store import DiscoveryStore
    from augmentum.tools.media_recommendations import (
        MediaRecommendationsTool,
    )
    backend = await _fresh_backend()
    store = DiscoveryStore(backend.conn)
    await _seed_library(backend.conn)
    sig = await store.log_signal(
        signal_type="media_play", source_url="augm:media:f6",
        source_title=_BOOK_TITLE, content_type="audiobook",
        weight=1.0, metadata={"file_id": "f6"}, user_id=_USER,
    )
    await assign_entity_signal(
        store, sig["id"], user_id=_USER, file_id="f6",
        fallback_title=_BOOK_TITLE,
    )
    app_state = SimpleNamespace(
        state_manager=SimpleNamespace(
            backend=SimpleNamespace(conn=backend.conn),
        ),
    )
    tool = MediaRecommendationsTool(app_state)
    result = await tool.execute(_context={"user_id": _USER})
    assert result.success
    assert "Vol. 7" in result.output
    assert "continue" in result.output


# ── Offer/accept rail ────────────────────────────────────────────────


def _session_for(backend, user_id: str = _USER):
    from types import SimpleNamespace

    from augmentum.intent.action import ReferentCache, SessionContext
    app_state = SimpleNamespace(
        state_manager=SimpleNamespace(
            backend=SimpleNamespace(conn=backend.conn),
        ),
    )
    session = SessionContext(user_id=user_id, session_id="s1")
    session.app_state = app_state
    session.referents = ReferentCache()
    return session


@pytest.mark.asyncio
async def test_media_recommend_verb_offers_cards_and_parks():
    from augmentum.discovery.entities import assign_entity_signal
    from augmentum.intent.builtin.media import _media_recommend
    from augmentum.state.discovery_store import DiscoveryStore
    backend = await _fresh_backend()
    store = DiscoveryStore(backend.conn)
    await _seed_library(backend.conn)
    sig = await store.log_signal(
        signal_type="media_play", source_url="augm:media:f6",
        source_title=_BOOK_TITLE, content_type="audiobook",
        weight=1.0, metadata={"file_id": "f6"}, user_id=_USER,
    )
    await assign_entity_signal(
        store, sig["id"], user_id=_USER, file_id="f6",
        fallback_title=_BOOK_TITLE,
    )
    session = _session_for(backend)
    result = await _media_recommend("recommend me something", session, {})
    assert result.short_circuit
    assert result.surface_emit is not None
    assert result.surface_emit["channel"] == "companion.candidates"
    cards = result.surface_emit["payload"]["candidates"]
    assert cards and cards[0]["file_id"]
    # Parked for the spoken/typed follow-up.
    assert session.referents.pending_candidates == cards
    # Speak names the option with its grounding, numbered for ordinals.
    assert "1:" in result.speak


@pytest.mark.asyncio
async def test_media_recommend_honest_when_library_empty():
    from augmentum.intent.builtin.media import _media_recommend
    backend = await _fresh_backend()
    session = _session_for(backend)
    result = await _media_recommend("recommend me something", session, {})
    assert result.short_circuit
    assert result.surface_emit is None
    # Honest miss now offers to widen / look outside the library.
    assert "library" in result.speak.lower()


@pytest.mark.asyncio
async def test_media_play_file_id_accept_fast_path():
    from augmentum.intent.builtin.media import _media_play
    backend = await _fresh_backend()
    await _seed_library(backend.conn)
    session = _session_for(backend)
    session.referents.pending_candidates = [{"file_id": "f7", "title": "x"}]
    result = await _media_play("the second one", session, {"file_id": "f7"})
    assert result.short_circuit
    assert result.surface_emit["channel"] == "media.resume"
    assert result.surface_emit["payload"]["file_id"] == "f7"
    # Accepting clears the parked offer.
    assert session.referents.pending_candidates == []


@pytest.mark.asyncio
async def test_media_play_file_id_is_ownership_checked():
    from augmentum.intent.builtin.media import _direct_play_by_file_id
    backend = await _fresh_backend()
    await _seed_library(backend.conn)
    session = _session_for(backend, user_id="usr_other")
    result = await _direct_play_by_file_id(session, "f7")
    assert result is None   # not their library → no trust-the-id play


def test_router_prompt_renders_offered_picks():
    from augmentum.architect.router import ConfidenceStack, _format_signals
    stack = ConfidenceStack(
        offered_candidates=[
            {"file_id": "f7", "title": "Vol. 7", "subtitle": "next in series"},
            {"file_id": "g1", "title": "Vending Machine"},
        ],
    )
    rendered = _format_signals(stack)
    assert "OFFERED PICKS" in rendered
    assert "file_id=f7" in rendered
    assert "media.play" in rendered
    # Declines must not turn into searches.
    assert "REJECT" in rendered


def test_router_prompt_omits_block_without_offer():
    from augmentum.architect.router import ConfidenceStack, _format_signals
    assert "OFFERED PICKS" not in _format_signals(ConfidenceStack())


@pytest.mark.asyncio
async def test_tool_parks_candidates_and_queues_cards():
    from types import SimpleNamespace

    from augmentum.discovery.entities import assign_entity_signal
    from augmentum.intent.dispatch import get_referent_cache
    from augmentum.state.discovery_store import DiscoveryStore
    from augmentum.tools.media_recommendations import (
        MediaRecommendationsTool,
    )
    backend = await _fresh_backend()
    store = DiscoveryStore(backend.conn)
    await _seed_library(backend.conn)
    sig = await store.log_signal(
        signal_type="media_play", source_url="augm:media:f6",
        source_title=_BOOK_TITLE, content_type="audiobook",
        weight=1.0, metadata={"file_id": "f6"}, user_id=_USER,
    )
    await assign_entity_signal(
        store, sig["id"], user_id=_USER, file_id="f6",
        fallback_title=_BOOK_TITLE,
    )
    app_state = SimpleNamespace(
        state_manager=SimpleNamespace(
            backend=SimpleNamespace(conn=backend.conn),
        ),
    )
    tool = MediaRecommendationsTool(app_state)
    result = await tool.execute(
        _context={"user_id": _USER, "session_id": "s1"},
    )
    assert result.success
    assert "tappable cards" in result.output
    refs = get_referent_cache(app_state, _USER, "s1")
    assert refs.pending_candidates
    assert refs.pending_candidates[0]["file_id"]
    assert refs.pending_surface_events
    ev = refs.pending_surface_events[-1]
    assert ev["surface"]["channel"] == "companion.candidates"


@pytest.mark.asyncio
async def test_media_recommendations_tool_honest_when_empty():
    from types import SimpleNamespace

    from augmentum.tools.media_recommendations import (
        MediaRecommendationsTool,
    )
    backend = await _fresh_backend()
    app_state = SimpleNamespace(
        state_manager=SimpleNamespace(
            backend=SimpleNamespace(conn=backend.conn),
        ),
    )
    tool = MediaRecommendationsTool(app_state)
    result = await tool.execute(_context={"user_id": _USER})
    assert result.success
    # Empty catalog + no history → honest miss (don't invent a title).
    assert "nothing" in result.output.lower()
    assert "library" in result.output.lower()
