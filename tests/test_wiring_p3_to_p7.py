"""Wiring program Phases 3-7 — registration pins + handler behavior.

P3 about-my-data reads, P4 introspection plane, P5 management verbs +
playlist provenance, P6 proactive bridges, P7 creation-tool pool +
artifact origin stamp. SQL-backed handlers run against a real
aiosqlite :memory: connection with minimal schemas — the real
migrations are exercised separately by the migration smoke.
"""

from __future__ import annotations

from types import SimpleNamespace

import aiosqlite
import pytest
import pytest_asyncio

from augmentum.intent.action import SessionContext

UID = "user-p37"


# ---------------------------------------------------------------------------
# Minimal schema fixture
# ---------------------------------------------------------------------------

_SCHEMAS = [
    """CREATE TABLE companion_journal (
        id INTEGER PRIMARY KEY, user_id TEXT, companion_id TEXT,
        entry_type TEXT, content TEXT, affect_tag TEXT,
        quarantined INTEGER DEFAULT 0, suppressed INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')))""",
    """CREATE TABLE companion_observations (
        id INTEGER PRIMARY KEY, companion_id TEXT, target_user_id TEXT,
        observation TEXT, surfaced INTEGER DEFAULT 0,
        confirmed INTEGER DEFAULT 0, denied INTEGER DEFAULT 0,
        ts TEXT DEFAULT (datetime('now')))""",
    """CREATE TABLE dream_entries (
        id TEXT PRIMARY KEY, user_id TEXT, content TEXT,
        entry_type TEXT, created_at TEXT DEFAULT (datetime('now')))""",
    """CREATE TABLE companion_skills (
        id INTEGER PRIMARY KEY, user_id TEXT, name TEXT,
        description TEXT, confidence REAL DEFAULT 0.5,
        instances_count INTEGER DEFAULT 0, status TEXT DEFAULT 'active',
        updated_at TEXT DEFAULT (datetime('now')))""",
    """CREATE TABLE playlists (
        id TEXT PRIMARY KEY, user_id TEXT, name TEXT,
        items_json TEXT DEFAULT '[]', origin TEXT NOT NULL DEFAULT '',
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')))""",
    """CREATE TABLE ui_sessions (
        id TEXT PRIMARY KEY, user_id TEXT, title TEXT,
        updated_at TEXT DEFAULT (datetime('now')))""",
    """CREATE TABLE file_index (
        id TEXT PRIMARY KEY, user_id TEXT, is_favorite INTEGER DEFAULT 0,
        updated_at TEXT DEFAULT (datetime('now')))""",
    """CREATE TABLE title_runs (
        id TEXT PRIMARY KEY, user_id TEXT, artifact_id TEXT,
        duration_s INTEGER)""",
    """CREATE TABLE artifacts (
        id TEXT PRIMARY KEY, display_name TEXT)""",
    """CREATE TABLE game_results (
        id INTEGER PRIMARY KEY, user_id TEXT, game_id TEXT,
        score INTEGER)""",
    """CREATE TABLE signal_events (
        id TEXT PRIMARY KEY, user_id TEXT, source TEXT, category TEXT,
        summary TEXT, occurrence_count INTEGER DEFAULT 1,
        status TEXT DEFAULT 'open',
        last_seen_at INTEGER DEFAULT 0)""",
    """CREATE TABLE companion_initiative_queue (
        id INTEGER PRIMARY KEY, companion_id TEXT, proposed_at REAL,
        kind TEXT, payload TEXT, score REAL, status TEXT,
        target_user_id TEXT)""",
]


@pytest_asyncio.fixture
async def conn():
    db = await aiosqlite.connect(":memory:")
    for ddl in _SCHEMAS:
        await db.execute(ddl)
    await db.commit()
    yield db
    await db.close()


def _ctx(conn, **extra):
    backend = SimpleNamespace(conn=conn)
    app_state = SimpleNamespace(
        state_manager=SimpleNamespace(backend=backend), **extra,
    )
    return SessionContext(user_id=UID, session_id="chat-1", app_state=app_state)


# ---------------------------------------------------------------------------
# P3 — about-my-data
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_my_playtime_aggregates(conn):
    from augmentum.intent.builtin.my_data import _my_playtime
    await conn.execute(
        "INSERT INTO artifacts (id, display_name) VALUES ('a1', 'Star Quest')",
    )
    for i, dur in enumerate((3600, 1800)):
        await conn.execute(
            "INSERT INTO title_runs (id, user_id, artifact_id, duration_s) "
            "VALUES (?, ?, 'a1', ?)", (f"r{i}", UID, dur),
        )
    await conn.execute(
        "INSERT INTO game_results (user_id, game_id, score) VALUES (?, 'bubble_pop', 420)",
        (UID,),
    )
    await conn.commit()
    res = await _my_playtime("", _ctx(conn), {})
    assert "Star Quest" in res.prompt_addendum
    assert "1.5h" in res.prompt_addendum
    assert "best score 420" in res.prompt_addendum


@pytest.mark.asyncio
async def test_my_jobs_uses_jobs_store(conn):
    from augmentum.intent.builtin.my_data import _my_jobs

    class FakeJobs:
        async def list_for_user(self, *, user_id, limit=10):
            assert user_id == UID
            return [
                {"job_type": "book_transcribe", "status": "running",
                 "progress": 0.4, "stage": "chapter 3"},
                {"job_type": "gguf_download", "status": "failed",
                 "error": "disk full"},
            ]

    res = await _my_jobs("", _ctx(conn, jobs_store=FakeJobs()), {})
    assert "book_transcribe: running 40% (chapter 3)" in res.prompt_addendum
    assert "disk full" in res.prompt_addendum


@pytest.mark.asyncio
async def test_my_jobs_includes_image_queue_snapshot(conn):
    from augmentum.intent.builtin.my_data import _my_jobs

    class FakeJobs:
        async def list_for_user(self, *, user_id, limit=10):
            return []

    job = SimpleNamespace(stage="Generating", steps_done=14, steps_total=30)

    class FakeImageQueue:
        _current_job_id = "img-1"
        queue_size = 2

        def get_job(self, job_id):
            return job if job_id == "img-1" else None

    res = await _my_jobs(
        "", _ctx(conn, jobs_store=FakeJobs(), image_queue=FakeImageQueue()), {},
    )
    assert "image generation: running — Generating (14/30)" in res.prompt_addendum
    assert "2 more queued" in res.prompt_addendum


@pytest.mark.asyncio
async def test_system_signals_reads_open_rows(conn):
    from augmentum.intent.builtin.my_data import _system_signals
    await conn.execute(
        "INSERT INTO signal_events (id, user_id, source, category, summary, "
        "occurrence_count) VALUES ('s1', ?, 'bug_finder', 'bug', "
        "'race in session sync', 3)", (UID,),
    )
    await conn.commit()
    res = await _system_signals("", _ctx(conn), {})
    assert "race in session sync" in res.prompt_addendum
    assert "seen 3x" in res.prompt_addendum


@pytest.mark.asyncio
async def test_system_health_reports_degraded_persistence(conn):
    from augmentum.intent.builtin.my_data import _system_health
    ctx = _ctx(conn)
    ctx.app_state.persistence_degraded = True
    res = await _system_health("", ctx, {})
    assert "PERSISTENCE DEGRADED" in res.prompt_addendum


# ---------------------------------------------------------------------------
# P4 — introspection plane
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_introspect_blends_facets_first_person(conn):
    from augmentum.intent.builtin.introspect import _introspect
    await conn.execute(
        "INSERT INTO companion_journal (user_id, entry_type, content) "
        "VALUES (?, 'wondering', 'whether he prefers mornings for hard work')",
        (UID,),
    )
    await conn.execute(
        "INSERT INTO companion_observations (target_user_id, observation, "
        "surfaced, confirmed) VALUES (?, 'plays the same game every friday', 0, 0)",
        (UID,),
    )
    await conn.execute(
        "INSERT INTO dream_entries (id, user_id, content, entry_type) "
        "VALUES ('d1', ?, 'a library with endless shelves', 'reflection')",
        (UID,),
    )
    await conn.commit()
    res = await _introspect("", _ctx(conn), {"facet": "all"})
    add = res.prompt_addendum
    assert "whether he prefers mornings" in add
    assert "never said aloud yet" in add
    assert "endless shelves" in add
    assert "first person" in add  # composition guidance rides along


@pytest.mark.asyncio
async def test_introspect_respects_visibility_flags(conn):
    from augmentum.intent.builtin.introspect import _introspect
    await conn.execute(
        "INSERT INTO companion_journal (user_id, entry_type, content, quarantined) "
        "VALUES (?, 'wondering', 'QUARANTINED THOUGHT', 1)", (UID,),
    )
    await conn.execute(
        "INSERT INTO companion_observations (target_user_id, observation, denied) "
        "VALUES (?, 'DENIED PATTERN', 1)", (UID,),
    )
    await conn.commit()
    res = await _introspect("", _ctx(conn), {})
    text = res.prompt_addendum or res.speak
    assert "QUARANTINED THOUGHT" not in text
    assert "DENIED PATTERN" not in text


@pytest.mark.asyncio
async def test_introspect_empty_is_honest(conn):
    from augmentum.intent.builtin.introspect import _introspect
    res = await _introspect("", _ctx(conn), {})
    assert res.short_circuit and "Nothing's accumulated" in res.speak


# ---------------------------------------------------------------------------
# P5 — management verbs + provenance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_playlist_create_stamps_companion_origin(conn):
    from augmentum.intent.builtin.manage import _playlist_create
    res = await _playlist_create("", _ctx(conn), {"name": "Late Night Coding"})
    assert "Late Night Coding" in res.speak
    cur = await conn.execute("SELECT name, origin FROM playlists")
    row = await cur.fetchone()
    assert row[0] == "Late Night Coding" and row[1] == "companion"


@pytest.mark.asyncio
async def test_playlist_delete_requires_confirm(conn):
    from augmentum.intent.builtin.manage import _playlist_delete
    await conn.execute(
        "INSERT INTO playlists (id, user_id, name) VALUES ('p1', ?, 'Old Mix')",
        (UID,),
    )
    await conn.commit()
    res = await _playlist_delete("", _ctx(conn), {"query": "old mix"})
    assert res.clarify is not None
    assert res.clarify["args"]["playlist_id"] == "p1"
    cur = await conn.execute("SELECT COUNT(*) FROM playlists")
    assert (await cur.fetchone())[0] == 1  # still there

    res = await _playlist_delete(
        "", _ctx(conn), {"playlist_id": "p1", "confirm": "yes"},
    )
    assert "Deleted" in res.speak
    cur = await conn.execute("SELECT COUNT(*) FROM playlists")
    assert (await cur.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_chat_rename_updates_current_session(conn):
    from augmentum.intent.builtin.manage import _chat_rename
    await conn.execute(
        "INSERT INTO ui_sessions (id, user_id, title) VALUES ('chat-1', ?, 'Untitled')",
        (UID,),
    )
    await conn.commit()
    res = await _chat_rename("", _ctx(conn), {"title": "Garden Planning"})
    assert "Garden Planning" in res.speak
    cur = await conn.execute("SELECT title FROM ui_sessions WHERE id='chat-1'")
    assert (await cur.fetchone())[0] == "Garden Planning"


@pytest.mark.asyncio
async def test_file_favorite_uses_last_file_referent(conn):
    from augmentum.intent.builtin.manage import _file_favorite
    await conn.execute(
        "INSERT INTO file_index (id, user_id) VALUES ('f1', ?)", (UID,),
    )
    await conn.commit()
    ctx = _ctx(conn)
    ctx.referents.last_file_id = "f1"
    res = await _file_favorite("favorite this", ctx, {})
    assert "Favorited" in res.speak
    cur = await conn.execute("SELECT is_favorite FROM file_index WHERE id='f1'")
    assert (await cur.fetchone())[0] == 1


# ---------------------------------------------------------------------------
# P6 — proactive bridges
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_job_terminal_listener_enqueues_initiative(conn, monkeypatch):
    from augmentum.companion_runtime import presence_mode
    from augmentum.companion_runtime.bridges import make_job_terminal_listener
    monkeypatch.setattr(presence_mode, "autonomy_allowed", lambda: True)
    app_state = SimpleNamespace(
        state_manager=SimpleNamespace(backend=SimpleNamespace(conn=conn)),
        companion_runtime=SimpleNamespace(companion_id="becca"),
    )
    listener = make_job_terminal_listener(app_state)
    event = SimpleNamespace(
        outcome="completed", job_type="book_transcribe", user_id=UID,
        job_id="j1", error="",
    )
    await listener(event)
    cur = await conn.execute(
        "SELECT kind, target_user_id, status FROM companion_initiative_queue",
    )
    row = await cur.fetchone()
    assert row == ("job_finished", UID, "pending")


@pytest.mark.asyncio
async def test_job_listener_skips_quiet_types_and_silent_mode(conn, monkeypatch):
    from augmentum.companion_runtime import presence_mode
    from augmentum.companion_runtime.bridges import make_job_terminal_listener
    app_state = SimpleNamespace(
        state_manager=SimpleNamespace(backend=SimpleNamespace(conn=conn)),
        companion_runtime=None,
    )
    listener = make_job_terminal_listener(app_state)

    # Quiet internal type — never enqueued, regardless of mode.
    monkeypatch.setattr(presence_mode, "autonomy_allowed", lambda: True)
    await listener(SimpleNamespace(
        outcome="completed", job_type="media_sync", user_id=UID,
        job_id="j2", error="",
    ))
    # SILENT floor — gated.
    monkeypatch.setattr(presence_mode, "autonomy_allowed", lambda: False)
    await listener(SimpleNamespace(
        outcome="completed", job_type="book_transcribe", user_id=UID,
        job_id="j3", error="",
    ))
    cur = await conn.execute("SELECT COUNT(*) FROM companion_initiative_queue")
    assert (await cur.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_signal_results_bridge(conn, monkeypatch):
    from augmentum.companion_runtime import presence_mode
    from augmentum.companion_runtime.bridges import bridge_signal_results
    monkeypatch.setattr(presence_mode, "autonomy_allowed", lambda: True)
    app_state = SimpleNamespace(
        state_manager=SimpleNamespace(backend=SimpleNamespace(conn=conn)),
        companion_runtime=None,
    )
    written = await bridge_signal_results(
        app_state,
        {UID: {"bug_finder": 2}, "other-user": {"bug_finder": 0}},
    )
    assert written == 1
    cur = await conn.execute(
        "SELECT kind, target_user_id FROM companion_initiative_queue",
    )
    assert (await cur.fetchone()) == ("signals_found", UID)


# ---------------------------------------------------------------------------
# P7 — creation pool + artifact origin
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_abilities_census_generated_from_registry():
    import augmentum.intent  # noqa: F401 — populate registry
    from augmentum.tools.context_peek import ContextPeekTool
    tool = ContextPeekTool(app_state=SimpleNamespace())
    res = await tool._peek_abilities("u1", "s1")
    assert res.success
    out = res.output
    # Census carries families and known verbs from every program phase.
    for marker in ("[media]", "[memory]", "[my]", "[playlist]",
                   "media.volume", "memory.forget", "my.taste",
                   "companion.introspect", "playlist.create"):
        assert marker in out, f"census missing {marker}"
    assert "always-on tools" in out
    assert res.metadata["verbs"] > 20  # the census, not a roster slice
    assert "answer from THIS" in out


def test_abilities_slot_registered():
    from augmentum.tools.context_peek import _SLOTS
    assert "abilities" in _SLOTS


def test_creation_tools_in_core_pool():
    from augmentum.companion_runtime.native_loop import CORE_TOOL_NAMES
    for name in ("create_document", "create_spreadsheet",
                 "create_presentation", "create_chart", "convert_document",
                 "youtube", "image_search", "remove_background"):
        assert name in CORE_TOOL_NAMES, f"{name} missing from loop pool"


def test_artifact_origin_contextvar_default_empty():
    from augmentum.tools.artifact_storage import ARTIFACT_ORIGIN
    assert ARTIFACT_ORIGIN.get() == ""
    token = ARTIFACT_ORIGIN.set("companion")
    assert ARTIFACT_ORIGIN.get() == "companion"
    ARTIFACT_ORIGIN.reset(token)
    assert ARTIFACT_ORIGIN.get() == ""


# ---------------------------------------------------------------------------
# Registration + bucket pins (all phases)
# ---------------------------------------------------------------------------

def test_p3_to_p5_verbs_registered_and_bucketed():
    import augmentum.intent  # noqa: F401
    from augmentum.intent.manifest import (
        VOICE_TOOLS_CORE,
        VOICE_TOOLS_DISRUPTIVE,
        VOICE_TOOLS_INTERACTIVE,
    )
    from augmentum.intent.registry import REGISTRY
    ids = {a.id for a in REGISTRY.all()}
    core = (
        "my.taste", "my.playtime", "my.jobs", "my.calls",
        "system.health", "system.signals", "companion.introspect",
        "playlist.create", "playlist.rename", "file.favorite",
        "chat.rename", "library.rename",
    )
    for verb in core:
        assert verb in ids, f"{verb} not registered"
        assert verb in VOICE_TOOLS_CORE, f"{verb} not in CORE bucket"
    assert "playlist.delete" in ids
    assert "playlist.delete" in VOICE_TOOLS_DISRUPTIVE
    for tool in ("create_document", "convert_document", "youtube",
                 "image_search"):
        assert tool in VOICE_TOOLS_INTERACTIVE
