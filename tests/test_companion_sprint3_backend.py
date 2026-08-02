"""Sprint 3 backend tests — Piece 12 endpoints + Piece 10 pre-context.

Tests at the SQL + module level (avoiding TestClient/auth-scope
fragility from earlier — pattern proven in test_companion_notes_pip.py).
"""

from __future__ import annotations

import json

import pytest


async def _boot_backend():
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_t', 'test', 'x', datetime('now'))",
    )
    await backend.conn.commit()
    return backend


# ── Migration 180 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mig180_creates_topic_mutes_table():
    backend = await _boot_backend()
    cur = await backend.conn.execute("PRAGMA table_info(companion_topic_mutes)")
    cols = {c[1] for c in await cur.fetchall()}
    await cur.close()
    expected = {"id", "user_id", "companion_id", "scope_json", "note_id",
                "created_at", "expires_at"}
    assert expected.issubset(cols)


@pytest.mark.asyncio
async def test_mig180_indexes_exist():
    backend = await _boot_backend()
    cur = await backend.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    )
    names = {r[0] for r in await cur.fetchall()}
    await cur.close()
    assert "idx_topic_mutes_user_active" in names
    assert "idx_topic_mutes_user_time" in names


# ── Endpoint registration smoke ──────────────────────────────────────


def test_endpoints_registered():
    """The 3 new endpoints (acknowledged, muted_topic, plus existing
    surfaced) all live on the router."""
    from augmentum.proxy.companion_routes import router
    paths = {getattr(r, "path", "") for r in router.routes}
    assert "/api/companion/notes/{note_id}/acknowledged" in paths
    assert "/api/companion/notes/{note_id}/muted_topic" in paths
    assert "/api/companion/notes/{note_id}/surfaced" in paths


# ── Mute scope extraction ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scope_extraction_pulls_keywords():
    """Note content → keywords in scope."""
    from augmentum.proxy.companion_routes import _extract_mute_scope
    backend = await _boot_backend()
    scope = await _extract_mute_scope(
        backend, user_id="usr_t",
        content="The prefix caching paper relates to KV restoration work.",
        refs=[],
    )
    assert isinstance(scope["keywords"], list)
    assert len(scope["keywords"]) > 0
    # Stopwords filtered
    assert "the" not in scope["keywords"]


@pytest.mark.asyncio
async def test_scope_extraction_pulls_domains_from_file_refs():
    """Refs of kind=file → domains from file_index.source_metadata.source_url."""
    from augmentum.proxy.companion_routes import _extract_mute_scope
    backend = await _boot_backend()
    # Plant a file_index row with a source_url
    await backend.conn.execute(
        "INSERT INTO file_index "
        "(id, user_id, source, source_id, name, mime_type, source_metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("fi_abc", "usr_t", "browse", "src_1", "article.html", "text/html",
         '{"source_url": "https://example.com/article"}'),
    )
    await backend.conn.commit()
    scope = await _extract_mute_scope(
        backend, user_id="usr_t",
        content="A topic note",
        refs=[{"kind": "file", "id": "fi_abc"}],
    )
    assert "example.com" in scope["domains"]


@pytest.mark.asyncio
async def test_scope_extraction_caps_scope_size():
    """A single mute can't shadow many domains/keywords."""
    from augmentum.proxy.companion_routes import _extract_mute_scope
    backend = await _boot_backend()
    # Plant 10 file_index rows with distinct domains
    for i in range(10):
        await backend.conn.execute(
            "INSERT INTO file_index "
            "(id, user_id, source, source_id, name, mime_type, source_metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"fi_{i}", "usr_t", "browse", f"src_{i}", "x.html", "text/html",
             f'{{"source_url": "https://domain{i}.com/a"}}'),
        )
    await backend.conn.commit()
    refs = [{"kind": "file", "id": f"fi_{i}"} for i in range(10)]
    scope = await _extract_mute_scope(
        backend, user_id="usr_t",
        content=" ".join(f"word{i}" for i in range(20)),
        refs=refs,
    )
    # Capped at 5 domains, 3 keywords
    assert len(scope["domains"]) <= 5
    assert len(scope["keywords"]) <= 3


# ── Topic mute persistence + wondering generator interaction ─────────


@pytest.mark.asyncio
async def test_topic_mute_persisted_with_expiry():
    """Writing a mute row sets expires_at in the future."""
    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO companion_topic_mutes "
        "(user_id, companion_id, scope_json, expires_at) "
        "VALUES ('usr_t', 'becca', ?, datetime('now', '+90 days'))",
        (json.dumps({"domains": ["example.com"], "keywords": []}),),
    )
    await backend.conn.commit()

    cur = await backend.conn.execute(
        "SELECT expires_at > datetime('now') FROM companion_topic_mutes "
        "WHERE user_id = 'usr_t'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == 1  # SQLite truthy


# ── Pre-context injection ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pre_context_disabled_returns_none(monkeypatch):
    """Master kill switch off → no injection."""
    from augmentum.companion_runtime.pre_context import maybe_inject_notes_context
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_pre_context_enabled", False)
    backend = await _boot_backend()

    # Build a minimal runtime stub
    class _RT:
        def __init__(self):
            self.backend = backend
            self.companion_id = "becca"

    result = await maybe_inject_notes_context(
        _RT(), user_id="usr_t", first_message="hello world",
    )
    assert result is None


@pytest.mark.asyncio
async def test_pre_context_no_match_returns_none(monkeypatch):
    """No keyword overlap → no injection."""
    from augmentum.companion_runtime.pre_context import maybe_inject_notes_context
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_pre_context_enabled", True)
    backend = await _boot_backend()
    # Insert a ready note about something unrelated
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, content_refs, quiet_share_ready) "
        "VALUES ('becca', 'usr_t', 'noticing', "
        "'A thought about quantum cryptography', '[]', 1)",
    )
    await backend.conn.commit()

    class _RT:
        def __init__(self):
            self.backend = backend
            self.companion_id = "becca"

    result = await maybe_inject_notes_context(
        _RT(), user_id="usr_t",
        first_message="what should I cook for dinner tonight",
    )
    assert result is None


@pytest.mark.asyncio
async def test_pre_context_keyword_overlap_triggers_injection(monkeypatch):
    """≥ min_keyword_overlap shared between note + message → inject."""
    from augmentum.companion_runtime.pre_context import maybe_inject_notes_context
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_pre_context_enabled", True)
    monkeypatch.setattr(settings, "companion_pre_context_min_keyword_overlap", 2)
    # Sprint 5 — pre-context is engaged-only; set presence_mode accordingly
    monkeypatch.setattr(settings, "companion_presence_mode", "engaged")

    backend = await _boot_backend()
    # Note with strong keyword signal
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, content_refs, "
        " quiet_share_ready) "
        "VALUES ('becca', 'usr_t', 'noticing', "
        "'Prefix caching connects to KV restoration work from April.', "
        "'[]', 1)",
    )
    await backend.conn.commit()

    class _RT:
        def __init__(self):
            self.backend = backend
            self.companion_id = "becca"

    result = await maybe_inject_notes_context(
        _RT(), user_id="usr_t",
        first_message="tell me about prefix caching and KV cache stability",
    )
    assert result is not None
    assert "Prefix caching" in result
    assert "becca's note" in result.lower()  # the wrap marker


@pytest.mark.asyncio
async def test_pre_context_skips_surfaced_notes(monkeypatch):
    """Already-surfaced notes are excluded."""
    from augmentum.companion_runtime.pre_context import maybe_inject_notes_context
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_pre_context_enabled", True)
    backend = await _boot_backend()
    # Note already surfaced — should not inject
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, content_refs, "
        " quiet_share_ready, surfaced_at) "
        "VALUES ('becca', 'usr_t', 'noticing', "
        "'Already-seen note about prefix caching and KV work.', "
        "'[]', 1, datetime('now'))",
    )
    await backend.conn.commit()

    class _RT:
        def __init__(self):
            self.backend = backend
            self.companion_id = "becca"

    result = await maybe_inject_notes_context(
        _RT(), user_id="usr_t",
        first_message="tell me about prefix caching",
    )
    assert result is None


@pytest.mark.asyncio
async def test_pre_context_skips_quarantined_notes(monkeypatch):
    """Quarantined notes are excluded even if keyword-matched."""
    from augmentum.companion_runtime.pre_context import maybe_inject_notes_context
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_pre_context_enabled", True)
    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, content_refs, "
        " quiet_share_ready, quarantined, quarantine_reason) "
        "VALUES ('becca', 'usr_t', 'noticing', "
        "'A flagged note about prefix caching and KV work.', "
        "'[]', 1, 1, 'adversarial_pattern')",
    )
    await backend.conn.commit()

    class _RT:
        def __init__(self):
            self.backend = backend
            self.companion_id = "becca"

    result = await maybe_inject_notes_context(
        _RT(), user_id="usr_t",
        first_message="tell me about prefix caching and KV work",
    )
    assert result is None


@pytest.mark.asyncio
async def test_pre_context_empty_message_returns_none(monkeypatch):
    """Empty first message → trivially no match."""
    from augmentum.companion_runtime.pre_context import maybe_inject_notes_context
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_pre_context_enabled", True)
    backend = await _boot_backend()

    class _RT:
        def __init__(self):
            self.backend = backend
            self.companion_id = "becca"

    result = await maybe_inject_notes_context(
        _RT(), user_id="usr_t", first_message="",
    )
    assert result is None


def test_extract_message_keywords_filters_stopwords():
    from augmentum.companion_runtime.pre_context import _extract_message_keywords
    keywords = _extract_message_keywords(
        "what is the prefix caching strategy in modern transformers"
    )
    assert "what" not in keywords  # stopword
    assert "the" not in keywords
    # Real content words preserved
    assert "prefix" in keywords or "caching" in keywords or "transformers" in keywords
