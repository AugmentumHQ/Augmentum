"""Per-voice TTS pronunciation lexicon — store CRUD + application.

Migration 261 / augmentum/voice/lexicon_store.py. The apply() path runs
on every speech request, so correctness pins here: precedence (voice
beats '', longest term first), literal replacement (backslash phonetics
must not crash), boundaries/casing, user scoping, cache invalidation.
"""

from __future__ import annotations

import pytest


async def _fresh_backend(user_id: str = "u1"):
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


@pytest.mark.asyncio
async def test_crud_round_trip_and_upsert():
    from augmentum.voice import lexicon_store as ls
    backend = await _fresh_backend()

    e = await ls.add_entry(
        backend.conn, user_id="u1", voice="af_heart",
        term="SQL", phonetics="sequel",
    )
    assert e and e["voice"] == "af_heart" and e["phonetics"] == "sequel"

    # Upsert: same (user, voice, term) updates phonetics in place.
    e2 = await ls.add_entry(
        backend.conn, user_id="u1", voice="af_heart",
        term="SQL", phonetics="ess cue ell",
    )
    assert e2 and e2["id"] == e["id"] and e2["phonetics"] == "ess cue ell"

    entries = await ls.list_entries(backend.conn, user_id="u1")
    assert len(entries) == 1

    assert await ls.remove_entry(
        backend.conn, entry_id=e["id"], user_id="u1") is True
    assert await ls.list_entries(backend.conn, user_id="u1") == []


@pytest.mark.asyncio
async def test_validation_refuses_garbage():
    from augmentum.voice import lexicon_store as ls
    backend = await _fresh_backend()
    assert await ls.add_entry(
        backend.conn, user_id="u1", voice="", term="", phonetics="x") is None
    assert await ls.add_entry(
        backend.conn, user_id="", voice="", term="x", phonetics="y") is None
    assert await ls.add_entry(
        backend.conn, user_id="u1", voice="", term="t" * 81, phonetics="y",
    ) is None


@pytest.mark.asyncio
async def test_apply_voice_specific_beats_global():
    from augmentum.voice import lexicon_store as ls
    backend = await _fresh_backend()
    await ls.add_entry(backend.conn, user_id="u1", voice="",
                       term="SQL", phonetics="sequel")
    await ls.add_entry(backend.conn, user_id="u1", voice="af_heart",
                       term="SQL", phonetics="ess cue ell")

    out_specific = await ls.apply(
        backend.conn, "I love SQL.", user_id="u1", voice="af_heart")
    out_other = await ls.apply(
        backend.conn, "I love SQL.", user_id="u1", voice="bm_daniel")
    assert out_specific == "I love ess cue ell."
    assert out_other == "I love sequel."


@pytest.mark.asyncio
async def test_apply_longest_term_first_and_boundaries():
    from augmentum.voice import lexicon_store as ls
    backend = await _fresh_backend()
    await ls.add_entry(backend.conn, user_id="u1", voice="",
                       term="SQL", phonetics="sequel")
    await ls.add_entry(backend.conn, user_id="u1", voice="",
                       term="SQL Server", phonetics="sequel server")

    out = await ls.apply(
        backend.conn, "SQL Server runs SQL. MySQL stays.",
        user_id="u1", voice="af_heart")
    # Longest first: "SQL Server" intact; boundary: MySQL untouched.
    assert out == "sequel server runs sequel. MySQL stays."


@pytest.mark.asyncio
async def test_apply_literal_replacement_no_template_crash():
    """Backslashes in phonetics must not be parsed as a sub() template
    — the crash class found in the settings-based lexicon review."""
    from augmentum.voice import lexicon_store as ls
    backend = await _fresh_backend()
    await ls.add_entry(backend.conn, user_id="u1", voice="",
                       term="pathvar", phonetics=r"back\slash \1 \g<0>")
    out = await ls.apply(
        backend.conn, "say pathvar now", user_id="u1", voice="x")
    assert out == r"say back\slash \1 \g<0> now"


@pytest.mark.asyncio
async def test_apply_case_insensitive_and_provider_prefix():
    from augmentum.voice import lexicon_store as ls
    backend = await _fresh_backend()
    await ls.add_entry(backend.conn, user_id="u1", voice="af_heart",
                       term="kubectl", phonetics="kube control")
    out = await ls.apply(
        backend.conn, "Run KUBECTL apply.",
        user_id="u1", voice="kokoro::af_heart",  # prefix normalized away
    )
    assert out == "Run kube control apply."


@pytest.mark.asyncio
async def test_apply_user_scoped_and_failsafe():
    from augmentum.voice import lexicon_store as ls
    backend = await _fresh_backend()
    await ls.add_entry(backend.conn, user_id="u1", voice="",
                       term="SQL", phonetics="sequel")
    # Another user's text is untouched.
    out = await ls.apply(
        backend.conn, "SQL here", user_id="someone_else", voice="x")
    assert out == "SQL here"
    # No conn / no user → text passes through unchanged, never raises.
    assert await ls.apply(None, "SQL", user_id="u1", voice="x") == "SQL"
    assert await ls.apply(backend.conn, "SQL", user_id="", voice="x") == "SQL"


@pytest.mark.asyncio
async def test_cache_invalidates_on_write():
    from augmentum.voice import lexicon_store as ls
    backend = await _fresh_backend()
    e = await ls.add_entry(backend.conn, user_id="u1", voice="",
                           term="SQL", phonetics="sequel")
    assert (await ls.apply(
        backend.conn, "SQL", user_id="u1", voice="v")) == "sequel"
    # Warm cache, then delete — next apply must see the removal.
    await ls.remove_entry(backend.conn, entry_id=e["id"], user_id="u1")
    assert (await ls.apply(
        backend.conn, "SQL", user_id="u1", voice="v")) == "SQL"


@pytest.mark.asyncio
async def test_empty_phonetics_rows_skipped_in_v1():
    from augmentum.voice import lexicon_store as ls
    backend = await _fresh_backend()
    await ls.add_entry(backend.conn, user_id="u1", voice="",
                       term="mm", phonetics="")
    out = await ls.apply(backend.conn, "mm, nice", user_id="u1", voice="v")
    assert out == "mm, nice"  # stored (shield reserved), not applied
