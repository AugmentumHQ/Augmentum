"""Auth session eviction ordering.

When ``auth_max_sessions_per_user`` is reached and a new login lands,
the old eviction logic picked the lowest-``created_at`` rows — which
quietly killed actively-used browser cookies just because they were
older than today's noisy login churn. The fix orders eviction by
``last_activity`` (most-recently-used wins) with ``created_at`` as
the tiebreaker so the new login itself can't be the first to go.
"""

from __future__ import annotations

import asyncio

import pytest

from augmentum.auth.session_manager import SessionManager, _hash_token
from augmentum.config import settings
from augmentum.state.backends.sqlite import SQLiteBackend


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def sm():
    backend = SQLiteBackend(":memory:")
    _run(backend.connect())
    manager = SessionManager(backend._conn)
    yield manager
    _run(backend.close())


@pytest.fixture
def user(sm):
    return _run(sm.create_user("alice", "supersecret"))


def _all_tokens(sm):
    async def _q():
        cur = await sm._db.execute(
            "SELECT token, last_activity, created_at FROM auth_sessions "
            "ORDER BY last_activity DESC, created_at DESC"
        )
        return await cur.fetchall()

    return _run(_q())


def test_eviction_keeps_recently_active_session(sm, user):
    """The session you just used must survive a flurry of new logins."""

    cap = settings.auth_max_sessions_per_user

    # The first session we create is the one we will mark as "actively
    # used right now" — simulating the browser cookie the user is
    # actually holding.
    active_token = _run(sm.create_session(user.id))
    active_hash = _hash_token(active_token)

    # Fill the table up to the cap with older sessions (created_at the
    # default, last_activity bumped to "far in the past" so they are
    # genuinely stale).
    for _ in range(cap - 1):
        _run(sm.create_session(user.id))

    async def _backdate_all_except_active():
        # Backdate everyone EXCEPT the active token, so the next
        # create_session triggers eviction with the active token having
        # the freshest last_activity.
        await sm._db.execute(
            """UPDATE auth_sessions
                  SET last_activity = datetime('now', '-1 day')
                WHERE token != ?""",
            (active_hash,),
        )
        await sm._db.commit()

    _run(_backdate_all_except_active())

    # New login — this pushes us 1 over the cap and triggers eviction.
    _run(sm.create_session(user.id))

    rows = _all_tokens(sm)
    assert len(rows) == cap
    surviving_tokens = {r[0] for r in rows}
    assert active_hash in surviving_tokens, (
        "actively-used session was evicted; the new-login churn killed "
        "the browser cookie the user is still holding"
    )


def test_new_session_itself_is_never_immediately_evicted(sm, user):
    """A brand-new session can never be its own eviction victim."""

    cap = settings.auth_max_sessions_per_user
    # Fill with cap stale sessions — all created in the past, all idle.
    for _ in range(cap):
        _run(sm.create_session(user.id))

    async def _backdate_all():
        await sm._db.execute(
            "UPDATE auth_sessions SET last_activity = datetime('now', '-1 day')"
        )
        await sm._db.commit()

    _run(_backdate_all())

    fresh_token = _run(sm.create_session(user.id))
    fresh_hash = _hash_token(fresh_token)

    rows = _all_tokens(sm)
    assert len(rows) == cap
    surviving_tokens = {r[0] for r in rows}
    assert fresh_hash in surviving_tokens, (
        "the new login was evicted on its own creation pass — "
        "created_at tiebreaker isn't sorting newest-first"
    )


def test_eviction_falls_back_to_created_at_when_activity_is_tied(sm, user):
    """When last_activity is identical across rows (mass-imported, or
    no validate_token has bumped them), created_at orders the win.

    Within a single test the SQLite default ``datetime('now')`` has
    second precision and several sessions can share the same value;
    we explicitly stagger ``created_at`` here so the tiebreaker has
    something distinguishable to sort on, mirroring real-world usage
    where sessions are minutes/hours apart.
    """

    cap = settings.auth_max_sessions_per_user
    tokens = [_run(sm.create_session(user.id)) for _ in range(cap)]
    hashes = [_hash_token(t) for t in tokens]

    # Stagger created_at: tokens[0] is the oldest, tokens[cap-1] is the
    # most recent. Flatten last_activity so the tiebreaker is the only
    # signal.
    async def _stagger():
        for i, h in enumerate(hashes):
            await sm._db.execute(
                "UPDATE auth_sessions SET created_at = datetime('now', ?), "
                "last_activity = datetime('now', '-2 hours') WHERE token = ?",
                (f"-{(cap - i) * 10} minutes", h),
            )
        await sm._db.commit()

    _run(_stagger())

    # New login — created_at is "now", so it wins the tiebreaker and
    # the oldest token in the staggered batch goes.
    fresh = _run(sm.create_session(user.id))
    fresh_hash = _hash_token(fresh)

    rows = _all_tokens(sm)
    assert len(rows) == cap
    surviving_tokens = {r[0] for r in rows}
    assert fresh_hash in surviving_tokens
    assert hashes[0] not in surviving_tokens, (
        "tied-activity tiebreaker did not order by created_at — "
        "oldest-created session should have been evicted"
    )


# ---------------------------------------------------------------------------
# Device-class sessions (cast_receiver / android) vs the browser LRU pool.
#
# An always-on TV is by definition the stalest session a user owns, so a
# shared pool meant browser/agent login churn silently evicted the TV's
# 1-year "home" credential and forced a QR re-pair (2026-07-01 incident).
# Device sources are pruned within their own source instead.
# ---------------------------------------------------------------------------


def _source_tokens(sm, source):
    async def _q():
        cur = await sm._db.execute(
            "SELECT token FROM auth_sessions WHERE source = ?", (source,)
        )
        return [r[0] for r in await cur.fetchall()]

    return _run(_q())


def test_device_session_survives_browser_churn(sm, user):
    """A stale cast-receiver session must outlive any amount of web login
    churn — that's the entire point of the device-source exemption."""

    cap = settings.auth_max_sessions_per_user
    tv_token = _run(sm.create_session(user.id, source="cast_receiver"))
    tv_hash = _hash_token(tv_token)

    # Make the TV the stalest row by far (it's been off for a month).
    async def _backdate_tv():
        await sm._db.execute(
            "UPDATE auth_sessions SET last_activity = datetime('now', '-30 days') "
            "WHERE token = ?",
            (tv_hash,),
        )
        await sm._db.commit()

    _run(_backdate_tv())

    # Flood the browser pool well past the cap.
    for _ in range(cap + 5):
        _run(sm.create_session(user.id, source="web"))

    assert tv_hash in _source_tokens(sm, "cast_receiver"), (
        "device-class session was evicted by web login churn — the "
        "browser LRU pool is counting/victimising device rows again"
    )
    # And the web pool itself is still capped: the device row must not
    # consume a browser slot.
    assert len(_source_tokens(sm, "web")) == cap


def test_device_sessions_capped_within_their_own_source(sm, user):
    """Device rows don't grow unboundedly — they LRU within the source."""

    cap = settings.auth_max_sessions_per_user
    for _ in range(cap + 3):
        _run(sm.create_session(user.id, source="cast_receiver"))

    assert len(_source_tokens(sm, "cast_receiver")) == cap


def test_repaired_device_replaces_its_own_previous_session(sm, user):
    """Two pairings of the SAME physical device keep exactly one row;
    a different device is untouched."""

    first = _run(sm.create_session(
        user.id, source="cast_receiver", source_device_id="tv-livingroom",
    ))
    other = _run(sm.create_session(
        user.id, source="cast_receiver", source_device_id="tv-bedroom",
    ))
    second = _run(sm.create_session(
        user.id, source="cast_receiver", source_device_id="tv-livingroom",
    ))

    remaining = set(_source_tokens(sm, "cast_receiver"))
    assert _hash_token(first) not in remaining, (
        "re-pairing the same device_id should replace its previous session"
    )
    assert _hash_token(second) in remaining
    assert _hash_token(other) in remaining, (
        "a different device's session must not be touched by the replacement"
    )


def test_device_replacement_requires_matching_source(sm, user):
    """source_device_id replacement is scoped to (user, source): an android
    row sharing a device_id string with a cast row must not be clobbered."""

    android = _run(sm.create_session(
        user.id, source="android", source_device_id="shared-id",
    ))
    _run(sm.create_session(
        user.id, source="cast_receiver", source_device_id="shared-id",
    ))

    assert _hash_token(android) in set(_source_tokens(sm, "android"))
