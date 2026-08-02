"""Action-handler registry — register/resolve/unregister/replace."""

from __future__ import annotations

import pytest

from augmentum.notifications.actions import (
    register_action_handler,
    registered_patterns,
    reset_registry,
    resolve_handler,
    unregister_action_handler,
)


async def _h_a(notification, action_id, request):  # type: ignore[no-untyped-def]
    return {"who": "a"}


async def _h_b(notification, action_id, request):  # type: ignore[no-untyped-def]
    return {"who": "b"}


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def test_register_and_resolve_exact_match() -> None:
    register_action_handler("connect.call.incoming", _h_a)
    handler = resolve_handler("connect.call.incoming")
    assert handler is _h_a


def test_glob_pattern_matches() -> None:
    register_action_handler("connect.call.*", _h_a)
    assert resolve_handler("connect.call.incoming") is _h_a
    assert resolve_handler("connect.call.missed") is _h_a
    # Non-matching:
    assert resolve_handler("connect.message") is None


def test_wildcard_pattern_matches_anything() -> None:
    register_action_handler("*", _h_a)
    assert resolve_handler("anything.at.all") is _h_a


def test_first_registered_pattern_wins() -> None:
    # Specific patterns registered before wildcards take precedence.
    register_action_handler("connect.call.*", _h_a)
    register_action_handler("*", _h_b)
    assert resolve_handler("connect.call.incoming") is _h_a
    assert resolve_handler("other.channel") is _h_b


def test_replace_pattern_preserves_order() -> None:
    # Re-registering the same pattern must replace in place — not
    # append a duplicate. Otherwise resolution order silently drifts.
    register_action_handler("connect.call.*", _h_a)
    register_action_handler("*", _h_b)
    register_action_handler("connect.call.*", _h_b)  # replace
    assert registered_patterns() == ["connect.call.*", "*"]
    assert resolve_handler("connect.call.incoming") is _h_b


def test_unregister_returns_presence() -> None:
    register_action_handler("connect.call.*", _h_a)
    assert unregister_action_handler("connect.call.*") is True
    assert unregister_action_handler("connect.call.*") is False
    assert resolve_handler("connect.call.incoming") is None


def test_unknown_channel_resolves_to_none() -> None:
    assert resolve_handler("totally.unknown") is None


def test_empty_pattern_rejected() -> None:
    with pytest.raises(ValueError, match="channel_pattern"):
        register_action_handler("", _h_a)
