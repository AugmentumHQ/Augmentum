"""Unit tests for the Cardsmith in-memory session state manager.

Covers:
- get_or_create_session creates with expected defaults
- get_session returns the same instance for the right user_id
- get_session enforces user-isolation
- get_session updates the LRU position
- get_session returns None for expired sessions
- drop_session removes from registry
- TTL eviction on access
- LRU eviction beyond cap
- commit_field scalar overwrite vs array append
- commit_field JSON value coercion (objects, lists, plain strings)
- to_preview snapshots the public state
"""

from __future__ import annotations

import time

import pytest

from augmentum.modes.narrative.cardsmith import state


@pytest.fixture(autouse=True)
def _clear_registry():
    """Reset the global session registry between tests."""
    state._sessions.clear()
    yield
    state._sessions.clear()


def test_get_or_create_session_returns_session_with_id():
    s = state.get_or_create_session(
        user_id="u1", card_type="single", source="describe",
    )
    assert s.session_id.startswith("cs_")
    assert s.user_id == "u1"
    assert s.card_type == "single"
    assert s.source == "describe"
    assert s.fields == {}
    assert s.messages == []
    assert s.finalized is False


def test_get_or_create_session_seed_prompt_stored_in_meta():
    s = state.get_or_create_session(
        user_id="u1", card_type="single", source="describe",
        seed_prompt="cyberpunk medic",
    )
    assert s.meta.get("seed_prompt") == "cyberpunk medic"


def test_get_session_returns_same_instance_for_owner():
    s = state.get_or_create_session(
        user_id="u1", card_type="single", source="describe",
    )
    same = state.get_session(s.session_id, user_id="u1")
    assert same is s


def test_get_session_blocks_other_user_id():
    s = state.get_or_create_session(
        user_id="u1", card_type="single", source="describe",
    )
    other = state.get_session(s.session_id, user_id="u2")
    assert other is None


def test_get_session_returns_none_for_unknown_id():
    assert state.get_session("cs_does_not_exist", user_id="u1") is None


def test_get_session_promotes_to_mru():
    s1 = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s2 = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s3 = state.get_or_create_session(user_id="u1", card_type="single", source="describe")

    # s1 was created first; access it to push to MRU end
    state.get_session(s1.session_id, user_id="u1")

    # Order in registry should now be: s2, s3, s1 (LRU → MRU)
    keys = list(state._sessions.keys())
    assert keys == [s2.session_id, s3.session_id, s1.session_id]


def test_drop_session_removes_from_registry():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    state.drop_session(s.session_id)
    assert state.get_session(s.session_id, user_id="u1") is None


def test_drop_session_no_op_for_unknown_id():
    state.drop_session("cs_unknown")  # must not raise


def test_get_session_returns_none_after_ttl_expiry():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    # Push last_active_at far into the past
    s.last_active_at = time.time() - state._TTL_SECONDS - 60
    out = state.get_session(s.session_id, user_id="u1")
    assert out is None
    # Also evicted from registry
    assert s.session_id not in state._sessions


def test_lru_evicts_oldest_beyond_cap(monkeypatch):
    monkeypatch.setattr(state, "_MAX_SESSIONS", 3)
    a = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    b = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    c = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    d = state.get_or_create_session(user_id="u1", card_type="single", source="describe")

    # `a` should have been evicted (oldest)
    assert state.get_session(a.session_id, user_id="u1") is None
    assert state.get_session(b.session_id, user_id="u1") is b
    assert state.get_session(c.session_id, user_id="u1") is c
    assert state.get_session(d.session_id, user_id="u1") is d


# ── commit_field semantics ────────────────────────────────────────────────

def test_commit_field_scalar_latest_wins():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s.commit_field("name", "First")
    s.commit_field("name", "Second")
    assert s.fields["name"] == "Second"


def test_commit_field_array_path_appends():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s.commit_field("tags[]", "a")
    s.commit_field("tags[]", "b")
    s.commit_field("tags[]", "c")
    assert s.fields["tags"] == ["a", "b", "c"]


def test_commit_field_array_path_creates_list_lazily():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    assert "tags" not in s.fields
    s.commit_field("tags[]", "first")
    assert s.fields["tags"] == ["first"]


def test_commit_field_json_object_value_coerced_to_dict():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s.commit_field("lorebook[]", '{"keys": ["X"], "content": "Y"}')
    assert isinstance(s.fields["lorebook"][0], dict)
    assert s.fields["lorebook"][0]["keys"] == ["X"]


def test_commit_field_json_list_value_coerced_to_list():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s.commit_field("misc[]", '["a", "b", "c"]')
    # The parser would normally expand the list, but commit_field stores raw.
    # We store the parsed list as a single appended item.
    assert s.fields["misc"] == [["a", "b", "c"]]


def test_commit_field_invalid_json_falls_back_to_raw_string():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s.commit_field("description", "{ malformed json")
    # Should keep as string since JSON parse failed
    assert s.fields["description"] == "{ malformed json"


def test_commit_field_plain_string_kept_as_string():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s.commit_field("description", "Hello world")
    assert s.fields["description"] == "Hello world"


def test_commit_field_non_string_value_passed_through():
    """Direct callers (tests, hypothetical other consumers) may pass non-strings."""
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s.commit_field("depthPromptDepth", 5)
    assert s.fields["depthPromptDepth"] == 5


# ── append_user / append_assistant ────────────────────────────────────────

def test_append_user_recorded_in_messages():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s.append_user("Hi there")
    assert s.messages == [{"role": "user", "content": "Hi there"}]


def test_append_assistant_recorded_in_messages():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s.append_assistant("Reply")
    assert s.messages == [{"role": "assistant", "content": "Reply"}]


def test_append_assistant_empty_content_not_recorded():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s.append_assistant("")
    assert s.messages == []


def test_append_user_updates_last_active():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    before = s.last_active_at
    time.sleep(0.01)
    s.append_user("hi")
    assert s.last_active_at > before


# ── to_preview ────────────────────────────────────────────────────────────

def test_to_preview_snapshot_is_a_safe_copy():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s.commit_field("name", "Lyra")
    s.commit_field("tags[]", "cyberpunk")

    snap = s.to_preview()
    assert snap["session_id"] == s.session_id
    assert snap["card_type"] == "single"
    assert snap["fields"]["name"] == "Lyra"
    assert snap["fields"]["tags"] == ["cyberpunk"]

    # Mutating the snapshot must not affect the session
    snap["fields"]["tags"].append("modified")
    assert s.fields["tags"] == ["cyberpunk"]


def test_to_preview_includes_message_count_not_messages():
    s = state.get_or_create_session(user_id="u1", card_type="single", source="describe")
    s.append_user("hi")
    s.append_assistant("hello")
    snap = s.to_preview()
    assert snap["message_count"] == 2
    assert "messages" not in snap


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
