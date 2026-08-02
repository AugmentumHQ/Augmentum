"""Tests for dream context window retrieval and clustering."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from augmentum.dream.context import DreamContextBuilder
from augmentum.state.tree_utils import linearize_to_node


def _make_session_tree():
    """12-message linear conversation (system + 5 user-assistant pairs + 1 trailing)."""
    tree = {}
    tree["root"] = {"id": "root", "role": "system", "content": "sys", "children": ["m1"]}
    for i in range(1, 12):
        mid = f"m{i}"
        role = "user" if i % 2 == 1 else "assistant"
        tree[mid] = {
            "id": mid, "role": role,
            "content": f"Message {i} content",
            "children": [f"m{i+1}"] if i < 11 else [],
        }
    return tree


def test_extract_context_window_center():
    """Context window centered on m6 should get 6 messages."""
    tree = _make_session_tree()
    builder = DreamContextBuilder()
    path = linearize_to_node(tree, "root", "m6")
    window = builder.extract_window(path, target_id="m6", pairs=3)
    ids = [m["id"] for m in window]
    assert len(ids) == 6
    assert "m6" in ids


def test_extract_context_window_start_clamp():
    tree = _make_session_tree()
    builder = DreamContextBuilder()
    path = linearize_to_node(tree, "root", "m1")
    window = builder.extract_window(path, target_id="m1", pairs=3)
    assert len(window) <= 6
    assert window[0]["id"] in ("root", "m1")


def test_cluster_overlapping_memories():
    builder = DreamContextBuilder()
    memories = [
        {"id": "mem1", "source_message_id": "m3", "session_id": "s1"},
        {"id": "mem2", "source_message_id": "m5", "session_id": "s1"},
    ]
    clusters = builder.cluster_by_proximity(memories, window_size=3)
    assert len(clusters) == 1
    assert len(clusters[0]["memories"]) == 2


def test_separate_sessions_dont_cluster():
    builder = DreamContextBuilder()
    memories = [
        {"id": "mem1", "source_message_id": "m3", "session_id": "s1"},
        {"id": "mem2", "source_message_id": "m3", "session_id": "s2"},
    ]
    clusters = builder.cluster_by_proximity(memories, window_size=3)
    assert len(clusters) == 2


def test_max_window_cap():
    builder = DreamContextBuilder()
    memories = [
        {"id": f"mem{i}", "source_message_id": f"m{i}", "session_id": "s1"}
        for i in range(1, 10)
    ]
    clusters = builder.cluster_by_proximity(memories, window_size=3)
    for cluster in clusters:
        assert cluster.get("window_size", 0) <= 12


def test_humanize_age_just_now():
    builder = DreamContextBuilder()
    now = datetime.now(timezone.utc)
    assert builder.humanize_age(now.isoformat(), now) == "just now"


def test_humanize_age_days():
    builder = DreamContextBuilder()
    now = datetime.now(timezone.utc)
    three_days_ago = (now - timedelta(days=3)).isoformat()
    assert builder.humanize_age(three_days_ago, now) == "3 days ago"


def test_humanize_age_invalid():
    builder = DreamContextBuilder()
    assert builder.humanize_age("not-a-date") == "some time ago"
