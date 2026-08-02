"""compute_direct_prefix_cache_key — stable per-conversation KV-slot key for
direct-mode (external harness) requests, so a harness conversation keeps slot
affinity (save/restore beats re-prefill) under slot-0 contention."""

from __future__ import annotations

from augmentum.models.base import InternalChatRequest, Message
from augmentum.proxy.handler_factory import compute_direct_prefix_cache_key


def _req(system: str, first_user: str, tail=None, tools=None) -> InternalChatRequest:
    msgs = [Message(role="system", content=system), Message(role="user", content=first_user)]
    for role, content in (tail or []):
        msgs.append(Message(role=role, content=content))
    return InternalChatRequest(model="d/Qwen3.6-27B", messages=msgs, tools=tools)


def test_stable_across_turns():
    # Same system + first user message, growing tail → SAME key (affinity holds
    # turn-over-turn, which is the whole point).
    a = compute_direct_prefix_cache_key("u1", _req("SYS", "fix the bug"))
    b = compute_direct_prefix_cache_key(
        "u1", _req("SYS", "fix the bug", tail=[("assistant", "ok"), ("user", "now add a test")])
    )
    assert a == b
    assert a.startswith("u1:oc:")


def test_distinct_first_user_message():
    # Two conversations sharing a system prompt but different first tasks must
    # NOT collide onto one slot (the bare-system-hash collision class).
    a = compute_direct_prefix_cache_key("u1", _req("SYS", "task A"))
    b = compute_direct_prefix_cache_key("u1", _req("SYS", "task B"))
    assert a != b


def test_distinct_system():
    a = compute_direct_prefix_cache_key("u1", _req("SYS-A", "task"))
    b = compute_direct_prefix_cache_key("u1", _req("SYS-B", "task"))
    assert a != b


def test_user_namespaced():
    a = compute_direct_prefix_cache_key("u1", _req("SYS", "task"))
    b = compute_direct_prefix_cache_key("u2", _req("SYS", "task"))
    assert a != b
    assert a.startswith("u1:") and b.startswith("u2:")


def test_anon_returns_empty():
    assert compute_direct_prefix_cache_key("", _req("SYS", "task")) == ""


def test_tools_fold_in():
    base = compute_direct_prefix_cache_key("u1", _req("SYS", "task"))
    withtools = compute_direct_prefix_cache_key(
        "u1", _req("SYS", "task", tools=[{"type": "function", "function": {"name": "read_file"}}])
    )
    assert base != withtools
