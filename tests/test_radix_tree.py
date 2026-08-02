"""Tests for the radix tree prefix cache (Phase 3)."""
from __future__ import annotations

import sys
from pathlib import Path

# Add engine source to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "engine"))

from radix_tree import RadixTree


def test_insert_and_match_single():
    """Single prefix insert and exact match."""
    tree = RadixTree()
    tree.insert([1, 2, 3, 4, 5], "pfx_a")

    length, pid, node = tree.match_prefix([1, 2, 3, 4, 5])
    assert length == 5
    assert pid == "pfx_a"
    assert node is not None


def test_match_prefix_subset():
    """Query longer than cached prefix should match the prefix portion."""
    tree = RadixTree()
    tree.insert([1, 2, 3], "pfx_sys")

    length, pid, _ = tree.match_prefix([1, 2, 3, 4, 5, 6])
    assert length == 3
    assert pid == "pfx_sys"


def test_no_match():
    """Query that doesn't match any prefix."""
    tree = RadixTree()
    tree.insert([1, 2, 3], "pfx_a")

    length, pid, _ = tree.match_prefix([9, 8, 7])
    assert length == 0
    assert pid == ""


def test_shared_prefix():
    """Two prefixes sharing a common start."""
    tree = RadixTree()
    tree.insert([1, 2, 3, 4, 5], "pfx_a")
    tree.insert([1, 2, 3, 6, 7], "pfx_b")

    # Query matching pfx_a
    length, pid, _ = tree.match_prefix([1, 2, 3, 4, 5, 8, 9])
    assert length == 5
    assert pid == "pfx_a"

    # Query matching pfx_b
    length, pid, _ = tree.match_prefix([1, 2, 3, 6, 7, 8, 9])
    assert length == 5
    assert pid == "pfx_b"

    # Query matching only shared prefix (no prefix node at [1,2,3])
    length, pid, _ = tree.match_prefix([1, 2, 3, 99])
    assert length == 0  # no prefix registered at the [1,2,3] split point


def test_nested_prefixes():
    """Prefix that is a subset of another prefix."""
    tree = RadixTree()
    tree.insert([1, 2, 3], "pfx_short")
    tree.insert([1, 2, 3, 4, 5], "pfx_long")

    # Query exactly matching short prefix
    length, pid, _ = tree.match_prefix([1, 2, 3, 99])
    assert length == 3
    assert pid == "pfx_short"

    # Query matching long prefix
    length, pid, _ = tree.match_prefix([1, 2, 3, 4, 5, 99])
    assert length == 5
    assert pid == "pfx_long"

    # Query matching only short
    length, pid, _ = tree.match_prefix([1, 2, 3])
    assert length == 3
    assert pid == "pfx_short"


def test_remove():
    """Remove a prefix and verify it no longer matches."""
    tree = RadixTree()
    tree.insert([1, 2, 3], "pfx_a")

    tokens = tree.remove("pfx_a")
    assert tokens == 3

    length, pid, _ = tree.match_prefix([1, 2, 3])
    assert length == 0
    assert pid == ""


def test_remove_preserves_siblings():
    """Removing one branch doesn't affect sibling branches."""
    tree = RadixTree()
    tree.insert([1, 2, 3, 4], "pfx_a")
    tree.insert([1, 2, 5, 6], "pfx_b")

    tree.remove("pfx_a")

    # pfx_b should still match
    length, pid, _ = tree.match_prefix([1, 2, 5, 6, 7])
    assert length == 4
    assert pid == "pfx_b"


def test_evict_lru():
    """LRU eviction removes the oldest leaf."""
    tree = RadixTree()

    # Insert three prefixes
    node_a = tree.insert([1, 2], "pfx_a")
    node_a.last_access = 100.0

    node_b = tree.insert([3, 4], "pfx_b")
    node_b.last_access = 200.0

    node_c = tree.insert([5, 6], "pfx_c")
    node_c.last_access = 300.0

    # Evict — should remove pfx_a (oldest)
    evicted_id, evicted_tokens = tree.evict_lru()
    assert evicted_id == "pfx_a"
    assert evicted_tokens == 2

    # pfx_b and pfx_c should remain
    assert tree.match_prefix([3, 4])[1] == "pfx_b"
    assert tree.match_prefix([5, 6])[1] == "pfx_c"


def test_evict_skips_refcounted():
    """Eviction skips prefixes with active references."""
    tree = RadixTree()

    node_a = tree.insert([1, 2], "pfx_a")
    node_a.last_access = 100.0
    node_a.refcount = 1  # in use

    node_b = tree.insert([3, 4], "pfx_b")
    node_b.last_access = 200.0

    # Should evict pfx_b (pfx_a is protected by refcount)
    evicted_id, _ = tree.evict_lru()
    assert evicted_id == "pfx_b"


def test_refcount():
    """Inc/dec refcount operations."""
    tree = RadixTree()
    tree.insert([1, 2, 3], "pfx_a")

    assert tree.inc_ref("pfx_a")
    assert tree.inc_ref("pfx_a")

    node = tree._prefix_map["pfx_a"]
    assert node.refcount == 2

    assert tree.dec_ref("pfx_a")
    assert node.refcount == 1

    # Non-existent prefix
    assert not tree.inc_ref("pfx_nonexistent")


def test_stats():
    """Stats reflect current tree state."""
    tree = RadixTree()
    tree.insert([1, 2, 3], "pfx_a")
    tree.insert([4, 5, 6, 7], "pfx_b")

    stats = tree.stats()
    assert stats["prefix_count"] == 2
    assert stats["total_cached_tokens"] == 7  # 3 + 4
    assert len(stats["prefixes"]) == 2


def test_visualize():
    """Visualization produces readable output."""
    tree = RadixTree()
    tree.insert([1, 2, 3, 4, 5], "pfx_sys")
    tree.insert([1, 2, 3, 6, 7], "pfx_alt")

    viz = tree.visualize()
    assert "Root" in viz
    assert "pfx_sys" in viz
    assert "pfx_alt" in viz


def test_edge_split():
    """Inserting a sequence that partially matches an existing edge splits it."""
    tree = RadixTree()
    tree.insert([1, 2, 3, 4, 5], "pfx_long")
    tree.insert([1, 2, 3], "pfx_short")

    # Both should be findable
    length, pid, _ = tree.match_prefix([1, 2, 3])
    assert pid == "pfx_short"
    assert length == 3

    length, pid, _ = tree.match_prefix([1, 2, 3, 4, 5])
    assert pid == "pfx_long"
    assert length == 5


def test_app_builder_scenario():
    """Simulate the app builder pipeline: shared prefix, multiple suffix calls."""
    tree = RadixTree()

    # System prompt + working document (shared across all pipeline calls)
    sys_tokens = list(range(1, 501))  # 500 tokens
    tree.insert(sys_tokens, "pfx_build_sys")

    # Pipeline call 1: generate index.html
    call1_tokens = sys_tokens + [1001, 1002, 1003]
    length, pid, _ = tree.match_prefix(call1_tokens)
    assert length == 500
    assert pid == "pfx_build_sys"

    # Pipeline call 2: generate styles.css
    call2_tokens = sys_tokens + [2001, 2002, 2003]
    length, pid, _ = tree.match_prefix(call2_tokens)
    assert length == 500
    assert pid == "pfx_build_sys"

    # All 10 calls would share the same 500-token prefix
    for i in range(10):
        call_tokens = sys_tokens + [3000 + i]
        length, pid, _ = tree.match_prefix(call_tokens)
        assert length == 500
        assert pid == "pfx_build_sys"


def test_narrative_branch_scenario():
    """Simulate narrative branching: shared conversation history, divergent branches."""
    tree = RadixTree()

    # Shared conversation history
    history = list(range(1, 201))  # 200 tokens
    tree.insert(history, "pfx_history")

    # Branch A: user chose door 1
    branch_a = history + [9001, 9002, 9003]
    tree.insert(branch_a, "pfx_branch_a")

    # Branch B: user chose door 2
    branch_b = history + [9101, 9102, 9103]
    tree.insert(branch_b, "pfx_branch_b")

    # Switching to branch A matches 203 tokens
    length, pid, _ = tree.match_prefix(branch_a + [8888])
    assert length == 203
    assert pid == "pfx_branch_a"

    # Switching to branch B matches 203 tokens
    length, pid, _ = tree.match_prefix(branch_b + [7777])
    assert length == 203
    assert pid == "pfx_branch_b"

    # New message on base history only matches 200
    new_msg = history + [5555]
    length, pid, _ = tree.match_prefix(new_msg)
    assert length == 200
    assert pid == "pfx_history"


def test_empty_tree():
    """Operations on empty tree don't crash."""
    tree = RadixTree()

    length, pid, _ = tree.match_prefix([1, 2, 3])
    assert length == 0
    assert pid == ""

    evicted_id, evicted_tokens = tree.evict_lru()
    assert evicted_id == ""
    assert evicted_tokens == 0

    assert tree.remove("nonexistent") == 0
    assert tree.stats()["prefix_count"] == 0


def test_empty_token_list():
    """Empty token lists handled gracefully."""
    tree = RadixTree()
    tree.insert([1, 2, 3], "pfx_a")

    length, pid, _ = tree.match_prefix([])
    assert length == 0
    assert pid == ""
