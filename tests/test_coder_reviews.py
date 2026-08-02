"""Tests for augmentum/coder/reviews.py — the review registry.

The registry is pure bookkeeping (no async futures, no timeouts — see
the module docstring for why reviews are async in the user dimension,
unlike permission requests). These tests lock in the semantic pieces
that route handlers will depend on:

* **publish → pending_for → resolve** is the happy path. After
  resolve, the bundle is no longer pending but the resolved status is
  preserved on the returned bundle so the caller knows what happened.
* **Multi-tenant filtering.** ``pending_for(user_id)`` must scope
  strictly by owner — a leak here lets user A see user B's diffs.
* **to_dict shape.** The frontend depends on this exact structure;
  drift breaks rendering silently. Tests freeze the contract.
* **Workspace clear.** Workspace deletion must remove orphaned
  bundles whose snapshot container is about to disappear.
"""
from __future__ import annotations

from augmentum.coder.reviews import ReviewBundle, ReviewRegistry
from augmentum.coder.turn_snapshot import DiffEntry


class _FakeSnapshot:
    """Stand-in for TurnSnapshot. ReviewBundle doesn't exercise it in
    these tests — only serialization (which excludes it) and registry
    bookkeeping (which treats it opaquely)."""
    turn_id = "t-1"


def _make_bundle(
    *,
    turn_id: str = "t-1",
    user_id: str = "user-a",
    workspace_id: str = "ws-a",
    session_id: str = "sess-a",
    files: list[DiffEntry] | None = None,
    user_message: str = "Add auth",
) -> ReviewBundle:
    return ReviewBundle(
        turn_id=turn_id,
        user_id=user_id,
        workspace_id=workspace_id,
        session_id=session_id,
        user_message=user_message,
        files=files or [],
        snapshot=_FakeSnapshot(),
    )


# ---------------------------------------------------------------------------
# publish / get / resolve
# ---------------------------------------------------------------------------


def test_publish_and_get_roundtrip():
    reg = ReviewRegistry()
    bundle = _make_bundle()
    reg.publish(bundle)
    assert reg.get("t-1") is bundle
    assert reg.size() == 1


def test_resolve_removes_from_pending_and_marks_status():
    reg = ReviewRegistry()
    reg.publish(_make_bundle())
    out = reg.resolve("t-1", "accepted")
    assert out is not None
    assert out.status == "accepted"
    assert reg.get("t-1") is None
    assert reg.size() == 0


def test_resolve_unknown_returns_none():
    """Route handlers treat None as 404 — must not raise or corrupt state."""
    reg = ReviewRegistry()
    assert reg.resolve("never-existed", "accepted") is None


def test_resolve_twice_second_call_is_none():
    """After resolve, the bundle is gone. A duplicate resolve must
    not re-apply or leak state — the second call is a no-op that
    surfaces as None so the route returns 404."""
    reg = ReviewRegistry()
    reg.publish(_make_bundle())
    first = reg.resolve("t-1", "accepted")
    second = reg.resolve("t-1", "rejected")
    assert first is not None
    assert second is None


# ---------------------------------------------------------------------------
# Multi-tenant filtering
# ---------------------------------------------------------------------------


def test_pending_for_filters_by_user_id():
    reg = ReviewRegistry()
    reg.publish(_make_bundle(turn_id="a", user_id="u1"))
    reg.publish(_make_bundle(turn_id="b", user_id="u2"))
    reg.publish(_make_bundle(turn_id="c", user_id="u1"))

    u1 = {b.turn_id for b in reg.pending_for("u1")}
    u2 = {b.turn_id for b in reg.pending_for("u2")}
    assert u1 == {"a", "c"}
    assert u2 == {"b"}


def test_pending_for_empty_string_returns_everything():
    """Single-tenant dev convention inherited from PermissionRegistry
    — ``pending_for("")`` sees the whole registry so the dev UI
    doesn't need to know user IDs."""
    reg = ReviewRegistry()
    reg.publish(_make_bundle(turn_id="a", user_id="u1"))
    reg.publish(_make_bundle(turn_id="b", user_id="u2"))
    assert {b.turn_id for b in reg.pending_for("")} == {"a", "b"}


def test_pending_for_excludes_resolved_bundles():
    reg = ReviewRegistry()
    reg.publish(_make_bundle(turn_id="a", user_id="u1"))
    reg.publish(_make_bundle(turn_id="b", user_id="u1"))
    reg.resolve("a", "accepted")
    assert {b.turn_id for b in reg.pending_for("u1")} == {"b"}


# ---------------------------------------------------------------------------
# to_dict serialization shape
# ---------------------------------------------------------------------------


def test_to_dict_includes_expected_top_level_fields():
    bundle = _make_bundle(user_message="add middleware")
    d = bundle.to_dict()
    assert d["turn_id"] == "t-1"
    assert d["workspace_id"] == "ws-a"
    assert d["session_id"] == "sess-a"
    assert d["user_message"] == "add middleware"
    assert d["status"] == "pending"
    assert "created_at" in d
    assert d["files"] == []


def test_to_dict_files_preserve_diff_entry_fields():
    entry = DiffEntry(
        path="/workspace/foo.py",
        status="added",
        unified_diff="@@ ...",
        old_size=0,
        new_size=42,
        reversible=True,
    )
    bundle = _make_bundle(files=[entry])
    d = bundle.to_dict()
    assert len(d["files"]) == 1
    f = d["files"][0]
    assert f == {
        "path":         "/workspace/foo.py",
        "status":       "added",
        "unified_diff": "@@ ...",
        "old_size":     0,
        "new_size":     42,
        "reversible":   True,
    }


def test_to_dict_summary_counts():
    files = [
        DiffEntry("/a", "added",    "", 0, 10, True),
        DiffEntry("/b", "modified", "", 5, 8,  True),
        DiffEntry("/c", "modified", "", 3, 3,  False),
        DiffEntry("/d", "deleted",  "", 4, 0,  True),
    ]
    bundle = _make_bundle(files=files)
    s = bundle.to_dict()["summary"]
    assert s == {
        "files_changed":  4,
        "added":          1,
        "modified":       2,
        "deleted":        1,
        "non_reversible": 1,
    }


def test_to_dict_does_not_include_snapshot_reference():
    """Snapshot holds raw pre-turn bytes — must NEVER serialise to
    the frontend. It lives on the bundle only for the reject path."""
    bundle = _make_bundle()
    d = bundle.to_dict()
    assert "snapshot" not in d


# ---------------------------------------------------------------------------
# Workspace clear
# ---------------------------------------------------------------------------


def test_clear_for_workspace_drops_matching_bundles():
    reg = ReviewRegistry()
    reg.publish(_make_bundle(turn_id="a", workspace_id="ws-x"))
    reg.publish(_make_bundle(turn_id="b", workspace_id="ws-y"))
    reg.publish(_make_bundle(turn_id="c", workspace_id="ws-x"))

    dropped = reg.clear_for_workspace("ws-x")
    assert dropped == 2
    assert {b.turn_id for b in reg.pending_for("")} == {"b"}


def test_clear_for_workspace_no_match_is_zero():
    reg = ReviewRegistry()
    reg.publish(_make_bundle(workspace_id="ws-x"))
    assert reg.clear_for_workspace("ws-absent") == 0
    assert reg.size() == 1
