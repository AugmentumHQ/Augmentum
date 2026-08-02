"""Tests for augmentum/proxy/coder_review_routes.py — HTTP flow for the
reviewable-turn panel.

These tests wire a FastAPI TestClient against a live ReviewRegistry,
a stub ContainerManager, and a TurnSnapshot pre-populated to mirror
"the agent just wrote X files". Coverage targets the bits route-level
logic owns that the registry tests can't reach:

* **Pending / one-shot fetch.** ``GET /pending`` lists bundles
  scoped to the current user; ``GET /<id>`` returns the single bundle
  with 404 / 403 semantics. Enabled:false shape when the registry
  isn't wired at all.
* **Accept.** Disk is already current; route stamps a git commit,
  resolves the bundle, returns commit hash + file list. No restore
  calls hit the container.
* **Reject.** Every touched path gets ``snapshot.restore`` called.
  Reversible paths come back to pre-turn state; non-reversible paths
  surface in ``failed_paths``. Bundle removed from registry.
* **Partial.** Body splits paths into accepted + rejected sets;
  rejected go through restore, accepted get committed; paths in
  neither default to accepted (safe default — the opposite would lose
  agent work silently).
* **Ownership.** Cross-tenant access is 403; unknown turn is 404.
* **Commit message.** Bundle's ``user_message`` becomes the commit
  message so ``git log`` reads as intent history.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from augmentum.coder.reviews import ReviewBundle, ReviewRegistry
from augmentum.coder.turn_snapshot import DiffEntry
from augmentum.proxy.coder_review_routes import router

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSnapshot:
    """Captures ``restore`` invocations for assertions. By default
    succeeds for every path; tests inject ``fail`` to simulate
    non-reversible paths."""

    def __init__(self, fail: set[str] | None = None):
        self.restore_calls: list[list[str]] = []
        self.fail = fail or set()

    async def restore(self, paths: list[str]) -> list[str]:
        self.restore_calls.append(list(paths))
        return [p for p in paths if p in self.fail]


class _FakeCM:
    """Records ``run_command`` calls for git-commit verification."""

    def __init__(self):
        self.commands: list[list[str]] = []
        self.git_output = "abc1234"  # fake short hash

    async def run_command(self, workspace_id: str, cmd: list[str], **_kw) -> str:
        self.commands.append(cmd)
        joined = cmd[-1] if cmd else ""
        if "rev-parse" in joined:
            return self.git_output
        return ""


def _make_app(
    *,
    registry: ReviewRegistry | None = None,
    cm: _FakeCM | None = None,
    user_id: str | None = None,
) -> FastAPI:
    """Build a test FastAPI with the review router + stub state.

    ``user_id=None`` means no auth middleware → no request.scope["user"]
    → single-tenant-dev path (registry returns everything). Pass a
    string to simulate a logged-in user.
    """
    app = FastAPI()
    app.state.review_registry = registry
    app.state.container_manager = cm
    app.include_router(router)

    if user_id is not None:
        @app.middleware("http")
        async def _inject_user(request, call_next):
            class _U:
                def __init__(self, uid): self.id = uid
            request.scope["user"] = _U(user_id)
            return await call_next(request)

    return app


def _make_bundle(
    *,
    turn_id: str = "t-1",
    user_id: str = "",
    workspace_id: str = "ws-1",
    paths: list[tuple[str, str]] | None = None,
    user_message: str = "Add foo",
    non_reversible: set[str] | None = None,
) -> tuple[ReviewBundle, _FakeSnapshot]:
    """Build a bundle + its fake snapshot. ``paths`` is a list of
    (path, status) tuples; defaults to one added file."""
    if paths is None:
        paths = [("/workspace/foo.py", "added")]
    nr = non_reversible or set()
    files = [
        DiffEntry(
            path=p, status=s, unified_diff=f"@@ {p} @@",
            old_size=0, new_size=10, reversible=(p not in nr),
        )
        for p, s in paths
    ]
    snap = _FakeSnapshot(fail=nr)
    bundle = ReviewBundle(
        turn_id=turn_id,
        user_id=user_id,
        workspace_id=workspace_id,
        session_id="sess-1",
        user_message=user_message,
        files=files,
        snapshot=snap,
    )
    return bundle, snap


# ---------------------------------------------------------------------------
# GET /pending
# ---------------------------------------------------------------------------


def test_pending_returns_enabled_false_when_no_registry():
    app = _make_app(registry=None)
    with TestClient(app) as client:
        r = client.get("/api/coder/reviews/pending")
    assert r.status_code == 200
    assert r.json() == {"enabled": False, "pending": []}


def test_pending_lists_bundles_scoped_to_user():
    reg = ReviewRegistry()
    b1, _ = _make_bundle(turn_id="t-a", user_id="alice")
    b2, _ = _make_bundle(turn_id="t-b", user_id="bob")
    b3, _ = _make_bundle(turn_id="t-c", user_id="alice")
    for b in (b1, b2, b3):
        reg.publish(b)

    app = _make_app(registry=reg, user_id="alice")
    with TestClient(app) as client:
        r = client.get("/api/coder/reviews/pending")
    body = r.json()
    assert body["enabled"] is True
    ids = {p["turn_id"] for p in body["pending"]}
    assert ids == {"t-a", "t-c"}


# ---------------------------------------------------------------------------
# GET /{turn_id}
# ---------------------------------------------------------------------------


def test_get_single_bundle_returns_full_shape():
    reg = ReviewRegistry()
    bundle, _ = _make_bundle()
    reg.publish(bundle)

    app = _make_app(registry=reg)
    with TestClient(app) as client:
        r = client.get("/api/coder/reviews/t-1")
    assert r.status_code == 200
    body = r.json()
    assert body["turn_id"] == "t-1"
    assert body["summary"]["files_changed"] == 1
    assert body["summary"]["added"] == 1


def test_get_unknown_turn_404():
    app = _make_app(registry=ReviewRegistry())
    with TestClient(app) as client:
        r = client.get("/api/coder/reviews/does-not-exist")
    assert r.status_code == 404


def test_get_cross_tenant_403():
    reg = ReviewRegistry()
    b, _ = _make_bundle(user_id="alice")
    reg.publish(b)

    app = _make_app(registry=reg, user_id="bob")
    with TestClient(app) as client:
        r = client.get("/api/coder/reviews/t-1")
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# POST /{turn_id}/accept
# ---------------------------------------------------------------------------


def test_accept_stamps_git_commit_and_resolves():
    reg = ReviewRegistry()
    bundle, snap = _make_bundle(
        paths=[("/workspace/a.py", "added"), ("/workspace/b.py", "modified")],
        user_message="Add auth middleware",
    )
    reg.publish(bundle)
    cm = _FakeCM()

    app = _make_app(registry=reg, cm=cm)
    with TestClient(app) as client:
        r = client.post("/api/coder/reviews/t-1/accept")

    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "accepted"
    assert body["commit"] == "abc1234"
    assert set(body["files"]) == {"/workspace/a.py", "/workspace/b.py"}

    # Registry no longer pending.
    assert reg.get("t-1") is None
    # Snapshot.restore must NOT have been called (accept = leave disk alone).
    assert snap.restore_calls == []
    # Commit messages include the user_message.
    joined_cmds = " ".join(c[-1] for c in cm.commands if c)
    assert "Turn: Add auth middleware" in joined_cmds


def test_accept_without_container_manager_still_resolves():
    """Deployments without a container_manager (tests, degraded mode)
    should still accept cleanly — commit is skipped, bundle removed."""
    reg = ReviewRegistry()
    bundle, _ = _make_bundle()
    reg.publish(bundle)

    app = _make_app(registry=reg, cm=None)
    with TestClient(app) as client:
        r = client.post("/api/coder/reviews/t-1/accept")
    assert r.status_code == 200
    body = r.json()
    assert body["commit"] is None
    assert reg.get("t-1") is None


def test_accept_unknown_turn_404():
    app = _make_app(registry=ReviewRegistry(), cm=_FakeCM())
    with TestClient(app) as client:
        r = client.post("/api/coder/reviews/nope/accept")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /{turn_id}/reject
# ---------------------------------------------------------------------------


def test_reject_restores_every_touched_path():
    reg = ReviewRegistry()
    bundle, snap = _make_bundle(
        paths=[("/workspace/a.py", "added"), ("/workspace/b.py", "modified")],
    )
    reg.publish(bundle)

    app = _make_app(registry=reg, cm=_FakeCM())
    with TestClient(app) as client:
        r = client.post("/api/coder/reviews/t-1/reject")
    body = r.json()

    assert r.status_code == 200
    assert body["status"] == "rejected"
    assert set(body["restored_paths"]) == {"/workspace/a.py", "/workspace/b.py"}
    assert body["failed_paths"] == []
    assert snap.restore_calls == [["/workspace/a.py", "/workspace/b.py"]]
    assert reg.get("t-1") is None


def test_reject_surfaces_non_reversible_paths_as_failed():
    reg = ReviewRegistry()
    bundle, snap = _make_bundle(
        paths=[("/workspace/ok.py", "modified"), ("/workspace/bad.py", "added")],
        non_reversible={"/workspace/bad.py"},
    )
    reg.publish(bundle)

    app = _make_app(registry=reg, cm=_FakeCM())
    with TestClient(app) as client:
        r = client.post("/api/coder/reviews/t-1/reject")
    body = r.json()

    assert body["restored_paths"] == ["/workspace/ok.py"]
    assert body["failed_paths"] == ["/workspace/bad.py"]


# ---------------------------------------------------------------------------
# POST /{turn_id}/partial
# ---------------------------------------------------------------------------


def test_partial_accepts_named_and_restores_rest():
    reg = ReviewRegistry()
    bundle, snap = _make_bundle(
        paths=[
            ("/workspace/a.py", "added"),
            ("/workspace/b.py", "modified"),
            ("/workspace/c.py", "added"),
        ],
    )
    reg.publish(bundle)
    cm = _FakeCM()

    app = _make_app(registry=reg, cm=cm)
    with TestClient(app) as client:
        r = client.post(
            "/api/coder/reviews/t-1/partial",
            json={
                "accepted_paths": ["/workspace/a.py"],
                "rejected_paths": ["/workspace/b.py"],
                # c.py not mentioned → default accept
            },
        )
    body = r.json()

    assert r.status_code == 200
    assert body["status"] == "partial"
    # c.py fell into the default-accept bucket.
    assert set(body["accepted_paths"]) == {"/workspace/a.py", "/workspace/c.py"}
    assert body["rejected_paths"] == ["/workspace/b.py"]
    # restore invoked ONLY on rejected.
    assert snap.restore_calls == [["/workspace/b.py"]]
    # Commit covers accepted set.
    commit_stream = " ".join(c[-1] for c in cm.commands if c)
    assert "git add '/workspace/a.py'" in commit_stream
    assert "git add '/workspace/c.py'" in commit_stream
    assert "git add '/workspace/b.py'" not in commit_stream


def test_partial_ignores_paths_not_in_bundle():
    """Stale UI state (path rejected that isn't in the current bundle)
    must not trigger a restore or a commit."""
    reg = ReviewRegistry()
    bundle, snap = _make_bundle(
        paths=[("/workspace/a.py", "added")],
    )
    reg.publish(bundle)
    cm = _FakeCM()

    app = _make_app(registry=reg, cm=cm)
    with TestClient(app) as client:
        r = client.post(
            "/api/coder/reviews/t-1/partial",
            json={
                "accepted_paths": [],
                "rejected_paths": ["/workspace/does-not-exist.py"],
                # a.py unmentioned → default accept
            },
        )
    body = r.json()
    assert body["accepted_paths"] == ["/workspace/a.py"]
    assert body["rejected_paths"] == []
    # No restore calls (nothing to restore).
    assert snap.restore_calls == []


def test_partial_invalid_json_400():
    reg = ReviewRegistry()
    bundle, _ = _make_bundle()
    reg.publish(bundle)

    app = _make_app(registry=reg, cm=_FakeCM())
    with TestClient(app) as client:
        r = client.post(
            "/api/coder/reviews/t-1/partial",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 400
