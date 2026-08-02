"""Behavior tests for the jobs REST endpoints (/api/jobs/*).

Focus areas:
* User isolation — a user must never see, fetch, or cancel another
  user's job. Isolation bugs here would leak background-task payloads
  (file paths, titles, prompts) across tenants.
* Cancel semantics — 200 on pending/running, 409 on terminal, 404 on
  missing-or-not-yours. The UI depends on this split to decide whether
  to surface "already done" vs. "not yours / deleted".
* Filter + limit behavior — clamp bounds, correct status/type narrowing,
  newest-first ordering.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


TEST_USER_ID = "usr_test"  # matches conftest.test_user


@pytest.fixture
def jobs_client(app):
    """Client with a real SQLite-backed JobsStore wired onto the app.

    The conftest ``app`` already provides a mock SessionManager that
    authenticates "Bearer test-token" as user_id=usr_test. We only need
    to plug in a real JobsStore so the routes can exercise actual CRUD.
    """
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.jobs_store import JobsStore
    from augmentum.state.manager import StateManager

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())
    app.state.state_manager = StateManager(backend)
    app.state.jobs_store = JobsStore(backend._conn)

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc, app.state.jobs_store
    _run(backend.close())


# ===========================================================================
# GET /api/jobs/
# ===========================================================================

class TestListJobs:
    def test_empty_for_fresh_user(self, jobs_client):
        client, _ = jobs_client
        r = client.get("/api/jobs/")
        assert r.status_code == 200
        assert r.json() == {"jobs": []}

    def test_returns_only_own_jobs(self, jobs_client):
        """Core isolation guarantee. A bug here leaks payloads across tenants."""
        client, store = jobs_client
        _run(store.create(user_id="usr_other", job_type="other_job", payload={"secret": "leak"}))
        my_id = _run(store.create(user_id=TEST_USER_ID, job_type="my_job"))

        r = client.get("/api/jobs/")
        jobs = r.json()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["id"] == my_id
        assert jobs[0]["job_type"] == "my_job"

    def test_newest_first(self, jobs_client):
        client, store = jobs_client
        _run(store.create(user_id=TEST_USER_ID, job_type="older"))
        time.sleep(1.1)  # created_at has 1s resolution in jobs_store
        _run(store.create(user_id=TEST_USER_ID, job_type="newer"))

        r = client.get("/api/jobs/")
        jobs = r.json()["jobs"]
        assert jobs[0]["job_type"] == "newer"
        assert jobs[1]["job_type"] == "older"

    def test_filter_by_status(self, jobs_client):
        client, store = jobs_client
        completed_id = _run(store.create(user_id=TEST_USER_ID, job_type="alpha"))
        _run(store.create(user_id=TEST_USER_ID, job_type="beta"))
        _run(store.mark_completed(completed_id, result=None))

        r = client.get("/api/jobs/?status=completed")
        jobs = r.json()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["id"] == completed_id

    def test_filter_by_type(self, jobs_client):
        client, store = jobs_client
        _run(store.create(user_id=TEST_USER_ID, job_type="gutenberg_fetch"))
        _run(store.create(user_id=TEST_USER_ID, job_type="media_sync"))

        r = client.get("/api/jobs/?type=gutenberg_fetch")
        jobs = r.json()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["job_type"] == "gutenberg_fetch"

    def test_limit_clamps_low(self, jobs_client):
        """limit=0 should be clamped to 1, not return everything / crash."""
        client, store = jobs_client
        for _ in range(3):
            _run(store.create(user_id=TEST_USER_ID, job_type="t"))
        r = client.get("/api/jobs/?limit=0")
        assert r.status_code == 200
        assert len(r.json()["jobs"]) == 1

    def test_limit_clamps_high(self, jobs_client):
        client, store = jobs_client
        for _ in range(5):
            _run(store.create(user_id=TEST_USER_ID, job_type="t"))
        # limit=9999 should clamp to 500 (the store max); we only have 5
        r = client.get("/api/jobs/?limit=9999")
        assert r.status_code == 200
        assert len(r.json()["jobs"]) == 5


# ===========================================================================
# GET /api/jobs/{job_id}
# ===========================================================================

class TestGetJob:
    def test_returns_own_job(self, jobs_client):
        client, store = jobs_client
        job_id = _run(store.create(user_id=TEST_USER_ID, job_type="test"))
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        assert r.json()["id"] == job_id
        assert r.json()["job_type"] == "test"
        assert r.json()["status"] == "pending"

    def test_returns_job_with_progress_fields(self, jobs_client):
        """Progress + stage must be surfaced — the UI polls these."""
        client, store = jobs_client
        job_id = _run(store.create(user_id=TEST_USER_ID, job_type="test"))
        _run(store.update_progress(job_id, progress=0.42, stage="downloading"))
        r = client.get(f"/api/jobs/{job_id}")
        data = r.json()
        assert data["progress"] == 0.42
        assert data["stage"] == "downloading"

    def test_missing_job_returns_404(self, jobs_client):
        client, _ = jobs_client
        r = client.get("/api/jobs/does-not-exist")
        assert r.status_code == 404

    def test_other_users_job_returns_404(self, jobs_client):
        """Isolation: user A must not be able to GET user B's job by guessing
        its ID. The 404 (not 403) also prevents ID-existence enumeration."""
        client, store = jobs_client
        other_id = _run(store.create(user_id="usr_other", job_type="secret_job"))
        r = client.get(f"/api/jobs/{other_id}")
        assert r.status_code == 404


# ===========================================================================
# POST /api/jobs/{job_id}/cancel
# ===========================================================================

class TestCancelJob:
    def test_cancels_pending_job(self, jobs_client):
        client, store = jobs_client
        job_id = _run(store.create(user_id=TEST_USER_ID, job_type="test"))
        r = client.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert _run(store.is_cancel_requested(job_id)) is True

    def test_cancel_completed_returns_409(self, jobs_client):
        client, store = jobs_client
        job_id = _run(store.create(user_id=TEST_USER_ID, job_type="test"))
        _run(store.mark_completed(job_id, result={"done": True}))
        r = client.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 409
        body = r.json()
        assert "terminal" in body["error"].lower()
        assert body["status"] == "completed"

    def test_cancel_failed_returns_409(self, jobs_client):
        client, store = jobs_client
        job_id = _run(store.create(user_id=TEST_USER_ID, job_type="test"))
        _run(store.mark_failed(job_id, error="boom", retryable=False))
        r = client.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 409
        assert r.json()["status"] == "failed"

    def test_cancel_already_cancelled_returns_409(self, jobs_client):
        client, store = jobs_client
        job_id = _run(store.create(user_id=TEST_USER_ID, job_type="test"))
        _run(store.mark_cancelled(job_id))
        r = client.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 409
        assert r.json()["status"] == "cancelled"

    def test_cancel_missing_returns_404(self, jobs_client):
        client, _ = jobs_client
        r = client.post("/api/jobs/does-not-exist/cancel")
        assert r.status_code == 404

    def test_cancel_other_users_job_returns_404(self, jobs_client):
        """User A cannot cancel user B's job — opacity matters so the attacker
        can't even confirm the job exists."""
        client, store = jobs_client
        other_id = _run(store.create(user_id="usr_other", job_type="secret"))
        r = client.post(f"/api/jobs/{other_id}/cancel")
        assert r.status_code == 404
        # And the job must remain in its original state on the other user's side
        assert _run(store.is_cancel_requested(other_id)) is False


# ===========================================================================
# Service-unavailable path
# ===========================================================================

class TestStoreUnavailable:
    def test_503_when_jobs_store_missing(self, app):
        """If jobs_store was never wired on app.state (e.g. boot error), the
        routes must return 503 rather than crash with AttributeError."""
        if hasattr(app.state, "jobs_store"):
            delattr(app.state, "jobs_store")
        tc = TestClient(app)
        tc.headers.update({"Authorization": "Bearer test-token"})

        r = tc.get("/api/jobs/")
        assert r.status_code == 503
        r = tc.get("/api/jobs/any-id")
        assert r.status_code == 503
        r = tc.post("/api/jobs/any-id/cancel")
        assert r.status_code == 503


# ===========================================================================
# Router sanity
# ===========================================================================

class TestRouterShape:
    def test_prefix(self):
        from augmentum.proxy.jobs_routes import router
        assert router.prefix == "/api/jobs"

    def test_registered_paths(self):
        from augmentum.proxy.jobs_routes import router
        paths = {r.path for r in router.routes}
        assert "/api/jobs/" in paths
        assert "/api/jobs/{job_id}" in paths
        assert "/api/jobs/{job_id}/cancel" in paths
