"""Direct route tests for queued media-server syncs."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeRequest:
    def __init__(self, *, user_id: str, jobs_store, job_runner, http_client=object()):
        self.scope = {"user": SimpleNamespace(id=user_id)}
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                jobs_store=jobs_store,
                job_runner=job_runner,
                http_client=http_client,
            ),
        )


def test_sync_route_reuses_existing_media_sync_job_for_same_server():
    async def go():
        from augmentum.proxy.media_routes import sync_server_route

        store = MagicMock()
        store.get = AsyncMock(return_value=SimpleNamespace(id="ms_abs"))
        store.update = AsyncMock()

        jobs_store = MagicMock()
        jobs_store.list_for_user = AsyncMock(return_value=[{
            "id": "job_existing",
            "status": "running",
            "payload": {"server_id": "ms_abs"},
            "stage": "Fetching catalog",
        }])
        jobs_store.create = AsyncMock()
        job_runner = MagicMock()

        req = _FakeRequest(
            user_id="u_abs",
            jobs_store=jobs_store,
            job_runner=job_runner,
        )

        with patch("augmentum.proxy.media_routes._get_store", return_value=store):
            resp = await sync_server_route("ms_abs", req)

        body = json.loads(resp.body)
        assert resp.status_code == 202
        assert body == {
            "status": "queued",
            "job_id": "job_existing",
            "server_id": "ms_abs",
        }
        store.update.assert_awaited_once_with(
            "ms_abs",
            user_id="u_abs",
            status="syncing",
            status_detail="Fetching catalog",
        )
        jobs_store.create.assert_not_called()
        job_runner.wake.assert_not_called()

    _run(go())


def test_sync_route_enqueues_background_job_and_marks_server_syncing():
    async def go():
        from augmentum.proxy.media_routes import sync_server_route

        store = MagicMock()
        store.get = AsyncMock(return_value=SimpleNamespace(id="ms_abs"))
        store.update = AsyncMock()

        jobs_store = MagicMock()
        jobs_store.list_for_user = AsyncMock(return_value=[])
        jobs_store.create = AsyncMock(return_value="job_new")
        job_runner = MagicMock()

        req = _FakeRequest(
            user_id="u_abs",
            jobs_store=jobs_store,
            job_runner=job_runner,
        )

        with patch("augmentum.proxy.media_routes._get_store", return_value=store):
            resp = await sync_server_route("ms_abs", req)

        body = json.loads(resp.body)
        assert resp.status_code == 202
        assert body == {
            "status": "queued",
            "job_id": "job_new",
            "server_id": "ms_abs",
        }
        store.update.assert_awaited_once_with(
            "ms_abs",
            user_id="u_abs",
            status="syncing",
            status_detail="Queued sync",
        )
        jobs_store.create.assert_awaited_once_with(
            user_id="u_abs",
            job_type="media_sync",
            payload={"server_id": "ms_abs"},
            priority=20,
            max_attempts=2,
        )
        job_runner.wake.assert_called_once_with()

    _run(go())
