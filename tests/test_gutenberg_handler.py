"""``gutenberg_fetch`` job handler — end-to-end with mocked network + fake file_index.

Covers the four outcomes the pin flow cares about:
  1. Happy path — downloaded, stripped, stored, metadata updated.
  2. Idempotency — second run on a 'fetched' row is a no-op.
  3. Unavailable — non-Gutenberg URL marks the row as permanently skipped.
  4. Transient failure — all HTTP URLs 404 exhausts retries and fails.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import aiosqlite
import httpx
import pytest

from augmentum.jobs import JobRunner, register_handler
from augmentum.jobs.handlers.gutenberg_fetch import make_gutenberg_fetch_handler
from augmentum.state.jobs_store import JobsStore


_SCHEMA_SQL = """
CREATE TABLE users (id TEXT PRIMARY KEY);
CREATE TABLE background_jobs (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users(id),
    job_type         TEXT NOT NULL,
    payload          TEXT NOT NULL DEFAULT '{}',
    status           TEXT NOT NULL DEFAULT 'pending',
    progress         REAL NOT NULL DEFAULT 0.0,
    stage            TEXT NOT NULL DEFAULT '',
    result           TEXT,
    error            TEXT,
    priority         INTEGER NOT NULL DEFAULT 0,
    attempts         INTEGER NOT NULL DEFAULT 0,
    max_attempts     INTEGER NOT NULL DEFAULT 3,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at       INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    started_at       INTEGER,
    updated_at       INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    completed_at     INTEGER
);
"""


_SAMPLE_BOOK = """The Project Gutenberg eBook of Foo

*** START OF THE PROJECT GUTENBERG EBOOK FOO ***

Body of the book — many many words to reach a plausible word count.
This line adds more words. And another. One more for good measure.

*** END OF THE PROJECT GUTENBERG EBOOK FOO ***

Licence footer
"""


@dataclass
class _FakeEntry:
    """Stand-in for ``FileEntry`` carrying only fields the handler reads."""

    id: str
    user_id: str
    source_metadata: dict[str, Any] = field(default_factory=dict)


class _FakeIndex:
    """In-memory file_index double with just the two methods the handler calls."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], _FakeEntry] = {}

    def seed(self, entry: _FakeEntry) -> None:
        self._rows[(entry.id, entry.user_id)] = entry

    async def get(self, file_id: str, *, user_id: str):
        return self._rows.get((file_id, user_id))

    async def update_source_metadata(
        self, file_id: str, metadata: dict, *, user_id: str,
    ) -> bool:
        key = (file_id, user_id)
        if key not in self._rows:
            return False
        self._rows[key].source_metadata = metadata
        return True


def _mock_http_client(text_by_url: dict[str, str]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        body = text_by_url.get(str(request.url))
        if body is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, text=body)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _mkstore() -> JobsStore:
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await conn.commit()
    return JobsStore(conn)


async def _wait_status(store: JobsStore, job_id: str, target: str, *, timeout: float = 5.0) -> dict:
    """Poll the store until the job hits ``target`` status. Returns the row."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        row = await store.get(job_id)
        if row and row["status"] == target:
            return row
        if row and row["status"] in {"cancelled"}:
            raise AssertionError(f"job went to cancelled, expected {target}")
        await asyncio.sleep(0.05)
    row = await store.get(job_id)
    raise AssertionError(
        f"job {job_id} did not reach {target} within {timeout}s "
        f"(final status={row and row['status']!r})",
    )


@pytest.mark.asyncio
async def test_happy_path_fetches_strips_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "augmentum.jobs.handlers.gutenberg_fetch.settings",
        SimpleNamespace(data_dir=str(tmp_path)),
    )
    idx = _FakeIndex()
    idx.seed(_FakeEntry(
        id="fi_abc", user_id="u1",
        source_metadata={"url_text_source": "https://www.gutenberg.org/ebooks/42"},
    ))
    http = _mock_http_client({
        "https://www.gutenberg.org/cache/epub/42/pg42.txt": _SAMPLE_BOOK,
    })
    app = SimpleNamespace(state=SimpleNamespace(http_client=http, file_index=idx))
    register_handler("gutenberg_fetch", make_gutenberg_fetch_handler(app))

    store = await _mkstore()
    job_id = await store.create(
        user_id="u1", job_type="gutenberg_fetch",
        payload={"file_id": "fi_abc", "url_text_source": "https://www.gutenberg.org/ebooks/42"},
    )
    runner = JobRunner(store)
    runner.start()
    try:
        row = await _wait_status(store, job_id, "completed")
    finally:
        await runner.stop()

    assert row["result"]["ebook_id"] == "42"
    assert row["result"]["word_count"] > 5
    assert (tmp_path / "gutenberg" / "fi_abc.txt").is_file()
    body = (tmp_path / "gutenberg" / "fi_abc.txt").read_text(encoding="utf-8")
    assert "Body of the book" in body
    assert "*** START OF" not in body
    assert "Licence footer" not in body

    updated = idx._rows[("fi_abc", "u1")].source_metadata
    assert updated["gutenberg_status"] == "fetched"
    assert updated["gutenberg_ebook_id"] == "42"
    assert updated["gutenberg_word_count"] == row["result"]["word_count"]


@pytest.mark.asyncio
async def test_idempotent_second_run_short_circuits(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "augmentum.jobs.handlers.gutenberg_fetch.settings",
        SimpleNamespace(data_dir=str(tmp_path)),
    )
    idx = _FakeIndex()
    idx.seed(_FakeEntry(
        id="fi_abc", user_id="u1",
        source_metadata={
            "url_text_source": "https://www.gutenberg.org/ebooks/42",
            "gutenberg_status": "fetched",
            "gutenberg_ebook_id": "42",
        },
    ))
    # No URLs in the mock — any outbound call would 404. If the guard
    # fails we'd see a retry loop instead of a clean skip.
    http = _mock_http_client({})
    app = SimpleNamespace(state=SimpleNamespace(http_client=http, file_index=idx))
    register_handler("gutenberg_fetch", make_gutenberg_fetch_handler(app))

    store = await _mkstore()
    job_id = await store.create(
        user_id="u1", job_type="gutenberg_fetch",
        payload={"file_id": "fi_abc", "url_text_source": "https://www.gutenberg.org/ebooks/42"},
    )
    runner = JobRunner(store)
    runner.start()
    try:
        row = await _wait_status(store, job_id, "completed")
    finally:
        await runner.stop()
    assert row["result"].get("skipped") == "already_fetched"


@pytest.mark.asyncio
async def test_non_gutenberg_url_marks_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "augmentum.jobs.handlers.gutenberg_fetch.settings",
        SimpleNamespace(data_dir=str(tmp_path)),
    )
    idx = _FakeIndex()
    idx.seed(_FakeEntry(
        id="fi_xyz", user_id="u1",
        source_metadata={"url_text_source": "https://librivox.org/some-page"},
    ))
    http = _mock_http_client({})
    app = SimpleNamespace(state=SimpleNamespace(http_client=http, file_index=idx))
    register_handler("gutenberg_fetch", make_gutenberg_fetch_handler(app))

    store = await _mkstore()
    job_id = await store.create(
        user_id="u1", job_type="gutenberg_fetch",
        payload={"file_id": "fi_xyz", "url_text_source": "https://librivox.org/some-page"},
    )
    runner = JobRunner(store)
    runner.start()
    try:
        row = await _wait_status(store, job_id, "completed")  # permanent skip
    finally:
        await runner.stop()
    assert row["result"].get("skipped") == "not_gutenberg"
    assert idx._rows[("fi_xyz", "u1")].source_metadata["gutenberg_status"] == "unavailable"


@pytest.mark.asyncio
async def test_all_http_fallbacks_404_retries_then_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "augmentum.jobs.handlers.gutenberg_fetch.settings",
        SimpleNamespace(data_dir=str(tmp_path)),
    )
    idx = _FakeIndex()
    idx.seed(_FakeEntry(
        id="fi_404", user_id="u1",
        source_metadata={"url_text_source": "https://www.gutenberg.org/ebooks/999999"},
    ))
    http = _mock_http_client({})
    app = SimpleNamespace(state=SimpleNamespace(http_client=http, file_index=idx))
    register_handler("gutenberg_fetch", make_gutenberg_fetch_handler(app))

    store = await _mkstore()
    job_id = await store.create(
        user_id="u1", job_type="gutenberg_fetch",
        payload={"file_id": "fi_404", "url_text_source": "https://www.gutenberg.org/ebooks/999999"},
        max_attempts=2,
    )
    runner = JobRunner(store)
    runner.start()
    try:
        row = await _wait_status(store, job_id, "failed", timeout=10.0)
    finally:
        await runner.stop()
    assert row["attempts"] == 2
    assert "gutenberg fetch failed" in (row["error"] or "")
    # Row metadata untouched — transient failure doesn't mark unavailable.
    assert "gutenberg_status" not in idx._rows[("fi_404", "u1")].source_metadata
