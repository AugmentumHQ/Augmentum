"""Integration test for the resume path of augmentum.builds.facade.run_build.

Uses fakes for the backend + container manager so the resume mechanics —
reuse the existing workspace, do NOT create a second build_runs row, use the
continuation prompt, persist the quality verdict — are exercised without
Docker or a real model.
"""

from __future__ import annotations

import gzip
import io
import tarfile

import pytest

from augmentum.models.base import InternalChatResponse, Message, Usage


def _make_targz(files: dict[str, str]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tf:
        for path, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=f"workspace/{path}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return gzip.compress(raw.getvalue())


class _FakeBackend:
    """Returns one tool-call-free completion -> stop_reason 'complete' on the
    first iteration. Records the messages it was handed so the test can assert
    the continuation prompt was used."""

    def __init__(self) -> None:
        self.seen_messages: list[Message] = []

    async def chat(self, request):  # noqa: ANN001
        self.seen_messages = list(request.messages)
        return InternalChatResponse(
            message=Message(role="assistant", content="Continued and verified."),
            model=request.model,
            usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )


class _FakeContainerManager:
    def __init__(self, archive: bytes) -> None:
        self._archive = archive
        self.created = 0

    async def create_workspace(self, **_kw):  # noqa: ANN003
        # Resume must NOT create a workspace — fail loudly if it does.
        self.created += 1
        raise AssertionError("run_build(resume=True) must reuse, not create, a workspace")

    async def workspace_archive_stream(self, workspace_id, excludes=None):  # noqa: ANN001
        yield self._archive


class _FakeArtifactStore:
    def __init__(self) -> None:
        self.saved: list[dict] = []

    async def save(self, **kw):  # noqa: ANN003
        self.saved.append(kw)
        return {"id": "art_resumed"}


async def _store_with_row(build_id: str, user_id: str):
    import aiosqlite

    from augmentum.builds.store import BuildRunStore

    db = await aiosqlite.connect(":memory:")
    await db.execute("CREATE TABLE users (id TEXT PRIMARY KEY)")
    await db.execute("INSERT INTO users (id) VALUES (?)", (user_id,))
    await db.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT)"
    )
    for mig in (
        "146_build_runs.sql",
        "217_build_runs_acked.sql",
        "269_build_runs_profile.sql",
        "278_build_runs_resume_count.sql",
    ):
        await db.executescript(
            open(f"augmentum/state/migrations/{mig}", encoding="utf-8").read()
        )
    await db.commit()
    store = BuildRunStore(db)
    # Pre-existing failed build, flipped to running for the resume.
    await store.create(build_id=build_id, user_id=user_id, name="Tip Calc",
                       request={"objective": "a tip calculator", "model": "deepseek"})
    await store.update(build_id, user_id=user_id, status="failed", error="budget")
    await store.begin_resume(build_id, user_id=user_id)
    return db, store


@pytest.mark.asyncio
async def test_run_build_resume_reuses_workspace_and_persists_verdict():
    from augmentum.builds.facade import run_build

    uid, bid = "usr_a", "build_resume"
    db, store = await _store_with_row(bid, uid)
    try:
        backend = _FakeBackend()
        cm = _FakeContainerManager(_make_targz({"index.html": "<html><body>hi</body></html>"}))
        artifacts = _FakeArtifactStore()

        result = await run_build(
            objective="a tip calculator",
            user_id=uid, backend=backend, model="deepseek",
            container_manager=cm, artifact_store=artifacts, build_run_store=store,
            profile_id="calculator", build_id=bid,
            reuse_workspace_id="ws_existing", resume=True,
            instructions="also add a dark mode toggle",
            prior_steps=[{"tool": "file_write", "preview": "index.html"}],
            prior_stop_reason="budget", kind="calculator",
        )

        # Reused the workspace, never created one.
        assert cm.created == 0
        assert result["workspace_id"] == "ws_existing"
        assert result["status"] == "completed"
        assert result["artifact_id"] == "art_resumed"

        # Used the continuation prompt with the user's new instructions.
        user_msg = next(m for m in backend.seen_messages if m.role == "user")
        assert "CONTINUING" in user_msg.content
        assert "dark mode toggle" in user_msg.content
        assert "budget" not in user_msg.content.lower() or "ran out" in user_msg.content.lower()

        # Quality gate ran: the fake never drove/asserted -> unverified.
        assert result["qualityStatus"] == "unverified"
        loaded = await store.get(bid, user_id=uid)
        assert loaded["status"] == "completed"
        assert loaded["result"]["verdict"]["passed"] is False
        assert loaded["result"]["qualityStatus"] == "unverified"
        # Still one row (resume reused it; create() was not called again).
        assert loaded["resume_count"] == 1
    finally:
        await db.close()
