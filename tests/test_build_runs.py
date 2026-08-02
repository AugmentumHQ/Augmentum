import pytest

# build_runs columns span several migrations (146 base, 217 acked_at, 269
# profile/target/capabilities/workspace_id, 278 resume_count). create()/update()
# write the later columns, so a test DB must apply the full set.
_BUILD_MIGRATIONS = (
    "146_build_runs.sql",
    "217_build_runs_acked.sql",
    "269_build_runs_profile.sql",
    "278_build_runs_resume_count.sql",
)


async def _apply_build_migrations(db) -> None:
    await db.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, description TEXT)"
    )
    for mig in _BUILD_MIGRATIONS:
        with open(f"augmentum/state/migrations/{mig}", encoding="utf-8") as fh:
            await db.executescript(fh.read())
    await db.commit()


@pytest.mark.asyncio
async def test_build_run_store_round_trip_user_scoped():
    import aiosqlite

    from augmentum.builds.store import BuildRunStore

    db = await aiosqlite.connect(":memory:")
    try:
        await db.execute("CREATE TABLE users (id TEXT PRIMARY KEY)")
        await db.execute("INSERT INTO users (id) VALUES ('usr_a'), ('usr_b')")
        await _apply_build_migrations(db)

        store = BuildRunStore(db)
        run = await store.create(
            build_id="build_a",
            user_id="usr_a",
            session_id="sess",
            task_id="task",
            name="Todo App",
            request={"description": "make a todo app"},
        )

        assert run["id"] == "build_a"
        assert run["user_id"] == "usr_a"
        assert await store.get("build_a", user_id="usr_b") is None

        await store.update(
            "build_a",
            user_id="usr_a",
            status="complete",
            artifact_id="art_1",
            progress={"passes": [{"name": "deliver", "status": "complete"}]},
            result={"project": {"artifactId": "art_1"}},
        )
        loaded = await store.get("build_a", user_id="usr_a")

        assert loaded["status"] == "completed"
        assert loaded["artifact_id"] == "art_1"
        assert loaded["progress"]["passes"][0]["name"] == "deliver"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_build_run_store_marks_stale_running_run_failed():
    import aiosqlite

    from augmentum.builds.store import BuildRunStore

    db = await aiosqlite.connect(":memory:")
    try:
        await db.execute("CREATE TABLE users (id TEXT PRIMARY KEY)")
        await db.execute("INSERT INTO users (id) VALUES ('usr_a')")
        await _apply_build_migrations(db)

        store = BuildRunStore(db)
        await store.create(build_id="build_stale", user_id="usr_a", name="Old Build")
        await db.execute(
            "UPDATE build_runs SET updated_at = datetime('now', '-20 minutes') WHERE id = ?",
            ("build_stale",),
        )
        await db.commit()

        marked = await store.mark_running_stale(
            "build_stale",
            user_id="usr_a",
            max_age_seconds=600,
            reason="No progress for too long.",
        )
        loaded = await store.get("build_stale", user_id="usr_a")

        assert marked is True
        assert loaded["status"] == "failed"
        assert loaded["error"] == "No progress for too long."
        assert loaded["completed_at"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_build_run_store_marks_startup_interrupted_runs_failed():
    import aiosqlite

    from augmentum.builds.store import BuildRunStore

    db = await aiosqlite.connect(":memory:")
    try:
        await db.execute("CREATE TABLE users (id TEXT PRIMARY KEY)")
        await db.execute("INSERT INTO users (id) VALUES ('usr_a')")
        await _apply_build_migrations(db)

        store = BuildRunStore(db)
        await store.create(build_id="build_running", user_id="usr_a")

        count = await store.mark_running_interrupted(reason="Server restarted.")
        loaded = await store.get("build_running", user_id="usr_a")

        assert count == 1
        assert loaded["status"] == "failed"
        assert loaded["error"] == "Server restarted."
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_latest_for_session_hides_acked_and_aged_terminal_runs():
    """Regression: terminal builds resurfacing on every device forever.

    Repro: complete a build, then ack it OR let it age past 24h. Both
    paths must keep the build out of latest_for_session() so the
    persistent monitor doesn't pop on every page load. Active builds
    must still surface regardless of age — switching devices mid-build
    should keep following the live run.
    """
    import aiosqlite

    from augmentum.builds.store import BuildRunStore

    db = await aiosqlite.connect(":memory:")
    try:
        await db.execute("CREATE TABLE users (id TEXT PRIMARY KEY)")
        await db.execute("INSERT INTO users (id) VALUES ('usr_a')")
        await _apply_build_migrations(db)

        store = BuildRunStore(db)

        # Recent terminal build: surfaces normally.
        await store.create(build_id="build_recent", user_id="usr_a", name="Recent")
        await store.update("build_recent", user_id="usr_a", status="complete")
        latest = await store.latest_for_session(user_id="usr_a")
        assert latest and latest["id"] == "build_recent"

        # After ack: hidden.
        acked = await store.mark_acked("build_recent", user_id="usr_a")
        assert acked is True
        latest = await store.latest_for_session(user_id="usr_a")
        assert latest is None

        # Second ack is a no-op (already stamped).
        assert await store.mark_acked("build_recent", user_id="usr_a") is False

        # Aged terminal build with NO ack: also hidden once past 24h.
        await store.create(build_id="build_old", user_id="usr_a", name="Old")
        await store.update("build_old", user_id="usr_a", status="failed")
        await db.execute(
            "UPDATE build_runs SET updated_at = datetime('now', '-25 hours') WHERE id = ?",
            ("build_old",),
        )
        await db.commit()
        latest = await store.latest_for_session(user_id="usr_a")
        assert latest is None

        # Active (non-terminal) build always surfaces — even aged.
        await store.create(build_id="build_active", user_id="usr_a", name="Active")
        await db.execute(
            "UPDATE build_runs SET updated_at = datetime('now', '-48 hours') WHERE id = ?",
            ("build_active",),
        )
        await db.commit()
        latest = await store.latest_for_session(user_id="usr_a")
        assert latest and latest["id"] == "build_active"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_begin_resume_reactivates_and_bumps_count():
    """A terminal build flips back to running and bumps resume_count; a build
    that's already running cannot be resumed (guard returns 0)."""
    import aiosqlite

    from augmentum.builds.store import BuildRunStore

    db = await aiosqlite.connect(":memory:")
    try:
        await db.execute("CREATE TABLE users (id TEXT PRIMARY KEY)")
        await db.execute("INSERT INTO users (id) VALUES ('usr_a')")
        await _apply_build_migrations(db)

        store = BuildRunStore(db)
        await store.create(build_id="build_r", user_id="usr_a", name="App")

        # Running build: not resumable.
        assert await store.begin_resume("build_r", user_id="usr_a") == 0

        # Fail it, then resume -> count 1, status running, error cleared.
        await store.update("build_r", user_id="usr_a", status="failed", error="boom")
        count = await store.begin_resume("build_r", user_id="usr_a")
        assert count == 1
        loaded = await store.get("build_r", user_id="usr_a")
        assert loaded["status"] == "running"
        assert loaded["error"] in (None, "")
        assert loaded["resume_count"] == 1

        # Cross-user isolation.
        await store.update("build_r", user_id="usr_a", status="completed")
        assert await store.begin_resume("build_r", user_id="usr_b") == 0

        # Resume a completed build (re-prompt) -> count 2.
        assert await store.begin_resume("build_r", user_id="usr_a") == 2
    finally:
        await db.close()


def test_build_status_from_run_surfaces_verdict_and_quality():
    """A persisted completed-but-unverified build round-trips its verdict +
    qualityStatus + warnings so the library can flag it before the user opens
    it."""
    from augmentum.builds.runtime import build_status_from_run

    snap = build_status_from_run({
        "id": "b1",
        "name": "Tip Calc",
        "status": "completed",
        "kind": "application",
        "session_id": "",
        "task_id": "",
        "artifact_id": "art_1",
        "request": {"objective": "tip calculator", "model": "deepseek"},
        "progress": {},
        "result": {
            "artifact_id": "art_1",
            "stop_reason": "complete",
            "verdict": {"passed": False, "unproven": ["asserted that behaviors actually work"]},
            "project": {
                "name": "Tip Calc",
                "qualityStatus": "unverified",
                "warnings": ["Not proven: the build never asserted that behaviors actually work."],
            },
        },
        "error": "",
        "resume_count": 1,
        "created_at": "2026-06-20T12:00:00Z",
    })
    assert snap["verdict"]["passed"] is False
    assert snap["qualityStatus"] == "unverified"
    assert snap["warnings"]
    assert snap["resume_count"] == 1


def test_project_progress_normalizer_matches_build_status_shape():
    from augmentum.builds.runtime import (
        apply_project_progress,
        build_status_from_run,
        progress_payload_from_state,
    )

    state = {"name": "", "passes": []}
    apply_project_progress(
        state,
        {
            "name": "Recipe App",
            "pass": "generate",
            "status": "running",
            "detail": "2/4 files",
            "iteration": 1,
            "max_iterations": 12,
            "files": [{"path": "index.html", "content": ""}],
            "planned_files": [{"path": "index.html"}, {"path": "app.js"}],
            "completed_files": ["index.html"],
            "totalTokens": 1200,
            "llmCalls": 3,
            "qualityStatus": "needs_review",
            "warnings": ["Validation found one issue."],
            "blockingErrors": ["app.js: missing handler"],
        },
    )

    progress = progress_payload_from_state(state)
    snap = build_status_from_run({
        "id": "build_agentic",
        "name": state["name"],
        "status": "running",
        "kind": "application",
        "session_id": "sess",
        "task_id": "task",
        "artifact_id": "",
        "progress": progress,
        "result": {},
        "error": "",
    })

    assert snap["active"] is True
    assert snap["passes"][0]["name"] == "generate"
    assert snap["passes"][0]["detail"] == "2/4 files"
    assert snap["passes"][0]["iterations"] == 1
    assert snap["passes"][0]["max_iterations"] == 12
    assert snap["filesComplete"] == ["index.html"]
    assert snap["filesRemaining"] == ["app.js"]
    assert snap["totalTokens"] == 1200
    assert progress["qualityStatus"] == "needs_review"
    assert progress["warnings"] == ["Validation found one issue."]
    assert progress["blockingErrors"] == ["app.js: missing handler"]
    assert snap["qualityStatus"] == "needs_review"
    assert snap["warnings"] == ["Validation found one issue."]


def test_build_status_from_run_exposes_model_and_started_iso():
    from augmentum.builds.runtime import build_status_from_run

    snap = build_status_from_run({
        "id": "b1",
        "name": "App",
        "status": "running",
        "kind": "application",
        "session_id": "",
        "task_id": "",
        "artifact_id": "",
        "request": {"description": "todo app", "model": "qwen3.6-35b"},
        "progress": {},
        "result": {},
        "error": "",
        "created_at": "2026-06-01T12:00:00Z",
    })
    assert snap["model"] == "qwen3.6-35b"
    assert snap["startedAtIso"] == "2026-06-01T12:00:00Z"


def test_build_status_from_run_exposes_failure_forensics():
    """A persisted failed build must round-trip the pass tombstone +
    traceback so the UI can render the rich error card without
    re-running the pipeline. Two sources for the data: progress blob
    (in-memory snapshot path) and project metadata (pipeline exception
    path) — accept either."""
    from augmentum.builds.runtime import build_status_from_run

    snap = build_status_from_run({
        "id": "b1",
        "name": "Snake Game",
        "status": "error",
        "kind": "application",
        "session_id": "",
        "task_id": "",
        "artifact_id": "",
        "request": {"description": "snake game", "model": "qwen3.6-35b"},
        "progress": {
            "failedPass": "generate",
            "lastCompletedPass": "plan",
            "errorDetail": "Traceback:\n  File \"x.py\"...\nRuntimeError: boom",
        },
        "result": {"project": {"name": "Snake Game"}},
        "error": "boom",
        "created_at": "2026-06-01T12:00:00Z",
    })
    assert snap["failedPass"] == "generate"
    assert snap["lastCompletedPass"] == "plan"
    assert "RuntimeError: boom" in snap["errorDetail"]
    # Fallback: when the progress blob is empty, project-level fields win.
    snap2 = build_status_from_run({
        "id": "b2",
        "name": "App",
        "status": "error",
        "kind": "application",
        "session_id": "",
        "task_id": "",
        "artifact_id": "",
        "request": {},
        "progress": {},
        "result": {"project": {
            "name": "App",
            "failed_pass": "validate",
            "last_completed_pass": "generate",
            "error_detail": "trace text",
        }},
        "error": "boom",
        "created_at": "2026-06-01T12:00:00Z",
    })
    assert snap2["failedPass"] == "validate"
    assert snap2["lastCompletedPass"] == "generate"
    assert snap2["errorDetail"] == "trace text"
