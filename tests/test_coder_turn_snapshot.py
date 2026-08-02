"""Tests for augmentum/coder/turn_snapshot.py — the pre-write capture layer
behind the reviewable-turn diff flow.

What we need to prove here is paranoid by design, because the reject
path is lossy if the snapshot layer mis-behaves:

* **Capture semantics.** First-write captures pre-state; subsequent
  writes for the same path don't re-capture (would stomp the true
  pre-state with mid-turn content). Missing files yield the ``None``
  sentinel (distinct from "empty file"). Read failures mark the path
  skipped rather than losing data silently.
* **Diff semantics.** Identical pre/post produces nothing (no-ops
  don't clutter review). Added/modified/deleted statuses match
  the pre/post reality. Unified diff contains git-style a/b prefix
  labels so downstream renderers don't need to re-shape them.
* **Restore semantics.** Previously-existing file → write pre-bytes
  back. Previously-missing file → delete (the snapshot captured
  absence, so restore = absence). Skipped paths fail loudly so the
  caller surfaces the non-reversibility.
* **Size-limit semantics.** Cap trips → further writes get
  ``skipped`` status, not silently lost. Already-captured paths
  stay captured.

The stub container manager captures every ``file_read`` / ``file_write``
/ ``run_command`` call so we can assert not just end state but the
exact sequence of calls — the snapshot must not re-read a file it's
already captured, and must not touch disk on the reject path for
skipped paths.
"""
from __future__ import annotations

import pytest

from augmentum.coder.turn_snapshot import (
    DiffEntry,
    TurnSnapshot,
    _classify_status,
    _unified_diff,
    _SIZE_LIMIT_BYTES,
)


# ---------------------------------------------------------------------------
# Stub container manager
# ---------------------------------------------------------------------------


class _StubCM:
    """In-memory stand-in for ContainerManager with a dict-backed FS.

    ``files`` maps absolute container path → str content. Missing key
    means the file doesn't exist; file_read raises FileNotFoundError.
    Track all mutating calls so tests can assert ordering + de-dup.
    """

    def __init__(self, files: dict[str, str] | None = None):
        self.files: dict[str, str] = dict(files or {})
        self.read_calls: list[str] = []
        self.write_calls: list[tuple[str, str]] = []
        self.run_calls: list[list[str]] = []
        # Inject read errors for specific paths: path → Exception
        self.read_errors: dict[str, Exception] = {}

    async def file_read(self, workspace_id: str, path: str) -> str:
        self.read_calls.append(path)
        if path in self.read_errors:
            raise self.read_errors[path]
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def file_write(self, workspace_id: str, path: str, content: str) -> None:
        self.write_calls.append((path, content))
        self.files[path] = content

    async def run_command(self, workspace_id: str, cmd: list[str], **_kw) -> str:
        self.run_calls.append(cmd)
        # Simulate rm -f: strip the file from self.files if the command
        # matches "rm -f <path>" after shell unwrap.
        joined = " ".join(cmd)
        if joined.startswith("sh -c") and "rm -f" in cmd[-1]:
            # Extract the quoted path — tests use the exact shape the
            # module emits.
            import re
            m = re.search(r"rm -f '([^']+)'", cmd[-1])
            if m:
                self.files.pop(m.group(1), None)
        return ""


def _make_snap(files: dict[str, str] | None = None) -> tuple[TurnSnapshot, _StubCM]:
    cm = _StubCM(files)
    snap = TurnSnapshot(turn_id="t-1", workspace_id="ws-1", container_manager=cm)
    return snap, cm


# ---------------------------------------------------------------------------
# Capture semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_captures_existing_file_bytes():
    snap, cm = _make_snap({"/workspace/a.py": "print('old')"})
    await snap.snapshot_before_write("/workspace/a.py")

    # One read; content captured as bytes.
    assert cm.read_calls == ["/workspace/a.py"]
    assert snap._snapshots["/workspace/a.py"] == b"print('old')"
    assert snap.total_bytes_captured == len(b"print('old')")


@pytest.mark.asyncio
async def test_snapshot_of_missing_file_uses_none_sentinel():
    """Path that doesn't exist pre-write → None sentinel, distinct
    from empty string. Diff classification + restore depend on this
    distinction."""
    snap, cm = _make_snap({})
    await snap.snapshot_before_write("/workspace/new.py")

    assert snap._snapshots["/workspace/new.py"] is None
    # Sentinel shouldn't count toward the size budget.
    assert snap.total_bytes_captured == 0


@pytest.mark.asyncio
async def test_snapshot_is_idempotent_per_path():
    """Repeated snapshot calls for the same path must not re-read —
    re-reading would overwrite the true pre-turn state with mid-turn
    content the agent already wrote."""
    snap, cm = _make_snap({"/workspace/a.py": "original"})
    await snap.snapshot_before_write("/workspace/a.py")

    # Simulate: tool writes "new" to disk, then another tool tries to
    # snapshot before its own write. Without idempotence we'd capture
    # "new" as the "pre-state" — which is WRONG.
    cm.files["/workspace/a.py"] = "mid-turn-content"
    await snap.snapshot_before_write("/workspace/a.py")

    assert cm.read_calls == ["/workspace/a.py"]  # only one read
    assert snap._snapshots["/workspace/a.py"] == b"original"


@pytest.mark.asyncio
async def test_snapshot_read_error_marks_skipped_not_lost():
    """A weird read failure (permission, container dropped, etc.)
    must not silently lose data. Path goes into ``skipped`` so the
    review bundle flags it ``reversible=False``."""
    snap, cm = _make_snap({"/workspace/weird.py": "unused"})
    cm.read_errors["/workspace/weird.py"] = RuntimeError("container down")
    await snap.snapshot_before_write("/workspace/weird.py")

    assert "/workspace/weird.py" in snap._skipped
    assert "/workspace/weird.py" not in snap._snapshots


@pytest.mark.asyncio
async def test_snapshot_size_limit_skips_further_captures(monkeypatch):
    """Big captures early in the turn shouldn't stop diff generation,
    but further snapshot calls past the limit go into ``skipped`` so
    the user sees them as non-reversible rather than losing the
    restore path silently."""
    import augmentum.coder.turn_snapshot as mod
    monkeypatch.setattr(mod, "_SIZE_LIMIT_BYTES", 20)

    big = "x" * 25
    snap, cm = _make_snap({
        "/workspace/big.txt": big,
        "/workspace/small.py": "k",
    })
    await snap.snapshot_before_write("/workspace/big.txt")
    await snap.snapshot_before_write("/workspace/small.py")

    # First captured (pushes past limit), second skipped.
    assert snap._snapshots["/workspace/big.txt"] == big.encode()
    assert "/workspace/small.py" in snap._skipped
    assert "/workspace/small.py" not in snap._snapshots


# ---------------------------------------------------------------------------
# Diff semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_diffs_emits_nothing_for_unchanged_file():
    """Tool wrote identical content → no-op → must not appear in the
    review bundle. Otherwise every touched-but-not-really-edited file
    clutters the UI."""
    snap, cm = _make_snap({"/workspace/a.py": "same"})
    await snap.snapshot_before_write("/workspace/a.py")
    # Agent wrote identical content.
    await cm.file_write("ws", "/workspace/a.py", "same")

    diffs = await snap.collect_diffs()
    assert diffs == []


@pytest.mark.asyncio
async def test_collect_diffs_detects_added_file():
    snap, cm = _make_snap({})
    await snap.snapshot_before_write("/workspace/new.py")
    await cm.file_write("ws", "/workspace/new.py", "print('hi')")

    diffs = await snap.collect_diffs()
    assert len(diffs) == 1
    entry = diffs[0]
    assert entry.path == "/workspace/new.py"
    assert entry.status == "added"
    assert entry.old_size == 0
    assert entry.new_size == len(b"print('hi')")
    assert entry.reversible is True
    assert "+print('hi')" in entry.unified_diff


@pytest.mark.asyncio
async def test_collect_diffs_detects_modified_file():
    snap, cm = _make_snap({"/workspace/a.py": "old"})
    await snap.snapshot_before_write("/workspace/a.py")
    await cm.file_write("ws", "/workspace/a.py", "new\ncontent")

    diffs = await snap.collect_diffs()
    assert len(diffs) == 1
    assert diffs[0].status == "modified"
    assert "-old" in diffs[0].unified_diff
    assert "+new" in diffs[0].unified_diff


@pytest.mark.asyncio
async def test_collect_diffs_detects_deleted_file():
    """Agent deleted the file mid-turn (via some path that called
    snapshot_before_write; shell-rm doesn't but that's out of scope
    per the module's non-goals)."""
    snap, cm = _make_snap({"/workspace/gone.py": "bye"})
    await snap.snapshot_before_write("/workspace/gone.py")
    # Simulate deletion.
    cm.files.pop("/workspace/gone.py")

    diffs = await snap.collect_diffs()
    assert len(diffs) == 1
    assert diffs[0].status == "deleted"
    assert diffs[0].new_size == 0
    assert "-bye" in diffs[0].unified_diff


@pytest.mark.asyncio
async def test_collect_diffs_stable_order():
    """Deterministic path order matters for tests + UI stability."""
    snap, cm = _make_snap({})
    for p in ["/workspace/c.py", "/workspace/a.py", "/workspace/b.py"]:
        await snap.snapshot_before_write(p)
        await cm.file_write("ws", p, "x")

    diffs = await snap.collect_diffs()
    assert [d.path for d in diffs] == [
        "/workspace/a.py", "/workspace/b.py", "/workspace/c.py",
    ]


@pytest.mark.asyncio
async def test_skipped_path_still_yields_diff_entry_flagged_non_reversible():
    """When capture failed (read error) but the file exists at turn
    end, we still want the user to SEE the change — just flagged so
    they know reject won't fully undo it."""
    snap, cm = _make_snap({"/workspace/a.py": "will-error"})
    cm.read_errors["/workspace/a.py"] = RuntimeError("read blew up")
    await snap.snapshot_before_write("/workspace/a.py")
    # Agent wrote new content AFTER our snapshot failed.
    cm.read_errors.clear()  # so collect_diffs can read the post-state
    await cm.file_write("ws", "/workspace/a.py", "final")

    diffs = await snap.collect_diffs()
    assert len(diffs) == 1
    assert diffs[0].reversible is False
    # Status falls back to "added" for a skipped file that exists at
    # turn end — see _classify_status rationale.
    assert diffs[0].status == "added"


# ---------------------------------------------------------------------------
# Restore semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restore_writes_pre_content_back():
    snap, cm = _make_snap({"/workspace/a.py": "original"})
    await snap.snapshot_before_write("/workspace/a.py")
    await cm.file_write("ws", "/workspace/a.py", "agent-version")

    failed = await snap.restore(["/workspace/a.py"])
    assert failed == []
    assert cm.files["/workspace/a.py"] == "original"


@pytest.mark.asyncio
async def test_restore_deletes_file_that_didnt_exist_pre_turn():
    """Pre-turn absence is encoded as None. Restore = delete."""
    snap, cm = _make_snap({})
    await snap.snapshot_before_write("/workspace/new.py")
    await cm.file_write("ws", "/workspace/new.py", "created-by-agent")

    failed = await snap.restore(["/workspace/new.py"])
    assert failed == []
    assert "/workspace/new.py" not in cm.files
    # run_command must have been invoked with an rm -f shape.
    assert any(
        "rm -f" in (cmd[-1] if len(cmd) >= 3 else "")
        for cmd in cm.run_calls
    )


@pytest.mark.asyncio
async def test_restore_reports_skipped_paths_as_failed():
    """Non-reversible paths must fail LOUDLY so callers surface to
    the user rather than claim a clean rollback."""
    snap, cm = _make_snap({"/workspace/a.py": "x"})
    cm.read_errors["/workspace/a.py"] = RuntimeError("nope")
    await snap.snapshot_before_write("/workspace/a.py")

    failed = await snap.restore(["/workspace/a.py"])
    assert failed == ["/workspace/a.py"]


@pytest.mark.asyncio
async def test_restore_unknown_path_fails():
    """A restore call for a path we never snapshotted must NOT silently
    no-op — the caller thinks it rolled back something it didn't."""
    snap, _cm = _make_snap({})
    failed = await snap.restore(["/workspace/never-touched.py"])
    assert failed == ["/workspace/never-touched.py"]


@pytest.mark.asyncio
async def test_restore_partial_failure_is_per_path():
    """Mixed batch — some restorable, some not — returns only the
    failed subset, restores the rest."""
    snap, cm = _make_snap({"/workspace/ok.py": "ok"})
    cm.read_errors["/workspace/bad.py"] = RuntimeError("nope")
    await snap.snapshot_before_write("/workspace/ok.py")
    await snap.snapshot_before_write("/workspace/bad.py")
    await cm.file_write("ws", "/workspace/ok.py", "changed")

    failed = await snap.restore([
        "/workspace/ok.py", "/workspace/bad.py",
    ])
    assert failed == ["/workspace/bad.py"]
    assert cm.files["/workspace/ok.py"] == "ok"


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_touched_paths_covers_snapshotted_and_skipped():
    snap, cm = _make_snap({"/workspace/a.py": "a", "/workspace/b.py": "b"})
    cm.read_errors["/workspace/b.py"] = RuntimeError("nope")
    await snap.snapshot_before_write("/workspace/a.py")
    await snap.snapshot_before_write("/workspace/b.py")
    assert snap.touched_paths == ["/workspace/a.py", "/workspace/b.py"]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_classify_status_matrix():
    # (pre_existed, post_exists, skipped) → status
    assert _classify_status(False, True, False) == "added"
    assert _classify_status(True, False, False) == "deleted"
    assert _classify_status(True, True, False) == "modified"
    # Skipped paths: fall back to presence-based heuristic.
    assert _classify_status(False, True, True) == "added"
    assert _classify_status(True, False, True) == "deleted"


def test_unified_diff_has_git_style_labels():
    out = _unified_diff("/workspace/x.py", b"a\n", b"b\n")
    assert "--- a/workspace/x.py" in out
    assert "+++ b/workspace/x.py" in out


def test_unified_diff_handles_added_file_from_none():
    out = _unified_diff("/workspace/new.py", None, b"fresh\n")
    assert "+++ b/workspace/new.py" in out
    assert "+fresh" in out


def test_unified_diff_handles_deleted_file_to_none():
    out = _unified_diff("/workspace/gone.py", b"bye\n", None)
    assert "--- a/workspace/gone.py" in out
    assert "-bye" in out


def test_unified_diff_binary_bytes_dont_crash():
    """Non-UTF8 bytes must decode-replace, not raise."""
    out = _unified_diff("/workspace/bin.dat", b"\xff\xfe\xfd", b"\xff\xfe\x00")
    assert out  # just prove it didn't crash
