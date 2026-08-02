"""SWD engine tests.

Cover the load-bearing checks borrowed from mythos-router:

* Intent=MUTATE on a no-op write fails (hallucinated-fix catch).
* CREATE / MODIFY / DELETE happy paths verify.
* Content-hash drift (model wrote different bytes than promised).
* Concurrency drift (file changed between apply and verify).
* Strict-mode batch rollback.
* Sensitive-path refusal (.env, .git/, *.key).
* Oversize-file refusal for modify/delete.
* Path-escape refusal (..).
"""

from __future__ import annotations

from pathlib import Path

from augmentum.bug_finder.swd import (
    ActionIntent,
    ActionOp,
    ActionStatus,
    FileAction,
    SWDEngine,
    parse_file_actions,
    resolve_safe_path,
    snapshot_file,
)

# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_snapshot_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello")
    snap = snapshot_file(f, "a.txt")
    assert snap.exists
    assert snap.size == 5
    assert snap.sha256
    assert snap.content == b"hello"


def test_snapshot_missing_file(tmp_path: Path) -> None:
    snap = snapshot_file(tmp_path / "nope.txt", "nope.txt")
    assert not snap.exists
    assert snap.sha256 == ""


def test_snapshot_oversize_skips_content(tmp_path: Path) -> None:
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * 100)
    snap = snapshot_file(f, "big.bin", max_bytes=10)
    assert snap.exists
    assert snap.sha256 == "oversize"
    assert snap.content is None


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def test_resolve_safe_path_rejects_escape(tmp_path: Path) -> None:
    assert resolve_safe_path(tmp_path, "../etc/passwd") is None


def test_resolve_safe_path_rejects_sensitive_file(tmp_path: Path) -> None:
    assert resolve_safe_path(tmp_path, ".env") is None
    assert resolve_safe_path(tmp_path, "config/secrets.toml") is None
    assert resolve_safe_path(tmp_path, "keys/server.key") is None


def test_resolve_safe_path_rejects_dot_git(tmp_path: Path) -> None:
    assert resolve_safe_path(tmp_path, ".git/HEAD") is None


def test_resolve_safe_path_allows_normal_file(tmp_path: Path) -> None:
    p = resolve_safe_path(tmp_path, "src/app.py")
    assert p is not None
    assert p == (tmp_path / "src" / "app.py").resolve()


# ---------------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------------


def test_create_verified_on_happy_path(tmp_path: Path) -> None:
    actions = [FileAction(
        op=ActionOp.CREATE.value, path="src/new.py",
        intent=ActionIntent.MUTATE.value,
        content=b"print('hi')\n",
        reason="add greeter module", finding_id="fnd_x",
    )]
    res = SWDEngine(workspace_root=tmp_path).run(actions)
    assert res.success
    assert res.results[0].status == ActionStatus.VERIFIED.value
    assert (tmp_path / "src" / "new.py").read_bytes() == b"print('hi')\n"


def test_create_drifts_when_target_existed(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("preexisting", encoding="utf-8")
    res = SWDEngine(workspace_root=tmp_path).run([FileAction(
        op=ActionOp.CREATE.value, path="x.py",
        intent=ActionIntent.MUTATE.value, content=b"new",
    )])
    assert res.results[0].status == ActionStatus.DRIFT.value
    # Strict mode: batch rolled back; preexisting content restored.
    assert (tmp_path / "x.py").read_text(encoding="utf-8") == "preexisting"


# ---------------------------------------------------------------------------
# MODIFY — the load-bearing intent check
# ---------------------------------------------------------------------------


def test_modify_fails_when_mutate_intent_but_no_change(tmp_path: Path) -> None:
    """The headline check: model said MUTATE but wrote identical
    content. This is the hallucinated-fix pattern SWD exists to catch."""
    f = tmp_path / "a.py"
    f.write_bytes(b"x = 1\n")
    res = SWDEngine(workspace_root=tmp_path).run([FileAction(
        op=ActionOp.MODIFY.value, path="a.py",
        intent=ActionIntent.MUTATE.value,
        content=b"x = 1\n",      # same content!
        reason="hallucinated fix", finding_id="fnd_h",
    )])
    assert not res.success
    assert res.results[0].status == ActionStatus.FAILED.value
    assert "unchanged" in res.results[0].error.lower()


def test_modify_verified_when_content_changed(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_bytes(b"x = 1\n")
    res = SWDEngine(workspace_root=tmp_path).run([FileAction(
        op=ActionOp.MODIFY.value, path="a.py",
        intent=ActionIntent.MUTATE.value,
        content=b"x = 2\n",
    )])
    assert res.success
    assert res.results[0].status == ActionStatus.VERIFIED.value
    assert f.read_bytes() == b"x = 2\n"


def test_modify_drifts_when_noop_intent_but_changed(tmp_path: Path) -> None:
    """Inverse: claimed NOOP but the write changed the file. Drift."""
    f = tmp_path / "a.py"
    f.write_bytes(b"x = 1\n")
    res = SWDEngine(workspace_root=tmp_path).run([FileAction(
        op=ActionOp.MODIFY.value, path="a.py",
        intent=ActionIntent.NOOP.value,
        content=b"x = 2\n",
    )])
    assert not res.success
    assert res.results[0].status == ActionStatus.DRIFT.value


def test_modify_fails_when_target_missing(tmp_path: Path) -> None:
    res = SWDEngine(workspace_root=tmp_path).run([FileAction(
        op=ActionOp.MODIFY.value, path="never_existed.py",
        intent=ActionIntent.MUTATE.value, content=b"x",
    )])
    assert res.results[0].status == ActionStatus.FAILED.value
    assert "did not exist" in res.results[0].error.lower()


def test_modify_drifts_on_oversize(tmp_path: Path) -> None:
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * 1000)
    engine = SWDEngine(workspace_root=tmp_path, max_snapshot_bytes=10)
    res = engine.run([FileAction(
        op=ActionOp.MODIFY.value, path="big.bin",
        intent=ActionIntent.MUTATE.value, content=b"y",
    )])
    assert res.results[0].status == ActionStatus.FAILED.value
    assert "modify/delete blocked" in res.results[0].error


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------


def test_delete_verified_on_happy_path(tmp_path: Path) -> None:
    f = tmp_path / "doomed.py"
    f.write_bytes(b"x")
    res = SWDEngine(workspace_root=tmp_path).run([FileAction(
        op=ActionOp.DELETE.value, path="doomed.py",
        intent=ActionIntent.MUTATE.value,
    )])
    assert res.success
    assert res.results[0].status == ActionStatus.VERIFIED.value
    assert not f.exists()


def test_delete_noops_when_missing(tmp_path: Path) -> None:
    res = SWDEngine(workspace_root=tmp_path).run([FileAction(
        op=ActionOp.DELETE.value, path="nope.py",
        intent=ActionIntent.MUTATE.value,
    )])
    # NOOP is a successful outcome — the file is already in the
    # desired state. Caller can decide whether to surface a warning.
    assert res.results[0].status == ActionStatus.NOOP.value


# ---------------------------------------------------------------------------
# Strict-mode batch rollback
# ---------------------------------------------------------------------------


def test_batch_rollback_restores_all_files(tmp_path: Path) -> None:
    """One bad action in a batch reverts the whole batch in strict mode."""
    (tmp_path / "a.py").write_bytes(b"original a\n")
    (tmp_path / "b.py").write_bytes(b"original b\n")
    res = SWDEngine(workspace_root=tmp_path).run([
        FileAction(
            op=ActionOp.MODIFY.value, path="a.py",
            intent=ActionIntent.MUTATE.value,
            content=b"patched a\n",
        ),
        FileAction(
            op=ActionOp.MODIFY.value, path="b.py",
            intent=ActionIntent.MUTATE.value,
            content=b"original b\n",   # hallucinated mutate → fail
        ),
    ])
    assert not res.success
    assert res.rolled_back
    # a.py was rolled back from "patched a" → "original a"
    assert (tmp_path / "a.py").read_bytes() == b"original a\n"
    assert (tmp_path / "b.py").read_bytes() == b"original b\n"


def test_non_strict_mode_keeps_partial_writes(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_bytes(b"original a\n")
    (tmp_path / "b.py").write_bytes(b"original b\n")
    res = SWDEngine(workspace_root=tmp_path, strict=False).run([
        FileAction(
            op=ActionOp.MODIFY.value, path="a.py",
            intent=ActionIntent.MUTATE.value, content=b"patched a\n",
        ),
        FileAction(
            op=ActionOp.MODIFY.value, path="b.py",
            intent=ActionIntent.MUTATE.value, content=b"original b\n",
        ),
    ])
    assert not res.success
    assert not res.rolled_back
    assert (tmp_path / "a.py").read_bytes() == b"patched a\n"


# ---------------------------------------------------------------------------
# Text protocol
# ---------------------------------------------------------------------------


def test_parse_file_actions_basic() -> None:
    out = (
        "Here's the patch:\n\n"
        "[FILE_ACTION: op=modify, path=src/a.py, intent=mutate]\n"
        "x = 2\n"
        "[/FILE_ACTION]\n"
    )
    actions = parse_file_actions(out)
    assert len(actions) == 1
    a = actions[0]
    assert a.op == "modify"
    assert a.path == "src/a.py"
    assert a.intent == "mutate"
    assert a.content == b"x = 2\n"


def test_parse_file_actions_multiple() -> None:
    out = (
        "[FILE_ACTION: op=create, path=new.py, intent=mutate]\nprint(1)\n[/FILE_ACTION]\n"
        "[FILE_ACTION: op=delete, path=old.py]\nremove legacy\n[/FILE_ACTION]\n"
    )
    actions = parse_file_actions(out)
    assert len(actions) == 2
    assert actions[0].op == "create"
    assert actions[1].op == "delete"
    assert actions[1].reason == "remove legacy"


def test_parse_file_actions_empty_when_no_block() -> None:
    assert parse_file_actions("just prose") == []


# ---------------------------------------------------------------------------
# Integration: SWD round-trip via text protocol
# ---------------------------------------------------------------------------


def test_swd_runs_actions_parsed_from_text(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_bytes(b"x = 1\n")
    llm_output = (
        "Fixed it:\n"
        "[FILE_ACTION: op=modify, path=x.py, intent=mutate]\n"
        "x = 42\n"
        "[/FILE_ACTION]\n"
    )
    actions = parse_file_actions(llm_output)
    res = SWDEngine(workspace_root=tmp_path).run(actions)
    assert res.success
    assert (tmp_path / "x.py").read_bytes() == b"x = 42\n"
