"""Receipt substrate tests.

Receipts persist per-action evidence to ``.augmentum/bug_finder/
receipts.jsonl``. Tests cover:

* Append/load round-trip preserves all fields.
* Multiple writes accumulate; ordering preserved.
* Lookup by finding_id and by path returns the right subset.
* SWD result → receipts adapter populates hashes correctly.
* TrustStatus correctly detects when a previously-verified fix is
  still in place vs. has been reverted.
"""

from __future__ import annotations

from pathlib import Path

from augmentum.bug_finder.receipts import (
    Receipt,
    append_receipt,
    append_receipts,
    check_fix_still_in_place,
    load_receipts,
    receipts_for_finding,
    receipts_for_path,
    receipts_from_swd_result,
)
from augmentum.bug_finder.swd import (
    ActionIntent,
    ActionOp,
    FileAction,
    SWDEngine,
)


# ---------------------------------------------------------------------------
# Append / load
# ---------------------------------------------------------------------------


def test_append_and_load_round_trip(tmp_path: Path) -> None:
    r = Receipt(
        finding_id="fnd_1", run_id="r1", op="modify",
        path="src/a.py", intent="mutate",
        status="verified",
        pre_hash="aaa", post_hash="bbb",
        pre_existed=True, post_existed=True,
        pre_size=100, post_size=120,
        model_id="openai:gpt-5", provider="openai",
        claim_signature="sql_injection",
        reason="parameterize query",
    )
    append_receipt(tmp_path, r)
    rows = load_receipts(tmp_path)
    assert len(rows) == 1
    loaded = rows[0]
    assert loaded.finding_id == "fnd_1"
    assert loaded.pre_hash == "aaa"
    assert loaded.post_hash == "bbb"
    assert loaded.model_id == "openai:gpt-5"
    assert loaded.claim_signature == "sql_injection"


def test_append_fills_timestamp_when_zero(tmp_path: Path) -> None:
    r = Receipt(
        finding_id="x", run_id="r1", op="read", path="a.py",
        intent="noop", status="noop",
    )
    append_receipt(tmp_path, r)
    loaded = load_receipts(tmp_path)[0]
    assert loaded.ts > 0


def test_load_respects_limit(tmp_path: Path) -> None:
    for i in range(15):
        append_receipt(tmp_path, Receipt(
            finding_id=f"f{i}", run_id="r", op="modify",
            path=f"src/{i}.py", intent="mutate", status="verified",
        ))
    rows = load_receipts(tmp_path, limit=5)
    assert len(rows) == 5
    assert rows[-1].finding_id == "f14"


def test_append_receipts_batch(tmp_path: Path) -> None:
    batch = [
        Receipt(finding_id="a", run_id="r", op="modify",
                path="a.py", intent="mutate", status="verified"),
        Receipt(finding_id="b", run_id="r", op="modify",
                path="b.py", intent="mutate", status="verified"),
    ]
    append_receipts(tmp_path, batch)
    rows = load_receipts(tmp_path)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def test_receipts_for_finding_filters(tmp_path: Path) -> None:
    for fid in ("fnd_1", "fnd_2", "fnd_1"):
        append_receipt(tmp_path, Receipt(
            finding_id=fid, run_id="r", op="modify",
            path="a.py", intent="mutate", status="verified",
        ))
    rows = receipts_for_finding(tmp_path, "fnd_1")
    assert len(rows) == 2
    assert all(r.finding_id == "fnd_1" for r in rows)


def test_receipts_for_path_filters(tmp_path: Path) -> None:
    for path in ("src/a.py", "src/b.py", "src/a.py"):
        append_receipt(tmp_path, Receipt(
            finding_id="f", run_id="r", op="modify",
            path=path, intent="mutate", status="verified",
        ))
    rows = receipts_for_path(tmp_path, "src/a.py")
    assert len(rows) == 2


def test_receipts_for_path_normalizes_windows_separators(tmp_path: Path) -> None:
    append_receipt(tmp_path, Receipt(
        finding_id="f", run_id="r", op="modify",
        path="src/a.py", intent="mutate", status="verified",
    ))
    # Query with backslashes — should still match
    assert len(receipts_for_path(tmp_path, "src\\a.py")) == 1


# ---------------------------------------------------------------------------
# SWD adapter
# ---------------------------------------------------------------------------


def test_receipts_from_swd_result_carries_hashes(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_bytes(b"x = 1\n")
    res = SWDEngine(workspace_root=tmp_path).run([FileAction(
        op=ActionOp.MODIFY.value, path="x.py",
        intent=ActionIntent.MUTATE.value,
        content=b"x = 2\n",
        finding_id="fnd_42", reason="constant bump",
    )])
    receipts = receipts_from_swd_result(
        res, run_id="run_abc", model_id="anthropic:claude-opus-4-7",
        provider="anthropic", claim_signature="logic_error",
        git_head="deadbeef",
    )
    assert len(receipts) == 1
    r = receipts[0]
    assert r.finding_id == "fnd_42"
    assert r.run_id == "run_abc"
    assert r.status == "verified"
    assert r.pre_hash and r.post_hash
    assert r.pre_hash != r.post_hash
    assert r.model_id == "anthropic:claude-opus-4-7"
    assert r.git_head == "deadbeef"


# ---------------------------------------------------------------------------
# TrustStatus — "is the fix we landed still on disk?"
# ---------------------------------------------------------------------------


def _seed_verified_fix(workspace: Path, finding_id: str = "fnd_X") -> str:
    target = workspace / "patched.py"
    target.write_bytes(b"x = 1\n")
    res = SWDEngine(workspace_root=workspace).run([FileAction(
        op=ActionOp.MODIFY.value, path="patched.py",
        intent=ActionIntent.MUTATE.value, content=b"x = safe()\n",
        finding_id=finding_id,
    )])
    receipts = receipts_from_swd_result(
        res, run_id="r", model_id="test", claim_signature="injection",
    )
    append_receipts(workspace, receipts)
    return receipts[0].post_hash


def test_trust_in_place_when_file_untouched(tmp_path: Path) -> None:
    _seed_verified_fix(tmp_path, finding_id="fnd_T1")
    status = check_fix_still_in_place(tmp_path, "fnd_T1", "patched.py")
    assert status.in_place
    assert status.current_hash == status.last_post_hash


def test_trust_drifts_when_file_reverted(tmp_path: Path) -> None:
    _seed_verified_fix(tmp_path, finding_id="fnd_T2")
    # Simulate developer reverting the fix
    (tmp_path / "patched.py").write_bytes(b"x = 1\n")
    status = check_fix_still_in_place(tmp_path, "fnd_T2", "patched.py")
    assert not status.in_place
    assert status.current_hash != status.last_post_hash


def test_trust_false_when_no_verified_receipt(tmp_path: Path) -> None:
    status = check_fix_still_in_place(tmp_path, "fnd_missing", "any.py")
    assert not status.in_place


def test_trust_false_when_file_deleted(tmp_path: Path) -> None:
    _seed_verified_fix(tmp_path, finding_id="fnd_T3")
    (tmp_path / "patched.py").unlink()
    status = check_fix_still_in_place(tmp_path, "fnd_T3", "patched.py")
    assert not status.in_place
