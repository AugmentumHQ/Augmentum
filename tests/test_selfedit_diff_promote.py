"""Tests for apply-diff promotion (apply verified file changes/removals to live).

Pure file ops on temp dirs — fully verifiable here (no Docker/token/git needed for
the apply core). Load-bearing:
  - changed files (new + modified) are written to live; removed files deleted;
  - a snapshot is taken FIRST so rollback.restore_tree can undo the apply;
  - path-traversal entries are refused;
  - classify_porcelain splits M/A (changed) vs D (removed), handles renames.
"""

from __future__ import annotations

import os

from augmentum.selfedit import rollback
from augmentum.selfedit.diff_promote import apply_diff, classify_porcelain


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# --- classify --------------------------------------------------------------

def test_classify_porcelain_splits_and_handles_rename():
    changed, removed = classify_porcelain([
        " M augmentum/a.py",
        "A  augmentum/new.py",
        " D augmentum/gone.py",
        "R  augmentum/old.py -> augmentum/renamed.py",
    ])
    assert "augmentum/a.py" in changed
    assert "augmentum/new.py" in changed
    assert "augmentum/renamed.py" in changed       # rename → destination is 'changed'
    assert removed == ["augmentum/gone.py"]


def test_classify_porcelain_refuses_traversal():
    changed, removed = classify_porcelain([" M ../../etc/passwd", "A  ok.py"])
    assert changed == ["ok.py"] and removed == []


# --- apply -----------------------------------------------------------------

def test_apply_writes_changed_and_deletes_removed(tmp_path):
    cand = tmp_path / "cand"
    live = tmp_path / "live"
    _write(str(cand / "a.py"), "new-a")          # modified in candidate
    _write(str(cand / "sub" / "b.py"), "added-b")  # added in candidate
    _write(str(live / "a.py"), "old-a")          # exists in live (to be overwritten)
    _write(str(live / "stale.py"), "remove-me")  # to be removed

    res = apply_diff(str(cand), str(live), changed=["a.py", "sub/b.py"],
                     removed=["stale.py"])
    assert res.applied
    assert _read(str(live / "a.py")) == "new-a"            # modified applied
    assert _read(str(live / "sub" / "b.py")) == "added-b"  # added applied (nested)
    assert not (live / "stale.py").exists()                 # removal applied
    assert set(res.written) == {"a.py", "sub/b.py"} and res.removed == ["stale.py"]


def test_apply_snapshots_first_so_rollback_can_undo(tmp_path):
    cand = tmp_path / "cand"
    live = tmp_path / "live"
    data = tmp_path / "data"
    _write(str(cand / "x.py"), "changed")
    _write(str(live / "x.py"), "original")

    res = apply_diff(str(cand), str(live), changed=["x.py"], removed=[],
                     data_dir=str(data))
    assert res.snapshotted is True
    assert _read(str(live / "x.py")) == "changed"          # applied
    # the parachute restores the pre-apply state
    assert rollback.restore_tree(str(data), str(live)) is True
    assert _read(str(live / "x.py")) == "original"         # undone via snapshot


def test_apply_without_data_dir_skips_snapshot_but_still_applies(tmp_path):
    cand = tmp_path / "cand"
    live = tmp_path / "live"
    _write(str(cand / "y.py"), "v2")
    _write(str(live / "y.py"), "v1")
    res = apply_diff(str(cand), str(live), changed=["y.py"], removed=[])
    assert res.applied and res.snapshotted is False
    assert _read(str(live / "y.py")) == "v2"


def test_apply_refuses_traversal_paths(tmp_path):
    cand = tmp_path / "cand"
    live = tmp_path / "live"
    _write(str(cand / "ok.py"), "ok")
    res = apply_diff(str(cand), str(live), changed=["../escape.py", "ok.py"], removed=[])
    assert res.written == ["ok.py"]                         # traversal skipped
    assert not (tmp_path / "escape.py").exists()
