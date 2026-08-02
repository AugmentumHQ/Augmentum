"""Staged-apply tests — the take-it-live path against a real temp git clone +
a fake live tree. Proves: pending = clone diff since baseline (served subtrees
only), apply materializes HEAD content + checkpoints the prior content, restore
puts it back, and added files revert by deletion."""

from __future__ import annotations

import subprocess

from augmentum.selfedit import apply as A


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _make_clone(tmp_path):
    """A git repo standing in for the --no-checkout clone (content via git show)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _write(repo, "augmentum/foo.py", "v1\n")
    _write(repo, "ui/scripts/bar.js", "old\n")
    _write(repo, "tests/test_x.py", "t\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _make_live(tmp_path):
    live = tmp_path / "live"
    _write(live, "augmentum/foo.py", "v1\n")
    _write(live, "ui/scripts/bar.js", "old\n")
    return live


async def test_pending_empty_at_baseline(tmp_path):
    repo = _make_clone(tmp_path)
    live = _make_live(tmp_path)
    await A.ensure_baseline(str(repo))
    pending = await A.compute_pending(str(repo), str(live))
    assert not pending.files
    assert pending.baseline == pending.head


async def test_pending_lists_served_changes_only(tmp_path):
    repo = _make_clone(tmp_path)
    live = _make_live(tmp_path)
    await A.ensure_baseline(str(repo))
    # a "kept" edit lands on main: backend + a non-served (tests/) file
    _write(repo, "augmentum/foo.py", "v2\n")
    _write(repo, "tests/test_x.py", "t2\n")
    _git(repo, "commit", "-aqm", "kept")
    pending = await A.compute_pending(str(repo), str(live))
    paths = [f.path for f in pending.files]
    assert "augmentum/foo.py" in paths
    assert "tests/test_x.py" not in paths        # not a served subtree
    assert all(f.applyable for f in pending.files)  # live/augmentum is writable


async def test_apply_materializes_and_checkpoints(tmp_path):
    repo = _make_clone(tmp_path)
    live = _make_live(tmp_path)
    await A.ensure_baseline(str(repo))
    _write(repo, "augmentum/foo.py", "v2\n")
    _git(repo, "commit", "-aqm", "kept")

    res = await A.apply_pending(str(repo), str(live), conn=None, user_id="")
    assert res.applied == ["augmentum/foo.py"]
    assert (live / "augmentum/foo.py").read_text() == "v2\n"     # live now updated
    assert res.checkpoint_id

    # the checkpoint captured the PRIOR (v1) content
    cps = A.list_checkpoints(str(repo))
    assert len(cps) == 1 and cps[0]["id"] == res.checkpoint_id

    # baseline advanced → nothing pending now
    again = await A.compute_pending(str(repo), str(live))
    assert not again.files


async def test_restore_reverts_to_prior(tmp_path):
    repo = _make_clone(tmp_path)
    live = _make_live(tmp_path)
    await A.ensure_baseline(str(repo))
    _write(repo, "augmentum/foo.py", "v2\n")
    _git(repo, "commit", "-aqm", "kept")
    res = await A.apply_pending(str(repo), str(live), conn=None, user_id="")
    assert (live / "augmentum/foo.py").read_text() == "v2\n"

    out = await A.restore_checkpoint(str(repo), str(live), checkpoint_id=res.checkpoint_id)
    assert "augmentum/foo.py" in out["restored"]
    assert (live / "augmentum/foo.py").read_text() == "v1\n"     # back to prior
    # baseline rolled back → the change is pending again
    pending = await A.compute_pending(str(repo), str(live))
    assert "augmentum/foo.py" in [f.path for f in pending.files]


async def test_added_file_reverts_by_deletion(tmp_path):
    repo = _make_clone(tmp_path)
    live = _make_live(tmp_path)
    await A.ensure_baseline(str(repo))
    _write(repo, "augmentum/new_mod.py", "brand new\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add")
    res = await A.apply_pending(str(repo), str(live), conn=None, user_id="")
    assert (live / "augmentum/new_mod.py").exists()

    out = await A.restore_checkpoint(str(repo), str(live), checkpoint_id=res.checkpoint_id)
    assert "augmentum/new_mod.py" in out["removed"]
    assert not (live / "augmentum/new_mod.py").exists()          # creation undone


async def test_blocked_subtree_is_reported_not_applied(tmp_path, monkeypatch):
    repo = _make_clone(tmp_path)
    live = _make_live(tmp_path)
    await A.ensure_baseline(str(repo))
    _write(repo, "ui/scripts/bar.js", "new\n")
    _git(repo, "commit", "-aqm", "kept ui")

    # simulate the ui:ro mount — ui not writable
    real = A.subtree_writable
    monkeypatch.setattr(A, "subtree_writable", lambda lv, s: False if s == "ui" else real(lv, s))
    pending = await A.compute_pending(str(repo), str(live))
    uifile = next(f for f in pending.files if f.path == "ui/scripts/bar.js")
    assert not uifile.applyable and "read-only" in uifile.reason

    res = await A.apply_pending(str(repo), str(live), conn=None, user_id="")
    assert res.applied == []                                     # nothing written
    assert any(s["path"] == "ui/scripts/bar.js" for s in res.skipped)
    assert (live / "ui/scripts/bar.js").read_text() == "old\n"    # untouched


async def test_boot_sync_baseline_ignores_host_commits(tmp_path):
    """A normal host commit landing since the last boot must NOT show as a pending
    self-edit: boot re-syncs the baseline to the clone HEAD (mirrors the clone
    reset prepare_writable_repo does), so only THIS session's cherry-picks pend."""
    repo = _make_clone(tmp_path)
    live = _make_live(tmp_path)
    await A.ensure_baseline(str(repo))                 # baseline = base commit
    # a normal (non-self-edit) commit advances the repo, as if dev work landed
    _write(repo, "augmentum/foo.py", "host-change\n")
    _git(repo, "commit", "-aqm", "host work")
    # if we kept the stale baseline, this would look pending:
    stale = await A.compute_pending(str(repo), str(live))
    assert stale.files                                  # (the stale-baseline bug)
    # boot-sync advances the baseline to HEAD → nothing pending
    await A.sync_baseline_to_head(str(repo))
    fresh = await A.compute_pending(str(repo), str(live))
    assert not fresh.files


def test_docker_http_base_from_env(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://docker-proxy:2375")
    assert A._docker_http_base() == "http://docker-proxy:2375"
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    assert A._docker_http_base() == ""
