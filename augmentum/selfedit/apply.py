"""Take collected self-edits live — the staged apply + checkpoint + restart path.

The model the operator chose: a kept edit is **staged, not instantly live**. Keep
cherry-picks the candidate's commit onto the isolated clone's main
(``promote.git_promote``), so the clone accumulates every accepted change as a
durable, reversible commit. This module is the *next* step — the human-triggered
"make the collected changes real":

* **Pending** — what's staged but not yet live: ``git diff baseline..HEAD`` in the
  clone, restricted to the subtrees the running app actually serves
  (``augmentum`` writable; ``ui`` writable only after the compose mount is flipped).
* **Checkpoint** — BEFORE writing anything, snapshot the *current* content of
  exactly the files an apply will touch (per-file, because ``ui/`` carries 500MB+
  of assets — a full-tree tar is a non-starter). This is the restore point.
* **Apply** — materialize the clone's HEAD content for each pending file into the
  live tree (``git show HEAD:path`` — the clone is ``--no-checkout``), advance the
  baseline, mark the attempts ``live``.
* **Restart** — the running process keeps its loaded bytecode until it restarts,
  so a backend change only takes effect on restart; ``restart_self`` asks the
  docker-proxy to restart this very container (confirmed allowed: ``POST=1`` +
  ``CONTAINERS=1``). Scheduled just after the response flushes, so the client
  gets its confirmation, then reconnects.
* **Revert** — restore a checkpoint's files to their pre-apply content + restart;
  the attempts it undid go to ``rolled_back`` (the lesson is kept — anti-Westworld).

Frontend (``ui/``) is read-only until the operator flips the compose mount and
recreates the container; until then those files are surfaced as pending but
flagged un-applyable rather than silently skipped.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from augmentum.selfedit import rollback as _rollback
from augmentum.selfedit import store as _store
from augmentum.selfedit.candidate import _git
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# The subtrees the running app serves (and thus can be taken live). Anything a
# self-edit touches outside these (tests/, docs/, migrations) is never applied to
# the live process by this path — it stays staged for a human/rebuild.
APPLYABLE_SUBTREES = ("augmentum", "ui")
_BASELINE_FILE = "applied_baseline"
_CHECKPOINTS_DIR = "checkpoints"
_MAX_CHECKPOINTS = 20


# --- paths / substrate ------------------------------------------------------

def live_tree(app_state: Any) -> str:
    """The running app's code root (``/app``) — where ``augmentum``/``ui`` live."""
    override = getattr(app_state, "selfedit_live_tree", "")
    if override:
        return override
    import augmentum
    return os.path.dirname(os.path.dirname(os.path.abspath(augmentum.__file__)))


def _selfedit_dir(repo_dir: str) -> str:
    """``/data/selfedit`` — the parent of the clone, our scratch/checkpoint home."""
    return os.path.dirname(repo_dir.rstrip("/")) or "/data/selfedit"


def _baseline_path(repo_dir: str) -> str:
    return os.path.join(_selfedit_dir(repo_dir), _BASELINE_FILE)


def subtree_writable(live: str, sub: str) -> bool:
    """Probe whether a live subtree can actually be written (the ui:ro gate)."""
    d = os.path.join(live, sub)
    if not os.path.isdir(d):
        return False
    probe = os.path.join(d, ".se_write_probe")
    try:
        with open(probe, "w") as f:
            f.write("x")
        os.remove(probe)
        return True
    except OSError:
        return False


def _path_subtree(path: str) -> str:
    return path.split("/", 1)[0] if path else ""


# --- baseline (what's currently live, as a clone sha) -----------------------

async def _clone_head(repo_dir: str) -> str:
    code, out = await _git(repo_dir, "rev-parse", "HEAD")
    return out.strip() if code == 0 else ""


async def ensure_baseline(repo_dir: str) -> str:
    """Record the clone HEAD as the live baseline IF not already set. Called at
    boot, BEFORE any Keep advances main — so the baseline honestly represents
    'what's running now'. Idempotent."""
    p = _baseline_path(repo_dir)
    if os.path.exists(p):
        with contextlib.suppress(OSError), open(p, encoding="utf-8") as f:
            return f.read().strip()
    head = await _clone_head(repo_dir)
    if head:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with contextlib.suppress(OSError), open(p, "w", encoding="utf-8") as f:
            f.write(head)
    return head


async def sync_baseline_to_head(repo_dir: str) -> str:
    """Force the baseline to the clone's current HEAD — call at BOOT, after
    ``prepare_writable_repo`` has reset the clone to the live source HEAD (it does:
    ``fetch HEAD`` + ``update-ref HEAD FETCH_HEAD``). Boot is a clean slate: the
    clone HEAD == what's running, and any prior un-applied cherry-picks were
    discarded by that reset. So the baseline must advance WITH it — otherwise
    normal host commits landed since the last boot masquerade as pending
    self-edits. Staged edits are session-scoped: Keep cherry-picks beyond this
    boot baseline → pending; a restart without applying drops them (the clone
    reset drops them too, so the two stay consistent)."""
    head = await _clone_head(repo_dir)
    if head:
        set_baseline(repo_dir, head)
    return head


async def get_baseline(repo_dir: str) -> str:
    p = _baseline_path(repo_dir)
    if os.path.exists(p):
        with contextlib.suppress(OSError), open(p, encoding="utf-8") as f:
            return f.read().strip()
    return await ensure_baseline(repo_dir)


def set_baseline(repo_dir: str, sha: str) -> None:
    if not sha:
        return
    p = _baseline_path(repo_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with contextlib.suppress(OSError), open(p, "w", encoding="utf-8") as f:
        f.write(sha)


# --- pending (staged-but-not-live) ------------------------------------------

@dataclass
class PendingFile:
    path: str
    change: str             # A | M | D
    subtree: str
    applyable: bool
    reason: str = ""


@dataclass
class Pending:
    baseline: str
    head: str
    files: list[PendingFile] = field(default_factory=list)
    diff: str = ""
    attempts: list[dict] = field(default_factory=list)   # promoted (staged) attempts

    @property
    def applyable_files(self) -> list[PendingFile]:
        return [f for f in self.files if f.applyable]

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline[:12], "head": self.head[:12],
            "has_changes": bool(self.files),
            "applyable_count": len(self.applyable_files),
            "blocked_count": len(self.files) - len(self.applyable_files),
            "files": [vars(f) for f in self.files],
            "diff": self.diff,
            "attempts": [{"id": a["id"], "objective": a["objective"],
                          "surface": a.get("surface", ""),
                          "tier": (a.get("gate_verdict") or {}).get("tier", ""),
                          "files": a.get("files_changed", [])}
                         for a in self.attempts],
        }


async def compute_pending(repo_dir: str, live: str, *, conn: Any = None,
                          user_id: str = "") -> Pending:
    """The staged set: files changed on the clone's main since the live baseline,
    restricted to the served subtrees, each tagged applyable (subtree writable)."""
    baseline = await get_baseline(repo_dir)
    head = await _clone_head(repo_dir)
    files: list[PendingFile] = []
    diff = ""
    if baseline and head and baseline != head:
        code, out = await _git(repo_dir, "diff", "--name-status", f"{baseline}..{head}")
        if code == 0:
            writable = {s: subtree_writable(live, s) for s in APPLYABLE_SUBTREES}
            for line in out.splitlines():
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                change, path = parts[0][:1], parts[-1].strip()
                sub = _path_subtree(path)
                if sub not in APPLYABLE_SUBTREES:
                    continue  # tests/docs/migrations — never live-applied here
                ok = writable.get(sub, False)
                reason = "" if ok else f"{sub}/ mount is read-only — recreate the container to enable"
                files.append(PendingFile(path=path, change=change, subtree=sub,
                                         applyable=ok, reason=reason))
        dcode, dout = await _git(repo_dir, "diff", "--stat", f"{baseline}..{head}")
        if dcode == 0:
            diff = dout
    attempts: list[dict] = []
    if conn is not None and user_id:
        with contextlib.suppress(Exception):
            rows = await _store.list_attempts(conn, user_id=user_id, limit=200)
            attempts = [a for a in rows if a.get("status") == "promoted"]
    return Pending(baseline=baseline, head=head, files=files, diff=diff, attempts=attempts)


# --- checkpoints (per-file restore points) ----------------------------------

async def _read_blob(repo_dir: str, ref: str, path: str) -> bytes | None:
    """Exact bytes of a tracked file at ``ref`` (the clone is --no-checkout, so we
    read from the object DB). Raw subprocess — NOT ``candidate._git``, which strips
    output and would silently drop a file's trailing newline."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "-c", "safe.directory=*", "-C", repo_dir, "show", f"{ref}:{path}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), 60)
    except Exception as exc:  # noqa: BLE001 — missing blob / git error → no content
        log.warning("selfedit_read_blob_failed", path=path, error=repr(exc))
        return None
    return out if proc.returncode == 0 else None


def _checkpoints_root(repo_dir: str) -> str:
    return os.path.join(_selfedit_dir(repo_dir), _CHECKPOINTS_DIR)


def make_checkpoint(repo_dir: str, live: str, *, files: list[str], label: str,
                    baseline_before: str, head_after: str,
                    attempt_ids: list[str]) -> dict:
    """Snapshot the CURRENT live content of exactly ``files`` (pre-apply) into a
    restore point. Captures absence too (a file the apply will create → restore
    deletes it). Returns the manifest."""
    cid = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    root = os.path.join(_checkpoints_root(repo_dir), cid)
    tree = os.path.join(root, "tree")
    os.makedirs(tree, exist_ok=True)
    captured: list[dict] = []
    for rel in files:
        src = os.path.join(live, rel)
        existed = os.path.isfile(src)
        if existed:
            dst = os.path.join(tree, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with contextlib.suppress(OSError), open(src, "rb") as r, \
                    open(dst, "wb") as w:
                w.write(r.read())
        captured.append({"path": rel, "existed": existed})
    manifest = {
        "id": cid, "label": label, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": captured, "baseline_before": baseline_before,
        "head_after": head_after, "attempt_ids": attempt_ids,
    }
    with contextlib.suppress(OSError), open(os.path.join(root, "manifest.json"),
                                            "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    _prune_checkpoints(repo_dir)
    return manifest


def list_checkpoints(repo_dir: str) -> list[dict]:
    root = _checkpoints_root(repo_dir)
    if not os.path.isdir(root):
        return []
    out: list[dict] = []
    for cid in os.listdir(root):
        mp = os.path.join(root, cid, "manifest.json")
        if os.path.isfile(mp):
            with contextlib.suppress(Exception):
                with open(mp, encoding="utf-8") as f:
                    m = json.load(f)
                m["file_count"] = len(m.get("files", []))
                out.append(m)
    out.sort(key=lambda m: m.get("created", ""), reverse=True)
    return out


def _prune_checkpoints(repo_dir: str) -> None:
    cps = list_checkpoints(repo_dir)
    import shutil
    for m in cps[_MAX_CHECKPOINTS:]:
        with contextlib.suppress(OSError):
            shutil.rmtree(os.path.join(_checkpoints_root(repo_dir), m["id"]))


# --- apply ------------------------------------------------------------------

@dataclass
class ApplyResult:
    applied: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)   # {path, reason}
    checkpoint_id: str = ""
    head: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {"applied": self.applied, "deleted": self.deleted,
                "skipped": self.skipped, "checkpoint_id": self.checkpoint_id,
                "head": self.head[:12], "error": self.error,
                "needs_restart": bool(self.applied or self.deleted)}


async def apply_pending(repo_dir: str, live: str, *, conn: Any, user_id: str,
                        label: str = "") -> ApplyResult:
    """Checkpoint, then write each applyable pending file's clone-HEAD content into
    the live tree (or delete it), advance the baseline, mark the attempts ``live``.
    Never raises — partial/blocked files are reported, not thrown."""
    pending = await compute_pending(repo_dir, live, conn=conn, user_id=user_id)
    res = ApplyResult(head=pending.head)
    applyable = pending.applyable_files
    if not applyable:
        res.skipped = [{"path": f.path, "reason": f.reason} for f in pending.files]
        return res

    attempt_ids = [a["id"] for a in pending.attempts]
    cp = make_checkpoint(repo_dir, live, files=[f.path for f in applyable],
                         label=label or "apply", baseline_before=pending.baseline,
                         head_after=pending.head, attempt_ids=attempt_ids)
    res.checkpoint_id = cp["id"]

    # L2 boot parachute (the other half of the self-edit floor, rollback.py): before
    # a BACKEND change — the only kind that can crash-loop boot — snapshot the
    # known-good augmentum/ tree so the entrypoint auto-restores it if the new code
    # fails to boot N times. (A bad frontend file can't boot-loop the server, so it's
    # covered only by the per-file checkpoint above.) The per-file checkpoint is the
    # clean app-level revert; this is the can't-even-boot floor.
    if any(f.subtree == "augmentum" for f in applyable):
        data_dir = os.path.dirname(_selfedit_dir(repo_dir))  # /data
        with contextlib.suppress(Exception):
            _rollback.snapshot_tree(os.path.join(live, "augmentum"), data_dir)
            _rollback.write_last_good_ref(data_dir, pending.baseline)

    for f in applyable:
        dst = os.path.join(live, f.path)
        try:
            if f.change == "D":
                if os.path.isfile(dst):
                    os.remove(dst)
                res.deleted.append(f.path)
                continue
            content = await _read_blob(repo_dir, pending.head, f.path)
            if content is None:
                res.skipped.append({"path": f.path, "reason": "content unavailable at HEAD"})
                continue
            os.makedirs(os.path.dirname(dst) or live, exist_ok=True)
            with open(dst, "wb") as w:
                w.write(content)
            res.applied.append(f.path)
        except OSError as exc:
            res.skipped.append({"path": f.path, "reason": repr(exc)})

    for f in pending.files:
        if not f.applyable:
            res.skipped.append({"path": f.path, "reason": f.reason})

    # Advance the baseline + mark the staged attempts live (only if something landed).
    if res.applied or res.deleted:
        set_baseline(repo_dir, pending.head)
        if conn is not None:
            for aid in attempt_ids:
                with contextlib.suppress(Exception):
                    await _store.finalize(
                        conn, attempt_id=aid, user_id=user_id, status="live",
                        outcome="applied to the live tree (pending restart)",
                        lesson="taken live by the user — reversible via checkpoint",
                        promoted_commit=pending.head)
    log.info("selfedit_apply", applied=len(res.applied), deleted=len(res.deleted),
             skipped=len(res.skipped), checkpoint=res.checkpoint_id)
    return res


async def restore_checkpoint(repo_dir: str, live: str, *, checkpoint_id: str,
                             conn: Any = None, user_id: str = "") -> dict:
    """Restore a checkpoint's files to their pre-apply content (re-creating or
    deleting as recorded), roll the baseline back, and mark its attempts
    ``rolled_back`` (the record/lesson is kept). Returns a summary."""
    root = os.path.join(_checkpoints_root(repo_dir), checkpoint_id)
    mp = os.path.join(root, "manifest.json")
    if not os.path.isfile(mp):
        return {"error": "checkpoint not found", "restored": [], "removed": []}
    with open(mp, encoding="utf-8") as f:
        manifest = json.load(f)
    tree = os.path.join(root, "tree")
    restored, removed = [], []
    for entry in manifest.get("files", []):
        rel = entry["path"]
        dst = os.path.join(live, rel)
        snap = os.path.join(tree, rel)
        try:
            if entry.get("existed") and os.path.isfile(snap):
                os.makedirs(os.path.dirname(dst) or live, exist_ok=True)
                with open(snap, "rb") as r, open(dst, "wb") as w:
                    w.write(r.read())
                restored.append(rel)
            elif not entry.get("existed"):
                # the apply CREATED this file → revert means remove it
                if os.path.isfile(dst):
                    os.remove(dst)
                removed.append(rel)
        except OSError as exc:
            log.warning("selfedit_restore_file_failed", path=rel, error=repr(exc))
    set_baseline(repo_dir, manifest.get("baseline_before", ""))
    if conn is not None and user_id:
        for aid in manifest.get("attempt_ids", []):
            with contextlib.suppress(Exception):
                await _store.finalize(
                    conn, attempt_id=aid, user_id=user_id, status="rolled_back",
                    outcome=f"reverted to checkpoint {checkpoint_id}",
                    lesson="reverted by the user — code restored, the record kept")
    log.info("selfedit_restore", checkpoint=checkpoint_id, restored=len(restored),
             removed=len(removed))
    return {"restored": restored, "removed": removed, "checkpoint_id": checkpoint_id,
            "needs_restart": bool(restored or removed)}


# --- self-restart (via the docker-proxy) ------------------------------------

def self_container_id() -> str:
    """This container's id — the Docker hostname inside the container."""
    import socket
    return socket.gethostname()


def _docker_http_base() -> str:
    host = os.environ.get("DOCKER_HOST", "")
    if host.startswith("tcp://"):
        return "http://" + host[len("tcp://"):]
    return ""


async def _post_restart() -> bool:
    base = _docker_http_base()
    if not base:
        log.warning("selfedit_restart_no_docker_host")
        return False
    cid = self_container_id()
    url = f"{base}/containers/{cid}/restart?t=2"
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url)
            return r.status_code in (204, 200, 404)  # 404 → already gone (restarting)
    except Exception as exc:  # noqa: BLE001 — the restart is best-effort
        log.warning("selfedit_restart_failed", error=repr(exc))
        return False


def schedule_restart(*, delay: float = 1.5) -> None:
    """Restart THIS container shortly — after the current response flushes so the
    client gets its confirmation and can start polling for the app's return.
    Fire-and-forget (the process will be replaced)."""
    async def _go() -> None:
        await asyncio.sleep(delay)
        log.info("selfedit_self_restart_initiated")
        await _post_restart()
    with contextlib.suppress(RuntimeError):
        asyncio.create_task(_go())
