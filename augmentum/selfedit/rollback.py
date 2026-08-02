"""The rollback floor — the parachute that stops a self-edit from bricking the box.

Two layers (Decision 3 — grounded in the real substrate: prod image has no
``.git``; the app container can never run git; ``/data`` is the one always-RW path;
Docker ``restart: unless-stopped`` + autoheal is the respawner):

* **L1 (git-native, smart).** A ``last_good`` SHA pointer the promote path advances
  on a confirmed-healthy boot, used to ``git revert`` a regression from the B1
  container. The git side lives in ``promote``; here is the pointer it pairs with.
* **L2 (zero-dependency, dumb).** A plain-file snapshot of known-good
  ``augmentum/`` under ``/data`` plus a boot-attempt counter. The entrypoint —
  which has NO git, NO network, NO other container, only ``/data`` — increments
  the counter each boot; the app resets it + re-snapshots once startup is healthy.
  If the counter crosses a threshold with no healthy stamp, the entrypoint restores
  the snapshot (plain copy) and clears the counter. Docker's respawn makes the
  counter climb deterministically, so the floor fires with no external watcher.
  Git is the ARCHIVE; this snapshot is the PARACHUTE.

Everything here is plain file/dir I/O under a data dir, so the whole floor is unit-
testable without Docker. The entrypoint shim calls these (a few lines of
bash+``python -c``); the app calls :func:`mark_boot_healthy` once startup
completes, and the promote path calls :func:`snapshot_tree` + :func:`write_last_good_ref`
after a confirmed-healthy promotion.
"""

from __future__ import annotations

import os
import shutil
import time

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".git", ".pytest_cache")


def state_dir(data_dir: str) -> str:
    """The self-edit floor's home under /data (created on demand)."""
    d = os.path.join(data_dir, "selfedit")
    os.makedirs(d, exist_ok=True)
    return d


def _counter_file(data_dir: str) -> str:
    return os.path.join(state_dir(data_dir), "boot_attempts")


def _ref_file(data_dir: str) -> str:
    return os.path.join(state_dir(data_dir), "last_good_ref")


def _healthy_file(data_dir: str) -> str:
    return os.path.join(state_dir(data_dir), "last_healthy_at")


def _snapshot_dir(data_dir: str) -> str:
    return os.path.join(state_dir(data_dir), "last_good")


# --- L2: boot-attempt counter (the crash-loop guard) ------------------------

def _read_int(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return int((f.read() or "0").strip() or "0")
    except (FileNotFoundError, ValueError):
        return 0
    except Exception as exc:  # noqa: BLE001 — a corrupt counter must not wedge boot
        log.warning("selfedit_counter_read_failed", path=path, error=repr(exc))
        return 0


def record_boot_attempt(data_dir: str) -> int:
    """Increment + return the boot-attempt counter. The entrypoint calls this
    BEFORE launching the app; a healthy startup later resets it."""
    n = _read_int(_counter_file(data_dir)) + 1
    with open(_counter_file(data_dir), "w", encoding="utf-8") as f:
        f.write(str(n))
    return n


def boot_attempts(data_dir: str) -> int:
    return _read_int(_counter_file(data_dir))


def should_rollback(data_dir: str, *, threshold: int = 3) -> bool:
    """True when the app has failed to reach a healthy startup ``threshold`` times
    in a row — the signal to deploy the parachute."""
    return _read_int(_counter_file(data_dir)) >= max(1, threshold)


def mark_boot_healthy(data_dir: str, *, ref: str = "") -> None:
    """Called once the app's startup completes cleanly: reset the crash counter,
    stamp the time, and (optionally) advance the last-good ref. This is what makes
    the counter mean 'consecutive *failed* boots'."""
    with open(_counter_file(data_dir), "w", encoding="utf-8") as f:
        f.write("0")
    with open(_healthy_file(data_dir), "w", encoding="utf-8") as f:
        f.write(str(time.time()))
    if ref:
        write_last_good_ref(data_dir, ref)


def last_healthy_at(data_dir: str) -> float:
    try:
        with open(_healthy_file(data_dir), encoding="utf-8") as f:
            return float((f.read() or "0").strip() or "0")
    except (FileNotFoundError, ValueError):
        return 0.0


# --- L1: last-good SHA pointer ----------------------------------------------

def write_last_good_ref(data_dir: str, sha: str) -> None:
    with open(_ref_file(data_dir), "w", encoding="utf-8") as f:
        f.write((sha or "").strip())


def read_last_good_ref(data_dir: str) -> str:
    try:
        with open(_ref_file(data_dir), encoding="utf-8") as f:
            return (f.read() or "").strip()
    except FileNotFoundError:
        return ""


# --- the parachute: a plain-file snapshot of known-good source --------------

def snapshot_tree(src_dir: str, data_dir: str) -> bool:
    """Copy known-good source (e.g. ``/app/augmentum``) into the /data snapshot,
    replacing any prior one. Plain files — no git, restorable by the entrypoint
    with nothing but a copy. Returns True on success."""
    if not os.path.isdir(src_dir):
        log.warning("selfedit_snapshot_src_missing", src=src_dir)
        return False
    dst = _snapshot_dir(data_dir)
    tmp = dst + ".new"
    try:
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src_dir, tmp, ignore=_IGNORE)
        # atomic-ish swap: remove old, rename new into place
        if os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True)
        os.replace(tmp, dst)
        return True
    except Exception as exc:  # noqa: BLE001 — a failed snapshot must not crash the app
        log.warning("selfedit_snapshot_failed", src=src_dir, error=repr(exc))
        shutil.rmtree(tmp, ignore_errors=True)
        return False


def has_snapshot(data_dir: str) -> bool:
    d = _snapshot_dir(data_dir)
    return os.path.isdir(d) and bool(os.listdir(d))


def restore_tree(data_dir: str, dst_dir: str) -> bool:
    """Restore the snapshot over ``dst_dir`` (e.g. ``/app/augmentum``). The L2
    parachute — a plain copy, callable from the entrypoint with no git/network.
    Returns True if a snapshot existed and was restored."""
    src = _snapshot_dir(data_dir)
    if not has_snapshot(data_dir):
        return False
    try:
        shutil.copytree(src, dst_dir, ignore=_IGNORE, dirs_exist_ok=True)
        log.info("selfedit_snapshot_restored", dst=dst_dir)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("selfedit_restore_failed", dst=dst_dir, error=repr(exc))
        return False


def clear_boot_counter(data_dir: str) -> None:
    """Reset the counter after the entrypoint deploys the parachute (so the
    restored-known-good boot starts from zero)."""
    with open(_counter_file(data_dir), "w", encoding="utf-8") as f:
        f.write("0")
