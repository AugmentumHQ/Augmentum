"""Files ingester — populates the ``files`` catalog.

Walks a small set of project directories (the ones other ingesters
care about) and upserts (path, mtime, sha, lang, subsystem) for each.

Lang detection: by suffix (.py, .js, .sql, .md, .css, .json).
Subsystem detection: 3rd path segment for ``augmentum/<subsystem>/...``,
``ui/scripts/<subsystem>/...``; otherwise None. Used by diagnosis to
cluster findings.

mtime/sha gating: rows are only re-touched if mtime AND sha differ
from what's stored. This keeps full-rebuild cost low even on a clean
working tree (just stat calls).
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

# Directories ingested. Ordered for deterministic walk; add to the
# end as new ingesters need new sources.
SCAN_DIRS = (
    "augmentum",
    "ui/scripts",
    "ui/styles",
    "tests",
    "docs",
)
# Files in the repo root that other ingesters reference.
SCAN_ROOT_FILES = (
    "CLAUDE.md",
)
# Skill files (relative to project_root)
SCAN_SKILL_FILES = (
    ".claude/skills/augmentum-dev/SKILL.md",
)

LANG_BY_SUFFIX = {
    ".py":   "python",
    ".js":   "javascript",
    ".mjs":  "javascript",
    ".ts":   "typescript",
    ".sql":  "sql",
    ".md":   "markdown",
    ".css":  "css",
    ".json": "json",
    ".html": "html",
    ".sh":   "shell",
}


def _lang(path: Path) -> str | None:
    return LANG_BY_SUFFIX.get(path.suffix.lower())


# Known augmentum subsystems — populated lazily from the project root.
# Used by _subsystem() to avoid assigning noise categories to test files.
_KNOWN_SUBSYSTEMS: set[str] | None = None


def _load_subsystems(project_root: Path) -> set[str]:
    """Scan augmentum/ for known subsystem directory names."""
    aug_dir = project_root / "augmentum"
    if not aug_dir.is_dir():
        return set()
    return {
        d.name for d in aug_dir.iterdir()
        if d.is_dir() and d.name != "__pycache__" and not d.name.startswith(".")
    }


def _subsystem(rel: Path, project_root: Path | None = None) -> str | None:
    """Derive a feature-area label from a project-relative path.

    augmentum/proxy/<X>_routes.py → "<X>" (the route file IS the
    feature signature; bucketing them all as "proxy" loses the
    diagnostic granularity). Other augmentum/ files use the second
    path segment. ui/scripts/<area>/... uses <area>. Test files
    derive from the stem (test_voice_routes.py → "voice") or the
    parent dir (tests/voice/ → "voice"). Everything else returns None.
    """
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "augmentum" and parts[1] == "proxy":
        # *_routes.py → subsystem is the stem minus trailing "_routes"
        stem = parts[2].removesuffix(".py")
        if stem.endswith("_routes"):
            return stem.removesuffix("_routes")
        return "proxy"
    if len(parts) >= 2 and parts[0] == "augmentum":
        return parts[1] if parts[1] != "__pycache__" else None
    if len(parts) >= 3 and parts[0] == "ui" and parts[1] == "scripts":
        return parts[2] if not parts[2].endswith(".js") else None
    # Test files: use parent dir name if it's a known augmentum subsystem.
    # For root-level tests, derive from stem only if it matches a real subsystem.
    if len(parts) >= 2 and parts[0] == "tests":
        known = _KNOWN_SUBSYSTEMS or set()
        # If in a named subdirectory (tests/coder/test_*.py → "coder")
        if len(parts) >= 3:
            candidate = parts[1]
            if candidate in known:
                return candidate
            return None
        # Root-level: test_voice_routes.py → "voice" if voice is a real subsystem
        stem = rel.stem
        if stem.startswith("test_"):
            stem = stem[5:]
            if "_" in stem:
                candidate = stem.split("_")[0]
                if candidate in known:
                    return candidate
        return None


def _sha_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _candidate_files(project_root: Path):
    for d in SCAN_DIRS:
        base = project_root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts or p.suffix in (".pyc",):
                continue
            if _lang(p) is None:
                continue
            yield p
    for f in SCAN_ROOT_FILES:
        p = project_root / f
        if p.is_file():
            yield p
    for f in SCAN_SKILL_FILES:
        p = project_root / f
        if p.is_file():
            yield p


def _cascade_delete(
    db: sqlite3.Connection, table: str, where_sql: str, params: tuple, depth: int = 0,
) -> None:
    """DELETE rows matching ``where_sql`` from ``table``, removing rows in
    any table that (transitively) REFERENCES it first. The FK graph comes
    from PRAGMA foreign_key_list at call time; depth-capped against cycles."""
    if depth > 8:
        return
    tables = [r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'",
    ).fetchall()]
    for child in tables:
        for fk in db.execute(f"PRAGMA foreign_key_list('{child}')").fetchall():
            # fk: (id, seq, parent_table, from_col, to_col, ...)
            if str(fk[2]).lower() != table.lower():
                continue
            parent_col = fk[4] or "id"
            sub = f"SELECT {parent_col} FROM {table} WHERE {where_sql}"
            _cascade_delete(db, child, f"{fk[3]} IN ({sub})", params, depth + 1)
    db.execute(f"DELETE FROM {table} WHERE {where_sql}", params)


def ingest(project_root: Path, db: sqlite3.Connection) -> None:
    """Refresh the ``files`` table. mtime/sha-gated."""
    global _KNOWN_SUBSYSTEMS
    if _KNOWN_SUBSYSTEMS is None:
        _KNOWN_SUBSYSTEMS = _load_subsystems(project_root)
    now = time.time()
    seen_ids: set[int] = set()
    # Pull existing state for fast comparison.
    existing = {
        row["path"]: (row["id"], row["mtime"], row["sha"])
        for row in db.execute("SELECT id, path, mtime, sha FROM files")
    }

    db.execute("BEGIN")
    try:
        for abs_path in _candidate_files(project_root):
            try:
                rel = abs_path.relative_to(project_root)
            except ValueError:
                continue
            rel_str = rel.as_posix()
            mtime = abs_path.stat().st_mtime
            prev = existing.get(rel_str)
            if prev is not None and prev[1] == mtime:
                # mtime unchanged → assume content unchanged. Skip sha.
                seen_ids.add(prev[0])
                continue
            sha = _sha_short(abs_path)
            lang = _lang(abs_path)
            sub = _subsystem(rel)
            if prev is not None and prev[2] == sha:
                # mtime touched (e.g. tar restore) but content same.
                # Still re-evaluate subsystem + lang — source heuristics
                # may have improved (e.g. test files now get subsystem).
                db.execute(
                    "UPDATE files SET mtime = ?, lang = ?, subsystem = ?, last_ingest_ts = ? WHERE id = ?",
                    (mtime, lang, sub, now, prev[0]),
                )
                seen_ids.add(prev[0])
                continue
            db.execute(
                """INSERT INTO files (path, mtime, sha, lang, subsystem, last_ingest_ts)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET
                       mtime = excluded.mtime,
                       sha = excluded.sha,
                       lang = excluded.lang,
                       subsystem = excluded.subsystem,
                       last_ingest_ts = excluded.last_ingest_ts""",
                (rel_str, mtime, sha, lang, sub, now),
            )
            # ON CONFLICT doesn't return lastrowid reliably; lookup by path.
            row = db.execute("SELECT id FROM files WHERE path = ?", (rel_str,)).fetchone()
            if row:
                seen_ids.add(int(row["id"]))

        # Drop rows for files that have disappeared from disk — children
        # first. Downstream tables REFERENCE files(id) without ON DELETE
        # CASCADE and their own ingesters only clean up in LATER
        # transactions, so a bare parent DELETE hits the FK constraint and
        # wedges every refresh (seen live 2026-07-10 when deleted syn_*
        # intent files stranded test_files/registrations rows). The FK
        # graph is discovered at runtime so a new child table can't
        # silently reintroduce the class.
        if seen_ids:
            placeholders = ",".join("?" for _ in seen_ids)
            _cascade_delete(
                db, "files", f"id NOT IN ({placeholders})", tuple(seen_ids),
            )
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
