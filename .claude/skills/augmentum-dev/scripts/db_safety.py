#!/usr/bin/env python3
"""Augmentum DB-safety scanner — static checks for SQLite footguns.

Born from the recurring `augmentum.db` corruption + recovery cycle (multiple
`*.corrupt*` snapshots, the `fix(sqlite): ...` commit series). These are the
patterns that either *cause* corruption or *amplify the blast radius* of a
recovery (silently dropping rows while files orphan, etc.):

  1. `AUTOINCREMENT` in a migration                 → use plain INTEGER PRIMARY KEY
  2. `CREATE TABLE/INDEX` without `IF NOT EXISTS`   → migration not re-run-safe
  3. `PRAGMA journal_mode` ≠ WAL in the state layer → loses crash-safety
  4. `shutil.copy*` / `os.rename` of a `.db` path   → torn-WAL backup; prefer VACUUM INTO
  5. `DELETE FROM <t>` with no WHERE / `DROP TABLE`  → unbounded data loss
  6. `os.remove` / `unlink` / `rmtree` of `.db` / `-wal` / `-shm`

Companion suppressions file (`db_safety_suppressions.json`) skips the
known-acceptable findings (intentional table rebuilds, the forensic copies in
the recovery path, legacy migrations). Run `--verbose` to see suppressed
entries. Exit 0 = clean (modulo warnings), 1 = errors found.

This is the *static* half. For *live* DB health (integrity_check, WAL size,
file⟷row reconciliation), see `db_health.py`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Windows consoles/pipes default to a non-UTF-8 codepage; ✓ / → / … in our
# output would raise UnicodeEncodeError. Make stdout/stderr lenient.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: Cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)


ROOT = _find_root()
MIGRATIONS = ROOT / "augmentum" / "state" / "migrations"
STATE_DIR = ROOT / "augmentum" / "state"

_COLOR = os.environ.get("TERM") or os.name != "nt"


def _red(s: str) -> str:    return f"\033[91m{s}\033[0m" if _COLOR else s
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m" if _COLOR else s
def _green(s: str) -> str:  return f"\033[92m{s}\033[0m" if _COLOR else s
def _cyan(s: str) -> str:   return f"\033[96m{s}\033[0m" if _COLOR else s
def _bold(s: str) -> str:   return f"\033[1m{s}\033[0m" if _COLOR else s


_SUPPRESSIONS_PATH = Path(__file__).resolve().parent / "db_safety_suppressions.json"
_SUPPRESSION_KEYS = (
    "autoincrement", "create_no_if_not_exists", "non_wal_journal",
    "db_file_copy", "unbounded_delete", "db_file_delete",
)


def _load_suppressions() -> dict[str, list[str]]:
    if _SUPPRESSIONS_PATH.is_file():
        try:
            data = json.loads(_SUPPRESSIONS_PATH.read_text(encoding="utf-8"))
            return {k: list(data.get(k, [])) for k in _SUPPRESSION_KEYS}
        except (json.JSONDecodeError, KeyError):
            pass
    default = {k: [] for k in _SUPPRESSION_KEYS}
    _SUPPRESSIONS_PATH.write_text(json.dumps(default, indent=2) + "\n", encoding="utf-8")
    return default


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _suppressed(entries: list[str], rel_path: str, line_no: int | None = None) -> bool:
    """An entry matches if it's the path, or `path:line`, or a prefix of the path."""
    keys = {rel_path}
    if line_no is not None:
        keys.add(f"{rel_path}:{line_no}")
    for e in entries:
        if e in keys or rel_path.startswith(e.rstrip("/") + "/") or rel_path == e:
            return True
    return False


def _strip_sql_comments(text: str) -> str:
    text = re.sub(r"--[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _migration_files() -> list[Path]:
    return sorted(MIGRATIONS.glob("*.sql")) if MIGRATIONS.is_dir() else []


def _py_files(*subdirs: str) -> list[Path]:
    out: list[Path] = []
    for sd in subdirs:
        base = ROOT / sd
        if base.is_dir():
            out += [p for p in base.rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(out)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_autoincrement(sup: list[str], verbose: bool) -> tuple[list[str], int]:
    findings, suppressed = [], 0
    for f in _migration_files():
        sql = _strip_sql_comments(f.read_text(encoding="utf-8", errors="replace"))
        if not re.search(r"\bAUTOINCREMENT\b", sql, re.IGNORECASE):
            continue
        rel = _rel(f)
        line = next((i for i, ln in enumerate(f.read_text(errors="replace").splitlines(), 1)
                     if re.search(r"\bAUTOINCREMENT\b", ln, re.IGNORECASE)), 1)
        if _suppressed(sup, rel, line) or _suppressed(sup, rel):
            suppressed += 1
            if verbose:
                findings.append(f"  [suppressed] {rel}:{line}  AUTOINCREMENT")
        else:
            findings.append(_yellow(f"  ! {rel}:{line}  AUTOINCREMENT — use plain `INTEGER PRIMARY KEY` (rowid reuse is fine; AUTOINCREMENT adds sqlite_sequence overhead and was implicated in past corruption)"))
    return findings, suppressed


def check_create_idempotent(sup: list[str], verbose: bool) -> tuple[list[str], int]:
    findings, suppressed = [], 0
    pat = re.compile(r"\bCREATE\s+(?:UNIQUE\s+)?(TABLE|INDEX|VIRTUAL\s+TABLE)\s+(?!IF\s+NOT\s+EXISTS)([A-Za-z_][\w.]*)", re.IGNORECASE)
    for f in _migration_files():
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        blob = _strip_sql_comments("\n".join(lines))
        for m in pat.finditer(blob):
            kind = m.group(1).upper().replace("VIRTUAL TABLE", "VIRTUAL TABLE")
            name = m.group(2)
            # Find line number (best-effort).
            line = next((i for i, ln in enumerate(lines, 1) if re.search(rf"CREATE\s+(?:UNIQUE\s+)?{re.escape(kind.split()[0])}\b", ln, re.IGNORECASE) and name in ln), 1)
            rel = _rel(f)
            if _suppressed(sup, rel, line) or _suppressed(sup, rel):
                suppressed += 1
                if verbose:
                    findings.append(f"  [suppressed] {rel}:{line}  CREATE {kind} {name} (no IF NOT EXISTS)")
            else:
                findings.append(_yellow(f"  ! {rel}:{line}  `CREATE {kind} {name}` without `IF NOT EXISTS` — migration won't survive a re-run / partial-apply"))
    return findings, suppressed


def check_journal_mode(sup: list[str], verbose: bool) -> tuple[list[str], int]:
    """Anything in the state layer that sets journal_mode to something other
    than WAL. (Side DBs — knowledge packs, ZIM readers — may legitimately use
    OFF/MEMORY for ephemeral builds; only the main app DB layer is in scope.)"""
    findings, suppressed = [], 0
    pat = re.compile(r"journal_mode\s*=\s*['\"]?\s*(\w+)", re.IGNORECASE)
    for f in _py_files("augmentum/state"):
        for i, ln in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            m = pat.search(ln)
            if not m:
                continue
            mode = m.group(1).upper()
            if mode == "WAL":
                continue
            rel = _rel(f)
            if _suppressed(sup, rel, i) or _suppressed(sup, rel):
                suppressed += 1
                if verbose:
                    findings.append(f"  [suppressed] {rel}:{i}  journal_mode={mode}")
            else:
                findings.append(_red(f"  x {rel}:{i}  journal_mode={mode} in the state layer — the main DB must run in WAL"))
    return findings, suppressed


def check_db_file_copy(sup: list[str], verbose: bool) -> tuple[list[str], int]:
    """shutil.copy*/copyfileobj/os.rename where a `.db` path is in the line.
    Copying a *live* DB captures a torn WAL state — prefer `VACUUM INTO`. The
    forensic copies in the recovery path (a dead/quarantined DB) are fine and
    should be in the suppressions file."""
    findings, suppressed = [], 0
    pat = re.compile(r"\b(shutil\.copy\w*|shutil\.copyfileobj|os\.rename|os\.replace)\s*\(", re.IGNORECASE)
    for f in _py_files("augmentum"):
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, ln in enumerate(lines, 1):
            if not pat.search(ln):
                continue
            ctx = " ".join(lines[max(0, i - 2):i + 1]).lower()
            if not re.search(r"\.db['\"]|\.db\b|sqlite|augmentum\.db|_db_path|db_path", ctx):
                continue
            rel = _rel(f)
            if _suppressed(sup, rel, i) or _suppressed(sup, rel):
                suppressed += 1
                if verbose:
                    findings.append(f"  [suppressed] {rel}:{i}  copy/rename of a .db path")
            else:
                findings.append(_yellow(f"  ! {rel}:{i}  `{ln.strip()[:80]}` — if this is a *live* DB, use `VACUUM INTO` instead (copy captures a torn WAL); if it's a quarantined/dead DB, add to db_safety_suppressions.json"))
    return findings, suppressed


def check_unbounded_delete(sup: list[str], verbose: bool) -> tuple[list[str], int]:
    """`DELETE FROM <t>` with no WHERE, or `DROP TABLE <t>` — in migrations or
    app code. Table rebuilds legitimately DROP the `_old` table; whitelist those.

    Multi-line awareness: a DELETE may be followed by WHERE on a subsequent line
    (Python triple-quoted SQL, formatted migrations). The lookahead scans up to
    5 lines for WHERE/USING/RETURNING qualifiers before flagging.
    """
    findings, suppressed = [], 0
    del_pat = re.compile(r"\bDELETE\s+FROM\s+([A-Za-z_][\w.]*)\s*(;|$)", re.IGNORECASE)
    drop_pat = re.compile(r"\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?([A-Za-z_][\w.]*)", re.IGNORECASE)
    qualifier_pat = re.compile(r"\b(WHERE|USING|RETURNING)\b", re.IGNORECASE)
    targets = _migration_files() + _py_files("augmentum")
    for f in targets:
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        stripped = [_strip_sql_comments(ln) for ln in lines]
        for i, ln in enumerate(lines, 1):
            for pat, kind, is_delete in (
                (del_pat, "DELETE FROM (no WHERE)", True),
                (drop_pat, "DROP TABLE", False),
            ):
                m = pat.search(stripped[i - 1])
                if not m:
                    continue
                if is_delete:
                    # Look ahead up to 5 lines for a WHERE/USING/RETURNING qualifier
                    # before treating this as unbounded. Skips false positives where
                    # the WHERE clause lives on a subsequent line of a triple-quoted
                    # SQL string.
                    lookahead = "\n".join(stripped[i - 1 : min(len(stripped), i + 5)])
                    if qualifier_pat.search(lookahead[m.end():]):
                        continue
                tbl = m.group(1)
                rel = _rel(f)
                if _suppressed(sup, rel, i) or _suppressed(sup, rel):
                    suppressed += 1
                    if verbose:
                        findings.append(f"  [suppressed] {rel}:{i}  {kind} {tbl}")
                else:
                    findings.append(_yellow(f"  ! {rel}:{i}  {kind} `{tbl}` — unbounded data loss; confirm intentional (and add to suppressions if it's a table rebuild / test fixture)"))
    return findings, suppressed


def check_db_file_delete(sup: list[str], verbose: bool) -> tuple[list[str], int]:
    findings, suppressed = [], 0
    pat = re.compile(r"\b(os\.remove|os\.unlink|Path\([^)]*\)\.unlink|shutil\.rmtree)\b")
    for f in _py_files("augmentum"):
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, ln in enumerate(lines, 1):
            if not pat.search(ln):
                continue
            if not re.search(r"\.db(['\"\)]|-wal|-shm|\b)|sqlite", ln, re.IGNORECASE):
                continue
            rel = _rel(f)
            if _suppressed(sup, rel, i) or _suppressed(sup, rel):
                suppressed += 1
                if verbose:
                    findings.append(f"  [suppressed] {rel}:{i}  deletes a .db / -wal / -shm file")
            else:
                findings.append(_yellow(f"  ! {rel}:{i}  deletes a `.db`/`-wal`/`-shm` file — make sure the DB is closed first and a backup/quarantine copy exists"))
    return findings, suppressed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    sup = _load_suppressions()
    print(_bold("Augmentum DB-safety scan"))
    print(_dim(f"  migrations: {len(_migration_files())}  state-layer .py: {len(_py_files('augmentum/state'))}"))
    print()

    checks = [
        ("AUTOINCREMENT in migrations",            check_autoincrement,   "autoincrement"),
        ("CREATE TABLE/INDEX idempotency",         check_create_idempotent, "create_no_if_not_exists"),
        ("journal_mode = WAL in state layer",      check_journal_mode,    "non_wal_journal"),
        ("copy/rename of .db paths",               check_db_file_copy,    "db_file_copy"),
        ("unbounded DELETE / DROP TABLE",          check_unbounded_delete, "unbounded_delete"),
        ("deletion of .db / -wal / -shm files",    check_db_file_delete,  "db_file_delete"),
    ]
    errors: list[str] = []
    warnings: list[str] = []
    suppressed_total = 0
    for idx, (label, fn, key) in enumerate(checks, 1):
        print(_cyan(f"  [{idx}/{len(checks)}] {label}..."))
        fnd, sup_n = fn(sup[key], verbose)
        suppressed_total += sup_n
        for line in fnd:
            if "[suppressed]" in line:
                print(line)
            elif _red("x") in line:
                errors.append(line)
            else:
                warnings.append(line)

    print()
    print(_bold("  Summary"))
    print(f"    Suppressions applied: {suppressed_total}")
    print()
    if errors:
        print(_red(f"  ERRORS ({len(errors)}):"))
        for e in errors:
            print(e)
        print()
    if warnings:
        print(_yellow(f"  WARNINGS ({len(warnings)}):"))
        for w in warnings:
            print(w)
        print()
    if not errors and not warnings:
        print(_green("  All clear — no DB-safety anti-patterns detected."))
        print()

    # Same closing line shape every Augmentum scanner uses (audit.py parses it).
    if errors:
        print(_red(f"  {len(errors)} error(s), {len(warnings)} warning(s)"))
        return 1
    print(_yellow(f"  0 errors, {len(warnings)} warning(s)") if warnings
          else _green("  0 errors, 0 warnings"))
    return 0


def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m" if _COLOR else s


if __name__ == "__main__":
    sys.exit(main())
