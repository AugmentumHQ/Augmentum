"""Per-workspace code-intelligence index — symbols, imports, repo map.

The structural sibling of ``indexer.py`` (semantic/embedding search):
where the indexer answers "what code is ABOUT X", this module answers
"where IS X" — symbol definitions, file outlines, and a compact
whole-repo structure map — without a single model iteration spent on
grep/read chains.

Architecture (shares the indexer's proven conventions):
- Storage: the SAME host-side sidecar SQLite as the semantic index
  (``{data_dir}/coder_indexes/{workspace_id[:12]}.db``), in its own
  ``ci_*`` tables. No embedding dependency — pure parsing, so it works
  even when the embedding service is cold or absent.
- Reads: batched base64 container execs via ``indexer._batch_read_files``
  (one exec per ~40 files, never one ``cat`` per file).
- Incremental: mtime+size partition from one ``find`` exec — unchanged
  files are skipped without a read; content hash catches touch-only.
- Extraction: Python via ``ast`` (accurate: classes, methods, functions,
  module constants, signatures); JS/TS family via line regexes
  (functions, classes, arrow consts, interfaces/types/enums). Other
  languages still appear in the repo map as files (structure is signal
  even without symbols).

Consumers:
- ``find_symbol`` / ``file_outline`` coder tools (one-hop answers).
- ``render_repo_map`` — the KV-stable prompt carrier block. BYTE-STABLE
  by construction: no line numbers, no timestamps, no counters that
  churn on unrelated edits; ordering is deterministic (path sort,
  kind/name sort) so the rendered block only changes when the actual
  structure (files / symbol names) changes.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
import sqlite3
import time
from pathlib import Path

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Extensions we parse for symbols. Everything else in the indexer's
# _INDEXABLE_EXTS still lands in ci_files (metadata only) so the repo
# map shows the full tree shape.
_SYMBOL_EXTS = {".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}

# Max file size to parse — mirrors the indexer's ceiling; bigger files
# are almost always generated/vendored and would bloat the map anyway.
_MAX_FILE_SIZE = 200_000

# The find listing cap. The semantic indexer stops at 500 (embedding
# cost scales with content); symbol extraction is cheap, so we map more
# of the tree. When a workspace actually hits this we log it — a silent
# cap would read as "mapped everything" when it didn't.
_LIST_CAP = 2000

# Per-workspace build locks — a double trigger (route + lazy tool build)
# must coalesce, not stack two full passes.
_build_locks: dict[str, asyncio.Lock] = {}


def _db_path(workspace_id: str) -> Path:
    from augmentum.config import settings
    return Path(settings.data_dir) / "coder_indexes" / f"{workspace_id[:12]}.db"


def _open_db(workspace_id: str) -> sqlite3.Connection:
    """Open the sidecar DB and ensure the ci_* schema exists.

    Plain sqlite3 (no sqlite-vec — the ci_* tables are relational only).
    WAL + busy_timeout because the semantic indexer may hold a second
    connection to the same file during a workspace open.
    """
    path = _db_path(workspace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=5000")
    # readonly FS or locked — non-fatal, journal defaults still work
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS ci_files (
        path TEXT PRIMARY KEY,
        hash TEXT NOT NULL DEFAULT '',
        mtime TEXT,
        size INTEGER,
        lang TEXT NOT NULL DEFAULT '',
        indexed_at REAL NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS ci_symbols (
        file_path TEXT NOT NULL,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,
        scope TEXT NOT NULL DEFAULT '',
        line INTEGER NOT NULL,
        end_line INTEGER,
        signature TEXT NOT NULL DEFAULT ''
    )""")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ci_symbols_name ON ci_symbols(name)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ci_symbols_file ON ci_symbols(file_path)"
    )
    conn.execute("""CREATE TABLE IF NOT EXISTS ci_imports (
        file_path TEXT NOT NULL,
        module TEXT NOT NULL,
        name TEXT NOT NULL DEFAULT ''
    )""")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ci_imports_file ON ci_imports(file_path)"
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Extraction — pure functions on (path, content)
# ---------------------------------------------------------------------------

def _lang_for(path: str) -> str:
    p = path.lower()
    if p.endswith(".py"):
        return "python"
    if p.endswith((".ts", ".tsx")):
        return "typescript"
    if p.endswith((".js", ".jsx", ".mjs", ".cjs")):
        return "javascript"
    return ""


def extract_python(content: str) -> tuple[list[dict], list[dict]]:
    """Extract (symbols, imports) from Python source via ``ast``.

    Returns ([], []) on syntax errors — a mid-edit file shouldn't kill
    the pass; its previous rows stay until it parses again.
    """
    import ast
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return [], []

    symbols: list[dict] = []
    imports: list[dict] = []

    def _sig(node) -> str:
        try:
            return "(" + ast.unparse(node.args) + ")"
        except Exception:
            return ""

    def _add(node, kind: str, scope: str = "") -> None:
        symbols.append({
            "name": node.name,
            "kind": kind,
            "scope": scope,
            "line": node.lineno,
            "end_line": getattr(node, "end_lineno", None),
            "signature": _sig(node) if hasattr(node, "args") else "",
        })

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            _add(node, "class")
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    _add(child, "method", scope=node.name)
                elif isinstance(child, ast.ClassDef):
                    _add(child, "class", scope=node.name)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            _add(node, "function")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    symbols.append({
                        "name": target.id, "kind": "const", "scope": "",
                        "line": node.lineno,
                        "end_line": getattr(node, "end_lineno", None),
                        "signature": "",
                    })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"module": alias.name, "name": ""})
        elif isinstance(node, ast.ImportFrom):
            mod = ("." * (node.level or 0)) + (node.module or "")
            for alias in node.names:
                imports.append({"module": mod, "name": alias.name})

    return symbols, imports


# JS/TS line regexes — deliberately regex-lite (no nesting awareness);
# accurate enough for definitions, which is what find_symbol serves.
_JS_FUNC = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s+(\w+)"
)
_JS_CLASS = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)")
_JS_ARROW = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*"
    r"(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>"
)
_JS_CONST = re.compile(r"^\s*export\s+(?:const|let|var)\s+(\w+)\s*[=:]")
_TS_TYPE = re.compile(r"^\s*(?:export\s+)?(?:interface|enum)\s+(\w+)|^\s*(?:export\s+)?type\s+(\w+)\s*=")
_JS_IMPORT = re.compile(r"""(?:from|import|require\()\s*['"]([^'"]+)['"]""")


def extract_js(content: str) -> tuple[list[dict], list[dict]]:
    """Extract (symbols, imports) from JS/TS source via line regexes."""
    symbols: list[dict] = []
    imports: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for lineno, line in enumerate(content.splitlines(), start=1):
        matched: tuple[str, str] | None = None
        m = _JS_FUNC.match(line)
        if m:
            matched = (m.group(1), "function")
        if matched is None:
            m = _JS_CLASS.match(line)
            if m:
                matched = (m.group(1), "class")
        if matched is None:
            m = _TS_TYPE.match(line)
            if m:
                matched = (m.group(1) or m.group(2), "type")
        if matched is None:
            m = _JS_ARROW.match(line)
            if m:
                matched = (m.group(1), "function")
        if matched is None:
            m = _JS_CONST.match(line)
            if m:
                matched = (m.group(1), "const")
        if matched and matched not in seen:
            seen.add(matched)
            symbols.append({
                "name": matched[0], "kind": matched[1], "scope": "",
                "line": lineno, "end_line": None, "signature": "",
            })
        im = _JS_IMPORT.search(line)
        if im and line.lstrip().startswith(("import", "export", "const", "let", "var")):
            imports.append({"module": im.group(1), "name": ""})
    return symbols, imports


def extract_symbols(path: str, content: str) -> tuple[list[dict], list[dict]]:
    """Language-dispatching extraction. Returns (symbols, imports)."""
    lang = _lang_for(path)
    if lang == "python":
        return extract_python(content)
    if lang in ("javascript", "typescript"):
        return extract_js(content)
    return [], []


# ---------------------------------------------------------------------------
# Build / incremental update
# ---------------------------------------------------------------------------

def _rel(path: str) -> str:
    return path.replace("/workspace/", "", 1) if path.startswith("/workspace/") else path


def _persist_file(
    conn: sqlite3.Connection,
    rel_path: str,
    *,
    content: str | None,
    mtime: str | None,
    size: int | None,
) -> None:
    """Replace one file's rows in a single transaction.

    ``content=None`` means metadata-only (non-symbol language) — the
    file still shows in the repo map, with no symbol rows.
    """
    file_hash = hashlib.md5(content.encode()).hexdigest() if content is not None else ""
    symbols, imports = extract_symbols(rel_path, content) if content is not None else ([], [])
    try:
        conn.execute("DELETE FROM ci_symbols WHERE file_path = ?", (rel_path,))
        conn.execute("DELETE FROM ci_imports WHERE file_path = ?", (rel_path,))
        for s in symbols:
            conn.execute(
                "INSERT INTO ci_symbols (file_path, name, kind, scope, line, end_line, signature) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rel_path, s["name"], s["kind"], s["scope"], s["line"],
                 s["end_line"], s["signature"]),
            )
        for i in imports:
            conn.execute(
                "INSERT INTO ci_imports (file_path, module, name) VALUES (?, ?, ?)",
                (rel_path, i["module"], i["name"]),
            )
        conn.execute(
            "INSERT OR REPLACE INTO ci_files (path, hash, mtime, size, lang, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (rel_path, file_hash, mtime, size, _lang_for(rel_path), time.time()),
        )
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        log.warning("code_intel_persist_failed", file=rel_path, exc_info=True)


async def build_code_intel(
    container_manager,
    workspace_id: str,
    force: bool = False,
) -> dict:
    """Build or incrementally update the code-intel index for a workspace.

    Coalesced per workspace: a second concurrent call returns a stub
    instead of stacking a duplicate pass. Cheap by design — unchanged
    files (mtime+size) never get read; only changed symbol-bearing
    files cost a batched container read + parse.
    """
    from augmentum.coder.indexer import (
        _INDEXABLE_EXTS,
        _SKIP_DIRS,
        _batch_read_files,
    )

    lock = _build_locks.setdefault(workspace_id, asyncio.Lock())
    if lock.locked():
        return {"status": "already_running", "indexed": 0, "unchanged": 0, "removed": 0}
    async with lock:
        t0 = time.monotonic()
        conn = await asyncio.to_thread(_open_db, workspace_id)
        try:
            def _existing() -> dict:
                return {
                    row[0]: (row[1], row[2], row[3])
                    for row in conn.execute(
                        "SELECT path, hash, mtime, size FROM ci_files"
                    ).fetchall()
                }
            existing = {} if force else await asyncio.to_thread(_existing)

            try:
                listing = await container_manager._run_command(
                    workspace_id,
                    ["bash", "-c",
                     "find /workspace -type f "
                     + " ".join(f"-not -path '*/{d}/*'" for d in _SKIP_DIRS)
                     + rf" -printf '%T@\t%s\t%p\n' 2>/dev/null | head -{_LIST_CAP}"],
                    timeout=15.0,
                )
            except Exception:
                log.warning("code_intel_list_failed", workspace=workspace_id)
                return {"status": "list_failed", "indexed": 0, "unchanged": 0, "removed": 0}

            files_meta: list[tuple[str, str, str, int]] = []
            for line in listing.splitlines():
                parts = line.split("\t", 2)
                if len(parts) != 3:
                    continue
                mtime_str, size_str, abs_path = parts
                if not any(abs_path.endswith(ext) for ext in _INDEXABLE_EXTS):
                    continue
                try:
                    size_int = int(size_str)
                except ValueError:
                    size_int = -1
                files_meta.append((_rel(abs_path), abs_path, mtime_str, size_int))
            if len(files_meta) >= _LIST_CAP:
                log.warning(
                    "code_intel_listing_cap_hit",
                    workspace=workspace_id, cap=_LIST_CAP,
                )

            indexed = 0
            unchanged = 0
            to_read: list[tuple[str, str, str, int]] = []
            meta_only: list[tuple[str, str, int]] = []
            for rel_path, abs_path, mtime_str, size_int in files_meta:
                prev = existing.get(rel_path)
                if prev is not None and prev[1] == mtime_str and prev[2] == size_int:
                    unchanged += 1
                    continue
                if _lang_for(rel_path) and size_int <= _MAX_FILE_SIZE:
                    to_read.append((rel_path, abs_path, mtime_str, size_int))
                else:
                    meta_only.append((rel_path, mtime_str, size_int))

            def _persist_meta_only() -> None:
                for rel_path, mtime_str, size_int in meta_only:
                    _persist_file(
                        conn, rel_path, content=None, mtime=mtime_str, size=size_int,
                    )
            if meta_only:
                await asyncio.to_thread(_persist_meta_only)
                indexed += len(meta_only)

            from augmentum.coder.indexer import _READ_BATCH_FILES
            for bstart in range(0, len(to_read), _READ_BATCH_FILES):
                batch = to_read[bstart:bstart + _READ_BATCH_FILES]
                contents = await _batch_read_files(
                    container_manager, workspace_id, [b[1] for b in batch],
                )
                def _persist_batch(batch=batch, contents=contents) -> int:
                    done = 0
                    for rel_path, abs_path, mtime_str, size_int in batch:
                        content = contents.get(abs_path)
                        if content is None:
                            continue
                        prev = existing.get(rel_path)
                        file_hash = hashlib.md5(content.encode()).hexdigest()
                        if prev is not None and prev[0] == file_hash:
                            # Touched but identical — refresh meta only.
                            conn.execute(
                                "UPDATE ci_files SET mtime=?, size=?, indexed_at=? WHERE path=?",
                                (mtime_str, size_int, time.time(), rel_path),
                            )
                            conn.commit()
                            continue
                        _persist_file(
                            conn, rel_path,
                            content=content, mtime=mtime_str, size=size_int,
                        )
                        done += 1
                    return done
                indexed += await asyncio.to_thread(_persist_batch)

            # Reconcile deletions — files in the DB that the listing no
            # longer contains lose all their rows together.
            def _reconcile() -> int:
                current = {m[0] for m in files_meta}
                gone = [p for p in existing if p not in current]
                for p in gone:
                    conn.execute("DELETE FROM ci_symbols WHERE file_path = ?", (p,))
                    conn.execute("DELETE FROM ci_imports WHERE file_path = ?", (p,))
                    conn.execute("DELETE FROM ci_files WHERE path = ?", (p,))
                conn.commit()
                return len(gone)
            removed = await asyncio.to_thread(_reconcile)

            duration_ms = int((time.monotonic() - t0) * 1000)
            log.info(
                "code_intel_built",
                workspace=workspace_id, indexed=indexed, unchanged=unchanged,
                removed=removed, files=len(files_meta), duration_ms=duration_ms,
            )
            return {
                "status": "ok", "indexed": indexed, "unchanged": unchanged,
                "removed": removed, "files": len(files_meta),
                "duration_ms": duration_ms,
            }
        finally:
            await asyncio.to_thread(conn.close)


async def reindex_paths(
    container_manager,
    workspace_id: str,
    paths: list[str],
) -> int:
    """Targeted re-extraction of specific files after an agent mutation.

    Called from the coder mutation hook so find_symbol stays fresh
    MID-turn (the next full build catches everything else). Missing
    files (deleted) lose their rows. Returns files updated.
    """
    from augmentum.coder.indexer import _batch_read_files

    abs_paths = []
    for p in paths:
        p = p.strip()
        if not p:
            continue
        abs_paths.append(p if p.startswith("/") else f"/workspace/{p}")
    abs_paths = [p for p in abs_paths if _lang_for(p)]
    if not abs_paths:
        return 0

    contents = await _batch_read_files(container_manager, workspace_id, abs_paths)
    conn = await asyncio.to_thread(_open_db, workspace_id)
    try:
        def _apply() -> int:
            done = 0
            for abs_path in abs_paths:
                rel_path = _rel(abs_path)
                content = contents.get(abs_path)
                if content is None:
                    conn.execute("DELETE FROM ci_symbols WHERE file_path = ?", (rel_path,))
                    conn.execute("DELETE FROM ci_imports WHERE file_path = ?", (rel_path,))
                    conn.execute("DELETE FROM ci_files WHERE path = ?", (rel_path,))
                    conn.commit()
                    continue
                if len(content) > _MAX_FILE_SIZE:
                    continue
                _persist_file(conn, rel_path, content=content, mtime=None, size=len(content))
                done += 1
            return done
        return await asyncio.to_thread(_apply)
    finally:
        await asyncio.to_thread(conn.close)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def _has_index(workspace_id: str) -> bool:
    """True if the sidecar DB exists and has at least one ci_files row."""
    path = _db_path(workspace_id)
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(str(path))
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ci_files'"
            ).fetchone()
            if not row:
                return False
            return conn.execute("SELECT 1 FROM ci_files LIMIT 1").fetchone() is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


async def has_index(workspace_id: str) -> bool:
    return await asyncio.to_thread(_has_index, workspace_id)


async def find_symbol(
    workspace_id: str,
    name: str,
    *,
    kind: str | None = None,
    limit: int = 25,
) -> list[dict]:
    """Look up symbol definitions by name.

    Accepts bare names (``search_index``), qualified names
    (``PackManager.search`` → scope=PackManager, name=search), and falls
    back to a case-insensitive substring match when nothing matches
    exactly — so a close-but-wrong guess still lands near the target.
    """
    def _query() -> list[dict]:
        if not _has_index(workspace_id):
            return []
        conn = sqlite3.connect(str(_db_path(workspace_id)))
        try:
            scope = None
            sym = name.strip()
            if "." in sym:
                scope, sym = sym.rsplit(".", 1)
            where = ["name = ?"]
            args: list = [sym]
            if scope:
                where.append("scope = ?")
                args.append(scope)
            if kind:
                where.append("kind = ?")
                args.append(kind)
            sql = (
                "SELECT file_path, name, kind, scope, line, end_line, signature "
                f"FROM ci_symbols WHERE {' AND '.join(where)} "
                "ORDER BY file_path, line LIMIT ?"
            )
            rows = conn.execute(sql, [*args, limit]).fetchall()
            exact = True
            if not rows:
                exact = False
                fuzzy_where = ["name LIKE ? COLLATE NOCASE"]
                fuzzy_args: list = [f"%{sym}%"]
                if kind:
                    fuzzy_where.append("kind = ?")
                    fuzzy_args.append(kind)
                rows = conn.execute(
                    "SELECT file_path, name, kind, scope, line, end_line, signature "
                    f"FROM ci_symbols WHERE {' AND '.join(fuzzy_where)} "
                    "ORDER BY length(name), file_path, line LIMIT ?",
                    [*fuzzy_args, limit],
                ).fetchall()
            return [
                {
                    "path": r[0], "name": r[1], "kind": r[2], "scope": r[3],
                    "line": r[4], "end_line": r[5], "signature": r[6],
                    "exact": exact,
                }
                for r in rows
            ]
        finally:
            conn.close()

    return await asyncio.to_thread(_query)


async def file_outline(workspace_id: str, path: str) -> dict | None:
    """Structured outline of one file: symbols in line order + imports.

    Returns None when the file isn't in the index (caller may trigger a
    targeted reindex and retry).
    """
    def _query() -> dict | None:
        if not _has_index(workspace_id):
            return None
        rel_path = _rel(path.strip())
        conn = sqlite3.connect(str(_db_path(workspace_id)))
        try:
            frow = conn.execute(
                "SELECT path, lang, size FROM ci_files WHERE path = ?", (rel_path,)
            ).fetchone()
            if frow is None:
                return None
            symbols = [
                {
                    "name": r[0], "kind": r[1], "scope": r[2],
                    "line": r[3], "end_line": r[4], "signature": r[5],
                }
                for r in conn.execute(
                    "SELECT name, kind, scope, line, end_line, signature "
                    "FROM ci_symbols WHERE file_path = ? ORDER BY line",
                    (rel_path,),
                ).fetchall()
            ]
            imports = [
                {"module": r[0], "name": r[1]}
                for r in conn.execute(
                    "SELECT module, name FROM ci_imports WHERE file_path = ? "
                    "ORDER BY module, name",
                    (rel_path,),
                ).fetchall()
            ]
            return {
                "path": frow[0], "lang": frow[1], "size": frow[2],
                "symbols": symbols, "imports": imports,
            }
        finally:
            conn.close()

    return await asyncio.to_thread(_query)


# Repo-map rendering caps. Symbols per file is a display cap, not a data
# cap — find_symbol/file_outline always see everything.
_MAP_SYMBOLS_PER_FILE = 24
_KIND_ORDER = {"class": 0, "type": 1, "function": 2, "const": 3, "method": 4}


def _render_repo_map_sync(workspace_id: str, max_chars: int) -> str:
    if not _has_index(workspace_id):
        return ""
    conn = sqlite3.connect(str(_db_path(workspace_id)))
    try:
        files = conn.execute("SELECT path FROM ci_files ORDER BY path").fetchall()
        if not files:
            return ""
        sym_rows = conn.execute(
            "SELECT file_path, name, kind, scope FROM ci_symbols"
        ).fetchall()
    finally:
        conn.close()

    by_file: dict[str, list[tuple[str, str, str]]] = {}
    methods: dict[tuple[str, str], list[str]] = {}
    for file_path, name, kind, scope in sym_rows:
        if kind == "method":
            methods.setdefault((file_path, scope), []).append(name)
        else:
            by_file.setdefault(file_path, []).append((name, kind, scope))

    header = (
        "<repo_map>\n"
        "Workspace structure — files with their top-level symbols "
        "(classes show {methods}). Generated from the code index; query "
        "precise locations with find_symbol / file_outline instead of "
        "grep-chains.\n"
    )
    footer = "</repo_map>"
    lines: list[str] = []
    used = len(header) + len(footer) + 1
    shown = 0
    for (path,) in files:
        syms = sorted(
            by_file.get(path, ()),
            key=lambda s: (_KIND_ORDER.get(s[1], 9), s[0]),
        )[:_MAP_SYMBOLS_PER_FILE]
        parts: list[str] = []
        for name, kind, _scope in syms:
            if kind == "class":
                ms = sorted(methods.get((path, name), ()))[:8]
                parts.append(f"{name}{{{','.join(ms)}}}" if ms else name)
            else:
                parts.append(name)
        line = f"{path}: {' '.join(parts)}" if parts else path
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
        shown += 1
    if shown < len(files):
        lines.append(
            f"(+{len(files) - shown} more files — use dir_tree / find_symbol)"
        )
    return header + "\n".join(lines) + "\n" + footer


async def render_repo_map(workspace_id: str, max_chars: int = 4000) -> str:
    """Render the byte-stable repo-map block for the stable prompt prefix.

    Deterministic: path-sorted files, kind/name-sorted symbols, no line
    numbers or timestamps — the output only changes when files or symbol
    NAMES change, so ordinary body edits keep the KV prefix cache-hot.
    Returns "" when the workspace has no index yet.
    """
    return await asyncio.to_thread(_render_repo_map_sync, workspace_id, max_chars)
