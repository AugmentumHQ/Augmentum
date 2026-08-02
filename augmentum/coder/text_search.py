"""Structured workspace text search for the Files-panel search UI.

The agent-facing ``CodeGrepTool`` (``augmentum/coder/tools.py``) formats
grep output as TEXT for the model's context window; this module returns
STRUCTURED matches (path / line / highlight spans) so the frontend can
render clickable, highlighted results. Both run inside the workspace
container through the same ``ContainerManager.run_command`` spine — no
host-side filesystem access, same isolation guarantees.

Engine: ripgrep (``rg --json``), installed in the workspace image (see
the apt package list in ``containers.py``). Workspaces built from
images that predate ripgrep fall back to ``grep -rn`` with best-effort
spans computed here; the response's ``engine`` field says which one
served the query so the UI can adjust expectations.
"""
from __future__ import annotations

import json
import re
import shlex
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Directories that are never useful in user-facing search results and
# routinely dwarf the actual project (one `npm install` makes
# node_modules 100x the code the agent wrote). rg additionally honors
# .gitignore; the grep fallback relies on this list alone.
_EXCLUDED_DIRS = (
    "node_modules", ".git", ".venv", "venv", "__pycache__",
    "dist", "build", "target", ".next", ".cache",
)
# Minified bundles / lockfiles match everything and mean nothing.
_EXCLUDED_FILE_GLOBS = ("*.min.js", "*.min.css", "*.lock", "package-lock.json")

# Transport cap piped through `head -c` so a pathological query against
# a huge tree can't stream unbounded JSON back through the exec socket.
# This caps the RAW stream; match-count capping happens at parse time.
_MAX_OUTPUT_BYTES = 3_000_000

# Matched lines longer than this are windowed around the first hit —
# a match inside a single-line JSON dump would otherwise ship the whole
# file as one "line". Display concern only; the file is untouched.
_LINE_WINDOW = 400

_NO_RG_SENTINEL = "__AUG_NO_RG__"


def _clamp_limit(limit: Any, default: int = 500) -> int:
    try:
        return max(1, min(2000, int(limit)))
    except (TypeError, ValueError):
        return default


def _byte_to_char(text: str, byte_off: int) -> int:
    """Convert a byte offset (rg submatch space) to a char offset."""
    raw = text.encode("utf-8", "surrogatepass")
    return len(raw[:byte_off].decode("utf-8", "ignore"))


def _window_line(text: str, spans: list[list[int]]) -> tuple[str, list[list[int]], bool]:
    """Trim a long matched line to a window around the first hit.

    Returns ``(text, adjusted_spans, clipped)``. Spans falling outside
    the window are dropped; the UI shows an ellipsis when clipped.
    """
    text = text.rstrip("\r\n")
    if len(text) <= _LINE_WINDOW:
        return text, spans, False
    anchor = spans[0][0] if spans else 0
    start = max(0, anchor - 80)
    end = start + _LINE_WINDOW
    adjusted = [
        [max(s - start, 0), min(e - start, _LINE_WINDOW)]
        for s, e in spans
        if s < end and e > start
    ]
    return text[start:end], adjusted, True


def _prepare_display_line(
    text: str, spans: list[list[int]],
) -> tuple[str, list[list[int]], bool]:
    """Make a matched line presentable: drop the trailing newline and the
    LEADING INDENTATION that sits before the first match, shifting spans.

    Without this, an indented code line (``white-space: pre`` in the UI)
    renders as a run of blank space next to the line number with the
    actual match pushed off the visible edge — so a real hit looks like
    an empty row. We strip only up to the earliest match start, never
    INTO a match, so a whitespace query still highlights correctly.
    Long lines are then windowed as before.
    """
    text = text.rstrip("\r\n")
    earliest = min((s[0] for s in spans), default=len(text))
    lead = len(text) - len(text.lstrip())
    strip_n = min(lead, earliest)
    if strip_n:
        text = text[strip_n:]
        spans = [[max(s - strip_n, 0), max(e - strip_n, 0)] for s, e in spans]
    return _window_line(text, spans)


async def search_workspace_text(
    cm,
    workspace_id: str,
    query: str,
    *,
    regex: bool = False,
    case_sensitive: bool = False,
    glob: str = "",
    max_results: int = 500,
) -> dict:
    """Run a text search across ``/workspace`` and return structured hits.

    ``regex=False`` searches the query as a literal string (``-F``) —
    the right default for a UI where users paste code fragments full
    of regex metacharacters. ``glob`` optionally narrows the file set
    (e.g. ``*.py``). Results are capped at ``max_results`` with a
    ``truncated`` flag; nothing is silently dropped without saying so.
    """
    if not query:
        return {"error": "query required", "matches": []}
    max_results = _clamp_limit(max_results)

    rg_flags: list[str] = [
        "--json", "--no-config", "--no-ignore-messages",
        "--max-count", "50", "--max-filesize", "2M",
    ]
    if not case_sensitive:
        rg_flags.append("-i")
    if not regex:
        rg_flags.append("-F")
    for d in _EXCLUDED_DIRS:
        rg_flags += ["-g", f"!{d}/**"]
    for g in _EXCLUDED_FILE_GLOBS:
        rg_flags += ["-g", f"!{g}"]
    if glob:
        rg_flags += ["-g", glob]

    rg_cmd = "rg " + " ".join(shlex.quote(f) for f in rg_flags) + \
        f" -e {shlex.quote(query)} . 2>&1"
    # stderr is kept in-stream on purpose: rg reports regex parse
    # errors there, and surfacing "regex parse error: ..." beats a
    # silent empty result. Non-JSON lines are collected below.
    script = (
        "cd /workspace && "
        f"if command -v rg >/dev/null 2>&1; then {rg_cmd}; "
        f"else echo {_NO_RG_SENTINEL}; fi"
    )
    output = await cm.run_command(
        workspace_id,
        ["bash", "-c", f"{script} | head -c {_MAX_OUTPUT_BYTES}"],
        timeout=25.0,
    )

    if _NO_RG_SENTINEL in (output or "")[:200]:
        return await _grep_fallback(
            cm, workspace_id, query,
            regex=regex, case_sensitive=case_sensitive,
            glob=glob, max_results=max_results,
        )

    matches: list[dict] = []
    files: set[str] = set()
    error_lines: list[str] = []
    truncated = len(output or "") >= _MAX_OUTPUT_BYTES - 1
    for line in (output or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            # rg's stderr (regex errors, permission warnings) or the
            # final line the `head -c` cap cut mid-object.
            error_lines.append(line)
            continue
        if not isinstance(ev, dict) or ev.get("type") != "match":
            continue
        d = ev.get("data") or {}
        text = (d.get("lines") or {}).get("text")
        if text is None:
            continue  # non-UTF8 line (rg emits "bytes") — skip
        path_text = (d.get("path") or {}).get("text") or ""
        if not path_text:
            continue
        rel = path_text[2:] if path_text.startswith("./") else path_text
        spans = [
            [_byte_to_char(text, int(sm.get("start", 0))),
             _byte_to_char(text, int(sm.get("end", 0)))]
            for sm in (d.get("submatches") or [])
        ]
        wtext, wspans, clipped = _prepare_display_line(text, spans)
        entry = {
            "path": f"/workspace/{rel}",
            "line": int(d.get("line_number") or 0),
            "text": wtext,
            "spans": wspans,
        }
        if clipped:
            entry["clipped"] = True
        matches.append(entry)
        files.add(rel)
        if len(matches) >= max_results:
            truncated = True
            break

    result: dict = {
        "engine": "rg",
        "query": query,
        "matches": matches,
        "files_with_matches": len(files),
        "total_returned": len(matches),
        "truncated": truncated,
    }
    if not matches and error_lines:
        # Distinguish "no hits" from "rg rejected the query" — the
        # first stderr line is the useful one ("regex parse error: …").
        first = error_lines[0]
        if "error" in first.lower():
            result["error"] = first[:300]
    return result


async def _grep_fallback(
    cm,
    workspace_id: str,
    query: str,
    *,
    regex: bool,
    case_sensitive: bool,
    glob: str,
    max_results: int,
) -> dict:
    """grep -rn fallback for workspace images without ripgrep."""
    flags = ["-rnI", "-m", "50"]
    flags.append("-E" if regex else "-F")
    if not case_sensitive:
        flags.append("-i")
    for d in _EXCLUDED_DIRS:
        flags.append(f"--exclude-dir={d}")
    for g in _EXCLUDED_FILE_GLOBS:
        flags.append(f"--exclude={g}")
    if glob:
        flags.append(f"--include={glob}")
    cmd = "grep " + " ".join(shlex.quote(f) for f in flags) + \
        f" -e {shlex.quote(query)} . 2>/dev/null"
    output = await cm.run_command(
        workspace_id,
        ["bash", "-c", f"cd /workspace && {cmd} | head -c {_MAX_OUTPUT_BYTES}"],
        timeout=25.0,
    )

    # Best-effort span computation — grep gives no offsets. Literal
    # queries use a plain find; regex queries re-run the pattern here.
    finder = None
    if regex:
        try:
            finder = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
        except re.error:
            finder = None

    matches: list[dict] = []
    files: set[str] = set()
    truncated = len(output or "") >= _MAX_OUTPUT_BYTES - 1
    line_re = re.compile(r"^\./(.+?):(\d+):(.*)$")
    for raw in (output or "").splitlines():
        m = line_re.match(raw)
        if not m:
            continue
        rel, lineno, text = m.group(1), int(m.group(2)), m.group(3)
        spans: list[list[int]] = []
        if finder is not None:
            hit = finder.search(text)
            if hit:
                spans = [[hit.start(), hit.end()]]
        elif not regex:
            hay = text if case_sensitive else text.lower()
            needle = query if case_sensitive else query.lower()
            idx = hay.find(needle)
            if idx >= 0:
                spans = [[idx, idx + len(needle)]]
        wtext, wspans, clipped = _prepare_display_line(text, spans)
        entry = {
            "path": f"/workspace/{rel}",
            "line": lineno,
            "text": wtext,
            "spans": wspans,
        }
        if clipped:
            entry["clipped"] = True
        matches.append(entry)
        files.add(rel)
        if len(matches) >= max_results:
            truncated = True
            break

    return {
        "engine": "grep",
        "query": query,
        "matches": matches,
        "files_with_matches": len(files),
        "total_returned": len(matches),
        "truncated": truncated,
    }
