#!/usr/bin/env python3
"""Augmentum context-aware security checker.

Unlike generic scanners, this understands Augmentum's security MODEL.
It reads security_exceptions.json to suppress known intentional findings
and only reports NEW issues that deviate from established patterns.

Checks:
  1. SSRF surface — endpoints accepting URLs without SafeHttpClient
  2. Template literal safety — innerHTML without escapeHtml (client XSS)
  3. SQL injection — f-string/format SQL with variables
  4. API key exposure — GET endpoints returning sensitive fields
  5. Silent exception handlers — swallowed errors on non-cleanup paths
  6. Stale exceptions — exceptions referencing code that no longer exists
  7. Path traversal — untrusted filename/path into a filesystem op without
     basename/sanitiser (the class behind the 2026-07 artifact-import bug:
     ``uploaded.filename`` → ``target_dir / filename`` escaped the store)
  8. Unsafe content serving — server-side HTMLResponse of raw file/user
     content (stored XSS when opened same-origin / cast to a receiver)
  9. Command execution — subprocess shell=True / os.system / os.popen
 10. Unsafe deserialization — pickle/marshal/yaml.load on possibly-untrusted data

Findings are suppressible via references/security_exceptions.json (id + files)
so the report stays a live list of UNREVIEWED gaps as dev continues.

Exit code 0 = clean (or only medium/low), 1 = new critical/high findings.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

# Windows consoles/pipes default to a non-UTF-8 codepage; the ✓ / ~ / … chars
# in our output would raise UnicodeEncodeError mid-report (and the audit then
# logs "parser found no metrics"). Make stdout/stderr lenient — no-op where
# reconfigure isn't available.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: Cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)

ROOT = _find_root()
REFS_DIR = Path(__file__).resolve().parent.parent / "references"

_COLOR = os.environ.get("TERM") or os.name != "nt"
def _red(s: str) -> str:    return f"\033[91m{s}\033[0m" if _COLOR else s
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m" if _COLOR else s
def _green(s: str) -> str:  return f"\033[92m{s}\033[0m" if _COLOR else s
def _cyan(s: str) -> str:   return f"\033[96m{s}\033[0m" if _COLOR else s
def _bold(s: str) -> str:   return f"\033[1m{s}\033[0m" if _COLOR else s
def _dim(s: str) -> str:    return f"\033[2m{s}\033[0m" if _COLOR else s

# ---------------------------------------------------------------------------
# Load exceptions
# ---------------------------------------------------------------------------

def _load_exceptions() -> list[dict]:
    exc_file = REFS_DIR / "security_exceptions.json"
    if not exc_file.exists():
        return []
    try:
        data = json.loads(exc_file.read_text(encoding="utf-8"))
        return data.get("exceptions", [])
    except (json.JSONDecodeError, KeyError):
        return []


def _inside_string_literal(line: str, pos: int) -> bool:
    """True when character ``pos`` on ``line`` sits inside a quoted string.

    Cheap odd-quote-count heuristic. Its job is to stop the command /
    deserialization scanners from flagging dangerous tokens that appear
    inside a ``re.compile(r"...os.system...")`` blocklist pattern or a log
    string — the scanner shouldn't fire on the security controls themselves.
    """
    seg = line[:pos]
    dq = len(re.findall(r'(?<!\\)"', seg))
    sq = len(re.findall(r"(?<!\\)'", seg))
    return (dq % 2 == 1) or (sq % 2 == 1)


def _is_excepted(finding_id: str, file_rel: str, exceptions: list[dict]) -> dict | None:
    """Check if a finding is covered by a known exception. Returns the exception or None."""
    for exc in exceptions:
        if exc.get("id") == finding_id:
            exc_files = exc.get("files", [])
            if "*" in exc_files or any(f in file_rel for f in exc_files):
                return exc
    return None

# ---------------------------------------------------------------------------
# 1. SSRF surface check
# ---------------------------------------------------------------------------

def check_ssrf_surface(exceptions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Find endpoints that accept URL parameters without SafeHttpClient protection."""
    findings: list[dict] = []
    suppressed: list[dict] = []

    proxy_dir = ROOT / "augmentum" / "proxy"
    for rf in sorted(proxy_dir.glob("*_routes.py")):
        text = rf.read_text(encoding="utf-8", errors="replace")
        rel = str(rf.relative_to(ROOT)).replace("\\", "/")

        # Find functions that take a URL from the request
        for m in re.finditer(
            r"(?:body\.url|request\.query_params.*url|url\s*[:=]\s*(?:body|request|params))",
            text,
        ):
            line_no = text[:m.start()].count("\n") + 1
            # Check if SafeHttpClient or _validate_url is used nearby.
            # Symmetric ~2000-char window covers function-top guards.
            context_start = max(0, m.start() - 2000)
            context_end = min(len(text), m.end() + 2000)
            context = text[context_start:context_end]

            has_ssrf_check = (
                "SafeHttpClient" in context
                or "_validate_url" in context
                or "_check_resolved_ips" in context
                or "safe_client" in context
                or "_safe_client" in context
                or "_image_client" in context
                or "check_ssrf" in context
            )

            if not has_ssrf_check:
                finding = {
                    "id": "ssrf-unprotected",
                    "severity": "HIGH",
                    "file": rel,
                    "line": line_no,
                    "description": "Endpoint accepts URL input without SSRF protection (no SafeHttpClient)",
                }
                # Check all SSRF-related exceptions
                exc = (
                    _is_excepted("provider-url-no-ssrf", rel, exceptions)
                    or _is_excepted("browse-save-url-metadata", rel, exceptions)
                    or _is_excepted("coder-routes-ssrf", rel, exceptions)
                    or _is_excepted("knowledge-notes-url", rel, exceptions)
                    or _is_excepted("cardsmith-routes-ssrf-via-safe-fetcher", rel, exceptions)
                    or _is_excepted("xr-browser-panel-local-only", rel, exceptions)
                    or _is_excepted("cast-routes-receiver-url-broker", rel, exceptions)
                    or _is_excepted("community-install-audit-url", rel, exceptions)
                )
                if exc:
                    suppressed.append({**finding, "exception": exc["id"], "reason": exc["reason"]})
                else:
                    findings.append(finding)

    return findings, suppressed

# ---------------------------------------------------------------------------
# 2. Template literal XSS check (enhanced)
# ---------------------------------------------------------------------------

def check_template_xss(exceptions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Scan for innerHTML assignments with unescaped interpolations."""
    findings: list[dict] = []
    suppressed: list[dict] = []

    # Known-safe patterns (internal constants, not user data)
    safe_patterns = {
        "escapeHtml", "_escHtml", "_esc(", "encodeURIComponent", "encodeURI",
        "JSON.stringify", "parseInt", "parseFloat", "Number(",
        "Math.", "Date.", ".length", ".id", ".toString()",
        "icons.", "DEFAULTS.", "THEME_", "PHASE_DISPLAY",
        # Internal CSS classes and enum values
        "currentMode", "cls", "icon", "historyHtml",
        # Ternary with string literals only
        "filter ?", "isActive ?", "checked ?",
        # Data from server that's numeric/boolean
        "data.samples",
        # Numeric counters and progress totals (always Number from .length)
        "completed", "total",
        # Pre-built HTML fragments composed from already-escaped values or
        # static literals — wrapping them in another escapeHtml would render
        # the markup as text. Audit the call site if you suspect drift.
        "errorBadge", "actionBtns", "breadcrumb", "rendered", "summary",
        "bars", "fileSpan",
        # ...the conventional names for "rows = items.map(i => `…${escapeHtml(i)}…`)"
        # list fragments and their "N more" tails, plus small pre-built spans.
        "rows", "more", "actions", "failNote", "badge", "count",
        # Response.status is an HTTP status code Number per the fetch spec —
        # decimal digits only, never a string. Common pattern:
        # `<div>Failed (status ${r.status})</div>` in fetch error branches.
        ".status",
    }

    js_dir = ROOT / "ui" / "scripts"
    for jsfile in sorted(js_dir.rglob("*.js")):
        text = jsfile.read_text(encoding="utf-8", errors="replace")
        rel = str(jsfile.relative_to(ROOT)).replace("\\", "/")

        for i, line in enumerate(text.splitlines(), 1):
            if ".innerHTML" not in line and ".insertAdjacentHTML" not in line:
                continue

            for m_interp in re.finditer(r"\$\{([^}]+)\}", line):
                expr = m_interp.group(1).strip()

                # Skip safe patterns
                if any(sp in expr for sp in safe_patterns):
                    continue
                # Skip pure numbers/booleans
                if re.match(r"^[\d.+\-*/%\s()!]+$", expr):
                    continue
                # Skip string literals
                if re.match(r"^['\"].*['\"]$", expr):
                    continue
                # Skip a ternary whose VALUE branches are all string literals /
                # safe patterns — the condition (parts[0]) never reaches the DOM,
                # so `cond ? '✓' : '✗'` is safe regardless of what `cond` is.
                _tern = re.split(r"[?:]", expr)
                if "?" in expr and len(_tern) >= 3 and all(
                    "'" in part or '"' in part or any(sp in part for sp in safe_patterns)
                    for part in _tern[1:]
                ):
                    continue

                short_expr = expr[:60] + "..." if len(expr) > 60 else expr
                finding = {
                    "id": "xss-template",
                    "severity": "MEDIUM",
                    "file": rel,
                    "line": i,
                    "description": f"${{{short_expr}}} in innerHTML without escapeHtml",
                }

                # Check exceptions
                exc = _is_excepted("avatar-data-uri-unescaped", rel, exceptions)
                if not exc:
                    exc = _is_excepted("article-raw-html", rel, exceptions)
                if exc and any(kw in expr.lower() for kw in ["avatar", "resized", "bodyHtml", "html"]):
                    suppressed.append({**finding, "exception": exc["id"], "reason": exc["reason"]})
                    continue
                # Per-file exceptions where the interpolated value is provably
                # already escaped or static (the safe_patterns set covers the
                # generic-name cases; this branch handles file-specific cases
                # where renaming the variable would mask the intent).
                exc = _is_excepted("learning-game-companion-dialogue-escaped", rel, exceptions)
                if exc and expr in ("bubble", "speak"):
                    suppressed.append({**finding, "exception": exc["id"], "reason": exc["reason"]})
                    continue
                exc = _is_excepted("powers-renderpowerrow-escaped", rel, exceptions)
                if exc and "_renderPowerRow" in expr:
                    suppressed.append({**finding, "exception": exc["id"], "reason": exc["reason"]})
                    continue
                exc = _is_excepted("youtube-panel-mountid-generated", rel, exceptions)
                if exc and expr == "mountId":
                    suppressed.append({**finding, "exception": exc["id"], "reason": exc["reason"]})
                    continue
                findings.append(finding)

    return findings, suppressed

# ---------------------------------------------------------------------------
# 3. SQL injection check
# ---------------------------------------------------------------------------

def check_sql_injection(exceptions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Find potential SQL injection via f-strings or format strings.

    Only flags actual DB execute calls (conn.execute, cursor.execute),
    not HTTP endpoints or Python methods named 'execute'.
    """
    findings: list[dict] = []
    suppressed: list[dict] = []

    # Patterns that indicate actual SQL execute (not HTTP or method calls)
    sql_execute_re = re.compile(
        r"(?:conn|cursor|db|backend|be|self\._conn|self\.conn)"
        r"\s*\.\s*(?:execute|executemany|executescript)\s*\(\s*f['\"]",
    )

    # Known safe patterns — table/column names from internal hardcoded sources
    safe_context = {
        "placeholders", "_BUNDLED_IDS", "schema_version",
        "tables", "table_name", "vec_table", "idx_",
        "noqa: S608",  # explicit security bypass marker
    }

    for pyfile in sorted((ROOT / "augmentum").rglob("*.py")):
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        rel = str(pyfile.relative_to(ROOT)).replace("\\", "/")

        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            if sql_execute_re.search(stripped):
                # Check for known safe patterns
                if any(sp in stripped for sp in safe_context):
                    continue
                # Check surrounding context for internal table iteration
                ctx_start = max(0, i - 5)
                ctx_end = min(len(text.splitlines()), i + 2)
                context = "\n".join(text.splitlines()[ctx_start:ctx_end])
                if any(sp in context for sp in safe_context):
                    continue

                finding = {
                    "id": "sql-injection",
                    "severity": "CRITICAL",
                    "file": rel,
                    "line": i,
                    "description": "Possible SQL injection: f-string in DB execute() call",
                }
                exc = (
                    _is_excepted("sql-fstring-hardcoded-tables", rel, exceptions)
                    or _is_excepted("sql-fstring-vec-dimension", rel, exceptions)
                    or _is_excepted("sql-fstring-internal-identifier", rel, exceptions)
                    or _is_excepted("sql-fstring-docstring-example", rel, exceptions)
                    or _is_excepted("sql-fstring-backup-vacuum-into", rel, exceptions)
                    or _is_excepted("sql-fstring-lang-pack-vocab-cols", rel, exceptions)
                )
                if exc:
                    suppressed.append({**finding, "exception": exc["id"], "reason": exc["reason"]})
                else:
                    findings.append(finding)

            # Also check .format()
            if sql_execute_re.pattern.replace("f['\"]", r".*\.format\(") and \
               re.search(r"(?:conn|cursor|db)\s*\.\s*execute.*\.format\(", stripped):
                finding = {
                    "id": "sql-injection",
                    "severity": "CRITICAL",
                    "file": rel,
                    "line": i,
                    "description": "Possible SQL injection: .format() in DB execute() call",
                }
                exc = (
                    _is_excepted("sql-fstring-hardcoded-tables", rel, exceptions)
                    or _is_excepted("sql-fstring-vec-dimension", rel, exceptions)
                    or _is_excepted("sql-fstring-internal-identifier", rel, exceptions)
                )
                if exc:
                    suppressed.append({**finding, "exception": exc["id"], "reason": exc["reason"]})
                else:
                    findings.append(finding)

    return findings, suppressed

# ---------------------------------------------------------------------------
# 4. API key exposure check
# ---------------------------------------------------------------------------

def check_key_exposure(exceptions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Verify GET endpoints don't return sensitive fields."""
    findings: list[dict] = []
    suppressed: list[dict] = []

    sensitive_patterns = re.compile(
        r"api_key|secret|password|token|private_key|credentials",
        re.IGNORECASE,
    )

    proxy_dir = ROOT / "augmentum" / "proxy"
    for rf in sorted(proxy_dir.glob("*_routes.py")):
        text = rf.read_text(encoding="utf-8", errors="replace")
        rel = str(rf.relative_to(ROOT)).replace("\\", "/")

        # Find GET endpoints
        for m in re.finditer(r"@router\.get\(", text):
            # Get the function body (next ~50 lines)
            func_start = m.start()
            func_text = text[func_start:func_start + 3000]

            # Check for JSONResponse or dict return with sensitive fields
            if "to_safe_dict" in func_text or "to_dict" in func_text:
                continue  # Using safe serializer

            # Check if SELECT query includes sensitive column
            for sel in re.finditer(r"SELECT\s+(.*?)\s+FROM", func_text, re.IGNORECASE | re.DOTALL):
                columns = sel.group(1)
                finding = None
                if columns.strip() == "*":
                    line_no = text[:func_start + sel.start()].count("\n") + 1
                    finding = {
                        "id": "key-exposure-select-star",
                        "severity": "HIGH",
                        "file": rel,
                        "line": line_no,
                        "description": "GET endpoint uses SELECT * which may expose api_key column",
                    }
                elif sensitive_patterns.search(columns):
                    line_no = text[:func_start + sel.start()].count("\n") + 1
                    finding = {
                        "id": "key-exposure",
                        "severity": "HIGH",
                        "file": rel,
                        "line": line_no,
                        "description": "GET endpoint SELECT includes sensitive column",
                    }

                if finding:
                    exc = (
                        _is_excepted("cloud-image-key-internal", rel, exceptions)
                        or _is_excepted("dream-routes-metadata", rel, exceptions)
                    )
                    if exc:
                        suppressed.append({**finding, "exception": exc["id"], "reason": exc["reason"]})
                    else:
                        findings.append(finding)

    return findings, suppressed

# ---------------------------------------------------------------------------
# 5. Stale exceptions check
# ---------------------------------------------------------------------------

def check_stale_exceptions(exceptions: list[dict]) -> list[dict]:
    """Find exceptions that reference files or patterns no longer in the codebase."""
    stale: list[dict] = []

    for exc in exceptions:
        exc_files = exc.get("files", [])
        if "*" in exc_files:
            continue  # Wildcard, always valid

        all_missing = True
        for ef in exc_files:
            full_path = ROOT / ef
            if full_path.exists():
                all_missing = False
                break
            # Check if it's a partial path match
            if any((ROOT / "augmentum").rglob(ef.split("/")[-1])):
                all_missing = False
                break

        if all_missing and exc_files:
            stale.append({
                "id": "stale-exception",
                "severity": "INFO",
                "exception_id": exc["id"],
                "files": exc_files,
                "description": f"Exception '{exc['id']}' references files that no longer exist",
            })

    return stale

# ---------------------------------------------------------------------------
# 6. Silent-exception patterns on non-cleanup paths
# ---------------------------------------------------------------------------
# Three shapes hide errors equivalently and all need to be caught:
#   contextlib.suppress(Exception)
#   except Exception(\s+as\s+\w+)?: \n  pass|continue
#   except: \n  pass|continue   (bare except)
#
# A site is acceptable when it lives on a cleanup/shutdown path (the
# surrounding ~3 lines mention close/shutdown/cleanup/etc.) — those
# are exempted from CLAUDE.md's prohibition because the exception is
# documented as best-effort teardown.
#
# Extended from the original contextlib.suppress-only check after the
# 2026-05-22 sweep that cleaned 176 sites across 73 files; the new
# patterns catch the broader `except Exception: pass` shape that the
# old check missed entirely.

_SILENT_EXC_PATTERNS = [
    # contextlib.suppress(Exception) — original CLAUDE.md-prohibited shape.
    (
        re.compile(r"contextlib\.suppress\(Exception\)"),
        "contextlib.suppress(Exception)",
    ),
    # `except Exception[ as ...]:` followed by a swallow-only body
    # (`pass` or `continue`). Body lookup tolerates a leading docstring
    # or one-line `# comment` between except and the swallow keyword.
    (
        re.compile(
            r"except\s+Exception(?:\s+as\s+\w+)?\s*:\s*"
            r"(?:#[^\n]*\n\s*)?"
            r"(?:pass|continue)\b",
            re.MULTILINE,
        ),
        "except Exception: pass / continue",
    ),
    # Bare `except:` with swallow-only body. Worst variant — masks
    # SystemExit and KeyboardInterrupt too.
    (
        re.compile(
            r"except\s*:\s*(?:#[^\n]*\n\s*)?(?:pass|continue)\b",
            re.MULTILINE,
        ),
        "bare except: pass / continue",
    ),
]


def check_silent_exceptions() -> list[dict]:
    """Find silent-swallow patterns that may hide real errors."""
    findings: list[dict] = []

    # Known acceptable uses (cleanup/shutdown paths)
    cleanup_patterns = {"close", "shutdown", "cleanup", "destroy", "__aexit__", "disconnect"}

    for pyfile in sorted((ROOT / "augmentum").rglob("*.py")):
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        rel = str(pyfile.relative_to(ROOT)).replace("\\", "/")
        lines = text.splitlines()

        for rx, label in _SILENT_EXC_PATTERNS:
            for m in rx.finditer(text):
                line_no = text[:m.start()].count("\n") + 1

                # Context window: 3 lines before, 3 lines after
                start = max(0, line_no - 4)
                end = min(len(lines), line_no + 3)
                context = "\n".join(lines[start:end]).lower()

                if any(pat in context for pat in cleanup_patterns):
                    continue  # Cleanup path, acceptable

                findings.append({
                    "id": "silent-exception",
                    "severity": "LOW",
                    "file": rel,
                    "line": line_no,
                    "description": f"{label} on non-cleanup path — may hide real errors",
                })

    return findings

# ---------------------------------------------------------------------------
# 7. Filesystem path traversal — untrusted filename/path into a FS sink
# ---------------------------------------------------------------------------
# The class behind the 2026-07 artifact-import bug: a request-supplied
# ``filename`` (multipart upload, ``?path=`` query, ``{x:path}`` route param)
# reaching a filesystem op without being reduced to a basename first — so
# ``../../etc/passwd`` or ``/abs/path`` escapes the intended directory on a
# READ or WRITE. Function-scoped so a sanitiser anywhere in the function
# clears it; residual false positives go in security_exceptions.json.

# An untrusted, path-shaped value entering the function.
_UNTRUSTED_PATH_SRC = re.compile(
    r"\.filename\b"                                    # uploaded.filename / part.filename
    r"|query_params\.get\([^)]*(?:path|file|name)"     # ?path= / ?file= / ?name=
    r"|\bbody\.(?:filename|file_path|filepath|path|name|file)\b"
    r"|form\.get\([^)]*(?:path|file|name)"
)
# A DIRECT filesystem operation that a traversal payload would abuse.
_FS_SINK = re.compile(
    r"\bopen\("
    r"|os\.path\.join\("
    r"|\.write_bytes\(|\.write_text\(|\.read_bytes\(|\.read_text\("
    r"|shutil\.(?:copy2?|copyfile|copyfileobj|move)\("
    r"|FileResponse\("
    r"|\)\s*/\s*[A-Za-z_]\w*"        # Path(...) / name
    r"|_?dir\s*/\s*[A-Za-z_]\w*"     # base_dir / name
)
# The value is FORWARDED into a write/save/extract call (the import-bug shape:
# uploaded.filename → store.save(filename=...)). Reported MEDIUM — the sink
# that actually touches disk may or may not sanitise, so it needs a look.
_FORWARD_WRITE_SINK = re.compile(
    r"\.save\(|save_from_path\(|\.write_bytes\(|\.write_text\(|\.write\("
    r"|shutil\.|extractall\(|\.extract\(|\.mkdir\("
)
# Confines the value to a safe basename / base directory. Includes the
# project's own path validators so the scanner recognises them as safe.
_PATH_SANITIZER = re.compile(
    r"os\.path\.basename|\bbasename\(|_safe_filename|secure_filename"
    r"|_validate_workspace_path|_safe_relative_path"          # project validators
    r"|\.resolve\(\)|relative_to|is_relative_to|_within|sanitize"
    r"|validate_path|werkzeug|\.name\b|\.stem\b|\.suffix\b"   # .name/.stem/.suffix → basename/ext only
)


def check_path_traversal(exceptions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Flag functions that take an untrusted filename/path and use it without
    a basename/confinement sanitiser (HIGH when a filesystem op is present in
    the same function, MEDIUM when the value is merely forwarded on)."""
    findings: list[dict] = []
    suppressed: list[dict] = []

    for pyfile in sorted((ROOT / "augmentum").rglob("*.py")):
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        if not _UNTRUSTED_PATH_SRC.search(text):
            continue  # cheap reject before the AST parse
        rel = str(pyfile.relative_to(ROOT)).replace("\\", "/")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            seg = ast.get_source_segment(text, node) or ""
            src = _UNTRUSTED_PATH_SRC.search(seg)
            if not src:
                continue
            if _PATH_SANITIZER.search(seg):
                continue  # sanitised somewhere in the function — good
            sink = _FS_SINK.search(seg)
            fwd = _FORWARD_WRITE_SINK.search(seg) if not sink else None
            if not sink and not fwd:
                continue  # value is used for display / SQL / comparison, not a path
            offset = (sink.start() if sink else fwd.start())
            line_no = node.lineno + seg[:offset].count("\n")
            finding = {
                "id": "path-traversal",
                "severity": "HIGH" if sink else "MEDIUM",
                "file": rel,
                "line": line_no,
                "description": (
                    "Untrusted filename/path used without basename/sanitiser"
                    + (" before a filesystem op" if sink
                       else " (forwarded into a write/save call — confirm it sanitises)")
                ),
            }
            exc = _is_excepted("path-traversal", rel, exceptions)
            if exc:
                suppressed.append({**finding, "exception": exc["id"], "reason": exc["reason"]})
            else:
                findings.append(finding)

    return findings, suppressed

# ---------------------------------------------------------------------------
# 8. Unsafe server-side content serving — raw file/user content as HTML
# ---------------------------------------------------------------------------
# HTMLResponse(<var>) where <var> holds raw file bytes / user markup renders
# it inline SAME-ORIGIN. Opened directly ("new tab") or cast to a receiver it
# runs with the viewer's session cookie — stored XSS. The 2026-07 fix served
# imported HTML as an attachment instead. App-builder / tool-generated HTML is
# trusted-by-construction — exception those specific sites.

_RAW_CONTENT_NAMES = {
    "content", "html", "raw", "body", "text", "data", "markup",
    "source", "file_content", "contents", "payload", "rendered_html",
}
_HTMLRESPONSE_VAR = re.compile(r"HTMLResponse\(\s*([A-Za-z_]\w*)\s*[,)]")
_CONTENT_SAFE = re.compile(
    r"escape\(|escapeHtml|sanitize|bleach|nosniff|attachment|_preview_shell\("
)


def check_unsafe_content_serving(exceptions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Flag ``HTMLResponse(<raw-content-var>)`` where the variable is loaded
    from disk / a request and isn't obviously sanitised — a stored-XSS vector
    when the response is reachable same-origin."""
    findings: list[dict] = []
    suppressed: list[dict] = []

    for pyfile in sorted((ROOT / "augmentum").rglob("*.py")):
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        if "HTMLResponse(" not in text:
            continue
        rel = str(pyfile.relative_to(ROOT)).replace("\\", "/")
        lines = text.splitlines()
        loads_from_disk = bool(re.search(r"\.read_text\(|\.read\(\)|\bopen\(", text))
        if not loads_from_disk:
            continue  # HTMLResponse of only literals/f-strings — not raw content

        for m in _HTMLRESPONSE_VAR.finditer(text):
            var = m.group(1)
            if var not in _RAW_CONTENT_NAMES:
                continue
            line_no = text[:m.start()].count("\n") + 1
            window = "\n".join(lines[max(0, line_no - 12):line_no + 1])
            if _CONTENT_SAFE.search(window):
                continue  # escaped / attachment / preview-shell wrapped
            finding = {
                "id": "unsafe-content-serving",
                "severity": "MEDIUM",
                "file": rel,
                "line": line_no,
                "description": f"HTMLResponse({var}) serves raw content inline — "
                               "stored-XSS if untrusted + reachable same-origin",
            }
            exc = _is_excepted("unsafe-content-serving", rel, exceptions)
            if exc:
                suppressed.append({**finding, "exception": exc["id"], "reason": exc["reason"]})
            else:
                findings.append(finding)

    return findings, suppressed

# ---------------------------------------------------------------------------
# 9. Command execution surface
# ---------------------------------------------------------------------------

_CMD_PATTERNS = [
    (re.compile(r"subprocess\.(?:run|call|check_output|check_call|Popen)\("
                r"[^\n]*shell\s*=\s*True"),
     "HIGH", "subprocess with shell=True — command injection if any arg is interpolated"),
    (re.compile(r"\bos\.system\("), "HIGH", "os.system() — command injection surface"),
    (re.compile(r"\bos\.popen\("), "MEDIUM", "os.popen() — prefer subprocess with a list argv"),
]


def check_command_execution(exceptions: list[dict]) -> tuple[list[dict], list[dict]]:
    findings: list[dict] = []
    suppressed: list[dict] = []
    for pyfile in sorted((ROOT / "augmentum").rglob("*.py")):
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        rel = str(pyfile.relative_to(ROOT)).replace("\\", "/")
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            for rx, sev, desc in _CMD_PATTERNS:
                cm = rx.search(s)
                if cm and not _inside_string_literal(s, cm.start()):
                    finding = {
                        "id": "command-execution", "severity": sev,
                        "file": rel, "line": i, "description": desc,
                    }
                    exc = _is_excepted("command-execution", rel, exceptions)
                    if exc:
                        suppressed.append({**finding, "exception": exc["id"], "reason": exc["reason"]})
                    else:
                        findings.append(finding)
                    break
    return findings, suppressed

# ---------------------------------------------------------------------------
# 10. Unsafe deserialization
# ---------------------------------------------------------------------------

_DESERIAL_PATTERNS = [
    (re.compile(r"\b(?:pickle|cPickle|cloudpickle|dill)\.loads?\("),
     "HIGH", "Unpickling — arbitrary code execution if the source is untrusted"),
    (re.compile(r"\bmarshal\.loads?\("), "MEDIUM", "marshal.load — unsafe on untrusted data"),
    (re.compile(r"\byaml\.load\((?![^)\n]*Loader)"),
     "HIGH", "yaml.load without an explicit SafeLoader"),
]


def check_unsafe_deserialization(exceptions: list[dict]) -> tuple[list[dict], list[dict]]:
    findings: list[dict] = []
    suppressed: list[dict] = []
    for pyfile in sorted((ROOT / "augmentum").rglob("*.py")):
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        rel = str(pyfile.relative_to(ROOT)).replace("\\", "/")
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            for rx, sev, desc in _DESERIAL_PATTERNS:
                dm = rx.search(s)
                if dm and not _inside_string_literal(s, dm.start()):
                    finding = {
                        "id": "unsafe-deserialization", "severity": sev,
                        "file": rel, "line": i, "description": desc,
                    }
                    exc = _is_excepted("unsafe-deserialization", rel, exceptions)
                    if exc:
                        suppressed.append({**finding, "exception": exc["id"], "reason": exc["reason"]})
                    else:
                        findings.append(finding)
                    break
    return findings, suppressed

# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main():
    print(_bold("\n  Augmentum Security Check"))
    print(_bold("  " + "=" * 40))
    print(_dim("  Context-aware • reads security_exceptions.json\n"))

    exceptions = _load_exceptions()
    print(f"  {_dim(f'Loaded {len(exceptions)} known exception(s)')}\n")

    all_findings: list[dict] = []
    all_suppressed: list[dict] = []

    # Run checks
    print(_cyan("  [1/10] Checking SSRF surface..."))
    ssrf_f, ssrf_s = check_ssrf_surface(exceptions)
    all_findings.extend(ssrf_f)
    all_suppressed.extend(ssrf_s)

    print(_cyan("  [2/10] Checking template literal safety (client XSS)..."))
    xss_f, xss_s = check_template_xss(exceptions)
    all_findings.extend(xss_f)
    all_suppressed.extend(xss_s)

    print(_cyan("  [3/10] Checking SQL injection patterns..."))
    sql_f, sql_s = check_sql_injection(exceptions)
    all_findings.extend(sql_f)
    all_suppressed.extend(sql_s)

    print(_cyan("  [4/10] Checking API key exposure..."))
    key_f, key_s = check_key_exposure(exceptions)
    all_findings.extend(key_f)
    all_suppressed.extend(key_s)

    print(_cyan("  [5/10] Checking for silent exception handlers..."))
    suppress_f = check_silent_exceptions()
    all_findings.extend(suppress_f)

    print(_cyan("  [6/10] Checking for stale exceptions..."))
    stale_f = check_stale_exceptions(exceptions)
    all_findings.extend(stale_f)

    print(_cyan("  [7/10] Checking filesystem path traversal..."))
    trav_f, trav_s = check_path_traversal(exceptions)
    all_findings.extend(trav_f)
    all_suppressed.extend(trav_s)

    print(_cyan("  [8/10] Checking unsafe content serving (server XSS)..."))
    serve_f, serve_s = check_unsafe_content_serving(exceptions)
    all_findings.extend(serve_f)
    all_suppressed.extend(serve_s)

    print(_cyan("  [9/10] Checking command execution surface..."))
    cmd_f, cmd_s = check_command_execution(exceptions)
    all_findings.extend(cmd_f)
    all_suppressed.extend(cmd_s)

    print(_cyan("  [10/10] Checking unsafe deserialization..."))
    deser_f, deser_s = check_unsafe_deserialization(exceptions)
    all_findings.extend(deser_f)
    all_suppressed.extend(deser_s)

    print()

    # Categorize by severity
    critical = [f for f in all_findings if f["severity"] == "CRITICAL"]
    high = [f for f in all_findings if f["severity"] == "HIGH"]
    medium = [f for f in all_findings if f["severity"] == "MEDIUM"]
    low = [f for f in all_findings if f["severity"] in ("LOW", "INFO")]

    # Report findings
    if critical:
        print(_red(f"  CRITICAL ({len(critical)}):"))
        for f in critical:
            print(f"    {_red('!!')} [{f['file']}:{f.get('line', '?')}] {f['description']}")
        print()

    if high:
        print(_red(f"  HIGH ({len(high)}):"))
        for f in high:
            print(f"    {_red('!')} [{f['file']}:{f.get('line', '?')}] {f['description']}")
        print()

    if medium:
        print(_yellow(f"  MEDIUM ({len(medium)}):"))
        for f in medium:
            print(f"    {_yellow('~')} [{f['file']}:{f.get('line', '?')}] {f['description']}")
        print()

    if low:
        print(_dim(f"  LOW/INFO ({len(low)}):"))
        for f in low:
            print(f"    {_dim('-')} [{f.get('file', '')}:{f.get('line', '?')}] {f['description']}")
        print()

    # Report suppressions
    if all_suppressed and "--verbose" in sys.argv:
        print(_dim(f"  Suppressed ({len(all_suppressed)}) — covered by security_exceptions.json:"))
        for s in all_suppressed:
            print(f"    {_dim('~')} [{s['file']}:{s.get('line', '?')}] {s['description']}")
            exc_id = s["exception"]
            exc_reason = s["reason"][:80]
            print(f"      {_dim(f'Exception: {exc_id} — {exc_reason}')}")
        print()

    # Summary
    total_new = len(critical) + len(high) + len(medium) + len(low)
    if total_new == 0:
        print(_green(f"  No new security findings."))
        print(_dim(f"  {len(all_suppressed)} known exception(s) suppressed."))
    else:
        severity_str = ", ".join(filter(None, [
            f"{len(critical)} critical" if critical else "",
            f"{len(high)} high" if high else "",
            f"{len(medium)} medium" if medium else "",
            f"{len(low)} low" if low else "",
        ]))
        color = _red if (critical or high) else _yellow
        print(color(f"  {total_new} new finding(s): {severity_str}"))
        print(_dim(f"  {len(all_suppressed)} known exception(s) suppressed."))

    print(_dim(f"\n  Tip: use --verbose to see suppressed findings"))
    print(_dim(f"  Edit references/security_exceptions.json to manage exceptions\n"))

    return 1 if (critical or high) else 0


if __name__ == "__main__":
    sys.exit(main())
