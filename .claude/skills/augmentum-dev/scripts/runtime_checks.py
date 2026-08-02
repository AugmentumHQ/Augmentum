#!/usr/bin/env python3
"""Augmentum runtime bug-pattern scanner — catches issues validate_wiring.py misses.

Detects:
  1. Empty model strings in InternalChatRequest() without subsequent override
  2. Silent exception swallowing (bare pass / log.debug on user-visible paths)
  3. Unhandled fetch() failures in frontend JS
  4. Unguarded app.state attribute access in proxy routes

Uses a companion suppressions file (runtime_suppressions.json) to skip
known-acceptable findings.  Run with --verbose to see suppressed entries.

Exit code 0 = clean, 1 = issues found.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import _common  # noqa: F401 — import side-effect: UTF-8-safe stdout/stderr

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------

def _find_root() -> Path:
    """Walk up from this script to find the project root (has augmentum/ and ui/)."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: Cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)

ROOT = _find_root()

# ---------------------------------------------------------------------------
# Terminal colors (skip on Windows without ANSI support)
# ---------------------------------------------------------------------------

_COLOR = os.environ.get("TERM") or os.name != "nt"

def _red(s: str) -> str:    return f"\033[91m{s}\033[0m" if _COLOR else s
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m" if _COLOR else s
def _green(s: str) -> str:  return f"\033[92m{s}\033[0m" if _COLOR else s
def _cyan(s: str) -> str:   return f"\033[96m{s}\033[0m" if _COLOR else s
def _bold(s: str) -> str:   return f"\033[1m{s}\033[0m" if _COLOR else s

# ---------------------------------------------------------------------------
# Suppressions
# ---------------------------------------------------------------------------

_SUPPRESSIONS_PATH = Path(__file__).resolve().parent / "runtime_suppressions.json"

def _load_suppressions() -> dict[str, list[str]]:
    """Load the suppressions file, creating an empty one if missing."""
    if _SUPPRESSIONS_PATH.is_file():
        try:
            data = json.loads(_SUPPRESSIONS_PATH.read_text(encoding="utf-8"))
            return {
                "empty_model": data.get("empty_model", []),
                "silent_exception": data.get("silent_exception", []),
                "fetch_resilience": data.get("fetch_resilience", []),
                "app_state_access": data.get("app_state_access", []),
            }
        except (json.JSONDecodeError, KeyError):
            pass
    # Create default
    default: dict[str, list[str]] = {
        "empty_model": [],
        "silent_exception": [],
        "fetch_resilience": [],
        "app_state_access": [],
    }
    _SUPPRESSIONS_PATH.write_text(
        json.dumps(default, indent=2) + "\n", encoding="utf-8",
    )
    return default

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel(path: Path) -> str:
    """Return a forward-slash relative path from ROOT."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _is_suppressed(suppressions: list[str], rel_path: str, line_no: int, func_name: str | None = None) -> bool:
    """Check if a finding matches any suppression entry."""
    key_line = f"{rel_path}:{line_no}"
    key_func = f"{rel_path}:{func_name}" if func_name else None
    for s in suppressions:
        if s == key_line or s == rel_path or (key_func and s == key_func):
            return True
        # Allow wildcard line matching (file only)
        if ":" not in s and rel_path.startswith(s):
            return True
    return False


def _python_files(subdir: str = "augmentum") -> list[Path]:
    """Collect all .py files under a subdirectory, excluding docs/."""
    base = ROOT / subdir
    return sorted(
        p for p in base.rglob("*.py")
        if "/docs/" not in _rel(p) and "\\docs\\" not in str(p)
    )


def _js_files() -> list[Path]:
    """Collect all .js files under ui/scripts/."""
    return sorted((ROOT / "ui" / "scripts").rglob("*.js"))


def _enclosing_function(lines: list[str], line_idx: int) -> str | None:
    """Walk backwards to find the nearest def/async def name."""
    for i in range(line_idx - 1, -1, -1):
        m = re.match(r"\s*(?:async\s+)?def\s+(\w+)", lines[i])
        if m:
            return m.group(1)
    return None


def _docstring_line_mask(lines: list[str]) -> list[bool]:
    """Return a list[bool] where True means 'this line is inside a
    triple-quoted string (docstring or otherwise)'.

    Tracks opening/closing of \"\"\" and ''' across lines. A single-line
    triple-quote `\"\"\"foo\"\"\"` opens and closes on the same line and
    that line is NOT marked (the contents are scanned normally).
    """
    mask = [False] * len(lines)
    open_quote: str | None = None
    for i, line in enumerate(lines):
        if open_quote is None:
            # Look for an unterminated opener on this line.
            for q in ('"""', "'''"):
                first = line.find(q)
                if first == -1:
                    continue
                rest = line[first + 3:]
                second = rest.find(q)
                if second == -1:
                    # Opens here, doesn't close — subsequent lines are inside.
                    open_quote = q
                    break
                # Opens and closes on the same line; no carry-over.
            # Current line keeps mask[i] = False so its code is scanned.
        else:
            mask[i] = True
            closer = line.find(open_quote)
            if closer != -1:
                open_quote = None
    return mask


_STRING_LITERAL_OPENERS = ('"', "'")


def _is_inside_string_literal(line: str, pos: int) -> bool:
    """Return True if column `pos` of `line` falls inside a single-line
    string literal. Handles plain ", ', and escaped quotes; does NOT
    handle triple-quoted strings (callers should pre-filter those via
    _docstring_line_mask)."""
    quote: str | None = None
    i = 0
    while i < pos and i < len(line):
        c = line[i]
        if quote is None and c in _STRING_LITERAL_OPENERS:
            quote = c
        elif quote is not None and c == "\\":
            i += 2
            continue
        elif quote is not None and c == quote:
            quote = None
        i += 1
    return quote is not None


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _finding_is_error(finding: str) -> bool:
    """Return True when a formatted finding has the hard-error marker.

    On Windows without ANSI color support, ``_red("x")`` returns plain ``x``.
    A substring check for that marker misclassifies any warning containing the
    letter x, including "except Exception", as an error.
    """
    plain = _ANSI_RE.sub("", finding).lstrip()
    return plain.startswith("x ")

# ---------------------------------------------------------------------------
# 1. Empty Model String Scanner
# ---------------------------------------------------------------------------

_EMPTY_MODEL_RE = re.compile(r"InternalChatRequest\(")
_MODEL_ASSIGN_RE = re.compile(r"\w+\.model\s*=")

def check_empty_model(suppressions: list[str], verbose: bool) -> tuple[list[str], int]:
    """Find InternalChatRequest(model='') without a model override within 5 lines."""
    findings: list[str] = []
    suppressed = 0

    for pyfile in _python_files():
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        rel = _rel(pyfile)

        for i, line in enumerate(lines):
            if not _EMPTY_MODEL_RE.search(line):
                continue

            # Gather the constructor call (may span multiple lines)
            # Look at current line + next 5 for model="" or model=''
            window = "\n".join(lines[i : i + 6])
            if not re.search(r'model\s*=\s*["\']["\']', window):
                continue  # model is non-empty or variable — fine

            # Check if model is overwritten within the next 5 lines after the call
            safe = False
            for j in range(i + 1, min(i + 6, len(lines))):
                if _MODEL_ASSIGN_RE.search(lines[j]):
                    safe = True
                    break

            if safe:
                continue

            line_no = i + 1
            if _is_suppressed(suppressions, rel, line_no):
                suppressed += 1
                if verbose:
                    findings.append(f"  [suppressed] {rel}:{line_no}  model=\"\" without override")
                continue

            findings.append(
                f"  {_yellow('!')} {rel}:{line_no}  "
                f"InternalChatRequest(model=\"\") — model never overwritten within 5 lines"
            )

    return findings, suppressed

# ---------------------------------------------------------------------------
# 2. Silent Exception Scanner
# ---------------------------------------------------------------------------

_CLEANUP_FUNCS = {
    "__del__", "close", "cleanup", "shutdown", "aclose", "_cleanup", "_shutdown",
    "reset", "soft_reset", "hard_reset", "_reset",  # state reset methods
    # Async lifecycle + connection teardown
    "lifespan", "_lifespan", "on_shutdown", "stop", "_stop", "stop_all",
    "disconnect", "_disconnect", "unregister", "_unregister", "deregister",
    "evict", "_evict", "release", "_release", "teardown", "_teardown",
    "dispose", "_dispose", "destroy", "_destroy", "sweep", "_sweep",
    # Idempotent cancel paths
    "cancel", "_cancel", "abort", "_abort",
}

# Directories/files where except-pass is expected (hardware probing, optional imports,
# feature detection, GPU capability checks).  These aren't bugs — they're intentional
# graceful degradation for optional subsystems.
_SILENT_EXCEPT_SAFE_DIRS = {
    "augmentum/image/",        # GPU/VRAM probing, pipeline feature detection
    "augmentum/models/llama_cpp",  # optional llama.cpp backend availability
    "augmentum/voice/vad",     # Silero model state reset — defensive
    "augmentum/utils/datetime_context",  # timezone probing — intentional fallback chain
}

# Function name patterns where except-pass / except-log.debug is intentional.
# Three classes:
#   * probing / detection — feature detection, capability discovery
#   * best-effort getters — metadata extraction, optional parsing, lazy loaders
#   * teardown / sweepers — eviction, release, unregister, dispose
_PROBE_FUNC_PATTERNS = re.compile(
    r"(?:detect|probe|check|test|_get_vram|_get_gpu|_try|_resolve|_safe_import|list_models)"
    r"|(?:_safe_|_safe$|^safe_)"  # any name with _safe_ infix or safe_ prefix
    r"|(?:^_?(?:extract|parse|load|render|serialise|serialize|dump|normalise|normalize)_)"
    r"|(?:_with_fallback$|_or_default$|_or_none$|_or_empty$|_or_skip$)"
    r"|(?:^_?(?:teardown|release|evict|unregister|unsubscribe|deregister|sweep|reap|prune|dispose|destroy|kill)_)"
    r"|(?:^_[a-z]+_available$)"  # _thing_available() functions
    r"|(?:^_?(?:webhook|notify|broadcast|publish|emit)(?:_|$))"  # fire-and-forget notifies
    r"|(?:^_?(?:enrich|hydrate|augment|annotate|backfill|hint)_)"  # nice-to-have additions
    r"|(?:^_?(?:capture|record|track|log|measure|stat|count|sample|snapshot)_)"  # telemetry recorders
    r"|(?:^_?(?:finish|finalize|finalise|close_out|commit|wrap_up)_)"  # finalizers
    r"|(?:^_?(?:verify|validate|sanity)_)"  # verifiers — return bool, not raise
    r"|(?:^_?(?:flush|persist|sync|push|store|save|drain)_)"  # background persistence
    r"|(?:^_?(?:bind|attach|hook|register)_)"  # late-binding registration
    r"|(?:^_?(?:recall|retrieve|inject|preload|prefetch)_)"  # best-effort retrieval
    r"|(?:^_?(?:peek|sniff|introspect|inspect)_)"  # observation-only
    r"|(?:^_get_[a-z_]+(?:_or_none|_or_default|_or_empty|_or_skip|_safe|_context|_guide|_state|_info|_summary|_snapshot)$)",  # safe getters
    re.IGNORECASE,
)

def check_silent_exceptions(suppressions: list[str], verbose: bool) -> tuple[list[str], int]:
    """Find except Exception blocks with only pass or only log.debug in non-cleanup contexts.

    Smart exclusions:
    - Cleanup functions (__del__, close, cleanup, shutdown, etc.)
    - finally: blocks
    - Hardware probing / feature detection directories (image/, models/llama_cpp)
    - Probe/detect function patterns
    - contextlib.suppress() — already reviewed in security audit
    """
    findings: list[str] = []
    suppressed = 0

    for pyfile in _python_files():
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        rel = _rel(pyfile)

        # Skip entire files in known-safe probe directories
        if any(rel.startswith(safe) for safe in _SILENT_EXCEPT_SAFE_DIRS):
            continue

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not re.match(r"except\s+Exception", stripped):
                continue

            line_no = i + 1

            # Determine the body of the except block
            except_indent = len(line) - len(line.lstrip())
            body_lines: list[str] = []
            for j in range(i + 1, min(i + 10, len(lines))):
                bline = lines[j]
                if not bline.strip():
                    continue
                bind = len(bline) - len(bline.lstrip())
                if bind <= except_indent:
                    break
                body_lines.append(bline.strip())

            if not body_lines:
                continue

            # Check enclosing function
            func_name = _enclosing_function(lines, i)
            is_cleanup = func_name in _CLEANUP_FUNCS if func_name else False

            # Check if inside a finally block
            in_finally = False
            for k in range(i - 1, max(i - 5, -1), -1):
                if lines[k].strip().startswith("finally:"):
                    in_finally = True
                    break

            if is_cleanup or in_finally:
                continue

            # Check if enclosing function is a probe/detect pattern
            if func_name and _PROBE_FUNC_PATTERNS.search(func_name):
                continue

            # Check if the try block contains an import (optional import pattern)
            try_block_start = None
            for k in range(i - 1, max(i - 20, -1), -1):
                if lines[k].strip().startswith("try:"):
                    try_block_start = k
                    break
            if try_block_start is not None:
                try_body = "\n".join(lines[try_block_start:i])
                if "import " in try_body:
                    continue  # Optional import — except pass is fine

            # Pattern A: body is only 'pass'
            if body_lines == ["pass"]:
                if _is_suppressed(suppressions, rel, line_no, func_name):
                    suppressed += 1
                    if verbose:
                        findings.append(f"  [suppressed] {rel}:{line_no}  except Exception: pass")
                    continue
                findings.append(
                    f"  {_red('x')} {rel}:{line_no}  "
                    f"except Exception: pass — should at minimum log the error"
                )
                continue

            # Pattern B: body is only log.debug (non-cleanup)
            if all(re.match(r"log\.debug\(", bl) for bl in body_lines):
                if _is_suppressed(suppressions, rel, line_no, func_name):
                    suppressed += 1
                    if verbose:
                        findings.append(f"  [suppressed] {rel}:{line_no}  except Exception: log.debug")
                    continue
                findings.append(
                    f"  {_yellow('!')} {rel}:{line_no}  "
                    f"except Exception with only log.debug — consider log.warning for user-visible paths"
                )

    return findings, suppressed

# ---------------------------------------------------------------------------
# 3. Frontend Fetch Resilience Scanner
# ---------------------------------------------------------------------------

_FETCH_RE = re.compile(r"(?<![\w-])fetch\s*\(")  # disallow hyphenated forms (auto-fetch, pre-fetch, ...)
_BEACON_RE = re.compile(r"navigator\.sendBeacon\s*\(")
_CATCH_CHAIN_RE = re.compile(r"\.catch\s*\(")
_TRY_RE = re.compile(r"\btry\s*\{")

def check_fetch_resilience(suppressions: list[str], verbose: bool) -> tuple[list[str], int]:
    """Find fetch() calls without error handling."""
    findings: list[str] = []
    suppressed = 0

    for jsfile in _js_files():
        text = jsfile.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        rel = _rel(jsfile)

        for i, line in enumerate(lines):
            if not _FETCH_RE.search(line):
                continue
            # Skip sendBeacon (fire-and-forget by design)
            if _BEACON_RE.search(line):
                continue
            # Skip fetch references inside JS comments (line or block),
            # documentation strings, or markdown inside template literals.
            stripped = line.lstrip()
            if stripped.startswith(("//", "*", "/*")):
                continue

            line_no = i + 1

            # Check 1: .catch() somewhere in this fetch's chain. Multi-line
            # fetches with options spread `fetch(url, { ... })` across
            # several lines, so the `.catch(...)` at the end of the chain
            # can be 5-15 lines below the `fetch(` call. Track brace +
            # paren depth from the start of the fetch line; the chain
            # ends when we return to baseline depth AND see a statement
            # terminator (`;`) that isn't inside a nested block. A
            # `console.debug(...);` inside a `.then((x) => { ... })`
            # body is at depth>=1 so doesn't terminate the chain.
            has_catch_chain = False
            depth = 0  # net (parens + braces) opened since fetch line
            for j in range(i, min(i + 24, len(lines))):
                lj = lines[j]
                if _CATCH_CHAIN_RE.search(lj):
                    has_catch_chain = True
                    break
                # Update depth (cheap approximation — ignores strings)
                depth += lj.count("(") + lj.count("{") - lj.count(")") - lj.count("}")
                # End-of-chain: depth back to 0 (or negative on the line
                # that contains the closing of an arg list / object) AND
                # the line ends with `;` not inside an open structure.
                if j > i and depth <= 0 and lj.rstrip().endswith(";"):
                    break

            if has_catch_chain:
                continue

            # Check 2: is the fetch inside a try block?
            # First check same-line: `try { await fetch(...); } catch {}`
            # is a valid one-liner the backward walk would miss.
            fetch_pos = _FETCH_RE.search(line).start()
            same_line_try = _TRY_RE.search(line)
            if same_line_try and same_line_try.start() < fetch_pos:
                continue

            # Walk backwards looking for 'try {' at same or lower indent.
            # UI handler bodies can be hundreds of lines (settings.js has
            # saveToolSettings() at ~340 lines, fetch ~200 lines past
            # the enclosing try) — so the walk must be wide. To keep
            # sibling-function tries from false-suppressing, stop at the
            # first top-level construct above the fetch: a line at indent 0
            # that opens a function / arrow-fn / class / object-method.
            # Brace-depth still filters tries that opened then closed
            # before reaching the fetch.
            fetch_indent = len(line) - len(line.lstrip())
            in_try = False
            brace_depth = 0
            _TOP_LEVEL_OPENER_RE = re.compile(
                r"^(?:export\s+)?(?:async\s+)?(?:function|class)\s|"
                r"^(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\(",
            )
            for k in range(i - 1, max(i - 400, -1), -1):
                bk = lines[k]
                brace_depth += bk.count("}") - bk.count("{")
                if _TRY_RE.search(bk):
                    try_indent = len(bk) - len(bk.lstrip())
                    if try_indent <= fetch_indent and brace_depth <= 0:
                        in_try = True
                        break
                # Top-level construct opener at column 0 → the fetch's
                # enclosing function starts at or before here. If we
                # haven't found an enclosing try yet, no try ever
                # enclosed the fetch.
                if _TOP_LEVEL_OPENER_RE.match(bk):
                    break

            if in_try:
                continue

            # Check 3: the fetch is in a helper that propagates errors
            # via `throw` — `if (!resp.ok) throw new Error(...)` is a
            # well-formed pattern that lets the caller's try/catch (or
            # awaiting code) handle the error. Look ahead ~16 lines from
            # the fetch for a `throw` statement; multi-line fetch +
            # options + status-check + detail-parse can push the throw
            # 10+ lines past the fetch call.
            propagates_via_throw = False
            for j in range(i, min(i + 24, len(lines))):
                if re.search(r"\bthrow\s+(?:new\s+\w+|\w+\b)", lines[j]):
                    propagates_via_throw = True
                    break

            if propagates_via_throw:
                continue

            # Not protected
            if _is_suppressed(suppressions, rel, line_no):
                suppressed += 1
                if verbose:
                    findings.append(f"  [suppressed] {rel}:{line_no}  unhandled fetch()")
                continue

            short_call = line.strip()[:80]
            findings.append(
                f"  {_yellow('!')} {rel}:{line_no}  "
                f"Unhandled fetch failure: {short_call}"
            )

    return findings, suppressed

# ---------------------------------------------------------------------------
# 4. app.state Consistency Scanner
# ---------------------------------------------------------------------------

_APP_STATE_RE = re.compile(r"(?:request\.app|app)\.state\.(\w+)")
_GETATTR_STATE_RE = re.compile(r"getattr\s*\(\s*(?:request\.app|app)\.state\s*,")

# Functions where direct app.state access is expected (startup, shutdown, lifespan —
# these are where state is being CREATED, not consumed defensively)
_APP_STATE_LIFECYCLE_FUNCS = {
    "lifespan", "_startup", "_shutdown", "startup", "shutdown",
    "on_startup", "on_shutdown", "_lifespan", "startup_event",
    "register_flow_tools_async",  # handler_factory startup wiring
    "_restore_settings", "_build_settings_restore_map",
}

# Route files that are compatibility passthrough layers — they always run with
# initialized backends and the entire route is non-functional without them.
# Bare app.state access is expected here, not a resilience concern.
_APP_STATE_PASSTHROUGH_FILES = {
    "augmentum/proxy/ollama_routes.py",   # Ollama API compat layer
    "augmentum/proxy/openai_routes.py",   # OpenAI API compat layer
    "augmentum/proxy/model_routes.py",    # Model management (requires running backends)
    "augmentum/proxy/provider_routes.py", # Provider management (infrastructure)
}

def check_app_state_access(suppressions: list[str], verbose: bool) -> tuple[list[str], int]:
    """Find unguarded app.state.X access in proxy/ route handlers.

    Excludes:
    - Lifecycle functions (startup/shutdown/lifespan) where state is being built
    - Assignments to app.state (initialization)
    - Lines already using getattr() or hasattr()
    - server.py entirely (it's the lifecycle orchestrator)
    - Passthrough proxy routes (ollama, openai) — always run with initialized state
    """
    findings: list[str] = []
    suppressed = 0

    proxy_dir = ROOT / "augmentum" / "proxy"
    for pyfile in sorted(proxy_dir.rglob("*.py")):
        rel = _rel(pyfile)

        # Skip server.py — it's the lifecycle orchestrator, not a consumer
        if rel.endswith("server.py"):
            continue

        # Skip passthrough proxy files — always run with initialized backends
        if rel in _APP_STATE_PASSTHROUGH_FILES:
            continue

        text = pyfile.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        # Pre-compute which lines are inside a triple-quoted string so we
        # don't flag docstring references like ``app.state.foo`` as code.
        in_docstring = _docstring_line_mask(lines)

        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith("#"):
                continue

            if in_docstring[i]:
                continue

            if _GETATTR_STATE_RE.search(line):
                continue

            # Skip assignments TO app.state (initialization)
            if re.search(r"(?:request\.app|app)\.state\.\w+\s*=", line):
                continue

            match = _APP_STATE_RE.search(line)
            if not match:
                continue

            # Skip if the matched reference is inside a string literal
            # on this single line (e.g. error messages that quote the
            # attribute name for diagnostic purposes).
            if _is_inside_string_literal(line, match.start()):
                continue

            attr = match.group(1)
            line_no = i + 1

            # Skip lifecycle functions
            func_name = _enclosing_function(lines, i)
            if func_name in _APP_STATE_LIFECYCLE_FUNCS:
                continue

            # Skip if the attribute is accessed inside a hasattr check nearby
            context_window = "\n".join(lines[max(0, i - 3) : i + 1])
            if f'hasattr(request.app.state, "{attr}")' in context_window:
                continue
            if f"hasattr(app.state, \"{attr}\")" in context_window:
                continue

            if _is_suppressed(suppressions, rel, line_no, func_name):
                suppressed += 1
                if verbose:
                    findings.append(f"  [suppressed] {rel}:{line_no}  app.state.{attr}")
                continue

            findings.append(
                f"  {_yellow('!')} {rel}:{line_no}  "
                f"Unguarded app.state.{attr} — use getattr(…, \"{attr}\", None) for resilience"
            )

    return findings, suppressed

# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print(_bold("\n  Augmentum Runtime Bug-Pattern Scanner"))
    print(_bold("  " + "=" * 44) + "\n")

    suppressions = _load_suppressions()

    all_errors: list[str] = []
    all_warnings: list[str] = []
    total_suppressed = 0

    # --- 1. Empty Model ---
    print(_cyan("  [1/4] Scanning empty model strings..."))
    model_findings, model_sup = check_empty_model(suppressions["empty_model"], verbose)
    total_suppressed += model_sup
    for f in model_findings:
        if "[suppressed]" in f:
            if verbose:
                print(f)
        else:
            all_warnings.append(f)

    # --- 2. Silent Exceptions ---
    print(_cyan("  [2/4] Scanning silent exception handlers..."))
    exc_findings, exc_sup = check_silent_exceptions(suppressions["silent_exception"], verbose)
    total_suppressed += exc_sup
    for f in exc_findings:
        if "[suppressed]" in f:
            if verbose:
                print(f)
        elif _finding_is_error(f):
            all_errors.append(f)
        else:
            all_warnings.append(f)

    # --- 3. Frontend Fetch ---
    print(_cyan("  [3/4] Scanning frontend fetch resilience..."))
    fetch_findings, fetch_sup = check_fetch_resilience(suppressions["fetch_resilience"], verbose)
    total_suppressed += fetch_sup
    for f in fetch_findings:
        if "[suppressed]" in f:
            if verbose:
                print(f)
        else:
            all_warnings.append(f)

    # --- 4. app.state Access ---
    print(_cyan("  [4/4] Scanning app.state access guards..."))
    state_findings, state_sup = check_app_state_access(suppressions["app_state_access"], verbose)
    total_suppressed += state_sup
    for f in state_findings:
        if "[suppressed]" in f:
            if verbose:
                print(f)
        else:
            all_warnings.append(f)

    # --- Report ---
    print()
    print(_bold("  Summary"))
    print(f"    Python files scanned:   {len(_python_files())}")
    print(f"    JS files scanned:       {len(_js_files())}")
    print(f"    Suppressions applied:   {total_suppressed}")
    print()

    if all_errors:
        print(_red(f"  ERRORS ({len(all_errors)}):"))
        for e in all_errors:
            print(e)
        print()

    if all_warnings:
        print(_yellow(f"  WARNINGS ({len(all_warnings)}):"))
        for w in all_warnings:
            print(w)
        print()

    if not all_errors and not all_warnings:
        print(_green("  All clear — no runtime bug patterns detected."))
        print()

    if all_errors:
        print(_red(f"  {len(all_errors)} error(s), {len(all_warnings)} warning(s)"))
        return 1
    elif all_warnings:
        print(_yellow(f"  0 errors, {len(all_warnings)} warning(s)"))
        return 0
    else:
        print(_green("  0 errors, 0 warnings"))
        return 0


if __name__ == "__main__":
    sys.exit(main())
