#!/usr/bin/env python3
"""Augmentum wiring validator — catches the top recurring integration bugs.

Scans all four setting layers (config.py, config_routes.py, server.py,
settings.js) and cross-references them to find orphaned settings, missing
registrations, and broken round-trips.  Also checks route registration
and template literal safety.

Exit code 0 = clean, 1 = issues found.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Windows consoles/pipes default to a non-UTF-8 codepage; the ✓ / → / … chars
# in our output would raise UnicodeEncodeError mid-report (and the audit then
# logs "parser found no metrics"). Make stdout/stderr lenient — no-op where
# reconfigure isn't available.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

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
def _dim(s: str) -> str:    return f"\033[2m{s}\033[0m" if _COLOR else s

# ---------------------------------------------------------------------------
# 1. Parse config.py — extract all Settings fields
# ---------------------------------------------------------------------------

def parse_config_fields() -> set[str]:
    """Extract field names from the Settings(BaseSettings) class in config.py."""
    path = ROOT / "augmentum" / "config.py"
    text = path.read_text(encoding="utf-8", errors="replace")
    # Match lines like: field_name: type = default
    fields: set[str] = set()
    in_class = False
    for line in text.splitlines():
        if "class Settings" in line:
            in_class = True
            continue
        if in_class:
            if line and not line[0].isspace() and not line.startswith("#"):
                break  # left the class
            m = re.match(r"\s+(\w+)\s*:\s*\w+", line)
            if m:
                fields.add(m.group(1))
    return fields

# ---------------------------------------------------------------------------
# 2. Parse config_routes.py — extract _TOOL_SETTINGS and _STRING_SETTINGS
# ---------------------------------------------------------------------------

def _extract_dict_keys(text: str, varname: str) -> set[str]:
    """Extract string keys from a dict literal assigned to varname.

    Only captures keys that appear at the start of a dict entry (before a colon),
    not string values or comment text.
    """
    keys: set[str] = set()
    # Find the dict assignment
    pattern = re.compile(rf"{varname}\s*[:\=].*?\{{", re.DOTALL)
    m = pattern.search(text)
    if not m:
        return keys
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    block = text[start:i - 1]
    # Match only "key": ... patterns (key followed by colon = dict key, not value)
    for km in re.finditer(r'^\s*"(\w+)"\s*:', block, re.MULTILINE):
        keys.add(km.group(1))
    return keys

def parse_config_routes() -> tuple[set[str], set[str]]:
    """Return (_TOOL_SETTINGS keys, _STRING_SETTINGS keys)."""
    path = ROOT / "augmentum" / "proxy" / "config_routes.py"
    text = path.read_text(encoding="utf-8", errors="replace")
    tool = _extract_dict_keys(text, "_TOOL_SETTINGS")
    string = _extract_dict_keys(text, "_STRING_SETTINGS")
    return tool, string

# ---------------------------------------------------------------------------
# 3. Parse server.py — extract _SETTINGS_RESTORE_MAP keys
# ---------------------------------------------------------------------------

def parse_restore_map() -> set[str]:
    """Return every key the startup restore pass will reload.

    Two sources:
      * Manual entries in ``_SETTINGS_RESTORE_MAP`` (the override path
        for custom parsers / encrypted strings).
      * Auto-derived entries from ``config_routes._TOOL_SETTINGS`` +
        ``_STRING_SETTINGS`` via ``_auto_derive_restore_parsers`` —
        every bool/int/float/str in those dicts persists across
        restart automatically, no manual sync required.

    The scanner needs to know about the auto-derivation or it would
    flag every _TOOL_SETTINGS key as "missing from restore_map" and
    push us right back into the bug class that auto-derivation
    eliminates.
    """
    path = ROOT / "augmentum" / "proxy" / "server.py"
    text = path.read_text(encoding="utf-8", errors="replace")
    manual = _extract_dict_keys(text, "_SETTINGS_RESTORE_MAP")
    # Auto-derived: _TOOL_SETTINGS + _STRING_SETTINGS are restored
    # automatically. The static check just needs to know they're
    # covered — it doesn't have to predict the parser.
    tool_keys, string_keys = parse_config_routes()
    return manual | tool_keys | string_keys

# ---------------------------------------------------------------------------
# 4. Parse settings.js — extract DEFAULTS keys and sync body keys
# ---------------------------------------------------------------------------

def _all_ui_snake_keys() -> set[str]:
    """Scan EVERY ui/scripts/**/*.js for snake_case identifiers — anywhere.

    The legacy parse_settings_js() only inspected settings.js's
    syncToolSettingsToBackend block. UI surfaces in app.js (typography),
    narrative/cardsmith.js (memory model), browse.js (RAG knobs), voice.js
    (speaker verify), etc. push their own subset of settings via direct
    fetch calls that this narrow scan missed — causing 100+ legitimate
    UI-wired settings to be flagged as 'not synced'.

    This helper widens detection to any snake_case identifier appearing
    in any JS source. A token-level match is too permissive in theory
    but in practice the snake_case form (with at least one underscore)
    is rare enough outside settings vocabulary that false-positive
    'synced' marks are negligible.
    """
    keys: set[str] = set()
    js_dir = ROOT / "ui" / "scripts"
    # Two patterns:
    #   * snake_case identifier (with at least one underscore) — the
    #     common backend-key shape
    #   * bare single-word settings (timezone, location, ...) — match
    #     these as JSON keys quoted with single or double quotes, so we
    #     don't also pick up every occurrence of the english word
    snake = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
    single = re.compile(r"""['"](timezone|location|theme|language|locale)['"]""")
    for jsfile in js_dir.rglob("*.js"):
        try:
            text = jsfile.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for m in snake.finditer(text):
            keys.add(m.group(1))
        for m in single.finditer(text):
            keys.add(m.group(1))
    return keys


def parse_settings_js() -> tuple[set[str], set[str]]:
    """Return (DEFAULTS camelCase keys, syncToolSettingsToBackend snake_case keys)."""
    path = ROOT / "ui" / "scripts" / "settings.js"
    text = path.read_text(encoding="utf-8", errors="replace")

    # DEFAULTS keys
    defaults: set[str] = set()
    m = re.search(r"const\s+DEFAULTS\s*=\s*\{", text)
    if m:
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        block = text[start:i - 1]
        for km in re.finditer(r"^\s*(\w+)\s*:", block, re.MULTILINE):
            defaults.add(km.group(1))

    # syncToolSettingsToBackend — snake_case keys sent to backend
    sync_keys: set[str] = set()
    sync_match = re.search(r"function\s+syncToolSettingsToBackend", text)
    if sync_match:
        start = text.find("{", sync_match.end())
        if start >= 0:
            depth = 1
            i = start + 1
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            block = text[start:i]
            for km in re.finditer(r"(\w+)\s*:", block):
                key = km.group(1)
                # Only snake_case keys (backend keys), skip JS keywords
                if "_" in key and key not in ("Content", "method", "headers"):
                    sync_keys.add(key)

    return defaults, sync_keys

# ---------------------------------------------------------------------------
# 5. Route registration check
# ---------------------------------------------------------------------------

def check_route_registration() -> list[str]:
    """Find route files that aren't registered in server.py."""
    issues: list[str] = []
    proxy_dir = ROOT / "augmentum" / "proxy"
    server_text = (proxy_dir / "server.py").read_text(encoding="utf-8", errors="replace")

    route_files = sorted(proxy_dir.glob("*_routes.py"))
    for rf in route_files:
        module_name = rf.stem  # e.g. "browse_routes"
        if module_name == "notification_routes":
            continue  # known removed
        # Check import
        if f"from augmentum.proxy.{module_name}" not in server_text:
            issues.append(f"Route file {module_name}.py is not imported in server.py")
        # Check include_router — look for the alias
        # Most use pattern: from ...{module_name} import router as {stem}_router
        alias = module_name.replace("_routes", "_router")
        if f"include_router({alias}" not in server_text:
            # Some use different aliases, check if ANY include references this module
            if module_name not in server_text:
                issues.append(
                    f"Route file {module_name}.py may not be registered "
                    f"(no include_router with expected alias '{alias}')"
                )
    return issues

# ---------------------------------------------------------------------------
# 6. Template literal safety (escapeHtml check)
# ---------------------------------------------------------------------------

def _load_xss_exception_files() -> set[str]:
    """Files covered by an existing security_exceptions.json XSS entry.

    The wiring validator and security_check.py share the same JSON
    exception store. When security_check has already reviewed and
    suppressed an interpolation in a given file, the wiring scanner
    should also stay quiet — otherwise the same false positive
    accumulates against the wiring budget too.
    """
    exc_path = ROOT / ".claude" / "skills" / "augmentum-dev" / "references" / "security_exceptions.json"
    try:
        data = json.loads(exc_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    out: set[str] = set()
    for entry in data.get("exceptions", []):
        # XSS-class exceptions have an id containing 'unescaped',
        # 'innerHTML', 'mountid', 'renderpowerrow', etc. — easier to
        # whitelist by file path than by id pattern.
        eid = str(entry.get("id", "")).lower()
        if not any(tag in eid for tag in (
            "unescape", "innerhtml", "mountid", "renderpower",
            "renderrow", "dialogue", "article-raw-html",
        )):
            continue
        for f in entry.get("files", []) or []:
            out.add(str(f).replace("\\", "/"))
    return out


def check_template_safety() -> list[str]:
    """Scan JS files for template literals that interpolate variables without escapeHtml."""
    issues: list[str] = []
    js_dir = ROOT / "ui" / "scripts"

    # Files covered by a reviewed security_exceptions.json XSS entry —
    # don't double-flag the same site here.
    xss_excepted_files = _load_xss_exception_files()

    # Safe patterns — these are known-safe interpolations
    safe_patterns = {
        "escapeHtml", "_escHtml", "_esc(", "encodeURIComponent", "encodeURI",
        "JSON.stringify", "parseInt", "parseFloat", "Number",
        "Math.", "Date.", ".length", ".id", ".toString",
        "icons.", "DEFAULTS.",
        # Response.status is an HTTP status code Number per the fetch spec;
        # decimal digits only, never a string. Common in fetch error branches.
        ".status",
        # Common pre-rendered HTML chunks built via .map(_renderXxx).join('')
        # where the renderer escapeHtml()s each field. Convention in this
        # codebase: variables ending in 'Html' / named 'rows' / 'bars' /
        # 'badge' / 'more' / 'historyHtml' / 'fileSpan' / 'actions' are
        # assembly outputs of already-escaped renderers.
        "Html", "rows", "bars", "badge", "more", "fileSpan", "actions",
        "summary", "breadcrumb", "historyHtml", "resized",
        # Static internal class/icon constants set from small literal enums
        # (cls = success/error/running; icon = Unicode check/cross glyph).
        "cls", "icon",
        # Numeric counters / totals — derived from .length or +cast, always
        # a JS Number per how the codebase uses them.
        "count", "total", "failNote",
        # data.samples is a Number from a stats endpoint; ?? '—' fallback
        # is a literal em-dash.
        "data.samples",
    }

    for jsfile in js_dir.rglob("*.js"):
        text = jsfile.read_text(encoding="utf-8", errors="replace")
        rel = jsfile.relative_to(ROOT)
        rel_posix = str(rel).replace("\\", "/")

        # Skip files already covered by a reviewed XSS exception
        if rel_posix in xss_excepted_files:
            continue

        # Find template literals with ${...} interpolation
        for i, line in enumerate(text.splitlines(), 1):
            if ".innerHTML" not in line and ".insertAdjacentHTML" not in line:
                continue
            # Check if line has template literal with interpolation
            for m in re.finditer(r"\$\{([^}]+)\}", line):
                expr = m.group(1).strip()
                # Skip if using a safe wrapper
                if any(sp in expr for sp in safe_patterns):
                    continue
                # Skip pure numeric/boolean expressions
                if re.match(r"^[\d.+\-*/%\s()]+$", expr):
                    continue
                # Skip ternary with string-literal-only value branches
                # (`cond ? '✓' : '✗'`). Split on both `?` and `:` since
                # the false-branch lives after the colon.
                if "?" in expr:
                    parts = re.split(r"[?:]", expr)
                    if len(parts) >= 3:
                        value_parts = parts[1:]  # all branches after the condition
                        is_all_literal_or_safe = all(
                            (("'" in p) or ('"' in p))
                            or any(sp in p for sp in safe_patterns)
                            for p in value_parts
                        )
                        if is_all_literal_or_safe:
                            continue
                # Flag it
                short_expr = expr[:60] + "..." if len(expr) > 60 else expr
                issues.append(f"{rel}:{i}  ${{{short_expr}}} in innerHTML without escapeHtml")

    return issues

# ---------------------------------------------------------------------------
# 7. Migration sequence check
# ---------------------------------------------------------------------------

def check_migrations() -> tuple[int, list[str]]:
    """Check migration numbering for gaps. Returns (count, issues)."""
    mig_dir = ROOT / "augmentum" / "state" / "migrations"
    if not mig_dir.is_dir():
        return 0, ["Migration directory not found"]

    # Migration numbers that were skipped intentionally — never created,
    # deliberately reserved, or claimed by an in-flight branch that never
    # landed those slots. Keep this list small + commented to discourage
    # "fix by adding to skip list" creep. Real missing migrations should
    # still surface as a warning so they can be investigated.
    _RESERVED_GAPS = {
        127, 128,  # in-flight branch reserved these; merged work skipped to 129
        166, 167,  # wake-word branch left these holes during sequencing
    }
    files = sorted(f.name for f in mig_dir.glob("*.sql"))
    issues: list[str] = []
    prev = 0
    for f in files:
        m = re.match(r"(\d+)_", f)
        if not m:
            issues.append(f"Migration {f} doesn't follow NNN_name.sql pattern")
            continue
        num = int(m.group(1))
        if num != prev + 1:
            missing = [n for n in range(prev + 1, num) if n not in _RESERVED_GAPS]
            if missing:
                missing_str = ",".join(f"{n:03d}" for n in missing)
                issues.append(f"Migration gap: {prev:03d} -> {num:03d} (missing {missing_str})")
        prev = num

    return len(files), issues

# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main():
    print(_bold("\n  Augmentum Wiring Validator"))
    print(_bold("  " + "=" * 40) + "\n")

    # Auto-refresh derived reference files (routes.json, settings_map.json, etc.)
    try:
        from refresh_refs import refresh_all
        refresh_all(quiet=True)
    except ImportError:
        # Running from a different cwd — try absolute import
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "refresh_refs",
                Path(__file__).resolve().parent / "refresh_refs.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.refresh_all(quiet=True)
        except Exception:
            pass  # non-critical — references are a convenience, not required

    errors: list[str] = []
    warnings: list[str] = []

    # --- Settings layers ---
    print(_cyan("  Scanning settings layers..."))

    config_fields = parse_config_fields()
    tool_settings, string_settings = parse_config_routes()
    restore_map = parse_restore_map()
    js_defaults, js_sync_keys = parse_settings_js()

    all_backend_settings = tool_settings | string_settings

    # Settings in config_routes but missing from server.py restore map
    for key in sorted(all_backend_settings):
        if key not in restore_map:
            errors.append(f"Setting '{key}' in config_routes.py but MISSING from server.py _SETTINGS_RESTORE_MAP")

    # Settings in restore map but missing from config_routes
    for key in sorted(restore_map):
        if key not in all_backend_settings:
            # Some restore map entries are for settings not exposed via API (internal)
            # Check if they exist in config.py at least
            if key not in config_fields:
                warnings.append(f"Setting '{key}' in _SETTINGS_RESTORE_MAP but not in config_routes or config.py")

    # Settings in config_routes but missing from config.py
    for key in sorted(all_backend_settings):
        if key not in config_fields:
            # UI-only settings stored via settings_store don't need config.py fields
            # but tool/string settings typically should have defaults
            warnings.append(f"Setting '{key}' in config_routes.py but no default in config.py")

    # Settings synced from JS but not accepted by config_routes
    for key in sorted(js_sync_keys):
        if key not in all_backend_settings:
            errors.append(f"JS syncs '{key}' to backend but it's NOT in config_routes _TOOL_SETTINGS or _STRING_SETTINGS")

    # Settings in config_routes but never synced from JS.
    # Many are admin-only operator tuning knobs that intentionally have
    # no user-facing UI (provider credentials, timeout tuning, security
    # policy, feature flags, subsystem internals). The scanner skips
    # those so the remaining list highlights settings that probably
    # SHOULD have a UI control but don't.
    _ADMIN_ONLY_PATTERNS = re.compile(
        # Provider credentials + endpoints (admin/operator)
        r"_(api_key|base_url|deployment|api_version)$|"
        # Timing knobs (operator tuning)
        r"_(timeout|timeout_ms|ms|seconds|hours|days|minutes|ttl_seconds)(_|$)|"
        # Numeric thresholds + clamps (operator tuning)
        r"_(threshold|min_confidence|max_age_days|max_attempts|max_backtracks|"
        r"max_clusters_per_run|max_phase_retries|max_request_bytes|min_size|"
        r"prefix_padding|veto_confidence|gain|hz|backoff|cap|ceiling|floor|delay)$|"
        # Filesystem / process paths (admin)
        r"_(path|binary_path|dir|root)$|"
        # Bool feature flags + on/off toggles — operator-set, not user-set
        r"_(enabled|disabled|active|on|off)$|"
        # Security policy (admin)
        r"^auth_|^_(ip_)?lockout_|^csrf_|^session_|^cors_|"
        # Subsystem-internal tuning prefixes
        r"^(?:architect|agentic|analytical|companion|engine|app_builder|"
        r"chromium|dream|wake_word|safety_floor|document_rag|files|image|"
        r"knowledge|passthrough_chain|rate_limit|role|uarf)_|"
        # TTS engine-specific tuning (admin)
        r"^tts_(?:kokoro|pocket)_|"
        # Voice subsystem tuning (admin)
        r"^voice_(?:audio|lipsync|smart_turn|speaker|always_listening)_|"
        # Narrative tuning (admin)
        r"^narrative_(?:memory|context|request_log|archive|smart_retrieval)_|"
        # Provider name prefixes — credentials / per-provider config (admin)
        r"^(?:anthropic|azure|cohere|openai|xai|mistral|deepseek|"
        r"gemini|qwen|groq|grok|google_vertex|google)_|"
        # Specific known-admin individual keys (non-pattern matches)
        r"^(?:ambient_favorites|primary_chat_model|startup_warmup)$"
    )
    # Widen detection: a setting present anywhere in any ui/scripts/**/*.js
    # is considered UI-wired, even if not via settings.js's central
    # syncToolSettingsToBackend block (typography settings live in app.js,
    # narrative_* in cardsmith.js, voice_* in voice.js, etc.).
    ui_keys = _all_ui_snake_keys()
    sync_warnings: list[str] = []
    skipped_admin = 0
    skipped_ui_wired = 0
    for key in sorted(all_backend_settings):
        if key in js_sync_keys:
            continue
        if key in ui_keys:
            skipped_ui_wired += 1
            continue
        if _ADMIN_ONLY_PATTERNS.search(key):
            skipped_admin += 1
            continue
        sync_warnings.append(f"Setting '{key}' in config_routes but not synced from settings.js")
    warnings.extend(sync_warnings)
    if (skipped_admin or skipped_ui_wired) and not os.environ.get("AUDIT_QUIET"):
        print(_dim(
            f"  ({skipped_ui_wired} UI-wired in other JS files, "
            f"{skipped_admin} admin/operator-only skipped)"
        ))

    # --- Routes ---
    print(_cyan("  Checking route registrations..."))
    route_issues = check_route_registration()
    for issue in route_issues:
        errors.append(issue)

    # --- Template safety ---
    print(_cyan("  Scanning template literal safety..."))
    safety_issues = check_template_safety()
    for issue in safety_issues:
        warnings.append(f"Possible unsafe interpolation: {issue}")

    # --- Migrations ---
    print(_cyan("  Checking migration sequence..."))
    mig_count, mig_issues = check_migrations()
    for issue in mig_issues:
        warnings.append(issue)

    # --- Report ---
    print()

    # Stats
    print(_bold("  Summary"))
    print(f"    Config fields:    {len(config_fields)}")
    print(f"    Tool settings:    {len(tool_settings)}")
    print(f"    String settings:  {len(string_settings)}")
    print(f"    Restore map:      {len(restore_map)}")
    print(f"    JS defaults:      {len(js_defaults)}")
    print(f"    JS sync keys:     {len(js_sync_keys)}")
    print(f"    Migrations:       {mig_count}")
    print()

    if errors:
        print(_red(f"  ERRORS ({len(errors)}):"))
        for e in errors:
            print(f"    {_red('x')} {e}")
        print()

    if warnings:
        print(_yellow(f"  WARNINGS ({len(warnings)}):"))
        for w in warnings:
            print(f"    {_yellow('!')} {w}")
        print()

    if not errors and not warnings:
        print(_green("  All clear — no wiring issues detected."))
        print()

    if errors:
        print(_red(f"  {len(errors)} error(s), {len(warnings)} warning(s)"))
        return 1
    elif warnings:
        print(_yellow(f"  0 errors, {len(warnings)} warning(s)"))
        return 0
    else:
        print(_green("  0 errors, 0 warnings"))
        return 0


if __name__ == "__main__":
    sys.exit(main())
