#!/usr/bin/env python3
"""Generate / refresh derived reference files for the augmentum-dev skill.

Produces three JSON files in the skill's references/ directory:

  routes.json          — every backend endpoint (method, path, handler, file:line)
  frontend_api_calls.json — every fetch()/WebSocket call from the frontend
  settings_map.json    — camelCase↔snake_case mapping + 4-layer coverage

Each file is regenerated ONLY when its source files are newer than the
existing JSON (mtime check).  Call with --force to skip the check.

Designed to be called from validate_wiring.py as a pre-step so references
stay fresh without manual effort.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Root and output paths
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

# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def _needs_refresh(output: Path, sources: list[Path]) -> bool:
    """Return True if output doesn't exist or any source is newer."""
    if not output.exists():
        return True
    out_mtime = output.stat().st_mtime
    for src in sources:
        if src.is_dir():
            for f in src.rglob("*.py"):
                if f.stat().st_mtime > out_mtime:
                    return True
            for f in src.rglob("*.js"):
                if f.stat().st_mtime > out_mtime:
                    return True
        elif src.exists() and src.stat().st_mtime > out_mtime:
            return True
    return False


def _write_json(path: Path, data, label: str) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Refreshed {label} ({path.name})")

# ---------------------------------------------------------------------------
# 1. Route map — every backend endpoint
# ---------------------------------------------------------------------------

def gen_route_map() -> bool:
    """Parse all route files AND server.py for endpoint definitions."""
    output = REFS_DIR / "routes.json"
    proxy_dir = ROOT / "augmentum" / "proxy"
    sources = []
    for candidate in sorted(proxy_dir.glob("*.py")):
        if candidate.name == "__init__.py":
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "APIRouter(" in text:
            sources.append(candidate)
    server_py = proxy_dir / "server.py"
    if server_py.exists():
        sources.append(server_py)

    if not _needs_refresh(output, sources) and "--force" not in sys.argv:
        return False

    endpoints: list[dict] = []

    for rf in sorted(sources):
        text = rf.read_text(encoding="utf-8", errors="replace")
        rel_path = f"augmentum/proxy/{rf.name}"

        # Extract ALL router prefixes: map variable name → prefix
        # e.g., "router" → "/api/models", "llamacpp_router" → "/api/llamacpp"
        router_prefixes: dict[str, str] = {"app": ""}  # @app routes have no prefix
        for rm in re.finditer(
            r'(\w+)\s*=\s*APIRouter\(([^)]*)\)',
            text,
        ):
            var_name = rm.group(1)
            args = rm.group(2)
            pm = re.search(r'prefix\s*=\s*["\']([^"\']+)["\']', args)
            router_prefixes[var_name] = pm.group(1) if pm else ""

        # Extract @<router_var>.method("/path") patterns
        for m in re.finditer(
            r'@(\w+)\.(get|post|put|delete|patch|websocket)\(\s*["\']([^"\']*)["\']',
            text,
        ):
            router_var = m.group(1)
            method = m.group(2).upper()
            path = m.group(3)

            # Look up prefix for this router variable
            prefix = router_prefixes.get(router_var, "")
            # WebSocket routes (`/ws/...`) live on un-prefixed routers in this
            # repo and use absolute paths. Tighten the suffix match so unrelated
            # paths like `/ws-ticket` (a normal POST) still pick up the prefix.
            full_path = prefix + path if not path.startswith("/ws/") else path

            # Strip FastAPI path converters like :path from {name:path}
            full_path = re.sub(r"\{(\w+):\w+\}", r"{\1}", full_path)

            # Find the function name on the next line(s)
            after = text[m.end():]
            func_match = re.search(r"(?:async\s+)?def\s+(\w+)", after)
            func_name = func_match.group(1) if func_match else "?"

            # Line number
            line_no = text[:m.start()].count("\n") + 1

            endpoints.append({
                "method": method,
                "path": full_path,
                "handler": func_name,
                "file": rel_path,
                "line": line_no,
            })

    # Sort by path then method
    endpoints.sort(key=lambda e: (e["path"], e["method"]))
    _write_json(output, {"endpoints": endpoints, "count": len(endpoints)}, "route map")
    return True

# ---------------------------------------------------------------------------
# 2. Frontend API calls — every fetch/WebSocket from JS
# ---------------------------------------------------------------------------

def gen_frontend_api_calls() -> bool:
    """Extract all fetch() and WebSocket calls from frontend JS files.

    Scans every JS surface under ``ui/`` (main app + cast surfaces like
    cast-control/cast-receiver/cast-pair/...) — they all hit the backend.
    Excludes ``ui/lib/`` (vendored third-party: highlight.js, hls.js,
    prism, three, …) which would otherwise pollute the call list with
    library-internal URLs.
    """
    output = REFS_DIR / "frontend_api_calls.json"
    js_dir = ROOT / "ui"

    if not _needs_refresh(output, [js_dir]) and "--force" not in sys.argv:
        return False

    calls: list[dict] = []

    # Module-level URL-alias resolution: many JS files declare
    # `const API = '/api/dream'` then call `fetch(\`${API}/journal\`)`.
    # The template-literal regex below normalizes `${API}` to `{param}` and
    # the call ends up looking like `{param}/journal` — guaranteed no match.
    # Pre-resolve those aliases by substituting the literal value back in
    # before the URL regexes run.
    _URL_ALIAS_RE = re.compile(
        r"""^\s*const\s+([A-Z_][A-Z0-9_]*)\s*=\s*['"`](/api/[^'"`\s]+)['"`]""",
        re.MULTILINE,
    )

    def _resolve_url_aliases(text: str) -> str:
        aliases = dict(_URL_ALIAS_RE.findall(text))
        if not aliases:
            return text
        # Substitute `${ALIAS}` ONLY when followed by a literal path segment
        # (`/something` or a closing backtick). When followed by another
        # interpolation like `${API}${path}` the alias is being used as a
        # *prefix wrapper* in a helper function — the real URL is constructed
        # at call sites of that helper, so substituting here would produce
        # spurious `/api/foo{param}` ghost calls.
        for name, path in aliases.items():
            placeholder = "${" + name + "}"
            # `${API}/x` or `${API}` at end of template → safe to substitute.
            # `${API}${path}` → skip.
            text = re.sub(
                re.escape(placeholder) + r"(?=[/`'\"?&\s])",
                path,
                text,
            )
        return text

    # Helper-function detection: many modules wrap fetch() in a per-module
    # `api(path)` helper that builds the full URL from a module-level alias.
    # Without this pass, every call through such a helper looks like a fetch
    # to a meta-URL (`/api/dream{param}`, `/api/surface-public/{token}{param}`)
    # that matches no real endpoint. We detect the helper, extract its URL
    # prefix template, and emit one synthetic fetch entry per call site.
    # Match a helper function header (signature only) — capture the function
    # name and the first param name. Don't try to match the closing brace;
    # we'll scan a fixed-size window after the header for the URL pattern.
    _HELPER_HEADER_RE = re.compile(
        r"""(?:async\s+)?function\s+(\w+)\s*\(\s*(\w+)[^)]*\)\s*\{""",
    )

    def _detect_helpers(text: str, aliases: dict) -> list[tuple[str, str, str]]:
        """Find (helper_name, url_prefix, default_method) triples.

        A helper qualifies if, within the first ~800 chars after the function
        header, we see either:
        - `fetch(\`<prefix>${path_param}…\`)`, or
        - `return \`<prefix>${path_param}…\`` (URL-builder helpers).
        `<prefix>` must resolve (via aliases) to a literal starting with /api
        or /v1.
        """
        out: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        # Bound the body window at the next top-level declaration to avoid
        # bleeding into the next function (same fix as block-URL-builder).
        _NEXT_DECL_FOR_HELPER = re.compile(
            r"""\n(?:export\s+)?(?:async\s+)?function\s+\w|\n(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>""",
        )
        for m in _HELPER_HEADER_RE.finditer(text):
            name, path_param = m.group(1), m.group(2)
            if name in seen:
                continue
            if name in {"fetchJson", "fetchOk", "fetch"}:
                # Thin fetch wrappers that take a URL, not a path — handled
                # already by the regular fetch regex.
                continue
            tail = text[m.end():]
            stop = _NEXT_DECL_FOR_HELPER.search(tail)
            body = tail[: stop.start()] if stop else tail[:1200]
            url_patterns = (
                rf"fetch\(\s*`([^`]*)\$\{{{re.escape(path_param)}\}}",
                rf"return\s+`([^`]*)\$\{{{re.escape(path_param)}\}}",
            )
            for upat in url_patterns:
                um = re.search(upat, body)
                if not um:
                    continue
                prefix = um.group(1)
                for alias_name, alias_val in aliases.items():
                    prefix = prefix.replace("${" + alias_name + "}", alias_val)
                prefix = re.sub(r"\$\{[^}]+\}", "{param}", prefix)
                if prefix.startswith(("/api", "/v1")):
                    out.append((name, prefix, "GET"))
                    seen.add(name)
                    break
        return out

    def _emit_helper_calls(text: str, helpers: list, rel: str, sink: list) -> None:
        for name, prefix, default_method in helpers:
            call_pat = re.compile(
                rf"""\b{re.escape(name)}\(\s*[`'"](/[^`'"]+)[`'"]""",
            )
            for m in call_pat.finditer(text):
                path = m.group(1)
                path = re.sub(r"\$\{[^}]+\}", "{param}", path)
                path = path.split("?", 1)[0]
                # Concatenate prefix + path, then collapse any `{param}{param}`
                # that result from prefix ending in `{param}` and path starting
                # with `{param}` (rare but possible) into a single segment.
                url = prefix.rstrip("/") + "/" + path.lstrip("/")
                url = re.sub(r"(\{param\}){2,}", r"\1", url)
                line_no = text[:m.start()].count("\n") + 1
                ctx = text[m.start():m.start() + 300]
                mm = re.search(r"method\s*:\s*['\"](\w+)['\"]", ctx)
                method = mm.group(1).upper() if mm else default_method
                sink.append({
                    "url": url,
                    "method": method,
                    "file": rel,
                    "line": line_no,
                })

    for jsfile in sorted(js_dir.rglob("*.js")):
        if "lib" in jsfile.relative_to(js_dir).parts:
            continue  # skip vendored third-party libraries
        raw = jsfile.read_text(encoding="utf-8", errors="replace")
        text = _resolve_url_aliases(raw)
        rel = str(jsfile.relative_to(ROOT)).replace("\\", "/")

        # Detect and resolve helper-function call sites first.
        aliases = dict(_URL_ALIAS_RE.findall(raw))
        helpers = _detect_helpers(text, aliases)
        _emit_helper_calls(text, helpers, rel, calls)

        def _sanitize_builder_url(raw: str) -> str:
            """Turn a captured URL-template fragment into a canonical path.

            Nested template literals are pre-truncated (\`${qs ? \`?...\` : ''}\`)
            so we may have an unterminated `${qs ` at the tail. Drop those, then
            normalize all `${...}` placeholders to `{param}`, then strip any
            trailing `{param}` not preceded by `/` (conditional query-string
            tack-ons), and strip query strings.
            """
            # Drop unterminated `${...` at the tail
            u = re.sub(r"\$\{[^}]*$", "", raw)
            u = re.sub(r"\$\{[^}]+\}", "{param}", u)
            u = re.sub(r"(?<=[^/]){param}$", "", u)
            u = u.split("?", 1)[0].rstrip()
            return u

        # URL-builder pattern: `export const downloadUrl = (id) => \`/api/...\``
        # These return URL strings consumed by <img src>, <a href>, <video src>
        # etc. — invisible to the fetch-call scanner but a real consumption
        # signal. Emit one call entry per builder so the dead-code matcher
        # sees the path is wired.
        _ARROW_URL_BUILDER_RE = re.compile(
            r"""(?:export\s+)?const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\n?\s*`(/(?:api|v1)/[^`]+)`""",
        )
        for m in _ARROW_URL_BUILDER_RE.finditer(text):
            url = _sanitize_builder_url(m.group(2))
            if not url.startswith(("/api", "/v1")):
                continue
            line_no = text[:m.start()].count("\n") + 1
            calls.append({
                "url": url,
                "method": "GET",
                "file": rel,
                "line": line_no,
            })

        # Block-body URL builder: `const NAME = (...) => { ... return \`/api/...\`; }`
        # Scan a window from the header until we hit either the next top-level
        # declaration (const/function/export/class on its own line) or a fixed
        # cap — bounding by braces fails on nested template literals like
        # `\`${qs ? \`?${qs}\` : ''}\`` and bleeds into the next function.
        _BLOCK_URL_BUILDER_HEADER_RE = re.compile(
            r"""(?:export\s+)?const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{""",
        )
        _NEXT_DECL_RE = re.compile(
            r"""\n(?:export\s+)?(?:async\s+)?function\s+\w|\n(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>""",
        )
        for m in _BLOCK_URL_BUILDER_HEADER_RE.finditer(text):
            tail = text[m.end():]
            stop = _NEXT_DECL_RE.search(tail)
            window = tail[: stop.start()] if stop else tail[:1200]
            ret_match = re.search(r"return\s+`(/(?:api|v1)/[^`]+)`", window)
            if not ret_match:
                continue
            url = _sanitize_builder_url(ret_match.group(1))
            if not url.startswith(("/api", "/v1")):
                continue
            line_no = text[:m.start()].count("\n") + 1
            calls.append({
                "url": url,
                "method": "GET",
                "file": rel,
                "line": line_no,
            })

        # fetch('url' or fetch(`url`) — extract URL and method
        # Skip URLs immediately followed by + (string concatenation — handled separately)
        for m in re.finditer(
            r"""fetch\(\s*[`'"](\/[^`'"$]+)[`'"](?!\s*\+)""",
            text,
        ):
            url = m.group(1)
            line_no = text[:m.start()].count("\n") + 1

            # Look for method in nearby context (within ~200 chars)
            context = text[m.start():m.start() + 300]
            method_match = re.search(r"method\s*:\s*['\"](\w+)['\"]", context)
            method = method_match.group(1).upper() if method_match else "GET"

            calls.append({
                "url": url,
                "method": method,
                "file": rel,
                "line": line_no,
            })

        # Template literal fetch with variable interpolation: fetch(`/api/thing/${id}`)
        for m in re.finditer(
            r"""fetch\(\s*`(\/[^`]+)`""",
            text,
        ):
            url = m.group(1)
            # Nested template literals (e.g. `${qs ? \`?${qs}\` : ''}`) confuse
            # the outer regex into stopping at the first inner backtick. The
            # captured tail looks like `...${qs ?` with no closing `}`. Strip
            # such fragments before normalizing.
            url = re.sub(r"\$\{[^}]*$", "", url).rstrip()
            # Normalize interpolations to {param} placeholders
            url = re.sub(r"\$\{[^}]+\}", "{param}", url)
            # `{param}` immediately tacked on (no `/` separator) is a
            # conditional query-string append, not a path segment — drop it
            # so URLs like `/api/powers${queryFn()}` match the `/api/powers`
            # route definition.
            url = re.sub(r"(?<=[^/]){param}$", "", url)
            line_no = text[:m.start()].count("\n") + 1

            context = text[m.start():m.start() + 300]
            method_match = re.search(r"method\s*:\s*['\"](\w+)['\"]", context)
            method = method_match.group(1).upper() if method_match else "GET"

            # Skip duplicates from the first regex
            if not any(c["file"] == rel and c["line"] == line_no for c in calls):
                calls.append({
                    "url": url,
                    "method": method,
                    "file": rel,
                    "line": line_no,
                })

        # String concatenation: fetch('/api/path/' + variable, ...)
        for m in re.finditer(
            r"""fetch\(\s*['"](/[^'"]+)['"]\s*\+""",
            text,
        ):
            url = m.group(1).rstrip("/") + "/{param}"
            line_no = text[:m.start()].count("\n") + 1
            context = text[m.start():m.start() + 300]
            method_match = re.search(r"method\s*:\s*['\"](\w+)['\"]", context)
            method = method_match.group(1).upper() if method_match else "GET"
            if not any(c["file"] == rel and c["line"] == line_no for c in calls):
                calls.append({
                    "url": url,
                    "method": method,
                    "file": rel,
                    "line": line_no,
                })

        # Broad helper-wrapper pattern: any non-method function call whose
        # FIRST argument is a string literal starting with `/api/` or `/v1/`.
        # Catches wrappers like `_confirmAndPost('/api/companion/rebuild', body)`
        # and the generic `apiCall(url, opts)` style.
        #
        # Filters (each fixes a false-positive class seen in practice):
        #   - `(?<![.])\b\w+\(` skips method calls like `url.startsWith('/api/...')`
        #     and `path.includes('/api/...')` which inspect URLs but don't issue
        #     HTTP requests.
        #   - URL must contain at least one `/` after the `/api/` or `/v1/`
        #     prefix — bare prefixes like `someFn('/api/surfaces')` (where the
        #     real endpoints are `/api/surfaces/...`) cause ghost flags.
        #   - URLs with `${...}` glued to a path segment without a `/` separator
        #     (e.g. `/api/powers${q}`) are conditional query strings, not real
        #     paths — strip them.
        for m in re.finditer(
            r"""(?<![.\w])\w+\(\s*[`'"](/(?:api|v1)/[^`'"]+)[`'"]\s*[,)]""",
            text,
        ):
            url = m.group(1)
            url = re.sub(r"\$\{[^}]+\}", "{param}", url)
            url = url.split("?", 1)[0].rstrip("/")
            # Strip any `{param}` not preceded by `/` (conditional QS tack-on)
            url = re.sub(r"(?<=[^/]){param}$", "", url)
            # Require at least one `/` after the prefix so bare `/api/x` doesn't
            # produce a ghost when real routes are `/api/x/y`.
            prefix_len = len("/api/") if url.startswith("/api/") else len("/v1/")
            tail = url[prefix_len:]
            if "/" not in tail and not tail.endswith("}"):
                # `/api/single-segment` — only emit if there's *also* a backend
                # endpoint exactly at this path. Skip otherwise.
                continue
            line_no = text[:m.start()].count("\n") + 1
            context = text[m.start():m.start() + 300]
            mm = re.search(r"method\s*:\s*['\"](\w+)['\"]", context)
            method = mm.group(1).upper() if mm else "GET"
            if not any(c["file"] == rel and c["line"] == line_no and c["url"] == url for c in calls):
                calls.append({
                    "url": url,
                    "method": method,
                    "file": rel,
                    "line": line_no,
                })

        # WebSocket connections
        for m in re.finditer(r"""new\s+WebSocket\(\s*[`'"](.*?)[`'"]""", text):
            url = m.group(1)
            url = re.sub(r"\$\{[^}]+\}", "{param}", url)
            line_no = text[:m.start()].count("\n") + 1
            calls.append({
                "url": url,
                "method": "WS",
                "file": rel,
                "line": line_no,
            })

    calls.sort(key=lambda c: (c["url"], c["method"]))

    # Deduplicate (same url+method from same file, keep first)
    seen: set[str] = set()
    unique: list[dict] = []
    for c in calls:
        key = f"{c['method']} {c['url']} {c['file']}"
        if key not in seen:
            seen.add(key)
            unique.append(c)

    _write_json(output, {"calls": unique, "count": len(unique)}, "frontend API calls")
    return True

# ---------------------------------------------------------------------------
# 3. Settings map — full camelCase↔snake_case mapping with coverage
# ---------------------------------------------------------------------------

def _extract_dict_keys(text: str, varname: str) -> set[str]:
    """Extract top-level string keys from a Python dict literal."""
    keys: set[str] = set()
    pattern = re.compile(rf"{varname}\s*[:\=].*?\{{", re.DOTALL)
    m = pattern.search(text)
    if not m:
        return keys
    start = m.end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    block = text[start:i - 1]
    for km in re.finditer(r'^\s*"(\w+)"\s*:', block, re.MULTILINE):
        keys.add(km.group(1))
    return keys


def gen_settings_map() -> bool:
    """Build a mapping of every setting across all 4 layers."""
    output = REFS_DIR / "settings_map.json"
    sources = [
        ROOT / "augmentum" / "config.py",
        ROOT / "augmentum" / "proxy" / "config_routes.py",
        ROOT / "augmentum" / "proxy" / "server.py",
        ROOT / "ui" / "scripts" / "settings.js",
    ]

    if not _needs_refresh(output, sources) and "--force" not in sys.argv:
        return False

    # Parse config.py fields
    config_text = sources[0].read_text(encoding="utf-8", errors="replace")
    config_fields: set[str] = set()
    in_class = False
    for line in config_text.splitlines():
        if "class Settings" in line:
            in_class = True
            continue
        if in_class:
            if line and not line[0].isspace() and not line.startswith("#"):
                break
            m = re.match(r"\s+(\w+)\s*:\s*\w+", line)
            if m:
                config_fields.add(m.group(1))

    # Parse config_routes.py
    routes_text = sources[1].read_text(encoding="utf-8", errors="replace")
    tool_settings = _extract_dict_keys(routes_text, "_TOOL_SETTINGS")
    string_settings = _extract_dict_keys(routes_text, "_STRING_SETTINGS")

    # Parse server.py restore map
    server_text = sources[2].read_text(encoding="utf-8", errors="replace")
    restore_map = _extract_dict_keys(server_text, "_SETTINGS_RESTORE_MAP")

    # Parse settings.js — extract the camelCase↔snake_case mapping from sync functions
    js_text = sources[3].read_text(encoding="utf-8", errors="replace")

    # Extract sync body: { snake_key: settings.camelKey, ... }
    js_to_backend: dict[str, str] = {}  # snake → camelCase
    sync_match = re.search(r"function\s+syncToolSettingsToBackend", js_text)
    if sync_match:
        start = js_text.find("{", sync_match.end())
        if start >= 0:
            depth, i = 1, start + 1
            while i < len(js_text) and depth > 0:
                if js_text[i] == "{":
                    depth += 1
                elif js_text[i] == "}":
                    depth -= 1
                i += 1
            block = js_text[start:i]
            # Permit common coercion prefixes between `:` and the
            # `settings.<camel>` reference: `!!`, `!`, `parseFloat(`,
            # `parseInt(`, `Number(`, `String(`, `Boolean(`. Without these
            # the regex false-negatives ~30 entries and inflates the
            # "missing settings.js" wiring count.
            value_re = re.compile(
                r"(\w+)\s*:\s*"
                r"(?:!!|!|parseFloat\(|parseInt\(|Number\(|String\(|Boolean\()?"
                r"(?:settings\.)?(\w+)"
            )
            for km in value_re.finditer(block):
                snake = km.group(1)
                camel = km.group(2)
                if "_" in snake and snake not in ("Content", "method", "headers"):
                    js_to_backend[snake] = camel

    # Build combined map
    all_backend = tool_settings | string_settings
    all_keys = sorted(config_fields | all_backend | restore_map | set(js_to_backend.keys()))

    settings: list[dict] = []
    for key in all_keys:
        # Only include keys that appear in at least one API layer
        in_config = key in config_fields
        in_routes = key in all_backend
        # Restore covers explicit _SETTINGS_RESTORE_MAP entries AND every
        # setting in _TOOL_SETTINGS / _STRING_SETTINGS (auto-derived by
        # _auto_derive_restore_parsers() at server startup).
        in_restore = key in restore_map or key in all_backend
        in_js = key in js_to_backend
        camel = js_to_backend.get(key, "")

        if in_routes or in_restore or in_js:
            layers = sum([in_config, in_routes, in_restore, in_js])
            settings.append({
                "backend_key": key,
                "frontend_key": camel,
                "config_py": in_config,
                "config_routes": in_routes,
                "restore_map": in_restore,
                "settings_js": in_js,
                "layers": layers,
                "category": "tool" if key in tool_settings else ("string" if key in string_settings else "other"),
            })

    _write_json(output, {
        "settings": settings,
        "totals": {
            "config_fields": len(config_fields),
            "tool_settings": len(tool_settings),
            "string_settings": len(string_settings),
            "restore_map": len(restore_map),
            "js_synced": len(js_to_backend),
            "fully_wired": sum(1 for s in settings if s["layers"] >= 3),
        },
    }, "settings map")
    return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def refresh_all(quiet: bool = False) -> int:
    """Refresh all reference files. Returns count of files refreshed."""
    REFS_DIR.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print("  Checking reference freshness...")

    count = 0
    count += gen_route_map()
    count += gen_frontend_api_calls()
    count += gen_settings_map()

    if count == 0 and not quiet:
        print("  All references up to date.")

    return count


if __name__ == "__main__":
    n = refresh_all()
    if n:
        print(f"\n  {n} reference file(s) refreshed.")
    else:
        print("\n  All references already current.")
