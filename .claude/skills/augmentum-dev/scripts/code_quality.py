#!/usr/bin/env python3
"""Augmentum code quality checker — catches non-security issues that cause real bugs.

Checks:
  1. CSS/JS class mismatches — classes referenced in JS but not defined in CSS (and vice versa)
  2. Silent JS catch blocks — empty catch {} that hide errors
  3. WebSocket message contract — server sends types that client doesn't handle
  4. API error response consistency — mixed {error} vs {detail} patterns
  5. Console.log in production — debug logging left in shipped code
  6. TODO/FIXME/HACK tracker — tech debt markers

Exit code 0 = clean, 1 = findings.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import _common  # noqa: F401 — import side-effect: UTF-8-safe stdout/stderr

def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: Cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)

ROOT = _find_root()

_COLOR = os.environ.get("TERM") or os.name != "nt"
def _red(s: str) -> str:    return f"\033[91m{s}\033[0m" if _COLOR else s
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m" if _COLOR else s
def _green(s: str) -> str:  return f"\033[92m{s}\033[0m" if _COLOR else s
def _cyan(s: str) -> str:   return f"\033[96m{s}\033[0m" if _COLOR else s
def _bold(s: str) -> str:   return f"\033[1m{s}\033[0m" if _COLOR else s
def _dim(s: str) -> str:    return f"\033[2m{s}\033[0m" if _COLOR else s

# ---------------------------------------------------------------------------
# 1. CSS/JS class audit
# ---------------------------------------------------------------------------

def check_css_js_classes() -> tuple[list[dict], list[dict]]:
    """Find CSS classes referenced in JS but not defined, and vice versa."""
    css_dir = ROOT / "ui" / "styles"
    js_dir = ROOT / "ui"  # widened from /ui/scripts so cast-*/cast-*.js + standalone harness JS are scanned too
    html_file = ROOT / "ui" / "index.html"

    # Collect all CSS class definitions. The leading `\.` anchors the
    # match to actual class selectors (`.foo`), not to pseudo-classes
    # (`:hover`, `:active`) which start with `:`. Common state class
    # names (active / hidden / disabled) ARE legitimate selectors and
    # used to be unconditionally skipped here, which made every
    # `classList.add('active')` look like a missing-CSS finding —
    # causing several hundred bogus warnings. Trust the regex.
    css_classes: dict[str, str] = {}  # class → file
    # Scan every CSS file under ui/ — not just ui/styles/. Standalone cast-*
    # surfaces (cast-control/cast-control.css, cast-video/cast-video.css,
    # etc.) define hundreds of classes used by their own JS but the legacy
    # css_dir.glob() only looked at the shared style bundle.
    #
    # Skip vendored 3rd-party libraries (highlight.js, milkdown, prism,
    # emulator-js) — their CSS classes are managed by the library and
    # have no relationship to project class-coverage.
    # Skip mockups — design scratchpads with self-contained styles.
    _CSS_SKIP_DIRS = ("ui/lib/", "ui/mockups/")
    # Class-selector terminators: { , : [ are the classic ones, but a
    # class can also be followed by another `.` in a compound selector
    # (`.a.b`), by a space for descendant (`a b`), by `>` for child,
    # or by `+`/`~` for sibling. Without these, classes that ONLY appear
    # in compound selectors are missed and flagged as undefined even
    # though they have rules.
    _CSS_RE = re.compile(r"\.([a-zA-Z_][\w-]*)(?=[\s{,.:>+~\[])")
    # Strip `/* ... */` comments — they often mention `.classname` in prose
    # ("set the .foo-bar class on body") which the selector regex would
    # otherwise treat as a real CSS class and flag as dead.
    _CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
    for cssfile in sorted((ROOT / "ui").rglob("*.css")):
        rel = str(cssfile.relative_to(ROOT)).replace("\\", "/")
        if any(rel.startswith(d) for d in _CSS_SKIP_DIRS):
            continue
        text = cssfile.read_text(encoding="utf-8", errors="replace")
        text = _CSS_COMMENT_RE.sub("", text)
        for m in _CSS_RE.finditer(text):
            cls = m.group(1)
            if cls not in css_classes:
                css_classes[cls] = rel
    # Also harvest inline <style> blocks from every HTML file (cast-* and
    # standalone harnesses commonly inline their CSS). The leading <style
    # ...> matches optional attributes; .*? is non-greedy to handle
    # multiple <style> blocks per file. re.DOTALL lets . span newlines.
    _STYLE_BLOCK_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)
    for htmlfile in sorted((ROOT / "ui").rglob("*.html")):
        rel = str(htmlfile.relative_to(ROOT)).replace("\\", "/")
        if any(rel.startswith(d) for d in _CSS_SKIP_DIRS):
            continue
        try:
            html_text = htmlfile.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for style_block in _STYLE_BLOCK_RE.finditer(html_text):
            block_text = _CSS_COMMENT_RE.sub("", style_block.group(1))
            for m in _CSS_RE.finditer(block_text):
                cls = m.group(1)
                if cls not in css_classes:
                    css_classes[cls] = rel

    # JS-injected stylesheets: self-contained widgets create a <style>
    # element and assign a CSS template literal to .textContent
    # (companion-candidates.js, epub-reader-controls.js). Those classes
    # ARE defined — harvest them or every class in the widget shows up
    # as a missing-CSS finding. Only files that actually create a style
    # element are scanned, and only braced templates, so ordinary
    # `el.textContent = \`...\`` message strings don't pollute the index.
    _VENDORED_JS_DIRS_CSS = ("ui/lib/", "ui/mockups/node_modules/")
    # A backtick template is a stylesheet if it contains a `.sel { prop:`
    # rule. This distinguishes real CSS-in-JS from HTML template literals
    # (which have `class="x"` but no dot-prefixed `.x{` rule bodies).
    _CSS_RULE_HINT = re.compile(r"[.#][\w-]+[^{}`]*\{[^{}`]*:", re.DOTALL)
    for jsfile in sorted((ROOT / "ui").rglob("*.js")):
        rel = str(jsfile.relative_to(ROOT)).replace("\\", "/")
        if any(rel.startswith(d) for d in _VENDORED_JS_DIRS_CSS):
            continue
        text = jsfile.read_text(encoding="utf-8", errors="replace")
        if "createElement('style')" not in text and 'createElement("style")' not in text:
            continue
        # Harvest CSS from EVERY backtick template in a style-injecting file
        # that looks like a stylesheet. The old code only matched CSS
        # assigned inline as `.textContent = \`...\``; it missed the equally
        # common `const CSS_TEXT = \`...\`; styleEl.textContent = CSS_TEXT`
        # shape (RHS is a named identifier, not a literal) — world-panel.js,
        # etc. — so every class in those widgets flagged as missing-CSS.
        # The `_CSS_RULE_HINT` guard keeps ordinary HTML/message templates
        # in the same file from polluting the class index.
        for bm in re.finditer(r"`([^`]*)`", text, re.DOTALL):
            block_text = bm.group(1)
            if not _CSS_RULE_HINT.search(block_text):
                continue
            block_text = _CSS_COMMENT_RE.sub("", block_text)
            for cm in _CSS_RE.finditer(block_text):
                cls = cm.group(1)
                if cls not in css_classes:
                    css_classes[cls] = rel

    # Collect JS class references — multiple discovery channels because
    # frontend code uses classes in many shapes: classList API, querySelector
    # selectors, className assignments, and (most commonly) class="X Y"
    # written inline in template literals or innerHTML.
    #
    # Skip directories of vendored 3rd-party JS: ui/lib/ holds drop-in
    # libraries (highlight.js, milkdown, prism, emulator-js) whose CSS
    # classes are managed by the library itself and not relevant to the
    # project's class-coverage signal. ui/mockups/node_modules/ is npm
    # tooling pulled in for design mockups, not shipped UI code.
    _VENDORED_JS_DIRS = ("ui/lib/", "ui/mockups/node_modules/")
    js_classes: dict[str, list[tuple[str, int]]] = {}  # class → [(file, line), ...]
    # Every class-shaped quoted string literal in JS. A CSS class named here
    # is applied dynamically (e.g. a lookup map `command_exec: 'cca-line--cmd'`
    # → applied via className) and must NOT be flagged dead even though no
    # `class="…"`/classList channel captures it. Keyed literal, precise.
    js_string_literals: set[str] = set()
    # Per-class "hook coverage": for each `class="…"` occurrence of a class,
    # was that element ALSO styled — i.e. does the same class attribute carry
    # a CSS-defined sibling class, or the tag an inline `style=`? A class whose
    # EVERY occurrence is covered is a behavior/querySelector hook (styling
    # comes from the sibling/inline), not a missing stylesheet. This kills the
    # `class="field-input vlex-term"` / `class="dm-x" style="…"` false
    # positives without hiding a genuinely unstyled element (which appears
    # bare, uncovered, and stays flagged).
    hook_cover: dict[str, list[bool]] = {}
    for jsfile in sorted(js_dir.rglob("*.js")):
        rel_check = str(jsfile.relative_to(ROOT)).replace("\\", "/")
        if any(rel_check.startswith(d) for d in _VENDORED_JS_DIRS):
            continue
        text = jsfile.read_text(encoding="utf-8", errors="replace")
        rel = str(jsfile.relative_to(ROOT)).replace("\\", "/")

        # Harvest class-shaped quoted string literals (Fix #3, dead check).
        for sm in re.finditer(r"""['"]([a-zA-Z_][\w-]{2,})['"]""", text):
            js_string_literals.add(sm.group(1))

        # classList.add('class'), classList.remove('class'), etc.
        for m in re.finditer(r"classList\.\w+\(\s*['\"]([a-zA-Z_][\w-]*)['\"]", text):
            cls = m.group(1)
            line = text[:m.start()].count("\n") + 1
            js_classes.setdefault(cls, []).append((rel, line))

        # querySelector('.class') or querySelectorAll('.class')
        for m in re.finditer(r"querySelector(?:All)?\(\s*['\"][^'\"]*\.([a-zA-Z_][\w-]*)", text):
            cls = m.group(1)
            line = text[:m.start()].count("\n") + 1
            js_classes.setdefault(cls, []).append((rel, line))

        # className = 'class' / className += 'class' / class="A B C" inside
        # a template literal or innerHTML payload. The latter is by far the
        # most common shape in this codebase and was being missed.
        #
        # Flatten `${...}` interpolations in the scan copy FIRST. The
        # capture regex stops at any quote character, so an interpolation
        # carrying quotes — `class="pill${active ? ' active' : ''}"` —
        # used to truncate the capture mid-expression: the real class
        # got dropped (its CSS rule then flagged as dead) and a stray JS
        # identifier from inside the ternary got recorded as a "missing
        # class". Flattening keeps the QUOTED STRING contents (' active',
        # '<span class="x">') — those are the markup/class fragments the
        # ternary chooses between — and drops the JS expression tokens.
        # Newlines are preserved so reported line numbers stay exact.
        # Only two shapes of quoted string inside an interpolation are
        # class material: fragments appended to the attribute (written
        # with a leading space by convention — `' active'`) and nested
        # markup (contains its own `class=`). Anything else is a JS
        # operand (`tab === 'history'`) and must NOT be inlined or it
        # shows up as a bogus missing-class finding.
        def _flatten_interp(m: "re.Match[str]") -> str:
            inner = m.group(0)[2:-1]
            parts = [a or b for a, b in
                     re.findall(r"'([^']*)'|\"([^\"]*)\"", inner)]
            kept = [p for p in parts
                    if p and (p[0].isspace() or "class=" in p)]
            return " " + " ".join(kept) + " " + "\n" * inner.count("\n")
        scan_text = re.sub(
            r"""\$\{(?:[^{}'"`]|'[^']*'|"[^"]*")*\}""",
            _flatten_interp,
            text,
        )
        # `className\s*[:+=]` — the `:` arm catches object-literal props
        # (`{className: 'action-overflow-row--danger', ...}`) passed to
        # render helpers, which are real class references.
        for m in re.finditer(
            r"""(?:className\s*[:+=]+|class\s*=)\s*['"`]([^'"`]+)['"`]""",
            scan_text,
        ):
            line_no = scan_text[:m.start()].count("\n") + 1
            # Skip if this match is on a comment line (// or /* prefix). The
            # scanner used to parse `// Fix missing quotes: class=foo -> ...`
            # as a real `class=foo` reference.
            line_text = scan_text.splitlines()[line_no - 1] if line_no <= scan_text.count("\n") + 1 else ""
            stripped_line = line_text.lstrip()
            if stripped_line.startswith("//") or stripped_line.startswith("*"):
                continue
            # Erase whole `${...}` interpolations before splitting — their
            # contents (`'on'`, `'active'`, `false`, `key`, etc.) are JS
            # expressions, not class names. Without this the scanner reports
            # `false` as a "missing CSS class" because the JS tokens
            # `false`, `key`, `null` happen to match the class-name regex.
            class_string = re.sub(r"\$\{[^}]*\}", "", m.group(1))
            group_tokens = [
                c for c in class_string.split()
                if re.match(r"^[a-zA-Z_][\w-]*$", c) and "${" not in c and "}" not in c
            ]
            # Is the element this class-attr sits on already styled? True if
            # the same tag carries an inline `style=`, or another token in the
            # same class="…" is a CSS-defined class. Bound the tag scan to the
            # enclosing <…> so a neighbouring element's style= can't leak in.
            lt = scan_text.rfind("<", 0, m.start())
            gt = scan_text.find(">", m.end())
            if lt != -1 and gt != -1 and (gt - lt) < 600:
                tag_seg = scan_text[lt:gt]
            else:
                tag_seg = scan_text[max(0, m.start() - 80):m.end() + 120]
            has_inline = "style=" in tag_seg
            for cls in group_tokens:
                js_classes.setdefault(cls, []).append((rel, line_no))
                covered = has_inline or any(
                    t != cls and t in css_classes for t in group_tokens
                )
                hook_cover.setdefault(cls, []).append(covered)

    # Scan every HTML file under ui/ (was: only index.html). The new
    # surfaces (artifact-session-edit, comic-detail-preview,
    # avatar-pose-harness, etc.) declare a lot of classes inline.
    #
    # Mockup / prototype HTML files are skipped — they're scratch design
    # artefacts and their classes intentionally don't live in shipped
    # CSS. We detect them in two ways:
    #   1. Any file under ui/mockups/ (the canonical mockup directory)
    #   2. Filename hints "mock" / "-v1" / "-v2" anywhere
    #   3. "preview" in the filename — except for the real
    #      comic-detail-preview surface
    _MOCKUP_HINTS = ("mock", "-v1", "-v2", "preview")
    _MOCKUP_DIRS = ("ui/mockups/", "ui/cast-comic/")  # cast-comic has
                                                       # its own .htmls
                                                       # built dynamically;
                                                       # CSS isn't keyed
                                                       # by class.
    for htmlfile in sorted((ROOT / "ui").rglob("*.html")):
        name = htmlfile.name.lower()
        rel = str(htmlfile.relative_to(ROOT)).replace("\\", "/")
        if any(rel.startswith(d) for d in _MOCKUP_DIRS):
            continue
        if any(h in name for h in _MOCKUP_HINTS) and name != "comic-detail-preview.html":
            # comic-detail-preview is a real surface despite the name
            continue
        try:
            html_text = htmlfile.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for m in re.finditer(r'class="([^"]+)"', html_text):
            for cls in m.group(1).split():
                js_classes.setdefault(cls, []).append((rel, 0))

    # Detect dynamic class-name construction: `'prefix-' + variant` /
    # `\`prefix-${variant}\`` — scan all JS for these patterns and record
    # the prefix as "valid root" so dead-CSS `prefix-X` rules and any
    # bare-prefix references stop flagging as drift.
    dynamic_class_prefixes: set[str] = set()
    for jsfile in sorted(js_dir.rglob("*.js")):
        rel_check = str(jsfile.relative_to(ROOT)).replace("\\", "/")
        if any(rel_check.startswith(d) for d in _VENDORED_JS_DIRS):
            continue
        text = jsfile.read_text(encoding="utf-8", errors="replace")
        # `'foo-' +` and `\`foo-${...}\``
        for m in re.finditer(r"['\"`]([a-z][\w-]*-)['\"`]\s*\+", text):
            dynamic_class_prefixes.add(m.group(1))
        # Backtick template with `prefix-${var}` anywhere inside — not just
        # at the start. Catches `class="bf-status-pill bf-status-${s}"` etc.
        # where the prefix appears mid-string.
        for m in re.finditer(r"([a-z][\w-]*-)\$\{", text):
            dynamic_class_prefixes.add(m.group(1))
        # `classList.add('foo-' + variant)` and similar
        for m in re.finditer(
            r"classList\.\w+\(\s*['\"`]([a-z][\w-]*-)['\"`]\s*\+", text,
        ):
            dynamic_class_prefixes.add(m.group(1))

    # Classes in JS but not in CSS (potential missing styles)
    missing_css: list[dict] = []
    # Skip dynamic/generated classes and common framework patterns
    skip_prefixes = ("js-", "is-", "has-", "data-", "ProseMirror", "milkdown",
                     "crepe", "mermaid", "hljs", "language-", "code-",
                     "lightbox", "img-lib",
                     # Editor framework classes (CodeMirror 5 + 6 API):
                     "CodeMirror", "EasyMDE", "cm-",
                     # KaTeX / Prism syntax highlighters:
                     "katex", "prism-",
                     # Bootstrap-style utility prefixes seen in 3rd party
                     # widgets and not styled in our CSS:
                     "btn-", "icon-")

    # Per-class allowlists. Two kinds:
    #   * missing_css_marker_classes — classes set in JS but intentionally
    #     have no styling (DOM markers, querySelector targets).
    #   * dead_css_acknowledged — classes the dead-CSS pass would flag
    #     even though they're live. Common reasons: dynamic prefix
    #     concatenation (`'pipeline-' + variant`) and compound-selector
    #     variants (`.parent.foo`) where `.foo` is set dynamically.
    suppress_path = Path(__file__).parent / "quality_suppressions.json"
    suppressed: set[str] = set()
    dead_acknowledged: set[str] = set()
    if suppress_path.exists():
        try:
            import json as _json
            data = _json.loads(suppress_path.read_text(encoding="utf-8"))
            suppressed = set(data.get("missing_css_marker_classes", []))
            dead_acknowledged = set(data.get("dead_css_acknowledged", []))
        except Exception:
            suppressed = set()
            dead_acknowledged = set()

    for cls, locations in sorted(js_classes.items()):
        if cls in css_classes:
            continue
        if any(cls.startswith(p) for p in skip_prefixes):
            continue
        if cls in suppressed:
            continue
        if len(cls) < 3:  # Skip very short classes (likely abbreviations)
            continue
        # Bare-prefix-with-trailing-dash markers (`'size-'`, `'pipeline-'`,
        # `'console-'`) are detected from dynamic concatenation sites
        # elsewhere in the codebase. They aren't real class names.
        if cls.endswith("-"):
            continue
        # Behavior/querySelector hook: every `class="…"` occurrence sits on an
        # already-styled element (inline `style=` or a CSS-defined sibling
        # class). The class carries no styling of its own by design — it's a
        # JS handle. Not a missing stylesheet. A genuinely unstyled element
        # appears bare (uncovered) in at least one spot and still flags.
        cov = hook_cover.get(cls)
        if cov and all(cov):
            continue
        # Only report first occurrence
        file, line = locations[0]
        missing_css.append({
            "class": cls,
            "file": file,
            "line": line,
            "refs": len(locations),
        })

    # Large CSS classes never referenced (potential dead styles)
    # Only report classes with distinctive names (not generic like "active")
    dead_css: list[dict] = []
    for cls, css_file in sorted(css_classes.items()):
        if cls in js_classes:
            continue
        if len(cls) < 8:  # Skip short generic classes
            continue
        if any(cls.startswith(p) for p in skip_prefixes):
            continue
        # Skip CSS rules whose names match a dynamic-prefix root from JS.
        # `.size-compact` is "live" if some JS does `'size-' + variant`
        # somewhere, even if `compact` isn't a literal anywhere.
        if any(cls.startswith(p) for p in dynamic_class_prefixes):
            continue
        # Referenced as a quoted string literal in JS — applied dynamically
        # via a lookup map / variable (e.g. `command_exec: 'cca-line--cmd'`
        # in coder.js, applied as className). Live, not dead. No JS channel
        # (classList/querySelector/class=) captures this shape, so without
        # this the whole map's worth of CSS flags as dead.
        if cls in js_string_literals:
            continue
        # Allowlist for known false-positive dead-CSS findings (compound
        # selectors with dynamically-set variants, vendor framework
        # overrides, etc.). Maintain in quality_suppressions.json.
        if cls in dead_acknowledged:
            continue
        dead_css.append({
            "class": cls,
            "file": css_file,
        })

    return missing_css, dead_css

# ---------------------------------------------------------------------------
# 2. Silent JS catch blocks
# ---------------------------------------------------------------------------

def check_silent_catches() -> list[dict]:
    """Find empty catch blocks that hide errors.

    Three classes of opt-in:

      1. Comment-only catches (`catch { /* ignore */ }`) — the comment IS
         the documentation. JS equivalent of `# noqa`.
      2. Single-line inline-fallback idiom (`try { x = parse(s); } catch {}`
         on one line) — the catch is structurally tied to a single
         expression in the try, so the failure mode is "leave x at its
         default value". Idiomatic.
      3. Catch blocks at most a single short line — the dev took some
         intentional action even if there's no comment.

    Only multi-line truly-empty `catch { }` blocks remain flagged —
    those have neither documentation nor structural tightness.
    """
    findings: list[dict] = []
    js_dir = ROOT / "ui" / "scripts"  # narrow: silent-catch signal is about app code, not vendored harnesses

    for jsfile in sorted(js_dir.rglob("*.js")):
        text = jsfile.read_text(encoding="utf-8", errors="replace")
        rel = str(jsfile.relative_to(ROOT)).replace("\\", "/")
        lines = text.splitlines()

        # Match catch blocks: catch { }, catch (e) { }, catch { /* comment */ }
        for m in re.finditer(
            r"catch\s*(?:\([^)]*\))?\s*\{([^}]*)\}",
            text,
        ):
            body = m.group(1).strip()
            line_no = text[:m.start()].count("\n") + 1
            end_line_no = text[:m.end()].count("\n") + 1

            # Comment-only catches are intentional (class 1).
            if body and re.match(r"^\s*(?:\/\/.*|\/\*.*\*\/)\s*$", body):
                continue

            # Inline-fallback idiom (class 2): the catch immediately
            # follows a `try { …; }` on the same line. The try-body is
            # a single expression that has a known fallback path.
            if not body and line_no == end_line_no:
                source_line = lines[line_no - 1] if line_no <= len(lines) else ""
                if "try" in source_line and "}" in source_line:
                    continue

            # Empty or whitespace-only — check for two more legitimate
            # patterns before flagging:
            #   * Optional-chain defensive call: the try body uses `?.(...)`
            #     or `?.[...]` — "call this if it exists, otherwise no-op"
            #     is the intent, and the silent catch handles the rare
            #     case where the method threw instead of being absent
            #     (e.g., recordRating?.() on a stale reference).
            #   * Try body is a single statement with a clear fallback
            #     pattern (return / continue / pass-equivalent default).
            if not body:
                # Find the start of the matching try block by walking back
                # from this catch's `}`. Cheap: look at the previous ~8
                # lines for the `try {` opener at any indent.
                try_body_window: list[str] = []
                for k in range(line_no - 2, max(line_no - 10, -1), -1):
                    try_body_window.insert(0, lines[k])
                    if re.search(r"\btry\s*\{", lines[k]):
                        break
                window_blob = "\n".join(try_body_window)
                # Optional-chain method call inside the try → silent fallback OK.
                if re.search(r"\?\.\s*[\w(]", window_blob):
                    continue
                # Try body contains a `.catch(...)` chain — the catch
                # outside is the sync-throw safety net for fetch().catch()
                # / .then().catch() / promise chains that already handle
                # async failure. The empty outer catch only fires if the
                # constructor itself throws synchronously, which is
                # universally non-actionable.
                if re.search(r"\.catch\s*\(", window_blob):
                    continue
                # `await fetch(...).then(...)` / single-statement awaits
                # where the await is inside the try and the value is
                # discarded — fire-and-forget pattern, silent fallback
                # tolerable when no other recovery is possible.
                if re.search(r"\bawait\s+\w+\(.*\)\.then\(", window_blob):
                    continue
                # Try has a `let x = ...; ... x = (something)` shape where
                # the silent catch lets `x` stay at its initialised value.
                # Detect a `let `/`const ` declaration in the window AND
                # a non-null/default-value check on the SAME identifier
                # within the next ~6 lines below the catch. This catches
                # the "vrmUrl stays null → caller falls through" pattern.
                init_match = re.search(
                    r"\b(?:let|const|var)\s+(\w+)\s*=\s*(?:null|undefined|''|\"\"|\[\]|\{\}|0|false)",
                    window_blob,
                )
                if init_match:
                    var_name = init_match.group(1)
                    tail_window = "\n".join(lines[line_no : line_no + 6])
                    if re.search(rf"\bif\s*\(\s*!?{re.escape(var_name)}\b", tail_window):
                        continue
                findings.append({
                    "file": rel, "line": line_no,
                    "type": "empty",
                    "description": "Empty catch block — errors silently swallowed",
                })

    return findings

# ---------------------------------------------------------------------------
# 3. WebSocket message contract
# ---------------------------------------------------------------------------

def check_websocket_contract() -> list[dict]:
    """Check that all WS message types sent by server have handlers in client.

    Scope: only types the server EMITS to the client (via _send_json,
    websocket.send_json, etc.) — not types the server READS from inbound
    client messages. Client-side, any reference to the type name anywhere
    in the frontend JS bundle counts as a handler (`case 'x'`,
    `meta.chain_step`, `data.type === 'x'`, even just the literal in a
    helper). False positives on the client side cost more than misses,
    because the contract is one-directional.
    """
    findings: list[dict] = []

    voice_routes = ROOT / "augmentum" / "proxy" / "voice_routes.py"
    if not voice_routes.exists():
        return findings

    server_text = voice_routes.read_text(encoding="utf-8", errors="replace")

    # Match only EMIT sites — `_send_json(ws, {"type": "X"})`,
    # `websocket.send_json({"type": "X"})`, `ws.send_json({"type": "X"})`,
    # `await websocket.send_text(json.dumps({"type": "X"}))`. Inbound
    # reads (`if data.get("type") == "X"`, `payload["type"] == "X"`)
    # are NOT server emissions.
    emit_pattern = re.compile(
        r"""(?:_send_json|send_json|send_text)\s*\(
            [^)]*?
            ["']type["']\s*:\s*["'](\w+)["']
        """,
        re.DOTALL | re.VERBOSE,
    )
    server_types: set[str] = set()
    for m in emit_pattern.finditer(server_text):
        server_types.add(m.group(1))

    # Client side: any reference to the type name in any JS bundle. We
    # already had false positives where chat/index.js handles
    # `meta.chain_step` and the prior regex didn't see that as a handler.
    js_files = list((ROOT / "ui" / "scripts").rglob("*.js"))
    client_blob_parts: list[str] = []
    for jsfile in js_files:
        try:
            client_blob_parts.append(jsfile.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
    client_blob = "\n".join(client_blob_parts)

    for stype in sorted(server_types):
        # Informational stream-control types don't need explicit handlers.
        if stype in {"error", "info", "debug", "ping", "pong", "ack",
                     "ready", "done", "complete"}:
            continue
        # Match the type name as a quoted literal OR as an attribute name
        # (e.g. `meta.chain_step`, `data.chain_step`).
        pattern = re.compile(
            rf"""(?:["']{re.escape(stype)}["']|\.{re.escape(stype)}\b)"""
        )
        if pattern.search(client_blob):
            continue
        findings.append({
            "type": stype,
            "direction": "server-to-client",
            "description": f"Server sends '{stype}' but no frontend code references it",
        })

    return findings

# ---------------------------------------------------------------------------
# 4. API error response consistency
# ---------------------------------------------------------------------------

def check_error_consistency() -> list[dict]:
    """Check for mixed error response patterns (JSONResponse vs HTTPException).

    Only counts JSONResponse with "error" as the SOLE top-level key —
    multi-key status payloads like ``{"state": "error", "error": ...}``
    or ``{"results": [], "error": ...}`` are non-error responses that
    happen to mention an error field, and shouldn't be conflated with
    pure error returns.
    """
    findings: list[dict] = []
    proxy_dir = ROOT / "augmentum" / "proxy"
    # JSONResponse({"error": ...})  with no other top-level keys.
    # Greedy-but-bounded body match; rejects anything with a comma at
    # depth zero before the closing brace (which would imply >1 key).
    sole_error = re.compile(
        r"""JSONResponse\(\s*\{\s*["']error["']\s*:\s*[^{}]*?\}""",
        re.DOTALL,
    )

    for rf in sorted(proxy_dir.glob("*_routes.py")):
        text = rf.read_text(encoding="utf-8", errors="replace")
        rel = str(rf.relative_to(ROOT)).replace("\\", "/")

        json_count = sum(
            1 for m in sole_error.finditer(text)
            if "," not in m.group(0).split('"error"', 1)[1].rstrip("}").strip().rsplit(":", 1)[1]
        )
        http_count = len(re.findall(r"HTTPException\(", text))

        if json_count and http_count:
            findings.append({
                "file": rel,
                "json_errors": json_count,
                "http_exceptions": http_count,
                "description": f"Mixed error patterns: {json_count}x JSONResponse(error) + {http_count}x HTTPException",
            })

    return findings

# ---------------------------------------------------------------------------
# 5. Console.log in production
# ---------------------------------------------------------------------------

def check_console_logs() -> list[dict]:
    """Find console.log calls in production JS (not test files).

    Filters out:
      - `console.warn` / `console.error` — intentional error surfacing.
      - `console.debug` — browsers default-hide debug-level output, so
        leaving it in production has no user-visible effect; the dev who
        wrote it explicitly chose the muted channel.
      - Calls that appear inside string literals (snippet templates,
        autocomplete entries, doc strings) — those are data, not code.
    """
    findings: list[dict] = []
    js_dir = ROOT / "ui" / "scripts"  # narrow: console.log signal is about app code, not vendored harnesses

    for jsfile in sorted(js_dir.rglob("*.js")):
        text = jsfile.read_text(encoding="utf-8", errors="replace")
        rel = str(jsfile.relative_to(ROOT)).replace("\\", "/")
        lines = text.splitlines()

        for m in re.finditer(r"console\.(log|info)\(", text):
            line_no = text[:m.start()].count("\n") + 1
            # Skip occurrences inside a quoted/backtick string literal on
            # the same line (snippet text, error messages built as
            # strings, etc.). Heuristic: count the number of unescaped
            # quote characters before the match position on this line —
            # an odd count means we're inside a literal.
            line_text = lines[line_no - 1] if line_no <= len(lines) else ""
            col = m.start() - (text.rfind("\n", 0, m.start()) + 1)
            prefix = line_text[:col]
            # Strip escaped quotes so they don't perturb the parity check.
            cleaned = prefix.replace("\\'", "").replace('\\"', "").replace("\\`", "")
            if cleaned.count("'") % 2 or cleaned.count('"') % 2 or cleaned.count("`") % 2:
                continue
            # Skip intentional tagged-diagnostic logs: the first arg is a
            # string literal starting with `[xxx]` (subsystem tag). This
            # is the codebase's convention for status/diagnostic logs that
            # an operator can grep for in DevTools. Example:
            #   console.info('[becca] persona mode active');
            #   console.log('[coder.preview] set src (no workspace)', ...);
            #   console.log('[ejs-net] fetch start', url);
            # The check is conservative — only the explicit `[tag]` form
            # qualifies; unprefixed console.log stays flagged.
            # Look at the next ~3 lines too so multi-line calls like
            #   console.info(
            #     '[agent-bridge] ' + ...,
            #   )
            # are caught alongside single-line ones. Two tag forms supported:
            #   '[name] ...'      bracketed (most common in this codebase)
            #   'name: ...'       colon-prefixed (xr-companion-binding etc.)
            # Both are explicit subsystem prefixes signalling intentional
            # operator-visible diagnostic, not stray debug noise.
            window_text = "\n".join(lines[line_no - 1 : line_no + 3])
            if re.search(
                r"console\.(?:log|info)\(\s*['\"`](?:\[|[a-z][\w-]*:\s)",
                window_text,
            ):
                continue
            findings.append({
                "file": rel,
                "line": line_no,
                "level": m.group(1),
            })

    return findings

# ---------------------------------------------------------------------------
# 6. TODO/FIXME/HACK tracker
# ---------------------------------------------------------------------------

def check_tech_debt() -> list[dict]:
    """Find TODO, FIXME, HACK, XXX comments across the codebase."""
    findings: list[dict] = []
    # Require the comment-prefix character to be at line start OR preceded
    # by whitespace, so it's actually a comment delimiter and not a CSS
    # ID selector inside a JSON string literal (`"#todo-list"` would match
    # the prior version because `#todo-list` looked like `# todo-list`).
    # Inline comments after code (`x = 1  # TODO: …`) still match because
    # they're preceded by whitespace.
    # Require AT LEAST ONE space between the comment marker (`#`/`//`/`/*`)
    # and the TODO keyword. CSS ID selectors inside string literals
    # (`"#todo-list"`) have no space, which is how the prior regex was
    # confusing JSON content for code comments.
    marker_re = re.compile(
        r"(?:^|\s)(?:#|//|/\*)\s+(TODO|FIXME|HACK|XXX|WORKAROUND)\b[:\s]*([^\r\n]*)",
        re.IGNORECASE | re.MULTILINE,
    )

    def _emit(text: str, rel: str) -> None:
        for m in marker_re.finditer(text):
            line = text[:m.start()].count("\n") + 1
            tail = m.group(2).strip()
            # Skip meta-comments that mention TODO as documentation of
            # what a regex does (`# "TODO" from passing the multiply check.`).
            # If the tail quotes another TODO marker, this is meta-text.
            if tail.startswith('"') and ('TODO' in tail or 'FIXME' in tail):
                continue
            findings.append({
                "marker": m.group(1).upper(),
                "text": tail[:80],
                "file": rel,
                "line": line,
            })

    # Python files
    for pyfile in sorted((ROOT / "augmentum").rglob("*.py")):
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        rel = str(pyfile.relative_to(ROOT)).replace("\\", "/")
        _emit(text, rel)

    # JS files
    for jsfile in sorted((ROOT / "ui" / "scripts").rglob("*.js")):
        text = jsfile.read_text(encoding="utf-8", errors="replace")
        rel = str(jsfile.relative_to(ROOT)).replace("\\", "/")
        _emit(text, rel)
    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 7. _model_map membership-as-locality misuse (2026-05-26 regression class)
# ---------------------------------------------------------------------------

def check_model_map_membership() -> list[dict]:
    """Catch ``X in _model_map`` checks that don't also accept the
    disambiguated key.

    Background: multi-provider models (same model name on N providers)
    are stored in ``_model_map`` ONLY under their disambiguated
    ``<model>@<backend>`` keys — never under the clean name. A bare
    ``clean_model in _model_map`` check silently classifies every
    disambiguated pick as "not local", which trips downstream fail-fast
    guards even though the ``@``-suffix branch correctly resolved a
    specific backend. The 2026-05-26 NVIDIA NIM regression
    (``z-ai/glm-5.1@vim`` rejected as "not served") was this bug; fixed
    in provider_registry.py + fabric_routes.py with a two-key check.

    What this checker enforces: every ``... in (self|registry)._model_map``
    membership check must EITHER be paired with an OR-fallback to a
    second key (the canonical fix pattern), OR be part of an iteration
    context (``for X in ... .items()`` / ``.keys()`` / ``.values()``,
    or direct iteration of the map).

    Suppressions live in ``quality_suppressions.json`` under
    ``model_map_membership_acknowledged`` as ``"file.py:lineno"`` entries.
    """
    findings: list[dict] = []
    pat = re.compile(r"\b[\w.]+\s+in\s+(?:self|registry)\._model_map\b(?!\.)")
    for_iter = re.compile(r"\bfor\s+[\w,\s()]+\s+in\s+(?:self|registry)\._model_map\b")

    sup_path = ROOT / ".claude" / "skills" / "augmentum-dev" / "scripts" / "quality_suppressions.json"
    suppressed: set[str] = set()
    if sup_path.is_file():
        try:
            data = json.loads(sup_path.read_text(encoding="utf-8"))
            suppressed = set(data.get("model_map_membership_acknowledged", []))
        except (json.JSONDecodeError, OSError):
            pass

    for pyfile in sorted((ROOT / "augmentum").rglob("*.py")):
        try:
            text = pyfile.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(pyfile.relative_to(ROOT)).replace("\\", "/")
        # Skip the canonical fix sites — they're the reference patterns,
        # not regressions. (They contain the fix BUT also match the
        # naive regex on the first line; the windowed OR check below
        # should clear them, but belt-and-suspenders.)
        lines = text.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""'):
                continue
            if not pat.search(line):
                continue
            # Iteration contexts — for-loop or comprehension over the map.
            if for_iter.search(line):
                continue
            if any(x in line for x in (".items(", ".keys(", ".values(")):
                continue
            # OR-fallback check: the matching line, the line above, AND
            # two lines after must collectively contain ``_model_map``
            # twice AND " or " somewhere — the canonical two-key shape.
            # Looking backward matters because the regex matches BOTH
            # halves of a multi-line ``X in map\n  or Y in map`` and
            # the second match's forward-only window misses the X-clause
            # on the line above.
            window = " ".join(lines[max(0, i - 1):i + 3])
            mm_count = len(re.findall(r"_model_map", window))
            if mm_count >= 2 and " or " in window:
                continue
            key = f"{rel}:{i + 1}"
            if key in suppressed:
                continue
            findings.append({
                "file": rel,
                "line": i + 1,
                "snippet": stripped[:80],
            })
    return findings


def main():
    print(_bold("\n  Augmentum Code Quality Check"))
    print(_bold("  " + "=" * 40) + "\n")

    has_actionable = False

    # 1. CSS/JS
    print(_cyan("  [1/7] Checking CSS/JS class alignment..."))
    missing_css, dead_css = check_css_js_classes()

    # 2. Silent catches
    print(_cyan("  [2/7] Checking silent JS catch blocks..."))
    silent = check_silent_catches()

    # 3. WebSocket contract
    print(_cyan("  [3/7] Checking WebSocket message contract..."))
    ws_issues = check_websocket_contract()

    # 4. Error consistency
    print(_cyan("  [4/7] Checking API error response consistency..."))
    error_mix = check_error_consistency()

    # 5. Console.log
    print(_cyan("  [5/7] Checking for console.log in production..."))
    console_logs = check_console_logs()

    # 6. Tech debt
    print(_cyan("  [6/7] Tracking TODO/FIXME/HACK markers..."))
    tech_debt = check_tech_debt()

    # 7. _model_map membership-as-locality misuse
    print(_cyan("  [7/7] Checking _model_map membership-as-locality usage..."))
    mm_membership = check_model_map_membership()

    print()

    # Report
    if missing_css and "--verbose" in sys.argv:
        print(_yellow(f"  Missing CSS Classes ({len(missing_css)}) — referenced in JS/HTML but not defined in CSS:"))
        for mc in missing_css[:20]:
            ref_count = f" ({mc['refs']}x)" if mc["refs"] > 1 else ""
            loc = f"{mc['file']}:{mc['line']}"
            print(f"    {_yellow('~')} .{mc['class']}{ref_count}  {_dim(loc)}")
        if len(missing_css) > 20:
            remaining = len(missing_css) - 20
            print(f"    {_dim(f'... and {remaining} more')}")
        print()

    if silent:
        has_actionable = True
        print(_yellow(f"  Silent Catch Blocks ({len(silent)}) — errors swallowed without logging:"))
        for s in silent[:15]:
            loc = f"{s['file']}:{s['line']}"
            print(f"    {_yellow('~')} {s['description'][:60]}  {_dim(loc)}")
        if len(silent) > 15:
            remaining = len(silent) - 15
            print(f"    {_dim(f'... and {remaining} more')}")
        print()

    if ws_issues:
        has_actionable = True
        print(_red(f"  WebSocket Contract Gaps ({len(ws_issues)}):"))
        for w in ws_issues:
            print(f"    {_red('!')} {w['description']}")
        print()

    if error_mix:
        print(_yellow(f"  Mixed Error Patterns ({len(error_mix)}) — inconsistent error response format:"))
        for e in error_mix:
            print(f"    {_yellow('~')} {e['file']}: {e['description']}")
        print()

    if console_logs:
        print(_dim(f"  Console.log ({len(console_logs)}) — debug logging in production:"))
        for cl in console_logs[:10]:
            loc = f"{cl['file']}:{cl['line']}"
            print(f"    {_dim('-')} console.{cl['level']}  {_dim(loc)}")
        if len(console_logs) > 10:
            remaining = len(console_logs) - 10
            print(f"    {_dim(f'... and {remaining} more')}")
        print()

    if mm_membership:
        has_actionable = True
        print(_red(f"  _model_map membership-as-locality misuse ({len(mm_membership)}):"))
        print(_dim("    Bare ``X in _model_map`` check that doesn't also accept the"))
        print(_dim("    disambiguated ``model@backend`` key — the 2026-05-26 regression"))
        print(_dim("    class. Pair with ``or model_name in _model_map`` or iterate."))
        for mm in mm_membership[:10]:
            loc = f"{mm['file']}:{mm['line']}"
            print(f"    {_red('!')} {mm['snippet']}  {_dim(loc)}")
        if len(mm_membership) > 10:
            remaining = len(mm_membership) - 10
            print(f"    {_dim(f'... and {remaining} more')}")
        print()

    if tech_debt:
        by_marker: dict[str, list] = {}
        for td in tech_debt:
            by_marker.setdefault(td["marker"], []).append(td)

        print(_dim(f"  Tech Debt Markers ({len(tech_debt)}):"))
        for marker in ("FIXME", "TODO", "HACK", "XXX", "WORKAROUND"):
            items = by_marker.get(marker, [])
            if items:
                header = f"{marker} ({len(items)}):"
                print(f"    {_dim(header)}")
                for td in items[:5]:
                    desc = td["text"][:50] if td["text"] else "(no description)"
                    loc = f"{td['file']}:{td['line']}"
                    print(f"      {_dim('-')} {desc}  {_dim(loc)}")
                if len(items) > 5:
                    remaining = len(items) - 5
                    print(f"      {_dim(f'... and {remaining} more')}")
        print()

    # Summary
    print(_bold("  Summary"))
    print(f"    Missing CSS classes:  {len(missing_css):3d}  {_dim('(use --verbose to list)')}")
    print(f"    Dead CSS classes:     {len(dead_css):3d}")
    print(f"    Silent JS catches:    {len(silent):3d}")
    print(f"    WS contract gaps:     {len(ws_issues):3d}")
    print(f"    Mixed error patterns: {len(error_mix):3d}")
    print(f"    Console.log:          {len(console_logs):3d}")
    print(f"    Tech debt markers:    {len(tech_debt):3d}")
    print(f"    _model_map misuse:    {len(mm_membership):3d}")
    print()

    return 1 if has_actionable else 0


if __name__ == "__main__":
    sys.exit(main())
