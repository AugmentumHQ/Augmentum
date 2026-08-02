"""Core engine for constraint-driven code synthesis.

Implements behaviors one at a time, running QuickJS tests after each
to verify correctness. Retries on failure, detects regressions, and
tracks progress via an async callback.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from augmentum.tools.constraint_compiler import (
    TestCompilationResult,
    compile_tests,
    generate_css_foundation,
    generate_skeleton,
)
from augmentum.tools.constraint_schema import AppSpec, Constraint, sort_constraints
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SynthesisResult:
    """Final output of the synthesis loop."""

    skeleton: str = ""
    css: str = ""
    js: str = ""
    assembled: str = ""
    constraint_results: list[dict] = field(default_factory=list)
    total_llm_calls: int = 0
    total_tokens: int = 0


class SynthesisLoop:
    """Iteratively implements behavioral constraints via LLM + QuickJS testing."""

    def __init__(
        self,
        call_llm: Callable[..., Coroutine[Any, Any, str]],
        max_attempts: int = 3,
        request_model: str = "",
    ) -> None:
        self._call_llm = call_llm
        self._max_attempts = max_attempts
        self._request_model = request_model

    async def run(
        self, spec: AppSpec, progress_cb: Callable | None = None
    ) -> SynthesisResult:
        """Run the full synthesis loop over all constraints in dependency order."""
        skeleton = generate_skeleton(spec)
        css = generate_css_foundation(spec)
        tests = compile_tests(spec)
        ordered = sort_constraints(spec.constraints)

        accumulated_js = ""
        constraint_results: list[dict] = []
        total_llm_calls = 0
        passed_ids: set[str] = set()

        for c in ordered:
            # Check dependencies
            unmet = [d for d in c.depends_on if d not in passed_ids]
            if unmet:
                c.status = "skipped"
                detail = "unmet dependencies: " + ", ".join(unmet)
                constraint_results.append(
                    {"id": c.id, "behavior": c.behavior, "status": "skipped",
                     "detail": detail}
                )
                await self._emit_progress(progress_cb, c, "skipped", detail)
                continue

            await self._emit_progress(progress_cb, c, "running", "")

            last_error: str | None = None
            succeeded = False

            for attempt in range(self._max_attempts):
                code = await self._generate_behavior(
                    c, spec, accumulated_js, skeleton, tests, last_error, attempt
                )
                total_llm_calls += 1
                code = self._clean_code(code)

                candidate_js = (
                    accumulated_js
                    + "\n// --- " + c.id + ": " + c.behavior + " ---\n"
                    + code
                )

                test_results = self._run_tests(
                    skeleton, css, candidate_js, tests, ordered
                )

                current_passed = c.id in test_results.get("passed", set())
                regressions = test_results.get("regressions", [])

                if current_passed and not regressions:
                    accumulated_js = candidate_js
                    c.status = "passed"
                    passed_ids.add(c.id)
                    succeeded = True
                    break

                if regressions:
                    last_error = "Regression: broke " + ", ".join(regressions)
                else:
                    errors = test_results.get("errors", [])
                    relevant = [e for e in errors if c.id in e]
                    last_error = (
                        relevant[0] if relevant
                        else (errors[0] if errors else "test did not pass")
                    )

            if not succeeded:
                c.status = "failed"

            constraint_results.append(
                {"id": c.id, "behavior": c.behavior, "status": c.status,
                 "detail": last_error or ""}
            )
            await self._emit_progress(progress_cb, c, c.status, last_error or "")

        assembled = self._assemble(skeleton, css, accumulated_js)

        return SynthesisResult(
            skeleton=skeleton,
            css=css,
            js=accumulated_js,
            assembled=assembled,
            constraint_results=constraint_results,
            total_llm_calls=total_llm_calls,
        )

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    async def _generate_behavior(
        self,
        constraint: Constraint,
        spec: AppSpec,
        accumulated_js: str,
        skeleton: str,
        tests: TestCompilationResult,
        last_error: str | None,
        attempt: int,
    ) -> str:
        """Build a prompt and call the LLM to generate JS for one constraint."""
        system = (
            "You are implementing ONE behavior for a web application.\n\n"
            "OUTPUT RULES (follow these EXACTLY):\n"
            "- Output ONLY the JavaScript code, nothing else\n"
            "- Do NOT wrap in markdown code fences (no ```)\n"
            "- Do NOT include filenames, explanations, or comments about what you're doing\n"
            "- Do NOT include __PASS_COMPLETE__ or any pipeline markers\n"
            "- Do NOT redeclare variables, functions, or classes from previous behaviors\n\n"
            "CODE RULES:\n"
            "- Wrap your code in an IIFE: (function() { 'use strict'; ... })();\n"
            "- Use document.getElementById() and document.querySelector() for DOM access\n"
            "- Use addEventListener() for event handling, never inline onclick\n"
            "- Share state between behaviors via window.appState (create it if it doesn't exist)\n"
            "- The HTML skeleton below shows all available element IDs — reference them exactly\n"
            "- The test below shows EXACTLY how your code will be verified — read it carefully\n"
        )

        # Compact accumulated JS if very long
        if accumulated_js:
            lines = accumulated_js.splitlines()
            if len(lines) > 60:
                shown = (
                    "\n".join(lines[:20])
                    + "\n// ... (middle omitted) ...\n"
                    + "\n".join(lines[-20:])
                )
            else:
                shown = accumulated_js
        else:
            shown = "(no previous code)"

        state_parts = []
        for k, v in spec.state_schema.items():
            state_parts.append(k + ": " + str(v))
        state_desc = ", ".join(state_parts) if state_parts else "none"

        test_code = tests.tests.get(constraint.id, "// no test")

        user_parts = [
            "## HTML Skeleton\n```html\n" + skeleton + "\n```",
            "## State Schema\n" + state_desc,
            "## Previous Behaviors (accumulated JS)\n```js\n" + shown + "\n```",
            "## Constraint to implement\nID: " + constraint.id
            + "\nBehavior: " + constraint.behavior
            + "\nDescription: " + constraint.description
            + "\nType: " + constraint.type,
        ]

        if constraint.trigger:
            user_parts.append("Trigger: " + str(constraint.trigger))
        if constraint.expected:
            user_parts.append("Expected: " + str(constraint.expected))

        user_parts.append("## Test that must pass\n```js\n" + test_code + "\n```")

        if attempt > 0 and last_error:
            user_parts.append(
                "## PREVIOUS ATTEMPT FAILED — FIX REQUIRED\n"
                "Attempt " + str(attempt + 1) + " of " + str(self._max_attempts) + ".\n\n"
                "Error:\n" + last_error + "\n\n"
                "IMPORTANT: Your previous code had a bug. Read the error above carefully.\n"
                "- If it says SyntaxError: you have a syntax bug (missing brace, semicolon, paren)\n"
                "- If it says 'no click listener': you forgot addEventListener\n"
                "- If it says 'not found': you used a wrong element ID\n"
                "- If it says 'Regression': your code broke a previously working behavior\n\n"
                "Output ONLY the corrected JavaScript IIFE. No explanations."
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

        return await self._call_llm(messages, model=self._request_model)

    # ------------------------------------------------------------------
    # Code cleaning
    # ------------------------------------------------------------------

    def _clean_code(self, code: str) -> str:
        """Strip LLM artifacts from generated code.

        Handles: markdown fences, thinking tokens, pipeline markers,
        SEARCH/REPLACE markers, prose preamble/postscript, HTML blocks.
        """
        # Strip thinking tokens from reasoning models (<think>...</think>)
        code = re.sub(r"<think>[\s\S]*?</think>", "", code)
        # Strip any XML-like tags that aren't HTML (e.g., <output>, <response>)
        code = re.sub(r"</?(?:output|response|answer|code|result)>", "", code)

        # Strip markdown code fences (```javascript, ```js, ```, etc.)
        code = re.sub(r"^```(?:javascript|js|html|css)?\s*\n?", "", code, flags=re.MULTILINE)
        code = re.sub(r"\n?```\s*$", "", code, flags=re.MULTILINE)

        # Strip pipeline markers
        code = code.replace("__PASS_COMPLETE__", "")
        code = code.replace("__NEEDS_ANOTHER_PASS__", "")

        # Strip SEARCH/REPLACE markers (models sometimes emit these)
        code = re.sub(r"^[<>=]{7}.*$", "", code, flags=re.MULTILINE)
        code = re.sub(
            r"<<<<<<< SEARCH.*?=======.*?>>>>>>> REPLACE", "", code, flags=re.DOTALL
        )

        # Strip prose preamble (lines before the first IIFE or function)
        # Models often write "Here's the code:" before the actual JS
        lines = code.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith("(function") or
                stripped.startswith("'use strict'") or
                stripped.startswith('"use strict"') or
                stripped.startswith("document.") or
                stripped.startswith("window.") or
                stripped.startswith("var ") or
                stripped.startswith("const ") or
                stripped.startswith("let ") or
                stripped.startswith("function ")):
                start_idx = i
                break
        if start_idx > 0:
            code = "\n".join(lines[start_idx:])

        # Strip trailing prose after the closing })(); of the IIFE
        iife_end = code.rfind("})();")
        if iife_end > 0:
            code = code[:iife_end + 5]

        return code.strip()

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------

    def _run_tests(
        self,
        skeleton: str,
        css: str,
        js: str,
        tests: TestCompilationResult,
        all_constraints: list[Constraint],
    ) -> dict:
        """Assemble HTML, run QuickJS, return {passed, failed, regressions, errors}."""
        assembled = self._assemble(skeleton, css, js)

        # Parse DOM from assembled HTML
        from augmentum.tools.artifact_application import ApplicationBuilderTool

        dom = ApplicationBuilderTool._parse_html_dom(assembled)

        html_ids = dom.get("ids", set())
        html_classes = dom.get("classes", set())
        html_tags = dom.get("tags", set())

        try:
            import quickjs  # type: ignore[import-untyped]
        except ImportError:
            log.warning("quickjs not available -- skipping test execution")
            return {"passed": set(), "failed": set(), "regressions": [], "errors": []}

        id_list = sorted(html_ids)
        class_list = sorted(html_classes)
        tag_list = sorted(html_tags)

        dom_mock_js = self._build_dom_mock(id_list, class_list, tag_list)

        errors: list[str] = []
        try:
            ctx = quickjs.Context()
            ctx.set_memory_limit(32 * 1024 * 1024)
            ctx.set_time_limit(5)

            ctx.eval(dom_mock_js)
            ctx.eval("var _verifyErrors = [];")

            # Execute the app JS.  SyntaxError is thrown by the engine
            # BEFORE execution (try/catch inside JS can't catch it), so
            # we catch it at the Python level, record the error, and
            # still run the constraint tests — they'll naturally fail
            # because the code didn't execute, giving precise feedback.
            code_executed = False
            try:
                # Split accumulated JS into blocks (separated by behavior
                # markers) and execute each independently.  This way a
                # SyntaxError in the LATEST behavior doesn't prevent
                # previously-passing behaviors from running.
                blocks = re.split(r'(?=\n// --- c\d+:)', js) if js.strip() else []
                if not blocks:
                    blocks = [js] if js.strip() else []
                for i, block in enumerate(blocks):
                    if not block.strip():
                        continue
                    wrapped = (
                        "try {\n" + block + "\n} catch(e) {"
                        " _verifyErrors.push('JS_ERROR: ' + e.constructor.name"
                        " + ': ' + e.message); }"
                    )
                    try:
                        ctx.eval(wrapped)
                        code_executed = True
                    except Exception as block_exc:
                        # SyntaxError in this block — extract the specific
                        # error and continue running other blocks
                        err_line = str(block_exc).split("\n")[0]
                        # Try to identify which constraint this block belongs to
                        cid_match = re.search(r'// --- (c\d+):', block[:80])
                        cid = cid_match.group(1) if cid_match else f"block-{i}"
                        errors.append(f"CONSTRAINT {cid}: SyntaxError in generated code: {err_line}")
                        log.warning("synthesis.block_syntax_error",
                                    constraint=cid, error=err_line,
                                    code_head=block[:120].replace("\n", " "))
            except Exception as js_exc:
                errors.append(f"JS_EXEC_ERROR: {js_exc}")
                log.warning("synthesis.js_execution_error", error=str(js_exc))

            # Fire DOMContentLoaded
            try:
                ctx.eval(
                    "if (typeof _dcl_listeners !== 'undefined') {"
                    "  _dcl_listeners.forEach(function(fn) {"
                    "    try { fn(); } catch(e) {"
                    "      _verifyErrors.push('DCL_ERROR: ' + e.constructor.name + ': ' + e.message);"
                    "    }"
                    "  });"
                    "}"
                )
            except Exception as dcl_exc:
                errors.append(f"DCL_ERROR: {dcl_exc}")

            # Fire deferred setTimeout callbacks
            try:
                ctx.eval(
                    "if (typeof _deferredTimers !== 'undefined') {"
                    "  _deferredTimers.forEach(function(fn) {"
                    "    try { fn(); } catch(e) {"
                    "      _verifyErrors.push('TIMER_ERROR: ' + e.constructor.name + ': ' + e.message);"
                    "    }"
                    "  });"
                    "}"
                )
            except Exception as timer_exc:
                errors.append(f"TIMER_ERROR: {timer_exc}")

            # Run the constraint tests — always runs even if code had errors
            try:
                ctx.eval(tests.combined_script())
            except Exception as test_exc:
                errors.append(f"TEST_EXEC_ERROR: {test_exc}")
                log.warning("synthesis.test_execution_error", error=str(test_exc))

            # Collect errors from the JS-side array
            try:
                raw = ctx.eval("JSON.stringify(_verifyErrors)")
                if raw:
                    errors.extend(json.loads(raw))
            except Exception as exc:
                log.debug("synthesis_loop_verify_errors_collect_failed", error=str(exc))

        except Exception as exc:
            log.warning("quickjs_mock_setup_failed", error=str(exc))
            errors = ["QUICKJS_ERROR: " + str(exc)]

        # Categorise results
        passed: set[str] = set()
        failed: set[str] = set()

        # Check for global errors (JS_ERROR, QUICKJS_ERROR, etc.) that aren't
        # constraint-specific -- these mean the code itself is broken
        global_errors = [
            e for e in errors
            if not e.startswith("CONSTRAINT ")
        ]

        constraint_ids = {c.id for c in all_constraints}
        if global_errors:
            # Code-level failure: all constraints are failed
            failed = set(constraint_ids)
        else:
            for cid in constraint_ids:
                if any("CONSTRAINT " + cid in e for e in errors):
                    failed.add(cid)
                else:
                    passed.add(cid)

        # Detect regressions: constraints previously marked "passed" that now fail
        regressions = [
            c.id for c in all_constraints
            if c.status == "passed" and c.id in failed
        ]

        return {
            "passed": passed,
            "failed": failed,
            "regressions": regressions,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # DOM mock builder
    # ------------------------------------------------------------------

    def _build_dom_mock(
        self,
        id_list: list[str],
        class_list: list[str],
        tag_list: list[str],
    ) -> str:
        """Generate a QuickJS-compatible DOM mock from parsed HTML metadata."""
        ids_js = ", ".join('"' + i + '"' for i in id_list)
        classes_js = ", ".join('"' + c + '"' for c in class_list)
        tags_js = ", ".join('"' + t + '"' for t in tag_list)

        return (
            "var _knownIds = new Set([" + ids_js + "]);\n"
            "var _knownClasses = new Set([" + classes_js + "]);\n"
            "var _knownTags = new Set([" + tags_js + "]);\n"
            "var _elementRegistry = {};\n"
            "var _eventLog = [];\n"
            "\n"
            "var _mockElement = function(tag, id) {\n"
            "    if (id && _elementRegistry[id]) return _elementRegistry[id];\n"
            "    var el = {\n"
            "        tagName: tag || 'DIV',\n"
            "        id: id || '',\n"
            "        textContent: '',\n"
            "        innerHTML: '',\n"
            "        innerText: '',\n"
            "        value: '',\n"
            "        checked: false,\n"
            "        disabled: false,\n"
            "        hidden: false,\n"
            "        _listeners: {},\n"
            "        style: {},\n"
            "        classList: {\n"
            "            _classes: new Set(),\n"
            "            add: function() { for(var i=0;i<arguments.length;i++) this._classes.add(arguments[i]); },\n"
            "            remove: function() { for(var i=0;i<arguments.length;i++) this._classes.delete(arguments[i]); },\n"
            "            toggle: function(c) { if(this._classes.has(c)) this._classes.delete(c); else this._classes.add(c); return this._classes.has(c); },\n"
            "            contains: function(c) { return this._classes.has(c); }\n"
            "        },\n"
            "        dataset: {},\n"
            "        children: [],\n"
            "        parentNode: null,\n"
            "        appendChild: function(c) { this.children.push(c); c.parentNode = this; return c; },\n"
            "        removeChild: function(c) { var i=this.children.indexOf(c); if(i>=0) this.children.splice(i,1); return c; },\n"
            "        insertBefore: function(c) { this.children.push(c); return c; },\n"
            "        addEventListener: function(type, handler) {\n"
            "            if (!this._listeners[type]) this._listeners[type] = [];\n"
            "            this._listeners[type].push(handler);\n"
            "            _eventLog.push({id: this.id, type: type});\n"
            "        },\n"
            "        removeEventListener: function() {},\n"
            "        setAttribute: function(k, v) { this[k] = v; },\n"
            "        getAttribute: function(k) { return this[k] || null; },\n"
            "        querySelector: function() { return null; },\n"
            "        querySelectorAll: function() { return []; },\n"
            "        getBoundingClientRect: function() { return {x:0,y:0,width:100,height:100,top:0,left:0,right:100,bottom:100}; },\n"
            "        click: function() { if(this._listeners['click']) this._listeners['click'].forEach(function(fn){ try{fn({preventDefault:function(){},target:el})}catch(e){} }); },\n"
            "        focus: function(){},\n"
            "        blur: function(){},\n"
            "        remove: function(){},\n"
            "        cloneNode: function() { return _mockElement(); },\n"
            "        closest: function() { return null; },\n"
            "        matches: function() { return false; },\n"
            "        offsetWidth: 100, offsetHeight: 100,\n"
            "        clientWidth: 100, clientHeight: 100,\n"
            "        width: 800, height: 600,\n"
            "        _contextCreated: false,\n"
            "        getContext: function(type) {\n"
            "            this._contextCreated = true;\n"
            "            if (type === '2d') return {\n"
            "                fillStyle: '', strokeStyle: '', lineWidth: 1, font: '',\n"
            "                save: function(){}, restore: function(){},\n"
            "                fillRect: function(){}, strokeRect: function(){}, clearRect: function(){},\n"
            "                beginPath: function(){}, closePath: function(){}, moveTo: function(){},\n"
            "                lineTo: function(){}, arc: function(){}, fill: function(){}, stroke: function(){},\n"
            "                fillText: function(){}, measureText: function() { return {width:0}; },\n"
            "                drawImage: function(){},\n"
            "                createLinearGradient: function() { return { addColorStop: function(){} }; },\n"
            "                translate: function(){}, rotate: function(){}, scale: function(){},\n"
            "            };\n"
            "            return {};\n"
            "        },\n"
            "    };\n"
            "    if (id) _elementRegistry[id] = el;\n"
            "    return el;\n"
            "};\n"
            "\n"
            "function _matchSelector(sel) {\n"
            "    if (!sel || typeof sel !== 'string') return false;\n"
            "    sel = sel.trim();\n"
            "    var parts = sel.split(/\\s+/).filter(function(p) { return p && p !== '>' && p !== '+' && p !== '~'; });\n"
            "    if (parts.length > 1) {\n"
            "        for (var i = 0; i < parts.length; i++) {\n"
            "            if (!_matchSingle(parts[i])) return false;\n"
            "        }\n"
            "        return true;\n"
            "    }\n"
            "    return _matchSingle(sel);\n"
            "}\n"
            "\n"
            "function _matchSingle(sel) {\n"
            "    if (sel.startsWith('#')) {\n"
            "        var id = sel.slice(1).split(/[.:[>+~]/, 1)[0];\n"
            "        return _knownIds.has(id);\n"
            "    }\n"
            "    if (sel.startsWith('.')) {\n"
            "        var cls = sel.slice(1).split(/[.:[>+~#]/, 1)[0];\n"
            "        return _knownClasses.has(cls);\n"
            "    }\n"
            "    var tag = sel.split(/[.:[>+~#]/, 1)[0].toLowerCase();\n"
            "    if (!tag) return false;\n"
            "    return _knownTags.has(tag);\n"
            "}\n"
            "\n"
            "var document = {\n"
            "    getElementById: function(id) {\n"
            "        if (_knownIds.has(id)) return _mockElement('DIV', id);\n"
            "        return null;\n"
            "    },\n"
            "    querySelector: function(sel) {\n"
            "        if (_matchSelector(sel)) return _mockElement();\n"
            "        return null;\n"
            "    },\n"
            "    querySelectorAll: function(sel) {\n"
            "        if (_matchSelector(sel)) return [_mockElement()];\n"
            "        return [];\n"
            "    },\n"
            "    createElement: function(tag) { return _mockElement(tag); },\n"
            "    createDocumentFragment: function() { return _mockElement('FRAGMENT'); },\n"
            "    createTextNode: function() { return { textContent: '' }; },\n"
            "    addEventListener: function() {},\n"
            "    removeEventListener: function() {},\n"
            "    body: _mockElement('BODY'),\n"
            "    head: _mockElement('HEAD'),\n"
            "    documentElement: _mockElement('HTML'),\n"
            "    cookie: '',\n"
            "    title: '',\n"
            "    readyState: 'complete',\n"
            "};\n"
            "\n"
            "var _dcl_listeners = [];\n"
            "var _orig_addEventListener = document.addEventListener;\n"
            "document.addEventListener = function(type, fn) {\n"
            "    if (type === 'DOMContentLoaded') _dcl_listeners.push(fn);\n"
            "};\n"
            "\n"
            "var window = globalThis;\n"
            "window.document = document;\n"
            "window.innerWidth = 1024;\n"
            "window.innerHeight = 768;\n"
            "window._listeners = {};\n"
            "window.addEventListener = function(type, fn) {\n"
            "    if (type === 'DOMContentLoaded') _dcl_listeners.push(fn);\n"
            "    if (!window._listeners[type]) window._listeners[type] = [];\n"
            "    window._listeners[type].push(fn);\n"
            "};\n"
            "window.removeEventListener = function() {};\n"
            "var _deferredTimers = [];\n"
            "window.setTimeout = function(fn, ms) { if (typeof fn === 'function') _deferredTimers.push(fn); return _deferredTimers.length; };\n"
            "window.setInterval = function() { return 1; };\n"
            "window.clearTimeout = function() {};\n"
            "window.clearInterval = function() {};\n"
            "window.requestAnimationFrame = function(fn) { return 1; };\n"
            "window.cancelAnimationFrame = function() {};\n"
            "window.getComputedStyle = function() { return {}; };\n"
            "window.matchMedia = function() { return { matches: false, addEventListener: function(){} }; };\n"
            "window.scrollTo = function() {};\n"
            "window.alert = function() {};\n"
            "window.confirm = function() { return false; };\n"
            "window.prompt = function() { return null; };\n"
            "window.fetch = function() { return new Promise(function(r){ r({ ok: true, json: function(){ return new Promise(function(r2){ r2({}); }); } }); }); };\n"
            "window.navigator = { userAgent: 'quickjs-verify', language: 'en' };\n"
            "window.location = { href: 'about:blank', origin: '', pathname: '/', search: '', hash: '' };\n"
            "window.history = { pushState: function(){}, replaceState: function(){}, back: function(){}, forward: function(){} };\n"
            "window.performance = { now: function() { return 0; } };\n"
            "\n"
            "var console = { log: function(){}, error: function(){}, warn: function(){}, info: function(){}, debug: function(){} };\n"
            "var localStorage = { _d: {}, getItem: function(k){ return this._d[k] || null; }, setItem: function(k,v){ this._d[k]=String(v); }, removeItem: function(k){ delete this._d[k]; }, clear: function(){ this._d={}; } };\n"
            "var sessionStorage = { _d: {}, getItem: function(k){ return this._d[k] || null; }, setItem: function(k,v){ this._d[k]=String(v); }, removeItem: function(k){ delete this._d[k]; }, clear: function(){ this._d={}; } };\n"
            "\n"
            "var MutationObserver = function() { this.observe = function(){}; this.disconnect = function(){}; };\n"
            "var Audio = function() { this.play = function(){}; this.pause = function(){}; this.addEventListener = function(){}; this.src = ''; };\n"
            "var AudioContext = function() { this.createGain = function(){ return {gain:{value:1}, connect:function(){}}; }; this.destination = {}; };\n"
        )

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble(self, skeleton: str, css: str, js: str) -> str:
        """Combine skeleton, CSS, and JS into a single HTML document."""
        css_tag = "<style>\n" + css + "\n</style>" if css else ""
        js_tag = "<script>\n" + js + "\n</script>" if js else ""

        if "</head>" in skeleton:
            assembled = skeleton.replace("</head>", css_tag + "\n</head>")
        else:
            assembled = css_tag + "\n" + skeleton

        if "</body>" in assembled:
            assembled = assembled.replace("</body>", js_tag + "\n</body>")
        else:
            assembled += "\n" + js_tag

        return assembled

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    async def _emit_progress(
        self,
        cb: Callable | None,
        constraint: Constraint,
        status: str,
        detail: str,
    ) -> None:
        """Fire the progress callback if one is registered."""
        if cb is None:
            return
        try:
            await cb({
                "constraint_id": constraint.id,
                "behavior": constraint.behavior,
                "status": status,
                "detail": detail,
            })
        except Exception:
            log.warning("progress callback failed", constraint_id=constraint.id)
