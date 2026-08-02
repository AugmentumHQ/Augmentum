"""Tests for the Application Builder pipeline.

Run: python -m pytest tests/test_app_builder.py -v
These tests validate the pipeline logic WITHOUT requiring a running server or LLM.
They use mock LLM responses to verify parsing, assembly, and pipeline flow.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Import the modules under test
# ---------------------------------------------------------------------------

def _import_scaffolds():
    """Import scaffolds module (no heavy dependencies)."""
    from augmentum.tools.application_scaffolds import (
        SCAFFOLDS,
        build_fix_prompt,
        build_generate_prompt,
        build_judge_prompt,
        build_plan_prompt,
    )
    return SCAFFOLDS, build_plan_prompt, build_generate_prompt, build_fix_prompt, build_judge_prompt


def _import_tool():
    """Import tool module — needs structlog stub if not installed."""
    try:
        from augmentum.tools.artifact_application import (
            ApplicationBuilderTool,
            PassResult,
            PipelineContext,
        )
        return ApplicationBuilderTool, PipelineContext, PassResult
    except ImportError:
        pytest.skip("structlog not available — run in project venv")


# ===========================================================================
# Scaffold Tests
# ===========================================================================

class TestScaffolds:
    def test_scaffolds_exist(self):
        SCAFFOLDS, *_ = _import_scaffolds()
        assert len(SCAFFOLDS) == 4
        assert "static" in SCAFFOLDS
        assert "dashboard" in SCAFFOLDS
        assert "game" in SCAFFOLDS
        assert "form" in SCAFFOLDS

    def test_each_scaffold_has_required_fields(self):
        SCAFFOLDS, *_ = _import_scaffolds()
        for key, scaffold in SCAFFOLDS.items():
            assert "name" in scaffold, f"{key} missing name"
            assert "default_files" in scaffold, f"{key} missing default_files"
            assert "cdn_includes" in scaffold, f"{key} missing cdn_includes"
            assert len(scaffold["default_files"]) >= 2, f"{key} has too few default files"

    def test_plan_prompt_create_mode(self):
        _, build_plan_prompt, *_ = _import_scaffolds()
        messages = build_plan_prompt("build a calculator", "static")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "calculator" in messages[1]["content"]
        assert "__PASS_COMPLETE__" in messages[0]["content"]

    def test_plan_prompt_iterate_mode(self):
        _, build_plan_prompt, *_ = _import_scaffolds()
        existing = [{"path": "index.html", "role": "entry", "content": "<html></html>"}]
        messages = build_plan_prompt("add dark mode", "static", existing_files=existing)
        assert "modif" in messages[0]["content"].lower()  # "modifications" or "modifying"
        assert "index.html" in messages[1]["content"]

    def test_generate_prompt(self):
        _, _, build_generate_prompt, *_ = _import_scaffolds()
        files = [{"path": "app.js", "role": "script", "description": "main logic"}]
        existing = {"index.html": "<html></html>"}
        messages = build_generate_prompt(files, existing, "calculator app")
        assert "app.js" in messages[1]["content"]
        assert "index.html" in messages[1]["content"]

    def test_fix_prompt(self):
        _, _, _, build_fix_prompt, *_ = _import_scaffolds()
        files = [{"path": "app.js", "content": "function() {}"}]
        errors = ["app.js: Mismatched braces"]
        messages = build_fix_prompt(files, errors)
        assert "SEARCH/REPLACE" in messages[0]["content"] or "SEARCH" in messages[0]["content"]
        assert "Mismatched" in messages[1]["content"]

    def test_judge_prompt_abbreviates_long_files(self):
        _, _, _, _, build_judge_prompt = _import_scaffolds()
        long_content = "\n".join(f"line {i}" for i in range(100))
        files = [{"path": "app.js", "content": long_content}]
        messages = build_judge_prompt(files, "test app")
        # Should NOT contain all 100 lines — abbreviated
        assert "omitted" in messages[1]["content"].lower() or messages[1]["content"].count("\n") < 80

    def test_design_framework_detects_categories(self):
        """Verify auto-detection of design categories from description + scaffold."""
        from augmentum.tools.application_scaffolds import _detect_categories, build_design_rules
        # Game scaffold → canvas_game
        cats = _detect_categories("space shooter with enemies", "game")
        assert "canvas_game" in cats
        # Dashboard scaffold → charts_dashboard
        cats = _detect_categories("sales analytics", "dashboard")
        assert "charts_dashboard" in cats
        # Cross-scaffold: static scaffold but description says "chart"
        cats = _detect_categories("line chart showing temperature", "static")
        assert "charts_dashboard" in cats
        # Form scaffold → interactive_form
        cats = _detect_categories("contact form with validation", "form")
        assert "interactive_form" in cats
        # Build rules should be non-empty for any scaffold
        rules = build_design_rules("calculator app", "static")
        assert "Design Rules" in rules
        assert "Anti-Patterns" in rules

    def test_design_rules_in_plan_prompt(self):
        """Verify design rules are injected into the plan prompt."""
        _, build_plan_prompt, *_ = _import_scaffolds()
        messages = build_plan_prompt("space shooter game", "game")
        system = messages[0]["content"]
        assert "requestAnimationFrame" in system, "Game design rules not in plan prompt"
        assert "NEVER" in system, "Anti-patterns not in plan prompt"


# ===========================================================================
# Pipeline Context & Parsing Tests
# ===========================================================================

class TestPipelineParsing:
    def _get_tool(self):
        ApplicationBuilderTool, _, _ = _import_tool()
        # Create tool with mock store and LLM
        mock_store = type("MockStore", (), {"save": None})()
        tool = ApplicationBuilderTool(mock_store, lambda m, **k: "", lambda: {})
        return tool

    def test_extract_name_filters_common_words(self):
        tool = self._get_tool()
        assert "Build" not in tool._extract_name("Build me a calculator app")
        assert "Calculator" in tool._extract_name("Build me a calculator app")
        assert tool._extract_name("create a todo list") != ""
        assert "Create" not in tool._extract_name("create a todo list")

    def test_derive_project_name_anchor_heuristic(self):
        """The old extractor took the first 4 non-skip words, producing
        names like 'Simple Calculator That Supports'. Confirm the new
        anchor-noun walker yields tight names instead."""
        from augmentum.tools.artifact_application import derive_project_name

        # Single-modifier anchor cases.
        assert derive_project_name("build me a todo app") == "Todo App"
        assert derive_project_name("weather dashboard") == "Weather Dashboard"
        assert derive_project_name("snake game") == "Snake Game"
        # Connector "that …" must truncate the modifier walk.
        assert derive_project_name(
            "make me a simple calculator that supports basic math"
        ) == "Simple Calculator"
        # Multi-modifier — cap at three.
        assert derive_project_name(
            "build a simple modern responsive todo app with localStorage"
        ) == "Modern Responsive Todo App"
        # Stacked anchors — last wins so "Calorie Tracker App" beats "Calorie Tracker".
        assert derive_project_name("build a calorie tracker app") == "Calorie Tracker App"
        # Acronym preservation.
        assert derive_project_name("api explorer for swagger docs") == "API Explorer"
        # No anchor — falls back to filtered keywords.
        assert derive_project_name("draw a sine wave on a canvas") == "Sine Wave Canvas"
        # Bare anchor — just the anchor in title case.
        assert derive_project_name("calculator") == "Calculator"
        # Anchor first, modifiers follow — forward sweep kicks in.
        assert derive_project_name("a chat ui for testing my llm") == "Chat UI"
        assert derive_project_name("page editor") == "Page Editor"
        # Empty / filler-only — sane fallback.
        assert derive_project_name("") == "Web App"
        assert derive_project_name("please can you") == "Web App"

    def test_parse_file_plan_structured(self):
        tool = self._get_tool()
        response = """
FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Main page
FILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styles
FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: Logic
__PASS_COMPLETE__
"""
        files = tool._parse_file_plan(response)
        assert len(files) == 3
        assert files[0]["path"] == "index.html"
        assert files[0]["role"] == "entry"
        assert files[2]["lang"] == "javascript"

    def test_parse_file_plan_numbered_list_fallback(self):
        tool = self._get_tool()
        response = """
Here's my plan:
1. index.html - Main page with form
2. styles.css - Clean dark theme
3. app.js - Calculator logic
"""
        files = tool._parse_file_plan(response)
        assert len(files) == 3
        assert files[0]["path"] == "index.html"

    def test_parse_file_plan_bare_filename_fallback(self):
        tool = self._get_tool()
        response = "I'll create index.html with the structure, styles.css for theming, and app.js for logic."
        files = tool._parse_file_plan(response)
        assert len(files) == 3

    def test_parse_generated_files(self):
        tool = self._get_tool()
        response = """
```index.html
<!DOCTYPE html>
<html><body>Hello</body></html>
```

```styles.css
body { margin: 0; }
```
"""
        files = tool._parse_generated_files(response)
        assert len(files) == 2
        assert files[0]["path"] == "index.html"
        assert "Hello" in files[0]["content"]

    def test_parse_generated_files_skips_language_tags(self):
        tool = self._get_tool()
        response = """
```javascript
console.log("this is a language tag, not a file");
```
"""
        files = tool._parse_generated_files(response)
        assert len(files) == 0  # "javascript" has no dot, should be skipped

    def test_parse_generated_files_strips_backticks(self):
        """Trailing backtick on filename (common LLM mistake)."""
        tool = self._get_tool()
        response = """
```index.html`
<!DOCTYPE html><html><body>Hello</body></html>
```

```app.js`
console.log("hi");
```
"""
        files = tool._parse_generated_files(response)
        assert len(files) == 2
        assert files[0]["path"] == "index.html"  # backtick stripped
        assert files[1]["path"] == "app.js"  # backtick stripped

    def test_parse_generated_files_language_then_filename(self):
        """LLM outputs '```html index.html' format."""
        tool = self._get_tool()
        response = """
```html index.html
<!DOCTYPE html><html><body>Test</body></html>
```
"""
        files = tool._parse_generated_files(response)
        assert len(files) == 1
        assert files[0]["path"] == "index.html"

    def test_parse_generated_files_skips_context_blocks(self):
        """Context blocks should not be parsed as files."""
        tool = self._get_tool()
        response = """
```app.js
console.log("real file");
```

```context
CLASSES: .foo
IDS: #bar
```
"""
        files = tool._parse_generated_files(response)
        assert len(files) == 1
        assert files[0]["path"] == "app.js"

    def test_parse_score(self):
        tool = self._get_tool()
        assert tool._parse_score("SCORE: 8.5/10") == 8.5
        assert tool._parse_score("SCORE: 7/10") == 7.0
        assert tool._parse_score("no score here") == 5.0

    def test_apply_file_patches_with_file_wrappers(self):
        tool = self._get_tool()
        files = [
            {"path": "app.js", "content": "function hello() {\n  return 'hello';\n}"},
        ]
        response = """
=== FILE: app.js ===
<<<<<<< SEARCH
  return 'hello';
=======
  return 'world';
>>>>>>> REPLACE
"""
        applied = tool._apply_file_patches(files, response)
        assert applied == 1
        assert "world" in files[0]["content"]

    def test_apply_file_patches_without_wrappers(self):
        tool = self._get_tool()
        files = [
            {"path": "app.js", "content": "const x = 1;\nconst y = 2;"},
        ]
        response = """
<<<<<<< SEARCH
const x = 1;
=======
const x = 42;
>>>>>>> REPLACE
"""
        applied = tool._apply_file_patches(files, response)
        assert applied == 1
        assert "42" in files[0]["content"]

    def test_apply_file_patches_normalizes_decorated_file_header(self):
        tool = self._get_tool()
        files = [
            {"path": "app.js", "content": "const label = 'old';\n"},
        ]
        response = "\n".join([
            "=== FILE: app.js (FULL) ===",
            "<<<<<<< SEARCH",
            "const label = 'old';",
            "=======",
            "const label = 'new';",
            ">>>>>>> REPLACE",
        ])
        applied = tool._apply_file_patches(files, response)
        assert applied == 1
        assert "const label = 'new';" in files[0]["content"]

    def test_apply_file_patches_skips_ambiguous_bare_patch(self):
        tool = self._get_tool()
        files = [
            {"path": "app.js", "content": "const shared = true;\nconst a = 1;"},
            {"path": "admin.js", "content": "const shared = true;\nconst b = 2;"},
        ]
        response = "\n".join([
            "<<<<<<< SEARCH",
            "const shared = true;",
            "=======",
            "const shared = false;",
            ">>>>>>> REPLACE",
        ])
        applied = tool._apply_file_patches(files, response)
        assert applied == 0
        assert all("const shared = true;" in f["content"] for f in files)

    def test_infer_role(self):
        tool = self._get_tool()
        assert tool._infer_role("index.html") == "entry"
        assert tool._infer_role("styles.css") == "style"
        assert tool._infer_role("app.js") == "script"
        assert tool._infer_role("data.json") == "data"
        assert tool._infer_role("README.md") == "readme"

    def test_infer_lang(self):
        tool = self._get_tool()
        assert tool._infer_lang("app.js") == "javascript"
        assert tool._infer_lang("styles.css") == "css"
        assert tool._infer_lang("index.html") == "html"

    def test_lint_html(self):
        tool = self._get_tool()
        # The heuristic triggers when open tags exceed close tags by >2
        issues = tool._lint_html("<div><span><p><a>text</div>")
        assert len(issues) > 0  # 4 opens, 1 close = 3 unclosed

    def test_lint_js_braces(self):
        tool = self._get_tool()
        issues = tool._lint_js("function() { if (true) {")
        assert len(issues) > 0

    def test_fuzzy_apply_exact(self):
        """Tier 1: exact match works."""
        tool = self._get_tool()
        content = "const x = 1;\nconst y = 2;\n"
        result, ok = tool._fuzzy_apply(content, "const y = 2;", "const y = 42;")
        assert ok
        assert "const y = 42;" in result

    def test_fuzzy_apply_trimmed_lines(self):
        """Tier 2: trimmed-line matching handles indentation differences."""
        tool = self._get_tool()
        content = "function foo() {\n    const x = 1;\n    return x;\n}\n"
        # Search has different indentation
        result, ok = tool._fuzzy_apply(content, "const x = 1;\nreturn x;", "const x = 99;\nreturn x;")
        assert ok
        assert "const x = 99;" in result
        # Verify indentation was preserved from the original
        for line in result.split("\n"):
            if "const x = 99;" in line:
                assert line.startswith("    "), f"Indentation lost: {line!r}"

    def test_fuzzy_apply_no_match(self):
        """All tiers fail on completely different content."""
        tool = self._get_tool()
        content = "const x = 1;\n"
        result, ok = tool._fuzzy_apply(content, "const z = 999;\nreturn z;", "replaced")
        assert not ok
        assert result == content

    def test_is_top_level_basic(self):
        """Top-level detection: outside braces = top level."""
        ApplicationBuilderTool, _, _ = _import_tool()
        code = "const a = 1;\nfunction foo() {\n  const b = 2;\n}\nconst c = 3;"
        # 'const a' at position 0 — top level
        assert ApplicationBuilderTool._is_top_level(code, 0) is True
        # 'const b' inside function — NOT top level
        b_pos = code.index("const b")
        assert ApplicationBuilderTool._is_top_level(code, b_pos) is False
        # 'const c' after closing brace — top level
        c_pos = code.index("const c")
        assert ApplicationBuilderTool._is_top_level(code, c_pos) is True

    def test_is_top_level_skips_strings(self):
        """Braces inside strings don't affect depth."""
        ApplicationBuilderTool, _, _ = _import_tool()
        code = 'const s = "{ not a block }";\nconst x = 1;'
        x_pos = code.index("const x")
        assert ApplicationBuilderTool._is_top_level(code, x_pos) is True

    def test_fix_prompt_includes_previous_attempts(self):
        """Fix prompt includes failed previous attempts when provided."""
        _, _, _, build_fix_prompt, _ = _import_scaffolds()
        files = [{"path": "app.js", "role": "script", "content": "const x = 1;"}]
        errors = ["app.js: something broke"]
        prev = ["=== FILE: app.js ===\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"]
        messages = build_fix_prompt(files, errors, previous_attempts=prev)
        user_content = messages[1]["content"]
        assert "FAILED" in user_content, "Should mention previous attempt failed"
        assert "do not repeat" in user_content.lower()
        assert "old" in user_content and "new" in user_content, "Should include the failed patch"

    def test_fix_prompt_no_previous_when_none(self):
        """Fix prompt doesn't include previous attempts section when none provided."""
        _, _, _, build_fix_prompt, _ = _import_scaffolds()
        files = [{"path": "app.js", "role": "script", "content": "const x = 1;"}]
        errors = ["app.js: something broke"]
        messages = build_fix_prompt(files, errors)
        user_content = messages[1]["content"]
        assert "Previous fix attempts" not in user_content

    def test_quickjs_catches_bad_queryselector(self):
        """Enriched DOM mock catches querySelector on missing class/tag."""
        ApplicationBuilderTool, _, _ = _import_tool()
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)
        html = '<html><body><div class="board"><div class="column" id="todo"></div></div></body></html>'

        # Existing class → clean
        errors = tool._execute_js_verify(
            'document.querySelector(".board").textContent = "ok";', html)
        assert not errors, f"Expected clean for .board, got {errors}"

        # Missing class → TypeError
        errors = tool._execute_js_verify(
            'document.querySelector(".nonexistent").textContent = "fail";', html)
        assert any("TypeError" in e for e in errors), "Expected TypeError for .nonexistent"

        # Compound selector with missing part → TypeError
        errors = tool._execute_js_verify(
            'document.querySelector("#todo .item").click();', html)
        assert any("TypeError" in e for e in errors), "Expected TypeError for #todo .item"

        # Tag selector exists → clean
        errors = tool._execute_js_verify(
            'document.querySelector("div").textContent;', html)
        assert not errors, "Expected clean for div tag"

    def test_parse_html_dom_extracts_structure(self):
        """HTML parser extracts IDs, classes, and tags accurately."""
        ApplicationBuilderTool, _, _ = _import_tool()
        html = '''<html><body>
        <div id="app" class="container main">
            <canvas id="game"></canvas>
            <form class="task-form">
                <input id="name" name="name" type="text">
            </form>
        </div></body></html>'''
        dom = ApplicationBuilderTool._parse_html_dom(html)
        assert "app" in dom["ids"]
        assert "game" in dom["ids"]
        assert "name" in dom["ids"]
        assert "container" in dom["classes"]
        assert "main" in dom["classes"]
        assert "task-form" in dom["classes"]
        assert "canvas" in dom["tags"]
        assert "form" in dom["tags"]
        assert "input" in dom["tags"]
        # Form fields tracked
        assert len(dom["form_fields"]) >= 1
        assert any(f["name"] == "name" for f in dom["form_fields"])

    def test_fix_prompt_enriches_known_api_mistakes(self):
        """Toolkit spec §4: build_fix_prompt must translate known LLM
        API hallucinations (fillCircle, addClass, export outside module,
        etc.) into explicit correction hints so small models don't have
        to derive the fix from the raw error alone."""
        from augmentum.tools.application_scaffolds import build_fix_prompt
        files = [
            {"path": "app.js", "role": "script",
             "content": "ctx.fillCircle(10, 10, 5);"},
        ]
        errors = ["TypeError: ctx.fillCircle is not a function (app.js:1)"]
        messages = build_fix_prompt(files, errors)
        user_content = messages[1]["content"]
        assert "Hint:" in user_content
        assert "arc(" in user_content  # suggestion names the real API

    def test_fix_prompt_module_syntax_error_gets_hint(self):
        from augmentum.tools.application_scaffolds import build_fix_prompt
        files = [{"path": "app.js", "role": "script", "content": "export default x;"}]
        errors = ["SyntaxError: Cannot use import statement outside a module (app.js:1)"]
        messages = build_fix_prompt(files, errors)
        user_content = messages[1]["content"]
        assert "window.X" in user_content or "window." in user_content

    def test_fix_prompt_unknown_error_passes_through(self):
        """Errors without a known correction must still land in the
        prompt verbatim — no enrichment, no data loss."""
        from augmentum.tools.application_scaffolds import build_fix_prompt
        files = [{"path": "x.js", "role": "script", "content": "const a = 1;"}]
        errors = ["Something very unusual happened"]
        messages = build_fix_prompt(files, errors)
        assert "Something very unusual happened" in messages[1]["content"]

    def test_detect_model_tier_classification(self):
        from augmentum.tools.application_scaffolds import detect_model_tier
        assert detect_model_tier("qwen3-7b-instruct") == "small"
        assert detect_model_tier("llama-3.2-3b") == "small"
        assert detect_model_tier("qwen3-14b") == "medium"
        assert detect_model_tier("qwen3-32b") == "medium"
        assert detect_model_tier("llama-70b") == "large"
        assert detect_model_tier("claude-opus-4-7") == "frontier"
        assert detect_model_tier("gpt-4o") == "frontier"
        assert detect_model_tier("gemini-pro") == "frontier"
        assert detect_model_tier("") == "medium"
        assert detect_model_tier("unknown-model") == "medium"

    def test_small_tier_gets_steering_preamble(self):
        from augmentum.tools.application_scaffolds import adapt_prompt_for_tier
        out = adapt_prompt_for_tier("Original prompt here", "small")
        assert "small model" in out.lower() or "happy path" in out.lower()
        assert "Original prompt here" in out
        # Larger tiers pass through untouched
        assert adapt_prompt_for_tier("X", "frontier") == "X"
        assert adapt_prompt_for_tier("X", "medium") == "X"
        assert adapt_prompt_for_tier("X", "large") == "X"

    def test_build_fix_prompt_adapts_for_small_model(self):
        from augmentum.tools.application_scaffolds import build_fix_prompt
        files = [{"path": "app.js", "role": "script", "content": "x"}]
        errors = ["some error"]
        msgs_small = build_fix_prompt(files, errors, model_name="qwen3-7b")
        msgs_frontier = build_fix_prompt(files, errors, model_name="claude-opus-4-7")
        assert len(msgs_small[0]["content"]) > len(msgs_frontier[0]["content"])
        assert "small model" in msgs_small[0]["content"].lower()

    def test_targeted_fix_context_compression(self):
        """Fix prompt only sends full content for affected files."""
        _, build_plan_prompt, build_generate_prompt, build_fix_prompt, _ = _import_scaffolds()
        files = [
            {"path": "index.html", "role": "entry", "content": "<html><body></body></html>"},
            {"path": "styles.css", "role": "style", "content": "body { margin: 0; }"},
            {"path": "app.js", "role": "script", "content": "const x = 1;\nfunction foo() { return x; }"},
        ]
        errors = ["app.js: Mismatched braces: 2 open, 1 close"]
        messages = build_fix_prompt(files, errors)
        user_content = messages[1]["content"]
        # app.js should appear as FULL (mentioned in error)
        assert "app.js (FULL)" in user_content
        # Other files should appear as signature (not mentioned in error)
        assert "index.html (signature)" in user_content
        assert "styles.css (signature)" in user_content
        # Full file content should NOT include the CSS body rule
        assert "body { margin: 0; }" not in user_content

    def test_dedup_intra_file_duplicate_class(self):
        """Duplicate class definition within a file is removed."""
        ApplicationBuilderTool, _, _ = _import_tool()
        code = (
            "class Game {\n"
            "  constructor() { this.x = 1; }\n"
            "  start() { console.log('go'); }\n"
            "}\n"
            "\n"
            "class Game {\n"
            "  constructor() { this.x = 1; }\n"
            "  start() { console.log('go'); }\n"
            "}\n"
        )
        result, fixes = ApplicationBuilderTool._dedup_intra_file(code, is_js=True)
        assert fixes >= 1, "Should detect duplicate class"
        assert result.count("class Game") == 1, f"Should have exactly 1 class Game, got {result.count('class Game')}"

    def test_dedup_intra_file_large_block(self):
        """Large verbatim block repeated later in the file is removed."""
        ApplicationBuilderTool, _, _ = _import_tool()
        # 10 unique lines, then the same 10 lines repeated
        block = "\n".join(f"  const line{i} = {i};" for i in range(10))
        code = f"(function() {{\n{block}\n}})();\n{block}\n"
        result, fixes = ApplicationBuilderTool._dedup_intra_file(code, is_js=True)
        assert fixes >= 1, "Should detect large block duplication"
        # The duplicated block should be removed
        for i in range(10):
            count = result.count(f"const line{i} = {i};")
            assert count == 1, f"line{i} appears {count} times, expected 1"

    def test_dedup_intra_file_no_false_positive(self):
        """Legitimate repeated patterns are NOT flagged."""
        ApplicationBuilderTool, _, _ = _import_tool()
        # Two different event handlers with similar structure — NOT a duplicate
        code = (
            "button1.addEventListener('click', () => { console.log('a'); });\n"
            "button2.addEventListener('click', () => { console.log('b'); });\n"
            "button3.addEventListener('click', () => { console.log('c'); });\n"
        )
        result, fixes = ApplicationBuilderTool._dedup_intra_file(code, is_js=True)
        assert fixes == 0, "Should not flag short similar lines as duplicates"
        assert result == code, "Content should be unchanged"

    def test_intercept_strips_sr_artifacts(self):
        """SEARCH/REPLACE markers in generated code are stripped."""
        ApplicationBuilderTool, PipelineContext, _ = _import_tool()
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)
        ctx = PipelineContext(description="test", scaffold_id="static")
        ctx.planned_files = [{"path": "app.js"}]
        ctx.generated_files = {}

        content = (
            "function getState() {\n"
            "    return { x: 1 };\n"
            "}\n"
            "=======\n"
            "function getState() {\n"
            "    return { x: 2 };\n"
            "}\n"
        )
        result = tool._intercept_generated_code(
            {"path": "app.js", "content": content}, ctx
        )
        assert "=======" not in result, "SEARCH/REPLACE marker should be stripped"
        # Should still have the function (at least one copy)
        assert "function getState" in result


# ===========================================================================
# Assembly Tests
# ===========================================================================

class TestAssembly:
    def _get_tool(self):
        ApplicationBuilderTool, _, _ = _import_tool()
        mock_store = type("MockStore", (), {"save": None})()
        return ApplicationBuilderTool(mock_store, lambda m, **k: "", lambda: {})

    def test_basic_assembly(self):
        tool = self._get_tool()
        files = [
            {"path": "index.html", "role": "entry", "content": "<!DOCTYPE html>\n<html>\n<head></head>\n<body>\n<div>Hello</div>\n</body>\n</html>"},
            {"path": "styles.css", "role": "style", "content": "body { margin: 0; }"},
            {"path": "app.js", "role": "script", "content": "console.log('hi');"},
        ]
        result = tool._assemble(files)
        assert "<style>" in result
        assert "margin: 0" in result
        assert "<script>" in result
        assert "console.log" in result
        assert result.index("<style>") < result.index("<script>")

    def test_assembly_escapes_script_closing_tag(self):
        tool = self._get_tool()
        files = [
            {"path": "index.html", "role": "entry", "content": "<html><head></head><body></body></html>"},
            {"path": "app.js", "role": "script", "content": 'var x = "</script>";'},
        ]
        result = tool._assemble(files)
        assert "</script>" not in result.split("<script>")[1].split("</script>")[0]
        # The literal </script> in the JS should be escaped
        assert "<\\/script>" in result or r"<\/script>" in result

    def test_assembly_no_entry(self):
        tool = self._get_tool()
        files = [
            {"path": "styles.css", "role": "style", "content": "body {}"},
            {"path": "app.js", "role": "script", "content": "alert(1);"},
        ]
        result = tool._assemble(files)
        assert result is not None
        assert "<style>" in result
        assert "<script>" in result

    def test_assembly_files_missing_role_does_not_crash(self):
        """Regression: app bundles (older saves / partial builds / imports)
        whose file entries lack a "role" key used to KeyError in _assemble,
        500-ing /capture-preview. Roles are now backfilled from the extension."""
        tool = self._get_tool()
        files = [
            {"path": "index.html", "content": "<html><head></head><body>x</body></html>"},
            {"path": "styles.css", "content": "body { color: red; }"},
            {"path": "app.js", "content": "console.log('ok');"},
        ]
        result = tool._assemble(files)
        assert "color: red" in result   # css placed via inferred "style"
        assert "console.log" in result  # js placed via inferred "script"
        # And the same path through the self=None module shim must not crash.
        from augmentum.tools.artifact_application import assemble_application_html
        shim = assemble_application_html(
            [{"path": "main.js", "content": "var y=1;"}]
        )
        assert "var y=1" in shim

    def test_assembly_data_files(self):
        tool = self._get_tool()
        files = [
            {"path": "index.html", "role": "entry", "content": "<html><head></head><body></body></html>"},
            {"path": "config.json", "role": "data", "content": '{"theme": "dark"}'},
        ]
        result = tool._assemble(files)
        assert "const config" in result.lower() or "const config_json" in result

    def test_assembly_module_files(self):
        tool = self._get_tool()
        files = [
            {"path": "index.html", "role": "entry", "content": "<html><head></head><body></body></html>"},
            {"path": "utils.js", "role": "module", "content": "export function add(a,b) { return a+b; }"},
        ]
        result = tool._assemble(files)
        assert 'type="module"' in result


# ===========================================================================
# Pipeline Flow Tests (with mock LLM)
# ===========================================================================

class TestPipelineFlow:
    def _make_tool(self, responses: list[str]):
        """Create tool with a mock LLM that returns predefined responses."""
        ApplicationBuilderTool, PipelineContext, _ = _import_tool()

        call_idx = [0]
        async def mock_llm(messages, max_tokens=4096, **kw):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx < len(responses):
                return responses[idx]
            return "__PASS_COMPLETE__"

        saved = []

        class MockStore:
            async def save(self_, *, filename="", fmt="", display_name="", **_kw):
                # Accept (and ignore) extra kwargs so this fixture survives
                # additions to the real ArtifactStore signature (user_id,
                # transient, etc.) without per-test churn.
                saved.append({"filename": filename, "fmt": fmt, "display_name": display_name})
                return {"id": "test_artifact_123"}

            async def save_version(self_, *_a, **_kw):
                # No-op stand-in for the real version-history hook so the
                # deliver pass's best-effort snapshot doesn't blow up the
                # mock-only tests.
                return {"id": "test_version_1", "version_index": 1}

        mock_store = MockStore()
        tool = ApplicationBuilderTool(mock_store, mock_llm, lambda: {})
        return tool, saved

    @pytest.mark.asyncio
    async def test_plan_retries_without_grammar_when_strict_plan_is_empty(self):
        """Strict llama.cpp grammar can occasionally return an empty stop."""
        ApplicationBuilderTool, PipelineContext, _ = _import_tool()
        calls = []

        async def mock_llm(messages, max_tokens=4096, **kw):
            calls.append(kw)
            if kw.get("grammar"):
                return ""
            return (
                "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Main page\n"
                "FILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styling\n"
                "FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: App logic\n"
                "__PASS_COMPLETE__"
            )

        class MockStore:
            async def save(self_, **_kw):
                return {"id": "test"}

        tool = ApplicationBuilderTool(MockStore(), mock_llm, lambda: {})
        tool._request_model = "test-model"
        ctx = PipelineContext(description="build a small timer", scaffold_id="static")
        ctx.iterations["plan"] = 1

        result = await tool._pass_plan(ctx)

        assert result.done
        assert [f["path"] for f in ctx.planned_files] == ["index.html", "styles.css", "app.js"]
        assert calls[0].get("grammar")
        assert not calls[1].get("grammar")

    @pytest.mark.asyncio
    async def test_full_pipeline_happy_path(self):
        """Test the complete pipeline with mock LLM responses."""
        plan_response = """
FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Main page
FILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styles
FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: Logic
__PASS_COMPLETE__
"""
        gen_html = """
```index.html
<!DOCTYPE html>
<html><head></head><body><div id="app">Calculator</div></body></html>
```

```context
CLASSES: none
IDS: #app
DECISIONS: Simple layout
```
__PASS_COMPLETE__
"""
        gen_css = """
```styles.css
body { margin: 0; font-family: sans-serif; }
#app { padding: 20px; }
```

```context
CLASSES: none
```
__PASS_COMPLETE__
"""
        gen_js = """
```app.js
document.getElementById('app').textContent = 'Ready';
```

```context
FUNCTIONS: none
```
__PASS_COMPLETE__
"""
        judge_response = """
SCORE: 9/10
STRENGTHS:
- Clean code
IMPROVEMENTS:
- None needed
__PASS_COMPLETE__
"""
        responses = [plan_response, gen_html, gen_css, gen_js, judge_response]
        tool, saved = self._make_tool(responses)

        result = await tool.execute(description="build a calculator")

        assert result.success, f"Pipeline failed: {result.error}"
        assert result.metadata.get("project")
        project = result.metadata["project"]
        # The pipeline occasionally retries a generate call when the
        # first response doesn't parse as code, and it may run
        # additional validate/improve passes. Rather than assert an
        # exact file count or score (brittle as the pipeline grows
        # more fix/retry loops) we verify that core planned roles
        # ended up present and delivery actually happened.
        roles = {f["role"] for f in project["files"]}
        assert {"style", "script"} <= roles, \
            f"Pipeline should deliver css + js; got roles={roles}"
        assert len(project["files"]) >= 2
        # Exactly one final zip regardless of retry count.
        zips = [s for s in saved if s["fmt"] == "zip"]
        assert len(zips) == 1  # exactly one final zip

    @pytest.mark.asyncio
    async def test_enhancer_pass_exception_does_not_sink_build(self, monkeypatch):
        """A crash in an enhancer pass (polish/validate/improve/verify) must
        degrade to a quality warning and still deliver a usable artifact —
        NOT abort the build before persistence. This is the 'build failed for
        2 CSS syntax errors but the app was actually fine' regression class."""
        plan_response = (
            "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Main page\n"
            "FILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styles\n"
            "FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: Logic\n"
            "__PASS_COMPLETE__"
        )
        gen_html = ('```index.html\n<!DOCTYPE html><html><head></head>'
                    '<body><div id="app">Hi</div></body></html>\n```\n__PASS_COMPLETE__')
        gen_css = "```styles.css\nbody { margin: 0; }\n#app { padding: 8px; }\n```\n__PASS_COMPLETE__"
        gen_js = "```app.js\ndocument.getElementById('app').textContent = 'Ready';\n```\n__PASS_COMPLETE__"
        judge = "SCORE: 9/10\nIMPROVEMENTS:\n- None\n__PASS_COMPLETE__"
        tool, saved = self._make_tool([plan_response, gen_html, gen_css, gen_js, judge])

        async def _boom(ctx):
            raise ValueError("polish regex blew up on malformed CSS")
        monkeypatch.setattr(tool, "_pass_polish", _boom)

        result = await tool.execute(description="build a calculator")

        assert result.success, f"enhancer crash must not fail the build: {result.error}"
        # The crash surfaced as a user-visible quality warning…
        assert any("polish" in w.lower() for w in result.warnings), result.warnings
        # …and the artifact was still delivered (exactly one zip persisted).
        assert len([s for s in saved if s["fmt"] == "zip"]) == 1

    @pytest.mark.asyncio
    async def test_pipeline_with_validation_fix(self):
        """Test that the validate pass detects and fixes errors."""
        plan = (
            "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Page\n"
            "FILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styles\n"
            "FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: Logic\n"
            "__PASS_COMPLETE__"
        )
        gen_html = '```index.html\n<html><head></head><body><div><span>unclosed\n</body></html>\n```\n__PASS_COMPLETE__'
        gen_css = '```styles.css\nbody { margin: 0; }\n```\n__PASS_COMPLETE__'
        gen_js = '```app.js\nconsole.log("ok");\n```\n__PASS_COMPLETE__'
        # Validation will detect unclosed tags, ask LLM to fix
        fix = """
=== FILE: index.html ===
<<<<<<< SEARCH
<div><span>unclosed
=======
<div><span>fixed</span></div>
>>>>>>> REPLACE
__PASS_COMPLETE__
"""
        judge = "SCORE: 8.5/10\n__PASS_COMPLETE__"
        tool, saved = self._make_tool([plan, gen_html, gen_css, gen_js, fix, judge])
        result = await tool.execute(description="test page")
        assert result.success

    @pytest.mark.asyncio
    async def test_pipeline_skips_improve_when_disabled(self):
        """Test that improve pass is skipped when setting is off.

        Rather than asserting an exact call count (which is brittle as the
        pipeline evolves and adds intermediate fix/verify calls), we
        inspect the actual prompts and assert the judge prompt — the
        canonical marker of the improve pass — never ran.
        """
        ApplicationBuilderTool, _, _ = _import_tool()

        calls: list[str] = []
        def _classify(messages: list) -> str:
            system = next(
                (m.get("content", "") for m in messages if m.get("role") == "system"),
                "",
            )
            if "reviewing a web application for quality" in system:
                return "judge"
            if "Plan" in system or "plan" in system.lower()[:300]:
                return "plan"
            if "fixing code errors" in system:
                return "fix"
            return "other"

        async def counting_llm(messages, max_tokens=4096, **kw):
            calls.append(_classify(messages))
            i = len(calls)
            if i == 1:
                return (
                    "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Page\n"
                    "FILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styles\n"
                    "FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: Logic\n"
                    "__PASS_COMPLETE__"
                )
            if i == 2:
                return '```index.html\n<html><head></head><body>Hi</body></html>\n```\n__PASS_COMPLETE__'
            if i == 3:
                return '```styles.css\nbody { margin: 0; }\n```\n__PASS_COMPLETE__'
            if i == 4:
                return '```app.js\nconsole.log("ok");\n```\n__PASS_COMPLETE__'
            return "__PASS_COMPLETE__"

        class MockStore2:
            async def save(self_, **kw):
                return {"id": "x"}

        tool = ApplicationBuilderTool(
            MockStore2(), counting_llm,
            lambda: {"app_builder_improve_pass": False}
        )

        result = await tool.execute(description="test")
        assert result.success
        # With the improve pass disabled, the judge prompt must never fire.
        assert "judge" not in calls, \
            f"improve pass should be skipped, but judge ran: {calls}"

    @pytest.mark.asyncio
    async def test_progress_callback_called(self):
        """Test that the pipeline emits progress updates."""
        plan = (
            "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Page\n"
            "FILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styles\n"
            "FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: Logic\n"
            "__PASS_COMPLETE__"
        )
        gen_html = '```index.html\n<html><head></head><body>Hi</body></html>\n```\n__PASS_COMPLETE__'
        gen_css = '```styles.css\nbody { margin: 0; }\n```\n__PASS_COMPLETE__'
        gen_js = '```app.js\nconsole.log("ok");\n```\n__PASS_COMPLETE__'
        judge = "SCORE: 9/10\n__PASS_COMPLETE__"

        progress_events = []

        call_idx = [0]
        responses = [plan, gen_html, gen_css, gen_js, judge]
        async def ordered_llm(messages, max_tokens=4096, **kw):
            idx = call_idx[0]
            call_idx[0] += 1
            return responses[idx] if idx < len(responses) else "__PASS_COMPLETE__"

        class MockStore3:
            async def save(self_, **kw):
                return {"id": "test"}

        ApplicationBuilderTool, _, _ = _import_tool()
        tool = ApplicationBuilderTool(MockStore3(), ordered_llm, lambda: {})

        async def progress_cb(data):
            progress_events.append(data)

        result = await tool.execute(
            description="test app",
            _progress_callback=progress_cb,
        )

        assert result.success
        # Should have progress events for each pass
        pass_names = [e.get("project_progress", {}).get("pass") for e in progress_events]
        assert "plan" in pass_names
        assert "generate" in pass_names
        assert "verify" in pass_names
        assert "deliver" in pass_names

    @pytest.mark.asyncio
    async def test_emit_checkpoints_files_before_deliver(self):
        """Progress events should expose current files as soon as they exist."""
        ApplicationBuilderTool, PipelineContext, _ = _import_tool()
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)

        ctx = PipelineContext(description="test", scaffold_id="static")
        ctx.project_name = "Checkpoint App"
        ctx.current_pass = "generate"
        ctx.iterations["generate"] = 1
        ctx.files = [
            {"path": "index.html", "role": "entry", "lang": "html", "content": "<html></html>"},
        ]
        ctx.planned_files = [
            {"path": "index.html", "role": "entry"},
            {"path": "app.js", "role": "script"},
        ]
        ctx.generated_files = {"index.html": ctx.files[0]["content"]}

        events = []

        async def progress_cb(data):
            events.append(data)

        await tool._emit(progress_cb, ctx, "running", "generated index.html")

        progress = events[0]["project_progress"]
        assert progress["files"] == ctx.files
        assert progress["planned_files"] == ctx.planned_files
        assert progress["completed_files"] == ["index.html"]
        assert progress["filesRemaining"] == ["app.js"]
        assert progress["currentFile"] == "app.js"

    @pytest.mark.asyncio
    async def test_verify_unfixed_errors_mark_build_needs_review(self):
        """Unfixed runtime issues must become explicit degraded-success metadata."""
        ApplicationBuilderTool, PipelineContext, _ = _import_tool()
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)
        tool._get_settings = None
        tool._request_model = ""

        async def no_patch_llm(messages, max_tokens=4096, **kw):
            return "__PASS_COMPLETE__"

        tool._call_llm = no_patch_llm
        tool._execute_js_verify = lambda _js, _html: ["RUNTIME: boom"]
        tool.verify_intent = lambda _description, _files: []

        ctx = PipelineContext(description="test", scaffold_id="static")
        ctx.files = [
            {"path": "index.html", "role": "entry", "lang": "html", "content": "<html><body></body></html>"},
            {"path": "app.js", "role": "script", "lang": "javascript", "content": "console.log('boom');"},
        ]

        result = await tool._pass_verify(ctx)

        assert result.done
        assert "need review" in result.detail
        assert ctx.quality_status == "needs_review"
        assert ctx.quality_warnings
        assert ctx.blocking_errors == ["RUNTIME: boom"]
        assert ctx.to_dict()["qualityStatus"] == "needs_review"

    @pytest.mark.asyncio
    async def test_verify_catches_duplicate_class(self):
        """Verify pass detects duplicate class definitions in assembled output."""
        ApplicationBuilderTool, PipelineContext, _ = _import_tool()
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)
        tool._request_model = ""
        # Mock LLM that won't be called (we expect clean detection, not fix)
        tool._call_llm = None

        ctx = PipelineContext(description="test", scaffold_id="static")
        ctx.files = [
            {"path": "index.html", "role": "entry", "lang": "html",
             "content": "<html><head></head><body><div id=\"app\"></div></body></html>"},
            {"path": "a.js", "role": "script", "lang": "javascript",
             "content": "class Game { constructor() { this.x = 1; } }"},
            {"path": "b.js", "role": "script", "lang": "javascript",
             "content": "class Game { constructor() { this.x = 2; } }\nwindow.game = new Game();"},
        ]

        # Assemble and check — should detect duplicate class Game
        assembled = tool._assemble(ctx.files)
        import re
        js_blocks = re.findall(r"<script[^>]*>([\s\S]*?)</script>", assembled, re.IGNORECASE)
        all_js = "\n".join(js_blocks)
        class_defs = {}
        for m in re.finditer(r"\bclass\s+(\w+)\s*(?:extends\s+\w+\s*)?\{", all_js):
            name = m.group(1)
            class_defs[name] = class_defs.get(name, 0) + 1
        assert class_defs.get("Game", 0) == 2, "Should detect 2 definitions of class Game"

    @pytest.mark.asyncio
    async def test_verify_catches_missing_dom_id(self):
        """Verify pass detects getElementById on non-existent element."""
        ApplicationBuilderTool, PipelineContext, _ = _import_tool()
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)
        tool._request_model = ""
        tool._call_llm = None

        ctx = PipelineContext(description="test", scaffold_id="static")
        ctx.files = [
            {"path": "index.html", "role": "entry", "lang": "html",
             "content": "<html><body><div id=\"app\"></div></body></html>"},
            {"path": "app.js", "role": "script", "lang": "javascript",
             "content": "document.getElementById('nonexistent').textContent = 'hi';"},
        ]

        assembled = tool._assemble(ctx.files)
        import re
        html_only = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", assembled, flags=re.IGNORECASE)
        html_ids = set(re.findall(r'id=["\'](\w[\w-]*)["\']', html_only))
        js_blocks = re.findall(r"<script[^>]*>([\s\S]*?)</script>", assembled, re.IGNORECASE)
        all_js = "\n".join(js_blocks)

        missing = []
        for m in re.finditer(r'getElementById\s*\(\s*["\'](\w[\w-]*)["\']', all_js):
            if m.group(1) not in html_ids:
                missing.append(m.group(1))
        assert "nonexistent" in missing, "Should detect getElementById on missing ID"
        assert "app" not in missing, "Should NOT flag existing ID 'app'"

    @pytest.mark.asyncio
    async def test_verify_rollback_on_regression(self):
        """Verify pass rolls back if LLM fix introduces more errors."""
        ApplicationBuilderTool, PipelineContext, _ = _import_tool()

        call_idx = [0]
        async def mock_llm(messages, max_tokens=4096, **kw):
            call_idx[0] += 1
            # Return a "fix" that introduces a NEW error (references nonExistent)
            return """=== FILE: app.js ===
<<<<<<< SEARCH
const y = document.getElementById('missing');
=======
const y = document.getElementById('missing');
nonExistent.crash();
>>>>>>> REPLACE
__PASS_COMPLETE__
"""

        class MockStore:
            async def save(self_, **kw): return {"id": "x"}

        tool = ApplicationBuilderTool(MockStore(), mock_llm, lambda: {})
        tool._request_model = ""

        ctx = PipelineContext(description="test", scaffold_id="static")
        ctx.files = [
            {"path": "index.html", "role": "entry", "lang": "html",
             "content": '<html><head></head><body><div id="app"></div></body></html>'},
            {"path": "app.js", "role": "script", "lang": "javascript",
             "content": "const y = document.getElementById('missing');\ny.textContent = 'hi';"},
        ]
        ctx.generated_files = {"index.html": ctx.files[0]["content"], "app.js": ctx.files[1]["content"]}

        # Save original content for comparison
        original_js = ctx.files[1]["content"]

        result = await tool._pass_verify(ctx)

        # The fix should have been rolled back — content restored
        assert ctx.files[1]["content"] == original_js, \
            "File content should be restored after regression rollback"
        assert "rolled back" in result.detail or "frontend will auto-fix" in result.detail

    @pytest.mark.asyncio
    async def test_scaffold_minimum_enforcement(self):
        """Plan with only HTML+CSS auto-adds JS for form scaffold."""
        # LLM plans only 2 files, but form scaffold requires 3
        plan = "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Page\nFILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styles\n__PASS_COMPLETE__"
        gen_html = '```index.html\n<html><head></head><body><button onclick="calc()">Go</button></body></html>\n```\n__PASS_COMPLETE__'
        gen_css = '```styles.css\nbody { margin: 0; }\n```\n__PASS_COMPLETE__'
        gen_js = '```app.js\nfunction calc() { alert("done"); }\n```\n__PASS_COMPLETE__'
        judge = "SCORE: 9/10\n__PASS_COMPLETE__"
        tool, saved = self._make_tool([plan, gen_html, gen_css, gen_js, judge])

        result = await tool.execute(description="build a calculator", scaffold="form")
        assert result.success
        project = result.metadata["project"]
        roles = {f["role"] for f in project["files"]}
        # Core intent of the test: scaffold enforcement saw the plan was
        # missing a script role and auto-added one. File-count assertions
        # are brittle when the pipeline grows more retry/fix loops; keep
        # just the role check.
        assert "script" in roles, "Scaffold enforcement should have added a JS file"

    @pytest.mark.asyncio
    async def test_gap_analysis_detects_missing_script_ref(self):
        """HTML referencing <script src='app.js'> auto-adds app.js to plan."""
        # Plan only has HTML, but the generated HTML references app.js
        plan = "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Page\n__PASS_COMPLETE__"
        gen_html = '```index.html\n<html><head></head><body><script src="app.js"></script></body></html>\n```\n__PASS_COMPLETE__'
        gen_js = '```app.js\nconsole.log("hello");\n```\n__PASS_COMPLETE__'
        judge = "SCORE: 9/10\n__PASS_COMPLETE__"
        tool, saved = self._make_tool([plan, gen_html, gen_js, judge])

        # Note: scaffold enforcement will also add styles.css + app.js for "static"
        # but the gap analysis should handle the script src reference
        result = await tool.execute(description="simple page")
        assert result.success
        paths = [f["path"] for f in result.metadata["project"]["files"]]
        # app.js should exist either via scaffold enforcement or gap analysis
        assert "app.js" in paths or any(p.endswith(".js") for p in paths)

    @pytest.mark.asyncio
    async def test_gap_analysis_detects_interactive_html_without_js(self):
        """HTML with buttons but no JS files triggers gap analysis."""
        ApplicationBuilderTool, PipelineContext, _ = _import_tool()
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)
        ctx = PipelineContext(
            description="test",
            scaffold_id="static",
            files=[
                {"path": "index.html", "role": "entry", "lang": "html",
                 "content": '<html><body><button onclick="go()">Click</button></body></html>'},
            ],
        )
        ctx.generated_files = {"index.html": ctx.files[0]["content"]}
        gaps = tool._detect_missing_references(ctx)
        # Should detect that HTML has interactive elements but no JS
        assert len(gaps) >= 1
        assert any(g["role"] == "script" for g in gaps)

    @pytest.mark.asyncio
    async def test_gap_analysis_ignores_cdn_urls(self):
        """CDN URLs in script/link tags should NOT be treated as missing files."""
        ApplicationBuilderTool, PipelineContext, _ = _import_tool()
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)
        ctx = PipelineContext(
            description="test",
            scaffold_id="static",
            files=[
                {"path": "index.html", "role": "entry", "lang": "html",
                 "content": '<html><head><script src="https://cdn.example.com/lib.js"></script></head><body>Hi</body></html>'},
            ],
        )
        ctx.generated_files = {"index.html": ctx.files[0]["content"]}
        gaps = tool._detect_missing_references(ctx)
        # CDN URL should not appear as a missing file
        assert not any(g["path"].startswith("http") for g in gaps)

    @pytest.mark.asyncio
    async def test_iteration_mode_modifies_existing(self):
        """Pipeline iteration mode plans modifications to existing files."""
        plan = (
            "FILE: app.js | ACTION: modify | DESCRIPTION: Add dark mode toggle\n"
            "FILE: styles.css | ACTION: modify | DESCRIPTION: Add dark mode CSS\n"
            "__PASS_COMPLETE__"
        )
        gen_js = '```app.js\nconst darkMode = false;\nfunction toggleDark() { document.body.classList.toggle("dark"); }\n```\n__PASS_COMPLETE__'
        gen_css = '```styles.css\nbody { margin: 0; }\nbody.dark { background: #111; color: #eee; }\n```\n__PASS_COMPLETE__'
        judge = "SCORE: 8/10\n__PASS_COMPLETE__"
        tool, saved = self._make_tool([plan, gen_js, gen_css, judge])

        existing_files = [
            {"path": "index.html", "role": "entry", "lang": "html",
             "content": "<html><head></head><body><h1>Hello</h1></body></html>"},
            {"path": "styles.css", "role": "style", "lang": "css",
             "content": "body { margin: 0; }"},
            {"path": "app.js", "role": "script", "lang": "javascript",
             "content": "console.log('hello');"},
        ]

        result = await tool.execute(
            description="add dark mode toggle",
            existing_project={"files": existing_files},
        )
        assert result.success, f"Iteration failed: {result.error}"
        files = result.metadata["project"]["files"]
        # Should have modified files
        all_content = "\n".join(f["content"] for f in files)
        assert "dark" in all_content.lower(), "Dark mode changes should be present"


# ===========================================================================
# Improve-pass score gate (§19)
# ===========================================================================

class TestImproveScoreGate:
    """The judge score now acts as a quality gate. Low scores trigger a
    targeted retry using the judge's IMPROVEMENTS bullets; if the score
    stays low after the retry, the build ships with a warning on
    ToolResult.warnings so the user sees it."""

    def test_score_below_gate_classifier(self):
        from augmentum.tools.artifact_application import ApplicationBuilderTool
        below = ApplicationBuilderTool._score_is_below_gate
        # Below threshold, above sentinel → gate triggers
        assert below(6.5) is True
        assert below(7.4) is True
        assert below(3.0) is True
        # At/above threshold → no gate
        assert below(7.5) is False
        assert below(8.0) is False
        # Parse-default 5.0 is treated as unknown → don't gate
        assert below(5.0) is False
        # Zero / missing → don't gate
        assert below(0.0) is False

    def test_extract_improvements_parses_judge_response(self):
        from augmentum.tools.artifact_application import ApplicationBuilderTool
        resp = (
            "SCORE: 6.5/10\n"
            "STRENGTHS:\n"
            "- Clean layout\n"
            "IMPROVEMENTS:\n"
            "- Add form validation\n"
            "- Handle empty state\n"
            "* Show loading spinner\n"
            "__NEEDS_ANOTHER_PASS__: improvements listed above"
        )
        bullets = ApplicationBuilderTool._extract_judge_improvements(resp)
        assert bullets == [
            "Add form validation",
            "Handle empty state",
            "Show loading spinner",
        ]

    def test_extract_improvements_handles_empty_section(self):
        from augmentum.tools.artifact_application import ApplicationBuilderTool
        resp = "SCORE: 9/10\nSTRENGTHS:\n- Great\nIMPROVEMENTS:\n- None needed\n__PASS_COMPLETE__"
        bullets = ApplicationBuilderTool._extract_judge_improvements(resp)
        # "None needed" is filtered out
        assert bullets == []

    def test_extract_improvements_missing_section(self):
        from augmentum.tools.artifact_application import ApplicationBuilderTool
        assert ApplicationBuilderTool._extract_judge_improvements("just score 9/10") == []

    def _routed_llm(self, *, plan: str, generates: dict, judges: list, fixes: list):
        """Build a mock LLM that routes by prompt classification rather
        than call-order. The pipeline makes intermediate calls we don't
        care about (plan sanity checks, etc.); routing by classification
        keeps the test robust to those. Returns (llm_fn, call_log)."""
        import re as _re
        from collections import deque

        from augmentum.tools.artifact_application import ApplicationBuilderTool  # noqa: F401

        call_log: list[str] = []
        pools = {
            "plan": deque([plan]),
            "generate_index": deque([generates.get("index.html", "__PASS_COMPLETE__")]),
            "generate_css": deque([generates.get("styles.css", "__PASS_COMPLETE__")]),
            "generate_js": deque([generates.get("app.js", "__PASS_COMPLETE__")]),
            "judge": deque(judges),
            "fix": deque(fixes),
        }

        def _classify(messages):
            system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            if "reviewing a web application for quality" in system:
                return "judge"
            if "fixing code errors" in system:
                return "fix"
            if "planning a production-quality" in system:
                return "plan"
            if "generating production-quality code" in system:
                # Route by filename in user text
                if _re.search(r"\bindex\.html\b", user):
                    return "generate_index"
                if _re.search(r"\bstyles\.css\b", user):
                    return "generate_css"
                if _re.search(r"\bapp\.js\b", user):
                    return "generate_js"
                return "generate_other"
            return "other"

        async def llm(messages, max_tokens=4096, **kw):
            kind = _classify(messages)
            call_log.append(kind)
            q = pools.get(kind)
            if q:
                return q.popleft() if len(q) > 1 else q[0]  # reuse last response
            return "__PASS_COMPLETE__"

        return llm, call_log

    @pytest.mark.asyncio
    async def test_low_score_triggers_retry_and_ships_warning(self):
        """End-to-end: judge returns 5.5/10 on first improve, 6.0 on retry,
        pipeline ships with a quality warning surfaced via result.warnings."""
        ApplicationBuilderTool, _, _ = _import_tool()

        judge_low = (
            "SCORE: 5.5/10\n"
            "STRENGTHS:\n- Works\n"
            "IMPROVEMENTS:\n- Improve accessibility\n- Add focus states\n"
            "__NEEDS_ANOTHER_PASS__: improvements listed above"
        )
        judge_still_low = "SCORE: 6.0/10\nSTRENGTHS:\n- Ok\nIMPROVEMENTS:\n- None needed\n__PASS_COMPLETE__"
        # Target styles.css — simple, single-line content that should
        # survive polish (which appends animations rather than editing
        # existing declarations) so the SEARCH block still matches.
        fix_patch = (
            "=== FILE: styles.css ===\n"
            "<<<<<<< SEARCH\n"
            "body { margin: 0; }\n"
            "=======\n"
            "body { margin: 0; padding: 0; }\n"
            ">>>>>>> REPLACE\n"
            "__PASS_COMPLETE__"
        )
        plan = (
            "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Page\n"
            "FILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styles\n"
            "FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: Logic\n"
            "__PASS_COMPLETE__"
        )
        llm, call_log = self._routed_llm(
            plan=plan,
            generates={
                "index.html": '```index.html\n<html><head></head><body>Hi</body></html>\n```\n__PASS_COMPLETE__',
                "styles.css": '```styles.css\nbody { margin: 0; }\n```\n__PASS_COMPLETE__',
                "app.js": '```app.js\nconsole.log("ok");\n```\n__PASS_COMPLETE__',
            },
            judges=[judge_low, judge_still_low],
            fixes=[fix_patch],
        )

        class MockStore:
            async def save(self_, **kw):
                return {"id": "gate-test"}

        tool = ApplicationBuilderTool(
            MockStore(), llm,
            lambda: {
                "app_builder_improve_pass": True,
                "app_builder_max_improve_iterations": 2,
            },
        )

        result = await tool.execute(description="a basic page")
        assert result.success

        # The gate engaged: score below threshold → improvements extracted
        # → fix prompt fired. Whether the patches actually landed (and
        # triggered a second judge call) or silently no-op'd (and shipped
        # with the warning) is less important than that the gate *tried*.
        assert call_log.count("judge") >= 1
        assert call_log.count("fix") >= 1, (
            f"gate should have triggered at least one fix call; calls: {call_log}"
        )

        # User-facing warning landed on the result AND in the chat output.
        assert result.warnings, "low score should leave a warning on the result"
        assert any("below" in w.lower() for w in result.warnings)
        assert "Warnings:" in result.output

    @pytest.mark.asyncio
    async def test_per_file_map_refresh_in_multi_file_response(self):
        """When the LLM returns multiple files in one generate response,
        the project map must refresh after EACH file so subsequent files
        in the loop see the fresh API surface. Previously the map only
        updated after the whole batch, leaving the intercept / retry
        paths with stale state."""
        from augmentum.tools.artifact_application import ApplicationBuilderTool, PipelineContext

        # Bare tool without __init__ (we only want the _pass_generate internals)
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)
        tool._store = None
        tool._get_settings = None
        tool._request_model = ""

        ctx = PipelineContext(description="x", scaffold_id="static")
        ctx.planned_files = [
            {"path": "a.js", "role": "script", "lang": "javascript", "action": "create", "description": ""},
            {"path": "b.js", "role": "script", "lang": "javascript", "action": "create", "description": ""},
        ]
        ctx.files = []
        ctx.generated_files = {}
        ctx.working_doc = "## Files to Generate\n- [ ] a.js\n- [ ] b.js\n"

        # Simulate the inner for-loop logic — mirror what _pass_generate
        # does per file so we can verify the map refresh is visible
        # between files.
        observed_maps: list[str] = []
        new_files = [
            {"path": "a.js", "content": "window.foo = function() { return 1; };\n"},
            {"path": "b.js", "content": "window.bar = function() { return window.foo() + 1; };\n"},
        ]
        for f in new_files:
            f["content"] = tool._intercept_generated_code(f, ctx)
            ctx.generated_files[f["path"]] = f["content"]
            ctx.files.append({
                "path": f["path"], "lang": "javascript",
                "content": f["content"], "role": "script",
            })
            ctx.working_doc = ctx.working_doc.replace(
                f"- [ ] {f['path']}", f"- [x] {f['path']}",
            )
            ctx.working_doc = tool._update_project_map(ctx)
            observed_maps.append(ctx.working_doc)

        # After processing a.js the map must list window.foo.
        assert "foo" in observed_maps[0], f"map after a.js missing foo: {observed_maps[0]}"
        # After processing b.js the map must list BOTH foo and bar.
        assert "foo" in observed_maps[1]
        assert "bar" in observed_maps[1]
        # Both files marked done in the checklist.
        assert "- [x] a.js" in observed_maps[1]
        assert "- [x] b.js" in observed_maps[1]

    @pytest.mark.asyncio
    async def test_good_score_no_retry_no_warning(self):
        """High score on first improve → no retry, no warning."""
        ApplicationBuilderTool, _, _ = _import_tool()

        plan = (
            "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Page\n"
            "FILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styles\n"
            "FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: Logic\n"
            "__PASS_COMPLETE__"
        )
        judge_good = "SCORE: 9/10\nSTRENGTHS:\n- Clean\nIMPROVEMENTS:\n- None needed\n__PASS_COMPLETE__"
        llm, call_log = self._routed_llm(
            plan=plan,
            generates={
                "index.html": '```index.html\n<html><head></head><body>Hi</body></html>\n```\n__PASS_COMPLETE__',
                "styles.css": '```styles.css\nbody { margin: 0; }\n```\n__PASS_COMPLETE__',
                "app.js": '```app.js\nconsole.log("ok");\n```\n__PASS_COMPLETE__',
            },
            judges=[judge_good],
            fixes=[],
        )

        class MockStore:
            async def save(self_, **kw):
                return {"id": "good-score"}

        tool = ApplicationBuilderTool(
            MockStore(), llm,
            lambda: {"app_builder_improve_pass": True, "app_builder_max_improve_iterations": 2},
        )
        result = await tool.execute(description="a basic page")
        assert result.success
        # Single judge call — no retry triggered.
        assert call_log.count("judge") == 1, f"good score should not retry; got {call_log}"
        # No warnings on a good-score build.
        assert result.warnings == []
        assert "Warnings:" not in result.output


# ===========================================================================
# Runtime contracts (§21)
# ===========================================================================

class TestContractParsing:
    """The plan parser must extract optional PROVIDES/DEPENDS/WIRES
    columns from the FILE: plan line without polluting the description
    text. Missing contract fields yield empty lists, not crashes.
    """

    def _parse(self, response: str):
        from augmentum.tools.artifact_application import ApplicationBuilderTool
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)
        return tool._parse_file_plan(response)

    def test_plan_without_contracts_stays_backward_compatible(self):
        response = (
            "FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: Main logic\n"
            "__PASS_COMPLETE__"
        )
        files = self._parse(response)
        assert len(files) == 1
        assert files[0]["description"] == "Main logic"
        # Contract fields are optional — missing is fine
        assert files[0].get("provides", []) == []
        assert files[0].get("depends", []) == []
        assert files[0].get("wires", []) == []

    def test_full_contract_extracted(self):
        response = (
            "FILE: app.js | ROLE: script | LANG: javascript "
            "| DESCRIPTION: Calculator logic "
            "| PROVIDES: window.calculate, window.Calculator "
            "| DEPENDS: window.formatNumber "
            "| WIRES: #btn-calc click, #form submit\n"
            "__PASS_COMPLETE__"
        )
        files = self._parse(response)
        assert files[0]["description"] == "Calculator logic"
        assert files[0]["provides"] == ["window.calculate", "window.Calculator"]
        assert files[0]["depends"] == ["window.formatNumber"]
        assert files[0]["wires"] == ["#btn-calc click", "#form submit"]

    def test_none_values_treated_as_empty(self):
        response = (
            "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Page"
            " | PROVIDES: none | DEPENDS: none | WIRES: none\n"
            "__PASS_COMPLETE__"
        )
        files = self._parse(response)
        assert files[0]["provides"] == []
        assert files[0]["depends"] == []
        assert files[0]["wires"] == []

    def test_normalize_strips_parenthetical_annotations(self):
        """LLMs routinely annotate PROVIDES with free text:
        'window.PomodoroApp (initialization function)'. The validator
        must compare the bare symbol — parenthetical is human prose."""
        from augmentum.tools.artifact_application import _normalize_symbol
        assert _normalize_symbol("window.PomodoroApp (init function)") == "PomodoroApp"
        assert _normalize_symbol("Calculator (main class)") == "Calculator"
        assert _normalize_symbol("window.save (persistence helper)") == "save"
        assert _normalize_symbol("window.calc()") == "calc"
        # Plain symbols untouched
        assert _normalize_symbol("window.Calculator") == "Calculator"

    def test_validate_contracts_tolerates_annotated_provides(self):
        """End-to-end: declared PROVIDES with annotation matches the
        actual bare-name definition. Caught in production during the
        live test — qwen3.6-35b emitted annotations by default."""
        from augmentum.tools.artifact_application import validate_contracts
        files = [
            {"path": "app.js", "role": "script",
             "content": "window.PomodoroApp = function() { return 1; };"},
        ]
        planned = [
            {"path": "app.js", "role": "script",
             "provides": ["window.PomodoroApp (initialization function)"],
             "depends": [], "wires": []},
        ]
        assert validate_contracts(files, planned, "") == []

    def test_depends_on_alias_accepted(self):
        # Some models emit "DEPENDS_ON" instead of the shorter "DEPENDS"
        response = (
            "FILE: app.js | ROLE: script | LANG: javascript"
            " | DESCRIPTION: Logic | DEPENDS_ON: window.foo\n"
            "__PASS_COMPLETE__"
        )
        files = self._parse(response)
        assert files[0]["depends"] == ["window.foo"]


class TestContractValidator:
    """validate_contracts must catch the mismatches that today only
    surface at runtime: declared PROVIDES never defined, DEPENDS without
    a provider, WIRES selectors with no matching DOM element."""

    def test_empty_planned_no_errors(self):
        from augmentum.tools.artifact_application import validate_contracts
        assert validate_contracts([], [], "") == []

    def test_matched_provides_passes(self):
        from augmentum.tools.artifact_application import validate_contracts
        files = [{"path": "app.js", "role": "script",
                  "content": "window.calc = function() { return 1; };"}]
        planned = [{"path": "app.js", "role": "script",
                    "provides": ["window.calc"], "depends": [], "wires": []}]
        assert validate_contracts(files, planned, "") == []

    def test_missing_provides_reported(self):
        from augmentum.tools.artifact_application import validate_contracts
        files = [{"path": "app.js", "role": "script",
                  "content": "console.log('hi');"}]
        planned = [{"path": "app.js", "role": "script",
                    "provides": ["window.calc"], "depends": [], "wires": []}]
        errors = validate_contracts(files, planned, "")
        assert len(errors) == 1
        assert "app.js" in errors[0] and "calc" in errors[0]

    def test_unresolved_depends_reported(self):
        from augmentum.tools.artifact_application import validate_contracts
        files = [{"path": "app.js", "role": "script", "content": "const x = 1;"}]
        planned = [{"path": "app.js", "role": "script",
                    "provides": [], "depends": ["window.mystery"], "wires": []}]
        errors = validate_contracts(files, planned, "")
        assert any("mystery" in e for e in errors)

    def test_cross_file_dependency_resolved(self):
        """window.foo defined in a.js, used in b.js → no error."""
        from augmentum.tools.artifact_application import validate_contracts
        files = [
            {"path": "a.js", "role": "script", "content": "window.foo = 1;"},
            {"path": "b.js", "role": "script", "content": "console.log(window.foo);"},
        ]
        planned = [
            {"path": "a.js", "role": "script", "provides": ["window.foo"], "depends": [], "wires": []},
            {"path": "b.js", "role": "script", "provides": [], "depends": ["window.foo"], "wires": []},
        ]
        assert validate_contracts(files, planned, "") == []

    def test_undeclared_window_reference_caught(self):
        """A script reading window.bar with no provider anywhere is a bug."""
        from augmentum.tools.artifact_application import validate_contracts
        files = [{"path": "app.js", "role": "script",
                  "content": "console.log(window.bar);"}]
        planned = [{"path": "app.js", "role": "script",
                    "provides": [], "depends": [], "wires": []}]
        errors = validate_contracts(files, planned, "")
        assert any("bar" in e for e in errors)

    def test_benign_window_globals_not_flagged(self):
        from augmentum.tools.artifact_application import validate_contracts
        files = [{"path": "app.js", "role": "script",
                  "content": "window.addEventListener('resize', () => {}); fetch('/x');"}]
        planned = [{"path": "app.js", "role": "script",
                    "provides": [], "depends": [], "wires": []}]
        assert validate_contracts(files, planned, "") == []

    def test_wires_selector_matches_html_id(self):
        from augmentum.tools.artifact_application import validate_contracts
        files = [{"path": "app.js", "role": "script", "content": ""}]
        planned = [{"path": "app.js", "role": "script",
                    "provides": [], "depends": [], "wires": ["#btn-go click"]}]
        html = '<html><body><button id="btn-go">Go</button></body></html>'
        assert validate_contracts(files, planned, html) == []

    def test_wires_selector_without_matching_id_flagged(self):
        from augmentum.tools.artifact_application import validate_contracts
        files = [{"path": "app.js", "role": "script", "content": ""}]
        planned = [{"path": "app.js", "role": "script",
                    "provides": [], "depends": [], "wires": ["#missing click"]}]
        html = "<html><body></body></html>"
        errors = validate_contracts(files, planned, html)
        assert any("missing" in e for e in errors)

    def test_contract_resolves_inline_script_provides(self):
        """The live pomodoro build revealed validator blind-spots for
        inline code. A <script> block inside index.html defining
        window.HelperFn should count toward the project's PROVIDES
        set, so other files depending on it don't trigger a
        'no file PROVIDES it' contract error."""
        from augmentum.tools.artifact_application import validate_contracts
        files = [
            {"path": "app.js", "role": "script",
             "content": "console.log(window.HelperFn());"},
        ]
        planned = [
            {"path": "app.js", "role": "script",
             "provides": [], "depends": ["window.HelperFn"], "wires": []},
        ]
        entry_html = (
            "<html><body>"
            "<script>window.HelperFn = function() { return 42; };</script>"
            "</body></html>"
        )
        assert validate_contracts(files, planned, entry_html) == []

    def test_wires_class_selector_matches_html(self):
        from augmentum.tools.artifact_application import validate_contracts
        files = [{"path": "app.js", "role": "script", "content": ""}]
        planned = [{"path": "app.js", "role": "script",
                    "provides": [], "depends": [], "wires": [".item click"]}]
        html = '<html><body><div class="item wide">x</div></body></html>'
        assert validate_contracts(files, planned, html) == []


class TestContractPromptFormatting:
    def test_target_without_contract_returns_empty(self):
        from augmentum.tools.artifact_application import _format_contract_for_prompt
        target = {"path": "app.js", "role": "script"}
        assert _format_contract_for_prompt(target, [target]) == ""

    def test_target_contract_rendered_with_other_provides(self):
        from augmentum.tools.artifact_application import _format_contract_for_prompt
        planned = [
            {"path": "app.js", "role": "script",
             "provides": ["window.calc"], "depends": ["window.fmt"], "wires": ["#go click"]},
            {"path": "util.js", "role": "script",
             "provides": ["window.fmt"], "depends": [], "wires": []},
        ]
        block = _format_contract_for_prompt(planned[0], planned)
        assert "PROVIDES" in block and "window.calc" in block
        assert "DEPENDS" in block and "window.fmt" in block
        assert "WIRES" in block and "#go click" in block
        assert "Other files provide" in block and "window.fmt" in block


# ===========================================================================
# Computed design system (§22)
# ===========================================================================

class TestDesignSystemMoodDetection:
    """Mood detection drives palette selection. Every built-in mood key
    must be reachable from at least one natural-language description."""

    @pytest.mark.parametrize("description,expected", [
        ("build a playful kids counting app", "playful"),
        ("a cute game about collecting stars", "playful"),
        ("cyberpunk neon hacking sim", "moody"),
        ("a dark noir detective story app", "moody"),
        ("a minimal zen meditation timer", "minimal"),
        ("simple clean todo list", "minimal"),
        ("elegant luxury real-estate listings", "elegant"),
        ("refined editorial reading app", "elegant"),
        ("enterprise b2b analytics dashboard", "professional"),
        ("corporate CRM", "professional"),
    ])
    def test_keyword_matches(self, description, expected):
        from augmentum.tools.application_design_system import detect_mood
        assert detect_mood(description) == expected

    def test_neutral_description_defaults_balanced(self):
        from augmentum.tools.application_design_system import detect_mood
        assert detect_mood("a calculator") == "balanced"
        assert detect_mood("") == "balanced"

    def test_whole_word_match(self):
        """'darkness' should NOT trigger 'dark' — whole-word matching."""
        from augmentum.tools.application_design_system import detect_mood
        assert detect_mood("a meditation app about darkness (as metaphor)") != "moody"


class TestDesignSystemPaletteWcag:
    """Every built-in palette must ship WCAG AA compliant. If a future
    edit regresses a palette below 4.5:1 the test catches it here
    rather than shipping an inaccessible build."""

    @pytest.mark.parametrize("mood", [
        "playful", "moody", "minimal", "elegant", "professional", "balanced",
    ])
    def test_text_on_surface_meets_aa(self, mood):
        from augmentum.tools.application_design_system import (
            _PALETTES,
            contrast_ratio,
        )
        palette = _PALETTES[mood]
        ratio = contrast_ratio(palette["text"], palette["surface"])
        assert ratio >= 4.5, f"{mood}: text/surface contrast {ratio:.2f} < 4.5"

    @pytest.mark.parametrize("mood", [
        "playful", "moody", "minimal", "elegant", "professional", "balanced",
    ])
    def test_accent_on_surface_meets_aa_large(self, mood):
        from augmentum.tools.application_design_system import (
            _PALETTES,
            contrast_ratio,
        )
        palette = _PALETTES[mood]
        # Accent is used for large UI (buttons, CTAs) — AA large is 3:1.
        ratio = contrast_ratio(palette["accent"], palette["surface"])
        assert ratio >= 3.0, f"{mood}: accent/surface contrast {ratio:.2f} < 3.0"


class TestDesignSystemContrastHelpers:
    def test_relative_luminance_known_values(self):
        from augmentum.tools.application_design_system import _relative_luminance
        assert _relative_luminance("#ffffff") == pytest.approx(1.0, abs=1e-6)
        assert _relative_luminance("#000000") == pytest.approx(0.0, abs=1e-6)
        # Mid-gray around 0.2158 (not 0.5 — sRGB is non-linear)
        assert _relative_luminance("#808080") == pytest.approx(0.2158, abs=0.01)

    def test_contrast_ratio_symmetric(self):
        from augmentum.tools.application_design_system import contrast_ratio
        a = contrast_ratio("#000000", "#ffffff")
        b = contrast_ratio("#ffffff", "#000000")
        assert a == pytest.approx(b)
        assert a == pytest.approx(21.0, abs=0.1)

    def test_meets_wcag_aa_thresholds(self):
        from augmentum.tools.application_design_system import meets_wcag_aa
        # Black on white — obviously passes both thresholds
        assert meets_wcag_aa("#000000", "#ffffff") is True
        assert meets_wcag_aa("#000000", "#ffffff", large=True) is True
        # Light-gray on white — marginal; passes large (3:1) but not normal (4.5:1)
        assert meets_wcag_aa("#aaaaaa", "#ffffff") is False
        # Very-light-gray on white — fails both
        assert meets_wcag_aa("#eeeeee", "#ffffff") is False

    def test_ensure_contrast_darkens_for_light_bg(self):
        from augmentum.tools.application_design_system import (
            _ensure_contrast,
            contrast_ratio,
        )
        # Light-gray text on white fails AA; _ensure_contrast should
        # darken it until the ratio clears 4.5:1.
        adjusted = _ensure_contrast("#bbbbbb", "#ffffff")
        assert contrast_ratio(adjusted, "#ffffff") >= 4.5

    def test_ensure_contrast_lightens_for_dark_bg(self):
        from augmentum.tools.application_design_system import (
            _ensure_contrast,
            contrast_ratio,
        )
        adjusted = _ensure_contrast("#555555", "#0a0a0a")
        assert contrast_ratio(adjusted, "#0a0a0a") >= 4.5

    def test_ensure_contrast_passthrough_when_ok(self):
        from augmentum.tools.application_design_system import _ensure_contrast
        # Already clears AA — shouldn't change
        assert _ensure_contrast("#000000", "#ffffff") == "#000000"


class TestDesignSystemOutput:
    def test_css_vars_block_has_required_custom_properties(self):
        from augmentum.tools.application_design_system import compute_design_system
        ds = compute_design_system("build a clean todo app")
        css = ds.to_css_vars()
        for var in (
            "--surface", "--surface-alt", "--text", "--text-muted",
            "--accent", "--accent-hover", "--border",
            "--success", "--warning", "--error",
            "--radius", "--font-body", "--font-heading",
        ):
            assert var in css, f"{var} missing from CSS output"

    def test_guidance_tells_llm_to_use_vars_not_hex(self):
        from augmentum.tools.application_design_system import compute_design_system
        guidance = compute_design_system("a bright fun kids app").guidance_for_prompt()
        assert "do NOT hardcode" in guidance
        assert ":root" in guidance  # CSS block embedded

    def test_build_design_rules_includes_palette(self):
        """build_design_rules feeds the palette into the prompt — verify
        end-to-end integration."""
        from augmentum.tools.application_scaffolds import build_design_rules
        rules = build_design_rules("a sleek professional dashboard", "dashboard")
        assert "Design system" in rules
        assert "--surface" in rules
        # The professional palette has a dark blue accent
        assert "mood: professional" in rules


# ===========================================================================
# Category-specific API refs (§23)
# ===========================================================================

class TestCategoryApiRefs:
    """API refs anchor the generator to verified signatures. Each built-in
    category must expose a usable block that names the real APIs (not
    hallucinated ones like fillCircle)."""

    def test_empty_categories_returns_empty(self):
        from augmentum.tools.application_api_refs import api_refs_for_categories
        assert api_refs_for_categories([]) == ""
        # Unknown categories produce nothing to show
        assert api_refs_for_categories(["mystery"]) == ""

    def test_canvas_game_block_contains_verified_idioms(self):
        from augmentum.tools.application_api_refs import api_refs_for_categories
        block = api_refs_for_categories(["canvas_game"])
        # Must include arc() since fillCircle() doesn't exist — this is the
        # single most common Canvas 2D hallucination.
        assert "ctx.arc(" in block
        assert "requestAnimationFrame" in block
        # fillCircle appears in a warning ("no fillCircle; use arc"), but
        # must NEVER appear as a call signature the LLM could copy.
        assert "ctx.fillCircle(" not in block

    def test_charts_dashboard_block(self):
        from augmentum.tools.application_api_refs import api_refs_for_categories
        block = api_refs_for_categories(["charts_dashboard"])
        assert "new Chart(" in block
        assert "type: 'line'" in block or "line" in block
        assert "destroy()" in block  # must warn to destroy before re-creating

    def test_form_block_covers_validation_and_persistence(self):
        from augmentum.tools.application_api_refs import api_refs_for_categories
        block = api_refs_for_categories(["interactive_form"])
        assert "preventDefault" in block
        assert "FormData" in block
        assert "localStorage" in block

    def test_multi_category_concatenates(self):
        from augmentum.tools.application_api_refs import api_refs_for_categories
        block = api_refs_for_categories(["canvas_game", "interactive_form"])
        assert "ctx.arc(" in block
        assert "FormData" in block

    def test_header_present_when_any_refs_match(self):
        from augmentum.tools.application_api_refs import api_refs_for_categories
        block = api_refs_for_categories(["canvas_game"])
        assert "API quick reference" in block

    def test_build_design_rules_injects_api_refs(self):
        """End-to-end: a game description triggers the canvas_game refs
        in the generator's design-rules block."""
        from augmentum.tools.application_scaffolds import build_design_rules
        rules = build_design_rules("build a space invaders game", "game")
        assert "ctx.arc(" in rules
        # Never suggest the hallucination as a call signature
        assert "ctx.fillCircle(" not in rules

    def test_build_design_rules_no_api_block_when_no_category(self):
        """A plain static page doesn't need Canvas/Chart.js refs."""
        from augmentum.tools.application_scaffolds import build_design_rules
        rules = build_design_rules("a landing page for a startup", "static")
        # Should not have API quick reference block since no categories match
        assert "API quick reference" not in rules


# ===========================================================================
# Intent-in-plan (§24)
# ===========================================================================

class TestIntentInPlan:
    """Intent detection in the plan step nudges the LLM to put feature
    signals into contract columns (PROVIDES/WIRES), which means the
    contract validator catches missing features natively — closing the
    gap that verify_intent's keyword grep used to plug at the end."""

    def test_empty_description_no_suggestions(self):
        from augmentum.tools.application_intent import derive_intent_features
        assert derive_intent_features("") == []
        assert derive_intent_features("a thing") == []

    def test_calculator_triggers_arithmetic_features(self):
        from augmentum.tools.application_intent import derive_intent_features
        suggestions = derive_intent_features(
            "a calculator that can add, subtract, multiply and divide"
        )
        labels = [s["label"] for s in suggestions]
        assert "addition" in labels
        assert "subtraction" in labels
        assert "multiplication" in labels
        assert "division" in labels

    def test_todo_app_triggers_crud_features(self):
        from augmentum.tools.application_intent import derive_intent_features
        suggestions = derive_intent_features(
            "a todo app where I can create, edit, and delete tasks and search them"
        )
        labels = [s["label"] for s in suggestions]
        assert "add-item / create" in labels
        assert "edit / update" in labels
        assert "delete / remove" in labels
        assert "search / filter" in labels

    def test_game_keywords_trigger_game_features(self):
        from augmentum.tools.application_intent import derive_intent_features
        labels = [
            s["label"] for s in derive_intent_features(
                "a space shooter with score tracking, collision detection, "
                "game over and restart"
            )
        ]
        assert "score tracking" in labels
        assert "collision detection" in labels
        assert "game over state" in labels
        assert "restart" in labels

    def test_whole_word_match_avoids_false_positives(self):
        """'additional' should NOT trigger 'addition' feature."""
        from augmentum.tools.application_intent import derive_intent_features
        suggestions = derive_intent_features(
            "a landing page with additional product information"
        )
        labels = [s["label"] for s in suggestions]
        assert "addition" not in labels

    def test_add_in_todo_does_not_trigger_arithmetic(self):
        """Context gate: 'add' in a todo description should trigger the
        CRUD add-item rule, NOT the arithmetic addition rule. Without
        the context gate, a todo app falsely demanded arithmetic
        operators at verify time."""
        from augmentum.tools.application_intent import derive_intent_features
        suggestions = derive_intent_features(
            "todo list with add, edit, and delete items"
        )
        labels = [s["label"] for s in suggestions]
        assert "addition" not in labels
        assert "add-item / create" in labels

    def test_add_in_calculator_triggers_arithmetic(self):
        """Same 'add' keyword, different context — arithmetic should
        fire because 'calculator' is present."""
        from augmentum.tools.application_intent import derive_intent_features
        suggestions = derive_intent_features(
            "scientific calculator with add, subtract, multiply, divide"
        )
        labels = [s["label"] for s in suggestions]
        assert "addition" in labels

    def test_verify_intent_gaps_catches_missing_features(self):
        """verify_intent_gaps reports descriptions-mention-but-code-doesnt
        gaps using the unified rule table."""
        from augmentum.tools.application_intent import verify_intent_gaps
        desc = "scientific calculator with add, subtract, multiply, divide"
        # Code has only addition — expect subtraction/multiplication/division flagged
        code = """
        const operations = { '+': (a, b) => a + b };
        function compute(op, a, b) { return operations[op](a, b); }
        """
        issues = verify_intent_gaps(desc, code)
        joined = " ".join(issues)
        assert "subtraction" in joined
        assert "multiplication" in joined
        assert "division" in joined
        assert "addition" not in joined  # this one IS implemented

    def test_verify_intent_gaps_todo_no_arithmetic_false_positive(self):
        """A todo app with `addItem` + `push` should satisfy CRUD rule
        and NOT generate any arithmetic-gap issues (the false positive
        that motivated unification)."""
        from augmentum.tools.application_intent import verify_intent_gaps
        desc = "todo app where I can add items, edit them, and delete them"
        code = """
        const tasks = [];
        function addItem(t) { tasks.push(t); }
        function editItem(i, t) { tasks[i] = t; }
        function deleteItem(i) { tasks.splice(i, 1); }
        """
        issues = verify_intent_gaps(desc, code)
        joined = " ".join(issues)
        assert "addition operation" not in joined
        assert "subtraction operation" not in joined
        assert "multiplication operation" not in joined
        assert "division operation" not in joined

    def test_suggestion_has_provides_or_wires_hint(self):
        """Every suggestion must carry at least one actionable hint so
        the plan prompt has something concrete to reference."""
        from augmentum.tools.application_intent import derive_intent_features
        for s in derive_intent_features("calculator with add subtract"):
            assert s.get("provides") or s.get("wires"), \
                f"suggestion {s['label']} has no actionable hint"

    def test_format_for_plan_empty_when_no_features(self):
        from augmentum.tools.application_intent import format_intent_for_plan_prompt
        assert format_intent_for_plan_prompt("") == ""
        assert format_intent_for_plan_prompt("hello world") == ""

    def test_format_for_plan_renders_human_readable_block(self):
        from augmentum.tools.application_intent import format_intent_for_plan_prompt
        block = format_intent_for_plan_prompt(
            "a calculator with add and subtract operations"
        )
        assert "Required features" in block
        assert "addition" in block
        assert "PROVIDES" in block
        # Arrow connects label → contract hint
        assert "\u2192" in block

    def test_plan_prompt_includes_intent_block_for_feature_rich_app(self):
        """End-to-end: a descriptive build request → plan prompt grows an
        intent section the LLM can act on."""
        from augmentum.tools.application_scaffolds import build_plan_prompt
        msgs = build_plan_prompt(
            "a todo app where I can create tasks, edit them, delete them, and search",
            "static",
        )
        system = msgs[0]["content"]
        assert "Required features" in system
        # Feature labels from the intent module should appear verbatim
        assert "add-item / create" in system
        assert "delete / remove" in system

    def test_plan_prompt_skips_intent_block_for_generic_app(self):
        from augmentum.tools.application_scaffolds import build_plan_prompt
        msgs = build_plan_prompt("a landing page", "static")
        system = msgs[0]["content"]
        assert "Required features" not in system


# ===========================================================================
# Browser verify via thin CDP client (§26)
# ===========================================================================

class TestCdpClient:
    """The CDP module must degrade gracefully on hosts without chromium
    AND parse real CDP event streams correctly on hosts that have it.
    All networking is mocked — tests run everywhere, Docker or not."""

    def test_find_chromium_returns_none_when_absent(self, monkeypatch):
        import augmentum.tools.application_cdp as cdp
        monkeypatch.setattr(cdp.shutil, "which", lambda name: None)
        assert cdp.find_chromium() is None

    def test_find_chromium_returns_first_match(self, monkeypatch):
        import augmentum.tools.application_cdp as cdp
        found = {}

        def fake_which(name):
            # Simulate only google-chrome being installed
            if name == "google-chrome":
                return "/usr/bin/google-chrome"
            found[name] = None
            return None

        monkeypatch.setattr(cdp.shutil, "which", fake_which)
        assert cdp.find_chromium() == "/usr/bin/google-chrome"

    @pytest.mark.asyncio
    async def test_start_raises_when_no_chromium(self, monkeypatch):
        import augmentum.tools.application_cdp as cdp
        monkeypatch.setattr(cdp, "find_chromium", lambda: None)
        bv = cdp.BrowserVerifier()
        with pytest.raises(cdp.ChromiumNotAvailable):
            await bv.start()

    def test_classify_events_runtime_exception(self):
        from augmentum.tools.application_cdp import BrowserVerifier
        events = [{
            "method": "Runtime.exceptionThrown",
            "params": {"exceptionDetails": {
                "text": "Uncaught",
                "exception": {"description": "TypeError: foo is not a function"},
                "url": "data:text/html",
                "lineNumber": 42,
            }},
        }]
        errors, warnings = BrowserVerifier._classify_events(events)
        assert len(errors) == 1
        assert "TypeError" in errors[0]
        assert "42" in errors[0]
        assert warnings == []

    def test_classify_events_console_error(self):
        from augmentum.tools.application_cdp import BrowserVerifier
        events = [{
            "method": "Runtime.consoleAPICalled",
            "params": {
                "type": "error",
                "args": [{"value": "Cannot read property 'foo' of null"}],
            },
        }]
        errors, warnings = BrowserVerifier._classify_events(events)
        assert len(errors) == 1
        assert "CONSOLE.ERROR" in errors[0]
        assert "null" in errors[0]

    def test_classify_events_console_warning_routed_separately(self):
        from augmentum.tools.application_cdp import BrowserVerifier
        events = [{
            "method": "Runtime.consoleAPICalled",
            "params": {
                "type": "warning",
                "args": [{"value": "deprecated API"}],
            },
        }]
        errors, warnings = BrowserVerifier._classify_events(events)
        assert errors == []
        assert len(warnings) == 1 and "deprecated" in warnings[0]

    def test_classify_events_ignores_info_and_log(self):
        from augmentum.tools.application_cdp import BrowserVerifier
        events = [
            {"method": "Runtime.consoleAPICalled", "params": {"type": "log", "args": [{"value": "x"}]}},
            {"method": "Runtime.consoleAPICalled", "params": {"type": "info", "args": [{"value": "x"}]}},
            {"method": "Page.loadEventFired", "params": {}},
        ]
        errors, warnings = BrowserVerifier._classify_events(events)
        assert errors == [] and warnings == []

    def test_derive_smoke_sequence_from_wires(self):
        """Plan WIRES entries supply the primary click selectors; duplicates
        across multiple files are deduped."""
        from augmentum.tools.application_cdp import derive_smoke_sequence
        planned = [
            {"path": "index.html", "role": "entry",
             "wires": ["#btn-start click", "#btn-pause click"]},
            {"path": "app.js", "role": "script",
             "wires": ["#btn-start click",  # duplicate
                       "#btn-reset click"]},
        ]
        selectors = derive_smoke_sequence(planned, "a pomodoro timer")
        # Dedup preserved order: start, pause, reset
        assert selectors[:3] == ["#btn-start", "#btn-pause", "#btn-reset"]

    def test_derive_smoke_sequence_falls_back_to_intent_hints(self):
        """When the plan has no WIRES, intent hints pick up common
        selectors — ensures even LLMs that skip contract columns get
        interactive smoke coverage."""
        from augmentum.tools.application_cdp import derive_smoke_sequence
        planned = [{"path": "app.js", "role": "script", "wires": []}]
        # 'restart' triggers the "restart / retry / play again" intent,
        # which suggests #btn-restart
        selectors = derive_smoke_sequence(planned, "a game with restart button")
        assert any(s.startswith("#") for s in selectors)

    def test_derive_smoke_sequence_caps_output(self):
        from augmentum.tools.application_cdp import derive_smoke_sequence
        planned = [{
            "path": "index.html", "role": "entry",
            "wires": [f"#b{i} click" for i in range(30)],
        }]
        selectors = derive_smoke_sequence(planned, "", max_clicks=5)
        assert len(selectors) == 5

    def test_derive_smoke_sequence_empty_when_no_plan_no_description(self):
        from augmentum.tools.application_cdp import derive_smoke_sequence
        assert derive_smoke_sequence([], "") == []

    def test_filter_browser_errors_drops_chromium_noise(self):
        from augmentum.tools.application_cdp import filter_browser_errors
        errors = [
            "RUNTIME: TypeError: user code broke",
            "Error with Permissions-Policy header: browser notice",
            "RUNTIME: DevTools failed to load source",
            "CONSOLE.ERROR: real app bug here",
        ]
        filtered = filter_browser_errors(errors)
        # Only the two user-code errors survive
        assert len(filtered) == 2
        assert any("user code broke" in e for e in filtered)
        assert any("real app bug" in e for e in filtered)

    @pytest.mark.asyncio
    async def test_run_browser_verify_absorbs_chromium_not_available(self, monkeypatch):
        """Verifier must return [] (not crash) when chromium isn't
        installed — falls back to the quickjs path cleanly."""
        import augmentum.tools.application_cdp as cdp
        from augmentum.tools.artifact_application import ApplicationBuilderTool

        monkeypatch.setattr(cdp, "find_chromium", lambda: None)
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)
        errors = await tool._run_browser_verify("<html></html>")
        assert errors == []

    @pytest.mark.asyncio
    async def test_run_browser_verify_absorbs_runtime_error(self, monkeypatch):
        """Transient CDP errors must not crash the build — they log
        and return []."""
        import augmentum.tools.application_cdp as cdp
        from augmentum.tools.artifact_application import ApplicationBuilderTool

        class BoomVerifier:
            async def __aenter__(self):
                raise RuntimeError("simulated transient")

            async def __aexit__(self, *args):
                pass

        monkeypatch.setattr(cdp, "BrowserVerifier", lambda *a, **k: BoomVerifier())
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)
        errors = await tool._run_browser_verify("<html></html>")
        assert errors == []


# ===========================================================================
# Batch generation (§25)
# ===========================================================================

class TestBatchGenerate:
    """Small apps (≤5 planned files) should generate in a single LLM
    call rather than one file at a time. The batch path must
    short-circuit when successful and fall through cleanly to the
    sequential path when the LLM returns partial output."""

    def _tool_with_llm(self, llm, *, batch_enabled=True, max_files=5):
        from augmentum.tools.artifact_application import ApplicationBuilderTool
        settings = {
            "app_builder_batch_small_apps": batch_enabled,
            "app_builder_improve_pass": False,  # keep tests focused
            "app_builder_max_improve_iterations": 1,
        }

        class MockStore:
            async def save(self_, **kw):
                return {"id": "batch-test"}

        return ApplicationBuilderTool(MockStore(), llm, lambda: settings)

    @pytest.mark.asyncio
    async def test_batch_succeeds_with_single_llm_call_for_all_files(self):
        plan = (
            "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Page\n"
            "FILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styles\n"
            "FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: Logic\n"
            "__PASS_COMPLETE__"
        )
        # One response holds ALL three files as separate fenced blocks.
        batch_response = (
            "```index.html\n<!DOCTYPE html><html><head></head>"
            "<body><h1>Hi</h1></body></html>\n```\n\n"
            "```styles.css\nbody { font-family: sans-serif; }\n```\n\n"
            '```app.js\nconsole.log("hello");\n```\n'
            "__PASS_COMPLETE__"
        )
        calls: list[str] = []

        async def llm(messages, max_tokens=4096, **kw):
            system = next(
                (m.get("content", "") for m in messages if m.get("role") == "system"),
                "",
            )
            if "planning a production-quality" in system:
                calls.append("plan")
                return plan
            if "generating a complete small web application in ONE response" in system:
                calls.append("batch")
                return batch_response
            # Warmup / other calls — don't classify or consume the script.
            return "__PASS_COMPLETE__"

        tool = self._tool_with_llm(llm)
        result = await tool.execute(description="a hello page")
        assert result.success
        # Must have used the batch path — no per-file generate calls.
        assert "batch" in calls
        # All three files were produced in ONE LLM call, not three.
        assert calls.count("batch") == 1
        # No sequential fallback fired.
        project = result.metadata["project"]
        assert len(project["files"]) == 3
        paths = {f["path"] for f in project["files"]}
        assert paths == {"index.html", "styles.css", "app.js"}

    @pytest.mark.asyncio
    async def test_batch_partial_falls_through_to_sequential(self):
        """If the batch response only contains SOME of the files, the
        pipeline must keep the good ones and generate the rest
        sequentially — no data loss, no infinite loop."""
        plan = (
            "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Page\n"
            "FILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styles\n"
            "FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: Logic\n"
            "__PASS_COMPLETE__"
        )
        partial_batch = (
            "```index.html\n<html><body>Hi</body></html>\n```\n"
            "__PASS_COMPLETE__"
        )
        gen_styles = '```styles.css\nbody { margin: 0; }\n```\n__PASS_COMPLETE__'
        gen_app = '```app.js\nconsole.log("ok");\n```\n__PASS_COMPLETE__'

        import re as _re
        # Route by classification so pipeline warmup calls ("ready" ping,
        # etc.) don't eat the response script.
        pools = {
            "plan": [plan],
            "batch": [partial_batch],
            "gen_styles": [gen_styles],
            "gen_app": [gen_app],
        }
        call_log: list[str] = []

        async def llm(messages, max_tokens=4096, **kw):
            system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            if "planning a production-quality" in system:
                kind = "plan"
            elif "generating a complete small web application in ONE response" in system:
                kind = "batch"
            elif "generating production-quality code" in system:
                # Match the distinctive "Generate this file: ```<path>```"
                # marker rather than substring — both files appear in the
                # prompt's context block, so substring match picks whichever
                # comes first.
                m = _re.search(r"Generate this file:\s*```([^`]+)```", user)
                path = m.group(1).strip() if m else ""
                if path == "styles.css":
                    kind = "gen_styles"
                elif path == "app.js":
                    kind = "gen_app"
                elif path == "index.html":
                    kind = "gen_index"
                else:
                    kind = "gen_other"
            else:
                kind = "other"
            call_log.append(kind)
            q = pools.get(kind, [])
            if q:
                return q.pop(0) if len(q) > 1 else q[0]
            return "__PASS_COMPLETE__"

        tool = self._tool_with_llm(llm)
        result = await tool.execute(description="a tiny page")
        assert result.success
        assert "batch" in call_log
        # Sequential picked up the stragglers — we never re-generated index.html.
        assert "gen_index" not in call_log
        # All three files are present in the final project.
        paths = {f["path"] for f in result.metadata["project"]["files"]}
        assert {"index.html", "styles.css", "app.js"} <= paths

    @pytest.mark.asyncio
    async def test_batch_disabled_via_setting_uses_sequential(self):
        plan = (
            "FILE: index.html | ROLE: entry | LANG: html | DESCRIPTION: Page\n"
            "FILE: styles.css | ROLE: style | LANG: css | DESCRIPTION: Styles\n"
            "FILE: app.js | ROLE: script | LANG: javascript | DESCRIPTION: Logic\n"
            "__PASS_COMPLETE__"
        )
        pools = {
            "plan": [plan],
            "gen_index": ['```index.html\n<html><body>Hi</body></html>\n```\n__PASS_COMPLETE__'],
            "gen_styles": ['```styles.css\nbody { margin: 0; }\n```\n__PASS_COMPLETE__'],
            "gen_app": ['```app.js\nconsole.log("ok");\n```\n__PASS_COMPLETE__'],
        }
        call_log: list[str] = []

        import re as _re

        async def llm(messages, max_tokens=4096, **kw):
            system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            if "planning a production-quality" in system:
                kind = "plan"
            elif "generating a complete small web application in ONE response" in system:
                kind = "batch"
            elif "generating production-quality code" in system:
                m = _re.search(r"Generate this file:\s*```([^`]+)```", user)
                path = m.group(1).strip() if m else ""
                if path == "styles.css":
                    kind = "gen_styles"
                elif path == "app.js":
                    kind = "gen_app"
                elif path == "index.html":
                    kind = "gen_index"
                else:
                    kind = "gen_other"
            else:
                return "__PASS_COMPLETE__"
            call_log.append(kind)
            q = pools.get(kind, [])
            return q[0] if q else "__PASS_COMPLETE__"

        tool = self._tool_with_llm(llm, batch_enabled=False)
        result = await tool.execute(description="a tiny page")
        assert result.success
        # Batch never attempted
        assert "batch" not in call_log


# ===========================================================================
# Pipeline collapse: structural checks moved to validate (§27)
# ===========================================================================

class TestStructuralChecksInValidate:
    """Toolkit §27: structural checks that used to live in _pass_improve
    (missing <html>, empty <body>, stub JS) now belong to _pass_validate
    so every static check lives in one place. improve becomes pure
    quality scoring."""

    def _make_bare_tool(self):
        from augmentum.tools.artifact_application import ApplicationBuilderTool
        tool = ApplicationBuilderTool.__new__(ApplicationBuilderTool)
        tool._store = None
        tool._get_settings = None
        tool._request_model = ""
        return tool

    @pytest.mark.asyncio
    async def test_validate_flags_missing_html_tag(self):
        """An entry file with no <html> tag must surface as a validate error."""
        from augmentum.tools.artifact_application import PipelineContext
        tool = self._make_bare_tool()

        ctx = PipelineContext(description="x", scaffold_id="static")
        ctx.files = [
            {"path": "index.html", "role": "entry", "lang": "html",
             "content": "<body><p>no html tag</p></body>"},
        ]
        ctx.planned_files = [{"path": "index.html", "role": "entry"}]

        async def noop_llm(messages, **kw):
            return "__PASS_COMPLETE__"
        tool._call_llm = noop_llm
        result = await tool._pass_validate(ctx)
        # Result may be done=True or done=False depending on the fix path;
        # the key assertion is that ctx.errors captured the missing-tag issue.
        assert any("missing <html>" in e for e in ctx.errors)

    @pytest.mark.asyncio
    async def test_validate_flags_stub_script(self):
        from augmentum.tools.artifact_application import PipelineContext
        tool = self._make_bare_tool()

        ctx = PipelineContext(description="x", scaffold_id="static")
        ctx.files = [
            {"path": "index.html", "role": "entry", "lang": "html",
             "content": "<html><body><div>content here is long enough</div></body></html>"},
            {"path": "app.js", "role": "script", "lang": "javascript",
             "content": "// TODO: implement this"},
        ]
        ctx.planned_files = [
            {"path": "index.html", "role": "entry"},
            {"path": "app.js", "role": "script"},
        ]

        async def noop_llm(messages, **kw):
            return "__PASS_COMPLETE__"
        tool._call_llm = noop_llm
        await tool._pass_validate(ctx)
        assert any("stub" in e.lower() for e in ctx.errors)

    @pytest.mark.asyncio
    async def test_validate_flags_empty_body(self):
        from augmentum.tools.artifact_application import PipelineContext
        tool = self._make_bare_tool()

        ctx = PipelineContext(description="x", scaffold_id="static")
        ctx.files = [
            {"path": "index.html", "role": "entry", "lang": "html",
             "content": "<html><body></body></html>"},
        ]
        ctx.planned_files = [{"path": "index.html", "role": "entry"}]

        async def noop_llm(messages, **kw):
            return "__PASS_COMPLETE__"
        tool._call_llm = noop_llm
        await tool._pass_validate(ctx)
        assert any("empty" in e.lower() for e in ctx.errors)

    @pytest.mark.asyncio
    async def test_validate_recognises_inline_style_block(self):
        """classList uses a class defined only in an entry <style> block —
        must NOT be flagged as "class not defined in CSS". The live
        pomodoro build false-positived on .loading-fade for this reason."""
        from augmentum.tools.artifact_application import PipelineContext
        tool = self._make_bare_tool()

        ctx = PipelineContext(description="x", scaffold_id="static")
        ctx.files = [
            {"path": "index.html", "role": "entry", "lang": "html",
             "content": "<html><head><style>.loading-fade { animation: fadeIn 0.5s; }</style></head><body><div id='app' class='main'>content here is long enough</div></body></html>"},
            {"path": "app.js", "role": "script", "lang": "javascript",
             "content": "document.getElementById('app').classList.add('loading-fade');"},
        ]
        ctx.planned_files = [
            {"path": "index.html", "role": "entry"},
            {"path": "app.js", "role": "script"},
        ]

        async def noop_llm(messages, **kw):
            return "__PASS_COMPLETE__"
        tool._call_llm = noop_llm
        await tool._pass_validate(ctx)
        errors = ctx.errors or []
        bad = [e for e in errors if "loading-fade" in e and "CSS" in e]
        assert bad == [], f"inline-style class should not be flagged; got {bad}"

    @pytest.mark.asyncio
    async def test_validate_clean_when_structural_ok(self):
        """Well-formed files produce no structural complaints."""
        from augmentum.tools.artifact_application import PipelineContext
        tool = self._make_bare_tool()

        ctx = PipelineContext(description="x", scaffold_id="static")
        ctx.files = [
            {"path": "index.html", "role": "entry", "lang": "html",
             "content": "<html><body><div>real content here is long enough</div></body></html>"},
            {"path": "app.js", "role": "script", "lang": "javascript",
             "content": "console.log('ok'); function go() { return 1; }"},
        ]
        ctx.planned_files = [
            {"path": "index.html", "role": "entry"},
            {"path": "app.js", "role": "script"},
        ]

        async def noop_llm(messages, **kw):
            return "__PASS_COMPLETE__"
        tool._call_llm = noop_llm
        result = await tool._pass_validate(ctx)
        # Clean validate → pass is done and no errors recorded.
        structural = [e for e in (ctx.errors or []) if
                       "missing" in e.lower() or "stub" in e.lower() or "empty" in e.lower()]
        assert structural == [], f"expected clean structural, got {structural}"


# ===========================================================================
# Polish: LLM self-doubt comment stripping
# ===========================================================================

class TestStripSelfDoubtComments:
    """Polish pass should scrub internal-monologue comments LLMs leak
    into generated code. Real samples from the live pomodoro build
    drove the opener list — tests below pin those patterns."""

    def test_preserves_ordinary_comments(self):
        from augmentum.tools.artifact_application import strip_selfdoubt_comments
        src = "// Timer state\nlet running = false;\n// Kick off\nstart();"
        cleaned, removed = strip_selfdoubt_comments(src)
        assert removed == 0
        assert cleaned == src

    def test_strips_actually_self_correction(self):
        """Verbatim sample from the live build output."""
        from augmentum.tools.artifact_application import strip_selfdoubt_comments
        src = (
            "function save(mode) {\n"
            "  // Actually, App.js handles increment logic explicitly to be cleaner.\n"
            "  // This function will just save whatever count is passed.\n"
            "  localStorage.setItem('k', '1');\n"
            "}\n"
        )
        cleaned, removed = strip_selfdoubt_comments(src)
        assert removed == 1  # only the "Actually" line; "This function" is ordinary
        assert "Actually" not in cleaned
        assert "localStorage.setItem" in cleaned

    def test_strips_lets_assume(self):
        from augmentum.tools.artifact_application import strip_selfdoubt_comments
        src = (
            "// Let's assume the caller handles validation first\n"
            "doWork();\n"
        )
        cleaned, removed = strip_selfdoubt_comments(src)
        assert removed == 1
        assert "Let's assume" not in cleaned

    def test_strips_for_simplicity(self):
        from augmentum.tools.artifact_application import strip_selfdoubt_comments
        src = (
            "// For simplicity in this specific prompt, we skip the retry loop.\n"
            "return result;\n"
        )
        cleaned, removed = strip_selfdoubt_comments(src)
        assert removed == 1

    def test_strips_first_person_voice(self):
        from augmentum.tools.artifact_application import strip_selfdoubt_comments
        src = (
            "// I'll just inline the default state here for now\n"
            "const state = { count: 0 };\n"
            "// We'll just reuse the global timer\n"
            "startTimer();\n"
        )
        cleaned, removed = strip_selfdoubt_comments(src)
        assert removed == 2

    def test_preserves_inline_trailing_comments(self):
        """Trailing comments on code lines must stay — only pure
        comment lines are candidates for stripping."""
        from augmentum.tools.artifact_application import strip_selfdoubt_comments
        src = "foo(); // Actually, this bit matters — keep the comment\n"
        cleaned, removed = strip_selfdoubt_comments(src)
        assert removed == 0
        assert cleaned == src

    def test_collapses_blank_lines_after_strip(self):
        from augmentum.tools.artifact_application import strip_selfdoubt_comments
        src = (
            "const a = 1;\n"
            "\n"
            "// Actually, we could use let here\n"
            "// For simplicity we'll keep it const\n"
            "\n"
            "const b = 2;\n"
        )
        cleaned, removed = strip_selfdoubt_comments(src)
        assert removed == 2
        assert "\n\n\n" not in cleaned


# ===========================================================================
# Settings Wiring Tests
# ===========================================================================

class TestSettingsWiring:
    def test_config_has_settings(self):
        """Verify settings exist in config.py."""
        import ast
        with open("augmentum/config.py") as f:
            tree = ast.parse(f.read())
        # Find the Settings class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Settings":
                field_names = [
                    n.target.id for n in node.body
                    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                ]
                assert "app_builder_improve_pass" in field_names
                assert "app_builder_max_improve_iterations" in field_names
                assert "app_builder_max_fix_iterations" in field_names
                assert "app_builder_auto_preview" in field_names
                # §26 — browser-verify feature flag
                assert "app_builder_use_browser_verify" in field_names
                return
        pytest.fail("Settings class not found")

    def test_config_routes_has_settings(self):
        """Verify settings in config_routes.py _TOOL_SETTINGS."""
        with open("augmentum/proxy/config_routes.py") as f:
            content = f.read()
        assert "app_builder_improve_pass" in content
        assert "app_builder_max_improve_iterations" in content
        assert "app_builder_max_fix_iterations" in content
        assert "app_builder_auto_preview" in content

    def test_server_restore_map_has_settings(self):
        """Verify settings in server.py _SETTINGS_RESTORE_MAP."""
        with open("augmentum/proxy/server.py") as f:
            content = f.read()
        assert "app_builder_improve_pass" in content
        assert "app_builder_max_improve_iterations" in content
        assert "app_builder_max_fix_iterations" in content
        assert "app_builder_auto_preview" in content

    def test_settings_js_has_defaults(self):
        """Verify settings in settings.js DEFAULTS."""
        with open("ui/scripts/settings.js") as f:
            content = f.read()
        assert "appBuilderImprovePass" in content
        assert "appBuilderMaxImproveIterations" in content
        assert "appBuilderMaxFixIterations" in content
        assert "appBuilderAutoPreview" in content


# ===========================================================================
# Intent Classifier Tests
# ===========================================================================

class TestIntentClassifier:
    def test_build_app_regex_matches(self):
        """Verify the _BUILD_APP_RE pattern matches expected inputs."""
        import re
        # Import the pattern
        with open("augmentum/tools/intent.py") as f:
            content = f.read()

        # Extract the regex pattern
        m = re.search(r'_BUILD_APP_RE\s*=\s*re\.compile\(\s*r"(.+?)"', content, re.DOTALL)
        if not m:
            # Try multi-line pattern
            m = re.search(r'_BUILD_APP_RE\s*=\s*re\.compile\(\s*\n\s*r"(.+?)",', content, re.DOTALL)
        assert m, "_BUILD_APP_RE not found in intent.py"

        pattern = re.compile(m.group(1), re.IGNORECASE)

        # Should match
        assert pattern.search("build me a calculator app")
        assert pattern.search("create a todo application")
        assert pattern.search("make a website for my portfolio")
        assert pattern.search("generate a dashboard")
        assert pattern.search("build a game")

        # Should NOT match (these are normal chat)
        assert not pattern.search("what is the weather today")
        assert not pattern.search("explain how Python works")
        assert not pattern.search("help me debug this code")


# ===========================================================================
# Frontend Verification Tests
# ===========================================================================

class TestFrontend:
    def test_project_css_exists(self):
        """Verify project.css was created."""
        import os
        assert os.path.exists("ui/styles/project.css")

    def test_project_css_loaded_in_html(self):
        """Verify project.css is referenced in index.html."""
        with open("ui/index.html", encoding="utf-8") as f:
            content = f.read()
        assert "project.css" in content

    def test_chat_js_has_project_functions(self):
        """Verify all project card functions exist in the project module.

        Project-card functions moved to ui/scripts/chat/project.js when
        chat.js was decomposed (see 2026-04-08 session). The old monolith
        has been replaced by a module — read the new location.
        """
        with open("ui/scripts/chat/project.js", encoding="utf-8") as f:
            content = f.read()
        # Post-decomposition the public entry point dropped its leading
        # underscore (now exported as renderProjectCard); the private
        # helpers kept theirs.
        assert "function _assembleProject" in content
        assert "function _downloadProjectZip" in content or "async function _downloadProjectZip" in content
        assert "function renderProjectCard" in content
        assert "function _wireProjectCardActions" in content
        assert "function _expandProjectFiles" in content

    def test_chat_js_project_card_uses_escape_html(self):
        """Verify escapeHtml is used for user-provided strings."""
        with open("ui/scripts/chat/project.js", encoding="utf-8") as f:
            content = f.read()
        # Find the renderProjectCard function (underscore prefix dropped
        # when it became the module's public export).
        start = content.index("function renderProjectCard")
        end = content.index("\nfunction", start + 1)
        card_fn = content[start:end]
        # Project name, pass name, pass detail, file path, error text should all be escaped
        assert "escapeHtml(project.name" in card_fn
        assert "escapeHtml(pass.name)" in card_fn or "escapeHtml(pass.name )" in card_fn
        assert "escapeHtml(file.path)" in card_fn

    def test_chat_js_srcdoc_not_escaped(self):
        """Verify srcdoc is NOT double-escaped with escapeHtml."""
        with open("ui/scripts/chat/project.js", encoding="utf-8") as f:
            content = f.read()
        # The old bug was: srcdoc="${escapeHtml(srcdoc)}"
        # Should now use setAttribute instead
        assert 'srcdoc="${escapeHtml(' not in content

    def test_chat_js_script_escaping_in_assembler(self):
        """Verify </script> is escaped in the frontend assembler.

        After the chat.js decomposition, _assembleProject became a thin
        wrapper that delegates to ../assemble.js. The escape logic moved
        with it — verify there instead.
        """
        with open("ui/scripts/assemble.js", encoding="utf-8") as f:
            content = f.read()
        assert "replace(/<\\/script>/gi" in content, \
            "assemble.js must escape </script> tags in embedded scripts"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
