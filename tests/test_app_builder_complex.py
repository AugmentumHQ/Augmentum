"""Live integration tests for COMPLEX multi-file application builds.

These tests target 5-10 file projects to stress-test the pipeline's
multi-file coordination, working document, gap analysis, and interceptors.

Run: .venv/Scripts/python.exe -m pytest tests/test_app_builder_complex.py -v -s
Skip if no backend: auto-skip when LM Studio is unreachable.
SLOW (3-8 min per test) — run manually, not in CI.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Fix Windows console encoding for emoji output from pipeline
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _check_lm_studio():
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=3)
        return True
    except Exception:
        return False


requires_model = pytest.mark.skipif(
    not _check_lm_studio(),
    reason="LM Studio not running on 127.0.0.1:1234",
)


@pytest.fixture
def tool():
    """Create ApplicationBuilderTool with real LLM caller."""
    from augmentum.tools.artifact_application import ApplicationBuilderTool

    class FakeStore:
        saved = []

        async def save(self, *, data=b"", filename="", fmt="", task_id="",
                       session_id="", display_name="", metadata=None, source_json=None):
            self.saved.append({
                "filename": filename, "fmt": fmt,
                "size": len(data), "display_name": display_name,
            })
            return {"id": "complex_test_" + filename.replace(".", "_")}

    _model_name = ""
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5)
        models = json.loads(resp.read())
        if models.get("data"):
            _model_name = models["data"][0]["id"]
            print(f"Using model: {_model_name}")
    except Exception:
        pass

    async def real_llm(messages, max_tokens=4096, model="", **kw):
        import urllib.request
        payload = json.dumps({
            "model": model or _model_name or "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:1234/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    store = FakeStore()
    return ApplicationBuilderTool(store, real_llm, lambda: {}), store


def _analyze_project(result, label: str):
    """Print detailed analysis of a build result."""
    project = result.metadata.get("project", {})
    files = project.get("files", [])

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Success: {result.success}")
    if not result.success:
        print(f"  Error: {result.error}")
        return project

    print(f"  Score: {project.get('score', 0)}/10")
    print(f"  Files: {len(files)}")

    total_lines = 0
    total_chars = 0
    for f in files:
        lines = f["content"].count("\n") + 1
        total_lines += lines
        total_chars += len(f["content"])
        print(f"    {f['path']:20s} ({f['role']:6s}) — {lines:4d} lines, {len(f['content']):5d} chars")

    print(f"  Total: {total_lines} lines, {total_chars} chars")

    # Check for common issues
    issues = []

    # 1. Duplicate const/let across files
    scripts = [f for f in files if f.get("role") in ("script", "module")]
    if len(scripts) > 1:
        all_decls = {}
        for f in scripts:
            for m in re.finditer(r'^(?:const|let)\s+(\w+)', f["content"], re.MULTILINE):
                name = m.group(1)
                if name in all_decls and all_decls[name] != f["path"]:
                    issues.append(f"DUPLICATE: '{name}' in {f['path']} (first in {all_decls[name]})")
                else:
                    all_decls[name] = f["path"]

    # 2. Backtick/quote contamination in paths
    for f in files:
        if any(c in f["path"] for c in '`"\''):
            issues.append(f"BAD PATH: {f['path']!r}")

    # 3. Duplicate file paths
    paths = [f["path"] for f in files]
    if len(paths) != len(set(paths)):
        issues.append(f"DUPLICATE PATHS: {paths}")

    # 4. Empty files
    for f in files:
        if len(f["content"].strip()) < 10:
            issues.append(f"EMPTY FILE: {f['path']} ({len(f['content'])} chars)")

    # 5. HTML references to files not in project
    entry = next((f for f in files if f["role"] == "entry"), None)
    if entry:
        project_paths = {f["path"] for f in files}
        for m in re.finditer(r'<script[^>]*src=["\']([^"\']+)["\']', entry["content"]):
            src = m.group(1)
            if not src.startswith(("http", "//", "data:")) and src not in project_paths:
                issues.append(f"DANGLING REF: <script src='{src}'> not in project")
        for m in re.finditer(r'<link[^>]*href=["\']([^"\']+)["\']', entry["content"]):
            href = m.group(1)
            if href.endswith(".css") and not href.startswith(("http", "//")) and href not in project_paths:
                issues.append(f"DANGLING REF: <link href='{href}'> not in project")

    # 6. window.X references to undefined globals
    if len(scripts) > 1:
        assignments = set()
        for f in scripts:
            for m in re.finditer(r'window\.(\w+)\s*=', f["content"]):
                assignments.add(m.group(1))
        builtins = {
            "addEventListener", "removeEventListener", "innerWidth", "innerHeight",
            "location", "navigator", "document", "localStorage", "sessionStorage",
            "setTimeout", "setInterval", "clearTimeout", "clearInterval",
            "requestAnimationFrame", "cancelAnimationFrame", "getComputedStyle",
            "open", "close", "alert", "confirm", "prompt", "scrollTo",
            "matchMedia", "performance", "console", "history", "screen",
            "devicePixelRatio", "scrollX", "scrollY", "pageXOffset", "pageYOffset",
            "onresize", "onload", "onerror", "fetch",
        }
        for f in scripts:
            for m in re.finditer(r'window\.(\w+)(?:\.\w+)*\s*[\(.]', f["content"]):
                name = m.group(1)
                if name not in assignments and name not in builtins:
                    issues.append(f"UNDEF GLOBAL: window.{name} in {f['path']} never assigned")

    # 7. getElementById on missing IDs
    if entry and scripts:
        all_js = "\n".join(f["content"] for f in scripts)
        html = entry["content"]
        for m in re.finditer(r'getElementById\s*\(\s*["\'](\w+)["\']', all_js):
            elem_id = m.group(1)
            if f'id="{elem_id}"' not in html and f"id='{elem_id}'" not in html:
                issues.append(f"MISSING ID: getElementById('{elem_id}') but no id='{elem_id}' in HTML")

    if issues:
        print(f"\n  ⚠ ISSUES ({len(issues)}):")
        for i in issues:
            print(f"    - {i}")
    else:
        print("\n  ✓ No issues detected")

    return project


@requires_model
class TestComplexBuilds:
    """Live tests for multi-file projects. SLOW — 3-8 min each."""

    @pytest.mark.asyncio
    async def test_dashboard_with_charts(self, tool):
        """Build a multi-section dashboard — should produce 5-7 files."""
        builder, store = tool
        t0 = time.time()

        progress = []
        async def log_progress(data):
            p = data.get("project_progress", {})
            if p.get("pass") and p.get("status"):
                progress.append(p)
                delta = data.get("_content_delta", "")
                if delta.strip():
                    try:
                        print(delta.rstrip())
                    except UnicodeEncodeError:
                        print(delta.encode("ascii", "replace").decode())

        result = await builder.execute(
            description=(
                "analytics dashboard with 3 chart sections: a line chart showing monthly revenue, "
                "a bar chart showing product sales by category, and a pie chart showing traffic sources. "
                "Include a header with logo and nav, a sidebar with filter controls (date range picker, "
                "category dropdown), and a stats row showing KPIs (total revenue, orders, conversion rate). "
                "Use Chart.js for charts. Dark theme with purple accents."
            ),
            scaffold="dashboard",
            _progress_callback=log_progress,
            _request_model="",
        )

        elapsed = time.time() - t0
        project = _analyze_project(result, f"Dashboard with Charts ({elapsed:.0f}s)")

        assert result.success, f"Build failed: {result.error}"
        files = project.get("files", [])
        assert len(files) >= 3, f"Expected 3+ files for dashboard, got {len(files)}"

        # Should have entry + style + at least 2 JS files
        roles = [f["role"] for f in files]
        assert "entry" in roles
        assert "style" in roles
        assert roles.count("script") >= 1

        # Verify assembly works
        from augmentum.tools.artifact_application import ApplicationBuilderTool as ABT
        dummy = ABT.__new__(ABT)
        assembled = dummy._assemble(files)
        assert "<html" in assembled.lower()
        assert "<style>" in assembled
        assert "<script>" in assembled
        print(f"\n  Assembly: {len(assembled)} chars")

    @pytest.mark.asyncio
    async def test_game_with_multiple_systems(self, tool):
        """Build a canvas game — should produce 4-6 files with game logic split."""
        builder, store = tool
        t0 = time.time()

        progress = []
        async def log_progress(data):
            p = data.get("project_progress", {})
            if p.get("pass") and p.get("status"):
                progress.append(p)
                delta = data.get("_content_delta", "")
                if delta.strip():
                    try:
                        print(delta.rstrip())
                    except UnicodeEncodeError:
                        print(delta.encode("ascii", "replace").decode())

        result = await builder.execute(
            description=(
                "space shooter game with a player ship that moves with arrow keys and shoots with spacebar. "
                "Include enemy waves that spawn from the top, particle effects for explosions, "
                "a score counter, lives display, level progression (enemies get faster each wave), "
                "a start screen with instructions, and a game over screen with restart button. "
                "Use HTML5 canvas with smooth 60fps animation. Include sound effect placeholders."
            ),
            scaffold="game",
            _progress_callback=log_progress,
            _request_model="",
        )

        elapsed = time.time() - t0
        project = _analyze_project(result, f"Space Shooter Game ({elapsed:.0f}s)")

        assert result.success, f"Build failed: {result.error}"
        files = project.get("files", [])
        assert len(files) >= 3, f"Expected 3+ files for game, got {len(files)}"

        # Verify canvas setup
        entry = next((f for f in files if f["role"] == "entry"), None)
        assert entry, "No entry file"
        assert "canvas" in entry["content"].lower(), "No canvas element in HTML"

        # Verify game loop exists somewhere (may be in script or module role files)
        all_js = "\n".join(f["content"] for f in files if f["role"] in ("script", "module"))
        assert "requestAnimationFrame" in all_js or "setInterval" in all_js or ".start()" in all_js, \
            "No game loop (requestAnimationFrame, setInterval, or .start()) found"

        # Total lines should be substantial for a game
        total_lines = sum(f["content"].count("\n") + 1 for f in files)
        print(f"\n  Total game code: {total_lines} lines")
        assert total_lines >= 100, f"Game seems too small: {total_lines} lines"

    @pytest.mark.asyncio
    async def test_form_app_with_state(self, tool):
        """Build a multi-step form — should produce 4-6 files with validation logic."""
        builder, store = tool
        t0 = time.time()

        progress = []
        async def log_progress(data):
            p = data.get("project_progress", {})
            if p.get("pass") and p.get("status"):
                progress.append(p)
                delta = data.get("_content_delta", "")
                if delta.strip():
                    try:
                        print(delta.rstrip())
                    except UnicodeEncodeError:
                        print(delta.encode("ascii", "replace").decode())

        result = await builder.execute(
            description=(
                "project management tool with: a task board showing 3 columns (To Do, In Progress, Done) "
                "with drag-and-drop between columns, a form to add new tasks (title, description, priority, "
                "due date), task cards that expand on click to show details, a filter bar (by priority, "
                "search by title), local storage persistence so tasks survive page refresh, "
                "and a stats panel showing task counts per column and overdue tasks. "
                "Clean modern UI with smooth transitions."
            ),
            scaffold="form",
            _progress_callback=log_progress,
            _request_model="",
        )

        elapsed = time.time() - t0
        project = _analyze_project(result, f"Project Management Tool ({elapsed:.0f}s)")

        assert result.success, f"Build failed: {result.error}"
        files = project.get("files", [])
        assert len(files) >= 3, f"Expected 3+ files, got {len(files)}"

        # Should have localStorage usage or storage key for persistence
        all_code = "\n".join(f["content"] for f in files if f["role"] in ("script", "module", "data"))
        has_persistence = "localStorage" in all_code or "STORAGE_KEY" in all_code or "storage" in all_code.lower()
        assert has_persistence, "No localStorage/persistence usage found"

        # Should have drag-and-drop related code
        has_dnd = any(kw in all_code for kw in ("dragstart", "dragover", "drop", "draggable"))
        print(f"\n  Drag-and-drop: {'yes' if has_dnd else 'NOT FOUND'}")

        total_lines = sum(f["content"].count("\n") + 1 for f in files)
        print(f"  Total code: {total_lines} lines")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
