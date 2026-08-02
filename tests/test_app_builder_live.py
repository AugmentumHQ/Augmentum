"""Live integration tests for the Application Builder pipeline.

These tests call the REAL LLM backend — they require a running model server.
Run: .venv/Scripts/python.exe -m pytest tests/test_app_builder_live.py -v -s

Skip if no backend available: tests auto-skip when the server is unreachable.
These are SLOW (2-5 min per test) — run manually, not in CI.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _check_server():
    """Check if the Augmentum server is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:7860/api/capabilities", timeout=3)
        return True
    except Exception:
        return False


def _check_lm_studio():
    """Check if LM Studio is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=3)
        return True
    except Exception:
        return False


requires_server = pytest.mark.skipif(
    not _check_server(),
    reason="Augmentum server not running on localhost:7860",
)

requires_model = pytest.mark.skipif(
    not _check_lm_studio(),
    reason="LM Studio not running on 127.0.0.1:1234",
)


@pytest.fixture
def tool():
    """Create ApplicationBuilderTool with real LLM caller."""
    from augmentum.tools.artifact_application import ApplicationBuilderTool

    class FakeStore:
        """Minimal artifact store that captures saves without touching disk."""
        saved = []

        async def save(self, *, data=b"", filename="", fmt="", task_id="",
                       session_id="", display_name="", metadata=None, source_json=None):
            self.saved.append({
                "filename": filename,
                "fmt": fmt,
                "size": len(data),
                "display_name": display_name,
            })
            return {"id": "live_test_" + filename.replace(".", "_")}

    # Auto-detect loaded model name from LM Studio
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
        """Call the real LM Studio backend."""
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
        resp = urllib.request.urlopen(req, timeout=180)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    store = FakeStore()
    return ApplicationBuilderTool(store, real_llm, lambda: {}), store


@requires_model
class TestLivePipeline:
    """Live tests against a real model. SLOW — 2-5 min each."""

    @pytest.mark.asyncio
    async def test_simple_calculator_build(self, tool):
        """Build a simple calculator and verify the output is clean."""
        builder, store = tool

        progress_log = []

        async def log_progress(data):
            text = data.pop("_content_delta", "")
            if text.strip():
                progress_log.append(text.strip())
            progress = data.get("project_progress", {})
            if progress.get("pass") and progress.get("status"):
                print(f"  [{progress['pass']}] {progress['status']}: {progress.get('detail', '')}")

        print("\n--- Building calculator app ---")
        result = await builder.execute(
            description="simple calculator with add subtract multiply divide",
            scaffold="form",
            _progress_callback=log_progress,
            _request_model="",
        )

        print(f"\nResult: success={result.success}")
        if not result.success:
            print(f"Error: {result.error}")

        # Basic assertions
        assert result.success, f"Pipeline failed: {result.error}"

        project = result.metadata.get("project", {})
        files = project.get("files", [])
        print(f"Files: {len(files)}")
        for f in files:
            print(f"  {f['path']} ({f['role']}) — {len(f['content'])} chars, {f['content'].count(chr(10))+1} lines")

        # Verify we got at least 2 files (HTML + something)
        assert len(files) >= 2, f"Expected at least 2 files, got {len(files)}"

        # Verify file path cleanliness — no backticks, no quotes
        for f in files:
            assert '`' not in f["path"], f"Backtick in path: {f['path']}"
            assert '"' not in f["path"], f"Quote in path: {f['path']}"
            assert "'" not in f["path"], f"Quote in path: {f['path']}"

        # Verify we have an entry HTML file
        entry = next((f for f in files if f["role"] == "entry"), None)
        assert entry, "No entry (HTML) file found"
        assert "<html" in entry["content"].lower(), "Entry file doesn't contain <html>"

        # Verify no duplicate filenames
        paths = [f["path"] for f in files]
        assert len(paths) == len(set(paths)), f"Duplicate paths: {paths}"

        # Verify artifact was saved
        assert len(store.saved) == 1, f"Expected 1 artifact save, got {len(store.saved)}"
        assert store.saved[0]["fmt"] == "zip"
        assert store.saved[0]["size"] > 0

        # Verify score was assigned
        score = project.get("score", 0)
        print(f"Score: {score}/10")
        assert score > 0, "No score assigned"

        # Check for common scope issues that autofix should have caught
        all_js = "\n".join(f["content"] for f in files if f["role"] in ("script", "module"))
        if all_js:
            # Count const declarations across files
            import re
            js_files = [f for f in files if f["role"] in ("script", "module")]
            if len(js_files) > 1:
                all_consts = {}
                for f in js_files:
                    for m in re.finditer(r'^const\s+(\w+)', f["content"], re.MULTILINE):
                        name = m.group(1)
                        if name in all_consts:
                            pytest.fail(
                                f"Duplicate const '{name}' in {f['path']} "
                                f"(first in {all_consts[name]}) — autofix should have caught this"
                            )
                        all_consts[name] = f["path"]

        print("\n[PASS] Calculator build passed all checks")

    @pytest.mark.asyncio
    async def test_build_produces_valid_assembly(self, tool):
        """Verify the assembled HTML is valid and contains all file contents."""
        builder, store = tool

        print("\n--- Building simple app for assembly test ---")
        result = await builder.execute(
            description="simple counter app with increment and decrement buttons",
            scaffold="static",
            _request_model="",
        )

        if not result.success:
            pytest.skip(f"Build failed: {result.error}")

        project = result.metadata.get("project", {})
        files = project.get("files", [])

        if len(files) < 2:
            pytest.skip(f"Only {len(files)} files generated — need 2+ for assembly test")

        # Import and test the assembly
        from augmentum.tools.artifact_application import ApplicationBuilderTool as ABT
        dummy = ABT.__new__(ABT)
        assembled = dummy._assemble(files)

        assert assembled, "Assembly returned None"
        assert "<html" in assembled.lower(), "Assembly missing <html>"

        # Verify CSS was inlined
        css_files = [f for f in files if f["role"] == "style"]
        if css_files:
            assert "<style>" in assembled, "CSS not inlined in assembly"

        # Verify JS was inlined
        js_files = [f for f in files if f["role"] == "script"]
        if js_files:
            assert "<script>" in assembled, "JS not inlined in assembly"

        # Verify external file references were stripped
        for f in files:
            if f["role"] == "entry":
                continue
            assert f'src="{f["path"]}"' not in assembled, \
                f"External <script src=\"{f['path']}\"> not stripped from assembly"
            assert f'href="{f["path"]}"' not in assembled, \
                f"External <link href=\"{f['path']}\"> not stripped from assembly"

        print(f"[PASS] Assembly test passed — {len(assembled)} chars, all files inlined")

    @pytest.mark.asyncio
    async def test_progress_streams_during_build(self, tool):
        """Verify progress callbacks fire during the pipeline."""
        builder, store = tool

        passes_seen = set()

        async def track_progress(data):
            data.pop("_content_delta", "")
            progress = data.get("project_progress", {})
            if progress.get("pass"):
                passes_seen.add(progress["pass"])

        print("\n--- Building app to test progress streaming ---")
        result = await builder.execute(
            description="simple timer with start stop reset",
            scaffold="static",
            _progress_callback=track_progress,
            _request_model="",
        )

        print(f"Passes seen: {passes_seen}")

        # Should have seen at least plan and generate
        assert "plan" in passes_seen, "Plan pass not seen in progress"
        assert "generate" in passes_seen, "Generate pass not seen in progress"

        if result.success:
            assert "deliver" in passes_seen, "Deliver pass not seen in progress"

        print(f"[PASS] Progress streaming test passed — {len(passes_seen)} passes observed")


@requires_server
class TestLiveServerEndpoint:
    """Tests that hit the actual Augmentum server API."""

    def test_build_tool_registered(self):
        """Verify build_application tool is registered on the server."""
        import urllib.request
        resp = urllib.request.urlopen(
            "http://localhost:7860/api/config/passthrough-tools",
            timeout=5,
        )
        data = json.loads(resp.read())
        tools = data.get("tools", [])
        names = [t["name"] for t in tools]
        assert "build_application" in names, \
            f"build_application not in tool list: {names}"
        print(f"[PASS] build_application registered — {len(tools)} total tools")

    def test_intent_classifier_detects_build(self):
        """Verify the intent classifier catches build requests."""
        from augmentum.tools.intent import classify_intent
        intent = classify_intent("build me a calculator app")
        assert intent.action == "build_app", f"Expected build_app, got {intent.action}"
        assert intent.confidence >= 0.8, f"Low confidence: {intent.confidence}"
        print(f"[PASS] Intent classifier: action={intent.action}, confidence={intent.confidence}")

    def test_intent_classifier_no_false_positives(self):
        """Verify normal messages don't trigger build_app."""
        from augmentum.tools.intent import classify_intent
        for msg in ["what's the weather", "explain python lists", "help me debug this code"]:
            intent = classify_intent(msg)
            assert intent.action != "build_app", \
                f"False positive: '{msg}' classified as build_app"
        print("[PASS] No false positives on normal messages")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
