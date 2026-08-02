"""Multi-model prompt testing harness for the Application Builder pipeline.

Runs a matrix of build descriptions × models and collects metrics per build:
- Plan completeness (scaffold enforcement triggered? gap analysis?)
- Interceptor fixes applied
- Validate errors found / auto-fixed / LLM-fixed
- Verify errors found / fixed / rolled back
- Final quickjs clean?
- File count, line count, total chars
- Score
- Total LLM calls
- Build time

Usage:
  # Run all tests against default model:
  python -m pytest tests/test_prompt_matrix.py -v -s

  # Run against a specific model:
  MATRIX_MODEL=zai-org/glm-4.7-flash python -m pytest tests/test_prompt_matrix.py -v -s

  # Run a single description:
  python -m pytest tests/test_prompt_matrix.py -v -s -k "calculator"

  # Run with multiple models (comma-separated):
  MATRIX_MODELS=nvidia/nemotron-3-nano-4b,zai-org/glm-4.7-flash python -m pytest tests/test_prompt_matrix.py -v -s

Output: results written to tests/matrix_results.json after each run.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Test matrix: descriptions covering every scaffold + complexity level
# ---------------------------------------------------------------------------

BUILD_MATRIX = [
    # --- FORM scaffold ---
    {"id": "calc-simple", "desc": "simple calculator with add subtract multiply divide", "scaffold": "form", "complexity": "micro"},
    {"id": "todo-crud", "desc": "todo list with add, delete, mark complete, and localStorage persistence", "scaffold": "form", "complexity": "small"},
    {"id": "quiz-app", "desc": "quiz app with 5 multiple choice questions, score counter, and results page", "scaffold": "form", "complexity": "small"},

    # --- DASHBOARD scaffold ---
    {"id": "weather-dash", "desc": "weather dashboard showing current temperature, 5-day forecast line chart, and humidity gauge. Use Chart.js. Blue theme.", "scaffold": "dashboard", "complexity": "small"},
    {"id": "analytics-dash", "desc": "analytics dashboard with 3 chart sections: line chart for monthly revenue, bar chart for product sales, pie chart for traffic sources. Sidebar with date range filter. Dark theme.", "scaffold": "dashboard", "complexity": "medium"},

    # --- GAME scaffold ---
    {"id": "snake-game", "desc": "snake game with arrow key controls, growing snake, random food, score counter, game over on wall or self collision, restart button", "scaffold": "game", "complexity": "small"},
    {"id": "shooter-game", "desc": "space shooter with player ship (arrow keys to move, spacebar to shoot), enemy waves from top, particle explosions, score, lives, and game over screen", "scaffold": "game", "complexity": "medium"},

    # --- STATIC scaffold ---
    {"id": "landing-page", "desc": "landing page with hero section, 3 feature cards, testimonial carousel, and contact form. Modern design with scroll animations.", "scaffold": "static", "complexity": "small"},
    {"id": "pomodoro-timer", "desc": "pomodoro timer with 25min work / 5min break cycles, start pause reset buttons, session counter, and notification sound", "scaffold": "static", "complexity": "small"},
    {"id": "kanban-board", "desc": "project management kanban board with 3 columns (To Do, In Progress, Done), drag and drop cards between columns, add new cards with title and priority, localStorage persistence", "scaffold": "form", "complexity": "medium"},
]


# ---------------------------------------------------------------------------
# LM Studio backend
# ---------------------------------------------------------------------------

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234")


def _check_lm_studio():
    import urllib.request
    try:
        urllib.request.urlopen(f"{LM_STUDIO_URL}/v1/models", timeout=3)
        return True
    except Exception:
        return False


def _get_models() -> list[str]:
    """Get model list from env or auto-detect from LM Studio."""
    env_models = os.environ.get("MATRIX_MODELS", os.environ.get("MATRIX_MODEL", ""))
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]

    # Auto-detect first loaded model
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"{LM_STUDIO_URL}/v1/models", timeout=5)
        data = json.loads(resp.read())
        if data.get("data"):
            return [data["data"][0]["id"]]
    except Exception:
        pass
    return []


requires_backend = pytest.mark.skipif(
    not _check_lm_studio(),
    reason=f"LM Studio not running at {LM_STUDIO_URL}",
)


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

class BuildMetrics:
    """Collects pipeline metrics during a build."""

    def __init__(self):
        self.llm_calls = 0
        self.passes_seen: set[str] = set()
        self.scaffold_enforcement = False
        self.gap_analysis = False
        self.interceptor_fixes = 0
        self.validate_errors = 0
        self.validate_autofixed = 0
        self.verify_errors = 0
        self.verify_fixed = 0
        self.verify_rollback = False
        self.verify_clean = False
        self.files: list[dict] = []
        self.score = 0.0
        self.success = False
        self.error = ""
        self.build_time = 0.0

    def to_dict(self) -> dict:
        total_lines = sum(f["content"].count("\n") + 1 for f in self.files)
        total_chars = sum(len(f["content"]) for f in self.files)
        return {
            "success": self.success,
            "error": self.error,
            "file_count": len(self.files),
            "total_lines": total_lines,
            "total_chars": total_chars,
            "score": self.score,
            "llm_calls": self.llm_calls,
            "build_time_s": round(self.build_time, 1),
            "passes": sorted(self.passes_seen),
            "scaffold_enforcement": self.scaffold_enforcement,
            "gap_analysis": self.gap_analysis,
            "interceptor_fixes": self.interceptor_fixes,
            "validate_errors": self.validate_errors,
            "verify_errors": self.verify_errors,
            "verify_rollback": self.verify_rollback,
            "verify_clean": self.verify_clean,
            "files": [{"path": f["path"], "role": f.get("role", ""), "lines": f["content"].count("\n") + 1} for f in self.files],
            # Quality checks
            "has_duplicate_classes": self._check_duplicate_classes(),
            "has_empty_files": any(len(f["content"].strip()) < 10 for f in self.files),
            "has_bad_paths": any(c in f["path"] for f in self.files for c in '`"\''),
        }

    def _check_duplicate_classes(self) -> bool:
        for f in self.files:
            if f.get("role") not in ("script", "module"):
                continue
            classes = re.findall(r"class\s+(\w+)", f["content"])
            if len(classes) != len(set(classes)):
                return True
        return False


async def run_build(desc: str, scaffold: str, model: str, max_tokens: int = 8192) -> BuildMetrics:
    """Run a single build and collect metrics."""
    from augmentum.tools.artifact_application import ApplicationBuilderTool
    import urllib.request

    metrics = BuildMetrics()

    # LLM caller with call counting
    async def counted_llm(messages, max_tokens=4096, model_name="", **kw):
        metrics.llm_calls += 1
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max(max_tokens, 4096),  # Ensure reasoning models get headroom
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
        return data["choices"][0]["message"].get("content", "")

    class FakeStore:
        async def save(self, **kw):
            return {"id": "matrix_test"}

    tool = ApplicationBuilderTool(
        FakeStore(), counted_llm,
        lambda: {"app_builder_max_tokens": max_tokens},
    )

    # Progress callback to track pipeline passes
    async def on_progress(data):
        p = data.get("project_progress", {})
        if p.get("pass"):
            metrics.passes_seen.add(p["pass"])

    t0 = time.time()
    try:
        result = await tool.execute(
            description=desc,
            scaffold=scaffold,
            _progress_callback=on_progress,
            _request_model=model,
        )
        metrics.build_time = time.time() - t0
        metrics.success = result.success
        metrics.error = result.error or ""

        project = result.metadata.get("project", {})
        metrics.files = project.get("files", [])
        metrics.score = project.get("score", 0)
    except Exception as exc:
        metrics.build_time = time.time() - t0
        metrics.error = str(exc)

    return metrics


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "matrix_results.json")


def _load_results() -> dict:
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"runs": []}


def _save_results(results: dict):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _save_run(model: str, build_id: str, desc: str, scaffold: str, metrics: BuildMetrics):
    results = _load_results()
    results["runs"].append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "build_id": build_id,
        "description": desc,
        "scaffold": scaffold,
        **metrics.to_dict(),
    })
    _save_results(results)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_backend
class TestPromptMatrix:
    """Run each build description against available models and collect metrics."""

    @pytest.fixture(autouse=True)
    def _models(self):
        self.models = _get_models()
        if not self.models:
            pytest.skip("No models available")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("build", BUILD_MATRIX, ids=[b["id"] for b in BUILD_MATRIX])
    async def test_build(self, build):
        for model in self.models:
            model_short = model.split("/")[-1][:20]
            print(f"\n{'='*60}")
            print(f"  {build['id']} | {model_short} | {build['scaffold']}")
            print(f"  {build['desc'][:70]}")
            print(f"{'='*60}")

            metrics = await run_build(
                desc=build["desc"],
                scaffold=build["scaffold"],
                model=model,
            )

            # Save results
            _save_run(model, build["id"], build["desc"], build["scaffold"], metrics)

            # Print summary
            m = metrics.to_dict()
            status = "PASS" if m["success"] else "FAIL"
            print(f"  {status} | {m['file_count']} files | {m['total_lines']} lines | {m['build_time_s']}s | {m['llm_calls']} LLM calls | Score {m['score']}")
            if m["verify_clean"]:
                print(f"  Verify: CLEAN")
            elif m["verify_errors"]:
                print(f"  Verify: {m['verify_errors']} errors {'(rolled back)' if m['verify_rollback'] else '(fixed)'}")
            if m["has_duplicate_classes"]:
                print(f"  WARNING: Duplicate class definitions found")
            if m["has_empty_files"]:
                print(f"  WARNING: Empty files detected")
            if not m["success"]:
                print(f"  Error: {m['error'][:100]}")

            # Basic assertions — we want data, not perfection
            assert metrics.success, f"Build failed: {metrics.error}"
            assert len(metrics.files) >= 2, f"Expected 2+ files, got {len(metrics.files)}"


@requires_backend
class TestMatrixSummary:
    """Print summary of all collected results."""

    def test_print_summary(self):
        results = _load_results()
        if not results["runs"]:
            pytest.skip("No results collected yet")

        # Group by model
        by_model: dict[str, list] = {}
        for run in results["runs"]:
            model = run["model"].split("/")[-1][:25]
            by_model.setdefault(model, []).append(run)

        print(f"\n{'='*80}")
        print(f"  PROMPT MATRIX RESULTS — {len(results['runs'])} runs across {len(by_model)} models")
        print(f"{'='*80}")

        for model, runs in sorted(by_model.items()):
            successes = sum(1 for r in runs if r["success"])
            avg_files = sum(r["file_count"] for r in runs) / len(runs)
            avg_lines = sum(r["total_lines"] for r in runs) / len(runs)
            avg_time = sum(r["build_time_s"] for r in runs) / len(runs)
            avg_calls = sum(r["llm_calls"] for r in runs) / len(runs)
            verify_clean = sum(1 for r in runs if r.get("verify_clean"))
            rollbacks = sum(1 for r in runs if r.get("verify_rollback"))
            dup_classes = sum(1 for r in runs if r.get("has_duplicate_classes"))

            print(f"\n  {model}")
            print(f"  {'─'*50}")
            print(f"  Pass rate:    {successes}/{len(runs)} ({successes/len(runs)*100:.0f}%)")
            print(f"  Avg files:    {avg_files:.1f}")
            print(f"  Avg lines:    {avg_lines:.0f}")
            print(f"  Avg time:     {avg_time:.0f}s")
            print(f"  Avg LLM calls: {avg_calls:.1f}")
            print(f"  Verify clean: {verify_clean}/{len(runs)}")
            print(f"  Rollbacks:    {rollbacks}")
            print(f"  Dup classes:  {dup_classes}")

            # Per-build breakdown
            for run in runs:
                status = "✓" if run["success"] else "✗"
                vfy = "clean" if run.get("verify_clean") else f"{run.get('verify_errors', 0)} err"
                print(f"    {status} {run['build_id']:20s} | {run['file_count']}f {run['total_lines']:4d}L | {run['build_time_s']:5.0f}s | {run['llm_calls']:2d} calls | vfy:{vfy} | score:{run.get('score', 0)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
