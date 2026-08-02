"""Application Builder evaluation matrix — instrumented pipeline for manual review.

Runs N prompts × M models, capturing the FULL pipeline trace at every stage:
- Every prompt sent to the LLM and every raw response
- Every autofix/intercept change (before→after diffs)
- Every patch attempt (applied, failed, rolled back)
- QuickJS errors, intent gaps, smoke test failures
- Final assembled HTML for browser testing

Results go to tests/eval_results/app_builder/<run_id>/

Usage:
  # Run full matrix (all models × all prompts):
  python tests/eval_app_builder_matrix.py

  # Run single model:
  python tests/eval_app_builder_matrix.py --model "gemma-3-4b-it"

  # Run single prompt:
  python tests/eval_app_builder_matrix.py --prompt 1

  # Run specific combo:
  python tests/eval_app_builder_matrix.py --model "gemma-3-4b-it" --prompt 0

  # List available models/prompts:
  python tests/eval_app_builder_matrix.py --list

  # Custom LM Studio address:
  python tests/eval_app_builder_matrix.py --api http://127.0.0.1:1234
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import difflib
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from augmentum.tools.artifact_application import ApplicationBuilderTool, PassResult, PipelineContext

# ---------------------------------------------------------------------------
# Test Matrix
# ---------------------------------------------------------------------------

MODELS = [
    {"id": "gemma-3-4b-it", "tier": "tiny", "label": "Gemma 3 4B"},
    {"id": "qwen3.5-4b-claude-4.6-opus-reasoning-distilled", "tier": "tiny+", "label": "Qwen 3.5 4B distill"},
    {"id": "qwopus3.5-9b-v3", "tier": "small", "label": "Qwopus 3.5 9B v3"},
    {"id": "google/gemma-4-26b-a4b", "tier": "mid-moe", "label": "Gemma 4 26B MoE"},
    {"id": "crow-9b-opus-4.6-distill-heretic_qwen3.5", "tier": "mid", "label": "Crow 9B"},
    {"id": "qwen3.5-40b-claude-4.6-opus-deckard-heretic-uncensored-thinking-i1", "tier": "strong", "label": "Qwen 3.5 40B"},
]

PROMPTS = [
    # --- Micro / single-purpose (micro scaffold, 2-3 files) -------------
    {
        "description": "pomodoro timer: 25 minute work, 5 minute break, start/pause/reset buttons, circular progress ring, sound alert at end of each session, persist work-session count to localStorage",
        "scaffold": "static",
        "label": "pomodoro",
        "stress": "state machine across sessions, interval/timing math, SVG ring progress, audio alert",
    },
    {
        "description": "scientific calculator with add, subtract, multiply, divide, plus percent, memory store/recall, and a history log of the last 10 expressions",
        "scaffold": "form",
        "label": "calculator-scientific",
        "stress": "expression parsing, memory state, history list, button grid layout",
    },
    {
        "description": "simple stopwatch with start, stop, reset, and a lap button that records lap times in a scrollable list",
        "scaffold": "static",
        "label": "stopwatch",
        "stress": "timing precision, laps list, monospace display, three-button state",
    },
    {
        "description": "tip calculator that takes bill amount, tip percent, and number of people, and shows tip per person and total per person",
        "scaffold": "form",
        "label": "tip-calculator",
        "stress": "form validation, live recompute on input, currency formatting",
    },

    # --- Small / state-heavy (static scaffold, 3-4 files) ---------------
    {
        "description": "todo list with add, edit inline, delete, mark complete, filter by status (all/active/done), and localStorage persistence",
        "scaffold": "static",
        "label": "todo-crud",
        "stress": "CRUD, inline edit, filter view, persistence round-trip",
    },
    {
        "description": "kanban board with draggable cards between columns, card editor modal, localStorage persistence, and undo/redo",
        "scaffold": "static",
        "label": "kanban",
        "stress": "drag-and-drop API, modal lifecycle, state history, multi-concern coordination",
    },
    {
        "description": "markdown note editor with live preview, syntax highlighting, multiple notes with sidebar navigation, search and filter, and local storage persistence",
        "scaffold": "static",
        "label": "markdown-editor",
        "stress": "text parsing, split-pane layout, multi-document state, search across corpus",
    },
    {
        "description": "palette picker: input a base color, show analogous / complementary / triadic / tetradic schemes with swatches, and a copy-to-clipboard hex button on each swatch",
        "scaffold": "static",
        "label": "palette-picker",
        "stress": "color math (HSL conversions), clipboard API, multi-variant display",
    },

    # --- Dashboard / charts ---------------------------------------------
    {
        "description": "personal finance dashboard with transaction entry form with categories, monthly spending breakdown pie chart, income vs expense bar chart, running balance line chart, and CSV export",
        "scaffold": "dashboard",
        "label": "finance-dashboard",
        "stress": "3 Chart.js chart types, form→data→chart flow, export feature, multi-file architecture",
    },
    {
        "description": "weather dashboard stub that reads 7-day forecast JSON from localStorage (seed with sample), shows today's temp large, a week line chart, and 7 day-cards",
        "scaffold": "dashboard",
        "label": "weather-dashboard",
        "stress": "Chart.js line, multi-card layout, responsive grid, mocked data source",
    },

    # --- Game / animation loop ------------------------------------------
    {
        "description": "asteroid shooter with ship rotation, momentum physics, bullet wrapping, wave spawner, particle explosions, and a persistent high score table",
        "scaffold": "game",
        "label": "asteroid-shooter",
        "stress": "complex game loop, trig math, entity management, particle system, localStorage",
    },
    {
        "description": "snake game with arrow key controls, food spawning, growing snake, wall collision, self collision, score counter, and a restart button on game over",
        "scaffold": "game",
        "label": "snake",
        "stress": "canvas grid rendering, input → state → render loop, collision detection, restart flow",
    },
]


# ---------------------------------------------------------------------------
# File-heuristic scanner (post-build quality signals)
# ---------------------------------------------------------------------------


def scan_final_files(files: list[dict], planned: list[dict]) -> dict:
    """Read delivered files and return per-build quality signals.

    This complements ctx.errors and the LLM judge score by catching
    things we KNOW are tells of low-quality output:

    * ``stub_markers``     — files still containing TODO/placeholder prose
    * ``selfdoubt_left``   — self-monologue comments the polish strip missed
    * ``uses_css_vars``    — True if any ``var(--X)`` use exists (design-system hit)
    * ``uses_hardcoded_hex`` — count of inline ``#rrggbb`` that SHOULD be vars
    * ``provides_honored`` — for contracts present, fraction delivered in code
    * ``lines_total``      — total non-blank lines across all files
    * ``file_count``       — number of delivered files
    """
    from augmentum.tools.artifact_application import (
        _extract_provides,
        _normalize_symbol,
    )

    stub_patterns = re.compile(r"\b(todo|placeholder|your code here|implement this)\b", re.IGNORECASE)
    selfdoubt_patterns = re.compile(
        r"//\s*(actually,|let's assume|for simplicity|i'll just|we'll just)",
        re.IGNORECASE,
    )

    lines_total = 0
    stub_hits = 0
    selfdoubt_hits = 0
    css_var_hits = 0
    hex_count = 0
    script_content = ""

    for f in files:
        content = f.get("content", "") or ""
        lines_total += sum(1 for ln in content.split("\n") if ln.strip())
        role = f.get("role", "")
        if role in ("script", "module"):
            script_content += "\n" + content
            stub_hits += len(stub_patterns.findall(content))
            selfdoubt_hits += len(selfdoubt_patterns.findall(content))
        if role == "style":
            css_var_hits += len(re.findall(r"var\(--[\w-]+\)", content))
            hex_count += len(re.findall(r"#[0-9a-fA-F]{3,6}\b", content))

    # Contract honour rate: how many declared PROVIDES actually show up?
    declared = []
    for p in planned:
        declared.extend(p.get("provides") or [])
    if declared:
        all_actual = _extract_provides(script_content)
        honored = sum(
            1 for d in declared if _normalize_symbol(d) in all_actual
        )
        provides_honored = round(honored / len(declared), 2)
    else:
        provides_honored = None  # No contracts declared — not applicable

    return {
        "file_count": len(files),
        "lines_total": lines_total,
        "stub_markers": stub_hits,
        "selfdoubt_comments_left": selfdoubt_hits,
        "uses_css_vars": css_var_hits > 0,
        "css_var_occurrences": css_var_hits,
        "hardcoded_hex_count": hex_count,
        "provides_honored": provides_honored,
        "provides_declared": len(declared),
    }


# ---------------------------------------------------------------------------
# Trace Capture
# ---------------------------------------------------------------------------

@dataclass
class LLMCall:
    """One LLM request/response pair."""
    stage: str
    messages: list[dict]
    raw_response: str
    tokens_prompt: int = 0
    tokens_completion: int = 0
    duration_s: float = 0.0
    kwargs: dict = field(default_factory=dict)


@dataclass
class StageTrace:
    """Trace for one pipeline stage."""
    name: str
    iterations: list[dict] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)
    autofix_diffs: list[dict] = field(default_factory=list)
    errors_detected: list[str] = field(default_factory=list)
    patches_applied: int = 0
    patches_failed: int = 0
    rollbacks: int = 0
    result_detail: str = ""
    duration_s: float = 0.0


@dataclass
class BuildTrace:
    """Full trace for one model × prompt build."""
    model_id: str
    model_label: str
    model_tier: str
    prompt_label: str
    prompt_description: str
    scaffold: str
    stages: dict[str, StageTrace] = field(default_factory=dict)
    files_planned: list[dict] = field(default_factory=list)
    files_generated: list[dict] = field(default_factory=list)
    assembled_html: str = ""
    score: float = 0.0
    total_tokens: int = 0
    total_llm_calls: int = 0
    total_duration_s: float = 0.0
    success: bool = False
    error: str = ""
    # --- New pipeline signals (toolkit features added this session) -----
    # Populated post-build by summarise_pipeline_signals() so downstream
    # tooling can see whether each lever actually helped vs a baseline.
    batch_gen_used: bool = False          # Did the single-call batch path land?
    batch_gen_complete: bool = False      # Did it produce all planned files?
    contracts_declared_count: int = 0     # Total PROVIDES+DEPENDS+WIRES entries across files
    contract_violations: int = 0          # From ctx.errors that start with "CONTRACT:"
    intent_features_declared: int = 0     # # of intent hints that fired on this description
    warnings: list[str] = field(default_factory=list)  # ToolResult.warnings (quality flags)
    smoke_click_sequence: list[str] = field(default_factory=list)  # CDP clicks exercised
    file_heuristics: dict = field(default_factory=dict)  # post-delivery quality scan

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        d = {
            "model": {"id": self.model_id, "label": self.model_label, "tier": self.model_tier},
            "prompt": {"label": self.prompt_label, "description": self.prompt_description, "scaffold": self.scaffold},
            "success": self.success,
            "error": self.error,
            "score": self.score,
            "total_tokens": self.total_tokens,
            "total_llm_calls": self.total_llm_calls,
            "total_duration_s": round(self.total_duration_s, 1),
            "files_planned": self.files_planned,
            "files_generated": [
                {"path": f["path"], "role": f.get("role", ""), "lines": f["content"].count("\n") + 1, "chars": len(f["content"])}
                for f in self.files_generated
            ],
            # New-pipeline signals — collapse to primitives for easy diff.
            "signals": {
                "batch_gen_used": self.batch_gen_used,
                "batch_gen_complete": self.batch_gen_complete,
                "contracts_declared_count": self.contracts_declared_count,
                "contract_violations": self.contract_violations,
                "intent_features_declared": self.intent_features_declared,
                "warnings_count": len(self.warnings),
                "warnings": self.warnings,
                "smoke_click_sequence": self.smoke_click_sequence,
                "file_heuristics": self.file_heuristics,
            },
            "stages": {},
        }
        for name, stage in self.stages.items():
            sd = {
                "duration_s": round(stage.duration_s, 1),
                "result_detail": stage.result_detail,
                "errors_detected": stage.errors_detected,
                "patches_applied": stage.patches_applied,
                "patches_failed": stage.patches_failed,
                "rollbacks": stage.rollbacks,
                "autofix_diffs": stage.autofix_diffs,
                "llm_calls": [],
            }
            for call in stage.llm_calls:
                sd["llm_calls"].append({
                    "stage": call.stage,
                    "system_prompt": call.messages[0]["content"] if call.messages else "",
                    "user_prompt": call.messages[-1]["content"] if len(call.messages) > 1 else "",
                    "user_prompt_length": len(call.messages[-1]["content"]) if len(call.messages) > 1 else 0,
                    "raw_response": call.raw_response,
                    "raw_response_length": len(call.raw_response),
                    "tokens_prompt": call.tokens_prompt,
                    "tokens_completion": call.tokens_completion,
                    "duration_s": round(call.duration_s, 1),
                })
            d["stages"][name] = sd
        return d


# ---------------------------------------------------------------------------
# Instrumented Builder
# ---------------------------------------------------------------------------

class InstrumentedBuilder(ApplicationBuilderTool):
    """Subclass that captures full trace data at every pipeline stage."""

    def __init__(self, api_base: str, model_id: str):
        self._api_base = api_base.rstrip("/")
        self._model_id_override = model_id
        self._trace: BuildTrace | None = None
        self._current_stage: str = ""
        self._stage_start: float = 0.0
        self._file_snapshots: dict[str, str] = {}  # path → content before stage

        # Create fake store
        class _Store:
            async def save(self, **kw):
                return {"id": "eval_" + kw.get("filename", "unknown").replace(".", "_")}

        # Wrap LLM caller as a plain function so the pipeline can set
        # _last_usage on it (can't set attrs on bound methods).
        async def _llm_wrapper(messages, max_tokens=4096, model="", **kw):
            return await self._instrumented_llm(messages, max_tokens=max_tokens, model=model, **kw)
        _llm_wrapper._last_usage = None
        self._llm_fn = _llm_wrapper

        super().__init__(_Store(), _llm_wrapper, lambda: {})

    async def _instrumented_llm(self, messages: list[dict], max_tokens: int = 4096,
                                 model: str = "", **kw) -> str:
        """LLM caller that captures every call."""
        import os as _os
        import urllib.request

        t0 = time.monotonic()
        payload = json.dumps({
            "model": self._model_id_override,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": kw.get("temperature", 0.7),
        }).encode()

        # X-Augmentum-Tools: none — suppress SSOS auto-tools (web search,
        # calc, datetime) that the passthrough handler otherwise attaches
        # to every completion. Bench runs want the raw model signal, not
        # a model that googled the prompt before answering.
        headers = {
            "Content-Type": "application/json",
            "X-Augmentum-Tools": "none",
        }
        token = _os.environ.get("AUGMENTUM_BENCH_TOKEN", "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            f"{self._api_base}/v1/chat/completions",
            data=payload,
            headers=headers,
        )

        try:
            resp = urllib.request.urlopen(req, timeout=300)
            data = json.loads(resp.read())
        except Exception as exc:
            elapsed = time.monotonic() - t0
            call = LLMCall(
                stage=self._current_stage,
                messages=messages,
                raw_response=f"ERROR: {exc}",
                duration_s=elapsed,
            )
            if self._trace and self._current_stage in self._trace.stages:
                self._trace.stages[self._current_stage].llm_calls.append(call)
            raise

        elapsed = time.monotonic() - t0
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        call = LLMCall(
            stage=self._current_stage,
            messages=messages,
            raw_response=content,
            tokens_prompt=usage.get("prompt_tokens", 0),
            tokens_completion=usage.get("completion_tokens", 0),
            duration_s=elapsed,
        )

        # Record in trace
        if self._trace and self._current_stage in self._trace.stages:
            self._trace.stages[self._current_stage].llm_calls.append(call)

        # Track usage — the pipeline reads _last_usage from the callable.
        self._llm_fn._last_usage = usage

        return content

    def _snapshot_files(self, ctx: PipelineContext) -> dict[str, str]:
        """Snapshot current file contents for before/after comparison."""
        return {f["path"]: f["content"] for f in ctx.files if f.get("content")}

    def _diff_snapshots(self, before: dict[str, str], after: dict[str, str]) -> list[dict]:
        """Compute diffs between two snapshots."""
        diffs = []
        all_paths = set(before.keys()) | set(after.keys())
        for path in sorted(all_paths):
            b = before.get(path, "")
            a = after.get(path, "")
            if b != a:
                diff_lines = list(difflib.unified_diff(
                    b.splitlines(keepends=True),
                    a.splitlines(keepends=True),
                    fromfile=f"before/{path}",
                    tofile=f"after/{path}",
                    lineterm="",
                ))
                diffs.append({
                    "path": path,
                    "diff": "\n".join(diff_lines[:200]),  # cap at 200 lines
                    "lines_changed": len([l for l in diff_lines if l.startswith("+") or l.startswith("-")]),
                })
        return diffs

    async def _pass_plan(self, ctx: PipelineContext) -> PassResult:
        self._current_stage = "plan"
        if "plan" not in self._trace.stages:
            self._trace.stages["plan"] = StageTrace(name="plan")
        t0 = time.monotonic()
        result = await super()._pass_plan(ctx)
        self._trace.stages["plan"].duration_s += time.monotonic() - t0
        self._trace.stages["plan"].result_detail = result.detail
        # Include contract fields (provides/depends/wires) so the signal
        # counter in _populate_new_signals sees the real declaration count.
        # Without these the bench showed contracts_declared=0 even when
        # validate was finding real CONTRACT: violations.
        self._trace.files_planned = [
            {
                "path": f["path"],
                "role": f.get("role", ""),
                "description": f.get("description", ""),
                "provides": list(f.get("provides") or []),
                "depends": list(f.get("depends") or []),
                "wires": list(f.get("wires") or []),
            }
            for f in ctx.planned_files
        ]
        return result

    async def _pass_generate(self, ctx: PipelineContext, progress_cb=None) -> PassResult:
        self._current_stage = "generate"
        if "generate" not in self._trace.stages:
            self._trace.stages["generate"] = StageTrace(name="generate")
        before = self._snapshot_files(ctx)
        t0 = time.monotonic()
        result = await super()._pass_generate(ctx, progress_cb)
        elapsed = time.monotonic() - t0
        self._trace.stages["generate"].duration_s += elapsed
        self._trace.stages["generate"].result_detail = result.detail

        # Capture what _intercept_generated_code changed
        after = self._snapshot_files(ctx)
        diffs = self._diff_snapshots(before, after)
        if diffs:
            self._trace.stages["generate"].autofix_diffs.extend(diffs)

        return result

    async def _pass_validate(self, ctx: PipelineContext) -> PassResult:
        self._current_stage = "validate"
        if "validate" not in self._trace.stages:
            self._trace.stages["validate"] = StageTrace(name="validate")
        before = self._snapshot_files(ctx)
        t0 = time.monotonic()
        result = await super()._pass_validate(ctx)
        elapsed = time.monotonic() - t0
        stage = self._trace.stages["validate"]
        stage.duration_s += elapsed
        stage.result_detail = result.detail
        stage.errors_detected = list(ctx.errors) if ctx.errors else []

        # Capture autofix diffs
        after = self._snapshot_files(ctx)
        diffs = self._diff_snapshots(before, after)
        if diffs:
            stage.autofix_diffs.extend(diffs)

        # Count patches from the result detail
        m = re.search(r"fixed (\d+)", result.detail)
        if m:
            stage.patches_applied += int(m.group(1))

        return result

    async def _pass_improve(self, ctx: PipelineContext) -> PassResult:
        self._current_stage = "improve"
        if "improve" not in self._trace.stages:
            self._trace.stages["improve"] = StageTrace(name="improve")
        before = self._snapshot_files(ctx)
        t0 = time.monotonic()
        result = await super()._pass_improve(ctx)
        elapsed = time.monotonic() - t0
        stage = self._trace.stages["improve"]
        stage.duration_s += elapsed
        stage.result_detail = result.detail

        after = self._snapshot_files(ctx)
        diffs = self._diff_snapshots(before, after)
        if diffs:
            stage.autofix_diffs.extend(diffs)

        return result

    async def _pass_polish(self, ctx: PipelineContext) -> PassResult:
        self._current_stage = "polish"
        if "polish" not in self._trace.stages:
            self._trace.stages["polish"] = StageTrace(name="polish")
        before = self._snapshot_files(ctx)
        t0 = time.monotonic()
        result = await super()._pass_polish(ctx)
        elapsed = time.monotonic() - t0
        stage = self._trace.stages["polish"]
        stage.duration_s += elapsed
        stage.result_detail = result.detail

        after = self._snapshot_files(ctx)
        diffs = self._diff_snapshots(before, after)
        if diffs:
            stage.autofix_diffs.extend(diffs)

        return result

    async def _pass_verify(self, ctx: PipelineContext) -> PassResult:
        self._current_stage = "verify"
        if "verify" not in self._trace.stages:
            self._trace.stages["verify"] = StageTrace(name="verify")
        before = self._snapshot_files(ctx)
        t0 = time.monotonic()
        result = await super()._pass_verify(ctx)
        elapsed = time.monotonic() - t0
        stage = self._trace.stages["verify"]
        stage.duration_s += elapsed
        stage.result_detail = result.detail

        # Capture errors from result detail
        m = re.search(r"(\d+) runtime issues", result.detail)
        if m:
            stage.errors_detected.append(f"{m.group(1)} runtime issues")
        if "rollback" in result.detail.lower():
            stage.rollbacks += 1

        after = self._snapshot_files(ctx)
        diffs = self._diff_snapshots(before, after)
        if diffs:
            stage.autofix_diffs.extend(diffs)

        return result

    async def _pass_deliver(self, ctx: PipelineContext) -> PassResult:
        self._current_stage = "deliver"
        if "deliver" not in self._trace.stages:
            self._trace.stages["deliver"] = StageTrace(name="deliver")
        t0 = time.monotonic()
        result = await super()._pass_deliver(ctx)
        elapsed = time.monotonic() - t0
        self._trace.stages["deliver"].duration_s += elapsed
        self._trace.stages["deliver"].result_detail = result.detail

        # Capture final state
        self._trace.files_generated = copy.deepcopy(ctx.files)
        self._trace.assembled_html = ctx.preview_html
        self._trace.score = ctx.score
        self._trace.total_tokens = ctx._total_tokens
        self._trace.total_llm_calls = ctx._total_llm_calls

        return result

    async def run_traced(self, description: str, scaffold: str, label: str,
                         model_info: dict) -> BuildTrace:
        """Run a full build with tracing enabled."""
        self._trace = BuildTrace(
            model_id=model_info["id"],
            model_label=model_info["label"],
            model_tier=model_info["tier"],
            prompt_label=label,
            prompt_description=description,
            scaffold=scaffold,
        )

        progress_log = []

        async def log_progress(data):
            text = data.pop("_content_delta", "")
            progress = data.get("project_progress", {})
            if progress.get("pass") and progress.get("status"):
                entry = {
                    "pass": progress["pass"],
                    "status": progress["status"],
                    "detail": progress.get("detail", ""),
                    "timestamp": time.monotonic(),
                }
                progress_log.append(entry)
                status_icon = {"running": "⏳", "complete": "✅", "failed": "❌"}.get(progress["status"], "•")
                print(f"    {status_icon} [{progress['pass']}] {progress['status']}: {progress.get('detail', '')}")

        t0 = time.monotonic()
        try:
            result = await self.execute(
                description=description,
                scaffold=scaffold,
                _progress_callback=log_progress,
                _request_model=self._model_id_override,
            )
            self._trace.success = result.success
            if not result.success:
                self._trace.error = result.error or "Unknown error"
            # Capture files from result if deliver didn't run
            if not self._trace.files_generated and result.metadata:
                project = result.metadata.get("project", {})
                if project.get("files"):
                    self._trace.files_generated = project["files"]
        except Exception as exc:
            self._trace.success = False
            self._trace.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

        self._trace.total_duration_s = time.monotonic() - t0

        # Ensure token/call counts are captured even on failure
        if self._trace.total_tokens == 0:
            for stage in self._trace.stages.values():
                for call in stage.llm_calls:
                    self._trace.total_tokens += call.tokens_prompt + call.tokens_completion
            self._trace.total_llm_calls = sum(
                len(s.llm_calls) for s in self._trace.stages.values()
            )

        # Populate the new-pipeline signal fields from result + planned files.
        self._populate_new_signals(locals().get("result", None))

        return self._trace

    def _populate_new_signals(self, result) -> None:
        """Summarise features added this session into trace signals so
        the bench can diff across runs. Must run AFTER the build so
        planned_files / files_generated / errors are final."""
        planned = self._trace.files_planned or []
        generated = self._trace.files_generated or []

        # Contract declarations — sum across all planned files.
        contracts_count = 0
        for p in planned:
            for key in ("provides", "depends", "wires"):
                contracts_count += len(p.get(key) or [])
        self._trace.contracts_declared_count = contracts_count

        # Contract violations — pull from validate-pass errors list.
        val = self._trace.stages.get("validate")
        if val:
            self._trace.contract_violations = sum(
                1 for e in val.errors_detected if str(e).startswith("CONTRACT:")
            )

        # Batch-gen usage — detect from generate pass detail.
        gen = self._trace.stages.get("generate")
        if gen and gen.result_detail:
            detail = gen.result_detail.lower()
            self._trace.batch_gen_used = "(batch)" in detail
            self._trace.batch_gen_complete = (
                self._trace.batch_gen_used
                and not any("incomplete" in e for e in (gen.errors_detected or []))
            )

        # Intent features — re-derive from description for the signal,
        # independent of whether the LLM surfaced them in the plan.
        try:
            from augmentum.tools.application_intent import derive_intent_features
            self._trace.intent_features_declared = len(
                derive_intent_features(self._trace.prompt_description)
            )
        except Exception:
            self._trace.intent_features_declared = 0

        # Warnings — captured from the result's ToolResult.warnings.
        if result is not None:
            self._trace.warnings = list(getattr(result, "warnings", []) or [])

        # Smoke-click sequence — derive what WOULD have been exercised
        # even when browser-verify is disabled, so we can see what the
        # bench thinks the right interaction surface is.
        try:
            from augmentum.tools.application_cdp import derive_smoke_sequence
            self._trace.smoke_click_sequence = derive_smoke_sequence(
                planned, self._trace.prompt_description,
            )
        except Exception:
            self._trace.smoke_click_sequence = []

        # File-heuristic scan over the delivered files.
        self._trace.file_heuristics = scan_final_files(generated, planned)


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def write_report(trace: BuildTrace, out_dir: Path) -> None:
    """Write trace data as JSON + human-readable report + assembled HTML."""
    slug = f"{trace.model_id.replace('/', '_')}--{trace.prompt_label}"

    # JSON trace (machine-readable)
    json_path = out_dir / f"{slug}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False)

    # Assembled HTML (open in browser to test)
    if trace.assembled_html:
        html_path = out_dir / f"{slug}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(trace.assembled_html)

    # Write each file separately for inspection
    files_dir = out_dir / slug
    files_dir.mkdir(exist_ok=True)
    for file_info in trace.files_generated:
        file_path = files_dir / file_info["path"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_info.get("content", ""))

    # Human-readable report
    report_path = out_dir / f"{slug}.report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Build Report: {trace.prompt_label}\n\n")
        f.write(f"**Model:** {trace.model_label} (`{trace.model_id}`) — tier: {trace.model_tier}\n")
        f.write(f"**Prompt:** {trace.prompt_description}\n")
        f.write(f"**Scaffold:** {trace.scaffold}\n")
        f.write(f"**Result:** {'✅ SUCCESS' if trace.success else '❌ FAILED: ' + trace.error[:200]}\n")
        f.write(f"**Score:** {trace.score}/10\n")
        f.write(f"**Duration:** {trace.total_duration_s:.1f}s\n")
        f.write(f"**Tokens:** {trace.total_tokens:,}\n")
        f.write(f"**LLM Calls:** {trace.total_llm_calls}\n\n")

        f.write("## Files Planned\n\n")
        for fp in trace.files_planned:
            f.write(f"- `{fp['path']}` ({fp.get('role', '?')}) — {fp.get('description', '')}\n")

        f.write("\n## Files Generated\n\n")
        for fg in trace.files_generated:
            lines = fg.get("content", "").count("\n") + 1
            chars = len(fg.get("content", ""))
            f.write(f"- `{fg['path']}` ({fg.get('role', '?')}) — {lines} lines, {chars} chars\n")

        f.write("\n---\n\n## Stage-by-Stage Trace\n\n")

        for stage_name in ["plan", "generate", "validate", "improve", "polish", "verify", "deliver"]:
            stage = trace.stages.get(stage_name)
            if not stage:
                f.write(f"### {stage_name.upper()} — skipped\n\n")
                continue

            f.write(f"### {stage_name.upper()} — {stage.result_detail} ({stage.duration_s:.1f}s)\n\n")

            if stage.errors_detected:
                f.write("**Errors detected:**\n")
                for err in stage.errors_detected[:20]:
                    f.write(f"- {err}\n")
                f.write("\n")

            if stage.patches_applied:
                f.write(f"**Patches applied:** {stage.patches_applied}\n")
            if stage.patches_failed:
                f.write(f"**Patches failed:** {stage.patches_failed}\n")
            if stage.rollbacks:
                f.write(f"**Rollbacks:** {stage.rollbacks}\n")

            if stage.autofix_diffs:
                f.write(f"\n**Autofix changes ({len(stage.autofix_diffs)} files):**\n\n")
                for diff in stage.autofix_diffs[:10]:
                    f.write(f"<details><summary>{diff['path']} ({diff['lines_changed']} lines changed)</summary>\n\n")
                    f.write(f"```diff\n{diff['diff'][:2000]}\n```\n\n")
                    f.write("</details>\n\n")

            if stage.llm_calls:
                f.write(f"\n**LLM Calls ({len(stage.llm_calls)}):**\n\n")
                for i, call in enumerate(stage.llm_calls):
                    tok = call.tokens_prompt + call.tokens_completion
                    f.write(f"<details><summary>Call {i+1}: {call.tokens_prompt}+{call.tokens_completion} tokens, {call.duration_s:.1f}s</summary>\n\n")
                    f.write(f"**System prompt** ({len(call.messages[0]['content'])} chars):\n")
                    sys_preview = call.messages[0]["content"][:1000]
                    f.write(f"```\n{sys_preview}\n{'... (truncated)' if len(call.messages[0]['content']) > 1000 else ''}\n```\n\n")
                    if len(call.messages) > 1:
                        user_msg = call.messages[-1]["content"]
                        f.write(f"**User prompt** ({len(user_msg)} chars):\n")
                        user_preview = user_msg[:1500]
                        f.write(f"```\n{user_preview}\n{'... (truncated)' if len(user_msg) > 1500 else ''}\n```\n\n")
                    f.write(f"**Response** ({len(call.raw_response)} chars):\n")
                    resp_preview = call.raw_response[:3000]
                    f.write(f"```\n{resp_preview}\n{'... (truncated)' if len(call.raw_response) > 3000 else ''}\n```\n\n")
                    f.write("</details>\n\n")

            f.write("\n")

    print(f"    📄 Report: {report_path}")
    print(f"    📦 JSON:   {json_path}")
    if trace.assembled_html:
        print(f"    🌐 HTML:   {out_dir / f'{slug}.html'}")


def write_summary(traces: list[BuildTrace], out_dir: Path) -> None:
    """Write comparative summary across all builds."""
    summary_path = out_dir / "SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# App Builder Evaluation Matrix — Summary\n\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Builds:** {len(traces)}\n\n")

        # Summary table
        f.write("## Results Matrix\n\n")
        f.write("| Model | Tier | Prompt | Result | Score | Tokens | LLM Calls | Time | Errors | Autofixes |\n")
        f.write("|-------|------|--------|--------|-------|--------|-----------|------|--------|----------|\n")

        for t in traces:
            result = "✅" if t.success else "❌"
            errors = sum(len(s.errors_detected) for s in t.stages.values())
            autofixes = sum(len(s.autofix_diffs) for s in t.stages.values())
            f.write(
                f"| {t.model_label} | {t.model_tier} | {t.prompt_label} | {result} "
                f"| {t.score} | {t.total_tokens:,} | {t.total_llm_calls} "
                f"| {t.total_duration_s:.0f}s | {errors} | {autofixes} |\n"
            )

        # --- New-pipeline signals table ------------------------------
        # Makes the effect of the toolkit additions (contracts, batch
        # gen, design system) visible at a glance across the matrix.
        f.write("\n## Pipeline-feature signals\n\n")
        f.write("| Prompt | Model | Batch | Contracts | Violations | Provides kept | Stubs | Self-doubt | Uses vars | Warnings |\n")
        f.write("|--------|-------|-------|-----------|------------|---------------|-------|------------|-----------|----------|\n")
        for t in traces:
            h = t.file_heuristics or {}
            batch = "✓" if t.batch_gen_complete else ("~" if t.batch_gen_used else "·")
            kept = h.get("provides_honored")
            kept_s = f"{int(kept*100)}%" if kept is not None else "—"
            vars_used = "✓" if h.get("uses_css_vars") else "·"
            f.write(
                f"| {t.prompt_label} | {t.model_label} | {batch} "
                f"| {t.contracts_declared_count} | {t.contract_violations} | {kept_s} "
                f"| {h.get('stub_markers', 0)} | {h.get('selfdoubt_comments_left', 0)} "
                f"| {vars_used} | {len(t.warnings)} |\n"
            )

        # Per-model summary
        f.write("\n## Per-Model Analysis\n\n")
        models_seen = {}
        for t in traces:
            if t.model_id not in models_seen:
                models_seen[t.model_id] = []
            models_seen[t.model_id].append(t)

        for model_id, model_traces in models_seen.items():
            first = model_traces[0]
            successes = sum(1 for t in model_traces if t.success)
            avg_score = sum(t.score for t in model_traces) / len(model_traces) if model_traces else 0
            avg_tokens = sum(t.total_tokens for t in model_traces) / len(model_traces) if model_traces else 0
            avg_time = sum(t.total_duration_s for t in model_traces) / len(model_traces) if model_traces else 0

            f.write(f"### {first.model_label} (`{model_id}`) — {first.model_tier}\n\n")
            f.write(f"- **Pass rate:** {successes}/{len(model_traces)}\n")
            f.write(f"- **Avg score:** {avg_score:.1f}/10\n")
            f.write(f"- **Avg tokens:** {avg_tokens:,.0f}\n")
            f.write(f"- **Avg time:** {avg_time:.0f}s\n\n")

            for t in model_traces:
                status = "✅" if t.success else "❌"
                f.write(f"- {status} **{t.prompt_label}**: score={t.score}, tokens={t.total_tokens:,}, "
                        f"time={t.total_duration_s:.0f}s")
                if t.error:
                    f.write(f" — {t.error[:100]}")
                f.write("\n")
            f.write("\n")

        # Stage failure patterns
        f.write("## Stage Failure Patterns\n\n")
        stage_names = ["plan", "generate", "validate", "improve", "polish", "verify", "deliver"]
        for stage_name in stage_names:
            issues = []
            for t in traces:
                stage = t.stages.get(stage_name)
                if not stage:
                    continue
                if stage.errors_detected:
                    issues.append(f"  - **{t.model_label} / {t.prompt_label}**: {', '.join(stage.errors_detected[:3])}")
                if stage.rollbacks > 0:
                    issues.append(f"  - **{t.model_label} / {t.prompt_label}**: {stage.rollbacks} rollback(s)")

            if issues:
                f.write(f"### {stage_name.upper()}\n\n")
                for issue in issues:
                    f.write(f"{issue}\n")
                f.write("\n")

    print(f"\n📊 Summary: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_single(api_base: str, model: dict, prompt: dict, out_dir: Path) -> BuildTrace:
    """Run a single model × prompt build."""
    print(f"\n{'='*70}")
    print(f"  Model:  {model['label']} ({model['id']})")
    print(f"  Prompt: {prompt['label']} — {prompt['description'][:80]}...")
    print(f"  Scaffold: {prompt['scaffold']}")
    print(f"{'='*70}")

    builder = InstrumentedBuilder(api_base, model["id"])
    trace = await builder.run_traced(
        description=prompt["description"],
        scaffold=prompt["scaffold"],
        label=prompt["label"],
        model_info=model,
    )

    # Write results
    write_report(trace, out_dir)

    status = "✅ SUCCESS" if trace.success else f"❌ FAILED: {trace.error[:100]}"
    print(f"\n  Result: {status}")
    print(f"  Score: {trace.score}/10 | Tokens: {trace.total_tokens:,} | Time: {trace.total_duration_s:.0f}s")

    return trace


def _load_prior_trace(out_dir: Path, model: dict, prompt: dict) -> BuildTrace | None:
    """Return a BuildTrace reconstructed from a prior JSON result, or None
    if no trace exists (or the stored record isn't a successful build).
    Used by --resume to skip pairs that already succeeded.
    """
    json_path = out_dir / f"{model['id']}--{prompt['label']}.json"
    if not json_path.exists():
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not data.get("success"):
        # Failed or partial — re-run to try again.
        return None
    # Reconstruct the minimum BuildTrace shape needed by write_summary().
    t = BuildTrace(
        model_id=data.get("model", {}).get("id", model["id"]),
        model_label=data.get("model", {}).get("label", model["label"]),
        model_tier=data.get("model", {}).get("tier", model["tier"]),
        prompt_label=data.get("prompt", {}).get("label", prompt["label"]),
        prompt_description=data.get("prompt", {}).get("description", prompt["description"]),
        scaffold=data.get("prompt", {}).get("scaffold", prompt["scaffold"]),
        success=True,
        score=float(data.get("score", 0.0)),
        total_tokens=int(data.get("total_tokens", 0)),
        total_llm_calls=int(data.get("total_llm_calls", 0)),
        total_duration_s=float(data.get("total_duration_s", 0.0)),
        files_planned=data.get("files_planned", []),
    )
    # Pipeline signals live on trace fields too — re-hydrate from JSON.
    signals = data.get("signals", {}) or {}
    for field_name in ("batch_gen_used", "batch_gen_complete", "contracts_declared_count",
                        "contract_violations", "intent_features_declared", "warnings",
                        "smoke_click_sequence", "file_heuristics"):
        if field_name in signals:
            setattr(t, field_name, signals[field_name])
    return t


async def run_matrix(api_base: str, models: list[dict], prompts: list[dict],
                     out_dir: Path, resume: bool = False) -> list[BuildTrace]:
    """Run the full matrix. With ``resume=True``, (model,prompt) pairs
    that already have a successful JSON in ``out_dir`` are skipped — so
    a run that got killed mid-way can be continued by re-invoking with
    ``--resume <run_id>``.
    """
    traces = []
    total = len(models) * len(prompts)
    current = 0

    for model in models:
        for prompt in prompts:
            current += 1

            if resume:
                prior = _load_prior_trace(out_dir, model, prompt)
                if prior is not None:
                    print(f"\n[skip {current}/{total}] {model['label']} × {prompt['label']} — "
                          f"already succeeded (score {prior.score}/10)")
                    traces.append(prior)
                    continue

            print(f"\n\n{'#'*70}")
            print(f"  BUILD {current}/{total}")
            print(f"{'#'*70}")

            try:
                trace = await run_single(api_base, model, prompt, out_dir)
                traces.append(trace)
            except Exception as exc:
                print(f"\n  ❌ FATAL: {exc}")
                traceback.print_exc()
                # Create a failed trace
                trace = BuildTrace(
                    model_id=model["id"], model_label=model["label"],
                    model_tier=model["tier"], prompt_label=prompt["label"],
                    prompt_description=prompt["description"], scaffold=prompt["scaffold"],
                    success=False, error=f"FATAL: {exc}",
                )
                traces.append(trace)

    return traces


def main():
    # Fix Windows console encoding for unicode output
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="App Builder evaluation matrix")
    parser.add_argument("--api", default="http://127.0.0.1:1234",
                        help="LM Studio API base URL")
    parser.add_argument("--model", help="Run only this model ID (substring match)")
    parser.add_argument("--prompt", type=int, help="Run only this prompt index (0-based)")
    parser.add_argument("--list", action="store_true", help="List available models and prompts")
    parser.add_argument("--out", help="Output directory override")
    parser.add_argument("--resume", metavar="RUN_ID",
                        help="Continue a prior run: reuse its output directory and "
                             "skip (model,prompt) pairs that already succeeded. "
                             "Use 'latest' to auto-pick the most recent run dir.")
    args = parser.parse_args()

    if args.list:
        print("\nModels:")
        for i, m in enumerate(MODELS):
            print(f"  [{i}] {m['label']:30s} ({m['id']})")
        print("\nPrompts:")
        for i, p in enumerate(PROMPTS):
            print(f"  [{i}] {p['label']:25s} scaffold={p['scaffold']:10s} — {p['description'][:60]}...")
        return

    # Check the LLM endpoint is reachable (LM Studio or Augmentum).
    import os as _os
    import urllib.request
    _tok = _os.environ.get("AUGMENTUM_BENCH_TOKEN", "")
    _hdrs = {"Authorization": f"Bearer {_tok}"} if _tok else {}
    try:
        req = urllib.request.Request(f"{args.api}/v1/models", headers=_hdrs)
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        print(f"❌ Cannot reach LLM endpoint at {args.api}: {exc}")
        print("   Start the server or pass --api <url>. If auth is required,")
        print("   export AUGMENTUM_BENCH_TOKEN=<bearer token> first.")
        sys.exit(1)

    # Filter models/prompts
    models = MODELS
    prompts = PROMPTS
    if args.model:
        models = [m for m in MODELS if args.model.lower() in m["id"].lower()]
        if not models:
            print(f"❌ No model matching '{args.model}'")
            print("   Use --list to see available models")
            sys.exit(1)
    if args.prompt is not None:
        if 0 <= args.prompt < len(PROMPTS):
            prompts = [PROMPTS[args.prompt]]
        else:
            print(f"❌ Prompt index {args.prompt} out of range (0-{len(PROMPTS)-1})")
            sys.exit(1)

    # Resolve output directory — either a fresh timestamped dir, or an
    # existing one (--resume) so partial results don't get orphaned.
    results_root = ROOT / "tests" / "eval_results" / "app_builder"
    if args.resume:
        if args.resume == "latest":
            candidates = sorted(
                (d for d in results_root.iterdir()
                 if d.is_dir() and d.name[:4].isdigit()),
                reverse=True,
            )
            if not candidates:
                print("❌ --resume latest: no prior runs found")
                sys.exit(1)
            resume_id = candidates[0].name
        else:
            resume_id = args.resume
        out_dir = results_root / resume_id
        if not out_dir.exists():
            print(f"❌ --resume {resume_id}: directory does not exist")
            sys.exit(1)
        run_id = resume_id
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out) if args.out else results_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

    print("\n🔬 App Builder Evaluation Matrix")
    print(f"   Models:  {len(models)}")
    print(f"   Prompts: {len(prompts)}")
    print(f"   Total:   {len(models) * len(prompts)} builds")
    print(f"   Output:  {out_dir}")
    print(f"   API:     {args.api}")
    if args.resume:
        print(f"   Resume:  {run_id} (skipping pairs with success=True)")

    # Save matrix config (overwrites on resume — that's fine, models/prompts
    # should match the prior run or the resume logic wouldn't find anything).
    config = {
        "run_id": run_id,
        "api": args.api,
        "models": models,
        "prompts": prompts,
        "timestamp": datetime.now().isoformat(),
        "resumed": bool(args.resume),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run
    traces = asyncio.run(run_matrix(args.api, models, prompts, out_dir, resume=bool(args.resume)))

    # Write summary
    write_summary(traces, out_dir)

    # Final stats
    successes = sum(1 for t in traces if t.success)
    print(f"\n{'='*70}")
    print(f"  COMPLETE: {successes}/{len(traces)} builds succeeded")
    print(f"  Results:  {out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
