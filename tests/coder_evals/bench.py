"""Hybrid Coder eval bench.

Runs coder eval YAML cases through the real :class:`CoderHandler` loop.
The bench has two backends:

* scripted: deterministic responses from the case YAML, useful for
  regression tests and harness development.
* live: a local Ollama or OpenAI-compatible model, useful for observing
  small/medium model behavior against the same tasks.

Outputs are intentionally inspection-friendly: a JSON result per run plus a
Markdown summary showing step-by-step tool decisions and the final outcome.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from augmentum.coder.models import ContainerInfo, FileEntry
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
    ModelDetails,
    ModelInfo,
    Usage,
)
from augmentum.models.ollama import OllamaBackend
from augmentum.models.openai_compat import OpenAIBackend
from augmentum.modes.coder.handler import CoderHandler
from tests.coder_evals.properties import apply_assertions

ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = ROOT / "tests" / "coder_evals" / "cases"
DEFAULT_OUT_DIR = ROOT / ".augmentum" / "coder-bench"

_SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".rb", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".swift", ".kt", ".sh",
    ".bash", ".yaml", ".yml", ".toml", ".json", ".html", ".css", ".scss",
    ".sql", ".md", ".txt",
}
_SKIP_DIRS = {
    ".git", ".augmentum", "__pycache__", ".pytest_cache", ".venv", "venv",
    "node_modules", "dist", "build", ".next", ".nuxt", "coverage",
}


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


def load_case(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        case = yaml.safe_load(f)
    required = {"name", "tier", "user_message"}
    missing = required - set(case or {})
    if missing:
        raise ValueError(f"{path}: missing required keys {sorted(missing)}")
    if case["tier"] not in {"reflex", "surgical", "composed", "project"}:
        raise ValueError(f"{path}: unknown tier {case['tier']!r}")
    return case


def discover_cases(case_filter: str = "") -> list[Path]:
    if not CASES_DIR.exists():
        return []
    paths = sorted(CASES_DIR.rglob("*.yaml"))
    if not case_filter:
        return paths
    needle = case_filter.lower()
    return [
        p for p in paths
        if needle in str(p.relative_to(CASES_DIR)).lower()
        or needle in p.stem.lower()
    ]


def materialize_workspace(root: Path, files: dict[str, str]) -> None:
    for rel, content in (files or {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def snapshot_workspace(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(part in _SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        try:
            result[rel] = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return result


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    paths = sorted(set(before) | set(after))
    return [p for p in paths if before.get(p) != after.get(p)]


# ---------------------------------------------------------------------------
# Scripted backend
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ScriptedResponse:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    thinking: str = ""


class ScriptedBackend(ModelBackend):
    """Deterministic streaming backend for YAML-driven cases."""

    def __init__(
        self,
        responses: list[ScriptedResponse],
        *,
        model: str = "scripted-coder-eval",
    ) -> None:
        self.responses = responses
        self.model = model
        self.calls_made = 0
        self.requests: list[InternalChatRequest] = []

    @classmethod
    def from_case(cls, case: dict[str, Any], *, model: str) -> ScriptedBackend:
        responses: list[ScriptedResponse] = [
            ScriptedResponse(content=_default_plan_for_case(case)),
        ]
        for raw in case.get("responses") or []:
            calls = _ensure_read_before_edits([
                _normalize_scripted_tool_call(tc)
                for tc in raw.get("tool_calls") or []
            ])
            content = str(raw.get("content") or "")
            if content and len(content) < 120:
                content = (
                    f"{content.rstrip()} The requested change is complete, "
                    "and the final workspace state should satisfy the case."
                )
            responses.append(
                ScriptedResponse(
                    content=content,
                    tool_calls=calls,
                    thinking=str(raw.get("thinking") or ""),
                ),
            )
        if len(responses) == 1:
            responses.append(ScriptedResponse(content="Done."))
        return cls(responses, model=model)

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        response = await self._next(request)
        return InternalChatResponse(
            message=Message(
                role="assistant",
                content=response.content,
                thinking=response.thinking or None,
                tool_calls=[
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["input"]),
                        },
                    }
                    for tc in response.tool_calls
                ] or None,
            ),
            model=request.model or self.model,
            finish_reason="tool_calls" if response.tool_calls else "stop",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def chat_stream(
        self,
        request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        response = await self._next(request)
        if response.thinking:
            yield InternalStreamChunk(
                thinking_delta=response.thinking,
                role="assistant",
                model=request.model or self.model,
            )
        if response.content:
            yield InternalStreamChunk(
                content_delta=response.content,
                role="assistant",
                model=request.model or self.model,
            )
        if response.tool_calls:
            yield InternalStreamChunk(
                augmentum={
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["input"]),
                            },
                        }
                        for i, tc in enumerate(response.tool_calls)
                    ],
                },
                model=request.model or self.model,
            )
        yield InternalStreamChunk(
            done=True,
            finish_reason="tool_calls" if response.tool_calls else "stop",
            model=request.model or self.model,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name=self.model, model=self.model, size=0)]

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails(details={"context_length": 8192})

    async def _next(self, request: InternalChatRequest) -> ScriptedResponse:
        self.requests.append(request)
        if self.calls_made >= len(self.responses):
            return ScriptedResponse(content="Done.")
        response = self.responses[self.calls_made]
        self.calls_made += 1
        return response


def _default_plan_for_case(case: dict[str, Any]) -> str:
    return (
        "Plan:\n"
        f"1. Inspect the workspace for the {case.get('tier', 'composed')} task.\n"
        "2. Make the smallest code changes that satisfy the request.\n"
        "3. Verify the result and summarize the outcome."
    )


def _normalize_scripted_tool_call(raw: dict[str, Any]) -> dict[str, Any]:
    name = str(raw.get("name") or raw.get("tool") or "")
    if name == "code_read":
        name = "file_read"
    inp = dict(raw.get("input") or raw.get("arguments") or {})
    if "file_path" in inp and "path" not in inp:
        inp["path"] = inp.pop("file_path")
    return {
        "id": raw.get("id") or f"scripted-{uuid.uuid4().hex[:10]}",
        "name": name,
        "input": inp,
    }


def _ensure_read_before_edits(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Patch older seed scripts into Hybrid's strict read-before-edit contract."""
    result: list[dict[str, Any]] = []
    seen_reads: set[str] = set()
    for call in calls:
        name = call.get("name")
        inp = call.get("input") or {}
        path = str(inp.get("path") or "")
        if name == "file_read" and path:
            seen_reads.add(path)
        if name in {"code_edit", "code_edit_batch"} and path and path not in seen_reads:
            result.append({
                "id": f"scripted-{uuid.uuid4().hex[:10]}",
                "name": "file_read",
                "input": {"path": path},
            })
            seen_reads.add(path)
        result.append(call)
    return result


# ---------------------------------------------------------------------------
# Local workspace manager
# ---------------------------------------------------------------------------


class LocalWorkspaceManager:
    """ContainerManager-compatible adapter over a temporary local directory."""

    def __init__(self, root: Path, workspace_id: str) -> None:
        self.root = root.resolve()
        self.workspace_id = workspace_id
        self._docker = None

    async def _get_workspace(self, workspace_id: str) -> ContainerInfo:
        self._check_workspace(workspace_id)
        return ContainerInfo(
            id=workspace_id,
            name=f"bench-{workspace_id}",
            container_id="local-bench",
            status="running",
            created_at=time.time(),
            last_active=time.time(),
            tooling_profile="bench",
            safeguards_enabled=True,
        )

    async def cancel_workspace_execs(self, workspace_id: str) -> int:
        self._check_workspace(workspace_id)
        return 0

    async def list_ports(self, workspace_id: str) -> list[dict[str, Any]]:
        self._check_workspace(workspace_id)
        return []

    async def git_checkpoint(self, workspace_id: str, message: str) -> str | None:
        self._check_workspace(workspace_id)
        return None

    async def git_diff(self, workspace_id: str, commit_hash: str) -> str:
        self._check_workspace(workspace_id)
        return ""

    async def file_download(self, workspace_id: str, path: str) -> bytes:
        self._check_workspace(workspace_id)
        return self._host_path(path).read_bytes()

    async def file_read(self, workspace_id: str, path: str) -> str:
        self._check_workspace(workspace_id)
        return self._host_path(path).read_text(encoding="utf-8")

    async def file_write(self, workspace_id: str, path: str, content: str) -> None:
        self._check_workspace(workspace_id)
        target = self._host_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def file_list(
        self,
        workspace_id: str,
        path: str = "/workspace",
    ) -> list[FileEntry]:
        self._check_workspace(workspace_id)
        target = self._host_path(path)
        if not target.exists():
            raise FileNotFoundError(path)
        if target.is_file():
            stat = target.stat()
            return [
                FileEntry(
                    name=target.name,
                    path=self._workspace_path(target),
                    is_dir=False,
                    size=stat.st_size,
                    modified=stat.st_mtime,
                ),
            ]
        entries: list[FileEntry] = []
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            stat = child.stat()
            entries.append(
                FileEntry(
                    name=child.name,
                    path=self._workspace_path(child),
                    is_dir=child.is_dir(),
                    size=stat.st_size,
                    modified=stat.st_mtime,
                ),
            )
        return entries

    async def run_command(
        self,
        workspace_id: str,
        cmd: list[str] | str,
        timeout: float = 30.0,
        *,
        idle_timeout: float | None = None,
        progress_path: str | None = None,
    ) -> str:
        return await self._run_command(
            workspace_id,
            cmd,
            timeout=timeout,
            idle_timeout=idle_timeout,
            progress_path=progress_path,
        )

    async def _run_command(
        self,
        workspace_id: str,
        cmd: list[str] | str,
        timeout: float = 30.0,
        *,
        idle_timeout: float | None = None,
        progress_path: str | None = None,
    ) -> str:
        del idle_timeout, progress_path
        self._check_workspace(workspace_id)
        if isinstance(cmd, str):
            return self._run_local_shell(cmd, timeout=timeout)
        if not cmd:
            return ""
        head = cmd[0]
        if head == "cat" and len(cmd) >= 2:
            return await self.file_read(workspace_id, cmd[1])
        if head == "mkdir" and "-p" in cmd and len(cmd) >= 3:
            self._host_path(cmd[-1]).mkdir(parents=True, exist_ok=True)
            return ""
        if head == "test":
            return self._run_test_command(cmd)
        if head == "ls":
            return self._run_ls(cmd)
        if head == "grep":
            return self._run_grep(cmd)
        if head == "find":
            return self._run_find(cmd)
        if head in {"bash", "sh"} and len(cmd) >= 3 and cmd[1] in {"-c", "-lc"}:
            return self._run_bash_like(cmd[2], timeout=timeout)
        return self._run_local_shell(_quote_cmd(cmd), timeout=timeout)

    def _check_workspace(self, workspace_id: str) -> None:
        if workspace_id != self.workspace_id:
            raise KeyError(f"unknown bench workspace: {workspace_id}")

    def _host_path(self, path: str) -> Path:
        raw = (path or "/workspace").strip()
        if raw in {"", "/workspace"}:
            return self.root
        if raw.startswith("/workspace/"):
            rel = raw[len("/workspace/"):]
        elif raw.startswith("./"):
            rel = raw[2:]
        elif raw.startswith("/"):
            raise ValueError(f"path outside /workspace is not available: {raw}")
        else:
            rel = raw
        resolved = (self.root / rel).resolve()
        if self.root not in (resolved, *resolved.parents):
            raise ValueError(f"path escapes bench workspace: {raw}")
        return resolved

    def _workspace_path(self, path: Path) -> str:
        rel = path.resolve().relative_to(self.root).as_posix()
        return "/workspace" if not rel else f"/workspace/{rel}"

    def _run_test_command(self, cmd: list[str]) -> str:
        if len(cmd) < 3:
            raise RuntimeError("unsupported test command")
        flag, raw_path = cmd[1], cmd[2]
        target = self._host_path(raw_path)
        ok = (
            (flag == "-e" and target.exists())
            or (flag == "-d" and target.is_dir())
            or (flag == "-f" and target.is_file())
        )
        if not ok:
            raise FileNotFoundError(raw_path)
        return ""

    def _run_ls(self, cmd: list[str]) -> str:
        path = next((part for part in reversed(cmd) if part.startswith("/workspace")), "/workspace")
        target = self._host_path(path)
        if not target.exists():
            raise FileNotFoundError(path)
        if target.is_file():
            return target.name + "\n"
        return "\n".join(child.name for child in sorted(target.iterdir())) + "\n"

    def _run_grep(self, cmd: list[str]) -> str:
        try:
            pattern = cmd[cmd.index("-m") + 2] if "-m" in cmd else cmd[-2]
        except (ValueError, IndexError):
            pattern = cmd[-2] if len(cmd) >= 3 else ""
        raw_path = cmd[-1] if cmd else "/workspace"
        root = self._host_path(raw_path)
        regex = re.compile(pattern)
        lines: list[str] = []
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for p in paths:
            if not p.is_file() or self._skip_path(p):
                continue
            with contextlib.suppress(UnicodeDecodeError):
                for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
                    if regex.search(line):
                        lines.append(f"{self._workspace_path(p)}:{i}:{line}")
        return "\n".join(lines) + ("\n" if lines else "")

    def _run_find(self, cmd: list[str]) -> str:
        raw_path = cmd[1] if len(cmd) > 1 else "/workspace"
        pattern = "*"
        if "-name" in cmd:
            with contextlib.suppress(IndexError):
                pattern = cmd[cmd.index("-name") + 1]
        root = self._host_path(raw_path)
        matches = []
        for p in sorted(root.rglob("*")):
            if p.is_file() and not self._skip_path(p) and fnmatch.fnmatch(p.name, pattern):
                matches.append(self._workspace_path(p))
        return "\n".join(matches) + ("\n" if matches else "")

    def _run_bash_like(self, script: str, *, timeout: float) -> str:
        if "printf 'python3\\t'" in script:
            return self._runtime_truth_output()
        if "## Project Detection" in script:
            return self._project_detection_output()
        if "find /workspace" in script and "wc -l" in script:
            return self._find_wc_output(limit=500 if "head -500" in script else 250)
        if script.strip().startswith("mkdir -p "):
            target = script.strip().split(None, 2)[-1]
            self._host_path(target).mkdir(parents=True, exist_ok=True)
            return ""
        if "-m py_compile" in script:
            return self._run_python_module_from_script("py_compile", script, timeout=timeout)
        if "-m pytest" in script:
            return self._run_python_module_from_script("pytest", script, timeout=timeout)
        stat_match = re.search(r"stat -c %Y ['\"]?([^'\" ]+)['\"]?", script)
        if stat_match:
            return str(int(self._host_path(stat_match.group(1)).stat().st_mtime)) + "\n"
        if "git rev-parse --is-inside-work-tree" in script:
            if not (self.root / ".git").exists():
                return ""
            return self._run_local_shell("git status -s", timeout=timeout)
        if _is_shell_test_expression(script):
            return self._evaluate_shell_tests(script)
        if script.startswith("cd /workspace && "):
            script = script[len("cd /workspace && "):]
        return self._run_local_shell(script, timeout=timeout)

    def _run_local_shell(self, script: str, *, timeout: float) -> str:
        script = _translate_workspace_command(script, self.root)
        env = dict(os.environ)
        env.setdefault("PYTHONUTF8", "1")
        try:
            completed = subprocess.run(
                script,
                cwd=self.root,
                env=env,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return f"{stdout}{stderr}\n\n[Command timed out after {timeout}s]"
        return (completed.stdout or "") + (completed.stderr or "")

    def _run_python_module_from_script(
        self,
        module: str,
        script: str,
        *,
        timeout: float,
    ) -> str:
        match = re.search(
            rf"(?:python3|python)\s+-m\s+{re.escape(module)}\s*([^|]*)",
            script,
        )
        tail = (match.group(1) if match else "")
        tail = tail.replace("2>&1", "").replace("2>", "").strip()
        tail = tail.replace("/workspace", self.root.as_posix())
        args = shlex.split(tail, posix=False) if tail else []
        cmd = [sys.executable, "-m", module, *args]
        try:
            completed = subprocess.run(
                cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return f"{stdout}{stderr}\n\n[Command timed out after {timeout}s]"
        return (completed.stdout or "") + (completed.stderr or "")

    def _runtime_truth_output(self) -> str:
        python_version = subprocess.run(
            [sys.executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        npm = shutil.which("npm")
        node = shutil.which("node")
        return "\n".join(
            [
                f"python3\t{python_version or 'missing'}",
                f"node\t{_cmd_version([node, '--version']) if node else 'missing'}",
                "go\tmissing",
                "rustc\tmissing",
                "java\tmissing",
                f"pip\t{_cmd_version([sys.executable, '-m', 'pip', '--version'])}",
                f"npm\t{_cmd_version([npm, '--version']) if npm else 'missing'}",
                "cargo\tmissing",
                "uv\tmissing",
                "pipx\tmissing",
                "pnpm\tmissing",
                "yarn\tmissing",
            ]
        ) + "\n"

    def _project_detection_output(self) -> str:
        lines = ["## Project Detection"]
        if (
            (self.root / "pytest.ini").exists()
            or (self.root / "pyproject.toml").exists()
            or any(self.root.glob("test_*.py"))
            or (self.root / "tests").exists()
        ):
            lines.append("- Python tests detected. Run: `python3 -m pytest`")
        if (self.root / "requirements.txt").exists():
            lines.append("- Python deps: `pip install -r requirements.txt`")
        elif (self.root / "pyproject.toml").exists():
            lines.append("- Python project: `pip install -e .`")
        if not any(p.name not in {".augmentum", ".git"} for p in self.root.iterdir()):
            lines.append("- Empty workspace. Ready for a new project.")
        return "\n".join(lines) + "\n"

    def _find_wc_output(self, *, limit: int) -> str:
        rows: list[str] = []
        for p in self._iter_source_files():
            try:
                line_count = len(p.read_text(encoding="utf-8").splitlines())
            except UnicodeDecodeError:
                continue
            rows.append(f"{line_count:7d} {self._workspace_path(p)}")
            if len(rows) >= limit:
                break
        if len(rows) > 1:
            total = sum(int(row.split(None, 1)[0]) for row in rows)
            rows.append(f"{total:7d} total")
        return "\n".join(rows) + ("\n" if rows else "")

    def _evaluate_shell_tests(self, script: str) -> str:
        for raw in re.findall(r"test\s+(-[edf])\s+(/workspace/[^\s|&;]+)", script):
            flag, path = raw
            target = self._host_path(path)
            ok = (
                (flag == "-e" and target.exists())
                or (flag == "-d" and target.is_dir())
                or (flag == "-f" and target.is_file())
            )
            if ok:
                return ""
        raise FileNotFoundError(script)

    def _iter_source_files(self) -> Iterable[Path]:
        for p in sorted(self.root.rglob("*")):
            if (
                p.is_file()
                and not self._skip_path(p)
                and (
                    p.suffix in _SOURCE_EXTENSIONS
                    or p.name in {"Dockerfile", "Makefile"}
                )
            ):
                yield p

    def _skip_path(self, p: Path) -> bool:
        rel_parts = p.resolve().relative_to(self.root).parts
        return any(part in _SKIP_DIRS for part in rel_parts)


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def _translate_workspace_command(script: str, root: Path) -> str:
    translated = script.replace("/workspace", root.as_posix())
    py_cmd = subprocess.list2cmdline([sys.executable])
    translated = re.sub(
        r"(?<![\w./\\-])(?:python3|python)(?![\w.])",
        lambda _m: py_cmd,
        translated,
    )
    translated = translated.replace("2>&1", "")
    return translated


def _is_shell_test_expression(script: str) -> bool:
    return bool(re.search(r"\btest\s+-[edf]\s+/workspace/", script))


def _cmd_version(cmd: list[str | None]) -> str:
    if not cmd or not cmd[0]:
        return "missing"
    try:
        completed = subprocess.run(
            [str(x) for x in cmd],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "missing"
    return (completed.stdout or completed.stderr or "missing").splitlines()[0].strip()


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BenchRunConfig:
    case_path: Path
    model: str
    model_label: str
    backend_kind: str
    strategy: str = "hybrid"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    keep_workspace: bool = False
    work_root: Path | None = None


async def run_case(config: BenchRunConfig) -> dict[str, Any]:
    case = load_case(config.case_path)
    workspace_ctx = (
        _kept_workspace(config)
        if config.keep_workspace
        else tempfile.TemporaryDirectory(prefix=f"coder-bench-{case['name']}-")
    )
    with workspace_ctx as raw_workspace:
        workspace = Path(raw_workspace)
        materialize_workspace(workspace, case.get("workspace", {}).get("files") or {})
        before = snapshot_workspace(workspace)
        workspace_id = f"bench-{uuid.uuid4().hex[:12]}"
        manager = LocalWorkspaceManager(workspace, workspace_id)
        backend, close_backend = await _make_backend(config, case)
        try:
            handler = CoderHandler(
                backend,
                session_id=f"{workspace_id}-{config.model_label}",
                container_manager=manager,
                workspace_id=workspace_id,
                coder_strategy=config.strategy,
            )
            request = InternalChatRequest(
                model=config.model,
                messages=[Message(role="user", content=str(case["user_message"]))],
                stream=True,
                temperature=0.0,
            )
            chunks = []
            started = time.time()
            async for chunk in handler.handle_stream(request):
                chunks.append(chunk)
            elapsed = time.time() - started
        finally:
            await close_backend()

        after = snapshot_workspace(workspace)
        result = _result_from_chunks(
            case=case,
            case_path=config.case_path,
            chunks=chunks,
            before=before,
            after=after,
            elapsed=elapsed,
            config=config,
            workspace=workspace if config.keep_workspace else None,
        )
        result["verification"] = run_post_verification(case, workspace)
        result["verification_output"] = result["verification"].pop("output", "")
        result["assertion_failures"] = apply_assertions(
            result,
            case.get("assertions") or [],
        )
        result["outcome"] = classify_outcome(result)
        return result


class _kept_workspace:
    def __init__(self, config: BenchRunConfig) -> None:
        root = config.work_root or DEFAULT_OUT_DIR / "workspaces"
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{config.case_path.stem}-{config.model_label}-{uuid.uuid4().hex[:6]}"

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=False)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


async def _make_backend(
    config: BenchRunConfig,
    case: dict[str, Any],
) -> tuple[ModelBackend, Any]:
    if config.backend_kind == "scripted":
        return ScriptedBackend.from_case(case, model=config.model), _noop_async

    timeout = httpx.Timeout(180.0, connect=10.0)
    client = httpx.AsyncClient(timeout=timeout)
    if config.backend_kind == "ollama":
        return OllamaBackend(client, config.base_url), client.aclose
    if config.backend_kind == "openai":
        base = config.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        return OpenAIBackend(client, base, config.api_key or None), client.aclose
    raise ValueError(f"unknown backend kind: {config.backend_kind}")


async def _noop_async() -> None:
    return None


def _result_from_chunks(
    *,
    case: dict[str, Any],
    case_path: Path,
    chunks: list[InternalStreamChunk],
    before: dict[str, str],
    after: dict[str, str],
    elapsed: float,
    config: BenchRunConfig,
    workspace: Path | None,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    tools: list[str] = []
    tier: str | None = None
    strategy = config.strategy
    termination_reason = ""
    iterations = 0
    tokens_used = 0
    assistant_text: list[str] = []

    for chunk in chunks:
        if chunk.content_delta:
            assistant_text.append(chunk.content_delta)
        if chunk.usage:
            tokens_used += chunk.usage.total_tokens
        meta = chunk.augmentum or {}
        status = meta.get("status")
        if not status:
            continue
        event = _summarize_meta_event(meta)
        events.append(event)
        if status == "tier_classified":
            tier = str(meta.get("tier") or "")
        elif status == "strategy":
            strategy = str(meta.get("strategy") or strategy)
        elif status == "tool_call":
            tool = ((meta.get("tool_call") or {}).get("tool") or "")
            if tool:
                tools.append(str(tool))
        elif status == "budget":
            budget = meta.get("budget") or {}
            iterations = max(iterations, int(budget.get("iteration") or 0))
        elif status == "complete":
            termination_reason = str(meta.get("termination_reason") or "")
            iterations = int(meta.get("iterations_used") or iterations)

    return {
        "case": case.get("name"),
        "case_path": str(case_path.relative_to(ROOT)),
        "tier": tier,
        "declared_tier": case.get("tier"),
        "strategy": strategy,
        "backend": config.backend_kind,
        "model": config.model,
        "model_label": config.model_label,
        "elapsed_seconds": round(elapsed, 3),
        "files": after,
        "files_changed": changed_files(before, after),
        "tools_used": tools,
        "iterations": iterations,
        "tokens_used": tokens_used,
        "termination_reason": termination_reason or "unknown",
        "assistant_text": "".join(assistant_text).strip(),
        "events": events,
        "workspace": str(workspace) if workspace else "",
    }


def _summarize_meta_event(meta: dict[str, Any]) -> dict[str, Any]:
    status = str(meta.get("status") or "")
    event: dict[str, Any] = {
        "phase": meta.get("phase"),
        "status": status,
    }
    for key in (
        "tier", "tier_reason", "strategy", "termination_reason",
        "iterations_used", "iteration", "summary_chars",
    ):
        if key in meta:
            event[key] = meta[key]
    if "tool_call" in meta:
        tc = meta["tool_call"] or {}
        event["tool"] = tc.get("tool")
        event["input"] = tc.get("input")
    if "tool_result" in meta:
        tr = meta["tool_result"] or {}
        event["tool"] = tr.get("tool")
        event["success"] = tr.get("success")
        event["preview"] = tr.get("output_preview")
        for key in ("denied", "preemptive_refusal", "workspace_tree_guard"):
            if key in tr:
                event[key] = tr[key]
    if "budget" in meta:
        budget = meta["budget"] or {}
        event["iteration"] = budget.get("iteration")
        event["remaining"] = budget.get("iterations_remaining")
    if "reminder" in meta:
        event["reminder"] = str(meta["reminder"])[:300]
    return event


def run_post_verification(case: dict[str, Any], workspace: Path) -> dict[str, Any]:
    specs = case.get("assertions") or []
    wants_verification = any(
        spec.get("property") == "verification_gate_passed" for spec in specs
    )
    command = (
        (case.get("verification") or {}).get("command")
        if isinstance(case.get("verification"), dict)
        else ""
    )
    if not command and wants_verification and list(workspace.glob("test_*.py")):
        command = f"{subprocess.list2cmdline([sys.executable])} -m pytest -q"
    if not command:
        return {"tests": True, "output": ""}
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return {"tests": completed.returncode == 0, "output": output[-4000:]}
    except Exception as exc:
        return {"tests": False, "output": str(exc)}


def classify_outcome(result: dict[str, Any]) -> str:
    failures = result.get("assertion_failures") or []
    verification = result.get("verification") or {}
    tests_ok = all(bool(v) for v in verification.values()) if verification else True
    if not failures and tests_ok:
        return "perfect"
    reason = str(result.get("termination_reason") or "")
    tools_used = result.get("tools_used") or []
    if reason in {"backend_error"}:
        return "backend_error"
    if "break" in reason or "streak" in reason or reason in {"max_iterations"}:
        return "loop_stopped"
    if not tools_used or reason.startswith("model_stop"):
        return "ended_early"
    return "partial"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_reports(results: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for result in results:
        name = f"{result['case']}-{result['model_label']}.json"
        (out_dir / _safe_name(name)).write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    (out_dir / "summary.md").write_text(render_markdown_summary(results), encoding="utf-8")


def render_markdown_summary(results: list[dict[str, Any]]) -> str:
    lines = ["# Hybrid Coder Bench", ""]
    totals: dict[str, int] = {}
    for result in results:
        totals[result["outcome"]] = totals.get(result["outcome"], 0) + 1
    lines.append("## Totals")
    for outcome, count in sorted(totals.items()):
        lines.append(f"- {outcome}: {count}")
    lines.append("")
    lines.append("## Runs")
    for result in results:
        lines.extend(_render_one_result(result))
    return "\n".join(lines).rstrip() + "\n"


def _render_one_result(result: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"### {result['case']} / {result['model_label']}",
        "",
        f"- outcome: **{result['outcome']}**",
        f"- model: `{result['model']}` via `{result['backend']}`",
        f"- tier: `{result.get('tier')}` declared `{result.get('declared_tier')}`",
        f"- strategy: `{result.get('strategy')}`",
        f"- termination: `{result.get('termination_reason')}`",
        f"- iterations: `{result.get('iterations')}`",
        f"- changed files: {', '.join(result.get('files_changed') or ['(none)'])}",
    ]
    failures = result.get("assertion_failures") or []
    if failures:
        lines.append("- assertion failures:")
        lines.extend(f"  - {failure}" for failure in failures)
    verification = result.get("verification") or {}
    if verification and not all(verification.values()):
        lines.append("- verification failed")
    lines.append("")
    lines.append("Step trace:")
    for event in result.get("events") or []:
        rendered = _render_event(event)
        if rendered:
            lines.append(f"- {rendered}")
    if result.get("assistant_text"):
        text = str(result["assistant_text"]).strip().replace("\n", " ")
        lines.append("")
        lines.append(f"Final prose: {text[:500]}")
    return lines


def _render_event(event: dict[str, Any]) -> str:
    status = event.get("status")
    phase = event.get("phase")
    if status == "tool_call":
        return f"{phase}: call `{event.get('tool')}` {json.dumps(event.get('input') or {})}"
    if status == "tool_result":
        marker = "ok" if event.get("success") else "fail"
        preview = str(event.get("preview") or "").replace("\n", " ")[:180]
        return f"{phase}: `{event.get('tool')}` {marker}: {preview}"
    if status == "tier_classified":
        return f"{phase}: tier `{event.get('tier')}` ({event.get('tier_reason')})"
    if status == "strategy":
        return f"{phase}: strategy `{event.get('strategy')}`"
    if status == "budget":
        return f"{phase}: iteration {event.get('iteration')} remaining {event.get('remaining')}"
    if status == "complete":
        return f"{phase}: complete `{event.get('termination_reason')}`"
    if status and any(word in status for word in ("nudge", "break", "error")):
        return f"{phase}: {status}"
    return ""


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_model_specs(values: list[str] | None, *, fallback: str) -> list[tuple[str, str]]:
    if not values:
        return [("default", fallback)]
    pairs: list[tuple[str, str]] = []
    for value in values:
        if "=" in value:
            label, model = value.split("=", 1)
        else:
            model = value
            label = value.rsplit("/", 1)[-1].replace(":", "-")
        pairs.append((label.strip() or model, model.strip()))
    return pairs


async def list_models(*, backend: str, base_url: str, api_key: str) -> bool:
    config = BenchRunConfig(
        case_path=CASES_DIR / "reflex" / "case_add_missing_import.yaml",
        model="",
        model_label="list",
        backend_kind=backend,
        base_url=base_url,
        api_key=api_key,
    )
    model_backend, close = await _make_backend(config, {"responses": []})
    try:
        try:
            models = await model_backend.list_models()
        except Exception as exc:
            print(f"Could not list models from {base_url}: {exc}", file=sys.stderr)
            return False
        for model in models:
            size = f" {model.size}" if model.size else ""
            print(f"{model.name}{size}")
        return True
    finally:
        await close()


async def amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Hybrid Coder eval bench cases.")
    parser.add_argument("--backend", choices=["scripted", "ollama", "openai"], default="scripted")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", action="append", help="Model or label=model. Repeat for small/medium.")
    parser.add_argument("--case", default="", help="Substring filter over case path/name.")
    parser.add_argument("--strategy", default="hybrid", choices=["hybrid", "native", "canonical", "legacy"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args(argv)

    if args.list_models:
        ok = await list_models(backend=args.backend, base_url=args.base_url, api_key=args.api_key)
        return 0 if ok else 1

    cases = discover_cases(args.case)
    if not cases:
        print("No coder eval cases matched.", file=sys.stderr)
        return 2

    fallback_model = "scripted-coder-eval" if args.backend == "scripted" else "llama3.1:8b"
    models = parse_model_specs(args.model, fallback=fallback_model)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or DEFAULT_OUT_DIR / stamp
    results: list[dict[str, Any]] = []
    for label, model in models:
        for case_path in cases:
            config = BenchRunConfig(
                case_path=case_path,
                model=model,
                model_label=label,
                backend_kind=args.backend,
                strategy=args.strategy,
                base_url=args.base_url,
                api_key=args.api_key,
                keep_workspace=args.keep_workspaces,
                work_root=out_dir / "workspaces",
            )
            print(f"running {case_path.relative_to(CASES_DIR)} on {label} ({model})")
            result = await run_case(config)
            results.append(result)
            print(
                f"  -> {result['outcome']} "
                f"term={result['termination_reason']} "
                f"iters={result['iterations']} tools={len(result['tools_used'])}",
            )

    try:
        write_reports(results, out_dir)
    except PermissionError as exc:
        print(f"\nCould not write bench report to {out_dir}: {exc}", file=sys.stderr)
        print(render_markdown_summary(results))
    else:
        print(f"\nWrote bench report: {out_dir / 'summary.md'}")
    return 1 if any(r["outcome"] != "perfect" for r in results) else 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
