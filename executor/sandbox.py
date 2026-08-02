"""Sandboxed Python code execution via subprocess."""

from __future__ import annotations

import ast
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

# Output size limits
MAX_STDOUT = 1_048_576  # 1MB
MAX_STDERR = 262_144    # 256KB

# Default resource limits (Linux only)
DEFAULT_CPU_LIMIT = 30     # seconds
DEFAULT_MEM_LIMIT = 256    # MB
DEFAULT_PIDS_LIMIT = 10
DEFAULT_FSIZE_LIMIT = 50   # MB


def _extract_last_expression(code: str) -> tuple[str, str | None]:
    """Parse code and extract the last expression for Jupyter-like capture.

    Returns (modified_code, capture_var_name) where modified_code has the last
    expression assigned to a variable, or (original_code, None) if not applicable.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None

    if not tree.body:
        return code, None

    last_node = tree.body[-1]
    if not isinstance(last_node, ast.Expr):
        return code, None

    # Replace last expression with assignment to __result__
    capture_var = "__augmentum_result__"
    assign = ast.Assign(
        targets=[ast.Name(id=capture_var, ctx=ast.Store())],
        value=last_node.value,
        lineno=last_node.lineno,
        col_offset=last_node.col_offset,
    )
    tree.body[-1] = ast.fix_missing_locations(assign)

    return ast.unparse(tree), capture_var


def _build_runner_script(code: str, capture_var: str | None) -> str:
    """Build the Python script that will run inside the subprocess."""
    lines = [
        "import sys, json",
        "",
    ]

    # Set resource limits on Linux
    if sys.platform != "win32":
        lines.extend([
            "import resource",
            f"resource.setrlimit(resource.RLIMIT_CPU, ({DEFAULT_CPU_LIMIT}, {DEFAULT_CPU_LIMIT}))",
            f"resource.setrlimit(resource.RLIMIT_AS, ({DEFAULT_MEM_LIMIT * 1024 * 1024}, {DEFAULT_MEM_LIMIT * 1024 * 1024}))",
            f"resource.setrlimit(resource.RLIMIT_NPROC, ({DEFAULT_PIDS_LIMIT}, {DEFAULT_PIDS_LIMIT}))",
            f"resource.setrlimit(resource.RLIMIT_FSIZE, ({DEFAULT_FSIZE_LIMIT * 1024 * 1024}, {DEFAULT_FSIZE_LIMIT * 1024 * 1024}))",
            "",
        ])

    lines.append("try:")
    # Indent user code
    for line in code.splitlines():
        lines.append(f"    {line}")

    # Capture result
    if capture_var:
        lines.extend([
            f"    __rv = {capture_var}",
            "    if __rv is not None:",
            '        print(f"\\n__RESULT__{json.dumps(repr(__rv))}__RESULT__")',
        ])

    lines.extend([
        "except Exception as __e:",
        "    import traceback",
        '    print(traceback.format_exc(), file=sys.stderr)',
        "    sys.exit(1)",
    ])

    return "\n".join(lines)


def execute_code(code: str, timeout: int = 30) -> dict:
    """Execute Python code in a subprocess sandbox.

    Returns dict with: success, stdout, stderr, return_value, error, metrics.
    """
    start_time = time.monotonic()
    modified_code, capture_var = _extract_last_expression(code)
    runner_script = _build_runner_script(modified_code, capture_var)

    # Write script to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/work" if os.path.isdir("/work") else None
    ) as f:
        f.write(runner_script)
        script_path = f.name

    try:
        kwargs = {}
        if sys.platform != "win32":
            kwargs["preexec_fn"] = os.setsid

        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/work" if os.path.isdir("/work") else None,
            **kwargs,
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the entire process group
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
            proc.wait()
            return {
                "success": False,
                "error": f"Execution timed out after {timeout}s",
                "stdout": "",
                "stderr": "",
                "return_value": None,
                "metrics": {"elapsed_seconds": timeout},
            }

        stdout = stdout_bytes.decode("utf-8", errors="replace")[:MAX_STDOUT]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:MAX_STDERR]

        # Extract captured return value
        return_value = None
        if "__RESULT__" in stdout:
            parts = stdout.split("__RESULT__")
            if len(parts) >= 3:
                with contextlib.suppress(json.JSONDecodeError, IndexError):
                    return_value = json.loads(parts[1])
                # Remove the result line from stdout
                stdout = stdout[: stdout.index("\n__RESULT__")]

        elapsed = time.monotonic() - start_time

        return {
            "success": proc.returncode == 0,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "return_value": return_value,
            "error": stderr.strip() if proc.returncode != 0 else None,
            "metrics": {"elapsed_seconds": round(elapsed, 3)},
        }

    finally:
        with contextlib.suppress(OSError):
            os.unlink(script_path)
