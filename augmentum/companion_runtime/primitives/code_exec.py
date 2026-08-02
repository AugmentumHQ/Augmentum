"""Code-exec primitive — sandboxed Python execution.

Wraps ``augmentum.tools.python_exec.PythonExecTool``. The underlying
tool already enforces a denylist (no os.system/subprocess/exec/eval)
and a timeout, so this adapter just passes through.
"""

from __future__ import annotations

from typing import Any

from augmentum.companion_runtime.primitives.base import (
    PrimitiveBase,
    PrimitiveContext,
    PrimitiveResult,
)
from augmentum.companion_runtime.primitives.registry import PrimitiveRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class CodeExecPrimitive(PrimitiveBase):
    name = "code_exec"
    description = "Execute a Python snippet in a sandboxed environment."

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        code = kwargs.get("code", "")
        if not code:
            return PrimitiveResult(ok=False, error="code_exec: empty code")
        timeout = int(kwargs.get("timeout", 30))

        try:
            from augmentum.tools.python_exec import PythonExecTool
        except Exception as exc:
            return PrimitiveResult(ok=False, error=f"code_exec_import_failed: {exc!s}")

        tool = PythonExecTool()
        try:
            result = await tool.execute(code=code, timeout=timeout)
        except Exception as exc:
            log.exception("code_exec_failed", error=str(exc))
            return PrimitiveResult(ok=False, error=f"code_exec_failed: {exc!s}")

        payload = getattr(result, "result", None) or getattr(result, "content", str(result))
        ok = not bool(getattr(result, "error", None))
        return PrimitiveResult(
            ok=ok,
            payload=payload,
            error=str(getattr(result, "error", "") or ""),
        )


PrimitiveRegistry.register(CodeExecPrimitive)
