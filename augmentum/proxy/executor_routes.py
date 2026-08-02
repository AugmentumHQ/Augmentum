"""Proxy endpoint for browser-initiated Python code execution."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from augmentum.config import settings
from augmentum.tools.python_exec import _BLOCKED_PATTERNS
from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import sanitize_error_detail

log = get_logger(__name__)

router = APIRouter(tags=["executor"])

_MAX_OUTPUT_BYTES = 100_000  # 100 KB
_MAX_TIMEOUT = 60


class ExecuteRequest(BaseModel):
    code: str
    timeout: int = Field(default=30, ge=1)


@router.post("/api/execute")
async def execute_code(body: ExecuteRequest, request: Request) -> JSONResponse:
    """Run Python code in the sandboxed executor container."""

    code = body.code
    timeout = min(body.timeout, _MAX_TIMEOUT)

    if not code.strip():
        return JSONResponse({"error": "No code provided"}, status_code=400)

    # Server-side static analysis — reject dangerous patterns before forwarding.
    for pattern, reason in _BLOCKED_PATTERNS:
        if pattern.search(code):
            return JSONResponse(
                {"error": f"Code rejected: {reason}"},
                status_code=400,
            )

    http_client = getattr(request.app.state, "http_client", None)
    if http_client is None:
        log.error("executor_route_no_http_client")
        return JSONResponse(
            {"error": "Server misconfigured: no HTTP client"},
            status_code=503,
        )

    base_url = settings.executor_base_url.rstrip("/")

    try:
        resp = await http_client.post(
            f"{base_url}/execute",
            json={"code": code, "timeout": timeout},
            timeout=float(timeout) + 10.0,
        )
    except Exception as exc:
        log.warning("executor_unreachable", error=str(exc))
        return JSONResponse(
            {"error": f"Executor unreachable: {sanitize_error_detail(str(exc))}"},
            status_code=502,
        )

    try:
        data = resp.json()
    except Exception:
        log.warning("executor_bad_response", status=resp.status_code)
        return JSONResponse(
            {"error": "Executor returned invalid JSON"},
            status_code=503,
        )

    if resp.status_code >= 400:
        error_msg = data.get("error", "Executor error")
        traceback_msg = data.get("traceback", "")
        log.warning("executor_error", status=resp.status_code, error=error_msg[:200])
        return JSONResponse(
            {
                "success": False,
                "error": sanitize_error_detail(error_msg[:2000]),
                "stderr": sanitize_error_detail(traceback_msg[:2000]) if traceback_msg else "",
                "stdout": "",
                "return_value": None,
            },
            status_code=200,  # Return 200 so the UI can parse and display the error
        )

    # Truncate oversized fields to 100 KB each.
    for key in ("stdout", "stderr"):
        val = data.get(key)
        if isinstance(val, str) and len(val) > _MAX_OUTPUT_BYTES:
            data[key] = val[:_MAX_OUTPUT_BYTES] + f"\n... (truncated at {_MAX_OUTPUT_BYTES} bytes)"

    return JSONResponse(data)
