"""OpenAI-compatible error-envelope normalization for the ``/v1`` surface.

Most ``/v1`` handlers already emit the OpenAI ``{"error": {...}}`` envelope,
but a few (notably the audio TTS/STT routes) ``raise HTTPException``, which
FastAPI renders as ``{"detail": "..."}`` — a shape OpenAI SDK clients don't
recognize, so their error handling breaks. These handlers normalize EVERY
error on a ``/v1/*`` path to the OpenAI envelope, making the whole compat
surface consistent. Non-``/v1`` paths delegate to FastAPI's own defaults, so
the rest of the app is untouched.

Wire it up with :func:`register_openai_compat_error_handlers` in ``create_app``.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler as _default_http_handler,
)
from fastapi.exception_handlers import (
    request_validation_exception_handler as _default_validation_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


def openai_error_type(status: int) -> str:
    """Map an HTTP status code to an OpenAI ``error.type`` value."""
    if status == 401:
        return "authentication_error"
    if status == 403:
        return "permission_error"
    if status == 404:
        return "not_found_error"
    if status == 429:
        return "rate_limit_error"
    if status >= 500:
        return "server_error"
    # 400, 413, 422 and any other 4xx
    return "invalid_request_error"


def openai_error_envelope(
    status: int, message: str, *, param: str | None = None, code: str | None = None
) -> dict[str, Any]:
    """Build the canonical OpenAI error body (matches ``_openai_image_error``)."""
    return {
        "error": {
            "message": message,
            "type": openai_error_type(status),
            "param": param,
            "code": code,
        }
    }


def _is_v1(request: Request) -> bool:
    return request.url.path.startswith("/v1/")


async def _http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    if not _is_v1(request):
        return await _default_http_handler(request, exc)
    detail = exc.detail
    message = (
        detail if isinstance(detail, str)
        else (str(detail) if detail is not None else "error")
    )
    return JSONResponse(
        openai_error_envelope(exc.status_code, message),
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    if not _is_v1(request):
        return await _default_validation_handler(request, exc)
    # Summarize the first field error for the OpenAI ``message`` (OpenAI
    # returns 400 invalid_request_error for bad params, not 422).
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    msg = first.get("msg", "invalid request")
    message = f"{loc}: {msg}" if loc else msg
    return JSONResponse(
        openai_error_envelope(400, message, param=loc or None),
        status_code=400,
    )


def register_openai_compat_error_handlers(app: FastAPI) -> None:
    """Install the ``/v1`` error-envelope normalizers on ``app``.

    Overrides FastAPI's built-in ``HTTPException`` + ``RequestValidationError``
    handlers (keyed by exception class), but only changes the response shape
    for ``/v1/*`` paths — everything else falls through to the defaults.
    """
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)


__all__ = [
    "openai_error_type",
    "openai_error_envelope",
    "register_openai_compat_error_handlers",
]
