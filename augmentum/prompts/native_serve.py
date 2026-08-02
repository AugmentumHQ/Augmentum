"""Native-primer serving — the F7 train==serve contract, applied at egress.

Models trained on the bare primer (Alethia seed: ``:C`` + minimal live state)
must be SERVED that primer, not the platform's full mode prompt — a format
mismatch silently degrades the model (see ``prompts/primer.py``).

This module applies :func:`augmentum.prompts.primer.build_primer` at the ONE
place every mode converges — the backend ``chat``/``chat_stream`` boundary —
via the same wrapper pattern as ``trace_context.install_capture_hook``.

Gating: inert unless ``settings.native_primer_models`` names the model
(comma-separated case-insensitive substrings, e.g. ``alethia``). The user
opts a model in explicitly; nothing is auto-selected.

Surface resolution: turn entry points publish their mode via
:func:`set_current_surface` (wired inside ``capture_turn``/``begin_capture``
BEFORE the capture gate, so it works with capture off). No published surface
falls back to ``chat`` (``:C``).
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime
from typing import Any

from augmentum.config import settings
from augmentum.prompts.primer import build_primer
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_SURFACE: ContextVar[str] = ContextVar("native_primer_surface", default="")

_HOOK_SENTINEL = "_native_primer_hook_installed"


def set_current_surface(mode: str) -> None:
    """Publish the active turn's mode/surface (ambient, task-scoped)."""
    if mode:
        _SURFACE.set(mode)


def current_surface() -> str:
    return _SURFACE.get()


def _is_primer_model(model: str) -> bool:
    patterns = (getattr(settings, "native_primer_models", "") or "").strip()
    if not patterns or not model:
        return False
    low = model.lower()
    return any(p.strip() and p.strip().lower() in low for p in patterns.split(","))


def apply_primer(request: Any) -> bool:
    """Replace the assembled system prompt with the trained primer.

    Mutates ``request.messages`` in place: all system messages are removed and
    one primer system message is inserted at index 0. Tool NAMES from the
    request ride the primer's ``tools:`` line (the trained format); the full
    schemas still travel in ``request.tools`` for the template/executor.
    Returns True when applied.
    """
    if not _is_primer_model(getattr(request, "model", "") or ""):
        return False
    messages = getattr(request, "messages", None)
    if not messages:
        return False
    surface = current_surface() or "chat"
    tool_names: list[str] = []
    for t in getattr(request, "tools", None) or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else ""
        if name:
            tool_names.append(name)
    primer = build_primer(
        surface,
        now=datetime.now().strftime("%Y-%m-%d %H:%M"),
        tools=tool_names,
    )
    msg_cls = type(messages[0])
    kept = [m for m in messages if getattr(m, "role", "") != "system"]
    messages[:] = [msg_cls(role="system", content=primer), *kept]
    log.info(
        "native_primer_applied",
        model=getattr(request, "model", ""),
        surface=surface,
        tools=len(tool_names),
    )
    return True


def install_primer_hook(backend: Any) -> None:
    """Wrap ``backend.chat``/``chat_stream`` to apply the primer at egress.

    Install AFTER ``install_capture_hook`` so this wrapper is outermost:
    the primer mutates the request BEFORE capture snapshots it, keeping
    captured traces faithful to what the model actually received.
    """
    if getattr(backend, _HOOK_SENTINEL, False):
        return
    orig_chat = backend.chat
    orig_chat_stream = backend.chat_stream

    async def _chat(request: Any) -> Any:
        try:
            apply_primer(request)
        except Exception:
            log.warning("native_primer_apply_failed", exc_info=True)
        return await orig_chat(request)

    async def _chat_stream(request: Any) -> Any:
        try:
            apply_primer(request)
        except Exception:
            log.warning("native_primer_apply_failed", exc_info=True)
        async for chunk in orig_chat_stream(request):
            yield chunk

    backend.chat = _chat  # type: ignore[attr-defined]
    backend.chat_stream = _chat_stream  # type: ignore[attr-defined]
    setattr(backend, _HOOK_SENTINEL, True)
