"""Structured logging configuration using structlog.

Level filtering is delegated to stdlib :mod:`logging` so the level can be
changed at runtime via :func:`set_log_level`. This is what powers the
admin "Log Level" toggle in Settings — flipping it in the UI takes effect
immediately, no restart required.

The previous configuration used ``structlog.make_filtering_bound_logger``
which bakes the level into a class at import-time and caches it per
logger; runtime changes were a no-op for already-imported modules. The
stdlib path doesn't have that problem because every ``log.info(...)``
call resolves the level via the stdlib logger hierarchy on each call.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.dev import ConsoleRenderer, RichTracebackFormatter

_VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

# Exception rendering: NEVER render frame locals.
#
# structlog's default ``ConsoleRenderer`` uses ``RichTracebackFormatter``
# with ``show_locals=True``. On any 500 the global ``generic_error_handler``
# logs ``exc_info=True``; for request handlers like ``import_chats`` the
# frame locals include the full request body (every chat tree, with the
# system prompt repeated per message). Rich's pretty-repr over that is
# tens of seconds of GIL-bound CPU on the main thread — which stalls the
# event loop, starves every aiosqlite worker thread, makes in-flight DB
# writes time out against ``busy_timeout`` ("database is locked"), which
# raises another 500, which renders another giant traceback: a feedback
# loop that freezes the app for minutes at a time. It also dumps prompt
# text (incl. NSFW character cards) to stdout in plaintext.
#
# ``show_locals=False`` keeps the traceback + source context (still useful
# for debugging) but cuts the locals dump. ``max_frames`` trimmed so deep
# ASGI stacks stay readable.
_EXC_FORMATTER = RichTracebackFormatter(
    show_locals=False,
    max_frames=20,
    width=120,
)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog + stdlib so structlog's output flows through
    stdlib's level filter. Call once at startup.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Stdlib root logger — this is the knob that ``set_log_level`` turns.
    # We clear existing handlers in case setup_logging gets called twice
    # (test fixtures, etc.) so we don't end up with duplicate output.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.NOTSET)  # let the logger decide; handler is permissive
    root.addHandler(handler)
    root.setLevel(level)

    # Chatty third-party loggers — silenced one notch above whatever the
    # root level is so they don't dominate stdout in production while
    # still surfacing when the operator flips to DEBUG.
    #
    # The big offender is ``httpx``: every outbound HTTP request emits
    # an INFO line ``HTTP Request: METHOD URL "HTTP/1.1 STATUS"``. Under
    # normal operation that's the bulk of stdout — model-load health
    # probes (500ms poll until /health flips from 503 → 200), provider
    # model-list refreshes, knowledge-pack fetches, every searxng call
    # the recommender fires. Useful for debugging, noise in production.
    #
    # ``httpcore`` mirrors httpx for the underlying transport; same
    # rationale. ``urllib3`` doesn't emit at INFO by default but pin it
    # too so a transitive dep flip doesn't suddenly spam.
    for noisy in ("httpx", "httpcore", "urllib3"):
        # Default behaviour: silenced to WARNING. Operators chasing
        # HTTP-level debugging can override via
        # ``logging.getLogger("httpx").setLevel(logging.DEBUG)`` in a
        # debugger or by bumping LOG_LEVEL=DEBUG which automatically
        # un-silences these (the per-logger filter is removed below).
        if level <= logging.DEBUG:
            logging.getLogger(noisy).setLevel(logging.NOTSET)  # inherit root
        else:
            logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,  # honour the stdlib level — KEY for runtime changes
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            ConsoleRenderer(exception_formatter=_EXC_FORMATTER),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def set_log_level(level_name: str) -> str:
    """Change the live log level for all loggers. Returns the normalized
    level name on success.

    Raises ``ValueError`` for unknown level names. Used by the admin UI
    toggle in Settings → Diagnostics; the change applies immediately to
    every emitted log line because filtering happens via stdlib at call
    time, not at logger-construction time.
    """
    level_upper = level_name.upper()
    if level_upper not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid log level: {level_name!r} (allowed: {', '.join(_VALID_LEVELS)})"
        )
    new_level = getattr(logging, level_upper)
    logging.getLogger().setLevel(new_level)
    # Keep the noisy-library silencing in lockstep with the new level:
    # at DEBUG they un-silence (inherit root), at INFO+ they go back to
    # WARNING. Same policy as ``setup_logging`` — see comment there.
    for noisy in ("httpx", "httpcore", "urllib3"):
        if new_level <= logging.DEBUG:
            logging.getLogger(noisy).setLevel(logging.NOTSET)
        else:
            logging.getLogger(noisy).setLevel(logging.WARNING)
    return level_upper


def get_log_level() -> str:
    """Return the current root logger level as a string (DEBUG/INFO/...)."""
    return logging.getLevelName(logging.getLogger().getEffectiveLevel())


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a bound logger instance."""
    return structlog.get_logger(name)
