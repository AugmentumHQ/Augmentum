"""System datetime context line for LLM system prompts."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timezone
from zoneinfo import ZoneInfo

_log = logging.getLogger(__name__)


def _get_local_tz() -> ZoneInfo | timezone:
    """Resolve the server's timezone.

    Priority: config setting > TZ env var > system local timezone > UTC.
    """
    # 1. Check config setting (lazy import to avoid circular)
    try:
        from augmentum.config import settings

        tz_name = settings.timezone
        if tz_name:
            try:
                return ZoneInfo(tz_name)
            except KeyError:
                _log.warning(
                    "timezone_invalid: %r is not a valid IANA timezone", tz_name,
                )
            except Exception:
                _log.warning(
                    "timezone_lookup_failed: %r — is tzdata installed?",
                    tz_name,
                    exc_info=True,
                )
    except ImportError:
        pass

    # 2. Check TZ env var
    tz_env = os.environ.get("TZ")
    if tz_env:
        try:
            return ZoneInfo(tz_env)
        except Exception:
            # ZoneInfoNotFoundError or platform-specific tzdata gap —
            # fall through to system-local.
            pass

    # 3. Fall back to system local timezone
    try:
        now = datetime.now().astimezone()
        if now.tzinfo is not None:
            return now.tzinfo  # type: ignore[return-value]
    except (OSError, ValueError):
        # astimezone() can fail on platforms without tzdata; UTC is the
        # documented contract for any unresolvable case.
        pass
    return UTC


def get_datetime_context() -> str:
    """Return a single-line datetime string for injection into system prompts.

    Uses the configured timezone (setting > TZ env > system local > UTC).
    Example: ``Current date/time: Saturday, 2026-03-07 19:34 UTC-05:00``
    """
    tz = _get_local_tz()
    now = datetime.now(tz)
    offset = now.strftime("%z")  # e.g. "-0500"
    tz_label = f"UTC{offset[:3]}:{offset[3:]}"
    return (
        "<current_time>\n"
        f"Current date: {now.strftime('%A, %B %d, %Y')}. "
        f"Current time: {now.strftime('%H:%M')} {tz_label}. "
        "This is the real current date from the server clock — trust it over your "
        "training cutoff. Never mention this timestamp to the user or say "
        "'the date is in the future.' Just use it naturally when relevant.\n"
        "</current_time>"
    )
