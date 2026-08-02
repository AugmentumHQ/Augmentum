"""Structured backend errors + rate-limit reset parsing.

Step 2 of the first-class load-balancer work. Adapters raise a
:class:`BackendError` (a plain ``RuntimeError`` subclass, so every existing
``except RuntimeError`` still catches it) carrying a ``retry_after`` derived
from the provider's own signal — the ``Retry-After`` / ``X-RateLimit-Reset``
headers for OpenAI-compat + Anthropic, the ``retryDelay`` in the body for
Gemini. The balancer facade reads ``exc.retry_after`` to cool a member for
exactly the right window instead of a blind default.

``parse_retry_after`` is header-shape tolerant on purpose: providers disagree
(``Retry-After`` seconds vs HTTP-date; reset as epoch vs delta vs ``"6m0s"``
duration vs RFC3339 timestamp). Anything it can't confidently read → None, and
the caller falls back to the blind exponential cooldown.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

# Reset windows longer than a day are almost certainly a misparse (epoch vs
# delta confusion, a bad date) — reject rather than bench a member for a week.
_MAX_REASONABLE_S = 86400.0

# Body hint, e.g. Gemini "Please retry in 56.016388495s." / '"retryDelay": "14s"',
# or "retry after 30 seconds".
_BODY_RETRY_RE = re.compile(
    r'retry[\s_"\'-]*(?:after|delay|in)?[\s:="\']*'
    r'([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|seconds?)\b',
    re.IGNORECASE,
)
# Duration strings some gateways use for reset, e.g. "6m0s", "1s", "500ms".
_DURATION_RE = re.compile(r'^(?:(\d+)m)?(\d+(?:\.\d+)?)s$', re.IGNORECASE)


class BackendError(RuntimeError):
    """A backend request failure carrying structured rate-limit metadata.

    Subclasses ``RuntimeError`` so existing catch sites are unaffected; the
    message is kept identical to the old ``raise RuntimeError(...)`` text so
    any downstream substring classification (429/503/etc.) still works.
    """

    def __init__(
        self, message: str, *, retry_after: float | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.status = status


def _to_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _http_date_delta(value: str) -> float | None:
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (dt - datetime.now(UTC)).total_seconds()


def _rfc3339_delta(value: str) -> float | None:
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (dt - datetime.now(UTC)).total_seconds()


def _reset_to_delta(value: object) -> float | None:
    """Interpret an ``X-RateLimit-Reset``-style value as seconds-from-now."""
    if not value:
        return None
    s = str(value).strip()
    m = _DURATION_RE.match(s)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2))
    n = _to_float(s)
    if n is not None:
        # Large absolute values are epoch seconds; small ones are a delta.
        return max(0.0, n - time.time()) if n > 1_000_000 else n
    return _rfc3339_delta(s)  # e.g. Anthropic's RFC3339 reset timestamp


def retry_after_from_body(body: str) -> float | None:
    m = _BODY_RETRY_RE.search(body or "")
    if not m:
        return None
    v = _to_float(m.group(1))
    return v if v is not None and 0 < v <= _MAX_REASONABLE_S else None


def parse_retry_after(headers: object = None, body: str = "") -> float | None:
    """Best-effort seconds-until-reset from response headers, then body.

    Priority: ``Retry-After`` → ``*-RateLimit-Reset`` family → body hint.
    Returns None when no trustworthy signal is present.
    """
    get = getattr(headers, "get", None)
    if callable(get):
        ra = get("retry-after")
        if ra:
            n = _to_float(ra)
            secs = n if n is not None else _http_date_delta(str(ra))
            if secs is not None and 0 < secs <= _MAX_REASONABLE_S:
                return secs
        for key in (
            "x-ratelimit-reset",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
            "anthropic-ratelimit-requests-reset",
            "anthropic-ratelimit-tokens-reset",
        ):
            secs = _reset_to_delta(get(key))
            if secs is not None and 0 < secs <= _MAX_REASONABLE_S:
                return secs
    return retry_after_from_body(body)
