"""Per-minute token-quota 429 classification.

A 429 from a per-minute token cap won't clear inside the 17s in-loop
retry budget — quota windows reset on the minute boundary. The legacy
``transient`` classification burned the full budget and then surfaced
the error anyway. The ``quota`` kind bails after the first attempt and
shows the user a quota-specific hint.
"""
from __future__ import annotations

from augmentum.modes.coder.handler import _classify_backend_error


def _runtime_error(body: str) -> Exception:
    # Match the shape openai_compat raises:
    # ``RuntimeError("Backend returned 429: <body>")``
    return RuntimeError(f"Backend returned 429: {body}")


def test_chatgpt_bridge_tpm_error_is_quota():
    # Exact shape from 2026-06-01T01:37 logs.
    exc = _runtime_error(
        '{"message":"Tokens per minute limit exceeded - too many tokens '
        'processed.","type":"too_many_tokens_error","param":"quota",'
        '"code":"token_quota_exceeded"}'
    )
    kind, status = _classify_backend_error(exc)
    assert kind == "quota"
    assert status == 429


def test_lowercase_underscore_form_is_quota():
    # Anthropic / some compat layers use the underscore form.
    exc = _runtime_error('{"type":"rate_limit_error","code":"tokens_per_minute"}')
    kind, status = _classify_backend_error(exc)
    assert kind == "quota"
    assert status == 429


def test_qps_surge_429_still_transient():
    # Generic queue-surge 429 with no quota markers — should still
    # retry through the 17s budget because the next 2-5s usually clears
    # it.
    exc = _runtime_error('{"message":"Rate limit, queue full","code":"rate_limit"}')
    kind, status = _classify_backend_error(exc)
    assert kind == "transient"
    assert status == 429


def test_403_still_permanent():
    # Sanity: the new quota branch only intercepts 429s, not other 4xx.
    exc = RuntimeError("Backend returned 403: forbidden")
    kind, status = _classify_backend_error(exc)
    assert kind == "permanent"
    assert status == 403


def test_503_still_transient():
    exc = RuntimeError("Backend returned 503: upstream overloaded")
    kind, status = _classify_backend_error(exc)
    assert kind == "transient"
    assert status == 503
