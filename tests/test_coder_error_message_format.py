"""Pin the user-visible coder iteration-error text shape.

The old `[Agent error on iteration N]` chunk dropped all error detail,
forcing the user to grep container logs. This test set ensures the
formatter keeps surfacing the provider's actual reply text (status code
+ truncated message) so future edits can't silently regress to the bare
form.
"""
from __future__ import annotations

from augmentum.modes.coder.phase_act import _format_iteration_error


def test_includes_status_code_and_message():
    out = _format_iteration_error(
        iteration=5,
        error_kind="permanent",
        error_status=400,
        error_message=(
            "Backend returned 400: Messages with role 'tool' must be a "
            "response to a preceding message with 'tool_calls'"
        ),
    )
    assert "iteration 5" in out
    assert "HTTP 400" in out
    assert "Messages with role 'tool'" in out


def test_strips_redundant_backend_prefix():
    out = _format_iteration_error(
        iteration=2,
        error_kind="permanent",
        error_status=429,
        error_message="Backend returned 429: Tokens per minute limit exceeded",
    )
    # Status is already in the HTTP label; repeating "Backend returned 429:"
    # is just noise. The detail after the colon must survive.
    assert "Backend returned 429:" not in out
    assert "Tokens per minute limit exceeded" in out
    assert "HTTP 429" in out


def test_truncates_long_messages():
    long_detail = "x" * 600
    out = _format_iteration_error(
        iteration=1, error_kind="permanent", error_status=400,
        error_message=long_detail,
    )
    # Hard cap at 240 chars + ellipsis. Prevents one ugly backend reply
    # from flooding the conversation surface.
    assert "…" in out
    assert len(out) < 500


def test_falls_back_to_kind_label_when_no_status():
    # Network blip — httpx exceptions don't carry an HTTP status. We
    # still want the user to know whether retrying is worth it.
    out = _format_iteration_error(
        iteration=3, error_kind="transient", error_status=None,
        error_message="ReadTimeout: timed out waiting for response",
    )
    assert "transient" in out
    assert "ReadTimeout" in out


def test_handles_empty_message():
    # Defensive: even with no exception text we still emit the iteration
    # number and the kind so the chunk isn't completely opaque.
    out = _format_iteration_error(
        iteration=4, error_kind="permanent", error_status=None,
        error_message="",
    )
    assert "iteration 4" in out
    assert "permanent" in out


def test_quota_kind_uses_quota_specific_label():
    # ``quota`` errors phrase differently from ``permanent`` — the
    # user's request isn't broken, they hit a per-minute cap. The label
    # has to read as "wait or switch model" rather than "your request
    # is malformed".
    out = _format_iteration_error(
        iteration=2, error_kind="quota", error_status=429,
        error_message="Tokens per minute limit exceeded",
    )
    assert "iteration 2" in out
    assert "HTTP 429" in out
    assert "token quota exceeded" in out
    assert "Tokens per minute limit exceeded" in out
