"""Tests for the window-scaled tool-result cap in the coder handler.

Covers:
  1. _scaled_output_cap_chars: 50k ceiling for big windows (byte-identical
     default behavior), ~25%-of-window scaling for small limits, and the
     8k floor for tiny limits.
  2. _truncate_output: no-op when the cap is at the 50k default (even for
     oversized text — the tools' own clip owns that regime), both-ends
     notice when a small cap fires, truthful totals.
  3. _append_tool_result_to_history wiring: a 50k-char result entering
     history under a 16384-token compaction limit is clipped to ~16k
     chars; under the default big-window limit it passes byte-identical.
"""

from __future__ import annotations

from types import SimpleNamespace

from augmentum.modes.analytical.tool_calling import ToolCallingTier
from augmentum.modes.coder.handler import (
    _MAX_OUTPUT_CHARS,
    _OUTPUT_CAP_FLOOR_CHARS,
    CoderHandler,
    _scaled_output_cap_chars,
    _truncate_output,
)

# --- #1 cap derivation ----------------------------------------------------


class TestScaledOutputCap:
    def test_big_window_keeps_50k_ceiling(self):
        # 256k derived default (and anything >= 50k tokens) → 50k chars.
        assert _scaled_output_cap_chars(256_000) == _MAX_OUTPUT_CHARS
        assert _scaled_output_cap_chars(200_000) == _MAX_OUTPUT_CHARS
        assert _scaled_output_cap_chars(50_000) == _MAX_OUTPUT_CHARS

    def test_small_window_scales_to_quarter(self):
        # 16384 tokens × 4 chars × 0.25 = 16384 chars.
        assert _scaled_output_cap_chars(16_384) == 16_384

    def test_floor_respected(self):
        # 4k tokens would derive 4k chars — floored at 8k.
        assert _scaled_output_cap_chars(4_000) == _OUTPUT_CAP_FLOOR_CHARS
        assert _scaled_output_cap_chars(1_000) == _OUTPUT_CAP_FLOOR_CHARS

    def test_missing_or_bad_limit_defaults_to_ceiling(self):
        assert _scaled_output_cap_chars(None) == _MAX_OUTPUT_CHARS
        assert _scaled_output_cap_chars(0) == _MAX_OUTPUT_CHARS
        assert _scaled_output_cap_chars("nope") == _MAX_OUTPUT_CHARS  # type: ignore[arg-type]


# --- #2 _truncate_output --------------------------------------------------


class TestTruncateOutput:
    def test_noop_at_default_cap_even_when_over(self):
        # Byte-identical guarantee for big-window deployments: the tools'
        # own 50k clip owns this regime, the handler must not re-clip.
        huge = "x" * (_MAX_OUTPUT_CHARS + 5_000)
        assert _truncate_output(huge, _MAX_OUTPUT_CHARS) is huge

    def test_noop_under_cap(self):
        small = "short content"
        assert _truncate_output(small, 16_384) == small

    def test_small_cap_clips_with_both_ends_notice(self):
        huge = "y" * 50_000
        out = _truncate_output(huge, 16_384)
        assert out.startswith("[TRUNCATED"), "header missing from start"
        assert "showing first 16384 of 50000 chars" in out
        assert "total chars" in out.rsplit("\n", 1)[-1], "trailer missing"
        # Header must survive compaction's 160-char preview.
        assert "TRUNCATED" in out[:160]

    def test_clipped_body_length_matches_cap(self):
        huge = "z" * 50_000
        out = _truncate_output(huge, 16_384)
        assert out.count("z") == 16_384


# --- #3 chokepoint wiring -------------------------------------------------


def _append(limit_tokens: int, output: str) -> str:
    """Run the real _append_tool_result_to_history with a stub self."""
    stub = SimpleNamespace(_coder_compact_token_limit=limit_tokens)
    messages: list = []
    tool_result = SimpleNamespace(success=True, output=output, error=None)
    CoderHandler._append_tool_result_to_history(
        stub, messages, "tid_1", "shell_exec", tool_result,
        ToolCallingTier.NATIVE,
    )
    assert len(messages) == 1
    return messages[0].content


class TestHistoryWiring:
    def test_big_window_byte_identical(self):
        huge = "a" * _MAX_OUTPUT_CHARS
        assert _append(256_000, huge) == huge

    def test_small_window_clips_50k_result(self):
        huge = "b" * _MAX_OUTPUT_CHARS
        content = _append(16_384, huge)
        assert content.startswith("[TRUNCATED")
        assert content.count("b") == 16_384
        assert content.rstrip().endswith("50000 total chars)")

    def test_floor_window_keeps_8k(self):
        huge = "c" * _MAX_OUTPUT_CHARS
        content = _append(2_000, huge)
        # Header prose contains stray 'c's — assert on the run length.
        assert "c" * _OUTPUT_CAP_FLOOR_CHARS in content
        assert "c" * (_OUTPUT_CAP_FLOOR_CHARS + 1) not in content
