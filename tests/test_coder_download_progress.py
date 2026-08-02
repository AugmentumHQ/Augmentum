"""Tests for the active-download liveness detector in the shell/container path.

Problem this feature solves: a silent ``wget -q`` / ``curl -s`` download
emits zero stdout for its entire runtime. The harness's idle-kill timer
(defined in run_command) treats silent stdout as "process hung" and
kills it — even when bytes are actively landing on disk. Observed
2026-04-22 with a 700MB RetroArch archive dying every 60 seconds.

Fix: when the shell tool recognises a download command AND can parse
the target file from its arguments, it passes ``progress_path`` through
to run_command. The idle handler then polls the file's size across the
idle window and treats growth as liveness — emitting a heartbeat into
the returned output and resetting the stall counter. Only files that
STOP growing across the full idle window are killed.

Coverage split:

* **Parser** (``_parse_download_target``) — unit tests for all the
  command shapes we expect to see in the wild.
* **Pattern-tier picker** — confirms that a bare ``wget ...`` without
  any ``_LONG_PATTERNS`` keywords still gets the long tier because it's
  a first-token download tool.
* **byte formatter** — readability is the whole point of the
  heartbeat message.

Live container integration is intentionally NOT tested here — docker
round-trips are expensive and flaky in CI. The parser + tier + format
functions are the risk surface; the `_stat_path_size` / idle-loop glue
is straightforward plumbing once its inputs are right.
"""
from __future__ import annotations

import pytest

from augmentum.coder.containers import _fmt_bytes
from augmentum.coder.tools import (
    _DOWNLOAD_FIRST_TOKENS,
    _parse_download_target,
)


# ---------------------------------------------------------------------------
# _parse_download_target — shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd,expected", [
    # Classic wget invocations
    ("wget https://example.com/file.7z -O /tmp/file.7z",       "/tmp/file.7z"),
    ("wget -q https://example.com/file.7z -O /tmp/file.7z",    "/tmp/file.7z"),
    ("wget -q -O /tmp/file.7z https://example.com/file.7z",    "/tmp/file.7z"),
    # Long-form wget
    ("wget --output-document=/tmp/file.7z https://example.com/file.7z",
                                                                "/tmp/file.7z"),
    ("wget --output-document /tmp/file.7z https://example.com/file.7z",
                                                                "/tmp/file.7z"),
    # curl variants
    ("curl -sSL -o /var/www/html/bundle.7z https://example.com/bundle.7z",
                                                                "/var/www/html/bundle.7z"),
    ("curl --output /tmp/out.bin https://example.com/resource",
                                                                "/tmp/out.bin"),
    ("curl --output=/tmp/out.bin https://example.com/resource",
                                                                "/tmp/out.bin"),
    # aria2
    ("aria2c --out=file.zip https://example.com/file.zip",
                                                                "file.zip"),
    ("aria2c --out file.zip https://example.com/file.zip",
                                                                "file.zip"),
])
def test_parse_download_target_positive(cmd, expected):
    assert _parse_download_target(cmd) == expected


@pytest.mark.parametrize("cmd", [
    # Not a download tool
    "echo hello -O /tmp/foo",
    "python3 script.py -o /tmp/output",
    "make install -O /tmp/thing",
    # Download tool but no output flag (piping)
    "curl https://example.com | tar xz",
    "wget https://example.com/file.zip",  # downloads to cwd with server name
    # Output flag but to stdout
    "wget -O - https://example.com/stream",
    "curl -o - https://example.com/stream",
    # Empty / junk
    "",
    "   ",
])
def test_parse_download_target_negative(cmd):
    assert _parse_download_target(cmd) is None


def test_parse_download_target_handles_env_prefix():
    """A leading ``FOO=bar wget ...`` shape shouldn't confuse the
    first-token check. Users legitimately prefix env vars for
    download configuration (e.g., ``https_proxy=... wget URL``)."""
    assert _parse_download_target(
        "https_proxy=http://proxy:8080 wget -q URL -O /tmp/file.bin",
    ) == "/tmp/file.bin"


def test_parse_download_target_handles_sh_c_wrapper():
    """shell_exec may wrap the command in ``sh -c '...'`` before the
    parser sees it. Strip that wrapper so the first-token check
    actually lands on wget/curl."""
    got = _parse_download_target(
        'sh -c "wget -q URL -O /tmp/file.bin"',
    )
    assert got == "/tmp/file.bin"


def test_parse_download_target_strips_quoted_paths():
    """Quoted output paths are common when the path has spaces."""
    assert _parse_download_target(
        'wget -q URL -O "/tmp/spaced name.bin"',
    ) == "/tmp/spaced name.bin"


# ---------------------------------------------------------------------------
# Pattern set — make sure the frozen tokens match what the tier picker expects
# ---------------------------------------------------------------------------


def test_download_tokens_include_expected_tools():
    """The liveness detector only fires for commands in this set. A
    regression that drops wget/curl would silently disable the whole
    feature. Explicit assertion locks the contract."""
    assert "wget" in _DOWNLOAD_FIRST_TOKENS
    assert "curl" in _DOWNLOAD_FIRST_TOKENS
    assert "aria2" in _DOWNLOAD_FIRST_TOKENS
    assert "aria2c" in _DOWNLOAD_FIRST_TOKENS
    # rsync / scp also bulk-transfer and can be silent for a long time.
    assert "rsync" in _DOWNLOAD_FIRST_TOKENS
    assert "scp" in _DOWNLOAD_FIRST_TOKENS


# ---------------------------------------------------------------------------
# _fmt_bytes — heartbeat readability
# ---------------------------------------------------------------------------


def test_fmt_bytes_zero():
    assert _fmt_bytes(0) == "0 B"


def test_fmt_bytes_sub_kilobyte():
    assert _fmt_bytes(512) == "512 B"


def test_fmt_bytes_kilobytes():
    assert _fmt_bytes(2048) == "2.0 KB"


def test_fmt_bytes_megabytes():
    assert _fmt_bytes(5 * 1024 * 1024) == "5.0 MB"


def test_fmt_bytes_gigabytes():
    assert _fmt_bytes(2.5 * 1024 * 1024 * 1024) == "2.5 GB"


def test_fmt_bytes_negative_clamps_to_zero():
    """Defensive: a stat that returns a garbage value shouldn't
    produce nonsense output in a heartbeat. Negative input is clamped
    rather than raising."""
    assert _fmt_bytes(-500) == "0 B"
