"""Check-writer parser + validator tests.

The full subagent run requires a backend; we cover the deterministic
pieces:

* ``parse_check_source`` extracts the fenced Python block from
  permissive LLM output shapes.
* ``is_valid_check_source`` enforces the module contract: parses,
  exports ``run``, and isn't the ``# skipped`` sentinel.
"""

from __future__ import annotations

from augmentum.bug_finder.check_writer import (
    is_valid_check_source,
    parse_check_source,
    slug_for_pillar,
)

# ---------------------------------------------------------------------------
# parse_check_source
# ---------------------------------------------------------------------------


def test_parse_extracts_fenced_python_block() -> None:
    out = (
        "Here's the check:\n\n"
        "```python\n"
        "def run(root):\n"
        "    return []\n"
        "```\n"
    )
    src = parse_check_source(out)
    assert "def run" in src


def test_parse_handles_bare_fence() -> None:
    """Some models emit ``` without the language tag."""
    out = (
        "Sure:\n\n"
        "```\n"
        "def run(root):\n"
        "    return []\n"
        "```\n"
    )
    assert "def run" in parse_check_source(out)


def test_parse_last_block_wins() -> None:
    out = (
        "Draft:\n```python\ndef run(root):\n    return [1]\n```\n"
        "Final:\n```python\ndef run(root):\n    return [2]\n```\n"
    )
    src = parse_check_source(out)
    assert "return [2]" in src
    assert "return [1]" not in src


def test_parse_empty_when_no_fence() -> None:
    assert parse_check_source("just prose, no code") == ""


def test_parse_empty_when_no_input() -> None:
    assert parse_check_source("") == ""


# ---------------------------------------------------------------------------
# is_valid_check_source
# ---------------------------------------------------------------------------


def test_valid_when_module_has_run_function() -> None:
    src = (
        "from pathlib import Path\n"
        "def run(root: Path) -> list[dict]:\n"
        "    return []\n"
    )
    ok, reason = is_valid_check_source(src)
    assert ok
    assert reason == ""


def test_invalid_when_empty() -> None:
    ok, reason = is_valid_check_source("")
    assert not ok
    assert "empty" in reason.lower()


def test_invalid_when_syntax_error() -> None:
    ok, reason = is_valid_check_source(
        "def run(  # missing colon\n    pass\n",
    )
    assert not ok
    assert "syntax" in reason.lower()


def test_invalid_when_no_run_function() -> None:
    src = "def main(): pass\n"
    ok, reason = is_valid_check_source(src)
    assert not ok
    assert "run" in reason.lower()


def test_invalid_when_skip_sentinel() -> None:
    """Check_writer emits ``# skipped: ...`` when it can't write a
    useful check; the orchestrator must not persist it."""
    ok, reason = is_valid_check_source("# skipped: pillar too vague\n")
    assert not ok
    assert "skipped" in reason.lower()


def test_invalid_when_skip_sentinel_no_message() -> None:
    ok, _ = is_valid_check_source("# skipped\n")
    assert not ok


# ---------------------------------------------------------------------------
# Safety guardrails — the generated module is EXECUTED on every audit,
# so unsafe code must never pass validation (= never reach disk).
# ---------------------------------------------------------------------------


def test_valid_allows_stdlib_module_constants() -> None:
    """A module-level ``re.compile`` constant is benign and common —
    the import allowlist bounds what the RHS can reach, so allow it."""
    src = (
        "import re\n"
        "from pathlib import Path\n"
        "_PAT = re.compile(r'x')\n"
        "def run(root: Path) -> list[dict]:\n"
        "    return []\n"
    )
    ok, reason = is_valid_check_source(src)
    assert ok, reason


def test_invalid_rejects_nonstdlib_import() -> None:
    for mod in ("os", "sys", "subprocess", "socket", "importlib", "shutil"):
        src = f"import {mod}\ndef run(root):\n    return []\n"
        ok, reason = is_valid_check_source(src)
        assert not ok, mod
        assert "disallowed import" in reason


def test_invalid_rejects_from_import_of_blocked_module() -> None:
    ok, reason = is_valid_check_source(
        "from subprocess import run as r\ndef run(root):\n    return []\n",
    )
    assert not ok
    assert "subprocess" in reason


def test_invalid_rejects_relative_import() -> None:
    ok, reason = is_valid_check_source(
        "from . import secrets\ndef run(root):\n    return []\n",
    )
    assert not ok
    assert "import" in reason.lower()


def test_invalid_rejects_toplevel_side_effect() -> None:
    """A bare top-level call executes at import time."""
    ok, reason = is_valid_check_source(
        "def run(root):\n    return []\nprint('side effect')\n",
    )
    assert not ok
    assert "top-level" in reason.lower()


def test_invalid_rejects_exec_builtin() -> None:
    for name in ("eval", "exec", "compile", "__import__"):
        src = f"def run(root):\n    return {name}('1')\n"
        ok, reason = is_valid_check_source(src)
        assert not ok, name
        assert "forbidden builtin" in reason


def test_invalid_rejects_dunder_escape() -> None:
    ok, reason = is_valid_check_source(
        "def run(root):\n    return ().__class__.__bases__\n",
    )
    assert not ok
    assert "forbidden attribute" in reason


# ---------------------------------------------------------------------------
# slug_for_pillar
# ---------------------------------------------------------------------------


def test_slug_is_filesystem_safe() -> None:
    assert slug_for_pillar("user_id scoping!! v2") == "user_id_scoping_v2"


def test_slug_never_empty() -> None:
    assert slug_for_pillar("!!!") == "pillar_check"


def test_slug_is_deterministic() -> None:
    assert slug_for_pillar("Auth Bypass") == slug_for_pillar("Auth Bypass")
