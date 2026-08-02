"""Tests for augmentum/coder/digest.py — the project masterfile block.

Covers the small-workspace-optimisation path: when the whole project
fits under the token budget, inline every file; otherwise return None
so the caller falls through to repo_map + on-demand file_read.
"""
from __future__ import annotations

import pytest

from augmentum.coder.digest import (
    _default_budget,
    _estimate_tokens,
    _file_footer,
    _file_header,
    build_project_digest,
)


# ---------------------------------------------------------------------------
# Container-manager stub. Matches the surface build_project_digest calls:
# ``_run_command`` for the find listing, ``file_read`` per file.
# ---------------------------------------------------------------------------

class _StubCM:
    def __init__(self, files: dict[str, str] | None = None) -> None:
        # files maps /workspace/<rel> -> content
        self.files = files or {}

    async def run_command(self, *args, **kwargs):
        return await self._run_command(*args, **kwargs)

    async def _run_command(self, workspace_id, cmd, timeout=None):  # noqa: ARG002
        """Return a fake ``wc -l`` listing for every file in ``self.files``."""
        lines: list[str] = []
        for path, content in sorted(self.files.items()):
            count = len(content.splitlines()) or 1
            lines.append(f"{count:>7} {path}")
        lines.append(f"{sum(len(c.splitlines()) or 1 for c in self.files.values()):>7} total")
        return "\n".join(lines)

    async def file_read(self, workspace_id, path):  # noqa: ARG002
        return self.files.get(path, "")


# ---------------------------------------------------------------------------
# Budget / estimation helpers
# ---------------------------------------------------------------------------


def test_default_budget_is_40k():
    assert _default_budget() == 40_000


def test_default_budget_scales_when_context_window_known(monkeypatch):
    monkeypatch.delenv("AUGMENTUM_CODER_DIGEST_BUDGET", raising=False)
    assert 75_000 <= _default_budget(context_window=261_376) <= 85_000


def test_default_budget_env_override(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_CODER_DIGEST_BUDGET", "10000")
    assert _default_budget() == 10_000


def test_default_budget_rejects_bogus_env(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_CODER_DIGEST_BUDGET", "not-a-number")
    assert _default_budget() == 40_000


def test_default_budget_rejects_negative_env(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_CODER_DIGEST_BUDGET", "-5")
    assert _default_budget() == 40_000


def test_estimate_tokens_rough():
    # ~4 chars/token. Ceiling on division so empty string is 0 tokens.
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("abcd") == 1
    assert _estimate_tokens("a" * 40) == 10


def test_file_header_footer_grep_friendly():
    # Regression guard — the boundaries are documented as "===== FILE: "
    # and "===== END: "; downstream tools may grep for these, so
    # changes here need a deliberate bump.
    assert _file_header("src/app.py", 42) == "===== FILE: src/app.py (42L) ====="
    assert _file_footer("src/app.py") == "===== END: src/app.py ====="


# ---------------------------------------------------------------------------
# build_project_digest — main behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_workspace_returns_none():
    cm = _StubCM(files={})
    assert await build_project_digest(cm, "ws") is None


@pytest.mark.asyncio
async def test_no_container_returns_none():
    assert await build_project_digest(None, "ws") is None


@pytest.mark.asyncio
async def test_small_workspace_inlines_all_files():
    cm = _StubCM(files={
        "/workspace/a.py": "print('a')\n",
        "/workspace/b.py": "print('b')\n",
    })
    digest = await build_project_digest(cm, "ws")
    assert digest is not None
    assert "===== FILE: a.py" in digest
    assert "===== FILE: b.py" in digest
    assert "print('a')" in digest
    assert "print('b')" in digest
    # Opening / closing XML-ish tags for the whole block
    assert digest.startswith("<project_digest>")
    assert digest.endswith("</project_digest>")


@pytest.mark.asyncio
async def test_deterministic_sort_by_path():
    # Files listed in reverse alphabetical order in the "find" output
    # should still appear sorted in the digest (prefix-caching relies
    # on this).
    cm = _StubCM(files={
        "/workspace/z.py": "z\n",
        "/workspace/a.py": "a\n",
        "/workspace/m.py": "m\n",
    })
    digest = await build_project_digest(cm, "ws")
    assert digest is not None
    pos_a = digest.index("FILE: a.py")
    pos_m = digest.index("FILE: m.py")
    pos_z = digest.index("FILE: z.py")
    assert pos_a < pos_m < pos_z


@pytest.mark.asyncio
async def test_over_budget_returns_none():
    # One giant file over the budget — preflight estimate rejects.
    huge = "x" * 200_000  # ~50k tokens, well over the 10k budget below
    cm = _StubCM(files={"/workspace/big.py": huge + "\n"})
    digest = await build_project_digest(cm, "ws", token_budget=10_000)
    assert digest is None


@pytest.mark.asyncio
async def test_oversized_file_forces_whole_digest_to_bail():
    # All-or-nothing contract: a single file that pushes the running
    # total over the char budget causes the entire digest to return
    # None rather than emit a truncated view. A partial digest is
    # worse than no digest because the preamble tells the model the
    # block is authoritative — invisible missing content produces
    # confident-but-wrong edits.
    tight_budget = 2_000  # ~8 KB char budget
    cm = _StubCM(files={
        "/workspace/small.py": "tiny\n",
        "/workspace/big.py": "y" * 50_000,  # way over budget on its own
    })
    digest = await build_project_digest(
        cm, "ws", token_budget=tight_budget,
    )
    assert digest is None


@pytest.mark.asyncio
async def test_no_truncation_markers_in_any_returned_digest():
    # Regression guard: if anyone re-adds silent truncation, this
    # test catches it. A digest must contain every file's complete
    # content or return None.
    cm = _StubCM(files={"/workspace/x.py": "hello\n"})
    digest = await build_project_digest(cm, "ws")
    assert digest is not None
    assert "[truncated" not in digest
    assert "truncated:" not in digest


@pytest.mark.asyncio
async def test_boundary_markers_wrap_every_file():
    cm = _StubCM(files={
        "/workspace/a.py": "A\n",
        "/workspace/b.py": "B\n",
    })
    digest = await build_project_digest(cm, "ws")
    assert digest is not None
    # Each file gets exactly one header + one footer
    assert digest.count("===== FILE: a.py") == 1
    assert digest.count("===== END: a.py") == 1
    assert digest.count("===== FILE: b.py") == 1
    assert digest.count("===== END: b.py") == 1


@pytest.mark.asyncio
async def test_relative_paths_not_absolute():
    # Digest reads like a project tour — relative to /workspace — not
    # an absolute-path dump.
    cm = _StubCM(files={"/workspace/src/app.py": "x\n"})
    digest = await build_project_digest(cm, "ws")
    assert digest is not None
    assert "FILE: src/app.py" in digest
    assert "/workspace/src/app.py" not in digest


@pytest.mark.asyncio
async def test_listing_command_failure_returns_none():
    class _BrokenCM:
        async def run_command(self, *args, **kwargs):
            return await self._run_command(*args, **kwargs)

        async def _run_command(self, *a, **kw):  # noqa: ARG002
            raise RuntimeError("container unreachable")

        async def file_read(self, *a, **kw):  # noqa: ARG002
            return ""

    assert await build_project_digest(_BrokenCM(), "ws") is None


@pytest.mark.asyncio
async def test_individual_file_read_failures_are_skipped():
    # If one file_read blows up, the digest continues with the rest.
    # Silent degradation is appropriate because digest is an
    # optimisation, not the source of truth.
    class _PartialCM(_StubCM):
        async def file_read(self, workspace_id, path):  # noqa: ARG002
            if path == "/workspace/broken.py":
                raise OSError("permission denied")
            return self.files.get(path, "")

    cm = _PartialCM(files={
        "/workspace/good.py": "good\n",
        "/workspace/broken.py": "should error\n",
    })
    digest = await build_project_digest(cm, "ws")
    assert digest is not None
    assert "good.py" in digest
    # Broken file's content isn't in the digest (but we don't assert
    # its HEADER is absent — the current impl skips it entirely on
    # read failure, which is the cleanest signal to the model).
    assert "should error" not in digest


@pytest.mark.asyncio
async def test_preamble_tells_model_not_to_redundant_read():
    # The preamble is what nudges the model away from unnecessary
    # file_read / dir_tree calls. Regression guard — if this text
    # changes, weak models may stop trusting the digest.
    cm = _StubCM(files={"/workspace/x.py": "x\n"})
    digest = await build_project_digest(cm, "ws")
    assert digest is not None
    assert "authoritative view" in digest
    assert "file_read" in digest
