"""Tests for the workspace-kernel substrate (Phase 1 migration).

Covers:
* ``WorkspaceKernel`` tier-conditional file set
* ``refresh()`` calls mkdir for non-REFLEX tiers, no-ops for REFLEX
* Best-effort contract: container failures swallowed without raising
* ``read_plan()`` returns ``""`` on miss / trimmed content on hit
* ``coder_kernel_v2`` flag drives the system-prompt ``<workspace_kernel>``
  hint via ``CoderHandler._build_messages``
* ``coder_kernel_v2`` flag suppresses the plan-anchor read in the act
  loop's sticky-reminder injection path

See ``docs/superpowers/specs/2026-05-16-workspace-kernel-design.md``.
"""
from __future__ import annotations

import pytest

from augmentum.coder.workspace_kernel import (
    IDENTITY_TOML,
    KERNEL_ROOT,
    OBJECTIVE_MD,
    OBSERVATIONS_JSONL,
    PLAN_MD,
    PROFILE_MD,
    RECENT_FAILURES_MD,
    WORLD_MD,
    WorkspaceKernel,
)
from augmentum.modes.coder.intent import Tier

# ---------------------------------------------------------------------------
# Fakes — minimal container manager mimicking the public surface the
# kernel touches. ``run_command_calls`` lets tests assert what the
# kernel actually did without standing up Docker.
# ---------------------------------------------------------------------------


class _FakeContainer:
    """Records run_command + file_read invocations; otherwise inert.

    ``files`` (added 2026-05-28) maps absolute container paths to
    contents — lets tests seed arbitrary workspace state without
    mocking individual reads. Writes are captured in ``file_writes``
    so tests assert what the kernel attempted to persist.
    """

    def __init__(
        self,
        *,
        plan_content: str | None = None,
        files: dict[str, str] | None = None,
        fail: bool = False,
    ):
        self.run_command_calls: list[tuple[str, list[str]]] = []
        self.file_read_calls: list[tuple[str, str]] = []
        self.file_writes: dict[str, str] = {}
        self._plan_content = plan_content
        self._files = dict(files or {})
        self._fail = fail

    async def _run_command(self, workspace_id, cmd, timeout=None):
        if self._fail:
            raise RuntimeError("container down")
        self.run_command_calls.append((workspace_id, list(cmd)))
        return ""

    async def file_read(self, workspace_id, path):
        if self._fail:
            raise RuntimeError("container down")
        self.file_read_calls.append((workspace_id, path))
        if path in self._files:
            return self._files[path]
        if path == PLAN_MD and self._plan_content is not None:
            return self._plan_content
        # Default: file-not-found shape (raise, mirroring real container)
        raise FileNotFoundError(path)

    async def file_write(self, workspace_id, path, content):
        if self._fail:
            raise RuntimeError("container down")
        self.file_writes[path] = content
        # After a write, subsequent reads should return the new content.
        self._files[path] = content


# ---------------------------------------------------------------------------
# files_for_tier — the tier → file-set mapping is the contract that drives
# every later migration. Pin it explicitly so reshuffling a tier's set
# can't slip in without a test failure.
# ---------------------------------------------------------------------------


class TestFilesForTier:
    def test_reflex_writes_nothing(self):
        k = WorkspaceKernel(_FakeContainer(), "ws")
        assert k.files_for_tier(Tier.REFLEX) == frozenset()

    def test_surgical_writes_plan_identity_objective(self):
        k = WorkspaceKernel(_FakeContainer(), "ws")
        assert k.files_for_tier(Tier.SURGICAL) == frozenset({
            PLAN_MD, IDENTITY_TOML, OBJECTIVE_MD,
        })

    def test_composed_writes_plan_failures_world_identity_observations_objective(self):
        k = WorkspaceKernel(_FakeContainer(), "ws")
        assert k.files_for_tier(Tier.COMPOSED) == frozenset({
            PLAN_MD, RECENT_FAILURES_MD, WORLD_MD, IDENTITY_TOML,
            OBJECTIVE_MD, OBSERVATIONS_JSONL,
        })

    def test_project_writes_full_set(self):
        k = WorkspaceKernel(_FakeContainer(), "ws")
        assert k.files_for_tier(Tier.PROJECT) == frozenset({
            PLAN_MD, RECENT_FAILURES_MD, WORLD_MD, PROFILE_MD, IDENTITY_TOML,
            OBJECTIVE_MD, OBSERVATIONS_JSONL,
        })


# ---------------------------------------------------------------------------
# refresh — at this slice it only ensures the .augmentum/ directory
# exists. REFLEX skips the mkdir; everything else mkdir -ps it.
# ---------------------------------------------------------------------------


class TestRefresh:
    @pytest.mark.asyncio
    async def test_reflex_is_a_no_op(self):
        fake = _FakeContainer()
        k = WorkspaceKernel(fake, "ws")
        await k.refresh(tier=Tier.REFLEX)
        assert fake.run_command_calls == []

    @pytest.mark.asyncio
    async def test_surgical_mkdirs_kernel_root(self):
        fake = _FakeContainer()
        k = WorkspaceKernel(fake, "ws")
        await k.refresh(tier=Tier.SURGICAL)
        assert fake.run_command_calls == [("ws", ["mkdir", "-p", KERNEL_ROOT])]

    @pytest.mark.asyncio
    async def test_composed_mkdirs_kernel_root(self):
        fake = _FakeContainer()
        k = WorkspaceKernel(fake, "ws")
        await k.refresh(tier=Tier.COMPOSED)
        assert fake.run_command_calls == [("ws", ["mkdir", "-p", KERNEL_ROOT])]

    @pytest.mark.asyncio
    async def test_project_mkdirs_kernel_root(self):
        fake = _FakeContainer()
        k = WorkspaceKernel(fake, "ws")
        await k.refresh(tier=Tier.PROJECT)
        assert fake.run_command_calls == [("ws", ["mkdir", "-p", KERNEL_ROOT])]

    @pytest.mark.asyncio
    async def test_refresh_is_idempotent(self):
        """Two refresh calls should produce two mkdir -ps — mkdir -p is
        itself idempotent on the filesystem, but the kernel must not
        short-circuit on a "we already ran" flag (that'd race with
        external dir deletions)."""
        fake = _FakeContainer()
        k = WorkspaceKernel(fake, "ws")
        await k.refresh(tier=Tier.SURGICAL)
        await k.refresh(tier=Tier.SURGICAL)
        assert len(fake.run_command_calls) == 2

    @pytest.mark.asyncio
    async def test_refresh_swallows_container_failure(self):
        """Best-effort contract: a kernel failure must never raise out
        of refresh — the caller wraps in conditional but shouldn't have
        to wrap in try/except too."""
        fake = _FakeContainer(fail=True)
        k = WorkspaceKernel(fake, "ws")
        await k.refresh(tier=Tier.SURGICAL)  # must not raise

    @pytest.mark.asyncio
    async def test_refresh_without_container_is_noop(self):
        k = WorkspaceKernel(None, "ws")
        await k.refresh(tier=Tier.PROJECT)  # must not raise

    @pytest.mark.asyncio
    async def test_refresh_without_workspace_is_noop(self):
        fake = _FakeContainer()
        k = WorkspaceKernel(fake, "")
        await k.refresh(tier=Tier.PROJECT)
        assert fake.run_command_calls == []


# ---------------------------------------------------------------------------
# read_plan — best-effort, returns "" on any miss/error. The handler's
# legacy ``_read_plan_md`` had the same contract; the kernel preserves
# it so downstream callers can swap source without behavior change.
# ---------------------------------------------------------------------------


class TestReadPlan:
    @pytest.mark.asyncio
    async def test_missing_file_returns_empty(self):
        fake = _FakeContainer()  # no plan_content set → file_read raises
        k = WorkspaceKernel(fake, "ws")
        assert await k.read_plan() == ""

    @pytest.mark.asyncio
    async def test_present_file_returns_trimmed(self):
        fake = _FakeContainer(plan_content="\n  hello plan  \n\n")
        k = WorkspaceKernel(fake, "ws")
        assert await k.read_plan() == "hello plan"

    @pytest.mark.asyncio
    async def test_container_failure_returns_empty(self):
        fake = _FakeContainer(fail=True)
        k = WorkspaceKernel(fake, "ws")
        assert await k.read_plan() == ""

    @pytest.mark.asyncio
    async def test_no_container_returns_empty(self):
        k = WorkspaceKernel(None, "ws")
        assert await k.read_plan() == ""


# ---------------------------------------------------------------------------
# Flag wiring — when ``coder_kernel_v2`` is True the handler injects a
# one-sentence ``<workspace_kernel>`` block into the system message
# telling the model that ``.augmentum/`` exists and is readable on
# demand. When False the block is absent (legacy behavior preserved).
# ---------------------------------------------------------------------------


def _system_prompt_with_flag(monkeypatch, *, flag: bool) -> str:
    """Build a CoderHandler, run _build_messages with both gates off so
    the only variable is the kernel_v2 flag, and return the concatenated
    system prompt text."""
    from augmentum.config import settings as _settings
    from augmentum.models.base import InternalChatRequest, Message
    from augmentum.modes.coder.handler import CoderHandler

    monkeypatch.setattr(_settings, "coder_kernel_v2", flag)

    handler = CoderHandler(
        backend=None, session_id="s", workspace_id="w",
        container_manager=None,
    )
    handler._cached_guide = "GUIDE"  # bypass async read in _build_messages

    request = InternalChatRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
    )
    messages = handler._build_messages(request, extra_system="EXTRA")
    return messages[0].content


def test_build_messages_includes_cached_facts_block(monkeypatch):
    """The cached facts block (populated by _refresh_kernel_facts at
    turn-start) must appear in every strategy's system prompt that
    routes through _build_messages — i.e. canonical, hybrid, plan,
    and the legacy strategies. Single source of truth across
    strategies."""
    from augmentum.config import settings as _settings
    from augmentum.models.base import InternalChatRequest, Message
    from augmentum.modes.coder.handler import CoderHandler

    monkeypatch.setattr(_settings, "coder_kernel_v2", True)

    handler = CoderHandler(
        backend=None, session_id="s", workspace_id="w",
        container_manager=None,
    )
    handler._cached_guide = "GUIDE"
    handler._cached_facts_block = (
        "<workspace_facts>\n"
        "Objective (user-pinned):\n  test obj\n\n"
        "Project: python (uv) · test=pytest\n"
        "</workspace_facts>"
    )
    request = InternalChatRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
    )
    messages = handler._build_messages(request, extra_system="EXTRA")
    sys_content = messages[0].content
    assert "<workspace_facts>" in sys_content
    assert "test obj" in sys_content
    assert "python (uv)" in sys_content
    # Sits adjacent to the kernel hint.
    hint_idx = sys_content.index("<workspace_kernel>")
    facts_idx = sys_content.index("<workspace_facts>")
    # Hint comes first (general orientation), facts follow.
    assert hint_idx < facts_idx


def test_build_messages_omits_facts_block_when_empty(monkeypatch):
    """Empty cache (no kernel content or flag off) → no <workspace_facts>
    block. The existing guide + extra_system must still survive."""
    from augmentum.config import settings as _settings
    from augmentum.models.base import InternalChatRequest, Message
    from augmentum.modes.coder.handler import CoderHandler

    monkeypatch.setattr(_settings, "coder_kernel_v2", True)

    handler = CoderHandler(
        backend=None, session_id="s", workspace_id="w",
        container_manager=None,
    )
    handler._cached_guide = "GUIDE"
    handler._cached_facts_block = ""
    request = InternalChatRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
    )
    messages = handler._build_messages(request, extra_system="EXTRA")
    sys_content = messages[0].content
    assert "<workspace_facts>" not in sys_content
    assert "GUIDE" in sys_content
    assert "EXTRA" in sys_content


def test_system_prompt_contains_workspace_kernel_when_flag_on(monkeypatch):
    sys_text = _system_prompt_with_flag(monkeypatch, flag=True)
    assert "<workspace_kernel>" in sys_text
    assert "/workspace/.augmentum/" in sys_text


def test_system_prompt_omits_workspace_kernel_when_flag_off(monkeypatch):
    sys_text = _system_prompt_with_flag(monkeypatch, flag=False)
    assert "<workspace_kernel>" not in sys_text
    # The legacy guide + extra_system must still survive — this test
    # exists partly to catch a regression where the kernel hint
    # accidentally replaces existing system content.
    assert "GUIDE" in sys_text
    assert "EXTRA" in sys_text


# ---------------------------------------------------------------------------
# hint_text — static, single-sourced. Same string for the canonical/hybrid
# path (via _build_messages) and the native path (via phase_act._act_native).
# Pinned by a direct test of the method so refactors can't drift the wording
# silently.
# ---------------------------------------------------------------------------


def test_hint_text_static_no_instance_needed():
    """Hint must work without a live kernel — the legacy handler only
    constructs ``_workspace_kernel`` when a container_manager exists,
    but the system-prompt block should render in unit-test contexts
    too (and on early-turn entry before the kernel is bound)."""
    hint = WorkspaceKernel.hint_text(enabled=True)
    assert "<workspace_kernel>" in hint
    assert "/workspace/.augmentum/" in hint
    assert "plan.md" in hint


def test_hint_text_returns_empty_when_disabled():
    assert WorkspaceKernel.hint_text(enabled=False) == ""


def test_hint_text_respects_global_setting(monkeypatch):
    """When ``enabled`` is None (default), hint_text reads the live
    setting. This is the path both handler.py and phase_act.py take."""
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "coder_kernel_v2", True)
    assert "<workspace_kernel>" in WorkspaceKernel.hint_text()

    monkeypatch.setattr(_settings, "coder_kernel_v2", False)
    assert WorkspaceKernel.hint_text() == ""


# ---------------------------------------------------------------------------
# Native sys_text wiring — phase_act._act_native renders the same kernel
# hint as canonical/hybrid. Pre-2026-05-28 native built sys_text directly
# and never told the model the .augmentum/ directory existed; live logs
# showed Qwen3.6 looping on silent shells trying to find credentials it
# would have found in plan.md if it had known to look.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_native_sys_text_contains_kernel_hint(monkeypatch):
    """Smoke test: a native turn's first request has the kernel hint in
    its system message when the flag is on."""
    from augmentum.config import settings as _settings
    from augmentum.models.base import InternalStreamChunk
    from augmentum.modes.coder.handler import CoderHandler

    monkeypatch.setattr(_settings, "coder_kernel_v2", True)

    # Reuse the existing native test scaffolding so we don't reimplement
    # backend fakes — they're well-tuned for the act loop.
    from tests.test_coder_handler import (
        _ExtendedContainerManager,
        _FakeChunk,
        _FakeTool,
        _make_request,
    )

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )

    class _RecordingBackend:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            yield _FakeChunk(content_delta=(
                "Hello there. The workspace looks ready for inspection."
            ))
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _RecordingBackend()
    handler = CoderHandler(
        backend,
        session_id="s",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        _make_request("inspect the workspace"), workspace_context="",
    ):
        chunks.append(c)

    assert backend.requests, "Expected at least one backend request"
    sys_msg = backend.requests[0].messages[0]
    assert sys_msg.role == "system"
    assert "<workspace_kernel>" in sys_msg.content
    assert "/workspace/.augmentum/" in sys_msg.content


# ---------------------------------------------------------------------------
# refresh_identity — detection at turn-start writes identity.toml,
# preserving [asserted] + [discovered] across calls.
# ---------------------------------------------------------------------------


class TestRefreshIdentity:
    @pytest.mark.asyncio
    async def test_refresh_identity_writes_manifest_for_python_project(self):
        fake = _FakeContainer(files={
            "/workspace/pyproject.toml": "[project]\nname = 'demo'\n",
            "/workspace/uv.lock": "",
        })
        k = WorkspaceKernel(fake, "ws")
        await k.refresh_identity()

        assert IDENTITY_TOML in fake.file_writes
        written = fake.file_writes[IDENTITY_TOML]
        assert "[meta]" in written
        assert "[detected]" in written
        assert '"python"' in written
        assert "uv" in written

    @pytest.mark.asyncio
    async def test_refresh_identity_preserves_asserted_and_discovered(self):
        """Refreshing detection must not clobber user-asserted facts
        or model-discovered observations. This is the load-bearing
        ownership invariant of the layer."""
        seeded = """\
[meta]
detector_version = 1
last_detected_at = 100.0

[detected]
languages = ["python"]

[detected.python]
package_manager = "pip"

[asserted]
deploy_target = "fly.io"

[[discovered]]
ts = 1.0
category = "env"
fact = "secrets live in /workspace/.env"
source = "shell_exec turn 1"
confidence = "confirmed"
"""
        fake = _FakeContainer(files={
            IDENTITY_TOML: seeded,
            "/workspace/pyproject.toml": "[project]\nname='demo'\n",
            "/workspace/uv.lock": "",  # now uv, not pip — refresh should update
        })
        k = WorkspaceKernel(fake, "ws")
        await k.refresh_identity()

        manifest = await k.read_identity()
        # detected REPLACED.
        assert manifest.detected["python"]["package_manager"] == "uv"
        # asserted PRESERVED.
        assert manifest.asserted == {"deploy_target": "fly.io"}
        # discovered PRESERVED.
        assert len(manifest.discovered) == 1
        assert manifest.discovered[0].fact == "secrets live in /workspace/.env"
        assert manifest.discovered[0].category == "env"

    @pytest.mark.asyncio
    async def test_refresh_identity_swallows_failure(self):
        """Best-effort contract: identity refresh failures never
        propagate. The agent's contract is "the file might be there",
        never "the file is guaranteed"."""
        fake = _FakeContainer(fail=True)
        k = WorkspaceKernel(fake, "ws")
        await k.refresh_identity()  # must not raise

    @pytest.mark.asyncio
    async def test_refresh_identity_empty_workspace_writes_empty_languages(self):
        fake = _FakeContainer(files={})  # no project files at all
        k = WorkspaceKernel(fake, "ws")
        await k.refresh_identity()

        manifest = await k.read_identity()
        assert manifest.detected.get("languages") == []

    @pytest.mark.asyncio
    async def test_refresh_at_surgical_tier_calls_refresh_identity(self):
        """The tier dispatch in refresh() must wire identity.toml for
        SURGICAL+ — otherwise the file set advertised by
        files_for_tier wouldn't actually appear on disk."""
        fake = _FakeContainer(files={
            "/workspace/pyproject.toml": "[project]\nname='x'\n",
        })
        k = WorkspaceKernel(fake, "ws")
        await k.refresh(tier=Tier.SURGICAL)
        assert IDENTITY_TOML in fake.file_writes

    @pytest.mark.asyncio
    async def test_refresh_at_reflex_does_not_touch_identity(self):
        """REFLEX is the spartan tier — no .augmentum/ writes at all."""
        fake = _FakeContainer(files={
            "/workspace/pyproject.toml": "[project]\nname='x'\n",
        })
        k = WorkspaceKernel(fake, "ws")
        await k.refresh(tier=Tier.REFLEX)
        assert IDENTITY_TOML not in fake.file_writes


# ---------------------------------------------------------------------------
# read_identity — returns parsed manifest or empty on miss
# ---------------------------------------------------------------------------


class TestObjective:
    """User-curated session goal anchor. Auto-seeded from the first
    substantive user message; never silently overwritten on subsequent
    seeds. Surfaces at the top of the facts block so the model can
    never lose the original ask."""

    @pytest.mark.asyncio
    async def test_read_objective_missing_returns_empty(self):
        fake = _FakeContainer()
        k = WorkspaceKernel(fake, "ws")
        assert await k.read_objective() == ""

    @pytest.mark.asyncio
    async def test_seed_writes_when_missing(self):
        fake = _FakeContainer()
        k = WorkspaceKernel(fake, "ws")
        seed_text = (
            "Add real-time request logging to the research agent UI."
        )
        ok = await k.seed_objective_if_missing(seed_text)
        assert ok is True
        assert OBJECTIVE_MD in fake.file_writes
        # The seed includes header scaffolding so the user has
        # context when they open the file.
        assert "# Session Objective" in fake.file_writes[OBJECTIVE_MD]
        assert seed_text in fake.file_writes[OBJECTIVE_MD]

    @pytest.mark.asyncio
    async def test_seed_skipped_for_short_text(self):
        """Short messages ('hi', 'thanks', 'continue') must not
        pollute the anchor. The 30-char floor is the gate."""
        fake = _FakeContainer()
        k = WorkspaceKernel(fake, "ws")
        ok = await k.seed_objective_if_missing("hi there")
        assert ok is False
        assert OBJECTIVE_MD not in fake.file_writes

    @pytest.mark.asyncio
    async def test_seed_idempotent_when_file_present(self):
        """Once seeded (or user-written), subsequent seeds are no-ops.
        Load-bearing for the user-curated contract — we must NEVER
        silently overwrite."""
        existing = "Build a snake game using Python turtle."
        fake = _FakeContainer(files={OBJECTIVE_MD: existing})
        k = WorkspaceKernel(fake, "ws")
        ok = await k.seed_objective_if_missing(
            "totally different objective that's long enough to qualify",
        )
        assert ok is False
        # File unchanged.
        assert OBJECTIVE_MD not in fake.file_writes

    @pytest.mark.asyncio
    async def test_seed_no_container_returns_false(self):
        k = WorkspaceKernel(None, "ws")
        ok = await k.seed_objective_if_missing(
            "valid long objective text for the seeding contract",
        )
        assert ok is False

    def test_strip_header_handles_seeded_file(self):
        """The seeded file has scaffolding; the render path strips it
        before showing to the model."""
        seeded = (
            "# Session Objective\n"
            "<!-- comment 1 -->\n"
            "<!-- comment 2 -->\n"
            "\n"
            "Add request logging.\n"
        )
        body = WorkspaceKernel._strip_objective_header(seeded)
        assert body == "Add request logging."

    def test_strip_header_keeps_user_edited_file_intact(self):
        """A user who hand-edits and drops the scaffolding still has
        their content surface correctly."""
        user_written = "Just a one-liner the user typed directly."
        body = WorkspaceKernel._strip_objective_header(user_written)
        assert body == user_written


class TestRenderFactsBlock:
    """Composite render of identity + observations as a system-prompt
    block. Used by _act_native to surface durable facts at turn-start
    without per-iteration re-injection."""

    @pytest.mark.asyncio
    async def test_empty_workspace_renders_nothing(self):
        fake = _FakeContainer(files={})
        k = WorkspaceKernel(fake, "ws")
        assert await k.render_facts_block() == ""

    @pytest.mark.asyncio
    async def test_identity_only_renders_summary_line(self):
        """A workspace with detected identity but no observations
        should render the project summary line + directive — no
        empty observations section."""
        identity_text = """\
[meta]
detector_version = 1
last_detected_at = 100.0

[detected]
languages = ["python"]
language_primary = "python"

[detected.python]
package_manager = "uv"
test_runner = "pytest"

[asserted]
"""
        fake = _FakeContainer(files={IDENTITY_TOML: identity_text})
        k = WorkspaceKernel(fake, "ws")
        block = await k.render_facts_block()
        assert "<workspace_facts>" in block
        assert "Project:" in block
        assert "python" in block
        assert "uv" in block
        # No observations exist → no "Established" header.
        assert "Established" not in block
        # The directive must appear so the model trusts the block.
        assert "trust the block" in block.lower() or "known true" in block.lower()

    @pytest.mark.asyncio
    async def test_observations_only_render_with_priority_categories(self):
        from augmentum.coder.observations import (
            Observation,
            serialize_observations,
        )
        from augmentum.coder.workspace_kernel import OBSERVATIONS_JSONL

        ledger_text = serialize_observations([
            Observation(ts=1.0, category="other", fact="random fact", source="t1"),
            Observation(ts=2.0, category="constraint", fact="node 18 locked", source="t2"),
        ])
        fake = _FakeContainer(files={OBSERVATIONS_JSONL: ledger_text})
        k = WorkspaceKernel(fake, "ws")
        block = await k.render_facts_block()
        assert "<workspace_facts>" in block
        assert "node 18 locked" in block
        # The "Established" header surfaces when observations are present.
        assert "Established" in block

    @pytest.mark.asyncio
    async def test_constraint_surfaces_before_other(self):
        from augmentum.coder.observations import (
            Observation,
            serialize_observations,
        )
        from augmentum.coder.workspace_kernel import OBSERVATIONS_JSONL

        ledger_text = serialize_observations([
            Observation(ts=10.0, category="other", fact="newer other-fact", source="t1"),
            Observation(ts=5.0, category="constraint", fact="must use uv", source="t2"),
        ])
        fake = _FakeContainer(files={OBSERVATIONS_JSONL: ledger_text})
        k = WorkspaceKernel(fake, "ws")
        block = await k.render_facts_block()
        # Even though `other` is newer, `constraint` should appear first.
        assert block.index("must use uv") < block.index("newer other-fact")

    @pytest.mark.asyncio
    async def test_full_block_renders_both_layers(self):
        from augmentum.coder.observations import (
            Observation,
            serialize_observations,
        )
        from augmentum.coder.workspace_kernel import OBSERVATIONS_JSONL

        identity_text = """\
[meta]
detector_version = 1
last_detected_at = 100.0

[detected]
languages = ["rust"]
language_primary = "rust"

[detected.rust]
package_manager = "cargo"
test_runner = "cargo test"

[asserted]
deploy_target = "fly.io"
"""
        ledger_text = serialize_observations([
            Observation(ts=5.0, category="gotcha", fact="watch out for X", source="t1"),
        ])
        fake = _FakeContainer(files={
            IDENTITY_TOML: identity_text,
            OBSERVATIONS_JSONL: ledger_text,
        })
        k = WorkspaceKernel(fake, "ws")
        block = await k.render_facts_block()
        assert "rust" in block
        assert "cargo" in block
        assert "fly.io" in block
        assert "watch out for X" in block

    @pytest.mark.asyncio
    async def test_budget_zero_returns_empty(self):
        fake = _FakeContainer(files={
            IDENTITY_TOML: "[meta]\ndetector_version = 1\n[detected]\nlanguages = [\"python\"]\n[asserted]\n",
        })
        k = WorkspaceKernel(fake, "ws")
        assert await k.render_facts_block(budget_chars=0) == ""

    @pytest.mark.asyncio
    async def test_container_failure_returns_empty(self):
        fake = _FakeContainer(fail=True)
        k = WorkspaceKernel(fake, "ws")
        assert await k.render_facts_block() == ""

    @pytest.mark.asyncio
    async def test_objective_appears_at_top_of_block(self):
        """Objective is the user's pinned anchor — must surface
        before identity and observations so the model's attention
        anchors on it first."""
        from augmentum.coder.observations import (
            Observation,
            serialize_observations,
        )
        from augmentum.coder.workspace_kernel import OBSERVATIONS_JSONL

        identity_text = (
            "[meta]\n"
            "detector_version = 1\n\n"
            "[detected]\n"
            'languages = ["python"]\n'
            'language_primary = "python"\n\n'
            "[detected.python]\n"
            'package_manager = "uv"\n\n'
            "[asserted]\n"
        )
        objective_seeded = (
            "# Session Objective\n"
            "<!-- header -->\n\n"
            "Ship the cross-modal moat feature this week.\n"
        )
        ledger_text = serialize_observations([
            Observation(
                ts=1.0, category="constraint",
                fact="don't break ports", source="t1",
            ),
        ])
        fake = _FakeContainer(files={
            IDENTITY_TOML: identity_text,
            OBJECTIVE_MD: objective_seeded,
            OBSERVATIONS_JSONL: ledger_text,
        })
        k = WorkspaceKernel(fake, "ws")
        block = await k.render_facts_block(budget_chars=800)
        assert "<workspace_facts>" in block

        # The "Objective" header MUST appear before "Project:" (identity)
        # and "Established" (observations).
        obj_idx = block.index("Objective (user-pinned")
        proj_idx = block.index("Project:")
        est_idx = block.index("Established")
        assert obj_idx < proj_idx < est_idx
        # Body text rendered (header stripped).
        assert "Ship the cross-modal moat" in block
        # Closing directive mentions the no-edit-without-permission
        # rule when an objective is present.
        assert "objective.md" in block.lower()
        assert "permission" in block.lower()

    @pytest.mark.asyncio
    async def test_objective_clipped_at_budget(self):
        """A user who writes a paragraph-long objective shouldn't blow
        the facts budget. The render clips with an ellipsis."""
        long_body = "X" * 1000
        seeded = (
            "# Session Objective\n<!-- header -->\n\n"
            f"{long_body}\n"
        )
        fake = _FakeContainer(files={OBJECTIVE_MD: seeded})
        k = WorkspaceKernel(fake, "ws")
        block = await k.render_facts_block(
            budget_chars=600, objective_budget=200,
        )
        # The objective section should be capped to ~200 chars.
        assert "…" in block
        # Total block should fit reasonably close to the budget.
        assert len(block) < 800

    @pytest.mark.asyncio
    async def test_objective_only_renders_without_identity_or_observations(self):
        """An objective with nothing else still produces a useful
        facts block — load-bearing for fresh workspaces where
        identity detection hasn't run yet."""
        seeded = "Add real-time request logging to the research agent UI."
        fake = _FakeContainer(files={OBJECTIVE_MD: seeded})
        k = WorkspaceKernel(fake, "ws")
        block = await k.render_facts_block()
        assert "<workspace_facts>" in block
        assert "Objective" in block
        assert seeded in block
        # No identity → no "Project:" line.
        assert "Project:" not in block


class TestReadIdentity:
    @pytest.mark.asyncio
    async def test_missing_file_returns_empty_manifest(self):
        from augmentum.coder.identity import IdentityManifest

        fake = _FakeContainer()
        k = WorkspaceKernel(fake, "ws")
        manifest = await k.read_identity()
        assert isinstance(manifest, IdentityManifest)
        assert manifest.detected == {}

    @pytest.mark.asyncio
    async def test_present_file_returns_parsed_manifest(self):
        text = """\
[meta]
detector_version = 1
last_detected_at = 100.0

[detected]
languages = ["rust"]

[detected.rust]
package_manager = "cargo"

[asserted]

"""
        fake = _FakeContainer(files={IDENTITY_TOML: text})
        k = WorkspaceKernel(fake, "ws")
        manifest = await k.read_identity()
        assert manifest.detected["languages"] == ["rust"]
        assert manifest.detected["rust"]["package_manager"] == "cargo"


@pytest.mark.asyncio
async def test_native_sys_text_includes_facts_block_when_present(monkeypatch):
    """End-to-end: when identity.toml + observations.jsonl exist in
    the workspace AND both kernel_v2 and inline_facts are on, native's
    first request system message must contain the <workspace_facts>
    block. This is the load-bearing wiring test."""
    from augmentum.coder.observations import (
        Observation,
        serialize_observations,
    )
    from augmentum.coder.workspace_kernel import OBSERVATIONS_JSONL
    from augmentum.config import settings as _settings
    from augmentum.models.base import InternalStreamChunk
    from augmentum.modes.coder.handler import CoderHandler

    monkeypatch.setattr(_settings, "coder_kernel_v2", True)
    monkeypatch.setattr(_settings, "coder_kernel_inline_facts", True)

    from tests.test_coder_handler import (
        _ExtendedContainerManager,
        _FakeChunk,
        _FakeTool,
        _make_request,
    )

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )

    class _ExtendedWithFacts(_ExtendedContainerManager):
        """Pre-seed the workspace with identity + observations files."""
        def __init__(self):
            super().__init__()
            self._seeded_files = {
                IDENTITY_TOML: (
                    "[meta]\n"
                    "detector_version = 1\n"
                    "last_detected_at = 100.0\n\n"
                    "[detected]\n"
                    'languages = ["python"]\n'
                    'language_primary = "python"\n\n'
                    "[detected.python]\n"
                    'package_manager = "uv"\n'
                    'test_runner = "pytest"\n\n'
                    "[asserted]\n"
                ),
                OBSERVATIONS_JSONL: serialize_observations([
                    Observation(
                        ts=1.0, category="constraint",
                        fact="node 18 is locked",
                        source="user turn 1",
                        confidence="user_asserted",
                    ),
                ]),
            }

        async def file_read(self, workspace_id, path):
            if path in self._seeded_files:
                return self._seeded_files[path]
            return await super().file_read(workspace_id, path)

    class _Backend:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            yield _FakeChunk(content_delta=(
                "Got it. The project is python+uv with pytest."
            ))
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _Backend()
    handler = CoderHandler(
        backend,
        session_id="s",
        container_manager=_ExtendedWithFacts(),
        workspace_id="ws-facts",
    )

    chunks: list[InternalStreamChunk] = []
    # The cache is populated by _handle_stream's turn-start hook. The
    # direct-call test pattern bypasses that, so we invoke the hook
    # ourselves to mirror real-turn semantics.
    req = _make_request("inspect the workspace")
    await handler._refresh_kernel_facts(req)
    async for c in handler._act_native(req, workspace_context=""):
        chunks.append(c)

    sys_content = backend.requests[0].messages[0].content
    assert "<workspace_facts>" in sys_content
    # Identity summary made it in.
    assert "python" in sys_content
    assert "uv" in sys_content
    # Observation made it in (constraint priority).
    assert "node 18 is locked" in sys_content
    # Directive present.
    assert "trust the block" in sys_content.lower() or "known true" in sys_content.lower()


@pytest.mark.asyncio
async def test_native_auto_seeds_objective_from_first_substantive_message(monkeypatch):
    """End-to-end: a fresh native turn with a substantive user message
    must seed objective.md AND render it in the facts block."""
    from augmentum.config import settings as _settings
    from augmentum.models.base import InternalStreamChunk
    from augmentum.modes.coder.handler import CoderHandler

    monkeypatch.setattr(_settings, "coder_kernel_v2", True)
    monkeypatch.setattr(_settings, "coder_kernel_inline_facts", True)
    monkeypatch.setattr(_settings, "coder_kernel_auto_seed_objective", True)

    from tests.test_coder_handler import (
        _ExtendedContainerManager,
        _FakeChunk,
        _FakeTool,
        _make_request,
    )

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )

    class _SeedingContainer(_ExtendedContainerManager):
        """Records writes so the test can verify objective.md got
        seeded with the user's first ask."""
        def __init__(self):
            super().__init__()
            self._files: dict[str, str] = {}
            self.writes: dict[str, str] = {}

        async def file_read(self, workspace_id, path):
            if path in self._files:
                return self._files[path]
            return await super().file_read(workspace_id, path)

        async def file_write(self, workspace_id, path, content):
            self.writes[path] = content
            self._files[path] = content
            return await super().file_write(workspace_id, path, content)

    class _Backend:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            yield _FakeChunk(content_delta=(
                "Acknowledged. The original ask is clearly captured."
            ))
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    cm = _SeedingContainer()
    backend = _Backend()
    handler = CoderHandler(
        backend,
        session_id="s",
        container_manager=cm,
        workspace_id="ws-fresh",
    )

    user_ask = (
        "Build a request-logging panel for the research agent UI "
        "that shows plan, step, and synthesis calls with timestamps."
    )
    chunks: list[InternalStreamChunk] = []
    # Cache populated by the turn-start hook (mirrors _handle_stream).
    req = _make_request(user_ask)
    await handler._refresh_kernel_facts(req)
    async for c in handler._act_native(req, workspace_context=""):
        chunks.append(c)

    # objective.md got seeded with the user's ask.
    assert OBJECTIVE_MD in cm.writes
    assert "request-logging panel" in cm.writes[OBJECTIVE_MD]
    # The seeded header is present so a user inspecting the file
    # knows what it is.
    assert "# Session Objective" in cm.writes[OBJECTIVE_MD]

    # The native sys_text contains the rendered objective.
    sys_content = backend.requests[0].messages[0].content
    assert "<workspace_facts>" in sys_content
    assert "Objective (user-pinned" in sys_content
    assert "request-logging panel" in sys_content
    # The "do not edit without permission" directive surfaces.
    assert "permission" in sys_content.lower()


@pytest.mark.asyncio
async def test_native_skips_seed_for_short_message(monkeypatch):
    """A short or conversational first message must NOT seed
    objective.md — pinning 'hi' as the session goal would be a
    pathological state."""
    from augmentum.config import settings as _settings
    from augmentum.modes.coder.handler import CoderHandler

    monkeypatch.setattr(_settings, "coder_kernel_v2", True)
    monkeypatch.setattr(_settings, "coder_kernel_inline_facts", True)
    monkeypatch.setattr(_settings, "coder_kernel_auto_seed_objective", True)

    from tests.test_coder_handler import (
        _ExtendedContainerManager,
        _FakeChunk,
        _FakeTool,
        _make_request,
    )

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )

    class _SeedingContainer(_ExtendedContainerManager):
        def __init__(self):
            super().__init__()
            self._files: dict[str, str] = {}
            self.writes: dict[str, str] = {}

        async def file_read(self, workspace_id, path):
            if path in self._files:
                return self._files[path]
            return await super().file_read(workspace_id, path)

        async def file_write(self, workspace_id, path, content):
            self.writes[path] = content
            self._files[path] = content
            return await super().file_write(workspace_id, path, content)

    class _Backend:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            yield _FakeChunk(content_delta=(
                "Hello back. Let me know how I can help today."
            ))
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    cm = _SeedingContainer()
    backend = _Backend()
    handler = CoderHandler(
        backend,
        session_id="s",
        container_manager=cm,
        workspace_id="ws-greeting",
    )

    req = _make_request("hi")
    # Call the turn-start hook explicitly so the seed gate is
    # exercised — the gate is what we're testing here.
    await handler._refresh_kernel_facts(req)
    async for _ in handler._act_native(req, workspace_context=""):
        pass

    # objective.md NOT seeded — message was too short.
    assert OBJECTIVE_MD not in cm.writes
    sys_content = backend.requests[0].messages[0].content
    assert "Objective (user-pinned" not in sys_content


@pytest.mark.asyncio
async def test_native_sys_text_omits_facts_block_when_flag_off(monkeypatch):
    """The inline-facts setting is the opt-out switch for strong
    models that prefer the pure on-demand pattern. With it disabled,
    the facts block must NOT appear even when both files exist."""
    from augmentum.coder.observations import (
        Observation,
        serialize_observations,
    )
    from augmentum.coder.workspace_kernel import OBSERVATIONS_JSONL
    from augmentum.config import settings as _settings
    from augmentum.models.base import InternalStreamChunk
    from augmentum.modes.coder.handler import CoderHandler

    monkeypatch.setattr(_settings, "coder_kernel_v2", True)
    monkeypatch.setattr(_settings, "coder_kernel_inline_facts", False)

    from tests.test_coder_handler import (
        _ExtendedContainerManager,
        _FakeChunk,
        _FakeTool,
        _make_request,
    )

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )

    class _ExtendedWithFacts(_ExtendedContainerManager):
        def __init__(self):
            super().__init__()
            self._seeded_files = {
                IDENTITY_TOML: (
                    "[meta]\ndetector_version = 1\n[detected]\n"
                    'languages = ["python"]\n[asserted]\n'
                ),
                OBSERVATIONS_JSONL: serialize_observations([
                    Observation(
                        ts=1.0, category="constraint",
                        fact="don't render me", source="t1",
                    ),
                ]),
            }

        async def file_read(self, workspace_id, path):
            if path in self._seeded_files:
                return self._seeded_files[path]
            return await super().file_read(workspace_id, path)

    class _Backend:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            yield _FakeChunk(content_delta=(
                "Acknowledged. Inspect-only turn — no further action needed."
            ))
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _Backend()
    handler = CoderHandler(
        backend,
        session_id="s",
        container_manager=_ExtendedWithFacts(),
        workspace_id="ws-no-facts",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        _make_request("hi"), workspace_context="",
    ):
        chunks.append(c)

    sys_content = backend.requests[0].messages[0].content
    assert "<workspace_facts>" not in sys_content
    assert "don't render me" not in sys_content
    # The kernel HINT is still present — opt-out is for the inline
    # facts block, not for telling the model the dir exists.
    assert "<workspace_kernel>" in sys_content


@pytest.mark.asyncio
async def test_native_sys_text_omits_kernel_hint_when_flag_off(monkeypatch):
    """Flag-off path: native sys_text falls back to the pre-kernel shape."""
    from augmentum.config import settings as _settings
    from augmentum.models.base import InternalStreamChunk
    from augmentum.modes.coder.handler import CoderHandler

    monkeypatch.setattr(_settings, "coder_kernel_v2", False)

    from tests.test_coder_handler import (
        _ExtendedContainerManager,
        _FakeChunk,
        _FakeTool,
        _make_request,
    )

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )

    class _RecordingBackend:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            yield _FakeChunk(content_delta=(
                "Hello there. No need for tools on this one."
            ))
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _RecordingBackend()
    handler = CoderHandler(
        backend,
        session_id="s",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        _make_request("hi"), workspace_context="",
    ):
        chunks.append(c)

    sys_msg = backend.requests[0].messages[0]
    assert "<workspace_kernel>" not in sys_msg.content
