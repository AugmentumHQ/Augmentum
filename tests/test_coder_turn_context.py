from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.modes.coder.handler import CoderHandler
from augmentum.modes.coder.runtime_truth import build_runtime_truth
from augmentum.modes.coder.turn_context import TurnContext, build_turn_context
from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeBackend,
    _FakeChunk,
    _make_request,
)


class _SnapshotStub:
    def __init__(self, tree: str = "") -> None:
        self.tree = tree
        self.refresh_calls = 0

    async def refresh_if_stale(self, force: bool = False) -> bool:  # noqa: ARG002
        self.refresh_calls += 1
        return True

    def render(self) -> str:
        return self.tree


class _RuntimeTruthCM:
    def __init__(self, *, image: str = "augmentum-workspace", output: str | None = None):
        self._docker = SimpleNamespace(
            containers=SimpleNamespace(get=self._get_container),
        )
        self._image = image
        self._output = output or (
            "python3\tPython 3.12.3\n"
            "node\tv18.19.1\n"
            "go\tgo version go1.24.12 linux/amd64\n"
            "rustc\tmissing\n"
            "java\tmissing\n"
            "pip\tpip 24.0 from /usr/lib/python3/dist-packages/pip (python 3.12)\n"
            "npm\t9.2.0\n"
            "cargo\tmissing\n"
        )

    async def _get_container(self, container_id):  # noqa: ARG002
        return SimpleNamespace(
            show=self._show_container,
        )

    async def _show_container(self):
        return {"Config": {"Image": self._image}}

    async def _get_workspace(self, workspace_id):  # noqa: ARG002
        return SimpleNamespace(container_id="cid-runtime")

    async def _run_command(self, workspace_id, cmd, timeout=None):  # noqa: ARG002
        return self._output


@pytest.mark.asyncio
async def test_build_runtime_truth_parses_probe_output():
    handler = SimpleNamespace(
        _container_manager=_RuntimeTruthCM(),
        _workspace_id="ws-runtime",
    )

    truth = await build_runtime_truth(handler=handler)

    assert truth.probe_succeeded is True
    assert truth.workspace_mode == "prebaked"
    assert truth.workspace_image == "augmentum-workspace"
    assert truth.observed_runtimes["python3"] == "Python 3.12.3"
    assert truth.observed_runtimes["go"].startswith("go version")
    assert truth.observed_runtimes["rustc"] == "missing"
    rendered = truth.render_block()
    assert "<runtime_truth>" in rendered
    assert "Workspace mode:" in rendered
    assert "Observed now (direct probe):" in rendered
    assert "the workspace guide describes the intended baseline, not proof" in rendered


@pytest.mark.asyncio
async def test_build_runtime_truth_marks_fallback_baseline_gaps():
    handler = SimpleNamespace(
        _container_manager=_RuntimeTruthCM(
            image="ubuntu:24.04",
            output=(
                "python3\tPython 3.12.3\n"
                "node\tv18.19.1\n"
                "go\tmissing\n"
                "rustc\tmissing\n"
                "java\tmissing\n"
                "pip\tpip 24.0\n"
                "npm\tmissing\n"
                "cargo\tmissing\n"
            ),
        ),
        _workspace_id="ws-runtime-fallback",
    )

    truth = await build_runtime_truth(handler=handler)

    assert truth.workspace_mode == "fallback"
    assert truth.workspace_image == "ubuntu:24.04"
    assert truth.missing_baseline == ("go", "npm")
    rendered = truth.render_block()
    assert "fallback image: ubuntu:24.04" in rendered
    assert "Not observed now (treat as unavailable until verified or installed):" in rendered
    assert "- go" in rendered
    assert "- npm" in rendered


@pytest.mark.asyncio
async def test_build_turn_context_prefers_digest(monkeypatch):
    async def _digest(cm, workspace_id):  # noqa: ARG001
        return "DIGEST"

    async def _repo_map(*args, **kwargs):  # noqa: ARG001, ANN002, ANN003
        raise AssertionError("repo_map should not run on digest path")

    monkeypatch.setattr("augmentum.coder.digest.build_project_digest", _digest)
    monkeypatch.setattr("augmentum.coder.repomap.build_repo_map", _repo_map)

    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-digest",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-digest",
    )
    snapshot = _SnapshotStub("<workspace_tree>")
    handler._workspace_snapshot = snapshot

    ctx = await build_turn_context(
        handler=handler,
        request=_make_request("rename foo to bar"),
    )

    assert ctx.digest == "DIGEST"
    assert ctx.runtime_truth is not None
    assert ctx.tree_is_authoritative is True
    assert ctx.to_plan_context() == "DIGEST"
    assert await ctx.to_act_context() == "DIGEST"
    assert snapshot.refresh_calls == 0


def test_turn_context_native_context_is_bounded_orientation():
    ctx = TurnContext(
        latest_input="fix the UI",
        user_goal="fix the UI",
        user_query="fix the UI",
        digest="DIGEST:" + ("x" * 4000),
        workspace_profile_block="<workspace_profile>\nnpm run dev\n</workspace_profile>",
    )

    native_context = ctx.to_native_context(
        prior_turns="<prior_turns>\nEdited: src/app.js\n</prior_turns>",
        runtime_truth_block="<runtime_truth>\nnode: v20\n</runtime_truth>",
        max_chars=5000,
    )

    assert native_context.startswith("## Native Context Prelude")
    assert "## Runtime Truth" in native_context
    assert "<prior_turns>" in native_context
    assert "## Workspace Profile" in native_context
    assert "npm run dev" in native_context
    assert "## Workspace Grounding" in native_context
    assert "DIGEST:" in native_context
    assert len(native_context) < 5400


@pytest.mark.asyncio
async def test_turn_context_native_context_skips_semantic_search(monkeypatch):
    search_called = False

    async def _search(*args, **kwargs):  # noqa: ARG001, ANN002, ANN003
        nonlocal search_called
        search_called = True
        return []

    monkeypatch.setattr("augmentum.coder.indexer.search_index", _search)

    ctx = TurnContext(
        latest_input="inspect app",
        user_goal="inspect app",
        user_query="inspect app",
        digest="DIGEST",
        _workspace_id="ws-native",
    )

    assert "DIGEST" in ctx.to_native_context()
    assert search_called is False


@pytest.mark.asyncio
async def test_build_turn_context_snapshot_repo_map_and_cached_search(monkeypatch):
    search_calls: list[tuple[str, str, int]] = []

    async def _digest(cm, workspace_id):  # noqa: ARG001
        return None

    async def _repo_map(cm, workspace_id, query="", skip_file_listing=False, **kwargs):  # noqa: ARG001, ANN003
        assert skip_file_listing is True
        return f"REPO::{query}"

    async def _search(workspace_id, query, limit=5):
        search_calls.append((workspace_id, query, limit))
        return [
            SimpleNamespace(
                file_path="src/app.py",
                start_line=10,
                end_line=18,
                content="File: src/app.py\nprint('hi')\n",
                score=0.91,
            ),
        ]

    monkeypatch.setattr("augmentum.coder.digest.build_project_digest", _digest)
    monkeypatch.setattr("augmentum.coder.repomap.build_repo_map", _repo_map)
    monkeypatch.setattr("augmentum.coder.indexer.search_index", _search)

    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-tree",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-tree",
    )
    snapshot = _SnapshotStub("<workspace_tree>\n- src/app.py")
    handler._workspace_snapshot = snapshot

    ctx = await build_turn_context(
        handler=handler,
        request=_make_request("inspect app.py"),
    )

    assert snapshot.refresh_calls == 1
    assert ctx.to_plan_context() == "<workspace_tree>\n- src/app.py\n\nREPO::inspect app.py"

    # Stable half (system prefix): grounding only — no query-dependent
    # semantic hits, which would mutate the prefix cache every turn.
    act_context = await ctx.to_act_context()
    assert act_context == "<workspace_tree>\n- src/app.py\n\nREPO::inspect app.py"
    assert "## Relevant Code" not in act_context

    # Dynamic half (runtime carrier): carries the semantic hits.
    dynamic_context = await ctx.to_act_dynamic_context()
    assert "## Relevant Code" in dynamic_context
    assert "src/app.py:10-18 (score 0.91)" in dynamic_context
    assert "print('hi')" in dynamic_context

    again = await ctx.to_act_dynamic_context()
    assert again == dynamic_context
    assert search_calls == [("ws-tree", "inspect app.py", 5)]


@pytest.mark.asyncio
async def test_build_turn_context_repo_map_only_branch(monkeypatch):
    async def _digest(cm, workspace_id):  # noqa: ARG001
        return None

    async def _repo_map(cm, workspace_id, query="", skip_file_listing=False, **kwargs):  # noqa: ARG001, ANN003
        assert skip_file_listing is False
        return "REPO-ONLY"

    monkeypatch.setattr("augmentum.coder.digest.build_project_digest", _digest)
    monkeypatch.setattr("augmentum.coder.repomap.build_repo_map", _repo_map)

    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-repo",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-repo",
    )
    snapshot = _SnapshotStub("")
    handler._workspace_snapshot = snapshot

    ctx = await build_turn_context(
        handler=handler,
        request=_make_request("explain the project"),
    )

    assert snapshot.refresh_calls == 1
    assert ctx.to_plan_context() == "REPO-ONLY"


@pytest.mark.asyncio
async def test_build_turn_context_fallback_only_applies_to_plan(monkeypatch):
    async def _digest(cm, workspace_id):  # noqa: ARG001
        return None

    async def _repo_map(cm, workspace_id, query="", skip_file_listing=False, **kwargs):  # noqa: ARG001, ANN003
        return ""

    monkeypatch.setattr("augmentum.coder.digest.build_project_digest", _digest)
    monkeypatch.setattr("augmentum.coder.repomap.build_repo_map", _repo_map)

    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-fallback",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-fallback",
    )
    handler._workspace_snapshot = _SnapshotStub("")

    async def _workspace_context():
        return "FALLBACK"

    handler._get_workspace_context = _workspace_context

    ctx = await build_turn_context(
        handler=handler,
        request=_make_request("summarize the workspace"),
    )

    assert ctx.to_plan_context() == "FALLBACK"
    assert await ctx.to_act_context() == ""


@pytest.mark.asyncio
async def test_build_turn_context_uses_300_char_query(monkeypatch):
    seen_queries: list[str] = []

    async def _digest(cm, workspace_id):  # noqa: ARG001
        return None

    async def _repo_map(cm, workspace_id, query="", skip_file_listing=False, **kwargs):  # noqa: ARG001, ANN003
        seen_queries.append(query)
        return "MAP"

    monkeypatch.setattr("augmentum.coder.digest.build_project_digest", _digest)
    monkeypatch.setattr("augmentum.coder.repomap.build_repo_map", _repo_map)

    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-query",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-query",
    )
    handler._workspace_snapshot = _SnapshotStub("")

    long_prompt = "x" * 350
    ctx = await build_turn_context(
        handler=handler,
        request=_make_request(long_prompt),
    )

    assert len(ctx.user_query) == 300
    assert seen_queries == ["x" * 300]


@pytest.mark.asyncio
async def test_handle_stream_builds_turn_context_once_for_continuation(monkeypatch):
    sentinel = SimpleNamespace(
        latest_input="continue please",
        user_goal="continue please",
        tree_is_authoritative=False,
        runtime_truth=None,
    )
    build_calls: list[str] = []
    act_contexts: list[object] = []
    monkeypatch.setenv("AUGMENTUM_STRICT_METADATA", "0")

    async def _build(*, handler, request):  # noqa: ARG001
        build_calls.append("called")
        return sentinel

    async def _act(self, request, turn_context):  # noqa: ARG002
        act_contexts.append(turn_context)
        yield self._meta_chunk(
            phase="executing",
            status="complete",
            model=request.model,
        )

    monkeypatch.setattr("augmentum.modes.coder.handler.build_turn_context", _build)
    monkeypatch.setattr(CoderHandler, "_act_phase", _act)

    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-cont",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-cont",
    )
    handler._state.plan = "Plan: keep going"

    chunks = []
    async for chunk in handler.handle_stream(_make_request("continue please")):
        chunks.append(chunk)

    assert build_calls == ["called"]
    assert act_contexts == [sentinel]
    assert any(
        c.augmentum
        and c.augmentum.get("phase") == "executing"
        and c.augmentum.get("status") == "continuation"
        for c in chunks
    )


@pytest.mark.asyncio
async def test_handle_stream_builds_one_turn_context_native_goes_straight_to_act(monkeypatch):
    """Native (the shipped strategy) bypasses the plan phase entirely and
    the single per-turn TurnContext flows straight to act. The old
    plan→act reuse assertion only holds on the FROZEN hybrid/canonical
    rollback paths (see the strategy switch in ``handle_stream``)."""
    sentinel = SimpleNamespace(
        latest_input="inspect the repo",
        user_goal="inspect the repo",
        tree_is_authoritative=False,
        runtime_truth=None,
    )
    build_calls: list[str] = []
    seen: list[tuple[str, object]] = []

    async def _build(*, handler, request):  # noqa: ARG001
        build_calls.append("called")
        return sentinel

    async def _plan(self, request, turn_context):  # noqa: ARG002
        seen.append(("plan", turn_context))
        self._state.plan = "Plan: inspect workspace\n1. Read files"
        yield self._meta_chunk(
            phase="planning",
            status="complete",
            model=request.model,
        )

    async def _act(self, request, turn_context):  # noqa: ARG002
        seen.append(("act", turn_context))
        yield self._meta_chunk(
            phase="executing",
            status="complete",
            model=request.model,
        )

    monkeypatch.setattr("augmentum.modes.coder.handler.build_turn_context", _build)
    monkeypatch.setattr(CoderHandler, "_plan_phase", _plan)
    monkeypatch.setattr(CoderHandler, "_act_phase", _act)

    handler = CoderHandler(
        _FakeBackend([_FakeChunk(done=True, finish_reason="stop")]),
        session_id="sess-fresh",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-fresh",
    )

    async for _ in handler.handle_stream(_make_request("inspect the repo")):
        pass

    assert build_calls == ["called"]
    assert seen == [("act", sentinel)]


@pytest.mark.asyncio
async def test_act_phase_native_receives_context_prelude(monkeypatch):
    seen_contexts: list[str] = []

    async def _native(self, request, workspace_context):  # noqa: ARG002
        seen_contexts.append(workspace_context)
        yield self._meta_chunk(
            phase="executing",
            status="complete",
            model=request.model,
        )

    monkeypatch.setattr(CoderHandler, "_act_native", _native)

    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-native-prelude",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-native-prelude",
        coder_strategy="native",
    )
    handler._runtime_truth_context_block = (
        "<runtime_truth>\nnode: v20\n</runtime_truth>"
    )
    handler._state.add_turn_summary({
        "turn_idx": 1,
        "user_goal": "fix previous bug",
        "files_read": ["src/app.js"],
        "files_edited": ["src/app.js"],
        "shell_commands": ["npm test"],
        "outcome": "done",
        "blockers": "",
        "created_at": 0,
    })

    ctx = TurnContext(
        latest_input="continue",
        user_goal="fix previous bug",
        user_query="continue",
        digest="DIGEST",
        workspace_profile_block="<workspace_profile>\nnpm run dev\n</workspace_profile>",
    )

    async for _ in handler._act_phase(_make_request("continue"), ctx):
        pass

    assert len(seen_contexts) == 1
    native_context = seen_contexts[0]
    assert "## Native Context Prelude" in native_context
    assert "node: v20" in native_context
    # ``<prior_turns>`` moved out of the native context prelude into a
    # user-role runtime carrier inserted by ``_act_native`` itself, so
    # the leading system prefix stays byte-stable across turns for
    # llama-server slot prefix-cache reuse. The prelude still carries
    # workspace profile + grounding (rarely-changing content).
    assert "<prior_turns" not in native_context
    assert "npm run dev" in native_context
    assert "DIGEST" in native_context
