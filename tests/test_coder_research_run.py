"""Tests for the ``coder_research_run`` job handler (autonomous
improvement loop: propose → implement → measure → keep/revert).

The handler's contract, locked in here:
  1. Malformed payloads fail loudly (ValueError) — never a silent no-op:
     missing fields, an unstated direction (never assumed), and missing
     intentions (no payload value AND no OBJECTIVES.md).
  2. An unparseable baseline is a hard error before any agent turn runs.
  3. The loop keeps a strictly-improving experiment (git checkpoint
     stays) and reverts a regressing one (git_revert back to the last
     good sha), with both outcomes appended to RESEARCH_LOG.md.
  4. Score parsing prefers the last ``SCORE:`` marker and falls back to
     the last bare-number line; garbage yields None (never zero).
  5. A completion notification summarizes kept/ran + baseline→best.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.jobs.context import JobContext
from augmentum.jobs.handlers import coder_research_run as mod
from augmentum.models.base import InternalStreamChunk

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestParseScore:
    def test_score_marker_wins(self):
        assert mod.parse_score("noise\nSCORE: 12.5\nmore") == 12.5

    def test_last_score_marker_wins(self):
        assert mod.parse_score("SCORE: 1\nblah\nSCORE: 2.75") == 2.75

    def test_bare_number_last_line_fallback(self):
        assert mod.parse_score("running bench...\n0.4173") == 0.4173

    def test_bare_number_skips_trailing_noise(self):
        assert mod.parse_score("3.14\ndone.") == 3.14

    def test_scientific_notation(self):
        assert mod.parse_score("SCORE: 1.2e-3") == 1.2e-3

    def test_negative(self):
        assert mod.parse_score("SCORE: -7") == -7.0

    def test_garbage_is_none_not_zero(self):
        assert mod.parse_score("error: no benchmark found") is None

    def test_empty_is_none(self):
        assert mod.parse_score("") is None


class TestIsImprovement:
    def test_minimize(self):
        assert mod.is_improvement(9.0, 10.0, "minimize")
        assert not mod.is_improvement(10.0, 10.0, "minimize")
        assert not mod.is_improvement(11.0, 10.0, "minimize")

    def test_maximize(self):
        assert mod.is_improvement(11.0, 10.0, "maximize")
        assert not mod.is_improvement(10.0, 10.0, "maximize")

    def test_min_delta_margin(self):
        assert not mod.is_improvement(9.99, 10.0, "minimize", min_delta=0.5)
        assert mod.is_improvement(9.4, 10.0, "minimize", min_delta=0.5)


class TestExperimentSummary:
    def test_experiment_line_extracted(self):
        text = "I changed the cache.\nEXPERIMENT: memoize the hot loop"
        assert mod.extract_experiment_summary(text) == "memoize the hot loop"

    def test_fallback_first_line(self):
        assert mod.extract_experiment_summary("Did a thing.\nmore") == "Did a thing."

    def test_empty(self):
        assert mod.extract_experiment_summary("") == "(no summary)"


class TestPromptRendering:
    def test_history_feeds_back_into_prompt(self):
        experiments = [
            {"index": 1, "verdict": "KEPT", "summary": "memoized", "score": 8.0},
            {"index": 2, "verdict": "REVERTED", "summary": "unrolled", "score": 12.0},
        ]
        prompt = mod.build_experiment_prompt(
            index=3, total=5, intentions="make bench.py fast",
            objective_command="python bench.py", direction="minimize",
            baseline=10.0, best=8.0, experiments=experiments,
        )
        assert "make bench.py fast" in prompt
        assert "python bench.py" in prompt
        assert "[KEPT] memoized" in prompt
        assert "[REVERTED] unrolled" in prompt
        assert "LOWER" in prompt
        assert "EXPERIMENT:" in prompt


# ---------------------------------------------------------------------------
# Handler stubs
# ---------------------------------------------------------------------------


class _StubJobStore:
    def __init__(self):
        self.progress: list[tuple[float, str]] = []
        self.cancel = False

    async def update_progress(self, job_id, *, progress, stage=""):
        self.progress.append((progress, stage))

    async def is_cancel_requested(self, job_id):
        return self.cancel


def _ctx(payload, store=None):
    return JobContext(
        job_id="job_research1",
        user_id="usr_test",
        job_type=mod.JOB_TYPE,
        payload=payload,
        store=store or _StubJobStore(),
    )


class _StubContainers:
    """Scriptable ContainerManager stand-in.

    ``objective_outputs`` is consumed one entry per objective run
    (baseline first). ``checkpoint_shas`` is consumed one entry per
    git_checkpoint call AFTER the baseline checkpoint (which returns
    None — clean tree).
    """

    def __init__(self, *, objective_outputs, checkpoint_shas, objectives_md=""):
        self.objective_outputs = list(objective_outputs)
        self.checkpoint_shas = list(checkpoint_shas)
        self.objectives_md = objectives_md
        self.files: dict[str, str] = {}
        self.checkpoints: list[str] = []
        self.reverts: list[str] = []

    async def file_read(self, workspace_id, path):
        if path in self.files:
            return self.files[path]
        if path == mod._OBJECTIVES_FILE and self.objectives_md:
            return self.objectives_md
        raise FileNotFoundError(path)

    async def file_write(self, workspace_id, path, content):
        self.files[path] = content

    async def git_checkpoint(self, workspace_id, message):
        self.checkpoints.append(message)
        if message == "research: baseline":
            return None  # clean tree at mission start
        return self.checkpoint_shas.pop(0) if self.checkpoint_shas else None

    async def git_revert(self, workspace_id, commit_hash):
        self.reverts.append(commit_hash)
        return True

    async def run_command(self, workspace_id, cmd, timeout=30.0, **kw):
        joined = " ".join(cmd)
        if "rev-parse" in joined:
            return "abc1234\n"
        # The objective run: ["bash", "-lc", objective_command]
        if not self.objective_outputs:
            raise AssertionError("objective run not scripted")
        return self.objective_outputs.pop(0)


class _StubRegistry:
    async def resolve_backend_with_fabric(self, model, *, user_id=""):
        return "backend", model


def _app(containers, broker=None):
    return SimpleNamespace(
        state=SimpleNamespace(
            provider_registry=_StubRegistry(),
            container_manager=containers,
            coder_run_broker=broker,
            state_manager=SimpleNamespace(backend=SimpleNamespace(conn=object())),
            notification_hub=None,
        ),
    )


def _turn_factory(answers):
    """Factory whose CoderHandler yields the next scripted answer."""
    remaining = list(answers)

    def factory(mode, backend, session_id, state, **kw):
        answer = remaining.pop(0) if remaining else "EXPERIMENT: nothing"

        class CoderHandler:  # noqa: N801 — name checked by handler under test
            async def handle_stream(self, request):
                yield InternalStreamChunk(
                    augmentum={"run_id": "run_r1", "phase": "act"},
                )
                yield InternalStreamChunk(content_delta=answer, done=True)

        return CoderHandler()

    return factory


class _StubPersistence:
    instances: list[_StubPersistence] = []

    def __init__(self, conn):
        self.saved = None
        _StubPersistence.instances.append(self)

    async def load_conversation(self, workspace_id, *, user_id=""):
        return []

    async def save_conversation(self, workspace_id, messages, *, user_id=""):
        self.saved = (workspace_id, messages, user_id)


@pytest.fixture(autouse=True)
def _patch_collaborators(monkeypatch):
    _StubPersistence.instances = []
    monkeypatch.setattr(
        "augmentum.state.coder_persistence.CoderPersistence", _StubPersistence,
    )
    notifications: list[dict] = []

    async def _fake_notify(app, **kw):
        notifications.append(kw)

    monkeypatch.setattr(mod, "_notify", _fake_notify)
    yield notifications


_PAYLOAD = {
    "workspace_id": "ws1",
    "model": "m-test",
    "objective_command": "python bench.py",
    "direction": "minimize",
    "intentions": "make bench.py as fast as possible",
    "experiments": 2,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        "missing", ["workspace_id", "model", "objective_command"],
    )
    async def test_missing_field_raises(self, missing):
        payload = dict(_PAYLOAD)
        payload[missing] = ""
        handler = mod.make_coder_research_run_handler(
            _app(_StubContainers(objective_outputs=[], checkpoint_shas=[])),
        )
        with pytest.raises(ValueError):
            await handler(_ctx(payload))

    async def test_direction_never_assumed(self):
        payload = dict(_PAYLOAD)
        payload["direction"] = ""
        handler = mod.make_coder_research_run_handler(
            _app(_StubContainers(objective_outputs=[], checkpoint_shas=[])),
        )
        with pytest.raises(ValueError, match="direction"):
            await handler(_ctx(payload))

    async def test_no_intentions_anywhere_raises(self):
        payload = dict(_PAYLOAD)
        payload["intentions"] = ""
        containers = _StubContainers(
            objective_outputs=[], checkpoint_shas=[], objectives_md="",
        )
        handler = mod.make_coder_research_run_handler(_app(containers))
        with pytest.raises(ValueError, match="intentions"):
            await handler(_ctx(payload))

    async def test_objectives_md_fallback_accepted(self, monkeypatch):
        payload = dict(_PAYLOAD)
        payload["intentions"] = ""
        payload["experiments"] = 1
        containers = _StubContainers(
            objective_outputs=["SCORE: 10", "SCORE: 9"],
            checkpoint_shas=["c1a1a1a"],
            objectives_md="speed up the renderer",
        )
        monkeypatch.setattr(
            "augmentum.proxy.handler_factory.get_handler_for_mode",
            _turn_factory(["EXPERIMENT: cache templates"]),
        )
        handler = mod.make_coder_research_run_handler(_app(containers))
        result = await handler(_ctx(payload))
        assert result["kept"] == 1
        log_text = containers.files[mod._LOG_FILE]
        assert "speed up the renderer" in log_text

    async def test_unparseable_baseline_is_hard_error(self):
        containers = _StubContainers(
            objective_outputs=["no numbers here"], checkpoint_shas=[],
        )
        handler = mod.make_coder_research_run_handler(_app(containers))
        with pytest.raises(RuntimeError, match="baseline"):
            await handler(_ctx(dict(_PAYLOAD)))


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class TestLoop:
    async def test_keep_then_revert(self, monkeypatch, _patch_collaborators):
        notifications = _patch_collaborators
        containers = _StubContainers(
            # baseline 10 → exp1 scores 8 (kept) → exp2 scores 9 (worse
            # than best=8, reverted).
            objective_outputs=["SCORE: 10", "SCORE: 8", "SCORE: 9"],
            checkpoint_shas=["c1a1a1a", "c2b2b2b"],
        )
        monkeypatch.setattr(
            "augmentum.proxy.handler_factory.get_handler_for_mode",
            _turn_factory([
                "done\nEXPERIMENT: memoize the hot loop",
                "done\nEXPERIMENT: unroll the parser",
            ]),
        )
        handler = mod.make_coder_research_run_handler(_app(containers))
        result = await handler(_ctx(dict(_PAYLOAD)))

        assert result["experiments"] == 2
        assert result["kept"] == 1
        assert result["baseline"] == 10.0
        assert result["best"] == 8.0

        # Exp2 reverted back to exp1's KEPT sha (the last good state).
        assert containers.reverts == ["c1a1a1a"]

        # The lab notebook is physically in the workspace and complete.
        log_text = containers.files[mod._LOG_FILE]
        assert "KEPT" in log_text and "memoize the hot loop" in log_text
        assert "REVERTED" in log_text and "unroll the parser" in log_text
        assert "Mission summary" in log_text

        # The log is git-ignored so reverts never eat it.
        assert "RESEARCH_LOG.md" in containers.files["/workspace/.gitignore"]

        # Completion notification summarizes the mission.
        assert notifications[-1]["ok"] is True
        assert "1/2" in notifications[-1]["title"]

        # One summary message appended to the workspace conversation.
        saved = _StubPersistence.instances[-1].saved
        assert saved is not None
        assert "Research mission finished" in saved[1][-1]["content"]

    async def test_broken_objective_reverts(self, monkeypatch):
        containers = _StubContainers(
            # exp1's change makes the objective print garbage → revert.
            objective_outputs=["SCORE: 10", "Traceback: boom"],
            checkpoint_shas=["c1a1a1a"],
        )
        monkeypatch.setattr(
            "augmentum.proxy.handler_factory.get_handler_for_mode",
            _turn_factory(["EXPERIMENT: risky rewrite"]),
        )
        payload = dict(_PAYLOAD)
        payload["experiments"] = 1
        handler = mod.make_coder_research_run_handler(_app(containers))
        result = await handler(_ctx(payload))
        assert result["kept"] == 0
        assert containers.reverts == ["abc1234"]  # back to baseline HEAD
        assert "BROKE OBJECTIVE" in containers.files[mod._LOG_FILE]

    async def test_noop_turn_skips_measurement(self, monkeypatch):
        containers = _StubContainers(
            # Only the baseline measurement happens — a no-op experiment
            # never runs the objective.
            objective_outputs=["SCORE: 10"],
            checkpoint_shas=[],  # checkpoint returns None → no changes
        )
        monkeypatch.setattr(
            "augmentum.proxy.handler_factory.get_handler_for_mode",
            _turn_factory(["EXPERIMENT: pondered only"]),
        )
        payload = dict(_PAYLOAD)
        payload["experiments"] = 1
        handler = mod.make_coder_research_run_handler(_app(containers))
        result = await handler(_ctx(payload))
        assert result["kept"] == 0
        assert containers.reverts == []
        assert "NO-OP" in containers.files[mod._LOG_FILE]
