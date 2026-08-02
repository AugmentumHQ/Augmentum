"""Tests for the Promise runtime (augmentum/promises/)."""
from __future__ import annotations

import pytest

from augmentum.promises import (
    ActEvent,
    ActEventKind,
    MissionRunner,
    Promise,
    PromiseContext,
    PromiseStatus,
    RunnerEventKind,
    Verification,
    VerificationKind,
    always_pass,
    file_verifier,
    parse_mission_json,
    parse_prose_plan,
    render_mission_log,
    shell_verifier,
)
from augmentum.promises.verify import default_verify_fns

# ---------------------------------------------------------------------------
# Models — serialization round-trip
# ---------------------------------------------------------------------------


class TestPromiseSerialization:
    def test_roundtrip_preserves_all_fields(self):
        original = Promise(
            description="install deps",
            verify=Verification(kind=VerificationKind.SHELL, spec={"cmd": "dpkg -l"}),
            max_attempts=3,
            metadata={"origin": "planner"},
        )
        restored = Promise.from_dict(original.to_dict())
        assert restored.description == "install deps"
        assert restored.verify.kind == VerificationKind.SHELL
        assert restored.verify.spec == {"cmd": "dpkg -l"}
        assert restored.max_attempts == 3
        assert restored.metadata == {"origin": "planner"}
        assert restored.status == PromiseStatus.PENDING

    def test_roundtrip_children(self):
        parent = Promise(
            description="build game",
            children=[
                Promise(description="compile", verify=Verification.always()),
                Promise(description="link", verify=Verification.always()),
            ],
        )
        restored = Promise.from_dict(parent.to_dict())
        assert len(restored.children) == 2
        assert restored.children[0].description == "compile"

    def test_verification_roundtrip(self):
        v = Verification(kind=VerificationKind.FILE, spec={"path": "/x", "must_exist": True})
        assert Verification.from_dict(v.to_dict()) == v


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_always_pass_returns_true():
    promise = Promise(description="x", verify=Verification.always())
    result = await always_pass(promise, evidence="hello")
    assert result.passed


@pytest.mark.asyncio
async def test_shell_verifier_pass_on_exit_zero():
    async def fake_shell(cmd: str, timeout: float) -> tuple[int, str]:
        assert cmd == "echo ok"
        return 0, "ok\n"

    verify = shell_verifier(fake_shell)
    promise = Promise(
        verify=Verification(kind=VerificationKind.SHELL, spec={"cmd": "echo ok"}),
    )
    result = await verify(promise, evidence=None)
    assert result.passed
    assert "ok" in result.reason


@pytest.mark.asyncio
async def test_shell_verifier_fail_on_exit_nonzero():
    async def fake_shell(cmd: str, timeout: float) -> tuple[int, str]:
        return 2, "not found\n"

    verify = shell_verifier(fake_shell)
    promise = Promise(
        verify=Verification(kind=VerificationKind.SHELL, spec={"cmd": "nope"}),
    )
    result = await verify(promise, evidence=None)
    assert not result.passed
    assert "exit 2" in result.reason


@pytest.mark.asyncio
async def test_shell_verifier_missing_cmd():
    verify = shell_verifier(lambda *_: None)  # type: ignore[arg-type]
    promise = Promise(verify=Verification(kind=VerificationKind.SHELL, spec={}))
    result = await verify(promise, evidence=None)
    assert not result.passed


@pytest.mark.asyncio
async def test_file_verifier_pass_when_exists():
    async def fake_shell(cmd: str, timeout: float) -> tuple[int, str]:
        assert "test -e" in cmd
        return 0, ""

    verify = file_verifier(fake_shell)
    promise = Promise(
        verify=Verification(kind=VerificationKind.FILE, spec={"path": "/workspace/a"}),
    )
    result = await verify(promise, evidence=None)
    assert result.passed


@pytest.mark.asyncio
async def test_file_verifier_fail_when_missing():
    async def fake_shell(cmd: str, timeout: float) -> tuple[int, str]:
        return 1, ""

    verify = file_verifier(fake_shell)
    promise = Promise(
        verify=Verification(kind=VerificationKind.FILE, spec={"path": "/nope"}),
    )
    result = await verify(promise, evidence=None)
    assert not result.passed
    assert "absent" in result.reason


@pytest.mark.asyncio
async def test_file_verifier_respects_must_exist_false():
    async def fake_shell(cmd: str, timeout: float) -> tuple[int, str]:
        return 1, ""  # file does NOT exist

    verify = file_verifier(fake_shell)
    promise = Promise(
        verify=Verification(
            kind=VerificationKind.FILE,
            spec={"path": "/should-be-gone", "must_exist": False},
        ),
    )
    result = await verify(promise, evidence=None)
    assert result.passed


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class TestRenderer:
    def test_empty_mission(self):
        out = render_mission_log([])
        assert "empty" in out.lower()

    def test_pending_status_shows_verify_preview(self):
        mission = [
            Promise(
                description="install deps",
                verify=Verification(kind=VerificationKind.SHELL, spec={"cmd": "apt-get install -y foo"}),
            ),
        ]
        out = render_mission_log(mission)
        assert "[ ]" in out
        assert "install deps" in out
        assert "verify:" in out
        assert "apt-get install" in out

    def test_fulfilled_shows_evidence(self):
        p = Promise(description="build", verify=Verification.always())
        p.status = PromiseStatus.FULFILLED
        p.evidence = "exit 0"
        out = render_mission_log([p])
        assert "[x]" in out
        assert "verified:" in out
        assert "exit 0" in out

    def test_rejected_shows_failure(self):
        p = Promise(description="run", verify=Verification.always())
        p.status = PromiseStatus.REJECTED
        p.evidence = "exit 127: command not found"
        out = render_mission_log([p])
        assert "[!]" in out
        assert "failed:" in out

    def test_nested_children_indent(self):
        child = Promise(description="subtask", verify=Verification.always())
        parent = Promise(description="top", verify=Verification.always(), children=[child])
        out = render_mission_log([parent])
        lines = out.splitlines()
        # Parent line is not indented past the header
        parent_line = next(line for line in lines if "top" in line)
        child_line = next(line for line in lines if "subtask" in line)
        assert child_line.index("[") > parent_line.index("[")

    def test_attempt_counter_shown_on_retry(self):
        p = Promise(description="retry me", verify=Verification.always(), max_attempts=3)
        p.attempts = 1
        out = render_mission_log([p])
        assert "attempt 2/3" in out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def _collect(gen):
    out = []
    async for ev in gen:
        out.append(ev)
    return out


def _make_act_fn(scripts: dict[str, list]):
    """Build an act_fn that emits pre-canned events per promise description.

    scripts: {promise_description: [ActEvent, ActEvent, ...]}
    """

    async def act(promise: Promise, ctx: PromiseContext):
        events = scripts.get(promise.description, [])
        for ev in events:
            yield ev

    return act


@pytest.mark.asyncio
async def test_runner_fulfills_simple_mission():
    mission = [
        Promise(description="step1", verify=Verification.always()),
        Promise(description="step2", verify=Verification.always()),
    ]
    act = _make_act_fn({
        "step1": [ActEvent(kind=ActEventKind.ATTEMPT_COMPLETE, evidence="did it")],
        "step2": [ActEvent(kind=ActEventKind.ATTEMPT_COMPLETE, evidence="also did")],
    })
    verify_fns = {VerificationKind.ALWAYS: always_pass}
    runner = MissionRunner()
    events = await _collect(runner.run(mission, act, verify_fns))

    kinds = [e.kind for e in events]
    assert kinds[0] == RunnerEventKind.MISSION_STARTED
    assert RunnerEventKind.MISSION_COMPLETED in kinds
    assert mission[0].status == PromiseStatus.FULFILLED
    assert mission[1].status == PromiseStatus.FULFILLED


@pytest.mark.asyncio
async def test_runner_retries_then_rejects_on_verify_failure():
    async def always_fail(promise, evidence):
        from augmentum.promises.verify import VerifyResult
        return VerifyResult(False, "nope")

    mission = [Promise(description="doomed", verify=Verification.always(), max_attempts=2)]
    act = _make_act_fn({
        "doomed": [ActEvent(kind=ActEventKind.ATTEMPT_COMPLETE, evidence="x")],
    })
    verify_fns = {VerificationKind.ALWAYS: always_fail}

    runner = MissionRunner()
    events = await _collect(runner.run(mission, act, verify_fns))

    kinds = [e.kind for e in events]
    assert RunnerEventKind.PROMISE_RETRY in kinds
    assert RunnerEventKind.PROMISE_REJECTED in kinds
    assert RunnerEventKind.MISSION_FAILED in kinds
    assert mission[0].status == PromiseStatus.REJECTED
    assert mission[0].attempts == 2


@pytest.mark.asyncio
async def test_runner_cannot_fulfill_skips_retry():
    """CANNOT_FULFILL rejects immediately, no retry, no verifier call."""
    verify_calls = []

    async def tracking_verify(promise, evidence):
        verify_calls.append(promise.id)
        from augmentum.promises.verify import VerifyResult
        return VerifyResult(True, "ok")

    mission = [Promise(description="giveup", verify=Verification.always(), max_attempts=5)]
    act = _make_act_fn({
        "giveup": [ActEvent(kind=ActEventKind.CANNOT_FULFILL, payload="no way")],
    })
    runner = MissionRunner()
    events = await _collect(runner.run(mission, act, {VerificationKind.ALWAYS: tracking_verify}))

    assert mission[0].status == PromiseStatus.REJECTED
    assert mission[0].attempts == 0  # never retried
    assert not verify_calls  # verifier never ran


@pytest.mark.asyncio
async def test_runner_progress_events_passthrough():
    progress_payloads = []

    async def tracking_act(promise: Promise, ctx: PromiseContext):
        yield ActEvent(kind=ActEventKind.PROGRESS, payload={"tool": "shell"})
        yield ActEvent(kind=ActEventKind.PROGRESS, payload={"tool": "file_read"})
        yield ActEvent(kind=ActEventKind.ATTEMPT_COMPLETE, evidence="done")

    runner = MissionRunner()
    events = await _collect(runner.run(
        [Promise(description="x", verify=Verification.always())],
        tracking_act,
        {VerificationKind.ALWAYS: always_pass},
    ))
    progress = [e.payload for e in events if e.kind == RunnerEventKind.PROMISE_PROGRESS]
    assert progress == [{"tool": "shell"}, {"tool": "file_read"}]


@pytest.mark.asyncio
async def test_runner_decomposition_children_run_first():
    """A NEEDS_DECOMPOSITION event installs children and the runner
    descends into them before re-running the parent."""

    parent_attempts = []

    async def parent_act(promise: Promise, ctx: PromiseContext):
        parent_attempts.append(promise.attempts)
        if not promise.children:
            # First attempt: decompose
            yield ActEvent(
                kind=ActEventKind.NEEDS_DECOMPOSITION,
                payload=[
                    Promise(description="child1", verify=Verification.always()),
                    Promise(description="child2", verify=Verification.always()),
                ],
            )
        else:
            # Re-entered after children fulfilled
            yield ActEvent(kind=ActEventKind.ATTEMPT_COMPLETE, evidence="rolled up")

    child_act = _make_act_fn({
        "child1": [ActEvent(kind=ActEventKind.ATTEMPT_COMPLETE, evidence="c1")],
        "child2": [ActEvent(kind=ActEventKind.ATTEMPT_COMPLETE, evidence="c2")],
    })

    async def composite_act(promise: Promise, ctx: PromiseContext):
        if promise.description == "parent":
            async for ev in parent_act(promise, ctx):
                yield ev
        else:
            async for ev in child_act(promise, ctx):
                yield ev

    mission = [Promise(description="parent", verify=Verification.always())]
    runner = MissionRunner()
    events = await _collect(runner.run(mission, composite_act, {VerificationKind.ALWAYS: always_pass}))

    kinds = [e.kind for e in events]
    assert RunnerEventKind.PROMISE_DECOMPOSED in kinds
    assert RunnerEventKind.MISSION_COMPLETED in kinds
    # Parent was visited twice: once for decomp, once for final attempt
    assert len(parent_attempts) == 2
    assert mission[0].status == PromiseStatus.FULFILLED
    assert all(c.status == PromiseStatus.FULFILLED for c in mission[0].children)


@pytest.mark.asyncio
async def test_runner_child_rejection_cascades_to_parent():
    async def always_fail(promise, evidence):
        from augmentum.promises.verify import VerifyResult
        return VerifyResult(False, "bad")

    async def composite_act(promise: Promise, ctx: PromiseContext):
        if promise.description == "parent":
            yield ActEvent(
                kind=ActEventKind.NEEDS_DECOMPOSITION,
                payload=[Promise(description="child", verify=Verification.always(), max_attempts=1)],
            )
        else:
            yield ActEvent(kind=ActEventKind.ATTEMPT_COMPLETE, evidence="x")

    mission = [Promise(description="parent", verify=Verification.always())]
    runner = MissionRunner()
    events = await _collect(runner.run(mission, composite_act, {VerificationKind.ALWAYS: always_fail}))

    assert mission[0].children[0].status == PromiseStatus.REJECTED
    assert mission[0].status == PromiseStatus.REJECTED  # cascaded
    assert any(e.kind == RunnerEventKind.MISSION_FAILED for e in events)


@pytest.mark.asyncio
async def test_runner_act_exception_rejects_cleanly():
    async def boom(promise: Promise, ctx: PromiseContext):
        yield ActEvent(kind=ActEventKind.PROGRESS, payload="ok")
        raise RuntimeError("backend fell over")

    mission = [Promise(description="x", verify=Verification.always())]
    runner = MissionRunner()
    events = await _collect(runner.run(mission, boom, default_verify_fns(lambda _c, _t: None)))  # type: ignore[arg-type]

    assert mission[0].status == PromiseStatus.REJECTED
    assert "backend fell over" in (mission[0].evidence or "")
    assert any(e.kind == RunnerEventKind.MISSION_FAILED for e in events)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParseMissionJson:
    def test_clean_array(self):
        text = '[{"desc": "install", "verify": {"kind": "shell", "cmd": "x"}}]'
        mission = parse_mission_json(text)
        assert len(mission) == 1
        assert mission[0].description == "install"
        assert mission[0].verify.kind == VerificationKind.SHELL
        assert mission[0].verify.spec == {"cmd": "x"}

    def test_flat_verify_form(self):
        """Accepts shell spec keys at top level of verify dict."""
        text = '[{"desc": "build", "verify": {"kind": "file", "path": "/a", "must_exist": true}}]'
        mission = parse_mission_json(text)
        assert mission[0].verify.spec == {"path": "/a", "must_exist": True}

    def test_nested_spec_form(self):
        """Also accepts the canonical {kind, spec} shape."""
        text = '[{"desc": "x", "verify": {"kind": "shell", "spec": {"cmd": "y"}}}]'
        mission = parse_mission_json(text)
        assert mission[0].verify.spec == {"cmd": "y"}

    def test_description_key_alias(self):
        text = '[{"description": "x", "verify": {"kind": "always"}}]'
        mission = parse_mission_json(text)
        assert mission[0].description == "x"

    def test_markdown_fences_stripped(self):
        text = '```json\n[{"desc": "a", "verify": {"kind": "always"}}]\n```'
        mission = parse_mission_json(text)
        assert len(mission) == 1

    def test_array_embedded_in_prose(self):
        text = 'Sure, here is the plan:\n[{"desc": "a", "verify": {"kind": "always"}}]\nHope that helps!'
        mission = parse_mission_json(text)
        assert len(mission) == 1

    def test_invalid_kind_falls_back_to_always(self):
        text = '[{"desc": "x", "verify": {"kind": "bogus"}}]'
        mission = parse_mission_json(text)
        assert mission[0].verify.kind == VerificationKind.ALWAYS

    def test_missing_description_dropped(self):
        text = '[{"verify": {"kind": "always"}}, {"desc": "ok", "verify": {"kind": "always"}}]'
        mission = parse_mission_json(text)
        assert [p.description for p in mission] == ["ok"]

    def test_not_a_list_returns_empty(self):
        assert parse_mission_json('{"desc": "x"}') == []
        assert parse_mission_json("not json") == []
        assert parse_mission_json("") == []


class TestParseProsePlan:
    def test_single_line_numbered_steps(self):
        text = (
            "Plan: Install and run nsnake "
            "1. Clone nsnake from GitHub to /workspace/nsnake "
            "2. Build the binary with `make` "
            "3. Run the game with a short timeout"
        )
        mission = parse_prose_plan(text)
        assert len(mission) == 3
        assert "Clone nsnake" in mission[0].description
        assert "Build" in mission[1].description
        assert all(p.verify.kind == VerificationKind.ALWAYS for p in mission)

    def test_multiline_numbered_steps(self):
        text = """Here's what I'll do:
1. Install dependencies
2. Clone the repository
3. Build the binary
4. Run the smoke test
"""
        mission = parse_prose_plan(text)
        assert [p.description for p in mission] == [
            "Install dependencies",
            "Clone the repository",
            "Build the binary",
            "Run the smoke test",
        ]

    def test_paren_markers(self):
        mission = parse_prose_plan("1) first 2) second 3) third")
        assert len(mission) == 3

    def test_rejects_decimal_numbers_in_prose(self):
        """'3.14' shouldn't split a single line into steps."""
        text = "pi is 3.14 and e is 2.71"
        assert parse_prose_plan(text) == []

    def test_rejects_single_step(self):
        """Single-step 'plan' is usually a false match."""
        assert parse_prose_plan("1. just do it") == []

    def test_rejects_non_monotonic_numbers(self):
        """Out-of-order numbers usually means we're matching citations, not a plan."""
        text = "Ref 5. says X. Ref 2. says Y."
        assert parse_prose_plan(text) == []

    def test_empty_input(self):
        assert parse_prose_plan("") == []
        assert parse_prose_plan(None) == []  # type: ignore[arg-type]

    def test_skips_section_headers(self):
        """Skip entries whose body starts with 'plan'/'steps'/etc."""
        text = "1. plan 2. Clone the repo 3. Build it"
        mission = parse_prose_plan(text)
        assert "plan" not in [p.description.lower() for p in mission]
        assert len(mission) == 2


@pytest.mark.asyncio
async def test_runner_cap_terminates():
    """max_promise_attempts cap trips when a verifier flip-flops forever.

    We simulate a promise that always fails verification but has an
    enormous max_attempts; the runner should stop via its own cap.
    """

    async def always_fail(promise, evidence):
        from augmentum.promises.verify import VerifyResult
        return VerifyResult(False, "keep failing")

    mission = [
        Promise(
            description=f"p{i}", verify=Verification.always(), max_attempts=1000,
        )
        for i in range(10)
    ]
    act = _make_act_fn({
        p.description: [ActEvent(kind=ActEventKind.ATTEMPT_COMPLETE, evidence="x")]
        for p in mission
    })
    runner = MissionRunner(max_promise_attempts=5)
    events = await _collect(runner.run(mission, act, {VerificationKind.ALWAYS: always_fail}))

    assert any(
        e.kind == RunnerEventKind.MISSION_FAILED and "cap" in str(e.payload).lower()
        for e in events
    )
