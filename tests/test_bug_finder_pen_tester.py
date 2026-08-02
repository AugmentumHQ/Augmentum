"""Phase 1c tests — pen_tester subagent role + orchestrator stage 5.5.

Covers:
* Parser — verdict + evidence shapes, malformed input, empty evidence
* Role allow-list — PEN_TESTER includes pen-test tools; other roles
  still disjoint
* role_models — Role.PEN_TESTER falls back to verifier when not set
* Orchestrator integration — _run_pen_test_leg attaches verdict notes,
  downgrades refuted findings, tears down the registry, never raises
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from augmentum.agents.tools import (
    DETECTOR_TOOL_NAMES,
    FIXER_TOOL_NAMES,
    INVESTIGATOR_TOOL_NAMES,
    LEAD_TOOL_NAMES,
    PEN_TEST_TOOL_NAMES,
    PEN_TESTER_TOOL_NAMES,
    PLANNER_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    VERIFIER_TOOL_NAMES,
)
from augmentum.bug_finder.findings import Finding, FindingStatus
from augmentum.bug_finder.pen_tester import (
    PenTestVerdict,
    ProbeEvidence,
    parse_pen_tester_output,
    verdict_to_note,
)
from augmentum.bug_finder.role_models import Role, RoleModelConfig

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parse_confirmed_verdict_with_evidence() -> None:
    text = """
Some reasoning text...

```json
{
  "finding_id": "f_001",
  "verdict": "confirmed",
  "rationale": "Single-quote payload triggered a SQL error in the response.",
  "evidence": [
    {"method": "POST", "url": "http://localhost:8080/api/login",
     "status": 500, "expected_status": 400, "note": "SQL error in body"}
  ]
}
```
"""
    v = parse_pen_tester_output(text)
    assert v is not None
    assert v.finding_id == "f_001"
    assert v.is_confirmed
    assert not v.is_refuted
    assert len(v.evidence) == 1
    e = v.evidence[0]
    assert e.method == "POST"
    assert e.status == 500


def test_parse_refuted_verdict_with_empty_evidence() -> None:
    text = """```json
{
  "finding_id": "f_002",
  "verdict": "refuted",
  "rationale": "Three payload variants all returned 400 with no error leak.",
  "evidence": []
}
```"""
    v = parse_pen_tester_output(text)
    assert v is not None
    assert v.is_refuted
    assert v.evidence == ()


def test_parse_inconclusive_verdict() -> None:
    text = """```json
{
  "finding_id": "f_003",
  "verdict": "inconclusive",
  "rationale": "Could not boot the app — no recognizable entrypoint.",
  "evidence": []
}
```"""
    v = parse_pen_tester_output(text)
    assert v is not None
    assert v.verdict == "inconclusive"


def test_parse_picks_last_json_block_when_multiple() -> None:
    """The model often emits a thinking block then a final answer
    block. We must take the FINAL one as authoritative."""
    text = """
First attempt:
```json
{"finding_id": "f", "verdict": "confirmed", "rationale": "", "evidence": []}
```

After reconsidering:
```json
{"finding_id": "f", "verdict": "refuted", "rationale": "thought again", "evidence": []}
```
"""
    v = parse_pen_tester_output(text)
    assert v is not None
    assert v.is_refuted


def test_parse_returns_none_on_invalid_verdict_value() -> None:
    text = """```json
{"finding_id": "f", "verdict": "maybe", "rationale": "x", "evidence": []}
```"""
    assert parse_pen_tester_output(text) is None


def test_parse_returns_none_on_missing_block() -> None:
    assert parse_pen_tester_output("just plain text") is None
    assert parse_pen_tester_output("") is None


def test_parse_drops_evidence_rows_with_no_url() -> None:
    text = """```json
{
  "finding_id": "f",
  "verdict": "confirmed",
  "rationale": "x",
  "evidence": [
    {"method": "GET", "status": 200, "url": ""},
    {"method": "GET", "status": 500, "url": "http://localhost/x"}
  ]
}
```"""
    v = parse_pen_tester_output(text)
    assert v is not None
    assert len(v.evidence) == 1
    assert v.evidence[0].url == "http://localhost/x"


def test_verdict_to_note_includes_count_and_rationale_head() -> None:
    v = PenTestVerdict(
        finding_id="f", verdict="confirmed",
        rationale="The defense leaked a SQL error stack trace.",
        evidence=(ProbeEvidence(method="POST", url="x", status=500),),
    )
    note = verdict_to_note(v)
    assert "pen_test: confirmed" in note
    assert "1 probe" in note
    assert "SQL error" in note


def test_verdict_to_note_handles_no_rationale() -> None:
    v = PenTestVerdict(
        finding_id="f", verdict="refuted", rationale="", evidence=(),
    )
    note = verdict_to_note(v)
    assert "pen_test: refuted" in note
    assert "0 probes" in note


# ---------------------------------------------------------------------------
# Role allow-list
# ---------------------------------------------------------------------------


def test_pen_tester_role_includes_pen_test_tools() -> None:
    """The pen_tester role is the ONLY consumer of the probing tools.
    Every probing tool must be in its allow-list."""
    for name in PEN_TEST_TOOL_NAMES:
        assert name in PEN_TESTER_TOOL_NAMES


def test_pen_tester_role_includes_read_only_and_deterministic() -> None:
    """Pen_tester surveys before probing — it needs the same read
    surface as detector/investigator."""
    assert READ_ONLY_TOOL_NAMES.issubset(PEN_TESTER_TOOL_NAMES)


def test_other_roles_still_disjoint_from_pen_test_tools() -> None:
    """The Phase 1a/1b safety invariant must hold after Phase 1c:
    NO ROLE OTHER THAN PEN_TESTER may contain a probing tool name.

    A regression that adds http_attack to e.g. DETECTOR_TOOL_NAMES
    would silently grant probing capability to detect runs. Catch
    that here."""
    for role_names in (
        READ_ONLY_TOOL_NAMES,
        PLANNER_TOOL_NAMES,
        DETECTOR_TOOL_NAMES,
        INVESTIGATOR_TOOL_NAMES,
        LEAD_TOOL_NAMES,
        VERIFIER_TOOL_NAMES,
        FIXER_TOOL_NAMES,
    ):
        assert PEN_TEST_TOOL_NAMES.isdisjoint(role_names), (
            f"probing tools leaked into role allow-list: "
            f"{PEN_TEST_TOOL_NAMES & role_names}"
        )


def test_tool_names_for_role_recognizes_pen_tester() -> None:
    """Legacy lookup must route pen_tester to its allow-list."""
    from augmentum.agents.tools import tool_names_for_role
    assert tool_names_for_role("pen_tester") == PEN_TESTER_TOOL_NAMES


# ---------------------------------------------------------------------------
# RoleModelConfig
# ---------------------------------------------------------------------------


def test_role_model_config_pen_tester_falls_back_to_verifier() -> None:
    """When the user doesn't override pen_tester, it should use the
    same model as the verifier — same job (active confirmation),
    different mechanism."""
    cfg = RoleModelConfig(
        planner="A", detector="A", verifier="B", fixer="A",
    )
    assert cfg.for_role(Role.PEN_TESTER) == "B"


def test_role_model_config_pen_tester_explicit_override() -> None:
    cfg = RoleModelConfig(
        planner="A", detector="A", verifier="B", fixer="A",
        pen_tester="C",
    )
    assert cfg.for_role(Role.PEN_TESTER) == "C"


# ---------------------------------------------------------------------------
# Orchestrator integration — _run_pen_test_leg
# ---------------------------------------------------------------------------


@dataclass
class _StubSubagentResult:
    output: str
    role: str = "pen_tester"
    instance_id: str = "pen_tester_X"
    iterations: int = 1
    tokens_in: int = 100
    tokens_out: int = 50
    wallclock_ms: int = 10
    stop_reason: str = "complete"
    stuck_pattern: str = ""


def _make_finding(
    *, id: str, severity: str = "high",
    status: str = FindingStatus.CONFIRMED.value,
    signature: str = "auth_bypass",
) -> Finding:
    return Finding(
        id=id,
        file="auth/routes.py",
        function="login",
        severity=severity,
        claim=f"finding {id}",
        claim_signature=signature,
        evidence_paths=("auth/routes.py:12",),
        suggested_repro="POST /api/login",
        status=status,
    )


@pytest.mark.asyncio
async def test_pen_test_leg_attaches_confirmed_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A confirmed-verdict pen_test run must stamp the finding's
    notes with the verdict + rationale."""
    from augmentum.bug_finder import orchestrator, pen_tester

    finding = _make_finding(id="f_confirmed", severity="high")

    async def _fake_pen_tester(*, finding, **kwargs):
        return pen_tester.PenTesterRunResult(
            verdict=PenTestVerdict(
                finding_id=finding.id, verdict="confirmed",
                rationale="SQL error leaked on a single-quote payload",
                evidence=(ProbeEvidence(
                    method="POST", url="http://localhost/api/login",
                    status=500, expected_status=400, note="leak",
                ),),
            ),
            subagent_result=_StubSubagentResult(output="ok"),
            runtime_seconds=0.1,
        )

    monkeypatch.setattr(pen_tester, "run_pen_tester", _fake_pen_tester)

    async def _fake_resolve(_model):
        return (object(), "fake-model")

    notes: list[str] = []
    out = await orchestrator._run_pen_test_leg(
        [finding],
        config=_make_config_with_pen_test_on(),
        workspace_root_for_probes=tmp_path,
        resolve_backend=_fake_resolve,
        ledger=[],
        notes=notes,
    )
    target = next(f for f in out if f.id == "f_confirmed")
    assert any("pen_test: confirmed" in n for n in target.notes), target.notes
    # Severity stays high — confirmation doesn't downgrade
    assert target.severity == "high"
    assert any("pen_test leg:" in n for n in notes), notes


@pytest.mark.asyncio
async def test_pen_test_leg_downgrades_refuted_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the verdict is ``refuted``, severity caps at low and a
    breadcrumb note explains the downgrade."""
    from augmentum.bug_finder import orchestrator, pen_tester

    finding = _make_finding(id="f_refuted", severity="high")

    async def _fake_pen_tester(*, finding, **kwargs):
        return pen_tester.PenTesterRunResult(
            verdict=PenTestVerdict(
                finding_id=finding.id, verdict="refuted",
                rationale="Three payloads all 400, no error leak",
                evidence=(),
            ),
            subagent_result=_StubSubagentResult(output="ok"),
            runtime_seconds=0.1,
        )

    monkeypatch.setattr(pen_tester, "run_pen_tester", _fake_pen_tester)

    async def _fake_resolve(_model):
        return (object(), "fake-model")

    out = await orchestrator._run_pen_test_leg(
        [finding],
        config=_make_config_with_pen_test_on(),
        workspace_root_for_probes=tmp_path,
        resolve_backend=_fake_resolve,
        ledger=[],
        notes=[],
    )
    target = next(f for f in out if f.id == "f_refuted")
    assert target.severity == "low"
    assert any("severity downgraded" in n for n in target.notes)
    assert any("pen_test: refuted" in n for n in target.notes)


@pytest.mark.asyncio
async def test_pen_test_leg_unparseable_output_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the LLM emits something the parser can't decode, mark the
    finding as inconclusive but don't blow up the run."""
    from augmentum.bug_finder import orchestrator, pen_tester

    finding = _make_finding(id="f_garbled", severity="medium")

    async def _fake_pen_tester(*, finding, **kwargs):
        return pen_tester.PenTesterRunResult(
            verdict=None,
            subagent_result=_StubSubagentResult(
                output="(no fenced JSON in this turn)",
            ),
            runtime_seconds=0.1,
        )

    monkeypatch.setattr(pen_tester, "run_pen_tester", _fake_pen_tester)

    async def _fake_resolve(_model):
        return (object(), "fake-model")

    out = await orchestrator._run_pen_test_leg(
        [finding],
        config=_make_config_with_pen_test_on(),
        workspace_root_for_probes=tmp_path,
        resolve_backend=_fake_resolve,
        ledger=[],
        notes=[],
    )
    target = next(f for f in out if f.id == "f_garbled")
    assert any("unparseable" in n for n in target.notes)
    # Severity unchanged — inconclusive is neither confirmation nor refutation
    assert target.severity == "medium"


@pytest.mark.asyncio
async def test_pen_test_leg_subagent_exception_is_contained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subagent crash on ONE finding must not abort the leg for the
    rest — the leg should attach a note and keep going."""
    from augmentum.bug_finder import orchestrator, pen_tester

    finding = _make_finding(id="f_boom", severity="high")

    async def _fake_pen_tester(*, finding, **kwargs):
        raise RuntimeError("simulated subagent crash")

    monkeypatch.setattr(pen_tester, "run_pen_tester", _fake_pen_tester)

    async def _fake_resolve(_model):
        return (object(), "fake-model")

    out = await orchestrator._run_pen_test_leg(
        [finding],
        config=_make_config_with_pen_test_on(),
        workspace_root_for_probes=tmp_path,
        resolve_backend=_fake_resolve,
        ledger=[],
        notes=[],
    )
    target = next(f for f in out if f.id == "f_boom")
    assert any("pen_test: error" in n for n in target.notes), target.notes


@pytest.mark.asyncio
async def test_pen_test_leg_skips_unconfirmed_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only CONFIRMED findings go through the pen-test leg — speculative
    ones haven't proven they exist yet."""
    from augmentum.bug_finder import orchestrator, pen_tester

    confirmed = _make_finding(id="f_real")
    speculative = _make_finding(
        id="f_specu", status=FindingStatus.SPECULATIVE.value,
    )

    invocation_log: list[str] = []

    async def _fake_pen_tester(*, finding, **kwargs):
        invocation_log.append(finding.id)
        return pen_tester.PenTesterRunResult(
            verdict=PenTestVerdict(
                finding_id=finding.id, verdict="confirmed",
                rationale="ok", evidence=(),
            ),
            subagent_result=_StubSubagentResult(output="ok"),
            runtime_seconds=0.1,
        )

    monkeypatch.setattr(pen_tester, "run_pen_tester", _fake_pen_tester)

    async def _fake_resolve(_model):
        return (object(), "fake-model")

    await orchestrator._run_pen_test_leg(
        [confirmed, speculative],
        config=_make_config_with_pen_test_on(),
        workspace_root_for_probes=tmp_path,
        resolve_backend=_fake_resolve,
        ledger=[],
        notes=[],
    )
    assert invocation_log == ["f_real"]


@pytest.mark.asyncio
async def test_pen_test_leg_no_deterministic_root_skips_with_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a workspace root the leg can't probe — note + bail."""
    from augmentum.bug_finder import orchestrator

    notes: list[str] = []
    out = await orchestrator._run_pen_test_leg(
        [_make_finding(id="f")],
        config=_make_config_with_pen_test_on(),
        workspace_root_for_probes=None,
        resolve_backend=lambda m: (None, ""),  # never reached
        ledger=[],
        notes=notes,
    )
    assert len(out) == 1
    assert any("pen_test leg skipped" in n for n in notes)


@pytest.mark.asyncio
async def test_pen_test_leg_event_emit_signature_matches_outer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``_run_pen_test_leg`` must call ``event_emit`` with a
    signature that the outer orchestrator's ``_emit`` accepts.

    Original bug: the leg's inner ``_e`` passed ``False`` as the third
    positional arg for ``terminal``, but the outer ``_emit`` has
    ``terminal`` as keyword-only. That raised
    ``TypeError: _emit() takes 2 positional arguments but 3 were given``
    on the first stage-emit attempt inside the leg, killing the run.

    Pin the contract by passing an emit function with the same
    kw-only-terminal shape as the real one and asserting the leg
    completes without TypeError.
    """
    from augmentum.bug_finder import orchestrator, pen_tester

    finding = _make_finding(id="f_emit", severity="high")

    async def _fake_pen_tester(*, finding, **kwargs):
        return pen_tester.PenTesterRunResult(
            verdict=PenTestVerdict(
                finding_id=finding.id, verdict="confirmed",
                rationale="ok", evidence=(),
            ),
            subagent_result=_StubSubagentResult(output="ok"),
            runtime_seconds=0.1,
        )

    monkeypatch.setattr(pen_tester, "run_pen_tester", _fake_pen_tester)

    async def _fake_resolve(_model):
        return (object(), "fake-model")

    captured: list[tuple[str, dict]] = []

    def _strict_emit(kind: str, payload: dict, *, terminal: bool = False) -> None:
        # Matches the production ``_emit`` defined in run_bug_finder —
        # ``terminal`` is keyword-only. Positional False would raise here.
        captured.append((kind, payload))

    await orchestrator._run_pen_test_leg(
        [finding],
        config=_make_config_with_pen_test_on(),
        workspace_root_for_probes=tmp_path,
        resolve_backend=_fake_resolve,
        ledger=[],
        event_emit=_strict_emit,
        notes=[],
    )
    # The leg fires several stage emits — assert at least one landed
    # and none raised. ``stage`` is the most reliable kind.
    assert any(kind == "stage" for kind, _ in captured), captured


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_config_with_pen_test_on():
    """Minimal BugFinderRunConfig stand-in for leg tests. We bypass
    the full config builder because the leg only touches a few
    fields."""

    @dataclass
    class _ConfigStub:
        enable_pen_test_leg: bool = True
        pen_test_boot_command: str = ""
        pen_test_boot_port: int = 0
        pen_test_healthcheck_path: str = "/"
        role_models: Any = None

    @dataclass
    class _RoleModelsStub:
        def for_role(self, role):
            return "fake-model"

    cfg = _ConfigStub()
    cfg.role_models = _RoleModelsStub()
    return cfg


# ---------------------------------------------------------------------------
# Config-level integration
# ---------------------------------------------------------------------------


def test_bug_finder_run_config_default_pen_test_disabled() -> None:
    """The leg must be opt-in. Defaulting it on would make every
    bug_finder run boot the workspace's app, which is heavier than
    callers usually want."""
    # Build with minimal required args
    import inspect

    from augmentum.bug_finder.orchestrator import BugFinderRunConfig
    sig = inspect.signature(BugFinderRunConfig)
    assert sig.parameters["enable_pen_test_leg"].default is False
