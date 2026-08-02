"""Independent cross-model verification of a completed background coder run.

The completion-relay honesty gate (design conversation 2026-07-27): before the
user is told a delegated run is "done", a DIFFERENT model reviews the run's
ACTUAL diff against the user's original ask and returns an honest, TIERED
verdict. This is the automated first verifier that must exist for the human to
be the *second* verifier (the exception-gate) rather than the depleted
first-pass checker on every run.

Deliberate constraints, each load-bearing:

* **Cross-model only.** The verifier is the user's pinned ``heavyweight_model``.
  If none is pinned, verification does NOT run — the verdict stays ``unchecked``
  (the honest hedge), never a self-graded "passed". If the heavyweight resolves
  to the SAME model that drove the run, that's not an independent check either,
  so the verdict is also ``unchecked`` with that reason stated. A model grading
  its own work is exactly the correlated-failure echo chamber we refuse to ship.
* **Grounded in the diff, not the claim.** It judges the real ``unified_diff``
  from the review bundle, not the driver's self-reported answer text — judging
  the claim alone is the lonely-runner false-proof hole.
* **Two oracles, honest tiers.** ``probable`` (ORACLE_JUDGMENT) = a stronger,
  independent model reviewed the actual diff and judges it correct. ``verified``
  (ORACLE_MECHANICAL) is only reached when the run's OWN recorded tests are
  RE-RUN in the workspace and pass AND the independent model also agrees — the
  strongest bar, requiring both. If the model and the tests DISAGREE (tests pass
  but the reviewer flagged a problem, or vice-versa) the verdict is
  ``human_required``, never a laundered ``verified``. Never over-claim a verdict
  we can't stand behind.

Shape mirrors ``agents/verify.py::judge_subagent_result`` (proven plumbing):
one second-LLM round-trip, ``stream=False``, ``temperature=0``, small
``max_tokens``, **never raises**. A failed/flaky verifier degrades to
``unchecked`` — it must never fail a completed mission, and a silent failure
must never masquerade as a "passed".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Mechanical re-run budget. Tests that hang past the wall-clock or go silent
# are treated as inconclusive (they don't demote a judged-good run), not as a
# pass. Cap the number of recorded commands re-run so a pathological run can't
# tie up the single-worker job queue.
_TEST_TIMEOUT_S = 120.0
_TEST_IDLE_S = 20.0
_MAX_TEST_CMDS = 4

# Verdict tiers — vocabulary shared with selfedit/verifier.py (weakest→strongest
# confirmation of INTENT). Kept as local constants to avoid coupling coder to
# the selfedit package.
TIER_FAILED = "failed"                # judge found the diff does not satisfy the ask
TIER_HUMAN_REQUIRED = "human_required"  # judge flagged a genuine intent ambiguity
TIER_PROBABLE = "probable"            # independent model reviewed the diff, judges it correct
TIER_UNCHECKED = "unchecked"          # not verified (no heavyweight / disabled / self / error)
TIER_VERIFIED = "verified"            # RESERVED: mechanical proof (tests ran + passed) — not emitted yet

ORACLE_JUDGMENT = "judgment"          # a model estimated; can be wrong
ORACLE_MECHANICAL = "mechanical"      # RESERVED
ORACLE_NONE = "none"

_MAX_DIFF_CHARS = 8000
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_JUDGE_PROMPT = """\
You are an INDEPENDENT verification judge — a DIFFERENT model than the one that \
wrote this code. A coding agent was given a task by a user and has reported it \
done. Below is the user's original request and the agent's ACTUAL diff. Judge \
ONLY whether the diff genuinely satisfies the user's request — from the evidence \
in the diff, not from the agent's optimism, and not by demanding more than was \
asked.

Respond with a JSON object only — no prose, no markdown fences:
{"ok": true, "needs_human": false, "tests_seen": false, "reason": "<evidence the request is met>"}
{"ok": false, "needs_human": false, "tests_seen": false, "reason": "<what is concretely missing or wrong>"}

Rules:
- Set ok=false if the diff does not accomplish the request, is empty, or contains \
an obvious defect that defeats the stated goal.
- Set needs_human=true (with ok=true) ONLY when the code is reasonable but the \
request contains a genuine PRODUCT/INTENT choice the agent had to guess at, and \
the user should confirm which they wanted — not for ordinary uncertainty.
- Set tests_seen=true if the diff itself adds or updates tests covering the change \
(this is a signal only; you are NOT running them).
- Judge substance, not wording. A minor stylistic gap is ok=true. A load-bearing \
claim with no supporting change in the diff is ok=false."""

# Appended only when an inbound contract is present — asks the judge to itemize
# unmet contract items. The verdict logic downgrades a nominally-ok run that
# still has unmet load-bearing items to human_required (surface the gap).
_CONTRACT_JUDGE_SUFFIX = """

An <inbound_contract> is provided. Add an "unmet" array listing the VERBATIM \
contract items the diff+evidence does not honestly satisfy (empty if all met):
{"ok": <bool>, "needs_human": <bool>, "tests_seen": <bool>, "unmet": ["<item>", ...], "reason": "..."}
If any load-bearing contract item is unmet, set ok=false and list it."""


@dataclass(frozen=True)
class RunVerdict:
    """Honest, tiered verdict for a completed coder run."""

    tier: str
    oracle: str = ORACLE_NONE
    reason: str = ""
    verifier_model: str = ""
    self_verified: bool = False
    # Inbound-contract gate (P3): the load-bearing items from the run's own
    # committed contract (mission promises + pending_objective_contract) that
    # the independent judge found NOT honestly satisfied by the diff+evidence.
    # Empty when there was no inbound contract (direct-user turns) or all items
    # were met — so a run with no mission behaves exactly as before.
    contract_unmet: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True only for a positive independent verdict."""
        return self.tier in (TIER_PROBABLE, TIER_VERIFIED)

    def to_envelope(self) -> dict[str, Any]:
        """The shape the brief consumes (envelope.verification)."""
        return {
            "tier": self.tier,
            "oracle": self.oracle,
            "reason": self.reason,
            "verifier_model": self.verifier_model,
            "self_verified": self.self_verified,
            "contract_unmet": list(self.contract_unmet),
        }


def _unchecked(reason: str, *, verifier_model: str = "", self_verified: bool = False) -> RunVerdict:
    return RunVerdict(
        tier=TIER_UNCHECKED, oracle=ORACLE_NONE, reason=reason,
        verifier_model=verifier_model, self_verified=self_verified,
    )


def _normalize_model(name: str) -> str:
    """Coarse identity for the driver-vs-verifier independence check — strip
    a provider/fabric prefix and casing so 'ollama/qwen3.6' and 'qwen3.6'
    compare equal (a fabric-routed same model is still self-verification)."""
    n = (name or "").strip().lower()
    return n.rsplit("/", 1)[-1] if "/" in n else n


def _conn(app_state: Any) -> Any:
    return getattr(
        getattr(getattr(app_state, "state_manager", None), "backend", None),
        "conn", None,
    )


def _collect_diff(app_state: Any, review_turn_id: str) -> str:
    """Pull the real unified diff from the (in-memory) review bundle. The
    bundle is still pending at verification time — read with get(), never
    resolve() (which pops it)."""
    reg = getattr(app_state, "review_registry", None)
    if reg is None or not review_turn_id:
        return ""
    try:
        bundle = reg.get(review_turn_id)
    except Exception:
        return ""
    if bundle is None:
        return ""
    parts: list[str] = []
    total = 0
    for f in getattr(bundle, "files", None) or []:
        header = f"--- {getattr(f, 'path', '?')} ({getattr(f, 'status', '')}) ---\n"
        chunk = header + (getattr(f, "unified_diff", "") or "")
        if total + len(chunk) > _MAX_DIFF_CHARS:
            parts.append(header + "[diff truncated for length]\n")
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


async def _load_contract(
    app_state: Any, *, workspace_id: str, user_id: str,
) -> tuple[str, ...]:
    """Load the run's INBOUND contract — the promises it committed to plus the
    cross-turn acceptance requirements — as flat criteria strings for the judge.

    Defensive: any miss (no state manager, no mission, non-SQLite backend)
    returns ``()`` so a run with no inbound contract verifies exactly as it did
    before the gate existed. This is the back-compat guarantee."""
    sm = getattr(app_state, "state_manager", None)
    loader = getattr(sm, "load_coder_state", None)
    if loader is None or not workspace_id:
        return ()
    try:
        state = await loader(workspace_id, user_id=user_id)
    except Exception:
        log.warning("coder_run_verify_contract_load_failed", exc_info=True)
        return ()
    if state is None:
        return ()
    criteria: list[str] = []
    # Mission = the structured Promise tree (flatten descriptions).
    for promise in getattr(state, "mission", None) or []:
        desc = str(getattr(promise, "description", "") or "").strip()
        if desc:
            criteria.append(desc[:300])
    # pending_objective_contract = acceptance-oriented "must be proven" items.
    contract = getattr(state, "pending_objective_contract", None)
    if isinstance(contract, dict):
        for key in ("requirements", "must_prove", "acceptance", "unresolved"):
            val = contract.get(key)
            if isinstance(val, list):
                criteria.extend(str(v)[:300] for v in val if v and str(v).strip())
            elif isinstance(val, str) and val.strip():
                criteria.append(val.strip()[:300])
    # De-dup, preserve order.
    seen: set[str] = set()
    return tuple(c for c in criteria if not (c in seen or seen.add(c)))


async def _load_evidence(
    app_state: Any, *, review_turn_id: str, user_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the OUTBOUND evidence for the turn: the citation rows (claim→proof)
    and the oracle summary (tests-not-gamed signal). Both keyed by the ledger
    ``ctr_`` id (== ``review_turn_id``). Defensive — returns ([], {})."""
    conn = _conn(app_state)
    if conn is None or not review_turn_id:
        return [], {}
    cites: list[dict[str, Any]] = []
    oracle: dict[str, Any] = {}
    try:
        from augmentum.coder.citations import load_citations
        cites = await load_citations(conn, turn_run_id=review_turn_id, user_id=user_id)
    except Exception:
        log.warning("coder_run_verify_citation_load_failed", exc_info=True)
    try:
        from augmentum.coder.ledger import CoderTurnLedgerStore
        run = await CoderTurnLedgerStore(conn).get_run(review_turn_id, user_id=user_id)
        metrics = (run or {}).get("metrics_json") or {}
        if isinstance(metrics, dict) and isinstance(metrics.get("oracle"), dict):
            oracle = metrics["oracle"]
    except Exception:
        log.warning("coder_run_verify_oracle_load_failed", exc_info=True)
    return cites, oracle


async def verify_coder_run(
    app_state: Any,
    *,
    user_id: str,
    review_turn_id: str,
    prompt: str,
    answer: str,
    driver_model: str,
    run_id: str = "",
    workspace_id: str = "",
) -> RunVerdict:
    """Independently verify a completed run. Never raises — returns an
    ``unchecked`` verdict on any gate miss or verifier failure.

    Three inputs, one judge: an independent cross-model review of the real
    diff, the run's INBOUND contract (mission + acceptance requirements), and
    the OUTBOUND evidence (citations + oracle tests-not-gamed signal). When the
    judgment is real, a MECHANICAL re-run of the run's own recorded tests
    confirms it — ``verified`` needs both to agree; a disagreement is
    ``human_required``. A run with no inbound contract behaves exactly as the
    pre-gate verifier did.
    """
    contract = await _load_contract(
        app_state, workspace_id=workspace_id, user_id=user_id,
    )
    citations, oracle = await _load_evidence(
        app_state, review_turn_id=review_turn_id, user_id=user_id,
    )
    base = await _judge_run(
        app_state, user_id=user_id, review_turn_id=review_turn_id,
        prompt=prompt, answer=answer, driver_model=driver_model,
        contract=contract, citations=citations, oracle=oracle,
    )
    # No independent judgment (gate missed / self / error) → don't run tests
    # either: verification is one gated feature (respects the heavyweight gate).
    if base.tier == TIER_UNCHECKED:
        return base
    # Mechanical re-run keys on the LEDGER id (ctr_ == review_turn_id), NOT the
    # broker run_id — get_run() indexes coder_turn_runs.id. Passing the broker
    # id made the mechanical tier silently unreachable (fixed with the gate).
    mech = await _run_tests(
        app_state, user_id=user_id, workspace_id=workspace_id,
        review_turn_id=review_turn_id,
    )
    return _combine(base, mech)


def _render_contract_block(contract: tuple[str, ...]) -> str:
    """Render the inbound contract + the extra judge instruction, or '' when
    there was no contract (so the base diff-vs-request judgment is unchanged)."""
    if not contract:
        return ""
    items = "\n".join(f"{i}. {c}" for i, c in enumerate(contract, start=1))
    return (
        "\n<inbound_contract>\nThe run committed to these load-bearing items "
        "up front. Judge whether the diff+evidence HONESTLY satisfies each — a "
        "bare claim with no supporting change or oracle is UNMET:\n"
        f"{items}\n</inbound_contract>\n"
    )


def _render_evidence_block(
    citations: list[dict[str, Any]], oracle: dict[str, Any],
) -> str:
    """Render the outbound evidence: what the run actually verified (oracle
    tests-not-gamed signal) + the claim→proof citations."""
    parts: list[str] = []
    if oracle:
        no_oracle = bool(oracle.get("no_oracle_done"))
        last = str(oracle.get("last_outcome") or "")
        kinds = ", ".join(oracle.get("kinds") or []) or "none"
        warn = (
            " ⚠ wrote files but ran NO verification oracle after the last edit "
            "— treat 'it works' claims skeptically."
            if no_oracle else ""
        )
        parts.append(
            f"<verification_telemetry>\noracle kinds used: {kinds}; "
            f"last outcome: {last or 'n/a'}.{warn}\n</verification_telemetry>"
        )
    if citations:
        rows = []
        for c in citations[:40]:
            loc = c.get("file") or c.get("evidence_ref") or "?"
            span = (
                f":{c['line_start']}-{c['line_end']}"
                if c.get("line_start") and c.get("line_end") else ""
            )
            out = f" [{c.get('outcome')}]" if c.get("outcome") else ""
            rows.append(f"- {c.get('evidence_kind')}: {loc}{span}{out}")
        parts.append("<citations>\n" + "\n".join(rows) + "\n</citations>")
    return ("\n" + "\n\n".join(parts) + "\n") if parts else ""


async def _judge_run(
    app_state: Any,
    *,
    user_id: str,
    review_turn_id: str,
    prompt: str,
    answer: str,
    driver_model: str,
    contract: tuple[str, ...] = (),
    citations: list[dict[str, Any]] | None = None,
    oracle: dict[str, Any] | None = None,
) -> RunVerdict:
    """The cross-model diff-review leg. Returns a judgment-oracle verdict.

    Beyond the diff-vs-request check, it folds in the INBOUND contract (mission
    + acceptance items) and OUTBOUND evidence (citations + oracle signal) when
    available — the P3 gate. With no contract the judgment is unchanged."""
    from augmentum.config import settings as _settings

    if not getattr(_settings, "coder_verify_enabled", True):
        return _unchecked("Independent verification is turned off.")

    heavy = (getattr(_settings, "heavyweight_model", "") or "").strip()
    if not heavy:
        return _unchecked(
            "No heavyweight model is set — pin one in Settings → Models to have "
            "a second model independently check completed runs.",
        )

    registry = getattr(app_state, "provider_registry", None)
    if registry is None:
        return _unchecked("Model registry unavailable for verification.")
    try:
        backend, vmodel = await registry.resolve_backend_with_fabric(heavy, user_id=user_id)
    except Exception:
        log.warning("coder_run_verify_resolve_failed", model=heavy, exc_info=True)
        backend, vmodel = None, ""
    if backend is None:
        return _unchecked(f"Heavyweight verifier model unavailable: {heavy}.", verifier_model=heavy)

    resolved = vmodel or heavy
    self_verified = _normalize_model(resolved) == _normalize_model(driver_model)
    if self_verified:
        # A model grading its own work is not an independent check.
        return _unchecked(
            "The heavyweight verifier is the same model that did the work — "
            "not an independent check. Pin a different model to verify.",
            verifier_model=resolved, self_verified=True,
        )

    diff = _collect_diff(app_state, review_turn_id)
    if not diff:
        return _unchecked(
            "No diff was available to review (nothing changed, or the review "
            "bundle expired).",
            verifier_model=resolved,
        )

    context = (
        f"<user_request>\n{(prompt or '').strip()[:2000]}\n</user_request>\n\n"
        f"<agent_report>\n{(answer or '').strip()[:1500]}\n</agent_report>\n\n"
        f"<diff>\n{diff}\n</diff>\n"
        f"{_render_contract_block(contract)}"
        f"{_render_evidence_block(citations or [], oracle or {})}"
        f"\n{_JUDGE_PROMPT}"
        f"{_CONTRACT_JUDGE_SUFFIX if contract else ''}"
    )

    from augmentum.models.base import InternalChatRequest, Message, response_text

    req = InternalChatRequest(
        model=resolved,
        messages=[Message(role="user", content=context)],
        tools=None, tool_choice=None, stream=False,
        temperature=0.0, max_tokens=400, chat_template_kwargs=None,
    )
    try:
        resp = await backend.chat(req)
    except Exception as exc:  # noqa: BLE001
        log.warning("coder_run_verify_backend_error", error=str(exc)[:200], model=resolved)
        return _unchecked("The verifier model failed to respond.", verifier_model=resolved)

    raw = response_text(resp, thinking_fallback=False).strip()
    parsed = _parse(raw)
    if parsed is None:
        log.warning("coder_run_verify_parse_failed", raw_preview=raw[:160], model=resolved)
        return _unchecked("The verifier produced no usable signal.", verifier_model=resolved)

    ok = parsed.get("ok")
    if not isinstance(ok, bool):
        return _unchecked("The verifier produced no usable signal.", verifier_model=resolved)
    reason = str(parsed.get("reason") or "")[:400]
    unmet_raw = parsed.get("unmet")
    contract_unmet: tuple[str, ...] = ()
    if isinstance(unmet_raw, list):
        contract_unmet = tuple(str(u)[:300] for u in unmet_raw if u and str(u).strip())

    if ok is False:
        return RunVerdict(
            TIER_FAILED, ORACLE_JUDGMENT, reason,
            verifier_model=resolved, contract_unmet=contract_unmet,
        )
    if parsed.get("needs_human") is True:
        return RunVerdict(
            TIER_HUMAN_REQUIRED, ORACLE_JUDGMENT, reason,
            verifier_model=resolved, contract_unmet=contract_unmet,
        )
    # ok=True but the judge still flagged unmet load-bearing contract items —
    # a nominally-passing run with a real gap. Surface it as the user's call
    # rather than laundering it to "probable/done".
    if contract_unmet:
        return RunVerdict(
            TIER_HUMAN_REQUIRED, ORACLE_JUDGMENT,
            (reason + " Unmet contract items remain — your call.").strip(),
            verifier_model=resolved, contract_unmet=contract_unmet,
        )
    # ok=True: a different model reviewed the actual diff and judges it correct.
    # Ceiling is PROBABLE — mechanical proof (running the tests) is a future tier.
    return RunVerdict(TIER_PROBABLE, ORACLE_JUDGMENT, reason, verifier_model=resolved)


@dataclass(frozen=True)
class _MechResult:
    # "pass" | "fail" | "none" (no tests recorded) | "inconclusive" (timeout/exec)
    status: str
    detail: str = ""


async def _run_tests(
    app_state: Any, *, user_id: str, workspace_id: str, review_turn_id: str,
) -> _MechResult:
    """Re-run the run's OWN recorded test commands in the workspace and report
    a mechanical pass/fail. ``none`` when nothing testable was recorded — the
    honest gate that keeps ``verified`` off runs that never ran tests.

    Keyed by ``review_turn_id`` — the ledger ``ctr_`` id that
    ``get_run`` indexes (``coder_turn_runs.id``). The broker ``run_id`` is a
    different namespace and would always miss."""
    if not review_turn_id or not workspace_id:
        return _MechResult("none")
    conn = _conn(app_state)
    cm = getattr(app_state, "container_manager", None)
    if conn is None or cm is None:
        return _MechResult("none")
    try:
        from augmentum.coder.ledger import CoderTurnLedgerStore
        run = await CoderTurnLedgerStore(conn).get_run(review_turn_id, user_id=user_id)
    except Exception:
        log.warning("coder_run_verify_ledger_read_failed", run_id=review_turn_id, exc_info=True)
        return _MechResult("none")
    cmds = [
        c for c in ((run or {}).get("tests_run") or [])
        if isinstance(c, str) and c.strip()
    ][:_MAX_TEST_CMDS]
    if not cmds:
        return _MechResult("none")

    from augmentum.coder.containers import ExecAborted
    for cmd in cmds:
        try:
            # -lc: login shell so the workspace's own venv/npm PATH is visible
            # (mirrors how the driver's test_run tool invoked it). strict=True
            # turns a non-zero exit into ExecAborted(kind="exit_code").
            await cm.run_command(
                workspace_id, ["bash", "-lc", cmd],
                timeout=_TEST_TIMEOUT_S, idle_timeout=_TEST_IDLE_S, strict=True,
            )
        except ExecAborted as exc:
            if getattr(exc, "kind", "") == "exit_code":
                return _MechResult("fail", cmd)
            # Timeout / idle-kill — can't claim pass OR fail honestly.
            return _MechResult("inconclusive", f"{getattr(exc, 'kind', 'timeout')} on `{cmd[:80]}`")
        except Exception as exc:  # noqa: BLE001 — container gone / transient
            log.warning("coder_run_verify_test_exec_error", cmd=cmd[:80], error=str(exc)[:160])
            return _MechResult("inconclusive", f"exec error on `{cmd[:80]}`")
    return _MechResult("pass", "; ".join(cmds))


def _combine(base: RunVerdict, mech: _MechResult) -> RunVerdict:
    """Fuse the judgment verdict with the mechanical test result — honestly.
    ``verified`` requires BOTH to agree; a disagreement is ``human_required``."""
    if mech.status == "none":
        return base  # no tests to run — judgment stands
    if mech.status == "inconclusive":
        note = f" (couldn't re-run tests to confirm: {mech.detail})"
        return replace(base, reason=(base.reason + note).strip())
    if mech.status == "fail":
        # Objective failure overrides any positive judgment.
        return RunVerdict(
            TIER_FAILED, ORACLE_MECHANICAL,
            f"Tests failed when re-run: {mech.detail}.",
            verifier_model=base.verifier_model, contract_unmet=base.contract_unmet,
        )
    # mech pass
    if base.tier == TIER_PROBABLE:
        who = base.verifier_model or "an independent model"
        return RunVerdict(
            TIER_VERIFIED, ORACLE_MECHANICAL,
            f"Tests passed on re-run and {who} agrees. {base.reason}".strip(),
            verifier_model=base.verifier_model, contract_unmet=base.contract_unmet,
        )
    if base.tier == TIER_FAILED:
        # Tests pass but the reviewer flagged a problem — a real disagreement.
        return RunVerdict(
            TIER_HUMAN_REQUIRED, ORACLE_MECHANICAL,
            f"Tests pass, but the independent reviewer flagged a concern: {base.reason} "
            "Your call.",
            verifier_model=base.verifier_model, contract_unmet=base.contract_unmet,
        )
    # base already human_required — keep it, note the tests passed.
    return replace(base, reason=f"Tests pass. {base.reason}".strip())


# ── Durable persistence (so a brief opened cold from a stale notification can
#    still show the verdict — the live path carries it in the envelope) ──────

async def save_verdict(
    conn: Any, *, run_id: str, user_id: str, workspace_id: str, verdict: RunVerdict,
) -> None:
    """Best-effort persist. A persistence failure must never fail a mission."""
    if conn is None or not run_id or not user_id:
        return
    try:
        await conn.execute(
            "INSERT OR REPLACE INTO coder_run_verifications "
            "(run_id, user_id, workspace_id, tier, oracle, reason, "
            " verifier_model, self_verified) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, user_id, workspace_id or "", verdict.tier, verdict.oracle,
                verdict.reason, verdict.verifier_model, 1 if verdict.self_verified else 0,
            ),
        )
        await conn.commit()
    except Exception:
        log.warning("coder_run_verify_persist_failed", run_id=run_id, exc_info=True)


async def load_verdict(conn: Any, *, run_id: str, user_id: str) -> dict[str, Any] | None:
    """Load a persisted verdict envelope for (run_id, user_id), or None."""
    if conn is None or not run_id or not user_id:
        return None
    try:
        cur = await conn.execute(
            "SELECT tier, oracle, reason, verifier_model, self_verified "
            "FROM coder_run_verifications WHERE run_id = ? AND user_id = ?",
            (run_id, user_id),
        )
        row = await cur.fetchone()
    except Exception:
        log.warning("coder_run_verify_load_failed", run_id=run_id, exc_info=True)
        return None
    if row is None:
        return None
    return {
        "tier": row[0], "oracle": row[1], "reason": row[2],
        "verifier_model": row[3], "self_verified": bool(row[4]),
    }


def _parse(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    cleaned = _FENCE_RE.sub("", raw).strip()
    a, b = cleaned.find("{"), cleaned.rfind("}")
    if a >= 0 and b > a:
        cleaned = cleaned[a : b + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


__all__ = [
    "ORACLE_JUDGMENT", "ORACLE_MECHANICAL", "ORACLE_NONE",
    "TIER_FAILED", "TIER_HUMAN_REQUIRED", "TIER_PROBABLE", "TIER_UNCHECKED", "TIER_VERIFIED",
    "RunVerdict", "verify_coder_run", "save_verdict", "load_verdict",
]
