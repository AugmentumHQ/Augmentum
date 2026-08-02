"""Test-first capability authoring — the SOTA self-evolution seam.

Where ``render.py`` template-fills bounded data (personalization), THIS drives
the LIVE edit engine to write ARBITRARY new logic, fenced not by a template but
by a test the system authors itself:

    request → synthesize_acceptance_test (LLM writes a pytest oracle)
            → build_authoring_objective (the contract handed to the engine)
            → run_self_edit(objective, driver, extra_verifiers=[acceptance oracle])
              ↳ the engine writes the real implementation in a candidate worktree,
                escalating model tiers until the test PASSES, in the sandbox
            → VERIFIED (mechanical confirm oracle) → propose → approve → archive

The capability space is therefore unbounded — limited only by what the system
can express as a *verifiable test*, which is the honest definition of "done."

Division of labor (no engine overlap): this module authors the **acceptance test
+ objective** and plugs the test in as an oracle via the engine's existing
``extra_verifiers`` seam (``pytest_confirm_verifier``). The other agent's engine
(candidate → driver → verify → archive) writes and verifies the code. The test
is injected at verify-time by US, not authored by the agent, so it can't be gamed.
"""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

from augmentum.selfedit.verifier import (
    ORACLE_MECHANICAL,
    Verifier,
    VerifierResult,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

ModelInvoke = Callable[[str], Awaitable[str]]

_TEST_PROMPT = """\
Write a pytest ACCEPTANCE TEST that defines "done" for a new capability in the
Augmentum codebase. The test is the contract: an implementer (another model)
will write code until THIS test passes.

The capability the user wants:
{request}

Rules:
- Return ONLY Python code (no prose, no code fence).
- Define one or more `def test_...` functions asserting the capability's
  OBSERVABLE behavior (what it does), not its implementation details.
- It must be deterministic and self-contained (no network, no sleeps, no
  reliance on external services — stub/inject those).
- It should FAIL today (the capability doesn't exist yet) and PASS once built.
- For a new primitive verb, assert it registers and dispatches:
    from augmentum.intent import REGISTRY
    import augmentum.intent.builtin.<module>  # noqa: F401
    def test_x():
        a = REGISTRY.get("surface.action")
        assert a is not None
        # ...assert the handler's ActionResult shape...
- Prefer asserting through the smallest real public surface.
"""

_REPAIR_SUFFIX = """\

Your previous answer was not usable:
{problem}

Return corrected Python code only (def test_... functions)."""


def _slug(request: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (request or "").lower()).strip("_")
    return (s[:40] or "capability").rstrip("_")


def _extract_code(text: str) -> str:
    """Strip a ```python fence if present; otherwise return as-is."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _validate_test(src: str) -> str:
    """Return '' if the source is a usable acceptance test, else the problem."""
    if not src.strip():
        return "empty"
    if "def test_" not in src:
        return "no `def test_` function found"
    try:
        compile(src, "<acceptance-test>", "exec")
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    return ""


async def synthesize_acceptance_test(
    request: str, *, model_invoke: ModelInvoke, slug: str = "",
) -> tuple[str | None, str]:
    """Author a pytest acceptance test for ``request``. Returns
    ``(test_rel_path, test_source)`` on success, or ``(None, error)``.

    This is the hard, valuable, net-new step: turning a desired capability into a
    machine-checkable definition of done. Pure + injected model → unit-testable."""
    rel = f"tests/test_authored_{slug or _slug(request)}.py"
    try:
        raw = await model_invoke(_TEST_PROMPT.format(request=request))
    except Exception as exc:  # noqa: BLE001 — model hiccup = no test, not a crash
        log.warning("acceptance_test_model_failed", error=repr(exc))
        return None, f"model call failed: {exc!r}"

    src = _extract_code(raw)
    problem = _validate_test(src)
    if not problem:
        log.info("acceptance_test_synthesized", path=rel, bytes=len(src))
        return rel, src

    # one repair retry
    try:
        raw2 = await model_invoke(
            _TEST_PROMPT.format(request=request) + _REPAIR_SUFFIX.format(problem=problem)
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"repair call failed: {exc!r}"
    src2 = _extract_code(raw2)
    problem2 = _validate_test(src2)
    if problem2:
        log.info("acceptance_test_rejected", problem=problem2)
        return None, problem2
    return rel, src2


def _write_test(candidate_dir: str, rel_path: str, source: str) -> None:
    dest = os.path.join(candidate_dir, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(source)


# A pytest-confirm factory: (test_paths, cwd) -> Verifier. Defaults to the
# engine's own pytest_confirm_verifier; injectable so this is unit-testable
# without spawning a real pytest subprocess.
PytestConfirmFactory = Callable[..., Verifier]


def acceptance_test_verifier(
    test_rel_path: str, test_source: str, *,
    pytest_confirm_factory: PytestConfirmFactory | None = None,
) -> Verifier:
    """A mechanical confirm oracle that, at verify-time, WRITES our canonical
    acceptance test into the candidate worktree (so the agent can't have weakened
    it) and runs it. Passing → confirms the intent → VERIFIED tier. Plugs into
    ``run_self_edit(extra_verifiers=[...])``."""
    async def _run(ctx: dict) -> VerifierResult:
        candidate_dir = ctx.get("candidate_dir") or "."
        try:
            _write_test(candidate_dir, test_rel_path, test_source)
        except Exception as exc:  # noqa: BLE001 — can't write the oracle → can't confirm
            return VerifierResult(
                "acceptance_test", ORACLE_MECHANICAL, "fail", confirms_intent=True,
                required=True, detail=f"could not inject acceptance test: {exc!r}",
            )
        factory = pytest_confirm_factory
        if factory is None:
            from augmentum.selfedit.adapters import pytest_confirm_verifier as factory
        inner = factory([test_rel_path], cwd=candidate_dir)
        result = await inner.run(ctx)
        # relabel so the verdict trail names the authored oracle
        result.name = "acceptance_test"
        return result

    return Verifier(
        "acceptance_test", ORACLE_MECHANICAL, _run,
        intent_classes=("*",), confirms_intent=True, cost=6, required=True,
    )


def build_authoring_objective(request: str, test_rel_path: str, test_source: str) -> str:
    """The contract handed to the live engine: build the capability so the
    acceptance test passes. The test source is shown as the spec; the canonical
    copy is (re)written by the verifier at check-time, so weakening it is futile."""
    return (
        f"Implement a new capability so the acceptance test at {test_rel_path} "
        "passes.\n\n"
        f"What the user wants:\n{request.strip()}\n\n"
        f"The acceptance test (the definition of done — make it pass):\n"
        f"```python\n{test_source}\n```\n\n"
        "Write the real implementation (a new verb/tool/source with actual logic, "
        "registered/wired as the codebase expects). Do NOT edit the acceptance "
        f"test — it is the contract and will be re-checked verbatim. Acceptance: "
        f"`pytest {test_rel_path}` passes."
    )


async def author_capability(
    request: str, *, repo_dir: str, conn: Any, driver: Any, model_invoke: ModelInvoke,
    user_id: str = "", run: Callable[..., Awaitable[Any]] | None = None,
    pytest_confirm_factory: PytestConfirmFactory | None = None, **run_kwargs: Any,
) -> tuple[Any | None, str]:
    """End-to-end: synthesize the acceptance test, then drive the LIVE edit engine
    to implement against it (the authored test plugged in as the confirm oracle).
    Returns ``(outcome, "")`` or ``(None, error)``. ``run`` defaults to the
    engine's ``run_self_edit`` (lazy-imported); injectable for tests."""
    rel, src_or_err = await synthesize_acceptance_test(request, model_invoke=model_invoke)
    if rel is None:
        return None, src_or_err
    test_source = src_or_err
    objective = build_authoring_objective(request, rel, test_source)
    verifier = acceptance_test_verifier(rel, test_source,
                                        pytest_confirm_factory=pytest_confirm_factory)

    runner = run
    if runner is None:
        from augmentum.selfedit.orchestrator import run_self_edit as runner
    outcome = await runner(
        repo_dir=repo_dir, objective=objective, user_id=user_id, conn=conn,
        driver=driver, extra_verifiers=[verifier], **run_kwargs,
    )
    log.info("capability_authored", request=request[:80],
             status=getattr(outcome, "status", "?"))
    return outcome, ""
