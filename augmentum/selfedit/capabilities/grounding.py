"""Grounded acceptance-test synthesis — fix the two failure modes the live
Qwen3.6-35B run exposed.

The free-form path (``authoring.synthesize_acceptance_test``) let the model guess
our API and it got the *mechanics* wrong (sync call of an async handler, wrong
``ActionResult`` import, ``REGISTRY.get`` treated as callable) while getting the
*behavior* right ("212 in the spoken line"). Two grounding moves close that gap:

1. **Scaffolded synthesis** — for verb-shaped capabilities, don't ask the model
   for the whole test. Ask only for what it's good at: the verb id, sample args,
   and the behavioral assertions on the result. We render the CORRECT harness
   (imports, SessionContext, ``asyncio.run(action.handler(...))``) around them, so
   the mechanics are right by construction.

2. **The oracle gate** — a synthesized test is only a trustworthy oracle if, run
   against the CURRENT (unbuilt) code, it FAILS for the RIGHT reason: the
   capability is missing. A test that passes already is vacuous; one that errors
   on a broken import is testing nothing real. ``oracle_verdict`` distinguishes
   valid / vacuous / broken so we never hand the engine an oracle it can satisfy
   by accident — or one it can never satisfy because the test itself is wrong.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from augmentum.selfedit.capabilities.render import syn_module_stem
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

ModelInvoke = Callable[[str], Awaitable[str]]
# (test_source) -> (passed, combined_output)
PytestRunner = Callable[[str], Awaitable[tuple[bool, str]]]

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")

_SCAFFOLD_PROMPT = """\
A new primitive VERB is being added to Augmentum's action registry. The pytest
harness that loads and dispatches it is ALREADY written for you — you do not
write imports or call the handler. You ONLY specify how to recognize success.

The capability the user wants:
{request}

Return ONE JSON object (no prose, no code fence):
{{
  "id": "surface.action",          // lowercase, e.g. "unit.c_to_f"
  "args": {{"celsius": 100}},      // sample args the handler is dispatched with
  "assertions": ["\\"212\\" in res.speak"],  // boolean expressions on `res`
  "summary": "one line: what the verb does"
}}

`res` is the ActionResult the handler returns. Its fields:
  res.speak (str), res.short_circuit (bool), res.toast (str),
  res.surface_emit (dict|None with 'channel'/'payload'), res.fulfilled (bool),
  res.prompt_addendum (str).
Each assertion must be a boolean Python EXPRESSION that references `res` and
captures the OBSERVABLE behavior the user asked for. Keep them minimal and real.
"""

_REPAIR = "\n\nYour previous answer was invalid: {problem}\nReturn corrected JSON only."


def _slug(s: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:40] or "verb").rstrip("_")


def _extract_json(text: str) -> dict | None:
    s = (text or "").strip()
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        obj = json.loads(s[a : b + 1])
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def _validate_scaffold(obj: dict) -> tuple[str, str]:
    """Return (problem, '') — empty problem means valid."""
    vid = str(obj.get("id", ""))
    if not _ID_RE.match(vid):
        return f"id {vid!r} must be lowercase surface.action", ""
    if not isinstance(obj.get("args"), dict):
        return "args must be a JSON object (the sample dispatch args)", ""
    assertions = obj.get("assertions")
    if not isinstance(assertions, list) or not assertions:
        return "assertions must be a non-empty list", ""
    for a in assertions:
        if not isinstance(a, str) or "res" not in a:
            return f"assertion {a!r} must be an expression referencing `res`", ""
        try:
            compile(a, "<assertion>", "eval")   # expression only — no statements/imports
        except SyntaxError as exc:
            return f"assertion {a!r} is not a valid expression: {exc}", ""
    return "", ""


def render_scaffolded_test(obj: dict) -> tuple[str, str]:
    """Render the correct harness around the model's id/args/assertions. Returns
    (test_rel_path, source). Mechanics are ours; behavior is the model's."""
    vid = obj["id"]
    safe = vid.replace(".", "_")
    args = dict(obj["args"])
    asserts = "\n".join(f"    assert {a}" for a in obj["assertions"])
    rel = f"tests/test_authored_{safe}.py"
    src = (
        f'"""Scaffolded acceptance oracle for {vid} — harness is machine-rendered '
        "(correct by construction); assertions are model-authored behavior.\n"
        "\n"
        "Verifies via the BOOT PATH: the package import runs intent/__init__ (the\n"
        "startup wiring), then asserts the verb is registered. No direct module\n"
        "import - that would mask a skipped intent/__init__ registration line, so\n"
        "the verb would pass its oracle yet be dead at real boot.\n"
        f'Summary: {str(obj.get("summary", "")).strip()[:200]}"""\n'
        "from __future__ import annotations\n"
        "\n"
        "import asyncio\n"
        "\n"
        "from augmentum.intent import REGISTRY  # runs intent/__init__ = the boot wiring\n"
        "from augmentum.intent.action import SessionContext\n"
        "\n"
        "\n"
        f"def test_{safe}_capability():\n"
        f"    action = REGISTRY.get({vid!r})\n"
        f"    assert action is not None, \"verb {vid} must be registered\"\n"
        "    ctx = SessionContext(user_id=\"u\", session_id=\"s\")\n"
        f"    res = asyncio.run(action.handler(\"\", ctx, {args!r}))\n"
        "    assert res is not None\n"
        f"{asserts}\n"
    )
    return rel, src


def syn_module_stem_for(vid: str) -> str:
    """The builtin module the engine should create the verb in (matches the
    scaffold's import + render.syn_module_stem convention)."""
    from augmentum.selfedit.capabilities.spec import CapabilitySpec
    return syn_module_stem(CapabilitySpec(id=vid, behavior="speak"))


async def synthesize_verb_acceptance(
    request: str, *, model_invoke: ModelInvoke,
) -> tuple[str | None, str]:
    """Scaffolded synthesis for a verb-shaped capability. Returns
    ``(test_rel_path, source)`` or ``(None, error)``. The model supplies only
    id/args/assertions; the harness is rendered correctly here."""
    async def _ask(prompt: str) -> dict | None:
        try:
            raw = await model_invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            log.warning("scaffold_model_failed", error=repr(exc))
            return None
        return _extract_json(raw)

    obj = await _ask(_SCAFFOLD_PROMPT.format(request=request))
    problem = _validate_scaffold(obj)[0] if obj is not None else "no JSON object returned"
    if obj is not None and not problem:
        rel, src = render_scaffolded_test(obj)
        log.info("scaffolded_oracle_synthesized", id=obj["id"], path=rel)
        return rel, src

    obj2 = await _ask(_SCAFFOLD_PROMPT.format(request=request) + _REPAIR.format(problem=problem))
    problem2 = _validate_scaffold(obj2)[0] if obj2 is not None else "no JSON object returned"
    if obj2 is None or problem2:
        log.info("scaffolded_oracle_rejected", problem=problem2)
        return None, problem2
    rel, src = render_scaffolded_test(obj2)
    return rel, src


# --- the oracle gate: does the test fail for the RIGHT reason? --------------

async def _default_run_pytest(test_source: str) -> tuple[bool, str]:
    """Run the test against the CURRENT code and return (passed, output).

    Writes to the SYSTEM temp dir (the repo's ``/app`` is a read-only dev-bind in
    the container) and puts the repo root on PYTHONPATH so ``import augmentum``
    resolves from the subprocess regardless of cwd."""
    # capabilities → selfedit → augmentum → <repo root>
    repo_root = str(Path(__file__).resolve().parents[3])
    fd, path = tempfile.mkstemp(prefix="oracle_", suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(test_source)
        env = dict(os.environ)
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pytest", path, "-q", "--no-header", "-p",
            "no:cacheprovider",
            cwd=tempfile.gettempdir(), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        return proc.returncode == 0, out.decode("utf-8", "replace")
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


async def oracle_verdict(
    test_source: str, *, expected_missing_module: str = "", run_pytest: PytestRunner | None = None,
) -> tuple[str, str]:
    """Classify a synthesized oracle by running it against the UNBUILT code:

      valid   — fails because the capability is missing (the right reason)
      vacuous — passes already (asserts nothing real)
      broken  — fails for the wrong reason (bad import, syntax, wrong API)

    Only ``valid`` oracles should be handed to the engine."""
    runner = run_pytest or _default_run_pytest
    passed, output = await runner(test_source)
    low = output.lower()
    # pytest not available to this process (the app container shares this gap with
    # the engine's own gate.pytest_check) — the real oracle runs at build time in
    # the candidate worktree. Honest "unverified", not a false "broken".
    if "no module named pytest" in low:
        return "unverified", "pytest unavailable here; oracle will be gated at build time in the worktree"
    if passed:
        return "vacuous", "the test passes before the capability exists — it asserts nothing real"
    em = (expected_missing_module or "").lower()
    right_signals = []
    if em and (f"no module named '{em}'" in low or em in low and "modulenotfounderror" in low):
        right_signals.append("impl module not yet present")
    if "must be registered" in low or ("verb" in low and "registered" in low):
        right_signals.append("verb-not-registered assertion")
    if right_signals:
        return "valid", "fails because " + " / ".join(right_signals) + " (right reason)"
    return "broken", f"fails for the wrong reason (not 'capability missing'):\n{output[-400:]}"
