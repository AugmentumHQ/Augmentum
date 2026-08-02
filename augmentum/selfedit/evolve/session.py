"""One end-to-end evolve session — the usable path.

Given a prompt to improve and a plain-language goal, this builds a synthetic eval
set, derives a rubric from the goal, wires the GEPA seams to a single injected
``chat`` callable, and runs ``evolve``. The result is the honest verdict: the
evolved prompt iff it beat the original on a held-out split, else the original.

Everything model-facing is the one injected ``chat`` (messages -> text), so the
session is pure + testable with a fake chat; the route supplies the real model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable

from augmentum.selfedit.evolve.dataset import build_synthetic, merge
from augmentum.selfedit.evolve.gepa import FailureExample, evolve, max_size_constraint
from augmentum.selfedit.evolve.rubric import Criterion, Rubric
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Injected: chat takes OpenAI-style messages and returns the model's text.
Chat = Callable[[list[dict]], Awaitable[str]]


def _json_list(text: str) -> list:
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    i, j = t.find("["), t.rfind("]")
    if i >= 0 and j > i:
        try:
            v = json.loads(t[i:j + 1])
            return v if isinstance(v, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _json_obj(text: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            v = json.loads(t[i:j + 1])
            return v if isinstance(v, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


async def run_evolve_session(
    *, prompt: str, goal: str, chat: Chat, n_cases: int = 6,
    max_iterations: int = 2, samples: int = 1, finalists: int = 3,
    success_threshold: float = 0.08, hard_cap: int = 4000,
):
    """Build dataset + rubric, wire the seams to ``chat``, and evolve. Returns the
    ``EvolveResult``. Never raises for model hiccups inside the loop (the seams
    swallow + degrade); a fatal setup failure (no usable dataset) raises."""
    rubric = Rubric(name="goal", criteria=[
        Criterion("meets_goal", f"Does the output satisfy this goal: {goal}?", 1.0),
        Criterion("quality", "Is the output specific, correct, and high quality "
                  "(not vague, padded, or off-format)?", 1.0),
    ])

    async def generate(artifact_text: str, n: int) -> list:
        txt = await chat([{"role": "user", "content":
            f"A system prompt will be evaluated:\n\"\"\"{artifact_text}\"\"\"\n\n"
            f"The goal for its outputs is: {goal}\n\n"
            f"Emit {n} varied, realistic test INPUTS that exercise it (edge cases too). "
            f'Return ONLY a JSON array of objects: [{{"input": "...", "expectation": "..."}}].'}])
        return _json_list(txt)

    cases = await build_synthetic(prompt, generate=generate, n=n_cases)
    if len(cases) < 3:
        raise ValueError(f"could not build a usable eval set (got {len(cases)} cases)")
    dataset = merge(cases, name="evolve-session")

    async def run(variant: str, inp: str) -> str:
        return (await chat([{"role": "system", "content": variant},
                            {"role": "user", "content": inp}])).strip()

    async def judge(rb: Rubric, inp: str, output: str) -> dict:
        crit = "\n".join(f"- {c.name}: {c.question}" for c in rb.criteria)
        txt = await chat([{"role": "user", "content":
            f"Score the OUTPUT from 0.0 to 1.0 on each criterion.\n"
            f"INPUT: {inp}\nOUTPUT: {output}\n\nCRITERIA:\n{crit}\n\n"
            f'Return ONLY JSON like {{"meets_goal": 0.0, "quality": 0.0}}.'}])
        obj = _json_obj(txt)
        return {c.name: float(obj.get(c.name, 0.0) or 0.0) for c in rb.criteria}

    async def mutate(current: str, failures: list[FailureExample]) -> list[str]:
        fx = "\n".join(f"- input: {f.inp}\n  weak output: {f.observed_output}\n  score: {f.score:.2f}"
                       for f in failures) or "(no specific failures captured)"
        txt = await chat([{"role": "user", "content":
            f"Improve this SYSTEM PROMPT. Goal for its outputs: {goal}\n\n"
            f"Current prompt:\n\"\"\"{current}\"\"\"\n\n"
            f"It scored poorly on these cases — reflect on WHY, then fix the root cause:\n{fx}\n\n"
            f"Propose 3 improved system prompts (concise, general — not tailored to these "
            f"specific inputs). Return ONLY a JSON array of 3 strings."}])
        return [str(x).strip() for x in _json_list(txt) if str(x).strip()][:3]

    result = await evolve(
        prompt, dataset, mutate=mutate, run=run, rubric=rubric, judge=judge,
        constraints=[max_size_constraint(hard_cap)], max_iterations=max_iterations,
        n_failures=3, success_threshold=success_threshold, samples=samples,
        finalists=finalists,
    )
    log.info("evolve_session_done", accepted=result.accepted,
             improvement=round(result.improvement, 4), cases=len(cases))
    return result
