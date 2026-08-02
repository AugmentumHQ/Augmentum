"""The clarify / triage gate — ask before you build.

The live 35B run's Case B ("make it handle my stuff better") confabulated a
meaningless capability because nothing stood between a vague request and the
synthesizer. This gate is that something. Before any spec/test is written, a
single closed-world pass classifies the request:

  ready   — clear + feasible enough to write a test for → proceed (carries a
            crisp ``normalized`` restatement + a ``shape`` routing hint)
  clarify — ambiguous in a way that would change the test → return targeted
            questions (AskUserQuestion-shaped) for the Workshop/voice to render
  refuse  — can't be a safe, verifiable capability (irreversible/external,
            needs absent infra, fundamentally unverifiable) → say why, early

The human-in-the-loop clusters HERE, at the point where intent is captured —
because in test-first authoring the test *is* the spec, so a wrong-or-vague
intent becomes a confidently-verified wrong build. ``apply_clarifications`` folds
the user's answers back into a clarified request for synthesis.

Pure + injected ``model_invoke`` → unit-testable; engine-agnostic in production.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

ModelInvoke = Callable[[str], Awaitable[str]]

STATUSES = frozenset({"ready", "clarify", "refuse"})
SHAPES = frozenset({"verb", "tool", "source", "other"})


@dataclass
class ClarifyOption:
    label: str
    description: str = ""


@dataclass
class ClarifyQuestion:
    question: str
    header: str = ""
    options: list[ClarifyOption] = field(default_factory=list)
    multi_select: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question, "header": self.header,
            "multiSelect": self.multi_select,
            "options": [{"label": o.label, "description": o.description} for o in self.options],
        }


@dataclass
class TriageResult:
    status: str                       # ready | clarify | refuse
    normalized: str = ""              # ready: a crisp restatement to synthesize from
    shape: str = "verb"              # ready: verb|tool|source|other (routing hint)
    reason: str = ""                 # refuse: the honest why
    questions: list[ClarifyQuestion] = field(default_factory=list)  # clarify

    def to_dict(self) -> dict:
        return {
            "status": self.status, "normalized": self.normalized, "shape": self.shape,
            "reason": self.reason, "questions": [q.to_dict() for q in self.questions],
        }


_PROMPT = """\
You are the intake gate for Augmentum's capability authoring. A capability is a
new primitive the assistant can DO — it will be built test-first (an acceptance
test is written, then code until it passes). Classify this request:

{request}

Return ONE JSON object (no prose, no code fence):
{{
  "status": "ready" | "clarify" | "refuse",
  "normalized": "crisp one-line restatement of exactly what to build",   // status=ready
  "shape": "verb" | "tool" | "source" | "other",                          // status=ready
  "reason": "one honest line",                                            // status=refuse
  "questions": [                                                           // status=clarify
    {{"question": "...?", "header": "<=12 chars",
      "multiSelect": false,
      "options": [{{"label": "...", "description": "..."}}, ...]}}   // 2-4 options each
  ]
}}

Rules:
- READY only if the request is specific enough that you could write a pass/fail
  test for it without guessing. Give the crisp `normalized` form + a `shape`.
- CLARIFY when it's ambiguous in a way that would change the TEST (which data?
  what format/units? what trigger phrasing? what scope?). Ask 1-3 TARGETED
  questions, each with 2-4 concrete options. Do NOT ask about trivia that
  wouldn't change the build.
- REFUSE when it can't be a safe, verifiable capability: anything irreversible or
  external (send/post/pay/delete/call), anything needing credentials or infra
  that isn't present, or anything fundamentally unverifiable ("make it better").
  Give a short honest `reason`.
"""

_REPAIR = "\n\nYour previous answer was invalid: {problem}\nReturn corrected JSON only."


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


def _parse_triage(obj: dict) -> tuple[TriageResult | None, str]:
    status = str(obj.get("status", ""))
    if status not in STATUSES:
        return None, f"status {status!r} must be one of {sorted(STATUSES)}"
    if status == "refuse":
        reason = str(obj.get("reason", "")).strip()
        if not reason:
            return None, "refuse requires a reason"
        return TriageResult("refuse", reason=reason), ""
    if status == "ready":
        normalized = str(obj.get("normalized", "")).strip()
        if not normalized:
            return None, "ready requires a normalized restatement"
        shape = str(obj.get("shape", "verb"))
        shape = shape if shape in SHAPES else "other"
        return TriageResult("ready", normalized=normalized, shape=shape), ""
    # clarify
    raw_qs = obj.get("questions")
    if not isinstance(raw_qs, list) or not raw_qs:
        return None, "clarify requires a non-empty questions list"
    questions: list[ClarifyQuestion] = []
    for q in raw_qs[:4]:
        if not isinstance(q, dict) or not str(q.get("question", "")).strip():
            return None, "each question needs a 'question' string"
        opts = [
            ClarifyOption(str(o.get("label", "")).strip(), str(o.get("description", "")).strip())
            for o in (q.get("options") or []) if isinstance(o, dict) and str(o.get("label", "")).strip()
        ]
        if len(opts) < 2:
            return None, f"question {q.get('question')!r} needs at least 2 options"
        questions.append(ClarifyQuestion(
            question=str(q["question"]).strip(), header=str(q.get("header", "")).strip()[:12],
            options=opts[:4], multi_select=bool(q.get("multiSelect", False)),
        ))
    return TriageResult("clarify", questions=questions), ""


async def triage_capability_request(
    request: str, *, model_invoke: ModelInvoke,
) -> TriageResult:
    """Classify a capability request as ready / clarify / refuse. Never raises —
    a model failure degrades to a single clarify question (safer than guessing)."""
    async def _ask(prompt: str) -> dict | None:
        try:
            return _extract_json(await model_invoke(prompt))
        except Exception as exc:  # noqa: BLE001
            log.warning("triage_model_failed", error=repr(exc))
            return None

    obj = await _ask(_PROMPT.format(request=request))
    res, problem = _parse_triage(obj) if obj is not None else (None, "no JSON object")
    if res is not None:
        log.info("capability_triage", status=res.status, shape=res.shape)
        return res

    obj2 = await _ask(_PROMPT.format(request=request) + _REPAIR.format(problem=problem))
    res2, problem2 = _parse_triage(obj2) if obj2 is not None else (None, problem)
    if res2 is not None:
        return res2

    # Degrade safely: ask rather than assume.
    log.info("capability_triage_degraded", problem=problem2)
    return TriageResult("clarify", questions=[ClarifyQuestion(
        question="I couldn't parse that request — what exactly should the new ability do?",
        header="Clarify",
        options=[ClarifyOption("Let me rephrase", "I'll describe it differently"),
                 ClarifyOption("Never mind", "Cancel this")],
    )])


def apply_clarifications(request: str, answers: dict[str, Any]) -> str:
    """Fold the user's answers to clarify questions back into a single clarified
    request for synthesis. ``answers`` maps question text → chosen label(s)."""
    if not answers:
        return request
    lines = []
    for q, a in answers.items():
        val = ", ".join(a) if isinstance(a, list) else str(a)
        lines.append(f"- {q.strip()} → {val.strip()}")
    return f"{request.strip()}\n\nClarifications:\n" + "\n".join(lines)
