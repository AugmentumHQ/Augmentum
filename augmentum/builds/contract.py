"""Behavior contract — the spec-derived checklist a build must actually pass.

The build process used to grade itself by the *shape* of what the agent did
(did it call ``browser_evaluate``?), never the *outcome* (did the assertion
pass?). This module derives, FROM THE OBJECTIVE — not the code — a frozen list
of concrete, browser-observable behaviors the finished app must satisfy. The
gate (``builds/verify.py``) then runs each behavior against the real running
app and the build is "done" only when they pass.

Deriving from the objective (and freezing the list BEFORE the build) is the
anti-gaming anchor: the agent can build to satisfy the behaviors, but it cannot
weaken them — the *what* is fixed by the spec; only the *how* (selectors) is
bound to the implementation later, at gate time.

A behavior is a plain dict so it persists cleanly into the build_runs blob and
rides into the snapshot/UI:

    {"id": "tip-basic", "description": "...", "status": "untested",
     "evidence": ""}

``status`` is one of ``untested`` / ``pass`` / ``fail``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from augmentum.models.base import InternalChatRequest, Message, response_text
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Bounds: enough behaviors to cover core paths + edges, not so many the gate
# (and the fix loop) balloon. The Power's per-kind floor is ~3-5 checks.
_MIN_BEHAVIORS = 3
_MAX_BEHAVIORS = 10

_DERIVE_SYSTEM = (
    "You are a meticulous QA engineer. Given a one-line request for a web app, "
    "you enumerate the concrete, user-observable behaviors the finished app MUST "
    "exhibit — the acceptance criteria. You judge INTENT from the request, not "
    "any implementation."
)


def _derive_prompt(objective: str, kind: str) -> str:
    return (
        f"Request for a web app:\n\"{objective}\"\n\n"
        f"(app kind: {kind or 'general'})\n\n"
        "List the concrete behaviors this app must satisfy to be considered "
        "correct and complete. Rules:\n"
        "- Each behavior is ONE specific, browser-observable outcome a user "
        "could check by interacting with the page (not an implementation "
        "detail, not 'the code is clean').\n"
        "- Cover the main happy paths AND the edge / error cases the request "
        "implies (empty input, zero, division by zero, invalid input, reset/"
        "clear, persistence across reload if claimed).\n"
        f"- Between {_MIN_BEHAVIORS} and {_MAX_BEHAVIORS} behaviors. Prefer the "
        "highest-value ones if you'd exceed the max.\n"
        "- Phrase each as a checkable statement, e.g. \"Entering a bill of 100 "
        "and a tip of 15% shows a tip of 15.00 and a total of 115.00\".\n\n"
        "Return ONLY a JSON object, no prose, no code fence:\n"
        '{"behaviors": [{"id": "short-kebab-id", "description": "..."}, ...]}'
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerates code fences,
    leading reasoning, a stray ```json). Returns {} on failure."""
    if not text:
        return {}
    s = text.strip()
    # Drop a leading ```json / ``` fence if present.
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    if "{" in s and "}" in s:
        s = s[s.index("{"): s.rindex("}") + 1]
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _slug(value: str, fallback: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return (out or fallback)[:48]


def normalize_behaviors(raw: Any) -> list[dict]:
    """Coerce a model's behavior list into the canonical dict shape, dropping
    junk and de-duplicating ids. Pure — unit-testable without a backend."""
    items = raw.get("behaviors") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for i, it in enumerate(items):
        if isinstance(it, str):
            desc, bid = it, ""
        elif isinstance(it, dict):
            desc = str(it.get("description") or it.get("behavior") or "").strip()
            bid = str(it.get("id") or "").strip()
        else:
            continue
        if not desc:
            continue
        bid = _slug(bid or desc, f"behavior-{i + 1}")
        # De-dup ids.
        base, n = bid, 2
        while bid in seen:
            bid = f"{base}-{n}"
            n += 1
        seen.add(bid)
        out.append({"id": bid, "description": desc[:240], "status": "untested", "evidence": ""})
        if len(out) >= _MAX_BEHAVIORS:
            break
    return out


async def derive_behaviors(
    backend: Any, *, model: str, objective: str, kind: str = "",
) -> list[dict]:
    """Derive the frozen behavior contract from the objective via one LLM call.

    Returns a list of behavior dicts (status ``untested``). Never raises — on a
    backend/parse failure it returns ``[]`` and the caller falls back to the
    trail-based floor, so a flaky derivation can't block a build.
    """
    req = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content=_DERIVE_SYSTEM),
            Message(role="user", content=_derive_prompt(objective, kind)),
        ],
        temperature=0.2,
        # Behavior enumeration is a structured task — no chain-of-thought needed,
        # and reasoning models otherwise blow the latency/token budget here.
        chat_template_kwargs={"enable_thinking": False},
    )
    try:
        resp = await backend.chat(req)
    except Exception:  # noqa: BLE001 — derivation is best-effort
        log.warning("build_contract.derive_failed", exc_info=True)
        return []
    behaviors = normalize_behaviors(extract_json_object(response_text(resp)))
    if len(behaviors) < _MIN_BEHAVIORS:
        log.info("build_contract.derive_thin", count=len(behaviors))
    log.info("build_contract.derived", count=len(behaviors), kind=kind)
    return behaviors


def render_behaviors_for_build(behaviors: list[dict]) -> str:
    """A compact block injected into the build prompt so the agent builds
    toward the same behaviors the gate will check."""
    if not behaviors:
        return ""
    lines = [f"  {i + 1}. {b['description']}" for i, b in enumerate(behaviors)]
    return (
        "This app MUST satisfy these behaviors — they will be checked "
        "automatically in a real browser before the build is accepted:\n"
        + "\n".join(lines)
    )


def render_failures_for_fix(behaviors: list[dict]) -> str:
    """The concrete fix-loop feedback: which behaviors failed and the evidence
    the gate observed, so the agent fixes the real defect instead of guessing."""
    failed = [b for b in behaviors if b.get("status") == "fail"]
    if not failed:
        return ""
    lines = []
    for b in failed:
        ev = (b.get("evidence") or "").strip()
        lines.append(f"  - {b['description']}" + (f"\n      (observed: {ev})" if ev else ""))
    return (
        "The automated browser check FOUND THESE BEHAVIORS BROKEN — fix the "
        "root cause in the code, do not just retry:\n" + "\n".join(lines)
    )
