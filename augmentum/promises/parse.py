"""Parse an LLM-emitted JSON mission into a ``list[Promise]``.

Accepts the schema produced by ``MISSION_PLAN_SYSTEM``:

    [
      {"desc": "install deps", "verify": {"kind": "shell", "cmd": "..."}},
      {"desc": "build", "verify": {"kind": "file", "path": "/..."}}
    ]

The parser is tolerant of:
- Leading/trailing prose or markdown fences (takes the first JSON array)
- Either ``desc`` or ``description`` keys
- Flat verify dicts (``{"kind": "shell", "cmd": "..."}``) as well as
  the canonical ``{"kind": "shell", "spec": {"cmd": "..."}}`` form
"""
from __future__ import annotations

import json
import re

from augmentum.promises.models import Promise, Verification, VerificationKind

_ARRAY_RE = re.compile(r"\[[\s\S]*\]")

# Keys that belong inside `spec` rather than at the top level of the verify dict.
_SPEC_KEYS = {
    "cmd", "path", "must_exist", "timeout",
    "url", "method", "expected_status",
    "prompt", "expected",
    "checks",  # any_of composite
}


def parse_mission_json(text: str) -> list[Promise]:
    """Parse an LLM response into a mission. Returns ``[]`` on failure.

    The model is supposed to output ONLY a JSON array, but real models
    leak prose and markdown. We extract the first ``[...]`` block we
    can parse.
    """
    if not text:
        return []
    candidate = _extract_array(text)
    if candidate is None:
        return []
    try:
        raw = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    mission: list[Promise] = []
    for item in raw:
        promise = _promise_from_raw(item)
        if promise is not None:
            mission.append(promise)
    return mission


def _extract_array(text: str) -> str | None:
    # Strip common markdown fences first
    stripped = text.strip()
    for fence in ("```json", "```"):
        if stripped.startswith(fence):
            stripped = stripped[len(fence):].lstrip("\n")
        if stripped.endswith("```"):
            stripped = stripped[: -3].rstrip()
    # Fast path: the whole thing is an array
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    # Slow path: find the first balanced [...] block
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(stripped):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start != -1:
                return stripped[start : i + 1]
    # Fall back to the simple regex — catches arrays buried in prose
    m = _ARRAY_RE.search(stripped)
    return m.group(0) if m else None


def _promise_from_raw(item: object) -> Promise | None:
    if not isinstance(item, dict):
        return None
    description = str(item.get("desc") or item.get("description") or "").strip()
    if not description:
        return None
    verify = _verify_from_raw(item.get("verify"))
    max_attempts = item.get("max_attempts") or 3
    try:
        max_attempts = max(1, int(max_attempts))
    except (TypeError, ValueError):
        max_attempts = 3
    return Promise(
        description=description,
        verify=verify,
        max_attempts=max_attempts,
    )


def parse_prose_plan(text: str) -> list[Promise]:
    """Fallback: extract a numbered plan from prose into always-verify promises.

    Smaller local models frequently emit plans as prose like
    ``1. Clone the repo 2. Build 3. Run`` instead of the requested JSON.
    This parser rescues those cases by splitting on ``N.`` / ``N)``
    boundaries. Resulting promises use ``always`` verification since
    we cannot infer postconditions from prose — the act layer is then
    responsible for getting each step to observable completion.

    Returns ``[]`` if fewer than 2 plausible steps are found, so the
    caller can treat this as "could not plan" rather than building a
    trivially-passing single-step mission from a greeting.
    """
    if not text:
        return []
    # Locate digit-dot or digit-paren markers that look like step numbers.
    # Require a preceding boundary (newline, start, space) so we don't
    # match e.g. "3.14" inside prose.
    markers = list(re.finditer(r"(?:^|\s|\r|\n)(\d+)[.)]\s+", text))
    if len(markers) < 2:
        return []

    # Verify the step numbers are sequential enough to be a list (tolerate
    # skips but require monotonic increase).
    numbers = [int(m.group(1)) for m in markers]
    if not all(b >= a for a, b in zip(numbers, numbers[1:], strict=False)):
        return []

    steps: list[str] = []
    for i, match in enumerate(markers):
        start = match.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        body = text[start:end].strip().rstrip(".").strip()
        # Strip stray markdown bullets / backticks that might leak in
        body = body.lstrip("-* ").strip()
        # Drop obvious section headers and oversize blobs
        if 4 <= len(body) <= 240 and not body.lower().startswith(("plan", "steps", "note")):
            steps.append(body)
    if len(steps) < 2:
        return []
    return [
        Promise(description=s, verify=Verification.always())
        for s in steps
    ]


def _verify_from_raw(raw: object) -> Verification:
    if not isinstance(raw, dict):
        return Verification.always()
    kind_str = raw.get("kind")
    if not isinstance(kind_str, str):
        return Verification.always()
    try:
        kind = VerificationKind(kind_str.lower())
    except ValueError:
        return Verification.always()
    # Accept either flat form (keys at top level) or {"spec": {...}}.
    spec = dict(raw.get("spec") or {})
    for k, v in raw.items():
        if k in ("kind", "spec"):
            continue
        if k in _SPEC_KEYS:
            spec[k] = v
    return Verification(kind=kind, spec=spec)
