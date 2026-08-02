"""Render a mission (list of Promises) as text for prompt injection.

The mission log is the ``program counter`` visible to the model on
every ACT iteration — the ledger that makes ``done`` a predicate
rather than a vibe.
"""
from __future__ import annotations

from augmentum.promises.models import Promise, PromiseStatus, VerificationKind

# Plain-ASCII icons so the renderer works in any terminal / log pipeline
# and round-trips through prompt tokenization cleanly.
_ICON = {
    PromiseStatus.PENDING: "[ ]",
    PromiseStatus.IN_PROGRESS: "[>]",
    PromiseStatus.FULFILLED: "[x]",
    PromiseStatus.REJECTED: "[!]",
}

_EVIDENCE_MAX = 120


def render_mission_log(mission: list[Promise], *, header: str = "Mission") -> str:
    """Render the mission as a human/model-readable checklist.

    Example::

        Mission
        [x] install ncurses + build tools
            verified: libncurses-dev ii
        [>] clone the repo
            verify: `test -e /workspace/curseofwar/Makefile`
        [ ] build the binary
        [ ] smoke test
    """
    if not mission:
        return f"{header}\n(empty)"
    lines: list[str] = [header]
    for p in mission:
        lines.extend(_render_promise(p, indent=0))
    return "\n".join(lines)


def _render_promise(p: Promise, *, indent: int) -> list[str]:
    pad = "  " * indent
    icon = _ICON[p.status]
    attempt_tag = ""
    if p.attempts > 0 and p.status != PromiseStatus.FULFILLED:
        attempt_tag = f" (attempt {p.attempts + 1}/{p.max_attempts})"
    head = f"{pad}{icon} {p.description}{attempt_tag}"

    detail = _detail_line(p)
    lines: list[str] = [head]
    if detail:
        lines.append(f"{pad}    {detail}")
    for child in p.children:
        lines.extend(_render_promise(child, indent=indent + 1))
    return lines


def _detail_line(p: Promise) -> str:
    if p.status == PromiseStatus.FULFILLED and p.evidence:
        return f"verified: {_summarize(p.evidence)}"
    if p.status == PromiseStatus.REJECTED and p.evidence:
        return f"failed: {_summarize(p.evidence)}"
    if p.status in (PromiseStatus.PENDING, PromiseStatus.IN_PROGRESS):
        return _verify_preview(p)
    return ""


def _verify_preview(p: Promise) -> str:
    kind = p.verify.kind
    spec = p.verify.spec
    if kind == VerificationKind.SHELL and isinstance(spec.get("cmd"), str):
        return f"verify: `{_summarize(spec['cmd'], limit=80)}`"
    if kind == VerificationKind.FILE and isinstance(spec.get("path"), str):
        must = spec.get("must_exist", True)
        state = "exists" if must else "absent"
        return f"verify: {spec['path']} {state}"
    if kind == VerificationKind.HTTP and isinstance(spec.get("url"), str):
        return f"verify: HTTP {spec.get('method', 'GET')} {spec['url']}"
    if kind == VerificationKind.USER_CONFIRM and isinstance(spec.get("prompt"), str):
        return f"verify: user confirms '{_summarize(spec['prompt'])}'"
    if kind == VerificationKind.ANY_OF:
        checks = spec.get("checks") or []
        return f"verify: any of {len(checks)} checks"
    if kind == VerificationKind.ALWAYS:
        return ""
    return f"verify: ({kind.value})"


def _summarize(text: str, *, limit: int = _EVIDENCE_MAX) -> str:
    if not text:
        return ""
    first = text.strip().splitlines()[0]
    if len(first) > limit:
        return first[: limit - 1] + "…"
    return first
