"""Diagnosis engine — turn a failed probe into a complete dossier.

The second half of the harness, and the higher-leverage one: a probe tells
you *that* an endpoint broke; this turns "broke" into "here is the kind of
break, exactly where, the source line, and how hard to act on it."

Oracle-agnostic by design — anything that produces a ``ProbeResult``
(a contract probe, the bug-finder verifier, a failing test) feeds the same
classify -> localize -> tier pipeline. Diff-attribution and call-chain
(which need a changed-file set + call graph) are separate enrichers layered
on by the self-edit / coder consumers; this module handles the parts that
apply even when validating a single tree with no diff.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from augmentum.contracts.discover import RouteSpec

# Failure modes — each routes to a different response (see ``classify``).
CRASH = "crash"                    # 5xx / unhandled exception
STATUS_REGRESSION = "status"       # unexpected status (route vanished, etc.)
AUTHZ_FLIP = "authz_flip"          # protected route answered unauthenticated — security
SCHEMA_DRIFT = "schema"            # 2xx but body doesn't match the declared model
HANG = "hang"                      # exceeded the probe timeout — never returned
OK = "ok"

# Severity / action tiers — "knowing what to do" once we know the break.
HARD_BLOCK = "hard_block"          # security regression — halt, never auto-retry
REGRESSION = "regression"         # real break — feed the fix loop
ANNOTATE = "annotate"             # record, don't block (soft / service-dependent)
SUPPRESS = "suppress"             # not a real break (expected 4xx, mock limitation)

_FILE_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+), in (.+)$')


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Raw outcome of calling one endpoint."""

    route: RouteSpec
    status: int | None                # HTTP status, or None if the call raised
    exception: str = ""               # repr of an unhandled exception
    traceback_text: str = ""          # full server-side traceback, if captured
    body_snippet: str = ""            # first chunk of the response body
    timed_out: bool = False           # the probe exceeded its wall-clock (hang)

    @property
    def raised(self) -> bool:
        return bool(self.exception) and not self.timed_out


@dataclass(slots=True)
class Frame:
    file: str
    line: int
    func: str


@dataclass(slots=True)
class Dossier:
    """The complete, actionable package handed to whatever fixes the break."""

    route: RouteSpec
    failure_mode: str
    severity: str
    locus: str = ""                   # "auth/session.py:212" — deepest in-repo frame
    source_line: str = ""             # the offending source line, when available
    exception: str = ""
    status: int | None = None
    frames: list[Frame] = field(default_factory=list)
    traceback: str = ""               # full server-side traceback, for the repair artifact
    note: str = ""

    def render(self) -> str:
        head = f"[{self.failure_mode} | {self.severity}] {self.route.key}"
        lines = [head]
        if self.locus:
            lines.append(f"    at: {self.locus}")
        if self.source_line:
            lines.append(f"    src: {self.source_line}")
        if self.exception:
            lines.append(f"    exc: {self.exception}")
        elif self.status is not None:
            lines.append(f"    status: {self.status}")
        lines.append(f"    handler: {self.route.handler}")
        if self.note:
            lines.append(f"    note: {self.note}")
        return "\n".join(lines)


# Protected-path heuristic for the authz-flip check: a route that answered
# 2xx with NO auth presented, yet isn't on the public allow-list, is a leak.
_PUBLIC_PREFIXES = (
    "/api/auth/login", "/api/auth/setup", "/api/auth/status",
    "/api/auth/invite", "/api/cast/pair", "/api/capabilities",
    "/ui", "/static", "/docs", "/openapi.json", "/redoc", "/health",
)


def _is_public(path: str) -> bool:
    return any(path.startswith(p) for p in _PUBLIC_PREFIXES)


def classify(
    probe: ProbeResult,
    *,
    expected_status: int | None = None,
    authless: bool = False,
    is_public_fn: Callable[[str], bool] | None = None,
) -> str:
    """Decide the failure mode of a probe.

    ``expected_status`` is the baseline status when a differential is
    available (self-edit / coder). Without one, we fall back to absolute
    rules: 5xx/raise = crash, everything else = ok (a 4xx to a synthesized
    request is normal, not a break).

    ``authless`` marks probes sent with NO credentials — used only for the
    security authz-flip check. ``is_public_fn`` is the app's OWN
    "is this path intentionally public" predicate (injected by the consumer
    so this core stays app-agnostic and never drifts from the real auth
    allow-list); falls back to a small built-in prefix set if not supplied.
    """
    if probe.timed_out:
        return HANG
    if probe.raised:
        return CRASH
    status = probe.status
    if status is None:
        return CRASH
    # Security: an authless call that got a 2xx on a path the app does NOT
    # consider public is a genuine bypass. (With an active auth middleware,
    # in-process, this should be empty — the real signal is the differential
    # across an edit, which the self-edit consumer supplies via a baseline.)
    is_pub = is_public_fn(probe.route.path) if is_public_fn else _is_public(probe.route.path)
    if authless and 200 <= status < 300 and not is_pub:
        return AUTHZ_FLIP
    if status >= 500:
        return CRASH
    if expected_status is not None:
        if status == expected_status:
            return OK
        # A route that went 2xx -> 404/405 is a wiring/registration regression.
        if expected_status < 400 <= status:
            return STATUS_REGRESSION
    return OK


def _is_mock_artifact(probe: ProbeResult) -> bool:
    """A crash caused by the probe harness's own mocks (an un-awaitable
    MagicMock, a mocked service returning the wrong type) — never a real code
    bug. In-process harnesses stub heavy services, so these are noise to be
    demoted, not regressions to be reported."""
    blob = f"{probe.exception} {probe.traceback_text}"
    return (
        "MagicMock" in blob
        or "AsyncMock" in blob
        or "can't be used in 'await' expression" in blob
    )


def _severity_for(mode: str, probe: ProbeResult, *, in_repo: bool) -> str:
    if mode == AUTHZ_FLIP:
        return HARD_BLOCK
    if mode == OK:
        return SUPPRESS
    if mode == HANG:
        # A hang can't be told apart from a slow-under-mock blocking call
        # without a baseline, so it's annotate here — but it's surfaced as
        # its own category (never folded into crash noise) so a NEW hang vs
        # baseline reads as the potential-deadlock signal it is.
        return ANNOTATE
    if mode in (CRASH, STATUS_REGRESSION):
        # Harness-mock artifacts are never real bugs — demote to annotate.
        if _is_mock_artifact(probe):
            return ANNOTATE
        # A crash localized to our own code is a real regression; one that
        # dies in the mock/framework layer (no in-repo frame) is a harness
        # limitation, not a codebase break — annotate, don't fail.
        return REGRESSION if in_repo else ANNOTATE
    return ANNOTATE


def _parse_frames(traceback_text: str) -> list[Frame]:
    frames: list[Frame] = []
    for ln in traceback_text.splitlines():
        m = _FILE_FRAME.match(ln)
        if not m:
            continue
        raw = m.group(1)
        if raw.startswith("<"):  # <string>, <frozen ...> — not real source
            continue
        frames.append(Frame(file=raw.replace("\\", "/"), line=int(m.group(2)), func=m.group(3)))
    return frames


def _deepest_repo_frame(frames: list[Frame], repo_root: Path) -> Frame | None:
    """The deepest frame that lives under the repo's ``augmentum/`` tree —
    that's the error site the fixer should navigate to (skip site-packages
    and the framework's own frames)."""
    root = str(repo_root).replace("\\", "/")
    chosen: Frame | None = None
    for fr in frames:
        f = fr.file
        in_repo = f.startswith(root) and "/augmentum/" in f
        # also accept relative frames (subprocess runs) rooted at augmentum/
        in_repo = in_repo or "/augmentum/" in f or f.startswith("augmentum/")
        if in_repo:
            chosen = fr  # keep overwriting → ends on the deepest
    return chosen


def _read_source_line(repo_root: Path, frame: Frame) -> str:
    path = Path(frame.file)
    if not path.is_absolute():
        path = repo_root / frame.file
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if 1 <= frame.line <= len(lines):
            return lines[frame.line - 1].strip()[:160]
    except Exception:  # noqa: BLE001 — source unavailable, dossier still useful
        pass
    return ""


def build_dossier(
    probe: ProbeResult,
    repo_root: Path,
    *,
    expected_status: int | None = None,
    authless: bool = False,
    is_public_fn: Callable[[str], bool] | None = None,
) -> Dossier:
    """Classify a probe and gather every locally-available bit of context."""
    mode = classify(
        probe, expected_status=expected_status, authless=authless,
        is_public_fn=is_public_fn,
    )
    frames = _parse_frames(probe.traceback_text)
    repo_frame = _deepest_repo_frame(frames, repo_root)
    in_repo = repo_frame is not None
    severity = _severity_for(mode, probe, in_repo=in_repo)

    locus = ""
    source_line = ""
    if repo_frame is not None:
        base = repo_frame.file.split("/augmentum/", 1)[-1]
        locus = f"augmentum/{base}:{repo_frame.line}" if "/augmentum/" in repo_frame.file else f"{repo_frame.file}:{repo_frame.line}"
        source_line = _read_source_line(repo_root, repo_frame)

    note = ""
    if mode in (CRASH, STATUS_REGRESSION) and not in_repo:
        note = "no in-repo frame — likely a probe-harness/mock limitation, not a code break"
    if mode == HANG:
        note = "exceeded the probe timeout — possible deadlock/infinite await, or a slow blocking call under mocks; raise --timeout or investigate"
    if mode == AUTHZ_FLIP:
        note = "SECURITY: non-public route answered without credentials"

    return Dossier(
        route=probe.route,
        failure_mode=mode,
        severity=severity,
        locus=locus,
        source_line=source_line,
        exception=probe.exception,
        status=probe.status,
        frames=frames,
        traceback=probe.traceback_text,
        note=note,
    )
