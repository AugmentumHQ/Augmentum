"""Turn atheris crashes into bug_finder ``Finding`` rows.

Triage is deterministic in v1 — same philosophy as the harness writer.
Atheris already gives us the most expensive parts (exception class,
stack trace, crashing bytes); mapping those to a ``ClaimSignature`` +
severity is a fixed table. LLM enrichment (attacker-perspective
severity, exploitability assessment) is a planned step 2.5 but the
deterministic path produces a complete Finding row today.

Three responsibilities:

1. Classify each crash → ``(ClaimSignature, Severity)``.
2. Deduplicate by stack signature so the dozens of artifacts atheris
   minimizes for one underlying bug collapse into one Finding.
3. Build a Finding row with ``status=CONFIRMED`` (the crash IS the
   PoC) and family-confirmation set so the result flows through the
   existing ensemble accounting without surprising downstream.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

from augmentum.bug_finder.findings import (
    ClaimSignature,
    Finding,
    FindingStatus,
    Severity,
    _finding_id,
)
from augmentum.bug_finder.fuzz.harness import Harness
from augmentum.bug_finder.fuzz.runner import FuzzCrash, FuzzRunResult


# Family slug for cross-modal confirmation. When an LLM detector AND
# this fuzz leg agree on a finding, families_to_confirm reaches 2 —
# the spec's "cross-modal confirmation is the gold-standard FP killer".
FUZZ_FAMILY = "fuzz"


# Exception class → (signature, severity). Anything not listed
# defaults to (OTHER, LOW). The HARNESS's expected-exceptions filter
# already swallows ValueError / TypeError / KeyError / IndexError /
# UnicodeDecodeError / OverflowError / struct.error / json.JSONDecodeError
# as "documented parse failures, not bugs", so the table below only
# needs entries for what *gets through* that filter.
_CLASSIFICATION: dict[str, tuple[str, str]] = {
    "AttributeError":    (ClaimSignature.NULL_DEREF.value,     Severity.MEDIUM.value),
    "RecursionError":    (ClaimSignature.RESOURCE_LEAK.value,  Severity.MEDIUM.value),
    "MemoryError":       (ClaimSignature.RESOURCE_LEAK.value,  Severity.MEDIUM.value),
    "OSError":           (ClaimSignature.RESOURCE_LEAK.value,  Severity.LOW.value),
    "IOError":           (ClaimSignature.RESOURCE_LEAK.value,  Severity.LOW.value),
    "ZeroDivisionError": (ClaimSignature.LOGIC_ERROR.value,    Severity.LOW.value),
    "AssertionError":    (ClaimSignature.LOGIC_ERROR.value,    Severity.LOW.value),
    "RuntimeError":      (ClaimSignature.OTHER.value,          Severity.LOW.value),
    "SystemError":       (ClaimSignature.TYPE_CONFUSION.value, Severity.HIGH.value),
}


def _classify_crash(crash: FuzzCrash) -> tuple[str, str]:
    """Pick (signature, severity) for one crash.

    Special cases worth surfacing distinctly from the table lookup:
      * ``AttributeError`` + ``NoneType`` in the message is the
        canonical null-deref pattern.
      * libfuzzer's ``deadly signal`` line in the stack means a
        SIGSEGV / SIGABRT from a native C extension — that's
        memory-safety territory regardless of Python class.
    """
    if (
        crash.exception_class == "AttributeError"
        and "NoneType" in crash.exception_message
    ):
        return ClaimSignature.NULL_DEREF.value, Severity.MEDIUM.value
    if "deadly signal" in crash.stack_trace.lower():
        return ClaimSignature.USE_AFTER_FREE.value, Severity.HIGH.value
    return _CLASSIFICATION.get(
        crash.exception_class,
        (ClaimSignature.OTHER.value, Severity.LOW.value),
    )


_FRAME_RE = re.compile(
    r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)',
)


def _stack_signature(stack_trace: str, *, top_n: int = 3) -> str:
    """Hash the top ``top_n`` traceback frames into a dedup key.

    Atheris minimizes a crashing input down to a tiny triggering case
    via several intermediate artifacts. They all point to the same
    leaf line in the target; collapsing on a few frames groups them
    into one Finding. Returns "" when no frames parsed — caller falls
    back to the crash basename as the dedup key.
    """
    frames: list[str] = []
    for m in _FRAME_RE.finditer(stack_trace):
        frames.append(f"{m.group('file')}:{m.group('func')}:{m.group('line')}")
        if len(frames) >= top_n:
            break
    if not frames:
        return ""
    return hashlib.sha256("|".join(frames).encode("utf-8")).hexdigest()[:16]


def _is_harness_frame(file_path: str, harness_filename: str) -> bool:
    """Recognise frames that point at the harness or atheris itself.

    The harness's ``TestOneInput`` frame is structurally always at the
    top of the trace; we want the Finding's anchor to be in user code,
    not in the auto-generated wrapper.
    """
    name = PurePosixPath(file_path).name
    if name == harness_filename:
        return True
    return "atheris" in file_path.replace("\\", "/").split("/")


def _leaf_user_frame(
    stack_trace: str,
    *,
    harness_filename: str,
    fallback_file: str,
    fallback_function: str,
) -> tuple[str, str]:
    """Return ``(file, function)`` of the bottommost non-harness frame.

    "Bottommost" because that's where the crash actually fires —
    the leaf of the call chain. We skip frames that point at the
    harness or atheris's instrumentation shim so the Finding lands
    on user code.
    """
    leaf_file = fallback_file
    leaf_func = fallback_function
    for m in _FRAME_RE.finditer(stack_trace):
        f = m.group("file")
        if _is_harness_frame(f, harness_filename):
            continue
        leaf_file = f
        leaf_func = m.group("func")
    return leaf_file, leaf_func


def triage_fuzz_run(
    result: FuzzRunResult,
    *,
    harness: Harness,
    chunk_file: str,
    chunk_function: str,
    artifact_dir: str = "",
) -> list[Finding]:
    """Turn a fuzz result's crashes into deduped Finding rows.

    When ``result.skipped`` is True or there are no crashes, returns
    an empty list. The orchestrator logs the skip context from the
    result itself.

    ``artifact_dir`` (if supplied) is prepended to each crash's
    ``input_basename`` to form ``repro_path``. Pass the same path the
    runner wrote artifacts to so a human can ``cat`` the file.

    Findings produced:
      * ``status = CONFIRMED`` — fuzz crashes are PoCs; no further
        verifier step.
      * ``runs_to_confirm = total_runs = 1`` — fuzz doesn't replicate
        across N detector runs the way LLM findings do; the crash is
        a single observation. The ensemble's family axis is what gives
        cross-modal confirmation a number to multiply against.
      * ``families_to_confirm = total_families = 1`` — this leg saw
        the bug. When merged with LLM detector findings on the same
        chunk, the family count goes up.
    """
    if result.skipped or not result.crashes:
        return []

    by_sig: dict[str, FuzzCrash] = {}
    for crash in result.crashes:
        sig = _stack_signature(crash.stack_trace) or crash.input_basename
        if sig not in by_sig:
            by_sig[sig] = crash

    findings: list[Finding] = []
    for crash in by_sig.values():
        claim_sig, severity = _classify_crash(crash)
        leaf_file, leaf_func = _leaf_user_frame(
            crash.stack_trace,
            harness_filename=harness.suggested_filename,
            fallback_file=chunk_file,
            fallback_function=chunk_function,
        )

        evidence: tuple[str, ...] = (f"{leaf_file}:{leaf_func}",)
        msg_tail = (
            f": {crash.exception_message}"
            if crash.exception_message else ""
        )
        claim = (
            f"Fuzzer triggered {crash.exception_class}{msg_tail} "
            f"via input saved at {crash.input_basename}."
        )
        repro_path = (
            f"{artifact_dir.rstrip('/')}/{crash.input_basename}"
            if artifact_dir else crash.input_basename
        )
        replay_module = harness.target_module
        replay_function = harness.target_function
        suggested = (
            f"Replay: python3 -c \"from {replay_module} import "
            f"{replay_function}; {replay_function}("
            f"open('{crash.input_basename}', 'rb').read())\""
        )
        fid = _finding_id(leaf_file, leaf_func, claim_sig, evidence)
        findings.append(Finding(
            id=fid,
            file=leaf_file,
            function=leaf_func,
            claim=claim,
            claim_signature=claim_sig,
            severity=severity,
            evidence_paths=evidence,
            suggested_repro=suggested,
            status=FindingStatus.CONFIRMED.value,
            runs_to_confirm=1,
            total_runs=1,
            families_to_confirm=1,
            total_families=1,
            repro_path=repro_path,
            repro_command=(
                f"python3 {harness.suggested_filename} "
                f"{repro_path}"
            ),
            repro_output=crash.stack_trace,
            notes=[
                f"fuzz family — atheris harness, "
                f"{result.iterations} iters in "
                f"{result.runtime_seconds:.1f}s",
            ],
        ))
    return findings
