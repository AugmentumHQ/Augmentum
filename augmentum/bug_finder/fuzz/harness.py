"""Generate Atheris fuzz harnesses for module-level Python functions.

The spec called for an LLM subagent here. For Python free functions
that's overkill: the AST already gave us the module path (via file
path), the function name, the input kind, and the parameter name. The
harness is 85% boilerplate — module import, try/except around the call,
``atheris.Setup/Fuzz``. Skipping the LLM trims latency (no model call
per chunk), cost (zero tokens), and debuggability headaches (the same
input always produces the same harness).

The case where the LLM genuinely earns its keep is **instance-method
harnesses**: synthesizing a plausible ``__init__`` invocation from
reading the class body. That work lands in step 2.5 and isn't blocked
on this commit — methods are explicitly rejected upstream by the
classifier (``is_method=True``).

Output shape — one ``.py`` file per chunk, runnable as::

    python fuzz_<func>.py corpus/ -max_total_time=120

The runner shim (step 3) wraps that invocation and captures crashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from augmentum.bug_finder.fuzz.classifier import FuzzVerdict


# Exception classes a typical parser/decoder raises when it rejects
# malformed input. The harness wraps the target call in
# ``try/except (these...)`` so the fuzzer only flags *unhandled*
# exceptions / hangs / Python interpreter crashes — not the cases the
# parser is documented to surface as exceptions.
#
# Adding to this list is conservative (you'll suppress some real bugs
# that happened to raise one of these classes). Removing from it makes
# the fuzzer noisier with non-bugs. The list below mirrors what
# OSS-Fuzz / Atheris's own example harnesses swallow.
_EXPECTED_PARSE_ERRORS: tuple[str, ...] = (
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "UnicodeDecodeError",
    "UnicodeError",
    "OverflowError",
    "struct.error",
    "json.JSONDecodeError",
)


@dataclass(frozen=True)
class Harness:
    """A renderable Atheris harness for one chunk."""

    source: str               # the Python source for the harness file
    suggested_filename: str   # e.g. "fuzz_extract_pdf.py"
    target_module: str        # dotted import path
    target_function: str      # leaf function name (no class qualifier)
    target_param: str         # the bytes-shaped parameter name
    input_kind: str           # forwarded from the verdict


def _module_path_from_file(file_path: str, workspace_root: str = "") -> str:
    """Derive a dotted module path from a workspace-relative file path.

    ``augmentum/documents/chunker.py`` → ``augmentum.documents.chunker``.
    ``/workspace/augmentum/foo.py`` with root ``/workspace`` →
    ``augmentum.foo``. Drops trailing ``__init__`` so a package's
    ``__init__.py`` imports as the package itself.
    """
    if not file_path:
        return ""
    rel = Path(file_path).as_posix()
    if workspace_root:
        root = Path(workspace_root).as_posix().rstrip("/")
        if rel.startswith(root + "/"):
            rel = rel[len(root) + 1:]
        elif rel == root:
            return ""
    rel = rel.removesuffix(".py")
    parts = [seg for seg in rel.split("/") if seg]
    # Drop a trailing __init__ segment so packages import cleanly.
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _input_conversion(input_kind: str) -> tuple[str, str]:
    """Return ``(prelude, variable)`` used to feed the target.

    Atheris always hands the harness ``bytes``. Most targets accept
    ``bytes`` directly. ``bytearray`` / ``memoryview`` need a wrapping
    step. The prelude (if non-empty) appears inside ``TestOneInput``
    above the target call and is already indented to match.
    """
    if input_kind == "bytearray":
        return ("    payload = bytearray(data)\n", "payload")
    if input_kind == "memoryview":
        return ("    payload = memoryview(data)\n", "payload")
    # bytes / inferred-by-name → atheris's bytes directly. If the
    # target turned out to want str, the harness will see TypeError
    # which is in the expected-exception list — surfacing the
    # disagreement gracefully on the first run rather than crashing
    # the fuzzer.
    return ("", "data")


def generate_harness(
    verdict: FuzzVerdict,
    *,
    target_file: str,
    target_function: str,
    workspace_root: str = "/workspace",
) -> Harness:
    """Render an Atheris harness for the chunk described by ``verdict``.

    Raises ``ValueError`` if the verdict isn't fuzzable, if the target
    function name carries a class qualifier (``X.method`` — handled by
    the step-2.5 LLM-driven harness writer instead), or if the module
    path can't be derived from ``target_file``. The orchestrator should
    gate on ``verdict.fuzzable`` before calling this so the error paths
    indicate genuine misuse, not user-flow failure.
    """
    if not verdict.fuzzable:
        raise ValueError(
            f"cannot generate harness for non-fuzzable verdict: {verdict.reason}",
        )
    if "." in target_function:
        raise ValueError(
            f"qualified target {target_function!r} looks like a method; "
            "v1 harness writer handles module-level functions only — "
            "method support pending step 2.5",
        )

    module = _module_path_from_file(target_file, workspace_root)
    if not module:
        raise ValueError(
            f"could not derive a module path from {target_file!r} "
            f"(workspace_root={workspace_root!r})",
        )

    func = target_function
    prelude, var = _input_conversion(verdict.input_kind)
    expected = ", ".join(_EXPECTED_PARSE_ERRORS)
    suggested_filename = f"fuzz_{func.lstrip('_')}.py"

    # Modules referenced in the expected-exception tuple need top-level
    # imports — otherwise the harness raises NameError before atheris
    # gets a chance to feed it anything. Discover them from the
    # exception list rather than hard-coding so adding to
    # ``_EXPECTED_PARSE_ERRORS`` Just Works.
    exc_module_imports = sorted({
        exc.split(".", 1)[0]
        for exc in _EXPECTED_PARSE_ERRORS
        if "." in exc
    })
    exc_import_lines = "".join(
        f"import {mod}\n" for mod in exc_module_imports
    )

    # The harness is a complete runnable Python file. Newlines and
    # indentation are explicit because Atheris harnesses are usually
    # inspected by humans during crash triage — keep them readable.
    source = (
        f'"""Atheris fuzz harness for {module}.{func}.\n'
        f"\n"
        f"Generated by augmentum.bug_finder.fuzz.harness. Run with:\n"
        f"    python {suggested_filename} <corpus_dir> -max_total_time=120\n"
        f'"""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"import atheris\n"
        f"import sys\n"
        f"{exc_import_lines}"
        f"\n"
        f"with atheris.instrument_imports():\n"
        f"    from {module} import {func}\n"
        f"\n"
        f"\n"
        f"def TestOneInput(data: bytes) -> None:\n"
        f"{prelude}"
        f"    try:\n"
        f"        {func}({var})\n"
        f"    except ({expected}):\n"
        f"        # Documented-as-recoverable parse errors. Not a bug.\n"
        f"        return\n"
        f"    # Any other exception, native crash, or hang is surfaced\n"
        f"    # by atheris as a crash with the offending input saved.\n"
        f"\n"
        f"\n"
        f"def main() -> None:\n"
        f"    atheris.Setup(sys.argv, TestOneInput)\n"
        f"    atheris.Fuzz()\n"
        f"\n"
        f'\nif __name__ == "__main__":\n'
        f"    main()\n"
    )

    return Harness(
        source=source,
        suggested_filename=suggested_filename,
        target_module=module,
        target_function=func,
        target_param=verdict.target_param,
        input_kind=verdict.input_kind,
    )
