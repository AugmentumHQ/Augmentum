"""Atheris harness writer tests.

The harness writer is deterministic, so the tests are exact: given a
verdict + path, we know the exact module import line, the input
conversion prelude, and the expected-exception list. We also parse
every generated harness with ``ast.parse`` to verify the output is
syntactically valid Python — a regression in the template that broke
parsing would otherwise only be caught at fuzz-run time.
"""

from __future__ import annotations

import ast

import pytest

from augmentum.bug_finder.fuzz.classifier import FuzzVerdict
from augmentum.bug_finder.fuzz.harness import (
    Harness,
    _EXPECTED_PARSE_ERRORS,
    _input_conversion,
    _module_path_from_file,
    generate_harness,
)


# ---------------------------------------------------------------------------
# _module_path_from_file
# ---------------------------------------------------------------------------


def test_module_path_strips_py_extension_and_dots() -> None:
    assert _module_path_from_file("augmentum/foo.py") == "augmentum.foo"
    assert _module_path_from_file("augmentum/sub/bar.py") == "augmentum.sub.bar"


def test_module_path_handles_workspace_root_prefix() -> None:
    assert _module_path_from_file(
        "/workspace/augmentum/foo.py", workspace_root="/workspace",
    ) == "augmentum.foo"


def test_module_path_handles_trailing_slash_root() -> None:
    assert _module_path_from_file(
        "/workspace/augmentum/foo.py", workspace_root="/workspace/",
    ) == "augmentum.foo"


def test_module_path_drops_init_segment() -> None:
    """``pkg/__init__.py`` imports as ``pkg``."""
    assert _module_path_from_file("augmentum/__init__.py") == "augmentum"


def test_module_path_handles_windows_separator() -> None:
    assert _module_path_from_file("augmentum\\foo\\bar.py") == "augmentum.foo.bar"


def test_module_path_returns_empty_when_no_input() -> None:
    assert _module_path_from_file("") == ""


# ---------------------------------------------------------------------------
# _input_conversion
# ---------------------------------------------------------------------------


def test_input_conversion_bytes_no_prelude() -> None:
    prelude, var = _input_conversion("bytes")
    assert prelude == ""
    assert var == "data"


def test_input_conversion_bytearray_wraps() -> None:
    prelude, var = _input_conversion("bytearray")
    assert "bytearray(data)" in prelude
    assert var == "payload"


def test_input_conversion_memoryview_wraps() -> None:
    prelude, var = _input_conversion("memoryview")
    assert "memoryview(data)" in prelude
    assert var == "payload"


def test_input_conversion_inferred_falls_back_to_bytes() -> None:
    prelude, var = _input_conversion("inferred-by-name")
    assert prelude == ""
    assert var == "data"


# ---------------------------------------------------------------------------
# generate_harness — happy path
# ---------------------------------------------------------------------------


def _fuzzable(kind: str = "bytes", param: str = "data") -> FuzzVerdict:
    return FuzzVerdict(
        fuzzable=True, target_param=param, input_kind=kind,
    )


def test_generate_harness_bytes_produces_valid_python() -> None:
    h = generate_harness(
        _fuzzable("bytes"),
        target_file="augmentum/documents/chunker.py",
        target_function="extract_text",
    )
    # The whole point: parse cleanly. A broken template here would
    # otherwise only fail at fuzz-run time.
    ast.parse(h.source)
    assert h.target_module == "augmentum.documents.chunker"
    assert h.target_function == "extract_text"
    assert h.suggested_filename == "fuzz_extract_text.py"
    assert "from augmentum.documents.chunker import extract_text" in h.source
    assert "extract_text(data)" in h.source
    assert "atheris.Setup(sys.argv, TestOneInput)" in h.source
    assert "atheris.Fuzz()" in h.source


def test_generate_harness_includes_expected_exception_list() -> None:
    h = generate_harness(
        _fuzzable("bytes"),
        target_file="augmentum/foo.py",
        target_function="parse",
    )
    for exc in _EXPECTED_PARSE_ERRORS:
        assert exc in h.source


def test_generate_harness_imports_modules_for_dotted_exceptions() -> None:
    """``struct.error`` / ``json.JSONDecodeError`` in the except tuple
    need their modules imported — without it the harness raises
    NameError before atheris gets a chance to feed it anything. The
    template derives the required imports from the exception list so
    extending ``_EXPECTED_PARSE_ERRORS`` Just Works."""
    h = generate_harness(
        _fuzzable("bytes"),
        target_file="augmentum/foo.py",
        target_function="parse",
    )
    expected_modules = {
        exc.split(".", 1)[0] for exc in _EXPECTED_PARSE_ERRORS if "." in exc
    }
    for mod in expected_modules:
        assert f"import {mod}\n" in h.source, (
            f"harness references {mod}.* in except but never imports {mod}"
        )


def test_generate_harness_compiles_with_module_globals() -> None:
    """Beyond ``ast.parse``, ``compile`` catches a wider class of
    output bugs (bad indentation, invalid f-string interpolation). We
    don't actually execute the harness — that would require atheris
    on the host — but we walk every name reference and confirm the
    stdlib modules in the except tuple are bound at module scope."""
    h = generate_harness(
        _fuzzable("bytes"),
        target_file="augmentum/foo.py",
        target_function="parse",
    )
    code = compile(h.source, "<harness>", "exec")
    # Module-scoped names declared by the harness. We expect each
    # dotted-exception's module to appear here.
    declared = set(code.co_names)
    for exc in _EXPECTED_PARSE_ERRORS:
        if "." in exc:
            mod = exc.split(".", 1)[0]
            assert mod in declared, (
                f"harness compiles but {mod} is not module-scope-declared"
            )


def test_generate_harness_bytearray_emits_wrap_prelude() -> None:
    h = generate_harness(
        _fuzzable("bytearray"),
        target_file="augmentum/foo.py",
        target_function="parse",
    )
    ast.parse(h.source)
    assert "payload = bytearray(data)" in h.source
    assert "parse(payload)" in h.source


def test_generate_harness_memoryview_emits_wrap_prelude() -> None:
    h = generate_harness(
        _fuzzable("memoryview"),
        target_file="augmentum/foo.py",
        target_function="parse",
    )
    ast.parse(h.source)
    assert "payload = memoryview(data)" in h.source
    assert "parse(payload)" in h.source


def test_generate_harness_inferred_uses_bytes_directly() -> None:
    h = generate_harness(
        _fuzzable("inferred-by-name", param="payload"),
        target_file="augmentum/foo.py",
        target_function="receive",
    )
    ast.parse(h.source)
    assert "receive(data)" in h.source


def test_generate_harness_strips_underscore_prefix_from_filename() -> None:
    """A target named ``_extract_pdf`` should yield ``fuzz_extract_pdf.py``,
    not ``fuzz__extract_pdf.py``. Filenames stay readable on disk."""
    h = generate_harness(
        _fuzzable("bytes"),
        target_file="augmentum/documents/chunker.py",
        target_function="_extract_pdf",
    )
    assert h.suggested_filename == "fuzz_extract_pdf.py"
    # But the actual import still uses the underscore-prefixed name
    assert "import _extract_pdf" in h.source


def test_generate_harness_with_workspace_root() -> None:
    h = generate_harness(
        _fuzzable("bytes"),
        target_file="/workspace/augmentum/documents/chunker.py",
        target_function="extract_text",
        workspace_root="/workspace",
    )
    assert h.target_module == "augmentum.documents.chunker"


# ---------------------------------------------------------------------------
# generate_harness — error paths
# ---------------------------------------------------------------------------


def test_generate_harness_refuses_non_fuzzable_verdict() -> None:
    v = FuzzVerdict(fuzzable=False, reason="generator function")
    with pytest.raises(ValueError, match="non-fuzzable"):
        generate_harness(
            v, target_file="x.py", target_function="f",
        )


def test_generate_harness_refuses_method_target() -> None:
    """Qualified ``ClassName.method`` is the step-2.5 LLM-driven path —
    v1 rejects with a clear pointer."""
    with pytest.raises(ValueError, match="method"):
        generate_harness(
            _fuzzable("bytes"),
            target_file="augmentum/foo.py",
            target_function="MyClass.parse",
        )


def test_generate_harness_refuses_unresolvable_module_path() -> None:
    """An empty file path can't be turned into a module — surface the
    misuse loudly rather than emitting a broken harness."""
    with pytest.raises(ValueError, match="module path"):
        generate_harness(
            _fuzzable("bytes"),
            target_file="",
            target_function="parse",
        )


# ---------------------------------------------------------------------------
# Harness dataclass shape
# ---------------------------------------------------------------------------


def test_harness_is_frozen_dataclass() -> None:
    h = Harness(
        source="...", suggested_filename="fuzz_x.py",
        target_module="m", target_function="f",
        target_param="data", input_kind="bytes",
    )
    try:
        h.source = "other"   # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("Harness should be frozen")
