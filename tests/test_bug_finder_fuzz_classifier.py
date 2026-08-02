"""Fuzz chunk classifier tests.

The classifier is pure AST + heuristics — no I/O — so the tests exercise
it with literal source strings. Coverage hits each disqualifying axis
(decorators, async, generator, test-file, missing positional arg,
non-bytes typing) plus the positive paths (typed bytes/str/Optional,
inferred-by-name).
"""

from __future__ import annotations

from augmentum.bug_finder.fuzz.classifier import (
    FuzzVerdict,
    classify_chunk,
    classify_function,
)

# ---------------------------------------------------------------------------
# Positive — first-arg accepts bytes-shaped input
# ---------------------------------------------------------------------------


def test_bytes_typed_param_is_fuzzable() -> None:
    src = (
        "def parse_record(data: bytes) -> dict:\n"
        "    return {'len': len(data)}\n"
    )
    v = classify_chunk(src, "parse_record")
    assert v.fuzzable is True
    assert v.target_param == "data"
    assert v.input_kind == "bytes"
    assert v.reason == ""


def test_bytearray_typed_param_is_fuzzable() -> None:
    src = "def decode(buf: bytearray) -> int:\n    return len(buf)\n"
    v = classify_chunk(src, "decode")
    assert v.fuzzable and v.input_kind == "bytearray"


def test_memoryview_typed_param_is_fuzzable() -> None:
    src = "def consume(view: memoryview) -> int:\n    return len(view)\n"
    v = classify_chunk(src, "consume")
    assert v.fuzzable and v.input_kind == "memoryview"


def test_str_typed_param_is_not_fuzzable_by_typing() -> None:
    """``str`` alone is too broad — 18% of Augmentum's own functions take
    a single ``str`` arg, and almost none of them are parsing/decoding
    targets that benefit from random-bytes input. A ``str``-shaped parse
    target lands via the parameter-name heuristic instead (see the
    inferred-by-name tests). Removing ``str`` from the typed-fuzzable
    set was the single biggest precision win in the first stress test."""
    src = "def parse_header(text: str) -> dict:\n    return {}\n"
    v = classify_chunk(src, "parse_header")
    assert not v.fuzzable
    assert "'str'" in v.reason


def test_optional_bytes_unwrapped_to_fuzzable() -> None:
    src = (
        "from typing import Optional\n"
        "def parse(data: Optional[bytes]) -> dict:\n"
        "    return {}\n"
    )
    v = classify_chunk(src, "parse")
    assert v.fuzzable and v.input_kind == "bytes"


def test_union_bytes_or_none_unwrapped() -> None:
    src = (
        "def parse(data: bytes | None) -> dict:\n"
        "    return {}\n"
    )
    v = classify_chunk(src, "parse")
    assert v.fuzzable and v.input_kind == "bytes"


def test_inferred_by_name_data() -> None:
    src = "def handle(data):\n    return len(data)\n"
    v = classify_chunk(src, "handle")
    assert v.fuzzable
    assert v.target_param == "data"
    assert v.input_kind == "inferred-by-name"


def test_inferred_by_name_payload() -> None:
    src = "def receive(payload):\n    return payload[:4]\n"
    v = classify_chunk(src, "receive")
    assert v.fuzzable and v.target_param == "payload"


def test_method_is_rejected_with_is_method_flag() -> None:
    """v1 harness writer can't synthesize instances for arbitrary classes,
    so methods are rejected. ``is_method=True`` lets callers separate
    this bucket from other rejections in their telemetry."""
    src = (
        "class Decoder:\n"
        "    def decode(self, data: bytes) -> int:\n"
        "        return len(data)\n"
    )
    v = classify_chunk(src, "Decoder.decode")
    assert not v.fuzzable
    assert v.is_method is True
    assert "method" in v.reason


def test_classmethod_is_rejected_as_method() -> None:
    src = (
        "class Decoder:\n"
        "    @classmethod\n"
        "    def from_bytes(cls, data: bytes) -> 'Decoder':\n"
        "        return cls()\n"
    )
    v = classify_chunk(src, "from_bytes")
    assert not v.fuzzable
    assert v.is_method is True


def test_classify_function_self_skip_at_direct_entry_point() -> None:
    """Even when called directly with ``inside_class=False`` (a misuse,
    but the contract should still degrade gracefully), the positional
    arg walker should skip ``self`` and consider the next arg. This
    documents the unit-level behaviour separately from the method
    rejection that ``classify_chunk`` performs at scope-walk time."""
    import ast
    src = (
        "def decode(self, data: bytes) -> int:\n"
        "    return len(data)\n"
    )
    tree = ast.parse(src)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    v = classify_function(func, inside_class=False)
    assert v.fuzzable
    assert v.target_param == "data"


# ---------------------------------------------------------------------------
# Negative — disqualifying signatures
# ---------------------------------------------------------------------------


def test_no_positional_params_not_fuzzable() -> None:
    src = "def ping():\n    return 'pong'\n"
    v = classify_chunk(src, "ping")
    assert not v.fuzzable
    assert "no positional parameters" in v.reason


def test_first_param_int_not_fuzzable() -> None:
    src = "def increment(n: int) -> int:\n    return n + 1\n"
    v = classify_chunk(src, "increment")
    assert not v.fuzzable
    assert "'int'" in v.reason


def test_unnamed_unannotated_first_param_not_fuzzable() -> None:
    """Unannotated + unknown param name → false (avoid burning fuzz time)."""
    src = "def helper(x):\n    return x * 2\n"
    v = classify_chunk(src, "helper")
    assert not v.fuzzable
    assert "no type hint" in v.reason


def test_union_with_two_non_none_members_not_fuzzable() -> None:
    """Union[bytes, int] is ambiguous — bail rather than mis-fuzz."""
    src = (
        "def parse(data: bytes | int) -> int:\n"
        "    return 0\n"
    )
    v = classify_chunk(src, "parse")
    assert not v.fuzzable


def test_generator_not_fuzzable() -> None:
    src = (
        "def stream(data: bytes):\n"
        "    for byte in data:\n"
        "        yield byte\n"
    )
    v = classify_chunk(src, "stream")
    assert not v.fuzzable
    assert "generator" in v.reason


def test_async_function_not_fuzzable() -> None:
    src = "async def fetch(data: bytes) -> bytes:\n    return data\n"
    v = classify_chunk(src, "fetch")
    assert not v.fuzzable
    assert "async" in v.reason


# ---------------------------------------------------------------------------
# Negative — disqualifying context (decorators, test files)
# ---------------------------------------------------------------------------


def test_test_function_name_not_fuzzable() -> None:
    src = "def test_my_thing(data: bytes):\n    assert data\n"
    v = classify_chunk(src, "test_my_thing")
    assert not v.fuzzable
    assert "test function" in v.reason


def test_test_file_path_not_fuzzable() -> None:
    src = "def helper(data: bytes):\n    return data\n"
    v = classify_chunk(src, "helper", file_path="tests/test_thing.py")
    assert not v.fuzzable
    assert "test file" in v.reason


def test_conftest_file_path_not_fuzzable() -> None:
    src = "def some_fixture_factory(data: bytes):\n    return data\n"
    v = classify_chunk(src, "some_fixture_factory", file_path="conftest.py")
    assert not v.fuzzable


def test_route_decorator_disqualifies() -> None:
    src = (
        "@app.route('/echo', methods=['POST'])\n"
        "def echo(data: bytes):\n"
        "    return data\n"
    )
    v = classify_chunk(src, "echo")
    assert not v.fuzzable
    assert "@app.route" in v.reason


def test_fastapi_get_decorator_disqualifies() -> None:
    src = (
        "@router.get('/items')\n"
        "def list_items(data: bytes):\n"
        "    return []\n"
    )
    v = classify_chunk(src, "list_items")
    assert not v.fuzzable


def test_websocket_decorator_disqualifies() -> None:
    src = (
        "@router.websocket('/ws')\n"
        "def ws_handler(data: bytes):\n"
        "    return data\n"
    )
    v = classify_chunk(src, "ws_handler")
    assert not v.fuzzable


def test_pytest_fixture_decorator_disqualifies() -> None:
    src = (
        "@pytest.fixture\n"
        "def make_payload(data: bytes):\n"
        "    return data\n"
    )
    v = classify_chunk(src, "make_payload")
    assert not v.fuzzable


def test_click_command_decorator_disqualifies() -> None:
    src = (
        "@click.command()\n"
        "def cli(data: bytes):\n"
        "    return data\n"
    )
    v = classify_chunk(src, "cli")
    assert not v.fuzzable


def test_hypothesis_given_decorator_disqualifies() -> None:
    """Already a fuzzer — don't double-fuzz."""
    src = (
        "@hypothesis.given(data=strategies.binary())\n"
        "def prop_test(data: bytes):\n"
        "    assert isinstance(data, bytes)\n"
    )
    v = classify_chunk(src, "prop_test")
    assert not v.fuzzable


# ---------------------------------------------------------------------------
# Robustness — bad input
# ---------------------------------------------------------------------------


def test_function_not_found_returns_verdict() -> None:
    src = "def other(data: bytes): return data\n"
    v = classify_chunk(src, "missing")
    assert not v.fuzzable
    assert "not found" in v.reason


def test_syntax_error_does_not_crash() -> None:
    src = "def broken(data: bytes\n    return data\n"
    v = classify_chunk(src, "broken")
    assert not v.fuzzable
    assert "did not parse" in v.reason


def test_free_function_with_same_name_as_method_is_still_fuzzable() -> None:
    """When a module has both ``parse`` (free fn) and ``X.parse``
    (method), the unqualified target ``parse`` should resolve to the
    free function and stay fuzzable. Tests that scope tracking doesn't
    accidentally over-match into class bodies."""
    src = (
        "def parse(data: bytes) -> int:\n"
        "    return len(data)\n"
        "\n"
        "class Foo:\n"
        "    def parse(self, data: bytes) -> int:\n"
        "        return len(data)\n"
    )
    v = classify_chunk(src, "parse")
    assert v.fuzzable
    assert v.is_method is False


def test_qualified_function_name_finds_method_then_rejects() -> None:
    """``MyClass.method`` should find the method (and reject it as a
    method for v1) — not fall through to "not found"."""
    src = (
        "class MyClass:\n"
        "    def parse(self, data: bytes) -> int:\n"
        "        return len(data)\n"
    )
    v = classify_chunk(src, "MyClass.parse")
    assert not v.fuzzable
    assert v.is_method is True
    assert "not found" not in v.reason


# ---------------------------------------------------------------------------
# FuzzVerdict — bool / dataclass semantics
# ---------------------------------------------------------------------------


def test_verdict_is_truthy_when_fuzzable() -> None:
    yes = FuzzVerdict(True, target_param="data", input_kind="bytes")
    no = FuzzVerdict(False, reason="x")
    assert bool(yes) is True
    assert bool(no) is False
    # Direct use in a conditional
    if yes:
        pass
    else:
        raise AssertionError("expected truthy")


def test_verdict_is_frozen() -> None:
    v = FuzzVerdict(False, reason="x")
    try:
        v.reason = "y"   # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("FuzzVerdict should be frozen")


# ---------------------------------------------------------------------------
# classify_function — direct AST entry point
# ---------------------------------------------------------------------------


def test_classify_function_direct_entry_point() -> None:
    import ast
    src = "def parse(data: bytes) -> int:\n    return len(data)\n"
    tree = ast.parse(src)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    v = classify_function(func)
    assert v.fuzzable
