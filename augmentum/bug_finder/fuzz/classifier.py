"""Decide whether a chunk is a Python fuzzing target.

Pure-AST pass with no I/O. The classifier is the cheapest first commit in
the fuzzer-integration spec because the answer is "no" for most chunks —
a route handler, a test fixture, a generator, an init method. Routing
those through the fuzzer would waste a harness-writer LLM call and a
multi-minute fuzz session for nothing.

A chunk is fuzzable when ALL of these hold:

1. The function exists, parses, and isn't an async / generator function.
   (Atheris drives sync callables that return; generators yield without
   ever returning, so a crash never lands on a stack we can capture.)

2. The function is not framework infrastructure — no ``@app.route`` /
   ``@router.X`` / ``@pytest.fixture`` / ``@click.command`` decorators,
   and not defined in a test file.

3. The first positional parameter (after skipping ``self`` / ``cls`` on
   methods) is annotated as ``bytes`` / ``bytearray`` / ``memoryview`` /
   ``str`` — or, if unannotated, named like a known bytes-input
   parameter (``data``, ``buf``, ``payload``, …). When ambiguous, the
   classifier returns ``False`` — the cost of a missed fuzz target is
   much smaller than the cost of repeatedly burning fuzz budget on
   non-targets.

The classifier is deliberately strict on first-arg shape. Spec'd
deepening (typed bytes anywhere in the signature, multi-param fuzzing
via structured input) is a follow-up; v1 keeps the harness writer's
prompt simple by guaranteeing it sees one bytes-shaped target param.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


# Parameter names that suggest a bytes-shaped input when no type hint is
# present. Conservative on purpose — adding a name here is cheap, but
# every false-positive entry routes a function through the fuzz leg and
# back via "fuzzer couldn't construct any seed". Names taken from a
# scan of common Python parsing/decoding APIs (stdlib + popular libs).
_BYTES_PARAM_NAMES = frozenset({
    "data", "buf", "buffer", "raw", "payload", "content",
    "bytes_data", "input_data", "input_bytes", "blob", "body",
    "src", "source", "stream", "octets",
})

# Type-annotation source strings that mark a parameter as fuzzable. The
# annotation is rendered back to text via ``ast.unparse`` so we can
# compare structurally without an import-resolution step.
#
# ``str`` is deliberately NOT here. A first stress-pass over Augmentum
# itself flagged 18% of functions fuzzable when ``str`` was included —
# mostly utility functions (slugify / get_by_id / format_iso) where
# every input is a string and "feed it random bytes" yields no
# interesting crashes. The conservative position: bytes-typed input
# usually means parser/decoder. ``str``-shaped parsing targets get
# picked up via the parameter-name heuristic instead, which requires
# names like ``payload`` / ``content`` to land.
_FUZZABLE_TYPE_HINTS = frozenset({
    "bytes", "bytearray", "memoryview",
})

# Decorator names (exact match on the dotted source form) that disqualify
# a function. These mark framework entry points / test scaffolding /
# CLI plumbing — none of which are unit-fuzz targets.
_FRAMEWORK_DECORATORS = frozenset({
    # Pytest / unittest
    "pytest.fixture", "pytest.mark.parametrize", "pytest.mark.asyncio",
    # Click / typer
    "click.command", "click.group", "typer.command",
    # Hypothesis is a fuzzer in its own right — don't double-fuzz
    "hypothesis.given", "given",
})

# Decorator suffixes that mark a framework route registration. We match
# on the *last* attribute name so ``app.route``, ``api_router.post``,
# ``self.router.websocket`` all rank as disqualifying.
_ROUTE_DECORATOR_SUFFIXES = frozenset({
    "route", "get", "post", "put", "delete", "patch",
    "head", "options", "websocket",
    "fixture",  # @pytest.fixture as ".fixture" suffix
    "command",  # @click.command, @typer.command
})


@dataclass(frozen=True)
class FuzzVerdict:
    """The classifier's answer about one chunk.

    ``fuzzable`` is the only field the caller branches on. The other
    fields exist so the downstream harness writer (Phase 1 step 2) and
    the per-run logging have enough structure to skip a second AST pass.

    ``is_method`` is informational — even when ``fuzzable=False``,
    callers can break out method-rejections from other rejection
    reasons in their telemetry. v1's deterministic harness writer
    handles module-level functions only; instance-method support
    requires LLM-driven ``__init__`` synthesis, which lands in
    a follow-up step.
    """

    fuzzable: bool
    reason: str = ""            # populated when fuzzable=False
    target_param: str = ""      # parameter name that takes fuzz input
    input_kind: str = ""        # "bytes" / "bytearray" / "memoryview" /
                                # "inferred-by-name"
    is_method: bool = False     # function is defined inside a class

    def __bool__(self) -> bool:
        return self.fuzzable


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _decorator_name(node: ast.expr) -> str:
    """Render a decorator expression as dotted source text.

    ``@route`` → ``"route"``,
    ``@app.route("/x")`` → ``"app.route"``,
    ``@functools.lru_cache()`` → ``"functools.lru_cache"``.
    Returns ``""`` for shapes we can't easily render (lambdas, walrus, …).
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _decorator_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def _annotation_source(node: ast.expr | None) -> str:
    """Render a type-annotation AST as the source string it came from."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 — be liberal: any unparseable hint
        return ""      # is treated as "no useful annotation".


def _strip_optional(annotation: str) -> str:
    """Reduce ``Optional[X]`` / ``X | None`` / ``None | X`` to ``X``.

    Leaves ``Union[X, Y]`` (with two non-None members) alone — that
    isn't a single shape and shouldn't be classified as fuzzable.
    """
    bare = annotation.strip()
    for prefix in ("Optional[", "typing.Optional["):
        if bare.startswith(prefix) and bare.endswith("]"):
            bare = bare[len(prefix):-1].strip()
            break
    if "|" in bare:
        members = [p.strip() for p in bare.split("|")]
        non_none = [m for m in members if m and m != "None"]
        if len(non_none) == 1:
            bare = non_none[0]
    return bare


def _contains_yield(func: ast.AST) -> bool:
    """True when the function body contains a yield / yield-from anywhere.

    Walks the whole body so nested ``yield`` inside try/except still
    counts. A nested function literal that yields doesn't make the
    enclosing function a generator — but it's rare enough that the
    simpler walk is fine; classifier errs on the side of "not fuzzable".
    """
    for child in ast.walk(func):
        if isinstance(child, (ast.Yield, ast.YieldFrom)):
            return True
    return False


def _is_test_path(file_path: str) -> bool:
    """``tests/x.py`` / ``foo_test.py`` / ``conftest.py`` count as tests."""
    if not file_path:
        return False
    p = file_path.replace("\\", "/")
    parts = p.split("/")
    if any(part == "tests" or part == "test" for part in parts):
        return True
    leaf = parts[-1]
    return (
        leaf.startswith("test_")
        or leaf.endswith("_test.py")
        or leaf == "conftest.py"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_function(
    func: ast.AsyncFunctionDef | ast.FunctionDef,
    *,
    file_path: str = "",
    inside_class: bool = False,
) -> FuzzVerdict:
    """Decide whether a single function AST node is a Python fuzz target.

    Caller-supplied ``file_path`` is used only for the test-file check;
    pass ``""`` if the AST didn't come from a file.

    ``inside_class`` should be True when ``func`` is a class method.
    v1's deterministic harness writer cannot synthesize an instance for
    arbitrary classes, so methods are rejected. The verdict's
    ``is_method`` flag is preserved so callers can separate this reject
    bucket from the others. ``classify_chunk`` tracks scope and passes
    this automatically.
    """
    if inside_class:
        return FuzzVerdict(
            False,
            reason=(
                "method (v1 harness writer requires module-level functions; "
                "instance synthesis pending LLM-driven step 2.5)"
            ),
            is_method=True,
        )

    name = func.name
    if name.startswith("test_"):
        return FuzzVerdict(False, reason="test function name")
    if _is_test_path(file_path):
        return FuzzVerdict(False, reason="defined in a test file")

    for dec in func.decorator_list:
        dname = _decorator_name(dec)
        if not dname:
            continue
        if dname in _FRAMEWORK_DECORATORS:
            return FuzzVerdict(False, reason=f"decorated with @{dname}")
        suffix = dname.rsplit(".", 1)[-1]
        if suffix in _ROUTE_DECORATOR_SUFFIXES:
            return FuzzVerdict(False, reason=f"decorated with @{dname}")

    if isinstance(func, ast.AsyncFunctionDef):
        # Atheris doesn't drive coroutines directly. A future enhancement
        # could wrap with ``asyncio.run`` in the harness, but v1 skips
        # to keep the harness writer's prompt simple.
        return FuzzVerdict(False, reason="async function")

    if _contains_yield(func):
        return FuzzVerdict(False, reason="generator function")

    args = func.args
    pos_args = list(args.args)
    if pos_args and pos_args[0].arg in ("self", "cls"):
        pos_args = pos_args[1:]

    if not pos_args:
        return FuzzVerdict(False, reason="no positional parameters to fuzz")

    first = pos_args[0]
    annotation = _annotation_source(first.annotation)
    bare = _strip_optional(annotation) if annotation else ""

    if bare in _FUZZABLE_TYPE_HINTS:
        return FuzzVerdict(
            fuzzable=True,
            target_param=first.arg,
            input_kind=bare,
        )

    if not annotation and first.arg in _BYTES_PARAM_NAMES:
        return FuzzVerdict(
            fuzzable=True,
            target_param=first.arg,
            input_kind="inferred-by-name",
        )

    if annotation:
        return FuzzVerdict(
            False,
            reason=f"first parameter typed as {annotation!r}, not bytes/str-like",
        )
    return FuzzVerdict(
        False,
        reason=(
            f"first parameter {first.arg!r} has no type hint and is not a "
            "known bytes-input name"
        ),
    )


def classify_chunk(
    source: str,
    function: str,
    *,
    file_path: str = "",
) -> FuzzVerdict:
    """Parse ``source``, find ``function`` by name, classify it.

    ``function`` may be qualified (``ClassName.method``) — the classifier
    splits on the last dot and matches the leaf name. When the qualified
    form names a class, only methods inside *that* class match. The walk
    also tracks class scope so methods are correctly tagged
    ``is_method=True`` and rejected for v1.

    Multiple matches at the same scope return the first; tighter
    targeting via line range is a follow-up once the planner emits
    stable chunk anchors.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return FuzzVerdict(False, reason=f"source did not parse: {exc.msg}")

    target_leaf = function.rsplit(".", 1)[-1]
    target_class = function.rsplit(".", 1)[0] if "." in function else None

    # Recursive descent with class-scope tracking. ``in_class`` is the
    # enclosing class name (None at module scope). We prefer a match in
    # the explicitly-named class when the caller qualified the target;
    # otherwise the first match wins.
    matched: tuple[ast.AsyncFunctionDef | ast.FunctionDef, str | None] | None = None

    def visit(node: ast.AST, in_class: str | None) -> None:
        nonlocal matched
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
            elif isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                if child.name == target_leaf:
                    if matched is None:
                        matched = (child, in_class)
                    elif target_class and in_class == target_class:
                        # Explicit class qualifier — prefer the match
                        # inside that class even if an unrelated one
                        # was seen first.
                        matched = (child, in_class)
                visit(child, in_class)
            else:
                visit(child, in_class)

    visit(tree, None)

    if matched is None:
        return FuzzVerdict(
            False, reason=f"function {function!r} not found in source",
        )
    node, in_class = matched
    return classify_function(
        node, file_path=file_path, inside_class=in_class is not None,
    )
