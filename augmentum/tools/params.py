"""Schema-driven coercion of LLM-supplied tool parameters.

Models routinely send parameters that don't match the declared JSON
Schema: ``"42"`` for an integer, a nested ``{"parameters": {...}}``
wrapper, or — the case this module exists for — a JSON *array* for a
parameter declared ``"type": "string"``.

That last one is not a malformed call. When a model emits::

    web_search(query=["mars sample return 2026", "esa budget 2026"])

it is expressing a real, coherent intent: *run two searches*. The tool
signature simply has no way to say that. Historically this reached
``execute(query=[...])`` and died on ``query.strip()`` with
``AttributeError: 'list' object has no attribute 'strip'`` — a Python
internal that told the model nothing, and (worse) was scored as "the
search returned nothing", which pushed the model into answering from
memory and doubting well-sourced material.

So the policy here is **fan-out, not rejection**: a list for a string
param is split into one call per element. The model's intent is
preserved exactly, no query is dropped, and no round-trip is wasted
teaching it a calling convention.

This module deliberately lives under ``tools/`` rather than ``modes/``
so the ``Tool`` base class can use it without importing a mode package
(a layering inversion). ``modes.analytical.tool_calling.coerce_tool_params``
delegates here for backwards compatibility.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.tools.base import Tool

log = get_logger(__name__)

# How many strings one fanned-out parameter may become. 1-4 is the
# accepted band: wide enough to absorb the phrasing differences between
# model families (some batch 2, some 3-4), narrow enough that a runaway
# can't hammer SearXNG or blow the context budget.
#
# Past 4 we ERROR rather than truncate. Silently running the first 4 of
# 9 queries would return a confident-looking answer built on a subset
# the model never agreed to — the failure would be invisible. An error
# the model can read and act on is strictly better than a quiet
# half-answer.
MAX_FANOUT = 4

# Key under which ``modes.analytical.tool_calling._parse_python_args`` records
# the parameter names it GUESSED rather than read off the model's output.
# Text-tier models emit positional calls — ``calculator("2 + 2")`` — and that
# parser has only the tool NAME, no schema, so it labels every positional
# ``"query"``. The marker lets :func:`_bind_stray_positional` rebind that guess
# onto the real parameter without also "fixing" genuine model typos, which
# would fabricate values. Stripped before the tool is ever called.
POSITIONAL_GUESS_KEY = "__positional_guess__"


def _string_props(schema: dict) -> set[str]:
    props = schema.get("properties") or {}
    return {
        key
        for key, prop in props.items()
        if isinstance(prop, dict) and prop.get("type") == "string"
    }


def unwrap_argument_envelope(params: dict, properties: dict) -> dict:
    """Unwrap ``{"parameters": {...}}`` / ``{"arguments": {...}}`` envelopes.

    Common with models trained against other function-calling dialects.
    Only unwraps when the inner dict matches the schema *better* than the
    outer one, so a tool that legitimately takes a ``parameters`` field is
    left alone.
    """
    for wrapper in ("parameters", "arguments"):
        inner = params.get(wrapper)
        if not isinstance(inner, dict):
            continue
        inner_hits = sum(1 for k in inner if k in properties)
        outer_hits = sum(1 for k in params if k in properties and k != wrapper)
        if inner_hits > outer_hits:
            log.info("tool_params_unwrapped", wrapper=wrapper)
            return dict(inner)
    return params


def coerce_params(
    tool: Tool, params: dict, *, stripped_out: list | None = None,
) -> dict:
    """Coerce ``params`` toward ``tool.input_schema``.

    Handles envelope unwrapping, unknown-key stripping and scalar type
    coercion. Does NOT fan out list-valued string params — that changes
    the number of calls, so it is a separate, explicit step
    (:func:`split_fanout`) the caller opts into.

    ``stripped_out``: when a list is passed, every model-supplied key
    removed as "unknown to this tool" is appended to it. ``Tool.invoke``
    surfaces those names back to the model rather than letting the drop
    stay silent — a dropped ``cwd`` or case-insensitive flag otherwise
    changes the result with zero signal to the caller.
    """
    schema = getattr(tool, "input_schema", None) or {}
    properties = schema.get("properties") or {}

    # Pop the parser's marker unconditionally, BEFORE the no-properties
    # early return. It is internal bookkeeping and starts with "_", so the
    # unknown-key strip below deliberately skips it — left in place it would
    # ride all the way into ``execute(**params)`` and raise TypeError on a
    # schema-less tool.
    guessed = params.pop(POSITIONAL_GUESS_KEY, None)

    if not properties:
        return params

    params = unwrap_argument_envelope(params, properties)

    _bind_stray_positional(tool, params, schema, properties, guessed)

    # Strip unknown keys — prevents TypeError on execute(**params).
    # Internal context keys (_context, _user_id, …) are injected by the
    # dispatch layer AFTER coercion, but strip defensively in case a
    # caller pre-injected them: they are not schema properties and the
    # signature-aware injector re-adds only what the tool accepts.
    for key in [k for k in params if k not in properties and not k.startswith("_")]:
        log.debug("tool_param_stripped", tool=tool.name, param=key)
        if stripped_out is not None:
            stripped_out.append(key)
        del params[key]

    for key, prop in properties.items():
        if key not in params or not isinstance(prop, dict):
            continue
        expected = prop.get("type")
        value = params[key]

        # A single-element list for ANY scalar type is unambiguous —
        # unwrap it here so it never reaches fan-out or execute().
        if isinstance(value, list) and len(value) == 1 and expected in (
            "string", "integer", "number", "boolean",
        ):
            value = params[key] = value[0]

        if expected == "integer" and isinstance(value, str):
            with contextlib.suppress(ValueError):
                params[key] = int(value)
        elif expected == "number" and isinstance(value, str):
            with contextlib.suppress(ValueError):
                params[key] = float(value)
        elif expected == "boolean" and isinstance(value, str):
            params[key] = value.lower() in ("true", "1", "yes")
        elif expected == "string" and isinstance(value, int | float | bool):
            # Mirror image of the str->int case: a model sending a bare
            # number for a string param. Cheap and lossless.
            params[key] = str(value)

    return params


def _bind_stray_positional(
    tool: Tool, params: dict, schema: dict, properties: dict, guessed: Any,
) -> None:
    """Rebind a parser-GUESSED arg name onto the one missing required param.

    Text-tier models emit positional calls — ``calculator("2 + 2")`` — and
    ``_parse_python_args`` has only the tool NAME, no schema, so it labels
    every positional ``"query"`` (the common case, wrong for everything
    else). The stray key was then stripped as unknown and the tool failed
    with "requires: expression" while the value sat right there.

    Only keys the parser flagged via ``POSITIONAL_GUESS_KEY`` are eligible.
    A model's own misspelling is NOT: rebinding ``web_search(catgory="news")``
    would silently run a search for the literal string "news" — a query the
    model never asked for, from a value that meant a category. Typos have to
    fail loudly so the model can read the schema hint and retry, and this
    guard is what keeps a category from becoming a search term. That matters
    across the model spread we support: the small text-tier models that need
    the positional rescue are the same ones that misspell parameter names,
    while native function-calling models never hit this path at all.

    Beyond the marker, the binding must also be unambiguous: exactly one
    required property missing and exactly one eligible scalar supplied. Two
    missing params is a model that genuinely didn't answer, not a naming
    slip — guessing there would invent data.
    """
    guessed_keys = {k for k in guessed if isinstance(k, str)} if isinstance(guessed, list) else set()
    if not guessed_keys:
        return
    required = [r for r in (schema.get("required") or []) if isinstance(r, str)]
    missing = [r for r in required if r not in params]
    if len(missing) != 1:
        return
    strays = [
        k for k, v in params.items()
        if k not in properties
        and k in guessed_keys
        and isinstance(v, str | int | float | bool)
    ]
    if len(strays) != 1:
        return
    params[missing[0]] = params.pop(strays[0])
    log.debug(
        "tool_param_rebound", tool=tool.name, frm=strays[0], to=missing[0],
    )


def split_fanout(tool: Tool, params: dict) -> tuple[str, list[Any], str]:
    """Detect a list supplied for a string-typed param.

    Returns ``(param_name, values, error)``. ``param_name`` is empty
    when there is nothing to fan out. ``error`` is a model-readable
    message when the list is too long to honor — in that case the call
    must NOT run, because a partial answer would look complete.

    Only ONE parameter is ever fanned out — the first string-typed one
    carrying a multi-element list. Fanning out two independently would
    imply a cross product, which no model means and which multiplies
    cost quadratically; the second list is left for ``coerce_params``
    to stringify so the call still runs.
    """
    schema = getattr(tool, "input_schema", None) or {}
    if not schema:
        return "", [], ""
    for key in _string_props(schema):
        value = params.get(key)
        if not isinstance(value, list) or len(value) <= 1:
            continue
        # Keep only usable scalars; a nested object here is not an
        # intent we can honor.
        values = [v for v in value if isinstance(v, str | int | float) and str(v).strip()]
        if len(values) <= 1:
            return "", [], ""
        if len(values) > MAX_FANOUT:
            return key, [], (
                f"'{key}' accepts 1 to {MAX_FANOUT} values per call, but "
                f"{len(values)} were supplied. Nothing was run — split this "
                f"into separate {tool.name} calls of at most {MAX_FANOUT} "
                f"each, so no value is silently skipped."
            )
        return key, values, ""
    return "", [], ""
