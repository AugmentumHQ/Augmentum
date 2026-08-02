"""The model-backed reshape CLASSIFIER — natural language → a concrete, catalog-
validated surface change.

This turns the live Adaptation path from "client sends explicit {key,value}" into
"user says 'make it darker and denser' and it happens." It implements the
``engine.Classifier`` seam, so it drops straight into ``run_reshape_request``.

Discipline that keeps it honest and safe:
  - **Closed-world**: the model picks ONLY from the catalog (the surfaces'
    published adaptable fields), like the ``app.act`` utility pick — it cannot
    invent a key.
  - **Validated**: every proposed change is checked against the catalog
    (`catalog.validate`); an unknown surface/key or a disallowed enum value →
    returns None (honest *unmapped*), never a wild guess.
  - **Confidence-gated** (optional): a low-confidence pick → None, so the caller
    can ask for clarification instead of acting.
  - **Reversible-by-construction downstream**: even a correct-looking misread is
    instantly undoable (the config oracle confirms the *set*; the NL reading is a
    judgment, so the interpretation is surfaced in ``intent`` — "set theme=dark" —
    for the see-it/keep-it loop).

The model call (``invoke``) is INJECTED, so this module is pure and testable with
canned JSON — it calls no model itself.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable

from augmentum.selfedit.surfaces.base import ReshapeChange
from augmentum.selfedit.surfaces.catalog import (
    SurfaceSchema,
    registered_schemas,
    render_catalog,
    validate,
)
from augmentum.selfedit.surfaces.engine import ReshapeRequest
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Injected: a prompt → raw model text (JSON expected). Model-agnostic.
ModelInvoke = Callable[[str], Awaitable[str]]
# Injected: pick the schemas relevant to the live surfaces (defaults to the registry).
SchemasProvider = Callable[[list[str]], list[SurfaceSchema]]

_PROMPT = """You map a user's request to ONE configuration change, chosen ONLY \
from the catalog below. Do not invent keys or values.

Catalog:
{catalog}

User request: "{ask}"

Respond with JSON only:
  {{"surface": "...", "key": "...", "value": "...", "confidence": 0.0-1.0}}
or, if no catalog entry fits the request:
  {{"unmapped": true}}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of the model text; tolerant of prose/fences."""
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def build_model_classifier(invoke: ModelInvoke, *,
                           schemas_provider: SchemasProvider | None = None,
                           min_confidence: float = 0.0):
    """Build an ``engine.Classifier`` backed by ``invoke``. ``schemas_provider``
    selects the catalog for the live surfaces (defaults to the registered schemas
    filtered to the available surfaces)."""

    def _schemas_for(surfaces: list[str]) -> list[SurfaceSchema]:
        if schemas_provider is not None:
            return list(schemas_provider(surfaces))
        return [s for name, s in registered_schemas().items() if name in surfaces]

    async def classify(request: ReshapeRequest, surfaces: list[str]) -> ReshapeChange | None:
        schemas = _schemas_for(surfaces)
        if not schemas:
            log.info("reshape_classify_no_catalog", surfaces=surfaces)
            return None

        prompt = _PROMPT.format(catalog=render_catalog(schemas), ask=request.ask)
        try:
            raw = await invoke(prompt)
        except Exception as exc:  # noqa: BLE001 — a model error is an honest unmapped, not a crash
            log.warning("reshape_classify_invoke_error", error=repr(exc))
            return None

        data = _extract_json(raw)
        if not data or data.get("unmapped"):
            return None
        if float(data.get("confidence", 1.0) or 1.0) < min_confidence:
            log.info("reshape_classify_low_confidence", ask=request.ask)
            return None

        surface = str(data.get("surface", "")).strip()
        key = str(data.get("key", "")).strip()
        value = data.get("value", "")
        ok, canonical, reason = validate(schemas, surface, key, value)
        if not ok:
            log.info("reshape_classify_rejected", reason=reason)  # out-of-catalog → unmapped
            return None

        schema = next(s for s in schemas if s.surface == surface)
        return ReshapeChange(
            surface=surface, change_class=schema.change_class,
            payload={"key": key, "value": canonical},
            intent=f"set {key}={canonical}",  # surfaced for the see-it/keep-it loop
            actor=request.actor)

    return classify
