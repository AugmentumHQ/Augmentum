"""The reshape CATALOG — the closed world a surface will let itself be reshaped
into.

The model-backed classifier must NOT be free to write arbitrary config — that's
both unsafe and un-verifiable. Instead each surface publishes a schema: the exact
fields it will accept and (for enums) their allowed values. The classifier picks
from this catalog (closed-world, like the ``app.act`` utility pick), and a
proposed change is validated against it before it ever reaches ``reshape`` — an
out-of-catalog key/value is rejected as *unmapped*, not guessed.

So the catalog is two things at once: the grounding the classifier reasons over,
AND the safety allowlist that bounds what NL can touch. Schemas are registered by
the wiring layer (built from the real settings registry); this module owns the
shape, the render-for-prompt, and the validate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from augmentum.selfedit.surfaces.base import CLASS_ADAPTATION
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class AdaptationField:
    """One adaptable key on a surface. ``values`` empty = a free string field;
    non-empty = an enum (the only allowed values)."""

    key: str
    description: str
    values: tuple[str, ...] = ()

    def allows(self, value: object) -> tuple[bool, str]:
        """Return (ok, canonical_value). Enums match case-insensitively and
        normalize to the declared spelling; free fields accept any string."""
        s = str(value).strip()
        if not self.values:
            return (s != ""), s
        for v in self.values:
            if s.lower() == v.lower():
                return True, v
        return False, s


@dataclass
class SurfaceSchema:
    surface: str
    fields: list[AdaptationField] = field(default_factory=list)
    change_class: str = CLASS_ADAPTATION
    description: str = ""

    def get_field(self, key: str) -> AdaptationField | None:
        for f in self.fields:
            if f.key == key:
                return f
        return None


# --- registry (mirrors the surface registry) ------------------------------

_SCHEMAS: dict[str, SurfaceSchema] = {}


def register_schema(schema: SurfaceSchema) -> None:
    _SCHEMAS[schema.surface] = schema
    log.info("reshape_schema_registered", surface=schema.surface,
             fields=[f.key for f in schema.fields])


def get_schema(surface: str) -> SurfaceSchema | None:
    return _SCHEMAS.get(surface)


def registered_schemas() -> dict[str, SurfaceSchema]:
    return dict(_SCHEMAS)


def clear_schemas() -> None:  # for tests
    _SCHEMAS.clear()


def render_catalog(schemas: list[SurfaceSchema]) -> str:
    """Compact textual catalog for the classifier prompt."""
    lines: list[str] = []
    for s in schemas:
        head = f"- surface '{s.surface}'" + (f" — {s.description}" if s.description else "")
        lines.append(head)
        for f in s.fields:
            allowed = f" (one of: {', '.join(f.values)})" if f.values else " (free text)"
            lines.append(f"    • {f.key}: {f.description}{allowed}")
    return "\n".join(lines)


def validate(schemas: list[SurfaceSchema], surface: str, key: str,
             value: object) -> tuple[bool, str, str]:
    """Validate a proposed change against the catalog. Returns
    (ok, canonical_value, reason). The safety allowlist: unknown surface/key or a
    disallowed enum value → not ok."""
    schema = next((s for s in schemas if s.surface == surface), None)
    if schema is None:
        return False, "", f"surface '{surface}' not in catalog"
    fld = schema.get_field(key)
    if fld is None:
        return False, "", f"key '{key}' not adaptable on '{surface}'"
    ok, canonical = fld.allows(value)
    if not ok:
        return False, canonical, f"value {value!r} not allowed for '{key}'"
    return True, canonical, "ok"


def example_adaptation_schema() -> SurfaceSchema:
    """An EXAMPLE config schema for tests/bootstrap. The LIVE wiring should build
    the real schema from the settings registry — these keys are illustrative, not
    a claim about Augmentum's actual setting names."""
    return SurfaceSchema(
        surface="config", change_class=CLASS_ADAPTATION,
        description="per-user UI adaptation",
        fields=[
            AdaptationField("theme", "color theme", ("light", "dark", "system")),
            AdaptationField("density", "layout density", ("comfortable", "compact")),
            AdaptationField("accent", "accent color (free)"),
        ])
