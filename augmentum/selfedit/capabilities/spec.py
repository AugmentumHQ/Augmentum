"""CapabilitySpec — the validated, closed-world description of a verb to author.

The LLM produces one of these (as JSON) from a natural-language request; nothing
downstream trusts it until ``validate_spec`` passes. The spec is intentionally
NARROW: it names a behavior from a fixed safe palette and fills data slots. It
cannot describe arbitrary handler logic — that's the whole safety story.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# id must be ``surface.action`` — lowercase, matches the registry convention
# (navigate.open_surface, weather.today, note.create, …).
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")

# The safe behavior palette. A synthesized verb's handler is RENDERED from one
# of these — never authored by the model. Extend deliberately.
#   surface_emit — short-circuit to a frontend WS action (open a surface, run a
#                  palette command). Pure UI signal; the safest "new capability."
#   speak        — a canned spoken/typed line (e.g. a verb that states something).
BEHAVIORS = frozenset({"surface_emit", "speak"})

# Synthesized verbs are capped to genuinely-reversible stakes. Anything that
# sends/posts/pays/touches-personal-data must be human-authored — the synthesizer
# refuses to mint it. (Full registry set is wider; this is the *synthesis* subset.)
SAFE_STAKES = frozenset({"trivial_reversible", "disruptive"})

KNOWN_SURFACES = frozenset({"voice", "chat", "cast", "xr", "becca"})
ARG_TYPES = frozenset({"string", "integer", "number", "boolean"})

_DEFAULT_SURFACES = ("voice", "chat", "becca")


@dataclass
class CapabilitySpec:
    """A verb to author. ``behavior`` selects the rendered handler; the rest are
    data slots the renderer substitutes as literals (never as code)."""

    # Defaulted so from_dict never crashes on a partial/garbage model response —
    # validate_spec is the gate, not the constructor.
    id: str = ""
    summary: str = ""                             # tier-3 tool description
    examples: list[str] = field(default_factory=list)  # trigger phrasings (seed tier-1 patterns)
    behavior: str = ""                            # one of BEHAVIORS

    # behavior=surface_emit
    channel: str = ""                             # WS channel, e.g. "navigate.open_surface"
    payload: dict[str, Any] = field(default_factory=dict)  # static payload (merged with args)
    toast: str = ""                               # small non-spoken confirmation chip

    # behavior=speak (also an optional spoken line for surface_emit)
    speak: str = ""

    arg_schema: dict[str, Any] = field(default_factory=dict)   # name -> {type, description}
    required: list[str] = field(default_factory=list)
    surfaces: list[str] = field(default_factory=lambda: list(_DEFAULT_SURFACES))
    stakes: str = "trivial_reversible"

    tier1: bool = True
    tier2: bool = True
    tier3: bool = True

    @property
    def module_name(self) -> str:
        """builtin module file stem — the surface prefix of the id."""
        return self.id.split(".", 1)[0]

    @property
    def func_name(self) -> str:
        """Generated handler function name, unique per verb."""
        return "_" + self.id.replace(".", "_")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CapabilitySpec:
        d = d or {}
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in d.items() if k in known})


def validate_spec(spec: CapabilitySpec) -> list[str]:
    """Return a list of human-readable problems (empty = valid). Closed-world:
    every field is checked against the registry contract + the safe palette."""
    errs: list[str] = []

    if not spec.id or not _ID_RE.match(spec.id):
        errs.append(f"id {spec.id!r} must match surface.action (lowercase, e.g. notes.pin)")
    if not (spec.summary or "").strip():
        errs.append("summary is required (it becomes the tool description)")
    if not spec.examples or not any((e or "").strip() for e in spec.examples):
        errs.append("at least one example phrasing is required (seeds matching)")
    if spec.behavior not in BEHAVIORS:
        errs.append(f"behavior {spec.behavior!r} not in safe palette {sorted(BEHAVIORS)}")
    if spec.stakes not in SAFE_STAKES:
        errs.append(
            f"stakes {spec.stakes!r} not synthesizable — only {sorted(SAFE_STAKES)}; "
            "higher-stakes verbs (send/post/pay/personal) must be human-authored"
        )

    # behavior-specific required slots
    if spec.behavior == "surface_emit":
        if not (spec.channel or "").strip():
            errs.append("surface_emit requires a non-empty channel")
        if not isinstance(spec.payload, dict):
            errs.append("surface_emit payload must be an object")
    if spec.behavior == "speak" and not (spec.speak or "").strip():
        errs.append("speak behavior requires a non-empty speak line")

    # arg schema shape
    if not isinstance(spec.arg_schema, dict):
        errs.append("arg_schema must be an object")
    else:
        for name, meta in spec.arg_schema.items():
            if not re.match(r"^[a-z][a-z0-9_]*$", str(name)):
                errs.append(f"arg name {name!r} must be a lowercase identifier")
            if not isinstance(meta, dict) or meta.get("type") not in ARG_TYPES:
                errs.append(f"arg {name!r} needs a type in {sorted(ARG_TYPES)}")
    for r in spec.required:
        if r not in spec.arg_schema:
            errs.append(f"required arg {r!r} is not declared in arg_schema")

    for s in spec.surfaces:
        if s not in KNOWN_SURFACES:
            errs.append(f"surface {s!r} not in {sorted(KNOWN_SURFACES)}")

    return errs
