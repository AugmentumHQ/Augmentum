"""Surface-agnostic reshape — the foundation working on every hat.

One contract (``SurfaceAdapter``: apply / revert / verifier / capture) + one
orchestration (``reshape``) make every Augmentum delivery surface — config,
media, document, VR scene, … — reshape-able by the same intent→verify→apply→
revert primitive. What differs per surface is only the actuator behind the
adapter. Add a surface by writing one adapter and registering it; the core never
changes.

See ``docs/superpowers/specs/2026-06-23-grows-with-user-hermes-build-reference.md``
(the "Surface-agnostic reshape" section).
"""

from __future__ import annotations

from augmentum.selfedit.surfaces.base import (
    CLASS_ADAPTATION,
    CLASS_BUILD,
    CaptureArtifact,
    ReshapeChange,
    ReshapeOutcome,
    SurfaceAdapter,
    clear_surfaces,
    get_surface,
    register_surface,
    registered_surfaces,
)
from augmentum.selfedit.surfaces.catalog import (
    AdaptationField,
    SurfaceSchema,
    clear_schemas,
    example_adaptation_schema,
    get_schema,
    register_schema,
    registered_schemas,
    render_catalog,
    validate,
)
from augmentum.selfedit.surfaces.classify import build_model_classifier
from augmentum.selfedit.surfaces.config_surface import (
    SURFACE_CONFIG,
    build_config_surface,
)
from augmentum.selfedit.surfaces.engine import (
    STATUS_APPLIED_PENDING,
    STATUS_FAILED,
    STATUS_PROMOTED,
    STATUS_REVERTED,
    STATUS_UNMAPPED,
    EngineResult,
    ReshapeRequest,
    run_reshape_request,
)
from augmentum.selfedit.surfaces.handler import handle_reshape_ask
from augmentum.selfedit.surfaces.live import (
    build_store_recorder,
    register_default_surfaces,
)
from augmentum.selfedit.surfaces.model_invoke import build_role_model_invoke
from augmentum.selfedit.surfaces.model_source import (
    DEFAULT_SOURCE,
    SOURCE_SETTING_KEY,
    ModelSource,
    SourceContext,
    clear_sources,
    get_source,
    list_sources,
    register_default_sources,
    register_source,
    resolve_invoke,
)
from augmentum.selfedit.surfaces.presentation import (
    ReshapeAction,
    ReshapePresentation,
    present,
    proposed_presentation,
)
from augmentum.selfedit.surfaces.reshape import ReshapeResult, reshape

__all__ = [
    "CLASS_ADAPTATION",
    "CLASS_BUILD",
    "DEFAULT_SOURCE",
    "SOURCE_SETTING_KEY",
    "SURFACE_CONFIG",
    "AdaptationField",
    "ModelSource",
    "ReshapeAction",
    "ReshapePresentation",
    "SourceContext",
    "present",
    "proposed_presentation",
    "clear_sources",
    "get_source",
    "handle_reshape_ask",
    "list_sources",
    "register_default_sources",
    "register_source",
    "resolve_invoke",
    "SurfaceSchema",
    "build_model_classifier",
    "build_role_model_invoke",
    "clear_schemas",
    "example_adaptation_schema",
    "get_schema",
    "register_schema",
    "registered_schemas",
    "render_catalog",
    "validate",
    "STATUS_APPLIED_PENDING",
    "STATUS_FAILED",
    "STATUS_PROMOTED",
    "STATUS_REVERTED",
    "STATUS_UNMAPPED",
    "CaptureArtifact",
    "EngineResult",
    "ReshapeChange",
    "ReshapeOutcome",
    "ReshapeRequest",
    "ReshapeResult",
    "SurfaceAdapter",
    "build_config_surface",
    "build_store_recorder",
    "clear_surfaces",
    "get_surface",
    "register_default_surfaces",
    "register_surface",
    "registered_surfaces",
    "reshape",
    "run_reshape_request",
]
