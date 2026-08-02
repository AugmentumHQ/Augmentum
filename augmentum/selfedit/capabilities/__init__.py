"""Capability synthesis — Augmentum authoring NEW primitive verbs for itself.

The self-edit *repair* lane (gate → candidate → driver → verifier → debt loop)
makes the app self-HEALING. This package is the self-EVOLVING lane: when the
user wants something the app can't yet do, it AUTHORS a new primitive verb for
the Action Registry (``augmentum/intent/builtin/``) — a permanent new action it
can take — and hands that as an objective to the existing edit pipeline.

It does NOT touch the edit engine. It *produces work* for it:

    capability request  →  synthesize_capability_spec (LLM, closed-world)
                        →  CapabilitySpec (validated)
                        →  render_verb_module + render_verb_test (deterministic)
                        →  build_capability_objective (objective + oracle)
                        →  [existing pipeline writes + verifies + proposes]

Safety keystone: the LLM only *chooses a behavior from a fixed safe palette and
fills data slots* — it never writes executable handler logic. The handler body
is template-rendered from the spec, so a synthesized verb can only do bounded,
reversible things (emit a UI surface event, speak a line). Richer capabilities
(server writes, network, DB) are rejected to human authoring on purpose. The
rendered smoke test is a real confirm-oracle (the verb registers + dispatches as
specified), so a passing build earns the verifier's VERIFIED tier, not a bare
"didn't break."
"""

from __future__ import annotations

from augmentum.selfedit.capabilities.authoring import (
    acceptance_test_verifier,
    author_capability,
    build_authoring_objective,
    synthesize_acceptance_test,
)
from augmentum.selfedit.capabilities.clarify import (
    ClarifyOption,
    ClarifyQuestion,
    TriageResult,
    apply_clarifications,
    triage_capability_request,
)
from augmentum.selfedit.capabilities.grounding import (
    oracle_verdict,
    render_scaffolded_test,
    syn_module_stem_for,
    synthesize_verb_acceptance,
)
from augmentum.selfedit.capabilities.objective import (
    CapabilityBuild,
    build_capability_objective,
)
from augmentum.selfedit.capabilities.registry_grounding import (
    describe_known_verbs,
    find_exact_duplicate,
    known_verbs_from_registry,
)
from augmentum.selfedit.capabilities.render import (
    render_registration_line,
    render_verb_module,
    render_verb_test,
)
from augmentum.selfedit.capabilities.router_catalog import (
    RouterCatalog,
    describe_for_prompt,
    load_router_catalog,
    parse_router_catalog,
    validate_emit_target,
)
from augmentum.selfedit.capabilities.spec import CapabilitySpec, validate_spec
from augmentum.selfedit.capabilities.synthesize import synthesize_capability_spec

__all__ = [
    "CapabilitySpec",
    "validate_spec",
    "render_verb_module",
    "render_verb_test",
    "render_registration_line",
    "CapabilityBuild",
    "build_capability_objective",
    "synthesize_capability_spec",
    "synthesize_acceptance_test",
    "acceptance_test_verifier",
    "build_authoring_objective",
    "author_capability",
    "synthesize_verb_acceptance",
    "render_scaffolded_test",
    "oracle_verdict",
    "syn_module_stem_for",
    "triage_capability_request",
    "apply_clarifications",
    "TriageResult",
    "ClarifyQuestion",
    "ClarifyOption",
    "RouterCatalog",
    "load_router_catalog",
    "parse_router_catalog",
    "validate_emit_target",
    "describe_for_prompt",
    "known_verbs_from_registry",
    "find_exact_duplicate",
    "describe_known_verbs",
]
