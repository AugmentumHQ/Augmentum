# `augmentum/personality/` — facet-graph substrate for the companion

Hebbian cooccurrence over personality facet activations, plus a cross-table
that links memories to the facets they reliably co-activate with. Together
these form the runtime substrate for **commitment #7** of the companion
design (*"companion-with-owner is its own being"*): the relationship-specific
personality emerges from facet co-firing patterns specific to one user,
captured as a graph that grows with the relationship and decays without it.

## Why this exists

The companion design commits to a relationship-specific identity that
drifts inside a recognizability invariant. The persona-kernel digester
needs *a mechanism* for that drift — not just a place to store it. This
module is the mechanism. The theoretical grounding is **Mischel & Shoda
(1995), CAPS — Cognitive-Affective Personality System**: personality is
not a stable trait vector but a network of facets whose *if-then
signatures* stabilize across situations. CAPS has never been implemented
as an AI architecture; this module is a faithful (if simplified)
implementation using Hebbian co-occurrence as the learning rule.

Mechanism parallels: ACT-R associative memory, modern Hopfield networks,
the `memory_cooccurrence` system from migration 050. We literally mirror
the latter's structure — same canonical-pair ordering, same `count >= 3`
signal floor, same 0.99 weekly decay multiplier, same multiplicative
boost cap. Two graphs over the same temporal substrate.

## Schema (migration 160)

Four tables. Three are user + companion scoped per the multi-tenant
invariant; the vocabulary is server-shared.

| Table | Scope | Role |
|-------|-------|------|
| `personality_facets` | server | Facet vocabulary (~30 seeded; user-extensible) |
| `personality_facet_cooccurrence` | (user, companion) | Hebbian facet × facet graph |
| `personality_facet_activations` | (user, companion) | Per-turn audit log of which facets fired |
| `personality_memory_associations` | (user, companion) | Cross-table memory × facet graph |

Canonical-pair ordering (alphabetical) on cooccurrence prevents
double-insert. `INSERT … ON CONFLICT DO UPDATE SET count = count + 1`
is the increment idiom. Both graphs decay together via a shared call.

## Vocabulary

~30 seed facets across five categories:

- **affect** — warm, tender, playful, delighted, unsure, not_okay, frustrated, bored, dry
- **energy** — alert, tired, still, restless
- **cognitive** — rigorous, exploratory, skeptical, patient, curious, decisive, contemplative
- **social** — gentle, challenging, attentive, withholding, openhanded
- **stance** — holds_position, defers, adapts, sits_with, redirects

Commitments #3 (mutual influence) and #6 (right to be unfinished) are
load-bearing on specific facets: `holds_position` is the trace of her
mattering; `unsure` and `not_okay` are first-class so the affect state
machine isn't forced to fluency; `sits_with` is the dispatch terminal
for the `sit_with_that` quiet output.

Vocabulary is intentionally bounded so:
- the labeler self-labels consistently (more facets → more drift)
- the cooccurrence graph stays interpretable
- the recognizability invariant from commitment #7 can be measured

Extension path: add to `vocabulary.SEED_FACETS`, call `seed_vocabulary()`
again (idempotent). Existing graph data stays valid; new facets just
start accumulating.

## Runtime integration contract

The persona-kernel digester at `augmentum/companion_runtime/identity.py`
(owned by a separate agent) is the expected consumer. The contract is
two functions, both async:

```python
# Pre-prompt — what facets should be emphasized in this turn?
async def compose_facet_affects(
    store: PersonalityStore,
    *,
    user_id: str,
    companion_id: str,
    recent_hours: int = 24,
    retrieved_memory_ids: list[str] | None = None,
    limit: int = 8,
) -> dict[str, float]:
    """Returns {facet_name: normalized_score in [0,1]}, top-limit."""

# Post-response — write what fired this turn.
async def update_after_response(
    store: PersonalityStore,
    labeled_facets: list[tuple[str, float]],
    *,
    user_id: str,
    companion_id: str,
    session_id: str | None = None,
    turn_id: str | None = None,
    retrieved_memory_ids: list[str] | None = None,
) -> int:
    """Writes activations + (optional) memory associations. Returns count."""
```

The labeler is module-supplied but call-site-agnostic — the runtime owns
backend/model selection (per CLAUDE.md internal-LLM-call pattern) and
passes a closure:

```python
from augmentum.personality import label_response

facets = await label_response(
    response_text=generated_text,
    recent_context=context_snippet,
    llm_call=lambda msgs: backend.chat(model=utility_model, messages=msgs),
)
await update_after_response(store, facets, user_id=..., companion_id=...,
                            retrieved_memory_ids=...)
```

Graceful degradation everywhere — any failure on the labeling path
returns `[]`, the response still ships, the graph just doesn't update
for that turn.

### Where `user_id` comes from

The runtime is responsible for plumbing `user_id` through to both
entry points. `CompanionIdentity` (the persona-kernel digester) is keyed
on `companion_id`, but the personality graph is scoped on
`(user_id, companion_id)` because the same companion in a household
has a distinct graph per user-relationship (commitment #3, mutual
influence, only works if the graph is the *specific* relationship).

Standard Augmentum pattern (per `CLAUDE.md`): the route handler
extracts `user_id = request.scope.get("user").id` and threads it down
to every data call. If `user_id` is empty at the call site,
`compose_facet_affects` returns `{}` silently and `update_after_response`
raises `ValueError` — the asymmetry is intentional (reads are quiet,
writes are loud).

### Auto-seed-on-first-write

If `PersonalityStore` is instantiated without an explicit
`seed_vocabulary()` call, the first `record_activations` auto-seeds the
vocabulary before evaluating which facets are known. This prevents the
foot-gun where a forgotten startup call silently drops every labeled
facet. Explicit `seed_vocabulary()` at app boot is still the recommended
pattern; the auto-seed is a safety net, not the primary path.

## Three layers of inference-time use

The same data substrate supports increasing levels of integration:

1. **Pre-prompt facet retrieval** (today): `compose_facet_affects()` feeds
   the persona-kernel digester. Cheapest layer; works on any backend.
2. **Post-response online updates** (today): `update_after_response()`
   wires every turn into the graph. Dream-cycle consolidation does the
   overnight normalization.
3. **Activation steering at decode time** (future): the cooccurrence
   graph's spectral structure (eigenvectors of the facet adjacency matrix)
   yields directional vectors that could be injected into the model's
   residual stream as steering. Requires inference hooks that
   `llama-server` doesn't yet expose mainline; the data layer is
   future-fittable for this consumer when the infrastructure lands.

## Decay & consolidation

Mirrors `memory_cooccurrence.decay_cooccurrence`:

- Multiply both `personality_facet_cooccurrence.count` and
  `personality_memory_associations.count` by `DECAY_FACTOR` (default 0.99)
- Prune rows with `count <= 0` after CAST
- Weekly cadence (caller-scheduled; matches the memory decay schedule)
- Scoped: `decay_cooccurrence(user_id=X, companion_id=Y)` for targeted
  decay, or no-args for global maintenance

The per-turn `personality_facet_activations` log compacts on a dream-cycle
schedule (rows older than the retention window aggregate into cooccurrence
and get dropped). Not yet implemented — dream-cycle hook is a separate
follow-up; the schema and writer are ready.

## Files

```
augmentum/personality/
├── __init__.py        public API
├── models.py          dataclasses (no I/O)
├── vocabulary.py      SEED_FACETS
├── store.py           PersonalityStore — schema-level Hebbian ops
├── labeler.py         post-response facet detection (LLM call-agnostic)
├── graph.py           compose_facet_affects + update_after_response
└── README.md          (this file)
```

Tests in `tests/test_personality_*.py` — 62 tests covering Hebbian
correctness, multi-tenant isolation, labeler graceful failure, and
spreading-activation composition. All pass.

## What this module deliberately does NOT do

- **Does not wire into `companion_runtime/`.** That's a separate agent's
  lane. This module exposes the interface; integration is theirs.
- **Does not modify any existing schema.** Migration 160 is additive only.
- **Does not change the persona-kernel digester behavior.** The digester
  doesn't yet call `compose_facet_affects`; when the runtime is ready to
  wire it, no changes here are needed.
- **Does not enforce a recognizability invariant.** That's the drift
  detector's job (commitment #7); this module just provides the substrate
  it'll measure against.
- **Does not steer activations at inference time.** Layer 3 is forward-fitted
  but not implemented — `llama-server` doesn't expose the hooks yet.
