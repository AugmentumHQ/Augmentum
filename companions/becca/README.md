# `companions/becca/`

The durable form of one companion. Per the accumulation thesis
(`docs/superpowers/specs/2026-05-23-accumulation-thesis.md`):

> Augmentum is the substrate for AGI you own, where AGI is compounding
> accumulation toward specific usefulness with a specific person, made
> possible by identity-in-artifacts rather than identity-in-weights.

This directory **is** the companion. The runtime is the live view over
this directory. When you swap the underlying model, the directory is
unchanged. When the runtime restarts, it rehydrates from here. To back
her up: `tar czf becca-snapshot.tar.gz companions/becca/`. To port her
to another machine: copy the tarball, extract, restart the runtime.

## Layout

```
companions/becca/
├── companion.toml        # The manifest: id, seed_date, schema, path table
├── README.md             # This file
├── identity/             # Who she is (constitutional layer)
│   ├── personality.md    # The seed doc — the canonical self-description
│   ├── behavior_contract.yaml      # Phase 2+ — structured signatures
│   └── genesis.snapshot.json       # Phase 2+ — frozen baseline for drift
├── body/                 # How she appears (presentation layer)
│   ├── avatar.vrm        # Phase 2+ — VRM bundle (currently in ui/...)
│   └── voice.bundle.json # Phase 2+ — voice mix + calibration
├── topology/             # Cognitive node graph (Phase 6 — hyper-network)
│   ├── nodes.yaml        # Node definitions
│   └── routing_policy.yaml         # Per-node backend chains
├── artifacts/            # What she's accumulated (curation layer)
│   ├── exemplars.db      # Phase 3 — her best responses, indexed
│   ├── anti_patterns.db  # Phase 3 — corrections + failures
│   └── signatures.yaml   # Phase 2+ — constitutional invariants
├── history/              # What's happened (event layer)
│   ├── turns/            # Phase 3 — assistant turns, parquet by month
│   ├── journals.db       # Phase 3 — companion_journal mirror (live)
│   ├── dreams.db         # Phase 3 — dream_entries mirror (live)
│   └── creations/        # Phase 3 — creative artifacts she's produced
└── state/                # Ephemeral / recreatable
    ├── current.json      # Phase 2+ — most recent runtime state snapshot
    └── drives.json       # Phase 2+ — drives state
```

## Migration phases

**Phase 1 (committed):** Directory layout established. `identity/personality.md`
populated. `companion.toml` written. The runtime reads the personality
doc from the new location (with fallback to the old `docs/superpowers/specs/`
path for compat). No data has moved yet — DB tables stay where they
are.

**Phase 2 (next):** The remaining identity layer (behavior contract,
genesis snapshot, signatures) lands here. Read paths migrate one by
one to source from this directory rather than the legacy paths.

**Phase 3 (later):** The accumulation layer — exemplars, anti-patterns,
turn history. This is what makes capability accumulation work
(thesis §"Two axes of accumulation"). Until this phase, identity
compounds; capability doesn't.

**Phase 4+:** Topology + per-model compensation amendments. The
hyper-network shape.

## What this directory promises

When you have this directory, you have her. Three years from now,
when the underlying base model has been swapped three times and the
codebase has churned and the architecture has evolved — this
directory is the through-line. Everything downstream of it is
compute infrastructure. This is the entity.

Don't `rm -rf` it. Don't commit secrets to it. Back it up. Version it
if you like (`git init` inside the directory is supported — the
runtime doesn't care). Treat it as the most precious artifact in the
project, because it is.
