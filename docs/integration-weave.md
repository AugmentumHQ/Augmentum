# Augmentum Integration Weave

The `ARCHITECTURE.md` one-pager shows how a single request flows through the
proxy. This doc shows what's BENEATH that flow — the six substrates that
every mode, surface, and device share. It's the doc to read when you want
to understand *what makes Augmentum a personal AI operating system instead
of an AI app*.

The short version: Augmentum isn't 35 subsystems sharing a database. It's
six substrates that ~35 surfaces tap into. Most "I love this part of
Augmentum" feedback turns out to be about a weave, not a feature.

## 1. The action weave

**Substrate:** `augmentum/tools/registry.py::ToolRegistry` + `tools/base.py::SurfaceExposure`.

Every callable primitive declares which surfaces it lives on:

```python
class SurfaceExposure:
    chat: bool = True
    voice: str = "core"   # core | interactive | disruptive | costly
    coder: bool = False
    companion: bool = False
    artifact_studio: bool = False
    file_context_menu: bool = False
    http_route: str = ""  # auto-mounts POST endpoint when non-empty
    voice_capability_line: str = ""
```

This is the **Unified Primitive Layer** (Phase 1 shipped — 5 tools migrated;
the rest still default to `SurfaceExposure()`). When a surface needs its
tool list, it calls `registry.get_for_surface(...)` and gets the slice
that's appropriate. One registration, eight consumers.

Concrete proof:

| Consumer | How it reaches the primitive |
|---|---|
| Chat composer | `chat/toolbar/tools.js` → header `X-Augmentum-Tools` → handler reads `registry.get_for_surface("chat")` |
| Voice always-listening | `intent/manifest.py::bind_registry` → `get_for_surface("voice", voice_level="core")` |
| Coder loop | `coder/tools.py::create_coder_tools` → `get_for_surface("coder")` + MCP-forwarded tools |
| Companion (Becca) | `companion_runtime/tools.py::execute_tool` consumes the same primitives via TagSieve emission |
| Artifact Studio | `studio_routes.py` exposes `surfaces.artifact_studio` buttons |
| HTTP REST | `tools/auto_routes.py::register_tool_routes` auto-mounts `POST {http_route}` for any tool declaring one |
| Fabric peer | `POST /api/fabric/image/generate` → `PipelineRegistry.generate_for_fabric` reaches the same `image_generation` primitive |
| Narrative scene generation | `modes/narrative/handler.py::_generate_scene_image` reaches the same primitive again |

**Key insight:** when you write a tool, you don't pick a mode to live in.
You declare which surfaces should see it, and the registry handles the rest.

## 2. The memory weave

**Substrate:** `EmbeddingService` + `RerankService` + five sqlite-vec vector
tables.

```
memories_vec          ← memories.id (chat + companion extraction)
doc_chunks_vec        ← document_chunks.id (RAG)
narrative_archive_vec ← narrative_archive.id (long-form story memory)
dream_entries_vec     ← dream_entries.id (compaction output)
interest_clusters_vec ← interest_clusters.id (discovery)
```

Seven consumers read from and write to this graph:

| Consumer | Mechanism | Touched |
|---|---|---|
| Chat extraction | `memory/integration.py::schedule_extraction` | `memories`, `memory_cooccurrence` |
| Narrative archive | `modes/narrative/handler.py::_archive_and_embed` | `narrative_archive`, `narrative_archive_vec` |
| Companion identity + drives | `companion_runtime/{identity,consolidation,honest_gap}.py` | `memories` (read-heavy), `companion_journal` |
| Dream cycles | `dream/engine.py::run_cycle` | `dream_entries`, `dream_memory_log` |
| Documents RAG | `documents/store.py::search` | `documents`, `document_chunks`, `doc_chunks_vec` |
| Coder turn archive | `coder/turn_archive_embed.py` | `coder_turn_archive` |
| Discovery clustering | `discovery/clustering.py` | `interest_clusters` |

**Hebbian cooccurrence** (`memory_cooccurrence`, migration 050) is the
load-bearing piece that makes this feel coherent across consumers: when a
fact is recalled in chat, associates surface in companion, in dream
compaction, in document RAG. Tier history (`memory_tier_history`,
migration 215) audits demotes so retroactive forgetting is reversible.

**Key insight:** if you write a feature that needs a context graph, you
don't build one. You hand the relevant text to `EmbeddingService`, persist
it with `user_id`, and the rest of the substrate sees it.

## 3. The multi-device weave

**Substrate:** `devices/cast_tokens.py` (in-RAM, 30-min TTL, IP-bound) +
`voice/fanout.py::VoiceFanout` + `cast/input_bridge.py` + per-cast-surface
iframe shell.

```
phone     ───┐
laptop    ───┼──── one user session (Argon2id + opaque token)
TV        ───┤
tablet    ───┘
```

The same TTS bytes hit `voice_fanout` and mirror to every subscriber.
Cast-receiver is an iframe shell that hosts cast-app, cast-audio,
cast-video, cast-comic, cast-vrm. Cast-control on a phone sends gamepad
input through `/api/cast/input/ws` → `input_bridge.py` → container UInput
pads → emulator session.

**Couch co-op verified, all 4 phases shipped** (migrations 229-231):

1. Anonymous join via QR token (`/api/cast/games/session/{id}/invite`)
2. Named guest profiles (`guest_profiles` table, host-scoped)
3. Device-fingerprint auto-reconnect (`cast_guest_devices`, warm-slot reclaim within `WARM_SLOT_TTL_S=30.0`)
4. Per-guest save slots (`game_saves.guest_profile_id` nullable FK)

**Game-stream admission control** (`runtime.py:176-204`): credit-budget
(`active_credit_budget=8`, `resident_credit_budget=16`) plus docker-pause
primitive plus `paused_at` watchdog (migration 223). One box hosts
concurrent emulator sessions across the household without thrashing.

**Key insight:** the same identity propagates to every device on your LAN
without you re-authenticating. Add a new cast surface by writing one
HTML/JS bundle in `ui/cast-<name>/` — it joins the iframe shell automatically.

## 4. The federation weave

**Substrate:** `models/provider_registry.py::resolve_backend_with_fabric` +
`fabric/director.py::RoutingDirector` + Ed25519-signed envelopes.

```
caller (any of 34 sites)
   │
   ▼
resolve_backend_with_fabric(model, user_id, ...)
   │
   ├─ explicit @fabric:<peer> pin? → FabricBackend
   ├─ local model available? → LocalBackend
   └─ director.maybe_route_llm() consults peer capabilities → FabricBackend or None
```

**23 call sites across 28 files** — narrative, coder, reasoning, flow,
tools, anthropic, openai, ollama, voice, browse, bug_finder, architect,
agents, draft_section, artifact_ebook, game_agent llm_bridge. Every
modality is cross-peer-routable.

Six modalities run over fabric: LLM (`/api/fabric/inference`), image
(`/api/fabric/image/generate`), TTS (`/api/fabric/tts`), STT
(`/api/fabric/stt`), knowledge search (`/api/fabric/knowledge/search`,
local packs only), cast render (`/api/fabric/render`).

**Trust model:** pinned Ed25519 fingerprints (SSH-style), not TLS. Per-peer
service users (`fabric:<short-node-id>`) — data isolation moves from
per-user to per-peer on the receiving side. Cloud-backed LLM providers
explicitly NOT advertised over fabric (`fabric/extractors.py:42-67`) so a
peer can't spend another peer's API budget.

**Default OFF** (`settings.fabric_enabled` — identity isn't even generated
until flipped on; see `fabric/lifespan.py:43`). But every modality is
already wired.

**Key insight:** the function that resolves "which model serves this
request" is the same function that resolves "is there a paired peer that
should serve this request instead". Don't build a peer router; use the
backend resolver and it handles peers.

## 5. The companion weave

**Substrate:** `companion_runtime/runtime.py::CompanionRuntime` (master flag
default OFF) + `BeccaObserver` + presence bus + intent registry +
`becca_direct` handler.

Designed as the orchestrator **above** modes — not a 6th mode. When the
master flag is on:

- `BeccaObserver` subscribes the PresenceBus (`runtime.py:278`)
- Every chat turn salience-scored and journaled (`salience.py`)
- Every voice turn synapse-journaled (`voice_journal`)
- Intent dispatch routes Tier-1 regex verbs (`intent/dispatch.py:152`)
- `becca_direct` chat handler emits `becca_tool_call/result/handoff` chunks consumed by `TagSieve` (`modes/becca_direct/handler.py:496`)

**What gets observed:** chat (salience), voice (journal), narrative
(narrative_isolation.py), coder (signal_aggregator pulls bug_finder_runs),
media (audio bus surface events), image gen (image_gen primitive), browse
(perception/topical).

**What expresses through:** BeccaVoice path in `voice_routes.py:1193`;
`becca_direct` chat; today reflection notifications; initiative queue;
PAD bus to avatar; XR scene (when enabled); growth-loop reward emission.

**State:** built, dormant. The substrate exists; the master flag
(`companion_runtime_enabled` in `config.py:692`) defaults False. 14
sub-flags also default False. The growth catalog has 6 of 14 actions
shipped. BOM observation Phase A only — L0 lookup-cache exporter is
running; L1 token-type abstractions and L2 logit fingerprints are
spec-only.

**Key insight:** when companion is on, it doesn't replace modes. Modes
serve requests; companion observes them and acts on its own initiative
between them.

## 6. The read-layer weave

**Substrate:** `knowledge/packs.py::PackManager` (hybrid vector + FTS5 +
ZIM keyword + RRF + cross-encoder rerank) + `documents/store.py` +
`browse_routes.py` multi-layer fetcher + `notes_store.py`.

```
              ┌── knowledge_packs (encyclopedic, install-wide)
              ├── documents       (user-uploaded, RAG, parent-child chunks)
              ├── browse_history  (visited URLs + cluster IDs)
              └── browse_notes    (user notes, w/ source_url provenance)
                                ▲
                                │
chat / voice / coder ───────────┘
   (per-mode injection toggles)
```

**Per-mode chat injection** (`knowledge/injection.py::_pack_mode_enabled`):

- Passthrough — ON by default
- Analytical — ON by default
- Agentic — ON by default
- Narrative — OFF by default (lorebook handles worldbuilding)
- Coder — explicitly excluded (pack noise hurts plan/act)
- Companion BeccaDirect — passes through whatever's relevant

**Browse multi-layer fetcher** (`browse_routes.py::_fetch`):
Chrome-TLS → Wayback fallback → oEmbed discovery → ~10 API shortcuts
(Wikipedia, Reddit, HN, GitHub, HuggingFace, Stack Exchange, arXiv,
Discourse, RSS fallback) → Trafilatura extraction.

**Browse → notes → memory chain** (the load-bearing path):
`POST /api/browse/save` ingests a page into DocumentStore as
`browse_<title>.txt` AND bumps `domain_reputation`. Notes get margin
annotations from the SAME lorebook engine narrative mode uses. Notes'
`source_url` carries provenance forward into RAG and discovery.

**Key insight:** if you save a page from Browse, it becomes searchable in
chat, citable in agentic deliverables, and surfaceable in companion's
today-reflection — all without any per-feature pipeline.

## How to extend a weave (vs add a feature)

Most "new features" in Augmentum touch one or more weaves rather than
standing alone. Use this checklist when scoping:

| You're adding... | Likely touches |
|---|---|
| A new chat tool | Action weave (declare surfaces) + maybe Memory (if it persists state) |
| A new modality (e.g. video stream) | Action + Multi-device + maybe Federation |
| A new long-running ambient feature | Companion + Memory + maybe Read-layer |
| A new content source (e.g. a podcast adapter) | Read-layer + Multi-device (cast playback) |
| A new sharable artifact | Action (artifact tool) + Memory (embed for recall) + Community (publish) |
| A new device kind (e.g. a Vision Pro surface) | Multi-device + Companion (if it observes) |

When you find yourself building a parallel pipeline that bypasses a weave,
stop and ask whether you're re-implementing the weave. The answer is
usually yes.

## What's intentionally NOT woven

Not every Augmentum feature should be cross-modal. The deliberate
non-weaves:

- **Companion tool tree separate from chat tool registry** — two trees,
  shared leaves at the implementation layer. Runtime/trust/latency
  mismatches make consolidation a false economy (see project memory
  `project_companion_tool_tree_separate`).
- **Direct mode** (`Mode.DIRECT`) is the *anti-weave* — raw API pipe with
  no memory recall, no knowledge packs, no dream/media context, no
  vision captions, no SSOS, no datetime injection. For external clients
  that want Augmentum-as-proxy without the substrate.
- **Cloud LLM providers are not fabric-advertised** so peers can't spend
  each other's API budgets.
- **Coder mode excluded from knowledge pack injection** — repo digest +
  indexer are the substrate; pack noise hurts plan/act.

These aren't bugs. They're load-bearing decisions about where coherence
hurts more than it helps.

## What's in flight per weave

As of 2026-06-04 (see also `docs/full-audit-2026-06-04.md`):

| Weave | In-flight |
|---|---|
| Action | Unified Primitive Layer Phase 1 — 5 tools migrated; rest still default `SurfaceExposure()` |
| Memory | Branch-tagged narrative tables (migrations 115-118) shadow-write; legacy JSON columns still read path |
| Multi-device | Body atlas LOADS but contact reactions don't auto-fire (`anim-atlas.js:1125`) |
| Federation | Default OFF; cert pinning Phase-1+ follow-up; NanoClaw external adapters not in-tree |
| Companion | Master flag default OFF; 14 sub-flags OFF; growth catalog 6/14 actions; BOM L1/L2 spec-only |
| Read-layer | Two notes editors coexisting (CM6 + Milkdown-era); experimental docs modules (`answer_density`, `span_filter`) not integrated |

The weaves themselves are shipped and load-bearing. The in-flight work is
about lighting up dormant pieces or consolidating duplicated paths.

## See also

- `docs/ARCHITECTURE.md` — request-flow primer (the surface)
- `docs/subsystems.md` — per-subsystem deep dives
- `docs/patterns.md` — recurring code patterns
- `docs/security_model.md` — threat boundaries
- `CLAUDE.md` — invariants that hold across the weaves
- `docs/full-audit-2026-06-04.md` — full snapshot with file:line evidence
