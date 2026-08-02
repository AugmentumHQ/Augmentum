# Augmentum — Architecture (one-page primer)

This is the 5-minute mental model for new contributors. For depth on
any specific subsystem, see [`docs/subsystems.md`](subsystems.md). For
the cross-modal view (what makes Augmentum a personal AI OS, not an AI
app), see [`docs/integration-weave.md`](integration-weave.md).

## What Augmentum is

A FastAPI **intelligence-layer proxy** between LLM frontends and
backends — but the proxy is the surface, not the product. The product
is the substrate beneath: one identity layer, one memory graph, one
action surface, one federation fabric, one cast bus, one companion
orchestrator, all shared across ~35 subsystems.

Frontends (Open WebUI, SillyTavern, Cursor, custom UI, Claude Desktop
via the in-box MCP server) talk to Augmentum using the same OpenAI /
Ollama API shape they already use. Augmentum chooses how to handle
the request — pass it through, run a multi-step analytical flow, build
narrative memory, plan + act in a code workspace — and streams the
result back through whatever surface the user happens to be on
(chat in a tab, voice on the phone, TTS mirroring to the TV).

The point is **"one proxy, many modes, many surfaces, many devices"**:
a single endpoint that adapts behaviour to what the user is actually
doing AND coheres across modalities so a chat on the laptop, voice on
the phone, and cast playback on the TV are all the same conversation.

## Request flow

```
   ┌──────────────────────────────────────────────────────────┐
   │  Frontend                                                │
   │  Open WebUI · SillyTavern · Cursor · custom UI · curl    │
   └──────────────────────────┬───────────────────────────────┘
                              │  /v1/chat/completions
                              │  /api/chat   /v1/audio/...
                              │  /api/coder/...   etc.
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Raw ASGI middleware                                     │
   │   · auth (session cookie / API key / WS ticket)          │
   │   · rate limit                                           │
   │   · attaches `request.scope["user"]`                     │
   └──────────────────────────┬───────────────────────────────┘
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Classifier  (augmentum/classifier/)                     │
   │   header / prefix override > heuristics > narrative      │
   │   detector > complexity > fallback                       │
   └──────────────────────────┬───────────────────────────────┘
                              ▼
   ┌────────────┬────────────┬────────────┬───────────┬───────┐
   │ Passthrough│ Analytical │ Narrative  │  Agentic  │ Coder │
   │  + SSOS    │  (UARF)    │  (3-layer  │ (plan +   │ (Plan/│
   │  auto-tools│  6-phase   │   memory)  │  artifacts│  Act, │
   │            │  + tools   │            │  + tools) │  in   │
   │            │            │            │           │ docker│
   │            │            │            │           │ ws)   │
   └────────────┴────────────┴────────────┴───────────┴───────┘
                              │
                              │  Mode handlers prepare the prompt,
                              │  fetch memory / docs / tools, call
                              │  the configured model backend, and
                              │  emit a stream.
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Model backend                                           │
   │   bundled llama-server subprocess (LlamaServerManager)   │
   │   Ollama · OpenAI-compatible · Anthropic · etc.          │
   │   provider_registry routes to whichever the user picked  │
   └──────────────────────────┬───────────────────────────────┘
                              │  NDJSON / SSE / WebSocket
                              ▼
                          Frontend
```

## The substrate beneath

The request flow shown above is one slice through ~35 subsystems that
share six substrates. Most "I love this part of Augmentum" feedback
turns out to be about a substrate, not a feature. (Detailed in
[`docs/integration-weave.md`](integration-weave.md).)

```
                ┌─────────────────────────────────────┐
                │   COMPANION (Becca, dormant flag)   │  observes everything
                │   companion_runtime/                 │  → expresses on its own initiative
                └─────────────────────────────────────┘
                                  ▲
                                  │ observes
   ┌──────────────────────────────┴───────────────────────────────┐
   │                  THE REQUEST FLOW ABOVE                       │
   │     classifier → mode handler → backend → stream              │
   └──┬──────────┬─────────────┬─────────────┬──────────────┬─────┘
      │ scopes   │ recalls     │ invokes     │ resolves     │ mirrors
      ▼          ▼             ▼             ▼              ▼
   IDENTITY    MEMORY       ACTION         FABRIC          CAST
   layer       weave        surface        federation      bus
   ----------  -----------  -------------  --------------  ----------
   Argon2id    sqlite-vec   ToolRegistry   resolve_        in-RAM
   + tokens    + FTS5 +     + Surface-     backend_with_   tokens
   + raw-ASGI  4 vec0       Exposure       fabric          + VoiceFanout
   middleware  tables       declaration    (34 sites)      + iframe
              + Hebbian    + auto-mount   + Ed25519        shell
              cooccurrence  HTTP route    envelopes
                                          (default OFF)
```

- **Identity** scopes every read and write to the calling user. The
  conversation your phone-voice has is the same conversation your
  laptop sees, because they're both authenticated as you.

- **Memory** is a shared graph (chat extraction, narrative archive,
  dream compaction, documents RAG, coder turn archive, discovery
  clustering, companion identity — seven consumers, one substrate).

- **Action** declares per-surface exposure once and shows up everywhere
  appropriate. `image_generation` is reached by chat composer auto-
  attach, narrative scene generation, ebook auto-illustrate, Studio
  embedding, dream portraits, and a fabric peer's box dispatching
  through to use your GPU.

- **Fabric** routes any of 6 modalities (LLM / image / TTS / STT /
  knowledge / cast) to a paired peer transparently. 34 call sites use
  `resolve_backend_with_fabric` so peer routing isn't a separate code
  path. Default OFF, opt-in via `settings.fabric_enabled`.

- **Cast** mirrors voice / image / video output across browser ↔ TV ↔
  phone subscribers on one user session. Couch co-op (4 phases) shares
  a single emulator stream with named guests and per-guest saves.

- **Companion** (Becca) sits above modes — when the master flag is on,
  she observes every modality and acts on her own initiative between
  user turns. Default OFF; substrate built; 6/14 growth-loop actions
  shipped.

## Persistence

Single SQLite database, accessed via `aiosqlite`.

- **`augmentum/state/migrations/`** — every schema change is a
  numbered SQL file that runs in alphabetical order at startup.
  `python .claude/skills/augmentum-dev/scripts/gen_migration.py`
  picks the next number.
- **`augmentum/state/backends/sqlite.py`** — runner + connection
  pool + sqlite-vec extension.
- **`augmentum/state/manager.py`** — `StateManager` is the
  per-app handle apps grab from `request.app.state.state_manager`.
- **`augmentum/state/settings_store.py`** — KV table
  (`SettingsStore.get/set`) for everything user-configurable.

**Multi-tenant invariant:** every user-data table has a `user_id`
column and every CRUD function takes `*, user_id: str = ""`.
CLAUDE.md's table list and the `audit.py` `doc_facts` checker keep
this honest. The auth middleware attaches `request.scope["user"]`;
route handlers extract it; data calls scope by it; handlers cache
by `(user_id, session_id)`.

## The 5 modes (one line each)

| Mode | Lives in | What it's for |
|---|---|---|
| **Passthrough** | `augmentum/modes/passthrough/` | Simple proxy with auto-tools (calc, datetime, units). Defaults for short questions. |
| **Analytical** | `augmentum/modes/analytical/` | UARF 6-phase pipeline — classify, search, fetch, verify, synthesize, refine. For research/lookup. |
| **Narrative** | `augmentum/modes/narrative/` | Long-running RP/story with 3-layer memory (STATE + LEDGER + ARCHIVE), characters, personas, lorebook. |
| **Agentic** | `augmentum/modes/agentic/` | Goal-driven plan-as-anchor with artifact tools (docx/pptx/xlsx/chart/ebook). |
| **Coder** | `augmentum/modes/coder/` + `augmentum/coder/` | Plan/Act loop inside a per-workspace Docker container. Permissions, reviews, mission/promise verifiers. |

Mode selection happens in `augmentum/classifier/router.py`. Frontends
can override via `X-Augmentum-Mode: narrative` (or `:passthrough` etc.)
or by prefixing the message (e.g. `/coder: …`).

## Key subsystems by directory

```
augmentum/
├── auth/        Argon2id passwords + opaque tokens + ASGI middleware
├── classifier/  Mode selection + complexity analysis
├── coder/       Workspace containers, indexer, permissions, reviews
├── discovery/   Quality filtering for search results
├── documents/   RAG: chunking + FTS5 + embeddings (sqlite-vec)
├── dream/       Persona introspection — journal + portrait
├── games/       itch.io / JS13K web-game discovery
├── image/       Local SD/GGUF + cloud (OpenAI/Together/Stability/...)
├── jobs/        Restart-survivable background queue (gguf, gutenberg…)
├── knowledge/   Lorebook packs, RAG sources
├── mcp/         MCP client (external tool servers)
├── media/       Emby/Jellyfin/Audiobookshelf/Komga/Suwayomi/LibriVox
├── memory/      Embeddings + consolidation + compaction
├── modes/       The 5 mode handlers (passthrough, analytical, ...)
├── models/      LlamaServerManager + provider_registry + catalog
├── powers/      Capability packs (bias coder at safe checkpoints)
├── promises/    Mission runner + verifiers (replaces free-text plans)
├── proxy/       FastAPI app + every *_routes.py file + middleware
├── reasoning/   User-defined reasoning flows
├── resource/    VRAM/RAM ledger
├── session/     Per-session lifecycle hooks
├── state/       SQLite backend + migrations
├── tools/       Tool implementations (artifacts, web, search, image)
├── utils/       safe_http (SSRF), thinking (parser), logging
├── vfs/         Unified File Index across uploads/media/docs/artifacts
└── voice/       Text cleaning, TTS chunking
ui/
├── scripts/     ES-module frontend (no framework, vanilla DOM)
└── styles/      Per-surface CSS files
```

## Streaming

Server emits NDJSON events on HTTP and structured frames on
WebSocket. Every streaming surface ships at least three sub-states
(`awaiting_first_token`, `thinking`, `responding`) so the user
isn't staring at a dead spinner during long prefills. KV cache
restoration on the bundled llama-server avoids re-prefilling on
follow-up turns (`ttft_ms=0`).

## Where to read next

- [`docs/integration-weave.md`](integration-weave.md) — **the cross-modal view** (what the substrates share)
- [`docs/subsystems.md`](subsystems.md) — deep-dive per subsystem (~35 sections)
- [`docs/patterns.md`](patterns.md) — recurring code patterns with examples
- [`docs/testing.md`](testing.md) — test rulebook
- [`docs/security_model.md`](security_model.md) — threat model + boundaries
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — how to actually add a thing
- [`CLAUDE.md`](../CLAUDE.md) — invariants you must not break
