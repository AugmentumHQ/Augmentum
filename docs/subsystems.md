# Augmentum Subsystems Reference

Deep-dive docs for each major subsystem. Read the relevant section when touching a subsystem; the top-level `SKILL.md` carries the cross-cutting patterns.

Companion docs:
- `patterns.md` — recurring code idioms
- `security_model.md` — threat model & trust boundaries
- `testing.md` — test rulebook

---

## Table of Contents

**Request path**
1. [Classifier / Mode Router](#classifier--mode-router)
2. [Mode Handlers](#mode-handlers)
3. [Companion (Becca)](#companion-becca)
4. [Coder Mode](#coder-mode)
5. [Narrative Engine](#narrative-engine)
6. [Agentic Mode](#agentic-mode)
7. [Reasoning Flows](#reasoning-flows)
8. [Intent Dispatch + Action Catalog](#intent-dispatch--action-catalog)

**Memory & knowledge**
9. [Memory System](#memory-system)
10. [Observation Substrate (BOM)](#observation-substrate-bom)
11. [Dream System](#dream-system)
12. [Documents / RAG](#documents--rag)
13. [VFS (File Index)](#vfs-file-index)

**Media & generation**
14. [Voice Pipeline](#voice-pipeline)
15. [Image Pipeline](#image-pipeline)
16. [Media System](#media-system)
17. [Avatar System](#avatar-system)
18. [Cast & Multi-Device](#cast--multi-device)
19. [Controllers](#controllers)
20. [Game Streaming](#game-streaming)
21. [Games Portal](#games-portal)

**Tooling & integration**
22. [Jobs Queue](#jobs-queue)
23. [MCP Integration](#mcp-integration)
24. [Powers System](#powers-system)
25. [Promises / Missions](#promises--missions)
26. [Discovery & Quality](#discovery--quality)
27. [Search & Tools](#search--tools)
28. [Character Cards](#character-cards)
29. [Fabric / Federation](#fabric--federation)
30. [Connect (Calls & Messaging)](#connect-calls--messaging)
31. [Community Marketplace](#community-marketplace)
32. [Bug Finder Agent](#bug-finder-agent)

**Infrastructure**
33. [Engine v2 (llama-server)](#engine-v2-llama-server)
34. [Resource Ledger](#resource-ledger)
35. [Docker & Deployment](#docker--deployment)

> For the cross-modal view (what these subsystems share), see
> [`docs/integration-weave.md`](integration-weave.md).

---

## Classifier / Mode Router

**Dir:** `augmentum/classifier/`

Stateless classifier chooses mode for incoming requests.

| File | Role |
|------|------|
| `router.py` | `classify_request(req) → ClassificationResult(mode, confidence, reason, metadata)` |
| `complexity_analyzer.py` | Heuristic complexity score (goals, constraints, tool patterns) |
| `narrative_detector.py` | Story/RP keyword + sentiment signals |

Ordering: explicit override (`X-Augmentum-Mode` header, or `p/`/`a/`/`n/`/`g/`/`c/` model prefix) → complexity gate → narrative detector → fallback.

**Modes emitted:** `PASSTHROUGH`, `ANALYTICAL`, `NARRATIVE`, `AGENTIC`, `CODER`.

---

## Mode Handlers

**Files:** `augmentum/modes/{passthrough,analytical,narrative,agentic,coder}/handler.py`
**Base:** `augmentum/modes/base.py`

| Mode | Handler | Purpose |
|------|---------|---------|
| Passthrough | `PassthroughHandler` | Direct proxy + SSOS auto-tools |
| Analytical | `AnalyticalHandler` | UARF 6-phase + tool calling |
| Narrative | `NarrativeHandler` | Story/RP + 3-layer memory + character cards |
| Agentic | `AgenticHandler` | Goal-driven plan + artifact generation |
| **Coder** | `CoderHandler` | Plan/Act + containerized workspace + mission/promises |

### Lifecycle
1. Request hits `openai_routes.py` / `ollama_routes.py` / `chat_routes.py`
2. Classifier picks mode
3. Handler instantiated or pulled from per-session cache (`app.state.*_handlers`)
4. `process_stream()` yields chunks via `chat_egress.emit()`
5. State saved after response (handlers own their own save/load)

### Cache keys
`(user_id, session_id)` — **never bare `session_id`** when auth is active.

### SSOS orchestrator (Passthrough)
`augmentum/modes/passthrough/orchestrator.py` — intent classifier (search/fetch/calc/convert/datetime/build_app). Self-gates on the per-user `ui.autoTools` preference (in `user_settings`); when on, auto-includes `calculator`, `datetime`, `unit_converter`. Known gap: `tool_synthesis_hint` is defined but never passed.

---

## Companion (Becca)

**Dir:** `augmentum/companion_runtime/` + `augmentum/companion/` + `augmentum/intent/` — routes: `companion_routes.py`, `companion_growth_routes.py`, `observation_routes.py`

Orchestrator layer ABOVE modes — observes cross-modal signals, journals
salient turns, dispatches autonomous "growth" actions on a tick loop, and
can route chat turns into a mode via subagent dispatcher when on. Master
flag default OFF (`companion_runtime_enabled` in `config.py:692`).

| File | Role |
|------|------|
| `runtime.py:70` | `CompanionRuntime` — lifecycle, tick loop, observer subscriptions |
| `behavior/tick.py` | Per-user tick loop — drives growth actions + reflection |
| `salience.py` | Chat turn salience scoring (threshold 0.55 → journal) |
| `consolidation.py` | Memory consolidation per-user, reuses `EmbeddingService` |
| `chat_router.py:106` | `resolve_chat_mode()` — companion dispatcher hook in chat routing |
| `tool_protocol.py::TagSieve` | Parses `<tool:NAME …/>` tags from Becca's stream |
| `modes/becca_direct/handler.py:65` | `becca_direct` chat handler — bridges chat to companion |
| `companion/growth/actions/` | Action catalog (6/14 shipped: recall.surface_connection, narrate_growth, discovery_surface, care_consolidate, proactive_offer, companionship) |

**Tables (24+, all user-scoped):** companion_identities, companion_state,
companion_state_log, companion_journal, companion_journal_archive,
companion_creations, companion_initiative_queue, companion_scene,
companion_observations, companion_skill_archive, companion_skills,
companion_skill_instances, companion_safety_floor_rolling_user_view,
companion_rebuild_log, companion_affect_baselines, companion_topic_mutes,
companion_note_feedback, companion_drive_state, companion_today_reflections,
companion_growth_backlog, companion_growth_log, companion_economy,
companion_economy_tx.

Observes: chat (salience), voice (synapse journal), narrative (isolation),
coder (signal aggregator), media (audio bus events), browse (perception).
Expresses through: BeccaVoice (`voice_routes.py:1193`), `becca_direct`
chat handler, initiative queue → notes, PAD bus → avatar.

**State:** built, dormant. Substrate exists; master flag + 14 sub-flags
default False. See [`integration-weave.md#5-the-companion-weave`](integration-weave.md#5-the-companion-weave)
for the cross-modal observation + expression channels.

---

## Coder Mode

**Dirs:** `augmentum/coder/` (core), `augmentum/modes/coder/` (handler)
**Routes:** `coder_routes.py`, `coder_permission_routes.py`, `coder_review_routes.py`
**WebSocket:** `/ws/terminal/{workspace_id}`

Containerized agentic coding: Plan/Act loop, persistent workspace, semantic index, structured missions.

### Components

| Layer | File | What it does |
|-------|------|--------------|
| Container | `coder/containers.py` | `ContainerManager` — async Docker lifecycle, labeled `augmentum.workspace=true` for restart survival |
| State | `coder/state.py` | `CoderState` — phase enum, working_set, plan/mission, turn_summaries (FIFO 10) |
| Snapshot | `coder/snapshot.py` | Auto-refreshing workspace tree, `[NEW]`/`[MOD]`/`[DEL]` markers, injected turn-start + every 8 iters |
| Digest | `coder/digest.py` | Inlines every file when total < ~40K tokens — skips dir_tree/file_read ceremony on small projects |
| Index | `coder/indexer.py` | sqlite-vec inside container's `.augmentum/index.db` — 100-line chunks, 20-line overlap |
| Scratch | `coder/scratch.py` | `ScratchStore` — large tool outputs; model reads by key |
| Tools | `coder/tools.py` | `create_coder_tools()` — shell, file, git, search, MCP-forwarded. Enforces read-before-edit |
| Plan | `modes/coder/phase_plan.py` | Plan generation + step parsing |
| Act | `modes/coder/phase_act.py` | Tool dispatch, verification, streak-break reflexion |
| Handler | `modes/coder/handler.py` | Orchestrator — strategy selector, streaming, terminal control |

### Persistence

`coder_sessions` — session_id, workspace_id, phase, plan, plan_steps, current_step, step_outputs, mission (JSON Promises list), turn_summaries (JSON).
`coder_workspaces` — metadata, resource limits.
`coder_turn_summaries`, `coder_tool_failures` — audit/debug.

### Strategies
Gated by `AUGMENTUM_CODER_STRATEGY`:
- `native` — minimal Claude-Code/Qwen-Code parity loop (`_act_native`), default
- `hybrid` — 4-innovation rebuttal loop (`_act_hybrid`)
- `canonical` — Codex/Claude Code-style consensus (`_act_canonical`)
- `legacy` — Phase 1 fallback (no containers); lives in `_legacy.py`

### Terminate conditions
Task-completion signal, consecutive no-op streak (triggers reflexion), hard iter cap, user `Ctrl+C` (propagates to `docker exec`).

### Cross-turn state
`turn_summaries` inject as `<prior_turns>` block in next turn's system prompt — stops model re-reading same files.

### Review & permissions
- `coder_review_routes.py` — accept/reject hunks per file before commit
- `coder_permission_routes.py` — path-prefix allow/deny enforced in `tools.py`

---

## Narrative Engine

**Files:** `augmentum/modes/narrative/*.py`

### Three-layer memory

| Layer | Purpose | Persistence |
|-------|---------|-------------|
| STATE | Current snapshot (characters, scene, mood) | `narrative_memory` table |
| LEDGER | Running event log (facts, changes, quotes) | `narrative_memory` table |
| ARCHIVE | Compressed summaries + embeddings | `narrative_memory` + embeddings |

### Flow
1. Every N messages → `llm_extractor.py` pulls entities/facts/plots
2. STATE updated, LEDGER appended
3. LEDGER over ceiling → continuous archiving compresses to ARCHIVE
4. Context builder assembles: system + STATE + LEDGER + retrieved ARCHIVE

### Critical: sync before save
```python
engine.sync_to_state()
await handler.save_narrative_state(session_id)
```

### Per-session settings
`memory_settings.py` — `SessionMemorySettings`, toggles for state/ledger/archive per session. Handler uses `_mem_setting()` for session → global fallback.

### Macro expansion
`macro_expander.py`: `{{char}}`, `{{user}}`, custom macros from card `extensions`.

### Refusal filter
`_is_refusal_text()` uses compound phrase matching — `"I cannot"` alone doesn't match, but `"I cannot generate"` does. Prevents false positives on legitimate text.

---

## Agentic Mode

**Files:** `augmentum/modes/agentic/*.py`

- Planner decomposes goal into step-by-step plan
- Autonomy dial: 4 levels (1 = human approval, 4 = fully autonomous)
- Working memory: goal, constraints, completed steps, observations
- Plan-as-attention-anchor: plan injected into every step's context

### Artifact tools (`augmentum/tools/artifact_*.py`)

| Tool | Stack |
|------|-------|
| `artifact_document.py` | python-docx (Word) |
| `artifact_presentation.py` | python-pptx (PowerPoint) |
| `artifact_spreadsheet.py` | openpyxl (Excel) |
| `artifact_chart.py` | matplotlib |

---

## Reasoning Flows

**Dir:** `augmentum/reasoning/` — routes: `reasoning_routes.py`, `flow_routes.py`

User-defined step pipelines for analytical mode. Replaces hardcoded phase logic.

| File | Role |
|------|------|
| `executor.py` | `execute_flow_stream(flow, req) → AsyncIterator[chunk]` |
| `templates.py` | Bundled flow definitions (1741 lines) — step templates, role behaviors |
| `store.py` | Flow persistence, versioning |
| `models.py` | `ReasoningFlow`, `FlowStep` dataclasses |
| `resolver.py` | Variable binding (`$QUERY`, `$SEARCH_RESULTS`, `$CONTEXT`, …) |

**Step roles**: `classify`, `search`, `verify`, `respond`.

Each step has a complexity gate — trivial queries skip search/verify. Substitution happens **before** LLM call (not runtime).

**Persistence:** `custom_flows` (user-scoped), `reasoning_flows` + `reasoning_flow_steps` (legacy).

---

## Intent Dispatch + Action Catalog

**Dir:** `augmentum/intent/` — wired to voice via `voice_routes.py:2078`, to chat via the `ToolRegistry`

Three-tier dispatch for the user-addressable verb library. Tier 1 regex
matchers compile from auto-derived templates; Tier 2 embedding similarity
is a slot (`Phase 10`, unbuilt); Tier 3 exposes actions to the LLM via
native function calling.

| File | Role |
|------|------|
| `dispatch.py:152` | `dispatch()` — matcher → registry → handler |
| `registry.py:45` | `REGISTRY = _ActionStore()` singleton (stakes-validated) |
| `matcher.py` | Tier 1 regex pattern compilation + matching |
| `tool_adapter.py` | `register_action_tools` — wires Tier 3 actions into native function calling |
| `manifest.py:90-141` | Voice-tool manifest, binds to `ToolRegistry` |
| `builtin/control.py` | `control.stop / repeat / slower / louder / goodbye / nevermind` |
| `builtin/navigation.py` | `navigate.open_surface / navigate.back` |
| `builtin/notes.py` | `note.create / append / show_sticky / start_capture / end_capture`, `memory.save / recall` |

**ReferentCache** (per-`(user_id, session_id)` on `app.state.intent_referents`,
TTL 24h, sweep every 60s) lets `note.append` target the recently-mentioned
note without re-naming.

Voice hook: `proxy/voice_routes.py:2078::_maybe_dispatch_intent` runs after
STT, before backchannel filter. Three outcomes: short-circuit (handled,
skip LLM), soft-augment (prompt addendum injected), pass-through.

---

## Memory System

**Files:** `augmentum/memory/*.py` — routes: `memory_routes.py`

- Core profile: persistent user summary, rebuilt periodically
- Memories: extracted facts with sqlite-vec embeddings
- Consolidation: merges similar memories
- Compaction: removes old/low-relevance
- Scope: global or per-mode (`memory_scope_by_mode`)

Embedding via sentence-transformers or API. Recall: top-K by cosine ≥ min_score.

---

## Observation Substrate (BOM)

**Dir:** `augmentum/observation/` + `augmentum/signals/` — routes: `observation_routes.py`

Cross-modal pattern memory store. **Phase A only**: L0 (tokenizer-agnostic
exact observations) shipped as a lookup-cache exporter for the llama-server
drafter. L1 (token-type abstractions) and L2 (logit fingerprints) are
spec-only, no code path yet. Multi-tenant cache replicas deferred.

| File | Role |
|------|------|
| `observation/store.py` | L0 observation store (migration 234, primary-user-keyed) |
| `observation/fingerprint.py` | Observation fingerprint computation |
| `observation/seeder.py` | Chat-history seed pipeline (off by default) |
| `observation/exporter.py` | Per-`(user, model)` lookup-cache export → llama-server drafter |
| `signals/aggregator.py` | Daily pass: `bug_finder_runs` + `companion_journal` → `signal_events` |

**Tables:** `bom_observations_exact` (migration 234), `signal_events` (migration 206).

Gated behind 5 ops-tier flags (default OFF). Single consumer today:
llama-server's `--lookup-cache-static` drafter when the cache file exists
for the current (user, model) pair. Future consumers (autocomplete,
companion expression policy, behavioral ranker) are spec-shaped.

---

## Dream System

**Dir:** `augmentum/dream/` — routes: `dream_routes.py`

Persona introspection engine. AI reflects on conversations and produces journal entries + evolved portrait.

| File | Role |
|------|------|
| `engine.py` | `DreamEngine` — cycle orchestrator (context build → prompt → LLM → parse → store) |
| `scheduler.py` | `DreamScheduler` — per-user message/approval counters, idle gating, cooldowns |
| `journal.py` | `DreamJournal` — entry storage, dedup |
| `portrait.py` | Persona portraits (appearance, summary, mood history) |
| `lifecycle.py` | State transitions |

**Tables:** `dream_entries`, `dream_cycles`, `dream_portraits`, `dream_memory_log`.

**Settings:** `dream_model`, `dream_max_context_tokens`, `dream_portrait_model`, `dream_recall_enabled`, `dream_recall_limit`. UI: `ui.dreamEnabled`, `ui.dreamMessageThreshold`, `ui.dreamIdleMinutes`, `ui.dreamCooldownMinutes`.

**Gotchas:**
- Scheduler is a process singleton with per-user counters; `user_id = ""` = legacy default bucket
- `DreamsDisabledError` → HTTP 409 (opted-out) vs HTTP 503 (system unavailable)

---

## Documents / RAG

**Dir:** `augmentum/documents/` — routes: `document_routes.py`

| File | Role |
|------|------|
| `store.py` | `DocumentStore` — CRUD, FTS5 index, retrieval |
| `chunker.py` | Recursive splitting with parent tracking, paragraph-aware boundaries |
| `query_analyzer.py` | Intent + complexity detection |
| `scoring.py` | BM25 + density + span coverage |
| `query_expansion.py` | Synonyms, rephrasing |
| `topic_coverage.py` | Topic diversity across result set |
| `dedup.py` | Identical chunk removal |
| `span_filter.py` | Token-level filter for sensitive data |

**Tables:** `documents`, `document_chunks` (+ FTS5 virtual table on content).

Chunks preserve parent↔child links for context expansion. Stop word list (123 terms), AND-first tokenization for 3+ word queries, OR fallback for 1-2 words.

---

## VFS (File Index)

**Dir:** `augmentum/vfs/` — routes: `files_routes.py`

Unified file catalog across upload sources with pluggable adapters.

**Adapters** (`vfs/adapters/`):
- `uploads` — user file uploads (blob dedup)
- `media_server` — Audiobookshelf/Emby/Jellyfin/Komga/LibriVox/Suwayomi
- `documents` — RAG document chunks
- `images` — image generations + chat image cache
- `artifacts` — scratch artifacts
- `chat_images` — inline chat images
- `bookmarks` — web bookmarks

**Table:** `file_index` — id, user_id, source, source_id, name, **kind** (audio/video/code/document/image/archive/executable/data/other), mime_type, size_bytes, real_path, tags (JSON), is_directory, parent_id, source_metadata (JSON), is_favorite, is_trashed. FTS5 virtual table for name+description.

Favorite/trash are soft deletes; explicit purge required. Real_path is adapter-dependent (local, URL, resource handle).

---

## Voice Pipeline

**Files:** `augmentum/voice/*.py`
**WebSocket:** `/ws/voice`
**Routes:** `voice_routes.py`, `voice_enrollment_routes.py`, `audio_routes.py`

### Data flow
```
Browser mic → WS chunks → VAD (Silero) → STT (Deepgram/Moonshine)
    → Handler.process_stream() → sentence splitter → emotion extraction
    → TTS queue[(text, instruct)] → TTS provider → WS audio chunks → browser
```

### Details
- VAD: Silero, silence threshold 400-3000ms
- STT: streaming (Deepgram) or batch (Moonshine local, Whisper)
- TTS chunking: sentence/clause/full (via `voice_tts_chunking`)
- Emotion: `extract_emotion_instruct()` returns prosody for Qwen3-TTS
- PCM streaming built but disabled (`_STREAM_PCM_PROVIDERS = frozenset()`)

### Gotchas
- Moonshine: `base_url="builtin"` must NOT be passed to httpx
- Backchannel filter `_BACKCHANNEL_RE` — don't add common words like "thanks"
- Speaker verification in `voice_enrollment` table

### TTS provider adapter table
See SKILL.md — Chatterbox (no /v1), Fish Speech (/v1/tts), Qwen, Deepgram, ElevenLabs, OpenAI each have custom auth/endpoints.

---

## Image Pipeline

**Dirs:** `augmentum/image/`, routes: `image_routes.py`, `cloud_image_routes.py`, `chat_image_routes.py`

### Local
Stable Diffusion (diffusers), GGUF (stable-diffusion.cpp), FreeU/ToMe/CFG rescale, hires fix.

### Cloud providers
| Provider | Auth | Payload notes |
|----------|------|---------------|
| OpenAI | `Bearer` | `size: "1024x1024"` string. **No `negative_prompt`.** Only DALL-E 3 takes `quality`/`style` — NOT GPT-Image models |
| Together | `Bearer` | `width`/`height` as ints |
| Stability | `Bearer` | **Multipart form-data** (`files=`, NOT `data=`) |
| BFL | `x-key` | Async polling — submit → poll `/v1/get_result` |
| Fal | `Key` | `image_size: {width, height}` object |

### Narrative scene images
Auto-background during narrative chat. Distiller model condenses scene for prompt. `narrative_scene_context_rounds` controls how many prior messages inform the scene.

---

## Media System

**Dir:** `augmentum/media/` — routes: `media_routes.py`

| File | Role |
|------|------|
| `sync.py` | Pull media server catalog into `file_index` |
| `library_store.py` | Per-media-type classification views |
| `comic_series_store.py` | Comic series grouping + scan checkpoints |
| `playback_selection.py` | Smart resume (last-played, audio/subtitle routing) |
| `providers/` | Adapters: audiobookshelf, emby, jellyfin, komga, librivox, suwayomi |
| `receivers/` | DLNA: device discovery, playlist streaming, profile detection |

**Tables:** `user_media_servers`, `media_library_views`, `comic_series`, `comic_scan_checkpoint`. Catalog rows populate `file_index.source_metadata` with provider-specific fields (author, narrator, chapters, duration_ms, progress_pct, cover_url).

**LibriVox** is wired as a permanent built-in via sentinel `server_id='builtin-librivox'` (browse-live + pin-to-persist).

---

## Avatar System

**Dir:** `augmentum/avatar/` — routes: `avatar_routes.py`

| File | Role |
|------|------|
| `store.py` | `AvatarStore` — per-user_id CRUD |
| `bundled.py` | Bundled avatars under `user_id IS NULL` (visible to all) |

**Table:** `avatars` — id (`avt_<ts>_<hash>`), user_id (NULL for bundled), vrm_path, thumbnail_path, mannerisms (JSON), is_bundled, type, segmentation_data.

Mannerisms is loose JSON (user-customizable). 64% of avatar endpoints have no frontend caller — feature incomplete, consider before adding.

---

## Cast & Multi-Device

**Dir:** `augmentum/cast/` + `augmentum/devices/` + 15 `ui/cast-*/` surfaces — routes: `cast_*_routes.py`, `device_routes.py`

Cross-device bus + receiver surfaces. Cast tokens (in-RAM, 30-min TTL,
IP-bound) bridge browser ↔ TV ↔ phone to one user session. Voice WS at
`/ws/voice` opens a `VoiceFanout`; cast receivers subscribe via
`/voice/sessions/{voice_session_id}/stream` so the same TTS bytes mirror
to every subscriber without double-synth.

| File | Role |
|------|------|
| `devices/registry.py` | `DeviceRegistry` — driver lifecycle, persistence, capability dispatch (~688 LOC) |
| `devices/cast_tokens.py` | In-RAM token store (30-min TTL, IP-bound, single-session revocable) |
| `cast/output_store.py` | Render output bytes store (TTL 5 min, max 256 entries) |
| `cast/input_bridge.py` | Phone gamepad → container UInput pad routing |
| `voice/fanout.py` | TTS / WS fan-out to subscribers — zero refactor of emit sites |
| `cast/executors.py` | Per-kind dispatch (image / TTS / video → cast targets) |
| `cast/dispatcher.py` | Cross-mode output routing to cast surfaces |

**Couch co-op all 4 phases shipped** (migrations 229-231): anonymous QR
join, named guest profiles (host-scoped), device-fingerprint auto-reconnect
(`WARM_SLOT_TTL_S = 30.0`), per-guest save slots.

**Surfaces** (`ui/cast-*/`): cast-home (TV idle), cast-app/audio/video/comic/vrm
(per-kind players), cast-control (phone remote), cast-receiver (iframe shell),
cast-stage (editorial), cast-pair (QR pairing), cast-guest-join/guests
(co-op onboarding).

Android-TV native receiver: `augmentum/cast/android-tv-receiver/` (Kotlin,
BootReceiver + Discovery + PlaybackService + AugmentumTvBridge).

---

## Controllers

**Dir:** `augmentum/controllers/` — routes: `controllers_routes.py`

Per-user gamepad/HID remapping. Merges system-default layouts with
per-`(user, system)` overrides; any non-null user binding wins per action.
`pad_routing` strategy (`'index'` | `'firstpress'`) decides slot assignment
when multiple phones join.

| File | Role |
|------|------|
| `service.py` | `ControllerService` — merges defaults + per-user remap (~161 LOC) |
| `store.py` | CRUD for `(user_id, system_id)` rows (~163 LOC) |
| `defaults.py` | System-default profiles per console family |

**Table:** `controller_remaps` (user-scoped, migration 126).

No host HID daemon — the in-container `cast-input-bridge.py` writes virtual
UInput pads. Resolved layouts feed `input_bridge.py` slot assignment.
Cast-control phone WS is the producer side.

---

## Game Streaming

**Dir:** `augmentum/game_stream/` — routes: `game_stream_routes.py`

Emulator container orchestration with admission control. Replaces the
2-per-user flat cap with a credit-budget model (active=8, resident=16)
plus docker-pause primitive plus paused-stop watchdog.

| File | Role |
|------|------|
| `runtime.py` | Port pool, lifecycle, admission, container adapter delegation (~1027 LOC) |
| `runtime.py:176-204` | `_admit()` — enforces active + resident credit budgets |
| `runtime.py:529, 618` | `pause()` / `resume()` — cgroup freezer via docker pause |
| `docker_adapter.py` | aiodocker impl, agsp-streamed emulator entrypoint, per-`(user, emulator)` save mount (~741 LOC) |
| `lifecycle.py:45-95` | State machine — adds PAUSED with legal-transition matrix |

**Tables:** `game_stream_sessions`, `game_stream_worlds`, `game_stream_telemetry`,
`game_saves` (with `guest_profile_id` from migration 231 for couch co-op).

Routes: `POST /api/game-stream/sessions`, `GET /sessions/{id}/readiness`,
`POST /heartbeat`, `WS /signal/{id}`.

Integrates with: titles catalog (BIOS classifier, ROM upload), controllers
(remapping → input_bridge), cast couch co-op (phase-4 per-guest saves).

---

## Games Portal

**Dir:** `augmentum/games/` — routes: `games_routes.py`

Aggregates playable web games. Providers (`games/providers/`):
- `itch.py` — itch.io API
- `js13k.py` — JS13K competition entries (JS games < 13KB)

`GameBrowseResult` matches `media.BrowseResult` shape (unified UI). `play_mode`: `embed` (iframe) or `local` (download + extract). Stateless — no dedicated tables; catalog fetched on demand.

See `project_game_portal.md` in memory for the roadmap (gamepad API, Web Bluetooth, WebHID, vibration, local co-op).

---

## Jobs Queue

**Dir:** `augmentum/jobs/` — routes: `jobs_routes.py`

Restart-survivable background runner.

| File | Role |
|------|------|
| `runner.py` | `JobRunner` — poll pending → lookup handler → run with context → update status |
| `context.py` | `JobContext` — progress reporting, cancellation, thread pool |
| `handlers/` | Type-specific: `gutenberg_fetch.py` (LibriVox catalog), `media_sync.py` (media server sync) |

**Table:** `background_jobs` — id, user_id, job_type, payload (JSON), status, progress, stage, result, error, attempts, max_attempts, cancel_requested.

### Contract
- **Single-worker** (no semaphore concurrency — CPU/GPU-bound assumption)
- **Non-blocking is strict** — `test_jobs_responsiveness.py` enforces
- **Handlers idempotent** — on restart, `running` → `pending` requeues
- `cancel_requested` cooperative (check between chunks)

Use for transcription, subtitle gen, media catalog sync, long-running pipelines.

---

## MCP Integration

**Dir:** `augmentum/mcp/` — routes: `mcp_routes.py`

| File | Role |
|------|------|
| `client.py` | `MCPClient` — session manager, connection pool, tool discovery, timeouts |
| `bridge.py` | Augmentum tool ↔ MCP tool call translation |
| `server.py` | Stdio/pipe server spawning |

**Timeouts:** init + list_tools 30s, call_tool 60s.

**Gotchas:** stdio servers inherit parent env+cwd; blocks on stdout/stderr if buffers fill. Transient failures **not retried**. Tool call forwarding happens in coder's phase_act dispatch; `create_coder_tools()` auto-adds MCP tools.

---

## Powers System

**Dir:** `augmentum/powers/` — routes: `power_routes.py`

Capability packs (metadata-driven behavior biases) activated at safe checkpoints.

| File | Role |
|------|------|
| `controller.py` | Checkpoint-aware activation (pre_plan, post_write, verify_failed, pre_finish) |
| `manifest.py` | Manifest parsing (kind, activation_policy, activation_windows) |
| `models.py` | `PowerManifest`, `PowerFile` dataclasses |
| `registry.py` | Discovery + catalog |
| `state.py` | Runtime (activated powers per turn) |

**Kinds:** `guidance`, `verifier`, `workflow`, `integration`, `bridge`.
**Activation policy:** `manual`, `controller`, `model_request`, `explicit_only`.

**Native powers:** `browser-verification`, `changelog-documenter`, `contract-keeper`, `dependency-doctor`, `failure-triage`, `mcp-builder`, `migration-safety`, `multi-agent-review`, `multi-tenant-auditor`, `observation-keeper`, `performance-profiler`, `power-audit`, `power-forge`, `release-review`, `subagent-router`, `test-author`, `test-baseline-keeper`, `workspace-onboarding`.

**Persistence rules:**
- Controller-activated → transient (turn-scoped)
- User-pinned → persist via settings
- Pinned pre_plan powers suppress controller overlays at implementation; controller still works at verifier checkpoints

---

## Promises / Missions

**Dir:** `augmentum/promises/` — consumed by coder

Structured plan model replacing free-text `plan_steps`.

| File | Role |
|------|------|
| `runner.py` | `MissionRunner.run(promises, act_fn, verify_fns, replan_fn) → AsyncIterator[event]` |
| `models.py` | `Promise`, `PromiseContext`, `ActEvent`, `VerificationKind` |
| `parse.py` | Promise-string parsing / AST |
| `render.py` | Tree rendering for display |
| `verify.py` | Verifier predicates (file exists, test passes, artifact matches, …) |

**Execution:** DFS — children complete + verify before parent. On verify pass, cascade eligibility + allow replan. On fail, cascade rejection to parents, skip siblings.

**Persistence:** JSON blob in `coder_sessions.mission` (supersedes `plan_steps`).

---

## Discovery & Quality

**Dir:** `augmentum/discovery/` — routes: `discovery_routes.py`

Consumer-specific quality filtering + ranking.

| File | Role |
|------|------|
| `quality.py` | Consumer pipelines: `filter_and_rank`, `filter_for_llm`, `filter_for_video_ui`, `filter_for_images`, `filter_for_docs` |
| `recommender.py` | Frecency-based ranking + interest clustering |
| `feeds.py` | Feed source config + ingestion |
| `clustering.py` | Interest-based clustering |
| `distiller.py` | Narrative extraction + metadata |
| `frecency.py` | Frecency score (recency × frequency) |

**Tables:** `domain_reputation` (server-level — blacklist/whitelist + fetchability history).

86% of discovery endpoints have no frontend caller today — feature is partially built.

---

## Search & Tools

### Search pipeline (Analytical)
1. Query expansion (`search_expansion_enabled`) — LLM variants
2. SearXNG search — federated
3. Direct fetch — top URL content extraction
4. Credibility scoring — source ranking
5. Relevance filtering — cosine threshold

### Tool calling (3-tier)
1. **Native** — provider supports function calling
2. **Structured** — XML/JSON injected into prompt
3. **Text** — regex extraction

### Preferred sources
511 domain list. Configurable.

### CodeMind (frontend)
`ui/scripts/codemind.js` — tree-sitter AST parsing for workspace editor + chat code blocks. Lazy-loaded grammars (JS/TS/HTML/CSS/Python/JSON). See SKILL.md for API.

---

## Character Cards

**Routes:** `character_routes.py`, `persona_routes.py`

### Formats supported
TavernCard V1/V2/V3, JanitorAI, Chub API, Pygmalion, RisuAI (CharX / PNG tEXt).

### Import pipeline
1. `_normalize_card()` — detect format, extract data dict
2. `_map_fields()` — normalize names + strip HTML
3. `_build_char()` — assemble canonical object + lorebook
4. `_download_avatar()` — fetch URL → base64 data URI
5. `_upsert_char()` — write to `ui_characters`

### HTML stripper rules
- `<img>` → markdown `![alt](src)` (preserve card images)
- Strip `<style>` (JanitorAI garbage)
- Extract `background-image: url(...)` from inline styles
- `<a>` transparent (image links preserved, wrapping links dropped)
- Allow `http://`, `https://`, `data:image/` URLs

---

## Fabric / Federation

**Dir:** `augmentum/fabric/` — routes: `fabric_routes.py`

Default-off, opt-in household-grade federation. Paired-peer fabric routes
the 6 modalities (LLM, image, TTS, STT, knowledge search, cast/render) to
other Augmentum boxes via Ed25519-signed envelopes. Connect is a per-user
overlay on the same trust substrate (separate section).

| File | Role |
|------|------|
| `identity.py:46` | `FabricIdentity` — node_id + Ed25519 keypair, fingerprint = `SHA256:<32-hex>` |
| `peer_auth.py:74-126` | `PairRequest` signed envelope; 300s replay window |
| `peer_middleware.py:108` | `FabricPeerMiddleware` — verifies signed HTTPS outside AuthMiddleware |
| `coordinator.py` | Peer registry, heartbeat sweeper, latency EMA per-(peer, kind) |
| `director.py` | `RoutingDirector.maybe_route_*` — capability-aware dispatch |
| `protocol.py:57` | `FabricEnvelope` + 9 msg types (hello / heartbeat / ack / error / cancel_request / job_*) |
| `models/provider_registry.py:212` | `resolve_backend_with_fabric` — **23 call sites across 28 files** in narrative/coder/reasoning/tools/openai/anthropic/ollama/etc |

**Tables:** `fabric_nodes` (server-level — paired peer registry).

**Trust:** pinned Ed25519 fingerprints (not TLS); per-peer service users
(`fabric:<short-node-id>`) on receiver; cloud-backed LLM providers
explicitly NOT advertised over fabric (`extractors.py:42-67`) so a peer
can't spend another peer's API budget.

**Default OFF.** Identity isn't generated until `settings.fabric_enabled`
flips on (`lifespan.py:43`). See
[`integration-weave.md#4-the-federation-weave`](integration-weave.md#4-the-federation-weave).

---

## Connect (Calls & Messaging)

**Dir:** `augmentum/connect/` — routes: `connect_routes.py`

Per-user voice / video / text overlay on the fabric trust substrate. Full
substrate shipped 2026-06-02: calls, messaging, typing receipts, reactions,
mid-call video escalation, missed-call timer.

| File | Role |
|------|------|
| `protocol.py` | `CALL_PROTOCOL_VERSION=1`, msg/event format, 64KB cap |
| `hub.py` | `{user_id → list[_Attachment]}` in-memory presence + fan-out |
| `call_lifecycle.py` | Invite timer (60s default) → missed-call flip + notification |
| `call_store.py`, `message_store.py`, `contact_store.py` | Idempotent CRUD against `connect_*` tables |

**Tables (migration 219, all user-scoped):** `connect_contacts`,
`call_sessions`, `call_events`, `connect_threads`, `connect_messages`,
plus `connect_message_reactions` (migration 233).

**Verbs:** `invite, accept, decline, offer, answer, candidates` (batched),
`select_answer`, **`negotiate`** (mid-call video on/off), `hangup`, text
`send / delivered / read / delete / edit / react`, `typing_start / stop`.

UI: `ui/scripts/connect/` (13 modules — client, dialer, incoming-modal,
calls-panel, thread-panel, messages, outbox, broadcast, ringtone,
rate-toast, icons, ui).

Contacts use `peer_did = user@instance` (forward-compat with
`did:augmentum:<keyfp>`); auto-surfaces when both peers opt in
(mutual-enablement-as-consent). External adapters (Telegram/Discord/WA/
Slack/Gmail) not in-tree — see project memory `project_nanoclaw_adapter_strategy`.

---

## Community Marketplace

**Dir:** `augmentum/proxy/community_routes.py` — spec: `augmentumhq-site/docs/specs/community-install.md`

Deep-link install flow from `augmentumhq.com/community/<cat>/<slug>` →
local Augmentum. Four categories ship working: characters,
reasoning-flows (per-user installs); powers, knowledge packs (admin-only,
install-wide).

| File | Role |
|------|------|
| `community_routes.py:214` | `GET /community-install` — auth-exempt preview UI with inline login |
| `community_routes.py:315` | `POST /api/community/install` — auth-gated dispatch |
| `community_routes.py::_install_character` | → `character_routes._upsert_char(..., uid=user_id)` |
| `community_routes.py::_install_reasoning_flow` | → `flow_store.import_flow(..., user_id=user_id)` |
| `community_routes.py::_install_power` | Writes POWER.md to `{data_dir}/community-powers/<slug>/`, `power_registry.rescan()` |
| `community_routes.py::_install_knowledge_pack` | Async download via `httpx.stream`, SHA-256 verify, `pack_manager.scan()` |
| `auth/middleware.py::_PUBLIC_PATHS` | `/community-install` listed so cross-origin nav doesn't 401 |
| `powers/registry.py:25-36` | Search roots include `{data_dir}/community-powers/` (source_kind="community") |

**Table:** `community_installs` (user-scoped audit row per install, migration 236).

**Safety:** trusted-origin allowlist (`_BUILTIN_TRUSTED_ORIGINS` + admin
`community_trusted_origins`); SafeHttpClient for manifest + artifact
fetches; per-category JSON schema validation; powers require kebab-case
slug + valid kind enum; knowledge size cap (`community_max_pack_size_mb`).

---

## Bug Finder Agent

**Dir:** `augmentum/bug_finder/` — routes: `bug_finder_routes.py`

5-surface callable agent for detecting bugs in user code. Triggerable
from companion, chat, coder, MCP, or HTTP. Uses the subagent dispatch
substrate under the hood.

| File | Role |
|------|------|
| `orchestrator.py` | End-to-end run lifecycle (scan, score, verify, persist) |
| `agnostic_stage.py` | Bandit/Ruff/Semgrep rule descriptions + result normalization |
| `auto_suppression.py` | Pattern-memory based auto-suppression candidates |
| `verifier.py` | Disproof-oriented verifier (PoC required, model isolation) |
| `proxy/bug_finder_routes.py` | List/get/cancel runs, SSE stream of findings |
| `mcp/server.py:434` | `bug_finder_*` MCP tools — run / status / list |

**Tables (migrations 225-228, 232):** `bug_finder_runs`,
`bug_finder_findings`, `bug_finder_findings_normalized`,
`bug_finder_patterns`, `bug_finder_codebase_knowledge`, `bug_finder_tasks`.

**Integration:** surfaces in coder as `BugFinderRunTool` /
`BugFinderStatusTool`, emits signal_events for companion observation,
runs as a background job via `JobRunner`.

---

## Engine v2 (llama-server)

**Manager:** `augmentum/models/llama_server_manager.py`

Bundled `llama-server` binary as a subprocess. Built from upstream llama.cpp via `Dockerfile.llama-server`, copied into the main image by `Dockerfile.gpu`.

- **Version pin:** `LLAMA_SERVER_VERSION` at repo root (e.g. `b8733`)
- **Upgrade:** `./scripts/upgrade_llama_server.sh <tag>` (or `--latest`), rebuild augmentum image
- **Verify:** `docker exec augmentum-augmentum-1 llama-server --version`
- **Stress test after bump:** `python scripts/stress_test_families.py`

**Engine v1 is retired.** LLM serving runs through the bundled `llama-server`
subprocess (`augmentum/models/llama_server_manager.py`). Do NOT resurrect an
older serving fork to support new models — bump `LLAMA_SERVER_VERSION` instead.

### Reasoning parser dispatch
`augmentum/utils/thinking.py` detects model family from GGUF `general.architecture` or name substring, then dispatches to the right parser.

| Format | Models |
|--------|--------|
| `<think>…</think>` | DeepSeek-R1, Qwen3/3.5/3.6, most |
| `<\|channel\|>analysis<\|message\|>…<\|end\|>` | Gemma 3, GPT-OSS (symmetric) |
| `<\|channel>thought\n…<channel\|>` | **Gemma 4 (asymmetric — closer is NOT slash-variant of opener)** |

**Gemma 4 gotcha:** upstream llama.cpp strips channel tokens if `skip_special_tokens=True`. Keep it False on decode or you silently lose all reasoning. `stress_test_families.py` catches the regression.

---

## Resource Ledger

**Dir:** `augmentum/resource/` — routes: `resource_routes.py`

Cross-subsystem model tracking.

| File | Role |
|------|------|
| `ledger.py` | `ResourceLedger`, `TrackedModel`, `ModelProfile` — record loads, snapshot devices, profile resource needs, cleanup low-priority models |

**Tables:** `resource_snapshots` (timestamp, device_id, vram/ram_used, models_loaded), `resource_profiles` (model_name, subsystem, backend, vram/ram, quantization, family).

Populated by model load/unload events (LLM manager, image pipeline, TTS/STT). Queried before loading a new model to check headroom.

---

## Docker & Deployment

### Compose overlays
```
compose.yaml                  base: augmentum (with bundled engine v2) + searxng + executor
compose.gpu.yaml              + GPU + image gen
compose.kokoro.yaml           + Kokoro TTS
compose.qwen-tts.yaml         + Qwen3-TTS
compose.chatterbox.yaml       + Chatterbox voice clone
compose.speaches.yaml         + Speaches TTS
```

Inference is served by the bundled engine (llama-server subprocess managed by
`augmentum/models/llama_server_manager.py`). Users with their own external
Ollama / llama.cpp / OpenAI-compatible servers configure them in
**Settings > Manage Providers** instead of bundling a containerized backend.

### Setup wizard
- `setup.bat` (Windows) / `setup.sh` (Linux/Mac)
- Writes selected overlays to `.augmentum.conf`
- `start.bat` / `start.sh` reads conf, merges compose files

**Windows gotchas:**
- em dashes (`—`) render as `ΓÇö` in CMD — use `--`
- `sed -i` converts CRLF → LF, breaks CMD batch labels

### Service URL envs
All passed through `compose.yaml` as `${VAR:-}` (empty default):
- `AUGMENTUM_TTS_QWEN_URL`, `AUGMENTUM_TTS_KOKORO_URL`, `AUGMENTUM_TTS_MY_URL`, `AUGMENTUM_MEDIA_AUDIOBOOKSHELF_URL`, etc.
