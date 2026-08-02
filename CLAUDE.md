# Augmentum — Development Standards

Guidance for contributors (and AI coding assistants) working in this
codebase. If you use Claude Code, Cursor, or a similar agent, point it at
this file.

## Project Overview
Augmentum is a self-hosted personal AI platform — NOT a proxy. It serves its own
models (bundled `llama-server`, Slots A/B/C) and ships its own multi-surface frontend
over one shared memory + identity. It speaks OpenAI/Ollama/Anthropic-compatible APIs
so any client can plug in and any provider can be a backend — but Augmentum is the
destination, not a middleman: external services (Ollama, cloud providers, OpenWebUI)
are optional, interchangeable plumbing *beneath* it. The OpenAI-compatible proxy is
the project's origin and one ingress among ~1,400 routes — no longer its identity.
Python 3.11, SQLite (aiosqlite), vanilla JS UI, self-hosted via Docker Compose.
Multi-tenant, privacy-first.

## What Augmentum Is (scope — read before scoping any change)

The OpenAI-compatible API is the origin and one ingress among ~1,400 routes, NOT the
identity. Scale: ~95 route modules, 7 dispatch modes, ~70 chat tools, ~45 coder
agentic tools, ~55 intent verbs, 209 UI modules, ~40 subsystem packages, 300
migrations, 882 settings. A change in one corner routinely affects others (shared
memory, model serving, identity, event bus) — default to sweeping ALL relevant
surfaces, never one (see the "verify wide" discipline below).

- **Serving**: bundled `llama-server` (Slots A/B/C) + optional cloud/local backends,
  all behind one model registry (see Model Slots below).
- **Modes** (classifier → `handler_factory`): passthrough · analytical (UARF) ·
  narrative (RP: cards, groups, lorebook) · agentic (planner + promises) · coder
  (containerized) · companion pipeline · direct (raw pipe).
- **Coder**: a full IDE-agent — file read/write/edit/patch, shell, git, test, real
  Docker workspaces + port publishing, a 7-tool CDP browser, subagent dispatch,
  bug-finder, per-workspace permission policy, and live dev-server preview proxying.
- **Companion**: autonomous, not a chatbot — dispatches every mode as a
  subagent; owns primitives (browse/code_exec/files/game_agent/image_gen/memory/
  stt/tts/voice_mixer/xr_scene); runs tick verbs (affect/drive/energy/journal/
  scheduler) and behavior loops (initiative/sleep-wake/drift); presence; safety
  floor. Grows with the user. Default OFF.
- **Embodiment**: VRM 3D avatar (IK, Rapier ragdoll physics, lipsync, poses, spatial
  director), WebXR binding, voice pipeline (STT/TTS/wake-word/enrollment/mixing).
- **Tools**: web/`research`, python/math exec, artifacts (chart/doc/ebook/slides/
  sheet + exports), image (gen/search/convert/bg-remove), scheduling substrate,
  memory, vocab, lorebook, notify, offers.
- **Whole subsystems, each self-hosted**: calling + connect (WebRTC user-to-user
  voice/video calls + text threads) · fabric (cross-instance peer federation —
  your tablet borrows your tower's GPU) · Titles/AXF game platform + game_stream
  (AGSP cloud gaming) + game_agent (AI plays games via vision) · knowledge packs
  (offline Wikipedia/ZIM hybrid retrieval) · dream (persona introspection) ·
  personality · discovery · notifications · observation (cross-modal pattern
  memory) · cast-to-TV · learning (language learning) · media-server integration
  (Audiobookshelf/Emby/Jellyfin) · vfs · ocr · security · marketplace · library ·
  powers · projects · builds.
- **Sovereign & multi-tenant**: self-hosted, Docker Compose, per-user isolation,
  encrypted secrets, no telemetry.

## Build & Test
- `start.sh` / `start.bat` — start Docker services (no rebuild)
- `start.sh build` — rebuild and start
- `start.sh -d` — start detached
- Tests: `python -m pytest tests/ -x` (from venv)
- Lint: `ruff check augmentum/`

See `README.md` for first-time setup and `CONTRIBUTING.md` for the
contribution workflow.

## Feature Implementation Checklist

BEFORE writing any code for a new feature, work through this checklist explicitly:

### 1. Data Flow (where does state live?)
- [ ] Identify every piece of state the feature creates or modifies
- [ ] For EACH piece: is it persisted server-side (SQLite) or client-side (localStorage)?
- [ ] **Default to server-side persistence.** localStorage is a cache/backup only
- [ ] If a new DB table is needed: write the migration SQL FIRST
- [ ] If adding fields to existing tables: write the ALTER TABLE migration
- [ ] Verify FK constraints won't block inserts (sessions table issue — use `get_or_create_session`)

### 2. Save/Load Round-Trip
- [ ] For every field saved: trace the FULL path: UI → API → handler → persistence layer → DB
- [ ] For every field loaded: trace the FULL path: DB → persistence layer → handler → API → UI
- [ ] Test: does the data survive a server restart?
- [ ] Test: does the data survive a page refresh?
- [ ] If the feature uses `StateManager`: remember `.backend` is a @property, `._backend` also works via getattr

### 3. API Design
- [ ] New endpoints: define route, method, request/response schema
- [ ] Register the router in server.py if it's a new file
- [ ] Add to config_routes.py `_TOOL_SETTINGS` or `_STRING_SETTINGS` if user-configurable
- [ ] Settings must persist via the settings_store (SQLite key-value)

### 4. UI Integration
- [ ] Every UI control must save to the server, not just localStorage
- [ ] Use `escapeHtml()` for ALL user-provided text in template literals (prevents backtick injection)
- [ ] Wire change handlers that call the API immediately (or debounced)
- [ ] On page load: fetch from server first, localStorage as fallback only
- [ ] Large payloads: use lazy loading (metadata first, full data on demand)

### 5. Error Visibility
- [ ] Never use `contextlib.suppress(Exception)` on save/load paths — use try/except with `log.warning`
- [ ] Never use `log.debug` for failures the user needs to know about — use `log.warning`
- [ ] API endpoints should return meaningful error messages, not silent 200s

### 6. Edge Cases
- [ ] What happens on first use (no existing data)?
- [ ] What happens after server restart (state restored correctly)?
- [ ] What happens with concurrent requests?
- [ ] What happens when the backend model is unavailable?
- [ ] For narrative: what happens on branch/edit/delete messages?

## Working Discipline

Two rules that override convenience, learned the hard way from recurring
regressions in this codebase:

1. **Fix the CLASS, not the symptom.** When a bug is reported, do NOT patch only
   the exact instance named. Find the general class and fix it everywhere it
   manifests. Most recurring regressions here (STT capture, layout-on-rotate,
   web-search) came back because the first fix was a point-patch on one surface.
   Before claiming done, ask: "what other surfaces share this code path, and did
   I fix them too?"

2. **Never auto-select on the user's behalf — surface the choice.** Models,
   providers, voices, the companion buddy: when multiple options exist, the user
   picks. Default-to-first / silent auto-pick is a regression, not a convenience —
   the whole point of a sovereign platform is that the user is in control.

**Verify wide.** The surface here is large and highly interconnected (shared
memory, model serving, identity, event bus). Default to verifying connections
across the whole codebase, not one area — a change or check in one section
routinely affects others. Don't scope narrowly, conclude, then discover breakage
on a second pass.

## Architecture Patterns

### Auth & Data Isolation
- Multi-tenant: Argon2id passwords, opaque session tokens, raw ASGI middleware (`augmentum/auth/`)
- **Every user-scoped table** has a `user_id` column. Every CRUD function accepts `*, user_id: str = ""`
- Route handlers extract `user_id = request.scope.get("user").id` and pass it to every data call
- Handler caches key by `(user_id, session_id)` not bare `session_id`
- User-scoped tables have a `user_id` column (added via CREATE TABLE or ALTER TABLE
  ADD COLUMN in `augmentum/state/migrations/`). Run
  `python .claude/skills/augmentum-dev/scripts/audit.py` to check for drift between
  the migrations and any doc that enumerates them.
- Server-level tables (NOT scoped): providers, app_settings, managed_services, audio_providers, image_providers, knowledge_packs, settings, users, domain_reputation, resource_snapshots, resource_profiles, marketplace_listings
- **NEW FEATURES MUST**: add user_id column to new tables, accept user_id in store functions, pass user_id from routes

### Persistence
- Server-side state: SQLite via `aiosqlite`, migrations in `augmentum/state/migrations/`
- Session data: `ui_sessions` table (full chat trees as JSON blobs)
- Character cards: `ui_characters` table
- Narrative state: `narrative_memory` table + entity/fact/plot tables
- Settings: `settings` key-value table via `SettingsStore`
- Frontend loads via `GET /api/...`, saves via `POST/PUT /api/...`
- Bulk session sync: `POST /api/chats/sync` (only send sessions with tree data, not metadata stubs)

### Narrative Mode
- NarrativeEngine (per-session, cached in `app.state.narrative_engines`)
- NarrativeHandler (per-session, cached in `app.state.narrative_handlers`)
- Three-layer memory: STATE snapshot + MEMORY ledger + embedded archive
- `sync_to_state()` must be called before `save_narrative_state()` to flush engine state
- `get_or_create_session()` must be called before any INSERT into FK-constrained tables
- Refusal filtering: `_is_refusal_text()` uses compound phrase matching (not single keywords)

### Frontend
- Vanilla JS modules in `ui/scripts/`
- `escapeHtml()` escapes `<`, `>`, `&`, `"`, backticks, and `${` (template literal safe)
- Sessions loaded as metadata stubs on init, full data on demand via `ensureSessionLoaded()`
- `saveSessions()` debounces to server; `_flushActiveSession()` for immediate sync after messages
- `sendBeacon` on unload sends only the active session (stays under 64KB limit)

### Docker
- `compose.yaml` — core services (augmentum, searxng, executor)
- `compose.gpu.yaml` — adds GPU + image generation
- Overlay compose files for ollama, llamacpp, kokoro, speaches, etc.
- `start.sh` / `start.bat` — reads `.augmentum.conf` for compose file list

### Inference Engine (Engine v2)
LLM serving uses a bundled `llama-server` binary, managed as a subprocess
by `augmentum/models/llama_server_manager.py`. The binary is built from
upstream llama.cpp via `Dockerfile.llama-server` and copied into the main
augmentum image by `Dockerfile.gpu`.

**Bundled-model capability cards**: `docs/model-cards/` holds fast-reference
cards for the always-on substrate models (classifier, vision sibling, TTS,
STT, VAD) — modalities, function-calling, sampling, and the per-model
gotchas (Gemma's `<|think|>` token, mmproj-for-vision, SmolLM's CPU budget).
Read/update them when wiring or bumping a bundled model.

- **Version pin**: `LLAMA_SERVER_VERSION` at repo root. Dockerfile reads it
  as a build arg.
- **Upgrade**: `./scripts/upgrade_llama_server.sh <tag>` (or `.bat`). With
  `--latest` it pulls the newest release from GitHub. Rebuild the augmentum
  image after so it picks up the new binary.
- **Verify**: `docker exec augmentum-augmentum-1 llama-server --version`.

**Do NOT revive `services/engine/`** — that's Engine v1, retired to
research/. See `services/engine/README.md` for the retirement rationale.
Bumping the v1 vendored llama-cpp-python fork is not the way to support new
models; bump the pinned `LLAMA_SERVER_VERSION` instead.

### Model Slots (A/B/C) & Vision
Three named llama-server slots, each its own subprocess/port, never competing
for a model slot:
- **Slot A** — the primary engine (`AugmentumEngineBackend`, port 8091), the
  user's chat model.
- **Slot B** — `models/secondary_slot.py::SecondarySlot` (port 8094): a second
  user-chosen *chat* model kept resident, reached via an explicit registry pin.
- **Slot C** — `models/classifier_slot.py::ClassifierSlot` (port 8093): the
  managed, **runtime-switchable** small workhorse for the `classifier` +
  `utility` (and, when its model is VL+mmproj, **vision**) roles. Registers
  under the `"classifier"` backend key so `resolve_model_for_role` needs no
  change; **resident** (`idle_timeout=0`) for the 2.5s voice/architect budget.
  Swap its model from the model manager ("Classifier" per-row button →
  `POST /api/engine/v2/classifier/load`) with **no container recreate**.
  **Precedence**: an external Docker classifier (`AUGMENTUM_CLASSIFIER_BASE_URL`,
  `compose.classifier.yaml`) still WINS the `"classifier"` key — Slot C
  registers only if it's empty, so existing installs are untouched.
  Setup (`setup.sh`/`.bat`) offers a VRAM-aware model choice (Gemma-4-E2B/E4B
  /SmolLM2) written to `.env` `AUGMENTUM_CLASSIFIER_*` (+ `_SLOT_*`).
  Open gap: the managed slot loads a LOCAL gguf only (no `-hf` auto-pull) — the
  external container remains the install-time serving path; the managed slot's
  model lands via the model-manager download.

**Vision is a capability of the classifier**, not a separate model. `augmentum/
vision/router.py` prefers, in order, a VL primary → the classifier (Gemma with
mmproj, via `ClassifierVisionProvider` gated live on `is_vision_capable()`) →
the **SmolVLM CPU fallback**. SmolVLM is **retired from the default path**: it
no longer spawns at boot and only **lazily** starts when neither the primary
nor the classifier can see (the no-GPU tier), so it costs nothing on GPU boxes.
`vision_provider_enabled` now means "allow the CPU fallback" (default on); its
model path (default SmolVLM2-500M) is swappable. The unified caption profile
(`provider.py::_CAPTION_SAMPLING`, seed-pinned) + grounded SEES/MAIN prompt
(`base.py::_caption_prompt_for`) are shared by both captioners.

### Reasoning Parser (Thinking/Channel Extraction)
Hidden reasoning comes in at least three wire formats. `augmentum/utils/
thinking.py` dispatches to the right parser by model family.

- `<think>…</think>` — DeepSeek-R1 / DeepSeek-V2/V3 / Qwen3 / Qwen3.5 /
  Qwen3.6 / Hunyuan Hy3 / Nemotron 3 Nano / most models (symmetric)
- `<think>…</think>` — GLM-4.x / DeepSeek-V3.2 / DeepSeek-V4 (Pro/Flash) /
  MiniMax M2.x / EXAONE 4.x (**asymmetric** — opener in prompt prefix; only
  `</think>` arrives in the response stream)
- `[THINK]…[/THINK]` — **Mistral Magistral** (symmetric; bracketed SPECIAL
  TOKENS, not literal text — requires `mistral-common ≥ 1.8.5` in the GGUF
  tokenizer config or markers get stripped). Vanilla Mistral 7B/Small/Large
  do NOT emit these — only Magistral does.
- `<|channel|>analysis<|message|>…<|end|>` — Gemma 3 / GPT-OSS (symmetric)
- `<|channel>thought\n…<channel|>` — **Gemma 4** (asymmetric; closer is
  not the slash-variant of the opener, they're different strings)

Family is detected from GGUF `general.architecture` (preferred) or a
substring match on the model name. Add new families by editing
`_FAMILY_PARSERS` in `thinking.py`.

**`--jinja` is mandatory** for correct thinking-mode behavior. Without
it, llama-server's fallback chat template doesn't recognize newer
reasoning models' thinking delimiters and chain-of-thought leaks into
the visible response. Wired by default in `engine_use_jinja_template`;
also enables `--reasoning-format deepseek`, which extracts reasoning
into the OpenAI-compat `reasoning_content` field. Disable only if a
specific GGUF has a buggy embedded template.

**Asymmetric closer families**: GLM-4.x, DeepSeek V3.2/V4 (Pro/Flash),
MiniMax M2.x, and EXAONE 4.x all share the same trick — the chat
template puts the opening `<think>` tag in the prompt prefix, so the
response stream starts INSIDE a think block and only `</think>` ever
arrives in the visible stream. The parser handles this via
`_STARTS_THINKING_FAMILIES` in `thinking.py`: when the family matches,
ThinkingStreamBuffer initializes with `_inside_think=True`, mirroring
Ollama's GLM47Parser. If `</think>` never arrives at all (small
distilled variants like GLM-4.7-Flash sometimes drop it), the entire
response routes to reasoning rather than leaking to content.

The chat-composer thinking button is wired for every family that
consumes an `enable_thinking` chat-template kwarg: Qwen 3.x, GLM-4.x,
EXAONE 4.x, and Nemotron 3 Nano. UI detection is in
`ui/scripts/settings.js::detectThinkingSupport`; backend forwarding in
`augmentum/models/openai_compat.py::_template_thinking_override` and
`augmentum/models/llama_cpp.py::_chat_template_kwargs`.

**Gemma 4 gotcha**: upstream llama.cpp strips the channel tokens if
`skip_special_tokens=True`. Keep it False on the decode path or you'll
silently lose all reasoning extraction. See `scripts/stress_test_families
.py` — the harness will catch this regression automatically if it
reappears.

### Knowledge Packs
Offline reference corpora (Wikipedia, MDWiki, Stack Exchange, DevDocs)
attached to chats for grounded retrieval, plus a browseable surface in
the Browse panel. Two on-disk formats:

- **`.augpack`** — SQLite + sqlite-vec + FTS5. Created by import path
  (CSV/JSON/PDF/etc.) or by ZIM→augpack conversion. Searched via
  hybrid vector + FTS5 with RRF merge. Chunks are pre-extracted at
  ingest, so search is fast but content isn't standalone-browseable.
- **`.zim`** — Kiwix archives (Wikipedia/MDWiki etc.). Searched via
  libzim's native Xapian index. Articles are HTML so they render as
  full pages in the Browse panel via `/api/knowledge/zim/{pack}/{path}`
  (sandboxed iframe, link rewriter, themed reader-mode CSS).

Hybrid retrieval (`augmentum/knowledge/packs.py::PackManager.search`)
runs all available legs in parallel — augpack vector + FTS, ZIM
keyword — and merges via Reciprocal Rank Fusion. Cross-encoder rerank
optional. Per-mode injection toggles in chat (`knowledge_packs_*`).

ZIM passage extraction (`augmentum/knowledge/zim_reader.py`) does the
heavy lifting: strips MediaWiki chrome (script/style/infobox/navbox),
splits on h1-h6 into ~900-char passages so the per-mode budget fits
real content. Cached in a sidecar SQLite per pack.

Performance: result LRU cache + persistent passage cache + startup
model pre-warm reduce repeat-query latency from ~5s to <50ms. All
gated by `knowledge_search_cache_*` and `knowledge_passage_cache_*`
settings.

Failed conversions (stuck mid-embedding install jobs) are detected
at scan and surfaced via `GET /api/knowledge/packs[failed_conversions]`
with a Discard endpoint that cleans up the empty shell + progress
file (preserving the original .zim).

Eval harness at `tests/live/test_live_pack_quality.py` — opportunistic
canonical queries against installed packs, runnable via
`pytest tests/live/test_live_pack_quality.py --run-live -v`. Add cases
when new packs land or retrieval behavior changes.

### Stress Testing the Engine
`scripts/stress_test_families.py` loads one representative model from each
family bucket (dense, MoE, reasoning, Gemma 4, Qwen 3.5/3.6), generates a
canned prompt, and verifies that:

- The correct parser family was selected
- Reasoning was extracted for reasoning-capable models
- Control tokens never leak into visible content
- Tokens-per-second and TTFT are plausible

Run after any `LLAMA_SERVER_VERSION` bump to catch regressions before
shipping.

### Action Registry (Companion Primitive Verbs)

`augmentum/intent/` gives the assistant a library of composable
primitive verbs (`note.create`, `memory.save`, `navigate.open_surface`,
`control.stop`, …) that ship through three dispatch tiers:

1. **Tier 1 — regex match** on the raw transcript. Sub-100ms.
   Conversation-control words (`stop`, `bye`, `slower`) and the most
   common phrasings of each verb live here.
2. **Tier 2 — embedding similarity** (planned) for paraphrase tolerance.
3. **Tier 3 — LLM tool exposure**. Every action registered with
   `fanout.tier3=True` becomes a tool the model can invoke via
   function-calling, alongside `web_search` / `image_generation` /
   etc. Composition (e.g., `note.create + note.start_capture`) is the
   model's responsibility.

Voice route hook lives in `voice_routes.py::_maybe_dispatch_intent`,
called after STT, before backchannel-filter / UARF. Side-channel WS
emission for LLM-invoked actions drains
`ReferentCache.pending_surface_events` at `turn_complete` boundaries.

The full architecture, including the referent cache, capture-mode
state machine, and "how to add a new verb" cookbook, is in
`augmentum/intent/README.md`. Tests at
`tests/test_smoke_intent.py` + `tests/test_integration_intent_*.py`.

When adding a new primitive: register it in
`augmentum/intent/builtin/<surface>.py`, import the module from
`augmentum/intent/__init__.py`, and (if voice-visible) add its id to
`_VOICE_TOOLS` in `voice_routes.py`. Action handlers MUST check
`session.user_id` and refuse to write into the anon row.

### Companion Headless Agency
The companion works **headless-first**: it gathers with tools and
answers in words; `surface_emit` is reserved for actions whose point
is the user's screen (play, open-on-request).

- **The shared FC loop** is `augmentum/companion_runtime/native_loop.py`
  (`native_loop_events` — event generator consumed by companion-direct chat;
  voice consumption pending). Do NOT re-implement tool loops for new
  companion surfaces — consume the events.
- **New headless capability tool** (returns data into the loop): add its
  registry name to `CORE_TOOL_NAMES`; if running it means the companion "went
  somewhere," map it in `_TRAIL_KINDS` so "take me there"
  (`intent/builtin/trail.py`) can jump the user to it.
- **New perceived surface** (presence): one `reportAttention(topic,
  payload)` call at the client choke-point (`ui/scripts/
  architect-observer.js`) + one topic branch in
  `companion_runtime/presence_context.py::observe_attention`. Topics
  must match the `/api/architect/observe` allow-list prefixes.
- **New verb rule of thumb**: if the user wants information, the verb
  should return data (the loop narrates it); if the user wants something
  to happen on screen, `surface_emit`. Don't dispatch info asks from the
  architect router — its INFORMATION vs SCREEN ACTION rule REJECTs them
  to the conversational layer by design.
- **App menu (long-tail actions)**: before authoring a new verb for an
  arg-less, context-bound, on-screen action, register the existing
  button instead: `registerCommand({..., agent: {description, speak}})`
  in the surface module (one line; `when` guard = liveness). The
  `app.act` verb matches intent against the synced catalog
  (`augmentum/intent/app_menu.py`, closed-world utility pick, stakes
  cap `trivial_reversible`) and fires it via the `palette.run` channel.
  Surfaces whose `when` context flips must call `refreshAgentCatalog()`.
  Verbs keep the head of the distribution: frequent, argument-carrying,
  headless-capable, or data-returning.

## Code Style
- Python: ruff-compliant, no TCH rules
- `from __future__ import annotations` in all Python files
- Type hints on function signatures
- structlog for logging (`get_logger(__name__)`)
- JS: no framework, vanilla DOM manipulation, ES modules
