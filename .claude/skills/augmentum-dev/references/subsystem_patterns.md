# Augmentum — Subsystem Cookbooks

> Loaded on demand by the `augmentum-dev` skill. These are the per-subsystem
> "how to add a new X" recipes. The high-frequency cross-cutting contracts
> (4-layer settings wiring, route registration, migrations, multi-tenant
> isolation, DB safety, the post-implementation checklist) live in
> [`../SKILL.md`](../SKILL.md) — read that first; come here when your change
> touches one of the subsystems below.

---

## Docker Overlay Pattern

New services follow this structure:

```yaml
# compose.myservice.yaml
services:
  augmentum:
    environment:
      - AUGMENTUM_MY_SERVICE_URL=http://myservice:8080
  myservice:
    image: myservice:latest
    ports:
      - "7000:8080"
```

Then in `compose.yaml`, pass the env var through:
```yaml
environment:
  - AUGMENTUM_MY_SERVICE_URL=${AUGMENTUM_MY_SERVICE_URL:-}
```

And in `config.py`:
```python
my_service_url: str = ""
```

Setup wizard integration: add an option in `setup.bat` and `setup.sh` that writes the overlay filename to `.augmentum.conf`.

---

## Coder Mode Patterns

Coder is a full handler (`augmentum/modes/coder/handler.py` — the largest handler in the codebase, several thousand lines; decomposition is on the backlog) backed by a containerized workspace subsystem (`augmentum/coder/`). The loop is `plan → act → verify → respond`, streamed via `chat_egress`.

### Key moving parts

| Concern | File | What it owns |
|---------|------|--------------|
| Container lifecycle | `coder/containers.py` | `ContainerManager` — async Docker start/stop/exec; labels `augmentum.workspace=true` to survive server restart |
| Per-session state | `coder/state.py` | `CoderState` dataclass: phase enum, working_set, plan, mission, step outputs, turn_summaries (FIFO 10) |
| Workspace tree | `coder/snapshot.py` | Auto-refreshing tree with `[NEW] [MOD] [DEL]` markers, injected at turn-start + every 8 iters |
| Inline project digest | `coder/digest.py` | Inlines every file when workspace < ~40K tokens — model skips `dir_tree`/`file_read` ceremony on small projects |
| Semantic search | `coder/indexer.py` | sqlite-vec inside container's `/workspace/.augmentum/index.db` (100-line chunks, 20-line overlap) |
| Scratch pad | `coder/scratch.py` | `ScratchStore` — oversized tool outputs land here; model reads by key instead of stuffing context |
| Tool registry | `coder/tools.py` | `create_coder_tools()` — shell, file, git, search, MCP forwarded tools. Auto-adds read-before-edit enforcement |
| Phase loop | `modes/coder/phase_plan.py`, `phase_act.py` | Plan generation, act dispatch, verification integration, streak-break reflexion |

### Strategy selector

`CoderHandler` picks one of four strategies based on `AUGMENTUM_CODER_STRATEGY`:

- **`native`** — minimal Claude-Code/Qwen-Code parity loop (`_act_native`). Default.
- **`hybrid`** — 4-innovation rebuttal loop (`_act_hybrid`).
- **`canonical`** — Codex/Claude Code-style consensus loop (`_act_canonical`)
- **`legacy`** — Phase 1 fallback (no containers, passthrough). Still lives in `_legacy.py`

Terminates on: task-completion signal, consecutive no-op streak (triggers reflexion), hard iter cap.

### Cross-turn persistence

`CoderState.turn_summaries` carries short per-turn digests across user turns. Injected into next turn's system prompt as `<prior_turns>` block. Fixes "model re-reads same files every turn".

### Mission / Promises

Newer sessions plan with **structured promises** (see `augmentum/promises/`) instead of free-text steps:

```python
from augmentum.promises.runner import MissionRunner
from augmentum.promises.models import Promise, VerificationKind

# Each promise has: statement, verify_kind (OUTPUT_CONTAINS, FILE_EXISTS, TEST_PASSES, ...),
# children (DFS: children run before parent).
runner = MissionRunner()
async for event in runner.run(promises, act_fn, verify_fns, replan_fn):
    ...
```

Stored as JSON blob in `coder_sessions.mission`. DFS execution: children must complete + verify before parent becomes runnable. Failure cascades to siblings.

### Powers (capability packs)

`augmentum/powers/` — metadata-driven behavior packs (guidance, verifier, workflow, integration, bridge). A `PowerController` engages them at safe checkpoints (`pre_plan`, `post_write`, `verify_failed`, `pre_finish`). Controller-activated powers are **transient** (turn-scoped); user-pinned powers **persist** via settings.

Built-in powers include: `browser-verification`, `contract-keeper`, `failure-triage`, `migration-safety`, `test-author`, `release-review`, `power-forge`. Add new packs in `powers/` with a manifest describing kind + activation_windows.

### Reviewing / permissions

- `coder_review_routes.py` — diff review queue (accept/reject hunks per file)
- `coder_permission_routes.py` — path-prefix allowlist/denylist enforced inside `tools.py`

### WebSocket live terminal

`WEBSOCKET /ws/terminal/{workspace_id}` — live stdout/stderr stream from the container. `Ctrl+C`/`Esc` cancellation is wired all the way to `docker exec` (fixed in commit `20ddec1`).

---

## Background Jobs Pattern

Restart-survivable async queue at `augmentum/jobs/` + `background_jobs` table. Use for long-running CPU/GPU/I-O-bound work (book transcription, media catalog sync, subtitle generation).

```python
from augmentum.jobs.runner import get_job_runner
from augmentum.jobs.context import JobContext

# 1. Define handler (async, receives JobContext):
async def my_handler(ctx: JobContext) -> dict:
    payload = ctx.payload  # JSON payload from submission
    for i, item in enumerate(items):
        if ctx.cancel_requested:
            return {"status": "cancelled"}
        await ctx.report_progress(i / len(items), stage="processing")
        # blocking work MUST go through ctx.run_in_thread():
        result = await ctx.run_in_thread(lambda: heavy_cpu_work(item))
    return {"processed": len(items)}

# 2. Register in handlers/__init__.py:
HANDLERS["my_job_type"] = my_handler

# 3. Submit via POST /api/jobs/submit with {"job_type": "my_job_type", "payload": {...}}
```

Rules:
- **Single-worker only**. No concurrency — if you need it, batch inside the handler.
- **Handlers must be idempotent** (or keep a checkpoint in `payload`) — on restart, status `running` → `pending` requeues them.
- **Non-blocking contract is strict**: any unwrapped blocking call freezes the FastAPI loop. `test_jobs_responsiveness.py` enforces this.
- `cancel_requested` is cooperative (check between chunks).

First consumers: `gutenberg_fetch` (LibriVox catalog pull), `media_sync` (Audiobookshelf/Emby/etc. → file_index).

---

## MCP Tool Bridge Pattern

`augmentum/mcp/` forwards tool calls to external MCP servers. Lifecycle:

```python
from augmentum.mcp.client import MCPClient, StdioServerParameters

client = MCPClient()
await client.add_server(
    "my-server",
    StdioServerParameters(command="npx", args=["-y", "@my/mcp-server"], env={...}),
)
# Tools now appear in client.list_tools() — coder.tools.create_coder_tools() auto-adds them.
result = await client.call_tool("my-server", "my_tool", {"arg": "value"})
```

Timeouts: init+list_tools 30s, call_tool 60s. Transport: stdio (inherits parent env+cwd) or pipe. Blocks on stdout/stderr if buffers fill — keep server output tight. Transient failures are **not retried**; caller handles.

Server configs live in user settings or workspace manifest. Cleanup via `_cm_exit`/`_session_exit` refs when the `ClientSession` context manager exits.

---

## Reasoning Flow Pattern

`augmentum/reasoning/` — user-defined step pipelines for analytical mode. Each step has a role (`classify`, `search`, `verify`, `respond`) and a template with variable slots (`$QUERY`, `$SEARCH_RESULTS`, `$DOCUMENT_CHUNKS`, …).

```python
from augmentum.reasoning.executor import execute_flow_stream

async for chunk in execute_flow_stream(flow, request):
    yield chunk  # InternalStreamChunk
```

Steps gate on a complexity predicate — trivial queries skip search+verify, complex ones run the full chain. Variable substitution happens **before** LLM invocation (not runtime binding). Custom flows stored in `custom_flows` (user-scoped); bundled flows in `reasoning/templates.py` (global via `user_id IS NULL`).

---

## Engine v2 (llama-server) Pattern

LLM serving uses a bundled `llama-server` binary as a subprocess managed by `augmentum/models/llama_server_manager.py`. Built from upstream llama.cpp via `Dockerfile.llama-server`, copied into the main image by `Dockerfile.gpu`.

- **Version pin**: `LLAMA_SERVER_VERSION` at repo root (e.g. `b8733`)
- **Upgrade**: `./scripts/upgrade_llama_server.sh <tag>` or `--latest`, then rebuild augmentum image
- **Verify**: `docker exec augmentum-augmentum-1 llama-server --version`
- **Stress test after bump**: `python scripts/stress_test_families.py` — loads one model per family bucket, verifies parser family selection and reasoning extraction

**Do NOT revive `services/engine/`** — Engine v1 is retired to research/. Support new models by bumping `LLAMA_SERVER_VERSION`, not the v1 fork.

### Reasoning (thinking) parser dispatch

`augmentum/utils/thinking.py` dispatches by model family. Three wire formats:

| Format | Models | Notes |
|--------|--------|-------|
| `<think>…</think>` | DeepSeek-R1, Qwen3, Qwen3.5, Qwen3.6 | Most common |
| `<\|channel\|>analysis<\|message\|>…<\|end\|>` | Gemma 3, GPT-OSS | Symmetric |
| `<\|channel>thought\n…<channel\|>` | **Gemma 4** | **Asymmetric** — closer is NOT a slash-variant of the opener. `skip_special_tokens` MUST be False on the decode path or all reasoning is lost. |

Family detected from GGUF `general.architecture` (preferred) or model name substring. Add families by editing `_FAMILY_PARSERS` in `thinking.py`.

---

## CodeMind — AST-Powered Code Intelligence

`ui/scripts/codemind.js` provides tree-sitter AST parsing for the workspace editor and chat code blocks. Grammars lazy-load from CDN (JS, TS, HTML, CSS, Python, JSON).

**Key API:**
- `init()` — lazy-load tree-sitter (UMD script injection, not ESM import)
- `parse(code, lang, fileKey)` — full parse with incremental caching
- `parseSync(code, lang, fileKey)` — sync parse if grammar already loaded
- `getErrors(code, lang)` — syntax errors with line/col positions
- `findBracketMatch(code, row, col, lang)` — AST-aware bracket matching
- `getScopeAt(code, row, lang)` — extract enclosing function/class + imports (for LLM context compression)
- `getDeclarations(code, lang)` — top-level names + signatures (for file summaries)
- `validate(code, lang)` — check LLM output for syntax errors before showing to user

**Integration points:**
- Workspace: real-time diagnostics on edit (debounced 150ms), error gutter, bracket matching, fold regions, symbol outline
- Autocomplete: AST declarations prioritized over regex word matching
- Chat: pre-execution validation for Python blocks, silent lint pre-check
- LLM context: `getScopeAt()` compresses context to current function + imports (40-60% token savings)

**Graceful degradation:** All features fall back if CodeMind unavailable (CDN offline, grammar not loaded).

---

## TTS Provider Pattern

Audio providers are stored in SQLite `audio_providers` table and auto-registered on startup from env vars.

**Two built-in (in-process, no sidecar) TTS engines** — `_BUILTIN_TTS_IDS` in `audio_routes.py`:
- `kokoro-builtin` — `augmentum/voice/kokoro_tts.py` (`KokoroTTS`), via `kokoro-onnx`. 54 voices, the bundled default (`tts_kokoro_builtin=True`). Quality tier. Has HBE upsampling + prosodic steering + the voice-walk cloner.
- `pockettts-builtin` — `augmentum/voice/pocket_tts.py` (`PocketTTS`), via the `pocket-tts` PyPI package (Kyutai). ~100M params / ~236MB weights, 6 languages (en/fr/de/it/pt/es), 8 named voices, voice cloning from short reference clips. CPU-real-time. Off by default (`tts_pocket_builtin=False`, `AUGMENTUM_TTS_POCKET_BUILTIN=true` to enable). Cached under `~/.cache/pocket_tts` (or `tts_pocket_model_dir`).

Both share the shape `instance()` / `load_model()` / `is_available` / `get_voices()` / `_resolve_voice()` (handles `name`, `a+b` blends, `a*0.7+b` weighted, `walk:` walked voices) / `stream_speech()` / `generate()`. Route code dispatches via `_builtin_tts_engine(provider_id)` (in `audio_routes.py`) — used by `tts_speech`, `tts_speech_routed`, `tts_synthesize_bytes`, `combine_voices`. The narration synth (`narration_synth.py::_resolve_synth_engine`) and the TTS-studio long-text path work with either. **Voice blending** works on both (same 256-dim style-vector averaging, per-family — you can't blend across engines). **Voice-walk cloning** (`voice_walk.py`) is Kokoro-only for now — it reaches into the underlying `kokoro-onnx` object's `create`/`get_voice_style`; generalising it is a follow-up. The Settings → Voice page's *inline* mixer is also Kokoro-only pending migration onto the reusable `ui/scripts/voice-mixer.js` (which IS engine-aware — used by the TTS studio).

**Adding a new TTS provider:**

1. **Compose overlay** (`compose.myservice.yaml`):
```yaml
services:
  augmentum:
    environment:
      - AUGMENTUM_TTS_MY_URL=http://myservice:8080
  myservice:
    image: ...
```

2. **Config** (`config.py`): `tts_my_url: str = ""`
3. **Env passthrough** (`compose.yaml`): `- AUGMENTUM_TTS_MY_URL=${AUGMENTUM_TTS_MY_URL:-}`
4. **Auto-register** (`server.py` `_BUNDLED` list): Add entry with `url_setting: "tts_my_url"`
5. **Bundled IDs** (`audio_routes.py` `_BUNDLED_IDS`): Add `"my-tts"`

**Provider-specific endpoint paths:**
- Most providers: `POST /v1/audio/speech` (OpenAI-compatible)
- travisvn/chatterbox: `POST /audio/speech` (no `/v1` prefix) — detected via `provider_id == "chatterbox-tts"`
- Fish Speech: `POST /v1/tts` with different payload format — detected via `_is_fish_provider()`
- Voice listing tries: `/v1/audio/voices`, `/v1/voices`, `/voices` (fallback chain)

**Provider detection helpers** in `audio_routes.py`:
- `_is_chatterbox_provider()`, `_is_fish_provider()`, `_is_qwen_provider()`
- `_is_deepgram()`, `_is_elevenlabs()`, `_is_openai_tts()`
- Each has custom auth headers, endpoint paths, and payload formats

**Voice cloning** uploads to `/v1/voices` (or `/voices` for Chatterbox standard). Clone-capable providers: Chatterbox, Chatterbox Turbo.

---

## Image Provider Pattern

Cloud image providers stored in `image_providers` table. Each provider has different auth, endpoints, and payload formats.

**Auth header format varies:**
- OpenAI/Stability/Together: `Authorization: Bearer {key}`
- BFL: `x-key: {key}`
- Fal: `Authorization: Key {key}`

**Payload format varies:**
- OpenAI: JSON with `size: "1024x1024"` (string)
- Together: JSON with `width`/`height` as separate integers
- Stability: Multipart form-data (use `files=` with tuples, NOT `data=`)
- BFL: JSON, async polling (submit → poll `/v1/get_result`)
- Fal: JSON with `image_size: {width, height}` object

**Important:** OpenAI does NOT support `negative_prompt`. Only send `quality`/`style` for DALL-E 3, not GPT-Image models.

---

## Code Editor Patterns

### Workspace Editor Architecture
Textarea + Prism.js overlay. **Critical:** Every text-layout CSS property must match between `.workspace-code` (textarea) and `.workspace-code-highlight` (pre). The Prism CDN theme must be overridden with `!important` on all font/spacing properties — but NOT `padding` on the `<pre>` (it must match the textarea's 8px 12px).

### Keyboard Shortcuts
All shortcuts are in the `_el.code?.addEventListener('keydown', ...)` handler in `workspace.js`. Chat code edit mode shares the same shortcuts (duplicated in `chat.js:_toggleCodeEdit`).

### Quick Actions
Defined in `_QUICK_ACTIONS_CATEGORIES` (chat.js). Each category has an icon, name, and array of `{label, instruction, mode}` objects. Language-specific extras in `_QUICK_ACTION_LANG_EXTRAS` (key = language, value = array or alias string).

### Auto-Fixers
In `chat.js`: `_fixHTML()`, `_fixCSS()`, `_fixJavaScript()`, `_fixPython()`, `_fixJSON()`. Each returns `{code, fixed, changes}`. `_silentLint()` runs CodeMind validation first, then the appropriate fixer. JS fixer excluded from silent path (async CDN dependency).

### Diff Rendering
`_renderDiffLines()` (chat.js) and `_renderHunkLines()` (workspace.js) both:
- Show dual line numbers (old/new columns)
- Compute intra-line character-level diffs for paired remove→add lines
- Use `diff-highlight` class for changed characters within a line

### Ghost Text
`_triggerGhostText()` fires on edit (600ms debounce), calls `/api/chat` with cursor context, renders suggestion in Prism overlay. Setting: `ghost_text_enabled` (bool) + `ghost_text_model` (string, empty = current model). Tab accepts, Escape/typing dismisses.

---

## Testing Patterns

### Three-Tier Test Architecture

Every module should have coverage at one or more tiers:

**Tier 1 — Smoke Tests** (`tests/test_smoke_*.py`): Module imports, class construction, basic operation. Fast (<100ms).

**Tier 2 — Contract & Integration Tests** (`tests/test_contract_*.py`, `tests/test_integration_*.py`):
- **Contract**: External service boundaries — verify request/response shapes with mocked HTTP
- **Integration**: Cross-subsystem chains — handler → tools → LLM → state

**Tier 3 — Live Tests** (`tests/live/test_live_*.py`): Real Docker services, real APIs. `@pytest.mark.live`, auto-skipped unless `--run-live`.

### Test Patterns Quick Reference

**Route test** (sync, TestClient):
```python
def test_list_items(sqlite_client):
    resp = sqlite_client.get("/api/items/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)
```

**Contract test** (mock external HTTP):
```python
def test_ollama_chat_request_shape(mock_backend):
    with patch("augmentum.models.ollama.httpx.AsyncClient") as mock_client:
        # Verify we send correct body shape to Ollama
        ...
```

**State test** (async, real SQLite):
```python
@pytest.mark.asyncio
async def test_settings_roundtrip():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    store = SettingsStore(backend)
    await store.set("key", "value")
    assert await store.get("key") == "value"
```

**Live test** (requires running services):
```python
@pytest.mark.live
@pytest.mark.asyncio
async def test_real_ollama_chat(live_ollama):
    resp = await live_ollama.post("/api/chat", json={...})
    assert resp.status_code == 200
```

### Adding Tests for New Modules

When creating `augmentum/foo/bar.py`, also create `tests/test_bar.py` (or `tests/test_foo_bar.py`). The `test_coverage.py` scanner enforces this. At minimum:
1. Smoke test: module imports, primary class constructs
2. Contract test: if the module calls external services
3. Round-trip test: if the module persists data

### Shared Fixtures (tests/conftest.py)

| Fixture | Use for |
|---------|---------|
| `client` | Stateless route tests |
| `sqlite_client` | Stateful route tests (fresh :memory: DB) |
| `mock_backend` | MockOllamaBackend instance |
| `mock_docker` | AsyncMock aiodocker.Docker |
| `load_fixture` | Load canned JSON from tests/fixtures/responses/ |
| `audio_silence` | 1s PCM silence for voice tests |
| `audio_tone` | 1s 440Hz sine for voice tests |

### Running Tests

```bash
pytest tests/ -x                    # all tests (no live)
pytest tests/ --run-live             # include live tests
pytest tests/ -m contract            # contract tests only
pytest tests/test_foo.py -x -v -s    # single file, verbose
```

---
