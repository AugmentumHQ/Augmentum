# Augmentum Code Patterns Reference

## Pattern 1: Setting Addition (4-Layer Wiring)

Every setting touches 4 files. Miss one and it silently breaks.

| Layer | File | What to add |
|-------|------|-------------|
| **Default** | `augmentum/config.py` | `my_setting: type = default` in Settings class |
| **Validation** | `augmentum/proxy/config_routes.py` | `_TOOL_SETTINGS["my_setting"] = (type, min, max)` or `_STRING_SETTINGS["my_setting"] = max_len` |
| **Restore** | `augmentum/proxy/server.py` | `_SETTINGS_RESTORE_MAP["my_setting"] = type_caster` |
| **Frontend** | `ui/scripts/settings.js` | DEFAULTS + loadToolSettingsFromBackend + syncToolSettingsToBackend + UI control |

**Type casters in _SETTINGS_RESTORE_MAP:**
- `bool` → use `_parse_bool` (handles string "true"/"false")
- `int` → use `int`
- `float` → use `float`
- `str` → use `str`

**Name mapping:** Python `snake_case` → JS `camelCase`
- `my_feature_enabled` → `myFeatureEnabled`
- Exception: some legacy settings don't follow this (check existing mappings)

## Pattern 2: Route Registration

```python
# 1. Define in augmentum/proxy/my_routes.py:
router = APIRouter(prefix="/api/myfeature", tags=["myfeature"])

# 2. Import in server.py (~line 1601):
from augmentum.proxy.my_routes import router as my_router

# 3. Register in server.py (~line 1631):
app.include_router(my_router)
```

## Pattern 3: Migration Creation

File: `augmentum/state/migrations/NNN_description.sql`
- Always `IF NOT EXISTS` for new tables
- `ALTER TABLE` for existing tables (may fail silently if column exists — that's OK)
- Runs on startup in alphabetical order
- FK to `ui_sessions` → call `get_or_create_session()` before INSERT

## Pattern 4: Session Persistence

```python
# Correct — ensures session exists before FK insert:
await get_or_create_session(backend, session_id)
await backend.conn.execute(
    "INSERT INTO my_table (session_id, ...) VALUES (?, ...)",
    (session_id, ...),
)
```

## Pattern 5: Handler Cache (Per-Session Singletons)

```python
# In server.py or handler_factory.py:
if session_id not in app.state.my_handlers:
    app.state.my_handlers[session_id] = MyHandler(...)
handler = app.state.my_handlers[session_id]

# Use OrderedDict for LRU-like eviction:
from collections import OrderedDict
app.state.my_handlers = OrderedDict()
```

## Pattern 6: Frontend Save/Load

```javascript
// Load: server first, localStorage fallback
async function loadFromServer() {
  try {
    const resp = await fetch('/api/my-feature');
    if (resp.ok) return await resp.json();
  } catch {}
  // fallback
  return JSON.parse(localStorage.getItem('my-feature') || 'null');
}

// Save: always to server, localStorage as backup
async function saveToServer(data) {
  localStorage.setItem('my-feature', JSON.stringify(data));
  await fetch('/api/my-feature', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}
```

## Pattern 7: Template Literal Safety

```javascript
// ALWAYS escape user text:
el.innerHTML = `<div class="name">${escapeHtml(userName)}</div>`;

// Safe without escaping (not user data):
el.innerHTML = `<div class="count">${messages.length}</div>`;
el.innerHTML = `<div>${icons.search}</div>`;
```

`escapeHtml()` handles: `<` `>` `&` `"` `` ` `` `${`

## Pattern 8: Docker Compose Overlay

```yaml
# compose.myservice.yaml
services:
  augmentum:
    environment:
      - AUGMENTUM_MY_URL=http://myservice:8080
  myservice:
    image: myorg/myservice:latest
    ports:
      - "7000:8080"
```

Passthrough in `compose.yaml`:
```yaml
environment:
  - AUGMENTUM_MY_URL=${AUGMENTUM_MY_URL:-}
```

## Pattern 9: Streaming Response

```python
# Ollama-style NDJSON:
async def stream_ndjson():
    async for chunk in upstream:
        yield json.dumps(chunk) + "\n"

# OpenAI-style SSE:
async def stream_sse():
    async for chunk in upstream:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
```

## Pattern 10: Voice Pipeline Queue

```python
# TTS queue carries (text, instruct) tuples for atomic pairing:
tts_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()

# Producer (per sentence):
instruct = extract_emotion_instruct(sentence, entity_state)
await tts_queue.put((sentence, instruct))

# Sentinel to signal end:
await tts_queue.put(None)
```

## Pattern 11: Narrative Memory Sync

```python
# MUST call sync_to_state() before save_narrative_state()
# to flush engine state into the DB-friendly format:
engine.sync_to_state()
await handler.save_narrative_state(session_id)
```

## Pattern 12: Settings Store (KV Persistence)

```python
# Read:
value = await settings_store.get("my_key")

# Write:
await settings_store.set("my_key", json.dumps(data))

# The settings_store wraps the `settings` SQLite table.
```

## Pattern 13: ASGI Middleware (Not BaseHTTPMiddleware)

```python
# BaseHTTPMiddleware breaks WebSocket connections.
# Use raw ASGI for middleware that must work with WebSocket:

class MyMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # modify headers, etc.
            pass
        await self.app(scope, receive, send)
```

## Pattern 14: Bundled Service Auto-Registration

```python
# In audio_routes.py or similar:
_BUNDLED_IDS = {"ollama", "llamacpp", "kokoro", "chatterbox-turbo", "qwen-tts"}

# Dynamic SQL for bundled services:
placeholders = ",".join("?" * len(_BUNDLED_IDS))
query = f"SELECT ... WHERE id NOT IN ({placeholders})"
```

## Pattern 15: Refusal Detection

```python
from augmentum.modes.narrative.handler import _is_refusal_text

# Uses compound phrase matching — NOT single keywords:
# "I cannot" alone won't match, but "I cannot generate" will.
# This prevents false positives on legitimate text containing refusal words.
```

## Pattern 16: User-Scoped CRUD (Explicit)

THE pattern for every user-scoped table. `user_id` is keyword-only to prevent positional mixups at the call site.

```python
# SELECT — append filter only when user_id is set (empty → system/legacy access):
async def get_item(self, item_id: str, *, user_id: str = "") -> dict | None:
    query = "SELECT * FROM items WHERE id = ?"
    params: list = [item_id]
    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    async with self._db.execute(query, params) as cur:
        return await cur.fetchone()

# INSERT — column list is dynamic:
async def create_item(self, item_id: str, data: str, *, user_id: str = "") -> None:
    cols = ["id", "data"]
    vals: list = [item_id, data]
    if user_id:
        cols.append("user_id")
        vals.append(user_id)
    placeholders = ",".join("?" * len(cols))
    await self._db.execute(
        f"INSERT INTO items ({','.join(cols)}) VALUES ({placeholders})", vals
    )

# Shared/builtin rows — user's own + global (user_id IS NULL):
async def list_flows(self, *, user_id: str = "") -> list[dict]:
    if user_id:
        rows = await self._db.fetch_all(
            "SELECT * FROM flows WHERE user_id = ? OR user_id IS NULL", [user_id]
        )
    else:
        rows = await self._db.fetch_all("SELECT * FROM flows", [])
    return rows

# Route handler — extract at the boundary:
def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""
```

Run `red_team_scan.py` after any change that touches a user-scoped table — it catches missing `AND user_id = ?` filters.

## Pattern 17: Background Job Submission

```python
# 1. Submit from a route:
import uuid
job_id = uuid.uuid4().hex[:12]
await backend.execute(
    "INSERT INTO background_jobs (id, user_id, job_type, payload, status, created_at) "
    "VALUES (?, ?, ?, ?, 'pending', datetime('now'))",
    (job_id, user_id, "my_job_type", json.dumps(payload)),
)

# 2. Handler (non-blocking, idempotent):
async def my_handler(ctx: JobContext) -> dict:
    for i, item in enumerate(ctx.payload["items"]):
        if ctx.cancel_requested:
            return {"status": "cancelled", "processed": i}
        await ctx.report_progress(i / len(ctx.payload["items"]), stage="processing")
        result = await ctx.run_in_thread(lambda: heavy_cpu_work(item))
    return {"processed": len(ctx.payload["items"])}

# 3. Register in augmentum/jobs/handlers/__init__.py:
HANDLERS["my_job_type"] = my_handler
```

**Never** call blocking code directly in a job handler — it freezes the FastAPI loop. Use `ctx.run_in_thread()` or `asyncio.create_subprocess_exec()`.

## Pattern 18: Workspace Snapshot (Coder)

```python
from augmentum.coder.snapshot import WorkspaceSnapshot

snapshot = WorkspaceSnapshot(workspace_path)
await snapshot.refresh()  # auto-detects [NEW]/[MOD]/[DEL] since last refresh
tree_text = snapshot.render(max_files=60, max_depth=5)
# → injected into the coder system prompt at turn-start + every 8 iters
```

Small-project shortcut: if the full project digest fits (<40K tokens), inline every file with `augmentum/coder/digest.py` — the model then skips dir_tree/file_read exploration entirely.

## Pattern 19: Reasoning Parser Dispatch

```python
from augmentum.utils.thinking import split_thinking

# Family auto-detected from GGUF metadata or model name:
visible_text, hidden_reasoning = split_thinking(raw_output, model_name=name)
```

Three wire formats. Adding a new family = edit `_FAMILY_PARSERS` in `augmentum/utils/thinking.py` and add a test model to `scripts/stress_test_families.py`.

**Gemma 4 trap:** closer is NOT `</|channel>` — it's `<channel|>` (asymmetric). `skip_special_tokens=True` silently drops the markers; keep it False on decode.

## Pattern 20: Mission / Promise Plan

```python
from augmentum.promises.models import Promise, VerificationKind
from augmentum.promises.runner import MissionRunner

mission = [
    Promise(
        statement="Create auth/routes.py with /login endpoint",
        verify_kind=VerificationKind.FILE_EXISTS,
        verify_target="auth/routes.py",
        children=[
            Promise(
                statement="Add pytest for login endpoint",
                verify_kind=VerificationKind.TEST_PASSES,
                verify_target="tests/test_auth_login.py",
            ),
        ],
    ),
]

async for event in MissionRunner().run(mission, act_fn, verify_fns, replan_fn):
    # DFS: children verify first, then parent becomes runnable
    emit(event)
```

Structured verify beats "write a plan in markdown". Stored as JSON in `coder_sessions.mission` (supersedes the old `plan_steps` string column).

## Pattern 21: MCP Tool Registration

```python
from augmentum.mcp.client import MCPClient, StdioServerParameters

client: MCPClient = app.state.mcp_client
await client.add_server(
    "my-server",
    StdioServerParameters(
        command="npx",
        args=["-y", "@my/mcp-server"],
        env={"API_KEY": "..."},
    ),
)
# coder.tools.create_coder_tools() auto-discovers and forwards these tools.
result = await client.call_tool("my-server", "my_tool", {"arg": "value"})
```

Timeouts: init+list 30s, call 60s. Transient failures are **not retried** — handle at the call site.

## Pattern 22: Unified Primitive Layer — Tool Surface Declaration

Tools declare which surfaces they live on at registration time. The
registry filters per surface so the same tool can appear on chat, voice,
coder, companion, artifact studio, file context menu, HTTP route, AND a
voice capability line — without that being eight separate registrations.

```python
from augmentum.tools.base import Tool, SurfaceExposure, ToolResult

class MyTool(Tool):
    name = "do_thing"
    category = ToolCategory.EXECUTE
    description = "Does the thing."
    input_schema = {"type": "object", "properties": {...}}

    # Declare WHERE this tool should be exposed:
    surfaces = SurfaceExposure(
        chat=True,
        voice="core",                   # "core" | "interactive" | "disruptive" | "costly"
        coder=True,
        companion=False,
        artifact_studio=False,
        file_context_menu=False,
        http_route="/api/tools/do-thing",  # non-empty → auto-mounts POST endpoint
        voice_capability_line="do the thing",
    )

    async def run(self, **kwargs) -> ToolResult:
        ...

# Registered once in tools/registry.py or proxy/server.py::_build_tool_registry
registry.register(MyTool(...))
```

Consumers fetch their slice via `registry.get_for_surface("coder")` or
`registry.get_for_phase("respond")`. Adding a new consumer surface is one
line on `SurfaceExposure` + one new `get_for_surface` call site — no per-
tool updates. **Phase 1 shipped (~5 tools migrated); rest still use
default `SurfaceExposure()`.**

## Pattern 23: Fabric-Aware Backend Resolution

Any LLM dispatch site MUST use `resolve_backend_with_fabric`, not
`resolve_backend_for_model`. The wrapper transparently routes to a paired
peer when capability + setting allows; the inner function silently falls
back to default when a model isn't local, breaking peer-only models.

```python
# WRONG — breaks peer-only models silently:
backend, model = await registry.resolve_backend_for_model(model_id, user_id=user_id)

# RIGHT — peer-aware, falls back gracefully:
backend, model = await registry.resolve_backend_with_fabric(
    model_id, user_id=user_id
)
```

**23 call sites across 28 files** in narrative/coder/reasoning/flow/tools/
anthropic/openai/ollama already use the wrapper. Audit script flags any
`resolve_backend_for_model` callers in non-test code. See
`augmentum/models/provider_registry.py:212`.

## Pattern 24: Subagent Dispatch

Spawn a sub-LLM with a role profile (`.augmentum/agents/*.md` YAML
frontmatter), filtered tool surface, and parent-context bridge — all
under depth-cap + per-role semaphore.

```python
from augmentum.agents.dispatch import SubagentDispatcher

dispatcher: SubagentDispatcher = app.state.subagent_dispatcher
async for event in dispatcher.dispatch(
    role="bug-finder",                  # resolved workspace > user > built-ins
    prompt="...",
    parent_session_id=session.id,       # context bridge
    user_id=user_id,
    model_override="@anthropic:claude-sonnet-4-6",  # @provider:peer syntax
):
    # event ∈ {iter_started, tool_call, tool_result, message_delta, finished, error}
    yield event
```

Roles are file-based (`.augmentum/agents/<slug>.md`) with hot-reload via
mtime. Persistence in `coder_subagent_runs` (migration 212). Depth cap +
cancellation wired via `_current_subagent_depth` contextvar and per-spawn
`POST /api/coder/subagents/{id}/cancel`. See `augmentum/agents/dispatch.py`.

## Pattern 25: Intent Action Registration

Add a primitive verb to the intent registry. Tier 1 regex matchers
auto-derive from declared templates; the dispatcher consults the registry
before the LLM sees the transcript.

```python
# augmentum/intent/builtin/<surface>.py
from augmentum.intent.action import Action, ActionFanout, Stakes
from augmentum.intent.registry import REGISTRY

@REGISTRY.register
class CreateNote(Action):
    id = "note.create"
    templates = [
        "make a note (about|saying)? {content}",
        "remind me {content}",
    ]
    stakes = Stakes.trivial_reversible
    fanout = ActionFanout(tier1=True, tier3=True)  # voice fast-path + LLM-callable

    async def handle(self, ctx, match) -> ActionResult:
        ...
```

If voice-visible, add the action id to `_VOICE_TOOLS` in
`voice_routes.py`. Action handlers MUST check `session.user_id` and
refuse to write into the anon row. Voice hook fires at
`voice_routes.py:2078::_maybe_dispatch_intent` after STT, before LLM.
See `augmentum/intent/dispatch.py:152`.

## Pattern 26: SafeHttpClient + Trusted Origin Allowlist

Every server-side outbound URL fetch that the user (or a community
artifact) influences MUST go through `SafeHttpClient`. Block loopback,
RFC1918, link-local, multicast targets; cap response size; rebinding-
proof DNS pinning.

```python
from augmentum.utils.safe_http import SafeHttpClient, SafeHttpError

# Outbound community artifact fetch:
client = SafeHttpClient(max_response_size=64 * 1024)
try:
    text, meta = await client.fetch(url, timeout=10.0)
except SafeHttpError as exc:
    return _error("Couldn't fetch", str(exc))
```

For categories where the user can configure additional trusted origins
beyond a built-in default list, pair the fetch with an allowlist check:

```python
_BUILTIN_TRUSTED_ORIGINS = (
    "https://raw.githubusercontent.com/AugmentumHQ/",
    ...
)

def _is_allowed(url: str, settings) -> bool:
    extra = getattr(settings, "my_trusted_origins", []) or []
    return any(url.startswith(p) for p in _BUILTIN_TRUSTED_ORIGINS + tuple(extra))

# Double-gate: allowlist before fetch, fetch with SafeHttpClient:
if not _is_allowed(url, settings):
    raise HTTPException(400, "Untrusted source")
text, _ = await SafeHttpClient().fetch(url, timeout=10.0)
```

Admin-configured persistent URLs (provider base URLs, media server URLs)
are exempt by design — they're treated like env vars. See
`augmentum/utils/safe_http.py:282` and the community-install spec at
`augmentumhq-site/docs/specs/community-install.md`.

## Pattern 27: Auth Middleware Exemption + Self-Resolve

Public-path routes that still WANT to know whether the user is logged in
(e.g. preview screens, status endpoints) need to handle auth themselves
because middleware short-circuits before attaching `scope["user"]`.

```python
# 1. Add to _PUBLIC_PATHS in augmentum/auth/middleware.py
_PUBLIC_PATHS = {
    "/",
    "/api/auth/login",
    "/my-public-route",   # ← new
    ...
}

# 2. In the handler, resolve auth manually:
async def my_handler(request: Request):
    user = await _resolve_user(request)  # parses Bearer or session cookie
    if not user:
        # render login form / redirect / show anon UI
        return _render_login_html()
    # authenticated path
    ...

async def _resolve_user(request: Request):
    """Mirror of auth_routes.py::auth_status — handles Bearer + cookie."""
    user = request.scope.get("user")
    if user:
        return user
    sm = getattr(request.app.state, "session_manager", None)
    if not sm:
        return None
    auth_header = request.headers.get("authorization", "")
    token = (
        auth_header[7:].strip() if auth_header.startswith("Bearer ")
        else _cookie_token(request.headers.get("cookie", ""))
    )
    if not token:
        return None
    try:
        return await sm.validate_token(token)
    except Exception:
        return None
```

Used by `/api/auth/status` and `/community-install`. The POST counterpart
(`/api/community/install`) stays auth-gated — only the public preview
GET handles self-resolution. See
`augmentum/proxy/community_routes.py::_resolve_user`.

## Pattern 28: Community Install Dispatcher

A community artifact installs by dispatching to existing per-category
helpers, not by reimplementing import. Powers + knowledge are admin-only
(install-wide); characters + reasoning-flows are per-user.

```python
async def community_install(request: Request):
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(401)

    body = await request.json()
    # ... validate manifest URL against trusted-origin allowlist ...

    if body["category"] == "characters":
        # Reuse the existing character import helpers
        result_id = await _install_character(request, body["artifact"], user_id)
    elif body["category"] == "reasoning-flows":
        result_id = await _install_reasoning_flow(request, body["artifact"], user_id)
    elif body["category"] == "powers":
        # Admin-only — Powers are install-wide
        if not is_admin(request):
            raise HTTPException(403)
        result_id = await _install_power(request, body["artifact"], user_id)
    elif body["category"] == "knowledge":
        if not is_admin(request):
            raise HTTPException(403)
        result_id = await _install_knowledge_pack(request, body["artifact"], user_id)

    # Audit row — user-scoped, never bypassed
    await _record_install(request, user_id=user_id, ...)
```

Write the audit row regardless of category. Failures during audit write
must NOT break the install (`try/except log.warning` — the user already
got their artifact). See `augmentum/proxy/community_routes.py:315`.
