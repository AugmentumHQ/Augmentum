# MCP Protocol — Integration TODO

## Current State (March 2026)

Foundation layer complete: client (stdio + HTTP), bridge (MCPToolWrapper), server (FastMCP at /mcp),
routes (4 REST endpoints), 60+ tests. All plumbing works. What follows is the roadmap to make MCP
a first-class, deeply integrated feature rather than a bolted-on capability.

---

## 1. Smart Category Inference (Priority: HIGH)

**Problem:** All MCP tools default to `ToolCategory.EXECUTE`, so they only appear in APPLY/GATHER/RESPOND
phases. A search MCP server's tools won't surface during RELEVANT/SEARCH phases where they'd be most useful.

**Solution:** Infer category from tool name + description at registration time.

```
Heuristic patterns:
  - name/desc contains "search", "find", "query", "lookup"  → SEARCH
  - name/desc contains "fetch", "read", "get", "download"   → FETCH
  - name/desc contains "verify", "validate", "check", "test" → VERIFY
  - name/desc contains "write", "create", "save", "upload"   → FILE
  - name/desc contains "image", "draw", "render", "generate" → IMAGE
  - name/desc contains "document", "presentation", "chart"   → ARTIFACT
  - fallback                                                  → EXECUTE
```

**Where:** `bridge.py` — `register_mcp_tools()` should run inference before wrapping.
Allow manual override via config (`mcp_servers` JSON could include `category_overrides` dict).

**Impact on each mode:**
- **Analytical:** MCP search tools appear in RELEVANT phase, verification tools in VERIFY phase.
  This is the biggest unlock — UARF can now leverage external search/verify infrastructure.
- **Passthrough:** No change (tools resolved by name, not phase).
- **Agentic:** Flow steps filter by category; correct assignment means MCP tools slot into the
  right step roles (search steps get search MCP tools, create steps get artifact MCP tools).
- **Narrative:** No impact (no tool use currently).

---

## 2. Rich Content Handling (Priority: HIGH)

**Problem:** `call_tool()` only extracts text blocks from MCP results. Images, files, embedded
resources, and `structuredContent` are silently dropped. This means any MCP tool that returns
images (screenshot tools, chart generators, image search) or files (exporters, converters)
produces empty results.

**Tasks:**
- [ ] Extract `ImageContent` blocks — convert to base64 data URIs or save to image_output dir
- [ ] Extract `EmbeddedResource` blocks — save to workdir, return file path
- [ ] Extract `structuredContent` field — pass through as JSON in ToolResult.metadata
- [ ] Handle mixed content (text + images) — text in output, images in metadata
- [ ] MCPToolWrapper.execute() should populate ToolResult.metadata with rich content
- [ ] Streaming handlers should be able to surface MCP images in chat responses

**Where:** `client.py` `call_tool()` return type changes from `str` to richer structure.
`bridge.py` MCPToolWrapper maps rich result to ToolResult fields.

**Impact on each mode:**
- **Analytical:** Tool results with images can be referenced in VERIFY/CONCLUDE phases.
- **Passthrough:** Rich tool results streamed to user with embedded images.
- **Agentic:** Artifact steps can consume MCP-generated files (charts, exports).
- **Narrative:** If narrative ever gets tool support, scene images from MCP tools.

---

## 3. MCP Resources — Inbound (Priority: HIGH)

**Problem:** MCP resources are the protocol's way of exposing files, databases, and structured
data. Augmentum ignores them entirely. This means if a user connects a filesystem MCP server,
a database MCP server, or a knowledge base MCP server, their resources are invisible.

**What resources enable for users:**
- Connect a filesystem MCP server → browse and inject files into context
- Connect a database MCP server → query tables, inject results into prompts
- Connect a knowledge base MCP server → RAG-style retrieval from external stores
- Connect a code repository MCP server → reference source files during analysis

**Tasks:**
- [ ] Add `list_resources(server_name)` to MCPClientManager
- [ ] Add `read_resource(server_name, uri)` to MCPClientManager
- [ ] Store discovered resources in MCPServerConnection
- [ ] Add REST endpoints: `GET /v1/mcp/resources`, `GET /v1/mcp/resources/{uri}`
- [ ] Resource injection into context — similar to document RAG injection
- [ ] Resource subscription (MCP supports `resources/updated` notifications)

**Integration with each mode:**
- **Analytical:** Resources injected during RELEVANT phase as supplementary context.
  "Here are relevant files from connected systems." Treated like document chunks.
- **Passthrough:** Resources available via explicit user reference ("use the database schema
  from my postgres MCP server").
- **Agentic:** Flow steps can pull resources as input data. Research steps query MCP knowledge
  bases. Draft steps reference MCP file templates.
- **Narrative:** World-building resources — MCP server exposes lore documents, character
  databases, map data. Injected into context builder alongside lorebook entries.
  This is genuinely useful: a community could run an MCP server with shared world state
  and every user's Augmentum instance pulls from it.

---

## 4. MCP Resources — Outbound (Priority: MEDIUM)

**Problem:** External MCP clients connecting to Augmentum's `/mcp` endpoint can call tools
but can't browse Augmentum's stored data (documents, artifacts, session exports, images).

**What to expose as MCP resources:**
- `augmentum://documents/{id}` — uploaded documents (PDF, DOCX, etc.)
- `augmentum://artifacts/{id}` — generated artifacts (reports, presentations, spreadsheets)
- `augmentum://images/{id}` — generated images
- `augmentum://sessions/{id}/export` — session exports (JSON)
- `augmentum://memory/{user_id}/profile` — core memory profile

**Tasks:**
- [ ] Implement resource handlers in `server.py` using FastMCP's resource API
- [ ] Resource templates for parameterized URIs (e.g., `augmentum://documents/{id}`)
- [ ] Pagination for resource listings (documents can be large collections)
- [ ] Access control — scope resources by user/session if multi-user

**Who uses this:**
- Claude Desktop connecting to Augmentum can browse generated artifacts
- OpenClaw connecting to Augmentum can read documents and session context
- Custom scripts can pull Augmentum's generated reports without the REST API
- Other Augmentum instances could share resources via MCP (federated setup)

---

## 5. MCP Sampling — Augmentum as Brain (Priority: MEDIUM)

**Problem:** MCP sampling lets servers request LLM completions from the host. Augmentum IS
the host with multiple backends, load balancing, and smart routing. Without sampling support,
MCP servers must bring their own LLM access — defeating the purpose of a centralized proxy.

**What this enables:**
- Lightweight MCP tool servers that do I/O work (filesystem, database, API calls) can
  delegate "thinking" back to Augmentum. A code analysis MCP server reads files locally
  but asks Augmentum to reason about them.
- Augmentum routes the sampling request through its provider registry — model selection,
  load balancing, rate limiting all apply automatically.
- The MCP server doesn't need its own API keys or model access.

**Tasks:**
- [ ] Implement sampling handler in FastMCP server setup
- [ ] Route sampling requests through ProviderRegistry.resolve_backend_for_model()
- [ ] Apply Augmentum's system prompt augmentation (memory injection, tool context)
- [ ] Respect rate limits and inference semaphore for sampling requests
- [ ] Config: `mcp_sampling_enabled`, `mcp_sampling_max_tokens`, `mcp_sampling_model`
- [ ] Audit trail — log sampling requests for visibility

**Integration with each mode:**
- Sampling is mode-independent — it's the MCP server asking Augmentum to think, not a
  user request flowing through the classifier. But the response benefits from Augmentum's
  memory injection and context augmentation.

**Security consideration:** Sampling means external MCP servers can consume LLM tokens
through Augmentum. Must have rate limiting, token budgets, and opt-in per server.

---

## 6. MCP Prompts — Inbound (Priority: LOW)

**Problem:** MCP prompt templates let servers provide reusable prompt patterns. Augmentum
already has its own prompt system (UARF phase prompts, narrative presets, reasoning flow
templates) but can't consume external prompt templates.

**What this enables:**
- A "legal analysis" MCP server provides prompt templates for contract review, compliance
  checks, etc. Augmentum surfaces these as available reasoning flows or system prompt
  augmentations.
- A "code review" MCP server provides language-specific review prompts.
- Community prompt packs distributed via MCP servers.

**Tasks:**
- [ ] Add `list_prompts(server_name)` to MCPClientManager
- [ ] Add `get_prompt(server_name, prompt_name, arguments)` to MCPClientManager
- [ ] Map MCP prompts → reasoning flow templates (auto-create flows from MCP prompts)
- [ ] Map MCP prompts → narrative prompt presets (if applicable)
- [ ] REST endpoint: `GET /v1/mcp/prompts`
- [ ] UI: show available MCP prompts alongside built-in flow templates

---

## 7. MCP Prompts — Outbound (Priority: LOW)

**What to expose as MCP prompts:**
- Augmentum's reasoning flow templates (Research, Code Review, Math Verification, etc.)
- UARF phase system prompts (for external tools that want Augmentum-style reasoning)
- Narrative prompt presets (system_prompt, jailbreak, author_note templates)

**Tasks:**
- [ ] Register reasoning flow templates as MCP prompts in server.py
- [ ] Support prompt arguments (e.g., `{query}`, `{complexity}` variables)
- [ ] Prompt listing with descriptions and argument schemas

---

## 8. Auto-Reconnection & Health Checks (Priority: HIGH for production)

**Problem:** If an MCP server crashes or network drops, the connection stays dead in
`_servers` dict. No detection, no recovery. Users must manually disconnect and reconnect.

**Tasks:**
- [ ] Periodic health check — call `list_tools()` or lightweight ping every N seconds
- [ ] On failure detection: mark server as disconnected, log warning
- [ ] Auto-reconnect with exponential backoff (1s, 2s, 4s, 8s, max 60s)
- [ ] Config: `mcp_health_check_interval` (default 30s), `mcp_auto_reconnect` (default true)
- [ ] Connection status exposed in REST API (`GET /v1/mcp/servers` includes health state)
- [ ] UI indicator for server health (connected/reconnecting/disconnected)
- [ ] Graceful degradation — if server down, MCP tools return error result instead of hanging

---

## 9. Parallel MCP Tool Execution (Priority: MEDIUM)

**Problem:** UARF's `parse_native_tool_calls_all()` can extract multiple tool calls, and
`asyncio.gather` runs them in parallel. But this already works for MCP tools since they're
standard Tool instances. The gap is at the MCP *client* level — concurrent calls to the
same server may queue on the transport.

**Tasks:**
- [ ] Verify concurrent `call_tool()` calls to same server work correctly
- [ ] If transport serializes: consider connection pooling (multiple sessions per server)
- [ ] For cross-server parallelism: already works (different MCPServerConnection instances)
- [ ] Test: fire 5 concurrent tool calls to same MCP server, verify no deadlock or data mix

---

## 10. Server-Side Timeout Enforcement (Priority: MEDIUM)

**Problem:** `_build_typed_handler()` in server.py doesn't enforce `tool.timeout`. External
MCP clients calling a slow Augmentum tool could block indefinitely.

**Tasks:**
- [ ] Wrap `tool.execute()` with `asyncio.wait_for(tool.execute(**kwargs), timeout=tool.timeout)`
- [ ] Return MCP error result on timeout, not exception
- [ ] Config: `mcp_server_default_timeout` (fallback if tool.timeout is 0)

---

## 11. Input Validation (Priority: MEDIUM)

**Problem:** MCPToolWrapper.validate_input() always returns True. No JSON Schema validation
before calling remote MCP servers, and no validation before executing tools for external clients.

**Tasks:**
- [ ] Client side: validate kwargs against `mcp_tool.inputSchema` before calling
- [ ] Server side: validate incoming params against `tool.input_schema` before executing
- [ ] Apply `coerce_tool_params()` to MCP tool inputs (same type-fixing as UARF tier 3)
- [ ] Return `ToolResult(validation_error=True)` for schema violations (allows LLM retry)

---

## 12. Registry Access Cleanup (Priority: LOW)

**Problem:** `unregister_mcp_tools()` accesses `registry._tools` directly (private field).

**Tasks:**
- [ ] Add `ToolRegistry.unregister(name)` public method (already exists — use it)
- [ ] Add `ToolRegistry.list_by_source(source)` or `list_by_type(MCPToolWrapper)` for filtering
- [ ] Update `unregister_mcp_tools()` to use public API

---

## 13. Per-Tool / Per-Server Configuration (Priority: LOW)

**Problem:** All MCP servers share the same timeouts (30s init, 60s call). No per-server
or per-tool overrides. No way to configure which tools from a server are exposed.

**Tasks:**
- [ ] Extend `mcp_servers` JSON config:
  ```json
  {
    "name": "my-server",
    "url": "https://...",
    "call_timeout": 120,
    "init_timeout": 60,
    "category_overrides": {"search_tool": "search", "write_tool": "file"},
    "tool_filter": ["allowed_tool_1", "allowed_tool_2"],
    "enabled": true
  }
  ```
- [ ] Per-tool timeout from MCP tool annotations (if server provides `execution.timeout`)
- [ ] Tool filtering — only register subset of server's tools (security, noise reduction)

---

## 14. UI Integration (Priority: MEDIUM)

**Problem:** MCP servers and tools aren't visible in the main UI during use. Settings UI
has server management but no runtime visibility.

**Tasks:**
- [ ] MCP panel in inspector sidebar — show connected servers, tool counts, health status
- [ ] Tool call attribution — when MCP tool is called during UARF, show server origin in
  reasoning panel ("Called my-server/search_docs → 3 results")
- [ ] Server connection indicator in status bar (like the backend connection indicator)
- [ ] Tool discovery browser — list all MCP tools with descriptions, test execution
- [ ] Resource browser — if resources implemented, browse/preview MCP resources

---

## 15. MCP-to-MCP Chaining (Priority: LOW, Future)

**Problem:** Can't call one MCP server's tool from another MCP server's context. If a
data-analysis MCP server needs to fetch data from a database MCP server, there's no path.

**What this enables:**
- Composable MCP server pipelines: fetch → transform → analyze → report
- Augmentum as the orchestration hub that wires MCP servers together
- Agentic flows that span multiple MCP servers in a single task

**This ties directly to agentic mode.** Flow steps could specify MCP server + tool combos,
and the executor chains results between servers. The chain system (`tools/chain.py`) with
wave-based DAG execution is already built for this — it just needs MCP-awareness.

---

## Per-Mode Integration Vision

### Passthrough + MCP
**User story:** "I connected my GitHub MCP server and my Jira MCP server. When I chat,
I want the LLM to be able to search issues and read code files naturally."

- MCP tools listed in tool selector (X-Augmentum-Tools header or UI toggle)
- Tool calling loop works as-is (MCPToolWrapper is a normal Tool)
- Resources from MCP servers browseable and injectable into chat context
- **Gap to close:** Smart tool filtering in UI — show MCP tools grouped by server

### Analytical + MCP
**User story:** "I connected a fact-checking MCP server. During UARF analysis, I want it
used in the VERIFY phase to cross-reference claims against authoritative sources."

- Category inference puts tools in correct phases (search→RELEVANT, verify→VERIFY)
- UARF phase prompts should mention available MCP tools by capability
- MCP resources injected during RELEVANT phase as evidence sources
- **Gap to close:** Category inference (item 1), resource injection (item 3)

### Narrative + MCP
**User story:** "My roleplay community runs a shared world-state MCP server. Every player's
Augmentum instance pulls the current world state, faction standings, and event timeline."

- MCP resources → context builder injection (world state, lore, character databases)
- MCP tools for dice rolling, random tables, name generators, map lookups
- MCP prompts → character-specific prompt templates from community servers
- Narrative doesn't need full tool calling — selective resource + prompt consumption
- **Gap to close:** Resources (item 3), narrative context builder integration, selective
  feature consumption without full tool calling overhead

### Agentic + MCP
**User story:** "I want to create a flow that researches a topic using my custom search MCP,
drafts a report, generates charts from my data MCP, and exports via my storage MCP."

- MCP tools slotted into flow steps by category (research→search MCP, create→artifact MCP)
- MCP resources as step inputs (pull data from external systems before processing)
- MCP sampling enables lightweight "worker" MCP servers that do I/O but delegate thinking
- Chain executor wires MCP servers together in DAG (item 15)
- **Gap to close:** Category inference (item 1), resources (item 3), sampling (item 5),
  flow step MCP-awareness in executor

---

## External Consumption Vision (Augmentum as MCP Server)

### Claude Desktop / Other AI Assistants
**User story:** "I connect Claude Desktop to Augmentum's /mcp endpoint. Now Claude can
search the web through Augmentum, recall my persistent memories, and access my documents."

- Already partially working (tools + memory tools exposed)
- Add: document resources, artifact resources, image resources
- Add: reasoning flow prompts ("run Research flow on this query")
- Add: sampling support so Augmentum handles the LLM calls efficiently

### OpenClaw / External Agents
**User story:** "OpenClaw handles my messaging. Augmentum handles intelligence. OpenClaw
calls Augmentum's MCP tools for search, verification, and memory."

- Augmentum as the "brain" behind consumer agents
- Memory persistence across all connected agents (shared brain)
- Verification tools ensure agent outputs are fact-checked
- Sampling means agents don't need their own API keys

### Other Augmentum Instances (Federation)
**User story:** "My team runs separate Augmentum instances. Each exposes its memory and
documents via MCP. We can query each other's knowledge bases."

- Each instance is both MCP client and server
- Cross-instance memory search (connect to teammate's /mcp, call memory_recall)
- Shared document pools without centralized storage
- **Future consideration:** authentication, access control, audit trail

---

## Implementation Order (Recommended)

### Phase A — Make Existing MCP Actually Useful
1. Smart category inference (item 1) — unlocks correct phase routing
2. Rich content handling (item 2) — stops dropping non-text results
3. Auto-reconnection (item 8) — production reliability
4. Server-side timeout (item 10) — safety for external clients
5. Input validation (item 11) — correctness + retry on bad params

### Phase B — Resources & Context
6. Inbound resources (item 3) — browse external data through MCP
7. Outbound resources (item 4) — expose documents/artifacts/images
8. UI integration (item 14) — make MCP visible and manageable

### Phase C — Advanced Protocol
9. Sampling support (item 5) — Augmentum as brain for MCP servers
10. Parallel execution verification (item 9) — performance
11. Per-server configuration (item 13) — operational control
12. Inbound prompts (item 6) — consume external prompt templates

### Phase D — Ecosystem
13. MCP-to-MCP chaining (item 15) — composable server pipelines
14. Outbound prompts (item 7) — expose reasoning flows
15. Registry cleanup (item 12) — code quality
