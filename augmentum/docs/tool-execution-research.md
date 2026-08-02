# Tool Execution Research: Improving Augmentum's Pipeline

> Research conducted March 2026. Covers OpenClaw (ClawdBot), Open WebUI, LangChain/LangGraph,
> Semantic Kernel, AutoGen, CrewAI, Gorilla/BFCL, ToolBench, NexusRaven, Phidata,
> Ollama native tool calling, structured output enforcement, OpenCode, Qwen Code/Qwen-Agent,
> and coding agent orchestration patterns (Microsoft, Aider, Cursor).

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [OpenClaw (ClawdBot) — Agent-First Tool Execution](#1-openclaw-clawdbot)
3. [Open WebUI — Plugin-Based Tool System](#2-open-webui)
4. [Framework Patterns — LangChain, AutoGen, CrewAI, Semantic Kernel](#3-framework-patterns)
5. [Tool Calling Benchmarks — Gorilla/BFCL, ToolBench, NexusRaven](#4-benchmarks)
6. [Structured Output Enforcement](#5-structured-output-enforcement)
7. [Coding Agents — OpenCode, Qwen Code, Aider](#6-coding-agents)
8. [Agent Orchestration Patterns](#7-orchestration-patterns)
9. [Cross-Project Patterns](#8-cross-project-patterns)
10. [Gap Analysis — Augmentum vs. Industry](#9-gap-analysis)
11. [Recommended Improvements](#10-recommended-improvements)

---

## Executive Summary

Augmentum's tool pipeline already implements several best practices that many projects lack:
fuzzy name resolution (4-level), phase-gated tool access, placeholder rejection, auto-search
bypass, refusal detection, balanced JSON extraction, and forgiving JSON recovery. However,
research across 15+ projects reveals five significant gaps compared to the state of the art:

1. **No native tool calling** — Augmentum uses text-based `TOOL_CALL:` parsing exclusively,
   missing Ollama/OpenAI's structured `tool_calls` API that guarantees valid JSON. Every
   major coding agent (OpenCode, Qwen Code, Claude Code, Cursor) uses native calling first.
2. **No structured output enforcement** — Not using Ollama's JSON Schema `format` parameter
   or llama.cpp's GBNF grammars to guarantee valid tool call output
3. **No type coercion** — Small models emit `"42"` instead of `42`; no automatic type fixing.
   OpenCode solves this with Zod's `z.coerce` pattern.
4. **No parallel tool call extraction** — Only the first `TOOL_CALL:` is parsed per response.
   Qwen-Agent, LangChain, and OpenCode all extract and execute multiple tool calls per turn.
5. **No thinking block awareness** — Qwen3-Coder and DeepSeek reasoning models emit
   `<think>...</think>` blocks that can confuse our text-based parser.

These gaps primarily affect users running local models through Ollama or llama.cpp.
Frontier models via OpenAI-compatible APIs work well with the current approach.

**The strongest signal from this research:** The coding agent ecosystem has converged on
native function calling as the primary tool execution method, with text-based parsing as a
fallback. Augmentum is doing it backwards — text-based primary with no native support.
Flipping this priority is the highest-impact change we can make.

---

## 1. OpenClaw (ClawdBot)

**What it is:** Open-source autonomous AI agent (134K GitHub stars) that lives in messaging
platforms (Telegram, Discord, WhatsApp, Slack, etc.) and executes tasks on the user's machine.
Formerly "ClawdBot", renamed to "Moltbot" then "OpenClaw" in Jan 2026.

**Repository:** github.com/openclaw/openclaw

### Architecture (5 Layers)

```
Message Channels (12 platforms)
    ↓ normalized envelope
Gateway Server (control plane)
    ↓ session routing
Lane Queue (per-session serial queue)
    ↓ task dispatch
Agent Runner (context assembly)
    ↓ assembled prompt
LLM API → Agentic Loop (tool call detection → execute → re-prompt)
    ↓ final response
Response Path → back through gateway to channel
```

### Key Design Decisions

**1. Delegate the agent loop, own everything else.**
OpenClaw uses the **Pi agent framework** for the core think-act cycle (tool calling, context
management, LLM interaction). OpenClaw's value-add is the layers *around* the loop: channel
normalization, session management, memory persistence, skill extensibility, and security.

**Lesson for Augmentum:** Our UARF pipeline IS the agent loop — we own it directly. This is
an advantage for control and customization, but we should study Pi's specific tool-call
parsing and retry patterns for reliability ideas.

**2. Skills as markdown, not code modules.**
Skills are `SKILL.md` files with YAML frontmatter containing name, description, and instructions.
They're injected into the agent's context as plain text when activated. This enables:
- Progressive disclosure (only names/descriptions loaded initially, ~97 chars each)
- Hot reloading with 250ms debounce
- Agent self-authoring of new skills
- Community marketplace (ClawHub, 5400+ skills)

**Lesson for Augmentum:** Our tool descriptions in `get_tool_prompt_section()` are already
concise (~50 words each with required/optional param hints). But we don't do progressive
disclosure — all phase-eligible tools are injected at once. For phases with many tools
(APPLY has 6 categories), this could be optimized.

**3. Per-session serial queuing (Lane Queue).**
Every session gets its own queue; tasks execute one at a time within a session. Session keys
are `workspace:channel:userId`. This prevents race conditions from concurrent tool calls.

**Lesson for Augmentum:** We have no concurrency control on tool execution within a session.
If two analytical requests for the same session arrive simultaneously, they could interleave
tool calls. The Lane Queue pattern would prevent this.

**4. Agentic loop = "does this response contain a tool call?"**
After every LLM response, the system checks for tool calls. If found: execute, inject result,
re-prompt. If not: return to user. LLM output ≠ final response — it enters judgment logic.

**Lesson for Augmentum:** This matches our `_run_phase_with_tools` loop, but OpenClaw's loop
is unbounded (runs until no more tool calls), while ours caps at `_MAX_TOOL_CALLS_PER_PHASE=3`.
Our cap is appropriate for small models but could be configurable.

**5. Memory as human-readable flat files.**
Markdown for notes, YAML for structured data, JSONL for conversation history. Hybrid search
combining vector similarity (sqlite-vec) with keyword search (FTS5), all local.

**Lesson for Augmentum:** Our state management (fact registry, entity registry, assumption
stack) uses structured SQLite. OpenClaw's approach of human-readable files is interesting for
debugging but our structured approach is better for the analytical pipeline.

### Security Concerns (Cautionary)

- CVE-2026-25253: Missing WebSocket origin validation → remote code execution
- 12-20% of ClawHub skills contain malicious prompt injection
- Plaintext credential storage under `~/.openclaw/`
- No process-level sandboxing for skill execution

**Lesson for Augmentum:** Our sandboxed Python executor (separate Docker container, no network)
is significantly more secure than OpenClaw's approach. Our tool execution is server-side with
no user-provided code execution in the main process.

---

## 2. Open WebUI

**What it is:** Self-hosted AI interface (Ollama/OpenAI compatible) with a plugin-based tool
system. The primary frontend our users connect through Augmentum.

### Tool Execution Modes

**Default Mode (Legacy/Prompt-Based):**
- Tool descriptions injected into the prompt as text
- LLM uses natural language to decide tool usage
- Universal compatibility — works with any model, including small/old ones
- **Drawback:** Breaks KV cache by changing prompts each turn → increased latency
- Similar to Augmentum's current `TOOL_CALL:` text-based approach

**Native Mode (Recommended/Structured):**
- Uses model's built-in function-calling capability (JSON schema definitions)
- Returns structured `tool_calls` JSON — no text parsing needed
- KV cache friendly — tool definitions are stable across turns
- **Requirement:** High-quality models (GPT-5, Claude 4.5+, Gemini 3+, or good local models
  with native tool calling like Llama 3.1+, Qwen 2.5+)
- **Critical caveat:** Many EventEmitter event types are incompatible with Native mode due to
  content snapshot overwriting

### Tool Definition Architecture

Tools are Python classes with:
```python
class Tools:
    class Valves(BaseModel):           # Admin-level config (API keys, URLs)
        api_key: str = ""
    class UserValves(BaseModel):       # Per-user config
        preferred_units: str = "metric"

    def __init__(self):
        self.valves = self.Valves()

    async def search_web(self, query: str, __event_emitter__=None) -> str:
        """Search the web for information."""
        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": "Searching..."}})
        # ... tool logic ...
        return result_string
```

**Reserved arguments** (dependency injection):
- `__user__` — user info (ID, email, role, user-level valves)
- `__request__` — FastAPI Request object (mandatory since v0.5)
- `__event_emitter__` — one-way UI communication (status, notifications, content)
- `__event_call__` — two-way interaction (user input/confirmation requests)
- `__messages__`, `__files__`, `__model__` — conversation context

### Tool Execution Flow (7 Steps)

1. User submits message with enabled tools
2. Backend assembles enabled tools and configurations
3. System determines Default vs Native mode
4. LLM receives tool descriptions (Default) or function schemas (Native)
5. Backend executes requested tools with validated parameters
6. Tool output integrates into conversation context
7. LLM generates response incorporating results

### Built-in System Tools (Native Mode)

When enabled, these are automatically available:
- **Search & Web:** `search_web`, `fetch_url`
- **Knowledge Base:** `query_knowledge_files`, `view_file`
- **Memory:** `add_memory`, `search_memories`, `delete_memory`
- **Notes:** `write_note`, `view_note`, `replace_note_content`
- **Code:** `execute_code` (sandboxed)
- **Chat History:** `search_chats`, `view_chat`
- **Image Generation:** `generate_image`, `edit_image`
- **Channels:** `search_channels`, `view_channel_message`

### Three-Layer Activation

All three must be enabled for a tool to be injected:
1. Global config enabled (Admin Panel)
2. Model capability enabled (Workspace > Models editor)
3. Per-chat toggle enabled (chat input bar)

### Pipelines (Separate Server)

For more complex tool pipelines, Open WebUI supports a separate Pipelines server:
- Executes on a separate process, reducing main server load
- Can install arbitrary Python dependencies
- More extensible than workspace tools
- OpenAI-compatible plugin framework

### Key Lessons for Augmentum

**1. Dual-mode approach works.**
Open WebUI's Default/Native mode split validates our need for both text-based and native
tool calling. Default mode for small models, Native mode for capable ones.

**2. EventEmitter pattern for live status.**
The `__event_emitter__` for real-time UI updates during tool execution is elegant. We
already do something similar with UARF phase streaming metadata, but we don't emit
per-tool-call status updates.

**3. Per-model tool capability gating.**
Open WebUI lets admins enable/disable tool categories per model. We could add similar
per-model configuration to optimize tool prompts for each model's capabilities.

**4. Knowledge tool behavior differs with context.**
When knowledge is attached, different tools are injected than when browsing freely. This
context-aware tool selection is more sophisticated than our static phase-to-category mapping.

**5. DeepSeek V3.2 produces malformed native tool calls.**
Open WebUI documents that some models produce malformed responses with native tool calling.
This reinforces the need for our robust text-based fallback.

---

## 3. Framework Patterns

### LangChain / LangGraph

**Tool Call Schema (standardized across providers):**
```python
class ToolCall(TypedDict):
    name: str       # tool function name
    args: dict      # parsed arguments (already deserialized)
    id: str         # unique call ID for correlation
```

**Key patterns:**
- **Normalized tool calls:** All providers (OpenAI, Anthropic, Google, Ollama) are converted
  to a uniform `ToolCall` format. Augmentum's `InternalChatRequest` does something similar.
- **`bind_tools()` fallback:** For models without native tool calling, injects JSON schemas
  into the prompt and parses output with `JsonOutputToolsParser`. Similar to our approach.
- **`ToolNode` batch execution:** Executes multiple tool calls from a single LLM response
  in parallel. We currently execute sequentially.
- **Error-as-result loop:** When `handle_tool_errors=True`, errors are sent back as
  `ToolMessage(status="error")` and the LLM can self-correct. This is the most effective
  retry pattern.
- **`RetryPolicy` on nodes:** Configurable retry with exponential backoff for transient failures.
- **Fallback chains:** `tool.with_fallbacks([backup_tool])` for graceful degradation.

### Semantic Kernel (Microsoft)

**Auto-Invocation Loop:**
```python
settings.function_call_behavior = FunctionCallBehavior.AutoInvokeKernelFunctions()
settings.max_auto_invoke_attempts = 5
```

Runs an internal loop: Send → detect tool calls → execute → append results → re-send →
repeat until text response or max attempts. Very similar to our `_run_phase_with_tools`.

**Key patterns:**
- **`FunctionInvocationFilter`:** Middleware that wraps tool execution for pre/post processing.
  Could be useful for adding timing, logging, rate limiting to our tools.
- **Max auto-invoke cap:** Default 5 attempts. Our cap of 3 per phase is conservative but
  appropriate for small models.

### AutoGen (Microsoft)

**Two-agent tool execution:**
- `AssistantAgent` (LLM) generates tool calls
- `UserProxyAgent` (Executor) executes them and returns results
- Natural conversation loop between the two

**Key patterns:**
- **Code execution as alternative:** For models without tool calling, the LLM writes Python
  code that calls the tools. We have `python_exec` but don't use it as a meta-tool.
- **`Handoff` pattern:** Failing agent hands off to a specialized error-handling agent.
  Could apply to our fallback-to-passthrough pattern.
- **`max_consecutive_auto_reply`:** Limits loop length. Similar to our `_MAX_TOOL_CALLS_PER_PHASE`.

### CrewAI

**ReAct-style text format:**
```
Thought: I need to search for this information
Action: Search Tool
Action Input: {"query": "example search"}
Observation: [tool result here]
```

**Key patterns:**
- **Model-agnostic:** ReAct format works with any LLM, including small local models.
  Similar philosophy to our `TOOL_CALL:` / `TOOL_INPUT:` format.
- **`max_iter = 15-25`:** Much higher tool call limit than our 3. CrewAI tasks are more
  complex (multi-step workflows).
- **Re-prompting on parse failure:** If the LLM doesn't follow the format, CrewAI
  re-prompts with formatting instructions. We don't currently do this.
- **Memory across tool calls:** Short-term, long-term, and entity memory. Our analytical
  state (fact registry, assumption stack) serves a similar purpose.
- **Delegation:** Agents can delegate to other agents. Maps to our mode fallback
  (narrative → passthrough).

### Phidata (now "Agno")

**Key patterns:**
- **`tool_call_limit`:** Maximum tool calls per response (prevents infinite loops).
- **`retry_on_tool_error`:** Boolean flag for automatic retry. Simple but effective.
- **`tool_choice`:** Can force tool usage (`"required"`) or let model decide (`"auto"`).
  We don't expose this control.
- **Toolkit pattern:** Tools grouped into toolkits. Similar to our `ToolCategory` grouping.

---

## 4. Benchmarks

### Berkeley Function Calling Leaderboard (BFCL v3)

The primary benchmark for LLM tool calling accuracy.

**Test categories:**
1. AST Simple — single function call, straightforward params
2. AST Multiple — choosing right function from many options
3. AST Parallel — multiple independent calls in one turn
4. AST Parallel Multiple — parallel calls selecting from multiple functions
5. Exec Simple/Multiple/Parallel — calls that are actually executed
6. Relevance Detection — knowing when NOT to call any tool
7. Multi-Turn — multi-step conversations with tool results
8. Live — real-world API calls

**Key findings relevant to Augmentum:**

| Model | Overall Accuracy | Notes |
|-------|-----------------|-------|
| GPT-4o / Claude 3.5 Sonnet | 85-92% | Reliable for all tool patterns |
| Llama 3.1 70B | 75-82% | Good for single/multiple, weak on parallel |
| Llama 3.1 8B | 60-70% | Frequent argument errors, hallucinated params |
| Qwen 2.5 (7-72B) | Strong relative to size | Often matches/exceeds Llama at same size |
| Models < 3B | 40-55% | Unreliable for tool calling |

**Critical findings:**
1. **Relevance detection is hardest for small models** — they call tools when they shouldn't.
   Validates our phase-gated tool access.
2. **Argument accuracy > tool selection** — models often pick the right tool but provide wrong
   arg types, miss required fields, or include fabricated args. Validates our schema validation.
3. **Multi-turn degrades significantly for <13B models** — maintaining context across
   tool-call-result-response cycles is hard. Our `_MAX_TOOL_CALLS_PER_PHASE=3` is appropriate.
4. **Fine-tuned models outperform larger general models** — Hermes, Firefunction, xLAM achieve
   near-GPT-4 accuracy at 7-13B. We should recommend these to users.
5. **Each additional tool reduces selection accuracy ~3-5% on small models** — reinforces
   keeping tool counts per phase low.

### ToolBench / ToolLLM

**DFSDT (Depth-First Search Decision Tree):**
A planning strategy where the LLM generates multiple tool call candidates at each step,
evaluates them, and backtracks if a path fails. This parallels our UARF verify-then-backtrack
pattern.

**API Retriever pattern:**
Instead of showing the LLM all tools, train a retriever model to select the most relevant
5-10 tools based on the query. This is a more sophisticated version of our phase-to-category
mapping.

### NexusRaven

**Key insight:** Fine-tuned CodeLlama 13B for function calling achieves GPT-4 level accuracy.
Uses Python-style function call output (not JSON), parsed with Python's `ast` module.
AST-based parsing is more robust than JSON parsing for function calls.

---

## 5. Structured Output Enforcement

### Ollama JSON Schema `format` Parameter

Since Ollama v0.5 (Dec 2024), the `format` parameter accepts a full JSON Schema object.
Ollama converts this to a GBNF grammar internally and uses llama.cpp's grammar-constrained
decoding. **The model physically cannot emit tokens that violate the schema.**

```json
{
  "model": "llama3.2",
  "messages": [...],
  "format": {
    "type": "object",
    "properties": {
      "tool_name": { "type": "string" },
      "tool_input": { "type": "object" }
    },
    "required": ["tool_name", "tool_input"]
  }
}
```

**Impact:** Eliminates all JSON parsing failures. Works with any Ollama model, including those
without native tool calling support. Zero-overhead guarantee — no retries needed.

**Current Augmentum gap:** `OllamaBackend._build_ollama_payload` already passes `request.format`
through, but the analytical engine only sets it to `"json"` or `None`. Passing a full tool-call
JSON Schema would guarantee valid output.

### llama.cpp GBNF Grammars

llama.cpp supports GBNF (GGML BNF) grammars via the `--grammar-file` parameter or `grammar`
field in the API request. The grammar constrains token generation at the sampling level.

```
root ::= "{" ws "\"name\"" ws ":" ws string "," ws "\"arguments\"" ws ":" ws object "}"
string ::= "\"" ([^"\\] | "\\" .)* "\""
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
```

**Current Augmentum gap:** `LlamaCppBackend._to_openai_payload` does not pass grammar or
response_format parameters. Adding grammar support would enable constrained decoding.

### OpenAI `strict: true` Mode

When `strict: true` is set on function definitions, OpenAI uses constrained decoding
server-side. Guarantees the output matches the declared JSON Schema.

**Constraints:** All fields must be `required`, `additionalProperties: false`, no `default`
values, unions via `anyOf` only.

### Other Libraries

- **Outlines** (dottxt-ai): Python library generating token masks from regex/JSON schemas
- **Guidance** (Microsoft): Template-based generation with `select()`/`gen()` constraints
- **LMQL**: SQL-like query language for LLMs with constraints
- **SGLang** (Stanford): Runtime with constrained decoding for structured generation

---

## 6. Coding Agents — OpenCode, Qwen Code, Aider

These projects are the most relevant to Augmentum because they implement the same core pattern:
an LLM in a loop with tools, parsing tool calls from model output, executing them, feeding
results back, and iterating until the task is complete. They've solved many of the same
reliability challenges we face.

### OpenCode

**What it is:** Open-source terminal-native AI coding agent (100K+ GitHub stars, 2.5M monthly
users). Built in TypeScript/Bun with Hono HTTP server. Uses Vercel AI SDK for LLM interaction.

**Repository:** github.com/opencode-ai/opencode

#### Architecture

```
TUI / IDE / Desktop
    ↓ prompt
HTTP Server (Hono)
    ↓ Session.prompt()
Prompt Assembly (system prompts + history + summaries + tools)
    ↓ streamText() with tool definitions
LLM Response → Tool Call Detection → Execute → Feed Back → Loop
    ↓ events
Event Bus → SSE → Real-time UI updates
    ↓ persist
Disk storage (conversation history)
```

#### Tool System (13 Built-in Tools)

| Tool | Purpose | Permission Default |
|------|---------|-------------------|
| bash | Shell command execution | ask |
| edit | Exact string replacement in files | ask |
| write | Create/overwrite files | ask |
| read | File contents with line ranges | allow |
| grep | Regex content search (ripgrep) | allow |
| glob | Pattern-matching file discovery | allow |
| list | Directory enumeration | allow |
| patch | Apply diff/patch files | ask |
| skill | Load SKILL.md documentation | allow |
| todowrite/todoread | Task list management | allow |
| webfetch | Retrieve web content | allow |
| websearch | Exa AI-powered search | allow |
| question | User interaction/clarification | allow |

**Tool Definition Format (TypeScript/Zod):**
```typescript
Tool.define("read", {
  description: "Retrieve file contents with optional line ranges",
  parameters: z.object({
    filePath: z.string().describe("The path to the file to read"),
    offset: z.coerce.number().describe("Line number to start from").optional(),
    limit: z.coerce.number().describe("Number of lines").optional(),
  }),
  async execute(params, ctx) {
    // path validation, file checks, content slicing
    return { title, output, metadata }
  }
})
```

**Key insight: `z.coerce.number()`** — Zod's `coerce` automatically converts string inputs
to numbers. This is the TypeScript equivalent of the type coercion we need to add.

#### Tool Execution Loop

OpenCode uses Vercel AI SDK's `streamText()` which handles the core loop:

1. LLM receives tool definitions (JSON Schema from Zod schemas)
2. Model outputs structured tool calls (native function calling)
3. SDK iterates through `fullStream` events: `tool-call`, `tool-result`, `tool-error`
4. Tool calls are executed via their registered `execute()` functions
5. Results feed back into the LLM's context window
6. Loop continues until `stopWhen` condition (max 1000 steps or `shouldStop`)

**Streaming event types:**
- `tool-call`: Logs call as running with input/timestamp
- `tool-result`: Updates completion status, output, metadata
- `tool-error`: Handles rejection or failure (sets `shouldStop = true` on permission denial)
- `text-delta`: Accumulates streaming text
- `start-step`: Creates Git snapshot for rollback capability
- `finish-step`: Calculates token usage and cost

#### Error Handling

- **`maxRetries: 3`** on `streamText()` for transient failures
- **Permission denial**: Sets `shouldStop = true`, graceful exit
- **Tool errors**: Captured as `tool-error` events; critical failures abort loop
- **Binary/image files**: Read tool rejects with clear error message
- **Output truncation**: Bash tool caps at `MAX_OUTPUT_LENGTH`
- **Git snapshots**: Created at each `start-step` for rollback on failure

#### Agent System (Plan vs Build)

Two primary agents with different tool permissions:
- **Build** (default): All tools enabled, full write access
- **Plan**: Read-only — `edit` and `bash` default to `ask` permission

Sub-agents invoked via **Task tool** get isolated sessions with separate context:
```json
{
  "review": {
    "mode": "subagent",
    "model": "anthropic/claude-sonnet-4-20250514",
    "tools": {"write": false, "edit": false}
  }
}
```

**Per-tool permission granularity:**
```json
{
  "permission": {
    "bash": {
      "git *": "ask",
      "grep *": "allow",
      "rm *": "deny"
    }
  }
}
```

#### MCP Integration

MCP servers spawned on startup, communicate via JSON-RPC over stdio. Tools fetched
automatically and added to the tool registry. MCP tool calls route through the client:
```typescript
for (const [key, item] of Object.entries(await MCP.tools())) {
  tools[key] = item
}
```

#### Key Lessons for Augmentum

1. **Native function calling via AI SDK** — OpenCode never does text-based tool call parsing.
   It relies entirely on the AI SDK's structured tool calling, which delegates to the
   provider's native function calling API. This eliminates parsing failures entirely.

2. **Zod coerce for type safety** — `z.coerce.number()` automatically handles string-to-number
   conversion. We need equivalent type coercion in our Python tool validation.

3. **Git snapshots for rollback** — Each tool execution step creates a snapshot. If tools
   produce bad results, the agent can roll back. Interesting for our file_ops tool.

4. **Per-command bash permissions** — Granular glob patterns for bash commands. We could
   apply similar patterns to our Python executor's allowed operations.

5. **1000-step max** — Much higher than our 3-per-phase. But OpenCode tasks are unbounded
   coding sessions, not structured analytical pipelines. Our cap is appropriate.

6. **Event bus for real-time updates** — Tool execution events stream in real-time via SSE.
   Similar to our UARF phase streaming but more granular (per-tool-call events).

7. **Skill loading = progressive disclosure** — SKILL.md files loaded on demand, not all
   at once. Matches the pattern we identified from OpenClaw.

---

### Qwen Code & Qwen-Agent

**What it is:** Open-source CLI coding agent by Alibaba (forked from Gemini CLI), optimized
for Qwen3-Coder model. Qwen-Agent is the underlying Python framework for building LLM
applications with function calling, MCP, code interpreter, and RAG.

**Repositories:**
- github.com/QwenLM/qwen-code (CLI tool)
- github.com/QwenLM/Qwen-Agent (framework)
- github.com/QwenLM/Qwen3-Coder (model)

#### Qwen3-Coder Function Calling Format (Hermes-Style)

Qwen3 uses XML-structured tool calls (Hermes format), not JSON like OpenAI:

```
<tool_call>
{"name": "web_search", "arguments": {"query": "current weather <city, region>"}}
</tool_call>
```

**Tool definitions** follow OpenAI JSON Schema format but are wrapped in `<tools>` XML tags
in the chat template:

```xml
<tools>
[{"type": "function", "function": {"name": "web_search", "description": "...", "parameters": {...}}}]
</tools>
```

**Tool results** use `<tool_response>` tags:
```xml
<tool_response>
{"temperature": 42, "condition": "cloudy"}
</tool_response>
```

**Key difference from OpenAI:** The container is XML tags (`<tool_call>`, `<tool_response>`)
rather than JSON `tool_calls` arrays. This has implications for parsing.

#### Qwen-Agent Framework Architecture

```python
# Core hierarchy
BaseChatModel → BaseFnCallModel  (adds function calling)
BaseTool → @register_tool        (tool registration)
Agent → FnCallAgent              (agent with tool loop)
```

**Tool Definition:**
```python
@register_tool("web_search")
class WebSearchTool(BaseTool):
    description = "Search the web for information"
    parameters = [{
        "name": "query",
        "type": "string",
        "description": "Search query",
        "required": True
    }]

    def call(self, params: str, **kwargs) -> str:
        # Tool implementation
        return result_string
```

**Function calling pipeline:**
1. `_preprocess_messages()` — Format messages for function calling
2. Model generation — Produces `<tool_call>` XML blocks
3. `_postprocess_messages()` — Extract function calls from output
4. `validate_num_fncall_results()` — Ensure result count matches call count
5. Results appended as `{"role": "function", "name": "...", "content": "..."}`

#### Dual Format Support

Qwen-Agent supports two prompt styles via `fncall_prompt_type`:
- **`"qwen"`** — Qwen's native format with `function_call` field
- **`"nous"`** — Hermes/Nous format with XML tags

And two result role formats:
- **Qwen-Agent**: `role: "function"` with `name` field
- **vLLM/OpenAI-compat**: `role: "tool"` with `tool_call_id` field

#### Thinking Budget for Reasoning Models

Unique to Qwen3: reasoning models have a "thinking budget" that limits reasoning tokens:
- Token ID `151668` marks closing `</think>` tags
- "Early-stopping prompt" with "limited time by user" when budget exhausted
- Two-stage generation: first up to budget, then continuation

**Important warning:** For reasoning models, avoid stopword-based parsing templates (like
ReAct format) because thinking output may contain stopwords that interfere with tool call
detection.

#### Qwen3-Coder Training Approach

Two distinct RL strategies:
- **Code RL**: Execution-driven large-scale RL across real-world tasks, auto-scaling test cases
- **Agent RL (Long-Horizon)**: Multi-turn RL with 20,000 parallel environments for planning,
  tool use, feedback reception, and decision making

The model achieves "state-of-the-art results among open models on Agentic Coding, Agentic
Browser-Use, and Agentic Tool-Use, comparable to Claude Sonnet 4."

#### Key Lessons for Augmentum

1. **XML-based tool call format** — Qwen uses `<tool_call>` XML tags, not `TOOL_CALL:` text
   lines. If we add native tool calling support, we need to handle this format for Qwen models
   even when not using Ollama's native `tools` parameter.

2. **Dual prompt type support** — The framework internally handles different tool call formats
   per model. This validates our need for a hybrid parsing approach.

3. **Validation of result count** — `validate_num_fncall_results()` ensures every tool call
   gets exactly one result. We don't verify this — if a tool call fails silently, the model
   never gets a result message.

4. **Thinking budget interaction** — Reasoning models' `</think>` tokens can interfere with
   tool call parsing. If users run Qwen3 reasoning models through Augmentum, our text-based
   parser needs to handle `<think>...</think>` blocks.

5. **Parallel function calls** — The framework supports multiple tool calls in one response
   via `parallel_function_calls` config. We currently only extract the first tool call.

6. **Agent RL training** — Models trained specifically for multi-turn tool use are dramatically
   more reliable. We should recommend Qwen3-Coder and similar models to users.

---

### Aider

**What it is:** Git-native CLI coding agent that works with multiple models. Known for
pioneering the "Repository Map" pattern and for reliable structured edits.

**Key tool patterns:**
- **Repository Map**: Uses tree-sitter to parse code into AST, extracts function signatures
  and class definitions, builds a dependency graph using PageRank to rank symbol importance,
  dynamically fits optimal content within token budgets
- **Edit formats**: Multiple output formats (diff, whole-file, search-replace) selected based
  on model capability. Smaller models get simpler formats.
- **Git integration**: Every edit is a commit. Rollback = `git revert`. No custom undo logic.
- **Model-specific edit format selection**: Aider benchmarks each model on its ability to
  produce correct edits in each format, then recommends the best format per model.

**Lesson for Augmentum:** The idea of benchmarking models to select the best tool-call format
per model is powerful. We could test which models work best with native tool calling vs.
`TOOL_CALL:` text format vs. structured JSON output.

---

## 7. Agent Orchestration Patterns

Microsoft's Azure Architecture Center documents five fundamental patterns that map directly
to how Augmentum and other tool-using systems organize work:

### Pattern 1: Sequential (Pipeline)

```
Agent 1 → Agent 2 → ... → Agent N → Result
```

Each agent processes the output of the previous one. **Maps to UARF's phase pipeline:**
ASSESS → IDENTIFY → RELEVANT → APPLY → VERIFY → CONCLUDE. Each phase builds on previous
phase output.

**Key insight from Microsoft:** "Avoid this pattern when the workflow requires backtracking
or iteration." UARF explicitly handles this with verify-then-backtrack, making it a hybrid
sequential + iterative pattern.

### Pattern 2: Concurrent (Fan-out/Fan-in)

```
Input → [Agent 1, Agent 2, ..., Agent N] → Aggregate → Result
```

Multiple agents process the same input in parallel. **Maps to our auto-search:** multiple
search queries execute in parallel via `asyncio.gather()`, results are deduplicated and
aggregated.

**Key insight:** "Choose a strategy: voting for classification, weighted merging for scored
recommendations, or LLM-synthesized summary when results need reconciling."

### Pattern 3: Maker-Checker Loop

```
Maker → Checker → (feedback) → Maker → Checker → ... → Approved
```

One agent creates, another evaluates. **This is exactly UARF's APPLY → VERIFY → (backtrack)
→ APPLY → VERIFY cycle.** Microsoft recommends:
- Clear acceptance criteria for the checker (our VERIFY has `CONFIDENCE: 0.0-1.0`)
- Iteration cap to prevent infinite loops (our `_MAX_PHASE_RETRIES = 2`)
- Fallback behavior when cap reached (our backtrack context injection)

### Pattern 4: Handoff (Routing/Triage)

```
Triage Agent → (assess) → Specialist Agent 1 or 2 or ... N
```

Dynamic delegation based on content analysis. **Maps to our Classifier:**
RequestClassifier routes to Passthrough, Analytical, or Narrative mode based on system prompt
analysis, content heuristics, and session history.

### Pattern 5: Group Chat (Collaborative)

```
Agent 1 ↔ Agent 2 ↔ ... ↔ Agent N (shared thread, managed by coordinator)
```

Multiple agents collaborate in a shared conversation. **Not currently in Augmentum** but
could apply to complex analytical queries where UARF, narrative, and passthrough insights
are combined.

### The Coding Agent Pattern (Recursive Tool Loop)

The pattern shared by OpenCode, Qwen Code, Claude Code, Cursor, and Aider:

```
while not done:
    response = llm.generate(messages + tool_definitions)
    if response.has_tool_calls:
        for call in response.tool_calls:
            result = execute(call)
            messages.append(tool_result(call.id, result))
    else:
        done = True
        final_answer = response.text
```

**Safety features across all implementations:**
- Bounded retries (3-1000 depending on context)
- Structured error parsing (error → tool result → retry)
- Prior failure context injection
- Permission gates before destructive actions
- Output truncation for large results
- Cost/token tracking per step

**How UARF differs:** Instead of one unbounded loop, UARF uses multiple bounded loops
(one per phase), each with different tool access. This provides more structure but less
flexibility. The phases act as guardrails — the model can't spend all its tool budget on
search when it should be reasoning.


## 8. Cross-Project Patterns (All Sources)

### Pattern 1: Error-as-Result Loop (Universal)

The most common and effective retry pattern across ALL frameworks:

```
LLM generates tool call → Tool execution fails → Error message returned as "tool result"
→ LLM sees error and self-corrects → Retry → Success (or max retries reached)
```

Used by: LangChain, Semantic Kernel, AutoGen, CrewAI, Phidata, OpenClaw, Open WebUI.

**Augmentum status:** Partially implemented. Validation errors are injected back as user
messages in `_run_phase_with_tools`, but the loop relies on the LLM emitting another
`TOOL_CALL:` line — small models often don't retry.

### Pattern 2: Dual-Mode Tool Calling (Emerging Standard)

All major projects support both:
- **Native function calling** (structured JSON) for capable models
- **Text-based parsing** (ReAct/TOOL_CALL) for fallback

Open WebUI calls these "Native Mode" and "Default Mode". LangChain has `bind_tools()` +
`JsonOutputToolsParser`. CrewAI uses ReAct format universally.

**Augmentum status:** Only text-based. The infrastructure for native mode exists
(`InternalChatRequest.tools`, `Message.tool_calls`, backend passthrough) but the analytical
engine doesn't use it.

### Pattern 3: Tool Count Management

Research consistently shows accuracy degrades with more tools:
- BFCL data: ~3-5% accuracy drop per additional tool on small models
- Best practice: 3-7 tools maximum per context
- Advanced: Hierarchical selection (category → tool) or RAG-based tool retrieval

**Augmentum status:** Already implemented via `_PHASE_CATEGORIES`. RELEVANT gets 2 categories,
APPLY gets 6, VERIFY gets 2. This is well-aligned with best practices.

### Pattern 4: Progressive Skill/Tool Disclosure (OpenClaw)

Load only tool names+descriptions initially (~97 chars each). Inject full tool details only
when activated. Reduces baseline context consumption.

**Augmentum status:** Not implemented. All eligible tools are fully described in the prompt.
For APPLY phase (which can have 10+ tools), this adds significant token overhead.

### Pattern 5: Corrective Re-prompting (CrewAI, LangChain)

When the LLM doesn't follow the tool-call format, explicitly re-prompt with formatting
instructions rather than silently giving up:

```
Your response did not include a tool call in the required format.
If you need to use a tool, please use this exact format:
TOOL_CALL: tool_name
TOOL_INPUT: {"param": "value"}
```

**Augmentum status:** Not implemented. If parsing fails, the phase continues without tool use.

### Pattern 6: Intent Extraction Fallback

When explicit tool-call format isn't found, look for implicit intent:
- "I should search for X" → auto-execute web_search(query="X")
- "Let me calculate 25 * 1.08" → auto-execute calculator(expression="25 * 1.08")
- "I need to check this URL" → auto-execute web_fetch(url=...)

**Augmentum status:** Not implemented. We have `_get_proactive_suggestions()` that detects
URLs, math patterns, and search keywords in the user query, but not in the LLM's output.

### Pattern 7: Per-Session Serial Queuing (OpenClaw)

Ensure tool executions within a session are serialized to prevent race conditions.
Session key = `workspace:channel:userId`.

**Augmentum status:** Not implemented. asyncio tasks are not serialized per session.

### Pattern 8: Max Iteration Caps (Universal)

Every framework has a hard cap on tool call iterations:
- Semantic Kernel: 5 (default)
- CrewAI: 15-25
- Phidata: configurable `tool_call_limit`
- AutoGen: `max_consecutive_auto_reply`
- Augmentum: 3 per phase

**Our cap is conservative but appropriate for small models.** BFCL data shows multi-turn
accuracy degrades significantly for <13B models beyond 3-4 tool calls.

### Pattern 9: Tool Result Caching (LangChain, Phidata)

Cache tool results by `(tool_name, canonical_args_hash)` within a session. Prevents
redundant API calls when the LLM repeats a tool call.

**Augmentum status:** We have `PromptCache` and `DedupCache` but no per-tool-call result cache.

### Pattern 10: Native Function Calling First, Text Fallback (Coding Agents)

OpenCode, Qwen Code, Claude Code, and Cursor all use native function calling as the primary
method. Text-based parsing (ReAct, `TOOL_CALL:`) is a fallback for models that don't support
structured tool calls.

**The industry consensus is clear:** if the model supports native tool calling, use it.
It eliminates an entire class of parsing errors.

**Augmentum status:** Only uses text-based parsing. The infrastructure for native mode exists
but the analytical engine doesn't leverage it.

### Pattern 11: Parallel Tool Call Extraction

OpenCode, Qwen-Agent, and LangChain all support extracting multiple tool calls from a single
LLM response and executing them in parallel.

**Augmentum status:** Only extracts the first `TOOL_CALL:` match. If the LLM wants to search
and calculate simultaneously, only the first is executed.

### Pattern 12: Model-Specific Format Selection (Aider)

Aider benchmarks each model on its ability to produce correct outputs in different formats,
then selects the best format per model. This data-driven approach means each model gets the
format it's best at.

**Augmentum status:** One format for all models. Adding a model capability registry with
per-model format selection would improve reliability across the diverse models our users run.

### Pattern 13: Git Snapshot Rollback (OpenCode)

OpenCode creates git snapshots at each tool execution step. If a sequence of tool calls
produces bad results, the entire sequence can be rolled back.

**Augmentum status:** Not applicable (we don't modify files in the analytical pipeline), but
interesting for future file_ops tool enhancements.

### Pattern 14: Thinking Block Awareness (Qwen3)

Reasoning models output `<think>...</think>` blocks that can contain tool-call-like text.
Parsers must be aware of thinking blocks to avoid false-positive tool call extraction.

**Augmentum status:** Not handled. If a user runs Qwen3-Coder through Augmentum in analytical
mode, thinking blocks could confuse our `TOOL_CALL:` parser.

### Pattern 15: Circuit Breaker (Production Pattern)

If a tool consistently fails (e.g., SearXNG down), disable it temporarily and inform the LLM:

```python
class ToolCircuitBreaker:
    failure_threshold = 3
    reset_timeout = 300  # seconds
```

**Augmentum status:** Not implemented. A failing SearXNG would cause repeated failures across
all analytical requests until manually restarted.

---

## 9. Gap Analysis — Augmentum vs. Industry

### What Augmentum Does Well (Already Best Practice)

| Feature | Augmentum Implementation | Industry Comparison |
|---------|-------------------------|---------------------|
| Fuzzy tool name resolution | 4-level: exact → normalized → alias (89 entries) → substring | More thorough than most (LangChain: exact only; CrewAI: exact + alias) |
| Phase-gated tool access | `_PHASE_CATEGORIES` mapping | Matches "tool count management" best practice |
| Placeholder value rejection | Detects `<query>`, `<code>`, template artifacts | Unique — most frameworks don't check for this |
| Extra field stripping | Removes unknown kwargs before execution | LangChain does similar; most don't |
| Auto-search bypass | System-level search prevents redundant LLM-initiated searches | Unique to Augmentum's UARF pipeline |
| Refusal detection | 30+ refusal phrases checked | Unique — most frameworks don't handle model refusals |
| Forgiving JSON parsing | Single-quote fix, trailing comma strip, balanced brace extraction | On par with best frameworks |
| Concrete few-shot examples | Real values in prompts, not `<placeholder>` | Matches best practice recommendations |
| Sandboxed code execution | Separate Docker container, no network | More secure than OpenClaw, on par with Open WebUI |

### What Augmentum Is Missing

| Gap | Impact | Effort | Priority |
|-----|--------|--------|----------|
| Native tool calling (Ollama `tools` param) | High — eliminates JSON parsing for capable models | Medium | P0 |
| Structured output enforcement (Ollama `format` schema) | High — guarantees valid JSON for all models | Low | P0 |
| Type coercion for tool params | Medium — fixes `"42"` vs `42` for small models | Low | P1 |
| Corrective re-prompting on format failure | Medium — recovers from missed tool-call format | Low | P1 |
| Intent extraction fallback | Medium — catches "I should search for X" | Medium | P1 |
| Tool result caching (per-session) | Medium — prevents redundant API calls | Low | P2 |
| Circuit breaker for failing tools | Medium — graceful degradation when SearXNG is down | Low | P2 |
| Progressive tool disclosure | Low-Medium — reduces token overhead in APPLY phase | Medium | P2 |
| Parallel tool call extraction | Medium — multiple tools per LLM response | Medium | P1 |
| Thinking block awareness | Medium — Qwen3/DeepSeek reasoning models | Low | P2 |
| Per-model tool capability config | Low-Medium — optimize format per model | Medium | P2 |
| Model capability registry | Low-Medium — auto-select best format per model | Medium | P3 |
| Per-session serial queuing | Low — prevents rare race condition | Low | P3 |

---

## 10. Recommended Improvements

### P0: Hybrid Native + Text-Based Tool Calling

**What:** Detect if the model supports native tool calling and route accordingly.

**How:**
1. Maintain a set of model families with known native tool-call support:
   `llama3.1, llama3.2, llama3.3, qwen2.5, mistral, command-r, hermes3, firefunction`
2. When the model matches, pass `tools` parameter to the backend and parse `tool_calls`
   from the response (structured JSON, no text parsing needed)
3. When the model doesn't match, fall back to current `TOOL_CALL:` text-based parsing
4. The `InternalChatRequest.tools` field and `Message.tool_calls` field already exist —
   only the analytical engine needs updating

**Expected impact:** Near-100% tool call parsing success for supported models (currently ~85%).

### P0: Structured Output via Ollama JSON Schema Format

**What:** For tool-calling phases, pass a JSON Schema as the `format` parameter.

**How:**
1. Define a tool-call JSON Schema:
   ```json
   {
     "type": "object",
     "properties": {
       "thinking": { "type": "string" },
       "tool_call": {
         "type": "object",
         "properties": {
           "name": { "type": "string" },
           "arguments": { "type": "object" }
         },
         "required": ["name", "arguments"]
       },
       "response": { "type": "string" }
     },
     "required": ["thinking"]
   }
   ```
2. Set `internal_req.format = schema_dict` for RELEVANT/APPLY/VERIFY phases when using Ollama
3. Parse the structured JSON response — if `tool_call` is present, execute it; otherwise,
   use `response` as the phase output

**Expected impact:** Eliminates all JSON parsing failures for Ollama models. Works with any
model, not just those with native tool-call support.

**Note:** This is complementary to native tool calling. Use native when available (structured
`tool_calls` in response), structured output as fallback (valid JSON with our schema),
and text-based `TOOL_CALL:` as the last resort.

### P1: Type Coercion in Tool Parameter Validation

**What:** Automatically coerce parameter types before validation.

**Where:** Add to `_execute_tool()` in `engine.py`, between input extraction and schema
validation.

**How:**
```python
def _coerce_types(params: dict, schema: dict) -> dict:
    properties = schema.get("properties", {})
    for key, value in list(params.items()):
        expected_type = properties.get(key, {}).get("type")
        if expected_type == "integer" and isinstance(value, str):
            with suppress(ValueError): params[key] = int(value)
        elif expected_type == "number" and isinstance(value, str):
            with suppress(ValueError): params[key] = float(value)
        elif expected_type == "boolean" and isinstance(value, str):
            params[key] = value.lower() in ("true", "1", "yes")
        elif expected_type == "array" and isinstance(value, str):
            with suppress(json.JSONDecodeError):
                parsed = json.loads(value)
                if isinstance(parsed, list): params[key] = parsed
    return params
```

### P1: Corrective Re-prompting on Parse Failure

**What:** When the LLM's output doesn't contain `TOOL_CALL:` but the phase expects tools,
add a follow-up message with formatting instructions.

**Where:** In `_run_phase_with_tools()`, after parsing fails.

**How:**
```python
if not tool_name and phase in ("relevant", "apply") and attempt == 0:
    messages.append(Message(
        role="user",
        content=(
            "Your response did not include a tool call. If you need information "
            "or calculations, use this EXACT format on its own line:\n"
            "TOOL_CALL: tool_name\n"
            'TOOL_INPUT: {"param": "value"}\n\n'
            "If you have everything needed, continue your analysis."
        ),
    ))
    # Re-prompt the LLM
    continue
```

### P1: Intent Extraction Fallback

**What:** When explicit `TOOL_CALL:` parsing fails, scan the LLM's natural language output
for tool-use intent and auto-execute.

**How:**
```python
_INTENT_PATTERNS = [
    (r"(?:search|look up|find out)\s+(?:about\s+|for\s+)?['\"]?(.{10,80})['\"]?",
     "web_search", lambda m: {"query": m.group(1).strip().rstrip(".")}),
    (r"(?:calculate|compute)\s+(.{5,50})",
     "calculator", lambda m: {"expression": m.group(1).strip()}),
    (r"(?:fetch|visit|check)\s+(?:the\s+)?(?:url\s+|page\s+)?(https?://\S+)",
     "web_fetch", lambda m: {"url": m.group(1).strip()}),
]
```

Only trigger as a last resort (after text-based and native parsing both fail).

### P2: Tool Result Caching

**What:** Cache tool results by `(tool_name, frozenset(args.items()))` within a session.

**Where:** Add to `_execute_tool()` in `engine.py`.

**How:**
```python
cache_key = (tool_name, tuple(sorted(tool_input.items())))
if cache_key in self._tool_result_cache:
    log.info("tool_result_cache_hit", tool=tool_name)
    return self._tool_result_cache[cache_key]
result = await tool.execute(**tool_input)
if result.success:
    self._tool_result_cache[cache_key] = result
return result
```

### P2: Circuit Breaker for External Tools

**What:** Track failure counts per tool; disable temporarily after N consecutive failures.

**How:**
```python
class ToolCircuitBreaker:
    def __init__(self, threshold=3, reset_seconds=300):
        self._failures: dict[str, int] = {}
        self._opened: dict[str, float] = {}

    def record_failure(self, tool_name: str) -> None:
        self._failures[tool_name] = self._failures.get(tool_name, 0) + 1
        if self._failures[tool_name] >= self._threshold:
            self._opened[tool_name] = time.time()

    def is_available(self, tool_name: str) -> bool:
        if tool_name not in self._opened:
            return True
        if time.time() - self._opened[tool_name] > self._reset_seconds:
            del self._opened[tool_name]
            self._failures[tool_name] = 0
            return True
        return False

    def record_success(self, tool_name: str) -> None:
        self._failures.pop(tool_name, None)
        self._opened.pop(tool_name, None)
```

---

## Sources

### OpenClaw / ClawdBot
- [OpenClaw GitHub Repository](https://github.com/clawdbot/clawdbot)
- [ClawBot's Architecture Explained — Towards AI](https://pub.towardsai.net/clawbots-architecture-explained-how-a-lobster-conquered-100k-github-stars-4c02a4eae078)
- [Lessons from OpenClaw's Architecture for Agent Builders](https://blog.agentailor.com/posts/openclaw-architecture-lessons-for-agent-builders)
- [OpenClaw Complete Guide 2026](https://www.jitendrazaa.com/blog/ai/clawdbot-complete-guide-open-source-ai-assistant-2026/)
- [Lobster Workflow Engine](https://github.com/openclaw/lobster)
- [Awesome OpenClaw Skills](https://github.com/VoltAgent/awesome-openclaw-skills)

### Open WebUI
- [Open WebUI Tools Documentation](https://docs.openwebui.com/features/extensibility/plugin/tools/)
- [Open WebUI Functions System — DeepWiki](https://deepwiki.com/open-webui/docs/4.2-functions-system)
- [Open WebUI Tools Development — DeepWiki](https://deepwiki.com/open-webui/docs/4.1-tools-development)
- [Open WebUI Pipelines GitHub](https://github.com/open-webui/pipelines)
- [Tools Usage Guide — Discussion #3134](https://github.com/open-webui/open-webui/discussions/3134)
- [Haervwe Open WebUI Tools Collection](https://github.com/Haervwe/open-webui-tools)

### Tool Calling Reliability
- [Local LLM Tool Calling Evaluation — Docker Blog](https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/)
- [Advanced Tool Calling in LLM Agents — SparkCo](https://sparkco.ai/blog/advanced-tool-calling-in-llm-agents-a-deep-dive)
- [Reliability for Unreliable LLMs — Stack Overflow](https://stackoverflow.blog/2025/06/30/reliability-for-unreliable-llms/)
- [Tool Calling Best Practices — Laurent Kubaski](https://medium.com/@laurentkubaski/tool-or-function-calling-best-practices-a5165a33d5f1)

### Structured Output
- [Ollama Structured Outputs Documentation](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama Structured Outputs Blog](https://ollama.com/blog/structured-outputs)
- [How Ollama's Structured Outputs Work](https://blog.danielclayton.co.uk/posts/ollama-structured-outputs/)
- [Structured Output with Ollama — Instructor](https://python.useinstructor.com/integrations/ollama/)

### Coding Agents
- [OpenCode — AI Coding Agent](https://opencode.ai/)
- [OpenCode GitHub](https://github.com/opencode-ai/opencode)
- [How Coding Agents Actually Work: Inside OpenCode](https://cefboud.com/posts/coding-agents-internals-opencode-deepdive/)
- [OpenCode Tools Documentation](https://opencode.ai/docs/tools/)
- [OpenCode Agents Documentation](https://opencode.ai/docs/agents/)
- [Qwen Code GitHub](https://github.com/QwenLM/qwen-code)
- [Qwen-Agent GitHub](https://github.com/QwenLM/Qwen-Agent)
- [Qwen3-Coder: Agentic Coding in the World](https://qwenlm.github.io/blog/qwen3-coder/)
- [Qwen Function Calling Documentation](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Qwen3 Function Calling and Tool Use — DeepWiki](https://deepwiki.com/QwenLM/Qwen3/4.3-function-calling-and-tool-use)
- [Qwen-Agent function_calling.py Source](https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/function_calling.py)

### Agent Orchestration Patterns
- [AI Agent Orchestration Patterns — Microsoft Azure](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
- [Best AI Coding Agents for 2026 — Faros AI](https://www.faros.ai/blog/best-ai-coding-agents-2026)
- [AI Coding Agents: Coherence Through Orchestration](https://mikemason.ca/writing/ai-coding-agents-jan-2026/)
- [Agent Loops: Foundation of Multi-Agent Ecosystems](https://www.ema.co/additional-blogs/addition-blogs/building-ai-agents-agent-loop)

### Docker / llama.cpp
- [Docker Desktop 4.42 llama.cpp Streaming and Tool Calling](https://www.ajeetraina.com/docker-desktop-4-42-llama-cpp-gets-streaming-and-tool-calling-support/)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
