# Reasoning Pipeline Editor — Design Document

## The Idea

Replace the passive "Analytical Reasoning" side panel with a **Reasoning Flow Editor** — a visual pipeline designer where users build, customize, and manage how Augmentum thinks through problems.

No other proxy, middleware, or AI frontend lets end-users design reasoning pipelines visually at runtime. Developer frameworks (LangChain, CrewAI) require code. ComfyUI does this for image generation with node graphs. **We do it for reasoning.**

---

## Core Concept: Reasoning Flows

A **Flow** is a named, ordered sequence of **Steps**. Each step is an LLM call with its own system prompt, tool access, and behavior flags. Flows are stored per-user, shareable as JSON, and selectable at runtime.

Instead of one hardcoded UARF pipeline, users get:

- **Built-in templates** that cover common patterns out of the box
- **Full customization** — add, remove, reorder, edit any step
- **Per-step tool control** — choose exactly which tools each step can use
- **Per-step prompt editing** — modify the system prompt for any step
- **Complexity gating** — steps can be set to only run for certain complexity levels
- **Model-aware defaults** — different flows can be pinned to different models

---

## Why This Is Novel

1. **No-code reasoning design** — Visual drag-and-drop, not Python chains
2. **Runtime, not build-time** — Change your reasoning pipeline mid-conversation without restarting anything
3. **Model-agnostic** — Same flow works across Ollama, OpenAI, llama.cpp, any backend
4. **Sits in the proxy** — Works transparently with any frontend (Open WebUI, SillyTavern, curl)
5. **Shareable** — Export/import flows as JSON. Community can share what works for specific models or domains
6. **Adaptive** — Complexity gating means a single flow handles simple and complex queries differently without needing separate flows

---

## Data Model

### Flow

```
Flow:
  id: string (uuid)
  name: string ("Research Deep Dive", "Quick Answer", "Code Review")
  description: string
  icon: string (emoji, optional)
  version: int (for conflict resolution)

  # Routing
  auto_select: bool (can the classifier pick this flow automatically?)
  trigger_domains: string[] (e.g. ["code", "math", "science"])
  trigger_keywords: string[] (e.g. ["debug", "review", "analyze"])
  pinned_models: string[] (e.g. ["qwen2.5:72b"] — auto-use this flow for these models)

  # Behavior
  auto_search: bool (enable web search phase when needed)
  max_tool_calls_per_step: int (default 3)

  steps: Step[]
```

### Step

```
Step:
  id: string (uuid)
  name: string (user-visible, e.g. "Assess", "Deep Research", "Verify Logic")

  # Prompt
  system_prompt: string (the instructions for this step)
  user_template: string (how to build the user message — uses variables)

  # Tools
  tool_categories: string[] (e.g. ["search", "execute", "verify"])
  tool_names: string[] (specific tools, overrides categories if set)

  # Behavior flags
  role: string (see "Step Roles" below)
  complexity_gate: string[] (e.g. ["moderate", "complex"] — skip on simple)
  stream_to_user: bool (false = internal step, true = stream output to user)
  output_cap: int (max chars forwarded to next step, 0 = unlimited)
  enabled: bool
```

### Step Roles

Roles define special parsing/behavior without hardcoding to phase names:

| Role | Behavior |
|------|----------|
| `classify` | Parse output for `COMPLEXITY:` line. Determines complexity gating for downstream steps. Must be first step (or early). |
| `search` | Triggers auto-search machinery. Injects search results into downstream steps. |
| `analyze` | Standard LLM reasoning step. Gets prior step outputs + search context. |
| `verify` | Parse output for `VERIFIED:` and `CONFIDENCE:`. Can trigger backtracking. |
| `respond` | Final step — output streams directly to the user. |
| `transform` | No LLM call — applies a template/regex to previous output. Cheap, instant. |

Most steps will be `analyze`. Roles are an **opt-in enhancement**, not a requirement. A flow with 3 generic `analyze` steps and one `respond` step works fine.

---

## Prompt Variables

System prompts and user templates support variables that resolve at runtime:

| Variable | Description |
|----------|-------------|
| `{query}` | The user's original question |
| `{conversation}` | Recent conversation history (formatted) |
| `{search_results}` | Web search results (if search step ran) |
| `{previous_output}` | Output from the immediately preceding step |
| `{step:Name}` | Output from a specific named step (e.g. `{step:Assess}`) |
| `{all_outputs}` | Concatenated outputs from all prior steps |
| `{complexity}` | Detected complexity level (from classify step) |
| `{model}` | Current model name |
| `{tools}` | Auto-generated tool description section (based on step's tool config) |

Default user template (if not customized):
```
## Query
{query}

{conversation}
{previous_output}
{search_results}
```

This means users who just want to tweak the system prompt don't need to touch the user template at all — the default handles data plumbing automatically.

---

## Built-in Flow Templates

### 1. Standard (current UARF behavior, default)
```
[Classify] → [Identify*] → [Research*] → [Analyze] → [Verify*] → [Respond]
  * = skipped on simple queries
```

### 2. Quick Answer (for powerful models)
```
[Respond]
```
Single step. No overhead. Just adds tools and search when needed. Good for GPT-4, Claude, Qwen 72B — models that don't need hand-holding.

### 3. Research
```
[Classify] → [Search] → [Cross-Reference] → [Synthesize] → [Fact-Check] → [Respond]
```
Heavy on search and verification. Good for current events, fact-checking, homework.

### 4. Code Review
```
[Parse Code] → [Find Issues] → [Suggest Fixes] → [Verify Fixes] → [Respond]
```
Each step has a code-focused system prompt. Parse step uses `python_exec` tool for syntax checking. Verify step re-runs code.

### 5. Debate / Steel Man
```
[Understand Position] → [Argue For] → [Argue Against] → [Synthesize] → [Respond]
```
Forces the model to consider both sides before concluding. Great for opinion questions, ethics, policy.

### 6. Socratic Teacher
```
[Understand] → [Identify Misconceptions] → [Ask Guiding Questions] → [Respond]
```
Instead of giving the answer directly, guides the user toward understanding. The respond step's prompt explicitly says "help them discover the answer, don't just tell them."

### 7. Math / Science
```
[Parse Problem] → [Set Up Equations] → [Solve (with calculator)] → [Verify (with calculator)] → [Respond]
```
Both Solve and Verify steps have the `math_verify` and `calculator` tools enabled. The verify step re-solves independently and compares.

### 8. Creative Writing
```
[Understand Intent] → [Brainstorm Approaches] → [Draft] → [Refine] → [Respond]
```
No tools needed. Prompts focus on creativity, voice, and style. Draft step streams to let user see the raw output.

### 9. Minimal (for weak models)
```
[Respond]
```
Same as Quick Answer but with a more structured, hand-holding system prompt that compensates for weaker model capabilities. Keeps token usage low.

### 10. Custom (blank canvas)
```
[Step 1]
```
Starts with one empty step. User builds from scratch.

---

## Complexity Gating

Each step has a `complexity_gate` field: an array of complexity levels it runs on.

- `["simple", "moderate", "complex"]` = always runs (default)
- `["moderate", "complex"]` = skips on simple
- `["complex"]` = only runs on complex queries
- `[]` = always runs (empty = no gate)

This replaces the current hardcoded `_SIMPLE_PHASES` / `_MODERATE_PHASES` / `_FULL_PHASES` pattern. Users get the same adaptive behavior but can customize it per-step.

For this to work, **one step must have the `classify` role** (or complexity defaults to "moderate" and all steps run).

---

## Flow Selection / Routing

How does the system decide which flow to use?

1. **Explicit user selection** — User picks a flow from a dropdown in the UI (or via API parameter)
2. **Model pinning** — If the current model matches a flow's `pinned_models`, use it
3. **Domain matching** — The classifier detects the query domain → match against flow `trigger_domains`
4. **Keyword matching** — Query contains a flow's `trigger_keywords`
5. **Default fallback** — Use the flow marked as default (Standard template)

Priority: explicit > model pin > domain > keyword > default

This means a user could pin "Code Review" to `deepseek-coder:33b` and "Quick Answer" to `claude-3-opus`, and the right flow activates automatically based on which model they're talking to.

---

## UI Design: The Flow Editor Panel

The side panel transforms from a passive phase viewer into a flow management & editing interface.

### Panel Layout

```
┌─────────────────────────────────────┐
│ Reasoning Flows          [+ New] [×] │
├─────────────────────────────────────┤
│ ┌─────────────────────────────────┐ │
│ │ ▼ Standard (default)            │ │
│ │   Quick Answer                  │ │
│ │   Research                      │ │
│ │   Code Review                   │ │
│ │   My Custom Flow ★              │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ ── Standard ──────────── [Edit] ── │
│                                     │
│  ● Classify          [always]  ✓    │
│  │                                  │
│  ● Identify      [mod+complex] ✓   │
│  │                                  │
│  ● Research      [mod+complex] ✓   │
│  │                                  │
│  ● Analyze           [always]  ✓   │
│  │                                  │
│  ● Verify        [mod+complex] ✓   │
│  │                                  │
│  ● Respond           [always]  ✓   │
│                                     │
│  [+ Add Step]  [Import] [Export]    │
├─────────────────────────────────────┤
│ ── Step: Analyze ──────────────── │
│                                     │
│ System Prompt:                      │
│ ┌─────────────────────────────────┐ │
│ │ Solve this query using the      │ │
│ │ evidence provided. Show your    │ │
│ │ reasoning at each step...       │ │
│ └─────────────────────────────────┘ │
│                                     │
│ Tools: ☑ web_search ☑ python_exec  │
│        ☑ calculator  ☐ web_fetch    │
│        ☐ file_ops    ☑ math_verify  │
│                                     │
│ Runs on: ☑ Simple ☑ Moderate       │
│          ☑ Complex                  │
│                                     │
│ Role: [analyze ▼]                   │
│ Stream to user: ☐                   │
│ Max output: [800] chars             │
│                                     │
│ Variables: {query} {previous_output}│
│           {search_results} {step:X} │
└─────────────────────────────────────┘
```

### Interactions

- **Click a flow** → loads it into the editor, shows its steps
- **Click a step** → expands inline to show prompt, tools, settings
- **Drag steps** → reorder (drag handle on left)
- **Toggle checkbox** → enable/disable a step without deleting it
- **[+ Add Step]** → adds a blank step at the bottom (or between existing steps)
- **Right-click step / ⋮ menu** → Duplicate, Delete, Move Up, Move Down
- **[Edit] on flow** → Edit name, description, routing rules
- **[+ New]** → Create from template or blank canvas
- **[Import/Export]** → JSON file for sharing
- **★ star** → Set as default flow

### During Active Generation

When a flow is running, the panel shows live progress (similar to current behavior but using the flow's actual steps):

```
┌─────────────────────────────────────┐
│ Running: Standard                    │
├─────────────────────────────────────┤
│  ✓ Classify (simple)       0.3s     │
│  — Identify (skipped)               │
│  — Research (skipped)               │
│  ● Analyze...              2.1s     │
│  ○ Verify                           │
│  ○ Respond                          │
├─────────────────────────────────────┤
│ Tools used: web_search (1)          │
│ Complexity: simple                   │
│ Tokens: ~340                         │
└─────────────────────────────────────┘
```

This gives the live view actual purpose — it's showing your custom flow executing, not a generic hardcoded pipeline.

---

## Storage & Persistence

### SQLite Schema (new migration)

```sql
CREATE TABLE reasoning_flows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    icon TEXT DEFAULT '',
    version INTEGER DEFAULT 1,
    is_default BOOLEAN DEFAULT 0,
    is_builtin BOOLEAN DEFAULT 0,
    auto_select BOOLEAN DEFAULT 1,
    trigger_domains TEXT DEFAULT '[]',   -- JSON array
    trigger_keywords TEXT DEFAULT '[]',  -- JSON array
    pinned_models TEXT DEFAULT '[]',     -- JSON array
    auto_search BOOLEAN DEFAULT 1,
    max_tool_calls_per_step INTEGER DEFAULT 3,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE reasoning_flow_steps (
    id TEXT PRIMARY KEY,
    flow_id TEXT NOT NULL REFERENCES reasoning_flows(id) ON DELETE CASCADE,
    sort_order INTEGER NOT NULL,
    name TEXT NOT NULL,
    system_prompt TEXT DEFAULT '',
    user_template TEXT DEFAULT '',
    role TEXT DEFAULT 'analyze',
    tool_categories TEXT DEFAULT '[]',   -- JSON array
    tool_names TEXT DEFAULT '[]',        -- JSON array
    complexity_gate TEXT DEFAULT '[]',   -- JSON array, empty = always
    stream_to_user BOOLEAN DEFAULT 0,
    output_cap INTEGER DEFAULT 800,
    enabled BOOLEAN DEFAULT 1
);

CREATE INDEX idx_flow_steps_flow ON reasoning_flow_steps(flow_id, sort_order);
```

### API Endpoints

```
GET    /api/reasoning/flows              — List all flows
GET    /api/reasoning/flows/{id}         — Get a flow with its steps
POST   /api/reasoning/flows              — Create a flow
PUT    /api/reasoning/flows/{id}         — Update a flow
DELETE /api/reasoning/flows/{id}         — Delete a flow (not builtins)
POST   /api/reasoning/flows/{id}/clone   — Clone a flow (including builtins)
POST   /api/reasoning/flows/import       — Import from JSON
GET    /api/reasoning/flows/{id}/export  — Export as JSON
PUT    /api/reasoning/flows/{id}/default — Set as default
GET    /api/reasoning/templates          — List available templates
```

---

## Engine Integration

### How the handler changes

Currently `handler.py` hardcodes `_SIMPLE_PHASES`, `_MODERATE_PHASES`, `_FULL_PHASES`. Instead:

1. **Flow resolution** — Before streaming begins, resolve which flow to use (explicit selection > model pin > domain match > default)
2. **Step filtering** — After ASSESS determines complexity, filter steps by `complexity_gate`
3. **Step execution** — Loop through remaining steps in order, each one:
   - Builds system prompt from step config (with variable substitution)
   - Builds user message from template (with variable substitution)
   - Resolves tools from step's `tool_categories` / `tool_names`
   - Calls the LLM
   - Parses output based on step's `role` (classify → complexity, verify → confidence, etc.)
   - Stores output for downstream variable resolution
4. **The last `respond` role step** streams to the user (all prior steps are internal)

### What stays the same

- Tool calling tiers (native/structured/text) — model-dependent, orthogonal to flows
- Auto-search machinery — triggered by the `search` role step
- Backtracking on failed verification — triggered by `verify` role step
- Thinking block display in messages — shows flow steps instead of hardcoded phases
- Streaming infrastructure — unchanged
- Prompt condensation, tool resolution, proactive suggestions — all still work

### Backward compatibility

- The **Standard** built-in flow replicates exact current UARF behavior
- If no custom flows exist, behavior is identical to today
- The `a/` model prefix still triggers analytical mode, which uses the active flow
- All existing config settings (`uarf_*`) still work as defaults for the Standard flow

---

## Implementation Phases

### Phase A: Data Model + Storage + API
- `ReasoningFlow` and `FlowStep` Pydantic models
- SQLite migration for `reasoning_flows` + `reasoning_flow_steps`
- `FlowStore` class (CRUD, import/export)
- Built-in templates seeded on first run
- REST API endpoints
- ~400 lines backend

### Phase B: Engine Integration
- `FlowResolver` — selects the active flow based on routing rules
- Refactor `handler.py` to read steps from the active flow instead of hardcoded phase lists
- Variable substitution system for prompts
- Role-based parsing (classify, verify, respond)
- ~300 lines refactor

### Phase C: UI — Flow List + Editor
- Replace reasoning panel contents with flow list and step editor
- Flow selection dropdown, create/clone/delete
- Step editing: prompt textarea, tool checkboxes, complexity gate, role selector
- Drag-to-reorder steps
- Import/export buttons
- ~500 lines JS + HTML + CSS

### Phase D: UI — Live Execution View
- When a flow is running, panel switches to live progress view
- Shows which step is executing, timing, tools used, complexity
- Returns to editor view when generation completes
- ~200 lines JS

### Phase E: Polish + Testing
- Tests for flow store, resolver, engine integration
- Template refinement based on testing with different models
- Edge case handling (empty flows, all steps disabled, no classify step, etc.)

---

## Open Questions

1. **Should built-in templates be editable, or clone-only?** Leaning toward clone-only — users clone a template and customize the clone. Prevents accidental destruction of defaults and makes reset easy.

2. **Per-session flow override?** Should users be able to switch flows mid-session, or is it session-wide? Leaning toward per-message — the dropdown selects the flow for the next message, not the whole session.

3. **Flow versioning?** If a user edits a flow, do previous messages that used the old version still show the old steps in their thinking block? Probably yes — the persisted reasoning data on each message captures the steps that were used, not the current flow definition.

4. **How granular should tool control be?** Currently: tool categories (search, execute, verify, fetch, file). Could also allow individual tool names. Leaning toward both — categories for quick config, individual names for fine-tuning.

5. **Should the `transform` role (no-LLM template step) be in v1?** It's useful (e.g., "reformat the output of step 2 as a bullet list" without burning an LLM call) but adds complexity. Could defer to v2.

6. **Community sharing?** Import/export covers local sharing. A future community hub where users upload flows for specific models/domains would be powerful but is out of scope for v1.

---

## What This Enables

- A user running Qwen 2.5 7B creates a "Careful Math" flow with extra verification steps because the model makes arithmetic errors
- A user running GPT-4o creates a "Quick Answer" flow with just one step because the model is capable enough
- A researcher creates a "Literature Review" flow: Search → Extract Claims → Cross-Reference → Synthesize → Cite
- A developer creates a "Bug Triage" flow: Parse Error → Search Docs → Hypothesize → Suggest Fix → Verify Fix
- A teacher creates a "Socratic" flow that never gives direct answers
- Users share flows as JSON files: "here's my optimized research flow for Mistral 7B"
- A power user pins different flows to different models so the right reasoning strategy auto-activates

This turns Augmentum from "a proxy that enhances LLM responses" into **"a platform where users design how AI thinks."**
