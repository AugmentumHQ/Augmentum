# Tool Chain Execution Design

## Problem

The current passthrough tool loop is reactive and single-threaded:

```
User → LLM picks tool → execute → feed result back → LLM picks next tool → execute → … → respond
```

This works for "search X and tell me" but breaks down for multi-step tasks because:

1. **Small models lose context** after 2-3 tool iterations — they forget what they were doing
2. **No parallelism** — even independent tools wait for sequential LLM round-trips
3. **No dependency awareness** — the model can't express "use step 1's result in step 3"
4. **Each iteration is expensive** — full prompt re-processing on every tool call

Meanwhile, agentic mode is too heavy — persistence, checkpoints, approval gates, flow templates, full UI panel. We need something **between** the simple loop and full agentic.

## Design: Two-Layer Chain System

### Layer 1: Adaptive Chains (model-planned)

For novel requests. The model creates the plan on the fly. No user setup needed.

### Layer 2: Custom Flows (user-defined)

For recurring patterns. Users define reusable chain templates. Faster and more reliable because they skip planning entirely.

**Both layers share the same wave executor.**

---

## Adaptive Chain Execution

### Phase 1: Detect + Plan + First Actions (one LLM call)

Zero-cost heuristic detects when planning is needed:

```python
def detect_complexity(query: str, matched_tools: list[Tool]) -> bool:
    # Multiple distinct tool categories = likely multi-step
    categories = {t.category for t in matched_tools}
    if len(categories) >= threshold:
        return True
    # Explicit multi-step language
    if MULTI_STEP_RE.search(query):  # "then", "after that", "first...then"
        return True
    return False
```

Simple queries skip planning entirely → zero overhead.

When complex, the **first LLM call combines planning with first tool calls**:

```
System: You have these tools: [schemas].
Plan your approach as a numbered list, then immediately call any tools you can.

User: [original request]
```

Model outputs plan text + native tool calls in one response. No wasted planning-only call.

### Phase 2: Wave execution with plan injection

After wave 1 tools complete, the next LLM call gets focused context:

```
System: You're executing a multi-step plan.

Plan progress:
1. ✓ Search for the video → [found: "Math in Cooking - YouTube"]
2. → Get the transcript — EXECUTE THIS NOW
3. ○ Verify math claims
4. ○ Create summary image

Call the next tool(s).
```

The model sees:
- **What it planned** (prevents drift)
- **What's done** with results (only relevant ones — not entire growing conversation)
- **What to do next** (focused instruction)

And responds with tool call(s), args naturally resolved from the visible results.

### Phase 3: Continue waves

Each wave:
1. Inject updated plan status + relevant prior results
2. Model makes next tool call(s)
3. Execute (parallel via `asyncio.gather` if multiple independent calls)
4. Update plan status
5. If model deviates from plan based on unexpected results → that's fine, plan is a compass not a railroad

### Phase 4: Final synthesis

```
All steps complete. Results:
[Step 1 - web_search]: ...
[Step 2 - youtube_transcript]: ...
[Step 3 - calculator]: ...
[Step 4 - image_generation]: Image generated.

Synthesize a complete response for the user.
```

### Why This Beats the Current Loop

Same ~4 LLM calls, but each one has:
- **Focused context** (plan + relevant results only, not entire growing history)
- **Plan anchor** (model always knows what it's doing)
- **Parallel execution** within waves (independent steps run simultaneously)
- **Adaptability** (model can deviate if results change the situation)

### Fallback

If plan parsing fails → still got tool calls from the first response → feed those into the existing simple loop. One wasted prompt line, business as usual.

---

## Custom Flows (User-Defined Chains)

Users create reusable chain templates for recurring workflows.

### Flow Definition

```json
{
  "name": "Video Analyzer",
  "description": "Search, transcribe, verify, visualize",
  "trigger": "analyze (this |the |a )?video",
  "steps": [
    {
      "id": 1,
      "tool": "web_search",
      "input": {"query": "{{query}}"},
      "reason": "Find the video"
    },
    {
      "id": 2,
      "tool": "youtube_transcript",
      "needs": [1],
      "reason": "Get the transcript"
    },
    {
      "id": 3,
      "tool": "calculator",
      "needs": [2],
      "reason": "Verify any math claims"
    },
    {
      "id": 4,
      "tool": "image_generation",
      "needs": [1, 2, 3],
      "reason": "Create summary image"
    }
  ]
}
```

### Template Variables

Steps can reference prior results and user context:
- `{{query}}` — the user's message text
- `{{step.N.output}}` — output text from step N
- `{{step.N.metadata.KEY}}` — metadata field from step N (e.g. `{{step.1.metadata.urls.0}}`)
- `{{user}}` — user's name from memory (if available)

When a step has `needs` but no `input`, args are resolved via a focused LLM call (same as adaptive chains).

When a step has `input` with template variables, Augmentum resolves templates directly — no LLM call needed.

### Triggering

Three ways to activate a custom flow:

1. **Explicit:** User types `/flow Video Analyzer` or `/f video` (fuzzy name match)
2. **Pattern-matched:** `trigger` regex matches the user's query automatically
3. **UI selector:** Dropdown in the chat toolbar showing saved flows

If no custom flow matches → adaptive chain planner kicks in as fallback.

### Storage

SQLite table (migration file):

```sql
CREATE TABLE IF NOT EXISTS custom_flows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    trigger_pattern TEXT DEFAULT '',
    steps_json TEXT NOT NULL,         -- JSON array of step definitions
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    enabled INTEGER DEFAULT 1
);
```

### REST API

```
GET    /api/flows              — list all flows
POST   /api/flows              — create a flow
GET    /api/flows/{id}         — get flow details
PUT    /api/flows/{id}         — update a flow
DELETE /api/flows/{id}         — delete a flow
POST   /api/flows/{id}/run     — manually trigger a flow with a query
GET    /api/flows/match?q=...  — check which flow matches a query
POST   /api/flows/import       — import flows from JSON
GET    /api/flows/export       — export all flows as JSON
```

### UI

Settings panel: flow list with create/edit/delete/toggle. Flow editor with:
- Name, description, trigger pattern (with regex test)
- Step builder: add/remove/reorder steps, pick tools from dropdown, set dependencies
- Template variable reference
- Test run button

---

## Shared Wave Executor

Both adaptive chains and custom flows feed into the same execution engine:

```python
@dataclass
class ChainStep:
    id: int
    tool: str
    input: dict | None = None     # pre-specified args (may contain {{templates}})
    needs: list[int] = field(default_factory=list)
    reason: str = ""

@dataclass
class ChainPlan:
    steps: list[ChainStep]
    source: str = "adaptive"      # "adaptive" or "custom:{flow_id}"

@dataclass
class StepResult:
    step_id: int
    tool_name: str
    output: str
    metadata: dict
    success: bool
```

### Wave Execution Algorithm

```python
async def execute_chain(plan, backend, tool_registry, ...):
    results: dict[int, StepResult] = {}
    remaining = list(plan.steps)

    while remaining:
        # Find steps whose dependencies are all satisfied
        ready = [s for s in remaining if all(d in results for d in s.needs)]
        if not ready:
            break  # circular dependency or unresolvable — bail

        # Execute ready steps in parallel
        wave_tasks = []
        for step in ready:
            wave_tasks.append(execute_step(step, results, backend, tool_registry))

        wave_results = await asyncio.gather(*wave_tasks, return_exceptions=True)

        for step, result in zip(ready, wave_results):
            if isinstance(result, Exception):
                results[step.id] = StepResult(step.id, step.tool, f"Error: {result}", {}, False)
            else:
                results[step.id] = result
            remaining.remove(step)

    return results


async def execute_step(step, prior_results, backend, tool_registry):
    tool = tool_registry.resolve(step.tool)

    if step.input:
        # Args pre-specified (possibly with templates) — resolve and execute
        args = resolve_templates(step.input, prior_results)
        # ... coerce + execute
    else:
        # Needs arg resolution — focused LLM call
        context = format_dependency_results(step.needs, prior_results)
        # LLM call: "Given these results, call {tool} for: {reason}"
        # Parse tool call from response, execute
```

### Error Handling

- **Step fails:** Mark as failed. Dependent steps get skipped (their `needs` never satisfied). Independent steps continue.
- **Plan parse fails (adaptive):** Fall back to existing simple loop.
- **Template resolution fails (custom):** Fall back to LLM arg resolution for that step.
- **All steps fail:** Return error summary to user.
- **Circular dependencies:** Detected in wave loop (no ready steps but remaining exists). Break and report.

---

## Streaming Integration

Each step emits metadata chunks via the existing `augmentum` channel:

```python
# Chain started
yield InternalStreamChunk(augmentum={"chain": {"status": "running", "total_steps": 4, "source": "adaptive"}})

# Step progress
yield InternalStreamChunk(augmentum={"chain_step": {"id": 1, "tool": "web_search", "status": "running"}})
yield InternalStreamChunk(augmentum={"chain_step": {"id": 1, "tool": "web_search", "status": "done", "preview": "Found 5 results"}})

# Wave progress (steps 2+3 starting in parallel)
yield InternalStreamChunk(augmentum={"chain_step": {"id": 2, "tool": "youtube_transcript", "status": "running"}})
yield InternalStreamChunk(augmentum={"chain_step": {"id": 3, "tool": "calculator", "status": "running"}})
```

UI renders as a progress pipeline:
```
[1/4 Web Search ✓] → [2/4 YouTube ✓] → [3/4 Calculator ⟳] → [4/4 Image Gen ○]
```

---

## Config

```python
# --- Tool Chains ---
passthrough_chain_enabled: bool = True         # enable multi-step chain execution
passthrough_chain_complexity_threshold: int = 2  # min tool categories to trigger adaptive planning
passthrough_chain_max_steps: int = 6           # max steps per chain
```

---

## Resolution Order

When a request comes in with tools enabled:

```
1. Check custom flows:
   a. Explicit /flow command? → use that flow
   b. Trigger pattern match? → use highest-priority matching flow
2. If no custom flow matched:
   a. detect_complexity() says complex? → adaptive chain planning
   b. Simple? → existing single-step tool loop
```

---

## Files

| File | Purpose |
|------|---------|
| `augmentum/tools/chain.py` | ChainStep, ChainPlan, StepResult, WaveExecutor, ToolChainPlanner |
| `augmentum/tools/custom_flows.py` | CustomFlowStore (SQLite CRUD), template resolution, trigger matching |
| `augmentum/modes/passthrough/handler.py` | Integration: `_run_chain()`, `_run_custom_flow()` |
| `augmentum/proxy/flow_routes.py` | REST API for custom flows |
| `augmentum/state/migrations/032_custom_flows.sql` | SQLite table |
| `augmentum/config.py` | 3 new settings |
| `tests/test_tool_chain.py` | Wave executor, adaptive planning, complexity detection |
| `tests/test_custom_flows.py` | Custom flow CRUD, trigger matching, template resolution |
| `ui/scripts/flows.js` | Flow editor UI |

## Implementation Order

1. **Wave executor** (`chain.py`) — the shared engine. Testable in isolation.
2. **Complexity detection + adaptive planning** — integrate into passthrough handler.
3. **Custom flow store** — SQLite CRUD + template resolution.
4. **Flow routes** — REST API.
5. **Handler integration** — resolution order (custom flow → adaptive → simple loop).
6. **Streaming metadata** — chain progress chunks.
7. **UI** — flow editor + chain progress display.
