# Speculative Tool Execution — Design Document

## Problem

When the LLM decides to use a tool during a voice conversation, there is a
silence gap between the user's question and the eventual spoken answer:

```
User speaks → STT → LLM decides tool call → [TOOL EXECUTES: 1-5s silence] → LLM response → TTS
```

Current mitigation: narration cues ("Searching the web…") fill some of the gap.
But complex tool chains (search → fetch → analyze) can take 5-10 seconds, and
a single cue doesn't cover that.

## Solution: Dual-Track Speculative Execution

Run tool execution and filler speech generation in parallel.  The LLM produces
filler/transition speech while tools run in the background.  When tools complete,
the LLM incorporates results into the final answer.

### Architecture

```
                     ┌─────────────────────┐
                     │   LLM Response       │
                     │   (streaming)        │
                     └──────┬──────────────┘
                            │
              ┌─────────────┼──────────────┐
              ▼                            ▼
    ┌──────────────────┐         ┌──────────────────┐
    │  Track A: Filler │         │  Track B: Tools   │
    │  Speech → TTS    │         │  Execute in bg    │
    │                  │         │                   │
    │ "That's a great  │         │ web_search(...)   │
    │  question, let   │         │ → results         │
    │  me look that    │         │                   │
    │  up for you..."  │         │ web_fetch(...)    │
    │                  │         │ → content         │
    └────────┬─────────┘         └────────┬──────────┘
             │                            │
             ▼                            ▼
    ┌──────────────────────────────────────┐
    │  Track C: Continuation               │
    │  LLM receives tool results,          │
    │  generates final answer → TTS        │
    └──────────────────────────────────────┘
```

### Detailed Flow

1. **LLM emits tool call + narration text** in the same response
   - Current: narration is TTS'd, then tool runs synchronously
   - Proposed: narration TTS and tool execution start simultaneously

2. **Filler speech generation** (two strategies):

   **Strategy A — LLM-generated filler (higher quality, needs model support)**
   - Before executing tools, send a follow-up prompt:
     `"You just decided to search for X.  While the search runs, say something
      brief and natural to keep the conversation going.  1-2 sentences max."`
   - TTS the filler speech immediately
   - Cost: one extra LLM call (~200ms for a short response)
   - Quality: contextual, natural, can reference what the user asked

   **Strategy B — Template filler (zero latency, lower quality)**
   - Pre-recorded or pre-synthesized filler phrases per tool category:
     - Search: "Let me look into that for you..."
     - Calculate: "Running the numbers now..."
     - Image: "Creating that image, give me just a moment..."
   - Optionally pre-synthesize these at connection start (TTS warmup)
   - Cost: zero LLM calls, ~0ms latency
   - Quality: generic but always fast

   **Recommended: Hybrid** — use template filler immediately (0ms), then
   if tools take >3s, generate contextual filler via LLM as a second sentence.

3. **Tool result injection** — when tools complete:
   - If filler TTS is still playing, let it finish naturally
   - Queue the continuation LLM call with tool results injected
   - The continuation produces the actual answer, which is TTS'd normally

4. **Early termination** — if tools finish before filler TTS ends:
   - Don't interrupt the filler (sounds unnatural)
   - Queue the real response right after filler finishes
   - Total perceived latency = filler duration (not tool duration)

### Implementation Sketch

```python
async def _speculative_tool_turn(
    tool_calls: list[ToolCall],
    narration: str,
    websocket: WebSocket,
    session: VoiceSession,
    handler: Handler,
    request: InternalChatRequest,
    app_state: Any,
    conn: Any,
) -> str:
    """Execute tools speculatively while filler speech plays."""

    # Track A: TTS the narration immediately
    filler_task = asyncio.create_task(
        _tts_filler(narration, websocket, session, conn)
    )

    # Track B: Execute tools in parallel
    tool_results = await asyncio.gather(*[
        _execute_tool(tc, app_state) for tc in tool_calls
    ], return_exceptions=True)

    # Wait for filler to finish (don't interrupt mid-sentence)
    await filler_task

    # Track C: Generate final answer with tool results
    request.messages.append(Message(
        role="tool",
        content=_format_tool_results(tool_calls, tool_results),
    ))
    return await _stream_llm_response(request, handler, websocket, session, conn)
```

### Latency Analysis

| Scenario              | Current         | With Speculation |
|-----------------------|-----------------|------------------|
| Fast tool (< 1s)     | ~1.5s silence   | ~0ms (filler covers) |
| Medium tool (1-3s)   | ~3s silence     | ~0ms (filler covers) |
| Slow tool (3-8s)     | ~5-8s silence   | ~0-3s (filler + contextual fill) |
| Multi-tool chain      | ~10s+ silence   | ~3-5s (staggered filler) |

### Risks and Mitigations

1. **Filler sounds awkward if tools fail** — if a tool errors, the continuation
   prompt should acknowledge the failure naturally: "I wasn't able to find that,
   but here's what I know..."

2. **Double-speak** — filler and real response must not overlap.  Strict queue
   discipline: real response only starts after filler TTS fully finishes.

3. **Context pollution** — filler text added to conversation history makes the
   context longer.  Mitigation: mark filler messages as `ephemeral` and strip
   them from history after the turn completes.

4. **Cost** — Strategy A adds one LLM call per tool use.  For voice-optimized
   use, recommend Strategy B (templates) by default, Strategy A opt-in.

### Configuration

```python
# In config.py
voice_speculative_tools: bool = False      # Enable speculative tool execution
voice_speculative_filler: str = "template" # "template" | "llm" | "hybrid"
voice_speculative_llm_threshold: float = 3.0  # Seconds before LLM filler kicks in (hybrid mode)
```

### Prerequisites

- Server-side VAD (done) — enables the tight audio pipeline needed
- Streaming STT (done) — reduces overall latency budget
- Tool narration callbacks (done) — already capture LLM narration text
- Sentence-buffered TTS (done) — plays filler while tools run

### Status

**Design only** — not implemented.  The infrastructure (narration callbacks,
sentence buffer, parallel tool execution) is in place.  Implementation requires:

1. Refactor `_resolve_tool_calls()` to return tool calls without executing
2. Add `_speculative_tool_turn()` orchestrator in voice_routes.py
3. Add filler template bank (`_FILLER_TEMPLATES` dict per tool category)
4. Add `ephemeral` flag to voice session messages for context cleanup
5. Wire configuration and toggle in voice settings UI
