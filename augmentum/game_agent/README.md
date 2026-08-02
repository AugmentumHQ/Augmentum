# Augmentum Game Agent

Universal substrate for AI agents controlling any of Augmentum's four gaming
surfaces. The cognition is uniform across surfaces; only the surface adapter
varies.

## What this is

A small, surface-agnostic Python package that gives an AI a way to play any
game we host:

- **js13k** (HTML5 games served from an iframe, sourced from `augmentum/games/`)
- **Luanti** (voxel server with a Lua agent mod; `scriptable=True` in
  `augmentum/game_stream/profiles.py`)
- **emulator-streamed** (Dolphin today, PCSX2/RPCS3/Citra later)
- **curated** (Cradle-style screen capture + synthetic input for anything else)

The same NDJSON log, the same strict prompt, and the same fast-path / slow-path
loop drive all four. Adding a new surface is one new module under `surfaces/`.

## Architecture

```
                  ┌───────────────────────┐
                  │   Surface Adapter     │
                  │  (caps + resolver +   │
                  │   observation source) │
                  └───────────┬───────────┘
                              │
              EventPayload    │   semantic input emit
              (observations)  │   (PlanAction resolved by resolver)
                              │
                              ▼
                  ┌───────────────────────┐
                  │     Orchestrator      │
                  │  ┌─────────────────┐  │
                  │  │ Live NDJSON Log │  │
                  │  └─────────────────┘  │
                  │   /\              /\  │
                  │  fast            slow │
                  │  path            path │
                  │ (rules)         (LLM) │
                  └───────────────────────┘
```

- The **adapter** owns everything game-specific: its `caps()` declares the
  semantic-input vocabulary, the log-schema descriptor, and which observation
  modalities (log / frame / ocr / memory) it offers. Its `resolver` binds each
  semantic id to a wire-format input emitter.
- The **orchestrator** opens an append-only NDJSON log, runs the adapter's
  background observation source, feeds every observation through the rule
  engine, schedules slow-path planning turns, and drains an action queue.
- The **fast path** (rule engine) responds to events at ~100 Hz with
  deterministic Python predicates. Rules emit `PlanAction`s; the orchestrator
  resolves them and writes `input` entries.
- The **slow path** (LLM) runs on the cadence the previous plan requested
  (default 2 s). It reads a tail of the log, optionally pulls a frame, calls
  the user-supplied LLM with the strict agnostic prompt, parses the JSON
  reply, and queues its actions.

Every action, observation, plan, and rule firing is in the log. Replaying a
session is just streaming the file.

## The strict, agnostic prompt

The slow-path prompt body is fixed at authoring time and lives in
`prompt.py:SLOW_PATH_PROMPT`. It names no specific game; all game-specific
information arrives at runtime via:

- `OBJECTIVE` — user-authored, plain English
- `SURFACE_CAPS` — `{semantic_inputs, log_schema, observation_modalities}`
- `LIVE_LOG_TAIL` — recent NDJSON entries from the same file
- `STATE` — the agent's own scratchpad from the previous turn
- `FRAME` — optional PNG, only on slow-path turns when vision is budgeted

Output is strict JSON matching `PlanPayload`; `parse_plan_output()` validates
shape, bounds, and that every action's semantic is in `SURFACE_CAPS.semantic_inputs`.

## NDJSON log schema

One line per entry, discriminated by `kind`:

| `kind` | Purpose |
|---|---|
| `session` | Header; written once at start |
| `surface_caps` | Adapter capability announcement |
| `event` | Observation from a `log` / `vlm` / `ocr` / `memory` channel |
| `input` | An action was emitted to the surface (by agent or rule) |
| `plan` | Slow-path planner output |
| `rule_fired` | Fast-path rule matched and emitted actions |
| `agent_error` | Non-fatal failure (parse, adapter, etc.) |
| `session_end` | Trailer; written once at termination |

See `schema.py` for the Pydantic models. Validation is enforced at the
`LiveLog.append` boundary; an invalid payload raises and is *not* written.

## Adding a new surface

1. Add a module under `surfaces/` (e.g. `surfaces/myengine.py`).
2. Implement the `SurfaceAdapter` Protocol from `surfaces/base.py`:
   - `resolver` — a `SemanticInputResolver` with one binding per semantic id you accept.
   - `caps()` — return a `SurfaceCapsPayload` declaring your vocabulary.
   - `start(emit)` — open whatever transport you need; push `EventPayload`s through `emit`.
   - `stop()` — tear down.
   - `snapshot_frame()` — return PNG bytes or `None`.
3. Re-export from `surfaces/__init__.py` and the top-level `__init__.py`.
4. (Optional) ship a per-game rule pack as `surfaces/myengine_rules.py`.

The agent prompt and orchestrator never need changes.

## Status

- Core: complete (schema, log, semantic resolver, rule engine, prompt, agent,
  orchestrator).
- Adapters:
  - `mock` — complete; drives all tests.
  - `js13k`, `luanti`, `emulator`, `curated` — scaffolds. `caps()` and resolver
    bindings are real; `start()` raises `NotImplementedError` until the wire
    transport (WebSocket to browser shim / Lua mod, Selkies bridge,
    xdotool subprocess) is configured.

## Tests

```bash
pytest augmentum/game_agent/tests/ -v
```

The end-to-end smoke test (`test_orchestrator.py::test_end_to_end_session_writes_well_formed_log`)
runs a full session with the mock adapter and a stub LLM, then asserts the
resulting NDJSON contains every entry kind in the right order.

## HTTP + WebSocket surface

The route layer lives at `augmentum/proxy/game_agent_routes.py`:

| Verb | Path | Purpose |
|---|---|---|
| POST | `/api/game-agent/sessions` | Start a session: `{surface, objective, semantic_inputs?, log_schema?}`. Returns `{session_id, status, bridge_ws_url?}`. |
| GET  | `/api/game-agent/sessions/{id}` | Session status: `{surface, objective, status, error?}`. |
| GET  | `/api/game-agent/sessions/{id}/log` | SSE stream of the live NDJSON log; one `data:` event per line. Ends after `session_end`. |
| POST | `/api/game-agent/sessions/{id}/stop` | Request graceful stop. |
| WS   | `/api/game-agent/surfaces/{kind}/bridge/{id}` | Adapter wire for `js13k` and `luanti`. The client connects after `POST /sessions` returns `pending_bridge`; the orchestrator starts once the WS is accepted. |

Server-side surfaces (`mock` today, `curated`/`emulator` once wired) start
immediately. Bridged surfaces (`js13k`, `luanti`) require the WS handshake.

### Bridge wire protocol

Client (browser shim / Lua mod) → server:

```jsonl
{"kind": "event", "data": {...surface-specific vocabulary...}}
{"kind": "event", "data": {...}, "confidence": 0.8}
{"kind": "frame", "png_b64": "<base64 PNG>"}     // cached for snapshot_frame()
{"kind": "ping"}                                  // heartbeat; ignored
{"kind": "bye"}                                   // request graceful stop
```

Server → client:

```jsonl
{"action": "<semantic_id>", "duration_ms": <int>}
{"action": "<primary>", "duration_ms": <int>, "chord": [{"button": "<semantic_id>", "wire_kind": "...", "wire_code": <int>}]}
```

The optional ``chord`` array (max 2 parts) asks the client to hold every
part *simultaneously* with the primary for the whole duration — real-time
games (run+jump). Clients that don't understand ``chord`` ignore it and
press only the primary; adapters without a chord path degrade to
sequential presses in the action worker.

### Mounting in `augmentum/proxy/server.py`

Two lines wire the router and one line attaches the LLM:

```python
from augmentum.proxy.game_agent_routes import router as game_agent_router

# Inside create_app():
app.include_router(game_agent_router)

# On startup, after your model provider is initialized:
app.state.game_agent_llm = your_slow_path_llm_callable
# Optional: app.state.game_agent_log_dir = Path("/data/game_agent_logs")
```

`your_slow_path_llm_callable` must satisfy
`async (prompt: str, frame: bytes | None) -> str` and return strict JSON
matching `PlanPayload`. The easiest wiring is a thin adapter around
Augmentum's `ProviderRegistry.get_backend(...).chat(...)` that returns the
assistant message verbatim.

### What the route layer covers in tests

`tests/test_routes.py` exercises the end-to-end vertical against a real
FastAPI app:

- `mock` session starts immediately, runs, stops cleanly with a finalized log.
- `js13k` session returns a `pending_bridge` status + `ws://…/bridge/…` URL.
- `js13k` rejected with 400 when `semantic_inputs` / `log_schema` missing.
- 503 when no LLM is configured.
- 404 for unknown session ids.
- Full WS bridge handshake: client connects, pushes an `event`, sends `bye`,
  and the resulting NDJSON log finalizes with `session_end` and contains the
  pushed event.
