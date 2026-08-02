# Action Registry — Becca's Primitive Verbs

The `augmentum.intent` package gives Becca a small library of
**composable primitive verbs** that the LLM can pick and chain to
satisfy a user request. Instead of pattern-matching every utterance
to a hard-coded outcome, the model decides what to do based on what
makes sense in context.

It's the substrate for "open a new note for me," "remember that I
prefer dark mode," and any future productivity primitive. Add one
verb here and every voice / chat / cast / XR entry point can use it.

## The three dispatch tiers

A user utterance flows through tiers in order. The first hit wins;
unhandled phrasings fall through to the next handler in the request
pipeline (UARF, narrative, etc.).

```
transcript
   ├─ Tier 1: regex match against patterns
   │       hit → handler → ActionResult
   │
   ├─ Tier 2: embedding similarity (TBD — Phase 10)
   │       hit > threshold → handler → ActionResult
   │
   └─ Tier 3: exposed as an LLM tool via the action registry
           model calls function → ActionTool.execute → handler
```

Tier 1 is the fast path — conversation-control words like `stop`
and `bye` short-circuit the LLM entirely so they respond sub-100ms.
Tier 3 is the smart path — the model receives every tier-3 action
as a function-callable tool in its prompt, so it can compose them
freely ("save what I just said to a note" → `note.create(content=…)`).

## Architecture

```
augmentum/intent/
├── __init__.py           — public exports
├── action.py             — Action, ActionResult, ReferentCache, SessionContext
├── registry.py           — register_action() decorator + REGISTRY singleton
├── matcher.py            — Tier 1 regex matcher
├── dispatch.py           — match + handler + per-session referent cache
├── tool_adapter.py       — ActionTool wraps Action for the Tools framework
└── builtin/
    ├── control.py        — stop, repeat, slower, louder, goodbye, nevermind
    ├── navigation.py     — open_surface, back
    └── notes.py          — note.create / append / show_sticky / capture, memory.save / recall
```

## Adding a new primitive

1. **Choose a verb name.** Use `<surface>.<verb>` (e.g., `image.show_last`,
   `media.queue_next`). The id IS the LLM-facing tool name.

2. **Write the handler.** Async, takes `(text, session, args)`, returns
   `ActionResult | None`. Return `None` to opt out at runtime (e.g.,
   no referent available) — the dispatcher falls through to UARF so
   the LLM can compose an alternative.

3. **Register it.**

   ```python
   from augmentum.intent.action import ActionResult, SessionContext
   from augmentum.intent.registry import register_action

   async def _set_timer(
       text: str, session: SessionContext, args: dict[str, Any],
   ) -> ActionResult | None:
       minutes = args.get("minutes")
       if not minutes:
           return None
       # ... schedule the timer ...
       return ActionResult(
           short_circuit=True,
           surface_emit={"channel": "timer.set", "payload": {...}},
           speak=f"Timer set for {minutes} minutes.",
       )

   register_action(
       id="timer.set",
       summary="Set a countdown timer that fires a chime.",
       examples=[
           "set a timer for 10 minutes",
           "ten-minute timer please",
           "alarm in 5 minutes",
       ],
       arg_schema={
           "minutes": {"type": "integer", "description": "Duration in minutes."},
           "label": {"type": "string", "description": "Optional label."},
       },
       required=["minutes"],
       patterns=[  # Optional — auto-derived from examples if omitted.
           r"\b(?:set\s+(?:a\s+)?)?(?:(\d+)[-\s]?)?minute timer\b",
       ],
       handler=_set_timer,
   )
   ```

4. **Import it from `augmentum/intent/__init__.py`.** Each builtin
   module is explicitly imported so the `@register_action` decorators
   run at process start.

5. **Add it to `_VOICE_TOOLS`** in `augmentum/proxy/voice_routes.py`
   if it should be available to the voice mode.

6. **Add a test in `tests/test_smoke_intent.py`** with a canonical
   phrase + expected action id.

## Surface emission — Tier 1 vs Tier 3

When the user speaks a Tier-1 matched phrase directly, the voice
route emits the surface payload as a WS `intent_action` event before
returning. The frontend router (`ui/scripts/intent-action-router.js`)
turns it into a UI action (open browse, cancel TTS, etc.).

When the LLM invokes the action via Tier 3 tool-calling, the chain
layer has no WebSocket handle. The surface payload is queued on the
session's `ReferentCache.pending_surface_events` list, and the voice
route drains the queue at the next `turn_complete` boundary.

## Referent cache

`ReferentCache` is keyed by `(user_id, session_id)` and holds the
"what we just talked about" anchors:

- `last_image_id` / `last_image_title`
- `last_url`, `last_quote`, `last_file_id`, `last_entity`
- `active_note_id` / `active_note_title` — set by `note.create`
- `note_capture_mode` + `note_capture_deadline`
- `pending_surface_events` — outbox for LLM-invoked actions
- `recent_note_fingerprints` — idempotency for `note.create` retries

Handlers default to these when args are missing — e.g., `note.append`
without an explicit `note_id` appends to `active_note_id`.

## Note-capture mode

The voice route intercepts utterances while
`refs.note_capture_mode is True` and appends them to the active note
via `_capture_append` instead of running through UARF. Exits via:

- The `note.end_capture` action ("save this", "stop noting",
  "we're done")
- An idle timeout (5 min, refreshed on each append)
- WebSocket disconnect (cleared in the voice WS cleanup block)

The LLM-cleanup pass (shaping raw transcript chunks into coherent
notes) is a planned follow-up; v1 appends raw.

## Testing

- `tests/test_smoke_intent.py` — registry shape + Tier 1 patterns +
  schema generator
- `tests/test_integration_intent_notes.py` — notes primitives with
  an in-memory store fake
- `tests/test_integration_intent_tool_adapter.py` — `ActionTool.execute`
  + side-channel queue + tier-3 registration

Run inside the container:

```bash
docker exec augmentum-augmentum-1 python -m pytest \
    tests/test_smoke_intent.py \
    tests/test_integration_intent_notes.py \
    tests/test_integration_intent_tool_adapter.py -x
```

## Design constraints worth knowing

- **Patterns are case-insensitive and tolerant to trailing
  punctuation** but otherwise literal. Paraphrase tolerance comes from
  Tier 2 (embedding similarity, not yet built) or Tier 3 (LLM picks
  the right tool). If a regex isn't matching what you expect, add an
  example to the action's `examples` list and a `patterns` entry.

- **Action handlers MUST check `session.user_id`.** Writing into the
  empty-string anon user violates the multi-tenant invariant.
  `note.create` / `memory.save` already refuse with a friendly
  spoken hint; mirror the pattern in new actions.

- **Idempotency is per-action.** Tool-call retries are real — a flaky
  network can cause the LLM to retry the same tool. Use the referent
  cache's `recent_note_fingerprints` (or your own per-action map) to
  dedupe.

- **Don't return None lightly.** A handler that returns `None` lets
  the LLM (or UARF) take over — useful when a precondition is missing
  ("no active note to append to"), wrong when you actually wanted to
  speak an error. Return `ActionResult(short_circuit=True, speak=…)`
  to tell the user something is off.

## Why not just make everything an LLM tool?

We could — Tier 3 covers everything. The reason Tier 1 exists is
**latency for conversation-control words**. "Stop" should cancel TTS
in tens of milliseconds, not after a 1.5-second LLM round-trip.
Anything that's a property of the conversation surface (volume, pace,
turn boundaries) belongs in Tier 1. Anything semantic ("save this to
memory", "show me the last image") should ideally flow through Tier
3 so the model can interpret context, but Tier 1 patterns let users
discover the verbs through direct phrasing while paraphrase tolerance
is being built up.
