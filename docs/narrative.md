# Narrative mode — roleplay & interactive fiction

Narrative mode turns Augmentum into a coherent storyteller: it plays characters,
tracks the world, remembers what happened, and keeps a long story consistent
across restarts and edits. It's a full engine, not a system-prompt trick.

## Entering narrative mode

- **Force it** — prefix the model with `n/` (e.g. `n/llama3`). Works from the web
  UI or any connected client.
- **Automatic** — loading a **character card** or writing in a roleplay style
  makes the classifier route there on its own. (See [Modes](modes.md) for how
  routing decides.)

## Character cards

Bring your own or author them in-app. Common community card formats are supported,
so a card you already have elsewhere generally just works — import it and start.
A card defines the character's persona, voice, example dialogue, and first
message; the engine parses it and stays in character.

Multiple characters can share a scene (group scenes), each with their own card and
relationships to the others.

## What the engine tracks

While you play, the engine maintains:

- **World state** — the current situation, who's present, where things stand.
- **Plot threads** — open storylines it keeps alive and returns to.
- **Lorebook** — your world's facts, injected only when relevant so the model
  stays grounded without drowning in backstory. Add entries for places, factions,
  history, rules.
- **Style profiles** — enforce tone/POV/formatting so the prose stays consistent.
- **Consistency checks** — it cross-checks for continuity so the story doesn't
  contradict itself over a long session.

Characters and scenes can **illustrate themselves** via image generation as the
story unfolds (see [Image generation](image.md)).

## Memory that survives

Narrative uses a **three-layer memory**:

1. A live **state snapshot** (the current world),
2. A **memory ledger** (what has happened),
3. An **embedded archive** (older history, retrieved when relevant).

Because state persists server-side, a long roleplay survives a server restart and
a page refresh — and it handles **branching, editing, and deleting** messages
correctly, rewinding world state to match rather than leaving stale facts behind.

## Tips

- Seed the **lorebook** early for anything the model must never get wrong — names,
  relationships, world rules. It's cheaper (and more reliable) than repeating
  facts in every message.
- Use **group scenes** for multi-character dynamics rather than one card
  role-playing everyone.
- If a scene drifts, an explicit correction sticks — the engine folds it into
  world state, so you don't have to keep re-correcting.

## Where this lives (for contributors)

`augmentum/modes/narrative/` — the `NarrativeEngine` (per-session world state) and
`NarrativeHandler`. State tables: `narrative_memory`, `character_cards`,
`lorebook_entries`, plus entity/fact/plot tables. `sync_to_state()` flushes engine
state before persistence; see the Narrative section of `CLAUDE.md` for the
save/branch invariants.
