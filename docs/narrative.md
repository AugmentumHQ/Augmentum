# Narrative mode — roleplay & interactive fiction

Narrative mode is not a system-prompt trick. It's a full story engine with
per-session world state, branch-aware memory, LLM-callable recall, and
card-declared game mechanics — one of Augmentum's largest subsystems (~17K lines
across 30+ files in the narrative packages alone). It's SillyTavern-compatible
where compatibility matters (cards, lorebooks, presets, regex, macros) and goes
well beyond it where the integrated engine lets it.

This page covers what you can actually *do* with it on your own hardware. For the
contributor-level map, jump to [Where this lives](#where-this-lives).

---

## Getting started

- **Force it** — prefix the model with `n/` (e.g. `n/llama3`), from the web UI or
  any connected client.
- **Automatic** — loading a character card or writing in a roleplay style makes
  the classifier route there on its own (see [Modes](modes.md)).
- **Bring your cards.** Augmentum parses the common community formats directly —
  Tavern (v2/v3), Character.AI, JanitorAI, RisuRealm, Boostyle, W++ — including
  embedded PNG cards. Import a file, or search Chub.ai / RisuRealm from the card
  panel (SFW-filtered and search-off by default; you opt in).
- **Everything persists server-side** (SQLite), so a session survives a restart
  and a page refresh, and is reachable from any device on your LAN — no
  browser-localStorage lock-in.

Because the model server is *built in*, there's nothing external to wire up: no
Ollama/Kobold/Tabby endpoint to configure, no ports to reconcile. Pick a model in
the UI and write.

---

## Architecture: engine + handler + persistence

Each session gets its own engine instance; the handler drives the turn loop and
background work; persistence tiers everything to SQLite.

```
NarrativeHandler — turn loop, recall loop, background checkpointing
    ├── NarrativeEngine — per-session world state
    │   ├── CardParser            unified character-card parsing
    │   ├── CharacterTracker      emotional / physical state extraction
    │   ├── WorldTracker          scene / location state
    │   ├── PlotTracker           narrative-arc tracking
    │   ├── RelationshipTracker   directed trust / affection / tension graph
    │   ├── LoreEngine            keyword / regex / recursive lorebook injection
    │   ├── ContextBuilder        token-budget-aware prompt assembly
    │   ├── BranchTracker         DAG-based message tracking + branch detection
    │   └── LLMExtractor          LLM-powered state extraction
    │
    ├── NarrativeState  (SQLite, per session)
    │   ├── Entities        character / location / item / faction
    │   ├── Facts           domain-tagged, confidence-scored
    │   ├── PlotThreads     active / resolved / abandoned
    │   ├── LorebookEntries selective logic + timed effects
    │   └── Contradictions  minor / moderate / major severity
    │
    └── NarrativePersistence — branch-tagged tiers, ancestor chains,
        full save/load of state snapshots + memory ledgers
```

---

## Three-layer memory

Narrative memory isn't a flat history. It's three layers, each with a distinct
job:

| Layer | What it holds | Update frequency | Purpose |
| --- | --- | --- | --- |
| **State snapshot** | Current scene: location, who's present, activity, time-of-day | Every batch of messages | Grounds the model in "now" |
| **Memory ledger** | Chronological significant events, round-stamped and categorized | Every batch of messages | A timeline of what happened |
| **Embedded archive** | Older history, compacted periodically | On compaction trigger | Long-range recall without blowing the context window |

The engine auto-detects card type (narrator, single character, ensemble) and
builds card-type-specific prompts for the state+memory extraction call. It also
detects AI refusals, so a safety-model response ("I can't roleplay that") never
poisons the state snapshot as if it were a real event.

---

## Branch-aware state

The feature most people don't know exists. The branch tracker builds a **DAG** of
messages and detects when you swipe, regenerate, or edit a message from earlier
in the history. When history diverges, it:

1. Detects the exact branch point (message index where the sequence forks)
2. Forks state there — clones world state, plot threads, and the memory ledger
3. Assigns a **content-based branch ID** (hash of the divergent sequence), so
   swapping back to the same path reuses the same state
4. Caches per-branch state, so you can flip between branches without losing each
   one's accumulated consequences
5. Keeps per-branch archive pointers, so summarization tracks branches
   independently

**What this gives you:** five different versions of a scene after the same
critical choice, each remembering its *own* consequences — swap between them
freely. SillyTavern has linear swipes; Augmentum has a full DAG with per-branch
world state.

---

## Recall — the model can query its own memory mid-turn

State isn't only injected into the prompt. The model can actively look things up
while writing, through LLM-callable tools:

| Tool | What it does |
| --- | --- |
| `recall_entity(name)` | Everything the engine knows about a character / location / faction / item |
| `recall_fact(entity, domain)` | Specific facts about an entity (appearance, motives, history…) |
| `recall_plot(thread_id)` | Full status of a plot thread |
| `recall_exchange(entity, limit)` | Recent dialogue/actions involving a character |
| `lorebook.check(query)` | Search the lorebook for relevant world info |
| `lorebook.create(content, category)` | Record a newly-established world fact (session lore — never touches the source card) |

These run through an iterative loop that streams the model's response, executes
any recall tool-calls against the persistence layer, appends the results, and
re-invokes the model with updated history — capped at a max-iterations budget so
it can't spin on recall-only loops. It carries a ladder of health guards adapted
from coder mode: a duplicate-read nudge (warns when the model issues identical
reads twice) and final-iteration synthesis enforcement (the last budgeted call
must produce prose, not another tool call). Per-turn metrics — iterations, tool
calls, latency, errors — are logged.

**Why it matters:** the character doesn't "forget" their own backstory. When the
model is unsure, it *asks the engine* instead of confabulating.

---

## Lore engine — SillyTavern World Info, fully

A complete World Info implementation:

- **Keyword matching:** literal substring *and* regex (`/pattern/flags`)
- **Secondary-keyword logic:** `AND_ANY`, `NOT_ALL`, `NOT_ANY`, `AND_ALL` (mirrors
  ST's `SelectiveLogic`)
- **Timed effects:** sticky entries, cooldown timers, delay-turns-before-activation
- **Priority budget:** entries compete for token budget by priority
- **Recursive scanning:** a triggered entry can trigger others
- **Position control:** `before_char`, `after_char`, `at_depth`, `EM_TOP`,
  `EM_BOTTOM`, outlet
- **Two tool surfaces:** an underscore family (`lorebook_search/create/update/delete`)
  for rich authoring, and a dot-named family (`lorebook.check/create`) with F1/F5
  semantics — grounded mid-scene retrieval and session-lore creation that never
  touch the source card

---

## World system — card-declared game mechanics

Character cards can declare game mechanics via `extensions.world_system`, and the
engine enforces them with no per-world code:

| Module | What it declares |
| --- | --- |
| **Trackers** | Band trackers (e.g. `health: healthy → wounded → critical`), counters, flags, scalars |
| **Tables** | Lookup tables with columns and rows |
| **Dice** | Dice systems (d20 by default), player-roller toggle |
| **Sheet** | Grouped tracker-display sections with reveal triggers |
| **Locations** | Named locations with descriptions |

The model shifts trackers via `world.track.shift`. The system validates every
write (band trackers move one band per call unless force-flagged), enforces user
locks (your corrections are sticky for a set number of turns), renders a
`[World State]` block *from the store* (never from prose), and keeps bounded
per-tracker history.

**What this gives you:** declare "this world has HP, sanity, inventory, and a
corruption track," and the engine runs the rules — no code. A card becomes a
definition file, not a code path.

---

## Cardsmith — build characters by talking

An interactive, conversational character builder with its own state machine and
SQLite persistence:

- **Field accumulation:** scalar fields take latest-wins; array fields append per
  emission
- **Description-slot decomposition:** `desc_physical` / `desc_personality` /
  `desc_depth` — the model emits each slot once instead of re-emitting a growing
  description every turn
- **Output mapper:** assembles the final `CharacterCard` from accumulated fields
- **Durable sessions:** survive a server restart

Describe the character you want; Cardsmith interviews you and produces an
exportable card.

---

## The SillyTavern toolkit, built in

If you're coming from SillyTavern, the authoring primitives you rely on are here
with matching semantics:

- **Prompt presets** — the exact injection pipeline: `system_prompt` (prepended),
  `author_note` (injected N turns from the end, depth-aware to preserve KV prefix
  stability), `post_history` (before the final user message), `jailbreak` (after
  it). Plus modular toggle-composition (role / tense / POV / length / tone /
  content / anti-slop) and a JSON anti-slop phrase blacklist.
- **Regex scripts** — ordered input/output/both transforms, global or
  character-scoped; invalid patterns are skipped, never crash the pipeline.
- **Macros** — `{{char}}` / `{{user}}` / `{{obj}}` (JanitorAI compat) /
  `{{persona}}`, `{{time}}` / `{{date}}` / `{{day}}`, `{{random}}` /
  `{{random:a,b,c}}`, `{{roll:NdM}}`, `{{idle_duration}}` — expanded before text
  reaches the model.
- **Group chats** — four speaker modes (round-robin, random, manual, `llm_decide`),
  per-member summaries, muted-but-present members, and per-speaker system-prompt
  swapping.
- **Automatic trackers** — character state (10 emotion categories + physical
  posture, location changes, inventory) and a directed relationship graph
  (trust / affection / tension, each −1…+1), both confidence-dampened so a single
  ambiguous line doesn't swing the model's understanding.

---

## How it compares to SillyTavern

SillyTavern is the reference AI-roleplay frontend. On the narrative primitives,
Augmentum matches it; on state, memory, and integration, it goes further. (This
table is scoped to *narrative* capabilities — Augmentum's voice, avatar,
companion, and coder features are separate subsystems, not counted here.)

| Capability | SillyTavern | Augmentum narrative |
| --- | --- | --- |
| Character cards | Tavern, CAI, W++, Boostyle, Risu… | Same formats (Tavern, CAI, JanitorAI, Risu, Boostyle, W++) |
| Group chats | Round-robin, random, manual, natural | Round-robin, random, manual, `llm_decide` |
| Lorebook / World Info | Keyword + regex + selective logic + timed | Full ST compatibility **+ native dot-named tool schemas** |
| Prompt presets | system / jailbreak / author-note / post-history | Same pipeline **+ modular toggle config** |
| Regex scripts | Input / output / both | Same **+ character-scoped** |
| Macro expansion | `{{char}}`, `{{random}}`, `{{roll}}`… | Same **+ `{{obj}}` / `{{idle_duration}}`** |
| Swipes / branches | Linear swipe, linear undo | **Full DAG — per-branch world state, content-based IDs** |
| Memory / summarization | ChromaDB vector memory (optional, external) | **Native three-layer memory (snapshot + ledger + archive)** |
| Mid-turn recall | No mid-turn tool calling | **LLM-callable `recall_entity/fact/plot/exchange` + loop** |
| World-state tracking | Manual author's note | **Automatic character / relationship / plot / scene tracking** |
| Game mechanics | — | **World System — card-declared trackers, tables, dice, sheets** |
| Character builder | External tools only | **Cardsmith — interactive AI builder, durable sessions** |
| Model serving | External (Ollama / Kobold / OpenAI) | **Built in — 3-slot engine, all local** |
| KV session persistence | Depends on backend | **Built in — slot save/restore + replay** |
| Multi-provider routing | Multiple API connections | **Unified — local + cloud + fabric peers in one registry** |
| Server-side persistence | Client-side localStorage | **SQLite — survives restart, multi-device, multi-user** |
| Scene image generation | SD / DALL·E / ComfyUI | **Built-in SD + FLUX; scenes auto-illustrate** |

### Where SillyTavern still leads

Being honest about it:

| Area | SillyTavern advantage |
| --- | --- |
| Extension ecosystem | 100+ community extensions and plugins |
| Frontend polish | Mature UI, themes, custom CSS, full mobile PWA |
| Character hub | Deep Chub.ai integration and community sharing |
| Connection breadth | KoboldCPP, TextGen, Aphrodite, Tabby, Horde, … |
| Community | Large Discord, active dev, tutorials |

### Where Augmentum's narrative mode leads

| Area | Augmentum advantage |
| --- | --- |
| Branch-aware state | ST swipes are linear; Augmentum's DAG preserves world state per branch — flip between five versions of a scene with consequences intact. |
| Memory architecture | ST's ChromaDB is bolted-on and vector-only; Augmentum's three-layer memory is native, structured, and branch-aware. |
| Mid-turn recall | ST injects static context; Augmentum's model can *query* memory while writing, so characters don't forget their own backstory. |
| Game mechanics | ST has no equivalent; Augmentum enforces HP / sanity / corruption tracks with model-driven updates and user-lock protection. |
| Integrated serving | ST is a frontend that connects to external backends; Augmentum serves models from the same process — one install, nothing to wire. |
| KV session resume | ST can't persist KV across backend restarts; Augmentum's save/restore/replay resumes sessions in milliseconds. |
| Server-side & multi-user | ST stores everything in your browser; Augmentum's SQLite survives anything and is reachable, per-user, from any device on your LAN. |

---

## Tips

- Seed the **lorebook** early for anything the model must never get wrong — names,
  relationships, world rules. It's cheaper and more reliable than repeating facts
  in every message, and the recall tools can query it mid-scene.
- Use **group scenes** for multi-character dynamics rather than one card
  role-playing everyone.
- If a scene drifts, an explicit correction sticks — the engine folds it into
  world state (and, under the world system, user-locks it), so you don't have to
  keep re-correcting.
- After a big fork in the plot, **swipe/branch freely** — each branch keeps its
  own world state, so you can explore alternatives without contaminating the
  original.

---

## Where this lives

For contributors: the engine and handler are in `augmentum/modes/narrative/`
(`engine.py`, `handler.py`, `recall_loop.py`, plus the trackers, lore engine,
world system, prompt presets, regex/macro processors, and group manager). Session
state persists through the narrative tables (`narrative_memory`,
`character_cards`, `lorebook_entries`, entity/fact/plot tables, branch-tagged
memory ledgers). `sync_to_state()` flushes engine state before persistence.
Cardsmith and the narrative API routes (`/api/narrative/`, `/api/cardsmith/`) are
wired in `augmentum/proxy/`. See the Narrative section of
[`CLAUDE.md`](../CLAUDE.md) for the save/branch invariants.
