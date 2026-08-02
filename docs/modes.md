# Modes — how Augmentum picks the right brain for the job

Augmentum doesn't run one prompt style for everything. A built-in **classifier**
reads each request and routes it to the mode that fits — a plain answer, a
structured reasoning pass, a roleplay engine, an autonomous task-runner, or a full
coding agent. You can always **force** a mode yourself.

## Forcing a mode

Prefix the model name with one letter (works on any endpoint — web UI, Ollama,
OpenAI-compatible):

| Prefix | Mode |
| --- | --- |
| `p/` | Passthrough (plain) |
| `a/` | Analytical (UARF reasoning) |
| `n/` | Narrative (roleplay) |
| `g/` | Agentic (task-running) |
| `c/` | Coder |
| `d/` | Direct — raw verbatim pass-through, no Augmentum features |

`d/` is the escape hatch: it bypasses **all** Augmentum machinery (memory,
classification, injection) and pipes your request straight to the model, byte for
byte — useful when you want the bare model or are debugging.

e.g. `a/llama3` runs that turn through the analytical pipeline; `c/llama3` through
the coder loop. Without a prefix, the classifier decides based on: an explicit
override → system-prompt patterns (e.g. character cards) → content heuristics →
session history → default (passthrough).

## Passthrough

The fast default. Forwards your request straight to the model with no extra
machinery — greetings, simple questions, anything that doesn't benefit from
processing. Streaming and non-streaming both work.

## Analytical (UARF)

A structured reasoning pipeline for complex, factual, or multi-step questions. It
walks explicit phases:

1. **ASSESS** — understand the query and context.
2. **IDENTIFY** — what's known vs unknown.
3. **RELEVANT** — gather external information via tools (web search, fetch).
4. **APPLY** — reason it through, running code or verifying math as needed.
5. **VERIFY** — cross-check conclusions for consistency.
6. **CONCLUDE** — produce a well-supported answer.

For very hard queries a **DECOMPOSE** step splits the problem into sub-tasks
first. Tools are made available per phase (search/fetch in RELEVANT; code, math,
and file tools in APPLY/VERIFY). It grounds itself in your offline knowledge packs
when relevant. Tune it via the `AUGMENTUM_UARF_*` settings (backtracks, tool-call
limits, confidence threshold, proactive search/math/code).

## Narrative

Creative writing and roleplay. It parses **character cards**, tracks **world
state** and **plot threads**, enforces **style profiles**, manages **lorebook**
entries, and runs consistency checks so a story stays coherent across long
conversations. Scenes and characters can illustrate themselves via image
generation as the story unfolds. State persists in a three-layer memory (a live
state snapshot + a memory ledger + an embedded archive), so a long RP survives
restarts and branch/edit/delete edits.

Bring existing character cards (the common card formats are supported), or author
them in the app.

## Agentic

Long-running, goal-directed work with an explicit **plan as the attention
anchor**. The agent decomposes a goal into steps, executes them with tools
(search, fetch, code execution, file operations), and produces **real
artifacts** — `.docx`, `.pptx`, `.xlsx`, `.epub`, charts, and e-books. An
**autonomy dial (four levels)** controls how much it acts on its own versus
checking in with you.

## Coder

A full IDE agent in a real container — its own deep topic. See
**[Coder mode](coder.md)**.

## Companion

Not a request-mode but an autonomous layer that can dispatch *any* of the above
as a subagent. Off by default. See **[Companion](companion.md)**.

## Where this lives (for contributors)

Modes are implemented under `augmentum/modes/` (`passthrough/`, `analytical/`,
`narrative/`, `agentic/`, `coder/`); the classifier routes each request to a
handler via `handler_factory`.
