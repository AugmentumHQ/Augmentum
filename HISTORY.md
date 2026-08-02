# The Making of Augmentum

*How a single LLM proxy became a full companion and agent platform — in four months.*

This is the narrative companion to [`CHANGELOG.md`](CHANGELOG.md). The changelog
records *what* changed per release; this document records *how the project grew* —
the arc from a three-mode proxy on day one to the multi-surface platform it is today.

The figures below are drawn directly from the project's own git history
(1,331 commits, 2026-03-01 → 2026-06-30).

---

## Day one was not a prototype

Augmentum's very first commit — `Initial commit: Augmentum intelligence layer`,
2026-03-01 — already shipped a working system of **144 files and 122 Python
modules**, backed by **912 tests**. It had clearly matured before it was ever
placed under version control. From the first commit, it was:

- A FastAPI proxy compatible with the Ollama and OpenAI APIs
- **Three processing modes** — Passthrough, Analytical (the UARF 6-phase engine
  with backtracking verification), and Narrative (character / world / plot / lore
  tracking)
- A 13-tool registry (SearXNG search, web fetch, Python executor, math verifier, …)
- Three-tier tool calling — native → structured → text-parsing fallback
- Thinking-block awareness for reasoning models
- SQLite state persistence with migrations
- A vanilla-JS chat UI with a mode switcher and reasoning/narrative panels
- Docker Compose orchestration across seven services

It also shipped under a modest **21-line MIT license** — a detail that changes
later.

---

## The growth curve

The project roughly **doubled in size every month** for its first four months:

| Date       | Files | Python files |
|------------|-------|--------------|
| 2026-03-01 |   144 |          122 |
| 2026-04-01 |   723 |          445 |
| 2026-05-01 | 1,623 |          880 |
| 2026-06-01 | 2,872 |        1,562 |
| 2026-06-30 | 4,119 |        2,552 |

That is **1,331 commits in 122 days** — an average of ~11 per day, sustained,
essentially solo (with AI pair-programming from the first commit onward). The
single busiest day was **2026-03-26, with 102 commits**.

| Month     | Commits |
|-----------|---------|
| March     |     434 |
| April     |     347 |
| May       |     360 |
| June      |     190 |

---

## The arc, in four movements

### I. The intelligence layer (March)

The opening month deepened the original premise: a smart proxy that sits between
frontends and LLM backends. Two subsystems that would become pillars arrived here:

- **2026-03-27 — Coder mode.** Agentic coding as a first-class surface.
- **2026-03-28 — Knowledge packs.** Offline reference corpora (Wikipedia, MDWiki,
  Stack Exchange, DevDocs) for grounded retrieval.

By the end of March the codebase had already grown 5× from its first commit.

### II. The multi-tenant foundation (April)

- **2026-04-06 — Auth.** Multi-tenant authentication landed in an 82-commit day —
  the second-busiest of the entire project. Argon2id passwords, opaque session
  tokens, raw-ASGI middleware, and per-user data isolation across every table.

This is the quiet inflection that made everything after it possible: once
Augmentum could serve isolated users, it could become more than a personal proxy.

### III. The companion pivot (May)

The single most important day in the project's evolution is **2026-05-17**, when
two subsystems landed *together*:

- **`companion_runtime`** — a persistent AI companion with memory, drive, and agency
- **`game_agent`** — an agent that perceives and acts from raw frames

Before this, Augmentum was an increasingly capable *tool*. After it, the whole
project reorients around a companion *with presence*. The rest of May built out
the machinery that pivot demanded:

- **2026-05-21 — Intent + Vision.** A registry of composable primitive verbs (the
  action layer) and a vision capability. The same day brought the first
  `pre-release hygiene` commit — the earliest sign the project was aiming at a
  public launch.

### IV. Voice, self-improvement, and training (June)

The final month turned the companion outward and inward at once:

- **2026-06-06 — Calling.** Real-time voice and telephony.
- **2026-06-22 — Self-edit.** The self-improvement subsystem, landing in a massive
  documentation-and-infrastructure checkpoint (the model cards alone added
  ~86,000 lines).
- **2026-06-28 — Training.** The model-training pipeline — capability synthesis,
  capture, and the beginnings of a self-improving loop (+24k lines of integration
  in a single day).

---

## Milestones at a glance

| Date       | Milestone                                             |
|------------|-------------------------------------------------------|
| 2026-03-01 | Initial commit — 3-mode proxy, UARF, Narrative, tools |
| 2026-03-27 | Coder mode                                            |
| 2026-03-28 | Knowledge packs                                       |
| 2026-04-06 | Multi-tenant auth (82-commit day)                     |
| 2026-05-17 | **Companion runtime + game agent — the pivot**        |
| 2026-05-21 | Intent (action registry) + Vision                     |
| 2026-06-06 | Calling (voice / telephony)                           |
| 2026-06-12 | License: AGPL-3.0-or-later (launch positioning) |
| 2026-06-22 | Self-edit (self-improvement)                          |
| 2026-06-28 | Training pipeline                                      |

---

## What the shape of this history says

Augmentum did not stumble toward a release. The `pre-release hygiene` commit in
May and the deliberate license change in June show a project that spent its second
half consciously preparing to be seen. The velocity was real and sustained; the
direction was not random. It began as an *intelligence layer* — a better proxy —
and became a platform for a persistent, capable AI companion. Everything after
2026-05-17 is in service of that.

*This document is a curated summary. The full commit-by-commit history is
preserved privately; this public repository begins from a clean initial release
commit.*
