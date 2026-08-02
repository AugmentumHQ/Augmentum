# Companion

The Companion is Augmentum's **autonomous assistant** — not a chatbot, but an
agent that can act on its own, dispatch any of Augmentum's modes as sub-tasks,
and grow with you over time through its own memory, personality, and reflection
loops. It's the connective tissue between everything else: it can summon a coder
workspace and check on the run, kick off research, illustrate something, or just
keep track of what matters to you.

> **Beta, and off by default.** The Companion is one of the most ambitious parts
> of Augmentum and is explicitly opt-in. A fresh install never runs a line of
> companion logic until you enable it.

## Enabling it

The master switch is the setting **`companion_runtime_enabled`** (default
`false`). Turn it on in the app's companion settings (or set the env var), then
restart. Until it's on, nothing below runs.

Because it's autonomous and resource-using, treat enabling it as a deliberate
choice — the same way the game agent, Fabric, and self-editing are all opt-in.

## What it does once enabled

- **Dispatches any mode as a subagent** — it can run coder, analytical,
  narrative, or agentic work on your behalf and bring the result back in words.
- **Takes initiative** — behavior loops (initiative, sleep-wake, drift) let it
  surface things at sensible moments rather than only when prompted.
- **Runs standing tasks** — recurring jobs you hand it (a daily briefing, a
  watch on a topic) that fire on a schedule.
- **Has internal state** — affect, drive, and energy states shape *when* and
  *how* it acts, so it isn't a firehose. It reflects, consolidates memory,
  captures lessons, and accumulates skills over time.
- **Is present across surfaces** — it can observe what you're doing (with your
  setup) and be reachable by voice or chat, and it can speak through the voice
  pipeline and the avatar.

## Configuring it

The companion is heavily tunable — its energy budget, how proactive it is, which
topics it tracks, its persona/identity, and its safety limits are all settings.
Start conservative and open it up as you get comfortable; the defaults lean
toward *unobtrusive*.

If you find it over-injecting memory or being too chatty, dial back its
initiative and memory-recall settings — the intent is for its memory to be
**relevant and subtractive**, not a running life story.

## Safety posture

- **Off by default**, gated behind `companion_runtime_enabled`.
- A **safety floor** bounds what it will do autonomously.
- It respects the same per-user isolation as everything else — it acts within
  your account's scope, never across users.
- Anything with real-world or resource cost (long runs, external actions) stays
  explicit rather than silent.

## Honest status

This is genuinely early. It works and is used daily by the author, but it's the
kind of system that could run for a very long time and still not be "finished"
against where it's headed. Expect rough edges, and file issues — feedback on the
companion is especially valuable.
