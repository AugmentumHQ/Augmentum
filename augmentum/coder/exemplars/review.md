# Review exemplar

The shape of a review turn: read the changeset, look for the specific
class of problem the user asked about, summarize findings with file
and line citations. No edits — the deliverable is the assessment.

## User goal

"Review the changes on branch `feature/dream-summary` for any
data-isolation bugs."

## Turn

[git action=diff branch="feature/dream-summary" base="main"]
  → 6 files changed, 312 insertions, 47 deletions
  → key files: augmentum/dream/store.py, augmentum/dream/routes.py,
    augmentum/state/migrations/187_dream_summary.sql, ...

[file_read augmentum/state/migrations/187_dream_summary.sql]
  → CREATE TABLE dream_summaries (
      id TEXT PRIMARY KEY,
      dream_id TEXT NOT NULL,
      content TEXT NOT NULL
    )
  # No user_id column. Violation of the multi-tenant invariant.

[file_read augmentum/dream/store.py]
  → async def get_summary(summary_id): ...WHERE id = ?
  → async def list_summaries(): ...SELECT * FROM dream_summaries
  # Neither method accepts user_id or filters on it.

[file_read augmentum/dream/routes.py]
  → @router.get("/api/dream/summaries")
  → async def list_summaries():
  →     return await store.list_summaries()
  # Route handler doesn't extract user_id from request.scope at all.

[code_grep "user_id" augmentum/dream/store.py]
  → 0 matches.

# Confirmed at all three layers. Summarize.

Answer to user:

**Branch `feature/dream-summary` leaks data across users at all three
layers.**

1. **Schema** (migrations/187_dream_summary.sql:1-5): `dream_summaries`
   has no `user_id` column. Every Augmentum user-scoped table must
   include `user_id TEXT REFERENCES users(id)` per CLAUDE.md.

2. **Persistence** (augmentum/dream/store.py:23, 41): `get_summary`
   and `list_summaries` don't accept the `*, user_id: str = ""` kwarg
   and don't filter on user_id. User A's `list_summaries()` would
   return User B's rows.

3. **Routing** (augmentum/dream/routes.py:18): the route doesn't
   extract `user_id` from `request.scope.get("user").id` — no user
   context is threaded anywhere.

This is the #1 security invariant in CLAUDE.md and must be fixed before
merge. The reference pattern is in `augmentum/state/sessions.py`.

[finish_task]

## What this exemplar teaches

- Review doesn't edit. Read, assess, report. No "I'll fix that for you"
  in a review turn unless the user explicitly extends the scope.
- Stay focused on the problem class the user named. "Review for data
  isolation bugs" focuses the inspection — don't drift into style
  nitpicks or unrelated concerns.
- Cite `file:line` for every finding. The user needs to navigate
  directly to each issue.
- Order findings by severity. Schema-level violations come first;
  small style issues come last (or get cut).
- Quote the rule that's being violated when one exists. "CLAUDE.md
  requires user_id on every user-scoped table" lands harder than
  "this looks wrong."
- Audit the changeset's tests as claims, not decorations: for each new
  or changed test, ask "could this fail if the code were wrong?" — an
  always-green test is a finding. Report missed coverage: the concrete
  cases (wrong-user, error path, restart survival) a wrong
  implementation would still pass without.
