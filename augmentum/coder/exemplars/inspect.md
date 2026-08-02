# Inspect exemplar

The shape of an inspect turn: locate the relevant code, read enough to
answer accurately, write the answer. No edits, no shell mutations.
The deliverable is the prose answer; `finish_task` closes the turn
after it's written.

## User goal

"How does Augmentum handle multi-tenant data isolation in chat sessions?"

## Turn

[code_grep "user_id" augmentum/state/sessions.py]
  → 8 matches; key one at line 47 (sessions table has user_id column)

[file_read augmentum/state/sessions.py lines 1-100]
  → SessionStore.get_by_user_id filters WHERE user_id = ?
  → INSERT/UPDATE accept *, user_id: str = "" kwarg

[code_grep "user_id = request" augmentum/proxy/chat_routes.py]
  → line 142: user_id = request.scope.get("user").id
  → threaded into every store call

[file_read augmentum/proxy/chat_routes.py lines 130-180]
  → every chat handler extracts user_id from request.scope, passes it
    to the session store; cache key is (user_id, session_id).

# Three-layer pattern is clear. Write the answer.

Answer to user:

Augmentum scopes chat sessions per-user at three layers:

1. **Schema** — the `sessions` table (and `ui_sessions`,
   `session_messages`, etc.) has a `user_id TEXT REFERENCES users(id)`
   column.
2. **Persistence** — every CRUD method in `SessionStore` accepts a
   `*, user_id: str = ""` kwarg and appends `AND user_id = ?` to its
   WHERE clauses.
3. **Routing** — each chat route extracts `user_id` from
   `request.scope["user"].id` and passes it to the store. The per-session
   handler cache is keyed by `(user_id, session_id)` rather than
   `session_id` alone.

This is the same pattern used across all 115 user-scoped tables; the
authoritative list lives in `CLAUDE.md`.

[finish_task]

## What this exemplar teaches

- Inspect doesn't write files. No `code_edit`, no `file_write`, no
  `shell_exec` that mutates state. `shell_read` is fine.
- Read JUST enough to answer accurately. Don't burn iterations reading
  files that don't bear on the question.
- Cite file paths and line numbers in the answer. They're free
  signal-density for the user.
- Don't summarize the code if you can quote the pattern. "Every store
  accepts `*, user_id: str = ''`" is more useful than "stores are
  parameterized by user."
- Close with `finish_task` after the answer prose. The prose IS the
  deliverable; `finish_task` just signals the turn is over.
