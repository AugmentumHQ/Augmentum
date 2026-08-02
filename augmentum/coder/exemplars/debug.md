# Debug exemplar

The shape of a debug turn: narrow the target, reproduce the failure with
a verifier, fix one thing, reverify with the same verifier. Applies to
failing tests, runtime exceptions, and wrong-output bugs equally.

## User goal

"test_session_loads_from_disk is failing — fix it"

## Turn

[file_read tests/test_sessions.py]
  → asserts session.user_id == "alice" after load_session("alice")
  → load_session calls SessionStore.get_by_user_id

[code_grep "def get_by_user_id"]
  → augmentum/state/sessions.py:142

[file_read augmentum/state/sessions.py lines 130-180]
  → query: SELECT id, created_at FROM sessions WHERE user_id = ?
  → returns row["user_id"]
  # The SELECT doesn't project user_id. Every returned row has no
  # such column; row["user_id"] is None.

[test_run tests/test_sessions.py::test_session_loads_from_disk]
  → AssertionError: assert None == "alice"
  # Reproduced. Apply the one-line fix.

[code_edit augmentum/state/sessions.py]
  - SELECT id, created_at FROM sessions WHERE user_id = ?
  + SELECT id, user_id, created_at FROM sessions WHERE user_id = ?

[test_run tests/test_sessions.py::test_session_loads_from_disk]
  → PASSED

[finish_task]

## What this exemplar teaches

- Inspect first. Don't edit on a hypothesis you haven't grounded in a read.
- Reproduce the failure with the verifier BEFORE changing anything. The
  reproduction is itself a guard against fixing the wrong thing.
- One change per fix. If the fix doesn't work, you want to know why
  without disentangling multiple edits.
- Re-run the SAME command that surfaced the bug — not a different test,
  not "let's also run the whole suite." The verifier that surfaced the
  bug is the verifier that closes it.
- If the bug arrives as a report with no failing test, write one first
  and watch it go red before touching the code. Red first, then green —
  a fix you never saw fail is a hypothesis, not a fix.
- Stop with `finish_task` once the verifier passes. The passing test is
  the evidence; no "I've fixed the bug" prose is needed.
