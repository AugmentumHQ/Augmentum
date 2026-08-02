# Research exemplar

The shape of a research turn: identify what's unknown, fetch
authoritative sources, synthesize a recommendation the user can act on.
No workspace changes — the deliverable is the synthesis.

## User goal

"What's the current best way to do streaming SSE responses in FastAPI
as of 2026? I keep seeing conflicting advice."

## Turn

[doc_search "FastAPI streaming response SSE 2026"]
  → top hits: fastapi.tiangolo.com, sse-starlette repo, recent issues
  → skipping medium.com tutorials (stale, AI-generated patterns)

[doc_fetch "https://fastapi.tiangolo.com/advanced/custom-response/"]
  → StreamingResponse class, sync vs async generators
  → SSE-specific: media_type="text/event-stream"

[doc_search "sse-starlette EventSourceResponse vs StreamingResponse"]
  → EventSourceResponse from sse-starlette is the maintained path
  → handles ping frames + client-disconnect detection

[doc_fetch "https://github.com/sysid/sse-starlette"]
  → last release 2026-04, 1.2k stars, active
  → recommended in FastAPI issue #4861 (closed 2026-03)

# Sources align. Write the synthesis.

Answer to user:

For SSE in FastAPI as of 2026-05, the consensus is:

- **Use `sse-starlette`'s `EventSourceResponse`**, not raw
  `StreamingResponse`. It handles ping frames, client-disconnect
  detection, and SSE wire-format details for you.
- `EventSourceResponse` accepts an async generator yielding
  `{"event": ..., "data": ...}` dicts or `ServerSentEvent` objects.
- Don't set `media_type="text/event-stream"` yourself — the response
  class sets it. Double-setting causes a header collision.
- For long-lived streams behind a proxy, configure `ping=15` to keep
  the connection warm.

Raw `StreamingResponse` still works but you implement the SSE framing
yourself. The `sse-starlette` path is what FastAPI's own docs point to.

[finish_task]

## What this exemplar teaches

- Research doesn't change the workspace. No `code_edit`, no
  `file_write`, no mutating `shell_exec`. `doc_search` + `doc_fetch`
  only.
- Look for AUTHORITATIVE sources first: official docs, the maintained
  library's own repo, recent issues. Skip stale blog posts and
  AI-generated tutorials.
- Cite versions and dates. "As of 2026-05" anchors the recommendation
  in time so the user knows whether it's still valid in 6 months.
- Lead with the recommendation, then the reason. Not a dump of every
  link you fetched.
- Close with `finish_task` after the synthesis.
