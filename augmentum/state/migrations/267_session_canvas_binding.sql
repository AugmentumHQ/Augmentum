-- 267_session_canvas_binding.sql
-- Session Canvas binding: which artifact is currently pinned to a session's
-- side-docked canvas. One row per session (session_id is a per-user UUID).
-- User-scoped so the canvas survives refresh / restart / device switch and
-- never leaks across tenants. The pinned artifact_id is a soft reference into
-- the user-scoped `artifacts` table; rows are cleaned up lazily (a stale pin
-- resolves to the session's latest artifact, see canvas_routes.py).

CREATE TABLE IF NOT EXISTS session_canvas (
    session_id   TEXT PRIMARY KEY,
    artifact_id  TEXT NOT NULL,
    user_id      TEXT NOT NULL REFERENCES users(id),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_session_canvas_user ON session_canvas(user_id);
