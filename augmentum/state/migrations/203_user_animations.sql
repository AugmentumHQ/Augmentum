-- 203_user_animations.sql
--
-- User-uploaded animations for the companion widget. Companion to the
-- code-defined ATLAS in ui/scripts/anim-atlas.js — bundled entries stay
-- in code (40 hand-curated entries with stable tagging), user uploads
-- live here and are merged into the atlas at runtime by the widget.
--
-- Phase B of [[project-dance-timeline-authoritative]]. Phase C builds
-- on top of this with `dance_loops` — curated subsets the conductor
-- picks from.
--
-- Schema mirrors the ATLAS entry shape so a row can be turned into an
-- atlas-compatible JS object with no transform beyond column-rename:
--
--   id            'user:<short_uuid>' — namespaced so an atlas
--                 collision with a future bundled id is impossible,
--                 AND so render code can tell uploads apart from
--                 bundled at a glance (e.g. to surface a "delete"
--                 control only on uploads).
--   user_id      ON DELETE CASCADE per user-deletion strands fix.
--   type         'vrma' | 'bvh' (the two formats chosen for v1;
--                 fbx/glb deferred — no retargeting infra yet).
--   source_path  on-disk path under {data_dir}/user_animations/
--                 <user_id>/<id>.<ext>. Stored explicitly (not derived)
--                 so a data-dir relocation can rewrite the column.
--   label        display name in the timeline + atlas. Defaults to the
--                 sanitized filename at upload time; user can edit.
--   roles        JSON array of role strings ('dance', 'celebrate',
--                 'greet', ...). Open vocabulary — matches the ATLAS
--                 convention. Stored as TEXT JSON to keep the column
--                 free-form.
--   emotion      JSON object {warmth, energy, openness, focus}, each
--                 in [0..1]. Defaults applied at upload time.
--   modes        JSON array of mode strings. Default
--                 ['chat-call','narrative'] matches the most common
--                 ATLAS configuration.
--   cost         REAL [0..1] — energy budget the conductor charges.
--   duration_sec REAL — actual clip length, used for slot sizing.
--   cooldown_sec REAL — minimum seconds before this id can fire again.
--   framing      'fullBody' | 'closeUp' | NULL. Camera preset hint.
--   trim_start   REAL seconds | NULL — skip leading dead frames.
--   trim_end     REAL seconds | NULL — skip trailing dead frames.
--   speed        REAL multiplier | NULL — defaults to 1.0 at render.
--   loop_flag    0 | 1 — ambient clips set to 1; default 0.
--   explicit_only 0 | 1 — if 1, only fires on direct user request.
--   notes         curation notes, optional.
--   thumbnail_path NULLABLE — set later by an autotag job (not in B v1).
--   created_at   REAL epoch seconds (matches the 197/198 family).
--   updated_at   REAL — bumped on tag edits.

CREATE TABLE IF NOT EXISTS user_animations (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type           TEXT NOT NULL,
    source_path    TEXT NOT NULL,
    label          TEXT NOT NULL,
    roles          TEXT NOT NULL DEFAULT '[]',
    emotion        TEXT NOT NULL DEFAULT '{}',
    modes          TEXT NOT NULL DEFAULT '["chat-call","narrative"]',
    cost           REAL NOT NULL DEFAULT 0.5,
    duration_sec   REAL NOT NULL DEFAULT 0,
    cooldown_sec   REAL NOT NULL DEFAULT 300,
    framing        TEXT,
    trim_start     REAL,
    trim_end       REAL,
    speed          REAL,
    loop_flag      INTEGER NOT NULL DEFAULT 0,
    explicit_only  INTEGER NOT NULL DEFAULT 0,
    notes          TEXT,
    thumbnail_path TEXT,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_animations_user
    ON user_animations(user_id, created_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (203, 'user_animations - user-uploaded VRMA/BVH for widget atlas');
