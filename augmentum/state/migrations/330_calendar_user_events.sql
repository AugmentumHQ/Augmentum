-- Native Augmentum calendar events — user-created, first-class events that
-- live in Augmentum itself (not just a CalDAV cache). The calendar surface
-- reads these alongside the CalDAV cache (calendar_events, migration 314) and
-- companion standing-task occurrences to render one unified grid.
--
-- Optional outbound sync: when the user flips "also add to my devices" and a
-- CalDAV service is connected, the event is mirrored to that server and the
-- resulting UID/href/service are recorded here so the two stay linked.
CREATE TABLE IF NOT EXISTS calendar_user_events (
    id              INTEGER PRIMARY KEY,
    user_id         TEXT    NOT NULL DEFAULT '',
    title           TEXT    NOT NULL DEFAULT '',
    start_dt        TEXT    NOT NULL,       -- ISO-8601 UTC datetime (or date when all_day)
    end_dt          TEXT    NOT NULL DEFAULT '',
    all_day         INTEGER NOT NULL DEFAULT 0,
    location        TEXT    NOT NULL DEFAULT '',
    description     TEXT    NOT NULL DEFAULT '',
    color           TEXT    NOT NULL DEFAULT '',   -- optional swatch key (e.g. 'blue')
    rrule           TEXT    NOT NULL DEFAULT '',   -- optional iCal RRULE for recurrence
    -- Outbound CalDAV linkage (all empty until the user opts into device sync).
    caldav_service_id TEXT  NOT NULL DEFAULT '',
    caldav_uid        TEXT  NOT NULL DEFAULT '',
    caldav_href       TEXT  NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_calendar_user_events_range
    ON calendar_user_events (user_id, start_dt, end_dt);
