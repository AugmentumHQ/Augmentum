-- Calendar event cache — synced from connected CalDAV servers.
-- One row per event (deduped on uid + service_id + user_id).
-- The companion reads this table for briefings and calendar.today.
CREATE TABLE IF NOT EXISTS calendar_events (
    id              INTEGER PRIMARY KEY,
    user_id         TEXT    NOT NULL DEFAULT '',
    service_id      TEXT    NOT NULL,       -- managed_services id (e.g. "radicale")
    uid             TEXT    NOT NULL,       -- iCalendar UID
    summary         TEXT    NOT NULL DEFAULT '',
    start_dt        TEXT    NOT NULL,       -- ISO-8601 datetime or date
    end_dt          TEXT    NOT NULL DEFAULT '',
    location        TEXT    NOT NULL DEFAULT '',
    description     TEXT    NOT NULL DEFAULT '',
    calendar_name   TEXT    NOT NULL DEFAULT '',
    calendar_path   TEXT    NOT NULL DEFAULT '',
    last_seen_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_calendar_events_uid
    ON calendar_events (user_id, service_id, uid);

CREATE INDEX IF NOT EXISTS idx_calendar_events_range
    ON calendar_events (user_id, start_dt, end_dt);
