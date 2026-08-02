-- Notification substrate — unified attention-worthy event store.
--
-- See docs/superpowers/specs/2026-06-01-notification-substrate-design.md
-- for the full design discussion + research synthesis (Android channels,
-- freedesktop Notifications, Apple UNNotificationRequest, Matrix push,
-- HA notify, Slack/Discord).
--
-- All tables strictly per-user per CLAUDE.md's multi-tenant pattern.
-- This shipment is publish-side only: store + catalog + a config flag.
-- HTTP surface, WS fan-out, and migrating existing surfacers
-- (background_chain SSE, companion_initiative_queue, coder run
-- complete) are separate follow-on tasks.
--
-- Three tables:
--   * notification_channels        — user-mutable importance/mute buckets
--   * notifications                — the events themselves
--   * notification_subscriptions   — delivery target registrations


CREATE TABLE IF NOT EXISTS notification_channels (
    -- Composite PK: a row is "this user's customization of this
    -- channel." System defaults live in Python (catalog.py); the
    -- store falls back to the catalog when no row exists for a
    -- (user_id, channel_id) pair. user_id='' rows would be system
    -- templates but are not used in v1 — kept reserved for a future
    -- admin-set-default-for-everyone feature.
    channel_id          TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    name                TEXT NOT NULL,
    description         TEXT NOT NULL DEFAULT '',
    -- Android-style importance ladder, 0..4. Matches default_sound
    -- + visibility behavior in the UI layer (low = no sound, no
    -- toast; high = sound + toast; critical = sound + persistent
    -- banner; see catalog.py for the mapping).
    importance          INTEGER NOT NULL DEFAULT 2,
    default_sound       TEXT NOT NULL DEFAULT '',
    -- Per-user mute. NULL = not muted; otherwise mutes until this
    -- timestamp (use a far-future date for "muted indefinitely").
    muted_until         TIMESTAMP,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, channel_id)
);


CREATE TABLE IF NOT EXISTS notifications (
    notification_id     TEXT PRIMARY KEY,           -- short UUID
    user_id             TEXT NOT NULL,
    channel_id          TEXT NOT NULL,
    -- Source subsystem. Used together with dedupe_key to scope
    -- uniqueness (so unrelated subsystems can pick overlapping
    -- dedupe_key strings without clashing).
    source              TEXT NOT NULL,
    -- Stable per-source key. Empty string = "no dedup, every
    -- publish is a new row." A non-empty value means "replace any
    -- existing row with the same (user_id, source, dedupe_key)."
    -- The partial unique index below enforces this without
    -- preventing many empty-string rows.
    dedupe_key          TEXT NOT NULL DEFAULT '',
    -- Optional UI grouping key. Sibling notifications with the
    -- same thread_id collapse together (e.g. multiple call_events
    -- for one call all share thread_id = call_id).
    thread_id           TEXT NOT NULL DEFAULT '',
    -- Snapshot of importance at publish time. The channel's
    -- importance is the default; the publisher can override on a
    -- per-event basis (e.g. "this particular invite is critical").
    importance          INTEGER NOT NULL DEFAULT 2,
    title               TEXT NOT NULL,
    body                TEXT NOT NULL DEFAULT '',
    icon                TEXT NOT NULL DEFAULT '',
    -- Actions are a JSON list of {id, label, style, href?} dicts.
    -- The UI renders them as buttons; clicks POST to an action
    -- callback endpoint (deferred — not in this turn) with the
    -- action id, which the publisher handles.
    actions_json        TEXT NOT NULL DEFAULT '[]',
    -- Opaque per-source state. The action handler reads this to
    -- decide what to do (e.g. call_id + party_id for the
    -- accept/decline actions on an incoming-call notification).
    payload_json        TEXT NOT NULL DEFAULT '{}',
    -- Transient = auto-dismiss after display, never persist in the
    -- feed. Useful for ephemeral UX (sound only, no list entry).
    -- Resident (transient=0) = stay in the feed until dismissed.
    transient           INTEGER NOT NULL DEFAULT 0,
    expires_at          TIMESTAMP,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- Refreshed on dedupe-replace (preserves created_at).
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- Lifecycle timestamps. NULL until the corresponding event
    -- fires. delivered_at = handed off to at least one subscriber;
    -- read_at = user acknowledged in some surface; dismissed_at
    -- = removed from the active feed.
    delivered_at        TIMESTAMP,
    read_at             TIMESTAMP,
    dismissed_at        TIMESTAMP
);

-- Partial unique index for dedupe. Empty dedupe_key never collides
-- (each row is distinct); non-empty rows collide on (user_id,
-- source, dedupe_key). The store layer uses a delete-by-key-tuple
-- then insert pattern to make repost-with-same-key update in place
-- while preserving created_at semantics.
CREATE UNIQUE INDEX IF NOT EXISTS idx_notifications_dedupe
    ON notifications(user_id, source, dedupe_key)
    WHERE dedupe_key != '';
-- Hot path: "show me this user's unread, undismissed notifications,
-- newest first." Pinning the partial filter columns first lets
-- SQLite skip to live rows without scanning dismissed ones.
CREATE INDEX IF NOT EXISTS idx_notifications_user_unread
    ON notifications(user_id, dismissed_at, read_at, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notifications_thread
    ON notifications(user_id, thread_id, created_at DESC);


CREATE TABLE IF NOT EXISTS notification_subscriptions (
    -- A delivery target subscription. Many per user (one per WS
    -- session, one per registered push endpoint, etc.). The fan-
    -- out resolver walks these at publish time to figure out where
    -- to deliver.
    subscription_id     TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    -- Glob pattern matched against channel_id. '*' = all channels;
    -- 'connect.*' = all Connect-class events; 'coder.run.*' = just
    -- coder run lifecycle. v1 store may treat anything other than
    -- exact-match as '*' until pattern resolution is wired.
    channel_pattern     TEXT NOT NULL DEFAULT '*',
    -- Delivery target kind. Vocabulary evolves over time:
    --   'ws'         — WebSocket session (in-process; address = session id)
    --   'webpush'    — Browser Push API endpoint
    --   'cast'       — Cast receiver (TV / browser tab)
    --   'voice'      — Voice/companion interstitial
    --   'phone_apk'  — Future Android companion APK
    target_kind         TEXT NOT NULL,
    target_address      TEXT NOT NULL,
    -- Drop events below this importance for this target. Lets the
    -- user say "only show coder-run-failed on the phone, not every
    -- run-complete." 0 = receive everything.
    importance_floor    INTEGER NOT NULL DEFAULT 0,
    -- JSON-encoded quiet hours. Empty string = always-on. Shape
    -- (when implemented): {"tz": "America/New_York", "ranges":
    -- [["22:00", "08:00"]]}.
    quiet_hours_json    TEXT NOT NULL DEFAULT '',
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_notification_subscriptions_user
    ON notification_subscriptions(user_id);
