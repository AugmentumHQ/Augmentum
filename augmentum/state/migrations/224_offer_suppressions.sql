-- Offer substrate — per-(user, kind, target_id) suppression record.
--
-- See docs/superpowers/specs/2026-06-02-offer-substrate-design.md.
--
-- Active offers themselves live in the existing `notifications` table
-- on channel `system.offer` (migration 221). This migration adds the
-- one extra table the offer substrate needs: a record of "the user
-- said Not now (snooze, 30d) or Never (permanent) on this kind+target."
--
-- The dispatcher consults this before publishing — a suppressed
-- offer is dropped before it ever reaches the UI, and the calling
-- tool gets back `suppressed=True` so the model can adjust prose.
--
-- Per-user isolation per CLAUDE.md.


CREATE TABLE IF NOT EXISTS offer_suppressions (
    -- Composite PK. One row per (user, kind, target_id) triple. The
    -- target_id is opaque to the table — it's the per-kind catalog
    -- key (e.g. 'gmail' for kind='mcp_server'). The catalog layer
    -- owns its meaning.
    user_id          TEXT NOT NULL,
    kind             TEXT NOT NULL,
    target_id        TEXT NOT NULL,
    -- '9999-12-31T00:00:00Z' = Never (permanent). Any earlier
    -- timestamp = snooze; the dispatcher treats `< CURRENT_TIMESTAMP`
    -- as "expired, may show again." A weekly sweep prunes expired
    -- snoozes; Never rows stay forever (that's the contract).
    suppressed_until TIMESTAMP NOT NULL,
    -- 'snooze' | 'never' — bookkeeping only, dispatcher just reads
    -- the date. Useful for the Settings UI to render the right
    -- "Snoozed for 30d" vs "Never" label.
    reason           TEXT NOT NULL DEFAULT 'snooze',
    created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, kind, target_id)
);

-- Hot path: "is (user, kind, target) currently suppressed?" The PK
-- already covers exact-match lookup; this secondary index covers
-- the sweep query that prunes expired snoozes.
CREATE INDEX IF NOT EXISTS idx_offer_suppressions_user_until
    ON offer_suppressions(user_id, suppressed_until);
