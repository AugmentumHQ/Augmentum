-- 295_connect_thread_e2e.sql
-- Connect E2E P3: per-thread end-to-end-encryption state.
--
-- A side table (not a column on connect_threads) so the E2E feature stays
-- self-contained and doesn't churn the shared message_store schema. A row
-- here means "this user has E2E on for this thread": outgoing messages are
-- sealed client-side and incoming bodies are ciphertext to be decrypted on
-- the device. The pinned peer master is recorded so the client can refuse
-- to seal to a swapped key.
--
-- Default state = ABSENT = host-trusted (the untouched fallback). E2E is
-- strictly opt-in per thread.
--
-- USER-SCOPED (composite PK with user_id).

CREATE TABLE IF NOT EXISTS connect_thread_e2e (
    thread_id        TEXT NOT NULL,
    user_id          TEXT NOT NULL,
    enabled          INTEGER NOT NULL DEFAULT 0,
    peer_master_did  TEXT NOT NULL DEFAULT '',   -- the verified peer master we seal to
    updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (thread_id, user_id)
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (295, 'connect_thread_e2e: per-thread opt-in E2E state + pinned peer master (Connect E2E P3)');
