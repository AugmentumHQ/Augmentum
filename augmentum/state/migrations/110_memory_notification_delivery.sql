-- 108_memory_notification_delivery.sql
-- Separate chat delivery from review status.

ALTER TABLE memory_notifications ADD COLUMN delivered_at TEXT;

CREATE INDEX IF NOT EXISTS idx_memory_notifications_delivery
    ON memory_notifications(user_id, delivered_at, created_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (110, 'Memory notification delivery tracking');
