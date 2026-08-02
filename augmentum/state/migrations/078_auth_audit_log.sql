-- Audit log for auth-related admin actions (user create/update/delete,
-- role changes, activation toggles, password changes, admin password resets).
-- Read by admins via GET /api/auth/audit.
CREATE TABLE IF NOT EXISTS auth_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_user_id TEXT,
    actor_username TEXT,
    target_user_id TEXT,
    target_username TEXT,
    action TEXT NOT NULL,
    detail TEXT,
    ip_address TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_auth_audit_created ON auth_audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_auth_audit_target ON auth_audit_log(target_user_id);
