---
name: Multi-Tenant Auditor
description: >
  Augmentum-specific: enforce user_id scoping on every CRUD touching a
  user-scoped table. Multi-tenant data isolation is the #1 security
  invariant in this codebase; violating it leaks data between users.
  Fires whenever migrations, route handlers, or store/CRUD functions
  are touched.
kind: guidance
activation_policy: controller
activation_windows:
  - pre_plan
  - post_write
  - pre_finish
modes:
  - coder
triggers:
  - migration
  - new table
  - user_id
  - route handler
  - new endpoint
  - crud
  - data isolation
  - tenant
  - persistence
preferred_tools:
  - file_read
  - code_grep
  - task_dispatch
verification_recipe:
  - Confirm every new table has user_id TEXT REFERENCES users(id) + index.
  - Confirm every store/CRUD takes `*, user_id: str = ""` and appends
    `AND user_id = ?` when non-empty.
  - Confirm every route handler extracts user_id and passes it through.
  - Confirm caches key by `(user_id, session_id)` not bare session_id.
memory_writes:
  - category: constraint
    key: multi_tenant_scoping
success_criteria:
  - No new user-scoped table ships without user_id column + index.
  - No new route handler reads/writes user data without user_id scoping.
  - Audit pass (grep for "user_id" in changed files) confirms wiring.
tags:
  - security
  - multi-tenant
  - data-isolation
  - augmentum-specific
---

# Multi-Tenant Auditor

Augmentum is multi-tenant. Every piece of user data MUST be scoped by
`user_id`. This is the load-bearing invariant in this codebase —
CLAUDE.md flags it as the #1 security rule, and violations silently
leak data between users.

## The contract (from CLAUDE.md)

Every function that touches a **user-scoped table** must accept
`*, user_id: str = ""` as a keyword-only argument. When `user_id` is
non-empty, all queries MUST include `AND user_id = ?`.

Three-layer scoping:

```
Route handler:   user_id = request.scope.get("user").id
       ↓         passes user_id= to every data call
Store/persist:   *, user_id: str = ""
       ↓         appends AND user_id = ? to SQL
Cache keys:      (user_id, session_id) instead of session_id
```

## When you must run this audit

- Creating a new table → user_id column + index in the migration
- Adding columns to an existing user-scoped table → no user_id needed
  there, but every new query path must respect existing scoping
- Writing a new route handler that reads/writes user data
- Modifying a cache structure that holds per-user state

## Workflow

1. **Identify**: is the table you're touching user-scoped? (List in
   CLAUDE.md; or grep `migrations/` for `user_id TEXT`.)
2. **Migration check**: does the schema include `user_id TEXT
   REFERENCES users(id) ON DELETE CASCADE` + an index on user_id?
3. **Store check**: does every CRUD method accept `*, user_id: str =
   ""` and append `AND user_id = ?` when non-empty?
4. **Route check**: does the handler extract `user_id` via
   `request.scope.get("user").id` and pass it to every data call?
5. **Cache check**: if cached in `app.state.*`, are keys
   `(user_id, session_id)` not bare `session_id`?
6. **Cross-tenant test**: at minimum, write a test that verifies User
   A's data is invisible to User B for any new endpoint.

## Patterns (copy verbatim)

```python
# SELECT — append user_id filter:
async def get_item(self, item_id: str, *, user_id: str = ""):
    query = "SELECT * FROM items WHERE id = ?"
    params = [item_id]
    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)

# INSERT — include user_id column:
async def create_item(self, item_id: str, data: str, *, user_id: str = ""):
    cols = "id, data"
    phs = "?, ?"
    vals = [item_id, data]
    if user_id:
        cols += ", user_id"
        phs += ", ?"
        vals.append(user_id)
    await db.execute(f"INSERT INTO items ({cols}) VALUES ({phs})", vals)

# Route handler — extract user_id:
def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""
```

## Subagent assist

When the change touches 3+ files or you're unsure if existing scoping
covers the new path, spawn:
`task_dispatch(role="security_review", prompt="audit user_id scoping on the changes to <files>")`.

## Guardrails

- Server-level tables (providers, app_settings, knowledge_packs, etc.)
  do NOT take user_id. The full list is in CLAUDE.md under "Server-
  level tables".
- Bundled / shared resources can use `WHERE user_id = ? OR user_id IS
  NULL` to surface user's items alongside global items.
- Never silently drop the user_id filter on a "convenience" admin path
  without explicit annotation that this is intentional.

## Good outputs

- "Migration 213 adds `notebook_entries`. Wired user_id + index. Store
  CRUD takes user_id. Route extracts user_id. Cache keys updated.
  Cross-tenant test added in test_notebook_isolation.py."
- "New route `/api/foo/{id}` shares the existing `foo_store` — confirmed
  the store already enforces user_id, no new scoping needed. Added an
  isolation test to pin the behavior."
- "Touched `providers` table (server-level) — no user_id needed.
  Recorded constraint to be explicit about it."
