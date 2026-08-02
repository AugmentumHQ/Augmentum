# Red Team / White Hat Review Framework

A structured adversarial thinking process for reviewing code, designs, and features.
Use this when building anything that touches auth, user data, external APIs, or system boundaries.

## The Pattern

Every feature review has two passes:

### Pass 1: Red Hat (Attacker)

Think like someone trying to break this system. For each component, ask:

**Input boundaries:**
- What happens if I send unexpected types, sizes, or encodings?
- Can I inject SQL, HTML, shell commands, or prompt instructions?
- Can I bypass validation by sending the request differently (different endpoint, different content-type, WebSocket instead of HTTP)?

**Authentication & authorization:**
- Can I access this without authenticating?
- Can I access another user's data by guessing/incrementing IDs?
- Can I escalate my privileges by modifying request fields?
- What if I steal/forge a token — what's the blast radius?

**State & persistence:**
- Can I corrupt state by sending concurrent conflicting requests?
- Does data survive where it shouldn't (logs, browser history, error messages)?
- If the database file is exposed, what can I decrypt?

**Timing & side channels:**
- Do error responses reveal whether a resource exists?
- Does response timing differ for valid vs invalid inputs?
- Can I enumerate users, sessions, or resources by observing response patterns?

**AI-specific vectors:**
- Can user-generated content contain prompt injections that affect other users?
- Are LLM context windows isolated per user?
- Can I craft inputs that cause the AI to leak system prompts, other users' data, or internal state?
- Are tool outputs sanitized before being shown to users or fed back to the LLM?

**Supply chain & infrastructure:**
- What if a dependency is compromised?
- What if the Docker host is compromised?
- What if the reverse proxy misconfigures HTTPS?
- What secrets exist on disk and who can read them?

### Pass 2: White Hat (Defender)

For each attack vector identified, design the countermeasure:

**Severity assessment:**
- CRITICAL: Data breach, full system compromise, privilege escalation
- HIGH: Single-user data exposure, authentication bypass, persistent XSS
- MEDIUM: Information disclosure, timing attacks, enumeration
- LOW: Theoretical attacks requiring unlikely preconditions

**Fix principles:**
- Defense in depth — don't rely on a single check
- Fail closed — deny by default, allow explicitly
- Least privilege — minimum access needed for the operation
- Explicit over implicit — a missing `user_id` parameter should be a TypeError, not a wildcard query
- Constant-time operations for auth — prevent timing attacks

**Verification questions for each fix:**
- Does the fix introduce a new attack surface?
- Can the fix be bypassed by a different code path?
- Is the fix enforced at the lowest possible layer (data access, not route handler)?
- Will a developer forgetting to use it cause a loud failure (crash) or a silent vulnerability?

## When to Use This Framework

**Always use for:**
- Authentication / authorization changes
- New API endpoints that access user data
- File upload/download features
- Any code that constructs SQL, HTML, shell commands, or LLM prompts
- WebSocket or streaming endpoints
- Data export / sharing features
- Admin functionality

**Quick pass for:**
- UI-only changes (still check XSS via template literals)
- Configuration changes (check for secrets in logs/errors)
- New tool implementations (check for SSRF, command injection)

## Common Augmentum-Specific Patterns

### Data isolation check
Every data-access function must take `user_id` as a required parameter.
```python
# WRONG — silent data leak if user_id forgotten:
async def get_sessions(db) -> list:
    return await db.execute("SELECT * FROM ui_sessions")

# RIGHT — TypeError if user_id missing:
async def get_sessions(db, *, user_id: str) -> list:
    return await db.execute("SELECT * FROM ui_sessions WHERE user_id = ?", (user_id,))
```

### Token in URL check
Never put long-lived tokens in URLs (logged by proxies, browser history).
```python
# WRONG — token visible in server logs:
ws://host/ws/voice?token=LONG_LIVED_TOKEN

# RIGHT — short-lived ticket, burned on use:
POST /api/auth/ws-ticket → {"ticket": "30_SECOND_ONE_TIME_USE"}
ws://host/ws/voice?ticket=SHORT_TICKET
```

### Error message check
Auth errors must not reveal which field was wrong.
```python
# WRONG — reveals valid usernames:
if not user: return {"error": "User not found"}
if not verify: return {"error": "Wrong password"}

# RIGHT — constant-time, generic:
if not user:
    argon2.hash("dummy")  # same timing as real check
    return {"error": "Invalid username or password"}
if not argon2.verify(user.password_hash, password):
    return {"error": "Invalid username or password"}
```

### LLM context isolation check
Per-user data must never leak across user boundaries in LLM context.
```python
# WRONG — shared prefix cache mixes user context:
cache_key = f"{model}:{system_prompt_hash}"

# RIGHT — user-scoped cache:
cache_key = f"{user_id}:{model}:{system_prompt_hash}"
```

### XSS in template literals check
All user-generated content in template literals must use escapeHtml().
```javascript
// WRONG — XSS via character name:
el.innerHTML = `<div>${character.name}</div>`;

// RIGHT:
el.innerHTML = `<div>${escapeHtml(character.name)}</div>`;
```
