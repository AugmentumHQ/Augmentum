/**
 * Authentication module — login, setup wizard, session management.
 */

import { extractErrorMessage } from './app.js';

let _currentUser = null;

/**
 * Resolve the `?next=` gate round-trip target, or null if absent/unsafe.
 *
 * Set by the front-gate login redirect (gate_routes._login_redirect) when an
 * unauthenticated hit to `<svc>.<gate_apex>` bounced the browser here. We only
 * honor an absolute https URL whose host is this page's host or a subdomain of
 * it — that's exactly the `<svc>.<gate_apex>` the user came from — so a crafted
 * `?next=https://evil.com` can never turn login into an open redirect.
 */
function _safeGateNext() {
    try {
        const raw = new URLSearchParams(location.search).get('next');
        if (!raw) return null;
        const u = new URL(raw, location.origin);
        if (u.protocol !== 'https:' && u.protocol !== location.protocol) return null;
        const here = location.hostname;
        if (u.hostname === here || u.hostname.endsWith('.' + here)) return u.href;
        // Also allow the sibling case: current host is a subdomain of next's
        // host is NOT permitted (would climb to apex); only same-or-deeper.
        return null;
    } catch { return null; }
}

/**
 * Mirror of `_currentUser` on window so classic (non-module) scripts —
 * e.g. discovery.js, loaded via `<script>` not `<script type="module">` —
 * can scope their localStorage keys by user without importing this module.
 * Read with `window.__augmentumUser?.id`; null when not logged in.
 *
 * Keeping this in sync with `_currentUser` is the responsibility of
 * every code path that mutates `_currentUser` below. The helper is
 * idempotent so over-calling is cheap.
 */
function _publishUser() {
    if (_currentUser && _currentUser.id) {
        window.__augmentumUser = { id: _currentUser.id, role: _currentUser.role || null };
    } else {
        window.__augmentumUser = null;
    }
}

/**
 * Build a per-user localStorage key. Returns `null` when the user is
 * not yet known — callers should treat null as "skip this read/write"
 * rather than fall back to a global key (which is how cross-profile
 * leaks happened in the first place).
 *
 * Format: `<baseKey>::u:<userId>`. The `::u:` separator is unlikely to
 * appear in any existing base key, which keeps the sweep in `logout()`
 * unambiguous and lets us migrate without colliding with v1 keys.
 */
export function userScopedKey(baseKey) {
    const uid = _currentUser?.id;
    if (!uid) return null;
    return `${baseKey}::u:${uid}`;
}

/**
 * Check auth status with the server.
 *
 * This call is the boot gate — `init()` in app.js awaits it before doing
 * anything else. A hung request here freezes the app on a blank screen,
 * which is the worst-possible first impression for a new user on a flaky
 * network or hitting a slow-starting Docker container. We retry with
 * short timeouts so a Tailscale/browser wake-up hiccup does not strand
 * the user on a false "server unreachable" screen.
 *
 * On timeout/network error we return `server_unreachable: true` so
 * `init()` can show a distinct overlay instead of falling into the
 * setup-wizard or login paths (which would render confusingly when the
 * server isn't actually reachable).
 *
 * @returns {{setup_required?: boolean, authenticated?: boolean, user?: object|null, server_unreachable?: boolean, db_error?: boolean, error?: string}}
 */
export async function checkStatus() {
    const timeouts = [10000, 15000, 25000];
    let lastErr = null;
    for (let i = 0; i < timeouts.length; i += 1) {
        if (i > 0) {
            await new Promise((resolve) => setTimeout(resolve, 750));
        }
        const result = await _checkStatusOnce(timeouts[i]);
        if (!result.server_unreachable) return result;
        lastErr = result.error;
    }
    return {
        server_unreachable: true,
        attempts: timeouts.length,
        error: lastErr || 'Timed out while checking server status',
    };
}


async function _checkStatusOnce(timeoutMs) {
    try {
        const resp = await fetch('/api/auth/status', {
            cache: 'no-store',
            signal: AbortSignal.timeout(timeoutMs),
        });
        if (!resp.ok) {
            // A response ARRIVED but with a non-2xx status — almost always
            // the proxy returning 502/503/504 because the backend is still
            // coming up (mid-restart). Treat it exactly like an unreachable
            // server: retry, then show the reconnecting overlay. NEVER fall
            // through to setup_required here. A non-OK status is an UNKNOWN
            // auth state, and rendering the create-admin wizard on "unknown"
            // is a fail-OPEN — it flashed a bogus "create a new admin"
            // screen on phones during restarts, and would be an account-
            // takeover vector if it ever lined up with a real empty-users
            // window server-side. Unknown auth → reconnect, never create-admin.
            return {
                server_unreachable: true,
                error: `auth status HTTP ${resp.status}`,
            };
        }
        const data = await resp.json();
        if (data.authenticated && data.user) {
            _currentUser = data.user;
            // Comms-only guests don't get the full app — bounce them to their
            // surface. The server already 403s every non-Connect API for a
            // guest (this is just clean UX so they don't stare at a broken
            // shell). Skip if we're already inside the guest surface.
            if (data.user.role === 'guest'
                && !location.pathname.startsWith('/ui/connect-guest/')) {
                location.replace('/ui/connect-guest/');
                return data;
            }
            _publishUser();
        }
        return data;
    } catch (err) {
        // Timeout or network failure — surface distinctly so the boot
        // UI can render a useful overlay rather than falling through
        // to setup-wizard. AbortError covers both AbortSignal.timeout()
        // and an explicit cancel; TypeError is what fetch throws when
        // the network request itself fails (DNS, connection refused).
        if (err?.name === 'TimeoutError' || err?.name === 'AbortError' || err?.name === 'TypeError') {
            return { server_unreachable: true, error: err.message || String(err) };
        }
        return { setup_required: false, authenticated: false, user: null };
    }
}

/**
 * Get the currently authenticated user.
 */
export function getCurrentUser() {
    return _currentUser;
}

/**
 * Return true when the current user has the admin role.
 */
export function isAdmin() {
    return !!(_currentUser && _currentUser.role === 'admin');
}

/**
 * Return true when the current user is family-content-filtered. Used by
 * surfaces that import external content (chub.ai / risurealm character
 * search) to hide SFW/NSFW toggles — the server forces SFW regardless,
 * but the UI is cleaner when the toggle isn't dangling.
 */
export function isFamilyFiltered() {
    return !!(_currentUser && _currentUser.content_level === 'family');
}

/**
 * Toggle `role-admin` / `role-user` classes on <body> so stylesheets can
 * hide admin-only controls via `.admin-only { display: none }` under
 * `body.role-user`. Call after checkStatus() resolves and any time the
 * user changes (login/logout).
 */
export function applyRoleBodyClass() {
    const b = document.body;
    if (!b) return;
    b.classList.remove('role-admin', 'role-user');
    if (!_currentUser) return;
    b.classList.add(isAdmin() ? 'role-admin' : 'role-user');
}

/**
 * Fetch /api/auth/me and refresh the cached user (role, display_name,
 * quota, etc.). Called after login or on routes that need fresh data.
 */
export async function refreshCurrentUser() {
    try {
        const resp = await fetch('/api/auth/me');
        if (!resp.ok) return null;
        const data = await resp.json();
        _currentUser = data;
        _publishUser();
        applyRoleBodyClass();
        return data;
    } catch {
        return null;
    }
}

/**
 * Show the first-run setup wizard. Returns a promise that resolves when setup is complete.
 */
export function showSetupWizard() {
    return new Promise((resolve) => {
        // First-user-wins admin claim is open until the first registration.
        // If the UI is being accessed from a non-localhost address, the
        // install is reachable on the LAN and a hostile network neighbour
        // could in principle race to register first. Warn the user — race
        // mitigation is "register now, before opening Augmentum on any
        // other device."
        const host = window.location.hostname;
        const isLoopback = host === 'localhost' || host === '127.0.0.1' || host === '::1';
        const lanWarning = isLoopback ? '' : `
            <div class="auth-lan-warning">
                <strong>⚠ This install is reachable on your local network.</strong>
                The first user to register becomes the administrator. Create
                your account now — before opening Augmentum on any other
                device — so nobody else on your network can claim admin.
            </div>
        `;

        const overlay = document.createElement('div');
        overlay.className = 'auth-overlay';
        overlay.innerHTML = `
            <div class="auth-card">
                <h2 class="auth-title">Welcome to Augmentum</h2>
                <p class="auth-subtitle">Create your admin account to get started.</p>
                ${lanWarning}
                <div class="auth-step">
                    <label class="auth-label" for="setup-username">Username</label>
                    <input class="auth-input" type="text" id="setup-username" placeholder="3-32 characters" autocomplete="username" minlength="3" maxlength="32">
                </div>
                <div class="auth-step">
                    <label class="auth-label" for="setup-password">Password</label>
                    <input class="auth-input" type="password" id="setup-password" placeholder="Minimum 8 characters" autocomplete="new-password" minlength="8">
                </div>
                <div class="auth-error" id="setup-error"></div>
                <button class="auth-btn" id="setup-submit">Create Admin Account</button>
            </div>
        `;
        document.body.appendChild(overlay);

        const btn = overlay.querySelector('#setup-submit');
        const errEl = overlay.querySelector('#setup-error');

        btn.addEventListener('click', async () => {
            const username = overlay.querySelector('#setup-username').value.trim();
            const password = overlay.querySelector('#setup-password').value;
            errEl.textContent = '';

            if (username.length < 3) { errEl.textContent = 'Username must be at least 3 characters.'; return; }
            if (password.length < 8) { errEl.textContent = 'Password must be at least 8 characters.'; return; }

            btn.disabled = true;
            btn.textContent = 'Creating...';

            try {
                const resp = await fetch('/api/auth/setup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password }),
                });
                const data = await resp.json();
                if (!resp.ok) {
                    errEl.textContent = extractErrorMessage(data, 'Setup failed.');
                    btn.disabled = false;
                    btn.textContent = 'Create Admin Account';
                    return;
                }
                _currentUser = data.user;
                _publishUser();
                overlay.remove();
                resolve();
            } catch (e) {
                errEl.textContent = 'Connection error. Is the server running?';
                btn.disabled = false;
                btn.textContent = 'Create Admin Account';
            }
        });

        // Enter key submits
        overlay.querySelectorAll('.auth-input').forEach(input => {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') btn.click();
            });
        });

        overlay.querySelector('#setup-username').focus();
    });
}

/**
 * Show the login screen. Returns a promise that resolves when login succeeds.
 */
export function showLogin() {
    return new Promise((resolve) => {
        const overlay = document.createElement('div');
        overlay.className = 'auth-overlay';
        overlay.innerHTML = `
            <div class="auth-card">
                <h2 class="auth-title">Sign In</h2>
                <p class="auth-subtitle">Enter your credentials to continue.</p>
                <div class="auth-step">
                    <label class="auth-label" for="login-username">Username</label>
                    <input class="auth-input" type="text" id="login-username" autocomplete="username">
                </div>
                <div class="auth-step">
                    <label class="auth-label" for="login-password">Password</label>
                    <input class="auth-input" type="password" id="login-password" autocomplete="current-password">
                </div>
                <div class="auth-error" id="login-error"></div>
                <button class="auth-btn" id="login-submit">Sign In</button>
            </div>
        `;
        document.body.appendChild(overlay);

        const btn = overlay.querySelector('#login-submit');
        const errEl = overlay.querySelector('#login-error');

        btn.addEventListener('click', async () => {
            const username = overlay.querySelector('#login-username').value.trim();
            const password = overlay.querySelector('#login-password').value;
            errEl.textContent = '';

            if (!username || !password) { errEl.textContent = 'Both fields are required.'; return; }

            btn.disabled = true;
            btn.textContent = 'Signing in...';

            try {
                const resp = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password }),
                });
                const data = await resp.json();
                if (!resp.ok) {
                    if (data.retry_after) {
                        const mins = Math.ceil(data.retry_after / 60);
                        errEl.textContent = `Too many attempts. Try again in ${mins} minute${mins > 1 ? 's' : ''}.`;
                    } else {
                        errEl.textContent = extractErrorMessage(data, 'Login failed.');
                    }
                    btn.disabled = false;
                    btn.textContent = 'Sign In';
                    return;
                }
                _currentUser = data.user;
                _publishUser();
                overlay.remove();
                // Front-gate round-trip: if we were sent here by a gated
                // service's forward_auth (gate_routes._login_redirect adds
                // ?next=<service-url>), forward the browser back now that the
                // session cookie is set for the gate domain. Guard against
                // open-redirect: only follow a next that stays on this host or
                // a subdomain of it (the <svc>.<gate_apex> the user came from).
                const nextTarget = _safeGateNext();
                if (nextTarget) { location.href = nextTarget; return; }
                resolve();
            } catch (e) {
                errEl.textContent = 'Connection error.';
                btn.disabled = false;
                btn.textContent = 'Sign In';
            }
        });

        overlay.querySelectorAll('.auth-input').forEach(input => {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') btn.click();
            });
        });

        overlay.querySelector('#login-username').focus();
    });
}

/**
 * Logout — POST to server, reload page.
 */
export async function logout() {
    try {
        await fetch('/api/auth/logout', { method: 'POST' });
    } catch { /* ignore */ }
    _currentUser = null;
    _publishUser();
    _clearUserScopedStorage();
    location.reload();
}

/**
 * Defense-in-depth: wipe any localStorage entries that belong to the
 * session we're leaving. Two classes get cleared:
 *
 *   1. Keys produced by `userScopedKey()` — anything containing `::u:`.
 *      These are safe to drop because the next login re-scopes them.
 *   2. Known-bad legacy keys that were written unscoped and would leak
 *      across profiles (audiobook resume, For-You cache, browse search
 *      history). Listed explicitly so an unrelated `augmentum*` key set
 *      by future code isn't nuked by accident.
 *
 * sessionStorage is NOT cleared — it dies with the tab anyway, and
 * some surfaces (XR seat lock, voice ticket) store transient state
 * there that they're equipped to rebuild after the reload.
 */
function _clearUserScopedStorage() {
    try {
        // Exact pre-migration (unscoped) keys that hold ONE user's state.
        // New code namespaces these via userScopedKey (swept by the
        // `::u:` rule below); this list clears values written before the
        // multi-tenant migration so they don't bleed into the next login.
        const legacy = [
            'augmentum-media-last-played',
            'augmentum-media-resume-dismissed',
            'augmentum-grove-last-played',
            'augmentum-grove-resume-dismissed',
            'augmentum-grove-orb-float',
            'augmentum-grove-orb-detach-hinted',
            'augmentum_for_you_cache_v1',
            'augmentum:browse_search_history',
            // Becca companion widget (now per-user; clear legacy values).
            'becca.presence.avatar',
            'becca.presence.audio_role',
            'becca.presence.size',
            'becca.dance.history',
            'becca.talk_mode',
            'becca.wake.enabled',
            'becca.wake.avatar_ids',
            'becca.followup.enabled',
            'becca.followup.window_s',
            // Connect outbox (now per-user; queued messages must not bleed).
            'augmentum:connect:outbox:v1',
            'augmentum:connect:outbox:failed:v1',
            // Coder workspace + per-user UI prefs.
            'augmentum.coder.activeWorkspaceId',
            'augmentum-coder-theme',
            'augmentum-coder-classic',
            // App chrome state tied to the user's layout.
            'augmentum-panel',
            'augmentum-inspector',
            'augmentum-inspector-width',
            'augmentum-inspector-section',
            'augmentum:doc-section-collapsed',
            // Saved/learned commands.
            'augmentum_learned_commands',
        ];
        // Per-user key PREFIXES (surface-suffixed keys we can't enumerate).
        // Genuinely device-global keys (augmentum-theme/typography/text-scale
        // /mode, companion.presence.mode, armed-device) are deliberately
        // NOT listed here so they survive a profile switch.
        const prefixes = ['becca.presence.pos.'];
        const toRemove = [];
        for (let i = 0; i < localStorage.length; i += 1) {
            const k = localStorage.key(i);
            if (!k) continue;
            if (
                k.includes('::u:')
                || legacy.includes(k)
                || prefixes.some((p) => k.startsWith(p))
            ) toRemove.push(k);
        }
        for (const k of toRemove) localStorage.removeItem(k);
    } catch { /* private mode or quota — non-critical */ }
}

/**
 * Get a short-lived WebSocket ticket for voice connections.
 *
 * Throws an Error with a `.status` property on failure so callers can
 * distinguish auth expiry (401 — show "sign in again") from transient
 * network/server failures (everything else — retry with backoff).
 *
 * @returns {string} ticket
 */
export async function getWsTicket() {
    const resp = await fetch('/api/auth/ws-ticket', { method: 'POST' });
    if (!resp.ok) {
        const err = new Error(`Failed to get WS ticket (${resp.status})`);
        err.status = resp.status;
        throw err;
    }
    const data = await resp.json();
    return data.ticket;
}
