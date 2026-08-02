/* ==========================================================================
   Chat Module — SessionStore
   Singleton class managing all chat session CRUD, persistence, and sync.
   Replaces the global `sessions` object and related functions from chat.js.
   ========================================================================== */

import { escapeHtml, showToast } from '../app.js';
import {
  STORAGE_SESSIONS,
  STORAGE_ACTIVE,
  STORAGE_ACTIVE_BY_MODE,
  STORAGE_SESSIONS_MIGRATED,
} from './constants.js';
import { migrateSessionToV2, sessionHasMessages, getDeepestLeaf } from './tree.js';
// Local copy of the prune window so it's easy to find + tweak.
const _PRUNE_STALE_MS = 6 * 60 * 60 * 1000; // 6 hours

class SessionStore {
  constructor() {
    this._sessions = {};
    this._activeId = null;
    // Per-mode "last viewed" memory. Populated as the user navigates; on
    // a mode-switch we look here first so the user returns to the chat
    // they were reading instead of the newest-created session in that
    // mode. Persisted to localStorage under STORAGE_ACTIVE_BY_MODE.
    this._lastActiveByMode = {};
    try {
      const raw = localStorage.getItem(STORAGE_ACTIVE_BY_MODE);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') {
          this._lastActiveByMode = parsed;
        }
      }
    } catch { /* corrupt localStorage — ignore */ }
    this._saveTimer = null;
    this._syncFailCount = 0;
    this._syncFailReported = false;
    this._syncRejectReported = false;
    this._syncDeferred = false;
    this._dirtyIds = new Set();
    this._retryTimer = null;
    this._connectivityWired = false;
    // Wire visibilitychange + online + pageshow so a mobile tab
    // wake / WiFi reconnect immediately retries deferred syncs.
    // Safe to call before document is fully ready -- the listener
    // attach is idempotent.
    this._attachConnectivityListeners();
    // Debounced cross-device reconcile, driven by the `sessions.changed` SSE
    // topic. Server pushes a change signal; we refetch the metadata list and
    // merge non-destructively. See _reconcileSessions below.
    this._reconcileTimer = null;
    this._attachSessionSyncListener();
  }

  // -------------------------------------------------------------------------
  // Load — fetch from server with localStorage fallback + one-time migration
  // -------------------------------------------------------------------------

  async load() {
    try {
      const resp = await fetch('/api/chats/?meta=1');
      if (resp.ok) {
        const data = await resp.json();
        this._sessions = data.sessions || {};

        // One-time migration: push localStorage sessions to server
        if (!localStorage.getItem(STORAGE_SESSIONS_MIGRATED)) {
          await this._migrateLocalStorage();
          localStorage.setItem(STORAGE_SESSIONS_MIGRATED, '1');
        }

        // Migrate any v1 format sessions from server
        const migratedIds = [];
        for (const id of Object.keys(this._sessions)) {
          if (this._sessions[id].version !== 2) {
            this._sessions[id] = migrateSessionToV2(this._sessions[id]);
            migratedIds.push(id);
          }
        }
        if (migratedIds.length > 0) {
          for (const id of migratedIds) this.markDirty(id);
          this.save();
        }

        // Restore active session ID
        this._activeId = localStorage.getItem(STORAGE_ACTIVE) || null;
        // NB: pruning is deferred to post-surface-restore (called from
        // app.js _bootSurfaces) so that surfaces about to be restored from
        // saved workspace JSON don't lose their referenced sessions to a
        // load-time sweep that only protects _activeId. (audit §7.5)
        return;
      }
    } catch { /* server unavailable — fall back to localStorage */ }

    // Fallback: load from localStorage
    try {
      const raw = localStorage.getItem(STORAGE_SESSIONS);
      this._sessions = raw ? JSON.parse(raw) : {};
    } catch {
      this._sessions = {};
      showToast('Could not load chat history', 'error');
    }

    const migratedIds = [];
    for (const id of Object.keys(this._sessions)) {
      if (this._sessions[id].version !== 2) {
        this._sessions[id] = migrateSessionToV2(this._sessions[id]);
        migratedIds.push(id);
      }
    }
    if (migratedIds.length > 0) {
      for (const id of migratedIds) this.markDirty(id);
      this.save();
    }

    this._activeId = localStorage.getItem(STORAGE_ACTIVE) || null;
    // pruning deferred to post-surface-restore (see note above)
  }

  // -------------------------------------------------------------------------
  // Cross-device reconcile — driven by the `sessions.changed` SSE topic
  // (chat_routes.py emits it on create/delete/sync). This is the sessions
  // equivalent of the model-cache "server pushes, client reconciles" pattern,
  // for the installed PWA which has no manual refresh. Debounced hard because
  // the active device's own autosave syncs emit constantly — we coalesce to
  // one lightweight metadata refetch per window.
  // -------------------------------------------------------------------------

  _attachSessionSyncListener() {
    window.addEventListener('system-event:sessions.changed', () => {
      clearTimeout(this._reconcileTimer);
      this._reconcileTimer = setTimeout(() => this._reconcileSessions(), 4000);
    });
  }

  async _reconcileSessions() {
    let serverSessions;
    try {
      const resp = await fetch('/api/chats/?meta=1');
      if (!resp.ok) return;
      const data = await resp.json();
      serverSessions = data.sessions || {};
    } catch { return; /* offline — leave local state untouched */ }

    let changed = false;
    const serverIds = new Set(Object.keys(serverSessions));

    // Add brand-new sessions; refresh metadata on safe-to-touch ones.
    for (const [id, srv] of Object.entries(serverSessions)) {
      const local = this._sessions[id];
      if (!local) {
        // Created on another device. The meta stub is already v2-shaped and
        // carries no tree, so it lazy-loads on first open like any stub.
        this._sessions[id] = srv;
        changed = true;
        continue;
      }
      // Never clobber unsynced local edits or the currently-open session.
      if (this._dirtyIds.has(id) || id === this._activeId) continue;
      // srv has no `tree`/`rootId`, so spreading it over local refreshes
      // metadata (title/mode/updatedAt) while preserving any loaded tree.
      if (local.title !== srv.title || local.mode !== srv.mode
          || local.updatedAt !== srv.updatedAt) {
        this._sessions[id] = { ...local, ...srv };
        changed = true;
      }
    }

    // Drop sessions deleted elsewhere — but ONLY pure server-originated stubs.
    // Anything with a local `tree` (a loaded chat, or a freshly-created one
    // that create() leaves un-dirty until its first message) is preserved so
    // we never delete local work. Such a stale entry clears on next reload.
    for (const id of Object.keys(this._sessions)) {
      if (serverIds.has(id) || this._dirtyIds.has(id) || id === this._activeId) continue;
      if ('tree' in this._sessions[id]) continue;
      delete this._sessions[id];
      changed = true;
    }

    if (changed) {
      // Re-render via the existing event seam — sessions.js stays decoupled
      // from the render layer (chat/index.js listens for this).
      document.dispatchEvent(new CustomEvent('augmentum:sessions-reconciled'));
    }
  }

  /** Push any localStorage-only sessions to the server (runs once). */
  async _migrateLocalStorage() {
    const localRaw = localStorage.getItem(STORAGE_SESSIONS);
    if (!localRaw) return;

    let local;
    try { local = JSON.parse(localRaw); } catch { return; }

    const localIds = Object.keys(local);
    if (localIds.length === 0) return;

    const serverIds = new Set(Object.keys(this._sessions));
    const toMigrate = {};
    for (const id of localIds) {
      if (!serverIds.has(id)) {
        toMigrate[id] = local[id].version === 2
          ? local[id]
          : migrateSessionToV2(local[id]);
      }
    }

    if (Object.keys(toMigrate).length === 0) return;

    await fetch('/api/chats/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessions: toMigrate }),
    });

    // Re-fetch merged list
    try {
      const resp = await fetch('/api/chats/');
      if (resp.ok) {
        const data = await resp.json();
        this._sessions = data.sessions || {};
      }
    } catch { /* keep what we have */ }
  }

  /**
   * Remove orphan sessions that have no messages (empty tree) and are older
   * than the prune window. This catches:
   *   - Fresh "New Chat" sessions never typed into (original reason).
   *   - Sessions created by the old drag-open path where a character or
   *     mode title got stamped but the user never engaged (we broadened
   *     the title filter to match any title now that sessions are also
   *     spawned from surface boot, not only the "+" button).
   *
   * Sessions with an empty tree can always be re-created from the UI — no
   * user content is lost by deletion. Anything with at least one tree node
   * (including a greeting) is left alone so the user can return to it.
   *
   * `protectedIds` is a Set of session ids referenced by mounted surfaces;
   * those are skipped regardless of staleness. Single-tab boot used to skip
   * only `_activeId`, which was correct for one tab but pruned non-focused
   * tab sessions out from under multi-surface workspaces (audit §7.5).
   * Public so the boot sequence can call it after surfaces have claimed
   * their session ids.
   */
  pruneStaleEmpty(protectedIds = null) {
    const cutoff = Date.now() - _PRUNE_STALE_MS;
    const stale = [];
    const protect = protectedIds instanceof Set ? protectedIds : new Set();
    for (const [id, s] of Object.entries(this._sessions)) {
      if (id === this._activeId) continue;
      if (protect.has(id)) continue;
      if (sessionHasMessages(s)) continue;
      const ts = s.updatedAt || s.createdAt || s.created || 0;
      if (ts < cutoff) stale.push(id);
    }
    for (const id of stale) {
      delete this._sessions[id];
      // Server-side delete is best-effort (this is housekeeping for
      // sessions the user already abandoned), but log non-2xx so a
      // misconfigured permissions / endpoint regression is debuggable.
      // 404 on already-deleted sessions is fine — silence it.
      fetch(`/api/chats/${encodeURIComponent(id)}`, { method: 'DELETE' })
        .then((resp) => {
          if (!resp.ok && resp.status !== 404) {
            console.warn('[chat-sessions] stale delete failed', { id, status: resp.status });
          }
        })
        .catch((err) => {
          console.warn('[chat-sessions] stale delete network error', { id, err: err?.message });
        });
    }
    if (stale.length > 0) this.save();
  }

  // -------------------------------------------------------------------------
  // Save — localStorage stubs + debounced server sync
  // -------------------------------------------------------------------------

  save(touchedSessionId = null) {
    // Write only metadata stubs to localStorage (titles, timestamps, mode).
    // Full tree data is persisted server-side via /api/chats/sync.

    // Bump the touched session's updatedAt so forMode() ranks it as
    // "most recently used". Multi-surface streaming can mutate a
    // non-active session — pass `touchedSessionId` to bump the right
    // one (audit §7.4).
    //
    // The previous fallback (when no id passed) marked every loaded
    // session with tree data as dirty, which produced runaway sync
    // payloads >50MB once the user had loaded enough chats (typical
    // after a few hours of dogfooding — an early test session got
    // stuck at 36MB across 607 sessions). Bare save() now writes localStorage
    // stubs without marking anything dirty; call sites that need a
    // sync must pass the id explicitly, or use markDirty() for bulk
    // (migration, import).
    if (touchedSessionId && this._sessions[touchedSessionId]) {
      this._sessions[touchedSessionId].updatedAt = Date.now();
      this._dirtyIds.add(touchedSessionId);
    }

    try {
      const stubs = {};
      for (const [id, s] of Object.entries(this._sessions)) {
        stubs[id] = {
          id: s.id || id,
          title: s.title || '',
          mode: s.mode || 'passthrough',
          model: s.model || '',
          createdAt: s.createdAt || s.created || 0,
          updatedAt: s.updatedAt || s.updated || Date.now(),
          version: s.version || 2,
        };
      }
      localStorage.setItem(STORAGE_SESSIONS, JSON.stringify(stubs));
    } catch (e) {
      console.warn('SessionStore.save: localStorage write failed, relying on server sync', e);
    }

    // Debounced server sync (500ms)
    clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => this._syncToServer(), 500);
  }

  /**
   * Mark a session for inclusion in the next sync without bumping its
   * timestamp. Use for bulk operations (migration, import) where each
   * affected session needs to be queued for upload but the caller
   * doesn't want N separate save() calls (each of which would trigger
   * its own debounced sync).
   */
  markDirty(id) {
    if (id && this._sessions[id]?.tree) this._dirtyIds.add(id);
  }

  // -------------------------------------------------------------------------
  // Server sync
  // -------------------------------------------------------------------------

  async _syncToServer() {
    // Don't fight the browser when it has good reason to drop our
    // requests: tab is backgrounded (mobile WebKit/Chrome suspend
    // pending fetches as soon as the tab is hidden), or the device
    // reports itself offline. We retry as soon as either condition
    // clears -- see ``_attachConnectivityListeners()`` below.
    if (typeof document !== 'undefined' && document.hidden) {
      this._syncDeferred = true;
      return;
    }
    if (typeof navigator !== 'undefined' && navigator.onLine === false) {
      this._syncDeferred = true;
      return;
    }
    this._syncDeferred = false;

    // Only sync sessions that changed and have tree data. Sending every
    // loaded chat after each message makes mobile/Tailscale page lifecycle
    // fetches much easier to drop, and a dropped sync should never make the
    // whole app look unreachable.
    const dirtyIds = Array.from(this._dirtyIds);
    if (dirtyIds.length === 0) return;

    const allToSync = {};
    for (const id of dirtyIds) {
      const s = this._sessions[id];
      if (s?.tree) allToSync[id] = s;
    }
    if (Object.keys(allToSync).length === 0) return;

    const stripped = this._stripImagesForStorage(allToSync);

    // 30MB per chunk. Server caps body at 50MB
    // (``settings.max_request_body_bytes``); headroom covers
    // JSON-encoding overhead and the occasional outlier session
    // pushing a chunk slightly past our estimate. Without chunking,
    // a user with enough loaded sessions hits the 50MB ceiling and
    // every subsequent sync 413s with the same payload, never
    // clearing.
    const CHUNK_MAX_BYTES = 30 * 1024 * 1024;
    const chunks = this._partitionForSync(stripped, CHUNK_MAX_BYTES);

    // Walk chunks until one transient failure. _handleSyncFailure
    // already schedules a retry that will re-attempt every still-
    // dirty session, so bailing here doesn't lose work; it just
    // avoids hammering a server that's clearly unhealthy this round.
    let transientFailure = false;
    for (const { body, ids } of chunks) {
      const ok = await this._postSyncChunk(body, ids);
      if (!ok) {
        transientFailure = true;
        break;
      }
    }

    if (!transientFailure) {
      this._syncFailCount = 0;
      if (this._syncFailReported) {
        // Recovery \u2014 only toast on transition (failed \u2192 ok), not
        // every successful sync, to avoid spam.
        showToast('Connection restored, chat saved.', 'info');
      }
      this._syncFailReported = false;
    }
  }

  /**
   * Greedy bin-pack ``stripped`` into chunks whose stringified body
   * stays under ``maxBytes``. Sessions keep insertion order. A single
   * session larger than the cap is sent on its own \u2014 the server may
   * still reject it with 413, but the rest of the batch goes through
   * and that one session's dirty bit gets cleared via the 4xx branch
   * in ``_postSyncChunk``.
   */
  _partitionForSync(stripped, maxBytes) {
    const chunks = [];
    let current = {};
    let currentIds = [];
    // Size of the JSON wrapper `{"sessions":{}}` so the running total
    // matches what JSON.stringify will produce.
    let currentSize = 16;

    const flush = () => {
      if (currentIds.length === 0) return;
      chunks.push({
        body: JSON.stringify({ sessions: current }),
        ids: currentIds.slice(),
      });
      current = {};
      currentIds = [];
      currentSize = 16;
    };

    for (const [id, s] of Object.entries(stripped)) {
      const entry = JSON.stringify(s);
      // entry + ``"id":`` key + comma between siblings
      const addSize = entry.length + id.length + 6;
      if (currentIds.length > 0 && currentSize + addSize > maxBytes) {
        flush();
      }
      current[id] = s;
      currentIds.push(id);
      currentSize += addSize;
    }
    flush();
    return chunks;
  }

  /**
   * POST one chunk. Returns true if we should keep processing later
   * chunks this round (success OR a 4xx that we handled), false on a
   * transient error (5xx / network) so the caller can stop early.
   *
   * 4xx \u2014 typically 413 when a single session won't fit, or 400 on
   * malformed payload \u2014 is treated as PERMANENT: we clear the chunk's
   * dirty bits and surface a one-shot toast. Retrying the same body
   * forever was the previous behaviour (pre-fix) that caused the
   * 45-syncs-in-30-min storm.
   */
  async _postSyncChunk(body, ids) {
    try {
      const resp = await fetch('/api/chats/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      });
      if (resp.ok) {
        for (const id of ids) this._dirtyIds.delete(id);
        // Stale guard: the server refuses any session whose stored blob
        // carries a newer client edit-stamp — another tab/device wrote it
        // since we last loaded it. Blind acceptance would have replaced
        // that device's turns with our stale tree. Recover by union-
        // merging the server copy into ours and re-syncing, so NEITHER
        // side's turns are lost.
        try {
          const data = await resp.json();
          const stale = Array.isArray(data?.stale) ? data.stale : [];
          if (stale.length) this._recoverStaleSessions(stale);
        } catch { /* response body is advisory */ }
        return true;
      }
      if (resp.status >= 400 && resp.status < 500) {
        for (const id of ids) this._dirtyIds.delete(id);
        console.warn(
          '[chat-sync] server rejected chunk',
          { status: resp.status, sessions: ids.length, bytes: body.length },
        );
        if (!this._syncRejectReported) {
          this._syncRejectReported = true;
          showToast(
            'Some chat changes were too large to sync; reload to retry.',
            'warning',
          );
        }
        return true;
      }
      this._handleSyncFailure(`server returned ${resp.status}`);
      return false;
    } catch (err) {
      this._handleSyncFailure(err?.message || String(err));
      return false;
    }
  }

  /**
   * Record a sync failure, log it, and surface a single toast after
   * a meaningful streak. Mobile browsers drop in-flight fetches when
   * the tab is backgrounded -- so the very first failure after a
   * screen-off is normal, not alarming. We retry with exponential
   * backoff and only toast once we've actually given the user
   * reason to worry (10 consecutive failures spread over ~2 minutes).
   * On recovery, ``_syncToServer`` toasts "Connection restored".
   */
  _handleSyncFailure(reason) {
    this._syncFailCount++;
    console.debug('[chat-sync] failed', { count: this._syncFailCount, reason });
    // Schedule a retry with exponential backoff (1s, 2s, 4s, 8s,
    // 16s capped). Cancelled if a later save() call wins -- the
    // debounce timer in save() bumps over this without coordination.
    const backoffMs = Math.min(16000, 1000 * (2 ** (this._syncFailCount - 1)));
    clearTimeout(this._retryTimer);
    this._retryTimer = setTimeout(() => this._syncToServer(), backoffMs);
    // Toast at 10 strikes -- with 60s per attempt + backoff, that's
    // ~2 minutes of failure. Long-running chats legitimately hold
    // sync queued past 15-30s; we only care about real outages.
    if (!this._syncFailReported && this._syncFailCount >= 10) {
      this._syncFailReported = true;
      showToast(
        "Chat sync paused -- we'll retry automatically. (Saved locally.)",
        'warning',
      );
    }
  }

  // Listen for the tab waking up + the OS reporting a network
  // recovery, and immediately retry pending syncs. Without this the
  // user has to type something for the next debounced save to even
  // attempt the network -- which means a sync that failed at
  // background-suspend stays unsynced indefinitely.
  _attachConnectivityListeners() {
    if (this._connectivityWired) return;
    this._connectivityWired = true;
    const retry = () => {
      if (this._syncDeferred || this._syncFailCount > 0) {
        clearTimeout(this._retryTimer);
        this._retryTimer = setTimeout(() => this._syncToServer(), 200);
      }
    };
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', () => {
        if (!document.hidden) retry();
      });
    }
    if (typeof window !== 'undefined') {
      window.addEventListener('online', retry);
      // ``pageshow`` fires when bfcache restores the tab on
      // navigation back -- another mobile-suspend recovery hook.
      window.addEventListener('pageshow', retry);
    }
  }

  /**
   * Create a shallow clone of sessions with base64 image data replaced by
   * placeholders. Preserve server URLs (/api/chat-images/...), only strip
   * inline base64.
   */
  _stripImagesForStorage(sessionsObj) {
    const out = {};
    for (const [id, session] of Object.entries(sessionsObj)) {
      const s = { ...session };
      if (s.tree) {
        const tree = {};
        for (const [nodeId, node] of Object.entries(s.tree)) {
          if (node.images && node.images.length > 0) {
            tree[nodeId] = {
              ...node,
              images: node.images.map(img =>
                (img && img.startsWith('/api/')) ? img : '[image]'
              ),
            };
          } else {
            tree[nodeId] = node;
          }
        }
        s.tree = tree;
      }
      // Strip transient data
      delete s._pendingImages;
      out[id] = s;
    }
    return out;
  }

  // -------------------------------------------------------------------------
  // Lazy-load full session data
  // -------------------------------------------------------------------------

  async _fetchFull(id) {
    try {
      const resp = await fetch(`/api/chats/${encodeURIComponent(id)}`);
      if (resp.ok) {
        const full = await resp.json();
        this._sessions[id] = full;
      }
    } catch { /* server unavailable — session stays as stub */ }
  }

  async ensureLoaded(id) {
    const session = this._sessions[id];
    if (!session) return;
    // If it's a metadata stub (no tree), fetch full data
    if (!session.tree) {
      await this._fetchFull(id);
    }
  }

  // -------------------------------------------------------------------------
  // CRUD
  // -------------------------------------------------------------------------

  get(id) {
    return this._sessions[id] || null;
  }

  all() {
    return this._sessions;
  }

  forMode(mode) {
    // Sort by "last touched" so the fallback that fires when the
    // per-mode memory is empty matches a user's intuition for "resume
    // where I left off". updatedAt is bumped on view (setActiveId) and
    // via save() whenever the session tree mutates; createdAt is the
    // tiebreaker for sessions that have never been written since load.
    return Object.values(this._sessions)
      .filter(s => (s.mode || 'passthrough') === mode)
      .sort((a, b) => {
        const aTs = a.updatedAt || a.createdAt || 0;
        const bTs = b.updatedAt || b.createdAt || 0;
        return bTs - aTs;
      });
  }

  create(mode) {
    const modeStr = (typeof mode === 'string' && mode) ? mode : 'passthrough';
    const id = 's_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    this._sessions[id] = {
      id,
      title: 'New Chat',
      mode: modeStr,
      version: 2,
      tree: {},
      rootId: null,
      activeLeafId: null,
      createdAt: Date.now(),
    };
    this.save();
    return id;
  }

  delete(id) {
    const session = this._sessions[id];
    const deletedMode = session?.mode || 'passthrough';
    delete this._sessions[id];
    this.save();

    // Also delete from server directly. The user already saw the chat
    // disappear from the UI, so don't toast on success. But if the
    // server-side delete fails, the next page load will resurrect the
    // chat from /api/chats/list — confusing and feels broken. Surface
    // the failure so the user knows to retry. 404 = already gone on
    // server (e.g. multi-tab race) = silent.
    fetch(`/api/chats/${encodeURIComponent(id)}`, { method: 'DELETE' })
      .then((resp) => {
        if (!resp.ok && resp.status !== 404) {
          console.warn('[chat-sessions] delete failed', { id, status: resp.status });
          showToast("Couldn't fully delete chat — it may reappear after reload", 'warning');
        }
      })
      .catch((err) => {
        console.warn('[chat-sessions] delete network error', { id, err: err?.message });
        showToast("Couldn't fully delete chat — it may reappear after reload", 'warning');
      });

    // Drop any per-mode memory that pointed at this session so the next
    // mode-switch falls back to the newest surviving session instead of
    // trying to restore a tombstone. getLastActiveForMode has the same
    // defense but skipping a doomed round-trip here is cheaper.
    for (const m of Object.keys(this._lastActiveByMode)) {
      if (this._lastActiveByMode[m] === id) {
        delete this._lastActiveByMode[m];
      }
    }
    try {
      localStorage.setItem(
        STORAGE_ACTIVE_BY_MODE,
        JSON.stringify(this._lastActiveByMode),
      );
    } catch (e) {
      console.warn('SessionStore: active-by-mode save failed', e);
    }

    // Centralized dispatch — any surface that renders session lists (narrative
    // recent-chats strip, sidebar session list, group accordion) can subscribe
    // once instead of hand-wiring a refresh call at every deletion callsite.
    document.dispatchEvent(new CustomEvent('augmentum:session-deleted', {
      detail: { sessionId: id, mode: deletedMode },
    }));

    return deletedMode;
  }

  // -------------------------------------------------------------------------
  // Active session
  // -------------------------------------------------------------------------

  getActiveId() {
    return this._activeId;
  }

  setActiveId(id) {
    if (id === this._activeId) return;  // no-op — don't re-fire listeners
    this._activeId = id;
    if (id) {
      localStorage.setItem(STORAGE_ACTIVE, id);
    } else {
      localStorage.removeItem(STORAGE_ACTIVE);
    }
    // Centralized dispatch — every listener that cares about which session
    // the user is on (inspector, narrative state, document list, agentic
    // restore, group tab visibility) hooks `augmentum:session-changed`.
    // Firing here means every callsite that changes the active session
    // (switchSession, narrative-start-chat, new-session, mode-changed,
    // anything future) automatically syncs the UI.
    const session = id ? this._sessions?.[id] : null;
    const mode = session ? (session.mode || null) : null;

    // Persisting here bumps the session's updatedAt via save()'s own
    // touch logic, so forMode() treats "just viewed" as "most recently
    // used". Explicit id required since save() no longer falls back
    // to _activeId on its own (the fallback used to cascade into a
    // mark-all-dirty path that blew up sync payloads).
    if (session) this.save(id);

    // Record per-mode "last viewed" so a later mode-switch can restore the
    // user to the exact session they were reading. Only record when we
    // actually know the session's mode — clearing the active id shouldn't
    // wipe the per-mode memory (that'd defeat the purpose).
    if (id && mode) {
      this._lastActiveByMode[mode] = id;
      try {
        localStorage.setItem(
          STORAGE_ACTIVE_BY_MODE,
          JSON.stringify(this._lastActiveByMode),
        );
      } catch { /* quota / disabled — best-effort */ }
    }

    document.dispatchEvent(new CustomEvent('augmentum:session-changed', {
      detail: { sessionId: id, mode },
    }));
  }

  /**
   * Return the session id the user was last viewing in ``mode``, or null
   * if there is no memory or the remembered session has been deleted.
   *
   * Used by the mode-change handler to restore per-mode context: clicking
   * the story orb after poking around in chat should land the user back
   * on the exact narrative they were reading, not on the newest-created
   * narrative session.
   */
  getLastActiveForMode(mode) {
    if (!mode) return null;
    const id = this._lastActiveByMode[mode];
    if (!id) return null;
    const s = this._sessions[id];
    if (!s) {
      // Session was deleted while away. Clean up the stale pointer so
      // we don't keep trying to restore it.
      delete this._lastActiveByMode[mode];
      try {
        localStorage.setItem(
          STORAGE_ACTIVE_BY_MODE,
          JSON.stringify(this._lastActiveByMode),
        );
      } catch { /* best-effort */ }
      return null;
    }
    return id;
  }

  // -------------------------------------------------------------------------
  // Flush — immediate sync for page unload
  // -------------------------------------------------------------------------

  flush() {
    clearTimeout(this._saveTimer);
    this._saveTimer = null;

    // Build per-session payloads. Sending each dirty session as its own
    // keepalive fetch means a single oversize session doesn't take down
    // the rest of the batch (the previous all-or-nothing path bailed
    // entirely as soon as the combined payload crossed 60KB — a
    // storybook chat with a chain trace + artifact card easily does).
    // Browsers cap keepalive bodies at ~64KB per origin, so sessions
    // that don't fit individually are logged and skipped; the next
    // post-load save will retry them through the normal sync path.
    const dirtyIds = Array.from(this._dirtyIds);
    if (dirtyIds.length === 0) return;

    const stripped = this._stripImagesForStorage(
      Object.fromEntries(
        dirtyIds.map(id => [id, this._sessions[id]]).filter(([, s]) => s?.tree),
      ),
    );

    for (const [id, session] of Object.entries(stripped)) {
      const payload = JSON.stringify({ sessions: { [id]: session } });
      if (payload.length > 60_000) {
        // Too big for unload-time channels — leave dirty so post-reload
        // sync retries. Sub-cases beyond the limit are rare (chat with
        // hundreds of tool cards / images) but used to silently lose data.
        console.warn(
          '[chat-sync] session too large for unload flush; will retry post-reload',
          { id, bytes: payload.length },
        );
        continue;
      }
      const blob = new Blob([payload], { type: 'application/json' });
      let sent = false;
      try {
        sent = navigator.sendBeacon('/api/chats/sync', blob);
      } catch { /* sendBeacon unavailable */ }
      if (sent) {
        this._dirtyIds.delete(id);
        continue;
      }
      // Fallback: keepalive fetch. Promise won't resolve here (the page
      // is going away) but the browser tries to complete it. Best-effort.
      try {
        fetch('/api/chats/sync', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: payload,
          keepalive: true,
        }).catch(() => {});
        this._dirtyIds.delete(id);
      } catch { /* nothing else to try */ }
    }
  }

  // -------------------------------------------------------------------------
  // Stale-sync recovery — the server rejected our copy because another
  // tab/device wrote a newer one. Fetch theirs, union-merge, re-sync.
  // -------------------------------------------------------------------------

  async _recoverStaleSessions(ids) {
    const merged = [];
    for (const id of ids) {
      const local = this._sessions[id];
      if (!local?.tree) continue;
      let server;
      try {
        const resp = await fetch(`/api/chats/${encodeURIComponent(id)}`);
        if (!resp.ok) { this._dirtyIds.add(id); continue; } // retry later
        server = await resp.json();
      } catch { this._dirtyIds.add(id); continue; }
      if (!server?.tree) { this._dirtyIds.add(id); continue; }
      this._mergeServerSession(local, server);
      merged.push(id);
      // save() stamps a fresh updatedAt and re-queues the sync — the
      // merged tree now supersedes both writers, so the next round is
      // accepted. Concurrent edits during the merge just trigger
      // another merge round; content is unioned each time, never lost.
      this.save(id);
      // Clock-skew guard: if this device's clock trails the other's,
      // the Date.now() stamp from save() could still be older than the
      // server blob's and the re-sync would bounce forever. Strictly
      // supersede the copy we just merged.
      local.updatedAt = Math.max(local.updatedAt, (server.updatedAt || 0) + 1);
      console.info('[sessions] union-merged newer server copy of', id);
    }
    if (merged.length) {
      document.dispatchEvent(new CustomEvent('augmentum:sessions-reconciled', {
        detail: { mergedIds: merged },
      }));
    }
  }

  _mergeServerSession(local, server) {
    // Union of both trees, server copy as the base (it holds the turns
    // the other device added). Node ids are globally unique, so any node
    // only WE have is exactly the work this tab did on its stale base —
    // graft it back in and re-link every node into its parent's children
    // list. A local node whose parent chain was deleted server-side stays
    // in the blob unreachable rather than being dropped: preserved, never
    // rendered — deliberate, this code must never destroy a turn.
    const mergedTree = server.tree;
    for (const [nid, node] of Object.entries(local.tree)) {
      if (!mergedTree[nid]) mergedTree[nid] = node;
    }
    for (const [nid, node] of Object.entries(mergedTree)) {
      const pid = node.parentId;
      if (pid && mergedTree[pid]) {
        const kids = mergedTree[pid].children || (mergedTree[pid].children = []);
        if (!kids.includes(nid)) kids.push(nid);
      }
    }
    local.tree = mergedTree;
    if (!mergedTree[local.activeLeafId]) {
      local.activeLeafId =
        (server.activeLeafId && mergedTree[server.activeLeafId])
          ? server.activeLeafId
          : (local.rootId || server.rootId || null);
    }
    if (!local.rootId && server.rootId) local.rootId = server.rootId;
    // Prefer the other device's title when ours is still the default.
    if ((!local.title || local.title === 'New Chat') && server.title) {
      local.title = server.title;
    }
  }

  // -------------------------------------------------------------------------
  // syncNow — bypass the 500ms debounce for state changes that absolutely
  // must reach the server before the user might refresh (e.g. an agentic
  // chain's final assistant message + artifact card, which can take 10+
  // minutes to generate — the user immediately wants to see the result and
  // a fast refresh inside the debounce window used to lose everything).
  // -------------------------------------------------------------------------

  syncNow(touchedSessionId = null) {
    if (touchedSessionId && this._sessions[touchedSessionId]) {
      this._sessions[touchedSessionId].updatedAt = Date.now();
      this._dirtyIds.add(touchedSessionId);
    }
    clearTimeout(this._saveTimer);
    this._saveTimer = null;
    return this._syncToServer();
  }

  // -------------------------------------------------------------------------
  // Import / Export
  // -------------------------------------------------------------------------

  import(sessionsObj) {
    const importedIds = [];
    for (const [id, session] of Object.entries(sessionsObj)) {
      // Ensure v2 format
      this._sessions[id] = session.version === 2
        ? session
        : migrateSessionToV2(session);
      importedIds.push(id);
    }
    // Each imported session needs to land on the server. Mark them
    // explicitly because save() no longer cascades to "all sessions
    // with tree data" when called without an id.
    for (const id of importedIds) this.markDirty(id);
    this.save();
  }

  export(id) {
    const session = this._sessions[id];
    if (!session) return null;
    return JSON.parse(JSON.stringify(session));
  }

  exportAll() {
    return JSON.parse(JSON.stringify(this._sessions));
  }
}

export const sessionStore = new SessionStore();
