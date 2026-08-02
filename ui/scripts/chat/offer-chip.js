/* ==========================================================================
   Offer Chip — inline render for chat-LLM-emitted Install/Save/Switch chips.

   The model calls the `propose_offer` tool; the dispatcher publishes a
   notification on channel `system.offer`; this module renders that
   notification as a chip in the chat stream with Install/Not now/Never
   buttons.

   Spec: docs/superpowers/specs/2026-06-02-offer-substrate-design.md
   ========================================================================== */

import { escapeHtml, extractErrorMessage } from '../app.js';

const OFFER_CHANNEL_ID = 'system.offer';

// Reconnect bounds mirror notifications.js so both substrates back off
// the same way under shared failure modes (auth subsystem down, server
// restart). Capped tighter than chat/voice WSes — the offer feed is
// auxiliary, not load-bearing.
const RECONNECT_BASE_MS = 1500;
const RECONNECT_MAX_MS = 30000;

/**
 * Render one offer notification into a DOM element.
 *
 * @param {object} notif  notification row (channel_id=system.offer)
 * @param {object} opts   {onAfterAction?: (offerId, actionId, result) => void}
 * @returns {HTMLElement}
 */
export function renderOfferChip(notif, opts = {}) {
  const el = document.createElement('div');
  el.className = 'offer-chip';
  el.dataset.offerId = notif.notification_id;
  const kind = (notif.payload && notif.payload.kind) || '';
  const targetId = (notif.payload && notif.payload.target_id) || '';
  if (kind) el.dataset.kind = kind;
  if (targetId) el.dataset.targetId = targetId;

  const scope = (notif.payload && notif.payload.scope) || 'user';
  const preview = (notif.payload && notif.payload.preview) || {};
  const isAdminScope = scope === 'admin';

  // Hover-grade markup, no framework. The styling lives in chat.css —
  // any host page importing this file is expected to ship those rules
  // (or accept the unstyled fallback gracefully).
  el.innerHTML = `
    <div class="offer-chip-icon" aria-hidden="true">${_iconSvg(notif.icon)}</div>
    <div class="offer-chip-body">
      <div class="offer-chip-title">${escapeHtml(notif.title || 'Suggestion')}</div>
      ${notif.body ? `<div class="offer-chip-why">${escapeHtml(notif.body)}</div>` : ''}
      ${_previewLine(preview)}
      ${isAdminScope ? '<div class="offer-chip-scope-hint">Admin-only — ask your admin to install.</div>' : ''}
      <div class="offer-chip-actions" role="group">
        <button type="button" class="offer-chip-btn primary" data-action="accept">${escapeHtml(_acceptLabel(notif))}</button>
        <button type="button" class="offer-chip-btn" data-action="snooze">Not now</button>
        <button type="button" class="offer-chip-btn ghost" data-action="never">Never</button>
      </div>
      <div class="offer-chip-status" aria-live="polite"></div>
    </div>
  `;

  const statusEl = el.querySelector('.offer-chip-status');
  const actionsEl = el.querySelector('.offer-chip-actions');

  actionsEl.addEventListener('click', async (ev) => {
    const btn = ev.target.closest('button[data-action]');
    if (!btn || btn.disabled) return;
    const actionId = btn.dataset.action;
    // Disable all buttons so a double-click can't fire twice.
    actionsEl.querySelectorAll('button').forEach(b => { b.disabled = true; });
    statusEl.textContent = actionId === 'accept' ? 'Working…' : '';

    try {
      const res = await fetch(
        `/api/notify/${encodeURIComponent(notif.notification_id)}/action/${encodeURIComponent(actionId)}`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' },
      );
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = extractErrorMessage(body, `Action ${actionId} failed`);
        statusEl.textContent = msg;
        // Re-enable so the user can retry (except for never — once
        // never has been posted, retrying makes no sense).
        if (actionId !== 'never') {
          actionsEl.querySelectorAll('button').forEach(b => { b.disabled = false; });
        }
        if (opts.onAfterAction) opts.onAfterAction(notif.notification_id, actionId, null);
        return;
      }
      _renderTerminalState(el, actionId, body.result || {});
      if (opts.onAfterAction) opts.onAfterAction(notif.notification_id, actionId, body.result || {});
    } catch (err) {
      statusEl.textContent = err && err.message ? err.message : 'Network error';
      actionsEl.querySelectorAll('button').forEach(b => { b.disabled = false; });
    }
  });

  return el;
}

function _previewLine(preview) {
  if (!preview || typeof preview !== 'object') return '';
  const label = preview.label ? `<span class="offer-chip-preview-label">${escapeHtml(preview.label)}</span>` : '';
  const hint = preview.hint ? `<span class="offer-chip-preview-hint">${escapeHtml(preview.hint)}</span>` : '';
  if (!label && !hint) return '';
  return `<div class="offer-chip-preview">${label}${label && hint ? ' — ' : ''}${hint}</div>`;
}

function _acceptLabel(notif) {
  // The catalog could ship a custom accept-button label via payload
  // some day; for v1 just key off the kind so a Mode-switch chip
  // doesn't say "Install."
  const kind = (notif.payload && notif.payload.kind) || '';
  if (kind === 'mode_switch') return 'Switch';
  if (kind === 'setting_tweak') return 'Apply';
  if (kind === 'memory_save') return 'Save';
  if (kind === 'character_pin') return 'Pin';
  if (kind === 'gated_tool') {
    // Heavy tools the model proposed — confirm to run them.
    const t = (notif.payload && notif.payload.target_id) || '';
    if (t === 'image_generation') return 'Generate';
    if (t === 'build_application') return 'Build it';
    return 'Go ahead';
  }
  return 'Install';
}

function _iconSvg(name) {
  // Lightweight lucide-style "lightbulb" by default; the rest come
  // from existing chat icons via the host stylesheet's data-attr.
  if (name === 'mail') {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4h16v16H4z"/><path d="M4 4l8 8 8-8"/></svg>';
  }
  return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M5 9a7 7 0 1 1 14 0c0 5-3 6-3 9H8c0-3-3-4-3-9z"/><path d="M9 18h6"/></svg>';
}

function _renderTerminalState(el, actionId, result) {
  const statusEl = el.querySelector('.offer-chip-status');
  const actionsEl = el.querySelector('.offer-chip-actions');

  if (actionId === 'accept') {
    if (result && result.ok === false) {
      statusEl.textContent = result.detail || result.error || 'Accept failed';
      actionsEl.querySelectorAll('button').forEach(b => { b.disabled = false; });
      return;
    }
    el.classList.add('accepted');
    // A build_application offer kicks off a coder-workspace build — attach the
    // live build card so the user tracks it (same surface as build mode).
    if (result && result.build_id) {
      import('./build-card.js')
        .then(m => m.handleBuildStarted({ build_id: result.build_id, name: result.name }))
        .catch(() => {});
    }
    // Snippet payloads (mcp_client_config kind) carry a paste-ready
    // config block; render with a Copy button instead of just the
    // generic "✓ done" line.
    if (result && result.kind === 'snippet' && typeof result.snippet === 'string') {
      _renderSnippetTerminalState(el, result);
      return;
    }
    const next = (result && (result.next_step || result.message)) || 'Done.';
    actionsEl.innerHTML = `<span class="offer-chip-done">✓ ${escapeHtml(next)}</span>`;
  } else if (actionId === 'snooze') {
    // "Not now" dismisses this chip and nothing more — no 30-day mute, so
    // the wording must not imply one. See migration 326.
    el.classList.add('snoozed');
    actionsEl.innerHTML = '<span class="offer-chip-done">Dismissed.</span>';
  } else if (actionId === 'never') {
    // Never is now the ONLY action that suppresses anything, so its undo has
    // to be real. It used to point at "Settings → Offers", a surface that was
    // never built — leaving the one permanent choice reversible only via the
    // API. The Undo lives on the chip instead: same DELETE endpoint, reachable
    // at the moment of doubt rather than three menus away.
    el.classList.add('nevered');
    actionsEl.innerHTML = `
      <span class="offer-chip-done">Won’t suggest again.</span>
      <button type="button" class="offer-chip-btn small" data-act="undo-never">Undo</button>
    `;
    _wireUndoNever(el, actionsEl);
  }
}


function _wireUndoNever(el, actionsEl) {
  const btn = actionsEl.querySelector('button[data-act="undo-never"]');
  if (!btn) return;
  const kind = el.dataset.kind || '';
  const targetId = el.dataset.targetId || '';
  if (!kind || !targetId) { btn.remove(); return; }

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = 'Undoing…';
    try {
      const res = await fetch(
        `/api/offers/suppressions/${encodeURIComponent(kind)}/${encodeURIComponent(targetId)}`,
        { method: 'DELETE', credentials: 'same-origin' },
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      el.classList.remove('nevered');
      actionsEl.innerHTML =
        '<span class="offer-chip-done">Restored — this can be suggested again.</span>';
    } catch (err) {
      btn.disabled = false;
      btn.textContent = 'Undo';
      const statusEl = el.querySelector('.offer-chip-status');
      if (statusEl) {
        statusEl.textContent =
          err && err.message ? `Undo failed: ${err.message}` : 'Undo failed';
      }
    }
  });
}


function _renderSnippetTerminalState(el, result) {
  const bodyEl = el.querySelector('.offer-chip-body');
  const next = result.next_step || 'Paste into your client config and reload.';
  const langClass = result.language ? ` language-${result.language}` : '';

  // Replace the post-accept area with a snippet block + copy button.
  // We append to body instead of replacing so the original title +
  // "why" lines stay visible — useful context once the snippet is
  // in the user's clipboard.
  const wrap = document.createElement('div');
  wrap.className = 'offer-chip-snippet-wrap';
  wrap.innerHTML = `
    <div class="offer-chip-snippet-meta">
      <span>${escapeHtml(result.file || 'Snippet')}</span>
      <button type="button" class="offer-chip-btn small" data-act="copy">Copy</button>
    </div>
    <pre class="offer-chip-snippet${langClass}"><code></code></pre>
    <div class="offer-chip-done">${escapeHtml(next)}</div>
  `;
  // Use textContent on the inner <code> so HTML inside the snippet
  // (very unlikely but possible in JSON values) doesn't render.
  wrap.querySelector('code').textContent = result.snippet;

  const copyBtn = wrap.querySelector('button[data-act="copy"]');
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(result.snippet);
      copyBtn.textContent = 'Copied!';
      setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
    } catch {
      copyBtn.textContent = 'Press ⌘C';
    }
  });

  const actionsEl = el.querySelector('.offer-chip-actions');
  actionsEl.replaceWith(wrap);
}


/**
 * Subscribe to the live offer feed and append chips into a container.
 *
 * Returns a {dispose()} handle. Re-entrant: calling mountOfferFeed
 * twice on the same container is undefined behaviour — the caller
 * owns dispose semantics.
 *
 * An offer belongs to the chat that produced it. `threadId` may be a
 * function so it re-resolves per notification — the feed is mounted once on
 * the primary container but the active session changes under it, and a
 * value captured at mount would pin the filter to whichever chat happened
 * to be open first.
 *
 * Filtering FAILS CLOSED: with a known thread, only offers carrying that
 * exact `thread_id` render. It used to fall open whenever either side was
 * empty, which is how one chat's offer chip appeared in every chat on every
 * signed-in device. Unthreaded (system-wide) offers belong to the
 * notification center, not the chat stream; pass `includeUnthreaded: true`
 * for a surface that genuinely wants them.
 *
 * @param {HTMLElement} container  where chips are appended
 * @param {object} opts  {threadId?: string|() => string, fetchExisting?: bool,
 *                        includeUnthreaded?: bool}
 */
export function mountOfferFeed(container, opts = {}) {
  if (!container) throw new Error('mountOfferFeed: container required');

  const resolveThreadId = () => {
    const t = typeof opts.threadId === 'function' ? opts.threadId() : opts.threadId;
    return t || '';
  };
  const includeUnthreaded = opts.includeUnthreaded === true;
  // Keyed by `${thread}:${id}` rather than bare id: switching sessions wipes
  // the chips out of the DOM along with the messages, so coming back has to
  // be able to re-render them.
  const renderedKeys = new Set();
  let ws = null;
  let closed = false;
  let reconnectMs = RECONNECT_BASE_MS;
  let reconnectTimer = null;

  const belongsHere = (notif, threadId) => {
    const own = notif.thread_id || '';
    if (!own) return includeUnthreaded;
    if (!threadId) return false;
    return own === threadId;
  };

  const insert = (notif) => {
    if (!notif || notif.channel_id !== OFFER_CHANNEL_ID) return;
    const threadId = resolveThreadId();
    if (!belongsHere(notif, threadId)) return;
    const key = `${threadId}:${notif.notification_id}`;
    if (renderedKeys.has(key)) return;
    renderedKeys.add(key);
    container.appendChild(renderOfferChip(notif));
  };

  // Backfill the chips for THIS chat. Scoped server-side via thread_id so a
  // long-lived account doesn't ship every other session's offers over the
  // wire just to have the client drop them. (Dismissed rows — including
  // "Not now" — are already excluded by include_dismissed=false.)
  const backfill = () => {
    if (opts.fetchExisting === false) return;
    const threadId = resolveThreadId();
    if (!threadId && !includeUnthreaded) return;
    const url = '/api/notify/feed?include_dismissed=false'
      + (threadId ? `&thread_id=${encodeURIComponent(threadId)}` : '');
    fetch(url, { credentials: 'same-origin' })
      .then(r => r.ok ? r.json() : { items: [] })
      .then(data => {
        for (const item of (data.items || [])) insert(item);
      })
      .catch(() => { /* silent — backfill is opportunistic */ });
  };

  backfill();

  // The notify WS is gated on a ws-ticket (see augmentum/auth/middleware.py
  // — only /api/coder/preview/, /api/cast/receiver/, and /api/voice/sessions/
  // accept a same-origin cookie). Mint one before each connect attempt.
  const fetchTicket = async () => {
    try {
      const resp = await fetch('/api/auth/ws-ticket', {
        method: 'POST',
        credentials: 'same-origin',
      });
      if (!resp.ok) return null;
      const data = await resp.json();
      return data.ticket || null;
    } catch {
      return null;
    }
  };

  const scheduleReconnect = () => {
    if (closed) return;
    const jitter = Math.floor(Math.random() * 500);
    const wait = Math.min(reconnectMs + jitter, RECONNECT_MAX_MS);
    reconnectMs = Math.min(reconnectMs * 2, RECONNECT_MAX_MS);
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, wait);
  };

  const connect = async () => {
    if (closed) return;
    const ticket = await fetchTicket();
    if (closed) return;
    if (!ticket) {
      scheduleReconnect();
      return;
    }
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl =
      `${proto}//${window.location.host}/api/notify/subscribe`
      + `?ticket=${encodeURIComponent(ticket)}`
      + `&channel_pattern=${encodeURIComponent(OFFER_CHANNEL_ID)}`;
    let next;
    try {
      next = new WebSocket(wsUrl);
    } catch {
      scheduleReconnect();
      return;
    }
    ws = next;
    next.addEventListener('open', () => {
      reconnectMs = RECONNECT_BASE_MS;
    });
    next.addEventListener('message', (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg && msg.type === 'notification') insert(msg.notification);
    });
    next.addEventListener('error', () => { /* close handler reconnects */ });
    next.addEventListener('close', () => {
      if (ws === next) ws = null;
      scheduleReconnect();
    });
  };

  connect();

  return {
    /** Re-backfill after the active session changed.
     *
     *  Switching sessions re-renders the message list, which wipes every chip
     *  out of the container — so the render bookkeeping has to be cleared too,
     *  or returning to a chat we've already visited would suppress its chips
     *  as "already rendered" when in fact nothing is on screen. */
    rescope() {
      if (closed) return;
      renderedKeys.clear();
      backfill();
    },
    dispose() {
      if (closed) return;
      closed = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      try { if (ws) ws.close(); } catch { /* ignore */ }
    },
  };
}
