// connect-guest.js — durable guest-pass surface (Connect Phase 3c).
//
// The saved homescreen app for an invited guest. On launch it exchanges its
// durable grant token (stored at claim) for a scoped session, then shows a
// single-relationship surface to text — or request a call from — the host who
// invited them. Goes dark when the host revokes (410). Self-contained: it talks
// to the same /api/connect endpoints the main app does, with the guest session.

const $ = (s, r = document) => r.querySelector(s);
const TOKEN_KEY = 'augmentum_guest_grant_token';

let HOST_DID = '';
let HOST_NAME = '';
let SCOPES = [];
let THREAD_ID = '';
let _poll = null;

function show(state) {
  for (const el of document.querySelectorAll('[data-state]')) {
    el.hidden = el.getAttribute('data-state') !== state;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function ended(title, sub) {
  if (_poll) clearInterval(_poll);
  $('[data-ended-title]').textContent = title;
  $('[data-ended-sub]').textContent = sub || '';
  show('ended');
}

async function api(path, opts = {}) {
  return fetch(path, { credentials: 'same-origin', ...opts });
}

async function establish() {
  const token = (localStorage.getItem(TOKEN_KEY) || '').trim();
  if (!token) {
    return ended('No guest pass found', 'Open the invite link your host sent you to set this up.');
  }
  let res;
  try {
    res = await api('/api/connect/guest/session', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ grant_token: token }),
    });
  } catch {
    return ended('Connection problem', 'Could not reach the server. Try again in a moment.');
  }
  if (res.status === 410) {
    localStorage.removeItem(TOKEN_KEY);
    return ended('Access ended', 'Your host has turned off this guest access.');
  }
  if (!res.ok) {
    return ended('Something went wrong', 'Please try again later.');
  }
  const data = await res.json();
  HOST_DID = data.host?.did || '';
  HOST_NAME = (data.host?.display_name || '').trim() || (HOST_DID.split('@')[0] || 'your host');
  SCOPES = data.scopes || ['text'];
  renderActive();
}

function renderActive() {
  $('[data-host-name]').textContent = HOST_NAME;
  $('[data-host-initial]').textContent = (HOST_NAME[0] || '?').toUpperCase();
  const callBtn = $('[data-call]');
  callBtn.hidden = !SCOPES.includes('call');
  callBtn.addEventListener('click', requestCall);
  $('[data-composer]').addEventListener('submit', onSend);
  const input = $('[data-input]');
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(e); }
  });
  show('active');
  maybeOfferInstall();
  loadThread();
  _poll = setInterval(loadThread, 5000);
}

async function loadThread() {
  // Find (or wait for) the thread with the host, then pull recent messages.
  try {
    if (!THREAD_ID) {
      const tr = await api('/api/connect/threads');
      if (tr.ok) {
        const threads = (await tr.json()).threads || [];
        const t = threads.find((x) => (x.peer_did || '') === HOST_DID);
        if (t) THREAD_ID = t.thread_id;
      }
    }
    if (!THREAD_ID) return;
    const mr = await api(`/api/connect/threads/${encodeURIComponent(THREAD_ID)}/messages`);
    if (mr.status === 410) return ended('Access ended', 'Your host has turned off this guest access.');
    if (!mr.ok) return;
    const messages = (await mr.json()).messages || [];
    renderMessages(messages);
  } catch { /* transient — next poll retries */ }
}

function renderMessages(messages) {
  const thread = $('[data-thread]');
  // API returns newest-first; show oldest→newest.
  const ordered = [...messages].reverse();
  thread.innerHTML = ordered.map((m) => {
    const mine = (m.sender_did || '') !== HOST_DID;
    const body = m.deleted_at ? '<em>deleted</em>' : escapeHtml(m.body || '');
    return `<div class="cg-msg ${mine ? 'cg-mine' : 'cg-theirs'}">${body}</div>`;
  }).join('');
  thread.scrollTop = thread.scrollHeight;
}

async function onSend(e) {
  e.preventDefault();
  const input = $('[data-input]');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  const messageId = `m_${Date.now()}_${Math.floor(performance.now())}`;
  const threadId = THREAD_ID || `g_${Date.now()}`;
  // Optimistic append.
  const thread = $('[data-thread]');
  thread.insertAdjacentHTML('beforeend', `<div class="cg-msg cg-mine">${escapeHtml(text)}</div>`);
  thread.scrollTop = thread.scrollHeight;
  try {
    await api(`/api/connect/threads/${encodeURIComponent(threadId)}/send`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ peer_did: HOST_DID, message_id: messageId, body: text }),
    });
    // Nudge the host so a closed app still surfaces it.
    api('/api/connect/guest/ping', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ kind: 'text', peer_did: HOST_DID }),
    }).catch(() => {});
    if (!THREAD_ID) THREAD_ID = threadId;
    loadThread();
  } catch { /* the optimistic bubble stays; next action retries */ }
}

async function requestCall() {
  const btn = $('[data-call]');
  btn.disabled = true;
  try {
    const r = await api('/api/connect/guest/ping', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ kind: 'call', peer_did: HOST_DID }),
    });
    btn.textContent = r.ok ? 'Asked…' : 'Call';
  } catch { /* ignore */ }
  setTimeout(() => { btn.disabled = false; btn.querySelector('span') || (btn.textContent = 'Call'); }, 4000);
}

// --- Add to home screen ---
let _deferredPrompt = null;
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  _deferredPrompt = e;
  const el = $('[data-install]');
  if (el && document.querySelector('[data-state="active"]')?.hidden === false) el.hidden = false;
});
function maybeOfferInstall() {
  const el = $('[data-install]');
  if (!el) return;
  if (_deferredPrompt) el.hidden = false;
  $('[data-install-btn]')?.addEventListener('click', async () => {
    el.hidden = true;
    if (_deferredPrompt) { _deferredPrompt.prompt(); _deferredPrompt = null; }
  });
}

establish();
