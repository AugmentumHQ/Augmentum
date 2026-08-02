/**
 * cast-stage.js — the literary management surface for paired
 * receivers. Talks to the same endpoints as the cast shelf modal,
 * but presents them as an editorial almanac rather than a settings
 * panel: live stages, standing-by stages, and a written chronicle
 * of every cast that's passed through.
 *
 * Talks to:
 *   GET    /api/cast/trusted-receivers
 *   PATCH  /api/cast/trusted-receivers/:id
 *   POST   /api/cast/trusted-receivers/:id/revoke
 *   GET    /api/cast/cast-events?limit=N
 *   GET    /api/cast/receivers     (live-connection cross-check)
 */

const POLL_MS = 10_000;
const EVENT_LIMIT = 25;
const REVOKE_CONFIRM_WINDOW_MS = 3000;
const TOAST_MS = 2400;

const state = {
  trusted: [],         // trusted_receivers rows (with currently_showing)
  events: [],          // recent cast events
  liveConnectedIds: new Set(),   // trusted_ids currently connected
  pollTimer: null,
  pendingRevokes: new Map(),     // trusted_id → setTimeout handle
  renameTarget: null,            // { id, label }
};

const els = {
  statLive:        document.querySelector('[data-cs-stat-live]'),
  statPaired:      document.querySelector('[data-cs-stat-paired]'),
  statLast:        document.querySelector('[data-cs-stat-last]'),
  activeBody:      document.querySelector('[data-cs-active]'),
  quietSection:    document.querySelector('[data-cs-movement="quiet"]'),
  quietBody:       document.querySelector('[data-cs-quiet]'),
  chronicleBody:   document.querySelector('[data-cs-chronicle]'),
  renameCard:      document.querySelector('[data-cs-rename]'),
  renamePrev:      document.querySelector('[data-cs-rename-prev]'),
  renameForm:      document.querySelector('[data-cs-rename-form]'),
  renameInput:     document.querySelector('[data-cs-rename-input]'),
  renameCancel:    document.querySelector('[data-cs-rename-cancel]'),
  toast:           document.querySelector('[data-cs-toast]'),
};

const FRIENDLY_KIND = {
  'vrm.avatar':   'the companion',
  'html.generic': 'a window',
  'media.image':  'an image',
  'media.video':  'a film',
  'media.audio':  'an audio piece',
};

const FRIENDLY_PLATFORM = {
  'android-tv': 'android tv',
  'browser':    'browser',
  'tizen':      'samsung tv',
  'webos':      'lg tv',
  'roku':       'roku',
  'apple-tv':   'apple tv',
};


/* ── escape ────────────────────────────────────────────────── */

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"'`$]/g, c => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;',
    '"':'&quot;', "'":'&#39;', '`':'&#96;', '$':'&#36;',
  }[c]));
}


/* ── fetchers ──────────────────────────────────────────────── */

async function loadAll() {
  // ``include_revoked=true`` so the Severed section below can offer a
  // restore button for accidental revokes. Render-side filtering keeps
  // the Active/Quiet movements clean.
  const [trusted, events, live] = await Promise.allSettled([
    fetchJson('/api/cast/trusted-receivers?include_revoked=true'),
    fetchJson(`/api/cast/cast-events?limit=${EVENT_LIMIT}`),
    fetchJson('/api/cast/receivers'),
  ]);

  state.trusted = (trusted.status === 'fulfilled' && Array.isArray(trusted.value?.receivers))
    ? trusted.value.receivers : [];
  state.events = (events.status === 'fulfilled' && Array.isArray(events.value?.events))
    ? events.value.events : [];
  state.liveConnectedIds = new Set(
    (live.status === 'fulfilled' && Array.isArray(live.value?.receivers))
      ? live.value.receivers.map(r => r.trusted_id).filter(Boolean)
      : []
  );
  render();
}

async function fetchJson(path, opts = {}) {
  const res = await fetch(path, {
    credentials: 'same-origin',
    headers: { Accept: 'application/json', ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${path} ${res.status}: ${text.slice(0, 120)}`);
  }
  return res.json();
}


/* ── render ────────────────────────────────────────────────── */

function render() {
  renderStats();
  renderActive();
  renderQuiet();
  renderSevered();
  renderChronicle();
}

function renderStats() {
  const active = state.trusted.filter(t =>
    state.liveConnectedIds.has(t.id) && !t.revoked
  );
  const paired = state.trusted.filter(t => !t.revoked);

  els.statLive.textContent = String(active.length);
  els.statLive.classList.toggle('is-accent', active.length > 0);
  els.statPaired.textContent = String(paired.length);

  const lastCastTs = state.events[0]?.started_at || '';
  els.statLast.textContent = lastCastTs ? humanRelative(lastCastTs) : '—';
}

function renderActive() {
  const active = state.trusted.filter(t =>
    state.liveConnectedIds.has(t.id) && !t.revoked
  );
  if (!active.length) {
    els.activeBody.innerHTML = `
      <p class="empty">
        No stage is awake at the moment.
        ${state.trusted.length === 0
          ? `When you <a href="/ui/cast-receiver/" target="_blank" rel="noopener">open the receiver page</a>
             on a TV, phone, or browser tab in your home and approve the pair,
             it will appear here — named, trusted, ready.`
          : `Your paired stages are listed below.`}
      </p>`;
    return;
  }
  els.activeBody.innerHTML = active.map((t, i) => stageActiveHtml(t, i)).join('');
  wireStageActions(els.activeBody);
}

function renderQuiet() {
  const quiet = state.trusted.filter(t =>
    !state.liveConnectedIds.has(t.id) && !t.revoked
  );
  if (!quiet.length) {
    els.quietSection.hidden = true;
    return;
  }
  els.quietSection.hidden = false;
  els.quietBody.innerHTML = quiet.map((t, i) => stageQuietHtml(t, i)).join('');
  wireStageActions(els.quietBody);
}

function renderSevered() {
  // Revoked entries surfaced last so the user has a recovery path
  // without DB surgery. If none, the section just doesn't exist —
  // we render directly into the chronicle's container area by
  // inserting a sibling section. To keep this minimal we render
  // into the quiet body's parent when needed; if a dedicated DOM
  // node `severedBody` exists in the page we use that.
  const severed = state.trusted.filter(t => t.revoked);
  // Ensure a host section exists. Created lazily so the rest of the
  // page layout doesn't have to change for this rescue feature.
  let host = document.querySelector('[data-cs-severed]');
  if (!severed.length) {
    if (host) host.hidden = true;
    return;
  }
  if (!host) {
    const main = document.querySelector('main') || document.body;
    host = document.createElement('section');
    host.className = 'movement movement-severed';
    host.dataset.csSevered = '';
    host.innerHTML = `
      <header class="movement-head">
        <span class="movement-numeral">IV.</span>
        <span class="movement-title">Severed stages</span>
        <span class="movement-rule" aria-hidden="true"></span>
      </header>
      <div class="movement-body" data-cs-severed-body></div>
    `;
    // Place before the chronicle (last movement) so revoked sit
    // between Standing-by and the chronicle.
    const chronicle = document.querySelector('[data-cs-movement="chronicle"]');
    if (chronicle && chronicle.parentNode) {
      chronicle.parentNode.insertBefore(host, chronicle);
    } else {
      main.appendChild(host);
    }
  }
  host.hidden = false;
  const body = host.querySelector('[data-cs-severed-body]');
  if (body) {
    body.innerHTML = severed.map((t, i) => stageSeveredHtml(t, i)).join('');
    wireSeveredActions(body);
  }
}

function stageSeveredHtml(t, idx) {
  const num = String(idx + 1).padStart(2, '0');
  const platform = FRIENDLY_PLATFORM[t.platform] || t.platform || 'screen';
  const when = t.revoked_at ? humanRelative(t.revoked_at) : 'an unknown moment';
  return `
    <article class="stage-quiet stage-severed" data-cs-id="${esc(t.id)}">
      <div class="stage-number">${esc(num)}</div>
      <div class="stage-main">
        <h3 class="stage-name">${esc(t.label || 'Unnamed stage')}</h3>
        <p class="stage-status">Severed ${esc(when)}. Connecting from this device is denied until restored.</p>
        <div class="stage-meta">
          <span>${esc(platform)}</span>
        </div>
      </div>
      <div class="stage-actions">
        <button class="link-button" data-cs-action="restore" data-cs-id="${esc(t.id)}">restore trust</button>
      </div>
    </article>
  `;
}

function wireSeveredActions(scope) {
  scope.querySelectorAll('[data-cs-action="restore"]').forEach(btn => {
    btn.addEventListener('click', () => doRestore(btn.dataset.csId));
  });
}

async function doRestore(trustedId) {
  try {
    const res = await fetch(
      `/api/cast/trusted-receivers/${encodeURIComponent(trustedId)}/restore`,
      { method: 'POST', credentials: 'same-origin' }
    );
    if (!res.ok) throw new Error(`status ${res.status}`);
    toast('Trust restored.');
    await loadAll();
  } catch (err) {
    toast(`Couldn't restore — ${err.message || err}`);
  }
}


function renderChronicle() {
  if (!state.events.length) {
    els.chronicleBody.innerHTML = `
      <p class="empty">
        Nothing has crossed these screens yet.
        When something does, the chronicle will keep it.
      </p>`;
    return;
  }
  els.chronicleBody.innerHTML = state.events
    .map(chronicleEntryHtml)
    .join('');
}


/* ── HTML fragments ────────────────────────────────────────── */

function stageActiveHtml(t, idx) {
  const num = String(idx + 1).padStart(2, '0');
  const showing = (t.currently_showing && t.currently_showing[0]) || null;
  const platform = FRIENDLY_PLATFORM[t.platform] || t.platform || 'screen';
  const paired = t.created_at ? humanRelative(t.created_at) : 'recently';
  const deviceTail = t.device_id ? t.device_id.slice(-6) : 'ephemeral';

  let status;
  if (showing) {
    status = `Right now, showing <em>${esc(FRIENDLY_KIND[showing.surface_kind] || showing.surface_kind || 'something')}</em> in the ${esc(showing.slot || 'main')} slot.`;
  } else {
    status = `Connected, but quiet — waiting for something to show.`;
  }

  return `
    <article class="stage is-live" data-cs-id="${esc(t.id)}">
      <div class="stage-number">${esc(num)}</div>
      <div class="stage-main">
        <div class="stage-name-row">
          <span class="orb" aria-hidden="true"></span>
          <h3 class="stage-name">${esc(t.label || 'Unnamed stage')}</h3>
        </div>
        <p class="stage-status">${status}</p>
        <div class="stage-meta">
          <span>${esc(platform)}</span>
          <span class="stage-meta-sep">·</span>
          <span>paired ${esc(paired)}</span>
          <span class="stage-meta-sep">·</span>
          <span>id ${esc(deviceTail)}</span>
        </div>
      </div>
      <div class="stage-actions">
        <button class="link-button" data-cs-action="rename" data-cs-id="${esc(t.id)}">rename</button>
        <button class="link-button link-button-warn" data-cs-action="revoke" data-cs-id="${esc(t.id)}">revoke</button>
      </div>
    </article>
  `;
}

function stageQuietHtml(t, idx) {
  const num = String(idx + 1).padStart(2, '0');
  const platform = FRIENDLY_PLATFORM[t.platform] || t.platform || 'screen';
  const lastSeen = t.last_seen_at ? humanRelative(t.last_seen_at) : 'never';
  const lastCast = t.last_cast_at ? humanRelative(t.last_cast_at) : 'never';

  return `
    <article class="stage-quiet" data-cs-id="${esc(t.id)}">
      <div class="stage-number">${esc(num)}</div>
      <div class="stage-main">
        <h3 class="stage-name">${esc(t.label || 'Unnamed stage')}</h3>
        <div class="stage-meta">
          <span>${esc(platform)}</span>
          <span class="stage-meta-sep">·</span>
          <span>last seen ${esc(lastSeen)}</span>
          <span class="stage-meta-sep">·</span>
          <span>last cast ${esc(lastCast)}</span>
        </div>
      </div>
      <div class="stage-actions">
        <button class="link-button" data-cs-action="rename" data-cs-id="${esc(t.id)}">rename</button>
        <button class="link-button link-button-warn" data-cs-action="revoke" data-cs-id="${esc(t.id)}">revoke</button>
      </div>
    </article>
  `;
}

function chronicleEntryHtml(ev) {
  const dt = chronicleDate(ev.started_at);
  const trusted = state.trusted.find(t => t.id === ev.trusted_id);
  const tvName = trusted ? (trusted.label || trusted.id) : (ev.trusted_id || 'a browser tab');
  const kind = FRIENDLY_KIND[ev.surface_kind] || ev.surface_kind || 'something';

  let body;
  if (ev.active) {
    body = `<strong>${esc(tvName)}</strong> is <span class="ongoing">showing ${esc(kind)}</span>.`;
  } else {
    const dur = duration(ev.started_at, ev.ended_at);
    const verb = endVerb(ev.end_reason);
    body = `<strong>${esc(tvName)}</strong>
      <span class="verb">showed</span> ${esc(kind)}${dur ? ' for ' + esc(dur) : ''}.
      ${esc(verb)}`;
  }

  return `
    <div class="chronicle-entry">
      <span class="chronicle-date">${esc(dt)}</span>
      <span class="chronicle-body">${body}</span>
    </div>
  `;
}

function endVerb(reason) {
  switch (reason) {
    case 'user_stop':     return 'You let it go.';
    case 'replaced':      return 'Another took its place.';
    case 'disconnected':  return 'The connection slipped away.';
    case 'ended':         return 'It finished gracefully.';
    default:              return '';
  }
}


/* ── time helpers ──────────────────────────────────────────── */

function parseIsoUtc(s) {
  if (!s) return null;
  const d = new Date(s.replace(' ', 'T') + 'Z');
  return Number.isNaN(d.getTime()) ? null : d;
}

function humanRelative(ts) {
  const d = parseIsoUtc(ts);
  if (!d) return 'never';
  const diffS = Math.max(0, (Date.now() - d.getTime()) / 1000);
  if (diffS < 45)        return 'just now';
  if (diffS < 90)        return 'a minute ago';
  if (diffS < 3600)      return `${Math.floor(diffS / 60)} minutes ago`;
  if (diffS < 5400)      return 'an hour ago';
  if (diffS < 86400)     return `${Math.floor(diffS / 3600)} hours ago`;
  if (diffS < 86400 * 2) return 'yesterday';
  if (diffS < 86400 * 7) return `${Math.floor(diffS / 86400)} days ago`;
  if (diffS < 86400 * 60) return `${Math.floor(diffS / 86400 / 7)} weeks ago`;
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function chronicleDate(ts) {
  const d = parseIsoUtc(ts);
  if (!d) return '—';
  // "Tue 18:42" within the past week, full date otherwise
  const diffS = (Date.now() - d.getTime()) / 1000;
  if (diffS < 86400 * 6) {
    return d.toLocaleString(undefined, {
      weekday: 'short',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).toLowerCase();
  }
  return d.toLocaleDateString(undefined, {
    month: 'short', day: '2-digit',
  }).toLowerCase();
}

function duration(startTs, endTs) {
  const s = parseIsoUtc(startTs);
  const e = parseIsoUtc(endTs);
  if (!s || !e) return '';
  const secs = Math.max(0, (e.getTime() - s.getTime()) / 1000);
  if (secs < 60)    return `${Math.round(secs)} seconds`;
  if (secs < 90)    return `a minute`;
  if (secs < 3600)  return `${Math.round(secs / 60)} minutes`;
  if (secs < 5400)  return `an hour`;
  if (secs < 86400) {
    const h = Math.floor(secs / 3600);
    const m = Math.round((secs % 3600) / 60);
    return m ? `${h}h ${m}m` : `${h} hours`;
  }
  return `${Math.floor(secs / 86400)} days`;
}


/* ── actions: rename + revoke ──────────────────────────────── */

function wireStageActions(scope) {
  scope.querySelectorAll('[data-cs-action="rename"]').forEach(btn => {
    btn.addEventListener('click', () => openRename(btn.dataset.csId));
  });
  scope.querySelectorAll('[data-cs-action="revoke"]').forEach(btn => {
    btn.addEventListener('click', () => stagedRevoke(btn));
  });
}

function openRename(trustedId) {
  const t = state.trusted.find(x => x.id === trustedId);
  if (!t) return;
  state.renameTarget = { id: t.id, label: t.label || '' };
  els.renamePrev.textContent = t.label || 'Unnamed stage';
  els.renameInput.value = t.label || '';
  els.renameInput.placeholder = 'A worthy name';
  els.renameCard.classList.remove('hidden');
  els.renameCard.setAttribute('aria-hidden', 'false');
  setTimeout(() => {
    els.renameInput.focus();
    els.renameInput.select();
  }, 60);
}

function closeRename() {
  els.renameCard.classList.add('hidden');
  els.renameCard.setAttribute('aria-hidden', 'true');
  state.renameTarget = null;
}

els.renameCancel.addEventListener('click', closeRename);
els.renameCard.addEventListener('click', (e) => {
  // Click on the card's backdrop area to dismiss (clicks on the
  // inner card bubble; this catches only the wrapper itself).
  if (e.target === els.renameCard) closeRename();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !els.renameCard.classList.contains('hidden')) {
    closeRename();
  }
});

els.renameForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!state.renameTarget) return;
  const newLabel = (els.renameInput.value || '').trim();
  if (!newLabel || newLabel === state.renameTarget.label) {
    closeRename();
    return;
  }
  try {
    const res = await fetch(
      `/api/cast/trusted-receivers/${encodeURIComponent(state.renameTarget.id)}`,
      {
        method: 'PATCH',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: newLabel }),
      }
    );
    if (!res.ok) throw new Error(`status ${res.status}`);
    toast(`Renamed.`);
    closeRename();
    await loadAll();
  } catch (err) {
    toast(`Couldn't rename — ${err.message || err}`);
  }
});


/**
 * Two-step revoke without a modal dialog:
 *   - First click swaps the label to "tap again to confirm".
 *   - Second click within the window does the irreversible thing.
 *   - The first click also disarms any other pending revoke.
 * Quieter than a confirm() while still preventing slips.
 */
function stagedRevoke(btn) {
  const trustedId = btn.dataset.csId;
  if (!trustedId) return;

  const existing = state.pendingRevokes.get(trustedId);
  if (existing) {
    clearTimeout(existing);
    state.pendingRevokes.delete(trustedId);
    doRevoke(trustedId);
    return;
  }

  // Disarm any other pending revoke — only one armed at a time.
  for (const [otherId, h] of state.pendingRevokes) {
    clearTimeout(h);
    const otherBtn = document.querySelector(
      `[data-cs-action="revoke"][data-cs-id="${otherId}"]`
    );
    if (otherBtn) restoreRevokeButton(otherBtn);
  }
  state.pendingRevokes.clear();

  const original = btn.textContent;
  btn.dataset.csOriginal = original;
  btn.textContent = 'tap again to confirm';
  btn.classList.add('is-confirming');

  const h = setTimeout(() => {
    restoreRevokeButton(btn);
    state.pendingRevokes.delete(trustedId);
  }, REVOKE_CONFIRM_WINDOW_MS);
  state.pendingRevokes.set(trustedId, h);
}

function restoreRevokeButton(btn) {
  btn.classList.remove('is-confirming');
  btn.textContent = btn.dataset.csOriginal || 'revoke';
  delete btn.dataset.csOriginal;
}

async function doRevoke(trustedId) {
  try {
    const res = await fetch(
      `/api/cast/trusted-receivers/${encodeURIComponent(trustedId)}/revoke`,
      { method: 'POST', credentials: 'same-origin' }
    );
    if (!res.ok) throw new Error(`status ${res.status}`);
    toast('Trust severed.');
    await loadAll();
  } catch (err) {
    toast(`Couldn't revoke — ${err.message || err}`);
  }
}


/* ── toast ─────────────────────────────────────────────────── */

let _toastTimer = null;
function toast(text) {
  els.toast.textContent = text;
  els.toast.classList.remove('hidden');
  // Re-trigger CSS entrance animation
  els.toast.style.animation = 'none';
  // eslint-disable-next-line no-unused-expressions
  els.toast.offsetWidth;
  els.toast.style.animation = '';
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => els.toast.classList.add('hidden'), TOAST_MS);
}


/* ── polling lifecycle ─────────────────────────────────────── */

function startPolling() {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(async () => {
    await loadAll().catch(() => {});
    startPolling();
  }, POLL_MS);
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) loadAll().catch(() => {});
});


/* ── boot ──────────────────────────────────────────────────── */

(async () => {
  try {
    await loadAll();
  } catch (err) {
    els.activeBody.innerHTML = `
      <p class="empty">
        Couldn't read the trust ledger. Are you signed in?
        <a href="/login.html">Open the sign-in page.</a>
      </p>`;
    els.chronicleBody.innerHTML = '';
  }
  startPolling();
})();
