/**
 * media-servers.js — Connected Devices panel (media sources + receivers).
 *
 * Two sections in one overlay:
 *
 *   1. Media Sources — per-user Audiobookshelf / Emby / Jellyfin / etc.
 *      credentials and catalog syncs (the original concern of this file).
 *      On first open we silently probe default ports on host.docker.internal
 *      + 127.0.0.1; anything found surfaces as an in-panel banner.
 *
 *   2. Receivers (TVs and Speakers) — devices the substrate can target
 *      with capability invocations (DLNA, Cast, AirPlay, Hue, etc).
 *      Rendered by `connected-devices.js`; this module just embeds it.
 *
 * One overlay, two concerns. "Things that supply media" + "Things that
 * play it." The user mental model that the device substrate spec calls
 * "Connected Devices" — see `docs/superpowers/specs/2026-05-07-device-
 * substrate-design.md`.
 */
import { escapeHtml, showToast } from './app.js';
import { openLibrivoxBrowse } from './librivox-browse.js';
import { renderDevicesSection } from './connected-devices.js';

let _overlay = null;
let _servers = [];
let _defaults = {
  audiobookshelf: 13378, emby: 8096, jellyfin: 8096,
  komga: 25600, kavita: 5000, suwayomi: 4567,
};
let _detected = [];
let _dirty = false;
// Tracked separately from `isAdmin()` in auth.js so a server response
// that omits the field (older deployments / offline cache) doesn't
// blow up the panel. The list response is the source of truth — it
// reflects whether THIS user's request is being served with admin
// privileges, including any role flips since page load.
let _viewerIsAdmin = false;
const _syncJobPollers = new Map();

const PROVIDERS = [
  {
    id: 'audiobookshelf',
    label: 'Audiobookshelf',
    default_port: 13378,
    blurb: 'Audiobooks and podcasts, streamed from your ABS server.',
    allow_no_auth: false,
  },
  {
    id: 'emby',
    label: 'Emby',
    default_port: 8096,
    blurb: 'Movies, shows, and other home-media libraries from an Emby server.',
    allow_no_auth: false,
  },
  {
    id: 'jellyfin',
    label: 'Jellyfin',
    default_port: 8096,
    blurb: 'Movies, shows, and music videos from Jellyfin, normalized into Files > Cloud.',
    allow_no_auth: false,
  },
  {
    id: 'komga',
    label: 'Komga',
    default_port: 25600,
    blurb: 'Self-hosted comic / manga server. HTTP Basic or API key.',
    allow_no_auth: false,
  },
  {
    id: 'suwayomi',
    label: 'Suwayomi',
    default_port: 4567,
    blurb: 'Tachiyomi-extension bridge: your curated manga library plus live-fetched chapters from sources like MangaDex.',
    // Most users run Suwayomi locally with auth disabled. Keep creds
    // optional so the happy path is "enter URL, click Connect".
    allow_no_auth: true,
  },
  // Emby/Jellyfin/Kavita/OPDS land in follow-up phases — placeholders
  // kept out of the UI until their provider clients are wired server-side.
];

export async function openMediaServers() {
  if (!_overlay) _buildOverlay();
  _overlay.classList.add('visible');
  document.body.classList.add('ms-lock-scroll');
  await _refresh({ runDetect: true });
}

export function closeMediaServers() {
  if (!_overlay) return;
  _overlay.classList.remove('visible');
  document.body.classList.remove('ms-lock-scroll');
  // Drop any confirmed Augmentum passwords held for open access panels —
  // never keep step-up credentials in memory past the panel's lifetime.
  _clearStepUp();
  // If the user ran a sync or added a server, the Files panel may be
  // showing a stale list — the caller refreshes on its next open.
  if (_dirty) {
    window.dispatchEvent(new CustomEvent('media-servers:changed'));
    _dirty = false;
  }
}

// --- Data ------------------------------------------------------------

async function _refresh({ runDetect = false } = {}) {
  const listResp = await fetch('/api/media/servers').catch(() => null);
  if (listResp && listResp.ok) {
    const data = await listResp.json();
    _servers = data.servers || [];
    _defaults = data.defaults || _defaults;
    _viewerIsAdmin = !!data.viewer_is_admin;
    const librariesByServer = await Promise.all(_servers.map(async (server) => {
      const resp = await fetch(`/api/media/servers/${encodeURIComponent(server.id)}/libraries`).catch(() => null);
      if (!resp || !resp.ok) return [];
      const payload = await resp.json().catch(() => ({}));
      return payload.libraries || [];
    }));
    _servers = _servers.map((server, idx) => ({
      ...server,
      libraries: librariesByServer[idx] || [],
    }));
  } else {
    _servers = [];
    _viewerIsAdmin = false;
  }
  if (runDetect) {
    const detResp = await fetch('/api/media/detect', { method: 'POST' }).catch(() => null);
    _detected = (detResp && detResp.ok) ? (await detResp.json()).detected || [] : [];
  }
  _render();
}

async function _addServer(payload) {
  let resp, body;
  try {
    resp = await fetch('/api/media/servers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    body = await resp.json().catch(() => ({}));
  } catch (err) {
    showToast(`Network error: ${err.message || 'could not reach server'}`, 'error', 4000);
    return false;
  }
  if (!resp.ok) {
    showToast(body.error || 'Failed to add server', 'error', 4000);
    return false;
  }
  showToast(`Connected to ${payload.name}`, 'success', 3000);
  _dirty = true;
  await _refresh({ runDetect: true });
  // Auto-kick a first sync so the Files panel has content right away.
  if (body.server?.id) {
    _syncServer(body.server.id, { silent: true });
  }
  return true;
}

async function _syncServer(id, { silent = false } = {}) {
  const row = _servers.find(s => s.id === id);
  if (row) { row.status = 'syncing'; row.status_detail = 'Syncing…'; _render(); }
  let resp, body;
  try {
    resp = await fetch(`/api/media/servers/${encodeURIComponent(id)}/sync`, { method: 'POST' });
    body = await resp.json().catch(() => ({}));
  } catch (err) {
    if (!silent) showToast(`Sync failed: ${err.message || 'network error'}`, 'error', 4000);
    await _refresh();
    return;
  }
  if (resp.ok && body.job_id) {
    if (!silent) showToast('Sync started', 'success', 2000);
    _pollSyncJob(id, body.job_id, { silent });
    return;
  }
  if (!resp.ok) {
    if (!silent) showToast(body.error || 'Sync failed', 'error', 4000);
  } else if (!silent) {
    showToast(`Indexed ${body.indexed ?? 0} items`, 'success', 3000);
  }
  _dirty = true;
  await _refresh();
}

function _pollSyncJob(serverId, jobId, { silent = false } = {}) {
  if (!jobId) return;
  _syncJobPollers.set(serverId, jobId);
  const INTERVAL_MS = 1500;

  const tick = async () => {
    if (_syncJobPollers.get(serverId) !== jobId) return;
    let resp, body;
    try {
      resp = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
      body = await resp.json().catch(() => ({}));
    } catch {
      setTimeout(tick, INTERVAL_MS);
      return;
    }

    if (!resp.ok) {
      // Job row can briefly lag right after enqueue on slower disks.
      setTimeout(tick, INTERVAL_MS);
      return;
    }

    const row = _servers.find(s => s.id === serverId);
    if (row && (body.status === 'pending' || body.status === 'running')) {
      row.status = 'syncing';
      row.status_detail = body.stage || 'Syncing…';
      _render();
    }

    if (body.status === 'completed') {
      _syncJobPollers.delete(serverId);
      _dirty = true;
      await _refresh();
      if (!silent) showToast(`Indexed ${body.result?.indexed ?? 0} items`, 'success', 3000);
      return;
    }

    if (body.status === 'failed' || body.status === 'cancelled') {
      _syncJobPollers.delete(serverId);
      _dirty = true;
      await _refresh();
      if (!silent) showToast(body.error || 'Sync failed', 'error', 4000);
      return;
    }

    setTimeout(tick, INTERVAL_MS);
  };

  setTimeout(tick, INTERVAL_MS);
}

async function _testServer(id) {
  let resp, body;
  try {
    resp = await fetch(`/api/media/servers/${encodeURIComponent(id)}/test`, { method: 'POST' });
    body = await resp.json().catch(() => ({}));
  } catch (err) {
    showToast(`Test failed: ${err.message || 'network error'}`, 'error', 4000);
    return;
  }
  if (!resp.ok || body.status !== 'ok') {
    showToast(body.detail || body.error || 'Test failed', 'error', 4000);
  } else {
    showToast('Connection OK', 'success', 2500);
  }
  await _refresh();
}

async function _toggleShareServer(id) {
  const row = _servers.find(s => s.id === id);
  if (!row) return;
  // Only admins can flip scope, and only on rows they own. The button
  // is hidden in those cases too — this is a belt-and-braces guard
  // so a stray click via DOM inspector doesn't silently 403.
  if (!_viewerIsAdmin || !row.is_owned_by_viewer) {
    showToast('Only the admin who connected this server can share it', 'warning', 4000);
    return;
  }
  const nextScope = row.is_shared ? 'private' : 'shared';
  const verb = nextScope === 'shared' ? 'Share' : 'Un-share';
  const blurb = nextScope === 'shared'
    ? (
      `Share "${row.name}" with every user on this box? They'll be able ` +
      `to browse and stream from it, but they can't edit the URL, ` +
      `token, or name. They won't see your watch history.`
    )
    : (
      `Un-share "${row.name}"? Other users will no longer see this ` +
      `connection. Their already-synced catalog entries stay until ` +
      `they remove the server from their own Files panel.`
    );
  if (!confirm(blurb)) return;
  let resp, body;
  try {
    resp = await fetch(`/api/media/servers/${encodeURIComponent(id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scope: nextScope }),
    });
    body = await resp.json().catch(() => ({}));
  } catch (err) {
    showToast(`${verb} failed: ${err.message || 'network error'}`, 'error', 4000);
    return;
  }
  if (!resp.ok) {
    showToast(body.error || `${verb} failed`, 'error', 4000);
    return;
  }
  showToast(
    nextScope === 'shared'
      ? `"${row.name}" is now visible to all users`
      : `"${row.name}" is back to private`,
    'success', 3000,
  );
  _dirty = true;
  await _refresh();
}

async function _removeServer(id) {
  const row = _servers.find(s => s.id === id);
  if (!row) return;
  if (!confirm(`Remove "${row.name}"? Synced catalog stays until you clear Files.`)) return;
  let resp;
  try {
    resp = await fetch(`/api/media/servers/${encodeURIComponent(id)}`, { method: 'DELETE' });
  } catch (err) {
    showToast(`Remove failed: ${err.message || 'network error'}`, 'error');
    return;
  }
  if (!resp.ok) {
    showToast('Failed to remove', 'error');
    return;
  }
  _dirty = true;
  await _refresh();
}

async function _updateLibrary(serverId, libraryId, payload) {
  let resp, body;
  try {
    resp = await fetch(
      `/api/media/servers/${encodeURIComponent(serverId)}/libraries/${encodeURIComponent(libraryId)}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      },
    );
    body = await resp.json().catch(() => ({}));
  } catch (err) {
    showToast(`Library update failed: ${err.message || 'network error'}`, 'error', 4000);
    return false;
  }
  if (!resp.ok) {
    showToast(body.error || 'Failed to update library', 'error', 4000);
    return false;
  }
  _dirty = true;
  await _refresh();
  return true;
}

// --- Rendering -------------------------------------------------------

function _buildOverlay() {
  _overlay = document.createElement('div');
  _overlay.className = 'ms-overlay';
  _overlay.innerHTML = `
    <div class="ms-panel" role="dialog" aria-modal="true" aria-label="Connected Devices">
      <div class="ms-header">
        <div>
          <div class="ms-title">Connected Devices</div>
          <div class="ms-subtitle">Your media libraries plus the TVs, speakers, and screens to play them on.</div>
        </div>
        <button class="ms-close" title="Close (Esc)" aria-label="Close">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
        </button>
      </div>
      <div class="ms-body" id="ms-body"></div>
    </div>
  `;
  document.body.appendChild(_overlay);

  _overlay.addEventListener('click', (e) => {
    if (e.target === _overlay) closeMediaServers();
  });
  _overlay.querySelector('.ms-close').addEventListener('click', closeMediaServers);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && _overlay?.classList.contains('visible')) {
      closeMediaServers();
    }
  });
}

function _render() {
  const body = _overlay?.querySelector('#ms-body');
  if (!body) return;

  // Filter detected entries down to providers we actually support AND
  // the user hasn't already added. Stays silent (empty) when nothing
  // interesting showed up.
  const supported = new Set(PROVIDERS.map(p => p.id));
  const fresh = (_detected || []).filter(d => supported.has(d.provider) && !d.already_added);

  body.innerHTML = `
    <div class="cd-self-actions">
      <span class="cd-self-actions-label">On this device:</span>
      <a class="cd-self-link" href="/ui/cast-receiver/" target="_blank" rel="noopener" title="Open this browser as a cast target — accepts a pairing code">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="13" rx="2"/><path d="M8 21h8M12 17v4"/></svg>
        <span>Use as receiver</span>
      </a>
      <a class="cd-self-link" href="/ui/cast-control/" target="_blank" rel="noopener" title="Open this browser as a remote control for another screen">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8a4 4 0 0 1 4-4h8a4 4 0 0 1 4 4v8a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4z"/><path d="M9 10h.01M15 10h.01M8 14h8"/></svg>
        <span>Use as remote</span>
      </a>
    </div>
    <section class="cd-section">
      <header class="cd-section-head">
        <div class="cd-section-titles">
          <h3 class="cd-section-title">Where your media comes from</h3>
          <p class="cd-section-sub">Connect existing libraries — Audiobookshelf, Emby, Jellyfin — so Augmentum can play and recommend from them.</p>
        </div>
      </header>
      ${_detectBannerHtml(fresh)}
      ${_builtinLibraryCardHtml()}
      ${_serverListHtml(_servers)}
      ${_addFormHtml()}
    </section>
    <div data-cd-host></div>
  `;

  // Mount the devices (receivers) section. Owns its own data fetch
  // and event wiring; we only hand it a container and a callback.
  const devicesHost = body.querySelector('[data-cd-host]');
  if (devicesHost) {
    renderDevicesSection(devicesHost, {
      onChanged: () => { _dirty = true; },
    }).catch(err => console.warn('[connected-devices] mount failed:', err));
  }

  // Built-in library card — always visible, no test/delete/sync.
  const browseLvBtn = body.querySelector('[data-builtin-browse="librivox"]');
  browseLvBtn?.addEventListener('click', () => openLibrivoxBrowse());

  // Wire dynamic controls.
  fresh.forEach(d => {
    const btn = body.querySelector(`[data-connect-detected="${CSS.escape(d.base_url)}"]`);
    btn?.addEventListener('click', () => _promptConnectDetected(d));
  });
  body.querySelectorAll('[data-dismiss-detected]').forEach(btn => {
    btn.addEventListener('click', () => {
      _detected = [];
      _render();
    });
  });
  body.querySelectorAll('[data-sync-id]').forEach(btn => {
    btn.addEventListener('click', () => _syncServer(btn.dataset.syncId));
  });
  body.querySelectorAll('[data-test-id]').forEach(btn => {
    btn.addEventListener('click', () => _testServer(btn.dataset.testId));
  });
  body.querySelectorAll('[data-remove-id]').forEach(btn => {
    btn.addEventListener('click', () => _removeServer(btn.dataset.removeId));
  });
  body.querySelectorAll('[data-share-id]').forEach(btn => {
    btn.addEventListener('click', () => _toggleShareServer(btn.dataset.shareId));
  });
  body.querySelectorAll('[data-access-id]').forEach(btn => {
    btn.addEventListener('click', () => _toggleAccessPanel(btn.dataset.accessId));
  });
  body.querySelectorAll('[data-open-gate]').forEach(btn => {
    btn.addEventListener('click', () => window.open(btn.dataset.openGate, '_blank', 'noopener'));
  });
  body.querySelectorAll('[data-library-id]').forEach(row => {
    _wireLibraryRow(row);
  });
  body.querySelectorAll('[data-show-skipped]').forEach(btn => {
    btn.addEventListener('click', () => {
      const server = _servers.find(s => s.id === btn.dataset.showSkipped);
      if (server) _showSkippedModal(server);
    });
  });
  const form = body.querySelector('#ms-add-form');
  form?.addEventListener('submit', (e) => {
    e.preventDefault();
    _submitAddForm(form);
  });

  // Keep the form responsive to provider selection:
  //   - port default updates unless the user has typed a custom one
  //   - blurb text reflects the picked provider
  //   - username/password/token placeholders adapt for no-auth providers
  //     so Suwayomi users see "Leave blank for local no-auth setup"
  //     instead of the generic required-looking field.
  const providerSel = body.querySelector('#ms-provider');
  const portInput = body.querySelector('#ms-port');
  const blurbEl = body.querySelector('#ms-provider-blurb');
  const userInput = body.querySelector('#ms-user');
  const passInput = body.querySelector('#ms-pass');
  const tokenInput = body.querySelector('#ms-token');

  const _syncProviderFields = () => {
    const p = PROVIDERS.find(x => x.id === providerSel.value) || PROVIDERS[0];
    const def = _defaults[p.id] || p.default_port || '';
    if (portInput && (!portInput.value || portInput.dataset.pristine === '1')) {
      portInput.value = String(def || '');
    }
    if (blurbEl) blurbEl.textContent = p.blurb || '';
    if (userInput) {
      userInput.placeholder = p.allow_no_auth
        ? 'Optional for local no-auth setup'
        : '';
    }
    if (passInput) {
      passInput.placeholder = p.allow_no_auth
        ? 'Optional'
        : '';
    }
    if (tokenInput) {
      tokenInput.placeholder = p.allow_no_auth
        ? 'Not needed for no-auth deployments'
        : 'Leave empty to use username + password';
    }
  };
  providerSel?.addEventListener('change', _syncProviderFields);
  portInput?.addEventListener('input', () => { portInput.dataset.pristine = '0'; });
  // Run once on mount so the initial selection's blurb + placeholders
  // populate without waiting for the first `change` event.
  _syncProviderFields();
}

// LibriVox tile — rendered above the user's connected servers so every
// user sees a free library option before they bother pointing at their
// own box. No credentials, no test button, no delete; the browse overlay
// owns all interaction with this catalog.
function _builtinLibraryCardHtml() {
  return `
    <div class="ms-section-title">Free public-domain library</div>
    <div class="ms-builtin-card">
      <div class="ms-builtin-icon" aria-hidden="true">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 19a2 2 0 0 1 2-2h14"/>
          <path d="M5 5a2 2 0 0 0-2 2v12l3-1h13V5H5z"/>
          <path d="M9 9h8M9 13h6"/>
        </svg>
      </div>
      <div class="ms-builtin-main">
        <div class="ms-builtin-title">
          LibriVox
          <span class="ms-builtin-pill">Included</span>
        </div>
        <div class="ms-builtin-sub">~20,000 free audiobooks, read by volunteers. Streams instantly, no sign-up.</div>
      </div>
      <div class="ms-builtin-action">
        <button class="btn btn-sm btn-primary" data-builtin-browse="librivox">
          Browse catalog →
        </button>
      </div>
    </div>
  `;
}


function _detectBannerHtml(fresh) {
  if (!fresh.length) return '';
  const intro = fresh.length === 1
    ? `We found a server running on this machine.`
    : `We found ${fresh.length} servers running on this machine.`;
  return `
    <div class="ms-banner" role="region" aria-label="Detected servers">
      <div class="ms-banner-head">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l4 4L19 7"/></svg>
        <span>${escapeHtml(intro)}</span>
        <button class="ms-banner-dismiss" data-dismiss-detected aria-label="Dismiss" title="Not now">&times;</button>
      </div>
      <ul class="ms-banner-list">
        ${fresh.map(d => `
          <li>
            <span class="ms-banner-label">
              <strong>${escapeHtml(_providerLabel(d.provider))}</strong>
              <span class="ms-banner-url">${escapeHtml(d.base_url)}</span>
              ${d.is_initialized === false ? '<span class="ms-badge warn" title="This server hasn\'t been set up yet">Not initialized</span>' : ''}
            </span>
            <button class="btn btn-sm btn-primary" data-connect-detected="${escapeHtml(d.base_url)}">Connect</button>
          </li>
        `).join('')}
      </ul>
    </div>
  `;
}

function _serverListHtml(servers) {
  if (!servers.length) {
    return `
      <div class="ms-empty">
        <div class="ms-empty-icon" aria-hidden="true">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/></svg>
        </div>
        <div class="ms-empty-title">Bring your own library</div>
        <div class="ms-empty-sub">
          Connect an Audiobookshelf, Emby, or Jellyfin server and your books and shows land in the Files panel —
          streamed from your server, indexed by Augmentum, searchable next to everything else.
        </div>
      </div>
    `;
  }
  return `
    <div class="ms-section-title">Connected</div>
    <ul class="ms-server-list">
      ${servers.map(s => _serverRowHtml(s)).join('')}
    </ul>
  `;
}

function _statusLabel(status) {
  switch (status) {
    case 'ok':       return 'Connected';
    case 'error':    return 'Error';
    case 'syncing':  return 'Syncing…';
    case 'untested': return 'Not tested';
    default:         return status;
  }
}

function _relativeTime(iso) {
  if (!iso) return 'never';
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t)) return 'never';
  const diff = Date.now() - t;
  const mins = Math.round(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins} min${mins === 1 ? '' : 's'} ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs} hr${hrs === 1 ? '' : 's'} ago`;
  const days = Math.round(hrs / 24);
  if (days < 14) return `${days} day${days === 1 ? '' : 's'} ago`;
  return new Date(iso).toLocaleDateString();
}

function _serverRowHtml(s) {
  const statusClass = s.status === 'ok' ? 'ok'
                    : s.status === 'error' ? 'err'
                    : s.status === 'syncing' ? 'syncing'
                    : 'idle';
  const synced = s.last_sync_at ? _relativeTime(s.last_sync_at) : 'never';
  const items = `${s.item_count} ${s.item_count === 1 ? 'item' : 'items'}`;
  // Diagnostic chip — only render when a sync has actually run (total_seen > 0).
  // Clickable when skipped > 0 so users can see exactly which titles we dropped.
  const skipped = Number(s.skipped_count) || 0;
  const totalSeen = Number(s.total_seen) || 0;
  let diagBlock = '';
  if (totalSeen > 0) {
    if (skipped === 0) {
      diagBlock = `<span class="ms-diag ms-diag-ok" title="Every book in the library was indexed">All ${totalSeen} indexed</span>`;
    } else {
      diagBlock = `<button type="button" class="ms-diag ms-diag-warn" data-show-skipped="${escapeHtml(s.id)}"
                    title="Click to see which titles didn't sync">
        ${skipped} skipped of ${totalSeen}
      </button>`;
    }
  }
  // Sharing affordances. Three states:
  //   1. Admin owns + already shared → "Shared with everyone" badge + Un-share button
  //   2. Admin owns + not shared      → "Private" (no badge) + Share button
  //   3. Non-owner viewing a shared row → "Shared by admin" badge, read-only
  //      (no Share button, no Test, no Remove — sync is still useful so they
  //       can populate their own catalog).
  const isShared = !!s.is_shared;
  const isOwned = s.is_owned_by_viewer !== false;
  let shareBadge = '';
  if (isShared && isOwned) {
    shareBadge = `<span class="ms-badge ms-badge-shared" title="Visible to every user on this box">Shared</span>`;
  } else if (isShared && !isOwned) {
    shareBadge = `<span class="ms-badge ms-badge-shared" title="An admin shared this connection — read-only for you">Shared by admin</span>`;
  }
  const canShare = _viewerIsAdmin && isOwned;
  const shareBtn = canShare
    ? `<button class="btn btn-sm" data-share-id="${escapeHtml(s.id)}" title="${isShared ? 'Stop sharing with other users' : 'Make this server visible to every user'}">${isShared ? 'Un-share' : 'Share'}</button>`
    : '';
  // Non-owners of a shared server only get Sync. Test would try to
  // overwrite the admin's status row (silently no-ops, but still
  // misleading), and Remove is explicitly rejected server-side.
  const testBtn = isOwned
    ? `<button class="btn btn-sm" data-test-id="${escapeHtml(s.id)}" title="Verify credentials">Test</button>`
    : '';
  const removeBtn = isOwned
    ? `<button class="btn btn-sm ms-danger" data-remove-id="${escapeHtml(s.id)}" title="Disconnect server">Remove</button>`
    : '';
  // Admin-only AND managed-instance-only: reveal the managed login + access
  // URLs at any time (the password is deterministic, so it's never truly
  // lost). Manually-connected external servers get no panel — Augmentum
  // doesn't manage their login, and the panel's URLs (built from this
  // host's address + the managed instance's ports) would be wrong for them.
  const accessBtn = _viewerIsAdmin && s.is_managed_instance
    ? `<button class="btn btn-sm" data-access-id="${escapeHtml(s.id)}" aria-expanded="false" aria-controls="ms-access-${escapeHtml(s.id)}" title="Show login & access URLs">Access &amp; login</button>`
    : '';
  // Front gate: dissolved-login "Open" for any user with access. Present only
  // when the gate is configured AND the server is gate-eligible (the server
  // list response attaches gate_url). Logging into the server is handled by
  // Augmentum's session — no password prompt.
  const openSignedInBtn = s.gate_url
    ? `<button class="btn btn-sm btn-primary" data-open-gate="${escapeHtml(s.gate_url)}" aria-label="Open ${escapeHtml(s.name)} signed in (new tab)" title="Open signed in — Augmentum trusts your session">Open ▸</button>`
    : '';
  return `
    <li class="ms-server${isShared && !isOwned ? ' ms-server-readonly' : ''}">
      <div class="ms-server-main">
        <div class="ms-server-name">
          <strong>${escapeHtml(s.name)}</strong>
          <span class="ms-server-type">${escapeHtml(_providerLabel(s.provider))}</span>
          <span class="ms-badge ${statusClass}">${escapeHtml(_statusLabel(s.status))}</span>
          ${shareBadge}
        </div>
        <div class="ms-server-url" title="${escapeHtml(s.base_url)}">${escapeHtml(s.base_url)}</div>
        <div class="ms-server-meta">
          <span>${items}</span>
          <span aria-hidden="true">&middot;</span>
          <span title="${s.last_sync_at ? escapeHtml(new Date(s.last_sync_at).toLocaleString()) : ''}">
            Last synced ${escapeHtml(synced)}
          </span>
          ${diagBlock ? `<span aria-hidden="true">&middot;</span>${diagBlock}` : ''}
          ${s.status_detail ? `<span class="ms-detail">&middot; ${escapeHtml(s.status_detail)}</span>` : ''}
        </div>
        ${_serverLibrariesHtml(s)}
      </div>
      <div class="ms-server-actions">
        ${openSignedInBtn}
        ${accessBtn}
        ${testBtn}
        <button class="btn btn-sm btn-primary" data-sync-id="${escapeHtml(s.id)}" title="Re-scan library">Sync</button>
        ${shareBtn}
        ${removeBtn}
      </div>
      ${accessBtn ? `<div class="ms-access-panel" id="ms-access-${escapeHtml(s.id)}" data-access-panel="${escapeHtml(s.id)}" role="region" aria-label="Access & login for ${escapeHtml(s.name)}" hidden></div>` : ''}
    </li>
  `;
}

// ── Access & login panel ────────────────────────────────────────────
//
// Admin-only AND step-up: revealing or changing the managed credential
// re-verifies the operator's OWN Augmentum password (an unattended or
// hijacked admin session shouldn't silently surface/rotate a server
// login). The panel opens to a "confirm your Augmentum password" prompt;
// a successful unlock reveals three ways to reach the server:
//   1. In a browser / iframe over real HTTPS (the dedicated front door).
//   2. From native TV/phone apps over the raw LAN host port.
//   3. The managed login (recover any time; change it to your own).
// We hold the confirmed password in memory only while the panel is open
// (so a change doesn't need a second prompt) and drop it on close. The
// managed credential is deterministic, so recovery is "confirm it's you,
// then read it again".

const _stepUpPw = new Map();   // serverId → confirmed Augmentum password (panel lifetime)

function _clearStepUp(serverId) {
  if (serverId == null) _stepUpPw.clear();
  else _stepUpPw.delete(serverId);
}

async function _toggleAccessPanel(serverId) {
  const panel = _overlay?.querySelector(
    `[data-access-panel="${(window.CSS && CSS.escape) ? CSS.escape(serverId) : serverId}"]`,
  );
  if (!panel) return;
  const btn = _overlay?.querySelector(
    `[data-access-id="${(window.CSS && CSS.escape) ? CSS.escape(serverId) : serverId}"]`,
  );
  if (!panel.hidden) {
    panel.hidden = true;
    panel.innerHTML = '';
    btn?.setAttribute('aria-expanded', 'false');
    _clearStepUp(serverId);
    return;
  }
  panel.hidden = false;
  btn?.setAttribute('aria-expanded', 'true');
  _renderUnlockPrompt(panel, serverId);
}

function _renderUnlockPrompt(panel, serverId, errMsg) {
  const pwId = `ms-access-pw-${serverId}`;
  panel.innerHTML = `
    <div class="ms-access-inner">
      <div class="ms-access-group-title">🔒 Confirm it's you</div>
      <div class="ms-access-row">
        <label class="ms-access-label" for="${escapeHtml(pwId)}">Augmentum password</label>
        <input type="password" id="${escapeHtml(pwId)}" class="ms-access-pw"
               autocomplete="current-password" aria-label="Augmentum password"
               placeholder="Your Augmentum login" />
        <button type="button" class="btn btn-sm btn-primary" data-unlock>Unlock</button>
      </div>
      <div class="ms-access-sub">Required to reveal or change this server's login.</div>
      ${errMsg ? `<div class="ms-access-err" role="alert">${escapeHtml(errMsg)}</div>` : ''}
    </div>`;
  const input = panel.querySelector('.ms-access-pw');
  const submit = () => _unlockAccessPanel(serverId, panel, input.value);
  panel.querySelector('[data-unlock]')?.addEventListener('click', submit);
  input?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); submit(); }
  });
  input?.focus();
}

async function _unlockAccessPanel(serverId, panel, password) {
  if (!password) {
    _renderUnlockPrompt(panel, serverId, 'Enter your Augmentum password.');
    return;
  }
  panel.innerHTML = '<div class="ms-access-loading" role="status">Verifying…</div>';
  try {
    const r = await fetch(
      `/api/media/servers/${encodeURIComponent(serverId)}/access`,
      {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ augmentum_password: password }),
      });
    const d = await r.json().catch(() => ({}));
    if (r.ok) {
      _stepUpPw.set(serverId, password);
      _renderAccessPanel(panel, serverId, d);
    } else {
      _renderUnlockPrompt(panel, serverId,
        d.error || d.detail || `Verification failed (${r.status})`);
    }
  } catch (err) {
    _renderUnlockPrompt(panel, serverId, String(err.message || err));
  }
}

function _renderAccessPanel(panel, serverId, d) {
  const host = location.hostname;
  const httpsUrl = d.https_port ? `https://${host}:${d.https_port}` : '';
  const rawUrl = d.raw_host_port ? `http://${host}:${d.raw_host_port}` : '';

  const linkRow = (label, url, sub, openable) => {
    if (!url) return '';
    return `
      <div class="ms-access-row">
        <span class="ms-access-label">${escapeHtml(label)}</span>
        <code class="ms-access-val">${escapeHtml(url)}</code>
        ${openable ? `<button type="button" class="btn btn-sm" data-open="${escapeHtml(url)}" aria-label="Open ${escapeHtml(url)} in a new tab">Open</button>` : ''}
        <button type="button" class="btn btn-sm" data-copy="${escapeHtml(url)}" aria-label="Copy ${escapeHtml(label)} URL" aria-live="polite">Copy</button>
        ${sub ? `<span class="ms-access-sub">${escapeHtml(sub)}</span>` : ''}
      </div>`;
  };
  const credRow = (label, value) => `
    <div class="ms-access-row">
      <span class="ms-access-label">${escapeHtml(label)}</span>
      <code class="ms-access-val">${escapeHtml(value)}</code>
      <button type="button" class="btn btn-sm" data-copy="${escapeHtml(value)}" aria-label="Copy ${escapeHtml(label)}" aria-live="polite">Copy</button>
    </div>`;

  let html = '<div class="ms-access-inner">';
  if (httpsUrl || rawUrl) {
    html += '<div class="ms-access-group-title">Open the server</div>';
    html += linkRow('Browser', httpsUrl,
      'Real HTTPS — opens in this browser', true);
    html += linkRow('Apps / TV', rawUrl,
      'Point the official app at this address on your network', false);
  }
  if (d.managed_auth) {
    html += '<div class="ms-access-group-title">🔒 Login (managed by Augmentum)</div>';
    html += credRow('User', d.username || '');
    html += credRow('Pass', d.password || '');
    if (d.can_change_password) {
      html += `<div class="ms-access-row">
        <button type="button" class="btn btn-sm" data-change-pass="1">Change password…</button>
        <span class="ms-access-sub">Set a memorable login for your apps</span>
      </div>`;
    }
  } else if (!httpsUrl && !rawUrl) {
    html += '<div class="ms-access-sub">No managed access info for this connection.</div>';
  }
  html += '</div>';
  panel.innerHTML = html;

  panel.querySelectorAll('[data-open]').forEach(b => {
    b.addEventListener('click', () => window.open(b.dataset.open, '_blank', 'noopener'));
  });
  panel.querySelectorAll('[data-copy]').forEach(b => {
    b.addEventListener('click', () => {
      try { navigator.clipboard?.writeText(b.dataset.copy || ''); } catch { /* no clipboard */ }
      const prev = b.textContent;
      b.textContent = 'Copied';
      setTimeout(() => { b.textContent = prev; }, 1200);
    });
  });
  panel.querySelector('[data-change-pass]')?.addEventListener('click',
    () => _changeServerPassword(serverId, panel));
}

async function _changeServerPassword(serverId, panel) {
  // The panel is already unlocked, so the operator's Augmentum password is
  // confirmed in memory — reuse it (no second prompt) to authorize the change.
  const stepUp = _stepUpPw.get(serverId);
  if (!stepUp) {
    _renderUnlockPrompt(panel, serverId, 'Please confirm your Augmentum password again.');
    return;
  }
  const pw = window.prompt('New password for this server’s login (at least 8 characters):');
  if (pw == null) return;
  if (pw.length < 8) { showToast('Password must be at least 8 characters.'); return; }
  try {
    const r = await fetch(
      `/api/media/servers/${encodeURIComponent(serverId)}/change-password`,
      {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_password: pw, augmentum_password: stepUp }),
      });
    const d = await r.json().catch(() => ({}));
    if (r.ok) {
      showToast(d.restarting
        ? 'Password changed — restarting the server to apply…'
        : 'Password changed.');
      if (panel) await _unlockAccessPanel(serverId, panel, stepUp);  // refresh reveal
    } else if (r.status === 401) {
      // Augmentum password no longer valid (e.g. changed mid-session) —
      // drop the cached value and re-prompt.
      _clearStepUp(serverId);
      _renderUnlockPrompt(panel, serverId,
        d.error || 'Augmentum password no longer valid — confirm again.');
    } else {
      showToast(d.error || d.detail || `Change failed (${r.status})`);
    }
  } catch (err) {
    showToast(String(err.message || err));
  }
}

function _serverLibrariesHtml(server) {
  const libraries = Array.isArray(server.libraries) ? server.libraries : [];
  if (!libraries.length) return '';
  return `
    <div class="ms-library-block">
      <div class="ms-library-title">Libraries</div>
      <div class="ms-library-list">
        ${libraries.map(lib => _libraryRowHtml(server.id, lib)).join('')}
      </div>
    </div>
  `;
}

function _libraryRowHtml(serverId, lib) {
  const surface = lib.surface_group || lib.detected_group || 'other';
  const confidence = Math.round(Number(lib.detection_confidence || 0) * 100);
  return `
    <div class="ms-library" data-server-id="${escapeHtml(serverId)}" data-library-id="${escapeHtml(lib.id)}">
      <div class="ms-library-head">
        <div class="ms-library-name">${escapeHtml(lib.display_name || lib.provider_name || 'Library')}</div>
        <div class="ms-library-badges">
          <span class="ms-badge">${escapeHtml(_surfaceGroupLabel(surface))}</span>
          <span class="ms-badge ${lib.needs_review ? 'warn' : 'ok'}">
            ${lib.needs_review ? 'Needs review' : `Auto ${confidence}%`}
          </span>
        </div>
      </div>
      <div class="ms-library-sub">
        ${escapeHtml(lib.provider_name || '')}
        ${lib.provider_collection_type ? ` · ${escapeHtml(lib.provider_collection_type)}` : ''}
      </div>
      <div class="ms-library-controls">
        <input type="text" class="ms-library-input" data-lib-display
               value="${escapeHtml(lib.display_name_override || '')}"
               placeholder="${escapeHtml(lib.provider_name || 'Display name')}">
        <select class="ms-library-select" data-lib-surface>
          <option value="" ${!lib.surface_group_override ? 'selected' : ''}>Auto (${escapeHtml(_surfaceGroupLabel(lib.detected_group || 'other'))})</option>
          <option value="shows" ${lib.surface_group_override === 'shows' ? 'selected' : ''}>Shows</option>
          <option value="movies" ${lib.surface_group_override === 'movies' ? 'selected' : ''}>Movies</option>
          <option value="music_videos" ${lib.surface_group_override === 'music_videos' ? 'selected' : ''}>Music videos</option>
          <option value="other" ${lib.surface_group_override === 'other' ? 'selected' : ''}>Other</option>
        </select>
        <label class="ms-library-toggle"><input type="checkbox" data-lib-hidden ${lib.is_hidden ? 'checked' : ''}> Hide</label>
        <label class="ms-library-toggle"><input type="checkbox" data-lib-search ${lib.include_in_search !== false ? 'checked' : ''}> Search</label>
        <label class="ms-library-toggle"><input type="checkbox" data-lib-overview ${lib.include_in_overview !== false ? 'checked' : ''}> Overview</label>
      </div>
    </div>
  `;
}

// Reasons come from the provider's _item_from_abs. Keep labels short
// and neutral — they appear in a dialog the user opens to understand
// why a batch of books isn't showing up.
const SKIP_REASON_LABELS = {
  no_audio_files:          'No audio files',
  folder_needs_detail_fetch: 'Folder-based book (needs per-item fetch)',
  hidden_library:          'Hidden library',
  unsupported_library_group: 'Deferred library type',
  unknown_shape:           'Unrecognised data shape',
  unknown:                 'Unknown',
};

function _showSkippedModal(server) {
  const items = Array.isArray(server.last_sync_skipped) ? server.last_sync_skipped : [];
  if (!items.length) {
    showToast('No skipped titles recorded on the last sync', 'info', 2500);
    return;
  }
  const groupedByReason = new Map();
  for (const it of items) {
    const key = it.reason || 'unknown';
    if (!groupedByReason.has(key)) groupedByReason.set(key, []);
    groupedByReason.get(key).push(it);
  }
  const overlay = document.createElement('div');
  overlay.className = 'ms-skipped-overlay';
  overlay.innerHTML = `
    <div class="ms-skipped-modal" role="dialog" aria-modal="true" aria-label="Skipped titles">
      <div class="ms-skipped-head">
        <div>
          <div class="ms-skipped-title">${server.skipped_count} titles not indexed</div>
          <div class="ms-skipped-sub">
            ${server.skipped_count === items.length
              ? `Showing all ${items.length}.`
              : `Showing first ${items.length} of ${server.skipped_count}.`}
          </div>
        </div>
        <button class="ms-close" data-skipped-close aria-label="Close">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
        </button>
      </div>
      <div class="ms-skipped-body">
        ${[...groupedByReason.entries()].map(([reason, list]) => `
          <section class="ms-skipped-group">
            <div class="ms-skipped-reason">
              ${escapeHtml(SKIP_REASON_LABELS[reason] || reason)}
              <span class="ms-skipped-count">${list.length}</span>
            </div>
            <ul class="ms-skipped-list">
              ${list.map(it => `
                <li>
                  <span class="ms-skipped-name">${escapeHtml(it.title || 'Untitled')}</span>
                  ${it.author ? `<span class="ms-skipped-author">${escapeHtml(it.author)}</span>` : ''}
                </li>
              `).join('')}
            </ul>
          </section>
        `).join('')}
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  const close = () => overlay.remove();
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });
  overlay.querySelector('[data-skipped-close]').addEventListener('click', close);
  document.addEventListener('keydown', function onEsc(e) {
    if (e.key === 'Escape') {
      close();
      document.removeEventListener('keydown', onEsc);
    }
  });
}

function _addFormHtml() {
  const providerOpts = PROVIDERS.map(p =>
    `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}</option>`
  ).join('');
  return `
    <div class="ms-section-title">Add a server</div>
    <form id="ms-add-form" class="ms-form">
      <div class="ms-field">
        <label>Type</label>
        <select id="ms-provider" required>${providerOpts}</select>
      </div>
      <p id="ms-provider-blurb" class="ms-provider-blurb"></p>
      <div class="ms-field">
        <label>Display name</label>
        <input type="text" id="ms-name" placeholder="Home Audiobookshelf" required autocomplete="off">
      </div>
      <div class="ms-field ms-field-host">
        <label>Host or URL</label>
        <input type="text" id="ms-host" placeholder="192.168.1.50 or http://abs.lan" required autocomplete="off">
      </div>
      <div class="ms-field ms-field-port">
        <label>Port</label>
        <input type="number" id="ms-port" min="1" max="65535" data-pristine="1" placeholder="${_defaults.audiobookshelf}">
      </div>
      <div class="ms-field">
        <label>Username</label>
        <input type="text" id="ms-user" autocomplete="off">
      </div>
      <div class="ms-field">
        <label>Password</label>
        <input type="password" id="ms-pass" autocomplete="new-password">
      </div>
      <div class="ms-field ms-field-token">
        <label>…or paste an API token</label>
        <input type="text" id="ms-token" autocomplete="off" placeholder="Leave empty to use username + password">
      </div>
      ${_viewerIsAdmin ? `
      <div class="ms-field ms-field-share">
        <label class="ms-share-toggle">
          <input type="checkbox" id="ms-scope-shared">
          <span>Share with all users</span>
        </label>
        <p class="ms-share-blurb">
          They'll see this connection read-only — same media, separate watch history. You can toggle this any time.
        </p>
      </div>` : ''}
      <div class="ms-form-actions">
        <button type="submit" class="btn btn-primary">Connect</button>
      </div>
      <p class="ms-form-hint">
        We connect from the Augmentum server, not your browser — your server URL can be private (192.168.x.x, Tailscale, etc.).
      </p>
    </form>
  `;
}

// --- Actions ---------------------------------------------------------

function _promptConnectDetected(d) {
  // Pre-fills the add form with the detected URL and focuses the username
  // field — faster than typing it in, and the user still reviews before
  // submitting. Adding a server silently without creds isn't useful.
  const body = _overlay.querySelector('#ms-body');
  const providerSel = body.querySelector('#ms-provider');
  const nameIn = body.querySelector('#ms-name');
  const hostIn = body.querySelector('#ms-host');
  const portIn = body.querySelector('#ms-port');
  const userIn = body.querySelector('#ms-user');

  if (providerSel) providerSel.value = d.provider;
  if (nameIn && !nameIn.value) nameIn.value = `${_providerLabel(d.provider)} (local)`;

  try {
    const u = new URL(d.base_url);
    if (hostIn) hostIn.value = `${u.protocol}//${u.hostname}`;
    if (portIn) { portIn.value = u.port || ''; portIn.dataset.pristine = '0'; }
  } catch {
    if (hostIn) hostIn.value = d.base_url;
  }
  userIn?.focus();
  hostIn?.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

async function _submitAddForm(form) {
  const provider = form.querySelector('#ms-provider').value;
  const name = form.querySelector('#ms-name').value.trim();
  const hostRaw = form.querySelector('#ms-host').value.trim();
  const port = form.querySelector('#ms-port').value.trim();
  const username = form.querySelector('#ms-user').value.trim();
  const password = form.querySelector('#ms-pass').value;
  const token = form.querySelector('#ms-token').value.trim();

  if (!name || !hostRaw) {
    showToast('Name and host are required', 'warning');
    return;
  }
  // Providers that run with auth disabled (Suwayomi local-only is the
  // canonical case) accept empty creds — their server-side login() will
  // validate the deployment itself. All other providers still require
  // a token or username+password pair.
  const providerCfg = PROVIDERS.find(x => x.id === provider);
  const allowNoAuth = !!providerCfg?.allow_no_auth;
  if (!allowNoAuth && !token && (!username || !password)) {
    showToast('Provide a token or username + password', 'warning');
    return;
  }
  const base_url = _buildUrl(hostRaw, port);
  if (!base_url) {
    showToast('Invalid URL', 'warning');
    return;
  }

  const submitBtn = form.querySelector('button[type="submit"]');
  if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Connecting…'; }
  const payload = { provider, name, base_url, username, password, access_token: token };
  // Admin-only checkbox. The backend re-enforces the admin role —
  // a non-admin posting scope='shared' via DOM injection gets 403.
  const shareEl = form.querySelector('#ms-scope-shared');
  if (shareEl?.checked && _viewerIsAdmin) {
    payload.scope = 'shared';
  }
  await _addServer(payload);
  if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Connect'; }
}

function _buildUrl(hostRaw, port) {
  let host = hostRaw.trim();
  if (!host) return '';
  // Accept bare hostnames; default to http:// when scheme is omitted.
  let scheme = 'http';
  if (/^https?:\/\//i.test(host)) {
    try {
      const u = new URL(host);
      scheme = u.protocol.replace(':', '');
      host = u.hostname + (u.pathname && u.pathname !== '/' ? u.pathname : '');
    } catch { return ''; }
  }
  const portPart = port ? `:${port}` : '';
  return `${scheme}://${host}${portPart}`;
}

function _providerLabel(id) {
  const p = PROVIDERS.find(x => x.id === id);
  return p ? p.label : id;
}

function _surfaceGroupLabel(slug) {
  switch (slug) {
    case 'shows': return 'Shows';
    case 'movies': return 'Movies';
    case 'music_videos': return 'Music videos';
    case 'live_tv': return 'Live TV';
    case 'collections': return 'Collections';
    case 'playlists': return 'Playlists';
    default: return slug ? slug.replace(/_/g, ' ') : 'Other';
  }
}

function _wireLibraryRow(row) {
  const serverId = row.dataset.serverId;
  const libraryId = row.dataset.libraryId;
  if (!serverId || !libraryId) return;
  const submit = () => _updateLibrary(serverId, libraryId, {
    display_name_override: row.querySelector('[data-lib-display]')?.value?.trim() || '',
    surface_group_override: row.querySelector('[data-lib-surface]')?.value || '',
    is_hidden: !!row.querySelector('[data-lib-hidden]')?.checked,
    include_in_search: !!row.querySelector('[data-lib-search]')?.checked,
    include_in_overview: !!row.querySelector('[data-lib-overview]')?.checked,
  });
  row.querySelector('[data-lib-display]')?.addEventListener('change', submit);
  row.querySelector('[data-lib-surface]')?.addEventListener('change', submit);
  row.querySelector('[data-lib-hidden]')?.addEventListener('change', submit);
  row.querySelector('[data-lib-search]')?.addEventListener('change', submit);
  row.querySelector('[data-lib-overview]')?.addEventListener('change', submit);
}
