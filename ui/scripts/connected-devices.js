/**
 * connected-devices.js — Devices section inside the Connected Devices panel.
 *
 * Sibling to media-servers.js. Where media servers are SOURCES of media
 * the user owns (Emby, Jellyfin, Audiobookshelf), devices are TARGETS
 * the user can send things to (TVs via DLNA/Cast, speakers via AirPlay,
 * lights via Hue/Matter, augmentum's own UI surfaces).
 *
 * Public API:
 *   renderDevicesSection(container, { onChanged })
 *
 * Owns its own data fetching and event wiring. Caller provides the
 * container element + an `onChanged` callback for when devices are
 * added/removed (so the parent panel can mark itself dirty / refresh
 * cross-modal surfaces).
 */

import { escapeHtml, extractErrorMessage, showToast } from './app.js';
import { detectLocalSubnet } from './lan-probe.js';
import {
  arm as armDevice,
  disarm as disarmDevice,
  isArmedDevice,
  subscribe as subscribeArmed,
} from './armed-device.js';


// Driver list known to the substrate. Augmented at runtime from
// `/api/devices/drivers`. The fallback is the conservative set we
// expect to always be present.
const FALLBACK_DRIVERS = [
  { id: 'dlna',  label: 'DLNA / UPnP', description: 'Smart TVs and AV receivers (Sony, LG, Samsung, etc).' },
];

let _state = {
  devices: [],
  mobileDevices: [],
  discoveredFresh: [],
  drivers: FALLBACK_DRIVERS,
  refreshing: false,
  hasRunDiscovery: false,
  lastDiscoveryErrors: {},
  lastDiscoveryDurationS: 0,
  pathsRun: [],
  browseProbe: { stage: '', done: 0, total: 0, found: 0 },
};

let _onChanged = null;
let _unsubArmed = null;
let _mobilePair = null;
let _mobilePairPollTimer = null;


/* ------------------------------------------------------------------ *\
   Public mount
\* ------------------------------------------------------------------ */


export async function renderDevicesSection(container, { onChanged } = {}) {
  if (!container) return;
  _onChanged = onChanged || null;
  container.innerHTML = `
    <div class="cd-root" data-cd-root>
      <section class="cd-section cd-section-devices">
        <header class="cd-section-head">
          <div class="cd-section-titles">
            <h3 class="cd-section-title">TVs and speakers</h3>
            <p class="cd-section-sub">Cast chat audio, music, and video to anything on your network.</p>
          </div>
          <button class="btn btn-sm cd-section-action" data-cd-refresh title="">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 15.5-6.36L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15.5 6.36L3 16"/><path d="M3 21v-5h5"/></svg>
            <span data-cd-refresh-label>Find new devices</span>
          </button>
        </header>
        <div data-cd-discovered></div>
        <div data-cd-saved></div>
        <div data-cd-add></div>
      </section>

      <section class="cd-section cd-section-phone">
        <div data-cd-mobile></div>
      </section>

      <div data-cd-history></div>
    </div>
  `;

  const root = container.querySelector('[data-cd-root]');
  root.querySelector('[data-cd-refresh]').addEventListener('click', () => {
    _runDiscovery({ refresh: true });
  });

  // Re-render saved rows whenever arming changes so the primary action
  // and accent border on the armed row stay in sync.
  if (_unsubArmed) _unsubArmed();
  _unsubArmed = subscribeArmed(() => _renderSaved());

  await _loadDrivers();
  await _refresh();
  // Auto-run discovery on first mount per session so the panel reflects
  // the live LAN state without the user having to click "Search". On
  // re-opens within the same session, the user clicks Search explicitly
  // to refresh.
  if (!_state.hasRunDiscovery) {
    _state.hasRunDiscovery = true;
    _runDiscovery({ refresh: true });
  }
}


/* ------------------------------------------------------------------ *\
   Data
\* ------------------------------------------------------------------ */


async function _loadDrivers() {
  try {
    const resp = await fetch('/api/devices/drivers');
    if (resp.ok) {
      const body = await resp.json();
      if (Array.isArray(body.drivers) && body.drivers.length) {
        _state.drivers = body.drivers;
      }
    }
  } catch {
    // Substrate may not be initialized yet on cold-boot — fall back.
  }
}


async function _refresh() {
  const [devicesResp, mobileResp] = await Promise.all([
    fetch('/api/devices').catch(() => null),
    fetch('/api/auth/pair/devices', { credentials: 'same-origin' }).catch(() => null),
  ]);
  if (devicesResp && devicesResp.ok) {
    const body = await devicesResp.json().catch(() => ({}));
    _state.devices = body.devices || [];
  } else {
    _state.devices = [];
  }
  if (mobileResp && mobileResp.ok) {
    const body = await mobileResp.json().catch(() => ({}));
    _state.mobileDevices = body.devices || [];
  } else {
    _state.mobileDevices = [];
  }
  _render();
}


async function _runDiscovery({ refresh = false } = {}) {
  if (_state.refreshing) return;
  _state.refreshing = true;
  _state.pathsRun = [];
  _state.browseProbe = { stage: 'starting', done: 0, total: 0, found: 0 };
  _renderControls();

  // Two discovery paths run in parallel — each handles a different
  // failure mode of the other:
  //
  //   1. Server-side SSDP — fast (~3s) but multicast, fails on Docker
  //      Desktop (container in a VM can't reach LAN multicast).
  //
  //   2. Server-side TCP subnet sweep with a browser-supplied subnet
  //      hint. The browser is the only thing guaranteed to be on the
  //      user's LAN, so it computes the local subnet (from window.
  //      location.hostname or WebRTC ICE) and hands it to the server.
  //      The server then runs the actual TCP probes from inside the
  //      container — the augmentum container CAN reach LAN IPs via
  //      Docker NAT even when multicast doesn't cross. CSP `connect-src`
  //      blocks browser-side LAN fetches entirely, so all probing has
  //      to happen server-side; the browser just provides the hint.
  //
  // Whichever finds something first wins; we merge by (driver, native_id).

  const merged = new Map();  // (driver|native_id) -> DiscoveredDevice
  const errors = {};
  const startedAt = performance.now();

  const _registerPath = (label) => {
    _state.pathsRun.push({ label, status: 'running' });
    _renderControls();
  };
  const _markPath = (label, status, count) => {
    const path = _state.pathsRun.find(p => p.label === label);
    if (path) {
      path.status = status;
      path.count = count ?? 0;
    }
    _renderControls();
  };

  const _addResults = (devices) => {
    for (const d of devices || []) {
      const key = `${d.driver}|${d.native_id}`;
      if (!merged.has(key)) merged.set(key, d);
    }
  };

  // Path 1 — server-side SSDP
  _registerPath('Multicast');
  const ssdpPromise = fetch(`/api/devices/discover?refresh=${refresh ? 1 : 0}`)
    .then(async (resp) => {
      if (!resp.ok) {
        errors.ssdp = `HTTP ${resp.status}`;
        _markPath('Multicast', 'failed');
        return;
      }
      const body = await resp.json();
      _addResults(body.discovered);
      Object.assign(errors, body.errors || {});
      _markPath('Multicast', 'done', (body.discovered || []).length);
    })
    .catch((err) => {
      errors.ssdp = String(err?.message || err);
      _markPath('Multicast', 'failed');
    });

  // Path 2 — server-side TCP sweep with browser-supplied subnet hint.
  // The browser knows its own LAN IP via window.location.hostname or
  // WebRTC ICE; the server can't easily figure that out behind a
  // reverse proxy. Hand the subnet to the server and let it probe.
  _registerPath('LAN scan');
  const sweepPromise = (async () => {
    try {
      const subnet = await detectLocalSubnet();
      const params = subnet ? `?subnet=${encodeURIComponent(subnet)}` : '';
      const resp = await fetch(`/api/devices/sweep${params}`);
      if (!resp.ok) {
        errors.sweep = `HTTP ${resp.status}`;
        _markPath('LAN scan', 'failed');
        return;
      }
      const body = await resp.json();
      _addResults(body.discovered);
      Object.assign(errors, body.errors || {});
      _markPath('LAN scan', 'done', (body.discovered || []).length);
    } catch (err) {
      errors.sweep = String(err?.message || err);
      _markPath('LAN scan', 'failed');
    }
  })();

  await Promise.all([ssdpPromise, sweepPromise]);

  // Hide Emby/Jellyfin client sessions from auto-discovery. They're not
  // TVs or speakers — they're other phones logged into the same media
  // server, and listing them here is confusing. Manual add still works
  // for users who genuinely want to push playback into a client session.
  const HIDDEN_DISCOVERY_DRIVERS = new Set(['emby_remote', 'jellyfin_remote']);
  _state.discoveredFresh = Array.from(merged.values())
    .filter(d => !HIDDEN_DISCOVERY_DRIVERS.has(String(d.driver || '').toLowerCase()));
  _state.lastDiscoveryErrors = errors;
  _state.lastDiscoveryDurationS = (performance.now() - startedAt) / 1000;
  _state.refreshing = false;

  await _refresh(); // pulls the now-online statuses for saved devices
}


async function _saveDiscovered(disc) {
  try {
    const resp = await fetch('/api/devices', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        driver: disc.driver,
        host: disc.address?.host || '',
        port: disc.address?.port || null,
        label: disc.label || disc.metadata?.model_name || '',
        // `discovered` carries the full DiscoveredDevice so the server
        // can save provider-bridged devices (Emby/Jellyfin sessions)
        // with their real native_id/address instead of falling through
        // to the "manual:host:port" + unverified path.
        hint: {
          location_url: disc.address?.location || '',
          discovered: disc,
        },
      }),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      showToast(extractErrorMessage(body, 'Failed to save device'), 'error', 4000);
      return;
    }
    showToast(`Added ${disc.label}`, 'success', 2000);
    _state.discoveredFresh = _state.discoveredFresh.filter(d =>
      !(d.driver === disc.driver && d.native_id === disc.native_id),
    );
    _onChanged?.();
    await _refresh();
  } catch (err) {
    showToast(`Network error: ${err.message || err}`, 'error', 4000);
  }
}


async function _addManual(form) {
  const driver = form.querySelector('[name="driver"]').value;
  const host = form.querySelector('[name="host"]').value.trim();
  const portRaw = form.querySelector('[name="port"]').value.trim();
  const label = form.querySelector('[name="label"]').value.trim();

  if (!driver || !host) {
    showToast('Driver and host are required', 'error', 3000);
    return;
  }

  try {
    const resp = await fetch('/api/devices', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        driver, host,
        port: portRaw ? Number(portRaw) : null,
        label,
      }),
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      showToast(extractErrorMessage(body, 'Failed to add device'), 'error', 4000);
      return;
    }
    if (body.verified) {
      showToast(`Added ${body.device.label}`, 'success', 2500);
    } else {
      showToast(
        `Saved ${body.device.label} as unverified — couldn't reach it`,
        'info', 4000,
      );
    }
    form.reset();
    _onChanged?.();
    await _refresh();
  } catch (err) {
    showToast(`Network error: ${err.message || err}`, 'error', 4000);
  }
}


async function _testDevice(deviceId) {
  try {
    const resp = await fetch(`/api/devices/${encodeURIComponent(deviceId)}/test`, {
      method: 'POST',
    });
    const body = await resp.json().catch(() => ({}));
    if (resp.ok && body.reachable) {
      showToast('Device is reachable', 'success', 2000);
    } else {
      showToast('Device is not reachable', 'error', 2500);
    }
    await _refresh();
  } catch (err) {
    showToast(`Network error: ${err.message || err}`, 'error', 3000);
  }
}


async function _removeDevice(deviceId, label) {
  if (!window.confirm(`Remove "${label}" from your devices?`)) return;
  try {
    const resp = await fetch(`/api/devices/${encodeURIComponent(deviceId)}`, {
      method: 'DELETE',
    });
    if (resp.ok) {
      showToast(`Removed ${label}`, 'success', 2000);
      _onChanged?.();
      await _refresh();
    } else {
      showToast('Failed to remove device', 'error', 3000);
    }
  } catch (err) {
    showToast(`Network error: ${err.message || err}`, 'error', 3000);
  }
}


/* ------------------------------------------------------------------ *\
   Render
\* ------------------------------------------------------------------ */


function _render() {
  _renderControls();
  _renderMobilePairing();
  _renderDiscovered();
  _renderSaved();
  _renderAdd();
  _renderHistorySection();
}


/* ------------------------------------------------------------------ *\
   Mobile app pairing
\* ------------------------------------------------------------------ */


function _renderMobilePairing() {
  const root = _root();
  if (!root) return;
  const host = root.querySelector('[data-cd-mobile]');
  if (!host) return;

  const devices = _state.mobileDevices || [];
  const active = devices.filter(d => !d.revoked);
  const revoked = devices.filter(d => d.revoked);

  host.innerHTML = `
    <header class="cd-section-head">
      <div class="cd-section-titles">
        <h3 class="cd-section-title">Your phone</h3>
        <p class="cd-section-sub">Pair an Android phone to chat, take voice calls, and get notifications on the go.</p>
      </div>
      <button class="btn btn-sm btn-primary cd-section-action" data-cd-mobile-start>
        ${_mobilePair ? 'New code' : 'Pair phone'}
      </button>
    </header>
    ${_mobilePair ? _mobilePairCardHtml(_mobilePair) : ''}
    ${active.length || revoked.length ? `
      <div class="cd-mobile-list">
        ${active.map(_mobileDeviceRowHtml).join('')}
        ${revoked.map(_mobileDeviceRowHtml).join('')}
      </div>
    ` : `
      <div class="cd-mobile-empty">No paired phones yet.</div>
    `}
  `;

  host.querySelector('[data-cd-mobile-start]')?.addEventListener('click', () => {
    _startMobilePair();
  });
  host.querySelector('[data-cd-mobile-cancel]')?.addEventListener('click', () => {
    _clearMobilePairPoll();
    _mobilePair = null;
    _renderMobilePairing();
  });
  host.querySelector('[data-cd-mobile-copy]')?.addEventListener('click', () => {
    _copyMobilePairLink();
  });
  host.querySelector('[data-cd-mobile-approve]')?.addEventListener('click', () => {
    _approveMobilePair();
  });
  host.querySelectorAll('[data-cd-mobile-revoke]').forEach(btn => {
    btn.addEventListener('click', () => {
      _revokeMobileDevice(btn.dataset.cdMobileRevoke);
    });
  });
}


function _mobilePairCardHtml(pair) {
  const state = String(pair.state || 'pending').toLowerCase();
  const claim = pair.claim || {};
  const claimed = state === 'claimed';
  const approved = state === 'approved';
  const consumed = state === 'consumed';
  const expired = state === 'expired' || Number(pair.expires_in || 0) <= 0;
  return `
    <div class="cd-mobile-pair cd-mobile-pair-${escapeHtml(state)}">
      <div class="cd-mobile-pair-main">
        <div class="cd-mobile-qr-wrap">
          ${pair.qr_url && !expired && !consumed
            ? `<img class="cd-mobile-qr" src="${escapeHtml(pair.qr_url)}" alt="">`
            : `<div class="cd-mobile-qr cd-mobile-qr-placeholder">${consumed ? 'Paired' : 'Expired'}</div>`}
        </div>
        <div class="cd-mobile-pair-copy">
          <div class="cd-mobile-pair-state">${escapeHtml(_mobilePairStatusText(pair))}</div>
          <div class="cd-mobile-code">${escapeHtml(pair.pair_code || '')}</div>
          ${claim.device_id ? _mobileClaimHtml(claim) : ''}
          ${pair.pair_url ? `<button class="btn btn-sm" data-cd-mobile-copy>Copy link</button>` : ''}
        </div>
      </div>
      <div class="cd-mobile-pair-actions">
        ${claimed ? `<button class="btn btn-sm btn-primary" data-cd-mobile-approve>Approve phone</button>` : ''}
        ${approved ? `<span class="cd-mobile-waiting">Waiting for phone</span>` : ''}
        <button class="btn btn-sm" data-cd-mobile-cancel>${consumed || expired ? 'Close' : 'Cancel'}</button>
      </div>
    </div>
  `;
}


function _mobileClaimHtml(claim) {
  const label = claim.label || 'Android phone';
  const bits = [
    claim.platform || '',
    claim.app_version ? `app ${claim.app_version}` : '',
  ].filter(Boolean).join(' · ');
  const caps = Array.isArray(claim.capabilities) ? claim.capabilities.slice(0, 4) : [];
  return `
    <div class="cd-mobile-claim">
      <div class="cd-mobile-claim-title">${escapeHtml(label)}</div>
      ${bits ? `<div class="cd-mobile-claim-sub">${escapeHtml(bits)}</div>` : ''}
      ${caps.length ? `<div class="cd-mobile-caps">${caps.map(cap => `<span>${escapeHtml(cap)}</span>`).join('')}</div>` : ''}
    </div>
  `;
}


function _mobileDeviceRowHtml(device) {
  const revoked = !!device.revoked;
  const title = device.label || device.device_id || 'Android phone';
  const sub = [
    device.platform || 'android',
    device.app_version ? `app ${device.app_version}` : '',
    device.last_seen_at ? `last seen ${device.last_seen_at}` : '',
  ].filter(Boolean).join(' · ');
  return `
    <div class="cd-mobile-row${revoked ? ' cd-mobile-row-revoked' : ''}">
      <span class="cd-row-dot ${revoked ? 'cd-dot-offline' : 'cd-dot-paired'}"></span>
      <div class="cd-mobile-row-main">
        <div class="cd-mobile-row-title">${escapeHtml(title)}</div>
        <div class="cd-mobile-row-sub">${escapeHtml(sub)}</div>
      </div>
      ${revoked
        ? `<span class="cd-mobile-revoked-label">Revoked</span>`
        : `<button class="btn btn-sm" data-cd-mobile-revoke="${escapeHtml(device.id)}">Revoke</button>`}
    </div>
  `;
}


function _mobilePairStatusText(pair) {
  const state = String(pair.state || 'pending').toLowerCase();
  const expires = _formatExpires(pair.expires_in);
  if (state === 'claimed') return 'Phone is ready to approve.';
  if (state === 'approved') return 'Approved. Finish on the phone.';
  if (state === 'consumed') return 'Phone paired.';
  if (state === 'expired' || Number(pair.expires_in || 0) <= 0) return 'Pair code expired.';
  return expires ? `Scan this code. ${expires}` : 'Scan this code.';
}


function _formatExpires(seconds) {
  const total = Math.max(0, Number(seconds || 0));
  if (!total) return '';
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  if (mins <= 0) return `${secs}s left`;
  return `${mins}m ${String(secs).padStart(2, '0')}s left`;
}


async function _startMobilePair() {
  _clearMobilePairPoll();
  try {
    const resp = await fetch('/api/auth/pair/start', {
      method: 'POST',
      credentials: 'same-origin',
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      showToast(extractErrorMessage(body, 'Could not start phone pairing'), 'error', 4000);
      return;
    }
    _mobilePair = body;
    _renderMobilePairing();
    _scheduleMobilePairPoll(1000);
  } catch (err) {
    showToast(`Pairing failed: ${err.message || err}`, 'error', 4000);
  }
}


async function _pollMobilePair() {
  if (!_mobilePair?.status_path) return;
  try {
    const resp = await fetch(_mobilePair.status_path, { credentials: 'same-origin' });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      _mobilePair = { ..._mobilePair, state: 'expired', expires_in: 0 };
      _renderMobilePairing();
      return;
    }
    _mobilePair = { ..._mobilePair, ...body };
    _renderMobilePairing();
    if (body.state === 'consumed') {
      showToast('Phone paired', 'success', 2500);
      await _refresh();
      setTimeout(() => {
        if (_mobilePair?.state === 'consumed') {
          _mobilePair = null;
          _renderMobilePairing();
        }
      }, 2200);
      return;
    }
    if (!['expired', 'consumed'].includes(String(body.state || '').toLowerCase())) {
      _scheduleMobilePairPoll(body.state === 'approved' ? 1000 : 1500);
    }
  } catch {
    _scheduleMobilePairPoll(2000);
  }
}


function _scheduleMobilePairPoll(delayMs = 1500) {
  _clearMobilePairPoll();
  _mobilePairPollTimer = window.setTimeout(_pollMobilePair, delayMs);
}


function _clearMobilePairPoll() {
  if (_mobilePairPollTimer) {
    window.clearTimeout(_mobilePairPollTimer);
    _mobilePairPollTimer = null;
  }
}


async function _approveMobilePair() {
  if (!_mobilePair?.pair_code) return;
  try {
    const resp = await fetch(`/api/auth/pair/approve/${encodeURIComponent(_mobilePair.pair_code)}`, {
      method: 'POST',
      credentials: 'same-origin',
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      showToast(extractErrorMessage(body, 'Could not approve phone'), 'error', 4000);
      return;
    }
    _mobilePair = { ..._mobilePair, ...body };
    _renderMobilePairing();
    _scheduleMobilePairPoll(1000);
  } catch (err) {
    showToast(`Approval failed: ${err.message || err}`, 'error', 4000);
  }
}


async function _copyMobilePairLink() {
  const link = _mobilePair?.pair_url || '';
  if (!link) return;
  try {
    await navigator.clipboard.writeText(link);
    showToast('Pair link copied', 'success', 1800);
  } catch {
    showToast('Could not copy pair link', 'warning', 2200);
  }
}


async function _revokeMobileDevice(mobileId) {
  if (!mobileId) return;
  const device = (_state.mobileDevices || []).find(d => d.id === mobileId);
  const label = device?.label || 'this phone';
  if (!window.confirm(`Revoke ${label}? Its Android sessions will stop working.`)) return;
  try {
    const resp = await fetch(`/api/auth/pair/devices/${encodeURIComponent(mobileId)}/revoke`, {
      method: 'POST',
      credentials: 'same-origin',
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      showToast(extractErrorMessage(body, 'Could not revoke phone'), 'error', 4000);
      return;
    }
    showToast('Phone revoked', 'success', 2200);
    await _refresh();
  } catch (err) {
    showToast(`Revoke failed: ${err.message || err}`, 'error', 4000);
  }
}

// ── Recent + Favorites + Pair flow ──────────────────────────────────
// Surfaces the play-history API as an expandable section. "Recent" shows
// recently-cast items so the user can resume; "Favorites" pins items
// they want quick access to. Toggling a favorite hits POST
// /api/devices/history/favorite. Pairing for devices that need a code
// uses a small prompt flow (pair/start → display code → pair/complete).

let _historyExpanded = false;

function _renderHistorySection() {
  const root = _root();
  if (!root) return;
  const host = root.querySelector('[data-cd-history]');
  if (!host) return;
  host.innerHTML = `
    <details class="cd-history" ${_historyExpanded ? 'open' : ''}>
      <summary class="cd-history-summary">
        <span>Recently cast &amp; favorites</span>
        <svg class="cd-history-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>
      </summary>
      <div class="cd-history-body">
        <div data-cd-recent></div>
        <div data-cd-favs></div>
      </div>
    </details>`;
  const details = host.querySelector('details');
  details.addEventListener('toggle', () => {
    _historyExpanded = details.open;
    if (details.open) {
      _loadRecentAndFavorites();
    }
  });
  if (_historyExpanded) _loadRecentAndFavorites();
}

async function _loadRecentAndFavorites() {
  const root = _root();
  if (!root) return;
  const recentHost = root.querySelector('[data-cd-recent]');
  const favsHost = root.querySelector('[data-cd-favs]');
  if (!recentHost || !favsHost) return;
  recentHost.innerHTML = '<div class="cd-history-loading">Loading recent…</div>';
  favsHost.innerHTML = '<div class="cd-history-loading">Loading favorites…</div>';

  const [recentResp, favsResp] = await Promise.all([
    fetch('/api/devices/history/recent?limit=15', { credentials: 'same-origin' }).catch(() => null),
    fetch('/api/devices/history/favorites?limit=15', { credentials: 'same-origin' }).catch(() => null),
  ]);

  const recentData = recentResp && recentResp.ok ? await recentResp.json() : { history: [] };
  const favsData = favsResp && favsResp.ok ? await favsResp.json() : { favorites: [] };

  const renderList = (rows, isFav) => {
    if (!rows || rows.length === 0) {
      return `<div class="cd-history-empty">No ${isFav ? 'favorites' : 'recent items'} yet.</div>`;
    }
    return rows.map(r => {
      const title = r.title || r.content_title || r.label || r.content_id || 'Unnamed';
      const sub = r.device_label || r.device_id || '';
      const star = isFav ? '★' : (r.is_favorite ? '★' : '☆');
      const contentKey = r.content_kind && r.content_id
        ? `${r.content_kind}:${r.content_id}`
        : '';
      return `
        <div class="cd-history-row">
          <button class="cd-history-fav" data-cd-fav-toggle="${escapeHtml(contentKey)}"
                  data-cd-fav-state="${isFav || r.is_favorite ? '1' : '0'}"
                  title="${isFav || r.is_favorite ? 'Remove favorite' : 'Add favorite'}">${star}</button>
          <div class="cd-history-meta">
            <div class="cd-history-title">${escapeHtml(String(title))}</div>
            ${sub ? `<div class="cd-history-sub">${escapeHtml(String(sub))}</div>` : ''}
          </div>
        </div>`;
    }).join('');
  };

  recentHost.innerHTML = `
    <div class="cd-history-group-title">Recent</div>
    ${renderList(recentData.history || [], false)}`;
  favsHost.innerHTML = `
    <div class="cd-history-group-title">Favorites</div>
    ${renderList(favsData.favorites || [], true)}`;

  // Wire favorite toggles. The backend takes (content_kind, content_id,
  // favorite=bool); we encoded content_key = "kind:id" in the button.
  [recentHost, favsHost].forEach(host => {
    host.querySelectorAll('[data-cd-fav-toggle]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const key = btn.dataset.cdFavToggle || '';
        if (!key.includes(':')) return;
        const [contentKind, contentId] = key.split(':', 2);
        const wasOn = btn.dataset.cdFavState === '1';
        btn.disabled = true;
        try {
          const r = await fetch('/api/devices/history/favorite', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'same-origin',
            body: JSON.stringify({
              content_kind: contentKind,
              content_id: contentId,
              favorite: !wasOn,
            }),
          });
          if (r.ok) {
            await _loadRecentAndFavorites();
          } else {
            btn.disabled = false;
          }
        } catch {
          btn.disabled = false;
        }
      });
    });
  });
}

// Pairing — Cast/Chromecast and some Sony BRAVIA require a pairing code
// shown on the TV screen. start returns a session token; complete
// submits the user-entered code from the TV. Exposed as a window helper
// so other surfaces (the saved-device row, the cast picker) can invoke
// it without duplicating the prompt logic.
window.augmentumPairDevice = async function pairDevice(deviceId, deviceLabel = '') {
  if (!deviceId) return false;
  let startResult;
  try {
    const r = await fetch(`/api/devices/${encodeURIComponent(deviceId)}/pair/start`, {
      method: 'POST', credentials: 'same-origin',
    });
    if (!r.ok) {
      showToast?.(`Pairing failed to start (status ${r.status})`, 'error');
      return false;
    }
    startResult = await r.json();
  } catch (err) {
    showToast?.(`Pairing failed: ${err.message || err}`, 'error');
    return false;
  }

  // Some drivers complete immediately (no code needed) — short-circuit.
  if (startResult.status === 'paired' || startResult.completed) {
    showToast?.(`Paired with ${deviceLabel || deviceId}`, 'success');
    return true;
  }

  const code = prompt(
    `Enter the pairing code shown on ${deviceLabel || deviceId}` +
    (startResult.message ? `\n\n${startResult.message}` : ''),
    '',
  );
  if (code === null) return false;
  try {
    const r2 = await fetch(`/api/devices/${encodeURIComponent(deviceId)}/pair/complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ code: code.trim() }),
    });
    if (!r2.ok) {
      showToast?.(`Pairing failed (status ${r2.status})`, 'error');
      return false;
    }
    showToast?.(`Paired with ${deviceLabel || deviceId}`, 'success');
    return true;
  } catch (err) {
    showToast?.(`Pairing failed: ${err.message || err}`, 'error');
    return false;
  }
};


function _renderControls() {
  const root = _root();
  if (!root) return;
  const label = root.querySelector('[data-cd-refresh-label]');
  const btn = root.querySelector('[data-cd-refresh]');
  if (!label || !btn) return;

  // Per-path detail (multicast vs LAN scan, success/fail counts) lives on
  // the button title — useful for debugging without occupying the surface.
  btn.title = _pathsTitle() || '';

  if (_state.refreshing) {
    label.textContent = 'Searching…';
    btn.disabled = true;
  } else {
    label.textContent = 'Find new devices';
    btn.disabled = false;
  }
}


function _pathsTitle() {
  if (!_state.pathsRun.length) return '';
  return _state.pathsRun.map(p => {
    const status = p.status === 'done'
      ? (p.count > 0 ? `${p.count} found` : 'no devices')
      : p.status;
    return `${p.label}: ${status}`;
  }).join(' · ');
}


function _renderDiscovered() {
  const root = _root();
  if (!root) return;
  const host = root.querySelector('[data-cd-discovered]');
  if (!host) return;
  const fresh = _state.discoveredFresh || [];
  if (!fresh.length) {
    host.innerHTML = '';
    return;
  }
  host.innerHTML = `
    <div class="cd-banner">
      <div class="cd-banner-head">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l4 4L19 7"/></svg>
        <span>Found ${fresh.length} new ${fresh.length === 1 ? 'device' : 'devices'} on your network.</span>
      </div>
      <div class="cd-banner-list">
        ${fresh.map(d => `
          <div class="cd-banner-row">
            <div class="cd-banner-meta">
              <div class="cd-banner-label">${escapeHtml(d.label)}</div>
              <div class="cd-banner-sub">
                ${escapeHtml(d.driver.toUpperCase())}
                ${d.metadata?.manufacturer ? ` &middot; ${escapeHtml(d.metadata.manufacturer)}` : ''}
                ${d.metadata?.model_name ? ` ${escapeHtml(d.metadata.model_name)}` : ''}
              </div>
            </div>
            <button class="btn btn-sm btn-primary" data-cd-add-discovered="${escapeHtml(d.driver)}|${escapeHtml(d.native_id)}">
              Add
            </button>
          </div>
        `).join('')}
      </div>
    </div>
  `;

  host.querySelectorAll('[data-cd-add-discovered]').forEach(btn => {
    btn.addEventListener('click', () => {
      const [driver, nativeId] = btn.dataset.cdAddDiscovered.split('|');
      const disc = fresh.find(d => d.driver === driver && d.native_id === nativeId);
      if (disc) _saveDiscovered(disc);
    });
  });
}


function _renderSaved() {
  const root = _root();
  if (!root) return;
  const host = root.querySelector('[data-cd-saved]');
  if (!host) return;
  if (!_state.devices.length) {
    const ran = _state.hasRunDiscovery && !_state.refreshing;
    const errs = _state.lastDiscoveryErrors || {};
    const hadErrors = Object.keys(errs).length > 0;
    host.innerHTML = `
      <div class="cd-empty">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="14" rx="2"/><path d="M12 18v3M8 21h8"/></svg>
        <div class="cd-empty-title">${ran ? `Nothing turned up` : `No devices yet`}</div>
        <div class="cd-empty-sub">
          ${ran
            ? `If you know your TV or speaker's IP address, add it below.`
            : `Tap "Find new devices" above to look on your network.`}
        </div>
        ${ran && !hadErrors ? `
          <details class="cd-empty-why">
            <summary>Why didn't it find anything?</summary>
            <div class="cd-empty-why-body">
              Either this computer isn't on the same Wi-Fi as your TV, or the TV has its "screen sharing" feature turned off.
              Most TVs still work if you add them by IP address — try the form below.
            </div>
          </details>` : ''}
      </div>
    `;
    return;
  }

  host.innerHTML = `
    <div class="cd-saved-list">
      ${_state.devices.map(d => _savedRowHtml(d)).join('')}
    </div>
  `;

  host.querySelectorAll('[data-cd-arm]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.cdArm;
      const device = _state.devices.find(x => x.id === id);
      if (device) {
        armDevice(device);
        showToast(`Casting to ${device.label}`, 'success', 2200);
      }
    });
  });
  host.querySelectorAll('[data-cd-disarm]').forEach(btn => {
    btn.addEventListener('click', () => disarmDevice());
  });
  host.querySelectorAll('[data-cd-kebab]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      _toggleKebab(btn.dataset.cdKebab);
    });
  });
  host.querySelectorAll('[data-cd-test]').forEach(btn => {
    btn.addEventListener('click', () => {
      _closeKebabs();
      _testDevice(btn.dataset.cdTest);
    });
  });
  host.querySelectorAll('[data-cd-remove]').forEach(btn => {
    btn.addEventListener('click', () => {
      _closeKebabs();
      const [id, label] = btn.dataset.cdRemove.split('|');
      _removeDevice(id, label);
    });
  });
}


function _savedRowHtml(d) {
  const armed = isArmedDevice(d.id);
  const statusClass = String(d.status || 'unverified').toLowerCase();
  const statusTitle = statusClass === 'online' ? 'Online'
    : statusClass === 'offline' ? 'Offline'
    : statusClass === 'paired' ? 'Paired'
    : 'Not yet verified';
  return `
    <div class="cd-row${armed ? ' cd-row-armed' : ''}" data-cd-id="${escapeHtml(d.id)}">
      <span class="cd-row-dot cd-dot-${escapeHtml(statusClass)}" title="${escapeHtml(statusTitle)}" aria-label="${escapeHtml(statusTitle)}"></span>
      <div class="cd-row-icon" aria-hidden="true">${_iconForDriver(d.driver)}</div>
      <div class="cd-row-meta">
        <div class="cd-row-label">${escapeHtml(d.label)}</div>
        <div class="cd-row-sub">${escapeHtml(_humanDriverPhrase(d))}</div>
      </div>
      <div class="cd-row-actions">
        ${armed
          ? `<button class="btn btn-sm cd-row-disarm" data-cd-disarm>Stop using</button>`
          : `<button class="btn btn-sm btn-primary" data-cd-arm="${escapeHtml(d.id)}">Use this device</button>`}
        <div class="cd-row-kebab-wrap">
          <button class="cd-row-kebab" data-cd-kebab="${escapeHtml(d.id)}" aria-label="More actions" title="More actions">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="5" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="12" cy="19" r="1.6"/></svg>
          </button>
          <div class="cd-row-kebab-menu" data-cd-kebab-menu="${escapeHtml(d.id)}">
            <button data-cd-test="${escapeHtml(d.id)}">Test reachability</button>
            <button data-cd-remove="${escapeHtml(d.id)}|${escapeHtml(d.label)}" class="cd-row-kebab-danger">Remove device</button>
          </div>
        </div>
      </div>
    </div>
  `;
}


function _toggleKebab(deviceId) {
  const wasOpen = document.querySelector(`[data-cd-kebab-menu="${deviceId}"]`)?.classList.contains('open');
  _closeKebabs();
  if (!wasOpen) {
    document.querySelector(`[data-cd-kebab-menu="${deviceId}"]`)?.classList.add('open');
  }
}


function _closeKebabs() {
  document.querySelectorAll('.cd-row-kebab-menu.open').forEach(el => el.classList.remove('open'));
}


// Close any open kebab when clicking outside. Mounted once; cheap.
if (typeof window !== 'undefined' && !window.__cdKebabCloseWired) {
  window.__cdKebabCloseWired = true;
  document.addEventListener('click', () => _closeKebabs());
}


function _renderAdd() {
  const root = _root();
  if (!root) return;
  const host = root.querySelector('[data-cd-add]');
  if (!host) return;

  const driverOptions = _state.drivers
    .filter(d => !(d.discovery_modes || []).includes('manual_only') || true)
    .map(d => `<option value="${escapeHtml(d.id)}">${escapeHtml(d.label)}</option>`)
    .join('');

  host.innerHTML = `
    <details class="cd-add">
      <summary class="cd-add-summary">
        <span>Add by IP address</span>
      </summary>
      <form class="cd-add-form" data-cd-add-form>
        <label class="cd-field">
          <span>Type</span>
          <select name="driver">${driverOptions}</select>
        </label>
        <label class="cd-field">
          <span>IP or hostname</span>
          <input type="text" name="host" placeholder="192.168.1.42" required>
        </label>
        <label class="cd-field">
          <span>Port (optional)</span>
          <input type="number" name="port" placeholder="auto-detect" min="1" max="65535">
        </label>
        <label class="cd-field cd-field-wide">
          <span>Name (optional)</span>
          <input type="text" name="label" placeholder="Living Room TV">
        </label>
        <div class="cd-add-actions">
          <button type="submit" class="btn btn-sm btn-primary">Add device</button>
        </div>
      </form>
    </details>
  `;

  host.querySelector('[data-cd-add-form]')?.addEventListener('submit', (e) => {
    e.preventDefault();
    _addManual(e.target);
  });
}


/* ------------------------------------------------------------------ *\
   Helpers
\* ------------------------------------------------------------------ */


function _root() {
  return document.querySelector('[data-cd-root]');
}


function _humanDriverPhrase(device) {
  // Surface a calm one-liner instead of "DLNA · http://192.168.x.x:1216".
  // Host details are still useful for debugging, but not as the default
  // user-visible subline.
  const driver = String(device.driver || '').toLowerCase();
  const map = {
    dlna: 'Smart TV',
    cast: 'Chromecast',
    cast_custom: 'Chromecast',
    airplay: 'AirPlay',
    emby_remote: 'via Emby',
    jellyfin_remote: 'via Jellyfin',
    hue: 'Philips Hue',
    matter: 'Matter device',
    sonos: 'Sonos',
    esphome: 'ESPHome',
    augmentum_surface: 'Augmentum panel',
  };
  if (map[driver]) return map[driver];
  const known = _state.drivers.find(x => x.id === driver);
  return known?.label || driver.toUpperCase();
}


function _iconForDriver(driverId) {
  // Single TV/UPnP icon for now. As more drivers ship (cast, hue, matter,
  // etc.) they'll get their own icons.
  if (driverId === 'dlna' || driverId === 'cast' || driverId === 'airplay') {
    return `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="13" rx="2"/><path d="M8 21h8M12 17v4"/></svg>`;
  }
  if (driverId === 'hue' || driverId === 'lighting') {
    return `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a7 7 0 0 0-4 12.7c.7.5 1 1.3 1 2.1V19a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1v-2.2c0-.8.3-1.6 1-2.1A7 7 0 0 0 12 2z"/><path d="M9 22h6"/></svg>`;
  }
  return `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v4l3 2"/></svg>`;
}
