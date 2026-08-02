/**
 * cast-guests.js — host-side Manage Guests UI.
 *
 * Loads from /ui/cast-guests/ (cookie-authed via the standard
 * AuthMiddleware path). Calls the /api/cast/guests endpoints to
 * list/rename/recolor/delete guest profiles + revoke per-device
 * links. No invite/QR flow here — that lives on cast-control.
 */

const PALETTE = [
  '#4ade80', '#60a5fa', '#fbbf24', '#f472b6',
  '#a78bfa', '#fb923c', '#34d399', '#f87171',
];

function _esc(s) {
  return String(s ?? '')
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}

function _formatRelative(unixSec) {
  if (!unixSec) return '';
  const now = Date.now() / 1000;
  const delta = Math.max(0, now - unixSec);
  if (delta < 60) return 'just now';
  if (delta < 3600) return `${Math.floor(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)} hr ago`;
  const days = Math.floor(delta / 86400);
  if (days < 30) return `${days} day${days === 1 ? '' : 's'} ago`;
  if (days < 365) return `${Math.floor(days / 30)} mo ago`;
  return `${Math.floor(days / 365)} yr ago`;
}


/* ── State + render ─────────────────────────────────────────────── */

let _guests = [];
let _activeProfileId = null;
let _activeDetail = null;

async function _loadGuests() {
  try {
    const r = await fetch('/api/cast/guests');
    if (!r.ok) {
      _renderError(`Could not load (HTTP ${r.status})`);
      return;
    }
    const body = await r.json();
    _guests = Array.isArray(body.guests) ? body.guests : [];
    _renderList();
  } catch (err) {
    _renderError(String(err?.message || err));
  }
}

function _renderList() {
  const list = document.querySelector('[data-list]');
  const empty = document.querySelector('[data-empty]');
  if (!list || !empty) return;
  if (_guests.length === 0) {
    list.innerHTML = '';
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  list.innerHTML = _guests.map((g) => {
    const color = g.color || _autoColor(g.id);
    const lastSeen = _formatRelative(g.last_seen_at);
    const plays = g.play_count
      ? `${g.play_count} session${g.play_count === 1 ? '' : 's'}`
      : 'no sessions yet';
    const devices = g.device_count
      ? `${g.device_count} device${g.device_count === 1 ? '' : 's'}`
      : 'no devices';
    return `
      <li class="cg-item" data-profile-id="${_esc(g.id)}">
        <span class="cg-item-dot" style="background:${_esc(color)}"></span>
        <div class="cg-item-body">
          <div class="cg-item-name">${_esc(g.display_name)}</div>
          <div class="cg-item-meta">
            <span>${_esc(plays)}</span>
            <span>${_esc(devices)}</span>
            <span>last seen ${_esc(lastSeen)}</span>
          </div>
        </div>
        <span class="cg-item-chev">›</span>
      </li>
    `;
  }).join('');
  list.querySelectorAll('.cg-item').forEach((el) => {
    el.addEventListener('click', () => _openDrawer(el.dataset.profileId));
  });
}

function _renderError(msg) {
  const list = document.querySelector('[data-list]');
  if (list) list.innerHTML = `<div class="cg-empty-line">${_esc(msg)}</div>`;
}

function _autoColor(id) {
  // Stable hash → palette index for guests without an assigned color.
  let h = 0;
  for (let i = 0; i < id.length; i++) {
    h = ((h << 5) - h) + id.charCodeAt(i);
    h |= 0;
  }
  return PALETTE[Math.abs(h) % PALETTE.length];
}


/* ── Drawer ─────────────────────────────────────────────────────── */

async function _openDrawer(profileId) {
  _activeProfileId = profileId;
  try {
    const r = await fetch(`/api/cast/guests/${encodeURIComponent(profileId)}`);
    if (!r.ok) return;
    _activeDetail = await r.json();
    _renderDrawer();
  } catch {}
}

function _closeDrawer() {
  _activeProfileId = null;
  _activeDetail = null;
  document.querySelector('[data-drawer]').hidden = true;
}

function _renderDrawer() {
  const drawer = document.querySelector('[data-drawer]');
  if (!drawer || !_activeDetail) return;
  drawer.hidden = false;
  const { profile, devices } = _activeDetail;
  const color = profile.color || _autoColor(profile.id);

  document.querySelector('[data-drawer-color]').style.background = color;
  document.querySelector('[data-drawer-name]').value = profile.display_name;
  document.querySelector('[data-drawer-played]').textContent =
    `${profile.play_count} session${profile.play_count === 1 ? '' : 's'}`;
  document.querySelector('[data-drawer-last]').textContent =
    `Last seen ${_formatRelative(profile.last_seen_at)}`;

  // Color swatches.
  const colorRow = document.querySelector('[data-color-row]');
  colorRow.innerHTML = PALETTE.map((c) => `
    <button class="cg-color-pip ${c === color ? 'is-active' : ''}"
            data-color="${c}"
            style="background:${c}"
            aria-label="${c}"></button>
  `).join('');
  colorRow.querySelectorAll('.cg-color-pip').forEach((pip) => {
    pip.addEventListener('click', () => _setColor(pip.dataset.color));
  });

  // Devices.
  const devList = document.querySelector('[data-devices]');
  const devEmpty = document.querySelector('[data-devices-empty]');
  if (devices.length === 0) {
    devList.innerHTML = '';
    devEmpty.hidden = false;
  } else {
    devEmpty.hidden = true;
    devList.innerHTML = devices.map((d) => `
      <li class="cg-device" data-device-id="${_esc(d.id)}">
        <div class="cg-device-label">
          <span>${_esc(d.label || 'Unnamed device')}</span>
          <small>first seen ${_esc(_formatRelative(d.first_seen_at))}, last ${_esc(_formatRelative(d.last_seen_at))}</small>
        </div>
        <button class="cg-device-revoke" data-revoke="${_esc(d.id)}">Revoke</button>
      </li>
    `).join('');
    devList.querySelectorAll('[data-revoke]').forEach((btn) => {
      btn.addEventListener('click', () => _revokeDevice(btn.dataset.revoke));
    });
  }
}

async function _setColor(color) {
  if (!_activeProfileId) return;
  await fetch(`/api/cast/guests/${encodeURIComponent(_activeProfileId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ color }),
  });
  await _openDrawer(_activeProfileId);
  await _loadGuests();
}

async function _saveName() {
  if (!_activeProfileId || !_activeDetail) return;
  const input = document.querySelector('[data-drawer-name]');
  const newName = (input?.value || '').trim();
  if (!newName || newName === _activeDetail.profile.display_name) return;
  const r = await fetch(`/api/cast/guests/${encodeURIComponent(_activeProfileId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ display_name: newName }),
  });
  if (r.status === 409) {
    alert('That name is already in use. Pick a different one.');
    input.value = _activeDetail.profile.display_name;
    return;
  }
  await _openDrawer(_activeProfileId);
  await _loadGuests();
}

async function _revokeDevice(deviceId) {
  if (!_activeProfileId) return;
  if (!confirm('Revoke this device? The guest will be asked for their name on next join.')) return;
  await fetch(
    `/api/cast/guests/${encodeURIComponent(_activeProfileId)}/devices/${encodeURIComponent(deviceId)}`,
    { method: 'DELETE' },
  );
  await _openDrawer(_activeProfileId);
}

async function _deleteGuest() {
  if (!_activeProfileId || !_activeDetail) return;
  const name = _activeDetail.profile.display_name;
  if (!confirm(`Delete ${name}? Their saved games will detach to your library; their device links and play history are removed.`)) return;
  await fetch(`/api/cast/guests/${encodeURIComponent(_activeProfileId)}`, {
    method: 'DELETE',
  });
  _closeDrawer();
  await _loadGuests();
}


/* ── Wire ────────────────────────────────────────────────────────── */

document.querySelector('[data-drawer-close]')?.addEventListener('click', _closeDrawer);
document.querySelector('[data-drawer-name]')?.addEventListener('blur', _saveName);
document.querySelector('[data-drawer-name]')?.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') ev.target.blur();
});
document.querySelector('[data-delete]')?.addEventListener('click', _deleteGuest);

_loadGuests();
