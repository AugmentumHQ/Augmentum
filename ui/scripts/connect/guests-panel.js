/* connect/guests-panel.js — host-side guest management (Connect Phase 3b).
 *
 * Docked panel (right side, parallel to calls-panel.js) listing the guests
 * this host has invited via durable guest-pass links. For each guest:
 *   - an "Allow calls" toggle  → PATCH /api/connect/guests/{id} {scopes}
 *   - a "Revoke" kill-switch    → POST  /api/connect/guests/{id}/revoke
 * Plus an "Invite a guest" action that mints an external_guest invite link.
 *
 * Reads:   GET   /api/connect/guests
 * Mutates: PATCH /api/connect/guests/{id}, POST /api/connect/guests/{id}/revoke
 *          POST  /api/auth/invites  (kind=external_guest)
 *
 * Created on first open (command palette: "Connect: Manage guests").
 */

import { escapeHtml, showConfirm, showToast } from '../app.js';
import { getSettings } from '../settings.js';
import { registerCommand } from '../command-palette.js';
import { icon } from './icons.js';
import { mountMintForm } from './invite-mint.js';
import { resolvePeerName } from './messages.js';

let _panel = null;
let _initialized = false;
let _deferredRetryArmed = false;

function _isEnabled() {
  const s = getSettings?.();
  return !!(s && s.connectEnabled);
}

export function initConnectGuestsUI() {
  if (_initialized) return;
  if (!_isEnabled()) {
    if (!_deferredRetryArmed) {
      _deferredRetryArmed = true;
      const retry = () => {
        if (_initialized || !_isEnabled()) return;
        try { initConnectGuestsUI(); } catch (e) { console.warn('[connect-guests] deferred init failed', e); }
      };
      window.addEventListener('augmentum:settings-loaded', retry);
      window.addEventListener('augmentum:connect-enabled', retry);
    }
    return;
  }
  _initialized = true;

  registerCommand({
    id: 'connect.manageGuests',
    label: 'Connect: Manage guests',
    hint: 'Invite outside people, see who has access, and revoke it',
    group: 'Connect',
    keywords: 'connect guest guests invite revoke access pass external visitor',
    run: () => import('./home.js').then((m) => m.openConnectHome('guests')),
    when: () => _isEnabled(),
  });

  window.augmentumConnectGuests = { open: openGuestsPanel, refresh: _refresh };
}

export async function openGuestsPanel() {
  if (!_isEnabled()) { showToast('Connect is disabled', 'warning'); return; }
  if (!_panel) _ensurePanel();
  _panel.classList.remove('hidden');
  await _refresh();
}

export function closeGuestsPanel() {
  if (_panel) _panel.classList.add('hidden');
}

/**
 * Embed the guests panel inside the Connect home's Guests section.
 * Mirrors thread-panel.js::mountMessagingInto.
 */
export async function mountGuestsInto(host) {
  if (!host) return;
  if (!_isEnabled()) { showToast('Connect is disabled', 'warning'); return; }
  if (!_panel) _ensurePanel();
  _panel.classList.add('is-embedded');
  _panel.classList.remove('hidden');
  if (_panel.parentElement !== host) host.appendChild(_panel);
  await _refresh();
}

function _ensurePanel() {
  _panel = document.createElement('div');
  _panel.className = 'connect-guests-panel hidden';
  _panel.setAttribute('role', 'dialog');
  _panel.setAttribute('aria-label', 'Manage guests');
  _panel.innerHTML = `
    <div class="connect-guests-card">
      <div class="connect-guests-header">
        <div class="connect-guests-title">${icon('users', { size: 16 })}<span>Guests</span></div>
        <div class="connect-guests-header-actions">
          <button class="connect-guests-invite" type="button">Invite a guest</button>
          <button class="connect-guests-close" type="button" aria-label="Close">&#x2715;</button>
        </div>
      </div>
      <p class="connect-guests-sub">People you've invited from outside. Revoke to cut their access instantly.</p>
      <div class="connect-guests-pending" data-pending></div>
      <div class="connect-guests-list" data-list></div>
    </div>`;
  _panel.addEventListener('click', (e) => { if (e.target === _panel) closeGuestsPanel(); });
  _panel.querySelector('.connect-guests-close').addEventListener('click', closeGuestsPanel);
  _panel.querySelector('.connect-guests-invite').addEventListener('click', _mintGuestInvite);
  document.body.appendChild(_panel);
}

async function _refresh() {
  // Pending guest registrations (awaiting this host's confirm) render above the
  // active-guests list. Without this, a guest who signs up through the portal is
  // stuck on "waiting for your host to confirm" forever — the host had no UI to
  // approve them even though the backend endpoints existed.
  await _refreshPending();

  const list = _panel?.querySelector('[data-list]');
  if (!list) return;
  list.innerHTML = '<div class="connect-guests-empty">Loading…</div>';
  let guests = [];
  try {
    const resp = await fetch('/api/connect/guests', { credentials: 'same-origin' });
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    guests = (await resp.json()).guests || [];
  } catch (err) {
    list.innerHTML = '<div class="connect-guests-empty">Could not load guests.</div>';
    return;
  }
  if (!guests.length) {
    list.innerHTML = `
      <div class="connect-guests-empty">
        <div>No guests yet.</div>
        <div class="connect-guests-empty-sub">Tap <strong>Invite a guest</strong> to send someone a link.</div>
      </div>`;
    return;
  }
  list.innerHTML = guests.map(_renderGuestRow).join('');
  for (const row of list.querySelectorAll('[data-grant]')) {
    const id = row.getAttribute('data-grant');
    row.querySelector('[data-call-toggle]')?.addEventListener('change', (e) => _toggleCall(id, e.target.checked));
    row.querySelector('[data-revoke]')?.addEventListener('click', () => _revoke(id, row.getAttribute('data-name')));
  }
}

async function _refreshPending() {
  const host = _panel?.querySelector('[data-pending]');
  if (!host) return;
  let pending = [];
  try {
    const resp = await fetch('/api/portal/pending', { credentials: 'same-origin' });
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    pending = (await resp.json()).pending || [];
  } catch (err) {
    host.innerHTML = '';  // don't block the guests list on a pending-fetch error
    return;
  }
  if (!pending.length) { host.innerHTML = ''; return; }
  host.innerHTML = `
    <div class="connect-guests-pending-head">${icon('user', { size: 14 })}<span>Waiting for you to approve (${pending.length})</span></div>
    ${pending.map(_renderPendingRow).join('')}`;
  for (const row of host.querySelectorAll('[data-reg]')) {
    const id = row.getAttribute('data-reg');
    const name = row.getAttribute('data-name');
    row.querySelector('[data-approve]')?.addEventListener('click', () => _approve(id, name));
    row.querySelector('[data-deny]')?.addEventListener('click', () => _deny(id, name));
  }
}

function _renderPendingRow(p) {
  const name = (p.display_name || '').trim() || 'Guest';
  const scopes = (p.scopes || '').split(',').filter(Boolean);
  const can = scopes.includes('call') ? 'text + calls' : 'text';
  const ip = p.requested_ip ? ` · from ${escapeHtml(p.requested_ip)}` : '';
  return `
    <div class="connect-guests-row is-pending" data-reg="${escapeHtml(p.registration_id)}" data-name="${escapeHtml(name)}">
      <div class="connect-guests-row-main">
        <div class="connect-guests-row-name">${escapeHtml(name)}</div>
        <div class="connect-guests-row-meta">Signed up · ${escapeHtml(can)}${ip}</div>
      </div>
      <button class="connect-guests-approve" type="button" data-approve>Approve</button>
      <button class="connect-guests-revoke" type="button" data-deny>Decline</button>
    </div>`;
}

async function _approve(registrationId, name) {
  try {
    const resp = await fetch(`/api/portal/registrations/${encodeURIComponent(registrationId)}/confirm`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({}),
    });
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    showToast(`${name || 'Guest'} approved — they can reach you now`, 'success');
  } catch {
    showToast('Could not approve — try again', 'error');
  }
  await _refresh();
}

async function _deny(registrationId, name) {
  const ok = await showConfirm({
    title: 'Decline this guest?',
    message: `${name || 'This guest'} won't be able to message or call you. They'll need a fresh invite to try again.`,
    confirmLabel: 'Decline',
    variant: 'danger',
  });
  if (!ok) return;
  try {
    const resp = await fetch(`/api/portal/registrations/${encodeURIComponent(registrationId)}/deny`, {
      method: 'POST',
      credentials: 'same-origin',
    });
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    showToast('Guest declined', 'info');
  } catch {
    showToast('Could not decline — try again', 'error');
  }
  await _refresh();
}

function _renderGuestRow(g) {
  const name = _guestName(g);
  const scopes = (g.scopes || '').split(',');
  const canCall = scopes.includes('call');
  const revoked = !!g.revoked;
  const last = g.last_used_at ? `Last seen ${escapeHtml(_relTime(g.last_used_at))}` : 'Not used yet';
  return `
    <div class="connect-guests-row ${revoked ? 'is-revoked' : ''}" data-grant="${escapeHtml(g.grant_id)}" data-name="${escapeHtml(name)}">
      <div class="connect-guests-row-main">
        <div class="connect-guests-row-name">${escapeHtml(name)}</div>
        <div class="connect-guests-row-meta">${revoked ? 'Access revoked' : escapeHtml(last)}</div>
      </div>
      ${revoked ? '' : `
      <label class="connect-guests-calltoggle" title="Let this guest call you, not just text">
        <input type="checkbox" data-call-toggle ${canCall ? 'checked' : ''}>
        <span>Calls</span>
      </label>
      <button class="connect-guests-revoke" type="button" data-revoke>Revoke</button>`}
    </div>`;
}

function _guestName(g) {
  // Through the shared resolver, same as every other peer label: a guest
  // whose DID local part is an auto-generated `usr_<hex>` should read as
  // "User adfce8", not as the raw id.
  const did = g.guest_did || '';
  return resolvePeerName(did) || g.guest_user_id || 'Guest';
}

async function _toggleCall(grantId, enabled) {
  const scopes = enabled ? 'text,call' : 'text';
  try {
    const resp = await fetch(`/api/connect/guests/${encodeURIComponent(grantId)}`, {
      method: 'PATCH',
      credentials: 'same-origin',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ scopes }),
    });
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    showToast(enabled ? 'Calls enabled for this guest' : 'Calls disabled', 'info');
  } catch {
    showToast('Could not update guest', 'error');
    await _refresh();
  }
}

async function _revoke(grantId, name) {
  const ok = await showConfirm({
    title: 'Revoke guest access?',
    message: `${name || 'This guest'} will lose access immediately and their saved link will stop working. This can't be undone.`,
    confirmLabel: 'Revoke access',
    variant: 'danger',
  });
  if (!ok) return;
  try {
    const resp = await fetch(`/api/connect/guests/${encodeURIComponent(grantId)}/revoke`, {
      method: 'POST',
      credentials: 'same-origin',
    });
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    showToast('Guest access revoked', 'info');
  } catch {
    showToast('Could not revoke', 'error');
  }
  await _refresh();
}

function _mintGuestInvite() {
  // Class fix (2026-07-16 guest-gateway spec): delegate to the ONE shared
  // mint component so this site asks recipient_scope and honours reach
  // metadata (blocked state, QR, #k= bundle) exactly like the other sites.
  const prior = document.querySelector('.connect-guests-linkdlg');
  if (prior) prior.remove();
  const wrap = document.createElement('div');
  wrap.className = 'connect-invite-dialog connect-guests-linkdlg';
  wrap.innerHTML = `
    <div class="connect-invite-card">
      <div class="connect-invite-title">${icon('users', { size: 16 })}<span>Guest invite</span></div>
      <p class="connect-invite-sub">They'll get a small app to text or call you — revoke it anytime from Guests.</p>
      <div class="connect-invite-mount" data-mount></div>
      <div class="connect-invite-actions"><button class="connect-invite-done" type="button">Close</button></div>
    </div>`;
  const close = () => { wrap.remove(); _refresh(); };
  wrap.addEventListener('click', (e) => { if (e.target === wrap) close(); });
  wrap.querySelector('.connect-invite-done').addEventListener('click', close);
  mountMintForm(wrap.querySelector('[data-mount]'), { role: 'guest' });
  document.body.appendChild(wrap);
  wrap.querySelector('.connect-invite-scope')?.focus();
}

function _relTime(iso) {
  // iso is a UTC "YYYY-MM-DD HH:MM:SS"; show a coarse relative string.
  const t = Date.parse((iso || '').replace(' ', 'T') + 'Z');
  if (!t) return iso;
  const mins = Math.max(0, Math.floor((Date.now() - t) / 60000));
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}
