// ui/scripts/connect/home.js
//
// The unified Connect home — one master-detail surface reached from the
// header voice-button long-press ("Connect" row) and the six
// `Connect: …` palette commands. A left section rail switches the
// content region between the six pages plus a settings footer:
//
//   Chats · Calls · People · Guests · Invite · Federation   (+ ⚙ settings)
//
// Design: docs/superpowers/specs/2026-06-23-connect-unified-home-design.md
//
// Phase 1: the Chats section renders INLINE by embedding the existing
// messaging master-detail (thread-panel.js::mountMessagingInto) — all of
// its composer / edit / voice-note logic is reused untouched. The other
// five sections bridge to their existing floating panels for now (a
// "classic view" hint + auto-open) so nothing is unreachable; Phases 2–3
// inline them into the content region.

import { showToast } from '../app.js';
import { getSettings } from '../settings.js';
import { icon } from './icons.js';
import { mountMessagingInto, openThreadForPeer } from './thread-panel.js';
import { mountCallsInto } from './calls-panel.js';
import { mountPeopleInto } from './people.js';

// Hand-offs People uses so it doesn't import the home (avoids a cycle).
async function _goToPeerChat(peerDid) {
  if (!peerDid) return;
  await _selectSection('chats');
  await openThreadForPeer(peerDid);
}
async function _callPeer(peerDid) {
  if (!peerDid) return;
  try {
    const { startCall } = await import('./ui.js');
    await startCall(peerDid);
  } catch (err) { console.warn('connect home: call peer failed', err); }
}

// Section registry.
//   embed: 'panel'  — relocate an existing floating panel into the host
//                     (Chats, Calls), preserved across switches.
//   embed: 'people' — custom inline view rendered into the host.
//   mode:  'legacy' — bridge to a classic opener until inlined (P3).
const SECTIONS = [
  { id: 'chats',      label: 'Chats',      glyph: 'message', embed: 'panel',
    mount: (host) => mountMessagingInto(host) },
  { id: 'calls',      label: 'Calls',      glyph: 'phone',   embed: 'panel',
    mount: (host) => mountCallsInto(host) },
  { id: 'people',     label: 'People',     glyph: 'users',   embed: 'people',
    mount: (host) => mountPeopleInto(host, { onMessage: _goToPeerChat, onCall: _callPeer }) },
  { id: 'guests',     label: 'Guests',     glyph: 'user',    embed: 'panel',
    mount: (host) => import('./guests-panel.js').then((m) => m.mountGuestsInto?.(host)) },
  { id: 'invite',     label: 'Invite',     glyph: 'send',    embed: 'custom',
    mount: (host) => import('./ui.js').then((m) => m.mountInviteInto?.(host)) },
  { id: 'federation', label: 'Federation', glyph: 'signal',  embed: 'custom',
    mount: (host) => import('./federation.js').then((m) => m.mountFederationInto?.(host)) },
];

// Classes of the relocatable floating panels (hidden, not destroyed,
// when switching between panel-embed sections so the switch is instant).
const PANEL_SELECTOR = '.connect-thread-panel, .connect-calls-panel, .connect-guests-panel';

let _home = null;
let _activeSection = 'chats';

function _isEnabled() {
  const s = getSettings?.();
  return !!(s && s.connectEnabled);
}

/**
 * Open the Connect home and route to a section.
 * @param {string} section one of the SECTIONS ids (default 'chats').
 */
export async function openConnectHome(section = 'chats') {
  if (!_isEnabled()) {
    showToast('Connect is disabled', 'warning');
    return;
  }
  _ensureHome();
  // .hidden uses display:none, so the slide-in keyframe only runs on the
  // hidden → visible transition (same pattern as the thread panel).
  _home.classList.remove('hidden');
  const valid = SECTIONS.some((s) => s.id === section) ? section : 'chats';
  await _selectSection(valid);
}

export function closeConnectHome() {
  if (!_home) return;
  _home.classList.add('hidden');
}

function _ensureHome() {
  if (_home) return _home;
  const el = document.createElement('div');
  el.className = 'connect-home hidden';
  el.setAttribute('role', 'dialog');
  el.setAttribute('aria-modal', 'false');
  el.setAttribute('aria-label', 'Connect');

  const rail = SECTIONS.map((s) => `
    <button class="connect-home-rail-item" type="button" role="tab"
            data-section="${s.id}" aria-selected="false"
            title="${s.label}" aria-label="${s.label}">
      <span class="connect-home-rail-icon">${icon(s.glyph, { size: 20 })}</span>
      <span class="connect-home-rail-label">${s.label}</span>
    </button>`).join('');

  el.innerHTML = `
    <div class="connect-home-card">
      <nav class="connect-home-rail" role="tablist" aria-label="Connect sections">
        <div class="connect-home-rail-brand">Connect</div>
        <div class="connect-home-rail-items">${rail}</div>
        <button class="connect-home-rail-item connect-home-rail-settings" type="button"
                data-section="settings" title="Connect settings" aria-label="Connect settings">
          <span class="connect-home-rail-icon">${icon('settings', { size: 20 })}</span>
          <span class="connect-home-rail-label">Settings</span>
        </button>
      </nav>
      <div class="connect-home-main">
        <div class="connect-home-topbar">
          <div class="connect-home-title"></div>
          <button class="connect-home-close" type="button" aria-label="Close Connect">&#x2715;</button>
        </div>
        <div class="connect-home-content"></div>
      </div>
    </div>
  `;
  document.body.appendChild(el);
  _home = el;

  el.querySelector('.connect-home-close')
    .addEventListener('click', closeConnectHome);

  for (const btn of el.querySelectorAll('.connect-home-rail-item')) {
    btn.addEventListener('click', () => {
      const id = btn.dataset.section;
      if (id === 'settings') { _openConnectSettings(); return; }
      _selectSection(id);
    });
  }

  // Escape closes the home — but not while focus is in a text field
  // (there Esc means "cancel this input": composer, search, edit modal),
  // and not when an inner overlay (edit/contact picker) is open.
  el.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Escape') return;
    if (ev.target.closest('input, textarea, [contenteditable="true"]')) return;
    if (el.querySelector('.connect-edit-overlay, .connect-contact-picker:not(.hidden)')) return;
    ev.stopPropagation();
    closeConnectHome();
  });

  return _home;
}

async function _selectSection(id) {
  if (!_home) return;
  _activeSection = id;
  const section = SECTIONS.find((s) => s.id === id) || SECTIONS[0];

  // Rail highlight.
  for (const btn of _home.querySelectorAll('.connect-home-rail-item')) {
    const on = btn.dataset.section === id;
    btn.classList.toggle('is-active', on);
    btn.setAttribute('aria-selected', on ? 'true' : 'false');
  }
  const titleEl = _home.querySelector('.connect-home-title');
  if (titleEl) titleEl.textContent = section.label;

  const content = _home.querySelector('.connect-home-content');
  if (!content) return;
  content.dataset.section = id;

  // Hide any relocatable panels from other sections so only the active
  // one shows (kept in the DOM for instant return).
  for (const p of content.querySelectorAll(PANEL_SELECTOR)) p.classList.add('hidden');

  if (section.embed === 'panel') {
    content.classList.add('is-embedded-host');
    // Drop stray non-panel markup (legacy placeholder / People view) but
    // keep the relocatable panels (mount re-appends + un-hides its own).
    for (const child of Array.from(content.children)) {
      if (!child.matches(PANEL_SELECTOR)) child.remove();
    }
    await section.mount(content);
    return;
  }

  if (section.embed === 'people' || section.embed === 'custom') {
    content.classList.add('is-embedded-host');
    // Custom inline views own the host; clear it first (this drops any
    // embedded panels, which re-mount + reload on return — cheap).
    content.innerHTML = '';
    await section.mount(content);
    return;
  }

  // Legacy bridge (fallback; no sections use this after P3). Show a hint
  // and open the classic panel.
  content.classList.remove('is-embedded-host');
  content.innerHTML = `
    <div class="connect-home-legacy">
      <div class="connect-home-legacy-icon">${icon(section.glyph, { size: 32 })}</div>
      <div class="connect-home-legacy-title">${section.label}</div>
      <div class="connect-home-legacy-desc">${section.desc}</div>
      <button class="connect-home-legacy-open" type="button">Open ${section.label}</button>
      <div class="connect-home-legacy-note">Opens in the classic view for now.</div>
    </div>
  `;
  const openBtn = content.querySelector('.connect-home-legacy-open');
  const fire = () => { try { section.open?.(); } catch (err) { console.warn('connect home: open', id, err); } };
  if (openBtn) openBtn.addEventListener('click', fire);
  // Auto-open on first navigation so the section isn't a dead end.
  fire();
}

function _openConnectSettings() {
  // Visibility / Connect prefs live in the main Settings surface today;
  // bridge there until a dedicated inline panel lands (P5).
  try {
    if (typeof window.openSettings === 'function') { window.openSettings('connect'); return; }
  } catch (_) { /* fall through to the toast fallback below */ }
  showToast('Connect settings live in Settings → Account', 'info');
}
