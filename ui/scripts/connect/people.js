// ui/scripts/connect/people.js
//
// The Connect home's People section: a searchable directory of everyone
// reachable on this machine (+ saved contacts), each row a one-tap
// Message or Call. Renders inline into the home content region.
//
// Reuses the same data the messaging picker uses
// (messages.js::listDirectory / searchPeers / listContacts) and live
// presence from client.js. The host wires onMessage/onCall so People
// can hand off to the Chats section or the dialer without importing the
// home (avoids a cycle).

import { escapeHtml } from '../app.js';
import { icon } from './icons.js';
import { getPeerStatus, seedPeerStatus } from './client.js';
import {
  listContacts, listDirectory, peerSubtitle, resolvePeerName, searchPeers,
} from './messages.js';

let _host = null;
let _cbs = {};
let _searchSeq = 0;
let _debounce = null;

export async function mountPeopleInto(host, { onMessage, onCall } = {}) {
  if (!host) return;
  _host = host;
  _cbs = { onMessage, onCall };
  host.innerHTML = `
    <div class="connect-people">
      <div class="connect-people-searchbar">
        <span class="connect-people-search-icon" aria-hidden="true">${icon('search', { size: 16 })}</span>
        <input type="text" class="connect-people-search"
               placeholder="Search people on this machine…"
               autocomplete="off" spellcheck="false" aria-label="Search people" />
      </div>
      <div class="connect-people-list" aria-label="People"></div>
    </div>
  `;
  const input = host.querySelector('.connect-people-search');
  if (input) {
    input.addEventListener('input', () => {
      if (_debounce) clearTimeout(_debounce);
      _debounce = setTimeout(() => _renderList(input.value), 220);
    });
  }
  await _renderList('');
}

async function _renderList(query) {
  if (!_host) return;
  const listEl = _host.querySelector('.connect-people-list');
  if (!listEl) return;
  const q = (query || '').trim();
  const seq = ++_searchSeq;

  let people = [];
  let contacts = [];
  try {
    if (q) {
      people = (await searchPeers(q)).people || [];
    } else {
      const [dir, cts] = await Promise.all([
        listDirectory().catch(() => ({ people: [] })),
        listContacts({ includeBlocked: false }).catch(() => []),
      ]);
      people = dir.people || [];
      contacts = cts || [];
    }
  } catch {
    if (seq !== _searchSeq) return;
    listEl.innerHTML = `<div class="connect-people-empty">Couldn’t load people.</div>`;
    return;
  }
  if (seq !== _searchSeq) return;  // a newer search superseded this one

  const seen = new Set();
  const rows = [];
  for (const p of people) {
    if (!p.peer_did || seen.has(p.peer_did)) continue;
    seen.add(p.peer_did);
    // The directory/search response carries an authoritative `online` snapshot;
    // seed it into the presence cache so rows don't render a stale "offline"
    // before the signaling WS delivers live presence (later WS events win).
    // Mirrors messages.js::refreshDisplayNameCache.
    seedPeerStatus(p.peer_did, p.online ? 'online' : 'offline');
    // `sub` is a disambiguator, not an identifier dump — a handle if the
    // peer has one, else the fabric instance, else nothing.
    rows.push({
      did: p.peer_did,
      name: p.display_name || resolvePeerName(p.peer_did),
      sub: p.handle || peerSubtitle(p.peer_did),
    });
  }
  for (const c of contacts) {
    if (!c.peer_did || seen.has(c.peer_did)) continue;
    seen.add(c.peer_did);
    rows.push({
      did: c.peer_did,
      name: (c.peer_display_name || '').trim() || resolvePeerName(c.peer_did),
      sub: peerSubtitle(c.peer_did),
    });
  }

  if (!rows.length) {
    listEl.innerHTML = `<div class="connect-people-empty">${q ? 'No one matches.' : 'No one to show yet.'}</div>`;
    return;
  }

  listEl.innerHTML = rows.map((r) => {
    const presence = (r.did ? getPeerStatus(r.did) : 'offline') || 'offline';
    const initial = (r.name || '?').trim().charAt(0).toUpperCase() || '?';
    return `
      <div class="connect-people-row" data-peer-did="${escapeHtml(r.did)}" role="button" tabindex="0">
        <span class="connect-people-avatar" aria-hidden="true">${escapeHtml(initial)}
          <span class="connect-people-presence" data-presence="${escapeHtml(presence)}" title="${escapeHtml(presence)}"></span>
        </span>
        <div class="connect-people-text">
          <div class="connect-people-name">${escapeHtml(r.name)}</div>
          ${r.sub ? `<div class="connect-people-sub">${escapeHtml(r.sub)}</div>` : ''}
        </div>
        <div class="connect-people-actions">
          <button class="connect-people-act" data-act="message" title="Message" aria-label="Message ${escapeHtml(r.name)}">${icon('message', { size: 16 })}</button>
          <button class="connect-people-act" data-act="call" title="Call" aria-label="Call ${escapeHtml(r.name)}">${icon('phone', { size: 16 })}</button>
        </div>
      </div>
    `;
  }).join('');

  for (const row of listEl.querySelectorAll('.connect-people-row')) {
    const did = row.dataset.peerDid;
    const msgBtn = row.querySelector('[data-act="message"]');
    const callBtn = row.querySelector('[data-act="call"]');
    if (msgBtn) msgBtn.addEventListener('click', (e) => { e.stopPropagation(); _cbs.onMessage?.(did); });
    if (callBtn) callBtn.addEventListener('click', (e) => { e.stopPropagation(); _cbs.onCall?.(did); });
    row.addEventListener('click', () => _cbs.onMessage?.(did));
    row.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); _cbs.onMessage?.(did); }
    });
  }
}
