/* connect/federation.js — the federated-PBX trust surface (D1-01).
 *
 * The security review's binding UI mandate: a federated peer's
 * attacker-controllable display name must NEVER appear without its
 * trust state next to it. `trustChip()` is that renderer and every
 * peer row here uses it. The surface has three parts:
 *
 *   1) My card    — generate a signed contact card (link + QR text) to share.
 *   2) Contacts   — pinned peers with verified / unverified / key-changed
 *                   chips, and the one-tap verification ceremony.
 *   3) Knocks      — the deny-by-default stranger inbox (accept / reject).
 *
 * All peer-supplied text goes through escapeHtml (template-literal safe).
 * Reached via the command palette ("Connect: Federation") and
 * window.augmentumFederation for console/debug.
 */

import { escapeHtml, showToast } from '../app.js';
import { registerCommand } from '../command-palette.js';

const API = '/api/fabric';

async function api(path, { method = 'GET', body } = {}) {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch { /* noop */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

/* ── D1-01: the binding trust chip ─────────────────────────────────
 * Renders the trust state that MUST accompany every peer name. The
 * unverified state is dominant (warning-coloured) on purpose — a fresh
 * pin is TOFU and the human needs to feel that before trusting it.
 */
const _TONE_CLASS = { good: 'fed-chip--ok', warn: 'fed-chip--warn', alert: 'fed-chip--alert' };

export function trustChip(peer) {
  // Prefer the server's plain-language presentation so every surface
  // shows the same professional copy.
  const p = peer && peer.presentation;
  if (p) {
    const cls = _TONE_CLASS[p.tone] || 'fed-chip--warn';
    return `<span class="fed-chip ${cls}" title="${escapeHtml(p.hint || '')}">${escapeHtml(p.icon || '')} ${escapeHtml(p.label || '')}</span>`;
  }
  if (peer && peer.key_changed) {
    return `<span class="fed-chip fed-chip--alert" title="Their security code changed — verify again before trusting.">! Identity changed</span>`;
  }
  if (peer && peer.verified) {
    return `<span class="fed-chip fed-chip--ok" title="You confirmed this is really them.">✓ Verified</span>`;
  }
  return `<span class="fed-chip fed-chip--warn" title="Tap Verify to be sure no one is impersonating them.">• Not verified yet</span>`;
}

function peerRow(peer) {
  const name = escapeHtml(peer.display_name || peer.handle || 'Unknown');
  // data-did MUST carry the real key the ceremony/verify endpoints look peers
  // up by; the visible line shows the human-readable safety code. (Bug: these
  // were conflated, so the "Safety code …" label was sent as peer_did_key and
  // the backend 404'd — the Verify ceremony was dead.)
  const key = escapeHtml(peer.peer_did_key || '');
  const label = escapeHtml(peer.safety_code ? `Safety code ${peer.safety_code}` : (peer.peer_did_key || ''));
  const canVerify = !peer.verified;
  return `
    <div class="fed-peer" data-did="${key}">
      <div class="fed-peer-name">${name} ${trustChip(peer)}</div>
      <div class="fed-peer-did">${label}</div>
      <div class="fed-peer-actions">
        ${canVerify ? `<button class="fed-btn" data-act="ceremony">Verify…</button>` : ''}
      </div>
      <div class="fed-ceremony" hidden></div>
    </div>`;
}

/* ── ceremony: show SAS words + safety number, confirm on match ──── */
async function showCeremony(rowEl, did) {
  const box = rowEl.querySelector('.fed-ceremony');
  box.hidden = false;
  box.innerHTML = `<div class="fed-muted">Computing safety number…</div>`;
  try {
    const c = await api('/peers/ceremony', { method: 'POST', body: { peer_did_key: did } });
    const words = (c.sas_words || []).map(escapeHtml).join(' &nbsp; ');
    box.innerHTML = `
      <div class="fed-sas">${words}</div>
      <div class="fed-muted">${escapeHtml(c.instruction || '')}</div>
      <div class="fed-num">${escapeHtml(c.safety_number || '')}</div>
      <div class="fed-peer-actions">
        <button class="fed-btn fed-btn--ok" data-act="confirm" data-method="sas">They match — verify</button>
        <button class="fed-btn" data-act="cancel">Cancel</button>
      </div>`;
  } catch (e) {
    box.innerHTML = `<div class="fed-err">${escapeHtml(e.message)}</div>`;
  }
}

async function confirmVerify(did, method) {
  await api('/peers/verify', { method: 'POST', body: { peer_did_key: did, method } });
  showToast('Contact verified.');
  renderContacts();
}

/* ── panels ───────────────────────────────────────────────────────── */
let _root = null;

async function renderContacts() {
  const host = _root?.querySelector('#fed-contacts');
  if (!host) return;
  host.innerHTML = `<div class="fed-muted">Loading contacts…</div>`;
  try {
    const data = await api('/peers/verified');
    const peers = data.peers || [];
    host.innerHTML = peers.length
      ? peers.map(peerRow).join('')
      : `<div class="fed-muted">No federated contacts yet. Share your card to add one.</div>`;
  } catch (e) {
    host.innerHTML = `<div class="fed-err">${escapeHtml(e.message)}</div>`;
  }
}

async function renderKnocks() {
  const host = _root?.querySelector('#fed-knocks');
  if (!host) return;
  host.innerHTML = `<div class="fed-muted">Loading knocks…</div>`;
  try {
    const data = await api('/knocks');
    const knocks = data.knocks || [];
    host.innerHTML = knocks.length
      ? knocks.map((k) => `
          <div class="fed-peer" data-knock="${escapeHtml(k.id)}">
            <div class="fed-peer-name">${escapeHtml(k.from_handle || 'stranger')} ${trustChip({ verified: false })}</div>
            <div class="fed-peer-did">${escapeHtml(k.from_did_key || '')}</div>
            ${k.intro_flagged ? `<div class="fed-err">⚠ flagged by the abuse classifier</div>` : ''}
            <div class="fed-muted">Intro hidden until you accept.</div>
            <div class="fed-peer-actions">
              <button class="fed-btn fed-btn--ok" data-act="knock-accept">Accept</button>
              <button class="fed-btn" data-act="knock-reject">Reject</button>
            </div>
          </div>`).join('')
      : `<div class="fed-muted">No pending knocks.</div>`;
  } catch (e) {
    // Endpoint optional until the knock routes are wired — fail soft.
    host.innerHTML = `<div class="fed-muted">Knock inbox unavailable.</div>`;
  }
}

async function shareMyCard() {
  const host = _root?.querySelector('#fed-mycard');
  if (!host) return;
  host.innerHTML = `<div class="fed-muted">Creating your invite…</div>`;
  try {
    const data = await api('/contact-card', { method: 'POST', body: {} });
    const s = data.share || {};
    const link = s.link || s.full_link || '';
    const code = s.code || '';
    const qrUrl = s.qr_url || '';
    // Professional share: a scannable QR + a short code people can say
    // aloud — never a raw blob.
    host.innerHTML = `
      <div class="fed-share">
        ${qrUrl ? `<img class="fed-qr" alt="Scan to connect" src="${escapeHtml(qrUrl)}" />` : ''}
        <div class="fed-share-body">
          <div class="fed-muted">Have them scan this, or enter your code:</div>
          ${code ? `<div class="fed-code">${escapeHtml(code)}</div>` : ''}
          ${link ? `<input class="fed-input" readonly value="${escapeHtml(link)}" onclick="this.select()" />` : ''}
          ${s.safety_code ? `<div class="fed-muted">Your safety code: <strong>${escapeHtml(s.safety_code)}</strong></div>` : ''}
        </div>
      </div>`;
  } catch (e) {
    host.innerHTML = `<div class="fed-err">${escapeHtml(e.message)}</div>`;
  }
}

async function acceptCardFromInput() {
  const ta = _root?.querySelector('#fed-accept-input');
  const raw = (ta?.value || '').trim();
  if (!raw) { showToast('Enter a connect code or paste an invite link first.'); return; }
  // Accept a short connect code, a link (?code= or #card=), or a raw card.
  let body = null;
  const codeInLink = raw.match(/[?&]code=([^&\s]+)/);
  const cardInLink = raw.match(/#card=([^&\s]+)/);
  if (cardInLink) body = { card_b64: cardInLink[1] };
  else if (codeInLink) body = { code: codeInLink[1] };
  else if (/^[0-9A-Za-z-]{6,12}$/.test(raw)) body = { code: raw }; // looks like a code
  else body = { card_b64: raw };
  try {
    const res = await api('/contact-card/accept', { method: 'POST', body });
    showToast(res.next_step || 'Contact added.');
    if (ta) ta.value = '';
    renderContacts();
  } catch (e) {
    showToast(e.message);
  }
}

/* ── dialog shell + event wiring ──────────────────────────────────── */
// Build the federation UI into a container. `embedded` drops the modal
// chrome (close button) since the Connect home owns dismissal.
function _buildFederationInto(container, { embedded = false } = {}) {
  _root = container;
  _root.innerHTML = `
    <div class="fed-dialog" role="dialog" aria-label="Connect Federation">
      <div class="fed-head">
        <strong>Connect Federation</strong>
        ${embedded ? '' : '<button class="fed-btn" data-act="close">Close</button>'}
      </div>
      <div class="fed-banner">Federated chats are readable by both servers unless end-to-end is on. Verify contacts before trusting them.</div>

      <section><h4>Your contact card</h4><div id="fed-mycard"></div>
        <button class="fed-btn fed-btn--ok" data-act="share">Generate my card</button></section>

      <section><h4>Add a contact</h4>
        <textarea id="fed-accept-input" class="fed-input" rows="2" placeholder="Enter a connect code (e.g. K7P2-9QX4) or paste an invite link…"></textarea>
        <button class="fed-btn fed-btn--ok" data-act="accept">Add contact</button></section>

      <section><h4>Contacts</h4><div id="fed-contacts"></div></section>
      <section><h4>Knocks <span class="fed-muted">(strangers)</span></h4><div id="fed-knocks"></div></section>
    </div>`;

  _root.addEventListener('click', async (ev) => {
    const btn = ev.target.closest('[data-act]');
    if (!btn) return;
    const act = btn.dataset.act;
    const peerEl = btn.closest('.fed-peer');
    const did = peerEl?.dataset.did;
    if (act === 'close') { if (!embedded) { _root.remove(); _root = null; } }
    else if (act === 'share') shareMyCard();
    else if (act === 'accept') acceptCardFromInput();
    else if (act === 'ceremony') showCeremony(peerEl, did);
    else if (act === 'cancel') { const b = peerEl.querySelector('.fed-ceremony'); b.hidden = true; b.innerHTML = ''; }
    else if (act === 'confirm') confirmVerify(did, btn.dataset.method || 'sas');
    else if (act === 'knock-accept') { try { await api(`/knocks/${encodeURIComponent(peerEl.dataset.knock)}/accept`, { method: 'POST' }); showToast('Knock accepted — contact pinned.'); renderKnocks(); renderContacts(); } catch (e) { showToast(e.message); } }
    else if (act === 'knock-reject') { try { await api(`/knocks/${encodeURIComponent(peerEl.dataset.knock)}/reject`, { method: 'POST' }); renderKnocks(); } catch (e) { showToast(e.message); } }
  });

  renderContacts();
  renderKnocks();
}

function openFederation() {
  if (_root) { _root.remove(); _root = null; }
  const overlay = document.createElement('div');
  overlay.className = 'fed-overlay';
  document.body.appendChild(overlay);
  _buildFederationInto(overlay, { embedded: false });
}

/**
 * Embed the federation UI inside the Connect home's Federation section.
 * Renders into a host-owned container (no modal overlay).
 */
export function mountFederationInto(host) {
  if (!host) return;
  if (_root && _root.parentElement && _root.parentElement !== host) {
    try { _root.remove(); } catch (_) {}
    _root = null;
  }
  const container = document.createElement('div');
  container.className = 'fed-embedded';
  host.appendChild(container);
  _buildFederationInto(container, { embedded: true });
}

export function initFederation() {
  registerCommand({
    id: 'connect.federation',
    title: 'Connect: Federation',
    run: () => import('./home.js').then((m) => m.openConnectHome('federation')),
  });
  window.augmentumFederation = { open: openFederation, trustChip };
}
