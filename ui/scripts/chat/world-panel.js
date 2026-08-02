/* World panel — card-declared world systems (trackers / dice / sheet).
 *
 * Spec: docs/superpowers/specs/2026-07-15-world-system-manifest-design.md
 *
 * Absence means invisible: the drawer mounts ONLY when
 * GET /api/narrative/world/{session} reports an active manifest. Plain
 * cards never see any of this UI. All strings are world-agnostic — the
 * manifest supplies the flavor.
 */

import { escapeHtml } from '../app.js';

let _sessionId = '';
let _state = null;      // last fetched world state (null = inactive)
let _collapsed = localStorage.getItem('worldPanelCollapsed') === '1';
let _suggestions = [];

// ---------------------------------------------------------------------------
// Data

async function fetchWorld(sessionId) {
  try {
    const r = await fetch(`/api/narrative/world/${encodeURIComponent(sessionId)}`);
    if (!r.ok) return null;
    const data = await r.json();
    return data && data.active ? data : null;
  } catch { return null; }
}

/** Refresh (or mount/unmount) the drawer for a session. */
export async function ensureWorldPanel(sessionId) {
  if (!sessionId) { unmount(); return; }
  _sessionId = sessionId;
  _state = await fetchWorld(sessionId);
  if (!_state) { unmount(); return; }
  try {
    const r = await fetch(`/api/narrative/world/${encodeURIComponent(sessionId)}/suggestions`);
    _suggestions = r.ok ? ((await r.json()).items || []) : [];
  } catch { _suggestions = []; }
  render();
}

async function resolveSuggestion(tracker, accept) {
  try {
    await fetch(`/api/narrative/world/${encodeURIComponent(_sessionId)}/suggestions/resolve`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tracker, accept }),
    });
  } catch { /* chip just re-fetches below */ }
  ensureWorldPanel(_sessionId);
}

// ---------------------------------------------------------------------------
// Drawer

function unmount() {
  document.getElementById('world-panel')?.remove();
  _state = null;
}

function render() {
  if (!_state) return;
  let el = document.getElementById('world-panel');
  if (!el) {
    el = document.createElement('div');
    el.id = 'world-panel';
    el.className = 'world-panel';
    document.body.appendChild(el);
  }
  const rows = (_state.trackers || []).map(t => trackerRowHtml(t)).join('');
  el.innerHTML = `
    <div class="world-panel-header" id="world-panel-toggle">
      <span class="world-panel-title">${escapeHtml(_state.world || 'World')}</span>
      <span class="world-panel-caret">${_collapsed ? '▸' : '▾'}</span>
    </div>
    <div class="world-panel-body" style="${_collapsed ? 'display:none' : ''}">
      ${rows || '<div class="world-panel-empty">No trackers yet</div>'}
      ${(_suggestions || []).map(sg => `
        <div class="world-suggestion" data-sg="${escapeHtml(sg.tracker)}">
          <span class="world-suggestion-text">${escapeHtml(sg.label || sg.tracker)} → <b>${escapeHtml(String(sg.to !== null && sg.to !== undefined ? sg.to : (sg.delta > 0 ? '+' + sg.delta : sg.delta)))}</b>?</span>
          <span class="world-suggestion-btns">
            <button class="world-chip-btn" data-sg-accept="${escapeHtml(sg.tracker)}" title="${escapeHtml(sg.reason || '')}">✓</button>
            <button class="world-chip-btn" data-sg-dismiss="${escapeHtml(sg.tracker)}">✕</button>
          </span>
        </div>`).join('')}
      <div class="world-panel-actions">
        ${_state.sheet_command ? `<button class="world-btn" data-act="sheet">Sheet</button>` : ''}
        ${_state.player_roller ? `<button class="world-btn" data-act="roll">🎲 Roll</button>` : ''}
      </div>
    </div>`;
  el.querySelector('#world-panel-toggle').onclick = () => {
    _collapsed = !_collapsed;
    localStorage.setItem('worldPanelCollapsed', _collapsed ? '1' : '0');
    render();
  };
  el.querySelectorAll('[data-tracker]').forEach(row => {
    row.addEventListener('click', () => beginCorrect(row.dataset.tracker, row.dataset.owner || ''));
  });
  el.querySelector('[data-act="sheet"]')?.addEventListener('click', () => {
    const input = document.getElementById('chat-input');
    if (input) { input.value = _state.sheet_command || '/status'; input.form?.requestSubmit?.(); }
  });
  el.querySelector('[data-act="roll"]')?.addEventListener('click', promptRoll);
  el.querySelectorAll('[data-sg-accept]').forEach(b =>
    b.addEventListener('click', (e) => { e.stopPropagation(); resolveSuggestion(b.dataset.sgAccept, true); }));
  el.querySelectorAll('[data-sg-dismiss]').forEach(b =>
    b.addEventListener('click', (e) => { e.stopPropagation(); resolveSuggestion(b.dataset.sgDismiss, false); }));
}

function trackerRowHtml(t) {
  const owner = t.owner ? `${escapeHtml(t.owner)} · ` : '';
  let value;
  if (t.kind === 'band' && Array.isArray(t.bands)) {
    const idx = Math.max(0, t.bands.indexOf(t.value));
    const pct = t.bands.length > 1 ? Math.round(100 * (1 - idx / (t.bands.length - 1))) : 100;
    value = `
      <div class="world-band">
        <div class="world-band-fill" style="width:${pct}%"></div>
        <span class="world-band-label">${escapeHtml(String(t.value))}</span>
      </div>`;
  } else {
    value = `<span class="world-value">${escapeHtml(String(t.value))}</span>`;
  }
  return `
    <div class="world-tracker" data-tracker="${escapeHtml(t.id)}" data-owner="${escapeHtml(t.owner || '')}" title="Tap to correct">
      <span class="world-tracker-label">${owner}${escapeHtml(t.label)}</span>${value}
    </div>`;
}

// User correction (spec D1): provenance=user, sticky lock server-side.
async function beginCorrect(trackerId, owner) {
  const t = (_state.trackers || []).find(x => x.id === trackerId && (x.owner || '') === owner);
  if (!t) return;
  let to = null; let delta = null;
  if (t.kind === 'band') {
    const pick = prompt(`Set ${t.label} to one of:\n${t.bands.join(' / ')}`, String(t.value));
    if (pick === null) return;
    to = pick.trim();
  } else if (t.kind === 'counter') {
    const raw = prompt(`Set ${t.label} (current: ${t.value})`, String(t.value));
    if (raw === null) return;
    to = Number(raw);
    if (Number.isNaN(to)) return;
    delta = to - Number(t.value);
    to = null;
  } else {
    const raw = prompt(`Set ${t.label}`, String(t.value));
    if (raw === null) return;
    to = t.kind === 'flag' ? /^(1|true|yes|on)$/i.test(raw.trim()) : Number(raw);
  }
  try {
    const r = await fetch(`/api/narrative/world/${encodeURIComponent(_sessionId)}/correct`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tracker: trackerId, owner, to, delta }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      console.warn('world correct failed', err);
    }
  } catch (e) { console.warn('world correct failed', e); }
  ensureWorldPanel(_sessionId);
}

// Player roll: real server dice, result lands in the composer so the
// user sends it as their action beat (model narrates around it).
async function promptRoll() {
  const expr = prompt('Roll dice (NdM+K):', 'd20');
  if (!expr) return;
  try {
    const r = await fetch(`/api/narrative/world/${encodeURIComponent(_sessionId)}/roll`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expression: expr }),
    });
    if (!r.ok) return;
    const data = await r.json();
    const input = document.getElementById('chat-input');
    if (input && data.total !== undefined) {
      const prefix = input.value ? input.value.replace(/\s+$/, '') + ' ' : '';
      input.value = `${prefix}(🎲 ${data.expression}: ${data.total})`;
      input.focus();
    }
  } catch { /* dice are optional; composer untouched on failure */ }
}

// ---------------------------------------------------------------------------
// Inline event cards (streaming + historical replay)

/** Render event cards into a message element. Used for both live streams
 *  and node replay on session load. */
export function renderWorldEventCards(events, messageEl) {
  if (!Array.isArray(events) || !events.length || !messageEl) return;
  const content = messageEl.querySelector('.message-content') || messageEl;
  let host = content.querySelector('.world-event-host');
  if (!host) {
    host = document.createElement('div');
    host.className = 'world-event-host';
    content.insertBefore(host, content.firstChild);
  }
  for (const ev of events) {
    const key = JSON.stringify(ev);
    if (host.querySelector(`[data-ev-key="${CSS.escape(key.slice(0, 80))}"]`)) continue;
    const card = document.createElement('div');
    card.className = 'world-event-card';
    card.dataset.evKey = key.slice(0, 80);
    card.innerHTML = eventCardHtml(ev);
    host.appendChild(card);
  }
}

function eventCardHtml(ev) {
  if (ev.kind === 'roll') {
    const dc = ev.dc !== undefined && ev.dc !== null ? ` vs DC ${escapeHtml(String(ev.dc))}` : '';
    const outcome = ev.outcome
      ? ` <span class="world-roll-${escapeHtml(ev.outcome)}">${escapeHtml(ev.outcome)}</span>` : '';
    const check = ev.check ? `${escapeHtml(ev.check)} — ` : '';
    return `🎲 ${check}${escapeHtml(ev.expression || '')} → <b>${escapeHtml(String(ev.total))}</b>${dc}${outcome}`;
  }
  if (ev.kind === 'tracker_shift') {
    const owner = ev.owner ? `${escapeHtml(ev.owner)} · ` : '';
    const reason = ev.reason ? ` <span class="world-event-reason">(${escapeHtml(ev.reason)})</span>` : '';
    return `◈ ${owner}${escapeHtml(String(ev.tracker))} → <b>${escapeHtml(String(ev.value))}</b>${reason}`;
  }
  if (ev.kind === 'sheet' && ev.sheet) {
    const s = ev.sheet;
    const sections = (s.sections || []).map(sec => `
      <div class="world-sheet-section">
        <div class="world-sheet-section-title">${escapeHtml(sec.label)}</div>
        ${sec.rows.map(r => `<div class="world-sheet-row"><span>${escapeHtml(r.label)}</span><span>${escapeHtml(String(r.value))}</span></div>`).join('')}
      </div>`).join('');
    return `<div class="world-sheet"><div class="world-sheet-title">${escapeHtml(s.world || 'World')} · ${escapeHtml(s.owner || '')}</div>${sections}</div>`;
  }
  return '';
}

/** Live-stream entry point from chat index _onMeta. */
export function handleWorldEvents(events) {
  // Attach to the most recent assistant message in the DOM.
  const msgs = document.querySelectorAll('.message.assistant');
  const last = msgs[msgs.length - 1];
  if (last) renderWorldEventCards(events, last);
  // Tracker values changed server-side — refresh the drawer.
  if (_sessionId) ensureWorldPanel(_sessionId);
}

// ---------------------------------------------------------------------------
// Styles (scoped, injected once) + session hook

const CSS_TEXT = `
#world-panel{position:fixed;right:12px;bottom:96px;width:230px;z-index:40;
  background:var(--bg-secondary,#1c1c22);border:1px solid var(--border-color,#333);
  border-radius:10px;font-size:12px;overflow:hidden;box-shadow:0 4px 14px rgba(0,0,0,.35)}
.world-panel-header{display:flex;justify-content:space-between;align-items:center;
  padding:7px 10px;cursor:pointer;user-select:none;font-weight:600}
.world-panel-body{padding:6px 10px 10px}
.world-panel-empty{opacity:.6;padding:4px 0}
.world-tracker{display:flex;justify-content:space-between;align-items:center;
  gap:8px;padding:4px 0;cursor:pointer}
.world-tracker:hover{opacity:.85}
.world-tracker-label{opacity:.85;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.world-band{position:relative;flex:0 0 92px;height:14px;border-radius:7px;
  background:var(--bg-tertiary,#2a2a32);overflow:hidden}
.world-band-fill{position:absolute;inset:0 auto 0 0;background:var(--accent-color,#4a8);opacity:.55}
.world-band-label{position:absolute;inset:0;display:flex;align-items:center;
  justify-content:center;font-size:10px}
.world-panel-actions{display:flex;gap:6px;margin-top:8px}
.world-btn{flex:1;padding:4px 0;font-size:11px;border-radius:6px;cursor:pointer;
  background:var(--bg-tertiary,#2a2a32);border:1px solid var(--border-color,#333);color:inherit}
.world-suggestion{display:flex;justify-content:space-between;align-items:center;
  gap:6px;margin-top:6px;padding:5px 8px;border-radius:8px;font-size:11px;
  background:rgba(120,160,255,.08);border:1px dashed var(--border-color,#446)}
.world-suggestion-btns{display:flex;gap:4px}
.world-chip-btn{width:22px;height:22px;border-radius:6px;cursor:pointer;
  background:var(--bg-tertiary,#2a2a32);border:1px solid var(--border-color,#333);color:inherit}
.world-event-host{display:flex;flex-direction:column;gap:4px;margin:2px 0 8px}
.world-event-card{padding:5px 10px;border-radius:8px;font-size:12px;
  background:var(--bg-secondary,#1c1c22);border:1px solid var(--border-color,#333)}
.world-roll-success{color:#5c5}
.world-roll-failure{color:#d66}
.world-event-reason{opacity:.6}
.world-sheet-title{font-weight:600;margin-bottom:4px}
.world-sheet-section-title{opacity:.7;margin:6px 0 2px;font-size:11px;text-transform:uppercase}
.world-sheet-row{display:flex;justify-content:space-between;gap:12px;padding:1px 0}
`;

function injectStyles() {
  if (document.getElementById('world-panel-css')) return;
  const s = document.createElement('style');
  s.id = 'world-panel-css';
  s.textContent = CSS_TEXT;
  document.head.appendChild(s);
}

injectStyles();
document.addEventListener('augmentum:turn-stats', () => {
  if (!_sessionId || !_state) return;
  setTimeout(() => ensureWorldPanel(_sessionId), 2500);
  setTimeout(() => ensureWorldPanel(_sessionId), 9000);
});
document.addEventListener('augmentum:session-changed', (e) => {
  const id = e.detail?.sessionId || e.detail?.id || '';
  ensureWorldPanel(id);
});
