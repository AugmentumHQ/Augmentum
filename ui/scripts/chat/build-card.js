/**
 * Chat build card — the unified, mode-agnostic progress surface for an
 * app build kicked off from ANY conversational surface (chat, build,
 * analytical, narrative…).
 *
 * Replaces the legacy top-bar build monitor (chat/project.js
 * `_ensureBuildMonitor`), which was purpose-built for the retired in-process
 * quickjs pipeline. Build mode now runs on the SAME coder-workspace builder as
 * the Library "Build an app" button (augmentum.builds.facade.run_build), so
 * this card consumes the SAME snapshot the Library monitor does — via
 * `GET /api/builds/{id}/stream` (SSE, poll fallback) — and renders it as a
 * compact, dockable, non-obstructive card the user can minimize while they
 * keep chatting.
 *
 * On completion it hands over the deliverable (download) AND a live door back
 * into the running coder workspace ("Open in Code"), where the preview keeps
 * running and the agent can continue the work — the recovery story the old
 * throwaway-artifact path never had.
 */

import { escapeHtml, showToast } from '../app.js';

const POLL_MS = 2500;
const MAX_CHIPS = 5;
const TERMINAL = new Set(['complete', 'completed', 'error', 'cancelled', 'canceled', 'failed']);

// build_id -> { id, name, snap, el, es, poll }
const _cards = new Map();
let _containerEl = null;

function _container() {
  if (_containerEl && document.body.contains(_containerEl)) return _containerEl;
  _containerEl = document.getElementById('chat-build-cards');
  if (!_containerEl) {
    _containerEl = document.createElement('div');
    _containerEl.id = 'chat-build-cards';
    document.body.appendChild(_containerEl);
  }
  _enableDrag(_containerEl);
  return _containerEl;
}

// Drag the whole stack by a card header, so it never has to sit on top of the
// composer/send button. Clamped to the viewport; position persists per session.
function _enableDrag(el) {
  if (el._dragInit) return;
  el._dragInit = true;
  let sx = 0, sy = 0, ox = 0, oy = 0, dragging = false;

  el.addEventListener('pointerdown', (e) => {
    const head = e.target.closest('.cbc-head');
    if (!head || e.target.closest('button')) return;  // buttons keep their click
    const rect = el.getBoundingClientRect();
    // Switch from right/top anchoring to absolute left/top for free movement.
    el.style.left = `${rect.left}px`;
    el.style.top = `${rect.top}px`;
    el.style.right = 'auto';
    el.style.bottom = 'auto';
    sx = e.clientX; sy = e.clientY; ox = rect.left; oy = rect.top;
    dragging = true;
    el.classList.add('cbc-dragging');
    head.setPointerCapture?.(e.pointerId);
    e.preventDefault();
  });
  el.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const w = el.offsetWidth || 320;
    const nx = Math.max(6, Math.min(window.innerWidth - w - 6, ox + (e.clientX - sx)));
    const ny = Math.max(6, Math.min(window.innerHeight - 44, oy + (e.clientY - sy)));
    el.style.left = `${nx}px`;
    el.style.top = `${ny}px`;
  });
  const end = () => { dragging = false; el.classList.remove('cbc-dragging'); };
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
}

/** Entry point: a build just started (meta.build_started). */
export function handleBuildStarted(payload) {
  if (!payload || !payload.build_id) return;
  const id = payload.build_id;
  if (_cards.has(id)) { _subscribe(id); return; }
  _mountCard(id, payload.name || 'Building…');
  _subscribe(id);
}

/** On page load / reconnect: re-attach cards for any in-flight builds. */
export async function recoverBuildCards() {
  try {
    const r = await fetch('/api/builds', { credentials: 'same-origin' });
    if (!r.ok) return;
    const data = await r.json().catch(() => ({}));
    const runs = Array.isArray(data) ? data : (data.runs || data.builds || []);
    for (const run of runs) {
      const id = run.build_id || run.id;
      const status = (run.status || '').toLowerCase();
      if (!id || TERMINAL.has(status)) continue;
      if ((run.kind || 'application') !== 'application') continue;
      if (_cards.has(id)) continue;
      _mountCard(id, run.name || 'Building…');
      _cards.get(id).snap = run;
      _render(id);
      _subscribe(id);
    }
  } catch { /* recovery is best-effort */ }
}

function _mountCard(id, name) {
  const el = document.createElement('div');
  el.className = 'chat-build-card is-running';
  el.dataset.buildId = id;
  _cards.set(id, { id, name, snap: { name, status: 'running' }, el, es: null, poll: null });
  _container().appendChild(el);
  _render(id);
}

function _subscribe(id) {
  const card = _cards.get(id);
  if (!card || card.es || card.poll) return;

  // Prefer SSE; the same stream the Library monitor could use.
  if (typeof EventSource !== 'undefined') {
    try {
      const es = new EventSource(`/api/builds/${encodeURIComponent(id)}/stream`);
      card.es = es;
      es.onmessage = (ev) => {
        try {
          const snap = JSON.parse(ev.data);
          card.snap = { ...card.snap, ...snap };
          _render(id);
          if (_isStop((snap.status || '').toLowerCase())) _finish(id);
        } catch { /* ignore malformed frame */ }
      };
      es.addEventListener('end', () => _finish(id));
      es.onerror = () => {
        // Network hiccup or terminal close — fall back to polling.
        es.close();
        card.es = null;
        if (!_isStop((card.snap.status || '').toLowerCase())) _startPoll(id);
      };
      return;
    } catch { /* fall through to poll */ }
  }
  _startPoll(id);
}

function _startPoll(id) {
  const card = _cards.get(id);
  if (!card || card.poll) return;
  const tick = async () => {
    try {
      const r = await fetch(`/api/builds/${encodeURIComponent(id)}`, { credentials: 'same-origin' });
      if (r.ok) {
        const data = await r.json().catch(() => ({}));
        const snap = data.run || data.snapshot || data;
        card.snap = { ...card.snap, ...snap };
        _render(id);
        if (_isStop((snap.status || '').toLowerCase())) { _finish(id); return; }
      }
    } catch { /* keep polling */ }
  };
  card.poll = setInterval(tick, POLL_MS);
  tick();
}

function _finish(id) {
  const card = _cards.get(id);
  if (!card) return;
  if (card.es) { card.es.close(); card.es = null; }
  if (card.poll) { clearInterval(card.poll); card.poll = null; }
  _render(id);
}

// A build stops (stream closes) on a terminal status OR on a paused checkpoint
// — paused is idle-awaiting-the-user, not still running.
const _isStop = (st) => TERMINAL.has(st) || st === 'paused';

const _STATUS = {
  running: { cls: 'is-running', label: 'Building' },
  complete: { cls: 'is-done', label: 'Ready' },
  completed: { cls: 'is-done', label: 'Ready' },
  error: { cls: 'is-error', label: 'Needs attention' },
  failed: { cls: 'is-error', label: 'Needs attention' },
  cancelled: { cls: 'is-cancelled', label: 'Cancelled' },
  canceled: { cls: 'is-cancelled', label: 'Cancelled' },
  paused: { cls: 'is-paused', label: 'Checkpoint' },
};

function _render(id) {
  const card = _cards.get(id);
  if (!card) return;
  const s = card.snap || {};
  const rawStatus = (s.status || 'running').toLowerCase();
  // "Keep what's built" collapses a paused checkpoint into a normal done card.
  const paused = rawStatus === 'paused' && !card.kept;
  const status = (rawStatus === 'paused' && card.kept) ? 'complete' : rawStatus;
  const meta = paused ? _STATUS.paused : (_STATUS[status] || _STATUS.running);
  const done = TERMINAL.has(status) || paused || (rawStatus === 'paused');
  const name = s.name || card.name || 'App build';

  // Unverified = shipped but the behavior gate wasn't fully satisfied.
  const quality = (s.qualityStatus || s.quality_status || '').toLowerCase();
  const unverified = status === 'complete' && quality && quality !== 'clean';

  card.el.className = `chat-build-card ${meta.cls}${unverified ? ' is-unverified' : ''}${card.el.classList.contains('is-min') ? ' is-min' : ''}`;

  const steps = Array.isArray(s.steps) ? s.steps : [];
  const chips = steps.slice(-MAX_CHIPS).map(
    (st) => `<span class="cbc-chip" title="${escapeHtml(st.preview || '')}">${escapeHtml(st.tool || '')}</span>`,
  ).join('');

  const behaviors = Array.isArray(s.behaviors) ? s.behaviors : [];
  const passed = behaviors.filter((b) => (b.status || '').toLowerCase() === 'pass').length;
  const behaviorLine = behaviors.length
    ? `<div class="cbc-behaviors"><span class="cbc-behaviors-count ${passed === behaviors.length ? 'all' : ''}">${passed}/${behaviors.length} checks</span></div>`
    : '';

  const detail = paused
    ? escapeHtml(`Reached a checkpoint${s.stop_reason ? ` (${s.stop_reason})` : ''} — it made progress but hasn't finished. Keep going, or take what's built so far.`)
    : done
      ? (s.error ? escapeHtml(String(s.error).slice(0, 160)) : (unverified ? 'Built, but not every check passed — open it to finish up.' : 'Verified in a real browser.'))
      : escapeHtml((s.currentFile || 'Working…') + (s.llmCalls ? ` · ${s.llmCalls} iter` : ''));

  const actions = paused
    ? _pausedActionsHtml()
    : (done ? _actionsHtml(s) : _runningActionsHtml(s));
  const spinner = done ? '' : '<span class="cbc-spinner" aria-hidden="true"></span>';

  card.el.innerHTML = `
    <div class="cbc-head">
      ${spinner}
      <div class="cbc-titles">
        <div class="cbc-name">${escapeHtml(name)}</div>
        <div class="cbc-status">${escapeHtml(meta.label)}${unverified ? ' · unverified' : ''}</div>
      </div>
      <button class="cbc-btn cbc-min" data-cbc-min title="Minimize" aria-label="Minimize">–</button>
      <button class="cbc-btn cbc-close" data-cbc-close title="Dismiss" aria-label="Dismiss">×</button>
    </div>
    ${!done ? '<div class="cbc-bar"><div class="cbc-bar-fill"></div></div>' : ''}
    <div class="cbc-body">
      <div class="cbc-detail">${detail}</div>
      ${chips ? `<div class="cbc-trail">${chips}</div>` : ''}
      ${behaviorLine}
      ${actions}
    </div>`;
}

function _runningActionsHtml(s) {
  const wsId = s.workspace_id || s.workspaceId || '';
  if (!wsId) return '';
  // Live viewing: jump into the coder workspace while it's still building —
  // watch the agent work and the preview update in real time.
  return `<div class="cbc-actions">
    <button class="cbc-action" data-cbc-open>Open in Code (live)</button>
  </div>`;
}

function _pausedActionsHtml() {
  // The checkpoint gate: hard limits become a choice, never a forced failure.
  return `<div class="cbc-actions">
    <button class="cbc-action cbc-primary" data-cbc-continue>Keep going</button>
    <button class="cbc-action" data-cbc-keep>Keep what's built</button>
  </div>`;
}

function _actionsHtml(s) {
  const btns = [];
  const artifactId = s.artifact_id || s.artifactId || '';
  const wsId = s.workspace_id || s.workspaceId || '';
  // The primary door: open the live workspace in Code — preview keeps running,
  // and the agent can continue the work. Falls back to rebuilding from the
  // artifact if the workspace is gone.
  if (wsId || artifactId) {
    btns.push(`<button class="cbc-action cbc-primary" data-cbc-open>Open in Code</button>`);
  }
  if (artifactId) {
    btns.push(`<a class="cbc-action" href="/api/artifacts/${encodeURIComponent(artifactId)}/download" download>Download</a>`);
  }
  return btns.length ? `<div class="cbc-actions">${btns.join('')}</div>` : '';
}

// Delegated actions (one listener for the whole stack).
document.addEventListener('click', async (ev) => {
  const cardEl = ev.target.closest('.chat-build-card');
  if (!cardEl) return;
  const id = cardEl.dataset.buildId;
  const card = _cards.get(id);
  if (!card) return;

  if (ev.target.closest('[data-cbc-close]')) {
    _finish(id);
    cardEl.remove();
    _cards.delete(id);
    return;
  }
  if (ev.target.closest('[data-cbc-min]')) {
    cardEl.classList.toggle('is-min');
    return;
  }
  if (ev.target.closest('[data-cbc-keep]')) {
    // Take what's built at the checkpoint — collapse the gate into a done card.
    card.kept = true;
    _render(id);
    return;
  }
  if (ev.target.closest('[data-cbc-continue]')) {
    ev.preventDefault();
    await _continueBuild(card);
    return;
  }
  if (ev.target.closest('[data-cbc-open]')) {
    ev.preventDefault();
    await _openInCode(card);
  }
});

// Resume a paused build on its existing workspace, then re-attach the stream.
async function _continueBuild(card) {
  try {
    const r = await fetch(`/api/builds/${encodeURIComponent(card.id)}/resume`, {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    card.kept = false;
    card.snap = { ...card.snap, status: 'running', awaiting_continue: false };
    card.es = null;
    card.poll = null;
    _render(card.id);
    _subscribe(card.id);
    showToast('Continuing the build…', 'info');
  } catch (err) {
    showToast(`Couldn't continue: ${err.message || err}`, 'error');
  }
}

async function _openInCode(card) {
  const s = card.snap || {};
  const wsId = s.workspace_id || s.workspaceId || '';
  try {
    const coder = await import('../coder.js');
    if (wsId && typeof coder.openWorkspaceById === 'function') {
      await coder.openWorkspaceById(wsId);
      document.querySelector('[data-mode="coder"]')?.click();
      showToast('Opened the build in Code — preview is live there', 'success');
      return;
    }
    // No live workspace — rebuild one from the published artifact.
    const buildId = card.id;
    showToast('Opening in Code…', 'info');
    const r = await fetch(`/api/builds/${encodeURIComponent(buildId)}/open-in-code`, {
      method: 'POST', credentials: 'same-origin',
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    if (data.workspace_id && typeof coder.openWorkspaceById === 'function') {
      await coder.openWorkspaceById(data.workspace_id);
    }
    document.querySelector('[data-mode="coder"]')?.click();
    showToast('Code workspace ready', 'success');
  } catch (err) {
    showToast(`Couldn't open in Code: ${err.message || err}`, 'error');
  }
}
