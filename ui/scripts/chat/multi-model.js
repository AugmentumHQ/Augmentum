/* ==========================================================================
   Chat Module — Multi-Model Fan-out (passthrough compare)

   One user turn dispatched to N models. Each response streams into its own
   compare card and lands in the session tree as a model-tagged SIBLING of
   the others (children of the same user node), so the existing branch-swipe
   navigation, persistence, and regenerate flows all work unchanged.

   Concurrency: the backend's /api/chats/fanout-plan groups models by
   backend. Models on a single-slot local engine (bundled llama-server /
   external llama.cpp) serialize within their group — a second model on the
   same process would force a mid-stream hot swap. Models on distinct
   backends (cloud providers, Ollama, the secondary engine slot) run fully
   in parallel.

   Tool calls work per-stream: each model's request is an ordinary
   /api/chat passthrough request, so the server-side tool loop runs
   independently for each model and tool activity renders as chips on
   that model's card.
   ========================================================================== */

import { app, escapeHtml, showToast } from '../app.js';
import { getSettings, save as saveSettings, openModelPickerFor } from '../settings.js';
import * as tree from './tree.js';
import { ChatStream } from './stream.js';
import { sessionStore } from './sessions.js';
import { renderMarkdown, highlightCode } from './markdown.js';

// --- module state -----------------------------------------------------------

let _extras = [];          // compare models beyond the composer's primary
let _running = false;
let _activeStreams = [];   // in-flight ChatStream instances (abort targets)
let _groupEl = null;       // the transient compare container in the DOM
let _keepNodeId = null;    // user's keep choice (applied at all-done)
let _renderMessages = null; // injected from chat/index.js
let _getRenderer = null;    // injected — primary MessageRenderer accessor

let _btnEl = null;
let _popoverEl = null;
let _wrapEl = null;

// --- selection state --------------------------------------------------------

export function getExtraModels() {
  return [..._extras];
}

export function isFanoutActive() {
  const s = getSettings();
  return !!s.multiModelEnabled && _extras.length >= 1;
}

export function isFanoutRunning() {
  return _running;
}

function _persistSelection() {
  const s = getSettings();
  s.multiModelEnabled = !!s.multiModelEnabled;
  s.multiModelModels = _extras.join(',');
  saveSettings();
  // Server-side persistence so the compare set survives refresh + restart.
  // Non-admin PUTs may be rejected — localStorage above still covers the
  // device, so this is best-effort by design.
  fetch('/api/config/tools', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({
      multi_model_enabled: !!s.multiModelEnabled,
      multi_model_models: s.multiModelModels,
    }),
  }).catch(() => { /* device-local persistence already done */ });
}

// --- composer UI -------------------------------------------------------------

const _ICON = `
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <rect x="3" y="3" width="8" height="18" rx="2"/>
    <rect x="13" y="3" width="8" height="18" rx="2"/>
  </svg>`;

/**
 * Mount the compare button + popover into the composer toolbar.
 * @param {object} hooks
 * @param {() => void} hooks.renderMessages — primary surface re-render
 * @param {() => object|null} hooks.getRenderer — primary MessageRenderer
 */
export function initMultiModel(hooks = {}) {
  _renderMessages = hooks.renderMessages || (() => {});
  _getRenderer = hooks.getRenderer || (() => null);

  const toolbar = document.getElementById('input-toolbar');
  if (!toolbar || document.getElementById('multi-model-btn')) return;

  _wrapEl = document.createElement('div');
  _wrapEl.className = 'toolbar-multi-model-wrap';
  // id + data-overflow so the mobile overflow menu (chat/toolbar/overflow.js)
  // collapses Compare into the floating ⋯ stack on phones. The wrap travels
  // as a unit so its popover stays anchored to the button.
  _wrapEl.id = 'multi-model-wrap';
  _wrapEl.setAttribute('data-overflow', '');
  _wrapEl.innerHTML = `
    <button class="icon-btn small" id="multi-model-btn" type="button"
            title="Compare models — send one message to several models">
      ${_ICON}
      <span class="mm-btn-badge hidden" id="multi-model-badge"></span>
    </button>
    <div class="mm-popover hidden" id="multi-model-popover"></div>
  `;
  // Sit next to the thinking toggle, before the toolbar divider.
  const divider = toolbar.querySelector('.input-toolbar-divider');
  if (divider) toolbar.insertBefore(_wrapEl, divider);
  else toolbar.appendChild(_wrapEl);

  _btnEl = _wrapEl.querySelector('#multi-model-btn');
  _popoverEl = _wrapEl.querySelector('#multi-model-popover');

  _btnEl.addEventListener('click', (e) => {
    e.stopPropagation();
    if (_popoverEl.classList.contains('hidden')) _openPopover();
    else _closePopover();
  });
  document.addEventListener('click', (e) => {
    if (!_popoverEl || _popoverEl.classList.contains('hidden')) return;
    if (e.target.closest('.toolbar-multi-model-wrap')) return;
    _closePopover();
  });

  // Restore the persisted compare set once settings have loaded. Settings
  // arrive asynchronously at boot, so re-read on a short delay AND on the
  // popover open path (which always re-reads).
  _restoreFromSettings();
  setTimeout(_restoreFromSettings, 2500);

  // Fan-out only applies to passthrough chat — hide elsewhere.
  document.addEventListener('augmentum:mode-changed', (e) => {
    _updateVisibility(e.detail?.mode);
  });
  _updateVisibility(app.state.mode);

  // Switching sessions mid-run: the compare container belongs to the old
  // session's DOM and the streams to its tree — abort cleanly.
  document.addEventListener('augmentum:session-changed', () => {
    if (_running) abortFanout();
  });
}

function _restoreFromSettings() {
  const s = getSettings();
  if (typeof s.multiModelModels === 'string' && s.multiModelModels.trim()) {
    _extras = s.multiModelModels.split(',').map(m => m.trim()).filter(Boolean);
  }
  _refreshBadge();
}

function _updateVisibility(mode) {
  if (!_wrapEl) return;
  const m = mode || app.state.mode || 'passthrough';
  _wrapEl.classList.toggle('hidden', m !== 'passthrough');
}

function _refreshBadge() {
  const badge = document.getElementById('multi-model-badge');
  if (!badge) return;
  const active = isFanoutActive();
  badge.classList.toggle('hidden', !active);
  if (active) badge.textContent = String(1 + _extras.length);
  _btnEl?.classList.toggle('mm-armed', active);
}

function _openPopover() {
  _renderPopover();
  _popoverEl.classList.remove('hidden');
}

function _closePopover() {
  _popoverEl?.classList.add('hidden');
}

function _renderPopover() {
  const s = getSettings();
  const primary = app.state.currentModel || 'default';
  const rows = _extras.map((m, i) => `
    <div class="mm-pop-row">
      <span class="mm-pop-model" title="${escapeHtml(m)}">${escapeHtml(m)}</span>
      <button class="mm-pop-remove" data-idx="${i}" title="Remove">&times;</button>
    </div>`).join('');

  _popoverEl.innerHTML = `
    <div class="mm-pop-header">Compare models</div>
    <label class="mm-pop-toggle">
      <input type="checkbox" id="mm-pop-enabled" ${s.multiModelEnabled ? 'checked' : ''}>
      <span>Send to all on each message</span>
    </label>
    <div class="mm-pop-row mm-pop-primary">
      <span class="mm-pop-model" title="${escapeHtml(primary)}">${escapeHtml(primary)}</span>
      <span class="mm-pop-tag">primary</span>
    </div>
    ${rows}
    <button class="mm-pop-add" id="mm-pop-add">+ Add model</button>
    <div class="mm-pop-hint">Replies stream side by side — pick the one to keep; the rest stay as swipeable branches.</div>
  `;

  _popoverEl.querySelector('#mm-pop-enabled')?.addEventListener('change', (e) => {
    const set = getSettings();
    set.multiModelEnabled = !!e.target.checked;
    _persistSelection();
    _refreshBadge();
  });
  _popoverEl.querySelectorAll('.mm-pop-remove').forEach(btn => {
    btn.addEventListener('click', () => {
      _extras.splice(Number(btn.dataset.idx), 1);
      _persistSelection();
      _refreshBadge();
      _renderPopover();
    });
  });
  _popoverEl.querySelector('#mm-pop-add')?.addEventListener('click', async () => {
    await openModelPickerFor({
      anchor: _btnEl,
      onSelect: (name) => {
        // Same model as the composer's primary is allowed — comparing a
        // model against itself is two independent samples (best-of-2).
        if (name && !_extras.includes(name)) {
          _extras.push(name);
          const set = getSettings();
          if (!set.multiModelEnabled) set.multiModelEnabled = true;
          _persistSelection();
          _refreshBadge();
        }
        _renderPopover();
        _popoverEl.classList.remove('hidden');
      },
    });
  });
}

// --- fan-out plan -------------------------------------------------------------

/**
 * Group models into serial chains that can run in parallel with each other.
 * Models sharing an exclusive (single-slot local) backend go into one chain;
 * everything else gets its own single-model chain.
 */
async function _buildRunGroups(models) {
  let plan = null;
  try {
    const resp = await fetch('/api/chats/fanout-plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ models }),
    });
    if (resp.ok) plan = (await resp.json())?.plan || null;
  } catch { /* plan is an optimization — fall back to all-parallel */ }

  // Groups are arrays of INDICES into `models` — the same model can
  // legitimately appear twice (compare a model against itself = two
  // samples), so identity has to be positional, not by name.
  if (!Array.isArray(plan)) return models.map((_, i) => [i]);

  const byModel = new Map(plan.map(p => [p.model, p]));
  const exclusiveChains = new Map(); // backend key -> chain of indices
  const groups = [];
  models.forEach((m, i) => {
    const entry = byModel.get(m);
    if (entry && entry.exclusive && entry.backend) {
      if (!exclusiveChains.has(entry.backend)) {
        const chain = [];
        exclusiveChains.set(entry.backend, chain);
        groups.push(chain);
      }
      exclusiveChains.get(entry.backend).push(i);
    } else {
      groups.push([i]);
    }
  });
  return groups;
}

// --- compare cards ------------------------------------------------------------

function _buildCompareEl(models, session) {
  const el = document.createElement('div');
  el.className = 'mm-compare';
  el.dataset.sessionId = session.id || '';
  el.innerHTML = `
    <div class="mm-compare-grid" style="--mm-cols:${Math.min(models.length, 3)}">
      ${models.map((m, i) => `
        <div class="mm-card" data-idx="${i}" data-model="${escapeHtml(m)}">
          <div class="mm-card-head">
            <span class="mm-card-model" title="${escapeHtml(m)}">${escapeHtml(m)}</span>
            <span class="mm-card-status">queued</span>
          </div>
          <div class="mm-card-think hidden"></div>
          <div class="mm-card-tools"></div>
          <div class="mm-card-body"></div>
          <div class="mm-card-foot">
            <span class="mm-card-metrics"></span>
            <button class="mm-card-keep" disabled>Use this reply</button>
          </div>
        </div>`).join('')}
    </div>
  `;
  return el;
}

function _card(idx) {
  return _groupEl?.querySelector(`.mm-card[data-idx="${idx}"]`) || null;
}

function _setStatus(idx, text, kind = '') {
  const card = _card(idx);
  if (!card) return;
  const el = card.querySelector('.mm-card-status');
  if (el) {
    el.textContent = text;
    el.dataset.kind = kind;
  }
  card.dataset.state = kind || 'live';
}

function _ensureAttached(renderer) {
  // A renderMessages() from elsewhere (memory glow, session list refresh)
  // rebuilds the message list and orphans our container — re-attach so
  // in-flight cards stay visible.
  if (_groupEl && !_groupEl.isConnected && renderer?.messagesEl) {
    renderer.messagesEl.appendChild(_groupEl);
  }
}

// --- the run -------------------------------------------------------------------

/**
 * Fan one already-rendered user turn out to the compare set.
 * Caller has appended the user node, set session.activeLeafId to it,
 * rendered messages, and set app.state.isStreaming = true.
 *
 * @param {object} session
 * @param {object} opts
 * @param {string} opts.tools — comma-separated tool list (same set every stream)
 */
export async function runFanout(session, opts = {}) {
  // The primary may legitimately appear again in the compare set —
  // same model twice = two independent samples (best-of-2). Cards and
  // results are index-keyed for exactly this reason; do NOT dedupe
  // against the primary here (that silently collapsed the set to one
  // model and fell back to a single stream whenever the user's compare
  // pick matched their composer model).
  const primary = app.state.currentModel || '';
  const models = [primary, ..._extras.filter(Boolean)];
  if (models.length < 2) return false;

  const renderer = _getRenderer ? _getRenderer() : null;
  if (!renderer || !renderer.messagesEl) return false;

  const userNodeId = session.activeLeafId;
  const userNode = session.tree?.[userNodeId];
  if (!userNode || userNode.role !== 'user') return false;

  _running = true;
  _keepNodeId = null;
  _activeStreams = [];
  _groupEl = _buildCompareEl(models, session);
  renderer.messagesEl.appendChild(_groupEl);
  renderer.scrollToBottom?.(false, true);

  // Keep buttons — delegated once per run.
  const results = new Map(); // card index -> { nodeId, ok }
  _groupEl.addEventListener('click', (e) => {
    const btn = e.target.closest('.mm-card-keep');
    if (!btn || btn.disabled) return;
    const idx = btn.closest('.mm-card')?.dataset.idx;
    const r = idx != null ? results.get(Number(idx)) : null;
    if (!r || !r.nodeId || !session.tree?.[r.nodeId]) return;
    _keepNodeId = r.nodeId;
    if (_running) {
      // Mid-run preselect: highlight now, applied at all-done.
      _groupEl.querySelectorAll('.mm-card').forEach(c => c.classList.toggle('mm-kept', c.dataset.idx === idx));
      return;
    }
    session.activeLeafId = tree.getDeepestLeaf(session, r.nodeId);
    sessionStore.syncNow(session.id);
    _groupEl?.remove();
    _groupEl = null;
    _renderMessages?.();
  });

  const groups = await _buildRunGroups(models);

  const runOne = (idx) => new Promise((resolve) => {
    const model = models[idx];
    const card = _card(idx);
    const bodyEl = card?.querySelector('.mm-card-body');
    const thinkEl = card?.querySelector('.mm-card-think');
    const toolsEl = card?.querySelector('.mm-card-tools');
    const metricsEl = card?.querySelector('.mm-card-metrics');
    let content = '';
    let thinking = '';
    const metrics = {};
    const openTools = new Map();

    _setStatus(idx, 'connecting…');

    const stream = new ChatStream({
      onContent: (text) => {
        _ensureAttached(renderer);
        content += text;
        if (bodyEl) {
          bodyEl.textContent = content;
          _setStatus(idx, 'streaming');
        }
      },
      onMeta: (meta) => {
        if (meta.model_thinking_delta) {
          thinking += meta.model_thinking_delta;
          if (thinkEl) {
            thinkEl.classList.remove('hidden');
            thinkEl.textContent = `thinking… (${thinking.length} chars)`;
          }
        }
        if (meta.stage_start) {
          const s = meta.stage_start;
          const label = s.label || s.stage || '';
          _setStatus(idx, s.detail ? `${label} · ${s.detail}` : label);
        }
        if (meta.thinking === true && !content) _setStatus(idx, 'thinking…');
        if (meta.stalled === true) _setStatus(idx, 'slow…', 'warn');
        if (meta.stalled === false && content) _setStatus(idx, 'streaming');
        if (meta.tool_start && toolsEl) {
          const chip = document.createElement('span');
          chip.className = 'mm-tool-chip';
          chip.textContent = meta.tool_start.tool || 'tool';
          toolsEl.appendChild(chip);
          if (meta.tool_start.id != null) openTools.set(meta.tool_start.id, chip);
        }
        if (meta.tool_complete && toolsEl) {
          const chip = openTools.get(meta.tool_complete.id);
          if (chip) chip.classList.add(meta.tool_complete.success === false ? 'mm-tool-fail' : 'mm-tool-ok');
        }
        if (meta.tool_call && toolsEl && !meta.tool_start) {
          // Legacy single-event path — draw a completed chip directly.
          const chip = document.createElement('span');
          chip.className = `mm-tool-chip ${meta.tool_call.success === false ? 'mm-tool-fail' : 'mm-tool-ok'}`;
          chip.textContent = meta.tool_call.tool || 'tool';
          toolsEl.appendChild(chip);
        }
        for (const k of ['tokens_per_second', 'eval_tokens', 'prompt_tokens', 'ttft_ms', 'total_duration_ms', 'context_length', 'context_used']) {
          if (meta[k] != null) metrics[k] = meta[k];
        }
      },
      onComplete: (result) => {
        const aborted = !!result?.aborted;
        _finalizeOne({ idx, model, session, userNodeId, content, thinking, metrics, results, bodyEl, metricsEl, error: null, aborted });
        resolve();
      },
      onError: (err) => {
        _finalizeOne({ idx, model, session, userNodeId, content, thinking, metrics, results, bodyEl, metricsEl, error: err, aborted: false });
        resolve();
      },
    });
    _activeStreams.push(stream);

    // All streams build context from the SAME active path — activeLeafId
    // stays parked on the user node until every model has finished, so a
    // serialized second model never sees a faster sibling's reply.
    stream.send(session, { model, mode: 'passthrough', tools: opts.tools || '' });
  });

  const chains = groups.map(group => (async () => {
    for (const cardIdx of group) {
      await runOne(cardIdx);
    }
  })());
  for (const g of groups) {
    for (let i = 1; i < g.length; i++) _setStatus(g[i], 'waiting for local engine…');
  }

  await Promise.allSettled(chains);

  // ---- all done -----------------------------------------------------------
  _running = false;
  _activeStreams = [];
  app.state.isStreaming = false;

  const ordered = models.map((_, i) => results.get(i)).filter(r => r && r.nodeId);
  const firstOk = ordered.find(r => r.ok) || ordered[0] || null;
  const keep = (_keepNodeId && session.tree?.[_keepNodeId]) ? _keepNodeId : (firstOk?.nodeId || null);

  if (!keep) {
    // Every stream failed with no content — clean up like a failed single
    // send: container gone, leaf stays on the user node, regenerate works.
    _groupEl?.remove();
    _groupEl = null;
    _renderMessages?.();
    showToast("None of the models returned a reply — try again.", 'error');
    return true;
  }

  session.activeLeafId = tree.getDeepestLeaf(session, keep);
  sessionStore.syncNow(session.id);

  if (_groupEl) {
    _groupEl.querySelectorAll('.mm-card').forEach(c => {
      const r = results.get(Number(c.dataset.idx));
      c.classList.toggle('mm-kept', !!r && r.nodeId === keep);
      const btn = c.querySelector('.mm-card-keep');
      if (btn && r && r.nodeId) btn.disabled = false;
    });
    const note = document.createElement('div');
    note.className = 'mm-compare-note';
    note.textContent = 'Kept reply is highlighted — click another to switch, or just keep typing.';
    _groupEl.appendChild(note);
  }
  return true;
}

function _finalizeOne({ idx, model, session, userNodeId, content, thinking, metrics, results, bodyEl, metricsEl, error, aborted }) {
  const hasContent = !!(content && content.trim());
  if (hasContent) {
    const node = tree.addChildNode(session, userNodeId, 'assistant', content);
    node.model_used = model;
    node.multi_model = true;
    if (thinking) node.reasoning = { thinking };
    if (metrics.tokens_per_second > 0) node.tokens_per_second = metrics.tokens_per_second;
    if (metrics.prompt_tokens > 0) node.prompt_tokens = metrics.prompt_tokens;
    if (metrics.eval_tokens > 0) node.eval_tokens = metrics.eval_tokens;
    if (metrics.context_length > 0) {
      node.context_length = metrics.context_length;
      node.context_used = metrics.context_used;
    }
    if (metrics.ttft_ms > 0) node.ttft_ms = metrics.ttft_ms;
    if (metrics.total_duration_ms > 0) node.total_duration_ms = metrics.total_duration_ms;
    if (error || aborted) {
      node.interrupted = true;
      if (error?.message) node.error_message = error.message;
    }
    results.set(idx, { nodeId: node.id, ok: !error && !aborted });
    sessionStore.save(session.id);
  } else {
    results.set(idx, { nodeId: null, ok: false });
  }

  if (bodyEl && hasContent) {
    bodyEl.innerHTML = renderMarkdown(content, { mode: 'passthrough' });
    try { highlightCode(bodyEl); } catch { /* highlight is cosmetic */ }
  }
  if (metricsEl) {
    const parts = [];
    if (metrics.tokens_per_second) parts.push(`${metrics.tokens_per_second} tok/s`);
    if (metrics.eval_tokens) parts.push(`${metrics.eval_tokens} tok`);
    if (metrics.ttft_ms) parts.push(`${(metrics.ttft_ms / 1000).toFixed(1)}s first token`);
    metricsEl.textContent = parts.join(' · ');
  }
  if (error) _setStatus(idx, error.message ? `error — ${error.message}` : 'error', 'error');
  else if (aborted) _setStatus(idx, hasContent ? 'stopped (partial kept)' : 'stopped', 'warn');
  else _setStatus(idx, 'done', 'done');
}

/** Abort every in-flight fan-out stream (Stop button / session switch). */
export function abortFanout() {
  for (const s of _activeStreams) {
    try { s.abort(); } catch { /* already settled */ }
  }
}
