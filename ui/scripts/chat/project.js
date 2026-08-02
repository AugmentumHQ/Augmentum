/* ==========================================================================
   Chat Module — Project Builder
   Handles the app builder pipeline: build monitor, project progress streaming,
   project card rendering, error capture/auto-fix, iteration, and file expansion.
   ========================================================================== */

import { app, escapeHtml, extractErrorMessage, showToast } from '../app.js';
import { icons } from './constants.js';
import { blockFingerprint, safeHighlightElement } from './markdown.js';
import {
  assembleProject as _sharedAssemble,
  getLastSourceMap as _getLastSourceMap,
  buildPreviewSrcdoc as _sharedBuildPreviewSrcdoc,
} from '../assemble.js';

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------

let _pendingProject = null;
let _lastAssemblySourceMap = [];
let _buildMonitor = null;
// Either a setInterval id (legacy poll) OR an EventSource (SSE feed).
// _stopBuildStatusFeed handles both — never call clearInterval directly.
let _buildMonitorPoll = null;
let _titleFlashInterval = null;
let _iterateInFlight = false;

function _stopBuildStatusFeed() {
  if (!_buildMonitorPoll) return;
  if (typeof _buildMonitorPoll.close === 'function') _buildMonitorPoll.close();
  else clearInterval(_buildMonitorPoll);
  _buildMonitorPoll = null;
}

// Header subtitle: "model · 1m 23s" — driven by a 1s tick when the build
// is running so the clock advances smoothly between status pushes. The
// last-known model + start ISO live on the monitor element so the ticker
// can re-render without re-fetching state.
let _elapsedTicker = null;
function _formatElapsed(ms) {
  if (!ms || ms < 0) return '';
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  if (m < 60) return `${m}m ${rem}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}
function _shortModelName(model) {
  if (!model) return '';
  // Strip provider prefix + GGUF qualifiers so we render "qwen3.6-35b"
  // rather than "ollama/qwen3.6-35b-instruct-Q4_K_M.gguf".
  const tail = model.split(/[\\/]/).pop() || model;
  return tail.replace(/\.(gguf|safetensors|bin)$/i, '')
             .replace(/-instruct-Q\d+(_[A-Z0-9]+)*$/i, '')
             .replace(/-Q\d+(_[A-Z0-9]+)*$/i, '');
}
function _renderSubtitle() {
  if (!_buildMonitor) return;
  const sub = _buildMonitor.querySelector('.build-monitor-subtitle');
  if (!sub) return;
  const model = _buildMonitor._model || '';
  const startIso = _buildMonitor._startedAtIso || '';
  const parts = [];
  if (model) parts.push(_shortModelName(model));
  if (startIso) {
    const elapsedMs = Date.now() - Date.parse(startIso);
    const fmt = _formatElapsed(elapsedMs);
    if (fmt) parts.push(fmt);
  }
  sub.textContent = parts.join(' · ');
}
function _startElapsedTicker() {
  if (_elapsedTicker) return;
  _elapsedTicker = setInterval(() => {
    if (!_buildMonitor || _buildMonitor.classList.contains('done')) {
      _stopElapsedTicker();
      return;
    }
    _renderSubtitle();
  }, 1000);
}
function _stopElapsedTicker() {
  if (_elapsedTicker) { clearInterval(_elapsedTicker); _elapsedTicker = null; }
}

function _qualityStatus(data = {}) {
  const project = data.project || {};
  const value = data.qualityStatus || data.quality_status || project.qualityStatus || project.quality_status || 'clean';
  return value && value !== 'clean' ? value : '';
}

function _qualityWarnings(data = {}) {
  const project = data.project || {};
  const w = data.warnings || data.qualityWarnings || project.warnings || project.qualityWarnings || [];
  return Array.isArray(w) ? w : [];
}
function _blockingErrors(data = {}) {
  const project = data.project || {};
  const e = data.blockingErrors || data.blocking_errors || project.blockingErrors || project.blocking_errors || [];
  return Array.isArray(e) ? e : [];
}

function _needsQualityReview(data = {}) {
  return Boolean(
    _qualityStatus(data)
    || _qualityWarnings(data).length
    || _blockingErrors(data).length,
  );
}

function _qualityLabel(data = {}) {
  return _qualityStatus(data) === 'warning' ? 'Review suggested' : 'Needs review';
}

function _setStatusTone(statusEl, tone = '') {
  if (!statusEl) return;
  statusEl.classList.remove('complete', 'error', 'warning');
  if (tone) statusEl.classList.add(tone);
}

// Pass icons + label rendering. Single source for the three sites
// (_applyBuildStatus / _updateBuildMonitor / _seedMonitorPasses) so the
// "(attempt N/M)" suffix on retry-budgeted passes lands everywhere at once.
const _PASS_ICONS = {
  complete: '<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M3 6l2 2 4-4" stroke="var(--success, #22c55e)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  running:  '<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="4" stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="6 19" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" from="0 6 6" to="360 6 6" dur="0.8s" repeatCount="indefinite"/></circle></svg>',
  error:    '<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M4 4l4 4M8 4l-4 4" stroke="var(--error)" stroke-width="1.5" stroke-linecap="round"/></svg>',
  pending:  '<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="3" stroke="var(--text-muted)" stroke-width="1" opacity="0.4"/></svg>',
};

function _renderPassEntry(p) {
  const icon = _PASS_ICONS[p.status] || _PASS_ICONS.pending;
  const max = Number(p.max_iterations || 0);
  const cur = Number(p.iterations || 0);
  // Show attempt budget only for retry-eligible passes (max > 1). Hides
  // noise on single-iter passes like polish/deliver. Cur=0 means "not
  // started yet" — show the budget so the user knows it CAN retry.
  const attempt = max > 1
    ? `<span class="build-monitor-pass-attempt"> · attempt ${Math.max(1, cur)}/${max}</span>`
    : '';
  const detail = p.detail ? `<span class="build-monitor-pass-detail">${escapeHtml(p.detail)}</span>` : '';
  return `<div class="build-monitor-pass ${p.status}">`
    + `<span class="build-monitor-pass-icon">${icon}</span>`
    + `<span class="build-monitor-pass-name">${escapeHtml(p.name)}</span>`
    + attempt
    + detail
    + `</div>`;
}

// Terminal-error card: pass tombstone + summary + collapsible detail +
// resume button when partial progress is recoverable. Keeps the default
// surface compact (one summary line + the action) and only expands the
// full traceback when the user explicitly clicks "Show details".
function _renderErrorCard(data, failedPass, lastDone) {
  const summary = escapeHtml(data.error || 'Unknown error');
  const detail = data.errorDetail || data.project?.error_detail || '';
  const project = data.project || {};
  const resumable = project.resumable && Array.isArray(project.files) && project.files.length > 0;
  const completed = project.completed_files?.length || 0;
  const total = project.planned_files?.length || 0;

  const tombstoneParts = [];
  if (failedPass) tombstoneParts.push(`Failed at <strong>${escapeHtml(failedPass)}</strong>`);
  if (lastDone) tombstoneParts.push(`last completed <strong>${escapeHtml(lastDone)}</strong>`);
  const tombstone = tombstoneParts.length
    ? `<div class="build-error-tombstone">${tombstoneParts.join(' · ')}</div>`
    : '';

  const summaryRow = `<div class="build-error-summary">${summary}</div>`;
  const detailBlock = detail
    ? `<button class="build-error-toggle" type="button" data-action="toggle-error-detail">Show details</button>`
      + `<pre class="build-error-detail hidden">${escapeHtml(detail)}</pre>`
    : '';
  const resumeBtn = resumable
    ? `<button class="project-action-btn primary" data-action="resume-failed">▶ Resume (${completed}/${total} files done)</button>`
    : '';

  return `<div class="build-error-card">${tombstone}${summaryRow}${detailBlock}${resumeBtn}</div>`;
}

function _wireErrorCardHandlers(actionsEl, data) {
  const toggle = actionsEl.querySelector('[data-action="toggle-error-detail"]');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const pre = actionsEl.querySelector('.build-error-detail');
      if (!pre) return;
      const showing = !pre.classList.toggle('hidden');
      toggle.textContent = showing ? 'Hide details' : 'Show details';
    });
  }
  const resume = actionsEl.querySelector('[data-action="resume-failed"]');
  if (resume) {
    resume.addEventListener('click', async () => {
      const project = data.project || {};
      try {
        const resp = await fetch('/api/artifacts/iterate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            description: project.name || data.name || '',
            scaffold: project.scaffold || 'static',
            model: app.state.currentModel || 'default',
            session_id: app.state.activeSessionId || '',
            files: project.files || [],
            planned_files: project.planned_files || [],
          }),
        });
        if (resp.ok) {
          _buildMonitor.remove();
          _buildMonitor = null;
          _stopBuildStatusFeed();
          _ensureBuildMonitor(project.name || data.name || 'Resuming...');
          showToast('Resuming build from where it failed', 'info');
        }
      } catch (e) {
        showToast('Resume failed: ' + e.message, 'error');
      }
    });
  }
}

function _renderIssueSection(title, items, tone) {
  // tone: 'warning' (amber/review notes) or 'error' (red/unresolved errors).
  // Limits to 5 visible rows + "(N more)" rollup; the user opens the
  // library to see the full set rather than the monitor scrolling forever.
  if (!items || !items.length) return '';
  const shown = items.slice(0, 5);
  const more = items.length > shown.length
    ? `<div class="build-monitor-quality-more">${items.length - shown.length} more</div>`
    : '';
  const rows = shown.map(msg =>
    `<div class="build-monitor-quality-row">${escapeHtml(msg)}</div>`
  ).join('');
  return `<div class="build-monitor-quality-title quality-${tone}">${escapeHtml(title)}</div>${rows}${more}`;
}

function _syncBuildMonitorQuality(data = {}) {
  if (!_buildMonitor) return;
  const body = _buildMonitor.querySelector('.build-monitor-body');
  if (!body) return;
  let qualityEl = body.querySelector('.build-monitor-quality');
  if (!_needsQualityReview(data)) {
    qualityEl?.remove();
    return;
  }
  if (!qualityEl) {
    qualityEl = document.createElement('div');
    qualityEl.className = 'build-monitor-quality';
    body.insertBefore(qualityEl, body.firstChild);
  }
  // Render warnings (review notes) and blocking errors as distinct
  // sections — they have different severity and the user shouldn't have
  // to guess which entries are "soft FYIs" and which are "this didn't
  // get fixed". When NEITHER list is populated but quality_status flags
  // a review (rare — defensive), fall back to the generic notice.
  const warnings = _qualityWarnings(data);
  const errors = _blockingErrors(data);
  qualityEl.classList.toggle('has-errors', errors.length > 0);
  // Section labels: warnings get the soft "Review notes" / "Needs review"
  // label, errors get the explicit "Unresolved issues". When both are
  // present we show both sections; when neither is populated but
  // quality_status still flags a review (rare — defensive), the generic
  // fallback explains why.
  const warningTitle = errors.length > 0 ? 'Review notes' : _qualityLabel(data);
  let html = '';
  html += _renderIssueSection(warningTitle, warnings, 'warning');
  html += _renderIssueSection('Unresolved issues', errors, 'error');
  if (!html) {
    html = `<div class="build-monitor-quality-title quality-warning">${escapeHtml(_qualityLabel(data))}</div>`
         + `<div class="build-monitor-quality-row">The project completed, but verification recommends review.</div>`;
  }
  qualityEl.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Chat bridge — injected by the host (chat.js or index.js) so this module
// can read/write session state without circular imports.
// ---------------------------------------------------------------------------

let _bridge = {
  getSessions: () => ({}),
  getActiveSessionId: () => null,
  addChildNode: () => null,
  renderMessages: () => {},
  saveSessions: () => {},
};

/**
 * Inject references to chat-level functions.  Call once at startup.
 *
 * @param {Object} bridge
 * @param {() => Object}  bridge.getSessions
 * @param {() => string}  bridge.getActiveSessionId
 * @param {Function}      bridge.addChildNode
 * @param {Function}      bridge.renderMessages
 * @param {Function}      bridge.saveSessions
 */
export function setChatBridge(bridge) {
  _bridge = { ..._bridge, ...bridge };
}

// ---------------------------------------------------------------------------
// Preview Storage Bridge (parent-side)
// Handles localStorage operations from sandboxed preview iframes via postMessage.
// Data is namespaced under 'preview_' to isolate from the main app's localStorage.
// ---------------------------------------------------------------------------

(function _initStorageBridge() {
  const PREFIX = 'preview_';
  window.addEventListener('message', (e) => {
    if (!e.data?.type?.startsWith('storage-')) return;
    // Verify the message comes from a sandboxed iframe (not the main page)
    if (e.source === window) return;
    switch (e.data.type) {
      case 'storage-set':
        try { localStorage.setItem(PREFIX + e.data.key, e.data.value); } catch {}
        break;
      case 'storage-remove':
        try { localStorage.removeItem(PREFIX + e.data.key); } catch {}
        break;
      case 'storage-clear':
        try {
          const keys = [];
          for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            if (k?.startsWith(PREFIX)) keys.push(k);
          }
          keys.forEach(k => localStorage.removeItem(k));
        } catch { /* ignore */ }
        break;
      case 'storage-init':
        // Send all preview_ data to the iframe
        try {
          const data = {};
          for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            if (k?.startsWith(PREFIX)) data[k.slice(PREFIX.length)] = localStorage.getItem(k);
          }
          e.source?.postMessage({ type: 'storage-init-response', data }, '*');
        } catch { /* ignore */ }
        break;
    }
  });
})();

// ---------------------------------------------------------------------------
// Assembly helpers
// ---------------------------------------------------------------------------

/**
 * Assembles project files into a single runnable HTML page.
 * Also builds a source map: array of { file, fileLineStart, assembledLineStart, lineCount }
 * so error line numbers in the assembled output can be traced back to source files.
 */
function _assembleProject(files) {
  const result = _sharedAssemble(files);
  _lastAssemblySourceMap = _getLastSourceMap();
  return result;
}

function _buildPreviewSrcdoc(rawCode) {
  return _sharedBuildPreviewSrcdoc(rawCode);
}

/**
 * Maps an assembled output line number back to a source file + line.
 * Returns { file, line, content } or null if not mappable.
 */
function _mapAssembledLineToSource(project, assembledLine) {
  const sourceMap = project._assemblySourceMap || _lastAssemblySourceMap;
  if (!sourceMap || !sourceMap.length) return null;

  for (const entry of sourceMap) {
    if (assembledLine >= entry.assembledLineStart &&
        assembledLine < entry.assembledLineStart + entry.lineCount) {
      const fileLine = assembledLine - entry.assembledLineStart + 1;
      const file = project.files.find(f => f.path === entry.file);
      const lines = file?.content?.split('\n') || [];
      return {
        file: entry.file,
        line: fileLine,
        content: lines[fileLine - 1] || '',
      };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Tab title notification
// ---------------------------------------------------------------------------

/** Flash tab title when build completes while user is on another tab. */
function _notifyBuildComplete(projectName) {
  if (document.hasFocus()) return; // user is already looking
  const original = document.title;
  _titleFlashInterval = setInterval(() => {
    document.title = document.title === original ? `\u2705 Build Complete \u2014 ${projectName}` : original;
  }, 1500);
  const stop = () => {
    if (_titleFlashInterval) { clearInterval(_titleFlashInterval); _titleFlashInterval = null; }
    document.title = original;
    window.removeEventListener('focus', stop);
  };
  window.addEventListener('focus', stop);
}

// ---------------------------------------------------------------------------
// Persistent Build Monitor
// Lives outside the chat message lifecycle.  Survives stream disconnects
// and page refreshes.
// ---------------------------------------------------------------------------

function _ensureBuildMonitor(projectName, buildId = '') {
  if (_buildMonitor) {
    if (buildId) _buildMonitor._buildId = buildId;
    return _buildMonitor;
  }
  _buildMonitor = document.createElement('div');
  _buildMonitor.id = 'build-monitor';
  _buildMonitor._buildId = buildId || '';
  // SVG progress ring: circumference = 2*PI*8 ≈ 50.27
  const circ = 50.27;
  _buildMonitor.innerHTML = `
    <div class="build-monitor-header">
      <div class="build-monitor-ring">
        <svg viewBox="0 0 22 22">
          <circle class="build-monitor-ring-bg" cx="11" cy="11" r="8"/>
          <circle class="build-monitor-ring-fill" cx="11" cy="11" r="8"
            stroke-dasharray="${circ}" stroke-dashoffset="${circ}"/>
          <path class="build-monitor-ring-check" d="M7 11l3 3 5-6" fill="none"
            stroke="var(--success, #22c55e)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="build-monitor-titleblock">
        <span class="build-monitor-title">${escapeHtml(projectName)}</span>
        <span class="build-monitor-subtitle"></span>
      </div>
      <span class="build-monitor-status">Building\u2026</span>
      <button class="build-monitor-cancel" title="Cancel build">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><rect x="2" y="2" width="8" height="8" rx="1.5"/></svg>
      </button>
      <button class="build-monitor-minimize" title="Minimize">\u2013</button>
      <button class="build-monitor-close" title="Dismiss">&times;</button>
    </div>
    <div class="build-monitor-body">
      <div class="build-monitor-passes"></div>
      <div class="build-monitor-actions hidden"></div>
    </div>
  `;
  _buildMonitor._ringCirc = circ;
  // Close — remove entirely. Fetch the current build id on dismiss so
  // we can flag it as acknowledged; the server will keep returning it
  // from /build-status for 30 min otherwise and every refresh would
  // resurrect the popup.
  _buildMonitor.querySelector('.build-monitor-close').addEventListener('click', async () => {
    try {
      const bid = _buildMonitor?._buildId;
      const r = await fetch('/api/artifacts/build-status' + (bid ? `?build_id=${encodeURIComponent(bid)}` : ''));
      if (r.ok) { const d = await r.json(); if (d?.id) _ackBuild(d.id); }
    } catch { /* best-effort ack — close happens anyway */ }
    _buildMonitor.remove(); _buildMonitor = null;
    _stopBuildStatusFeed();
    _stopElapsedTicker();
  });
  // Minimize — collapse body
  _buildMonitor.querySelector('.build-monitor-minimize').addEventListener('click', (e) => {
    e.stopPropagation();
    _buildMonitor.classList.toggle('minimized');
  });
  // Click header to toggle expand/collapse
  _buildMonitor.querySelector('.build-monitor-header').addEventListener('click', () => {
    _buildMonitor.classList.toggle('minimized');
  });
  // Cancel
  _buildMonitor.querySelector('.build-monitor-cancel').addEventListener('click', async (e) => {
    e.stopPropagation();
    try {
      const bid = _buildMonitor?._buildId;
      // build_id in the JSON body (the handler reads body OR query) — a static
      // path keeps the dead-code scanner from mis-reading the query string as a
      // path segment and flagging a ghost call.
      await fetch('/api/artifacts/build-cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bid ? { build_id: bid } : {}),
      });
      const statusEl = _buildMonitor?.querySelector('.build-monitor-status');
      if (statusEl) { statusEl.textContent = 'Cancelled'; statusEl.classList.add('error'); }
      const cancelBtn = _buildMonitor?.querySelector('.build-monitor-cancel');
      if (cancelBtn) cancelBtn.style.display = 'none';
      _stopBuildStatusFeed();
      showToast('Build cancelled', 'info');
    } catch { /* best-effort — UI already shows cancelled */ }
  });
  // Insert directly above chat-scroll so it's an inline bar. chat-scroll
  // gets reparented into a surface cell by the surface-grid system, so
  // anchor off its current parent rather than .main-area.
  const chatScroll = document.getElementById('chat-scroll');
  const anchorParent = chatScroll?.parentNode || document.querySelector('.main-area');
  if (anchorParent && chatScroll && chatScroll.parentNode === anchorParent) {
    anchorParent.insertBefore(_buildMonitor, chatScroll);
  } else if (anchorParent) {
    anchorParent.appendChild(_buildMonitor);
  } else {
    document.body.appendChild(_buildMonitor);
  }

  // Build-status updates can arrive two ways:
  //   1. SSE — `/build-status/stream` pushes one event per state change.
  //   2. Polling — 2s setInterval, used when SSE fails (legacy proxies,
  //      reverse-proxies that buffer text/event-stream, etc.).
  // Both feed the same _applyBuildStatus(data) handler so the UI logic
  // stays in one place. _buildMonitorPoll holds either the interval id
  // OR the EventSource (we tear down whichever is active).
  let _pollFailCount = 0;

  function _applyBuildStatus(data) {
    if (!_buildMonitor) return;
    if (!data || !data.status) return;
    if (data.id || data.build_id) _buildMonitor._buildId = data.id || data.build_id;
    if (data.model && !_buildMonitor._model) _buildMonitor._model = data.model;
    if (data.startedAtIso && !_buildMonitor._startedAtIso) _buildMonitor._startedAtIso = data.startedAtIso;
    _renderSubtitle();
    if (data.active) _startElapsedTicker(); else _stopElapsedTicker();
    const statusEl = _buildMonitor.querySelector('.build-monitor-status');
      _syncBuildMonitorQuality(data);
      const passesEl = _buildMonitor.querySelector('.build-monitor-passes');
      const actions = _buildMonitor.querySelector('.build-monitor-actions');

      // Always update passes — whether active or complete
      if (data.passes && passesEl) {
        passesEl.innerHTML = data.passes.map(_renderPassEntry).join('');
      }

      // Update progress ring + status text
      if (data.passes && _buildMonitor._ringCirc) {
        const total = data.passes.length || 1;
        const done = data.passes.filter(p => p.status === 'complete').length;
        const pct = done / total;
        const ring = _buildMonitor.querySelector('.build-monitor-ring-fill');
        if (ring) ring.setAttribute('stroke-dashoffset', String(_buildMonitor._ringCirc * (1 - pct)));
      }
      if (data.active && statusEl) {
        const runningPass = data.passes?.find(p => p.status === 'running');
        const tokens = data.totalTokens ? ` \u00B7 ${data.totalTokens.toLocaleString()} tok` : '';
        _setStatusTone(statusEl);
        statusEl.textContent = (runningPass ? runningPass.name + '\u2026' : 'Building\u2026') + tokens;
      }

      // Sync in-chat project card from poll data (keeps both views in sync)
      if (data.active && data.passes) {
        const runningPass = data.passes.find(p => p.status === 'running');
        if (runningPass) {
          handleProjectProgress({
            name: data.name,
            pass: runningPass.name,
            status: runningPass.status,
            detail: runningPass.detail || '',
            iteration: runningPass.iterations || 0,
            filesComplete: data.filesComplete || [],
            filesRemaining: data.filesRemaining || [],
            score: 0,
            build_id: data.id || data.build_id || '',
            qualityStatus: data.qualityStatus || data.quality_status || 'clean',
            warnings: data.warnings || [],
            blockingErrors: data.blockingErrors || data.blocking_errors || [],
          });
        }
      }

      // Hide cancel button when build is no longer running
      if (!data.active) {
        const cancelBtn = _buildMonitor?.querySelector('.build-monitor-cancel');
        if (cancelBtn) cancelBtn.style.display = 'none';
        // Mark this build as seen. The server keeps terminal builds in
        // memory for 30 min, so without this a page refresh would pull
        // the same completion popup back every time.
        _ackBuild(data.id);
      }

      if (data.status === 'cancelled' && statusEl) {
        statusEl.textContent = 'Cancelled';
        _setStatusTone(statusEl, 'error');
        // Show resume button if there are partial files
        if (actions && data.project?.resumable) {
          actions.classList.remove('hidden');
          const completed = data.project.completed_files?.length || 0;
          const total = data.project.planned_files?.length || 0;
          actions.innerHTML = `
            <button class="project-action-btn primary" id="build-resume-btn">\u25B6 Continue (${completed}/${total} done)</button>
          `;
          document.getElementById('build-resume-btn').addEventListener('click', async () => {
            try {
              const resp2 = await fetch('/api/artifacts/iterate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  description: data.project.name || data.name || '',
                  scaffold: data.project.scaffold || 'static',
                  model: app.state.currentModel || 'default',
                  session_id: app.state.activeSessionId || '',
                  files: data.project.files || [],
                  planned_files: data.project.planned_files || [],
                }),
              });
              if (resp2.ok) {
                // Tear down current monitor and create fresh one (reuses existing poll infrastructure)
                _buildMonitor.remove(); _buildMonitor = null;
                _stopBuildStatusFeed();
                _ensureBuildMonitor(data.project.name || data.name || 'Resuming...');
                showToast('Build resuming from where it stopped', 'info');
              }
            } catch(e) { showToast('Resume failed: ' + e.message, 'error'); }
          });
        }
        _stopBuildStatusFeed();
      } else if (data.status === 'complete' && statusEl) {
        const needsReview = _needsQualityReview(data);
        statusEl.textContent = needsReview ? _qualityLabel(data) : 'Complete!';
        _setStatusTone(statusEl, needsReview ? 'warning' : 'complete');
        _buildMonitor.classList.add('done');

        // If this was a modify/iterate, update the existing project card
        const monitorCard = _buildMonitor?._projectCard;
        const monitorNode = _buildMonitor?._projectNode;
        if (monitorCard && monitorNode && data.project?.files?.length > 0) {
          const proj = monitorNode.projectArtifact;
          if (proj) {
            proj.files = data.project.files;
            proj.score = data.project.score || proj.score;
            proj.artifactId = data.project.artifactId || data.artifact_id || proj.artifactId || '';
            proj.buildId = data.id || data.build_id || proj.buildId || '';
            proj.qualityStatus = data.qualityStatus || data.quality_status || data.project.qualityStatus || data.project.quality_status || proj.qualityStatus || 'clean';
            proj.warnings = data.warnings || data.project.warnings || proj.warnings || [];
            proj.blockingErrors = data.blockingErrors || data.blocking_errors || data.project.blockingErrors || data.project.blocking_errors || proj.blockingErrors || [];
            proj.previewHtml = _assembleProject(proj.files);

            // Auto-generate version changelog
            if (!proj.versions) proj.versions = [];
            const iterateDesc = _buildMonitor?._iterateDescription || 'Modified project';
            proj.versions.push({
              timestamp: Date.now(),
              label: iterateDesc.length > 80 ? iterateDesc.slice(0, 77) + '...' : iterateDesc,
              files: proj.files.map(f => ({ ...f, content: f.content })),
            });

            const previewEl = monitorCard.querySelector('.project-preview iframe');
            if (previewEl && proj.previewHtml) {
              const nf = document.createElement('iframe');
              nf.setAttribute('sandbox', 'allow-scripts allow-forms allow-modals allow-popups');
              nf.setAttribute('srcdoc', _buildPreviewSrcdoc(proj.previewHtml));
              previewEl.replaceWith(nf);
            }
            const badge = monitorCard.querySelector('.project-card-badge');
            if (badge) { badge.className = `project-card-badge ${needsReview ? 'warning' : 'complete'}`; badge.textContent = needsReview ? 'Review' : 'Ready'; }
            showToast(needsReview ? 'Project modified - review recommended' : `Project modified - ${proj.files.length} files`, needsReview ? 'warning' : 'success');
            _bridge.saveSessions();
          }
        } else if (data.project?.files?.length > 0 && !monitorNode) {
          // Background build completed — deliver project card into chat
          _deliverBuildToChat(data);
        }

        if (actions && data.project) {
          actions.classList.remove('hidden');
          const artifactId = data.project.artifactId || '';
          actions.innerHTML = artifactId
            ? `<a href="/api/artifacts/${escapeHtml(artifactId)}/download" class="project-action-btn primary" target="_blank">\uD83D\uDCE6 Download</a>
               <button class="project-action-btn" onclick="window.open('/api/artifacts/${escapeHtml(artifactId)}/preview','_blank')">\u2197 Preview</button>
               <button class="project-action-btn" onclick="document.querySelector('[data-tab=\\'library\\']')?.click()">Library</button>`
            : `<button class="project-action-btn" onclick="document.querySelector('[data-tab=\\'library\\']')?.click()">View in Library</button>`;
        }
        _notifyBuildComplete(data.name || 'Web App');
        _stopBuildStatusFeed();
      } else if (data.status === 'error' && statusEl) {
        const failedPass = data.failedPass || data.project?.failed_pass || '';
        const lastDone = data.lastCompletedPass || data.project?.last_completed_pass || '';
        statusEl.textContent = failedPass ? `Failed at ${failedPass}` : 'Failed';
        _setStatusTone(statusEl, 'error');
        if (actions) {
          actions.classList.remove('hidden');
          actions.innerHTML = _renderErrorCard(data, failedPass, lastDone);
          _wireErrorCardHandlers(actions, data);
        }
        _stopBuildStatusFeed();
        _stopElapsedTicker();
      }
  }

  function _startPollingFallback() {
    _buildMonitorPoll = setInterval(async () => {
      if (!_buildMonitor) { _stopBuildStatusFeed(); return; }
      try {
        const bid = _buildMonitor?._buildId;
        const resp = await fetch('/api/artifacts/build-status' + (bid ? `?build_id=${encodeURIComponent(bid)}` : ''));
        if (!resp.ok) {
          _pollFailCount++;
          if (_pollFailCount >= 5) {
            const statusEl = _buildMonitor.querySelector('.build-monitor-status');
            if (statusEl) statusEl.textContent = 'Connection lost';
          }
          return;
        }
        _pollFailCount = 0;
        const data = await resp.json();
        _applyBuildStatus(data);
      } catch { /* ignore */ }
    }, 2000);
  }

  // Prefer SSE so the monitor reflects state changes within tens of
  // milliseconds instead of up to 2 seconds. EventSource sends cookies
  // by default, so the existing auth middleware works without extra
  // wiring. Fall back to polling on connection error or in browsers
  // without EventSource (older Edge/IE).
  if (typeof EventSource === 'function') {
    try {
      const bid = _buildMonitor?._buildId;
      const es = new EventSource('/api/artifacts/build-status/stream' + (bid ? `?build_id=${encodeURIComponent(bid)}` : ''));
      _buildMonitorPoll = es;
      es.onmessage = (ev) => {
        try { _applyBuildStatus(JSON.parse(ev.data)); } catch { /* ignore parse errors */ }
      };
      es.addEventListener('end', () => _stopBuildStatusFeed());
      es.onerror = () => {
        // Either the server closed the stream (terminal status) or a
        // proxy is buffering — either way, switch to polling so we
        // don't sit silent.
        if (_buildMonitorPoll === es) {
          es.close();
          _buildMonitorPoll = null;
          _startPollingFallback();
        }
      };
    } catch {
      _startPollingFallback();
    }
  } else {
    _startPollingFallback();
  }

  return _buildMonitor;
}

// ---------------------------------------------------------------------------
// Build monitor update (from streaming progress)
// ---------------------------------------------------------------------------

function _updateBuildMonitor(progress) {
  const monitor = _buildMonitor;
  if (!monitor) return;
  _syncBuildMonitorQuality(progress);

  const passesEl = monitor.querySelector('.build-monitor-passes');
  if (!passesEl) return;

  // Rebuild pass list with SVG icons
  if (_pendingProject?.passes) {
    passesEl.innerHTML = _pendingProject.passes.map(_renderPassEntry).join('');

    // Update progress ring
    if (monitor._ringCirc) {
      const total = _pendingProject.passes.length || 1;
      const done = _pendingProject.passes.filter(p => p.status === 'complete').length;
      const ring = monitor.querySelector('.build-monitor-ring-fill');
      if (ring) ring.setAttribute('stroke-dashoffset', String(monitor._ringCirc * (1 - done / total)));
    }
  }

  // Update file progress list
  const done = progress.filesComplete || [];
  const remaining = progress.filesRemaining || [];
  if (done.length > 0 || remaining.length > 0) {
    let filesEl = monitor.querySelector('.build-monitor-files');
    if (!filesEl) {
      filesEl = document.createElement('div');
      filesEl.className = 'build-monitor-files';
      const body = monitor.querySelector('.build-monitor-body');
      if (body) body.appendChild(filesEl);
    }
    const checkSvg = '<svg width="10" height="10" viewBox="0 0 12 12" fill="none"><path d="M3 6l2 2 4-4" stroke="var(--success, #22c55e)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    const dotSvg = '<svg width="10" height="10" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="3" stroke="var(--text-muted)" stroke-width="1" opacity="0.4"/></svg>';
    const spinSvg = '<svg width="10" height="10" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="4" stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="6 19" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" from="0 6 6" to="360 6 6" dur="0.8s" repeatCount="indefinite"/></circle></svg>';
    const lines = [];
    for (const f of done) {
      lines.push(`<div class="build-file-entry done">${checkSvg}<span>${escapeHtml(f)}</span></div>`);
    }
    // Show only the next file as "in progress", rest as pending
    for (let i = 0; i < remaining.length; i++) {
      const fileIcon = i === 0 ? spinSvg : dotSvg;
      const cls = i === 0 ? 'active' : 'pending';
      lines.push(`<div class="build-file-entry ${cls}">${fileIcon}<span>${escapeHtml(remaining[i])}</span></div>`);
    }
    filesEl.innerHTML = `<div class="build-files-label">Files (${done.length}/${done.length + remaining.length})</div>` + lines.join('');
  }

  // Update status text with token count
  const statusEl = monitor.querySelector('.build-monitor-status');
  if (statusEl && progress.status !== 'complete') {
    const tokens = progress.totalTokens ? ` \u00B7 ${progress.totalTokens.toLocaleString()} tok` : '';
    _setStatusTone(statusEl, _needsQualityReview(progress) ? 'warning' : '');
    statusEl.textContent = progress.pass + '\u2026' + tokens;
  }
  if (statusEl && progress.status === 'complete' && progress.pass === 'deliver') {
    const needsReview = _needsQualityReview(progress);
    const finalTokens = progress.totalTokens ? ` \u00B7 ${progress.totalTokens.toLocaleString()} tok` : '';
    statusEl.textContent = (needsReview ? _qualityLabel(progress) : 'Complete!') + finalTokens;
    _setStatusTone(statusEl, needsReview ? 'warning' : 'complete');
    monitor.classList.add('done');
    _stopBuildStatusFeed();
  }
}

// ---------------------------------------------------------------------------
// Seed monitor passes from build-status data
// ---------------------------------------------------------------------------

function _seedMonitorPasses(data) {
  if (!data.passes || !_buildMonitor) return;
  const passesEl = _buildMonitor.querySelector('.build-monitor-passes');
  if (passesEl) {
    passesEl.innerHTML = data.passes.map(_renderPassEntry).join('');
  }
  if (_buildMonitor._ringCirc) {
    const total = data.passes.length || 1;
    const done = data.passes.filter(p => p.status === 'complete').length;
    const ring = _buildMonitor.querySelector('.build-monitor-ring-fill');
    if (ring) ring.setAttribute('stroke-dashoffset', String(_buildMonitor._ringCirc * (1 - done / total)));
  }
  const statusEl = _buildMonitor.querySelector('.build-monitor-status');
  const runningPass = data.passes.find(p => p.status === 'running');
  if (statusEl && runningPass) {
    const tokens = data.totalTokens ? ` \u00B7 ${data.totalTokens.toLocaleString()} tok` : '';
    _setStatusTone(statusEl);
    statusEl.textContent = runningPass.name + '\u2026' + tokens;
  }
}

// ---------------------------------------------------------------------------
// Deliver completed background build into chat
// ---------------------------------------------------------------------------

/**
 * Deliver a completed background build into the chat as a new assistant message
 * with a full project card. Called when the build monitor detects completion
 * for a build that has no existing in-chat project card.
 */
function _deliverBuildToChat(data) {
  const sessions = _bridge.getSessions();
  const activeSessionId = _bridge.getActiveSessionId();
  const session = sessions[activeSessionId];
  if (!session) return;

  // Build the project artifact object
  const project = data.project;
  const artifact = {
    id: 'proj_' + Date.now().toString(36),
    name: project.name || data.name || 'Web App',
    status: 'complete',
    files: project.files || [],
    passes: [],
    score: project.score || 0,
    artifactId: project.artifactId || '',
    buildId: data.id || data.build_id || project.buildId || '',
    scaffold: project.scaffold || 'static',
    qualityStatus: data.qualityStatus || data.quality_status || project.qualityStatus || project.quality_status || 'clean',
    warnings: data.warnings || project.warnings || [],
    blockingErrors: data.blockingErrors || data.blocking_errors || project.blockingErrors || project.blocking_errors || [],
    previewHtml: '',
  };

  // Assemble preview
  if (artifact.files.length > 0) {
    artifact.previewHtml = _assembleProject(artifact.files);
    artifact._assemblySourceMap = _lastAssemblySourceMap;
  }

  // Initialize version history
  if (artifact.files.length > 0) {
    artifact.versions = [{
      timestamp: Date.now(),
      label: 'Initial build',
      files: artifact.files.map(f => ({ ...f })),
    }];
  }

  // Build completion summary text
  const fileCount = artifact.files.length;
  const scoreText = artifact.score ? ` \u00B7 Score ${artifact.score}/10` : '';
  const content = `\u2705 **${escapeHtml(artifact.name)}** — ${fileCount} files${scoreText}`;

  // Add as a new assistant message in the chat tree
  const node = _bridge.addChildNode(session, session.activeLeafId, 'assistant', content);
  node.projectArtifact = artifact;
  session.activeLeafId = node.id;

  // Render the message + project card
  _bridge.renderMessages();
  _bridge.saveSessions();

  // Store references on monitor so modify/iterate can find the card
  if (_buildMonitor) {
    const msgEl = app.dom.chatMessages.querySelector(`[data-node-id="${node.id}"]`);
    if (msgEl) {
      _buildMonitor._projectCard = msgEl.querySelector('.project-card');
      _buildMonitor._projectNode = node;
    }
  }

  showToast(`${artifact.name} — build complete!`, 'success');
}

// ---------------------------------------------------------------------------
// Project progress (streaming)
// ---------------------------------------------------------------------------

function handleProjectProgress(progress) {
  // Always update the persistent build monitor (survives stream disconnects)
  _ensureBuildMonitor(progress.name || 'Web App', progress.build_id || progress.buildId || '');
  _updateBuildMonitor(progress);

  // Find the project card — in streaming message OR in the last finalized message
  const streamMsg = document.getElementById('streaming-message');
  let contentEl = streamMsg?.querySelector('.message-content');

  // If streaming message is gone, find the card in the most recent assistant message
  if (!contentEl) {
    const allCards = app.dom.chatMessages.querySelectorAll('.project-card');
    if (allCards.length > 0) {
      contentEl = allCards[allCards.length - 1].closest('.message-content');
    }
  }
  if (!contentEl) return;

  // Build the pending project artifact
  if (!_pendingProject) {
    _pendingProject = {
      id: 'proj_' + Date.now().toString(36),
      name: progress.name || 'Web App',
      status: 'building',
      files: [],
      passes: [],
      previewHtml: '',
      score: 0,
      buildId: progress.build_id || progress.buildId || '',
    };
  }

  const pa = _pendingProject;
  pa.name = progress.name || pa.name;
  pa.score = progress.score || pa.score;
  pa.buildId = progress.build_id || progress.buildId || pa.buildId || '';
  pa.qualityStatus = progress.qualityStatus || progress.quality_status || pa.qualityStatus || 'clean';
  pa.warnings = progress.warnings || progress.qualityWarnings || pa.warnings || [];
  pa.blockingErrors = progress.blockingErrors || progress.blocking_errors || pa.blockingErrors || [];

  // Update or add the pass
  const existing = pa.passes.find(p => p.name === progress.pass);
  if (existing) {
    existing.status = progress.status;
    existing.detail = progress.detail || existing.detail;
    existing.iterations = progress.iteration;
  } else {
    pa.passes.push({
      name: progress.pass,
      status: progress.status,
      detail: progress.detail || '',
      iterations: progress.iteration || 0,
    });
  }

  // Render or update the card in a SEPARATE container (not inside .response-body,
  // which gets replaced on every streaming text update)
  let card = contentEl.querySelector('.project-card');
  if (!card) {
    // Remove streaming dots if present
    const dots = contentEl.querySelector('.streaming-dots');
    if (dots) dots.remove();

    card = document.createElement('div');
    card.className = 'project-card';
    card.innerHTML = `<div class="project-card-header">
      <span class="project-card-title">${escapeHtml(pa.name)}</span>
      <span class="project-card-badge building">Building</span>
    </div>
    <div class="project-pipeline"></div>
    <div class="project-progress">
      <div class="project-progress-bar"><div class="project-progress-fill" style="width:0%"></div></div>
      <div class="project-progress-text">Starting...</div>
    </div>`;
    // Insert at the TOP of message-content, before any .response-body
    contentEl.insertBefore(card, contentEl.firstChild);
  }

  // Update pipeline rows
  const pipeline = card.querySelector('.project-pipeline');
  if (pipeline) {
    let pipelineHtml = '';
    for (const pass of pa.passes) {
      const ps = pass.status || 'pending';
      const passIcon = ps === 'complete' ? icons.checkSmall
        : ps === 'running' ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="14" height="14"><circle cx="12" cy="12" r="10" stroke-dasharray="31.4 31.4" stroke-dashoffset="10"/></svg>`
        : ps === 'failed' ? (icons.xSmall || '\u274C')
        : (icons.dotEmpty || '\u25CB');
      pipelineHtml += `<div class="project-pass ${ps}">
        <span class="project-pass-icon">${passIcon}</span>
        <span class="project-pass-name">${escapeHtml(pass.name)}</span>
        <span class="project-pass-detail">${escapeHtml(pass.detail || '')}</span>
      </div>`;
    }
    pipeline.innerHTML = pipelineHtml;
  }

  // Update progress bar
  const fill = card.querySelector('.project-progress-fill');
  const text = card.querySelector('.project-progress-text');
  if (fill && text) {
    const total = Math.max(pa.passes.length, 5);
    const complete = pa.passes.filter(p => p.status === 'complete').length;
    const pct = Math.round((complete / total) * 100);
    fill.style.width = `${pct}%`;
    text.textContent = `${progress.pass}: ${progress.detail || progress.status}`;
  }
  _updateBuildMonitor(progress);
}

// ---------------------------------------------------------------------------
// Project result (final from tool_call metadata)
// ---------------------------------------------------------------------------

/**
 * Handle the final project result from the tool_call metadata.
 * Merges with _pendingProject and stores for node attachment at finalization.
 */
function handleProjectResult(projectData) {
  _pendingProject = {
    ...(_pendingProject || {}),
    ...projectData,
    status: projectData.status || 'complete',
    buildId: projectData.buildId || projectData.build_id || _pendingProject?.buildId || '',
    qualityStatus: projectData.qualityStatus || projectData.quality_status || _pendingProject?.qualityStatus || 'clean',
    warnings: projectData.warnings || projectData.qualityWarnings || _pendingProject?.warnings || [],
    blockingErrors: projectData.blockingErrors || projectData.blocking_errors || _pendingProject?.blockingErrors || [],
  };

  // Initialize version history with the initial build
  if (_pendingProject.files?.length > 0 && !_pendingProject.versions) {
    _pendingProject.versions = [{
      timestamp: Date.now(),
      label: 'Initial build',
      files: _pendingProject.files.map(f => ({ ...f, content: f.content })),
    }];
  }

  // Assemble preview if not present
  if (!_pendingProject.previewHtml && _pendingProject.files?.length > 0) {
    _pendingProject.previewHtml = _assembleProject(_pendingProject.files);
    _pendingProject._assemblySourceMap = _lastAssemblySourceMap;
  }
}

// ---------------------------------------------------------------------------
// Build monitor recovery (page refresh / tab switch)
// Recovers ALL build states — running, complete, error, cancelled.
// ---------------------------------------------------------------------------

// localStorage key for the last build id the user has already seen in a
// terminal state (complete / error / cancelled). The server keeps that
// build around in ACTIVE_BUILDS for 30 min after completion, so without
// this every page refresh during that window re-surfaces the "Build
// finished" popup. Running builds always recover — they're load-bearing
// for in-flight work.
const ACKED_BUILD_KEY = 'augmentum-acked-build-id';

function _ackBuild(id) {
  if (!id) return;
  try { localStorage.setItem(ACKED_BUILD_KEY, String(id)); } catch {}
  // Cross-device dismissal: the server falls through to latest_for_session()
  // when ACTIVE_BUILDS is cold (every restart), so without a server-side
  // flag the same terminal build resurfaces on every other device forever.
  // Fire-and-forget; same-tab UX never blocks on this.
  try {
    fetch(`/api/builds/${encodeURIComponent(id)}/ack`, { method: 'POST' }).catch(() => {});
  } catch { /* best-effort */ }
}

function _isBuildAcked(id) {
  if (!id) return false;
  try { return localStorage.getItem(ACKED_BUILD_KEY) === String(id); } catch { return false; }
}

async function recoverBuildMonitor() {
  if (_buildMonitor) return; // already active
  try {
    const resp = await fetch('/api/artifacts/build-status');
    if (!resp.ok) return;
    const data = await resp.json();
    if (!data.status) return;

    // Terminal builds stay in server memory for 30 min so /build-status
    // keeps returning them. Suppress the popup once the user has seen
    // this particular build id — otherwise every refresh re-surfaces
    // the same "Build finished" / "Resume available" card forever.
    const terminal = data.status === 'complete' || data.status === 'error' || data.status === 'cancelled';
    if (terminal && _isBuildAcked(data.id)) return;

    const name = data.name || 'Web App';

    if (data.status === 'running' || data.active) {
      // Build still in progress — re-create the monitor with polling
      _ensureBuildMonitor(name, data.id || data.build_id || '');
      _seedMonitorPasses(data);
      showToast('Build in progress \u2014 monitor recovered', 'info');

    } else if (data.status === 'complete' && data.project) {
      _ackBuild(data.id);
      // Build finished while we were away — show result notification
      _ensureBuildMonitor(name, data.id || data.build_id || '');
      _seedMonitorPasses(data);
      _syncBuildMonitorQuality(data);
      const statusEl = _buildMonitor?.querySelector('.build-monitor-status');
      if (statusEl) {
        const needsReview = _needsQualityReview(data);
        statusEl.textContent = needsReview ? _qualityLabel(data) : 'Complete!';
        _setStatusTone(statusEl, needsReview ? 'warning' : 'complete');
      }
      _buildMonitor?.classList.add('done');
      // Fill ring to 100%
      const ring = _buildMonitor?.querySelector('.build-monitor-ring-fill');
      if (ring && _buildMonitor._ringCirc) ring.setAttribute('stroke-dashoffset', '0');
      // Show actions
      const actions = _buildMonitor?.querySelector('.build-monitor-actions');
      if (actions && data.project) {
        actions.classList.remove('hidden');
        const aid = data.project.artifactId || '';
        actions.innerHTML = aid
          ? `<a href="/api/artifacts/${escapeHtml(aid)}/download" class="project-action-btn primary" target="_blank">\uD83D\uDCE6 Download</a>
             <button class="project-action-btn" onclick="window.open('/api/artifacts/${escapeHtml(aid)}/preview','_blank')">\u2197 Preview</button>
             <button class="project-action-btn" onclick="document.querySelector('[data-tab=\\'library\\']')?.click()">Library</button>`
          : `<button class="project-action-btn" onclick="document.querySelector('[data-tab=\\'library\\']')?.click()">View in Library</button>`;
      }
      // Stop polling — build is done
      _stopBuildStatusFeed();
      // Tab title notification
      _notifyBuildComplete(name);
      showToast(_needsQualityReview(data) ? `Build finished: ${name} - review recommended` : `Build finished: ${name}`, _needsQualityReview(data) ? 'warning' : 'success');

    } else if ((data.status === 'error' || data.status === 'cancelled') && data.project?.resumable) {
      // Build failed/cancelled with partial progress — show resume option
      _ackBuild(data.id);
      _ensureBuildMonitor(name, data.id || data.build_id || '');
      _seedMonitorPasses(data);
      const statusEl = _buildMonitor?.querySelector('.build-monitor-status');
      if (statusEl) {
        statusEl.textContent = data.status === 'error' ? 'Failed' : 'Cancelled';
        _setStatusTone(statusEl, 'error');
      }
      const actions = _buildMonitor?.querySelector('.build-monitor-actions');
      if (actions) {
        actions.classList.remove('hidden');
        const completed = data.project.completed_files?.length || 0;
        const total = data.project.planned_files?.length || 0;
        const errorMsg = data.error ? `<div style="color:var(--error);font-size:var(--text-xs);margin-bottom:6px">${escapeHtml(data.error)}</div>` : '';
        actions.innerHTML = errorMsg + `<button class="project-action-btn primary" id="build-resume-btn">\u25B6 Resume (${Number(completed)}/${Number(total)} files)</button>`;
        document.getElementById('build-resume-btn')?.addEventListener('click', async () => {
          try {
            const r = await fetch('/api/artifacts/iterate', {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                description: data.project.name || name, scaffold: data.project.scaffold || 'static',
                model: app.state.currentModel || 'default', session_id: app.state.activeSessionId || '',
                files: data.project.files || [], planned_files: data.project.planned_files || [],
              }),
            });
            if (r.ok) {
              const resumed = await r.json().catch(() => ({}));
              _buildMonitor.remove(); _buildMonitor = null;
              _stopBuildStatusFeed();
              _ensureBuildMonitor(name, resumed.id || resumed.build_id || '');
              showToast('Resuming build', 'info');
            }
          } catch(e) { showToast('Resume failed: ' + e.message, 'error'); }
        });
      }
      _stopBuildStatusFeed();
      showToast(`Build ${data.status} \u2014 resume available`, 'warning');
    }
  } catch { /* server not reachable, ignore */ }
}

// ---------------------------------------------------------------------------
// Version history / revert
// ---------------------------------------------------------------------------

/**
 * Restore the most recent in-memory version, or fetch the server's
 * version list and show a small picker if no in-memory state exists
 * (e.g. the user opened the project from the library after a refresh).
 *
 * Server reverts go through POST /api/artifacts/{id}/revert/{version}
 * which auto-snapshots the current state — so the revert is itself
 * undoable.
 */
async function _handleProjectRevert(project, card) {
  // Fast path: in-memory revert (works during the active session).
  if (project.versions && project.versions.length > 1) {
    const prev = project.versions[project.versions.length - 2];
    project.files = prev.files.map(f => ({ ...f, content: f.content }));
    project.versions.pop();
    _refreshProjectPreviewFromFiles(project, card);
    showToast(`Reverted to: ${prev.label}`, 'info');
    if (card._resetErrorCapture) card._resetErrorCapture();
    if (project.versions.length <= 1) {
      card.querySelector('[data-action="project-revert"]')?.remove();
    }
    _bridge.saveSessions();
    return;
  }

  // Server path: project loaded from library, no in-memory history.
  const artifactId = project.artifactId;
  if (!artifactId) {
    showToast('No previous version available', 'info');
    return;
  }
  let versions = [];
  try {
    const resp = await fetch(`/api/artifacts/${encodeURIComponent(artifactId)}/versions`);
    if (resp.ok) {
      const data = await resp.json();
      versions = data.versions || [];
    }
  } catch { /* ignore — handled below */ }
  if (!versions.length) {
    showToast('No saved versions found', 'info');
    return;
  }
  // Skip the most recent version — that's the current state.
  const restorable = versions.slice(1);
  if (!restorable.length) {
    showToast('No earlier versions to revert to', 'info');
    return;
  }
  const target = await _pickVersion(restorable);
  if (!target) return;
  try {
    const resp = await fetch(
      `/api/artifacts/${encodeURIComponent(artifactId)}/revert/${encodeURIComponent(target.id)}`,
      { method: 'POST' },
    );
    if (!resp.ok) {
      showToast('Revert failed', 'error');
      return;
    }
    const result = await resp.json();
    project.files = result.files;
    _refreshProjectPreviewFromFiles(project, card);
    if (card._resetErrorCapture) card._resetErrorCapture();
    showToast(`Reverted to v${result.version_index}`, 'success');
    _bridge.saveSessions();
  } catch (e) {
    showToast(`Revert failed: ${e.message}`, 'error');
  }
}

function _refreshProjectPreviewFromFiles(project, card) {
  project.previewHtml = _assembleProject(project.files);
  project._assemblySourceMap = _lastAssemblySourceMap;
  const previewEl = card.querySelector('.project-preview');
  if (!previewEl) return;
  const oldIframe = previewEl.querySelector('iframe');
  if (!oldIframe) return;
  const newIframe = document.createElement('iframe');
  newIframe.setAttribute('sandbox', 'allow-scripts allow-forms allow-modals allow-popups');
  newIframe.setAttribute('srcdoc', _buildPreviewSrcdoc(project.previewHtml));
  oldIframe.replaceWith(newIframe);
}

/**
 * Show a one-shot picker for the given restorable versions and
 * resolve to the user's choice (or null on cancel). Inline overlay
 * rather than a modal — keeps the workspace surface visible.
 */
function _pickVersion(versions) {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'version-picker-overlay';
    overlay.style.cssText = (
      'position:fixed;inset:0;background:rgba(0,0,0,0.4);z-index:1000;'
      + 'display:flex;align-items:center;justify-content:center;'
    );
    const panel = document.createElement('div');
    panel.style.cssText = (
      'background:var(--bg-elevated, #1f2937);color:var(--text-primary, #fff);'
      + 'border:1px solid var(--border, #374151);border-radius:var(--radius-md, 8px);'
      + 'padding:var(--space-md, 16px);max-width:90vw;max-height:80vh;overflow:auto;'
      + 'min-width:280px;'
    );
    panel.innerHTML = `<div style="font-weight:600;margin-bottom:var(--space-sm, 8px)">Pick a version to restore</div>`;
    for (const v of versions) {
      const row = document.createElement('button');
      row.className = 'project-action-btn';
      row.style.cssText = 'display:block;width:100%;text-align:left;margin-bottom:6px';
      const label = v.label || `v${v.version_index}`;
      const date = v.created_at ? ` · ${v.created_at.split('T')[0]}` : '';
      row.innerHTML = `<strong>v${escapeHtml(String(v.version_index))}</strong> ${escapeHtml(label)}${escapeHtml(date)}`;
      row.addEventListener('click', () => { document.body.removeChild(overlay); resolve(v); });
      panel.appendChild(row);
    }
    const cancel = document.createElement('button');
    cancel.className = 'project-action-btn';
    cancel.textContent = 'Cancel';
    cancel.style.cssText = 'margin-top:var(--space-sm, 8px)';
    cancel.addEventListener('click', () => { document.body.removeChild(overlay); resolve(null); });
    panel.appendChild(cancel);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) { document.body.removeChild(overlay); resolve(null); }
    });
    overlay.appendChild(panel);
    document.body.appendChild(overlay);
  });
}

// ---------------------------------------------------------------------------
// Download ZIP
// ---------------------------------------------------------------------------

async function _downloadProjectZip(project) {
  let JSZipLib = window.JSZip;
  if (!JSZipLib) {
    try {
      const mod = await import('https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm');
      JSZipLib = mod.default || mod;
    } catch (err) {
      showToast('Failed to load zip library', 'error');
      return;
    }
  }

  const zip = new JSZipLib();
  for (const file of project.files) {
    zip.file(file.path, file.content);
  }
  if (!project.files.some(f => f.path.toLowerCase() === 'readme.md')) {
    zip.file('README.md', `# ${project.name || 'Web App'}\n\nGenerated by Augmentum.\n\nOpen index.html in your browser to run.\n`);
  }

  const blob = await zip.generateAsync({ type: 'blob' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${(project.name || 'project').toLowerCase().replace(/\s+/g, '-')}.zip`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Project card rendering
// ---------------------------------------------------------------------------

function renderProjectCard(node, container) {
  const project = node.projectArtifact;
  if (!project) return;

  const card = document.createElement('div');
  card.className = 'project-card';
  card.dataset.projectId = project.id || '';

  const status = project.status || 'complete';
  const needsReview = _needsQualityReview(project);
  const badgeClass = status === 'building' ? 'building' : status === 'error' ? 'error' : needsReview ? 'warning' : 'complete';
  const badgeText = status === 'building' ? 'Building' : status === 'error' ? 'Failed' : needsReview ? 'Review' : 'Ready';

  let html = `<div class="project-card-header">
    <span class="project-card-title">${escapeHtml(project.name || 'Web App')}</span>
    <span class="project-card-badge ${badgeClass}">${badgeText}</span>
  </div>`;

  if (needsReview) {
    const messages = _qualityWarnings(project).slice(0, 3);
    const rows = messages.length
      ? messages.map(msg => `<div class="project-warning-row">${escapeHtml(msg)}</div>`).join('')
      : '<div class="project-warning-row">Verification recommends review before sharing this app.</div>';
    html += `<div class="project-warning">
      <div class="project-warning-title">${escapeHtml(_qualityLabel(project))}</div>
      ${rows}
    </div>`;
  }

  // Pipeline progress
  if (project.passes && project.passes.length > 0) {
    const collapsed = status === 'complete' ? ' collapsed' : '';
    html += `<div class="project-pipeline${collapsed}">`;
    for (const pass of project.passes) {
      const ps = pass.status || 'pending';
      const passIcon = ps === 'complete' ? icons.checkSmall
        : ps === 'running' ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="14" height="14"><circle cx="12" cy="12" r="10" stroke-dasharray="31.4 31.4" stroke-dashoffset="10"/></svg>`
        : icons.dotEmpty;
      html += `<div class="project-pass ${ps}">
        <span class="project-pass-icon">${passIcon}</span>
        <span class="project-pass-name">${escapeHtml(pass.name)}</span>
        <span class="project-pass-detail">${escapeHtml(pass.detail || '')}</span>
      </div>`;
    }
    html += '</div>';
  }

  // Progress bar during building
  if (status === 'building') {
    html += `<div class="project-progress">
      <div class="project-progress-bar"><div class="project-progress-fill" style="width:0%"></div></div>
      <div class="project-progress-text">Starting...</div>
    </div>`;
  }

  // Live preview (iframe built programmatically to avoid srcdoc escaping issues)
  const _projectPreviewHtml = (status === 'complete' && (project.previewHtml || project.files?.length > 0))
    ? (project.previewHtml || _assembleProject(project.files))
    : null;
  if (_projectPreviewHtml) {
    html += `<div class="project-preview">
      <div class="project-preview-iframe-slot"></div>
      <div class="project-preview-overlay"></div>
      <div class="project-preview-controls">
        <button class="project-preview-btn" data-action="project-fullscreen" title="Open in new tab">\u2197</button>
        <button class="project-preview-btn" data-action="project-console" title="Toggle console">\u2328</button>
      </div>
    </div>
    <div class="project-console" hidden></div>`;
  }

  // File chips
  if (project.files && project.files.length > 0) {
    html += '<div class="project-files">';
    for (const file of project.files) {
      html += `<button class="project-file-chip" data-path="${escapeHtml(file.path)}">${escapeHtml(file.path)}</button>`;
    }
    html += '</div>';
  }

  // Stats
  if (status === 'complete') {
    const passCount = project.passes ? project.passes.filter(p => p.status === 'complete').length : 0;
    const scoreText = project.score ? ` \u00B7 Score ${project.score}/10` : '';
    html += `<div class="project-stats" data-action="project-toggle-pipeline">
      <span class="project-stats-toggle">\u25B8</span>
      \u2705 Built \u00B7 ${project.files.length} files${scoreText} \u00B7 ${passCount} passes
    </div>`;
  }

  // Error
  if (status === 'error' && project.error) {
    html += `<div class="project-error">
      <div class="project-error-text">${escapeHtml(project.error)}</div>
    </div>`;
  }

  // Actions
  html += '<div class="project-actions">';
  if (status === 'complete') {
    html += `<button class="project-action-btn primary" data-action="project-open-workspace">Open</button>`;
    html += `<button class="project-action-btn" data-action="project-download">\uD83D\uDCE6 ZIP</button>`;
    html += `<button class="project-action-btn" data-action="project-edit">\uD83D\uDCDD Files</button>`;
    html += `<button class="project-action-btn" data-action="project-iterate">\u2728 Modify</button>`;
    if (project.artifactId || project.buildId) {
      html += `<button class="project-action-btn" data-action="project-open-code">Code</button>`;
    }
    html += `<button class="project-action-btn" data-action="project-fullscreen">\u2197 New Tab</button>`;
    // Revert button if either: (a) we have multiple in-memory versions
    // captured this session, or (b) the artifact is persisted server-
    // side and may have stored versions we can fetch on demand.
    if ((project.versions && project.versions.length > 1) || project.artifactId) {
      html += `<button class="project-action-btn" data-action="project-revert">\u21A9 Revert</button>`;
    }
  } else if (status === 'error') {
    html += `<button class="project-action-btn primary" data-action="project-retry">\uD83D\uDD04 Retry</button>`;
    html += `<button class="project-action-btn" data-action="project-edit">\uD83D\uDCDD Edit & Fix</button>`;
  }
  html += '</div>';

  card.innerHTML = html;

  // Inject preview iframe programmatically (avoids srcdoc escaping in template literal)
  if (_projectPreviewHtml) {
    const slot = card.querySelector('.project-preview-iframe-slot');
    if (slot) {
      const iframe = document.createElement('iframe');
      iframe.setAttribute('sandbox', 'allow-scripts allow-forms allow-modals allow-popups');
      const srcdoc = _buildPreviewSrcdoc(_projectPreviewHtml);
      iframe.setAttribute('srcdoc', srcdoc);
      slot.replaceWith(iframe);
    }
  }

  container.appendChild(card);

  // Wire event handlers
  _wireProjectCardActions(card, node);
}

// ---------------------------------------------------------------------------
// Project card action wiring
// ---------------------------------------------------------------------------

function _wireProjectCardActions(card, node) {
  const project = node.projectArtifact;

  card.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const action = btn.dataset.action;

    if (action === 'project-download') {
      _downloadProjectZip(project);
    } else if (action === 'project-open-workspace') {
      _openProjectWorkspace(project, 'play');
    } else if (action === 'project-open-code') {
      _openProjectInCode(project);
    } else if (action === 'project-fullscreen') {
      const assembled = project.previewHtml || _assembleProject(project.files);
      if (assembled) {
        const blob = new Blob([assembled], { type: 'text/html' });
        window.open(URL.createObjectURL(blob), '_blank');
      }
    } else if (action === 'project-console') {
      const consoleEl = card.querySelector('.project-console');
      if (consoleEl) consoleEl.hidden = !consoleEl.hidden;
    } else if (action === 'project-toggle-pipeline') {
      const pipeline = card.querySelector('.project-pipeline');
      const toggle = card.querySelector('.project-stats-toggle');
      if (pipeline) {
        pipeline.classList.toggle('collapsed');
        toggle?.classList.toggle('expanded');
      }
    } else if (action === 'project-edit') {
      _expandProjectFiles(card, node);
    } else if (action === 'project-revert') {
      _handleProjectRevert(project, card);
    } else if (action === 'project-iterate') {
      _showProjectIterateBar(card, node);
    } else if (action === 'project-retry') {
      // Check if we can resume from partial progress (faster than starting over)
      if (project.resumable && project.files?.length > 0 && project.planned_files?.length > 0) {
        showToast(`Resuming build (${project.completed_files?.length || 0} files already done)...`, 'info');
        _iterateProject(card, node, project._originalDescription || project.name, true);
      } else {
        showToast('Retrying build...', 'info');
        const input = document.querySelector('#chat-input, .chat-input, textarea');
        if (input && project.name) {
          input.value = `Rebuild: ${project.name}`;
          input.focus();
        }
      }
    }
  });

  // File chip clicks
  card.querySelectorAll('.project-file-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      _expandProjectFiles(card, node, chip.dataset.path);
    });
  });

  // Preview iframe console/error capture + auto-fix
  // Uses card-level state so it works across iframe replacements (after fixes)
  _initProjectErrorCapture(card, node);
}

async function _openProjectWorkspace(project, mode = 'play') {
  try {
    const artifactId = project?.artifactId || project?.artifact_id || '';
    const { openWorkspace } = await import('../workspace.js');
    if (artifactId) {
      await openWorkspace({ id: artifactId }, mode);
      return;
    }
    await openWorkspace({
      name: project?.name || 'Web App',
      source_json: JSON.stringify({
        type: 'application',
        name: project?.name || 'Web App',
        scaffold: project?.scaffold || 'static',
        files: project?.files || [],
      }),
    }, mode);
  } catch (err) {
    showToast(`Workspace open failed: ${err.message || err}`, 'error');
  }
}

async function _openProjectInCode(project) {
  const buildId = project?.buildId || project?.build_id || '';
  const artifactId = project?.artifactId || project?.artifact_id || '';
  if (!buildId && !artifactId) {
    showToast('This project needs a saved artifact before Code can open it', 'warning');
    return;
  }
  try {
    showToast('Creating Code workspace...', 'info');
    const url = buildId
      ? `/api/builds/${encodeURIComponent(buildId)}/open-in-code`
      : `/api/builds/artifacts/${encodeURIComponent(artifactId)}/open-in-code`;
    const resp = await fetch(url, {
      method: 'POST',
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(extractErrorMessage(data, 'Code workspace failed'));
    const coder = await import('../coder.js');
    if (typeof coder.openWorkspaceById === 'function') {
      await coder.openWorkspaceById(data.workspace_id);
    }
    document.querySelector('[data-mode="coder"]')?.click();
    showToast('Code workspace ready', 'success');
  } catch (err) {
    showToast(`Code handoff failed: ${err.message || err}`, 'error');
  }
}

// ---------------------------------------------------------------------------
// Error capture from preview iframe
// ---------------------------------------------------------------------------

function _initProjectErrorCapture(card, node) {
  const consoleEl = card.querySelector('.project-console');
  const state = { errors: [], fixBarTimer: null };

  // Single global listener that checks ALL iframes in this card
  const handler = (e) => {
    // Find the current iframe (may have been replaced after a fix)
    const currentIframe = card.querySelector('.project-preview iframe');
    if (!currentIframe || !currentIframe.contentWindow || e.source !== currentIframe.contentWindow) return;

    if (e.data?.type === 'code-console' && consoleEl) {
      const el = document.createElement('div');
      el.className = 'console-' + (e.data.level || 'log');
      el.textContent = (e.data.args || []).join(' ');
      consoleEl.appendChild(el);
    } else if (e.data?.type === 'code-error') {
      const errorText = e.data.detail || 'Unknown error';

      // Extract line number from error text (e.g. "(line 337)" or "line 42")
      const lineMatch = errorText.match(/\(line\s+(\d+)\)|line\s+(\d+)/i);
      const assembledLine = lineMatch ? parseInt(lineMatch[1] || lineMatch[2], 10) : null;

      // Map to source file
      const proj = node.projectArtifact;
      let sourceInfo = null;
      if (assembledLine && proj) {
        sourceInfo = _mapAssembledLineToSource(proj, assembledLine);
      }

      const enrichedError = {
        text: errorText,
        assembledLine,
        sourceFile: sourceInfo?.file || null,
        sourceLine: sourceInfo?.line || null,
        sourceContent: sourceInfo?.content || null,
      };
      state.errors.push(enrichedError);

      if (consoleEl) {
        const el = document.createElement('div');
        el.className = 'console-error';
        el.textContent = sourceInfo
          ? `${sourceInfo.file}:${sourceInfo.line} — ${errorText}`
          : errorText;
        consoleEl.appendChild(el);
        consoleEl.hidden = false;
      }
      // Debounce: wait 2s after last error, then auto-fix or show bar
      if (state.fixBarTimer) clearTimeout(state.fixBarTimer);
      state.fixBarTimer = setTimeout(() => {
        const existingBar = card.querySelector('.project-fix-bar');
        if (existingBar) existingBar.remove();
        const proj2 = node.projectArtifact;
        const autoFix = proj2 && (!proj2._fixAttempts || proj2._fixAttempts === 0);
        if (autoFix) {
          // First round: auto-attempt fix silently (zero-click validation)
          _fixProjectErrors(card, node, [...state.errors]);
        } else {
          // Subsequent rounds: show fix bar for user decision
          _showProjectFixBar(card, node, [...state.errors]);
        }
      }, 2000);
    }
  };
  window.addEventListener('message', handler);

  // Store reset function on the card so _fixProjectErrors can call it after fixing
  card._resetErrorCapture = () => {
    state.errors = [];
    if (state.fixBarTimer) clearTimeout(state.fixBarTimer);
    if (consoleEl) consoleEl.innerHTML = '';
    const existingBar = card.querySelector('.project-fix-bar');
    if (existingBar) existingBar.remove();
  };
}

// ---------------------------------------------------------------------------
// Fix bar (shown after auto-fix limit or for user decision)
// ---------------------------------------------------------------------------

function _showProjectFixBar(card, node, errors) {
  // Don't show if already present
  if (card.querySelector('.project-fix-bar')) return;

  // Deduplicate by error text
  const seen = new Set();
  const uniqueErrors = errors.filter(e => {
    const key = typeof e === 'string' ? e : e.text;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  // Build error summary with source file info
  let errorListHtml = '';
  for (const err of uniqueErrors) {
    const errObj = typeof err === 'string' ? { text: err } : err;
    const loc = errObj.sourceFile
      ? `<span class="project-error-loc">${escapeHtml(errObj.sourceFile)}:${errObj.sourceLine}</span> `
      : '';
    errorListHtml += `<div class="project-error-item">${loc}${escapeHtml(errObj.text)}</div>`;
  }

  const bar = document.createElement('div');
  bar.className = 'project-fix-bar';
  bar.innerHTML = `
    <div class="project-fix-info">
      <span class="project-fix-icon">\u26A0\uFE0F</span>
      <span>${uniqueErrors.length} error${uniqueErrors.length > 1 ? 's' : ''} detected</span>
    </div>
    <div class="project-error-list">${errorListHtml}</div>
    <div class="project-error-explain" style="display:none"></div>
    <div class="project-fix-actions">
      <button class="code-fix-explain-btn" title="Ask AI to explain these errors">Explain</button>
      <button class="code-fix-retry-btn" title="Send errors to AI to fix">Fix Errors</button>
      <button class="code-fix-show-btn" title="Show error in code">Show in Code</button>
      <button class="code-fix-ask-btn" title="Toggle console">Console</button>
    </div>
  `;

  // Explain errors via LLM — shows structured explanation before fixing
  bar.querySelector('.code-fix-explain-btn').addEventListener('click', async () => {
    const explainEl = bar.querySelector('.project-error-explain');
    const explainBtn = bar.querySelector('.code-fix-explain-btn');
    if (explainEl.style.display !== 'none') {
      explainEl.style.display = 'none';
      return;
    }
    explainBtn.disabled = true;
    explainBtn.textContent = 'Thinking\u2026';
    explainEl.style.display = '';
    explainEl.innerHTML = '<div style="color:var(--text-muted);font-style:italic">Analyzing errors\u2026</div>';

    // Build context: error messages + relevant code snippets
    const project = node.projectArtifact;
    let contextLines = uniqueErrors.map((e, i) => {
      const err = typeof e === 'string' ? { text: e } : e;
      let detail = `Error ${i + 1}: ${err.text}`;
      if (err.sourceFile && err.sourceLine) {
        detail = `Error ${i + 1}: ${err.sourceFile}:${err.sourceLine} — ${err.text}`;
        const file = project?.files?.find(f => f.path === err.sourceFile);
        if (file) {
          const lines = file.content.split('\\n');
          const start = Math.max(0, (err.sourceLine || 1) - 4);
          const end = Math.min(lines.length, (err.sourceLine || 1) + 4);
          detail += '\\nCode context:\\n' + lines.slice(start, end).map((l, j) => {
            const ln = start + j + 1;
            return `${ln === err.sourceLine ? '→' : ' '} ${ln}: ${l}`;
          }).join('\\n');
        }
      }
      return detail;
    }).join('\\n\\n');

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: app.state.currentModel || 'default',
          messages: [
            {
              role: 'system',
              content: 'You are a helpful coding tutor. Explain errors clearly and concisely. For each error: (1) What it means in plain English, (2) Why it happened (the root cause in the code), (3) How to fix it (specific steps). Use markdown formatting. Keep each explanation to 2-3 sentences per section. Be encouraging — errors are normal and learning opportunities.'
            },
            {
              role: 'user',
              content: `Explain these errors from my web app:\\n\\n${contextLines}`
            }
          ],
          stream: false,
          think: false,
        }),
      });
      if (resp.ok) {
        const data = await resp.json();
        const explanation = data.choices?.[0]?.message?.content || data.content || 'Could not generate explanation.';
        // Render markdown-lite (bold, code, headers)
        let rendered = escapeHtml(explanation)
          .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
          .replace(/`([^`]+)`/g, '<code style="background:var(--code-bg);padding:1px 4px;border-radius:3px">$1</code>')
          .replace(/^### (.+)$/gm, '<div style="font-weight:600;margin-top:8px;color:var(--accent)">$1</div>')
          .replace(/^## (.+)$/gm, '<div style="font-weight:700;margin-top:10px;font-size:13px">$1</div>')
          .replace(/\n/g, '<br>');
        explainEl.innerHTML = '<div class="project-error-explain-content">' + rendered + '</div>';
      } else {
        explainEl.innerHTML = '<div style="color:var(--error)">Failed to get explanation. Try Fix Errors instead.</div>';
      }
    } catch {
      explainEl.innerHTML = '<div style="color:var(--error)">Could not reach the model. Check your connection.</div>';
    }
    explainBtn.disabled = false;
    explainBtn.textContent = 'Explain';
  });

  bar.querySelector('.code-fix-retry-btn').addEventListener('click', () => {
    bar.remove();
    _fixProjectErrors(card, node, uniqueErrors);
  });

  bar.querySelector('.code-fix-show-btn').addEventListener('click', () => {
    // Open files and highlight error lines
    const firstError = uniqueErrors[0];
    const errObj = typeof firstError === 'string' ? { text: firstError } : firstError;
    _expandProjectFiles(card, node, errObj.sourceFile || null, errObj.sourceLine || null);
  });

  bar.querySelector('.code-fix-ask-btn').addEventListener('click', () => {
    const consoleEl = card.querySelector('.project-console');
    if (consoleEl) consoleEl.hidden = !consoleEl.hidden;
  });

  // Insert after the preview, before file chips
  const preview = card.querySelector('.project-preview');
  const filesEl = card.querySelector('.project-files');
  if (filesEl) filesEl.before(bar);
  else if (preview) preview.after(bar);
  else card.appendChild(bar);
}

// ---------------------------------------------------------------------------
// Auto-fix project errors
// ---------------------------------------------------------------------------

async function _fixProjectErrors(card, node, errors) {
  const project = node.projectArtifact;
  if (!project || !project.files) return;

  // Circuit breaker: max 2 automatic fix attempts (learning from Lovable's fix spiral problem)
  if (!project._fixAttempts) project._fixAttempts = 0;
  project._fixAttempts++;
  if (project._fixAttempts > 2) {
    showToast('Auto-fix limit reached — edit files manually or revert', 'warning');
    // Show the errors with "Edit" as primary action
    const firstErr = errors[0];
    const errObj = typeof firstErr === 'string' ? { text: firstErr } : firstErr;
    _expandProjectFiles(card, node, errObj.sourceFile || null, errObj.sourceLine || null);
    return;
  }

  // Snapshot current state for version history (revert if fix makes things worse)
  if (!project.versions) project.versions = [];
  project.versions.push({
    timestamp: Date.now(),
    label: project.versions.length === 0 ? 'Initial build' : `Before fix (${errors.length} errors)`,
    files: project.files.map(f => ({ ...f, content: f.content })),
  });

  showToast('Fixing errors...', 'info');

  // Build targeted context: error file with line context + global exports from other files
  const errorDetails = errors.map((e, i) => {
    const err = typeof e === 'string' ? { text: e } : e;
    let detail = `${i + 1}. ${err.text}`;
    if (err.sourceFile && err.sourceLine) {
      detail = `${i + 1}. ${err.sourceFile}:${err.sourceLine} — ${err.text}`;
      // Add context lines from the file
      const file = project.files.find(f => f.path === err.sourceFile);
      if (file) {
        const lines = file.content.split('\n');
        const start = Math.max(0, err.sourceLine - 5);
        const end = Math.min(lines.length, err.sourceLine + 5);
        const context = lines.slice(start, end).map((l, j) => {
          const lineNum = start + j + 1;
          const marker = lineNum === err.sourceLine ? ' → ' : '   ';
          return `${marker}${lineNum}: ${l}`;
        }).join('\n');
        detail += `\n\nContext:\n${context}`;
      }
    }
    return detail;
  }).join('\n\n');

  // Build compact file summary (only affected files get full content, others get signature)
  const affectedFiles = new Set(errors.map(e => (typeof e === 'string' ? null : e.sourceFile)).filter(Boolean));
  const fileContext = project.files.map(f => {
    if (affectedFiles.has(f.path) || affectedFiles.size === 0) {
      return `=== ${f.path} (FULL — contains error) ===\n${f.content}`;
    }
    // For non-error files, just show global exports
    const globals = [];
    for (const m of f.content.matchAll(/(?:window\.(\w+)\s*=|(?:function|class|const|var|let)\s+(\w+))/g)) {
      globals.push(m[1] || m[2]);
    }
    return `=== ${f.path} (summary) ===\nExports: ${globals.join(', ') || 'none'}`;
  }).join('\n\n');

  // Use unified backend fix endpoint — same logic as pipeline verify pass:
  // build_fix_prompt (targeted context compression), _apply_file_patches
  // (3-tier fuzzy matching), rollback on regression, error preservation.
  const model = app.state.currentModel || 'default';
  const errTexts = errors.map(e => typeof e === 'string' ? e : e.text);
  {

    try {
      const resp = await fetch('/api/artifacts/fix', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          errors: errTexts, model,
          files: project.files.map(f => ({ path: f.path, role: f.role, lang: f.lang || '', content: f.content })),
          previous_attempts: project._fixPreviousAttempts || null,
        }),
      });
      if (!resp.ok) { showToast('Fix request failed', 'error'); return; }
      const data = await resp.json();
      const totalApplied = data.patches_applied || 0;
      if (!project._fixPreviousAttempts) project._fixPreviousAttempts = [];
      if (data.fix_response) project._fixPreviousAttempts.push(data.fix_response);
      if (!data.success && data.error) showToast(data.error, 'warning');
      if (data.files) { for (const updated of data.files) { const local = project.files.find(f => f.path === updated.path); if (local) local.content = updated.content; } }
      if (totalApplied > 0) {
        // Auto-generate changelog for version history
        const fixSummary = `Fixed ${totalApplied} issue${totalApplied > 1 ? 's' : ''}: ${errTexts.slice(0, 3).map(e => e.slice(0, 60)).join('; ')}${errTexts.length > 3 ? ` (+${errTexts.length - 3} more)` : ''}`;
        project.versions.push({
          timestamp: Date.now(),
          label: fixSummary,
          files: project.files.map(f => ({ ...f, content: f.content })),
        });
        showToast(`Applied ${totalApplied} fix${totalApplied > 1 ? 'es' : ''} — reloading preview...`, 'success');
        if (card._resetErrorCapture) card._resetErrorCapture();
        project.previewHtml = _assembleProject(project.files);
        project._assemblySourceMap = _lastAssemblySourceMap;
        const previewEl = card.querySelector('.project-preview');
        if (previewEl) {
          const oldIframe = previewEl.querySelector('iframe');
          if (oldIframe) { const nf = document.createElement('iframe'); nf.setAttribute('sandbox', 'allow-scripts allow-forms allow-modals allow-popups');
            nf.setAttribute('srcdoc', _buildPreviewSrcdoc(project.previewHtml)); oldIframe.replaceWith(nf); }
        }
        _bridge.saveSessions();
        setTimeout(() => { if (!card.querySelector('.project-fix-bar')) showToast('Preview is clean', 'success'); }, 3000);
      } else { showToast('Could not apply fixes automatically — try editing files manually', 'warning'); }
    } catch (err) { showToast(`Fix failed: ${err.message}`, 'error'); }
  }
}

// ---------------------------------------------------------------------------
// Project iteration (modify)
// ---------------------------------------------------------------------------

function _showProjectIterateBar(card, node) {
  const existing = card.querySelector('.project-iterate-bar');
  if (existing) { existing.remove(); return; }

  const bar = document.createElement('div');
  bar.className = 'project-iterate-bar';
  bar.innerHTML = `
    <div class="project-iterate-input-row">
      <input type="text" class="project-iterate-input" placeholder="Describe changes... e.g. 'add dark mode toggle' or 'fix the layout on mobile'">
      <button class="project-iterate-submit">Apply</button>
    </div>
    <div class="project-iterate-hint">Modifies the existing project via the full pipeline.</div>
  `;
  const input = bar.querySelector('.project-iterate-input');
  const submit = bar.querySelector('.project-iterate-submit');
  const doIterate = () => { const desc = input.value.trim(); if (!desc) return; bar.remove(); _iterateProject(card, node, desc); };
  submit.addEventListener('click', doIterate);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); doIterate(); } if (e.key === 'Escape') bar.remove(); });
  const actions = card.querySelector('.project-actions');
  if (actions) actions.after(bar); else card.appendChild(bar);
  input.focus();
}

async function _iterateProject(card, node, description, isResume = false) {
  // Double-click guard
  if (_iterateInFlight) { showToast('Build already in progress', 'info'); return; }
  _iterateInFlight = true;

  const project = node.projectArtifact;
  if (!project || !project.files) { _iterateInFlight = false; return; }

  // Snapshot for revert
  if (!project.versions) project.versions = [];
  project.versions.push({ timestamp: Date.now(), label: `Before: ${description.slice(0, 40)}`, files: project.files.map(f => ({ ...f, content: f.content })) });

  const model = app.state.currentModel || 'default';
  try {
    const reqBody = {
      description, scaffold: project.scaffold || 'static', model,
      session_id: app.state.activeSessionId || '',
      task_id: project.taskId || project.task_id || '',
      files: project.files.map(f => ({ path: f.path, role: f.role, lang: f.lang || '', content: f.content })),
    };
    if (isResume && project.planned_files) reqBody.planned_files = project.planned_files;

    // Fire-and-forget: start the build in the background
    const resp = await fetch('/api/artifacts/iterate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(reqBody),
    });
    if (!resp.ok) { showToast('Modification failed to start', 'error'); return; }

    const data = await resp.json();
    if (!data.started) { showToast('Modification failed to start', 'error'); return; }
    project.buildId = data.build_id || project.buildId || '';

    // Open the build monitor — it polls /api/artifacts/build-status for progress
    _ensureBuildMonitor(description.slice(0, 50), data.build_id || '');
    showToast('Modification started — monitor shows progress', 'info');

    // Store card reference so we can update it when the build completes
    if (_buildMonitor) {
      _buildMonitor._projectCard = card;
      _buildMonitor._projectNode = node;
      _buildMonitor._iterateDescription = description;
    }
  } catch (err) { showToast(`Modification failed: ${err.message}`, 'error'); }
  _iterateInFlight = false;
}

function _revertIterateUI(card, badge, overlay) {
  if (badge) { badge.className = 'project-card-badge complete'; badge.textContent = 'Ready'; }
  if (overlay) overlay.classList.remove('active');
}

// ---------------------------------------------------------------------------
// Expand project files (code blocks)
// ---------------------------------------------------------------------------

function _expandProjectFiles(card, node, focusPath, errorLine) {
  let expanded = card.querySelector('.project-expanded-files');
  if (expanded && !focusPath) { expanded.remove(); return; }
  if (expanded) { expanded.remove(); }

  expanded = document.createElement('div');
  expanded.className = 'project-expanded-files';

  const project = node.projectArtifact;
  for (const file of project.files) {
    const lang = file.lang || '';
    const encoded = encodeURIComponent(file.content);
    const blockId = blockFingerprint(lang, file.content);
    const isErrorFile = focusPath && file.path === focusPath;

    const header = document.createElement('div');
    header.className = 'code-header' + (isErrorFile ? ' code-header-error' : '');
    header.dataset.rawCode = encoded;
    header.dataset.lang = lang;
    header.dataset.versionIdx = '0';
    header.dataset.blockId = blockId;

    let actionBtns = '';
    if (lang) {
      actionBtns += `<button class="code-action-btn" data-action="ask-ai-edit" title="Ask AI to edit">Ask AI</button>`;
      actionBtns += `<button class="code-action-btn code-quick-actions-trigger" data-action="quick-actions" title="Quick actions">&#9662;</button>`;
      actionBtns += `<button class="code-action-btn" data-action="edit-code" title="Edit code">Edit</button>`;
    }
    actionBtns += `<button class="copy-code-btn" data-copy="${encoded}">Copy</button>`;

    const errorBadge = isErrorFile && errorLine ? ` <span class="code-error-badge">error line ${Number(errorLine)}</span>` : '';
    header.innerHTML = '<span>' + escapeHtml(file.path) + errorBadge + '</span><div class="code-header-actions">' + actionBtns + '</div>';

    const pre = document.createElement('pre');
    const codeEl = document.createElement('code');
    codeEl.className = lang ? `language-${escapeHtml(lang)}` : '';
    codeEl.textContent = file.content;
    pre.appendChild(codeEl);

    expanded.appendChild(header);
    expanded.appendChild(pre);

    safeHighlightElement(codeEl);

    // Highlight error line if this is the error file
    if (isErrorFile && errorLine && codeEl.dataset.numbered) {
      const lineEls = codeEl.querySelectorAll('.code-line');
      if (lineEls[errorLine - 1]) {
        lineEls[errorLine - 1].classList.add('code-line-error');
        // Scroll to error line after card is appended
        setTimeout(() => lineEls[errorLine - 1].scrollIntoView({ behavior: 'smooth', block: 'center' }), 100);
      }
    }
  }

  card.appendChild(expanded);
  expanded.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ---------------------------------------------------------------------------
// Accessors for pending project state (used by stream finalization in chat.js)
// ---------------------------------------------------------------------------

/** Return and clear the pending project artifact. */
export function consumePendingProject() {
  const p = _pendingProject;
  _pendingProject = null;
  return p;
}

/** Return the current pending project without clearing it. */
export function getPendingProject() {
  return _pendingProject;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export {
  handleProjectProgress,
  handleProjectResult,
  renderProjectCard,
  _ensureBuildMonitor as ensureBuildMonitor,
  recoverBuildMonitor,
  _deliverBuildToChat as deliverBuildToChat,
};
