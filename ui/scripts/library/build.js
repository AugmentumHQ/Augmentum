/**
 * library/build.js — in-Library "Build an app" surface.
 *
 * Kicks off the autonomous builder (POST /api/builds → the coder-harness
 * build-test-fix loop), shows a live "building" card in the main pane that
 * accumulates the agent's tool trail in realtime (polling /api/builds/{id}),
 * and on completion hands off to the orchestrator to surface the finished
 * artifact in the grid.
 *
 * The controller owns its own card DOM (a strip the main pane mounts at the
 * top of its body) so it survives selection changes and updates in place
 * without re-fetching the library. Active build ids persist to localStorage
 * so a refresh re-attaches the monitors.
 *
 * Labels are persona-agnostic ("Build an app", "Building…") — Augmentum is
 * OSS and deployments name their own surfaces.
 */

import { escapeHtml } from '../app.js';

const POLL_MS = 2000;
const LS_ACTIVE = 'augmentum.library.activeBuilds';
const TERMINAL = new Set(['complete', 'completed', 'error', 'cancelled', 'canceled', 'failed']);

const _spinnerIcon = '<svg class="lib-build-spin" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M12 3a9 9 0 1 0 9 9" /></svg>';
const _checkIcon = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12l5 5L21 6"/></svg>';
const _alertIcon = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>';
const _closeIcon = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
const _buildIcon = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 7l-1.5-1.5a3 3 0 0 0-4 4L10 11l-5 5 3 3 5-5 1.5 1.5a3 3 0 0 0 4-4L17 9"/></svg>';


export class BuildController {
  /**
   * @param {object} opts
   * @param {import('./main-pane.js').MainPane} opts.mainPane
   * @param {(item:object)=>void} [opts.onComplete] called when a build finishes
   *   successfully (orchestrator reloads the library to surface the artifact)
   */
  constructor({ mainPane, onComplete } = {}) {
    this._main = mainPane;
    this._onComplete = onComplete || (() => {});
    this._active = new Map();    // build_id -> item
    this._monitors = new Map();  // build_id -> interval id
    this._dialog = null;
    this._wireClicks();
    this._restore();
  }

  /** Active building items, newest first — consumed by MainPane's strip. */
  list() {
    return [...this._active.values()].reverse();
  }

  hasActive() {
    return this._active.size > 0;
  }

  // ── card interactions (delegated; survives strip re-renders) ────────

  _wireClicks() {
    const host = this._main?.host;
    if (!host) return;
    host.addEventListener('click', (ev) => {
      const cancel = ev.target.closest('[data-build-cancel]');
      if (cancel) {
        ev.preventDefault();
        ev.stopPropagation();
        this.cancel(cancel.getAttribute('data-build-cancel'));
        return;
      }
      const dismiss = ev.target.closest('[data-build-dismiss]');
      if (dismiss) {
        ev.preventDefault();
        ev.stopPropagation();
        this._remove(dismiss.getAttribute('data-build-dismiss'));
        return;
      }
      const openWs = ev.target.closest('[data-build-open-ws]');
      if (openWs) {
        ev.preventDefault();
        ev.stopPropagation();
        this._openWorkspace(openWs.getAttribute('data-build-open-ws'));
        return;
      }
      const openCode = ev.target.closest('[data-build-open-code]');
      if (openCode) {
        ev.preventDefault();
        ev.stopPropagation();
        this._openInCode(openCode.getAttribute('data-build-open-code'));
        return;
      }
      const retry = ev.target.closest('[data-build-retry]');
      if (retry) {
        ev.preventDefault();
        ev.stopPropagation();
        this._retry(retry.getAttribute('data-build-retry'));
        return;
      }
      const resume = ev.target.closest('[data-build-resume]');
      if (resume) {
        ev.preventDefault();
        ev.stopPropagation();
        this._resume(resume.getAttribute('data-build-resume'), false);
        return;
      }
      const resumePrompt = ev.target.closest('[data-build-resume-prompt]');
      if (resumePrompt) {
        ev.preventDefault();
        ev.stopPropagation();
        this._resume(resumePrompt.getAttribute('data-build-resume-prompt'), true);
        return;
      }
      const details = ev.target.closest('[data-build-details]');
      if (details) {
        ev.preventDefault();
        ev.stopPropagation();
        const id = details.getAttribute('data-build-details');
        const pre = host.querySelector(`[data-build-detail-for="${_cssEscape(id)}"]`);
        if (pre) pre.classList.toggle('hidden');
      }
    });
  }

  // ── failed-build recovery ───────────────────────────────────────────

  /** Jump into coder mode on the workspace the failed build left behind. */
  async _openWorkspace(id) {
    const item = this._active.get(id);
    const wsId = item?.workspace_id;
    if (!wsId) { this._toast('No workspace was created for this build', 'warning'); return; }
    try {
      const coder = await import('../coder.js');
      if (typeof coder.openWorkspaceById === 'function') {
        await coder.openWorkspaceById(wsId);
      }
      document.querySelector('[data-mode="coder"]')?.click();
      this._toast('Opened the build workspace in Code', 'success');
    } catch (err) {
      this._toast(`Couldn't open the workspace: ${err.message || err}`, 'error');
    }
  }

  /** Fallback when no live workspace remains: create one from the artifact. */
  async _openInCode(id) {
    const item = this._active.get(id);
    if (!item) return;
    try {
      this._toast('Creating Code workspace…', 'info');
      const buildId = item.build_id || id;
      const r = await fetch(`/api/builds/${encodeURIComponent(buildId)}/open-in-code`, {
        method: 'POST', credentials: 'same-origin',
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      const coder = await import('../coder.js');
      if (typeof coder.openWorkspaceById === 'function' && data.workspace_id) {
        await coder.openWorkspaceById(data.workspace_id);
      }
      document.querySelector('[data-mode="coder"]')?.click();
      this._toast('Code workspace ready', 'success');
    } catch (err) {
      this._toast(`Code handoff failed: ${err.message || err}`, 'error');
    }
  }

  /** Re-run the same build description as a fresh build. */
  _retry(id) {
    const item = this._active.get(id);
    const desc = item?.description;
    if (!desc) { this._toast('Nothing to retry — original description is gone', 'warning'); return; }
    const model = item.model || '';
    this._remove(id);
    this.start(desc, model);
  }

  /**
   * Continue a stopped/finished build on its existing workspace (resume), or
   * re-prompt it with new instructions. Unlike retry, this keeps the work and
   * picks up where the agent left off.
   * @param {string} id
   * @param {boolean} withPrompt  prompt the user for new instructions first
   */
  async _resume(id, withPrompt) {
    const item = this._active.get(id);
    if (!item) return;
    const buildId = item.build_id || id;
    let instructions = '';
    if (withPrompt) {
      instructions = (await this._promptInstructions(item)) || '';
      // Empty + cancelled prompt => abort; empty + confirmed => plain continue.
      if (instructions === null) return;
    }
    try {
      this._toast(instructions ? 'Continuing with your changes…' : 'Continuing the build…', 'info');
      const body = {};
      if (instructions) body.instructions = instructions;
      if (item.model) body.model = item.model;
      const r = await fetch(`/api/builds/${encodeURIComponent(buildId)}/resume`, {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      // Re-activate the card and re-attach the monitor.
      item.status = 'running';
      item.error = '';
      item.errorDetail = '';
      item.qualityStatus = '';
      item.resume_count = data.resume_count || (item.resume_count || 0) + 1;
      if (data.workspace_id) item.workspace_id = data.workspace_id;
      this._persist();
      this._updateCard(id);
      this._monitor(id);
    } catch (err) {
      this._toast(`Couldn't continue the build: ${err.message || err}`, 'error');
    }
  }

  /** Minimal inline prompt for resume-with-changes. Resolves to the entered
   *  text, '' for a confirmed-but-empty continue, or null if cancelled. */
  _promptInstructions(item) {
    return new Promise((resolve) => {
      const wrap = document.createElement('div');
      wrap.className = 'lib-build-modal';
      wrap.setAttribute('role', 'dialog');
      wrap.setAttribute('aria-modal', 'true');
      wrap.setAttribute('aria-label', 'Continue build with changes');
      wrap.innerHTML = `
        <div class="lib-build-modal-backdrop" data-rp-close></div>
        <div class="lib-build-modal-panel" role="document">
          <h2 class="lib-build-modal-title"><span>Continue with changes</span></h2>
          <p class="lib-build-modal-hint">
            Tell the builder what to change or add. It keeps the existing app and
            continues from where it left off. Leave blank to just keep going.
          </p>
          <textarea class="lib-build-input" rows="3"
            placeholder="e.g. add a dark-mode toggle and fix the broken total"></textarea>
          <div class="lib-build-modal-actions">
            <button type="button" class="lib-build-btn-cancel" data-rp-close>Cancel</button>
            <button type="button" class="lib-build-btn-go"><span>Continue</span></button>
          </div>
        </div>`;
      document.body.appendChild(wrap);
      const input = wrap.querySelector('.lib-build-input');
      const go = wrap.querySelector('.lib-build-btn-go');
      const done = (val) => { wrap.remove(); resolve(val); };
      wrap.addEventListener('click', (ev) => {
        if (ev.target.closest('[data-rp-close]')) { ev.preventDefault(); done(null); }
      });
      wrap.addEventListener('keydown', (ev) => {
        if (ev.key === 'Escape') { ev.preventDefault(); done(null); }
        if (ev.key === 'Enter' && (ev.metaKey || ev.ctrlKey)) { ev.preventDefault(); done((input.value || '').trim()); }
      });
      go.addEventListener('click', () => done((input.value || '').trim()));
      setTimeout(() => input.focus(), 0);
    });
  }

  // ── kickoff dialog ──────────────────────────────────────────────────

  openDialog() {
    if (this._dialog) {
      this._dialog.remove();
      this._dialog = null;
    }
    const wrap = document.createElement('div');
    wrap.className = 'lib-build-modal';
    wrap.setAttribute('role', 'dialog');
    wrap.setAttribute('aria-modal', 'true');
    wrap.setAttribute('aria-label', 'Build an app');
    wrap.innerHTML = `
      <div class="lib-build-modal-backdrop" data-build-modal-close></div>
      <div class="lib-build-modal-panel" role="document">
        <h2 class="lib-build-modal-title">${_buildIcon}<span>Build an app</span></h2>
        <p class="lib-build-modal-hint">
          Describe the app you want. An agent writes it, runs it in a real
          browser, verifies every behavior, then publishes it here — ready to play.
        </p>
        <textarea class="lib-build-input" rows="4"
          placeholder="e.g. a tip calculator with split-by-people and a clear button"></textarea>
        <label class="lib-build-model">
          <span>Model <span class="lib-build-model-opt">(optional)</span></span>
          <input type="text" class="lib-build-model-input"
                 placeholder="leave blank for your default model"
                 autocomplete="off" spellcheck="false">
        </label>
        <div class="lib-build-modal-actions">
          <button type="button" class="lib-build-btn-cancel" data-build-modal-close>Cancel</button>
          <button type="button" class="lib-build-btn-go">${_buildIcon}<span>Build</span></button>
        </div>
      </div>
    `;
    document.body.appendChild(wrap);
    this._dialog = wrap;

    const input = wrap.querySelector('.lib-build-input');
    const modelInput = wrap.querySelector('.lib-build-model-input');
    const go = wrap.querySelector('.lib-build-btn-go');

    const close = () => {
      wrap.remove();
      if (this._dialog === wrap) this._dialog = null;
    };
    const submit = () => {
      const desc = (input.value || '').trim();
      if (!desc) {
        input.focus();
        input.classList.add('lib-build-input-error');
        return;
      }
      close();
      this.start(desc, (modelInput.value || '').trim());
    };

    wrap.addEventListener('click', (ev) => {
      if (ev.target.closest('[data-build-modal-close]')) {
        ev.preventDefault();
        close();
      }
    });
    wrap.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') { ev.preventDefault(); close(); }
      // ⌘/Ctrl+Enter submits from the textarea.
      if (ev.key === 'Enter' && (ev.metaKey || ev.ctrlKey)) { ev.preventDefault(); submit(); }
    });
    go.addEventListener('click', submit);
    input.addEventListener('input', () => input.classList.remove('lib-build-input-error'));
    setTimeout(() => input.focus(), 0);
  }

  // ── start + monitor ─────────────────────────────────────────────────

  async start(description, model = '') {
    const body = { description };
    if (model) body.model = model;
    let data;
    try {
      const r = await fetch('/api/builds', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const txt = await r.text().catch(() => '');
        throw new Error(`HTTP ${r.status} ${txt.slice(0, 140)}`);
      }
      data = await r.json();
    } catch (err) {
      this._toast(`Couldn't start the build: ${err.message || err}`, 'error');
      return;
    }
    const id = data.build_id;
    if (!id) {
      this._toast("Couldn't start the build: no build id returned", 'error');
      return;
    }
    this._active.set(id, {
      build_id: id,
      id,                       // some surfaces key on .id
      _type: 'building',
      description,
      display_name: data.name || description.slice(0, 48),
      model: data.model || model || '',
      status: 'running',
      steps: [],
      llmCalls: 0,
      currentFile: '',
      artifact_id: '',
      workspace_id: '',
      error: '',
      errorDetail: '',
      blockingErrors: [],
      qualityStatus: '',
      verdict: null,
      warnings: [],
      behaviors: [],
      resume_count: 0,
    });
    this._persist();
    this._main?.refreshBuilds?.();
    this._monitor(id);
  }

  _monitor(id) {
    if (this._monitors.has(id)) return;
    const tick = async () => {
      let snap;
      try {
        const r = await fetch(`/api/builds/${encodeURIComponent(id)}`, {
          credentials: 'same-origin',
          headers: { Accept: 'application/json' },
        });
        if (!r.ok) return;  // transient (e.g. 404 before the run row lands) — keep polling
        snap = (await r.json()).run || {};
      } catch {
        return;  // network blip — keep polling
      }
      const item = this._active.get(id);
      if (!item) return;
      item.status = snap.status || item.status;
      if (snap.name) item.display_name = snap.name;
      if (Array.isArray(snap.steps)) item.steps = snap.steps;
      item.llmCalls = snap.llmCalls || item.llmCalls;
      item.currentFile = snap.currentFile || item.currentFile;
      item.artifact_id = snap.artifact_id || item.artifact_id;
      item.workspace_id = snap.workspace_id || item.workspace_id;
      if (snap.error) item.error = snap.error;
      if (snap.errorDetail) item.errorDetail = snap.errorDetail;
      if (Array.isArray(snap.blockingErrors) && snap.blockingErrors.length) {
        item.blockingErrors = snap.blockingErrors;
      }
      if (snap.qualityStatus) item.qualityStatus = snap.qualityStatus;
      if (snap.verdict && typeof snap.verdict === 'object') item.verdict = snap.verdict;
      if (Array.isArray(snap.warnings) && snap.warnings.length) item.warnings = snap.warnings;
      if (Array.isArray(snap.behaviors) && snap.behaviors.length) item.behaviors = snap.behaviors;
      if (typeof snap.resume_count === 'number') item.resume_count = snap.resume_count;
      this._updateCard(id);
      if (TERMINAL.has(String(item.status).toLowerCase())) {
        this._stopMonitor(id);
        this._onTerminal(item);
      }
    };
    const timer = setInterval(tick, POLL_MS);
    this._monitors.set(id, timer);
    tick();
  }

  _onTerminal(item) {
    const ok = ['complete', 'completed'].includes(String(item.status).toLowerCase());
    const unverified = String(item.qualityStatus || '').toLowerCase() === 'unverified';
    this._persist();
    if (ok && !unverified) {
      this._toast(`Built “${item.display_name}”`, 'success');
      // Surface the finished artifact in the grid, then retire the card.
      try { this._onComplete(item); } catch { /* orchestrator owns reload */ }
      setTimeout(() => this._remove(item.build_id), 4000);
    } else if (ok && unverified) {
      // Built, but the agent never proved it works. Surface it in the grid
      // (it IS published) but keep the card so the user can finish verifying.
      this._toast(`Built “${item.display_name}” — not fully verified`, 'warning');
      try { this._onComplete(item); } catch { /* orchestrator owns reload */ }
      this._updateCard(item.build_id);
    } else {
      // Leave a dismissable error/cancelled card so the failure is visible.
      this._updateCard(item.build_id);
    }
  }

  async cancel(id) {
    const item = this._active.get(id);
    if (item && TERMINAL.has(String(item.status).toLowerCase())) {
      this._remove(id);
      return;
    }
    try {
      await fetch(`/api/builds/${encodeURIComponent(id)}/cancel`, {
        method: 'POST', credentials: 'same-origin',
      });
    } catch { /* best-effort */ }
    if (item) { item.status = 'cancelled'; this._updateCard(id); }
    this._stopMonitor(id);
    setTimeout(() => this._remove(id), 2500);
  }

  // ── card rendering / in-place update ────────────────────────────────

  _updateCard(id) {
    const host = this._main?.host;
    const item = this._active.get(id);
    if (!host || !item) return;
    const existing = host.querySelector(`[data-build-id="${_cssEscape(id)}"]`);
    if (existing) {
      const next = _cardElement(item);
      existing.replaceWith(next);
    } else {
      // Card not mounted (strip absent) — ask the pane to (re)render it.
      this._main?.refreshBuilds?.();
    }
  }

  _remove(id) {
    this._stopMonitor(id);
    this._active.delete(id);
    this._persist();
    this._main?.refreshBuilds?.();
  }

  _stopMonitor(id) {
    const t = this._monitors.get(id);
    if (t) { clearInterval(t); this._monitors.delete(id); }
  }

  // ── persistence (re-attach monitors across a refresh) ───────────────

  _persist() {
    try {
      const running = [...this._active.values()]
        .filter((it) => !TERMINAL.has(String(it.status).toLowerCase()))
        .map((it) => it.build_id);
      localStorage.setItem(LS_ACTIVE, JSON.stringify(running));
    } catch { /* private mode — fine */ }
  }

  _restore() {
    let ids = [];
    try { ids = JSON.parse(localStorage.getItem(LS_ACTIVE) || '[]'); } catch { ids = []; }
    for (const id of ids) {
      if (!id || this._active.has(id)) continue;
      this._active.set(id, {
        build_id: id, id, _type: 'building',
        description: '', display_name: 'Building…', model: '',
        status: 'running', steps: [], llmCalls: 0,
        currentFile: '', artifact_id: '', workspace_id: '', error: '',
        errorDetail: '', blockingErrors: [],
        qualityStatus: '', verdict: null, warnings: [], behaviors: [], resume_count: 0,
      });
      this._monitor(id);
    }
  }

  _toast(msg, kind) {
    import('../app.js').then((m) => m.showToast?.(msg, kind)).catch(() => {});
  }
}


// ── card HTML (string + element) ──────────────────────────────────────

/** HTML string for one building card. Used by MainPane's strip render. */
export function renderBuildCard(item) {
  if (!item) return '';
  const status = String(item.status || 'running').toLowerCase();
  const id = item.build_id || item.id || '';
  const title = item.display_name || item.description || 'Building…';
  const steps = Array.isArray(item.steps) ? item.steps : [];
  const stepCount = steps.length;
  const behaviors = Array.isArray(item.behaviors) ? item.behaviors : [];
  const bChecked = behaviors.filter((b) => b.status === 'pass' || b.status === 'fail').length;
  const bPassed = behaviors.filter((b) => b.status === 'pass').length;

  let stateClass = 'is-running';
  let icon = _spinnerIcon;
  let statusText;
  let action;
  let errorBlock = '';
  if (status === 'complete' || status === 'completed') {
    const unverified = String(item.qualityStatus || '').toLowerCase() === 'unverified';
    if (unverified) {
      // Built + published, but the agent never proved it works (no serve/drive/
      // assert). Surface the gap and let the user finish verifying on the same
      // workspace instead of discovering a broken app on open.
      stateClass = 'is-unverified';
      icon = _alertIcon;
      const warnings = Array.isArray(item.warnings) ? item.warnings : [];
      const unproven = (item.verdict && Array.isArray(item.verdict.unproven)) ? item.verdict.unproven : [];
      const n = unproven.length || warnings.length;
      statusText = n
        ? `Built — but ${n} check${n === 1 ? '' : 's'} not verified`
        : 'Built — but not fully verified';
      action = `<button type="button" class="lib-build-x" data-build-dismiss="${escapeHtml(id)}" aria-label="Dismiss" title="Dismiss">${_closeIcon}</button>`;
      const detail = (warnings.length ? warnings : unproven).join('\n');
      const hasDetail = !!detail;
      const btns = [
        `<button type="button" class="lib-build-recover-btn primary" data-build-resume="${escapeHtml(id)}">Finish verifying</button>`,
        `<button type="button" class="lib-build-recover-btn" data-build-resume-prompt="${escapeHtml(id)}">Continue with changes&hellip;</button>`,
      ];
      if (item.workspace_id) {
        btns.push(`<button type="button" class="lib-build-recover-btn ghost" data-build-open-ws="${escapeHtml(id)}">Open workspace &rarr;</button>`);
      }
      if (hasDetail) {
        btns.push(`<button type="button" class="lib-build-recover-btn ghost" data-build-details="${escapeHtml(id)}">Details</button>`);
      }
      const detailPre = hasDetail
        ? `<pre class="lib-build-error-detail hidden" data-build-detail-for="${escapeHtml(id)}">${escapeHtml(detail)}</pre>`
        : '';
      errorBlock = `<div class="lib-build-recover">${btns.join('')}</div>${detailPre}`;
    } else {
      stateClass = 'is-done';
      icon = _checkIcon;
      statusText = bChecked
        ? `Built · ${bPassed}/${bChecked} behaviors verified`
        : `Built · ${stepCount} step${stepCount === 1 ? '' : 's'} verified`;
      action = '';  // auto-retires
    }
  } else if (status === 'cancelled' || status === 'canceled') {
    stateClass = 'is-cancelled';
    icon = _alertIcon;
    statusText = 'Cancelled';
    action = `<button type="button" class="lib-build-x" data-build-dismiss="${escapeHtml(id)}" aria-label="Dismiss" title="Dismiss">${_closeIcon}</button>`;
  } else if (status === 'error' || status === 'failed') {
    stateClass = 'is-error';
    icon = _alertIcon;
    action = `<button type="button" class="lib-build-x" data-build-dismiss="${escapeHtml(id)}" aria-label="Dismiss" title="Dismiss">${_closeIcon}</button>`;
    // A failed build still saves its work: the agent's workspace stays live
    // and (when any files were produced) the artifact is published. So this
    // is a recovery surface, not a tombstone — offer the doors the user would
    // otherwise have to find by hand.
    const wsId = item.workspace_id || '';
    const artifactId = item.artifact_id || '';
    const desc = item.description || '';
    const blocking = Array.isArray(item.blockingErrors) ? item.blockingErrors : [];
    const detail = [
      ...(blocking.length ? blocking : []),
      item.errorDetail ? String(item.errorDetail) : '',
    ].filter(Boolean).join('\n');
    const hasDetail = !!(detail || item.error);
    statusText = (wsId || artifactId)
      ? 'Build stopped early — your work was saved'
      : (item.error ? String(item.error).slice(0, 140) : 'Build failed');

    // Resume keeps the work and continues where the agent stopped — the right
    // first move for a budget/stuck/interrupted build. Possible whenever a
    // workspace or a published artifact survives (the backend rebuilds from
    // the artifact if the container is gone).
    const canResume = !!(wsId || artifactId);
    const btns = [];
    if (canResume) {
      btns.push(`<button type="button" class="lib-build-recover-btn primary" data-build-resume="${escapeHtml(id)}">Continue build</button>`);
      btns.push(`<button type="button" class="lib-build-recover-btn" data-build-resume-prompt="${escapeHtml(id)}">Continue with changes&hellip;</button>`);
    }
    if (wsId) {
      btns.push(`<button type="button" class="lib-build-recover-btn ghost" data-build-open-ws="${escapeHtml(id)}">Open workspace &rarr;</button>`);
    } else if (artifactId) {
      btns.push(`<button type="button" class="lib-build-recover-btn ghost" data-build-open-code="${escapeHtml(id)}">Open in Code</button>`);
    }
    if (desc) {
      btns.push(`<button type="button" class="lib-build-recover-btn ghost" data-build-retry="${escapeHtml(id)}">Start over</button>`);
    }
    if (hasDetail) {
      btns.push(`<button type="button" class="lib-build-recover-btn ghost" data-build-details="${escapeHtml(id)}">Details</button>`);
    }
    const detailPre = hasDetail
      ? `<pre class="lib-build-error-detail hidden" data-build-detail-for="${escapeHtml(id)}">${escapeHtml(detail || item.error || '')}</pre>`
      : '';
    errorBlock = `
      <div class="lib-build-recover">${btns.join('')}</div>
      ${detailPre}`;
  } else {
    const cur = (item.currentFile || '').replace(/\s+/g, ' ').trim();
    statusText = cur
      ? `${stepCount} step${stepCount === 1 ? '' : 's'} · ${cur.slice(0, 90)}`
      : 'Starting…';
    action = `<button type="button" class="lib-build-x" data-build-cancel="${escapeHtml(id)}" aria-label="Cancel build" title="Cancel">${_closeIcon}</button>`;
  }

  const running = stateClass === 'is-running';
  const bar = running
    ? '<div class="lib-build-bar"><span class="lib-build-bar-fill"></span></div>'
    : '';
  const trail = running && steps.length
    ? `<div class="lib-build-trail">${steps.slice(-5)
        .map((s) => `<span class="lib-build-chip">${escapeHtml(String(s.tool || ''))}</span>`)
        .join('')}</div>`
    : '';
  // Spec-derived acceptance checklist with per-behavior pass/fail. Shown on
  // terminal + verifying states so the user sees exactly what was proven.
  const behaviorsBlock = (behaviors.length && stateClass !== 'is-cancelled')
    ? renderBehaviorsBlock(behaviors)
    : '';
  const model = item.model
    ? `<span class="lib-build-model-tag">${escapeHtml(item.model)}</span>`
    : '';
  const resumed = (Number(item.resume_count) > 0)
    ? `<span class="lib-build-model-tag" title="continued ${Number(item.resume_count)}×">&#8635; ${Number(item.resume_count)}</span>`
    : '';

  return `
    <div class="lib-build-card ${stateClass}" data-build-id="${escapeHtml(id)}">
      <div class="lib-build-head">
        <span class="lib-build-icon" aria-hidden="true">${icon}</span>
        <span class="lib-build-name">${escapeHtml(title)}</span>
        ${model}
        ${resumed}
        ${action}
      </div>
      <div class="lib-build-status">${escapeHtml(statusText)}</div>
      ${bar}
      ${trail}
      ${behaviorsBlock}
      ${errorBlock}
    </div>`;
}

/** Compact acceptance-checklist: one row per behavior with a pass/fail/pending
 *  marker. Collapses to a count header so long lists stay tidy. */
function renderBehaviorsBlock(behaviors) {
  const rows = behaviors.map((b) => {
    const st = String(b.status || 'untested');
    const mark = st === 'pass' ? '✓' : st === 'fail' ? '✗' : '·';
    const ev = (st === 'fail' && b.evidence) ? ` — ${String(b.evidence)}` : '';
    return `<li class="lib-build-behavior is-${escapeHtml(st)}">
      <span class="lib-build-behavior-mark" aria-hidden="true">${mark}</span>
      <span class="lib-build-behavior-text">${escapeHtml(String(b.description || ''))}${escapeHtml(ev)}</span>
    </li>`;
  }).join('');
  const checked = behaviors.filter((b) => b.status === 'pass' || b.status === 'fail').length;
  const passed = behaviors.filter((b) => b.status === 'pass').length;
  const head = checked
    ? `${passed}/${checked} behaviors verified`
    : `${behaviors.length} acceptance checks`;
  return `<details class="lib-build-behaviors"${passed < checked ? ' open' : ''}>
      <summary>${escapeHtml(head)}</summary>
      <ul class="lib-build-behavior-list">${rows}</ul>
    </details>`;
}

function _cardElement(item) {
  const tpl = document.createElement('template');
  tpl.innerHTML = renderBuildCard(item).trim();
  return tpl.content.firstElementChild;
}

function _cssEscape(s) {
  if (window.CSS && CSS.escape) return CSS.escape(s);
  return String(s).replace(/["\\]/g, '\\$&');
}
