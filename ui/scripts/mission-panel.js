/**
 * mission-panel.js — Renders the coder mode mission log as a live checklist.
 *
 * Subscribes to typed events from CoderStream and updates the panel DOM
 * in place.  Each promise is a row with status icon, description, and
 * evidence.  The panel pins to the top of the conversation pane.
 *
 * The mission data model is driven by the backend augmentum.promises
 * runtime — this module is purely presentational.
 */

import { escapeHtml } from './app.js';

const STATUS_ICON = {
  pending:      '<span class="mission-icon mission-icon--pending" aria-label="pending">○</span>',
  in_progress:  '<span class="mission-icon mission-icon--running" aria-label="in progress">◐</span>',
  fulfilled:    '<span class="mission-icon mission-icon--done" aria-label="done">✓</span>',
  rejected:     '<span class="mission-icon mission-icon--failed" aria-label="failed">✕</span>',
};

/**
 * @typedef {Object} PromiseSummary
 * @property {string} id
 * @property {string} description
 * @property {string} status
 * @property {number} attempts
 * @property {number} max_attempts
 * @property {string} verify_kind
 * @property {string|null} [evidence]
 */

export class MissionPanel {
  /**
   * @param {HTMLElement} mountEl — container DOM element
   */
  constructor(mountEl) {
    this._mount = mountEl;
    /** @type {Map<string, PromiseSummary>} */
    this._promises = new Map();
    /** @type {string[]} Preserves planner's original ordering */
    this._order = [];
    this._collapsed = false;
    this._failed = false;
    this._completed = false;
    this._activeId = null;
    /** @type {Map<string, {until:number, attempt:number, maxRetries:number, reason:string}>} */
    this._rateLimitById = new Map();
    this._countdownTimer = null;
    this._render();
  }

  /**
   * Initialize a fresh mission. Resets state and renders placeholders.
   * @param {PromiseSummary[]} mission
   */
  start(mission) {
    this._promises.clear();
    this._order = [];
    this._failed = false;
    this._completed = false;
    this._activeId = null;
    this._rateLimitById.clear();
    this._stopCountdownTick();
    for (const p of mission || []) {
      if (!p?.id) continue;
      this._promises.set(p.id, { ...p });
      this._order.push(p.id);
    }
    this._render();
    this.show();
  }

  /** Clear the panel entirely (e.g. mission aborted, workspace switched). */
  clear() {
    this._promises.clear();
    this._order = [];
    this._activeId = null;
    this._failed = false;
    this._completed = false;
    this._rateLimitById.clear();
    this._stopCountdownTick();
    this._render();
  }

  /** Hide the panel from the DOM. */
  hide() {
    this._mount.classList.add('mission-panel--hidden');
  }

  /** Show the panel. */
  show() {
    this._mount.classList.remove('mission-panel--hidden');
  }

  /**
   * Merge a partial promise update. If id is unknown we insert it at the
   * end (covers decomposition children injected mid-mission).
   * @param {PromiseSummary} update
   */
  _merge(update) {
    if (!update?.id) return;
    const prev = this._promises.get(update.id) || {};
    this._promises.set(update.id, { ...prev, ...update });
    if (!this._order.includes(update.id)) {
      this._order.push(update.id);
    }
  }

  // ── Event handlers ────────────────────────────────────────────────────

  onPromiseStarted(promise) {
    this._merge({ ...promise, status: 'in_progress' });
    this._activeId = promise.id;
    this._rateLimitById.delete(promise.id);  // new attempt — clear old backoff
    this._render();
  }

  onPromiseVerifying(promise) {
    this._merge(promise);
    this._flashRow(promise.id, 'verifying');
  }

  onPromiseFulfilled(promise) {
    this._merge({ ...promise, status: 'fulfilled' });
    if (this._activeId === promise.id) this._activeId = null;
    this._rateLimitById.delete(promise.id);
    this._render();
  }

  onPromiseRetry(promise, reason) {
    this._merge({ ...promise, status: 'in_progress', evidence: reason });
    this._flashRow(promise.id, 'retry');
  }

  onPromiseRejected(promise, reason) {
    this._merge({
      ...promise,
      status: 'rejected',
      evidence: promise.evidence || reason,
    });
    if (this._activeId === promise.id) this._activeId = null;
    this._failed = true;
    this._render();
  }

  /**
   * Backend hit a transient error (429 / 503 / timeout) and is backing
   * off. The active row gets a "paused" indicator with a live countdown
   * until the next retry fires.
   *
   * @param {Object} info
   * @param {Object} [info.promise]     — Promise summary (may be missing)
   * @param {number} info.waitSeconds   — Backoff duration in seconds
   * @param {number} info.attempt       — 1-based retry index
   * @param {number} [info.maxRetries]  — Total retries allowed
   * @param {string} [info.reason]      — Short reason label (e.g. "rate limited (429)")
   */
  onRateLimited({ promise, waitSeconds, attempt, maxRetries = 3, reason = '' }) {
    const id = promise?.id || this._activeId;
    if (!id) return;
    if (!this._promises.has(id) && promise) this._merge(promise);
    this._rateLimitById.set(id, {
      until: Date.now() + Math.max(1, waitSeconds) * 1000,
      attempt,
      maxRetries,
      reason,
    });
    this._render();
    this._ensureCountdownTick();
  }

  onPromiseDecomposed(promise, children) {
    this._merge(promise);
    // Insert children directly after the parent in the display order
    const parentIdx = this._order.indexOf(promise.id);
    let insertAt = parentIdx >= 0 ? parentIdx + 1 : this._order.length;
    for (const child of children || []) {
      if (!child?.id) continue;
      this._promises.set(child.id, { ...child, _parentId: promise.id });
      this._order.splice(insertAt++, 0, child.id);
    }
    this._render();
  }

  onMissionCompleted() {
    this._completed = true;
    this._activeId = null;
    this._render();
  }

  onMissionFailed() {
    this._failed = true;
    this._activeId = null;
    this._render();
  }

  // ── Rendering ─────────────────────────────────────────────────────────

  _counts() {
    let done = 0, failed = 0, total = 0;
    for (const p of this._promises.values()) {
      total++;
      if (p.status === 'fulfilled') done++;
      else if (p.status === 'rejected') failed++;
    }
    return { done, failed, total };
  }

  _render() {
    if (!this._mount) return;
    if (this._order.length === 0) {
      this._mount.innerHTML = '';
      return;
    }

    const { done, failed, total } = this._counts();
    let statusLabel = `${done}/${total}`;
    let statusClass = 'mission-panel--running';
    if (this._completed && failed === 0) {
      statusLabel = `${done}/${total} complete`;
      statusClass = 'mission-panel--complete';
    } else if (this._failed) {
      statusLabel = `${done}/${total} · ${failed} failed`;
      statusClass = 'mission-panel--failed';
    }

    this._mount.className = `mission-panel ${statusClass}${this._collapsed ? ' mission-panel--collapsed' : ''}`;

    const rows = this._order
      .map((id) => this._renderRow(this._promises.get(id)))
      .filter(Boolean)
      .join('');

    this._mount.innerHTML = `
      <div class="mission-panel-header" role="button" tabindex="0" aria-expanded="${!this._collapsed}">
        <div class="mission-panel-header-title">
          <span class="mission-panel-chip">${this._headerIcon()}</span>
          <span class="mission-panel-label">Mission</span>
          <span class="mission-panel-progress">${escapeHtml(statusLabel)}</span>
        </div>
        <button class="mission-panel-toggle" type="button" aria-label="${this._collapsed ? 'Expand mission' : 'Collapse mission'}">
          <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="${this._collapsed ? '6 9 12 15 18 9' : '18 15 12 9 6 15'}"/>
          </svg>
        </button>
      </div>
      <ol class="mission-panel-list" role="list">
        ${rows}
      </ol>
    `;

    const header = this._mount.querySelector('.mission-panel-header');
    header?.addEventListener('click', (ev) => {
      // Clicks on the toggle button should still toggle, but avoid double-fire
      if (ev.target.closest('.mission-panel-toggle')) return;
      this._toggle();
    });
    this._mount.querySelector('.mission-panel-toggle')?.addEventListener('click', (ev) => {
      ev.stopPropagation();
      this._toggle();
    });
  }

  _renderRow(p) {
    if (!p) return '';
    const icon = STATUS_ICON[p.status] || STATUS_ICON.pending;
    const isActive = p.id === this._activeId;
    const klass = [
      'mission-panel-item',
      `mission-panel-item--${p.status || 'pending'}`,
      isActive ? 'mission-panel-item--active' : '',
      p._parentId ? 'mission-panel-item--child' : '',
    ].filter(Boolean).join(' ');

    const attemptBadge = (p.attempts && p.status !== 'fulfilled')
      ? `<span class="mission-panel-attempts" title="Retry attempt">attempt ${p.attempts + 1}/${p.max_attempts || 2}</span>`
      : '';

    const detail = this._detailLine(p);

    return `
      <li class="${klass}" data-promise-id="${escapeHtml(p.id)}">
        ${icon}
        <div class="mission-panel-item-body">
          <div class="mission-panel-item-desc">
            ${escapeHtml(p.description || '(untitled step)')}
            ${attemptBadge}
          </div>
          ${detail ? `<div class="mission-panel-item-detail">${detail}</div>` : ''}
        </div>
      </li>
    `;
  }

  _detailLine(p) {
    // Rate-limit backoff takes visual priority over the running tag
    const rl = this._rateLimitById.get(p.id);
    if (rl && Date.now() < rl.until) {
      const remaining = Math.max(1, Math.ceil((rl.until - Date.now()) / 1000));
      const reason = rl.reason ? escapeHtml(rl.reason) : 'backend busy';
      return (
        `<span class="mission-panel-paused-tag">paused:</span> ` +
        `${reason} — retry ${rl.attempt}/${rl.maxRetries} in ${remaining}s`
      );
    }
    if (p.status === 'fulfilled' && p.evidence) {
      return `<span class="mission-panel-verified">verified:</span> ${escapeHtml(this._truncate(p.evidence, 140))}`;
    }
    if (p.status === 'rejected' && p.evidence) {
      return `<span class="mission-panel-failed-tag">failed:</span> ${escapeHtml(this._truncate(p.evidence, 140))}`;
    }
    if (p.status === 'in_progress') {
      return `<span class="mission-panel-running-tag">running…</span>`;
    }
    return '';
  }

  /**
   * Drive the countdown label on paused rows. Runs at 1 Hz only while
   * at least one rate-limit backoff is active; stops itself once all
   * backoffs have elapsed so the panel is idle.
   */
  _ensureCountdownTick() {
    if (this._countdownTimer) return;
    this._countdownTimer = setInterval(() => {
      const now = Date.now();
      // Prune expired entries
      for (const [id, rl] of this._rateLimitById) {
        if (now >= rl.until) this._rateLimitById.delete(id);
      }
      if (this._rateLimitById.size === 0) {
        this._stopCountdownTick();
        this._render();
        return;
      }
      this._render();
    }, 1000);
  }

  _stopCountdownTick() {
    if (this._countdownTimer) {
      clearInterval(this._countdownTimer);
      this._countdownTimer = null;
    }
  }

  _truncate(text, limit) {
    if (!text) return '';
    const first = String(text).split(/\r?\n/)[0] || '';
    return first.length > limit ? first.slice(0, limit - 1) + '…' : first;
  }

  _toggle() {
    this._collapsed = !this._collapsed;
    this._render();
  }

  _headerIcon() {
    if (this._completed && !this._failed) return '✓';
    if (this._failed) return '!';
    return '●';
  }

  /**
   * Briefly mark a row with a transient state (verifying/retry). The CSS
   * class auto-clears after a short delay; we don't track it in state.
   */
  _flashRow(id, variant) {
    const row = this._mount.querySelector(`[data-promise-id="${CSS.escape(id)}"]`);
    if (!row) return;
    const klass = `mission-panel-item--flash-${variant}`;
    row.classList.add(klass);
    setTimeout(() => row.classList.remove(klass), 1500);
  }
}
