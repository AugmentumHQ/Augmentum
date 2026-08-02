/**
 * companion-becca-day.js — Becca's day timeline.
 *
 * Companion verbs architecture, Phase 5. Renders the management-verb
 * activity for the current user, sourced from companion_verb_log via
 * GET /api/companion/day.
 *
 * Two views:
 *   - Summary cards (per verb): fires / ok / skipped / errors / avg latency
 *   - Timeline list: recent invocations, most-recent first
 *
 * Mount via `BeccaDayPanel.mount(hostEl)` from companion-self.js (a small
 * hook there opens the panel into a side drawer or replaces a section).
 * Standalone usage: import + `panel.mount(document.body)` to drop a
 * fullscreen overlay; close with `panel.detach()`.
 */

const REFRESH_MS = 30_000;
const DEFAULT_WINDOW_HOURS = 24;

function _esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/`/g, '&#96;')
    .replace(/\$\{/g, '&#36;{');
}

function _relTime(unix) {
  if (!Number.isFinite(unix) || unix <= 0) return '';
  const now = Date.now() / 1000;
  const d = Math.max(0, now - unix);
  if (d < 60) return `${Math.floor(d)}s ago`;
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

function _outcomeColor(outcome) {
  if (outcome === 'ok') return '#7ed957';
  if (outcome === 'cooldown_skipped' || outcome === 'deduped') return '#888';
  if (outcome === 'error' || outcome === 'auto_paused') return '#e57373';
  if (outcome === 'autonomy_gated' || outcome === 'chain_depth_exceeded') return '#f0c674';
  if (outcome === 'budget_exceeded') return '#f4a261';
  return '#aaa';
}

function _safetyChipBg(safety) {
  if (safety === 'READ') return 'rgba(126, 217, 87, 0.12)';
  if (safety === 'WRITE_SELF') return 'rgba(96, 153, 230, 0.18)';
  if (safety === 'WRITE_USER') return 'rgba(244, 162, 97, 0.22)';
  return 'rgba(255, 255, 255, 0.08)';
}

export class BeccaDayPanel {
  constructor(opts = {}) {
    this._windowHours = opts.windowHours || DEFAULT_WINDOW_HOURS;
    this._refreshTimer = null;
    this._root = null;
    this._summaryEl = null;
    this._timelineEl = null;
    this._metaEl = null;
    this._busy = false;
  }

  async _fetch() {
    try {
      const resp = await fetch(
        `/api/companion/day?window_hours=${this._windowHours}&limit=200`,
        { credentials: 'same-origin' },
      );
      if (!resp.ok) return null;
      return await resp.json();
    } catch {
      return null;
    }
  }

  async refresh() {
    if (this._busy) return;
    this._busy = true;
    try {
      const data = await this._fetch();
      if (!data || !data.enabled) {
        this._renderEmpty(data);
        return;
      }
      this._renderSummary(data.summary || {});
      this._renderTimeline(data.timeline || []);
      this._renderMeta(data);
    } finally {
      this._busy = false;
    }
  }

  _renderEmpty(data) {
    if (!this._summaryEl) return;
    const message = data && !data.enabled
      ? "Becca's runtime is off."
      : 'No verb activity in this window.';
    this._summaryEl.innerHTML = `<div class="becca-day-empty">${_esc(message)}</div>`;
    this._timelineEl.innerHTML = '';
  }

  _renderSummary(summary) {
    if (!this._summaryEl) return;
    const names = Object.keys(summary).sort();
    if (names.length === 0) {
      this._summaryEl.innerHTML = `<div class="becca-day-empty">No verb activity in this window.</div>`;
      return;
    }
    const cards = names.map((name) => {
      const s = summary[name] || {};
      const fires = Number(s.fires || 0);
      const ok = Number(s.ok || 0);
      const skipped = Number(s.skipped || 0);
      const errors = Number(s.errors || 0);
      const lat = Number(s.avg_latency_ms || 0);
      const last = Number(s.last_fired_at || 0);
      const safety = String(s.safety_class || '');
      const dispatch = String(s.dispatch_class || '');
      const safetyBg = _safetyChipBg(safety);
      return `
        <div class="becca-day-card">
          <div class="becca-day-card-head">
            <span class="becca-day-card-name">${_esc(name)}</span>
            <span class="becca-day-card-chips">
              ${safety ? `<span class="becca-day-chip" style="background:${safetyBg}">${_esc(safety)}</span>` : ''}
              ${dispatch ? `<span class="becca-day-chip">${_esc(dispatch)}</span>` : ''}
            </span>
          </div>
          <div class="becca-day-card-stats">
            <span title="fires">${fires}×</span>
            <span title="ok" style="color:#7ed957">${ok} ok</span>
            ${skipped ? `<span title="skipped" style="color:#888">${skipped} skipped</span>` : ''}
            ${errors ? `<span title="errors" style="color:#e57373">${errors} err</span>` : ''}
            <span class="becca-day-card-lat" title="avg latency">${lat.toFixed(0)}ms avg</span>
          </div>
          <div class="becca-day-card-foot">${_esc(_relTime(last) || 'never')}</div>
        </div>`;
    }).join('');
    this._summaryEl.innerHTML = cards;
  }

  _renderTimeline(rows) {
    if (!this._timelineEl) return;
    if (!rows || rows.length === 0) {
      this._timelineEl.innerHTML = '';
      return;
    }
    const items = rows.slice(0, 80).map((r) => {
      const color = _outcomeColor(r.outcome);
      const cited = (r.cited || []).slice(0, 2).map(
        (c) => `${_esc(c.table || '')}${c.row_id ? `:${_esc(String(c.row_id).slice(0, 12))}` : ''}`,
      ).join(', ');
      return `
        <li class="becca-day-row" title="${_esc(r.event_topic || '')}">
          <span class="becca-day-dot" style="background:${color}"></span>
          <span class="becca-day-verb">${_esc(r.verb)}</span>
          <span class="becca-day-outcome" style="color:${color}">${_esc(r.outcome)}</span>
          <span class="becca-day-lat">${_esc(String(r.latency_ms))}ms</span>
          ${cited ? `<span class="becca-day-cited" title="cited substrate">${_esc(cited)}</span>` : ''}
          <span class="becca-day-when">${_esc(_relTime(r.fired_at))}</span>
        </li>`;
    }).join('');
    this._timelineEl.innerHTML = items;
  }

  _renderMeta(data) {
    if (!this._metaEl) return;
    const verbs = Object.keys(data.summary || {}).length;
    const events = (data.timeline || []).length;
    this._metaEl.textContent = `${verbs} verbs · ${events} events · last ${data.window_hours}h`;
  }

  mount(host) {
    if (this._root) return this._root;
    const parent = host || document.body;
    const root = document.createElement('div');
    root.className = 'becca-day-panel';
    root.innerHTML = `
      <div class="becca-day-head">
        <div>
          <div class="becca-day-title">Becca's day</div>
          <div class="becca-day-meta">—</div>
        </div>
        <button class="becca-day-refresh" type="button">Refresh</button>
      </div>
      <div>
        <div class="becca-day-section-title">Per verb</div>
        <div class="becca-day-summary"></div>
      </div>
      <div>
        <div class="becca-day-section-title">Timeline</div>
        <ul class="becca-day-timeline"></ul>
      </div>`;
    parent.appendChild(root);
    this._root = root;
    this._metaEl = root.querySelector('.becca-day-meta');
    this._summaryEl = root.querySelector('.becca-day-summary');
    this._timelineEl = root.querySelector('.becca-day-timeline');
    root.querySelector('.becca-day-refresh').addEventListener(
      'click', () => this.refresh(),
    );

    this.refresh();
    this._refreshTimer = setInterval(() => this.refresh(), REFRESH_MS);
    return root;
  }

  detach() {
    if (this._refreshTimer) {
      clearInterval(this._refreshTimer);
      this._refreshTimer = null;
    }
    if (this._root && this._root.parentNode) {
      this._root.parentNode.removeChild(this._root);
    }
    this._root = null;
  }
}

// Convenience: window-level mount helper so the panel can be opened from
// the browser console or wired into a settings link without importing.
if (typeof window !== 'undefined') {
  window.openBeccaDay = function openBeccaDay(host) {
    const panel = new BeccaDayPanel();
    panel.mount(host || document.body);
    return panel;
  };
}
