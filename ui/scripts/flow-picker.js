/* ==========================================================================
   Flow Picker — popover card grid for selecting reasoning flows
   ========================================================================== */

import { escapeHtml } from './app.js';

export class FlowPicker {
  /**
   * @param {Object} opts
   * @param {HTMLElement}  opts.anchorEl    — element to position relative to
   * @param {Function}     opts.onSelect    — (flow) => void
   * @param {Function}     opts.onCreate    — () => void
   * @param {Function}     opts.onEdit      — () => void
   * @param {Function}     opts.onDismiss   — () => void  (backdrop/Escape close)
   * @param {string}       opts.accentColor — CSS color variable
   */
  constructor({ anchorEl, onSelect, onCreate, onEdit, onDismiss, accentColor }) {
    this._anchor = anchorEl;
    this._onSelect = onSelect;
    this._onCreate = onCreate;
    this._onEdit = onEdit;
    this._onDismiss = onDismiss;
    this._accent = accentColor || 'var(--mode-analytical)';
    this._el = null;
    this._backdrop = null;
    this._boundKeydown = this._handleKeydown.bind(this);
    this._boundResize = this._handleResize.bind(this);
  }

  show(flows, activeFlowId) {
    this.hide();
    this._flows = flows || [];
    this._activeId = activeFlowId || '';
    this._render();
    document.addEventListener('keydown', this._boundKeydown);
    window.addEventListener('resize', this._boundResize);
  }

  hide() {
    if (this._backdrop) { this._backdrop.remove(); this._backdrop = null; }
    if (this._el) { this._el.remove(); this._el = null; }
    document.removeEventListener('keydown', this._boundKeydown);
    window.removeEventListener('resize', this._boundResize);
  }

  /** Internal dismiss — hides DOM and notifies the parent via onDismiss. */
  _dismiss() {
    this.hide();
    if (this._onDismiss) this._onDismiss();
  }

  _handleKeydown(e) {
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      this._dismiss();
    }
  }

  _handleResize() {
    if (this._el) this._positionAboveAnchor();
  }

  _render() {
    // Backdrop (click-to-dismiss)
    this._backdrop = document.createElement('div');
    this._backdrop.className = 'flow-picker__backdrop';
    this._backdrop.addEventListener('click', () => this._dismiss());
    document.body.appendChild(this._backdrop);

    // Popover
    this._el = document.createElement('div');
    this._el.className = 'flow-picker';
    this._el.setAttribute('role', 'listbox');
    this._el.setAttribute('aria-label', 'Available reasoning flows');
    this._el.style.setProperty('--fp-accent', this._accent);

    const cards = this._flows.map(f => this._renderCard(f)).join('');

    this._el.innerHTML = `
      <div class="flow-picker__header">
        <input class="flow-picker__search" type="text"
               placeholder="Search flows\u2026" autocomplete="off" spellcheck="false"
               aria-label="Filter flows by name">
        <button class="flow-picker__create-btn" aria-label="Create new flow">+ New</button>
      </div>
      <div class="flow-picker__grid">${cards || this._renderEmpty()}</div>
      <div class="flow-picker__footer">
        <button class="flow-picker__edit-link">Edit Flows\u2026</button>
      </div>
    `;

    // Position above anchor
    this._positionAboveAnchor();
    document.body.appendChild(this._el);

    // Bind card clicks
    this._el.querySelectorAll('.flow-picker__card').forEach(card => {
      card.addEventListener('click', () => {
        const flowId = card.dataset.flowId;
        const flow = this._flows.find(f => f.id === flowId);
        if (flow && this._onSelect) this._onSelect(flow);
      });
      // Keyboard: Enter/Space selects
      card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          card.click();
        }
      });
    });

    // Bind header buttons
    this._el.querySelector('.flow-picker__create-btn')
      ?.addEventListener('click', () => { if (this._onCreate) this._onCreate(); });
    this._el.querySelector('.flow-picker__edit-link')
      ?.addEventListener('click', () => { if (this._onEdit) this._onEdit(); });

    // Search filter
    const searchInput = this._el.querySelector('.flow-picker__search');
    if (searchInput) {
      searchInput.addEventListener('input', () => this._filterCards(searchInput.value));
      // Skip autofocus when the picker is in compact / bottom-sheet mode
      // (matches _positionAboveAnchor) so we don't pop a soft keyboard
      // open the moment the sheet slides up.
      if (window.innerWidth >= 900) requestAnimationFrame(() => searchInput.focus());
    }
  }

  _positionAboveAnchor() {
    if (!this._el || !this._anchor) return;
    // Bottom-sheet treatment kicks in below 900px so halfscreen on common
    // desktops (1080p/1440p) gets the same always-reachable slide-up
    // layout as mobile — the anchored popover at this width gets clipped
    // by the viewport edge when the anchor sits right of center.
    const isCompact = window.innerWidth < 900;

    // Clear any previously set inline positioning so CSS can take over
    this._el.style.removeProperty('bottom');
    this._el.style.removeProperty('left');
    this._el.style.removeProperty('width');
    this._el.style.removeProperty('max-width');

    if (isCompact) {
      // Bottom sheet — CSS rules handle everything (position: fixed; bottom: 0; etc.)
      return;
    }

    const rect = this._anchor.getBoundingClientRect();
    // Width: prefer the anchor's width, but clamp to a reasonable range
    // so the picker is never narrower than 420px (cards become unreadable)
    // or wider than 560px (looks adrift).
    const desiredWidth = Math.min(560, Math.max(rect.width, 420));
    // Position: anchor's left edge by default, clamped to keep the picker
    // fully on-screen with an 8px safe-area on both sides. Without this,
    // a right-of-center anchor at halfscreen pushes the picker off the
    // viewport edge.
    const maxLeft = window.innerWidth - desiredWidth - 8;
    const left = Math.max(8, Math.min(rect.left, maxLeft));

    this._el.style.bottom = `${window.innerHeight - rect.top + 8}px`;
    this._el.style.left = `${left}px`;
    this._el.style.width = `${desiredWidth}px`;
  }

  _filterCards(query) {
    const q = (query || '').toLowerCase().trim();
    this._el?.querySelectorAll('.flow-picker__card').forEach(card => {
      const name = (card.dataset.flowName || '').toLowerCase();
      const desc = (card.dataset.flowDesc || '').toLowerCase();
      card.style.display = (!q || name.includes(q) || desc.includes(q)) ? '' : 'none';
    });
  }

  _renderCard(flow) {
    const isActive = flow.id === this._activeId;
    const steps = (flow.steps || []).filter(s => s.enabled !== false);
    const pipeline = this._renderMiniPipeline(steps);
    const caps = this._extractCaps(steps);

    return `
      <div class="flow-picker__card${isActive ? ' flow-picker__card--active' : ''}"
           role="option" tabindex="0"
           aria-selected="${isActive}"
           data-flow-id="${escapeHtml(flow.id)}"
           data-flow-name="${escapeHtml(flow.name || '')}"
           data-flow-desc="${escapeHtml(flow.description || '')}">
        <div class="flow-picker__card-name">
          ${escapeHtml(flow.name || 'Untitled')}
          ${isActive ? '<span class="flow-picker__card-active" aria-label="Currently active">\u25C6</span>' : ''}
        </div>
        <div class="flow-picker__card-desc">${escapeHtml(flow.description || '\u2014')}</div>
        <div class="flow-picker__card-footer">
          <div class="flow-picker__card-pipeline" aria-hidden="true">${pipeline}</div>
          <span class="flow-picker__card-count">${steps.length}</span>
          <div class="flow-picker__card-caps" aria-hidden="true">${caps}</div>
        </div>
      </div>
    `;
  }

  _renderMiniPipeline(steps) {
    if (!steps.length) return '';
    return steps.map((_, i) =>
      (i > 0 ? '<span class="flow-picker__line"></span>' : '') +
      '<span class="flow-picker__dot"></span>'
    ).join('');
  }

  _extractCaps(steps) {
    const allTools = new Set(steps.flatMap(s => s.tool_names || []));
    const allCats = new Set(steps.flatMap(s => s.tool_categories || []));
    const caps = [];
    if (allTools.has('web_search') || allCats.has('web'))
      caps.push('<span title="Web Search">\u26B2</span>');
    if (allTools.has('web_fetch') || allTools.has('document_parse'))
      caps.push('<span title="Document Fetch">\u2637</span>');
    if (steps.some(s => s.role === 'verify'))
      caps.push('<span title="Verification">\u2713</span>');
    if (allTools.has('calculator') || allTools.has('math_verify'))
      caps.push('<span title="Math / Calculate">\u03A3</span>');
    if (allTools.has('python_exec'))
      caps.push('<span title="Code Execution">\u27E8/\u27E9</span>');
    return caps.join(' ');
  }

  _renderEmpty() {
    return `<div class="flow-picker__empty">
      No flows available. Click <strong>+ New</strong> to create one.
    </div>`;
  }
}
