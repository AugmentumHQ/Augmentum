/* ==========================================================================
   Augmentum — Reusable Flow Editor
   A class-based flow editor that can be instantiated for both analytical
   and agentic panels with different configs.
   ========================================================================== */

import { escapeHtml, showToast } from './app.js';
import { openFlowCreateSheet } from './flow-create-sheet.js';

// ---------------------------------------------------------------------------
// Template variable hints per mode
// ---------------------------------------------------------------------------

const TEMPLATE_VARS = {
  analytical: '{query} {previous_output} {step:Name} {all_outputs} {complexity} {search_results} {model} {tools} {conversation} {current_date}',
  agentic: '{query} {previous_output} {step:Name} {all_outputs} {plan} {search_results} {model} {tools} {conversation} {current_date}',
};

// ---------------------------------------------------------------------------
// Role icon map — inline Lucide-style SVGs (currentColor stroke, 24px
// viewBox). Sized by .fe-role-card__icon svg in CSS; inheriting
// currentColor lets hover/selected states tint the glyph via the card.
// ---------------------------------------------------------------------------

const _SVG_ATTRS = 'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"';

const _ROLE_ICONS = {
  classify:  `<svg ${_SVG_ATTRS}><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>`,
  search:    `<svg ${_SVG_ATTRS}><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`,
  analyze:   `<svg ${_SVG_ATTRS}><path d="M9.5 2a3 3 0 0 0-3 3v.5a3 3 0 0 0-2 2.83V11a3 3 0 0 0 2 2.83V15a3 3 0 0 0 3 3 2 2 0 0 0 2-2V4a2 2 0 0 0-2-2z"/><path d="M14.5 2a3 3 0 0 1 3 3v.5a3 3 0 0 1 2 2.83V11a3 3 0 0 1-2 2.83V15a3 3 0 0 1-3 3 2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"/></svg>`,
  verify:    `<svg ${_SVG_ATTRS}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>`,
  respond:   `<svg ${_SVG_ATTRS}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
  plan:      `<svg ${_SVG_ATTRS}><rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4v2h6V4"/><line x1="9" y1="10" x2="15" y2="10"/><line x1="9" y1="14" x2="15" y2="14"/><line x1="9" y1="18" x2="13" y2="18"/></svg>`,
  draft:     `<svg ${_SVG_ATTRS}><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4z"/></svg>`,
  create:    `<svg ${_SVG_ATTRS}><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/></svg>`,
  review:    `<svg ${_SVG_ATTRS}><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/></svg>`,
  deliver:   `<svg ${_SVG_ATTRS}><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>`,
  transform: `<svg ${_SVG_ATTRS}><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>`,
};

const _ROLE_ICON_FALLBACK = `<svg ${_SVG_ATTRS}><path d="M12 2 L22 12 L12 22 L2 12 Z"/></svg>`;

// How many roles to show before the "More" toggle
const _PRIMARY_ROLE_COUNT = 5;

// Plain-English labels + one-line hints per backend ToolCategory value.
// Keep the keys aligned with augmentum/tools/base.py::ToolCategory. Adding
// a new category there without an entry here falls back to the raw value,
// which is recoverable but not pretty.
const _CATEGORY_META = {
  search:   { label: 'Web Search',  hint: 'Search the internet' },
  fetch:    { label: 'Fetch & Read', hint: 'Pull URLs, pages, and documents' },
  execute:  { label: 'Run Code',    hint: 'Execute Python or JavaScript' },
  verify:   { label: 'Verify',      hint: 'Cross-check facts and math' },
  file:     { label: 'Files',       hint: 'Read, write, and manage files' },
  image:    { label: 'Images',      hint: 'Generate and edit images' },
  artifact: { label: 'Artifacts',   hint: 'Build documents, slides, and charts' },
  code:     { label: 'Code',        hint: 'Write, lint, and refactor code' },
  shell:    { label: 'Terminal',    hint: 'Run shell commands' },
};

// ---------------------------------------------------------------------------
// FlowEditor Class
// ---------------------------------------------------------------------------

export class FlowEditor {
  /**
   * @param {object} config
   * @param {HTMLElement} config.containerEl - Root element to render into
   * @param {HTMLElement} [config.mainContainerEl] - Optional separate column to host
   *   the step/settings editor forms. When provided, the editor renders in
   *   true two-column layout: list/info live in containerEl (sidebar) and the
   *   editor panels move to mainContainerEl. The step list stays visible
   *   while editing — opening an editor no longer hides it. Used by the
   *   fullscreen flow editor overlay.
   * @param {'analytical'|'agentic'} config.mode - Editor mode
   * @param {Array<{value: string, label: string}>} config.roles - Available roles
   * @param {string} [config.accentColor] - CSS color variable
   * @param {function} [config.onFlowChanged] - Callback when a flow is modified
   */
  constructor({ containerEl, mainContainerEl, mode, roles, accentColor, onFlowChanged }) {
    this.el = containerEl;
    this.mainEl = mainContainerEl || null;
    this.mode = mode;
    this.roles = roles;
    this.accentColor = accentColor || 'var(--mode-analytical)';
    this.onFlowChanged = onFlowChanged || (() => {});
    this.flows = [];
    this.selectedFlowId = '';
    this.selectedStepIndex = -1;
    this.currentFlow = null;
    this.availableTools = [];  // fetched from backend
    this.toolCategories = [];  // fetched from backend
    this._toolsFetched = false;
    this._dragSrcIdx = -1;
    this._dropTargetIdx = -1;
    // Unsaved-changes guard for the step editor. Flipped true by any real
    // user edit to a step form field, cleared on (re)open and on save.
    // Guards the back button and a page reload so a typed-out prompt isn't
    // silently lost. (Covers form inputs — name/prompts/templates/caps/
    // toggles; role-card / tool-chip clicks are quick to redo and left
    // unguarded to avoid over-prompting.)
    this._stepDirty = false;
    this._unloadGuardBound = false;
  }

  // -------------------------------------------------------------------------
  // Scoped DOM helpers
  // -------------------------------------------------------------------------

  $(id) {
    // Two-column mode: the step/settings editors get relocated to mainEl
    // after render, so resolve against both containers.
    return this.el.querySelector(`[data-fe="${id}"]`)
      || (this.mainEl ? this.mainEl.querySelector(`[data-fe="${id}"]`) : null);
  }

  $$(sel) {
    if (!this.mainEl) return this.el.querySelectorAll(sel);
    // Manually merge the two NodeLists when in two-column mode so callers
    // see both regions transparently.
    const a = Array.from(this.el.querySelectorAll(sel));
    const b = Array.from(this.mainEl.querySelectorAll(sel));
    return a.concat(b);
  }

  /** True if running in the fullscreen overlay's two-column layout. */
  get _isTwoColumn() {
    return !!this.mainEl;
  }

  // -------------------------------------------------------------------------
  // render() — Generate full editor HTML inside containerEl
  // -------------------------------------------------------------------------

  render() {
    const accent = this.accentColor;
    const modeLabel = this.mode === 'agentic' ? 'Creator' : 'Thinker';
    const templateHint = TEMPLATE_VARS[this.mode] || TEMPLATE_VARS.analytical;

    this.el.innerHTML = `
      <div class="flow-editor" style="--fe-accent: ${accent}">
        <!-- Flow rail — horizontal scrollable list of flows + New chip.
             Replaces the old <select> dropdown so all flows are discoverable
             at a glance. The active flow auto-loads on init. -->
        <div class="flow-editor-rail-wrap">
          <div data-fe="flow-rail" class="flow-editor-rail" role="tablist" aria-label="Available flows">
            <div class="flow-editor-rail__skeleton" data-fe="flow-rail-skeleton">
              <span></span><span></span><span></span>
            </div>
          </div>
          <button data-fe="flow-new-btn" class="flow-editor-rail__new" title="Create a new flow" aria-label="New flow">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
          </button>
          <!-- Hidden legacy <select> retained so external code that targets
               data-fe="flow-select" (and the rest of this class) keeps
               working without churn. -->
          <select data-fe="flow-select" class="hidden" aria-hidden="true">
            <option value="">Select a flow...</option>
          </select>
        </div>

        <!-- Flow info section (hidden until a flow is selected) -->
        <div data-fe="flow-info-section" class="flow-editor-info" style="display:none">
          <div class="flow-editor-info-header">
            <div class="flow-editor-info-title">
              <span data-fe="flow-info-name" class="flow-editor-info-name"></span>
              <span data-fe="flow-badge-default" class="flow-editor-badge flow-editor-badge-default hidden">Default</span>
              <span data-fe="flow-badge-builtin" class="flow-editor-badge flow-editor-badge-builtin hidden">Built-in</span>
            </div>
            <div data-fe="flow-info-desc" class="flow-editor-info-desc"></div>
          </div>
          <div class="flow-editor-info-actions">
            ${this.mode === 'analytical' ? `
            <button data-fe="flow-test-btn" class="flow-editor-btn" title="Test the flow with a sample query">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12" aria-hidden="true">
                <polygon points="5 3 19 12 5 21 5 3"/>
              </svg>
              Test run
            </button>` : ''}
            <button data-fe="flow-edit-btn" class="flow-editor-btn flow-editor-btn-primary" title="Edit flow settings">Edit</button>
            <div class="flow-editor-info-overflow" data-fe="flow-overflow">
              <button data-fe="flow-overflow-btn" class="flow-editor-btn-icon" title="More actions" aria-haspopup="true" aria-expanded="false" aria-label="More actions">
                <svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16" aria-hidden="true"><circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/></svg>
              </button>
              <div data-fe="flow-overflow-menu" class="flow-editor-info-menu" role="menu" hidden>
                <button data-fe="flow-clone-btn" class="flow-editor-btn" role="menuitem">Clone</button>
                <button data-fe="flow-set-default-btn" class="flow-editor-btn" role="menuitem">Set as default</button>
                <button data-fe="flow-export-btn" class="flow-editor-btn" role="menuitem">Export</button>
                <div class="flow-editor-info-menu-sep"></div>
                <button data-fe="flow-delete-btn" class="flow-editor-btn flow-editor-btn-danger" role="menuitem">Delete</button>
              </div>
            </div>
          </div>
          ${this.mode === 'analytical' ? `
          <!-- Test-run drawer — analytical mode only. The agentic Build
               pipeline runs through a different execution path
               (TaskState + autonomy gates + artifact creation) that the
               shared executor doesn't cover, so the drawer is hidden in
               that mode rather than silently producing wrong results. -->
          <div data-fe="flow-test-drawer" class="flow-editor-test-drawer" hidden>
            <div class="flow-editor-test-composer">
              <textarea data-fe="flow-test-query"
                        class="flow-editor-input flow-editor-test-query"
                        rows="2"
                        placeholder="Sample query — what would the user ask?"></textarea>
              <div class="flow-editor-test-controls">
                <label class="flow-editor-test-control">
                  <span class="flow-editor-test-control__label">Complexity</span>
                  <select data-fe="flow-test-complexity" class="flow-editor-input flow-editor-input-sm">
                    <option value="">Auto</option>
                    <option value="simple">Simple</option>
                    <option value="moderate">Moderate</option>
                    <option value="complex">Complex</option>
                  </select>
                </label>
                <label class="flow-editor-checkbox-label flow-editor-test-control" title="When off, tool calls are skipped so the dry-run is cheap and side-effect-free.">
                  <input data-fe="flow-test-tools" type="checkbox">
                  <span>Run tools</span>
                </label>
                <span style="flex:1"></span>
                <button data-fe="flow-test-cancel" class="flow-editor-btn flow-editor-btn-sm" hidden>Cancel</button>
                <button data-fe="flow-test-run" class="flow-editor-btn flow-editor-btn-primary flow-editor-btn-sm">Run</button>
              </div>
            </div>
            <div data-fe="flow-test-output" class="flow-editor-test-output" hidden></div>
          </div>` : ''}
        </div>

        <!-- Step list section -->
        <div data-fe="flow-steps-container" class="flow-editor-steps-container" style="display:none">
          <div class="flow-editor-steps-header">
            <span class="flow-editor-steps-title">Steps</span>
            <button data-fe="flow-add-step-btn" class="flow-editor-btn flow-editor-btn-sm" title="Add step">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12">
                <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
              </svg>
              Add Step
            </button>
          </div>
          <div data-fe="flow-step-list" class="flow-editor-step-list"></div>
          <div data-fe="drag-indicator" class="flow-editor-drag-indicator"></div>
        </div>

        <!-- Flow settings editor (hidden until Edit is clicked) -->
        <div data-fe="flow-settings-editor" class="flow-editor-step-editor hidden">
          <div class="flow-editor-step-editor-header">
            <button data-fe="settings-editor-back" class="flow-editor-btn flow-editor-btn-sm" title="Back to steps">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                <polyline points="15 18 9 12 15 6"/>
              </svg>
              Back
            </button>
            <span class="flow-editor-step-editor-title">Flow Settings</span>
          </div>

          <div class="flow-editor-form">
            <label class="flow-editor-label">
              Name
              <input data-fe="settings-name" type="text" class="flow-editor-input" placeholder="Flow name">
            </label>

            <label class="flow-editor-label">
              Description
              <textarea data-fe="settings-desc" class="flow-editor-textarea" rows="2" placeholder="What does this flow do?"></textarea>
            </label>

            <label class="flow-editor-label">
              Trigger Keywords
              <input data-fe="settings-keywords" type="text" class="flow-editor-input" placeholder="debug, error, fix (comma-separated)">
              <span class="flow-editor-hint">Auto Routing uses these to match queries to this flow.</span>
              <span data-fe="settings-keyword-collision" class="flow-editor-hint flow-editor-hint-warn hidden"></span>
            </label>

            <div class="flow-editor-label">
              Routing Preview
              <div class="flow-editor-row">
                <input data-fe="settings-routing-query" type="text" class="flow-editor-input"
                       placeholder="Type a query to see which flow would handle it">
                <button data-fe="settings-routing-preview-btn" type="button" class="flow-editor-btn flow-editor-btn-sm">Preview</button>
              </div>
              <div data-fe="settings-routing-result" class="flow-editor-routing-result"></div>
            </div>

            <div class="flow-editor-row">
              <label class="flow-editor-label">
                Max Tool Calls / Step
                <input data-fe="settings-max-tools" type="number" class="flow-editor-input flow-editor-input-sm" value="3" min="0" max="20">
              </label>
              <label class="flow-editor-checkbox-label flow-editor-checkbox-standalone">
                <input data-fe="settings-auto-search" type="checkbox">
                Auto Search
              </label>
              <label class="flow-editor-checkbox-label flow-editor-checkbox-standalone">
                <input data-fe="settings-auto-select" type="checkbox">
                Auto Select
              </label>
            </div>

            <div class="flow-editor-form-actions">
              <button data-fe="settings-save-btn" class="flow-editor-btn flow-editor-btn-primary">Save Settings</button>
            </div>
          </div>
        </div>

        <!-- Step editor form (hidden until a step is opened) -->
        <div data-fe="flow-step-editor" class="flow-editor-step-editor hidden">
          <div class="flow-editor-step-editor-header">
            <button data-fe="step-editor-back" class="flow-editor-btn flow-editor-btn-sm" title="Back to step list">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                <polyline points="15 18 9 12 15 6"/>
              </svg>
              Back
            </button>
            <span data-fe="step-editor-title" class="flow-editor-step-editor-title">Edit Step</span>
          </div>

          <div class="flow-editor-form">
            <label class="flow-editor-label">
              Name
              <input data-fe="step-name-input" type="text" class="flow-editor-input" placeholder="Step name">
            </label>

            <div class="flow-editor-label">
              What should this step do?
              <div data-fe="step-role-cards" class="fe-role-grid" role="radiogroup" aria-label="Step role">
                ${this.roles.slice(0, _PRIMARY_ROLE_COUNT).map(r => `<div class="fe-role-card" data-role="${r.value}" tabindex="0" role="radio" aria-checked="false" title="${escapeHtml(r.description || '')}">
                  <span class="fe-role-card__icon">${_ROLE_ICONS[r.value] || _ROLE_ICON_FALLBACK}</span>
                  <span class="fe-role-card__name">${escapeHtml(r.label)}</span>
                  ${r.description ? `<span class="fe-role-card__desc">${escapeHtml(r.description)}</span>` : ''}
                </div>`).join('')}
              </div>
              ${this.roles.length > _PRIMARY_ROLE_COUNT ? `
              <button data-fe="role-more-toggle" class="fe-role-more-toggle">More roles\u2026</button>
              <div data-fe="role-more-grid" class="fe-role-grid" role="radiogroup" style="display:none">
                ${this.roles.slice(_PRIMARY_ROLE_COUNT).map(r => `<div class="fe-role-card" data-role="${r.value}" tabindex="0" role="radio" aria-checked="false" title="${escapeHtml(r.description || '')}">
                  <span class="fe-role-card__icon">${_ROLE_ICONS[r.value] || _ROLE_ICON_FALLBACK}</span>
                  <span class="fe-role-card__name">${escapeHtml(r.label)}</span>
                  ${r.description ? `<span class="fe-role-card__desc">${escapeHtml(r.description)}</span>` : ''}
                </div>`).join('')}
              </div>` : ''}
              <select data-fe="step-role-select" class="hidden">
                ${this.roles.map(r => `<option value="${escapeHtml(r.value)}">${escapeHtml(r.label)}</option>`).join('')}
              </select>
            </div>

            <div class="flow-editor-label">
              Tools
              <div data-fe="step-tools-grid" class="flow-editor-tools-grid"></div>
            </div>

            <button data-fe="advanced-toggle" class="fe-advanced-toggle" type="button">
              <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 4 10 8 6 12"/></svg>
              Advanced Options
            </button>
            <div data-fe="advanced-section" class="fe-advanced-section">
              <label class="flow-editor-label">
                System Prompt
                <textarea data-fe="step-system-prompt" class="flow-editor-textarea" rows="4" placeholder="System prompt for this step..."></textarea>
              </label>

              <label class="flow-editor-label">
                User Template
                <textarea data-fe="step-user-template" class="flow-editor-textarea" rows="4" placeholder="User template..."></textarea>
                <span class="flow-editor-hint">Variables: ${escapeHtml(templateHint)}</span>
              </label>

              <div class="flow-editor-row">
                <label class="flow-editor-label flow-editor-label-inline">
                  Complexity Gate
                  <div class="flow-editor-gate-group">
                    <label class="flow-editor-checkbox-label">
                      <input data-fe="step-gate-simple" type="checkbox"> Simple
                    </label>
                    <label class="flow-editor-checkbox-label">
                      <input data-fe="step-gate-moderate" type="checkbox"> Moderate
                    </label>
                    <label class="flow-editor-checkbox-label">
                      <input data-fe="step-gate-complex" type="checkbox"> Complex
                    </label>
                  </div>
                  <span class="flow-editor-hint">None checked = always runs. Check to restrict when this step executes.</span>
                </label>
              </div>

              <label class="flow-editor-label">
                Model Override
                <input data-fe="step-model-override" type="text" class="flow-editor-input" placeholder="e.g. qwen2.5:7b, gpt-4o, or &quot;verify&quot;">
                <span class="flow-editor-hint">Leave empty to use the flow's default model. Use "verify" for cross-model verification.</span>
              </label>

              <div class="flow-editor-row">
                <label class="flow-editor-label">
                  Output Cap (tokens)
                  <input data-fe="step-output-cap" type="number" class="flow-editor-input flow-editor-input-sm" value="800" min="0" max="16000">
                  <span class="flow-editor-hint">0 = unlimited (use for final response steps)</span>
                </label>
                <label class="flow-editor-checkbox-label flow-editor-checkbox-standalone" title="When checked, this step's output is shown directly in the chat. At least one step should have this enabled.">
                  <input data-fe="step-stream-to-user" type="checkbox">
                  Stream to user
                </label>
              </div>
            </div>

            <div class="flow-editor-form-actions">
              <button data-fe="step-save-btn" class="flow-editor-btn flow-editor-btn-primary">Save Step</button>
              <button data-fe="step-delete-btn" class="flow-editor-btn flow-editor-btn-danger">Delete Step</button>
            </div>
          </div>
        </div>

        <!-- Import bar -->
        <div class="flow-editor-import-bar">
          <button data-fe="flow-import-btn" class="flow-editor-btn flow-editor-btn-sm">Import Flow</button>
          <input data-fe="flow-import-file" type="file" accept=".json" class="hidden">
        </div>
      </div>
    `;
  }

  // -------------------------------------------------------------------------
  // init() — Bind all event listeners and load flows
  // -------------------------------------------------------------------------

  init() {
    this.render();
    this._relocateForTwoColumn();
    this._bindEvents();
    this._fetchTools();
    this._refreshMainPanel();
    this.loadFlows();

    // Sync when flow bar selects a flow — fully load it so the editor
    // body (info + step list) mirrors the active flow instead of only
    // updating the chip-rail visual selection.
    document.addEventListener('augmentum:flow-bar-selected', (e) => {
      const { flowId, flow } = e.detail || {};
      if (!flowId || flowId === this.selectedFlowId) return;
      // Same-mode check: don't react to the *other* editor's selection.
      if (flow) {
        const hasAgentic = (flow.trigger_domains || []).includes('agentic');
        if ((this.mode === 'agentic') !== hasAgentic) return;
      }
      this.selectFlow(flowId);
    });
  }

  /**
   * Two-column mode: physically move the step/settings editor panels out
   * of the sidebar container and into mainEl. Done once at init time,
   * after render(). The editor's `data-fe` lookups still work because
   * `$()` searches both containers.
   */
  _relocateForTwoColumn() {
    if (!this._isTwoColumn) return;
    const stepEditor = this.el.querySelector('[data-fe="flow-step-editor"]');
    const settingsEditor = this.el.querySelector('[data-fe="flow-settings-editor"]');
    if (stepEditor) this.mainEl.appendChild(stepEditor);
    if (settingsEditor) this.mainEl.appendChild(settingsEditor);
  }

  /**
   * Two-column mode: toggle the empty-state placeholder vs. an active
   * editor panel based on which one is currently `.hidden`. CSS uses the
   * `fe-overlay-main--editing` class on mainEl to hide the empty state.
   * Single-column mode: no-op.
   */
  _refreshMainPanel() {
    if (!this._isTwoColumn) return;
    const stepEditor = this.$('flow-step-editor');
    const settingsEditor = this.$('flow-settings-editor');
    const stepVisible = stepEditor && !stepEditor.classList.contains('hidden');
    const settingsVisible = settingsEditor && !settingsEditor.classList.contains('hidden');
    this.mainEl.classList.toggle('fe-overlay-main--editing',
                                  !!(stepVisible || settingsVisible));
  }

  async _fetchTools() {
    try {
      // surface=flow → only tools declared useful inside flow steps
      // (SurfaceExposure.flow). Keeps ~80 conversational action verbs
      // (note.create, media.play, device.*) out of the grid.
      const resp = await fetch('/api/tools?surface=flow');
      if (!resp.ok) {
        this._toolsFetchFailed = true;
        return;
      }
      const data = await resp.json();
      this.availableTools = data.tools || [];
      this.toolCategories = new Set(data.categories || []);
      this._toolsFetched = true;
      this._toolsFetchFailed = false;
    } catch {
      this._toolsFetchFailed = true;
    }
  }

  _bindEvents() {
    // Flow selector
    const flowSelect = this.$('flow-select');
    if (flowSelect) {
      flowSelect.addEventListener('change', (e) => this.selectFlow(e.target.value));
    }

    // Flow action buttons
    const newBtn = this.$('flow-new-btn');
    if (newBtn) newBtn.addEventListener('click', () => this.openCreateSheet());

    const editBtn = this.$('flow-edit-btn');
    if (editBtn) editBtn.addEventListener('click', () => this.openFlowSettings());

    // Test-run drawer toggle + run/cancel
    const testBtn = this.$('flow-test-btn');
    if (testBtn) testBtn.addEventListener('click', () => this._toggleTestDrawer());
    const testRunBtn = this.$('flow-test-run');
    if (testRunBtn) testRunBtn.addEventListener('click', () => this._startTestRun());
    const testCancelBtn = this.$('flow-test-cancel');
    if (testCancelBtn) testCancelBtn.addEventListener('click', () => this._cancelTestRun());
    const testQuery = this.$('flow-test-query');
    if (testQuery) {
      testQuery.addEventListener('keydown', (e) => {
        // Cmd/Ctrl-Enter sends; bare Enter inserts newline (keeps Markdown
        // pasting + multi-line prompts ergonomic).
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
          e.preventDefault();
          this._startTestRun();
        }
      });
    }

    const cloneBtn = this.$('flow-clone-btn');
    if (cloneBtn) cloneBtn.addEventListener('click', () => this.cloneFlow());

    const defaultBtn = this.$('flow-set-default-btn');
    if (defaultBtn) defaultBtn.addEventListener('click', () => this.setDefaultFlow());

    const exportBtn = this.$('flow-export-btn');
    if (exportBtn) exportBtn.addEventListener('click', () => this.exportFlow());

    const deleteBtn = this.$('flow-delete-btn');
    if (deleteBtn) deleteBtn.addEventListener('click', () => this.deleteFlow());

    // Overflow menu (Clone / Set Default / Export / Delete live behind a `⋯`
    // trigger so the info-actions row stays as a clean primary CTA + menu).
    const overflowBtn = this.$('flow-overflow-btn');
    const overflowMenu = this.$('flow-overflow-menu');
    const overflowWrap = this.$('flow-overflow');
    if (overflowBtn && overflowMenu && overflowWrap) {
      // The menu is `position: fixed` (set in CSS) so it escapes the
      // `.panel-content` scroll container — otherwise the inspector's
      // `overflow-x: hidden` / `overflow-y: auto` would clip the dropdown
      // against the panel edge. Position is recomputed every open from
      // the trigger's getBoundingClientRect(); scroll/resize close it.
      const positionMenu = () => {
        const rect = overflowBtn.getBoundingClientRect();
        // Pin top to just below the trigger, then anchor the menu's
        // right edge to the trigger's right edge. min-width 160px so
        // we use that as a lower bound for the left-edge calc; the menu
        // is allowed to grow rightward only if it doesn't fit leftward.
        const menuWidth = Math.max(overflowMenu.offsetWidth || 160, 160);
        const gutter = 8;
        let left = rect.right - menuWidth;
        if (left < gutter) left = Math.min(rect.left, window.innerWidth - menuWidth - gutter);
        if (left < gutter) left = gutter;
        let top = rect.bottom + 4;
        // Flip up if there isn't room below.
        if (top + (overflowMenu.offsetHeight || 200) > window.innerHeight - gutter) {
          top = Math.max(gutter, rect.top - (overflowMenu.offsetHeight || 200) - 4);
        }
        overflowMenu.style.top = `${Math.round(top)}px`;
        overflowMenu.style.left = `${Math.round(left)}px`;
        overflowMenu.style.right = 'auto';
      };
      const close = () => {
        overflowWrap.classList.remove('is-open');
        overflowMenu.setAttribute('hidden', '');
        overflowBtn.setAttribute('aria-expanded', 'false');
        window.removeEventListener('scroll', close, true);
        window.removeEventListener('resize', close);
      };
      const open = () => {
        overflowWrap.classList.add('is-open');
        overflowMenu.removeAttribute('hidden');
        overflowBtn.setAttribute('aria-expanded', 'true');
        positionMenu();
        // Close on any scroll in any ancestor (capture phase) or window resize.
        // Simpler + more predictable than tracking the trigger's rect every frame.
        window.addEventListener('scroll', close, true);
        window.addEventListener('resize', close);
      };
      overflowBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (overflowWrap.classList.contains('is-open')) close(); else open();
      });
      // Close on any menu-item click (action fires from its own listener first).
      overflowMenu.addEventListener('click', (e) => {
        if (e.target.closest('button')) close();
      });
      // Click outside or Escape closes the menu. The menu itself lives in the
      // body's coordinate space (position:fixed) but stays a DOM child of
      // overflowWrap, so `overflowWrap.contains(e.target)` still works.
      document.addEventListener('click', (e) => {
        if (!overflowWrap.contains(e.target) && !overflowMenu.contains(e.target)) close();
      });
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && overflowWrap.classList.contains('is-open')) close();
      });
    }

    // Flow settings editor
    const settingsBackBtn = this.$('settings-editor-back');
    if (settingsBackBtn) settingsBackBtn.addEventListener('click', () => this.closeFlowSettings());

    const settingsSaveBtn = this.$('settings-save-btn');
    if (settingsSaveBtn) settingsSaveBtn.addEventListener('click', () => this.saveFlowSettings());

    // Routing preview + live keyword-collision hint
    const routingBtn = this.$('settings-routing-preview-btn');
    if (routingBtn) routingBtn.addEventListener('click', () => this._runRoutingPreview());
    const routingQuery = this.$('settings-routing-query');
    if (routingQuery) {
      routingQuery.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); this._runRoutingPreview(); }
      });
    }
    const keywordsField = this.$('settings-keywords');
    if (keywordsField) {
      keywordsField.addEventListener('input', () => this._checkKeywordCollisions());
    }

    // Step list
    const addStepBtn = this.$('flow-add-step-btn');
    if (addStepBtn) addStepBtn.addEventListener('click', () => this.addStep());

    // Step editor
    const backBtn = this.$('step-editor-back');
    if (backBtn) backBtn.addEventListener('click', () => this.closeStepEditor());

    const saveBtn = this.$('step-save-btn');
    if (saveBtn) saveBtn.addEventListener('click', () => this.saveStep());

    const stepDeleteBtn = this.$('step-delete-btn');
    if (stepDeleteBtn) stepDeleteBtn.addEventListener('click', () => this.deleteStep());

    // Mark the step editor dirty on any real form edit. Programmatic
    // .value/.checked assignments in openStepEditor do NOT fire these
    // events, so this only flips on genuine user input — no false positives.
    const stepEditor = this.$('flow-step-editor');
    if (stepEditor) {
      const markDirty = () => { this._stepDirty = true; };
      stepEditor.addEventListener('input', markDirty);
      stepEditor.addEventListener('change', markDirty);
    }

    // Page-reload / tab-close net: warn only while the step editor is open
    // AND has unsaved edits. Bound once (re-render calls _bindEvents again).
    if (!this._unloadGuardBound) {
      this._unloadGuardBound = true;
      window.addEventListener('beforeunload', (e) => {
        const editor = this.$('flow-step-editor');
        const open = editor && !editor.classList.contains('hidden');
        if (open && this._stepDirty) {
          e.preventDefault();
          e.returnValue = '';
        }
      });
    }

    // Role cards — click to select role.
    // Use $$ (which spans containerEl + mainContainerEl) since the cards
    // live inside the step editor, which gets relocated to the main column
    // in the fullscreen overlay's two-column layout.
    this.$$('.fe-role-card').forEach(card => {
      const handler = () => {
        const role = card.dataset.role;
        if (!role) return;
        // Update hidden select
        const roleSelect = this.$('step-role-select');
        if (roleSelect) roleSelect.value = role;
        // Update visual + ARIA state
        this.$$('.fe-role-card').forEach(c => {
          c.classList.remove('fe-role-card--selected');
          c.setAttribute('aria-checked', 'false');
        });
        card.classList.add('fe-role-card--selected');
        card.setAttribute('aria-checked', 'true');
      };
      card.addEventListener('click', handler);
      card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handler(); }
      });
    });

    // "More roles" toggle
    const roleMoreToggle = this.$('role-more-toggle');
    const roleMoreGrid = this.$('role-more-grid');
    if (roleMoreToggle && roleMoreGrid) {
      roleMoreToggle.addEventListener('click', () => {
        const isHidden = roleMoreGrid.style.display === 'none';
        roleMoreGrid.style.display = isHidden ? '' : 'none';
        roleMoreToggle.textContent = isHidden ? 'Fewer roles' : 'More roles\u2026';
      });
    }

    // Advanced toggle
    const advancedToggle = this.$('advanced-toggle');
    if (advancedToggle) {
      advancedToggle.addEventListener('click', () => {
        advancedToggle.classList.toggle('open');
      });
    }

    // Import
    const importBtn = this.$('flow-import-btn');
    if (importBtn) importBtn.addEventListener('click', () => this.importFlow());

    const importFile = this.$('flow-import-file');
    if (importFile) importFile.addEventListener('change', (e) => this._handleImportFile(e));
  }

  // -------------------------------------------------------------------------
  // loadFlows() — Fetch flows from backend, filter by mode
  // -------------------------------------------------------------------------

  async loadFlows() {
    try {
      const resp = await fetch('/api/reasoning/flows');
      if (!resp.ok) return;
      const data = await resp.json();
      this.flows = this._filterFlowsByMode(data);
      this.renderFlowSelector();

      // Auto-load the active flow on first open so the editor opens
      // "where the user left off" instead of an empty placeholder.
      // Preference order: 1) the flow already selected by an external
      // event (flow-bar), 2) the default flow, 3) the first flow.
      if (!this.selectedFlowId && this.flows.length) {
        const pick = this.flows.find(f => f.is_default) || this.flows[0];
        if (pick && pick.id) {
          await this.selectFlow(pick.id);
        }
      } else if (this.selectedFlowId && !this.currentFlow) {
        // External event already picked an id but we haven't fetched
        // the full flow yet (race). Fetch + render now.
        await this.selectFlow(this.selectedFlowId);
      }
    } catch {
      // backend not reachable
    }
  }

  /**
   * Filter flows based on mode.
   * - analytical: flows where trigger_domains does NOT include "agentic"
   * - agentic: flows where trigger_domains DOES include "agentic"
   */
  _filterFlowsByMode(allFlows) {
    if (!Array.isArray(allFlows)) return [];
    return allFlows.filter(f => {
      const domains = f.trigger_domains || [];
      const hasAgentic = domains.includes('agentic');
      if (this.mode === 'agentic') {
        return hasAgentic;
      }
      return !hasAgentic;
    });
  }

  // -------------------------------------------------------------------------
  // renderFlowSelector() — Update the dropdown with filtered flows
  // -------------------------------------------------------------------------

  renderFlowSelector() {
    // Keep the hidden <select> in sync for accessibility + any external
    // code that still reads its value.
    const select = this.$('flow-select');
    if (select) {
      select.innerHTML = '<option value="">Select a flow...</option>';
      this.flows.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f.id;
        opt.textContent = f.name + (f.is_default ? ' (default)' : '');
        if (f.id === this.selectedFlowId) opt.selected = true;
        select.appendChild(opt);
      });
    }

    // Render the visible chip rail.
    const rail = this.$('flow-rail');
    if (!rail) return;

    if (!this.flows.length) {
      rail.innerHTML = `
        <div class="flow-editor-rail__empty">
          <span class="flow-editor-rail__empty-glyph">✧</span>
          <span class="flow-editor-rail__empty-text">No flows yet — create one to begin.</span>
        </div>`;
      return;
    }

    rail.innerHTML = this.flows.map(f => {
      // The list endpoint returns `step_count` and omits `steps`; the
      // per-flow endpoint returns full `steps`. Prefer the count we
      // already have, then fall back to the steps array (set after
      // selectFlow() so re-renders post-edit stay accurate), then
      // fall back to a neutral em-dash so we never render "0 steps"
      // for a flow we just haven't fetched yet.
      let stepLabel;
      if (typeof f.step_count === 'number') {
        stepLabel = f.step_count === 1 ? '1 step' : `${f.step_count} steps`;
      } else if (Array.isArray(f.steps)) {
        const n = f.steps.filter(s => s.enabled !== false).length;
        stepLabel = n === 1 ? '1 step' : `${n} steps`;
      } else {
        stepLabel = '—';
      }
      const isActive = f.id === this.selectedFlowId;
      const isDefault = !!f.is_default;
      const isBuiltin = !!f.is_builtin;
      const initials = this._flowInitials(f.name || 'Flow');
      return `
        <button type="button"
                class="flow-editor-chip${isActive ? ' is-active' : ''}${isDefault ? ' is-default' : ''}${isBuiltin ? ' is-builtin' : ''}"
                data-flow-id="${escapeHtml(f.id)}"
                role="tab"
                aria-selected="${isActive ? 'true' : 'false'}"
                title="${escapeHtml(f.description || f.name || '')}">
          <span class="flow-editor-chip__mark" aria-hidden="true">${escapeHtml(initials)}</span>
          <span class="flow-editor-chip__body">
            <span class="flow-editor-chip__name">${escapeHtml(f.name || 'Untitled')}</span>
            <span class="flow-editor-chip__meta">${escapeHtml(stepLabel)}${isDefault ? ' · default' : ''}${isBuiltin ? ' · built-in' : ''}</span>
          </span>
          ${isActive ? '<span class="flow-editor-chip__pulse" aria-hidden="true"></span>' : ''}
        </button>`;
    }).join('');

    // Bind chip clicks (delegation would also work, but flow lists are short).
    rail.querySelectorAll('.flow-editor-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const id = chip.dataset.flowId;
        if (id && id !== this.selectedFlowId) this.selectFlow(id);
      });
    });

    // Auto-scroll the active chip into view so reopening the panel
    // (or switching sessions) puts the current flow under the user's eye.
    const active = rail.querySelector('.flow-editor-chip.is-active');
    if (active && typeof active.scrollIntoView === 'function') {
      // inline:'nearest' keeps it from jerking the parent panel.
      try { active.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'auto' }); } catch { /* noop */ }
    }
  }

  /** Build a 1-2 character mark for the chip avatar. Prefers initials
   *  of multi-word names ("Build Document" → "BD"); falls back to the
   *  first letter for single-word names. */
  _flowInitials(name) {
    const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
    if (!parts.length) return '?';
    if (parts.length === 1) return parts[0].slice(0, 1).toUpperCase();
    return (parts[0][0] + parts[1][0]).toUpperCase();
  }

  // -------------------------------------------------------------------------
  // selectFlow(flowId) — Fetch full flow and render info + steps
  // -------------------------------------------------------------------------

  async selectFlow(flowId) {
    this.selectedFlowId = flowId;
    this.selectedStepIndex = -1;

    const infoSection = this.$('flow-info-section');
    const stepsContainer = this.$('flow-steps-container');
    const stepEditor = this.$('flow-step-editor');
    const settingsEditor = this.$('flow-settings-editor');

    if (!flowId) {
      if (infoSection) infoSection.style.display = 'none';
      if (stepsContainer) stepsContainer.style.display = 'none';
      if (stepEditor) stepEditor.classList.add('hidden');
      if (settingsEditor) settingsEditor.classList.add('hidden');
      this._refreshMainPanel();
      return;
    }

    try {
      const resp = await fetch(`/api/reasoning/flows/${flowId}`);
      if (!resp.ok) return;
      const flow = await resp.json();
      this.currentFlow = flow;
      // Refresh chip rail so the new active chip lights up; cache the
      // freshly-fetched steps + recompute step_count so post-edit
      // re-renders pick up changes without a /flows roundtrip.
      const idx = this.flows.findIndex(f => f.id === flowId);
      if (idx >= 0) {
        const enabledCount = Array.isArray(flow.steps)
          ? flow.steps.filter(s => s.enabled !== false).length
          : 0;
        this.flows[idx] = {
          ...this.flows[idx],
          steps: flow.steps,
          step_count: enabledCount,
        };
      }
      this.renderFlowSelector();
      this._renderFlowInfo(flow);
      this.renderStepList(flow);
      if (stepEditor) stepEditor.classList.add('hidden');
      if (settingsEditor) settingsEditor.classList.add('hidden');
      // Reset the test drawer when switching flows so stale step
      // accordions from a different flow don't bleed through. An
      // in-flight run is also aborted.
      this._cancelTestRun();
      const drawer = this.$('flow-test-drawer');
      const output = this.$('flow-test-output');
      if (drawer) drawer.setAttribute('hidden', '');
      if (output) { output.innerHTML = ''; output.hidden = true; }
      this._refreshMainPanel();

      // Notify flow bar of selection change
      document.dispatchEvent(new CustomEvent('augmentum:flow-editor-selected', {
        detail: { flowId, flow },
      }));
    } catch {
      showToast('Failed to load flow', 'error');
    }
  }

  // -------------------------------------------------------------------------
  // _renderFlowInfo(flow) — Show flow name, description, badges
  // -------------------------------------------------------------------------

  _renderFlowInfo(flow) {
    const section = this.$('flow-info-section');
    if (!section) return;
    section.style.display = '';

    const nameEl = this.$('flow-info-name');
    const descEl = this.$('flow-info-desc');
    const defaultBadge = this.$('flow-badge-default');
    const builtinBadge = this.$('flow-badge-builtin');

    if (nameEl) nameEl.textContent = flow.name;
    if (descEl) descEl.textContent = flow.description || '';
    if (defaultBadge) defaultBadge.classList.toggle('hidden', !flow.is_default);
    if (builtinBadge) builtinBadge.classList.toggle('hidden', !flow.is_builtin);
  }

  // -------------------------------------------------------------------------
  // renderStepList(flow) — Render draggable step items
  // -------------------------------------------------------------------------

  renderStepList(flow) {
    const container = this.$('flow-steps-container');
    const list = this.$('flow-step-list');
    if (!container || !list) return;
    container.style.display = '';

    if (!flow.steps || flow.steps.length === 0) {
      list.innerHTML = '<div class="flow-editor-empty-steps">No steps defined.</div>';
      return;
    }

    const isBuiltin = !!flow.is_builtin;

    list.innerHTML = flow.steps.map((step, idx) => {
      const disabled = step.enabled === false;
      const eyeIcon = disabled
        ? '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/><line x1="1" y1="1" x2="23" y2="23"/>'
        : '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>';
      return `
      <div class="flow-editor-step-item${idx === this.selectedStepIndex ? ' selected' : ''}${disabled ? ' is-disabled' : ''}"
           data-step-idx="${idx}"
           draggable="true">
        <span class="flow-editor-step-grip" title="Drag to reorder">
          <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
            <circle cx="9" cy="6" r="1.5"/><circle cx="15" cy="6" r="1.5"/>
            <circle cx="9" cy="12" r="1.5"/><circle cx="15" cy="12" r="1.5"/>
            <circle cx="9" cy="18" r="1.5"/><circle cx="15" cy="18" r="1.5"/>
          </svg>
        </span>
        <span class="flow-editor-step-number">${idx + 1}</span>
        <span class="flow-editor-step-name">${escapeHtml(step.name || 'Untitled')}</span>
        ${step.model_override ? `<span class="flow-editor-step-model" title="Model: ${escapeHtml(step.model_override)}">${escapeHtml(step.model_override.length > 12 ? step.model_override.slice(0, 11) + '\u2026' : step.model_override)}</span>` : ''}
        <span class="flow-editor-step-role">${escapeHtml(step.role || 'analyze')}</span>
        ${isBuiltin ? `
        <span class="flow-editor-step-actions flow-editor-step-actions--builtin" aria-hidden="true">
          <span class="flow-editor-step-locked" title="Built-in flow \u2014 clone it to edit">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="12" height="12">
              <rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>
            </svg>
          </span>
        </span>` : `
        <span class="flow-editor-step-actions" role="group" aria-label="Step actions">
          <button type="button" class="flow-editor-step-action" data-action="toggle" title="${disabled ? 'Enable step' : 'Disable step'}" aria-label="${disabled ? 'Enable step' : 'Disable step'}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14" aria-hidden="true">${eyeIcon}</svg>
          </button>
          <button type="button" class="flow-editor-step-action" data-action="duplicate" title="Duplicate step" aria-label="Duplicate step">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14" aria-hidden="true">
              <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>
          </button>
          <button type="button" class="flow-editor-step-action flow-editor-step-action--danger" data-action="delete" title="Delete step" aria-label="Delete step">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14" aria-hidden="true">
              <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
            </svg>
          </button>
        </span>`}
      </div>
    `;
    }).join('');

    // Bind click and drag events
    this._bindStepListEvents(flow);
  }

  // -------------------------------------------------------------------------
  // Drag-and-drop support
  // -------------------------------------------------------------------------

  _bindStepListEvents(flow) {
    const list = this.$('flow-step-list');
    if (!list) return;

    const items = list.querySelectorAll('.flow-editor-step-item');
    const indicator = this.$('drag-indicator');

    items.forEach(el => {
      // Click to open step editor
      el.addEventListener('click', (e) => {
        // Ignore clicks on the grip handle during drag
        if (e.target.closest('.flow-editor-step-grip')) return;
        // Ignore clicks on the row's quick-action buttons — those have
        // their own handler below and shouldn't also open the editor.
        const actionBtn = e.target.closest('.flow-editor-step-action');
        if (actionBtn) {
          const action = actionBtn.dataset.action;
          const idx = parseInt(el.dataset.stepIdx, 10);
          this._handleStepRowAction(action, idx).catch(err =>
            console.warn('[flow-editor] step action failed', err.message || err));
          return;
        }
        const idx = parseInt(el.dataset.stepIdx, 10);
        this.openStepEditor(flow, idx);
      });

      // Drag start
      el.addEventListener('dragstart', (e) => {
        this._dragSrcIdx = parseInt(el.dataset.stepIdx, 10);
        el.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', String(this._dragSrcIdx));
      });

      // Drag end
      el.addEventListener('dragend', () => {
        el.classList.remove('dragging');
        this._dragSrcIdx = -1;
        this._dropTargetIdx = -1;
        if (indicator) {
          indicator.style.display = 'none';
        }
        // Remove all drag-over hints
        items.forEach(item => item.classList.remove('drag-over-above', 'drag-over-below'));
      });

      // Drag over
      el.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';

        const targetIdx = parseInt(el.dataset.stepIdx, 10);
        if (targetIdx === this._dragSrcIdx) {
          this._dropTargetIdx = -1;
          if (indicator) indicator.style.display = 'none';
          return;
        }

        // Determine drop position by comparing mouse Y to item midpoint
        const rect = el.getBoundingClientRect();
        const midY = rect.top + rect.height / 2;
        const above = e.clientY < midY;

        // Remove previous highlights
        items.forEach(item => item.classList.remove('drag-over-above', 'drag-over-below'));

        if (above) {
          el.classList.add('drag-over-above');
          this._dropTargetIdx = targetIdx;
        } else {
          el.classList.add('drag-over-below');
          this._dropTargetIdx = targetIdx + 1;
        }

        // Position the indicator line
        if (indicator) {
          const containerRect = this.$('flow-steps-container').getBoundingClientRect();
          const indicatorY = above ? rect.top : rect.bottom;
          indicator.style.display = 'block';
          indicator.style.top = (indicatorY - containerRect.top) + 'px';
        }
      });

      // Drop
      el.addEventListener('drop', (e) => {
        e.preventDefault();
        if (indicator) indicator.style.display = 'none';
        items.forEach(item => item.classList.remove('drag-over-above', 'drag-over-below'));

        if (this._dragSrcIdx < 0 || this._dropTargetIdx < 0) return;
        if (this._dragSrcIdx === this._dropTargetIdx) return;

        this._reorderSteps(flow, this._dragSrcIdx, this._dropTargetIdx);
      });
    });

    // Also handle dragover/drop on the list container itself to allow dropping at the end
    list.addEventListener('dragover', (e) => {
      e.preventDefault();
    });

    list.addEventListener('drop', (e) => {
      // Only handle if not caught by an item
      if (e.target === list && this._dragSrcIdx >= 0) {
        e.preventDefault();
        if (indicator) indicator.style.display = 'none';
        // Drop at end
        this._reorderSteps(flow, this._dragSrcIdx, flow.steps.length);
      }
    });
  }

  async _reorderSteps(flow, fromIdx, toIdx) {
    if (flow.is_builtin) {
      showToast('Clone this flow to edit it', 'warning');
      return;
    }

    // Perform the array reorder
    const steps = [...flow.steps];
    const [moved] = steps.splice(fromIdx, 1);

    // Adjust target index after removal
    const adjustedIdx = toIdx > fromIdx ? toIdx - 1 : toIdx;
    steps.splice(adjustedIdx, 0, moved);

    flow.steps = steps;

    // Save to backend
    try {
      const resp = await fetch(`/api/reasoning/flows/${flow.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps: flow.steps }),
      });
      if (resp.ok) {
        showToast('Steps reordered', 'success');
        this.currentFlow = flow;
        this.renderStepList(flow);
        this.onFlowChanged(flow);
      } else {
        showToast('Failed to reorder steps', 'error');
        // Reload to restore original order
        await this.selectFlow(flow.id);
      }
    } catch {
      showToast('Failed to reorder steps', 'error');
      await this.selectFlow(flow.id);
    }
  }

  // -------------------------------------------------------------------------
  // openStepEditor(flow, stepIdx) — Show the step editor form
  // -------------------------------------------------------------------------

  openStepEditor(flow, stepIdx) {
    this.currentFlow = flow;
    this.selectedStepIndex = stepIdx;
    const step = flow.steps[stepIdx];
    if (!step) return;

    const editor = this.$('flow-step-editor');
    const stepsContainer = this.$('flow-steps-container');
    if (!editor) return;

    // Single-column mode hides the step list since the editor takes its slot.
    // Two-column mode keeps it visible — editor lives in the other column.
    if (!this._isTwoColumn && stepsContainer) stepsContainer.style.display = 'none';
    editor.classList.remove('hidden');
    this._refreshMainPanel();

    // Update title
    const titleEl = this.$('step-editor-title');
    if (titleEl) titleEl.textContent = `Edit Step ${stepIdx + 1}: ${step.name || 'Untitled'}`;

    // Populate fields
    const nameInput = this.$('step-name-input');
    const roleSelect = this.$('step-role-select');
    const systemPrompt = this.$('step-system-prompt');
    const userTemplate = this.$('step-user-template');
    const outputCap = this.$('step-output-cap');
    const streamToUser = this.$('step-stream-to-user');

    if (nameInput) nameInput.value = step.name || '';
    if (roleSelect) roleSelect.value = step.role || (this.roles.length > 0 ? this.roles[0].value : 'analyze');
    if (systemPrompt) systemPrompt.value = step.system_prompt || '';
    if (userTemplate) userTemplate.value = step.user_template || '';
    if (outputCap) outputCap.value = step.output_cap != null ? step.output_cap : 800;
    if (streamToUser) streamToUser.checked = !!step.stream_to_user;

    // Highlight the matching role card. Spans both columns in two-column mode.
    const currentRole = step.role || 'analyze';
    this.$$('.fe-role-card').forEach(card => {
      const isSelected = card.dataset.role === currentRole;
      card.classList.toggle('fe-role-card--selected', isSelected);
      card.setAttribute('aria-checked', String(isSelected));
    });
    // If the role is in the "more" section, expand it
    const isPrimaryRole = this.roles.slice(0, _PRIMARY_ROLE_COUNT).some(r => r.value === currentRole);
    if (!isPrimaryRole) {
      const moreGrid = this.$('role-more-grid');
      const moreToggle = this.$('role-more-toggle');
      if (moreGrid) moreGrid.style.display = '';
      if (moreToggle) moreToggle.textContent = 'Fewer roles';
    }

    // Collapse advanced section by default (unless step has custom prompts)
    const advancedToggle = this.$('advanced-toggle');
    const hasAdvanced = (step.system_prompt || '').trim() || (step.user_template || '').trim()
      || (step.complexity_gate || []).length > 0 || (step.model_override || '').trim();
    if (advancedToggle) {
      advancedToggle.classList.toggle('open', !!hasAdvanced);
    }

    const modelOverride = this.$('step-model-override');
    if (modelOverride) modelOverride.value = step.model_override || '';

    // Complexity gate
    const gates = step.complexity_gate || [];
    ['simple', 'moderate', 'complex'].forEach(g => {
      const cb = this.$(`step-gate-${g}`);
      if (cb) cb.checked = gates.includes(g);
    });

    // Tools grid
    this._renderToolsGrid(step);

    // Disable form fields for builtin flows
    const isBuiltin = !!flow.is_builtin;
    const formInputs = editor.querySelectorAll('input, select, textarea');
    formInputs.forEach(input => {
      input.disabled = isBuiltin;
    });
    const saveBtn = this.$('step-save-btn');
    const delBtn = this.$('step-delete-btn');
    if (saveBtn) saveBtn.disabled = isBuiltin;
    if (delBtn) delBtn.disabled = isBuiltin;
    // Built-in: surface the read-only state with a Clone CTA banner
    // instead of leaving the user staring at a row of disabled inputs.
    this._renderReadonlyBanner(editor, isBuiltin);

    // Re-render step list to show selection highlight
    this.renderStepList(flow);
    // Step list is hidden but keeps selection state for when we return

    // Fresh load — baseline is clean. Any edit from here flips _stepDirty.
    this._stepDirty = false;
  }

  // -------------------------------------------------------------------------
  // _renderToolsGrid(step) — Render the tools checkbox grid
  // -------------------------------------------------------------------------

  _renderToolsGrid(step) {
    const grid = this.$('step-tools-grid');
    if (!grid) return;

    if (this._toolsFetchFailed) {
      grid.innerHTML = '<div class="flow-editor-tools-error">Could not load tools from backend. Previously configured tools are preserved.</div>';
      return;
    }
    if (!this._toolsFetched) {
      grid.innerHTML = '<div class="flow-editor-tools-loading">Loading tools...</div>';
      return;
    }

    const activeCategories = new Set(step.tool_categories || []);
    const activeNames = new Set(step.tool_names || []);

    // Group tools by their backend `category` field so non-technical users
    // see capability buckets ("Web Search", "Run Code") instead of a flat
    // wall of API names. Each group has a header checkbox that toggles the
    // category-level membership (preserves the existing
    // tool_categories/tool_names split — categories are an "all current and
    // future tools in this bucket" toggle; individual checks pin specific
    // tools).
    const toolsByCategory = new Map();
    for (const tool of this.availableTools) {
      const cat = tool.category || 'other';
      if (!toolsByCategory.has(cat)) toolsByCategory.set(cat, []);
      toolsByCategory.get(cat).push(tool);
    }

    let html = '';
    // Iterate the backend-declared category order if available so groups
    // appear in a stable, intentional sequence; trailing custom categories
    // get appended.
    const orderedCats = [
      ...Array.from(this.toolCategories).filter(c => toolsByCategory.has(c)),
      ...Array.from(toolsByCategory.keys()).filter(c => !this.toolCategories.has(c)),
    ];

    for (const cat of orderedCats) {
      const tools = toolsByCategory.get(cat) || [];
      if (tools.length === 0) continue;
      const meta = _CATEGORY_META[cat] || { label: cat, hint: '' };
      const groupChecked = activeCategories.has(cat);

      html += `
        <div class="flow-editor-tools-group${groupChecked ? ' is-active' : ''}">
          <label class="flow-editor-tools-group-header" title="${escapeHtml(meta.hint)}">
            <input type="checkbox" data-kind="category" value="${escapeHtml(cat)}"
                   ${groupChecked ? 'checked' : ''}>
            <span class="flow-editor-tools-group-name">${escapeHtml(meta.label)}</span>
            ${meta.hint ? `<span class="flow-editor-tools-group-hint">${escapeHtml(meta.hint)}</span>` : ''}
          </label>
          <div class="flow-editor-tools-group-items">
      `;
      for (const tool of tools) {
        // Friendly display name: convert snake_case → Title Case so
        // "web_search" reads as "Web search" instead of an identifier.
        const friendlyName = tool.name
          .split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        const titleAttr = tool.description
          ? `${tool.name} — ${tool.description}`
          : tool.name;
        html += `
          <label class="flow-editor-tool-label" title="${escapeHtml(titleAttr)}">
            <input type="checkbox" data-kind="tool" value="${escapeHtml(tool.name)}"
                   ${activeNames.has(tool.name) ? 'checked' : ''}>
            <span>${escapeHtml(friendlyName)}</span>
          </label>`;
      }
      html += `</div></div>`;
    }

    grid.innerHTML = html;
  }

  // -------------------------------------------------------------------------
  // closeStepEditor() — Return to step list view
  // -------------------------------------------------------------------------

  closeStepEditor({ force = false } = {}) {
    // Guard against losing unsaved form edits. ``force`` is passed by the
    // save/delete paths where the step is already persisted or gone, so the
    // prompt only appears on the back button with genuine pending edits.
    const editor = this.$('flow-step-editor');
    if (!force && this._stepDirty) {
      const open = editor && !editor.classList.contains('hidden');
      if (open && !confirm('Discard unsaved changes to this step?')) return;
    }
    this._stepDirty = false;
    const stepsContainer = this.$('flow-steps-container');
    if (editor) editor.classList.add('hidden');
    // Single-column: restore the step list that openStepEditor hid.
    // Two-column: list was never hidden — leave it alone.
    if (!this._isTwoColumn && stepsContainer) stepsContainer.style.display = '';
    this._refreshMainPanel();
    this.selectedStepIndex = -1;

    // Re-render step list without selection
    if (this.currentFlow) {
      this.renderStepList(this.currentFlow);
    }
  }

  // -------------------------------------------------------------------------
  // openFlowSettings() — Show flow-level settings editor
  // -------------------------------------------------------------------------

  openFlowSettings() {
    if (!this.currentFlow) return;

    const editor = this.$('flow-settings-editor');
    const stepsContainer = this.$('flow-steps-container');
    const stepEditor = this.$('flow-step-editor');
    const infoSection = this.$('flow-info-section');
    if (!editor) return;

    // Hide other panels, show settings.
    // Single-column hides list + info to free the slot for the editor.
    // Two-column keeps the sidebar visible (editor lives in the other column).
    if (!this._isTwoColumn && stepsContainer) stepsContainer.style.display = 'none';
    if (!this._isTwoColumn && infoSection) infoSection.style.display = 'none';
    if (stepEditor) stepEditor.classList.add('hidden');
    editor.classList.remove('hidden');
    this._refreshMainPanel();

    const flow = this.currentFlow;
    const isBuiltin = !!flow.is_builtin;

    // Populate fields
    const nameInput = this.$('settings-name');
    const descInput = this.$('settings-desc');
    const keywordsInput = this.$('settings-keywords');
    const maxToolsInput = this.$('settings-max-tools');
    const autoSearchCb = this.$('settings-auto-search');
    const autoSelectCb = this.$('settings-auto-select');

    if (nameInput) nameInput.value = flow.name || '';
    if (descInput) descInput.value = flow.description || '';
    if (keywordsInput) keywordsInput.value = (flow.trigger_keywords || []).join(', ');
    if (maxToolsInput) maxToolsInput.value = flow.max_tool_calls_per_step ?? 3;
    if (autoSearchCb) autoSearchCb.checked = flow.auto_search !== false;
    if (autoSelectCb) autoSelectCb.checked = flow.auto_select !== false;

    // Surface pre-existing keyword collisions immediately; reset any stale
    // routing-preview output from the previously edited flow.
    this._checkKeywordCollisions();
    const routingResult = this.$('settings-routing-result');
    if (routingResult) routingResult.innerHTML = '';
    const routingQuery = this.$('settings-routing-query');
    if (routingQuery) routingQuery.disabled = false;

    // Disable form for builtins — except the routing preview, which is a
    // read-only dry-run and just as useful on a builtin flow.
    const formInputs = editor.querySelectorAll('input, select, textarea');
    formInputs.forEach(input => {
      if ((input.dataset.fe || '').startsWith('settings-routing')) return;
      input.disabled = isBuiltin;
    });
    const saveBtn = this.$('settings-save-btn');
    if (saveBtn) saveBtn.disabled = isBuiltin;

    // Built-in flows can't be modified directly. Surface that with a banner
    // + Clone CTA at the top of the form so users know what to do instead
    // of staring at a wall of disabled fields.
    this._renderReadonlyBanner(editor, isBuiltin);
  }

  /**
   * Inject (or remove) a read-only banner at the top of an editor panel.
   * Centralized so we can mount the same affordance above the step editor
   * later without duplicating markup. The banner element is owned by the
   * editor — re-rendered on every open call to stay in sync with the
   * current flow's `is_builtin`.
   */
  _renderReadonlyBanner(editorEl, isBuiltin) {
    if (!editorEl) return;
    const existing = editorEl.querySelector('.fe-overlay-readonly-banner');
    if (existing) existing.remove();
    if (!isBuiltin) return;
    const banner = document.createElement('div');
    banner.className = 'fe-overlay-readonly-banner';
    banner.innerHTML = `
      <span class="fe-overlay-readonly-banner__icon" aria-hidden="true">\u{1F512}</span>
      <span class="fe-overlay-readonly-banner__msg">
        This is a built-in flow. Clone it to make changes.
      </span>
      <button type="button" class="fe-overlay-readonly-banner__action">Clone &amp; Edit</button>
    `;
    banner.querySelector('.fe-overlay-readonly-banner__action')
      ?.addEventListener('click', () => this.cloneFlow());
    // Insert at the top of the editor, before the form/header.
    editorEl.insertBefore(banner, editorEl.firstChild);
  }

  // -------------------------------------------------------------------------
  // closeFlowSettings() — Return to step list view
  // -------------------------------------------------------------------------

  closeFlowSettings() {
    const editor = this.$('flow-settings-editor');
    if (editor) editor.classList.add('hidden');
    this._refreshMainPanel();

    if (this.currentFlow) {
      this._renderFlowInfo(this.currentFlow);
      this.renderStepList(this.currentFlow);
    }
  }

  // -------------------------------------------------------------------------
  // Routing preview + keyword collision hint
  // -------------------------------------------------------------------------

  /**
   * Dry-run auto-routing for the query typed in the settings panel.
   * Backed by POST /api/reasoning/flows/routing-preview, which uses the
   * SAME scoring as live dispatch — so this is ground truth, not a guess.
   */
  async _runRoutingPreview() {
    const query = (this.$('settings-routing-query')?.value || '').trim();
    const resultEl = this.$('settings-routing-result');
    if (!resultEl) return;
    if (!query) {
      resultEl.innerHTML = '<span class="flow-editor-hint">Type a query first.</span>';
      return;
    }
    resultEl.innerHTML = '<span class="flow-editor-hint">Checking…</span>';
    try {
      const resp = await fetch('/api/reasoning/flows/routing-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        resultEl.innerHTML = `<span class="flow-editor-hint">${escapeHtml(data.error || 'Preview failed')}</span>`;
        return;
      }
      const rows = (data.candidates || []).map(c => {
        const isThis = this.currentFlow && c.id === this.currentFlow.id;
        const isWinner = data.winner && c.id === data.winner.id;
        const matched = (c.matched || []).map(m => `"${escapeHtml(m)}"`).join(', ');
        return `<div class="flow-editor-routing-row${isWinner ? ' is-winner' : ''}">
          ${isWinner ? '→' : '&nbsp;&nbsp;'} <strong>${escapeHtml(c.name)}</strong>${isThis ? ' (this flow)' : ''}
          — score ${c.score}${matched ? ` (${matched})` : ''}
        </div>`;
      }).join('');
      const headline = data.mode === 'default_flow'
        ? `<strong>${escapeHtml(data.winner?.name || '')}</strong> handles everything`
        : data.winner
          ? `<strong>${escapeHtml(data.winner.name)}</strong> would handle this query`
          : 'Standard analytical pipeline would handle this query';
      resultEl.innerHTML = `
        <div class="flow-editor-routing-headline">${headline}</div>
        ${rows}
        ${data.note ? `<div class="flow-editor-hint flow-editor-hint-warn">${escapeHtml(data.note)}</div>` : ''}
      `;
    } catch (err) {
      resultEl.innerHTML = `<span class="flow-editor-hint">Preview failed: ${escapeHtml(String(err.message || err))}</span>`;
    }
  }

  /**
   * Warn (non-blocking) when this flow's keywords exactly overlap another
   * auto-select flow — overlapping keywords make auto-routing tie, and
   * ties resolve by creation order, which reads as "my flow randomly
   * doesn't fire". Uses the already-loaded flow summaries.
   */
  _checkKeywordCollisions() {
    const el = this.$('settings-keyword-collision');
    if (!el) return [];
    const mine = (this.$('settings-keywords')?.value || '')
      .split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
    const collisions = [];
    if (mine.length) {
      for (const f of this.flows || []) {
        if (!f || f.id === this.currentFlow?.id) continue;
        if (f.auto_select === false) continue;
        const theirs = new Set((f.trigger_keywords || []).map(k => String(k).toLowerCase()));
        const shared = mine.filter(k => theirs.has(k));
        if (shared.length) collisions.push({ name: f.name, shared });
      }
    }
    if (collisions.length) {
      const detail = collisions
        .map(c => `${c.name} (${c.shared.map(s => `"${s}"`).join(', ')})`)
        .join('; ');
      el.textContent = `⚠ Shares keywords with ${detail} — ties resolve by creation order. Differentiate to route reliably.`;
      el.classList.remove('hidden');
    } else {
      el.textContent = '';
      el.classList.add('hidden');
    }
    return collisions;
  }

  // -------------------------------------------------------------------------
  // saveFlowSettings() — Save flow-level metadata to backend
  // -------------------------------------------------------------------------

  async saveFlowSettings() {
    if (!this.currentFlow) return;

    if (this.currentFlow.is_builtin) {
      showToast('Clone this flow to edit it', 'warning');
      return;
    }

    const name = (this.$('settings-name')?.value || '').trim();
    if (!name) {
      showToast('Flow name is required', 'warning');
      return;
    }

    const keywords = (this.$('settings-keywords')?.value || '')
      .split(',')
      .map(s => s.trim())
      .filter(Boolean);

    // Non-blocking: saving overlapping keywords is allowed (the user may
    // be mid-rework), but they should leave knowing routing will tie.
    const collisions = this._checkKeywordCollisions();
    if (collisions.length) {
      showToast(
        `Keywords overlap with ${collisions.map(c => c.name).join(', ')} — routing ties resolve by creation order`,
        'warning', 5000,
      );
    }

    const updates = {
      name,
      description: this.$('settings-desc')?.value || '',
      trigger_keywords: keywords,
      max_tool_calls_per_step: parseInt(this.$('settings-max-tools')?.value || '3', 10),
      auto_search: this.$('settings-auto-search')?.checked ?? true,
      auto_select: this.$('settings-auto-select')?.checked ?? true,
    };

    try {
      const resp = await fetch(`/api/reasoning/flows/${this.currentFlow.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });
      if (resp.ok) {
        showToast('Settings saved', 'success');
        await this.loadFlows();
        await this.selectFlow(this.currentFlow.id);
        this.onFlowChanged(this.currentFlow);
        this.closeFlowSettings();
      } else {
        const data = await resp.json().catch(() => ({}));
        showToast(data.error || 'Failed to save settings', 'error');
      }
    } catch (err) {
      showToast('Failed to save: ' + err.message, 'error');
    }
  }

  // -------------------------------------------------------------------------
  // saveStep() — Read form values, update step, PUT to backend
  // -------------------------------------------------------------------------

  async saveStep() {
    if (!this.currentFlow || this.selectedStepIndex < 0) return;

    if (this.currentFlow.is_builtin) {
      showToast('Clone this flow to edit it', 'warning');
      return;
    }

    const step = this.currentFlow.steps[this.selectedStepIndex];
    if (!step) return;

    // Read values from editor form
    const nameInput = this.$('step-name-input');
    const roleSelect = this.$('step-role-select');
    const systemPrompt = this.$('step-system-prompt');
    const userTemplate = this.$('step-user-template');
    const outputCap = this.$('step-output-cap');
    const streamToUser = this.$('step-stream-to-user');

    const newName = (nameInput?.value || '').trim();
    if (!newName) {
      showToast('Step name is required', 'warning');
      return;
    }

    // Warn about duplicate step names (breaks {step:Name} variable references)
    const dupIdx = this.currentFlow.steps.findIndex(
      (s, i) => i !== this.selectedStepIndex && s.name === newName
    );
    if (dupIdx >= 0) {
      showToast(`Warning: step "${newName}" duplicates step ${dupIdx + 1}. Variable {step:${newName}} will use the first match.`, 'warning');
    }

    step.name = newName;
    step.role = roleSelect?.value || step.role;
    step.system_prompt = systemPrompt?.value || '';
    step.user_template = userTemplate?.value || '';
    step.output_cap = parseInt(outputCap?.value || '800', 10);
    step.stream_to_user = streamToUser?.checked || false;

    const modelOverrideInput = this.$('step-model-override');
    step.model_override = (modelOverrideInput?.value || '').trim();

    // Complexity gate
    step.complexity_gate = [];
    ['simple', 'moderate', 'complex'].forEach(g => {
      const cb = this.$(`step-gate-${g}`);
      if (cb?.checked) step.complexity_gate.push(g);
    });

    // Tools — separate categories from individual tool names
    const grid = this.$('step-tools-grid');
    if (grid) {
      step.tool_categories = [];
      step.tool_names = [];
      grid.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
        if (cb.dataset.kind === 'category') {
          step.tool_categories.push(cb.value);
        } else {
          step.tool_names.push(cb.value);
        }
      });
    }

    // Warn if no step streams to user after this save
    const anyStreams = this.currentFlow.steps.some(s => s.stream_to_user);
    if (!anyStreams) {
      showToast('No step has "Stream to user" enabled — output will be auto-synthesized', 'warning');
    }

    // Save to backend
    try {
      const resp = await fetch(`/api/reasoning/flows/${this.currentFlow.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps: this.currentFlow.steps }),
      });
      if (resp.ok) {
        showToast('Step saved', 'success');
        this.onFlowChanged(this.currentFlow);
        this.closeStepEditor({ force: true });  // already persisted
        await this.selectFlow(this.currentFlow.id);
      } else {
        const data = await resp.json().catch(() => ({}));
        showToast(data.error || 'Failed to save step', 'error');
      }
    } catch (err) {
      showToast('Failed to save: ' + err.message, 'error');
    }
  }

  // -------------------------------------------------------------------------
  // deleteStep() — Remove step from array, PUT to backend
  // -------------------------------------------------------------------------

  async deleteStep() {
    if (!this.currentFlow || this.selectedStepIndex < 0) return;

    if (this.currentFlow.is_builtin) {
      showToast('Clone this flow to edit it', 'warning');
      return;
    }

    this.currentFlow.steps.splice(this.selectedStepIndex, 1);

    try {
      const resp = await fetch(`/api/reasoning/flows/${this.currentFlow.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps: this.currentFlow.steps }),
      });
      if (resp.ok) {
        showToast('Step deleted', 'success');
        this.onFlowChanged(this.currentFlow);
        this.closeStepEditor({ force: true });  // step is gone
        await this.selectFlow(this.currentFlow.id);
      } else {
        showToast('Failed to delete step', 'error');
      }
    } catch {
      showToast('Failed to delete step', 'error');
    }
  }

  // -------------------------------------------------------------------------
  // addStep() — Append new step to flow, PUT to backend
  // -------------------------------------------------------------------------

  async addStep() {
    if (!this.selectedFlowId) return;

    const flow = this.flows.find(f => f.id === this.selectedFlowId);
    if (flow?.is_builtin) {
      showToast('Clone this flow to edit it', 'warning');
      return;
    }

    try {
      // Fetch full flow to get current steps
      const resp = await fetch(`/api/reasoning/flows/${this.selectedFlowId}`);
      if (!resp.ok) return;
      const fullFlow = await resp.json();

      const defaultRole = this.roles.length > 0 ? this.roles[0].value : 'analyze';

      const newStep = {
        name: `Step ${(fullFlow.steps || []).length + 1}`,
        role: defaultRole,
        system_prompt: '',
        user_template: '',
        tool_names: [],
        tool_categories: [],
        complexity_gate: [],
        stream_to_user: false,
        output_cap: 800,
        enabled: true,
      };
      fullFlow.steps.push(newStep);

      const saveResp = await fetch(`/api/reasoning/flows/${this.selectedFlowId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps: fullFlow.steps }),
      });
      if (saveResp.ok) {
        showToast('Step added', 'success');
        this.onFlowChanged(fullFlow);
        await this.selectFlow(this.selectedFlowId);
      } else {
        showToast('Failed to add step', 'error');
      }
    } catch {
      showToast('Failed to add step', 'error');
    }
  }

  // -------------------------------------------------------------------------
  // Test-run drawer — sample-query composer + per-step accordion of
  // streamed events. Driven by POST /api/reasoning/flows/{id}/test (SSE).
  // Lives entirely client-side after the initial response — no session
  // state, no memory side-effects. Cancel aborts the fetch in-flight.
  // -------------------------------------------------------------------------

  _toggleTestDrawer() {
    const drawer = this.$('flow-test-drawer');
    if (!drawer) return;
    const open = drawer.hasAttribute('hidden');
    if (open) {
      drawer.removeAttribute('hidden');
      // Pre-fill the query with a reasonable default the first time.
      const query = this.$('flow-test-query');
      if (query && !query.value && this.currentFlow) {
        // Use the flow's first trigger keyword as a starter query if
        // there is one — otherwise leave blank so the placeholder shows.
        const kws = this.currentFlow.trigger_keywords || [];
        if (kws.length) query.value = kws[0];
        // Defer focus until after the layout settles so the textarea
        // doesn't fight scroll-into-view.
        requestAnimationFrame(() => { try { query.focus(); } catch { /* noop */ } });
      }
    } else {
      drawer.setAttribute('hidden', '');
      this._cancelTestRun();
    }
  }

  async _startTestRun() {
    if (this._testRunActive) return;
    if (!this.currentFlow) return;
    const query = (this.$('flow-test-query')?.value || '').trim();
    if (!query) {
      showToast('Enter a sample query to test', 'warning');
      return;
    }
    const complexity = this.$('flow-test-complexity')?.value || '';
    const allowTools = !!this.$('flow-test-tools')?.checked;

    const output = this.$('flow-test-output');
    if (output) {
      output.hidden = false;
      output.innerHTML = '<div class="flow-editor-test-status">Connecting…</div>';
    }
    this._setTestRunButtons(true);
    this._testStepStates = new Map();  // phase -> {status, content, tools}
    this._testStepOrder = [];
    this._testRunActive = true;
    this._testAbort = new AbortController();

    try {
      const resp = await fetch(`/api/reasoning/flows/${this.currentFlow.id}/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          complexity: complexity || null,
          allow_tools: allowTools,
        }),
        signal: this._testAbort.signal,
      });

      if (!resp.ok) {
        const errBody = await resp.json().catch(() => ({}));
        this._renderTestError(errBody.error || `HTTP ${resp.status}`);
        return;
      }
      if (!resp.body) {
        this._renderTestError('Streaming not supported by this browser');
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE frames are terminated by `\n\n`; loop until we've consumed
        // all complete frames in the buffer.
        let sep;
        while ((sep = buffer.indexOf('\n\n')) >= 0) {
          const frame = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          const dataLine = frame.split('\n').find(l => l.startsWith('data:'));
          if (!dataLine) continue;
          try {
            const evt = JSON.parse(dataLine.slice(5).trim());
            this._handleTestEvent(evt);
          } catch {
            // malformed frame — skip
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        this._renderTestStatus('Cancelled', 'cancelled');
      } else {
        this._renderTestError(err.message || String(err));
      }
    } finally {
      this._testRunActive = false;
      this._testAbort = null;
      this._setTestRunButtons(false);
    }
  }

  _cancelTestRun() {
    if (this._testAbort) {
      try { this._testAbort.abort(); } catch { /* noop */ }
    }
  }

  _setTestRunButtons(running) {
    const runBtn = this.$('flow-test-run');
    const cancelBtn = this.$('flow-test-cancel');
    if (runBtn) {
      runBtn.disabled = running;
      runBtn.textContent = running ? 'Running…' : 'Run';
    }
    if (cancelBtn) {
      cancelBtn.hidden = !running;
    }
  }

  _handleTestEvent(evt) {
    if (!evt || !evt.type) return;
    if (evt.type === 'start') {
      this._renderTestStatus(`Streaming ${evt.flow || 'flow'} on ${evt.model || 'model'}…`, 'running');
      return;
    }
    if (evt.type === 'done') {
      this._renderTestStatus('Run complete', 'complete');
      return;
    }
    if (evt.type === 'error') {
      this._renderTestError(evt.message || 'Unknown error');
      return;
    }
    const phase = evt.phase || '(unknown)';
    let state = this._testStepStates.get(phase);
    if (!state) {
      state = { status: 'pending', content: '', tools: [] };
      this._testStepStates.set(phase, state);
      this._testStepOrder.push(phase);
    }
    if (evt.type === 'step') {
      state.status = evt.status || state.status;
      if (evt.complexity) state.complexity = evt.complexity;
      if (evt.model) state.model = evt.model;
    } else if (evt.type === 'delta') {
      state.content += evt.content || '';
    } else if (evt.type === 'tool') {
      state.tools.push({ tool: evt.tool, status: evt.status });
    }
    this._renderTestSteps();
  }

  _renderTestStatus(message, kind = '') {
    const output = this.$('flow-test-output');
    if (!output) return;
    let strip = output.querySelector('.flow-editor-test-status');
    if (!strip) {
      strip = document.createElement('div');
      strip.className = 'flow-editor-test-status';
      output.prepend(strip);
    }
    strip.textContent = message;
    strip.dataset.kind = kind;
  }

  _renderTestError(message) {
    const output = this.$('flow-test-output');
    if (!output) return;
    output.innerHTML = `<div class="flow-editor-test-status" data-kind="error">Test run failed: ${escapeHtml(message)}</div>`;
  }

  _renderTestSteps() {
    const output = this.$('flow-test-output');
    if (!output) return;
    // Preserve the leading status strip across re-renders.
    const status = output.querySelector('.flow-editor-test-status');
    const statusHtml = status ? status.outerHTML : '';
    const stepsHtml = this._testStepOrder.map(phase => {
      const s = this._testStepStates.get(phase) || {};
      const status = s.status || 'pending';
      const tools = (s.tools || []).map(t => `
        <span class="flow-editor-test-tool" data-status="${escapeHtml(t.status || '')}">
          <span class="flow-editor-test-tool__dot"></span>
          ${escapeHtml(t.tool || 'tool')}
        </span>`).join('');
      const content = s.content
        ? `<pre class="flow-editor-test-content">${escapeHtml(s.content)}</pre>`
        : (status === 'running' ? '<div class="flow-editor-test-pending">…</div>' : '');
      return `
        <details class="flow-editor-test-step" data-status="${escapeHtml(status)}" ${status === 'running' ? 'open' : ''}>
          <summary class="flow-editor-test-step__head">
            <span class="flow-editor-test-step__dot"></span>
            <span class="flow-editor-test-step__name">${escapeHtml(phase)}</span>
            <span class="flow-editor-test-step__status">${escapeHtml(status)}</span>
          </summary>
          <div class="flow-editor-test-step__body">
            ${tools ? `<div class="flow-editor-test-tools">${tools}</div>` : ''}
            ${content}
          </div>
        </details>`;
    }).join('');
    output.innerHTML = statusHtml + stepsHtml;
  }

  // -------------------------------------------------------------------------
  // _handleStepRowAction(action, idx) — quick-actions on a step row
  //   toggle    flips step.enabled in place
  //   duplicate inserts a clone immediately after the source
  //   delete    removes after a confirm
  //   All three PUT the modified steps array via the existing flow
  //   update endpoint and then re-render. Built-ins are filtered out
  //   at render time, so we don't need a guard here — but we keep one
  //   defensively for future callers.
  // -------------------------------------------------------------------------

  async _handleStepRowAction(action, idx) {
    if (!this.currentFlow || !Number.isFinite(idx)) return;
    if (this.currentFlow.is_builtin) {
      showToast('Clone this flow to edit it', 'warning');
      return;
    }
    const steps = Array.isArray(this.currentFlow.steps) ? [...this.currentFlow.steps] : [];
    if (idx < 0 || idx >= steps.length) return;

    let toastSuccess = '';
    if (action === 'toggle') {
      const next = !(steps[idx].enabled === false);
      steps[idx] = { ...steps[idx], enabled: !next };
      toastSuccess = next ? 'Step disabled' : 'Step enabled';
    } else if (action === 'duplicate') {
      const src = steps[idx];
      // Drop the id so the backend assigns a fresh one; bump the name
      // to keep them visually distinct in the list.
      const { id: _ignored, ...rest } = src;
      const clone = { ...rest, name: `${src.name || 'Step'} copy` };
      steps.splice(idx + 1, 0, clone);
      toastSuccess = 'Step duplicated';
    } else if (action === 'delete') {
      const name = steps[idx].name || `Step ${idx + 1}`;
      if (!confirm(`Delete "${name}"?`)) return;
      steps.splice(idx, 1);
      toastSuccess = 'Step deleted';
      // If the deleted step is the one open in the editor, close it
      // so a stale form doesn't reopen against a different step's data.
      if (this.selectedStepIndex === idx) this.closeStepEditor({ force: true });
      else if (this.selectedStepIndex > idx) this.selectedStepIndex -= 1;
    } else {
      return;
    }

    try {
      const resp = await fetch(`/api/reasoning/flows/${this.currentFlow.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps }),
      });
      if (!resp.ok) {
        showToast('Failed to save step', 'error');
        return;
      }
      showToast(toastSuccess, 'success');
      this.onFlowChanged(this.currentFlow);
      await this.selectFlow(this.currentFlow.id);
    } catch {
      showToast('Failed to save step', 'error');
    }
  }

  // -------------------------------------------------------------------------
  // openCreateSheet() — Open the dedicated Create Flow sheet.
  //   After the sheet posts successfully, it calls back here so the editor
  //   can reload its flow list and select the newly created flow.  Works
  //   in both the inspector panel and inside the full-screen overlay (same
  //   FlowEditor instance either way).
  // -------------------------------------------------------------------------

  openCreateSheet() {
    openFlowCreateSheet({
      mode: this.mode,
      onCreated: async (flow) => {
        try {
          await this.loadFlows();
          await this.selectFlow(flow.id);
          this.onFlowChanged(flow);
        } catch (e) {
          console.warn('[flow-editor] Post-create refresh failed:', e.message || e);
        }
      },
    });
  }

  // -------------------------------------------------------------------------
  // cloneFlow() — Clone selected flow
  // -------------------------------------------------------------------------

  async cloneFlow() {
    if (!this.selectedFlowId) return;
    try {
      const resp = await fetch(`/api/reasoning/flows/${this.selectedFlowId}/clone`, { method: 'POST' });
      if (resp.ok) {
        const clone = await resp.json();
        showToast(`Cloned: ${clone.name}`, 'success');
        await this.loadFlows();
        await this.selectFlow(clone.id);
        this.onFlowChanged(clone);
      } else {
        showToast('Clone failed', 'error');
      }
    } catch {
      showToast('Clone failed', 'error');
    }
  }

  // -------------------------------------------------------------------------
  // setDefaultFlow() — Set selected flow as default
  // -------------------------------------------------------------------------

  async setDefaultFlow() {
    if (!this.selectedFlowId) return;
    try {
      const resp = await fetch(`/api/reasoning/flows/${this.selectedFlowId}/default`, { method: 'PUT' });
      if (resp.ok) {
        showToast('Set as default', 'success');
        await this.loadFlows();
        this.onFlowChanged(this.currentFlow);
      } else {
        showToast('Failed to set default', 'error');
      }
    } catch {
      showToast('Failed to set default', 'error');
    }
  }

  // -------------------------------------------------------------------------
  // exportFlow() — Download flow as JSON file
  // -------------------------------------------------------------------------

  async exportFlow() {
    if (!this.selectedFlowId) return;
    try {
      const resp = await fetch(`/api/reasoning/flows/${this.selectedFlowId}/export`);
      if (!resp.ok) return;
      const data = await resp.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `flow-${data.name || 'export'}.json`;
      a.click();
      URL.revokeObjectURL(url);
      showToast('Flow exported', 'success');
    } catch {
      showToast('Export failed', 'error');
    }
  }

  // -------------------------------------------------------------------------
  // deleteFlow() — Delete selected flow (not builtin)
  // -------------------------------------------------------------------------

  async deleteFlow() {
    if (!this.selectedFlowId) return;

    const flow = this.flows.find(f => f.id === this.selectedFlowId);
    if (flow?.is_builtin) {
      showToast('Cannot delete built-in flows', 'warning');
      return;
    }

    const flowName = flow?.name?.trim();
    const label = flowName ? `"${flowName}"` : 'this flow';
    if (!confirm(`Delete ${label}? This cannot be undone.`)) return;

    try {
      const resp = await fetch(`/api/reasoning/flows/${this.selectedFlowId}`, { method: 'DELETE' });
      if (resp.ok) {
        showToast('Flow deleted', 'success');
        this.selectedFlowId = '';
        this.currentFlow = null;
        await this.loadFlows();
        await this.selectFlow('');
        this.onFlowChanged(null);
      } else {
        const data = await resp.json().catch(() => ({}));
        showToast(data.error || 'Delete failed', 'error');
      }
    } catch {
      showToast('Delete failed', 'error');
    }
  }

  // -------------------------------------------------------------------------
  // importFlow() — Trigger file input for JSON import
  // -------------------------------------------------------------------------

  importFlow() {
    const fileInput = this.$('flow-import-file');
    if (!fileInput) return;
    fileInput.click();
  }

  async _handleImportFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      // Validate JSON before sending
      JSON.parse(text);

      const resp = await fetch('/api/reasoning/flows/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: text,
      });
      if (resp.ok) {
        const flow = await resp.json();
        showToast(`Imported: ${flow.name}`, 'success');
        await this.loadFlows();
        await this.selectFlow(flow.id);
        this.onFlowChanged(flow);
      } else {
        const data = await resp.json().catch(() => ({}));
        showToast(data.error || 'Import failed — check flow structure', 'error');
      }
    } catch (err) {
      showToast('Invalid JSON: ' + err.message, 'error');
    }
    e.target.value = '';
  }

  // -------------------------------------------------------------------------
  // Utility: destroy — cleanup if needed
  // -------------------------------------------------------------------------

  destroy() {
    this.el.innerHTML = '';
    // Two-column mode: also remove the editor panels we relocated to
    // mainEl, but preserve any non-flow-editor children (e.g. the
    // .fe-overlay-empty placeholder owned by the overlay HTML).
    if (this.mainEl) {
      this.mainEl.querySelectorAll('[data-fe="flow-step-editor"], [data-fe="flow-settings-editor"]')
        .forEach(node => node.remove());
      this.mainEl.classList.remove('fe-overlay-main--editing');
    }
    this.flows = [];
    this.selectedFlowId = '';
    this.selectedStepIndex = -1;
    this.currentFlow = null;
  }
}


// ==========================================================================
// Flow Editor Overlay — Full-screen workspace launcher
// ==========================================================================

let _overlayEditor = null;

const ANALYTICAL_ROLES = [
  { value: 'classify',  label: 'Classify',  description: 'Triage the question and pick a path' },
  { value: 'search',    label: 'Search',    description: 'Look things up using tools' },
  { value: 'analyze',   label: 'Analyze',   description: 'Reason over the gathered information' },
  { value: 'verify',    label: 'Verify',    description: 'Cross-check the previous answer' },
  { value: 'respond',   label: 'Respond',   description: 'Write the final user-facing reply' },
  { value: 'plan',      label: 'Plan',      description: 'Break the task into ordered steps' },
  { value: 'draft',     label: 'Draft',     description: 'Write a first version' },
  { value: 'create',    label: 'Create',    description: 'Generate the artifact (doc, image, etc.)' },
  { value: 'review',    label: 'Review',    description: 'Critique and improve the draft' },
  { value: 'deliver',   label: 'Deliver',   description: 'Package and present the result' },
  { value: 'transform', label: 'Transform', description: 'Convert between formats' },
];

/**
 * Open the full-screen flow editor overlay.
 * @param {'analytical'|'agentic'} mode
 * @param {string} [flowId] - If provided, auto-select this flow
 */
export function openFlowEditorOverlay(mode = 'analytical', flowId = '') {
  const overlay = document.getElementById('flow-editor-overlay');
  const sidebar = document.getElementById('fe-overlay-sidebar');
  const title = document.getElementById('fe-overlay-title');
  const varsBar = document.getElementById('fe-overlay-vars');
  if (!overlay || !sidebar) return;

  // Clean up previous instance
  if (_overlayEditor) {
    _overlayEditor.destroy();
    _overlayEditor = null;
  }

  const modeLabel = mode === 'agentic' ? 'Creator' : 'Thinker';
  const accent = mode === 'agentic' ? 'var(--mode-agentic)' : 'var(--mode-analytical)';

  title.innerHTML = `Reasoning Flows <span class="fe-overlay-title-badge">${escapeHtml(modeLabel)}</span>`;

  // Variable reference bar. Each chip inserts its token at the caret of the
  // currently-focused textarea/input. mousedown.preventDefault keeps focus
  // on the field so we can read selectionStart/End in the click handler —
  // otherwise the chip would steal focus and we'd lose the caret position.
  const vars = TEMPLATE_VARS[mode] || TEMPLATE_VARS.analytical;
  if (varsBar) {
    varsBar.innerHTML = '<span>Variables:</span> ' + vars.split(' ')
      .map(v => `<code data-var="${escapeHtml(v)}" title="Click to insert at cursor">${escapeHtml(v)}</code>`).join(' ');

    varsBar.onmousedown = (e) => {
      if (e.target.closest('code[data-var]')) e.preventDefault();
    };
    varsBar.onclick = (e) => {
      const chip = e.target.closest('code[data-var]');
      if (!chip) return;
      const token = chip.dataset.var;
      const active = document.activeElement;
      const isField = active && (active.tagName === 'TEXTAREA' || active.tagName === 'INPUT');
      if (!isField) {
        showToast('Click inside a text field first, then a variable', 'info');
        return;
      }
      const start = active.selectionStart ?? active.value.length;
      const end   = active.selectionEnd   ?? active.value.length;
      if (typeof active.setRangeText === 'function') {
        active.setRangeText(token, start, end, 'end');
      } else {
        active.value = active.value.slice(0, start) + token + active.value.slice(end);
        const caret = start + token.length;
        active.setSelectionRange(caret, caret);
      }
      active.focus();
      active.dispatchEvent(new Event('input', { bubbles: true }));
    };
  }

  // Create the editor as a true two-column layout: list/info in the
  // sidebar, step/settings editors in the main column. The HTML for both
  // already exists; FlowEditor relocates the editor panels into mainEl
  // after rendering and toggles `fe-overlay-main--editing` on it so the
  // empty-state placeholder hides while editing.
  _overlayEditor = new FlowEditor({
    containerEl: sidebar,
    mainContainerEl: document.getElementById('fe-overlay-main'),
    mode,
    roles: ANALYTICAL_ROLES,
    accentColor: accent,
    onFlowChanged: () => {},
  });

  _overlayEditor.init();

  // "+ New" button in overlay header — routes through the shared create sheet.
  // The FlowEditor's own flow-new-btn is hidden in the overlay (see the
  // .fe-overlay-sidebar .flow-editor-selector-bar { display: none; } rule),
  // so this is the discoverable creation entry point while the overlay is open.
  const actionsEl = document.getElementById('fe-overlay-actions');
  if (actionsEl) {
    actionsEl.innerHTML = `<button class="flow-editor-btn flow-editor-btn-primary" id="fe-overlay-new-btn">+ New</button>`;
    document.getElementById('fe-overlay-new-btn')?.addEventListener('click', () => {
      if (_overlayEditor) _overlayEditor.openCreateSheet();
    });
  }

  // "+ Add step" CTA inside the empty-state placeholder. Visible whenever
  // no step/settings editor is active — i.e. fresh open, or after deleting
  // the last step. Uses `onclick` (not addEventListener) so reopening the
  // overlay doesn't stack duplicate handlers on the persistent element.
  const emptyAddBtn = document.getElementById('fe-overlay-empty-add-btn');
  if (emptyAddBtn) {
    emptyAddBtn.onclick = () => {
      if (_overlayEditor) _overlayEditor.addStep();
    };
  }

  // If a specific flow was requested, select it after loading
  if (flowId) {
    const waitForLoad = setInterval(() => {
      if (_overlayEditor.flows.length > 0) {
        clearInterval(waitForLoad);
        _overlayEditor.selectFlow(flowId);
      }
    }, 100);
    setTimeout(() => clearInterval(waitForLoad), 5000);
  }

  // Show the overlay
  overlay.classList.remove('hidden');

  // Close handler
  document.getElementById('fe-overlay-back')?.addEventListener('click', closeFlowEditorOverlay, { once: true });

  // Escape key
  const escHandler = (e) => {
    if (e.key === 'Escape') {
      closeFlowEditorOverlay();
      document.removeEventListener('keydown', escHandler);
    }
  };
  document.addEventListener('keydown', escHandler);
}

export function closeFlowEditorOverlay() {
  const overlay = document.getElementById('flow-editor-overlay');
  if (overlay) overlay.classList.add('hidden');
  if (_overlayEditor) {
    _overlayEditor.destroy();
    _overlayEditor = null;
  }
  document.getElementById('fe-overlay-sidebar').innerHTML = '';
}

// Expose globally for settings.js onclick handlers
window.openFlowEditorOverlay = openFlowEditorOverlay;
window.closeFlowEditorOverlay = closeFlowEditorOverlay;
