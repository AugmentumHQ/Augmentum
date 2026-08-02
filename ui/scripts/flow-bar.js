/* ==========================================================================
   Flow Command Bar — compact flow selector + execution progress indicator
   Visible in analytical & agentic modes, above the chat input.
   ========================================================================== */

import { escapeHtml } from './app.js';
import { FlowPicker } from './flow-picker.js';
import { PHASE_NAMES } from './analytical-phases.js';
import { openFlowCreateSheet } from './flow-create-sheet.js';

// Re-export so existing consumers (e.g. `import { PHASE_NAMES } from './flow-bar.js'`)
// keep working.  The canonical definition lives in analytical-phases.js.
export { PHASE_NAMES };

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let _el = null;               // Root DOM element
let _flows = [];              // Cached flow list (filtered by mode)
let _currentFlow = null;      // Full flow object
let _mode = '';               // 'analytical' | 'agentic'
let _picker = null;           // FlowPicker instance (or null)
let _executing = false;       // Whether a flow is actively running
let _phases = [];             // Current execution phases [{name, status}, ...]
let _execStart = 0;           // Execution start timestamp
let _timerHandle = null;      // Interval handle for elapsed time
let _completionTimeout = null; // Handle for the 3-second idle-return timeout
let _loadGeneration = 0;      // Monotonic counter to invalidate stale loads
let _tuneOpen = false;        // Whether the quick-tune panel is expanded
let _tuneSkipSteps = new Set(); // sort_order values of steps to skip (ephemeral)
let _tuneDisableTools = new Set(); // tool names to exclude (ephemeral)

// DOM shorthand (scoped to flow-bar element IDs)
const $ = id => document.getElementById(id);

function getPrimaryInputArea() {
  return document.getElementById('chat-input')?.closest('.input-area') || null;
}

// ---------------------------------------------------------------------------
// Pipeline dot rendering
// ---------------------------------------------------------------------------
function _renderPipeline(steps, opts = {}) {
  if (!steps || !steps.length) return '';
  return steps.map((s, i) => {
    let cls;
    if (opts.executing) {
      cls = `flow-bar__dot--${s.status || 'pending'}`;
    } else if (s.enabled === false) {
      cls = 'flow-bar__dot--disabled';
    } else {
      cls = 'flow-bar__dot--idle';
    }
    const connector = i > 0 ? '<span class="flow-bar__connector"></span>' : '';
    return `${connector}<span class="flow-bar__dot ${cls}" title="${escapeHtml(s.name || '')}"></span>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// DOM creation (called once during init)
// ---------------------------------------------------------------------------
function _createEl() {
  const bar = document.createElement('div');
  bar.className = 'flow-bar flow-bar--hidden';
  bar.id = 'flow-bar';
  bar.setAttribute('role', 'toolbar');
  bar.setAttribute('aria-label', 'Reasoning flow selector');
  bar.innerHTML = `
    <div class="flow-bar__name-zone" id="flow-bar-name-zone"
         role="button" tabindex="0" aria-haspopup="listbox"
         aria-expanded="false" aria-label="Select reasoning flow">
      <span class="flow-bar__diamond" aria-hidden="true"></span>
      <span class="flow-bar__name" id="flow-bar-name">Reasoning Flow</span>
    </div>
    <div class="flow-bar__pipeline" id="flow-bar-pipeline" aria-hidden="true"></div>
    <button class="flow-bar__tune-btn" id="flow-bar-tune-btn" aria-label="Quick tune" title="Adjust steps and tools">
      <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="2" y1="4" x2="14" y2="4"/><circle cx="5" cy="4" r="1.5"/>
        <line x1="2" y1="12" x2="14" y2="12"/><circle cx="11" cy="12" r="1.5"/>
      </svg>
    </button>
    <div class="flow-bar__meta-zone">
      <span class="flow-bar__meta" id="flow-bar-meta"></span>
      <span class="flow-bar__status hidden" id="flow-bar-status" aria-live="polite"></span>
      <svg class="flow-bar__chevron" id="flow-bar-chevron"
           role="button" tabindex="0" aria-label="Toggle flow picker"
           viewBox="0 0 16 16" width="14" height="14"
           fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round">
        <polyline points="4 6 8 10 12 6"/>
      </svg>
    </div>
    <div class="flow-bar__progress" id="flow-bar-progress"></div>
    <div class="flow-bar__tune" id="flow-bar-tune" role="region" aria-label="Quick tune"></div>
  `;
  return bar;
}

// ---------------------------------------------------------------------------
// Flow loading & selection
// ---------------------------------------------------------------------------
async function _loadFlows() {
  // Don't early-return on a concurrent load. If the user switches mode
  // while a previous load is in flight, dropping the new fetch leaves
  // _flows stuck on whatever (possibly empty) state the in-flight one
  // settles to — and the stale-check below makes its result no-op.
  // Result: the flow picker shows nothing in the new mode. Letting
  // both fetches run is cheap; the stale check ensures only the
  // latest one actually populates _flows.
  const gen = ++_loadGeneration;
  try {
    const resp = await fetch('/api/reasoning/flows');
    if (!resp.ok) {
      console.warn('[flow-bar] Failed to load flows:', resp.status);
      return;
    }
    if (gen !== _loadGeneration) return;  // a newer load superseded us

    const all = await resp.json();
    _flows = _mode === 'agentic'
      ? all.filter(f => (f.trigger_domains || []).includes('agentic'))
      : all.filter(f => !(f.trigger_domains || []).includes('agentic'));

    // Auto-select the default flow if nothing selected
    if (!_currentFlow && _flows.length) {
      const def = _flows.find(f => f.is_default) || _flows[0];
      await _selectFlow(def.id, true);
    } else {
      _updateDisplay();
    }
  } catch (e) {
    console.warn('[flow-bar] Error loading flows:', e.message || e);
  }
}

async function _selectFlow(flowId, skipEvent) {
  try {
    const resp = await fetch(`/api/reasoning/flows/${flowId}`);
    if (!resp.ok) {
      console.warn('[flow-bar] Failed to load flow', flowId, resp.status);
      return;
    }
    _currentFlow = await resp.json();
    // Clear tune overrides — sort_order values are flow-specific
    _tuneSkipSteps.clear();
    _tuneDisableTools.clear();
    _closeTune();
    _updateDisplay();
    if (!skipEvent) {
      document.dispatchEvent(new CustomEvent('augmentum:flow-bar-selected', {
        detail: { flowId, flow: _currentFlow },
      }));
    }
  } catch (e) {
    console.warn('[flow-bar] Error selecting flow:', e.message || e);
  }
}

// ---------------------------------------------------------------------------
// Display update (idle state)
// ---------------------------------------------------------------------------
function _updateDisplay() {
  if (!_el || !_currentFlow) return;
  const nameEl = $('flow-bar-name');
  const pipeEl = $('flow-bar-pipeline');
  const metaEl = $('flow-bar-meta');

  if (nameEl) nameEl.textContent = _currentFlow.name || 'Default';
  if (pipeEl) {
    // Apply tune overrides: mark skipped steps as disabled so dots reflect the override
    const steps = (_currentFlow.steps || []).filter(s => s.enabled !== false).map(s => {
      if (_tuneSkipSteps.size > 0 && _tuneSkipSteps.has(s.sort_order)) {
        return { ...s, enabled: false };
      }
      return s;
    });
    pipeEl.innerHTML = _renderPipeline(steps);
  }
  if (metaEl) {
    const allSteps = (_currentFlow.steps || []).filter(s => s.enabled !== false);
    const activeSteps = allSteps.filter(s => !_tuneSkipSteps.has(s.sort_order));
    const activeTools = new Set(activeSteps.flatMap(s => s.tool_names || []));
    // Subtract tune-disabled tools
    for (const t of _tuneDisableTools) activeTools.delete(t);
    const stepCount = activeSteps.length;
    const toolCount = activeTools.size;
    metaEl.textContent = `${stepCount} step${stepCount !== 1 ? 's' : ''} \u00B7 ${toolCount} tool${toolCount !== 1 ? 's' : ''}`;
  }
}

// ---------------------------------------------------------------------------
// Execution state management
// ---------------------------------------------------------------------------
function _startTimer() {
  _execStart = Date.now();
  _timerHandle = setInterval(() => {
    const elapsed = ((Date.now() - _execStart) / 1000).toFixed(1);
    const statusEl = $('flow-bar-status');
    if (!statusEl) return;
    const running = _phases.find(p => p.status === 'running');
    const label = running ? (PHASE_NAMES[running.name] || running.name) : '';
    statusEl.textContent = `${label}${label ? '\u2026' : ''} ${elapsed}s`;
  }, 500);
}

function _stopTimer() {
  if (_timerHandle) { clearInterval(_timerHandle); _timerHandle = null; }
}

function _clearCompletionTimeout() {
  if (_completionTimeout) { clearTimeout(_completionTimeout); _completionTimeout = null; }
}

function _updatePipelineExec() {
  const pipeEl = $('flow-bar-pipeline');
  if (pipeEl) pipeEl.innerHTML = _renderPipeline(_phases, { executing: true });

  // Update progress bar width
  const total = _phases.length || 1;
  const done = _phases.filter(p => p.status === 'complete' || p.status === 'skipped').length;
  const running = _phases.some(p => p.status === 'running') ? 0.5 : 0;
  const pct = Math.min(100, ((done + running) / total) * 100);
  const progEl = $('flow-bar-progress');
  if (progEl) progEl.style.width = `${pct}%`;
}

/** Restore the bar to idle visual state (shared by completion timeout and reset). */
function _restoreIdleVisuals() {
  if (_el) _el.classList.remove('flow-bar--complete');
  const metaEl = $('flow-bar-meta');
  const statusEl = $('flow-bar-status');
  const chevronEl = $('flow-bar-chevron');
  const progEl = $('flow-bar-progress');
  if (metaEl) metaEl.classList.remove('hidden');
  if (statusEl) statusEl.classList.add('hidden');
  if (chevronEl) chevronEl.classList.remove('hidden');
  if (progEl) progEl.style.width = '0%';
  // Update aria state
  const nameZone = $('flow-bar-name-zone');
  if (nameZone) nameZone.setAttribute('aria-expanded', 'false');
  _updateDisplay();
}

// ---------------------------------------------------------------------------
// Quick Tune panel
// ---------------------------------------------------------------------------
function _toggleTune() {
  if (_executing) return;
  // Tune overrides not yet wired for agentic flows — disable until connected
  if (_mode === 'agentic') return;
  if (_tuneOpen) {
    _closeTune();
    return;
  }
  _closePicker();
  _tuneOpen = true;
  if (_el) _el.classList.add('flow-bar--tuning');
  _renderTunePanel();
}

function _closeTune() {
  _tuneOpen = false;
  if (_el) _el.classList.remove('flow-bar--tuning');
  const tuneEl = $('flow-bar-tune');
  if (tuneEl) tuneEl.innerHTML = '';
}

function _renderTunePanel() {
  const tuneEl = $('flow-bar-tune');
  if (!tuneEl || !_currentFlow) return;

  const steps = (_currentFlow.steps || []).filter(s => s.enabled !== false);
  // Collect unique tools from enabled (non-skipped) steps
  const activeSteps = steps.filter(s => !_tuneSkipSteps.has(s.sort_order));
  const allTools = new Set(activeSteps.flatMap(s => s.tool_names || []));
  // Also include tools from skipped steps (so they can be re-enabled)
  steps.forEach(s => (s.tool_names || []).forEach(t => allTools.add(t)));

  const stepPills = steps.map((s, i) => {
    const disabled = _tuneSkipSteps.has(s.sort_order);
    const connCls = disabled ? 'flow-bar__tune-conn--disabled' : '';
    const conn = i > 0 ? `<span class="flow-bar__tune-conn ${connCls}"></span>` : '';
    return `${conn}<span class="flow-bar__tune-pill${disabled ? ' flow-bar__tune-pill--disabled' : ''}"
      data-sort="${s.sort_order}" role="switch" aria-checked="${!disabled}" tabindex="0"
      title="${escapeHtml(s.name)}">${escapeHtml(s.name)}</span>`;
  }).join('');

  const toolChips = [...allTools].sort().map(t => {
    const disabled = _tuneDisableTools.has(t);
    return `<span class="flow-bar__tune-tool${disabled ? ' flow-bar__tune-tool--disabled' : ''}"
      data-tool="${t}" role="switch" aria-checked="${!disabled}" tabindex="0">${escapeHtml(t)}</span>`;
  }).join('');

  tuneEl.innerHTML = `
    <div class="flow-bar__tune-row">
      <span class="flow-bar__tune-label">Steps</span>
      ${stepPills}
    </div>
    ${allTools.size > 0 ? `<div class="flow-bar__tune-row">
      <span class="flow-bar__tune-label">Tools</span>
      ${toolChips}
    </div>` : ''}
    <div class="flow-bar__tune-footer">
      <button class="flow-bar__tune-done">Done</button>
      <button class="flow-bar__tune-edit">Edit full flow \u2192</button>
    </div>
  `;

  // Bind pill clicks
  tuneEl.querySelectorAll('.flow-bar__tune-pill').forEach(pill => {
    const handler = () => {
      const sortOrder = parseInt(pill.dataset.sort, 10);
      // Prevent disabling all steps
      const enabledCount = steps.filter(s => !_tuneSkipSteps.has(s.sort_order)).length;
      if (!_tuneSkipSteps.has(sortOrder)) {
        if (enabledCount <= 1) return; // Must keep at least 1
        _tuneSkipSteps.add(sortOrder);
      } else {
        _tuneSkipSteps.delete(sortOrder);
      }
      _renderTunePanel(); // Re-render with updated state
      _updateDisplay();   // Update pipeline dots
    };
    pill.addEventListener('click', handler);
    pill.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handler(); }
    });
  });

  // Bind tool clicks
  tuneEl.querySelectorAll('.flow-bar__tune-tool').forEach(chip => {
    const handler = () => {
      const toolName = chip.dataset.tool;
      if (_tuneDisableTools.has(toolName)) {
        _tuneDisableTools.delete(toolName);
      } else {
        _tuneDisableTools.add(toolName);
      }
      _renderTunePanel();
    };
    chip.addEventListener('click', handler);
    chip.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handler(); }
    });
  });

  // Bind footer buttons
  tuneEl.querySelector('.flow-bar__tune-done')
    ?.addEventListener('click', _closeTune);
  tuneEl.querySelector('.flow-bar__tune-edit')
    ?.addEventListener('click', () => {
      _closeTune();
      if (window.openFlowEditorOverlay) window.openFlowEditorOverlay(_mode);
    });
}

// ---------------------------------------------------------------------------
// Picker toggle
// ---------------------------------------------------------------------------
function _togglePicker() {
  if (_executing) return;
  if (_picker) {
    _closePicker();
    return;
  }

  _closeTune(); // Close tune panel if open
  $('flow-bar-chevron')?.classList.add('flow-bar__chevron--open');
  const nameZone = $('flow-bar-name-zone');
  if (nameZone) nameZone.setAttribute('aria-expanded', 'true');

  _picker = new FlowPicker({
    anchorEl: _el,
    accentColor: _mode === 'agentic' ? 'var(--mode-agentic)' : 'var(--mode-analytical)',
    onSelect: (flow) => {
      _selectFlow(flow.id);
      _closePicker();
    },
    onCreate: () => {
      _closePicker();
      openFlowCreateSheet({ mode: _mode });
    },
    onEdit: () => {
      _closePicker();
      if (window.openFlowEditorOverlay) window.openFlowEditorOverlay(_mode);
    },
    onDismiss: () => {
      // Called when user clicks backdrop or presses Escape inside the picker.
      // The picker has already removed its own DOM; we just need to sync state.
      _picker = null;
      $('flow-bar-chevron')?.classList.remove('flow-bar__chevron--open');
      const nz = $('flow-bar-name-zone');
      if (nz) nz.setAttribute('aria-expanded', 'false');
    },
  });
  _picker.show(_flows, _currentFlow?.id);
}

function _closePicker() {
  if (_picker) { _picker.hide(); _picker = null; }
  $('flow-bar-chevron')?.classList.remove('flow-bar__chevron--open');
  const nameZone = $('flow-bar-name-zone');
  if (nameZone) nameZone.setAttribute('aria-expanded', 'false');
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Create the command bar and insert it into the DOM. Call once at startup. */
export function initFlowBar() {
  _el = _createEl();

  // Insert inline into the input toolbar, between context controls and mode toggles
  // Fallback: before the input-wrapper if the toolbar is not present
  const toolbar = document.getElementById('input-toolbar');
  const divider = toolbar?.querySelector('.input-toolbar-divider');
  if (toolbar && divider) {
    toolbar.insertBefore(_el, divider);
  } else if (toolbar) {
    toolbar.appendChild(_el);
  } else {
    const inputArea = getPrimaryInputArea();
    const inputWrapper = inputArea?.querySelector('.input-wrapper');
    if (inputArea && inputWrapper) {
      inputArea.insertBefore(_el, inputWrapper);
    }
  }

  // Click handlers
  $('flow-bar-name-zone')?.addEventListener('click', _togglePicker);
  $('flow-bar-name-zone')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); _togglePicker(); }
  });
  // Pipeline zone → opens quick tune (desktop)
  $('flow-bar-pipeline')?.addEventListener('click', (e) => {
    e.stopPropagation();
    _toggleTune();
  });
  // Tune button → opens quick tune (always visible, used on mobile where pipeline is hidden)
  $('flow-bar-tune-btn')?.addEventListener('click', (e) => {
    e.stopPropagation();
    _toggleTune();
  });
  $('flow-bar-chevron')?.addEventListener('click', (e) => {
    e.stopPropagation();
    _togglePicker();
  });
  $('flow-bar-chevron')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); _togglePicker(); }
  });

  // Listen for mode changes
  document.addEventListener('augmentum:mode-changed', (e) => {
    const mode = e.detail?.mode;
    if (mode === 'analytical' || mode === 'agentic') {
      showFlowBar(mode);
    } else {
      hideFlowBar();
    }
  });

  // Listen for flow selection from inspector's FlowEditor
  document.addEventListener('augmentum:flow-editor-selected', (e) => {
    if (e.detail?.flow) {
      _currentFlow = e.detail.flow;
      _updateDisplay();
    }
  });

  // Listen for new flow creation from the shared Create Flow sheet.
  // Refresh our cached flow list so the next picker open shows it, and if
  // the new flow matches our current mode, select it immediately.
  document.addEventListener('augmentum:flow-created', (e) => {
    const flow = e.detail?.flow;
    if (!flow) return;
    // Reload so the picker grid is up to date
    _loadFlows();
    // If the new flow belongs to the current mode, make it active
    const belongsToMode = _mode === 'agentic'
      ? (flow.trigger_domains || []).includes('agentic')
      : !(flow.trigger_domains || []).includes('agentic');
    if (belongsToMode) _selectFlow(flow.id);
  });

  // Keyboard shortcut: Ctrl/Cmd + Shift + F
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'f') {
      if (_el && !_el.classList.contains('flow-bar--hidden')) {
        e.preventDefault();
        _togglePicker();
      }
    }
  });
}

/** Show the command bar for the given mode. */
export function showFlowBar(mode) {
  _mode = mode || 'analytical';
  ++_loadGeneration; // Invalidate any in-flight loads for a previous mode
  if (!_el) return;
  _el.classList.remove('flow-bar--hidden');
  _el.style.setProperty('--flow-bar-accent',
    _mode === 'agentic' ? 'var(--mode-agentic)' : 'var(--mode-analytical)');
  _loadFlows();
}

/** Hide the command bar. */
export function hideFlowBar() {
  if (_el) _el.classList.add('flow-bar--hidden');
  _closePicker();
  _closeTune();
  _stopTimer();
  _clearCompletionTimeout();
}

/** Called when a flow starts executing (first streaming metadata arrives). */
export function onExecutionStart(flowName, phases) {
  _executing = true;
  _closeTune();
  _clearCompletionTimeout();
  _phases = (phases || []).map(p => ({ ...p }));
  if (!_el) return;

  _el.classList.add('flow-bar--executing');
  _el.classList.remove('flow-bar--complete');

  // Update name if provided
  if (flowName) {
    const nameEl = $('flow-bar-name');
    if (nameEl) nameEl.textContent = flowName;
  }

  // Swap meta → status
  const metaEl = $('flow-bar-meta');
  const statusEl = $('flow-bar-status');
  const chevronEl = $('flow-bar-chevron');
  if (metaEl) metaEl.classList.add('hidden');
  if (statusEl) statusEl.classList.remove('hidden');
  if (chevronEl) chevronEl.classList.add('hidden');

  _updatePipelineExec();
  _startTimer();
}

/** Called during streaming as phases progress. */
export function onPhaseUpdate(phases) {
  if (!phases) return;
  _phases = phases.map(p => ({ ...p }));
  _updatePipelineExec();
}

/** Called when flow execution completes. */
export function onExecutionComplete() {
  _executing = false;
  _stopTimer();
  _clearCompletionTimeout();
  if (!_el) return;

  _el.classList.remove('flow-bar--executing');
  _el.classList.add('flow-bar--complete');

  // Show final elapsed time
  const elapsed = ((Date.now() - _execStart) / 1000).toFixed(1);
  const statusEl = $('flow-bar-status');
  if (statusEl) statusEl.textContent = `Complete \u00B7 ${elapsed}s`;

  // Progress bar full
  const progEl = $('flow-bar-progress');
  if (progEl) progEl.style.width = '100%';

  // After 3s, return to idle state
  _completionTimeout = setTimeout(() => {
    _completionTimeout = null;
    if (_executing) return; // New execution started during the wait
    _restoreIdleVisuals();
  }, 3000);
}

/** Reset the bar to idle (e.g. on new message/session). */
export function resetFlowBar() {
  _executing = false;
  _stopTimer();
  _clearCompletionTimeout();
  _phases = [];
  if (!_el) return;
  _el.classList.remove('flow-bar--executing', 'flow-bar--complete');
  _restoreIdleVisuals();
}

/** Get the currently selected flow (for other modules to query). */
export function getCurrentFlow() {
  return _currentFlow;
}

/**
 * Get ephemeral tune overrides for the next message.
 * Returns null if no overrides are active.
 * Format: { skip_steps: number[], disable_tools: string[] }
 */
export function getTuneOverrides() {
  if (_tuneSkipSteps.size === 0 && _tuneDisableTools.size === 0) return null;
  return {
    skip_steps: [..._tuneSkipSteps],
    disable_tools: [..._tuneDisableTools],
  };
}

/** Clear tune overrides and collapse the tune panel. Called after message send. */
export function clearTuneOverrides() {
  _tuneSkipSteps.clear();
  _tuneDisableTools.clear();
  _closeTune();
  _updateDisplay();
}
