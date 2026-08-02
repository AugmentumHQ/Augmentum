/* ==========================================================================
   Augmentum — Analytical Module
   Reasoning panel: flow editor, live execution view, UARF phase rendering
   ========================================================================== */

import { app, escapeHtml, showToast } from './app.js';
import { FlowEditor } from './flow-editor.js';
import { PHASE_NAMES as PHASE_DISPLAY_NAMES, PHASE_DESCRIPTIONS } from './analytical-phases.js';

// Icons from the shared icon set (set by chat.js on window).
// Uses a Proxy so lookups resolve at call-time (after all modules have loaded).
const _iconFallback = { checkSmall: '\u2713', xSmall: '\u2717', chevronDown: '\u25BC', chevronRightSmall: '\u25B6', dotFilled: '\u25CF', dotEmpty: '\u25CB' };
const icons = new Proxy(_iconFallback, { get: (fb, key) => (window.icons && window.icons[key]) || fb[key] });

// ---------------------------------------------------------------------------
// Streaming State — accumulated during streaming
// ---------------------------------------------------------------------------

let currentPhases = [];
let currentComplexity = '';
let currentConfidence = null;
let currentFlowName = '';
let phaseContent = {};
let phaseTimings = {};       // { phaseName: { start: ms, end: ms } }
let toolCalls = [];
let expandedPhases = new Set();
let expandedToolCards = new Set();

export function resetReasoningState() {
  currentPhases = [];
  currentComplexity = '';
  currentConfidence = null;
  currentFlowName = '';
  phaseContent = {};
  phaseTimings = {};
  toolCalls = [];
  expandedPhases = new Set();
  expandedToolCards = new Set();
}

/** Update the active flow name (called from handleAugmentumMeta). */
export function updateFlowName(name) {
  if (name && name !== currentFlowName) {
    currentFlowName = name;
    renderReasoningPhases(currentPhases, currentComplexity);
  }
}

/** Handle backtrack events from the verify step. */
export function handleBacktrack(data) {
  const { step, confidence, reason, count } = data;
  const key = `${step}_backtrack_${count}`;
  if (!phaseContent[key]) phaseContent[key] = '';
  phaseContent[key] = `Backtrack #${count}: confidence ${Math.round((confidence || 0) * 100)}% — ${reason || 'verification failed'}`;
  renderReasoningPhases(currentPhases, currentComplexity);
}

/** Handle revision events from the review step. */
export function handleRevision(data) {
  const { review_step, draft_step, count } = data;
  const key = `${draft_step}_revision_${count}`;
  if (!phaseContent[key]) phaseContent[key] = '';
  phaseContent[key] = `Revision #${count}: re-running "${draft_step}" based on review from "${review_step}"`;
  renderReasoningPhases(currentPhases, currentComplexity);
}

// Global toggle handler (called from inline onclick)
window.__togglePhaseExpand = function(phaseName) {
  if (expandedPhases.has(phaseName)) {
    expandedPhases.delete(phaseName);
  } else {
    expandedPhases.add(phaseName);
  }
  renderReasoningPhases(currentPhases, currentComplexity);
};

// Global tool card expand handler
window.__toggleToolExpand = function(idx) {
  if (expandedToolCards.has(idx)) {
    expandedToolCards.delete(idx);
  } else {
    expandedToolCards.add(idx);
  }
  renderReasoningPhases(currentPhases, currentComplexity);
};

// Global tool detail modal
window.__showToolDetail = function(idx) {
  const tc = toolCalls[idx];
  if (!tc) return;

  const existing = document.querySelector('.tool-detail-modal-overlay');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.className = 'tool-detail-modal-overlay';
  overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };

  const inputStr = tc.input ? JSON.stringify(tc.input, null, 2) : '';
  const outputStr = tc.output || tc.error || '';
  const statusLabel = tc.success ? 'Success' : 'Failed';
  const statusClass = tc.success ? 'success' : 'error';

  overlay.innerHTML = `
    <div class="tool-detail-modal">
      <div class="tool-detail-modal-header">
        <div class="tool-detail-modal-title">
          <span class="reasoning-tool-status ${statusClass}">${tc.success ? icons.checkSmall : icons.xSmall}</span>
          <span>${escapeHtml(tc.tool)}</span>
          <span class="tool-detail-modal-phase">${escapeHtml(tc.phase || '')}</span>
        </div>
        <button class="tool-detail-modal-close" onclick="this.closest('.tool-detail-modal-overlay').remove()">&times;</button>
      </div>
      ${inputStr ? `
        <div class="tool-detail-modal-section">
          <div class="tool-detail-modal-section-label">Input</div>
          <pre class="tool-detail-modal-code">${escapeHtml(inputStr)}</pre>
        </div>
      ` : ''}
      <div class="tool-detail-modal-section">
        <div class="tool-detail-modal-section-label">${tc.success ? 'Output' : 'Error'}</div>
        <pre class="tool-detail-modal-code ${statusClass}">${escapeHtml(outputStr)}</pre>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
};

// ---------------------------------------------------------------------------
// View State
// ---------------------------------------------------------------------------

let editorView = 'editor'; // 'editor' | 'live' | 'streaming'

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Phase timing helpers
// ---------------------------------------------------------------------------

function trackPhaseTiming(phases) {
  const now = Date.now();
  for (const p of phases) {
    if (!phaseTimings[p.name]) {
      phaseTimings[p.name] = {};
    }
    if (p.status === 'running' && !phaseTimings[p.name].start) {
      phaseTimings[p.name].start = now;
    }
    if (p.status === 'complete' && !phaseTimings[p.name].end) {
      phaseTimings[p.name].end = now;
    }
  }
}

function formatDuration(ms) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function getPhaseDuration(phaseName) {
  const t = phaseTimings[phaseName];
  if (!t || !t.start) return null;
  const end = t.end || Date.now();
  return end - t.start;
}

function getTotalDuration() {
  let earliest = Infinity, latest = 0;
  for (const t of Object.values(phaseTimings)) {
    if (t.start && t.start < earliest) earliest = t.start;
    if (t.end && t.end > latest) latest = t.end;
  }
  if (earliest === Infinity) return null;
  return (latest || Date.now()) - earliest;
}

// ---------------------------------------------------------------------------
// SVG Phase Pipeline Builder
// ---------------------------------------------------------------------------

function buildPhaseSvg(phases, tools) {
  if (!phases || phases.length === 0) return '';

  const nodeW = 64;
  const nodeH = 32;
  const gapX = 8;
  const paddingX = 12;
  const paddingY = 12;

  const svgW = phases.length * (nodeW + gapX) - gapX + paddingX * 2;
  const svgH = nodeH + paddingY * 2 + 4;

  let svg = `<svg class="phase-pipeline-svg" viewBox="0 0 ${svgW} ${svgH}" xmlns="http://www.w3.org/2000/svg">`;

  // Progress track (background)
  const trackY = paddingY + nodeH / 2;
  const trackX1 = paddingX + nodeW / 2;
  const trackX2 = paddingX + (phases.length - 1) * (nodeW + gapX) + nodeW / 2;
  if (phases.length > 1) {
    svg += `<line class="phase-track" x1="${trackX1}" y1="${trackY}" x2="${trackX2}" y2="${trackY}"/>`;

    // Completed progress overlay
    const lastComplete = phases.reduce((acc, p, i) => p.status === 'complete' ? i : acc, -1);
    const runningIdx = phases.findIndex(p => p.status === 'running');
    const progressEnd = runningIdx >= 0 ? runningIdx : lastComplete >= 0 ? lastComplete : -1;
    if (progressEnd > 0) {
      const pX2 = paddingX + progressEnd * (nodeW + gapX) + nodeW / 2;
      svg += `<line class="phase-track-progress" x1="${trackX1}" y1="${trackY}" x2="${pX2}" y2="${trackY}"/>`;
    }
  }

  // Draw nodes
  phases.forEach((p, i) => {
    const cx = paddingX + i * (nodeW + gapX) + nodeW / 2;
    const cy = paddingY + nodeH / 2;
    const x = paddingX + i * (nodeW + gapX);
    const y = paddingY;
    const displayName = PHASE_DISPLAY_NAMES[p.name] || p.name;
    const status = p.status || 'pending';

    // Tool count badge
    const phaseToolCount = (tools || []).filter(tc => (tc.phase || 'APPLY') === p.name).length;

    svg += `<g class="phase-node" data-phase="${escapeHtml(p.name)}">`;

    // Node background
    svg += `<rect class="phase-node-bg ${status}" x="${x}" y="${y}" width="${nodeW}" height="${nodeH}"/>`;

    // Status indicator dot
    if (status === 'complete') {
      svg += `<circle class="phase-dot complete" cx="${cx}" cy="${cy - 4}" r="4"/>`;
      svg += `<text class="phase-check" x="${cx}" y="${cy - 1}">\u2713</text>`;
    } else if (status === 'running') {
      svg += `<circle class="phase-dot running" cx="${cx}" cy="${cy - 4}" r="4"/>`;
    } else if (status === 'skipped') {
      svg += `<circle class="phase-dot skipped" cx="${cx}" cy="${cy - 4}" r="3"/>`;
    } else {
      svg += `<circle class="phase-dot pending" cx="${cx}" cy="${cy - 4}" r="3"/>`;
    }

    // Label
    svg += `<text class="phase-label ${status}" x="${cx}" y="${cy + 10}">${escapeHtml(displayName)}</text>`;

    // Tool count indicator
    if (phaseToolCount > 0) {
      const badgeX = x + nodeW - 6;
      const badgeY = y + 6;
      svg += `<circle class="phase-tool-badge" cx="${badgeX}" cy="${badgeY}" r="6"/>`;
      svg += `<text class="phase-tool-badge-text" x="${badgeX}" y="${badgeY + 3.5}">${phaseToolCount}</text>`;
    }

    svg += '</g>';
  });

  svg += '</svg>';
  return svg;
}

// ---------------------------------------------------------------------------
// Reasoning Panel Rendering (streaming phases)
// ---------------------------------------------------------------------------

export function renderReasoningPhases(phases, complexity) {
  const container = $('reasoning-content');
  if (!container) return;

  currentPhases = phases || currentPhases;
  if (complexity) currentComplexity = complexity;

  // Track timing
  if (currentPhases.length > 0) {
    trackPhaseTiming(currentPhases);
  }

  // Switch to streaming view
  switchView('streaming');

  if (!currentPhases || currentPhases.length === 0) {
    container.innerHTML = `
      <div class="reasoning-empty">
        <div class="reasoning-empty-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
          </svg>
        </div>
        <p class="reasoning-empty-text">Reasoning phases will appear here during analytical mode</p>
      </div>`;
    return;
  }

  const completedCount = currentPhases.filter(p => p.status === 'complete').length;
  const totalCount = currentPhases.length;
  const allComplete = completedCount === totalCount;
  const runningPhase = currentPhases.find(p => p.status === 'running');
  const totalTime = getTotalDuration();

  // Header with progress ring
  const progressPct = totalCount > 0 ? (completedCount / totalCount) * 100 : 0;
  const circumference = 2 * Math.PI * 10;
  const dashOffset = circumference - (progressPct / 100) * circumference;

  const headerLabel = allComplete
    ? 'Analysis complete'
    : runningPhase
      ? escapeHtml(PHASE_DISPLAY_NAMES[runningPhase.name] || runningPhase.name)
      : 'Preparing...';

  const headerSub = allComplete
    ? `${totalCount} phases${totalTime ? ' \u00b7 ' + formatDuration(totalTime) : ''}`
    : `${completedCount} of ${totalCount} phases`;

  let html = `
    <div class="reasoning-panel-inner">
      <div class="reasoning-header-bar">
        <div class="reasoning-progress-ring">
          <svg viewBox="0 0 24 24">
            <circle class="reasoning-ring-bg" cx="12" cy="12" r="10"/>
            <circle class="reasoning-ring-fill ${allComplete ? 'complete' : ''}" cx="12" cy="12" r="10"
                    stroke-dasharray="${circumference}" stroke-dashoffset="${dashOffset}"
                    transform="rotate(-90 12 12)"/>
          </svg>
          ${allComplete
            ? `<span class="reasoning-ring-check">${icons.checkSmall}</span>`
            : `<span class="reasoning-ring-count">${completedCount}</span>`}
        </div>
        <div class="reasoning-header-text">
          <span class="reasoning-header-label">${headerLabel}</span>
          <span class="reasoning-header-sub">${headerSub}</span>
        </div>
        <div class="reasoning-header-badges">
          ${currentFlowName ? `<span class="reasoning-badge reasoning-badge-flow">${escapeHtml(currentFlowName)}</span>` : ''}
          ${currentComplexity ? `<span class="reasoning-badge reasoning-badge-complexity">${escapeHtml(currentComplexity)}</span>` : ''}
          ${currentConfidence !== null ? `<span class="reasoning-badge reasoning-badge-confidence">${Math.round(currentConfidence * 100)}%</span>` : ''}
        </div>
      </div>

      ${buildPhaseSvg(currentPhases, toolCalls)}

      <div class="reasoning-phases">
  `;

  currentPhases.forEach(p => {
    const displayName = PHASE_DISPLAY_NAMES[p.name] || p.name;
    const description = PHASE_DESCRIPTIONS[p.name] || '';
    const status = p.status || 'pending';
    const duration = getPhaseDuration(p.name);

    const content = phaseContent[p.name] || p.output || '';
    const hasContent = content.length > 0;
    const isExpanded = expandedPhases.has(p.name);
    const phaseToolCount = toolCalls.filter(tc => (tc.phase || 'APPLY') === p.name).length;

    html += `
      <div class="reasoning-phase-block ${status}" data-phase="${escapeHtml(p.name)}">
        <div class="reasoning-phase-header${hasContent ? ' clickable' : ''}">
          <div class="reasoning-phase-status-icon ${status}">
            ${status === 'complete' ? '<svg viewBox="0 0 16 16"><polyline points="3.5 8 6.5 11 12.5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            : status === 'running' ? '<div class="reasoning-phase-spinner"></div>'
            : status === 'skipped' ? '<svg viewBox="0 0 16 16"><line x1="4" y1="8" x2="12" y2="8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
            : '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="3" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>'}
          </div>
          <div class="reasoning-phase-info">
            <span class="reasoning-phase-name">${escapeHtml(displayName)}</span>
            ${!hasContent && description ? `<span class="reasoning-phase-desc">${escapeHtml(description)}</span>` : ''}
            ${p.step_model ? `<span class="reasoning-phase-model-badge" title="Using model: ${escapeHtml(p.step_model)}">${escapeHtml(p.step_model)}</span>` : ''}
          </div>
          <div class="reasoning-phase-meta">
            ${phaseToolCount > 0 ? `<span class="reasoning-phase-tool-count" title="${phaseToolCount} tool call${phaseToolCount > 1 ? 's' : ''}">${phaseToolCount}</span>` : ''}
            ${duration !== null ? `<span class="reasoning-phase-time">${formatDuration(duration)}</span>` : ''}
            ${hasContent ? `<span class="reasoning-phase-toggle">${isExpanded ? icons.chevronDown : icons.chevronRightSmall}</span>` : ''}
          </div>
        </div>
        ${hasContent ? `
          <div class="reasoning-phase-content-box ${isExpanded ? 'expanded' : 'collapsed'}"
               id="phase-content-${escapeHtml(p.name)}">
            <pre class="reasoning-phase-text">${escapeHtml(content)}</pre>
          </div>
        ` : ''}
      </div>
    `;
  });

  html += '</div>';

  // Tool calls section
  if (toolCalls.length > 0) {
    const passCount = toolCalls.filter(tc => tc.success).length;
    const failCount = toolCalls.length - passCount;

    html += `
      <div class="reasoning-tools-section">
        <div class="reasoning-tools-header">
          <svg class="reasoning-tools-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M9.5 2.5L14 7l-4.5 4.5M6.5 2.5L2 7l4.5 4.5"/>
          </svg>
          <span class="reasoning-tools-label">${toolCalls.length} Tool Call${toolCalls.length > 1 ? 's' : ''}</span>
          <div class="reasoning-tools-stats">
            ${passCount > 0 ? `<span class="reasoning-tools-stat success">${passCount} passed</span>` : ''}
            ${failCount > 0 ? `<span class="reasoning-tools-stat error">${failCount} failed</span>` : ''}
          </div>
        </div>
        <div class="reasoning-tools-list">
    `;

    toolCalls.forEach((tc, idx) => {
      const statusClass = tc.success ? 'success' : 'error';
      const isExpanded = expandedToolCards.has(idx);
      const inputStr = tc.input ? JSON.stringify(tc.input) : '';
      const outputStr = tc.output || tc.error || '';
      const truncatedInput = inputStr.length > 120 ? inputStr.substring(0, 120) + '\u2026' : inputStr;
      const truncatedOutput = outputStr.length > 200 ? outputStr.substring(0, 200) + '\u2026' : outputStr;
      const hasLongContent = inputStr.length > 120 || outputStr.length > 200;

      // Tools that returned a structured card use the rich renderer; fall
      // back to the legacy input/output dump for plain-text tools.
      let bodyHtml;
      if (tc.card && tc.success) {
        bodyHtml = `<div class="reasoning-tool-card-host" data-card-idx="${idx}"></div>`;
      } else {
        bodyHtml = `
          ${inputStr ? `
            <div class="reasoning-tool-detail">
              <span class="reasoning-tool-detail-label">Input</span>
              <code class="reasoning-tool-detail-code">${escapeHtml(isExpanded ? inputStr : truncatedInput)}</code>
            </div>
          ` : ''}
          ${outputStr ? `
            <div class="reasoning-tool-detail">
              <span class="reasoning-tool-detail-label">${tc.success ? 'Output' : 'Error'}</span>
              <div class="reasoning-tool-detail-output ${statusClass}">${escapeHtml(isExpanded ? outputStr : truncatedOutput)}</div>
            </div>
          ` : ''}
          ${hasLongContent ? `
            <button class="reasoning-tool-view-full" onclick="event.stopPropagation(); window.__showToolDetail(${idx})">
              View Full Result
            </button>
          ` : ''}
        `;
      }

      html += `
        <div class="reasoning-tool-card ${statusClass}">
          <div class="reasoning-tool-header" onclick="window.__toggleToolExpand(${idx})">
            <span class="reasoning-tool-status-dot ${statusClass}"></span>
            <span class="reasoning-tool-name">${escapeHtml(tc.tool)}</span>
            <span class="reasoning-tool-phase-badge">${escapeHtml(tc.phase || '')}</span>
            <span class="reasoning-tool-chevron">${isExpanded ? icons.chevronDown : icons.chevronRightSmall}</span>
          </div>
          <div class="reasoning-tool-body ${isExpanded ? 'expanded' : 'collapsed'}">
            ${bodyHtml}
          </div>
        </div>
      `;
    });

    html += '</div></div>';
  }

  html += '</div>';
  container.innerHTML = html;

  // Wire phase-header clicks through a delegated handler that reads the
  // phase name from the parent block's `data-phase` attribute. Previously
  // this used inline `onclick="window.__togglePhaseExpand('${phaseName}')"`,
  // which is unsafe when `phaseName` is model output containing `'`:
  // HTML attribute parsing decodes character references BEFORE the event
  // handler JS parses, so `&#39;` becomes a literal `'` and breaks out
  // of the JS string. Reading from the data attribute is safe — the
  // browser doesn't re-parse data attribute values as code.
  container.querySelectorAll('.reasoning-phase-header.clickable').forEach((header) => {
    header.addEventListener('click', () => {
      const phaseName = header.closest('[data-phase]')?.dataset.phase;
      if (phaseName) window.__togglePhaseExpand(phaseName);
    });
  });

  // Hydrate ToolCard hosts for tool calls that returned a structured card.
  const hosts = container.querySelectorAll('.reasoning-tool-card-host[data-card-idx]');
  if (hosts.length) {
    import('./chat/tool-card.js').then(m => {
      hosts.forEach(host => {
        const idx = parseInt(host.dataset.cardIdx, 10);
        const tc = toolCalls[idx];
        if (!tc || !tc.card) return;
        const cardHtml = m.renderToolCard(tc.card);
        if (cardHtml) host.innerHTML = cardHtml;
      });
    }).catch(() => { /* fallback: hosts stay empty, legacy view still works */ });
  }
}

// ---------------------------------------------------------------------------
// Phase content delta handler
// ---------------------------------------------------------------------------

export function addPhaseContentDelta(phaseName, delta) {
  if (!phaseContent[phaseName]) phaseContent[phaseName] = '';
  phaseContent[phaseName] += delta;

  // Auto-expand the running phase and update its content box in-place
  // (avoids full re-render on every delta for performance)
  const contentBox = document.getElementById(`phase-content-${phaseName}`);
  if (contentBox) {
    // Auto-expand when content first arrives
    if (!expandedPhases.has(phaseName) && phaseContent[phaseName].length === delta.length) {
      expandedPhases.add(phaseName);
      contentBox.classList.remove('collapsed');
      contentBox.classList.add('expanded');
      // Update toggle arrow
      const block = contentBox.closest('.reasoning-phase-block');
      const toggle = block?.querySelector('.reasoning-phase-toggle');
      if (toggle) toggle.innerHTML = icons.chevronDown;
    }
    const pre = contentBox.querySelector('.reasoning-phase-text');
    if (pre) {
      pre.textContent = phaseContent[phaseName];
      contentBox.scrollTop = contentBox.scrollHeight;
    }
  } else {
    // Content box doesn't exist yet — trigger a re-render
    renderReasoningPhases(currentPhases, currentComplexity);
  }
}

// ---------------------------------------------------------------------------
// Tool call handler
// ---------------------------------------------------------------------------

export function addToolCall(toolCall) {
  toolCalls.push(toolCall);
  renderReasoningPhases(currentPhases, currentComplexity);
}

// ---------------------------------------------------------------------------
// Confidence update handler
// ---------------------------------------------------------------------------

export function updateConfidence(confidence) {
  if (confidence !== undefined && confidence !== null) {
    currentConfidence = confidence;
  }
}

export function getConfidence() {
  return currentConfidence;
}

// ---------------------------------------------------------------------------
// Restore from persisted message data (after refresh / session switch)
// ---------------------------------------------------------------------------

export function restoreReasoningFromStored(reasoning) {
  if (!reasoning || !reasoning.phases || reasoning.phases.length === 0) {
    resetReasoningState();
    return;
  }

  // Restore state from the persisted node.reasoning object
  currentPhases = reasoning.phases.map(p => ({ name: p.name, status: p.status || 'complete', ...(p.step_model ? { step_model: p.step_model } : {}) }));
  currentComplexity = reasoning.complexity || '';
  currentConfidence = reasoning.confidence ?? null;
  phaseContent = reasoning.phaseContent ? { ...reasoning.phaseContent } : {};
  toolCalls = (reasoning.toolCalls || []).map(tc => ({
    tool: tc.tool, phase: tc.phase, success: tc.success,
    input: tc.input, output: tc.output, error: tc.error,
  }));

  // Phases with content should start expanded
  expandedPhases = new Set();
  expandedToolCards = new Set();
  for (const [name, content] of Object.entries(phaseContent)) {
    if (content && content.length > 0) {
      expandedPhases.add(name);
    }
  }

  // Reconstruct approximate timings (all complete, no real timestamps)
  phaseTimings = {};

  // Render the restored state in the inspector panel
  renderReasoningPhases(currentPhases, currentComplexity);
}

// ---------------------------------------------------------------------------
// Update from streaming metadata (convenience)
// ---------------------------------------------------------------------------

export function updateReasoningFromMeta(meta) {
  if (meta.confidence !== undefined) {
    updateConfidence(meta.confidence);
  }
  if (!meta.phases) return;
  renderReasoningPhases(meta.phases, meta.complexity);
}

// ---------------------------------------------------------------------------
// View Switching (editor / live / streaming)
// ---------------------------------------------------------------------------

function switchView(view) {
  editorView = view;
  const editorEl = $('reasoning-editor-view');
  const liveEl = $('reasoning-live-view');
  const streamEl = $('reasoning-content');
  const title = $('reasoning-panel-title');

  if (editorEl) editorEl.classList.toggle('hidden', view !== 'editor');
  if (liveEl) liveEl.classList.toggle('hidden', view !== 'live');
  if (streamEl) streamEl.classList.toggle('hidden', view !== 'streaming');

  if (title) {
    title.textContent = view === 'editor' ? 'Reasoning Flows'
      : view === 'live' ? 'Live Execution'
      : 'Reasoning';
  }
}

function toggleView() {
  if (editorView === 'editor') {
    switchView('streaming');
  } else {
    switchView('editor');
  }
}

// ---------------------------------------------------------------------------
// Live Execution View
// ---------------------------------------------------------------------------

export function showLiveExecution(flowName, complexity, steps) {
  switchView('live');

  const nameEl = $('live-flow-name');
  const complexityEl = $('live-flow-complexity');
  const listEl = $('live-step-list');

  if (nameEl) nameEl.textContent = flowName || '';
  if (complexityEl) {
    complexityEl.textContent = complexity || '';
    complexityEl.className = `live-flow-complexity ${complexity || ''}`;
  }

  if (listEl && steps) {
    listEl.innerHTML = steps.map(s => `
      <div class="live-step ${s.status || 'pending'}">
        <span class="live-step-icon">${
          s.status === 'complete' ? icons.checkSmall
          : s.status === 'running' ? icons.dotFilled
          : icons.dotEmpty
        }</span>
        <span class="live-step-name">${escapeHtml(s.name)}</span>
      </div>
    `).join('');
  }
}

export function updateLiveStats(stats) {
  const el = $('live-stats');
  if (!el || !stats) return;
  const parts = [];
  if (stats.tokens) parts.push(`${stats.tokens} tokens`);
  if (stats.elapsed) parts.push(`${stats.elapsed}s`);
  if (stats.tools) parts.push(`${stats.tools} tool calls`);
  el.textContent = parts.join(' \u2022 ');
}

// Return to editor view when streaming completes
export function onStreamingComplete() {
  // Don't auto-switch — user can toggle manually
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

export function initAnalytical() {
  // View toggle button
  const toggleBtn = $('reasoning-view-toggle');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', toggleView);
  }

  // Flow editor (shared component)
  const containerEl = document.getElementById('reasoning-editor-view');
  if (containerEl) {
    const editor = new FlowEditor({
      containerEl,
      mode: 'analytical',
      roles: [
        { value: 'analyze', label: 'Analyze' },
        { value: 'classify', label: 'Classify' },
        { value: 'search', label: 'Search' },
        { value: 'verify', label: 'Verify' },
        { value: 'respond', label: 'Respond' },
      ],
      accentColor: 'var(--mode-analytical)',
    });
    editor.init();
  }
}
