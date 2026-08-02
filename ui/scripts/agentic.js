/* ==========================================================================
   Augmentum — Agentic Module
   Flow editor for agentic workflows + view switching between editor and
   task execution views.  Settings popover for theme, image model, autonomy.
   ========================================================================== */

import { app } from './app.js';
import { FlowEditor } from './flow-editor.js';
import { getImageModels, getToolSettings } from './model-cache.js';
import { dispatch as dispatchFlowMeta, loadBuiltins as loadFlowRenderers } from './agentic-panels/registry.js';

// Side-load per-flow renderers (storybook, future: app builder, document, …).
// Fire-and-forget — dispatchFlowMeta safely no-ops until the registry has
// loaded its builtins, so a slow import doesn't drop early events.
loadFlowRenderers().catch(err => console.warn('[agentic] flow renderers failed to load', err));

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AGENTIC_ROLES = [
  { value: 'plan', label: 'Plan' },
  { value: 'draft', label: 'Draft' },
  { value: 'create', label: 'Create' },
  { value: 'illustrate', label: 'Illustrate' },
  { value: 'review', label: 'Review' },
  { value: 'deliver', label: 'Deliver' },
  { value: 'transform', label: 'Transform' },
  { value: 'analyze', label: 'Analyze' },
  { value: 'search', label: 'Search' },
  { value: 'verify', label: 'Verify' },
];

const AUTONOMY_NAMES = ['', 'Suggest', 'Ask', 'Inform', 'Autonomous'];
const AUTONOMY_DESCS = [
  '',
  'Asks for approval before every step in the pipeline',
  'Pauses before high-impact actions like creating files',
  'Runs everything freely, notifies you about key actions',
  'Runs to completion silently — you see only the final result',
];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let editor = null;
let currentView = 'editor'; // 'editor' | 'task'

// ---------------------------------------------------------------------------
// View Switching
// ---------------------------------------------------------------------------

function switchAgenticView(view) {
  currentView = view;
  const editorEl = document.getElementById('agentic-editor-view');
  const taskEl = document.getElementById('task-content');
  const title = document.getElementById('agentic-panel-title');
  const flowsTab = document.getElementById('agentic-tab-flows');
  const taskTab = document.getElementById('agentic-tab-task');

  if (editorEl) editorEl.classList.toggle('hidden', view !== 'editor');
  if (taskEl) taskEl.classList.toggle('hidden', view !== 'task');

  // Segmented tab visual + ARIA state
  if (flowsTab) {
    const active = view === 'editor';
    flowsTab.classList.toggle('is-active', active);
    flowsTab.setAttribute('aria-selected', active ? 'true' : 'false');
  }
  if (taskTab) {
    const active = view === 'task';
    taskTab.classList.toggle('is-active', active);
    taskTab.setAttribute('aria-selected', active ? 'true' : 'false');
  }

  // Kept-in-sync legacy title (hidden by default; some code paths still write to it)
  if (title) {
    title.textContent = view === 'editor' ? 'Agentic Flows' : 'Task';
  }
}

/** Public hook: set the Task tab's running indicator. Called by chat.js / agentic.js
 *  task-meta renderer so the segmented control reflects task lifecycle. */
export function setAgenticTaskRunning(running) {
  const taskTab = document.getElementById('agentic-tab-task');
  if (taskTab) taskTab.classList.toggle('is-running', !!running);
}

/** Called by chat.js when an agentic task starts streaming.
 *
 *  Cheap to call repeatedly — only refreshes the task-history list once
 *  per session, then debounces. Without this guard, every streamed chunk
 *  would trigger a /api/agentic/tasks GET (we observed hundreds/min in
 *  the logs).
 */
let _historyLoadedForSession = '';
export function showAgenticTaskView() {
  switchAgenticView('task');
  const empty = document.getElementById('task-empty-state');
  if (empty) empty.style.display = 'none';
  // Only fetch history when entering the view for a NEW session.
  const sid = (app.state.currentSessionId || '');
  if (sid && sid !== _historyLoadedForSession) {
    _historyLoadedForSession = sid;
    loadAgenticTaskHistory().catch(() => { /* best effort */ });
  }
}

/** Optimistic first-paint used immediately after the user sends in Build mode.
 *
 *  The backend creates the durable task id a moment later and regular
 *  renderAgenticTaskMeta() takes over. Until then, clear any stale task
 *  chrome so the inspector visibly responds to the send action.
 */
export function showAgenticTaskStarting(prompt = '') {
  switchAgenticView('task');
  _activeTaskId = null;
  _latestTaskMeta = null;
  _lastPhases = [];
  _approvalPending = false;
  _stepOutputs = new Map();
  _expandedStepOutputs = new Set();
  _stepTimings = new Map();

  const title = String(prompt || '').trim();
  _setText('task-title-label', title ? title.slice(0, 80) : 'Starting build');
  const titleLabel = document.getElementById('task-title-label');
  if (titleLabel) titleLabel.dataset.status = 'planning';
  const bar = document.getElementById('task-progress-bar');
  if (bar) bar.style.width = '0%';
  _setText('task-progress-text', 'Starting...');
  _setVisible('task-progress-section', true);
  _setVisible('task-plan', false);
  _setVisible('task-pipeline', false);
  _setVisible('task-artifacts', false);
  _setVisible('task-activity', false);
  _clearApprovalSlot();

  const stepsEl = document.getElementById('task-pipeline-steps');
  if (stepsEl) stepsEl.innerHTML = '';
  const activityEl = document.getElementById('task-activity-list');
  if (activityEl) activityEl.innerHTML = '';
  const artifactEl = document.getElementById('task-artifact-list');
  if (artifactEl) artifactEl.innerHTML = '';
  const empty = document.getElementById('task-empty-state');
  if (empty) empty.style.display = 'none';
}

/** Called when switching to agentic mode to show editor by default. */
export function showAgenticEditorView() {
  switchAgenticView('editor');
}

// ---------------------------------------------------------------------------
// Live task render — driven by InternalStreamChunk.augmentum from the
// AgenticHandler. Backend emits {mode:"agentic", task_id, task_status,
// task_title, current_step, total_steps, progress, plan_md, autonomy_level}
// on every state transition. We just project that onto the existing DOM.
// ---------------------------------------------------------------------------

let _activeTaskId = null;
let _latestTaskMeta = null;

function _setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function _setVisible(id, visible) {
  const el = document.getElementById(id);
  if (el) el.style.display = visible ? '' : 'none';
}

/** Parse plan_md (markdown checklist) into an array of {done, text} items. */
function _parsePlanChecklist(planMd) {
  if (!planMd || typeof planMd !== 'string') return [];
  const items = [];
  for (const raw of planMd.split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    // - [x] item / - [ ] item / * item / 1. item / - item
    let m = line.match(/^[-*]\s*\[([ xX])\]\s*(.+)$/);
    if (m) { items.push({ done: m[1].toLowerCase() === 'x', text: m[2].trim() }); continue; }
    m = line.match(/^(?:[-*]|\d+\.)\s+(.+)$/);
    if (m) { items.push({ done: false, text: m[1].trim() }); continue; }
  }
  return items;
}

const _AUTONOMY_LABELS = ['', 'Suggest', 'Ask', 'Inform', 'Autonomous'];

// Keyed cache of the last known phases array for the active task so chain
// sub-step events (which only carry a chain_step payload, not a full phases
// update) can find the right pipeline-step DOM node to attach to.
let _lastPhases = [];
// Per-task approval state — remembers whether a pending card is showing,
// so task_status flipping back to running can retire it cleanly.
let _approvalPending = false;
// Per-step transparency state — Build mode already streams step outputs and
// status transitions, but we keep them here so the inspector can preserve
// output blocks across re-renders instead of only showing dots.
let _stepOutputs = new Map();
let _expandedStepOutputs = new Set();
let _stepTimings = new Map();
const _flowPhaseCache = new Map();

function _stepKey(name) {
  return typeof name === 'string' ? name.trim() : '';
}

function _statusLabel(status) {
  const s = (status || '').toLowerCase();
  if (s === 'complete' || s === 'completed') return 'Complete';
  if (s === 'running') return 'Running';
  if (s === 'warning') return 'Tool warning';
  if (s === 'failed' || s === 'error') return 'Failed';
  if (s === 'approval_pending') return 'Awaiting approval';
  if (s === 'paused') return 'Paused';
  return 'Pending';
}

function _previewText(text, max = 140) {
  const flat = String(text || '').replace(/\s+/g, ' ').trim();
  if (!flat) return '';
  return flat.length > max ? flat.slice(0, Math.max(0, max - 1)) + '…' : flat;
}

function _normalizeInspectorText(text) {
  let out = String(text || '').trim();
  if (!out) return '';
  out = out.replace(/^\[([^\]]+):\s*/, '');
  out = out.replace(/^\[([a-z0-9_]+)\s*\((ok|failed|error)\)\]:\s*/i, '');
  out = out.replace(/\]$/, '');
  return out.replace(/\s+/g, ' ').trim();
}

function _countLines(text) {
  const raw = String(text || '');
  if (!raw) return 0;
  return raw.split(/\r?\n/).length;
}

function _shouldSuppressStepOutput(stepEl, text) {
  if (!stepEl) return false;
  const raw = String(text || '').trim();
  if (!raw || raw.length > 280) return false;
  const substeps = Array.from(stepEl.querySelectorAll('.chain-substep'))
    .filter((row) => !row.classList.contains('running'));
  if (substeps.length !== 1) return false;
  const preview = substeps[0].querySelector('.chain-substep-preview')?.textContent || '';
  const rawNorm = _normalizeInspectorText(raw);
  const previewNorm = _normalizeInspectorText(preview);
  return !!rawNorm && rawNorm === previewNorm;
}

function _trackStepTimings(phases) {
  if (!Array.isArray(phases) || !phases.length) return;
  const now = Date.now();
  phases.forEach((phase) => {
    const key = _stepKey(phase && phase.name);
    if (!key) return;
    const status = String((phase && phase.status) || '').toLowerCase();
    let timing = _stepTimings.get(key);
    if (!timing) {
      timing = {};
      _stepTimings.set(key, timing);
    }
    if ((status === 'running' || status === 'warning') && !timing.start) {
      timing.start = now;
    }
    if ((status === 'complete' || status === 'failed' || status === 'error') && !timing.end) {
      if (!timing.start) timing.start = now;
      timing.end = now;
    }
  });
}

function _formatDuration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 10000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms / 1000)}s`;
}

function _stepDuration(name) {
  const key = _stepKey(name);
  if (!key) return '';
  const timing = _stepTimings.get(key);
  if (!timing || !timing.start) return '';
  const end = timing.end || Date.now();
  return _formatDuration(end - timing.start);
}

function _findPipelineStep(stepName) {
  const key = _stepKey(stepName);
  if (!key) return null;
  const stepsEl = document.getElementById('task-pipeline-steps');
  if (!stepsEl) return null;
  return Array.from(stepsEl.children).find((el) => {
    const rowKey = _stepKey(el.dataset.stepName || '');
    if (rowKey && rowKey === key) return true;
    const label = _stepKey(el.querySelector('.pipeline-step-label')?.textContent || '');
    return label === key;
  }) || null;
}

function _ensureStepOutputShell(stepEl, stepKeyName) {
  if (!stepEl) return null;
  const outputEl = stepEl.querySelector('.pipeline-step-output');
  if (!outputEl) return null;

  if (outputEl.dataset.bound !== '1') {
    outputEl.innerHTML = `
      <div class="pipeline-step-output-summary">
        <button type="button" class="pipeline-step-output-toggle">Output</button>
        <span class="pipeline-step-output-preview"></span>
        <span class="pipeline-step-output-meta"></span>
      </div>
      <div class="pipeline-step-output-body"></div>
    `;
    const toggle = outputEl.querySelector('.pipeline-step-output-toggle');
    if (toggle) {
      toggle.addEventListener('click', () => {
        const activeKey = _stepKey(outputEl.dataset.stepKey || '');
        if (!activeKey) return;
        if (_expandedStepOutputs.has(activeKey)) {
          _expandedStepOutputs.delete(activeKey);
        } else {
          _expandedStepOutputs.add(activeKey);
        }
        _renderStepOutput(stepEl, activeKey, _stepOutputs.get(activeKey) || '', stepEl.dataset.stepStatus || '');
      });
    }
    outputEl.dataset.bound = '1';
  }

  outputEl.dataset.stepKey = stepKeyName;
  return outputEl;
}

function _renderStepOutput(stepEl, stepName, text, status) {
  const key = _stepKey(stepName);
  if (!key) return;
  const outputEl = _ensureStepOutputShell(stepEl, key);
  if (!outputEl) return;

  const raw = typeof text === 'string' ? text : String(text || '');
  const hasText = raw.trim().length > 0;
  const suppress = _shouldSuppressStepOutput(stepEl, raw);
  outputEl.hidden = !hasText || suppress;
  if (!hasText || suppress) return;

  const toggle = outputEl.querySelector('.pipeline-step-output-toggle');
  const preview = outputEl.querySelector('.pipeline-step-output-preview');
  const meta = outputEl.querySelector('.pipeline-step-output-meta');
  const body = outputEl.querySelector('.pipeline-step-output-body');
  const isExpanded = _expandedStepOutputs.has(key);
  const lineCount = _countLines(raw);
  const previewText = _previewText(raw);
  const metaParts = [];
  if (status) metaParts.push(_statusLabel(status));
  if (lineCount > 0) metaParts.push(`${lineCount} line${lineCount === 1 ? '' : 's'}`);

  outputEl.dataset.status = status || '';
  outputEl.classList.toggle('expanded', isExpanded);

  if (toggle) {
    toggle.textContent = isExpanded
      ? 'Hide output'
      : ((status === 'running' || status === 'warning') ? 'Live output' : 'Show output');
    toggle.setAttribute('aria-expanded', isExpanded ? 'true' : 'false');
  }
  if (preview) {
    preview.textContent = previewText;
    preview.hidden = isExpanded || !previewText;
    preview.title = !isExpanded ? raw : '';
  }
  if (meta) meta.textContent = metaParts.join(' · ');
  if (body) {
    body.textContent = raw;
    if (isExpanded) body.scrollTop = body.scrollHeight;
  }
}

function _syncStepOutput(stepEl) {
  if (!stepEl) return;
  const key = _stepKey(stepEl.dataset.stepName || '');
  if (!key) return;
  _renderStepOutput(stepEl, key, _stepOutputs.get(key) || '', stepEl.dataset.stepStatus || '');
}

function _appendStepOutput(stepName, delta) {
  const key = _stepKey(stepName);
  if (!key || !delta) return;
  const prev = _stepOutputs.get(key) || '';
  const next = prev + delta;
  _stepOutputs.set(key, next);
  if (!prev) _expandedStepOutputs.add(key);
  const stepEl = _findPipelineStep(key);
  if (stepEl) {
    _renderStepOutput(stepEl, key, next, stepEl.dataset.stepStatus || 'running');
  }
}

async function _loadTaskPhases(task) {
  if (!task || typeof task !== 'object') return [];

  let phases = [];
  const flowId = _stepKey(task.flow_id || '');
  if (flowId) {
    if (!_flowPhaseCache.has(flowId)) {
      _flowPhaseCache.set(flowId, (async () => {
        const resp = await fetch(`/api/reasoning/flows/${encodeURIComponent(flowId)}`);
        if (!resp.ok) return [];
        const flow = await resp.json();
        return Array.isArray(flow.steps)
          ? flow.steps
            .map((step, index) => ({
              name: step.name || `Step ${index + 1}`,
              role: step.role || '',
              tools: Array.isArray(step.tool_names) ? step.tool_names.filter(Boolean) : [],
              _sourceIndex: index,
              enabled: step.enabled !== false,
            }))
            .filter((step) => step.enabled)
          : [];
      })().catch(() => []));
    }
    phases = await _flowPhaseCache.get(flowId);
  }

  if (!Array.isArray(phases) || !phases.length) {
    const checklist = _parsePlanChecklist(task.plan_md || '');
    phases = checklist.map((item, index) => ({
      name: item.text || `Step ${index + 1}`,
      _sourceIndex: index,
    }));
  }

  if (!Array.isArray(phases) || !phases.length) {
    const total = Number(task.total_steps) || 0;
    phases = Array.from({ length: total }, (_, index) => ({
      name: `Step ${index + 1}`,
      _sourceIndex: index,
    }));
  }

  const status = String(task.status || '').toLowerCase();
  const activeRawIndex = Number(task.current_step);
  const completedRaw = new Set(
    Object.keys(task.step_outputs || {})
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value)),
  );

  return phases.map((phase, index) => {
    const sourceIndex = Number.isFinite(phase._sourceIndex) ? phase._sourceIndex : index;
    let phaseStatus = 'pending';
    if (status === 'completed') {
      phaseStatus = 'complete';
    } else if (completedRaw.has(sourceIndex)) {
      phaseStatus = 'complete';
    } else if ((status === 'failed' || status === 'error') && sourceIndex === activeRawIndex) {
      phaseStatus = 'failed';
    } else if ((status === 'running' || status === 'approval_pending' || status === 'paused') && sourceIndex === activeRawIndex) {
      phaseStatus = 'running';
    }
    return {
      name: phase.name,
      role: phase.role || '',
      tools: Array.isArray(phase.tools) ? phase.tools : [],
      status: phaseStatus,
      _sourceIndex: sourceIndex,
    };
  });
}

/** Project a single agentic stream meta payload onto the inspector DOM. */
export function renderAgenticTaskMeta(meta) {
  if (!meta || meta.mode !== 'agentic') return;
  const taskId = meta.task_id || _activeTaskId;
  if (!taskId) return;
  const isNewTask = taskId !== _activeTaskId;
  _activeTaskId = taskId;
  _latestTaskMeta = {
    ...(_latestTaskMeta || {}),
    ...meta,
    task_id: taskId,
  };

  if (isNewTask) {
    // Reset transient lists so a fresh task doesn't show stale steps.
    const stepsEl = document.getElementById('task-pipeline-steps');
    if (stepsEl) stepsEl.innerHTML = '';
    const activityEl = document.getElementById('task-activity-list');
    if (activityEl) activityEl.innerHTML = '';
    _setVisible('task-activity', false);
    _clearApprovalSlot();
    _lastPhases = [];
    _approvalPending = false;
    _stepOutputs = new Map();
    _expandedStepOutputs = new Set();
    _stepTimings = new Map();
  }

  // Route structured sub-events first — these don't affect the title/progress
  // render path, but they update ancillary sections of the inspector.
  if (meta.chain_step) {
    _renderChainSubstep(meta.chain_step);
  }
  if (meta.chain_replan) {
    _renderChainReplan(meta.chain_replan);
  }
  if (meta.informed_action) {
    _renderActivityItem(meta.informed_action);
  }
  if (meta.phase_content_delta && meta.phase) {
    _appendStepOutput(meta.phase, meta.phase_content_delta);
  }

  // Approval surfacing — the backend emits approval_request for both the
  // plan-level pause (autonomy=1) and mid-flow step pauses. The distinction
  // is carried on the request itself (step_role === 'plan' means the
  // card should show the full plan body, otherwise show step description).
  const hasStatus = typeof meta.task_status === 'string' && meta.task_status;
  const statusLower = hasStatus ? meta.task_status.toLowerCase() : '';
  if (meta.approval_request) {
    const req = meta.approval_request;
    const isPlanApproval = (req.step_role === 'plan') || (req.step_name === 'Plan Approval');
    _renderApprovalCard({
      kind: isPlanApproval ? 'plan' : 'step',
      request: req,
      plan_md: meta.plan_md || '',
      task_title: meta.task_title || '',
    });
    _approvalPending = true;
  } else if (hasStatus && statusLower !== 'approval_pending') {
    // Task moved on — dismiss the card if it's still in the slot. Checked
    // unconditionally (rather than gated on the prior _approvalPending
    // flag) so the resolved card from the previous turn gets cleared
    // even when the user clicked Approve before the next meta arrived.
    _clearApprovalSlot();
    _approvalPending = false;
  }

  // Title — only update when the meta carries a fresh value. Bare content
  // chunks (just {mode, task_id}) leave it alone.
  if (typeof meta.task_title === 'string' && meta.task_title) {
    _setText('task-title-label', meta.task_title);
  }

  // Progress — guard on the presence of progress/total fields so a bare
  // content chunk can't reset the bar to 0%. The envelope helper always
  // includes these for state-bearing chunks. ``total`` / ``cur`` / ``pct``
  // are also consumed by the pipeline fallback below, so they stay
  // function-scoped with safe defaults for the no-update path.
  const hasProgressFields = (
    'progress' in meta || 'total_steps' in meta || 'current_step' in meta
  );
  let total = 0;
  let cur = 0;
  let pct = 0;
  if (hasProgressFields) {
    total = Number(meta.total_steps) || 0;
    cur = Number(meta.current_step) || 0;
    pct = Number(meta.progress);
    if (!Number.isFinite(pct)) pct = total > 0 ? Math.round((cur / total) * 100) : 0;
    pct = Math.max(0, Math.min(100, Math.round(pct)));
    const bar = document.getElementById('task-progress-bar');
    if (bar) bar.style.width = pct + '%';
    _setText('task-progress-text',
      total > 0 ? `${cur}/${total} · ${pct}%` : `${pct}%`);
    _setVisible('task-progress-section', true);
  }

  // Status pill (lives on the progress row label) — same guard. A
  // missing task_status means "no change", not "clear the pill".
  // ``status`` is also read by the pipeline-fallback render below.
  const status = statusLower;
  if (hasStatus) {
    const titleLabel = document.getElementById('task-title-label');
    if (titleLabel) {
      titleLabel.dataset.status = statusLower;
    }
    // Tab indicator: running/planning pulse on the Task tab so the
    // segmented control shows liveness even while the user is on Flows.
    const tab = document.getElementById('agentic-tab-task');
    if (tab) {
      const live = status === 'running' || status === 'planning' || status === 'approval_pending';
      tab.classList.toggle('is-running', live);
    }
  }

  // Plan checklist — parse plan_md whenever a fresh one arrives. State per
  // item is then derived from the same `phases` array used by the pipeline
  // so the two stay perfectly in sync.
  if (typeof meta.plan_md === 'string' && meta.plan_md) {
    const items = _parsePlanChecklist(meta.plan_md);
    const planEl = document.getElementById('task-plan-content');
    if (planEl && items.length) {
      const phasesArr = Array.isArray(meta.phases) ? meta.phases : null;
      planEl.innerHTML = items.map((it, i) => {
        let done, active;
        if (phasesArr) {
          const ps = phasesArr[i] && phasesArr[i].status;
          done = ps === 'complete' || it.done;
          active = ps === 'running' && !done;
        } else {
          const checkedThrough = cur > 0 ? Math.min(cur, items.length) : 0;
          done = it.done || i < checkedThrough;
          active = !done && i === checkedThrough;
        }
        const cls = ['plan-step'];
        if (done) cls.push('done');
        if (active) cls.push('active');
        const mark = done ? '\u2713' : (active ? '\u25B6' : '\u25CB');
        return `<div class="${cls.join(' ')}"><span class="plan-step-check">${mark}</span><span class="plan-step-text"></span></div>`;
      }).join('');
      const textNodes = planEl.querySelectorAll('.plan-step-text');
      items.forEach((it, i) => { if (textNodes[i]) textNodes[i].textContent = it.text; });
      _setVisible('task-plan', true);
    }
  } else if (Array.isArray(meta.phases) && meta.phases.length) {
    // No new plan_md, but phases changed — refresh the existing plan-item
    // marks so the checklist tracks step transitions in real time.
    const planEl = document.getElementById('task-plan-content');
    if (planEl) {
      const phasesArr = meta.phases;
      const planItems = planEl.querySelectorAll('.plan-step');
      planItems.forEach((el, i) => {
        const ps = phasesArr[i] && phasesArr[i].status;
        const done = ps === 'complete' || el.classList.contains('done');
        const active = ps === 'running' && !done;
        el.classList.toggle('done', done);
        el.classList.toggle('active', active);
        const mark = el.querySelector('.plan-step-check');
        if (mark) mark.textContent = done ? '\u2713' : (active ? '\u25B6' : '\u25CB');
      });
    }
  }

  // Pipeline step list — prefer the explicit `phases` array from the
  // backend (each entry: {name, status: "pending"|"running"|"complete"})
  // because it's the single source of truth per step. Fall back to
  // current_step arithmetic only when phases are absent (e.g. the
  // chain-planner path that hasn't been migrated yet).
  const stepsEl = document.getElementById('task-pipeline-steps');
  const phases = Array.isArray(meta.phases) ? meta.phases : null;
  const haveSteps = (phases && phases.length) || total > 0;
  if (stepsEl && haveSteps) {
    if (phases) {
      _lastPhases = phases;
      _trackStepTimings(phases);
    }
    const phaseNames = phases ? phases.map(p => (p && p.name) || '') : [];
    const want = phases ? phases.length : total;
    // Rebuild dots when the count OR labels change so a new task with
    // different step names doesn't show stale text.
    const currentLabels = Array.from(stepsEl.querySelectorAll('.pipeline-step-label'))
      .map(el => el.textContent || '');
    const labelsChanged = phaseNames.length === stepsEl.children.length
      && phaseNames.some((n, i) => n && n !== currentLabels[i]);
    if (stepsEl.children.length !== want || labelsChanged) {
      stepsEl.innerHTML = Array.from({ length: want }, (_, i) => {
        // Each step row carries two layers: a main line (number + name +
        // role chip + tool count) and two hidden slots — a substeps
        // container for live chain events, and an output slot for
        // resume-hydrated step outputs. Keeping them inline preserves
        // the single-column inspector rhythm.
        return `<div class="pipeline-step" data-idx="${i + 1}">
          <div class="pipeline-step-main">
            <span class="pipeline-step-num">${String(i + 1).padStart(2, '0')}</span>
            <div class="pipeline-step-copy">
              <span class="pipeline-step-label"></span>
              <span class="pipeline-step-state"></span>
            </div>
          </div>
          <div class="pipeline-step-meta"></div>
          <div class="pipeline-step-tooltip" hidden></div>
          <div class="pipeline-step-substeps" hidden></div>
          <div class="pipeline-step-output" hidden></div>
        </div>`;
      }).join('');
    }
    // Refresh labels, role chips, and tooltip contents on every update —
    // phases may swap role/tools between events (e.g. mid-flow re-planning
    // inserts a new step) and we want the UI to reflect that.
    Array.from(stepsEl.children).forEach((el, i) => {
      const phase = phases ? phases[i] : null;
      const label = phaseNames[i] || '';
      const labelEl = el.querySelector('.pipeline-step-label');
      if (labelEl) labelEl.textContent = label;
      el.dataset.stepName = label;
      el.dataset.stepStatus = String((phase && phase.status) || '').toLowerCase();
      el.dataset.sourceIndex = phase && Number.isFinite(phase._sourceIndex) ? String(phase._sourceIndex) : String(i);
      _applyStepMeta(el, phase);
      _syncStepOutput(el);
    });
    if (phases) {
      // Phase array → per-step state directly.
      Array.from(stepsEl.children).forEach((el, i) => {
        const ps = String((phases[i] && phases[i].status) || '').toLowerCase();
        el.dataset.stepStatus = ps;
        el.classList.toggle('complete', ps === 'complete');
        el.classList.toggle('running', ps === 'running');
        el.classList.toggle('active', ps === 'running' || ps === 'warning');
        el.classList.toggle('failed', ps === 'failed' || ps === 'error');
        el.classList.toggle('warning', ps === 'warning');
        // When a step leaves the running state its live substeps log is
        // no longer interesting — collapse it to keep the inspector tidy.
        if (ps === 'complete' || ps === 'failed' || ps === 'error') {
          const sub = el.querySelector('.pipeline-step-substeps');
          if (sub && !sub.hasChildNodes()) sub.hidden = true;
        }
        _applyStepMeta(el, phases[i] || null);
        _syncStepOutput(el);
      });
    } else {
      // current_step fallback. ``status`` is the *task* status here
      // (running / completed / failed), used to decorate the index that
      // matches current_step.
      Array.from(stepsEl.children).forEach((el) => {
        const idx = Number(el.dataset.idx);
        const isComplete = idx < cur;
        const isRunning = idx === cur && (status === 'running' || status === '');
        const isFailed = idx === cur && status === 'failed';
        el.dataset.stepStatus = isFailed ? 'failed' : (isRunning ? 'running' : (isComplete ? 'complete' : 'pending'));
        el.classList.toggle('complete', isComplete);
        el.classList.toggle('running', isRunning);
        el.classList.toggle('active', isRunning);
        el.classList.toggle('failed', isFailed);
        el.classList.remove('warning');
        _applyStepMeta(el, null);
        _syncStepOutput(el);
      });
    }
    // Whole-task completion → mark every dot done.
    if (status === 'completed' || pct >= 100) {
      Array.from(stepsEl.children).forEach((el) => {
        el.classList.add('complete');
        el.classList.remove('running');
        el.classList.remove('active');
        el.classList.remove('failed');
        el.classList.remove('warning');
        el.dataset.stepStatus = 'complete';
        _applyStepMeta(el, phases ? phases[Number(el.dataset.idx) - 1] || null : null);
        _syncStepOutput(el);
      });
    }
    _setVisible('task-pipeline', true);
  }

  if (hasProgressFields) {
    const textEl = document.getElementById('task-progress-text');
    if (textEl) {
      const phaseList = phases || _lastPhases || [];
      const runningIdx = Array.isArray(phaseList)
        ? phaseList.findIndex((p) => ['running', 'warning', 'failed', 'error'].includes(String((p && p.status) || '').toLowerCase()))
        : -1;
      const completedCount = Array.isArray(phaseList)
        ? phaseList.filter((p) => String((p && p.status) || '').toLowerCase() === 'complete').length
        : 0;
      const activeName = runningIdx >= 0 ? _stepKey(phaseList[runningIdx]?.name || '') : '';
      if (status === 'completed' && Array.isArray(phaseList) && phaseList.length) {
        textEl.textContent = `${phaseList.length}/${phaseList.length} · All steps complete · ${pct}%`;
      } else if (status === 'approval_pending') {
        textEl.textContent = activeName
          ? `Waiting for approval · ${activeName}`
          : `Waiting for approval · ${pct}%`;
      } else if (status === 'failed') {
        textEl.textContent = activeName
          ? `Stopped at ${activeName} · ${pct}%`
          : `Task failed · ${pct}%`;
      } else if (activeName && Array.isArray(phaseList) && phaseList.length) {
        const activeCount = runningIdx >= 0 ? (runningIdx + 1) : Math.max(1, completedCount);
        textEl.textContent = `${activeCount}/${phaseList.length} · ${activeName} · ${pct}%`;
      }
    }
  }

  // Autonomy level (purely informational — control lives elsewhere)
  const lvl = Number(meta.autonomy_level);
  if (Number.isFinite(lvl) && lvl >= 1 && lvl <= 4) {
    _setText('autonomy-label', _AUTONOMY_LABELS[lvl] || '');
    const dots = document.querySelectorAll('#autonomy-dots .autonomy-dot');
    dots.forEach((d, i) => d.classList.toggle('active', i < lvl));
  }

  // Hand off to a per-flow renderer if one is registered for meta.flow_id.
  // The renderer owns the ``#task-flow-body`` slot below the pipeline and
  // adds tailored UI (chapter strip / file tree / etc.). Fall-through is
  // safe: when no renderer matches, the slot stays empty and the user sees
  // the standard pipeline + artifacts only.
  dispatchFlowMeta(meta);

  // Hide the build-mode placeholder once any real progress exists.
  _setVisible('task-empty-state', false);
  _emitTaskSnapshot();
}

// ---------------------------------------------------------------------------
// Pipeline-step detail rendering — role chip + tools tooltip
// ---------------------------------------------------------------------------

/** Paint role chip, tool count, and hover tooltip onto a pipeline-step row.
 *
 *  Called from renderAgenticTaskMeta on every meta event. Tolerates phase
 *  objects from the legacy "name + status only" shape (older streams or
 *  the ad-hoc chain-planner path) by skipping fields it doesn't find.
 */
function _applyStepMeta(stepEl, phase) {
  if (!stepEl) return;
  const metaEl = stepEl.querySelector('.pipeline-step-meta');
  const tipEl = stepEl.querySelector('.pipeline-step-tooltip');
  const stateEl = stepEl.querySelector('.pipeline-step-state');
  if (!metaEl || !tipEl) return;

  const role = (phase && typeof phase.role === 'string') ? phase.role.trim() : '';
  const tools = (phase && Array.isArray(phase.tools)) ? phase.tools.filter(Boolean) : [];
  const status = String((phase && phase.status) || stepEl.dataset.stepStatus || '').toLowerCase();
  const stepName = _stepKey((phase && phase.name) || stepEl.dataset.stepName || '');
  const duration = stepName ? _stepDuration(stepName) : '';

  if (stateEl) {
    stateEl.textContent = _statusLabel(status);
  }

  // Meta line: role chip + optional tool count. Cleared rather than
  // hidden so a step that drops its role (unusual) updates correctly.
  metaEl.innerHTML = '';
  if (role) {
    const chip = document.createElement('span');
    chip.className = 'pipeline-step-role';
    chip.textContent = role;
    metaEl.appendChild(chip);
  }
  if (tools.length) {
    const count = document.createElement('span');
    count.className = 'pipeline-step-tool-count';
    count.textContent = tools.length === 1 ? '1 tool' : `${tools.length} tools`;
    metaEl.appendChild(count);
  }
  if (duration) {
    const time = document.createElement('span');
    time.className = 'pipeline-step-tool-count';
    time.textContent = duration;
    metaEl.appendChild(time);
  }

  // Tooltip body: tool list. Hidden when there's nothing to show so the
  // :hover rule doesn't flash an empty popover.
  tipEl.innerHTML = '';
  if (tools.length) {
    const label = document.createElement('span');
    label.className = 'pipeline-step-tooltip-label';
    label.textContent = 'Tools';
    tipEl.appendChild(label);
    tipEl.appendChild(document.createTextNode(tools.join(', ')));
    tipEl.hidden = false;
  } else {
    tipEl.hidden = true;
  }
}

// ---------------------------------------------------------------------------
// Chain-step sub-events — per-tool rows under the active pipeline step
// ---------------------------------------------------------------------------

/** Attach (or update) a single chain sub-step row under the active step. */
function _renderChainSubstep(ev) {
  if (!ev || !ev.tool) return;
  const stepsEl = document.getElementById('task-pipeline-steps');
  if (!stepsEl) return;
  // Find the active (running) pipeline step — chain_step events are only
  // meaningful while their parent flow-step is executing.
  const activeStep = stepsEl.querySelector('.pipeline-step.active')
                    || stepsEl.querySelector('.pipeline-step.running');
  if (!activeStep) return;

  const substeps = activeStep.querySelector('.pipeline-step-substeps');
  if (!substeps) return;
  substeps.hidden = false;

  const key = ev.id != null ? `sub-${ev.id}` : `sub-${ev.tool}-${substeps.children.length}`;
  let row = substeps.querySelector(`[data-sub-id="${CSS.escape(key)}"]`);
  if (!row) {
    row = document.createElement('div');
    row.className = 'chain-substep';
    row.dataset.subId = key;
    row.innerHTML = `
      <span class="chain-substep-status"></span>
      <span class="chain-substep-tool"></span>
      <span class="chain-substep-preview"></span>
    `;
    substeps.appendChild(row);
  }

  const status = (ev.status || 'running').toLowerCase();
  row.classList.toggle('running', status === 'running');
  row.classList.toggle('done', status === 'done' || status === 'complete');
  row.classList.toggle('failed', status === 'failed' || status === 'error');

  const toolEl = row.querySelector('.chain-substep-tool');
  if (toolEl) toolEl.textContent = ev.tool;
  const previewEl = row.querySelector('.chain-substep-preview');
  if (previewEl) {
    // Prefer `preview` (from on_step_done) over `reason` (from on_step_start)
    // so the final row shows actual output rather than the planning rationale.
    const text = ev.preview || ev.reason || '';
    previewEl.textContent = text;
    previewEl.title = text; // full text on hover for truncated previews
  }
}

/** Render a replan decision as a compact notice inside the active step. */
function _renderChainReplan(ev) {
  if (!ev || !ev.decision) return;
  const stepsEl = document.getElementById('task-pipeline-steps');
  if (!stepsEl) return;
  const activeStep = stepsEl.querySelector('.pipeline-step.active')
                    || stepsEl.querySelector('.pipeline-step.running');
  if (!activeStep) return;
  const substeps = activeStep.querySelector('.pipeline-step-substeps');
  if (!substeps) return;
  substeps.hidden = false;

  const row = document.createElement('div');
  row.className = 'chain-substep running';
  row.innerHTML = `
    <span class="chain-substep-status"></span>
    <span class="chain-substep-tool">replan</span>
    <span class="chain-substep-preview"></span>
  `;
  row.querySelector('.chain-substep-preview').textContent = ev.decision;
  substeps.appendChild(row);
}

// ---------------------------------------------------------------------------
// Activity log — inform events from autonomy 3/4 where the agent reports
// actions it took without pausing for approval.
// ---------------------------------------------------------------------------

const _INFORM_ICON = `
  <svg class="task-activity-icon" viewBox="0 0 24 24" fill="none"
       stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="20 6 9 17 4 12"></polyline>
  </svg>`;

function _renderActivityItem(action) {
  if (!action || typeof action !== 'object') return;
  const stepName = action.step_name || '';
  const actionText = action.action || '';
  if (!stepName && !actionText) return;

  const list = document.getElementById('task-activity-list');
  if (!list) return;
  _setVisible('task-activity', true);

  const item = document.createElement('div');
  item.className = 'task-activity-item';
  item.innerHTML = `
    ${_INFORM_ICON}
    <div class="task-activity-body">
      <span class="task-activity-step"></span><span class="task-activity-text"></span>
    </div>
  `;
  item.querySelector('.task-activity-step').textContent = stepName ? `${stepName}:` : '';
  item.querySelector('.task-activity-text').textContent = ' ' + actionText;

  // Newest at top, cap at 50 entries to keep the list bounded.
  list.insertBefore(item, list.firstChild);
  while (list.children.length > 50) list.removeChild(list.lastChild);
}

// ---------------------------------------------------------------------------
// Approval card — one-click Approve / Skip / Modify in the inspector
// ---------------------------------------------------------------------------

const _APPROVAL_ICON = `
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
       stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 9v4"></path>
    <path d="M12 17h.01"></path>
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
  </svg>`;

function _clearApprovalSlot() {
  const slot = document.getElementById('task-approval-slot');
  if (!slot) return;
  slot.innerHTML = '';
  _setVisible('task-approval-slot', false);
}

/** Render the approval card into the inspector's approval slot.
 *
 *  Button clicks resubmit the matching keyword through the normal chat
 *  submit path so the backend's _handle_approval_response stays the single
 *  source of truth for approval semantics — no special backend endpoint.
 */
function _renderApprovalCard({ kind, request, plan_md, task_title }) {
  const slot = document.getElementById('task-approval-slot');
  if (!slot) return;

  const stepName = (request && request.step_name) || 'Approval';
  const description = (request && request.description) || '';
  const stepRole = (request && request.step_role) || '';
  const isPlan = kind === 'plan';

  // Mark the pending step in the pipeline so the user sees WHICH step is
  // waiting, even if they scroll past the approval card.
  const stepsEl = document.getElementById('task-pipeline-steps');
  if (stepsEl) {
    Array.from(stepsEl.children).forEach(el => el.classList.remove('awaiting-approval'));
    const targetLabel = Array.from(stepsEl.querySelectorAll('.pipeline-step-label'))
      .find(lbl => lbl.textContent === stepName);
    if (targetLabel && targetLabel.closest('.pipeline-step')) {
      targetLabel.closest('.pipeline-step').classList.add('awaiting-approval');
    }
  }

  slot.innerHTML = '';
  _setVisible('task-approval-slot', true);

  const card = document.createElement('div');
  card.className = 'approval-card' + (isPlan ? ' approval-card-plan' : '');
  card.innerHTML = `
    <div class="approval-card-header">
      ${_APPROVAL_ICON}
      <span class="approval-card-step-name"></span>
    </div>
    <div class="approval-card-body"></div>
    ${isPlan && plan_md ? `<div class="approval-card-detail approval-card-plan-body"></div>` : ''}
    ${stepRole && !isPlan ? `<div class="approval-card-detail">Role: <span class="approval-card-role"></span></div>` : ''}
    <div class="approval-card-modify">
      <textarea placeholder="Describe what to change..." aria-label="Plan modification"></textarea>
    </div>
    <div class="approval-card-actions">
      <button class="approval-btn primary" data-action="approve">Approve</button>
      <button class="approval-btn" data-action="modify">Modify</button>
      <button class="approval-btn danger" data-action="cancel">${isPlan ? 'Cancel' : 'Skip'}</button>
    </div>
  `;

  card.querySelector('.approval-card-step-name').textContent =
    isPlan ? `Plan ready — ${task_title || 'review before running'}` : `Approval needed — ${stepName}`;
  card.querySelector('.approval-card-body').textContent =
    description || (isPlan ? 'The planner has drafted a sequence of steps. Approve to start, modify to adjust, or cancel to abort.' : '');
  if (isPlan && plan_md) {
    // Render plan_md as plaintext (safe) to keep any unknown markdown
    // renderer out of the trust boundary. The chat bubble still shows
    // the formatted version for rich context.
    card.querySelector('.approval-card-plan-body').textContent = plan_md;
  }
  const roleEl = card.querySelector('.approval-card-role');
  if (roleEl) roleEl.textContent = stepRole;

  // ---- Interactions -----------------------------------------------------
  const textarea = card.querySelector('textarea');

  const submit = (text) => {
    if (!text) return;
    card.classList.add('resolved');
    _approvalPending = false;
    // Dispatch the same event the chat composer uses so normal message
    // flow (persistence, streaming, middleware) still runs.
    document.dispatchEvent(new CustomEvent('augmentum:send', {
      detail: { text },
    }));
    // Slot is left in place with `.resolved` until the next meta update
    // clears it, so the user sees confirmation that their choice landed.
  };

  card.querySelector('[data-action="approve"]').addEventListener('click', () => {
    submit('approve');
  });
  card.querySelector('[data-action="cancel"]').addEventListener('click', () => {
    submit(isPlan ? 'cancel' : 'skip');
  });
  const modifyBtn = card.querySelector('[data-action="modify"]');
  modifyBtn.addEventListener('click', () => {
    if (!card.classList.contains('modifying')) {
      card.classList.add('modifying');
      modifyBtn.textContent = 'Submit';
      if (textarea) textarea.focus();
      return;
    }
    const text = textarea ? textarea.value.trim() : '';
    if (text) submit(text);
  });
  // Enter-to-submit from the textarea (matches composer behaviour).
  if (textarea) {
    textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const text = textarea.value.trim();
        if (text) submit(text);
      }
    });
  }

  slot.appendChild(card);
}

function _bytes(n) {
  if (!n || n < 0) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Render an artifact card emitted by the deliver step into the artifacts list.
 *
 *  Accepts both the ToolCard envelope shape ({artifact_id, title, kind, …})
 *  and the agentic deliver payload shape ({id, name, format, download_url,
 *  size_bytes, …}). Either is fine — _firstStr keys the lookup across both
 *  vocabularies so we don't care which side updates first.
 */
const _firstStr = (...vals) => {
  for (const v of vals) {
    if (typeof v === 'string' && v) return v;
  }
  return '';
};

export function renderAgenticArtifactCard(payload) {
  if (!payload) return;
  const list = document.getElementById('task-artifact-list');
  if (!list) return;

  const id = _firstStr(payload.id, payload.artifact_id);
  const title = _firstStr(payload.title, payload.display_name, payload.name) || 'Artifact';
  const kind = _firstStr(payload.kind, payload.format, payload.page_type);
  const downloadUrl = _firstStr(
    payload.download_url,
    id ? `/api/artifacts/${encodeURIComponent(id)}/download` : '',
  );
  const sizeStr = _bytes(Number(payload.size_bytes) || 0);

  // De-dup: if a card with this id is already in the list, replace it
  // (handles deliver step re-runs without piling up duplicates).
  if (id) {
    const existing = list.querySelector(`.task-artifact-card[data-artifact-id="${CSS.escape(id)}"]`);
    if (existing) existing.remove();
  }

  const card = document.createElement('div');
  card.className = 'task-artifact-card';
  // Intermediate visuals (slide images etc.) are build inputs, not
  // deliverables — render them compactly so the deliverable(s) stand out.
  if (payload.intermediate) card.dataset.intermediate = '1';
  if (id) card.dataset.artifactId = id;
  card.innerHTML = `
    <div class="task-artifact-title"></div>
    <div class="task-artifact-meta"></div>
    <div class="task-artifact-actions"></div>
  `;
  card.querySelector('.task-artifact-title').textContent = title;
  const meta = [kind, sizeStr].filter(Boolean).join(' · ');
  card.querySelector('.task-artifact-meta').textContent = meta;
  const actions = card.querySelector('.task-artifact-actions');

  if (id) {
    const overview = document.createElement('button');
    overview.className = 'task-artifact-btn';
    overview.type = 'button';
    overview.textContent = 'Overview';
    overview.addEventListener('click', () => {
      document.dispatchEvent(new CustomEvent('artifact:preview', {
        detail: { artifact_id: id },
      }));
    });
    actions.appendChild(overview);
    if (!payload.intermediate) {
      const edit = document.createElement('button');
      edit.className = 'task-artifact-btn';
      edit.type = 'button';
      edit.textContent = 'Edit';
      edit.addEventListener('click', () => {
        document.dispatchEvent(new CustomEvent('artifact:edit', {
          detail: { artifact_id: id },
        }));
      });
      actions.appendChild(edit);
    }
  }
  if (downloadUrl) {
    const dl = document.createElement('a');
    dl.className = 'task-artifact-btn';
    dl.href = downloadUrl;
    dl.textContent = 'Download';
    dl.setAttribute('download', '');
    actions.appendChild(dl);
  }

  list.appendChild(card);
  _setVisible('task-artifacts', true);
  _emitTaskSnapshot();
}

/** Fetch and render the recent task list for the active session.
 *
 *  When there's at least one prior task for this session, also auto-restore
 *  the MOST RECENT task into the live render (plan, progress, artifacts) so
 *  the panel survives page reload AND server restart. The live render is
 *  then immediately overwritten if the user is mid-task and a fresh
 *  agentic stream chunk arrives.
 */
export async function loadAgenticTaskHistory() {
  const list = document.getElementById('task-history-list');
  if (!list) return;
  const sessionId = app.state.currentSessionId || '';
  if (!sessionId) {
    list.innerHTML = '<div class="task-history-empty">Open a session to see its tasks.</div>';
    _setVisible('task-history', true);
    return;
  }
  const url = `/api/agentic/tasks?session_id=${encodeURIComponent(sessionId)}`;
  let data;
  try {
    const resp = await fetch(url);
    if (!resp.ok) return;
    data = await resp.json();
  } catch { return; }
  const tasks = Array.isArray(data) ? data : (data.tasks || data.items || []);
  if (!tasks.length) {
    list.innerHTML = '<div class="task-history-empty">No prior tasks for this session.</div>';
    _setVisible('task-history', true);
    return;
  }
  // Status → short symbol + tooltip text so the row is self-explanatory.
  const _STATUS = {
    completed: { mark: '\u2713', tip: 'Completed' },
    failed:    { mark: '\u2717', tip: 'Failed' },
    running:   { mark: '\u25B6', tip: 'In progress' },
    planning:  { mark: '\u00B7', tip: 'Planning' },
    plan_ready:{ mark: '\u00B7', tip: 'Plan ready' },
    cancelled: { mark: '\u2298', tip: 'Cancelled' },
  };
  const hint = `<div class="task-history-hint">Click a past task to reload its plan, progress, and artifacts into the panel above.</div>`;
  const rows = tasks.map(t => {
    const id = String(t.id || '');
    const status = (t.status || '').toLowerCase();
    const cur = Number(t.current_step) || 0;
    const total = Number(t.total_steps) || 0;
    const stepStr = total > 0 ? `${cur}/${total} steps` : '';
    return `<button class="task-history-item ${status}" data-task-id="${id}" data-status="${status}" title="">
      <span class="task-history-status"></span>
      <span class="task-history-title"></span>
      <span class="task-history-meta">${stepStr}</span>
    </button>`;
  }).join('');
  list.innerHTML = hint + rows;
  const items = list.querySelectorAll('.task-history-item');
  items.forEach((el, i) => {
    const t = tasks[i];
    const status = (t.status || '').toLowerCase();
    const meta = _STATUS[status] || { mark: '\u00B7', tip: status || 'Unknown' };
    el.querySelector('.task-history-title').textContent = t.title || '(untitled task)';
    el.querySelector('.task-history-status').textContent = meta.mark;
    el.title = `${meta.tip} \u2014 click to reload this task into the panel`;
    el.addEventListener('click', () => _openTaskDetail(t.id));
  });
  _setVisible('task-history', true);

  // Auto-restore the most recent task into the live render — but only if
  // we don't already have a live task in progress. This is what gives
  // panel persistence across reload + server restart.
  const mostRecent = tasks[0];
  if (mostRecent && mostRecent.id && !_activeTaskId) {
    _openTaskDetail(mostRecent.id);
  }
}

async function _openTaskDetail(taskId) {
  if (!taskId) return;
  try {
    const resp = await fetch(`/api/agentic/tasks/${encodeURIComponent(taskId)}`);
    if (!resp.ok) return;
    const t = await resp.json();
    const phases = await _loadTaskPhases(t);

    // Clear the artifact list so the restored task doesn't accumulate cards
    // from a previously displayed task.
    const artList = document.getElementById('task-artifact-list');
    if (artList) artList.innerHTML = '';
    _setVisible('task-artifacts', false);

    // Reset _activeTaskId so renderAgenticTaskMeta treats this as a "new"
    // task (clears pipeline list, snapshots fresh state).
    _activeTaskId = null;

    // Reuse the live renderer with the persisted task as a synthetic meta event.
    renderAgenticTaskMeta({
      mode: 'agentic',
      task_id: t.id,
      task_status: t.status || 'completed',
      task_title: t.title || '',
      current_step: t.current_step || 0,
      total_steps: phases.length || t.total_steps || 0,
      progress: t.progress || 0,
      plan_md: t.plan_md || '',
      autonomy_level: t.autonomy_level || 0,
      phases,
      flow_id: t.flow_id || '',
    });

    // Hydrate per-step outputs into collapsible detail rows so reload
    // preserves what each step produced, not just the task-level progress.
    if (t.step_outputs && typeof t.step_outputs === 'object') {
      _hydrateStepOutputs(t.step_outputs);
    }

    // Restore the artifacts produced by this task. Each row from the
    // backend already carries id + name + format + size_bytes + url.
    if (Array.isArray(t.artifacts)) {
      for (const a of t.artifacts) {
        renderAgenticArtifactCard(a);
      }
    }
    _emitTaskSnapshot();
  } catch { /* noop */ }
}

/** Populate the expandable output slot beneath each completed pipeline step.
 *
 *  ``stepOutputs`` is a map from step index (serialised as string by the API)
 *  to that step's output text. Steps with no output or that don't exist in
 *  the current pipeline are skipped silently.
 */
function _hydrateStepOutputs(stepOutputs) {
  const stepsEl = document.getElementById('task-pipeline-steps');
  if (!stepsEl) return;
  const children = Array.from(stepsEl.children);
  for (const [rawIdx, output] of Object.entries(stepOutputs)) {
    const idx = Number(rawIdx);
    if (!Number.isFinite(idx)) continue;
    const step = children.find((el) => Number(el.dataset.sourceIndex) === idx) || children[idx];
    if (!step) continue;
    const text = typeof output === 'string' ? output : String(output || '');
    if (!text.trim()) continue;
    const stepName = _stepKey(step.dataset.stepName || step.querySelector('.pipeline-step-label')?.textContent || `Step ${idx + 1}`);
    if (!stepName) continue;
    _stepOutputs.set(stepName, text);
    if (_countLines(text) <= 3 || text.length <= 220) {
      _expandedStepOutputs.add(stepName);
    }
    _renderStepOutput(step, stepName, text, step.dataset.stepStatus || 'complete');
  }
}

function _collectTaskPlanSnapshot() {
  const items = Array.from(document.querySelectorAll('#task-plan-content .plan-step'));
  return items.map((item, index) => ({
    index: index + 1,
    text: item.querySelector('.plan-step-text')?.textContent?.trim() || '',
    done: item.classList.contains('done'),
    active: item.classList.contains('active'),
  })).filter((item) => item.text);
}

function _collectTaskStepSnapshot() {
  const rows = Array.from(document.querySelectorAll('#task-pipeline-steps .pipeline-step'));
  return rows.map((row, index) => {
    const label = row.querySelector('.pipeline-step-label')?.textContent?.trim() || `Step ${index + 1}`;
    const output = _stepOutputs.get(_stepKey(label)) || '';
    return {
      index: Number(row.dataset.idx) || (index + 1),
      label,
      status: row.dataset.stepStatus || '',
      state: row.querySelector('.pipeline-step-state')?.textContent?.trim() || _statusLabel(row.dataset.stepStatus || ''),
      timing: row.querySelector('.pipeline-step-meta .pipeline-step-tool-count:last-child')?.textContent?.trim() || '',
      output,
      output_preview: _previewText(output, 180),
      expanded: _expandedStepOutputs.has(_stepKey(label)),
    };
  }).filter((step) => step.label);
}

function _collectTaskArtifactSnapshot() {
  const cards = Array.from(document.querySelectorAll('#task-artifact-list .task-artifact-card'));
  return cards.map((card) => {
    const id = card.dataset.artifactId || '';
    const title = card.querySelector('.task-artifact-title')?.textContent?.trim() || 'Artifact';
    const meta = card.querySelector('.task-artifact-meta')?.textContent?.trim() || '';
    const download = card.querySelector('a.task-artifact-btn[href]')?.getAttribute('href') || '';
    return {
      id,
      title,
      meta,
      download_url: download,
    };
  }).filter((artifact) => artifact.title);
}

export function getAgenticTaskSnapshot() {
  const titleEl = document.getElementById('task-title-label');
  const progressBar = document.getElementById('task-progress-bar');
  const progressText = document.getElementById('task-progress-text');
  const latest = _latestTaskMeta || {};
  const progressPct = Number(progressBar?.style?.width?.replace('%', '') || latest.progress || 0) || 0;
  return {
    task_id: _activeTaskId || latest.task_id || '',
    title: titleEl?.textContent?.trim() || latest.task_title || '',
    status: titleEl?.dataset?.status || latest.task_status || '',
    progress_pct: progressPct,
    progress_text: progressText?.textContent?.trim() || '',
    flow_id: latest.flow_id || '',
    plan: _collectTaskPlanSnapshot(),
    steps: _collectTaskStepSnapshot(),
    artifacts: _collectTaskArtifactSnapshot(),
    updated_at: Date.now(),
  };
}

function _emitTaskSnapshot() {
  const snapshot = getAgenticTaskSnapshot();
  document.dispatchEvent(new CustomEvent('augmentum:agentic-task-snapshot', {
    detail: snapshot,
  }));
  return snapshot;
}

// ---------------------------------------------------------------------------
// Settings Popover
// ---------------------------------------------------------------------------

function _toggleSettingsPopover() {
  const popover = document.getElementById('agentic-settings-popover');
  if (popover) popover.classList.toggle('hidden');
}

async function _loadAgenticSettings() {
  // Load all tool/string settings in one call
  try {
    const data = await getToolSettings();

    // Theme
    const themeSel = document.getElementById('agentic-theme-select');
    if (themeSel && data.agentic_artifact_theme) {
      themeSel.value = data.agentic_artifact_theme;
    }

    // Image model
    const imgSel = document.getElementById('agentic-image-model-select');
    if (imgSel && data.agentic_image_model) {
      imgSel.value = data.agentic_image_model;
    }

    // Autonomy level
    const autonomy = parseInt(data.agentic_default_autonomy, 10);
    if (autonomy >= 1 && autonomy <= 4) {
      _setAutonomyDots(autonomy);
    }
  } catch { /* ignore */ }

  // Populate image model dropdown from available models
  _populateImageModels();
}

async function _populateImageModels() {
  const sel = document.getElementById('agentic-image-model-select');
  if (!sel) return;

  // Keep the current value
  const current = sel.value;

  try {
    const models = await getImageModels();

    // Clear all but the default option
    while (sel.options.length > 1) sel.remove(1);

    for (const m of models) {
      const name = typeof m === 'string' ? m : (m.name || m.id || '');
      if (!name) continue;
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name.length > 30 ? name.slice(0, 30) + '...' : name;
      sel.appendChild(opt);
    }

    if (current) sel.value = current;
  } catch { /* ignore — image gen may not be enabled */ }
}

async function _saveSetting(key, value) {
  try {
    await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value }),
    });
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Autonomy Dots — clickable
// ---------------------------------------------------------------------------

function _setAutonomyDots(level) {
  const container = document.getElementById('autonomy-dots');
  const label = document.getElementById('autonomy-label');
  const desc = document.getElementById('autonomy-desc');
  if (!container) return;

  container.querySelectorAll('.autonomy-dot').forEach(dot => {
    const dotLevel = parseInt(dot.dataset.level, 10);
    dot.classList.toggle('active', dotLevel <= level);
  });
  if (label) label.textContent = AUTONOMY_NAMES[level] || '';
  if (desc) desc.textContent = AUTONOMY_DESCS[level] || '';
}

function _onAutonomyDotClick(e) {
  const dot = e.target.closest('.autonomy-dot');
  if (!dot) return;
  const level = parseInt(dot.dataset.level, 10);
  if (!level || level < 1 || level > 4) return;

  _setAutonomyDots(level);
  _saveSetting('agentic_default_autonomy', String(level));
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

export function initAgentic() {
  // Segmented tabs (Flows / Task) — replaces the old icon-button toggle
  document.querySelectorAll('.agentic-tabs .agentic-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const view = btn.dataset.view;
      if (view) switchAgenticView(view);
    });
  });

  // Settings popover toggle
  const settingsBtn = document.getElementById('agentic-settings-btn');
  if (settingsBtn) {
    settingsBtn.addEventListener('click', _toggleSettingsPopover);
  }

  // Settings change handlers
  const themeSelect = document.getElementById('agentic-theme-select');
  if (themeSelect) {
    themeSelect.addEventListener('change', () => {
      _saveSetting('agentic_artifact_theme', themeSelect.value);
    });
  }

  const imageModelSelect = document.getElementById('agentic-image-model-select');
  if (imageModelSelect) {
    imageModelSelect.addEventListener('change', () => {
      _saveSetting('agentic_image_model', imageModelSelect.value);
    });
  }

  // Clickable autonomy dots
  const dotsContainer = document.getElementById('autonomy-dots');
  if (dotsContainer) {
    dotsContainer.addEventListener('click', _onAutonomyDotClick);
  }

  // Load current settings
  _loadAgenticSettings();

  // Create flow editor in the agentic editor container
  const containerEl = document.getElementById('agentic-editor-view');
  if (containerEl) {
    editor = new FlowEditor({
      containerEl,
      mode: 'agentic',
      roles: AGENTIC_ROLES,
      accentColor: 'var(--mode-agentic)',
    });
    editor.init();
  }

  // Restore the panel whenever the user enters agentic mode or switches
  // session — so a refresh / server restart presents "where I left off"
  // instead of an empty task view. Reset the once-per-session debounce so
  // a real new session triggers a single fresh fetch.
  const restoreIfAgentic = () => {
    if ((app.state.mode || '') !== 'agentic') return;
    _activeTaskId = null;
    _historyLoadedForSession = '';  // allow one fetch for the new session
    loadAgenticTaskHistory().catch(() => { /* best effort */ });
  };
  document.addEventListener('augmentum:mode-changed', restoreIfAgentic);
  document.addEventListener('augmentum:session-changed', restoreIfAgentic);
  if ((app.state.mode || '') === 'agentic') {
    restoreIfAgentic();
  }
}
