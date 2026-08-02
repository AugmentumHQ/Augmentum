/**
 * coder-subagents-panel.js — UI panel for subagent dispatch
 *
 * Two surfaces, one modal shell, lazily mounted to <body>:
 *   - Roles tab: list of available roles (built-in + workspace + user),
 *     source pill, model + fallbacks, tool set, budget, context mode.
 *     Read-only in v1 — inline editing is a follow-up.
 *   - Runs tab: paginated history of recent task_dispatch spawns. Click
 *     a row to open the full transcript + tool_call_log in a side
 *     drawer. Optionally scoped to one parent_run_id.
 *
 * Entry points:
 *   - openSubagentsPanel({ tab, parentRunId })  — public, called from
 *     settings.js "Open panel" button + chat subagent card "history"
 *     button (when wired in a later pass).
 *
 * API endpoints consumed:
 *   - GET /api/coder/subagents/roles                     (roles tab)
 *   - GET /api/coder/subagents[?parent_run_id=...]       (runs tab)
 *   - GET /api/coder/subagents/{subagent_id}              (run detail drawer)
 */

import { escapeHtml } from './app.js';


const _state = {
  mounted: false,
  open: false,
  tab: 'roles',           // 'roles' | 'runs'
  parentRunId: '',
  roles: [],
  runs: [],
  loadingRoles: false,
  loadingRuns: false,
  detailRunId: '',
  detailData: null,
};


// --------------------------------------------------------------------------
// Mount + render
// --------------------------------------------------------------------------

function _mount() {
  if (_state.mounted) return;
  _state.mounted = true;

  const root = document.createElement('div');
  root.id = 'subagents-panel';
  root.className = 'subagents-panel hidden';
  root.innerHTML = `
    <div class="subagents-panel-backdrop" data-close="1"></div>
    <div class="subagents-panel-card">
      <header class="subagents-panel-head">
        <div class="subagents-panel-title">
          <span class="subagents-panel-eyebrow">CODER</span>
          <h2>Agents</h2>
        </div>
        <button class="subagents-panel-close" type="button" aria-label="Close" data-close="1">×</button>
      </header>
      <nav class="subagents-panel-tabs">
        <button class="subagents-panel-tab" data-tab="roles">Roles</button>
        <button class="subagents-panel-tab" data-tab="runs">Subagent Runs</button>
      </nav>
      <div class="subagents-panel-body">
        <div class="subagents-panel-tab-pane" data-pane="roles"></div>
        <div class="subagents-panel-tab-pane hidden" data-pane="runs"></div>
      </div>
      <aside class="subagents-panel-drawer hidden" data-drawer="1"></aside>
    </div>
  `;
  document.body.appendChild(root);

  // Backdrop + close
  root.querySelectorAll('[data-close="1"]').forEach(el => {
    el.addEventListener('click', closeSubagentsPanel);
  });

  // Tab switching
  root.querySelectorAll('.subagents-panel-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      _state.tab = btn.dataset.tab || 'roles';
      _renderTabs();
      _renderActivePane();
    });
  });

  // ESC to close
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && _state.open) closeSubagentsPanel();
  });
}

function _renderTabs() {
  document.querySelectorAll('#subagents-panel .subagents-panel-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === _state.tab);
  });
  document.querySelectorAll('#subagents-panel .subagents-panel-tab-pane').forEach(pane => {
    pane.classList.toggle('hidden', pane.dataset.pane !== _state.tab);
  });
}

function _renderActivePane() {
  if (_state.tab === 'roles') {
    _renderRolesPane();
    _loadRoles();
  } else if (_state.tab === 'runs') {
    _renderRunsPane();
    _loadRuns();
  }
}


// --------------------------------------------------------------------------
// Roles tab
// --------------------------------------------------------------------------

function _renderRolesPane() {
  const pane = document.querySelector('#subagents-panel [data-pane="roles"]');
  if (!pane) return;
  if (_state.loadingRoles) {
    pane.innerHTML = `<div class="subagents-empty">Loading roles…</div>`;
    return;
  }
  if (!_state.roles.length) {
    pane.innerHTML = `
      <div class="subagents-empty">
        <div class="subagents-empty-title">No roles registered yet.</div>
        <div class="subagents-empty-hint">Drop role files into <code>.augmentum/agents/*.md</code> (workspace) or <code>~/.augmentum/agents/*.md</code> (global). Built-ins should always appear here — if you see nothing, the registry isn't initialized.</div>
      </div>`;
    return;
  }
  pane.innerHTML = `
    <div class="subagents-roles-grid">
      ${_state.roles.map(_renderRoleCard).join('')}
    </div>
  `;
}

function _renderRoleCard(role) {
  const source = String(role.source || 'builtin');
  const tools = Array.isArray(role.tools) ? role.tools : [];
  const fallbacks = Array.isArray(role.fallback_models) ? role.fallback_models : [];
  const budget = role.budget || {};
  const preferred = role.preferred_model || '(parent\'s model)';
  return `
    <article class="subagents-role-card subagents-role-card--${escapeHtml(source)}">
      <header class="subagents-role-head">
        <h3 class="subagents-role-name">${escapeHtml(role.name || '')}</h3>
        <span class="subagents-source-pill subagents-source-pill--${escapeHtml(source)}">${escapeHtml(source)}</span>
      </header>
      ${role.description ? `<p class="subagents-role-desc">${escapeHtml(role.description)}</p>` : ''}
      <dl class="subagents-role-meta">
        <div>
          <dt>Preferred model</dt>
          <dd>${escapeHtml(preferred)}</dd>
        </div>
        ${fallbacks.length ? `
          <div>
            <dt>Fallbacks</dt>
            <dd>${fallbacks.map(f => `<code>${escapeHtml(f)}</code>`).join(', ')}</dd>
          </div>
        ` : ''}
        <div>
          <dt>Context mode</dt>
          <dd><code>${escapeHtml(role.context_mode || 'workspace')}</code></dd>
        </div>
        <div>
          <dt>Tool guard</dt>
          <dd><code>${escapeHtml(role.tool_guard || 'detector')}</code></dd>
        </div>
        <div>
          <dt>Budget</dt>
          <dd>${budget.max_iterations ?? '—'} iters · ${budget.max_wallclock_seconds ?? '—'}s · ${budget.max_tokens ? `${Math.round(budget.max_tokens / 1000)}k tok` : '—'}</dd>
        </div>
        <div>
          <dt>Parallelism</dt>
          <dd>${role.max_concurrent ?? 4} concurrent · spawn-subagents ${role.can_spawn_subagents ? 'yes' : 'no'}</dd>
        </div>
      </dl>
      <details class="subagents-role-tools">
        <summary>${tools.length} tool${tools.length === 1 ? '' : 's'}</summary>
        <div class="subagents-role-tools-list">
          ${tools.map(t => `<code class="subagents-tool-chip">${escapeHtml(t)}</code>`).join('')}
        </div>
      </details>
      ${role.file_path ? `<div class="subagents-role-path">${escapeHtml(role.file_path)}</div>` : ''}
    </article>
  `;
}


// --------------------------------------------------------------------------
// Runs tab
// --------------------------------------------------------------------------

function _renderRunsPane() {
  const pane = document.querySelector('#subagents-panel [data-pane="runs"]');
  if (!pane) return;
  if (_state.loadingRuns) {
    pane.innerHTML = '<div class="subagents-empty">Loading runs…</div>';
    _prependRunsFilter(pane);
    return;
  }
  if (!_state.runs.length) {
    pane.innerHTML = `
      <div class="subagents-empty">
        <div class="subagents-empty-title">No subagent runs yet.</div>
        <div class="subagents-empty-hint">When the lead coder model calls <code>task_dispatch</code>, each spawn lands here with full transcript + tool call log.</div>
      </div>
    `;
    _prependRunsFilter(pane);
    return;
  }
  pane.innerHTML = `
    <table class="subagents-runs-table">
      <thead>
        <tr>
          <th>Role</th>
          <th>Model</th>
          <th>Stop</th>
          <th class="num">Iters</th>
          <th class="num">Tokens</th>
          <th class="num">Wall</th>
          <th>Started</th>
        </tr>
      </thead>
      <tbody>
        ${_state.runs.map(_renderRunRow).join('')}
      </tbody>
    </table>
  `;
  _prependRunsFilter(pane);
  pane.querySelectorAll('.subagents-run-row').forEach(row => {
    row.addEventListener('click', () => _openDetail(row.dataset.subagentId));
  });
}

function _prependRunsFilter(pane) {
  if (!_state.parentRunId) return;

  const filter = document.createElement('div');
  filter.className = 'subagents-runs-filter';
  filter.appendChild(document.createTextNode('Scoped to parent run '));

  const code = document.createElement('code');
  code.textContent = _state.parentRunId;
  filter.appendChild(code);
  filter.appendChild(document.createTextNode(' '));

  const clear = document.createElement('button');
  clear.className = 'subagents-runs-clear';
  clear.type = 'button';
  clear.textContent = 'clear';
  clear.addEventListener('click', () => {
    _state.parentRunId = '';
    _loadRuns();
  });
  filter.appendChild(clear);

  pane.prepend(filter);
}

function _renderRunRow(run) {
  const started = run.started_at
    ? new Date(Number(run.started_at) * 1000).toISOString().slice(0, 16).replace('T', ' ')
    : '—';
  const tokens = (Number(run.tokens_in) || 0) + (Number(run.tokens_out) || 0);
  const wallSec = ((Number(run.wallclock_ms) || 0) / 1000).toFixed(1);
  const stop = String(run.stop_reason || '');
  return `
    <tr class="subagents-run-row" data-subagent-id="${escapeHtml(run.subagent_id || '')}">
      <td><span class="subagents-role-tag">${escapeHtml(run.role || '')}</span></td>
      <td><code>${escapeHtml(run.model_resolved || run.model_spec || '—')}</code></td>
      <td><span class="subagents-stop-pill subagents-stop-pill--${escapeHtml(stop)}">${escapeHtml(stop || '—')}</span></td>
      <td class="num">${Number(run.iterations) || 0}</td>
      <td class="num">${tokens.toLocaleString()}</td>
      <td class="num">${wallSec}s</td>
      <td>${escapeHtml(started)}</td>
    </tr>
  `;
}


// --------------------------------------------------------------------------
// Run-detail drawer
// --------------------------------------------------------------------------

async function _openDetail(subagentId) {
  if (!subagentId) return;
  _state.detailRunId = subagentId;
  _state.detailData = null;
  _renderDetail();
  try {
    const resp = await fetch(`/api/coder/subagents/${encodeURIComponent(subagentId)}`, {
      credentials: 'include',
    });
    if (!resp.ok) {
      _state.detailData = { error: `Fetch failed (${resp.status})` };
    } else {
      _state.detailData = await resp.json();
    }
  } catch (err) {
    _state.detailData = { error: String(err?.message || err) };
  }
  _renderDetail();
}

function _closeDetail() {
  _state.detailRunId = '';
  _state.detailData = null;
  const drawer = document.querySelector('#subagents-panel [data-drawer="1"]');
  if (drawer) drawer.classList.add('hidden');
}

function _renderDetail() {
  const drawer = document.querySelector('#subagents-panel [data-drawer="1"]');
  if (!drawer) return;
  if (!_state.detailRunId) {
    drawer.classList.add('hidden');
    return;
  }
  drawer.classList.remove('hidden');
  if (!_state.detailData) {
    drawer.innerHTML = `<div class="subagents-drawer-loading">Loading…</div>`;
    return;
  }
  if (_state.detailData.error) {
    drawer.innerHTML = `
      <div class="subagents-drawer-header">
        <button class="subagents-drawer-close" type="button">←</button>
        <span>Detail</span>
      </div>
      <div class="subagents-drawer-error">${escapeHtml(_state.detailData.error)}</div>
    `;
    drawer.querySelector('.subagents-drawer-close')?.addEventListener('click', _closeDetail);
    return;
  }
  const d = _state.detailData;
  const log = Array.isArray(d.tool_call_log) ? d.tool_call_log : [];
  drawer.innerHTML = `
    <div class="subagents-drawer-header">
      <button class="subagents-drawer-close" type="button">←</button>
      <span class="subagents-role-tag">${escapeHtml(d.role || '')}</span>
      <code class="subagents-drawer-id">${escapeHtml(d.subagent_id || '')}</code>
    </div>
    <dl class="subagents-drawer-meta">
      <div><dt>Model</dt><dd><code>${escapeHtml(d.model_resolved || '')}</code> ${d.model_spec && d.model_spec !== d.model_resolved ? `(<code>${escapeHtml(d.model_spec)}</code>)` : ''}</dd></div>
      <div><dt>Backend</dt><dd><code>${escapeHtml(d.backend_key || '—')}</code></dd></div>
      <div><dt>Stop</dt><dd>${escapeHtml(d.stop_reason || '')}${d.stop_detail ? ` — ${escapeHtml(d.stop_detail)}` : ''}</dd></div>
      <div><dt>Iters / tools / tokens / wall</dt><dd>${d.iterations} / ${d.tool_calls} / ${(Number(d.tokens_in) || 0) + (Number(d.tokens_out) || 0)} / ${((Number(d.wallclock_ms) || 0) / 1000).toFixed(1)}s</dd></div>
      <div><dt>Context mode</dt><dd><code>${escapeHtml(d.context_mode || '')}</code></dd></div>
      <div><dt>Parent run</dt><dd><code>${escapeHtml(d.parent_run_id || '—')}</code></dd></div>
    </dl>
    <details class="subagents-drawer-section" open>
      <summary>Prompt</summary>
      <pre class="subagents-drawer-prompt">${escapeHtml(d.prompt || '')}</pre>
    </details>
    <details class="subagents-drawer-section" open>
      <summary>Output</summary>
      <pre class="subagents-drawer-output">${escapeHtml(d.output_text || '')}</pre>
    </details>
    <details class="subagents-drawer-section">
      <summary>Tool call log (${log.length})</summary>
      ${log.length
        ? `<ol class="subagents-drawer-log">${log.map(_renderLogEntry).join('')}</ol>`
        : `<div class="subagents-drawer-empty">No tool calls recorded.</div>`}
    </details>
  `;
  drawer.querySelector('.subagents-drawer-close')?.addEventListener('click', _closeDetail);
}

function _renderLogEntry(entry) {
  if (!entry || typeof entry !== 'object') return '';
  const outcome = String(entry.outcome || '');
  const reason = entry.reason ? ` — ${entry.reason}` : '';
  let args = '';
  try {
    args = JSON.stringify(entry.args || {}, null, 0).slice(0, 200);
  } catch {
    args = '{}';
  }
  return `
    <li class="subagents-drawer-log-row subagents-drawer-log-row--${escapeHtml(outcome)}">
      <span class="subagents-drawer-log-iter">#${entry.iteration ?? '?'}</span>
      <code class="subagents-drawer-log-tool">${escapeHtml(entry.tool || '')}</code>
      <span class="subagents-drawer-log-outcome">${escapeHtml(outcome)}${escapeHtml(reason)}</span>
      <code class="subagents-drawer-log-args">${escapeHtml(args)}</code>
      <span class="subagents-drawer-log-elapsed">${entry.elapsed_ms || 0}ms</span>
    </li>
  `;
}


// --------------------------------------------------------------------------
// Data loaders
// --------------------------------------------------------------------------

async function _loadRoles() {
  _state.loadingRoles = true;
  _renderRolesPane();
  try {
    const resp = await fetch('/api/coder/subagents/roles', { credentials: 'include' });
    if (resp.ok) {
      const data = await resp.json();
      _state.roles = Array.isArray(data) ? data : [];
    } else {
      _state.roles = [];
    }
  } catch {
    _state.roles = [];
  } finally {
    _state.loadingRoles = false;
    _renderRolesPane();
  }
}

async function _loadRuns() {
  _state.loadingRuns = true;
  _renderRunsPane();
  try {
    const qs = _state.parentRunId
      ? `?parent_run_id=${encodeURIComponent(_state.parentRunId)}&limit=100`
      : '?limit=50';
    const resp = await fetch(`/api/coder/subagents${qs}`, { credentials: 'include' });
    if (resp.ok) {
      const data = await resp.json();
      _state.runs = Array.isArray(data) ? data : [];
    } else {
      _state.runs = [];
    }
  } catch {
    _state.runs = [];
  } finally {
    _state.loadingRuns = false;
    _renderRunsPane();
  }
}


// --------------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------------

export function openSubagentsPanel({ tab = 'roles', parentRunId = '' } = {}) {
  _mount();
  _state.tab = ['roles', 'runs'].includes(tab) ? tab : 'roles';
  _state.parentRunId = parentRunId || '';
  _state.open = true;
  const root = document.getElementById('subagents-panel');
  if (root) root.classList.remove('hidden');
  _renderTabs();
  _renderActivePane();
}

export function closeSubagentsPanel() {
  _state.open = false;
  const root = document.getElementById('subagents-panel');
  if (root) root.classList.add('hidden');
  _closeDetail();
}

// Auto-wire the settings "Open panel" button if it exists when this
// module loads. Settings re-render keeps the button — handler delegates
// to the current openSubagentsPanel via the global.
if (typeof window !== 'undefined') {
  window.openSubagentsPanel = openSubagentsPanel;
  document.addEventListener('click', (e) => {
    const target = e.target;
    if (!(target instanceof Element)) return;
    if (target.id === 'subagents-open-panel-btn') {
      e.preventDefault();
      openSubagentsPanel({ tab: 'roles' });
    }
  });
}
