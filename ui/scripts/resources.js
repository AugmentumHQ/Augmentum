/* ==========================================================================
   Resource Monitor — polls /api/resources/status and renders header widget
   ========================================================================== */

import { escapeHtml } from './app.js';

let _popoverEl = null;
let _triggerEl = null;
let _pollTimer = null;
let _popoverTimer = null;  // Faster refresh while popover is open
let _lastData = null;

const POLL_INTERVAL = 15_000;       // 15 s background
const POPOVER_POLL_INTERVAL = 3_000; // 3 s while popover is visible
const DEVICE_ICONS = {
  gpu: '\u{1F7E2}',     // green circle
  cpu: '\u{1F535}',     // blue circle
  'gpu+cpu': '\u{1F7E1}', // yellow circle
  remote: '\u{1F7E0}',  // orange circle
  unknown: '\u2B1C',     // white square
};

// ---------------------------------------------------------------------------
// Public: inject into header-right and start polling
// ---------------------------------------------------------------------------

export function initResources() {
  const headerRight = document.querySelector('.header-right');
  if (!headerRight) return;

  // Build the wrapper
  const wrap = document.createElement('div');
  wrap.className = 'resource-popover-wrap';
  wrap.innerHTML = _triggerHTML() + _popoverHTML();

  // Insert before settings button
  const settingsBtn = document.getElementById('settings-btn');
  if (settingsBtn) {
    headerRight.insertBefore(wrap, settingsBtn);
  } else {
    headerRight.appendChild(wrap);
  }

  _triggerEl = wrap.querySelector('.resource-trigger');
  _popoverEl = wrap.querySelector('.resource-popover');

  // Toggle popover
  _triggerEl.addEventListener('click', (e) => {
    e.stopPropagation();
    _toggle();
  });

  // Refresh button
  wrap.querySelector('.resource-pop-refresh').addEventListener('click', (e) => {
    e.stopPropagation();
    _refresh(true);
  });

  // Close on outside click
  document.addEventListener('click', (e) => {
    if (_popoverEl.dataset.visible === 'true' && !wrap.contains(e.target)) {
      _close();
    }
  });

  // Initial fetch + poll
  _refresh(false);
  _startBackgroundPoll();

  // Pause polling when the tab is hidden — otherwise 15s requests keep
  // firing while the user is playing a game, in another tab, or on a
  // different surface entirely, feeding 502s when the backend is busy.
  document.addEventListener('visibilitychange', _onVisibilityChange);

  // Clear poll timer on page unload to avoid orphaned intervals
  window.addEventListener('beforeunload', stopResources);
}

export function stopResources() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

function _startBackgroundPoll() {
  if (_pollTimer) return;
  _pollTimer = setInterval(() => _refresh(false), POLL_INTERVAL);
}

function _onVisibilityChange() {
  if (document.hidden) {
    stopResources();
  } else {
    // Back to visible — refresh immediately so the widget doesn't
    // sit on stale data, then resume the cadence.
    _refresh(false);
    _startBackgroundPoll();
  }
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

function _toggle() {
  const vis = _popoverEl.dataset.visible === 'true';
  if (vis) {
    _close();
  } else {
    _popoverEl.dataset.visible = 'true';
    _triggerEl.dataset.state = 'open';
    _refresh(false);
    // Start fast polling while popover is visible
    _popoverTimer = setInterval(() => _refresh(false), POPOVER_POLL_INTERVAL);
  }
}

function _close() {
  _popoverEl.dataset.visible = 'false';
  _triggerEl.dataset.state = 'idle';
  // Stop fast polling
  if (_popoverTimer) { clearInterval(_popoverTimer); _popoverTimer = null; }
}

async function _refresh(showSpin) {
  const refreshBtn = _popoverEl?.querySelector('.resource-pop-refresh');
  if (showSpin && refreshBtn) refreshBtn.classList.add('spinning');

  try {
    // showSpin = explicit user click on refresh; bypass the server's
    // ledger TTL cache so the user gets a guaranteed-fresh reading.
    // Background polls + popover auto-refresh hit the cache normally.
    const url = showSpin
      ? '/api/resources/status?fresh=1'
      : '/api/resources/status';
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status}`);
    _lastData = await res.json();
    _render(_lastData);
  } catch {
    _renderError();
  } finally {
    if (showSpin && refreshBtn) {
      setTimeout(() => refreshBtn.classList.remove('spinning'), 600);
    }
  }
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function _render(data) {
  const { gpu, ram, models, gpu_processes, unattributed_vram_mb, cpu_pct, cpu_scope, host } = data;
  const hasGPU = gpu && gpu.total_mb > 0;
  const hasRAM = ram && ram.total_mb > 0;
  const hostOK = !!(host && host.available && host.ram && host.ram.total_mb > 0);
  const isContainer = ram?.scope === 'runtime' || cpu_scope === 'runtime';

  // Update trigger bar + label (always reflects live usage)
  _updateTrigger(gpu, ram, models, hostOK ? host : null);

  // Bars
  const barsEl = _popoverEl.querySelector('.resource-bars');
  barsEl.innerHTML = '';

  if (hasGPU) {
    barsEl.appendChild(_buildBar('GPU', gpu.name || 'GPU', gpu.used_mb, gpu.total_mb));
  } else {
    barsEl.appendChild(_buildUnavailable('GPU', 'No GPU detected'));
  }

  // RAM — host first (matches Task Manager), then the container view.
  if (hostOK) {
    const hostSub = host.hostname ? `host · ${host.hostname}` : 'host';
    barsEl.appendChild(_buildBar('RAM', hostSub, host.ram.used_mb, host.ram.total_mb));
    if (hasRAM) {
      barsEl.appendChild(_buildBar('RAM', isContainer ? 'Augmentum container' : 'runtime', ram.used_mb, ram.total_mb));
    }
  } else if (hasRAM) {
    barsEl.appendChild(_buildBar('RAM', _scopeLabel(ram.scope), ram.used_mb, ram.total_mb));
  }

  // CPU — same host-then-container ordering.
  if (hostOK && host.cpu_pct != null && Number.isFinite(host.cpu_pct) && host.cpu_pct >= 0) {
    barsEl.appendChild(_buildCpuRow(host.cpu_pct, host.hostname ? `host · ${host.hostname}` : 'host'));
  }
  if (cpu_pct != null && Number.isFinite(cpu_pct) && cpu_pct >= 0) {
    barsEl.appendChild(_buildCpuRow(cpu_pct, hostOK ? (isContainer ? 'Augmentum container' : 'runtime') : _scopeLabel(cpu_scope)));
  }

  const scopeNote = hostOK
    ? `Host = your ${host.os || 'machine'} system (via the host stats agent). Container = the Augmentum runtime — these differ on Docker Desktop.`
    : (isContainer
      ? 'Showing the Augmentum container’s view (on Docker Desktop this is the WSL2/Linux VM, not the host — Task Manager will read differently). Run scripts/host_stats_agent.py on the host to also show host RAM/CPU.'
      : '');
  if (scopeNote) {
    const noteEl = document.createElement('div');
    noteEl.className = 'resource-scope-note';
    noteEl.textContent = scopeNote;
    barsEl.appendChild(noteEl);
  }

  // GPU process breakdown (if available)
  if (gpu_processes && gpu_processes.length > 0) {
    barsEl.appendChild(_buildProcessBreakdown(gpu_processes, unattributed_vram_mb || 0, gpu));
  }

  // Models. Only rewrite the DOM when the rendered markup actually changed —
  // re-setting identical innerHTML every 3s poll tears down + rebuilds the rows,
  // which reads as a flicker. With the backend carrying last-known stats
  // forward, steady-state polls produce identical markup and skip the rebuild.
  const listEl = _popoverEl.querySelector('.resource-model-list');
  const countEl = _popoverEl.querySelector('.resource-models-count');
  countEl.textContent = models.length;

  const modelsHtml = models.length === 0
    ? '<div class="resource-models-empty">No models loaded</div>'
    : models.map(_modelCard).join('');
  if (listEl._lastHtml !== modelsHtml) {
    listEl.innerHTML = modelsHtml;
    listEl._lastHtml = modelsHtml;
  }
}

function _renderError() {
  const barsEl = _popoverEl.querySelector('.resource-bars');
  barsEl.innerHTML = `
    <div class="resource-error">
      <div class="resource-error-icon">&#x26A0;</div>
      Could not reach resource API
    </div>`;

  const listEl = _popoverEl.querySelector('.resource-model-list');
  listEl.innerHTML = '';

  // Show idle state on trigger (not zeroed out)
  if (_triggerEl) {
    _triggerEl.style.removeProperty('--resource-pct');
    delete _triggerEl.dataset.live;
    const label = _triggerEl.querySelector('.resource-trigger-label');
    if (label) label.textContent = '--';
    const dot = _triggerEl.querySelector('.resource-trigger-dot');
    if (dot) dot.dataset.count = '0';
  }
}

function _updateTrigger(gpu, ram, models, host) {
  if (!_triggerEl) return;

  const label = _triggerEl.querySelector('.resource-trigger-label');
  const dot = _triggerEl.querySelector('.resource-trigger-dot');

  // Prefer GPU; if no GPU, prefer host RAM (matches Task Manager) over the
  // container's RAM view.
  const ramView = (host && host.ram && host.ram.total_mb > 0) ? host.ram : ram;

  if (gpu && gpu.total_mb > 0) {
    // GPU available — show VRAM utilization
    const pct = Math.round((gpu.used_mb / gpu.total_mb) * 100);
    const level = pct >= 90 ? 'crit' : pct >= 70 ? 'warn' : 'ok';
    _triggerEl.style.setProperty('--resource-pct', `${pct}%`);
    _triggerEl.dataset.live = level;
    label.textContent = `${_fmtMB(gpu.used_mb)}/${_fmtMB(gpu.total_mb)}`;
  } else if (ramView && ramView.total_mb > 0) {
    // No GPU — show RAM utilization instead
    const pct = Math.round((ramView.used_mb / ramView.total_mb) * 100);
    const level = pct >= 90 ? 'crit' : pct >= 70 ? 'warn' : 'ok';
    _triggerEl.style.setProperty('--resource-pct', `${pct}%`);
    _triggerEl.dataset.live = level;
    label.textContent = `${_fmtMB(ramView.used_mb)} RAM`;
  } else {
    // No data yet — show idle indicator
    _triggerEl.style.removeProperty('--resource-pct');
    delete _triggerEl.dataset.live;
    label.textContent = '--';
  }

  dot.dataset.count = models ? models.length : 0;
}

function _buildCpuRow(pct, scopeLabel) {
  const level = pct >= 90 ? 'crit' : pct >= 70 ? 'warn' : 'ok';
  const el = document.createElement('div');
  el.className = 'resource-bar-group resource-bar-compact';
  el.innerHTML = `
    <div class="resource-bar-label-row">
      <span class="resource-bar-label">CPU${scopeLabel ? ` <span style="font-weight:400;opacity:0.6;font-size:9px">${escapeHtml(scopeLabel)}</span>` : ''}</span>
      <span class="resource-bar-label-sub">${Math.round(pct)}%</span>
    </div>
    <div class="resource-bar-track">
      <div class="resource-bar-fill" data-level="${level}" style="width:${Math.max(0, Math.min(100, pct))}%"></div>
    </div>`;
  return el;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _buildBar(title, subtitle, used, total) {
  const pct = total > 0 ? Math.round((used / total) * 100) : 0;
  const level = pct >= 90 ? 'crit' : pct >= 70 ? 'warn' : 'ok';
  const free = total - used;

  const el = document.createElement('div');
  el.className = 'resource-bar-group';
  el.innerHTML = `
    <div class="resource-bar-label-row">
      <span class="resource-bar-label">${escapeHtml(title)}${subtitle ? ` <span style="font-weight:400;opacity:0.6;font-size:9px">${escapeHtml(subtitle)}</span>` : ''}</span>
      <span class="resource-bar-label-sub">${_fmtMB(used)} / ${_fmtMB(total)} &middot; ${_fmtMB(free)} free</span>
    </div>
    <div class="resource-bar-track">
      <div class="resource-bar-fill" data-level="${level}" style="width:${pct}%"></div>
    </div>`;
  return el;
}

function _buildProcessBreakdown(processes, unattributed, gpu) {
  // Aggregate by label (merge multiple ollama_llama_server PIDs, etc.)
  const grouped = new Map();
  for (const p of processes) {
    const key = p.label || p.name;
    const existing = grouped.get(key);
    if (existing) {
      existing.vram_mb += p.vram_mb;
      existing.pids.push(p.pid);
    } else {
      grouped.set(key, { label: key, vram_mb: p.vram_mb, pids: [p.pid] });
    }
  }

  const total = gpu ? gpu.total_mb : 0;
  const entries = [...grouped.values()].sort((a, b) => b.vram_mb - a.vram_mb);

  let html = '<div class="resource-proc-title">VRAM Breakdown</div>';
  for (const entry of entries) {
    const pct = total > 0 ? Math.round((entry.vram_mb / total) * 100) : 0;
    html += `
      <div class="resource-proc-row">
        <span class="resource-proc-name">${escapeHtml(entry.label)}</span>
        <span class="resource-proc-bar-wrap">
          <span class="resource-proc-bar-fill" style="width:${pct}%"></span>
        </span>
        <span class="resource-proc-val">${_fmtMB(entry.vram_mb)}</span>
      </div>`;
  }

  if (unattributed > 50) {
    const pct = total > 0 ? Math.round((unattributed / total) * 100) : 0;
    html += `
      <div class="resource-proc-row resource-proc-unattr">
        <span class="resource-proc-name">Other / driver</span>
        <span class="resource-proc-bar-wrap">
          <span class="resource-proc-bar-fill unattr" style="width:${pct}%"></span>
        </span>
        <span class="resource-proc-val">${_fmtMB(unattributed)}</span>
      </div>`;
  }

  const el = document.createElement('div');
  el.className = 'resource-proc-breakdown';
  el.innerHTML = html;
  return el;
}

function _buildUnavailable(title, msg) {
  const el = document.createElement('div');
  el.className = 'resource-bar-group';
  el.innerHTML = `
    <div class="resource-bar-label-row">
      <span class="resource-bar-label">${escapeHtml(title)}</span>
    </div>
    <div class="resource-bar-unavailable">${escapeHtml(msg)}</div>`;
  return el;
}

function _scopeLabel(scope) {
  if (scope === 'runtime') return 'container';
  if (scope === 'host') return 'host';
  return '';
}

// Backends that support unloading
const _UNLOADABLE = new Set(['ollama', 'llamacpp', 'llama.cpp', 'lm studio', 'engine', 'diffusers']);

// Sidecar containers (TTS/STT/classifier/vision) carry a `container` handle
// and `controllable` flag. Per-process VRAM isn't available on WSL2, so they
// show device + run-state instead of a (fake) MB number.
const _PAUSE_SVG = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="9" y1="5" x2="9" y2="19"/><line x1="15" y1="5" x2="15" y2="19"/></svg>';
const _RELOAD_SVG = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>';
const _UNLOAD_SVG = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';

function _modelCard(m) {
  const device = m.device || 'unknown';
  const icon = DEVICE_ICONS[device] || DEVICE_ICONS.unknown;
  const isContainer = !!m.container;
  const paused = m.status === 'paused';

  // Memory / state label. ``≈`` prefixes non-measured figures (declared
  // model-card constants, or the engine's estimated residual split) so the
  // panel is honest about fidelity (spec §4.6).
  const conf = m.confidence || 'measured';
  const approx = (conf === 'declared' || conf === 'estimated') ? '≈ ' : '';
  let memLabel = '';
  if (isContainer) {
    const dev = device === 'gpu' ? 'GPU' : (device === 'cpu' ? 'CPU' : device);
    if (paused) {
      memLabel = 'paused';
    } else {
      // VRAM (LLM siblings, from the llama-server log banner) + RAM/CPU
      // (measured from container stats: working-set + cpu delta).
      const bits = [];
      if (m.vram_mb) bits.push(`VRAM ${_fmtMB(m.vram_mb)}`);
      if (m.ram_mb) bits.push(`RAM ${_fmtMB(m.ram_mb)}`);
      if (m.cpu_pct != null && m.cpu_pct >= 0.1) bits.push(`${m.cpu_pct}%`);
      memLabel = bits.length ? `${dev} · ${bits.join(' · ')}` : `${dev} · active`;
    }
  } else {
    const memBits = [];
    if (m.vram_mb) memBits.push(`VRAM ${_fmtMB(m.vram_mb)}`);
    if (m.ram_mb) memBits.push(`RAM ${_fmtMB(m.ram_mb)}`);
    memLabel = memBits.length ? approx + memBits.join(' · ') : '';
  }

  const metaTags = [];
  if (!isContainer && m.status && m.status !== 'ready') metaTags.push('Loading');
  if (m.parameter_size) metaTags.push(m.parameter_size);
  if (m.quantization) metaTags.push(m.quantization);
  if (isContainer) {
    metaTags.push((m.subsystem || 'service').toUpperCase());
  } else {
    if (m.backend) metaTags.push(m.backend);
    if (m.subsystem && m.subsystem !== 'llm') metaTags.push(m.subsystem);
  }

  // Expiry countdown (ledger models only).
  let expiryHtml = '';
  if (!isContainer && m.expires_at) {
    const expiresMs = new Date(m.expires_at).getTime() - Date.now();
    if (expiresMs > 0) {
      const mins = Math.round(expiresMs / 60000);
      expiryHtml = mins > 60
        ? `<span class="resource-model-expiry">${Math.round(mins / 60)}h</span>`
        : `<span class="resource-model-expiry">${mins}m</span>`;
    }
  }

  // Controls: container → pause/reload; in-process/engine → unload.
  let controlHtml = '';
  if (isContainer && m.controllable) {
    controlHtml = paused
      ? `<button class="resource-model-reload" title="Reload (start container)" data-container="${escapeHtml(m.container)}" onclick="window._resourceResume(this)">${_RELOAD_SVG}</button>`
      : `<button class="resource-model-pause" title="Pause (stop container, frees VRAM)" data-container="${escapeHtml(m.container)}" onclick="window._resourcePause(this)">${_PAUSE_SVG}</button>`;
  } else if (_UNLOADABLE.has((m.backend || '').toLowerCase())) {
    controlHtml = `<button class="resource-model-unload" title="Unload from VRAM" data-model="${escapeHtml(m.name)}" data-backend="${escapeHtml(m.backend)}" onclick="window._resourceUnload(this)">${_UNLOAD_SVG}</button>`;
  }

  return `
    <div class="resource-model-card${paused ? ' is-paused' : ''}">
      <div class="resource-model-device" data-device="${escapeHtml(device)}">${icon}</div>
      <div class="resource-model-info">
        <div class="resource-model-name" title="${escapeHtml(m.name)}">${escapeHtml(m.name)}</div>
        ${metaTags.length ? `<div class="resource-model-meta">${metaTags.map(t => `<span class="resource-model-meta-tag">${escapeHtml(t)}</span>`).join('')}</div>` : ''}
      </div>
      ${expiryHtml}
      ${memLabel ? `<div class="resource-model-vram" data-confidence="${escapeHtml(conf)}" title="${escapeHtml(conf)} reading">${memLabel}</div>` : ''}
      ${controlHtml}
    </div>`;
}

window._resourceUnload = async function(btn) {
  const name = btn.dataset.model;
  const backend = btn.dataset.backend;
  btn.disabled = true;
  btn.classList.add('unloading');
  try {
    // Image models use the dedicated image unload endpoint
    const isDiffusers = backend.toLowerCase() === 'diffusers';
    const url = isDiffusers ? '/api/image/unload' : '/api/resources/unload';
    const body = isDiffusers ? {} : { name, backend };

    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.ok || data.unloaded) {
      // Refresh after short delay to let the backend release VRAM
      setTimeout(() => _refresh(true), 1000);
    } else {
      btn.classList.add('failed');
      btn.title = data.error || data.reason || 'Unload failed';
      setTimeout(() => { btn.classList.remove('failed'); btn.disabled = false; }, 2000);
    }
  } catch {
    btn.classList.add('failed');
    setTimeout(() => { btn.classList.remove('failed'); btn.disabled = false; }, 2000);
  }
};

async function _containerControl(btn, url) {
  const container = btn.dataset.container;
  btn.disabled = true;
  btn.classList.add('working');
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ container }),
    });
    const data = await res.json();
    if (data.ok) {
      // Stop/start takes a moment; refresh to flip pause↔reload + free VRAM.
      setTimeout(() => _refresh(true), 1500);
    } else {
      btn.classList.remove('working');
      btn.classList.add('failed');
      btn.title = data.error || 'Failed';
      setTimeout(() => { btn.classList.remove('failed'); btn.disabled = false; }, 2500);
    }
  } catch {
    btn.classList.remove('working');
    btn.classList.add('failed');
    setTimeout(() => { btn.classList.remove('failed'); btn.disabled = false; }, 2500);
  }
}

window._resourcePause = function(btn) { return _containerControl(btn, '/api/resources/pause'); };
window._resourceResume = function(btn) { return _containerControl(btn, '/api/resources/resume'); };

// ---------------------------------------------------------------------------
// Manual reclaim (spec §7.1) — preview, then execute.
//
// Two steps on purpose. The whole point of the manual phase is that the user
// sees what would be given up, and what is refused and why, BEFORE anything is
// unloaded. A one-click "free memory" button that silently evicts is the
// automatic governor with none of its safeguards.
// ---------------------------------------------------------------------------

function _reclaimPanel() {
  return _popoverEl?.querySelector('.resource-reclaim-panel');
}

window._resourceReclaimPreview = async function(btn) {
  const panel = _reclaimPanel();
  if (!panel) return;
  if (!panel.hidden) { panel.hidden = true; panel.innerHTML = ''; return; }

  btn.disabled = true;
  panel.hidden = false;
  panel.innerHTML = '<div class="resource-reclaim-note">Checking&hellip;</div>';
  try {
    const res = await fetch('/api/resources/reclaim/preview');
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Preview failed');
    panel.innerHTML = _reclaimPreviewHTML(data);
  } catch (err) {
    panel.innerHTML = `<div class="resource-reclaim-note failed">${escapeHtml(String(err.message || err))}</div>`;
  } finally {
    btn.disabled = false;
  }
};

function _reclaimPreviewHTML(data) {
  const cands = data.candidates || [];
  const blocked = data.blocked || [];

  const rows = cands.map(c => `
    <label class="resource-reclaim-row">
      <input type="checkbox" value="${escapeHtml(c.key)}" checked>
      <span class="resource-reclaim-label">${escapeHtml(c.label)}</span>
      <span class="resource-reclaim-size">${c.est ? '&mdash;' : _fmtMB(c.mib)}</span>
      ${c.restore_s ? `<span class="resource-reclaim-cost">~${c.restore_s}s to restore</span>` : ''}
      ${c.reason ? `<span class="resource-reclaim-why">${escapeHtml(c.reason)}</span>` : ''}
    </label>`).join('');

  // The blocked list is not filler — it is the "why did that barely free
  // anything" answer, and it is where mlocked weights become visible as
  // something that has to be refused at load time instead.
  const blockedHtml = blocked.length === 0 ? '' : `
    <div class="resource-reclaim-blocked">
      <div class="resource-reclaim-blocked-title">Not reclaimable</div>
      ${blocked.map(b => `
        <div class="resource-reclaim-row muted">
          <span class="resource-reclaim-label">${escapeHtml(b.label)}</span>
          <span class="resource-reclaim-size">${b.mib ? _fmtMB(b.mib) : ''}</span>
          <span class="resource-reclaim-why">${escapeHtml(b.reason || '')}</span>
        </div>`).join('')}
    </div>`;

  const total = data.estimated_mib || 0;
  const upTo = data.estimate_is_partial
    ? `at least ${_fmtMB(total)} (allocator slack is not knowable in advance)`
    : _fmtMB(total);

  return `
    ${cands.length === 0
      ? '<div class="resource-reclaim-note">Nothing is currently reclaimable.</div>'
      : `<div class="resource-reclaim-rows">${rows}</div>`}
    ${blockedHtml}
    <div class="resource-reclaim-actions">
      <span class="resource-reclaim-total">Frees ${upTo}</span>
      <button class="resource-reclaim-go" ${cands.length === 0 ? 'disabled' : ''}
              onclick="window._resourceReclaimRun(this)">Reclaim now</button>
    </div>`;
}

window._resourceReclaimRun = async function(btn) {
  const panel = _reclaimPanel();
  if (!panel) return;
  const keys = [...panel.querySelectorAll('input[type=checkbox]:checked')].map(i => i.value);
  if (keys.length === 0) return;

  btn.disabled = true;
  btn.textContent = 'Reclaiming…';
  try {
    const res = await fetch('/api/resources/reclaim', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keys }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || 'Reclaim failed');

    // Report the MEASURED container delta, never the sum of what each item
    // claimed to release — "freed" memory that went back to an allocator
    // arena instead of the kernel would otherwise read as a success.
    const skipped = (data.skipped || []).map(s =>
      `<div class="resource-reclaim-row muted">
         <span class="resource-reclaim-label">${escapeHtml(s.label)}</span>
         <span class="resource-reclaim-why">skipped &mdash; ${escapeHtml(s.reason || '')}</span>
       </div>`).join('');
    panel.innerHTML = `
      <div class="resource-reclaim-note">
        Freed <strong>${_fmtMB(data.measured_freed_mib || 0)}</strong>
        &mdash; measured working set ${_fmtMB(data.before?.used_mib || 0)}
        &rarr; ${_fmtMB(data.after?.used_mib || 0)}.
      </div>
      ${skipped}`;
    setTimeout(() => _refresh(true), 800);
  } catch (err) {
    btn.disabled = false;
    btn.textContent = 'Reclaim now';
    panel.insertAdjacentHTML('beforeend',
      `<div class="resource-reclaim-note failed">${escapeHtml(String(err.message || err))}</div>`);
  }
};

function _fmtMB(mb) {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb} MB`;
}

// ---------------------------------------------------------------------------
// HTML fragments
// ---------------------------------------------------------------------------

function _triggerHTML() {
  return `
    <button class="resource-trigger" data-state="idle" title="Resource monitor — GPU/RAM usage">
      <span class="resource-trigger-dot" data-count="0"></span>
      <span class="resource-trigger-label">--</span>
    </button>`;
}

function _popoverHTML() {
  return `
    <div class="resource-popover" data-visible="false">
      <div class="resource-pop-header">
        <span class="resource-pop-title">Resources</span>
        <button class="resource-pop-refresh" title="Refresh">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="23 4 23 10 17 10"/>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
          </svg>
        </button>
      </div>
      <div class="resource-bars"></div>
      <div class="resource-models-section">
        <div class="resource-models-header">
          <span class="resource-models-title">Loaded Models</span>
          <span class="resource-models-count">0</span>
        </div>
        <div class="resource-model-list"></div>
      </div>
      <div class="resource-reclaim-section">
        <button class="resource-reclaim-btn" onclick="window._resourceReclaimPreview(this)">
          Reclaim memory&hellip;
        </button>
        <div class="resource-reclaim-panel" hidden></div>
      </div>
    </div>`;
}
