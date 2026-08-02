import { escapeHtml, showToast } from './app.js';

function _workspaceQuery(workspaceId) {
  return workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : '';
}

function _statusLabel(power) {
  if (power.status === 'disabled') return 'Disabled';
  if (power.status === 'needs_setup') return 'Needs setup';
  return 'Ready';
}

function _statusTone(power) {
  if (power.status === 'disabled') return 'var(--text-muted)';
  if (power.status === 'needs_setup') return 'var(--warning)';
  if (power.active) return 'var(--accent)';
  return 'var(--success)';
}

function _summaryCounts(powers) {
  return {
    installed: powers.length,
    active: powers.filter(p => p.active).length,
    ready: powers.filter(p => p.status === 'ready').length,
    setup: powers.filter(p => p.status === 'needs_setup').length,
  };
}

function _requirementsText(power) {
  const parts = [];
  if (power.required_mcp_servers?.length) parts.push(`MCP: ${power.required_mcp_servers.join(', ')}`);
  if (power.required_bins?.length) parts.push(`Bins: ${power.required_bins.join(', ')}`);
  if (power.required_env?.length) parts.push(`Env: ${power.required_env.join(', ')}`);
  return parts.join(' • ');
}

function _labelize(value) {
  return String(value || '')
    .split(/[_-]/g)
    .filter(Boolean)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

export async function fetchPowers(workspaceId = '') {
  const resp = await fetch(`/api/powers${_workspaceQuery(workspaceId)}`);
  if (!resp.ok) throw new Error('Failed to load Powers');
  return resp.json();
}

export async function activatePower(powerId, workspaceId = '') {
  const resp = await fetch(`/api/powers/${encodeURIComponent(powerId)}/activate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ workspace_id: workspaceId, source: 'manual', scope: 'workspace' }),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || 'Failed to activate Power');
  }
  return resp.json();
}

export async function clearActivePower(workspaceId = '') {
  const resp = await fetch('/api/powers/clear-activation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ workspace_id: workspaceId }),
  });
  if (!resp.ok) throw new Error('Failed to clear active Power');
  return resp.json();
}

export async function setPowerEnabled(powerId, enabled, workspaceId = '') {
  const resp = await fetch(`/api/powers/${encodeURIComponent(powerId)}/${enabled ? 'enable' : 'disable'}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ workspace_id: workspaceId }),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || 'Failed to update Power');
  }
  return resp.json();
}

function _renderPowerRow(power) {
  const badgeStyle = `display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:999px;border:1px solid color-mix(in srgb, ${_statusTone(power)} 35%, var(--border));color:${_statusTone(power)};font-size:var(--text-xs);font-weight:600;`;
  const meta = [
    power.kind ? `Kind: ${_labelize(power.kind)}` : '',
    power.activation_policy ? `Activation: ${_labelize(power.activation_policy)}` : '',
    power.invocation ? `Invocation: ${power.invocation}` : '',
    power.source_label ? `Source: ${power.source_label}` : '',
    power.modes?.length ? `Modes: ${power.modes.join(', ')}` : '',
  ].filter(Boolean).join(' • ');
  const requirements = _requirementsText(power);
  return `
  <div class="mcp-server-item" data-power-id="${escapeHtml(power.id)}" style="padding:var(--space-md);display:flex;flex-direction:column;gap:var(--space-sm);border-left:3px solid ${_statusTone(power)}">
    <div style="display:flex;justify-content:space-between;gap:var(--space-md);align-items:flex-start">
      <div style="min-width:0;flex:1">
        <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:6px">
          <span style="${badgeStyle}">${escapeHtml(_statusLabel(power))}</span>
          ${power.active ? '<span style="display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:999px;border:1px solid color-mix(in srgb, var(--accent) 40%, var(--border));color:var(--accent);font-size:var(--text-xs);font-weight:600;">Active</span>' : ''}
          <strong style="font-size:var(--text-sm)">${escapeHtml(power.display_name)}</strong>
        </div>
        <div class="settings-desc" style="margin-bottom:4px">${escapeHtml(power.description || 'No description provided.')}</div>
        <div style="font-size:var(--text-xs);color:var(--text-muted);line-height:1.6">${escapeHtml(meta)}</div>
        ${requirements ? `<div style="font-size:var(--text-xs);color:var(--text-muted);line-height:1.6;margin-top:4px">${escapeHtml(requirements)}</div>` : ''}
        ${power.health?.blocked_reasons?.length ? `<div style="font-size:var(--text-xs);color:var(--warning);line-height:1.6;margin-top:4px">${escapeHtml(power.health.blocked_reasons.join(' • '))}</div>` : ''}
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end">
        ${power.active ? '<button class="btn btn-sm" data-action="clear">Turn Off</button>' : `<button class="btn btn-primary btn-sm" data-action="activate"${power.status !== 'ready' || !power.enabled ? ' disabled' : ''}>Use Now</button>`}
        <button class="btn btn-sm" data-action="${power.enabled ? 'disable' : 'enable'}">${power.enabled ? 'Disable' : 'Enable'}</button>
      </div>
    </div>
    <details>
      <summary style="cursor:pointer;color:var(--text-muted);font-size:var(--text-xs)">Inspect</summary>
      <div style="display:grid;gap:var(--space-sm);margin-top:var(--space-sm)">
        <div style="font-size:var(--text-xs);color:var(--text-muted);line-height:1.7">
          <div><strong>Manifest:</strong> ${escapeHtml(power.manifest_name || '')}</div>
          <div><strong>Files:</strong> ${escapeHtml((power.files || []).join(', ') || 'Manifest only')}</div>
          ${power.activation_windows?.length ? `<div><strong>Activation windows:</strong> ${escapeHtml(power.activation_windows.map(_labelize).join(', '))}</div>` : ''}
          ${power.triggers?.length ? `<div><strong>Triggers:</strong> ${escapeHtml(power.triggers.join(', '))}</div>` : ''}
          ${power.preferred_tools?.length ? `<div><strong>Preferred tools:</strong> ${escapeHtml(power.preferred_tools.join(', '))}</div>` : ''}
          ${power.blocked_tools?.length ? `<div><strong>Blocked tools:</strong> ${escapeHtml(power.blocked_tools.join(', '))}</div>` : ''}
        </div>
        ${power.instruction_excerpt ? `<pre style="margin:0;padding:var(--space-sm);border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface-1);white-space:pre-wrap;font:var(--text-xs)/1.55 var(--font-mono)">${escapeHtml(power.instruction_excerpt)}</pre>` : ''}
      </div>
    </details>
  </div>
  `;
}

export async function renderPowersPanel(container, { workspaceId = '' } = {}) {
  if (!container) return;
  container.innerHTML = '<div style="padding:var(--space-sm);color:var(--text-muted);font-size:var(--text-xs)">Loading Powers…</div>';
  try {
    const data = await fetchPowers(workspaceId);
    const powers = data.powers || [];
    const counts = _summaryCounts(powers);
    const summary = `
      <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--space-sm);margin-bottom:var(--space-md)">
        ${[
          ['Installed', counts.installed],
          ['Active', counts.active],
          ['Ready', counts.ready],
          ['Needs setup', counts.setup],
        ].map(([label, value]) => `
          <div style="padding:var(--space-sm);border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface-1)">
            <div style="font-size:var(--text-xs);color:var(--text-muted)">${label}</div>
            <div style="font-size:var(--text-lg);font-weight:700">${value}</div>
          </div>
        `).join('')}
      </div>
    `;
    if (!powers.length) {
      container.innerHTML = `${summary}<div style="padding:var(--space-md);border:1px dashed var(--border);border-radius:var(--radius-sm);color:var(--text-muted);font-size:var(--text-sm)">No Powers found yet. Add native packs under <code>.augmentum/powers/&lt;slug&gt;/POWER.md</code> or compatible packs under <code>.claude/skills/&lt;slug&gt;/SKILL.md</code>.</div>`;
      return;
    }
    container.innerHTML = `${summary}<div class="powers-grid">${powers.map(_renderPowerRow).join('')}</div>`;
    container.onclick = async (event) => {
      const button = event.target.closest('button[data-action]');
      if (!button) return;
      const row = button.closest('[data-power-id]');
      const powerId = row?.dataset.powerId;
      if (!powerId) return;
      button.disabled = true;
      try {
        if (button.dataset.action === 'activate') {
          await activatePower(powerId, workspaceId);
          showToast('Power activated', 'success');
        } else if (button.dataset.action === 'clear') {
          await clearActivePower(workspaceId);
          showToast('Power cleared', 'success');
        } else if (button.dataset.action === 'enable') {
          await setPowerEnabled(powerId, true, workspaceId);
          showToast('Power enabled', 'success');
        } else if (button.dataset.action === 'disable') {
          await setPowerEnabled(powerId, false, workspaceId);
          showToast('Power disabled', 'success');
        }
        await renderPowersPanel(container, { workspaceId });
      } catch (err) {
        console.warn('[powers] action failed', err);
        showToast(err.message || 'Power action failed', 'error');
      } finally {
        button.disabled = false;
      }
    };
  } catch (err) {
    console.warn('[powers] render failed', err);
    container.innerHTML = '<div style="padding:var(--space-sm);color:var(--danger);font-size:var(--text-xs)">Failed to load Powers.</div>';
  }
}

export async function refreshCoderPowerTile(tileEl, labelEl, workspaceId = '') {
  if (!tileEl || !labelEl) return;
  try {
    const data = await fetchPowers(workspaceId);
    const powers = data.powers || [];
    const active = powers.find(p => p.active);
    if (active) {
      labelEl.textContent = `Power: ${active.display_name}`;
      tileEl.classList.remove('hidden');
      return;
    }
    if (powers.length) {
      labelEl.textContent = `${powers.length} Power${powers.length !== 1 ? 's' : ''}`;
      tileEl.classList.remove('hidden');
      return;
    }
    tileEl.classList.add('hidden');
  } catch {
    tileEl.classList.add('hidden');
  }
}

export function powersHelpHtml(data, { workspaceId = '' } = {}) {
  const powers = data?.powers || [];
  const active = powers.find(p => p.active);
  const rows = powers.map(power => `
    <li>
      <code>${escapeHtml(power.id)}</code> — ${escapeHtml(power.display_name)}
      <span style="color:var(--text-muted)">(${escapeHtml(_statusLabel(power))}${power.active ? ', active' : ''})</span>
    </li>
  `).join('');
  return `
    <div class="coder-help-title">Powers</div>
    <div class="coder-help-section">
      <div class="coder-help-heading">Current workspace</div>
      <ul class="coder-help-list">
        <li>${active ? `Active Power: <code>${escapeHtml(active.display_name)}</code>` : 'No active Power for this workspace.'}</li>
        ${workspaceId ? `<li>Workspace: <code>${escapeHtml(workspaceId)}</code></li>` : ''}
      </ul>
    </div>
    <div class="coder-help-section">
      <div class="coder-help-heading">Installed</div>
      <ul class="coder-help-list">
        ${rows || '<li>No Powers discovered.</li>'}
      </ul>
    </div>
    <div class="coder-help-section">
      <div class="coder-help-heading">Commands</div>
      <ul class="coder-help-list">
        <li><code>/powers</code> — list installed Powers.</li>
        <li><code>/power &lt;id&gt;</code> — activate a Power for this workspace.</li>
        <li><code>/power off</code> — clear the active Power.</li>
        <li><code>/power why</code> — explain the current Power.</li>
      </ul>
    </div>
  `.trim();
}

export function activePowerWhyHtml(payload) {
  if (!payload?.power) {
    return `
      <div class="coder-help-title">Power</div>
      <div class="coder-help-section">
        <div class="coder-help-heading">Status</div>
        <ul class="coder-help-list">
          <li>No active Power for this workspace.</li>
        </ul>
      </div>
    `.trim();
  }
  const power = payload.power;
  return `
    <div class="coder-help-title">${escapeHtml(power.display_name)}</div>
    <div class="coder-help-section">
      <div class="coder-help-heading">Why it is active</div>
      <ul class="coder-help-list">
        <li>Status: ${escapeHtml(_statusLabel(power))}</li>
        ${power.kind ? `<li>Kind: ${escapeHtml(_labelize(power.kind))}</li>` : ''}
        ${power.activation_policy ? `<li>Activation: ${escapeHtml(_labelize(power.activation_policy))}</li>` : ''}
        ${power.activation_windows?.length ? `<li>Windows: ${escapeHtml(power.activation_windows.map(_labelize).join(', '))}</li>` : ''}
        ${power.active_reason ? `<li>Reason: ${escapeHtml(power.active_reason)}</li>` : ''}
        ${power.preferred_tools?.length ? `<li>Preferred tools: ${escapeHtml(power.preferred_tools.join(', '))}</li>` : ''}
        ${power.required_mcp_servers?.length ? `<li>Related MCP: ${escapeHtml(power.required_mcp_servers.join(', '))}</li>` : ''}
      </ul>
    </div>
    <div class="coder-help-section">
      <div class="coder-help-heading">Guidance</div>
      <pre style="margin:0;padding:var(--space-sm);border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface-1);white-space:pre-wrap;font:var(--text-xs)/1.55 var(--font-mono)">${escapeHtml(power.instruction_excerpt || power.description || '')}</pre>
    </div>
  `.trim();
}

export function powerActivationHtml(payload) {
  if (!payload) return '';
  const title = payload.source === 'controller' ? 'Power Engaged' : 'Power Active';
  return `
    <div class="coder-event-title">${escapeHtml(title)}</div>
    <div class="coder-event-body">
      <div><strong>${escapeHtml(payload.display_name || payload.id || 'Power')}</strong></div>
      <div class="coder-event-meta">
        ${payload.kind ? `<span>${escapeHtml(_labelize(payload.kind))}</span>` : ''}
        ${payload.checkpoint ? `<span>${escapeHtml(_labelize(payload.checkpoint))}</span>` : ''}
        ${payload.activation_policy ? `<span>${escapeHtml(_labelize(payload.activation_policy))}</span>` : ''}
        ${payload.scope ? `<span>${escapeHtml(_labelize(payload.scope))}</span>` : ''}
      </div>
      ${payload.reason ? `<div class="coder-event-reason">${escapeHtml(payload.reason)}</div>` : ''}
    </div>
  `.trim();
}
