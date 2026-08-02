/**
 * Dream Journal — UI module for the dream system.
 * Provides journal timeline, portrait card, entry management, and checkpoint controls.
 */

const API = '/api/dream';

// ── State ──
let _entries = [];
let _portrait = null;
let _status = null;
let _checkpoints = [];

// ── Helpers ──

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/`/g, '&#96;').replace(/\$\{/g, '&#36;{');
}

function relativeDate(iso) {
  if (!iso) return '';
  try {
    // Server stores created_at in two shapes:
    //   * Python ISO with offset: "2026-05-02T04:37:14.231721+00:00"
    //   * SQLite datetime('now'): "2026-05-02 05:13:35"  (NO timezone)
    // The second is naive UTC, but `new Date(...)` treats unmarked
    // timestamps as LOCAL, so EST users saw entries dated +4 hours
    // ahead of "now" and the diff went negative — producing the
    // "-1 days ago" labels on every entry. Normalize: if the string
    // has no offset and no trailing Z, append Z so JS parses it as
    // UTC (the server's actual intent).
    let s = iso;
    const hasTimezone = /[Z]|[+-]\d{2}:?\d{2}$/.test(s);
    if (!hasTimezone) {
      // Convert space separator to 'T' for ISO 8601 strict parsing,
      // then mark as UTC.
      s = s.replace(' ', 'T') + 'Z';
    }
    const d = new Date(s);
    const now = new Date();
    const diff = now - d;
    // Future timestamps (clock skew, malformed parse) clamp to "just now"
    // rather than producing negative-day labels.
    if (diff < 0) return 'just now';
    const days = Math.floor(diff / 86400000);
    if (days === 0) return 'today';
    if (days === 1) return 'yesterday';
    if (days < 7) return `${days} days ago`;
    if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
    return `${Math.floor(days / 30)} months ago`;
  } catch { return ''; }
}

const TYPE_ICONS = {
  reflection: '💭',
  voice_note: '🗣️',
  active_thread: '💡',
  impression: '💫',
};

const TYPE_LABELS = {
  reflection: 'Reflection',
  voice_note: 'Voice Note',
  active_thread: 'Active Thread',
  impression: 'Impression',
};

// ── API Calls ──

async function api(path, opts = {}) {
  try {
    const resp = await fetch(`${API}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch { return null; }
}

export async function fetchJournal(limit = 50, offset = 0) {
  const data = await api(`/journal?limit=${limit}&offset=${offset}`);
  if (data) { _entries = data.entries || []; }
  return data;
}

export async function fetchPortrait() {
  _portrait = await api('/portrait');
  return _portrait;
}

export async function fetchStatus() {
  _status = await api('/status');
  return _status;
}

export async function fetchCheckpoints() {
  const data = await api('/portrait/checkpoints');
  _checkpoints = data?.checkpoints || [];
  return _checkpoints;
}

// Module-level cache for the recent-cycles audit log (op visibility).
let _cycles = [];

export async function fetchCycles(limit = 10) {
  const data = await api(`/cycles?limit=${limit}`);
  _cycles = data?.cycles || [];
  return _cycles;
}

export async function fetchCycleDetail(cycleId) {
  return await api(`/cycles/${encodeURIComponent(cycleId)}`);
}

export async function triggerDream() {
  return await api('/trigger', { method: 'POST', body: '{}' });
}

export async function updateEntry(id, updates) {
  return await api(`/journal/${id}`, { method: 'PUT', body: JSON.stringify(updates) });
}

export async function deleteEntry(id) {
  return await api(`/journal/${id}`, { method: 'DELETE' });
}

export async function saveCheckpoint(name) {
  return await api('/portrait/checkpoint', { method: 'POST', body: JSON.stringify({ name }) });
}

export async function restoreCheckpoint(id) {
  return await api(`/portrait/restore/${id}`, { method: 'POST' });
}

export async function regeneratePortrait() {
  return await api('/portrait/regenerate', { method: 'POST', body: '{}' });
}

export async function resetToFoundation() {
  return await api('/portrait/reset', { method: 'POST', body: '{}' });
}

// ── Rendering ──

export function renderPortraitCard(container) {
  if (!container) return;
  if (!_portrait) {
    // Empty-state Generate action — without this, users with journal
    // entries but no portrait yet (e.g. after a cycle that produced
    // entries but failed at portrait synthesis) had no way to trigger
    // synthesis from the UI. The Regenerate button further down only
    // renders when a portrait already exists, creating a chicken-and-
    // egg trap. The button is a no-op when there are zero entries —
    // the route returns 404 and the existing handler shows that.
    container.innerHTML = `
      <div class="dream-empty">
        <p>No dream portrait yet.</p>
        <p class="dream-hint">As you chat and approve memories, your AI will reflect on your shared experiences.</p>
        <div class="dream-portrait-actions" style="margin-top:0.75rem">
          <button class="dream-btn dream-btn-sm" onclick="window._dreamRegenPortrait()">Generate from journal</button>
        </div>
      </div>`;
    return;
  }
  container.innerHTML = `
    <div class="dream-portrait">
      <div class="dream-portrait-section">
        <h4>Voice</h4>
        <p>${escapeHtml(_portrait.voice_notes)}</p>
      </div>
      <div class="dream-portrait-section">
        <h4>Active Threads</h4>
        <p>${escapeHtml(_portrait.active_threads)}</p>
      </div>
      <div class="dream-portrait-section">
        <h4>Impressions</h4>
        <p>${escapeHtml(_portrait.impressions)}</p>
      </div>
      <div class="dream-portrait-actions">
        <button class="dream-btn dream-btn-sm" onclick="window._dreamRegenPortrait()">Regenerate</button>
        <button class="dream-btn dream-btn-sm" onclick="window._dreamSaveCheckpoint()">Checkpoint</button>
        <button class="dream-btn dream-btn-sm dream-btn-danger" onclick="window._dreamReset()">Reset</button>
      </div>
    </div>`;
}

export function renderJournal(container) {
  if (!container) return;
  if (!_entries.length) {
    container.innerHTML = '<p class="dream-hint">No dream entries yet.</p>';
    return;
  }

  let html = '';
  let lastDate = '';

  for (const entry of _entries) {
    const dateStr = entry.created_at ? entry.created_at.split('T')[0] : '';
    if (dateStr !== lastDate) {
      html += `<div class="dream-date-separator">${escapeHtml(relativeDate(entry.created_at))} &mdash; ${escapeHtml(dateStr)}</div>`;
      lastDate = dateStr;
    }

    const typeIcon = TYPE_ICONS[entry.entry_type] || '💭';
    const typeLabel = TYPE_LABELS[entry.entry_type] || 'Reflection';
    const pinnedClass = entry.pinned ? ' dream-entry-pinned' : '';

    html += `
      <div class="dream-entry${pinnedClass}" data-id="${escapeHtml(entry.id)}">
        <div class="dream-entry-header">
          <span class="dream-entry-type" title="${escapeHtml(typeLabel)}">${typeIcon}</span>
          <span class="dream-entry-date" title="${escapeHtml(entry.created_at)}">${escapeHtml(relativeDate(entry.created_at))}</span>
          ${entry.pinned ? '<span class="dream-pin-badge">📌</span>' : ''}
        </div>
        <div class="dream-entry-content">${escapeHtml(entry.content)}</div>
        <div class="dream-entry-actions">
          <button class="dream-btn-icon" title="${entry.pinned ? 'Unpin' : 'Pin'}" onclick="window._dreamTogglePin('${entry.id}', ${!entry.pinned})">📌</button>
          <button class="dream-btn-icon" title="Delete" onclick="window._dreamDeleteEntry('${entry.id}')">🗑️</button>
        </div>
      </div>`;
  }

  container.innerHTML = html;
}

export function renderStatus(container) {
  if (!container || !_status) return;
  const eligible = _status.next_dream_eligible ? '✅ Ready' : '⏳ Waiting';
  container.innerHTML = `
    <div class="dream-status">
      <span>Messages: ${_status.messages_since_dream} / ${_status.approved_memories_since_dream} approved</span>
      <span>${eligible}</span>
      <button class="dream-btn dream-btn-sm" onclick="window._dreamTrigger()">Dream Now</button>
    </div>`;
}

export function renderCheckpoints(container) {
  if (!container) return;
  if (!_checkpoints.length) {
    container.innerHTML = '';
    return;
  }
  let html = '<h4>Checkpoints</h4>';
  for (const cp of _checkpoints) {
    html += `
      <div class="dream-checkpoint-item">
        <span>${escapeHtml(cp.checkpoint_name)}</span>
        <span class="dream-checkpoint-date">${escapeHtml(relativeDate(cp.created_at))}</span>
        <button class="dream-btn-icon" title="Restore" onclick="window._dreamRestoreCheckpoint('${cp.id}')">↩️</button>
      </div>`;
  }
  container.innerHTML = html;
}

// Recent dream cycles — operator visibility into the introspection
// pipeline. Each row summarises one cycle (status, trigger, model used,
// tokens, duration, error if any) so failures are diagnosable without
// touching the DB. Clicking a row opens the cycle detail (entry IDs).
export function renderCycles(container) {
  if (!container) return;
  if (!_cycles.length) {
    container.innerHTML = '<h4>Recent cycles</h4><div class="dream-cycle-empty">No cycles yet.</div>';
    return;
  }
  let html = '<h4>Recent cycles</h4>';
  for (const c of _cycles) {
    const status = (c.status || 'unknown').toLowerCase();
    const statusIcon = status === 'completed' ? '✅'
      : status === 'failed' || status === 'error' ? '❌'
      : status === 'running' ? '⏳' : '·';
    const duration = c.duration_ms ? `${Math.round(c.duration_ms / 1000)}s` : '—';
    const tokens = c.tokens_used ? c.tokens_used.toLocaleString() : '—';
    const errSnip = c.error ? ` <span class="dream-cycle-err" title="${escapeHtml(c.error)}">${escapeHtml(c.error.slice(0, 60))}</span>` : '';
    html += `
      <div class="dream-cycle-row" data-cycle-id="${escapeHtml(c.id)}" title="Click for entry IDs">
        <span class="dream-cycle-status">${statusIcon}</span>
        <span class="dream-cycle-reason">${escapeHtml(c.trigger_reason || 'auto')}</span>
        <span class="dream-cycle-date">${escapeHtml(relativeDate(c.started_at || ''))}</span>
        <span class="dream-cycle-stats">${c.entries_count || 0} entries · ${duration} · ${tokens} tok</span>
        ${errSnip}
      </div>`;
  }
  container.innerHTML = html;
  container.querySelectorAll('.dream-cycle-row').forEach((row) => {
    row.addEventListener('click', async () => {
      const cycleId = row.dataset.cycleId;
      if (!cycleId) return;
      const detail = await fetchCycleDetail(cycleId);
      if (!detail) return;
      const entryIds = (detail.entry_ids || []).join(', ') || '(none)';
      const lines = [
        `Cycle ${cycleId}`,
        `Status: ${detail.status || '?'}`,
        `Model: ${detail.model_used || '?'}`,
        `Tokens: ${detail.tokens_used || 0}`,
        `Duration: ${detail.duration_ms ? Math.round(detail.duration_ms / 1000) + 's' : '?'}`,
        `Started: ${detail.started_at || '?'}`,
        `Completed: ${detail.completed_at || '?'}`,
        detail.error ? `Error: ${detail.error}` : '',
        `Entries: ${entryIds}`,
      ].filter(Boolean);
      alert(lines.join('\n'));
    });
  });
}

// ── Global Handlers ──

window._dreamTogglePin = async (id, pinned) => {
  await updateEntry(id, { pinned });
  await refreshJournal();
};

window._dreamDeleteEntry = async (id) => {
  if (!confirm('Delete this dream entry?')) return;
  await deleteEntry(id);
  await refreshJournal();
};

window._dreamTrigger = async () => {
  const result = await triggerDream();
  if (result?.cycle_id) {
    setTimeout(() => refreshAll(), 5000); // Refresh after cycle likely completes
  }
};

window._dreamRegenPortrait = async () => {
  await regeneratePortrait();
  await refreshPortrait();
};

window._dreamSaveCheckpoint = async () => {
  const name = prompt('Checkpoint name:');
  if (!name) return;
  await saveCheckpoint(name);
  await fetchCheckpoints();
  renderCheckpoints(document.getElementById('dream-checkpoints'));
};

window._dreamRestoreCheckpoint = async (id) => {
  if (!confirm('Restore this checkpoint? Current portrait will be replaced.')) return;
  await restoreCheckpoint(id);
  await refreshPortrait();
};

window._dreamReset = async () => {
  if (!confirm('Reset to foundation? This will delete ALL dream entries, portraits, and checkpoints.')) return;
  await resetToFoundation();
  await refreshAll();
};

// ── Refresh Helpers ──

async function refreshPortrait() {
  await fetchPortrait();
  renderPortraitCard(document.getElementById('dream-portrait-card'));
}

async function refreshJournal() {
  await fetchJournal();
  renderJournal(document.getElementById('dream-journal-entries'));
}

async function refreshAll() {
  await Promise.all([fetchPortrait(), fetchJournal(), fetchStatus(), fetchCheckpoints(), fetchCycles()]);
  renderPortraitCard(document.getElementById('dream-portrait-card'));
  renderJournal(document.getElementById('dream-journal-entries'));
  renderStatus(document.getElementById('dream-status-bar'));
  renderCheckpoints(document.getElementById('dream-checkpoints'));
  renderCycles(document.getElementById('dream-cycles'));
}

// ── Init ──

export async function initDreamJournal() {
  _status = await fetchStatus();
  if (!_status?.enabled) return;
  _portrait = await fetchPortrait();
}

export async function openDreamPanel() {
  await refreshAll();
}
