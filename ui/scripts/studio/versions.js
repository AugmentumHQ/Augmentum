/* ==========================================================================
   Studio version history — right-edge drawer + restore flow
   --------------------------------------------------------------------------
   Lists manual-save snapshots for the active artifact, newest first.
   Per-version "Restore" → confirm modal → POST restore endpoint → toast +
   close drawer + reload artifact.

   Restore is itself reversible — the backend snapshots the current state
   before overwriting, so the user can open the drawer again and restore
   that auto-snapshot to undo. This is why we don't need a per-version
   preview pane in v1: the cost of "tried it, didn't like it" is one more
   click.

   Public:
     openVersionsDrawer({ artifactId, onRestored }) → opens panel
     closeVersionsDrawer()                          → idempotent close
   ========================================================================== */

import { escapeHtml, showToast } from '../app.js';

let _activeDrawer = null;

/**
 * @param {object} opts
 * @param {string} opts.artifactId
 * @param {() => void} [opts.onRestored] - called after a successful restore
 *   so the host can refresh its editor view from the new source.
 */
export async function openVersionsDrawer({ artifactId, onRestored }) {
  if (!artifactId) {
    showToast('No artifact loaded.', 'warn');
    return;
  }
  closeVersionsDrawer();

  const overlay = document.createElement('div');
  overlay.className = 'studio-versions-overlay';
  overlay.innerHTML = `
    <div class="studio-versions-panel" role="dialog" aria-label="Version history">
      <div class="studio-versions-head">
        <span class="studio-versions-title">Version history</span>
        <button type="button" class="studio-versions-close" aria-label="Close" title="Close (Esc)">&times;</button>
      </div>
      <div class="studio-versions-sub">Only manual saves leave a version. Restoring is reversible.</div>
      <div class="studio-versions-list" id="studio-versions-list">
        <div class="studio-versions-loading">Loading…</div>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  _activeDrawer = overlay;

  // Close handlers
  const close = () => closeVersionsDrawer();
  overlay.querySelector('.studio-versions-close').addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  const esc = (e) => { if (e.key === 'Escape') close(); };
  document.addEventListener('keydown', esc);
  overlay._esc = esc;

  // Fetch + render
  await _renderVersions(overlay, artifactId, onRestored);
}

export function closeVersionsDrawer() {
  if (!_activeDrawer) return;
  if (_activeDrawer._esc) document.removeEventListener('keydown', _activeDrawer._esc);
  _activeDrawer.remove();
  _activeDrawer = null;
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

async function _renderVersions(overlay, artifactId, onRestored) {
  const listEl = overlay.querySelector('#studio-versions-list');
  let versions;
  try {
    const resp = await fetch(
      `/api/studio/${encodeURIComponent(artifactId)}/versions`,
      { credentials: 'same-origin' },
    );
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const body = await resp.json();
    versions = Array.isArray(body.versions) ? body.versions : [];
  } catch (err) {
    listEl.innerHTML = `
      <div class="studio-versions-empty">
        <div class="studio-versions-empty-title">Couldn’t load history</div>
        <div class="studio-versions-empty-sub">${escapeHtml(err.message || String(err))}</div>
      </div>
    `;
    return;
  }

  if (!versions.length) {
    listEl.innerHTML = `
      <div class="studio-versions-empty">
        <div class="studio-versions-empty-title">No saved versions yet</div>
        <div class="studio-versions-empty-sub">Click Save (or Ctrl+S) to leave a version behind.</div>
      </div>
    `;
    return;
  }

  listEl.innerHTML = versions.map((v) => _renderRow(v)).join('');
  listEl.querySelectorAll('[data-version-restore]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const versionId = btn.dataset.versionRestore;
      const versionIndex = btn.dataset.versionIndex;
      const ok = window.confirm(
        `Restore Version ${versionIndex}? Your current state will be saved as a new version so you can undo this.`,
      );
      if (!ok) return;
      await _doRestore(artifactId, versionId, versionIndex, onRestored);
    });
  });
}

function _renderRow(v) {
  const idx = Number(v.version_index || 0);
  const label = v.label ? String(v.label) : 'Manual save';
  const rel = _relativeTime(v.created_at);
  const absolute = _formatAbsolute(v.created_at);
  return `
    <div class="studio-versions-row" data-version-row="${escapeHtml(String(v.id))}">
      <div class="studio-versions-row-meta">
        <div class="studio-versions-row-title">Version ${idx}</div>
        <div class="studio-versions-row-time" title="${escapeHtml(absolute)}">${escapeHtml(rel)}</div>
        <div class="studio-versions-row-label">${escapeHtml(label)}</div>
      </div>
      <button type="button" class="studio-versions-restore-btn"
              data-version-restore="${escapeHtml(String(v.id))}"
              data-version-index="${idx}">
        Restore
      </button>
    </div>
  `;
}

async function _doRestore(artifactId, versionId, versionIndex, onRestored) {
  const row = _activeDrawer?.querySelector(`[data-version-row="${CSS.escape(versionId)}"]`);
  const btn = row?.querySelector('.studio-versions-restore-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Restoring…'; }
  try {
    const resp = await fetch(
      `/api/studio/${encodeURIComponent(artifactId)}/restore-version/${encodeURIComponent(versionId)}`,
      { method: 'POST', credentials: 'same-origin' },
    );
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try { const j = await resp.json(); detail = j.error || j.detail || detail; }
      catch { /* keep status */ }
      throw new Error(detail);
    }
    const data = await resp.json();
    showToast(`Restored Version ${versionIndex}. Undo lives in this drawer.`, 'success');
    closeVersionsDrawer();
    if (typeof onRestored === 'function') {
      try { onRestored(data); }
      catch (err) { console.warn('[studio] onRestored handler threw:', err); }
    }
  } catch (err) {
    showToast(`Couldn’t restore: ${err.message || err}`, 'error');
    if (btn) { btn.disabled = false; btn.textContent = 'Restore'; }
  }
}

// ---------------------------------------------------------------------------
// Time formatting — server returns "YYYY-MM-DD HH:MM:SS" in UTC.
// We render relative ("2 min ago") with absolute as a tooltip.
// ---------------------------------------------------------------------------

function _parseServerTime(s) {
  if (!s) return null;
  // SQLite datetime('now') returns "YYYY-MM-DD HH:MM:SS" with no tz; treat as UTC.
  const iso = String(s).replace(' ', 'T') + 'Z';
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : null;
}

function _relativeTime(s) {
  const t = _parseServerTime(s);
  if (t == null) return '—';
  const delta = Date.now() - t;
  if (delta < 5_000) return 'just now';
  if (delta < 60_000) return `${Math.round(delta / 1000)}s ago`;
  if (delta < 3_600_000) return `${Math.round(delta / 60_000)}m ago`;
  if (delta < 86_400_000) return `${Math.round(delta / 3_600_000)}h ago`;
  return `${Math.round(delta / 86_400_000)}d ago`;
}

function _formatAbsolute(s) {
  const t = _parseServerTime(s);
  if (t == null) return String(s || '');
  try { return new Date(t).toLocaleString(); }
  catch { return String(s); }
}
