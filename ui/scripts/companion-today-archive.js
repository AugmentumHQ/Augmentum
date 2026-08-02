/**
 * companion-today-archive.js — modal listing prior days' reflections.
 *
 * Lazy-loaded by companion-notes.js when the user clicks "See archive"
 * in the Today zone. Keeps the drawer's first-load footprint small.
 *
 * Renders last 30 days, newest first. Each entry shows date + the
 * reflection prose with [kind:id] citations rewritten to spans (visual
 * only — clicking a citation in the archive opens a focused view of
 * the source, no in-place actions since archived days are immutable).
 */

const POLL_LIMIT = 30;

let _modal = null;

function _escapeHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/`/g, '&#96;')
    .replace(/\$\{/g, '&#36;{');
}

function _renderCitations(text, refs) {
  const realRefs = new Set(
    (refs || [])
      .filter((r) => r && r.kind && Number.isInteger(r.id))
      .map((r) => `${r.kind}:${r.id}`),
  );
  const parts = [];
  const re = /\[(note|wondering|journal):(\d+)\]/g;
  let cursor = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > cursor) parts.push(_escapeHtml(text.slice(cursor, m.index)));
    const kind = m[1];
    const id = m[2];
    if (realRefs.has(`${kind}:${id}`)) {
      parts.push(
        `<span class="companion-today-citation archive" data-kind="${_escapeHtml(kind)}" `
        + `data-id="${_escapeHtml(id)}">${_escapeHtml(`[${kind}:${id}]`)}</span>`,
      );
    }
    cursor = m.index + m[0].length;
  }
  if (cursor < text.length) parts.push(_escapeHtml(text.slice(cursor)));
  return parts.join('');
}

function _formatDate(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso + 'T00:00:00');
    return d.toLocaleDateString(undefined, {
      weekday: 'short', month: 'short', day: 'numeric', year: 'numeric',
    });
  } catch (_) { return iso; }
}

async function _fetchArchive() {
  try {
    const resp = await fetch(`/api/companion/today/archive?limit=${POLL_LIMIT}`,
                             { credentials: 'same-origin' });
    if (!resp.ok) return [];
    const data = await resp.json();
    return Array.isArray(data.archive) ? data.archive : [];
  } catch (_) { return []; }
}

function _buildModal() {
  if (_modal) return _modal;
  const root = document.createElement('div');
  root.className = 'companion-today-archive-modal hidden';
  root.innerHTML = `
    <div class="companion-today-archive-backdrop"></div>
    <div class="companion-today-archive-panel" role="dialog" aria-label="Reflection archive">
      <header class="companion-today-archive-header">
        <span class="companion-today-archive-title">Reflection archive</span>
        <button type="button" class="companion-today-archive-close" aria-label="Close">×</button>
      </header>
      <div class="companion-today-archive-body">
        <div class="companion-today-archive-loading">Loading…</div>
      </div>
    </div>
  `;
  document.body.appendChild(root);
  root.querySelector('.companion-today-archive-close')
    .addEventListener('click', close);
  root.querySelector('.companion-today-archive-backdrop')
    .addEventListener('click', close);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && _modal && !_modal.classList.contains('hidden')) {
      close();
    }
  });
  _modal = root;
  return root;
}

function _renderEntries(entries) {
  const body = _modal.querySelector('.companion-today-archive-body');
  if (!body) return;
  if (!entries.length) {
    body.innerHTML = `<p class="companion-today-archive-empty">No reflections yet.</p>`;
    return;
  }
  body.innerHTML = entries.map((entry) => {
    const dateLabel = _escapeHtml(_formatDate(entry.date));
    const prose = entry.content
      ? _renderCitations(entry.content, entry.source_refs || [])
      : `<em class="companion-today-archive-quiet">Stayed in the background.</em>`;
    return `
      <article class="companion-today-archive-entry">
        <h4 class="companion-today-archive-date">${dateLabel}</h4>
        <p class="companion-today-archive-prose">${prose}</p>
      </article>
    `;
  }).join('\n');
}

export async function open() {
  _buildModal();
  _modal.classList.remove('hidden');
  const entries = await _fetchArchive();
  _renderEntries(entries);
}

export function close() {
  if (_modal) _modal.classList.add('hidden');
}
