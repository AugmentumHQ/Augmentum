/* ==========================================================================
   ToolCard — typed presentation envelope for tool results.

   Backend tools may return ``ToolResult.card`` (see augmentum/tools/base.py)
   instead of dumping a markdown blob. This module renders those cards into
   HTML and dispatches the ``actions`` clicks as global custom events that
   existing editors (Artifact Studio, Image Studio, etc.) listen for.

   Card schema is documented in base.py; see ``make_artifact_card``.
   ========================================================================== */

import { escapeHtml } from '../app.js';

const _ICON = {
  eye:      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
  edit:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>',
  download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
  play:     '<svg viewBox="0 0 24 24" fill="currentColor" stroke="none" width="14" height="14"><polygon points="6 4 20 12 6 20 6 4"/></svg>',
};

const _KIND_ICON = {
  artifact:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
  image:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
  search:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
  article:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="13" y2="17"/></svg>',
  code_exec: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
  // Language-partner card kinds — emitted by tools in
  // augmentum/tools/language_partner.py
  vocab_lookup:      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
  vocab_breakdown:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/></svg>',
  drill_suggestion:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8" fill="currentColor" stroke="none"/></svg>',
};

function _bytes(n) {
  if (!n) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function _renderArtifactPreview(p) {
  const sections = p.chapters || p.sections || p.slides || [];
  const sheets = p.sheets || [];
  let html = '';

  if (sections.length) {
    const lim = sections.slice(0, 6);
    html += '<ul class="tool-card-list">';
    for (const s of lim) {
      html += `<li>${escapeHtml(s.heading || '')}</li>`;
    }
    html += '</ul>';
    if (sections.length > 6) {
      html += `<div class="tool-card-more">+${sections.length - 6} more</div>`;
    }
  } else if (sheets.length) {
    html += '<ul class="tool-card-list">';
    for (const sh of sheets) {
      html += `<li>${escapeHtml(sh.name || '')} <span class="tool-card-dim">(${sh.row_count || 0} rows)</span></li>`;
    }
    html += '</ul>';
  }

  const meta = [];
  if (p.format) meta.push(p.format.toUpperCase());
  if (p.size_bytes) meta.push(_bytes(p.size_bytes));
  if (p.image_count) meta.push(`${p.image_count} image${p.image_count !== 1 ? 's' : ''}`);
  if (meta.length) {
    html += `<div class="tool-card-meta">${meta.map(escapeHtml).join(' · ')}</div>`;
  }
  return html;
}

function _renderImagePreview(p) {
  const url = p.image_url || '';
  if (!url) return '';
  return `<div class="tool-card-image"><img src="${escapeHtml(url)}" alt=""></div>`;
}

function _renderActions(actions = []) {
  if (!actions.length) return '';
  let html = '<div class="tool-card-actions">';
  for (const a of actions) {
    const icon = _ICON[a.icon] || '';
    if (a.href) {
      html += `<a class="tool-card-btn" href="${escapeHtml(a.href)}" download>${icon}<span>${escapeHtml(a.label)}</span></a>`;
    } else {
      const payload = a.payload ? escapeHtml(JSON.stringify(a.payload)) : '';
      html += `<button class="tool-card-btn" data-event="${escapeHtml(a.event || '')}" data-payload="${payload}">${icon}<span>${escapeHtml(a.label)}</span></button>`;
    }
  }
  html += '</div>';
  return html;
}

/**
 * Render a card to an HTML string. Returns '' when card is unusable.
 * @param {object} card — backend ToolCard envelope
 * @param {object} [opts]
 * @param {boolean} [opts.compact] — drop preview body, header + actions only
 */
export function renderToolCard(card, opts = {}) {
  if (!card || !card.kind) return '';
  const kindIcon = _KIND_ICON[card.kind] || _KIND_ICON.artifact;
  const kindLabel = String(card.kind || 'result').replace(/_/g, ' ');
  let body = '';
  if (!opts.compact && card.preview) {
    if (card.kind === 'image') body = _renderImagePreview(card.preview);
    else if (card.kind === 'artifact') body = _renderArtifactPreview(card.preview);
  }
  const summary = card.summary
    ? `<div class="tool-card-summary">${escapeHtml(card.summary)}</div>`
    : '';
  return `
    <div class="tool-card tool-card--typed tool-card--${escapeHtml(card.kind)}" data-artifact-id="${escapeHtml(card.artifact_id || '')}">
      <div class="tool-card-head">
        <span class="tool-card-icon">${kindIcon}</span>
        <div class="tool-card-title-block">
          <div class="tool-card-kicker">${escapeHtml(kindLabel)}</div>
          <div class="tool-card-title">${escapeHtml(card.title || '')}</div>
          ${card.subtitle ? `<div class="tool-card-subtitle">${escapeHtml(card.subtitle)}</div>` : ''}
        </div>
      </div>
      ${body ? `<div class="tool-card-body">${body}</div>` : ''}
      ${summary}
      ${_renderActions(card.actions)}
    </div>
  `;
}

// Global click delegator for action buttons. Dispatches the named custom
// event with the payload so feature modules (artifact studio, image studio)
// can listen without coupling to this module.
let _wired = false;
export function ensureToolCardActionsWired() {
  if (_wired) return;
  _wired = true;
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.tool-card-btn[data-event]');
    if (!btn) return;
    const eventName = btn.dataset.event;
    if (!eventName) return;
    let payload = {};
    try {
      payload = btn.dataset.payload ? JSON.parse(btn.dataset.payload) : {};
    } catch { /* ignore malformed payload */ }
    document.dispatchEvent(new CustomEvent(eventName, { detail: payload }));
  });
}
