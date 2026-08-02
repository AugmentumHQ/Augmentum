/**
 * note-margin.js — Margin Intelligence UI
 *
 * Scans note content for annotations (lorebook, memory, related notes)
 * and renders them as underlined terms with margin/sheet annotations.
 */

const SOURCE_COLORS = {
  lorebook: { border: 'rgba(234,179,8,0.3)', underline: 'rgba(234,179,8,0.4)', label: 'Lorebook' },
  memory:   { border: 'rgba(99,102,241,0.3)', underline: 'rgba(99,102,241,0.4)', label: 'Memory' },
  related_note: { border: 'rgba(168,85,247,0.3)', underline: 'rgba(168,85,247,0.4)', label: 'Related Note' },
};

let _marginCol = null;
let _noteId = null;
let _debounceTimer = null;
let _annotations = [];       // active margin annotations
let _sheet = null;           // mobile bottom sheet element
const MAX_MARGIN = 3;

/* ── public API ── */

export function init(marginColumnEl) {
  _marginCol = marginColumnEl;
}

export function setNoteId(id) {
  _noteId = id;
}

export function scheduleRescan(content) {
  clearTimeout(_debounceTimer);
  _debounceTimer = setTimeout(() => _scan(content), 800);
}

export function clear() {
  clearTimeout(_debounceTimer);
  _removeAllUnderlines();
  _removeAllAnnotations();
  _dismissSheet();
}

/* ── scan ── */

async function _scan(content) {
  if (!_noteId) return;
  try {
    const res = await fetch(`/api/notes/${_noteId}/scan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    });
    if (!res.ok) return;
    const data = await res.json();
    _applyAnnotations(data.annotations || []);
  } catch { /* network failures are non-fatal */ }
}

/* ── underline injection ── */

function _applyAnnotations(annotations) {
  _removeAllUnderlines();
  _removeAllAnnotations();

  const pm = document.querySelector('#note-editor-body .ProseMirror');
  if (!pm) return;

  const matched = new Set();

  for (const ann of annotations) {
    if (matched.has(ann.term)) continue;
    const color = SOURCE_COLORS[ann.source];
    if (!color) continue;

    const found = _wrapFirstOccurrence(pm, ann.term, color.underline);
    if (found) {
      matched.add(ann.term);
      found.addEventListener('click', () => _showAnnotation(found, ann, color));
    }
  }
}

/**
 * Walk text nodes in `root`, find first occurrence of `term`,
 * wrap it in a styled span, return the span (or null).
 */
function _wrapFirstOccurrence(root, term, underlineColor) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const termLower = term.toLowerCase();

  while (walker.nextNode()) {
    const node = walker.currentNode;
    const idx = node.textContent.toLowerCase().indexOf(termLower);
    if (idx === -1) continue;

    const range = document.createRange();
    range.setStart(node, idx);
    range.setEnd(node, idx + term.length);

    const span = document.createElement('span');
    span.className = 'note-intel-term';
    span.style.textDecoration = 'underline dotted';
    span.style.textDecorationColor = underlineColor;
    span.style.cursor = 'pointer';
    range.surroundContents(span);
    return span;
  }
  return null;
}

/* ── annotation display ── */

function _showAnnotation(termEl, ann, color) {
  const isDesktop = window.innerWidth > 1080;
  if (isDesktop) {
    _showMarginAnnotation(termEl, ann, color);
  } else {
    _showBottomSheet(ann, color);
  }
}

function _showMarginAnnotation(termEl, ann, color) {
  if (!_marginCol) return;

  // enforce max 3
  while (_annotations.length >= MAX_MARGIN) {
    const oldest = _annotations.shift();
    oldest.remove();
  }

  const rect = termEl.getBoundingClientRect();
  const colRect = _marginCol.getBoundingClientRect();
  const top = rect.top - colRect.top + _marginCol.scrollTop;

  const el = document.createElement('div');
  el.className = 'note-margin-annotation';
  el.style.borderLeftColor = color.border;
  el.style.background = color.border.replace(/[\d.]+\)$/, '0.05)');
  el.style.top = `${top}px`;

  el.innerHTML = `
    <div class="note-margin-label" style="color:${color.underline}">${color.label}</div>
    <div class="note-margin-content">${_esc(ann.content)}</div>
    <div class="note-margin-actions">
      <button class="note-margin-action" data-action="dismiss">Dismiss</button>
    </div>`;

  el.querySelector('[data-action="dismiss"]').addEventListener('click', () => {
    const i = _annotations.indexOf(el);
    if (i !== -1) _annotations.splice(i, 1);
    el.remove();
  });

  _marginCol.appendChild(el);
  _annotations.push(el);
}

function _showBottomSheet(ann, color) {
  _dismissSheet();

  const el = document.createElement('div');
  el.className = 'note-margin-sheet';
  el.innerHTML = `
    <div class="note-margin-sheet-handle"></div>
    <div class="note-margin-label" style="color:${color.underline}">${color.label}</div>
    <div class="note-margin-content">${_esc(ann.content)}</div>
    <button class="note-margin-sheet-close">Dismiss</button>`;

  el.querySelector('.note-margin-sheet-close').addEventListener('click', () => _dismissSheet());
  document.body.appendChild(el);
  _sheet = el;

  // trigger transition
  requestAnimationFrame(() => el.classList.add('visible'));
}

function _dismissSheet() {
  if (!_sheet) return;
  _sheet.classList.remove('visible');
  const s = _sheet;
  setTimeout(() => s.remove(), 260);
  _sheet = null;
}

/* ── cleanup ── */

function _removeAllUnderlines() {
  document.querySelectorAll('.note-intel-term').forEach(span => {
    const parent = span.parentNode;
    while (span.firstChild) parent.insertBefore(span.firstChild, span);
    parent.removeChild(span);
    parent.normalize();
  });
}

function _removeAllAnnotations() {
  _annotations.forEach(el => el.remove());
  _annotations = [];
  _dismissSheet();
}

/* ── util ── */

function _esc(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}
