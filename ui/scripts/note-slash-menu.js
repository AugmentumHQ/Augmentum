/* ==========================================================================
   Note Slash Menu — opens on `/` in the CodeMirror 6 notes editor.

   Format items insert markdown directly via a CM6 transaction. Generate
   items (image / AI / knowledge) strip the `/query` and hand off to a
   callback. Legacy `attach(container, callbacks)` shape preserved so
   existing browse.js wiring keeps working; add `setView(view)` +
   `handleUpdate(update)` for CM6-native integration.

   Wiring (in notes-editor.js):
     import * as SlashMenu from './note-slash-menu.js';
     extensions: [
       ...,
       EditorView.updateListener.of(SlashMenu.handleUpdate),
     ]
     // after `new EditorView(...)`:
     SlashMenu.setView(view);
   ========================================================================== */

let _view = null;                // CodeMirror 6 EditorView
let _menu = null;
let _active = false;
let _slashPos = -1;              // doc position of the `/` that opened the menu
let _filter = '';
let _selectedIdx = 0;
let _filtered = [];
let _callbacks = {};

const ITEMS = [
  // Format — insert markdown (CM6's markdown lang highlights as soon as typed)
  { id: 'h1',      group: 'Format',   label: 'Heading 1',     hint: 'Big title',          icon: 'H1', insert: '# ' },
  { id: 'h2',      group: 'Format',   label: 'Heading 2',     hint: 'Section',            icon: 'H2', insert: '## ' },
  { id: 'h3',      group: 'Format',   label: 'Heading 3',     hint: 'Subsection',         icon: 'H3', insert: '### ' },
  { id: 'quote',   group: 'Format',   label: 'Quote',         hint: 'Blockquote',         icon: '“', insert: '> ' },
  { id: 'code',    group: 'Format',   label: 'Code block',    hint: 'Fenced code',        icon: '〈〉', insert: '```\n\n```' },
  { id: 'bullet',  group: 'Format',   label: 'Bullet list',   hint: 'Unordered',          icon: '•',  insert: '- ' },
  { id: 'number',  group: 'Format',   label: 'Numbered list', hint: 'Ordered',            icon: '1.', insert: '1. ' },
  { id: 'check',   group: 'Format',   label: 'Checklist',     hint: 'Tasks',              icon: '✓', insert: '- [ ] ' },
  { id: 'divider', group: 'Format',   label: 'Divider',       hint: 'Horizontal rule',    icon: '—', insert: '\n\n---\n\n' },
  // Generate — fire callbacks for multi-modal blocks
  { id: 'image',     group: 'Generate', label: 'Image',     hint: 'Generate from prompt', icon: '✨', callback: 'onImage' },
  { id: 'ai',        group: 'Generate', label: 'Ask AI',    hint: 'Draft / continue',     icon: '✨', callback: 'onAi' },
  { id: 'knowledge', group: 'Generate', label: 'Knowledge', hint: 'Cite from packs',      icon: '✨', callback: 'onKnowledge' },
];

// ---------------------------------------------------------------------------
// Public API — `attach` kept for backward-compat (browse.js calls it at
// init before the CM6 view exists). Real wiring happens via setView +
// handleUpdate below.
// ---------------------------------------------------------------------------
export function attach(container, callbacks = {}) {
  _callbacks = callbacks;
}

export function setView(view) {
  _view = view || null;
  if (!view && _active) _close();
}

/**
 * CM6 updateListener callback. Install via:
 *   EditorView.updateListener.of(SlashMenu.handleUpdate)
 * in the extensions array.
 */
export function handleUpdate(update) {
  if (!_view) _view = update.view;

  if (update.docChanged) {
    // Detect a fresh `/` typed at the current cursor position.
    if (!_active) {
      const head = update.state.selection.main.head;
      if (head > 0 && update.state.doc.sliceString(head - 1, head) === '/') {
        const lineFrom = update.state.doc.lineAt(head).from;
        const prevChar = head - 1 > lineFrom
          ? update.state.doc.sliceString(head - 2, head - 1)
          : '';
        if (!prevChar || /\s/.test(prevChar)) {
          _slashPos = head - 1;
          _open();
          return;
        }
      }
    } else {
      _syncFilter();
    }
  } else if (_active && update.selectionSet) {
    // Caret moved without editing — check if we've strayed out of the slash range.
    _syncFilter();
  }
}

/**
 * Explicit open — used by the Mod-/ keybinding in notes-editor.js.
 * Inserts a `/` at the cursor and opens the menu.
 */
export function openAtCursor(view) {
  if (view) _view = view;
  if (!_view) return;
  const pos = _view.state.selection.main.head;
  _slashPos = pos;
  _view.dispatch({
    changes: { from: pos, insert: '/' },
    selection: { anchor: pos + 1 },
    userEvent: 'input.complete',
  });
  _open();
}

export function close() { _close(); }

// ---------------------------------------------------------------------------
// Insertion helpers — used by browse.js (image generation) and any future
// callback that needs to drop content at the caret.
// ---------------------------------------------------------------------------
export function insertTextAtCaret(text) {
  if (!_view || !text) return;
  const pos = _view.state.selection.main.head;
  _view.dispatch({
    changes: { from: pos, insert: text },
    selection: { anchor: pos + text.length },
    userEvent: 'input',
  });
  _view.focus();
}

export function insertImageAtCaret({ url, alt = '', prompt = '' }) {
  const md = `\n![${alt || prompt || 'image'}](${url}${prompt ? ` "${prompt.replace(/"/g, '\\"')}"` : ''})\n`;
  insertTextAtCaret(md);
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------
function _normFilterText(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function _callbackQueryForItem(filter, item) {
  const raw = String(filter || '').trim();
  if (!raw || !item) return '';

  const q = _normFilterText(raw);
  const id = _normFilterText(item.id);
  const label = _normFilterText(item.label);

  // The slash filter is usually just how the user found the menu item:
  // "/ai" -> Ask AI, "/image" -> Image, "/knowledge" -> Knowledge.
  // Do not send that filter as the actual generation prompt/question.
  if (q && (
    id.startsWith(q) ||
    label.startsWith(q) ||
    label.endsWith(q) ||
    label.includes(q)
  )) {
    return '';
  }

  return raw;
}

function _syncFilter() {
  if (!_view || _slashPos < 0) return;
  const head = _view.state.selection.main.head;
  if (head <= _slashPos) { _close(); return; }
  const text = _view.state.doc.sliceString(_slashPos + 1, head);
  if (/\s/.test(text)) { _close(); return; }
  _filter = text;
  _selectedIdx = 0;
  _render();
}

function _open() {
  _active = true;
  _filter = '';
  _selectedIdx = 0;
  _ensureMenu();
  _render();
  _position();
  document.addEventListener('keydown', _onDocKey, true);
  document.addEventListener('mousedown', _onDocClick, true);
}

function _close() {
  if (!_active && !_menu?.classList.contains('hidden')) {
    if (_menu) _menu.classList.add('hidden');
    return;
  }
  _active = false;
  _slashPos = -1;
  _filter = '';
  if (_menu) _menu.classList.add('hidden');
  document.removeEventListener('keydown', _onDocKey, true);
  document.removeEventListener('mousedown', _onDocClick, true);
}

function _onDocKey(e) {
  if (!_active) return;
  if (e.key === 'ArrowDown') {
    e.preventDefault(); e.stopPropagation();
    _selectedIdx = (_selectedIdx + 1) % Math.max(1, _filtered.length);
    _renderSelection();
  } else if (e.key === 'ArrowUp') {
    e.preventDefault(); e.stopPropagation();
    _selectedIdx = (_selectedIdx - 1 + _filtered.length) % Math.max(1, _filtered.length);
    _renderSelection();
  } else if (e.key === 'Enter' || e.key === 'Tab') {
    if (_filtered[_selectedIdx]) {
      e.preventDefault(); e.stopPropagation();
      _commit(_filtered[_selectedIdx]);
    }
  } else if (e.key === 'Escape') {
    e.preventDefault(); e.stopPropagation();
    _close();
    _view?.focus();
  }
}

function _onDocClick(e) {
  if (!_menu || _menu.contains(e.target)) return;
  _close();
}

function _ensureMenu() {
  if (_menu) return;
  _menu = document.createElement('div');
  _menu.className = 'note-slash-menu';
  _menu.setAttribute('role', 'listbox');
  _menu.setAttribute('aria-label', 'Insert');
  document.body.appendChild(_menu);
}

function _render() {
  if (!_menu) return;
  const q = _filter.toLowerCase();
  _filtered = q
    ? ITEMS.filter(it => it.label.toLowerCase().includes(q) || it.id.includes(q))
    : ITEMS.slice();
  if (!_filtered.length) _filtered = ITEMS.slice();
  if (_selectedIdx >= _filtered.length) _selectedIdx = 0;

  _menu.classList.remove('hidden');
  _menu.innerHTML = '';
  let lastGroup = null;
  _filtered.forEach((item, i) => {
    if (item.group !== lastGroup) {
      const g = document.createElement('div');
      g.className = 'note-slash-group';
      g.textContent = item.group;
      _menu.appendChild(g);
      lastGroup = item.group;
    }
    const btn = document.createElement('button');
    btn.className = 'note-slash-item' + (i === _selectedIdx ? ' selected' : '');
    btn.dataset.idx = String(i);
    btn.innerHTML = `
      <span class="note-slash-icon">${item.icon}</span>
      <span class="note-slash-label">${item.label}</span>
      <span class="note-slash-hint">${item.hint}</span>
    `;
    btn.addEventListener('mousedown', (e) => {
      e.preventDefault();
      _commit(item);
    });
    btn.addEventListener('mouseenter', () => {
      _selectedIdx = i;
      _renderSelection();
    });
    _menu.appendChild(btn);
  });
}

function _renderSelection() {
  if (!_menu) return;
  _menu.querySelectorAll('.note-slash-item').forEach((el, i) => {
    el.classList.toggle('selected', i === _selectedIdx);
    if (i === _selectedIdx) el.scrollIntoView({ block: 'nearest' });
  });
}

function _position() {
  if (!_view || _slashPos < 0 || !_menu) return;
  const coords = _view.coordsAtPos(_slashPos);
  if (!coords) return;
  const menuH = Math.min(320, _menu.offsetHeight || 320);
  const menuW = Math.min(320, _menu.offsetWidth || 280);
  const top = coords.bottom + 6;
  const flipUp = top + menuH > window.innerHeight;
  const left = Math.min(coords.left, window.innerWidth - menuW - 12);
  _menu.style.position = 'fixed';
  _menu.style.left = `${Math.max(8, left)}px`;
  _menu.style.top = flipUp
    ? `${Math.max(8, coords.top - menuH - 6)}px`
    : `${top}px`;
}

function _commit(item) {
  if (!item || !_view) { _close(); return; }
  const head = _view.state.selection.main.head;
  const from = _slashPos;
  const to = head;
  const userQuery = _callbackQueryForItem(_filter, item);
  _close();

  if (item.insert != null) {
    _view.dispatch({
      changes: { from, to, insert: item.insert },
      selection: { anchor: from + item.insert.length },
      userEvent: 'input.complete',
    });
    _view.focus();
  } else if (item.callback && typeof _callbacks[item.callback] === 'function') {
    // Strip the /query text so the callback inserts cleanly at the caret.
    _view.dispatch({
      changes: { from, to, insert: '' },
      selection: { anchor: from },
      userEvent: 'delete.complete',
    });
    _view.focus();
    _callbacks[item.callback](userQuery);
  }
}
