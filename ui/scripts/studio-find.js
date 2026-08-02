/**
 * Shared Find & Replace modal for Studio editors.
 *
 * One call site per editor: register a "provider" describing how to read
 * the text, how to apply a replacement, and (optionally) how to focus a
 * match visually. The modal handles its own DOM — editors only supply the
 * data adapter.
 *
 * Usage:
 *   import { openFind, registerProvider } from './studio-find.js';
 *   registerProvider('doc', { getMatches, applyReplace, focusMatch });
 *   // Ctrl+F → openFind('doc');   Ctrl+H → openFind('doc', { replace: true });
 *
 * Providers return an array of { id, text, context? } records. `id` is
 * opaque to the modal — it's passed back verbatim in applyReplace and
 * focusMatch so editors can store whatever they need (cell coords, DOM
 * node ref, slide index, etc.).
 */

const _providers = new Map();

/**
 * Register a provider for a scope name. Calling openFind(scope) resolves
 * the provider by name. Late registration is fine — editors register on
 * mount and the modal reads the registry at open-time.
 */
export function registerProvider(scope, provider) {
  _providers.set(scope, provider);
}

export function unregisterProvider(scope) {
  _providers.delete(scope);
}

let _modal = null;
let _state = null;

export function openFind(scope, opts = {}) {
  const provider = _providers.get(scope);
  if (!provider) return;
  _ensureModal();
  _state = {
    scope,
    provider,
    query: '',
    replace: '',
    caseSensitive: false,
    wholeWord: false,
    mode: opts.replace ? 'replace' : 'find',
    matches: [],
    index: 0,
  };
  _modal.classList.add('is-open');
  _modal.dataset.mode = _state.mode;
  const q = /** @type {HTMLInputElement} */ (_modal.querySelector('[data-find-input]'));
  q?.focus();
  q?.select();
}

export function closeFind() {
  if (_modal) _modal.classList.remove('is-open');
  _state = null;
}

export function isFindOpen() { return !!_modal?.classList.contains('is-open'); }

function _ensureModal() {
  if (_modal) return;
  _modal = document.createElement('div');
  _modal.className = 'studio-find-modal';
  _modal.innerHTML = `
    <div class="studio-find-row">
      <span class="studio-find-icon" aria-hidden="true">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
      </span>
      <input type="text" class="studio-find-input" data-find-input placeholder="Find" autocomplete="off" spellcheck="false">
      <span class="studio-find-counter" data-find-counter>0 / 0</span>
      <button class="studio-find-icon-btn" data-find-action="prev" title="Previous (Shift+Enter)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"/></svg>
      </button>
      <button class="studio-find-icon-btn" data-find-action="next" title="Next (Enter)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <button class="studio-find-icon-btn" data-find-action="toggle-replace" title="Replace (Ctrl+H)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 4l6 6-6 6M20 10H4"/></svg>
      </button>
      <button class="studio-find-icon-btn" data-find-action="case" title="Match case">Aa</button>
      <button class="studio-find-icon-btn" data-find-action="word" title="Whole word">W</button>
      <button class="studio-find-icon-btn studio-find-close" data-find-action="close" title="Close (Esc)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
    <div class="studio-find-row studio-find-replace-row">
      <span class="studio-find-icon" aria-hidden="true">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 7l-3-3h-4L8 10l6 6 6-6V7h-3m-3 9l-6-6-4 4v3h3l6-6"/></svg>
      </span>
      <input type="text" class="studio-find-input" data-replace-input placeholder="Replace with" autocomplete="off" spellcheck="false">
      <button class="studio-find-btn" data-find-action="replace-one">Replace</button>
      <button class="studio-find-btn" data-find-action="replace-all">All</button>
    </div>
  `;
  document.body.appendChild(_modal);

  const qInput = /** @type {HTMLInputElement} */ (_modal.querySelector('[data-find-input]'));
  const rInput = /** @type {HTMLInputElement} */ (_modal.querySelector('[data-replace-input]'));

  qInput.addEventListener('input', () => {
    if (!_state) return;
    _state.query = qInput.value;
    _recomputeMatches();
    _focusCurrent();
  });
  rInput.addEventListener('input', () => {
    if (!_state) return;
    _state.replace = rInput.value;
  });

  qInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      _step(e.shiftKey ? -1 : 1);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      closeFind();
    }
  });
  rInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (e.shiftKey) _replaceAll(); else _replaceOne();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      closeFind();
    }
  });

  _modal.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-find-action]');
    if (!btn) return;
    const a = btn.dataset.findAction;
    if (a === 'close')           closeFind();
    else if (a === 'next')       _step(1);
    else if (a === 'prev')       _step(-1);
    else if (a === 'toggle-replace') _toggleReplace();
    else if (a === 'case')       { _state.caseSensitive = !_state.caseSensitive; btn.classList.toggle('active', _state.caseSensitive); _recomputeMatches(); _focusCurrent(); }
    else if (a === 'word')       { _state.wholeWord     = !_state.wholeWord;     btn.classList.toggle('active', _state.wholeWord);     _recomputeMatches(); _focusCurrent(); }
    else if (a === 'replace-one') _replaceOne();
    else if (a === 'replace-all') _replaceAll();
  });
}

function _toggleReplace() {
  if (!_state || !_modal) return;
  _state.mode = _state.mode === 'replace' ? 'find' : 'replace';
  _modal.dataset.mode = _state.mode;
}

// Build a regex honoring case-sensitive + whole-word toggles. Escapes the
// query so special regex chars behave as literals.
function _buildRegex() {
  if (!_state?.query) return null;
  let pattern = _state.query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  if (_state.wholeWord) pattern = `\\b${pattern}\\b`;
  const flags = 'g' + (_state.caseSensitive ? '' : 'i');
  try { return new RegExp(pattern, flags); } catch { return null; }
}

function _recomputeMatches() {
  if (!_state) return;
  if (!_state.query) { _state.matches = []; _state.index = 0; _updateCounter(); return; }
  const re = _buildRegex();
  if (!re) { _state.matches = []; _updateCounter(); return; }
  const records = _state.provider.getMatches(re) || [];
  _state.matches = records;
  _state.index = records.length ? Math.min(_state.index, records.length - 1) : 0;
  _updateCounter();
}

function _updateCounter() {
  const c = _modal?.querySelector('[data-find-counter]');
  if (!c) return;
  c.textContent = _state.matches.length
    ? `${_state.index + 1} / ${_state.matches.length}`
    : _state.query ? 'No matches' : '';
}

function _step(delta) {
  if (!_state?.matches.length) return;
  _state.index = (_state.index + delta + _state.matches.length) % _state.matches.length;
  _updateCounter();
  _focusCurrent();
}

function _focusCurrent() {
  if (!_state?.matches.length) return;
  const m = _state.matches[_state.index];
  _state.provider.focusMatch?.(m);
}

function _replaceOne() {
  if (!_state?.matches.length) return;
  const m = _state.matches[_state.index];
  _state.provider.applyReplace?.(m, _state.replace);
  // Replacement shifts all later offsets — easiest to recompute fully.
  _recomputeMatches();
  _focusCurrent();
}

function _replaceAll() {
  if (!_state?.matches.length) return;
  // Apply from last to first so earlier-offset replacements don't invalidate
  // the indices of later matches.
  const all = [..._state.matches].reverse();
  for (const m of all) _state.provider.applyReplace?.(m, _state.replace);
  _recomputeMatches();
  _focusCurrent();
}

/**
 * Build a simple-text provider around any HTMLElement whose innerText should
 * be searchable. Works for contenteditable surfaces (Milkdown's .ProseMirror,
 * textarea content mirrored from a DOM string, etc.) with a `commit` hook
 * the editor supplies so it can re-sync state after a replacement.
 *
 * Returns `{ getMatches, focusMatch, applyReplace }`. Wire it to the modal
 * via `registerProvider(scope, proseMirrorProvider(el, commit))`.
 */
export function contentEditableProvider(rootGetter, commit) {
  return {
    getMatches(re) {
      const root = rootGetter();
      if (!root) return [];
      const text = root.innerText || '';
      const out = [];
      let m;
      while ((m = re.exec(text)) !== null) {
        if (m[0].length === 0) { re.lastIndex++; continue; }
        out.push({ start: m.index, end: m.index + m[0].length, text: m[0] });
      }
      return out;
    },
    focusMatch(match) {
      const root = rootGetter();
      if (!root) return;
      // Build a range that spans [start, end) across all text nodes under
      // root. Walk the tree accumulating lengths until we hit the offsets.
      const range = _rangeFromOffsets(root, match.start, match.end);
      if (!range) return;
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      // Scroll the start of the match into view without jumping the whole page.
      const rect = range.getBoundingClientRect();
      if (rect.bottom > window.innerHeight - 80 || rect.top < 80) {
        const anchor = range.startContainer.parentElement;
        anchor?.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    },
    applyReplace(match, replacement) {
      const root = rootGetter();
      if (!root) return;
      const range = _rangeFromOffsets(root, match.start, match.end);
      if (!range) return;
      range.deleteContents();
      range.insertNode(document.createTextNode(replacement));
      commit?.();
    },
  };
}

function _rangeFromOffsets(root, start, end) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let offset = 0;
  let startNode = null, startOff = 0, endNode = null, endOff = 0;
  let n;
  while ((n = walker.nextNode())) {
    const len = n.nodeValue.length;
    if (startNode === null && offset + len >= start) {
      startNode = n;
      startOff = start - offset;
    }
    if (offset + len >= end) {
      endNode = n;
      endOff = end - offset;
      break;
    }
    offset += len;
  }
  if (!startNode || !endNode) return null;
  const range = document.createRange();
  range.setStart(startNode, startOff);
  range.setEnd(endNode, endOff);
  return range;
}

// Close on Escape even when the editor itself has focus.
if (typeof window !== 'undefined') {
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && isFindOpen()) {
      e.preventDefault();
      closeFind();
    }
  });
}
