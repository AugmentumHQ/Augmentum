/**
 * coder-search.js — Workspace search pane for coder mode's Files panel.
 *
 * Two legs, one pane:
 *   • Text  — live literal/regex grep over the working tree via
 *             /api/coder/workspaces/{id}/search-text. Exact, never
 *             stale: a file the agent wrote a second ago is findable.
 *   • Semantic — meaning-based lookup via the codebase index
 *             (/api/coder/search/{id}). Finds "auth handling" when the
 *             code says `verify_session`. Requires the index; if it's
 *             not built we surface a one-click build prompt rather than
 *             an empty result.
 *
 * Results are grouped by file, collapsible, with highlighted match
 * spans. Clicking a match opens the editor at that line. The pane
 * owns its own DOM + state; it reaches back into coder.js only through
 * the two callbacks it can't own — the active workspace id and the
 * open-file-at-line action.
 *
 * Performance discipline (this panel renders while agent runs stream):
 *   • Input is debounced (250ms); Enter searches immediately.
 *   • Each query is tagged with a monotonic token; stale responses
 *     (user typed again before the fetch returned) are dropped.
 *   • Results render into a DocumentFragment in one DOM write.
 *   • Match clicks use a single delegated listener.
 */
import { escapeHtml } from './app.js';

let _cfg = null;          // { getWorkspaceId, openResult, buildIndex }
let _els = null;          // cached DOM refs
let _open = false;
let _mode = 'text';       // 'text' | 'semantic'
const _opts = { regex: false, case: false };
let _debounce = null;
let _queryToken = 0;
let _lastQuery = '';

// ---------------------------------------------------------------------------
// Init / lifecycle
// ---------------------------------------------------------------------------

/**
 * @param {object} cfg
 * @param {() => string} cfg.getWorkspaceId  — active workspace id ('' if none).
 * @param {(hit: {path,line,spans,name}) => void} cfg.openResult — open file at line.
 * @param {() => Promise<boolean>} cfg.buildIndex — kick a semantic index build.
 */
export function initCoderSearch(cfg) {
  _cfg = cfg;
  _els = {
    root: document.getElementById('coder-search'),
    input: document.getElementById('coder-search-input'),
    glob: document.getElementById('coder-search-glob'),
    summary: document.getElementById('coder-search-summary'),
    results: document.getElementById('coder-search-results'),
    tree: document.getElementById('coder-file-tree'),
  };
  if (!_els.root || !_els.input) return;

  _els.input.addEventListener('input', () => _scheduleSearch());
  _els.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); _runSearch(true); }
    else if (e.key === 'Escape') { e.preventDefault(); closeCoderSearch(); }
  });
  _els.glob?.addEventListener('input', () => _scheduleSearch());
  _els.glob?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); _runSearch(true); }
  });

  // Option toggles (case / regex) + mode switch (text / semantic).
  _els.root.querySelectorAll('.coder-search-opt').forEach((btn) => {
    btn.addEventListener('click', () => {
      const opt = btn.dataset.opt;
      _opts[opt] = !_opts[opt];
      btn.classList.toggle('active', _opts[opt]);
      btn.setAttribute('aria-pressed', String(_opts[opt]));
      _runSearch(true);
    });
  });
  _els.root.querySelectorAll('.coder-search-mode-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (_mode === btn.dataset.mode) return;
      _mode = btn.dataset.mode;
      _els.root.querySelectorAll('.coder-search-mode-btn').forEach((b) =>
        b.classList.toggle('active', b.dataset.mode === _mode));
      _els.root.dataset.mode = _mode;
      // Case/regex are meaningless for semantic search — disable them.
      _els.root.querySelectorAll('.coder-search-opt').forEach((b) => {
        b.disabled = _mode === 'semantic';
      });
      if (_els.glob) _els.glob.disabled = _mode === 'semantic';
      _runSearch(true);
    });
  });

  document.getElementById('coder-search-close')
    ?.addEventListener('click', () => closeCoderSearch());

  // Delegated: match-row click opens the file; file-group header toggles.
  _els.results.addEventListener('click', (e) => {
    const groupHead = e.target.closest('.coder-search-group-head');
    if (groupHead) {
      groupHead.parentElement.classList.toggle('is-collapsed');
      return;
    }
    const row = e.target.closest('.coder-search-match');
    if (!row) return;
    _cfg.openResult({
      path: row.dataset.path,
      name: row.dataset.name,
      line: parseInt(row.dataset.line, 10) || 1,
      spans: _decodeSpans(row.dataset.spans),
    });
  });

  // Delegated: the "build index" affordance in the semantic empty state.
  _els.results.addEventListener('click', async (e) => {
    const buildBtn = e.target.closest('.coder-search-build-index');
    if (!buildBtn) return;
    buildBtn.disabled = true;
    buildBtn.textContent = 'Building index…';
    await _cfg.buildIndex?.();
    // Re-run once the build kicks off; the index populates progressively
    // so an immediate re-query may still be thin, but it stops looking
    // broken and the user can re-search as it fills.
    _runSearch(true);
  });
}

export function isCoderSearchOpen() { return _open; }

export function openCoderSearch() {
  if (!_els?.root) return;
  _open = true;
  _els.root.classList.remove('hidden');
  if (_els.tree) _els.tree.classList.add('coder-file-tree--search-hidden');
  document.getElementById('coder-search-toggle-btn')?.classList.add('active');
  // Seed the box with the editor's current selection if there is one —
  // matches the "search for what I've got highlighted" muscle memory.
  const sel = window.getSelection?.()?.toString?.().trim();
  if (sel && sel.length <= 200 && !_els.input.value) _els.input.value = sel;
  _els.input.focus();
  _els.input.select();
  if (_els.input.value) _runSearch(true);
}

export function closeCoderSearch() {
  if (!_els?.root) return;
  _open = false;
  _els.root.classList.add('hidden');
  if (_els.tree) _els.tree.classList.remove('coder-file-tree--search-hidden');
  document.getElementById('coder-search-toggle-btn')?.classList.remove('active');
}

export function toggleCoderSearch() {
  if (_open) closeCoderSearch();
  else openCoderSearch();
}

// ---------------------------------------------------------------------------
// Search execution
// ---------------------------------------------------------------------------

function _scheduleSearch() {
  if (_debounce) clearTimeout(_debounce);
  _debounce = setTimeout(() => _runSearch(false), 250);
}

async function _runSearch(immediate) {
  if (_debounce) { clearTimeout(_debounce); _debounce = null; }
  const query = _els.input.value.trim();
  const wid = _cfg.getWorkspaceId();
  const token = ++_queryToken;
  _lastQuery = query;

  if (!query) {
    _els.results.innerHTML = '';
    _setSummary('');
    return;
  }
  if (!wid) { _setSummary('No workspace selected'); return; }

  _setSummary('Searching…');
  try {
    const data = _mode === 'semantic'
      ? await _fetchSemantic(wid, query)
      : await _fetchText(wid, query);
    if (token !== _queryToken) return; // superseded by a newer query
    if (_mode === 'semantic') _renderSemantic(data, query);
    else _renderText(data, query);
  } catch (err) {
    if (token !== _queryToken) return;
    _els.results.innerHTML = '';
    _setSummary('Search failed');
    console.warn('[coder-search] failed', err);
  }
}

async function _fetchText(wid, query) {
  const params = new URLSearchParams({ q: query, limit: '600' });
  if (_opts.regex) params.set('regex', '1');
  if (_opts.case) params.set('case', '1');
  const glob = _els.glob?.value.trim();
  if (glob) params.set('glob', glob);
  const resp = await fetch(
    `/api/coder/workspaces/${encodeURIComponent(wid)}/search-text?${params}`,
    { credentials: 'include' },
  );
  return resp.json();
}

async function _fetchSemantic(wid, query) {
  const params = new URLSearchParams({ q: query, limit: '25' });
  const resp = await fetch(
    `/api/coder/search/${encodeURIComponent(wid)}?${params}`,
    { credentials: 'include' },
  );
  return resp.json();
}

// ---------------------------------------------------------------------------
// Rendering — text
// ---------------------------------------------------------------------------

function _renderText(data, query) {
  if (data?.error) {
    _els.results.innerHTML =
      `<div class="coder-search-note is-error">${escapeHtml(data.error)}</div>`;
    _setSummary('Invalid search');
    return;
  }
  const matches = data?.matches || [];
  if (!matches.length) {
    _els.results.innerHTML =
      `<div class="coder-search-note">No matches for “${escapeHtml(query)}”.</div>`;
    _setSummary('0 results');
    return;
  }

  // Group by file, preserving first-seen order.
  const groups = new Map();
  for (const m of matches) {
    if (!groups.has(m.path)) groups.set(m.path, []);
    groups.get(m.path).push(m);
  }

  const frag = document.createDocumentFragment();
  for (const [path, hits] of groups) {
    frag.appendChild(_renderGroup(path, hits));
  }
  _els.results.innerHTML = '';
  _els.results.appendChild(frag);

  const enginePart = data.engine === 'grep' ? ' · basic (no ripgrep)' : '';
  const cap = data.truncated ? '+ (narrow to see more)' : '';
  _setSummary(
    `${data.total_returned}${cap} match${data.total_returned === 1 ? '' : 'es'} ` +
    `in ${data.files_with_matches} file${data.files_with_matches === 1 ? '' : 's'}${enginePart}`,
  );
}

function _renderGroup(path, hits) {
  const group = document.createElement('div');
  group.className = 'coder-search-group';
  const rel = _rel(path);
  const name = rel.split('/').pop();
  const dir = rel.slice(0, rel.length - name.length);

  const head = document.createElement('div');
  head.className = 'coder-search-group-head';
  head.innerHTML =
    `<span class="coder-search-chevron">▾</span>` +
    `<span class="coder-search-file-icon">${_fileIcon(name)}</span>` +
    `<span class="coder-search-file-name">${escapeHtml(name)}</span>` +
    (dir ? `<span class="coder-search-file-dir">${escapeHtml(dir.replace(/\/$/, ''))}</span>` : '') +
    `<span class="coder-search-file-count">${hits.length}</span>`;
  group.appendChild(head);

  const body = document.createElement('div');
  body.className = 'coder-search-group-body';
  for (const m of hits) {
    const row = document.createElement('div');
    row.className = 'coder-search-match';
    row.dataset.path = path;
    row.dataset.name = name;
    row.dataset.line = String(m.line);
    row.dataset.spans = _encodeSpans(m.spans);
    row.title = `Open ${name}:${m.line}`;
    // Full line as the code span's tooltip so a trace that still clips at
    // the current panel width is readable on hover (complements the
    // drag-to-widen panel). escapeHtml escapes quotes → attr-safe.
    row.innerHTML =
      `<span class="coder-search-lineno">${m.line}</span>` +
      `<span class="coder-search-code" title="${escapeHtml(m.text || '')}">${_highlight(m.text || '', m.spans)}` +
      (m.clipped ? '<span class="coder-search-ellipsis"> …</span>' : '') +
      `</span>`;
    body.appendChild(row);
  }
  group.appendChild(body);
  return group;
}

// ---------------------------------------------------------------------------
// Rendering — semantic
// ---------------------------------------------------------------------------

function _renderSemantic(data, query) {
  if (data?.error || !data) {
    // The dominant error here is "index not built yet". Offer the build.
    _els.results.innerHTML =
      `<div class="coder-search-note">` +
      `<p>Semantic search needs a codebase index for this workspace.</p>` +
      `<button class="btn btn-sm coder-search-build-index">Build index now</button>` +
      `</div>`;
    _setSummary('Index not ready');
    return;
  }
  const results = data.results || [];
  if (!results.length) {
    _els.results.innerHTML =
      `<div class="coder-search-note">No semantic matches for “${escapeHtml(query)}”.` +
      ` <button class="btn btn-sm coder-search-build-index">Rebuild index</button></div>`;
    _setSummary('0 results');
    return;
  }

  const frag = document.createDocumentFragment();
  for (const r of results) {
    const path = r.file_path?.startsWith('/workspace')
      ? r.file_path
      : `/workspace/${(r.file_path || '').replace(/^\/+/, '')}`;
    const rel = _rel(path);
    const name = rel.split('/').pop();
    const row = document.createElement('div');
    row.className = 'coder-search-match coder-search-semantic-hit';
    row.dataset.path = path;
    row.dataset.name = name;
    row.dataset.line = String(r.start_line || 1);
    row.dataset.spans = '';
    row.title = `Open ${name}:${r.start_line || 1}`;
    const pct = Math.round((r.score || 0) * 100);
    row.innerHTML =
      `<div class="coder-search-sem-head">` +
      `<span class="coder-search-file-icon">${_fileIcon(name)}</span>` +
      `<span class="coder-search-file-name">${escapeHtml(name)}</span>` +
      `<span class="coder-search-file-dir">:${r.start_line || 1}</span>` +
      `<span class="coder-search-score" title="Relevance">${pct}%</span></div>` +
      `<pre class="coder-search-sem-snippet">${escapeHtml((r.content || '').slice(0, 240))}</pre>`;
    frag.appendChild(row);
  }
  _els.results.innerHTML = '';
  _els.results.appendChild(frag);
  _setSummary(`${results.length} semantic result${results.length === 1 ? '' : 's'}`);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _setSummary(text) {
  if (_els.summary) _els.summary.textContent = text;
}

function _rel(path) { return (path || '').replace(/^\/workspace\//, ''); }

// Highlight match spans with <mark>. Spans are char offsets into text;
// we escape each segment independently so highlighting can't break out
// of the escaped context.
function _highlight(text, spans) {
  if (!Array.isArray(spans) || !spans.length) return escapeHtml(text);
  const sorted = spans
    .filter((s) => Array.isArray(s) && s.length === 2)
    .map((s) => [Math.max(0, s[0]), Math.min(text.length, s[1])])
    .filter((s) => s[1] > s[0])
    .sort((a, b) => a[0] - b[0]);
  let html = '';
  let cursor = 0;
  for (const [s, e] of sorted) {
    if (s < cursor) continue; // drop overlaps
    html += escapeHtml(text.slice(cursor, s));
    html += `<mark class="coder-search-hit">${escapeHtml(text.slice(s, e))}</mark>`;
    cursor = e;
  }
  html += escapeHtml(text.slice(cursor));
  return html;
}

// Spans ride on a data-attribute as compact "s:e,s:e" pairs so the
// click handler can reconstruct them without a parallel JS-side map.
function _encodeSpans(spans) {
  if (!Array.isArray(spans)) return '';
  return spans.map((s) => `${s[0]}:${s[1]}`).join(',');
}
function _decodeSpans(str) {
  if (!str) return [];
  return str.split(',').map((p) => p.split(':').map(Number)).filter((s) => s.length === 2);
}

function _fileIcon(name) {
  const ext = name.split('.').pop()?.toLowerCase();
  const icons = {
    js: 'JS', ts: 'TS', py: 'PY', md: 'MD', json: '{}',
    html: '<>', css: '#', sh: '$', yml: '~', yaml: '~',
    rs: 'RS', go: 'GO', c: 'C', cpp: 'C+', h: '.h',
    java: 'JV', rb: 'RB', toml: '~',
  };
  return icons[ext] || '○';
}
