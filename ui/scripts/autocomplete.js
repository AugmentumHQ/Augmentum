/**
 * Autocomplete for workspace code editor.
 * Uses CodeMind AST declarations when available, falls back to regex word extraction.
 * Language keywords, snippets, fuzzy matching.
 */
import * as CodeMind from './codemind.js';

// ── Caret Position (mirror div technique) ──────────────────────────

const _mirrorProps = [
  'direction','boxSizing','width','height','overflowX','overflowY',
  'borderTopWidth','borderRightWidth','borderBottomWidth','borderLeftWidth','borderStyle',
  'paddingTop','paddingRight','paddingBottom','paddingLeft',
  'fontStyle','fontVariant','fontWeight','fontStretch','fontSize','fontSizeAdjust',
  'lineHeight','fontFamily','textAlign','textTransform','textIndent','textDecoration',
  'letterSpacing','wordSpacing','tabSize','MozTabSize'
];

function getCaretCoordinates(textarea, position) {
  const div = document.createElement('div');
  div.id = 'autocomplete-mirror';
  document.body.appendChild(div);
  const style = div.style;
  const computed = window.getComputedStyle(textarea);
  style.whiteSpace = 'pre-wrap';
  style.wordWrap = 'break-word';
  style.position = 'absolute';
  style.visibility = 'hidden';
  style.overflow = 'hidden';
  for (const prop of _mirrorProps) style[prop] = computed[prop];
  // Firefox overflow fix
  if (textarea.scrollHeight > parseInt(computed.height)) style.overflowY = 'scroll';
  div.textContent = textarea.value.substring(0, position);
  const span = document.createElement('span');
  span.textContent = textarea.value.substring(position) || '.';
  div.appendChild(span);
  const coords = {
    top: span.offsetTop + parseInt(computed.borderTopWidth),
    left: span.offsetLeft + parseInt(computed.borderLeftWidth),
    height: parseInt(computed.lineHeight) || parseInt(computed.fontSize) * 1.2
  };
  document.body.removeChild(div);
  return coords;
}

// ── Completion Sources ─────────────────────────────────────────────

function getWordCompletions(files, currentFile, prefix) {
  const completions = [];
  const seen = new Set();

  // AST-first: use CodeMind declarations for structured, accurate completions
  if (CodeMind.isReady()) {
    for (const f of files) {
      const lang = _extToLang(f.path);
      const decls = CodeMind.getDeclarations(f.content || '', lang, f.path);
      for (const d of decls) {
        if (d.name && !seen.has(d.name) && d.name !== prefix) {
          seen.add(d.name);
          const typeMap = { function: 'function', class: 'variable', export: 'variable' };
          completions.push({
            label: d.name,
            detail: d.signature ? d.signature.slice(0, 60) : '',
            type: typeMap[d.type] || 'variable',
            insertText: d.name,
          });
        }
      }
    }
  }

  // Regex fallback: catch identifiers the AST didn't surface (local vars, etc.)
  for (const f of files) {
    const matches = f.content.matchAll(/\b([a-zA-Z_$][\w$]{1,})\b/g);
    for (const m of matches) {
      if (!seen.has(m[1]) && m[1] !== prefix) {
        seen.add(m[1]);
        completions.push({ label: m[1], detail: '', type: 'variable', insertText: m[1] });
      }
    }
  }

  return completions;
}

function _extToLang(path) {
  if (path.endsWith('.html') || path.endsWith('.htm')) return 'html';
  if (path.endsWith('.css') || path.endsWith('.scss')) return 'css';
  if (path.endsWith('.json')) return 'json';
  if (path.endsWith('.py')) return 'python';
  if (path.endsWith('.ts') || path.endsWith('.tsx')) return 'typescript';
  return 'javascript';
}

const JS_KEYWORDS = ['function','const','let','var','if','else','for','while','do','switch','case','break','continue','return','class','extends','new','this','super','import','export','default','async','await','try','catch','finally','throw','typeof','instanceof','in','of','delete','void','yield','true','false','null','undefined','console','document','window','Math','Array','Object','String','Number','Boolean','Promise','setTimeout','setInterval','clearTimeout','clearInterval','requestAnimationFrame','addEventListener','removeEventListener','getElementById','querySelector','querySelectorAll','createElement','appendChild','removeChild','innerHTML','textContent','classList','style','setAttribute','getAttribute','preventDefault','stopPropagation','JSON','fetch','localStorage','sessionStorage'];

const CSS_KEYWORDS = ['display','position','top','right','bottom','left','width','height','min-width','max-width','min-height','max-height','margin','padding','border','border-radius','background','background-color','background-image','color','font-size','font-weight','font-family','line-height','text-align','text-decoration','text-transform','letter-spacing','opacity','overflow','z-index','cursor','transition','transform','animation','box-shadow','flex','flex-direction','flex-wrap','justify-content','align-items','align-content','gap','grid','grid-template-columns','grid-template-rows','visibility','pointer-events','user-select','backdrop-filter','filter','outline','resize','white-space','word-break','content','var()','calc()','color-mix()','rgba()','linear-gradient()','radial-gradient()'];

const HTML_TAGS = ['div','span','p','a','button','input','form','label','select','option','textarea','h1','h2','h3','h4','h5','h6','ul','ol','li','table','tr','td','th','thead','tbody','img','video','audio','canvas','svg','section','article','header','footer','nav','main','aside','figure','figcaption','details','summary','dialog','template','slot','script','style','link','meta','title','head','body','html'];

function getKeywordCompletions(language, prefix) {
  let keywords;
  if (language === 'css') keywords = CSS_KEYWORDS;
  else if (language === 'markup' || language === 'html') keywords = HTML_TAGS.map(t => `<${t}>`);
  else keywords = JS_KEYWORDS;
  return keywords.map(k => ({ label: k, detail: 'keyword', type: 'keyword', insertText: k }));
}

const JS_SNIPPETS = [
  { label: 'function', detail: 'function declaration', type: 'snippet', insertText: 'function name() {\n  \n}' },
  { label: 'arrow', detail: '() => {}', type: 'snippet', insertText: '() => {\n  \n}' },
  { label: 'foreach', detail: '.forEach()', type: 'snippet', insertText: '.forEach((item) => {\n  \n})' },
  { label: 'map', detail: '.map()', type: 'snippet', insertText: '.map((item) => {\n  \n})' },
  { label: 'filter', detail: '.filter()', type: 'snippet', insertText: '.filter((item) => {\n  \n})' },
  { label: 'ael', detail: 'addEventListener', type: 'snippet', insertText: "addEventListener('click', (e) => {\n  \n})" },
  { label: 'gel', detail: 'getElementById', type: 'snippet', insertText: "document.getElementById('')" },
  { label: 'qs', detail: 'querySelector', type: 'snippet', insertText: "document.querySelector('')" },
  { label: 'qsa', detail: 'querySelectorAll', type: 'snippet', insertText: "document.querySelectorAll('')" },
  { label: 'log', detail: 'console.log', type: 'snippet', insertText: 'console.log()' },
  { label: 'ce', detail: 'createElement', type: 'snippet', insertText: "document.createElement('')" },
  { label: 'trycatch', detail: 'try/catch', type: 'snippet', insertText: 'try {\n  \n} catch (err) {\n  console.error(err);\n}' },
  { label: 'ifelse', detail: 'if/else', type: 'snippet', insertText: 'if (condition) {\n  \n} else {\n  \n}' },
  { label: 'forloop', detail: 'for loop', type: 'snippet', insertText: 'for (let i = 0; i < length; i++) {\n  \n}' },
  { label: 'class', detail: 'class declaration', type: 'snippet', insertText: 'class Name {\n  constructor() {\n    \n  }\n}' },
  { label: 'promise', detail: 'new Promise', type: 'snippet', insertText: 'new Promise((resolve, reject) => {\n  \n})' },
  { label: 'asyncfn', detail: 'async function', type: 'snippet', insertText: 'async function name() {\n  \n}' },
  { label: 'fetch', detail: 'fetch request', type: 'snippet', insertText: "fetch(url)\n  .then(r => r.json())\n  .then(data => {\n    \n  })" },
  { label: 'raf', detail: 'requestAnimationFrame', type: 'snippet', insertText: 'requestAnimationFrame(function loop(t) {\n  \n  requestAnimationFrame(loop);\n})' },
  { label: 'iife', detail: 'IIFE', type: 'snippet', insertText: "(function() {\n  'use strict';\n  \n})();" },
];

const CSS_SNIPPETS = [
  { label: 'flexcenter', detail: 'flex center', type: 'snippet', insertText: 'display: flex;\nalign-items: center;\njustify-content: center;' },
  { label: 'grid', detail: 'grid layout', type: 'snippet', insertText: 'display: grid;\ngrid-template-columns: repeat(auto-fill, minmax(250px, 1fr));\ngap: 16px;' },
  { label: 'transition', detail: 'transition shorthand', type: 'snippet', insertText: 'transition: all 0.2s ease;' },
  { label: 'absolute', detail: 'absolute fill', type: 'snippet', insertText: 'position: absolute;\ninset: 0;' },
  { label: 'truncate', detail: 'text truncate', type: 'snippet', insertText: 'white-space: nowrap;\noverflow: hidden;\ntext-overflow: ellipsis;' },
  { label: 'media', detail: 'media query', type: 'snippet', insertText: '@media (max-width: 768px) {\n  \n}' },
  { label: 'keyframes', detail: '@keyframes', type: 'snippet', insertText: '@keyframes name {\n  from { }\n  to { }\n}' },
  { label: 'var', detail: 'CSS variable', type: 'snippet', insertText: 'var(--)' },
  { label: 'root', detail: ':root variables', type: 'snippet', insertText: ':root {\n  --color-primary: #;\n  --color-bg: #;\n}' },
  { label: 'hover', detail: ':hover state', type: 'snippet', insertText: ':hover {\n  \n}' },
];

function getSnippetCompletions(language) {
  if (language === 'css') return CSS_SNIPPETS;
  return JS_SNIPPETS;
}

// ── Fuzzy Matcher ──────────────────────────────────────────────────

function fuzzyMatch(pattern, candidate) {
  if (!pattern) return { score: 0, positions: [] };
  const p = pattern.toLowerCase();
  const c = candidate.toLowerCase();

  // Tier 1: exact prefix
  if (c.startsWith(p)) return { score: 1000 - candidate.length, positions: Array.from({length: p.length}, (_, i) => i) };

  // Tier 2: substring
  const subIdx = c.indexOf(p);
  if (subIdx >= 0) return { score: 500 - subIdx * 10 - candidate.length, positions: Array.from({length: p.length}, (_, i) => subIdx + i) };

  // Tier 3: camelCase / by-word fuzzy
  const positions = [];
  let pi = 0;
  for (let ci = 0; ci < candidate.length && pi < pattern.length; ci++) {
    if (candidate[ci].toLowerCase() === p[pi]) {
      positions.push(ci);
      pi++;
    }
  }
  if (pi === pattern.length) {
    let score = 300;
    for (let i = 1; i < positions.length; i++) {
      const gap = positions[i] - positions[i - 1] - 1;
      score -= gap * 10;
    }
    if (positions[0] === 0) score += 100;
    score -= candidate.length;
    return { score, positions };
  }

  return null; // no match
}

function filterAndSort(items, prefix) {
  if (!prefix) return items.slice(0, 50);
  const results = [];
  for (const item of items) {
    const match = fuzzyMatch(prefix, item.label);
    if (match) results.push({ ...item, score: match.score, matchPositions: match.positions });
  }
  results.sort((a, b) => {
    const aBonus = a.type === 'snippet' ? 50 : 0;
    const bBonus = b.type === 'snippet' ? 50 : 0;
    return (b.score + bBonus) - (a.score + aBonus);
  });
  return results.slice(0, 30);
}

// ── Popup Renderer ─────────────────────────────────────────────────

let _popup = null;
let _selectedIdx = 0;
let _filteredItems = [];
let _startPos = 0;

function showPopup(items, textarea, prefix) {
  _filteredItems = items;
  _selectedIdx = 0;

  if (!_popup) {
    _popup = document.createElement('div');
    _popup.className = 'ac-popup';
    _popup.setAttribute('role', 'listbox');
    document.body.appendChild(_popup);
  }

  // Position
  const coords = getCaretCoordinates(textarea, textarea.selectionEnd);
  const rect = textarea.getBoundingClientRect();
  let top = rect.top + coords.top - textarea.scrollTop + coords.height + 4;
  let left = rect.left + coords.left - textarea.scrollLeft;

  // Flip if near bottom
  const popupHeight = Math.min(items.length * 28, 280);
  if (top + popupHeight > window.innerHeight - 10) {
    top = rect.top + coords.top - textarea.scrollTop - popupHeight - 4;
  }
  // Constrain to viewport — account for popup width on narrow screens
  const popupW = Math.min(380, Math.max(220, window.innerWidth - 32));
  left = Math.max(8, Math.min(left, window.innerWidth - popupW - 8));

  _popup.style.top = top + 'px';
  _popup.style.left = left + 'px';
  _popup.style.display = '';

  _renderPopupItems();
}

function _renderPopupItems() {
  if (!_popup) return;
  _popup.innerHTML = _filteredItems.map((item, i) => {
    const active = i === _selectedIdx ? ' ac-active' : '';
    const icon = _getTypeIcon(item.type);
    const label = _highlightMatches(item.label, item.matchPositions);
    const detail = item.detail ? `<span class="ac-detail">${item.detail}</span>` : '';
    return `<div class="ac-item${active}" data-idx="${i}" role="option">${icon}<span class="ac-label">${label}</span>${detail}</div>`;
  }).join('');

  // Wire click
  _popup.querySelectorAll('.ac-item').forEach(el => {
    el.addEventListener('mousedown', (e) => {
      e.preventDefault(); // prevent textarea blur
      _selectedIdx = parseInt(el.dataset.idx);
      acceptCompletion();
    });
  });

  // Scroll selected into view
  const activeEl = _popup.querySelector('.ac-active');
  if (activeEl) activeEl.scrollIntoView({ block: 'nearest' });
}

function _getTypeIcon(type) {
  const icons = {
    variable: '<span class="ac-icon ac-icon-var">V</span>',
    keyword: '<span class="ac-icon ac-icon-kw">K</span>',
    snippet: '<span class="ac-icon ac-icon-sn">S</span>',
    function: '<span class="ac-icon ac-icon-fn">F</span>',
  };
  return icons[type] || icons.variable;
}

function _highlightMatches(label, positions) {
  if (!positions?.length) return label;
  let result = '';
  for (let i = 0; i < label.length; i++) {
    if (positions.includes(i)) result += `<b>${label[i]}</b>`;
    else result += label[i];
  }
  return result;
}

function hidePopup() {
  if (_popup) _popup.style.display = 'none';
  _filteredItems = [];
  _selectedIdx = 0;
}

export function isPopupVisible() {
  return _popup && _popup.style.display !== 'none';
}

// ── Main Controller ────────────────────────────────────────────────

let _textarea = null;
let _getFiles = null;
let _getLang = null;
let _debounceTimer = null;

export function initAutocomplete(textarea, getFilesFn, getLangFn) {
  _textarea = textarea;
  _getFiles = getFilesFn;
  _getLang = getLangFn;

  textarea.addEventListener('input', _onInput);
  textarea.addEventListener('keydown', _onKeyDown);
  textarea.addEventListener('blur', () => setTimeout(hidePopup, 150));
  textarea.addEventListener('scroll', hidePopup);

  // Dismiss on click outside
  document.addEventListener('mousedown', (e) => {
    if (_popup && !_popup.contains(e.target) && e.target !== textarea) hidePopup();
  });
}

export function destroyAutocomplete() {
  if (_textarea) {
    _textarea.removeEventListener('input', _onInput);
    _textarea.removeEventListener('keydown', _onKeyDown);
  }
  if (_popup) { _popup.remove(); _popup = null; }
  _textarea = null;
}

function _onInput() {
  clearTimeout(_debounceTimer);
  _debounceTimer = setTimeout(() => {
    const pos = _textarea.selectionEnd;
    const text = _textarea.value;
    const prefix = _getPrefix(text, pos);

    if (!prefix || prefix.length < 2) { hidePopup(); return; }

    _startPos = pos - prefix.length;
    const lang = _getLang ? _getLang() : 'javascript';
    const files = _getFiles ? _getFiles() : [];

    // Gather completions from all sources
    const all = [
      ...getWordCompletions(files, text, prefix),
      ...getKeywordCompletions(lang, prefix),
      ...getSnippetCompletions(lang),
    ];

    const filtered = filterAndSort(all, prefix);
    if (filtered.length === 0) { hidePopup(); return; }

    showPopup(filtered, _textarea, prefix);
  }, 100);
}

function _onKeyDown(e) {
  if (!isPopupVisible()) return;

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    _selectedIdx = (_selectedIdx + 1) % _filteredItems.length;
    _renderPopupItems();
    return;
  }
  if (e.key === 'ArrowUp') {
    e.preventDefault();
    _selectedIdx = (_selectedIdx - 1 + _filteredItems.length) % _filteredItems.length;
    _renderPopupItems();
    return;
  }
  if (e.key === 'Tab' || e.key === 'Enter') {
    if (_filteredItems.length > 0) {
      e.preventDefault();
      acceptCompletion();
      return;
    }
  }
  if (e.key === 'Escape') {
    e.preventDefault();
    hidePopup();
    return;
  }
}

function acceptCompletion() {
  const item = _filteredItems[_selectedIdx];
  if (!item || !_textarea) return;

  const before = _textarea.value.slice(0, _startPos);
  const after = _textarea.value.slice(_textarea.selectionEnd);
  _textarea.value = before + item.insertText + after;
  const newPos = _startPos + item.insertText.length;
  _textarea.selectionStart = _textarea.selectionEnd = newPos;
  _textarea.dispatchEvent(new Event('input')); // trigger highlight + save
  hidePopup();
}

function _getPrefix(text, pos) {
  let start = pos;
  while (start > 0) {
    const ch = text[start - 1];
    if (/[\w$]/.test(ch)) start--;
    else break;
  }
  return text.slice(start, pos);
}
