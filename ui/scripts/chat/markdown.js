/* ==========================================================================
   Chat Module — Markdown Rendering
   Markdown→HTML pipeline, code blocks, tables, citations, artifact cards
   ========================================================================== */

import { escapeHtml } from '../app.js';
import { icons } from './constants.js';
import { applyInlineEmphasis } from './emphasis.js';

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

export function unescapeHtml(str) {
  const doc = new DOMParser().parseFromString(str, 'text/html');
  return doc.documentElement.textContent;
}

export function blockFingerprint(lang, code) {
  let hash = 5381;
  const str = (lang || '') + ':' + code.slice(0, 200);
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) & 0x7fffffff;
  }
  return 'cb_' + hash.toString(36);
}

// ---------------------------------------------------------------------------
// Narrative-mode "panel" blocks — collapsible cards for stat/scene/system
// blocks that LLMs and character cards emit via ```md / ```stats / etc.
// Visually distinct from code blocks AND from prose. Excluded from TTS.
// ---------------------------------------------------------------------------

const _NARRATIVE_PANEL_LANGS = new Set([
  'md', 'markdown', 'stats', 'stat', 'scene', 'system', 'gm', 'note',
]);

const _PANEL_SUBTYPE_LABEL = {
  stats: 'Stats',
  stat: 'Stats',
  scene: 'Scene',
  system: 'System',
  gm: 'GM',
  note: 'Note',
  generic: 'Note',
};

/** Heuristic for bare ```md / ```markdown blocks. Returns one of:
 *  'stats' | 'scene' | 'system' | 'generic'. Explicit lang tags bypass this. */
function _detectNarrativeSubtype(lang, content) {
  const tag = (lang || '').toLowerCase();
  if (tag === 'stat') return 'stats';
  if (_PANEL_SUBTYPE_LABEL[tag]) return tag;
  // Bare md/markdown — sniff the content.
  const text = content.trim();
  // Stat block: many "KEY: NN" or "KEY: NN/NN" lines.
  const statLines = (text.match(/^[ \t]*[\w][\w \-/]{0,30}[ \t]*:[ \t]*[\d+\-]+\s*(?:\/\s*\d+)?/gm) || []).length;
  if (statLines >= 2) return 'stats';
  // Scene markers: arrow transitions, or known scene keys.
  if (/\u2192|->/.test(text)) return 'scene';
  if (/^\s*(?:location|time|place|setting|weather|atmosphere|now|when|where)\s*:/im.test(text)) return 'scene';
  // System/GM cues
  if (/^\s*(?:\[(?:gm|system|narrator|ooc)\b|\(\(|<system\b)/im.test(text)) return 'system';
  return 'generic';
}

/** Build a one-line summary for the collapsed panel. ESCAPED. May be empty. */
function _summarizePanel(subtype, content) {
  const text = content.trim();
  if (!text) return '';

  if (subtype === 'stats' || subtype === 'stat') {
    // Pluck first few KEY: VAL pairs. Anchor to same-line only ([ \t] not \s)
    // so a stat-line followed by another stat-line doesn't get glued together
    // (e.g. "HP: 45/100\nMP: ..." would otherwise consume "MP" as a unit).
    const pairs = [];
    const re = /^[ \t]*([\w][\w \-/]{0,30}?)[ \t]*:[ \t]*([\d+\-]+(?:[ \t]*\/[ \t]*\d+)?)/gim;
    let m;
    while ((m = re.exec(text)) && pairs.length < 3) {
      pairs.push(`${m[1].trim()} ${m[2].replace(/[ \t]+/g, '')}`);
    }
    if (pairs.length) return escapeHtml(pairs.join(' \u00B7 '));
  }

  if (subtype === 'scene') {
    // First line that looks like KEY: VAL or contains an arrow.
    for (const raw of text.split('\n')) {
      const line = raw.trim();
      if (!line) continue;
      if (/[:\u2192]|->/.test(line)) {
        return escapeHtml(_truncate(line, 80));
      }
    }
  }

  // Generic / system — first non-empty line, truncated. Strip leading AND
  // trailing markdown punctuation so action lines like *She smiles.* read
  // cleanly in the summary slot.
  for (const raw of text.split('\n')) {
    const line = raw.trim().replace(/^[*_~`>#\-\s]+/, '').replace(/[*_~`]+$/, '').trim();
    if (line) return escapeHtml(_truncate(line, 80));
  }
  return '';
}

function _truncate(s, max) {
  if (s.length <= max) return s;
  return s.slice(0, max - 1).trimEnd() + '\u2026';
}

/** Build the collapsible panel HTML. Body is rendered recursively as
 *  narrative markdown so dialogue/action styling fires inside.
 *  ``defaultCollapsed`` controls the initial aria-expanded state. The user
 *  toggles per-element via the click handler in renderer.js. */
function _buildNarrativePanel(lang, rawCode, mode, options) {
  const subtype = _detectNarrativeSubtype(lang, rawCode);
  const label = _PANEL_SUBTYPE_LABEL[subtype] || _PANEL_SUBTYPE_LABEL.generic;
  const summary = _summarizePanel(subtype, rawCode);
  const collapsed = options?.narrativePanelsCollapsed !== false;
  // Render body recursively so nested markdown (italics for actions, dialogue
  // styling, lists) all works. Pass mode through; force collapsed flag off in
  // the recursion so we never render a panel inside a panel via heuristic.
  const bodyHtml = renderMarkdown(rawCode, { mode, narrativePanelsCollapsed: true, _insidePanel: true });
  const expandedAttr = collapsed ? 'false' : 'true';
  const summaryHtml = summary ? `<span class="narrative-panel-summary">${summary}</span>` : '';
  const langAttr = lang ? ` data-lang="${escapeHtml(lang)}"` : '';
  return (
    `<div class="narrative-panel narrative-panel-${escapeHtml(subtype)}" data-tts-skip="true"${langAttr}>` +
      `<button type="button" class="narrative-panel-toggle" aria-expanded="${expandedAttr}">` +
        `<svg class="narrative-panel-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" width="12" height="12"><polyline points="9 18 15 12 9 6"/></svg>` +
        `<span class="narrative-panel-label">${escapeHtml(label)}</span>` +
        summaryHtml +
      `</button>` +
      `<div class="narrative-panel-body">${bodyHtml}</div>` +
    `</div>`
  );
}

/** Rewrite near-miss panel fences (2 or 4+ backticks) into canonical
 *  3-backtick form. Scoped to known panel langs so we never rewrite code
 *  fences, where a weird backtick count may be intentional.
 *
 *  Matches patterns like:
 *    ``md\ncontent\n``         → ```md\ncontent\n```
 *    ````scene\ncontent\n````  → ```scene\ncontent\n```
 *
 *  Requires matching open/close counts and a newline after the lang tag. */
export function _normalizeNearMissPanelFences(text) {
  if (!text || !text.includes('`')) return text;
  return text.replace(
    /(^|\n)(`{2}|`{4,})(md|markdown|stats|stat|scene|system|gm|note)[ \t]*\n([\s\S]*?)\n\2(?=\n|$)/gi,
    (_m, pre, _open, lang, body) => `${pre}\`\`\`${lang}\n${body}\n\`\`\``,
  );
}

/** Returns true if ``text`` is a single narrative-md fence wrapping ~all of
 *  the message — the "LLM panicked and wrapped its whole reply" anti-pattern.
 *  In that case we want to unwrap, NOT render a collapsed panel (which would
 *  leave the chat bubble effectively empty). Only triggers for bare md /
 *  markdown — explicit stats/scene/system tags are intentional. */
export function unwrapWholeMessageMarkdownFence(text) {
  if (!text) return text;
  const m = text.match(/^\s*```(md|markdown)\s*\n([\s\S]*?)\n```\s*$/);
  if (!m) return text;
  // Reject if there are multiple fences in the original (possibly nested).
  // ``^...$`` already requires single fence-pair from start to end of trimmed
  // text; this guards against pathological inputs the regex might still match.
  const inner = m[2];
  if (inner.includes('```')) return text;
  // Only unwrap if the inner content is the dominant part (>= 50% of the
  // outer length). For very short messages this still triggers; that's OK.
  if (inner.length < text.trim().length * 0.5) return text;
  return inner;
}

function _sanitizeUrl(href) {
  const decoded = decodeURIComponent(href);
  if (/^(https?:\/\/|\/[^\/])/i.test(decoded)) return href;
  return null;
}

function _sanitizeMediaUrl(src) {
  const decoded = decodeURIComponent(src);
  if (/^\/api\//i.test(decoded)) return src;
  if (/^data:image\//i.test(decoded)) return src;
  // External http(s) URLs go through the SSRF-safe proxy so hotlink-protected
  // CDNs (Referer checks, CORS) still render inline.
  if (/^https?:\/\//i.test(decoded)) {
    return `/api/browse/image?url=${encodeURIComponent(src)}`;
  }
  try { console.debug('[markdown] image URL rejected by sanitizer:', src.slice(0, 120)); } catch (_) {}
  return null;
}

// Match bare image URLs (http/https ending in common image extensions) that
// the model emits without markdown syntax. Skips URLs already wrapped in
// markdown image/link syntax or HTML attributes via the lookbehind.
const _BARE_IMAGE_URL_RE =
  /(^|[\s>])(https?:\/\/[^\s<>"'`)]+\.(?:jpe?g|png|gif|webp|avif|svg)(?:\?[^\s<>"'`)]*)?)/gi;

function _promoteBareImageUrls(text) {
  // Split on fenced code blocks; only transform outside of them.
  const parts = text.split(/(```[\s\S]*?```)/g);
  for (let i = 0; i < parts.length; i += 2) {
    parts[i] = parts[i].replace(
      _BARE_IMAGE_URL_RE,
      (_m, lead, url) => `${lead}![image](${url})`,
    );
  }
  return parts.join('');
}

// ---------------------------------------------------------------------------
// Citation links
// ---------------------------------------------------------------------------

function _linkCitations(html) {
  const urlMap = {};
  const sourcePattern = /\[(\d+)\]:?\s+(?:[^<\n]*?)?(https?:\/\/[^\s<\[\]]+)/g;
  let m;
  while ((m = sourcePattern.exec(html)) !== null) {
    if (!urlMap[m[1]]) urlMap[m[1]] = m[2];
  }
  const urlLinePattern = /\[(\d+)\][^\n]*?\n[^[]*?URL:\s*(https?:\/\/[^\s<\[\]]+)/g;
  while ((m = urlLinePattern.exec(html)) !== null) {
    if (!urlMap[m[1]]) urlMap[m[1]] = m[2];
  }
  if (Object.keys(urlMap).length === 0) return html;

  html = html.replace(/\[(\d+)\](?!:?\s*https?:\/\/)(?!:?\s+[^\n]*?https?:\/\/)(?!\()/g, (full, num) => {
    const url = urlMap[num];
    if (!url) return full;
    return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-link" title="${url}">[${num}]</a>`;
  });

  for (const [num, url] of Object.entries(urlMap)) {
    const escapedUrl = url.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const sourceLineRe = new RegExp(`(\\[${num}\\]\\s*)(${escapedUrl})`, 'g');
    html = html.replace(sourceLineRe, (_match, prefix, href) => {
      return `${prefix}<a href="${href}" target="_blank" rel="noopener noreferrer" class="md-link">${href}</a>`;
    });
  }
  return html;
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

function _buildTable(lines) {
  const rows = lines
    .filter(l => !/^\|[\s:|-]+\|$/.test(l))
    .map(l => l.replace(/^\||\|$/g, '').split('|').map(c => c.trim()));
  if (rows.length === 0) return lines.join('\n');

  let tableHtml = '<table class="md-table"><thead><tr>';
  rows[0].forEach(cell => { tableHtml += `<th>${cell}</th>`; });
  tableHtml += '</tr></thead><tbody>';
  for (let i = 1; i < rows.length; i++) {
    tableHtml += '<tr>';
    rows[i].forEach(cell => { tableHtml += `<td>${cell}</td>`; });
    tableHtml += '</tr>';
  }
  tableHtml += '</tbody></table>';
  return tableHtml;
}

function _renderTables(html) {
  const lines = html.split('\n');
  const result = [];
  let tableLines = [];
  let inTable = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    const isTableRow = /^\|(.+)\|$/.test(line);
    const isSeparator = /^\|[\s:|-]+\|$/.test(line);
    if (isTableRow || isSeparator) {
      tableLines.push(line);
      inTable = true;
    } else {
      if (inTable && tableLines.length >= 2) {
        result.push(_buildTable(tableLines));
      } else if (inTable) {
        result.push(...tableLines);
      }
      tableLines = [];
      inTable = false;
      result.push(lines[i]);
    }
  }
  if (inTable && tableLines.length >= 2) {
    result.push(_buildTable(tableLines));
  } else if (inTable) {
    result.push(...tableLines);
  }
  return result.join('\n');
}

// ---------------------------------------------------------------------------
// Artifact card placeholder
// ---------------------------------------------------------------------------

function _buildArtifactCardPlaceholder(artifactId) {
  const _icons = {
    pdf: icons.filePdf, docx: icons.fileDocx, pptx: icons.filePptx,
    xlsx: icons.fileXlsx, png: icons.fileImage, default: icons.fileDefault
  };
  return `<div class="artifact-card" data-artifact-id="${artifactId}">` +
    `<div class="artifact-card-icon">${_icons.default}</div>` +
    `<div class="artifact-card-body">` +
    `<div class="artifact-card-name">Loading...</div>` +
    `<div class="artifact-card-meta"></div>` +
    `</div>` +
    `<button class="artifact-card-preview" data-preview-id="${artifactId}" title="Preview" onclick="window._toggleArtifactPreview(this)">${icons.eye}</button>` +
    `<button class="artifact-card-preview" data-cast-id="${artifactId}" title="Cast to TV" onclick="window._castArtifact(this)">` +
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" width="14" height="14" stroke-linecap="round" stroke-linejoin="round"><path d="M2 16.1A5 5 0 0 1 5.9 20"/><path d="M2 12.05A9 9 0 0 1 9.95 20"/><path d="M2 8V6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-6"/><line x1="2" y1="20" x2="2.01" y2="20"/></svg>` +
    `</button>` +
    `<button class="artifact-card-preview" data-edit-id="${artifactId}" title="Edit in Studio" onclick="window._openArtifactStudio(this)">` +
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>` +
    `</button>` +
    `<button class="artifact-card-preview" data-canvas-id="${artifactId}" title="Open in canvas" onclick="window._openArtifactCanvas(this)">` +
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" width="14" height="14" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><line x1="14" y1="4" x2="14" y2="20"/></svg>` +
    `</button>` +
    `<a href="/api/artifacts/${artifactId}/download" class="artifact-card-download" ` +
    `target="_blank" rel="noopener noreferrer" title="Download">${icons.download}</a>` +
    `</div>`;
}

// ---------------------------------------------------------------------------
// Line numbers
// ---------------------------------------------------------------------------

export function addLineNumbers(codeEl) {
  if (codeEl.dataset.numbered) return;
  codeEl.dataset.numbered = '1';
  const lines = codeEl.innerHTML.split('\n');
  if (lines.length < 2) return;
  if (lines[lines.length - 1].trim() === '') lines.pop();
  const digits = String(lines.length).length;
  const width = Math.max(digits, 2);
  codeEl.innerHTML = lines.map((line, i) => {
    const num = String(i + 1).padStart(width, ' ');
    return `<span class="code-line"><span class="code-line-number">${num}</span>${line}</span>`;
  }).join('\n');
}

// ---------------------------------------------------------------------------
// Defensive: rewrite hallucinated/malformed image-card HTML into markdown.
// ---------------------------------------------------------------------------

// Matches the malformed shape narrative LLMs produce when they imitate the
// rendered image card but drop the opening tag chars. Looks for an
// /api/image/{id} URL in quotes followed by an alt= attribute fragment,
// and any leading "md-image-wrapper">" noise. Captures the image id and
// the alt text. Also matches partial well-formed <img src=…> with the same
// API path, so a stray <img …> from the model is still rewritten.
// Matches the malformed shape narrative LLMs produce when they imitate the
// rendered image card but drop the opening tag chars. Quote chars in these
// patterns accept ANY of: straight " ' or curly "" '' — narrative mode's
// dialogue-styling pass can lead the model to emit curly quotes for what
// it thinks is "quoted text", which is how the broken pattern slipped past
// the original sanitizer.
//
//   Q  = any opening/closing quote codepoint
const _Q = `["'\\u2018\\u2019\\u201C\\u201D]`;

// URL shape inside the hallucinated markup: local /api/image/ID, any /api/ path,
// or an absolute http(s) URL. Kept permissive because narrative LLMs paste
// external image hosts (character card avatars, CDN URLs) verbatim.
const _IMG_URL = `((?:\\/api\\/|https?:\\/\\/)[^\\s${_Q.slice(1, -1)}<>]+)`;

const _HALLUCINATED_IMG_PATTERNS = [
  // Well-formed-ish <img src="URL" alt="ALT" …> — run FIRST so the
  // stripped-tag pattern below doesn't claim half of it.
  new RegExp(
    `<img\\s+[^>]*?src=${_Q}${_IMG_URL}${_Q}` +
    `[^>]*?(?:\\s+alt=${_Q}([^${_Q.slice(1, -1)}]*)${_Q})?[^>]*?\\/?>`,
    'gi',
  ),
  // Stripped-tag form: [Q md-image-wrapper Q >] Q URL Q  alt= Q ALT Q  …
  // [> Q md-image-caption Q > caption]
  //
  // Whitespace is permissive everywhere (\s*) because models emit this
  // pattern with inconsistent spacing — no space before `alt=`, extra
  // spaces, tabs, or line breaks between attributes. Tightening any of
  // these to \s+ re-breaks cases we've already seen in the wild.
  new RegExp(
    `(?:${_Q}[a-z][\\w-]*${_Q}\\s*>\\s*)*` +
    `${_Q}${_IMG_URL}${_Q}` +
    `\\s*(?:[a-z-]+=${_Q}[^${_Q.slice(1, -1)}]*${_Q}\\s*)*` +
    `alt=${_Q}([^${_Q.slice(1, -1)}]*)${_Q}` +
    `[^<\\n]*(?:>\\s*${_Q}md-image-caption${_Q}\\s*>\\s*[^<\\n]*)?`,
    'gi',
  ),
];

function _rewriteHallucinatedImageHtml(text) {
  let out = text;
  // Some pipelines (persisted narrative content, certain character-card
  // greetings, some streaming paths) hand us text where HTML has been
  // entity-escaped already — `&quot;md-image-wrapper&quot;&gt;&quot;/api/…`.
  // The regexes below look for literal quote/angle chars, so they'd miss
  // the entity form. Decode the small entity set that appears in this
  // specific shape before running the patterns. Safe because the later
  // escapeHtml pass in renderMarkdown re-escapes anything that survives.
  if (/&(?:quot|#34|gt|lt|amp);(?:[a-z/]|md-image)/i.test(out)) {
    out = out
      .replace(/&quot;/g, '"')
      .replace(/&#34;/g, '"')
      .replace(/&apos;/g, "'")
      .replace(/&#39;/g, "'")
      .replace(/&gt;/g, '>')
      .replace(/&lt;/g, '<')
      .replace(/&amp;/g, '&');
  }
  for (const pat of _HALLUCINATED_IMG_PATTERNS) {
    out = out.replace(pat, (_m, url, alt) => `![${(alt || 'Scene').trim()}](${url})`);
  }

  // Last-resort sweep, line-by-line: any line that (a) contains an
  // Augmentum image URL and (b) has HTML-attribute soup signatures gets
  // replaced with a clean `![image](url)` markdown image. This catches
  // whatever exotic shape the structured patterns above missed —
  // stray `<` that breaks HTML parsing into attribute fragments, odd
  // quote arrangements, models mixing tag types, etc. The prose on that
  // line was garbage anyway, so dropping it is strictly better than
  // leaking `<span`-style fragments into the DOM.
  out = out.split('\n').map((line) => {
    const urlMatch = line.match(
      /(\/api\/(?:browse\/image\?url=|image\/|artifacts\/[a-zA-Z0-9_-]+\/preview)[^\s"'`<>)\u2018\u2019\u201C\u201D]+)/,
    );
    if (!urlMatch) return line;
    const hasSoup = /(?:^|[\s"'\u2018\u2019\u201C\u201D])(?:class|alt|loading|referrerpolicy|crossorigin)\s*=|md-image|md-image-wrapper|md-image-caption/i.test(line);
    if (!hasSoup) return line;
    return `![image](${urlMatch[1]})`;
  }).join('\n');

  return out;
}

// ---------------------------------------------------------------------------
// Main Markdown→HTML renderer
// ---------------------------------------------------------------------------

/**
 * Render markdown text to HTML.
 * @param {string} text - Raw markdown text
 * @param {object} options
 * @param {string} options.mode - Current UI mode ('narrative', 'passthrough', etc.)
 * @returns {string} HTML string
 */
export function renderMarkdown(text, options = {}) {
  if (!text) return '';
  const mode = options.mode || 'passthrough';
  // Compact surfaces (browse / studio / youtube AI blocks) share this
  // renderer for visual consistency with chat — same headings, tables,
  // images, links, and syntax highlighting — but drop the chat-only code
  // toolbar (Run / Edit / Ask AI / Library / Download / quick-actions),
  // whose click handlers are delegated from the chat surface and would be
  // dead buttons elsewhere. The Copy button stays (no host wiring needed).
  const compact = options.compact === true;

  // Defensive: small / weak narrative LLMs sometimes hallucinate the
  // *rendered* image-card markup we'd produce ourselves, but with the
  // opening `<` of every tag dropped. Result is plain text like:
  //   "md-image-wrapper">"/api/image/abc" alt="Scene" class="md-image" …
  // Detect that shape and rewrite it to a clean ![alt](url) before the
  // markdown pass so the image actually renders. The src= form covers
  // the case where the model emits a partial img tag too.
  text = _rewriteHallucinatedImageHtml(text);
  text = _promoteBareImageUrls(text);

  // Miscounted-fence normalization: LLMs in narrative mode frequently emit
  // 2 or 4+ backticks instead of 3 when opening a panel block. Rewrite those
  // near-miss fences to the canonical 3-backtick form so the code-block regex
  // and panel routing can see them. Scoped to known panel langs only —
  // never touches code fences where a weird backtick count might be deliberate.
  text = _normalizeNearMissPanelFences(text);

  // Narrative-mode anti-pattern: LLMs sometimes wrap an ENTIRE reply in a
  // single ```md fence. Collapsing such a panel by default would leave the
  // bubble looking empty. Unwrap before any code-block routing so the
  // content renders as ordinary narrative prose. Only applies in narrative
  // mode and only at the top level (not inside recursive panel rendering).
  if (mode === 'narrative' && !options._insidePanel) {
    text = unwrapWholeMessageMarkdownFence(text);
  }

  let html = escapeHtml(text);

  // Sentinel for placeholder slots. \x01 (SOH) never appears in user content
  // and survives escapeHtml / DOM serialization, so a later underscore-bold
  // pass can't accidentally devour `__CODE_BLOCK_0__`-shaped tokens.
  const SE = '\x01';

  // Code blocks — protect content inside (including mermaid)
  const _seenBlockIds = {};
  const codeBlocks = [];
  html = html.replace(/```(\w*)\s*\n([\s\S]*?)```/g, (_match, lang, code) => {
    const placeholder = `${SE}CB${codeBlocks.length}${SE}`;
    const rawCode = unescapeHtml(code.trimEnd());
    const langLowerEarly = (lang || '').toLowerCase();

    // Narrative panel routing — ```md / ```stats / ```scene / ```system /
    // ```gm / ```note become collapsible panels instead of code blocks.
    // Only in narrative mode and only at top-level (not inside a recursive
    // panel render — guarded by _insidePanel option).
    if (mode === 'narrative' && !options._insidePanel && _NARRATIVE_PANEL_LANGS.has(langLowerEarly) && rawCode.trim()) {
      codeBlocks.push(_buildNarrativePanel(lang, rawCode, mode, options));
      return placeholder;
    }

    if (lang === 'mermaid') {
      codeBlocks.push(`<div class="mermaid-block" data-mermaid="${encodeURIComponent(rawCode)}">${escapeHtml(rawCode)}</div>`);
    } else {
      const langClass = lang ? ` class="language-${escapeHtml(lang)}"` : '';
      const langLower = lang.toLowerCase();
      const encodedRaw = encodeURIComponent(rawCode);

      let blockId = blockFingerprint(langLower, rawCode);
      if (_seenBlockIds[blockId]) { blockId += '_' + (++_seenBlockIds[blockId]); }
      else { _seenBlockIds[blockId] = 1; }

      // Compact surfaces drop the code-header entirely: its buttons
      // (Copy/Run/Edit/…) are delegated from the chat container and would
      // be dead elsewhere. The highlighted ``<pre><code class="language-…">``
      // below still renders identically — only the toolbar differs.
      let langLabel = '';
      if (!compact) {
        let actionBtns = '';
        if (langLower === 'html') {
          actionBtns += `<button class="code-action-btn" data-action="preview-code" title="Preview in sandbox">Preview</button>`;
          actionBtns += `<button class="code-action-btn" data-action="save-to-library" title="Save to Library">Library</button>`;
        } else if (langLower === 'svg') {
          actionBtns += `<button class="code-action-btn" data-action="preview-svg" title="Render SVG">Preview</button>`;
        } else if (langLower === 'python' || langLower === 'py') {
          actionBtns += `<button class="code-action-btn" data-action="run-code" title="Run in sandbox">Run</button>`;
        } else if (langLower === 'javascript' || langLower === 'js') {
          actionBtns += `<button class="code-action-btn" data-action="run-code" title="Run in browser sandbox">Run</button>`;
        }
        if (lang) {
          if (['html', 'htm', 'javascript', 'js', 'jsx', 'css', 'scss'].includes(langLower)) {
            actionBtns += `<button class="code-action-btn" data-action="auto-fix" title="Auto-fix syntax errors (no LLM)">Fix</button>`;
          }
          actionBtns += `<button class="code-action-btn" data-action="ask-ai-edit" title="Ask AI to edit">Ask AI</button>`;
          actionBtns += `<button class="code-action-btn" data-action="edit-code" title="Edit code">Edit</button>`;
          actionBtns += `<button class="code-action-btn code-quick-actions-trigger" data-action="quick-actions" title="Quick actions">&#9662;</button>`;
        }
        actionBtns += `<button class="copy-code-btn" data-copy="${encodedRaw}">Copy</button>`;
        actionBtns += `<button class="code-action-btn" data-action="download-code" title="Download as file">Download</button>`;

        langLabel = lang
          ? `<div class="code-header" data-raw-code="${encodedRaw}" data-lang="${escapeHtml(langLower)}" data-version-idx="0" data-block-id="${blockId}"><span>${escapeHtml(lang)}</span><div class="code-header-actions">${actionBtns}</div></div>`
          : '';
      }
      codeBlocks.push(`${langLabel}<pre><code${langClass}>${code.trimEnd()}</code></pre>`);
    }
    return placeholder;
  });

  // Inline code
  const inlineCodes = [];
  html = html.replace(/`([^`\n]+)`/g, (_match, code) => {
    const placeholder = `${SE}IC${inlineCodes.length}${SE}`;
    inlineCodes.push(`<code>${code}</code>`);
    return placeholder;
  });

  // Artifact download links → rich cards
  const artifactCards = [];
  html = html.replace(
    /Download:\s*\/api\/artifacts\/([a-zA-Z0-9_-]+)\/download/g,
    (_match, artifactId) => {
      const placeholder = `${SE}AC${artifactCards.length}${SE}`;
      artifactCards.push(_buildArtifactCardPlaceholder(artifactId));
      return placeholder;
    }
  );

  // Images: ![alt](url) — placeholder so the URL can't be eaten by a later
  // bold/italic pass (e.g. `?q=*test*` becoming `?q=<em>test</em>`).
  const images = [];
  html = html.replace(
    /!\[([^\]]*)\]\(([^)]+)\)/g,
    (_match, alt, src) => {
      const safeSrc = _sanitizeMediaUrl(src);
      if (!safeSrc) return _match;
      const safeAlt = alt || '';
      const captionHtml = safeAlt && safeAlt !== 'image'
        ? `<span class="md-image-caption">${safeAlt}</span>` : '';
      const placeholder = `${SE}IM${images.length}${SE}`;
      images.push(`<div class="md-image-wrapper"><img src="${safeSrc}" alt="${safeAlt || 'image'}" class="md-image" loading="lazy" referrerpolicy="no-referrer" crossorigin="anonymous">${captionHtml}</div>`);
      return placeholder;
    }
  );

  // Links: [text](url) — placeholder for same URL-safety reason as images.
  // Link text itself ALSO can't run inline emphasis after this point, which
  // is the documented tradeoff (`[**bold link**](url)` will render literally).
  // The asterisks-in-URLs class of bug is the higher-volume one to prevent.
  const links = [];
  html = html.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    (_match, text, href) => {
      const safeHref = _sanitizeUrl(href);
      if (!safeHref) return text;
      const placeholder = `${SE}LK${links.length}${SE}`;
      links.push(`<a href="${safeHref}" target="_blank" rel="noopener noreferrer" class="md-link">${text}</a>`);
      return placeholder;
    }
  );

  // Smart typography on prose-only (code/links/images are sentineled now).
  // `--` between non-whitespace, non-dash chars → em-dash. The non-dash
  // lookarounds keep `----------` (long HR) and `foo----bar` intact so the
  // HR pass below can still recognise them.
  // `...` → ellipsis. Smart quotes deliberately skipped — narrative dialogue
  // handles its own quote pair below, and outside narrative mode straight
  // quotes are correct (command lines, code-style snippets in prose).
  html = html.replace(/(?<=[^\s-])--(?=[^\s-])/g, '—');
  html = html.replace(/\.\.\./g, '…');

  // Citations
  html = _linkCitations(html);

  // Autolinks — wrap bare http/https URLs in prose. Runs AFTER citations
  // (which already linkifies source-line URLs) and BEFORE bold/italic /
  // dialogue so the captured URLs are sealed as `${SE}LK${SE}` placeholders
  // and can't be eaten by `*X*` emphasis spanning them (`*see https://x*`).
  //
  // Negative lookbehind excludes chars that signal "we're already inside
  // an HTML tag/attribute" — letter/digit (middle-of-word), `>` (just
  // closed an opening tag), `="'\`` (inside an attribute value), `/` (URL
  // continuation). Everything else — whitespace, `(`, `[`, `*`, `_`, `.`,
  // `,`, `:` etc. — is a valid prose-position to start a link from. This
  // lets `*see https://x*` linkify the URL inside the emphasis run.
  //
  // Trailing sentence punctuation is trimmed back into prose: `https://x.`
  // becomes a link to `https://x` followed by literal `.`. Closing parens
  // are trimmed only when they'd leave the URL unbalanced — Wikipedia-style
  // `Function_(mathematics)` keeps the inner `)` because it has a matching
  // `(` inside the URL.
  html = html.replace(/(?<![a-zA-Z0-9>="'`/])(https?:\/\/[^\s<>"'`‘’“”]+)/g, (url) => {
    let clean = url;
    let trailing = '';
    while (clean.length > 0 && /[.,;:!?]$/.test(clean)) {
      trailing = clean.slice(-1) + trailing;
      clean = clean.slice(0, -1);
    }
    while (clean.endsWith(')')) {
      const opens = (clean.match(/\(/g) || []).length;
      const closes = (clean.match(/\)/g) || []).length;
      if (closes <= opens) break;
      trailing = clean.slice(-1) + trailing;
      clean = clean.slice(0, -1);
    }
    if (clean.length < 10) return url;
    const safe = _sanitizeUrl(clean);
    if (!safe) return url;
    const placeholder = `${SE}LK${links.length}${SE}`;
    links.push(`<a href="${safe}" target="_blank" rel="noopener noreferrer" class="md-link">${clean}</a>`);
    return `${placeholder}${trailing}`;
  });

  // Tables
  html = _renderTables(html);

  // Horizontal rules \u2014 CommonMark accepts three or more of `-`, `*`, or `_`,
  // optionally separated by spaces. Each form needs its own line to avoid
  // matching a setext-style heading underline (`===` / `---` below text).
  html = html.replace(/^[ ]{0,3}-(?:[ ]*-){2,}[ ]*$/gm, '<hr>');
  html = html.replace(/^[ ]{0,3}\*(?:[ ]*\*){2,}[ ]*$/gm, '<hr>');
  html = html.replace(/^[ ]{0,3}_(?:[ ]*_){2,}[ ]*$/gm, '<hr>');

  // Headings \u2014 H1 through H6. Order longest-first so `####` doesn't get
  // consumed by the H3 pattern.
  html = html.replace(/^###### (.+)$/gm, '<h6>$1</h6>');
  html = html.replace(/^##### (.+)$/gm, '<h5>$1</h5>');
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold / italic \u2014 moved to emphasis.js (bounded, linear-time bodies;
  // the old `[\s\S]+?` inner regexes were quadratic on content with
  // unmatched `*`/`_` \u2014 see the module header for the numbers). Same
  // guards as before: `2 * 3` math, `* ` list markers, `my__var__name`
  // and snake_case all stay literal; narrative `*action*` spans still
  // cross a soft line break (never a blank line \u2014 which produced broken
  // HTML once the paragraph pass split the tag pair anyway).
  html = applyInlineEmphasis(html);

  // Quoted dialogue (narrative mode only). escapeHtml turns ASCII `"` into
  // `&#34;` for XSS-safe attribute interpolation, so the straight-quote arm
  // matches the *encoded* form \u2014 the literal `"` arm would never fire on
  // user content. Curly quotes pass through escapeHtml untouched, so that
  // arm matches them directly.
  if (mode === 'narrative') {
    html = html.replace(/&#34;([^&<>]+?)&#34;/g,
      '<span class="dialogue"><span class="dq">\u201C</span>$1<span class="dq">\u201D</span></span>');
    html = html.replace(/\u201C([^\u201D<>]+?)\u201D/g,
      '<span class="dialogue"><span class="dq">\u201C</span>$1<span class="dq">\u201D</span></span>');
  }

  // Strikethrough
  html = html.replace(/~~(.+?)~~/g, '<del>$1</del>');

  // Restore link/image placeholders BEFORE the block-level passes below \u2014
  // blockquote merging and the paragraph cleanup at the bottom both check
  // for actual `<div class="md-image-wrapper">` / link tags to hoist them
  // out of `<p>` wrappers correctly. Code blocks stay sentineled longer
  // because they don't need to be visible to the list/blockquote scanners.
  //
  // ONE scan for all slots \u2014 the old per-placeholder `html.replace(str)`
  // loop rescanned the whole string per slot (O(slots \u00d7 message), ~100ms
  // at 500 slots in a 1MB message). The function replacer also closes a
  // latent bug: a string-arg replacement containing `$&`/`$'` was
  // pattern-interpolated by String.replace and corrupted the output.
  html = html.replace(/\x01(IM|LK)(\d+)\x01/g, (m, kind, i) => {
    const slot = kind === 'IM' ? images[+i] : links[+i];
    return slot !== undefined ? slot : m;
  });

  // GFM-style alert callouts: a blockquote opening with `[!NOTE]` /
  // `[!TIP]` / `[!IMPORTANT]` / `[!WARNING]` / `[!CAUTION]` becomes a
  // distinguished panel rather than a generic blockquote. Must run BEFORE
  // the blockquote merger below or those alert lines would just collapse
  // into a flat <blockquote>. Body lines stay `&gt; `-prefixed in the
  // capture and get stripped here.
  const _ALERT_LABEL = {
    NOTE: 'Note', TIP: 'Tip', IMPORTANT: 'Important',
    WARNING: 'Warning', CAUTION: 'Caution',
  };
  html = html.replace(
    /^&gt; \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*\n((?:&gt; [^\n]*(?:\n|$))*)/gm,
    (_m, type, body) => {
      const kind = type.toLowerCase();
      const label = _ALERT_LABEL[type];
      const innerBody = body
        .replace(/\n$/, '')
        .split('\n')
        .map(l => l.replace(/^&gt; ?/, ''))
        .filter(Boolean)
        .join('<br>');
      const bodyHtml = innerBody ? `<div class="md-alert-body">${innerBody}</div>` : '';
      return `<div class="md-alert md-alert-${kind}"><div class="md-alert-title">${label}</div>${bodyHtml}</div>\n`;
    }
  );

  // Blockquotes \u2014 fold consecutive `&gt; ` lines into one <blockquote>
  // so a 4-line citation block reads as one quote, not four glued boxes.
  // Inner line-breaks become <br> so the original line shape survives.
  html = html.replace(/(?:^&gt; [^\n]*(?:\n|$))+/gm, (block) => {
    const inner = block
      .replace(/\n$/, '')
      .split('\n')
      .map(l => l.replace(/^&gt; /, ''))
      .join('<br>');
    return `<blockquote>${inner}</blockquote>\n`;
  });

  // Lists \u2014 accept `-`, `*`, `+` as unordered bullet markers (CommonMark);
  // tag UL vs OL with sentinels so the wrap pass can separate them cleanly.
  // A mixed `- foo\n1. bar` now produces `<ul>\u2026</ul><ol>\u2026</ol>`, not one
  // misclassified `<ul>` containing both. Task-list prefixes `[ ]` / `[x]`
  // turn into a disabled checkbox + `.task-list-item` class so the list
  // renders as a GFM-style todo.
  html = html.replace(/^[ ]{0,3}[-*+] (\[[ xX]\] )?(.+)$/gm, (_m, check, content) => {
    if (!check) return `${SE}UL${SE}<li>${content}</li>`;
    const checked = /[xX]/.test(check);
    const box = `<input type="checkbox" disabled${checked ? ' checked' : ''} class="task-list-checkbox">`;
    return `${SE}UL${SE}<li class="task-list-item">${box} ${content}</li>`;
  });
  html = html.replace(/^[ ]{0,3}\d+\. (.+)$/gm, `${SE}OL${SE}<li>$1</li>`);
  // Wrap regexes accept <li> with optional attributes so .task-list-item
  // lines are caught alongside plain <li>.
  html = html.replace(new RegExp(`(?:${SE}UL${SE}<li[^>]*>[^\\n]*</li>\\n?)+`, 'g'),
    (m) => `<ul>${m.split(`${SE}UL${SE}`).join('')}</ul>`);
  html = html.replace(new RegExp(`(?:${SE}OL${SE}<li[^>]*>[^\\n]*</li>\\n?)+`, 'g'),
    (m) => `<ol>${m.split(`${SE}OL${SE}`).join('')}</ol>`);

  // Paragraphs
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  html = '<p>' + html + '</p>';

  // Clean up
  html = html.replace(/<p><\/p>/g, '');
  html = html.replace(/<p>(<h[1-6]>)/g, '$1');
  html = html.replace(/(<\/h[1-6]>)<\/p>/g, '$1');
  html = html.replace(/<p>(<ul>)/g, '$1');
  html = html.replace(/(<\/ul>)<\/p>/g, '$1');
  html = html.replace(/<p>(<ol>)/g, '$1');
  html = html.replace(/(<\/ol>)<\/p>/g, '$1');
  html = html.replace(/<p>(<blockquote>)/g, '$1');
  html = html.replace(/(<\/blockquote>)<\/p>/g, '$1');
  html = html.replace(/<p>(<pre>)/g, '$1');
  html = html.replace(/(<\/pre>)<\/p>/g, '$1');
  html = html.replace(/<p>(<hr>)<\/p>/g, '$1');
  html = html.replace(/<p>(<hr>)/g, '$1');
  html = html.replace(/(<hr>)<\/p>/g, '$1');
  html = html.replace(/<p>(<div class="code-header">)/g, '$1');
  html = html.replace(/<p>(<div class="md-image-wrapper">)/g, '$1');
  html = html.replace(/<p>(<div class="narrative-panel)/g, '$1');
  html = html.replace(/<p>(<div class="md-alert)/g, '$1');
  html = html.replace(/(<\/div>)<\/p>/g, '$1');
  html = html.replace(/<p>(<table)/g, '$1');
  html = html.replace(/(<\/table>)<\/p>/g, '$1');

  // Restore remaining sentinel placeholders. Images and links were restored
  // earlier (before block-level passes) so their cleanup rules fire above.
  // Single scan for the same reasons as the IM/LK pass above (a coder
  // summary with hundreds of inline-code spans paid the quadratic loop
  // hardest here).
  html = html.replace(/\x01(CB|IC|AC)(\d+)\x01/g, (m, kind, i) => {
    const slot = kind === 'CB' ? codeBlocks[+i]
      : kind === 'IC' ? inlineCodes[+i]
      : artifactCards[+i];
    return slot !== undefined ? slot : m;
  });

  return html;
}

// ---------------------------------------------------------------------------
// Code highlighting
// ---------------------------------------------------------------------------

/**
 * Per-block size cap for syntax highlighting. hljs runs dozens of regex
 * passes over the block; on a multi-hundred-KB block a single
 * highlightElement call blocks the main thread for seconds (hljs itself
 * documents ~30KB as the sensible ceiling). Oversized blocks keep their
 * plain <pre><code> rendering — still monospaced, still correct — and
 * are stamped so no later pass retries them. addLineNumbers shares the
 * cap: it re-splits and rebuilds the block's innerHTML, so it's the
 * same order of cost on the same blocks.
 */
const _HIGHLIGHT_MAX_CHARS = 100_000;

/** Highlight one code block, but never let an UNKNOWN fence language spam the
 *  console. When a block is tagged `language-spectra` (or any grammar hljs
 *  doesn't have), highlight.js logs "Could not find the language 'spectra'"
 *  PER BLOCK and falls back to no-highlight anyway. We pre-map such blocks to
 *  `plaintext` first: identical result (monospace + .hljs styling), zero
 *  warnings. Exported so EVERY highlightElement caller shares the guard —
 *  code-edit / code-actions / project re-highlight blocks that may carry an
 *  arbitrary chat fence language, so the fix belongs at the call, not one
 *  surface. No-op when hljs isn't loaded. */
export function safeHighlightElement(block) {
  if (typeof hljs === 'undefined' || !block) return;
  const m = (block.className || '').match(/\blanguage-([^\s]+)/);
  const lang = m && m[1];
  if (lang && typeof hljs.getLanguage === 'function' && !hljs.getLanguage(lang)) {
    block.classList.remove(`language-${lang}`);
    block.classList.add('language-plaintext');
  }
  hljs.highlightElement(block);
}

/** Highlight + number one block, honoring the size cap. Safe to call on
 *  an already-processed block (dataset guards make it a no-op). */
function _highlightBlock(block) {
  if (!block.dataset.highlighted) {
    if ((block.textContent || '').length > _HIGHLIGHT_MAX_CHARS) {
      // Stamp so neither this pass nor any future sweep retries; the
      // tooltip tells a curious user why this block has no colors.
      block.dataset.highlighted = 'skipped-size';
      const pre = block.closest('pre');
      if (pre && !pre.title) pre.title = 'Syntax highlighting skipped (large block)';
    } else {
      safeHighlightElement(block);
    }
  }
  if (
    !block.dataset.numbered
    && block.dataset.highlighted !== 'skipped-size'
    && block.closest('.code-header + pre, .message-content pre')
  ) {
    addLineNumbers(block);
  }
}

/**
 * Highlight code blocks in a container using hljs.
 * Accepts optional post-processing hooks for features that live in other modules.
 * @param {HTMLElement} container - DOM element to search for code blocks
 * @param {object} hooks - Optional post-processors
 * @param {function} hooks.mermaid - Render mermaid diagrams
 * @param {function} hooks.artifactCards - Load artifact card metadata
 * @param {function} hooks.codeVersions - Restore code block version indicators
 */
export function highlightCode(container, hooks = {}) {
  if (typeof hljs !== 'undefined') {
    const blocks = (container || document).querySelectorAll("pre code[class*='language-']");
    blocks.forEach(_highlightBlock);
  }
  if (hooks.mermaid) hooks.mermaid(container);
  if (hooks.artifactCards) hooks.artifactCards(container);
  if (hooks.codeVersions) hooks.codeVersions(container);
}

// Containers with an idle-slice loop already in flight. A stream that
// promotes paragraph-by-paragraph calls highlightCodeDeferred once per
// promotion; without this guard each call queued its own full-container
// sweep. The in-flight loop re-queries unprocessed blocks every slice,
// so it naturally picks up blocks added after it started.
const _deferredInFlight = new WeakSet();

/**
 * Deferred variant of highlightCode — processes blocks on the browser's
 * idle queue in TIME-BUDGETED SLICES so the user sees styled text
 * immediately, colors fill in once the main thread is free, and no single
 * slice holds the thread longer than the idle deadline (a hundred-block
 * thread used to highlight in ONE idle callback — "deferred" but still a
 * multi-hundred-ms task once it fired; worse, the old 500ms timeout FORCED
 * it to run mid-jank instead of waiting for actual idle).
 *
 * Falls back to setTimeout(0) with a synthetic ~8ms deadline where
 * requestIdleCallback is unavailable (Safari <16.4).
 *
 * Post-hooks (mermaid / artifact cards / code versions) and the math
 * renderer run once, after the last slice — they operate on the whole
 * container and are cheap relative to highlighting.
 */
export function highlightCodeDeferred(container, hooks = {}) {
  const target = container || document;
  if (_deferredInFlight.has(target)) return;

  const schedule = (fn) => {
    if (typeof window !== 'undefined' && typeof window.requestIdleCallback === 'function') {
      window.requestIdleCallback(fn);
    } else {
      setTimeout(() => fn({ timeRemaining: () => 8, didTimeout: false }), 0);
    }
  };

  const finish = () => {
    _deferredInFlight.delete(target);
    try {
      if (hooks.mermaid) hooks.mermaid(container);
      if (hooks.artifactCards) hooks.artifactCards(container);
      if (hooks.codeVersions) hooks.codeVersions(container);
    } catch { /* hooks are best-effort */ }
    // Dynamic import so the math module (which pulls in KaTeX lazily)
    // only costs anything when there's actually content to render.
    import('../math-renderer.js')
      .then(m => m.renderMathIn(container))
      .catch(() => { /* math-renderer missing is fine, render as source */ });
  };

  const step = (deadline) => {
    if (typeof hljs === 'undefined') {
      // hljs script not loaded yet — don't spin; the next caller retries.
      finish();
      return;
    }
    let processed = 0;
    try {
      const pending = target.querySelectorAll("pre code[class*='language-']:not([data-highlighted])");
      for (const block of pending) {
        // Always make progress (>=1 per slice) so a stingy deadline can't
        // stall the loop forever; otherwise stop when the idle budget is
        // spent and let the browser paint / handle input.
        if (processed > 0 && deadline.timeRemaining() <= 3) break;
        try { _highlightBlock(block); } catch { /* hljs grammar hiccup — skip block */ }
        processed++;
      }
      if (target.querySelector("pre code[class*='language-']:not([data-highlighted])")) {
        schedule(step);
        return;
      }
    } catch { /* container detached mid-loop — fall through to finish */ }
    finish();
  };

  _deferredInFlight.add(target);
  schedule(step);
}
