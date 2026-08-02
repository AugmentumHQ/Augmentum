/**
 * coder-conversation.js — Structured conversation renderer for coder mode.
 *
 * Renders agent interactions as typed messages: user prompts, tool activity
 * cards (collapsible with diffs/shell output), and agent responses.
 *
 * Reuses:
 *   - renderMarkdown() from chat/markdown.js
 *   - computeLineDiff() + renderDiffLines() from chat/code-edit.js
 *   - escapeHtml() from app.js
 *   - icons from chat/constants.js
 */

import { escapeHtml } from './app.js';
import { renderMarkdown, highlightCodeDeferred } from './chat/markdown.js';
import { renderStreamSplit, newSplitState } from './chat/stream-render.js';
import { computeLineDiff, renderDiffLines } from './chat/code-edit.js';
import { nodesToDetach, nodesToRehydrate } from './coder-window.js';
import { yieldToPaint } from './coder-stream.js';

// DOM windowing — cap the number of LIVE top-level conversation nodes so the
// per-frame layout cost during a stream stays O(window) instead of O(entire
// session). Detached nodes are retained (not destroyed) and re-attached
// exactly on scroll-up. See coder-window.js for the rationale + the pure math.
const _MAX_LIVE_NODES = 80;   // newest message/tool nodes kept attached
const _TRIM_BATCH = 20;       // hysteresis: only trim once this far over the cap
const _REHYDRATE_BATCH = 30;  // nodes re-attached per scroll-to-top
const _REHYDRATE_SCROLL_PX = 120; // re-hydrate when within this of the top

// History-load budgeting. loadHistory used to render the ENTIRE session in
// one synchronous task — ~900 messages = a multi-second main-thread block on
// every page load / workspace switch, in every connected browser. Now the
// newest _HISTORY_TAIL_SYNC messages render synchronously (instant visible
// tail) and everything older backfills upward in _HISTORY_BUDGET_MS slices
// with a paint/input yield between slices — the same budget discipline as
// the stream pump's _DISPATCH_BUDGET_MS in coder-stream.js.
const _HISTORY_TAIL_SYNC = 80;   // matches the _MAX_LIVE_NODES window
const _HISTORY_BUDGET_MS = 8;

// Lazy-loaded CodeMind for syntax validation of agent edits
let _codeMindLoaded = false;
let _codeMindValidate = null;

async function _ensureCodeMind() {
  if (_codeMindLoaded) return;
  try {
    const cm = await import('./codemind.js');
    _codeMindValidate = cm.validate;
    _codeMindLoaded = true;
  } catch {
    _codeMindLoaded = true; // Don't retry on failure
  }
}

// Map file extensions to CodeMind language names
const _EXT_TO_LANG = {
  js: 'javascript', jsx: 'javascript', mjs: 'javascript',
  ts: 'typescript', tsx: 'typescript',
  py: 'python',
  html: 'html', htm: 'html',
  css: 'css', scss: 'css',
  json: 'json',
};

function _langFromPath(path) {
  const ext = (path || '').split('.').pop()?.toLowerCase() || '';
  return _EXT_TO_LANG[ext] || null;
}

// ── Tool display config ─────────────────────────────────────────────────────
// Pre-2026-04-21 these were emoji (📄 📝 ✏️ …). Problems: inconsistent
// rendering per OS, oversized on some platforms, and mixed visual
// weight with the rest of the app which uses SVG line icons
// everywhere else. Swapping to Feather-style inline SVG matches the
// existing icon-btn vocabulary (see the panel headers / workspace bar)
// and scales cleanly with font size. `currentColor` + 14×14 means they
// inherit tool-status color for free.

const _SVG_ATTRS = 'viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"';

const TOOL_LABELS = {
  file_read:    { verb: 'Read',    icon: `<svg ${_SVG_ATTRS}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="13" y2="17"/></svg>` },
  file_write:   { verb: 'Write',   icon: `<svg ${_SVG_ATTRS}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="15" y2="15"/></svg>` },
  file_list:    { verb: 'List',    icon: `<svg ${_SVG_ATTRS}><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>` },
  code_edit:    { verb: 'Edit',    icon: `<svg ${_SVG_ATTRS}><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>` },
  code_grep:    { verb: 'Grep',    icon: `<svg ${_SVG_ATTRS}><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>` },
  find_files:   { verb: 'Glob',    icon: `<svg ${_SVG_ATTRS}><polyline points="3 7 9 12 3 17"/><line x1="13" y1="17" x2="21" y2="17"/></svg>` },
  code_search:  { verb: 'Search',  icon: `<svg ${_SVG_ATTRS}><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>` },
  shell_exec:   { verb: 'Shell',   icon: `<svg ${_SVG_ATTRS}><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>` },
  shell_read:   { verb: 'Shell',   icon: `<svg ${_SVG_ATTRS}><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>` },
  doc_search:   { verb: 'Docs',    icon: `<svg ${_SVG_ATTRS}><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>` },
  doc_fetch:    { verb: 'Fetch',   icon: `<svg ${_SVG_ATTRS}><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>` },
  dir_tree:     { verb: 'Tree',    icon: `<svg ${_SVG_ATTRS}><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><circle cx="3.5" cy="6" r="1"/><circle cx="3.5" cy="12" r="1"/><circle cx="3.5" cy="18" r="1"/></svg>` },
  git:          { verb: 'Git',     icon: `<svg ${_SVG_ATTRS}><circle cx="6" cy="3" r="2"/><circle cx="6" cy="21" r="2"/><circle cx="18" cy="12" r="2"/><path d="M6 5v14"/><path d="M6 12h8a4 4 0 0 0 4-2"/></svg>` },
  test_run:     { verb: 'Test',    icon: `<svg ${_SVG_ATTRS}><path d="M10 2v7.5L4 18a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3l-6-8.5V2"/><line x1="8" y1="2" x2="16" y2="2"/></svg>` },
  env_info:     { verb: 'Env',     icon: `<svg ${_SVG_ATTRS}><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>` },
  multi_edit:   { verb: 'Edit×',   icon: `<svg ${_SVG_ATTRS}><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/><line x1="3" y1="20" x2="7" y2="20"/></svg>` },
  code_edit_batch: { verb: 'Edit', icon: `<svg ${_SVG_ATTRS}><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/><line x1="3" y1="20" x2="7" y2="20"/></svg>` },
  apply_patch:  { verb: 'Patch',   icon: `<svg ${_SVG_ATTRS}><polygon points="13 2 13 11 22 11 22 22 2 22 2 2 13 2"/><line x1="2" y1="11" x2="13" y2="11"/></svg>` },
  subagent:     { verb: 'Subtask', icon: `<svg ${_SVG_ATTRS}><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>` },
  task_dispatch: { verb: 'Dispatch', icon: `<svg ${_SVG_ATTRS}><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>` },
};
const _FALLBACK_ICON = `<svg ${_SVG_ATTRS}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`;

/**
 * Build a one-line description of a tool call for the collapsed summary.
 * `icon` is a raw SVG string (trusted source, safe to interpolate
 * unescaped); `verb` and `desc` must be escapeHtml'd at the call site.
 */
function _toolSummary(tool, input) {
  const cfg = TOOL_LABELS[tool] || { icon: _FALLBACK_ICON, verb: tool };
  let desc = '';
  if (tool === 'shell_exec' || tool === 'shell_read') {
    desc = input.command || '';
  } else if (tool === 'code_grep') {
    desc = `"${input.pattern || ''}"`;
  } else if (tool === 'code_edit') {
    desc = input.path || '';
  } else if (tool === 'code_edit_batch') {
    const editCount = Array.isArray(input.edits) ? input.edits.length : 0;
    const path = input.path || '';
    desc = path ? `${path} · ${editCount} edit${editCount === 1 ? '' : 's'}` : `${editCount} edits`;
  } else if (tool === 'apply_patch') {
    const paths = _patchFilePaths(input.patch || '');
    desc = paths.length === 0
      ? '(empty patch)'
      : paths.length === 1
        ? paths[0]
        : `${paths.length} files`;
  } else if (tool === 'task_dispatch') {
    const role = String(input.role || '').trim() || 'subagent';
    const prompt = String(input.prompt || '').trim();
    const preview = prompt.length > 80 ? prompt.slice(0, 77) + '…' : prompt;
    desc = preview ? `${role}: ${preview}` : role;
  } else if (input.path) {
    desc = input.path;
  } else if (input.query) {
    desc = input.query;
  } else if (input.pattern) {
    desc = input.pattern;
  } else if (input.url) {
    desc = input.url;
  }
  // Strip the /workspace/ prefix — users already know that's the
  // workspace root; seeing it on every card is noise.
  desc = desc.replace(/^\/workspace\//, '');
  return { icon: cfg.icon, verb: cfg.verb, desc };
}

/**
 * Extract file paths from a unified-diff patch string. Mirrors the
 * server-side parser in ApplyPatchTool._workspace_paths_from_patch so
 * the card summary reflects what the patch actually touched.
 */
function _patchFilePaths(patch) {
  const paths = [];
  const seen = new Set();
  if (!patch) return paths;
  for (const line of String(patch).split('\n')) {
    if (!(line.startsWith('+++ ') || line.startsWith('--- '))) continue;
    let raw = line.slice(4).trim().split('\t', 1)[0].split(' ', 1)[0];
    if (raw === '/dev/null') continue;
    if (raw.startsWith('a/') || raw.startsWith('b/')) raw = raw.slice(2);
    if (!raw) continue;
    if (raw.startsWith('/workspace/')) raw = raw.slice('/workspace/'.length);
    if (!seen.has(raw)) {
      seen.add(raw);
      paths.push(raw);
    }
  }
  return paths;
}

/**
 * Parse a unified-diff patch into per-file segments. Returns
 * ``[{path, hunks: [{header, lines: [{type, text}]}]}, ...]``.
 * ``type`` is 'add' | 'del' | 'ctx' | 'meta'. Best-effort — handles
 * the standard `--- a/x +++ b/x @@ ...` shape. Non-standard hunks
 * (binary diffs, rename-only entries) pass through as a single
 * meta-only segment so the user still sees what changed at file
 * granularity.
 */
function _parsePatch(patch) {
  const segments = [];
  if (!patch) return segments;
  const lines = String(patch).split('\n');
  let current = null;
  let currentHunk = null;
  for (const line of lines) {
    if (line.startsWith('--- ') || line.startsWith('+++ ')) {
      let raw = line.slice(4).trim().split('\t', 1)[0].split(' ', 1)[0];
      if (raw === '/dev/null') continue;
      if (raw.startsWith('a/') || raw.startsWith('b/')) raw = raw.slice(2);
      if (raw.startsWith('/workspace/')) raw = raw.slice('/workspace/'.length);
      if (current && current.path === raw) continue;
      // New file segment. Flush the previous.
      if (current) segments.push(current);
      current = { path: raw, hunks: [] };
      currentHunk = null;
    } else if (line.startsWith('@@')) {
      if (!current) continue;
      currentHunk = { header: line, lines: [] };
      current.hunks.push(currentHunk);
    } else if (currentHunk) {
      if (line.startsWith('+') && !line.startsWith('+++')) {
        currentHunk.lines.push({ type: 'add', text: line.slice(1) });
      } else if (line.startsWith('-') && !line.startsWith('---')) {
        currentHunk.lines.push({ type: 'del', text: line.slice(1) });
      } else if (line.startsWith(' ')) {
        currentHunk.lines.push({ type: 'ctx', text: line.slice(1) });
      } else if (line.startsWith('\\')) {
        // "\ No newline at end of file" — meta marker, skip.
      } else {
        // Trailing junk between hunks — ignore.
      }
    }
  }
  if (current) segments.push(current);
  return segments;
}

// ── Coalescing for fast-arriving same-type tool calls ─────────────────────────
// Pre-2026-05-31 every tool call rendered its own card. Models that fan
// out reads (Qwen-3.6 often does ~5 file_reads in a row) flooded the
// chat with N cards in a few hundred ms — Cline calls this the
// "firehose." We coalesce same-type read-only calls that arrive within
// _COALESCE_WINDOW_MS into a single "Read N files" card with sub-rows.
// Mutating tools (code_edit, file_write, shell_exec, etc.) never
// coalesce: each is significant on its own and deserves its own card.

const _COALESCE_TOOLS = new Set([
  'file_read', 'file_list', 'code_grep', 'find_files',
  'code_search', 'dir_tree',
]);
const _COALESCE_WINDOW_MS = 1500;
// Cap so an absurd fan-out (50 reads) still renders compactly. Beyond
// the cap, the card switches to "Read 50 files (+12 more)" style and
// stops adding sub-rows; the input list lives in the message log if
// the user really wants to scroll history.
const _COALESCE_MAX_VISIBLE = 24;

const _COALESCE_VERBS = {
  file_read: { running: 'Reading', done: 'Read', noun: 'files' },
  file_list: { running: 'Listing', done: 'Listed', noun: 'dirs' },
  code_grep: { running: 'Grepping', done: 'Grepped', noun: 'searches' },
  find_files: { running: 'Globbing', done: 'Globbed', noun: 'queries' },
  code_search: { running: 'Searching', done: 'Searched', noun: 'queries' },
  dir_tree: { running: 'Walking', done: 'Walked', noun: 'trees' },
};

function _coalesceLabel(tool, count, status) {
  const cfg = _COALESCE_VERBS[tool] || { running: tool, done: tool, noun: 'calls' };
  const verb = status === 'running' ? cfg.running : cfg.done;
  return `${verb} ${count} ${cfg.noun}`;
}

function _coalescePath(tool, input) {
  if (tool === 'code_grep' || tool === 'code_search') {
    return `"${input.pattern || input.query || ''}"`;
  }
  if (tool === 'find_files') {
    return input.pattern || input.glob || '';
  }
  if (tool === 'dir_tree') {
    return (input.path || '/').replace(/^\/workspace\//, '');
  }
  return (input.path || '').replace(/^\/workspace\//, '');
}

/**
 * Format elapsed milliseconds for the tool-card meta row. Sub-100ms
 * calls render as "—" (not interesting; would dominate the row); 100ms-
 * 1s as "850ms"; 1s+ as "4.3s" with one decimal.
 */
function _formatToolElapsed(ms) {
  if (!ms || ms < 100) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const min = Math.floor(ms / 60_000);
  const sec = Math.round((ms % 60_000) / 1000);
  return `${min}m ${sec}s`;
}

/**
 * Count line additions/deletions for a single code_edit and return
 * the HTML chip for the summary metric slot. Counts REAL added/removed
 * lines from the line diff — NOT the net line-count delta, which reported
 * +0/-0 for an equal-size replacement (e.g. 4 lines swapped for 4 others
 * is +4/-4, but ``replaceLines - searchLines`` is 0). Mirrors the
 * code_edit_batch counting path so single + batch edits agree.
 */
function _editMetric(search, replace) {
  const diff = computeLineDiff(search || '', replace || '');
  let adds = 0, dels = 0;
  for (const d of diff) {
    if (d.type === 'added') adds++;
    else if (d.type === 'removed') dels++;
  }
  // Always render both so the chrome is consistent across cards.
  return `<span class="coder-tool-metric-add">+${adds}</span><span class="coder-tool-metric-del">-${dels}</span>`;
}

/**
 * Truncate a text block to the first ``maxLines`` lines for an
 * expandable preview. Returns ``{text, truncated}`` — caller renders
 * a "more" footer when truncated. Newline at the end is preserved
 * when present so the pre block doesn't render with a stray cursor
 * spot.
 */
function _previewLines(text, maxLines) {
  if (!text) return { text: '', truncated: false };
  const lines = String(text).split('\n');
  if (lines.length <= maxLines) {
    return { text, truncated: false };
  }
  return { text: lines.slice(0, maxLines).join('\n'), truncated: true };
}

/**
 * Render a unified-diff segment list as a stack of per-file diffs.
 * Each file is its own `<details>` so a multi-file patch doesn't
 * dump everything at once on a tap.
 */
function _renderPatchSegments(segments) {
  if (!segments.length) {
    return '<div class="coder-tool-empty">(empty patch)</div>';
  }
  const parts = [];
  for (const seg of segments) {
    let adds = 0, dels = 0;
    for (const h of seg.hunks) {
      for (const l of h.lines) {
        if (l.type === 'add') adds++;
        else if (l.type === 'del') dels++;
      }
    }
    const summary = `<summary class="coder-tool-subfile">
      <span class="coder-tool-subfile-path">${escapeHtml(seg.path)}</span>
      <span class="coder-tool-subfile-metric">
        <span class="coder-tool-metric-add">+${adds}</span>
        <span class="coder-tool-metric-del">-${dels}</span>
      </span>
    </summary>`;
    const body = seg.hunks.map((h) => {
      const hunkLines = h.lines.map((l) => {
        const cls = l.type === 'add' ? 'diff-line added'
          : l.type === 'del' ? 'diff-line removed'
          : 'diff-line';
        const prefix = l.type === 'add' ? '+' : l.type === 'del' ? '-' : ' ';
        return `<div class="${cls}"><span class="diff-prefix">${prefix}</span>${escapeHtml(l.text)}</div>`;
      }).join('');
      return `<div class="coder-tool-hunk">
        <div class="coder-tool-hunk-header">${escapeHtml(h.header)}</div>
        ${hunkLines}
      </div>`;
    }).join('');
    parts.push(`<details class="coder-tool-subcard" open>${summary}<div class="coder-tool-subcard-body">${body}</div></details>`);
  }
  return parts.join('');
}


/**
 * Render the in-progress body of a task_dispatch card. Shown while the
 * subagent is running; surfaces role / model spec / prompt preview /
 * spinner + a "view live transcript" link to the run-detail JSON
 * endpoint. Replaced with `_renderSubagentResultBody` when the tool
 * completes.
 *
 * Kept self-contained in this module so coder-conversation doesn't
 * import from the chat surface (chat/tool-result-view.js handles the
 * non-coder case identically).
 */
function _renderSubagentRunningBody(input) {
  const role = String(input.role || '').trim() || '(unknown)';
  const modelSpec = String(input.model || '').trim();
  const modelLabel = modelSpec || '(parent\'s model)';
  const prompt = String(input.prompt || '').trim();
  const promptPreview = prompt.length > 240
    ? prompt.slice(0, 237) + '…' : prompt;
  const contextMode = String(input.context || 'workspace');
  // The .coder-subagent-activity-log is populated incrementally by
  // ``updateSubagentProgress``; .coder-subagent-cancel is wired on
  // first progress event when we learn the subagent_id (we don't
  // know it at card-create time — the dispatcher mints it). Both
  // start empty/hidden; the progress handler shows the cancel button
  // + reveals the activity panel on first event.
  return `
    <div class="coder-subagent-card coder-subagent-card--running">
      <div class="coder-subagent-card-head">
        <div class="coder-subagent-card-id">
          <span class="coder-subagent-role-tag">${escapeHtml(role)}</span>
          <span class="coder-subagent-model"><code>${escapeHtml(modelLabel)}</code></span>
        </div>
        <div class="coder-subagent-running-indicator">
          <span class="coder-subagent-spinner" aria-hidden="true"></span>
          <span class="coder-subagent-running-label">running…</span>
          <button type="button" class="coder-subagent-cancel" hidden
                  title="Cancel this subagent (siblings keep running)">
            cancel
          </button>
        </div>
      </div>
      ${promptPreview ? `
        <div class="coder-subagent-prompt">
          <div class="coder-subagent-prompt-label">PROMPT</div>
          <div class="coder-subagent-prompt-body">${escapeHtml(promptPreview)}</div>
        </div>
      ` : ''}
      <div class="coder-subagent-activity coder-subagent-activity--empty">
        <div class="coder-subagent-activity-head">
          <span class="coder-subagent-activity-label">ACTIVITY</span>
          <span class="coder-subagent-activity-stats"></span>
        </div>
        <ol class="coder-subagent-activity-log" aria-live="polite"></ol>
      </div>
      <div class="coder-subagent-card-foot">
        <span class="coder-subagent-context">context: <code>${escapeHtml(contextMode)}</code></span>
      </div>
    </div>
  `;
}


/**
 * Synthesize a "result was lost" object for tool calls that come back from
 * the persisted conversation without a recorded result. Without this,
 * ``loadHistory`` would leave the card in perpetual running state — and
 * the running-banner would pull the user to the top of the conversation
 * every load until the message was edited away.
 *
 * Happens whenever a tool was in flight at the moment the session ended:
 * page closed mid-call, container restarted, server crashed, network died.
 * The server never gets a chance to attach the result before persisting
 * the message, so the on-disk shape is ``{role: 'tool', ..., result: null}``.
 *
 * Shape mirrors the result objects that the runtime emits on completion
 * (see updateToolResult + _renderSubagentResultBody) so the existing
 * renderers handle it without special cases. ``stop_reason='interrupted'``
 * surfaces as a tinted pill instead of the green "complete".
 */
function _synthesizeInterruptedResult(tool, input) {
  const detail = 'Run interrupted — page reload / server restart before result landed.';
  if (tool === 'task_dispatch') {
    return {
      success: false,
      output: '',
      metadata: {
        role: input?.role || '',
        model_spec: input?.model || '',
        stop_reason: 'interrupted',
        stop_detail: detail,
      },
    };
  }
  return { success: false, error: detail };
}

/** Truncate a string with an ellipsis for activity-log previews. */
function _truncForActivity(s, n) {
  if (!s) return '';
  const str = String(s);
  return str.length > n ? str.slice(0, n - 1) + '…' : str;
}


function _formatSubagentTokens(n) {
  const v = Number(n) || 0;
  if (v < 1000) return String(v);
  if (v < 1_000_000) return `${(v / 1000).toFixed(1)}k`;
  return `${(v / 1_000_000).toFixed(2)}M`;
}

function _formatSubagentDuration(ms) {
  const v = Number(ms) || 0;
  if (v < 1000) return `${v}ms`;
  if (v < 60_000) return `${(v / 1000).toFixed(1)}s`;
  const min = Math.floor(v / 60_000);
  const sec = Math.round((v % 60_000) / 1000);
  return `${min}m ${sec}s`;
}


/**
 * Render the completed-state body of a task_dispatch card. Replaces
 * the running body when the tool result lands. Surfaces: role + final
 * resolved model + iter / tool / token / wall stats + stop-reason
 * pill + optional inspect link to the run-detail JSON.
 *
 * Reads from result.metadata for the structured fields; falls back to
 * result.output for the prose summary.
 */
function _renderSubagentResultBody(input, result) {
  const meta = result?.metadata || {};
  const role = String(meta.role || input.role || '').trim() || '(unknown)';
  const modelResolved = String(meta.model_resolved || '');
  const modelSpec = String(meta.model_spec || input.model || '');
  const modelLabel = modelSpec && modelSpec !== modelResolved
    ? `${modelResolved} (${modelSpec})`
    : (modelResolved || '(parent\'s model)');
  const iters = Number(meta.iterations) || 0;
  const tools = Number(meta.tool_calls) || 0;
  const tokensIn = Number(meta.tokens_in) || 0;
  const tokensOut = Number(meta.tokens_out) || 0;
  const wallMs = Number(meta.wallclock_ms) || 0;
  const stopReason = String(meta.stop_reason || '');
  const stopDetail = String(meta.stop_detail || '');
  const stuckPattern = String(meta.stuck_pattern || '');
  const subagentId = String(meta.subagent_id || '');
  const stopClass = `coder-subagent-stop-${stopReason || 'unknown'}`;

  const output = String(result?.output || '').trim();
  const outputPreview = output.length > 400 ? output.slice(0, 397) + '…' : output;
  // The recovery hint is the structured guidance the inner loop
  // attaches when stop_reason != complete (see _compute_recovery_hint
  // in augmentum/agents/loop.py). Showing it as a tinted callout in
  // the result card means the user immediately sees WHY the subagent
  // stopped and WHAT the lead should do next — no scrolling, no
  // inferring from stop_reason + stop_detail.
  const recoveryHint = String(meta.recovery_hint || '').trim();

  const inspectLink = subagentId
    ? `<a class="coder-subagent-inspect" href="/api/coder/subagents/${encodeURIComponent(subagentId)}" target="_blank" rel="noopener">view full transcript →</a>`
    : '';

  return `
    <div class="coder-subagent-card coder-subagent-card--done">
      <div class="coder-subagent-card-head">
        <div class="coder-subagent-card-id">
          <span class="coder-subagent-role-tag">${escapeHtml(role)}</span>
          <span class="coder-subagent-model"><code>${escapeHtml(modelLabel)}</code></span>
        </div>
        <span class="coder-subagent-stop-pill ${stopClass}">${escapeHtml(stopReason || 'unknown')}</span>
      </div>
      <dl class="coder-subagent-stats">
        <div><dt>iters</dt><dd>${iters}</dd></div>
        <div><dt>tools</dt><dd>${tools}</dd></div>
        <div><dt>tokens</dt><dd>${_formatSubagentTokens(tokensIn + tokensOut)} <span class="coder-subagent-stat-sub">(${_formatSubagentTokens(tokensIn)} in / ${_formatSubagentTokens(tokensOut)} out)</span></dd></div>
        <div><dt>wall</dt><dd>${_formatSubagentDuration(wallMs)}</dd></div>
      </dl>
      ${stopDetail ? `<div class="coder-subagent-detail">${escapeHtml(stopDetail)}</div>` : ''}
      ${stuckPattern ? `<div class="coder-subagent-detail">stuck pattern: ${escapeHtml(stuckPattern)}</div>` : ''}
      ${recoveryHint ? `
        <div class="coder-subagent-recovery">
          <span class="coder-subagent-recovery-label">→ next move</span>
          <div class="coder-subagent-recovery-body">${escapeHtml(recoveryHint)}</div>
        </div>
      ` : ''}
      ${outputPreview ? `
        <details class="coder-subagent-output">
          <summary>OUTPUT</summary>
          <pre>${escapeHtml(outputPreview)}</pre>
        </details>
      ` : ''}
      ${inspectLink ? `<div class="coder-subagent-card-foot">${inspectLink}</div>` : ''}
    </div>
  `;
}

// ─────────────────────────────────────────────────────────────────────────────

// ---------------------------------------------------------------------------
// Attachment rendering (shared by addUserMessage + loadHistory)
// ---------------------------------------------------------------------------

/**
 * Serialise an attachment descriptor for conversation history. Strips
 * client-only fields (``dataUrl``, ``pending``) since they aren't
 * meaningful after a refresh — ``dataUrl`` for images is the
 * ephemeral data URL we used for the optimistic thumbnail; once
 * saved, the ``url`` field (``/api/chat-images/<id>``) rehydrates
 * the image on load.
 */
function _serializeAttachment(att) {
  return {
    id:   att.id,
    kind: att.kind,
    name: att.name,
    size: att.size,
    mime: att.mime,
    url:  att.url,
  };
}


/**
 * Render an attachment tile inside a posted user message.
 *
 * Images → thumbnail anchor that opens the full image in a new tab
 * (no lightbox yet — that's polish). Clicks respect middle-click /
 * ctrl-click for tab behaviour since they're real <a> elements.
 *
 * Files → compact chip with type-icon + name + size. Not clickable
 * by default (the file lives under /workspace/.augmentum/attachments/
 * and is accessible via the file tree); a future enhancement could
 * wire click → jump-to-file-in-tree.
 */
function _renderMessageAttachment(att) {
  if (att.kind === 'image') {
    const a = document.createElement('a');
    a.className = 'coder-msg-attachment coder-msg-attachment--image';
    a.href = att.url;
    a.target = '_blank';
    a.rel = 'noopener';
    a.title = `${att.name} — click to open`;
    const img = document.createElement('img');
    img.src = att.url;
    img.alt = att.name;
    img.loading = 'lazy';
    a.appendChild(img);
    return a;
  }
  // Non-image: chip with icon + meta
  const chip = document.createElement('div');
  chip.className = 'coder-msg-attachment coder-msg-attachment--file';
  chip.title = att.url;
  chip.innerHTML = `
    <span class="coder-msg-attachment-icon" aria-hidden="true">${_fileIconForMime(att.mime, att.name)}</span>
    <span class="coder-msg-attachment-meta">
      <span class="coder-msg-attachment-name">${escapeHtml(att.name)}</span>
      <span class="coder-msg-attachment-path">${escapeHtml(att.url)}</span>
    </span>
  `;
  return chip;
}


function _fileIconForMime(mime, name) {
  const lower = (mime || '').toLowerCase();
  const ext = (name || '').toLowerCase().split('.').pop() || '';
  if (lower.startsWith('text/') || ['txt', 'md', 'log', 'csv'].includes(ext)) return '≡';
  if (lower === 'application/pdf' || ext === 'pdf') return 'PDF';
  if (lower === 'application/json' || ext === 'json') return '{}';
  if (['zip', 'tar', 'gz', '7z', 'rar'].includes(ext) || lower.includes('zip')) return '□';
  if (['js', 'ts', 'py', 'rs', 'go', 'c', 'cpp', 'java'].includes(ext)) return '</>';
  return '▢';
}


export class CoderConversation {
  /**
   * @param {HTMLElement} container  — #coder-conv-messages element
   * @param {HTMLElement} scrollEl   — #coder-conv-scroll element
   */
  constructor(container, scrollEl) {
    this._container = container;
    this._scrollEl = scrollEl;
    this._messages = [];          // Serializable history [{id, role, content, metadata}]
    this._toolCards = new Map();   // id → DOM element (for updating results)
    // subagent_id (minted by the dispatcher) → the toolCardId of the
    // task_dispatch card hosting that subagent. Populated on the first
    // ``subagent_progress`` event for an unseen instance_id by binding
    // to the most recent running task_dispatch card. Cleared in
    // updateToolResult when the dispatch completes.
    this._subagentCardByInstance = new Map();
    this._streamEl = null;        // Current streaming response element
    this._lastContentDelta = '';
    this._userScrolledUp = false;
    this._idCounter = 0;
    // History-replay bookkeeping (see loadHistory). ``_historyMode`` makes
    // the render helpers display-only while a persisted message replays;
    // ``_historyGen`` cancels an in-flight async backfill when the
    // conversation is cleared or reloaded underneath it.
    this._historyMode = false;
    this._historyGen = 0;
    // Coalescing state (Phase 2): when same-type read-only tool calls
    // arrive back-to-back within _COALESCE_WINDOW_MS and no prose lands
    // between them, fold them into one card with sub-rows. ``cardEl``
    // is the live card; ``tool`` and ``lastAdded`` are the dedup keys.
    // Reset when a different tool fires OR when the streaming bubble
    // gets flushed (prose between tool calls breaks the run).
    this._coalesce = {
      cardEl: null,
      tool: '',
      lastAdded: 0,
      subRows: new Map(),  // tool_id -> {li, statusEl, input}
    };
    // Phase 3 — sticky running banner. Pinned to the top of the chat
    // surface so when the user scrolls away from a long-running tool
    // they still see "⏳ Read 3 files · 1.4s" with a tap-to-jump back.
    // Visibility is reactive: shows when at least one running card is
    // present AND that card is below the visible viewport. Updates the
    // elapsed counter on a 200ms tick while visible.
    this._runningBanner = {
      el: document.getElementById('coder-running-banner'),
      labelEl: document.getElementById('coder-running-banner-detail'),
      elapsedEl: document.getElementById('coder-running-banner-elapsed'),
      tickHandle: null,
    };
    if (this._runningBanner.el) {
      this._runningBanner.el.addEventListener('click', () => this._jumpToRunningCard());
    }
    // Active workspace id for tool results that need a workspace-scoped
    // URL (e.g. browser_screenshot's inline ``<img>`` embed). Set by
    // ``setWorkspaceId`` when the surrounding coder UI switches workspaces.
    this._workspaceId = '';

    // DOM windowing state. ``_detachedTop`` holds the oldest message/tool
    // nodes that have been pulled OUT of the document to keep the live DOM
    // bounded — retained in visual order (oldest first) so re-attaching them
    // on scroll-up is exact. ``_windowSentinel`` is the "show earlier" control
    // pinned to the top of the container while anything is detached.
    this._detachedTop = [];
    this._windowSentinel = null;

    // Single passive scroll handler. Reading scroll position is cheap; the
    // expensive part — _refreshRunningBanner's querySelectorAll over every
    // tool card + getBoundingClientRect (a forced reflow) — is rAF-coalesced
    // inside _refreshRunningBanner itself. Previously TWO scroll listeners
    // fired per event AND the banner refresh ran synchronously, so a single
    // smooth-scroll's storm of scroll events thrashed layout — a big part of
    // the tool-call-flurry freeze.
    this._scrollEl?.addEventListener('scroll', () => {
      const el = this._scrollEl;
      if (el) {
        this._userScrolledUp = el.scrollTop + el.clientHeight < el.scrollHeight - 40;
        // Infinite-scroll-up: when the user reaches the top and there are
        // windowed-out nodes, re-attach the next batch. Scroll position is
        // compensated inside _rehydrateOlder so the viewport never jumps.
        if (el.scrollTop <= _REHYDRATE_SCROLL_PX && this._detachedTop.length) {
          this._rehydrateOlder();
        }
      }
      this._refreshRunningBanner();
    }, { passive: true });
  }

  // ── Public API ────────────────────────────────────────────────────────

  /**
   * Add a user message bubble.
   *
   * @param {string}   text         — the user's prompt (clean, no footer)
   * @param {object[]} [attachments] — optional attachment descriptors to
   *                                   render inline as chips below the
   *                                   text. Each entry is the descriptor
   *                                   shape returned by
   *                                   coder-attachments.js#ingestFile.
   *                                   Stored in the conversation history
   *                                   so attachments survive refresh /
   *                                   conversation reload.
   */
  addUserMessage(text, attachments = []) {
    const id = this._nextId();
    const el = document.createElement('div');
    el.className = 'coder-msg coder-msg-user';
    // Stamp the conversation-local message id on the element so callers
    // can find this exact bubble later by id (e.g.,
    // _markConversationMessageAsQueued after an interject fetch returns).
    // Without this, the old code used `container.lastElementChild` which
    // is racy: any chunk that lands during the fetch await leaves the
    // wrong bubble at the bottom, and the "failed" tag ends up on an
    // unrelated message (or worse, the chained turn's auto-rendered
    // duplicate bubble).
    el.dataset.convMsgId = String(id);

    // Text block — textContent (not innerHTML) so arbitrary user
    // input can't inject markup. Whitespace preserved via CSS
    // white-space:pre-wrap on .coder-msg-user.
    const textEl = document.createElement('div');
    textEl.className = 'coder-msg-user-text';
    textEl.textContent = text;
    el.appendChild(textEl);

    // Attachment grid — only when at least one attachment carries a
    // usable URL. Images render as thumbnails with click-to-expand;
    // files render as compact chips with type-icon + name. The
    // rendering uses the same CSS as composer chips but with
    // .is-sent for a slightly muted look (already-sent, can't be
    // removed).
    const ready = (attachments || []).filter(a => a && a.url);
    if (ready.length) {
      const grid = document.createElement('div');
      grid.className = 'coder-msg-attachments';
      for (const att of ready) {
        grid.appendChild(_renderMessageAttachment(att));
      }
      el.appendChild(grid);
    }

    this._append(el);
    this._messages.push({
      id, role: 'user', content: text,
      attachments: ready.length ? ready.map(_serializeAttachment) : undefined,
    });
    // Remove empty state if present
    document.getElementById('coder-conv-empty')?.remove();
    return id;
  }

  /** Add a thinking/planning block (streaming). */
  addThinking() {
    const el = document.createElement('div');
    el.className = 'coder-msg coder-msg-thinking';
    this._append(el);
    return el;
  }

  /** Append text to a thinking block. Append a text node rather than
   *  ``textContent += text`` — the latter re-materializes the whole
   *  accumulated string each delta (O(n²) over a long reasoning stream).
   *  The scroll is already rAF-coalesced in _scrollToBottom. */
  appendThinking(el, text) {
    el.appendChild(document.createTextNode(text));
    this._scrollToBottom();
  }

  /**
   * Add a live reasoning block (model chain-of-thought, ``reasoning_delta``
   * chunks). Distinct from the plan-text thinking bubble above: reasoning
   * is a byproduct the user peeks at, not a deliverable, so it renders as
   * a collapsible card — expanded and height-clamped while streaming
   * (tokens scroll inside the card, the page itself never reflows or
   * scrolls per delta), then auto-collapsed to a one-line "Reasoned for
   * Xs" header when the model moves on. Ephemeral by design, like the
   * thinking bubble — not serialized into _messages.
   *
   * @returns {object} block handle for appendReasoning/finalizeReasoning
   */
  addReasoning() {
    const wrap = document.createElement('div');
    wrap.className = 'coder-msg coder-reasoning';
    wrap.dataset.state = 'live';

    const head = document.createElement('button');
    head.type = 'button';
    head.className = 'coder-reasoning-head';
    head.setAttribute('aria-expanded', 'true');
    // Static markup only — no user/model text flows through innerHTML.
    head.innerHTML =
      '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" '
      + 'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
      + 'aria-hidden="true"><path d="M6 4l4 4-4 4"/></svg>'
      + '<span class="coder-reasoning-label">Reasoning…</span>';

    const body = document.createElement('div');
    body.className = 'coder-reasoning-body';
    const textEl = document.createElement('div');
    textEl.className = 'coder-reasoning-text';
    body.appendChild(textEl);

    wrap.appendChild(head);
    wrap.appendChild(body);

    const block = {
      el: wrap,
      body,
      textEl,
      labelEl: head.querySelector('.coder-reasoning-label'),
      startedAt: performance.now(),
      chars: 0,
      done: false,
      _stickBottom: true,
      _scrollPending: false,
    };

    head.addEventListener('click', () => {
      const expanded = head.getAttribute('aria-expanded') === 'true';
      head.setAttribute('aria-expanded', String(!expanded));
      wrap.classList.toggle('is-collapsed', expanded);
      if (!expanded) {
        // Re-opened — jump the inner scroll to the tail (where the
        // freshest reasoning is) and resume sticking.
        block._stickBottom = true;
        body.scrollTop = body.scrollHeight;
      }
    });
    // Inner-scroll stickiness: reading older reasoning mid-stream must
    // not fight the autoscroll. Passive — position reads only.
    body.addEventListener('scroll', () => {
      block._stickBottom =
        body.scrollTop + body.clientHeight >= body.scrollHeight - 12;
    }, { passive: true });

    this._append(wrap);
    return block;
  }

  /** Append reasoning text. O(1) per delta: one text node append + an
   *  rAF-coalesced INNER scroll. The card's outer height is clamped by
   *  CSS while live, so the page layout is untouched per delta — this
   *  is what keeps long reasoning streams off the main-thread budget. */
  appendReasoning(block, text) {
    if (!block || block.done || !text) return;
    block.textEl.appendChild(document.createTextNode(text));
    block.chars += text.length;
    if (block._stickBottom && !block._scrollPending) {
      block._scrollPending = true;
      requestAnimationFrame(() => {
        block._scrollPending = false;
        if (block._stickBottom) {
          block.body.scrollTop = block.body.scrollHeight;
        }
      });
    }
  }

  /**
   * Close a reasoning block: collapse it to a one-line summary header
   * ("Reasoned for 12s", expandable). Called when a tool call or prose
   * interrupts the reasoning stream, on turn completion, or with
   * ``interrupted: true`` when a provider retry cut the stream.
   * A block that never received text is removed outright.
   */
  finalizeReasoning(block, { interrupted = false } = {}) {
    if (!block || block.done) return;
    block.done = true;
    if (!block.chars) {
      block.el.remove();
      return;
    }
    const secs = (performance.now() - block.startedAt) / 1000;
    const dur = secs >= 10 ? `${Math.round(secs)}s`
              : secs >= 0.95 ? `${secs.toFixed(1)}s`
              : '';
    block.el.dataset.state = 'done';
    block.el.classList.add('is-collapsed');
    block.el.querySelector('.coder-reasoning-head')
      ?.setAttribute('aria-expanded', 'false');
    block.labelEl.textContent = interrupted
      ? 'Reasoning interrupted'
      : (dur ? `Reasoned for ${dur}` : 'Reasoned');
  }

  /** Add a step indicator (decompose strategy). */
  addStep(step, total, description) {
    const el = document.createElement('div');
    el.className = 'coder-step-indicator';
    el.textContent = `Step ${step}/${total}: ${description}`;
    this._append(el);
  }

  /**
   * Add a tool activity card (collapsed by default).
   * @returns {string} Tool call ID for later updateToolResult().
   */
  addToolCall(id, tool, input) {
    // Close any streaming assistant bubble first so the tool card
    // appears BELOW the prose the model emitted immediately before
    // it, not as a sibling of all prose in this turn. Without this
    // every tool_call during a turn stacks at the bottom while all
    // prose accumulates at the top — a "tool dump" layout that
    // loses the model's natural "I did X, then Y, then Z" narrative.
    // ``_flushStreamingBubble`` also breaks coalescing — by the time
    // we get here _streamEl is null, but we tracked whether prose
    // landed via the existence of an element that just got closed.
    const proseSinceLastTool = this._streamEl !== null;
    this._flushStreamingBubble();

    // Phase 2 — coalesce check. Fold this call into the previous card
    // when: same tool name, within window, no prose interrupted the
    // run, and the tool is in the read-only coalesce set. Mutating
    // calls always get their own card.
    const now = performance.now();
    if (
      _COALESCE_TOOLS.has(tool)
      && this._coalesce.tool === tool
      && this._coalesce.cardEl
      && !proseSinceLastTool
      && (now - this._coalesce.lastAdded) < _COALESCE_WINDOW_MS
    ) {
      this._appendCoalesceSubRow(id, tool, input);
      this._coalesce.lastAdded = now;
      // Register in _toolCards so updateToolResult finds the right
      // sub-row by id. ``el`` points at the group card; the sub-row
      // is looked up via _coalesce.subRows.
      this._toolCards.set(id, {
        el: this._coalesce.cardEl,
        tool, input,
        isCoalesced: true,
      });
      // History replay is display-only — the message is already in
      // _messages (loadHistory sets the full array up-front).
      if (!this._historyMode) {
        this._messages.push({ id, role: 'tool', tool, input, result: null });
      }
      return id;
    }

    const { icon, verb, desc } = _toolSummary(tool, input);
    const startedAt = performance.now();

    const details = document.createElement('details');
    details.className = 'coder-tool-card';
    details.dataset.toolId = id;
    details.dataset.tool = tool;
    details.dataset.startedAt = String(startedAt);

    const summary = document.createElement('summary');
    summary.innerHTML = `
      <span class="coder-tool-icon">${icon}</span>
      <span class="coder-tool-name">${escapeHtml(verb)}</span>
      <span class="coder-tool-desc">${escapeHtml(desc)}</span>
      <span class="coder-tool-metric"></span>
      <span class="coder-tool-elapsed"></span>
      <span class="coder-tool-status running">running</span>
    `;
    details.appendChild(summary);

    // Body will be populated when result arrives
    const body = document.createElement('div');
    body.className = 'coder-tool-body';

    // task_dispatch gets a rich in-progress card so the user sees what
    // the subagent is doing while it runs (role, model, prompt, spin)
    // instead of an empty body until the result lands. Replaced with
    // the final result card in updateToolResult. Inspired by Claude
    // Code's Task tool transparency.
    if (tool === 'task_dispatch') {
      body.innerHTML = _renderSubagentRunningBody(input);
      // Auto-expand running subagent so the user sees what's happening
      // without an extra click.
      details.open = true;
    }

    details.appendChild(body);

    this._append(details);
    this._toolCards.set(id, { el: details, tool, input });

    // Phase 2 — register this card as the head of a potential coalesce
    // run. The next same-type call within the window will fold into
    // here. Non-coalescable tools clear the state so a code_edit
    // doesn't get an unrelated file_read appended to it.
    if (_COALESCE_TOOLS.has(tool)) {
      this._coalesce.cardEl = details;
      this._coalesce.tool = tool;
      this._coalesce.lastAdded = performance.now();
      this._coalesce.subRows = new Map();
      // Stash the "first call" path so _convertToCoalesce can move it
      // into the sub-rows list when a second same-type call arrives.
      details.dataset.firstPath = _coalescePath(tool, input);
      details.dataset.firstToolId = id;
    } else {
      this._coalesce.cardEl = null;
      this._coalesce.tool = '';
    }

    // History replay is display-only — see the coalesced branch above.
    if (!this._historyMode) {
      this._messages.push({ id, role: 'tool', tool, input, result: null });
    }
    // Refresh the banner — a new running card may now be eligible to show.
    this._refreshRunningBanner();
    return id;
  }

  /**
   * Append a sub-row to the active coalesce card. On the first
   * additional call (the 2nd same-type within the window) we convert
   * the card's body into a sub-rows list, moving the first call's
   * path/preview into a sub-row.
   */
  _appendCoalesceSubRow(id, tool, input) {
    const card = this._coalesce.cardEl;
    if (!card) return;
    const isFirstAppend = !card.dataset.coalesce;
    if (isFirstAppend) {
      this._convertCardToCoalesce(card, tool);
    }
    const list = card.querySelector('.coder-tool-coalesce-list');
    if (!list) return;

    const visible = this._coalesce.subRows.size;
    const overflow = visible >= _COALESCE_MAX_VISIBLE;

    if (!overflow) {
      const li = document.createElement('li');
      li.className = 'coder-tool-coalesce-row';
      li.dataset.toolId = id;
      const statusEl = document.createElement('span');
      statusEl.className = 'coder-tool-coalesce-status running';
      const pathEl = document.createElement('span');
      pathEl.className = 'coder-tool-coalesce-path';
      pathEl.textContent = _coalescePath(tool, input);
      li.appendChild(statusEl);
      li.appendChild(pathEl);
      list.appendChild(li);
      this._coalesce.subRows.set(id, { li, statusEl, input });
    } else {
      // Overflow marker — render once on the (cap+1)th item, update
      // the count on subsequent items.
      let overflowEl = list.querySelector('.coder-tool-coalesce-overflow');
      const extra = (visible - _COALESCE_MAX_VISIBLE) + 1;
      if (!overflowEl) {
        overflowEl = document.createElement('li');
        overflowEl.className = 'coder-tool-coalesce-overflow';
        list.appendChild(overflowEl);
      }
      overflowEl.textContent = `+${extra} more (open inspector to view all)`;
      this._coalesce.subRows.set(id, { li: null, statusEl: null, input });
    }

    // Update card title to reflect the new count.
    const total = this._coalesce.subRows.size + 1;  // +1 = the first call (pre-conversion)
    const nameEl = card.querySelector('.coder-tool-name');
    const descEl = card.querySelector('.coder-tool-desc');
    if (nameEl) nameEl.textContent = _coalesceLabel(tool, total, 'running');
    if (descEl) descEl.textContent = '';
    card.dataset.coalesceCount = String(total);
    this._scrollToBottom();
  }

  /**
   * One-time mutation that turns a single-call card into the coalesce
   * shell. Hoists the first call's body content out (we lose the
   * per-file preview deliberately — at fan-out scale users want the
   * file list, not 12 lines of file #1) and replaces it with a
   * sub-rows list seeded with the first call's row.
   */
  // ── Phase 3: sticky running banner ─────────────────────────────

  /**
   * Find the latest running tool card. Returns the DOM element or
   * ``null`` when nothing is in flight.
   */
  _findLatestRunningCard() {
    // Iterate the message container's children in reverse — newest
    // running card wins. Could maintain a separate index but the
    // tool-card count per turn is small (<100) so a quick scan is fine.
    const messages = document.getElementById('coder-conv-messages');
    if (!messages) return null;
    const cards = messages.querySelectorAll('.coder-tool-card .coder-tool-status.running');
    if (!cards.length) return null;
    // Last running status indicator → its enclosing card.
    return cards[cards.length - 1].closest('.coder-tool-card');
  }

  /**
   * rAF-coalesced entry point. Many code paths poke the banner per
   * mutation (addToolCall, updateToolResult, every coalesce sub-row, and
   * every scroll event). The actual refresh forces a layout via
   * getBoundingClientRect, so during a dozens-of-tool-calls flurry the
   * synchronous version reflowed the page dozens of times per frame.
   * Collapse every poke to one refresh per animation frame.
   */
  _refreshRunningBanner() {
    if (this._bannerRefreshScheduled) return;
    this._bannerRefreshScheduled = true;
    const run = () => {
      this._bannerRefreshScheduled = false;
      this._refreshRunningBannerNow();
    };
    if (typeof window !== 'undefined'
        && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(run);
    } else {
      run();
    }
  }

  _refreshRunningBannerNow() {
    const banner = this._runningBanner.el;
    if (!banner) return;
    // Fast path — the banner exists ONLY to surface a running card the user
    // has scrolled away from. When we're at/near the bottom (the streaming
    // auto-scroll common case, where the scroll handler pokes this every
    // animation frame), nothing is out of view to surface. Bail before the
    // full-conversation querySelectorAll + getBoundingClientRect (a forced
    // reflow) so per-frame streaming cost stays flat. Behavior-preserving:
    // when not scrolled up, the checks below would hide the banner anyway.
    if (!this._userScrolledUp) {
      this._hideRunningBanner();
      return;
    }
    const card = this._findLatestRunningCard();
    if (!card) {
      this._hideRunningBanner();
      return;
    }
    // Show only when the running card is above the visible top of the
    // scroll area (i.e. the user scrolled past it OR was already
    // scrolled up when it started running). We give a small buffer
    // (top + 40px) so the banner doesn't strobe when the card is just
    // peeking into view from below.
    const scrollEl = this._scrollEl;
    if (!scrollEl) return;
    const cardRect = card.getBoundingClientRect();
    const scrollRect = scrollEl.getBoundingClientRect();
    const aboveViewport = cardRect.bottom < (scrollRect.top + 40);
    const belowViewport = cardRect.top > (scrollRect.bottom - 40);
    if (aboveViewport || belowViewport) {
      this._showRunningBanner(card);
    } else {
      this._hideRunningBanner();
    }
  }

  _showRunningBanner(card) {
    const banner = this._runningBanner.el;
    if (!banner) return;
    // Compose the label from the card's chrome. Don't render an SVG
    // icon in the banner — keep the chrome minimal at small widths.
    const verb = card.querySelector('.coder-tool-name')?.textContent || 'Running';
    const desc = card.querySelector('.coder-tool-desc')?.textContent || '';
    if (this._runningBanner.labelEl) {
      this._runningBanner.labelEl.textContent = desc ? `${verb} · ${desc}` : verb;
    }
    banner.classList.remove('hidden');
    banner.dataset.targetId = card.dataset.toolId || '';
    // Tick elapsed at 200ms — fast enough for sub-second feedback,
    // slow enough to not thrash the GC. Only ticks while shown.
    if (!this._runningBanner.tickHandle) {
      this._runningBanner.tickHandle = setInterval(
        () => this._tickRunningBanner(),
        200,
      );
    }
    this._tickRunningBanner();
  }

  _hideRunningBanner() {
    const banner = this._runningBanner.el;
    if (!banner || banner.classList.contains('hidden')) return;
    banner.classList.add('hidden');
    if (this._runningBanner.tickHandle) {
      clearInterval(this._runningBanner.tickHandle);
      this._runningBanner.tickHandle = null;
    }
  }

  _tickRunningBanner() {
    const banner = this._runningBanner.el;
    if (!banner || banner.classList.contains('hidden')) return;
    const targetId = banner.dataset.targetId;
    if (!targetId) return;
    const card = document.querySelector(
      `.coder-tool-card[data-tool-id="${CSS.escape(targetId)}"]`,
    );
    if (!card) {
      // Running card vanished (e.g., result arrived and status flipped) —
      // recompute. If another card is still running, swap; otherwise hide.
      this._refreshRunningBanner();
      return;
    }
    // Status may have flipped to done/error since the banner was shown.
    const status = card.querySelector('.coder-tool-status');
    if (!status?.classList.contains('running')) {
      this._refreshRunningBanner();
      return;
    }
    const startedAt = Number(card.dataset.startedAt) || 0;
    if (this._runningBanner.elapsedEl && startedAt > 0) {
      const elapsed = _formatToolElapsed(performance.now() - startedAt) || '';
      this._runningBanner.elapsedEl.textContent = elapsed;
    }
  }

  _jumpToRunningCard() {
    const banner = this._runningBanner.el;
    if (!banner) return;
    const targetId = banner.dataset.targetId;
    if (!targetId) return;
    const card = document.querySelector(
      `.coder-tool-card[data-tool-id="${CSS.escape(targetId)}"]`,
    );
    if (!card) return;
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    // Brief highlight pulse so the user spots the card after the
    // scroll lands — pure-CSS animation triggered by class toggle.
    card.classList.add('coder-tool-card-highlight');
    setTimeout(() => card.classList.remove('coder-tool-card-highlight'), 1400);
  }

  /**
   * Update the sub-row status when a coalesced tool call completes.
   * Promotes the group card's overall status to:
   *   - success if every sub-row is success
   *   - error if any sub-row is error
   *   - running otherwise
   * Elapsed time is computed as "since first call added" so it tracks
   * the wall-clock the user actually saw.
   */
  _updateCoalesceSubRow(cardEl, id, result) {
    const sub = this._coalesce.subRows.get(id);
    if (sub && sub.statusEl) {
      sub.statusEl.classList.remove('running');
      sub.statusEl.classList.add(result.success ? 'success' : 'error');
      if (!result.success) {
        const errPreview = (result.error || result.output_preview || '').slice(0, 200);
        if (errPreview) sub.li.title = errPreview;
      }
    }
    // Aggregate state across all sub-rows.
    const groupStatus = cardEl.querySelector('.coder-tool-status');
    if (!groupStatus) return;
    const subStatuses = [...this._coalesce.subRows.values()]
      .map((s) => s.statusEl)
      .filter(Boolean);
    let anyRunning = false, anyError = false;
    for (const s of subStatuses) {
      if (s.classList.contains('running')) anyRunning = true;
      if (s.classList.contains('error')) anyError = true;
    }
    groupStatus.classList.remove('running', 'success', 'error');
    if (anyRunning) {
      groupStatus.classList.add('running');
      groupStatus.textContent = 'running';
    } else if (anyError) {
      groupStatus.classList.add('error');
      groupStatus.textContent = 'failed';
    } else {
      groupStatus.classList.add('success');
      groupStatus.textContent = 'done';
    }
    // Update verb tense in the title (Reading → Read).
    const nameEl = cardEl.querySelector('.coder-tool-name');
    if (nameEl) {
      const tool = cardEl.dataset.tool;
      const count = Number(cardEl.dataset.coalesceCount) || 1;
      nameEl.textContent = _coalesceLabel(
        tool, count, anyRunning ? 'running' : 'done',
      );
    }
    // Group elapsed = since first call added (most informative metric
    // for the user — "this fan-out took 1.4s").
    if (!anyRunning) {
      const elapsedEl = cardEl.querySelector('.coder-tool-elapsed');
      const startedAt = Number(cardEl.dataset.startedAt) || 0;
      if (elapsedEl && startedAt > 0) {
        elapsedEl.textContent = _formatToolElapsed(performance.now() - startedAt);
      }
    }
    // Update message log for the sub-row id. Skipped during history
    // replay — the result came FROM _messages, and the linear find per
    // tool was O(n²) across a big loadHistory.
    if (!this._historyMode) {
      const msg = this._messages.find((m) => m.id === id);
      if (msg) msg.result = result;
    }
    // Coalesced sub-row finished — group card status may have flipped
    // to done/error; refresh banner accordingly.
    this._refreshRunningBanner();
  }

  _convertCardToCoalesce(card, tool) {
    card.dataset.coalesce = '1';
    const body = card.querySelector('.coder-tool-body');
    if (!body) return;
    const firstPath = card.dataset.firstPath || '';
    const firstId = card.dataset.firstToolId || '';
    body.innerHTML = `<ul class="coder-tool-coalesce-list"></ul>`;
    const list = body.querySelector('.coder-tool-coalesce-list');
    const firstLi = document.createElement('li');
    firstLi.className = 'coder-tool-coalesce-row';
    if (firstId) firstLi.dataset.toolId = firstId;
    // First call may already have completed by the time the 2nd arrives;
    // mark it ready if the original status was 'done'/'failed', else
    // running. We re-sync on next updateToolResult anyway.
    const origStatus = card.querySelector('.coder-tool-status');
    const origCls = origStatus?.classList.contains('success') ? 'success'
      : origStatus?.classList.contains('error') ? 'error' : 'running';
    const statusEl = document.createElement('span');
    statusEl.className = `coder-tool-coalesce-status ${origCls}`;
    const pathEl = document.createElement('span');
    pathEl.className = 'coder-tool-coalesce-path';
    pathEl.textContent = firstPath;
    firstLi.appendChild(statusEl);
    firstLi.appendChild(pathEl);
    list.appendChild(firstLi);
    // Register the first row in the subRows map so updateToolResult
    // for the original tool id can update its status alongside the rest.
    // CRITICAL: also flip the first tool's _toolCards entry to
    // ``isCoalesced=true`` so its later result.update goes through
    // the sub-row path, not the per-tool body renderer (which would
    // clobber the coalesce list we just built).
    if (firstId) {
      this._coalesce.subRows.set(firstId, { li: firstLi, statusEl, input: null });
      const firstEntry = this._toolCards.get(firstId);
      if (firstEntry) {
        firstEntry.isCoalesced = true;
      }
    }
    // The collapse-card UI doesn't need the per-call metric chip on
    // the summary — the per-file rows carry status themselves.
    const metricEl = card.querySelector('.coder-tool-metric');
    if (metricEl) metricEl.innerHTML = '';
  }

  /**
   * Update a tool card with its result. Auto-expands on error.
   * For code_edit: renders inline diff.
   * For shell_exec: renders output in monospace.
   */
  updateToolResult(id, result) {
    const card = this._toolCards.get(id);
    if (!card) return;

    const { el, tool, input } = card;
    const statusEl = el.querySelector('.coder-tool-status');
    const body = el.querySelector('.coder-tool-body');
    const elapsedEl = el.querySelector('.coder-tool-elapsed');
    const metricEl = el.querySelector('.coder-tool-metric');

    // Coalesced sub-row: update just the sub-row's status dot + the
    // group's aggregate state. No per-tool body rendering — the
    // collapsed view is intentionally just a file list.
    if (card.isCoalesced) {
      this._updateCoalesceSubRow(el, id, result);
      return;
    }

    // Elapsed timing on the summary row. Computed from the perf
    // timestamp stamped at addToolCall. Surfaces "Edited X · 0.8s"
    // style metadata so the user can see at a glance which calls
    // took meaningful time vs which fired in milliseconds.
    if (elapsedEl) {
      const startedAt = Number(el.dataset.startedAt) || 0;
      if (startedAt > 0) {
        const elapsedMs = performance.now() - startedAt;
        elapsedEl.textContent = _formatToolElapsed(elapsedMs);
      }
    }

    // Update status indicator with verb-tense pattern (Cline/Codex
    // style): running → done/failed. The leading word is mirrored
    // from TOOL_LABELS so e.g. "Edit" → "Edited" reads naturally.
    if (statusEl) {
      statusEl.classList.remove('running');
      statusEl.classList.add(result.success ? 'success' : 'error');
      statusEl.textContent = result.success ? 'done' : 'failed';
    }

    // task_dispatch: replace the in-progress card with the final
    // structured result card (role / model / iters / tokens / stop
    // pill / output preview / inspect link). Done EARLY so the
    // other generic-body renderers below don't accidentally append
    // their own preview onto the structured card.
    if (tool === 'task_dispatch') {
      body.innerHTML = _renderSubagentResultBody(input, result);
      return;
    }

    // Render body based on tool type
    if (tool === 'code_edit' && result.success && input.search && input.replace) {
      // Inline diff
      const diff = computeLineDiff(input.search, input.replace);
      body.innerHTML = renderDiffLines(diff);
      if (metricEl) {
        metricEl.innerHTML = _editMetric(input.search, input.replace);
      }
      if (result.checkpoint) {
        const revertBtn = document.createElement('button');
        revertBtn.className = 'coder-tool-revert';
        revertBtn.textContent = 'Revert';
        revertBtn.dataset.checkpoint = result.checkpoint;
        body.appendChild(revertBtn);
      }
      // Async: validate the replacement with CodeMind (non-blocking)
      this._validateEdit(input.path, input.replace, el);
    } else if (tool === 'code_edit_batch' && result.success && Array.isArray(input.edits)) {
      // Multi-edit-in-one-file: render each {search, replace} pair
      // as its own sub-card with a mini-diff. Pre-2026-05-31 this
      // path fell through to the generic renderer and showed nothing
      // about what was actually changed — Codex/Cline both render
      // batches as sub-rows so the user can scan per-edit.
      const subs = [];
      let totalAdds = 0;
      let totalDels = 0;
      for (let i = 0; i < input.edits.length; i++) {
        const edit = input.edits[i] || {};
        const search = edit.search || '';
        const replace = edit.replace || '';
        const diff = computeLineDiff(search, replace);
        let adds = 0, dels = 0;
        for (const d of diff) {
          if (d.type === 'added') adds++;
          else if (d.type === 'removed') dels++;
        }
        totalAdds += adds;
        totalDels += dels;
        const diffHtml = renderDiffLines(diff);
        subs.push(`
          <details class="coder-tool-subcard" open>
            <summary class="coder-tool-subfile">
              <span class="coder-tool-subfile-path">Edit ${i + 1}</span>
              <span class="coder-tool-subfile-metric">
                <span class="coder-tool-metric-add">+${adds}</span>
                <span class="coder-tool-metric-del">-${dels}</span>
              </span>
            </summary>
            <div class="coder-tool-subcard-body">${diffHtml}</div>
          </details>
        `);
      }
      body.innerHTML = subs.join('');
      if (metricEl) {
        metricEl.innerHTML = `<span class="coder-tool-metric-add">+${totalAdds}</span><span class="coder-tool-metric-del">-${totalDels}</span>`;
      }
    } else if (tool === 'apply_patch' && result.success && input.patch) {
      // Unified-diff patch — parse client-side and render per-file
      // sub-cards. Same scan pattern as code_edit_batch but with
      // the patch's own +/-/space line markers preserved instead of
      // re-computing a diff.
      const segments = _parsePatch(input.patch);
      body.innerHTML = _renderPatchSegments(segments);
      let totalAdds = 0;
      let totalDels = 0;
      for (const seg of segments) {
        for (const h of seg.hunks) {
          for (const l of h.lines) {
            if (l.type === 'add') totalAdds++;
            else if (l.type === 'del') totalDels++;
          }
        }
      }
      if (metricEl) {
        metricEl.innerHTML = `<span class="coder-tool-metric-add">+${totalAdds}</span><span class="coder-tool-metric-del">-${totalDels}</span>`;
      }
    } else if (tool === 'file_write' && result.success && input.content) {
      // New-file preview: render the content with a "first N lines"
      // collapse so users can verify what the agent created without
      // an explicit file_read after the fact. Auto-expand a small
      // file (<= 12 lines) since the whole thing fits.
      const lineCount = input.content.split('\n').length;
      const preview = _previewLines(input.content, 12);
      body.innerHTML = `
        <div class="coder-tool-content-preview">
          <pre class="coder-tool-shell">${escapeHtml(preview.text)}</pre>
          ${preview.truncated ? `<div class="coder-tool-more">… ${lineCount - 12} more lines</div>` : ''}
        </div>
      `;
      if (metricEl) {
        metricEl.innerHTML = '';
        const added = document.createElement('span');
        added.className = 'coder-tool-metric-add';
        added.textContent = `+${lineCount}`;
        const label = document.createElement('span');
        label.className = 'coder-tool-lines-label';
        label.textContent = 'lines';
        metricEl.append(added, label);
      }
      // Validate new file content (existing CodeMind path)
      this._validateEdit(input.path, input.content, el);
    } else if (tool === 'file_read' && result.success && result.output_preview) {
      // Read preview: show the first N lines of what the agent saw.
      // Mirrors file_write but reads the preview from the result.
      const preview = _previewLines(result.output_preview, 12);
      body.innerHTML = `
        <pre class="coder-tool-shell">${escapeHtml(preview.text)}</pre>
        ${preview.truncated ? `<div class="coder-tool-more">… more · expand</div>` : ''}
      `;
    }

    if ((tool === 'shell_exec' || tool === 'shell_read') && result.output_preview) {
      const pre = document.createElement('div');
      pre.className = 'coder-tool-shell';
      pre.textContent = result.output_preview;
      body.appendChild(pre);
    } else if (
      result.output_preview
      && tool !== 'code_edit'
      && tool !== 'code_edit_batch'
      && tool !== 'apply_patch'
      && tool !== 'file_write'
      && tool !== 'file_read'
    ) {
      // Default output_preview path for any tool we don't have a
      // structured renderer for. The explicit deny-list above keeps
      // structured renderers from getting a generic output_preview
      // tacked on after the diff/preview content.
      const pre = document.createElement('div');
      pre.className = 'coder-tool-shell';
      pre.textContent = result.output_preview;
      body.appendChild(pre);
    }

    // Browser screenshot — render the captured PNG inline so the user
    // doesn't have to dig through the files browser to see what the
    // agent actually saw. Path comes through metadata.browser.path
    // (per BrowserScreenshotTool's metadata shape). The /raw route
    // serves with image/* Content-Type so the browser renders rather
    // than downloads.
    if (tool === 'browser_screenshot' && result?.metadata?.browser?.path) {
      const path = result.metadata.browser.path;
      const ws = this._workspaceId;
      if (ws) {
        const img = document.createElement('img');
        img.className = 'coder-tool-screenshot';
        img.loading = 'lazy';
        img.src = `/api/coder/files/${encodeURIComponent(ws)}/raw?path=${encodeURIComponent(path)}`;
        img.alt = result.metadata.browser.title || 'browser screenshot';
        // Click → open full-size in a new tab (the embed is width-capped
        // for readability; users sometimes want to inspect details).
        img.addEventListener('click', () => window.open(img.src, '_blank', 'noopener'));
        // Graceful fallback: if the /raw fetch fails (the workspace was
        // reaped between capture and view, or the file was cleaned up),
        // swap the broken-image glyph for a quiet note.
        img.addEventListener('error', () => {
          const note = document.createElement('div');
          note.className = 'coder-tool-empty';
          note.textContent = '(screenshot unavailable — workspace may have been cleaned up)';
          img.replaceWith(note);
        });
        body.appendChild(img);
        el.open = true;  // screenshots are the point — auto-expand
      }
    }

    // Auto-expand on error or for shell/read tools with output
    const autoExpandTools = new Set([
      'shell_exec', 'shell_read', 'dir_tree', 'test_run', 'env_info', 'git',
    ]);
    if (!result.success) {
      el.open = true;
      if (result.output_preview) {
        const errPre = body.querySelector('.coder-tool-shell') || document.createElement('div');
        errPre.className = 'coder-tool-shell';
        errPre.textContent = result.output_preview;
        if (!body.contains(errPre)) body.appendChild(errPre);
      }
    } else if (autoExpandTools.has(tool) && result.output_preview) {
      // Show output for tools where the result IS the point
      el.open = true;
    }

    // Update serialized message. Skipped during history replay — the
    // result came FROM _messages (see loadHistory), and the linear find
    // per tool card made a big replay O(n²).
    if (!this._historyMode) {
      const msg = this._messages.find(m => m.id === id);
      if (msg) msg.result = result;
    }

    this._scrollToBottom();
    // Status flipped from running → done/error; refresh the banner so
    // it either swaps to the next running card or hides.
    this._refreshRunningBanner();
  }

  /** Append shell output to the active tool card's body. */
  appendShellOutput(text) {
    // Find the last running tool card
    const cards = [...this._toolCards.values()];
    const active = cards[cards.length - 1];
    if (!active) return;

    const body = active.el.querySelector('.coder-tool-body');
    if (!body) return;

    let shellEl = body.querySelector('.coder-tool-shell');
    if (!shellEl) {
      shellEl = document.createElement('div');
      shellEl.className = 'coder-tool-shell';
      body.appendChild(shellEl);
    }
    // Append-only text node — ``textContent += text`` is O(n²) over a long
    // shell stream (re-materializes the whole buffer each chunk). Scroll is
    // already rAF-coalesced in _scrollToBottom.
    shellEl.appendChild(document.createTextNode(text));
    this._scrollToBottom();
  }

  /**
   * Update a task_dispatch card with a SubagentProgress event from the
   * inner loop. Binds the subagent's instance_id to the most-recent
   * running task_dispatch card on first sight (the dispatcher mints
   * instance_id after the card is created, so we can't pre-bind),
   * then appends one row per event into the .coder-subagent-activity-
   * log list. Also surfaces the cancel button + live token/iter stats.
   */
  updateSubagentProgress(progress) {
    if (!progress || !progress.instance_id) return;
    const instanceId = String(progress.instance_id);

    // Locate the card. First sighting binds to the most recent
    // running task_dispatch card without a data-subagent-id; later
    // events go to the same card directly.
    let cardId = this._subagentCardByInstance.get(instanceId);
    let entry = cardId ? this._toolCards.get(cardId) : null;
    if (!entry) {
      // First event for this instance_id — find an unbound running
      // task_dispatch card from the back of the insertion order.
      const all = [...this._toolCards.entries()];
      for (let i = all.length - 1; i >= 0; i--) {
        const [tid, ent] = all[i];
        if (ent.tool !== 'task_dispatch') continue;
        const cardEl = ent.el.querySelector('.coder-subagent-card--running');
        if (cardEl && !cardEl.dataset.subagentId) {
          cardEl.dataset.subagentId = instanceId;
          this._subagentCardByInstance.set(instanceId, tid);
          cardId = tid;
          entry = ent;
          // Reveal cancel button + wire its click handler.
          const cancelBtn = ent.el.querySelector('.coder-subagent-cancel');
          if (cancelBtn) {
            cancelBtn.hidden = false;
            cancelBtn.addEventListener('click', (ev) => {
              ev.stopPropagation();
              this._cancelSubagent(instanceId, cancelBtn);
            });
          }
          break;
        }
      }
    }
    if (!entry) return;

    const card = entry.el.querySelector('.coder-subagent-card--running');
    if (!card) return;

    const activity = card.querySelector('.coder-subagent-activity');
    const log = card.querySelector('.coder-subagent-activity-log');
    const stats = card.querySelector('.coder-subagent-activity-stats');
    if (!log) return;
    if (activity) activity.classList.remove('coder-subagent-activity--empty');

    // Per-row entry. Each phase gets a distinct visual treatment
    // (responding=thinking, tool_call=⚙, tool_result=←, stuck=⚠).
    const iter = Number(progress.iteration) || 0;
    const phase = String(progress.phase || '');
    const toolName = String(progress.tool_name || '');
    const preview = _truncForActivity(progress.text_preview || '', 120);
    let icon = '·';
    let phaseLabel = phase;
    if (phase === 'responding') { icon = '◇'; phaseLabel = 'thinking'; }
    else if (phase === 'tool_call') { icon = '⚙'; phaseLabel = `→ ${toolName}`; }
    else if (phase === 'tool_result') { icon = '←'; phaseLabel = `${toolName} done`; }
    else if (phase === 'stuck') { icon = '⚠'; phaseLabel = 'stuck'; }
    else if (phase === 'done') { icon = '✓'; phaseLabel = 'done'; }

    const li = document.createElement('li');
    li.className = `coder-subagent-activity-row coder-subagent-activity-row--${escapeHtml(phase)}`;
    li.innerHTML = `
      <span class="coder-subagent-activity-icon" aria-hidden="true">${icon}</span>
      <span class="coder-subagent-activity-iter">i${iter}</span>
      <span class="coder-subagent-activity-phase">${escapeHtml(phaseLabel)}</span>
      ${preview ? `<span class="coder-subagent-activity-preview">${escapeHtml(preview)}</span>` : ''}
    `;
    log.appendChild(li);

    // Trim log to the most recent 30 rows so a long-running subagent
    // doesn't unboundedly grow the DOM. Older rows are still in the
    // server-side run record (GET /api/coder/subagents/{id}).
    while (log.children.length > 30) log.removeChild(log.firstChild);

    // Live stats row — tokens + wallclock at the top of the activity
    // panel so the user sees the run "spending" without scrolling.
    if (stats) {
      const tIn = Number(progress.tokens_in) || 0;
      const tOut = Number(progress.tokens_out) || 0;
      const wall = Number(progress.wallclock_ms) || 0;
      stats.textContent = `iter ${iter} · ${_formatSubagentTokens(tIn + tOut)} tok · ${_formatSubagentDuration(wall)}`;
    }

    this._scrollToBottom();
  }

  /**
   * POST /api/coder/subagents/{id}/cancel. Updates the cancel button's
   * label to reflect pending → cancelled while the request is in
   * flight, falls back on failure. The dispatch loop synthesises a
   * cancelled-SubagentResult so the result card surfaces normally
   * with stop_reason="cancelled" and the recovery hint.
   */
  async _cancelSubagent(instanceId, btn) {
    if (!instanceId || !btn || btn.disabled) return;
    const originalLabel = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'cancelling…';
    try {
      const res = await fetch(
        `/api/coder/subagents/${encodeURIComponent(instanceId)}/cancel`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason: 'user cancelled from chat card' }),
        },
      );
      if (!res.ok) {
        // 404 = already finished — let the result card land normally.
        // Anything else = real failure; restore the button.
        if (res.status === 404) {
          btn.textContent = 'finishing…';
          // Hide a beat later; the result chunk should arrive imminently.
          setTimeout(() => { btn.hidden = true; }, 1500);
          return;
        }
        throw new Error(`HTTP ${res.status}`);
      }
      btn.textContent = 'cancelled';
      // Don't hide — leave the visible "cancelled" label until the
      // result card replaces the running body.
    } catch (err) {
      console.warn('subagent cancel failed', err);
      btn.disabled = false;
      btn.textContent = originalLabel;
    }
  }

  /** Start streaming an agent response. Returns the element for appending. */
  startResponse() {
    const el = document.createElement('div');
    el.className = 'coder-msg coder-msg-assistant';
    this._append(el);
    this._streamEl = el;
    // Per-stream bookkeeping for the shared incremental renderer.
    this._streamSplit = newSplitState();
    this._streamRenderScheduled = false;
    return el;
  }

  /**
   * Append content to the streaming response.
   *
   * Accumulates raw text synchronously (so finalize/flush always see the
   * latest) and coalesces the DOM render to one pass per animation frame
   * via the shared stable/active split renderer. The old path did
   * ``innerHTML = renderMarkdown(WHOLE raw)`` on every NDJSON delta — pure
   * O(n²) that pinned the main thread for tens of seconds when a model
   * streamed a long file-write at a fast tok/s. See chat/stream-render.js.
   */
  appendContent(text) {
    if (this._shouldSuppressDuplicateDelta(text)) return;
    if (!this._streamEl) this.startResponse();
    this._streamEl._rawContent = (this._streamEl._rawContent || '') + text;

    if (this._streamRenderScheduled) return;
    this._streamRenderScheduled = true;
    const flush = () => {
      this._streamRenderScheduled = false;
      const el = this._streamEl;
      if (!el) return;
      renderStreamSplit(el, el._rawContent || '', this._streamSplit, { mode: 'coder' });
      this._scrollToBottom(false);
    };
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(flush);
    } else {
      flush(); // test / non-browser host
    }
  }

  /** Finalize the streaming response. */
  finalizeResponse() {
    this._flushStreamingBubble();
  }

  /**
   * Finalize the streaming response after a cancellation.
   *
   * The normal ``finalizeResponse`` commits whatever raw text is in
   * the streaming bubble — on cancel, that text is often mid-
   * sentence or a half-formed tool-call JSON header. The model on
   * the next turn reads the message history verbatim (via
   * ``getMessagesForLLM``), so trailing junk like
   * ``{"path": "/snake.py"`` ends up in its context and disorients
   * its continuation.
   *
   * This variant trims the trailing partial-JSON / dangling-token
   * artifact and appends a clear ``[cancelled: reason]`` marker.
   * Sibling-of fix: ``_render_interruption_stanza`` in the backend
   * puts the matching signal in the ``<prior_turns>`` block — UI
   * and prompt context agree on what happened.
   *
   * Called from ``_stopActiveCoderRun`` *before* the underlying
   * stream is aborted, so this runs while ``_streamEl`` is still
   * live and the regular ``finalizeResponse`` (fired by the abort's
   * ``onComplete('')`` callback a tick later) becomes a no-op.
   *
   * @param {string} [reason='user_cancel'] Echoed into the marker
   *   so the model can tell user_cancel from slash_clear etc.
   */
  finalizeResponseCancelled(reason = 'user_cancel') {
    if (!this._streamEl) {
      // No active bubble — nothing visible was streamed before the
      // cancel landed. We do NOT push a synthetic marker message in
      // this case: that would put a spurious bubble in history for
      // a turn where the model emitted nothing. The backend's
      // turn_summary already records the cancellation; an empty UI
      // is honest about what was visible.
      return;
    }
    const raw = this._streamEl._rawContent || this._streamEl.textContent || '';
    // Trim trailing partial-JSON artifacts. The most common shapes
    // are an unmatched `{` / `[` followed by partial keys, or a
    // trailing comma after a property — both leave the markdown
    // renderer confused and the model unable to tell where the
    // prose stopped.
    let cleaned = raw.trimEnd();
    // Drop a trailing fragment that starts at the last unmatched
    // brace/bracket so we don't keep partial JSON in history. The
    // regex finds the latest opening token with no matching close
    // *after* the last newline boundary; conservative — only
    // strips when there's clearly a dangling structure.
    const lastOpen = Math.max(cleaned.lastIndexOf('{'), cleaned.lastIndexOf('['));
    if (lastOpen > -1) {
      const tail = cleaned.slice(lastOpen);
      const opens = (tail.match(/[\{\[]/g) || []).length;
      const closes = (tail.match(/[\}\]]/g) || []).length;
      if (opens > closes) {
        cleaned = cleaned.slice(0, lastOpen).trimEnd();
      }
    }
    const marker = `\n\n_[cancelled: ${reason}]_`;
    const final = (cleaned || '') + marker;
    this._streamEl._rawContent = final;
    this._streamEl.innerHTML = renderMarkdown(final);
    highlightCodeDeferred(this._streamEl);
    this._flushStreamingBubble();
  }

  /**
   * Close out the active streaming assistant bubble (if any), commit
   * its accumulated content to the serialized history, and null the
   * pointer so the next ``appendContent`` opens a fresh bubble.
   *
   * Called in two places:
   *   - ``addToolCall`` — to keep prose and tool cards in chronological
   *     order as the model interleaves them within one turn.
   *   - ``finalizeResponse`` — at stream end, when the model is done
   *     for this turn.
   *
   * Empty / whitespace-only bubbles are discarded so a tool-first turn
   * (model starts with ``file_read`` before any prose) doesn't leave
   * an invisible empty-paragraph artifact in the DOM or history.
   *
   * @private
   */
  _flushStreamingBubble() {
    if (!this._streamEl) return;
    const raw = this._streamEl._rawContent || this._streamEl.textContent || '';
    const priorAssistant = [...this._messages].reverse().find(
      (msg) => msg.role === 'assistant' && (msg.content || '').trim(),
    );
    if (
      raw.trim()
      && priorAssistant
      && (priorAssistant.content || '').trim() === raw.trim()
    ) {
      this._streamEl.remove();
    } else if (raw.trim()) {
      // Settle the still-unrendered tail via the SAME bounded incremental
      // renderer the live stream uses — NOT a from-scratch
      // ``renderMarkdown(raw)`` over the WHOLE bubble. This path fires once
      // per INTERLEAVED TOOL CALL (see ``addToolCall`` above), so in an
      // agentic coder turn with dozens of tool calls the old code re-parsed
      // the entire prose bubble dozens of times — O(bubble) × (tool calls),
      // the coder-mode freeze. renderStreamSplit only promotes the unsettled
      // tail (bounded chunks) and leaves the already-rendered DOM intact, so
      // each flush is O(tail). The committed bubble's split scaffold uses
      // display:contents, so it's visually identical to the flat render that
      // loadHistory produces on reload. Highlight + hooks deferred to idle.
      if (this._streamEl.querySelector(':scope > .response-body')) {
        renderStreamSplit(this._streamEl, raw, this._streamSplit, { mode: 'coder' });
        highlightCodeDeferred(this._streamEl);
      }
      const id = this._nextId();
      this._messages.push({ id, role: 'assistant', content: raw });
    } else {
      // Empty bubble — remove the DOM node so we don't leave a
      // zero-height margin-only element between the tool cards.
      this._streamEl.remove();
    }
    this._streamEl = null;
    this._lastContentDelta = '';
  }

  /** Update the active workspace id (callers should fire this whenever
   *  the surrounding UI switches workspaces). Used by tool-result
   *  renderers that need workspace-scoped URLs — currently the
   *  browser_screenshot inline embed.
   */
  setWorkspaceId(id) {
    this._workspaceId = id || '';
  }

  /** Show an error message. */
  addError(message) {
    const el = document.createElement('div');
    el.className = 'coder-msg-error';
    el.textContent = message;
    this._append(el);
  }

  /**
   * Show a recoverable-error pill with Try Again / Stop actions.
   *
   * Used when the backend exhausted its transient-retry budget (429,
   * 5xx, network blip) — the same request is likely to succeed if the
   * user retries in a moment. Visually distinct from ``addError`` so the
   * user reads it as "the agent paused, your choice what's next" rather
   * than "your work is dead". Warning-tinted, not error-tinted.
   *
   * @param {string} message - subtitle copy shown under the title
   * @param {object} [opts]
   * @param {string} [opts.title="Backend timeout"] - bold lead line
   * @param {() => Promise<void> | void} [opts.onRetry] - fires on Try Again
   * @param {() => void} [opts.onDismiss] - fires on Stop
   * @returns {HTMLElement} the pill element (caller can remove if needed)
   */
  addRecoverableError(message, {
    title, onRetry, onDismiss, permanent = false,
    retryLabel = 'Try Again', busyLabel = 'Retrying…', dismissLabel,
  } = {}) {
    const el = document.createElement('div');
    el.className = permanent
      ? 'coder-msg-recoverable coder-msg-recoverable-permanent'
      : 'coder-msg-recoverable';
    const _esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    const dismissText = dismissLabel || (permanent ? 'Dismiss' : 'Stop');
    // Permanent (4xx) errors: retrying the same payload hits the same
    // wall, so the primary button just frustrates the user. Show only
    // Dismiss. Transient/interrupted errors keep the two-button pill;
    // ``retryLabel`` lets callers say "Resume" instead of "Try Again".
    const actionsHtml = permanent
      ? `<button class="coder-msg-recoverable-btn coder-msg-recoverable-btn-secondary" data-action="dismiss" type="button">${_esc(dismissText)}</button>`
      : `<button class="coder-msg-recoverable-btn coder-msg-recoverable-btn-primary" data-action="retry" type="button">${_esc(retryLabel)}</button>`
        + `<button class="coder-msg-recoverable-btn coder-msg-recoverable-btn-secondary" data-action="dismiss" type="button">${_esc(dismissText)}</button>`;
    el.innerHTML = `
      <div class="coder-msg-recoverable-icon" aria-hidden="true">&#9888;</div>
      <div class="coder-msg-recoverable-body">
        <div class="coder-msg-recoverable-title"></div>
        <div class="coder-msg-recoverable-subtitle"></div>
      </div>
      <div class="coder-msg-recoverable-actions">${actionsHtml}</div>
    `;
    el.querySelector('.coder-msg-recoverable-title').textContent =
      title || (permanent ? 'Provider rejected request' : 'Backend timeout');
    el.querySelector('.coder-msg-recoverable-subtitle').textContent = message || '';
    const dismissEl = () => { if (el.parentElement) el.remove(); };
    const retryBtn = el.querySelector('[data-action="retry"]');
    const stopBtn = el.querySelector('[data-action="dismiss"]');
    if (retryBtn) {
      retryBtn.addEventListener('click', () => {
        retryBtn.disabled = true;
        stopBtn.disabled = true;
        retryBtn.textContent = busyLabel;
        // Dismiss after the retry kicks off — the new turn will render
        // its own stream into the conversation, so the pill's job is done.
        Promise.resolve(onRetry?.()).finally(dismissEl);
      });
    }
    stopBtn.addEventListener('click', () => {
      onDismiss?.();
      dismissEl();
    });
    this._append(el);
    return el;
  }

  /** Clear all messages and reset state. */
  clear() {
    // Cancel any in-flight history backfill — its DOM nodes are about to
    // be wiped and its message range no longer exists.
    this._historyGen++;
    this._container.innerHTML = '';
    this._messages = [];
    this._toolCards.clear();
    this._streamEl = null;
    this._lastContentDelta = '';
    this._idCounter = 0;
    this._resetWindow();
  }

  /**
   * Drop the most recent user → assistant exchange from the
   * conversation, including any interleaved tool cards.
   *
   * Called by coder.js after the backend ``/rewind`` endpoint reports
   * success. Walks from the end of ``_messages`` backwards: pops the
   * tail until (and including) the most recent user message, then
   * removes every matching DOM node from the conversation container.
   *
   * Returns the count of messages removed (0 when the conversation
   * was empty / had no user message — the rewind never reaches this
   * path in that case because the button is hidden).
   */
  rewindLastTurn() {
    if (!this._messages.length) return 0;
    // Find the index of the most recent user message — everything
    // from there to the end is the "last turn" (the user's prompt
    // plus the assistant's response and any tool cards in between).
    let cutoff = -1;
    for (let i = this._messages.length - 1; i >= 0; i--) {
      if (this._messages[i].role === 'user') {
        cutoff = i;
        break;
      }
    }
    if (cutoff < 0) return 0;
    const removed = this._messages.splice(cutoff);
    // Remove the message ids' tool cards from the lookup map so
    // subsequent renders don't try to update phantom DOM nodes.
    for (const m of removed) {
      if (m.id) this._toolCards.delete(m.id);
    }
    // Discard a still-streaming bubble — rewind cancels mid-flight too.
    // Element.remove() on a detached node is a no-op, so no guard needed.
    if (this._streamEl) {
      this._streamEl.remove();
      this._streamEl = null;
      this._lastContentDelta = '';
    }
    // Re-render from the surviving _messages. Cheaper than a precise
    // per-node removal (which would need to track turn boundaries on
    // every DOM node) and avoids the risk of orphan tool cards or
    // dangling thinking blocks bleeding through.
    const survivors = this._messages.slice();
    this._messages = [];
    this._container.innerHTML = '';
    this._idCounter = 0;
    this._resetWindow();
    if (survivors.length === 0) {
      // Re-surface the onboarding empty state if everything is gone.
      const empty = document.getElementById('coder-conv-empty');
      if (empty) this._container.appendChild(empty);
    } else {
      this.loadHistory(survivors);
    }
    return removed.length;
  }

  _shouldSuppressDuplicateDelta(text) {
    if (!text) return true;
    const duplicate = (
      this._lastContentDelta === text
      && (text.includes('\n') || text.trim().length >= 80)
    );
    this._lastContentDelta = text;
    return duplicate;
  }

  /**
   * Load history from a serialized message array.
   *
   * Two-phase render (see _HISTORY_TAIL_SYNC): the newest tail renders
   * synchronously so the conversation is instantly usable at its normal
   * anchor (the bottom); older messages backfill upward in budgeted
   * slices without blocking paint or input. ``_messages`` is complete
   * from the moment this returns — getMessagesForLLM / getHistory never
   * see a partial history while the DOM is still filling.
   */
  loadHistory(messages) {
    this.clear();
    // Fresh load lands at the bottom — reset any stale scroll flag so the
    // windowing trim (which no-ops while scrolled up) and the closing
    // scroll-to-bottom both run.
    this._userScrolledUp = false;
    // Serialized history is authoritative immediately (the render below is
    // display-only — _historyMode suppresses the per-call _messages pushes).
    this._messages = [...messages];

    const cut = Math.max(0, messages.length - _HISTORY_TAIL_SYNC);
    for (let i = cut; i < messages.length; i++) {
      this._renderHistoryMessage(messages[i]);
    }
    // A tool card coalesce-run must never bridge the sync tail and a LIVE
    // tool call that arrives right after load (or the async backfill).
    this._coalesce.cardEl = null;
    this._coalesce.tool = '';

    // Window the freshly-loaded history: a long session would otherwise paint
    // thousands of nodes and then pay O(all) layout on the first stream. Trim
    // detaches everything past the cap in one pass (retained for scroll-up).
    this._trimWindowIfNeeded();
    this._scrollToBottom(false);

    if (cut > 0) {
      void this._backfillHistory(messages, cut, this._historyGen);
    }
  }

  /**
   * Render one persisted message into ``this._container``. Extracted from
   * the old loadHistory loop verbatim; ``_historyMode`` makes addToolCall /
   * updateToolResult display-only (no _messages mutation) for the duration
   * of the call — the flag is scoped synchronously so live events landing
   * between backfill slices behave normally.
   * @private
   */
  _renderHistoryMessage(msg) {
    this._historyMode = true;
    try {
      if (msg.role === 'user') {
        const el = document.createElement('div');
        el.className = 'coder-msg coder-msg-user';
        // Split text + attachment rendering — same layout as
        // addUserMessage so a refreshed conversation looks identical
        // to a freshly-rendered one.
        const textEl = document.createElement('div');
        textEl.className = 'coder-msg-user-text';
        textEl.textContent = msg.content || '';
        el.appendChild(textEl);
        const ready = (msg.attachments || []).filter(a => a && a.url);
        if (ready.length) {
          const grid = document.createElement('div');
          grid.className = 'coder-msg-attachments';
          for (const att of ready) {
            grid.appendChild(_renderMessageAttachment(att));
          }
          el.appendChild(grid);
        }
        this._container.appendChild(el);
      } else if (msg.role === 'tool') {
        this.addToolCall(msg.id, msg.tool, msg.input || {});
        // No persisted result → the tool was in flight when the session
        // ended. Without this synthetic reconcile the card stays stuck
        // in running state forever and the top-of-conv running banner
        // follows the user back to it on every load.
        const reconciled = msg.result
          ? msg.result
          : _synthesizeInterruptedResult(msg.tool, msg.input || {});
        this.updateToolResult(msg.id, reconciled);
      } else if (msg.role === 'assistant') {
        const el = document.createElement('div');
        el.className = 'coder-msg coder-msg-assistant';
        el.innerHTML = renderMarkdown(msg.content || '');
        this._container.appendChild(el);
        highlightCodeDeferred(el);
      }
    } finally {
      this._historyMode = false;
    }
  }

  /**
   * Backfill messages[0..cut) ABOVE the already-rendered tail in budgeted
   * slices. Chunks are taken newest-first (the range adjacent to the tail
   * fills in first — matching what a scrolling-up user reaches first), each
   * chunk rendered in document order into a detached host and inserted
   * above the previous content in one move. Scroll position is preserved
   * the same way _rehydrateOlder does it.
   *
   * Cancellation: ``gen`` is compared against this._historyGen after every
   * yield — clear() / a newer loadHistory bumps the generation and any
   * in-flight backfill aborts without touching the (already replaced) DOM.
   * @private
   */
  async _backfillHistory(messages, cut, gen) {
    let anchor = this._container.firstChild; // current oldest rendered node
    let end = cut; // exclusive — render [start, end) per chunk

    while (end > 0) {
      await yieldToPaint();
      if (gen !== this._historyGen) return; // superseded — stop silently

      const sliceStart = performance.now();
      // Render whole chunks under one budget window: walk messages backward
      // in fixed-size bites until the slice budget is spent.
      while (end > 0 && (performance.now() - sliceStart) < _HISTORY_BUDGET_MS) {
        const start = Math.max(0, end - 10);
        const chunkHost = document.createElement('div');
        const realContainer = this._container;
        this._container = chunkHost;
        try {
          // A coalesce-run must not bridge chunks: the previous chunk holds
          // NEWER messages, so folding this chunk's reads into its card
          // would misorder the transcript.
          this._coalesce.cardEl = null;
          this._coalesce.tool = '';
          for (let i = start; i < end; i++) {
            this._renderHistoryMessage(messages[i]);
          }
        } finally {
          this._container = realContainer;
        }
        // Move the chunk above everything rendered so far, preserving the
        // user's viewport (manual compensation, mirroring _rehydrateOlder —
        // when pinned at the bottom the delta math is a no-op-equivalent).
        const scrollEl = this._scrollEl;
        const beforeHeight = scrollEl ? scrollEl.scrollHeight : 0;
        const beforeTop = scrollEl ? scrollEl.scrollTop : 0;
        const newAnchor = chunkHost.firstChild;
        // Ordered move: insert children first-to-last before the anchor so
        // document order is preserved. `anchor` may be null on the very
        // first insert into an empty container — insertBefore(node, null)
        // is appendChild, which is correct there.
        while (chunkHost.firstChild) {
          this._container.insertBefore(chunkHost.firstChild, anchor);
        }
        if (newAnchor) anchor = newAnchor;
        if (scrollEl) {
          if (this._userScrolledUp) {
            scrollEl.scrollTop = beforeTop + (scrollEl.scrollHeight - beforeHeight);
          } else {
            scrollEl.scrollTop = scrollEl.scrollHeight;
          }
        }
        end = start;
      }
    }
    // Backfill complete — drop any coalesce state left by the last chunk so
    // the next LIVE tool call starts a fresh card.
    this._coalesce.cardEl = null;
    this._coalesce.tool = '';
  }

  /** Get serializable message history. */
  getHistory() {
    return this._messages;
  }

  /** Get the messages formatted for LLM context (user + assistant only). */
  getMessagesForLLM() {
    return this._messages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .map(m => ({ role: m.role, content: m.content }));
  }

  /**
   * Async-validate edited code with CodeMind (tree-sitter).
   * Shows a warning badge on the tool card if syntax errors are found.
   * Non-blocking — doesn't prevent the tool card from rendering.
   * @private
   */
  async _validateEdit(path, code, cardEl) {
    const lang = _langFromPath(path);
    if (!lang || !code) return;
    await _ensureCodeMind();
    if (!_codeMindValidate) return;
    try {
      const { valid, errors } = await _codeMindValidate(code, lang);
      if (!valid && errors.length > 0) {
        const badge = document.createElement('span');
        badge.className = 'coder-tool-lint-badge';
        badge.title = errors.map(e => `Line ${e.startPosition?.row + 1}: ${e.type}`).join('\n');
        badge.textContent = `⚠ ${errors.length} syntax ${errors.length === 1 ? 'error' : 'errors'}`;
        const summary = cardEl.querySelector('summary');
        if (summary) summary.appendChild(badge);
      }
    } catch {
      // CodeMind unavailable — silent, validation is best-effort
    }
  }

  // ── Private ───────────────────────────────────────────────────────────

  _nextId() {
    return `msg_${Date.now()}_${++this._idCounter}`;
  }

  _append(el) {
    this._container.appendChild(el);
    // Bound the live DOM. Runs after the append (so the new node is counted)
    // and before the scroll re-pin (so detaching far-above nodes is invisible
    // while we're anchored at the bottom — the streaming common case).
    this._trimWindowIfNeeded();
    this._scrollToBottom();
  }

  /**
   * Top-level conversation nodes currently in the document, oldest first,
   * excluding the windowing sentinel itself.
   * @private
   */
  _liveNodes() {
    const out = [];
    for (const el of this._container.children) {
      if (el !== this._windowSentinel) out.push(el);
    }
    return out;
  }

  /**
   * Detach the oldest live nodes when the live DOM grows past the cap, so the
   * document never holds more than ~_MAX_LIVE_NODES message/tool nodes. The
   * detached nodes are RETAINED (not destroyed) in ``_detachedTop`` for exact
   * re-attach on scroll-up. No-op while the user has scrolled up.
   * @private
   */
  _trimWindowIfNeeded() {
    // DISABLED. The JS detach/re-attach windowing this drove was buggy on fast
    // scroll-up (detached nodes weren't re-hydrated in time → blank/grey region
    // where history should be). Replaced by CSS `content-visibility: auto` on
    // the conversation children (see coder.css): the browser skips layout/paint
    // for off-screen cards — same O(visible) win — while keeping every node in
    // the DOM, so nothing goes missing, taps work, and scroll height is exact.
    // Left inert (rather than ripped out) to keep this change minimal; with no
    // detach, _detachedTop stays empty so the sentinel + rehydrate paths never
    // fire. nodesToDetach/nodesToRehydrate/coder-window.js retained for tests.
    void nodesToDetach; void nodesToRehydrate;
  }

  /**
   * Keep the "↑ show earlier" affordance pinned to the top of the container
   * while nodes are windowed out; remove it when none remain.
   * @private
   */
  _updateWindowSentinel() {
    const hidden = this._detachedTop.length;
    if (hidden === 0) {
      if (this._windowSentinel) {
        this._windowSentinel.remove();
        this._windowSentinel = null;
      }
      return;
    }
    if (!this._windowSentinel) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'coder-conv-load-earlier';
      btn.addEventListener('click', () => this._rehydrateOlder());
      this._windowSentinel = btn;
    }
    const batch = nodesToRehydrate({ detachedCount: hidden, batch: _REHYDRATE_BATCH });
    this._windowSentinel.textContent = `↑ Show ${batch} earlier · ${hidden} hidden`;
    if (this._container.firstChild !== this._windowSentinel) {
      this._container.insertBefore(this._windowSentinel, this._container.firstChild);
    }
  }

  /**
   * Re-attach the next batch of windowed-out nodes to the top of the
   * conversation, preserving the user's scroll position (content added above
   * the viewport would otherwise shift everything down).
   * @private
   */
  _rehydrateOlder() {
    const count = nodesToRehydrate({
      detachedCount: this._detachedTop.length,
      batch: _REHYDRATE_BATCH,
    });
    if (count <= 0) return;
    const scrollEl = this._scrollEl;
    const beforeHeight = scrollEl ? scrollEl.scrollHeight : 0;
    const beforeTop = scrollEl ? scrollEl.scrollTop : 0;

    // Pop the most-recently-detached `count` nodes (those closest to the
    // current top) and re-insert them in correct visual order: oldest of the
    // batch right after the sentinel, each newer one below it.
    const batch = [];
    for (let i = 0; i < count; i++) batch.push(this._detachedTop.pop());
    batch.reverse(); // now oldest-first
    this._updateWindowSentinel(); // ensure sentinel exists / count is current
    let anchor = this._windowSentinel;
    for (const node of batch) {
      if (anchor && anchor.nextSibling) {
        this._container.insertBefore(node, anchor.nextSibling);
      } else if (anchor) {
        this._container.appendChild(node);
      } else {
        this._container.insertBefore(node, this._container.firstChild);
      }
      anchor = node;
    }
    this._updateWindowSentinel(); // drops the sentinel if we just emptied it

    // Anchor the viewport: content grew above, so bump scrollTop by the delta.
    if (scrollEl) {
      const afterHeight = scrollEl.scrollHeight;
      scrollEl.scrollTop = beforeTop + (afterHeight - beforeHeight);
    }
  }

  /**
   * Reset windowing bookkeeping. Called wherever the container is cleared
   * wholesale (clear / rewind) so stale detached-node refs don't leak and the
   * sentinel pointer matches the (now empty) DOM.
   * @private
   */
  _resetWindow() {
    this._detachedTop = [];
    this._windowSentinel = null;
  }

  /**
   * Follow the conversation to the bottom — coalesced to one scroll per
   * animation frame, always instant.
   *
   * Auto-follow during a fast stream / tool-call flurry used to call
   * ``scrollTo({behavior:'smooth'})`` synchronously on every append, shell-
   * output line, and tool result. ``smooth`` restarts its animation on each
   * call (the target keeps moving anyway), and each call plus the scroll
   * events it spawns forced a layout — the residual "screen unresponsive
   * between requests" jank. One instant scroll per rAF is both smoother-
   * feeling and orders of magnitude cheaper. The ``smooth`` parameter is
   * kept for call-site compatibility but ignored; discrete user jumps
   * (e.g. _jumpToRunningCard) keep their own smooth scrollIntoView.
   */
  _scrollToBottom(_smooth = true) {
    if (this._userScrolledUp) return;
    if (this._scrollScheduled) return;
    this._scrollScheduled = true;
    const doScroll = () => {
      this._scrollScheduled = false;
      if (this._userScrolledUp) return;
      const el = this._scrollEl;
      if (el) el.scrollTop = el.scrollHeight;
    };
    if (typeof window !== 'undefined'
        && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(doScroll);
    } else {
      doScroll();
    }
  }
}
