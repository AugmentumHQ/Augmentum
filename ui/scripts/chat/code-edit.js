/* ==========================================================================
   Code Edit — AI Edit Flow, Diff Patching, Quick Actions, Lint/Fixers
   Extracted from chat.js. Handles all LLM-powered code editing and the
   deterministic auto-fix pipeline.
   ========================================================================== */

import { app, escapeHtml, showToast } from '../app.js';
import { icons } from './constants.js';
import { getSettings } from '../settings.js';
import * as CodeMind from '../codemind.js';
import {
  closePreviewModal,
  toggleHtmlPreview,
  getBlock,
  getBlocksForMessage,
  updateBlock,
  getSessionNode,
  showVersion,
  codeMindValidate,
} from './code-actions.js';
import { safeHighlightElement } from './markdown.js';

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------

const _aiEditInProgress = new WeakSet();

let _acornLoose = null;
let _astring = null;
let _jsFixerLoading = false;

// ---------------------------------------------------------------------------
// Diff Patching — 3-tier fuzzy SEARCH/REPLACE + legacy LINE/INSERT/DELETE
// ---------------------------------------------------------------------------

export function applyDiffPatches(original, diffOutput) {
  let applied = 0;
  const failedPatches = [];

  // --- Phase 1: SEARCH/REPLACE (primary) ---
  const srPatches = [];
  const srRe = /<<<<<<<?\.?\s*SEARCH\n([\s\S]*?)\n?={3,}\n([\s\S]*?)\n?>>>>>>>?\.?\s*REPLACE/g;
  let m;
  while ((m = srRe.exec(diffOutput)) !== null) {
    srPatches.push({ search: m[1].trimEnd(), replace: m[2].trimEnd() });
  }

  if (srPatches.length > 0) {
    let result = original;
    for (const patch of srPatches) {
      // Tier 1: Exact match
      if (result.includes(patch.search)) {
        result = result.replace(patch.search, patch.replace);
        applied++;
        continue;
      }

      // Tier 2: Fuzzy match (trimmed lines, indent rebase)
      const searchLines = patch.search.split('\n');
      const resultLines = result.split('\n');
      const searchTrimmed = searchLines.map(l => l.trim());
      let found = false;

      for (let i = 0; i <= resultLines.length - searchLines.length; i++) {
        const windowTrimmed = resultLines.slice(i, i + searchLines.length).map(l => l.trim());
        if (searchTrimmed.every((line, j) => line === windowTrimmed[j])) {
          const targetIndent = resultLines[i].match(/^(\s*)/)[1];
          const srcLines = patch.replace.split('\n');
          const srcIndent = srcLines[0].match(/^(\s*)/)[1];
          const replaceLines = srcLines.map((line) => {
            if (!line.trim()) return line;
            if (line.startsWith(srcIndent)) return targetIndent + line.slice(srcIndent.length);
            return targetIndent + line.trimStart();
          });
          result = [...resultLines.slice(0, i), ...replaceLines, ...resultLines.slice(i + searchLines.length)].join('\n');
          resultLines.length = 0;
          resultLines.push(...result.split('\n'));
          applied++;
          found = true;
          break;
        }
      }

      // Tier 3: Whitespace-normalized match (collapse runs of spaces/tabs)
      if (!found) {
        const normalize = s => s.replace(/[ \t]+/g, ' ').trim();
        const searchNorm = searchLines.map(normalize);
        const resultLines2 = result.split('\n');

        for (let i = 0; i <= resultLines2.length - searchLines.length; i++) {
          const windowNorm = resultLines2.slice(i, i + searchLines.length).map(normalize);
          if (searchNorm.every((line, j) => line === windowNorm[j])) {
            const targetIndent = resultLines2[i].match(/^(\s*)/)[1];
            const srcLines = patch.replace.split('\n');
            const srcIndent = srcLines[0].match(/^(\s*)/)[1];
            const replaceLines = srcLines.map((line) => {
              if (!line.trim()) return line;
              if (line.startsWith(srcIndent)) return targetIndent + line.slice(srcIndent.length);
              return targetIndent + line.trimStart();
            });
            result = [...resultLines2.slice(0, i), ...replaceLines, ...resultLines2.slice(i + searchLines.length)].join('\n');
            applied++;
            found = true;
            break;
          }
        }
      }

      if (!found) failedPatches.push(patch);
    }

    if (applied > 0) return { code: result, applied, failed: failedPatches };
  }

  // --- Phase 2: @@@ LINE/INSERT/DELETE (legacy fallback) ---
  const lines = original.split('\n');
  const lineOps = [];
  const lineRe = /@@@\s*(?:LINE|LINES?)\s+(\d+)(?:\s*-\s*(\d+))?\s*\n([\s\S]*?)@@@/gi;
  const insertRe = /@@@\s*INSERT\s+AFTER\s+(\d+)\s*\n([\s\S]*?)@@@/gi;
  const deleteRe = /@@@\s*DELETE\s+(\d+)(?:\s*-\s*(\d+))?\s*@@@/gi;

  while ((m = lineRe.exec(diffOutput)) !== null) {
    lineOps.push({ type: 'replace', start: parseInt(m[1], 10), end: m[2] ? parseInt(m[2], 10) : parseInt(m[1], 10), content: m[3].trimEnd() });
  }
  while ((m = insertRe.exec(diffOutput)) !== null) {
    lineOps.push({ type: 'insert', after: parseInt(m[1], 10), content: m[2].trimEnd() });
  }
  while ((m = deleteRe.exec(diffOutput)) !== null) {
    lineOps.push({ type: 'delete', start: parseInt(m[1], 10), end: m[2] ? parseInt(m[2], 10) : parseInt(m[1], 10) });
  }

  if (lineOps.length > 0) {
    lineOps.sort((a, b) => (b.start || b.after || 0) - (a.start || a.after || 0));
    for (const op of lineOps) {
      if (op.type === 'replace') {
        if (op.start >= 1 && op.end <= lines.length) {
          lines.splice(op.start - 1, op.end - op.start + 1, ...op.content.split('\n'));
          applied++;
        } else {
          failedPatches.push({ search: `Lines ${op.start}-${op.end}`, replace: op.content });
        }
      } else if (op.type === 'insert') {
        if (op.after >= 0 && op.after <= lines.length) {
          lines.splice(op.after, 0, ...op.content.split('\n'));
          applied++;
        } else {
          failedPatches.push({ search: `Insert after ${op.after}`, replace: op.content });
        }
      } else if (op.type === 'delete') {
        if (op.start >= 1 && op.end <= lines.length) {
          lines.splice(op.start - 1, op.end - op.start + 1);
          applied++;
        } else {
          failedPatches.push({ search: `Delete ${op.start}-${op.end}`, replace: '' });
        }
      }
    }
    if (applied > 0) return { code: lines.join('\n'), applied, failed: failedPatches };
  }

  // --- Phase 3: Assume full code (last resort) ---
  // Python-tuned models emit "# CHANGES:" despite the prompt asking for
  // "//" — accept both comment styles everywhere we parse the summary.
  const stripped = diffOutput.replace(/(?:\/\/|#)\s*CHANGES?:.*/gi, '').trim();
  if (stripped.includes('<') || stripped.includes('def ') || stripped.includes('function ')) {
    return { code: stripped, applied: 1, failed: [] };
  }
  return { code: original, applied: 0, failed: [] };
}

// ---------------------------------------------------------------------------
// Streaming Diff
// ---------------------------------------------------------------------------

/**
 * Creates and manages a streaming diff view.
 * Shows live diff as tokens arrive in full mode.
 * Returns an object with update() and destroy() methods.
 */
export function createStreamingDiff(originalCode, container) {
  const originalLines = originalCode.split('\n');

  const wrapper = document.createElement('div');
  wrapper.className = 'code-streaming-diff';
  container.after(wrapper);
  container.hidden = true;

  let rafId = null;
  let dirty = false;
  let currentBuffer = '';
  let lastRenderTime = 0;
  const RENDER_INTERVAL = 200;

  function render() {
    if (!dirty) return;
    dirty = false;

    const diff = computeLineDiff(originalCode, currentBuffer);
    let html = '';
    let sOldLn = 1, sNewLn = 1;
    for (const d of diff) {
      const oln = d.type !== 'added' ? sOldLn : '';
      const nln = d.type !== 'removed' ? sNewLn : '';
      const lns = `<span class="diff-ln diff-ln-old">${oln}</span><span class="diff-ln diff-ln-new">${nln}</span>`;
      if (d.type === 'same') {
        html += `<div class="diff-line diff-same">${lns}${_escapeForDiff(d.line)}</div>`;
      } else if (d.type === 'added') {
        html += `<div class="diff-line diff-added">${lns}<span class="diff-prefix">+</span>${_escapeForDiff(d.line)}</div>`;
      } else if (d.type === 'removed') {
        html += `<div class="diff-line diff-removed">${lns}<span class="diff-prefix">-</span>${_escapeForDiff(d.line)}</div>`;
      }
      if (d.type !== 'added') sOldLn++;
      if (d.type !== 'removed') sNewLn++;
    }

    // Show remaining original lines as dim (not yet generated)
    const newLineCount = currentBuffer.split('\n').length;
    if (newLineCount < originalLines.length) {
      for (let i = newLineCount; i < originalLines.length; i++) {
        html += `<div class="diff-line diff-pending">${_escapeForDiff(originalLines[i])}</div>`;
      }
    }

    html += '<div class="diff-cursor">\u2588</div>';
    wrapper.innerHTML = html;
    wrapper.scrollTop = wrapper.scrollHeight;
  }

  function scheduleRender() {
    if (rafId) return;
    const now = Date.now();
    const delay = Math.max(0, RENDER_INTERVAL - (now - lastRenderTime));
    rafId = setTimeout(() => {
      rafId = null;
      lastRenderTime = Date.now();
      requestAnimationFrame(render);
    }, delay);
  }

  return {
    update(buffer) {
      currentBuffer = buffer;
      dirty = true;
      scheduleRender();
    },
    destroy() {
      if (rafId) clearTimeout(rafId);
      wrapper.remove();
      container.hidden = false;
    },
    getElement() { return wrapper; }
  };
}

// ---------------------------------------------------------------------------
// Multi-Block Response Parsing
// ---------------------------------------------------------------------------

/**
 * Parses a multi-block LLM response into per-block patch sets.
 * Splits output by === FILE: <label> === markers, runs applyDiffPatches on each.
 * Returns Map<label, result> or null if no FILE markers found.
 */
export function parseMultiBlockResponse(diffOutput, blockLabels) {
  const sections = new Map();
  const sectionRe = /===\s*FILE:\s*(.+?)\s*===\s*\n([\s\S]*?)(?=\n===\s*FILE:|$)/gi;
  let m;

  while ((m = sectionRe.exec(diffOutput)) !== null) {
    sections.set(m[1].trim(), m[2].trim());
  }

  if (sections.size === 0) return null;

  // Extract change summary from end of full output ("//" or "#" style)
  let changeSummary = '';
  const changesRe = /(?:\/\/|#)\s*CHANGES?:\s*(.+)/i;
  const changesMatch = changesRe.exec(diffOutput);
  if (changesMatch) changeSummary = changesMatch[1].trim();

  const results = new Map();
  for (const { label, block } of blockLabels) {
    const sectionContent = sections.get(label);
    if (!sectionContent) continue;

    const patchResult = applyDiffPatches(block.code, sectionContent);
    results.set(label, {
      block,
      oldCode: block.code,
      newCode: patchResult.code,
      applied: patchResult.applied,
      failed: patchResult.failed,
      changeSummary,
      accepted: true,
    });
  }

  return results;
}

function _findCodeHeaderForBlock(anyHeader, block) {
  const msgEl = anyHeader.closest('[data-node-id]');
  if (!msgEl) return null;
  return msgEl.querySelector(`.code-header[data-block-id="${CSS.escape(block.id)}"]`);
}

// ---------------------------------------------------------------------------
// Quick Actions
// ---------------------------------------------------------------------------

const _QUICK_ACTIONS_CATEGORIES = [
  {
    name: 'Understand',
    icon: '\uD83D\uDCA1',
    actions: [
      { label: 'Explain code', instruction: 'Add a detailed block comment at the top explaining what this code does, its inputs, outputs, key logic, and any edge cases. Use clear, concise language a junior developer would understand.', mode: 'full' },
      { label: 'Review code', instruction: 'Review this code for bugs, performance issues, security vulnerabilities, and style problems. Add a structured comment at the top with sections: BUGS, PERFORMANCE, SECURITY, STYLE. List each finding with severity (critical/warning/info). Do not fix anything \u2014 only document.', mode: 'full' },
    ],
  },
  {
    name: 'Improve',
    icon: '\u2728',
    actions: [
      { label: 'Fix bugs', instruction: 'Fix any bugs in this code. Preserve the intended behavior. Add a brief comment above each fix explaining what was wrong.', mode: 'diff' },
      { label: 'Optimize', instruction: 'Optimize this code for performance and readability. Reduce complexity, eliminate redundancy, use more efficient patterns. Preserve the existing behavior. Add brief comments for non-obvious optimizations.', mode: 'diff' },
      { label: 'Add error handling', instruction: 'Add comprehensive error handling: try/catch blocks, input validation, null checks, edge case guards, and meaningful error messages. Preserve existing behavior for valid inputs.', mode: 'diff' },
      { label: 'Simplify', instruction: 'Simplify this code. Reduce nesting, extract complex conditions into named variables, replace verbose patterns with concise alternatives. The code should be easier to read without changing behavior.', mode: 'diff' },
    ],
  },
  {
    name: 'Document',
    icon: '\uD83D\uDCDD',
    actions: [
      { label: 'Add comments', instruction: 'Add clear, concise inline comments explaining the logic at key decision points. Comment the "why", not the "what". Do not change any code behavior.', mode: 'diff' },
      { label: 'Generate docs', instruction: 'Add comprehensive documentation: JSDoc/docstring for every function (params, returns, throws, examples), a module-level description, and inline comments for complex logic. Use the appropriate doc format for the language.', mode: 'diff' },
    ],
  },
  {
    name: 'Generate',
    icon: '\uD83D\uDD27',
    actions: [
      { label: 'Generate tests', instruction: 'Write a comprehensive test suite for this code. Include: unit tests for each function, edge case tests, error handling tests, and integration tests where relevant. Use the standard test framework for the language (pytest for Python, describe/it for JS/TS, testing for Go). Output the complete test file.', mode: 'full' },
      { label: 'Extract function', instruction: 'Extract the selected code (or the most complex logic block) into a well-named helper function. Add the function definition above the current code and replace the original with a call to it. Choose a descriptive name that explains the purpose.', mode: 'diff' },
      { label: 'Add logging', instruction: 'Add structured logging at key points: function entry/exit with parameters, error conditions, important state changes, and performance-sensitive operations. Use console.log for JS, logging module for Python, or the appropriate logging pattern for the language.', mode: 'diff' },
    ],
  },
  {
    name: 'Transform',
    icon: '\uD83D\uDD04',
    actions: [
      { label: 'Make responsive', instruction: 'Add responsive design: media queries for mobile/tablet/desktop breakpoints, flexible layouts (flexbox/grid), relative units (rem, %, vh/vw), touch-friendly tap targets (min 44px), and fluid typography. Ensure the design works from 320px to 1440px+ width.', mode: 'diff' },
      { label: 'Add dark mode', instruction: 'Add dark mode support using CSS custom properties. Define a color system with --bg, --text, --accent, --border variables. Add a .dark-theme class (or prefers-color-scheme media query) that switches to dark values. Ensure sufficient contrast ratios (WCAG AA).', mode: 'diff' },
      { label: 'Add loading states', instruction: 'Add loading and error state handling for all async operations (fetch calls, API requests, data loading). Include: loading indicators (spinners/skeletons), error messages with retry buttons, empty states, and disabled buttons during submission. Show optimistic UI where appropriate.', mode: 'diff' },
      { label: 'Async/await', instruction: 'Convert all callback-based and Promise.then() patterns to async/await syntax. Add proper try/catch error handling around each await. Preserve the exact same behavior and error semantics.', mode: 'diff' },
    ],
  },
];

const _QUICK_ACTION_LANG_EXTRAS = {
  python: [
    { label: 'Add type hints', instruction: 'Add type hints to all function parameters, return types, and class attributes. Use modern Python typing (3.10+ syntax: X | Y instead of Union[X,Y], list[X] instead of List[X]). Do not change any behavior.', mode: 'diff', category: 'Document' },
    { label: 'Add dataclasses', instruction: 'Convert plain classes or dicts-as-records to @dataclass (or @dataclass(frozen=True) where appropriate). Add type annotations, __post_init__ validation where needed. Preserve behavior.', mode: 'diff', category: 'Transform' },
  ],
  py: 'python',
  javascript: [
    { label: 'Convert to TypeScript', instruction: 'Convert this JavaScript to TypeScript. Add type annotations for all variables, function parameters, return types, and interfaces for objects. Use strict types (no `any`). Output as a .ts file.', mode: 'full', category: 'Transform' },
    { label: 'Add JSDoc', instruction: 'Add JSDoc documentation to every function and class. Include @param, @returns, @throws, @example tags. Use TypeScript-style type annotations in JSDoc ({string}, {number}, {Object}).', mode: 'diff', category: 'Document' },
  ],
  js: 'javascript',
  html: [
    { label: 'Add accessibility', instruction: 'Add accessibility improvements: ARIA labels/roles, alt text for images, semantic HTML elements (nav, main, article, aside), keyboard navigation (tabindex, focus styles), skip-to-content link, and form labels. Do not change visual appearance.', mode: 'diff', category: 'Improve' },
    { label: 'Add SEO meta', instruction: 'Add SEO meta tags: title, description, og:title, og:description, og:image, twitter:card, canonical URL, viewport, charset, lang attribute. Add structured data (JSON-LD) if appropriate for the content.', mode: 'diff', category: 'Document' },
  ],
  htm: 'html',
  css: [
    { label: 'Add animations', instruction: 'Add subtle, professional CSS animations and transitions: hover effects, focus states, entrance animations, micro-interactions. Use @keyframes for complex animations, transition for simple state changes. Keep durations short (150-300ms) and use ease-out curves. Respect prefers-reduced-motion.', mode: 'diff', category: 'Transform' },
    { label: 'Use CSS Grid/Flex', instruction: 'Refactor the layout to use modern CSS Grid and Flexbox. Replace floats, absolute positioning hacks, and table layouts. Use grid-template-areas for page layout, flex for component-level alignment. Add gap instead of margin hacks.', mode: 'diff', category: 'Transform' },
  ],
  scss: 'css',
  typescript: [
    { label: 'Strict types', instruction: 'Strengthen TypeScript types: replace `any` with proper types, add discriminated unions where appropriate, use branded types for IDs, add readonly where values should not be mutated, and ensure no implicit any. Add utility types (Partial, Required, Pick, Omit) where they simplify.', mode: 'diff', category: 'Improve' },
  ],
  ts: 'typescript',
  go: [
    { label: 'Add error wrapping', instruction: 'Improve error handling: wrap errors with fmt.Errorf("context: %w", err) for stack context, add sentinel errors where appropriate, use errors.Is/errors.As for checking. Preserve existing behavior.', mode: 'diff', category: 'Improve' },
  ],
  rust: [
    { label: 'Use Result/Option', instruction: 'Replace unwrap() and expect() calls with proper Result/Option handling using ? operator, map, and_then, unwrap_or_else. Add custom error types where appropriate. Make the code production-safe.', mode: 'diff', category: 'Improve' },
  ],
};

const _PORT_LANGUAGES = ['Python', 'JavaScript', 'TypeScript', 'Go', 'Rust', 'Java', 'C++', 'C#', 'Ruby', 'PHP', 'Swift', 'Kotlin'];

export function showQuickActionsMenu(codeHeader) {
  // Remove any existing menu
  document.querySelector('.code-quick-actions-menu')?.remove();

  const lang = (codeHeader.dataset.lang || '').toLowerCase();

  // Resolve language aliases for extras lookup
  function _resolveLangExtras(key) {
    const val = _QUICK_ACTION_LANG_EXTRAS[key];
    if (typeof val === 'string') return _resolveLangExtras(val);
    return val || [];
  }
  const langExtras = _resolveLangExtras(lang);

  // Build categorized menu
  let menuHtml = '';
  for (const cat of _QUICK_ACTIONS_CATEGORIES) {
    // Collect category actions + any language extras for this category
    const catActions = [...cat.actions];
    for (const ex of langExtras) {
      if (ex.category === cat.name) catActions.push(ex);
    }
    if (catActions.length === 0) continue;

    menuHtml += `<div class="code-quick-action-group">`;
    menuHtml += `<div class="code-quick-action-header">${cat.icon} ${escapeHtml(cat.name)}</div>`;
    menuHtml += catActions.map(a =>
      `<div class="code-quick-action-item" data-instruction="${escapeHtml(a.instruction)}" data-mode="${a.mode}">${escapeHtml(a.label)}</div>`
    ).join('');
    menuHtml += `</div>`;
  }

  // Language extras not in any category (standalone)
  const uncategorized = langExtras.filter(ex => !ex.category);
  if (uncategorized.length > 0) {
    menuHtml += `<div class="code-quick-action-group">`;
    menuHtml += `<div class="code-quick-action-header">\uD83D\uDD24 ${escapeHtml(lang.toUpperCase())}</div>`;
    menuHtml += uncategorized.map(a =>
      `<div class="code-quick-action-item" data-instruction="${escapeHtml(a.instruction)}" data-mode="${a.mode}">${escapeHtml(a.label)}</div>`
    ).join('');
    menuHtml += `</div>`;
  }

  // Port submenu
  menuHtml += `<div class="code-quick-action-group">`;
  menuHtml += `<div class="code-quick-action-item code-quick-action-port">\uD83C\uDF10 Port to \u25B8
    <div class="code-quick-actions-submenu">
      ${_PORT_LANGUAGES.map(l =>
        `<div class="code-quick-action-item" data-instruction="${escapeHtml('Rewrite this code in ' + l + '. Preserve the exact same behavior and logic. Use idiomatic patterns for ' + l + '.')}" data-mode="full">${escapeHtml(l)}</div>`
      ).join('')}
    </div>
  </div>`;
  menuHtml += `</div>`;

  const menu = document.createElement('div');
  menu.className = 'code-quick-actions-menu';
  menu.innerHTML = menuHtml;

  // Position below the trigger button
  const trigger = codeHeader.querySelector('[data-action="quick-actions"]');
  if (trigger) trigger.after(menu);
  else codeHeader.appendChild(menu);

  // Handle item clicks
  menu.addEventListener('click', (e) => {
    const item = e.target.closest('.code-quick-action-item[data-instruction]');
    if (!item) return;
    e.stopPropagation();
    const instruction = item.dataset.instruction;
    const useDiff = item.dataset.mode === 'diff';
    menu.remove();
    document.removeEventListener('click', dismiss);
    document.removeEventListener('keydown', dismissKey);
    executeAiEdit(codeHeader, instruction, useDiff, false, { skipPlan: true });
  });

  // Dismiss on outside click or Escape
  const dismiss = (e) => {
    if (!menu.contains(e.target) && e.target !== trigger) {
      menu.remove();
      document.removeEventListener('click', dismiss);
      document.removeEventListener('keydown', dismissKey);
    }
  };
  const dismissKey = (e) => {
    if (e.key === 'Escape') {
      menu.remove();
      document.removeEventListener('click', dismiss);
      document.removeEventListener('keydown', dismissKey);
    }
  };
  setTimeout(() => {
    document.addEventListener('click', dismiss);
    document.addEventListener('keydown', dismissKey);
  }, 0);
}

// ---------------------------------------------------------------------------
// Ask AI Edit
// ---------------------------------------------------------------------------

export function showAskAiPrompt(codeHeader) {
  const pre = codeHeader.nextElementSibling;
  if (!pre) return;

  const existing = codeHeader.nextElementSibling;
  if (existing?.classList.contains('code-ask-ai-bar')) {
    existing.remove();
    codeHeader.querySelector('[data-action="ask-ai-edit"]')?.classList.remove('active');
    return;
  }

  codeHeader.querySelector('[data-action="ask-ai-edit"]')?.classList.add('active');

  // Detect text selection within the code block
  let selectionRange = null;
  const sel = window.getSelection();
  if (sel && sel.rangeCount > 0 && !sel.isCollapsed) {
    const codePre = codeHeader.nextElementSibling;
    const codeEl = codePre?.querySelector('code');
    if (codeEl && codeEl.contains(sel.anchorNode) && codeEl.contains(sel.focusNode)) {
      const fullText = codeEl.textContent || '';
      const range = sel.getRangeAt(0);
      const preRange = document.createRange();
      preRange.selectNodeContents(codeEl);
      preRange.setEnd(range.startContainer, range.startOffset);
      const startOffset = preRange.toString().length;
      const endOffset = startOffset + range.toString().length;

      const startLine = fullText.slice(0, startOffset).split('\n').length;
      const endLine = fullText.slice(0, endOffset).split('\n').length;
      const totalLines = fullText.split('\n').length;

      if (endLine > startLine || range.toString().trim().length > 0) {
        selectionRange = { startLine, endLine, totalLines };
      }
    }
  }

  const allBlocks = getBlocksForMessage(codeHeader);
  const hasMultipleBlocks = allBlocks.length > 1;

  const scopeToggleHtml = hasMultipleBlocks
    ? `<label class="code-ask-ai-scope" title="Edit all code blocks in this message together">
         <input type="checkbox" class="code-ask-ai-scope-toggle" checked>
         <span class="code-ask-ai-scope-label">All blocks</span>
       </label>`
    : '';

  const bar = document.createElement('div');
  bar.className = 'code-ask-ai-bar';
  bar.innerHTML = `
    <input class="code-ask-ai-input" type="text" placeholder="What should I change?" autofocus>
    <label class="code-ask-ai-mode" title="Diff mode sends less tokens (best for cloud APIs). Full mode is more reliable for small local models.">
      <input type="checkbox" class="code-ask-ai-diff-toggle">
      <span class="code-ask-ai-mode-label">Full</span>
    </label>
    ${scopeToggleHtml}
    <button class="code-ask-ai-submit" title="Send">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
    </button>
    <button class="code-ask-ai-cancel" title="Cancel">&times;</button>
  `;

  const input = bar.querySelector('.code-ask-ai-input');
  const submitBtn = bar.querySelector('.code-ask-ai-submit');
  const cancelBtn = bar.querySelector('.code-ask-ai-cancel');
  const diffToggle = bar.querySelector('.code-ask-ai-diff-toggle');
  const modeLabel = bar.querySelector('.code-ask-ai-mode-label');

  diffToggle.addEventListener('change', () => {
    modeLabel.textContent = diffToggle.checked ? 'Diff' : 'Full';
  });

  const scopeToggle = bar.querySelector('.code-ask-ai-scope-toggle');
  const scopeLabel = bar.querySelector('.code-ask-ai-scope-label');
  if (scopeToggle) {
    scopeToggle.addEventListener('change', () => {
      scopeLabel.textContent = scopeToggle.checked ? 'All blocks' : 'This block';
    });
  }

  const submit = () => {
    const instruction = input.value.trim();
    if (!instruction) return;
    const useDiff = diffToggle.checked;
    const useMultiBlock = scopeToggle?.checked || false;
    const selStart = bar.dataset.selStart;
    const selEnd = bar.dataset.selEnd;
    // Show brief loading state before removing bar
    submitBtn.classList.add('loading');
    input.disabled = true;
    input.style.opacity = '0.5';
    requestAnimationFrame(() => {
      bar.remove();
      codeHeader.querySelector('[data-action="ask-ai-edit"]')?.classList.remove('active');
    });

    let finalInstruction = instruction;
    if (selStart && selEnd) {
      finalInstruction = `${instruction}\n\nIMPORTANT: Only modify lines ${selStart}-${selEnd}. Do not change any code outside this range. Include context lines from the original in your SEARCH blocks.`;
    }
    executeAiEdit(codeHeader, finalInstruction, useDiff, useMultiBlock);
  };

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); submit(); }
    if (e.key === 'Escape') { bar.remove(); codeHeader.querySelector('[data-action="ask-ai-edit"]')?.classList.remove('active'); }
  });
  submitBtn.addEventListener('click', submit);
  cancelBtn.addEventListener('click', () => {
    bar.remove();
    codeHeader.querySelector('[data-action="ask-ai-edit"]')?.classList.remove('active');
  });

  codeHeader.after(bar);
  input.focus();

  if (selectionRange) {
    const scopeInfo = document.createElement('div');
    scopeInfo.className = 'code-ask-ai-scope-info';
    scopeInfo.textContent = `Editing lines ${selectionRange.startLine}\u2013${selectionRange.endLine} of ${selectionRange.totalLines}`;
    bar.appendChild(scopeInfo);
    bar.dataset.selStart = selectionRange.startLine;
    bar.dataset.selEnd = selectionRange.endLine;
  }
}

async function _getEditPlan(model, lang, rawCode, instruction) {
  const resp = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Augmentum-Mode': 'passthrough',
      'X-Augmentum-Tools': 'none',
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: 'You are a code review assistant. The user will show you code and an edit instruction. Describe what changes you would make in 2-4 bullet points. Be specific about which parts of the code you will modify. Do NOT output any code \u2014 only describe the plan.' },
        { role: 'user', content: `Here is ${lang || 'code'} (${rawCode.split('\n').length} lines):\n\n${rawCode}\n\n---\nEdit instruction: ${instruction}` },
      ],
      stream: false,
      think: false,
      options: { num_predict: 256 },
    }),
  });
  if (!resp.ok) return null;
  const data = await resp.json();
  return data?.message?.content || null;
}

export async function executeAiEdit(codeHeader, instruction, useDiff = false, useMultiBlock = false, { skipPlan = false } = {}) {
  const rawCode = decodeURIComponent(codeHeader.dataset.rawCode || '');
  const lang = codeHeader.dataset.lang || '';
  if (!rawCode) return;

  // Guard against concurrent edits on the same block
  if (_aiEditInProgress.has(codeHeader)) {
    showToast('Already editing this block.', 'info');
    return;
  }
  _aiEditInProgress.add(codeHeader);

  const askBtn = codeHeader.querySelector('[data-action="ask-ai-edit"]');
  const pre = codeHeader.nextElementSibling?.tagName === 'PRE' ? codeHeader.nextElementSibling : codeHeader.nextElementSibling?.nextElementSibling;
  const codeEl = pre?.querySelector('code');
  const model = app.state.currentModel || 'default';

  // Show prominent progress indicator
  const label = skipPlan ? (useDiff ? 'Generating diff...' : 'Editing...') : 'Planning edit...';
  if (askBtn) { askBtn.classList.add('active'); askBtn.textContent = label; askBtn.disabled = true; }
  let statusBar = null;
  if (pre) {
    statusBar = document.createElement('div');
    statusBar.className = 'code-edit-progress';
    statusBar.innerHTML = `
      <div class="code-edit-progress-bar"><div class="code-edit-progress-fill" style="width:${skipPlan ? '5' : '15'}%"></div></div>
      <span class="code-edit-progress-text">${label}</span>
    `;
    pre.before(statusBar);
  }

  let plan = null;
  if (!skipPlan) {
    try { plan = await _getEditPlan(model, lang, rawCode, instruction); } catch { plan = null; }
  }

  if (!plan) {
    if (statusBar) statusBar.remove();
    if (askBtn) { askBtn.textContent = useDiff ? 'Diffing...' : 'Editing...'; }
    try {
      await _executeAiEditPhase2(codeHeader, instruction, useDiff, rawCode, lang, model, askBtn, pre, codeEl, useMultiBlock);
    } finally {
      _aiEditInProgress.delete(codeHeader);
    }
    return;
  }

  if (statusBar) statusBar.remove();
  if (askBtn) { askBtn.classList.remove('active'); askBtn.textContent = 'Ask AI'; askBtn.disabled = false; }

  const planBar = document.createElement('div');
  planBar.className = 'code-edit-plan-bar';
  planBar.innerHTML = `
    <div class="code-edit-plan-content">${escapeHtml(plan).replace(/\n/g, '<br>')}</div>
    <div class="code-edit-plan-actions">
      <button class="code-edit-plan-approve">Apply</button>
      <button class="code-edit-plan-cancel">Cancel</button>
    </div>
  `;

  if (pre) pre.before(planBar);
  else codeHeader.after(planBar);

  const userDecision = await new Promise(resolve => {
    planBar.querySelector('.code-edit-plan-approve').addEventListener('click', () => resolve(true));
    planBar.querySelector('.code-edit-plan-cancel').addEventListener('click', () => resolve(false));
  });

  planBar.remove();
  if (!userDecision) {
    _aiEditInProgress.delete(codeHeader);
    return;
  }

  if (askBtn) { askBtn.classList.add('active'); askBtn.textContent = useDiff ? 'Diffing...' : 'Editing...'; }
  try {
    await _executeAiEditPhase2(codeHeader, instruction, useDiff, rawCode, lang, model, askBtn, pre, codeEl, useMultiBlock);
  } finally {
    _aiEditInProgress.delete(codeHeader);
  }
}

async function _executeAiEditPhase2(codeHeader, instruction, useDiff, rawCode, lang, model, askBtn, pre, codeEl, useMultiBlock = false) {
  const systemFull = `You are a code editor. The user will give you existing code and an edit instruction. Output the complete updated code, preserving the original structure and only changing what the instruction asks for. Output ONLY the raw code — no explanations, no commentary before or after, no markdown fences. Your entire response is pasted directly into the file.\n\nAfter the code, on the very last line, write a brief change summary prefixed with "// CHANGES: " (e.g. // CHANGES: Added dark theme, fixed button alignment). This summary line must be the last line of your output.`;

  const systemDiff = `You are a code editor. The user will give you code and an edit instruction. Output ONLY the changed sections using this format:\n\n<<<<<<< SEARCH\nexact lines to find in the original code\n=======\nreplacement lines\n>>>>>>> REPLACE\n\nRules:\n- SEARCH content must match the original code EXACTLY \u2014 same whitespace, indentation, everything\n- Include 1-3 surrounding context lines in SEARCH to ensure a unique match\n- You may output multiple SEARCH/REPLACE blocks for different changes\n- To delete lines: leave the REPLACE section empty\n- To insert after a location: SEARCH the lines above, REPLACE with those lines plus new ones\n- Only output blocks that change \u2014 do not repeat the full file\n- After all blocks, write: // CHANGES: <brief summary>`;

  const systemMultiBlock = `You are a code editor. The user will give you multiple labeled code blocks and an edit instruction. Apply changes to whichever blocks need them.\n\nFor each block that needs changes, output:\n\n=== FILE: <label> ===\n<<<<<<< SEARCH\nexact lines to find\n=======\nreplacement lines\n>>>>>>> REPLACE\n\nRules:\n- Labels must match those provided in the user message exactly\n- SEARCH content must match the original code EXACTLY\n- Blocks that don't need changes should be omitted entirely\n- After all edits, write: // CHANGES: <brief summary>`;

  const startTime = Date.now();
  let tokenCount = 0;

  const progressBar = document.createElement('div');
  progressBar.className = 'code-edit-progress';
  progressBar.innerHTML = `
    <div class="code-edit-progress-bar"><div class="code-edit-progress-fill"></div></div>
    <span class="code-edit-progress-text">Starting...</span>
  `;
  if (pre) pre.before(progressBar);
  else codeHeader.after(progressBar);

  const progressFill = progressBar.querySelector('.code-edit-progress-fill');
  const progressText = progressBar.querySelector('.code-edit-progress-text');

  function updateProgress(tokens) {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const tokPerSec = tokens > 0 ? (tokens / (Date.now() - startTime) * 1000).toFixed(0) : 0;
    if (useDiff) {
      progressText.textContent = `Generating diff... ${tokens} tokens (${tokPerSec} tok/s, ${elapsed}s)`;
    } else {
      const estimatedTokens = Math.max(rawCode.length / 3, 500);
      const pct = Math.min(95, (tokens / estimatedTokens) * 100);
      progressFill.style.width = `${pct}%`;
      progressText.textContent = `Generating... ${tokens} tokens (${tokPerSec} tok/s, ${elapsed}s)`;
    }
  }

  let streamingDiff = null;

  try {
    let codeForPrompt;
    let blockLabels = null;

    if (useMultiBlock) {
      const blocks = getBlocksForMessage(codeHeader);
      blockLabels = [];
      const parts = [];
      const langCounts = {};

      for (const block of blocks) {
        const langKey = block.lang || 'code';
        langCounts[langKey] = (langCounts[langKey] || 0) + 1;
        const label = langCounts[langKey] > 1 ? `${langKey} (${langCounts[langKey]})` : langKey;
        blockLabels.push({ label, block });
        parts.push(`=== ${label} ===\n${block.code}`);
      }
      codeForPrompt = parts.join('\n\n');
    } else {
      codeForPrompt = rawCode;
    }

    const systemPrompt = useMultiBlock ? systemMultiBlock : (useDiff ? systemDiff : systemFull);

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Here is the current ${lang || 'code'}:\n\n${codeForPrompt}\n\n---\nEdit instruction: ${instruction}` },
    ];

    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Augmentum-Mode': 'passthrough',
        'X-Augmentum-Tools': 'none',
      },
      body: JSON.stringify({
        model, messages, stream: true,
        think: false,
        options: { num_predict: useDiff ? 4096 : 8192 },
      }),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    if (!useDiff && !useMultiBlock && pre) {
      streamingDiff = createStreamingDiff(rawCode, pre);
    } else if (useDiff && codeEl) {
      codeEl.textContent = '\u23F3 Generating diff...';
      codeEl.removeAttribute('data-highlighted');
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let newCode = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const lines = decoder.decode(value, { stream: true }).split('\n');
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const chunk = JSON.parse(line);
          const text = chunk?.message?.content || '';
          if (text) {
            newCode += text;
            tokenCount++;
            updateProgress(tokenCount);
            if (streamingDiff) {
              let preview = newCode;
              if (preview.startsWith('```')) preview = preview.replace(/^```\w*\n?/, '');
              streamingDiff.update(preview);
            }
          }
        } catch (parseErr) {
          if (line.trim() && !line.startsWith('data:')) {
            console.warn('Ask AI: unparseable chunk:', line.slice(0, 100));
          }
        }
      }
    }

    if (streamingDiff) {
      streamingDiff.destroy();
      streamingDiff = null;
    }

    progressFill.style.width = '100%';
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    progressText.textContent = useDiff
      ? `\u2699 Matching patches \u00B7 ${tokenCount} tokens \u00B7 ${elapsed}s`
      : `\u2699 Processing \u00B7 ${tokenCount} tokens \u00B7 ${elapsed}s`;

    if (!newCode.trim()) {
      throw new Error('Model returned empty response \u2014 try a different model or simplify the instruction');
    }

    newCode = newCode.trim();
    if (newCode.startsWith('```')) {
      newCode = newCode.replace(/^```\w*\n?/, '').replace(/\n?```\s*$/, '');
    }

    // Multi-block response handling
    if (useMultiBlock && blockLabels) {
      const multiResults = parseMultiBlockResponse(newCode, blockLabels);

      if (multiResults && multiResults.size > 0) {
        progressBar.remove();
        const accepted = await _showMultiBlockDiffReview(codeHeader, multiResults);
        if (accepted) {
          let lintCount = 0;
          for (const [label, result] of multiResults) {
            if (!result.accepted) continue;
            const header = _findCodeHeaderForBlock(codeHeader, result.block);
            if (header) {
              let finalCode = result.newCode;
              updateBlock(header, finalCode, result.changeSummary, true);

              const lintResult = silentLint(result.block.lang, finalCode);
              if (lintResult.fixed) {
                updateBlock(header, lintResult.code, null, false);
                lintCount += lintResult.changes.length;
              }

              const blk = getBlock(header);
              if (blk) showVersion(header, blk.versions.length - 1);
            }
          }
          if (lintCount > 0) {
            showToast(`Cleaned up ${lintCount} small syntax thing${lintCount > 1 ? 's' : ''}.`, 'info');
          }
        }
        if (askBtn) { askBtn.classList.remove('active'); askBtn.textContent = 'Ask AI'; askBtn.disabled = false; }
        return;
      }
      // If parsing failed, fall through to single-block flow
    }

    // Extract the model's change summary from its RAW output, before diff
    // patching — in diff mode the "// CHANGES:" line sits outside the
    // SEARCH/REPLACE blocks and never survives into the patched code, so
    // extracting afterwards always missed it. Fallback: first line of the
    // instruction, truncated (the full instruction used to put entire
    // error tracebacks into the diff header and version label).
    let changeSummary = instruction.split('\n')[0].slice(0, 120);
    const changesRe = /(?:\/\/|#)\s*CHANGES?:\s*(.+)/i;
    const codeLines = newCode.split('\n');
    for (let i = codeLines.length - 1; i >= Math.max(0, codeLines.length - 3); i--) {
      const match = changesRe.exec(codeLines[i]);
      if (match) {
        changeSummary = match[1].trim();
        codeLines[i] = codeLines[i].replace(changesRe, '').trimEnd();
        if (!codeLines[i].trim()) codeLines.splice(i, 1);
        newCode = codeLines.join('\n').trimEnd();
        break;
      }
    }

    let diffFailures = [];
    if (useDiff) {
      const diffResult = applyDiffPatches(rawCode, newCode);
      newCode = diffResult.code;
      diffFailures = diffResult.failed || [];
    }

    const accepted = (newCode.trim() === rawCode.trim())
      ? (showToast('Nothing to change.', 'info'), false)
      : await _showDiffReview(codeHeader, pre, rawCode, newCode, changeSummary, lang);

    if (accepted) {
      updateBlock(codeHeader, newCode, changeSummary, true);

      // Silent lint: catch LLM syntax mistakes
      const lintResult = silentLint(lang, newCode);
      if (lintResult.fixed) {
        updateBlock(codeHeader, lintResult.code, null, false);
        newCode = lintResult.code;
        showToast(`Cleaned up ${lintResult.changes.length} small syntax thing${lintResult.changes.length > 1 ? 's' : ''}.`, 'info');
      }

      const block = getBlock(codeHeader);
      if (block) showVersion(codeHeader, block.versions.length - 1);

      // Write directly into the captured element too — showVersion can miss
      // it (e.g. node not in tree yet), which left the diff-mode
      // "Generating diff..." placeholder stuck on screen.
      if (codeEl) {
        codeEl.textContent = newCode;
        codeEl.removeAttribute('data-highlighted');
        safeHighlightElement(codeEl);
      }

      if (lang === 'html' && document.getElementById('code-preview-modal')?.classList.contains('visible')) {
        toggleHtmlPreview(codeHeader);
      }

      if (diffFailures.length > 0) {
        _showDiffRetryBar(codeHeader, instruction, diffFailures, rawCode);
      }
    } else {
      if (codeEl) {
        codeEl.textContent = rawCode;
        codeEl.removeAttribute('data-highlighted');
        safeHighlightElement(codeEl);
      }
    }

  } catch (err) {
    if (streamingDiff) { streamingDiff.destroy(); streamingDiff = null; }
    // Diff mode replaced the visible code with a placeholder — restore it
    if (useDiff && codeEl) {
      codeEl.textContent = rawCode;
      codeEl.removeAttribute('data-highlighted');
      safeHighlightElement(codeEl);
    }
    showToast(`Couldn't edit — ${err.message}`, 'error');
  } finally {
    if (streamingDiff) { streamingDiff.destroy(); streamingDiff = null; }
    progressBar.remove();
    if (askBtn) { askBtn.classList.remove('active'); askBtn.textContent = 'Ask AI'; askBtn.disabled = false; }
  }
}

function _showDiffRetryBar(codeHeader, originalInstruction, failedPatches, originalCode) {
  const existing = codeHeader.parentElement.querySelector('.code-diff-retry-bar');
  if (existing) existing.remove();

  const failedCount = failedPatches.length;
  const failedPreview = failedPatches.map(p => p.search.split('\n')[0].trim().slice(0, 50)).join(', ');

  const bar = document.createElement('div');
  bar.className = 'code-diff-retry-bar';
  bar.innerHTML = `
    <div class="code-diff-retry-info">
      <span class="code-diff-retry-icon">&#9888;</span>
      <span>${failedCount} patch${failedCount > 1 ? 'es' : ''} failed to apply</span>
      <span class="code-diff-retry-detail" title="${escapeHtml(failedPreview)}">${escapeHtml(failedPreview.slice(0, 60))}${failedPreview.length > 60 ? '...' : ''}</span>
    </div>
    <div class="code-diff-retry-actions">
      <button class="code-diff-retry-btn" title="Retry with error context">Retry</button>
      <button class="code-diff-retry-full-btn" title="Retry in Full mode">Retry Full</button>
      <button class="code-diff-retry-dismiss">&times;</button>
    </div>
  `;

  bar.querySelector('.code-diff-retry-btn').addEventListener('click', () => {
    bar.remove();
    const errorContext = failedPatches.map(p => `SEARCH block that failed to match:\n${p.search.slice(0, 200)}`).join('\n\n');
    executeAiEdit(codeHeader, `${originalInstruction}\n\nNOTE: Your previous edit had ${failedCount} SEARCH block(s) that didn't match the code. The blocks that failed:\n${errorContext}\n\nPlease check the original code carefully and output corrected SEARCH/REPLACE blocks.`, true);
  });
  bar.querySelector('.code-diff-retry-full-btn').addEventListener('click', () => {
    bar.remove();
    executeAiEdit(codeHeader, originalInstruction, false);
  });
  bar.querySelector('.code-diff-retry-dismiss').addEventListener('click', () => bar.remove());

  const pre = codeHeader.nextElementSibling?.tagName === 'PRE' ? codeHeader.nextElementSibling : codeHeader.nextElementSibling?.nextElementSibling;
  if (pre) pre.after(bar);
  else codeHeader.after(bar);
}

// ---------------------------------------------------------------------------
// Visual Diff Review
// ---------------------------------------------------------------------------

export function computeLineDiff(oldText, newText) {
  const oldLines = oldText.split('\n');
  const newLines = newText.split('\n');

  if (oldLines.length + newLines.length > 5000) {
    return [
      { type: 'removed', line: `(${oldLines.length} lines removed \u2014 diff too large to display)` },
      { type: 'added', line: `(${newLines.length} lines added \u2014 use version arrows to compare)` },
    ];
  }

  const diff = [];
  let oi = 0, ni = 0;
  while (oi < oldLines.length || ni < newLines.length) {
    if (oi < oldLines.length && ni < newLines.length && oldLines[oi] === newLines[ni]) {
      diff.push({ type: 'same', line: oldLines[oi] });
      oi++; ni++;
    } else {
      let foundOld = -1, foundNew = -1;
      const searchWindow = 10;
      for (let k = 1; k <= searchWindow; k++) {
        if (foundNew < 0 && ni + k < newLines.length && oldLines[oi] === newLines[ni + k]) foundNew = k;
        if (foundOld < 0 && oi + k < oldLines.length && oldLines[oi + k] === newLines[ni]) foundOld = k;
        if (foundOld >= 0 || foundNew >= 0) break;
      }

      if (foundOld >= 0 && (foundNew < 0 || foundOld <= foundNew)) {
        for (let k = 0; k < foundOld; k++) {
          if (ni < newLines.length) { diff.push({ type: 'added', line: newLines[ni] }); ni++; }
        }
      } else if (foundNew >= 0) {
        for (let k = 0; k < foundNew; k++) {
          if (oi < oldLines.length) { diff.push({ type: 'removed', line: oldLines[oi] }); oi++; }
        }
      } else {
        if (oi < oldLines.length) { diff.push({ type: 'removed', line: oldLines[oi] }); oi++; }
        if (ni < newLines.length) { diff.push({ type: 'added', line: newLines[ni] }); ni++; }
      }
    }
  }
  return diff;
}

/**
 * Compute intra-line character-level diff between two strings.
 * Returns HTML with <span class="diff-highlight"> around changed characters.
 * Uses common prefix/suffix detection (fast, no LCS needed).
 */
function _intraLineDiff(oldLine, newLine) {
  // Find common prefix
  let prefixLen = 0;
  const minLen = Math.min(oldLine.length, newLine.length);
  while (prefixLen < minLen && oldLine[prefixLen] === newLine[prefixLen]) prefixLen++;

  // Find common suffix (from end)
  let suffixLen = 0;
  while (suffixLen < (minLen - prefixLen) &&
    oldLine[oldLine.length - 1 - suffixLen] === newLine[newLine.length - 1 - suffixLen]) suffixLen++;

  const oldChanged = oldLine.slice(prefixLen, oldLine.length - suffixLen);
  const newChanged = newLine.slice(prefixLen, newLine.length - suffixLen);

  // Only highlight if the change is meaningful (not the entire line)
  if (prefixLen + suffixLen < 3 && oldChanged.length > oldLine.length * 0.7) {
    return { oldHtml: null, newHtml: null }; // too different, skip intra-line
  }

  const prefix = _escapeForDiff(oldLine.slice(0, prefixLen));
  const suffix = _escapeForDiff(oldLine.slice(oldLine.length - suffixLen));

  return {
    oldHtml: oldChanged.length > 0
      ? `${prefix}<span class="diff-highlight">${_escapeForDiff(oldChanged)}</span>${suffix}`
      : `${prefix}${suffix}`,
    newHtml: newChanged.length > 0
      ? `${_escapeForDiff(newLine.slice(0, prefixLen))}<span class="diff-highlight">${_escapeForDiff(newChanged)}</span>${_escapeForDiff(newLine.slice(newLine.length - suffixLen))}`
      : `${_escapeForDiff(newLine.slice(0, prefixLen))}${_escapeForDiff(newLine.slice(newLine.length - suffixLen))}`,
  };
}

export function renderDiffLines(diff) {
  const diffLines = [];
  let lastShown = -1;

  // Pre-compute intra-line diffs for paired remove->add lines
  for (let i = 0; i < diff.length - 1; i++) {
    if (diff[i].type === 'removed' && diff[i + 1].type === 'added') {
      const result = _intraLineDiff(diff[i].line, diff[i + 1].line);
      if (result.oldHtml) {
        diff[i]._intraHtml = result.oldHtml;
        diff[i + 1]._intraHtml = result.newHtml;
      }
    }
  }

  // Pre-compute line numbers: old (left) and new (right)
  let oldLn = 1, newLn = 1;
  for (const d of diff) {
    d._oldLn = d.type !== 'added' ? oldLn : null;
    d._newLn = d.type !== 'removed' ? newLn : null;
    if (d.type !== 'added') oldLn++;
    if (d.type !== 'removed') newLn++;
  }

  function _lineHtml(d, cls, prefix) {
    const oln = d._oldLn != null ? d._oldLn : '';
    const nln = d._newLn != null ? d._newLn : '';
    // Use intra-line highlight HTML if available, otherwise escape the full line
    const content = d._intraHtml || _escapeForDiff(d.line);
    return `<div class="diff-line ${cls}"><span class="diff-ln diff-ln-old">${oln}</span><span class="diff-ln diff-ln-new">${nln}</span><span class="diff-prefix">${prefix}</span>${content}</div>`;
  }

  for (let i = 0; i < diff.length; i++) {
    const d = diff[i];
    if (d.type !== 'same') {
      const contextStart = Math.max(lastShown + 1, i - 2);
      if (contextStart > lastShown + 1 && lastShown >= 0) {
        diffLines.push('<div class="diff-separator">\u00B7\u00B7\u00B7</div>');
      }
      for (let j = contextStart; j < i; j++) {
        if (diff[j].type === 'same') {
          diffLines.push(_lineHtml(diff[j], 'diff-same', ' '));
        }
      }
      const cls = d.type === 'added' ? 'diff-added' : 'diff-removed';
      const prefix = d.type === 'added' ? '+' : '-';
      diffLines.push(_lineHtml(d, cls, prefix));
      lastShown = i;

      for (let j = i + 1; j <= Math.min(i + 2, diff.length - 1); j++) {
        if (diff[j].type === 'same') {
          diffLines.push(_lineHtml(diff[j], 'diff-same', ' '));
          lastShown = j;
        } else break;
      }
    }
  }

  if (diffLines.length === 0) {
    diffLines.push('<div class="diff-line diff-same" style="opacity:0.5">No visible changes</div>');
  }
  return diffLines.join('');
}

function _showDiffReview(codeHeader, pre, oldCode, newCode, changeSummary, lang, options = {}) {
  const diff = computeLineDiff(oldCode, newCode);
  const added = diff.filter(d => d.type === 'added').length;
  const removed = diff.filter(d => d.type === 'removed').length;
  const unchanged = diff.filter(d => d.type === 'same').length;

  const diffLinesHtml = renderDiffLines(diff);

  const panel = document.createElement('div');
  panel.className = 'code-diff-review';
  panel.innerHTML = `
    <div class="code-diff-review-header">
      <span class="code-diff-review-stats">
        <span class="diff-stat-added">+${added} line${added !== 1 ? 's' : ''}</span>
        <span class="diff-stat-removed">\u2212${removed} line${removed !== 1 ? 's' : ''}</span>
        <span class="diff-stat-unchanged">${unchanged} unchanged</span>
      </span>
      <span class="code-diff-review-summary">${escapeHtml(changeSummary || '')}</span>
    </div>
    <div class="code-diff-review-body">${diffLinesHtml}</div>
    <div class="code-diff-review-actions">
      ${lang === 'html' ? '<button class="code-diff-review-preview">Preview</button>' : ''}
      <button class="code-diff-review-accept">${escapeHtml(options.acceptLabel || 'Accept')} <kbd>Enter</kbd></button>
      <button class="code-diff-review-reject">Reject <kbd>Esc</kbd></button>
    </div>
  `;

  if (pre) pre.before(panel);
  else codeHeader.after(panel);

  if (pre) pre.hidden = true;

  return new Promise(resolve => {
    const previewBtn = panel.querySelector('.code-diff-review-preview');
    if (previewBtn) {
      previewBtn.addEventListener('click', () => {
        const originalRaw = codeHeader.dataset.rawCode;
        codeHeader.dataset.rawCode = encodeURIComponent(newCode);
        toggleHtmlPreview(codeHeader);
        codeHeader.dataset.rawCode = originalRaw;
      });
    }

    const _accept = () => { closePreviewModal(); panel.remove(); if (pre) pre.hidden = false; document.removeEventListener('keydown', _kbHandler); resolve(true); };
    const _reject = () => { closePreviewModal(); panel.remove(); if (pre) pre.hidden = false; document.removeEventListener('keydown', _kbHandler); resolve(false); };

    panel.querySelector('.code-diff-review-accept').addEventListener('click', _accept);
    panel.querySelector('.code-diff-review-reject').addEventListener('click', _reject);

    // Keyboard: Enter = Accept, Escape = Reject (when diff panel is visible)
    const _kbHandler = (e) => {
      // Don't handle if user is typing in an input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); _accept(); }
      if (e.key === 'Escape') { e.preventDefault(); _reject(); }
    };
    document.addEventListener('keydown', _kbHandler);
  });
}

async function _showMultiBlockDiffReview(codeHeader, multiResults) {
  const pre = codeHeader.nextElementSibling?.tagName === 'PRE'
    ? codeHeader.nextElementSibling
    : codeHeader.nextElementSibling?.nextElementSibling;

  const panel = document.createElement('div');
  panel.className = 'code-diff-review code-diff-review-multi';

  let tabsHtml = '';
  let sectionsHtml = '';
  let blockIdx = 0;

  for (const [label, result] of multiResults) {
    const diff = computeLineDiff(result.oldCode, result.newCode);
    const added = diff.filter(d => d.type === 'added').length;
    const removed = diff.filter(d => d.type === 'removed').length;
    const safeLabel = escapeHtml(label);
    const diffLinesHtml = renderDiffLines(diff);

    tabsHtml += `<label class="code-diff-multi-tab">
      <input type="checkbox" data-idx="${blockIdx}" checked>
      <span>${safeLabel}</span>
      <span class="diff-stat-added">+${added}</span>
      <span class="diff-stat-removed">-${removed}</span>
    </label>`;

    sectionsHtml += `<div class="code-diff-multi-section" data-idx="${blockIdx}">
      <div class="code-diff-multi-section-header">${safeLabel}</div>
      <div class="code-diff-review-body">${diffLinesHtml}</div>
    </div>`;
    blockIdx++;
  }

  const resultsArr = [...multiResults.values()];

  panel.innerHTML = `
    <div class="code-diff-review-header">
      <span class="code-diff-review-summary">${escapeHtml(resultsArr[0]?.changeSummary || '')}</span>
    </div>
    <div class="code-diff-multi-tabs">${tabsHtml}</div>
    <div class="code-diff-multi-sections">${sectionsHtml}</div>
    <div class="code-diff-review-actions">
      <button class="code-diff-review-accept">Accept Selected</button>
      <button class="code-diff-review-reject">Reject All</button>
    </div>
  `;

  if (pre) pre.before(panel);
  else codeHeader.after(panel);
  if (pre) pre.hidden = true;

  panel.querySelectorAll('.code-diff-multi-tab input').forEach(cb => {
    cb.addEventListener('change', () => {
      const idx = parseInt(cb.dataset.idx, 10);
      const result = resultsArr[idx];
      if (result) result.accepted = cb.checked;
      const section = panel.querySelector(`.code-diff-multi-section[data-idx="${idx}"]`);
      if (section) section.style.opacity = cb.checked ? '1' : '0.4';
    });
  });

  return new Promise(resolve => {
    panel.querySelector('.code-diff-review-accept').addEventListener('click', () => {
      panel.remove();
      if (pre) pre.hidden = false;
      resolve(true);
    });
    panel.querySelector('.code-diff-review-reject').addEventListener('click', () => {
      panel.remove();
      if (pre) pre.hidden = false;
      resolve(false);
    });
  });
}

function _escapeForDiff(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\t/g, '  ')
    || '&nbsp;';
}

// ---------------------------------------------------------------------------
// Auto-Fix (no LLM) — Deterministic per-language syntax fixers
// ---------------------------------------------------------------------------

async function _loadJsFixer() {
  if (_acornLoose && _astring) return true;
  if (_jsFixerLoading) {
    while (_jsFixerLoading) await new Promise(r => setTimeout(r, 50));
    return !!(_acornLoose && _astring);
  }
  _jsFixerLoading = true;
  try {
    const [acornMod, astringMod] = await Promise.all([
      import('https://cdn.jsdelivr.net/npm/acorn-loose@8/+esm'),
      import('https://cdn.jsdelivr.net/npm/astring@1/+esm'),
    ]);
    _acornLoose = acornMod;
    _astring = astringMod;
    return true;
  } catch (err) {
    console.warn('Failed to load JS fixer libraries:', err);
    return false;
  } finally {
    _jsFixerLoading = false;
  }
}

async function _fixJavaScript(code) {
  const loaded = await _loadJsFixer();
  if (!loaded) return { code, fixed: false, changes: ['JS fixer unavailable \u2014 CDN may be unreachable'] };

  const changes = [];
  try {
    const ast = _acornLoose.parse(code, { ecmaVersion: 2022, sourceType: 'module' });
    const fixed = _astring.generate(ast);

    try {
      const { parse } = await import('https://cdn.jsdelivr.net/npm/acorn@8/+esm');
      parse(fixed, { ecmaVersion: 2022, sourceType: 'module' });
    } catch (validationErr) {
      changes.push(`Fix produced invalid JS: ${validationErr.message} \u2014 keeping original`);
      return { code, fixed: false, changes };
    }

    if (fixed !== code) {
      changes.push('Repaired syntax structure via tolerant parsing');
      const oldLines = code.split('\n').length;
      const newLines = fixed.split('\n').length;
      if (oldLines !== newLines) changes.push(`Line count: ${oldLines} \u2192 ${newLines}`);
    }

    return { code: fixed, fixed: fixed !== code, changes };
  } catch (err) {
    changes.push(`Parse failed: ${err.message}`);
    return { code, fixed: false, changes };
  }
}

export function fixHTML(code) {
  const changes = [];
  try {
    let fixed = code;

    // Fix missing quotes on attributes: class=foo -> class="foo"
    const beforeAttr = fixed;
    fixed = fixed.replace(/(<[a-z][^>]*\s)([\w-]+)=([^\s"'>][^\s>]*)/gi, (m, pre, attr, val) => {
      if (val.startsWith('"') || val.startsWith("'")) return m;
      return `${pre}${attr}="${val}"`;
    });
    if (fixed !== beforeAttr) changes.push('Added missing quotes on attributes');

    // Fix duplicate IDs — append suffix to duplicates
    const idCounts = {};
    fixed = fixed.replace(/id="([^"]+)"/g, (m, id) => {
      idCounts[id] = (idCounts[id] || 0) + 1;
      return idCounts[id] > 1 ? `id="${id}-${idCounts[id]}"` : m;
    });
    if (Object.values(idCounts).some(c => c > 1)) changes.push('Fixed duplicate IDs');

    // Fix missing alt on img tags
    const beforeAlt = fixed;
    fixed = fixed.replace(/<img(?![^>]*alt=)([^>]*?)(\s*\/?>)/gi, '<img$1 alt=""$2');
    if (fixed !== beforeAlt) changes.push('Added missing alt attributes to images');

    // DOMParser structural fix (closes tags, normalizes)
    const parser = new DOMParser();
    const doc = parser.parseFromString(fixed, 'text/html');
    const errors = doc.querySelectorAll('parsererror');
    if (errors.length > 0) changes.push('HTML parse errors detected (may not be fully fixable)');

    const hasDoctype = fixed.trim().toLowerCase().startsWith('<!doctype') || fixed.trim().toLowerCase().startsWith('<html');
    let structured = hasDoctype ? '<!DOCTYPE html>\n' + doc.documentElement.outerHTML : doc.body.innerHTML;

    if (structured.length < code.length * 0.5 && code.length > 100) {
      changes.push('Structural fix removed too much content \u2014 keeping pre-structural fixes only');
      return { code: fixed, fixed: fixed !== code, changes };
    }

    if (structured !== fixed) {
      const unclosedTags = (fixed.match(/<(?!\/|!|br|hr|img|input|meta|link)[a-z][^>]*>/gi) || []).length;
      const closeTags = (fixed.match(/<\/[a-z]+>/gi) || []).length;
      if (unclosedTags > closeTags) changes.push(`Fixed ${unclosedTags - closeTags} unclosed tag(s)`);
      else if (changes.length === 0) changes.push('Normalized HTML structure');
      fixed = structured;
    }

    return { code: fixed, fixed: fixed !== code, changes };
  } catch (err) {
    return { code, fixed: false, changes: [`Fix failed: ${err.message}`] };
  }
}

export function fixCSS(code) {
  const changes = [];
  let fixed = code;

  // Missing semicolons
  const beforeSemifix = fixed;
  fixed = fixed.replace(/([a-zA-Z0-9%"')\s])(\s*\n\s*[a-zA-Z-]+\s*:)/g, '$1;$2');
  fixed = fixed.replace(/([a-zA-Z0-9%"')])(\s*\})/g, '$1;$2');
  fixed = fixed.replace(/;{2,}/g, ';');
  if (fixed !== beforeSemifix) changes.push('Added missing semicolons');

  // Unclosed braces
  const opens = (fixed.match(/\{/g) || []).length;
  const closes = (fixed.match(/\}/g) || []).length;
  if (opens > closes) {
    fixed += '\n}'.repeat(opens - closes);
    changes.push(`Closed ${opens - closes} unclosed brace(s)`);
  }

  // Missing colons in declarations
  const beforeColon = fixed;
  fixed = fixed.replace(/^\s*([\w-]+)\s+((?:#|rgb|hsl|[0-9]|"|'|[a-z]).+?;)/gm, '  $1: $2');
  if (fixed !== beforeColon) changes.push('Added missing colons in declarations');

  // Remove duplicate properties (keep last occurrence within each rule)
  const beforeDups = fixed;
  fixed = fixed.replace(/\{([^}]+)\}/g, (match, block) => {
    const lines = block.split('\n');
    const seen = new Map();
    const cleaned = [];
    for (const line of lines) {
      const propMatch = line.match(/^\s*([\w-]+)\s*:/);
      if (propMatch) {
        const prop = propMatch[1].toLowerCase();
        if (seen.has(prop)) {
          // Remove previous occurrence, keep this one
          cleaned[seen.get(prop)] = null;
        }
        seen.set(prop, cleaned.length);
      }
      cleaned.push(line);
    }
    return '{' + cleaned.filter(l => l !== null).join('\n') + '}';
  });
  if (fixed !== beforeDups) changes.push('Removed duplicate CSS properties (kept last)');

  // Fix common unit mistakes: 0px -> 0 (except 0%)
  const beforeUnits = fixed;
  fixed = fixed.replace(/:\s*0(px|em|rem|vh|vw|vmin|vmax)\b/g, ': 0');
  if (fixed !== beforeUnits) changes.push('Removed unnecessary units on zero values');

  // Fix missing space after colon
  const beforeSpace = fixed;
  fixed = fixed.replace(/([\w-]+):([^\s/])/g, '$1: $2');
  if (fixed !== beforeSpace) changes.push('Added missing space after colons');

  return { code: fixed, fixed: fixed !== code, changes };
}

/**
 * Fix Python code — deterministic whitespace and common pattern fixes.
 * No AST parsing (can't run Python parser in browser), but catches
 * the most common LLM generation errors.
 */
export function fixPython(code) {
  const changes = [];
  let fixed = code;

  // Fix mixed tabs and spaces (convert tabs to 4 spaces)
  if (fixed.includes('\t')) {
    fixed = fixed.replace(/\t/g, '    ');
    changes.push('Converted tabs to 4 spaces');
  }

  // Fix inconsistent indentation (detect dominant indent and normalize)
  const indents = fixed.match(/^( +)\S/gm);
  if (indents) {
    const sizes = indents.map(m => m.match(/^ +/)[0].length);
    const minIndent = Math.min(...sizes.filter(s => s > 0));
    if (minIndent === 2 || minIndent === 3) {
      // Normalize to 4-space indent (Python standard)
      const ratio = 4 / minIndent;
      if (Number.isInteger(ratio)) {
        const beforeIndent = fixed;
        fixed = fixed.replace(/^( +)/gm, (m, spaces) => ' '.repeat(spaces.length * ratio));
        if (fixed !== beforeIndent) changes.push(`Normalized indentation to 4 spaces`);
      }
    }
  }

  // Fix trailing whitespace
  const beforeTrail = fixed;
  fixed = fixed.split('\n').map(l => l.trimEnd()).join('\n');
  if (fixed !== beforeTrail) changes.push('Removed trailing whitespace');

  // Fix missing colons after def/class/if/elif/else/for/while/try/except/finally/with
  const beforeColon = fixed;
  fixed = fixed.replace(/^(\s*(?:def|class|if|elif|else|for|while|try|except|finally|with)\b[^:\n]*?)(\s*)$/gm, (m, stmt, trail) => {
    if (stmt.trimEnd().endsWith(':') || stmt.trimEnd().endsWith(',') || stmt.trimEnd().endsWith('\\')) return m;
    return stmt + ':' + trail;
  });
  if (fixed !== beforeColon) changes.push('Added missing colons after statements');

  // Fix common f-string issues: print(f"...{var}") — ensure f prefix
  const beforeFstr = fixed;
  fixed = fixed.replace(/(?<![fFrRbBuU])(["'])([^"']*\{[^}]+\}[^"']*)\1/g, (m, q, content) => {
    // Only add f-prefix if it looks like an f-string (has {varname} pattern)
    if (/\{[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*(?:\[.*?\])?\}/.test(content)) {
      return `f${q}${content}${q}`;
    }
    return m;
  });
  if (fixed !== beforeFstr) changes.push('Added missing f-string prefix');

  return { code: fixed, fixed: fixed !== code, changes };
}

/**
 * Fix JSON — parse and re-format, or fix common issues.
 */
export function fixJSON(code) {
  const changes = [];
  let fixed = code;

  // Try to parse as-is
  try {
    const obj = JSON.parse(fixed);
    const formatted = JSON.stringify(obj, null, 2);
    if (formatted !== fixed) {
      changes.push('Formatted JSON');
      fixed = formatted;
    }
    return { code: fixed, fixed: fixed !== code, changes };
  } catch { /* needs fixing */ }

  // Fix trailing commas (common LLM mistake)
  const beforeTrailing = fixed;
  fixed = fixed.replace(/,(\s*[}\]])/g, '$1');
  if (fixed !== beforeTrailing) changes.push('Removed trailing commas');

  // Fix single quotes -> double quotes
  const beforeQuotes = fixed;
  fixed = fixed.replace(/'/g, '"');
  if (fixed !== beforeQuotes) changes.push('Converted single quotes to double quotes');

  // Try to parse again after fixes
  try {
    const obj = JSON.parse(fixed);
    fixed = JSON.stringify(obj, null, 2);
    changes.push('Reformatted JSON');
    return { code: fixed, fixed: true, changes };
  } catch (err) {
    changes.push(`JSON still invalid: ${err.message}`);
    return { code: fixed, fixed: fixed !== code, changes };
  }
}

/**
 * Silently lint code after an AI edit. Returns the fixed code if issues found.
 *
 * Pipeline: CodeMind AST validation (fast) -> deterministic fixers (per-language).
 * JS is excluded from deterministic fixers (async CDN load, too heavy for silent path).
 */
export function silentLint(lang, code) {
  const langLower = (lang || '').toLowerCase();

  // Phase 1: CodeMind AST validation (sub-5ms, catches structural errors)
  if (CodeMind.isReady()) {
    const cmLang = CodeMind.resolveLanguage(langLower);
    if (cmLang && CodeMind.isLanguageLoaded(cmLang)) {
      const diag = CodeMind.getDiagnostics(code, cmLang);
      if (diag.length > 0) {
        // Log AST errors for debugging but don't block — fixers may handle them
        console.debug(`[CodeMind] ${diag.length} syntax issue(s) in ${lang}:`,
          diag.slice(0, 3).map(d => `Ln ${d.line}: ${d.message}`).join('; '));
      }
    }
  }

  // Phase 2: Deterministic fixers (per-language pattern repair)
  try {
    if (['html', 'htm'].includes(langLower)) return fixHTML(code);
    if (['css', 'scss'].includes(langLower)) return fixCSS(code);
    if (['python', 'py'].includes(langLower)) return fixPython(code);
    if (['json'].includes(langLower)) return fixJSON(code);
  } catch { /* fixer failed — return unfixed */ }
  return { code, fixed: false, changes: [] };
}

export async function autoFixCodeBlock(codeHeader) {
  const rawCode = decodeURIComponent(codeHeader.dataset.rawCode || '');
  const lang = (codeHeader.dataset.lang || '').toLowerCase();
  if (!rawCode) return;

  const fixBtn = codeHeader.querySelector('[data-action="auto-fix"]');
  if (fixBtn) { fixBtn.classList.add('active'); fixBtn.textContent = 'Fixing...'; }

  let result;
  try {
    if (['html', 'htm'].includes(lang)) result = fixHTML(rawCode);
    else if (['javascript', 'js', 'jsx'].includes(lang)) result = await _fixJavaScript(rawCode);
    else if (['css', 'scss'].includes(lang)) result = fixCSS(rawCode);
    else if (['python', 'py'].includes(lang)) result = fixPython(rawCode);
    else if (['json'].includes(lang)) result = fixJSON(rawCode);
    else result = { code: rawCode, fixed: false, changes: ['No fixer available for this language'] };

    if (result.fixed) {
      const summary = 'Auto-fix: ' + result.changes.join(', ');
      const pre = codeHeader.nextElementSibling?.tagName === 'PRE'
        ? codeHeader.nextElementSibling
        : codeHeader.nextElementSibling?.nextElementSibling;
      const rawCode = decodeURIComponent(codeHeader.dataset.rawCode || '');

      const accepted = await _showDiffReview(codeHeader, pre, rawCode, result.code, summary, lang, { acceptLabel: 'Accept Fix' });
      if (accepted) {
        updateBlock(codeHeader, result.code, summary, true);
        const block = getBlock(codeHeader);
        if (block) showVersion(codeHeader, block.versions.length - 1);
        showToast(summary, 'success');
      } else {
        showToast('Skipped the fix.', 'info');
      }
    } else {
      showToast(result.changes.length > 0 ? result.changes[0] : 'Nothing to fix.', 'info');
    }
  } catch (err) {
    showToast(`Couldn't auto-fix — ${err.message}`, 'error');
  } finally {
    if (fixBtn) { fixBtn.classList.remove('active'); fixBtn.textContent = 'Fix'; }
  }
}
