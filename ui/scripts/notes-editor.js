/**
 * Augmentum Notes Editor — CodeMirror 6 + Live Preview
 *
 * A seamless markdown editor where the page IS the writing surface.
 * The document stores raw markdown; a ViewPlugin decorates the syntax
 * tree so headings render as headings, bold as bold, blockquote-as-
 * blockquote — all inline, without rewriting text. Formatting marks
 * fade to 30% when the caret leaves their line and return to full
 * opacity when the caret re-enters. The net feel is Obsidian Live
 * Preview / Typora / Bear: no toolbar, no preview pane, just a page.
 *
 * Public API (mirrors EasyMDE/Milkdown-era shims so browse.js doesn't
 * need to know which editor is underneath):
 *
 *   const editor = await createNotesEditor({ element, value, onChange });
 *   editor.value()                -> string
 *   editor.value(next)            -> void (replaces doc)
 *   editor.focus()
 *   editor.destroy()              -> void
 *   editor.on('change', fn)       -> unsubscribe fn
 *   editor.codemirror             -> EditorView (escape hatch)
 *   editor.getMarkdown()          -> alias for value() (legacy callers)
 *   editor.toTextArea()           -> alias for destroy() (legacy callers)
 *
 * Single-file by design; splitting is premature until a second surface
 * embeds it.
 */

import * as SlashMenu from './note-slash-menu.js';

let _cm = null;   // cached module bundle after first load

// ---------------------------------------------------------------------------
// Notes-AI surface (cross-module helpers exposed on window)
// ---------------------------------------------------------------------------
// The notes-editor doesn't currently own a "transform selected text" UI
// surface — the slash menu handles new-content insertion, not in-place
// transforms. These helpers expose the backend AI actions so any future
// UI (right-click menu, toolbar dropdown, command palette) can wire to
// them without duplicating the fetch logic.
//
// Backend: /api/notes/{note_id}/ai accepts {action, selected_text, context, option}
// and returns {result}. Actions: rewrite, expand, compress, research, define.
// Backend: /api/notes/tags returns {tags: [...]} for autocomplete.

window.augmentumNoteAiAction = async function noteAiAction(noteId, action, selectedText, opts = {}) {
  if (!noteId || !action || !selectedText) return null;
  try {
    const resp = await fetch(`/api/notes/${encodeURIComponent(noteId)}/ai`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        action,
        selected_text: selectedText,
        context: opts.context || '',
        option: opts.option || '',
      }),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch (_) {
    return null;
  }
};

window.augmentumNoteTags = async function noteTags() {
  try {
    const resp = await fetch('/api/notes/tags', { credentials: 'same-origin' });
    if (!resp.ok) return [];
    const data = await resp.json();
    return Array.isArray(data?.tags) ? data.tags : [];
  } catch (_) {
    return [];
  }
};

// CM6 is a constellation of sibling packages (state/view/language/
// commands/search/autocomplete/lang-markdown/…). Every one of them
// has its own dependency on @codemirror/state. If any two packages
// resolve against DIFFERENT @codemirror/state modules, Compartment /
// Facet / Extension instanceof checks fail at editor creation with:
//
//   "Unrecognized extension value in extension set ([object Object]).
//    This sometimes happens because multiple instances of
//    @codemirror/state are loaded, breaking instanceof checks."
//
// Fix: import map in ui/index.html pins each CM package + `?external=`
// on every non-leaf package so their internal `import '@codemirror/state'`
// statements stay as BARE specifiers that the browser resolves through
// the same map — guaranteeing one shared @codemirror/state instance.
// Here we just import the bare specifiers and trust the map.
async function loadCM6() {
  if (_cm) return _cm;
  const [
    stateMod,
    viewMod,
    languageMod,
    commandsMod,
    searchMod,
    autocompleteMod,
    markdownMod,
    highlightMod,
  ] = await Promise.all([
    import('@codemirror/state'),
    import('@codemirror/view'),
    import('@codemirror/language'),
    import('@codemirror/commands'),
    import('@codemirror/search'),
    import('@codemirror/autocomplete'),
    import('@codemirror/lang-markdown'),
    import('@lezer/highlight'),
  ]);
  _cm = {
    ...stateMod,
    ...viewMod,
    ...languageMod,
    ...commandsMod,
    ...searchMod,
    ...autocompleteMod,
    markdown: markdownMod.markdown,
    markdownLanguage: markdownMod.markdownLanguage,
    tags: highlightMod.tags,
  };
  return _cm;
}

// ---------------------------------------------------------------------------
// Live Preview: walk the syntax tree and emit decorations that make the
// source markdown render inline. Mark-fade on cursor-line is handled
// entirely in CSS via the `.cm-md-cursor-line` class on the active line.
// ---------------------------------------------------------------------------
function buildLivePreviewExtension(cm) {
  const { ViewPlugin, Decoration, WidgetType, EditorView } = cm;
  const { syntaxTree } = cm;

  // Inline image widget — replaces `![alt](url)` with the rendered image
  // (like the checkbox, rendered unconditionally so the layout stays stable
  // and the caret treats it atomically: backspace deletes the whole image).
  // This is what makes paste/drop of a photo actually SHOW the photo rather
  // than leaving raw markdown text.
  class ImageWidget extends WidgetType {
    constructor(url, alt) { super(); this.url = url; this.alt = alt; }
    eq(other) { return other.url === this.url && other.alt === this.alt; }
    toDOM() {
      const wrap = document.createElement('span');
      // `is-loading` reserves a placeholder box (CSS) so the surrounding
      // text doesn't reflow when the decoded image swaps in — the image
      // CLS we'd otherwise pay every time one scrolls into view. The
      // widget is already viewport-scoped (only visible-range images get a
      // widget), so `loading="lazy"` would add pop-in shift for nothing.
      wrap.className = 'cm-md-image-embed is-loading';
      const img = document.createElement('img');
      img.alt = this.alt || '';
      img.decoding = 'async';
      img.addEventListener('load', () => wrap.classList.remove('is-loading'));
      img.addEventListener('error', () => wrap.classList.remove('is-loading'));
      img.src = this.url;
      wrap.appendChild(img);
      return wrap;
    }
    ignoreEvent() { return false; }
  }

  // Task checkbox widget — replaces `[ ]` / `[x]` with a clickable box
  // when the caret isn't on that line. Lets users toggle with a tap.
  class CheckboxWidget extends WidgetType {
    constructor(checked, pos) { super(); this.checked = checked; this.pos = pos; }
    eq(other) { return other.checked === this.checked && other.pos === this.pos; }
    toDOM(view) {
      const box = document.createElement('span');
      box.className = 'cm-md-checkbox' + (this.checked ? ' checked' : '');
      box.setAttribute('aria-checked', this.checked ? 'true' : 'false');
      box.setAttribute('role', 'checkbox');
      box.addEventListener('mousedown', (e) => {
        e.preventDefault();
        const from = this.pos;
        const to = this.pos + 3;   // `[ ]` or `[x]`
        const replacement = this.checked ? '[ ]' : '[x]';
        view.dispatch({ changes: { from, to, insert: replacement } });
      });
      return box;
    }
    ignoreEvent() { return false; }
  }

  // Active-line marker is split into its OWN plugin (below). Caret moves
  // fire `selectionSet` on every click/tap/arrow-key; folding the one
  // cursor-line decoration into the heavy syntax walk meant each of those
  // re-iterated the whole visible syntax tree to produce IDENTICAL syntax
  // decorations — the dominant INP cost on long notes (the mark-fade is
  // pure CSS via `.cm-md-cursor-line`, so the syntax decorations never
  // actually depend on cursor position). The syntax plugin now rebuilds
  // only when the doc or viewport changes; the cursor-line plugin carries
  // the per-selection cost, which is a single line decoration.
  const syntaxPlugin = ViewPlugin.fromClass(class {
    constructor(view) { this.decorations = this.buildDecorations(view); }
    update(update) {
      if (update.docChanged || update.viewportChanged) {
        this.decorations = this.buildDecorations(update.view);
      }
    }
    buildDecorations(view) {
      const decos = [];
      const { state } = view;

      // (Earlier builds compressed blank lines here; removed because
      // changing a line's font-size at CSS-level makes CM6's virtual
      // scroller re-measure on every scroll/resize and emit "Measure
      // loop restarted more than 5 times" warnings — the same uniform-
      // line-height principle that keeps clicking a code block from
      // shifting layout. The uniform 1.97rem line box gives acceptable
      // density for web-imported markdown without per-line margins.)

      for (const { from, to } of view.visibleRanges) {
        syntaxTree(state).iterate({
          from, to,
          enter: (node) => {
            const nStart = state.doc.lineAt(node.from).number;
            const nEnd = state.doc.lineAt(node.to).number;

            switch (node.name) {
              // ── Headings ───────────────────────────────────────────
              case 'ATXHeading1':
              case 'ATXHeading2':
              case 'ATXHeading3':
              case 'ATXHeading4':
              case 'ATXHeading5':
              case 'ATXHeading6': {
                const level = node.name.slice(-1);
                const line = state.doc.lineAt(node.from);
                decos.push(
                  Decoration.line({ class: `cm-md-heading cm-md-h${level}` })
                    .range(line.from),
                );
                break;
              }
              case 'HeaderMark':
                decos.push(
                  Decoration.mark({ class: 'cm-md-mark cm-md-mark-header' })
                    .range(node.from, node.to),
                );
                break;

              // ── Emphasis ──────────────────────────────────────────
              case 'StrongEmphasis':
                decos.push(Decoration.mark({ class: 'cm-md-bold' }).range(node.from, node.to));
                break;
              case 'Emphasis':
                decos.push(Decoration.mark({ class: 'cm-md-italic' }).range(node.from, node.to));
                break;
              case 'Strikethrough':
                decos.push(Decoration.mark({ class: 'cm-md-strike' }).range(node.from, node.to));
                break;
              case 'EmphasisMark':
              case 'StrikethroughMark':
                decos.push(
                  Decoration.mark({ class: 'cm-md-mark cm-md-mark-emph' })
                    .range(node.from, node.to),
                );
                break;

              // ── Inline code ────────────────────────────────────────
              case 'InlineCode':
                decos.push(
                  Decoration.mark({ class: 'cm-md-code-inline' })
                    .range(node.from, node.to),
                );
                break;
              case 'CodeMark':
                decos.push(
                  Decoration.mark({ class: 'cm-md-mark cm-md-mark-code' })
                    .range(node.from, node.to),
                );
                break;

              // ── Links ──────────────────────────────────────────────
              case 'Link':
                decos.push(
                  Decoration.mark({ class: 'cm-md-link' })
                    .range(node.from, node.to),
                );
                break;
              case 'LinkMark':
                decos.push(
                  Decoration.mark({ class: 'cm-md-mark cm-md-mark-link' })
                    .range(node.from, node.to),
                );
                break;
              case 'URL':
                // URL is always rendered (just styled faint + monospace).
                // Toggling visibility per-cursor-line caused large CLS
                // scores (every caret move reflowed the line). Keep it
                // stable; the mark class already fades by default.
                decos.push(
                  Decoration.mark({ class: 'cm-md-url' })
                    .range(node.from, node.to),
                );
                break;
              case 'Image': {
                // Parse `![alt](url)` and render the actual image. Only when
                // the URL is a safe scheme (http(s) or a same-origin path);
                // otherwise fall back to the faint markdown mark so a
                // half-typed or unsafe ref doesn't vanish or load junk.
                const raw = state.doc.sliceString(node.from, node.to);
                const m = raw.match(/^!\[([^\]]*)\]\(\s*(\S+?)\s*(?:"[^"]*")?\)$/);
                const src = m ? _safeImgSrc(m[2]) : null;
                if (src) {
                  decos.push(
                    Decoration.replace({
                      widget: new ImageWidget(src, m[1]),
                    }).range(node.from, node.to),
                  );
                  return false;  // don't descend into the URL/LinkMark children
                }
                decos.push(
                  Decoration.mark({ class: 'cm-md-image' })
                    .range(node.from, node.to),
                );
                break;
              }

              // ── Blockquote ─────────────────────────────────────────
              case 'Blockquote': {
                // Clamp to the visible window. A blockquote that runs past
                // the screen would otherwise emit a line decoration for
                // every off-screen line on every scroll frame — O(block)
                // per frame, the main fast-scroll stutter risk on long docs.
                const qFrom = Math.max(nStart, state.doc.lineAt(from).number);
                const qTo = Math.min(nEnd, state.doc.lineAt(to).number);
                for (let ln = qFrom; ln <= qTo; ln++) {
                  const line = state.doc.line(ln);
                  decos.push(
                    Decoration.line({ class: 'cm-md-quote-line' }).range(line.from),
                  );
                }
                break;
              }
              case 'QuoteMark':
                decos.push(
                  Decoration.mark({ class: 'cm-md-mark cm-md-mark-quote' })
                    .range(node.from, node.to),
                );
                break;

              // ── Lists ──────────────────────────────────────────────
              case 'ListMark':
                decos.push(
                  Decoration.mark({ class: 'cm-md-mark cm-md-mark-list' })
                    .range(node.from, node.to),
                );
                break;
              case 'Task': {
                // Render `[ ]` / `[x]` as a checkbox widget unconditionally.
                // Replace decorations are atomic in CM6 so the caret
                // skips over them. Rendering regardless of cursor
                // position keeps layout stable (no CLS from widget ↔
                // text toggling on every pointer move).
                const text = state.doc.sliceString(node.from, node.from + 3);
                if (text === '[ ]' || text === '[x]' || text === '[X]') {
                  const checked = text.toLowerCase() === '[x]';
                  decos.push(
                    Decoration.replace({ widget: new CheckboxWidget(checked, node.from) })
                      .range(node.from, node.from + 3),
                  );
                  if (checked) {
                    const line = state.doc.lineAt(node.from);
                    decos.push(
                      Decoration.line({ class: 'cm-md-task-checked' }).range(line.from),
                    );
                  }
                }
                break;
              }

              // ── Code block ─────────────────────────────────────────
              case 'FencedCode':
              case 'CodeBlock': {
                // Clamp the per-line loop to the visible window (see
                // Blockquote above). The first/last rounded-corner classes
                // still compare against the TRUE block bounds (nStart/nEnd),
                // so a block whose top scrolled off-screen correctly omits
                // the rounded top until its real first line is in view.
                const cFrom = Math.max(nStart, state.doc.lineAt(from).number);
                const cTo = Math.min(nEnd, state.doc.lineAt(to).number);
                for (let ln = cFrom; ln <= cTo; ln++) {
                  const line = state.doc.line(ln);
                  const first = ln === nStart ? ' cm-md-code-block-first' : '';
                  const last = ln === nEnd ? ' cm-md-code-block-last' : '';
                  decos.push(
                    Decoration.line({
                      class: `cm-md-code-block${first}${last}`,
                    }).range(line.from),
                  );
                }
                break;
              }
              case 'CodeInfo':
                decos.push(
                  Decoration.mark({ class: 'cm-md-code-info' })
                    .range(node.from, node.to),
                );
                break;

              // ── Horizontal rule ────────────────────────────────────
              case 'HorizontalRule': {
                const line = state.doc.lineAt(node.from);
                decos.push(
                  Decoration.line({ class: 'cm-md-hr-line' }).range(line.from),
                );
                break;
              }
            }
          },
        });
      }

      return Decoration.set(decos, /*sort=*/ true);
    }
  }, {
    // Decoration.replace widgets (our checkbox) are atomic by default
    // in CM6; mark decorations (bold/italic/headings) must stay non-
    // atomic so the caret can navigate inside them. No atomicRanges
    // provider needed — the defaults are correct.
    decorations: v => v.decorations,
  });

  // Cursor-line marker — the only selection-dependent decoration. Kept
  // separate (and trivially cheap) so a click/tap doesn't re-walk the
  // syntax tree. Rebuilds only when the caret LINE changes, not on every
  // selectionSet (clicking within the same line, or extending a selection
  // on one line, leaves the marker untouched).
  const cursorLinePlugin = ViewPlugin.fromClass(class {
    constructor(view) {
      this.line = view.state.doc.lineAt(view.state.selection.main.head).from;
      this.decorations = Decoration.set([
        Decoration.line({ class: 'cm-md-cursor-line' }).range(this.line),
      ]);
    }
    update(update) {
      if (!update.selectionSet && !update.docChanged) return;
      const from = update.state.doc.lineAt(update.state.selection.main.head).from;
      if (from === this.line && !update.docChanged) return;
      this.line = from;
      this.decorations = Decoration.set([
        Decoration.line({ class: 'cm-md-cursor-line' }).range(from),
      ]);
    }
  }, {
    decorations: v => v.decorations,
  });

  return [syntaxPlugin, cursorLinePlugin];
}

// ---------------------------------------------------------------------------
// Dark editorial theme — turns CM6's structural classes into our page.
// Kept small; all markdown-token styling lives in notes-editor.css so
// designers can tweak without round-tripping through JS.
// ---------------------------------------------------------------------------
function buildTheme(cm) {
  const { EditorView } = cm;
  return EditorView.theme({
    '&': {
      flex: '1',
      minHeight: '0',
      backgroundColor: 'transparent',
      color: 'var(--notes-ink, var(--text-primary))',
      fontFamily: 'var(--font-editorial), "Crimson Pro", Georgia, serif',
      fontSize: '1.0625rem',
      // Editorial leading for comfortable long-form reading. Expressed as an
      // ABSOLUTE length (≈ 1.0625rem × 1.85) rather than a unitless ratio so
      // EVERY line — prose, lists, quotes, and code (which sets the same
      // 1.97rem in notes-editor.css) — has an identical line box. CM6 holds
      // one height estimate for off-screen lines; uniform boxes make that
      // estimate exact, eliminating the re-measure shifts (CLS) and layout
      // cost (INP) that clicking around a mixed-height document produced.
      lineHeight: '1.97rem',
      letterSpacing: '0',
    },
    '&.cm-focused': { outline: 'none' },
    '.cm-scroller': {
      fontFamily: 'inherit',
      padding: '6px 0 96px',
      overflowX: 'hidden',
    },
    '.cm-content': {
      maxWidth: '100%',
      margin: '0',
      caretColor: 'var(--accent, #6c8aff)',
      padding: '0',
    },
    '.cm-line': { padding: '0' },
    '.cm-cursor, .cm-dropCursor': {
      borderLeftWidth: '2px',
      borderLeftColor: 'var(--accent, #6c8aff)',
    },
    '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': {
      background: 'color-mix(in srgb, var(--accent, #6c8aff) 22%, transparent)',
    },
    '.cm-placeholder': {
      color: 'var(--notes-ink-ghost, var(--text-muted))',
      fontStyle: 'italic',
    },
  }, { dark: true });
}

// ---------------------------------------------------------------------------
// Format helpers (exported so the mobile toolbar can use the same logic).
//
// `toggleWrap` — toggle paired markers around the selection (bold/italic/
// code). Checks three cases so repeated Mod-B on the same word removes
// the markers instead of stacking them:
//   1. Selection itself reads as `**text**` → strip
//   2. Text around the selection is `**text**` → strip the surrounds
//   3. Otherwise → wrap
// Empty selection: if the caret sits between existing markers, strip;
// otherwise insert a paired `**|**` with the caret between.
//
// `toggleLinePrefix` — toggle a line-start marker (heading / list /
// checkbox / quote). Accepts an optional regex so variants are
// recognised (e.g. "### " counts as a heading even when the toggle
// inserts "# ").
// ---------------------------------------------------------------------------
export function toggleWrap(view, open, close = open) {
  const { state } = view;
  const r = state.selection.main;
  const openLen = open.length;
  const closeLen = close.length;

  if (!r.empty) {
    const selText = state.doc.sliceString(r.from, r.to);
    if (selText.length >= openLen + closeLen
        && selText.startsWith(open) && selText.endsWith(close)) {
      const inner = selText.slice(openLen, selText.length - closeLen);
      view.dispatch({
        changes: { from: r.from, to: r.to, insert: inner },
        selection: { anchor: r.from, head: r.from + inner.length },
        userEvent: 'input.format',
      });
      view.focus();
      return;
    }
    const before = state.doc.sliceString(Math.max(0, r.from - openLen), r.from);
    const after = state.doc.sliceString(r.to, r.to + closeLen);
    if (before === open && after === close) {
      view.dispatch({
        changes: [
          { from: r.from - openLen, to: r.from, insert: '' },
          { from: r.to, to: r.to + closeLen, insert: '' },
        ],
        selection: { anchor: r.from - openLen, head: r.to - openLen },
        userEvent: 'input.format',
      });
      view.focus();
      return;
    }
    view.dispatch({
      changes: { from: r.from, to: r.to, insert: open + selText + close },
      selection: {
        anchor: r.from + openLen,
        head: r.to + openLen,
      },
      userEvent: 'input.format',
    });
    view.focus();
    return;
  }

  // Empty selection: collapse if caret sits in `open|close`
  const before = state.doc.sliceString(Math.max(0, r.from - openLen), r.from);
  const after = state.doc.sliceString(r.from, r.from + closeLen);
  if (before === open && after === close) {
    view.dispatch({
      changes: [
        { from: r.from - openLen, to: r.from, insert: '' },
        { from: r.from, to: r.from + closeLen, insert: '' },
      ],
      selection: { anchor: r.from - openLen },
      userEvent: 'input.format',
    });
    view.focus();
    return;
  }
  view.dispatch({
    changes: { from: r.from, insert: open + close },
    selection: { anchor: r.from + openLen },
    userEvent: 'input.format',
  });
  view.focus();
}

export function toggleLinePrefix(view, prefix, detectPattern = null) {
  const { state } = view;
  const r = state.selection.main;
  const fromLine = state.doc.lineAt(r.from);
  const toLine = state.doc.lineAt(r.to);
  const pattern = detectPattern
    || new RegExp('^' + prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));

  const changes = [];
  for (let ln = fromLine.number; ln <= toLine.number; ln++) {
    const line = state.doc.line(ln);
    const match = line.text.match(pattern);
    if (match) {
      changes.push({ from: line.from, to: line.from + match[0].length, insert: '' });
    } else {
      changes.push({ from: line.from, insert: prefix });
    }
  }
  if (!changes.length) return;
  view.dispatch({ changes, userEvent: 'input.format' });
  view.focus();
}

// ---------------------------------------------------------------------------
// Markdown shortcuts + keybindings
// ---------------------------------------------------------------------------
function buildKeymap(cm, handlers) {
  const { keymap } = cm;

  return keymap.of([
    { key: 'Mod-b', run: (v) => { toggleWrap(v, '**'); return true; } },
    { key: 'Mod-i', run: (v) => { toggleWrap(v, '*');  return true; } },
    { key: 'Mod-`', run: (v) => { toggleWrap(v, '`');  return true; } },
    { key: 'Mod-Shift-x', run: (v) => { toggleWrap(v, '~~'); return true; } },
    {
      key: 'Mod-k',
      run: (v) => {
        const { state } = v;
        const r = state.selection.main;
        const text = state.doc.sliceString(r.from, r.to) || 'link text';
        v.dispatch({
          changes: { from: r.from, to: r.to, insert: `[${text}](url)` },
          selection: { anchor: r.from + text.length + 3, head: r.from + text.length + 6 },
        });
        return true;
      },
    },
    {
      key: 'Mod-/',
      run: () => { handlers.openSlashMenu?.(); return true; },
    },
    {
      key: 'Mod-j',
      run: () => { handlers.askAi?.(); return true; },
    },
  ]);
}

// Coarse-pointer (touch) detection — used to flip link-open from
// "modifier-click" (desktop) to "tap" (phone/tablet) and to suppress the
// floating selection bubble where the mobile keyboard toolbar already
// covers formatting + AI.
function _isCoarsePointer() {
  try { return window.matchMedia('(pointer: coarse)').matches; }
  catch (_) { return false; }
}

const _SAFE_LINK_RE = /^(https?:\/\/|\/|mailto:|tel:|#)/i;

// Image src is narrower than link href: only http(s) or a same-origin path
// (e.g. /api/chat-images/...). Blocks data:/javascript: from rendering as an
// <img>. Returns the url if safe, else null.
function _safeImgSrc(url) {
  return /^(https?:\/\/|\/)/i.test(String(url || '')) ? url : null;
}

// Read a File as a data URL, POST it to the shared user-scoped image store
// (/api/chat-images — reused, not chat-specific), and return the stable
// served URL. Throws on failure so the caller can surface it.
async function _uploadImageFile(file) {
  const dataUrl = await new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = () => reject(fr.error || new Error('read failed'));
    fr.readAsDataURL(file);
  });
  const resp = await fetch('/api/chat-images', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({ data_url: dataUrl }),
  });
  if (!resp.ok) throw new Error(`upload failed (${resp.status})`);
  const data = await resp.json();
  if (!data || !data.url) throw new Error('no url in response');
  return data.url;
}

// ---------------------------------------------------------------------------
// Paste / drop image attachment. Pasting a screenshot or dragging an image
// file into the note uploads it and inserts `![name](url)`, which the live-
// preview then renders inline via ImageWidget. (Apple-Notes-style photos —
// previously paste only handled text/html and image blobs were dropped.)
// ---------------------------------------------------------------------------
function buildImageAttachHandler(cm) {
  const { EditorView } = cm;

  const insertAt = (view, pos, md) => {
    view.dispatch({
      changes: { from: pos, insert: md },
      selection: { anchor: pos + md.length },
      userEvent: 'input',
    });
    view.focus();
  };

  const handleFiles = async (view, files) => {
    const imgs = Array.from(files).filter((f) => f.type && f.type.startsWith('image/'));
    if (!imgs.length) return false;
    for (const f of imgs) {
      try {
        const url = await _uploadImageFile(f);
        const alt = (f.name || 'image').replace(/\.[^.]+$/, '');
        // Re-read the caret each iteration — the doc grew on the previous insert.
        insertAt(view, view.state.selection.main.head, `\n![${alt}](${url})\n`);
      } catch (e) {
        console.warn('[notes] image attach failed', e);
        // Let the orchestrator surface a toast if it's listening.
        document.dispatchEvent(new CustomEvent('note-image-attach-error', { detail: { error: String(e) } }));
      }
    }
    return true;
  };

  const hasImage = (list) => list && list.length
    && Array.from(list).some((f) => f.type && f.type.startsWith('image/'));

  return EditorView.domEventHandlers({
    paste(event, view) {
      const files = event.clipboardData && event.clipboardData.files;
      if (!hasImage(files)) return false;
      event.preventDefault();
      void handleFiles(view, files);
      return true;
    },
    drop(event, view) {
      const files = event.dataTransfer && event.dataTransfer.files;
      if (!hasImage(files)) return false;
      event.preventDefault();
      // Move the caret to the drop point so the image lands where dropped.
      const pos = view.posAtCoords({ x: event.clientX, y: event.clientY });
      if (pos != null) view.dispatch({ selection: { anchor: pos } });
      void handleFiles(view, files);
      return true;
    },
  });
}

// Resolve the URL of any link under `pos`, or null. Handles `[text](url)`
// Link nodes, bare URL nodes, and `<url>` Autolinks.
function _resolveLinkUrlAt(cm, view, pos) {
  const { syntaxTree } = cm;
  const tree = syntaxTree(view.state);
  let node = tree.resolveInner(pos);
  let linkNode = null;
  while (node) {
    const n = node.type.name;
    if (n === 'Link' || n === 'Autolink' || n === 'URL') { linkNode = node; break; }
    node = node.parent;
  }
  if (!linkNode) return null;

  let url = null;
  if (linkNode.type.name === 'URL') {
    url = view.state.doc.sliceString(linkNode.from, linkNode.to);
  } else {
    tree.iterate({
      from: linkNode.from, to: linkNode.to,
      enter: ({ type, from, to }) => {
        if (url) return false;
        if (type.name === 'URL') url = view.state.doc.sliceString(from, to);
      },
    });
    if (!url && linkNode.type.name === 'Autolink') {
      url = view.state.doc.sliceString(linkNode.from, linkNode.to).replace(/^<|>$/g, '');
    }
  }
  return url || null;
}

// ---------------------------------------------------------------------------
// Open-link affordances, pointer-appropriate:
//   • Desktop (mouse): Cmd/Ctrl-click opens; plain click positions the caret
//     (so link text stays editable).
//   • Touch (coarse pointer): a clean TAP on a link opens it — phones/tablets
//     are the read-heavy devices and had no way to follow a link before.
//     A LONG-PRESS (>500ms) or a drag (>10px, i.e. a scroll) does NOT open,
//     leaving the native select/place-caret behaviour intact for editing.
// ---------------------------------------------------------------------------
function buildLinkClickHandler(cm) {
  const { EditorView } = cm;
  let touch = null;  // { x, y, t, url } captured at touchstart

  const open = (url, event) => {
    if (!url || !_SAFE_LINK_RE.test(url)) return false;
    event.preventDefault();
    window.open(url, '_blank', 'noopener,noreferrer');
    return true;
  };

  return EditorView.domEventHandlers({
    mousedown(event, view) {
      if (!event.ctrlKey && !event.metaKey) return false;
      if (event.button !== 0) return false;
      const pos = view.posAtCoords({ x: event.clientX, y: event.clientY });
      if (pos == null) return false;
      return open(_resolveLinkUrlAt(cm, view, pos), event);
    },
    touchstart(event, view) {
      touch = null;
      if (!event.touches || event.touches.length !== 1) return false;
      const t = event.touches[0];
      const pos = view.posAtCoords({ x: t.clientX, y: t.clientY });
      const url = pos == null ? null : _resolveLinkUrlAt(cm, view, pos);
      if (url) touch = { x: t.clientX, y: t.clientY, t: event.timeStamp, url };
      return false;  // never block caret placement or scroll start
    },
    touchend(event, view) {
      const rec = touch; touch = null;
      if (!rec) return false;
      const t = event.changedTouches && event.changedTouches[0];
      if (!t) return false;
      const moved = Math.hypot(t.clientX - rec.x, t.clientY - rec.y);
      const held = event.timeStamp - rec.t;
      if (moved > 10 || held > 500) return false;  // scroll or long-press → don't hijack
      return open(rec.url, event);
    },
  });
}

// ---------------------------------------------------------------------------
// Floating selection bubble — appears above a non-empty selection with
// quick formatting + AI actions. Rendered as a sibling of the editor
// so it can escape overflow:hidden parents.
// ---------------------------------------------------------------------------
function mountBubbleMenu(view, host, handlers) {
  const bubble = document.createElement('div');
  bubble.className = 'notes-bubble';
  bubble.setAttribute('role', 'toolbar');
  bubble.innerHTML = `
    <button data-act="bold" title="Bold (⌘B)"><b>B</b></button>
    <button data-act="italic" title="Italic (⌘I)"><i>I</i></button>
    <button data-act="strike" title="Strikethrough (⌘⇧X)"><s>S</s></button>
    <button data-act="code" title="Inline code (⌘\`)">&lt;/&gt;</button>
    <button data-act="link" title="Link (⌘K)">↗</button>
    <span class="notes-bubble-sep"></span>
    <button data-act="ai" title="Ask AI (⌘J)">✦ AI</button>
  `;
  host.appendChild(bubble);

  const act = (name) => {
    const { state } = view;
    const r = state.selection.main;
    const text = state.doc.sliceString(r.from, r.to);
    if (name === 'ai') { handlers.askAi?.(text); return; }
    if (r.empty && name !== 'link') return;
    if (name === 'bold')   { toggleWrap(view, '**'); return; }
    if (name === 'italic') { toggleWrap(view, '*');  return; }
    if (name === 'strike') { toggleWrap(view, '~~'); return; }
    if (name === 'code')   { toggleWrap(view, '`');  return; }
    if (name === 'link') {
      // Link is an insert-not-toggle: wrap if selected text is plain,
      // do nothing if already inside a `[x](url)` syntax.
      view.dispatch({
        changes: { from: r.from, to: r.to, insert: `[${text || 'text'}](url)` },
        selection: {
          anchor: r.from + (text || 'text').length + 3,
          head: r.from + (text || 'text').length + 6,
        },
        userEvent: 'input.format',
      });
      view.focus();
      return;
    }
  };

  bubble.addEventListener('mousedown', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    e.preventDefault();
    act(btn.dataset.act);
  });

  const reposition = () => {
    const sel = view.state.selection.main;
    // On touch, suppress the floating bubble entirely: it fought the OS's
    // native selection callout for the same screen space, and the mobile
    // keyboard toolbar (note-mobile-toolbar.js) already covers formatting +
    // AI on these devices.
    if (sel.empty || _isCoarsePointer()) {
      bubble.classList.remove('visible');
      return;
    }
    const coords = view.coordsAtPos(sel.from);
    const coordsEnd = view.coordsAtPos(sel.to);
    if (!coords || !coordsEnd) return;
    const hostRect = host.getBoundingClientRect();
    const midX = (coords.left + coordsEnd.right) / 2 - hostRect.left;
    const top = coords.top - hostRect.top - 44;
    bubble.style.left = `${Math.max(8, midX - bubble.offsetWidth / 2)}px`;
    bubble.style.top = `${Math.max(8, top)}px`;
    bubble.classList.add('visible');
  };

  const off = () => {
    // Keep the bubble alive while the user drags inside it
    if (document.activeElement === bubble || bubble.contains(document.activeElement)) return;
    bubble.classList.remove('visible');
  };

  // CM6 selection updates don't fire native 'selectionchange'; use the
  // view's update plugin hook by subscribing via an updateListener.
  return {
    bubble,
    reposition,
    off,
  };
}

// ---------------------------------------------------------------------------
// Public factory
// ---------------------------------------------------------------------------
export async function createNotesEditor({ element, value = '', onChange, handlers = {} }) {
  if (!element) throw new Error('createNotesEditor: element required');

  const cm = await loadCM6();
  const {
    EditorState, EditorView, ViewPlugin, Decoration,
    keymap, placeholder, highlightSpecialChars, drawSelection,
    dropCursor, crosshairCursor, rectangularSelection,
    lineNumbers, highlightActiveLineGutter,   // not used but imported
    defaultKeymap, history, historyKeymap, indentWithTab,
    search, searchKeymap, closeSearchPanel, openSearchPanel,
    autocompletion, completionKeymap, closeBracketsKeymap, closeBrackets,
    bracketMatching, indentOnInput, HighlightStyle, syntaxHighlighting,
    foldKeymap,
    markdown, markdownLanguage, tags,
  } = cm;

  // Wrapper element — host for the bubble menu + any future overlays
  const host = document.createElement('div');
  host.className = 'notes-editor';
  element.innerHTML = '';
  element.appendChild(host);

  // Syntax coloring inside fenced code blocks
  const highlightStyle = HighlightStyle.define([
    { tag: tags.keyword, color: 'var(--notes-syntax-keyword, #a855f7)' },
    { tag: tags.comment, color: 'var(--notes-syntax-comment, #6b7280)', fontStyle: 'italic' },
    { tag: tags.string, color: 'var(--notes-syntax-string, #15803d)' },
    { tag: tags.number, color: 'var(--notes-syntax-number, #c2410c)' },
    { tag: tags.variableName, color: 'var(--notes-syntax-variable, var(--notes-ink))' },
    { tag: tags.function(tags.variableName), color: 'var(--notes-syntax-function, #2563eb)' },
    { tag: tags.typeName, color: 'var(--notes-syntax-type, #b45309)' },
    { tag: tags.operator, color: 'var(--notes-syntax-operator, #0891b2)' },
    { tag: tags.punctuation, color: 'var(--notes-syntax-punctuation, #0891b2)' },
    { tag: tags.heading, color: 'var(--notes-ink)', fontWeight: '600' },
    { tag: tags.emphasis, fontStyle: 'italic' },
    { tag: tags.strong, fontWeight: '700', color: 'var(--notes-ink)' },
    { tag: tags.link, color: 'var(--accent, #6c8aff)' },
  ]);

  const listeners = new Set();
  const notifyChange = () => { for (const fn of listeners) { try { fn(); } catch {} } };

  const menuHandlers = {
    openSlashMenu: () => SlashMenu.openAtCursor(view),
    askAi: (selectedText) => handlers.askAi?.(selectedText, view),
  };

  const updateListener = EditorView.updateListener.of((update) => {
    // Slash-menu observes all updates (watches for `/` + filter changes)
    SlashMenu.handleUpdate(update);
    if (update.docChanged) {
      notifyChange();
      if (onChange) onChange();
    }
    if (update.selectionSet || update.docChanged) {
      queueMicrotask(() => bubble?.reposition());
    }
  });

  const state = EditorState.create({
    doc: value || '',
    extensions: [
      history(),
      drawSelection(),
      dropCursor(),
      indentOnInput(),
      bracketMatching(),
      closeBrackets(),
      autocompletion({ defaultKeymap: false }),
      rectangularSelection(),
      crosshairCursor(),
      highlightSpecialChars(),
      EditorView.lineWrapping,
      // Native CM6 find/replace. `top: true` floats the panel at the top
      // of the editor (matches the old find-bar position). This replaces
      // the previous custom find bar in note-editor.js, which walked a
      // `.ProseMirror` node that no longer exists under the CM6 editor and
      // so always reported "No matches". searchKeymap (Mod-f / Mod-g /
      // Escape) is already wired below.
      search({ top: true }),
      placeholder('Start writing…'),
      markdown({ base: markdownLanguage, codeLanguages: [] }),
      syntaxHighlighting(highlightStyle),
      buildLivePreviewExtension(cm),
      buildLinkClickHandler(cm),
      buildImageAttachHandler(cm),
      buildTheme(cm),
      buildKeymap(cm, menuHandlers),
      keymap.of([
        ...closeBracketsKeymap,
        ...defaultKeymap,
        ...searchKeymap,
        ...historyKeymap,
        ...foldKeymap,
        ...completionKeymap,
        indentWithTab,
      ]),
      updateListener,
    ],
  });

  const view = new EditorView({ state, parent: host });

  // Wire the slash menu to this view. Its updateListener (registered
  // above) already ran once during construction; setView ensures any
  // slash-item insertion / image drop resolves against the correct
  // view when multiple notes have been opened in the session.
  SlashMenu.setView(view);

  // Bubble menu mounts after the view so it can call coordsAtPos.
  const bubble = mountBubbleMenu(view, host, menuHandlers);

  // Modifier-key cursor hint — when Cmd/Ctrl is held, links show a
  // pointer cursor so the Cmd-click-to-open affordance is discoverable.
  // Scoped to the host so a held modifier on any other surface doesn't
  // flicker other editors in the page.
  const onModDown = (e) => {
    if (e.ctrlKey || e.metaKey) host.classList.add('notes-editor--mod-held');
  };
  const onModUp = (e) => {
    if (!e.ctrlKey && !e.metaKey) host.classList.remove('notes-editor--mod-held');
  };
  window.addEventListener('keydown', onModDown);
  window.addEventListener('keyup', onModUp);
  window.addEventListener('blur', () => host.classList.remove('notes-editor--mod-held'));

  // Public API
  const api = {
    get codemirror() { return view; },
    value(next) {
      if (next === undefined) return view.state.doc.toString();
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: next || '' },
        userEvent: 'set',
      });
    },
    focus() { view.focus(); },
    // Open the native search panel. Exposed so the orchestrator / mobile
    // toolbar / command palette can trigger find without re-implementing it.
    find() { try { openSearchPanel(view); } catch {} },
    destroy() {
      try { SlashMenu.close(); } catch {}
      try { SlashMenu.setView(null); } catch {}
      try { window.removeEventListener('keydown', onModDown); } catch {}
      try { window.removeEventListener('keyup', onModUp); } catch {}
      try { view.destroy(); } catch {}
      try { host.remove(); } catch {}
      listeners.clear();
    },
    on(event, fn) {
      if (event !== 'change' || typeof fn !== 'function') return () => {};
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    // Legacy shims for code still calling the Milkdown/EasyMDE API
    getMarkdown() { return api.value(); },
    toTextArea() { api.destroy(); },
  };

  return api;
}

/**
 * Eager-prefetch the CM6 bundle so the first note-open feels instant.
 * Call from surface init; harmless if omitted.
 */
export function prefetchNotesEditor() {
  return loadCM6().catch(() => null);
}
