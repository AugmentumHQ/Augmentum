/**
 * CodeMirror 6 Editor Module — replaces the textarea+Prism overlay.
 *
 * Lazy-loads CM6 from CDN. Provides a clean API for workspace.js to:
 * - Create/destroy editor instances
 * - Get/set content
 * - Listen for changes
 * - Add ghost text decorations
 * - Show diagnostics (lint markers)
 *
 * All CM6 specifics are contained here — workspace.js never imports CM6 directly.
 */

// ---------------------------------------------------------------------------
// CDN URLs — using esm.sh which deduplicates shared dependencies
// ---------------------------------------------------------------------------
// esm.sh resolves @codemirror/state to a single instance across all packages,
// avoiding the "multiple instances" error that breaks instanceof checks.
// jsdelivr's +esm does NOT do this — each package gets its own copy.

const ESM = 'https://esm.sh';

const IMPORTS = {
  // Core (only exports basicSetup, minimalSetup, EditorView)
  core: `${ESM}/codemirror@6`,
  // Sub-packages (need separate imports for everything else)
  state: `${ESM}/@codemirror/state@6`,
  view: `${ESM}/@codemirror/view@6`,
  commands: `${ESM}/@codemirror/commands@6`,
  autocomplete: `${ESM}/@codemirror/autocomplete@6`,
  lint: `${ESM}/@codemirror/lint@6`,
  language: `${ESM}/@codemirror/language@6`,
  search: `${ESM}/@codemirror/search@6`,
  // Languages
  langJs: `${ESM}/@codemirror/lang-javascript@6`,
  langHtml: `${ESM}/@codemirror/lang-html@6`,
  langCss: `${ESM}/@codemirror/lang-css@6`,
  langJson: `${ESM}/@codemirror/lang-json@6`,
  langPython: `${ESM}/@codemirror/lang-python@6`,
  // Theme
  themeDark: `${ESM}/@codemirror/theme-one-dark@6`,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _loaded = false;
let _loadPromise = null;

// CM6 modules (populated after load)
let CM = {};  // { EditorView, basicSetup, EditorState, keymap, ... }
let LANGS = {};  // { javascript, html, css, json, python }

// Active editors: Map<containerId, { view, onChange }>
const _editors = new Map();

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/**
 * Lazy-load all CM6 modules from CDN. Call before creating any editor.
 * Returns true if loaded, false if CDN unavailable.
 */
export async function load() {
  if (_loaded) return true;
  if (_loadPromise) return _loadPromise;

  _loadPromise = (async () => {
    try {
      // Load all CM6 packages in parallel from esm.sh.
      // esm.sh deduplicates @codemirror/state across all packages automatically.
      const [
        coreMod, stateMod, viewMod, cmdMod, autoMod, lintMod, langMod, searchMod,
        jsMod, htmlMod, cssMod, jsonMod, pyMod, themeMod,
      ] = await Promise.all([
        import(IMPORTS.core),
        import(IMPORTS.state),
        import(IMPORTS.view),
        import(IMPORTS.commands),
        import(IMPORTS.autocomplete),
        import(IMPORTS.lint),
        import(IMPORTS.language),
        import(IMPORTS.search),
        import(IMPORTS.langJs),
        import(IMPORTS.langHtml),
        import(IMPORTS.langCss),
        import(IMPORTS.langJson),
        import(IMPORTS.langPython),
        import(IMPORTS.themeDark),
      ]);

      CM = {
        // Core (from codemirror package)
        basicSetup: coreMod.basicSetup,
        minimalSetup: coreMod.minimalSetup,
        // View (from @codemirror/view)
        EditorView: viewMod.EditorView,
        keymap: viewMod.keymap,
        Decoration: viewMod.Decoration,
        ViewPlugin: viewMod.ViewPlugin,
        WidgetType: viewMod.WidgetType,
        // State (from @codemirror/state)
        EditorState: stateMod.EditorState,
        StateField: stateMod.StateField,
        StateEffect: stateMod.StateEffect,
        Compartment: stateMod.Compartment,
        // Commands (from @codemirror/commands)
        indentWithTab: cmdMod.indentWithTab,
        defaultKeymap: cmdMod.defaultKeymap,
        historyKeymap: cmdMod.historyKeymap,
        history: cmdMod.history,
        // Search (from @codemirror/search)
        searchKeymap: searchMod.searchKeymap,
        // Autocomplete (from @codemirror/autocomplete)
        autocompletion: autoMod.autocompletion,
        completionKeymap: autoMod.completionKeymap,
        closeBrackets: autoMod.closeBrackets,
        closeBracketsKeymap: autoMod.closeBracketsKeymap,
        // Lint (from @codemirror/lint)
        linter: lintMod.linter,
        lintGutter: lintMod.lintGutter,
        // Language (from @codemirror/language)
        indentOnInput: langMod.indentOnInput,
        bracketMatching: langMod.bracketMatching,
        foldGutter: langMod.foldGutter,
        foldKeymap: langMod.foldKeymap,
        syntaxHighlighting: langMod.syntaxHighlighting,
        defaultHighlightStyle: langMod.defaultHighlightStyle,
        // Theme
        oneDark: themeMod.oneDark,
      };

      // Verify critical exports
      if (!CM.EditorView || !CM.EditorState) {
        throw new Error('Core CM6 exports missing — CDN bundle may be incomplete');
      }

      LANGS = {
        javascript: jsMod.javascript,
        html: htmlMod.html,
        css: cssMod.css,
        json: jsonMod.json,
        python: pyMod.python,
      };

      _loaded = true;
      return true;
    } catch (err) {
      console.warn('[CM6] Failed to load from CDN:', err.message);
      _loadPromise = null;
      return false;
    }
  })();

  return _loadPromise;
}

export function isLoaded() {
  return _loaded;
}

// ---------------------------------------------------------------------------
// Language Resolution
// ---------------------------------------------------------------------------

const LANG_MAP = {
  javascript: 'javascript', js: 'javascript', jsx: 'javascript',
  typescript: 'javascript', ts: 'javascript', tsx: 'javascript', // JS mode handles TS with jsx()
  html: 'html', htm: 'html', markup: 'html',
  css: 'css', scss: 'css',
  json: 'json',
  python: 'python', py: 'python',
};

function _getLanguage(lang) {
  const key = LANG_MAP[lang?.toLowerCase()] || null;
  if (!key || !LANGS[key]) return null;
  return LANGS[key]();
}

// ---------------------------------------------------------------------------
// Theme (matches Augmentum dark theme)
// ---------------------------------------------------------------------------

function _buildTheme() {
  return CM.EditorView.theme({
    '&': {
      fontSize: '12px',
      fontFamily: 'var(--font-mono)',
      height: '100%',
    },
    '.cm-content': {
      padding: '8px 0',
      fontFamily: 'var(--font-mono)',
      caretColor: 'var(--text)',
    },
    '.cm-gutters': {
      background: 'var(--bg-elevated)',
      borderRight: '1px solid var(--border)',
      color: 'var(--text-muted)',
      fontFamily: 'var(--font-mono)',
      fontSize: '11px',
    },
    '.cm-activeLineGutter': {
      background: 'color-mix(in srgb, var(--accent) 10%, transparent)',
    },
    '.cm-activeLine': {
      background: 'color-mix(in srgb, var(--accent) 5%, transparent)',
    },
    '.cm-cursor': {
      borderLeftColor: 'var(--text)',
    },
    '.cm-selectionBackground': {
      background: 'color-mix(in srgb, var(--accent) 25%, transparent) !important',
    },
    '.cm-matchingBracket': {
      background: 'color-mix(in srgb, var(--accent) 30%, transparent)',
      outline: '1px solid color-mix(in srgb, var(--accent) 50%, transparent)',
    },
    '.cm-foldGutter .cm-gutterElement': {
      cursor: 'pointer',
      padding: '0 4px',
    },
    '.cm-tooltip': {
      background: 'var(--bg-elevated)',
      border: '1px solid var(--border)',
      borderRadius: '6px',
      boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
    },
    '.cm-tooltip-autocomplete': {
      background: 'var(--bg-elevated)',
    },
    '.cm-tooltip-autocomplete > ul > li[aria-selected]': {
      background: 'color-mix(in srgb, var(--accent) 15%, transparent)',
    },
    '.cm-panels': {
      background: 'var(--bg-elevated)',
      borderBottom: '1px solid var(--border)',
    },
    '.cm-search': {
      fontSize: '12px',
    },
    '.cm-diagnostic-error': {
      borderBottom: '2px solid var(--error)',
    },
    '.cm-diagnostic-warning': {
      borderBottom: '2px solid var(--warning, #f59e0b)',
    },
    '.cm-scroller': {
      overflow: 'auto',
      scrollbarWidth: 'thin',
      scrollbarColor: 'var(--scrollbar-thumb) var(--scrollbar-track)',
    },
  }, { dark: true });
}

// ---------------------------------------------------------------------------
// Editor Creation
// ---------------------------------------------------------------------------

/**
 * Create a CM6 editor in the given container element.
 *
 * @param {HTMLElement} container — DOM element to mount into (will be cleared)
 * @param {object} options
 * @param {string} options.content — initial code content
 * @param {string} options.language — language key (js, html, css, python, json)
 * @param {function} options.onChange — called on every edit: (content: string) => void
 * @param {function} options.onSave — called on Ctrl+S: () => void
 * @param {boolean} options.readOnly — make editor read-only
 * @returns {string} editor ID for future operations
 */
export function create(container, options = {}) {
  if (!_loaded) throw new Error('CM6 not loaded — call load() first');

  const id = container.id || `cm-${Date.now()}`;

  // Destroy existing editor in this container
  if (_editors.has(id)) destroy(id);

  const langCompartment = new CM.Compartment();
  const lang = _getLanguage(options.language);

  // Build custom keymap for Ctrl+S
  const customKeymap = [];
  if (options.onSave) {
    customKeymap.push({
      key: 'Mod-s',
      run: () => { options.onSave(); return true; },
    });
  }

  // Ghost text Tab acceptance keymap — Tab accepts ghost text if showing, else indent
  const ghostKeymap = [{
    key: 'Tab',
    run: (view) => {
      const editorId = container.id || id;
      if (_ghostState.has(editorId)) {
        acceptGhostText(editorId);
        return true;
      }
      return false; // fall through to indentWithTab
    },
  }, {
    key: 'Escape',
    run: (view) => {
      const editorId = container.id || id;
      if (_ghostState.has(editorId)) {
        dismissGhostText(editorId);
        return true;
      }
      return false;
    },
  }];

  // Initialize ghost text extension
  const ghostExt = _initGhostExtension();

  // Additional completion source from CodeMind AST declarations.
  // Registered via languageData facet so it adds to (not replaces) basicSetup's completions.
  // CodeMind completion source — adds AST-extracted declarations to autocomplete.
  // We use a simple completion function, not autocompletion() (which would conflict
  // with basicSetup's built-in autocomplete). Instead we register via languageData.
  const codeMindCompletionSource = [];  // Will be empty if no completion needed

  // CodeMind linter — reads diagnostics from _pendingDiagnostics (set externally)
  const codeMindLinter = CM.linter ? CM.linter((view) => {
    return _pendingDiagnostics.get(id) || [];
  }, { delay: 0 }) : [];

  // Build extensions defensively — skip any that aren't available.
  // Note: basicSetup already includes history, bracketMatching, foldGutter,
  // indentOnInput, autocompletion, closeBrackets, search, etc.
  // We only add things NOT in basicSetup: ghost text, CodeMind linter, theme, lang.
  const extensions = [
    CM.basicSetup,
    CM.lintGutter ? CM.lintGutter() : [],
    codeMindLinter,
    ghostExt,
    codeMindCompletionSource,
    langCompartment.of(lang ? lang : []),
    CM.keymap.of([
      ...ghostKeymap,
      ...customKeymap,
      ...(CM.indentWithTab ? [CM.indentWithTab] : []),
    ]),
    _buildTheme(),
    CM.oneDark,
    CM.EditorView.lineWrapping,
  ].filter(Boolean);

  // onChange listener
  if (options.onChange) {
    extensions.push(CM.EditorView.updateListener.of((update) => {
      if (update.docChanged) {
        options.onChange(update.state.doc.toString());
      }
    }));
  }

  // Read-only
  if (options.readOnly) {
    extensions.push(CM.EditorState.readOnly.of(true));
  }

  container.innerHTML = '';

  const view = new CM.EditorView({
    state: CM.EditorState.create({
      doc: options.content || '',
      extensions,
    }),
    parent: container,
  });

  _editors.set(id, { view, langCompartment, container });
  return id;
}

/**
 * Destroy an editor instance.
 */
export function destroy(id) {
  const editor = _editors.get(id);
  if (!editor) return;
  editor.view.destroy();
  _editors.delete(id);
}

// ---------------------------------------------------------------------------
// Content Access
// ---------------------------------------------------------------------------

/**
 * Get the current content of an editor.
 */
export function getContent(id) {
  const editor = _editors.get(id);
  return editor ? editor.view.state.doc.toString() : '';
}

/**
 * Set the content of an editor (replaces all).
 */
export function setContent(id, content) {
  const editor = _editors.get(id);
  if (!editor) return;
  const { view } = editor;
  view.dispatch({
    changes: { from: 0, to: view.state.doc.length, insert: content },
  });
}

/**
 * Set the language mode of an editor.
 */
export function setLanguage(id, lang) {
  const editor = _editors.get(id);
  if (!editor) return;
  const langExt = _getLanguage(lang);
  editor.view.dispatch({
    effects: editor.langCompartment.reconfigure(langExt ? langExt : []),
  });
}

// ---------------------------------------------------------------------------
// Cursor & Selection
// ---------------------------------------------------------------------------

/**
 * Get cursor position as { line, col } (1-based).
 */
export function getCursor(id) {
  const editor = _editors.get(id);
  if (!editor) return { line: 1, col: 1 };
  const pos = editor.view.state.selection.main.head;
  const line = editor.view.state.doc.lineAt(pos);
  return { line: line.number, col: pos - line.from + 1 };
}

/**
 * Set cursor position (1-based line and col).
 */
export function setCursor(id, line, col = 1) {
  const editor = _editors.get(id);
  if (!editor) return;
  const lineObj = editor.view.state.doc.line(Math.min(line, editor.view.state.doc.lines));
  const pos = lineObj.from + Math.min(col - 1, lineObj.length);
  // scrollIntoView centers the target line — without it a jump to a
  // match/diagnostic deep in a long file moves the cursor off-screen
  // and the user sees no change. Harmless for near-top targets.
  editor.view.dispatch({ selection: { anchor: pos }, scrollIntoView: true });
  editor.view.focus();
}

/**
 * Focus the editor.
 */
export function focus(id) {
  const editor = _editors.get(id);
  if (editor) editor.view.focus();
}

// ---------------------------------------------------------------------------
// Diagnostics (CodeMind integration)
// ---------------------------------------------------------------------------

// Store pending diagnostics per editor — the linter callback reads from here
const _pendingDiagnostics = new Map(); // id → Diagnostic[]

/**
 * Set lint diagnostics from CodeMind AST errors.
 * Uses a linter() callback registered during editor creation — we store the
 * diagnostics and request a lint refresh.
 * @param {string} id — editor ID
 * @param {Array} errors — [{startRow, startCol, endRow, endCol, message, type}]
 */
export function setDiagnosticsFromCodeMind(id, errors) {
  const editor = _editors.get(id);
  if (!editor) return;

  const doc = editor.view.state.doc;
  const diagnostics = [];

  for (const err of errors) {
    try {
      const fromLine = doc.line(err.startRow + 1);
      const toLine = doc.line(Math.min(err.endRow + 1, doc.lines));
      const from = fromLine.from + Math.min(err.startCol, fromLine.length);
      const to = toLine.from + Math.min(err.endCol, toLine.length);
      if (from <= to && from >= 0 && to <= doc.length) {
        diagnostics.push({
          from,
          to: Math.max(to, from + 1),
          severity: err.type === 'missing' ? 'warning' : 'error',
          message: err.message,
        });
      }
    } catch { /* skip invalid positions */ }
  }

  _pendingDiagnostics.set(id, diagnostics);
  // Force lint refresh by dispatching a no-op change
  // CM6's linter will re-call our callback which reads from _pendingDiagnostics
  editor.view.dispatch({});
}

// ---------------------------------------------------------------------------
// Ghost Text — inline suggestion rendered as CM6 widget decoration
// ---------------------------------------------------------------------------
// Shows semi-transparent italic text after the cursor. Tab to accept.
// Implemented as a StateField + Decoration.widget so CM6 manages lifecycle.

const _ghostEffect = { set: null, clear: null };
let _ghostField = null;

function _initGhostExtension() {
  if (_ghostField) return _ghostField;

  _ghostEffect.set = CM.StateEffect.define();
  _ghostEffect.clear = CM.StateEffect.define();

  // Widget that renders the ghost text as a DOM span
  class GhostWidget extends CM.WidgetType {
    constructor(text) { super(); this.text = text; }
    toDOM() {
      const span = document.createElement('span');
      span.className = 'cm-ghost-text';
      span.textContent = this.text;
      return span;
    }
    eq(other) { return this.text === other.text; }
  }

  _ghostField = CM.StateField.define({
    create() { return CM.Decoration.none; },
    update(deco, tr) {
      for (const effect of tr.effects) {
        if (effect.is(_ghostEffect.set)) {
          const { pos, text } = effect.value;
          const widget = CM.Decoration.widget({
            widget: new GhostWidget(text),
            side: 1,
          });
          return CM.Decoration.set([widget.range(pos)]);
        }
        if (effect.is(_ghostEffect.clear)) {
          return CM.Decoration.none;
        }
      }
      // Clear ghost on any document change (user typed something)
      if (tr.docChanged) return CM.Decoration.none;
      return deco;
    },
    provide: (f) => CM.EditorView.decorations.from(f),
  });

  return _ghostField;
}

/**
 * Get the ghost text extension to include in editor setup.
 * Must be called during editor creation.
 */
/**
 * Show ghost text suggestion at current cursor position.
 */
export function showGhostText(id, text) {
  const editor = _editors.get(id);
  if (!editor || !text || !_ghostEffect.set) return;

  const pos = editor.view.state.selection.main.head;
  _ghostState.set(id, { text, pos });

  editor.view.dispatch({
    effects: _ghostEffect.set.of({ pos, text }),
  });
}

const _ghostState = new Map();

/**
 * Dismiss ghost text.
 */
export function dismissGhostText(id) {
  const editor = _editors.get(id);
  _ghostState.delete(id);
  if (editor && _ghostEffect.clear) {
    editor.view.dispatch({ effects: _ghostEffect.clear.of(null) });
  }
}

/**
 * Accept ghost text — insert it at cursor.
 */
export function acceptGhostText(id) {
  const editor = _editors.get(id);
  const ghost = _ghostState.get(id);
  if (!editor || !ghost) return;

  // Clear the ghost decoration first
  if (_ghostEffect.clear) {
    editor.view.dispatch({ effects: _ghostEffect.clear.of(null) });
  }

  // Insert the text
  editor.view.dispatch({
    changes: { from: ghost.pos, insert: ghost.text },
    selection: { anchor: ghost.pos + ghost.text.length },
  });
  _ghostState.delete(id);
}

/**
 * Check if ghost text is currently showing.
 */
export function hasGhostText(id) {
  return _ghostState.has(id);
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/**
 * Get all loaded language keys.
 */
export function getLanguages() {
  return Object.keys(LANGS);
}

/**
 * Get the EditorView instance (escape hatch for advanced operations).
 */
export function getView(id) {
  return _editors.get(id)?.view || null;
}
