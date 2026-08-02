/**
 * CodeMind — AST-powered code intelligence for Augmentum.
 *
 * Lazy-loads web-tree-sitter + per-language grammars from CDN.
 * Provides: parsing, syntax error detection, bracket matching,
 * scope extraction (for LLM context compression), and diagnostics.
 *
 * Designed for real-time editor use — all operations are sub-5ms
 * after initial grammar load.
 */

// ---------------------------------------------------------------------------
// CDN URLs — tree-sitter core + grammar WASMs
// ---------------------------------------------------------------------------

const TS_VERSION = '0.24.7';
const TS_BASE = `https://cdn.jsdelivr.net/npm/web-tree-sitter@${TS_VERSION}`;
const TS_WASM_CDN = `${TS_BASE}/tree-sitter.wasm`;

// Grammar WASM URLs — loaded on demand per language
const GRAMMAR_URLS = {
  javascript: `https://cdn.jsdelivr.net/npm/tree-sitter-wasms@0.1.11/out/tree-sitter-javascript.wasm`,
  typescript: `https://cdn.jsdelivr.net/npm/tree-sitter-wasms@0.1.11/out/tree-sitter-typescript.wasm`,
  html:       `https://cdn.jsdelivr.net/npm/tree-sitter-wasms@0.1.11/out/tree-sitter-html.wasm`,
  css:        `https://cdn.jsdelivr.net/npm/tree-sitter-wasms@0.1.11/out/tree-sitter-css.wasm`,
  python:     `https://cdn.jsdelivr.net/npm/tree-sitter-wasms@0.1.11/out/tree-sitter-python.wasm`,
  json:       `https://cdn.jsdelivr.net/npm/tree-sitter-wasms@0.1.11/out/tree-sitter-json.wasm`,
};

// Map file extensions / language labels to grammar keys
const LANG_MAP = {
  js: 'javascript', jsx: 'javascript', javascript: 'javascript',
  ts: 'typescript', tsx: 'typescript', typescript: 'typescript',
  html: 'html', htm: 'html', markup: 'html',
  css: 'css', scss: 'css',
  py: 'python', python: 'python',
  json: 'json',
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _TreeSitter = null;      // The TreeSitter class (loaded from CDN)
let _initPromise = null;     // Singleton init promise
let _parsers = {};           // lang → TreeSitter.Parser instance
let _languages = {};         // lang → TreeSitter.Language instance
let _trees = new Map();      // fileKey → { tree, lang, version }
let _loadingLangs = {};      // lang → Promise (dedup concurrent loads)

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/**
 * Initialize tree-sitter core. Call once — subsequent calls are no-ops.
 * Returns true if ready, false if CDN unavailable.
 */
export async function init() {
  if (_TreeSitter) return true;
  if (_initPromise) return _initPromise;

  _initPromise = (async () => {
    try {
      // web-tree-sitter is a UMD module — load via script tag, not ES import.
      // After loading, it exposes window.TreeSitter (or Module for older versions).
      if (!window.TreeSitter) {
        await new Promise((resolve, reject) => {
          const script = document.createElement('script');
          script.src = `${TS_BASE}/tree-sitter.js`;
          script.onload = resolve;
          script.onerror = () => reject(new Error('Failed to load tree-sitter.js'));
          document.head.appendChild(script);
        });
      }

      _TreeSitter = window.TreeSitter;
      if (!_TreeSitter || typeof _TreeSitter.init !== 'function') {
        throw new Error('TreeSitter global not found after script load');
      }

      await _TreeSitter.init({
        locateFile: (scriptName) => {
          if (scriptName.includes('tree-sitter.wasm')) return TS_WASM_CDN;
          return `${TS_BASE}/${scriptName}`;
        },
      });

      return true;
    } catch (err) {
      console.warn('[CodeMind] tree-sitter init failed:', err.message);
      _TreeSitter = null;
      _initPromise = null;
      return false;
    }
  })();

  return _initPromise;
}

/**
 * Check if CodeMind is initialized and ready.
 */
export function isReady() {
  return _TreeSitter !== null;
}

// ---------------------------------------------------------------------------
// Language Loading
// ---------------------------------------------------------------------------

/**
 * Load a grammar for a language. Returns the Language object or null.
 * Deduplicates concurrent loads for the same language.
 */
async function _loadLanguage(lang) {
  const key = LANG_MAP[lang?.toLowerCase()] || lang?.toLowerCase();
  if (!key || !GRAMMAR_URLS[key]) return null;
  if (_languages[key]) return _languages[key];
  if (_loadingLangs[key]) return _loadingLangs[key];

  _loadingLangs[key] = (async () => {
    try {
      const language = await _TreeSitter.Language.load(GRAMMAR_URLS[key]);
      _languages[key] = language;

      // Create a parser for this language
      const parser = new _TreeSitter();
      parser.setLanguage(language);
      _parsers[key] = parser;

      return language;
    } catch (err) {
      console.warn(`[CodeMind] Failed to load grammar for ${key}:`, err.message);
      return null;
    } finally {
      delete _loadingLangs[key];
    }
  })();

  return _loadingLangs[key];
}

/**
 * Resolve a language string to a grammar key.
 */
export function resolveLanguage(lang) {
  return LANG_MAP[lang?.toLowerCase()] || lang?.toLowerCase() || null;
}

/**
 * Check if a language grammar is loaded.
 */
export function isLanguageLoaded(lang) {
  const key = resolveLanguage(lang);
  return key ? !!_languages[key] : false;
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/**
 * Parse code and return the tree. Uses incremental parsing if a previous
 * tree exists for the same fileKey.
 *
 * @param {string} code - Source code to parse
 * @param {string} lang - Language identifier (e.g., 'javascript', 'js', 'python')
 * @param {string} [fileKey] - Unique key for caching (e.g., file path). Enables incremental parsing.
 * @returns {object|null} - { tree, errors, lang } or null if grammar unavailable
 */
export async function parse(code, lang, fileKey = null) {
  if (!_TreeSitter) {
    const ok = await init();
    if (!ok) return null;
  }

  const key = resolveLanguage(lang);
  if (!key) return null;

  const language = _languages[key] || await _loadLanguage(key);
  if (!language) return null;

  const parser = _parsers[key];
  if (!parser) return null;

  // Incremental parsing: pass previous tree if same file
  let oldTree = null;
  if (fileKey && _trees.has(fileKey)) {
    const cached = _trees.get(fileKey);
    if (cached.lang === key) oldTree = cached.tree;
  }

  const tree = parser.parse(code, oldTree);

  // Cache for incremental parsing
  if (fileKey) {
    _trees.set(fileKey, { tree, lang: key, version: Date.now() });
  }

  const errors = _collectErrors(tree.rootNode);

  return { tree, errors, lang: key };
}

/**
 * Synchronous parse — only works if grammar is already loaded.
 * Use for real-time editing where async is not acceptable.
 */
export function parseSync(code, lang, fileKey = null) {
  const key = resolveLanguage(lang);
  if (!key || !_parsers[key]) return null;

  const parser = _parsers[key];
  let oldTree = null;
  if (fileKey && _trees.has(fileKey)) {
    const cached = _trees.get(fileKey);
    if (cached.lang === key) oldTree = cached.tree;
  }

  const tree = parser.parse(code, oldTree);

  if (fileKey) {
    _trees.set(fileKey, { tree, lang: key, version: Date.now() });
  }

  const errors = _collectErrors(tree.rootNode);
  return { tree, errors, lang: key };
}

// ---------------------------------------------------------------------------
// Error Detection
// ---------------------------------------------------------------------------

/**
 * Collect syntax errors from a tree-sitter AST.
 * tree-sitter marks error nodes with type "ERROR" or isMissing().
 */
function _collectErrors(node, errors = []) {
  const isMissing = typeof node.isMissing === 'function' ? node.isMissing() : node.isMissing;
  const isError = node.type === 'ERROR' || node.type === 'MISSING' || isMissing;
  if (isError) {
    errors.push({
      type: isMissing ? 'missing' : 'error',
      message: isMissing
        ? `Missing ${node.type === 'ERROR' ? 'syntax element' : node.type}`
        : `Syntax error: unexpected ${(node.text || '').slice(0, 30)}`,
      startRow: node.startPosition.row,
      startCol: node.startPosition.column,
      endRow: node.endPosition.row,
      endCol: node.endPosition.column,
    });
  }

  for (let i = 0; i < node.childCount; i++) {
    _collectErrors(node.child(i), errors);
  }

  return errors;
}

/**
 * Get syntax errors for code without caching.
 */
export async function getErrors(code, lang) {
  const result = await parse(code, lang);
  return result ? result.errors : [];
}

// ---------------------------------------------------------------------------
// Bracket Matching (AST-aware)
// ---------------------------------------------------------------------------

/**
 * Find the matching bracket at the given position.
 * Uses the AST to find the enclosing paired node — more reliable
 * than character scanning (handles strings, comments, nested structures).
 *
 * @returns {{ row, col }|null} Position of matching bracket, or null.
 */
export function findBracketMatch(code, row, col, lang, fileKey = null) {
  const key = resolveLanguage(lang);
  let cached = fileKey ? _trees.get(fileKey) : null;
  if (!cached || cached.lang !== key) {
    // Try sync parse if grammar is loaded
    const result = parseSync(code, lang, fileKey);
    if (!result) return null;
    cached = _trees.get(fileKey) || { tree: result.tree };
  }

  const tree = cached.tree;
  const point = { row, column: col };
  let node = tree.rootNode.descendantForPosition(point);
  if (!node) return null;

  // Walk up to find a bracket-containing parent
  const BRACKET_TYPES = new Set([
    'arguments', 'formal_parameters', 'parenthesized_expression',
    'object', 'array', 'dictionary', 'list', 'tuple',
    'block', 'statement_block', 'compound_statement',
    'template_string', 'string', 'interpolation',
    'jsx_element', 'jsx_self_closing_element',
    'element', // HTML
  ]);

  // Check if cursor is on a bracket character
  const ch = code.split('\n')[row]?.[col];
  const OPENERS = new Set(['(', '[', '{', '<']);
  const CLOSERS = new Set([')', ']', '}', '>']);

  if (!OPENERS.has(ch) && !CLOSERS.has(ch)) {
    // Check one char before cursor too
    const prevCh = col > 0 ? code.split('\n')[row]?.[col - 1] : null;
    if (!OPENERS.has(prevCh) && !CLOSERS.has(prevCh)) return null;
  }

  // Find the enclosing bracketed node
  let current = node;
  while (current) {
    if (BRACKET_TYPES.has(current.type) || current.type.includes('block') || current.type.includes('body')) {
      const start = current.startPosition;
      const end = current.endPosition;

      // If cursor is near start, return end position
      if (row === start.row && Math.abs(col - start.column) <= 1) {
        // Return the position of the closing bracket
        return { row: end.row, col: Math.max(0, end.column - 1) };
      }
      // If cursor is near end, return start position
      if (row === end.row && Math.abs(col - (end.column - 1)) <= 1) {
        return { row: start.row, col: start.column };
      }
    }
    current = current.parent;
  }

  return null;
}

// ---------------------------------------------------------------------------
// Scope Extraction (for LLM context compression)
// ---------------------------------------------------------------------------

/**
 * Extract the enclosing function/class scope at the given cursor position.
 * Returns a compact context object suitable for sending to an LLM.
 *
 * @returns {{ scopeText, scopeName, scopeType, startLine, endLine, imports }|null}
 */
export function getScopeAt(code, row, lang, fileKey = null) {
  const key = resolveLanguage(lang);
  let cached = fileKey ? _trees.get(fileKey) : null;
  if (!cached || cached.lang !== key) {
    const result = parseSync(code, lang, fileKey);
    if (!result) return null;
    cached = { tree: result.tree };
  }

  const tree = cached.tree;
  const point = { row, column: 0 };
  let node = tree.rootNode.descendantForPosition(point);

  // Walk up to find the enclosing function or class
  const SCOPE_TYPES = new Set([
    'function_declaration', 'function_definition', 'method_definition',
    'arrow_function', 'generator_function_declaration',
    'class_declaration', 'class_definition',
    'function_item', 'impl_item', // Rust
    'func_declaration', // Go
  ]);

  let scopeNode = null;
  let current = node;
  while (current) {
    if (SCOPE_TYPES.has(current.type)) {
      scopeNode = current;
      break;
    }
    current = current.parent;
  }

  // Extract imports from the top of the file
  const imports = [];
  for (let i = 0; i < tree.rootNode.childCount; i++) {
    const child = tree.rootNode.child(i);
    if (child.type === 'import_statement' || child.type === 'import_declaration' ||
        child.type === 'import_from_statement' || child.type.startsWith('import')) {
      imports.push(child.text);
    }
  }

  if (scopeNode) {
    // Find the scope name
    let scopeName = '';
    const nameNode = scopeNode.childForFieldName('name');
    if (nameNode) scopeName = nameNode.text;

    return {
      scopeText: scopeNode.text,
      scopeName,
      scopeType: scopeNode.type,
      startLine: scopeNode.startPosition.row + 1,
      endLine: scopeNode.endPosition.row + 1,
      imports,
    };
  }

  // No enclosing scope — return the full file with a reasonable window
  const lines = code.split('\n');
  const windowStart = Math.max(0, row - 20);
  const windowEnd = Math.min(lines.length, row + 20);

  return {
    scopeText: lines.slice(windowStart, windowEnd).join('\n'),
    scopeName: '(module level)',
    scopeType: 'module',
    startLine: windowStart + 1,
    endLine: windowEnd,
    imports,
  };
}

/**
 * Extract all top-level declarations (function names, class names, exports)
 * from the AST. Useful for building file summaries for multi-file context.
 */
export function getDeclarations(code, lang, fileKey = null) {
  const key = resolveLanguage(lang);
  let cached = fileKey ? _trees.get(fileKey) : null;
  if (!cached || cached.lang !== key) {
    const result = parseSync(code, lang, fileKey);
    if (!result) return [];
    cached = { tree: result.tree };
  }

  const tree = cached.tree;
  const declarations = [];

  for (let i = 0; i < tree.rootNode.childCount; i++) {
    const child = tree.rootNode.child(i);
    const type = child.type;
    const nameNode = child.childForFieldName('name');

    if (nameNode) {
      declarations.push({
        type: type.replace(/_declaration|_definition|_statement/, ''),
        name: nameNode.text,
        line: child.startPosition.row + 1,
        signature: child.text.split('\n')[0].slice(0, 120),
      });
    } else if (type === 'expression_statement') {
      // Handle: export default, module.exports, window.X = ...
      const text = child.text;
      if (text.startsWith('export') || text.includes('module.exports') || text.includes('window.')) {
        declarations.push({
          type: 'export',
          name: text.slice(0, 60),
          line: child.startPosition.row + 1,
          signature: text.split('\n')[0].slice(0, 120),
        });
      }
    }
  }

  return declarations;
}

// ---------------------------------------------------------------------------
// Diagnostics Rendering (HTML overlay helpers)
// ---------------------------------------------------------------------------

/**
 * Build error squiggly underline data for the editor.
 * Returns an array of { line, startCol, endCol, message, severity } suitable
 * for rendering as styled overlays.
 */
export function getDiagnostics(code, lang, fileKey = null) {
  const result = parseSync(code, lang, fileKey);
  if (!result) return [];

  return result.errors.map(e => ({
    line: e.startRow + 1,
    startCol: e.startCol,
    endCol: e.endCol,
    message: e.message,
    severity: e.type === 'missing' ? 'warning' : 'error',
  }));
}

// ---------------------------------------------------------------------------
// Validation (for LLM output checking)
// ---------------------------------------------------------------------------

/**
 * Validate code generated by the LLM. Returns { valid, errors }.
 * Used by the Repair Cascade to catch broken generations before
 * showing them to the user.
 */
export async function validate(code, lang) {
  const result = await parse(code, lang);
  if (!result) return { valid: true, errors: [] }; // can't validate, assume ok

  return {
    valid: result.errors.length === 0,
    errors: result.errors,
  };
}

// ---------------------------------------------------------------------------
// Cache Management
// ---------------------------------------------------------------------------

/**
 * Clear cached tree for a file (e.g., when file is closed).
 */
export function clearCache(fileKey) {
  _trees.delete(fileKey);
}

/**
 * Clear all cached data.
 */
export function clearAll() {
  _trees.clear();
}

/**
 * Get supported languages.
 */
export function getSupportedLanguages() {
  return Object.keys(GRAMMAR_URLS);
}

/**
 * Get loaded languages.
 */
export function getLoadedLanguages() {
  return Object.keys(_languages);
}
