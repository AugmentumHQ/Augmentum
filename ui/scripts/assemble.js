/* ==========================================================================
   Augmentum — Assemble Module
   Shared project assembly and preview helpers used by chat.js and workspace.
   ========================================================================== */

// Module-level source map from the last assembleProject() call
let lastSourceMap = [];

/**
 * Returns the source map from the most recent assembleProject() call.
 */
export function getLastSourceMap() {
  return lastSourceMap;
}

/**
 * Assembles project files into a single runnable HTML page.
 * Also builds a source map: array of { file, fileLineStart, assembledLineStart, lineCount }
 * so error line numbers in the assembled output can be traced back to source files.
 */
export function assembleProject(files) {
  const entry = files.find(f => f.role === 'entry');
  if (!entry) return null;

  let html = entry.content;

  const styles = files.filter(f => f.role === 'style');
  if (styles.length > 0) {
    const styleBlock = styles.map(f => `/* ${f.path} */\n${f.content}`).join('\n\n');
    if (html.includes('</head>')) {
      html = html.replace('</head>', `<style>\n${styleBlock}\n</style>\n</head>`);
    } else {
      html = `<style>\n${styleBlock}\n</style>\n` + html;
    }
  }

  const scripts = files.filter(f => f.role === 'script');
  const modules = files.filter(f => f.role === 'module');
  if (scripts.length > 0 || modules.length > 0) {
    let scriptBlock = '';
    for (const f of scripts) scriptBlock += `<script>\n/* ${f.path} */\n${f.content.replace(/<\/script>/gi, '<\\/script>')}\n</script>\n`;
    for (const f of modules) scriptBlock += `<script type="module">\n/* ${f.path} */\n${f.content.replace(/<\/script>/gi, '<\\/script>')}\n</script>\n`;
    if (html.includes('</body>')) {
      html = html.replace('</body>', `${scriptBlock}</body>`);
    } else {
      html += `\n${scriptBlock}`;
    }
  }

  const dataFiles = files.filter(f => f.role === 'data');
  if (dataFiles.length > 0) {
    let dataBlock = '<script>\n';
    for (const f of dataFiles) {
      const varName = f.path.replace(/\.\w+$/, '').replace(/[^a-zA-Z0-9]/g, '_');
      dataBlock += `const ${varName} = ${f.content};\n`;
    }
    dataBlock += '</script>\n';
    if (html.includes('</head>')) {
      html = html.replace('</head>', `${dataBlock}</head>`);
    } else {
      html = dataBlock + html;
    }
  }

  // Build source map by finding where each file's content appears in the assembled HTML
  const sourceMap = [];
  const assembledLines = html.split('\n');
  for (const file of files) {
    if (file.role === 'entry' || !file.content) continue;
    // Find where this file's content starts in the assembled output (look for the /* filename */ comment)
    for (let i = 0; i < assembledLines.length; i++) {
      if (assembledLines[i].includes(`/* ${file.path} */`)) {
        sourceMap.push({
          file: file.path,
          role: file.role,
          assembledLineStart: i + 2, // +1 for 0-index, +1 for the comment line
          fileLineStart: 1,
          lineCount: file.content.split('\n').length,
        });
        break;
      }
    }
  }

  lastSourceMap = sourceMap;
  return html;
}

function _isRelativeUrl(url) {
  return url && !(/^(https?:)?\/\/|^data:|^blob:/i.test(url));
}

function _stripRelativeResources(html) {
  html = html.replace(/<script\b[^>]*\bsrc\s*=\s*["']([^"']+)["'][^>]*>\s*<\/script>/gi,
    (match, src) => _isRelativeUrl(src) ? `<!-- stripped: ${src} -->` : match);
  html = html.replace(/<link\b[^>]*\brel\s*=\s*["'](?:stylesheet|preload)["'][^>]*\bhref\s*=\s*["']([^"']+)["'][^>]*\/?>/gi,
    (match, href) => _isRelativeUrl(href) ? `<!-- stripped: ${href} -->` : match);
  html = html.replace(/<link\b[^>]*\bhref\s*=\s*["']([^"']+)["'][^>]*\brel\s*=\s*["'](?:stylesheet|preload)["'][^>]*\/?>/gi,
    (match, href) => _isRelativeUrl(href) ? `<!-- stripped: ${href} -->` : match);
  return html;
}

function _getPreviewThemeInfo() {
  const theme = document.documentElement.getAttribute('data-theme') || 'light';
  return { theme, isDark: theme === 'dark' };
}

function _buildStoragePolyfillScript({ syncToParent } = {}) {
  const initResponseListener = syncToParent
    ? `window.addEventListener('message', (e) => {
  if (e.data?.type === 'storage-init-response' && e.data.data) {
    try { Object.assign(window._bridgedStorage, e.data.data); } catch {}
  }
});`
    : '';
  const setItemHook = syncToParent
    ? `parent.postMessage({type:'storage-set', key:k, value:String(v)}, '*');`
    : '';
  const removeItemHook = syncToParent
    ? `parent.postMessage({type:'storage-remove', key:k}, '*');`
    : '';
  const clearHook = syncToParent
    ? `parent.postMessage({type:'storage-clear'}, '*');`
    : '';
  const initHook = syncToParent
    ? `parent.postMessage({type:'storage-init'}, '*');`
    : '';

  return `<script>
${initResponseListener}
window._bridgedStorage = {};
(function installStoragePolyfill() {
  var sandboxed = false;
  try { window.localStorage.getItem('_probe'); } catch { sandboxed = true; }
  if (!sandboxed) return;
  var lsShim = {
    _d: window._bridgedStorage,
    getItem: function(k) { return this._d[k] !== undefined ? this._d[k] : null; },
    setItem: function(k, v) { this._d[k] = String(v); ${setItemHook} },
    removeItem: function(k) { delete this._d[k]; ${removeItemHook} },
    clear: function() { this._d = {}; window._bridgedStorage = this._d; ${clearHook} },
    get length() { return Object.keys(this._d).length; },
    key: function(i) { return Object.keys(this._d)[i] || null; },
  };
  var ssShim = {
    _d: {}, getItem: function(k) { return this._d[k] !== undefined ? this._d[k] : null; },
    setItem: function(k, v) { this._d[k] = String(v); }, removeItem: function(k) { delete this._d[k]; },
    clear: function() { this._d = {}; }, get length() { return Object.keys(this._d).length; },
    key: function(i) { return Object.keys(this._d)[i] || null; },
  };
  var lsInstalled = false;
  try {
    Object.defineProperty(window, 'localStorage', { value: lsShim, configurable: true, writable: true });
    Object.defineProperty(window, 'sessionStorage', { value: ssShim, configurable: true, writable: true });
    lsInstalled = true;
  } catch (e) { /* ignore */ }
  if (!lsInstalled) {
    try { globalThis.localStorage = lsShim; } catch {}
    try { globalThis.sessionStorage = ssShim; } catch {}
  }
  ${initHook}
})();
<\/script>`;
}

function _buildPreviewThemeStyle(theme, isDark, { libraryMode = false } = {}) {
  // Don't override body bg/color/font — that fights apps with their own
  // styling (the agentic-built todo, calc, etc. all set their own theme).
  // Just expose color-scheme for the form-control fallback and ensure the
  // viewport has height so apps using `100vh` measure something.
  // libraryMode/isDark/theme retained for callers + future re-tightening.
  void libraryMode; void isDark; void theme;
  return `<style>
  html, body { height: 100%; margin: 0; }
</style>`;
}

function _wrapPreviewSrcdoc(rawCode, headInsert, theme) {
  let code = _stripRelativeResources(rawCode);
  const isFullHtml = /^\s*<!DOCTYPE|^\s*<html/i.test(code);

  if (isFullHtml) {
    let html = code;
    if (html.includes('</head>')) {
      html = html.replace('</head>', headInsert + '\n</head>');
    } else if (/<html[^>]*>/i.test(html)) {
      html = html.replace(/<html([^>]*)>/i, `<html$1><head>${headInsert}</head>`);
    } else {
      html = headInsert + '\n' + html;
    }
    if (/<html[^>]*>/i.test(html) && !html.includes('data-theme')) {
      html = html.replace(/<html([^>]*)>/i, `<html$1 data-theme="${theme}">`);
    }
    return html;
  }

  return `<!DOCTYPE html>
<html data-theme="${theme}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
${headInsert}
</head>
<body>
${code}
<\/body>
</html>`;
}

/**
 * Builds a sandboxed srcdoc HTML string for previewing raw code in an iframe.
 * Includes error capture bridge, console override, localStorage polyfill,
 * sessionStorage polyfill, and theme-aware styles.
 */
export function buildPreviewSrcdoc(rawCode) {
  const { theme, isDark } = _getPreviewThemeInfo();

  const diagnosticsScript = `<script>
window.onerror = (msg, src, line, col) => {
  parent.postMessage({type:'code-error', detail: msg + (line ? ' (line ' + line + ')' : '')}, '*');
};
window.addEventListener('unhandledrejection', (e) => {
  parent.postMessage({type:'code-error', detail: 'Unhandled: ' + (e.reason?.message || e.reason)}, '*');
});
['log','warn','error','info'].forEach(m => {
  const orig = console[m];
  console[m] = (...args) => {
    parent.postMessage({type:'code-console', level:m, args: args.map(a => {
      try { return typeof a === 'object' ? JSON.stringify(a, null, 2) : String(a); }
      catch { return String(a); }
    })}, '*');
    orig.apply(console, args);
  };
});
window.addEventListener('message', (e) => {
  if (e.data?.type === 'code-ping') parent.postMessage({type:'code-pong'}, '*');
});
<\/script>`;
  const storageScript = _buildStoragePolyfillScript({ syncToParent: true });
  const themeStyle = _buildPreviewThemeStyle(theme, isDark);
  return _wrapPreviewSrcdoc(rawCode, `${diagnosticsScript}\n${storageScript}\n${themeStyle}`, theme);
}

export function buildLibraryPreviewSrcdoc(rawCode) {
  const { theme, isDark } = _getPreviewThemeInfo();
  // No muzzle: silently swallowing every error is what made the "blank
  // thumbnails, no console, no hint" rabbit hole impossible to debug.
  // If a card's app errors, we want it to error out loud in devtools
  // so the next reader actually has a thread to pull on.
  const storageScript = _buildStoragePolyfillScript({ syncToParent: false });
  const themeStyle = _buildPreviewThemeStyle(theme, isDark, { libraryMode: true });
  return _wrapPreviewSrcdoc(rawCode, `${storageScript}\n${themeStyle}`, theme);
}

/**
 * Maps an assembled output line number back to a source file + line.
 * Returns { file, line } or null if not mappable.
 */
export function mapAssembledLineToSource(sourceMap, assembledLine) {
  if (!sourceMap || !sourceMap.length) return null;

  for (const entry of sourceMap) {
    if (assembledLine >= entry.assembledLineStart &&
        assembledLine < entry.assembledLineStart + entry.lineCount) {
      const fileLine = assembledLine - entry.assembledLineStart + 1;
      return {
        file: entry.file,
        line: fileLine,
      };
    }
  }
  return null;
}
