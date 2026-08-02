/* ==========================================================================
   Code Block Actions — Preview, Run, Edit, Versioning, Download
   Extracted from chat.js. Handles all code-block UI interactions that
   don't involve LLM round-trips (those live in code-edit.js).
   ========================================================================== */

import { app, escapeHtml, extractErrorMessage, showToast } from '../app.js';
import { icons } from './constants.js';
import { renderMarkdown, highlightCode, blockFingerprint, safeHighlightElement } from './markdown.js';
import { getSettings } from '../settings.js';
import * as CodeMind from '../codemind.js';
import { buildPreviewSrcdoc as _sharedBuildPreviewSrcdoc } from '../assemble.js';
import { copyToClipboard } from '../clipboard.js';

// Lazy imports from code-edit.js to avoid circular deps.
// We use a getter pattern: call _codeEdit().fn(...)
let _codeEditModule = null;
async function _loadCodeEdit() {
  if (!_codeEditModule) _codeEditModule = await import('./code-edit.js');
  return _codeEditModule;
}
function _codeEdit() { return _codeEditModule; }

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------

/** Registry of active preview iframes for message isolation. */
const _activeIframes = new WeakSet();
const _heartbeatIntervals = new Map(); // iframe -> intervalId

let _previewModalIframe = null;
let _previewModalHandler = null;
let _previewModalHeartbeat = null;
let _previewModalSrcdoc = '';
let _previewModalCodeHeader = null;
let _previewModalErrors = [];
let _previewModalEscHandler = null;

let _fixRetryCount = new WeakMap();

const _pendingBlockEdits = [];

let _pyodide = null;
let _pyodideReady = null;

/** Blocked patterns for client-side Python safety check (mirrors server). */
const _PY_BLOCKED = [
  /\bos\.system\s*\(/, /\bsubprocess\b/, /\b__import__\s*\(/,
  /\bexec\s*\(/, /\beval\s*\(/, /\bcompile\s*\(/, /\bopen\s*\(/,
];

// ---------------------------------------------------------------------------
// Session / block helpers — require sessionStore + getActiveSession from caller
// ---------------------------------------------------------------------------

let _sessionAccessor = null;

/**
 * Provide session accessor functions so code-actions can read/write
 * session state without importing from index.js (which would be circular).
 *
 * Called once during initCodeBlockActions().
 *
 * @param {{ getActiveSession: Function, saveSessions: Function }} accessor
 */
export function setSessionAccessor(accessor) {
  _sessionAccessor = accessor;
}

function _getActiveSession() {
  return _sessionAccessor?.getActiveSession() || null;
}

function _saveSessions() {
  _sessionAccessor?.saveSessions();
}

// ---------------------------------------------------------------------------
// Output Panel
// ---------------------------------------------------------------------------

/**
 * Get or create the output panel below a code block's <pre> element.
 */
export function getOutputPanel(codeHeader) {
  const pre = codeHeader.nextElementSibling;
  if (!pre) return null;
  let panel = pre.nextElementSibling;
  if (panel && panel.classList.contains('code-output-panel')) return panel;

  panel = document.createElement('div');
  panel.className = 'code-output-panel';
  panel.innerHTML = `
    <div class="code-output-header">
      <div class="code-output-tabs">
        <button class="code-output-tab active" data-tab="output">Output</button>
        <button class="code-output-tab" data-tab="errors">Errors</button>
      </div>
      <div class="code-output-actions">
        <button title="Copy output" data-action="copy-output">Copy</button>
        <button title="Clear" data-action="clear-output">Clear</button>
        <button title="Close" data-action="close-output">&times;</button>
      </div>
    </div>
    <div class="code-output-body">
      <div class="code-output-content" data-tab="output"></div>
      <div class="code-output-content" data-tab="errors" hidden></div>
    </div>
    <div class="code-output-preview"></div>
  `;

  // Tab switching
  panel.querySelectorAll('.code-output-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      panel.querySelectorAll('.code-output-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.dataset.tab;
      panel.querySelectorAll('.code-output-content').forEach(c => {
        c.hidden = c.dataset.tab !== target;
      });
    });
  });

  // Action buttons
  panel.querySelector('[data-action="copy-output"]').addEventListener('click', () => {
    const active = panel.querySelector('.code-output-content:not([hidden])');
    if (active) copyToClipboard(active.textContent);
  });
  panel.querySelector('[data-action="clear-output"]').addEventListener('click', () => {
    panel.querySelectorAll('.code-output-content').forEach(c => { c.innerHTML = ''; });
    const badge = panel.querySelector('.error-badge');
    if (badge) badge.remove();
  });
  panel.querySelector('[data-action="close-output"]').addEventListener('click', () => {
    _cleanupPanel(panel);
    panel.remove();
  });

  pre.after(panel);
  return panel;
}

function _cleanupPanel(panel) {
  const iframe = panel.querySelector('iframe');
  if (iframe) {
    _activeIframes.delete(iframe);
    const hbId = _heartbeatIntervals.get(iframe);
    if (hbId) { clearInterval(hbId); _heartbeatIntervals.delete(iframe); }
  }
}

export function appendOutput(panel, text, className) {
  const output = panel.querySelector('.code-output-content[data-tab="output"]');
  const el = document.createElement('div');
  el.className = className;
  el.textContent = text;
  output.appendChild(el);
}

export function appendError(panel, text) {
  const errors = panel.querySelector('.code-output-content[data-tab="errors"]');
  const el = document.createElement('div');
  el.className = 'stderr';
  el.textContent = text;
  errors.appendChild(el);

  const tab = panel.querySelector('.code-output-tab[data-tab="errors"]');
  let badge = tab.querySelector('.error-badge');
  const count = errors.children.length;
  if (!badge) {
    badge = document.createElement('span');
    badge.className = 'error-badge';
    tab.appendChild(badge);
  }
  badge.textContent = count;
}

// ---------------------------------------------------------------------------
// HTML Preview (fullscreen modal)
// ---------------------------------------------------------------------------

function _buildPreviewSrcdoc(rawCode) {
  return _sharedBuildPreviewSrcdoc(rawCode);
}

export function closePreviewModal() {
  const modal = document.getElementById('code-preview-modal');
  if (modal) modal.classList.remove('visible');
  if (_previewModalIframe) {
    _activeIframes.delete(_previewModalIframe);
    _previewModalIframe = null;
  }
  if (_previewModalHandler) {
    window.removeEventListener('message', _previewModalHandler);
    _previewModalHandler = null;
  }
  if (_previewModalHeartbeat) {
    clearInterval(_previewModalHeartbeat);
    _previewModalHeartbeat = null;
  }
  _previewModalCodeHeader = null;
  _previewModalErrors = [];
  if (_previewModalEscHandler) {
    document.removeEventListener('keydown', _previewModalEscHandler);
    _previewModalEscHandler = null;
  }
  const fixBtn = document.getElementById('code-preview-fix-errors');
  if (fixBtn) fixBtn.hidden = true;
  const wrap = document.getElementById('code-preview-iframe-wrap');
  if (wrap) wrap.innerHTML = '';
  const consoleEl = document.getElementById('code-preview-console-content');
  if (consoleEl) consoleEl.innerHTML = '';
}

export function toggleHtmlPreview(codeHeader) {
  const rawCode = decodeURIComponent(codeHeader.dataset.rawCode || '');
  if (!rawCode) return;

  closePreviewModal();

  const modal = document.getElementById('code-preview-modal');
  const wrap = document.getElementById('code-preview-iframe-wrap');
  const consoleEl = document.getElementById('code-preview-console-content');
  const consolePanel = document.getElementById('code-preview-console');
  const titleEl = document.getElementById('code-preview-modal-title');
  if (!modal || !wrap) return;

  _previewModalSrcdoc = _buildPreviewSrcdoc(rawCode);
  _previewModalCodeHeader = codeHeader;
  _previewModalErrors = [];

  const iframe = document.createElement('iframe');
  iframe.sandbox = 'allow-scripts allow-forms allow-modals allow-popups';
  iframe.srcdoc = _previewModalSrcdoc;
  _activeIframes.add(iframe);
  _previewModalIframe = iframe;

  wrap.innerHTML = '';
  wrap.appendChild(iframe);
  if (consolePanel) consolePanel.hidden = true;
  if (titleEl) titleEl.textContent = 'HTML Preview';
  modal.classList.add('visible');

  let lastPong = Date.now();
  _previewModalHeartbeat = setInterval(() => {
    if (_previewModalIframe !== iframe) return;
    try { iframe.contentWindow?.postMessage({ type: 'code-ping' }, '*'); } catch {}
    if (Date.now() - lastPong > 8000) {
      if (titleEl) titleEl.textContent = 'HTML Preview (unresponsive)';
    }
  }, 3000);

  _previewModalHandler = (e) => {
    if (e.source !== iframe.contentWindow) return;
    const d = e.data;
    if (!d || !d.type) return;

    if (d.type === 'code-error' && consoleEl) {
      const errorText = extractErrorMessage(d, 'Unknown error');
      const el = document.createElement('div');
      el.className = 'console-error';
      el.textContent = errorText;
      consoleEl.appendChild(el);
      _previewModalErrors.push(errorText);
      if (consolePanel) consolePanel.hidden = false;
      const toggleBtn = document.getElementById('code-preview-console-toggle');
      if (toggleBtn) toggleBtn.classList.add('active');
      const fixBtn = document.getElementById('code-preview-fix-errors');
      if (fixBtn) fixBtn.hidden = false;
      const autoFixBtn = document.getElementById('code-preview-auto-fix');
      if (autoFixBtn) autoFixBtn.hidden = false;
    } else if (d.type === 'code-console' && consoleEl) {
      const el = document.createElement('div');
      el.className = 'console-' + (d.level || 'log');
      el.textContent = (d.args || []).join(' ');
      consoleEl.appendChild(el);
      consoleEl.scrollTop = consoleEl.scrollHeight;
    } else if (d.type === 'code-pong') {
      lastPong = Date.now();
      if (titleEl) titleEl.textContent = 'HTML Preview';
    }
  };
  window.addEventListener('message', _previewModalHandler);

  document.getElementById('code-preview-close').onclick = closePreviewModal;
  document.getElementById('code-preview-reload').onclick = () => {
    iframe.srcdoc = _previewModalSrcdoc;
    lastPong = Date.now();
    if (consoleEl) consoleEl.innerHTML = '';
    if (titleEl) titleEl.textContent = 'HTML Preview';
  };
  document.getElementById('code-preview-console-toggle').onclick = () => {
    if (!consolePanel) return;
    consolePanel.hidden = !consolePanel.hidden;
    document.getElementById('code-preview-console-toggle')?.classList.toggle('active', !consolePanel.hidden);
  };
  document.getElementById('code-preview-new-tab').onclick = () => {
    const blob = new Blob([_previewModalSrcdoc], { type: 'text/html' });
    window.open(URL.createObjectURL(blob), '_blank');
  };
  document.getElementById('code-preview-download').onclick = () => {
    if (!_previewModalCodeHeader) return;
    downloadCodeBlock(_previewModalCodeHeader);
  };
  document.getElementById('code-preview-auto-fix').onclick = async () => {
    if (!_previewModalCodeHeader) return;
    const targetHeader = _previewModalCodeHeader;
    closePreviewModal();
    await _loadCodeEdit();
    _codeEdit().autoFixCodeBlock(targetHeader);
  };
  document.getElementById('code-preview-fix-errors').onclick = async () => {
    if (!_previewModalCodeHeader || _previewModalErrors.length === 0) return;
    const targetHeader = _previewModalCodeHeader;
    const uniqueErrors = [...new Set(_previewModalErrors)];
    closePreviewModal();
    const errorSummary = uniqueErrors.map((e, i) => `${i + 1}. ${e}`).join('\n');
    const instruction = `Fix these JavaScript errors from the preview:\n${errorSummary}`;
    await _loadCodeEdit();
    _codeEdit().executeAiEdit(targetHeader, instruction, true, false, { skipPlan: true });
  };

  _previewModalEscHandler = (e) => {
    if (e.key === 'Escape') closePreviewModal();
  };
  document.addEventListener('keydown', _previewModalEscHandler);
}

// ---------------------------------------------------------------------------
// Python Runner
// ---------------------------------------------------------------------------

const _PYODIDE_BUILTINS = new Set([
  'sys', 'os', 'io', 'base64', 'json', 'math', 'random', 'datetime',
  're', 'collections', 'itertools', 'functools', 'typing', 'string',
  'copy', 'time', 'hashlib', 'pathlib', 'abc', 'enum', 'dataclasses',
  'decimal', 'fractions', 'statistics', 'textwrap', 'struct', 'csv',
  'urllib', 'http', 'html', 'xml', 'logging', 'unittest', 'pprint',
]);

async function _loadPyodide() {
  if (_pyodide) return _pyodide;
  if (_pyodideReady) return _pyodideReady;

  _pyodideReady = (async () => {
    const mod = await import('https://cdn.jsdelivr.net/pyodide/v0.27.4/full/pyodide.mjs');
    _pyodide = await mod.loadPyodide();
    return _pyodide;
  })();

  return _pyodideReady;
}

async function _runWithPyodide(code) {
  const pyodide = await _loadPyodide();

  // Detect imports and load packages
  const imports = code.match(/^\s*(?:import|from)\s+(\w+)/gm) || [];
  const packages = imports
    .map(m => m.replace(/^\s*(?:import|from)\s+/, '').trim())
    .filter(p => !_PYODIDE_BUILTINS.has(p));

  for (const pkg of packages) {
    try {
      await pyodide.loadPackage(pkg);
    } catch {
      try {
        await pyodide.runPythonAsync(`import micropip; await micropip.install('${pkg}')`);
      } catch { /* package unavailable — will fail naturally in code */ }
    }
  }

  // Redirect stdout/stderr
  pyodide.runPython('import sys, io; sys.stdout = io.StringIO(); sys.stderr = io.StringIO()');

  let success = true, error = '';
  const startTime = Date.now();
  try {
    const preamble = getPythonPreamble();
    await pyodide.runPythonAsync(preamble + '\n' + code);
  } catch (err) {
    success = false;
    error = err.message;
  }

  const stdout = pyodide.runPython('sys.stdout.getvalue()');
  const stderr = pyodide.runPython('sys.stderr.getvalue()');
  const elapsed = (Date.now() - startTime) / 1000;

  // Reset stdout/stderr
  pyodide.runPython('sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__');

  return {
    success,
    stdout: stdout || '',
    stderr: stderr || '',
    error,
    return_value: null,
    metrics: { elapsed_seconds: elapsed },
  };
}

/**
 * Returns a Python preamble that patches matplotlib and pandas for rich output.
 * matplotlib plt.show() captures as base64 PNG with __IMG_BASE64__ marker.
 * pandas DataFrame repr emits HTML table with __HTML__ marker.
 */
export function getPythonPreamble() {
  return [
    'import sys as _sys, io as _io, base64 as _b64',
    'try:',
    '    import matplotlib as _mpl',
    '    _mpl.use("Agg")',
    '    import matplotlib.pyplot as _plt',
    '    _orig_show = _plt.show',
    '    def _patched_show(*a, **kw):',
    '        _buf = _io.BytesIO()',
    '        _plt.savefig(_buf, format="png", bbox_inches="tight", dpi=100)',
    '        _buf.seek(0)',
    '        print(f"__IMG_BASE64__:{_b64.b64encode(_buf.read()).decode()}")',
    '        _plt.close("all")',
    '    _plt.show = _patched_show',
    'except ImportError:',
    '    pass',
    'try:',
    '    import pandas as _pd',
    '    _orig_df_repr = _pd.DataFrame.__repr__',
    '    def _patched_df_repr(self):',
    '        _text = _orig_df_repr(self)',
    '        if len(self) <= 100:',
    '            print(f"__HTML__:{self.to_html(classes=\\"df-table\\", max_rows=50)}")',
    '        return _text',
    '    _pd.DataFrame.__repr__ = _patched_df_repr',
    'except ImportError:',
    '    pass',
  ].join('\n');
}

/**
 * Renders execution output into the output panel, handling rich markers.
 * Shared between container execution and Pyodide execution.
 */
export function renderExecutionOutput(panel, outputEl, data) {
  if (data.stdout) {
    const lines = data.stdout.split('\n');
    for (const line of lines) {
      if (line.startsWith('__IMG_BASE64__:')) {
        const img = document.createElement('img');
        img.className = 'code-output-image';
        img.src = `data:image/png;base64,${line.slice('__IMG_BASE64__:'.length)}`;
        outputEl.appendChild(img);
      } else if (line.startsWith('__HTML__:')) {
        const container = document.createElement('div');
        container.className = 'code-output-rich-html';
        container.innerHTML = line.slice('__HTML__:'.length);
        outputEl.appendChild(container);
      } else if (line.startsWith('data:image/') || (/^iVBOR/.test(line))) {
        // Legacy image detection (backward compat)
        const img = document.createElement('img');
        img.className = 'code-output-image';
        img.src = line.startsWith('data:') ? line : `data:image/png;base64,${line}`;
        outputEl.appendChild(img);
      } else if (line) {
        appendOutput(panel, line, 'stdout');
      }
    }
  }
  if (data.return_value != null) appendOutput(panel, String(data.return_value), 'return-value');
  if (data.stderr && data.success) appendOutput(panel, data.stderr, 'console-warn');
  if (data.error && !data.success) appendError(panel, data.error);
  if (data.metrics?.elapsed_seconds != null) {
    const timeEl = document.createElement('div');
    timeEl.className = 'execution-time';
    timeEl.textContent = `Executed in ${data.metrics.elapsed_seconds.toFixed(2)}s`;
    outputEl.appendChild(timeEl);
  }
}

/**
 * Attach the Fix & Retry / Ask AI bar to a run's error tab.
 * Shared by container, Pyodide, and browser-JS execution paths.
 */
function _attachFixRetryBar(panel, codeHeader, errorText, opts = {}) {
  const errorTab = panel.querySelector('.code-output-content[data-tab="errors"]');
  if (!errorTab) return;
  const btnBar = document.createElement('div');
  btnBar.className = 'code-fix-retry-bar';
  btnBar.innerHTML = `
    <button class="code-fix-retry-btn" title="Send error to AI, accept fix, re-run automatically">Fix &amp; Retry</button>
    <button class="code-fix-ask-btn" title="Open Ask AI with error context">Ask AI about this</button>
  `;
  btnBar.querySelector('.code-fix-retry-btn').addEventListener('click', () => {
    btnBar.remove();
    _fixAndRetry(codeHeader, errorText, opts);
  });
  btnBar.querySelector('.code-fix-ask-btn').addEventListener('click', async () => {
    btnBar.remove();
    await _loadCodeEdit();
    _codeEdit().showAskAiPrompt(codeHeader);
    // Pre-fill the input with error context
    setTimeout(() => {
      const input = codeHeader.nextElementSibling?.querySelector?.('.code-ask-ai-input')
        || document.querySelector('.code-ask-ai-bar .code-ask-ai-input');
      if (input) {
        input.value = `Fix this error: ${_errorHeadline(errorText)}`;
        input.focus();
      }
    }, 50);
  });
  errorTab.appendChild(btnBar);
}

export async function runPythonCode(codeHeader, { preserveRetryState = false } = {}) {
  const rawCode = decodeURIComponent(codeHeader.dataset.rawCode || '');
  // A manual Run is a fresh start; a Fix & Retry re-run must keep the
  // attempt counter or the retry circuit breaker never trips.
  if (!preserveRetryState) {
    _fixRetryCount.delete(codeHeader);
    _fixAttemptHistory.delete(codeHeader);
  }
  if (!rawCode) return;

  for (const pat of _PY_BLOCKED) {
    if (pat.test(rawCode)) {
      const panel = getOutputPanel(codeHeader);
      appendError(panel, `Blocked: ${pat.source}`);
      return;
    }
  }

  const panel = getOutputPanel(codeHeader);
  const outputEl = panel.querySelector('.code-output-content[data-tab="output"]');

  // Pre-validate with CodeMind AST (catches syntax errors before server round-trip)
  if (CodeMind.isReady()) {
    const validation = await codeMindValidate(rawCode, 'python');
    if (!validation.valid && validation.errors.length > 0) {
      const err = validation.errors[0];
      appendError(panel, `Syntax error at line ${err.startRow + 1}: ${err.message}`);
      // Still show the output panel so user sees the error
      panel.classList.remove('hidden');
      return;
    }
  }

  outputEl.innerHTML = '<div class="code-output-spinner">Running...</div>';

  const runBtn = codeHeader.querySelector('[data-action="run-code"]');
  if (runBtn) { runBtn.classList.add('active'); runBtn.textContent = 'Running...'; }

  try {
    const resp = await fetch('/api/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: getPythonPreamble() + '\n' + rawCode, timeout: 30 }),
    });

    if (!resp.ok) {
      const errBody = await resp.text();
      let errMsg = '';
      try { errMsg = JSON.parse(errBody)?.error || ''; } catch { /* not JSON */ }
      outputEl.innerHTML = '';
      if (resp.status === 400 && errMsg) {
        // Sandbox policy rejection ("Code rejected: open() is not allowed")
        // — show the clean message and offer Fix & Retry so the AI can
        // rewrite around the restriction.
        appendError(panel, errMsg);
        _attachFixRetryBar(panel, codeHeader, errMsg, { env: 'sandbox' });
      } else {
        appendError(panel, `Server error: ${(errMsg || errBody).slice(0, 200)}`);
      }
      return;
    }

    const data = await resp.json();
    outputEl.innerHTML = '';

    // Show environment indicator
    const envEl = document.createElement('div');
    envEl.className = 'code-output-env';
    envEl.textContent = '\uD83D\uDC33 Sandbox';
    outputEl.appendChild(envEl);

    renderExecutionOutput(panel, outputEl, data);

    // Show Fix & Retry buttons on execution errors
    if (!data.success && (data.error || data.stderr)) {
      _attachFixRetryBar(panel, codeHeader, data.error || data.stderr, {
        env: 'sandbox',
        stdout: data.stdout || '',
      });
    } else if (data.success) {
      _fixRetryCount.delete(codeHeader);
      _fixAttemptHistory.delete(codeHeader);
    }
  } catch (fetchErr) {
    // Executor unreachable — try Pyodide fallback
    try {
      outputEl.innerHTML = '<div class="code-output-spinner">Loading Python runtime...</div>';
      const data = await _runWithPyodide(rawCode);
      outputEl.innerHTML = '';

      // Show environment indicator
      const envEl = document.createElement('div');
      envEl.className = 'code-output-env';
      envEl.textContent = '\uD83C\uDF10 Browser (Pyodide)';
      outputEl.appendChild(envEl);

      renderExecutionOutput(panel, outputEl, data);

      // Show Fix & Retry buttons on Pyodide errors too
      if (!data.success && (data.error || data.stderr)) {
        _attachFixRetryBar(panel, codeHeader, data.error || data.stderr, {
          env: 'pyodide',
          stdout: data.stdout || '',
        });
      } else if (data.success) {
        _fixRetryCount.delete(codeHeader);
        _fixAttemptHistory.delete(codeHeader);
      }
    } catch (pyErr) {
      outputEl.innerHTML = '';
      appendError(panel, `Executor unavailable and Pyodide failed: ${pyErr.message}\n\nStart Docker with start.bat to enable full execution.`);
    }
  } finally {
    if (runBtn) { runBtn.classList.remove('active'); runBtn.textContent = 'Run'; }
  }
}

// ---------------------------------------------------------------------------
// JavaScript Execution — sandboxed iframe runner
// ---------------------------------------------------------------------------

const _JS_RUN_HARD_TIMEOUT_MS = 15_000;
// After the main body settles, keep collecting late async output
// (setTimeout/interval demos) for this long; each new message resets it.
const _JS_RUN_GRACE_MS = 1200;

/**
 * Run JavaScript in a hidden sandboxed iframe (allow-scripts only, opaque
 * origin). Console output and errors stream back via postMessage. The user
 * code is wrapped in an async IIFE, so top-level await and dynamic
 * `await import('https://...')` work; static import statements do not
 * (they can't appear inside a function body).
 *
 * Returns the same result shape as the Python executor so
 * renderExecutionOutput can be shared.
 */
function _runJsInIframe(code) {
  const token = 'jsrun_' + Math.random().toString(36).slice(2);
  const startTime = Date.now();

  // JSON.stringify handles quotes/newlines; the </ escape stops a literal
  // </script> inside user code from terminating the srcdoc script block.
  const codeJson = JSON.stringify(code).replace(/<\//g, '<\\/');

  const bootstrap = `<!DOCTYPE html><html><head><meta charset="utf-8"><script>
(function () {
  var TOKEN = ${JSON.stringify(token)};
  function post(msg) { msg.__jsrun = TOKEN; window.parent.postMessage(msg, '*'); }
  function fmt(v) {
    if (typeof v === 'string') return v;
    if (v instanceof Error) return v.stack || (v.name + ': ' + v.message);
    try { var s = JSON.stringify(v); if (s !== undefined) return s; } catch (e) {}
    return String(v);
  }
  ['log', 'info', 'warn', 'error', 'debug'].forEach(function (level) {
    var orig = console[level];
    console[level] = function () {
      post({ type: 'console', level: level, text: Array.prototype.map.call(arguments, fmt).join(' ') });
      orig.apply(console, arguments);
    };
  });
  window.onerror = function (msg, src, line) {
    post({ type: 'error', text: line ? msg + ' (line ' + line + ')' : String(msg) });
  };
  window.onunhandledrejection = function (e) {
    var r = e.reason;
    post({ type: 'error', text: 'Unhandled rejection: ' + (r && r.stack ? r.stack : String(r)) });
  };
  var USER_CODE = ${codeJson};
  Promise.resolve().then(function () {
    return new Function('return (async () => {\\n' + USER_CODE + '\\n})()')();
  }).then(function (result) {
    post({ type: 'done', result: result === undefined ? null : fmt(result) });
  }, function (err) {
    post({ type: 'error', text: err && err.stack ? err.stack : String(err) });
    post({ type: 'done', result: null });
  });
})();
<\/script></head><body></body></html>`;

  return new Promise((resolve) => {
    const iframe = document.createElement('iframe');
    iframe.setAttribute('sandbox', 'allow-scripts');
    iframe.style.display = 'none';

    const stdout = [];
    const stderrLines = [];
    let errorText = '';
    let returnValue = null;
    let doneSeen = false;
    let settled = false;
    let graceTimer = null;
    let hardTimer = null;

    const finish = (timedOut) => {
      if (settled) return;
      settled = true;
      window.removeEventListener('message', onMsg);
      clearTimeout(graceTimer);
      clearTimeout(hardTimer);
      iframe.remove();
      if (timedOut && !doneSeen && !errorText) {
        errorText = `Timed out after ${_JS_RUN_HARD_TIMEOUT_MS / 1000}s — execution stopped (infinite loop or a never-settling await?)`;
      }
      resolve({
        success: !errorText,
        stdout: stdout.join('\n'),
        stderr: stderrLines.join('\n'),
        error: errorText,
        return_value: returnValue,
        metrics: { elapsed_seconds: (Date.now() - startTime) / 1000 },
      });
    };

    const scheduleGrace = () => {
      clearTimeout(graceTimer);
      graceTimer = setTimeout(() => finish(false), _JS_RUN_GRACE_MS);
    };

    const onMsg = (e) => {
      const d = e.data;
      if (!d || d.__jsrun !== token) return;
      if (d.type === 'console') {
        // console.error/warn are user output, not failures — only thrown
        // errors and unhandled rejections mark the run as failed.
        if (d.level === 'warn' || d.level === 'error') stderrLines.push(d.text);
        else stdout.push(d.text);
        if (doneSeen) scheduleGrace();
      } else if (d.type === 'error') {
        if (!errorText) errorText = d.text;
        if (doneSeen) scheduleGrace();
      } else if (d.type === 'done') {
        doneSeen = true;
        if (d.result != null) returnValue = d.result;
        scheduleGrace();
      }
    };

    window.addEventListener('message', onMsg);
    hardTimer = setTimeout(() => finish(true), _JS_RUN_HARD_TIMEOUT_MS);
    iframe.setAttribute('srcdoc', bootstrap);
    document.body.appendChild(iframe);
  });
}

export async function runJavaScriptCode(codeHeader, { preserveRetryState = false } = {}) {
  const rawCode = decodeURIComponent(codeHeader.dataset.rawCode || '');
  if (!preserveRetryState) {
    _fixRetryCount.delete(codeHeader);
    _fixAttemptHistory.delete(codeHeader);
  }
  if (!rawCode) return;

  const panel = getOutputPanel(codeHeader);
  const outputEl = panel.querySelector('.code-output-content[data-tab="output"]');

  // Pre-validate with CodeMind AST (catches syntax errors before running)
  if (CodeMind.isReady()) {
    const validation = await codeMindValidate(rawCode, 'javascript');
    if (!validation.valid && validation.errors.length > 0) {
      const err = validation.errors[0];
      appendError(panel, `Syntax error at line ${err.startRow + 1}: ${err.message}`);
      panel.classList.remove('hidden');
      return;
    }
  }

  outputEl.innerHTML = '<div class="code-output-spinner">Running...</div>';

  const runBtn = codeHeader.querySelector('[data-action="run-code"]');
  if (runBtn) { runBtn.classList.add('active'); runBtn.textContent = 'Running...'; }

  try {
    const data = await _runJsInIframe(rawCode);
    outputEl.innerHTML = '';

    const envEl = document.createElement('div');
    envEl.className = 'code-output-env';
    envEl.textContent = '🌐 Browser';
    outputEl.appendChild(envEl);

    renderExecutionOutput(panel, outputEl, data);

    if (data.success && !data.stdout && !data.stderr && data.return_value == null) {
      appendOutput(panel, '(no output — use console.log to print results)', 'stdout');
    }

    if (!data.success && (data.error || data.stderr)) {
      _attachFixRetryBar(panel, codeHeader, data.error || data.stderr, {
        env: 'browser-js',
        stdout: data.stdout || '',
      });
    } else if (data.success) {
      _fixRetryCount.delete(codeHeader);
      _fixAttemptHistory.delete(codeHeader);
    }
  } finally {
    if (runBtn) { runBtn.classList.remove('active'); runBtn.textContent = 'Run'; }
  }
}

/**
 * Language dispatch for the code-header Run button.
 */
export function runCodeBlock(codeHeader, opts = {}) {
  const lang = (codeHeader.dataset.lang || '').toLowerCase();
  if (lang === 'javascript' || lang === 'js') return runJavaScriptCode(codeHeader, opts);
  return runPythonCode(codeHeader, opts);
}

// --- Error Fix & Retry ---

/** Per-block memory of what prior fix attempts changed, so retry prompts
 *  can steer the model away from repeating a fix that didn't work. */
const _fixAttemptHistory = new WeakMap();

/** Runtime constraints injected into fix prompts so the model doesn't
 *  produce fixes the environment will reject (e.g. open() in the sandbox,
 *  static imports in the iframe runner). */
const _RUN_ENV_NOTES = {
  sandbox:
    'The code runs in a locked-down Python sandbox: open(), subprocess, '
    + 'exec(), eval(), compile() and __import__() are all blocked. Standard '
    + 'library plus common data packages (numpy, pandas, matplotlib) are '
    + 'available. Results must be print()ed.',
  pyodide:
    'The code runs in Pyodide (Python in the browser). Only pure-Python '
    + 'packages and Pyodide-bundled wheels (numpy, pandas, matplotlib, etc.) '
    + 'are available — packages with native extensions or network '
    + 'dependencies may be unavailable. Results must be print()ed.',
  'browser-js':
    'The code runs as JavaScript in a sandboxed browser iframe, wrapped in '
    + 'an async IIFE: top-level await works, but static `import` statements '
    + 'do NOT — use `await import("https://...")` for modules. No Node.js '
    + 'APIs (no require, fs, process). Output must go through console.log. '
    + 'Line numbers reported in errors may be off by one because of the wrapper.',
};

/**
 * Truncate an error for prompting while keeping both ends — Python
 * tracebacks put the exception on the LAST line, JS stacks put the
 * message FIRST. Naive head-truncation used to drop the actual error
 * message from long Python tracebacks.
 */
function _truncateErrorForPrompt(text, max = 1200) {
  const t = (text || '').trim();
  if (t.length <= max) return t;
  const head = t.slice(0, Math.floor(max * 0.3));
  const tail = t.slice(-Math.floor(max * 0.7));
  return `${head}\n... (${t.length - max} chars omitted) ...\n${tail}`;
}

/**
 * Pick the single most informative line of an error for short contexts
 * (Ask AI prefill). JS stacks lead with the message; Python tracebacks
 * end with it.
 */
function _errorHeadline(errorText) {
  const lines = (errorText || '').split('\n').map(l => l.trim()).filter(Boolean);
  if (lines.length === 0) return (errorText || '').slice(0, 100);
  const first = lines[0];
  if (/(error|exception|rejection)/i.test(first) && !/^at\s/.test(first)) {
    return first.slice(0, 160);
  }
  return lines[lines.length - 1].slice(0, 160);
}

/** Strip iframe-runner-internal stack frames (eval / new Function /
 *  <anonymous>) that only add noise to fix prompts. */
function _cleanJsStack(text) {
  return (text || '').split('\n')
    .filter(l => !(/^\s*at\s/.test(l) && /(\beval\b|new Function|<anonymous>)/.test(l)))
    .join('\n');
}

async function _fixAndRetry(codeHeader, errorText, opts = {}) {
  const count = (_fixRetryCount.get(codeHeader) || 0) + 1;
  _fixRetryCount.set(codeHeader, count);

  if (count > 3) {
    showToast('Reached the retry limit. Try Ask AI for guidance.', 'warning');
    return;
  }

  const block = getBlock(codeHeader);
  const versionsBefore = block ? block.versions.length : 0;

  const cleanError = opts.env === 'browser-js' ? _cleanJsStack(errorText) : errorText;

  const parts = [
    'Fix the following error in this code:',
    '',
    _truncateErrorForPrompt(cleanError),
  ];

  const stdoutTail = (opts.stdout || '').trim().slice(-500);
  if (stdoutTail) {
    parts.push('', 'Output printed before the error:', stdoutTail);
  }

  const envNote = _RUN_ENV_NOTES[opts.env];
  if (envNote) parts.push('', `Runtime constraints: ${envNote}`);

  const history = _fixAttemptHistory.get(codeHeader) || [];
  if (history.length > 0) {
    parts.push('', 'Previous fix attempts FAILED — the error persisted after each. Do NOT repeat them:');
    history.forEach((h, i) => parts.push(`${i + 1}. ${h}`));
    parts.push('Try a different approach.');
  }

  parts.push('', 'The error occurred when running the code. Fix the root cause.');

  await _loadCodeEdit();
  await _codeEdit().executeAiEdit(codeHeader, parts.join('\n'), true, false, { skipPlan: true });

  // Check if user accepted (new version was created)
  const blockAfter = getBlock(codeHeader);
  if (blockAfter && blockAfter.versions.length > versionsBefore) {
    // Remember what this attempt changed for the next retry's prompt
    const summary = blockAfter.summaries?.[blockAfter.summaries.length - 1] || 'unlabeled edit';
    history.push(summary.slice(0, 200));
    _fixAttemptHistory.set(codeHeader, history);
    // preserveRetryState: the re-run must NOT reset the attempt counter,
    // or the 3-attempt circuit breaker can never trip (it was dead code —
    // every re-run cleared the count back to zero).
    await runCodeBlock(codeHeader, { preserveRetryState: true });
  }
}

// ---------------------------------------------------------------------------
// SVG Preview
// ---------------------------------------------------------------------------

export function toggleSvgPreview(codeHeader) {
  const rawCode = decodeURIComponent(codeHeader.dataset.rawCode || '');
  if (!rawCode) return;

  const pre = codeHeader.nextElementSibling;
  if (!pre) return;

  const existing = pre.nextElementSibling;
  if (existing && existing.classList.contains('svg-preview-container')) {
    existing.remove();
    codeHeader.querySelector('[data-action="preview-svg"]')?.classList.remove('active');
    return;
  }

  codeHeader.querySelector('[data-action="preview-svg"]')?.classList.add('active');

  const sanitized = _sanitizeSvg(rawCode);
  if (!sanitized) return;

  const container = document.createElement('div');
  container.className = 'svg-preview-container';
  container.innerHTML = sanitized;
  pre.after(container);
}

function _sanitizeSvg(svgStr) {
  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(svgStr, 'image/svg+xml');
    const err = doc.querySelector('parsererror');
    if (err) return null;

    const walk = (node) => {
      if (node.nodeType !== 1) return;
      const tag = node.tagName.toLowerCase();
      if (['script', 'foreignobject', 'iframe', 'object', 'embed'].includes(tag)) {
        node.remove();
        return;
      }
      const attrs = [...node.attributes];
      for (const attr of attrs) {
        const name = attr.name.toLowerCase();
        if (name.startsWith('on')) {
          node.removeAttribute(attr.name);
        } else if (['href', 'xlink:href', 'action', 'formaction'].includes(name)) {
          if (attr.value.trim().toLowerCase().startsWith('javascript:')) {
            node.removeAttribute(attr.name);
          }
        }
      }
      [...node.children].forEach(walk);
    };

    walk(doc.documentElement);
    return new XMLSerializer().serializeToString(doc.documentElement);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Code Editing (textarea)
// ---------------------------------------------------------------------------

export function toggleCodeEdit(codeHeader) {
  const pre = codeHeader.nextElementSibling;
  if (!pre) return;

  const editBtn = codeHeader.querySelector('[data-action="edit-code"]');

  const existingTextarea = pre.nextElementSibling;
  if (existingTextarea && existingTextarea.classList.contains('code-edit-textarea')) {
    existingTextarea.remove();
    pre.hidden = false;
    if (editBtn) editBtn.textContent = 'Edit';
    return;
  }

  const rawCode = decodeURIComponent(codeHeader.dataset.rawCode || '');
  const textarea = document.createElement('textarea');
  textarea.className = 'code-edit-textarea';
  textarea.value = rawCode;
  textarea.spellcheck = false;

  textarea.style.height = Math.max(120, pre.offsetHeight) + 'px';
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, window.innerHeight * 0.6) + 'px';
  });

  // Full keyboard shortcuts — shared with workspace editor
  textarea.addEventListener('keydown', (e) => {
    const ta = textarea;
    const ctrl = e.ctrlKey || e.metaKey;

    if (e.key === 'Escape') {
      ta.remove(); pre.hidden = false;
      if (editBtn) editBtn.textContent = 'Edit';
      return;
    }

    // Ctrl+/ — toggle comment
    if (ctrl && e.key === '/') {
      e.preventDefault();
      const val = ta.value, start = ta.selectionStart, end = ta.selectionEnd;
      const lang = (codeHeader.dataset.lang || '').toLowerCase();
      const prefix = (lang === 'css' || lang === 'scss') ? '/* ' : (lang === 'html' || lang === 'htm') ? '<!-- ' : '// ';
      const suffix = (lang === 'css' || lang === 'scss') ? ' */' : (lang === 'html' || lang === 'htm') ? ' -->' : '';
      const lineStart = val.lastIndexOf('\n', start - 1) + 1;
      const blockEnd = (end === start ? val.indexOf('\n', end) : val.indexOf('\n', end - 1));
      const be = blockEnd === -1 ? val.length : blockEnd;
      const lines = val.slice(lineStart, be).split('\n');
      const allComm = lines.every(l => l.trimStart().startsWith(prefix.trimEnd()));
      const toggled = lines.map(l => {
        if (allComm) { const i = l.indexOf(prefix.trimEnd()); let s = i === -1 ? l : l.slice(0, i) + l.slice(i + prefix.length); if (suffix && s.trimEnd().endsWith(suffix.trimEnd())) s = s.slice(0, s.lastIndexOf(suffix.trimEnd())); return s; }
        return prefix + l + suffix;
      }).join('\n');
      ta.value = val.slice(0, lineStart) + toggled + val.slice(be);
      ta.selectionStart = lineStart; ta.selectionEnd = lineStart + toggled.length;
      return;
    }

    // Ctrl+D — duplicate line/selection
    if (ctrl && e.key === 'd') {
      e.preventDefault();
      const val = ta.value, start = ta.selectionStart, end = ta.selectionEnd;
      if (start !== end) {
        const sel = val.slice(start, end);
        ta.value = val.slice(0, end) + sel + val.slice(end);
        ta.selectionStart = end; ta.selectionEnd = end + sel.length;
      } else {
        const ls = val.lastIndexOf('\n', start - 1) + 1;
        let le = val.indexOf('\n', start); if (le === -1) le = val.length;
        const line = val.slice(ls, le);
        ta.value = val.slice(0, le) + '\n' + line + val.slice(le);
        ta.selectionStart = ta.selectionEnd = start + line.length + 1;
      }
      return;
    }

    // Ctrl+Shift+K — delete line
    if (ctrl && e.shiftKey && e.key === 'K') {
      e.preventDefault();
      const val = ta.value, start = ta.selectionStart;
      const ls = val.lastIndexOf('\n', start - 1) + 1;
      let le = val.indexOf('\n', start); if (le === -1) le = val.length; else le++;
      ta.value = val.slice(0, ls) + val.slice(le);
      ta.selectionStart = ta.selectionEnd = Math.min(ls, ta.value.length);
      return;
    }

    // Alt+Up/Down — move line
    if (e.altKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
      e.preventDefault();
      const val = ta.value, pos = ta.selectionStart;
      const ls = val.lastIndexOf('\n', pos - 1) + 1;
      let le = val.indexOf('\n', pos); if (le === -1) le = val.length;
      if (e.key === 'ArrowUp' && ls > 0) {
        const pls = val.lastIndexOf('\n', ls - 2) + 1;
        const cur = val.slice(ls, le), prev = val.slice(pls, ls - 1);
        ta.value = val.slice(0, pls) + cur + '\n' + prev + val.slice(le);
        ta.selectionStart = ta.selectionEnd = pls + (pos - ls);
      } else if (e.key === 'ArrowDown' && le < val.length) {
        let nle = val.indexOf('\n', le + 1); if (nle === -1) nle = val.length;
        const cur = val.slice(ls, le), next = val.slice(le + 1, nle);
        ta.value = val.slice(0, ls) + next + '\n' + cur + val.slice(nle);
        ta.selectionStart = ta.selectionEnd = ls + next.length + 1 + (pos - ls);
      }
      return;
    }

    // Tab — indent (2 spaces, multi-line support)
    if (e.key === 'Tab' && !e.shiftKey) {
      e.preventDefault();
      const start = ta.selectionStart, end = ta.selectionEnd, val = ta.value;
      if (start !== end) {
        const fl = val.lastIndexOf('\n', start - 1) + 1;
        const block = val.slice(fl, end);
        const indented = block.split('\n').map(l => '  ' + l).join('\n');
        ta.value = val.slice(0, fl) + indented + val.slice(end);
        ta.selectionStart = fl; ta.selectionEnd = fl + indented.length;
      } else {
        ta.value = val.slice(0, start) + '  ' + val.slice(end);
        ta.selectionStart = ta.selectionEnd = start + 2;
      }
      return;
    }

    // Shift+Tab — dedent
    if (e.key === 'Tab' && e.shiftKey) {
      e.preventDefault();
      const start = ta.selectionStart, end = ta.selectionEnd, val = ta.value;
      const fl = val.lastIndexOf('\n', start - 1) + 1;
      const blockEnd = start === end ? (val.indexOf('\n', end) === -1 ? val.length : val.indexOf('\n', end)) : end;
      const block = val.slice(fl, blockEnd);
      const dedented = block.split('\n').map(l => l.startsWith('  ') ? l.slice(2) : l).join('\n');
      ta.value = val.slice(0, fl) + dedented + val.slice(fl + block.length);
      ta.selectionStart = fl; ta.selectionEnd = fl + dedented.length;
      return;
    }

    // Enter — smart auto-indent
    if (e.key === 'Enter') {
      e.preventDefault();
      const start = ta.selectionStart, val = ta.value;
      const ls = val.lastIndexOf('\n', start - 1) + 1;
      const indent = val.slice(ls, start).match(/^(\s*)/)[1];
      const lastChar = val.slice(start - 1, start);
      const nextChar = val.slice(start, start + 1);
      const extra = (lastChar === '{' || lastChar === '(' || lastChar === ':') ? '  ' : '';
      let closing = '';
      if ((lastChar === '{' && nextChar === '}') || (lastChar === '(' && nextChar === ')')) {
        closing = '\n' + indent;
      }
      ta.value = val.slice(0, start) + '\n' + indent + extra + closing + val.slice(start);
      ta.selectionStart = ta.selectionEnd = start + 1 + indent.length + extra.length;
      return;
    }
  });

  // Bracket/quote auto-close
  textarea.addEventListener('keypress', (e) => {
    const ta = textarea;
    const pairs = { '(': ')', '{': '}', '[': ']', '"': '"', "'": "'", '`': '`' };
    const close = pairs[e.key];
    if (!close) return;
    const start = ta.selectionStart, end = ta.selectionEnd, val = ta.value;
    if (start !== end) {
      e.preventDefault();
      ta.value = val.slice(0, start) + e.key + val.slice(start, end) + close + val.slice(end);
      ta.selectionStart = start + 1; ta.selectionEnd = end + 1;
      return;
    }
    if (e.key === close && val[start] === close) {
      e.preventDefault();
      ta.selectionStart = ta.selectionEnd = start + 1;
      return;
    }
    e.preventDefault();
    ta.value = val.slice(0, start) + e.key + close + val.slice(end);
    ta.selectionStart = ta.selectionEnd = start + 1;
  });

  textarea.addEventListener('blur', () => {
    const newCode = textarea.value;
    const rawCode = decodeURIComponent(codeHeader.dataset.rawCode || '');
    if (newCode !== rawCode) {
      updateBlock(codeHeader, newCode, 'Manual edit', true);
    }
    const codeEl = pre.querySelector('code');
    if (codeEl) {
      codeEl.textContent = newCode;
      codeEl.removeAttribute('data-highlighted');
      safeHighlightElement(codeEl);
    }
    textarea.remove();
    pre.hidden = false;
    if (editBtn) editBtn.textContent = 'Edit';
  });

  pre.hidden = true;
  pre.after(textarea);
  textarea.focus();
  if (editBtn) editBtn.textContent = 'Cancel';
}

// ---------------------------------------------------------------------------
// Structured Code Block Storage
// ---------------------------------------------------------------------------

/**
 * Extracts code blocks from markdown content into structured codeBlocks array.
 * Called once per node on first edit (legacy session migration).
 * Mermaid blocks are excluded — they are rendering-only, not editable.
 *
 * CRITICAL INVARIANT: This function and renderMarkdown must produce block IDs
 * in the same order using the same blockFingerprint + collision-counting
 * algorithm, both skipping mermaid blocks. Any divergence silently breaks
 * getBlock lookups.
 */
export function hydrateCodeBlocks(node) {
  if (node.codeBlocks) return node.codeBlocks;
  if (!node.content) { node.codeBlocks = []; return node.codeBlocks; }

  const blocks = [];
  const seenIds = {};
  const fenceRe = /```(\w*)\s*\n([\s\S]*?)(?:```|$)/g;
  let m;

  while ((m = fenceRe.exec(node.content)) !== null) {
    const lang = m[1].toLowerCase();
    if (lang === 'mermaid') continue;

    const code = m[2].trimEnd();
    let id = blockFingerprint(lang, code);
    if (seenIds[id]) { id += '_' + (++seenIds[id]); }
    else { seenIds[id] = 1; }

    // Migrate existing versions from legacy codeVersions map
    let versions = [code];
    let summaries = ['Original'];
    if (node.codeVersions?.[id]) {
      versions = node.codeVersions[id];
      summaries = node.codeVersionSummaries?.[id] || ['Original'];
    }

    blocks.push({ id, lang: m[1] || '', code: versions[versions.length - 1], versions, summaries });
  }

  node.codeBlocks = blocks;

  // Clean up legacy fields
  if (node.codeVersions) delete node.codeVersions;
  if (node.codeVersionSummaries) delete node.codeVersionSummaries;

  return blocks;
}

/**
 * Rebuilds node.content markdown from codeBlocks array.
 * Walks the existing markdown, replacing each non-mermaid fence body
 * with the corresponding codeBlocks[i].code. Prose and mermaid blocks
 * are preserved exactly.
 */
export function regenerateMarkdown(node) {
  if (!node.codeBlocks || !node.content) return;

  let blockIdx = 0;
  node.content = node.content.replace(/```(\w*)\s*\n([\s\S]*?)(?:```|$)/g, (match, lang) => {
    if (lang.toLowerCase() === 'mermaid') return match;
    const block = node.codeBlocks[blockIdx++];
    if (!block) return match;
    return '```' + (block.lang || lang) + '\n' + block.code + '\n```';
  });
}

/**
 * Returns the codeBlocks entry for a given code header DOM element.
 * Hydrates the node on first access if needed (legacy migration).
 * Returns null if the node isn't in the session tree yet (streaming).
 */
export function getBlock(codeHeader) {
  const node = getSessionNode(codeHeader);
  if (!node) return null;

  const blocks = hydrateCodeBlocks(node);
  const blockId = codeHeader.dataset.blockId;
  return blocks.find(b => b.id === blockId) || null;
}

/**
 * Returns all codeBlocks for the message containing a given code header.
 * Hydrates on first access.
 */
export function getBlocksForMessage(codeHeader) {
  const node = getSessionNode(codeHeader);
  if (!node) return [];
  return hydrateCodeBlocks(node);
}

/**
 * Updates a code block's content, pushes a new version, regenerates markdown, and saves.
 * This is THE write path — all edits (manual, AI, auto-fix, version nav) go through here.
 */
export function updateBlock(codeHeader, newCode, summary = null, addVersion = true) {
  const node = getSessionNode(codeHeader);
  const block = node ? getBlock(codeHeader) : null;

  if (!block) {
    // Node not in tree yet (streaming) — queue for later
    const blockId = codeHeader.dataset.blockId;
    const rawCode = decodeURIComponent(codeHeader.dataset.rawCode || '');
    _pendingBlockEdits.push({
      blockId,
      code: newCode,
      versions: [rawCode, newCode],
      summaries: ['Original', summary || 'Edit'],
    });
    // Still update the DOM immediately
    codeHeader.dataset.rawCode = encodeURIComponent(newCode);
    const copyBtn = codeHeader.querySelector('.copy-code-btn');
    if (copyBtn) copyBtn.dataset.copy = encodeURIComponent(newCode);
    return;
  }

  block.code = newCode;
  if (addVersion) {
    block.versions.push(newCode);
    block.summaries.push(summary || 'Edit');
  }

  // Update DOM
  codeHeader.dataset.rawCode = encodeURIComponent(newCode);
  const copyBtn = codeHeader.querySelector('.copy-code-btn');
  if (copyBtn) copyBtn.dataset.copy = encodeURIComponent(newCode);

  // Regenerate markdown from structured data
  regenerateMarkdown(node);
  _saveSessions();
}

export function getSessionNode(codeHeader) {
  const msgEl = codeHeader.closest('[data-node-id]');
  if (!msgEl) return null;
  const nodeId = msgEl.dataset.nodeId;
  if (!nodeId || nodeId === 'pending') return null;
  const session = _getActiveSession();
  return session?.tree[nodeId] || null;
}

export function restoreCodeVersions(container) {
  const headers = (container || document).querySelectorAll('.code-header[data-version-idx]');
  headers.forEach(codeHeader => {
    const block = getBlock(codeHeader);
    if (block && block.versions.length > 1) {
      showVersion(codeHeader, block.versions.length - 1);
    }
  });
}

export function getVersionIdx(codeHeader) {
  return parseInt(codeHeader.dataset.versionIdx || '0', 10);
}

export function showVersion(codeHeader, idx) {
  const block = getBlock(codeHeader);
  if (!block) return;
  if (idx < 0 || idx >= block.versions.length) return;

  codeHeader.dataset.versionIdx = String(idx);

  const code = block.versions[idx];
  const encoded = encodeURIComponent(code);
  codeHeader.dataset.rawCode = encoded;

  const copyBtn = codeHeader.querySelector('.copy-code-btn');
  if (copyBtn) copyBtn.dataset.copy = encoded;

  // Walk past transient bars (progress, ask-ai, plan) inserted between the
  // header and its <pre>; stop if we reach another code block's header.
  let pre = codeHeader.nextElementSibling;
  while (pre && pre.tagName !== 'PRE' && !pre.classList.contains('code-header')) {
    pre = pre.nextElementSibling;
  }
  if (pre && pre.tagName === 'PRE') {
    const codeEl = pre.querySelector('code');
    if (codeEl) {
      codeEl.textContent = code;
      codeEl.removeAttribute('data-highlighted');
      safeHighlightElement(codeEl);
    }
  }

  updateVersionIndicator(codeHeader);

  // Sync block's current code and regenerate markdown (no new version entry)
  block.code = code;
  const node = getSessionNode(codeHeader);
  if (node) { regenerateMarkdown(node); _saveSessions(); }
}

export function updateVersionIndicator(codeHeader) {
  const block = getBlock(codeHeader);
  const versions = block ? block.versions : [];
  const summaries = block ? block.summaries : [];
  const idx = getVersionIdx(codeHeader);

  let indicator = codeHeader.querySelector('.code-version-nav');
  if (versions.length <= 1) {
    if (indicator) indicator.remove();
    return;
  }

  if (!indicator) {
    indicator = document.createElement('div');
    indicator.className = 'code-version-nav';
    const langSpan = codeHeader.querySelector(':scope > span');
    if (langSpan) langSpan.after(indicator);
    else codeHeader.prepend(indicator);
  }

  const summary = summaries[idx] || '';
  const escapedSummary = summary.replace(/"/g, '&quot;').replace(/</g, '&lt;');
  const prevSummary = idx > 0 ? (summaries[idx - 1] || '').replace(/"/g, '&quot;').replace(/</g, '&lt;') : '';
  const nextSummary = idx < versions.length - 1 ? (summaries[idx + 1] || '').replace(/"/g, '&quot;').replace(/</g, '&lt;') : '';

  indicator.innerHTML = `
    <button class="code-version-btn" data-action="version-prev" ${idx === 0 ? 'disabled' : ''} title="${prevSummary || 'Previous version'}">&lsaquo;</button>
    <span class="code-version-label" title="${escapedSummary}">v${idx + 1}/${versions.length}</span>
    <button class="code-version-btn" data-action="version-next" ${idx === versions.length - 1 ? 'disabled' : ''} title="${nextSummary || 'Next version'}">&rsaquo;</button>
  `;

  let summaryEl = codeHeader.querySelector('.code-version-summary');
  if (summary && summary !== 'Original') {
    if (!summaryEl) {
      summaryEl = document.createElement('span');
      summaryEl.className = 'code-version-summary';
      indicator.after(summaryEl);
    }
    summaryEl.textContent = summary;
  } else if (summaryEl) {
    summaryEl.remove();
  }
}

// ---------------------------------------------------------------------------
// Code Download
// ---------------------------------------------------------------------------

const _LANG_EXTENSIONS = {
  html: 'html', htm: 'html', svg: 'svg',
  python: 'py', py: 'py',
  javascript: 'js', js: 'js', jsx: 'jsx', ts: 'ts', tsx: 'tsx',
  css: 'css', scss: 'scss', less: 'less',
  json: 'json', yaml: 'yaml', yml: 'yml', toml: 'toml',
  markdown: 'md', md: 'md',
  sql: 'sql', sh: 'sh', bash: 'sh', zsh: 'sh',
  rust: 'rs', go: 'go', java: 'java', kotlin: 'kt',
  cpp: 'cpp', c: 'c', cs: 'cs', ruby: 'rb', php: 'php',
  swift: 'swift', dart: 'dart', lua: 'lua', r: 'r',
  xml: 'xml', csv: 'csv', txt: 'txt',
};

export function downloadCodeBlock(codeHeader) {
  const rawCode = decodeURIComponent(codeHeader.dataset.rawCode || '');
  if (!rawCode) { showToast('Nothing in this code block yet.', 'info'); return; }

  const lang = (codeHeader.dataset.lang || '').toLowerCase();
  const ext = _LANG_EXTENSIONS[lang] || lang || 'txt';
  let filename = _inferFilename(rawCode, lang) || 'code';
  if (!filename.includes('.')) filename += '.' + ext;

  let content = rawCode;
  if (lang === 'html' && !rawCode.trim().toLowerCase().startsWith('<!doctype') && !rawCode.trim().toLowerCase().startsWith('<html')) {
    content = `<!DOCTYPE html>\n<html>\n<head><meta charset="UTF-8"><title>${filename}</title></head>\n<body>\n${rawCode}\n</body>\n</html>`;
  }

  const blob = new Blob([content], { type: _getMimeType(ext) });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function _inferFilename(code, lang) {
  if (lang === 'html' || lang === 'htm') {
    const titleMatch = code.match(/<title[^>]*>([^<]+)<\/title>/i);
    if (titleMatch) return titleMatch[1].trim().replace(/[^a-zA-Z0-9_\- ]/g, '').replace(/\s+/g, '-').toLowerCase().slice(0, 40);
  }
  if (lang === 'python' || lang === 'py') {
    const commentMatch = code.match(/^#\s*(.+)/m);
    if (commentMatch) {
      const name = commentMatch[1].trim().replace(/[^a-zA-Z0-9_\- ]/g, '').replace(/\s+/g, '_').toLowerCase().slice(0, 30);
      if (name.length > 3) return name;
    }
  }
  if (['javascript', 'js', 'jsx', 'ts', 'tsx'].includes(lang)) {
    const classMatch = code.match(/(?:export\s+)?class\s+(\w+)/);
    if (classMatch) return classMatch[1];
    const funcMatch = code.match(/(?:export\s+)?function\s+(\w+)/);
    if (funcMatch) return funcMatch[1];
  }
  return null;
}

function _getMimeType(ext) {
  const types = {
    html: 'text/html', htm: 'text/html', svg: 'image/svg+xml',
    js: 'text/javascript', jsx: 'text/javascript', ts: 'text/typescript',
    css: 'text/css', json: 'application/json',
    py: 'text/x-python', rb: 'text/x-ruby', php: 'text/x-php',
    md: 'text/markdown', xml: 'text/xml', csv: 'text/csv',
    sql: 'text/x-sql', sh: 'text/x-sh', yaml: 'text/yaml', yml: 'text/yaml',
  };
  return types[ext] || 'text/plain';
}

// ---------------------------------------------------------------------------
// CodeMind Validation
// ---------------------------------------------------------------------------

/**
 * Validate code with CodeMind AST before presenting to user.
 * Used in the Repair Cascade — fast pre-check before heavyweight fixers.
 */
export async function codeMindValidate(code, lang) {
  if (!CodeMind.isReady()) {
    await CodeMind.init();
  }
  return CodeMind.validate(code, lang);
}

// ---------------------------------------------------------------------------
// Preview Storage Bridge (parent-side)
// ---------------------------------------------------------------------------
// Handles localStorage operations from sandboxed preview iframes via postMessage.
// Data is namespaced under 'preview_' to isolate from the main app's localStorage.

function _initStorageBridge() {
  const PREFIX = 'preview_';
  window.addEventListener('message', (e) => {
    if (!e.data?.type?.startsWith('storage-')) return;
    // Verify the message comes from a sandboxed iframe (not the main page)
    if (e.source === window) return;
    switch (e.data.type) {
      case 'storage-set':
        try { localStorage.setItem(PREFIX + e.data.key, e.data.value); } catch {}
        break;
      case 'storage-remove':
        try { localStorage.removeItem(PREFIX + e.data.key); } catch {}
        break;
      case 'storage-clear':
        try {
          const keys = [];
          for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            if (k?.startsWith(PREFIX)) keys.push(k);
          }
          keys.forEach(k => localStorage.removeItem(k));
        } catch { /* ignore */ }
        break;
      case 'storage-init':
        // Send all preview_ data to the iframe
        try {
          const data = {};
          for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            if (k?.startsWith(PREFIX)) data[k.slice(PREFIX.length)] = localStorage.getItem(k);
          }
          e.source?.postMessage({ type: 'storage-init-response', data }, '*');
        } catch { /* ignore */ }
        break;
    }
  });
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/**
 * Initialize code block action handling. Call once on app boot.
 * Sets up event delegation and the storage bridge.
 *
 * @param {{ getActiveSession: Function, saveSessions: Function }} accessor
 */
export function initCodeBlockActions(accessor) {
  setSessionAccessor(accessor);
  _initStorageBridge();

  // Eagerly load code-edit module so it's ready when user clicks Ask AI
  _loadCodeEdit();
}

// ---------------------------------------------------------------------------
// Expose pendingBlockEdits for stream finalization
// ---------------------------------------------------------------------------

export function getPendingBlockEdits() {
  return _pendingBlockEdits;
}

export function clearPendingBlockEdits() {
  _pendingBlockEdits.length = 0;
}

export function getFixRetryCount() {
  return _fixRetryCount;
}
