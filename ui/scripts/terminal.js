/**
 * terminal.js — xterm.js wrapper for Coder mode.
 *
 * Lazy-loads xterm.js from CDN (same pattern as cm-editor.js).
 * Creates terminal instances connected to workspace containers via WebSocket.
 */

import { getWsTicket } from './auth.js';
import { copyToClipboard } from './clipboard.js';

let _loaded = false;
let _Terminal, _FitAddon, _WebLinksAddon, _SearchAddon, _WebglAddon;

const _instances = new Map(); // sessionId → { terminal, fitAddon, ws, resizeObserver }

/**
 * Lazy-load xterm.js and addons from CDN.
 */
export async function load() {
  if (_loaded) return;

  const [
    xtermCore,
    fitMod,
    webLinksMod,
    searchMod,
  ] = await Promise.all([
    import('https://esm.sh/@xterm/xterm@6.0.0'),
    import('https://esm.sh/@xterm/addon-fit@0.11.0'),
    import('https://esm.sh/@xterm/addon-web-links@0.12.0'),
    import('https://esm.sh/@xterm/addon-search@0.16.0'),
  ]);

  // WebGL renderer — loaded separately and best-effort: the terminal is
  // fully functional on the default DOM renderer, just much slower under
  // output floods (every write mutates spans; the GPU renderer makes
  // sustained build/log output cheap). A CDN miss here must never take
  // the terminal down with it.
  try {
    const webglMod = await import('https://esm.sh/@xterm/addon-webgl@0.19.0');
    _WebglAddon = webglMod.WebglAddon;
  } catch {
    _WebglAddon = null;
  }

  _Terminal = xtermCore.Terminal;
  _FitAddon = fitMod.FitAddon;
  _WebLinksAddon = webLinksMod.WebLinksAddon;
  _SearchAddon = searchMod.SearchAddon;
  _loaded = true;

  // Load xterm CSS — await it so the terminal doesn't render before
  // its stylesheet is applied (missing CSS → invisible canvas)
  if (!document.getElementById('xterm-css')) {
    await new Promise((resolve) => {
      const link = document.createElement('link');
      link.id = 'xterm-css';
      link.rel = 'stylesheet';
      link.href = 'https://esm.sh/@xterm/xterm@6.0.0/css/xterm.css';
      link.onload = resolve;
      link.onerror = resolve; // proceed even if CSS fails
      document.head.appendChild(link);
      // Safety timeout
      setTimeout(resolve, 3000);
    });
  }
}

// Curated 16-color ANSI palettes. xterm needs concrete colors for the
// palette, and CSS vars only cover bg/fg — the bright codes assume a DARK
// canvas, so a single hardcoded palette renders bright-white and the
// pastels invisible on the light theme's near-white --bg. Pick the palette
// by background luminance so program output stays legible in every theme.
const _ANSI_DARK = {
  black: '#1c1c2e', red: '#ef4444', green: '#22c55e', yellow: '#eab308',
  blue: '#6c8aff', magenta: '#c084fc', cyan: '#22d3ee', white: '#ececf1',
  brightBlack: '#71717a', brightRed: '#f87171', brightGreen: '#4ade80',
  brightYellow: '#facc15', brightBlue: '#93c5fd', brightMagenta: '#d8b4fe',
  brightCyan: '#67e8f9', brightWhite: '#ffffff',
};
// Light-canvas palette: darker, saturated colors that meet contrast on a
// near-white background; the white/bright-white ends collapse to ink so a
// "bright white" SGR code doesn't paint invisible text on white.
const _ANSI_LIGHT = {
  black: '#1c1917', red: '#c0392b', green: '#1e7a3c', yellow: '#9a6700',
  blue: '#2950c8', magenta: '#9333ea', cyan: '#0e7490', white: '#4a4a4a',
  brightBlack: '#6b6b80', brightRed: '#dc2626', brightGreen: '#15803d',
  brightYellow: '#b45309', brightBlue: '#1d4ed8', brightMagenta: '#7c3aed',
  brightCyan: '#0891b2', brightWhite: '#1c1917',
};

/** Perceived-luminance test (Rec.601) — true for light backgrounds. */
function _isLightColor(color) {
  let r, g, b;
  const hex = (color || '').replace('#', '').trim();
  if (/^[0-9a-f]{3}$/i.test(hex)) {
    r = parseInt(hex[0] + hex[0], 16);
    g = parseInt(hex[1] + hex[1], 16);
    b = parseInt(hex[2] + hex[2], 16);
  } else if (/^[0-9a-f]{6}$/i.test(hex)) {
    r = parseInt(hex.slice(0, 2), 16);
    g = parseInt(hex.slice(2, 4), 16);
    b = parseInt(hex.slice(4, 6), 16);
  } else {
    const m = (color || '').match(/(\d+)[,\s]+(\d+)[,\s]+(\d+)/);
    if (!m) return false;
    [, r, g, b] = m.map(Number);
  }
  return (0.299 * r + 0.587 * g + 0.114 * b) > 140;
}

/**
 * Build xterm.js theme from Augmentum CSS variables.
 */
function _buildTheme() {
  // Read from #app, not documentElement: the coder-scoped theme override
  // (data-coder-theme on .app) redefines --bg/--text-primary there, and the
  // app-wide theme cascades through it either way.
  const s = getComputedStyle(document.getElementById('app') || document.documentElement);
  const v = (name) => s.getPropertyValue(name).trim();
  const bg = v('--bg') || '#0f0f1a';
  const isLight = _isLightColor(bg);
  const fg = v('--text-primary') || (isLight ? '#1c1917' : '#ececf1');
  return {
    background: bg,
    foreground: fg,
    cursor: v('--accent') || '#6c8aff',
    cursorAccent: bg,
    selectionBackground:
      v('--surface-3') || v('--surface-active') || (isLight ? '#dfdcda' : '#2d2d45'),
    selectionForeground: fg,
    ...(isLight ? _ANSI_LIGHT : _ANSI_DARK),
  };
}

/**
 * Create a terminal instance connected to a workspace container.
 *
 * @param {HTMLElement} container - DOM element to mount terminal into
 * @param {string} workspaceId - Workspace ID for WebSocket connection
 * @returns {string} Session ID for this terminal instance
 */
export async function create(container, workspaceId) {
  if (!_loaded) await load();

  const sessionId = `term_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  const fitAddon = new _FitAddon();
  const terminal = new _Terminal({
    theme: _buildTheme(),
    fontFamily: '"JetBrains Mono", "Cascadia Code", "SF Mono", monospace',
    fontSize: parseInt(localStorage.getItem('augmentum-terminal-font-size')) || 14,
    cursorBlink: true,
    cursorStyle: 'bar',
    scrollback: 5000,
    allowProposedApi: true,
    overviewRulerWidth: 10,
  });

  terminal.loadAddon(fitAddon);
  terminal.loadAddon(new _WebLinksAddon());
  terminal.loadAddon(new _SearchAddon());

  // Wait for the container to have non-zero dimensions before opening
  // xterm — fitAddon.fit() computes rows/cols from clientWidth/Height,
  // and a 0×0 measurement produces an invisible terminal that never
  // recovers (the ResizeObserver won't fire if size stays 0→0).
  if (!container.clientHeight || !container.clientWidth) {
    await new Promise((resolve) => {
      const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          if (entry.contentRect.width > 0 && entry.contentRect.height > 0) {
            ro.disconnect();
            resolve();
            return;
          }
        }
      });
      ro.observe(container);
      // Safety timeout — don't wait forever
      setTimeout(() => { ro.disconnect(); resolve(); }, 3000);
    });
  }

  terminal.open(container);
  fitAddon.fit();

  // GPU renderer (must load AFTER open() — the addon binds to the live
  // canvas). Fully optional: WebGL unavailable (headless, old GPU,
  // blocklisted driver) throws here and we stay on the DOM renderer with
  // identical behavior. On context loss (GPU reset, tab backgrounded too
  // long on some drivers) xterm's documented recovery is to dispose the
  // addon, which transparently falls back to the DOM renderer.
  try {
    if (_WebglAddon) {
      const webgl = new _WebglAddon();
      webgl.onContextLoss(() => {
        try { webgl.dispose(); } catch { /* already disposed */ }
      });
      terminal.loadAddon(webgl);
    }
  } catch {
    // DOM renderer fallback — no user-visible difference beyond speed.
  }

  // Connection state — shared by connectWs(), input handlers, and destroy().
  // Input handlers read `ws` from the outer closure; reassignment inside
  // connectWs() propagates automatically via JS closure-over-variable.
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const MAX_RECONNECT = 4;
  let ws = null;
  let reconnectAttempt = 0;
  let reconnectTimer = null;
  let destroyed = false;
  let hadConnection = false;

  // Debounced terminal output event for intent bar updates
  let _outputDebounce = null;
  const _emitOutput = () => {
    clearTimeout(_outputDebounce);
    _outputDebounce = setTimeout(() => {
      document.dispatchEvent(new CustomEvent('coder:terminal-output', { detail: { sessionId } }));
    }, 500);
  };

  const scheduleReconnect = (reason) => {
    if (destroyed) return;
    if (reconnectAttempt >= MAX_RECONNECT) {
      // Retry budget exhausted — tell coder.js to clear its cached terminal
      // id so the next _onEnterCoderMode() can rebuild from scratch. Without
      // this event, the stale _activeTerminalId blocks re-creation and the
      // user stares at a dead canvas until they recreate the workspace.
      terminal.writeln(`\r\n\x1b[31m${reason} — could not reconnect after ${MAX_RECONNECT} attempts.\x1b[0m`);
      terminal.writeln('\x1b[90mThe container may be stopped. Use the workspace selector to recreate, or reload the page.\x1b[0m');
      document.dispatchEvent(new CustomEvent('coder:terminal-disconnected', {
        detail: { sessionId, workspaceId, reason },
      }));
      return;
    }
    reconnectAttempt += 1;
    const delay = 500 * Math.pow(2, reconnectAttempt - 1); // 500, 1000, 2000, 4000
    terminal.writeln(`\r\n\x1b[33m${reason} — reconnecting (${reconnectAttempt}/${MAX_RECONNECT}) in ${Math.round(delay / 1000) || 1}s...\x1b[0m`);
    reconnectTimer = setTimeout(() => { reconnectTimer = null; connectWs(); }, delay);
  };

  const connectWs = async () => {
    if (destroyed) return;
    let ticket;
    try {
      ticket = await getWsTicket();
    } catch (err) {
      // Surface the actual HTTP status so the user (and we) can tell
      // 401 (cookie expired / never set) from 403 (CSRF Origin mismatch)
      // from 503 (auth subsystem down) from a network failure (server
      // unreachable). Generic "auth ticket failed" was hiding which
      // wire it was — the three causes have different fixes.
      const status = err?.status;
      let detail;
      if (status === 401) {
        detail = 'session expired (401) — sign in again';
      } else if (status === 403) {
        detail = 'CSRF blocked (403) — origin/referer mismatch, check console';
      } else if (status === 503) {
        detail = 'auth unavailable (503) — server is restarting';
      } else if (status) {
        detail = `HTTP ${status}`;
      } else {
        detail = 'no response (server unreachable)';
      }
      terminal.writeln(`\r\n\x1b[31mFailed to get auth ticket: ${detail}\x1b[0m`);
      scheduleReconnect(`Auth ticket failed (${detail})`);
      return;
    }
    if (destroyed) return;
    const url = `${proto}://${location.host}/ws/terminal/${workspaceId}?ticket=${encodeURIComponent(ticket)}`;
    let next;
    try {
      next = new WebSocket(url);
    } catch (err) {
      scheduleReconnect(`Connection failed: ${err.message || 'socket error'}`);
      return;
    }
    next.binaryType = 'arraybuffer';
    ws = next;

    ws.onopen = () => {
      reconnectAttempt = 0;
      if (hadConnection) {
        terminal.writeln('\r\n\x1b[32mReconnected\x1b[0m\r\n');
      } else {
        terminal.writeln('\x1b[32mConnected to workspace\x1b[0m\r\n');
        hadConnection = true;
      }
      const { cols, rows } = terminal;
      try { ws.send(JSON.stringify({ type: 'resize', cols, rows })); } catch { /* ignore */ }
      document.dispatchEvent(new CustomEvent('coder:terminal-connected', {
        detail: { sessionId, workspaceId },
      }));
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        terminal.write(new Uint8Array(event.data));
        _emitOutput();
      }
    };

    // onerror fires before onclose on failure — let onclose drive the
    // retry logic so we don't emit duplicate "reconnecting" lines.
    ws.onerror = () => { /* noop — handled in onclose */ };

    ws.onclose = (event) => {
      if (destroyed) return;
      // Always surface a close reason. Pre-2026-04-21 code 1006 (abnormal
      // close, no close frame) was silently swallowed — which is precisely
      // the "container died / proxy reset / network blip" case the user
      // needs to see, and was the root cause of silently-empty terminals.
      const reason = event.reason
        || (event.code === 1006 ? 'Connection lost'
          : event.code === 1011 ? 'Server error (container may not be running)'
          : event.code === 4403 ? 'Unauthorized'
          : `Disconnected (code ${event.code})`);
      scheduleReconnect(reason);
    };
  };

  // Open the initial connection. If this fails synchronously it still
  // schedules reconnection and the terminal stays alive for retry.
  await connectWs();

  // Keyboard overrides — Ctrl+C copies when text is selected, Ctrl+V pastes
  terminal.attachCustomKeyEventHandler((e) => {
    // Escape OR Ctrl+C during agent execution → cancel.
    // Pre-2026-04-20 only Escape was wired, so users hitting ^C on a
    // runaway turn would see nothing happen (keystrokes went to the
    // shell, not the agent-cancel event). Observed in a "what's in
    // this project?" thrash where the user had to hit ^C five times
    // and still couldn't stop the loop. ^C is the intuitive cancel
    // key; Escape stays as an alias. During agent execution we
    // intercept ^C regardless of selection — the stuck loop matters
    // more than a copy-to-clipboard shortcut.
    const inst = _instances.get(sessionId);
    const agentActive = !!inst?.agentActive;
    if (agentActive && (
      e.key === 'Escape'
      || (e.ctrlKey && (e.key === 'c' || e.key === 'C'))
    )) {
      document.dispatchEvent(new CustomEvent('coder:agent-cancel'));
      return false;
    }
    // Ctrl+= / Ctrl+- to adjust terminal font size
    if (e.ctrlKey && (e.key === '=' || e.key === '+')) {
      const size = Math.min(24, (terminal.options.fontSize || 14) + 1);
      terminal.options.fontSize = size;
      localStorage.setItem('augmentum-terminal-font-size', String(size));
      fitAddon.fit();
      return false;
    }
    if (e.ctrlKey && e.key === '-') {
      const size = Math.max(8, (terminal.options.fontSize || 14) - 1);
      terminal.options.fontSize = size;
      localStorage.setItem('augmentum-terminal-font-size', String(size));
      fitAddon.fit();
      return false;
    }
    // Ctrl+C with selection → browser copy
    if (e.ctrlKey && e.key === 'c' && terminal.hasSelection()) {
      copyToClipboard(terminal.getSelection());
      terminal.clearSelection();
      return false; // prevent sending to terminal
    }
    // Ctrl+V → paste from clipboard into terminal. Async readText is the
    // only way to read the clipboard — there is no execCommand fallback,
    // so on browsers without it we silently no-op and let the user use
    // the terminal's native paste.
    if (e.ctrlKey && e.key === 'v') {
      if (navigator.clipboard?.readText) {
        navigator.clipboard.readText().then(text => {
          if (text && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'input', data: text }));
          }
        }).catch(() => {});
      }
      return false;
    }
    // Ctrl+A → select all terminal content
    if (e.ctrlKey && e.key === 'a') {
      terminal.selectAll();
      return false;
    }
    return true; // all other keys pass through to terminal
  });

  // Forward terminal input to WebSocket — intercept // prefix for agent mode
  // Read the CURRENT terminal line when Enter is pressed instead of tracking keystrokes
  terminal.onData((data) => {
    const inst = _instances.get(sessionId);
    if (inst?.agentActive) return;

    if (data === '\r' || data === '\n') {
      // Enter pressed — assemble the full logical command from the terminal
      // buffer. A long command wraps across visual lines; reading only the
      // cursor's row drops the prompt and the head of the command. Walk
      // backwards until we find a prompt-bearing line, then join forward.
      const buf = terminal.buffer.active;
      const cursorY = buf.cursorY + buf.viewportY;
      const PROMPT_RE = /[$#>]\s*(.*)$/;
      let promptRow = cursorY;
      let promptMatch = null;
      for (let i = 0; i < 40 && promptRow >= 0; i++, promptRow--) {
        const text = buf.getLine(promptRow)?.translateToString(true) ?? '';
        const m = text.match(PROMPT_RE);
        if (m) {
          promptMatch = m;
          break;
        }
      }
      let cmd = '';
      if (promptMatch) {
        const parts = [promptMatch[1]];
        for (let row = promptRow + 1; row <= cursorY; row++) {
          const text = buf.getLine(row)?.translateToString(true) ?? '';
          parts.push(text);
        }
        cmd = parts.join('').trim();
      } else {
        const fallback = buf.getLine(cursorY)?.translateToString(true).trim() ?? '';
        cmd = fallback;
      }

      // Tolerate `// foo`, `//foo`, or stray leading slashes/whitespace —
      // anything that starts with two slashes and has a non-empty tail is
      // an agent request, regardless of spacing.
      const agentMatch = cmd.match(/^\/{2,}\s*(\S.*)$/);
      if (agentMatch) {
        const request = agentMatch[1].trim();
        if (request) {
          // Agent mode — don't send Enter to shell
          // Clear the line in the terminal (send Ctrl+U to kill the line, then our output)
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'input', data: '\x15' })); // Ctrl+U clears line
          }
          terminal.write('\r\n');
          document.dispatchEvent(new CustomEvent('coder:agent-request', {
            detail: { sessionId, request },
          }));
          return;
        }
      }

      // Normal command — send Enter to shell
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'input', data }));
      }
    } else {
      // All non-Enter input: pass through to shell directly
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'input', data }));
      }
    }
  });

  // Forward resize events
  terminal.onResize(({ cols, rows }) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'resize', cols, rows }));
    }
  });

  // Touch scrollback — xterm.js canvas captures touch events, so we
  // manually translate vertical swipes into terminal scrolling.
  let _touchStartY = null;
  container.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) _touchStartY = e.touches[0].clientY;
  }, { passive: true });
  container.addEventListener('touchmove', (e) => {
    if (_touchStartY === null || e.touches.length !== 1) return;
    const dy = _touchStartY - e.touches[0].clientY;
    _touchStartY = e.touches[0].clientY;
    const lines = Math.round(dy / 20); // ~20px per terminal line
    if (lines !== 0) terminal.scrollLines(lines);
  }, { passive: true });
  container.addEventListener('touchend', () => { _touchStartY = null; }, { passive: true });

  // Auto-fit on container resize. If the terminal was opened into a
  // 0×0 container (common when coder mode mounts before the grid has
  // reflowed, or when the terminal pane starts hidden on mobile), some
  // xterm.js versions cache the degenerate init state and never repaint
  // after a later fit(). Detect the 0→visible transition and force an
  // explicit refresh so the terminal draws its first frame.
  let _hadDimensions = container.clientWidth > 0 && container.clientHeight > 0;
  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const w = entry.contentRect.width;
      const h = entry.contentRect.height;
      try { fitAddon.fit(); } catch { /* ignore */ }
      if (!_hadDimensions && w > 0 && h > 0) {
        _hadDimensions = true;
        try { terminal.refresh(0, Math.max(0, terminal.rows - 1)); } catch { /* ignore */ }
      }
    }
  });
  resizeObserver.observe(container);

  // Re-fit xterm when the on-screen keyboard opens/closes. Keyboard
  // height is handled natively by interactive-widget=resizes-content
  // in the viewport meta (100dvh shrinks when keyboard opens).
  if (window.visualViewport) {
    const _onViewportResize = () => {
      if (window.innerWidth >= 768) return;
      try { fitAddon.fit(); } catch { /* ignore */ }
    };
    window.visualViewport.addEventListener('resize', _onViewportResize);
    container._vpHandler = _onViewportResize;
  }

  // `getWs` returns the live socket (not a snapshot) so reconnects
  // transparently update what destroy() / sendInput() see.
  const instData = {
    terminal,
    fitAddon,
    resizeObserver,
    agentActive: false,
    getWs: () => ws,
    teardown: () => {
      destroyed = true;
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
      if (ws) { try { ws.close(); } catch { /* ignore */ } }
    },
  };
  _instances.set(sessionId, instData);

  return sessionId;
}

/** Destroy a terminal instance and close its WebSocket. */
export function destroy(sessionId) {
  const inst = _instances.get(sessionId);
  if (!inst) return;
  inst.resizeObserver.disconnect();
  inst.teardown();
  inst.terminal.dispose();
  _instances.delete(sessionId);
}

/** Focus a terminal instance (skip on mobile to avoid popping the virtual keyboard). */
export function focus(sessionId) {
  if (window.innerWidth < 768) return;
  const inst = _instances.get(sessionId);
  if (inst) inst.terminal.focus();
}

/** Refit terminal to container (call after layout changes). */
export function fit(sessionId) {
  const inst = _instances.get(sessionId);
  if (inst) inst.fitAddon.fit();
}

/** Update theme for all terminal instances (call on theme change). */
export function updateTheme() {
  const theme = _buildTheme();
  for (const inst of _instances.values()) {
    inst.terminal.options.theme = theme;
  }
}

/** Get the raw terminal instance (for attaching temporary onData listeners). */
export function getTerminalInstance(sessionId) {
  const inst = _instances.get(sessionId);
  return inst ? inst.terminal : null;
}

/** Send input data to a terminal's WebSocket. */
export function sendInput(sessionId, data) {
  const inst = _instances.get(sessionId);
  if (!inst) return;
  const ws = inst.getWs();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'input', data }));
  }
}

/** Destroy all terminal instances. */
/** Write text/ANSI directly to the terminal display (not to the shell). */
export function write(sessionId, data) {
  const inst = _instances.get(sessionId);
  if (inst) inst.terminal.write(data);
}

function _logicalScrollbackLines(buf) {
  const lines = [];

  for (let row = 0; row < buf.length; row++) {
    const line = buf.getLine(row);
    if (!line) continue;

    const text = line.translateToString(false);
    if (line.isWrapped && lines.length > 0) {
      lines[lines.length - 1] += text;
      continue;
    }

    lines.push(text);
  }

  return lines.map(line => line.replace(/\s+$/u, ''));
}

/** Get the last N lines of terminal scrollback as plain text. */
export function getScrollback(sessionId, lines = 50) {
  const inst = _instances.get(sessionId);
  if (!inst) return '';
  const limit = Math.max(0, lines);
  if (limit === 0) return '';
  const buf = inst.terminal.buffer.active;
  const result = _logicalScrollbackLines(buf).slice(-limit);
  // Trim trailing empty lines
  while (result.length > 0 && !result[result.length - 1].trim()) result.pop();
  return result.join('\n');
}

/** Set agent active flag — blocks terminal input during agent execution. */
export function setAgentActive(sessionId, active) {
  const inst = _instances.get(sessionId);
  if (inst) inst.agentActive = active;
}

export function destroyAll() {
  for (const id of _instances.keys()) {
    destroy(id);
  }
}
