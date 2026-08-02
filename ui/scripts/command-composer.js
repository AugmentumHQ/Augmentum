/* ==========================================================================
   Command Composer — Self-designing command system

   The system knows its own capabilities as an action dictionary.
   It observes user patterns, designs useful command chains, and
   proposes them: "Hey, should I learn this? I designed it to do X."

   Architecture:
   - Action Dictionary: every atomic action the system can perform
   - Command Chain: named sequence of actions with parameters
   - Pattern Observer: watches behavior, proposes new commands
   - Command Store: persists learned commands to settings
   ========================================================================== */

import { SurfaceRegistry } from './surface-registry.js';
import { LayoutManager } from './layout-manager.js';
import { SurfaceFlows } from './surface-flows.js';
import { app, showToast } from './app.js';
import { openSurfaceAlongside } from './orb-nav.js';

// ---------------------------------------------------------------------------
// Action Dictionary — every atomic thing the system can do
// ---------------------------------------------------------------------------

const ACTION_DICTIONARY = {
  // --- Surface Management ---
  'surface.open': {
    description: 'Open a new surface',
    params: { type: 'string (chat|narrative|coder|browse|image)', mode: 'string? (passthrough|analytical|agentic)' },
    execute: async ({ type, mode, config }) => {
      // Canonical ladder (orb-nav) — singleton guard, cap toast, boot-race
      // guard, workspace save. Command chains want fresh-session semantics,
      // so session inheritance is off.
      const res = await openSurfaceAlongside({ type: type || 'chat', mode, config, inherit: false });
      return res.ok ? { ok: true, surfaceId: res.surface.id } : { ok: false, reason: res.reason };
    },
  },
  'surface.close': {
    description: 'Close the focused surface (or by type)',
    params: { type: 'string? — close by type instead of focused' },
    execute: async ({ type }) => {
      let target;
      if (type) {
        const surfaces = SurfaceRegistry.ofType(type);
        target = surfaces[0];
      } else {
        target = SurfaceRegistry.getFocused();
      }
      if (!target) return { ok: false, reason: 'no target' };
      if (SurfaceRegistry.all().length <= 1) return { ok: false, reason: 'last surface' };
      if (typeof target.isCloseable === 'function' && !target.isCloseable()) {
        return { ok: false, reason: 'surface not closeable' };
      }
      LayoutManager.closeSurface(target.id);
      return { ok: true };
    },
  },
  'surface.focus': {
    description: 'Switch focus to a surface by type',
    params: { type: 'string (chat|narrative|coder|browse|image)' },
    execute: async ({ type }) => {
      const surfaces = SurfaceRegistry.ofType(type);
      if (surfaces.length === 0) return { ok: false, reason: 'not open' };
      SurfaceRegistry.focus(surfaces[0].id);
      return { ok: true };
    },
  },

  // --- Browse ---
  'browse.search': {
    description: 'Search the web for a query',
    params: { query: 'string' },
    execute: async ({ query }) => {
      document.dispatchEvent(new CustomEvent('augmentum:browse-search', { detail: { query } }));
      return { ok: true };
    },
  },
  'browse.open': {
    description: 'Open a URL in the browse reader',
    params: { url: 'string' },
    execute: async ({ url }) => {
      document.dispatchEvent(new CustomEvent('augmentum:browse-url', { detail: { url } }));
      return { ok: true };
    },
  },

  // --- Image ---
  'image.generate': {
    description: 'Generate an image from a prompt',
    params: { prompt: 'string', style: 'string?' },
    execute: async ({ prompt, style }) => {
      document.dispatchEvent(new CustomEvent('augmentum:generate-image', { detail: { prompt, style } }));
      return { ok: true };
    },
  },

  // --- Voice ---
  'voice.minimize': {
    description: 'Minimize voice call to pill',
    params: {},
    execute: async () => {
      document.dispatchEvent(new CustomEvent('voice:minimize'));
      return { ok: true };
    },
  },

  // --- Mode ---
  'mode.set': {
    description: 'Switch the primary mode',
    params: { mode: 'string (passthrough|analytical|narrative|agentic|coder)' },
    execute: async ({ mode }) => {
      app.setMode(mode);
      return { ok: true };
    },
  },

  // --- Context ---
  'context.inject': {
    description: 'Inject text context into the focused chat surface',
    params: { text: 'string', source: 'string?' },
    execute: async ({ text, source }) => {
      // Route through SurfaceFlows.emit so subscribers (chat surfaces that
      // called onFlow('context', ...)) actually run. Dispatching the raw
      // DOM event only populates the debug bus, not the handler map.
      SurfaceFlows.emit('context', source || 'command', { text });
      return { ok: true };
    },
  },

  // --- UI ---
  'ui.toast': {
    description: 'Show a notification toast',
    params: { message: 'string', type: 'string? (info|success|warning|error)' },
    execute: async ({ message, type }) => {
      showToast(message, type || 'info');
      return { ok: true };
    },
  },
  'ui.inspector': {
    description: 'Toggle the inspector panel',
    params: { visible: 'boolean?' },
    execute: async ({ visible }) => {
      if (visible !== undefined) {
        app.state.inspectorVisible = visible;
        app.dom.app.setAttribute('data-inspector', visible ? 'visible' : 'hidden');
      } else {
        app.toggleInspector();
      }
      return { ok: true };
    },
  },

  // --- Wait ---
  'wait': {
    description: 'Pause between actions (milliseconds)',
    params: { ms: 'number' },
    execute: async ({ ms }) => {
      await new Promise(r => setTimeout(r, ms || 500));
      return { ok: true };
    },
  },
};

// ---------------------------------------------------------------------------
// Command Chain Execution
// ---------------------------------------------------------------------------

/**
 * A command is a named sequence of actions.
 * {
 *   name: "Research Mode",
 *   trigger: "research mode",
 *   description: "Opens analytical chat alongside browse for deep research",
 *   steps: [
 *     { action: "surface.open", params: { type: "chat", mode: "analytical" } },
 *     { action: "wait", params: { ms: 300 } },
 *     { action: "browse.search", params: { query: "$input" } },
 *   ]
 * }
 *
 * $input is replaced with the user's extra text after the trigger phrase.
 */

async function executeCommand(command, input = '') {
  const results = [];
  for (const step of command.steps) {
    const actionDef = ACTION_DICTIONARY[step.action];
    if (!actionDef) {
      results.push({ action: step.action, ok: false, reason: 'unknown action' });
      continue;
    }

    // Replace $input placeholders in params
    const params = {};
    for (const [key, value] of Object.entries(step.params || {})) {
      params[key] = typeof value === 'string' ? value.replace(/\$input/g, input) : value;
    }

    try {
      const result = await actionDef.execute(params);
      results.push({ action: step.action, ...result });
      if (!result.ok) break; // Stop chain on failure
    } catch (err) {
      results.push({ action: step.action, ok: false, reason: err.message });
      break;
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Command Store — persisted learned commands
// ---------------------------------------------------------------------------

let _commands = [];  // Learned command chains
const STORAGE_KEY = 'augmentum_learned_commands';

function _loadCommands() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    _commands = raw ? JSON.parse(raw) : [];
  } catch {
    _commands = [];
  }
}

function _saveCommands() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(_commands));
  } catch { /* quota */ }
}

function addCommand(command) {
  // Validate
  if (!command.name || !command.trigger || !command.steps?.length) return false;
  for (const step of command.steps) {
    if (!ACTION_DICTIONARY[step.action]) return false;
  }
  // Remove existing with same trigger
  _commands = _commands.filter(c => c.trigger !== command.trigger);
  _commands.push(command);
  _saveCommands();
  return true;
}

function removeCommand(trigger) {
  _commands = _commands.filter(c => c.trigger !== trigger);
  _saveCommands();
}

function getCommands() {
  return [..._commands];
}

// ---------------------------------------------------------------------------
// Pattern Observer — watches behavior, designs commands
// ---------------------------------------------------------------------------

const _actionLog = [];       // Recent user actions
const MAX_LOG = 100;
const _proposedOnce = new Set(); // Don't re-propose dismissed commands

/** Record a user action for pattern detection. */
function recordAction(action, params = {}) {
  _actionLog.push({
    action,
    params,
    timestamp: Date.now(),
  });
  if (_actionLog.length > MAX_LOG) _actionLog.shift();
}

/**
 * Analyze recent actions and propose useful commands.
 * Returns array of { command, reason } proposals.
 */
function analyzePatterns() {
  const proposals = [];
  if (_actionLog.length < 10) return proposals;

  // --- Pattern: user frequently opens same surface pair ---
  const surfaceOpens = _actionLog.filter(a => a.action === 'surface.open');
  const openPairs = {};
  for (let i = 0; i < surfaceOpens.length - 1; i++) {
    const a = surfaceOpens[i];
    const b = surfaceOpens[i + 1];
    // Within 30 seconds = likely intentional pair
    if (b.timestamp - a.timestamp < 30000) {
      const key = `${a.params.type || 'chat'}+${b.params.type || 'chat'}`;
      openPairs[key] = (openPairs[key] || 0) + 1;
    }
  }
  for (const [pair, count] of Object.entries(openPairs)) {
    if (count >= 2 && !_proposedOnce.has(pair)) {
      const [typeA, typeB] = pair.split('+');
      const nameA = typeA.charAt(0).toUpperCase() + typeA.slice(1);
      const nameB = typeB.charAt(0).toUpperCase() + typeB.slice(1);
      proposals.push({
        command: {
          name: `${nameA} + ${nameB}`,
          trigger: `${typeA} and ${typeB}`,
          description: `Open ${nameA} and ${nameB} side by side`,
          steps: [
            { action: 'surface.open', params: { type: typeA } },
            { action: 'wait', params: { ms: 300 } },
            { action: 'surface.open', params: { type: typeB } },
            { action: 'surface.focus', params: { type: typeA } },
          ],
        },
        reason: `You've opened ${nameA} and ${nameB} together ${count} times`,
      });
    }
  }

  // --- Pattern: user searches after switching to analytical ---
  const modeSearchPairs = _actionLog.filter(a =>
    a.action === 'mode.set' && a.params.mode === 'analytical'
  );
  let searchAfterAnalytical = 0;
  for (const modeSet of modeSearchPairs) {
    const nextSearch = _actionLog.find(a =>
      a.action === 'browse.search' && a.timestamp > modeSet.timestamp && a.timestamp - modeSet.timestamp < 60000
    );
    if (nextSearch) searchAfterAnalytical++;
  }
  if (searchAfterAnalytical >= 2 && !_proposedOnce.has('analytical+search')) {
    proposals.push({
      command: {
        name: 'Research Mode',
        trigger: 'research mode',
        description: 'Switch to analytical mode and open browse for research',
        steps: [
          { action: 'surface.open', params: { type: 'chat', mode: 'analytical' } },
          { action: 'wait', params: { ms: 300 } },
          { action: 'browse.search', params: { query: '$input' } },
        ],
      },
      reason: `You often search the web right after switching to analytical mode`,
    });
  }

  return proposals;
}

/** Mark a proposal as dismissed (don't re-propose). */
function dismissProposal(triggerOrPairKey) {
  _proposedOnce.add(triggerOrPairKey);
}

// ---------------------------------------------------------------------------
// Command Matching (for voice/text input)
// ---------------------------------------------------------------------------

/**
 * Try to match text against learned commands.
 * Returns { command, input } if matched, null otherwise.
 */
function matchCommand(text) {
  if (!text) return null;
  const lower = text.toLowerCase().trim();
  for (const cmd of _commands) {
    const trigger = cmd.trigger.toLowerCase();
    if (lower.startsWith(trigger)) {
      const input = lower.slice(trigger.length).trim();
      return { command: cmd, input };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

function init() {
  _loadCommands();

  // Pre-seed with useful default commands if none exist
  if (_commands.length === 0) {
    addCommand({
      name: 'Research Mode',
      trigger: 'research mode',
      description: 'Open analytical chat alongside browse for deep research',
      steps: [
        { action: 'surface.open', params: { type: 'chat', mode: 'analytical' } },
        { action: 'wait', params: { ms: 300 } },
        { action: 'ui.toast', params: { message: 'Research mode active — search or ask questions', type: 'info' } },
      ],
    });
    addCommand({
      name: 'Creative Writing',
      trigger: 'writing mode',
      description: 'Open narrative surface with image panel for illustrated storytelling',
      steps: [
        { action: 'surface.open', params: { type: 'narrative' } },
        { action: 'ui.toast', params: { message: 'Creative writing mode — start your story', type: 'info' } },
      ],
    });
    addCommand({
      name: 'Code + Chat',
      trigger: 'coding mode',
      description: 'Open coder alongside chat for AI-assisted development',
      steps: [
        { action: 'surface.open', params: { type: 'coder' } },
        { action: 'wait', params: { ms: 300 } },
        { action: 'surface.open', params: { type: 'chat', mode: 'passthrough' } },
        { action: 'surface.focus', params: { type: 'coder' } },
      ],
    });
  }

  // Wire into voice command system — check learned commands before built-in
  document.addEventListener('voice:pre-send', (e) => {
    if (e.defaultPrevented) return; // Already handled by built-in commands
    const text = e.detail?.text;
    if (!text) return;
    const match = matchCommand(text);
    if (match) {
      e.preventDefault();
      executeCommand(match.command, match.input).then(results => {
        const failed = results.find(r => !r.ok);
        if (failed) {
          showToast(`Command failed: ${failed.reason}`, 'warning');
        }
      });
    }
  }, { capture: false }); // Run after built-in voice commands

  // Record surface opens for pattern detection
  document.addEventListener('surface:focus-changed', (e) => {
    recordAction('surface.focus', { type: e.detail.type, mode: e.detail.mode });
  });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export const CommandComposer = {
  init,
  executeCommand,
  addCommand,
  removeCommand,
  getCommands,
  matchCommand,
  recordAction,
  analyzePatterns,
  dismissProposal,
  getDictionary: () => ({ ...ACTION_DICTIONARY }),
};

export { ACTION_DICTIONARY };
