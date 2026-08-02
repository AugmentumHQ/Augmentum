/* ==========================================================================
   Voice Commands — Route voice instructions to surface actions
   Phase 3 foundation of the workspace architecture.

   Listens for recognized voice commands and translates them into
   surface creation, switching, and cross-surface actions.

   Commands are detected from the user's transcript text after STT
   processes it. This runs BEFORE the text is sent to the LLM, so
   commands that match are intercepted and executed locally.

   Examples:
     "open story with Luna"     → create narrative surface, load Luna
     "open chat"                → create/switch to chat surface
     "search for medieval art"  → open browse, run search
     "show my code"             → switch to coder surface
     "close this"               → close focused surface
     "go back to chat"          → switch to chat tab
     "generate an image of..."  → open image panel, start generation
   ========================================================================== */

import { SurfaceRegistry } from './surface-registry.js';
import { LayoutManager } from './layout-manager.js';
import { app, showToast } from './app.js';
import { openSurfaceAlongside } from './orb-nav.js';

// ---------------------------------------------------------------------------
// Command Patterns
// ---------------------------------------------------------------------------

const COMMANDS = [
  {
    pattern: /^(?:open|start|new)\s+(?:a\s+)?(?:chat|conversation)/i,
    action: 'open-surface',
    surface: 'chat',
    mode: 'passthrough',
  },
  {
    pattern: /^(?:open|start|new)\s+(?:an?\s+)?analy/i,
    action: 'open-surface',
    surface: 'chat',
    mode: 'analytical',
  },
  {
    pattern: /^(?:open|start|new)\s+(?:a\s+)?(?:story|narrative|roleplay)(?:\s+(?:with|about)\s+(.+))?/i,
    action: 'open-narrative',
  },
  {
    pattern: /^(?:open|start|show)\s+(?:my\s+)?(?:code|coder|terminal|editor)/i,
    action: 'open-surface',
    surface: 'coder',
    mode: 'coder',
  },
  {
    pattern: /^(?:search|look up|find|browse)\s+(?:for\s+)?(.+)/i,
    action: 'search',
  },
  {
    pattern: /^(?:close|dismiss|remove)\s+(?:this|current|focused)/i,
    action: 'close-focused',
  },
  {
    pattern: /^(?:go\s+)?(?:back\s+)?(?:to|switch\s+to)\s+(?:the\s+)?(?:chat|conversation)/i,
    action: 'focus-type',
    surface: 'chat',
  },
  {
    pattern: /^(?:go\s+)?(?:back\s+)?(?:to|switch\s+to)\s+(?:the\s+)?(?:story|narrative)/i,
    action: 'focus-type',
    surface: 'narrative',
  },
  {
    pattern: /^(?:go\s+)?(?:back\s+)?(?:to|switch\s+to)\s+(?:the\s+)?(?:code|coder)/i,
    action: 'focus-type',
    surface: 'coder',
  },
  {
    pattern: /^(?:generate|create|make)\s+(?:an?\s+)?image\s+(?:of\s+)?(.+)/i,
    action: 'generate-image',
  },
];

// ---------------------------------------------------------------------------
// Command Execution
// ---------------------------------------------------------------------------

function _executeCommand(cmd, match) {
  switch (cmd.action) {
    case 'open-surface': {
      // Canonical ladder (orb-nav): gains the singleton guard this path was
      // missing — "open coder" with a coder tab already open used to create
      // a second instance that stole the singleton DOM from the first; now
      // it focuses the existing tab with a toast. "new chat" is an explicit
      // fresh-session ask, so inheritance is off.
      openSurfaceAlongside({ type: cmd.surface, mode: cmd.mode, inherit: false }).then((res) => {
        if (res.ok) showToast(`Opened ${cmd.mode || cmd.surface}`, 'info', 2000);
      });
      return true;
    }

    case 'open-narrative': {
      const charName = match[1]?.trim() || '';
      openSurfaceAlongside({
        type: 'narrative',
        mode: 'narrative',
        inherit: false,
        config: { characterName: charName },
      }).then((res) => {
        if (!res.ok) return;
        showToast(charName ? `Opened story with ${charName}` : 'Opened story', 'info', 2000);
        // If character name given, try to start a character chat
        if (charName) {
          document.dispatchEvent(new CustomEvent('voice:find-character', {
            detail: { name: charName, surfaceId: res.surface.id },
          }));
        }
      });
      return true;
    }

    case 'search': {
      const query = match[1]?.trim();
      if (!query) return false;
      // Open browse panel with the search query
      document.dispatchEvent(new CustomEvent('augmentum:browse-search', {
        detail: { query },
      }));
      showToast(`Searching: ${query.slice(0, 40)}`, 'info', 2000);
      return true;
    }

    case 'close-focused': {
      const focused = SurfaceRegistry.getFocused();
      if (!focused) return false;
      // Don't close the last surface
      if (SurfaceRegistry.all().length <= 1) {
        showToast('Cannot close the last surface', 'warning');
        return true;
      }
      if (typeof focused.isCloseable === 'function' && !focused.isCloseable()) {
        showToast('This surface can\'t be closed', 'warning');
        return true;
      }
      LayoutManager.closeSurface(focused.id);
      showToast('Surface closed', 'info', 1500);
      return true;
    }

    case 'focus-type': {
      const surfaces = SurfaceRegistry.ofType(cmd.surface);
      if (surfaces.length > 0) {
        SurfaceRegistry.focus(surfaces[0].id);
        return true;
      }
      return false; // No surface of that type open — let LLM handle it
    }

    case 'generate-image': {
      const prompt = match[1]?.trim();
      if (!prompt) return false;
      document.dispatchEvent(new CustomEvent('augmentum:generate-image', {
        detail: { prompt },
      }));
      showToast('Generating image...', 'info', 2000);
      return true;
    }

    default:
      return false;
  }
}

// ---------------------------------------------------------------------------
// Command Detection
// ---------------------------------------------------------------------------

/**
 * Check if text matches a voice command. Returns true if handled.
 * Called from the voice module before sending transcript to LLM.
 */
export function tryVoiceCommand(text) {
  if (!text || text.length < 4) return false;

  const trimmed = text.trim();
  for (const cmd of COMMANDS) {
    const match = trimmed.match(cmd.pattern);
    if (match) {
      return _executeCommand(cmd, match);
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

export function initVoiceCommands() {
  // Listen for voice transcripts and check for commands
  document.addEventListener('voice:pre-send', (e) => {
    const text = e.detail?.text;
    if (text && tryVoiceCommand(text)) {
      // Command was handled — prevent sending to LLM
      e.preventDefault();
    }
  });

  // Also expose for keyboard command palette (future)
  window.__voiceCommand = tryVoiceCommand;
}
