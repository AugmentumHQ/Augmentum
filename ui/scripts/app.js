/* ==========================================================================
   Augmentum — Core App Module
   Handles initialization, mode switching, panels, theme, toasts, input
   ========================================================================== */

import { checkStatus, showLogin, showSetupWizard, getCurrentUser, applyRoleBodyClass } from './auth.js';
import { scheduleAutosize } from './utils/textarea-autosize.js';
import { hydrateArmedDevice } from './armed-device.js';
import { initChat, chat } from './chat.js';
import { createImageProgressLoader } from './chat/image-progress.js';
import { initSettings, openSettings, closeModelDropdown, fetchModels as settingsFetchModels, fetchCapabilities, getSettings, save as saveSettings, loadPersonalizationFromServer, loadVoicePrefsFromServer } from './settings.js';
import { initAnalytical } from './analytical.js';
import { initAgentic } from './agentic.js';
import { initFlowBar, showFlowBar, hideFlowBar } from './flow-bar.js';
import { initNarrative, narrative } from './narrative/index.js';
import { initImage, getImageSettings, closeImagePanel } from './image.js';
import { initBrowse, openBrowsePanel, closeBrowsePanel } from './browse.js';
import { initModels } from './models.js';
import { initResources } from './resources.js';
import { initVoice, voice } from './voice.js';
import './presence-fullscreen.js'; // self-inits only on ?presence=1 (lock-screen avatar)
import { warmup as warmupModelCache, getModels, getImageModels, getToolSettings, getCloudImageModels } from './model-cache.js';
import { initFiles, openFiles, closeFiles, toggleFiles } from './files.js';
import * as Coder from './coder.js';
import { initOrbNav, returnToMode, openSurfaceAlongside } from './orb-nav.js';
import { SurfaceRegistry } from './surface-registry.js';
import { LayoutManager } from './layout-manager.js';
import { ViewStack } from './view-stack.js';
import { initFlows } from './surface-flows.js';
import { initVoiceCommands } from './voice-commands.js';
import { CommandComposer } from './command-composer.js';
import { initXrSurfaceBridge } from './xr-surface-bridge.js';
import './surfaces/chat-surface.js';
import './surfaces/narrative-surface.js';
import './surfaces/coder-surface.js';
import './surfaces/browse-surface.js';
import './surfaces/image-surface.js';
import './resize-guard.js';
import { flashToolbarBtn, syncToggleToBackend } from './chat/toolbar/util.js';
import { wireAutoSearch } from './chat/toolbar/auto-search.js';
import { wireAutoRead } from './chat/toolbar/auto-read.js';
import { wireReadingRoom } from './chat/toolbar/reading-room.js';
import { wireNarrativeBubbles } from './chat/toolbar/narrative-bubbles.js';
import { wireAutoBg } from './chat/toolbar/auto-bg.js';
import { wireInstantScene } from './chat/toolbar/instant-scene.js';
import { wireBgRotation } from './chat/toolbar/bg-rotation.js';
import { wireWebSearch } from './chat/toolbar/web-search.js';
import { wireChatTuning } from './chat/toolbar/tuning.js';
import { wireTools } from './chat/toolbar/tools.js';
import { wireOverflow } from './chat/toolbar/overflow.js';
import { initNotifications } from './notifications.js';
import { initConnectUI } from './connect/ui.js';
import { initConnectMessagingUI } from './connect/thread-panel.js';
import { initConnectCallsUI } from './connect/calls-panel.js';
import { initConnectGuestsUI } from './connect/guests-panel.js';
import { initFederation } from './connect/federation.js';
import { ensureConnected as ensureConnectSignaling } from './connect/client.js';

// ---------------------------------------------------------------------------
// Per-tab client id + fetch shim (health/strain monitor)
// ---------------------------------------------------------------------------
// A stable-per-tab id so the server can tell concurrent browsers/tabs/devices
// apart (the strain monitor counts distinct active clients to correlate
// multi-browser contention). sessionStorage = per-tab grain; a new tab/window
// gets its own id. Attached as X-Augmentum-Client on every same-origin request
// via a one-time fetch shim — no per-call wiring needed.
export const CLIENT_ID = (() => {
  try {
    let id = sessionStorage.getItem('augmentum-client-id');
    if (!id) {
      id = (crypto?.randomUUID?.() || `c_${Date.now()}_${Math.floor(Math.random() * 1e9)}`);
      sessionStorage.setItem('augmentum-client-id', id);
    }
    return id;
  } catch {
    return `c_${Date.now()}_${Math.floor(Math.random() * 1e9)}`;
  }
})();

(function _installClientIdFetchShim() {
  if (typeof window === 'undefined' || window.__augClientShim) return;
  const orig = window.fetch;
  if (typeof orig !== 'function') return;
  window.__augClientShim = true;
  window.fetch = function (input, init) {
    try {
      // Only same-origin requests (don't leak the id to external providers).
      const url = typeof input === 'string' ? input : (input && input.url) || '';
      const sameOrigin = url.startsWith('/') || url.startsWith(window.location.origin);
      if (sameOrigin) {
        const headers = new Headers((init && init.headers) || (input && input.headers) || undefined);
        if (!headers.has('X-Augmentum-Client')) headers.set('X-Augmentum-Client', CLIENT_ID);
        init = { ...(init || {}), headers };
      }
    } catch { /* fall through to a normal fetch */ }
    return orig.call(this, input, init);
  };
})();

// ---------------------------------------------------------------------------
// Utility: escape HTML to prevent XSS
// ---------------------------------------------------------------------------
export function escapeHtml(str) {
  if (typeof str !== 'string') str = str == null ? '' : String(str);
  const div = document.createElement('div');
  div.textContent = str;
  // Browser's textContent→innerHTML serialization handles `<`, `>`, `&`.
  // We additionally escape:
  //   ${  → `&#36;{`  — prevents tagged-template-literal injection
  //   '   → `&#39;`   — required for safe interpolation into single-quoted
  //                     HTML *data* attributes (`href='...'`, `title='...'`).
  //                     Without this, an apostrophe in the data closes the
  //                     attribute early and lets an attacker inject more
  //                     attributes (e.g. `onclick=`).
  //   "   → `&#34;`   — same risk in double-quoted data attributes.
  // Backticks are intentionally NOT escaped: renderMarkdown depends on them
  // being literal for code-fence detection (```lang\n...\n```).
  //
  // ⚠ NOT SAFE FOR INLINE EVENT HANDLERS (onclick="..." etc.). Per the
  // HTML5 spec, character references in attribute values are decoded BEFORE
  // the event-handler JS parses the value — so `&#39;` becomes `'` and the
  // injection still works. For data that flows into onclick/onmouse*/onload
  // etc., use the data-attribute + delegated event listener pattern instead
  // of inline handlers (see chat/index.js _onAction handler for a reference).
  return div.innerHTML
    .replace(/\$\{/g, '&#36;{')
    .replace(/'/g, '&#39;')
    .replace(/"/g, '&#34;');
}

// Parse a JSON string safely, returning `fallback` on any failure. The common
// `JSON.parse(localStorage.getItem(k) || '{}')` idiom only guards the MISSING
// case — a corrupt/truncated/tampered value still throws and takes down the
// caller (often at module init). Use this for any parse of untrusted storage.
export function safeParseJSON(raw, fallback = null) {
  if (raw == null || raw === '') return fallback;
  try {
    return JSON.parse(raw);
  } catch (e) {
    console.warn('[safeParseJSON] discarding corrupt value:', e);
    return fallback;
  }
}

// ---------------------------------------------------------------------------
// DOM cache — resolved once at boot
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

let dom = {};

function cacheDom() {
  dom = {
    app:              $('app'),
    // Header
    menuBtn:          $('menu-btn'),
    modeSelector:     $('mode-selector'),
    modelSelector:    $('model-selector'),
    modelName:        $('model-name'),
    thinkingToggle:   $('thinking-toggle'),
    inspectorToggle:  $('inspector-toggle-btn'),
    settingsBtn:      $('settings-btn'),
    themeBtn:         $('theme-btn'),
    themeIcon:         $('theme-icon'),
    // Panels
    panelBackdrop:    $('panel-backdrop'),
    leftPanel:        $('left-panel'),
    inspectorPanel:   $('inspector-panel'),
    closeInspector:   $('close-inspector-btn'),
    // Panel views
    sessionsView:     $('sessions-view'),
    charactersView:   $('characters-view'),
    passthroughView:  $('passthrough-view'),
    reasoningView:    $('reasoning-view'),
    taskView:         $('task-view'),
    cardView:         $('card-view'),
    coderView:        $('coder-view'),
    // Chat
    chatScroll:       $('chat-scroll'),
    chatMessages:     $('chat-messages'),
    emptyState:       $('empty-state'),
    // Input
    chatInput:        $('chat-input'),
    sendBtn:          $('send-btn'),
    commandBtn:       $('command-btn'),
    // (bottom nav removed)
    // Toast
    toastContainer:   $('toast-container'),
    // Inspector section selector (narrative)
    sectionSelect:    $('inspector-section-select'),
    // Library
    libraryOpenBtn:   $('library-open-btn'),
    discoverOpenBtn:  $('discover-open-btn'),
    // Media console (comfort-first rebuild; parallel to Library)
    mediaOpenBtn:     $('media-open-btn'),
    devicesOpenBtn:   $('devices-open-btn'),
    // Workshop (self-improvement surface)
    workshopOpenBtn:  $('workshop-open-btn'),
    // Files
    filesBtn:         $('files-btn'),
  };
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  mode: 'passthrough',        // passthrough | analytical | narrative | agentic
  theme: 'dark',              // dark | light | midnight | sepia
  typography: 'system',       // typography preset key
  textScale: 1.0,             // global text size multiplier (0.7–1.4)
  panelOpen: false,           // mobile left-panel open
  inspectorVisible: false,
  currentModel: 'default',
  currentSessionId: null,
  passthroughTools: [],    // tool names enabled for passthrough mode
};

// isStreaming as a property: every assignment refreshes the send-button UI
// (swap send icon ↔ stop icon) so all 12+ call sites work without changes.
let _isStreamingValue = false;
let _streamingStartedAt = 0;
Object.defineProperty(state, 'isStreaming', {
  enumerable: true,
  get() { return _isStreamingValue; },
  set(v) {
    const next = !!v;
    if (next === _isStreamingValue) return;
    _isStreamingValue = next;
    _streamingStartedAt = next ? performance.now() : 0;
    _refreshSendButton();
  },
});

const SEND_ICON_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
const STOP_ICON_SVG = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';

function _refreshSendButton() {
  const btn = dom?.sendBtn || document.getElementById('send-btn');
  if (!btn) return;
  if (_isStreamingValue) {
    btn.innerHTML = STOP_ICON_SVG;
    btn.title = 'Stop generation';
    btn.classList.add('is-stop');
  } else {
    btn.innerHTML = SEND_ICON_SVG;
    btn.title = 'Send (Enter)';
    btn.classList.remove('is-stop');
  }
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
const THEME_KEY = 'augmentum-theme';

const sunIcon = '<path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10z"/>';
const moonIcon = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
const midnightIcon = '<path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>';
const sepiaIcon = '<path d="M12 2c-2 3-5 5.5-5 8.5a5 5 0 0 0 10 0c0-3-3-5.5-5-8.5z"/><path d="M12 18v4" stroke-linecap="round"/>';

// Theme cycle: dark → light → midnight → sepia (all modes)
const THEMES = ['dark', 'light', 'midnight', 'sepia'];
const THEME_ICONS = { dark: moonIcon, light: sunIcon, midnight: midnightIcon, sepia: sepiaIcon };
const THEME_LABELS = { dark: 'Dark', light: 'Light', midnight: 'Midnight', sepia: 'Sepia' };

// User-facing mode display names (internal enum unchanged for compatibility)
const MODE_DISPLAY = {
  passthrough: 'Chat',
  analytical:  'Analyze',
  narrative:   'Story',
  agentic:     'Build',
  coder:       'Code',
};
function modeLabel(mode) { return MODE_DISPLAY[mode] || mode; }
const THEME_META_COLORS = { dark: '#0f0f1a', light: '#fafaf9', midnight: '#060810', sepia: '#1a1610' };

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  // Migrate old narrative-only theme to unified key
  const oldNarrative = localStorage.getItem('augmentum-narrative-theme');
  if (oldNarrative && THEMES.includes(oldNarrative) && !THEMES.includes(saved)) {
    state.theme = oldNarrative;
    localStorage.removeItem('augmentum-narrative-theme');
  } else if (THEMES.includes(saved)) {
    state.theme = saved;
  } else if (window.matchMedia('(prefers-color-scheme: light)').matches) {
    state.theme = 'light';
  }
  // Clean up old key if still present
  if (oldNarrative) localStorage.removeItem('augmentum-narrative-theme');
  localStorage.setItem(THEME_KEY, state.theme);
  applyTheme();
}

function applyTheme() {
  const theme = state.theme;
  document.documentElement.setAttribute('data-theme', theme);

  // Icon (theme-icon may not exist if Grove panel replaced the theme button)
  if (dom.themeIcon) dom.themeIcon.innerHTML = THEME_ICONS[theme] || moonIcon;

  // Switch highlight.js theme (light is the only light variant)
  const isLight = theme === 'light';
  const hljsLink = document.getElementById('hljs-theme');
  if (hljsLink) {
    hljsLink.href = isLight
      ? 'lib/highlight.js/github.min.css'
      : 'lib/highlight.js/github-dark.min.css';
  }

  // Update meta theme-color
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) {
    meta.content = THEME_META_COLORS[theme] || '#0f0f1a';
  }

  // Push the live theme to the native Android shell (no-op off-Android) so the
  // status bar / nav bar / native surfaces match whatever theme the web UI is
  // on. Fires here = on init and on every toggle, since applyTheme is the one
  // chokepoint that sets data-theme.
  pushThemeToNative();
}

// Relative-luminance check so e.g. dark-sepia (#1a1610) is correctly "dark"
// even though it's a warm paper-ish hue.
function _isDarkColor(css) {
  try {
    let r, g, b;
    const hex = (css || '').match(/^#?([0-9a-f]{6})$/i);
    if (hex) {
      r = parseInt(hex[1].slice(0, 2), 16);
      g = parseInt(hex[1].slice(2, 4), 16);
      b = parseInt(hex[1].slice(4, 6), 16);
    } else {
      const rgb = (css || '').match(/rgba?\(([^)]+)\)/i);
      if (!rgb) return true;
      [r, g, b] = rgb[1].split(',').map((s) => parseFloat(s));
    }
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 < 0.5;
  } catch (_) {
    return true;
  }
}

// Hand the web app's live theme tokens to the native shell. Reads computed
// CSS custom properties (so it always reflects the *current* themes.css, never
// a stale hardcoded copy) and ships them across the AugmentumAndroid bridge.
function pushThemeToNative() {
  const bridge = window.AugmentumAndroid;
  if (!bridge || typeof bridge.setTheme !== 'function') return;
  try {
    const cs = getComputedStyle(document.documentElement);
    const tok = (n) => cs.getPropertyValue(n).trim();
    const bg = tok('--bg');
    bridge.setTheme(JSON.stringify({
      theme: document.documentElement.getAttribute('data-theme') || '',
      bg,
      surface: tok('--surface'),
      accent: tok('--accent'),
      text: tok('--text-primary'),
      textSecondary: tok('--text-secondary'),
      border: tok('--border'),
      isDark: _isDarkColor(bg),
    }));
  } catch (_) {
    // Theme push must never break the page.
  }
}

// Let the native side pull the current theme once its WebView finishes loading
// (covers the cold-start race where the bridge attaches after initTheme ran).
window.__augPushTheme = pushThemeToNative;

// Device corner — Android only. A folded page-corner at the bottom-right that
// opens a tiny menu into the native phone surfaces (Library / Voice Hub /
// Device Settings) via the bridge. Stays hidden in a normal browser.
function initDeviceCorner() {
  const bridge = window.AugmentumAndroid;
  const corner = document.getElementById('device-corner');
  const menu = document.getElementById('device-corner-menu');
  if (!corner || !menu) return;
  if (!bridge || typeof bridge.openNative !== 'function') return;
  corner.hidden = false;
  const close = () => { menu.hidden = true; corner.setAttribute('aria-expanded', 'false'); };
  const open = () => { menu.hidden = false; corner.setAttribute('aria-expanded', 'true'); };
  corner.addEventListener('click', (e) => {
    e.stopPropagation();
    if (menu.hidden) open(); else close();
  });
  menu.querySelectorAll('button[data-surface]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      try { bridge.openNative(btn.dataset.surface); } catch (_) { /* ignore */ }
      close();
    });
  });
  document.addEventListener('click', () => { if (!menu.hidden) close(); });
}
if (document.readyState !== 'loading') initDeviceCorner();
else window.addEventListener('DOMContentLoaded', initDeviceCorner);

// On-device dictation (Android only). Native delivers the Moonshine transcript
// here; insert it into the composer (DON'T auto-send) so the user reviews/edits
// before sending.
window.__augReceiveTranscript = function (text) {
  try {
    const t = (text || '').trim();
    if (!t) return;
    const input = document.getElementById('chat-input');
    if (!input) return;
    const cur = input.value || '';
    input.value = cur && !/\s$/.test(cur) ? cur + ' ' + t : cur + t;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.focus();
  } catch (_) { /* never break on a transcript */ }
};

// On-device dictation (Android) is wired onto the single #mic-btn next to Send
// by chat/stt.js::initMicButton — press-and-hold there runs Moonshine STT. The
// native side delivers the resulting transcript to window.__augReceiveTranscript
// above. No separate dictation button (consolidated 2026-06-18).

function toggleTheme(e) {
  const idx = THEMES.indexOf(state.theme);
  state.theme = THEMES[(idx + 1) % THEMES.length];
  localStorage.setItem(THEME_KEY, state.theme);

  const doSwap = () => {
    applyTheme();
    document.dispatchEvent(new CustomEvent('augmentum:theme-changed', { detail: { theme: state.theme } }));
    showToast(`Theme: ${THEME_LABELS[state.theme]}`, 'success');
  };

  // Instant swap when View Transitions API unavailable or reduced motion preferred
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!document.startViewTransition || prefersReduced) { doSwap(); return; }

  // Circular reveal from the theme/grove button
  const btn = dom.themeBtn || document.getElementById('grove-btn');
  if (!btn) { doSwap(); return; }
  const rect = btn.getBoundingClientRect();
  const x = rect.left + rect.width / 2;
  const y = rect.top + rect.height / 2;
  const radius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));

  const transition = document.startViewTransition(doSwap);
  transition.ready.then(() => {
    document.documentElement.animate(
      { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] },
      { duration: 450, easing: 'ease-in-out', pseudoElement: '::view-transition-new(root)' }
    );
  });
}

// ---------------------------------------------------------------------------
// Typography Presets
// ---------------------------------------------------------------------------
const TYPO_KEY = 'augmentum-typography';
const TYPO_SCALE_KEY = 'augmentum-text-scale';

const TYPO_PRESETS = {
  default: {
    label: 'Augmentum',
    desc: 'Source Sans 3 — bundled, no external fonts',
    // Self-hosted via @font-face in variables.css; needs no CDN load
    // (deliberately absent from _PRESET_FONTS). System stack trails it
    // as the swap fallback before the woff2 paints.
    body: '"Source Sans 3", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif',
    display: null,
    mono: '"JetBrains Mono", "SF Mono", "Cascadia Code", "Fira Code", Consolas, monospace',
    category: 'general',
  },
  system: {
    label: 'System',
    desc: 'Your OS fonts (no bundled face)',
    body: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif',
    display: null,
    mono: '"SF Mono", "Cascadia Code", "Fira Code", Consolas, monospace',
    category: 'general',
  },
  literary: {
    label: 'Literary',
    desc: 'Elegant serif for immersive reading',
    body: '"Literata", Georgia, serif',
    display: '"EB Garamond", "Palatino Linotype", serif',
    mono: '"Fira Code", "Cascadia Code", Consolas, monospace',
    category: 'narrative',
  },
  classic: {
    label: 'Classic',
    desc: 'Traditional novel typography',
    body: '"Lora", "Palatino Linotype", Georgia, serif',
    display: '"Crimson Text", "Book Antiqua", Palatino, serif',
    mono: '"Fira Code", Consolas, monospace',
    category: 'narrative',
  },
  editorial: {
    label: 'Editorial',
    desc: 'Professional magazine clarity',
    body: '"Source Serif 4", "Georgia", serif',
    display: '"Source Sans 3", "Segoe UI", sans-serif',
    mono: '"JetBrains Mono", "Fira Code", monospace',
    category: 'general',
  },
  modern: {
    label: 'Modern',
    desc: 'Clean contemporary sans-serif',
    body: '"DM Sans", "Inter", sans-serif',
    display: '"Inter", "DM Sans", sans-serif',
    mono: '"JetBrains Mono", "Fira Code", monospace',
    category: 'general',
  },
  technical: {
    label: 'Technical',
    desc: 'Optimized for data and analysis',
    body: '"Inter", -apple-system, sans-serif',
    display: '"Inter", sans-serif',
    mono: '"JetBrains Mono", "Cascadia Code", monospace',
    category: 'analytical',
  },
  readable: {
    label: 'Readable',
    desc: 'Maximum legibility (Braille Institute)',
    body: '"Atkinson Hyperlegible", "Verdana", sans-serif',
    display: '"Atkinson Hyperlegible", "Verdana", sans-serif',
    mono: '"JetBrains Mono", Consolas, monospace',
    category: 'general',
  },
  typewriter: {
    label: 'Typewriter',
    desc: 'Monospace everything',
    body: '"JetBrains Mono", "Fira Code", monospace',
    display: '"JetBrains Mono", "Fira Code", monospace',
    mono: '"JetBrains Mono", "Fira Code", monospace',
    category: 'general',
  },
};

// --- Lazy Font Loading ---
// Each preset maps to the Google Fonts families it needs.
// Fonts are injected on-demand when a preset is first activated.
const _PRESET_FONTS = {
  // system: no Google Fonts needed — uses OS fonts
  literary:   ['Literata:ital,opsz,wght@0,7..72,400;0,7..72,500;0,7..72,600;1,7..72,400;1,7..72,500',
               'EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500',
               'Fira+Code:wght@400;500'],
  classic:    ['Lora:ital,wght@0,400;0,500;0,600;1,400;1,500',
               'Crimson+Text:ital,wght@0,400;0,600;1,400',
               'Fira+Code:wght@400;500'],
  editorial:  ['Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400;1,8..60,500',
               'Source+Sans+3:ital,wght@0,400;0,500;0,600;1,400',
               'JetBrains+Mono:wght@400;500'],
  modern:     ['DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;1,9..40,400',
               'Inter:wght@400;500;600',
               'JetBrains+Mono:wght@400;500'],
  technical:  ['Inter:wght@400;500;600',
               'JetBrains+Mono:wght@400;500'],
  readable:   ['Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400',
               'JetBrains+Mono:wght@400;500'],
  typewriter: ['JetBrains+Mono:wght@400;500',
               'Fira+Code:wght@400;500'],
};

// Narrative display fallback (system preset) needs Cormorant Garamond
const _NARRATIVE_FALLBACK_FONTS = [
  'Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500;1,600',
];

const _loadedFontLinks = new Set();  // track injected link IDs

/** Inject Google Fonts <link> for a preset. Dedupes by family name. */
function _ensurePresetFonts(presetKey) {
  const families = _PRESET_FONTS[presetKey];
  if (!families) return;  // system preset — nothing to load
  _injectFontFamilies(families);
}

/** Inject the narrative fallback fonts (Cormorant Garamond for system preset). */
function _ensureNarrativeFallbackFonts() {
  _injectFontFamilies(_NARRATIVE_FALLBACK_FONTS);
}

function _injectFontFamilies(families) {
  for (const spec of families) {
    const familyName = spec.split(':')[0];
    const linkId = 'gfont-preset-' + familyName.toLowerCase();
    if (_loadedFontLinks.has(linkId) || document.getElementById(linkId)) {
      _loadedFontLinks.add(linkId);
      continue;
    }
    const link = document.createElement('link');
    link.id = linkId;
    link.rel = 'stylesheet';
    link.href = `https://fonts.googleapis.com/css2?family=${spec}&display=swap`;
    document.head.appendChild(link);
    _loadedFontLinks.add(linkId);
  }
}

// --- Custom Google Fonts ---

let _typoCustomLoaded = false;  // true after server fetch completes

/** Generate a stable preset key from a font name. */
function _typoFontKey(name) {
  return 'custom-' + name.trim().replace(/\s+/g, '-').toLowerCase();
}

/** Inject a Google Fonts <link> into <head>. No-op if already present. */
function _typoInjectLink(fontName) {
  const id = 'gfont-' + fontName.replace(/\s+/g, '-').toLowerCase();
  if (document.getElementById(id)) return null;  // already loaded
  const link = document.createElement('link');
  link.id = id;
  link.rel = 'stylesheet';
  link.href = 'https://fonts.googleapis.com/css2?family='
    + encodeURIComponent(fontName)
    + ':ital,wght@0,300;0,400;0,500;0,600;0,700;1,400;1,500&display=swap';
  document.head.appendChild(link);
  return link;
}

/** Remove a Google Fonts <link> from <head>. */
function _typoRemoveLink(fontName) {
  const id = 'gfont-' + fontName.replace(/\s+/g, '-').toLowerCase();
  document.getElementById(id)?.remove();
}

/** Register a custom font as a TYPO_PRESETS entry. */
function _typoRegisterCustom(name) {
  const key = _typoFontKey(name);
  TYPO_PRESETS[key] = {
    label: name,
    desc: 'Google Fonts',
    body: `"${name}", sans-serif`,
    display: `"${name}", sans-serif`,
    mono: TYPO_PRESETS.system.mono,
    category: 'custom',
    custom: true,
  };
  return key;
}

/** Persist custom fonts array to the server. */
async function _typoSaveCustomFonts() {
  const fonts = Object.entries(TYPO_PRESETS)
    .filter(([, v]) => v.custom)
    .map(([key, v]) => ({ name: v.label, key }));
  try {
    await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ typography_custom_fonts: JSON.stringify(fonts) }),
    });
  } catch { /* silent — localStorage is still the cache */ }
  localStorage.setItem('augmentum-custom-fonts', JSON.stringify(fonts));
}

/** Load custom fonts from the server, inject links, register presets. */
async function _typoLoadCustomFonts() {
  let fonts = [];
  try {
    const data = await getToolSettings();
    fonts = JSON.parse(data.typography_custom_fonts || '[]');
    // Also pick up server-persisted selection + scale
    const sel = data.typography_selected;
    if (sel) {
      localStorage.setItem(TYPO_KEY, sel);
    }
    const scale = parseFloat(data.typography_text_scale);
    if (scale && scale >= 0.7 && scale <= 1.4) {
      state.textScale = scale;
      localStorage.setItem(TYPO_SCALE_KEY, scale);
    }
  } catch {
    // Fallback to localStorage cache
    try {
      fonts = JSON.parse(localStorage.getItem('augmentum-custom-fonts') || '[]');
    } catch { fonts = []; }
  }

  for (const f of fonts) {
    if (!f.name) continue;
    _typoInjectLink(f.name);
    _typoRegisterCustom(f.name);
  }

  _typoCustomLoaded = true;

  // Re-apply: server selection may differ from localStorage or custom font
  // may now be available as a preset that wasn't registered yet
  const serverSel = localStorage.getItem(TYPO_KEY);
  if (serverSel && TYPO_PRESETS[serverSel]) {
    state.typography = serverSel;
    applyTypography();
  }
  renderTypoDropdown();
}

/** Add a custom Google Font by name. Validates via <link> load event. */
async function addCustomFont(name) {
  name = name.trim().split(',')[0].trim();  // strip fallbacks if pasted
  if (!name) return;

  const key = _typoFontKey(name);
  if (TYPO_PRESETS[key]) {
    showToast(`"${name}" is already added`, 'info');
    return;
  }

  // Inject the stylesheet and wait for it to load or fail
  const link = _typoInjectLink(name);
  if (!link) {
    // Already injected (shouldn't happen since we checked TYPO_PRESETS)
    showToast(`"${name}" is already loaded`, 'info');
    return;
  }

  try {
    await new Promise((resolve, reject) => {
      link.onload = resolve;
      link.onerror = () => reject(new Error('not found'));
      setTimeout(() => reject(new Error('timeout')), 6000);
    });
  } catch {
    link.remove();
    showToast(`Could not load "${name}" from Google Fonts. Check the spelling.`, 'error');
    return;
  }

  // Register, auto-select, save
  _typoRegisterCustom(name);
  setTypography(key);
  await _typoSaveCustomFonts();
  showToast(`Added "${name}"`, 'success');
}

/** Remove a custom font. */
async function removeCustomFont(key) {
  const preset = TYPO_PRESETS[key];
  if (!preset || !preset.custom) return;

  const name = preset.label;
  _typoRemoveLink(name);
  delete TYPO_PRESETS[key];

  // If active, fall back to system
  if (state.typography === key) {
    setTypography('system');
  }

  renderTypoDropdown();
  await _typoSaveCustomFonts();
  showToast(`Removed "${name}"`, 'info');
}

// --- Core typography functions ---

function initTypography() {
  const saved = localStorage.getItem(TYPO_KEY);
  if (saved && TYPO_PRESETS[saved]) {
    state.typography = saved;
  }
  const savedScale = parseFloat(localStorage.getItem(TYPO_SCALE_KEY));
  if (savedScale && savedScale >= 0.7 && savedScale <= 1.4) {
    state.textScale = savedScale;
  }
  applyTypography();
  renderTypoDropdown();

  const btn = document.getElementById('typo-btn');
  const dropdown = document.getElementById('typo-dropdown');
  if (btn && dropdown) {
    btn.addEventListener('click', () => {
      dropdown.classList.toggle('hidden');
      // Auto-focus the input when opening
      if (!dropdown.classList.contains('hidden')) {
        setTimeout(() => dropdown.querySelector('#typo-add-input')?.focus(), 50);
      }
    });
    document.addEventListener('click', (e) => {
      if (!e.target.closest('#typo-menu-wrap')) {
        dropdown.classList.add('hidden');
      }
    });
  }

  // Load custom fonts + server-persisted selection (async, re-renders when done)
  _typoLoadCustomFonts();

  // Listen for typography changes from The Grove (keeps state + fonts in sync)
  document.addEventListener('augmentum:typography-changed', (e) => {
    const key = e.detail?.key;
    if (key && TYPO_PRESETS[key]) {
      setTypography(key);
    }
  });

  // Settings loaded fresh typography config from the backend (custom
  // fonts or preset selection). Re-run the font loader so presets get
  // registered and the active choice applies without a page reload.
  document.addEventListener('augmentum:typography-reload', () => {
    _typoLoadCustomFonts();
  });
}

function renderTypoDropdown() {
  const dropdown = document.getElementById('typo-dropdown');
  if (!dropdown) return;

  const current = state.typography || 'default';

  const categories = [
    { key: 'general', label: 'General' },
    { key: 'narrative', label: 'Narrative' },
    { key: 'analytical', label: 'Analytical' },
    { key: 'custom', label: 'Custom' },
  ];

  let html = '';
  for (const cat of categories) {
    const presets = Object.entries(TYPO_PRESETS).filter(([, v]) => v.category === cat.key);
    // Always show Custom section header (even if empty — the add input lives there)
    if (presets.length === 0 && cat.key !== 'custom') continue;

    html += `<div class="typo-group-label">${escapeHtml(cat.label)}</div>`;

    for (const [key, preset] of presets) {
      const isActive = key === current;
      const removeBtn = preset.custom
        ? `<button class="typo-remove-btn" data-typo-remove="${escapeHtml(key)}" title="Remove">&times;</button>`
        : '';
      html += `
        <button class="typo-option${isActive ? ' active' : ''}" data-typo="${escapeHtml(key)}">
          ${removeBtn}
          <span class="typo-option-label">${escapeHtml(preset.label)}</span>
          <span class="typo-option-desc">${escapeHtml(preset.desc)}</span>
          <span class="typo-option-preview" style="font-family:${preset.body}">Aa</span>
        </button>`;
    }
  }

  // Text size slider
  const scalePercent = Math.round((state.textScale ?? 1) * 100);
  html += `
    <div class="typo-scale-section">
      <div class="typo-scale-row">
        <span class="typo-scale-label">A</span>
        <input type="range" id="typo-scale-slider" class="typo-scale-slider"
               min="0.7" max="1.4" step="0.05" value="${state.textScale ?? 1}">
        <span class="typo-scale-label typo-scale-label-lg">A</span>
        <span class="typo-scale-val" id="typo-scale-val">${scalePercent}%</span>
      </div>
    </div>`;

  // Add-font input + Google Fonts link
  html += `
    <div class="typo-add-section">
      <div class="typo-add-row">
        <input type="text" id="typo-add-input" class="typo-add-input"
               placeholder="Font name..." spellcheck="false" autocomplete="off">
        <button class="typo-add-btn" id="typo-add-btn" title="Add Google Font">+</button>
      </div>
      <a href="https://fonts.google.com" target="_blank" rel="noopener" class="typo-browse-link">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
        Browse Google Fonts
      </a>
    </div>`;

  dropdown.innerHTML = html;

  // Wire preset selection
  dropdown.querySelectorAll('.typo-option').forEach(opt => {
    opt.addEventListener('click', (e) => {
      if (e.target.closest('.typo-remove-btn')) return;
      setTypography(opt.dataset.typo);
      dropdown.classList.add('hidden');
    });
  });

  // Wire remove buttons
  dropdown.querySelectorAll('.typo-remove-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      removeCustomFont(btn.dataset.typoRemove);
    });
  });

  // Wire add input + button
  const addInput = dropdown.querySelector('#typo-add-input');
  const addBtn = dropdown.querySelector('#typo-add-btn');
  if (addBtn && addInput) {
    const doAdd = () => {
      const val = addInput.value.trim();
      if (val) { addCustomFont(val); addInput.value = ''; }
    };
    addBtn.addEventListener('click', doAdd);
    addInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); doAdd(); }
      e.stopPropagation();  // don't trigger global shortcuts while typing
    });
    addInput.addEventListener('click', (e) => e.stopPropagation());
  }

  // Wire text-size slider
  const scaleSlider = dropdown.querySelector('#typo-scale-slider');
  if (scaleSlider) {
    scaleSlider.addEventListener('input', () => setTextScale(parseFloat(scaleSlider.value)));
    scaleSlider.addEventListener('click', (e) => e.stopPropagation());
    // Double-click resets to 100%
    scaleSlider.addEventListener('dblclick', () => setTextScale(1.0));
  }
}

function setTypography(key) {
  if (!TYPO_PRESETS[key]) return;
  state.typography = key;
  localStorage.setItem(TYPO_KEY, key);
  applyTypography();
  renderTypoDropdown();
  showToast(`Typography: ${TYPO_PRESETS[key].label}`, 'success');

  // Persist selection to server (fire-and-forget)
  fetch('/api/config/tools', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ typography_selected: key }),
  }).catch(() => {});
}

function applyTypography() {
  const key = state.typography || 'default';
  const preset = TYPO_PRESETS[key];
  if (!preset) return;

  // Lazy-load only the fonts this preset needs
  if (preset.custom) {
    // Custom fonts are loaded via _typoInjectLink (existing path)
  } else {
    _ensurePresetFonts(key);
  }

  const root = document.documentElement;
  root.style.setProperty('--font-sans', preset.body);
  root.style.setProperty('--font-mono', preset.mono);

  // Narrative display font for action text / headings
  if (preset.display) {
    root.style.setProperty('--font-narrative-display', preset.display);
  } else {
    // Falls back to CSS default (Cormorant Garamond) — ensure it's loaded
    root.style.removeProperty('--font-narrative-display');
    _ensureNarrativeFallbackFonts();
  }

  // Also set body font for narrative mode
  root.style.setProperty('--font-narrative-body', preset.body);

  // Global text size scale — drives all --text-* tokens via CSS calc()
  root.style.setProperty('--text-size-scale', state.textScale ?? 1);
}

/** Update the text size scale (0.7–1.4). Persists to localStorage + server. */
function setTextScale(value) {
  const clamped = Math.round(Math.max(0.7, Math.min(1.4, value)) * 100) / 100;
  state.textScale = clamped;
  localStorage.setItem(TYPO_SCALE_KEY, clamped);
  document.documentElement.style.setProperty('--text-size-scale', clamped);

  // Update the slider label if dropdown is open
  const label = document.getElementById('typo-scale-val');
  if (label) label.textContent = Math.round(clamped * 100) + '%';
  const slider = document.getElementById('typo-scale-slider');
  if (slider && parseFloat(slider.value) !== clamped) slider.value = clamped;

  // Persist to server (fire-and-forget)
  fetch('/api/config/tools', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ typography_text_scale: String(clamped) }),
  }).catch(() => {});
}

// ---------------------------------------------------------------------------
// Mode Switching
// ---------------------------------------------------------------------------
const MODE_KEY = 'augmentum-mode';

function initMode() {
  const saved = localStorage.getItem(MODE_KEY);
  // ViewStack owns mode transitions after this point. Hand it the pure DOM
  // flipper so it can apply modes without re-entering setMode (which would
  // recurse). Boot then does a single applyMode via the injected fn.
  ViewStack._registerApplyMode((mode) => {
    state.mode = mode;
    localStorage.setItem(MODE_KEY, mode);
    applyMode();
  });
  ViewStack.boot({ savedMode: saved });
}

function _findSurfaceContainer(surfaceId) {
  return Array.from(document.querySelectorAll('.surface-container'))
    .find(el => el.dataset.surfaceId === surfaceId) || null;
}

function _getPrimaryInputArea() {
  return document.getElementById('chat-input')?.closest('.input-area') || null;
}

function _rehomePrimaryChatDom() {
  // Primary chat/story still use singleton DOM during the surface migration.
  // Keep those nodes inside the focused primary surface so flex layout cannot
  // leave the composer floating in the middle of the main pane after switches.
  const targetType = state.mode === 'narrative' ? 'narrative' : 'chat';
  const focused = SurfaceRegistry.getFocused?.();
  const surface = (focused?._isPrimary && focused.constructor?.type === targetType)
    ? focused
    : (SurfaceRegistry.all?.() || []).find(s => s._isPrimary && s.constructor?.type === targetType);
  if (!surface) return;

  const container = _findSurfaceContainer(surface.id);
  const target = container?.querySelector('.surface-content');
  if (!target) return;

  const chatScroll = document.getElementById('chat-scroll');
  const inputArea = _getPrimaryInputArea();
  if (chatScroll && chatScroll.parentElement !== target) target.appendChild(chatScroll);
  if (inputArea && inputArea.parentElement !== target) target.appendChild(inputArea);
  chatScroll?.classList.remove('hidden');
  inputArea?.classList.remove('hidden');
}

function _stabilizeAfterViewportResize() {
  if (!dom?.app) return;

  if (state.mode !== 'coder') {
    _rehomePrimaryChatDom();
  }

  // A breakpoint resize can leave all stacked surface containers visually
  // unfocused for a frame. Re-emit the current focus so LayoutManager writes
  // data-focused back onto the active container. If the focused surface has
  // no mounted container (rare race), fall through to the primary fallback
  // for the current mode so we don't end up with no focused container.
  const focused = SurfaceRegistry.getFocused?.();
  if (focused?.id && LayoutManager.hasContainer?.(focused.id)) {
    SurfaceRegistry.focus(focused.id);
    return;
  }

  const targetType = state.mode === 'narrative'
    ? 'narrative'
    : state.mode === 'coder'
      ? 'coder'
      : 'chat';
  const all = SurfaceRegistry.all?.() || [];
  const fallback = all.find(
    s => s._isPrimary && s.constructor?.type === targetType && LayoutManager.hasContainer?.(s.id),
  ) || all.find(s => LayoutManager.hasContainer?.(s.id));
  if (fallback?.id) SurfaceRegistry.focus(fallback.id);
}

window.addEventListener('augmentum:resize-settled', () => {
  requestAnimationFrame(_stabilizeAfterViewportResize);
});

export function applyMode() {
  const { mode } = state;
  dom.app.setAttribute('data-mode', mode);

  const isNarrative = mode === 'narrative';
  const isAnalytical = mode === 'analytical';
  const isAgentic = mode === 'agentic';
  const isCoder = mode === 'coder';

  // Left panel: narrative gets its own panel; other modes use sessions view
  dom.sessionsView.classList.toggle('hidden', isNarrative || isCoder);
  dom.charactersView.classList.toggle('hidden', !isNarrative);

  // Inspector: passthrough vs reasoning vs card vs task vs coder
  const isPassthrough = mode === 'passthrough';
  if (dom.passthroughView) dom.passthroughView.classList.toggle('hidden', !isPassthrough);
  dom.reasoningView.classList.toggle('hidden', !isAnalytical);
  dom.cardView.classList.toggle('hidden', !isNarrative);
  if (dom.taskView) dom.taskView.classList.toggle('hidden', !isAgentic);
  if (dom.coderView) dom.coderView.classList.toggle('hidden', !isCoder);

  // Inspector toggle button picks up the active mode color via CSS so the
  // small mode-dot on the toggle matches whichever mode is rendering.
  if (dom.inspectorToggle) dom.inspectorToggle.dataset.mode = mode;

  // Start / stop the coder inspector poll loop. Lazy import keeps the
  // module out of cold-boot critical path for users who never enter
  // coder mode.
  if (isCoder) {
    import('./coder-inspector.js').then((mod) => {
      const insp = mod.createCoderInspector();
      insp.open();
    }).catch((err) => console.debug('coder-inspector load failed', err));
  } else {
    import('./coder-inspector.js').then((mod) => {
      const insp = mod.getCoderInspector?.();
      insp?.close();
    }).catch(() => {});
  }

  // Coder mode views
  const coderFilesView = document.getElementById('coder-files-view');
  const coderLayout = document.getElementById('coder-layout');
  const coderTerminalPane = document.getElementById('coder-terminal-pane');
  const coderStatus = document.getElementById('coder-status');
  const coderIntent = document.getElementById('coder-intent');
  const coderMobileTabs = document.getElementById('coder-mobile-tabs');
  if (coderFilesView) coderFilesView.classList.toggle('hidden', !isCoder);
  if (coderLayout) coderLayout.classList.toggle('hidden', !isCoder);
  if (coderTerminalPane) coderTerminalPane.classList.toggle('hidden', !isCoder);
  if (coderStatus) coderStatus.classList.toggle('hidden', !isCoder);
  if (coderIntent) coderIntent.classList.toggle('hidden', !isCoder);
  if (coderMobileTabs) coderMobileTabs.classList.toggle('hidden', !isCoder);

  // Hide main chat area in coder mode
  const chatScroll = document.getElementById('chat-scroll');
  const inputArea = _getPrimaryInputArea();
  if (chatScroll) chatScroll.classList.toggle('hidden', isCoder);
  if (inputArea) inputArea.classList.toggle('hidden', isCoder);

  // Reconcile chat/story singleton DOM after mode changes. If the composer
  // is left in .main-area or a stale surface, flex layout can make it float
  // mid-pane instead of sitting at the bottom of the active chat surface.
  if (!isCoder) {
    _rehomePrimaryChatDom();
    if (typeof requestAnimationFrame === 'function') {
      const appliedMode = mode;
      requestAnimationFrame(() => {
        if (state.mode === appliedMode && state.mode !== 'coder') {
          _rehomePrimaryChatDom();
        }
      });
    }
  }

  // Update passthrough dashboard if visible
  if (isPassthrough) updatePassthroughDashboard();

  // Mark active mode option in sidebar
  if (dom.modeSelector) {
    dom.modeSelector.querySelectorAll('.panel-mode-option').forEach(opt => {
      opt.classList.toggle('active', opt.dataset.mode === mode);
    });
  }

  // Scene image button in header (narrative only)
  const sceneGenWrap = document.getElementById('scene-gen-wrap');
  if (sceneGenWrap) {
    sceneGenWrap.classList.toggle('hidden', !isNarrative);
    if (!isNarrative) {
      // Close popover when leaving narrative mode
      const pop = document.getElementById('scene-gen-popover');
      if (pop) pop.classList.add('hidden');
      const btn = document.getElementById('scene-gen-btn');
      if (btn) btn.dataset.state = 'idle';
    }
  }

  // Instant scene gen button (narrative only — input toolbar)
  const instantSceneBtn = document.getElementById('instant-scene-btn');
  if (instantSceneBtn) {
    instantSceneBtn.classList.toggle('hidden', !isNarrative);
  }

  // Quick-insert buttons (narrative only — *action* and "dialogue")
  const quickInsert = document.getElementById('quick-insert');
  if (quickInsert) {
    quickInsert.classList.toggle('hidden', !isNarrative);
  }

  // Auto background button (narrative only). Hide the button outside
  // narrative, but do NOT touch the user's saved preference — the previous
  // _setAutoBgState('off') call here silently flipped narrativeAutoBackground
  // false on every mode change and synced that to the server, so switching
  // modes would kill the feature. We clear the on-screen background (a
  // visual cleanup) but leave the setting alone so re-entering narrative
  // restores the armed state.
  const autoBgBtn = document.getElementById('auto-bg-btn');
  if (autoBgBtn) {
    autoBgBtn.classList.toggle('hidden', !isNarrative);
    const s = getSettings();
    if (!isNarrative) {
      if (window.clearNarrativeBackground) window.clearNarrativeBackground();
      // Reflect the saved preference in the button state, even while hidden,
      // so re-enter is seamless.
      autoBgBtn.dataset.state = s.narrativeAutoBackground ? 'armed' : 'off';
    } else {
      autoBgBtn.dataset.state = s.narrativeAutoBackground ? 'armed' : 'off';
    }
  }

  // Reading room button (narrative only)
  const readingBtn = document.getElementById('reading-room-btn');
  if (readingBtn) {
    readingBtn.classList.toggle('hidden', !isNarrative);
    if (isNarrative) {
      const isRR = document.getElementById('app')?.dataset.readingRoom === 'true';
      readingBtn.dataset.active = isRR ? 'true' : 'false';
    }
  }

  // Chat bubbles button (narrative only)
  const bubblesBtn = document.getElementById('narrative-bubbles-btn');
  if (bubblesBtn) {
    bubblesBtn.classList.toggle('hidden', !isNarrative);
    if (isNarrative) {
      const hasBubbles = document.getElementById('app')?.dataset.narrativeBubbles === 'true';
      bubblesBtn.dataset.active = hasBubbles ? 'true' : 'false';
    }
  }

  // Background rotation button — available in every mode except Coder.
  // The rotation engine itself already no-ops in coder; "Narrative only"
  // vs "All modes" is picked in the button's own config dropdown, so the
  // toggle no longer has to be enabled from Narrative first.
  const bgRotBtn = document.getElementById('bg-rotation-btn');
  if (bgRotBtn) {
    const rotState = window._bgRotationState;
    const showRotation = !isCoder;
    bgRotBtn.classList.toggle('hidden', !showRotation);
    if (showRotation) {
      bgRotBtn.dataset.active = rotState?.enabled ? 'true' : 'false';
      bgRotBtn.dataset.modeColor = mode;
      bgRotBtn.title = rotState?.enabled
        ? `Background rotation (${rotState.interval}s)`
        : 'Background rotation (off)';
    }
    // Hide config dropdown when switching modes
    const bgRotConfig = document.getElementById('bg-rotation-config');
    if (bgRotConfig && !showRotation) bgRotConfig.classList.add('hidden');
  }

  // Auto-search button (Thinker + Creator)
  const searchBtn = document.getElementById('auto-search-btn');
  const showAutoSearch = isAnalytical || isAgentic;
  if (searchBtn) {
    searchBtn.classList.toggle('hidden', !showAutoSearch);
    if (showAutoSearch) searchBtn.dataset.active = getSettings().autoSearch !== false ? 'true' : 'false';
  }

  // Web search button in input toolbar (hide in narrative — uses its own tools)
  const webSearchBtn = document.getElementById('web-search-btn');
  if (webSearchBtn) webSearchBtn.classList.toggle('hidden', isNarrative);

  // Knowledge source bar (chat/analyze/build only)
  const docCtxBar = document.getElementById('doc-context-bar');
  if (docCtxBar) docCtxBar.classList.toggle('hidden', !_modeSupportsDocContext(mode));

  // Tools toggle button (visible in Chat, Thinker, Creator — not Narrative or Coder)
  const toolsWrap = document.getElementById('tools-toggle-wrap');
  const showTools = isPassthrough || isAnalytical || isAgentic;
  if (toolsWrap) {
    toolsWrap.classList.toggle('hidden', !showTools);
    // Update active state based on current selection
    const toolsBtn = document.getElementById('tools-toggle-btn');
    if (toolsBtn) {
      toolsBtn.dataset.active = (state.passthroughTools || []).length > 0 ? 'true' : 'false';
    }
  }

  // Auto-read button (always available if TTS configured, not mode-specific)
  const autoReadBtn = document.getElementById('auto-read-btn');
  if (autoReadBtn) {
    // Show if any TTS is configured (check will be lazy after init)
    const s = getSettings();
    autoReadBtn.dataset.active = s.voiceAutoRead ? 'true' : 'false';
    // Visible in any mode — voice is cross-cutting
    autoReadBtn.classList.remove('hidden');
  }

  // Input placeholder
  if (isNarrative) {
    dom.chatInput.placeholder = 'Write a message... (use *asterisks* for actions)';
  } else if (isAnalytical) {
    dom.chatInput.placeholder = 'Ask a question for deep analysis...';
  } else if (isAgentic) {
    dom.chatInput.placeholder = 'Describe a task... (e.g., "Create a report on...")';
  } else {
    dom.chatInput.placeholder = 'Type a message...';
  }

  // Sync mobile bottom bar mode label
  const mobLabel = document.getElementById('mob-mode-label');
  if (mobLabel) mobLabel.textContent = modeLabel(mode);

}

function setMode(newMode, opts = {}) {
  if (state.mode === newMode) return;
  // Delegate to ViewStack so overlays (voice call etc.) are popped first,
  // SurfaceRegistry is re-focused to match the destination mode, and the
  // legacy applyMode path still runs (ViewStack calls the registered fn).
  ViewStack.setBaseMode(newMode, opts);
  // Notify other modules of mode change (legacy — predates ViewStack)
  document.dispatchEvent(new CustomEvent('augmentum:mode-changed', { detail: { mode: newMode } }));
  // Companion presence: which area of the app the user is in.
  import('./architect-observer.js')
    .then(m => m.reportAttention('surface.attention.mode_changed', { mode: newMode }))
    .catch(() => {});
  // "Catch up on this chat" handoff — the chat transcript lives in
  // #chat-messages. Registered on entering chat, cleared on leaving.
  // (Coder registers its own 'file' provider at file-open; browse its
  // 'page' provider at page-open — those aren't modes.)
  import('./companion-context.js')
    .then(m => {
      // Chat and narrative both render their active conversation into the
      // shared #chat-messages shell, so the same DOM provider serves both
      // — narrative just carries the 'scene' kind so deixis lines up with
      // its perception slot. Empty container → setLoadableFromDom no-ops
      // and the chip stays hidden (graceful).
      if (newMode === 'chat') m.setLoadableFromDom('chat', 'this chat', '#chat-messages');
      else m.clearCompanionLoadable('chat');
      if (newMode === 'narrative') m.setLoadableFromDom('scene', 'this scene', '#chat-messages');
      else m.clearCompanionLoadable('scene');
      // Coder registers its own 'file' provider at file-open; clear it
      // here when the user leaves the coder surface entirely.
      if (newMode !== 'coder') m.clearCompanionLoadable('file');
    })
    .catch(() => {});
  // Fire-and-forget passive hint to the companion dispatcher (Lane 3 §X.x).
  // The legacy mode toggle continues to drive routing directly; this lets
  // the runtime see the user's preference as a bus event for telemetry +
  // future decisions. 503 if companion_runtime_enabled is off — ignored.
  try {
    fetch('/api/companion/mode_hint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ mode: newMode }),
    }).catch(() => {});
  } catch (_) {}
}

// ---------------------------------------------------------------------------
// Left Panel (mobile drawer + desktop toggle)
// ---------------------------------------------------------------------------
const PANEL_KEY = 'augmentum-panel';

function isDesktop() {
  return window.matchMedia('(min-width: 768px)').matches;
}

function openPanel() {
  state.panelOpen = true;
  if (isDesktop()) {
    dom.leftPanel.classList.remove('desktop-collapsed');
    dom.app.setAttribute('data-panel', 'visible');
    localStorage.setItem(PANEL_KEY, 'visible');
  } else {
    dismissOverlays();
    closeInspectorMobile();
    dom.leftPanel.classList.add('open');
    dom.panelBackdrop.classList.add('visible');
    document.body.style.overflow = 'hidden';
  }
}

function closePanel() {
  state.panelOpen = false;
  if (isDesktop()) {
    dom.leftPanel.classList.add('desktop-collapsed');
    dom.app.setAttribute('data-panel', 'hidden');
    localStorage.setItem(PANEL_KEY, 'hidden');
  } else {
    dom.leftPanel.classList.remove('open');
    dom.panelBackdrop.classList.remove('visible');
    document.body.style.overflow = '';
  }
}

function togglePanel() {
  if (isDesktop()) {
    const collapsed = dom.leftPanel.classList.contains('desktop-collapsed');
    collapsed ? openPanel() : closePanel();
  } else {
    state.panelOpen ? closePanel() : openPanel();
  }
}

function restorePanelState() {
  const saved = localStorage.getItem(PANEL_KEY);
  if (saved === 'hidden' && isDesktop()) {
    dom.leftPanel.classList.add('desktop-collapsed');
    dom.app.setAttribute('data-panel', 'hidden');
    state.panelOpen = false;
  }
}

// ---------------------------------------------------------------------------
// Mobile Drawer Swipe Gestures
// ---------------------------------------------------------------------------
// Swipe left on the open drawer → close it (with real-time tracking).
// Swipe from the left edge of the screen → open it.

// Swipe-to-close for left/right panels removed.
// The gesture handlers intercepted touch events on buttons and interactive
// elements inside panels, causing taps to be swallowed on mobile.
// Panels are closed by tapping the backdrop or pressing Escape instead.

// ---------------------------------------------------------------------------
// Inspector Panel
// ---------------------------------------------------------------------------
const INSPECTOR_KEY = 'augmentum-inspector';

function initInspector() {
  // Restore saved state on desktop; always start closed on mobile
  const saved = localStorage.getItem(INSPECTOR_KEY);
  if (window.innerWidth >= 1024 && saved === 'visible') {
    state.inspectorVisible = true;
  } else {
    state.inspectorVisible = false;
  }
  applyInspector();
}

function applyInspector() {
  dom.app.setAttribute('data-inspector', state.inspectorVisible ? 'visible' : 'hidden');
  dom.inspectorPanel.classList.toggle('mobile-open', false);
}

function toggleInspector() {
  // Below desktop breakpoint, use mobile overlay instead of grid toggle
  if (window.innerWidth < 1024) {
    if (dom.inspectorPanel.classList.contains('mobile-open')) {
      closeInspectorMobile();
    } else {
      // Mobile: inspector is full-width, close other mobile drawers
      closeImagePanel();
      closePanel();
      openInspectorMobile();
    }
    return;
  }
  const opening = !state.inspectorVisible;
  state.inspectorVisible = opening;
  localStorage.setItem(INSPECTOR_KEY, opening ? 'visible' : 'hidden');
  // Desktop: inspector is a grid column, only close competing side panel
  if (opening) { closeImagePanel(); }
  applyInspector();

  // When opening the inspector, trigger a fresh narrative state poll
  // so the panel always shows current data (not stale from before it was hidden)
  if (opening && state.mode === 'narrative') {
    document.dispatchEvent(new CustomEvent('augmentum:inspector-opened'));
  }
  // Same pattern for coder: kick a refresh so the panel doesn't show
  // stale-from-before-hidden state when re-opened.
  if (opening && state.mode === 'coder') {
    import('./coder-inspector.js').then((mod) => {
      const insp = mod.getCoderInspector?.() || mod.createCoderInspector();
      insp?.open();
    }).catch(() => {});
  } else if (!opening && state.mode === 'coder') {
    // Closing in coder mode: stop the polling loop to save resources.
    import('./coder-inspector.js').then((mod) => {
      mod.getCoderInspector?.()?.close();
    }).catch(() => {});
  }
}

/** Close the inspector without toggling (for use by other panel openers). */
function closeInspector() {
  if (window.innerWidth < 1024) {
    if (dom.inspectorPanel.classList.contains('mobile-open')) closeInspectorMobile();
  } else if (state.inspectorVisible) {
    state.inspectorVisible = false;
    localStorage.setItem(INSPECTOR_KEY, 'hidden');
    applyInspector();
  }
}

// Mobile: open inspector as overlay
function openInspectorMobile() {
  dom.inspectorPanel.classList.add('mobile-open');
  dom.panelBackdrop.classList.add('visible');
  document.body.style.overflow = 'hidden';
}

function closeInspectorMobile() {
  dom.inspectorPanel.classList.remove('mobile-open');
  dom.panelBackdrop.classList.remove('visible');
  document.body.style.overflow = '';
}

// ---------------------------------------------------------------------------
// Inspector Resize (drag-to-resize on desktop)
// ---------------------------------------------------------------------------
const INSPECTOR_WIDTH_KEY = 'augmentum-inspector-width';
const INSPECTOR_MIN_W = 280;
const INSPECTOR_MAX_VW = 0.5; // 50vw

function initInspectorResize() {
  const handle = document.getElementById('inspector-resize-handle');
  if (!handle) return;

  let dragging = false;
  let startX = 0;
  let startW = 0;

  function maxPx() { return window.innerWidth * INSPECTOR_MAX_VW; }

  function onPointerDown(e) {
    if (window.innerWidth < 1024) return; // no resize on mobile
    e.preventDefault();
    dragging = true;
    startX = e.clientX;
    startW = dom.inspectorPanel.offsetWidth;
    handle.classList.add('dragging');
    handle.setPointerCapture(e.pointerId);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }

  function onPointerMove(e) {
    if (!dragging) return;
    // Dragging left = larger panel (panel is on the right)
    const delta = startX - e.clientX;
    const w = Math.max(INSPECTOR_MIN_W, Math.min(maxPx(), startW + delta));
    document.documentElement.style.setProperty('--inspector-width', w + 'px');
  }

  function onPointerUp(e) {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    handle.releasePointerCapture(e.pointerId);
    // Persist
    const w = dom.inspectorPanel.offsetWidth;
    localStorage.setItem(INSPECTOR_WIDTH_KEY, String(w));
  }

  handle.addEventListener('pointerdown', onPointerDown);
  handle.addEventListener('pointermove', onPointerMove);
  handle.addEventListener('pointerup', onPointerUp);
  handle.addEventListener('pointercancel', onPointerUp);

  // Restore persisted width
  const saved = localStorage.getItem(INSPECTOR_WIDTH_KEY);
  if (saved) {
    const w = Math.max(INSPECTOR_MIN_W, Math.min(maxPx(), parseInt(saved, 10)));
    if (w) document.documentElement.style.setProperty('--inspector-width', w + 'px');
  }
}

// ---------------------------------------------------------------------------
// Left Panel Resize (drag-to-resize on tablet/desktop)
// ---------------------------------------------------------------------------
// Mirror of the inspector resize, on the LEFT panel's right edge. The
// panel drives --panel-width, which is the left grid column; widening it
// lets the file-search pane show full match traces instead of clipping.
// The `1fr` main column absorbs the delta, so the layout reflows smoothly.
const PANEL_WIDTH_KEY = 'augmentum-panel-width';
const PANEL_MIN_W = 240;
const PANEL_MAX_VW = 0.6; // 60vw — generous so long search lines fit

function initLeftPanelResize() {
  const handle = document.getElementById('left-panel-resize-handle');
  if (!handle || !dom.leftPanel) return;

  let dragging = false;
  let startX = 0;
  let startW = 0;

  function maxPx() { return window.innerWidth * PANEL_MAX_VW; }

  function onPointerDown(e) {
    if (window.innerWidth < 768) return; // panel is a swipe drawer below this
    e.preventDefault();
    dragging = true;
    startX = e.clientX;
    startW = dom.leftPanel.offsetWidth;
    handle.classList.add('dragging');
    handle.setPointerCapture(e.pointerId);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }

  function onPointerMove(e) {
    if (!dragging) return;
    // Dragging right = larger panel (panel is on the left).
    const delta = e.clientX - startX;
    const w = Math.max(PANEL_MIN_W, Math.min(maxPx(), startW + delta));
    document.documentElement.style.setProperty('--panel-width', w + 'px');
  }

  function onPointerUp(e) {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    handle.releasePointerCapture(e.pointerId);
    localStorage.setItem(PANEL_WIDTH_KEY, String(dom.leftPanel.offsetWidth));
  }

  handle.addEventListener('pointerdown', onPointerDown);
  handle.addEventListener('pointermove', onPointerMove);
  handle.addEventListener('pointerup', onPointerUp);
  handle.addEventListener('pointercancel', onPointerUp);

  // Double-click the grip resets to the default width — an easy escape
  // from an over-wide panel without hunting for the exact original size.
  handle.addEventListener('dblclick', () => {
    document.documentElement.style.removeProperty('--panel-width');
    localStorage.removeItem(PANEL_WIDTH_KEY);
  });

  // Restore persisted width.
  const saved = localStorage.getItem(PANEL_WIDTH_KEY);
  if (saved) {
    const w = Math.max(PANEL_MIN_W, Math.min(maxPx(), parseInt(saved, 10)));
    if (w) document.documentElement.style.setProperty('--panel-width', w + 'px');
  }
}

// ---------------------------------------------------------------------------
// Inspector Section Selector
// ---------------------------------------------------------------------------
const INSPECTOR_SECTION_KEY = 'augmentum-inspector-section';

function switchInspectorSection(tabId) {
  if (!dom.cardView) return;
  dom.cardView.querySelectorAll('.tab-content').forEach(tc => {
    tc.classList.toggle('hidden', tc.id !== tabId);
  });
  if (dom.sectionSelect) dom.sectionSelect.value = tabId;
  localStorage.setItem(INSPECTOR_SECTION_KEY, tabId);
}

function initTabs() {
  if (!dom.sectionSelect) return;

  // Restore last chosen section (or default to first option)
  const saved = localStorage.getItem(INSPECTOR_SECTION_KEY);
  const validIds = Array.from(dom.sectionSelect.options).map(o => o.value);
  const initial = (saved && validIds.includes(saved)) ? saved : validIds[0];
  switchInspectorSection(initial);

  dom.sectionSelect.addEventListener('change', (e) => {
    switchInspectorSection(e.target.value);
    // Trigger LTM render when tab is selected (it may not have rendered yet)
    if (e.target.value === 'chat-settings-tab') {
      if (typeof window._nsRenderLtm === 'function') window._nsRenderLtm();
    }
  });
}

// (bottom nav removed)

// ---------------------------------------------------------------------------
// Textarea Auto-Resize
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Slash Command Autocomplete
// ---------------------------------------------------------------------------
let _slashMenuEl = null;
let _slashActiveIdx = -1;
let _slashFlowCache = null;
let _slashFlowCacheTime = 0;

const _SLASH_COMMANDS = [
  { cmd: '/v', desc: 'Generate an image' },
  { cmd: '/flow', desc: 'Run a saved flow' },
];

async function _fetchFlowNames() {
  if (_slashFlowCache && Date.now() - _slashFlowCacheTime < 30_000) return _slashFlowCache;
  try {
    const resp = await fetch('/api/flows');
    if (resp.ok) {
      const flows = await resp.json();
      _slashFlowCache = flows.filter(f => f.enabled).map(f => f.name);
      _slashFlowCacheTime = Date.now();
      return _slashFlowCache;
    }
  } catch { /* ignore */ }
  return [];
}

function _buildSlashItems(query) {
  const items = [];
  const q = query.toLowerCase();

  if (q.startsWith('/flow ')) {
    // Show flow name completions
    const partial = query.slice(6).toLowerCase();
    _fetchFlowNames().then(names => {
      const filtered = partial ? names.filter(n => n.toLowerCase().includes(partial)) : names;
      _renderSlashMenu(filtered.map(n => ({ cmd: `/flow ${n}`, desc: 'Run flow' })));
    });
    return null; // async render
  }

  for (const c of _SLASH_COMMANDS) {
    if (c.cmd.startsWith(q) || q.startsWith(c.cmd)) {
      items.push(c);
    }
  }
  return items;
}

function _renderSlashMenu(items) {
  if (!items || items.length === 0) { _hideSlashMenu(); return; }
  if (!_slashMenuEl) {
    _slashMenuEl = document.createElement('div');
    _slashMenuEl.className = 'slash-menu';
    dom.chatInput.parentElement.style.position = 'relative';
    dom.chatInput.parentElement.appendChild(_slashMenuEl);
  }
  _slashActiveIdx = 0;
  _slashMenuEl.innerHTML = items.map((item, i) =>
    `<div class="slash-menu-item${i === 0 ? ' active' : ''}" data-idx="${i}" data-cmd="${escapeHtml(item.cmd)}">
      <span class="slash-cmd">${escapeHtml(item.cmd)}</span>
      <span class="slash-desc">${escapeHtml(item.desc)}</span>
    </div>`
  ).join('');
  _slashMenuEl.classList.remove('hidden');
  _slashMenuEl.querySelectorAll('.slash-menu-item').forEach(el => {
    el.addEventListener('click', () => _selectSlashItem(el.dataset.cmd));
    el.addEventListener('mouseenter', () => {
      _slashMenuEl.querySelectorAll('.slash-menu-item').forEach(e => e.classList.remove('active'));
      el.classList.add('active');
      _slashActiveIdx = parseInt(el.dataset.idx, 10);
    });
  });
}

function _showSlashMenu(query) {
  const items = _buildSlashItems(query);
  if (items !== null) _renderSlashMenu(items);
}

function _hideSlashMenu() {
  if (_slashMenuEl) { _slashMenuEl.classList.add('hidden'); }
  _slashActiveIdx = -1;
}

function _navigateSlashMenu(dir) {
  if (!_slashMenuEl) return;
  const items = _slashMenuEl.querySelectorAll('.slash-menu-item');
  if (items.length === 0) return;
  items[_slashActiveIdx]?.classList.remove('active');
  _slashActiveIdx = (_slashActiveIdx + dir + items.length) % items.length;
  items[_slashActiveIdx]?.classList.add('active');
  items[_slashActiveIdx]?.scrollIntoView({ block: 'nearest' });
}

function _selectSlashItem(cmd) {
  if (!cmd && _slashMenuEl) {
    const active = _slashMenuEl.querySelector('.slash-menu-item.active');
    cmd = active?.dataset.cmd;
  }
  if (cmd) {
    dom.chatInput.value = cmd + (cmd.endsWith(' ') ? '' : ' ');
    autoResize(dom.chatInput);
    dom.chatInput.focus();
  }
  _hideSlashMenu();
}

function _isSlashMenuVisible() {
  return _slashMenuEl && !_slashMenuEl.classList.contains('hidden');
}

function initInput() {
  const input = dom.chatInput;

  input.addEventListener('input', () => {
    autoResize(input);
    // Slash command autocomplete
    const val = input.value;
    if (val.startsWith('/') && val.length < 30) {
      _showSlashMenu(val);
    } else {
      _hideSlashMenu();
    }
  });

  // Send on Enter (without Shift)
  input.addEventListener('keydown', (e) => {
    if (_isSlashMenuVisible()) {
      if (e.key === 'ArrowUp') { e.preventDefault(); _navigateSlashMenu(-1); return; }
      if (e.key === 'ArrowDown') { e.preventDefault(); _navigateSlashMenu(1); return; }
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); _selectSlashItem(); return; }
      if (e.key === 'Escape') { e.preventDefault(); _hideSlashMenu(); return; }
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  dom.sendBtn.addEventListener('click', handleSend);
}

function autoResize(textarea) {
  // Delegate to the shared rAF-deferred helper. The inline pattern
  // (height='auto' + read scrollHeight + write height) forces sync
  // layout on every keystroke. The 200px cap differs from the
  // helper's default 50vh — pass it explicitly.
  scheduleAutosize(textarea, 200);
}

function handleSend() {
  // Mid-generation: send button doubles as stop. Click/Enter aborts the
  // active stream instead of being a silent no-op.
  if (state.isStreaming) {
    if (performance.now() - _streamingStartedAt < 1800) {
      showToast('Message is starting...', 'info', 1200);
      return;
    }
    document.dispatchEvent(new CustomEvent('augmentum:stop'));
    return;
  }
  const text = dom.chatInput.value.trim();
  if (!text && pendingImages.length === 0 && pendingDocuments.length === 0) return;

  // Dispatch custom event for chat module to handle
  const images = pendingImages.length > 0 ? [...pendingImages] : undefined;
  const docs = pendingDocuments.length > 0 ? [...pendingDocuments] : undefined;
  const event = new CustomEvent('augmentum:send', { detail: { text, images, docs } });
  document.dispatchEvent(event);

  // Clear input and attachments
  dom.chatInput.value = '';
  dom.chatInput.placeholder = 'Type a message...';
  clearPendingImages();
  clearPendingDocuments();
  autoResize(dom.chatInput);
  // Focus input on desktop only — on mobile this re-opens the keyboard
  if (window.innerWidth >= 768) dom.chatInput.focus();
}

// ---------------------------------------------------------------------------
// Quick Insert — *action* and "dialogue" buttons for narrative mode
// ---------------------------------------------------------------------------

function _initQuickInsert() {
  const container = document.getElementById('quick-insert');
  const input = dom.chatInput;
  if (!container || !input) return;

  container.querySelectorAll('.quick-insert-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const wrap = btn.dataset.wrap; // '"' or '*'
      const start = input.selectionStart;
      const end = input.selectionEnd;
      const text = input.value;

      if (start !== end) {
        // Wrap selected text
        const selected = text.slice(start, end);
        const wrapped = wrap + selected + wrap;
        input.value = text.slice(0, start) + wrapped + text.slice(end);
        input.selectionStart = start + wrap.length;
        input.selectionEnd = end + wrap.length;
      } else {
        // Insert pair and place cursor between them
        input.value = text.slice(0, start) + wrap + wrap + text.slice(start);
        input.selectionStart = start + wrap.length;
        input.selectionEnd = start + wrap.length;
      }

      input.focus();
      input.dispatchEvent(new Event('input')); // trigger auto-resize
    });
  });
}

// ---------------------------------------------------------------------------
// File Attachment (images + documents)
// ---------------------------------------------------------------------------
let pendingImages = [];
let pendingDocuments = []; // { id, filename, chunk_count }

function initAttach() {
  const btn = document.getElementById('attach-btn');
  const input = document.getElementById('attach-input');
  if (!btn || !input) return;

  btn.addEventListener('click', () => input.click());

  input.addEventListener('change', () => {
    if (input.files) addFiles(input.files);
    input.value = '';
  });

  // Paste from clipboard: images become attachments, bare URLs auto-fetch
  // and attach the page (Bridge 4 — chat ↔ browse). Mixed/text paste with
  // surrounding prose falls through to default text paste behavior.
  dom.chatInput.addEventListener('paste', (e) => {
    const files = [];
    for (const item of e.clipboardData?.items || []) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }
    if (files.length > 0) {
      e.preventDefault();
      addFiles(files);
      return;
    }
    const text = (e.clipboardData?.getData('text') || '').trim();
    if (text && text.length < 500 && /^https?:\/\/\S+$/i.test(text) && typeof window._attachWebPage === 'function') {
      e.preventDefault();
      window._attachWebPage(text, '');
    }
  });

  // Drag and drop (any file type)
  const inputArea = _getPrimaryInputArea();
  if (inputArea) {
    inputArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      inputArea.classList.add('drag-over');
    });
    inputArea.addEventListener('dragleave', () => {
      inputArea.classList.remove('drag-over');
    });
    inputArea.addEventListener('drop', (e) => {
      e.preventDefault();
      inputArea.classList.remove('drag-over');
      if (e.dataTransfer?.files) addFiles(e.dataTransfer.files);
    });
  }
}

/** Route files by type: images → base64 inline, documents → upload + bind. */
function addFiles(files) {
  const imageFiles = [];
  const docFiles = [];
  for (const file of files) {
    if (file.type.startsWith('image/')) {
      imageFiles.push(file);
    } else {
      docFiles.push(file);
    }
  }
  if (imageFiles.length > 0) addImageFiles(imageFiles);
  for (const doc of docFiles) uploadAndAttachDocument(doc);
}

/**
 * Attach an artifact to the chat input as if the user had dropped it there.
 *
 * Fetches the binary, wraps it in a File, and routes through `addFiles` so
 * image artifacts become inline vision attachments and everything else gets
 * ingested into the documents store (chunked, session-bound, searchable).
 *
 * A new session is created first so the artifact lands in a fresh context —
 * "Ask AI about this file" feels wrong if the artifact gets dumped into
 * whatever unrelated chat the user had on screen.
 */
// Formats whose raw download isn't ingestible by the document store, but whose
// TEXT we can derive server-side: audio/video → STT transcript (/transcribe),
// epub → chapter text (/epub-text). These route through a derive-then-attach
// path instead of the binary download. Mirrors _AUDIO_VIDEO_EXTS in
// augmentum/proxy/artifact_routes.py.
const _ARTIFACT_TRANSCRIBE_FORMATS = new Set([
  'mp3', 'wav', 'm4a', 'flac', 'ogg', 'opus', 'aac',
  'mp4', 'mov', 'webm', 'mkv', 'avi',
]);

/** Strip a trailing extension so the derived text file gets a clean stem. */
function _artifactStem(name) {
  const base = name || 'artifact';
  const dot = base.lastIndexOf('.');
  return dot > 0 ? base.slice(0, dot) : base;
}

/**
 * Attach already-extracted plain text to the chat as a synthetic .txt document,
 * routed through the normal upload pipeline (session bind, pendingDocuments,
 * context bar). Used for derived-text formats (transcripts, ebook text).
 */
function _attachDerivedTextDocument(text, fileName) {
  const file = new File([text], fileName, { type: 'text/plain' });
  // Fresh session so the derived doc lands in a clean context (matches the
  // binary attach flow); event dispatch is synchronous.
  document.dispatchEvent(new CustomEvent('augmentum:new-session'));
  addFiles([file]);
  dom.chatInput?.focus();
}

/** Audio/video → transcribe via STT, then attach the transcript as text. */
async function _attachTranscribedArtifact(artifactId, displayName, toastId) {
  updateToast(toastId, `Transcribing ${displayName}…`, 'loading');
  const resp = await fetch(
    `/api/artifacts/${encodeURIComponent(artifactId)}/transcribe`,
    { method: 'POST' },
  );
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || err.error || `Transcription failed (${resp.status})`);
  }
  const data = await resp.json();
  const transcript = (data.transcript || '').trim();
  if (!transcript) {
    updateToast(toastId, `No speech detected in ${displayName}`, 'warning');
    return;
  }
  _attachDerivedTextDocument(transcript, `${_artifactStem(displayName)}.transcript.txt`);
  updateToast(toastId, `Attached transcript of ${displayName}`, 'success');
}

/** eBook → derive chapter text via /epub-text, then attach as a text doc. */
async function _attachEbookText(artifactId, displayName, toastId) {
  updateToast(toastId, `Extracting ${displayName}…`, 'loading');
  const resp = await fetch(`/api/artifacts/${encodeURIComponent(artifactId)}/epub-text`);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || err.error || `Extraction failed (${resp.status})`);
  }
  const data = await resp.json();
  const chapters = Array.isArray(data.chapters) ? data.chapters : [];
  const text = chapters
    .map((c) => {
      const heading = (c.heading || '').trim();
      const body = (c.text || '').trim();
      return heading ? `# ${heading}\n\n${body}` : body;
    })
    .filter(Boolean)
    .join('\n\n')
    .trim();
  if (!text) {
    updateToast(toastId, `No text found in ${displayName}`, 'warning');
    return;
  }
  _attachDerivedTextDocument(text, `${_artifactStem(displayName)}.txt`);
  updateToast(toastId, `Attached text of ${displayName}`, 'success');
}

async function attachArtifactToChat(artifactId, filename, format = '') {
  if (!artifactId) return;
  const displayName = filename || 'artifact';
  const fmt = String(format || '').toLowerCase();
  const toastId = showToast(`Attaching ${displayName}…`, 'loading');
  try {
    // Derived-text formats: the raw binary isn't ingestible, so extract text
    // server-side and attach that instead of the download blob.
    if (_ARTIFACT_TRANSCRIBE_FORMATS.has(fmt)) {
      await _attachTranscribedArtifact(artifactId, displayName, toastId);
      return;
    }
    if (fmt === 'epub') {
      await _attachEbookText(artifactId, displayName, toastId);
      return;
    }

    const resp = await fetch(`/api/artifacts/${encodeURIComponent(artifactId)}/download`);
    if (!resp.ok) throw new Error(`Download failed (${resp.status})`);
    const blob = await resp.blob();
    const file = new File(
      [blob],
      displayName,
      { type: blob.type || 'application/octet-stream' },
    );

    // Start a fresh chat session so the artifact attaches to a clean
    // context — event dispatch is synchronous (see chat/index.js handler).
    document.dispatchEvent(new CustomEvent('augmentum:new-session'));

    // Route through the same path as drag-drop — this handles image-vs-doc
    // routing, session binding, pendingDocuments, context bar refresh.
    addFiles([file]);

    updateToast(toastId, `Attached ${displayName}`, 'success');
    // Focus the input so the user can type their question immediately.
    dom.chatInput?.focus();
  } catch (err) {
    updateToast(toastId, `Attach failed: ${err.message}`, 'error');
  }
}

// Studio's "Ask AI" button on each viewer dispatches this — listener is
// module-level so it's available even when Studio hasn't been opened yet.
document.addEventListener('augmentum:ask-ai-about-artifact', (e) => {
  const { artifactId, filename, format } = e.detail || {};
  attachArtifactToChat(artifactId, filename, format);
});

/** Max pixel dimension (longest edge) for images sent to the LLM.
 *  2048 preserves fine text in screenshots while matching OpenAI's highest
 *  detail tier.  Images already within this bound pass through untouched. */
const IMAGE_MAX_EDGE = 2048;

/** JPEG quality for photo-type images (0-1).  0.92 is visually lossless
 *  while typically reducing a phone photo from ~4MB to ~200-400KB. */
const IMAGE_JPEG_QUALITY = 0.92;

/** Types that may carry an alpha channel (transparency). */
const _ALPHA_TYPES = new Set([
  'image/png', 'image/webp', 'image/gif', 'image/avif',
  'image/svg+xml', 'image/x-icon', 'image/vnd.microsoft.icon',
]);

/** Types that createImageBitmap can't reliably decode in all browsers. */
const _NEEDS_IMG_DECODE = new Set([
  'image/heic', 'image/heif', 'image/svg+xml', 'image/tiff',
]);

function addImageFiles(files) {
  for (const file of files) {
    if (!file.type.startsWith('image/')) continue;
    _processImageFile(file);
  }
}

async function _processImageFile(file) {
  let dataUrl;
  try {
    const bitmap = await _decodeToBitmap(file);
    const { width, height } = bitmap;
    const needsResize = width > IMAGE_MAX_EDGE || height > IMAGE_MAX_EDGE;

    if (!needsResize) {
      // Already within bounds — use original if LLM-friendly, otherwise re-encode
      if (_isLlmFriendlyType(file.type)) {
        dataUrl = await _readAsDataUrl(file);
        bitmap.close();
      } else {
        // Exotic format (HEIC, TIFF, BMP, AVIF, etc.) — re-encode to PNG/JPEG
        dataUrl = _bitmapToDataUrl(bitmap, file.type, width, height);
      }
    } else {
      // Scale down preserving aspect ratio
      const scale = IMAGE_MAX_EDGE / Math.max(width, height);
      const newW = Math.round(width * scale);
      const newH = Math.round(height * scale);
      dataUrl = _bitmapToDataUrl(bitmap, file.type, newW, newH);
    }
  } catch {
    // Last resort: send raw bytes (may be an exotic format the LLM can't read)
    dataUrl = await _readAsDataUrl(file);
  }

  // Upload to server for persistent storage, fall back to inline base64
  const url = await _uploadChatImage(dataUrl);
  pendingImages.push(url || dataUrl);
  renderAttachPreviews();
}

/** Decode a file to an ImageBitmap, falling back to <img> for exotic types. */
async function _decodeToBitmap(file) {
  if (!_NEEDS_IMG_DECODE.has(file.type)) {
    return createImageBitmap(file);
  }
  // For HEIC/SVG/TIFF: load via <img> element which has broader codec support
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      // SVG needs explicit dimensions to rasterize
      if (file.type === 'image/svg+xml') {
        img.width = IMAGE_MAX_EDGE;
        img.height = IMAGE_MAX_EDGE;
      }
      img.src = url;
    });
    return createImageBitmap(img);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** Draw a bitmap to canvas and export as a data URL with appropriate format. */
function _bitmapToDataUrl(bitmap, originalType, targetW, targetH) {
  const canvas = document.createElement('canvas');
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext('2d');

  // If the format might have transparency, check if any pixel actually uses it
  const hasAlpha = _ALPHA_TYPES.has(originalType) && _detectAlpha(bitmap, ctx, targetW, targetH);

  if (hasAlpha) {
    // Transparent → PNG (preserves alpha, lossless edges)
    ctx.clearRect(0, 0, targetW, targetH);
    ctx.drawImage(bitmap, 0, 0, targetW, targetH);
    bitmap.close();
    return canvas.toDataURL('image/png');
  }

  // Opaque → JPEG for photos, PNG for screenshots/diagrams
  // Heuristic: PNG source likely means screenshot/diagram → keep PNG for edge clarity
  const usePng = originalType === 'image/png';

  // Draw on white background (prevents black-where-alpha-was for JPEG output)
  if (!usePng) {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, targetW, targetH);
  }
  ctx.drawImage(bitmap, 0, 0, targetW, targetH);
  bitmap.close();

  return usePng
    ? canvas.toDataURL('image/png')
    : canvas.toDataURL('image/jpeg', IMAGE_JPEG_QUALITY);
}

/** Sample pixels to detect if an image actually uses its alpha channel. */
function _detectAlpha(bitmap, ctx, w, h) {
  // Draw at small size for fast sampling
  const sampleW = Math.min(w, 128);
  const sampleH = Math.min(h, 128);
  const sampleCanvas = document.createElement('canvas');
  sampleCanvas.width = sampleW;
  sampleCanvas.height = sampleH;
  const sCtx = sampleCanvas.getContext('2d');
  sCtx.clearRect(0, 0, sampleW, sampleH);
  sCtx.drawImage(bitmap, 0, 0, sampleW, sampleH);
  const data = sCtx.getImageData(0, 0, sampleW, sampleH).data;
  // Check every 4th pixel's alpha channel (stride of 16 bytes = every 4th pixel)
  for (let i = 3; i < data.length; i += 16) {
    if (data[i] < 250) return true;  // Found a non-opaque pixel
  }
  return false;
}

/** Types that LLM vision APIs universally accept (no re-encoding needed). */
function _isLlmFriendlyType(type) {
  return type === 'image/jpeg' || type === 'image/png' ||
         type === 'image/gif' || type === 'image/webp';
}

function _readAsDataUrl(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(file);
  });
}

/** Upload a data-URL image to the server. Returns the serving URL, or null on failure. */
async function _uploadChatImage(dataUrl) {
  try {
    const resp = await fetch('/api/chat-images', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        data_url: dataUrl,
        session_id: state.currentSessionId || '',
      }),
    });
    if (!resp.ok) return null;
    const data = await resp.json();
    return data.url || null;  // "/api/chat-images/<id>"
  } catch {
    return null;  // graceful fallback to inline base64
  }
}

async function uploadAndAttachDocument(file) {
  const formData = new FormData();
  formData.append('file', file);

  const toastId = showToast(`Uploading ${file.name}...`, 'loading');

  try {
    const resp = await fetch('/api/documents', {
      method: 'POST',
      body: formData,
    });
    const data = await resp.json();

    if (!resp.ok) {
      updateToast(toastId, extractErrorMessage(data, 'Upload failed'), 'error');
      return;
    }

    // Auto-bind to current session if one exists.
    // Default to "full" for inline attachments — the user dropped a file into
    // chat, they want the model to read it, not search fragments of it.
    const sessionId = state.currentSessionId;
    if (sessionId) {
      try {
        await fetch(`/api/documents/session/${encodeURIComponent(sessionId)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ document_id: data.id, inject_mode: 'full' }),
        });
      } catch (_) { /* session bind is best-effort */ }
    }

    pendingDocuments.push({
      id: data.id,
      filename: data.filename || file.name,
      chunk_count: data.chunk_count || 0,
    });

    renderAttachPreviews();
    updateToast(toastId, `${file.name}: ${data.chunk_count || 0} chunks indexed`, 'success');

    // Refresh sidebar document list and context bar
    if (typeof refreshDocumentList === 'function') refreshDocumentList();
    refreshDocContextBar();
  } catch (err) {
    updateToast(toastId, `Upload failed: ${err.message}`, 'error');
  }
}

function removeImage(index) {
  pendingImages.splice(index, 1);
  renderAttachPreviews();
}

function removeDocument(index) {
  pendingDocuments.splice(index, 1);
  renderAttachPreviews();
}

function clearPendingImages() {
  pendingImages = [];
  renderAttachPreviews();
}

function clearPendingDocuments() {
  pendingDocuments = [];
  renderAttachPreviews();
}

function renderAttachPreviews() {
  const preview = document.getElementById('attach-preview');
  const btn = document.getElementById('attach-btn');
  if (!preview) return;

  const hasAttachments = pendingImages.length > 0 || pendingDocuments.length > 0;

  if (!hasAttachments) {
    preview.classList.add('hidden');
    preview.innerHTML = '';
    btn?.classList.remove('has-attachments');
    return;
  }

  btn?.classList.add('has-attachments');
  preview.classList.remove('hidden');

  const imageHtml = pendingImages.map((src, i) => `
    <div class="attach-thumb attach-image">
      <img src="${escapeHtml(src)}" alt="Attached image ${i + 1}" />
      <button class="remove-btn" data-type="image" data-index="${i}" title="Remove">&times;</button>
    </div>
  `).join('');

  const docHtml = pendingDocuments.map((doc, i) => {
    // Web content attachment (from browse Discuss button or web search popover)
    if (doc._webTitle) {
      let domain = '';
      try { domain = new URL(doc._webUrl || '').hostname.replace(/^www\./, ''); } catch {}
      return `<div class="attach-thumb attach-web" title="${escapeHtml(doc._webTitle)}">
        <svg class="attach-web-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/>
          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
        </svg>
        <div class="attach-web-info">
          <span class="attach-web-title">${escapeHtml(doc._webTitle)}</span>
          ${domain ? `<span class="attach-web-domain">${escapeHtml(domain)}</span>` : ''}
        </div>
        <button class="remove-btn" data-type="doc" data-index="${i}" title="Remove">&times;</button>
      </div>`;
    }
    // Regular file attachment
    return `<div class="attach-thumb attach-doc" title="${escapeHtml(doc.filename)}">
      <svg class="attach-doc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
      <span class="attach-doc-name">${escapeHtml(doc.filename)}</span>
      <button class="remove-btn" data-type="doc" data-index="${i}" title="Remove">&times;</button>
    </div>`;
  }).join('');

  preview.innerHTML = imageHtml + docHtml;

  preview.querySelectorAll('.remove-btn').forEach(b => {
    b.addEventListener('click', () => {
      if (b.dataset.type === 'image') removeImage(parseInt(b.dataset.index, 10));
      else removeDocument(parseInt(b.dataset.index, 10));
    });
  });
}

// ---------------------------------------------------------------------------
// Document Context Bar (bound docs for current session)
// ---------------------------------------------------------------------------

function _modeSupportsDocContext(mode = state.mode) {
  return mode === 'passthrough' || mode === 'analytical' || mode === 'agentic';
}

function _docContextAddButtonHtml(expanded = false) {
  return `<button class="doc-ctx-add${expanded ? ' doc-ctx-add--inline' : ''}" title="Add knowledge source" aria-label="Add knowledge source">
    <span class="doc-ctx-add-plus" aria-hidden="true">+</span>
    ${expanded ? '<span class="doc-ctx-add-label">Knowledge</span>' : ''}
  </button>`;
}

async function _openDocPickerFromContextBar(addBtn, currentDocBindings, currentPackBindings, sessionId, onChange = null) {
  let effectiveSessionId = sessionId || state.currentSessionId;
  if (!effectiveSessionId) {
    effectiveSessionId = chat.createSession(state.mode || 'passthrough');
    await refreshDocContextBar();
  }
  // Per-surface bars pass onChange, whose closure re-anchors to their own
  // bar; the singleton falls back to re-querying its global elements.
  const currentBar = onChange ? addBtn.closest('.doc-context-bar') : document.getElementById('doc-context-bar');
  const currentAddBtn = currentBar?.querySelector('.doc-ctx-add') || addBtn;
  await _showDocPicker(currentAddBtn, currentDocBindings, currentPackBindings, effectiveSessionId, onChange);
}

function _bindDocContextAddButton(bar, currentDocBindings, currentPackBindings, sessionId, onChange = null) {
  const addBtn = bar.querySelector('.doc-ctx-add');
  if (!addBtn) return;
  addBtn.addEventListener('click', () => {
    _openDocPickerFromContextBar(addBtn, currentDocBindings, currentPackBindings, sessionId, onChange);
  });
}

/** Refresh the singleton knowledge context bar (primary composer). */
export async function refreshDocContextBar() {
  const bar = document.getElementById('doc-context-bar');
  if (!bar) return;
  await renderDocContextBarInto(bar, state.currentSessionId, state.mode);
}

/**
 * Render a knowledge context bar (bound docs + packs as pills) into an
 * arbitrary bar element for an arbitrary session. Extracted from the
 * singleton refreshDocContextBar so per-surface composer toolbars can host
 * their own bar against their own sessionId (surface-owned composer spec).
 *
 * @param {HTMLElement} bar       target `.doc-context-bar` element
 * @param {string|null} sessionId session whose bindings to show/edit
 * @param {string}      mode      owning surface's mode (gates visibility)
 * @param {Function}    [onChange] called after picker attaches change
 *                                bindings — per-surface bars re-render
 *                                themselves; singleton defaults to
 *                                refreshDocContextBar
 */
export async function renderDocContextBarInto(bar, sessionId, mode, onChange = null) {
  if (!bar) return;

  if (!_modeSupportsDocContext(mode)) {
    bar.classList.add('hidden');
    bar.innerHTML = '';
    return;
  }

  if (!sessionId) {
    bar.classList.remove('hidden');
    bar.classList.add('doc-ctx-empty');
    bar.innerHTML = _docContextAddButtonHtml(true);
    _bindDocContextAddButton(bar, [], [], null, onChange);
    return;
  }

  // Fetch document bindings and pack bindings in parallel
  let docBindings = [];
  let packBindings = [];
  try {
    const [docResp, packResp] = await Promise.all([
      fetch(`/api/documents/session/${encodeURIComponent(sessionId)}`),
      fetch(`/api/documents/session/${encodeURIComponent(sessionId)}/packs`),
    ]);
    if (docResp.ok) {
      const data = await docResp.json();
      docBindings = data.bindings || [];
    }
    if (packResp.ok) {
      const data = await packResp.json();
      packBindings = data.packs || [];
    }
  } catch { /* ignore */ }

  const isEmpty = docBindings.length === 0 && packBindings.length === 0;
  bar.classList.remove('hidden');
  bar.classList.toggle('doc-ctx-empty', isEmpty);

  // Document pills (existing behavior)
  const docPillsHtml = docBindings.map(b => {
    const mode = b.inject_mode === 'full' ? 'Full' : 'RAG';
    const modeClass = b.inject_mode === 'full' ? ' mode-full' : '';
    return `<div class="doc-ctx-pill${modeClass}" data-doc-id="${escapeHtml(b.document_id)}" data-mode="${escapeHtml(b.inject_mode)}" data-type="doc" title="Click mode to toggle RAG/Full">
      <svg class="doc-ctx-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
      <span class="doc-ctx-name">${escapeHtml(b.filename || 'Document')}</span>
      <span class="doc-ctx-mode">${mode}</span>
      <button class="doc-ctx-dismiss" title="Remove from chat">&times;</button>
    </div>`;
  }).join('');

  // Knowledge pack pills (RAG-only, no mode toggle)
  const packPillsHtml = packBindings.map(p => {
    return `<div class="doc-ctx-pill" data-pack-id="${escapeHtml(p.pack_id)}" data-type="pack" title="${escapeHtml(p.name)} (${p.chunk_count} chunks)">
      <svg class="doc-ctx-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
      </svg>
      <span class="doc-ctx-name">${escapeHtml(p.name || p.pack_id)}</span>
      <span class="doc-ctx-mode">RAG</span>
      <button class="doc-ctx-dismiss" title="Remove from chat">&times;</button>
    </div>`;
  }).join('');

  const addBtnHtml = _docContextAddButtonHtml(isEmpty);
  bar.innerHTML = docPillsHtml + packPillsHtml + addBtnHtml;

  // Wire mode toggle for DOCUMENT pills only (click on pill body, not dismiss)
  bar.querySelectorAll('.doc-ctx-pill[data-type="doc"]').forEach(pill => {
    pill.addEventListener('click', async (e) => {
      if (e.target.closest('.doc-ctx-dismiss')) return;
      const docId = pill.dataset.docId;
      const currentMode = pill.dataset.mode;
      const newMode = currentMode === 'search' ? 'full' : 'search';
      try {
        const resp = await fetch(
          `/api/documents/session/${encodeURIComponent(sessionId)}/${docId}/mode`,
          { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ inject_mode: newMode }) }
        );
        if (resp.ok) {
          pill.dataset.mode = newMode;
          pill.classList.toggle('mode-full', newMode === 'full');
          pill.querySelector('.doc-ctx-mode').textContent = newMode === 'full' ? 'Full' : 'RAG';
        }
      } catch { /* ignore */ }
    });
  });

  // Wire dismiss for ALL pills (docs + packs)
  bar.querySelectorAll('.doc-ctx-dismiss').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const pill = btn.closest('.doc-ctx-pill');
      if (!pill) return;

      try {
        if (pill.dataset.type === 'pack') {
          await fetch(`/api/documents/session/${encodeURIComponent(sessionId)}/packs/${encodeURIComponent(pill.dataset.packId)}`, { method: 'DELETE' });
        } else {
          await fetch(`/api/documents/session/${encodeURIComponent(sessionId)}/${pill.dataset.docId}`, { method: 'DELETE' });
        }
        pill.remove();
        if (!bar.querySelector('.doc-ctx-pill')) {
          bar.classList.add('doc-ctx-empty');
        }
      } catch { /* ignore */ }
    });
  });

  // Wire add button — unified picker
  _bindDocContextAddButton(bar, docBindings, packBindings, sessionId, onChange);
}

/** Coarse scale word for a knowledge pack based on chunk count. Tells
 *  the user at a glance whether a pack is "small reference" vs "vast
 *  encyclopedia" without making them parse a 7-digit number. */
function _packScale(chunks) {
  if (chunks >= 100000) return 'Vast';
  if (chunks >= 10000) return 'Large';
  if (chunks >= 1000) return 'Medium';
  return 'Small';
}

/** Human-friendly chunk count: 8432 → "8.4k", 102 → "102". */
function _formatChunkCount(n) {
  if (typeof n !== 'number' || !Number.isFinite(n)) return '0';
  if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
  return String(n);
}

/** Show a unified picker dropdown of docs and packs not yet bound. */
async function _showDocPicker(anchor, currentDocBindings, currentPackBindings, sessionId, onChange = null) {
  document.querySelectorAll('.doc-ctx-picker').forEach(p => p.remove());

  // Fetch docs and packs in parallel
  let allDocs = [];
  let allPacks = [];
  try {
    const [docResp, packResp] = await Promise.all([
      fetch('/api/documents'),
      fetch('/api/knowledge/packs'),
    ]);
    if (docResp.ok) { allDocs = (await docResp.json()).documents || []; }
    if (packResp.ok) { allPacks = (await packResp.json()).packs || []; }
  } catch { return; }

  const boundDocIds = new Set(currentDocBindings.map(b => b.document_id));
  const boundPackIds = new Set(currentPackBindings.map(p => p.pack_id));
  const unboundDocs = allDocs.filter(d => !boundDocIds.has(d.id));
  const unboundPacks = allPacks.filter(p => !boundPackIds.has(p.pack_id));

  const picker = document.createElement('div');
  picker.className = 'doc-ctx-picker';

  let html = '';

  // Documents section. Each row has two action buttons (RAG / Full) on
  // the right — tapping either one adds the doc with that mode in a
  // single gesture. Was previously a single-button row with no mode
  // choice; users had to bind first, then toggle from the context bar.
  if (unboundDocs.length > 0) {
    html += '<div class="doc-ctx-picker-label">Documents</div>';
    html += unboundDocs.map(d => {
      const ext = (d.filename || '').split('.').pop().toLowerCase();
      const iconType = _DOC_ICON_EXT[ext] || 'doc';
      const sizeLabel = d.file_size ? formatFileSize(d.file_size) : `${d.chunk_count} passages`;
      return `
      <div class="doc-ctx-picker-item" data-type="doc" data-doc-id="${escapeHtml(d.id)}">
        <span class="doc-ctx-picker-icon-wrap" data-icon-type="${escapeHtml(iconType)}" aria-hidden="true">${_docIconSvg(d.filename)}</span>
        <span class="doc-ctx-picker-text">
          <span class="doc-ctx-picker-name">${escapeHtml(d.filename)}</span>
          <span class="doc-ctx-picker-meta">${escapeHtml(sizeLabel)}</span>
        </span>
        <span class="doc-ctx-picker-actions">
          <button class="doc-ctx-add-mode" data-mode="search" aria-label="Attach ${escapeHtml(d.filename)} as RAG">RAG</button>
          <button class="doc-ctx-add-mode" data-mode="full" aria-label="Attach ${escapeHtml(d.filename)} as Full">Full</button>
        </span>
      </div>`;
    }).join('');
  }

  // Knowledge packs section. RAG-only by design — packs are too large
  // to inject in full — so each row gets a single "Attach" action.
  if (unboundPacks.length > 0) {
    if (unboundDocs.length > 0) html += '<div class="doc-ctx-picker-divider"></div>';
    html += '<div class="doc-ctx-picker-label">Knowledge Packs</div>';
    html += unboundPacks.map(p => {
      const scale = _packScale(p.chunk_count);
      const passages = _formatChunkCount(p.chunk_count);
      return `
      <div class="doc-ctx-picker-item" data-type="pack" data-pack-id="${escapeHtml(p.pack_id)}">
        <span class="doc-ctx-picker-icon-wrap" data-icon-type="pack" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
          </svg>
        </span>
        <span class="doc-ctx-picker-text">
          <span class="doc-ctx-picker-name">${escapeHtml(p.name || p.pack_id)}</span>
          <span class="doc-ctx-picker-meta">${escapeHtml(scale)} reference · ${escapeHtml(passages)} passages</span>
        </span>
        <span class="doc-ctx-picker-actions">
          <button class="doc-ctx-add-mode doc-ctx-add-mode--solo" data-mode="search" aria-label="Attach pack ${escapeHtml(p.name || p.pack_id)}">Attach</button>
        </span>
      </div>`;
    }).join('');
  }

  if (!html) {
    html = '<div class="doc-ctx-picker-empty">No knowledge sources available</div>';
  }

  picker.innerHTML = html;
  picker.classList.add('doc-ctx-picker-floating');
  document.body.appendChild(picker);

  // Positioning. Was: left-anchored to the button with a left-only
  // viewport clamp — when the + button sat near the right edge of the
  // viewport (which it does once any pills are bound) the 300px-wide
  // picker hung off the right side, hiding the RAG/Full controls. Now
  // right-edge-anchored to the button (picker grows leftward) with
  // both-side viewport clamps, and the max-height is set BEFORE
  // measuring so offsetHeight reflects the clamped size, not the
  // natural content height — keeps the picker on-screen on iPhone.
  const rect = anchor.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const spaceAbove = rect.top - 16;
  const spaceBelow = vh - rect.bottom - 16;
  picker.style.maxHeight = `${Math.min(400, Math.max(spaceAbove, spaceBelow))}px`;

  const pickerHeight = picker.offsetHeight;
  const pickerWidth = picker.offsetWidth;

  // Prefer above (natural for a bottom-anchored input toolbar) but
  // fall through to below if there isn't room.
  let top = (spaceAbove >= pickerHeight + 6)
    ? rect.top - pickerHeight - 6
    : rect.bottom + 6;
  top = Math.max(8, Math.min(top, vh - pickerHeight - 8));

  let left = rect.right - pickerWidth;
  left = Math.max(8, Math.min(left, vw - pickerWidth - 8));

  picker.style.left = `${left}px`;
  picker.style.top = `${top}px`;

  // Shared add helper — used by both the row-level default click and
  // the per-mode button clicks.
  const attach = async (row, mode) => {
    if (!row) return;
    try {
      if (row.dataset.type === 'pack') {
        await fetch(`/api/documents/session/${encodeURIComponent(sessionId)}/packs`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pack_id: row.dataset.packId }),
        });
      } else {
        await fetch(`/api/documents/session/${encodeURIComponent(sessionId)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ document_id: row.dataset.docId, inject_mode: mode }),
        });
      }
      picker.remove();
      if (onChange) onChange();
      else refreshDocContextBar();
    } catch { /* ignore */ }
  };

  // Row-level click — tap anywhere outside the mode buttons attaches
  // with RAG (the safe default, matches prior behavior). The Full
  // button stops propagation so it never double-fires through here.
  picker.querySelectorAll('.doc-ctx-picker-item').forEach(row => {
    row.addEventListener('click', () => attach(row, 'search'));
  });

  // Per-mode buttons — explicit mode choice, stop propagation so the
  // row default doesn't also fire.
  picker.querySelectorAll('.doc-ctx-add-mode').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const row = btn.closest('.doc-ctx-picker-item');
      attach(row, btn.dataset.mode || 'search');
    });
  });

  const closeHandler = (e) => {
    if (!picker.contains(e.target) && e.target !== anchor) {
      picker.remove();
      document.removeEventListener('click', closeHandler, true);
    }
  };
  setTimeout(() => document.addEventListener('click', closeHandler, true), 0);
}

// ---------------------------------------------------------------------------
// Toast Notification System (Sonner-inspired)
// ---------------------------------------------------------------------------
let toastCounter = 0;
const _toastTimers = new Map();   // id → { timeout, remaining, startedAt }
let _toastsPaused = false;

const _TOAST_ICONS = {
  success: '<svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>',
  error:   '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
  warning: '<svg viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  info:    '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
};

/**
 * Show a toast notification.
 * Backward-compatible: showToast(message, type, duration)
 * Enhanced: showToast(message, type, duration, opts)
 *
 * @param {string} message
 * @param {'success'|'warning'|'error'|'info'|'loading'} type
 * @param {number} [duration] - Auto-dismiss ms (0=persistent). Defaults: error=5000, loading=0, others=3000
 * @param {object} [opts]
 * @param {string} [opts.description] - Secondary text
 * @param {{label:string, onClick:Function}} [opts.action] - Action button
 * @param {boolean} [opts.dismissible] - Show dismiss X (default true)
 * @returns {number} Toast ID
 */
/**
 * Extract a human-readable error message from a FastAPI-shaped error body.
 *
 * FastAPI emits errors in several distinct shapes depending on which layer
 * raised them, and template-literalising the wrong one renders
 * "[object Object]" to the user:
 *
 *   { detail: "string" }              ← HTTPException with a string
 *   { detail: [{type,loc,msg,input}]} ← Pydantic 422 validation
 *   { detail: { msg, ...} }           ← HTTPException with a dict
 *   { error: "string" }               ← Augmentum's own exception handlers
 *   { error: { ... } }                ← rare; legacy custom handler
 *
 * This helper walks the candidates in priority order and produces a string
 * for any of them. Pydantic 422 arrays are joined with "; ". Anything truly
 * unrepresentable falls through to the caller's ``fallback`` (or a generic
 * "Request failed" if not supplied).
 *
 * Always prefer this over ``err.detail || err.error || 'x'`` template-string
 * concatenation in toast/error rendering paths.
 *
 * @param {object|null|undefined} body - Parsed JSON body from a non-OK response.
 * @param {string} [fallback] - Used when body has nothing extractable.
 * @returns {string} Human-readable single-line error message.
 */
export function extractErrorMessage(body, fallback = 'Request failed') {
  if (!body || typeof body !== 'object') {
    return typeof body === 'string' && body ? body : fallback;
  }
  const fromDetail = _extractFromField(body.detail);
  if (fromDetail) return fromDetail;
  const fromError = _extractFromField(body.error);
  if (fromError) return fromError;
  if (typeof body.message === 'string' && body.message) return body.message;
  if (typeof body.hint === 'string' && body.hint) return body.hint;
  return fallback;
}

function _extractFromField(field) {
  if (!field) return '';
  if (typeof field === 'string') return field;
  if (Array.isArray(field)) {
    // Pydantic 422 — list of {type, loc, msg, input} dicts.
    const parts = field.map((entry) => {
      if (typeof entry === 'string') return entry;
      if (entry && typeof entry === 'object' && typeof entry.msg === 'string') {
        const loc = Array.isArray(entry.loc) && entry.loc.length
          ? ` (${entry.loc.filter((p) => p !== 'body').join('.')})`
          : '';
        return entry.msg + loc;
      }
      try { return JSON.stringify(entry); } catch { return String(entry); }
    });
    return parts.join('; ');
  }
  if (typeof field === 'object') {
    if (typeof field.msg === 'string') return field.msg;
    if (typeof field.message === 'string') return field.message;
    if (typeof field.error === 'string') return field.error;
    try { return JSON.stringify(field); } catch { return String(field); }
  }
  return String(field);
}

export function showToast(message, type = 'info', duration, opts = {}) {
  if (duration === undefined) {
    duration = type === 'error' ? 5000 : type === 'loading' ? 0 : 3000;
  }

  const id = ++toastCounter;
  const toast = document.createElement('div');
  toast.className = `toast ${type} toast-enter`;
  toast.setAttribute('role', type === 'error' ? 'alert' : 'status');
  toast.setAttribute('aria-live', type === 'error' ? 'assertive' : 'polite');
  toast.dataset.id = id;

  let html = '';

  // Icon or spinner
  if (type === 'loading') {
    html += '<div class="toast-spinner"></div>';
  } else if (_TOAST_ICONS[type]) {
    html += `<div class="toast-icon">${_TOAST_ICONS[type]}</div>`;
  }

  // Body
  html += '<div class="toast-body">';
  html += `<div class="toast-message">${escapeHtml(message)}</div>`;
  if (opts.description) {
    html += `<div class="toast-description">${escapeHtml(opts.description)}</div>`;
  }
  html += '</div>';

  // Action button
  if (opts.action) {
    html += `<button class="toast-action">${escapeHtml(opts.action.label)}</button>`;
  }

  // Dismiss X
  if (opts.dismissible !== false && type !== 'loading') {
    html += '<div class="toast-dismiss" role="button" aria-label="Dismiss">&#x2715;</div>';
  }

  toast.innerHTML = html;

  // Wire action
  if (opts.action) {
    toast.querySelector('.toast-action')?.addEventListener('click', () => {
      opts.action.onClick();
      _dismissToastEl(toast, id);
    });
  }

  // Wire dismiss
  toast.querySelector('.toast-dismiss')?.addEventListener('click', () => _dismissToastEl(toast, id));

  // Append (lazy fallback if called before cacheDom)
  const container = dom.toastContainer || document.getElementById('toast-container');
  if (!container) return id;
  container.appendChild(toast);

  // Trigger entry animation next frame
  requestAnimationFrame(() => {
    requestAnimationFrame(() => toast.classList.remove('toast-enter'));
  });

  // Auto-dismiss timer
  if (duration > 0) {
    _startToastTimer(id, toast, duration);
  }

  // Pause timers on hover
  toast.addEventListener('mouseenter', _pauseToastTimers);
  toast.addEventListener('mouseleave', _resumeToastTimers);

  return id;
}

/**
 * Update an existing toast (e.g., promote loading → success/error).
 */
export function updateToast(id, message, type = 'success', opts = {}) {
  const toast = dom.toastContainer?.querySelector(`.toast[data-id="${id}"]`);
  if (!toast) return;
  _clearToastTimer(id);

  toast.className = `toast ${type}`;
  toast.dataset.id = id;

  let html = '';
  if (type === 'loading') {
    html += '<div class="toast-spinner"></div>';
  } else if (_TOAST_ICONS[type]) {
    html += `<div class="toast-icon">${_TOAST_ICONS[type]}</div>`;
  }
  html += '<div class="toast-body">';
  html += `<div class="toast-message">${escapeHtml(message)}</div>`;
  if (opts.description) html += `<div class="toast-description">${escapeHtml(opts.description)}</div>`;
  html += '</div>';
  if (opts.action) html += `<button class="toast-action">${escapeHtml(opts.action.label)}</button>`;
  if (opts.dismissible !== false && type !== 'loading') {
    html += '<div class="toast-dismiss" role="button" aria-label="Dismiss">&#x2715;</div>';
  }
  toast.innerHTML = html;

  if (opts.action) {
    toast.querySelector('.toast-action')?.addEventListener('click', () => {
      opts.action.onClick();
      _dismissToastEl(toast, id);
    });
  }
  toast.querySelector('.toast-dismiss')?.addEventListener('click', () => _dismissToastEl(toast, id));

  const dur = type === 'error' ? 5000 : type === 'loading' ? 0 : 3000;
  if (dur > 0) _startToastTimer(id, toast, dur);
}

/**
 * Dismiss a toast by ID or element.
 */
export function dismissToast(idOrEl) {
  if (typeof idOrEl === 'number') {
    const toast = dom.toastContainer?.querySelector(`.toast[data-id="${idOrEl}"]`);
    if (toast) _dismissToastEl(toast, idOrEl);
  } else if (idOrEl instanceof HTMLElement) {
    _dismissToastEl(idOrEl, Number(idOrEl.dataset.id));
  }
}

/**
 * Persistent toast presenting several actions the user must choose between —
 * for surfacing a decision instead of silently defaulting (never auto-select).
 * Each choice: { label, onClick, primary? }. The toast stays until a choice is
 * made or (if allowed) dismissed; clicking a choice runs its onClick then
 * closes. ``opts.onDismiss`` fires when the user closes it without choosing.
 * Returns the toast id.
 */
export function showChoiceToast(message, choices = [], opts = {}) {
  const type = opts.type || 'warning';
  const id = ++toastCounter;
  const toast = document.createElement('div');
  toast.className = `toast ${type} toast-choice toast-enter`;
  toast.setAttribute('role', 'alertdialog');
  toast.setAttribute('aria-live', 'assertive');
  toast.dataset.id = id;

  let html = '';
  if (_TOAST_ICONS[type]) html += `<div class="toast-icon">${_TOAST_ICONS[type]}</div>`;
  html += '<div class="toast-body">';
  html += `<div class="toast-message">${escapeHtml(message)}</div>`;
  if (opts.description) html += `<div class="toast-description">${escapeHtml(opts.description)}</div>`;
  html += '<div class="toast-choices">';
  choices.forEach((c, i) => {
    html += `<button class="toast-action${c.primary ? ' toast-action-primary' : ''}" data-choice="${i}">${escapeHtml(c.label)}</button>`;
  });
  html += '</div></div>';
  if (opts.dismissible !== false) {
    html += '<div class="toast-dismiss" role="button" aria-label="Dismiss">&#x2715;</div>';
  }
  toast.innerHTML = html;

  toast.querySelectorAll('.toast-action').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const choice = choices[Number(btn.dataset.choice)];
      _dismissToastEl(toast, id);
      try { await choice?.onClick?.(); } catch { /* action owns its own errors */ }
    });
  });
  // Dismissal is an outcome too. Callers that await a decision (wrapping this
  // in a Promise) hang forever without it — and a caller that hangs mid-flow
  // usually leaves a busy flag set, disabling the button that opened it.
  toast.querySelector('.toast-dismiss')?.addEventListener('click', () => {
    _dismissToastEl(toast, id);
    try { opts.onDismiss?.(); } catch { /* dismissal owns its own errors */ }
  });

  const container = dom.toastContainer || document.getElementById('toast-container');
  if (!container) return id;
  container.appendChild(toast);
  requestAnimationFrame(() => requestAnimationFrame(() => toast.classList.remove('toast-enter')));
  // Persistent by design — a decision shouldn't time out from under the user.
  return id;
}

function _dismissToastEl(toast, id) {
  _clearToastTimer(id);
  toast.classList.add('toast-exit');
  toast.addEventListener('transitionend', () => toast.remove(), { once: true });
  setTimeout(() => { if (toast.parentNode) toast.remove(); }, 300);
}

function _startToastTimer(id, toast, duration) {
  _clearToastTimer(id);
  const timeout = setTimeout(() => {
    const el = dom.toastContainer?.querySelector(`.toast[data-id="${id}"]`);
    if (el) _dismissToastEl(el, id);
  }, duration);
  _toastTimers.set(id, { timeout, remaining: duration, startedAt: Date.now() });
}

function _clearToastTimer(id) {
  const timer = _toastTimers.get(id);
  if (timer) {
    clearTimeout(timer.timeout);
    _toastTimers.delete(id);
  }
}

function _pauseToastTimers() {
  if (_toastsPaused) return;
  _toastsPaused = true;
  for (const [, timer] of _toastTimers) {
    clearTimeout(timer.timeout);
    timer.remaining -= (Date.now() - timer.startedAt);
  }
}

function _resumeToastTimers() {
  if (!_toastsPaused) return;
  _toastsPaused = false;
  for (const [id, timer] of _toastTimers) {
    const toast = dom.toastContainer?.querySelector(`.toast[data-id="${id}"]`);
    if (toast && timer.remaining > 0) {
      timer.startedAt = Date.now();
      timer.timeout = setTimeout(() => {
        const el = dom.toastContainer?.querySelector(`.toast[data-id="${id}"]`);
        if (el) _dismissToastEl(el, id);
      }, timer.remaining);
    }
  }
}

// Pause all toast timers when tab is hidden
document.addEventListener('visibilitychange', () => {
  if (document.hidden) _pauseToastTimers();
  else _resumeToastTimers();
});

// ---------------------------------------------------------------------------
// Confirmation Dialog
// ---------------------------------------------------------------------------

/**
 * Show a confirmation dialog. Returns a Promise that resolves to true (confirmed) or false (cancelled).
 *
 * @param {object} opts
 * @param {string} opts.title - Dialog title
 * @param {string} opts.message - Body message (can include HTML for emphasis)
 * @param {string} [opts.confirmLabel='Delete'] - Confirm button text (should name the action)
 * @param {string} [opts.cancelLabel='Cancel'] - Cancel button text
 * @param {'danger'|'primary'} [opts.variant='danger'] - Confirm button style
 * @param {string} [opts.confirmInput] - If set, user must type this string to enable confirm (type-to-confirm)
 * @returns {Promise<boolean>}
 */
export function showConfirm(opts) {
  const {
    title,
    message,
    confirmLabel = 'Delete',
    cancelLabel = 'Cancel',
    variant = 'danger',
    confirmInput,
  } = opts;

  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'confirm-overlay';

    const btnClass = variant === 'primary' ? 'confirm-btn-primary' : 'confirm-btn-danger';

    let inputHtml = '';
    if (confirmInput) {
      inputHtml = `
        <p class="confirm-input-hint">Type <strong>${escapeHtml(confirmInput)}</strong> to confirm</p>
        <input class="confirm-input" type="text" autocomplete="off" spellcheck="false">`;
    }

    overlay.innerHTML = `
      <div class="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-dlg-title">
        <div class="confirm-title" id="confirm-dlg-title">${escapeHtml(title)}</div>
        <div class="confirm-message">${escapeHtml(message)}</div>
        ${inputHtml}
        <div class="confirm-actions">
          <button class="confirm-btn confirm-btn-cancel">${escapeHtml(cancelLabel)}</button>
          <button class="confirm-btn ${btnClass}"${confirmInput ? ' disabled' : ''}>${escapeHtml(confirmLabel)}</button>
        </div>
      </div>`;

    const cancelBtn = overlay.querySelector('.confirm-btn-cancel');
    const confirmBtn = overlay.querySelector(`.${btnClass}`);
    const input = overlay.querySelector('.confirm-input');

    let settled = false;

    function close(result) {
      if (settled) return;
      settled = true;
      overlay.classList.add('confirm-exit');
      setTimeout(() => {
        overlay.remove();
        const appEl = document.getElementById('app');
        if (appEl) appEl.removeAttribute('inert');
        resolve(result);
      }, 150);
    }

    // Type-to-confirm input
    if (input) {
      input.addEventListener('input', () => {
        confirmBtn.disabled = input.value !== confirmInput;
      });
    }

    confirmBtn.addEventListener('click', () => close(true));
    cancelBtn.addEventListener('click', () => close(false));

    // Click outside dialog = cancel
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) close(false);
    });

    // Escape key = cancel
    overlay.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') close(false);
    });

    document.body.appendChild(overlay);

    // Trap focus by marking app inert
    const appEl = document.getElementById('app');
    if (appEl) appEl.setAttribute('inert', '');

    // Entry animation + focus
    requestAnimationFrame(() => {
      overlay.classList.add('confirm-visible');
      if (input) {
        input.focus();
      } else {
        cancelBtn.focus();
      }
    });
  });
}

export function showDangerConfirm(title, message, confirmLabel = 'Delete') {
  return showConfirm({ title, message, confirmLabel, variant: 'danger' });
}

// ---------------------------------------------------------------------------
// Document RAG Management
// ---------------------------------------------------------------------------
const _DOC_SECTION_COLLAPSED_KEY = 'augmentum:doc-section-collapsed';

function _setDocSectionCollapsed(collapsed) {
  const section = document.getElementById('doc-section');
  const toggle = document.getElementById('doc-section-toggle');
  if (!section || !toggle) return;
  section.classList.toggle('collapsed', !!collapsed);
  toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
  toggle.setAttribute('title', collapsed ? 'Show documents' : 'Hide documents');
  try { localStorage.setItem(_DOC_SECTION_COLLAPSED_KEY, collapsed ? '1' : '0'); } catch {}
}

function initDocuments() {
  const uploadBtn = document.getElementById('upload-doc-btn');
  const fileInput = document.getElementById('doc-upload-input');
  if (!uploadBtn || !fileInput) return;

  // Upload button: don't let the click bubble to the header toggle, and
  // auto-expand the section so the user sees their newly-added doc land
  // in the list. Treats "I'm adding a doc" as implicit intent to review.
  uploadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    _setDocSectionCollapsed(false);
    fileInput.click();
  });
  fileInput.addEventListener('change', async () => {
    if (!fileInput.files) return;
    for (const file of fileInput.files) {
      await uploadDocument(file);
    }
    fileInput.value = '';
  });

  // Collapsible header. Default state = collapsed so chat history owns
  // most of the panel; respect any prior user choice from localStorage.
  const toggle = document.getElementById('doc-section-toggle');
  if (toggle) {
    let saved = null;
    try { saved = localStorage.getItem(_DOC_SECTION_COLLAPSED_KEY); } catch {}
    // Default: collapsed. Only honor an explicit '0' (expanded) preference.
    _setDocSectionCollapsed(saved !== '0');
    toggle.addEventListener('click', (e) => {
      // Clicks on the nested upload button are handled above and shouldn't
      // reach here, but guard defensively.
      if (e.target.closest('#upload-doc-btn')) return;
      const section = document.getElementById('doc-section');
      _setDocSectionCollapsed(!section?.classList.contains('collapsed') ? true : false);
    });
    toggle.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      if (e.target.closest('#upload-doc-btn')) return;
      e.preventDefault();
      const section = document.getElementById('doc-section');
      _setDocSectionCollapsed(!section?.classList.contains('collapsed') ? true : false);
    });
  }

  // Load initial list
  refreshDocumentList();

  // Refresh bindings when active session changes
  document.addEventListener('augmentum:session-changed', () => {
    refreshDocumentList();
    refreshDocContextBar();
  });
}

async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);

  const toastId = showToast(`Uploading ${file.name}...`, 'loading');

  try {
    const resp = await fetch('/api/documents', {
      method: 'POST',
      body: formData,
    });
    const data = await resp.json();

    if (resp.ok) {
      updateToast(toastId, `${file.name}: ${data.chunk_count} chunks indexed`, 'success');
      refreshDocumentList();
    } else {
      updateToast(toastId, extractErrorMessage(data, 'Upload failed'), 'error');
    }
  } catch (err) {
    updateToast(toastId, `Upload failed: ${err.message}`, 'error');
  }
}

const _DOC_ICON_SVGS = {
  doc: '<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/><path d="M14 2v6h6"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="14" y2="17"/></svg>',
  code: '<svg viewBox="0 0 24 24"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
  sheet: '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>',
  image: '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="9" r="1.5"/><path d="m21 15-5-5-9 9"/></svg>',
  book: '<svg viewBox="0 0 24 24"><path d="M12 6v15"/><path d="M3 4a2 2 0 0 1 2-2h5a3 3 0 0 1 2 1 3 3 0 0 1 2-1h5a2 2 0 0 1 2 2v13a2 2 0 0 1-2 2h-5a3 3 0 0 0-2 1 3 3 0 0 0-2-1H5a2 2 0 0 1-2-2z"/></svg>',
};

const _DOC_ICON_EXT = {
  js: 'code', ts: 'code', py: 'code', html: 'code', htm: 'code', xml: 'code',
  json: 'code', yaml: 'code', yml: 'code', toml: 'code', sh: 'code', bash: 'code',
  c: 'code', h: 'code', cpp: 'code', rs: 'code', go: 'code',
  csv: 'sheet', tsv: 'sheet', xlsx: 'sheet', xls: 'sheet', ods: 'sheet',
  png: 'image', jpg: 'image', jpeg: 'image', gif: 'image', webp: 'image',
  bmp: 'image', svg: 'image', avif: 'image', tiff: 'image',
  epub: 'book', mobi: 'book', azw: 'book', azw3: 'book',
};

function _docIconSvg(filename) {
  const ext = (filename || '').split('.').pop().toLowerCase();
  return _DOC_ICON_SVGS[_DOC_ICON_EXT[ext] || 'doc'];
}

export async function refreshDocumentList() {
  const list = document.getElementById('doc-list');
  const countEl = document.getElementById('doc-section-count');
  if (!list) return;

  try {
    const resp = await fetch('/api/documents');
    if (!resp.ok) return;
    const data = await resp.json();

    const docCount = (data.documents || []).length;
    if (countEl) countEl.textContent = docCount ? String(docCount) : '';

    if (!data.documents || data.documents.length === 0) {
      list.innerHTML = '<div class="doc-empty">No documents uploaded</div>';
      return;
    }

    // Get current session bindings. Fall back to sessionStore for the
    // page-load race where localStorage has restored an active session
    // but the session-changed event hasn't propagated to app.state yet.
    let sessionId = app?.state?.currentSessionId || null;
    if (!sessionId) {
      try {
        const { sessionStore } = await import('./chat/sessions.js');
        sessionId = sessionStore?.getActiveId?.() || null;
      } catch { /* ignore */ }
    }
    let bindings = {};
    if (sessionId) {
      try {
        const bResp = await fetch(`/api/documents/session/${encodeURIComponent(sessionId)}`);
        if (bResp.ok) {
          const bData = await bResp.json();
          for (const b of (bData.bindings || [])) {
            bindings[b.document_id] = b.inject_mode;
          }
        }
      } catch { /* ignore */ }
    }

    list.innerHTML = data.documents.map(doc => {
      const isBound = doc.id in bindings;
      const mode = bindings[doc.id] || 'search';
      const cls = `doc-item${isBound ? ' bound' : ''}`;
      const tip = `${doc.filename} — ${doc.chunk_count} chunks, ${formatFileSize(doc.file_size)}`;
      return `
      <label class="${cls}" data-id="${escapeHtml(doc.id)}" title="${escapeHtml(tip)}">
        <input type="checkbox" class="doc-bind-check" ${isBound ? 'checked' : ''} />
        <span class="doc-item-icon" aria-hidden="true">${_docIconSvg(doc.filename)}</span>
        <div class="doc-item-info">
          <span class="doc-item-name">${escapeHtml(doc.filename)}</span>
          <span class="doc-item-meta">${escapeHtml(doc.file_size ? formatFileSize(doc.file_size) : `${doc.chunk_count} passages`)}</span>
        </div>
        <button type="button" class="doc-mode-btn ${mode === 'full' ? 'mode-full' : ''}"
          title="Toggle: search (RAG) vs full (inject entire document)"
          ${isBound ? '' : 'hidden'}>${mode === 'full' ? 'Full' : 'RAG'}</button>
        <button type="button" class="icon-btn small doc-delete-btn" title="Delete document">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
          </svg>
        </button>
      </label>`;
    }).join('');

    // Wire checkboxes — bind/unbind document to session
    list.querySelectorAll('.doc-bind-check').forEach(cb => {
      cb.addEventListener('change', async (e) => {
        const item = e.target.closest('.doc-item');
        const docId = item?.dataset.id;
        if (!docId) return;
        const modeBtn = item.querySelector('.doc-mode-btn');

        // Resolve session id: prefer render-time value, then live
        // app.state, then sessionStore (which reflects localStorage on
        // page load before session-changed fires). Only mint a brand-new
        // session as a last resort — otherwise a click during the page-
        // load race would orphan the user's real active chat.
        let effectiveSessionId = sessionId || app?.state?.currentSessionId;
        if (!effectiveSessionId) {
          try {
            const { sessionStore } = await import('./chat/sessions.js');
            effectiveSessionId = sessionStore?.getActiveId?.() || null;
          } catch { /* ignore */ }
        }
        if (!effectiveSessionId && e.target.checked) {
          try {
            effectiveSessionId = chat?.createSession?.(app?.state?.mode || 'passthrough');
          } catch { /* fallthrough */ }
          if (!effectiveSessionId) {
            e.target.checked = false;
            showToast('Start a chat to attach documents', 'info');
            return;
          }
        }
        if (!effectiveSessionId) { e.target.checked = false; return; }

        if (e.target.checked) {
          try {
            await fetch(`/api/documents/session/${encodeURIComponent(effectiveSessionId)}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ document_id: docId, inject_mode: 'search' }),
            });
            item.classList.add('bound');
            if (modeBtn) { modeBtn.hidden = false; modeBtn.textContent = 'RAG'; modeBtn.classList.remove('mode-full'); }
            showToast('Document enabled for this chat', 'success');
          } catch { /* ignore */ }
        } else {
          try {
            await fetch(`/api/documents/session/${encodeURIComponent(effectiveSessionId)}/${docId}`, {
              method: 'DELETE',
            });
            item.classList.remove('bound');
            if (modeBtn) modeBtn.hidden = true;
            showToast('Document removed from this chat', 'success');
          } catch { /* ignore */ }
        }
        refreshDocumentList();
        refreshDocContextBar();
      });
    });

    // Wire mode toggle buttons
    list.querySelectorAll('.doc-mode-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const item = e.target.closest('.doc-item');
        const docId = item?.dataset.id;
        if (!docId || !sessionId) return;

        const currentMode = btn.textContent.trim().toLowerCase() === 'full' ? 'full' : 'search';
        const newMode = currentMode === 'search' ? 'full' : 'search';

        try {
          const resp = await fetch(
            `/api/documents/session/${encodeURIComponent(sessionId)}/${docId}/mode`,
            {
              method: 'PUT',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ inject_mode: newMode }),
            }
          );
          if (resp.ok) {
            btn.textContent = newMode === 'full' ? 'Full' : 'RAG';
            btn.classList.toggle('mode-full', newMode === 'full');
            showToast(newMode === 'full' ? 'Full document will be injected' : 'Switched to RAG search mode', 'success');
            refreshDocContextBar();
          }
        } catch { /* ignore */ }
      });
    });

    // Wire delete buttons
    list.querySelectorAll('.doc-delete-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const item = e.target.closest('.doc-item');
        const id = item?.dataset.id;
        if (!id) return;
        const name = (item.querySelector('.doc-item-name')?.textContent || '').trim();
        const label = name ? `"${name}"` : 'this document';
        if (!confirm(`Delete ${label}? This cannot be undone.`)) return;
        try {
          const resp = await fetch(`/api/documents/${id}`, { method: 'DELETE' });
          if (resp.ok) {
            showToast('Document deleted', 'success');
            refreshDocumentList();
            refreshDocContextBar();
          }
        } catch { /* ignore */ }
      });
    });
  } catch { /* ignore */ }
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

// ---------------------------------------------------------------------------
// Keyboard Shortcuts
// ---------------------------------------------------------------------------
function initKeyboard() {
  document.addEventListener('keydown', (e) => {
    // Escape: stop TTS / close voice / close overlays
    if (e.key === 'Escape') {
      if (voice.isConnected) {
        voice.endVoiceCall();
        return;
      }
      chat.ttsStopCurrent();
      closeModelDropdown();
      document.getElementById('typo-dropdown')?.classList.add('hidden');
      dismissOverlays();
      if (state.panelOpen) closePanel();
      if (dom.inspectorPanel.classList.contains('mobile-open')) closeInspectorMobile();
      closeImagePanel();
      return;
    }

    // Ctrl+/ — focus input
    if (e.ctrlKey && e.key === '/') {
      e.preventDefault();
      dom.chatInput.focus();
      return;
    }

    // Ctrl+, — settings
    if (e.ctrlKey && e.key === ',') {
      e.preventDefault();
      openSettings();
      return;
    }

    // Ctrl+Shift+S — new session
    if (e.ctrlKey && e.shiftKey && e.key === 'S') {
      e.preventDefault();
      document.dispatchEvent(new CustomEvent('augmentum:new-session'));
      return;
    }
  });
}

// ---------------------------------------------------------------------------
// Click-Outside Handling
// ---------------------------------------------------------------------------
function initClickOutside() {
  document.addEventListener('click', () => {
    // (mode selector is now inline in sidebar, no dropdown to close)
  });
}

// ---------------------------------------------------------------------------
// Scroll to Bottom
// ---------------------------------------------------------------------------
export function scrollToBottom(smooth = true, force = false) {
  if (!force) {
    const el = dom.chatScroll;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (!atBottom) return;
  }
  dom.chatScroll.scrollTo({
    top: dom.chatScroll.scrollHeight,
    behavior: smooth ? 'smooth' : 'instant',
  });
}

// ---------------------------------------------------------------------------
// Empty State
// ---------------------------------------------------------------------------
export function updateEmptyState(hasMessages) {
  if (dom.emptyState) {
    dom.emptyState.style.display = hasMessages ? 'none' : '';
  }
  // Re-entry pull for language learners: when a passthrough session
  // is empty (fresh chat or just-cleared), surface any materialised
  // partners with their due-review counts above the generic prompt
  // chips. Lazy import keeps non-language users out of the partner
  // bundle. Best-effort: failures inside the renderer don't block
  // empty-state rendering. Gated on passthrough so narrative/coder/
  // agentic empty-states don't sprout a card. See
  // [[project-language-partner]].
  if (!hasMessages && dom.emptyState && state.mode === 'passthrough') {
    import('./learning_games/home_partner_card.js')
      .then(mod => mod.renderHomePartnerCards(dom.emptyState))
      .catch(err => console.warn('[empty-state] partner cards failed', err));
  }
}

// ---------------------------------------------------------------------------
// Passthrough Inspector Dashboard
// ---------------------------------------------------------------------------
function updatePassthroughDashboard() {
  // Status
  const dot = document.getElementById('pt-status-dot');
  const statusText = document.getElementById('pt-status-text');
  if (dot && statusText) {
    fetch('/api/health/services', { cache: 'no-store', signal: AbortSignal.timeout(3000) })
      .then(async r => {
        if (r.ok) {
          const services = await r.json().catch(() => ({}));
          const degraded = Object.values(services || {}).some(s => (
            s && typeof s === 'object' && ['degraded', 'down'].includes(s.status)
          ));
          dot.className = 'pt-status-dot connected';
          statusText.textContent = degraded ? 'Degraded' : 'Connected';
        } else {
          dot.className = 'pt-status-dot disconnected';
          statusText.textContent = 'Error';
        }
      })
      .catch(() => {
        dot.className = 'pt-status-dot disconnected';
        statusText.textContent = 'Disconnected';
      });
  }

  // Model
  const modelEl = document.getElementById('pt-model-name');
  if (modelEl) {
    modelEl.textContent = state.currentModel || 'default';
  }

  // Session stats — count messages from the DOM
  const msgEl = document.getElementById('pt-stat-messages');
  const tokEl = document.getElementById('pt-stat-tokens');
  if (msgEl) {
    const msgCount = dom.chatMessages ? dom.chatMessages.querySelectorAll('.message').length : 0;
    msgEl.textContent = msgCount;
  }
  if (tokEl) {
    // Rough estimate: ~4 chars per token from visible messages
    if (dom.chatMessages) {
      let totalChars = 0;
      dom.chatMessages.querySelectorAll('.message-content').forEach(el => {
        totalChars += (el.textContent || '').length;
      });
      const est = Math.round(totalChars / 4);
      tokEl.textContent = est > 0 ? (est > 1000 ? `${(est / 1000).toFixed(1)}k` : est) : '--';
    } else {
      tokEl.textContent = '--';
    }
  }

  // Mode quick-switch buttons
  document.querySelectorAll('.pt-mode-btn').forEach(btn => {
    btn.onclick = () => setMode(btn.dataset.mode);
  });
}

// ---------------------------------------------------------------------------
// Mobile Bottom Bar
// ---------------------------------------------------------------------------
let mobileModePicker = null;

function initMobileBottomBar() {
  const mobMenu = document.getElementById('mob-menu');
  const mobMode = document.getElementById('mob-mode');
  const mobInspector = document.getElementById('mob-inspector');

  if (!mobMenu) return; // Not present in DOM

  mobMenu.addEventListener('click', () => {
    closeMobileModePicker();
    togglePanel();
  });

  const mobBrowse = document.getElementById('mob-browse');
  if (mobBrowse) {
    mobBrowse.addEventListener('click', () => {
      closeMobileModePicker();
      document.getElementById('toggle-browse-btn')?.click();
    });
  }

  mobInspector.addEventListener('click', () => {
    closeMobileModePicker();
    toggleInspector();
  });

  mobMode.addEventListener('click', () => {
    if (mobileModePicker && mobileModePicker.classList.contains('visible')) {
      closeMobileModePicker();
    } else {
      openMobileModePicker();
    }
  });

  // Update mode label on mode change
  document.addEventListener('augmentum:mode-changed', (e) => {
    const label = document.getElementById('mob-mode-label');
    if (label) label.textContent = modeLabel(e.detail.mode);
    closeMobileModePicker();
  });
}

function _initHeaderOverflow() {
  const btn = $('header-overflow-btn');
  const menu = $('header-overflow-menu');
  if (!btn || !menu) return;

  // Action map — overflow items trigger the original hidden buttons.
  // Grove uses requestAnimationFrame to defer past the click event cycle,
  // preventing its click-outside handler from immediately closing the panel.
  const actions = {
    'manage-models': () => $('manage-models-btn')?.click(),
    'image':         () => $('toggle-image-btn')?.click(),
    'browse':        () => $('toggle-browse-btn')?.click(),
    'inspector':     () => toggleInspector(),
    'files':         () => $('files-btn')?.click(),
    'voice':         () => $('voice-call-btn')?.click(),
    'settings':      () => $('settings-btn')?.click(),
    'grove':         () => requestAnimationFrame(() => $('grove-btn')?.click()),
    'schedule':      () => import('./calendar/index.js')
                             .then((m) => m.open?.())
                             .catch((e) => console.warn('[calendar] open failed', e)),
  };

  const closeMenu = () => {
    menu.classList.add('hidden');
    btn.setAttribute('aria-expanded', 'false');
  };

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpening = menu.classList.contains('hidden');
    menu.classList.toggle('hidden');
    btn.setAttribute('aria-expanded', isOpening ? 'true' : 'false');
  });

  menu.addEventListener('click', (e) => {
    e.stopPropagation();
    const item = e.target.closest('.header-overflow-item');
    if (!item) return;
    const action = item.dataset.action;
    if (actions[action]) actions[action]();
    closeMenu();
  });

  // Close on outside click
  document.addEventListener('click', (e) => {
    if (!menu.classList.contains('hidden') && !menu.contains(e.target) && e.target !== btn) {
      closeMenu();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !menu.classList.contains('hidden')) {
      closeMenu();
      btn.focus();
    }
  });
}

function openMobileModePicker() {
  if (!mobileModePicker) {
    mobileModePicker = document.createElement('div');
    mobileModePicker.className = 'mobile-mode-picker';
    const modes = [
      { id: 'passthrough', color: 'var(--mode-passthrough)' },
      { id: 'analytical',  color: 'var(--mode-analytical)' },
      { id: 'narrative',   color: 'var(--mode-narrative)' },
      { id: 'agentic',     color: 'var(--mode-agentic)' },
      { id: 'coder',       color: 'var(--mode-coder)' },
    ];
    mobileModePicker.innerHTML = modes.map(m =>
      `<button class="mobile-mode-picker-option${state.mode === m.id ? ' active' : ''}" data-mode="${m.id}">
        <span class="mode-dot" style="background:${m.color}"></span>
        ${modeLabel(m.id)}
      </button>`
    ).join('');
    document.body.appendChild(mobileModePicker);

    mobileModePicker.querySelectorAll('.mobile-mode-picker-option').forEach(btn => {
      btn.addEventListener('click', () => {
        setMode(btn.dataset.mode);
        closeMobileModePicker();
      });
    });
  } else {
    // Update active state
    mobileModePicker.querySelectorAll('.mobile-mode-picker-option').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.mode === state.mode);
    });
  }

  requestAnimationFrame(() => {
    mobileModePicker.classList.add('visible');
  });

  // Close on outside click
  setTimeout(() => {
    document.addEventListener('click', _closeMobilePickerOutside, { once: true });
  }, 0);
}

function closeMobileModePicker() {
  if (mobileModePicker) mobileModePicker.classList.remove('visible');
}

function _closeMobilePickerOutside(e) {
  if (mobileModePicker && !mobileModePicker.contains(e.target) && e.target.id !== 'mob-mode') {
    closeMobileModePicker();
  }
}

// ---------------------------------------------------------------------------
// Toolbar Toggle Helpers
// ---------------------------------------------------------------------------

/** Flash scale animation on a toolbar button for visual feedback. */
// flashToolbarBtn + syncToggleToBackend moved to ./chat/toolbar/util.js
// as part of the surface-owned composer migration (Step 2). Imports at top.


// ---------------------------------------------------------------------------
// Scene Image Generation — header button with popover quick-settings
// ---------------------------------------------------------------------------

let _sceneGenModelsPopulated = false;

async function _populateSceneGenModels() {
  const sel = document.getElementById('scene-gen-model');
  if (!sel || _sceneGenModelsPopulated) return;
  _sceneGenModelsPopulated = true;

  // Local image models
  try {
    const models = await getImageModels();
    if (models.length > 0) {
      const group = document.createElement('optgroup');
      group.label = 'Local';
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name;
        group.appendChild(opt);
      }
      sel.appendChild(group);
    }
  } catch { /* ignore */ }

  // Cloud image models
  try {
    const cloudModels = await getCloudImageModels();
    if (cloudModels.length > 0) {
      const group = document.createElement('optgroup');
      group.label = 'Cloud';
      for (const m of cloudModels) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = `${m.name} (${m.provider || 'cloud'})`;
        group.appendChild(opt);
      }
      sel.appendChild(group);
    }
  } catch { /* ignore */ }
}

export async function _fireSceneGenerate() {
  const sessionId = state.currentSessionId;
  if (!sessionId) {
    showToast('No active session', 'error');
    return;
  }

  const btn = document.getElementById('scene-gen-btn');
  const btn2 = document.getElementById('instant-scene-btn');
  const pop = document.getElementById('scene-gen-popover');

  if (pop) pop.classList.add('hidden');
  if (btn) btn.dataset.state = 'loading';
  if (btn2) btn2.classList.add('loading');

  const instruction = dom.chatInput.value.trim();
  if (instruction) dom.chatInput.value = '';

  // Anchor result on the latest assistant bubble (or the messages container
  // itself if none exists). Rendering as a detached DOM element — never as a
  // chat-tree node — keeps the image out of the narrative model's context on
  // the next turn, so it can't imitate the rendered <img> markup as text.
  const messagesEl = document.querySelector('.chat-messages');
  const lastAssistant = messagesEl
    ? messagesEl.querySelector('.message.message-assistant:last-of-type .message-bubble')
    : null;
  const target = lastAssistant || messagesEl;
  if (!target) {
    showToast('No chat surface to render scene', 'error');
    if (btn) btn.dataset.state = 'idle';
    if (btn2) btn2.classList.remove('loading');
    return;
  }

  // Shared progress loader \u2014 polls /api/image/generation-status so
  // distill / load / step-by-step gen / save phases all surface as
  // real text + determinate bar instead of a generic spinner.
  const progress = createImageProgressLoader({
    session_id: sessionId,
    variant: 'scene',
  });
  progress.element.classList.add('scene-gen-loading');
  target.appendChild(progress.element);
  progress.start();
  if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;

  try {
    const session = chat.getActiveSession();
    const allMsgs = session ? chat.buildMessagesForAPI(session) : [];
    const convMsgs = allMsgs
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .slice(-4)
      .map(m => ({ role: m.role, content: m.content }));

    const activeChar = narrative.activeCharacter;

    // Image model/resolution/steps/cfg/sampler/etc. come from the server-side
    // image_active_settings (pushed by the image panel via _pushActiveSettings).
    // Sending no overrides forces the backend to honor the panel exactly —
    // matches illustrate.js so both paths use identical quality settings.
    const res = await fetch('/api/image/generate-scene', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        instruction,
        messages: convMsgs,
        character_name: activeChar?.name || '',
        visual_traits: activeChar?.visualTraits || '',
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(extractErrorMessage(err, `HTTP ${res.status}`));
    }

    const data = await res.json();

    progress.stop();
    const imgContainer = document.createElement('div');
    imgContainer.className = 'illustrate-result scene-gen-result';
    imgContainer.innerHTML = `
      <div class="illustrate-result-header">
        <span class="illustrate-result-label">Scene image</span>
        <button class="illustrate-result-close" title="Remove">&times;</button>
      </div>
      <img src="${escapeHtml(data.url)}" alt="Generated scene" class="illustrate-result-img" loading="lazy">
    `;
    imgContainer.querySelector('.illustrate-result-close')
      .addEventListener('click', () => imgContainer.remove());
    imgContainer.querySelector('.illustrate-result-img')
      .addEventListener('click', () => {
        if (window.openImageLightbox) {
          window.openImageLightbox(data.url);
        } else {
          window.open(data.url, '_blank');
        }
      });
    target.appendChild(imgContainer);
    if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;

    showToast('Scene image generated', 'success');
  } catch (err) {
    progress.stop();
    showToast(`Scene image failed: ${err.message}`, 'error');
  } finally {
    if (btn) btn.dataset.state = 'idle';
    if (btn2) btn2.classList.remove('loading');
  }
}


// ---------------------------------------------------------------------------
// Auto Background — 3-state button (off → config → armed → off)
// ---------------------------------------------------------------------------

let _autoBgDropdownsPopulated = false;

/**
 * Transition the auto-background button between states.
 * off    → greyed out, feature disabled
 * config → glowing, model dropdowns visible for selection
 * armed  → dropdowns hidden, feature active with breathing pulse
 */
export function _setAutoBgState(newState) {
  const btn = document.getElementById('auto-bg-btn');
  const config = document.getElementById('auto-bg-config');
  if (!btn) return;

  btn.dataset.state = newState;
  const s = getSettings();

  if (newState === 'off') {
    // Fully disable — hide dropdowns, sync to backend, clear current scene
    if (config) config.classList.add('hidden');
    s.narrativeAutoBackground = false;
    _syncAutoBgToBackend(s);
    btn.title = 'Auto scene backgrounds (off)';
    // Clear the current auto-generated background
    if (window.clearNarrativeBackground) window.clearNarrativeBackground();
  } else if (newState === 'config') {
    // Show dropdowns for model selection
    if (config) config.classList.remove('hidden');
    if (!_autoBgDropdownsPopulated) {
      _populateAutoBgDropdowns();
      _autoBgDropdownsPopulated = true;
    }
    btn.title = 'Choose models, then click again to activate';
  } else if (newState === 'armed') {
    // Hide dropdowns, capture selections, enable feature
    const distillerSel = document.getElementById('auto-bg-distiller');
    const imageSel = document.getElementById('auto-bg-image');
    if (distillerSel) s.narrativeAutoBgDistillerModel = distillerSel.value;
    if (imageSel) s.narrativeAutoBgImageModel = imageSel.value;
    if (config) config.classList.add('hidden');
    s.narrativeAutoBackground = true;
    _syncAutoBgToBackend(s);
    btn.title = 'Auto scene backgrounds (active) — click to disable';
    showToast('Auto backgrounds enabled', 'success');
  }

  // Persist to localStorage
  saveSettings();
}

async function _populateAutoBgDropdowns() {
  const distillerSel = document.getElementById('auto-bg-distiller');
  const imageSel = document.getElementById('auto-bg-image');
  const s = getSettings();

  // Populate LLM models
  if (distillerSel && distillerSel.options.length <= 1) {
    try {
      const models = (await getModels()).filter(m => !m.name.startsWith('g/'));
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.display_name || m.name;
        distillerSel.appendChild(opt);
      }
    } catch { /* ignore */ }
    distillerSel.value = s.narrativeAutoBgDistillerModel || '';
  }

  // Populate image models
  if (imageSel && imageSel.options.length <= 1) {
    try {
      const models = await getImageModels();
      if (models.length > 0) {
        const group = document.createElement('optgroup');
        group.label = 'Local';
        for (const m of models) {
          const opt = document.createElement('option');
          opt.value = m.name;
          opt.textContent = m.name;
          group.appendChild(opt);
        }
        imageSel.appendChild(group);
      }
    } catch { /* ignore */ }
    try {
      const cloudModels = await getCloudImageModels();
      if (cloudModels.length > 0) {
        const group = document.createElement('optgroup');
        group.label = 'Cloud';
        for (const m of cloudModels) {
          const opt = document.createElement('option');
          opt.value = m.name;
          opt.textContent = `${m.name} (${m.provider || 'cloud'})`;
          group.appendChild(opt);
        }
        imageSel.appendChild(group);
      }
    } catch { /* ignore */ }
    imageSel.value = s.narrativeAutoBgImageModel || '';
  }
}

async function _syncAutoBgToBackend(s) {
  try {
    await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        narrative_auto_background: s.narrativeAutoBackground,
        narrative_auto_background_interval: s.narrativeAutoBackgroundInterval || 4,
        narrative_auto_bg_distiller_model: s.narrativeAutoBgDistillerModel || '',
        narrative_auto_bg_image_model: s.narrativeAutoBgImageModel || '',
      }),
    });
  } catch { /* ignore */ }
}


// ---------------------------------------------------------------------------
// Passthrough Tools — wired via ./chat/toolbar/tools.js (Step 2 of the
// surface-owned composer migration). State lives on app.state.passthroughTools
// and is mirrored to localStorage by that module.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Public API — expose state and actions for other modules
// ---------------------------------------------------------------------------
/**
 * Dismiss full-screen overlays (library, browse, artifact studio).
 *
 * Only closes overlays that compete for the full viewport. Side panels
 * (image panel, inspector) are independent drawers — they float alongside
 * any overlay and are never force-closed by view switching.
 *
 * Accepts an optional `except` string to keep a specific overlay open
 * (e.g. dismissOverlays('browse') keeps browse while closing the rest).
 */
function dismissOverlays(except = '') {
  if (except !== 'browse') {
    // Browse can be mounted either as a ViewStack overlay (legacy
    // header-button path) or as a Surface tab (orb-drag). dismissOverlays
    // is for clearing OVERLAY-class panels — tab-mounted browse is its
    // own window and would be destroyed by closeBrowsePanel's tab branch.
    // Explicit close paths (X button, Esc inside browse, "discuss this
    // page") call closeBrowsePanel directly and are unaffected.
    const browseIsTab = SurfaceRegistry.ofType('browse').length > 0;
    if (!browseIsTab) closeBrowsePanel();
  }
  if (except !== 'files')   closeFiles();
  if (except !== 'library') {
    import('./library.js').then(m => m.closeLibrary()).catch(() => {});
  }
  if (except !== 'media') {
    const mediaOverlay = document.getElementById('media-overlay');
    if (mediaOverlay && !mediaOverlay.classList.contains('hidden')) {
      import('./media.js').then(m => m.closeMedia?.()).catch(() => {});
    }
  }
  if (except !== 'studio') {
    const studioOverlay = document.getElementById('studio-overlay');
    if (studioOverlay && !studioOverlay.classList.contains('hidden')) {
      import('./studio.js').then(m => { if (m.closeStudio) m.closeStudio(); }).catch(() => {});
    }
  }
}

// Cross-script event hook for overlay dismissal — non-module scripts
// (discovery.js, etc.) can't import dismissOverlays directly, but they
// can dispatch this event to navigate between overlays cleanly. Called
// by the Files panel's files:open-with-filter listener so a Discovery
// click that opens Files behind the still-visible Browse panel correctly
// closes Browse first instead of leaving the user staring at a stale
// overlay.
window.addEventListener('augmentum:dismiss-overlays', (e) => {
  const except = ((e.detail || {}).except || '').trim();
  dismissOverlays(except);
});

export { modeLabel, setMode };
export { setTextScale };

/** Public: ensure a typography preset's Google Fonts are loaded. */
export function ensureTypographyFonts(key) {
  _ensurePresetFonts(key);
}

export const app = {
  get state() { return state; },
  get dom() { return dom; },
  setMode,
  toggleTheme,
  togglePanel,
  openPanel,
  closePanel,
  toggleInspector,
  closeInspector,
  closeImagePanel,
  closeBrowsePanel,
  openInspectorMobile,
  closeInspectorMobile,
  switchInspectorSection,
  dismissOverlays,
  showToast,
  scrollToBottom,
  updateEmptyState,
  escapeHtml,
  enableDragScroll,
};

/**
 * Enable click-and-drag AND touch-drag horizontal scrolling on a container.
 * Call once per element — safe to call multiple times (idempotent).
 */
function enableDragScroll(el) {
  if (!el || el._dragScrollInit) return;
  el._dragScrollInit = true;
  let isDown = false, startX, scrollLeft;

  // --- Mouse (desktop) ---
  el.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    isDown = true;
    startX = e.pageX - el.offsetLeft;
    scrollLeft = el.scrollLeft;
    el.style.cursor = 'grabbing';
    el.style.userSelect = 'none';
  });
  el.addEventListener('mouseleave', () => { isDown = false; el.style.cursor = ''; el.style.userSelect = ''; });
  el.addEventListener('mouseup', () => { isDown = false; el.style.cursor = ''; el.style.userSelect = ''; });
  el.addEventListener('mousemove', (e) => {
    if (!isDown) return;
    e.preventDefault();
    el.scrollLeft = scrollLeft - (e.pageX - el.offsetLeft - startX);
  });

  // --- Touch (mobile) ---
  let touchStartX = 0, touchScrollLeft = 0, touchTracking = false;
  el.addEventListener('touchstart', (e) => {
    if (e.touches.length !== 1) return;
    touchStartX = e.touches[0].clientX;
    touchScrollLeft = el.scrollLeft;
    touchTracking = true;
  }, { passive: true });
  el.addEventListener('touchmove', (e) => {
    if (!touchTracking) return;
    const dx = e.touches[0].clientX - touchStartX;
    el.scrollLeft = touchScrollLeft - dx;
  }, { passive: true });
  el.addEventListener('touchend', () => { touchTracking = false; }, { passive: true });
}

// ---------------------------------------------------------------------------
// Surface bootstrap — restore saved workspace, then ensure a primary surface
// matching the current mode is present. Split out of init() because it's
// the one piece of startup that depends on the surface registry + layout
// manager being initialised AND on the chat module having already loaded
// sessions (so we can prune references to deleted sessions cleanly).
// ---------------------------------------------------------------------------
const _MODE_TO_SURFACE = {
  passthrough: 'chat',
  analytical: 'chat',
  narrative: 'narrative',
  coder: 'coder',
  agentic: 'chat',
};

async function _fetchSavedWorkspace() {
  try {
    const resp = await fetch('/api/config/ui');
    if (!resp.ok) return null;
    const data = await resp.json();
    if (!data?.workspace) return null;
    const parsed = typeof data.workspace === 'string'
      ? JSON.parse(data.workspace)
      : data.workspace;
    return (parsed && Array.isArray(parsed.surfaces)) ? parsed : null;
  } catch {
    return null;
  }
}

// Boot-lifecycle gate. A deferred promise that resolves only when
// _bootSurfaces finishes restoring saved workspace. Initialised at
// module load (not at init()) because orb-nav clicks can race the
// window between initOrbNav and `await _bootSurfaces()` — if
// waitForBootSurfaces resolved trivially during that window, orb-nav
// would create a default-config singleton that silently clobbers the
// saved one boot was about to restore. Eager init closes the window.
let _bootSurfacesComplete = false;
let _surfaceLayoutInitialized = false;
let _surfaceBootStarted = false;
let _resolveBootSurfaces;
const _bootSurfacesPromise = new Promise((resolve) => {
  _resolveBootSurfaces = resolve;
});
export function waitForBootSurfaces() {
  return _bootSurfacesPromise;
}
export function bootSurfacesComplete() {
  return _bootSurfacesComplete;
}

function _ensureSurfaceLayout() {
  if (_surfaceLayoutInitialized) return;
  _surfaceLayoutInitialized = true;
  LayoutManager.init();
}

function _mountPrimarySurfaceForCurrentMode() {
  const surfaceType = _MODE_TO_SURFACE[state.mode] || 'chat';
  const existing = (SurfaceRegistry.all?.() || []).find(
    s => s._isPrimary
      && s.constructor?.type === surfaceType
      && LayoutManager.hasContainer?.(s.id),
  );
  if (existing) return existing;
  if (!SurfaceRegistry.hasType(surfaceType)) return null;
  const primary = SurfaceRegistry.create(surfaceType, {
    analytical: state.mode === 'analytical',
    mode: state.mode,
    primary: true,
  });
  LayoutManager.mountSurface(primary);
  return primary;
}

function _mountInitialSurfaceShell() {
  _ensureSurfaceLayout();
  _mountPrimarySurfaceForCurrentMode();
}

async function _ensureSurfaceBoot() {
  if (_surfaceBootStarted) return _bootSurfacesPromise;
  _surfaceBootStarted = true;
  _ensureSurfaceLayout();
  await _bootSurfaces();
  return _bootSurfacesPromise;
}

async function _bootSurfaces() {
  try {
    // Fetch EVERYTHING we need before we touch the surface registry.
    // Doing both awaits up front removes the race window where an orb
    // click between "mount primary" and "restore extras" could create
    // a second singleton. After this Promise.all resolves, the rest
    // of the function is synchronous — no more yields.
    const [saved, sessionsModule] = await Promise.all([
      _fetchSavedWorkspace(),
      import('./chat/sessions.js'),
    ]);
    const { sessionStore } = sessionsModule;

    _bootSurfacesMountAll(saved, sessionStore);

    // Prune empty/stale sessions now that surfaces have claimed their
    // session ids. Doing this here (vs. inside sessionStore.load()) means
    // a non-focused tab's empty New-Chat session isn't pruned out from
    // under it (audit §7.5).
    const referencedIds = new Set(
      SurfaceRegistry.all()
        .map((s) => s._sessionId)
        .filter(Boolean),
    );
    sessionStore.pruneStaleEmpty(referencedIds);
  } finally {
    _bootSurfacesComplete = true;
    _resolveBootSurfaces();
  }
}

/**
 * Synchronous portion of surface boot. Called once from _bootSurfaces
 * after all I/O has resolved. Split out so every call here runs in one
 * microtask, eliminating the interleaving window where external
 * creators could race.
 */
function _bootSurfacesMountAll(saved, sessionStore) {
  // 1. Always create a fresh primary for the current mode. Primary adopts
  //    singleton DOM (#chat-scroll, .input-area) so it must be created
  //    first and must match the mode applyMode() rendered at boot.
  _mountPrimarySurfaceForCurrentMode();

  if (!saved) return;

  // 2. Restore extra (non-primary) surfaces from saved workspace. Each
  //    surface type that owns singleton DOM (coder/browse/image) is capped
  //    at one total — the primary may already be that type, in which case
  //    skip any saved copies. Drop surfaces whose pinned session is gone.
  const singletonTypes = new Set(['coder', 'browse', 'image']);
  let focusTargetId = null;

  for (const surfaceState of saved.surfaces) {
    if (!surfaceState?.type || !SurfaceRegistry.hasType(surfaceState.type)) continue;
    // Never re-create the primary — we already own the only one.
    if (surfaceState.primary) {
      if (surfaceState.id === saved.focused) focusTargetId = null; // primary will be focused as default
      continue;
    }
    // Enforce single-instance cap — skip a saved coder/browse/image if one
    // already exists (most likely the primary we just created).
    if (singletonTypes.has(surfaceState.type) && SurfaceRegistry.ofType(surfaceState.type).length > 0) {
      continue;
    }
    // Drop surfaces whose session was deleted while the user was away.
    if (surfaceState.sessionId && !sessionStore.get(surfaceState.sessionId)) continue;
    // Respect the 4-total cap.
    if (SurfaceRegistry.all().length >= 4) break;

    try {
      const surface = SurfaceRegistry.create(surfaceState.type, {
        ...(surfaceState.config || {}),
        id: surfaceState.id,
        sessionId: surfaceState.sessionId,
        mode: surfaceState.mode,
        characterId: surfaceState.characterId,
        characterName: surfaceState.characterName,
        url: surfaceState.url,
        pageTitle: surfaceState.pageTitle,
        workspaceId: surfaceState.workspaceId,
        workspaceName: surfaceState.workspaceName,
      });
      surface.restoreState(surfaceState);
      LayoutManager.mountSurface(surface);
      if (surface.id === saved.focused) focusTargetId = surface.id;
    } catch (err) {
      console.warn('Failed to restore surface:', surfaceState, err);
    }
  }

  // 3. Restore focus (if target survived) — otherwise the primary stays
  //    focused as a result of its mountSurface → focus() call.
  if (focusTargetId) SurfaceRegistry.focus(focusTargetId);
}

// ---------------------------------------------------------------------------
// XR web embed deep links
// ---------------------------------------------------------------------------
const _XR_EMBED_MODES = new Set(['passthrough', 'analytical', 'narrative', 'agentic', 'coder']);
const _XR_SURFACE_TO_MODE = Object.freeze({
  chat: 'passthrough',
  analytical: 'analytical',
  analyze: 'analytical',
  agentic: 'agentic',
  build: 'agentic',
  narrative: 'narrative',
  story: 'narrative',
  coder: 'coder',
});

function _isXrEmbedPage() {
  try {
    return new URLSearchParams(window.location.search).get('xrEmbed') === '1';
  } catch {
    return false;
  }
}

function _normalizeXrEmbedMode(mode) {
  const value = String(mode || '').trim();
  if (_XR_EMBED_MODES.has(value)) return value;
  if (value === 'chat') return 'passthrough';
  if (value === 'analyze') return 'analytical';
  if (value === 'story') return 'narrative';
  if (value === 'build') return 'agentic';
  return '';
}

function _applyXrEmbedChrome() {
  if (!_isXrEmbedPage()) return false;
  document.documentElement.dataset.xrEmbed = 'true';
  document.body?.classList.add('xr-embed-page');
  if (dom?.app) {
    dom.app.dataset.xrEmbed = 'true';
    dom.app.dataset.panel = 'hidden';
    dom.app.dataset.inspector = 'hidden';
  }
  return true;
}

function _switchXrBrowseTab(tab) {
  if (!tab) return;
  document.dispatchEvent(new CustomEvent('augmentum:switch-browse-tab', {
    detail: { tab },
  }));
  requestAnimationFrame(() => {
    document.querySelector(`.browse-tab-btn[data-tab="${CSS.escape(tab)}"]`)?.click();
  });
}

function _nextPaint() {
  return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

async function _openXrEmbedSurface(surface, params) {
  switch (surface) {
    case 'chat':
      setMode('passthrough');
      await _nextPaint();
      document.getElementById('chat-input')?.focus?.({ preventScroll: true });
      break;
    case 'analytical':
    case 'analyze':
      setMode('analytical');
      break;
    case 'agentic':
    case 'build':
      setMode('agentic');
      break;
    case 'narrative':
    case 'story':
      setMode('narrative');
      break;
    case 'coder':
      setMode('coder');
      break;
    case 'browse':
      openBrowsePanel({ skipAutoFocus: true });
      _switchXrBrowseTab('browse');
      break;
    case 'files':
      openFiles({ focusSearch: params.get('focus') === 'search' });
      break;
    case 'notes':
      openBrowsePanel({ skipAutoFocus: true });
      _switchXrBrowseTab('notes');
      break;
    case 'studio':
      if (document.getElementById('image-panel')?.classList.contains('hidden')) {
        document.getElementById('toggle-image-btn')?.click();
      }
      break;
    case 'media':
      await import('./youtube-panel.js').catch(() => {});
      window.dispatchEvent(new CustomEvent('media:open-panel', {
        detail: { tab: params.get('tab') || 'discover' },
      }));
      break;
    case 'devices': {
      await _openConnectedDevices();
      break;
    }
    case 'games': {
      const mod = await import('./library.js').catch(() => null);
      // Land directly on the Games type group in the three-pane sidebar.
      // The sidebar uses Title-Cased label ids ("Games"); the open-by-
      // selection contract is satisfied by passing { kind, id }.
      await mod?.openLibrary?.({ initialSelection: { kind: 'type', id: 'Games' } });
      break;
    }
  }
}

async function _openConnectedDevices() {
  dismissOverlays('devices');
  const mod = await import('./media-servers.js').catch((err) => {
    console.warn('[connected-devices] open failed:', err);
    return null;
  });
  await mod?.openMediaServers?.();
}

async function _handleXrEmbedDeepLink() {
  if (!_applyXrEmbedChrome()) return;
  const params = new URLSearchParams(window.location.search);
  const requestedMode = _normalizeXrEmbedMode(params.get('mode'));
  const requestedSurface = String(params.get('surface') || params.get('xrSurface') || '').trim();
  const mode = requestedMode || _XR_SURFACE_TO_MODE[requestedSurface] || '';

  if (mode) setMode(mode);
  await _nextPaint();
  if (requestedSurface) await _openXrEmbedSurface(requestedSurface, params);
}

async function _handleSurfaceDeepLink() {
  const params = new URLSearchParams(window.location.search);
  if (params.get('xrEmbed') === '1') return;
  const requestedSurface = String(params.get('surface') || '').trim();
  if (requestedSurface === 'devices') {
    await _openConnectedDevices();
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
// Boot transition (see #app-boot in index.html). setBootStage updates the
// status line during the launch wait; dismissBoot fades + removes the overlay
// once the shell is interactive (and cancels the inline safety timeout).
function setBootStage(text) {
  const el = document.getElementById('app-boot-stage');
  if (el) el.textContent = text;
}
// If the tab was open *while the server was still booting*, some of the ~40
// stylesheet / JS-module requests fired at page load can race the upstream's
// cold-start window and get handed Caddy's `service-starting.html` holding page
// (text/html) in place of the real asset (see handle_errors 502/503 in
// caddy_front_door.py). The browser silently drops the mis-typed stylesheet and
// never retries it, so revealing the shell exposes an UNSTYLED app (orbs on
// white). This is asset-agnostic — any CSS/JS that raced the window is affected
// — so we don't point-patch one file: at the single reveal choke point we check
// whether the theme CSS actually applied and, if not, force exactly ONE reload
// (which now happens fully after the server is ready). sessionStorage guards
// against a reload loop if something else is wrong.
function themeCssLoaded() {
  // --panel-width is set on :root by styles/variables.css. If it resolves, the
  // core theme stylesheets parsed; if it's empty they were dropped mid-boot.
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue('--panel-width').trim();
  return v.length > 0;
}
function dismissBoot() {
  // Boot-race self-heal: reveal would expose an unstyled shell — reload once.
  if (!themeCssLoaded() && !sessionStorage.getItem('__bootCssReloaded')) {
    sessionStorage.setItem('__bootCssReloaded', '1');
    location.reload();
    return;
  }
  // Reached a good paint — clear the one-shot guard for future cold starts.
  sessionStorage.removeItem('__bootCssReloaded');
  if (window.__bootSafety) { clearTimeout(window.__bootSafety); window.__bootSafety = null; }
  const b = document.getElementById('app-boot');
  if (!b || b.classList.contains('app-boot--hide')) return;
  b.classList.add('app-boot--hide');
  setTimeout(() => b.remove(), 500);
}

async function init() {
  _applyXrEmbedChrome();
  setBootStage('Connecting…');

  // Auth gate — must authenticate before loading app
  const authStatus = await checkStatus();
  if (authStatus.server_unreachable || authStatus.db_error
      || authStatus.setup_required || !authStatus.authenticated) {
    // Any path that shows its own full-screen UI (error card, setup wizard,
    // login) must clear the boot overlay first so it doesn't sit on top.
    dismissBoot();
  }
  if (authStatus.server_unreachable) {
    // Several attempts elapsed with no response from /api/auth/status —
    // server is either down, still cold-starting, or unreachable from
    // this host. Show a clear overlay with a manual retry instead of a
    // frozen blank screen.
    const overlay = document.createElement('div');
    overlay.className = 'auth-overlay';
    overlay.innerHTML = `
      <div class="auth-card">
        <h2 class="auth-title" style="color:var(--color-error, #ef4444)">⚠ Server Unreachable</h2>
        <p class="auth-subtitle">Could not reach Augmentum after several attempts.</p>
        <p class="auth-hint" style="margin-top:1rem;opacity:0.7">
          Common fixes:<br>
          • Confirm the server is running (<code>docker ps</code>)<br>
          • If it just started, give it a moment and retry<br>
          • Check the URL — are you on the right host/port?<br>
          • Network/firewall blocking the connection?
        </p>
        <button class="auth-btn" onclick="location.reload()" style="margin-top:1.5rem">Retry</button>
      </div>
    `;
    document.body.appendChild(overlay);
    return;
  }
  if (authStatus.db_error) {
    // Database unavailable — show error overlay instead of broken setup wizard
    const overlay = document.createElement('div');
    overlay.className = 'auth-overlay';
    overlay.innerHTML = `
      <div class="auth-card">
        <h2 class="auth-title" style="color:var(--color-error, #ef4444)">⚠ Database Unavailable</h2>
        <p class="auth-subtitle">${authStatus.error || 'The server is running in memory-only mode. Data cannot be saved.'}</p>
        <p class="auth-hint" style="margin-top:1rem;opacity:0.7">
          Common fixes:<br>
          • Check Docker volume permissions on <code>/data</code><br>
          • Ensure <code>augmentum_data</code> volume is writable<br>
          • Check server logs for <code>sqlite_connect_failed</code><br>
          • Restart the container after fixing permissions
        </p>
        <button class="auth-btn" onclick="location.reload()" style="margin-top:1.5rem">Retry</button>
      </div>
    `;
    document.body.appendChild(overlay);
    return;
  }
  if (authStatus.setup_required) {
    await showSetupWizard();
    location.reload();
    return;
  }
  if (!authStatus.authenticated) {
    await showLogin();
    location.reload();
    return;
  }
  state.user = getCurrentUser();
  setBootStage('Loading your space…');
  // Tag <body> with role-admin/role-user so CSS can hide admin-only UI
  // from non-admin tenants (see .admin-only in components.css).
  applyRoleBodyClass();
  hydrateArmedDevice();

  cacheDom();
  _applyXrEmbedChrome();
  initTheme();
  initTypography();
  initMode();
  initInspector();
  initInspectorResize();
  initLeftPanelResize();
  initTabs();
  initInput();
  initAttach();
  _initQuickInsert();
  initDocuments();
  refreshDocContextBar();

  // Empty state suggested prompts — click to populate input
  document.getElementById('empty-state-chips')?.addEventListener('click', (e) => {
    const chip = e.target.closest('.empty-state-chip');
    if (!chip || !chip.dataset.prompt) return;
    dom.chatInput.value = chip.dataset.prompt;
    dom.chatInput.focus();
    dom.chatInput.dispatchEvent(new Event('input')); // trigger auto-resize
  });

  // Library overlay — three-pane orchestrator handles its own lifecycle
  // (ViewStack push, localStorage persistence, refresh recovery).
  dom.libraryOpenBtn?.addEventListener('click', () => {
    dismissOverlays('library');
    import('./library.js').then(m => m.openLibrary());
  });

  // Media console — comfort-first parallel build. Retires Library on parity.
  dom.mediaOpenBtn?.addEventListener('click', () => {
    dismissOverlays('media');
    import('./media.js').then(m => m.openMedia());
  });
  // Workshop — the self-improvement surface (own overlay lifecycle).
  dom.workshopOpenBtn?.addEventListener('click', () => {
    dismissOverlays('workshop');
    import('./workshop.js').then(m => m.openWorkshop());
  });
  dom.devicesOpenBtn?.addEventListener('click', () => {
    _openConnectedDevices();
  });
  dom.discoverOpenBtn?.addEventListener('click', () => {
    dismissOverlays('discover');
    import('./discover/index.js').then(m => m.openDiscover()).catch((e) => {
      console.warn('[discover] open failed:', e);
    });
  });
  initKeyboard();
  initClickOutside();
  restorePanelState();
  // Swipe-to-close removed — tap-on-backdrop is more intentional and
  // the gesture handlers were swallowing taps on buttons inside panels.

  // Wire header buttons
  dom.menuBtn.addEventListener('click', togglePanel);
  // Theme and typography now handled by The Grove (grove.js)
  // dom.themeBtn.addEventListener('click', toggleTheme);
  dom.inspectorToggle.addEventListener('click', toggleInspector);
  dom.closeInspector.addEventListener('click', () => {
    // On desktop, toggle inspector. On mobile, close overlay.
    if (dom.inspectorPanel.classList.contains('mobile-open')) {
      closeInspectorMobile();
    } else {
      toggleInspector();
    }
  });
  dom.panelBackdrop.addEventListener('click', () => {
    closePanel();
    closeInspectorMobile();
  });

  // Agentic inspector close button
  const closeAgenticBtn = $('close-inspector-btn-agentic');
  if (closeAgenticBtn) {
    closeAgenticBtn.addEventListener('click', () => {
      if (dom.inspectorPanel.classList.contains('mobile-open')) {
        closeInspectorMobile();
      } else {
        toggleInspector();
      }
    });
  }

  // Card view inspector close button
  const closeCardBtn = $('close-inspector-btn-card');
  if (closeCardBtn) {
    closeCardBtn.addEventListener('click', () => {
      if (dom.inspectorPanel.classList.contains('mobile-open')) {
        closeInspectorMobile();
      } else {
        toggleInspector();
      }
    });
  }

  // Passthrough inspector close button
  const closePassthroughBtn = $('close-inspector-btn-passthrough');
  if (closePassthroughBtn) {
    closePassthroughBtn.addEventListener('click', () => {
      if (dom.inspectorPanel.classList.contains('mobile-open')) {
        closeInspectorMobile();
      } else {
        toggleInspector();
      }
    });
  }

  // Coder inspector close button
  const closeCoderBtn = $('close-inspector-btn-coder');
  if (closeCoderBtn) {
    closeCoderBtn.addEventListener('click', () => {
      if (dom.inspectorPanel.classList.contains('mobile-open')) {
        closeInspectorMobile();
      } else {
        toggleInspector();
      }
    });
  }

  // Orb navigation (replaces old mobile bottom bar)
  initOrbNav({
    setMode,
    // Let orb-nav defer singleton creation until _bootSurfaces finishes
    // restoring saved workspace. Prevents the fast-click-during-boot
    // race where orb-nav would create a default-config coder/browse/
    // image surface that then blocks the saved one from being restored.
    waitForBoot: waitForBootSurfaces,
    openBrowse: () => { document.getElementById('toggle-browse-btn')?.click(); },
    openNotes: () => {
      document.getElementById('toggle-browse-btn')?.click();
      setTimeout(() => document.getElementById('browse-tab-notes')?.click(), 100);
    },
    openDiscovery: () => {
      document.getElementById('toggle-browse-btn')?.click();
      setTimeout(() => document.getElementById('browse-tab-discovery')?.click(), 100);
    },
    closeBrowse: () => closeBrowsePanel(),
    closeNotes: () => closeBrowsePanel(),
    closeDiscovery: () => closeBrowsePanel(),
    initialMode: state.mode,
  });

  // Mobile header overflow menu
  _initHeaderOverflow();

  // Mode option clicks (in sidebar)
  // Normal click = switch mode. Ctrl+click (or Cmd+click) = open alongside as new surface.
  if (dom.modeSelector) {
    dom.modeSelector.querySelectorAll('.panel-mode-option').forEach(opt => {
      opt.addEventListener('click', (e) => {
        if (e.ctrlKey || e.metaKey) {
          // Open alongside — create new surface without switching mode.
          // Routed through the canonical ladder in orb-nav so this path
          // gets the same cap toast, singleton-focus toast, boot-restore
          // race guard, and session inheritance as the orb long-press
          // gesture — this handler used to be a drifted copy that failed
          // silently at the 4-tab cap (audit §4.6 class).
          e.preventDefault();
          openSurfaceAlongside({ orbId: opt.dataset.mode }).then((res) => {
            if (res.ok) {
              showToast(`Opened ${opt.querySelector('.mode-option-name')?.textContent || opt.dataset.mode} alongside`, 'info', 2000);
            }
          });
        } else {
          setMode(opt.dataset.mode);
        }
      });
    });
  }

  // Scene image button (header — narrative mode with popover)
  const sceneGenBtn = $('scene-gen-btn');
  const sceneGenPop = $('scene-gen-popover');
  const sceneGenGo = $('scene-gen-go');

  if (sceneGenBtn && sceneGenPop) {
    // Toggle popover on button click
    sceneGenBtn.addEventListener('click', () => {
      const isOpen = sceneGenBtn.dataset.state === 'open';
      if (isOpen) {
        sceneGenPop.classList.add('hidden');
        sceneGenBtn.dataset.state = 'idle';
      } else {
        sceneGenPop.classList.remove('hidden');
        sceneGenBtn.dataset.state = 'open';
        _populateSceneGenModels();
      }
    });

    // Close popover on outside click
    document.addEventListener('click', (e) => {
      const wrap = document.getElementById('scene-gen-wrap');
      if (wrap && !wrap.contains(e.target) && sceneGenBtn.dataset.state === 'open') {
        sceneGenPop.classList.add('hidden');
        sceneGenBtn.dataset.state = 'idle';
      }
    });
  }

  // Generate button inside popover
  if (sceneGenGo) {
    sceneGenGo.addEventListener('click', () => _fireSceneGenerate());
  }

  // Narrative-themed buttons — wired via per-control modules
  // (Step 2 of the surface-owned composer migration).
  const _toolbarRoot = document.getElementById('input-toolbar');
  wireAutoBg(_toolbarRoot, null);
  wireInstantScene(_toolbarRoot, null);
  wireReadingRoom(_toolbarRoot, null);
  wireNarrativeBubbles(_toolbarRoot, null);

  // Background rotation button — wired via ./chat/toolbar/bg-rotation.js (Step 2).
  wireBgRotation(_toolbarRoot, null);

  // Auto-search button (analytical mode) — wired via ./chat/toolbar/auto-search.js
  // (Step 2 of the surface-owned composer migration).
  wireAutoSearch(_toolbarRoot, null);

  // TTS auto-read — wired via ./chat/toolbar/auto-read.js (Step 2).
  wireAutoRead(_toolbarRoot, null);

  // Per-chat sampling ("Tuning for this chat") — wired via
  // ./chat/toolbar/tuning.js. Sets session.chatSampling, merged into the
  // request by chat/stream.js as the per-call layer over the per-model profile.
  wireChatTuning(_toolbarRoot);

  // Warm TTS on page load if auto-read is already enabled
  { const _s = getSettings(); if (_s.voiceAutoRead) chat.ttsChatWarmup(); }

  // Passthrough tools toggle — wired via ./chat/toolbar/tools.js (Step 2).
  wireTools(_toolbarRoot, null);

  // Web search popover — wired via ./chat/toolbar/web-search.js (Step 2).
  // MUST be wired BEFORE wireOverflow. On mobile, wireOverflow's deferred
  // relayout() re-parents #web-search-wrap out of the toolbar into the
  // body-mounted overflow popover. That relayout is a setTimeout(0) macrotask
  // which fires during the first `await` further down in init — so if web
  // search were wired after those awaits (where it used to live), its
  // toolbarEl.querySelector('#web-search-btn') would miss the already-moved
  // button and never attach a handler (the "browse web does nothing in the ⋯
  // menu" bug). Wiring it here, synchronously before wireOverflow, guarantees
  // the handler is on the button before it can move. Re-parenting preserves
  // handlers, and window._attachWebPage is only read at click-time, so this is
  // safe to do before the rest of init runs.
  wireWebSearch(_toolbarRoot, null);

  // Mobile overflow ("More") menu — collapses the current mode's secondary
  // buttons into a labeled popover on phones. Wired last so its placeholder
  // pass sees the final toolbar order. Desktop is a no-op.
  wireOverflow(_toolbarRoot);

  // Initialize sub-modules
  initSettings();
  await loadPersonalizationFromServer();  // restore server-persisted personalization
  await loadVoicePrefsFromServer();       // restore server-persisted voice preferences
  await fetchCapabilities();   // fetch before initModels so it can check backends
  await initChat();
  _mountInitialSurfaceShell();
  initAnalytical();
  initAgentic();
  initFlowBar();
  // Apply initial mode to flow bar (initMode runs before flow bar exists)
  if (state.mode === 'analytical' || state.mode === 'agentic') showFlowBar(state.mode);
  await initNarrative();
  initImage();
  initBrowse();
  // Ambient per-turn stats strip above the composer: ttft / tok/s / gen
  // tokens as text, plus a context-window fill meter along its top edge
  // (folds in the retired ctx-bar pill). Replaces the per-bubble inline
  // stats line that overflowed on narrow viewports. Subscribes to
  // augmentum:turn-stats.
  const { initStatsBar } = await import('./chat/stats-bar.js');
  initStatsBar();
  // Universal mouse-wheel → horizontal scroll for tab strips, chip
  // rails, model pickers, etc. Document-level delegation; existing
  // per-surface wheel handlers (grove chips, settings nav) win via
  // defaultPrevented so no double-scroll.
  const { initHorizontalScroll } = await import('./utils/h-scroll.js');
  initHorizontalScroll();
  const browseBtn = document.getElementById('toggle-browse-btn');
  if (browseBtn) browseBtn.addEventListener('click', () => {
    dismissOverlays('browse');
    openBrowsePanel();
  });
  initFiles();
  dom.filesBtn?.addEventListener('click', () => {
    dismissOverlays('files');
    toggleFiles();
  });
  // Persistent mini-player (bottom-docked) — survives navigation so
  // an audiobook keeps playing while the user browses chat / notes /
  // Grove. Dynamic import so the module's cost is only paid for users
  // who connect a media server.
  import('./media-mini-player.js').then(m => m.initMediaMiniPlayer())
    .catch(err => console.error('[media-player] init failed:', err));
  import('./tts-mini-player.js').then(m => m.initTtsMiniPlayer())
    .catch(err => console.error('[tts-mini-player] init failed:', err));
  import('./sticky-note.js').then(m => m.initStickyNotes())
    .catch(err => console.error('[sticky-note] init failed:', err));
  import('./media-keyboard.js').then(m => m.initMediaKeyboard())
    .catch(err => console.error('[media-keyboard] init failed:', err));
  // Resume toast for the audiobook / video pair (parity with the Grove
  // radio + ambient YT resume). Fires once per page load when there's a
  // fresh last-played entry and the user hasn't permanently dismissed.
  // Slight delay so the toast isn't competing with first-paint chrome.
  setTimeout(() => {
    import('./media-resume.js').then(m => m.offerMediaResume())
      .catch(err => console.error('[media-resume] init failed:', err));
  }, 1200);
  // Cast shelf — floating cast affordance + active-cast transport.
  // Lazy import so single-machine non-cast users don't pay the boot
  // cost. Folds in what used to be a separate cast-remote pill (now
  // deleted) — active casts get inline transport controls per row.
  import('./cast-shelf.js').then(m => m.initCastShelf())
    .catch(err => console.error('[cast-shelf] init failed:', err));
  // Paused-comic dock — same shell aesthetic as the audio mini-player,
  // materialises only when the reader's Minimize is used.
  import('./comic-mini-player.js').then(m => m.initComicMiniPlayer())
    .catch(err => console.error('[comic-mini-player] init failed:', err));
  initModels();
  initResources();
  initVoice();
  Coder.init().catch(err => console.error('[Coder] init failed:', err));
  initXrSurfaceBridge({
    app,
    openBrowsePanel,
    openFiles,
    Coder,
  });

  // Pre-fetch models and voices into centralized cache
  warmupModelCache();

  // Fetch models from backend (via settings module)
  settingsFetchModels();

  // Recover overlay state if user refreshed while a panel was open
  try {
    const wsState = localStorage.getItem('augmentum_workspace');
    const libState = localStorage.getItem('augmentum_library_open');
    const browseState = localStorage.getItem('augmentum_browse_open');
    if (wsState) {
      import('./workspace.js').then(m => m.recoverWorkspace()).catch(() => {});
    } else if (libState) {
      import('./library.js').then(m => m.openLibrary()).catch(() => {});
    } else if (browseState) {
      openBrowsePanel();
      const savedTab = localStorage.getItem('augmentum_browse_tab');
      if (savedTab === 'notes') {
        // switchTab is internal to browse.js — trigger via tab button click
        setTimeout(() => document.querySelector('.browse-tab-btn[data-tab="notes"]')?.click(), 100);
      }
    }
  } catch { /* ignore */ }

  // Horizontal scroll-wheel support for hidden-scrollbar tab bars
  // Uses delegation so dynamically-created elements (model dropdown tabs) are covered
  document.addEventListener('wheel', (e) => {
    const el = e.target.closest('.panel-mode-selector, .model-dropdown-tabs, .recent-chats-strip, .files-chips');
    if (!el || el.scrollWidth <= el.clientWidth) return;
    e.preventDefault();
    el.scrollLeft += e.deltaY;
  }, { passive: false });

  // Desktop drag-scroll for horizontal strips
  document.querySelectorAll('.panel-mode-selector, .recent-chats-strip').forEach(enableDragScroll);

  // Connection status monitoring
  const connBadge = document.getElementById('connection-badge');
  const updateConn = (online) => {
    if (connBadge) connBadge.classList.toggle('hidden', online);
    if (!online) showToast('Connection lost — changes may not save', 'warning', 0);
    else {
      // Dismiss any persistent offline toast
      document.querySelectorAll('.toast.warning').forEach(t => {
        if (t.querySelector('.toast-message')?.textContent.includes('Connection lost')) {
          dismissToast(t);
        }
      });
    }
  };
  window.addEventListener('offline', () => updateConn(false));
  window.addEventListener('online', () => {
    updateConn(true);
    showToast('Connection restored', 'success');
  });
  if (!navigator.onLine) updateConn(false);

  // Browse → Chat bridge: attach web content to chat input
  document.addEventListener('augmentum:web-to-chat', (e) => {
    const { docId, filename, chunkCount, title, url } = e.detail;
    if (!docId) return;

    // Bind document to current session
    const sessionId = state.currentSessionId;
    if (sessionId) {
      fetch(`/api/documents/session/${encodeURIComponent(sessionId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_id: docId, inject_mode: 'full' }),
      }).catch(() => {});
    }

    // Add to pending documents with web metadata
    pendingDocuments.push({
      id: docId,
      filename: filename,
      chunk_count: chunkCount,
      _webTitle: title,
      _webUrl: url,
    });
    renderAttachPreviews();

    // Focus chat input with contextual placeholder
    dom.chatInput.focus();
    dom.chatInput.placeholder = `Ask about "${title.slice(0, 40)}${title.length > 40 ? '...' : ''}"...`;

    // Refresh document context bar
    refreshDocContextBar();
  });

  // Shared function: fetch a web page and attach to chat. Exposed as
  // window._attachWebPage below; the web-search popover (wired earlier, before
  // wireOverflow) reads it lazily at click-time, so defining it here is fine.
  async function _attachWebPage(url, title) {
    try {
      // Fetch the page content
      const fetchResp = await fetch(`/api/browse/fetch?url=${encodeURIComponent(url)}`);
      const page = await fetchResp.json();
      const content = page.text || '';

      if (!content || content.length < 100) {
        showToast('Page has no extractable content', 'warning');
        return;
      }

      // Save to document store
      const saveResp = await fetch('/api/browse/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, title: title || page.title || url, content }),
      });

      if (!saveResp.ok) {
        showToast('Failed to save page', 'error');
        return;
      }

      const data = await saveResp.json();

      // Bind to session
      const sessionId = state.currentSessionId;
      if (sessionId) {
        await fetch(`/api/documents/session/${encodeURIComponent(sessionId)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ document_id: data.id, inject_mode: 'full' }),
        }).catch(() => {});
      }

      // Add as web attachment
      pendingDocuments.push({
        id: data.id,
        filename: data.filename || `${title || 'page'}.txt`,
        chunk_count: data.chunk_count || 0,
        _webTitle: title || page.title || url,
        _webUrl: url,
      });
      renderAttachPreviews();
      refreshDocContextBar();
      dom.chatInput.focus();

      const displayTitle = title || page.title || url;
      dom.chatInput.placeholder = `Ask about "${displayTitle.slice(0, 40)}${displayTitle.length > 40 ? '...' : ''}"...`;
      showToast(`"${displayTitle.slice(0, 30)}..." attached`, 'success');
    } catch (err) {
      showToast(`Failed: ${err.message}`, 'error');
    }
  }

  // Make _attachWebPage available for the Discuss bridge too
  window._attachWebPage = _attachWebPage;

  // Make app globally accessible for debugging
  window.__augmentum = app;

  // --- Surface System Bootstrap ---
  await _ensureSurfaceBoot();

  // --- Workspace flush on page teardown ---
  // saveWorkspace runs on every surface open/close/focus change, but a user
  // who drags a tab and immediately closes the browser can lose that last
  // write. pagehide is the reliable cross-browser unload signal (fires
  // before beforeunload on mobile + bfcache). Using keepalive fetch so the
  // request survives teardown.
  window.addEventListener('pagehide', () => {
    SurfaceRegistry.flushWorkspace();
  });

  // --- Flows + Voice Commands + Command Composer ---
  initFlows();
  initVoiceCommands();
  CommandComposer.init();

  // --- Notification substrate client ---
  // No-op when notificationsEnabled is off. Opens a WS to
  // /api/notify/subscribe + fetches initial backlog so missed
  // notifications (e.g. incoming calls during disconnect) surface
  // on reload. See ui/scripts/notifications.js for the loop.
  initNotifications().catch((e) => {
    console.warn('[notifications] init failed', e);
  });

  // Wire the SW postMessage → window CustomEvent bridge so that
  // when the user clicks an OS-level push notification, the focused
  // tab can react (route to the right panel, etc.). Zero permission
  // required — this only installs a message listener. The actual
  // subscribe-to-push flow lives behind the Settings toggle.
  import('./notifications/push-subscribe.js').then(({ installClickListener }) => {
    try { installClickListener(); } catch (e) {
      console.warn('[notifications] click-listener install failed', e);
    }
  }).catch((e) => {
    console.warn('[notifications] push-subscribe module load failed', e);
  });

  // --- Connect (peer-to-peer call) UI ---
  // No-op when connectEnabled is off. Registers the command-palette
  // entry "Connect: Place a call" and exposes window.augmentumConnect
  // for debug. See ui/scripts/connect/.
  try {
    initConnectUI();
  } catch (e) {
    console.warn('[connect] ui init failed', e);
  }

  // Federated-PBX trust surface ("Connect: Federation"): contact cards,
  // the verification ceremony, the verified/unverified trust chips
  // (D1-01), and the knock inbox. No-op visually until opened.
  try {
    initFederation();
  } catch (e) {
    console.warn('[connect] federation ui init failed', e);
  }

  // --- Connect signaling WS (eager) ---
  // Open the signaling WS at boot when Connect is enabled, so this
  // tab can RECEIVE incoming calls/text without first initiating one
  // itself. Without this the WS is lazy and the inbox is silent —
  // the caller side gets routed=0 and the recipient never rings.
  // Fire-and-forget: the client owns its own reconnect loop; we just
  // need to kick it off. Errors here are not fatal to boot.
  try {
    const _s = getSettings();
    if (_s && _s.connectEnabled !== false) {
      ensureConnectSignaling().catch((e) => {
        console.warn('[connect] signaling autostart failed', e);
      });
    }
  } catch (e) {
    console.warn('[connect] signaling autostart guard failed', e);
  }

  // --- Connect text-messaging panel ---
  // No-op when connectEnabled is off. Registers "Connect: Open messages"
  // in the command palette, listens for incoming WS text events, and
  // wires the connect.message notification "Open" action.
  try {
    initConnectMessagingUI();
  } catch (e) {
    console.warn('[connect] messaging ui init failed', e);
  }

  // --- Connect call-history panel ---
  // No-op when connectEnabled is off. Registers "Connect: Open call
  // history" in the command palette. Reads /api/connect/calls and
  // /api/connect/calls/{id}, posts quality ratings via /rate.
  try {
    initConnectCallsUI();
  } catch (e) {
    console.warn('[connect] calls ui init failed', e);
  }

  // Guest management — "Connect: Manage guests" in the command palette.
  // Lists durable guest passes with per-guest scope toggle + revoke kill-switch.
  try {
    initConnectGuestsUI();
  } catch (e) {
    console.warn('[connect] guests ui init failed', e);
  }

  // --- Surface cleanup on session delete ---
  // When a session is deleted, any non-primary surface pinned to it becomes
  // a dead tab — its renderer would show nothing on re-activation and its
  // tab title + close button would still offer to re-focus an orphaned id.
  // Unmount proactively so the tab bar never has phantom entries. Primary
  // is left alone (it tracks the global active session via its listener).
  document.addEventListener('augmentum:session-deleted', (e) => {
    const deletedId = e.detail?.sessionId;
    if (!deletedId) return;
    for (const s of SurfaceRegistry.all()) {
      if (s._isPrimary) continue;
      if (s._sessionId === deletedId) {
        LayoutManager.unmountSurface(s.id);
        SurfaceRegistry.destroy(s.id);
      }
    }
    SurfaceRegistry.saveWorkspace();
  });

  // --- Surface-focus → app-state sync ---
  // The surface registry is the single source of truth for "what the user is
  // looking at." When focus changes (tab click, drag-to-open, programmatic
  // focus), the surface's mode becomes the active mode.
  //
  // Audit §1 — the two historical mode-change paths (legacy setMode vs.
  // surface focus) were asymmetric: setMode went through ViewStack
  // (overlay-pop, primary-swap, applyMode), focus-changed wrote state.mode
  // directly and skipped ViewStack entirely. _baseMode silent-drifted from
  // state.mode whenever the user switched tabs. Phase 1 unification routes
  // both paths through ViewStack.setBaseMode, with `fromFocus: true`
  // marking the focus-driven case so it skips overlay-pop and primary-swap
  // (focus on a non-primary tab must not dismiss voice or destroy the
  // existing primary). See ViewStack.setBaseMode docstring for the matrix.
  const _VALID_MODES = ['passthrough', 'analytical', 'narrative', 'agentic', 'coder'];
  document.addEventListener('surface:focus-changed', (e) => {
    const { mode } = e.detail;
    if (!mode) return;

    if (_VALID_MODES.includes(mode) && state.mode !== mode) {
      // Unified mode-change path. ViewStack runs the registered applyMode
      // wrapper (initMode → state.mode + localStorage + applyMode()) for us
      // and emits viewstack:mode-changed; we still emit the legacy
      // augmentum:mode-changed so chat/narrative/coder/flow-bar listeners
      // that pre-date ViewStack continue to fire.
      ViewStack.setBaseMode(mode, { fromFocus: true, reason: 'surface-focus' });
      document.dispatchEvent(new CustomEvent('augmentum:mode-changed', { detail: { mode } }));
      return;
    }

    // Same mode, different surface instance (e.g. two chat tabs). applyMode
    // would be a no-op, but the inspector and left panel visibility toggles
    // still need to be correct. Panel CONTENT refresh happens via
    // augmentum:session-changed, which surface.activate() fires.
    const isPassthrough = mode === 'passthrough';
    const isAnalytical = mode === 'analytical';
    const isNarrative = mode === 'narrative';
    const isAgentic = mode === 'agentic';
    const isCoder = mode === 'coder';
    if (dom.passthroughView) dom.passthroughView.classList.toggle('hidden', !isPassthrough);
    if (dom.reasoningView) dom.reasoningView.classList.toggle('hidden', !isAnalytical);
    if (dom.cardView) dom.cardView.classList.toggle('hidden', !isNarrative);
    if (dom.taskView) dom.taskView.classList.toggle('hidden', !isAgentic);
    if (dom.sessionsView) dom.sessionsView.classList.toggle('hidden', isNarrative || isCoder);
    if (dom.charactersView) dom.charactersView.classList.toggle('hidden', !isNarrative);
    const coderFilesView = document.getElementById('coder-files-view');
    if (coderFilesView) coderFilesView.classList.toggle('hidden', !isCoder);
  });

  // PWA manifest shortcuts — jump-list entries append `?shortcut=<key>`
  // to the start URL. We handle them after all panels/modes are wired
  // so the open-functions are live, then strip the param via
  // history.replaceState so a refresh doesn't re-trigger the shortcut
  // and the URL stays clean for sharing. Same scaffolding will serve
  // future deep-links (share targets, etc.).
  await _handleXrEmbedDeepLink();
  await _handleSurfaceDeepLink();

  try {
    const params = new URLSearchParams(window.location.search);
    const shortcut = params.get('shortcut');
    if (shortcut) {
      params.delete('shortcut');
      const qs = params.toString();
      const cleanUrl = window.location.pathname + (qs ? `?${qs}` : '') + window.location.hash;
      window.history.replaceState(null, '', cleanUrl);
      switch (shortcut) {
        case 'new-chat':
          setMode('passthrough');
          chat.createSession();
          document.getElementById('chat-input')?.focus();
          break;
        case 'files':
          openFiles();
          break;
        case 'browse':
          openBrowsePanel();
          break;
        case 'coder':
          setMode('coder');
          break;
        case 'voice-vr':
          setMode('passthrough');
          voice.openVrEntry?.();
          break;
      }
    }
  } catch (err) {
    console.warn('[shortcut] deep-link handling failed:', err);
  }

  // First-run check (lazy-loaded). The living-avatar greeting decides whether
  // to run the 3D-avatar welcome or delegate to the classic onboarding card.
  import('./first-run-avatar.js').then(m => m.checkFirstRun()).catch(() => {});

  // Shell is wired and interactive — fade the boot overlay once the first
  // post-init frame has painted, so the user sees the real UI underneath
  // rather than a hard cut. Surface-specific waits beyond this point (coder
  // workspace bring-up, chat model load) carry their own inline progress.
  requestAnimationFrame(() => requestAnimationFrame(dismissBoot));
}

// Boot when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
