/* ==========================================================================
   Studio Tool Palette — host module
   --------------------------------------------------------------------------
   Right-rail container managing 5 tool tabs (Image / Design / Layout / AI /
   Structure). Tools register via registerTool({id, icon, label, mount,
   unmount, isAvailable}); the palette draws the icon column + active drawer
   and dispatches focus events to the active tool. Per-artifact-type
   collapsed/expanded state lives in localStorage.

   Backend contract: tools receive a `ctx` object with {artifactId, source,
   focusSlot, onSlotChange(value), getFocusSlot(), api{search, generate,
   commit, discard}}. Tools don't talk to the rest of Studio directly — all
   editor mutations route through ctx.onSlotChange().

   Phase 1 ships with the Image tool only. Design / Layout / AI / Structure
   icons render disabled and surface a "Coming soon" hint until Phase 2-3.
   ========================================================================== */

import { escapeHtml } from '../app.js';

const STORAGE_KEY_PREFIX = 'studio.palette';

// Default tools — order = icon column order. Items can be disabled per
// artifact type or per-phase by setting `.disabled = true` here. Phase 1
// only enables Image; the rest render greyed so the visual rhythm of all
// 5 icons is established from day one.
const DEFAULT_TOOLS = [
  { id: 'image',     icon: 'image',     label: 'Image',     phase: 1 },
  { id: 'design',    icon: 'palette',   label: 'Design',    phase: 2 },
  { id: 'layout',    icon: 'layout',    label: 'Layout',    phase: 3 },
  { id: 'ai',        icon: 'sparkles',  label: 'AI Assist', phase: 3 },
  { id: 'structure', icon: 'list',      label: 'Structure', phase: 3 },
];

// Inline SVG icons. Single-file means no extra fetch + reliable theming.
const ICONS = {
  image:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
  palette:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="8.5" cy="9" r="1.2" fill="currentColor"/><circle cx="15.5" cy="9" r="1.2" fill="currentColor"/><circle cx="7" cy="14" r="1.2" fill="currentColor"/><circle cx="16.5" cy="14.5" r="1.2" fill="currentColor"/><path d="M12 21c1.5 0 2-1 2-2s-.7-2-1.5-2-1.5.5-1.5 1.5S10.5 21 12 21z"/></svg>',
  layout:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>',
  sparkles: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l2 5 5 2-5 2-2 5-2-5-5-2 5-2z"/><path d="M19 14l1 2 2 1-2 1-1 2-1-2-2-1 2-1z"/></svg>',
  list:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><circle cx="4" cy="6" r="1.5" fill="currentColor"/><circle cx="4" cy="12" r="1.5" fill="currentColor"/><circle cx="4" cy="18" r="1.5" fill="currentColor"/></svg>',
  chevron:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"/></svg>',
};

/**
 * Create a palette instance bound to a host element.
 *
 *   const palette = createPalette({ host, artifactType, ctx });
 *   palette.registerTool({ id: 'image', mount, unmount });
 *   palette.activate('image');
 *
 * @param {object} opts
 * @param {HTMLElement} opts.host - container DOM node to mount into
 * @param {string} opts.artifactType - presentation | document | ebook | spreadsheet | chart
 * @param {object} opts.ctx - tool context (artifactId, focusSlot, onSlotChange, api)
 */
export function createPalette({ host, artifactType, ctx }) {
  if (!host) throw new Error('createPalette requires a host element');
  const storageKey = `${STORAGE_KEY_PREFIX}.${artifactType || 'default'}.collapsed`;
  const tools = new Map();
  const state = {
    activeId: null,
    collapsed: _readCollapsed(storageKey),
    artifactType: artifactType || 'default',
    ctx: ctx || {},
  };

  // ----- DOM scaffold --------------------------------------------------------
  host.classList.add('studio-palette');
  host.dataset.state = state.collapsed ? 'collapsed' : 'expanded';
  host.innerHTML = `
    <div class="studio-palette-icons" role="tablist" aria-label="Tool palette"></div>
    <div class="studio-palette-drawer" role="tabpanel">
      <div class="studio-palette-drawer-head">
        <span class="studio-palette-drawer-title">Tools</span>
        <button type="button" class="studio-palette-collapse"
                aria-label="Collapse palette" title="Collapse">${ICONS.chevron}</button>
      </div>
      <div class="studio-palette-drawer-body"></div>
    </div>
  `;
  const iconsEl = host.querySelector('.studio-palette-icons');
  const drawerEl = host.querySelector('.studio-palette-drawer');
  const titleEl = host.querySelector('.studio-palette-drawer-title');
  const bodyEl = host.querySelector('.studio-palette-drawer-body');
  const collapseBtn = host.querySelector('.studio-palette-collapse');

  // ----- Icon column ---------------------------------------------------------
  for (const def of DEFAULT_TOOLS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'studio-palette-icon';
    btn.dataset.toolId = def.id;
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-pressed', 'false');
    btn.setAttribute('aria-label', def.label);
    btn.setAttribute('title', def.label);
    btn.innerHTML = ICONS[def.icon] || ICONS.image;
    iconsEl.appendChild(btn);
    btn.addEventListener('click', () => {
      if (btn.disabled) return;
      // Click an active tab → collapse drawer. Click another → switch.
      if (state.activeId === def.id && !state.collapsed) {
        setCollapsed(true);
      } else {
        activate(def.id);
      }
    });
  }

  collapseBtn.addEventListener('click', () => setCollapsed(true));

  // ----- API -----------------------------------------------------------------
  function registerTool(tool) {
    if (!tool?.id) throw new Error('Tool requires id');
    tools.set(tool.id, tool);
    const btn = iconsEl.querySelector(`[data-tool-id="${tool.id}"]`);
    if (btn) {
      btn.disabled = false;
      btn.removeAttribute('aria-disabled');
      btn.removeAttribute('title');
      btn.setAttribute('title', tool.label || tool.id);
    }
  }

  function unregisterTool(toolId) {
    if (state.activeId === toolId) _teardownActive();
    tools.delete(toolId);
    const btn = iconsEl.querySelector(`[data-tool-id="${toolId}"]`);
    if (btn) {
      btn.disabled = true;
      btn.setAttribute('aria-disabled', 'true');
      btn.setAttribute('title', `${btn.getAttribute('aria-label') || ''} (coming soon)`);
    }
  }

  function activate(toolId) {
    const tool = tools.get(toolId);
    if (!tool) {
      // Greyed-out future tool — surface a hint instead of silently nothing.
      _showPlaceholder(toolId);
      return;
    }
    if (state.activeId === toolId && !state.collapsed) return;
    _teardownActive();
    state.activeId = toolId;
    if (state.collapsed) setCollapsed(false);
    _setPressedState(toolId);
    titleEl.textContent = tool.label || toolId;
    bodyEl.innerHTML = '';
    try {
      tool.mount?.(bodyEl, state.ctx);
    } catch (err) {
      bodyEl.innerHTML = `<div class="studio-tool-image-empty">Failed to load tool: ${escapeHtml(String(err))}</div>`;
    }
  }

  function setCollapsed(collapsed) {
    state.collapsed = !!collapsed;
    host.dataset.state = state.collapsed ? 'collapsed' : 'expanded';
    _writeCollapsed(storageKey, state.collapsed);
    if (state.collapsed) {
      _setPressedState(null);
    } else if (state.activeId) {
      _setPressedState(state.activeId);
    }
  }

  function setCtx(partial) {
    Object.assign(state.ctx, partial || {});
    // Active tool may have wired in onCtxChange — let it know
    const tool = tools.get(state.activeId);
    if (tool?.onCtxChange) {
      try { tool.onCtxChange(state.ctx); }
      catch (err) { console.warn('palette tool onCtxChange threw:', err); }
    }
  }

  function destroy() {
    _teardownActive();
    tools.clear();
    host.innerHTML = '';
    host.classList.remove('studio-palette');
    delete host.dataset.state;
  }

  // ----- Helpers -------------------------------------------------------------
  function _setPressedState(toolId) {
    iconsEl.querySelectorAll('.studio-palette-icon').forEach((btn) => {
      btn.setAttribute('aria-pressed', btn.dataset.toolId === toolId ? 'true' : 'false');
    });
  }

  function _teardownActive() {
    const tool = tools.get(state.activeId);
    if (tool?.unmount) {
      try { tool.unmount(bodyEl, state.ctx); }
      catch (err) { console.warn('palette tool unmount threw:', err); }
    }
    state.activeId = null;
    bodyEl.innerHTML = '';
  }

  function _showPlaceholder(toolId) {
    const def = DEFAULT_TOOLS.find((d) => d.id === toolId);
    if (!def) return;
    if (state.collapsed) setCollapsed(false);
    _setPressedState(toolId);
    state.activeId = toolId;
    titleEl.textContent = def.label;
    bodyEl.innerHTML = `
      <div class="studio-tool-image-empty">
        <strong>${escapeHtml(def.label)}</strong> arrives in Phase ${def.phase}.<br>
        For now, use the editor's existing controls.
      </div>
    `;
  }

  return {
    registerTool,
    unregisterTool,
    activate,
    setCollapsed,
    setCtx,
    destroy,
    get activeId() { return state.activeId; },
    get collapsed() { return state.collapsed; },
  };
}

// ---------------------------------------------------------------------------
// Storage helpers — defensive against localStorage being unavailable
// (private window, quota errors, etc.)
// ---------------------------------------------------------------------------

function _readCollapsed(key) {
  try {
    return window.localStorage?.getItem(key) === '1';
  } catch { return false; }
}

function _writeCollapsed(key, value) {
  try {
    window.localStorage?.setItem(key, value ? '1' : '0');
  } catch { /* ignore */ }
}

