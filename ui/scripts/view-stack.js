/**
 * ViewStack — coordinates what the user actually sees.
 *
 * Sits on top of three independent state models that pre-dated it:
 *   1. app.state.mode + #app[data-mode]     (legacy mode switcher)
 *   2. SurfaceRegistry / LayoutManager      (first-class surface instances)
 *   3. Ad-hoc overlays (voice / library / image / artifact studio)
 *
 * Before this module, each overlay tore itself down without coordinating:
 *   - ending a voice call revealed whatever was underneath (sometimes blank)
 *   - exiting coder left chat hidden until the user clicked a tab
 *   - persisted-but-dead state (deleted workspace) put up a blocking modal
 *
 * The stack looks like:
 *     [ overlay N ]  ← transient, callback-owned (voice, library, image …)
 *     [   …        ]
 *     [ overlay 0 ]
 *     [ base mode ]  ← one of passthrough / analytical / narrative / agentic / coder
 *
 * Public API:
 *   pushOverlay(id, { onClose, restoreFocus, validate })
 *   popOverlay(id?)
 *   hasOverlay(id)
 *   setBaseMode(mode, { reason?, fromFocus? })
 *   getBaseMode()
 *   boot({ savedMode })
 *   current()
 */

import { SurfaceRegistry } from './surface-registry.js';
import { LayoutManager } from './layout-manager.js';

/** Modes that share the chat-family DOM (chat-scroll, input-area). */
const CHAT_FAMILY = new Set(['passthrough', 'analytical', 'narrative', 'agentic']);

/** Surface type preferred for each base mode (when picking a focus target). */
const MODE_SURFACE_TYPE = {
  passthrough: 'chat',
  analytical: 'chat',
  agentic: 'chat',
  narrative: 'narrative',
  coder: 'coder',
};

const _overlays = [];          // stack of { id, onClose, restoreFocus, validate }
let _baseMode = 'passthrough'; // canonical; mirrors app.state.mode
let _applyMode = null;         // setter injected by app.js (wraps legacy setMode body)
let _boot = false;             // guard: boot() has run at least once

function _emit(name, detail) {
  try {
    document.dispatchEvent(new CustomEvent('viewstack:' + name, { detail }));
  } catch { /* best effort */ }
}

/** Find the best surface to focus for a given mode. Does not create. */
function _pickSurfaceForMode(mode) {
  const type = MODE_SURFACE_TYPE[mode];
  if (!type) return null;
  const matches = SurfaceRegistry.ofType(type);
  if (matches.length === 0) return null;
  // Base-mode navigation should land on the primary surface. Non-primary tabs
  // may share a type, but they own independent composers and should not receive
  // the singleton chat/story DOM during a global mode switch.
  const primary = matches.find(s => s._isPrimary);
  if (primary) return primary;
  // Fallback for pre-primary workspace states.
  const focused = SurfaceRegistry.getFocused();
  if (focused && focused.constructor.type === type) return focused;
  return matches[0];
}

/** Find the current primary surface (at most one exists in a healthy state). */
function _findPrimary() {
  return SurfaceRegistry.all().find(s => s._isPrimary) || null;
}

/**
 * Swap the primary surface when its type no longer matches the active mode.
 *
 * Contract: exactly one primary exists, and its type must match the mode's
 * preferred surface type. Non-chat-family ↔ chat-family transitions (and
 * chat ↔ narrative transitions inside the chat-family DOM) require tearing
 * down one primary and standing up another so singleton DOM (chat-scroll +
 * input-area for chat/narrative, coder-terminal-pane etc. for coder) lives
 * inside the focused surface container rather than leaking into main-area
 * as flex:1 siblings of #surface-grid.
 *
 * Non-primary surfaces (user-pinned via _openAlongside drag-drop) are
 * untouched — they keep their tabs.
 *
 * Only runs at runtime; boot creates its primary through _bootSurfacesMountAll.
 */
function _swapPrimaryForMode(mode) {
  const targetType = MODE_SURFACE_TYPE[mode];
  if (!targetType || !SurfaceRegistry.hasType(targetType)) return;

  const oldPrimary = _findPrimary();
  if (oldPrimary && oldPrimary.constructor.type === targetType) {
    // Same type: chat-family mode flips (passthrough ↔ analytical ↔ agentic)
    // are handled in-place by the primary's _primaryModeListener.
    return;
  }

  // Tear down old primary (if any). Its unmount re-applies `hidden` to any
  // singleton DOM it adopted, so the brief window before the new primary
  // mounts doesn't flash unhidden singletons as flex:1 siblings.
  if (oldPrimary) {
    try {
      LayoutManager.unmountSurface(oldPrimary.id, { preserveFocus: true });
      SurfaceRegistry.destroy(oldPrimary.id);
    } catch (e) {
      console.error('[ViewStack] failed to tear down old primary:', e);
    }
  }

  // Stand up a fresh primary for the new mode. Its mount adopts the new
  // singleton DOM cluster, so applyMode's subsequent visibility toggles land
  // inside the surface container rather than in main-area.
  try {
    const surface = SurfaceRegistry.create(targetType, {
      analytical: mode === 'analytical',
      mode,
      primary: true,
    });
    LayoutManager.mountSurface(surface);
  } catch (e) {
    console.error('[ViewStack] failed to create new primary for mode', mode, e);
  }
}

/** Default restore: orb-nav snaps back via its own listener; focus chat input if visible. */
function _defaultRestore() {
  const input = document.getElementById('chat-input');
  const touchKeyboard = document.documentElement.classList.contains('touch-keyboard');
  if (!touchKeyboard && input && input.offsetParent !== null) {
    try { input.focus({ preventScroll: true }); } catch { /* ignore */ }
  }
}

export const ViewStack = {
  /** Inject the apply-mode function from app.js. Called once during init. */
  _registerApplyMode(fn) { _applyMode = fn; },

  /** Current top-of-stack descriptor — for debugging / introspection. */
  current() {
    if (_overlays.length) {
      const top = _overlays[_overlays.length - 1];
      return { kind: 'overlay', id: top.id };
    }
    return { kind: 'mode', id: _baseMode };
  },

  getBaseMode() { return _baseMode; },

  hasOverlay(id) { return _overlays.some(o => o.id === id); },

  /**
   * Push a transient overlay on top of the current view.
   *
   * @param {string} id  unique identifier (e.g. 'voice', 'library')
   * @param {object} opts
   * @param {Function} [opts.onClose]       called when popped; module tears down its own DOM/state here
   * @param {Function} [opts.restoreFocus]  called AFTER onClose; defaults to focusing chat input
   * @returns {boolean} true if pushed, false if already on stack
   */
  pushOverlay(id, opts = {}) {
    if (this.hasOverlay(id)) return false;
    _overlays.push({
      id,
      onClose: opts.onClose || (() => {}),
      restoreFocus: opts.restoreFocus || _defaultRestore,
      // sticky overlays survive setBaseMode — side-channel UI like the
      // voice call's pet-mode pill, which is content-mode-agnostic. The
      // user can still pop them explicitly via popOverlay(id).
      sticky: !!opts.sticky,
    });
    _emit('overlay-pushed', { id, depth: _overlays.length });
    return true;
  },

  /**
   * Pop an overlay. With no id, pops the top. With an id, pops that specific
   * overlay (and any above it, top-down — preserves LIFO teardown order).
   * Safe to call for overlays that aren't currently pushed (no-op).
   */
  popOverlay(id) {
    if (_overlays.length === 0) return;
    let target;
    if (id == null) {
      target = _overlays.length - 1;
    } else {
      target = _overlays.findIndex(o => o.id === id);
      if (target < 0) return;
    }
    // Tear down from top down to (and including) target — LIFO.
    while (_overlays.length > target) {
      const entry = _overlays.pop();
      try { entry.onClose(); }
      catch (e) { console.error('[ViewStack] onClose failed for', entry.id, e); }
      _emit('overlay-popped', { id: entry.id, depth: _overlays.length });
    }
    // Run restoreFocus of the now-top (overlay or base).
    if (_overlays.length > 0) {
      const top = _overlays[_overlays.length - 1];
      try { top.restoreFocus(); } catch (e) { console.error('[ViewStack] restoreFocus failed:', e); }
    } else {
      _defaultRestore();
      // Re-focus the base mode's surface so LayoutManager shows the right
      // content — guards the "voice ended, screen blank" case.
      const surface = _pickSurfaceForMode(_baseMode);
      if (surface) SurfaceRegistry.focus(surface.id);
    }
  },

  /**
   * Change the base mode.
   *
   * Two callsites with different semantics share this entry point — the
   * `fromFocus` flag distinguishes them (audit §1 — "two-path mode-change
   * asymmetry"). Before unification, focus-driven mode flips wrote
   * `state.mode` directly without going through ViewStack, leaving
   * `_baseMode` to drift out of sync.
   *
   * **Default (deliberate mode change — orb tap, command composer,
   * legacy setMode):**
   * - Pops non-sticky overlays (voice/library shouldn't survive a mode jump)
   * - Swaps the primary surface if its type doesn't match the new mode
   * - Runs applyMode
   * - Re-focuses the new mode's matching surface
   *
   * **`fromFocus: true` (focus-driven sync — tab click, drag-drop,
   * boot focus restore):**
   * - Does NOT pop overlays (focusing a tab shouldn't dismiss a voice call)
   * - Does NOT swap the primary (the focused surface may be non-primary;
   *   destroying the existing primary would be wrong)
   * - Runs applyMode (so visibility toggles realign)
   * - Does NOT re-focus a surface (we're already there — that's what
   *   triggered the call)
   *
   * Both paths still emit `viewstack:mode-changed`.
   *
   * @param {string} mode
   * @param {object} [opts]
   * @param {string} [opts.reason]    diagnostic, emitted in the event
   * @param {boolean} [opts.fromFocus] called from the `surface:focus-changed`
   *                                   listener — skip overlay-pop + primary-
   *                                   swap + post-call re-focus (see above)
   */
  setBaseMode(mode, opts = {}) {
    const fromFocus = !!opts.fromFocus;

    // Drop non-sticky overlays first (deliberate mode-change only). Their
    // onClose runs synchronously so the underlying mode isn't re-hidden by
    // a late teardown once we've switched. Sticky overlays (e.g. a
    // minimized voice call) are kept in place — they are content-mode-
    // agnostic by design and the user explicitly tears them down via their
    // own controls.
    //
    // Focus-driven sync skips this entirely: tab-clicking through tabs
    // should not dismiss a voice call or library overlay (audit §1).
    if (!fromFocus) {
      const kept = [];
      while (_overlays.length > 0) {
        const entry = _overlays.pop();
        if (entry.sticky) {
          kept.push(entry);
          continue;
        }
        try { entry.onClose(); }
        catch (e) { console.error('[ViewStack] onClose during mode change failed for', entry.id, e); }
        _emit('overlay-popped', { id: entry.id, depth: 0, reason: 'mode-change' });
      }
      // Restore sticky overlays preserving original LIFO order (we popped
      // them off the top, so push them back in reverse).
      while (kept.length > 0) _overlays.push(kept.pop());
    }

    const prev = _baseMode;
    _baseMode = mode;

    // Runtime mode switch: the primary surface IS the current mode. If the
    // primary's type doesn't match the new mode's type, swap it in place so
    // the user sees a single surface of the new mode — not an accumulating
    // tab list. Non-primary (user-pinned alongside) surfaces are preserved.
    //
    // Skipped during boot (_bootSurfacesMountAll handles initial primary
    // creation, and no primary exists yet when boot's applyMode runs).
    //
    // Skipped on focus-driven sync: the user may be focusing a non-primary
    // tab — destroying the existing primary to "match the new mode" would
    // be exactly the bug the audit named (§1 — primary swap on focus is
    // not what the user asked for).
    if (!fromFocus && _boot && SurfaceRegistry.all().length > 0) {
      _swapPrimaryForMode(mode);
    }

    if (_applyMode) {
      try { _applyMode(mode); }
      catch (e) { console.error('[ViewStack] applyMode failed:', e); }
    }

    // Coordinate with SurfaceRegistry: focus a matching surface so
    // LayoutManager shows the right container. No-op if no surfaces exist
    // yet (boot path — surfaces get restored later and pick up via
    // surface:focus-changed).
    //
    // Focus-driven sync skips this: the focused surface IS what triggered
    // the call. Re-focusing would either be a no-op or loop through the
    // surface:focus-changed listener again.
    if (!fromFocus) {
      const surface = _pickSurfaceForMode(mode);
      if (surface && SurfaceRegistry.getFocused()?.id !== surface.id) {
        SurfaceRegistry.focus(surface.id);
      }
    }

    _emit('mode-changed', {
      from: prev,
      to: mode,
      reason: opts.reason || null,
      fromFocus,
    });
  },

  /**
   * Boot entry. Call once, after cacheDom() and before any surface
   * restoration. Sets the base mode and primes the stack state.
   *
   * Validation philosophy: we do NOT pre-check the saved mode against the
   * server here — that would add 200ms+ to every boot. Instead the mode's
   * own enter handler is responsible for falling back via setBaseMode(
   * 'passthrough') if it can't render (see coder.js _onEnterCoderMode).
   */
  boot({ savedMode } = {}) {
    _boot = true;
    const allowed = ['passthrough', 'analytical', 'narrative', 'agentic', 'coder'];
    const target = allowed.includes(savedMode) ? savedMode : 'passthrough';
    _baseMode = target;
    if (_applyMode) {
      try { _applyMode(target); }
      catch (e) { console.error('[ViewStack] boot applyMode failed:', e); }
    }
    _emit('booted', { mode: target });
  },

  /** @internal test helper — reset state between harness runs. */
  _reset() {
    _overlays.length = 0;
    _baseMode = 'passthrough';
    _boot = false;
  },

  get depth() { return _overlays.length; },
  get booted() { return _boot; },
};
