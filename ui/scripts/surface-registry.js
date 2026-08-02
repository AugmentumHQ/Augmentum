/**
 * SurfaceRegistry — manages surface types and active instances.
 *
 * Surface classes register themselves on import. The registry creates
 * instances, tracks them, and handles workspace persistence.
 */

const _types = {};
const _instances = new Map();
let _focused = null;

export const SurfaceRegistry = {
  register(type, SurfaceClass) {
    _types[type] = SurfaceClass;
  },

  create(type, config = {}) {
    const Cls = _types[type];
    if (!Cls) throw new Error(`Unknown surface type: ${type}. Registered: ${Object.keys(_types).join(', ')}`);
    const id = config.id || `${type}-${crypto.randomUUID().slice(0, 8)}`;

    // Defensive invariant checks — currently impossible by UI code paths,
    // but workspace-JSON corruption, manual edits, and future race
    // conditions could produce them silently. Logging here gives the
    // first hint when something goes wrong (audit §7.10, §7.18).
    if (config.sessionId) {
      for (const existing of _instances.values()) {
        if (existing._sessionId === config.sessionId) {
          console.warn(
            '[surface-registry] sessionId collision — two surfaces will mutate the same session tree',
            { sessionId: config.sessionId, existingId: existing.id, newId: id, existingType: existing.constructor.type, newType: type },
          );
          break;
        }
      }
    }
    if (config.primary) {
      for (const existing of _instances.values()) {
        if (existing._isPrimary && existing.constructor.type === type) {
          console.warn(
            '[surface-registry] duplicate primary — singleton DOM adoption will steal from the earlier instance',
            { type, existingId: existing.id, newId: id },
          );
          break;
        }
      }
    }

    const surface = new Cls(id, config);
    _instances.set(id, surface);
    return surface;
  },

  get(id) { return _instances.get(id); },

  all() { return [..._instances.values()]; },

  ofType(type) { return this.all().filter(s => s.constructor.type === type); },

  getFocused() { return _focused ? _instances.get(_focused) : null; },

  focus(id) {
    if (_focused && _focused !== id) {
      const prev = _instances.get(_focused);
      if (prev) prev.deactivate();
    }
    const surface = _instances.get(id);
    if (surface) {
      surface.activate();
      _focused = id;
      const context = surface.getContext ? surface.getContext() : {};
      const surfaceType = surface.constructor.type;
      // Only carry `mode` when the surface declares one. Feature surfaces
      // (browse, image) used to fall back to their type name here — that
      // produced detail.mode='browse', which the app.js listener then
      // failed to match against _VALID_MODES and silently no-op'd
      // (audit §4.2). `surfaceType` lets listeners distinguish "mode
      // focus" from "feature focus" without abusing the mode field.
      document.dispatchEvent(new CustomEvent('surface:focus-changed', {
        detail: {
          surfaceId: id,
          type: surfaceType,
          surfaceType,
          mode: context.mode || null,
        },
      }));
    }
  },

  destroy(id) {
    const surface = _instances.get(id);
    if (!surface) return;
    if (_focused === id) _focused = null;
    surface.destroy();
    _instances.delete(id);
    if (_instances.size === 0) {
      document.dispatchEvent(new CustomEvent('surface:all-closed'));
    }
  },

  getTypes() { return Object.keys(_types); },

  hasType(type) { return type in _types; },

  getWorkspaceState() {
    return {
      surfaces: this.all().map(s => s.getState()),
      focused: _focused,
    };
  },

  restoreWorkspace(state) {
    for (const id of [..._instances.keys()]) {
      this.destroy(id);
    }
    for (const surfaceState of (state.surfaces || [])) {
      try {
        const surface = this.create(surfaceState.type, {
          ...surfaceState.config,
          id: surfaceState.id,
          sessionId: surfaceState.sessionId,
        });
        surface.restoreState(surfaceState);
      } catch (e) {
        console.error('Failed to restore surface:', surfaceState, e);
      }
    }
    if (state.focused && _instances.has(state.focused)) {
      _focused = state.focused;
    }
  },

  async saveWorkspace() {
    const state = this.getWorkspaceState();
    try {
      await fetch('/api/config/ui', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workspace: JSON.stringify(state) }),
      });
    } catch { /* best effort */ }
  },

  /**
   * Workspace flush for pagehide/beforeunload. The config endpoint is PUT
   * (sendBeacon only supports POST), so we rely on fetch + keepalive to let
   * the request outlive the page teardown. Best-effort: if the browser
   * drops it, saveWorkspace() still runs on every surface open/close/focus
   * so the window of loss is small.
   */
  flushWorkspace() {
    let payload;
    try {
      payload = JSON.stringify({ workspace: JSON.stringify(this.getWorkspaceState()) });
    } catch {
      return;
    }
    try {
      fetch('/api/config/ui', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
        keepalive: true,
      }).catch(() => {});
    } catch { /* best effort */ }
  },
};
