/**
 * Surface — the universal primitive for the Augmentum workspace.
 *
 * Every capability (chat, narrative, coder, browse, image) extends this
 * class. A surface owns its DOM, agent context, and state. Multiple
 * surfaces can coexist, each mounted in its own container.
 *
 * Lifecycle: create → mount → activate/deactivate → unmount → destroy
 * States: created, mounted, active, visible, unmounted, destroyed
 */

let _idCounter = 0;

export class Surface {
  /** @type {string} Override in subclass: 'chat', 'narrative', 'coder', etc. */
  static type = '';

  constructor(id, config = {}) {
    this.id = id || `${this.constructor.type}-${(++_idCounter).toString(36)}`;
    this.config = config;
    this.sessionId = config.sessionId || '';
    this._state = 'created';
    this._container = null;
    this._listeners = {};
  }

  // --- Lifecycle ---

  mount(container) {
    if (this._state === 'destroyed') throw new Error(`Surface ${this.id} is destroyed`);
    this._container = container;
    this._state = 'mounted';
    this.emit('surface:ready', { id: this.id });
  }

  unmount() {
    if (this._container) {
      this._container.innerHTML = '';
    }
    this._container = null;
    this._state = 'unmounted';
  }

  activate() {
    this._state = 'active';
    this.emit('surface:activated', { id: this.id });
  }

  deactivate() {
    if (this._state === 'active') {
      this._state = 'visible';
      this.emit('surface:deactivated', { id: this.id });
    }
  }

  destroy() {
    this.unmount();
    this._state = 'destroyed';
    this._listeners = {};
    this.emit('surface:destroyed', { id: this.id });
  }

  // --- State ---

  getState() {
    return {
      type: this.constructor.type,
      id: this.id,
      sessionId: this.sessionId,
      config: this.config,
    };
  }

  restoreState(state) {
    if (state.sessionId) this.sessionId = state.sessionId;
  }

  // --- Context (for ambient/orchestrator) ---

  getContext() {
    return {
      type: this.constructor.type,
      id: this.id,
      summary: '',
      capabilities: [],
      recentArtifacts: [],
    };
  }

  // --- UI Metadata ---

  getTitle() { return this.constructor.type; }
  getIcon() { return this.constructor.type; }
  getBadge() { return null; }

  /**
   * Whether the tab's close button should be rendered. Default true.
   *
   * Surfaces that ADOPT singleton DOM (chat/narrative primary, via
   * ``_mountPrimary``) must return false — closing them moves the adopted
   * elements back to ``.main-area`` where they sit as siblings of
   * ``#surface-grid``. Both have ``flex: 1``, so .main-area splits 50/50
   * top-to-bottom. The user sees a "glitched splitscreen" instead of the
   * expected clean hand-off to the remaining tab. This is a landmine
   * during the ongoing singleton → instance-scoped migration; once primary
   * adoption is gone the close can be re-enabled.
   */
  isCloseable() { return true; }

  get isMounted() {
    return this._state === 'mounted' || this._state === 'active' || this._state === 'visible';
  }

  get isActive() { return this._state === 'active'; }

  // --- Events ---

  on(event, handler) {
    if (!this._listeners[event]) this._listeners[event] = new Set();
    this._listeners[event].add(handler);
    return () => this._listeners[event]?.delete(handler);
  }

  off(event, handler) {
    this._listeners[event]?.delete(handler);
  }

  emit(event, data = {}) {
    this._listeners[event]?.forEach(fn => {
      try { fn(data); } catch (e) { console.error(`Surface event error [${event}]:`, e); }
    });
    // Generic bus for analytics/devtools (details carry the event name
    // so filters can pick it up).
    document.dispatchEvent(new CustomEvent('surface:event', {
      detail: { surfaceId: this.id, event, data },
    }));
    // Also re-dispatch under the specific name so listeners like
    // surface-flows.js:addEventListener('surface:destroyed', ...) can
    // react without subscribing to the entire bus.
    document.dispatchEvent(new CustomEvent(event, {
      detail: { surfaceId: this.id, ...data },
    }));
  }
}
