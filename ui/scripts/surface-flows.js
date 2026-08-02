/* ==========================================================================
   Surface Flows — Event-based data pipelines between surfaces
   Phase 2 of the workspace architecture.

   Surfaces emit typed events. Other surfaces subscribe to them.
   The flow system routes events, maintains subscriptions, and
   provides built-in flows for common cross-surface patterns.

   Example: narrative surface generates scene → image surface illustrates
   Example: browse surface extracts article → chat surface gets context
   Example: voice link tap → browse surface opens URL
   ========================================================================== */

import { SurfaceRegistry } from './surface-registry.js';

// ---------------------------------------------------------------------------
// Flow Event Types
// ---------------------------------------------------------------------------

/**
 * Typed flow events that surfaces can emit/subscribe to.
 * Each type has a defined payload shape.
 *
 * content:       { text, format?, source? }          — text content to share
 * context:       { text, metadata? }                 — context injection (RAG-like)
 * image:         { url, prompt?, style? }             — generated/found image
 * url:           { url, title?, excerpt? }            — web URL to open/read
 * code:          { code, language?, filename? }       — code snippet
 * scene:         { description, characters?, setting? } — scene to illustrate
 * artifact:      { id, type, name }                   — created artifact
 * command:       { action, target?, params? }          — cross-surface command
 */

// ---------------------------------------------------------------------------
// Flow Registry
// ---------------------------------------------------------------------------

const _subscriptions = new Map();  // eventType → Set<{ surfaceId, handler }>
const _history = [];               // recent flow events for debugging
const MAX_HISTORY = 50;

/**
 * Subscribe a surface to a flow event type.
 * @param {string} eventType — one of the typed events above
 * @param {string} surfaceId — subscribing surface's ID
 * @param {function} handler — (payload, sourceSurfaceId) => void
 * @returns {function} unsubscribe function
 */
export function subscribe(eventType, surfaceId, handler) {
  if (!_subscriptions.has(eventType)) {
    _subscriptions.set(eventType, new Set());
  }
  const entry = { surfaceId, handler };
  _subscriptions.get(eventType).add(entry);

  return () => {
    const subs = _subscriptions.get(eventType);
    if (subs) subs.delete(entry);
  };
}

/**
 * Emit a flow event from a surface.
 * All subscribers to that event type receive it (except the emitter).
 * @param {string} eventType
 * @param {string} sourceSurfaceId — emitting surface's ID
 * @param {object} payload — typed payload
 */
export function emit(eventType, sourceSurfaceId, payload) {
  // Record in history
  _history.push({
    type: eventType,
    source: sourceSurfaceId,
    payload,
    timestamp: Date.now(),
  });
  if (_history.length > MAX_HISTORY) _history.shift();

  // Dispatch to subscribers
  const subs = _subscriptions.get(eventType);
  if (!subs) return;

  for (const { surfaceId, handler } of subs) {
    // Don't send back to emitter
    if (surfaceId === sourceSurfaceId) continue;
    try {
      handler(payload, sourceSurfaceId);
    } catch (err) {
      console.error(`[Flow] Error in ${eventType} handler for surface ${surfaceId}:`, err);
    }
  }

  // Also dispatch as a DOM event for loose coupling
  document.dispatchEvent(new CustomEvent('surface:flow', {
    detail: { type: eventType, source: sourceSurfaceId, payload },
  }));
}

/**
 * Remove all subscriptions for a surface (call on surface destroy).
 */
export function unsubscribeAll(surfaceId) {
  for (const [, subs] of _subscriptions) {
    for (const entry of subs) {
      if (entry.surfaceId === surfaceId) subs.delete(entry);
    }
  }
}

/**
 * Get recent flow history (for debugging / ambient context).
 */
export function getHistory() {
  return [..._history];
}

// ---------------------------------------------------------------------------
// Built-in Flow Wiring
// ---------------------------------------------------------------------------

/**
 * Initialize built-in flows that connect common surface interactions.
 * Called once at startup.
 */
export function initFlows() {
  // --- Browse → Chat context injection ---
  // When the browse panel extracts an article, inject it as context
  // into the focused chat surface.
  document.addEventListener('augmentum:browse-extracted', (e) => {
    const { url, title, text } = e.detail || {};
    if (!text) return;
    emit('context', 'browse', {
      text: text.slice(0, 2000),
      metadata: { url, title, source: 'browse' },
    });
  });

  // --- Voice → Browse URL opening ---
  // Already wired via augmentum:browse-url in voice.js + browse.js

  // --- Image generation results → any subscriber ---
  document.addEventListener('augmentum:image-generated', (e) => {
    const { url, prompt } = e.detail || {};
    if (url) {
      emit('image', 'image-panel', { url, prompt });
    }
  });

  // --- Clean up subscriptions when surfaces are destroyed ---
  document.addEventListener('surface:destroyed', (e) => {
    const surfaceId = e.detail?.surfaceId;
    if (surfaceId) unsubscribeAll(surfaceId);
  });
}

// ---------------------------------------------------------------------------
// Surface Flow Mixin
// ---------------------------------------------------------------------------

/**
 * Add flow capabilities to a surface instance.
 * Call this in the surface constructor or mount:
 *   addFlowMethods(this);
 *
 * Gives the surface:
 *   surface.emitFlow(type, payload)
 *   surface.onFlow(type, handler) → unsubscribe fn
 */
export function addFlowMethods(surface) {
  const _unsubs = [];

  surface.emitFlow = (eventType, payload) => {
    emit(eventType, surface.id, payload);
  };

  surface.onFlow = (eventType, handler) => {
    const unsub = subscribe(eventType, surface.id, handler);
    _unsubs.push(unsub);
    return unsub;
  };

  // Auto-cleanup on destroy
  const origDestroy = surface.destroy?.bind(surface);
  surface.destroy = () => {
    _unsubs.forEach(fn => fn());
    _unsubs.length = 0;
    unsubscribeAll(surface.id);
    if (origDestroy) origDestroy();
  };
}

export const SurfaceFlows = {
  subscribe,
  emit,
  unsubscribeAll,
  getHistory,
  initFlows,
  addFlowMethods,
};
