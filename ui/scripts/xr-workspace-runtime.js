/*
 * XR Workspace Runtime
 *
 * A side-effect-light coordinator for headset-native Augmentum surfaces.
 * It stores spatial workspace intent/state and emits XR-only events, leaving
 * desktop DOM panels, iframes, and app routing under their existing owners.
 */

import { describeXrSurface } from './xr-surface-adapters.js';
import { buildXrMediaNavigationState } from './xr-media-library.js';

export const XR_WORKSPACE_EVENTS = Object.freeze({
  intent: 'augmentum:xr-workspace-intent',
  state: 'augmentum:xr-workspace-state',
});

export const XR_WORKSPACE_INTENTS = Object.freeze({
  OPEN_SURFACE: 'open_surface',
  FOCUS_SURFACE: 'focus_surface',
  CLOSE_SURFACE: 'close_surface',
  INVOKE_SURFACE_ACTION: 'invoke_surface_action',
  UPDATE_CAPABILITIES: 'update_capabilities',
  SET_PRESENTATION: 'set_presentation',
  SET_MEDIA_LIBRARY: 'set_media_library',
});

function _defaultEventTarget() {
  return typeof window !== 'undefined' ? window : null;
}

function _key(value) {
  return String(value || '').trim();
}

function _now() {
  return new Date().toISOString();
}

function _emit(target, name, detail) {
  if (!target?.dispatchEvent || typeof CustomEvent !== 'function') return false;
  try {
    target.dispatchEvent(new CustomEvent(name, { detail }));
    return true;
  } catch {
    return false;
  }
}

function _clonePanel(panel) {
  return {
    action: panel.action,
    label: panel.label,
    openedAt: panel.openedAt,
    focusedAt: panel.focusedAt,
    selectedAction: panel.selectedAction,
    state: panel.state,
  };
}

class XrWorkspaceRuntime {
  constructor({
    surfaces = [],
    sessionId = '',
    presentation = 'vr',
    capabilities = {},
    eventTarget = _defaultEventTarget(),
    mediaItems = [],
    nowPlaying = null,
  } = {}) {
    this.sessionId = sessionId || '';
    this.presentation = presentation || 'vr';
    this.capabilities = { ...(capabilities || {}) };
    this.eventTarget = eventTarget;
    this.surfaces = new Map();
    this.panels = new Map();
    this.activeSurface = 'voice';
    this.focusedSurface = 'voice';
    this.intentJournal = [];
    this.disposed = false;
    this.mediaNavigation = buildXrMediaNavigationState({ items: mediaItems, nowPlaying });
    this.registerSurfaces(surfaces);
  }

  registerSurfaces(surfaces = []) {
    if (!Array.isArray(surfaces)) return this.snapshot();
    surfaces.forEach((surface) => this.registerSurface(surface));
    this._emitState('surfaces_registered');
    return this.snapshot();
  }

  registerSurface(surface = {}) {
    const action = _key(surface.action || surface.id);
    if (!action) return null;
    const normalized = {
      id: _key(surface.id || action),
      action,
      label: _key(surface.label || action),
      hint: _key(surface.hint || surface.hubHint || surface.voiceCue),
      placement: _key(surface.placement),
      panelKind: _key(surface.panelKind),
      embedUrl: _key(surface.embedUrl),
      primaryActions: Array.isArray(surface.primaryActions) ? [...surface.primaryActions] : [],
      contextSources: Array.isArray(surface.contextSources) ? [...surface.contextSources] : [],
    };
    this.surfaces.set(action, normalized);
    return normalized;
  }

  describeSurface(action, context = {}) {
    const key = _key(action);
    const surface = this.surfaces.get(key) || { action: key, label: key };
    const description = describeXrSurface(surface, context);
    if (key !== 'media') return description;
    return {
      ...description,
      mediaNavigation: this.mediaNavigation,
    };
  }

  openSurface(action, detail = {}, options = {}) {
    const key = _key(action);
    if (!key || this.disposed) return this.snapshot();
    const surface = this.surfaces.get(key) || this.registerSurface({ action: key, label: detail.label });
    const existing = this.panels.get(key);
    const panel = existing || {
      action: key,
      label: surface?.label || key,
      openedAt: _now(),
      focusedAt: '',
      selectedAction: '',
      state: 'open',
    };
    panel.state = 'open';
    panel.focusedAt = _now();
    this.panels.set(key, panel);
    this.activeSurface = key;
    this.focusedSurface = key;
    this._recordIntent(XR_WORKSPACE_INTENTS.OPEN_SURFACE, {
      action: key,
      label: panel.label,
      ...detail,
    }, options);
    return this.snapshot();
  }

  focusSurface(action, detail = {}, options = {}) {
    const key = _key(action);
    if (!key || this.disposed) return this.snapshot();
    if (!this.panels.has(key)) {
      return this.openSurface(key, detail, options);
    }
    const panel = this.panels.get(key);
    panel.focusedAt = _now();
    this.activeSurface = key;
    this.focusedSurface = key;
    this._recordIntent(XR_WORKSPACE_INTENTS.FOCUS_SURFACE, { action: key, ...detail }, options);
    return this.snapshot();
  }

  closeSurface(action, detail = {}, options = {}) {
    const key = _key(action);
    if (!key || this.disposed) return this.snapshot();
    this.panels.delete(key);
    if (this.activeSurface === key || this.focusedSurface === key) {
      const next = this.panels.keys().next().value || 'voice';
      this.activeSurface = next;
      this.focusedSurface = next;
    }
    this._recordIntent(XR_WORKSPACE_INTENTS.CLOSE_SURFACE, { action: key, ...detail }, options);
    return this.snapshot();
  }

  invokeSurfaceAction(action, panelAction, detail = {}, options = {}) {
    const key = _key(action);
    const selectedAction = _key(panelAction);
    if (!key || this.disposed) return this.snapshot();
    if (!this.panels.has(key)) this.openSurface(key, detail, { emit: false });
    const panel = this.panels.get(key);
    panel.selectedAction = selectedAction;
    panel.focusedAt = _now();
    this.activeSurface = key;
    this.focusedSurface = key;
    if (key === 'media' && selectedAction) {
      this.mediaNavigation = buildXrMediaNavigationState({
        activeSection: selectedAction,
        items: this.mediaNavigation?.sections?.flatMap((section) => section.items || []) || [],
        nowPlaying: this.mediaNavigation?.nowPlaying || null,
      });
    }
    this._recordIntent(XR_WORKSPACE_INTENTS.INVOKE_SURFACE_ACTION, {
      action: key,
      panelAction: selectedAction,
      ...detail,
    }, options);
    return this.snapshot();
  }

  updateCapabilities(capabilities = {}, detail = {}, options = {}) {
    if (this.disposed) return this.snapshot();
    this.capabilities = { ...(capabilities || {}) };
    this._recordIntent(XR_WORKSPACE_INTENTS.UPDATE_CAPABILITIES, detail, {
      emit: options.emit ?? false,
    });
    return this.snapshot();
  }

  setPresentation(presentation = 'vr', detail = {}, options = {}) {
    if (this.disposed) return this.snapshot();
    this.presentation = presentation || 'vr';
    this._recordIntent(XR_WORKSPACE_INTENTS.SET_PRESENTATION, {
      presentation: this.presentation,
      ...detail,
    }, options);
    return this.snapshot();
  }

  setMediaLibrary({ items = [], nowPlaying = null, activeSection = '' } = {}, options = {}) {
    if (this.disposed) return this.snapshot();
    this.mediaNavigation = buildXrMediaNavigationState({ items, nowPlaying, activeSection });
    this._recordIntent(XR_WORKSPACE_INTENTS.SET_MEDIA_LIBRARY, {
      activeSection: this.mediaNavigation.activeSection,
      itemCount: items.length,
    }, options);
    return this.snapshot();
  }

  snapshot() {
    return {
      sessionId: this.sessionId,
      presentation: this.presentation,
      activeSurface: this.activeSurface,
      focusedSurface: this.focusedSurface,
      capabilities: { ...this.capabilities },
      panels: Array.from(this.panels.values()).map(_clonePanel),
      surfaces: Array.from(this.surfaces.values()).map((surface) => ({ ...surface })),
      mediaNavigation: this.mediaNavigation,
      intentJournal: this.intentJournal.slice(-20),
    };
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.panels.clear();
    this.intentJournal.length = 0;
  }

  _recordIntent(intent, detail = {}, { emit = true } = {}) {
    const entry = {
      intent,
      at: _now(),
      sessionId: this.sessionId,
      presentation: this.presentation,
      activeSurface: this.activeSurface,
      focusedSurface: this.focusedSurface,
      detail: { ...(detail || {}) },
    };
    this.intentJournal.push(entry);
    if (this.intentJournal.length > 80) this.intentJournal.splice(0, this.intentJournal.length - 80);
    if (emit) _emit(this.eventTarget, XR_WORKSPACE_EVENTS.intent, entry);
    this._emitState(intent);
  }

  _emitState(reason) {
    if (this.disposed) return;
    _emit(this.eventTarget, XR_WORKSPACE_EVENTS.state, {
      reason,
      snapshot: this.snapshot(),
    });
  }
}

export function createXrWorkspaceRuntime(options = {}) {
  return new XrWorkspaceRuntime(options);
}
