/**
 * video-companion-session.js — shared state for the app-wide video companion.
 *
 * FloatingVideo owns the DOM/video element lifecycle. This module owns the
 * durable session snapshot other surfaces can read without poking into the
 * player implementation directly.
 */

const LAYOUT_KEY = 'augmentum.videoCompanion.layout.v1';

const _state = {
  isOpen: false,
  shellMode: 'companion',
  deviceKind: 'desktop',
  supportsNativePiP: false,
  isNativePiPActive: false,
  fileId: null,
  videoId: null,
  title: '',
  channel: '',
  thumbnail: '',
  nextItem: null,
  currentTimeS: 0,
  durationS: 0,
  isPlaying: false,
  hasPlaybackOptions: false,
  remoteSessionActive: false,
  remoteSourceType: '',
  remoteTransportKind: '',
  remoteProvider: '',
  remoteServerId: '',
  remoteSessionId: '',
  remoteDeviceName: '',
  remoteSupportedCommands: [],
  isMuted: false,
  volumeLevel: null,
  canSeek: false,
  layout: {
    x: null,
    y: null,
    width: 360,
    height: 203,
  },
};

const _subscribers = new Set();

function _cloneLayout(layout = {}) {
  return {
    x: Number.isFinite(layout.x) ? Number(layout.x) : null,
    y: Number.isFinite(layout.y) ? Number(layout.y) : null,
    width: Number.isFinite(layout.width) ? Number(layout.width) : 360,
    height: Number.isFinite(layout.height) ? Number(layout.height) : 203,
  };
}

function _snapshot() {
  return {
    ..._state,
    nextItem: _state.nextItem ? { ..._state.nextItem } : null,
    remoteSupportedCommands: Array.isArray(_state.remoteSupportedCommands)
      ? [..._state.remoteSupportedCommands]
      : [],
    layout: _cloneLayout(_state.layout),
  };
}

function _notify() {
  const snap = _snapshot();
  for (const fn of _subscribers) {
    try {
      fn(snap);
    } catch (err) {
      console.warn('[video-companion-session] subscriber threw:', err);
    }
  }
}

function _loadPersistedLayout() {
  if (typeof window === 'undefined') return;
  try {
    const raw = window.localStorage.getItem(LAYOUT_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return;
    _state.layout = {
      ..._state.layout,
      ..._cloneLayout(parsed),
    };
  } catch {
    // Ignore bad persisted state.
  }
}

function _persistLayout() {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(LAYOUT_KEY, JSON.stringify(_cloneLayout(_state.layout)));
  } catch {
    // Ignore storage failures.
  }
}

export function normalizeShellMode(mode) {
  const raw = String(mode || '').trim().toLowerCase();
  if (raw === 'native_pip' || raw === 'native-pip' || raw === 'pip' || raw === 'picture-in-picture') {
    return 'native_pip';
  }
  if (raw === 'fullscreen' || raw === 'full') return 'fullscreen';
  if (raw === 'collapsed' || raw === 'audio') return 'collapsed';
  return 'companion';
}

export function inferDeviceKind() {
  if (typeof window === 'undefined') return 'desktop';
  const width = Number(window.innerWidth || 0);
  const coarse = !!window.matchMedia?.('(pointer: coarse)').matches;
  if (width <= 640) return 'mobile';
  if (coarse || width <= 1024) return 'tablet';
  return 'desktop';
}

export function subscribe(fn) {
  if (typeof fn !== 'function') return () => {};
  _subscribers.add(fn);
  try {
    fn(_snapshot());
  } catch (err) {
    console.warn('[video-companion-session] subscriber threw on init:', err);
  }
  return () => _subscribers.delete(fn);
}

export function getState() {
  return _snapshot();
}

export function openSession(patch = {}) {
  updateSession({
    ...patch,
    isOpen: true,
  });
}

export function updateSession(patch = {}) {
  const next = { ...patch };
  if (Object.prototype.hasOwnProperty.call(next, 'shellMode')) {
    _state.shellMode = normalizeShellMode(next.shellMode);
    delete next.shellMode;
  }
  if (Object.prototype.hasOwnProperty.call(next, 'layout')) {
    _state.layout = {
      ..._state.layout,
      ..._cloneLayout(next.layout),
    };
    delete next.layout;
    _persistLayout();
  }
  Object.assign(_state, next);
  _notify();
}

export function setShellMode(mode) {
  updateSession({ shellMode: mode });
}

export function setLayout(layoutPatch = {}) {
  updateSession({ layout: layoutPatch });
}

export function setDeviceKind(kind = inferDeviceKind()) {
  _state.deviceKind = kind || 'desktop';
  _notify();
}

export function closeSession() {
  _state.isOpen = false;
  _state.shellMode = 'companion';
  _state.fileId = null;
  _state.videoId = null;
  _state.supportsNativePiP = false;
  _state.isNativePiPActive = false;
  _state.title = '';
  _state.channel = '';
  _state.thumbnail = '';
  _state.nextItem = null;
  _state.currentTimeS = 0;
  _state.durationS = 0;
  _state.isPlaying = false;
  _state.hasPlaybackOptions = false;
  _state.remoteSessionActive = false;
  _state.remoteSourceType = '';
  _state.remoteTransportKind = '';
  _state.remoteProvider = '';
  _state.remoteServerId = '';
  _state.remoteSessionId = '';
  _state.remoteDeviceName = '';
  _state.remoteSupportedCommands = [];
  _state.isMuted = false;
  _state.volumeLevel = null;
  _state.canSeek = false;
  _notify();
}

_loadPersistedLayout();
_state.deviceKind = inferDeviceKind();
