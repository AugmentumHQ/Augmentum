/**
 * armed-device.js — Single source of truth for the "armed cast target".
 *
 * When a device is armed, the next media play action routes to that
 * device instead of playing locally. Arm persists across reloads via
 * localStorage but is scoped to the current user — arming carries no
 * state into another user's session on the same browser.
 */

import { getCurrentUser } from './auth.js';

const STORAGE_KEY = 'augmentum-armed-device';

let _armed = null;
const _subs = new Set();


function _readPersisted() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    const user = getCurrentUser();
    if (!user || !data || data.userId !== user.id) return null;
    return data;
  } catch {
    return null;
  }
}


function _writePersisted(state) {
  try {
    if (state) localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    // localStorage may be full / disabled — in-memory state still works.
  }
}


function _notify() {
  for (const fn of _subs) {
    try { fn(_armed); }
    catch (err) { console.warn('[armed-device] subscriber threw:', err); }
  }
}


/**
 * Restore armed state from localStorage. Call once after auth completes.
 */
export function hydrateArmedDevice() {
  _armed = _readPersisted();
  _notify();
}


export function getArmed() {
  return _armed;
}


export function isArmed() {
  return _armed !== null;
}


export function isArmedDevice(deviceId) {
  return _armed !== null && _armed.deviceId === deviceId;
}


/**
 * Arm a device. `device` is a Device dict from /api/devices.
 */
export function arm(device) {
  if (!device || !device.id) return;
  const user = getCurrentUser();
  _armed = {
    deviceId: device.id,
    label: device.label || 'Unnamed device',
    driver: device.driver || '',
    capabilities: Array.isArray(device.capabilities) ? device.capabilities.slice() : [],
    userId: user ? user.id : '',
  };
  _writePersisted(_armed);
  _notify();
}


export function disarm() {
  if (_armed === null) return;
  _armed = null;
  _writePersisted(null);
  _notify();
}


/**
 * Subscribe to armed-state changes. Fires immediately with current state.
 * Returns an unsubscribe function.
 */
export function subscribe(fn) {
  if (typeof fn !== 'function') return () => {};
  _subs.add(fn);
  try { fn(_armed); }
  catch (err) { console.warn('[armed-device] subscriber threw:', err); }
  return () => _subs.delete(fn);
}
