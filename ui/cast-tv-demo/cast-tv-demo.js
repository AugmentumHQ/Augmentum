/**
 * cast-tv-demo.js — self-testing surface for the cast protocol.
 *
 * Demonstrates the postMessage protocol surfaces use to talk to the
 * receiver shell:
 *
 *   ← parent: { type: 'augmentum.surface_init', surface_id, kind, slot, state }
 *   ← parent: { type: 'augmentum.surface_state', surface_id, patch }
 *
 *   → parent: any plain object — gets forwarded to the server as
 *             a surface_event keyed to this surface's id.
 *
 * The surface emits a 'demo.loaded' event back on init so the
 * orchestrator's automated test fixture can verify the full
 * round-trip (server → receiver → iframe → receiver → server).
 *
 * No external dependencies — runs in any browser context.
 */

const $ = (sel) => document.querySelector(sel);

const titleEl = $('[data-demo-title]');
const statusEl = $('[data-demo-status]');
const sidEl = $('[data-demo-sid]');
const kindEl = $('[data-demo-kind]');
const slotEl = $('[data-demo-slot]');
const stateEl = $('[data-demo-state]');
const patchCountEl = $('[data-demo-patch-count]');
const pulseEl = document.getElementById('pulse');

let surfaceId = '';
let patchCount = 0;
let currentState = {};


function emit(payload) {
  // postMessage to the parent (receiver shell). Plain object → the
  // shell wraps it as a surface_event addressed to our surface_id.
  try { window.parent.postMessage(payload, '*'); } catch {}
}


function renderState(state) {
  currentState = state || {};
  try {
    stateEl.textContent = JSON.stringify(currentState, null, 2);
  } catch {
    stateEl.textContent = String(currentState);
  }
}


function showReady() {
  statusEl.textContent = 'Ready — waiting for state patches';
  pulseEl.style.display = '';
}


window.addEventListener('message', (ev) => {
  const msg = ev.data;
  if (!msg || typeof msg !== 'object') return;

  if (msg.type === 'augmentum.surface_init') {
    surfaceId = String(msg.surface_id || '');
    titleEl.textContent = 'Cast TV Demo';
    sidEl.textContent = surfaceId || '—';
    kindEl.textContent = String(msg.kind || '—');
    slotEl.textContent = String(msg.slot || '—');
    renderState(msg.state);
    showReady();
    // Echo back so the test harness can confirm the round trip.
    emit({
      event: 'demo.loaded',
      data: {
        received_surface_id: surfaceId,
        received_state_keys: Object.keys(msg.state || {}),
      },
    });
    return;
  }

  if (msg.type === 'augmentum.surface_state') {
    patchCount += 1;
    patchCountEl.textContent = String(patchCount);
    // Merge patch into local state for display.
    renderState({ ...currentState, ...(msg.patch || {}) });
    // Acknowledge the patch so the harness can verify it landed.
    emit({
      event: 'demo.patched',
      data: {
        patch_count: patchCount,
        applied_keys: Object.keys(msg.patch || {}),
      },
    });
    return;
  }
});
