/**
 * universal-input-adapter.js — adapter pipeline loader for cast surfaces.
 *
 * Loaded by /ui/play/ and /ui/play-web/ (and Phase-3 origin-proxied games
 * via injection). Listens for two postMessage shapes from the receiver
 * shell:
 *
 *   {kind: 'augmentum.cast_input', slot, pad_index, buttons, axes}
 *     — input frame; fanned out to every active adapter.
 *
 *   {type: 'augmentum.cast_input_config', adapters: [id, ...], keymap: {...}}
 *     — adapter chain reconfiguration; loaded asynchronously, swaps in
 *       atomically once the next frame arrives.
 *
 * Default chain is ['gamepad_api'] — every play surface gets the legacy
 * shim for free, and games that need keyboard/touch/pointer get a new
 * chain configured via CastProfile (Phase 2) or pre-injected by the
 * origin proxy (Phase 3).
 *
 * Telemetry: each frame increments ``frames_received``; each adapter
 * dispatch increments ``dispatches``. The loader emits a periodic
 * ``augmentum.input_telemetry`` event back to the parent (receiver shell)
 * so Phase 4's classifier can demote underperforming chains.
 */

const ADAPTER_BASE = '/ui/scripts/cast-input/adapters/';
const TELEMETRY_TICK_MS = 5000;
const KNOWN_ADAPTERS = new Set(['gamepad_api', 'keyboard', 'touch', 'pointer']);

const _activeAdapters = new Map();   // id → {module, deactivate}
let _frameSubs = [];                  // [{cb}]
let _currentKeymap = null;
// Captured from surface_init so the demotion telemetry can name the
// title + the strategy that's currently in play. Empty until the
// receiver shell delivers the surface state bag.
let _titleId = '';
let _strategy = '';
const _telemetry = {
  frames_received: 0,
  dispatches: 0,
  adapter: '',
  started_at: Date.now(),
};

function _scanTargets() {
  // Count same-realm-reachable vs cross-origin-unreachable iframes.
  // An unreachable iframe with input flowing is the POSITIVE signal the
  // server's demotion loop keys on: a cheap-shim cast whose game lives
  // behind a cross-origin boundary the adapters can't reach. (The
  // dispatch counter can't see this — gamepad_api counts a dispatch per
  // frame whether or not a target received it.)
  let reachable = 0;
  let unreachable = 0;
  try {
    for (const fr of document.querySelectorAll('iframe')) {
      try {
        const w = fr.contentWindow;
        if (w && w.document) reachable += 1;
        else unreachable += 1;
      } catch (_) {
        unreachable += 1;   // cross-origin SecurityError on .document
      }
    }
  } catch (_) { /* document not ready */ }
  return { reachable, unreachable };
}

function _enumerateTargets() {
  const targets = [window];
  try {
    for (const fr of document.querySelectorAll('iframe')) {
      try {
        const w = fr.contentWindow;
        if (w && w.document) targets.push(w);
      } catch (_) { /* cross-origin — skip */ }
    }
  } catch (_) { /* document not ready */ }
  return targets;
}

function _publishFrame(frame) {
  _telemetry.frames_received += 1;
  for (const sub of _frameSubs.slice()) {
    try { sub.cb(frame); } catch (err) {
      console.warn('[cast-adapter] subscriber threw', err);
    }
  }
}

function _registerRecv(cb) {
  const sub = { cb };
  _frameSubs.push(sub);
  return () => {
    _frameSubs = _frameSubs.filter(s => s !== sub);
  };
}

async function _loadAdapterModule(id) {
  if (!KNOWN_ADAPTERS.has(id)) {
    throw new Error(`unknown adapter: ${id}`);
  }
  return import(`${ADAPTER_BASE}${id}.js`);
}

async function _activate(ids, keymap) {
  // Validate + dedupe
  const wanted = [];
  for (const raw of (ids || [])) {
    const id = String(raw || '').trim();
    if (id && KNOWN_ADAPTERS.has(id) && !wanted.includes(id)) {
      wanted.push(id);
    }
  }
  if (!wanted.length) wanted.push('gamepad_api');

  // Deactivate anything no longer wanted
  for (const [id, entry] of Array.from(_activeAdapters.entries())) {
    if (!wanted.includes(id)) {
      try { entry.deactivate?.(); } catch (err) {
        console.warn('[cast-adapter] deactivate threw', id, err);
      }
      _activeAdapters.delete(id);
    }
  }

  _currentKeymap = keymap || null;

  // Activate anything newly wanted
  for (const id of wanted) {
    if (_activeAdapters.has(id)) continue;
    let mod;
    try { mod = await _loadAdapterModule(id); }
    catch (err) {
      console.warn('[cast-adapter] load failed', id, err);
      continue;
    }
    if (typeof mod.activate !== 'function') {
      console.warn('[cast-adapter] adapter has no activate()', id);
      continue;
    }
    const ctx = {
      recv: _registerRecv,
      keymap: _currentKeymap,
      telemetry: _telemetry,
      targets: _enumerateTargets,
    };
    try {
      const deactivate = mod.activate(ctx);
      _activeAdapters.set(id, { module: mod, deactivate });
    } catch (err) {
      console.warn('[cast-adapter] activate threw', id, err);
    }
  }

  _telemetry.adapter = wanted.join(',');
}

function _maybeEmitTelemetry() {
  if (!_telemetry.frames_received && !_telemetry.dispatches) return;
  const { reachable, unreachable } = _scanTargets();
  try {
    window.parent?.postMessage({
      type: 'augmentum.input_telemetry',
      title_id: _titleId,
      strategy: _strategy,
      adapter: _telemetry.adapter,
      frames_received: _telemetry.frames_received,
      dispatches: _telemetry.dispatches,
      reachable_targets: reachable,
      unreachable_targets: unreachable,
      window_ms: Date.now() - _telemetry.started_at,
    }, '*');
  } catch (_) { /* not embedded */ }
  // Roll the window so each tick reports its own delta — keeps the
  // demotion threshold meaningful instead of summing forever.
  _telemetry.frames_received = 0;
  _telemetry.dispatches = 0;
  _telemetry.started_at = Date.now();
}

window.addEventListener('message', (ev) => {
  const msg = ev.data;
  if (!msg || typeof msg !== 'object') return;

  if (msg.type === 'augmentum.cast_input_config'
      && Array.isArray(msg.adapters)) {
    _activate(msg.adapters, msg.keymap || null);
    return;
  }

  // surface_init carries the receiver's initial state bag. When the
  // cast was routed through the classifier (library/cast-launch.js),
  // ``state.cast_input_config`` describes the adapter chain + keymap
  // the classifier chose; we apply it here so the loader doesn't need
  // a second round-trip through the receiver shell.
  if (msg.type === 'augmentum.surface_init' && msg.state) {
    // Capture title + strategy for the demotion telemetry regardless of
    // whether an explicit input chain was supplied (the default
    // gamepad_api chain still needs naming so the server can demote it).
    if (msg.state.artifact_id) _titleId = String(msg.state.artifact_id);
    if (msg.state.cast_strategy) _strategy = String(msg.state.cast_strategy);
    if (msg.state.cast_input_config?.adapters) {
      const cfg = msg.state.cast_input_config;
      _activate(cfg.adapters, cfg.keymap || null);
    }
    // Fall through — other listeners (play.js / play-web.js) also
    // consume surface_init.
  }

  if (msg.kind === 'augmentum.cast_input') {
    _publishFrame(msg);
  }
});

// Boot with the default chain. The receiver / origin proxy can override
// via a later cast_input_config message at any time.
_activate(['gamepad_api'], null);

setInterval(_maybeEmitTelemetry, TELEMETRY_TICK_MS);

// Diagnostic surface for the smoke test.
window.__augCastAdapter = {
  active: () => Array.from(_activeAdapters.keys()),
  telemetry: () => ({ ..._telemetry }),
  targets: () => _scanTargets(),
  context: () => ({ title_id: _titleId, strategy: _strategy }),
  reconfigure: (ids, keymap) => _activate(ids, keymap),
};

// Backwards-compat alias — the smoke checklist + existing receiver-side
// debug calls reference __augCastInput from the legacy shim.
window.__augCastInput = window.__augCastInput || {
  pads: () => {
    const ga = _activeAdapters.get('gamepad_api');
    return ga?.module?._diagnostics?.()?.pads || [];
  },
  frameCount: () => _telemetry.frames_received,
  rewalk: () => { /* no-op — adapters re-walk per frame */ },
};
