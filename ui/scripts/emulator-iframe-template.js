// ui/scripts/emulator-iframe-template.js
//
// Generates the srcdoc HTML for the iframe that hosts EmulatorJS. The
// template includes:
//
//   1. The standard EmulatorJS bootstrap (sets EJS_player, EJS_core,
//      EJS_gameUrl, EJS_pathtodata from the launch handle's metadata)
//   2. An inline bridge script that wires EmulatorJS' save callbacks
//      to postMessage so the parent EmulatorBridge can proxy saves
//      to /api/titles/{id}/saves/*
//
// The template is exported as a function so the template literals
// don't get parsed at module-load time and tests can render it with
// fixture inputs.
//
// Why srcdoc (vs same-origin iframe URL): no extra route to maintain;
// the iframe inherits the parent origin so it can hit the relative
// /ui/lib/emulator-js/ + /api/* paths directly. The sandbox attribute
// (allow-scripts allow-same-origin) is set by the parent at iframe
// creation -- this template assumes that's been done.

/**
 * Render the srcdoc HTML.
 *
 * @param {object} args
 *   @param {object} args.config - LaunchHandle.metadata (system/core/rom_url/...)
 *   @param {object} args.protocol - PROTOCOL constants from emulator-bridge.js
 *   @param {string} args.titleId - title artifact id (used in error messages)
 * @returns {string} Full HTML document for the iframe srcdoc.
 */
export function renderEmulatorIframeSrcdoc({ config, protocol, titleId }) {
  if (!config || typeof config !== 'object') {
    throw new Error('renderEmulatorIframeSrcdoc: config required');
  }
  if (!protocol || typeof protocol !== 'object') {
    throw new Error('renderEmulatorIframeSrcdoc: protocol required');
  }

  // Defensive: every interpolated value is escaped + quoted on the
  // JS side. We never inline user-controlled HTML.
  const cfgJson = JSON.stringify(_normaliseConfig(config));
  const protoJson = JSON.stringify(protocol);
  const titleIdJson = JSON.stringify(String(titleId || ''));
  // Inject the parent's REAL origin. The iframe's own
  // window.location.origin returns "null" / null for srcdoc iframes
  // even when allow-same-origin is set, which means a reply sent with
  // that as the targetOrigin would never reach the parent. We compute
  // it on the parent side once at render-time and embed the literal.
  const parentOriginJson = JSON.stringify(
    typeof window !== 'undefined' && window.location ? window.location.origin : '*'
  );

  // The iframe-side bridge script. Lives inline so the iframe never
  // makes a separate fetch beyond the EmulatorJS loader + ROM blob.
  // Style: a single IIFE; no module imports (srcdoc + ESM is finicky
  // across browsers, and we want the smallest possible surface area
  // here).
  const bridge = `
(function() {
  'use strict';
  var CFG = ${cfgJson};
  var PROTO = ${protoJson};
  var TITLE_ID = ${titleIdJson};
  // NOTICE: parent's real origin is injected at render time because
  // a srcdoc iframe's own window.location.origin can be "null" /
  // null, which would cause reply postMessages to never reach the
  // parent.
  var PARENT_ORIGIN = ${parentOriginJson};

  function post(payload) {
    try { parent.postMessage(payload, PARENT_ORIGIN); }
    catch (e) { /* parent may have torn down */ }
  }

  function postError(message, recoverable) {
    post({ type: PROTO.ERROR, message: String(message || 'unknown'), recoverable: !!recoverable });
  }

  // ── WebGL framebuffer readback patch ─────────────────────────────
  //
  // EmulatorJS's libretro GL backend creates its WebGL context with
  // the browser default preserveDrawingBuffer:false, which means
  // after each frame is composited to the screen the canvas
  // framebuffer is discarded -- any later toDataURL / drawImage reads
  // back as black even though the pixels you see on screen still
  // exist in the compositor. That's why the agent's canvas readback
  // sees a uniform-black image for the entire session.
  //
  // We patch HTMLCanvasElement.prototype.getContext at IIFE start --
  // before EmulatorJS's loader.js runs -- so every WebGL context
  // created in this iframe gets preserveDrawingBuffer:true. The GPU
  // cost is the extra memory to retain a single framebuffer (240x160
  // for GBA, scaled) per swap; negligible at retro resolutions.
  // Game-graphics-agnostic; works for any libretro core.
  (function patchPreserveDrawingBuffer() {
    var orig = HTMLCanvasElement.prototype.getContext;
    if (!orig) return;
    HTMLCanvasElement.prototype.getContext = function(type, attrs) {
      if (type === 'webgl' || type === 'webgl2' ||
          type === 'experimental-webgl' || type === 'experimental-webgl2') {
        attrs = Object.assign({}, attrs || {}, { preserveDrawingBuffer: true });
      }
      return orig.call(this, type, attrs);
    };
  })();

  // ── Frame-uniformity probe (agent vision gate) ────────────────────
  //
  // Detect a still-loading or core-not-yet-rendering canvas so the
  // parent can skip the frame send for this tick. Implementation:
  // render the source canvas downscaled into a small offscreen 2D
  // context, then sample every pixel and compare against the first.
  // Works for both 2D and WebGL source canvases (drawImage handles
  // both transparently). The downscale is fixed at 8×8 = 64 samples;
  // empirically enough to detect uniform colour without false
  // positives on real game content.
  //
  // Cost: a single drawImage + 256-byte getImageData per tick.
  // Compared to the toDataURL('image/png') we'd run anyway, this is
  // ~1% extra and saves the ~15 KB upstream PNG plus an LLM call on
  // every blank tick during boot.
  //
  // Reused across calls — instantiating a canvas every 2 s would
  // churn through Web platform objects unnecessarily.
  var _UNIFORM_CHECK_SIZE = 8;          // sample grid edge (8 → 64 pixels)
  var _UNIFORM_TOLERANCE  = 5;          // per-channel max delta (PNG-compression headroom)
  var _uniformProbeCanvas = null;
  var _uniformProbeCtx    = null;
  // Cached framebuffer canvas resolution. EmulatorJS keeps multiple
  // canvases at runtime (libretro framebuffer + menu overlay +
  // occasionally a debug canvas), and window.EJS_emulator.canvas does
  // NOT always point at the framebuffer for every core — we observed
  // an mgba session where it was a static black overlay.
  // _resolveFramebufferCanvas scans every <canvas> in the document and
  // returns the largest one whose pixels aren't uniform; that pick is
  // cached here so we don't enumerate on every tick once we've found it.
  var _cachedFramebuffer = null;
  function _isCanvasUniform(srcCanvas) {
    try {
      if (!_uniformProbeCanvas) {
        _uniformProbeCanvas = document.createElement('canvas');
        _uniformProbeCanvas.width = _UNIFORM_CHECK_SIZE;
        _uniformProbeCanvas.height = _UNIFORM_CHECK_SIZE;
        // willReadFrequently hints the browser to keep the backing
        // store in CPU-readable memory; without it Chromium triggers a
        // GPU readback warning every tick.
        _uniformProbeCtx = _uniformProbeCanvas.getContext('2d', { willReadFrequently: true });
      }
      _uniformProbeCtx.drawImage(
        srcCanvas, 0, 0, _UNIFORM_CHECK_SIZE, _UNIFORM_CHECK_SIZE,
      );
      var data = _uniformProbeCtx.getImageData(
        0, 0, _UNIFORM_CHECK_SIZE, _UNIFORM_CHECK_SIZE,
      ).data;
      var r0 = data[0], g0 = data[1], b0 = data[2];
      for (var i = 4; i < data.length; i += 4) {
        if (Math.abs(data[i]     - r0) > _UNIFORM_TOLERANCE ||
            Math.abs(data[i + 1] - g0) > _UNIFORM_TOLERANCE ||
            Math.abs(data[i + 2] - b0) > _UNIFORM_TOLERANCE) {
          return false;
        }
      }
      return true;
    } catch (e) {
      // If the probe fails (tainted canvas, GL context lost, etc.) fail
      // OPEN: let the frame through. False negatives on the gate are
      // strictly better than dropping every frame.
      return false;
    }
  }

  // Sample a small fingerprint of the framebuffer canvas — used to
  // compare BEFORE vs AFTER an agent input and produce an
  // "effect_score" the agent can read back. Returns a Uint8ClampedArray
  // of length 256 (8x8 RGBA), or null on failure. Reuses the
  // _uniformProbeCanvas to avoid allocating per-call.
  //
  // Why 8x8 not full-resolution: sums-of-absolute-differences scale
  // linearly with sample count, and 64 pixels is plenty to detect any
  // meaningful change (dialog appears, character moves, menu opens).
  // Full-res diffing would cost 240*160 = 38400 px per input -- way
  // more than needed, with no signal improvement.
  function _sampleCanvasFingerprint(canvas) {
    try {
      if (!canvas) return null;
      if (!_uniformProbeCanvas) {
        _uniformProbeCanvas = document.createElement('canvas');
        _uniformProbeCanvas.width = _UNIFORM_CHECK_SIZE;
        _uniformProbeCanvas.height = _UNIFORM_CHECK_SIZE;
        _uniformProbeCtx = _uniformProbeCanvas.getContext('2d', { willReadFrequently: true });
      }
      _uniformProbeCtx.drawImage(
        canvas, 0, 0, _UNIFORM_CHECK_SIZE, _UNIFORM_CHECK_SIZE,
      );
      return _uniformProbeCtx.getImageData(
        0, 0, _UNIFORM_CHECK_SIZE, _UNIFORM_CHECK_SIZE,
      ).data;
    } catch (e) {
      return null;
    }
  }

  // Pixel-delta between two fingerprints. Returns sum of per-channel
  // absolute differences across the RGB triples (alpha ignored — it's
  // always 255 for a rendered framebuffer). The agent reads this as
  // "did anything change visually after my input?":
  //   0      = identical fingerprints (input had zero effect)
  //   1-30   = noise / minor animation tick
  //   30-200 = small change (cursor moved, single sprite shifted)
  //   200+   = significant change (dialog appeared, menu opened, scene
  //            transition)
  // Numbers are calibrated empirically for the 240x160 GBA framebuffer.
  function _fingerprintDelta(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    var total = 0;
    for (var i = 0; i < a.length; i += 4) {
      total += Math.abs(a[i]     - b[i]);
      total += Math.abs(a[i + 1] - b[i + 1]);
      total += Math.abs(a[i + 2] - b[i + 2]);
    }
    return total;
  }

  // Resolve the libretro framebuffer canvas.
  //
  // Strategy:
  //   1. If we have a cached canvas that's still in the DOM and not
  //      uniform, reuse it.
  //   2. Otherwise enumerate every <canvas> in the document and pick
  //      the LARGEST non-uniform one. Largest because EmulatorJS often
  //      keeps a small (e.g. 32×32) menu/icon canvas plus the real
  //      framebuffer (e.g. 240×160 for GBA, scaled).
  //   3. If none qualifies, fall back to EJS_emulator.canvas so we at
  //      least try the official reference — better than returning
  //      nothing on rare cores where every canvas is briefly uniform.
  //
  // Re-resolution cost is bounded (~5 canvases enumerated, 64-pixel
  // probe each); only paid until a good cache hits.
  function _resolveFramebufferCanvas() {
    if (_cachedFramebuffer &&
        _cachedFramebuffer.isConnected &&
        !_isCanvasUniform(_cachedFramebuffer)) {
      return _cachedFramebuffer;
    }
    var nodes = document.querySelectorAll('canvas');
    var best = null;
    var bestArea = 0;
    for (var i = 0; i < nodes.length; i++) {
      var c = nodes[i];
      if (!c.width || !c.height) continue;
      if (_isCanvasUniform(c)) continue;
      var area = c.width * c.height;
      if (area > bestArea) {
        bestArea = area;
        best = c;
      }
    }
    if (best) {
      _cachedFramebuffer = best;
      return best;
    }
    var emu = window.EJS_emulator;
    return (emu && emu.canvas) || null;
  }

  // ── Network instrumentation ────────────────────────────────────
  // EmulatorJS has multiple silent failure modes ("Download game data"
  // spinner hangs forever, no console error). Wrapping fetch +
  // XMLHttpRequest lets the parent see each URL being requested,
  // its HTTP status, and whether it ever resolved -- so we can
  // tell the user "ROM 404" vs "core .data 200 but never delivered"
  // without DevTools spelunking.
  (function instrumentNetwork() {
    var origFetch = window.fetch;
    window.fetch = function(input, init) {
      var url = (typeof input === 'string') ? input : (input && input.url) || '?';
      console.log('[ejs-net] fetch start', url);
      var t0 = performance.now();
      return origFetch.apply(this, arguments).then(function(r) {
        var dt = (performance.now() - t0).toFixed(0);
        console.log('[ejs-net] fetch done', url, r.status, dt + 'ms');
        return r;
      }, function(err) {
        var dt = (performance.now() - t0).toFixed(0);
        console.error('[ejs-net] fetch FAIL', url, dt + 'ms', err.message);
        throw err;
      });
    };
    var origXHROpen = XMLHttpRequest.prototype.open;
    var origXHRSend = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function(method, url) {
      this._ejsUrl = url;
      this._ejsT0 = performance.now();
      console.log('[ejs-net] xhr open', method, url);
      return origXHROpen.apply(this, arguments);
    };
    XMLHttpRequest.prototype.send = function() {
      var self = this;
      this.addEventListener('load', function() {
        var dt = (performance.now() - (self._ejsT0 || 0)).toFixed(0);
        console.log('[ejs-net] xhr done', self._ejsUrl, self.status, dt + 'ms');
      });
      this.addEventListener('error', function() {
        var dt = (performance.now() - (self._ejsT0 || 0)).toFixed(0);
        console.error('[ejs-net] xhr FAIL', self._ejsUrl, dt + 'ms');
      });
      return origXHRSend.apply(this, arguments);
    };
  })();

  // ── EmulatorJS globals ─────────────────────────────────────────
  // Matches the documented EmulatorJS configuration surface:
  //   https://emulatorjs.org/docs/options
  // Anything we don't know about is left to default.
  window.EJS_player = '#emu';
  window.EJS_core = CFG.core || '';
  window.EJS_gameUrl = CFG.rom_url || '';
  window.EJS_pathtodata = CFG.emulator_js_path || '/ui/lib/emulator-js/data/';
  // EmulatorJS expects EJS_pathtodata to end in '/'.
  if (!window.EJS_pathtodata.endsWith('/')) window.EJS_pathtodata += '/';
  window.EJS_gameName = CFG.title || 'Untitled';
  window.EJS_color = '#9580ff';
  window.EJS_startOnLoaded = true;

  // Optional BIOS — when bios_required is set the launch refused at the
  // route layer, so by the time we land here the user's already
  // confirmed the BIOS is uploaded. We pass the URL through if the
  // caller supplied one in the metadata.
  if (CFG.bios_url) window.EJS_biosUrl = CFG.bios_url;

  // ── Save / load hooks ──────────────────────────────────────────
  // EmulatorJS exposes onSaveState/onLoadState callbacks. The exact
  // shape varies by release; we hook the documented one and a small
  // set of known aliases so version drift doesn't silently break us.
  function bytesToBase64(arrayLike) {
    if (typeof arrayLike === 'string') return arrayLike;
    var arr = arrayLike instanceof Uint8Array
      ? arrayLike
      : new Uint8Array(arrayLike);
    var CHUNK = 0x8000;
    var s = '';
    for (var i = 0; i < arr.length; i += CHUNK) {
      s += String.fromCharCode.apply(null, arr.subarray(i, i + CHUNK));
    }
    return btoa(s);
  }

  function base64ToBytes(b64) {
    var bin = atob(b64);
    var arr = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return arr;
  }

  // Pending state to load once EmulatorJS finishes init.
  var pendingSram = null;
  var pendingStates = {};

  window.EJS_onSaveState = function(payload) {
    try {
      var slot = (payload && typeof payload.slot === 'number') ? payload.slot : 1;
      var bytes = (payload && payload.screenshot)
        ? null  // EmulatorJS forks differ; primary callers send 'state' or 'data'
        : null;
      // Try the common keys; if the EmulatorJS version uses something
      // else we'll surface an error rather than silently dropping the save.
      var stateBytes =
           (payload && (payload.state || payload.data || payload.savefile));
      if (!stateBytes) {
        postError('save callback fired but state bytes were absent', true);
        return;
      }
      post({
        type: PROTO.STATE_SAVED,
        slot: slot,
        data_b64: bytesToBase64(stateBytes),
        screenshot_b64: payload.screenshot ? bytesToBase64(payload.screenshot) : null,
      });
    } catch (e) {
      postError('onSaveState handler crashed: ' + e.message, true);
    }
  };

  // Some releases call this onLoadSave; track both.
  window.EJS_onSaveSave = function(payload) {
    try {
      var bytes = (payload && (payload.data || payload.savefile));
      if (!bytes) return;
      post({
        type: PROTO.SRAM_SAVED,
        data_b64: bytesToBase64(bytes),
      });
    } catch (e) {
      postError('onSaveSave handler crashed: ' + e.message, true);
    }
  };

  // ── EmulatorJS runtime patches ────────────────────────────────────
  //
  // Decide which memory API surface the loaded core actually offers
  // and stash a single getMemoryRegion(name) accessor on the emulator
  // object so AGENT_READ_MEMORY has one code path regardless of
  // which EmulatorJS / core build is loaded.
  //
  // Three possible API surfaces, in priority order:
  //
  // 1. Module.EmulatorJSGetMemoryData(key) -- the API EmulatorJS
  //    added in upstream commit ed32657 (Feb 2026). Returns a
  //    Uint8Array view into HEAPU8 directly. This is the path our
  //    Tier 2 rebuild produces. Cleanest.
  //
  // 2. gameManager.functions.getMemoryData/getMemorySize cwrap'd by
  //    us at runtime if Module exposes _retro_get_memory_data /
  //    _retro_get_memory_size as raw symbols. This is the Tier 1
  //    fallback path for older EJS builds that compiled the C symbols
  //    in (rare; v4.2.3 does NOT).
  //
  // 3. Neither available -- the core build stripped the symbols.
  //    AGENT_READ_MEMORY surfaces a clean error and the slow path
  //    runs vision-only.
  //
  // Why three paths instead of one: backward / forward compatibility.
  // The new core can land at any time without us shipping a synchronous
  // UI update; older cores that limped via the cwrap workaround keep
  // working until they're replaced.
  function _installAgentRuntimePatches() {
    var report = { memory_api: 'none', system_ram_size: 0, notes: [] };
    try {
      var emu = window.EJS_emulator;
      var gm = emu && emu.gameManager;
      var Module = gm && gm.Module;
      if (!emu || !gm || !Module) {
        report.notes.push('emulator not ready at patch time');
      }
      // Priority 1: upstream EmulatorJS API (memory-capable core).
      else if (typeof Module.EmulatorJSGetMemoryData === 'function') {
        // The function's PRESENCE means the core supports memory reads —
        // that alone is what we gate on. We deliberately do NOT require
        // the sanity probe to succeed: this patch runs on
        // EJS_onGameStart, and at that instant the work-RAM region often
        // isn't mapped yet, so EmulatorJSGetMemoryData throws "memory
        // access out of bounds" even though a read a second later (once
        // the core has run a frame) returns valid data. Gating on the
        // probe here permanently disabled memory for the whole session
        // over a transient startup race. Instead we mark the API
        // available and let the per-tick AGENT_READ_MEMORY reads — each
        // independently try/caught — succeed once the region maps; an
        // early miss just skips that tick.
        report.memory_api = 'EmulatorJSGetMemoryData';
        try {
          var sysRam = Module.EmulatorJSGetMemoryData('RETRO_MEMORY_SYSTEM_RAM');
          report.system_ram_size = (sysRam && sysRam.length) || 0;
          if (!report.system_ram_size) {
            report.notes.push(
              'sanity probe empty at game-start (region not mapped yet; ' +
              'reads will retry per tick)');
          }
        } catch (e) {
          report.notes.push(
            'sanity probe deferred at game-start (region not mapped yet; ' +
            'reads will retry per tick): ' + (e && e.message || e));
        }
      }
      // Priority 2: legacy cwrap path. Some non-standard core builds
      // export the raw libretro symbols without the EmulatorJS helper.
      else if (Module.cwrap &&
               typeof Module._retro_get_memory_data === 'function' &&
               typeof Module._retro_get_memory_size === 'function') {
        var fns = gm.functions;
        fns.getMemoryData = Module.cwrap('retro_get_memory_data', 'number', ['number']);
        fns.getMemorySize = Module.cwrap('retro_get_memory_size', 'number', ['number']);
        try {
          var sz = fns.getMemorySize(2);
          if (sz > 0) {
            report.memory_api = 'cwrap_legacy';
            report.system_ram_size = sz | 0;
          } else {
            report.notes.push('cwrap_legacy probe returned size=0; unwiring');
            fns.getMemoryData = undefined;
            fns.getMemorySize = undefined;
          }
        } catch (e) {
          fns.getMemoryData = undefined;
          fns.getMemorySize = undefined;
          report.notes.push('cwrap_legacy probe threw: ' + (e && e.message || e));
        }
      }
      // Priority 3: nothing. Memory probes will return empty.
      else {
        report.notes.push(
          'no memory API exposed by this core build (neither ' +
          'Module.EmulatorJSGetMemoryData nor _retro_get_memory_data*)',
        );
      }
    } catch (err) {
      report.notes.push('patch threw: ' + (err && err.message || err));
    }
    post({ type: PROTO.READY, agent_runtime_patches: report });
  }

  // ── Virtual gamepad (touch devices) ──────────────────────────────
  //
  // EmulatorJS auto-enables its on-screen gamepad when its own
  // UA-based isMobile returns true. That check misses iPads on
  // Safari 13+, Android tablets that ship a desktop UA string, and
  // touchscreen Chromebooks. We re-evaluate using maxTouchPoints +
  // matchMedia(pointer:coarse) — the same heuristic the streamed
  // stage uses — and call the toggleVirtualGamepad API directly
  // after start. Idempotent: a second toggle(true) when EJS already
  // shows it just resets the same style.display assignment.
  //
  // toggleVirtualGamepad is set on the emulator object during init
  // (setVirtualGamepad runs before onGameStart), but we still poll
  // with a short retry because some forks reorder init.
  function _isTouchDevice() {
    try {
      var hasTouchPoints = (navigator.maxTouchPoints || 0) > 0;
      var coarsePointer = window.matchMedia
        && window.matchMedia('(pointer: coarse)').matches;
      return hasTouchPoints && coarsePointer;
    } catch (_e) {
      return false;
    }
  }
  function _enableVirtualGamepadIfTouch() {
    if (!_isTouchDevice()) return;
    var attempts = 0;
    function tryToggle() {
      attempts += 1;
      var emu = window.EJS_emulator;
      if (emu && typeof emu.toggleVirtualGamepad === 'function') {
        try {
          emu.toggleVirtualGamepad(true);
          post({ type: PROTO.READY, virtual_gamepad_enabled: true });
        } catch (e) {
          postError('virtual gamepad toggle failed: ' + e.message, true);
        }
        return;
      }
      // ~3s total backoff; if it never appears the EJS build doesn't
      // expose the API at all and there's nothing useful we can do.
      if (attempts < 30) setTimeout(tryToggle, 100);
    }
    tryToggle();
  }

  // Game-start → ready. Forks differ on naming; hook the most stable.
  // We also install agent-only runtime patches BEFORE posting READY so
  // the parent knows by the time the bridge opens whether memory
  // probes will actually work.
  window.EJS_onGameStart = function() {
    _installAgentRuntimePatches();
    _enableVirtualGamepadIfTouch();
    // If we received saves before the engine was ready, push them now.
    if (pendingSram) {
      try {
        // EmulatorJS exposes EJS_emulator only after init.
        if (window.EJS_emulator && window.EJS_emulator.gameManager) {
          window.EJS_emulator.gameManager.loadSaveFiles(
            base64ToBytes(pendingSram),
          );
        }
      } catch (e) {
        postError('failed to push SRAM on ready: ' + e.message, true);
      }
      pendingSram = null;
    }
  };

  // ── Listen for parent commands ─────────────────────────────────
  //
  // NOTICE: we gate on (e.source === parent) — sender identity — rather
  // than the origin string. In Chromium srcdoc iframes (even with
  // allow-same-origin), window.location.origin can evaluate to "null"
  // while incoming e.origin is the parent's real origin string, so a
  // strict origin-equality filter drops every parent->iframe command.
  // Sender identity is a stronger contract anyway: the iframe can ONLY
  // receive postMessages from its parent.
  window.addEventListener('message', function(e) {
    if (e.source !== parent) return;
    var msg = e.data;
    if (!msg || typeof msg !== 'object') return;

    switch (msg.type) {
      case PROTO.SAVES_SNAPSHOT:
        if (msg.sram_b64) pendingSram = msg.sram_b64;
        // state index is metadata only; full bodies arrive on LOAD_STATE
        break;

      case PROTO.LOAD_STATE:
        try {
          if (window.EJS_emulator && msg.data_b64) {
            var bytes = base64ToBytes(msg.data_b64);
            // Forks expose loadState via different paths; try the
            // known ones in priority order.
            if (window.EJS_emulator.gameManager &&
                window.EJS_emulator.gameManager.loadState) {
              window.EJS_emulator.gameManager.loadState(bytes);
            } else if (window.EJS_emulator.callEvent) {
              window.EJS_emulator.callEvent('loadState', { state: bytes });
            } else {
              postError('emulator does not expose a load-state API', true);
            }
          }
        } catch (err) {
          postError('load-state failed: ' + err.message, true);
        }
        break;

      case PROTO.SAVE_NOW:
        try {
          if (window.EJS_emulator) {
            if (window.EJS_emulator.gameManager &&
                window.EJS_emulator.gameManager.getState) {
              var state = window.EJS_emulator.gameManager.getState();
              post({
                type: PROTO.STATE_SAVED,
                slot: msg.slot != null ? msg.slot : 1,
                data_b64: bytesToBase64(state),
              });
            } else if (window.EJS_emulator.callEvent) {
              window.EJS_emulator.callEvent('saveState', {});
              // EmulatorJS will fire onSaveState; we don't echo here.
            }
          }
        } catch (err) {
          postError('save-now failed: ' + err.message, true);
        }
        break;

      case PROTO.SET_PAUSED:
        // Best-effort across EmulatorJS forks: pause/play accept
        // optional bool; some forks expose togglePause only.
        try {
          var emu = window.EJS_emulator;
          if (!emu) break;
          var gm = emu.gameManager || emu;
          if (msg.paused) {
            (emu.pause || gm.pause)?.call(gm);
          } else {
            (emu.play || gm.play || emu.resume)?.call(gm);
          }
          post({ type: PROTO.PAUSED, paused: !!msg.paused });
        } catch (err) {
          postError('set-paused failed: ' + err.message, true);
        }
        break;

      case PROTO.RESET:
        // Warm reboot of the libretro core. Forks differ; try the
        // documented restart() first, then a callEvent fallback.
        try {
          var emu2 = window.EJS_emulator;
          if (!emu2) break;
          if (emu2.gameManager && emu2.gameManager.restart) {
            emu2.gameManager.restart();
          } else if (emu2.callEvent) {
            emu2.callEvent('reset', {});
          } else if (emu2.restart) {
            emu2.restart();
          }
        } catch (err) {
          postError('reset failed: ' + err.message, true);
        }
        break;

      case PROTO.SET_VOLUME:
        // Several setVolume signatures across forks; try the most
        // common surfaces.
        try {
          var emu3 = window.EJS_emulator;
          if (!emu3) break;
          var v = Math.max(0, Math.min(1, Number(msg.volume) || 0));
          if (emu3.setVolume) {
            emu3.setVolume(v);
          } else if (emu3.audio && emu3.audio.setVolume) {
            emu3.audio.setVolume(v);
          } else if (emu3.gameManager && emu3.gameManager.setVolume) {
            emu3.gameManager.setVolume(v);
          }
        } catch (err) {
          postError('set-volume failed: ' + err.message, true);
        }
        break;

      case PROTO.OPEN_NATIVE_MENU:
        // EmulatorJS's bottom menu wakes on mousemove inside the
        // canvas — useless on touch. Force it open from the parent so
        // mobile users can reach settings/shaders/controls. We try the
        // public API first, then fall back to the auto-hide CSS class.
        try {
          var menuEmu = window.EJS_emulator;
          if (!menuEmu) break;
          var opened = false;
          if (menuEmu.menu && typeof menuEmu.menu.open === 'function') {
            menuEmu.menu.open();
            opened = true;
          } else if (typeof menuEmu.openSettingsMenu === 'function') {
            menuEmu.openSettingsMenu();
            opened = true;
          }
          if (!opened) {
            // CSS fallback: forcibly strip the auto-hide class. Same
            // class name in 4.2.x; if EJS ever renames we surface a
            // recoverable error so the bar can show a hint.
            var bar = document.querySelector('.ejs_menu_bar');
            if (bar) {
              bar.classList.remove('ejs_menu_bar_hidden');
              opened = true;
            }
          }
          if (!opened) {
            postError('native menu unavailable in this EmulatorJS build', true);
          }
        } catch (err) {
          postError('open-native-menu failed: ' + err.message, true);
        }
        break;

      case PROTO.SET_PERF:
        // Runtime perf knobs. EmulatorJS stores settings under
        // changeSettingOption(key, value); the keys vary by build.
        // We try the documented names and fall through silently on
        // a mismatch — the parent's "Open native menu" is the
        // always-works escape hatch.
        try {
          var perfEmu = window.EJS_emulator;
          if (!perfEmu) break;
          var setOpt = perfEmu.changeSettingOption
            || (perfEmu.settings && perfEmu.settings.changeSettingOption);
          if (typeof setOpt !== 'function') {
            postError('set-perf: changeSettingOption not exposed', true);
            break;
          }
          if (msg.frame_skip !== undefined) {
            // "frameSkip" is the upstream key in EmulatorJS 4.x; some
            // forks use "frame-skip". Try both.
            try { setOpt.call(perfEmu, 'frameSkip', String(msg.frame_skip)); }
            catch (_) { try { setOpt.call(perfEmu, 'frame-skip', String(msg.frame_skip)); } catch (__) {} }
          }
          if (msg.audio_filter !== undefined) {
            try {
              setOpt.call(perfEmu, 'audioFilter', msg.audio_filter ? 'enabled' : 'disabled');
            } catch (audioErr) {
              // Fork-name drift is expected; log so a real
              // permission/init bug isn't silenced alongside it.
              console.warn('[ejs-bridge] audioFilter set failed:', audioErr.message);
            }
          }
        } catch (err) {
          postError('set-perf failed: ' + err.message, true);
        }
        break;

      // ── Game-agent extension ──────────────────────────────────
      //
      // The parent EmulatorBridge polls memory + canvas at low rates
      // and forwards into a server-side game-agent session. This
      // iframe answers those requests using the libretro APIs
      // EmulatorJS already exposes. Nothing here interferes with
      // the user's own keyboard input — agent inputs and user
      // inputs co-exist via simulateInput at the gameManager level.

      case PROTO.AGENT_READ_MEMORY:
        try {
          var emuA = window.EJS_emulator;
          var ModA = emuA && emuA.gameManager && emuA.gameManager.Module;
          var fnsA = emuA && emuA.gameManager && emuA.gameManager.functions;
          // Mapped-region path: the parent addressed a region by its bus
          // start from the core's published memory map. Re-resolve the
          // descriptor's heap pointer FRESH here (EmulatorJSGetMemoryMap
          // re-reads it) so WASM heap growth never leaves us on a stale
          // pointer, then return the whole descriptor's bytes.
          if (typeof msg.map_start === 'number'
              && ModA && typeof ModA.EmulatorJSGetMemoryMap === 'function'
              && ModA.HEAPU8) {
            var mSlice = null;
            try {
              var mmA = ModA.EmulatorJSGetMemoryMap() || [];
              for (var mi = 0; mi < mmA.length; mi++) {
                var dd = mmA[mi];
                if (dd && (dd.start >>> 0) === (msg.map_start >>> 0) && dd.len > 0) {
                  var startA = (dd.ptr >>> 0) + (dd.offset >>> 0);
                  mSlice = ModA.HEAPU8.subarray(startA, startA + (dd.len >>> 0));
                  break;
                }
              }
            } catch (_eMapRead) { mSlice = null; }
            post({ type: PROTO.AGENT_MEMORY_DATA,
                   request_id: msg.request_id,
                   region: msg.region,
                   bytes_b64: (mSlice && mSlice.length) ? bytesToBase64(mSlice) : '' });
            break;
          }
          // Region name -> both the integer ID (legacy cwrap path)
          // and the upstream string key (EmulatorJSGetMemoryData).
          var REGION_IDS  = { save_ram: 0, rtc: 1, system_ram: 2, video_ram: 3 };
          var REGION_KEYS = {
            save_ram:   'RETRO_MEMORY_SAVE_RAM',
            rtc:        'RETRO_MEMORY_RTC',
            system_ram: 'RETRO_MEMORY_SYSTEM_RAM',
            video_ram:  'RETRO_MEMORY_VIDEO_RAM',
          };
          var regionId = REGION_IDS[msg.region];
          var regionKey = REGION_KEYS[msg.region];
          if (regionId === undefined || !regionKey) {
            post({ type: PROTO.ERROR,
                   message: 'unknown memory region: ' + msg.region,
                   recoverable: true });
            break;
          }
          var slice = null;
          // Priority 1: upstream EmulatorJS API (Tier 2 rebuilt core).
          // Returns a Uint8Array view directly -- no manual pointer
          // arithmetic, no HEAPU8 lifetime worries.
          if (ModA && typeof ModA.EmulatorJSGetMemoryData === 'function') {
            try {
              var view = ModA.EmulatorJSGetMemoryData(regionKey);
              if (view && view.length > 0) slice = view;
            } catch (_eA) { /* fall through to legacy */ }
          }
          // Priority 2: cwrap'd legacy path. Only viable when the
          // runtime patch detected and installed it.
          if (!slice && fnsA && fnsA.getMemoryData && fnsA.getMemorySize) {
            var ptr  = fnsA.getMemoryData(regionId);
            var size = fnsA.getMemorySize(regionId);
            if (size > 0 && ModA && ModA.HEAPU8) {
              slice = ModA.HEAPU8.subarray(ptr, ptr + size);
            }
          }
          if (!slice || !slice.length) {
            // Either no API available, or the region is empty for this
            // core. The bridge treats empty as "no data" and the probe
            // tick moves on; the parent's _agentTickProbe handles it.
            post({ type: PROTO.AGENT_MEMORY_DATA,
                   request_id: msg.request_id,
                   region: msg.region,
                   bytes_b64: '' });
            break;
          }
          post({ type: PROTO.AGENT_MEMORY_DATA,
                 request_id: msg.request_id,
                 region: msg.region,
                 bytes_b64: bytesToBase64(slice) });
        } catch (errMem) {
          postError('agent read-memory failed: ' + errMem.message, true);
        }
        break;

      case PROTO.AGENT_LIST_REGIONS:
        // Enumerate which libretro regions this core actually exposes,
        // with their byte sizes, so the parent can build a bus-address
        // memory map (a 32 KB system_ram means IWRAM on GBA; 256 KB
        // means EWRAM). We probe every known region key once; empty /
        // throwing regions are simply omitted.
        try {
          var emuL = window.EJS_emulator;
          var ModL = emuL && emuL.gameManager && emuL.gameManager.Module;
          var fnsL = emuL && emuL.gameManager && emuL.gameManager.functions;
          var REGION_KEYS_L = {
            save_ram:   'RETRO_MEMORY_SAVE_RAM',
            rtc:        'RETRO_MEMORY_RTC',
            system_ram: 'RETRO_MEMORY_SYSTEM_RAM',
            video_ram:  'RETRO_MEMORY_VIDEO_RAM',
            // Some rebuilt cores also publish EWRAM directly. Harmless
            // to probe: unknown keys just return empty and are omitted.
            ewram:      'RETRO_MEMORY_EWRAM',
            iwram:      'RETRO_MEMORY_IWRAM',
          };
          var REGION_IDS_L = { save_ram: 0, rtc: 1, system_ram: 2, video_ram: 3 };
          var regionsL = [];
          for (var rn in REGION_KEYS_L) {
            var len = 0;
            if (ModL && typeof ModL.EmulatorJSGetMemoryData === 'function') {
              try {
                var vL = ModL.EmulatorJSGetMemoryData(REGION_KEYS_L[rn]);
                if (vL && vL.length) len = vL.length;
              } catch (_eL) { /* region not present */ }
            }
            if (!len && fnsL && fnsL.getMemorySize && REGION_IDS_L[rn] !== undefined) {
              try { len = fnsL.getMemorySize(REGION_IDS_L[rn]) || 0; } catch (_eL2) { /* ignore */ }
            }
            if (len > 0) regionsL.push({ region: rn, size: len });
          }
          // Preferred: the core's FULL published libretro memory map
          // (get_memory_map export). Each descriptor is a bus region the
          // parent can address directly (EWRAM/IWRAM/VRAM/... on GBA,
          // WRAM on GB). When present the parent uses these over the four
          // named keys above. Older cores lack the export → empty, and
          // the parent falls back to the named-key + per-console layout.
          if (ModL && typeof ModL.EmulatorJSGetMemoryMap === 'function') {
            try {
              var mm = ModL.EmulatorJSGetMemoryMap() || [];
              for (var di = 0; di < mm.length; di++) {
                var d = mm[di];
                if (!d || !(d.len > 0)) continue;
                regionsL.push({
                  region: 'map@0x' + (d.start >>> 0).toString(16),
                  start: d.start >>> 0,
                  size: d.len >>> 0,
                  mapped: true,
                });
              }
            } catch (_eMap) { /* no map on this core */ }
          }
          post({ type: PROTO.AGENT_REGIONS_DATA,
                 request_id: msg.request_id,
                 regions: regionsL });
        } catch (errList) {
          post({ type: PROTO.AGENT_REGIONS_DATA,
                 request_id: msg.request_id, regions: [] });
        }
        break;

      case PROTO.AGENT_READ_FRAME:
        try {
          // Resolve the framebuffer canvas dynamically each tick.
          // window.EJS_emulator.canvas isn't reliable: for some
          // libretro cores it points at a static menu/overlay canvas
          // rather than the actual framebuffer. _resolveFramebufferCanvas
          // enumerates every <canvas> and picks the largest non-uniform
          // one, caching the result so the enumeration cost is paid
          // once per session rather than per-tick.
          var canvas = _resolveFramebufferCanvas();
          if (!canvas) {
            post({ type: PROTO.AGENT_FRAME_DATA,
                   request_id: msg.request_id, data_url: '' });
            break;
          }
          // The resolver already rejects uniform canvases, but the
          // selected one might have gone uniform between resolution
          // and capture (e.g., during a state load). Double-check; if
          // uniform, return empty and let the next tick re-resolve.
          if (_isCanvasUniform(canvas)) {
            post({ type: PROTO.AGENT_FRAME_DATA,
                   request_id: msg.request_id, data_url: '' });
            break;
          }
          var url;
          try {
            url = canvas.toDataURL('image/png');
          } catch (eFrame) {
            // Tainted canvas — rare with our same-origin setup but
            // possible. Surface empty so parent can decide to retry.
            url = '';
          }
          post({ type: PROTO.AGENT_FRAME_DATA,
                 request_id: msg.request_id, data_url: url });
        } catch (errFrame) {
          postError('agent read-frame failed: ' + errFrame.message, true);
        }
        break;

      case PROTO.AGENT_SIMULATE_INPUT:
        // Helper: always emit exactly one AGENT_INPUT_ACK per request_id
        // so the server-side BridgedAdapter._pending_acks resolves.
        // Without this, any failure path (missing sim, unknown button,
        // sim throws) leaves the server waiting until input_ack_timeout
        // — which manifests as the agent emitting inputs that all time
        // out silently (the exact symptom we hit on Pokémon Sapphire).
        var _ackInput = function (extra) {
          post(Object.assign({
            type: PROTO.AGENT_INPUT_ACK,
            request_id: msg.request_id,
            button: msg.button,
            held_ms: 0,
            tick_count: 0,
            effect_score: 0,
          }, extra || {}));
        };
        try {
          var emuI = window.EJS_emulator;
          var gm = emuI && emuI.gameManager;
          var simFns = gm && gm.functions;
          var sim = simFns && simFns.simulateInput;
          if (!sim) {
            // Surface the diagnostic AND release the pending ack so
            // the agent sees a clean failure on the next planning turn
            // instead of a stalled timeout.
            post({ type: PROTO.ERROR,
                   message: 'core does not expose simulateInput',
                   recoverable: true });
            _ackInput({ error: 'no_simulate_input' });
            break;
          }
          // Two paths converge on a libretro button id:
          //
          //   1. Universal control schema (Phase G). Parent forwards
          //      wire_kind + wire_code; for 'libretro_joypad' the
          //      wire_code is already the RETRO_DEVICE_ID_JOYPAD_*
          //      integer the core wants. We trust it directly so a
          //      new game/console JSON pair can ship without an
          //      iframe-side update.
          //
          //   2. Legacy semantic-name path. msg.button is a short
          //      letter (a/b/up/start/...) and we look it up in the
          //      built-in BTN table. Kept for backwards compatibility
          //      with sessions started without controller_profile +
          //      game_profile.
          //
          // Standard libretro joypad button ids
          // (RETRO_DEVICE_ID_JOYPAD_*). Y is the third face button on
          // NDS / SNES layouts; X is the fourth. GBA uses A/B/L/R +
          // Start/Select; NDS uses A/B/X/Y/L/R + Start/Select.
          var BTN = {
            b: 0, y: 1, select: 2, start: 3,
            up: 4, down: 5, left: 6, right: 7,
            a: 8, x: 9, l: 10, r: 11,
          };
          var _resolveButtonId = function (wireKind, wireCode, name) {
            if (wireKind === 'libretro_joypad'
                && typeof wireCode === 'number'
                && wireCode >= 0
                && wireCode < 256
                && Number.isInteger(wireCode)) {
              return wireCode;
            }
            return BTN[String(name || '').toLowerCase()];
          };
          var buttonId = _resolveButtonId(msg.wire_kind, msg.wire_code, msg.button);
          if (buttonId === undefined) {
            post({ type: PROTO.ERROR,
                   message: 'unknown agent button: ' + msg.button,
                   recoverable: true });
            _ackInput({ error: 'unknown_button' });
            break;
          }
          // Chord parts (real-time games: hold run while pressing
          // jump). Every resolvable part is held simultaneously with
          // the primary for the whole duration; unresolvable parts
          // are reported but never abort the primary press.
          var buttonIds = [buttonId];
          if (Array.isArray(msg.chord)) {
            for (var ci = 0; ci < msg.chord.length && ci < 2; ci++) {
              var part = msg.chord[ci] || {};
              var pid = _resolveButtonId(part.wire_kind, part.wire_code, part.button);
              if (pid === undefined) {
                post({ type: PROTO.ERROR,
                       message: 'unknown chord button: ' + part.button,
                       recoverable: true });
              } else if (buttonIds.indexOf(pid) === -1) {
                buttonIds.push(pid);
              }
            }
          }
          // Hold for at least 180 ms (~11 frames at 60 Hz). Libretro
          // input poll happens once per retro_run() and some games
          // (e.g., Pokémon GBA title screens, Ruby/Sapphire Yes/No
          // prompts, party-summary screens) debounce presses for
          // several frames before accepting. Sub-150 ms presses
          // silently miss on these games. Was 120 ms (~7 frames) but
          // that left a real-world gap on GBA Pokémon — the agent
          // would emit a confirm during a debouncing screen and the
          // press never registered. 180 ms is still well within
          // human-tap range and doesn't measurably hurt fast-action
          // games. The agent's per-action duration_ms can request
          // longer holds and they're respected up to the 2000 ms
          // ceiling.
          var dur = Math.max(180, Math.min(2000, Number(msg.duration_ms) || 180));

          // Press on every frame for the duration; release after.
          // EmulatorJS's gameManager polls input each frame and a
          // single one-shot simulateInput(player, btn, 1) can be
          // overwritten by the keyboard scanner on the very next
          // frame, so we re-assert the press in a requestAnimationFrame
          // loop. The release at the end is done identically to make
          // sure the input falls back to 0 even if the keyboard
          // scanner missed our state.
          //
          // We also fingerprint the framebuffer BEFORE the press and
          // AFTER release + a brief settle delay so the parent can
          // tell whether the input had any visible effect. Useful for
          // diagnosing "agent presses Start but nothing happens"
          // scenarios -- effect_score=0 means the press silently
          // dropped at the core level.
          var pressStart = performance.now();
          var stopped = false;
          var tickCount = 0;
          var preFingerprint =
            _sampleCanvasFingerprint(_resolveFramebufferCanvas());
          var _simAll = function (state) {
            for (var bi = 0; bi < buttonIds.length; bi++) {
              try { sim(0, buttonIds[bi], state); } catch (_e) { /* ignore */ }
            }
          };
          function holdLoop() {
            if (stopped) return;
            _simAll(1);
            tickCount += 1;
            if (performance.now() - pressStart < dur) {
              requestAnimationFrame(holdLoop);
            } else {
              stopped = true;
              _simAll(0);
              // Re-release on next frame in case the keyboard scanner
              // raced us at the exact release tick.
              requestAnimationFrame(function () {
                _simAll(0);
              });
              var heldMs = Math.round(performance.now() - pressStart);
              // Settle delay: the core needs a few frames to render
              // the consequence of the input. 150 ms = ~9 frames at
              // 60 Hz, enough for menu transitions, dialog appearance,
              // and most one-tap responses; short enough that the
              // next agent tick isn't blocked.
              setTimeout(function () {
                var postFingerprint =
                  _sampleCanvasFingerprint(_resolveFramebufferCanvas());
                var effectScore = _fingerprintDelta(
                  preFingerprint, postFingerprint,
                );
                post({
                  type: PROTO.AGENT_INPUT_ACK,
                  request_id: msg.request_id,
                  button: msg.button,
                  held_ms: heldMs,
                  tick_count: tickCount,
                  effect_score: effectScore,
                });
              }, 150);
            }
          }
          // Initial press immediately (don't wait for the first rAF
          // tick — that adds ~16 ms latency). Wrapped in try so a
          // throwing sim() doesn't bypass the rAF loop AND skip the
          // ACK — both failures release the server's pending input.
          try {
            sim(0, buttonId, 1);
          } catch (eInit) {
            postError(
              'agent simulate-input initial press failed: ' + eInit.message,
              true,
            );
            _ackInput({ error: 'sim_initial_threw: ' + eInit.message });
            break;
          }
          // Chord extras join immediately too (errors on extras never
          // abort the primary press — the hold loop re-asserts them).
          for (var xi = 1; xi < buttonIds.length; xi++) {
            try { sim(0, buttonIds[xi], 1); } catch (_eX) { /* ignore */ }
          }
          requestAnimationFrame(holdLoop);
        } catch (errIn) {
          postError('agent simulate-input failed: ' + errIn.message, true);
          // The setup-time throw path also leaves the server waiting.
          // Release the pending input with the error attached.
          _ackInput({ error: 'sim_setup_threw: ' + errIn.message });
        }
        break;
    }
  });

  // Surface bootstrap failures (script not vendored, etc.).
  window.addEventListener('error', function(e) {
    postError(
      'iframe error: ' + ((e.error && e.error.message) || e.message || 'unknown'),
      false,
    );
  });
})();
`;

  // Final HTML. Minimal styling so the emulator gets the full viewport.
  return `<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Augmentum Emulator</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; background: #000; overflow: hidden; }
  #emu { width: 100%; height: 100%; }
</style>
</head><body>
<div id="emu"></div>
<script>${bridge}</script>
<script src="${_safeAttr((config.emulator_js_path || '/ui/lib/emulator-js/data/').replace(/\/?$/, '/'))}loader.js"></script>
</body></html>`;
}


// ── Internals ───────────────────────────────────────────────────────


// Strip / reshape config keys before they cross the wire. We don't
// want the iframe to see internal bookkeeping (run_id etc.); pass
// only the keys the bootstrap actually reads.
function _normaliseConfig(config) {
  return {
    system: String(config.system || ''),
    core: String(config.core || ''),
    rom_url: String(config.rom_url || ''),
    save_bridge_url: String(config.save_bridge_url || ''),
    title: String(config.title || ''),
    bios_required: !!config.bios_required,
    bios_url: config.bios_url ? String(config.bios_url) : '',
    emulator_js_path: String(config.emulator_js_path || '/ui/lib/emulator-js/data/'),
  };
}


// Defensive: even attribute strings get escaped. The path is operator-
// supplied (settings) so the surface is small, but we don't trust it
// to be clean.
function _safeAttr(s) {
  return String(s).replace(/[<>"']/g, (c) => ({
    '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}


export const __test = { _normaliseConfig, _safeAttr };
