// ui/scripts/emulator-bridge.js
//
// Parent-side bridge for the EmulatorJS browser runtime.
//
// Mount an iframe → wire postMessage → proxy save events to the
// /api/titles/{id}/saves/* surface → expose a clean programmatic API
// for the rest of the UI. The iframe is generated via srcdoc with an
// inline bridge script that talks to EmulatorJS' EJS_* globals; see
// `emulator-iframe-template.js` for that side of the wire.
//
// Lifecycle:
//   const bridge = new EmulatorBridge(container, handle, titleId);
//   await bridge.mount();         // awaits 'emu:ready' from iframe
//   bridge.on('state-saved', ...) // listen for events
//   await bridge.saveState(1);    // programmatic save
//   bridge.unmount();             // clean teardown
//
// Save flow:
//   1. On mount, parent fetches all existing saves from the API and
//      sends them to the iframe as 'emu:saves-snapshot'.
//   2. EmulatorJS' onSaveState fires → iframe posts 'emu:state-saved'
//      → parent PUTs /api/titles/{id}/saves/state/{slot}.
//   3. Same for SRAM auto-saves.
//   4. To load a slot, parent fetches the bytes and posts
//      'emu:load-state' → iframe calls EJS_emulator's load API.

import { renderEmulatorIframeSrcdoc } from './emulator-iframe-template.js';

// ── Protocol constants (kept in sync with the iframe template) ──────

const PROTOCOL = {
  // iframe → parent
  READY:        'emu:ready',
  STATE_SAVED:  'emu:state-saved',
  SRAM_SAVED:   'emu:sram-saved',
  PAUSED:       'emu:paused',          // {paused: bool}
  ERROR:        'emu:error',
  // parent → iframe
  SAVES_SNAPSHOT: 'emu:saves-snapshot',
  LOAD_STATE:     'emu:load-state',
  SAVE_NOW:       'emu:save-now',
  SET_PAUSED:     'emu:set-paused',    // {paused: bool}
  RESET:          'emu:reset',
  SET_VOLUME:     'emu:set-volume',    // {volume: 0..1}
  OPEN_NATIVE_MENU: 'emu:open-native-menu',  // ask iframe to surface EJS's own menu
  SET_PERF:       'emu:set-perf',      // {frame_skip?: 0|1|2|4|'auto', audio_filter?: bool}
  // Game-agent extension — iframe ↔ parent
  // Parent asks iframe for a memory region snapshot (libretro
  // RETRO_MEMORY_* by name), or for a canvas frame, or to dispatch a
  // synthetic gamepad input. The iframe replies with *_DATA / *_ACK
  // carrying the request_id so parent-side waiters can resolve.
  AGENT_READ_MEMORY:    'agent:read-memory',    // {region, request_id}
  AGENT_LIST_REGIONS:   'agent:list-regions',   // {request_id} — enumerate exposed libretro regions + sizes
  AGENT_READ_FRAME:     'agent:read-frame',     // {request_id}
  AGENT_SIMULATE_INPUT: 'agent:simulate-input', // {button, duration_ms, request_id, wire_kind?, wire_code?}
  AGENT_MEMORY_DATA:    'agent:memory-data',    // {region, bytes_b64, request_id}
  AGENT_REGIONS_DATA:   'agent:regions-data',   // {regions:[{region,size}], request_id}
  AGENT_FRAME_DATA:     'agent:frame-data',     // {data_url, request_id}
  AGENT_INPUT_ACK:      'agent:input-ack',      // {request_id}
};


// ── Console memory-map layout (agnostic addressing) ─────────────────
//
// The bridge addresses memory by ABSOLUTE console-bus address, not by
// cherry-picked named regions. A preset probe carries a bus address
// (e.g. GBA 0x02024029, GB 0xD163); the resolver walks the exposed
// regions to find which one contains it and reads at the right offset.
//
// Each libretro region the core exposes (system_ram / video_ram /
// save_ram) maps to a bus base per console. `base` is the bus address
// the region starts at; `size` (when set) pins the base when a single
// region name is ambiguous across cores. The GBA `system_ram` name is
// the ambiguous case: a STANDARD mGBA core returns EWRAM (256 KB @
// 0x02000000) for it, while the EmulatorJS nightly core returns IWRAM
// (32 KB @ 0x03000000). We disambiguate by the reported size so the
// SAME preset (absolute addresses) works on either core. When the core
// exposes EWRAM under its own key, that entry supplies 0x02000000
// directly and the ambiguity disappears.
//
// This table is a fallback used until the core publishes its own
// memory map (RETRO_ENVIRONMENT_SET_MEMORY_MAPS); a published map, when
// available, overrides these defaults region-for-region.
const CONSOLE_BUS_LAYOUT = {
  gba: {
    // size-keyed disambiguation for system_ram (iwram 32K vs ewram 256K)
    system_ram: [
      { size: 0x8000,  base: 0x03000000 },   // IWRAM
      { size: 0x40000, base: 0x02000000 },   // EWRAM
    ],
    ewram:      { base: 0x02000000 },
    iwram:      { base: 0x03000000 },
    video_ram:  { base: 0x06000000 },
    save_ram:   { base: 0x0E000000 },
  },
  gb: {
    system_ram: { base: 0xC000 },   // WRAM 0xC000-0xDFFF (+banks on CGB)
    video_ram:  { base: 0x8000 },
    save_ram:   { base: 0xA000 },   // cart RAM
  },
  gbc: {
    system_ram: { base: 0xC000 },
    video_ram:  { base: 0x8000 },
    save_ram:   { base: 0xA000 },
  },
  nes: {
    system_ram: { base: 0x0000 },   // 2 KB internal RAM (0x0000-0x07FF)
    save_ram:   { base: 0x6000 },   // PRG-RAM / battery
  },
};

// GB/GBC/NES pointers are native-endian bus addresses within a small
// space; GBA pointers are full 32-bit bus addresses. No masking needed
// for deref — the stored pointer value IS the bus address.


// ── Public API ──────────────────────────────────────────────────────


export class EmulatorBridge {
  /**
   * @param {HTMLElement} container - DOM element to mount the iframe into
   * @param {object} handle - LaunchHandle from POST /api/titles/{id}/launch
   * @param {string} titleId - the title artifact id
   * @param {object} [options]
   *   @param {function} [options.fetchImpl=fetch] - test injection
   *   @param {Window}   [options.windowImpl=window] - test injection
   */
  constructor(container, handle, titleId, options = {}) {
    if (!container || !handle || !titleId) {
      throw new Error('EmulatorBridge requires container, handle, titleId');
    }
    if (handle.runtime_id !== 'emulator-browser' || handle.kind !== 'emulator') {
      throw new Error(
        `EmulatorBridge expects an emulator-browser handle (got ${handle.runtime_id}/${handle.kind})`,
      );
    }

    this._container = container;
    this._handle = handle;
    this._titleId = titleId;
    this._win = options.windowImpl || ((typeof window !== 'undefined') ? window : null);
    // ``fetch`` is a Window method and throws "Illegal invocation" when
    // called as a bare function reference (the browser's WebIDL binding
    // requires its receiver to be Window/globalThis). Bind on storage so
    // every later ``this._fetch(...)`` is safe regardless of call site.
    // A test-injected fetchImpl is used as-is on the assumption that
    // tests provide a plain function with no implicit receiver.
    const rawFetch = options.fetchImpl
      || ((typeof fetch !== 'undefined') ? fetch.bind(this._win || globalThis) : null);
    this._fetch = rawFetch;
    if (!this._fetch || !this._win) {
      throw new Error('EmulatorBridge needs fetch + window (or test injections)');
    }

    this._iframe = null;
    this._messageListener = null;
    this._listeners = new Map();   // eventName → Set<callback>
    this._readyResolver = null;    // resolved by the 'emu:ready' message
    this._readyPromise = null;
    this._mounted = false;
  }

  // ── Mount / unmount ───────────────────────────────────────────────

  async mount() {
    if (this._mounted) return;
    this._mounted = true;

    // Build the iframe with srcdoc carrying EmulatorJS + the iframe-side
    // bridge. The handle's metadata supplies system/core/rom_url/etc.
    const srcdoc = renderEmulatorIframeSrcdoc({
      config: this._handle.metadata,
      protocol: PROTOCOL,
      titleId: this._titleId,
    });

    const iframe = this._win.document.createElement('iframe');
    iframe.className = 'emulator-iframe';
    iframe.setAttribute('sandbox', 'allow-scripts allow-same-origin');
    iframe.style.width = '100%';
    iframe.style.height = '100%';
    iframe.style.border = '0';
    iframe.srcdoc = srcdoc;
    this._iframe = iframe;
    this._container.appendChild(iframe);

    this._readyPromise = new Promise((resolve) => {
      this._readyResolver = resolve;
    });

    this._messageListener = (e) => this._onMessage(e);
    this._win.addEventListener('message', this._messageListener);

    // Wait for the iframe to report ready (gives us a chance to
    // surface bootstrap errors uniformly). Then push existing saves.
    await this._readyPromise;
    await this._sendSavesSnapshot();
  }

  unmount() {
    if (!this._mounted) return;
    this._mounted = false;
    if (this._messageListener) {
      this._win.removeEventListener('message', this._messageListener);
      this._messageListener = null;
    }
    if (this._iframe) {
      try { this._iframe.parentNode?.removeChild(this._iframe); } catch (_) {}
      this._iframe = null;
    }
    this._listeners.clear();
    this._readyResolver = null;
    this._readyPromise = null;
  }

  // ── Programmatic API ──────────────────────────────────────────────

  /** Trigger a save state at the given slot. */
  saveState(slot) {
    this._post({ type: PROTOCOL.SAVE_NOW, slot: Number(slot) });
  }

  /** Load a saved slot (fetches from the API + forwards to iframe). */
  async loadState(slot) {
    const data = await this._fetchSaveBytes('state', Number(slot));
    if (!data) return false;
    this._post({
      type: PROTOCOL.LOAD_STATE,
      slot: Number(slot),
      data_b64: _bytesToBase64(data),
    });
    return true;
  }

  /** Pause / resume the running game. */
  setPaused(paused) {
    this._post({ type: PROTOCOL.SET_PAUSED, paused: !!paused });
  }

  /** Reset the running game (warm reboot of the libretro core). */
  reset() {
    this._post({ type: PROTOCOL.RESET });
  }

  /** Set audio volume (0..1). */
  setVolume(volume) {
    const clamped = Math.max(0, Math.min(1, Number(volume) || 0));
    this._post({ type: PROTOCOL.SET_VOLUME, volume: clamped });
  }

  /** Surface EmulatorJS's own settings/menu panel inside the iframe.
   *  Touch-only path — EmulatorJS's bottom menu wakes on mousemove
   *  which never fires on phones. */
  openNativeMenu() {
    this._post({ type: PROTOCOL.OPEN_NATIVE_MENU });
  }

  /** Apply runtime perf knobs (frame skip, audio filter). Passes
   *  through to EJS_emulator.changeSettingOption inside the iframe. */
  setPerf(opts) {
    const payload = { type: PROTOCOL.SET_PERF };
    if (opts && opts.frame_skip !== undefined) payload.frame_skip = String(opts.frame_skip);
    if (opts && opts.audio_filter !== undefined) payload.audio_filter = !!opts.audio_filter;
    this._post(payload);
  }

  // ── Event subscription ────────────────────────────────────────────

  on(event, handler) {
    if (!this._listeners.has(event)) this._listeners.set(event, new Set());
    this._listeners.get(event).add(handler);
    return () => this._listeners.get(event)?.delete(handler);
  }

  _emit(event, payload) {
    const set = this._listeners.get(event);
    if (!set) return;
    for (const fn of set) {
      try { fn(payload); } catch (_) { /* swallow */ }
    }
  }

  // ── Message handling ──────────────────────────────────────────────

  _onMessage(e) {
    if (!this._iframe || e.source !== this._iframe.contentWindow) return;
    const msg = e.data;
    if (!msg || typeof msg !== 'object' || typeof msg.type !== 'string') return;
    if (!msg.type.startsWith('emu:') && !msg.type.startsWith('agent:')) return;

    switch (msg.type) {
      case PROTOCOL.READY:
        // Iframe runtime-patch report rides on READY. Surface it to
        // the console so we know at a glance whether the libretro
        // memory API was successfully wrapped for this core build.
        // Forward to listeners too (the agent panel can show probe
        // capability in its UI later).
        if (msg.agent_runtime_patches) {
          const r = msg.agent_runtime_patches;
          // The iframe reports the read API it found as `memory_api`
          // ('EmulatorJSGetMemoryData' / 'cwrap_legacy' / 'none'). A
          // non-'none' value means per-tick reads work. (An earlier bug
          // checked a `wrapped_memory_api` field the iframe never sets,
          // so this always logged a false "NOT wrapped" warning.)
          const apiOk = r.memory_api && r.memory_api !== 'none';
          if (apiOk) {
            console.info(
              '[agent-bridge] libretro memory API available (' + r.memory_api +
              '); system_ram_size=' + r.system_ram_size +
              ' (0x' + (r.system_ram_size || 0).toString(16) + ')',
            );
          } else {
            console.warn(
              '[agent-bridge] libretro memory API not detected at game-start ' +
              '(may map on a later tick); notes:',
              r.notes,
            );
          }
        }
        this._readyResolver?.();
        this._emit('ready', msg.agent_runtime_patches
          ? { agent_runtime_patches: msg.agent_runtime_patches }
          : {});
        break;
      case PROTOCOL.STATE_SAVED:
        this._handleStateSaved(msg).catch((err) => {
          this._emit('error', { message: `state save proxy failed: ${err.message}` });
        });
        break;
      case PROTOCOL.SRAM_SAVED:
        this._handleSramSaved(msg).catch((err) => {
          this._emit('error', { message: `sram save proxy failed: ${err.message}` });
        });
        break;
      case PROTOCOL.PAUSED:
        this._emit('paused-changed', { paused: !!msg.paused });
        break;
      case PROTOCOL.ERROR:
        this._emit('error', {
          message: String(msg.message || 'unknown emulator error'),
          recoverable: !!msg.recoverable,
        });
        break;
      case PROTOCOL.AGENT_MEMORY_DATA:
      case PROTOCOL.AGENT_FRAME_DATA:
      case PROTOCOL.AGENT_REGIONS_DATA:
        this._resolveAgentRequest(msg);
        break;
      case PROTOCOL.AGENT_INPUT_ACK:
        // Inputs are fire-and-forget at the parent layer (no pending
        // entry to resolve), but the iframe's ACK carries effect
        // tracing: how long we held, how many rAF re-asserts ran, and
        // a pixel-delta score for canvas change after release.
        // Forward to:
        //   1. Console -- live diagnostic for "why didn't this button
        //      do anything?"
        //   2. Server log as an event -- the agent sees it in
        //      LIVE_LOG_TAIL on the next planning turn so it can
        //      adapt ("my last press of A had effect_score=0, the
        //      game didn't register it; try Start instead").
        //   3. 'agent-action-ack' event for the panel.
        this._handleAgentInputAck(msg);
        this._resolveAgentRequest(msg);
        break;
    }
  }

  _handleAgentInputAck(msg) {
    const trace = {
      // request_id is the server-minted (or fallback-local) identifier
      // the server uses to resolve its pending-ack dict. Always include
      // it so the delivery-guarantee layer doesn't time out a press
      // that actually succeeded.
      request_id: msg.request_id || '',
      button: msg.button,
      held_ms: Number(msg.held_ms) || 0,
      tick_count: Number(msg.tick_count) || 0,
      effect_score: Number(msg.effect_score) || 0,
    };
    // effect_score=0 means the canvas was BYTE-IDENTICAL before vs
    // after the press settled — the press silently dropped. Surface
    // that loudly so we don't keep flailing the same bad button.
    if (trace.effect_score === 0) {
      console.warn('[agent-bridge] input had no visible effect:', trace);
    } else {
      console.info('[agent-bridge] input ack:', trace);
    }
    // Server-side log entry. Rides the same "event" channel the rest
    // of the bridge uses; appears in LIVE_LOG_TAIL of the next agent
    // planning turn so the model can adapt.
    this._agentSend({
      kind: 'event',
      data: { event: 'input_ack', ...trace },
    });
    this._emit('agent-action-ack', trace);
  }

  // ── Game-agent session ────────────────────────────────────────────
  //
  // Opens a WebSocket to the Augmentum game-agent route, drives
  // periodic memory + canvas reads against the iframe, forwards
  // resolved values upstream, and routes incoming actions back
  // down to the iframe as simulated inputs. Lifecycle:
  //
  //   bridge.startGameAgent({sessionId, bridgeUrl, probesName,
  //                          tickHz, frameHz, mode})
  //   bridge.stopGameAgent()
  //
  // ``mode`` is 'co-pilot' (agent emits inputs alongside user) or
  // 'watch' (agent observes only; we still receive its action
  // messages but never forward them). The mode is mutable mid-session
  // via ``setAgentMode(mode)``.

  async startGameAgent(opts = {}) {
    if (this._agent) {
      throw new Error('game-agent already running on this bridge');
    }
    const { sessionId, bridgeUrl, probesName } = opts;
    if (!sessionId || !bridgeUrl) {
      throw new Error('startGameAgent: sessionId + bridgeUrl required');
    }
    const tickHz  = Number(opts.tickHz)  > 0 ? Number(opts.tickHz)  : 4;
    const frameHz = Number(opts.frameHz) >= 0 ? Number(opts.frameHz) : 0.5;
    const mode    = opts.mode === 'watch' ? 'watch' : 'co-pilot';

    // Probe preset is optional. When absent we run vision-only:
    // no RAM polling, agent reasons from frames + objective alone.
    // Per-game presets ship under augmentum/game_agent/probes/ and
    // are listed on GET /api/game-agent/probes/{name}; passing a
    // name fetches it here.
    let preset = null;
    if (probesName) {
      const presetResp = await this._fetch(
        `/api/game-agent/probes/${encodeURIComponent(probesName)}`,
      );
      if (!presetResp.ok) {
        throw new Error(`probe preset ${probesName} fetch failed: ${presetResp.status}`);
      }
      preset = await presetResp.json();
    }

    const agent = {
      sessionId,
      preset,
      mode,
      ws: null,
      tickTimer: null,
      frameTimer: null,
      pending: new Map(),       // request_id -> {resolve, reject, kind}
      lastSnapshot: {},         // probe_name -> JSON-serialised value
      requestSeq: 0,
      tickMs:  Math.max(50,  Math.round(1000 / Math.max(0.5, tickHz))),
      frameMs: frameHz > 0 ? Math.max(200, Math.round(1000 / frameHz)) : 0,
    };

    // Open WS first — if the route says no, we want to fail before
    // starting any pollers.
    agent.ws = new WebSocket(bridgeUrl);
    agent.ws.addEventListener('open', () => {
      this._emit('agent-status', { status: 'open', sessionId });
      // Memory tick only when we have a preset to read against.
      if (agent.preset) {
        agent.tickTimer = setInterval(() => this._agentTickMemory(), agent.tickMs);
      }
      if (agent.frameMs > 0) {
        agent.frameTimer = setInterval(() => this._agentTickFrame(), agent.frameMs);
      }
    });
    agent.ws.addEventListener('message', (e) => this._agentOnServerMessage(e));
    agent.ws.addEventListener('close', () => {
      this._emit('agent-status', { status: 'closed', sessionId });
      this.stopGameAgent();
    });
    agent.ws.addEventListener('error', () => {
      this._emit('agent-status', { status: 'error', sessionId });
    });

    this._agent = agent;
  }

  setAgentMode(mode) {
    if (!this._agent) return;
    this._agent.mode = (mode === 'watch') ? 'watch' : 'co-pilot';
    this._emit('agent-mode-changed', { mode: this._agent.mode });
  }

  stopGameAgent() {
    const a = this._agent;
    if (!a) return;
    if (a.tickTimer)  clearInterval(a.tickTimer);
    if (a.frameTimer) clearInterval(a.frameTimer);
    // Best-effort polite stop signal.
    try {
      if (a.ws && a.ws.readyState === WebSocket.OPEN) {
        a.ws.send(JSON.stringify({ kind: 'bye' }));
      }
    } catch (_e) { /* ignore */ }
    try { a.ws?.close(); } catch (_e) { /* ignore */ }
    for (const p of a.pending.values()) p.reject(new Error('agent stopped'));
    a.pending.clear();
    this._agent = null;
  }

  // Build a bus-address memory map from the core's region manifest +
  // the per-console layout. Self-heals: returns false (and keeps trying
  // on later ticks) until the core actually exposes ≥1 region, so a slow
  // boot doesn't permanently disable probing.
  async _ensureMemMap(a) {
    if (a.memMap && a.memMap.length) return true;
    if (a._mapBuilding) return false;   // a build is already in flight
    a._mapBuilding = true;
    try {
      let regions = [];
      try {
        regions = await this._agentRequest(PROTOCOL.AGENT_LIST_REGIONS, {});
      } catch (_e) {
        regions = [];   // old iframe without manifest support → legacy path
      }
      a.memMap = _buildMemoryMap(a.preset, regions);
      if (a.memMap.length && !a._mapLogged) {
        a._mapLogged = true;
        console.info(
          '[agent-bridge] memory map: ' +
          a.memMap.map((r) => `${r.region}@0x${r.base.toString(16)}(${r.size}B)`).join(' '),
        );
      }
    } finally {
      a._mapBuilding = false;
    }
    return !!(a.memMap && a.memMap.length);
  }

  // Read each named region once this tick (cached by caller). For regions
  // backed by a published memory-map descriptor, pass its bus start so the
  // iframe reads the descriptor's heap pointer FRESH (robust to WASM heap
  // growth). Named-key regions read via EmulatorJSGetMemoryData as before.
  async _readRegions(a, regionSet) {
    const out = {};
    const byName = {};
    for (const e of (a.memMap || [])) byName[e.region] = e;
    for (const region of regionSet) {
      const entry = byName[region];
      const payload = { region };
      if (entry && typeof entry.mapStart === 'number') payload.map_start = entry.mapStart;
      try {
        const b = await this._agentRequest(PROTOCOL.AGENT_READ_MEMORY, payload);
        if (b && b.length) out[region] = b;
      } catch (_e) { /* region unavailable this tick — skip */ }
    }
    return out;
  }

  async _agentTickMemory() {
    const a = this._agent;
    if (!a) return;
    if (a._ticking) return;             // avoid overlapping ticks
    a._ticking = true;
    try {
      if (!(await this._ensureMemMap(a))) return;
      const probes = a.preset.probes || [];

      // Pass 1: resolve each probe. A direct probe resolves its own bus
      // address; a deref probe first needs its pointer read, so we
      // resolve the pointer's location now and finish in pass 2.
      const need = new Set();
      const plans = [];
      const gridPlans = [];
      for (const probe of probes) {
        if (probe.type === 'grid' && probe.grid && typeof probe.grid.header_at === 'number') {
          // Grid probe: resolve its {width,height,map*} header now; the
          // window itself decodes in pass 3, after the anchor probes'
          // values are known.
          const hlen = ((probe.grid.map_ptr_offset == null ? 8 : probe.grid.map_ptr_offset) >>> 0) + 4;
          const hr = _resolveBus(a.memMap, probe.grid.header_at >>> 0, hlen);
          if (!hr) { plans.push(null); continue; }
          need.add(hr.region);
          const plan = { probe, gridHeader: hr, hlen };
          plans.push(plan);
          gridPlans.push(plan);
          continue;
        }
        if (probe.pointer && typeof probe.pointer.at === 'number') {
          const psize = probe.pointer.size || 4;
          const pr = _resolveBus(a.memMap, probe.pointer.at >>> 0, psize);
          if (!pr) { plans.push(null); continue; }
          need.add(pr.region);
          plans.push({ probe, ptr: pr, psize });
        } else {
          const r = _resolveBus(a.memMap, probe.address >>> 0, probe.length);
          if (!r) { plans.push(null); continue; }
          need.add(r.region);
          plans.push({ probe, direct: r });
        }
      }

      const bytes = await this._readRegions(a, need);

      // Pass 2: dereference pointers, resolve the final targets, and
      // gather any extra regions those targets land in.
      const need2 = new Set();
      for (const plan of plans) {
        if (!plan || !plan.ptr) continue;
        const buf = bytes[plan.ptr.region];
        if (!buf || plan.ptr.offset + plan.psize > buf.length) { plan.dead = true; continue; }
        const target = _readUintLE(buf, plan.ptr.offset, plan.psize);
        const finalAddr = ((target >>> 0) + (plan.probe.pointer.offset || 0)) >>> 0;
        const fr = _resolveBus(a.memMap, finalAddr, plan.probe.length);
        if (!fr) { plan.dead = true; continue; }
        plan.final = fr;
        need2.add(fr.region);
      }
      const extra = [...need2].filter((r) => !(r in bytes));
      if (extra.length) Object.assign(bytes, await this._readRegions(a, new Set(extra)));

      // Decode + diff + emit only changed probes. `decoded` keeps every
      // value from THIS tick (changed or not) so grid probes can anchor
      // on the player-position probes' current readings.
      const out = {};
      const decoded = {};
      let any = false;
      for (const plan of plans) {
        if (!plan || plan.dead) continue;
        const loc = plan.direct || plan.final;
        if (!loc) continue;
        const buf = bytes[loc.region];
        if (!buf || loc.offset < 0 || loc.offset + plan.probe.length > buf.length) continue;
        const slice = buf.subarray(loc.offset, loc.offset + plan.probe.length);
        const value = _decodeProbe(slice, plan.probe);
        decoded[plan.probe.name] = value;
        const serialised = JSON.stringify(value);
        if (a.lastSnapshot[plan.probe.name] !== serialised) {
          a.lastSnapshot[plan.probe.name] = serialised;
          out[plan.probe.name] = value;
          any = true;
        }
      }

      // Pass 3: grid probes (collision-map windows around an anchor).
      for (const plan of gridPlans) {
        const value = await this._agentDecodeGrid(a, bytes, plan, decoded);
        if (value === null) continue;
        const serialised = JSON.stringify(value);
        if (a.lastSnapshot[plan.probe.name] !== serialised) {
          a.lastSnapshot[plan.probe.name] = serialised;
          out[plan.probe.name] = value;
          any = true;
        }
      }
      if (any) this._agentSend({ kind: 'event', data: { event: 'ram', probes: out } });
    } finally {
      a._ticking = false;
    }
  }

  // Decode one grid probe: read the {width, height, map-pointer} header,
  // then sample a window of collision cells centred on the anchor
  // probes' values. Emits {x0, y0, rows} in the SAME coordinate space
  // as the anchors ('.'=walkable '#'=blocked '?'=outside the stored
  // grid). Returns null (honest skip) whenever any prerequisite —
  // header region, sane dimensions, anchor values, map region — is
  // unavailable this tick.
  async _agentDecodeGrid(a, bytes, plan, decoded) {
    const g = plan.probe.grid;
    const hbuf = bytes[plan.gridHeader.region];
    if (!hbuf || plan.gridHeader.offset + plan.hlen > hbuf.length) return null;
    const at = plan.gridHeader.offset;
    const w = _readUintLE(hbuf, at + ((g.width_offset == null ? 0 : g.width_offset) >>> 0), 4) | 0;
    const h = _readUintLE(hbuf, at + ((g.height_offset == null ? 4 : g.height_offset) >>> 0), 4) | 0;
    const mapPtr = _readUintLE(hbuf, at + ((g.map_ptr_offset == null ? 8 : g.map_ptr_offset) >>> 0), 4) >>> 0;
    if (w <= 0 || h <= 0 || w > 4096 || h > 4096) return null;
    const px = decoded[g.anchor_x];
    const py = decoded[g.anchor_y];
    if (typeof px !== 'number' || typeof py !== 'number') return null;
    const cell = g.cell_bytes || 2;
    const mr = _resolveBus(a.memMap, mapPtr, w * h * cell);
    if (!mr) return null;
    let mbuf = bytes[mr.region];
    if (!mbuf) {
      Object.assign(bytes, await this._readRegions(a, new Set([mr.region])));
      mbuf = bytes[mr.region];
      if (!mbuf) return null;
    }
    const border = g.border || 0;
    const win = g.window || 15;
    const half = win >> 1;
    const shift = g.collision_shift || 0;
    const mask = (g.collision_mask == null ? 3 : g.collision_mask);
    const rows = [];
    for (let dy = -half; dy <= half; dy++) {
      const gy = py + border + dy;
      let row = '';
      for (let dx = -half; dx <= half; dx++) {
        const gx = px + border + dx;
        if (gx < 0 || gy < 0 || gx >= w || gy >= h) { row += '?'; continue; }
        const off = mr.offset + (gy * w + gx) * cell;
        if (off + cell > mbuf.length) { row += '?'; continue; }
        const v = _readUintLE(mbuf, off, cell);
        row += (((v >> shift) & mask) === 0) ? '.' : '#';
      }
      rows.push(row);
    }
    return { x0: px - half, y0: py - half, rows };
  }

  async _agentTickFrame() {
    const a = this._agent;
    if (!a) return;
    let dataUrl;
    try {
      dataUrl = await this._agentRequest(PROTOCOL.AGENT_READ_FRAME, {});
    } catch (_e) {
      // Timeout (1 s, see _agentRequest) — iframe didn't reply. We
      // skip this tick and the next setInterval fire will try again.
      // The iframe's message handler is well-tested at session start
      // (READY round-trips successfully through the same channel), so
      // a steady stream of timeouts in practice points at the iframe
      // being torn down — already surfaced by the bridge-disconnect
      // path.
      return;
    }
    if (typeof dataUrl !== 'string') return;
    const comma = dataUrl.indexOf(',');
    if (comma < 0) {
      // Empty reply. Two normal causes during a session:
      //   1. Canvas not yet attached to the emulator (very first tick
      //      after WS open, before the core has run a frame).
      //   2. Iframe-side uniformity gate rejected the frame (canvas
      //      is still all-black / all-same-colour — see
      //      _isCanvasUniform in emulator-iframe-template.js).
      // Both clear on the next tick; nothing to log per-frame.
      a.uniformSkipCount = (a.uniformSkipCount || 0) + 1;
      if (a.uniformSkipCount === 10) {
        // 10 consecutive skips = ~20 s of no real frames. Surface a
        // single warning so operators can investigate (frozen core,
        // black ROM, etc.) without spamming on normal boot.
        console.warn(
          '[agent-bridge] no usable frame in ~20 s; canvas may be stuck',
        );
      }
      return;
    }
    const png_b64 = dataUrl.slice(comma + 1);
    a.uniformSkipCount = 0;
    if (!a.frameLogged) {
      a.frameLogged = true;
      console.info('[agent-bridge] first frame captured', png_b64.length, 'b64 chars');
    }
    this._agentSend({ kind: 'frame', png_b64 });
  }

  _agentOnServerMessage(ev) {
    const a = this._agent;
    if (!a) return;
    let msg;
    try { msg = JSON.parse(ev.data); } catch (_e) { return; }
    if (!msg || typeof msg !== 'object') return;

    // Companion speech: audio frame produced by the server-side
    // VoiceBridge in response to a non-empty ``plan.say``. Decoded
    // and queued for sequential playback so two utterances don't
    // overlap mid-sentence.
    if (msg.kind === 'audio' && typeof msg.bytes_b64 === 'string') {
      this._queueAgentSpeech(msg);
      return;
    }

    if (typeof msg.action === 'string') {
      if (a.mode === 'watch') {
        // Observe-only — still surface to listeners so the panel can
        // visualize what the agent *would* have done.
        this._emit('agent-action', { semantic: msg.action, duration_ms: msg.duration_ms, dispatched: false });
        return;
      }
      // Prefer the server-minted request_id so the server-side
      // ack-tracking pending dict resolves cleanly when the iframe's
      // AGENT_INPUT_ACK rides back. Falls back to a local seq when the
      // server didn't supply one (older sessions, mock surfaces).
      const requestId = (typeof msg.request_id === 'string' && msg.request_id)
        ? msg.request_id
        : ('sim-' + (++a.requestSeq));
      // Universal control schema (Phase G): when the server attached
      // a ComposedProfile, the WS payload carries the wire transport
      // kind + the wire-level code (libretro joypad id, etc.). The
      // iframe prefers them over its built-in name->id table so a new
      // game/console combo can ship without an iframe-side update.
      // ``button`` (the semantic name) is still forwarded for replay
      // logs and ``watch`` mode UI; the iframe ignores it when a
      // wire_code is present.
      const forwarded = {
        type: PROTOCOL.AGENT_SIMULATE_INPUT,
        button: msg.action,
        duration_ms: msg.duration_ms,
        request_id: requestId,
      };
      if (typeof msg.wire_kind === 'string' && msg.wire_kind) {
        forwarded.wire_kind = msg.wire_kind;
      }
      if (typeof msg.wire_code === 'number' || typeof msg.wire_code === 'string') {
        forwarded.wire_code = msg.wire_code;
      }
      // Chord parts: extra buttons the iframe holds simultaneously
      // with the primary (real-time games: run+jump). Sanitized to
      // the three fields the iframe consumes.
      if (Array.isArray(msg.chord) && msg.chord.length) {
        forwarded.chord = msg.chord.slice(0, 2).map((p) => ({
          button: typeof p.button === 'string' ? p.button : '',
          wire_kind: typeof p.wire_kind === 'string' ? p.wire_kind : undefined,
          wire_code: (typeof p.wire_code === 'number' || typeof p.wire_code === 'string')
            ? p.wire_code : undefined,
        }));
      }
      this._post(forwarded);
      this._emit('agent-action', { semantic: msg.action, duration_ms: msg.duration_ms, dispatched: true });
    }
  }

  // ── Companion audio playback ──────────────────────────────────
  //
  // The server pushes one ``{kind:'audio', bytes_b64, mime, utterance}``
  // frame per non-empty plan.say. We decode to a Blob, hand the URL
  // to a single ``<audio>`` element, and gate the queue so a fast
  // sequence of utterances plays back-to-back instead of layered.

  _queueAgentSpeech(msg) {
    const a = this._agent;
    if (!a) return;
    if (!a._speechQueue) a._speechQueue = [];
    const item = { msg };
    a._speechQueue.push(item);
    this._drainSpeechQueue();
  }

  _drainSpeechQueue() {
    const a = this._agent;
    if (!a || a._speechPlaying) return;
    const queue = a._speechQueue || [];
    if (!queue.length) return;
    const { msg } = queue.shift();
    a._speechPlaying = true;

    let blob;
    try {
      const bin = atob(msg.bytes_b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      blob = new Blob([bytes], { type: msg.mime || 'audio/mpeg' });
    } catch (_e) {
      a._speechPlaying = false;
      return this._drainSpeechQueue();
    }

    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);

    this._emit('agent-speak-start', { text: msg.utterance || '', audio });
    audio.addEventListener('ended', () => {
      URL.revokeObjectURL(url);
      this._emit('agent-speak-end', { text: msg.utterance || '' });
      a._speechPlaying = false;
      this._drainSpeechQueue();
    });
    audio.addEventListener('error', () => {
      URL.revokeObjectURL(url);
      this._emit('agent-speak-end', { text: msg.utterance || '' });
      a._speechPlaying = false;
      this._drainSpeechQueue();
    });
    audio.play().catch((err) => {
      // Autoplay policy may have blocked us. Surface the text via
      // the same event so the panel can still render the bubble;
      // the audio just won't make noise.
      console.warn('[agent] audio playback blocked:', err.message || err);
      this._emit('agent-speak-end', { text: msg.utterance || '' });
      a._speechPlaying = false;
      this._drainSpeechQueue();
    });
  }

  _agentSend(obj) {
    const a = this._agent;
    if (!a || !a.ws || a.ws.readyState !== WebSocket.OPEN) return;
    a.ws.send(JSON.stringify(obj));
  }

  _agentRequest(type, payload) {
    const a = this._agent;
    if (!a) return Promise.reject(new Error('no agent'));
    const requestId = type + '-' + (++a.requestSeq);
    return new Promise((resolve, reject) => {
      a.pending.set(requestId, { resolve, reject, kind: type });
      this._post({ type, request_id: requestId, ...payload });
      // 1s ceiling: the iframe is local; if it cannot answer in 1s
      // something is wrong and we shouldn't hang the next tick.
      setTimeout(() => {
        if (a.pending.has(requestId)) {
          a.pending.delete(requestId);
          reject(new Error('agent request timeout: ' + type));
        }
      }, 1000);
    });
  }

  _resolveAgentRequest(msg) {
    const a = this._agent;
    if (!a) return;
    const p = a.pending.get(msg.request_id);
    if (!p) return;
    a.pending.delete(msg.request_id);
    if (msg.type === PROTOCOL.AGENT_MEMORY_DATA) {
      try {
        p.resolve(_b64ToBytes(String(msg.bytes_b64 || '')));
      } catch (e) {
        p.reject(e);
      }
    } else if (msg.type === PROTOCOL.AGENT_FRAME_DATA) {
      p.resolve(String(msg.data_url || ''));
    } else if (msg.type === PROTOCOL.AGENT_REGIONS_DATA) {
      p.resolve(Array.isArray(msg.regions) ? msg.regions : []);
    } else if (msg.type === PROTOCOL.AGENT_INPUT_ACK) {
      p.resolve(true);
    } else {
      p.reject(new Error('unknown agent reply type: ' + msg.type));
    }
  }

  _post(payload) {
    if (!this._iframe?.contentWindow) return;
    // srcdoc inherits parent origin, so explicit origin is OK.
    this._iframe.contentWindow.postMessage(
      payload,
      this._win.location.origin,
    );
  }

  // ── Save proxy (parent → API) ─────────────────────────────────────

  async _handleStateSaved(msg) {
    const slot = Number(msg.slot);
    const dataB64 = String(msg.data_b64 || '');
    if (!dataB64 || !Number.isFinite(slot)) return;
    const coreId = String(this._handle.metadata?.core || '');
    const label = String(msg.label || '');
    await this._putSave('state', slot, dataB64, { core_id: coreId, label });
    this._emit('state-saved', { slot, label });
    // Screenshot (optional) lands in the same slot under kind=screenshot.
    if (msg.screenshot_b64) {
      await this._putSave('screenshot', slot, String(msg.screenshot_b64), {});
    }
  }

  async _handleSramSaved(msg) {
    const dataB64 = String(msg.data_b64 || '');
    if (!dataB64) return;
    await this._putSave('sram', 0, dataB64, {});
    this._emit('sram-saved', {});
  }

  async _putSave(kind, slot, dataB64, extra = {}) {
    const url = `/api/titles/${encodeURIComponent(this._titleId)}/saves/${kind}/${slot}`;
    const r = await this._fetch(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: dataB64, ...extra }),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.error || `HTTP ${r.status}`);
    }
  }

  // ── Save loading (parent → iframe) ────────────────────────────────

  async _sendSavesSnapshot() {
    // Pull the index then fetch any blobs we want to push to the iframe.
    // We always push SRAM (game's normal save) on launch -- the
    // emulator boots from it. State slots are loaded on demand when
    // the user clicks one in the UI; we just send the slot index here
    // so the iframe knows what's available.
    let saves = [];
    try {
      const r = await this._fetch(
        `/api/titles/${encodeURIComponent(this._titleId)}/saves`,
      );
      if (r.ok) {
        const body = await r.json();
        saves = Array.isArray(body.saves) ? body.saves : [];
      }
    } catch (_) { /* swallow -- empty snapshot is fine on first launch */ }

    const sramRecord = saves.find((s) => s.kind === 'sram' && s.slot === 0);
    let sramBytes = null;
    if (sramRecord) {
      const data = await this._fetchSaveBytes('sram', 0);
      if (data) sramBytes = _bytesToBase64(data);
    }

    const states = saves
      .filter((s) => s.kind === 'state')
      .map((s) => ({
        slot: s.slot, label: s.label,
        size_bytes: s.size_bytes, updated_at: s.updated_at,
      }));

    this._post({
      type: PROTOCOL.SAVES_SNAPSHOT,
      sram_b64: sramBytes,           // null if none
      state_index: states,           // metadata only; bodies fetched on demand
    });
  }

  async _fetchSaveBytes(kind, slot) {
    const url = `/api/titles/${encodeURIComponent(this._titleId)}/saves/${kind}/${slot}`;
    try {
      const r = await this._fetch(url);
      if (!r.ok) return null;
      const buf = await r.arrayBuffer();
      return new Uint8Array(buf);
    } catch (_) {
      return null;
    }
  }
}


// ── Internals shared with tests ─────────────────────────────────────


function _bytesToBase64(bytes) {
  // Convert Uint8Array → base64 without blowing the call stack on
  // big inputs. We chunk because btoa(String.fromCharCode(...arr))
  // can hit the JS arg-count limit for ROM-sized payloads.
  const CHUNK = 0x8000;
  let s = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    s += String.fromCharCode.apply(
      null, bytes.subarray(i, i + CHUNK),
    );
  }
  // btoa is unavailable in some non-browser environments; tests inject
  // a window with btoa. Fall back to a Buffer-based path when present.
  if (typeof btoa === 'function') return btoa(s);
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(s, 'binary').toString('base64');
  }
  throw new Error('no btoa or Buffer available');
}


function _b64ToBytes(s) {
  if (!s) return new Uint8Array(0);
  if (typeof atob === 'function') {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  if (typeof Buffer !== 'undefined') {
    return new Uint8Array(Buffer.from(s, 'base64'));
  }
  throw new Error('no atob or Buffer available');
}


// ── Probe decoders ──────────────────────────────────────────────────
//
// Match the Python ``Probe.type`` vocabulary in
// augmentum/game_agent/probes/. Adding a new decoder name here must
// be done in lockstep with adding it to the preset's ``type`` Literal
// in Python.

// Little-endian unsigned integer read, used for pointer dereferencing.
// >>> 0 keeps the result an unsigned 32-bit value (bus addresses).
function _readUintLE(bytes, off, size) {
  let v = 0;
  for (let i = 0; i < size; i++) v |= (bytes[off + i] << (8 * i));
  return v >>> 0;
}

// Build a bus-address memory map from the core's region manifest + the
// per-console layout. Pure so it can be unit-tested without a live core.
function _buildMemoryMap(preset, regions) {
  regions = regions || [];
  // If the core PUBLISHED its libretro memory map (get_memory_map export),
  // that's authoritative and complete for every console — build directly
  // from the descriptors (base=start, size=len) and read each via the
  // mapped-heap path. Overrides the per-console fallback below.
  const mapped = regions.filter(
    (r) => r && r.mapped && typeof r.start === 'number' && (r.size >>> 0) > 0,
  );
  if (mapped.length) {
    return mapped.map((r) => ({
      region: r.region || ('map@0x' + (r.start >>> 0).toString(16)),
      base: r.start >>> 0,
      size: r.size >>> 0,
      mapStart: r.start >>> 0,
    }));
  }
  const system = String(preset.system || '').toLowerCase();
  const layout = CONSOLE_BUS_LAYOUT[system];
  const map = [];
  for (const r of regions || []) {
    const name = r.region;
    const size = (r.size >>> 0);
    if (!size) continue;
    let base = null;
    const spec = layout && layout[name];
    if (Array.isArray(spec)) {
      const hit = spec.find((s) => s.size === size);
      base = (hit || spec[0]).base;
    } else if (spec && typeof spec.base === 'number') {
      base = spec.base;
    }
    // Fall back to the preset's declared single region base.
    if (base === null && name === preset.memory_region
        && typeof preset.region_base_address === 'number') {
      base = preset.region_base_address >>> 0;
    }
    if (base === null) continue;   // unknown region on unknown system
    map.push({ region: name, base: base >>> 0, size });
  }
  // Legacy fallback: manifest empty (old core/iframe) but the preset
  // declares a flat region → synthesise one entry. The generous size
  // defers bounds-checking to decode-time (which already clamps).
  if (!map.length && preset.memory_region
      && typeof preset.region_base_address === 'number') {
    map.push({ region: preset.memory_region,
               base: preset.region_base_address >>> 0, size: 0x7fffffff });
  }
  return map;
}

// Resolve an absolute bus address to (region, offset), or null if no
// exposed region covers it (e.g. an EWRAM address on a core that only
// exposes IWRAM — the probe is skipped, honestly, rather than reading
// the wrong region).
function _resolveBus(memMap, addr, length) {
  for (const r of memMap) {
    if (addr >= r.base && addr + length <= r.base + r.size) {
      return { region: r.region, offset: addr - r.base };
    }
  }
  return null;
}

function _decodeProbe(bytes, probe) {
  switch (probe.type) {
    case 'u8':    return bytes[0];
    case 'u16le': return bytes[0] | (bytes[1] << 8);
    case 'u16be': return (bytes[0] << 8) | bytes[1];
    case 's16le': { const v = bytes[0] | (bytes[1] << 8); return v >= 0x8000 ? v - 0x10000 : v; }
    case 's16be': { const v = (bytes[0] << 8) | bytes[1]; return v >= 0x8000 ? v - 0x10000 : v; }
    case 'u32le': {
      const v = _readUintLE(bytes, 0, 4);
      if (probe.value_labels) {
        // Function-pointer values carry the ARM Thumb bit — mask it so
        // the lookup matches the symbol address. Unknown values render
        // as hex: still a legible "state changed" signal.
        const key = (v & ~1) >>> 0;
        return probe.value_labels[String(key)] || ('0x' + key.toString(16));
      }
      return v;
    }
    case 'bcd3':  return _bcdNibbles(bytes[0]) * 10000
                       + _bcdNibbles(bytes[1]) * 100
                       + _bcdNibbles(bytes[2]);
    case 'bitfield8': {
      const out = {};
      const v = bytes[0];
      const labels = probe.labels || [];
      for (let i = 0; i < 8; i++) {
        const label = labels[i] || ('bit' + i);
        if (label.startsWith('_')) continue;
        out[label] = !!(v & (1 << i));
      }
      return out;
    }
    case 'raw':   return Array.from(bytes);
    case 'text':  return _decodeText(bytes, probe.charmap || 'ascii');
    default:      return null;
  }
}

function _bcdNibbles(byte) {
  return ((byte >> 4) & 0x0F) * 10 + (byte & 0x0F);
}

// ── In-game text decoding ("translate game dialogue") ──────────────────
//
// Retro games store dialogue in proprietary character encodings, not
// ASCII. A `text`-type probe names its `charmap` and the decoder below
// turns the raw buffer into a readable string the agent can treat as
// world lore (tutorial text literally teaches the controls). Decoding
// stops at the charmap's terminator; unmappable bytes render as nothing
// (control codes) so the output stays clean prose.
//
// gen3: Gen-III Pokémon (Ruby/Sapphire/Emerald/FRLG) Latin charset.
//       Reference: pret/pokeemerald charmap.inc. 0xFF terminates,
//       0xFE = line break, 0xFB = paragraph scroll, 0x00 = space.
const _GEN3_CHARS = (() => {
  const m = {};
  m[0x00] = ' ';
  // 0xA1-0xAA: digits 0-9
  for (let i = 0; i <= 9; i++) m[0xA1 + i] = String.fromCharCode(48 + i);
  // 0xBB-0xD4: A-Z
  for (let i = 0; i < 26; i++) m[0xBB + i] = String.fromCharCode(65 + i);
  // 0xD5-0xEE: a-z
  for (let i = 0; i < 26; i++) m[0xD5 + i] = String.fromCharCode(97 + i);
  Object.assign(m, {
    0xAB: '!', 0xAC: '?', 0xAD: '.', 0xAE: '-', 0xB0: '…',
    0xB1: '“', 0xB2: '”', 0xB3: '‘', 0xB4: '’', 0xB5: '♂', 0xB6: '♀',
    0xB7: '$', 0xB8: ',', 0xB9: '×', 0xBA: '/',
    0xF0: ':', 0x2D: '&', 0x2E: '+', 0x1B: 'é',
    0xFE: '\n', 0xFB: '\n',
  });
  return m;
})();

function _decodeText(bytes, charmap) {
  let out = '';
  if (charmap === 'gen3') {
    for (let i = 0; i < bytes.length; i++) {
      const b = bytes[i];
      if (b === 0xFF) break;                 // EOS
      if (b === 0xFC || b === 0xFD) { i += 1; continue; } // ctrl/placeholder + arg
      const ch = _GEN3_CHARS[b];
      if (ch !== undefined) out += ch;
    }
  } else {
    // ascii: printable range only, NUL-terminated.
    for (let i = 0; i < bytes.length; i++) {
      const b = bytes[i];
      if (b === 0x00) break;
      if (b >= 0x20 && b < 0x7F) out += String.fromCharCode(b);
      else if (b === 0x0A) out += '\n';
    }
  }
  // Collapse whitespace runs so diffing is stable across scroll frames.
  return out.replace(/[ \t]+/g, ' ').replace(/\n{2,}/g, '\n').trim();
}


// Test surface (intentionally not on the public API)
export const __test = {
  _bytesToBase64, _b64ToBytes, _decodeProbe, _readUintLE, _decodeText,
  _buildMemoryMap, _resolveBus, CONSOLE_BUS_LAYOUT, PROTOCOL,
};
