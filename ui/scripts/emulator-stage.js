// ui/scripts/emulator-stage.js
//
// Player surface for emulator-rom titles. Mirrors the API of
// ``game-surface.js`` (``openEmulatorStage(artifact)`` to mount,
// ``closeEmulatorStage()`` to tear down) so the library can dispatch
// uniformly. Internally it:
//
//   1. POSTs ``/api/titles/{id}/launch`` to get a LaunchHandle
//   2. Mounts an EmulatorBridge into a fullscreen overlay
//   3. Wires save events to toasts so the user sees "Saved slot 3 ✓"
//   4. Calls ``/api/titles/{id}/runs/{run_id}/end`` on close so the
//      title_runs row gets a clean exit_reason

import { showToast } from './app.js';
import { createAgentPanel } from './agent/agent-panel.js';
import { EmulatorBridge } from './emulator-bridge.js';

let _state = {
  overlay: null,
  bridge: null,
  artifact: null,
  runId: null,
  closeHandler: null,
  // Chrome state
  paused: false,
  volume: 1,
  activeSlot: 1,        // last save/load slot the user touched
  frameSkip: 'auto',    // 'auto' | '0' | '1' | '2' | '4'
  pausedOverlay: null,
  savesPopover: null,
  loadsPopover: null,
  volumePopover: null,
  settingsPopover: null,
  controlBar: null,
  iframe: null,
  // Side-by-side game-agent panel (created at mount, destroyed on close).
  agentPanel: null,
};

const _SAVE_SLOTS = [1, 2, 3, 4, 5, 6, 7, 8, 9];

// SVG icons sized for the 32px chrome buttons. Single-color so we
// can theme with currentColor.
const _ICON = {
  fullscreen: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3"/></svg>',
  fullscreenExit: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3v3a2 2 0 0 1-2 2H3M21 8h-3a2 2 0 0 1-2-2V3M3 16h3a2 2 0 0 1 2 2v3M16 21v-3a2 2 0 0 1 2-2h3"/></svg>',
  pause: '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/></svg>',
  play: '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M8 5v14l11-7L8 5z"/></svg>',
  reset: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg>',
  save: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>',
  load: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17v3a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-3"/><polyline points="8 12 12 16 16 12"/><line x1="12" y1="2" x2="12" y2="16"/></svg>',
  volume: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>',
  volumeMute: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>',
  close: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
  settings: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
};


// ── Public API ──────────────────────────────────────────────────────


export async function openEmulatorStage(artifact, opts = {}) {
  // ``opts.startWithAgent`` (boolean) wires the panel into partner
  // mode: panel mounts expanded, mode defaults to co-pilot, agent
  // auto-starts once the bridge fires 'ready'. Used by the
  // Launch-with-Partner button in the library card chooser.
  //
  // ``opts.kiosk`` (boolean) strips the chrome bar + agent panel for
  // headless / cast-receiver embeds. The bridge still mounts so the
  // game plays; only the player-facing UI is suppressed. ESC + Space
  // keyboard fallbacks remain wired so a hardware keyboard on the TV
  // can still pause / close (most TV setups don't have one — the
  // cast controller phone closes via the WS instead).
  if (!artifact || !artifact.id) {
    showToast('Cannot open: missing artifact', 'error');
    return;
  }
  const kiosk = !!opts.kiosk;
  // If a previous emulator is still up, tear it down first. Avoids
  // two iframes fighting for the GamePad API or save endpoints.
  if (_state.bridge) {
    await closeEmulatorStage('replaced');
  }
  _state.artifact = artifact;

  // 1) Launch -- this allocates a title_runs row server-side and
  //    returns the runtime handle the bridge consumes.
  let launched;
  try {
    const r = await fetch(
      `/api/titles/${encodeURIComponent(artifact.id)}/launch`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' },
    );
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${r.status}`);
    }
    launched = await r.json();
  } catch (e) {
    showToast(`Couldn't launch ${artifact.display_name || artifact.title || 'title'}: ${e.message}`, 'error');
    _state.artifact = null;
    return;
  }
  _state.runId = launched.run_id;

  // Streaming-runtime handoff. ROMs whose system is marked
  // streaming_required (gamecube, wii — Dolphin) come back from
  // /launch with a webrtc-kind handle pointing at an AGSP session.
  // EmulatorJS isn't applicable; bail out of the EmulatorBridge
  // path and mount stream-stage instead. The session was already
  // created server-side by AgspStreamedRuntime, so we pass it as
  // ``preStarted`` to skip stream-stage's find-or-create logic.
  const handle = launched.handle || {};
  if (handle.kind === 'webrtc') {
    const md = handle.metadata || {};
    const profile = {
      id: md.profile_id || 'emulator-streamed',
      display_name: artifact.display_name || artifact.title || 'Streamed Emulator',
    };
    const preStarted = {
      session_id: handle.session_id,
      profile_id: md.profile_id,
      status: 'starting',
      stream_port: md.stream_port,
      game_port: md.game_port,
      bitrate_mbps: md.bitrate_mbps,
      resolution: md.resolution,
    };
    // Drop emulator-stage's own state — stream-stage owns the
    // overlay, lifecycle, and close handlers from here.
    _state.artifact = null;
    _state.runId = null;
    try {
      const m = await import('./stream-stage.js');
      await m.openStreamStage(profile, { preStarted });
    } catch (err) {
      console.error('Stream stage open failed:', err);
      showToast(`Failed to open streaming session: ${err.message || 'Unknown error'}`, 'error');
    }
    return;
  }

  // 2) Build the overlay. Fullscreen-bg, professional chrome bar
  //    pattern (RetroArch / OpenEmu reference): system badge +
  //    title on the left, control icon row on the right, with
  //    popovers for save/load slot pickers and a volume slider.
  const overlay = document.createElement('div');
  overlay.className = 'emulator-stage-overlay';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-label', 'Emulator');

  // Kiosk mode skips the chrome bar entirely — receiver-side embed
  // wants the game pixel-for-pixel with no player UI.
  let bar = null;
  if (!kiosk) {
    bar = _buildControlBar(artifact);
    _state.controlBar = bar;
  }

  const slot = document.createElement('div');
  slot.className = 'emulator-stage-slot';
  if (kiosk) slot.classList.add('is-kiosk');

  // Pause overlay -- centered icon over a dim layer. Hidden by
  // default; toggled when the bridge reports paused-changed.
  const pausedOverlay = document.createElement('div');
  pausedOverlay.className = 'emulator-stage-paused-overlay';
  pausedOverlay.innerHTML = `
    <div class="emulator-stage-paused-icon">${_ICON.pause}</div>
    <div class="emulator-stage-paused-label">Paused</div>
  `;
  pausedOverlay.style.display = 'none';
  _state.pausedOverlay = pausedOverlay;
  slot.appendChild(pausedOverlay);

  // Side-by-side play area: emulator slot on the left, game-agent
  // panel on the right. Wrapping slot in a flex parent lets the
  // bridge keep mounting the iframe inside ``slot`` unchanged, while
  // the agent panel claims a fixed-width column next to it. Kiosk
  // mode drops the wrapper and the agent panel — the slot owns the
  // full viewport.
  const playArea = document.createElement('div');
  playArea.className = 'emulator-stage-play-area';
  if (kiosk) playArea.classList.add('is-kiosk');
  playArea.appendChild(slot);

  if (bar) overlay.appendChild(bar);
  overlay.appendChild(playArea);
  if (kiosk) overlay.classList.add('is-kiosk');
  document.body.appendChild(overlay);
  _state.overlay = overlay;

  // Close any popover when clicking outside it (in the bar OR slot).
  overlay.addEventListener('click', (e) => {
    const inPopover = e.target.closest('.emulator-stage-popover');
    const onTrigger = e.target.closest('[data-popover-trigger]');
    if (!inPopover && !onTrigger) _closeAllPopovers();
  });

  // ESC closes the stage; F11 / 'f' toggles fullscreen.
  _state.closeHandler = (e) => {
    if (e.key === 'Escape' && !document.fullscreenElement) {
      // Don't fight Chrome's native ESC for fullscreen exit; only
      // close when not in fullscreen.
      closeEmulatorStage('user');
      return;
    }
    if ((e.key === 'F11' || (e.key === 'f' && (e.ctrlKey || e.metaKey)))
        && !_isInputFocused()) {
      e.preventDefault();
      _toggleFullscreen();
    }
    if (e.key === ' ' && !_isInputFocused()
        && document.activeElement === document.body) {
      // Space toggles pause when game frame has focus.
      e.preventDefault();
      _togglePaused();
    }
  };
  document.addEventListener('keydown', _state.closeHandler);

  // Track fullscreen state to swap the icon.
  document.addEventListener('fullscreenchange', _onFullscreenChange);

  // 3) Mount the bridge.
  const bridge = new EmulatorBridge(slot, launched.handle, artifact.id);
  _state.bridge = bridge;

  // 3b) Mount the game-agent side panel. The panel reads the bridge
  //     for ``startGameAgent`` / ``stopGameAgent`` and ``setAgentMode``.
  //     System lookup comes from the launch handle metadata so probe
  //     presets pick themselves; unsupported systems render a quiet
  //     "not wired yet" note.
  //
  //     Kiosk mode (cast receiver embed) suppresses the panel entirely —
  //     no AI overlay on the TV, and the receiver has no input surface
  //     to drive it anyway. The bridge still mounts so the game plays.
  if (!kiosk) {
    const systemId = (launched && launched.handle &&
                      launched.handle.metadata && launched.handle.metadata.system) || '';
    const agentPanel = createAgentPanel({
      titleId: artifact.id,
      system: systemId,
      bridge,
      startWithAgent: !!opts.startWithAgent,
      // Carries through to the session POST as ``character_id``; the
      // server uses it to load persona + voice for companion mode.
      // ``null`` (or omitted) means anonymous companion.
      characterId: opts.characterId || null,
    });
    playArea.appendChild(agentPanel.element);
    _state.agentPanel = agentPanel;

    // Mobile scrim — tap to collapse the panel. Only visible on phone
    // when the panel is expanded (CSS gates on data-panel-state). Lives
    // as a sibling of the panel so the scrim covers the iframe area.
    const scrim = document.createElement('div');
    scrim.className = 'agent-panel-scrim';
    scrim.setAttribute('aria-hidden', 'true');
    scrim.addEventListener('click', () => {
      if (!agentPanel.element.classList.contains('agent-panel-collapsed')) {
        agentPanel.element.querySelector('.agent-panel-toggle')?.click();
      }
    });
    playArea.appendChild(scrim);
  }
  bridge.on('state-saved', ({ slot: s }) => {
    showToast(`Saved slot ${s}`, 'info');
  });
  bridge.on('sram-saved', () => {
    // SRAM saves fire often (every cartridge write). Don't toast.
  });
  bridge.on('paused-changed', ({ paused }) => {
    _state.paused = !!paused;
    _updatePauseUI();
  });
  bridge.on('error', ({ message, recoverable }) => {
    showToast(
      `Emulator: ${message}${recoverable ? '' : ' (fatal)'}`,
      recoverable ? 'warn' : 'error',
    );
  });

  try {
    await bridge.mount();
    // Capture the iframe for fullscreen targeting -- after mount(),
    // the bridge has appended exactly one <iframe> to the slot.
    _state.iframe = slot.querySelector('iframe');
  } catch (e) {
    showToast(`Bootstrap failed: ${e.message}`, 'error');
    await closeEmulatorStage('crash');
  }
}


export async function closeEmulatorStage(reason = 'user') {
  if (_state.agentPanel) {
    try { _state.agentPanel.destroy(); } catch (_) {}
    _state.agentPanel = null;
  }
  if (_state.bridge) {
    try { _state.bridge.unmount(); } catch (_) {}
  }
  if (_state.closeHandler) {
    document.removeEventListener('keydown', _state.closeHandler);
  }
  document.removeEventListener('fullscreenchange', _onFullscreenChange);
  if (document.fullscreenElement) {
    try { document.exitFullscreen(); } catch (_) {}
  }
  if (_state.overlay && _state.overlay.parentNode) {
    _state.overlay.parentNode.removeChild(_state.overlay);
  }

  // End the run server-side so the title_runs row closes cleanly.
  if (_state.artifact && _state.runId) {
    try {
      await fetch(
        `/api/titles/${encodeURIComponent(_state.artifact.id)}/runs/${encodeURIComponent(_state.runId)}/end`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            runtime_id: 'emulator-browser',
            exit_reason: reason === 'user' ? 'clean'
                       : reason === 'replaced' ? 'abandon'
                       : reason === 'crash' ? 'crash'
                       : 'clean',
          }),
        },
      );
    } catch (_) { /* best-effort */ }
  }

  _state = {
    overlay: null,
    bridge: null,
    artifact: null,
    runId: null,
    closeHandler: null,
    paused: false,
    volume: 1,
    activeSlot: 1,
    pausedOverlay: null,
    savesPopover: null,
    loadsPopover: null,
    controlBar: null,
    iframe: null,
  };
}


// ── Chrome construction ────────────────────────────────────────────


function _buildControlBar(artifact) {
  const meta = artifact?.metadata || {};
  const systemLabel = meta.system_label || meta.system_id || meta.system || '';
  const titleText = artifact.display_name || artifact.title || 'Emulator';

  const bar = document.createElement('div');
  bar.className = 'emulator-stage-bar';

  // Left: system badge + title
  const left = document.createElement('div');
  left.className = 'emulator-stage-bar-left';
  if (systemLabel) {
    const badge = document.createElement('span');
    badge.className = 'emulator-stage-system-badge';
    badge.textContent = systemLabel.toUpperCase();
    left.appendChild(badge);
  }
  const title = document.createElement('span');
  title.className = 'emulator-stage-title';
  title.textContent = titleText;
  left.appendChild(title);
  bar.appendChild(left);

  // Right: control row
  const right = document.createElement('div');
  right.className = 'emulator-stage-bar-right';

  // Save (popover)
  const saveBtn = _iconButton(_ICON.save, 'Save state (S)');
  saveBtn.dataset.popoverTrigger = 'save';
  saveBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    _toggleSavesPopover(saveBtn);
  });
  right.appendChild(saveBtn);

  // Load (popover)
  const loadBtn = _iconButton(_ICON.load, 'Load state (L)');
  loadBtn.dataset.popoverTrigger = 'load';
  loadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    _toggleLoadsPopover(loadBtn);
  });
  right.appendChild(loadBtn);

  // Pause/Play
  const pauseBtn = _iconButton(_ICON.pause, 'Pause (Space)');
  pauseBtn.dataset.role = 'pause';
  pauseBtn.addEventListener('click', _togglePaused);
  right.appendChild(pauseBtn);

  // Reset
  const resetBtn = _iconButton(_ICON.reset, 'Reset game');
  resetBtn.addEventListener('click', () => {
    if (!confirm('Reset the game? Unsaved progress will be lost.')) return;
    _state.bridge?.reset();
    showToast('Game reset.', 'info');
  });
  right.appendChild(resetBtn);

  // Volume (popover with slider)
  const volBtn = _iconButton(_ICON.volume, 'Volume');
  volBtn.dataset.popoverTrigger = 'volume';
  volBtn.dataset.role = 'volume';
  volBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    _toggleVolumePopover(volBtn);
  });
  right.appendChild(volBtn);

  // Settings (perf knobs + native menu opener)
  const settingsBtn = _iconButton(_ICON.settings, 'Settings');
  settingsBtn.dataset.popoverTrigger = 'settings';
  settingsBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    _toggleSettingsPopover(settingsBtn);
  });
  right.appendChild(settingsBtn);

  // Fullscreen
  const fsBtn = _iconButton(_ICON.fullscreen, 'Fullscreen (F11)');
  fsBtn.dataset.role = 'fullscreen';
  fsBtn.addEventListener('click', _toggleFullscreen);
  right.appendChild(fsBtn);

  // Close (kept distinct visually -- same shape as the rest but red on hover)
  const closeBtn = _iconButton(_ICON.close, 'Close (Esc)');
  closeBtn.classList.add('emulator-stage-close-btn');
  closeBtn.addEventListener('click', () => closeEmulatorStage('user'));
  right.appendChild(closeBtn);

  bar.appendChild(right);
  return bar;
}


function _iconButton(svg, title) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'emulator-stage-iconbtn';
  btn.title = title;
  btn.setAttribute('aria-label', title);
  btn.innerHTML = svg;
  return btn;
}


// ── Popovers ───────────────────────────────────────────────────────


function _closeAllPopovers() {
  for (const key of ['savesPopover', 'loadsPopover', 'volumePopover', 'settingsPopover']) {
    const p = _state[key];
    if (p?.parentNode) p.parentNode.removeChild(p);
    _state[key] = null;
  }
}


function _renderSlotGrid(action) {
  // Slot 1-9 grid. Click → save or load that slot.
  const grid = document.createElement('div');
  grid.className = 'emulator-stage-slot-grid';
  for (const slot of _SAVE_SLOTS) {
    const cell = document.createElement('button');
    cell.type = 'button';
    cell.className = 'emulator-stage-slot-cell';
    if (slot === _state.activeSlot) cell.classList.add('is-active');
    cell.textContent = String(slot);
    cell.addEventListener('click', async () => {
      _state.activeSlot = slot;
      _closeAllPopovers();
      if (action === 'save') {
        _state.bridge?.saveState(slot);
      } else {
        const ok = await _state.bridge?.loadState(slot);
        if (!ok) showToast(`No save in slot ${slot}.`, 'warn');
      }
    });
    grid.appendChild(cell);
  }
  return grid;
}


function _toggleSavesPopover(anchor) {
  if (_state.savesPopover) { _closeAllPopovers(); return; }
  _closeAllPopovers();
  const pop = _popover(anchor, 'Save to slot');
  pop.appendChild(_renderSlotGrid('save'));
  _state.savesPopover = pop;
}


function _toggleLoadsPopover(anchor) {
  if (_state.loadsPopover) { _closeAllPopovers(); return; }
  _closeAllPopovers();
  const pop = _popover(anchor, 'Load from slot');
  pop.appendChild(_renderSlotGrid('load'));
  _state.loadsPopover = pop;
}


function _toggleVolumePopover(anchor) {
  if (_state.volumePopover) { _closeAllPopovers(); return; }
  _closeAllPopovers();
  const pop = _popover(anchor, 'Volume');
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = '100';
  slider.value = String(Math.round(_state.volume * 100));
  slider.className = 'emulator-stage-volume-slider';
  slider.addEventListener('input', () => {
    const v = Number(slider.value) / 100;
    _state.volume = v;
    _state.bridge?.setVolume(v);
    _updateVolumeIcon();
  });
  pop.appendChild(slider);
  _state.volumePopover = pop;
}


function _toggleSettingsPopover(anchor) {
  if (_state.settingsPopover) { _closeAllPopovers(); return; }
  _closeAllPopovers();
  const pop = _popover(anchor, 'Settings');
  pop.classList.add('emulator-stage-popover-settings');

  // Frame skip row — biggest stutter knob on phones. "Auto" lets the
  // core decide; higher numbers drop more frames in exchange for
  // smoother input/audio.
  const fsRow = document.createElement('div');
  fsRow.className = 'emulator-stage-setting-row';
  fsRow.innerHTML = `
    <label class="emulator-stage-setting-label" for="emu-frameskip">Frame skip</label>
    <select id="emu-frameskip" class="emulator-stage-setting-select">
      <option value="auto">Auto</option>
      <option value="0">Off</option>
      <option value="1">1</option>
      <option value="2">2</option>
      <option value="4">4 (max)</option>
    </select>
  `;
  const fsSelect = fsRow.querySelector('select');
  fsSelect.value = _state.frameSkip;
  fsSelect.addEventListener('change', () => {
    _state.frameSkip = fsSelect.value;
    _state.bridge?.setPerf({ frame_skip: _state.frameSkip });
    showToast(`Frame skip: ${_state.frameSkip}`, 'info');
  });
  pop.appendChild(fsRow);

  const sep = document.createElement('div');
  sep.className = 'emulator-stage-setting-sep';
  pop.appendChild(sep);

  // Escape hatch: open EmulatorJS's full native settings panel.
  // Secondary styling — the perf knob above is the primary action.
  // The native menu is for the "I need to tweak shaders/controls/
  // cheats" case where the popover's curated options aren't enough.
  const openBtn = document.createElement('button');
  openBtn.type = 'button';
  openBtn.className = 'emulator-stage-setting-btn is-secondary';
  openBtn.textContent = 'Open all settings…';
  openBtn.addEventListener('click', () => {
    _state.bridge?.openNativeMenu();
    _closeAllPopovers();
  });
  pop.appendChild(openBtn);

  const hint = document.createElement('div');
  hint.className = 'emulator-stage-setting-hint';
  hint.textContent = 'Stutter on first launch is normal — cores warm up over the first few seconds.';
  pop.appendChild(hint);

  _state.settingsPopover = pop;
}


function _popover(anchor, label) {
  const rect = anchor.getBoundingClientRect();
  const pop = document.createElement('div');
  pop.className = 'emulator-stage-popover';
  pop.style.top = `${rect.bottom + 6}px`;
  pop.style.right = `${window.innerWidth - rect.right}px`;
  if (label) {
    const lab = document.createElement('div');
    lab.className = 'emulator-stage-popover-label';
    lab.textContent = label;
    pop.appendChild(lab);
  }
  pop.addEventListener('click', (e) => e.stopPropagation());
  _state.overlay.appendChild(pop);
  return pop;
}


// ── Action handlers ────────────────────────────────────────────────


function _togglePaused() {
  if (!_state.bridge) return;
  const next = !_state.paused;
  _state.bridge.setPaused(next);
  // Optimistic UI update; the bridge will echo the actual state via
  // 'paused-changed' which calls _updatePauseUI() again.
  _state.paused = next;
  _updatePauseUI();
}


function _updatePauseUI() {
  if (_state.pausedOverlay) {
    _state.pausedOverlay.style.display = _state.paused ? 'flex' : 'none';
  }
  const btn = _state.controlBar?.querySelector('[data-role="pause"]');
  if (btn) {
    btn.innerHTML = _state.paused ? _ICON.play : _ICON.pause;
    btn.title = _state.paused ? 'Resume (Space)' : 'Pause (Space)';
  }
}


function _toggleFullscreen() {
  const target = _state.overlay;
  if (!target) return;
  if (document.fullscreenElement) {
    document.exitFullscreen?.();
  } else {
    target.requestFullscreen?.();
  }
}


function _onFullscreenChange() {
  const btn = _state.controlBar?.querySelector('[data-role="fullscreen"]');
  if (!btn) return;
  btn.innerHTML = document.fullscreenElement ? _ICON.fullscreenExit : _ICON.fullscreen;
  btn.title = document.fullscreenElement ? 'Exit fullscreen (F11)' : 'Fullscreen (F11)';
}


function _updateVolumeIcon() {
  const btn = _state.controlBar?.querySelector('[data-role="volume"]');
  if (!btn) return;
  btn.innerHTML = _state.volume === 0 ? _ICON.volumeMute : _ICON.volume;
}


function _isInputFocused() {
  const a = document.activeElement;
  if (!a) return false;
  const tag = (a.tagName || '').toLowerCase();
  return tag === 'input' || tag === 'textarea' || a.isContentEditable;
}
