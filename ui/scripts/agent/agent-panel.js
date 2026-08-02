// ui/scripts/agent/agent-panel.js
//
// Side-by-side game-agent panel for the emulator stage.
//
// Mounts next to the EmulatorJS iframe. Default mode is "Off" — the
// user plays the game alone and sees the panel idle. When the user
// hits Start, the panel:
//
//   1. POSTs /api/game-agent/sessions with a sensible default for
//      this title's system (e.g. Pokémon RBY for Game Boy).
//   2. Connects the bridge WebSocket via the parent EmulatorBridge
//      (which forwards memory + canvas reads to the iframe and routes
//      action commands back as simulateInput calls).
//   3. Streams the session's NDJSON log over Server-Sent Events and
//      renders observations, the agent's scratchpad, and recent
//      button presses.
//
// Modes:
//   off       — no session running.
//   watch     — bridge open, agent reasons, but inputs are not
//               forwarded into the emulator (visible-only for the
//               user).
//   co-pilot  — bridge open, agent inputs are forwarded. User
//               keyboard still works; both share the same emulator.
//
// Multiplayer is *not* wired yet — the panel and bridge are designed
// so a future second-session column slots in next to this one without
// changing this file's data flow.

import { showToast } from '../app.js';
import { getWsTicket } from '../auth.js';

/**
 * Whether the agent has a configured default for this libretro system.
 *
 * Drives the Launch / Launch-with-Partner chooser: systems without an
 * entry (e.g. PSP, PSX, arcade) get the partner path hidden so users
 * are not offered something that would render a quiet "not wired"
 * panel.
 *
 * @param {string} system - libretro system id (e.g. "gb", "gba")
 * @returns {boolean}
 */
export function isAgentSupported(system) {
  return Object.prototype.hasOwnProperty.call(_SYSTEM_DEFAULTS, system);
}


// Per-system defaults. Add a new entry when you ship a new probe preset.
//
// ``probesName`` may be null — that means "vision-only": no RAM
// polling, the agent reasons from frames alone. Vision-only is the
// safe default for systems whose RAM maps are dynamically allocated
// (Pokémon Gen 3+ uses pointer-chased SaveBlocks that move between
// ROM revisions), or simply not yet mapped to a Python preset.
//
// ``controllerProfile`` + ``gameProfile`` opt into the universal
// control schema (Phase G). When both are non-null the session POST
// carries the ids and the server composes them into a ComposedProfile;
// the agent's vocabulary becomes the profile's Layer-1 action names
// (``confirm`` / ``nav_up`` / ...) instead of raw button letters, and
// the iframe receives wire_code on every press so it can dispatch
// directly to the libretro core without re-resolving names.
//
// Leave them ``null`` for systems we haven't shipped a profile for
// yet (e.g., NDS) — the legacy ``semanticInputs`` path still works.
const _SYSTEM_DEFAULTS = {
  gb: {
    probesName: 'pokemon_rby',
    logSchema: 'pokemon_rby.v1',
    semanticInputs: ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select'],
    controllerProfile: 'gambatte',
    gameProfile: 'pokemon_rby',
    sampleObjective: 'Walk south out of the starting town, then explore. Advance any textbox you see.',
  },
  gbc: {
    probesName: 'pokemon_rby',
    logSchema: 'pokemon_rby.v1',
    semanticInputs: ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select'],
    controllerProfile: 'gambatte',
    gameProfile: 'pokemon_rby',
    sampleObjective: 'Walk south out of the starting town, then explore. Advance any textbox you see.',
  },
  // GBA: Pokémon Emerald RAM preset.
  //
  // Emerald Alloc()-es its SaveBlocks at runtime and randomises their
  // location, so player position is read by dereferencing gSaveBlock1Ptr
  // (0x03005D8C, IWRAM) — the bridge's generic pointer-deref primitive —
  // rather than a fixed address. Party data is at fixed EWRAM symbols.
  // The preset surfaces player x/y, current map (group, num), party
  // count, and lead Pokémon level + HP: the overlay the LLM reasons over
  // alongside frames. The final EWRAM reads light up once the core
  // exposes EWRAM (see the memory-map export work); the IWRAM pointer
  // read already works. Targets US (BPEE) revisions. (Ruby/Sapphire
  // keep static SaveBlocks — use the pokemon_rs preset for those ROMs.)
  gba: {
    probesName: 'pokemon_emerald',
    logSchema: 'pokemon_emerald.v1',
    semanticInputs: ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select', 'l', 'r'],
    controllerProfile: 'gba',
    gameProfile: 'pokemon_emerald',
    // The objective is a GOAL + how to operate the controller/harness —
    // never a game walkthrough. Game knowledge (what screens mean, which
    // button clears a menu, where to go) is the agent's job to DISCOVER
    // through play and RECORD in its persistent journal, so the same
    // agent generalizes to any GBA title and gets better across sessions.
    sampleObjective:
      "LONG-TERM GOAL: progress the game and WIN your first battle. " +
      "Controller: confirm = A (advance text / choose / interact); cancel = B " +
      "(back out); nav_up/nav_down/nav_left/nav_right = walk one step or move a " +
      "cursor; menu = START. " +
      "Discipline: READ THE SCREEN before every action and describe what you " +
      "actually see — never assume. Figure the game out by trying inputs and " +
      "watching what actually changes. If an input does nothing twice, try a " +
      "different one (menu/START often skips or confirms whole screens). " +
      "LEARN AND REMEMBER: keep your journal current every turn via " +
      "journal_update — status = where you are now; progress = what you have " +
      "achieved so far; objectives = your goal stack (long-term goal first, then " +
      "the sub-goal you are working on); notes = concrete facts you discovered " +
      "and want to reuse (what a screen means, which button worked, where a door " +
      "leads, what failed). Your journal survives across sessions: trust it over " +
      "guessing, and resume from it.",
  },
  // NDS: now wired through the universal control schema. Stylus is
  // still vision-only — the BridgedAdapter doesn't carry pointer events
  // yet (separate wire kind, separate adapter scope).
  nds: {
    probesName: null,
    logSchema: 'nds_vision.v1',
    semanticInputs: ['up', 'down', 'left', 'right', 'a', 'b', 'x', 'y', 'start', 'select', 'l', 'r'],
    controllerProfile: 'nds',
    gameProfile: 'generic_nds',
    sampleObjective: 'Walk to the next visible exit. Advance any textbox you see.',
  },
  // NES (fceumm). Two face buttons + d-pad + Start/Select. The agent
  // runs vision-only (no probe preset yet); every NES title gets a
  // sensible default mapping via the generic_nes game profile.
  nes: {
    probesName: null,
    logSchema: 'nes_vision.v1',
    semanticInputs: ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select'],
    controllerProfile: 'nes',
    gameProfile: 'generic_nes',
    sampleObjective: 'Walk right to find an enemy or obstacle. Press A to interact.',
  },
  // SNES (snes9x). Four-face diamond + shoulders + Start/Select.
  snes: {
    probesName: null,
    logSchema: 'snes_vision.v1',
    semanticInputs: ['up', 'down', 'left', 'right', 'a', 'b', 'x', 'y', 'start', 'select', 'l', 'r'],
    controllerProfile: 'snes',
    gameProfile: 'generic_snes',
    sampleObjective: 'Explore the current screen. Press A to advance any dialogue.',
  },
  // Sega Genesis / Mega Drive (genesis_plus_gx, 3-button). Three face
  // buttons (A/B/C) in a horizontal row + Start. 6-button pads are not
  // covered by this profile.
  genesis: {
    probesName: null,
    logSchema: 'genesis_vision.v1',
    semanticInputs: ['up', 'down', 'left', 'right', 'a', 'b', 'c', 'start'],
    controllerProfile: 'genesis',
    gameProfile: 'generic_genesis',
    sampleObjective: 'Move right until you encounter something. Press C to interact.',
  },
  // Sega Master System (genesis_plus_gx SMS mode). Two face buttons +
  // d-pad + Pause (on the console; libretro maps to Start slot).
  sms: {
    probesName: null,
    logSchema: 'sms_vision.v1',
    semanticInputs: ['up', 'down', 'left', 'right', '1', '2', 'pause'],
    controllerProfile: 'sms',
    gameProfile: 'generic_sms',
    sampleObjective: 'Move right and engage. Press 2 to attack, 1 to jump.',
  },
  // Sega Game Gear (genesis_plus_gx GG mode). Same buttons as SMS plus
  // a Start button on the handheld.
  gg: {
    probesName: null,
    logSchema: 'gg_vision.v1',
    semanticInputs: ['up', 'down', 'left', 'right', '1', '2', 'start'],
    controllerProfile: 'gg',
    gameProfile: 'generic_gg',
    sampleObjective: 'Move right and engage. Press 2 to attack, 1 to jump.',
  },
  // PC Engine / TurboGrafx-16 (mednafen_pce). Two face buttons (I, II)
  // + d-pad + Run/Select.
  pce: {
    probesName: null,
    logSchema: 'pce_vision.v1',
    semanticInputs: ['up', 'down', 'left', 'right', 'i', 'ii', 'run', 'select'],
    controllerProfile: 'pce',
    gameProfile: 'generic_pce',
    sampleObjective: 'Explore the current screen. Press I to perform the primary action.',
  },
  // Sony PlayStation 1 (pcsx_rearmed). Four face buttons + L1/R1 + L2/R2
  // + L3/R3 + d-pad + Start/Select. Analog sticks are NOT represented
  // (v1 BridgedAdapter has no analog wire channel).
  psx: {
    probesName: null,
    logSchema: 'psx_vision.v1',
    semanticInputs: [
      'up', 'down', 'left', 'right',
      'cross', 'circle', 'square', 'triangle',
      'l1', 'r1', 'l2', 'r2', 'l3', 'r3',
      'start', 'select',
    ],
    controllerProfile: 'psx',
    gameProfile: 'generic_psx',
    sampleObjective: 'Look around the current scene. Press X to advance dialogue.',
  },
};


// Per-title overrides on top of the per-system defaults. Title-id is a
// human-readable artifact id (typically the ROM filename or display
// name), so a substring/regex check is the pragmatic route — we don't
// have a separate metadata field like "internal_engine_id" yet.
//
// Each entry's ``overrides`` are merged on top of the matched system's
// defaults; the first matching pattern wins. Unmatched titles fall
// through to the per-system default (vision-only or generic profile).
//
// Add an entry here when you ship a new probe preset that targets a
// specific game family rather than the whole console.
const _TITLE_OVERRIDES = [
  {
    // Pokémon Gold / Silver / Crystal share an engine and one preset.
    pattern: /pok(?:[ée]|%C3%A9)?mon[\s_-]*(?:gold|silver|crystal)/i,
    overrides: {
      probesName: 'pokemon_gsc',
      logSchema: 'pokemon_gsc.v1',
      gameProfile: 'pokemon_gsc',
      sampleObjective: 'Walk south out of New Bark Town toward Cherrygrove. Advance any textbox you see.',
    },
  },
  {
    // Zelda: Link's Awakening (DX). Matches both the DX and original
    // titles; the preset targets DX (US v1.0) addresses but the input
    // map is unchanged between revisions.
    pattern: /zelda.*awakening|link.{0,3}(?:'s)?\s*awakening/i,
    overrides: {
      probesName: 'zelda_links_awakening_dx',
      logSchema: 'zelda_links_awakening_dx.v1',
      gameProfile: 'zelda_links_awakening_dx',
      sampleObjective: 'Reach the next dungeon entrance. Use the A-slot item to clear obstacles in your path.',
    },
  },
];


/**
 * Compose the effective defaults for a (system, titleId) pair.
 *
 * Per-title overrides win on a per-key basis; anything the override
 * doesn't set falls through to the system default. Returns ``null`` if
 * the system itself has no defaults registered (unsupported console).
 *
 * @param {string} system - libretro system id (e.g. "gb")
 * @param {string} titleId - artifact id used as override-pattern subject
 * @returns {object|null}
 */
function _resolveDefaults(system, titleId) {
  const base = _SYSTEM_DEFAULTS[system];
  if (!base) return null;
  const subject = String(titleId || '');
  for (const entry of _TITLE_OVERRIDES) {
    if (entry.pattern.test(subject)) {
      return { ...base, ...entry.overrides };
    }
  }
  return base;
}


/**
 * Build and return a panel for one emulator-stage mount.
 *
 * @param {object} opts
 *   @param {string}  opts.titleId         - title artifact id
 *   @param {string}  opts.system          - rom system id (e.g. 'gb')
 *   @param {EmulatorBridge} opts.bridge   - the live emulator bridge
 *   @param {boolean} [opts.startWithAgent=false]
 *       If true, the panel mounts expanded, defaults to co-pilot mode,
 *       and auto-starts the session as soon as the bridge fires
 *       'ready'. Used by the Launch-with-Partner button.
 *   @param {string|null} [opts.characterId=null]
 *       When set (and the session starts in co-pilot mode), the
 *       session POST carries ``character_id``; the server loads the
 *       persona + voice from ``ui_characters`` and threads them into
 *       the slow-path prompt and TTS calls.
 *   @param {function} [opts.fetchImpl=fetch]
 * @returns {{element: HTMLElement, destroy: function}}
 */
export function createAgentPanel({ titleId, system, bridge, startWithAgent, characterId, fetchImpl }) {
  const fetchFn = fetchImpl || fetch;
  const defaults = _resolveDefaults(system, titleId);
  const partnerLaunch = !!startWithAgent && !!defaults;

  // ── DOM scaffold ────────────────────────────────────────────────
  //
  // Collapsed by default: the panel is a thin sliver against the
  // right edge so the emulator keeps its full width. The user
  // clicks the chevron to expand into the 280px control column.
  //
  // Exception: partner launches mount expanded so the user can see
  // the agent's thinking from the first frame.
  const root = document.createElement('aside');
  root.className = 'agent-panel' + (partnerLaunch ? '' : ' agent-panel-collapsed');
  root.setAttribute('aria-label', 'Game agent');

  const header = document.createElement('header');
  header.className = 'agent-panel-header';

  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'agent-panel-toggle';
  toggle.setAttribute('aria-label', 'Toggle agent panel');
  toggle.innerHTML = '<span class="agent-panel-toggle-icon" aria-hidden="true">‹</span>';

  const dot = document.createElement('span');
  dot.className = 'agent-panel-status-dot';
  dot.setAttribute('data-status', 'off');
  dot.setAttribute('role', 'status');
  dot.setAttribute('aria-label', 'Agent off');

  const titleEl = document.createElement('span');
  titleEl.className = 'agent-panel-title';
  titleEl.textContent = 'Game Agent';

  // Inline status text — italic, right-aligned, hidden when the dot
  // alone is enough. Used for transient states like "thinking…" or
  // "reconnecting" where a one-word annotation makes the dot legible.
  const statusText = document.createElement('span');
  statusText.className = 'agent-panel-status-text';
  statusText.setAttribute('hidden', '');
  statusText.setAttribute('aria-live', 'polite');

  header.appendChild(toggle);
  header.appendChild(dot);
  header.appendChild(titleEl);
  header.appendChild(statusText);

  const body = document.createElement('div');
  body.className = 'agent-panel-body';

  root.appendChild(header);
  root.appendChild(body);

  // Reflect collapsed state on both the panel and (on phone) the play
  // area's data-panel-state, which drives the scrim visibility via CSS.
  function syncCollapseState() {
    const collapsed = root.classList.contains('agent-panel-collapsed');
    toggle.querySelector('.agent-panel-toggle-icon').textContent =
      collapsed ? '‹' : '›';
    const playArea = root.parentNode;
    if (playArea && playArea.classList.contains('emulator-stage-play-area')) {
      playArea.dataset.panelState = collapsed
        ? 'collapsed'
        : (window.innerWidth <= 720 ? 'expanded-mobile' : 'expanded');
    }
  }
  toggle.addEventListener('click', () => {
    root.classList.toggle('agent-panel-collapsed');
    syncCollapseState();
  });
  // Sync once after mount (the parent appends shortly after; defer one
  // frame so playArea.parentNode is set).
  requestAnimationFrame(syncCollapseState);
  // Keep state in sync when the viewport flips between desktop and
  // mobile breakpoints (rotation, devtools resize).
  window.addEventListener('resize', syncCollapseState, { passive: true });

  // Unsupported system path — render a quiet "coming soon" panel.
  if (!defaults) {
    const note = document.createElement('p');
    note.className = 'agent-panel-unsupported';
    note.textContent =
      'The agent is not yet wired for "' + (system || 'this system') +
      '". Probe presets ship per system; ' +
      'Pokémon GB/GBC is the first.';
    body.appendChild(note);
    return { element: root, destroy() {} };
  }

  // ── Controls (objective + mode + start/stop) ────────────────────
  const objWrap = document.createElement('label');
  objWrap.className = 'agent-panel-field';
  const objLabel = document.createElement('span');
  objLabel.textContent = 'Objective';
  const objInput = document.createElement('textarea');
  objInput.className = 'agent-panel-objective';
  objInput.rows = 3;
  objInput.placeholder = defaults.sampleObjective;
  objInput.value = '';
  objWrap.appendChild(objLabel);
  objWrap.appendChild(objInput);

  const modeWrap = document.createElement('div');
  modeWrap.className = 'agent-panel-modes';
  const modes = [
    { id: 'off',      label: 'Off',      desc: 'No agent.' },
    { id: 'watch',    label: 'Watch',    desc: 'Reason only.' },
    { id: 'co-pilot', label: 'Co-pilot', desc: 'User + agent both play.' },
  ];
  const modeRadios = {};
  for (const m of modes) {
    const w = document.createElement('label');
    w.className = 'agent-panel-mode';
    w.title = m.desc;
    const r = document.createElement('input');
    r.type = 'radio';
    r.name = 'agent-panel-mode-' + titleId;
    r.value = m.id;
    // Partner launches default to co-pilot so Start fires straight
    // into useful behaviour. Solo launches default to Off so the
    // user opts in deliberately.
    if (partnerLaunch ? m.id === 'co-pilot' : m.id === 'off') r.checked = true;
    modeRadios[m.id] = r;
    const t = document.createElement('span');
    t.textContent = m.label;
    w.appendChild(r);
    w.appendChild(t);
    modeWrap.appendChild(w);
  }

  const action = document.createElement('button');
  action.type = 'button';
  action.className = 'agent-panel-action';
  action.textContent = 'Start';

  // ── Live thinking section ──────────────────────────────────────
  const thinking = document.createElement('section');
  thinking.className = 'agent-panel-thinking';
  const thinkingTitle = document.createElement('h4');
  thinkingTitle.textContent = 'Current thought';
  const obsList = document.createElement('ul');
  obsList.className = 'agent-panel-observations';
  const obsEmpty = document.createElement('div');
  obsEmpty.className = 'agent-panel-observations-empty';
  const stateBox = document.createElement('div');
  stateBox.className = 'agent-panel-state is-empty';
  stateBox.textContent = 'No scratchpad yet — the agent will write what it\'s tracking here.';
  thinking.appendChild(thinkingTitle);
  thinking.appendChild(obsList);
  thinking.appendChild(obsEmpty);
  thinking.appendChild(stateBox);

  const actionStrip = document.createElement('section');
  actionStrip.className = 'agent-panel-recent';
  const actionTitle = document.createElement('h4');
  actionTitle.textContent = 'Recent inputs';
  const actionRow = document.createElement('div');
  actionRow.className = 'agent-panel-action-row';
  actionStrip.appendChild(actionTitle);
  actionStrip.appendChild(actionRow);

  // Speech bubble for companion-mode utterances. Anchored at the top
  // of the panel (above the controls), hidden by default; fades in
  // when the bridge starts an audio frame and out when audio ends.
  const speech = document.createElement('div');
  speech.className = 'agent-panel-speech';
  speech.setAttribute('aria-live', 'polite');
  speech.setAttribute('hidden', '');

  body.appendChild(speech);
  body.appendChild(objWrap);
  body.appendChild(modeWrap);
  body.appendChild(action);
  body.appendChild(thinking);
  body.appendChild(actionStrip);

  // ── State ──────────────────────────────────────────────────────
  let session = null;   // {session_id, bridge_ws_url, sse, mode}
  let starting = false;

  // Status label map. The dot does the at-a-glance colour; the inline
  // text adds a one-word annotation for transient states so a stalled
  // session is legible without staring at a yellow dot trying to guess
  // what it means.
  const STATUS_LABELS = {
    off:           { aria: 'Agent off',             inline: ''               },
    starting:      { aria: 'Agent starting',        inline: 'starting…'      },
    watch:         { aria: 'Agent watching',        inline: ''               },
    'co-pilot':    { aria: 'Agent co-piloting',     inline: ''               },
    thinking:      { aria: 'Agent thinking',        inline: 'thinking…'      },
    reconnecting:  { aria: 'Agent reconnecting',    inline: 'reconnecting…'  },
    error:         { aria: 'Agent error',           inline: 'error'          },
  };

  function setStatus(status) {
    dot.setAttribute('data-status', status);
    const meta = STATUS_LABELS[status] || STATUS_LABELS.off;
    dot.setAttribute('aria-label', meta.aria);
    if (meta.inline) {
      statusText.textContent = meta.inline;
      statusText.removeAttribute('hidden');
    } else {
      statusText.textContent = '';
      statusText.setAttribute('hidden', '');
    }
  }

  // Active session status (watch / co-pilot) is the steady-state we
  // bounce back to after a transient "thinking…" indicator. Tracked
  // separately so handleLogEntry can flip thinking on/off without
  // forgetting the underlying mode.
  let steadyStatus = 'off';
  function setSteadyStatus(status) {
    steadyStatus = status;
    setStatus(status);
  }

  function readMode() {
    for (const id of Object.keys(modeRadios)) {
      if (modeRadios[id].checked) return id;
    }
    return 'off';
  }

  function applyMode() {
    const mode = readMode();
    if (mode === 'off') {
      stop();
      return;
    }
    // If a session exists, just flip its dispatch behavior.
    if (session !== null) {
      if (typeof bridge.setAgentMode === 'function') {
        bridge.setAgentMode(mode === 'co-pilot' ? 'co-pilot' : 'watch');
      }
      setStatus(mode);
      return;
    }
    // Otherwise the user must press Start to actually open a session.
    // We keep the radio choice; Start uses it.
  }

  for (const r of Object.values(modeRadios)) {
    r.addEventListener('change', applyMode);
  }

  action.addEventListener('click', () => {
    if (session !== null) {
      stop();
    } else {
      start();
    }
  });

  // Surface agent actions live as they're dispatched. EmulatorBridge.on
  // returns its own unsubscribe; capture it so destroy() can run a
  // clean teardown — there is no bridge.off() to call.
  const onAgentAction = (payload) => {
    addActionChip(payload.semantic, payload.dispatched);
  };
  const unsubAction = bridge.on?.('agent-action', onAgentAction) || (() => {});

  // Companion speech — show a bubble while audio is playing, clear it
  // when the audio ends. Auto-expand the panel if collapsed so the
  // user sees the bubble.
  const onSpeakStart = ({ text }) => {
    if (!text) return;
    speech.textContent = text;
    speech.removeAttribute('hidden');
    if (root.classList.contains('agent-panel-collapsed')) {
      root.classList.remove('agent-panel-collapsed');
      const icon = toggle.querySelector('.agent-panel-toggle-icon');
      if (icon) icon.textContent = '›';
    }
  };
  const onSpeakEnd = () => {
    // Linger briefly so the user can finish reading after audio
    // ends, then fade away.
    speech.classList.add('agent-panel-speech-fading');
    setTimeout(() => {
      speech.classList.remove('agent-panel-speech-fading');
      speech.setAttribute('hidden', '');
      speech.textContent = '';
    }, 1200);
  };
  const unsubSpeakStart = bridge.on?.('agent-speak-start', onSpeakStart) || (() => {});
  const unsubSpeakEnd   = bridge.on?.('agent-speak-end',   onSpeakEnd)   || (() => {});

  // Partner launch: auto-start as soon as the emulator iframe is
  // ready (no point starting before the libretro core is alive).
  // Done as a one-shot so a teardown-and-remount doesn't spawn
  // duplicate sessions.
  let unsubReady = () => {};
  if (partnerLaunch) {
    const onBridgeReady = () => {
      unsubReady();
      unsubReady = () => {};
      start().catch((err) => {
        console.error('[agent-panel] partner auto-start failed', err);
      });
    };
    unsubReady = bridge.on?.('ready', onBridgeReady) || (() => {});
  }

  // ── Lifecycle ──────────────────────────────────────────────────

  async function start() {
    if (starting || session !== null) return;
    const mode = readMode();
    if (mode === 'off') {
      showToast('Pick Watch or Co-pilot first', 'info');
      return;
    }
    starting = true;
    action.disabled = true;
    action.textContent = 'Starting…';
    setStatus('starting');
    try {
      const objective = (objInput.value.trim() || defaults.sampleObjective);
      const isCompanion = mode === 'co-pilot';
      const resp = await fetchFn('/api/game-agent/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          surface: 'emulatorjs',
          objective,
          semantic_inputs: defaults.semanticInputs,
          log_schema: defaults.logSchema,
          // Companion mode is on by default in co-pilot mode. The
          // panel unlocks the agent's ``say`` field; the bridge
          // routes audio frames; the speech bubble shows the text.
          companion: isCompanion,
          // character_id is only meaningful when companion is true;
          // server ignores it otherwise. ``null`` = anonymous helper.
          character_id: isCompanion ? (characterId || null) : null,
          // Library artifact id of the game being played. Used by the
          // server-side CompanionJournal to key persistent memory by
          // (user_id, title_id) so the agent remembers state across
          // session restarts. Falsy = no cross-session memory for this
          // session.
          title_id: titleId || null,
          // Universal control schema (Phase G). When both ids are
          // present the server composes a ComposedProfile, INPUT_HINTS
          // get rendered in the slow-path prompt, and the WS payload
          // carries wire_code so the iframe doesn't re-resolve. Either
          // null = legacy semantic-name path; the server demands both
          // or neither, and rejects one-alone as a 400.
          controller_profile: defaults.controllerProfile || null,
          game_profile: defaults.gameProfile || null,
        }),
      });
      if (!resp.ok) {
        const body = await resp.text();
        throw new Error('session start failed: ' + resp.status + ' ' + body);
      }
      const body = await resp.json();
      const sessionId = body.session_id;
      const rawBridgeUrl = body.bridge_ws_url;
      if (!rawBridgeUrl) {
        throw new Error('server did not return bridge_ws_url');
      }
      // Auth middleware requires `?ticket=` on every WS handshake;
      // browser WS API can't send cookies cross-protocol, so a
      // short-lived ticket minted off the user's session is the
      // approved escape hatch (same pattern voice.js uses).
      const ticket = await getWsTicket();
      const sep = rawBridgeUrl.includes('?') ? '&' : '?';
      const bridgeUrl = `${rawBridgeUrl}${sep}ticket=${encodeURIComponent(ticket)}`;
      await bridge.startGameAgent({
        sessionId,
        bridgeUrl,
        probesName: defaults.probesName,
        tickHz: 4,
        // 3 Hz capture ("more live-like"): the ring buffer gains real
        // temporal resolution so action-effect clips (frames straddling
        // a button press) and the scene narrator see motion, not 1 s
        // strobes. Bandwidth ~45 KB/s of base64 PNG — still trivial.
        frameHz: 3,
        mode: mode === 'co-pilot' ? 'co-pilot' : 'watch',
      });
      session = {
        sessionId,
        bridgeUrl,
        mode,
        sse: openLogStream(sessionId),
      };
      setSteadyStatus(mode);
      action.disabled = false;
      action.textContent = 'Stop';
      action.setAttribute('data-state', 'running');
    } catch (err) {
      console.error('[agent-panel] start failed', err);
      showToast('Agent start failed: ' + err.message, 'error');
      setStatus('error');
      action.disabled = false;
      action.textContent = 'Start';
      action.removeAttribute('data-state');
    } finally {
      starting = false;
    }
  }

  function stop() {
    if (session === null) {
      // Also reset radio to Off so the UI doesn't lie.
      modeRadios.off.checked = true;
      setSteadyStatus('off');
      action.textContent = 'Start';
      action.removeAttribute('data-state');
      return;
    }
    try { session.sse?.close(); } catch (_e) { /* ignore */ }
    try { bridge.stopGameAgent(); } catch (_e) { /* ignore */ }
    session = null;
    modeRadios.off.checked = true;
    setSteadyStatus('off');
    action.textContent = 'Start';
    action.removeAttribute('data-state');
  }

  function openLogStream(sessionId) {
    const url = '/api/game-agent/sessions/' + encodeURIComponent(sessionId) + '/log';
    const sse = new EventSource(url, { withCredentials: true });
    sse.onmessage = (ev) => {
      let entry;
      try { entry = JSON.parse(ev.data); } catch (_e) { return; }
      handleLogEntry(entry);
    };
    sse.onerror = () => {
      // SSE auto-reconnects; surfacing the dot keeps the UI honest.
      setStatus('reconnecting');
    };
    return sse;
  }

  function handleLogEntry(entry) {
    if (!entry || typeof entry.kind !== 'string') return;
    if (entry.kind === 'plan') {
      const p = entry.payload || {};
      renderObservations(Array.isArray(p.observations) ? p.observations : []);
      const stateText = (p.state_update || '').trim();
      if (stateText) {
        stateBox.classList.remove('is-empty');
        stateBox.textContent = stateText;
        // Brief flash so the user catches the update without the
        // scratchpad rewriting silently. The CSS animation auto-decays
        // after ~1.4s; we just toggle the class and let the browser
        // restart the keyframe by removing+re-adding it.
        stateBox.classList.remove('is-fresh');
        // Force reflow so the keyframe re-fires on repeated updates.
        void stateBox.offsetWidth;
        stateBox.classList.add('is-fresh');
      } else {
        stateBox.classList.add('is-empty');
        stateBox.textContent = 'No scratchpad yet — the agent will write what it\'s tracking here.';
      }
      // After we've processed a plan we're back in the steady state
      // (watch / co-pilot). Refresh so any transient "reconnecting…"
      // text clears.
      if (steadyStatus !== 'off' && steadyStatus !== 'error') {
        setStatus(steadyStatus);
      }
    } else if (entry.kind === 'session_end') {
      // Server ended the session — surface and reset.
      stop();
    }
  }

  // Diff-based render: keep existing <li> nodes when the text is
  // unchanged so the browser doesn't repaint the whole list on every
  // plan tick, and mark the newly-added items with .is-fresh so the
  // CSS highlights them for ~2 s. Capped at 5 items for sidebar fit.
  function renderObservations(items) {
    const capped = (items || []).slice(0, 5).map(String);
    const existing = Array.from(obsList.children);
    const existingText = existing.map(el => el.textContent);

    // Identify the longest common prefix so we don't churn unchanged
    // items at the top of the list.
    let prefix = 0;
    while (prefix < capped.length
           && prefix < existingText.length
           && capped[prefix] === existingText[prefix]) {
      prefix += 1;
    }

    // Drop stale tail items.
    while (existing.length > prefix) {
      const old = existing.pop();
      obsList.removeChild(old);
    }

    // Append (and mark-as-fresh) anything new past the shared prefix.
    for (let i = prefix; i < capped.length; i++) {
      const li = document.createElement('li');
      li.textContent = capped[i];
      li.classList.add('is-fresh');
      obsList.appendChild(li);
    }

    // Toggle the empty-state placeholder visibility (CSS handles its
    // styling; we just show/hide the helper element).
    obsEmpty.style.display = capped.length === 0 ? '' : 'none';
  }

  function addActionChip(semantic, dispatched) {
    const chip = document.createElement('span');
    chip.className = 'agent-panel-action-chip';
    if (!dispatched) chip.classList.add('watching');
    chip.textContent = semantic;
    actionRow.appendChild(chip);
    // Smooth overflow: fade leading chips out, then remove them after
    // the transition completes. Without this they pop out of existence
    // and the strip jitters.
    while (actionRow.children.length > 12) {
      const oldest = actionRow.firstChild;
      oldest.classList.add('is-leaving');
      const node = oldest;
      setTimeout(() => {
        if (node.parentNode === actionRow) actionRow.removeChild(node);
      }, 220);
      // Break the visible cap immediately for layout so the next
      // appendChild doesn't force the row to wrap mid-transition.
      if (actionRow.children.length <= 13) break;
    }
  }

  function destroy() {
    unsubAction();
    unsubReady();
    unsubSpeakStart();
    unsubSpeakEnd();
    stop();
    if (root.parentNode) root.parentNode.removeChild(root);
  }

  return { element: root, destroy };
}
