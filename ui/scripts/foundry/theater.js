// ui/scripts/foundry/theater.js
//
// The Game Foundry — a LIVE theater, not a form. You describe a game, then
// watch the agent work: code streaming out, the Blender render resolving, the
// game booting, and the agent playing its own creation while its decisions
// scroll past. The magic is watching it think and improve in real time.
//
// Data: POST /api/foundry/runs starts a run; GET /api/foundry/runs/{id}/events
// is an SSE feed of the loop's stage events (events.py). The left pane shows
// the Blender render + a live preview of the game + the pass timeline with
// climbing scores; the right pane is the agent's decision stream.

import { escapeHtml, showToast } from '../app.js';

let _overlay = null;
let _es = null;           // EventSource for the active run
let _state = null;

// ── Public entry ──────────────────────────────────────────────────────

export function openFoundryTheater() {
  if (_overlay) { _overlay.classList.remove('hidden'); return; }
  _overlay = document.createElement('div');
  _overlay.className = 'foundry-theater';
  _overlay.innerHTML = _setupMarkup();
  document.body.appendChild(_overlay);
  _wireSetup();
  _loadPickers();
}

function _close() {
  if (_es) { try { _es.close(); } catch (_) {} _es = null; }
  if (_overlay) { _overlay.remove(); _overlay = null; }
  _state = null;
}

// ── Setup view ────────────────────────────────────────────────────────

function _setupMarkup() {
  return `
    <div class="ft-scrim"></div>
    <div class="ft-shell ft-setup">
      <header class="ft-head">
        <span class="ft-glyph">&#9874;</span>
        <h2>The Game Foundry</h2>
        <button class="ft-close" title="Close">&times;</button>
      </header>
      <div class="ft-setup-body">
        <p class="ft-lede">Describe a game. Watch it get built, played, and improved &mdash; live.</p>
        <label class="ft-field">
          <span>Concept</span>
          <textarea id="ft-concept" rows="2" placeholder="a fast coin-dash platformer with a double jump"></textarea>
        </label>
        <div class="ft-row">
          <label class="ft-field ft-grow">
            <span>Title</span>
            <input id="ft-title" type="text" placeholder="Coin Dash">
          </label>
          <label class="ft-field ft-grow">
            <span>Objective (what "winning" is)</span>
            <input id="ft-objective" type="text" placeholder="collect all 10 coins">
          </label>
        </div>
        <div class="ft-row">
          <div class="ft-field">
            <span>Style</span>
            <div class="ft-seg" id="ft-dimension">
              <button data-dim="2d" class="active">2D</button>
              <button data-dim="3d">3D &middot; Blender</button>
            </div>
          </div>
          <label class="ft-field ft-grow">
            <span>Workspace</span>
            <select id="ft-workspace"><option value="">Loading&hellip;</option></select>
          </label>
          <label class="ft-field ft-grow">
            <span>Model</span>
            <select id="ft-model"><option value="">Loading&hellip;</option></select>
          </label>
        </div>
        <div class="ft-row">
          <label class="ft-field">
            <span>Passes <b id="ft-passes-val">2</b></span>
            <input id="ft-passes" type="range" min="1" max="5" value="2">
          </label>
          <label class="ft-field">
            <span>Playtest seconds <b id="ft-secs-val">90</b></span>
            <input id="ft-secs" type="range" min="30" max="240" step="15" value="90">
          </label>
        </div>
        <p class="ft-hint" id="ft-3d-hint" hidden>3D needs a <b>creative</b> workspace (Blender). Pick one above, or create one in the Coder surface.</p>
      </div>
      <footer class="ft-foot">
        <button class="ft-forge" id="ft-forge">Forge &#9654;</button>
      </footer>
    </div>`;
}

function _wireSetup() {
  const q = (id) => _overlay.querySelector(id);
  q('.ft-close').onclick = _close;
  q('.ft-scrim').onclick = _close;
  q('#ft-passes').oninput = (e) => { q('#ft-passes-val').textContent = e.target.value; };
  q('#ft-secs').oninput = (e) => { q('#ft-secs-val').textContent = e.target.value; };
  _overlay.querySelectorAll('#ft-dimension button').forEach((b) => {
    b.onclick = () => {
      _overlay.querySelectorAll('#ft-dimension button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      q('#ft-3d-hint').hidden = b.dataset.dim !== '3d';
    };
  });
  q('#ft-forge').onclick = _startRun;
}

async function _loadPickers() {
  // Workspaces — label the profile so a creative one is easy to spot.
  try {
    const r = await fetch('/api/coder/workspaces');
    const body = await r.json().catch(() => ({}));
    const ws = (body.workspaces || []);
    const sel = _overlay?.querySelector('#ft-workspace');
    if (sel) {
      sel.innerHTML = ws.length
        ? ws.map((w) => {
            const prof = w.profile || w.tooling_profile || '';
            const tag = prof ? ` (${escapeHtml(prof)})` : '';
            return `<option value="${escapeHtml(w.id)}">${escapeHtml(w.name || w.id)}${tag}</option>`;
          }).join('')
        : '<option value="">No workspaces — create one in Coder</option>';
    }
  } catch (_) { /* leave placeholder */ }

  // Models — the user picks; never auto-selected.
  try {
    const r = await fetch('/v1/models');
    const body = await r.json().catch(() => ({}));
    const models = (body.data || []).map((m) => m.id).filter(Boolean);
    const sel = _overlay?.querySelector('#ft-model');
    if (sel) {
      sel.innerHTML = ['<option value="">Choose a model&hellip;</option>']
        .concat(models.map((id) => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`))
        .join('');
    }
  } catch (_) { /* leave placeholder */ }
}

async function _startRun() {
  const q = (id) => _overlay.querySelector(id);
  const concept = q('#ft-concept').value.trim();
  const title = q('#ft-title').value.trim() || 'Untitled Game';
  const objective = q('#ft-objective').value.trim();
  const dimension = _overlay.querySelector('#ft-dimension .active')?.dataset.dim || '2d';
  const workspace_id = q('#ft-workspace').value;
  const model = q('#ft-model').value;
  const passes = parseInt(q('#ft-passes').value, 10);
  const play_seconds = parseInt(q('#ft-secs').value, 10);

  if (!concept) { showToast('Describe the game first', 'warn'); return; }
  if (!objective) { showToast('Set an objective — the agent needs a goal to play toward', 'warn'); return; }
  if (!workspace_id) { showToast('Pick a workspace', 'warn'); return; }
  if (!model) { showToast('Pick a model', 'warn'); return; }

  q('#ft-forge').disabled = true;
  try {
    const r = await fetch('/api/foundry/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id, model, title, concept, objective,
                             dimension, passes, play_seconds }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok || !body.run_id) {
      showToast(body.error || 'Could not start the foundry', 'error');
      q('#ft-forge').disabled = false;
      return;
    }
    _enterLiveView({ run_id: body.run_id, title, passes });
  } catch (err) {
    showToast('Foundry request failed', 'error');
    q('#ft-forge').disabled = false;
  }
}

// ── Live theater view ─────────────────────────────────────────────────

function _enterLiveView({ run_id, title, passes }) {
  _state = { run_id, title, passes, current: 0, scores: {} };
  const shell = _overlay.querySelector('.ft-shell');
  shell.classList.remove('ft-setup');
  shell.classList.add('ft-live');
  shell.innerHTML = `
    <header class="ft-head">
      <span class="ft-glyph">&#9874;</span>
      <h2>${escapeHtml(title)}</h2>
      <span class="ft-status" id="ft-status">starting&hellip;</span>
      <button class="ft-close" title="Close">&times;</button>
    </header>
    <div class="ft-stage">
      <section class="ft-preview">
        <div class="ft-render" id="ft-render"><span class="ft-render-ph">render appears here</span></div>
        <div class="ft-game" id="ft-game"><span class="ft-game-ph">the game plays here once it boots</span></div>
        <div class="ft-timeline" id="ft-timeline"></div>
      </section>
      <aside class="ft-stream">
        <div class="ft-stream-title">What the agent is doing</div>
        <ol class="ft-log" id="ft-log"></ol>
      </aside>
    </div>`;
  shell.querySelector('.ft-close').onclick = _close;
  _renderTimeline();
  _subscribe(run_id);
}

function _subscribe(run_id) {
  _es = new EventSource(`/api/foundry/runs/${encodeURIComponent(run_id)}/events`);
  _es.onmessage = (e) => {
    let ev; try { ev = JSON.parse(e.data); } catch (_) { return; }
    _handleEvent(ev);
  };
  _es.onerror = () => { /* SSE auto-reconnects; backlog replays on reconnect */ };
}

function _handleEvent(ev) {
  const status = _overlay?.querySelector('#ft-status');
  switch (ev.type) {
    case 'run_start':
      _state.passes = ev.passes || _state.passes;
      _renderTimeline();
      if (status) status.textContent = 'building';
      break;
    case 'pass_start':
      _state.current = ev.index;
      _renderTimeline();
      _log(`— Pass ${ev.index} —`, 'ft-sep');
      break;
    case 'asset_building':
      _log('sculpting the 3D asset in Blender…', 'ft-act');
      break;
    case 'asset_render': {
      const el = _overlay?.querySelector('#ft-render');
      if (el && ev.image) el.innerHTML = `<img src="${ev.image}" alt="render">`;
      _log('Blender render ready', 'ft-ok');
      break;
    }
    case 'generating':
      _log('writing the game code…', 'ft-act');
      if (status) status.textContent = 'generating';
      break;
    case 'generated':
      _log(`generated ${ev.file_count || 0} files`
        + (ev.violations?.length ? ` — ${ev.violations.length} contract issue(s)` : ''),
        ev.violations?.length ? 'ft-warn' : 'ft-ok');
      break;
    case 'play_start':
      _log('booting the game and starting to play…', 'ft-act');
      if (status) status.textContent = 'playtesting';
      break;
    case 'play_session':
      _embedGame(ev.play_url);
      break;
    case 'observation':
      _log(ev.text, 'ft-obs');
      break;
    case 'pass_scored': {
      _state.scores[ev.index ?? _state.current] = ev;
      _renderTimeline();
      if (ev.score == null) {
        _log('build could not be played — feeding fixes back', 'ft-warn');
      } else {
        _log(`score ${Number(ev.score).toFixed(2)}`
          + (ev.defects?.length ? ` · ${ev.defects.length} issue(s) found` : ' · clean'),
          'ft-score');
        (ev.defects || []).forEach((d) => _log(`⚠ ${d.kind}: ${d.detail}`, 'ft-defect'));
      }
      break;
    }
    case 'regenerating':
      _log('↻ regenerating from the playtest feedback…', 'ft-act');
      break;
    case 'done':
      if (status) status.textContent = ev.improved ? 'improved ✓' : 'done';
      _log(ev.improved ? 'The game got measurably better across passes ✓'
                       : 'Run complete.', ev.improved ? 'ft-ok' : 'ft-sep');
      if (_es) { _es.close(); _es = null; }
      break;
    case 'error':
      if (status) status.textContent = 'error';
      _log(`error: ${ev.message}`, 'ft-warn');
      break;
    default:
      break;
  }
}

function _embedGame(playUrl) {
  const el = _overlay?.querySelector('#ft-game');
  if (!el || !playUrl) return;
  // view=1 renders the game WITHOUT a second controlling shim.
  const sep = playUrl.includes('?') ? '&' : '?';
  el.innerHTML = `<iframe src="${escapeHtml(playUrl + sep + 'view=1')}" `
    + `sandbox="allow-scripts allow-same-origin" title="live game"></iframe>`;
}

function _renderTimeline() {
  const el = _overlay?.querySelector('#ft-timeline');
  if (!el || !_state) return;
  const dots = [];
  for (let i = 1; i <= _state.passes; i++) {
    const sc = _state.scores[i];
    let cls = 'ft-dot';
    let label = `${i}`;
    if (sc) {
      cls += sc.score == null ? ' ft-dot-fail' : ' ft-dot-done';
      if (sc.score != null) label = `${i} · ${Number(sc.score).toFixed(1)}`;
    } else if (i === _state.current) {
      cls += ' ft-dot-active';
    }
    dots.push(`<span class="${cls}">${escapeHtml(label)}</span>`);
  }
  el.innerHTML = `<span class="ft-tl-label">Passes</span>${dots.join('<span class="ft-tl-arrow">→</span>')}`;
}

function _log(text, cls) {
  const el = _overlay?.querySelector('#ft-log');
  if (!el || !text) return;
  const li = document.createElement('li');
  li.className = cls || '';
  li.textContent = text;
  el.appendChild(li);
  el.scrollTop = el.scrollHeight;
}
