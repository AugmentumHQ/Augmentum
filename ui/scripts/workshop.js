/*
 * Workshop — where Augmentum improves itself.
 *
 * The human end of the self-improvement system, built as a first-class Augmentum
 * "space" (the chrome, motion, and editorial voice of Library/Media): a lane-rail
 * surface that sits below the global header, never covering the app's nav.
 *
 * Lanes:
 *   Overview — the system at a glance: health, posture, the three engines, stats
 *   Lineage  — the never-pruned archive of every attempt + the human verdict
 *   Adapt    — reshape a surface live (config/Adaptation runs now)
 *   Debt     — the debt-paydown plan (auto-lane vs. propose-to-you)
 *   Evolve   — eval-driven evolution of the prompts that shape behavior
 *
 * APIs: GET /api/selfedit/{health,attempts}, POST /api/selfedit/{propose,reshape},
 * POST /api/selfedit/attempts/{id}/verdict, GET/PUT /api/config/tools.
 * Vanilla DOM; all server text through esc() (template-literal safe).
 */

const API = '/api/selfedit';

let _overlay = null;
let _opened = false;
let _lane = 'overview';
let _attempts = [];
let _selectedId = '';
let _settings = { enabled: false, autonomy: 'propose', editModel: '', frontierModel: '' };

function esc(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/`/g, '&#96;').replace(/\$\{/g, '&#36;{');
}

async function _json(url, opts) {
  try {
    const res = await fetch(url, opts);
    let body = {};
    try { body = await res.json(); } catch { /* empty */ }
    return { ok: res.ok, status: res.status, body };
  } catch (e) {
    return { ok: false, status: 0, body: { error: String(e) } };
  }
}

function _timeAgo(iso) {
  if (!iso) return '';
  const t = Date.parse(iso.includes('Z') || iso.includes('+') ? iso : iso + 'Z');
  if (Number.isNaN(t)) return '';
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

const STATUS_META = {
  proposed: { label: 'Proposed', cls: 'ws-st-proposed' },
  editing: { label: 'Editing', cls: 'ws-st-editing' },
  gated: { label: 'Awaiting you', cls: 'ws-st-gated' },
  promoted: { label: 'Promoted', cls: 'ws-st-promoted' },
  rejected: { label: 'Rejected', cls: 'ws-st-rejected' },
  rolled_back: { label: 'Rolled back', cls: 'ws-st-reverted' },
  failed: { label: 'Failed', cls: 'ws-st-failed' },
};

const TIER_NOTE = {
  verified: 'A mechanical oracle confirmed the change did what was asked — safe to auto-promote.',
  human_confirmed: 'You kept this.',
  probable: 'A judgment model thinks it is right — worth a glance.',
  human_required: 'No regression, but only you can say it is right.',
  failed: 'A required check failed or it regressed the app.',
};

const SURFACE_ICON = {
  config: '◧', frontend: '◑', backend: '⬡', migration: '⛁', mixed: '◐', '': '·',
};

const LANES = [
  { id: 'overview', label: 'Overview', glyph: '◉' },
  { id: 'lineage', label: 'Lineage', glyph: '❡' },
  { id: 'adapt', label: 'Adapt', glyph: '◧' },
  { id: 'debt', label: 'Debt', glyph: '⚖' },
  { id: 'coverage', label: 'Coverage', glyph: '◬' },
  { id: 'apply', label: 'Go live', glyph: '⤴' },
  { id: 'evolve', label: 'Evolve', glyph: '✶' },
  { id: 'learned', label: 'Learned', glyph: '◈' },
];

// ── build (once) ─────────────────────────────────────────────────────────

function _build() {
  if (_overlay) return;
  _overlay = document.createElement('div');
  _overlay.id = 'workshop-overlay';
  _overlay.className = 'ws-overlay hidden';
  _overlay.setAttribute('role', 'dialog');
  _overlay.setAttribute('aria-modal', 'true');
  _overlay.setAttribute('aria-label', 'Workshop');
  _overlay.setAttribute('tabindex', '-1');

  _overlay.innerHTML = `
    <header class="ws-header">
      <div class="ws-titlewrap">
        <h2 class="ws-title">Workshop</h2>
        <span class="ws-tagline">where Augmentum improves itself</span>
      </div>
      <div class="ws-header-actions">
        <span class="ws-health" id="ws-health" title="Application health signal">
          <span class="ws-health-dot"></span><span class="ws-health-label">health</span>
        </span>
        <label class="ws-toggle" title="Master switch — when off, nothing self-edits. Experimental (early access): this commits to Augmentum's own git repo, and its checks prove a change didn't break the build rather than that it's correct. Keep autonomy on Propose unless you're supervising. Turning this off also hides the Workshop from the sidebar.">
          <input type="checkbox" id="ws-enabled"><span class="ws-toggle-track"></span>
          <span class="ws-toggle-label">Self-edit</span>
        </label>
        <label class="ws-autonomy ws-model-pick" title="Rung 1 of the escalation ladder — the model that does the editing. Empty = the small utility role does the groundwork.">
          <span class="ws-toggle-label">Edits with</span>
          <select id="ws-edit-model" aria-label="Edit model"></select>
        </label>
        <label class="ws-autonomy ws-model-pick" title="The top rung — climbed only on a hard target when a run has “Allow frontier” ticked. Empty = the ladder stops at your primary model.">
          <span class="ws-toggle-label">Frontier</span>
          <select id="ws-frontier-model" aria-label="Frontier model"></select>
        </label>
        <label class="ws-autonomy" title="How verified changes are handled">
          <select id="ws-autonomy" aria-label="Autonomy">
            <option value="propose">Propose · you decide</option>
            <option value="auto_verified">Auto-promote verified</option>
          </select>
        </label>
        <button class="ws-close" id="ws-close-btn" aria-label="Close Workshop">Close</button>
      </div>
    </header>
    <div class="ws-body">
      <nav class="ws-rail" id="ws-rail" aria-label="Workshop lanes">
        ${LANES.map((l) => `
          <button class="ws-rail-item" data-lane="${l.id}">
            <span class="ws-rail-glyph" aria-hidden="true">${l.glyph}</span>
            <span class="ws-rail-label">${l.label}</span>
          </button>`).join('')}
        <div class="ws-rail-foot" id="ws-rail-foot"></div>
      </nav>
      <main class="ws-content" id="ws-content"></main>
    </div>
  `;
  document.body.appendChild(_overlay);

  _overlay.querySelector('#ws-close-btn').addEventListener('click', closeWorkshop);
  _overlay.querySelector('#ws-enabled').addEventListener('change', _onToggleEnabled);
  _overlay.querySelector('#ws-autonomy').addEventListener('change', _onChangeAutonomy);
  _overlay.querySelector('#ws-edit-model').addEventListener('change',
    (ev) => _onChangeModel('selfedit_edit_model', ev));
  _overlay.querySelector('#ws-frontier-model').addEventListener('change',
    (ev) => _onChangeModel('selfedit_frontier_model', ev));
  _overlay.querySelectorAll('.ws-rail-item').forEach((b) =>
    b.addEventListener('click', () => _setLane(b.dataset.lane)));
  _overlay.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') closeWorkshop(); });
}

// ── settings + health (header) ───────────────────────────────────────────

async function _loadSettings() {
  const { ok, body } = await _json('/api/config/tools');
  if (!ok || !body) return;
  _settings.enabled = body.selfedit_enabled === true || body.selfedit_enabled === 1 || body.selfedit_enabled === '1';
  _settings.autonomy = body.selfedit_autonomy_level || 'propose';
  _settings.editModel = body.selfedit_edit_model || '';
  _settings.frontierModel = body.selfedit_frontier_model || '';
  const cb = _overlay.querySelector('#ws-enabled');
  const sel = _overlay.querySelector('#ws-autonomy');
  if (cb) cb.checked = _settings.enabled;
  if (sel) sel.value = _settings.autonomy;
  _renderRailFoot();
  _modelCatalog = null;  // refresh the catalog each open — models come and go
  _populateModelPickers();
}

// ── ladder model pickers (header) ────────────────────────────────────────
// The user chooses the ladder's models — never auto-picked. Empty is a real,
// labeled state ("utility role" / "no frontier rung"), not a silent default.

let _modelCatalog = null;  // model ids from /v1/models, fetched once per open

async function _populateModelPickers() {
  const editSel = _overlay.querySelector('#ws-edit-model');
  const frontierSel = _overlay.querySelector('#ws-frontier-model');
  if (!editSel || !frontierSel) return;
  if (!_modelCatalog) {
    const { ok, body } = await _json('/v1/models');
    _modelCatalog = (ok && body && Array.isArray(body.data))
      ? body.data.map((m) => m && m.id).filter(Boolean) : [];
  }
  const fill = (sel, current, emptyLabel) => {
    const ids = [..._modelCatalog];
    // A saved model missing from the live catalog (backend offline) stays
    // visible as the selection — never silently re-read as the default.
    if (current && !ids.includes(current)) ids.unshift(current);
    sel.innerHTML = `<option value="">${esc(emptyLabel)}</option>`
      + ids.map((id) => `<option value="${esc(id)}">${esc(id)}</option>`).join('');
    sel.value = current || '';
  };
  fill(editSel, _settings.editModel, 'utility role (default)');
  fill(frontierSel, _settings.frontierModel, 'no frontier rung');
}

async function _onChangeModel(key, ev) {
  const value = ev.target.value;
  const isEdit = key === 'selfedit_edit_model';
  if (isEdit) _settings.editModel = value; else _settings.frontierModel = value;
  const saved = await _saveSetting(key, value);
  if (!saved) return;
  _toast(isEdit
    ? (value ? `Edits will start on ${value} — applies to the next run.`
             : 'Edits will start on the utility role — applies to the next run.')
    : (value ? `Frontier rung: ${value} — climbed only when a run allows frontier.`
             : 'Frontier rung removed — the ladder stops at your primary model.'));
}

async function _saveSetting(key, value) {
  const { ok, body } = await _json('/api/config/tools', {
    method: 'PUT', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ [key]: value }),
  });
  if (!ok) _toast(body.error || 'Could not save (admin only).', true);
  return ok;
}

async function _onToggleEnabled(ev) {
  _settings.enabled = ev.target.checked;
  const ok = await _saveSetting('selfedit_enabled', ev.target.checked);
  // The Workshop nav pill is gated on this switch (data-feature-flag in
  // index.html), so flipping it here has to move the pill too — otherwise
  // turning self-edit off from inside the Workshop leaves the entry point
  // advertised until the next reload. settings.js owns the mirror; announce
  // rather than reach into it. Only on a successful write: a rejected
  // (non-admin) save must not hide the pill the server still shows.
  if (ok) {
    try {
      window.dispatchEvent(new CustomEvent('augmentum:feature-flag-changed', {
        detail: { key: 'selfeditEnabled', value: ev.target.checked },
      }));
    } catch (_) {}
  }
  _toast(ev.target.checked ? 'Self-edit enabled.' : 'Self-edit disabled — nothing will change itself.');
  _renderRailFoot();
  if (_lane === 'overview') _renderLane();
}

async function _onChangeAutonomy(ev) {
  _settings.autonomy = ev.target.value;
  await _saveSetting('selfedit_autonomy_level', ev.target.value);
  _toast(`Autonomy: ${ev.target.value === 'auto_verified' ? 'auto-promote verified' : 'propose — you decide'}.`);
  _renderRailFoot();
}

let _healthCache = null;
let _healthAt = 0;

async function _loadHealth(force = false) {
  // The health probe runs a PRAGMA quick_check on the main DB (~1s); cache it for
  // a few seconds so re-rendering Overview doesn't re-hit it.
  if (!force && _healthCache && (Date.now() - _healthAt) < 8000) return _healthCache;
  const pill = _overlay.querySelector('#ws-health');
  const { ok, status, body } = await _json(`${API}/health`);
  if (!pill) return null;
  if (status === 401) { pill.classList.add('ws-hidden'); return null; }
  _healthCache = body;
  _healthAt = Date.now();
  const score = typeof body.score === 'number' ? body.score : null;
  const healthy = ok && body.ok;
  pill.classList.toggle('ws-health-ok', healthy);
  pill.classList.toggle('ws-health-warn', !healthy && score !== null);
  pill.querySelector('.ws-health-label').textContent =
    score === null ? 'health —' : `${healthy ? 'healthy' : 'degraded'} ${(score * 100).toFixed(0)}%`;
  return body;
}

// ── lane routing ─────────────────────────────────────────────────────────

function _setLane(lane) {
  _lane = lane;
  _overlay.querySelectorAll('.ws-rail-item').forEach((b) =>
    b.classList.toggle('is-active', b.dataset.lane === lane));
  _renderLane();
}

function _renderRailFoot() {
  const el = _overlay.querySelector('#ws-rail-foot');
  if (!el) return;
  const on = _settings.enabled;
  el.innerHTML = `
    <div class="ws-posture ${on ? 'is-on' : ''}">
      <span class="ws-posture-dot"></span>
      <span>${on ? 'Self-edit on' : 'Self-edit off'}</span>
    </div>
    <div class="ws-posture-sub">${esc(_settings.autonomy === 'auto_verified' ? 'auto-promote verified' : 'propose · you decide')}</div>`;
}

async function _renderLane() {
  const c = _overlay.querySelector('#ws-content');
  if (!c) return;
  if (_lane === 'theater') return _renderTheater(c);
  if (_lane === 'overview') return _renderOverview(c);
  if (_lane === 'lineage') return _renderLineage(c);
  if (_lane === 'adapt') return _renderAdapt(c);
  if (_lane === 'debt') return _renderDebt(c);
  if (_lane === 'coverage') return _renderCoverage(c);
  if (_lane === 'apply') return _renderApply(c);
  if (_lane === 'evolve') return _renderEvolve(c);
  if (_lane === 'learned') return _renderLearned(c);
}

// ── Learned (the learning loop) ──────────────────────────────────────────

async function _renderLearned(c) {
  c.innerHTML = `
    <div class="ws-lane ws-lane-narrow">
      <div class="ws-lane-h">What it's learned about you</div>
      <p class="ws-lede">Every <strong>Keep</strong> or <strong>Revert</strong> teaches the system which
        kinds of change you trust. A shape you consistently keep becomes <em>trusted</em> — and the next
        time a change of that shape only passes the no-regression checks, your history lifts it from
        <em>"needs you"</em> toward <em>probable</em>. Trust is earned by accumulation, never given on one
        outcome, and never silently auto-ships.</p>
      <div id="ws-learned">${_skeletonList()}</div>

      <div class="ws-lane-h" style="margin-top:1.4rem">What it's learned works</div>
      <p class="ws-lede">The <em>verified skill graph</em> — derived from every self-edit's outcome (no
        new store; it's a reading of the never-pruned archive). Regions where edits <strong>ship and
        stick</strong> rise; regions of <strong>repeated rollback</strong> fall. It only advises — biasing
        which work to tackle and letting a failure-prone region skip the doomed cheap pass on the
        escalation ladder. It never auto-ships anything.</p>
      <div id="ws-regions">${_skeletonList()}</div>

      <div class="ws-lane-h" style="margin-top:1.4rem">Your taste — the Palate</div>
      <p class="ws-lede">A legible, per-you model of your <em>judgment</em>, distilled from your keep/revert
        history — the third oracle, between the mechanical checks and you. It predicts whether you'll keep
        a change so your attention goes where it teaches most. It's <strong>cold-start honest</strong>: until
        it has enough of your verdicts on a shape, it says so and defers. It only advises — it never
        auto-ships, and you can always overrule it.</p>
      <div id="ws-palate">${_skeletonList()}</div>
    </div>`;
  const box = c.querySelector('#ws-learned');
  const { ok, body } = await _json(`${API}/preferences`);
  if (!ok) { box.innerHTML = '<div class="ws-muted ws-err">Could not load preferences.</div>'; return; }
  const rows = body.preferences || [];
  if (!rows.length) {
    box.innerHTML = `<div class="ws-placeholder"><span class="ws-placeholder-glyph">◈</span>
      <p>Nothing learned yet. As you Keep and Revert changes in Lineage, the shapes you trust appear here
      (a shape is trusted after ${esc(body.min_samples || 3)} verdicts at ≥${Math.round((body.trust_threshold || 0.8) * 100)}% kept).</p></div>`;
  } else {
    box.innerHTML = rows.map(_learnedRow).join('');
  }
  await _renderRegions(c.querySelector('#ws-regions'));
  await _renderPalate(c.querySelector('#ws-palate'));
}

async function _renderPalate(box) {
  if (!box) return;
  const { ok, body } = await _json(`${API}/palate`);
  if (!ok) { box.innerHTML = '<div class="ws-muted ws-err">Could not load the Palate.</div>'; return; }
  const prof = body.profile || {};
  const statements = prof.statements || [];
  if (!statements.length) {
    box.innerHTML = `<div class="ws-placeholder"><span class="ws-placeholder-glyph">❋</span>
      <p>The Palate is warming up. As you Keep and Revert changes, it learns your taste per kind of change
      and starts predicting what you'll want — it needs about ${esc(prof.min_confident_evidence || 4)}
      verdicts before it speaks with confidence.</p></div>`;
    return;
  }
  const banner = prof.warming_up
    ? `<div class="ws-callout">Still warming up (${esc(prof.n_labels)} verdict(s)) — early reads, defers to you.</div>`
    : '';
  box.innerHTML = banner + statements.map((s) => {
    const pct = Math.round((s.keep_rate || 0) * 100);
    const cls = s.keep_rate >= 0.6 ? 'pos' : s.keep_rate <= 0.4 ? 'neg' : 'mixed';
    const firm = s.firm ? '' : ' <span class="ws-muted">(early)</span>';
    return `<div class="ws-palate-row ws-region-${cls}">
      <span class="ws-palate-stmt">${esc(s.statement)}${firm}</span>
      <span class="ws-palate-eq">${pct}% kept · ${esc(s.samples)} verdict(s)</span>
    </div>`;
  }).join('');
}

async function _renderRegions(box) {
  if (!box) return;
  const { ok, body } = await _json(`${API}/activation`);
  if (!ok) { box.innerHTML = '<div class="ws-muted ws-err">Could not load the skill graph.</div>'; return; }
  const graph = body.graph || {};
  const cal = body.calibration || {};
  const calHtml = _calibrationBanner(cal);
  const regions = (graph.top_regions || []).filter(([, w]) => Math.abs(w) > 0.01);
  if (!regions.length) {
    box.innerHTML = calHtml + `<div class="ws-placeholder"><span class="ws-placeholder-glyph">◈</span>
      <p>No verified region signal yet. As self-edits ship or roll back, the regions the system trusts
      (and the ones to be cautious in) appear here.${graph.attempts ? ` Folded ${esc(graph.attempts)} attempt(s) so far.` : ''}</p></div>`;
    return;
  }
  const max = Math.max(...regions.map(([, w]) => Math.abs(w)), 0.0001);
  box.innerHTML = calHtml + regions.map(([atom, w]) => _regionRow(atom, w, max)).join('');
}

function _calibrationBanner(cal) {
  if (!cal || !cal.n_attempts) return '';
  const grad = !!cal.graduated;
  const acc = Math.round((cal.accuracy || 0) * 100);
  const label = grad ? 'Calibrated — acting on the signal' : 'Shadow — observing, not yet acting';
  const detail = cal.n_predictions
    ? `${esc(cal.n_correct)}/${esc(cal.n_predictions)} predictions right (${acc}%) over ${esc(cal.n_attempts)} attempts`
    : `Watching ${esc(cal.n_attempts)} attempt(s); no confident calls yet`;
  return `
    <div class="ws-calib ${grad ? 'is-grad' : 'is-shadow'}" title="The router applies the skill graph only after its backtested accuracy clears the floor.">
      <span class="ws-calib-dot"></span>
      <span class="ws-calib-label">${esc(label)}</span>
      <span class="ws-calib-detail">${detail}</span>
    </div>`;
}

function _prettyAtom(atom) {
  const i = String(atom).indexOf(':');
  if (i < 0) return { kind: '', name: atom };
  return { kind: atom.slice(0, i), name: atom.slice(i + 1) };
}

function _regionRow(atom, w, max) {
  const { kind, name } = _prettyAtom(atom);
  const pos = w >= 0;
  const pct = Math.round((Math.abs(w) / max) * 100);
  return `
    <div class="ws-region-row ${pos ? 'is-pos' : 'is-neg'}">
      <div class="ws-region-top">
        <span class="ws-region-name">${esc(name)}${kind && kind !== 'sub' ? ` <span class="ws-region-kind">${esc(kind)}</span>` : ''}</span>
        <span class="ws-region-w">${pos ? '+' : ''}${w.toFixed(2)}</span>
      </div>
      <div class="ws-region-bar"><i style="width:${pct}%"></i></div>
    </div>`;
}

function _learnedRow(p) {
  const pct = Math.round((p.confidence || 0) * 100);
  return `
    <div class="ws-learn-row ${p.trusted ? 'is-trusted' : ''}">
      <div class="ws-learn-top">
        <span class="ws-learn-shape">${esc(p.shape)}</span>
        ${p.trusted ? '<span class="ws-badge ws-st-promoted">trusted</span>' : `<span class="ws-learn-progress">${esc(p.samples)} verdicts</span>`}
      </div>
      <div class="ws-learn-bar"><i style="width:${pct}%"></i></div>
      <div class="ws-learn-meta">${esc(p.kept)} kept · ${esc(p.reverted)} reverted · ${pct}% kept</div>
    </div>`;
}

// ── Overview ─────────────────────────────────────────────────────────────

async function _renderOverview(c) {
  const health = await _loadHealth();
  const counts = _attempts.reduce((a, x) => { a[x.status] = (a[x.status] || 0) + 1; return a; }, {});
  const total = _attempts.length;
  const score = health && typeof health.score === 'number' ? Math.round(health.score * 100) : null;
  const dims = (health && health.dimensions || []).filter((d) => d.measured);

  c.innerHTML = `
    <div class="ws-lane ws-lane-overview">
      <p class="ws-lede">Augmentum can change <em>itself</em> — its settings, its surfaces, its code, the
        prompts that shape how it thinks — and every change is <strong>verified</strong>,
        <strong>reversible</strong>, and <strong>never forgotten</strong>. This is where you watch and steer it.</p>

      <div class="ws-cards ws-cards-3">
        ${_engineCard('Surfaces', 'reshapes what it shows you', 'live', 'config / Adaptation runs now — read-back verified, instant, reversible')}
        ${_engineCard('Code', 'edits its own source', 'staged', 'worktree → verify → promote → rollback; awaiting the live driver')}
        ${_engineCard('Prompts', 'evolves how it thinks', 'building', 'GEPA eval-driven prompt evolution; scored against your own usage')}
      </div>

      <div class="ws-cards ws-cards-2">
        <section class="ws-card ws-card-health">
          <div class="ws-card-h">Application health</div>
          ${score === null ? '<div class="ws-muted">Sign in to read the live signal.</div>' : `
            <div class="ws-score"><span class="ws-score-num">${score}</span><span class="ws-score-unit">/100</span></div>
            <div class="ws-dims">${dims.map((d) => `
              <div class="ws-dim ${d.ok ? 'ok' : 'bad'}" title="${esc(d.detail || '')}">
                <span class="ws-dim-name">${esc(d.name)}</span>
                <span class="ws-dim-bar"><i style="width:${Math.round((d.score || 0) * 100)}%"></i></span>
              </div>`).join('')}</div>`}
        </section>
        <section class="ws-card ws-card-stats">
          <div class="ws-card-h">The archive</div>
          <div class="ws-stats">
            ${_stat(total, 'attempts')}
            ${_stat(counts.promoted || 0, 'promoted', 'ok')}
            ${_stat(counts.gated || 0, 'awaiting you', 'gated')}
            ${_stat((counts.rolled_back || 0) + (counts.rejected || 0) + (counts.failed || 0), 'reverted/failed', 'bad')}
          </div>
          <button class="ws-link" data-goto="lineage">Open the lineage →</button>
        </section>
      </div>

      ${!_settings.enabled ? `<div class="ws-callout">Self-edit is <strong>off</strong>. Turn it on in the header to let Augmentum
        propose and (on the green lane) apply improvements. Everything stays reversible and recorded.</div>` : ''}
    </div>`;
  c.querySelectorAll('[data-goto]').forEach((b) => b.addEventListener('click', () => _setLane(b.dataset.goto)));
}

function _engineCard(name, what, state, detail) {
  return `
    <section class="ws-card ws-engine">
      <div class="ws-engine-top"><span class="ws-engine-name">${esc(name)}</span>
        <span class="ws-pill ws-pill-${state}">${state}</span></div>
      <div class="ws-engine-what">${esc(what)}</div>
      <div class="ws-engine-detail">${esc(detail)}</div>
    </section>`;
}

function _stat(n, label, cls = '') {
  return `<div class="ws-stat ${cls}"><span class="ws-stat-n">${esc(n)}</span><span class="ws-stat-l">${esc(label)}</span></div>`;
}

// ── Lineage ──────────────────────────────────────────────────────────────

async function _renderLineage(c) {
  c.innerHTML = `
    <div class="ws-lane ws-lane-split">
      <div class="ws-split-list">
        <div class="ws-lane-h">Lineage <span class="ws-count" id="ws-count"></span>
          <span class="ws-lane-note">never pruned — rollback restores code, not the lesson</span></div>
        <div class="ws-list" id="ws-list">${_skeletonList()}</div>
      </div>
      <div class="ws-split-detail" id="ws-detail">${_emptyDetail()}</div>
    </div>`;
  await _loadList();
}

function _skeletonList() {
  return Array.from({ length: 4 }).map(() => '<div class="ws-skel"></div>').join('');
}
function _emptyDetail() {
  return '<div class="ws-placeholder"><span class="ws-placeholder-glyph">❡</span><p>Select an attempt to see what was tried, the verdict, and what was learned.</p></div>';
}

async function _loadList() {
  const list = _overlay.querySelector('#ws-list');
  const { status, body, ok } = await _json(`${API}/attempts?limit=100`);
  if (status === 401) { if (list) list.innerHTML = '<div class="ws-muted">Sign in to view the lineage.</div>'; return; }
  if (!ok) { if (list) list.innerHTML = `<div class="ws-muted ws-err">${esc(body.error || 'Could not load attempts.')}</div>`; return; }
  _attempts = body.attempts || [];
  const count = _overlay.querySelector('#ws-count');
  if (count) count.textContent = _attempts.length ? _attempts.length : '';
  if (!list) return;
  if (!_attempts.length) {
    list.innerHTML = `<div class="ws-placeholder"><span class="ws-placeholder-glyph">✶</span>
      <p>No attempts yet. The archive fills as Augmentum improves itself — start in <button class="ws-link" data-goto="adapt">Adapt</button>.</p></div>`;
    list.querySelectorAll('[data-goto]').forEach((b) => b.addEventListener('click', () => _setLane(b.dataset.goto)));
    return;
  }
  list.innerHTML = _attempts.map(_attemptRow).join('');
  list.querySelectorAll('.ws-row').forEach((el) =>
    el.addEventListener('click', () => _showDetail(el.dataset.id)));
}

function _attemptRow(a) {
  const meta = STATUS_META[a.status] || { label: a.status, cls: '' };
  const tier = (a.gate_verdict && a.gate_verdict.tier) || '';
  const nfiles = (a.files_changed || []).length;
  // provenance: ingested rows (git history, coder turns) sit in the same
  // never-pruned lineage as the engine's own attempts — badge them so a
  // backfilled commit is never mistaken for an autonomous self-edit.
  const src = a.source && a.source !== 'autonomous' ? a.source : '';
  return `
    <button class="ws-row ${a.id === _selectedId ? 'is-selected' : ''}" data-id="${esc(a.id)}">
      <span class="ws-row-surface" aria-hidden="true">${esc(SURFACE_ICON[a.surface] ?? '·')}</span>
      <span class="ws-row-main">
        <span class="ws-row-obj">${esc(a.objective || '(no objective)')}</span>
        <span class="ws-row-meta">${esc(a.surface || 'system')}${src ? ' · from ' + esc(src) : ''}${tier ? ' · ' + esc(tier) : ''}${nfiles ? ' · ' + nfiles + 'f' : ''} · ${esc(_timeAgo(a.updated_at || a.created_at))}</span>
      </span>
      <span class="ws-badge ${meta.cls}">${esc(meta.label)}</span>
    </button>`;
}

function _showDetail(id) {
  _selectedId = id;
  _overlay.querySelectorAll('.ws-row').forEach((el) => el.classList.toggle('is-selected', el.dataset.id === id));
  const a = _attempts.find((x) => x.id === id);
  const pane = _overlay.querySelector('#ws-detail');
  if (!a || !pane) return;
  const meta = STATUS_META[a.status] || { label: a.status, cls: '' };
  const tier = (a.gate_verdict && a.gate_verdict.tier) || '';
  const files = (a.files_changed || []).map((f) => `<li>${esc(f)}</li>`).join('') || '<li class="ws-muted">none</li>';
  const canVerdict = a.status === 'gated';
  pane.innerHTML = `
    <div class="ws-d">
      <div class="ws-d-top"><span class="ws-badge ${meta.cls}">${esc(meta.label)}</span>
        ${tier ? `<span class="ws-d-tier" title="${esc(TIER_NOTE[tier] || '')}">${esc(tier)}</span>` : ''}</div>
      <h3 class="ws-d-obj">${esc(a.objective || '')}</h3>
      <div class="ws-d-sub">${esc(a.surface || 'system')} · tier ${esc(a.tier || '—')}${a.source && a.source !== 'autonomous' ? ' · from ' + esc(a.source) : ''} · ${esc(_timeAgo(a.updated_at || a.created_at))}</div>
      ${a.outcome ? `<div class="ws-d-block"><span class="ws-d-k">Outcome</span><p>${esc(a.outcome)}</p></div>` : ''}
      ${a.lesson ? `<div class="ws-d-block ws-d-lesson"><span class="ws-d-k">Lesson</span><p>${esc(a.lesson)}</p></div>` : ''}
      <div class="ws-d-block"><span class="ws-d-k">Files</span><ul class="ws-d-files">${files}</ul></div>
      ${canVerdict ? `
        <div class="ws-verdict">
          <span class="ws-verdict-q">Your verdict</span>
          <button class="ws-btn ws-keep" data-id="${esc(a.id)}">Keep</button>
          <button class="ws-btn ws-revert" data-id="${esc(a.id)}">Revert</button>
        </div>` : ''}
    </div>`;
  if (canVerdict) {
    pane.querySelector('.ws-keep').addEventListener('click', () => _verdict(a.id, 'keep'));
    pane.querySelector('.ws-revert').addEventListener('click', () => _verdict(a.id, 'revert'));
  }
}

async function _verdict(id, decision) {
  const { ok, status, body } = await _json(`${API}/attempts/${encodeURIComponent(id)}/verdict`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision }),
  });
  if (status === 403) { _toast('Self-edit is disabled.', true); return; }
  if (!ok) { _toast(body.error || 'Verdict failed.', true); return; }
  _toast(decision === 'keep'
    ? 'Kept — staged in Go live (apply when you’re ready).'
    : (body.applied ? 'Reverted — code restored, the record kept.' : 'Rejected — recorded.'));
  await _loadList();
  const pane = _overlay.querySelector('#ws-detail');
  if (pane) pane.innerHTML = _emptyDetail();
}

// ── Adapt ────────────────────────────────────────────────────────────────

async function _renderAdapt(c) {
  const recent = _attempts.filter((a) => a.surface === 'config').slice(0, 8);
  c.innerHTML = `
    <div class="ws-lane ws-lane-narrow">
      <div class="ws-lane-h">Adapt the app to you</div>
      <p class="ws-lede">Change a setting and it takes effect — a config change is <em>data, not code</em>,
        so it's verified by reading the value back, applied live, and fully reversible. Pick from the
        list below (conversational asks like "make it denser" arrive with the classifier).</p>
      <div id="ws-adapt-list">${_skeletonList()}</div>
      <details class="ws-adapt-advanced">
        <summary>Advanced — set any setting by key</summary>
        <div class="ws-adapt">
          <label class="ws-field"><span>Setting key</span>
            <input type="text" id="ws-rk" class="ws-input" placeholder="e.g. ui.voiceSpeed" autocomplete="off"></label>
          <label class="ws-field"><span>Value</span>
            <input type="text" id="ws-rv" class="ws-input" placeholder="value" autocomplete="off"></label>
          <button class="ws-btn" id="ws-adapt-btn">Set →</button>
        </div>
        <div class="ws-adapt-result" id="ws-adapt-result"></div>
      </details>
      <div class="ws-lane-h ws-mt">Recent adaptations</div>
      <div class="ws-list ws-list-compact" id="ws-adapt-recent">
        ${recent.length ? recent.map(_attemptRow).join('') : '<div class="ws-muted">None yet.</div>'}
      </div>
    </div>`;
  c.querySelector('#ws-adapt-btn').addEventListener('click', _reshape);
  c.querySelectorAll('#ws-adapt-recent .ws-row').forEach((el) =>
    el.addEventListener('click', () => { _setLane('lineage'); setTimeout(() => _showDetail(el.dataset.id), 60); }));

  const list = c.querySelector('#ws-adapt-list');
  const { ok, status, body } = await _json(`${API}/adaptables`);
  if (status === 403) { list.innerHTML = '<div class="ws-callout">Self-edit is off — turn it on in the header.</div>'; return; }
  const items = (ok && body.adaptables) || [];
  if (!items.length) { list.innerHTML = '<div class="ws-muted">No adaptable settings available.</div>'; return; }
  list.innerHTML = items.map(_adaptRow).join('');
  list.querySelectorAll('.ws-ad-apply').forEach((b) => b.addEventListener('click', () => {
    const ctrl = _overlay.querySelector('#' + b.dataset.id);
    if (ctrl) _setAdaptable(b.dataset.skey, ctrl.value, b);
  }));
}

function _adaptRow(a) {
  const id = `ws-ad-${a.key}`;
  let control;
  if (a.type === 'bool') {
    const on = String(a.value) === 'true';
    control = `<select id="${id}" class="ws-input ws-ad-control"><option value="true"${on ? ' selected' : ''}>on</option><option value="false"${!on ? ' selected' : ''}>off</option></select>`;
  } else if (a.type === 'choice' && (a.options || []).length) {
    control = `<select id="${id}" class="ws-input ws-ad-control">${a.options.map((o) => `<option${String(a.value) === o ? ' selected' : ''}>${esc(o)}</option>`).join('')}</select>`;
  } else {
    control = `<input id="${id}" class="ws-input ws-ad-control" value="${esc(a.value || '')}" placeholder="${esc(a.default || '')}"${a.type === 'number' ? ' inputmode="decimal"' : ''}>`;
  }
  return `<div class="ws-ad-row">
    <div class="ws-ad-meta">
      <span class="ws-ad-label">${esc(a.label)}${a.is_set ? ' <span class="ws-ad-set">set</span>' : ''}</span>
      ${a.description ? `<span class="ws-ad-desc">${esc(a.description)}</span>` : ''}
    </div>
    <div class="ws-ad-controls">${control}
      <button class="ws-btn ws-ad-apply" data-id="${id}" data-skey="${esc(a.settings_key)}">Set</button></div>
  </div>`;
}

async function _setAdaptable(settingsKey, value, btn) {
  if (btn) { btn.disabled = true; btn.textContent = '…'; }
  const { ok, status, body } = await _json(`${API}/reshape`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ surface: 'config', key: settingsKey, value, ask: `set ${settingsKey}` }),
  });
  if (btn) { btn.disabled = false; btn.textContent = 'Set'; }
  if (status === 403) { _toast('Self-edit is off.', true); return; }
  const st = (body && body.status) || '';
  if (!ok || st === 'reverted' || st === 'failed') {
    _toast((body && body.detail) || 'Could not apply.', true); return;
  }
  _toast('Applied — verified by read-back. Takes effect on the next load.');
  await _loadList();
}

async function _reshape() {
  const key = _overlay.querySelector('#ws-rk').value.trim();
  const val = _overlay.querySelector('#ws-rv').value;
  const out = _overlay.querySelector('#ws-adapt-result');
  if (!key) { out.innerHTML = '<div class="ws-muted">Enter a setting key.</div>'; return; }
  out.innerHTML = '<div class="ws-muted">Adapting…</div>';
  const { ok, status, body } = await _json(`${API}/reshape`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ surface: 'config', key, value: val, ask: `set ${key}` }),
  });
  if (status === 403) { out.innerHTML = '<div class="ws-callout">Self-edit is off — turn it on in the header.</div>'; return; }
  if (!ok) { out.innerHTML = `<div class="ws-muted ws-err">${esc(body.error || 'Reshape failed.')}</div>`; return; }
  const st = body.status || 'done';
  const good = st === 'promoted';
  out.innerHTML = `<div class="ws-adapt-verdict ${good ? 'ok' : st === 'reverted' ? 'bad' : ''}">
    <span class="ws-badge ${good ? 'ws-st-promoted' : st === 'reverted' ? 'ws-st-reverted' : 'ws-st-gated'}">${esc(st)}</span>
    <span>${good ? `Verified by read-back and applied — ${esc(key)} is now ${esc(val)}.`
      : st === 'reverted' ? 'Verification failed; auto-reverted.' : esc(body.detail || st)}</span></div>`;
  await _loadList();
  const recent = _overlay.querySelector('#ws-adapt-recent');
  if (recent) {
    const r = _attempts.filter((a) => a.surface === 'config').slice(0, 8);
    recent.innerHTML = r.length ? r.map(_attemptRow).join('') : '<div class="ws-muted">None yet.</div>';
    recent.querySelectorAll('.ws-row').forEach((el) =>
      el.addEventListener('click', () => { _setLane('lineage'); setTimeout(() => _showDetail(el.dataset.id), 60); }));
  }
}

// ── Debt ─────────────────────────────────────────────────────────────────

let _debtTargets = {};   // id → target, for expand
let _debtGreenLane = false;  // does the live coder driver exist?

async function _renderDebt(c) {
  c.innerHTML = `
    <div class="ws-lane">
      <div class="ws-lane-h">Debt-paydown plan
        <span class="ws-h-actions">
          <label class="ws-toggle ws-frontier" title="On a hard target, climb local → primary → frontier model, carrying findings forward. Costs more.">
            <input type="checkbox" id="ws-frontier"><span class="ws-toggle-track"></span>
            <span class="ws-toggle-label">Allow frontier</span>
          </label>
          <button class="ws-btn ws-primary" id="ws-advise-btn" title="Let the agent read the list and recommend the best choices">✦ Ask the agent</button>
          <button class="ws-btn" id="ws-greenlane-btn" title="Fix the top auto-lane finding">Fix next</button>
          <button class="ws-btn" id="ws-reaudit-btn" title="Re-run the full audit (slow)">Re-audit</button>
        </span></div>
      <p class="ws-lede">A snapshot of the codebase, triaged. The <strong>auto-lane</strong> is mechanical —
        a fix the audit itself can confirm. <strong>Needs you</strong> is taste or risk: surfaced, never
        auto-touched. Expand any finding and <strong>Fix this</strong> to watch the agent work it live, or
        ask the agent to recommend an order.</p>
      <div id="ws-resume"></div>
      <div class="ws-capability">
        <span class="ws-cap-label">Or build something new</span>
        <input type="text" id="ws-cap-input" class="ws-input" placeholder="describe a capability — e.g. “a verb that opens the Workshop”" autocomplete="off">
        <button class="ws-btn ws-primary" id="ws-cap-go">Build →</button>
      </div>
      <div id="ws-advice"></div>
      <div id="ws-plan">${_skeletonList()}</div>
    </div>`;
  c.querySelector('#ws-reaudit-btn').addEventListener('click', () => _previewPlan(true));
  c.querySelector('#ws-greenlane-btn').addEventListener('click', () => _runGreenLane());
  c.querySelector('#ws-advise-btn').addEventListener('click', _adviseDebt);
  c.querySelector('#ws-cap-go').addEventListener('click', _buildCapability);
  c.querySelector('#ws-cap-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') _buildCapability(); });
  _maybeShowResume(c);
  _previewPlan(false);
}

async function _previewPlan(fresh) {
  const pane = _overlay.querySelector('#ws-plan');
  if (!pane) return;
  pane.innerHTML = `<div class="ws-muted">${fresh
    ? 'Running a full audit — this takes a few minutes…'
    : 'Reading the latest audit…'}</div>`;
  const { ok, status, body } = await _json(`${API}/propose`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dry_run: true, fresh: !!fresh }),
  });
  if (status === 403) { pane.innerHTML = '<div class="ws-callout">Self-edit is off — turn it on in the header to preview the plan.</div>'; return; }
  if (!ok) { pane.innerHTML = `<div class="ws-muted ws-err">${esc(body.error || 'Could not load the plan.')}</div>`; return; }
  _debtGreenLane = ('driver_ready' in body)
    ? !!body.driver_ready
    : (!!body.live || !(body.note || '').includes("isn't connected"));
  const gl = _overlay.querySelector('#ws-greenlane-btn');
  if (gl) {
    gl.disabled = !_debtGreenLane;
    gl.title = _debtGreenLane ? 'Run the mechanical auto-lane' : 'Needs the coder driver — connect it to enable';
  }
  const mech = body.targets || [], structural = body.structural || [];
  _debtTargets = {};
  [...mech, ...structural].forEach((t) => { _debtTargets[`${t.scanner}.${t.metric}`] = t; });
  const score = typeof body.baseline_score === 'number' ? body.baseline_score.toFixed(1) : '—';
  pane.innerHTML = `
    <div class="ws-plan-head">Baseline health <strong>${esc(score)}</strong>${body.deferred ? ` · ${esc(body.deferred)} more in the queue` : ''}${fresh ? '' : ' · cached'}</div>
    <div class="ws-plan-section">Auto-lane <span class="ws-plan-tag ok">mechanical</span></div>
    <div class="ws-cards ws-cards-2">${mech.length ? mech.map((t) => _debtCard(t, true)).join('')
      : '<div class="ws-muted">No mechanical debt — clean on the auto-lane.</div>'}</div>
    ${structural.length ? `<div class="ws-plan-section ws-mt">Needs you <span class="ws-plan-tag warn">judgment</span></div>
      <div class="ws-cards ws-cards-2">${structural.map((t) => _debtCard(t, false)).join('')}</div>` : ''}`;
  pane.querySelectorAll('.ws-debt').forEach((el) =>
    el.addEventListener('click', () => _toggleDebt(el)));
}

function _debtCard(t, mech) {
  const id = `${t.scanner}.${t.metric}`;
  const demand = t.origin === 'demand';
  // Demand items are lived user friction, not a scanner class — show their
  // provenance, not a raw scanner.metric, and badge them so they read as
  // "the user hit this," the whole point of surfacing them here.
  const meta = demand
    ? `${esc(t.note || 'from the user')}`
    : `${esc(t.scanner)}.${esc(t.metric)} · ${mech ? 'auto-lane' : 'needs you'}`;
  return `<section class="ws-card ws-debt ${mech ? 'mech' : 'struct'}${demand ? ' demand' : ''}" data-debt="${esc(id)}" tabindex="0" role="button" aria-expanded="false">
    <div class="ws-debt-top">
      <span class="ws-debt-title">${demand ? '<span class="ws-demand-tag">from the user</span> ' : ''}${esc(t.title)}</span>
      <span class="ws-debt-count"${demand ? ' title="times the user hit this"' : ''}>${esc(t.count)}</span>
    </div>
    <div class="ws-debt-meta">${meta}</div>
    ${_trustBadge(t.region_trust)}
    ${_palateBadge(t.palate)}
    <div class="ws-debt-expand" hidden></div>
  </section>`;
}

// The Palate's read-only taste prediction, shown beside a needs-you item so you
// see "the system thinks you'll keep/revert this" at the decision point. Only
// present when the Palate is confident enough to speak (cold-start honest).
function _palateBadge(pv) {
  if (!pv || !pv.speaks) return '';
  const pct = Math.round((pv.p_keep || 0) * 100);
  const cls = pv.lean === 'keep' ? 'pos' : pv.lean === 'revert' ? 'neg' : 'mixed';
  const label = pv.lean === 'keep' ? "you'll likely keep this"
    : pv.lean === 'revert' ? "you'll likely revert this"
    : 'could go either way';
  return `<div class="ws-debt-trust ws-region-${cls}" title="${esc(pv.rationale || '')}">
    ❋ ${esc(label)} · ${pct}%</div>`;
}

// The verified skill graph's per-class track record, shown beside a "needs you"
// item so the human decides with the archive's history in view. Read-only; absent
// (renders nothing) until enough attempts of this class have accrued.
function _trustBadge(rt) {
  if (!rt || typeof rt.score !== 'number') return '';
  const s = rt.score, conf = Math.round((rt.confidence || 0) * 100);
  const cls = s > 0.15 ? 'pos' : s < -0.15 ? 'neg' : 'mixed';
  const label = s > 0.15 ? 'landed this class before'
    : s < -0.15 ? 'this class often reverts'
    : 'mixed history here';
  return `<div class="ws-debt-trust ws-region-${cls}" title="${esc(rt.rationale || '')}">
    ${esc(label)} · ${conf}% conf</div>`;
}

// ── Coverage (the Oracle Foundry) ────────────────────────────────────────
// The autonomy frontier: (surface × intent-class) → strongest oracle tier the
// archive has ever achieved there. Cells that keep interrupting the human are
// the foundry worklist — each carries a composed ask for the capability lane.
// The human reads the ask and fires it; nothing is auto-selected or auto-run.

let _coverageWorklist = [];  // index → worklist cell (objectives are multiline)

async function _renderCoverage(c) {
  c.innerHTML = `
    <div class="ws-lane">
      <div class="ws-lane-h">Verification coverage</div>
      <p class="ws-lede">Every change class either has a <strong>mechanical oracle</strong> that can
        confirm it — or it lands on you. This map shows where the archive says coverage exists and
        where you keep getting interrupted. <strong>Propose oracle</strong> asks the engine to author
        the missing check itself — as an ordinary red-tier self-edit you review before anything ships.</p>
      <div id="ws-cov-gauge"></div>
      <div id="ws-cov-work">${_skeletonList()}</div>
      <div id="ws-cov-matrix"></div>
    </div>`;
  const { ok, body } = await _json(`${API}/coverage`);
  const work = c.querySelector('#ws-cov-work');
  if (!ok) { work.innerHTML = `<div class="ws-muted ws-err">${esc((body && body.error) || 'Could not load the coverage map.')}</div>`; return; }
  const g = body.gauge || {};
  c.querySelector('#ws-cov-gauge').innerHTML = `
    <div class="ws-plan-head">
      <strong>${esc(g.verified_attempts || 0)}</strong>/${esc(g.graded_attempts || 0)} graded attempts
      mechanically verified (${Math.round((g.verified_share || 0) * 100)}%) ·
      ${esc(g.interruptions || 0)} human interruption(s) ·
      ${esc(g.cells_covered || 0)}/${esc(g.cells_total || 0)} classes covered
      ${body.note ? ` · ${esc(body.note)}` : ''}</div>`;
  _coverageWorklist = body.worklist || [];
  if (!_coverageWorklist.length) {
    work.innerHTML = `<div class="ws-placeholder"><span class="ws-placeholder-glyph">◬</span>
      <p>No oracle-worthy clusters yet. A class joins the worklist once it has interrupted you
      ${esc(body.min_cluster || 2)}+ times without a mechanical oracle — the map below fills in as
      self-edits accrue.</p></div>`;
  } else {
    work.innerHTML = `
      <div class="ws-plan-section">Foundry worklist <span class="ws-plan-tag warn">keeps needing you</span></div>
      <div class="ws-cards ws-cards-2">${_coverageWorklist.map(_coverageCard).join('')}</div>`;
    work.querySelectorAll('[data-cov-idx]').forEach((btn) =>
      btn.addEventListener('click', () => _proposeOracle(Number(btn.dataset.covIdx), btn)));
  }
  const cells = body.cells || [];
  if (cells.length) {
    c.querySelector('#ws-cov-matrix').innerHTML = `
      <div class="ws-plan-section ws-mt">Full map</div>
      <div class="ws-cards ws-cards-2">${cells.map((cell) => `
        <section class="ws-card">
          <div class="ws-debt-top">
            <span class="ws-debt-title">${esc(cell.intent_class)} · ${esc(cell.surface)}</span>
            <span class="ws-plan-tag ${cell.covered ? 'ok' : 'warn'}">${esc(cell.best_tier || 'ungraded')}</span>
          </div>
          <div class="ws-debt-meta">${esc(cell.total)} attempt(s) · ${esc(cell.interruptions)} needed you ·
            ${esc(cell.kept)} kept / ${esc(cell.reverted)} reverted</div>
        </section>`).join('')}</div>`;
  }
}

function _coverageCard(cell, idx) {
  return `<section class="ws-card">
    <div class="ws-debt-top">
      <span class="ws-debt-title">${esc(cell.intent_class)} · ${esc(cell.surface)}</span>
      <span class="ws-debt-count">${esc(cell.interruptions + cell.probables)}</span>
    </div>
    <div class="ws-debt-meta">best so far: ${esc(cell.best_tier || 'ungraded')} ·
      ${esc(cell.kept)} kept / ${esc(cell.reverted)} reverted</div>
    ${cell.evidence && cell.evidence.length
      ? `<p class="ws-debt-obj">${esc(cell.evidence[0])}</p>` : ''}
    <div class="ws-debt-foot">
      <span class="ws-debt-confirm">an authored oracle would convert this class to verified</span>
      <button class="ws-btn ws-primary" data-cov-idx="${idx}"
        title="Author the missing check as a red-tier self-edit — you review before it lands">Propose oracle →</button>
    </div>
  </section>`;
}

async function _proposeOracle(idx, btn) {
  const cell = _coverageWorklist[idx];
  if (!cell || !cell.oracle_objective) { _toast('That cell is stale — reload the map.', true); return; }
  if (btn) { btn.disabled = true; btn.textContent = 'Starting…'; }
  const { ok, status, body } = await _json(`${API}/capability`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request: cell.oracle_objective }),
  });
  if (btn) { btn.disabled = false; btn.textContent = 'Propose oracle →'; }
  if (status === 403) { _toast('Self-edit is disabled — turn it on in the header.', true); return; }
  if (status === 409) { _toast((body && body.error) || "The edit driver isn't connected.", true); return; }
  if (!ok || !body.run_id) { _toast((body && body.error) || 'Could not start.', true); return; }
  _openTheater(body);  // streams live; the gated oracle lands in Go live for your verdict
}

function _toggleDebt(el) {
  const t = _debtTargets[el.dataset.debt];
  const exp = el.querySelector('.ws-debt-expand');
  if (!t || !exp) return;
  const open = el.getAttribute('aria-expanded') === 'true';
  el.setAttribute('aria-expanded', String(!open));
  exp.hidden = open;
  if (open) return;
  const mech = t.kind === 'mechanical';
  exp.innerHTML = `
    <p class="ws-debt-obj">${esc(t.objective || t.note || '')}</p>
    <div class="ws-debt-foot">
      <span class="ws-debt-confirm">${mech
        ? `confirmed by: ${esc(t.confirms_via === 'test' ? 'a new passing test' : 'the audit (finding gone + no regression)')}`
        : `resolution: ${esc(t.note || 'your judgment — surfaced, never auto-touched')}`}</span>
      ${mech
        ? `<button class="ws-btn ws-debt-fix" ${_debtGreenLane ? '' : 'disabled title="Needs the coder driver"'}>Fix this →</button>`
        : '<span class="ws-badge ws-st-gated">needs you</span>'}
    </div>`;
  const fix = exp.querySelector('.ws-debt-fix');
  if (fix) fix.addEventListener('click', (ev) => { ev.stopPropagation(); _runGreenLane(el.dataset.debt); });
}

function _frontierOn() {
  const cb = _overlay.querySelector('#ws-frontier');
  return !!(cb && cb.checked);
}

async function _runGreenLane(target = '') {
  if (!_debtGreenLane) {
    _toast("The coder driver isn't connected yet — the auto-lane runs once it is.", true);
    return;
  }
  const frontier = _frontierOn();
  const { ok, status, body } = await _json(`${API}/run`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target, allow_frontier: frontier, max_attempts: 1 }),
  });
  if (status === 403) { _toast('Self-edit is disabled.', true); return; }
  if (status === 409) { _toast(body.error || "The edit driver isn't connected.", true); return; }
  if (!ok || !body.run_id) { _toast(body.error || 'Could not start the run.', true); return; }
  _openTheater(body);
}

// ── Live run theater (the guided, streamed view) ──────────────────────────
//
// Better than the reference coder stream: the self-edit PIPELINE is the story.
// A ladder stepper (local → primary → frontier), a phase rail that lights up
// (target → evidence → candidate → agent → verify → verdict), the agent's tool
// activity with real subjects + edit previews, every verifier turning green as
// it passes, and a verdict card — all live, all reversible.

const _RUN_KEY = 'augmentum.workshop.activeRun';

const PHASES = [
  { id: 'target', label: 'Target', glyph: '⊙' },
  { id: 'evidence', label: 'Evidence', glyph: '✦' },
  { id: 'candidate', label: 'Worktree', glyph: '⎇' },
  { id: 'agent', label: 'Agent', glyph: '⚙' },
  { id: 'verify', label: 'Verify', glyph: '✓' },
  { id: 'verdict', label: 'Verdict', glyph: '◈' },
];

const AGENT_ICON = {
  message: '💬', tool_call: '🔍', file_change: '✎',
  command_exec: '⌘', mcp_call: '⚡', completed: '✓', failed: '✕',
};

const VERIFIER_ICON = { pass: '✓', fail: '✕', skip: '·' };

let _run = null;        // live theater state
let _runAbort = null;   // AbortController for the SSE fetch
let _prevLane = 'debt'; // where to return on Close

function _saveActiveRun(id) {
  try { id ? sessionStorage.setItem(_RUN_KEY, id) : sessionStorage.removeItem(_RUN_KEY); }
  catch { /* sessionStorage unavailable */ }
}
function _readActiveRun() {
  try { return sessionStorage.getItem(_RUN_KEY) || ''; } catch { return ''; }
}

function _openTheater(meta, { since = 0, resuming = false } = {}) {
  _run = {
    id: meta.run_id, title: meta.title || 'Self-edit', target: meta.target || '',
    ladder: meta.ladder || [], since,
    reached: {}, rungs: [], feed: [], verifiers: [], verdict: null,
    baseline: null, status: 'running',
  };
  _saveActiveRun(_run.id);
  if (_lane !== 'theater') _prevLane = _lane;
  _lane = 'theater';
  _overlay.querySelectorAll('.ws-rail-item').forEach((b) => b.classList.remove('is-active'));
  _renderTheater(_overlay.querySelector('#ws-content'));
  if (resuming) _toast('Reattached to the live run.');
  _streamRun(_run.id, since);
}

function _renderTheater(c) {
  if (!c || !_run) return;
  const running = _run.status === 'running';
  c.innerHTML = `
    <div class="ws-theater">
      <div class="ws-th-head">
        <button class="ws-link ws-th-back" id="ws-th-back">← Debt</button>
        <div class="ws-th-titlewrap">
          <span class="ws-th-title">${esc(_run.title)}</span>
          ${_run.target ? `<span class="ws-th-target">${esc(_run.target)}</span>` : ''}
        </div>
        <div class="ws-th-actions">
          <span class="ws-th-live ${running ? 'is-live' : ''}" id="ws-th-live">${running ? '● live' : 'done'}</span>
          ${running ? '<button class="ws-btn ws-th-stop" id="ws-th-stop">Stop</button>' : ''}
        </div>
      </div>
      ${_run.ladder.length > 1 ? `<div class="ws-th-ladder" id="ws-th-ladder">${_ladderHtml()}</div>` : ''}
      <div class="ws-th-rail" id="ws-th-rail">${_phaseRailHtml()}</div>
      <div class="ws-th-cols">
        <section class="ws-th-feed-wrap">
          <div class="ws-th-h">Working</div>
          <div class="ws-th-feed" id="ws-th-feed">${_feedHtml() || '<div class="ws-muted">Starting…</div>'}</div>
        </section>
        <aside class="ws-th-side">
          <div class="ws-th-h">Verification</div>
          <div class="ws-th-verifiers" id="ws-th-verifiers">${_verifiersHtml() || '<div class="ws-muted">Pending the edit…</div>'}</div>
          <div id="ws-th-verdict">${_run.verdict ? _verdictHtml(_run.verdict) : ''}</div>
        </aside>
      </div>
    </div>`;
  c.querySelector('#ws-th-back').addEventListener('click', _closeTheater);
  const stop = c.querySelector('#ws-th-stop');
  if (stop) stop.addEventListener('click', _stopRun);
  _bindVerdictButtons(c);
}

function _ladderHtml() {
  return _run.ladder.map((m, i) => {
    const r = _run.rungs[i] || {};
    const cls = r.state === 'landed' ? 'landed' : r.state === 'done'
      ? (r.status === 'gated' ? 'ok' : 'spent') : r.state === 'start' ? 'active' : '';
    return `<span class="ws-rung ${cls}" title="${esc(r.tier || r.status || 'queued')}">
      <span class="ws-rung-dot"></span>${esc(m)}${r.tier ? `<span class="ws-rung-tier">${esc(r.tier)}</span>` : ''}</span>`;
  }).join('<span class="ws-rung-arrow">→</span>');
}

function _phaseRailHtml() {
  const order = PHASES.map((p) => p.id);
  const maxReached = Math.max(-1, ...Object.keys(_run.reached).map((k) => order.indexOf(k)));
  return PHASES.map((p, i) => {
    const done = i < maxReached || (_run.status !== 'running' && i <= maxReached);
    const active = i === maxReached && _run.status === 'running';
    return `<div class="ws-th-phase ${done ? 'done' : ''} ${active ? 'active' : ''}">
      <span class="ws-th-phase-glyph">${p.glyph}</span>
      <span class="ws-th-phase-label">${esc(p.label)}</span>
    </div>`;
  }).join('<span class="ws-th-phase-link"></span>');
}

function _feedHtml() {
  return _run.feed.map(_feedRow).join('');
}
function _feedRow(f) {
  const icon = AGENT_ICON[f.sub] || '·';
  const subj = f.tool ? `<span class="ws-fd-tool">${esc(f.tool)}</span>` : '';
  const main = esc(f.text || f.path || '');
  const detail = f.detail
    ? `<pre class="ws-fd-detail${f.sub === 'file_change' ? ' diff' : ''}">${esc(f.detail)}</pre>` : '';
  return `<div class="ws-fd ws-fd-${esc(f.sub)}">
    <span class="ws-fd-icon">${icon}</span>
    <div class="ws-fd-body">${subj}<span class="ws-fd-text">${main}</span>${detail}</div>
  </div>`;
}

function _verifiersHtml() {
  return _run.verifiers.map((v) => {
    const ic = VERIFIER_ICON[v.status] || '·';
    return `<div class="ws-vf ws-vf-${esc(v.status)}${v.required ? ' req' : ''}">
      <span class="ws-vf-ic">${ic}</span>
      <div class="ws-vf-body">
        <span class="ws-vf-name">${esc(v.name)}<span class="ws-vf-oracle">${esc(v.oracle)}</span></span>
        ${v.detail ? `<span class="ws-vf-detail">${esc(v.detail)}</span>` : ''}
      </div>
    </div>`;
  }).join('');
}

function _verdictHtml(v) {
  const tier = v.tier || v.final_status || 'done';
  const tierCls = tier === 'verified' ? 'ws-st-promoted'
    : tier === 'probable' ? 'ws-st-gated'
    : tier === 'failed' ? 'ws-st-failed' : 'ws-st-gated';
  const gated = v.status === 'gated';
  return `
    <div class="ws-th-verdict ${v.auto_promotable ? 'is-verified' : ''}">
      <div class="ws-th-verdict-top">
        <span class="ws-badge ${tierCls}">${esc(tier)}</span>
        ${v.auto_promotable ? '<span class="ws-th-auto">mechanically proven · safe to auto-promote</span>' : ''}
      </div>
      ${v.outcome ? `<p class="ws-th-outcome">${esc(v.outcome)}</p>` : ''}
      ${(v.files || []).length ? `<div class="ws-th-files">${v.files.map((f) => `<span class="ws-chip">${esc(f)}</span>`).join('')}</div>` : ''}
      ${v.lesson ? `<p class="ws-th-lesson"><span class="ws-d-k">Lesson</span> ${esc(v.lesson)}</p>` : ''}
      <div class="ws-th-verdict-actions">
        ${gated ? `<button class="ws-btn ws-keep" data-vid="${esc(v.attempt_id || '')}">Keep</button>
                   <button class="ws-btn ws-revert" data-vid="${esc(v.attempt_id || '')}">Revert</button>` : ''}
        <button class="ws-link" id="ws-th-tolineage">Open in Lineage →</button>
      </div>
    </div>`;
}

function _bindVerdictButtons(c) {
  const box = c.querySelector('#ws-th-verdict');
  if (!box) return;
  const keep = box.querySelector('.ws-keep');
  const rev = box.querySelector('.ws-revert');
  const line = box.querySelector('#ws-th-tolineage');
  if (keep) keep.addEventListener('click', () => _verdict(keep.dataset.vid, 'keep').then(() => _closeTheater('lineage')));
  if (rev) rev.addEventListener('click', () => _verdict(rev.dataset.vid, 'revert').then(() => _closeTheater('lineage')));
  if (line) line.addEventListener('click', () => _closeTheater('lineage'));
}

// incremental DOM updates (avoid a full re-render per event — keep it smooth)
function _appendFeed(f) {
  _run.feed.push(f);
  const box = _overlay.querySelector('#ws-th-feed');
  if (!box) return;
  if (box.querySelector('.ws-muted')) box.innerHTML = '';
  box.insertAdjacentHTML('beforeend', _feedRow(f));
  box.scrollTop = box.scrollHeight;
}
function _refreshRail() {
  const rail = _overlay.querySelector('#ws-th-rail');
  if (rail) rail.innerHTML = _phaseRailHtml();
  const lad = _overlay.querySelector('#ws-th-ladder');
  if (lad) lad.innerHTML = _ladderHtml();
}
function _refreshVerifiers() {
  const box = _overlay.querySelector('#ws-th-verifiers');
  if (box) box.innerHTML = _verifiersHtml() || '<div class="ws-muted">Pending the edit…</div>';
}

function _onRunEvent(ev) {
  if (!_run) return;
  switch (ev.kind) {
    case 'run':
      if (ev.ladder && ev.ladder.length) _run.ladder = ev.ladder;
      _refreshRail();
      break;
    case 'phase':
      _run.reached[ev.phase] = true;
      if (ev.phase === 'target' && typeof ev.baseline_score === 'number') _run.baseline = ev.baseline_score;
      if (ev.text) _appendFeed({ sub: 'message', text: `▸ ${ev.text}` });
      if (ev.phase === 'evidence' && (ev.findings || []).length) {
        _appendFeed({ sub: 'message', text: `   ${ev.findings.join(', ')}` });
      }
      _refreshRail();
      break;
    case 'rung': {
      _run.rungs[ev.index] = { ...(_run.rungs[ev.index] || {}), ...ev };
      if (ev.state === 'start' && ev.text) _appendFeed({ sub: 'message', text: `⟐ ${ev.text}` });
      if (ev.state === 'landed' && ev.text) _appendFeed({ sub: 'completed', text: ev.text });
      _refreshRail();
      break;
    }
    case 'agent':
      _appendFeed({ sub: ev.sub, tool: ev.tool, path: ev.path, text: ev.text, detail: ev.detail });
      if (ev.sub === 'file_change') { _run.reached.agent = true; }
      break;
    case 'verifier':
      _run.verifiers.push(ev);
      _run.reached.verify = true;
      _refreshVerifiers();
      _refreshRail();
      break;
    case 'verdict':
      _run.verdict = ev;
      _run.reached.verdict = true;
      _refreshRail();
      { const box = _overlay.querySelector('#ws-th-verdict');
        if (box) { box.innerHTML = _verdictHtml(ev); _bindVerdictButtons(_overlay.querySelector('#ws-content')); } }
      break;
    case 'done':
      _run.status = ev.status || 'done';
      _saveActiveRun('');
      _refreshRail();
      { const live = _overlay.querySelector('#ws-th-live');
        if (live) { live.classList.remove('is-live'); live.textContent = _run.status === 'cancelled' ? 'stopped' : 'done'; } }
      const stop = _overlay.querySelector('#ws-th-stop');
      if (stop) stop.remove();
      if (!_run.verdict && _run.status !== 'cancelled') {
        _appendFeed({ sub: ev.ok ? 'completed' : 'failed',
          text: ev.error || `Finished: ${ev.final_status || _run.status}` });
      }
      _loadList();  // refresh the lineage in the background
      break;
    case 'failed':
      _appendFeed({ sub: 'failed', text: ev.text || 'run error' });
      break;
  }
}

async function _streamRun(runId, since = 0) {
  if (_runAbort) _runAbort.abort();
  _runAbort = new AbortController();
  try {
    const resp = await fetch(`${API}/run/${encodeURIComponent(runId)}/stream?since=${since}`,
      { signal: _runAbort.signal });
    if (!resp.ok || !resp.body) { _toast('Could not attach to the run.', true); return; }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          if (typeof ev.seq === 'number') _run.since = ev.seq;
          _onRunEvent(ev);
        } catch { /* partial line */ }
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') {
      // dropped mid-run — the server-owned run keeps going; offer reattach
      if (_run && _run.status === 'running') {
        _appendFeed({ sub: 'message', text: '… connection dropped — the run continues on the server. Reattaching…' });
        setTimeout(() => { if (_run && _run.status === 'running') _streamRun(runId, _run.since || 0); }, 1500);
      }
    }
  }
}

async function _stopRun() {
  if (!_run) return;
  await _json(`${API}/run/${encodeURIComponent(_run.id)}/stop`, { method: 'POST' });
  _toast('Stopping the run…');
}

function _closeTheater(goLane = '') {
  if (_runAbort) { _runAbort.abort(); _runAbort = null; }
  const lane = goLane || _prevLane || 'debt';
  _run = null;
  _setLane(lane);
}

// ── Go live (staged apply + checkpoints + restart) ────────────────────────
//
// Kept code edits COLLECT in the isolated clone. Here the user takes the whole
// set live in one deliberate step — checkpoint → write → restart — and can
// revert to any prior checkpoint. Nothing touches the running app until you
// press the button.

const FILE_ICON = { A: '＋', M: '∼', D: '－' };

async function _renderApply(c) {
  c.innerHTML = `
    <div class="ws-lane ws-lane-apply">
      <div class="ws-lane-h">Go live
        <span class="ws-h-actions"><button class="ws-btn" id="ws-apply-refresh">Refresh</button></span>
      </div>
      <p class="ws-lede">Accepted code edits <strong>collect</strong> here, committed and reversible in an
        isolated copy — <em>nothing changes the running app until you say so</em>. Apply takes the whole set
        live (a quick restart); every apply leaves a <strong>checkpoint</strong> you can revert to.</p>
      <div id="ws-apply-body">${_skeletonList()}</div>
      <div class="ws-lane-h ws-mt">Checkpoints <span class="ws-lane-note">restore points — revert to a prior state</span></div>
      <div id="ws-apply-checkpoints">${_skeletonList()}</div>
    </div>`;
  c.querySelector('#ws-apply-refresh').addEventListener('click', () => _renderApply(c));
  _loadPending();
}

async function _loadPending() {
  const body = _overlay.querySelector('#ws-apply-body');
  const cpBox = _overlay.querySelector('#ws-apply-checkpoints');
  if (!body) return;
  const { ok, status, body: d } = await _json(`${API}/pending`);
  if (status === 403) { body.innerHTML = '<div class="ws-callout">Self-edit is off — turn it on in the header.</div>'; return; }
  if (status === 409) { body.innerHTML = '<div class="ws-muted">No self-edit repo is wired.</div>'; return; }
  if (!ok) { body.innerHTML = `<div class="ws-muted ws-err">${esc((d && d.error) || 'Could not load pending changes.')}</div>`; return; }

  if (cpBox) {
    const cps = d.checkpoints || [];
    cpBox.innerHTML = cps.length ? cps.map(_checkpointRow).join('')
      : '<div class="ws-muted">No checkpoints yet — they appear after your first apply.</div>';
    cpBox.querySelectorAll('.ws-cp-revert').forEach((b) =>
      b.addEventListener('click', () => _revertCheckpoint(b.dataset.cid)));
  }

  if (!d.has_changes) {
    body.innerHTML = `<div class="ws-placeholder"><span class="ws-placeholder-glyph">✓</span>
      <p>Everything accepted is already live. Kept edits from the Debt lane will collect here.</p></div>`;
    return;
  }
  const blocked = d.blocked_count || 0;
  const files = (d.files || []).map(_pendingFileRow).join('');
  const attempts = (d.attempts || []).map((a) => `
    <div class="ws-pend-attempt">
      <span class="ws-pend-obj">${esc(a.objective || '(edit)')}</span>
      <span class="ws-pend-meta">${esc(a.surface || 'system')}${a.tier ? ' · ' + esc(a.tier) : ''} · ${(a.files || []).length}f</span>
    </div>`).join('');
  body.innerHTML = `
    <div class="ws-pend">
      <div class="ws-pend-top">
        <div class="ws-pend-count"><span class="ws-pend-n">${esc(d.applyable_count)}</span> ready to apply${blocked ? ` · <span class="ws-pend-blocked">${esc(blocked)} blocked</span>` : ''}</div>
        <button class="ws-btn ws-primary ws-apply-go" ${d.applyable_count ? '' : 'disabled'}>Apply &amp; restart →</button>
      </div>
      ${blocked ? `<div class="ws-callout ws-pend-warn">${esc(blocked)} frontend file(s) can't apply yet — the <code>ui/</code> mount is read-only.
        Recreate the container (<code>start.bat</code>) after the compose change to enable, then they'll apply too.</div>` : ''}
      ${attempts ? `<div class="ws-pend-attempts">${attempts}</div>` : ''}
      <div class="ws-pend-files">${files}</div>
      ${d.diff ? `<details class="ws-pend-diff"><summary>Diff stat</summary><pre>${esc(d.diff)}</pre></details>` : ''}
    </div>`;
  const go = body.querySelector('.ws-apply-go');
  if (go) go.addEventListener('click', _applyAndRestart);
}

function _pendingFileRow(f) {
  return `<div class="ws-pend-file ${f.applyable ? '' : 'blocked'}">
    <span class="ws-pend-ic" title="${esc(f.change)}">${FILE_ICON[f.change] || '∼'}</span>
    <span class="ws-pend-path">${esc(f.path)}</span>
    ${f.applyable ? '<span class="ws-pend-ok">ready</span>'
      : `<span class="ws-pend-block" title="${esc(f.reason)}">blocked</span>`}
  </div>`;
}

function _checkpointRow(cp) {
  return `<div class="ws-cp">
    <div class="ws-cp-main">
      <span class="ws-cp-when">${esc((cp.created || '').replace('T', ' '))}</span>
      <span class="ws-cp-meta">${esc(cp.file_count || (cp.files || []).length)} file(s) · ${esc(cp.label || 'apply')}</span>
    </div>
    <button class="ws-btn ws-cp-revert" data-cid="${esc(cp.id)}">Revert to this</button>
  </div>`;
}

async function _applyAndRestart() {
  const go = _overlay.querySelector('.ws-apply-go');
  if (go) { go.disabled = true; go.textContent = 'Applying…'; }
  const { ok, status, body } = await _json(`${API}/apply`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ restart: true }),
  });
  if (status === 403) { _toast('Self-edit is disabled.', true); return; }
  if (!ok) {
    _toast((body && body.error) || 'Apply failed.', true);
    if (go) { go.disabled = false; go.textContent = 'Apply & restart →'; }
    return;
  }
  const n = (body.applied || []).length + (body.deleted || []).length;
  if (body.restarting) {
    await _restartOverlay(`Applied ${n} file(s) — restarting the server…`);
  } else {
    _toast(n ? `Applied ${n} file(s) (no restart needed).` : 'Nothing applied.');
    _loadPending();
  }
}

async function _revertCheckpoint(cid) {
  if (!cid) return;
  const { ok, status, body } = await _json(`${API}/checkpoints/${encodeURIComponent(cid)}/restore`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ restart: true }),
  });
  if (status === 403) { _toast('Self-edit is disabled.', true); return; }
  if (!ok) { _toast((body && body.error) || 'Revert failed.', true); return; }
  if (body.restarting) {
    await _restartOverlay('Reverted to the checkpoint — restarting the server…');
  } else {
    _toast('Reverted.');
    _loadPending();
  }
}

// A full-screen "restarting…" cover that polls until the app is back, then
// reloads the page (so any new frontend assets are picked up).
function _restartOverlay(message) {
  return new Promise((resolve) => {
    let el = document.getElementById('ws-restart-cover');
    if (!el) {
      el = document.createElement('div');
      el.id = 'ws-restart-cover';
      el.className = 'ws-restart-cover';
      document.body.appendChild(el);
    }
    const start = Date.now();
    el.innerHTML = `
      <div class="ws-restart-card">
        <div class="ws-restart-spin"></div>
        <div class="ws-restart-msg">${esc(message)}</div>
        <div class="ws-restart-sub" id="ws-restart-sub">waiting for the server to come back…</div>
      </div>`;
    el.classList.add('show');
    const sub = el.querySelector('#ws-restart-sub');
    let downSeen = false;
    const poll = async () => {
      const elapsed = Math.round((Date.now() - start) / 1000);
      if (sub) sub.textContent = `waiting for the server to come back… ${elapsed}s`;
      let up = false;
      try {
        const r = await fetch('/api/ui-version', { cache: 'no-store' });
        up = r.ok || r.status === 200;
      } catch { up = false; }
      // require seeing it go DOWN first (the restart begins ~1.5s in), so we
      // don't false-positive on the still-alive pre-restart server.
      if (!up) downSeen = true;
      if (up && (downSeen || elapsed > 4)) {
        if (sub) sub.textContent = 'back online — reloading…';
        setTimeout(() => window.location.reload(), 600);
        resolve(true);
        return;
      }
      if (elapsed > 120) {
        if (sub) sub.textContent = 'taking longer than expected — reload manually when ready.';
        resolve(false);
        return;
      }
      setTimeout(poll, 1500);
    };
    setTimeout(poll, 2000);
  });
}

async function _buildCapability() {
  const input = _overlay.querySelector('#ws-cap-input');
  const ask = input && input.value.trim();
  if (!ask) { _toast('Describe the capability first.', true); return; }
  if (!_debtGreenLane) { _toast("The edit driver isn't connected yet.", true); return; }
  const go = _overlay.querySelector('#ws-cap-go');
  if (go) { go.disabled = true; go.textContent = 'Starting…'; }
  const { ok, status, body } = await _json(`${API}/capability`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request: ask }),
  });
  if (go) { go.disabled = false; go.textContent = 'Build →'; }
  if (status === 403) { _toast('Self-edit is disabled.', true); return; }
  if (status === 409) { _toast(body.error || "The edit driver isn't connected.", true); return; }
  if (!ok || !body.run_id) { _toast((body && body.error) || 'Could not start.', true); return; }
  _openTheater(body);   // streams in the theater; a verified build lands in Go live
}

async function _maybeShowResume(c) {
  const box = c.querySelector('#ws-resume');
  const id = _readActiveRun();
  if (!box || !id) return;
  // Confirm it's genuinely still live on the server before offering resume.
  const { ok, body } = await _json(`${API}/run/${encodeURIComponent(id)}`);
  const live = ok && body && body.status === 'running';
  if (!live) { _saveActiveRun(''); return; }
  box.innerHTML = `<div class="ws-callout ws-resume">
    <span>A self-edit is running live${body.target ? ` on <strong>${esc(body.target)}</strong>` : ''}.</span>
    <button class="ws-btn ws-primary" id="ws-resume-btn">Watch it →</button></div>`;
  box.querySelector('#ws-resume-btn').addEventListener('click', () =>
    _openTheater({ run_id: id, title: body.title, target: body.target, ladder: body.ladder },
      { resuming: true }));
}

async function _adviseDebt() {
  const box = _overlay.querySelector('#ws-advice');
  if (!box) return;
  box.innerHTML = '<div class="ws-advice-loading">The agent is reading the list…</div>';
  const { ok, status, body } = await _json(`${API}/debt/advise`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
  });
  if (status === 403) { box.innerHTML = '<div class="ws-callout">Self-edit is off — turn it on to ask the agent.</div>'; return; }
  if (!ok || body.available === false) {
    box.innerHTML = `<div class="ws-muted">${esc((body && body.note) || 'The agent is unavailable right now — the triaged list below still stands.')}</div>`;
    return;
  }
  const recs = body.recommendations || [];
  if (!recs.length) {
    box.innerHTML = `<div class="ws-muted">${esc(body.summary || 'The agent had nothing to prioritize.')}</div>`;
    return;
  }
  box.innerHTML = `
    <section class="ws-advice">
      <div class="ws-advice-head"><span class="ws-advice-glyph">✦</span>
        <span>${esc(body.summary || 'Recommended order')}</span></div>
      <ol class="ws-advice-list">
        ${recs.map((r) => `
          <li class="ws-advice-item">
            <span class="ws-advice-rank">${esc(r.rank)}</span>
            <div class="ws-advice-body">
              <div class="ws-advice-title">${esc(r.title)}
                <span class="ws-badge ${r.kind === 'mechanical' ? 'ws-st-promoted' : 'ws-st-gated'}">${esc(r.kind === 'mechanical' ? 'auto-lane' : 'needs you')}</span>
                ${r.effort ? `<span class="ws-advice-effort">${esc(r.effort)}</span>` : ''}
                ${r.group ? `<span class="ws-advice-group">${esc(r.group)}</span>` : ''}</div>
              <div class="ws-advice-why">${esc(r.rationale)}</div>
              ${r.approach ? `<div class="ws-advice-how">→ ${esc(r.approach)}</div>` : ''}
            </div>
          </li>`).join('')}
      </ol>
    </section>`;
}

// ── Evolve ───────────────────────────────────────────────────────────────

const EVOLVE_PRESETS = [
  { name: 'Commit messages', prompt: 'Write a commit message for the change.',
    goal: 'a single conventional-commit line — type(scope): summary, imperative mood, under 72 chars, no body or fluff' },
  { name: 'Code-review summary', prompt: 'Summarize this code review.',
    goal: 'a crisp prioritized summary: blockers first then nits, each with file:line and a concrete fix' },
  { name: 'PR description', prompt: 'Write a pull-request description.',
    goal: 'what changed, why, and how to test — skimmable, specific, no filler' },
  { name: 'Error message', prompt: 'Write a user-facing error message.',
    goal: 'clear, blameless, actionable — what happened and exactly what to do next' },
];

let _evPollTimer = null;
let _evArtifacts = [];     // registered overridable prompts (the real apply targets)
let _evTarget = null;      // the selected artifact spec, or null for free-form

async function _renderEvolve(c) {
  c.innerHTML = `
    <div class="ws-lane ws-lane-narrow">
      <div class="ws-lane-h">Evolve a prompt</div>
      <p class="ws-lede">Give a prompt and what good looks like. Augmentum builds an eval set from the
        goal, reflects on where the prompt falls short, mutates it, and keeps a new version
        <em>only</em> if it beats the original on <strong>held-out</strong> cases. Pick a real
        <strong>target</strong> to make the winner live, or run free-form to just see the result.</p>
      <label class="ws-field"><span>Target (what to change in the app)</span>
        <select id="ws-ev-target" class="ws-input"><option value="">Free-form — just show me the result</option></select></label>
      <div id="ws-ev-target-state"></div>
      <div class="ws-chips" id="ws-ev-presets">
        ${EVOLVE_PRESETS.map((p, i) => `<button class="ws-chip" data-preset="${i}">${esc(p.name)}</button>`).join('')}
      </div>
      <label class="ws-field"><span>Prompt to improve</span>
        <textarea id="ws-ev-prompt" class="ws-input ws-textarea" rows="4" placeholder="paste a system prompt to evolve…"></textarea></label>
      <label class="ws-field"><span>What good looks like (the goal)</span>
        <textarea id="ws-ev-goal" class="ws-input ws-textarea" rows="2" placeholder="describe a great output…"></textarea></label>
      <div class="ws-evolve-actions">
        <button class="ws-btn ws-primary" id="ws-ev-run">✶ Evolve</button>
        <span class="ws-toolbar-note" id="ws-ev-note"></span>
      </div>
      <div id="ws-ev-result"></div>
    </div>`;
  c.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => {
    const p = EVOLVE_PRESETS[+b.dataset.preset];
    c.querySelector('#ws-ev-prompt').value = p.prompt;
    c.querySelector('#ws-ev-goal').value = p.goal;
  }));
  c.querySelector('#ws-ev-run').addEventListener('click', _startEvolve);
  c.querySelector('#ws-ev-target').addEventListener('change', _onEvTarget);

  // load the real overridable targets
  const { ok, body } = await _json(`${API}/prompts`);
  _evArtifacts = (ok && body.prompts) || [];
  const sel = c.querySelector('#ws-ev-target');
  if (sel && _evArtifacts.length) {
    _evArtifacts.forEach((p) => {
      const o = document.createElement('option');
      o.value = p.slug;
      o.textContent = `${p.label}${p.overridden ? ' (overridden)' : ''}${p.user_facing ? '' : ' · internal'}`;
      sel.appendChild(o);
    });
  }
}

function _onEvTarget(ev) {
  const slug = ev.target.value;
  const stateEl = _overlay.querySelector('#ws-ev-target-state');
  const presets = _overlay.querySelector('#ws-ev-presets');
  _evTarget = _evArtifacts.find((p) => p.slug === slug) || null;
  if (!_evTarget) {
    stateEl.innerHTML = '';
    if (presets) presets.style.display = '';
    return;
  }
  if (presets) presets.style.display = 'none';
  _overlay.querySelector('#ws-ev-prompt').value = _evTarget.effective || _evTarget.default || '';
  const goalEl = _overlay.querySelector('#ws-ev-goal');
  if (!goalEl.value.trim()) goalEl.value = _evTarget.description || '';
  stateEl.innerHTML = `<div class="ws-ev-target-note">${esc(_evTarget.description || '')}
    ${_evTarget.overridden ? ' · <button class="ws-link" id="ws-ev-revert">revert to default</button>' : ''}</div>`;
  const rev = stateEl.querySelector('#ws-ev-revert');
  if (rev) rev.addEventListener('click', _revertPrompt);
}

async function _startEvolve() {
  const prompt = _overlay.querySelector('#ws-ev-prompt').value.trim();
  const goal = _overlay.querySelector('#ws-ev-goal').value.trim();
  const note = _overlay.querySelector('#ws-ev-note');
  const out = _overlay.querySelector('#ws-ev-result');
  if (!prompt || !goal) { note.textContent = 'Enter a prompt and a goal.'; return; }
  note.textContent = '';
  out.innerHTML = '<div class="ws-advice-loading">Evolving — building an eval set, mutating, and judging on held-out cases. A few minutes on the model…</div>';
  const { ok, status, body } = await _json(`${API}/evolve/start`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, goal }),
  });
  if (status === 403) { out.innerHTML = '<div class="ws-callout">Self-edit is off — turn it on in the header.</div>'; return; }
  if (!ok || !body.run_id) { out.innerHTML = `<div class="ws-muted ws-err">${esc(body.error || 'Could not start evolution.')}</div>`; return; }
  if (_evPollTimer) clearTimeout(_evPollTimer);
  _pollEvolve(body.run_id, 0);
}

async function _pollEvolve(runId, tries) {
  if (!_opened || _lane !== 'evolve') return;
  const out = _overlay.querySelector('#ws-ev-result');
  if (!out) return;
  const { ok, body } = await _json(`${API}/evolve/${encodeURIComponent(runId)}`);
  if (!ok) { out.innerHTML = '<div class="ws-muted ws-err">Lost track of the evolve run.</div>'; return; }
  if (body.status === 'running') {
    if (tries > 240) { out.innerHTML = '<div class="ws-muted">Still running — give it another moment and re-open Evolve.</div>'; return; }
    const dots = '.'.repeat((tries % 3) + 1);
    out.innerHTML = `<div class="ws-advice-loading">Evolving${dots} building an eval set, mutating, judging on held-out cases (a few minutes).</div>`;
    _evPollTimer = setTimeout(() => _pollEvolve(runId, tries + 1), 3000);
    return;
  }
  if (body.status === 'failed') {
    out.innerHTML = `<div class="ws-muted ws-err">Evolution couldn't finish: ${esc(body.error || 'unknown error')}</div>`;
    return;
  }
  _renderEvolveResult(out, body.result || {});
}

function _renderEvolveResult(out, r) {
  const acc = !!r.accepted;
  const pct = (x) => (typeof x === 'number' ? `${(x * 100).toFixed(0)}%` : '—');
  const delta = typeof r.improvement === 'number' ? `${r.improvement >= 0 ? '+' : ''}${(r.improvement * 100).toFixed(0)}%` : '';
  const evolved = r.evolved_prompt || '';
  const canApply = acc && _evTarget;
  out.innerHTML = `
    <div class="ws-ev-verdict ${acc ? 'ok' : ''}">
      <span class="ws-badge ${acc ? 'ws-st-promoted' : 'ws-st-gated'}">${acc ? 'improved' : 'kept original'}</span>
      <span>held-out: baseline ${pct(r.baseline_holdout)} → evolved ${pct(r.best_holdout)} <strong>${esc(delta)}</strong>
        · ${esc(r.iterations || 0)} round${r.iterations === 1 ? '' : 's'}</span>
    </div>
    ${acc
      ? `<div class="ws-d-block"><span class="ws-d-k">Evolved prompt (beat the original on held-out cases)</span>
           <pre class="ws-ev-prompt">${esc(evolved)}</pre></div>
         ${canApply
           ? `<div class="ws-evolve-actions"><button class="ws-btn ws-keep" id="ws-ev-apply">Apply to ${esc(_evTarget.label)} →</button>
                <span class="ws-toolbar-note">makes the app use this prompt — verified by read-back, reversible</span></div>`
           : '<div class="ws-muted">Free-form run — pick a Target above to make a winner live.</div>'}`
      : '<div class="ws-muted">No variant beat the original on the held-out split — the honest call is to keep yours. Try a sharper goal or run it again.</div>'}`;
  const ap = out.querySelector('#ws-ev-apply');
  if (ap) ap.addEventListener('click', () => _applyEvolved(evolved));
}

async function _applyEvolved(text) {
  if (!_evTarget) return;
  const { ok, status, body } = await _json(`${API}/reshape`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ surface: 'config', key: _evTarget.key, value: text, ask: `evolve ${_evTarget.slug}` }),
  });
  if (status === 403) { _toast('Self-edit is off.', true); return; }
  if (!ok || body.status === 'failed') { _toast((body && body.detail) || 'Apply failed.', true); return; }
  _toast(`Applied — the app now uses the evolved "${_evTarget.label}" prompt.`);
  // reflect new state locally
  _evTarget.effective = text; _evTarget.overridden = true;
}

async function _revertPrompt() {
  if (!_evTarget) return;
  const { ok, status } = await _json(`${API}/reshape`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ surface: 'config', key: _evTarget.key, value: '', ask: `revert ${_evTarget.slug}` }),
  });
  if (status === 403) { _toast('Self-edit is off.', true); return; }
  if (!ok) { _toast('Revert failed.', true); return; }
  _toast(`Reverted "${_evTarget.label}" to the default.`);
  _evTarget.effective = _evTarget.default; _evTarget.overridden = false;
  _overlay.querySelector('#ws-ev-prompt').value = _evTarget.default || '';
  const st = _overlay.querySelector('#ws-ev-target-state');
  if (st) st.innerHTML = `<div class="ws-ev-target-note">${esc(_evTarget.description || '')}</div>`;
}

// ── toast (header note) ──────────────────────────────────────────────────

let _toastTimer = null;
function _toast(text, bad = false) {
  let el = _overlay.querySelector('#ws-toast');
  if (!el) {
    el = document.createElement('div');
    el.id = 'ws-toast';
    el.className = 'ws-toast';
    _overlay.appendChild(el);
  }
  el.textContent = text || '';
  el.classList.toggle('bad', bad);
  el.classList.add('show');
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
}

// ── public surface ───────────────────────────────────────────────────────

export async function openWorkshop() {
  _build();
  _overlay.classList.remove('hidden');
  _overlay.focus();
  _opened = true;
  // A theater needs its live state; if we lost it (overlay was closed), fall back
  // to Debt — the resume banner there reattaches to any still-running server run.
  if (_lane === 'theater' && !_run) _lane = 'debt';
  _setLane(_lane || 'overview');
  await Promise.all([_loadSettings(), _loadHealth(), _loadList().then(() => { if (_lane === 'overview') _renderLane(); })]);
}

export function closeWorkshop() {
  if (!_overlay) return;
  // Detach the live stream only — the server-owned run keeps going; the Debt
  // lane's resume banner reattaches next time.
  if (_runAbort) { _runAbort.abort(); _runAbort = null; }
  _overlay.classList.add('hidden');
  _opened = false;
}

export function isOpen() { return _opened; }
