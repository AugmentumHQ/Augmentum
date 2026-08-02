/**
 * coder-inspector.js — Right-side inspector for coder mode.
 *
 * Surfaces the hidden-state the agent reads each turn (objective,
 * mission, workspace facts, observations, prior turns, cost) in a
 * cooperative editable form: edits to objective + observations
 * mirror to /workspace/.augmentum/ files so the next turn's
 * kernel render reads them automatically.
 *
 * See docs/superpowers/specs/2026-05-28-coder-inspector-design.md.
 */

import { escapeHtml, extractErrorMessage, showToast } from './app.js';
import { getActiveWorkspaceId } from './coder.js';

const POLL_INTERVAL_MS = 1500;
const OBJECTIVE_MAX = 2048;
const OBSERVATION_FACT_MAX = 1024;

const CATEGORIES = [
  'build', 'test', 'deploy', 'api', 'data',
  'env', 'constraint', 'gotcha', 'style', 'other',
];

const PROMISE_ICON = {
  fulfilled:    '✓',
  in_progress:  '◐',
  pending:      '○',
  rejected:     '✗',
};

const TASK_ICON = {
  completed:    '✓',
  in_progress:  '◐',
  pending:      '○',
};

// Termination Quality Gate verdicts worth surfacing. We don't show
// every reason — "substantive_active" / "substantive_passive" are just
// "ok, the model answered", noise on the row. The ones below change
// how a reader interprets the stop: "loop capped a chatty model",
// "loop nudged for partial work", etc.
const _SHOWN_VERDICT_REASONS = new Set([
  'already_nudged',
  'nudge_bailout_under_action',
  'nudge_insistent_zero_writes',
  'nudge_empty_prose',
  'max_iterations_reached',
]);
const _VERDICT_LABELS = {
  'already_nudged': 'nudge cap',
  'nudge_bailout_under_action': 'bailout',
  'nudge_insistent_zero_writes': 'demanded',
  'nudge_empty_prose': 'empty',
};
const _VERDICT_TOOLTIPS = {
  'already_nudged': 'Loop hit the nudge cap — model kept producing prose without tool calls. Raise coder_native_nudge_max in Loop tuning if this fires often.',
  'nudge_bailout_under_action': 'Model emitted a short one-sentence stop under an action request — nudged once to continue.',
  'nudge_insistent_zero_writes': 'User explicitly demanded completion ("don\'t stop", "until finished") but the model produced zero writes — nudged.',
  'nudge_empty_prose': 'Model stopped without saying anything — nudged.',
};

let _instance = null;

export function createCoderInspector() {
  if (_instance) return _instance;
  _instance = new CoderInspector();
  return _instance;
}

export function getCoderInspector() { return _instance; }

class CoderInspector {
  constructor() {
    this._mount = document.getElementById('coder-view');
    this._workspaceId = '';
    this._pollTimer = null;
    this._pollInFlight = false;
    this._activeFilter = 'all';
    this._objectiveDraft = null;
    this._objectiveMtime = 0;
    this._observations = [];
    this._observationsTotal = 0;
    this._editingObsIdx = null;
    this._newObsCategory = null;  // 'constraint' | 'gotcha' | null
    this._lastInspectorState = null;
    this._currentStrategy = '';
    this._costAllLocal = true;
    this._destroyed = false;
    this._tuningManifest = null;
    this._tuningSaveTimer = null;
    this._tuningOpen = false;
    this._bindHandlers();
  }

  // ── Lifecycle ────────────────────────────────────────────────

  async open(workspaceId) {
    if (this._destroyed) return;
    const wsid = workspaceId || getActiveWorkspaceId();
    if (!wsid) {
      this._stopPolling();
      this._workspaceId = '';
      this._resetRenderedState();
      this._renderEmptyState();
      return;
    }
    if (wsid === this._workspaceId && this._pollTimer) {
      // Already open for this workspace; just kick a refresh.
      this._poll();
      return;
    }
    // Workspace changed (or first open) — clear all per-workspace
    // caches so the panel never shows the previous container's
    // objective / observations / mission / cost while the new
    // workspace's data is fetching.
    if (this._workspaceId && this._workspaceId !== wsid) {
      this._resetRenderedState();
    }
    this._workspaceId = wsid;
    await this._loadAll();
    this._startPolling();
  }

  _resetRenderedState() {
    this._objectiveDraft = null;
    this._objectiveMtime = 0;
    this._observations = [];
    this._observationsTotal = 0;
    this._editingObsIdx = null;
    this._newObsCategory = null;
    this._lastInspectorState = null;
    this._activeFilter = 'all';
    // Section signatures must reset with the DOM: the bodies below get
    // re-seeded with "Loading…", so an unchanged-data skip on the next
    // poll would leave the placeholder on screen forever.
    this._sectionSigs = {};
    this._idleStreak = 0;
    // Wipe DOM bodies so cross-workspace content can't visually bleed
    // through while the new fetches are in flight.
    const display = document.getElementById('coder-objective-display');
    if (display) display.innerHTML = `<span class="coder-inspector-muted">Loading…</span>`;
    const obsList = document.getElementById('coder-observations-list');
    if (obsList) obsList.innerHTML = `<li class="coder-inspector-empty">Loading…</li>`;
    const constraintsList = document.getElementById('coder-constraints-list');
    if (constraintsList) constraintsList.innerHTML = `<li class="coder-inspector-empty">Loading…</li>`;
    const gotchasList = document.getElementById('coder-gotchas-list');
    if (gotchasList) gotchasList.innerHTML = `<li class="coder-inspector-empty">Loading…</li>`;
    const turnsList = document.getElementById('coder-turns-list');
    if (turnsList) turnsList.innerHTML = `<li class="coder-inspector-empty">Loading…</li>`;
    const missionList = document.getElementById('coder-mission-list');
    if (missionList) { missionList.innerHTML = ''; missionList.classList.add('hidden'); }
    const missionEmpty = document.getElementById('coder-mission-empty');
    if (missionEmpty) { missionEmpty.textContent = 'Loading…'; missionEmpty.classList.remove('hidden'); }
    const identityCard = document.getElementById('coder-identity-card');
    if (identityCard) identityCard.innerHTML = `<span class="coder-inspector-muted">Loading…</span>`;
    const costTotal = document.getElementById('coder-cost-total');
    if (costTotal) costTotal.textContent = '$0.0000';
    const costEmpty = document.getElementById('coder-cost-empty');
    const costTable = document.getElementById('coder-cost-table');
    if (costEmpty) { costEmpty.textContent = 'Loading…'; costEmpty.classList.remove('hidden'); }
    if (costTable) costTable.classList.add('hidden');
    const oracleRate = document.getElementById('coder-oracle-rate');
    if (oracleRate) oracleRate.textContent = '—';
    const oracleEmpty = document.getElementById('coder-oracle-empty');
    if (oracleEmpty) { oracleEmpty.textContent = 'Loading…'; oracleEmpty.classList.remove('hidden'); }
    const oracleDetail = document.getElementById('coder-oracle-detail');
    if (oracleDetail) oracleDetail.classList.add('hidden');
    // Reset header chrome so stale status doesn't linger.
    const nowLine = document.getElementById('coder-inspector-now-line');
    if (nowLine) nowLine.textContent = 'Loading…';
    const counters = document.getElementById('coder-inspector-counters');
    if (counters) counters.textContent = '';
    const cancelBtn = document.getElementById('coder-inspector-cancel-btn');
    if (cancelBtn) cancelBtn.classList.add('hidden');
  }

  close() {
    this._stopPolling();
    this._cancelObjectiveEdit();
    this._editingObsIdx = null;
    this._newObsCategory = null;
  }

  destroy() {
    this._destroyed = true;
    this.close();
    _instance = null;
  }

  refresh() { return this._poll(); }

  // ── Polling ──────────────────────────────────────────────────

  _startPolling() {
    this._stopPolling();
    this._pollTimer = setInterval(() => {
      if (document.hidden) return;
      // Idle backoff: when several consecutive polls saw no running turn,
      // drop to every 3rd tick (effective ~4.5s). The SSE nudge below
      // calls _poll directly, so a mid-turn state mutation still lands
      // within ~100ms regardless of the backoff — this only slows the
      // "nothing is happening" heartbeat.
      this._tickCount = (this._tickCount || 0) + 1;
      if ((this._idleStreak || 0) >= 4 && this._tickCount % 3 !== 0) return;
      this._poll();
    }, POLL_INTERVAL_MS);
    this._attachSystemEventListener();
  }

  _stopPolling() {
    if (this._pollTimer) {
      clearInterval(this._pollTimer);
      this._pollTimer = null;
    }
    this._detachSystemEventListener();
  }

  // Subscribe to the SSE bus so mid-turn task_list / mission mutations
  // refresh the panel within ~100ms instead of waiting for the next
  // POLL_INTERVAL_MS tick. Filter on workspace_id so cross-workspace
  // events don't cause noisy re-fetches.
  _attachSystemEventListener() {
    if (this._sysEventHandler) return;
    this._sysEventHandler = (ev) => {
      const data = ev?.detail?.data || {};
      if (data.workspace_id && data.workspace_id !== this._workspaceId) return;
      if (document.hidden) return;
      this._poll();
    };
    window.addEventListener('system-event:coder.state_updated', this._sysEventHandler);
  }

  _detachSystemEventListener() {
    if (this._sysEventHandler) {
      window.removeEventListener('system-event:coder.state_updated', this._sysEventHandler);
      this._sysEventHandler = null;
    }
  }

  async _poll() {
    if (this._pollInFlight || !this._workspaceId) return;
    // The panel may not be on screen at all — createCoderInspector().open()
    // fires on ENTERING coder mode, not on opening the side panel, so this
    // loop used to fetch + rebuild five sections' innerHTML every 1.5s for
    // an invisible view. offsetParent is null under any hidden ancestor
    // (the #coder-view panel-view, a collapsed side panel, display:none).
    const panel = document.getElementById('coder-view');
    if (panel && !panel.offsetParent) return;
    this._pollInFlight = true;
    try {
      const data = await this._fetchInspectorState();
      if (data) {
        this._lastInspectorState = data;
        const running = (data.run_status || {}).state === 'running';
        this._idleStreak = running ? 0 : (this._idleStreak || 0) + 1;
        const state = data.state || {};
        // Header always renders — it's a handful of textContent writes and
        // carries the live elapsed/idle counters. The five sections below
        // are innerHTML REBUILDS, so each is gated on a signature of the
        // exact data it consumes: unchanged data = untouched DOM (no
        // churn, no style recalc, no GC pressure at idle).
        this._renderHeader(data);
        this._renderIfChanged('tasks', [state.tasks],
          () => this._renderTasks(state));
        this._renderIfChanged('mission', [state.mission, this._currentStrategy],
          () => this._renderMission(state));
        this._renderIfChanged('identity', [data.identity],
          () => this._renderIdentity(data.identity || {}));
        this._renderIfChanged('turns',
          [state.turn_summaries, (data.cost || {}).turn_count, (data.run_status || {}).state],
          () => this._renderTurns(state));
        this._renderIfChanged('cost', [data.cost],
          () => this._renderCost(data.cost || {}));
        this._renderIfChanged('oracle', [data.oracle],
          () => this._renderOracle(data.oracle || {}));
      }
    } catch (err) {
      // Network/server hiccups — keep last-known state, log once
      console.debug('coder-inspector poll failed', err);
    } finally {
      this._pollInFlight = false;
    }
  }

  /** Run ``render`` only when ``input``'s JSON signature changed since the
   *  last call for ``key``. Signatures cover exactly what the renderer
   *  reads — see the call sites in _poll. */
  _renderIfChanged(key, input, render) {
    let sig;
    try { sig = JSON.stringify(input); } catch { sig = undefined; }
    if (!this._sectionSigs) this._sectionSigs = {};
    // Unstringifiable input (circular?) — never skip, correctness first.
    if (sig !== undefined && this._sectionSigs[key] === sig) return;
    this._sectionSigs[key] = sig;
    render();
  }

  // ── Data fetching ────────────────────────────────────────────

  async _loadAll() {
    if (!this._workspaceId) return;
    await Promise.all([
      this._poll(),
      this._loadObjective(),
      this._loadObservations(),
      this._loadTuning(),
    ]);
  }

  async _fetchInspectorState() {
    try {
      const r = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(this._workspaceId)}/inspector-state`,
        { credentials: 'include' },
      );
      if (!r.ok) return null;
      return await r.json();
    } catch (err) {
      // Network failure (server restarting, mobile losing connection)
      // — poll caller logs the broader debug message; returning null
      // keeps last-known state on screen.
      return null;
    }
  }

  async _loadObjective() {
    try {
      const r = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(this._workspaceId)}/objective`,
        { credentials: 'include' },
      );
      if (!r.ok) {
        this._renderObjective({ content: '', mtime: 0, seeded: false });
        return;
      }
      const data = await r.json();
      this._objectiveMtime = data.mtime || 0;
      if (this._objectiveDraft === null) {
        this._renderObjective(data);
      }
    } catch (err) {
      console.debug('coder-inspector objective load failed', err);
    }
  }

  async _loadObservations() {
    try {
      const r = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(this._workspaceId)}/observations?limit=200`,
        { credentials: 'include' },
      );
      if (!r.ok) {
        this._observations = [];
        this._observationsTotal = 0;
      } else {
        const data = await r.json();
        this._observations = Array.isArray(data.items) ? data.items : [];
        this._observationsTotal = data.total || 0;
      }
      this._renderObservations();
      this._renderConstraintsAndGotchas();
    } catch (err) {
      console.debug('coder-inspector observations load failed', err);
    }
  }

  // ── Header / status ──────────────────────────────────────────

  _renderHeader(data) {
    const dot = document.getElementById('coder-inspector-status-dot');
    const strategy = document.getElementById('coder-inspector-strategy');
    const model = document.getElementById('coder-inspector-model');
    const counters = document.getElementById('coder-inspector-counters');
    const nowLine = document.getElementById('coder-inspector-now-line');
    const cancelBtn = document.getElementById('coder-inspector-cancel-btn');

    const runStatus = data.run_status || {};
    const isRunning = runStatus.state === 'running';
    const state = data.state || {};

    if (dot) {
      if (isRunning) dot.dataset.state = 'running';
      else if (runStatus.last_status === 'cancelled') dot.dataset.state = 'cancelled';
      else if (runStatus.last_status === 'errored' || runStatus.last_status === 'error') dot.dataset.state = 'error';
      else if (runStatus.last_status === 'completed') dot.dataset.state = 'completed';
      else dot.dataset.state = 'idle';
    }

    const lastModel = runStatus.last_model || '';
    const liveModel = isRunning ? (lastModel || '') : (runStatus.last_model || '');
    const lastStrategy = runStatus.last_strategy || '';
    const liveStrategy = isRunning ? (lastStrategy || '') : (runStatus.last_strategy || '');
    // Track the most recent strategy so other panels can read it
    // without a separate poll — Mission visibility, for instance,
    // depends on whether this is a legacy-strategy session.
    this._currentStrategy = (liveStrategy || '').toLowerCase();
    if (strategy) {
      strategy.textContent = (liveStrategy || '—').slice(0, 6).toUpperCase();
    }
    if (model) {
      model.textContent = _shortModel(liveModel) || '—';
      model.title = liveModel || '';
    }

    // Counters: while running we show only "iter N" — the elapsed is
    // already on the now-line below ("Running · 5m 27s"), so showing
    // it in both places duplicates without adding signal. Idle/done
    // turns flip to "idle 2m · iter 17" so the cool-down has context.
    const iters = isRunning ? runStatus.seq : runStatus.last_iterations;
    const counterParts = [];
    if (!isRunning && runStatus.last_idle_s != null) {
      counterParts.push(`idle ${_formatElapsed(runStatus.last_idle_s)}`);
    }
    if (iters) counterParts.push(`iter ${iters}`);
    if (counters) counters.textContent = counterParts.join(' · ');

    // Now line
    if (nowLine) {
      if (isRunning) {
        const stuck = runStatus.cancel_requested ? ' (cancelling)' : '';
        nowLine.textContent = `Running · ${_formatElapsed(runStatus.elapsed_s)}${stuck}`;
      } else if (runStatus.last_status) {
        const finishReason = runStatus.last_finish_reason || runStatus.last_status;
        nowLine.textContent = `Last turn: ${finishReason}`;
      } else {
        nowLine.textContent = 'Idle';
      }
    }

    if (cancelBtn) cancelBtn.classList.toggle('hidden', !isRunning);
  }

  // ── Task list ────────────────────────────────────────────────
  // The Claude-Code/Codex-style task list the model maintains via
  // the ``task_list`` tool. Items: {content, activeForm, status}.
  // Distinct from Mission (which is the structured Promise list the
  // legacy strategy generates). Both can be live in the same turn.

  _renderTasks(state) {
    const list = document.getElementById('coder-tasks-list');
    const empty = document.getElementById('coder-tasks-empty');
    const progress = document.getElementById('coder-tasks-progress');
    const tasks = Array.isArray(state.tasks) ? state.tasks : [];

    if (!list || !empty || !progress) return;

    if (tasks.length === 0) {
      list.classList.add('hidden');
      empty.classList.remove('hidden');
      progress.textContent = '0 of 0';
      return;
    }

    const completed = tasks.filter((t) => t.status === 'completed').length;
    const inProgress = tasks.filter((t) => t.status === 'in_progress').length;
    // Show "completed of total" with an in-progress hint when present
    // so the user sees "3 of 5 (1 active)" without clicking through.
    progress.textContent = inProgress
      ? `${completed} of ${tasks.length} · ${inProgress} active`
      : `${completed} of ${tasks.length}`;

    list.innerHTML = tasks.map((t) => {
      const status = t.status || 'pending';
      const icon = TASK_ICON[status] || TASK_ICON.pending;
      const content = t.content || '';
      const activeForm = t.activeForm || '';
      // Display ``activeForm`` while in-progress ("Running tests…") and
      // ``content`` otherwise ("Run the tests"). Tooltip carries the
      // other form so the user can see both shapes at a glance.
      const display = (status === 'in_progress' && activeForm) ? activeForm : content;
      const tooltip = (status === 'in_progress' && activeForm && content)
        ? content
        : (activeForm || content);
      return `
        <li class="coder-inspector-task-item" data-status="${escapeHtml(status)}" title="${escapeHtml(tooltip)}">
          <span class="coder-inspector-task-icon">${escapeHtml(icon)}</span>
          <span class="coder-inspector-task-text">${escapeHtml(display)}</span>
        </li>`;
    }).join('');
    list.classList.remove('hidden');
    empty.classList.add('hidden');
  }

  // ── Mission ──────────────────────────────────────────────────

  _renderMission(state) {
    const section = document.querySelector('.coder-inspector-section[data-section="mission"]');
    const list = document.getElementById('coder-mission-list');
    const empty = document.getElementById('coder-mission-empty');
    const progress = document.getElementById('coder-mission-progress');
    const mission = Array.isArray(state.mission) ? state.mission : [];

    if (!list || !empty || !progress) return;

    // Mission is only populated by the legacy strategy's promise
    // runner. Native / hybrid / canonical never write here, so for
    // those sessions the whole section is noise — hide it entirely
    // rather than showing a misleading "agent will populate" message.
    // Show when: strategy unknown (first poll), legacy, or list non-empty.
    const strategy = this._currentStrategy;
    const isLegacy = strategy === 'legacy';
    const shouldHide = strategy && !isLegacy && mission.length === 0;
    if (section) section.classList.toggle('hidden', shouldHide);
    if (shouldHide) return;

    if (mission.length === 0) {
      list.classList.add('hidden');
      empty.classList.remove('hidden');
      // _resetRenderedState seeds this with "Loading…" — replace with
      // the real empty message once polling has produced a state doc.
      // For the strategy-unknown case we keep neutral language; for
      // legacy we point at how the agent populates it.
      empty.textContent = isLegacy
        ? 'No mission yet — the agent generates promises during the plan phase.'
        : 'No mission for this session. Mission only appears when running the `legacy` strategy.';
      progress.textContent = '0 of 0';
      return;
    }

    const fulfilled = mission.filter(p => p.status === 'fulfilled').length;
    progress.textContent = `${fulfilled} of ${mission.length} verified`;

    list.innerHTML = mission.map((p) => {
      const status = p.status || 'pending';
      const icon = PROMISE_ICON[status] || PROMISE_ICON.pending;
      const desc = p.description || '';
      const evidence = p.evidence ? `\n\nEvidence: ${p.evidence}` : '';
      const attempts = p.attempts ? `\n\nAttempts: ${p.attempts}` : '';
      const tooltip = `${desc}${evidence}${attempts}`;
      return `
        <li class="coder-inspector-mission-item" data-status="${escapeHtml(status)}" title="${escapeHtml(tooltip)}">
          <span class="coder-inspector-mission-icon">${escapeHtml(icon)}</span>
          <span class="coder-inspector-mission-text">${escapeHtml(desc)}</span>
        </li>`;
    }).join('');
    list.classList.remove('hidden');
    empty.classList.add('hidden');
  }

  // ── Identity ─────────────────────────────────────────────────

  _renderIdentity(identity) {
    const card = document.getElementById('coder-identity-card');
    if (!card) return;
    const detected = identity.detected || {};
    const languages = Array.isArray(detected.languages) ? detected.languages : [];
    const meta = identity.meta || {};
    const lastDetected = meta.last_detected_at || 0;

    const rows = [];
    if (languages.length > 0) {
      rows.push(['Languages', languages.map(_titleCase).join(', ')]);
    }
    // Surface a handful of common per-language facts (package manager,
    // test runner, version) without enumerating every field — the
    // detected[] map is intentionally open-ended.
    for (const lang of languages) {
      const facts = detected[lang];
      if (!facts || typeof facts !== 'object') continue;
      const pieces = [];
      if (facts.version) pieces.push(facts.version);
      if (facts.package_manager) pieces.push(facts.package_manager);
      if (facts.test_runner) pieces.push(facts.test_runner);
      if (facts.runtime) pieces.push(facts.runtime);
      if (pieces.length > 0) {
        rows.push([_titleCase(lang), pieces.join(' · ')]);
      }
    }

    if (rows.length === 0) {
      if (lastDetected > 0) {
        card.innerHTML = `<span class="coder-inspector-muted">No language detected yet.</span>`;
      } else {
        card.innerHTML = `<span class="coder-inspector-muted">Detecting…</span>`;
      }
      return;
    }
    card.innerHTML = rows.map(([k, v]) =>
      `<div class="coder-inspector-identity-row">
        <span class="coder-inspector-identity-key">${escapeHtml(k)}</span>
        <span class="coder-inspector-identity-val">${escapeHtml(v)}</span>
      </div>`,
    ).join('');
  }

  // ── Objective ────────────────────────────────────────────────

  _renderObjective(data) {
    const display = document.getElementById('coder-objective-display');
    if (!display) return;
    const content = (data && data.content) || '';
    if (!content) {
      // Surface BOTH the auto-seed contract and an explicit CTA. The
      // 30-char gate misses casual asks ("fix the bug"), and a vibe
      // coder reading "Loading…" or "no objective" doesn't know they
      // can write one themselves.
      display.innerHTML = `
        <div class="coder-inspector-empty-cta">
          <div class="coder-inspector-muted">No objective pinned yet.</div>
          <div class="coder-inspector-empty-sub">The first substantive ask (≥30 chars) auto-seeds it, or pin one now:</div>
          <button type="button" class="coder-inspector-empty-btn" id="coder-objective-empty-add-btn">+ Pin an objective</button>
        </div>
      `;
      // Wire the CTA every render — the previous button is gone.
      display.querySelector('#coder-objective-empty-add-btn')?.addEventListener('click', () => {
        this._enterObjectiveEdit();
      });
    } else {
      display.textContent = _stripObjectiveHeader(content);
    }
  }

  _enterObjectiveEdit() {
    const editor = document.getElementById('coder-objective-editor');
    const display = document.getElementById('coder-objective-display');
    const textarea = document.getElementById('coder-objective-textarea');
    const charCount = document.getElementById('coder-objective-char-count');
    if (!editor || !display || !textarea) return;

    // Seed the textarea with the current rendered content
    const current = (display.textContent || '').trim();
    textarea.value = current;
    this._objectiveDraft = current;
    if (charCount) charCount.textContent = `${current.length} / ${OBJECTIVE_MAX}`;
    editor.classList.remove('hidden');
    display.classList.add('hidden');
    textarea.focus();
  }

  _cancelObjectiveEdit() {
    const editor = document.getElementById('coder-objective-editor');
    const display = document.getElementById('coder-objective-display');
    if (editor) editor.classList.add('hidden');
    if (display) display.classList.remove('hidden');
    this._objectiveDraft = null;
  }

  async _saveObjective() {
    const textarea = document.getElementById('coder-objective-textarea');
    if (!textarea) return;
    const content = (textarea.value || '').trim();
    if (!content) {
      showToast('Objective cannot be empty', 'error');
      return;
    }
    try {
      const r = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(this._workspaceId)}/objective`,
        {
          method: 'PUT',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content,
            if_mtime_unchanged: this._objectiveMtime || null,
          }),
        },
      );
      if (r.status === 409) {
        const data = await r.json();
        this._objectiveMtime = data.current_mtime || 0;
        showToast('Objective was modified by the agent — refreshing', 'warn');
        await this._loadObjective();
        return;
      }
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        showToast(extractErrorMessage(data, 'Failed to save objective'), 'error');
        return;
      }
      const data = await r.json();
      this._objectiveMtime = data.mtime || 0;
      this._objectiveDraft = null;
      this._cancelObjectiveEdit();
      this._renderObjective(data);
      showToast('Objective saved', 'success');
    } catch (err) {
      console.warn('save objective failed', err);
      showToast('Network error saving objective', 'error');
    }
  }

  // ── Observations / constraints / gotchas ─────────────────────

  _renderConstraintsAndGotchas() {
    this._renderObsCategoryList('constraint', 'coder-constraints-list');
    this._renderObsCategoryList('gotcha', 'coder-gotchas-list');
  }

  _renderObsCategoryList(category, listId) {
    const list = document.getElementById(listId);
    if (!list) return;
    const items = this._observations.filter(o => o.category === category);
    if (items.length === 0 && this._newObsCategory !== category) {
      // Explain the purpose, since first-time users won't know what
      // a "constraint" or "gotcha" is for in this context.
      const explainer = category === 'constraint'
        ? 'Hard rules the agent must respect — e.g. "no new dependencies", "Python 3.11 only", "keep public API stable".'
        : 'Surprising behaviors the agent should remember — e.g. "this lib breaks under proxy", "test runner needs --no-cov on Windows".';
      list.innerHTML = `<li class="coder-inspector-empty coder-inspector-empty-explain">${escapeHtml(explainer)}</li>`;
      return;
    }
    // hideCategory: the section header already says "Constraints" /
    // "Gotchas" — stamping the same word on every row is noise. The
    // all-observations panel keeps the category chip for scanning.
    const rows = items.map(o => this._renderObsItemHtml(o, { hideCategory: true })).join('');
    const adder = this._newObsCategory === category ? this._renderInlineAdderHtml(category) : '';
    list.innerHTML = rows + adder;
  }

  _renderObservations() {
    const list = document.getElementById('coder-observations-list');
    const counter = document.getElementById('coder-observations-count');
    if (!list) return;
    if (counter) counter.textContent = String(this._observationsTotal);

    let items = this._observations;
    if (this._activeFilter && this._activeFilter !== 'all') {
      items = items.filter(o => o.category === this._activeFilter);
    }
    if (items.length === 0) {
      // Differentiate "empty because nothing's been recorded yet" from
      // "empty because filter excludes everything". The former is the
      // expected state on most workspaces and shouldn't read as broken.
      const isFiltered = this._activeFilter && this._activeFilter !== 'all'
        && this._observations.length > 0;
      const msg = isFiltered
        ? `No <strong>${escapeHtml(this._activeFilter)}</strong> observations yet — try a different category or <em>all</em>.`
        : 'Durable facts the agent learns about this codebase land here automatically — test runner, API shapes, version locks, gotchas. Future sessions read them back so the agent doesn\'t re-discover the same thing twice.';
      list.innerHTML = '';
      const empty = document.createElement('li');
      empty.className = 'coder-inspector-empty coder-inspector-empty-explain';
      if (isFiltered) {
        empty.innerHTML = msg;
      } else {
        empty.textContent = msg;
      }
      list.appendChild(empty);
      return;
    }
    list.innerHTML = items.map(o => this._renderObsItemHtml(o)).join('');
  }

  _renderObsItemHtml(obs, opts = {}) {
    const source = obs.source || '';
    const userEdited = source.startsWith('user-edit');
    const autoObserved = source.startsWith('auto');
    const hideCategory = !!opts.hideCategory;
    if (this._editingObsIdx === obs.idx) {
      return `<li class="coder-inspector-obs-item" data-idx="${obs.idx}">
        ${this._renderObsEditorHtml(obs)}
      </li>`;
    }
    const categoryChip = hideCategory
      ? ''
      : `<span class="coder-inspector-obs-category" data-category="${escapeHtml(obs.category)}">${escapeHtml(obs.category)}</span>`;
    // Source attribution: distinguish user-edited from agent-recorded
    // via the ``observe`` tool from auto-extracted via the pattern
    // matcher. Auto entries are pattern-recognized; we tint them more
    // neutrally so the user can scan for agent-curated vs auto signal.
    const sourceLabel = userEdited ? 'You' : autoObserved ? 'Auto' : 'Agent';
    const sourceData = userEdited ? '1' : autoObserved ? '2' : '0';
    return `<li class="coder-inspector-obs-item" data-idx="${obs.idx}">
      <div class="coder-inspector-obs-row">
        ${categoryChip}
        <span class="coder-inspector-obs-source" data-user="${sourceData}" title="${escapeHtml(source)}">${sourceLabel}</span>
      </div>
      <div class="coder-inspector-obs-fact">${escapeHtml(obs.fact || '')}</div>
      <div class="coder-inspector-obs-actions">
        <button type="button" class="coder-inspector-obs-action" data-action="edit" data-idx="${obs.idx}">Edit</button>
        <button type="button" class="coder-inspector-obs-action" data-action="delete" data-idx="${obs.idx}">Delete</button>
      </div>
    </li>`;
  }

  _renderObsEditorHtml(obs) {
    const optionsHtml = CATEGORIES.map(c =>
      `<option value="${escapeHtml(c)}"${c === obs.category ? ' selected' : ''}>${escapeHtml(c)}</option>`,
    ).join('');
    return `<div class="coder-inspector-obs-editor">
      <select class="coder-inspector-obs-category-select" data-idx="${obs.idx}">${optionsHtml}</select>
      <textarea class="coder-inspector-textarea" rows="2" maxlength="${OBSERVATION_FACT_MAX}" data-idx="${obs.idx}">${escapeHtml(obs.fact || '')}</textarea>
      <div class="coder-inspector-editor-actions">
        <button type="button" class="btn-secondary tiny" data-action="cancel-edit">Cancel</button>
        <button type="button" class="btn-primary tiny" data-action="save-edit" data-idx="${obs.idx}">Save</button>
      </div>
    </div>`;
  }

  _renderInlineAdderHtml(category) {
    const optionsHtml = CATEGORIES.map(c =>
      `<option value="${escapeHtml(c)}"${c === category ? ' selected' : ''}>${escapeHtml(c)}</option>`,
    ).join('');
    return `<li class="coder-inspector-obs-item" data-new="1">
      <div class="coder-inspector-obs-editor">
        <select class="coder-inspector-obs-new-category">${optionsHtml}</select>
        <textarea class="coder-inspector-textarea coder-inspector-obs-new-fact" rows="2" maxlength="${OBSERVATION_FACT_MAX}" placeholder="Describe the ${escapeHtml(category)}…"></textarea>
        <div class="coder-inspector-editor-actions">
          <button type="button" class="btn-secondary tiny" data-action="cancel-new">Cancel</button>
          <button type="button" class="btn-primary tiny" data-action="save-new">Add</button>
        </div>
      </div>
    </li>`;
  }

  async _saveObservationEdit(idx) {
    const list = this._mount?.querySelector(`[data-idx="${idx}"]`);
    if (!list) return;
    const select = list.querySelector('select');
    const textarea = list.querySelector('textarea');
    if (!select || !textarea) return;
    const category = select.value;
    const fact = (textarea.value || '').trim();
    if (!fact) {
      showToast('Observation cannot be empty', 'error');
      return;
    }
    try {
      const r = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(this._workspaceId)}/observations/${idx}`,
        {
          method: 'PATCH',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ category, fact }),
        },
      );
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        showToast(extractErrorMessage(data, 'Failed to update observation'), 'error');
        return;
      }
      this._editingObsIdx = null;
      await this._loadObservations();
    } catch (err) {
      console.warn('save observation failed', err);
      showToast('Network error', 'error');
    }
  }

  async _deleteObservation(idx) {
    if (!confirm('Delete this observation?')) return;
    try {
      const r = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(this._workspaceId)}/observations/${idx}`,
        { method: 'DELETE', credentials: 'include' },
      );
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        showToast(extractErrorMessage(data, 'Failed to delete observation'), 'error');
        return;
      }
      await this._loadObservations();
    } catch (err) {
      console.warn('delete observation failed', err);
      showToast('Network error', 'error');
    }
  }

  async _saveNewObservation() {
    const node = this._mount?.querySelector('li[data-new="1"]');
    if (!node) return;
    const select = node.querySelector('.coder-inspector-obs-new-category');
    const textarea = node.querySelector('.coder-inspector-obs-new-fact');
    if (!select || !textarea) return;
    const category = select.value;
    const fact = (textarea.value || '').trim();
    if (!fact) {
      showToast('Observation cannot be empty', 'error');
      return;
    }
    try {
      const r = await fetch(
        `/api/coder/workspaces/${encodeURIComponent(this._workspaceId)}/observations`,
        {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ category, fact, confidence: 'user_asserted' }),
        },
      );
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        showToast(extractErrorMessage(data, 'Failed to add observation'), 'error');
        return;
      }
      this._newObsCategory = null;
      await this._loadObservations();
      showToast('Observation added', 'success');
    } catch (err) {
      console.warn('add observation failed', err);
      showToast('Network error', 'error');
    }
  }

  // ── Prior turns ──────────────────────────────────────────────

  _renderTurns(state) {
    const list = document.getElementById('coder-turns-list');
    if (!list) return;
    const summaries = Array.isArray(state.turn_summaries) ? state.turn_summaries : [];
    if (summaries.length === 0) {
      // Differentiate "fresh workspace" from "we have completed turns
      // but no summaries on file". The latter happens after a native /
      // canonical run that predates the summary writer, or when every
      // recent turn ended without any tool exchanges (pure-prose
      // turns leave no trace). Either way, point at the load-bearing
      // fact: a summary is written when a turn ends with tool use.
      const cost = (this._lastInspectorState && this._lastInspectorState.cost) || {};
      const completedTurns = Number(cost.turn_count || 0);
      const runStatus = (this._lastInspectorState && this._lastInspectorState.run_status) || {};
      const running = runStatus.state === 'running';
      let msg;
      if (running) {
        msg = 'Current turn still running — a compact summary is written here when it ends.';
      } else if (completedTurns > 0) {
        msg = `No summaries on file yet (${completedTurns} prior turn${completedTurns === 1 ? '' : 's'} predate this writer or used no tools). Next tool-using turn populates this.`;
      } else {
        msg = 'No completed turns yet. After each turn, a compact summary lands here so the agent remembers what it just did.';
      }
      list.innerHTML = `<li class="coder-inspector-empty coder-inspector-empty-explain">${escapeHtml(msg)}</li>`;
      return;
    }
    const recent = summaries.slice(-10).reverse();
    list.innerHTML = recent.map(t => {
      const cancelled = (t.outcome || '').toLowerCase().includes('cancel');
      const cancelTag = cancelled ? '<span class="coder-inspector-turn-tag">INTERRUPTED</span>' : '';
      const verdictReason = (t.verdict_reason || '').trim();
      // Surface the TQG verdict tag for stops where the granular reason
      // changes how a reader interprets "done". "already_nudged" in
      // particular is a stop the user wants to see — it means the loop
      // capped a chatty model that never called a tool.
      const verdictTag = verdictReason && _SHOWN_VERDICT_REASONS.has(verdictReason)
        ? `<span class="coder-inspector-turn-verdict" data-verdict="${escapeHtml(verdictReason)}" title="${escapeHtml(_VERDICT_TOOLTIPS[verdictReason] || verdictReason)}">${escapeHtml(_VERDICT_LABELS[verdictReason] || verdictReason)}</span>`
        : '';
      const goal = (t.user_goal || '').slice(0, 200);
      const reads = Array.isArray(t.files_read) ? t.files_read.length : 0;
      const edits = Array.isArray(t.files_edited) ? t.files_edited.length : 0;
      const outcome = t.outcome || '—';
      const turnIdx = t.turn_idx != null ? `Turn ${t.turn_idx}` : 'Turn';
      const ago = t.created_at ? _formatAgo(t.created_at) : '';
      return `<li class="coder-inspector-turn" data-cancelled="${cancelled ? '1' : '0'}">
        <div class="coder-inspector-turn-head">
          <span>${escapeHtml(turnIdx)}</span>
          ${ago ? `<span>· ${escapeHtml(ago)}</span>` : ''}
          ${cancelTag}${verdictTag}
        </div>
        <div class="coder-inspector-turn-goal">${escapeHtml(goal)}</div>
        <div class="coder-inspector-turn-stats">Read ${reads} · Edited ${edits} · ${escapeHtml(outcome)}</div>
      </li>`;
    }).join('');
  }

  // ── Cost ─────────────────────────────────────────────────────

  _renderOracle(stats) {
    const rate = document.getElementById('coder-oracle-rate');
    const empty = document.getElementById('coder-oracle-empty');
    const detail = document.getElementById('coder-oracle-detail');
    if (!rate || !empty || !detail) return;
    const writeRuns = Number(stats.write_runs || 0);
    if (!writeRuns) {
      rate.textContent = '—';
      empty.textContent = 'No write-turns yet.';
      empty.classList.remove('hidden');
      detail.classList.add('hidden');
      return;
    }
    const unverified = Number(stats.no_oracle_done || 0);
    const verified = writeRuns - unverified;
    rate.textContent = `${verified}/${writeRuns} verified`;
    empty.classList.add('hidden');
    detail.classList.remove('hidden');
    const kinds = stats.kinds || {};
    const kindsLine = Object.keys(kinds).sort()
      .map((k) => `${escapeHtml(k)} ×${Number(kinds[k]) || 0}`)
      .join(' · ');
    const outcomes = stats.last_outcomes || {};
    const red = Number(outcomes.red || 0);
    const parts = [];
    if (kindsLine) parts.push(kindsLine);
    if (unverified) parts.push(`<span class="coder-inspector-oracle-warn">${unverified} unverified write-turn${unverified === 1 ? '' : 's'}</span>`);
    if (red) parts.push(`${red} ended red`);
    detail.innerHTML = parts.length
      ? parts.join(' · ')
      : '<span class="coder-inspector-muted">All write-turns verified.</span>';
  }

  _renderCost(cost) {
    const total = document.getElementById('coder-cost-total');
    const table = document.getElementById('coder-cost-table');
    const tbody = document.getElementById('coder-cost-tbody');
    const empty = document.getElementById('coder-cost-empty');
    if (!total || !table || !tbody || !empty) return;
    const inputUsd = Number(cost.input_usd || 0);
    const outputUsd = Number(cost.output_usd || 0);
    const totalUsd = inputUsd + outputUsd;
    const rows = Array.isArray(cost.by_model) ? cost.by_model : [];

    // All-local: every row is $0. The 4-column table is noise — pros
    // can confirm at a glance from the header total, vibe coders just
    // see "is this costing me money?" answered. Collapse to a single
    // human-readable summary line.
    const allLocal = rows.length > 0 && rows.every(
      (r) => (Number(r.input_usd || 0) + Number(r.output_usd || 0)) === 0,
    );
    this._costAllLocal = allLocal;

    if (allLocal) {
      const totalTurns = rows.reduce((acc, r) => acc + (r.turns || 0), 0);
      const distinctModels = rows.length;
      total.textContent = 'Local';
      empty.classList.add('hidden');
      table.classList.add('hidden');
      let summary = document.getElementById('coder-cost-local-summary');
      if (!summary) {
        summary = document.createElement('div');
        summary.id = 'coder-cost-local-summary';
        summary.className = 'coder-inspector-card coder-inspector-cost-local';
        table.parentNode.insertBefore(summary, table.nextSibling);
      }
      summary.classList.remove('hidden');
      const modelLabel = distinctModels === 1
        ? escapeHtml(_shortModel(rows[0].model || 'local model'))
        : `${distinctModels} local models`;
      // Header total already says "Local" — repeating the chip below
      // is noise. Lead with the turn count (the actual measurement)
      // and keep the model breakdown as the secondary line.
      summary.innerHTML = `
        <div class="coder-inspector-cost-local-headline">
          <span class="coder-inspector-cost-local-turns">${totalTurns} turn${totalTurns === 1 ? '' : 's'} this session</span>
        </div>
        <div class="coder-inspector-cost-local-detail">${modelLabel} · no API cost</div>
      `;
      return;
    }

    // Mixed or cloud-only: hide any prior local summary, show the
    // full table so pros can audit per-model spend.
    const stale = document.getElementById('coder-cost-local-summary');
    if (stale) stale.classList.add('hidden');

    total.textContent = _formatUsd(totalUsd);
    if (rows.length === 0) {
      empty.classList.remove('hidden');
      table.classList.add('hidden');
      return;
    }
    empty.classList.add('hidden');
    table.classList.remove('hidden');
    tbody.innerHTML = rows.map(r => {
      const isZero = (r.input_usd + r.output_usd) === 0;
      const cls = isZero ? ' class="zero"' : '';
      return `<tr>
        <td title="${escapeHtml(r.model || '')}">${escapeHtml(_shortModel(r.model || ''))}</td>
        <td${cls}>${escapeHtml(_formatUsd(r.input_usd))}</td>
        <td${cls}>${escapeHtml(_formatUsd(r.output_usd))}</td>
        <td>${r.turns || 0}</td>
      </tr>`;
    }).join('');
  }

  // ── Misc ─────────────────────────────────────────────────────

  _renderEmptyState() {
    if (!this._mount) return;
    // Minimal placeholder until a workspace is active.
    const display = document.getElementById('coder-objective-display');
    if (display) {
      display.innerHTML = `<span class="coder-inspector-muted">Select a workspace to inspect.</span>`;
    }
  }

  async _cancelRun() {
    const runId = this._lastInspectorState?.run_status?.run_id;
    if (!runId) return;
    if (!confirm('Cancel the current run?')) return;
    try {
      const r = await fetch(
        `/api/coder/runs/${encodeURIComponent(runId)}/cancel`,
        {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason: 'user_cancel' }),
        },
      );
      if (!r.ok) {
        showToast('Failed to cancel run', 'error');
        return;
      }
      showToast('Cancel requested', 'info');
      this._poll();
    } catch (err) {
      console.warn('cancel run failed', err);
      showToast('Network error', 'error');
    }
  }

  // ── Event wiring (delegated for editable lists) ──────────────

  _bindHandlers() {
    if (!this._mount) return;

    // Workspace switch — coder.js fires this from _switchWorkspace and
    // from create-workspace. Re-bind so the panel doesn't keep showing
    // the previous container's objective / observations / cost.
    document.addEventListener('coder-workspace-changed', (ev) => {
      const newId = ev?.detail?.workspaceId || '';
      // Always re-open: open() handles the empty + same-id + different-id
      // branches itself (clear + refetch when the id flipped).
      this.open(newId);
    });

    // Objective edit
    document.getElementById('coder-objective-edit-btn')?.addEventListener('click', () => {
      this._enterObjectiveEdit();
    });
    document.getElementById('coder-objective-cancel-btn')?.addEventListener('click', () => {
      this._cancelObjectiveEdit();
    });
    document.getElementById('coder-objective-save-btn')?.addEventListener('click', () => {
      this._saveObjective();
    });
    document.getElementById('coder-objective-textarea')?.addEventListener('input', (ev) => {
      const charCount = document.getElementById('coder-objective-char-count');
      if (charCount) charCount.textContent = `${(ev.target.value || '').length} / ${OBJECTIVE_MAX}`;
    });

    // Cancel button + close button
    document.getElementById('coder-inspector-cancel-btn')?.addEventListener('click', () => {
      this._cancelRun();
    });

    // Strategy chip is now a READ-ONLY status indicator — it shows what the
    // last run used (always "native" unless an operator sets the
    // AUGMENTUM_CODER_STRATEGY backend env var for rollback). The in-UI
    // picker was removed when native became the sole shipped strategy; the
    // frozen hybrid/canonical/legacy loops are env-only (see
    // augmentum/modes/coder/README.md).

    // Filter chips
    document.getElementById('coder-observations-filters')?.addEventListener('click', (ev) => {
      const chip = ev.target.closest('.coder-inspector-chip');
      if (!chip) return;
      const category = chip.dataset.category || 'all';
      this._activeFilter = category;
      this._mount.querySelectorAll('#coder-observations-filters .coder-inspector-chip').forEach((c) => {
        c.classList.toggle('active', c.dataset.category === category);
      });
      this._renderObservations();
    });

    // "+ Add" constraint/gotcha
    this._mount.addEventListener('click', (ev) => {
      const addBtn = ev.target.closest('.coder-inspector-add-btn');
      if (addBtn) {
        ev.preventDefault();
        const cat = addBtn.dataset.category || 'constraint';
        this._newObsCategory = cat;
        this._renderConstraintsAndGotchas();
        return;
      }
      // Cancel new
      if (ev.target.dataset?.action === 'cancel-new') {
        this._newObsCategory = null;
        this._renderConstraintsAndGotchas();
        return;
      }
      // Save new
      if (ev.target.dataset?.action === 'save-new') {
        this._saveNewObservation();
        return;
      }
      // Edit observation
      const editBtn = ev.target.closest('[data-action="edit"]');
      if (editBtn) {
        this._editingObsIdx = Number(editBtn.dataset.idx);
        this._renderConstraintsAndGotchas();
        this._renderObservations();
        return;
      }
      // Delete observation
      const delBtn = ev.target.closest('[data-action="delete"]');
      if (delBtn) {
        this._deleteObservation(Number(delBtn.dataset.idx));
        return;
      }
      // Cancel edit
      if (ev.target.dataset?.action === 'cancel-edit') {
        this._editingObsIdx = null;
        this._renderConstraintsAndGotchas();
        this._renderObservations();
        return;
      }
      // Save edit
      const saveBtn = ev.target.closest('[data-action="save-edit"]');
      if (saveBtn) {
        this._saveObservationEdit(Number(saveBtn.dataset.idx));
        return;
      }
    });

    // ── Loop tuning ────────────────────────────────────────────────
    // Tunables are install-wide, not workspace-scoped — admin-only.
    // Toggle button reveals the form; debounced PUTs persist changes.

    document.getElementById('coder-tuning-toggle-btn')?.addEventListener('click', () => {
      this._tuningOpen = !this._tuningOpen;
      const card = document.getElementById('coder-tuning-card');
      if (card) card.classList.toggle('hidden', !this._tuningOpen);
      if (this._tuningOpen && !this._tuningManifest) {
        this._loadTuning();
      }
    });

    document.getElementById('coder-tuning-reset-all-btn')?.addEventListener('click', () => {
      this._resetAllTuning();
    });

    // Per-row input change → debounced save. Listening on the rows
    // container (delegated) so re-renders don't lose handlers.
    document.getElementById('coder-tuning-rows')?.addEventListener('change', (ev) => {
      const input = ev.target.closest('.coder-tuning-input');
      if (!input) return;
      this._scheduleTuningSave(input.dataset.key, input.value);
    });

    // Per-row reset button → set override to 0.
    document.getElementById('coder-tuning-rows')?.addEventListener('click', (ev) => {
      const btn = ev.target.closest('.coder-tuning-reset');
      if (!btn) return;
      const key = btn.dataset.key;
      if (!key) return;
      this._saveTuningValue(key, 0);
    });

    // Request pacing — toggle + seconds. Toggling immediately enables/disables
    // the seconds field and saves; editing the seconds saves (debounced).
    document.getElementById('coder-pacing-enabled')?.addEventListener('change', () => {
      const secs = document.getElementById('coder-pacing-seconds');
      const cb = document.getElementById('coder-pacing-enabled');
      if (secs && cb) secs.disabled = !cb.checked;
      this._savePacing();
    });
    document.getElementById('coder-pacing-seconds')?.addEventListener('change', () => {
      this._savePacing();
    });
  }

  async _savePacing() {
    const cb = document.getElementById('coder-pacing-enabled');
    const secsEl = document.getElementById('coder-pacing-seconds');
    const enabled = cb ? !!cb.checked : false;
    // Clamp to the registered bounds (0–120). NaN/blank → keep pacing safe
    // by falling back to the 5s default rather than 0 when enabled.
    let seconds = Number(secsEl ? secsEl.value : 0);
    if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
    seconds = Math.min(seconds, 120);
    if (enabled && seconds <= 0) { seconds = 5; if (secsEl) secsEl.value = '5'; }
    try {
      const r = await fetch('/api/config/tools', {
        method: 'PUT',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          coder_request_delay_enabled: enabled ? 1 : 0,
          coder_request_delay_seconds: seconds,
        }),
      });
      if (r.ok) {
        showToast(enabled ? `Request pacing → ${seconds}s` : 'Request pacing off', 'info');
      } else if (r.status === 401 || r.status === 403) {
        showToast('Admin access required to change pacing', 'error');
      } else {
        showToast('Failed to save pacing', 'error');
      }
    } catch (err) {
      console.debug('pacing save failed', err);
      showToast('Failed to save pacing', 'error');
    }
  }

  // ── Strategy picker popover ─────────────────────────────────────

  // ── Loop tuning ─────────────────────────────────────────────────

  async _loadTuning() {
    try {
      const r = await fetch('/api/coder/tuning', { credentials: 'include' });
      if (!r.ok) {
        // Non-admin reads land here; the panel stays empty + the
        // section header still shows so the user knows it exists.
        this._tuningManifest = null;
        this._renderTuningError(r.status);
        return;
      }
      this._tuningManifest = await r.json();
      this._renderTuning();
    } catch (err) {
      console.debug('coder-inspector tuning load failed', err);
      this._renderTuningError(0);
    }
  }

  _renderPacing() {
    // Request-pacing control (bool toggle + float seconds). Separate from the
    // int breaker/cap rows — populated straight from the manifest's ``pacing``
    // block, saved via ``_savePacing``.
    const pacing = (this._tuningManifest && this._tuningManifest.pacing) || {};
    const cb = document.getElementById('coder-pacing-enabled');
    const secs = document.getElementById('coder-pacing-seconds');
    if (cb) cb.checked = !!pacing.enabled;
    if (secs) {
      // Keep the field usable even when disabled (0) — show the last set value
      // or the 5s default rather than blanking to 0.
      const v = Number(pacing.seconds);
      if (Number.isFinite(v) && v > 0) secs.value = String(v);
    }
    if (secs) secs.disabled = cb ? !cb.checked : false;
  }

  _renderTuning() {
    this._renderPacing();
    const container = document.getElementById('coder-tuning-rows');
    if (!container || !this._tuningManifest) return;
    const data = this._tuningManifest;
    // Iter caps first (highest-level knobs), then breakers grouped by
    // kind so "hard breaks" are visually distinct from "nudges".
    const caps = data.iter_caps || [];
    const breakers = data.breakers || [];
    const breaks = breakers.filter((b) => b.kind === 'break');
    const nudges = breakers.filter((b) => b.kind === 'nudge');
    const rows = [...caps, ...breaks, ...nudges];
    if (!rows.length) {
      container.innerHTML = '<div class="coder-inspector-muted">No tunables registered.</div>';
      this._renderTuningSummary(0, 0);
      return;
    }
    container.innerHTML = rows.map((row) => this._renderTuningRow(row)).join('');
    const overrides = rows.filter((r) => Number(r.override) > 0).length;
    this._renderTuningSummary(overrides, rows.length);
  }

  _renderTuningSummary(overrides, total) {
    // Surface the override count in the section header so the user
    // can see at a glance whether anything is non-default without
    // expanding the card. "defaults" when nothing's overridden.
    const summary = document.getElementById('coder-tuning-summary');
    if (!summary) return;
    if (!total) {
      summary.textContent = '';
      return;
    }
    summary.textContent = overrides > 0
      ? `${overrides} override${overrides === 1 ? '' : 's'}`
      : 'defaults';
  }

  _renderTuningRow(row) {
    const kindBadge = row.kind === 'break' ? 'break'
      : row.kind === 'nudge' ? 'nudge'
      : 'cap';
    const overridden = row.override > 0;
    const desc = row.description || '';
    const env = row.env_var ? ` · env: ${row.env_var}` : '';
    return `
      <div class="coder-tuning-row ${overridden ? 'overridden' : ''}" data-key="${escapeHtml(row.settings_key)}">
        <div class="coder-tuning-info">
          <div class="coder-tuning-name-row">
            <span class="coder-tuning-badge coder-tuning-badge-${kindBadge}">${kindBadge}</span>
            <span class="coder-tuning-name-text">${escapeHtml(row.name)}</span>
            <span class="coder-tuning-default" title="Registered default">default ${row.registered_default}</span>
            ${overridden ? `<span class="coder-tuning-effective" title="Effective threshold this turn">→ ${row.effective}</span>` : ''}
          </div>
          <div class="coder-tuning-desc">${escapeHtml(desc)}${env}</div>
        </div>
        <div class="coder-tuning-controls">
          <input
            type="number"
            class="coder-tuning-input"
            data-key="${escapeHtml(row.settings_key)}"
            value="${row.override}"
            min="0"
            max="10000"
            title="0 = use default (${row.registered_default})"
          />
          <button
            type="button"
            class="coder-tuning-reset"
            data-key="${escapeHtml(row.settings_key)}"
            title="Reset to default"
          >↺</button>
        </div>
      </div>
    `;
  }

  _renderTuningError(status) {
    const container = document.getElementById('coder-tuning-rows');
    if (!container) return;
    const msg = status === 401 || status === 403
      ? 'Admin access required to view tuning.'
      : 'Tuning unavailable — backend not reachable.';
    container.innerHTML = '';
    const muted = document.createElement('div');
    muted.className = 'coder-inspector-muted';
    muted.textContent = msg;
    container.appendChild(muted);
    const summary = document.getElementById('coder-tuning-summary');
    if (summary) summary.textContent = '';
  }

  _scheduleTuningSave(key, value) {
    if (this._tuningSaveTimer) clearTimeout(this._tuningSaveTimer);
    this._tuningSaveTimer = setTimeout(() => {
      this._tuningSaveTimer = null;
      this._saveTuningValue(key, value);
    }, 500);
  }

  async _saveTuningValue(key, value) {
    const v = Math.max(0, Math.floor(Number(value) || 0));
    try {
      const r = await fetch('/api/config/tools', {
        method: 'PUT',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: v }),
      });
      if (r.ok) {
        const label = key.replace('coder_breaker_', '').replace('coder_hybrid_', '');
        showToast(`${label} → ${v || 'default'}`, 'info');
        await this._loadTuning();
      } else if (r.status === 401 || r.status === 403) {
        showToast('Admin access required to change tuning', 'error');
      } else {
        showToast('Failed to save tuning', 'error');
      }
    } catch (err) {
      console.debug('tuning save failed', err);
      showToast('Failed to save tuning', 'error');
    }
  }

  async _resetAllTuning() {
    if (!this._tuningManifest) return;
    const rows = [
      ...(this._tuningManifest.iter_caps || []),
      ...(this._tuningManifest.breakers || []),
    ];
    const hasOverrides = rows.some((r) => r.override > 0);
    if (!hasOverrides) {
      showToast('Already at defaults', 'info');
      return;
    }
    if (!confirm('Reset all loop tuning to registered defaults?')) return;
    const body = {};
    for (const r of rows) body[r.settings_key] = 0;
    try {
      const resp = await fetch('/api/config/tools', {
        method: 'PUT',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (resp.ok) {
        showToast('All loop tuning reset to defaults', 'info');
        await this._loadTuning();
      } else {
        showToast('Reset failed', 'error');
      }
    } catch (err) {
      console.debug('tuning reset failed', err);
    }
  }
}

// ── Helpers ────────────────────────────────────────────────────

function _shortModel(model) {
  if (!model) return '';
  // Strip provider prefix + keep last segment compact
  const noFabric = model.split('@')[0];
  const parts = noFabric.split('/');
  let name = parts[parts.length - 1];
  if (name.length > 32) name = name.slice(0, 30) + '…';
  return name;
}

function _formatElapsed(seconds) {
  if (!seconds && seconds !== 0) return '0s';
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function _formatAgo(epochSeconds) {
  if (!epochSeconds) return '';
  const ageSec = Math.max(0, Date.now() / 1000 - epochSeconds);
  return _formatElapsed(ageSec) + ' ago';
}

function _formatUsd(amount) {
  const n = Number(amount || 0);
  if (n === 0) return '$0.0000';
  if (n < 0.01) return `$${n.toFixed(4)}`;
  return `$${n.toFixed(2)}`;
}

function _titleCase(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function _stripObjectiveHeader(text) {
  // Mirrors WorkspaceKernel._strip_objective_header — drop the seeded
  // # Session Objective / <!-- ... --> scaffolding so the panel shows
  // just the user-meaningful body. Hand-written objectives pass
  // through unchanged.
  if (!text) return '';
  const lines = text.split('\n');
  const out = [];
  let started = false;
  for (const line of lines) {
    const s = line.trim();
    if (!started) {
      if (!s) continue;
      if (s.startsWith('#')) continue;
      if (s.startsWith('<!--')) continue;
      started = true;
    }
    out.push(line);
  }
  return out.join('\n').trim();
}
