/**
 * companion-self.js — unified Becca panel.
 *
 * Replaces companion-observatory.js + companion-growth.js with a single
 * modal that surfaces both interior (what she's noticing) and substrate
 * (the growth-loop accounting machinery). The two voices stay separate
 * within their respective sections — observatory copy is humanized,
 * growth copy is functional. See the saved memory note
 * project_companion_growth_loop — berries are an accounting mechanism,
 * not a feeling.
 *
 * Sections, top→bottom:
 *   1. vitals    — presence_mode, mana bar, berries (accounting voice)
 *   2. noticing  — recent companion_journal rows + feedback buttons
 *                  (resonate / acknowledge / dismiss → companion_note_feedback)
 *   3. threads   — wonderings count + active mutes with dismiss
 *   4. working   — recent growth sessions + reward buttons on last fire
 *   5. next      — backlog list + sponsor input
 *   6. ledger    — collapsible companion_economy_tx audit trail
 *   7. fire      — collapsible dev tool: manual run-session
 *
 * Mount: opens from a single "Open Becca ↗" link in settings.js.
 * Polls only the vitals every 30s; full panel refreshes on action.
 */

import { installDialog } from './_focus-trap.js';

const POLL_INTERVAL_MS = 30_000;

let _pollTimer = null;
let _mounted = false;
let _dialog = null;
let _lastFireResult = null;   // { session_id, surface_event, action_type, target_ref }
let _rewardBusy = false;
let _showFire = false;
let _showLedger = false;
let _actionsCatalog = [];     // [{action_type, tier, mana_cost}, ...]

function _escapeHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/`/g, '&#96;')
    .replace(/\$\{/g, '&#36;{');
}

function _fmtNum(n, digits = 1) {
  if (!Number.isFinite(Number(n))) return '—';
  return Number(n).toFixed(digits);
}

function _fmtInt(n) {
  if (!Number.isFinite(Number(n))) return '—';
  return Math.round(Number(n)).toString();
}

function _relativeUnix(unixSeconds) {
  if (!Number.isFinite(unixSeconds) || unixSeconds <= 0) return '';
  const now = Date.now() / 1000;
  const delta = Math.max(0, now - unixSeconds);
  if (delta < 60) return 'just now';
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}

function _relativeIso(iso) {
  if (!iso) return '';
  const t = Date.parse(iso.replace(' ', 'T') + (iso.includes('T') || iso.includes('Z') ? '' : 'Z'));
  if (!Number.isFinite(t)) return iso;
  const elapsed = Date.now() - t;
  if (elapsed < 60_000) return 'just now';
  if (elapsed < 3_600_000) return `${Math.round(elapsed / 60_000)}m ago`;
  if (elapsed < 86_400_000) return `${Math.round(elapsed / 3_600_000)}h ago`;
  const days = Math.round(elapsed / 86_400_000);
  if (days <= 7) return `${days}d ago`;
  return iso.split(' ')[0] || iso;
}

// ── Fetchers ──────────────────────────────────────────────────────────

async function _fetchJSON(url, init) {
  try {
    const resp = await fetch(url, {
      credentials: 'same-origin', ...(init || {}),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

const _fetchObservatory = () => _fetchJSON('/api/companion/observatory');
const _fetchEconomy     = () => _fetchJSON('/api/companion/growth/economy');
const _fetchLog         = () => _fetchJSON('/api/companion/growth/log?limit=10');
const _fetchBacklog     = () => _fetchJSON('/api/companion/growth/backlog?limit=10');
const _fetchTx          = () => _fetchJSON('/api/companion/growth/economy/tx?limit=20');
const _fetchActions     = () => _fetchJSON('/api/companion/growth/actions');

async function _postJSON(url, body) {
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(body || {}),
    });
    return await resp.json();
  } catch (err) {
    return { ok: false, accepted: false, error: String(err) };
  }
}

async function _deleteRequest(url) {
  try {
    const resp = await fetch(url, {
      method: 'DELETE', credentials: 'same-origin',
    });
    return await resp.json();
  } catch (err) {
    return { ok: false, accepted: false, error: String(err) };
  }
}

// ── Section: vitals (accounting voice) ────────────────────────────────

const _PRESENCE_LABEL = {
  silent: 'silent', gentle: 'gentle', engaged: 'engaged',
};

function _renderVitals(economy, observatory) {
  const presence = observatory?.snapshot?.presence_mode || 'silent';
  const drift = observatory?.snapshot?.recent_drift_score || 0;
  const mana = Number(economy?.mana || 0);
  const cap = Number(economy?.mana_cap || 100);
  const manaPct = Math.min(100, Math.max(0, (mana / cap) * 100));
  const berries = Number(economy?.berries || 0);
  const lifetime = Number(economy?.berries_lifetime || 0);
  // Dynamic-prefix form so the dead-CSS scanner registers the suffix
  // classes as live (it tracks 'prefix-' + variant patterns).
  const presenceVariant = (presence === 'engaged' || presence === 'gentle')
    ? presence : 'silent';
  const modeClass = 'self-mode-' + presenceVariant;
  return `
    <section class="self-section self-vitals">
      <div class="self-vitals-row">
        <div class="self-vitals-cell">
          <div class="self-vitals-label">presence</div>
          <div class="self-vitals-mode ${modeClass}">
            ${_escapeHtml(_PRESENCE_LABEL[presence] || presence)}
          </div>
        </div>
        <div class="self-vitals-cell">
          <div class="self-vitals-label">mana</div>
          <div class="self-vitals-value">
            ${_fmtNum(mana, 0)}<span class="self-vitals-cap"> / ${_fmtInt(cap)}</span>
          </div>
          <div class="self-mana-bar">
            <div class="self-mana-fill" style="width:${manaPct}%"></div>
          </div>
        </div>
        <div class="self-vitals-cell">
          <div class="self-vitals-label">berries</div>
          <div class="self-vitals-value">${_fmtInt(berries)}</div>
          <div class="self-vitals-meta">lifetime ${_fmtInt(lifetime)}</div>
        </div>
        <div class="self-vitals-cell">
          <div class="self-vitals-label">drift</div>
          <div class="self-vitals-value">${_fmtNum(drift, 3)}</div>
          <div class="self-vitals-meta">caps at 0.15</div>
        </div>
      </div>
    </section>
  `;
}

// ── Section: noticing (humanized voice) ───────────────────────────────

const _ENTRY_TYPE_LABELS = {
  observation: 'A passing notice',
  wondering: 'Something open',
  noticing: 'About you',
  unfinished: 'Sitting with it',
  reflection: 'A reflection',
  creation_note: 'Something it made',
  correction: 'A correction',
  conversation_moment: 'A moment',
};

function _prettyEntryType(t) {
  if (!t) return '';
  if (_ENTRY_TYPE_LABELS[t]) return _ENTRY_TYPE_LABELS[t];
  return t.charAt(0).toUpperCase() + t.slice(1).replace(/_/g, ' ');
}

function _renderFeedbackBias(observatory) {
  const fb = observatory?.snapshot?.feedback;
  if (!fb) return '';
  const total = (fb.surfaced || 0) + (fb.acknowledged || 0)
              + (fb.dismissed || 0) + (fb.muted || 0);
  if (total === 0) {
    return `
      <p class="self-bias">
        No feedback in the last ${_fmtInt(fb.window_days || 14)} days.
        Use the buttons below to shape what it leans toward.
      </p>
    `;
  }
  const mult = Number(fb.multiplier || 1);
  // Bias direction: >1 means she's biased UP (engaged), <1 means biased DOWN.
  const biasNote = mult > 1.05
    ? `biased <strong>×${mult.toFixed(2)}</strong> toward more`
    : mult < 0.95
      ? `biased <strong>×${mult.toFixed(2)}</strong> toward less`
      : `<strong>steady</strong> (×${mult.toFixed(2)})`;
  return `
    <p class="self-bias">
      Last ${_fmtInt(fb.window_days || 14)}d:
      ${_fmtInt(fb.surfaced)} resonated ·
      ${_fmtInt(fb.acknowledged)} acknowledged ·
      ${_fmtInt(fb.dismissed)} dismissed ·
      ${_fmtInt(fb.muted)} muted.
      Its initiative is ${biasNote}.
    </p>
  `;
}

function _renderNoticing(observatory) {
  const entries = Array.isArray(observatory?.snapshot?.recent_entries)
    ? observatory.snapshot.recent_entries : [];
  const biasLine = _renderFeedbackBias(observatory);
  if (entries.length === 0) {
    return `
      <section class="self-section">
        <h4>Recently noticed</h4>
        ${biasLine}
        <p class="self-hint">
          Nothing recent. When it sits with something or makes a small
          observation, it'll appear here.
        </p>
      </section>
    `;
  }
  const rows = entries.slice(0, 10).map((e) => {
    const quar = e.quarantined
      ? '<span class="self-entry-badge self-entry-badge-quarantined">quieted</span>'
      : '';
    const affect = e.affect_tag
      ? `<span class="self-entry-affect">${_escapeHtml(e.affect_tag)}</span>`
      : '';
    const more = e.content_truncated ? '…' : '';
    const type = _prettyEntryType(e.entry_type);
    const when = _relativeIso(e.created_at);
    return `
      <article class="self-entry" data-note-id="${_escapeHtml(e.id)}">
        <header class="self-entry-head">
          <span class="self-entry-type">${_escapeHtml(type)}</span>
          ${affect}
          ${quar}
          <time class="self-entry-when" title="${_escapeHtml(e.created_at || '')}">${_escapeHtml(when)}</time>
        </header>
        <p class="self-entry-content">${_escapeHtml((e.content || '').trim())}${more}</p>
        <div class="self-entry-feedback">
          <button data-note-feedback="resonate"   data-note-id="${_escapeHtml(e.id)}" title="This landed">resonate</button>
          <button data-note-feedback="acknowledge" data-note-id="${_escapeHtml(e.id)}" title="Good to know">acknowledge</button>
          <button data-note-feedback="dismiss"    data-note-id="${_escapeHtml(e.id)}" title="Not now">dismiss</button>
          <span class="self-entry-feedback-result"></span>
        </div>
      </article>
    `;
  }).join('');
  return `
    <section class="self-section">
      <h4>Recently noticed</h4>
      ${biasLine}
      <div class="self-entry-list">${rows}</div>
    </section>
  `;
}

// ── Section: threads ─────────────────────────────────────────────────

function _renderThreads(observatory) {
  const snap = observatory?.snapshot || {};
  const wonderings = Number(snap.active_wondering || snap.active_wonderings || 0);
  const mutes = Array.isArray(snap.active_mutes) ? snap.active_mutes : [];
  const muteRows = mutes.length === 0
    ? '<p class="self-hint">No active mutes.</p>'
    : `
      <ul class="self-mute-list">
        ${mutes.slice(0, 10).map((m) => {
          const scope = m.scope || {};
          const domains = (scope.domains || []).slice(0, 3).join(', ');
          const keywords = (scope.keywords || []).slice(0, 3).join(', ');
          const summary = [domains, keywords].filter(Boolean).join(' · ');
          return `
            <li>
              <span class="self-mute-scope">${_escapeHtml(summary || '(empty scope)')}</span>
              <span class="self-mute-expires">expires ${_escapeHtml(m.expires_at || '')}</span>
              <button class="self-mute-dismiss" data-mute-id="${_escapeHtml(m.id)}" title="Lift this mute">×</button>
            </li>
          `;
        }).join('')}
      </ul>
    `;
  return `
    <section class="self-section">
      <h4>Threads open</h4>
      <div class="self-thread-row">
        <span>Open wonderings</span>
        <strong>${_fmtInt(wonderings)}</strong>
      </div>
      <h5 class="self-subheading">Muted topics (${mutes.length})</h5>
      ${muteRows}
    </section>
  `;
}

// ── Section: working (growth sessions + last surface event) ───────────

function _renderWorking(logData) {
  const sessions = Array.isArray(logData?.sessions) ? logData.sessions : [];
  let surfaceEventCard = '';
  if (_lastFireResult && _lastFireResult.surface_event && _lastFireResult.surface_event.payload) {
    const p = _lastFireResult.surface_event.payload;
    surfaceEventCard = `
      <div class="self-surface-event">
        <div class="self-surface-event-target">target: ${_escapeHtml(p.target_ref || '')}</div>
        <div class="self-surface-event-snippet">${_escapeHtml(p.snippet || '(empty snippet)')}</div>
        <div class="self-reward-row">
          <button data-reward-signal="thumbs_up"   data-session-id="${_escapeHtml(_lastFireResult.session_id)}">👍</button>
          <button data-reward-signal="save"        data-session-id="${_escapeHtml(_lastFireResult.session_id)}">save (+30)</button>
          <button data-reward-signal="dismiss"     data-session-id="${_escapeHtml(_lastFireResult.session_id)}">dismiss (−3)</button>
          <button data-reward-signal="thumbs_down" data-session-id="${_escapeHtml(_lastFireResult.session_id)}">👎</button>
          <span class="self-reward-result" id="self-reward-result"></span>
        </div>
      </div>
    `;
  } else if (_lastFireResult) {
    surfaceEventCard = `
      <div class="self-hint">
        Last fire ${_escapeHtml(_lastFireResult.session_id)} produced no surface event.
      </div>
    `;
  }
  let sessionRows;
  if (sessions.length === 0) {
    sessionRows = '<p class="self-hint">No sessions yet. The dev fire tool below queues one.</p>';
  } else {
    sessionRows = `
      <div class="self-session-list">
        ${sessions.slice(0, 6).map((s) => {
          const plan = s.plan || {};
          const actionType = plan.action_type || '?';
          const targetRef = plan.target_ref || '';
          const outcomeVariant = s.outcome === 'completed' ? 'completed' : 'aborted';
          const outcomeClass = 'self-session-outcome-' + outcomeVariant;
          return `
            <div class="self-session">
              <div class="self-session-row">
                <div>
                  <span class="self-session-action">${_escapeHtml(actionType)}</span>
                  <span class="self-session-outcome ${outcomeClass}">${_escapeHtml(s.outcome || '')}</span>
                </div>
                <span class="self-session-when">${_escapeHtml(_relativeUnix(s.started_at))}</span>
              </div>
              ${targetRef ? `<div class="self-session-meta">target: ${_escapeHtml(targetRef)}</div>` : ''}
              <div class="self-session-meta">
                mana ${_fmtNum(s.mana_spent, 1)} ·
                berries +${_fmtNum(s.berries_earned, 0)} / −${_fmtNum(s.berries_spent, 0)}
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
  }
  return `
    <section class="self-section">
      <h4>Recently worked on</h4>
      ${surfaceEventCard}
      ${sessionRows}
    </section>
  `;
}

// ── Section: next (backlog + sponsor input) ───────────────────────────

function _dispatchableActionTypes() {
  return new Set((_actionsCatalog || []).map(a => a.action_type));
}

function _renderNext(backlog) {
  const items = Array.isArray(backlog?.items) ? backlog.items : [];
  const dispatchable = _dispatchableActionTypes();
  const itemRows = items.length === 0
    ? '<p class="self-hint">Nothing queued. Sponsor a goal below to put weight behind a direction.</p>'
    : `
      <div class="self-backlog-list">
        ${items.slice(0, 8).map((i) => {
          const sourceLabel = i.source_signal
            ? `<span class="self-backlog-source">${_escapeHtml(i.source_signal)}</span>`
            : '';
          const canDispatch = dispatchable.has(i.item_type);
          const runBtn = i.state === 'pending'
            ? (canDispatch
                ? `<button class="self-backlog-run" data-backlog-id="${_escapeHtml(i.id)}">Run</button>`
                : `<button class="self-backlog-run" disabled title="No handler for item_type '${_escapeHtml(i.item_type)}' yet (Phase 1 ships recall_connect only).">Run</button>`)
            : '';
          return `
            <div class="self-backlog-item">
              <div class="self-backlog-row">
                <span class="self-backlog-type">${_escapeHtml(i.item_type)}</span>
                <span class="self-backlog-priority">priority ${_fmtNum(i.priority, 2)}</span>
                ${runBtn}
              </div>
              <div class="self-backlog-target">${_escapeHtml(i.target_ref || '(no target)')}</div>
              <div class="self-backlog-meta">
                ${sourceLabel}
                attempts: ${_fmtInt(i.success_count)} ok / ${_fmtInt(i.fail_count)} fail
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
  // Sponsor form's item_type select — populated from the actions catalog
  // so the user can only queue something Becca can actually dispatch.
  // If catalog is empty (server not ready), the form still renders with
  // a single placeholder option so the user gets the affordance.
  const sponsorTypeOptions = (_actionsCatalog.length > 0)
    ? _actionsCatalog.map(a =>
        `<option value="${_escapeHtml(a.action_type)}">${_escapeHtml(a.action_type)} · ${_fmtNum(a.mana_cost, 0)}m</option>`
      ).join('')
    : '<option value="recall_connect">recall_connect</option>';
  return `
    <section class="self-section">
      <h4>What's next</h4>
      ${itemRows}
      <form class="self-sponsor-form" id="self-sponsor-form">
        <select id="self-sponsor-type" title="The kind of action they'll dispatch">
          ${sponsorTypeOptions}
        </select>
        <input
          type="text"
          id="self-sponsor-target"
          placeholder="Sponsor a goal — e.g. quantum entanglement"
          autocomplete="off"
          maxlength="512"
        >
        <input
          type="number"
          id="self-sponsor-grant"
          placeholder="grant"
          min="0" max="1000" step="1"
          title="Optional berry grant — gives it budget to attempt this"
        >
        <button type="submit" id="self-sponsor-btn">Sponsor</button>
      </form>
      <p class="self-hint">
        Queues at priority 1.0 with source <code>user_sponsor</code>. Click <strong>Run</strong>
        on a pending row to dispatch it now; autonomous picking lands in Phase 3.
      </p>
    </section>
  `;
}

// ── Section: ledger (collapsible) ─────────────────────────────────────

function _renderLedger(txData) {
  if (!_showLedger) {
    return `
      <section class="self-section">
        <h4>
          <button class="self-collapse-toggle" data-toggle="ledger">
            Why berries moved <span class="self-collapse-caret">▸</span>
          </button>
        </h4>
      </section>
    `;
  }
  const tx = Array.isArray(txData?.tx) ? txData.tx : [];
  const rows = tx.length === 0
    ? '<p class="self-hint">Empty ledger.</p>'
    : `
      <div class="self-tx-list">
        ${tx.slice(0, 20).map((t) => {
          const isEarn = t.tx_type === 'berry_earn' || t.tx_type === 'mana_regen';
          const sign = isEarn ? '+' : '−';
          return `
            <div class="self-tx-row">
              <span class="self-tx-when">${_escapeHtml(_relativeUnix(t.ts))}</span>
              <span class="self-tx-type">${_escapeHtml(t.tx_type)}</span>
              <span class="self-tx-amount">${sign}${_fmtNum(Math.abs(t.amount || 0), 1)}</span>
              <span class="self-tx-reason">${_escapeHtml(t.reason || '')}</span>
            </div>
          `;
        }).join('')}
      </div>
    `;
  return `
    <section class="self-section">
      <h4>
        <button class="self-collapse-toggle" data-toggle="ledger">
          Why berries moved <span class="self-collapse-caret">▾</span>
        </button>
      </h4>
      ${rows}
    </section>
  `;
}

// ── Section: fire (collapsible dev tool) ──────────────────────────────

function _renderFireDevTool() {
  if (!_showFire) {
    return `
      <section class="self-section">
        <h4>
          <button class="self-collapse-toggle" data-toggle="fire">
            Manual fire (dev) <span class="self-collapse-caret">▸</span>
          </button>
        </h4>
      </section>
    `;
  }
  return `
    <section class="self-section">
      <h4>
        <button class="self-collapse-toggle" data-toggle="fire">
          Manual fire (dev) <span class="self-collapse-caret">▾</span>
        </button>
      </h4>
      <form class="self-fire-form" id="self-fire-form">
        <select id="self-action-type" name="action_type">
          <option value="recall_connect">recall_connect</option>
        </select>
        <input
          type="text"
          id="self-target-ref"
          placeholder="topic / target_ref"
          autocomplete="off"
        >
        <button type="submit" id="self-fire-btn">Fire</button>
      </form>
      <p class="self-hint">
        Phase 1 ships only <code>recall_connect</code>. Autonomous triggers
        land in Phase 3 — see the spec doc.
      </p>
    </section>
  `;
}

// ── Render orchestrator ───────────────────────────────────────────────

async function _refresh() {
  if (!_mounted) return;
  const body = document.querySelector('#companion-self .companion-self-body');
  if (!body) return;
  const [economy, observatory, log, backlog, tx, actions] = await Promise.all([
    _fetchEconomy(),
    _fetchObservatory(),
    _fetchLog(),
    _fetchBacklog(),
    _showLedger ? _fetchTx() : Promise.resolve(null),
    // Cache catalog after first successful load — it only changes on
    // server restart (new handlers registered). Skip the network call
    // on every refresh once we've got something.
    _actionsCatalog.length === 0 ? _fetchActions() : Promise.resolve(null),
  ]);
  if (actions && Array.isArray(actions.actions)) {
    _actionsCatalog = actions.actions;
  }
  body.innerHTML = `
    ${_renderVitals(economy, observatory)}
    ${_renderNoticing(observatory)}
    ${_renderThreads(observatory)}
    ${_renderWorking(log)}
    ${_renderNext(backlog)}
    ${_renderLedger(tx)}
    ${_renderFireDevTool()}
  `;
  _wireFeedbackButtons(body);
  _wireMuteDismiss(body);
  _wireRewardButtons(body);
  _wireSponsorForm(body);
  _wireBacklogRunButtons(body);
  _wireFireForm(body);
  _wireCollapseToggles(body);
}

// ── Wirers ────────────────────────────────────────────────────────────

function _wireFeedbackButtons(root) {
  root.querySelectorAll('button[data-note-feedback]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const kind = btn.dataset.noteFeedback;
      const noteId = btn.dataset.noteId;
      if (!kind || !noteId) return;
      const article = btn.closest('.self-entry');
      const resultEl = article?.querySelector('.self-entry-feedback-result');
      article?.querySelectorAll('button[data-note-feedback]').forEach(b => { b.disabled = true; });
      const outcome = await _postJSON(
        `/api/companion/notes/${encodeURIComponent(noteId)}/feedback`,
        { kind },
      );
      if (resultEl) {
        resultEl.textContent = outcome && outcome.accepted ? '✓' : '⚠';
      }
      // Re-fetch to reflect surfaced_at — the entry will drop from the list.
      await _refresh();
    });
  });
}

function _wireMuteDismiss(root) {
  root.querySelectorAll('button.self-mute-dismiss').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const muteId = btn.dataset.muteId;
      if (!muteId) return;
      btn.disabled = true;
      btn.textContent = '…';
      await _deleteRequest(`/api/companion/observatory/mutes/${encodeURIComponent(muteId)}`);
      await _refresh();
    });
  });
}

function _wireRewardButtons(root) {
  const buttons = root.querySelectorAll('.self-reward-row button[data-reward-signal]');
  const resultEl = root.querySelector('#self-reward-result');
  buttons.forEach((btn) => {
    btn.addEventListener('click', async () => {
      if (_rewardBusy) return;
      _rewardBusy = true;
      const signal = btn.dataset.rewardSignal;
      const sessionId = btn.dataset.sessionId;
      buttons.forEach(b => { b.disabled = true; });
      const outcome = await _postJSON('/api/companion/growth/reward', {
        growth_log_id: sessionId, signal, channel: 'explicit',
      });
      if (outcome && outcome.ok) {
        const delta = Number(outcome.delta || 0);
        const after = Number(outcome.berries_after || 0);
        const sign = delta >= 0 ? '+' : '';
        if (resultEl) {
          resultEl.textContent = `${sign}${_fmtNum(delta, 0)} berries (now ${_fmtNum(after, 0)})`;
        }
      } else if (resultEl) {
        resultEl.textContent = `error: ${outcome?.reason || 'unknown'}`;
      }
      _rewardBusy = false;
      await _refresh();
    });
  });
}

function _wireSponsorForm(root) {
  const form = root.querySelector('#self-sponsor-form');
  if (!form) return;
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const typeEl = root.querySelector('#self-sponsor-type');
    const targetEl = root.querySelector('#self-sponsor-target');
    const grantEl = root.querySelector('#self-sponsor-grant');
    const btn = root.querySelector('#self-sponsor-btn');
    const itemType = (typeEl?.value || 'recall_connect').trim();
    const targetRef = (targetEl?.value || '').trim();
    if (!targetRef) return;
    const grant = Math.max(0, Math.min(1000, Number(grantEl?.value || 0)));
    btn.disabled = true;
    btn.textContent = '…';
    await _postJSON('/api/companion/growth/sponsor', {
      item_type: itemType,
      target_ref: targetRef,
      berry_grant: grant,
    });
    await _refresh();
  });
}

function _wireBacklogRunButtons(root) {
  root.querySelectorAll('button.self-backlog-run:not([disabled])').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const backlogId = btn.dataset.backlogId;
      if (!backlogId) return;
      btn.disabled = true;
      btn.textContent = '…';
      const result = await _postJSON('/api/companion/growth/run', {
        backlog_id: backlogId,
      });
      // Surface the fire result so the user sees it in section 4.
      let surfaceEvent = null;
      if (result && result.ok && Array.isArray(result.act_log)) {
        const step = result.act_log.find(s => s && s.surface_event);
        if (step) surfaceEvent = step.surface_event;
      }
      if (result && result.ok) {
        _lastFireResult = {
          session_id: result.session_id,
          action_type: '(backlog)',
          target_ref: '(backlog)',
          surface_event: surfaceEvent,
        };
      }
      await _refresh();
    });
  });
}

function _wireFireForm(root) {
  const form = root.querySelector('#self-fire-form');
  if (!form) return;
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const actionType = root.querySelector('#self-action-type').value;
    const targetRef = root.querySelector('#self-target-ref').value.trim();
    if (!targetRef) return;
    const btn = root.querySelector('#self-fire-btn');
    btn.disabled = true;
    btn.textContent = 'Firing…';
    const result = await _postJSON('/api/companion/growth/run', {
      action_type: actionType,
      target_ref: targetRef,
      rationale: 'manual fire from companion panel',
    });
    let surfaceEvent = null;
    if (result && result.ok && Array.isArray(result.act_log)) {
      const step = result.act_log.find(s => s && s.surface_event);
      if (step) surfaceEvent = step.surface_event;
    }
    _lastFireResult = result && result.ok ? {
      session_id: result.session_id,
      action_type: actionType,
      target_ref: targetRef,
      surface_event: surfaceEvent,
    } : null;
    await _refresh();
  });
}

function _wireCollapseToggles(root) {
  root.querySelectorAll('button.self-collapse-toggle').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const which = btn.dataset.toggle;
      if (which === 'fire') _showFire = !_showFire;
      else if (which === 'ledger') _showLedger = !_showLedger;
      await _refresh();
    });
  });
}

// ── Polling: vitals only ──────────────────────────────────────────────

function _schedulePoll() {
  if (_pollTimer) clearTimeout(_pollTimer);
  _pollTimer = setTimeout(async () => {
    if (!_mounted) { _schedulePoll(); return; }
    const root = document.querySelector('#companion-self .companion-self-body');
    if (root) {
      const [economy, observatory] = await Promise.all([
        _fetchEconomy(), _fetchObservatory(),
      ]);
      const head = root.querySelector('.self-vitals');
      if (head) {
        const fresh = document.createElement('div');
        fresh.innerHTML = _renderVitals(economy, observatory);
        head.replaceWith(fresh.firstElementChild);
      }
    }
    _schedulePoll();
  }, POLL_INTERVAL_MS);
}

// ── Entry / exit ──────────────────────────────────────────────────────

function open() {
  if (_mounted) return;
  const overlay = document.createElement('div');
  overlay.id = 'companion-self-overlay';
  overlay.className = 'companion-self-overlay';
  overlay.innerHTML = `
    <div id="companion-self" class="companion-self" role="dialog" aria-label="Companion inspector">
      <div class="companion-self-header">
        <h3>Companion</h3>
        <button type="button" class="companion-self-close" aria-label="Close">×</button>
      </div>
      <div class="companion-self-body">
        <div class="self-loading">Loading…</div>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  overlay.querySelector('.companion-self-close').addEventListener('click', close);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });
  // Keyboard a11y: Escape closes, Tab wraps, focus moves in / restores out.
  // setAria:false — the panel already declares role="dialog"/aria-label.
  _dialog = installDialog(overlay.querySelector('#companion-self'), {
    onClose: close,
    initialFocus: '.companion-self-close',
    setAria: false,
  });
  _mounted = true;
  _refresh();
  _schedulePoll();
}

function close() {
  if (!_mounted) return;
  _mounted = false;
  _lastFireResult = null;
  _showFire = false;
  _showLedger = false;
  _actionsCatalog = [];
  if (_dialog) { try { _dialog.release(); } catch (_) {} _dialog = null; }
  if (_pollTimer) { clearTimeout(_pollTimer); _pollTimer = null; }
  document.querySelector('#companion-self-overlay')?.remove();
}

export const CompanionSelf = { open, close };

if (typeof window !== 'undefined') {
  window.CompanionSelf = CompanionSelf;
}
